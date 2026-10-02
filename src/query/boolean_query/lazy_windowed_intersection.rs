use common::TinySet;

use crate::docset::{DocSet, BLOCK_NUM_TINYBITSETS, BLOCK_WINDOW, TERMINATED};
use crate::postings::SegmentPostings;
use crate::query::scorer::PruningScorer;
use crate::query::term_query::TermScorer;
use crate::query::Scorer;
use crate::{DocId, Score};

/// ANDs `other` into `mask` in-place. Returns `true` if the result is all zeros.
#[inline]
fn and_blocks_and_return_is_empty(
    mask: &mut [TinySet; BLOCK_NUM_TINYBITSETS],
    update: &[TinySet; BLOCK_NUM_TINYBITSETS],
) -> bool {
    let mut all_empty = true;
    for (mask_tinyset, update_tinyset) in mask.iter_mut().zip(update.iter()) {
        *mask_tinyset = mask_tinyset.intersect(*update_tinyset);
        all_empty &= mask_tinyset.is_empty();
    }
    all_empty
}

/// A lazy, windowed bitset intersection scorer for dense conjunctions.
///
/// Operates on sliding windows of 1,024 doc IDs (`BLOCK_WINDOW`).
/// For each term, maintains two separate cursors through the skipdata:
/// 1. `matchers`: doc-only postings cursors with frequency reading disabled. These decode doc IDs
///    into fixed-size 128-byte stack bitmasks.
/// 2. `scorers`: full `TermScorer`s that lag behind and only seek forward monotonically to evaluate
///    BM25 scores on confirmed surviving matches.
pub struct LazyWindowedIntersectionScorer {
    matchers: Vec<SegmentPostings>,
    scorers: Vec<TermScorer>,
    threshold: Score,
    scoring_enabled: bool,
    window_base: DocId,
    window_mask: [TinySet; BLOCK_NUM_TINYBITSETS],
    window_cursor: usize,
    current: (DocId, Score),
    maximum_possible_score: Score,
    suffix_max_scores: Vec<Score>,
}

impl LazyWindowedIntersectionScorer {
    /// Creates a new `LazyWindowedIntersectionScorer`.
    pub fn new(scorers: Vec<TermScorer>, threshold: Score) -> Self {
        Self::build(scorers, threshold, true)
    }

    pub(crate) fn new_without_scoring(scorers: Vec<TermScorer>) -> Self {
        Self::build(scorers, Score::MIN, false)
    }

    fn build(mut scorers: Vec<TermScorer>, threshold: Score, scoring_enabled: bool) -> Self {
        assert!(scorers.len() >= 2);
        // Sort scorers by cost ascending (lowest doc freq = leader)
        scorers.sort_by_key(|s| s.cost());
        let maximum_possible_score = scorers.iter().map(|s| s.max_score()).sum();

        let mut suffix_max_scores = vec![0.0; scorers.len()];
        let mut sum = 0.0;
        for i in (0..scorers.len()).rev() {
            suffix_max_scores[i] = sum;
            sum += scorers[i].max_score();
        }

        // Create doc-only matching cursors by cloning postings and disabling freqs
        let matchers: Vec<SegmentPostings> = scorers
            .iter()
            .map(|s| {
                let mut postings = s.postings().clone();
                postings.disable_freq_reading();
                postings
            })
            .collect();

        let initial_leader_doc = matchers[0].doc();
        let initial_base = if initial_leader_doc < TERMINATED {
            (initial_leader_doc / BLOCK_WINDOW) * BLOCK_WINDOW
        } else {
            TERMINATED
        };

        let mut scorer = Self {
            matchers,
            scorers,
            threshold,
            scoring_enabled,
            window_base: initial_base,
            window_mask: [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS],
            window_cursor: BLOCK_NUM_TINYBITSETS,
            current: (0, Score::MIN),
            maximum_possible_score,
            suffix_max_scores,
        };

        if initial_base < TERMINATED {
            scorer.advance_to_next_window(initial_base);
        }
        scorer.advance();
        scorer
    }

    #[inline]
    fn pop_next_candidate_in_window(&mut self) -> Option<DocId> {
        while self.window_cursor < BLOCK_NUM_TINYBITSETS {
            if let Some(bit) = self.window_mask[self.window_cursor].pop_lowest() {
                let doc = self.window_base + (self.window_cursor as u32 * 64) + bit;
                return Some(doc);
            }
            self.window_cursor += 1;
        }
        None
    }

    fn advance_to_next_window(&mut self, mut next_base: DocId) -> bool {
        let leader_doc = self.matchers[0].doc();
        if leader_doc == TERMINATED {
            return false;
        }
        if leader_doc >= next_base + BLOCK_WINDOW {
            next_base = (leader_doc / BLOCK_WINDOW) * BLOCK_WINDOW;
        }
        self.window_base = next_base;

        while self.window_base < TERMINATED {
            let window_start = self.window_base;
            let window_end = window_start + BLOCK_WINDOW;

            // 1. Leader fills window_mask.
            self.window_mask = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
            self.matchers[0].fill_bitset_window(window_start, window_end, &mut self.window_mask);

            let mut is_empty = self.window_mask.iter().all(|t| t.is_empty());
            if is_empty {
                let next_leader = self.matchers[0].doc();
                if next_leader == TERMINATED {
                    return false;
                }
                self.window_base = (next_leader / BLOCK_WINDOW)
                    .max(self.window_base / BLOCK_WINDOW + 1)
                    * BLOCK_WINDOW;
                continue;
            }

            // 2. Cascading secondary intersection.
            for secondary in &mut self.matchers[1..] {
                let mut tail_mask = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
                secondary.fill_bitset_window(window_start, window_end, &mut tail_mask);

                is_empty = and_blocks_and_return_is_empty(&mut self.window_mask, &tail_mask);
                if is_empty {
                    // Secondary eliminated all candidates in this window.
                    // Skip remaining secondaries for this window.
                    break;
                }
            }

            if is_empty {
                self.window_base += BLOCK_WINDOW;
                continue;
            }

            // Confirmed matches found in window!
            self.window_cursor = 0;
            return true;
        }

        false
    }
}

impl DocSet for LazyWindowedIntersectionScorer {
    fn advance(&mut self) -> DocId {
        if self.scoring_enabled && self.maximum_possible_score <= self.threshold {
            self.current = (TERMINATED, Score::MIN);
            return TERMINATED;
        }

        loop {
            // Drain remaining matches in the current window.
            'candidate: while let Some(candidate_doc) = self.pop_next_candidate_in_window() {
                if !self.scoring_enabled {
                    self.current = (candidate_doc, 1.0);
                    return candidate_doc;
                }
                // candidate_doc is guaranteed to be in every term's postings!
                // Compute BM25 score across terms with early threshold pruning.
                let mut total_score: Score = 0.0;
                for (idx, scorer) in self.scorers.iter_mut().enumerate() {
                    let d = scorer.seek(candidate_doc);
                    debug_assert_eq!(d, candidate_doc);
                    total_score += scorer.score();

                    if total_score + self.suffix_max_scores[idx] <= self.threshold {
                        continue 'candidate;
                    }
                }

                if total_score > self.threshold {
                    self.current = (candidate_doc, total_score);
                    return candidate_doc;
                }
            }

            // Window is exhausted. Advance to next non-empty window.
            if !self.advance_to_next_window(self.window_base + BLOCK_WINDOW) {
                self.current = (TERMINATED, Score::MIN);
                return TERMINATED;
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if target <= self.doc() {
            return self.doc();
        }
        if target >= TERMINATED || self.doc() == TERMINATED {
            self.current = (TERMINATED, Score::MIN);
            return TERMINATED;
        }

        let window_end = self.window_base + BLOCK_WINDOW;
        if target < window_end {
            // Target is in the current window.
            let delta = target - self.window_base;
            let target_bucket = (delta / 64) as usize;
            let target_bit = delta % 64;

            for bucket in self.window_cursor..target_bucket.min(BLOCK_NUM_TINYBITSETS) {
                self.window_mask[bucket] = TinySet::EMPTY;
            }
            if target_bucket < BLOCK_NUM_TINYBITSETS {
                self.window_mask[target_bucket] = self.window_mask[target_bucket]
                    .intersect(TinySet::range_greater_or_equal(target_bit));
            }
            self.window_cursor = target_bucket;
            return self.advance();
        }

        // Target is past current window.
        let next_base = (target / BLOCK_WINDOW) * BLOCK_WINDOW;
        if self.matchers[0].doc() < target {
            self.matchers[0].seek(target);
        }
        if !self.advance_to_next_window(next_base) {
            self.current = (TERMINATED, Score::MIN);
            return TERMINATED;
        }

        // Mask out any candidates in the newly loaded window that are < target.
        if self.window_base <= target {
            let delta = target - self.window_base;
            let target_bucket = (delta / 64) as usize;
            let target_bit = delta % 64;
            for bucket in 0..target_bucket.min(BLOCK_NUM_TINYBITSETS) {
                self.window_mask[bucket] = TinySet::EMPTY;
            }
            if target_bucket < BLOCK_NUM_TINYBITSETS {
                self.window_mask[target_bucket] = self.window_mask[target_bucket]
                    .intersect(TinySet::range_greater_or_equal(target_bit));
            }
            self.window_cursor = target_bucket;
        }
        self.advance()
    }

    #[inline]
    fn doc(&self) -> DocId {
        self.current.0
    }

    fn size_hint(&self) -> u32 {
        self.matchers[0].size_hint()
    }
}

impl Scorer for LazyWindowedIntersectionScorer {
    #[inline]
    fn score(&mut self) -> Score {
        self.current.1
    }
}

impl PruningScorer for LazyWindowedIntersectionScorer {
    #[inline]
    fn set_threshold(&mut self, score: Score) {
        self.threshold = score;
    }
}

#[cfg(test)]
mod tests {
    use common::HasLen;

    use super::*;
    use crate::docset::DocSet;
    use crate::query::term_query::TermScorer;
    use crate::query::Bm25Weight;
    use crate::Score;

    fn make_test_scorers(doc_freqs: &[Vec<u32>]) -> Vec<TermScorer> {
        let fieldnorm_reader = crate::fieldnorm::FieldNormReader::constant(10, 10);
        doc_freqs
            .iter()
            .map(|docs| {
                let segment_postings = SegmentPostings::create_from_docs(docs);
                let bm25_weight = Bm25Weight::for_one_term(
                    1,
                    segment_postings.len() as u64,
                    10.0,
                    crate::Bm25Params::default(),
                );
                TermScorer::new(segment_postings, fieldnorm_reader.clone(), bm25_weight)
            })
            .collect()
    }

    #[test]
    fn test_lazy_windowed_basic() {
        let scorers = make_test_scorers(&[
            vec![1, 5, 10, 100, 500, 1024, 1025, 2000],
            vec![5, 10, 50, 100, 500, 1025, 2000],
            vec![2, 5, 10, 100, 500, 1025, 2000, 3000],
        ]);
        let mut scorer = LazyWindowedIntersectionScorer::new(scorers, Score::MIN);
        let mut matches = Vec::new();
        while scorer.doc() != TERMINATED {
            matches.push(scorer.doc());
            scorer.advance();
        }
        assert_eq!(matches, vec![5, 10, 100, 500, 1025, 2000]);
    }

    #[test]
    fn test_lazy_windowed_seek() {
        let scorers = make_test_scorers(&[
            vec![5, 10, 100, 500, 1025, 2000],
            vec![5, 10, 100, 500, 1025, 2000],
        ]);
        let mut scorer = LazyWindowedIntersectionScorer::new(scorers, Score::MIN);
        assert_eq!(scorer.seek(10), 10);
        assert_eq!(scorer.seek(500), 500);
        assert_eq!(scorer.seek(1024), 1025);
        assert_eq!(scorer.seek(2001), TERMINATED);
    }

    #[test]
    fn test_lazy_windowed_no_matches() {
        let scorers = make_test_scorers(&[vec![1, 2, 3], vec![4, 5, 6]]);
        let scorer = LazyWindowedIntersectionScorer::new(scorers, Score::MIN);
        assert_eq!(scorer.doc(), TERMINATED);
    }
    #[test]
    fn test_unscored_windowed_matches_oracle() {
        for seed in 0u32..24 {
            let lists: Vec<Vec<u32>> = (0..3)
                .map(|term| {
                    (0u32..10000)
                        .filter(|doc| {
                            doc.wrapping_mul(2654435761)
                                .wrapping_add(seed * 97 + term * 113)
                                .rotate_left(term * 7)
                                % 10
                                < 7
                        })
                        .collect()
                })
                .collect();
            let expected: Vec<u32> = lists[0]
                .iter()
                .copied()
                .filter(|doc| {
                    lists[1].binary_search(doc).is_ok() && lists[2].binary_search(doc).is_ok()
                })
                .collect();
            let mut scorer =
                LazyWindowedIntersectionScorer::new_without_scoring(make_test_scorers(&lists));
            let mut actual = Vec::new();
            while scorer.doc() != TERMINATED {
                actual.push(scorer.doc());
                scorer.advance();
            }
            assert_eq!(actual, expected);
            let mut scorer =
                LazyWindowedIntersectionScorer::new_without_scoring(make_test_scorers(&lists));
            for target in (0..11000).step_by(137) {
                assert_eq!(
                    scorer.seek(target),
                    expected
                        .iter()
                        .copied()
                        .find(|doc| *doc >= target)
                        .unwrap_or(TERMINATED)
                );
            }
        }
    }
}
