use std::ops::{Deref, DerefMut};

use common::TinySet;

use crate::docset::{fill_all_candidates_mask, ProbeCost, SeekDangerResult};
use crate::query::exclude::BatchSizeRamp;
use crate::query::scorer::{BlockMaxScorer, PruningScorer};
use crate::query::term_query::TermScorer;
#[cfg(test)]
use crate::query::weight::for_each_pruning_scorer;
use crate::query::Scorer;
use crate::{DocId, DocSet, Score, TERMINATED};

/// How an embedded WAND filter is evaluated relative to alignment and scoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WandFilterMode {
    /// No filter is attached.
    None,
    /// Probe the filter before aligning secondary scorers or computing BM25 scores
    /// (beneficial for low-probe-cost filters such as bitsets).
    Early,
    /// Align/score terms and confirm the score beats the current threshold before probing
    /// the filter one doc at a time.
    AfterScoreScalar,
    /// Align/score terms, collect surviving candidates into a batch, and probe the filter via
    /// `seek_danger_batch` when `ProbeCost::prefers_batching()` is true.
    AfterScoreBatch,
}

impl WandFilterMode {
    pub(super) fn select(filter: Option<&dyn DocSet>, early_probe_budget: u64) -> Self {
        match filter {
            None => Self::None,
            Some(f) => {
                let probe = f.cost().probe;
                if probe.prefers_batching() {
                    Self::AfterScoreBatch
                } else if probe.scalar() > early_probe_budget {
                    Self::AfterScoreScalar
                } else {
                    Self::Early
                }
            }
        }
    }
}

/// Seeks `scorer` to `filter.doc()` if it is behind, so the filter is never
/// asked about a doc below its current one: `PhraseScorer::seek_danger` asserts
/// against that, and an `Intersection` of `RangeDocSet`s overflows on it. A new
/// filter is on its first match, so the skipped docs cannot match.
pub(super) fn seek_to_filter_start<T: DocSet + ?Sized>(scorer: &mut T, filter: &dyn DocSet) {
    let filter_doc = filter.doc();
    if scorer.doc() < filter_doc {
        scorer.seek(filter_doc);
    }
}

/// Takes a term_scorers sorted by their current doc() and a threshold and returns
/// Returns (pivot_len, pivot_ord) defined as follows:
/// - `pivot_doc` lowest document that has a chance of exceeding (>) the threshold score.
/// - `before_pivot_len` number of term_scorers such that term_scorer.doc() < pivot.
/// - `pivot_len` number of term_scorers such that term_scorer.doc() <= pivot.
///
/// We always have `before_pivot_len` < `pivot_len`.
///
/// `None` is returned if we establish that no document can exceed the threshold.
fn find_pivot_doc<T: BlockMaxScorer>(
    term_scorers: &[TermScorerWithMaxScore<T>],
    threshold: Score,
) -> Option<(usize, usize, DocId)> {
    let mut max_score = 0.0;
    let mut before_pivot_len = 0;
    let mut pivot_doc = TERMINATED;
    while before_pivot_len < term_scorers.len() {
        let term_scorer = &term_scorers[before_pivot_len];
        max_score += term_scorer.max_score;
        if max_score > threshold {
            pivot_doc = term_scorer.doc();
            break;
        }
        before_pivot_len += 1;
    }
    if pivot_doc == TERMINATED {
        return None;
    }
    // Right now i is an ordinal, we want a len.
    let mut pivot_len = before_pivot_len + 1;
    // Some other term_scorer may be positioned on the same document.
    pivot_len += term_scorers[pivot_len..]
        .iter()
        .take_while(|term_scorer| term_scorer.doc() == pivot_doc)
        .count();
    Some((before_pivot_len, pivot_len, pivot_doc))
}

/// Advance the scorer with best score among the scorers[..pivot_len] to
/// the next doc candidate defined by the min of `last_doc_in_block + 1` for
/// scorer in scorers[..pivot_len] and `scorer.doc()` for scorer in scorers[pivot_len..].
/// Note: before and after calling this method, scorers need to be sorted by their `.doc()`.
fn block_max_was_too_low_advance_one_scorer<T: BlockMaxScorer>(
    scorers: &mut [TermScorerWithMaxScore<T>],
    pivot_len: usize,
) {
    debug_assert!(scorers.iter().map(|scorer| scorer.doc()).is_sorted());
    let mut scorer_to_seek = pivot_len - 1;
    let mut global_max_score = scorers[scorer_to_seek].max_score;
    let mut doc_to_seek_after = scorers[scorer_to_seek].last_doc_in_block();
    for scorer_ord in (0..pivot_len - 1).rev() {
        let scorer = &scorers[scorer_ord];
        if scorer.last_doc_in_block() <= doc_to_seek_after {
            doc_to_seek_after = scorer.last_doc_in_block();
        }
        if scorers[scorer_ord].max_score > global_max_score {
            global_max_score = scorers[scorer_ord].max_score;
            scorer_to_seek = scorer_ord;
        }
    }
    // Add +1 to go to the next block unless we are already at the end.
    if doc_to_seek_after != TERMINATED {
        doc_to_seek_after += 1;
    }
    for scorer in &scorers[pivot_len..] {
        if scorer.doc() <= doc_to_seek_after {
            doc_to_seek_after = scorer.doc();
        }
    }
    scorers[scorer_to_seek].seek(doc_to_seek_after);

    restore_ordering(scorers, scorer_to_seek);
    debug_assert!(scorers.iter().map(|scorer| scorer.doc()).is_sorted());
}

// Given a list of term_scorers and a `ord` and assuming that `term_scorers[ord]` is sorted
// except term_scorers[ord] that might be in advance compared to its ranks,
// bubble up term_scorers[ord] in order to restore the ordering.
fn restore_ordering<T: BlockMaxScorer>(term_scorers: &mut [TermScorerWithMaxScore<T>], ord: usize) {
    let doc = term_scorers[ord].doc();
    for i in ord + 1..term_scorers.len() {
        if term_scorers[i].doc() >= doc {
            break;
        }
        term_scorers.swap(i, i - 1);
    }
    debug_assert!(term_scorers.iter().map(|scorer| scorer.doc()).is_sorted());
}

// Attempts to advance all term_scorers between `&term_scorers[0..before_len]` to the pivot.
// If this works, return true.
// If this fails (ie: one of the term_scorer does not contain `pivot_doc` and seek goes past the
// pivot), reorder the term_scorers to ensure the list is still sorted and returns `false`.
// If a term_scorer reach TERMINATED in the process return false remove the term_scorer and return.
fn align_scorers<T: BlockMaxScorer>(
    term_scorers: &mut Vec<TermScorerWithMaxScore<T>>,
    pivot_doc: DocId,
    before_pivot_len: usize,
) -> bool {
    debug_assert_ne!(pivot_doc, TERMINATED);
    for i in (0..before_pivot_len).rev() {
        let new_doc = term_scorers[i].seek(pivot_doc);
        if new_doc != pivot_doc {
            if new_doc == TERMINATED {
                term_scorers.swap_remove(i);
            }
            // We went past the pivot.
            // We just go through the outer loop mechanic (Note that pivot is
            // still a possible candidate).
            //
            // Termination is still guaranteed since we can only consider the same
            // pivot at most term_scorers.len() - 1 times.
            restore_ordering(term_scorers, i);
            return false;
        }
    }
    true
}

// Assumes terms_scorers[..pivot_len] are positioned on the same doc (pivot_doc).
// Advance term_scorers[..pivot_len] and out of these removes the terminated scores.
// Restores the ordering of term_scorers.
fn advance_all_scorers_on_pivot<T: BlockMaxScorer>(
    term_scorers: &mut Vec<TermScorerWithMaxScore<T>>,
    pivot_len: usize,
) {
    if pivot_len == 1 {
        if term_scorers[0].advance() == TERMINATED {
            term_scorers.swap_remove(0);
        }
        if !term_scorers.is_empty() {
            restore_ordering(term_scorers, 0);
        }
        return;
    }
    for term_scorer in &mut term_scorers[..pivot_len] {
        term_scorer.advance();
    }
    // TODO use drain_filter when available.
    let mut i = 0;
    while i != term_scorers.len() {
        if term_scorers[i].doc() == TERMINATED {
            term_scorers.swap_remove(i);
        } else {
            i += 1;
        }
    }
    term_scorers.sort_by_key(|scorer| scorer.doc());
}

#[cfg(test)]
pub fn block_wand(
    mut scorers: Vec<TermScorer>,
    threshold: Score,
    callback: &mut dyn FnMut(u32, Score) -> Score,
) {
    if scorers.len() == 1 {
        let scorer = scorers.pop().unwrap();
        return block_wand_single_scorer(scorer, threshold, callback);
    }
    let mut scorer = BlockWandUnionScorer::new(scorers, threshold);
    for_each_pruning_scorer(&mut scorer, callback);
}

#[cfg(test)]
pub fn block_wand_single_scorer(
    scorer: TermScorer,
    threshold: Score,
    callback: &mut dyn FnMut(u32, Score) -> Score,
) {
    let mut scorer = BlockWandSingleScorer::new(scorer, threshold);
    for_each_pruning_scorer(&mut scorer, callback);
}

struct TermScorerWithMaxScore<T: BlockMaxScorer = TermScorer> {
    scorer: Box<T>,
    max_score: Score,
}

impl<T: BlockMaxScorer> From<T> for TermScorerWithMaxScore<T> {
    fn from(scorer: T) -> Self {
        let max_score = scorer.max_score();
        let scorer = Box::new(scorer);
        TermScorerWithMaxScore { scorer, max_score }
    }
}

impl<T: BlockMaxScorer> Deref for TermScorerWithMaxScore<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.scorer
    }
}

impl<T: BlockMaxScorer> DerefMut for TermScorerWithMaxScore<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.scorer
    }
}

const UNION_ACTIVE_THRESHOLD_BATCH: usize = 128;
const SURVIVOR_MAX_BATCH_WORDS: usize = BatchSizeRamp::Full.limit().div_ceil(64);

/// Parallel `(DocId, Score)` buffer shared by WAND scorers in `WandFilterMode::AfterScoreBatch`.
///
/// Collects candidates that pass score pruning, filters them in place via
/// [`DocSet::seek_danger_batch`], and drains surviving hits while honoring live threshold updates.
pub(super) struct ScoredSurvivorBuffer {
    docs: Vec<DocId>,
    scores: Vec<Score>,
    idx: usize,
}

impl ScoredSurvivorBuffer {
    pub(super) fn new() -> Self {
        Self {
            docs: Vec::new(),
            scores: Vec::new(),
            idx: 0,
        }
    }

    #[inline]
    pub(super) fn clear(&mut self) {
        self.docs.clear();
        self.scores.clear();
        self.idx = 0;
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.docs.len()
    }

    #[inline]
    pub(super) fn last_doc(&self) -> Option<DocId> {
        self.docs.last().copied()
    }

    #[inline]
    pub(super) fn push(&mut self, doc: DocId, score: Score) {
        self.docs.push(doc);
        self.scores.push(score);
    }

    /// Advances past buffered docs `< target`. Returns `true` if at least one buffered doc remains
    /// at or above `target`, or clears the buffer and returns `false` if exhausted.
    #[inline]
    pub(super) fn skip_below(&mut self, target: DocId) -> bool {
        while self.idx < self.docs.len() && self.docs[self.idx] < target {
            self.idx += 1;
        }
        if self.idx == self.docs.len() {
            self.clear();
            false
        } else {
            true
        }
    }

    #[inline]
    pub(super) fn next_above(&mut self, threshold: Score) -> Option<(DocId, Score)> {
        while self.idx < self.docs.len() {
            let i = self.idx;
            self.idx += 1;
            let score = self.scores[i];
            if score > threshold {
                return Some((self.docs[i], score));
            }
        }
        self.clear();
        None
    }

    /// Filters `self.docs` in place with `filter.seek_danger_batch`, returning the skip-horizon
    /// `DocId` from the filter.
    pub(super) fn filter_with(&mut self, filter: &mut dyn DocSet) -> DocId {
        let n = self.docs.len();
        debug_assert!(n > 0 && n <= BatchSizeRamp::Full.limit());
        let words = n.div_ceil(64);
        let mut mask = [TinySet::EMPTY; SURVIVOR_MAX_BATCH_WORDS];
        fill_all_candidates_mask(n, &mut mask[..words]);
        let next_bound = filter.seek_danger_batch(&self.docs, &mut mask[..words]);
        let mut write = 0usize;
        for (word_idx, &word) in mask[..words].iter().enumerate() {
            let base = word_idx * 64;
            for bit in word {
                let i = base + bit as usize;
                self.docs[write] = self.docs[i];
                self.scores[write] = self.scores[i];
                write += 1;
            }
        }
        self.docs.truncate(write);
        self.scores.truncate(write);
        self.idx = 0;
        next_bound
    }
}

/// Implements the WAND (Weak AND) algorithm for dynamic pruning
/// described in the paper "Faster Top-k Document Retrieval Using Block-Max Indexes".
/// Link: <http://engineering.nyu.edu/~suel/papers/bmw.pdf>
///
/// When an optional non-scoring filter docset is provided, pivot documents are checked
/// against the filter either early (before aligning secondary scorers, for low-probe-cost
/// filters), scalar after scoring, or in batches via `seek_danger_batch` when the filter's
/// `ProbeCost::prefers_batching()` is true.
///
/// # Preconditions
/// - `scorers` has at least 2 elements
/// - All scorers read frequencies (`FreqReadingOption::ReadFreq`)
pub struct BlockWandUnionScorer<T: BlockMaxScorer = TermScorer> {
    scorers: Vec<TermScorerWithMaxScore<T>>,
    filter: Option<Box<dyn DocSet>>,
    filter_mode: WandFilterMode,
    filter_boost: Score,
    threshold: Score,
    survivors: ScoredSurvivorBuffer,
    batch_ramp: BatchSizeRamp,
    current: (DocId, Score),
}
impl<T: BlockMaxScorer> BlockWandUnionScorer<T> {
    /// Construction positions `current` on the first match
    pub fn new(scorers: Vec<T>, threshold: Score) -> Self {
        Self::with_filter_and_boost(scorers, threshold, None, 0.0)
    }

    /// Creates a new `BlockWandUnionScorer` with an embedded non-scoring filter.
    ///
    /// Filter evaluation placement is chosen from `filter.cost().probe` (`Early`,
    /// `AfterScoreScalar`, or batched `AfterScoreBatch` via `seek_danger_batch`).
    /// `filter_boost` is added to matching document scores and accounted for in the
    /// internal pruning threshold.
    pub fn with_filter(
        scorers: Vec<T>,
        threshold: Score,
        filter: Box<dyn DocSet>,
        filter_boost: Score,
    ) -> Self {
        Self::with_filter_and_boost(scorers, threshold, Some(filter), filter_boost)
    }

    fn with_filter_and_boost(
        mut scorers: Vec<T>,
        threshold: Score,
        filter: Option<Box<dyn DocSet>>,
        filter_boost: Score,
    ) -> Self {
        debug_assert!(scorers.len() > 1);
        if let Some(filter) = &filter {
            for scorer in &mut scorers {
                seek_to_filter_start(scorer, filter.as_ref());
            }
        }
        scorers.retain(|scorer| scorer.doc() < TERMINATED);
        let mut scorers: Vec<TermScorerWithMaxScore<T>> = scorers
            .into_iter()
            .map(TermScorerWithMaxScore::from)
            .collect();
        // At this point we need to ensure that the scorers are sorted!
        scorers.sort_by_key(|scorer| scorer.doc());

        let align_and_score_cost = (scorers.len().saturating_sub(1) as u64)
            .saturating_mul(ProbeCost::TERM.scalar())
            .saturating_add(scorers.len() as u64);
        let filter_mode = WandFilterMode::select(filter.as_deref(), align_and_score_cost);

        let mut scorer = Self {
            scorers,
            filter,
            filter_mode,
            filter_boost,
            threshold,
            survivors: ScoredSurvivorBuffer::new(),
            batch_ramp: BatchSizeRamp::Small,
            current: (0, Score::MIN),
        };
        // advance to fill current with actual (doc id, score)
        scorer.advance();
        scorer
    }

    fn seek_scorers_to(&mut self, target: DocId) {
        if target >= TERMINATED {
            self.scorers.clear();
            return;
        }
        let mut any_moved = false;
        for scorer in &mut self.scorers {
            if scorer.doc() < target {
                scorer.seek(target);
                any_moved = true;
            }
        }
        if any_moved {
            self.scorers.retain(|scorer| scorer.doc() < TERMINATED);
            self.scorers.sort_by_key(|scorer| scorer.doc());
        }
    }

    fn advance_after_score_batch(&mut self, inner_threshold: Score) -> DocId {
        if let Some((doc, raw_score)) = self.survivors.next_above(inner_threshold) {
            self.current = (doc, raw_score + self.filter_boost);
            return doc;
        }

        loop {
            let batch_cap = if self.threshold == Score::MIN {
                let cap = self.batch_ramp.limit();
                self.batch_ramp.promote();
                cap
            } else {
                UNION_ACTIVE_THRESHOLD_BATCH
            };
            while self.survivors.len() < batch_cap {
                let Some((before_pivot_len, pivot_len, pivot_doc)) =
                    find_pivot_doc(&self.scorers[..], inner_threshold)
                else {
                    break;
                };
                debug_assert!(self.scorers.iter().map(|scorer| scorer.doc()).is_sorted());
                debug_assert_ne!(pivot_doc, TERMINATED);
                debug_assert!(before_pivot_len < pivot_len);

                let block_max_score_upperbound: Score = self.scorers[..pivot_len]
                    .iter_mut()
                    .map(|scorer| {
                        scorer.seek_block(pivot_doc);
                        scorer.block_max_score()
                    })
                    .sum();

                if block_max_score_upperbound <= inner_threshold {
                    block_max_was_too_low_advance_one_scorer(&mut self.scorers, pivot_len);
                    continue;
                }

                if !align_scorers(&mut self.scorers, pivot_doc, before_pivot_len) {
                    continue;
                }

                let score: Score = self.scorers[..pivot_len]
                    .iter_mut()
                    .map(|scorer| scorer.score())
                    .sum();
                advance_all_scorers_on_pivot(&mut self.scorers, pivot_len);

                if score > inner_threshold {
                    self.survivors.push(pivot_doc, score);
                }
            }

            let Some(last_candidate) = self.survivors.last_doc() else {
                self.current = (TERMINATED, Score::MIN);
                return TERMINATED;
            };
            if let Some(ref mut filter) = self.filter {
                let next_bound = self.survivors.filter_with(filter.as_mut());
                if next_bound == TERMINATED || next_bound > last_candidate.saturating_add(1) {
                    self.seek_scorers_to(next_bound);
                }
            }

            if let Some((doc, raw_score)) = self.survivors.next_above(inner_threshold) {
                self.current = (doc, raw_score + self.filter_boost);
                return doc;
            }
        }
    }
}
impl<T: BlockMaxScorer> Scorer for BlockWandUnionScorer<T> {
    #[inline]
    fn score(&mut self) -> Score {
        self.current.1
    }
}
impl<T: BlockMaxScorer> PruningScorer for BlockWandUnionScorer<T> {
    #[inline]
    fn set_threshold(&mut self, score: Score) {
        self.threshold = score;
    }
}
impl<T: BlockMaxScorer> DocSet for BlockWandUnionScorer<T> {
    fn advance(&mut self) -> DocId {
        if self.current.0 == TERMINATED {
            return TERMINATED;
        }
        // hoist threshold into a local while we loop to avoid going to memory
        let threshold = self.threshold;
        let inner_threshold = if threshold == Score::MIN {
            Score::MIN
        } else {
            (threshold - self.filter_boost) - Score::EPSILON * threshold.abs()
        };
        if self.filter_mode == WandFilterMode::AfterScoreBatch {
            return self.advance_after_score_batch(inner_threshold);
        }
        while let Some((before_pivot_len, pivot_len, pivot_doc)) =
            find_pivot_doc(&self.scorers[..], inner_threshold)
        {
            debug_assert!(self.scorers.iter().map(|scorer| scorer.doc()).is_sorted());
            debug_assert_ne!(pivot_doc, TERMINATED);
            debug_assert!(before_pivot_len < pivot_len);

            let block_max_score_upperbound: Score = self.scorers[..pivot_len]
                .iter_mut()
                .map(|scorer| {
                    scorer.seek_block(pivot_doc);
                    scorer.block_max_score()
                })
                .sum();

            // Beware after shallow advance, skip readers can be in advance compared to
            // the segment posting lists.
            //
            // `block_segment_postings.load_block()` need to be called separately.
            if block_max_score_upperbound <= inner_threshold {
                // Block max condition was not reached
                // We could get away by simply advancing the scorers to DocId + 1 but it would
                // be inefficient. The optimization requires proper explanation and was
                // isolated in a different function.
                block_max_was_too_low_advance_one_scorer(&mut self.scorers, pivot_len);
                continue;
            }

            // Embedded Filter Check (Early Rejection)
            // Test filter membership on pivot_doc BEFORE aligning any secondary scorers
            // when the filter is cheaper to probe than aligning and scoring.
            if self.filter_mode == WandFilterMode::Early {
                if let Some(ref mut filter) = self.filter {
                    match filter.seek_danger(pivot_doc) {
                        SeekDangerResult::Found => {}
                        SeekDangerResult::SeekLowerBound(bound) => {
                            // Crucial win: We DO NOT call align_scorers on secondary scorers.
                            // We advance or seek the pivot scorer past the non-matching document or
                            // range.
                            if bound > pivot_doc + 1 {
                                self.scorers[before_pivot_len].seek(bound);
                            } else {
                                self.scorers[before_pivot_len].advance();
                            }
                            if self.scorers[before_pivot_len].doc() == TERMINATED {
                                self.scorers.swap_remove(before_pivot_len);
                                self.scorers.sort_by_key(|scorer| scorer.doc());
                            } else {
                                restore_ordering(&mut self.scorers, before_pivot_len);
                            }
                            continue;
                        }
                    }
                }
            }

            // Block max condition is observed.
            //
            // Let's try and advance all scorers before the pivot to the pivot.
            if !align_scorers(&mut self.scorers, pivot_doc, before_pivot_len) {
                // At least one of the scorers does not contain the pivot.
                //
                // Let's stop scoring this pivot and go through the pivot selection again.
                // Note that the current pivot is not necessarily a bad candidate and it
                // may be picked again.
                continue;
            }

            // At this point, all scorers are positioned on the doc.
            let score: Score = self.scorers[..pivot_len]
                .iter_mut()
                .map(|scorer| scorer.score())
                .sum();

            if score > inner_threshold {
                let in_filter = if self.filter_mode == WandFilterMode::AfterScoreScalar {
                    match self.filter {
                        Some(ref mut filter) => {
                            filter.seek_danger(pivot_doc) == SeekDangerResult::Found
                        }
                        None => true,
                    }
                } else {
                    true
                };
                // Advance all scorers positioned on the pivot before returning or continuing so
                // future pivot_docs are strictly greater than this one.
                advance_all_scorers_on_pivot(&mut self.scorers, pivot_len);
                if in_filter {
                    self.current = (pivot_doc, score + self.filter_boost);
                    return pivot_doc;
                }
            } else {
                // let's advance all of the scorers that are currently positioned on the pivot.
                advance_all_scorers_on_pivot(&mut self.scorers, pivot_len);
            }
        }

        self.current = (TERMINATED, Score::MIN);
        TERMINATED
    }

    #[inline]
    fn doc(&self) -> DocId {
        self.current.0
    }

    /// The number of elements returned by a PruningScorer depends on the threshold and cannot be
    /// computed ahead of time, so just defer to the largest internal scorer
    fn size_hint(&self) -> u32 {
        self.scorers
            .iter()
            .map(|s| s.scorer.size_hint())
            .max()
            .unwrap_or(0u32)
    }
}

const SINGLE_WARMUP_BATCH: usize = 64;
const SINGLE_BLOCK_BATCH: usize = 128;

/// Specialized version of [`BlockWandUnionScorer`] for a single scorer.
/// In this case, the algorithm is simple, readable and faster (~ x3)
/// than the generic algorithm.
///
/// When an optional non-scoring filter docset is provided, entire blocks are skipped
/// if `filter.is_empty_in_range` indicates no matches in the block, and candidate
/// documents are tested against the filter before scoring (for low-probe-cost filters),
/// scalar after checking `score > threshold`, or in batches via `seek_danger_batch` when
/// `ProbeCost::prefers_batching()` is true.
///
/// The algorithm behaves as follows:
/// - While we don't hit the end of the docset:
///   - While the block max score is under the `threshold`, go to the next block.
///   - On a block, advance until the end and return the current doc when the doc score is greater
///     or equal to the `threshold`.
///
/// # Preconditions
/// - The scorer reads frequencies (`FreqReadingOption::ReadFreq`)
pub struct BlockWandSingleScorer<T: BlockMaxScorer = TermScorer> {
    scorer: T,
    filter: Option<Box<dyn DocSet>>,
    filter_mode: WandFilterMode,
    filter_boost: Score,
    threshold: Score,
    next_search_doc: DocId,
    survivors: ScoredSurvivorBuffer,
    current: (DocId, Score),
}
impl<T: BlockMaxScorer> BlockWandSingleScorer<T> {
    /// Construction positions `current` on the first match
    pub fn new(term_scorer: T, threshold: Score) -> Self {
        Self::with_filter_and_boost(term_scorer, threshold, None, 0.0)
    }

    /// Creates a new `BlockWandSingleScorer` with an embedded non-scoring filter.
    ///
    /// Entire blocks are skipped if `filter` has no matches in the block range.
    /// `filter_boost` is added to matching document scores and accounted for in the
    /// internal pruning threshold.
    pub fn with_filter(
        term_scorer: T,
        threshold: Score,
        filter: Box<dyn DocSet>,
        filter_boost: Score,
    ) -> Self {
        Self::with_filter_and_boost(term_scorer, threshold, Some(filter), filter_boost)
    }

    fn with_filter_and_boost(
        mut term_scorer: T,
        threshold: Score,
        filter: Option<Box<dyn DocSet>>,
        filter_boost: Score,
    ) -> Self {
        if let Some(filter) = &filter {
            seek_to_filter_start(&mut term_scorer, filter.as_ref());
        }
        let filter_mode = WandFilterMode::select(filter.as_deref(), ProbeCost::BITSET.scalar());
        let mut scorer = Self {
            scorer: term_scorer,
            filter,
            filter_mode,
            filter_boost,
            threshold,
            next_search_doc: 0,
            survivors: ScoredSurvivorBuffer::new(),
            current: (0, Score::MIN),
        };
        // advance to fill current
        scorer.advance();
        scorer
    }
}
impl<T: BlockMaxScorer> Scorer for BlockWandSingleScorer<T> {
    #[inline]
    fn score(&mut self) -> Score {
        self.current.1
    }
}
impl<T: BlockMaxScorer> PruningScorer for BlockWandSingleScorer<T> {
    #[inline]
    fn set_threshold(&mut self, score: Score) {
        self.threshold = score;
    }
}

impl<T: BlockMaxScorer> DocSet for BlockWandSingleScorer<T> {
    fn advance(&mut self) -> DocId {
        if self.doc() == TERMINATED {
            return TERMINATED;
        }
        // hoist threshold to a local so we avoid going to memory in the loop
        let threshold = self.threshold;
        let inner_threshold = if threshold == Score::MIN {
            Score::MIN
        } else {
            (threshold - self.filter_boost) - Score::EPSILON * threshold.abs()
        };
        if let Some((doc, raw_score)) = self.survivors.next_above(inner_threshold) {
            self.current = (doc, raw_score + self.filter_boost);
            return doc;
        }

        if self.scorer.doc() == TERMINATED || self.next_search_doc >= TERMINATED {
            self.current = (TERMINATED, Score::MIN);
            return TERMINATED;
        }
        let mut doc = self.scorer.doc().max(self.next_search_doc);
        if doc > self.scorer.last_doc_in_block() {
            self.scorer.seek_block(doc);
        }
        'outer: loop {
            if doc >= TERMINATED {
                self.current = (TERMINATED, Score::MIN);
                return TERMINATED;
            }
            // We position the scorer on a block that can reach
            // the threshold.
            while self.scorer.block_max_score() <= inner_threshold {
                let last_doc_in_block = self.scorer.last_doc_in_block();
                if last_doc_in_block == TERMINATED {
                    self.current = (TERMINATED, Score::MIN);
                    return TERMINATED;
                }
                doc = last_doc_in_block + 1;
                self.scorer.seek_block(doc);
            }
            // Block-level filter pruning:
            // If the filter has no matches in this 128-doc block, skip the entire block.
            let last_doc_in_block = self.scorer.last_doc_in_block();
            if let Some(ref mut filter) = self.filter {
                if filter.is_empty_in_range(doc, last_doc_in_block) {
                    if last_doc_in_block == TERMINATED {
                        self.current = (TERMINATED, Score::MIN);
                        return TERMINATED;
                    }
                    doc = last_doc_in_block + 1;
                    self.scorer.seek_block(doc);
                    continue;
                }
            }
            // Seek will effectively load that block.
            doc = self.scorer.seek(doc);
            if doc == TERMINATED {
                self.current = (TERMINATED, Score::MIN);
                return TERMINATED;
            }
            if self.filter_mode == WandFilterMode::AfterScoreBatch {
                let batch_cap = if threshold == Score::MIN {
                    SINGLE_WARMUP_BATCH
                } else {
                    SINGLE_BLOCK_BATCH
                };
                let block_end = self.scorer.last_doc_in_block();
                let next_doc;
                loop {
                    let score = self.scorer.score();
                    if score > inner_threshold {
                        self.survivors.push(doc, score);
                    }
                    debug_assert!(doc <= block_end);
                    if doc == block_end {
                        next_doc = if block_end == TERMINATED {
                            TERMINATED
                        } else {
                            block_end + 1
                        };
                        break;
                    }
                    doc = self.scorer.advance();
                    if doc == TERMINATED {
                        next_doc = TERMINATED;
                        break;
                    }
                    if self.survivors.len() >= batch_cap {
                        next_doc = doc;
                        break;
                    }
                }
                self.next_search_doc = next_doc;
                if !self.survivors.is_empty() {
                    if let Some(ref mut filter) = self.filter {
                        let next_bound = self.survivors.filter_with(filter.as_mut());
                        self.next_search_doc = self.next_search_doc.max(next_bound);
                    }
                    if let Some((doc, raw_score)) = self.survivors.next_above(inner_threshold) {
                        self.current = (doc, raw_score + self.filter_boost);
                        return doc;
                    }
                }
                doc = self.next_search_doc;
                if doc >= TERMINATED {
                    self.current = (TERMINATED, Score::MIN);
                    return TERMINATED;
                }
                if doc > self.scorer.last_doc_in_block() {
                    self.scorer.seek_block(doc);
                }
                continue 'outer;
            }
            loop {
                if self.filter_mode == WandFilterMode::AfterScoreScalar {
                    let score = self.scorer.score();
                    if score > inner_threshold {
                        let in_filter = match self.filter {
                            Some(ref mut filter) => {
                                filter.seek_danger(doc) == SeekDangerResult::Found
                            }
                            None => true,
                        };
                        if in_filter {
                            self.current = (doc, score + self.filter_boost);
                            self.scorer.advance();
                            break 'outer;
                        }
                    }
                } else {
                    let in_filter = match self.filter {
                        Some(ref mut filter) => filter.seek_danger(doc) == SeekDangerResult::Found,
                        None => true,
                    };
                    if in_filter {
                        let score = self.scorer.score();
                        if score > inner_threshold {
                            self.current = (doc, score + self.filter_boost);
                            self.scorer.advance();
                            break 'outer;
                        }
                    }
                }
                debug_assert!(doc <= self.scorer.last_doc_in_block());
                if doc == self.scorer.last_doc_in_block() {
                    break;
                }
                doc = self.scorer.advance();
                if doc == TERMINATED {
                    self.current = (TERMINATED, Score::MIN);
                    return TERMINATED;
                }
            }
            doc += 1;
            self.scorer.seek_block(doc);
        }
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.doc() >= target {
            return self.doc();
        }
        if target == TERMINATED {
            self.survivors.clear();
            self.current = (TERMINATED, Score::MIN);
            return TERMINATED;
        }
        if !self.survivors.skip_below(target) {
            self.next_search_doc = self.next_search_doc.max(target);
            if self.next_search_doc >= TERMINATED {
                self.current = (TERMINATED, Score::MIN);
                return TERMINATED;
            }
            if self.scorer.doc() < self.next_search_doc {
                self.scorer.seek_block(self.next_search_doc);
                self.scorer.seek(self.next_search_doc);
            }
        }
        self.advance()
    }

    #[inline]
    fn doc(&self) -> DocId {
        self.current.0
    }

    /// The number of elements returned by a PruningScorer depends on the threshold and cannot be
    /// computed ahead of time, so just defer to the internal scorer
    fn size_hint(&self) -> u32 {
        self.scorer.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    use proptest::prelude::*;

    use super::{BlockWandSingleScorer, BlockWandUnionScorer};
    use crate::index::Bm25Params;
    use crate::query::score_combiner::SumCombiner;
    use crate::query::term_query::TermScorer;
    use crate::query::weight::for_each_pruning_scorer;
    use crate::query::{Bm25Weight, BufferedUnionScorer, Scorer};
    use crate::{DocId, DocSet, ProbeCost, Score, SeekDangerResult, TERMINATED};

    #[test]
    fn test_block_wand_single_seek_termination() {
        let fieldnorms = vec![2; 4096];
        let postings: Vec<_> = (0..4096).map(|doc| (doc, 1)).collect();
        let weight = Bm25Weight::for_one_term(4096, 4096, 2.0, Bm25Params::default());
        let term = TermScorer::create_for_test(&postings, &fieldnorms, weight);
        let mut scorer = super::BlockWandSingleScorer::new(term, 0.0);
        let score = scorer.score();
        assert_eq!(scorer.doc(), 0);
        assert_eq!(scorer.seek(3000), 3000);
        assert_eq!(scorer.score(), score);
        assert_eq!(scorer.seek(3000), 3000);
        assert_eq!(scorer.score(), score);
        assert_eq!(scorer.advance(), 3001);
        let next_doc = scorer.scorer.doc();
        assert_eq!(scorer.seek(TERMINATED), TERMINATED);
        assert_eq!(scorer.scorer.doc(), next_doc);
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(scorer.seek(TERMINATED), TERMINATED);
        assert_eq!(scorer.scorer.doc(), next_doc);
    }

    #[test]
    fn test_block_wand_single_seek_matches_advance() {
        for len in [1, 127, 128, 129, 513] {
            let fieldnorms = vec![8; len * 3];
            let postings: Vec<_> = (0..len as u32)
                .map(|doc| (doc * 3, 1 + (doc / 128) % 2))
                .collect();
            let weight = Bm25Weight::for_one_term(
                len as u64,
                fieldnorms.len() as u64,
                8.0,
                Bm25Params::default(),
            );
            let term = TermScorer::create_for_test(&postings, &fieldnorms, weight.clone());
            for threshold in [Score::MIN, weight.score(8, 1), Score::MAX] {
                crate::postings::tests::test_skip_against_unoptimized(
                    || Box::new(super::BlockWandSingleScorer::new(term.clone(), threshold)),
                    vec![0, 1, 3, 381, 384, 385, 768, 769, 1536, 1537, TERMINATED],
                );
            }
        }
    }

    struct Float(Score);

    impl Eq for Float {}

    impl PartialEq for Float {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }

    impl PartialOrd for Float {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Float {
        fn cmp(&self, other: &Self) -> Ordering {
            other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
        }
    }

    fn nearly_equals(left: Score, right: Score) -> bool {
        (left - right).abs() < 0.0002 * (left + right).abs()
    }

    #[derive(Clone, Copy, Debug)]
    enum PruningMode {
        NoPruning,
        Wand,
        MaxScore(u32),
    }

    #[derive(Clone, Debug)]
    enum TestFilter {
        None,
        Range(f64, f64),
        EveryNth(u32),
    }

    impl TestFilter {
        fn to_docs(&self, max_doc: u32) -> Option<Vec<DocId>> {
            match self {
                TestFilter::None => None,
                TestFilter::Range(start_f, end_f) => {
                    let start = (*start_f * max_doc as f64) as DocId;
                    let end = (*end_f * max_doc as f64) as DocId;
                    let (min, max) = (start.min(end), start.max(end));
                    Some((min..=max.min(max_doc.saturating_sub(1))).collect())
                }
                TestFilter::EveryNth(n) => Some((0..max_doc).filter(|d| d % n == 0).collect()),
            }
        }
    }

    fn arb_test_filter() -> impl Strategy<Value = TestFilter> {
        prop_oneof![
            3 => Just(TestFilter::None),
            2 => (0.0f64..1.0f64, 0.0f64..1.0f64).prop_map(|(a, b)| TestFilter::Range(a, b)),
            1 => (2u32..10u32).prop_map(TestFilter::EveryNth),
        ]
    }

    struct FilterProbeStats {
        probes: std::sync::atomic::AtomicUsize,
        batch_calls: std::sync::atomic::AtomicUsize,
    }

    impl FilterProbeStats {
        fn new() -> std::sync::Arc<Self> {
            std::sync::Arc::new(Self {
                probes: std::sync::atomic::AtomicUsize::new(0),
                batch_calls: std::sync::atomic::AtomicUsize::new(0),
            })
        }

        fn snapshot(&self) -> (usize, usize) {
            (
                self.probes.load(std::sync::atomic::Ordering::Relaxed),
                self.batch_calls.load(std::sync::atomic::Ordering::Relaxed),
            )
        }
    }

    struct CostedFilter {
        inner: crate::query::VecDocSet,
        probe: ProbeCost,
        stats: std::sync::Arc<FilterProbeStats>,
    }

    impl CostedFilter {
        fn new(
            docs: Vec<DocId>,
            probe: ProbeCost,
            stats: std::sync::Arc<FilterProbeStats>,
        ) -> Self {
            Self {
                inner: crate::query::VecDocSet::from(docs),
                probe,
                stats,
            }
        }
    }

    impl DocSet for CostedFilter {
        fn advance(&mut self) -> DocId {
            self.inner.advance()
        }

        fn seek(&mut self, target: DocId) -> DocId {
            self.inner.seek(target)
        }

        fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
            self.stats
                .probes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.seek_danger(target)
        }

        fn seek_danger_batch(&mut self, docs: &[DocId], mask: &mut [common::TinySet]) -> DocId {
            self.stats
                .batch_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if docs.is_empty() {
                return 0;
            }
            let last = *docs.last().unwrap();
            let words = docs.len().div_ceil(64);
            let mut lower_bound = 0;
            for (word_idx, word) in mask[..words].iter_mut().enumerate() {
                if word.is_empty() {
                    continue;
                }
                let base = word_idx * 64;
                let mut keep = common::TinySet::EMPTY;
                for bit in *word {
                    let doc = docs[base + bit as usize];
                    if doc < lower_bound {
                        continue;
                    }
                    match self.seek_danger(doc) {
                        SeekDangerResult::Found => {
                            keep.insert_mut(bit);
                        }
                        SeekDangerResult::SeekLowerBound(bound) => lower_bound = bound,
                    }
                }
                *word = keep;
            }
            lower_bound.max(last.saturating_add(1)).min(TERMINATED)
        }

        fn doc(&self) -> DocId {
            self.inner.doc()
        }

        fn size_hint(&self) -> u32 {
            self.inner.size_hint()
        }

        fn cost(&self) -> crate::DocSetCost {
            crate::DocSetCost::new(crate::DriveCost(self.inner.size_hint() as u64), self.probe)
        }
    }

    fn compute_checkpoints(
        term_scorers: Vec<TermScorer>,
        n: usize,
        max_doc: u32,
        mode: PruningMode,
        filter_docs: Option<&[DocId]>,
    ) -> Vec<(DocId, Score)> {
        compute_checkpoints_with_probe(term_scorers, n, max_doc, mode, filter_docs, ProbeCost::TERM)
            .0
    }

    fn compute_checkpoints_with_probe(
        mut term_scorers: Vec<TermScorer>,
        n: usize,
        max_doc: u32,
        mode: PruningMode,
        filter_docs: Option<&[DocId]>,
        filter_probe: ProbeCost,
    ) -> (Vec<(DocId, Score)>, usize, usize) {
        let mut heap: BinaryHeap<Float> = BinaryHeap::with_capacity(n);
        let mut checkpoints: Vec<(DocId, Score)> = Vec::new();
        let mut limit: Score = 0.0;
        let stats = FilterProbeStats::new();

        let callback = &mut |doc, score| {
            heap.push(Float(score));
            if heap.len() > n {
                heap.pop().unwrap();
            }
            if heap.len() == n {
                limit = heap.peek().unwrap().0;
            }
            if !nearly_equals(score, limit) {
                checkpoints.push((doc, score));
            }
            limit
        };

        match mode {
            PruningMode::NoPruning => {
                return (
                    compute_checkpoints_manual(term_scorers, n, max_doc, filter_docs),
                    0,
                    0,
                );
            }
            PruningMode::Wand => {
                if let Some(docs) = filter_docs {
                    let filter = Box::new(CostedFilter::new(
                        docs.to_vec(),
                        filter_probe,
                        std::sync::Arc::clone(&stats),
                    ));
                    if term_scorers.len() == 1 {
                        let mut scorer = BlockWandSingleScorer::with_filter(
                            term_scorers.pop().unwrap(),
                            Score::MIN,
                            filter,
                            0.0,
                        );
                        for_each_pruning_scorer(&mut scorer, callback);
                    } else {
                        let mut scorer = BlockWandUnionScorer::with_filter(
                            term_scorers,
                            Score::MIN,
                            filter,
                            0.0,
                        );
                        for_each_pruning_scorer(&mut scorer, callback);
                    }
                } else {
                    super::block_wand(term_scorers, Score::MIN, callback);
                }
            }
            PruningMode::MaxScore(window) => {
                if let Some(docs) = filter_docs {
                    let mut filter = CostedFilter::new(
                        docs.to_vec(),
                        filter_probe,
                        std::sync::Arc::clone(&stats),
                    );
                    super::super::block_maxscore::block_maxscore_filtered(
                        term_scorers,
                        Score::MIN,
                        Some(&mut filter),
                        0.0,
                        window,
                        callback,
                    );
                } else {
                    super::super::block_maxscore::block_maxscore(
                        term_scorers,
                        Score::MIN,
                        window,
                        callback,
                    );
                }
            }
        }
        let (probes, batch_calls) = stats.snapshot();
        (checkpoints, probes, batch_calls)
    }

    fn compute_checkpoints_manual(
        term_scorers: Vec<TermScorer>,
        n: usize,
        max_doc: u32,
        filter_docs: Option<&[DocId]>,
    ) -> Vec<(DocId, Score)> {
        let mut heap: BinaryHeap<Float> = BinaryHeap::with_capacity(n);
        let mut checkpoints: Vec<(DocId, Score)> = Vec::new();
        let mut scorer = BufferedUnionScorer::build(term_scorers, SumCombiner::default, max_doc);

        let mut limit = Score::MIN;
        loop {
            if scorer.doc() == TERMINATED {
                break;
            }
            let doc = scorer.doc();
            let in_filter = match filter_docs {
                Some(docs) => docs.binary_search(&doc).is_ok(),
                None => true,
            };
            if in_filter {
                let score = scorer.score();
                if score > limit {
                    heap.push(Float(score));
                    if heap.len() > n {
                        heap.pop().unwrap();
                    }
                    if heap.len() == n {
                        limit = heap.peek().unwrap().0;
                    }
                    if !nearly_equals(score, limit) {
                        checkpoints.push((doc, score));
                    }
                }
            }
            scorer.advance();
        }
        checkpoints
    }

    const MAX_TERM_FREQ: u32 = 100u32;

    fn posting_list(max_doc: u32) -> BoxedStrategy<Vec<(DocId, u32)>> {
        (1..max_doc + 1)
            .prop_flat_map(move |doc_freq| {
                (
                    proptest::bits::bitset::sampled(doc_freq as usize, 0..max_doc as usize),
                    proptest::collection::vec(1u32..MAX_TERM_FREQ, doc_freq as usize),
                )
            })
            .prop_map(|(docset, term_freqs)| {
                docset
                    .iter()
                    .map(|doc| doc as u32)
                    .zip(term_freqs.iter().cloned())
                    .collect::<Vec<_>>()
            })
            .boxed()
    }

    #[expect(clippy::type_complexity)]
    fn gen_term_scorers(num_scorers: usize) -> BoxedStrategy<(Vec<Vec<(DocId, u32)>>, Vec<u32>)> {
        (1u32..200u32)
            .prop_flat_map(move |max_doc: u32| {
                (
                    proptest::collection::vec(posting_list(max_doc), num_scorers),
                    proptest::collection::vec(2u32..10u32 * MAX_TERM_FREQ, max_doc as usize),
                )
            })
            .boxed()
    }

    fn make_term_scorers(
        posting_lists: &[Vec<(DocId, u32)>],
        fieldnorms: &[u32],
    ) -> (Vec<TermScorer>, u32) {
        // We virtually repeat all docs 64 times in order to emulate blocks of 2 documents
        // and surface blogs more easily.
        const REPEAT: usize = 64;
        let fieldnorms_expanded = fieldnorms
            .iter()
            .cloned()
            .flat_map(|fieldnorm| std::iter::repeat_n(fieldnorm, REPEAT))
            .collect::<Vec<u32>>();

        let postings_lists_expanded: Vec<Vec<(DocId, u32)>> = posting_lists
            .iter()
            .map(|posting_list| {
                posting_list
                    .iter()
                    .cloned()
                    .flat_map(|(doc, term_freq)| {
                        (0_u32..REPEAT as u32).map(move |offset| {
                            (
                                doc * (REPEAT as u32) + offset,
                                if offset == 0 { term_freq } else { 1 },
                            )
                        })
                    })
                    .collect::<Vec<(DocId, u32)>>()
            })
            .collect::<Vec<_>>();

        let total_fieldnorms: u64 = fieldnorms_expanded
            .iter()
            .cloned()
            .map(|fieldnorm| fieldnorm as u64)
            .sum();
        let average_fieldnorm = (total_fieldnorms as Score) / (fieldnorms_expanded.len() as Score);
        let max_doc = fieldnorms_expanded.len();

        let term_scorers: Vec<TermScorer> = postings_lists_expanded
            .iter()
            .map(|postings| {
                let bm25_weight = Bm25Weight::for_one_term(
                    postings.len() as u64,
                    max_doc as u64,
                    average_fieldnorm,
                    Bm25Params::default(),
                );
                TermScorer::create_for_test(postings, &fieldnorms_expanded[..], bm25_weight)
            })
            .collect();
        (term_scorers, max_doc as u32)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1100))]
        #[test]
        fn test_disjunction_pruning(
            (posting_lists, fieldnorms) in prop_oneof![
                gen_term_scorers(1),
                gen_term_scorers(2),
                gen_term_scorers(10),
            ],
            test_filter in arb_test_filter(),
            filter_probe in prop_oneof![
                Just(ProbeCost::BITSET),
                Just(ProbeCost::per_doc(100)),
                Just(ProbeCost::DICT_BLOCK),
            ],
        ) {
            let (term_scorers, max_doc) = make_term_scorers(&posting_lists, &fieldnorms);
            let filter_docs = test_filter.to_docs(max_doc);
            for top_k in [1, 3, 10, 100] {
                let no_pruning =
                    compute_checkpoints(term_scorers.clone(), top_k, max_doc, PruningMode::NoPruning, filter_docs.as_deref());
                for mode in [
                    PruningMode::Wand,
                    PruningMode::MaxScore(0),
                    PruningMode::MaxScore(512),
                    PruningMode::MaxScore(4096),
                    PruningMode::MaxScore(super::super::block_maxscore::MIN_BOUND_WINDOW),
                ] {
                    let (actual, _, _) = compute_checkpoints_with_probe(
                        term_scorers.clone(),
                        top_k,
                        max_doc,
                        mode,
                        filter_docs.as_deref(),
                        filter_probe,
                    );
                    assert_eq!(actual.len(), no_pruning.len(), "{mode:?}, k={top_k}, filter={test_filter:?}, probe={filter_probe:?}");
                    for (&(doc, score), &(expected_doc, expected_score)) in
                        actual.iter().zip(&no_pruning)
                    {
                        assert_eq!(doc, expected_doc, "{mode:?}, k={top_k}, filter={test_filter:?}, probe={filter_probe:?}");
                        assert!(
                            nearly_equals(score, expected_score),
                            "{mode:?}, k={top_k}, filter={test_filter:?}, probe={filter_probe:?}: {score} != {expected_score}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn high_probe_cost_defers_filter_probes_across_wand_and_maxscore() {
        let num_docs = 8_192u32;
        let fieldnorms = vec![10u32; num_docs as usize];
        let make_scorer = |postings: &[(DocId, u32)]| {
            let bm25 = Bm25Weight::for_one_term(
                postings.len() as u64,
                num_docs as u64,
                10.0,
                Bm25Params::default(),
            );
            TermScorer::create_for_test(postings, &fieldnorms, bm25)
        };
        // Dense terms with `tf = 10` in `d < 64` so top-5 warms to `bm25_a(10) + bm25_b(10)`,
        // plus one non-overlapping `tf = 20` doc per block in each list (`d % 256 == 2`,
        // `d % 384 == 3`, `d % 640 == 5`, `d % 896 == 7`) so block-max upper bounds stay strictly
        // above threshold while actual per-doc scores for `d >= 64` fall below it.
        let postings_a: Vec<(DocId, u32)> = (0..num_docs)
            .step_by(2)
            .map(|d| {
                (
                    d,
                    if d < 64 {
                        10
                    } else if d % 256 == 2 {
                        20
                    } else {
                        1
                    },
                )
            })
            .collect();
        let postings_b: Vec<(DocId, u32)> = (0..num_docs)
            .step_by(3)
            .map(|d| {
                (
                    d,
                    if d < 64 {
                        10
                    } else if d % 384 == 3 {
                        20
                    } else {
                        1
                    },
                )
            })
            .collect();
        let postings_c: Vec<(DocId, u32)> = (0..num_docs)
            .step_by(5)
            .map(|d| {
                (
                    d,
                    if d < 64 {
                        10
                    } else if d % 640 == 5 {
                        20
                    } else {
                        1
                    },
                )
            })
            .collect();
        let postings_d: Vec<(DocId, u32)> = (0..num_docs)
            .step_by(7)
            .map(|d| {
                (
                    d,
                    if d < 64 {
                        10
                    } else if d % 896 == 7 {
                        20
                    } else {
                        1
                    },
                )
            })
            .collect();
        let filter_docs: Vec<DocId> = (0..num_docs).filter(|d| d % 4 != 3).collect();

        let cases: [(&str, Vec<TermScorer>, PruningMode); 3] = [
            (
                "single_wand",
                vec![make_scorer(&postings_a)],
                PruningMode::Wand,
            ),
            (
                "union_wand",
                vec![make_scorer(&postings_a), make_scorer(&postings_b)],
                PruningMode::Wand,
            ),
            (
                "block_maxscore",
                vec![
                    make_scorer(&postings_a),
                    make_scorer(&postings_b),
                    make_scorer(&postings_c),
                    make_scorer(&postings_d),
                ],
                PruningMode::MaxScore(super::super::block_maxscore::MIN_BOUND_WINDOW),
            ),
        ];

        for (name, scorers, mode) in cases {
            let (early_cp, early_probes, _) = compute_checkpoints_with_probe(
                scorers.clone(),
                5,
                num_docs,
                mode,
                Some(&filter_docs),
                ProbeCost::BITSET,
            );
            let (deferred_cp, deferred_probes, deferred_batches) = compute_checkpoints_with_probe(
                scorers,
                5,
                num_docs,
                mode,
                Some(&filter_docs),
                ProbeCost::DICT_BLOCK,
            );
            assert_eq!(early_cp.len(), deferred_cp.len(), "{name}");
            assert!(
                deferred_probes < early_probes,
                "{name}: expected deferred filter probes ({deferred_probes}) < early \
                 ({early_probes})"
            );
            assert!(
                deferred_batches > 0 && deferred_batches < deferred_probes,
                "{name}: expected batched filter calls ({deferred_batches}) in (0, \
                 {deferred_probes})"
            );
        }

        // Full streaming scan (`threshold == Score::MIN`, as in Scanner) with `filter_boost`:
        // `BlockWandUnionScorer` should ramp batch sizes `64 -> 256 -> 1024` rather than
        // stalling at 64 or calling scalar `seek_danger`.
        let stats = FilterProbeStats::new();
        let filter = Box::new(CostedFilter::new(
            filter_docs.clone(),
            ProbeCost::DICT_BLOCK,
            std::sync::Arc::clone(&stats),
        ));
        let filter_boost = 1.5;
        let mut union = BlockWandUnionScorer::with_filter(
            vec![make_scorer(&postings_a), make_scorer(&postings_b)],
            Score::MIN,
            filter,
            filter_boost,
        );
        let mut got = Vec::new();
        while union.doc() != TERMINATED {
            got.push((union.doc(), union.score()));
            union.advance();
        }
        let mut expected_union = BufferedUnionScorer::build(
            vec![make_scorer(&postings_a), make_scorer(&postings_b)],
            SumCombiner::default,
            num_docs,
        );
        let mut expected = Vec::new();
        while expected_union.doc() != TERMINATED {
            let doc = expected_union.doc();
            if filter_docs.binary_search(&doc).is_ok() {
                expected.push((doc, expected_union.score() + filter_boost));
            }
            expected_union.advance();
        }
        assert_eq!(got.len(), expected.len());
        for (&(d, s), &(ed, es)) in got.iter().zip(&expected) {
            assert_eq!(d, ed);
            assert!(nearly_equals(s, es), "{s} != {es}");
        }
        let (_probes, batch_calls) = stats.snapshot();
        // ~5,462 union candidates across 8,192 docs: 64 + 256 + 6 * 1024 -> <= 10 batch calls.
        assert!(
            batch_calls > 0 && batch_calls <= 10,
            "expected ramped union batch calls in (0, 10], got {batch_calls}"
        );
    }

    #[test]
    fn test_advance_single_pivot_ties_and_termination() {
        for (first, expected) in [
            (vec![(0, 1), (5, 1)], vec![(2, 2), (5, 1), (5, 3), (5, 4)]),
            (vec![(0, 1)], vec![(2, 2), (5, 4), (5, 3)]),
        ] {
            let mut scorers: Vec<super::TermScorerWithMaxScore> =
                [first, vec![(2, 2)], vec![(5, 3)], vec![(5, 4)]]
                    .iter()
                    .map(|postings| {
                        TermScorer::create_for_test(
                            postings,
                            &[4; 6],
                            Bm25Weight::for_one_term(
                                postings.len() as u64,
                                6,
                                4.0,
                                Bm25Params::default(),
                            ),
                        )
                        .into()
                    })
                    .collect();
            super::advance_all_scorers_on_pivot(&mut scorers, 1);
            assert_eq!(
                scorers
                    .iter()
                    .map(|scorer| (scorer.doc(), scorer.term_freq()))
                    .collect::<Vec<_>>(),
                expected
            );
            scorers.truncate(1);
            super::advance_all_scorers_on_pivot(&mut scorers, 1);
            assert!(scorers.is_empty());
        }
    }

    #[test]
    fn test_fn_reproduce_proptest() {
        let postings_lists = &[
            vec![
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (6, 1),
                (7, 7),
                (8, 1),
                (10, 1),
                (12, 1),
                (13, 1),
                (14, 1),
                (15, 1),
                (16, 1),
                (19, 1),
                (20, 1),
                (21, 1),
                (22, 1),
                (24, 1),
                (25, 1),
                (26, 1),
                (28, 1),
                (30, 1),
                (31, 1),
                (33, 1),
                (34, 1),
                (35, 1),
                (36, 95),
                (37, 1),
                (39, 1),
                (41, 1),
                (44, 1),
                (46, 1),
            ],
            vec![
                (0, 5),
                (2, 1),
                (4, 1),
                (5, 84),
                (6, 47),
                (7, 26),
                (8, 50),
                (9, 34),
                (11, 73),
                (12, 11),
                (13, 51),
                (14, 45),
                (15, 18),
                (18, 60),
                (19, 80),
                (20, 63),
                (23, 79),
                (24, 69),
                (26, 35),
                (28, 82),
                (29, 19),
                (30, 2),
                (31, 7),
                (33, 40),
                (34, 1),
                (35, 33),
                (36, 27),
                (37, 24),
                (38, 65),
                (39, 32),
                (40, 85),
                (41, 1),
                (42, 69),
                (43, 11),
                (45, 45),
                (47, 97),
            ],
            vec![
                (2, 1),
                (4, 1),
                (7, 94),
                (8, 1),
                (9, 1),
                (10, 1),
                (12, 1),
                (15, 1),
                (22, 1),
                (23, 1),
                (26, 1),
                (27, 1),
                (32, 1),
                (33, 1),
                (34, 1),
                (36, 96),
                (39, 1),
                (41, 1),
            ],
        ];
        let fieldnorms = &[
            685, 239, 780, 564, 664, 827, 5, 56, 930, 887, 263, 665, 167, 127, 120, 919, 292, 92,
            489, 734, 814, 724, 700, 304, 128, 779, 311, 877, 774, 15, 866, 368, 894, 371, 982,
            502, 507, 669, 680, 76, 594, 626, 578, 331, 170, 639, 665, 186,
        ][..];
        let (term_scorers, max_doc) = make_term_scorers(postings_lists, fieldnorms);
        for top_k in [1, 3, 10, 100] {
            let no_pruning = compute_checkpoints(
                term_scorers.clone(),
                top_k,
                max_doc,
                PruningMode::NoPruning,
                None,
            );
            for mode in [
                PruningMode::Wand,
                PruningMode::MaxScore(0),
                PruningMode::MaxScore(512),
                PruningMode::MaxScore(4096),
            ] {
                let actual = compute_checkpoints(term_scorers.clone(), top_k, max_doc, mode, None);
                assert_eq!(actual.len(), no_pruning.len(), "{mode:?}, k={top_k}");
                for (&(doc, score), &(expected_doc, expected_score)) in
                    actual.iter().zip(&no_pruning)
                {
                    assert_eq!(doc, expected_doc, "{mode:?}, k={top_k}");
                    assert!(
                        nearly_equals(score, expected_score),
                        "{mode:?}, k={top_k}: {score} != {expected_score}"
                    );
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]
        #[ignore]
        #[test]
        #[ignore]
        fn test_block_wand_three_term_scorers((posting_lists, fieldnorms) in gen_term_scorers(3)) {
            let (term_scorers, max_doc) = make_term_scorers(&posting_lists, &fieldnorms);
            for top_k in [1, 3, 10, 100] {
                let no_pruning =
                    compute_checkpoints(term_scorers.clone(), top_k, max_doc, PruningMode::NoPruning, None);
                for mode in [
                    PruningMode::Wand,
                    PruningMode::MaxScore(0),
                    PruningMode::MaxScore(512),
                    PruningMode::MaxScore(4096),
                ] {
                    let actual = compute_checkpoints(term_scorers.clone(), top_k, max_doc, mode, None);
                    assert_eq!(actual.len(), no_pruning.len(), "{mode:?}, k={top_k}");
                    for (&(doc, score), &(expected_doc, expected_score)) in
                        actual.iter().zip(&no_pruning)
                    {
                        assert_eq!(doc, expected_doc, "{mode:?}, k={top_k}");
                        assert!(
                            nearly_equals(score, expected_score),
                            "{mode:?}, k={top_k}: {score} != {expected_score}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_disjunction_pruning_filtered_regression() {
        let posting_lists: Vec<Vec<(DocId, u32)>> = vec![
            vec![(15, 1)],
            vec![(7, 1)],
            vec![(10, 1)],
            vec![(11, 1)],
            vec![(12, 1)],
            vec![(3, 1)],
            vec![(3, 1)],
            vec![(0, 1)],
            vec![(7, 1)],
            vec![
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 6),
                (4, 82),
                (5, 81),
                (6, 20),
                (7, 60),
                (8, 43),
                (9, 91),
                (10, 27),
                (11, 79),
                (12, 61),
                (13, 11),
                (14, 87),
                (15, 79),
                (16, 35),
            ],
        ];
        let fieldnorms = vec![
            527, 53, 411, 364, 624, 412, 785, 548, 844, 106, 440, 32, 830, 805, 832, 944, 436,
        ];
        let test_filter = TestFilter::Range(0.7456032710964235, 0.1400998666088333);
        let (term_scorers, max_doc) = make_term_scorers(&posting_lists, &fieldnorms);
        let filter_docs = test_filter.to_docs(max_doc);
        let top_k = 100;
        let no_pruning = compute_checkpoints(
            term_scorers.clone(),
            top_k,
            max_doc,
            PruningMode::NoPruning,
            filter_docs.as_deref(),
        );
        let actual = compute_checkpoints(
            term_scorers.clone(),
            top_k,
            max_doc,
            PruningMode::MaxScore(0),
            filter_docs.as_deref(),
        );
        assert_eq!(actual.len(), no_pruning.len());
        for (&(doc, score), &(expected_doc, expected_score)) in actual.iter().zip(&no_pruning) {
            assert_eq!(doc, expected_doc);
            assert!(nearly_equals(score, expected_score));
        }
    }
}
