use crate::query::term_query::TermScorer;
use crate::query::Scorer;
use crate::{DocId, DocSet, Score, TERMINATED};

// Benchmarks favored 8192-doc windows to reduce bound setup work.
// Scoring uses separate 4096-entry buffers.
pub(super) const MIN_BOUND_WINDOW: u32 = 8192;
const WINDOW: usize = 4096;

pub(super) fn should_use_block_maxscore(scorers: &[TermScorer], max_doc: DocId) -> bool {
    if scorers.len() == 2 {
        let left = u64::from(scorers[0].size_hint());
        let right = u64::from(scorers[1].size_hint());
        if left.min(right) * 64 < left.max(right) {
            return false;
        }
    }
    // Keep WAND for a single term or fewer than 256 postings.
    // Require one posting per 256 doc IDs to avoid sparse-query regressions:
    // 32 expected term matches per 8192-doc window, counting overlapping terms.
    let postings: u64 = scorers
        .iter()
        .map(|scorer| u64::from(scorer.size_hint()))
        .sum();
    let max_docs_per_posting = 256;
    scorers.len() >= 2
        && postings >= 256
        && postings.saturating_mul(max_docs_per_posting) >= u64::from(max_doc)
}

pub(super) trait BlockMaxScorer: Scorer {
    fn seek_block(&mut self, target: DocId);
    fn block_max_score_up_to(&mut self, target: DocId) -> (Score, DocId);
    fn block_score_hint(&self) -> Score;
    fn refine_block_max_score(&mut self) -> Score;
    fn for_each_score_until(&mut self, end: DocId, callback: impl FnMut(DocId, Score));

    #[inline]
    fn score_at(&mut self, doc: DocId) -> Option<Score> {
        if self.doc() < doc {
            self.seek(doc);
        }
        (doc != TERMINATED && self.doc() == doc).then(|| self.score())
    }
}

impl BlockMaxScorer for TermScorer {
    #[inline]
    fn seek_block(&mut self, target: DocId) {
        self.seek_block(target);
    }

    #[inline]
    fn block_max_score_up_to(&mut self, target: DocId) -> (Score, DocId) {
        self.block_max_score_up_to(target)
    }

    #[inline]
    fn block_score_hint(&self) -> Score {
        self.block_score_hint()
    }

    #[inline]
    fn refine_block_max_score(&mut self) -> Score {
        self.refine_block_max_score()
    }

    #[inline]
    fn for_each_score_until(&mut self, end: DocId, callback: impl FnMut(DocId, Score)) {
        self.for_each_score_until(end, callback);
    }
}

struct Term<TScorer> {
    scorer: Box<TScorer>,
    bound: f64,
    inv_cost: f64,
    priority: f64,
    remaining: f64,
}

pub(super) fn block_maxscore<TScorer: BlockMaxScorer>(
    scorers: Vec<TScorer>,
    mut threshold: Score,
    min_window: u32,
    callback: &mut dyn FnMut(DocId, Score) -> Score,
) {
    // Keep bounds conservative relative to rounding in the collector's f32 scores.
    let rounding = 1.0 + scorers.len() as f64 * Score::EPSILON as f64;
    let mut terms: Vec<_> = scorers
        .into_iter()
        .map(|scorer| Term {
            inv_cost: 1.0 / scorer.cost().max(1) as f64,
            priority: 0.0,
            scorer: Box::new(scorer),
            bound: 0.0,
            remaining: 0.0,
        })
        .collect();
    let mut scores = vec![0.0f64; WINDOW];
    let mut candidates = [0u64; WINDOW / 64];
    let mut matches = vec![(0, 0.0); WINDOW];
    let mut start = terms
        .iter()
        .map(|t| t.scorer.doc())
        .min()
        .unwrap_or(TERMINATED);
    while start < TERMINATED {
        terms.retain(|term| term.scorer.doc() < TERMINATED);
        if terms.is_empty() {
            break;
        }
        // Reuse each term's maximum over whole posting blocks covering the outer window.
        let target = start.saturating_add(min_window.saturating_sub(1));
        let mut end = TERMINATED;
        for term in &mut terms {
            term.scorer.seek_block(start);
            let (bound, last) = term.scorer.block_max_score_up_to(target);
            term.bound = bound as f64 * rounding;
            term.priority = term.bound * term.inv_cost;
            end = end.min(last.saturating_add(1));
        }
        if min_window == 0
            && terms.iter().map(|term| term.bound).sum::<f64>() > threshold as f64
            && terms
                .iter()
                .map(|term| term.scorer.block_score_hint() as f64)
                .sum::<f64>()
                * rounding
                <= threshold as f64
        {
            for term in &mut terms {
                term.bound = term.scorer.refine_block_max_score() as f64 * rounding;
                term.priority = term.bound * term.inv_cost;
            }
        }
        // Low bound per posting cost goes first: these terms are candidates for deferred scoring.
        terms.sort_unstable_by(|a, b| a.priority.total_cmp(&b.priority));
        let mut bound = 0.0;
        let mut weak_count = 0;
        for term in &mut terms {
            term.remaining = bound;
            bound += term.bound;
            if bound <= threshold as f64 {
                weak_count += 1;
            }
        }
        if weak_count == terms.len() {
            start = end;
            continue;
        }
        // The weak prefix cannot beat the threshold alone, so every hit must match a strong term.
        let (weak, strong) = terms.split_at_mut(weak_count);
        let weak_bound = weak.iter().map(|t| t.bound).sum::<f64>();
        for term in strong.iter_mut() {
            if term.scorer.doc() < start {
                term.scorer.seek(start);
            }
        }
        loop {
            let base = strong.iter().map(|t| t.scorer.doc()).min().unwrap();
            if base >= end {
                break;
            }
            let window_end = end.min(base.saturating_add(WINDOW as u32));
            let mut matches_len = 0;
            if strong.len() == 1 {
                let scorer = &mut strong[0].scorer;
                scorer.for_each_score_until(window_end, |doc, score| {
                    let score = score as f64;
                    let keep = score * rounding + weak_bound > threshold as f64;
                    matches[matches_len] = (doc, score);
                    matches_len += keep as usize;
                });
            } else {
                // Accumulate strong terms into a dense batch; the bitmap tracks touched entries.
                for term in strong.iter_mut() {
                    term.scorer.for_each_score_until(window_end, |doc, score| {
                        let offset = (doc - base) as usize;
                        candidates[offset / 64] |= 1u64 << (offset % 64);
                        scores[offset] += score as f64;
                    });
                }
                for (word, bits) in candidates.iter_mut().enumerate() {
                    while *bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        *bits &= *bits - 1;
                        let offset = word * 64 + bit;
                        let score = scores[offset];
                        scores[offset] = 0.0;
                        let keep = score * rounding + weak_bound > threshold as f64;
                        matches[matches_len] = (base + offset as u32, score);
                        matches_len += keep as usize;
                    }
                }
            }
            // Add weak terms in reverse order, pruning with the bounds of the unvisited prefix.
            for term in weak.iter_mut().rev() {
                let mut len = 0;
                for i in 0..matches_len {
                    let (doc, mut score) = matches[i];
                    if score * rounding + term.bound + term.remaining <= threshold as f64 {
                        continue;
                    }
                    if let Some(contribution) = term.scorer.score_at(doc) {
                        score += contribution as f64;
                    }
                    let keep = score * rounding + term.remaining > threshold as f64;
                    matches[len] = (doc, score);
                    len += keep as usize;
                }
                matches_len = len;
            }
            for &(doc, score) in &matches[..matches_len] {
                if score as Score > threshold {
                    threshold = callback(doc, score as Score);
                }
            }
        }
        start = end;
    }
}
