use crate::query::term_query::TermScorer;
use crate::query::Scorer;
use crate::{DocId, DocSet, Score, TERMINATED};

const WINDOW: usize = 4096;

pub(super) fn should_use_block_maxscore(scorers: &[TermScorer], max_doc: DocId) -> bool {
    let postings: u64 = scorers
        .iter()
        .map(|scorer| u64::from(scorer.size_hint()))
        .sum();
    let density = 256;
    scorers.len() >= 3 && postings >= 256 && postings.saturating_mul(density) >= u64::from(max_doc)
}

struct Term {
    scorer: Box<TermScorer>,
    bound: f64,
    inv_cost: f64,
    priority: f64,
    remaining: f64,
}

pub(super) fn block_maxscore(
    scorers: Vec<TermScorer>,
    mut threshold: Score,
    min_window: u32,
    callback: &mut dyn FnMut(DocId, Score) -> Score,
) {
    let rounding = 1.0 + scorers.len() as f64 * Score::EPSILON as f64;
    let mut terms: Vec<_> = scorers
        .into_iter()
        .map(|scorer| Term {
            inv_cost: 1.0 / scorer.size_hint().max(1) as f64,
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
        let target = start.saturating_add(min_window.saturating_sub(1));
        let mut end = TERMINATED;
        for term in &mut terms {
            term.scorer.seek_block(start);
            let (bound, last) = term.scorer.block_max_score_up_to(target);
            term.bound = bound as f64 * rounding;
            term.priority = term.bound * term.inv_cost;
            end = end.min(last.saturating_add(1));
        }
        terms.sort_unstable_by(|a, b| a.priority.total_cmp(&b.priority));
        let mut bound = 0.0;
        let mut essential = 0;
        for term in &mut terms {
            term.remaining = bound;
            bound += term.bound;
            if bound <= threshold as f64 {
                essential += 1;
            }
        }
        if essential == terms.len() {
            start = end;
            continue;
        }
        let (weak, strong) = terms.split_at_mut(essential);
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
                while scorer.doc() < window_end {
                    matches[matches_len] = (scorer.doc(), scorer.score() as f64);
                    matches_len += 1;
                    scorer.advance();
                }
            } else {
                for term in strong.iter_mut() {
                    while term.scorer.doc() < window_end {
                        let offset = (term.scorer.doc() - base) as usize;
                        candidates[offset / 64] |= 1u64 << (offset % 64);
                        scores[offset] += term.scorer.score() as f64;
                        term.scorer.advance();
                    }
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
            for term in weak.iter_mut().rev() {
                let mut len = 0;
                for i in 0..matches_len {
                    let (doc, mut score) = matches[i];
                    if score * rounding + term.bound + term.remaining <= threshold as f64 {
                        continue;
                    }
                    if term.scorer.doc() < doc {
                        term.scorer.seek(doc);
                    }
                    if term.scorer.doc() == doc {
                        score += term.scorer.score() as f64;
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
