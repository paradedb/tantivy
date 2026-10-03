use crate::docset::SeekDangerResult;
use crate::query::term_query::TermScorer;
use crate::query::Scorer;
use crate::{DocId, DocSet, Score, TERMINATED};

// Benchmarks favored 8192-doc windows to reduce bound setup work.
// Scoring uses separate 4096-entry buffers.
pub(super) const MIN_BOUND_WINDOW: u32 = 8192;
const WINDOW: usize = 4096;

pub(super) fn should_use_block_maxscore(scorers: &[TermScorer], max_doc: DocId) -> bool {
    // Benchmark cutoffs keep WAND for 1-2 terms or fewer than 256 postings.
    // Require one posting per 256 doc IDs to avoid sparse-query regressions:
    // 32 expected term matches per 8192-doc window, counting overlapping terms.
    let postings: u64 = scorers
        .iter()
        .map(|scorer| u64::from(scorer.size_hint()))
        .sum();
    let max_docs_per_posting = 256;
    scorers.len() >= 3
        && postings >= 256
        && postings.saturating_mul(max_docs_per_posting) >= u64::from(max_doc)
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
    threshold: Score,
    min_window: u32,
    callback: &mut dyn FnMut(DocId, Score) -> Score,
) {
    block_maxscore_filtered(
        scorers,
        threshold,
        None::<&mut dyn DocSet>,
        0.0,
        min_window,
        callback,
    );
}

pub(super) fn block_maxscore_filtered<F: DocSet + ?Sized>(
    scorers: Vec<TermScorer>,
    mut threshold: Score,
    mut filter: Option<&mut F>,
    filter_boost: Score,
    min_window: u32,
    callback: &mut dyn FnMut(DocId, Score) -> Score,
) {
    // Keep bounds conservative relative to rounding in the collector's f32 scores.
    let rounding = 1.0 + (scorers.len() + 1) as f64 * Score::EPSILON as f64;
    let mut inner_threshold: f64 = if threshold == Score::MIN {
        Score::MIN as f64
    } else {
        (threshold - filter_boost) as f64 - (Score::EPSILON * threshold.abs()) as f64
    };
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
    let mut scores = vec![0.0 as Score; WINDOW];
    let mut candidates = [0u64; WINDOW / 64];
    let mut matches = vec![(0, 0.0 as Score); WINDOW];
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
        if let Some(ref mut f) = filter {
            if f.is_empty_in_range(start, end.saturating_sub(1)) {
                start = end;
                continue;
            }
        }
        // Low bound per posting cost goes first: these terms are candidates for deferred scoring.
        terms.sort_unstable_by(|a, b| a.priority.total_cmp(&b.priority));
        let mut bound = 0.0;
        let mut weak_count = 0;
        for term in &mut terms {
            term.remaining = bound;
            bound += term.bound;
            if bound <= inner_threshold {
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
                while scorer.doc() < window_end {
                    let doc = scorer.doc();
                    let in_filter = match filter {
                        Some(ref mut f) => f.seek_danger(doc) == SeekDangerResult::Found,
                        None => true,
                    };
                    if in_filter {
                        let score = scorer.score();
                        let keep = score as f64 * rounding + weak_bound > inner_threshold;
                        matches[matches_len] = (doc, score);
                        matches_len += keep as usize;
                    }
                    scorer.advance();
                }
            } else {
                // Accumulate strong terms into a dense batch; the bitmap tracks touched entries.
                for term in strong.iter_mut() {
                    while term.scorer.doc() < window_end {
                        let offset = (term.scorer.doc() - base) as usize;
                        candidates[offset / 64] |= 1u64 << (offset % 64);
                        scores[offset] += term.scorer.score();
                        term.scorer.advance();
                    }
                }
                let num_words = ((window_end - base) as usize + 63) / 64;
                for (word, bits) in candidates[..num_words].iter_mut().enumerate() {
                    while *bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        *bits &= *bits - 1;
                        let offset = word * 64 + bit;
                        let doc = base + offset as u32;
                        let score = scores[offset];
                        scores[offset] = 0.0;
                        let in_filter = match filter {
                            Some(ref mut f) => f.seek_danger(doc) == SeekDangerResult::Found,
                            None => true,
                        };
                        if in_filter {
                            let keep = score as f64 * rounding + weak_bound > inner_threshold;
                            matches[matches_len] = (doc, score);
                            matches_len += keep as usize;
                        }
                    }
                }
            }
            // Add weak terms in reverse order, pruning with the bounds of the unvisited prefix.
            for term in weak.iter_mut().rev() {
                let mut len = 0;
                for i in 0..matches_len {
                    let (doc, mut score) = matches[i];
                    if score as f64 * rounding + term.bound + term.remaining <= inner_threshold {
                        continue;
                    }
                    if term.scorer.doc() < doc {
                        term.scorer.seek(doc);
                    }
                    if term.scorer.doc() == doc {
                        score += term.scorer.score();
                    }
                    let keep = score as f64 * rounding + term.remaining > inner_threshold;
                    matches[len] = (doc, score);
                    len += keep as usize;
                }
                matches_len = len;
            }
            for &(doc, score) in &matches[..matches_len] {
                let final_score = score + filter_boost;
                if final_score > threshold {
                    threshold = callback(doc, final_score);
                    inner_threshold = if threshold == Score::MIN {
                        Score::MIN as f64
                    } else {
                        (threshold - filter_boost) as f64
                            - (Score::EPSILON * threshold.abs()) as f64
                    };
                }
            }
        }
        start = end;
    }
}
