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
    mut threshold: Score,
    min_window: u32,
    callback: &mut dyn FnMut(DocId, Score) -> Score,
) {
    // Keep bounds conservative relative to rounding in the collector's f32 scores.
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
                let handled =
                    scorer.for_each_block_slice_with_term_norms(window_end, |docs, sc| {
                        for (&doc, &score) in docs.iter().zip(sc) {
                            let score_f64 = score as f64;
                            let keep = score_f64 * rounding + weak_bound > threshold as f64;
                            matches[matches_len] = (doc, score_f64);
                            matches_len += keep as usize;
                        }
                    });
                if !handled {
                    while scorer.doc() < window_end {
                        let score = scorer.score() as f64;
                        let keep = score * rounding + weak_bound > threshold as f64;
                        matches[matches_len] = (scorer.doc(), score);
                        matches_len += keep as usize;
                        scorer.advance();
                    }
                }
            } else {
                // Accumulate strong terms into a dense batch; the bitmap tracks touched entries.
                for term in strong.iter_mut() {
                    let handled =
                        term.scorer
                            .for_each_block_slice_with_term_norms(window_end, |docs, sc| {
                                for (&doc, &score) in docs.iter().zip(sc) {
                                    let offset = (doc - base) as usize;
                                    candidates[offset / 64] |= 1u64 << (offset % 64);
                                    scores[offset] += score as f64;
                                }
                            });
                    if !handled {
                        while term.scorer.doc() < window_end {
                            let offset = (term.scorer.doc() - base) as usize;
                            candidates[offset / 64] |= 1u64 << (offset % 64);
                            scores[offset] += term.scorer.score() as f64;
                            term.scorer.advance();
                        }
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
            // Add weak terms in reverse order, pruning with the bounds of the unvisited prefix.
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

#[cfg(test)]
mod tests {
    use crate::collector::TopDocs;
    use crate::query::BooleanQuery;
    use crate::schema::{Schema, TEXT};
    use crate::Index;

    #[test]
    fn test_block_maxscore_with_pnorms_matches_standard() -> crate::Result<()> {
        let mut top_results = Vec::new();
        for pnorms in [false, true] {
            let mut schema = Schema::builder();
            let text = schema.add_text_field(
                "text",
                TEXT.set_indexing_options(
                    TEXT.get_indexing_options()
                        .unwrap()
                        .clone()
                        .set_pnorms(pnorms),
                ),
            );
            let index = Index::create_in_ram(schema.build());
            let mut writer = index.writer_for_tests()?;
            for id in 0..400 {
                let mut words = Vec::new();
                if id % 2 == 0 {
                    words.push("apple");
                }
                if id % 3 == 0 {
                    words.push("banana");
                }
                if id % 5 == 0 {
                    words.push("cherry");
                }
                let padding = "fruit ".repeat((id % 17) + 1);
                writer.add_document(doc!(text => format!("{} {}", words.join(" "), padding)))?;
            }
            writer.commit()?;

            let reader = index.reader()?;
            let searcher = reader.searcher();
            let query = BooleanQuery::union(vec![
                Box::new(crate::query::TermQuery::new(
                    crate::Term::from_field_text(text, "apple"),
                    crate::schema::IndexRecordOption::WithFreqs,
                )),
                Box::new(crate::query::TermQuery::new(
                    crate::Term::from_field_text(text, "banana"),
                    crate::schema::IndexRecordOption::WithFreqs,
                )),
                Box::new(crate::query::TermQuery::new(
                    crate::Term::from_field_text(text, "cherry"),
                    crate::schema::IndexRecordOption::WithFreqs,
                )),
            ]);

            let top_docs = searcher.search(&query, &TopDocs::with_limit(50).order_by_score())?;
            assert!(!top_docs.is_empty());

            // Verify top docs have valid positive decreasing scores
            let mut prev_score = f32::INFINITY;
            for &(score, _doc_address) in &top_docs {
                assert!(score > 0.0);
                assert!(score <= prev_score);
                prev_score = score;
            }
            top_results.push(top_docs);
        }

        // Both pnorms and standard Tantivy should retrieve the same top hits.
        // Due to 1-byte fieldnorm quantization vs exact pnorm lengths, scores may vary slightly,
        // but the top matching documents overlap heavily.
        let docs_no_pnorms: Vec<_> = top_results[0].iter().map(|&(_score, doc)| doc).collect();
        let docs_pnorms: Vec<_> = top_results[1].iter().map(|&(_score, doc)| doc).collect();
        let common = docs_no_pnorms
            .iter()
            .filter(|doc| docs_pnorms.contains(doc))
            .count();
        assert!(common >= 45, "Expected high top-k overlap, got {common}/50");

        Ok(())
    }
}
