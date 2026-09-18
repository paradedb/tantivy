use std::io;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::phrase_scorer::PhrasePruningScorer;
use super::{PhraseQuery, PhraseScorer};
use crate::collector::{Count, TopDocs};
use crate::directory::{FileHandle, FileSlice, OwnedBytes};
use crate::fieldnorm::FieldNormReader;
use crate::merge_policy::NoMergePolicy;
use crate::postings::{LoadedPostings, Postings};
use crate::query::bm25::Bm25Weight;
use crate::query::scorer::BasicPruningScorer;
use crate::query::weight::for_each_pruning_scorer;
use crate::query::{BoostQuery, PruningScorer};
use crate::schema::{Schema, TEXT};
use crate::{DocId, DocSet, HasLen, Index, Score, Term, TERMINATED};

struct CountedPostings {
    postings: LoadedPostings,
    reads: Arc<AtomicUsize>,
}

impl DocSet for CountedPostings {
    fn advance(&mut self) -> DocId {
        self.postings.advance()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        self.postings.seek(target)
    }
    fn doc(&self) -> DocId {
        self.postings.doc()
    }
    fn size_hint(&self) -> u32 {
        self.postings.size_hint()
    }
}

impl Postings for CountedPostings {
    fn term_freq(&self) -> u32 {
        self.postings.term_freq()
    }
    fn append_positions_with_offset(&mut self, offset: u32, output: &mut Vec<u32>) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.postings.append_positions_with_offset(offset, output);
    }
}

#[derive(Debug)]
struct CountedNorms {
    bytes: Vec<u8>,
    reads: AtomicUsize,
}

impl HasLen for CountedNorms {
    fn len(&self) -> usize {
        self.bytes.len()
    }
}

impl FileHandle for CountedNorms {
    fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
        self.reads.fetch_add(range.len(), Ordering::Relaxed);
        Ok(OwnedBytes::new(self.bytes[range].to_vec()))
    }
}

#[test]
fn test_phrase_score_bound_avoids_positions_and_norms() {
    for num_terms in [2, 3] {
        for anchor in [false, true] {
            crate::postings::set_phrase_anchor_filter(anchor);
            let mut observations = Vec::new();
            for enabled in [false, true] {
                let positions = Arc::new(AtomicUsize::new(0));
                let norms = Arc::new(CountedNorms {
                    bytes: vec![FieldNormReader::fieldnorm_to_id(100); 128],
                    reads: AtomicUsize::new(0),
                });
                let postings = (0..num_terms)
                    .map(|offset| {
                        let docs = (0..128).collect();
                        let values = (0..128)
                            .map(|doc| {
                                (0..if doc == 0 { 10 } else { 1 })
                                    .map(|repeat| repeat * num_terms as u32 + offset as u32)
                                    .collect()
                            })
                            .collect();
                        (
                            offset,
                            CountedPostings {
                                postings: LoadedPostings::from((docs, values)),
                                reads: positions.clone(),
                            },
                        )
                    })
                    .collect();
                let scorer = PhraseScorer::new(
                    postings,
                    Some(Bm25Weight::for_one_term(
                        128,
                        10_000,
                        100.0,
                        crate::Bm25Params::default(),
                    )),
                    FieldNormReader::open(FileSlice::new(norms.clone())),
                    0,
                );
                let mut pruning: Box<dyn PruningScorer> = if enabled {
                    Box::new(PhrasePruningScorer::new(scorer, Score::MIN))
                } else {
                    Box::new(BasicPruningScorer::new(Box::new(scorer), Score::MIN))
                };
                let mut hits = Vec::new();
                for_each_pruning_scorer(pruning.as_mut(), &mut |doc, score| {
                    hits.push((doc, score));
                    score
                });
                assert_eq!(pruning.advance(), TERMINATED);
                observations.push((
                    hits,
                    positions.load(Ordering::Relaxed),
                    norms.reads.load(Ordering::Relaxed),
                ));
            }
            assert_eq!(observations[0].0, observations[1].0);
            assert_eq!(observations[0].0.len(), 1);
            assert_eq!(observations[0].1, 128 * num_terms);
            assert_eq!(observations[1].1, num_terms);
            assert_eq!(observations[0].2, 128);
            assert_eq!(observations[1].2, 1);
        }
    }
    crate::postings::set_phrase_anchor_filter(false);
}

#[test]
fn test_phrase_score_bound_uses_exact_phrase_frequency_before_norm() {
    let weight = Bm25Weight::for_one_term(128, 10_000, 100.0, crate::Bm25Params::default());
    let norm = FieldNormReader::fieldnorm_to_id(100);
    let initial_cutoff = weight.score(norm, 5);
    assert!(weight.score(0, 1) < initial_cutoff);
    assert!(initial_cutoff < weight.score(norm, 10));
    assert!(weight.score(norm, 10) < weight.score(0, 10));
    for num_terms in [2, 3] {
        for anchor in [false, true] {
            crate::postings::set_phrase_anchor_filter(anchor);
            for threshold in [Score::MIN, initial_cutoff] {
                let mut observations = Vec::new();
                for enabled in [false, true] {
                    let positions = Arc::new(AtomicUsize::new(0));
                    let norms = Arc::new(CountedNorms {
                        bytes: vec![norm; 128],
                        reads: AtomicUsize::new(0),
                    });
                    let postings = (0..num_terms)
                        .map(|offset| {
                            let values = (0..128)
                                .map(|doc| {
                                    (0..10)
                                        .map(|repeat| {
                                            let position = if doc == 1 || repeat == 0 {
                                                repeat * num_terms
                                            } else {
                                                100 * (offset + 1) + repeat * 10
                                            };
                                            (position + offset) as u32
                                        })
                                        .collect()
                                })
                                .collect();
                            (
                                offset,
                                CountedPostings {
                                    postings: LoadedPostings::from(((0..128).collect(), values)),
                                    reads: positions.clone(),
                                },
                            )
                        })
                        .collect();
                    let scorer = PhraseScorer::new(
                        postings,
                        Some(weight.clone()),
                        FieldNormReader::open(FileSlice::new(norms.clone())),
                        0,
                    );
                    let mut pruning: Box<dyn PruningScorer> = if enabled {
                        Box::new(PhrasePruningScorer::new(scorer, threshold))
                    } else {
                        Box::new(BasicPruningScorer::new(Box::new(scorer), threshold))
                    };
                    let mut hits = Vec::new();
                    for_each_pruning_scorer(pruning.as_mut(), &mut |doc, score| {
                        hits.push((doc, score));
                        score
                    });
                    observations.push((
                        hits,
                        positions.load(Ordering::Relaxed),
                        norms.reads.load(Ordering::Relaxed),
                    ));
                }
                assert_eq!(observations[0].0, observations[1].0);
                let expected_norms = if threshold == Score::MIN { 2 } else { 1 };
                assert_eq!(observations[1].0.len(), expected_norms);
                assert_eq!(observations[0].1, 128 * num_terms);
                assert_eq!(observations[0].1, observations[1].1);
                assert_eq!(observations[0].2, 128);
                assert_eq!(observations[1].2, expected_norms);
            }
        }
    }
    crate::postings::set_phrase_anchor_filter(false);
}

#[test]
fn test_phrase_score_bound_topk_ties_offsets_repetition_and_fallback() -> crate::Result<()> {
    let mut schema = Schema::builder();
    let text = schema.add_text_field("text", TEXT);
    let index = Index::create_in_ram(schema.build());
    let mut writer = index.writer_for_tests()?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for segment in 0..4 {
        for doc in 0..60 {
            let pattern = match (segment + doc) % 5 {
                0 => "alpha beta gamma ",
                1 => "alpha alpha alpha ",
                2 => "alpha x beta gamma ",
                3 => "alpha beta alpha beta ",
                _ => "beta gamma alpha ",
            };
            writer.add_document(crate::doc!(text => pattern.repeat(1 + doc % 7)))?;
        }
        writer.commit()?;
    }
    let searcher = index.reader()?.searcher();
    for (spec, slop) in [
        (vec![(0, "alpha"), (1, "beta")], 0),
        (vec![(0, "alpha"), (1, "beta"), (2, "gamma")], 0),
        (vec![(0, "alpha"), (1, "alpha"), (2, "alpha")], 0),
        (vec![(0, "alpha"), (2, "beta"), (3, "gamma")], 0),
        (vec![(0, "alpha"), (1, "beta"), (2, "gamma")], 1),
    ] {
        let phrase = PhraseQuery::new_with_offset_and_slop(
            spec.iter()
                .map(|&(offset, term)| (offset, Term::from_field_text(text, term)))
                .collect(),
            slop,
        );
        for boost in [1.0, 0.0, -1.0] {
            let query = BoostQuery::new(Box::new(phrase.clone()), boost);
            for anchor in [false, true] {
                crate::postings::set_phrase_anchor_filter(anchor);
                for (limit, offset) in [(1, 0), (10, 0), (10, 7)] {
                    let mut outcomes = Vec::new();
                    for enabled in [false, true] {
                        crate::postings::set_phrase_score_bound(enabled);
                        let top = searcher.search(
                            &query,
                            &TopDocs::with_limit(limit)
                                .and_offset(offset)
                                .order_by_score(),
                        )?;
                        let count = searcher.search(&query, &Count)?;
                        outcomes.push((top, count));
                    }
                    assert_eq!(
                        outcomes[0], outcomes[1],
                        "spec={spec:?} slop={slop} boost={boost} anchor={anchor} limit={limit} \
                         offset={offset}"
                    );
                }
            }
        }
    }
    crate::postings::set_phrase_score_bound(false);
    crate::postings::set_phrase_anchor_filter(false);
    Ok(())
}

#[test]
fn test_phrase_score_bound_positive_bm25_params() {
    for (k1, b) in [(0.0, 0.75), (1.2, 0.0), (1.2, 0.75), (2.4, 1.0)] {
        let params = crate::Bm25Params::new(k1, b);
        for boost in [0.0, 0.5, 3.0] {
            let weight = Bm25Weight::for_one_term(100, 10_000, 100.0, params).boost_by(boost);
            for min_tf in [1, 2, 7, 100] {
                let bound = weight.score(0, min_tf);
                for phrase_tf in 1..=min_tf {
                    for norm in 0..=255 {
                        assert!(weight.score(norm, phrase_tf) <= bound);
                    }
                }
            }
        }
    }
}
