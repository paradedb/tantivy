use rustc_hash::FxHashMap;

use super::PhraseScorer;
use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentReader;
use crate::postings::SegmentPostings;
use crate::query::bm25::Bm25Weight;
use crate::query::explanation::does_not_match;
use crate::query::scorer::{BasicPruningScorer, PruningScorer};
use crate::query::{EmptyScorer, Explanation, Scorer, Weight};
use crate::schema::{IndexRecordOption, Term};
use crate::{DocId, DocSet, Score};

pub struct PhraseWeight {
    phrase_terms: Vec<(usize, Term)>,
    grouped_terms: Vec<(usize, Vec<u32>)>,
    similarity_weight_opt: Option<Bm25Weight>,
    slop: u32,
}

impl PhraseWeight {
    /// Creates a new phrase weight.
    /// If `similarity_weight_opt` is None, then scoring is disabled
    pub fn new(
        phrase_terms: Vec<(usize, Term)>,
        similarity_weight_opt: Option<Bm25Weight>,
    ) -> PhraseWeight {
        let mut grouped_terms = Vec::new();
        if phrase_terms.len() > 2 {
            let mut term_groups = FxHashMap::default();
            let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
            for (term_index, (offset, term)) in phrase_terms.iter().enumerate() {
                let group = *term_groups.entry(term).or_insert_with(|| {
                    groups.push((term_index, Vec::new()));
                    groups.len() - 1
                });
                groups[group].1.push(*offset);
            }
            if groups.len() >= 2 && groups.len() < phrase_terms.len() {
                let max_offset = phrase_terms
                    .iter()
                    .map(|(offset, _)| *offset)
                    .max()
                    .unwrap();
                grouped_terms = groups
                    .into_iter()
                    .map(|(term_index, mut offsets)| {
                        offsets.sort_unstable();
                        offsets.dedup();
                        (
                            term_index,
                            offsets
                                .into_iter()
                                .map(|offset| (max_offset - offset) as u32)
                                .collect(),
                        )
                    })
                    .collect();
            }
        }
        let slop = 0;
        PhraseWeight {
            phrase_terms,
            grouped_terms,
            similarity_weight_opt,
            slop,
        }
    }

    fn fieldnorm_reader(&self, reader: &SegmentReader) -> crate::Result<FieldNormReader> {
        let field = self.phrase_terms[0].1.field();
        if self.similarity_weight_opt.is_some() {
            return reader.scoring_fieldnorm_reader(field);
        }
        Ok(FieldNormReader::constant(reader.max_doc(), 1))
    }

    pub(crate) fn phrase_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<Option<PhraseScorer<SegmentPostings>>> {
        let similarity_weight_opt = self
            .similarity_weight_opt
            .as_ref()
            .map(|similarity_weight| similarity_weight.boost_by(boost));
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        if self.slop == 0 && !self.grouped_terms.is_empty() {
            let mut postings = Vec::with_capacity(self.grouped_terms.len());
            for (term_index, offsets) in &self.grouped_terms {
                let term = &self.phrase_terms[*term_index].1;
                let Some(term_postings) = reader
                    .inverted_index(term.field())?
                    .read_postings(term, IndexRecordOption::WithFreqsAndPositions)?
                else {
                    return Ok(None);
                };
                postings.push((offsets.as_slice(), term_postings));
            }
            return Ok(Some(PhraseScorer::new_grouped(
                postings,
                similarity_weight_opt,
                fieldnorm_reader,
            )));
        }
        let mut term_postings_list = Vec::new();
        for &(offset, ref term) in &self.phrase_terms {
            if let Some(postings) = reader
                .inverted_index(term.field())?
                .read_postings(term, IndexRecordOption::WithFreqsAndPositions)?
            {
                term_postings_list.push((offset, postings));
            } else {
                return Ok(None);
            }
        }
        Ok(Some(PhraseScorer::new(
            term_postings_list,
            similarity_weight_opt,
            fieldnorm_reader,
            self.slop,
        )))
    }

    pub fn slop(&mut self, slop: u32) {
        self.slop = slop;
    }
}

impl Weight for PhraseWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        if let Some(scorer) = self.phrase_scorer(reader, boost)? {
            Ok(Box::new(scorer))
        } else {
            Ok(Box::new(EmptyScorer))
        }
    }

    fn pruning_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
        init_threshold: Score,
    ) -> crate::Result<Box<dyn PruningScorer>> {
        if let Some(scorer) = self.phrase_scorer(reader, boost)? {
            Ok(Box::new(BasicPruningScorer::new(
                Box::new(scorer),
                init_threshold,
            )))
        } else {
            Ok(Box::new(EmptyScorer))
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let scorer_opt = self.phrase_scorer(reader, 1.0)?;
        if scorer_opt.is_none() {
            return Err(does_not_match(doc));
        }
        let mut scorer = scorer_opt.unwrap();
        if scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        let fieldnorm_id = scorer.fieldnorm_id();
        let phrase_count = scorer.phrase_count();
        let mut explanation = Explanation::new("Phrase Scorer", scorer.score());
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            explanation.add_detail(similarity_weight.explain(fieldnorm_id, phrase_count));
        }
        Ok(explanation)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::create_index;
    use super::*;
    use crate::docset::TERMINATED;
    use crate::query::{EnableScoring, PhraseQuery};
    use crate::{DocSet, Term};

    #[test]
    pub fn test_phrase_count() -> crate::Result<()> {
        let index = create_index(&["a c", "a a b d a b c", " a b"])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "a"),
            Term::from_field_text(text_field, "b"),
        ]);
        let enable_scoring = EnableScoring::enabled_from_searcher(&searcher);
        let phrase_weight = phrase_query.phrase_weight(enable_scoring).unwrap();
        let mut phrase_scorer = phrase_weight
            .phrase_scorer(searcher.segment_reader(0u32), 1.0)?
            .unwrap();
        assert_eq!(phrase_scorer.doc(), 1);
        assert_eq!(phrase_scorer.phrase_count(), 2);
        assert_eq!(phrase_scorer.advance(), 2);
        assert_eq!(phrase_scorer.doc(), 2);
        assert_eq!(phrase_scorer.phrase_count(), 1);
        assert_eq!(phrase_scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_repeated_groups_in_each_intersection_position() -> crate::Result<()> {
        for order in [
            ["a", "b", "c"],
            ["a", "c", "b"],
            ["b", "a", "c"],
            ["b", "c", "a"],
            ["c", "a", "b"],
            ["c", "b", "a"],
        ] {
            let mut texts = vec![
                "a b a c b ".repeat(512),
                "a b a c b".to_string(),
                "a b c a b".to_string(),
                "a b a c b a b a c b".to_string(),
            ];
            for (rank, term) in order.iter().enumerate() {
                texts.extend(std::iter::repeat_n(term.to_string(), rank + 1));
            }
            let index = create_index(&texts)?;
            let searcher = index.reader()?.searcher();
            assert_eq!(searcher.segment_readers().len(), 1);
            let reader = searcher.segment_reader(0);
            let field = index.schema().get_field("text")?;
            let inverted = reader.inverted_index(field)?;
            let costs: Vec<_> = order
                .iter()
                .map(|term| {
                    inverted
                        .read_postings(
                            &Term::from_field_text(field, term),
                            IndexRecordOption::WithFreqsAndPositions,
                        )
                        .unwrap()
                        .unwrap()
                        .size_hint()
                })
                .collect();
            assert!(costs.windows(2).all(|pair| pair[0] < pair[1]));
            let query = PhraseQuery::new(
                ["a", "b", "a", "c", "b"]
                    .iter()
                    .map(|term| Term::from_field_text(field, term))
                    .collect(),
            );
            let weight = query.phrase_weight(EnableScoring::enabled_from_searcher(&searcher))?;
            let mut grouped = weight.phrase_scorer(reader, 1.0)?.unwrap();
            let mut postings = Vec::new();
            for (offset, term) in &weight.phrase_terms {
                postings.push((
                    *offset,
                    inverted
                        .read_postings(term, IndexRecordOption::WithFreqsAndPositions)?
                        .unwrap(),
                ));
            }
            let mut legacy = PhraseScorer::new(
                postings,
                weight.similarity_weight_opt.clone(),
                weight.fieldnorm_reader(reader)?,
                0,
            );
            let mut counts = Vec::new();
            while legacy.doc() != TERMINATED {
                assert_eq!(grouped.doc(), legacy.doc(), "{order:?}");
                assert_eq!(grouped.phrase_count(), legacy.phrase_count(), "{order:?}");
                assert_eq!(
                    grouped.score().to_bits(),
                    legacy.score().to_bits(),
                    "{order:?}"
                );
                counts.push(grouped.phrase_count());
                grouped.advance();
                legacy.advance();
            }
            assert_eq!(grouped.doc(), TERMINATED);
            assert_eq!(counts, [512, 1, 2], "{order:?}");
        }
        Ok(())
    }

    #[test]
    fn test_grouped_phrases_match_ungrouped() -> crate::Result<()> {
        let mut texts = vec![
            "a b".to_string(),
            "a a b d a b c".to_string(),
            "a x b".to_string(),
            "b a".to_string(),
            "a a a".to_string(),
            "a c b".to_string(),
            "c".to_string(),
            String::new(),
            "a b a a b a b a a a a b a a b a b a a a".to_string(),
            "a b a b a b a b a b a b a b".to_string(),
            "a b a c b a b a c b".to_string(),
            "a b a c a b a c b".to_string(),
        ];
        let mut seed = 42u32;
        for len in 1..320 {
            let mut text = String::new();
            for _ in 0..len {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                text.push_str(["a ", "b ", "c ", "x "][(seed >> 24) as usize % 4]);
            }
            texts.push(text);
        }
        for pnorms in [false, true] {
            let mut schema = crate::schema::Schema::builder();
            let options = crate::schema::TEXT.set_indexing_options(
                crate::schema::TEXT
                    .get_indexing_options()
                    .unwrap()
                    .clone()
                    .set_pnorms(pnorms),
            );
            let field = schema.add_text_field("text", options);
            let index = crate::Index::create_in_ram(schema.build());
            let mut writer = index.writer_for_tests()?;
            writer.set_merge_policy(Box::new(crate::merge_policy::NoMergePolicy));
            for (ordinal, text) in texts.iter().enumerate() {
                writer.add_document(doc!(field => text.as_str()))?;
                if ordinal == 7 {
                    writer.commit()?;
                }
            }
            writer.commit()?;
            drop(writer);
            let field = index.schema().get_field("text")?;
            let searcher = index.reader()?.searcher();
            for offsets in [
                vec![(0, "a"), (1, "b")],
                vec![(0, "a"), (1, "a")],
                vec![(0, "a"), (1, "b"), (2, "c")],
                vec![(0, "a"), (2, "b")],
                vec![(0, "missing"), (1, "b")],
                vec![(0, "a"), (1, "b"), (2, "a")],
                vec![(0, "a"), (1, "b"), (2, "b"), (3, "a")],
                vec![(0, "a"), (1, "b"), (2, "a"), (3, "b"), (4, "a")],
                vec![(0, "a"), (1, "b"), (2, "a"), (3, "c"), (4, "b")],
                vec![(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "c")],
                vec![(2, "a"), (5, "b"), (8, "a")],
                vec![(0, "a"), (0, "a"), (1, "b")],
                vec![(0, "a"), (1, "a"), (2, "a")],
                vec![
                    (0, "a"),
                    (1, "b"),
                    (2, "a"),
                    (3, "a"),
                    (4, "b"),
                    (5, "a"),
                    (6, "b"),
                    (7, "a"),
                    (8, "a"),
                    (9, "a"),
                ],
            ] {
                for slop in [0, 1, 2] {
                    let mut query = PhraseQuery::new_with_offset(
                        offsets
                            .iter()
                            .map(|(offset, text)| (*offset, Term::from_field_text(field, text)))
                            .collect(),
                    );
                    query.set_slop(slop);
                    for scoring in [true, false] {
                        let enable_scoring = if scoring {
                            EnableScoring::enabled_from_searcher(&searcher)
                        } else {
                            EnableScoring::disabled_from_schema(searcher.schema())
                        };
                        let weight = query.phrase_weight(enable_scoring)?;
                        for reader in searcher.segment_readers() {
                            for boost in [0.0, 1.0, 2.5, -1.0] {
                                let mut baseline = weight.scorer(reader, boost)?;
                                let mut expected = Vec::new();
                                while baseline.doc() != TERMINATED {
                                    expected.push((baseline.doc(), baseline.score()));
                                    baseline.advance();
                                }
                                let mut term_postings = Vec::new();
                                for (offset, term) in &weight.phrase_terms {
                                    let Some(postings) =
                                        reader.inverted_index(term.field())?.read_postings(
                                            term,
                                            IndexRecordOption::WithFreqsAndPositions,
                                        )?
                                    else {
                                        term_postings.clear();
                                        break;
                                    };
                                    term_postings.push((*offset, postings));
                                }
                                let mut legacy_matches = Vec::new();
                                if !term_postings.is_empty() {
                                    let mut legacy = PhraseScorer::new(
                                        term_postings,
                                        weight
                                            .similarity_weight_opt
                                            .as_ref()
                                            .map(|weight| weight.boost_by(boost)),
                                        weight.fieldnorm_reader(reader)?,
                                        slop,
                                    );
                                    while legacy.doc() != TERMINATED {
                                        legacy_matches.push((legacy.doc(), legacy.score()));
                                        legacy.advance();
                                    }
                                }
                                assert_eq!(
                                    expected, legacy_matches,
                                    "{offsets:?}, slop={slop}, scoring={scoring}, boost={boost}"
                                );
                                let mut thresholds = vec![Score::MIN, -1.0, 0.0, Score::MAX];
                                for &(_, score) in expected.iter().step_by(17) {
                                    thresholds.extend([score.next_down(), score, score.next_up()]);
                                }
                                for threshold in thresholds {
                                    let mut scorer =
                                        weight.pruning_scorer(reader, boost, threshold)?;
                                    let mut actual = Vec::new();
                                    while scorer.doc() != TERMINATED {
                                        actual.push((scorer.doc(), scorer.score()));
                                        scorer.advance();
                                    }
                                    let expected: Vec<_> = expected
                                        .iter()
                                        .copied()
                                        .filter(|(_, score)| *score > threshold)
                                        .collect();
                                    assert_eq!(
                                        actual, expected,
                                        "{offsets:?}, slop={slop}, scoring={scoring}, \
                                         boost={boost}, threshold={threshold}"
                                    );
                                }
                                let mut baseline = BasicPruningScorer::new(
                                    weight.scorer(reader, boost)?,
                                    Score::MIN,
                                );
                                let mut optimized =
                                    weight.pruning_scorer(reader, boost, Score::MIN)?;
                                while baseline.doc() != TERMINATED {
                                    assert_eq!(optimized.doc(), baseline.doc());
                                    assert_eq!(optimized.score(), baseline.score());
                                    let threshold = baseline.score();
                                    baseline.set_threshold(threshold);
                                    optimized.set_threshold(threshold);
                                    baseline.advance();
                                    optimized.advance();
                                }
                                assert_eq!(optimized.doc(), TERMINATED);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
