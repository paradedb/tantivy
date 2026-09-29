use super::phrase_scorer::BlockPruningPhraseScorer;
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
        let slop = 0;
        PhraseWeight {
            phrase_terms,
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
        let indexing_average = if similarity_weight_opt.is_some() {
            reader
                .inverted_index(self.phrase_terms[0].1.field())?
                .total_num_tokens() as Score
                / reader.max_doc() as Score
        } else {
            Score::NAN
        };
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
        Ok(Some(
            PhraseScorer::new(
                term_postings_list,
                similarity_weight_opt,
                fieldnorm_reader,
                self.slop,
            )
            .with_indexing_average(indexing_average),
        ))
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
            let can_prune_positions = self.slop == 0
                && self
                    .similarity_weight_opt
                    .as_ref()
                    .is_some_and(|weight| weight.supports_pruning(boost));
            if can_prune_positions {
                let indexing_average = reader
                    .inverted_index(self.phrase_terms[0].1.field())?
                    .total_num_tokens() as Score
                    / reader.max_doc() as Score;
                Ok(Box::new(BlockPruningPhraseScorer::new(
                    scorer,
                    init_threshold,
                    indexing_average,
                )))
            } else {
                Ok(Box::new(BasicPruningScorer::new(
                    Box::new(scorer),
                    init_threshold,
                )))
            }
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
    fn test_phrase_pruning_matches_unpruned() -> crate::Result<()> {
        let mut texts = vec![
            "a b".to_string(),
            "a a b d a b c".to_string(),
            "a x b".to_string(),
            "b a".to_string(),
            "a a a".to_string(),
            "a c b".to_string(),
            "c".to_string(),
            String::new(),
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
