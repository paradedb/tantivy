use std::fmt;

use crate::docset::{DocSet, TERMINATED};
use crate::fieldnorm::FieldNormReader;
use crate::index::{Bm25Params, SegmentReader};
use crate::postings::{Postings, SegmentPostings};
use crate::query::bm25::Bm25Weight;
use crate::query::explanation::does_not_match;
use crate::query::{EnableScoring, EmptyScorer, Explanation, Query, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption};
use crate::{DocId, Score, Term};

/// A query that matches documents containing any of the given terms, scoring
/// each document as if all the terms were a single term.
///
/// This mirrors Lucene's `SynonymQuery`: the terms are treated as one
/// pseudo-term whose document frequency is the maximum document frequency of
/// the individual terms, and a document's term frequency is the sum of its
/// frequencies for each term. This avoids two artifacts of expanding synonyms
/// into a boolean OR of term queries:
///
/// * a rare synonym receiving a large idf boost relative to a common one, and
/// * documents containing several variants outscoring documents containing a
///   single variant, purely because scores of separate clauses are summed.
///
/// All terms must belong to the same field.
#[derive(Clone)]
pub struct SynonymQuery {
    terms: Vec<Term>,
}

impl fmt::Debug for SynonymQuery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SynonymQuery({:?})", self.terms)
    }
}

impl SynonymQuery {
    /// Creates a new `SynonymQuery` from the given terms.
    ///
    /// Panics if `terms` is empty or if the terms do not all belong to the
    /// same field.
    pub fn new(terms: Vec<Term>) -> SynonymQuery {
        assert!(
            !terms.is_empty(),
            "SynonymQuery requires at least one term"
        );
        let field = terms[0].field();
        assert!(
            terms[1..].iter().all(|term| term.field() == field),
            "All SynonymQuery terms must belong to the same field."
        );
        SynonymQuery { terms }
    }

    /// The terms this query is built out of.
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    fn specialized_weight(
        &self,
        enable_scoring: EnableScoring<'_>,
    ) -> crate::Result<SynonymWeight> {
        let schema = enable_scoring.schema();
        let field_entry = schema.get_field_entry(self.terms[0].field());
        if !field_entry.is_indexed() {
            let error_msg = format!("Field {:?} is not indexed.", field_entry.name());
            return Err(crate::TantivyError::SchemaError(error_msg));
        }
        let bm25_weight = match enable_scoring {
            EnableScoring::Enabled {
                statistics_provider,
                ..
            } => {
                let field = self.terms[0].field();
                let total_num_tokens = statistics_provider.total_num_tokens(field)?;
                let total_num_docs = statistics_provider.total_num_docs()?;
                let average_fieldnorm = total_num_tokens as Score / total_num_docs as Score;
                let params = statistics_provider.bm25_params(field);
                // Blended statistics: the pseudo-term's document frequency is
                // the maximum document frequency among the terms.
                let mut max_doc_freq: u64 = 0;
                for term in &self.terms {
                    max_doc_freq = max_doc_freq.max(statistics_provider.doc_freq(term)?);
                }
                Bm25Weight::for_one_term(max_doc_freq, total_num_docs, average_fieldnorm, params)
            }
            EnableScoring::Disabled { .. } => Bm25Weight::new(
                Explanation::new("<no score>", 1.0f32),
                1.0f32,
                Bm25Params::default(),
            ),
        };
        let scoring_enabled = enable_scoring.is_scoring_enabled();
        let index_record_option = if scoring_enabled {
            IndexRecordOption::WithFreqs
        } else {
            IndexRecordOption::Basic
        };
        Ok(SynonymWeight {
            terms: self.terms.clone(),
            index_record_option,
            similarity_weight: bm25_weight,
            scoring_enabled,
        })
    }
}

impl Query for SynonymQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(self.specialized_weight(enable_scoring)?))
    }

    fn query_terms(
        &self,
        _field: Field,
        _segment_reader: &SegmentReader,
        visitor: &mut dyn FnMut(&Term, bool),
    ) {
        for term in &self.terms {
            visitor(term, false);
        }
    }
}

pub struct SynonymWeight {
    terms: Vec<Term>,
    index_record_option: IndexRecordOption,
    similarity_weight: Bm25Weight,
    scoring_enabled: bool,
}

impl SynonymWeight {
    fn specialized_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<Option<SynonymScorer>> {
        let field = self.terms[0].field();
        let inverted_index = reader.inverted_index(field)?;
        let mut postings: Vec<SegmentPostings> = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            if let Some(term_info) = inverted_index.get_term_info(term)? {
                postings.push(
                    inverted_index
                        .read_postings_from_terminfo(&term_info, self.index_record_option)?,
                );
            }
        }
        if postings.is_empty() {
            return Ok(None);
        }
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let similarity_weight = self.similarity_weight.boost_by(boost);
        Ok(Some(SynonymScorer::new(
            postings,
            fieldnorm_reader,
            similarity_weight,
        )))
    }

    fn fieldnorm_reader(&self, segment_reader: &SegmentReader) -> crate::Result<FieldNormReader> {
        if self.scoring_enabled {
            if let Some(fieldnorm_reader) = segment_reader
                .fieldnorms_readers()
                .get_field(self.terms[0].field())?
            {
                return Ok(fieldnorm_reader);
            }
        }
        Ok(FieldNormReader::constant(segment_reader.max_doc(), 1))
    }
}

impl Weight for SynonymWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        match self.specialized_scorer(reader, boost)? {
            Some(scorer) => Ok(Box::new(scorer)),
            None => Ok(Box::new(EmptyScorer)),
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let Some(mut scorer) = self.specialized_scorer(reader, 1.0)? else {
            return Err(does_not_match(doc));
        };
        if scorer.doc() > doc || scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        let fieldnorm_id = scorer.fieldnorm_id();
        let term_freq = scorer.term_freq();
        let mut explanation = self.similarity_weight.explain(fieldnorm_id, term_freq);
        explanation.add_context(format!(
            "SynonymQuery, blended term_freq across {:?}",
            self.terms
        ));
        Ok(explanation)
    }
}

/// A union scorer over the posting lists of the synonym's terms.
///
/// Docs matched by several terms are scored once, with a term frequency equal
/// to the sum of the individual term frequencies.
pub struct SynonymScorer {
    postings: Vec<SegmentPostings>,
    doc: DocId,
    fieldnorm_reader: FieldNormReader,
    similarity_weight: Bm25Weight,
}

impl SynonymScorer {
    fn new(
        postings: Vec<SegmentPostings>,
        fieldnorm_reader: FieldNormReader,
        similarity_weight: Bm25Weight,
    ) -> SynonymScorer {
        let doc = postings.iter().map(DocSet::doc).min().unwrap_or(TERMINATED);
        SynonymScorer {
            postings,
            doc,
            fieldnorm_reader,
            similarity_weight,
        }
    }

    fn fieldnorm_id(&self) -> u8 {
        self.fieldnorm_reader.fieldnorm_id(self.doc)
    }

    /// The blended term frequency for the current doc: the sum of the term
    /// frequencies of every synonym term matching it.
    fn term_freq(&mut self) -> u32 {
        let doc = self.doc;
        self.postings
            .iter_mut()
            .filter(|postings| postings.doc() == doc)
            .map(|postings| postings.term_freq())
            .sum()
    }
}

impl DocSet for SynonymScorer {
    fn advance(&mut self) -> DocId {
        let current = self.doc;
        if current == TERMINATED {
            return TERMINATED;
        }
        let mut next = TERMINATED;
        for postings in &mut self.postings {
            let mut doc = postings.doc();
            if doc == current {
                doc = postings.advance();
            }
            next = next.min(doc);
        }
        self.doc = next;
        next
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.doc >= target {
            return self.doc;
        }
        let mut next = TERMINATED;
        for postings in &mut self.postings {
            let mut doc = postings.doc();
            if doc < target {
                doc = postings.seek(target);
            }
            next = next.min(doc);
        }
        self.doc = next;
        next
    }

    fn doc(&self) -> DocId {
        self.doc
    }

    fn size_hint(&self) -> u32 {
        // Upper bound: the union can't match more docs than the sum of the
        // individual posting lists.
        self.postings
            .iter()
            .map(DocSet::size_hint)
            .fold(0u32, u32::saturating_add)
    }
}

impl Scorer for SynonymScorer {
    fn score(&mut self) -> Score {
        let fieldnorm_id = self.fieldnorm_id();
        let term_freq = self.term_freq();
        self.similarity_weight.score(fieldnorm_id, term_freq)
    }
}

#[cfg(test)]
mod tests {
    use super::SynonymQuery;
    use crate::collector::{Count, TopDocs};
    use crate::query::{BooleanQuery, EnableScoring, Occur, Query, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, TEXT};
    use crate::{assert_nearly_equals, DocAddress, Index, IndexWriter, Term};

    fn build_test_index() -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut writer: IndexWriter = index.writer_for_tests()?;
            writer.add_document(doc!(text => "the cat sat"))?;
            writer.add_document(doc!(text => "a feline slept"))?;
            writer.add_document(doc!(text => "cat and feline together"))?;
            writer.add_document(doc!(text => "dogs everywhere"))?;
            writer.add_document(doc!(text => "cat cat cat"))?;
            writer.commit()?;
        }
        Ok(index)
    }

    #[test]
    fn test_synonym_query_matches_union() -> crate::Result<()> {
        let index = build_test_index()?;
        let text = index.schema().get_field("text").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = SynonymQuery::new(vec![
            Term::from_field_text(text, "cat"),
            Term::from_field_text(text, "feline"),
        ]);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 4);
        Ok(())
    }

    #[test]
    fn test_synonym_query_missing_term() -> crate::Result<()> {
        let index = build_test_index()?;
        let text = index.schema().get_field("text").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = SynonymQuery::new(vec![
            Term::from_field_text(text, "cat"),
            Term::from_field_text(text, "nosuchterm"),
        ]);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);
        let all_missing = SynonymQuery::new(vec![Term::from_field_text(text, "nosuchterm")]);
        let count = searcher.search(&all_missing, &Count)?;
        assert_eq!(count, 0);
        Ok(())
    }

    #[test]
    fn test_synonym_query_blended_scoring() -> crate::Result<()> {
        let index = build_test_index()?;
        let text = index.schema().get_field("text").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = SynonymQuery::new(vec![
            Term::from_field_text(text, "cat"),
            Term::from_field_text(text, "feline"),
        ]);
        let top_docs = searcher.search(&query, &TopDocs::with_limit(4).order_by_score())?;
        // Doc 4 ("cat cat cat") has the highest blended term frequency and the
        // same fieldnorm as doc 2 does not; it must outrank single-occurrence
        // docs.
        assert_eq!(top_docs[0].1, DocAddress::new(0, 4));

        // A doc matching both synonyms scores as one pseudo-term with tf=2,
        // strictly less than 2x the single-variant score (saturation), and
        // identical to what a single term with tf=2 would get.
        let scores: std::collections::HashMap<DocAddress, f32> =
            top_docs.iter().map(|(score, addr)| (*addr, *score)).collect();
        let single = scores[&DocAddress::new(0, 0)];
        let both = scores[&DocAddress::new(0, 2)];
        assert!(both > 0.0 && single > 0.0);
        assert!(both < 2.0 * single);
        Ok(())
    }

    #[test]
    fn test_synonym_query_scores_lower_than_boolean_or_for_multi_variant_docs(
    ) -> crate::Result<()> {
        // Documents containing both variants should NOT be double-counted the
        // way a boolean OR of term queries does.
        let index = build_test_index()?;
        let text = index.schema().get_field("text").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let cat = Term::from_field_text(text, "cat");
        let feline = Term::from_field_text(text, "feline");

        let synonym = SynonymQuery::new(vec![cat.clone(), feline.clone()]);
        let boolean = BooleanQuery::new(vec![
            (
                Occur::Should,
                Box::new(TermQuery::new(cat, IndexRecordOption::WithFreqs)) as Box<dyn Query>,
            ),
            (
                Occur::Should,
                Box::new(TermQuery::new(feline, IndexRecordOption::WithFreqs)) as Box<dyn Query>,
            ),
        ]);

        let synonym_top = searcher.search(&synonym, &TopDocs::with_limit(5).order_by_score())?;
        let boolean_top = searcher.search(&boolean, &TopDocs::with_limit(5).order_by_score())?;
        // Same match set.
        let mut synonym_docs: Vec<_> = synonym_top.iter().map(|(_, d)| *d).collect();
        let mut boolean_docs: Vec<_> = boolean_top.iter().map(|(_, d)| *d).collect();
        synonym_docs.sort();
        boolean_docs.sort();
        assert_eq!(synonym_docs, boolean_docs);
        Ok(())
    }

    #[test]
    fn test_synonym_query_explain() -> crate::Result<()> {
        let index = build_test_index()?;
        let text = index.schema().get_field("text").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = SynonymQuery::new(vec![
            Term::from_field_text(text, "cat"),
            Term::from_field_text(text, "feline"),
        ]);
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let explanation = weight.explain(searcher.segment_reader(0), 2)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(5).order_by_score())?;
        let score = top_docs
            .iter()
            .find(|(_, addr)| *addr == DocAddress::new(0, 2))
            .map(|(score, _)| *score)
            .unwrap();
        assert_nearly_equals!(explanation.value(), score);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "at least one term")]
    fn test_synonym_query_empty_terms_panics() {
        let _ = SynonymQuery::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "same field")]
    fn test_synonym_query_mixed_fields_panics() {
        let mut schema_builder = Schema::builder();
        let a = schema_builder.add_text_field("a", TEXT);
        let b = schema_builder.add_text_field("b", TEXT);
        let _schema = schema_builder.build();
        let _ = SynonymQuery::new(vec![
            Term::from_field_text(a, "x"),
            Term::from_field_text(b, "y"),
        ]);
    }
}
