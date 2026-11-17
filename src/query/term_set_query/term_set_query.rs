use std::collections::HashMap;
use std::io;
use std::sync::Arc;

use tantivy_fst::Map;

use super::term_predicate::ExactFastField;
use super::term_set_strategy::TermSetStrategyConfig;
use super::term_set_weight::{SetDfaWrapper, TermSetWeight};
use crate::query::score_combiner::DoNothingCombiner;
use crate::query::{AutomatonWeight, BooleanWeight, EnableScoring, Occur, Query, Weight};
use crate::schema::{Field, Schema};
use crate::{SegmentReader, Term};

fn group_terms_by_field<T: IntoIterator<Item = Term>>(terms: T) -> HashMap<Field, Vec<Term>> {
    let mut terms_map: HashMap<Field, Vec<Term>> = HashMap::new();
    for term in terms {
        terms_map.entry(term.field()).or_default().push(term);
    }
    terms_map
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexRequirement {
    Required,
    Optional,
}

fn build_multi_field_weight(
    terms_map: &HashMap<Field, Vec<Term>>,
    strategy_config: &TermSetStrategyConfig,
    exact_fast_field: ExactFastField,
    schema: &Schema,
    bitmap_enabled: bool,
    index_requirement: IndexRequirement,
) -> crate::Result<BooleanWeight<DoNothingCombiner>> {
    let mut sub_queries: Vec<(_, Box<dyn Weight>)> = Vec::with_capacity(terms_map.len());
    for (&field, terms) in terms_map {
        let field_entry = schema.get_field_entry(field);
        let field_type = field_entry.field_type();
        if index_requirement == IndexRequirement::Required && !field_type.is_indexed() {
            return Err(crate::TantivyError::SchemaError(format!(
                "Field {:?} is not indexed.",
                field_entry.name()
            )));
        }
        sub_queries.push((
            Occur::Should,
            Box::new(
                TermSetWeight::from_terms(
                    field,
                    terms,
                    field_type,
                    exact_fast_field,
                    strategy_config.clone(),
                )?
                .with_bitmap_postings(bitmap_enabled),
            ),
        ));
    }
    Ok(
        BooleanWeight::new(sub_queries, false, Box::new(DoNothingCombiner::default))
            .with_bitmap_postings(bitmap_enabled),
    )
}

/// A `TermSetQuery` matches all documents containing any of the provided `Term`s.
///
/// Dispatches each field via [`TermSetWeight`] across fast fields or inverted-index
/// postings depending on field schema and selectivity. Requires every referenced
/// field to be indexed (use [`FastFieldTermSetQuery`] for `FAST`-only fields, or
/// [`InvertedIndexTermSetQuery`] to force inverted-index execution).
#[derive(Debug, Clone)]
pub struct TermSetQuery {
    terms_map: HashMap<Field, Vec<Term>>,
    strategy_config: TermSetStrategyConfig,
    exact_fast_field: ExactFastField,
}

impl TermSetQuery {
    /// Create a `TermSetQuery`.
    pub fn new<T: IntoIterator<Item = Term>>(terms: T) -> Self {
        Self {
            terms_map: group_terms_by_field(terms),
            strategy_config: TermSetStrategyConfig::default(),
            exact_fast_field: ExactFastField::default(),
        }
    }

    /// Override the strategy thresholds and `strategy_sink`.
    pub fn with_strategy_config(mut self, cfg: TermSetStrategyConfig) -> Self {
        self.strategy_config = cfg;
        self
    }

    /// Specify whether `Str` fast-field dictionaries store the exact same terms as their
    /// inverted index (for example when indexing and fast-field tokenizers use aliased names).
    pub fn with_exact_fast_field(mut self, exact_fast_field: ExactFastField) -> Self {
        self.exact_fast_field = exact_fast_field;
        self
    }
}

impl Query for TermSetQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(build_multi_field_weight(
            &self.terms_map,
            &self.strategy_config,
            self.exact_fast_field,
            enable_scoring.schema(),
            enable_scoring.bitmap_postings_enabled(),
            IndexRequirement::Required,
        )?))
    }

    fn query_terms(
        &self,
        _field: Field,
        _segment_reader: &SegmentReader,
        visitor: &mut dyn FnMut(&Term, bool),
    ) {
        for terms in self.terms_map.values() {
            for term in terms {
                visitor(term, false);
            }
        }
    }
}

/// Thin `Query` wrapper around [`TermSetWeight`] that also permits fast-only
/// (non-indexed) fields.
#[derive(Debug, Clone)]
pub struct FastFieldTermSetQuery {
    terms_map: HashMap<Field, Vec<Term>>,
    strategy_config: TermSetStrategyConfig,
    exact_fast_field: ExactFastField,
}

impl FastFieldTermSetQuery {
    /// Create a new `FastFieldTermSetQuery`.
    pub fn new<T: IntoIterator<Item = Term>>(terms: T) -> Self {
        Self {
            terms_map: group_terms_by_field(terms),
            strategy_config: TermSetStrategyConfig::default(),
            exact_fast_field: ExactFastField::default(),
        }
    }

    /// Override the strategy thresholds and `strategy_sink`.
    pub fn with_strategy_config(mut self, cfg: TermSetStrategyConfig) -> Self {
        self.strategy_config = cfg;
        self
    }

    /// Specify whether `Str` fast-field dictionaries store the exact same terms as their
    /// inverted index (for example when indexing and fast-field tokenizers use aliased names).
    pub fn with_exact_fast_field(mut self, exact_fast_field: ExactFastField) -> Self {
        self.exact_fast_field = exact_fast_field;
        self
    }
}

impl Query for FastFieldTermSetQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(build_multi_field_weight(
            &self.terms_map,
            &self.strategy_config,
            self.exact_fast_field,
            enable_scoring.schema(),
            enable_scoring.bitmap_postings_enabled(),
            IndexRequirement::Optional,
        )?))
    }
}

/// `InvertedIndexTermSetQuery` is the same as [`TermSetQuery`] but only uses the inverted index.
#[derive(Debug, Clone)]
pub struct InvertedIndexTermSetQuery {
    terms_map: HashMap<Field, Vec<Term>>,
}

impl InvertedIndexTermSetQuery {
    /// Create a new `InvertedIndexTermSetQuery`.
    pub fn new<T: IntoIterator<Item = Term>>(terms: T) -> Self {
        let mut terms_map = group_terms_by_field(terms);
        for terms in terms_map.values_mut() {
            terms.sort_unstable();
            terms.dedup();
        }
        Self { terms_map }
    }
}

impl Query for InvertedIndexTermSetQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let mut sub_queries: Vec<(_, Box<dyn Weight>)> = Vec::with_capacity(self.terms_map.len());
        for (&field, sorted_terms) in &self.terms_map {
            let schema = enable_scoring.schema();
            let field_entry = schema.get_field_entry(field);
            if !field_entry.field_type().is_indexed() {
                return Err(crate::TantivyError::SchemaError(format!(
                    "Field {:?} is not indexed.",
                    field_entry.name()
                )));
            }
            let map = Map::from_iter(
                sorted_terms
                    .iter()
                    .map(|key| (key.serialized_value_bytes(), 0u64)),
            )
            .map_err(io::Error::other)?;

            sub_queries.push((
                Occur::Should,
                Box::new(
                    AutomatonWeight::new(field, SetDfaWrapper(Arc::new(map)))
                        .with_scoring_enabled(enable_scoring.is_scoring_enabled())
                        .with_bitmap_postings(enable_scoring.bitmap_postings_enabled()),
                ),
            ));
        }
        Ok(Box::new(
            BooleanWeight::new(sub_queries, false, Box::new(DoNothingCombiner::default))
                .with_bitmap_postings(enable_scoring.bitmap_postings_enabled()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::collector::TopDocs;
    use crate::query::{QueryParser, TermSetQuery};
    use crate::schema::{Schema, TEXT};
    use crate::{assert_nearly_equals, Index, IndexWriter, Term};

    #[test]
    pub fn test_term_set_query() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let field1 = schema_builder.add_text_field("field1", TEXT);
        let field2 = schema_builder.add_text_field("field2", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(field1 => "doc1", field2 => "val1"))?;
            index_writer.add_document(doc!(field1 => "doc2", field2 => "val2"))?;
            index_writer.add_document(doc!(field1 => "doc3", field2 => "val3"))?;
            index_writer.add_document(doc!(field1 => "val3", field2 => "doc3"))?;
            index_writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let top_docs = searcher.search(
            &TermSetQuery::new([Term::from_field_text(field1, "doc1")]),
            &TopDocs::with_limit(2).order_by_score(),
        )?;
        assert_eq!(top_docs.len(), 1);
        assert_nearly_equals!(1.0, top_docs[0].0);

        assert!(searcher
            .search(
                &TermSetQuery::new([Term::from_field_text(field1, "doc4")]),
                &TopDocs::with_limit(1).order_by_score()
            )?
            .is_empty());

        let top_docs = searcher.search(
            &TermSetQuery::new([
                Term::from_field_text(field1, "doc1"),
                Term::from_field_text(field1, "doc1"),
                Term::from_field_text(field2, "val2"),
            ]),
            &TopDocs::with_limit(3).order_by_score(),
        )?;
        assert_eq!(top_docs.len(), 2);
        for (score, _) in top_docs {
            assert_nearly_equals!(1.0, score);
        }

        // No field crosstalk
        assert_eq!(
            searcher
                .search(
                    &TermSetQuery::new([Term::from_field_text(field1, "doc3")]),
                    &TopDocs::with_limit(3).order_by_score()
                )?
                .len(),
            1
        );
        assert_eq!(
            searcher
                .search(
                    &TermSetQuery::new([
                        Term::from_field_text(field1, "doc3"),
                        Term::from_field_text(field2, "doc3"),
                    ]),
                    &TopDocs::with_limit(3).order_by_score()
                )?
                .len(),
            2
        );

        let qp = QueryParser::for_index(&index, vec![]);
        let top_docs = searcher.search(
            &qp.parse_query("field1: IN [doc1 doc2]")?,
            &TopDocs::with_limit(3).order_by_score(),
        )?;
        assert_eq!(top_docs.len(), 2);
        Ok(())
    }
}
