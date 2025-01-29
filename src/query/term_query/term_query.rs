use std::fmt;

use super::term_weight::TermWeight;
use crate::query::bm25::Bm25Weight;
use crate::query::{EnableScoring, Explanation, Query, Weight};
use crate::schema::IndexRecordOption;
use crate::Term;

/// A Term query matches all of the documents
/// containing a specific term.
///
/// The score associated is defined as
/// `idf` *  sqrt(`term_freq` / `field norm`)
/// in which :
/// * `idf`        - inverse document frequency.
/// * `term_freq`  - number of occurrences of the term in the field
/// * `field norm` - number of tokens in the field.
///
/// ```rust
/// use tantivy::collector::{Count, TopDocs};
/// use tantivy::query::TermQuery;
/// use tantivy::schema::{Schema, TEXT, IndexRecordOption};
/// use tantivy::{doc, Index, IndexWriter, Term};
/// # fn test() -> tantivy::Result<()> {
/// let mut schema_builder = Schema::builder();
/// let title = schema_builder.add_text_field("title", TEXT);
/// let schema = schema_builder.build();
/// let index = Index::create_in_ram(schema);
/// {
///     let mut index_writer: IndexWriter = index.writer(15_000_000)?;
///     index_writer.add_document(doc!(
///         title => "The Name of the Wind",
///     ))?;
///     index_writer.add_document(doc!(
///         title => "The Diary of Muadib",
///     ))?;
///     index_writer.add_document(doc!(
///         title => "A Dairy Cow",
///     ))?;
///     index_writer.add_document(doc!(
///         title => "The Diary of a Young Girl",
///     ))?;
///     index_writer.commit()?;
/// }
/// let reader = index.reader()?;
/// let searcher = reader.searcher();
/// let query = TermQuery::new(
///     Term::from_field_text(title, "diary"),
///     IndexRecordOption::Basic,
/// );
/// let (top_docs, count) = searcher.search(&query, &(TopDocs::with_limit(2), Count))?;
/// assert_eq!(count, 2);
/// Ok(())
/// # }
/// # assert!(test().is_ok());
/// ```
#[derive(Clone)]
pub struct TermQuery {
    term: Term,
    index_record_option: IndexRecordOption,
}

impl fmt::Debug for TermQuery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TermQuery({:?})", self.term)
    }
}

impl TermQuery {
    /// Creates a new term query.
    pub fn new(term: Term, segment_postings_options: IndexRecordOption) -> TermQuery {
        println!(
            "TermQuery::new called with term: {:?}, index_record_option: {:?}",
            term, segment_postings_options
        );
        TermQuery {
            term,
            index_record_option: segment_postings_options,
        }
    }

    /// The `Term` this query is built out of.
    pub fn term(&self) -> &Term {
        println!("TermQuery::term called. Returning term: {:?}", self.term);
        &self.term
    }

    /// Returns a weight object.
    ///
    /// While `.weight(...)` returns a boxed trait object,
    /// this method return a specific implementation.
    /// This is useful for optimization purpose.
    pub fn specialized_weight(
        &self,
        enable_scoring: EnableScoring<'_>,
    ) -> crate::Result<TermWeight> {
        let schema = enable_scoring.schema();
        println!("Obtained schema from EnableScoring.");
        let field_entry = schema.get_field_entry(self.term.field());
        println!(
            "Retrieved field entry for term's field: {:?}",
            field_entry.name()
        );
        if !field_entry.is_indexed() {
            let error_msg = format!("Field {:?} is not indexed.", field_entry.name());
            println!("Error: {}", error_msg);
            return Err(crate::TantivyError::SchemaError(error_msg));
        }
        let bm25_weight = match enable_scoring {
            EnableScoring::Enabled {
                statistics_provider,
                ..
            } => {
                println!("Scoring is enabled. Calculating BM25 weight.");
                Bm25Weight::for_terms(statistics_provider, &[self.term.clone()])?
            }
            EnableScoring::Disabled { .. } => {
                println!("Scoring is disabled. Using default BM25 weight.");
                Bm25Weight::new(Explanation::new("<no score>", 1.0f32), 1.0f32)
            }
        };
        let scoring_enabled = enable_scoring.is_scoring_enabled();
        println!(
            "Scoring enabled: {}, index_record_option: {:?}",
            scoring_enabled, self.index_record_option
        );
        let index_record_option = if scoring_enabled {
            self.index_record_option
        } else {
            IndexRecordOption::Basic
        };
        println!(
            "Final index_record_option set to: {:?}",
            index_record_option
        );

        let term_weight = TermWeight::new(
            self.term.clone(),
            index_record_option,
            bm25_weight,
            scoring_enabled,
        );
        Ok(term_weight)
    }
}

impl Query for TermQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        println!("TermQuery::weight called.");
        let specialized_weight = self.specialized_weight(enable_scoring)?;
        Ok(Box::new(specialized_weight))
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        println!(
            "TermQuery::query_terms called. Visiting term: {:?}",
            self.term
        );
        visitor(&self.term, false);
    }
}
