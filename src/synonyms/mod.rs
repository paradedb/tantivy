//! Solr synonym rules indexed as terms, and the query-time expansion that reads them.
//!
//! A field that holds one rule per value, such as `couch, sofa, divan` or
//! `sneakers => running shoes`, is indexed with a [`SynonymRuleTokenizer`]. It emits each key of a
//! rule as one term, so the postings of the field map a key to the documents of its rules.
//!
//! [`expand`](fn@expand) looks the keys up in every segment and rewrites a list of query tokens
//! into [`Expansion`]s, the longest key first, the way Lucene's `SynonymGraphFilter` does. It reads
//! the rules through a [`SynonymSource`], which gives the rules of a matched document from where
//! the caller keeps their text, so it can also cache them. [`SynonymExpander`] expands with no
//! cache and reads the text from the field's column, so it needs the field fast with the `raw`
//! tokenizer. [`expansion_query`] turns each [`Expansion`] into the query that matches it.
//!
//! ```
//! use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
//! use tantivy::synonyms::{SynonymExpander, SynonymRuleTokenizer};
//! use tantivy::tokenizer::{LowerCaser, SimpleTokenizer, TextAnalyzer};
//! use tantivy::{doc, Index, IndexWriter};
//!
//! # fn main() -> tantivy::Result<()> {
//! let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
//!     .filter(LowerCaser)
//!     .build();
//! let indexing = TextFieldIndexing::default()
//!     .set_tokenizer("synonym_rule")
//!     .set_index_option(IndexRecordOption::Basic);
//! let mut schema_builder = Schema::builder();
//! let rule = schema_builder.add_text_field(
//!     "rule",
//!     TextOptions::default()
//!         .set_indexing_options(indexing)
//!         .set_fast("raw"),
//! );
//! let index = Index::create_in_ram(schema_builder.build());
//! index
//!     .tokenizers()
//!     .register("synonym_rule", SynonymRuleTokenizer::new(analyzer.clone()));
//! let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
//! writer.add_document(doc!(rule => "couch, sofa, divan"))?;
//! writer.commit()?;
//!
//! let searcher = index.reader()?.searcher();
//! let expander = SynonymExpander::new(searcher.segment_readers(), rule, analyzer)?;
//! let expansions = expander.expand(&["red".to_string(), "sofa".to_string()])?;
//! assert_eq!(expansions[0].alternatives, None);
//! assert_eq!(
//!     expansions[1].alternatives,
//!     Some(vec![vec!["couch".to_string()], vec!["divan".to_string()], vec!["sofa".to_string()]])
//! );
//! # Ok(())
//! # }
//! ```

mod expand;
mod query;
mod rule;
#[cfg(test)]
mod tests;
mod tokenizer;

pub use self::expand::{
    expand, Expansion, KeyDocs, SegmentSynonyms, SynonymExpander, SynonymSource,
    DEFAULT_MAX_RULES_PER_KEY,
};
pub use self::query::expansion_query;
pub use self::rule::{CompiledRule, RuleError, SynonymRule, MAX_RULE_TERMS};
pub use self::tokenizer::SynonymRuleTokenizer;
