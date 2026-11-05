//! Synonym rules compiled per segment, and the query-time expansion that reads them.
//!
//! A text field set with
//! [`TextOptions::set_synonym_rules`](crate::schema::TextOptions::set_synonym_rules) holds one Solr
//! synonym rule per value, such as `couch, sofa, divan` or `sneakers => running shoes`. Each
//! segment then gets a `.syn` component, a dictionary from the analyzed tokens of a rule side to
//! its alternatives. The field's tokenizer analyzes both sides, so a stemmer acts on the rules the
//! way it acts on a query. A merge copies the compiled rules of the alive documents into the
//! merged segment.
//!
//! [`SynonymExpander`] reads the dictionaries of every segment and rewrites a list of query tokens
//! into [`Expansion`]s, the longest key first, the way Lucene's `SynonymGraphFilter` does.
//!
//! ```
//! use tantivy::schema::{Schema, TEXT};
//! use tantivy::synonyms::SynonymExpander;
//! use tantivy::{doc, Index, IndexWriter};
//!
//! # fn main() -> tantivy::Result<()> {
//! let mut schema_builder = Schema::builder();
//! let rule = schema_builder.add_text_field("rule", TEXT.set_synonym_rules());
//! let index = Index::create_in_ram(schema_builder.build());
//! let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
//! writer.add_document(doc!(rule => "couch, sofa, divan"))?;
//! writer.commit()?;
//!
//! let searcher = index.reader()?.searcher();
//! let expander = SynonymExpander::new(searcher.segment_readers())?;
//! let expansions = expander.expand(&["red".to_string(), "sofa".to_string()])?;
//! assert_eq!(expansions[0].alternatives, None);
//! assert_eq!(
//!     expansions[1].alternatives,
//!     Some(vec![vec!["couch".to_string()], vec!["sofa".to_string()], vec!["divan".to_string()]])
//! );
//! # Ok(())
//! # }
//! ```

mod dictionary;
mod expand;
pub(crate) mod plugin;
mod rule;
#[cfg(test)]
mod tests;

pub use self::dictionary::{RuleAlternatives, SynonymDictionary, SYNONYM_EXT};
pub use self::expand::{Expansion, SynonymExpander};
pub use self::rule::{RuleError, SynonymRule, MAX_RULE_TERMS};
