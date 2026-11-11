//! Solr synonym rules indexed as terms, and the query-time expansion that reads them.
//!
//! A field that holds one rule per value, such as `couch, sofa, divan` or
//! `sneakers => running shoes`, is indexed with a [`SynonymRuleTokenizer`]. It emits each key of a
//! rule as one term, so the postings of the field map a key to the documents of its rules.
//!
//! [`expand`](fn@expand) looks the keys up in every segment and rewrites a list of query tokens
//! into [`Expansion`]s, the longest key first, the way Lucene's `SynonymGraphFilter` does. The
//! index keeps only the keys. The caller keeps the text of the rules, and gives it through a
//! [`SynonymSource`], which can also cache what it reads. [`expansion_query`] turns each
//! [`Expansion`] into the query that matches it.
//!
//! ```
//! use std::sync::Arc;
//!
//! use tantivy::schema::Schema;
//! use tantivy::synonyms::{
//!     expand, rule_field_options, CompiledRule, KeyDocs, SegmentSynonyms, SynonymRuleTokenizer,
//!     SynonymSource, DEFAULT_MAX_RULES_PER_KEY,
//! };
//! use tantivy::tokenizer::{LowerCaser, SimpleTokenizer, TextAnalyzer};
//! use tantivy::{doc, DocId, Index, IndexWriter};
//!
//! /// One segment, and the text of each rule at the position of its document.
//! struct Rules {
//!     segment: SegmentSynonyms,
//!     texts: Vec<&'static str>,
//!     analyzer: TextAnalyzer,
//! }
//!
//! impl SynonymSource for Rules {
//!     fn token_keys(&mut self, token: &str) -> tantivy::Result<Vec<Arc<[KeyDocs]>>> {
//!         Ok(vec![self.segment.token_keys(token, DEFAULT_MAX_RULES_PER_KEY)?.into()])
//!     }
//!
//!     fn alive_rules(
//!         &mut self,
//!         _segment: usize,
//!         doc: DocId,
//!     ) -> tantivy::Result<Option<Arc<[CompiledRule]>>> {
//!         let rule = CompiledRule::from_line(self.texts[doc as usize], &mut self.analyzer);
//!         Ok(Some(rule.into_iter().collect()))
//!     }
//! }
//!
//! # fn main() -> tantivy::Result<()> {
//! let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
//!     .filter(LowerCaser)
//!     .build();
//! let mut schema_builder = Schema::builder();
//! let rule = schema_builder.add_text_field("rule", rule_field_options("synonym_rule"));
//! let index = Index::create_in_ram(schema_builder.build());
//! index
//!     .tokenizers()
//!     .register("synonym_rule", SynonymRuleTokenizer::new(analyzer.clone()));
//! let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
//! let texts = vec!["couch, sofa, divan"];
//! for text in &texts {
//!     writer.add_document(doc!(rule => *text))?;
//! }
//! writer.commit()?;
//!
//! let searcher = index.reader()?.searcher();
//! let mut rules = Rules {
//!     segment: SegmentSynonyms::open(searcher.segment_reader(0), rule)?,
//!     texts,
//!     analyzer,
//! };
//! let expansions = expand(&["red".to_string(), "sofa".to_string()], &mut rules)?;
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
    expand, rule_field_options, Expansion, KeyDocs, SegmentSynonyms, SynonymSource,
    DEFAULT_MAX_RULES_PER_KEY,
};
pub use self::query::expansion_query;
pub use self::rule::{CompiledRule, RuleError, SynonymRule, MAX_RULE_TERMS};
pub use self::tokenizer::SynonymRuleTokenizer;
