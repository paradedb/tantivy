//! Synonym rules compiled per segment, and the query-time expansion that reads them.
//!
//! The `.syn` component of a segment is a dictionary from the analyzed tokens of a rule side to
//! its alternatives. [`SynonymExpander`] reads the dictionaries of every segment and rewrites a
//! list of query tokens into [`Expansion`]s, the longest key first, the way Lucene's
//! `SynonymGraphFilter` does.

mod dictionary;
mod expand;
mod rule;

pub use self::dictionary::{RuleAlternatives, SynonymDictionary, SYNONYM_EXT};
pub use self::expand::{Expansion, SynonymExpander};
pub use self::rule::{RuleError, SynonymRule, MAX_RULE_TERMS};
