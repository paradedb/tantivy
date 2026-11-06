//! The query of an expansion. The query shape decides how a group scores and whether Block-WAND
//! can prune it, so it is built here and not by each caller.

use super::Expansion;
use crate::query::{DisjunctionMaxQuery, PhraseQuery, Query, SynonymQuery, TermQuery};
use crate::schema::IndexRecordOption;
use crate::{TantivyError, Term};

/// The query that matches `expansion`. `term` turns a token into a term of the field the rules
/// expand, and `record` is the option of the term queries.
///
/// A run that no rule matched and a group with one alternative are that one term or phrase. A
/// group of single-token alternatives is a [`SynonymQuery`], which scores them as one term, so
/// Block-WAND can prune it. Any other group is a [`DisjunctionMaxQuery`] whose multi-token
/// alternatives are phrases. A phrase needs `record` to have positions.
pub fn expansion_query(
    expansion: &Expansion,
    term: impl Fn(&str) -> Term,
    record: IndexRecordOption,
) -> crate::Result<Box<dyn Query>> {
    let alternatives = expansion.alternatives.as_deref().unwrap_or_default();
    match alternatives {
        [] => tokens_query(&expansion.tokens, &term, record),
        [alternative] => tokens_query(alternative, &term, record),
        _ if alternatives
            .iter()
            .all(|alternative| alternative.len() == 1) =>
        {
            let terms = alternatives
                .iter()
                .map(|alternative| term(&alternative[0]))
                .collect();
            Ok(Box::new(SynonymQuery::new(terms)))
        }
        _ => {
            let disjuncts = alternatives
                .iter()
                .map(|alternative| tokens_query(alternative, &term, record))
                .collect::<crate::Result<_>>()?;
            Ok(Box::new(DisjunctionMaxQuery::new(disjuncts)))
        }
    }
}

fn tokens_query(
    tokens: &[String],
    term: &impl Fn(&str) -> Term,
    record: IndexRecordOption,
) -> crate::Result<Box<dyn Query>> {
    match tokens {
        [token] => Ok(Box::new(TermQuery::new(term(token), record))),
        _ if !record.has_positions() => Err(TantivyError::InvalidArgument(format!(
            "the synonym phrase {tokens:?} needs a field indexed with positions"
        ))),
        _ => Ok(Box::new(PhraseQuery::new(
            tokens.iter().map(|token| term(token)).collect(),
        ))),
    }
}
