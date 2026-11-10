//! Query-time expansion with the synonym rules of an index's segments.

use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::sync::Arc;

use super::rule::CompiledRule;
use super::tokenizer::KEY_SEPARATOR;
use crate::schema::{Field, FieldType, IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
use crate::{DocId, DocSet, InvertedIndexReader, SegmentReader, TantivyError, TERMINATED};

/// A run of query tokens and what it expands to. `alternatives` is `None` when no rule matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expansion {
    /// The query tokens of the run.
    pub tokens: Vec<String>,
    /// The alternatives of the run, each a list of tokens. They are sorted, so the order does not
    /// depend on how the rules are spread over the segments.
    pub alternatives: Option<Vec<Vec<String>>>,
}

impl Expansion {
    /// True when a query for this expansion is one term, which Block-WAND prunes.
    pub fn is_single_term(&self) -> bool {
        match &self.alternatives {
            None => true,
            Some(alternatives) => alternatives.len() == 1 && alternatives[0].len() == 1,
        }
    }

    /// Keeps the first `max` alternatives, and the typed tokens when a rule lists them, so a long
    /// rule never drops what the user typed. A cap of zero keeps one alternative, because an
    /// expansion with none matches nothing.
    pub fn cap(&mut self, max: usize) {
        let Some(alternatives) = &mut self.alternatives else {
            return;
        };
        let max = max.max(1);
        if alternatives.len() <= max {
            return;
        }
        match alternatives.iter().position(|a| *a == self.tokens) {
            Some(typed) if typed >= max => {
                let typed = alternatives.remove(typed);
                alternatives.truncate(max - 1);
                alternatives.push(typed);
            }
            _ => alternatives.truncate(max),
        }
    }
}

/// The rules a key reads by default. A word in more rules reads only the first ones, so a common
/// word costs a bounded lookup, the same way `PhrasePrefixQuery` caps its expansions. Lucene caps
/// a boolean query at the same 1024 clauses.
pub const DEFAULT_MAX_RULES_PER_KEY: usize = 1024;

/// A key of a segment, with the documents of the rules that list it, deleted ones included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyDocs {
    /// The analyzed tokens of the key.
    pub key: Vec<String>,
    /// The documents whose rules list the key, in document order.
    pub docs: Vec<DocId>,
}

/// Options for a field of synonym rules, indexed with the tokenizer registered as
/// `tokenizer_name`. Rules are never scored, so the field has no positions or fieldnorms. The
/// caller keeps the text of the rules and gives it through a [`SynonymSource`].
pub fn rule_field_options(tokenizer_name: &str) -> TextOptions {
    let indexing = TextFieldIndexing::default()
        .set_tokenizer(tokenizer_name)
        .set_index_option(IndexRecordOption::Basic)
        .set_fieldnorms(false);
    TextOptions::default().set_indexing_options(indexing)
}

fn check_rule_field(schema: &Schema, field: Field) -> crate::Result<()> {
    let entry = schema.get_field_entry(field);
    match entry.field_type() {
        FieldType::Str(options) if options.get_indexing_options().is_some() => Ok(()),
        _ => Err(TantivyError::SchemaError(format!(
            "synonym rule field '{}' must be an indexed text field",
            entry.name()
        ))),
    }
}

/// The synonym keys of one segment. The rule field is indexed with a
/// [`SynonymRuleTokenizer`](super::SynonymRuleTokenizer).
#[derive(Clone)]
pub struct SegmentSynonyms {
    keys: Arc<InvertedIndexReader>,
}

impl SegmentSynonyms {
    /// Opens the rules in `field`.
    pub fn open(segment_reader: &SegmentReader, field: Field) -> crate::Result<Self> {
        check_rule_field(segment_reader.schema(), field)?;
        Ok(SegmentSynonyms {
            keys: segment_reader.inverted_index(field)?,
        })
    }

    /// Every key that starts with `token`, with the documents of its rules, at most `max_rules`
    /// of them for a key. Past the limit a key keeps its first documents, so a merge can change
    /// which rules count. The result holds no reader, so a caller can keep it for the life of the
    /// segment.
    pub fn token_keys(&self, token: &str, max_rules: usize) -> crate::Result<Vec<KeyDocs>> {
        let mut found = Vec::new();
        if token.contains(KEY_SEPARATOR) {
            return Ok(found);
        }
        // The keys that start with the token are the token itself and the token followed by the
        // separator, and all of them sort before the token followed by the next byte.
        let mut end = token.as_bytes().to_vec();
        end.push(KEY_SEPARATOR as u8 + 1);
        let mut keys = self
            .keys
            .terms()
            .range()
            .ge(token.as_bytes())
            .lt(&end)
            .into_stream()?;
        while keys.advance() {
            let Ok(key) = std::str::from_utf8(keys.key()) else {
                continue;
            };
            let mut postings = self
                .keys
                .read_postings_from_terminfo(keys.value(), IndexRecordOption::Basic)?;
            let mut docs = Vec::new();
            let mut doc = postings.doc();
            while doc != TERMINATED && docs.len() < max_rules {
                docs.push(doc);
                doc = postings.advance();
            }
            found.push(KeyDocs {
                key: key.split(KEY_SEPARATOR).map(str::to_string).collect(),
                docs,
            });
        }
        Ok(found)
    }
}

/// Where [`expand`](fn@expand) reads the rules of each segment. What a segment holds for a token
/// never changes, so a source can cache [`SegmentSynonyms::token_keys`] for the life of the
/// segment. The text of a document's rules never changes either, so a source can cache them
/// compiled with [`CompiledRule::from_line`]. A rule can be deleted, so its liveness is asked on
/// every expansion.
pub trait SynonymSource {
    /// Every key that starts with `token`, with the documents of its rules, one entry for each
    /// segment. [`alive_rules`](Self::alive_rules) names a segment by its position in this list.
    fn token_keys(&mut self, token: &str) -> crate::Result<Vec<Arc<[KeyDocs]>>>;

    /// The compiled rules of `doc` in the segment, or `None` when the document is deleted.
    fn alive_rules(
        &mut self,
        segment: usize,
        doc: DocId,
    ) -> crate::Result<Option<Arc<[CompiledRule]>>>;
}

/// Expands query tokens with the rules of `source`. At each position, from left to right, the
/// longest key wins. An alternative is never expanded again, the same as in Lucene's
/// `SynonymGraphFilter`. A key can match part of the tokens, so `red running shoes` expands
/// `running shoes` with a rule on that key.
pub fn expand(tokens: &[String], source: &mut impl SynonymSource) -> crate::Result<Vec<Expansion>> {
    let mut expansions = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        match longest_match(&tokens[i..], source)? {
            Some((len, alternatives)) => {
                expansions.push(Expansion {
                    tokens: tokens[i..i + len].to_vec(),
                    alternatives: Some(alternatives),
                });
                i += len;
            }
            None => {
                expansions.push(Expansion {
                    tokens: vec![tokens[i].clone()],
                    alternatives: None,
                });
                i += 1;
            }
        }
    }
    Ok(expansions)
}

/// The longest key that starts `run` and has an alive rule, with its alternatives.
fn longest_match(
    run: &[String],
    source: &mut impl SynonymSource,
) -> crate::Result<Option<(usize, Vec<Vec<String>>)>> {
    let fetched = source.token_keys(&run[0])?;
    let mut candidates: Vec<(usize, &KeyDocs)> = Vec::new();
    for (segment, keys) in fetched.iter().enumerate() {
        for key_docs in keys.iter() {
            if run.starts_with(&key_docs.key) {
                candidates.push((segment, key_docs));
            }
        }
    }
    candidates.sort_by_key(|(_, key_docs)| Reverse(key_docs.key.len()));
    for same_len in candidates.chunk_by(|(_, left), (_, right)| left.key.len() == right.key.len()) {
        let key = &run[..same_len[0].1.key.len()];
        let mut alternatives: BTreeSet<Vec<String>> = BTreeSet::new();
        for &(segment, key_docs) in same_len {
            for &doc in &key_docs.docs {
                let Some(rules) = source.alive_rules(segment, doc)? else {
                    continue;
                };
                // A document can hold several rules, and only the ones with this key count.
                for rule in rules
                    .iter()
                    .filter(|rule| rule.keys.iter().any(|k| k == key))
                {
                    alternatives.extend(rule.alternatives.iter().cloned());
                }
            }
        }
        // A key whose alternatives are only the key itself changes nothing, so the scan goes on to
        // a shorter key. The check runs on the union, because `sofa` alone and `sofa => couch`
        // together expand `sofa` to both words.
        let only_the_key =
            alternatives.len() == 1 && alternatives.first().map(Vec::as_slice) == Some(key);
        if !alternatives.is_empty() && !only_the_key {
            return Ok(Some((key.len(), alternatives.into_iter().collect())));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(terms: &[&str]) -> Vec<String> {
        terms.iter().map(|term| term.to_string()).collect()
    }

    #[test]
    fn the_cap_keeps_the_typed_tokens() {
        let group = || {
            Some(vec![
                strings(&["couch"]),
                strings(&["sofa"]),
                strings(&["divan"]),
            ])
        };
        let mut divan = Expansion {
            tokens: strings(&["divan"]),
            alternatives: group(),
        };
        divan.cap(2);
        assert_eq!(
            divan.alternatives,
            Some(vec![strings(&["couch"]), strings(&["divan"])])
        );
        let mut couch = Expansion {
            tokens: strings(&["couch"]),
            alternatives: group(),
        };
        couch.cap(2);
        assert_eq!(
            couch.alternatives,
            Some(vec![strings(&["couch"]), strings(&["sofa"])])
        );
        let mut zero = Expansion {
            tokens: strings(&["couch"]),
            alternatives: group(),
        };
        zero.cap(0);
        assert_eq!(zero.alternatives, Some(vec![strings(&["couch"])]));
    }
}
