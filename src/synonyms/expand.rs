//! Query-time expansion with the synonym rules of an index's segments.

use std::fmt;
use std::sync::Arc;

use columnar::StrColumn;

use super::rule::{SynonymRule, MAX_RULE_TERMS};
use super::tokenizer::{join_key, key_tokens, KEY_SEPARATOR};
use crate::fastfield::AliveBitSet;
use crate::postings::TermInfo;
use crate::schema::{Field, FieldType, IndexRecordOption, Schema};
use crate::tokenizer::TextAnalyzer;
use crate::{DocId, DocSet, InvertedIndexReader, SegmentReader, TantivyError, TERMINATED};

/// A run of query tokens and what it expands to. `alternatives` is `None` when no rule matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expansion {
    /// The query tokens of the run.
    pub tokens: Vec<String>,
    /// The alternatives of the run, each a list of tokens. Within a segment they come in
    /// document order, and the segments come in the order the expander was given them. A
    /// searcher lists its segments in no fixed order.
    pub alternatives: Option<Vec<Vec<String>>>,
}

impl Expansion {
    /// True when a query for this expansion is one term, which Block-WAND prunes. A token in no
    /// rule, or a rule whose one alternative is one word, is a single term.
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

#[derive(Clone)]
struct SegmentRules {
    /// The ordinal `expand_with` passes to its liveness check.
    ord: usize,
    keys: Arc<InvertedIndexReader>,
    rules: StrColumn,
    alive: Option<AliveBitSet>,
}

impl SegmentRules {
    fn is_alive(&self, doc: DocId) -> bool {
        self.alive
            .as_ref()
            .is_none_or(|alive| doc < alive.bitset().max_value() && alive.is_alive(doc))
    }

    /// The index keeps the text of a rule and not its alternatives, so the rule is compiled again
    /// here. A document can hold several rules, and only the ones with `key` count.
    fn add_alternatives(
        &self,
        doc: DocId,
        key: &[String],
        analyzer: &mut TextAnalyzer,
        alternatives: &mut Vec<Vec<String>>,
    ) -> crate::Result<()> {
        let mut text = String::new();
        for ord in self.rules.term_ords(doc) {
            text.clear();
            if !self.rules.ord_to_str(ord, &mut text)? {
                continue;
            }
            let Ok(Some(rule)) = SynonymRule::parse(&text) else {
                continue;
            };
            let Ok(Some(rule)) = rule.compile(analyzer) else {
                continue;
            };
            if !rule.keys.iter().any(|rule_key| rule_key.as_slice() == key) {
                continue;
            }
            for alternative in rule.alternatives {
                if !alternatives.contains(&alternative) {
                    alternatives.push(alternative);
                }
            }
        }
        Ok(())
    }
}

fn check_rule_field(schema: &Schema, field: Field) -> crate::Result<()> {
    let entry = schema.get_field_entry(field);
    match entry.field_type() {
        FieldType::Str(options)
            if options.get_indexing_options().is_some() && options.is_fast() =>
        {
            Ok(())
        }
        _ => Err(TantivyError::SchemaError(format!(
            "synonym rule field '{}' must be an indexed and fast text field",
            entry.name()
        ))),
    }
}

/// Expands query tokens with the synonym rules of every segment of an index. The rules are
/// spread over the segments, so a key is looked up in each one.
#[derive(Clone)]
pub struct SynonymExpander {
    segments: Vec<SegmentRules>,
    analyzer: TextAnalyzer,
}

impl fmt::Debug for SynonymExpander {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SynonymExpander")
            .field("segments", &self.segments.len())
            .finish()
    }
}

impl SynonymExpander {
    /// An expander over the segments of a searcher and the rules in `field`. The field is indexed
    /// with a [`SynonymRuleTokenizer`](super::SynonymRuleTokenizer) that wraps `analyzer`, and it
    /// is fast with the `raw` tokenizer.
    ///
    /// A rule whose document is deleted in its segment is skipped. The ordinal of a segment is its
    /// position in `segment_readers`.
    pub fn new(
        segment_readers: &[SegmentReader],
        field: Field,
        analyzer: TextAnalyzer,
    ) -> crate::Result<Self> {
        let mut segments = Vec::with_capacity(segment_readers.len());
        for (ord, segment_reader) in segment_readers.iter().enumerate() {
            let schema = segment_reader.schema();
            check_rule_field(schema, field)?;
            // A segment where no document has a rule has no column.
            let Some(rules) = segment_reader
                .fast_fields()
                .str(schema.get_field_name(field))?
            else {
                continue;
            };
            segments.push(SegmentRules {
                ord,
                keys: segment_reader.inverted_index(field)?,
                rules,
                alive: segment_reader.alive_bitset().cloned(),
            });
        }
        Ok(SynonymExpander { segments, analyzer })
    }

    /// Longest key first, left to right, one pass. An alternative is never expanded again, the
    /// same as Lucene's `SynonymGraphFilter`.
    pub fn expand(&self, tokens: &[String]) -> crate::Result<Vec<Expansion>> {
        self.expand_with(tokens, |_, _| true)
    }

    /// Expands like [`expand`](Self::expand), and skips a rule when `is_alive` returns false for
    /// the ordinal of its segment and its document. A caller whose deletes are not in the
    /// segments' delete bitsets yet, such as a Postgres scan under an MVCC snapshot, checks them
    /// here.
    pub fn expand_with(
        &self,
        tokens: &[String],
        mut is_alive: impl FnMut(usize, DocId) -> bool,
    ) -> crate::Result<Vec<Expansion>> {
        let mut analyzer = self.analyzer.clone();
        let mut expansions = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            // A token with the separator would join into a key of several tokens.
            let run = tokens[i..]
                .iter()
                .take(MAX_RULE_TERMS)
                .take_while(|token| !token.as_bytes().contains(&KEY_SEPARATOR))
                .count();
            match self.longest_match(&tokens[i..i + run], &mut analyzer, &mut is_alive)? {
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
    ///
    /// The keys that start `run` all sort between its first token and `run` itself, so one range
    /// scan per segment finds every candidate length.
    fn longest_match(
        &self,
        run: &[String],
        analyzer: &mut TextAnalyzer,
        is_alive: &mut impl FnMut(usize, DocId) -> bool,
    ) -> crate::Result<Option<(usize, Vec<Vec<String>>)>> {
        if run.is_empty() {
            return Ok(None);
        }
        let joined = join_key(run);
        let joined = joined.as_bytes();
        let mut candidates: Vec<(usize, usize, TermInfo)> = Vec::new();
        for (segment, rules) in self.segments.iter().enumerate() {
            let mut keys = rules
                .keys
                .terms()
                .range()
                .ge(run[0].as_bytes())
                .le(joined)
                .into_stream()?;
            while keys.advance() {
                let key = keys.key();
                let ends_on_a_token = joined
                    .get(key.len())
                    .is_none_or(|&byte| byte == KEY_SEPARATOR);
                if joined.starts_with(key) && ends_on_a_token {
                    candidates.push((key_tokens(key), segment, keys.value().clone()));
                }
            }
        }
        // Longest first. The sort is stable, so the segments of one length keep their order.
        candidates.sort_by(|left, right| right.0.cmp(&left.0));
        for same_len in candidates.chunk_by(|left, right| left.0 == right.0) {
            let len = same_len[0].0;
            let mut alternatives: Vec<Vec<String>> = Vec::new();
            for (_, segment, term_info) in same_len {
                let rules = &self.segments[*segment];
                let mut postings = rules
                    .keys
                    .read_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
                let mut doc = postings.doc();
                while doc != TERMINATED {
                    if rules.is_alive(doc) && is_alive(rules.ord, doc) {
                        rules.add_alternatives(doc, &run[..len], analyzer, &mut alternatives)?;
                    }
                    doc = postings.advance();
                }
            }
            // A key whose alternatives are only the key itself changes nothing, so the scan goes on
            // to a shorter key. The check runs on the union, because `sofa` alone and
            // `sofa => couch` together expand `sofa` to both words.
            let only_the_key = alternatives.len() == 1 && alternatives[0] == run[..len];
            if !alternatives.is_empty() && !only_the_key {
                return Ok(Some((len, alternatives)));
            }
        }
        Ok(None)
    }
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
