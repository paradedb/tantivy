//! Query-time expansion with the synonym rules of an index's segments.

use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::Arc;

use columnar::StrColumn;

use super::rule::{CompiledRule, SynonymRule};
use super::tokenizer::KEY_SEPARATOR;
use crate::fastfield::AliveBitSet;
use crate::schema::{Field, FieldType, IndexRecordOption, Schema};
use crate::tokenizer::{TextAnalyzer, RAW_TOKENIZER_NAME};
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

/// The alternatives that one rule gives a key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleAlternatives {
    /// The rule's document.
    pub doc: DocId,
    /// Each alternative is a list of analyzed tokens.
    pub alternatives: Vec<Vec<String>>,
}

/// A key of a segment, with the rules that list it in document order, deleted ones included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyRules {
    /// The analyzed tokens of the key.
    pub key: Vec<String>,
    /// The rules that list the key.
    pub rules: Vec<RuleAlternatives>,
}

fn check_rule_field(schema: &Schema, field: Field) -> crate::Result<()> {
    let entry = schema.get_field_entry(field);
    match entry.field_type() {
        FieldType::Str(options)
            if options.get_indexing_options().is_some()
                && options.get_fast_field_tokenizer_name() == Some(RAW_TOKENIZER_NAME) =>
        {
            Ok(())
        }
        _ => Err(TantivyError::SchemaError(format!(
            "synonym rule field '{}' must be an indexed text field, fast with the '{}' tokenizer",
            entry.name(),
            RAW_TOKENIZER_NAME
        ))),
    }
}

/// The synonym rules of one segment. The rule field is indexed with a
/// [`SynonymRuleTokenizer`](super::SynonymRuleTokenizer), and it is fast with the `raw` tokenizer.
#[derive(Clone)]
pub struct SegmentSynonyms {
    keys: Arc<InvertedIndexReader>,
    rules: StrColumn,
}

impl SegmentSynonyms {
    /// Opens the rules in `field`.
    pub fn open(segment_reader: &SegmentReader, field: Field) -> crate::Result<Self> {
        let schema = segment_reader.schema();
        check_rule_field(schema, field)?;
        let name = schema.get_field_name(field);
        // Every segment has a column for each fast field, so a missing column is an error.
        let rules = segment_reader.fast_fields().str(name)?.ok_or_else(|| {
            TantivyError::InternalError(format!("synonym rule field '{name}' has no column"))
        })?;
        Ok(SegmentSynonyms {
            keys: segment_reader.inverted_index(field)?,
            rules,
        })
    }

    /// Every key that starts with `token`, with its rules. `analyzer` is the one the field's
    /// tokenizer wraps. The result holds no reader, so a caller can keep it for the life of the
    /// segment.
    pub fn token_rules(
        &self,
        token: &str,
        analyzer: &mut TextAnalyzer,
    ) -> crate::Result<Vec<KeyRules>> {
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
        let mut compiled: HashMap<DocId, Vec<CompiledRule>> = HashMap::new();
        while keys.advance() {
            let Ok(key) = std::str::from_utf8(keys.key()) else {
                continue;
            };
            let key: Vec<String> = key.split(KEY_SEPARATOR).map(str::to_string).collect();
            let mut postings = self
                .keys
                .read_postings_from_terminfo(keys.value(), IndexRecordOption::Basic)?;
            let mut rules = Vec::new();
            let mut doc = postings.doc();
            while doc != TERMINATED {
                let doc_rules = match compiled.entry(doc) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(self.compile(doc, analyzer)?),
                };
                // A document can hold several rules, and only the ones with this key count.
                for rule in doc_rules.iter().filter(|rule| rule.keys.contains(&key)) {
                    rules.push(RuleAlternatives {
                        doc,
                        alternatives: rule.alternatives.clone(),
                    });
                }
                doc = postings.advance();
            }
            found.push(KeyRules { key, rules });
        }
        Ok(found)
    }

    /// The index keeps the text of a rule and not its alternatives, so the rules of a document are
    /// compiled again here.
    fn compile(&self, doc: DocId, analyzer: &mut TextAnalyzer) -> crate::Result<Vec<CompiledRule>> {
        let mut compiled = Vec::new();
        let mut text = String::new();
        for ord in self.rules.term_ords(doc) {
            text.clear();
            if !self.rules.ord_to_str(ord, &mut text)? {
                continue;
            }
            if let Ok(Some(rule)) = SynonymRule::parse(&text) {
                if let Ok(rule) = rule.compile(analyzer) {
                    compiled.push(rule);
                }
            }
        }
        Ok(compiled)
    }
}

/// Where [`expand`](fn@expand) reads the rules of each segment. What a segment holds for a token
/// never changes, so a source can cache [`SegmentSynonyms::token_rules`] for the life of the
/// segment. A rule can be deleted, so its liveness is asked on every expansion.
pub trait SynonymSource {
    /// The number of segments. The other methods name a segment by its ordinal.
    fn num_segments(&self) -> usize;

    /// Every key of the segment that starts with `token`, with its rules.
    fn token_rules(&mut self, segment: usize, token: &str) -> crate::Result<Arc<[KeyRules]>>;

    /// False when the rule in `doc` of the segment is deleted.
    fn is_alive(&mut self, segment: usize, doc: DocId) -> bool;
}

/// Expands query tokens with the rules of `source`. At each position, from left to right, the
/// longest key wins. An alternative is never expanded again, the same as in Lucene's
/// `SynonymGraphFilter`.
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
    let mut fetched = Vec::with_capacity(source.num_segments());
    for segment in 0..source.num_segments() {
        fetched.push(source.token_rules(segment, &run[0])?);
    }
    let mut candidates: Vec<(usize, &KeyRules)> = Vec::new();
    for (segment, keys) in fetched.iter().enumerate() {
        for key_rules in keys.iter() {
            if run.starts_with(&key_rules.key) {
                candidates.push((segment, key_rules));
            }
        }
    }
    candidates.sort_by_key(|(_, key_rules)| Reverse(key_rules.key.len()));
    for same_len in candidates.chunk_by(|(_, left), (_, right)| left.key.len() == right.key.len()) {
        let len = same_len[0].1.key.len();
        let mut alternatives: BTreeSet<&[String]> = BTreeSet::new();
        for &(segment, key_rules) in same_len {
            for rule in &key_rules.rules {
                if source.is_alive(segment, rule.doc) {
                    alternatives.extend(rule.alternatives.iter().map(Vec::as_slice));
                }
            }
        }
        // A key whose alternatives are only the key itself changes nothing, so the scan goes on to
        // a shorter key. The check runs on the union, because `sofa` alone and `sofa => couch`
        // together expand `sofa` to both words.
        let only_the_key = alternatives.len() == 1 && alternatives.first() == Some(&&run[..len]);
        if !alternatives.is_empty() && !only_the_key {
            let alternatives = alternatives.into_iter().map(<[String]>::to_vec).collect();
            return Ok(Some((len, alternatives)));
        }
    }
    Ok(None)
}

#[derive(Clone)]
struct ReaderSegment {
    synonyms: SegmentSynonyms,
    alive: Option<AliveBitSet>,
}

/// Expands query tokens with the synonym rules of every segment of an index, read from the
/// segment readers on each expansion. The rules are spread over the segments, so a key is looked
/// up in each one.
#[derive(Clone)]
pub struct SynonymExpander {
    segments: Vec<ReaderSegment>,
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
    /// An expander over the segments of a searcher and the rules in `field`. The field's
    /// [`SynonymRuleTokenizer`](super::SynonymRuleTokenizer) wraps `analyzer`.
    ///
    /// A rule whose document is deleted in its segment is skipped. The ordinal of a segment is its
    /// position in `segment_readers`.
    pub fn new(
        segment_readers: &[SegmentReader],
        field: Field,
        analyzer: TextAnalyzer,
    ) -> crate::Result<Self> {
        let segments = segment_readers
            .iter()
            .map(|segment_reader| {
                Ok(ReaderSegment {
                    synonyms: SegmentSynonyms::open(segment_reader, field)?,
                    alive: segment_reader.alive_bitset().cloned(),
                })
            })
            .collect::<crate::Result<_>>()?;
        Ok(SynonymExpander { segments, analyzer })
    }

    /// Expands like [`expand`](fn@expand), and skips a rule whose document is deleted in its
    /// segment.
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
        is_alive: impl FnMut(usize, DocId) -> bool,
    ) -> crate::Result<Vec<Expansion>> {
        let mut source = ReaderSource {
            segments: &self.segments,
            analyzer: self.analyzer.clone(),
            is_alive,
        };
        expand(tokens, &mut source)
    }
}

struct ReaderSource<'a, F> {
    segments: &'a [ReaderSegment],
    analyzer: TextAnalyzer,
    is_alive: F,
}

impl<F: FnMut(usize, DocId) -> bool> SynonymSource for ReaderSource<'_, F> {
    fn num_segments(&self) -> usize {
        self.segments.len()
    }

    fn token_rules(&mut self, segment: usize, token: &str) -> crate::Result<Arc<[KeyRules]>> {
        Ok(self.segments[segment]
            .synonyms
            .token_rules(token, &mut self.analyzer)?
            .into())
    }

    fn is_alive(&mut self, segment: usize, doc: DocId) -> bool {
        let in_bitset = self.segments[segment]
            .alive
            .as_ref()
            .is_none_or(|alive| alive.is_alive(doc));
        in_bitset && (self.is_alive)(segment, doc)
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
