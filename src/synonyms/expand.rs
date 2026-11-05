//! Query-time expansion over the synonym components of an index's segments.

use std::fmt;
use std::ops::Range;

use super::dictionary::{encode_key, key_tokens, SynonymDictionary, KEY_SEPARATOR};
use crate::fastfield::AliveBitSet;
use crate::{DocId, SegmentReader};

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
struct SegmentSynonyms {
    /// The ordinal `expand_with` passes to its liveness check.
    ord: usize,
    dictionary: SynonymDictionary,
    alive: Option<AliveBitSet>,
}

impl SegmentSynonyms {
    fn is_alive(&self, doc: DocId) -> bool {
        self.alive
            .as_ref()
            .is_none_or(|alive| doc < alive.bitset().max_value() && alive.is_alive(doc))
    }
}

/// Expands query tokens with the synonym rules of every segment of an index.
///
/// Until a merge, the rules of an index are spread over its segments, so a key is looked up in
/// every segment and a deleted rule is skipped.
#[derive(Clone)]
pub struct SynonymExpander {
    segments: Vec<SegmentSynonyms>,
    max_key_tokens: usize,
}

impl fmt::Debug for SynonymExpander {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SynonymExpander")
            .field("segments", &self.segments.len())
            .field("max_key_tokens", &self.max_key_tokens)
            .finish()
    }
}

impl SynonymExpander {
    /// An expander over the segments of a searcher. A rule whose document is deleted in its
    /// segment is skipped. The ordinal of a segment is its position in `segment_readers`.
    pub fn new(segment_readers: &[SegmentReader]) -> crate::Result<Self> {
        let mut segments = Vec::with_capacity(segment_readers.len());
        for (ord, segment_reader) in segment_readers.iter().enumerate() {
            if let Some(dictionary) = segment_reader.synonym_dictionary()? {
                segments.push(SegmentSynonyms {
                    ord,
                    dictionary,
                    alive: segment_reader.alive_bitset().cloned(),
                });
            }
        }
        Ok(Self::from_segments(segments))
    }

    /// An expander over dictionaries the caller opened. Every rule is alive unless the check of
    /// `expand_with` says otherwise. The ordinal of a dictionary is its position here.
    pub fn from_dictionaries(dictionaries: Vec<SynonymDictionary>) -> Self {
        Self::from_segments(
            dictionaries
                .into_iter()
                .enumerate()
                .map(|(ord, dictionary)| SegmentSynonyms {
                    ord,
                    dictionary,
                    alive: None,
                })
                .collect(),
        )
    }

    fn from_segments(segments: Vec<SegmentSynonyms>) -> Self {
        let max_key_tokens = segments
            .iter()
            .map(|segment| segment.dictionary.max_key_tokens())
            .max()
            .unwrap_or(0);
        SynonymExpander {
            segments,
            max_key_tokens,
        }
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
        let mut expansions = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            // A token with the separator would join into a key of several tokens.
            let run = tokens[i..]
                .iter()
                .take(self.max_key_tokens)
                .take_while(|token| !token.as_bytes().contains(&KEY_SEPARATOR))
                .count();
            match self.longest_match(&tokens[i..i + run], &mut is_alive)? {
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
        is_alive: &mut impl FnMut(usize, DocId) -> bool,
    ) -> crate::Result<Option<(usize, Vec<Vec<String>>)>> {
        if run.is_empty() {
            return Ok(None);
        }
        let joined = encode_key(run);
        let mut candidates: Vec<(usize, usize, Range<u64>)> = Vec::new();
        for (segment, synonyms) in self.segments.iter().enumerate() {
            let mut keys = synonyms
                .dictionary
                .keys_between(run[0].as_bytes(), &joined)?;
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
            for (_, segment, refs) in same_len {
                let synonyms = &self.segments[*segment];
                for group in synonyms.dictionary.groups_in(refs.clone())? {
                    if !synonyms.is_alive(group.doc) || !is_alive(synonyms.ord, group.doc) {
                        continue;
                    }
                    for alternative in synonyms.dictionary.alternatives(&group)? {
                        if !alternatives.contains(&alternative) {
                            alternatives.push(alternative);
                        }
                    }
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
    use common::file_slice::FileSlice;

    use super::*;
    use crate::synonyms::dictionary::{GroupRef, SynonymFileWriter};
    use crate::synonyms::SynonymRule;
    use crate::tokenizer::{LowerCaser, SimpleTokenizer, StopWordFilter, TextAnalyzer};

    fn plain() -> TextAnalyzer {
        TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .build()
    }

    #[cfg(feature = "stemmer")]
    fn stemming() -> TextAnalyzer {
        TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .filter(crate::tokenizer::Stemmer::default())
            .build()
    }

    /// One group per rule, the layout the segment writer produces.
    fn dictionary(mut analyzer: TextAnalyzer, rules: &[&str]) -> SynonymDictionary {
        let mut file = SynonymFileWriter::new(Vec::new());
        let mut keys: std::collections::BTreeMap<Vec<u8>, Vec<GroupRef>> = Default::default();
        for (doc, rule) in rules.iter().enumerate() {
            let Some(rule) = SynonymRule::parse(rule).unwrap() else {
                continue;
            };
            let Some(compiled) = rule.compile(&mut analyzer).unwrap() else {
                continue;
            };
            let group = file
                .add_group(doc as DocId, &compiled.alternatives)
                .unwrap();
            for key in &compiled.keys {
                keys.entry(encode_key(key)).or_default().push(group);
            }
        }
        let mut writer = file.into_keys();
        for (key, groups) in &keys {
            writer.insert(key, groups).unwrap();
        }
        SynonymDictionary::open(FileSlice::from(writer.finish().unwrap())).unwrap()
    }

    fn expander(analyzer: TextAnalyzer, rules: &[&str]) -> SynonymExpander {
        SynonymExpander::from_dictionaries(vec![dictionary(analyzer, rules)])
    }

    fn strings(terms: &[&str]) -> Vec<String> {
        terms.iter().map(|term| term.to_string()).collect()
    }

    fn expanded(tokens: &[&str], alternatives: Option<&[&[&str]]>) -> Expansion {
        Expansion {
            tokens: strings(tokens),
            alternatives: alternatives
                .map(|alternatives| alternatives.iter().map(|alt| strings(alt)).collect()),
        }
    }

    fn expand(expander: &SynonymExpander, tokens: &[&str]) -> Vec<Expansion> {
        expander.expand(&strings(tokens)).unwrap()
    }

    #[test]
    fn equivalents_expand_to_the_group() {
        let expander = expander(plain(), &["couch, sofa, divan"]);
        assert_eq!(
            expand(&expander, &["red", "sofa"]),
            vec![
                expanded(&["red"], None),
                expanded(&["sofa"], Some(&[&["couch"], &["sofa"], &["divan"]])),
            ]
        );
    }

    #[test]
    fn directed_rules_replace_the_left_side() {
        let expander = expander(plain(), &["sneakers => running shoes"]);
        assert_eq!(
            expand(&expander, &["sneakers"]),
            vec![expanded(&["sneakers"], Some(&[&["running", "shoes"]]))]
        );
        assert_eq!(
            expand(&expander, &["running", "shoes"]),
            vec![expanded(&["running"], None), expanded(&["shoes"], None)]
        );
    }

    #[test]
    fn longest_key_wins_and_outputs_are_not_expanded_again() {
        let expander = expander(
            plain(),
            &[
                "running shoes, sneakers",
                "shoes, footwear",
                "running shirts => shirts",
            ],
        );
        assert_eq!(
            expand(&expander, &["running", "shoes"]),
            vec![expanded(
                &["running", "shoes"],
                Some(&[&["running", "shoes"], &["sneakers"]])
            )]
        );
        assert_eq!(
            expand(&expander, &["blue", "shoes"]),
            vec![
                expanded(&["blue"], None),
                expanded(&["shoes"], Some(&[&["shoes"], &["footwear"]])),
            ]
        );
    }

    #[cfg(feature = "stemmer")]
    #[test]
    fn rules_go_through_the_analyzer() {
        let expander = expander(
            stemming(),
            &[
                "Couch, Sofa, Divan",
                "Sneakers => Running Shoes",
                "run, running, runs",
                "run => jog",
            ],
        );
        assert_eq!(
            expand(&expander, &["sneaker"]),
            vec![expanded(&["sneaker"], Some(&[&["run", "shoe"]]))]
        );
        assert_eq!(
            expand(&expander, &["divan"]),
            vec![expanded(
                &["divan"],
                Some(&[&["couch"], &["sofa"], &["divan"]])
            )]
        );
        assert_eq!(
            expand(&expander, &["sneakers"]),
            vec![expanded(&["sneakers"], None)]
        );
        assert_eq!(
            expand(&expander, &["run"]),
            vec![expanded(&["run"], Some(&[&["run"], &["jog"]]))]
        );
    }

    #[test]
    fn a_key_that_maps_to_itself_keeps_the_typed_term_beside_another_rule() {
        let both = expander(plain(), &["sofa", "sofa => couch"]);
        assert_eq!(
            expand(&both, &["sofa"]),
            vec![expanded(&["sofa"], Some(&[&["sofa"], &["couch"]]))]
        );
        let alone = expander(plain(), &["sofa", "# note", ""]);
        assert_eq!(expand(&alone, &["sofa"]), vec![expanded(&["sofa"], None)]);
    }

    #[test]
    fn a_right_side_of_stopwords_keeps_the_typed_term() {
        let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .filter(StopWordFilter::remove(vec!["the".to_string()]))
            .build();
        let expander = expander(analyzer, &["sofa => the", "couch => the, settee"]);
        assert_eq!(
            expand(&expander, &["sofa", "couch"]),
            vec![
                expanded(&["sofa"], None),
                expanded(&["couch"], Some(&[&["settee"]])),
            ]
        );
    }

    #[test]
    fn the_caller_check_drops_a_rule_and_a_shorter_key_takes_over() {
        let first = dictionary(plain(), &["running shoes, sneakers", "shoes, footwear"]);
        let second = dictionary(plain(), &["shoes => boots", "shoes, footwear"]);
        let expander = SynonymExpander::from_dictionaries(vec![first, second]);
        let expansions = expander
            .expand_with(&strings(&["running", "shoes"]), |segment, doc| {
                (segment, doc) != (0, 0)
            })
            .unwrap();
        assert_eq!(
            expansions,
            vec![
                expanded(&["running"], None),
                expanded(&["shoes"], Some(&[&["shoes"], &["footwear"], &["boots"]])),
            ]
        );
    }

    #[test]
    fn a_query_token_with_a_zero_byte_matches_no_key() {
        let expander = expander(plain(), &["a b, c"]);
        assert_eq!(
            expand(&expander, &["a\0b"]),
            vec![expanded(&["a\0b"], None)]
        );
        assert_eq!(
            expand(&expander, &["a", "b"]),
            vec![expanded(&["a", "b"], Some(&[&["a", "b"], &["c"]]))]
        );
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
