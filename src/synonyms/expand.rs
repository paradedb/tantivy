//! Query-time expansion over the synonym components of an index's segments.

use super::dictionary::SynonymDictionary;
use crate::fastfield::AliveBitSet;
use crate::SegmentReader;

/// A run of query tokens and what it expands to. `alternatives` is `None` when no rule matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expansion {
    /// The query tokens of the run.
    pub tokens: Vec<String>,
    /// The alternatives of the run, each a list of tokens. Within a segment they come in
    /// document order, and the segments come in the order the expander was given them.
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

struct SegmentSynonyms {
    dictionary: SynonymDictionary,
    alive: Option<AliveBitSet>,
}

impl SegmentSynonyms {
    fn is_alive(&self, doc: crate::DocId) -> bool {
        self.alive
            .as_ref()
            .is_none_or(|alive| doc < alive.bitset().max_value() && alive.is_alive(doc))
    }
}

/// Expands query tokens with the synonym rules of every segment of an index.
///
/// Until a merge, the rules of an index are spread over its segments, so a key is looked up in
/// every segment and a deleted rule is skipped.
pub struct SynonymExpander {
    segments: Vec<SegmentSynonyms>,
    max_key_tokens: usize,
}

impl SynonymExpander {
    /// An expander over the segments of a searcher. A segment without a synonym component
    /// contributes nothing.
    pub fn new(segment_readers: &[SegmentReader]) -> crate::Result<Self> {
        let mut segments = Vec::with_capacity(segment_readers.len());
        for segment_reader in segment_readers {
            if let Some(dictionary) = segment_reader.synonym_dictionary()? {
                segments.push(SegmentSynonyms {
                    dictionary,
                    alive: segment_reader.alive_bitset().cloned(),
                });
            }
        }
        Ok(Self::from_segments(segments))
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

    /// The alternatives of `key` from the alive rules of every segment, or `None` when no alive
    /// rule has the key.
    ///
    /// A key whose alternatives are only the key itself changes nothing and is `None` as well,
    /// so the expansion tries a shorter key at the same position. The check runs on the union,
    /// because `sofa` alone and `sofa => couch` together expand `sofa` to both words.
    pub fn alternatives(&self, key: &[String]) -> crate::Result<Option<Vec<Vec<String>>>> {
        let mut alternatives: Vec<Vec<String>> = Vec::new();
        for segment in &self.segments {
            let Some(groups) = segment.dictionary.groups_of(key)? else {
                continue;
            };
            for group in groups {
                if !segment.is_alive(group.doc) {
                    continue;
                }
                for alternative in segment.dictionary.alternatives(&group)? {
                    if !alternatives.contains(&alternative) {
                        alternatives.push(alternative);
                    }
                }
            }
        }
        let only_the_key = alternatives.len() == 1 && alternatives[0] == key;
        Ok((!alternatives.is_empty() && !only_the_key).then_some(alternatives))
    }

    /// Longest key first, left to right, one pass. An alternative is never expanded again, the
    /// same as Lucene's `SynonymGraphFilter`, so a pasted Elasticsearch file behaves the same.
    pub fn expand(&self, tokens: &[String]) -> crate::Result<Vec<Expansion>> {
        let mut expansions = Vec::with_capacity(tokens.len());
        let mut i = 0;
        'positions: while i < tokens.len() {
            for len in (1..=self.max_key_tokens.min(tokens.len() - i)).rev() {
                let key = &tokens[i..i + len];
                if let Some(alternatives) = self.alternatives(key)? {
                    expansions.push(Expansion {
                        tokens: key.to_vec(),
                        alternatives: Some(alternatives),
                    });
                    i += len;
                    continue 'positions;
                }
            }
            expansions.push(Expansion {
                tokens: vec![tokens[i].clone()],
                alternatives: None,
            });
            i += 1;
        }
        Ok(expansions)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use common::file_slice::FileSlice;

    use super::*;
    use crate::synonyms::dictionary::{encode_key, GroupRef, SynonymFileWriter};
    use crate::synonyms::SynonymRule;
    use crate::tokenizer::{LowerCaser, SimpleTokenizer, Stemmer, StopWordFilter, TextAnalyzer};
    use crate::DocId;

    fn plain() -> TextAnalyzer {
        TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .build()
    }

    fn stemming() -> TextAnalyzer {
        TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .filter(Stemmer::default())
            .build()
    }

    /// One group per rule, every rule alive, the layout the segment writer produces.
    fn expander(mut analyzer: TextAnalyzer, rules: &[&str]) -> SynonymExpander {
        let mut file = SynonymFileWriter::new(Vec::new());
        let mut keys: BTreeMap<Vec<u8>, Vec<GroupRef>> = BTreeMap::new();
        for (doc, rule) in rules.iter().enumerate() {
            let Some(rule) = SynonymRule::parse(rule).unwrap() else {
                continue;
            };
            let Some(compiled) = rule.compile(&mut analyzer) else {
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
        let bytes = writer.finish().unwrap();
        let dictionary = SynonymDictionary::open(FileSlice::from(bytes)).unwrap();
        SynonymExpander::from_segments(vec![SegmentSynonyms {
            dictionary,
            alive: None,
        }])
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
        let expander = expander(plain(), &["running shoes, sneakers", "shoes, footwear"]);
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

    #[test]
    fn rules_go_through_the_analyzer() {
        let expander = expander(
            stemming(),
            &["Couch, Sofa, Divan", "Sneakers => Running Shoes"],
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
    }

    #[test]
    fn same_key_in_two_rules_merges() {
        let expander = expander(plain(), &["sofa, couch", "sofa => settee"]);
        assert_eq!(
            expand(&expander, &["sofa"]),
            vec![expanded(
                &["sofa"],
                Some(&[&["sofa"], &["couch"], &["settee"]])
            )]
        );
    }

    #[test]
    fn a_key_that_maps_to_itself_keeps_the_typed_term_beside_another_rule() {
        let both = expander(plain(), &["sofa", "sofa => couch"]);
        assert_eq!(
            expand(&both, &["sofa"]),
            vec![expanded(&["sofa"], Some(&[&["sofa"], &["couch"]]))]
        );
        let stemmed = expander(stemming(), &["run, running, runs", "run => jog"]);
        assert_eq!(
            expand(&stemmed, &["run"]),
            vec![expanded(&["run"], Some(&[&["run"], &["jog"]]))]
        );
        let alone = expander(stemming(), &["run, running, runs", "sofa", "# note", ""]);
        assert_eq!(
            expand(&alone, &["run", "sofa"]),
            vec![expanded(&["run"], None), expanded(&["sofa"], None)]
        );
    }

    #[test]
    fn a_right_side_of_stopwords_keeps_the_typed_term() {
        let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(LowerCaser)
            .filter(StopWordFilter::remove(vec!["the".to_string()]))
            .build();
        let expander = expander(analyzer, &["sofa => the", "couch => the settee"]);
        assert_eq!(
            expand(&expander, &["sofa", "couch"]),
            vec![
                expanded(&["sofa"], None),
                expanded(&["couch"], Some(&[&["settee"]])),
            ]
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
