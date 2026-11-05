//! Tests of synonym expansion over an index, across writes, deletes and merges.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::*;
use crate::indexer::NoMergePolicy;
use crate::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, FAST, INDEXED, TEXT,
};
use crate::tokenizer::{LowerCaser, SimpleTokenizer, TextAnalyzer};
use crate::{doc, DocId, Index, IndexWriter, Searcher, TantivyDocument, TantivyError, Term};

const RULE_TOKENIZER: &str = "synonym_rule";

struct RulesIndex {
    index: Index,
    writer: IndexWriter,
    analyzer: TextAnalyzer,
    id: Field,
    rule: Field,
}

impl RulesIndex {
    fn new(analyzer: TextAnalyzer) -> Self {
        let mut schema_builder = Schema::builder();
        let id = schema_builder.add_u64_field("id", FAST | INDEXED);
        let indexing = TextFieldIndexing::default()
            .set_tokenizer(RULE_TOKENIZER)
            .set_index_option(IndexRecordOption::Basic);
        let rule = schema_builder.add_text_field(
            "rule",
            TextOptions::default()
                .set_indexing_options(indexing)
                .set_fast("raw"),
        );
        let index = Index::create_in_ram(schema_builder.build());
        index
            .tokenizers()
            .register(RULE_TOKENIZER, SynonymRuleTokenizer::new(analyzer.clone()));
        let writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();
        writer.set_merge_policy(Box::new(NoMergePolicy));
        RulesIndex {
            index,
            writer,
            analyzer,
            id,
            rule,
        }
    }

    /// One rule per document, the ids counting from zero, in one segment.
    fn with_rules(analyzer: TextAnalyzer, rules: &[&str]) -> Self {
        let mut index = Self::new(analyzer);
        for (id, rule) in rules.iter().enumerate() {
            index.add(id as u64, rule);
        }
        index.commit();
        index
    }

    fn add(&mut self, id: u64, rule: &str) {
        self.writer
            .add_document(doc!(self.id => id, self.rule => rule))
            .unwrap();
    }

    fn delete(&mut self, id: u64) {
        self.writer.delete_term(Term::from_field_u64(self.id, id));
    }

    fn commit(&mut self) {
        self.writer.commit().unwrap();
    }

    fn merge_all(&mut self) {
        let segment_ids = self.index.searchable_segment_ids().unwrap();
        self.writer.merge(&segment_ids).wait().unwrap();
    }

    fn searcher(&self) -> Searcher {
        self.index.reader().unwrap().searcher()
    }

    fn expander(&self) -> SynonymExpander {
        let searcher = self.searcher();
        SynonymExpander::new(searcher.segment_readers(), self.rule, self.analyzer.clone()).unwrap()
    }

    fn expand(&self, tokens: &[&str]) -> Vec<Expansion> {
        self.expander().expand(&strings(tokens)).unwrap()
    }

    /// The alternatives of a query that is exactly one key, or `None` when another expansion
    /// wins.
    fn alternatives(&self, key: &[&str]) -> Option<Vec<Vec<String>>> {
        let key = strings(key);
        match self.expander().expand(&key).unwrap().as_slice() {
            [expansion] if expansion.tokens == key => expansion.alternatives.clone(),
            _ => None,
        }
    }
}

fn plain() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .build()
}

fn strings(terms: &[&str]) -> Vec<String> {
    terms.iter().map(|term| term.to_string()).collect()
}

fn lists(lists: &[&[&str]]) -> Vec<Vec<String>> {
    lists.iter().map(|list| strings(list)).collect()
}

fn expanded(tokens: &[&str], alternatives: Option<&[&[&str]]>) -> Expansion {
    Expansion {
        tokens: strings(tokens),
        alternatives: alternatives.map(lists),
    }
}

#[test]
fn directed_rules_replace_the_left_side() {
    let index = RulesIndex::with_rules(plain(), &["sneakers => running shoes"]);
    assert_eq!(
        index.expand(&["sneakers"]),
        vec![expanded(&["sneakers"], Some(&[&["running", "shoes"]]))]
    );
    assert_eq!(
        index.expand(&["running", "shoes"]),
        vec![expanded(&["running"], None), expanded(&["shoes"], None)]
    );
}

#[test]
fn longest_key_wins_and_outputs_are_not_expanded_again() {
    let index = RulesIndex::with_rules(
        plain(),
        &[
            "running shoes, sneakers",
            "shoes, footwear",
            "running shirts => shirts",
        ],
    );
    assert_eq!(
        index.expand(&["running", "shoes"]),
        vec![expanded(
            &["running", "shoes"],
            Some(&[&["running", "shoes"], &["sneakers"]])
        )]
    );
    assert_eq!(
        index.expand(&["blue", "shoes"]),
        vec![
            expanded(&["blue"], None),
            expanded(&["shoes"], Some(&[&["footwear"], &["shoes"]])),
        ]
    );
}

#[cfg(feature = "stemmer")]
#[test]
fn rules_go_through_the_analyzer() {
    let stemming = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(crate::tokenizer::Stemmer::default())
        .build();
    let index = RulesIndex::with_rules(
        stemming,
        &[
            "Couch, Sofa, Divan",
            "Sneakers => Running Shoes",
            "run, running, runs",
            "run => jog",
        ],
    );
    assert_eq!(
        index.expand(&["sneaker"]),
        vec![expanded(&["sneaker"], Some(&[&["run", "shoe"]]))]
    );
    assert_eq!(
        index.expand(&["divan"]),
        vec![expanded(
            &["divan"],
            Some(&[&["couch"], &["divan"], &["sofa"]])
        )]
    );
    assert_eq!(
        index.expand(&["sneakers"]),
        vec![expanded(&["sneakers"], None)]
    );
    assert_eq!(
        index.expand(&["run"]),
        vec![expanded(&["run"], Some(&[&["jog"], &["run"]]))]
    );
}

#[test]
fn a_key_that_maps_to_itself_keeps_the_typed_term_beside_another_rule() {
    let both = RulesIndex::with_rules(plain(), &["sofa", "sofa => couch"]);
    assert_eq!(
        both.expand(&["sofa"]),
        vec![expanded(&["sofa"], Some(&[&["couch"], &["sofa"]]))]
    );
    let alone = RulesIndex::with_rules(plain(), &["sofa", "# note", ""]);
    assert_eq!(alone.expand(&["sofa"]), vec![expanded(&["sofa"], None)]);
}

#[test]
fn a_query_token_with_a_zero_byte_matches_no_key() {
    let index = RulesIndex::with_rules(plain(), &["a b, c"]);
    assert_eq!(index.expand(&["a\0b"]), vec![expanded(&["a\0b"], None)]);
    assert_eq!(
        index.expand(&["a", "b"]),
        vec![expanded(&["a", "b"], Some(&[&["a", "b"], &["c"]]))]
    );
}

#[test]
fn two_values_of_one_document_are_two_rules_of_that_document() {
    let mut index = RulesIndex::new(plain());
    let mut document = TantivyDocument::new();
    document.add_u64(index.id, 1);
    document.add_text(index.rule, "sofa, couch");
    document.add_text(index.rule, "sofa => settee");
    index.writer.add_document(document).unwrap();
    index.add(2, "red, crimson");
    index.commit();
    assert_eq!(
        index.alternatives(&["sofa"]),
        Some(lists(&[&["couch"], &["settee"], &["sofa"]]))
    );
    assert_eq!(
        index.alternatives(&["couch"]),
        Some(lists(&[&["couch"], &["sofa"]]))
    );
    index.delete(1);
    index.commit();
    assert_eq!(index.alternatives(&["sofa"]), None);
    assert!(index.alternatives(&["red"]).is_some());
}

#[test]
fn the_caller_check_drops_a_rule_in_any_segment() {
    let mut index = RulesIndex::new(plain());
    index.add(1, "running shoes, sneakers");
    index.add(2, "shoes, footwear");
    index.commit();
    index.add(3, "shoes => boots");
    index.add(4, "shoes, footwear");
    index.commit();
    let searcher = index.searcher();
    assert_eq!(searcher.segment_readers().len(), 2);
    let ids: Vec<_> = searcher
        .segment_readers()
        .iter()
        .map(|segment_reader| segment_reader.fast_fields().u64("id").unwrap())
        .collect();
    let expander = SynonymExpander::new(searcher.segment_readers(), index.rule, plain()).unwrap();
    let expansions = expander
        .expand_with(&strings(&["running", "shoes"]), |segment, doc| {
            ids[segment].first(doc) != Some(1)
        })
        .unwrap();
    assert_eq!(
        expansions,
        vec![
            expanded(&["running"], None),
            expanded(&["shoes"], Some(&[&["boots"], &["footwear"], &["shoes"]])),
        ]
    );
}

#[test]
fn a_merge_keeps_the_alive_rules() {
    let mut index = RulesIndex::new(plain());
    index.add(1, "couch, sofa");
    index.add(2, "sneakers => running shoes");
    index.commit();
    index.add(3, "red, crimson");
    index.commit();
    // An update is a delete and an add of the same id.
    index.delete(2);
    index.add(2, "sneakers => trainers");
    index.delete(3);
    index.commit();
    index.merge_all();
    assert_eq!(index.searcher().segment_readers().len(), 1);
    assert_eq!(
        index.alternatives(&["sneakers"]),
        Some(lists(&[&["trainers"]]))
    );
    assert_eq!(index.alternatives(&["red"]), None);
    assert_eq!(
        index.alternatives(&["sofa"]),
        Some(lists(&[&["couch"], &["sofa"]]))
    );
}

#[test]
fn an_invalid_rule_indexes_no_key_and_the_writer_goes_on() {
    let mut index = RulesIndex::new(plain());
    index.add(1, "a => b => c");
    index.add(2, "sofa, couch");
    index.commit();
    assert_eq!(index.alternatives(&["a"]), None);
    assert!(index.alternatives(&["sofa"]).is_some());
}

#[test]
fn the_rule_field_is_indexed_and_fast_with_raw() {
    for options in [TEXT, TEXT.set_fast("default")] {
        let mut schema_builder = Schema::builder();
        let rule = schema_builder.add_text_field("rule", options);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();
        writer.add_document(doc!(rule => "sofa, couch")).unwrap();
        writer.commit().unwrap();
        let searcher = index.reader().unwrap().searcher();
        let err = SynonymExpander::new(searcher.segment_readers(), rule, plain()).unwrap_err();
        assert!(matches!(err, TantivyError::SchemaError(_)), "{err:?}");
    }
}

#[test]
fn token_rules_returns_only_the_keys_that_start_with_the_token() {
    let index = RulesIndex::with_rules(
        plain(),
        &[
            "sofa, couch",
            "sofas => settees",
            "sofa bed => futon",
            "sof => x",
        ],
    );
    let searcher = index.searcher();
    let segment = SegmentSynonyms::open(&searcher.segment_readers()[0], index.rule)
        .unwrap()
        .unwrap();
    let keys: Vec<_> = segment
        .token_rules("sofa", &mut plain())
        .unwrap()
        .into_iter()
        .map(|key_rules| (key_rules.key, key_rules.rules))
        .collect();
    let rule = |doc, alternatives: &[&[&str]]| RuleAlternatives {
        doc,
        alternatives: lists(alternatives),
    };
    assert_eq!(
        keys,
        vec![
            (strings(&["sofa"]), vec![rule(0, &[&["sofa"], &["couch"]])]),
            (strings(&["sofa", "bed"]), vec![rule(2, &[&["futon"]])]),
        ]
    );
}

/// Keeps what each segment holds for a token across expansions, and asks for liveness each time.
struct CachingSource {
    segments: Vec<SegmentSynonyms>,
    analyzer: TextAnalyzer,
    cache: HashMap<(usize, String), Arc<[KeyRules]>>,
    fetches: usize,
    deleted: HashSet<DocId>,
}

impl SynonymSource for CachingSource {
    fn num_segments(&self) -> usize {
        self.segments.len()
    }

    fn token_rules(&mut self, segment: usize, token: &str) -> crate::Result<Arc<[KeyRules]>> {
        let key = (segment, token.to_string());
        if let Some(rules) = self.cache.get(&key) {
            return Ok(rules.clone());
        }
        self.fetches += 1;
        let rules: Arc<[KeyRules]> = self.segments[segment]
            .token_rules(token, &mut self.analyzer)?
            .into();
        self.cache.insert(key, rules.clone());
        Ok(rules)
    }

    fn is_alive(&mut self, _segment: usize, doc: DocId) -> bool {
        !self.deleted.contains(&doc)
    }
}

#[test]
fn a_cached_lookup_still_drops_a_rule_deleted_after_it() {
    let index = RulesIndex::with_rules(plain(), &["running shoes, sneakers", "shoes, footwear"]);
    let searcher = index.searcher();
    let mut source = CachingSource {
        segments: searcher
            .segment_readers()
            .iter()
            .filter_map(|segment_reader| SegmentSynonyms::open(segment_reader, index.rule).unwrap())
            .collect(),
        analyzer: plain(),
        cache: HashMap::new(),
        fetches: 0,
        deleted: HashSet::new(),
    };
    let tokens = strings(&["running", "shoes"]);
    assert_eq!(
        expand(&tokens, &mut source).unwrap(),
        index.expand(&["running", "shoes"])
    );
    let fetches = source.fetches;
    source.deleted.insert(0);
    // Only `shoes` is new, since the first expansion matched both tokens at once.
    assert_eq!(
        expand(&tokens, &mut source).unwrap(),
        vec![
            expanded(&["running"], None),
            expanded(&["shoes"], Some(&[&["footwear"], &["shoes"]])),
        ]
    );
    assert_eq!(source.fetches, fetches + 1);
}
