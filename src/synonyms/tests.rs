//! Tests of synonym expansion over an index, across writes, deletes and merges.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::panic;
use std::sync::Arc;

use columnar::Column;

use super::*;
use crate::fastfield::AliveBitSet;
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
            index.add(id as u64, &[rule]);
        }
        index.commit();
        index
    }

    fn add(&mut self, id: u64, rules: &[impl AsRef<str>]) {
        let mut document = TantivyDocument::new();
        document.add_u64(self.id, id);
        for rule in rules {
            document.add_text(self.rule, rule.as_ref());
        }
        self.writer.add_document(document).unwrap();
    }

    fn delete(&mut self, id: u64) {
        self.writer.delete_term(Term::from_field_u64(self.id, id));
    }

    fn commit(&mut self) {
        self.writer.commit().unwrap();
    }

    fn merge_all(&mut self) {
        let segment_ids = self.index.searchable_segment_ids().unwrap();
        if !segment_ids.is_empty() {
            self.writer.merge(&segment_ids).wait().unwrap();
        }
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
    let segment = SegmentSynonyms::open(&searcher.segment_readers()[0], index.rule).unwrap();
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

/// The same sequence on every run, so a failing seed replays.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as usize % n
    }

    fn pick(&mut self, words: &[&'static str]) -> &'static str {
        words[self.below(words.len())]
    }
}

/// Few words, so the keys of different rules overlap and share prefixes.
const WORDS: [&str; 6] = ["a", "b", "c", "d", "e", "f"];

fn random_term(rng: &mut Lcg) -> String {
    let words = 1 + rng.below(2);
    (0..words)
        .map(|_| rng.pick(&WORDS))
        .collect::<Vec<_>>()
        .join(" ")
}

fn random_side(rng: &mut Lcg) -> String {
    let terms = 1 + rng.below(3);
    (0..terms)
        .map(|_| random_term(rng))
        .collect::<Vec<_>>()
        .join(", ")
}

fn random_rule(rng: &mut Lcg) -> String {
    match rng.below(10) {
        0 => "a => b => c".to_string(),
        1..=5 => random_side(rng),
        _ => format!("{} => {}", random_side(rng), random_side(rng)),
    }
}

fn random_query(rng: &mut Lcg) -> Vec<String> {
    let tokens = 1 + rng.below(4);
    (0..tokens)
        .map(|_| rng.pick(&["a", "b", "c", "d", "e", "f", "z"]).to_string())
        .collect()
}

/// The matching rules of [`expand`](fn@expand) in their plainest form. Every key gets the union of
/// its rules' alternatives, and the longest key wins unless its only alternative is itself.
fn model_expand(rules: &BTreeMap<u64, Vec<String>>, tokens: &[String]) -> Vec<Expansion> {
    let mut by_key: HashMap<Vec<String>, BTreeSet<Vec<String>>> = HashMap::new();
    for text in rules.values().flatten() {
        let Ok(Some(rule)) = SynonymRule::parse(text) else {
            continue;
        };
        let Ok(compiled) = rule.compile(&mut plain()) else {
            continue;
        };
        for key in compiled.keys {
            by_key
                .entry(key)
                .or_default()
                .extend(compiled.alternatives.iter().cloned());
        }
    }
    let mut expansions = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let matched = (1..=tokens.len() - i).rev().find_map(|len| {
            let key = &tokens[i..i + len];
            let alternatives = by_key.get(key)?;
            let only_the_key = alternatives.len() == 1 && alternatives.contains(key);
            (!only_the_key).then(|| (len, alternatives.iter().cloned().collect()))
        });
        let len = matched.as_ref().map_or(1, |(len, _)| *len);
        expansions.push(Expansion {
            tokens: tokens[i..i + len].to_vec(),
            alternatives: matched.map(|(_, alternatives)| alternatives),
        });
        i += len;
    }
    expansions
}

struct CachedSegment {
    synonyms: SegmentSynonyms,
    alive: Option<AliveBitSet>,
    ids: Column<u64>,
}

/// Caches what each segment holds for a token, and checks liveness on every call. `deleted` holds
/// the deletes that are not in the bitsets yet.
struct CachingSource {
    segments: Vec<CachedSegment>,
    analyzer: TextAnalyzer,
    cache: HashMap<(usize, String), Arc<[KeyRules]>>,
    deleted: BTreeSet<u64>,
}

impl CachingSource {
    fn new(searcher: &Searcher, rule: Field) -> Self {
        let segments = searcher
            .segment_readers()
            .iter()
            .map(|segment_reader| CachedSegment {
                synonyms: SegmentSynonyms::open(segment_reader, rule).unwrap(),
                alive: segment_reader.alive_bitset().cloned(),
                ids: segment_reader.fast_fields().u64("id").unwrap(),
            })
            .collect();
        CachingSource {
            segments,
            analyzer: plain(),
            cache: HashMap::new(),
            deleted: BTreeSet::new(),
        }
    }
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
        let rules: Arc<[KeyRules]> = self.segments[segment]
            .synonyms
            .token_rules(token, &mut self.analyzer)?
            .into();
        self.cache.insert(key, rules.clone());
        Ok(rules)
    }

    fn is_alive(&mut self, segment: usize, doc: DocId) -> bool {
        let segment = &self.segments[segment];
        let in_bitset = segment
            .alive
            .as_ref()
            .is_none_or(|alive| alive.is_alive(doc));
        in_bitset && !self.deleted.contains(&segment.ids.first(doc).unwrap())
    }
}

fn without(
    rules: &BTreeMap<u64, Vec<String>>,
    deleted: &BTreeSet<u64>,
) -> BTreeMap<u64, Vec<String>> {
    rules
        .iter()
        .filter(|(id, _)| !deleted.contains(id))
        .map(|(id, rules)| (*id, rules.clone()))
        .collect()
}

/// Set it to a seed to run the model test for that seed only.
const MODEL_SEED_VAR: &str = "SYNONYM_MODEL_SEED";

/// The number of seeds a run checks when `MODEL_SEED_VAR` is not set.
const MODEL_SEEDS_PER_RUN: u64 = 16;

/// An expansion must not depend on how the rules are spread over segments, deleted, cached or
/// merged.
#[test]
fn expand_matches_a_plain_model_of_the_rules() {
    let seeds: Vec<u64> = match std::env::var(MODEL_SEED_VAR) {
        Ok(seed) => {
            let seed: u64 = seed
                .parse()
                .unwrap_or_else(|_| panic!("{MODEL_SEED_VAR} must be a number, not {seed:?}"));
            vec![seed]
        }
        // A new start on every run, so the runs together check more cases than a fixed list.
        Err(_) => {
            let start = u64::from(rand::random::<u32>());
            (start..start + MODEL_SEEDS_PER_RUN).collect()
        }
    };
    for seed in seeds {
        if let Err(payload) = panic::catch_unwind(|| check_against_the_model(seed)) {
            let features = if cfg!(feature = "quickwit") {
                " --features quickwit"
            } else {
                ""
            };
            eprintln!(
                "seed {seed} fails. Run it again with:\n{MODEL_SEED_VAR}={seed} cargo test \
                 --lib{features} expand_matches_a_plain_model_of_the_rules"
            );
            panic::resume_unwind(payload);
        }
    }
}

fn check_against_the_model(seed: u64) {
    let mut rng = Lcg(seed);
    let queries: Vec<Vec<String>> = (0..100).map(|_| random_query(&mut rng)).collect();
    let mut index = RulesIndex::new(plain());
    let mut alive: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    let mut next_id = 0;
    for _ in 0..5 {
        for _ in 0..rng.below(8) {
            let rules: Vec<String> = (0..1 + rng.below(2))
                .map(|_| random_rule(&mut rng))
                .collect();
            index.add(next_id, &rules);
            alive.insert(next_id, rules);
            next_id += 1;
        }
        // A delete, or an update, which is a delete and an add of the same id.
        for _ in 0..rng.below(3) {
            let Some(&id) = alive.keys().nth(rng.below(alive.len().max(1))) else {
                break;
            };
            index.delete(id);
            alive.remove(&id);
            if rng.below(2) == 0 {
                let rules = vec![random_rule(&mut rng)];
                index.add(id, &rules);
                alive.insert(id, rules);
            }
        }
        index.commit();
    }
    let searcher = index.searcher();
    let expander = SynonymExpander::new(searcher.segment_readers(), index.rule, plain()).unwrap();
    for query in &queries {
        let expansions = expander.expand(query).unwrap();
        assert_eq!(
            expansions,
            model_expand(&alive, query),
            "with the deletes in the bitsets, query {query:?}"
        );
    }

    let deleted: BTreeSet<u64> = alive
        .keys()
        .copied()
        .filter(|_| rng.below(3) == 0)
        .collect();
    let visible = without(&alive, &deleted);
    let ids: Vec<Column<u64>> = searcher
        .segment_readers()
        .iter()
        .map(|segment_reader| segment_reader.fast_fields().u64("id").unwrap())
        .collect();
    for query in &queries {
        let expansions = expander
            .expand_with(query, |segment, doc| {
                !deleted.contains(&ids[segment].first(doc).unwrap())
            })
            .unwrap();
        assert_eq!(
            expansions,
            model_expand(&visible, query),
            "with the caller check, query {query:?}"
        );
    }

    let mut source = CachingSource::new(&searcher, index.rule);
    source.deleted = deleted;
    for query in &queries {
        let expansions = expand(query, &mut source).unwrap();
        assert_eq!(
            expansions,
            model_expand(&visible, query),
            "with a cold cache, query {query:?}"
        );
    }
    source
        .deleted
        .extend(visible.keys().copied().filter(|_| rng.below(3) == 0));
    let visible = without(&alive, &source.deleted);
    for query in &queries {
        let expansions = expand(query, &mut source).unwrap();
        assert_eq!(
            expansions,
            model_expand(&visible, query),
            "with a warm cache and more deletes, query {query:?}"
        );
    }

    index.merge_all();
    let expander = index.expander();
    for query in &queries {
        let expansions = expander.expand(query).unwrap();
        assert_eq!(
            expansions,
            model_expand(&alive, query),
            "after a merge, query {query:?}"
        );
    }
}
