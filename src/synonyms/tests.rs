//! Tests of the synonym component across writes, deletes and merges.

use std::collections::BTreeMap;

use super::*;
use crate::directory::Directory;
use crate::index::{IndexSettings, IndexSortByField, Order, SegmentComponent};
use crate::indexer::operation::AddOperation;
use crate::indexer::{NoMergePolicy, SegmentWriter};
use crate::schema::{
    Field, Schema, SchemaBuilder, TextFieldIndexing, TextOptions, FAST, INDEXED, STORED, TEXT,
};
use crate::tokenizer::{LowerCaser, SimpleTokenizer, TextAnalyzer};
use crate::{doc, Index, IndexWriter, TantivyDocument, TantivyError, Term};

const RULES_TOKENIZER: &str = "rules";

/// Each key with the id and the alternatives of each alive rule.
type AliveKeys = BTreeMap<Vec<u8>, Vec<(u64, Vec<Vec<String>>)>>;

struct RulesIndex {
    index: Index,
    writer: IndexWriter,
    id: Field,
    rule: Field,
}

impl RulesIndex {
    fn new(analyzer: TextAnalyzer) -> Self {
        Self::with_settings(analyzer, IndexSettings::default())
    }

    fn with_settings(analyzer: TextAnalyzer, settings: IndexSettings) -> Self {
        let mut schema_builder = Schema::builder();
        let id = schema_builder.add_u64_field("id", FAST | INDEXED);
        let rule = schema_builder.add_text_field(
            "rule",
            TextOptions::default()
                .set_indexing_options(TextFieldIndexing::default().set_tokenizer(RULES_TOKENIZER))
                .set_synonym_rules(),
        );
        let index = Index::builder()
            .schema(schema_builder.build())
            .settings(settings)
            .create_in_ram()
            .unwrap();
        index.tokenizers().register(RULES_TOKENIZER, analyzer);
        let writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();
        writer.set_merge_policy(Box::new(NoMergePolicy));
        RulesIndex {
            index,
            writer,
            id,
            rule,
        }
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

    fn expander(&self) -> SynonymExpander {
        let searcher = self.index.reader().unwrap().searcher();
        SynonymExpander::new(searcher.segment_readers()).unwrap()
    }

    fn num_segments(&self) -> usize {
        self.index.searchable_segment_ids().unwrap().len()
    }

    /// The keys of every segment, sorted, so two indexes with the same alive rules compare equal
    /// however their documents are laid out.
    fn alive_keys(&self) -> AliveKeys {
        let searcher = self.index.reader().unwrap().searcher();
        let mut keys = AliveKeys::new();
        for segment_reader in searcher.segment_readers() {
            let ids = segment_reader.fast_fields().u64("id").unwrap();
            let dictionary = segment_reader.synonym_dictionary().unwrap().unwrap();
            let mut reader = dictionary.read_keys().unwrap();
            while reader.advance().unwrap() {
                for group in reader.groups().unwrap() {
                    if !segment_reader.is_deleted(group.doc) {
                        let id = ids.first(group.doc).unwrap();
                        let alternatives = dictionary.alternatives(&group).unwrap();
                        let entry = keys.entry(reader.key().to_vec()).or_default();
                        entry.push((id, alternatives));
                        entry.sort();
                    }
                }
            }
        }
        keys
    }

    fn dictionary(&self) -> SynonymDictionary {
        let searcher = self.index.reader().unwrap().searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        searcher.segment_readers()[0]
            .synonym_dictionary()
            .unwrap()
            .unwrap()
    }
}

fn plain() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .build()
}

/// The alternatives of a query that is exactly one key, or `None` when another expansion wins.
fn alternatives(expander: &SynonymExpander, key: &[&str]) -> Option<Vec<Vec<String>>> {
    let key = strings(key);
    match expander.expand(&key).unwrap().as_slice() {
        [expansion] if expansion.tokens == key => expansion.alternatives.clone(),
        _ => None,
    }
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
        alternatives(&index.expander(), &["sofa"]),
        Some(vec![
            strings(&["sofa"]),
            strings(&["couch"]),
            strings(&["settee"])
        ])
    );
    let rules = index
        .dictionary()
        .get(&strings(&["sofa"]))
        .unwrap()
        .unwrap();
    assert_eq!(
        rules.iter().map(|rule| rule.doc).collect::<Vec<_>>(),
        vec![0, 0]
    );
    index.delete(1);
    index.commit();
    let expander = index.expander();
    assert_eq!(alternatives(&expander, &["sofa"]), None);
    assert!(alternatives(&expander, &["red"]).is_some());
}

#[test]
fn a_key_is_looked_up_in_every_segment() {
    let mut index = RulesIndex::new(plain());
    index.add(1, "sofa, couch");
    index.commit();
    index.add(2, "sofa => settee");
    index.add(3, "running shoes, sneakers");
    index.commit();
    assert_eq!(index.num_segments(), 2);
    let mut expansions = index
        .expander()
        .expand(&strings(&["sofa", "running", "shoes"]))
        .unwrap();
    // The searcher lists segments in its own order, and the alternatives follow it.
    for expansion in &mut expansions {
        expansion.alternatives.as_mut().unwrap().sort();
    }
    assert_eq!(
        expansions,
        vec![
            expanded(&["sofa"], Some(&[&["couch"], &["settee"], &["sofa"]])),
            expanded(
                &["running", "shoes"],
                Some(&[&["running", "shoes"], &["sneakers"]])
            ),
        ]
    );
}

#[test]
fn a_deleted_rule_stops_expanding_before_a_merge() {
    let mut index = RulesIndex::new(plain());
    index.add(1, "running shoes, sneakers");
    index.add(2, "shoes, footwear");
    index.commit();
    index.delete(1);
    index.commit();
    assert_eq!(
        index
            .expander()
            .expand(&strings(&["running", "shoes"]))
            .unwrap(),
        vec![
            expanded(&["running"], None),
            expanded(&["shoes"], Some(&[&["shoes"], &["footwear"]])),
        ]
    );
}

#[test]
fn a_merge_matches_a_fresh_build_of_the_same_rules() {
    let mut merged = RulesIndex::new(plain());
    merged.add(1, "couch, sofa, divan");
    merged.add(2, "sneakers => running shoes");
    merged.commit();
    merged.add(3, "running shoes, sneakers");
    merged.add(4, "red, crimson");
    merged.commit();
    merged.add(5, "tv, television");
    merged.commit();
    // An update is a delete and an add of the same id.
    merged.delete(2);
    merged.add(2, "sneakers => trainers");
    merged.delete(4);
    merged.commit();
    assert_eq!(merged.num_segments(), 4);
    let before = merged.alive_keys();
    merged.merge_all();
    assert_eq!(merged.num_segments(), 1);

    let mut fresh = RulesIndex::new(plain());
    fresh.add(1, "couch, sofa, divan");
    fresh.add(3, "running shoes, sneakers");
    fresh.add(5, "tv, television");
    fresh.add(2, "sneakers => trainers");
    fresh.commit();
    assert_eq!(merged.alive_keys(), fresh.alive_keys());
    assert_eq!(merged.alive_keys(), before);

    let dictionary = merged.dictionary();
    // The deleted rule's keys are gone from the merged file, not only masked.
    assert_eq!(dictionary.get(&strings(&["red"])).unwrap(), None);
    assert_eq!(dictionary.max_key_tokens(), 2);
    assert_eq!(
        merged.expander().expand(&strings(&["sneakers"])).unwrap(),
        vec![expanded(
            &["sneakers"],
            Some(&[&["running", "shoes"], &["sneakers"], &["trainers"]])
        )]
    );
}

#[test]
fn a_merge_of_a_sorted_index_keeps_each_rule_on_its_document() {
    let settings = || IndexSettings {
        sort_by_field: Some(IndexSortByField {
            field: "id".to_string(),
            order: Order::Desc,
        }),
        ..Default::default()
    };
    let mut merged = RulesIndex::with_settings(plain(), settings());
    merged.add(1, "sofa, couch");
    merged.add(4, "red, crimson");
    merged.commit();
    merged.add(2, "tv, television");
    merged.add(5, "sofa => settee");
    merged.add(3, "big, large");
    merged.commit();
    merged.delete(4);
    merged.delete(2);
    merged.commit();
    merged.merge_all();
    assert_eq!(merged.num_segments(), 1);

    let mut fresh = RulesIndex::with_settings(plain(), settings());
    fresh.add(1, "sofa, couch");
    fresh.add(5, "sofa => settee");
    fresh.add(3, "big, large");
    fresh.commit();
    assert_eq!(merged.alive_keys(), fresh.alive_keys());
}

#[test]
fn an_index_without_rules_has_no_synonym_component() {
    let mut schema_builder = Schema::builder();
    let text = schema_builder.add_text_field("text", TEXT);
    let index = Index::create_in_ram(schema_builder.build());
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();
    writer.add_document(doc!(text => "couch, sofa")).unwrap();
    writer.commit().unwrap();
    writer.add_document(doc!(text => "red, crimson")).unwrap();
    writer.commit().unwrap();
    writer
        .merge(&index.searchable_segment_ids().unwrap())
        .wait()
        .unwrap();
    let searcher = index.reader().unwrap().searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    let segment_reader = &searcher.segment_readers()[0];
    assert!(segment_reader.synonym_dictionary().unwrap().is_none());
    let segment = &index.searchable_segments().unwrap()[0];
    let path = segment.relative_path(SegmentComponent::Custom(SYNONYM_EXT.to_string()));
    assert!(!index.directory().exists(&path).unwrap());
    assert_eq!(
        SynonymExpander::new(searcher.segment_readers())
            .unwrap()
            .expand(&strings(&["sofa"]))
            .unwrap(),
        vec![expanded(&["sofa"], None)]
    );
}

fn add_to_new_segment(index: &Index, document: TantivyDocument) -> crate::Result<()> {
    let mut segment_writer = SegmentWriter::for_segment(15_000_000, index.new_segment(), false)?;
    segment_writer.add_document(AddOperation {
        opstamp: 0,
        document,
    })
}

#[test]
fn an_invalid_rule_fails_its_document() {
    let index = RulesIndex::new(plain());
    let err = add_to_new_segment(&index.index, doc!(index.rule => "a => b => c")).unwrap_err();
    assert!(
        matches!(&err, TantivyError::InvalidArgument(msg)
            if msg == "invalid synonym rule 'a => b => c': a rule can have at most one '=>'"),
        "{err:?}"
    );
}

#[test]
fn a_missing_synonym_component_is_an_error() {
    let mut index = RulesIndex::new(plain());
    index.add(1, "sofa, couch");
    index.commit();
    let segment = &index.index.searchable_segments().unwrap()[0];
    let path = segment.relative_path(SegmentComponent::Custom(SYNONYM_EXT.to_string()));
    index.index.directory().delete(&path).unwrap();
    let searcher = index.index.reader().unwrap().searcher();
    assert!(SynonymExpander::new(searcher.segment_readers()).is_err());
}

#[test]
fn a_schema_holds_rules_in_one_indexed_field() {
    let create = |schema_builder: SchemaBuilder| {
        Index::builder()
            .schema(schema_builder.build())
            .create_in_ram()
            .map(drop)
    };
    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("first", TEXT.set_synonym_rules());
    schema_builder.add_text_field("second", TEXT.set_synonym_rules());
    let err = create(schema_builder).unwrap_err();
    assert!(matches!(err, TantivyError::SchemaError(_)), "{err:?}");

    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("rule", TextOptions::from(STORED).set_synonym_rules());
    let err = create(schema_builder).unwrap_err();
    assert!(matches!(err, TantivyError::SchemaError(_)), "{err:?}");
}

#[test]
fn the_synonym_rules_flag_is_left_out_of_a_schema_without_it() {
    let json = serde_json::to_value(TEXT).unwrap();
    assert!(json.get("synonym_rules").is_none());
    assert!(!serde_json::from_value::<TextOptions>(json)
        .unwrap()
        .holds_synonym_rules());
}
