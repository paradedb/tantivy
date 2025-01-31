use std::fmt;
use std::sync::Arc;

use common::JsonPathWriter;
// use explode::explode_document;

use crate::core::searcher::Searcher;
use crate::query::{
    BooleanQuery, EnableScoring, Explanation, Occur, ParentBitSetProducer, Query, QueryClone,
    ScoreMode, Scorer, TermQuery, ToParentBlockJoinQuery, Weight,
};
use crate::schema::{Field, IndexRecordOption, Term};
use crate::{DocAddress, DocId, DocSet, Score, SegmentReader, TantivyError, TERMINATED};

pub struct NestedQuery {
    path: Vec<String>,
    child_query: Box<dyn Query>,
    score_mode: ScoreMode,
    ignore_unmapped: bool,
}

#[allow(unused)]
impl NestedQuery {
    pub fn new(
        path: Vec<String>,
        child_query: Box<dyn Query>,
        score_mode: ScoreMode,
        ignore_unmapped: bool,
    ) -> Self {
        Self {
            path,
            child_query,
            score_mode,
            ignore_unmapped,
        }
    }

    pub fn path(&self) -> &Vec<String> {
        &self.path
    }
    pub fn child_query(&self) -> &dyn Query {
        self.child_query.as_ref()
    }
    pub fn score_mode(&self) -> ScoreMode {
        self.score_mode
    }
    pub fn ignore_unmapped(&self) -> bool {
        self.ignore_unmapped
    }
}

impl fmt::Debug for NestedQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NestedQuery")
            .field("path", &self.path)
            .field("score_mode", &self.score_mode)
            .field("ignore_unmapped", &self.ignore_unmapped)
            .finish()
    }
}

impl QueryClone for NestedQuery {
    fn box_clone(&self) -> Box<dyn Query> {
        Box::new(NestedQuery {
            path: self.path.clone(),
            child_query: self.child_query.box_clone(),
            score_mode: self.score_mode,
            ignore_unmapped: self.ignore_unmapped,
        })
    }
}

impl Query for NestedQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let mut path_builder = JsonPathWriter::new();
        for seg in &self.path {
            path_builder.push(seg);
        }

        let path_str = path_builder.as_str();
        let parent_flag_name = format!("_is_parent_{}", path_str);

        let schema = enable_scoring.schema();
        let parent_field = match schema.get_field(&parent_flag_name) {
            Ok(f) => f,
            Err(_) if self.ignore_unmapped => {
                return Ok(Box::new(NoMatchWeight));
            }
            Err(_) => {
                return Err(TantivyError::SchemaError(format!(
                    "NestedQuery path '{}' not mapped, and ignore_unmapped=false",
                    self.path.join(".")
                )));
            }
        };

        let exclude_parent_term = Term::from_field_bool(parent_field, true);
        let exclude_parent_q =
            TermQuery::new(exclude_parent_term.clone(), IndexRecordOption::Basic);

        let child_plus_exclude = BooleanQuery::new(vec![
            (Occur::Must, self.child_query.box_clone()),
            (Occur::MustNot, Box::new(exclude_parent_q)),
        ]);

        let block_join_query = ToParentBlockJoinQuery::new(
            Box::new(child_plus_exclude),
            Arc::new(NestedParentBitSetProducer::new(parent_field)),
            self.score_mode,
        );

        let weight = block_join_query.weight(enable_scoring)?;
        Ok(weight)
    }

    fn explain(&self, searcher: &Searcher, doc_address: DocAddress) -> crate::Result<Explanation> {
        let w = self.weight(EnableScoring::enabled_from_searcher(searcher))?;
        let explanation = w.explain(
            searcher.segment_reader(doc_address.segment_ord),
            doc_address.doc_id,
        )?;
        Ok(explanation)
    }

    fn count(&self, searcher: &Searcher) -> crate::Result<usize> {
        let w = self.weight(EnableScoring::disabled_from_searcher(searcher))?;
        let mut sum = 0usize;
        for seg_reader in searcher.segment_readers() {
            let seg_count = w.count(seg_reader)? as usize;
            sum += seg_count;
        }
        Ok(sum)
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        self.child_query.query_terms(visitor);
    }
}

pub struct NoMatchWeight;

impl Weight for NoMatchWeight {
    fn scorer(&self, _reader: &SegmentReader, _boost: Score) -> crate::Result<Box<dyn Scorer>> {
        Ok(Box::new(NoMatchScorer))
    }
    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> crate::Result<Explanation> {
        Ok(Explanation::new("No-match query", 0.0))
    }
    fn count(&self, _reader: &SegmentReader) -> crate::Result<u32> {
        Ok(0)
    }
    fn for_each_pruning(
        &self,
        _threshold: Score,
        _reader: &SegmentReader,
        _callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> crate::Result<()> {
        Ok(())
    }
}

pub struct NoMatchScorer;

impl crate::docset::DocSet for NoMatchScorer {
    fn advance(&mut self) -> DocId {
        TERMINATED
    }
    fn doc(&self) -> DocId {
        TERMINATED
    }
    fn size_hint(&self) -> u32 {
        0
    }
}

impl Scorer for NoMatchScorer {
    fn score(&mut self) -> Score {
        0.0
    }
}

pub struct NestedParentBitSetProducer {
    parent_field: Field,
}

impl NestedParentBitSetProducer {
    pub fn new(parent_field: Field) -> Self {
        Self { parent_field }
    }
}

impl ParentBitSetProducer for NestedParentBitSetProducer {
    fn produce(&self, reader: &SegmentReader) -> crate::Result<common::BitSet> {
        let max_doc = reader.max_doc();
        let mut bitset = common::BitSet::with_max_value(max_doc);

        let inverted = reader.inverted_index(self.parent_field)?;
        let term_true = Term::from_field_bool(self.parent_field, true);

        if let Some(mut postings) = inverted.read_postings(&term_true, IndexRecordOption::Basic)? {
            while postings.doc() != TERMINATED {
                bitset.insert(postings.doc());
                postings.advance();
            }
        }

        Ok(bitset)
    }
}

#[cfg(test)]
mod tests {
    use crate::collector::TopDocs;
    use crate::query::nested_query::explode::explode;
    use crate::query::{BooleanQuery, NestedQuery, QueryParser, ScoreMode};
    use crate::schema::{
        IndexRecordOption, JsonObjectOptions, NumericOptions, Schema, SchemaBuilder,
        TextFieldIndexing,
    };
    use crate::tokenizer::SimpleTokenizer;
    use crate::{doc, Index, IndexWriter, TantivyDocument};
    use serde_json::json;

    // Bring in the "explode" function from your `splode` module.
    // (You might adjust the import path depending on where `splode` is defined.

    /// A small helper that uses your `splode::explode(...)` function to produce `TantivyDocument`s.
    ///
    /// - `json_val`: The raw JSON value to explode.
    /// - `path`: The path of keys leading to nested objects (if top-level is nested, pass `&["driver_json".into()]`, etc.).
    /// - `json_field`: The `Field` in tantivy's schema used to store this JSON.
    /// - `schema`: The tantivy `Schema` (used if you need to get `_is_parent_<path>` fields).
    /// - `opts`: The `JsonObjectOptions` describing how to handle nesting, indexing, etc.
    ///
    /// Returns a vector of `TantivyDocument` that you can index via `IndexWriter::add_documents(...)`.

    #[test]
    fn test_regular_json_indexing() -> crate::Result<()> {
        let driver_json_options = JsonObjectOptions::default().set_indexing_options(
            TextFieldIndexing::default().set_index_option(IndexRecordOption::WithFreqsAndPositions),
        );

        let mut schema_builder = SchemaBuilder::default();
        let driver_field = schema_builder.add_json_field("driver", driver_json_options);

        let schema: Schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut writer = index.writer(50_000_000)?;

        let doc_json = json!({
            "vehicle": [
                { "make": "Powell Motors", "model": "Canyonero" },
                { "make": "Miller-Meteor", "model": "Ecto-1" }
            ],
            "last_name": "McQueen"
        });
        writer.add_document(doc! {
            driver_field => doc_json
        })?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&index, vec![]);
        let query = query_parser.parse_query("driver.vehicle.make:Powell")?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;

        assert_eq!(1, top_docs.len(), "We expect exactly 1 doc to match.");

        let query = query_parser.parse_query("driver.vehicle.model:Canyonero")?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        assert_eq!(1, top_docs.len(), "We expect exactly 1 doc to match.");

        let query2 = query_parser.parse_query("driver.vehicle.model:Canyonero")?;
        let top_docs2 = searcher.search(&query2, &TopDocs::with_limit(10))?;
        assert_eq!(
            1,
            top_docs2.len(),
            "Should match the same doc via 'Canyonero'."
        );

        let query3 = query_parser.parse_query("driver.last_name:McQueen")?;
        let top_docs3 = searcher.search(&query3, &TopDocs::with_limit(10))?;
        assert_eq!(
            1,
            top_docs3.len(),
            "Should match the doc via 'McQueen' too."
        );

        Ok(())
    }

    #[test]
    fn test_multi_level_nested_query() -> crate::Result<()> {
        use crate::{
            collector::TopDocs,
            index::Index,
            query::QueryParser,
            schema::{IndexRecordOption, JsonObjectOptions, SchemaBuilder, TextFieldIndexing},
            TantivyDocument,
        };
        use serde_json::json;

        // 1) Define schema: a top-level "driver" is nested with a nested subfield "vehicle".
        let mut schema_builder = SchemaBuilder::new();

        // We'll index text in "raw"/untokenized mode, or you could pick a standard tokenizer.
        let indexing_opts = TextFieldIndexing::default()
            .set_tokenizer("raw")
            .set_index_option(IndexRecordOption::Basic);

        let mut driver_opts =
            JsonObjectOptions::nested().set_indexing_options(indexing_opts.clone());
        driver_opts.subfields.insert(
            "vehicle".to_string(),
            JsonObjectOptions::nested().set_indexing_options(indexing_opts),
        );

        // This will automatically add internal `_is_parent_driver` or `_is_parent_driver\u{1}vehicle` fields.
        let driver_field = schema_builder.add_json_field_auto_nested("driver", driver_opts.clone());

        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());

        let mut writer = index.writer(50_000_000)?;

        // 2) Our JSON doc has an array of vehicles under driver
        let doc_json = json!({
            "vehicle": [
                { "make": "Powell Motors", "model": "Canyonero" },
                { "make": "Miller-Meteor", "model": "Ecto-1" }
            ],
            "last_name": "McQueen"
        });

        // 3) Explode + add docs
        let exploded_docs =
            crate::query::nested_query::explode::explode(&[], doc_json, Some(&driver_opts));
        let tantivy_docs: Vec<TantivyDocument> = exploded_docs
            .into_iter()
            .map(|val| {
                TantivyDocument::from_json_object(&schema, val.as_object().unwrap().clone())
                    .unwrap()
            })
            .collect();
        writer.add_documents(tantivy_docs)?;
        writer.commit()?;

        // 4) Basic test: Searching for driver.vehicle.model:Canyonero
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![driver_field]);
        let query = query_parser.parse_query("driver.vehicle.model:Canyonero")?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;

        assert_eq!(
            1,
            top_docs.len(),
            "We expect exactly one doc to match 'Canyonero'"
        );
        Ok(())
    }

    /// Example usage of the new parameter in your test scenario.
    #[test]
    fn test_nested_query_scenario() -> crate::Result<()> {
        use crate::query::nested_query::{NestedQuery, ScoreMode};

        let mut schema_builder = SchemaBuilder::new();

        // "driver_json" top-level is nested => let's do something like "include_in_parent=false".
        let mut driver_json_opts = JsonObjectOptions::default()
            .set_nested()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_index_option(IndexRecordOption::Basic),
            );

        // We also declare a "vehicle" subfield (not nested here for simplicity).
        driver_json_opts.subfields.insert(
            "vehicle".to_string(),
            JsonObjectOptions::default()
                .set_nested()
                .set_indexing_options(
                    TextFieldIndexing::default()
                        .set_tokenizer("raw")
                        .set_index_option(IndexRecordOption::Basic),
                ),
        );

        let driver_field = schema_builder.add_json_field("driver_json", driver_json_opts.clone());
        let bool_opts = NumericOptions::default().set_stored().set_indexed();
        let _is_parent_driver_json =
            schema_builder.add_bool_field("_is_parent_driver_json", bool_opts.clone());
        let _is_parent_driver_json_vehicle =
            schema_builder.add_bool_field("_is_parent_driver_json\u{1}vehicle", bool_opts);

        let schema = schema_builder.build();

        // Create index + writer
        let index = Index::create_in_ram(schema.clone());
        index
            .tokenizers()
            .register("raw", SimpleTokenizer::default());
        let mut writer: IndexWriter<TantivyDocument> =
            index.writer_with_num_threads(2, 50_000_000)?;

        let big_json = json!({
            "driver_json": {
                "last_name": "McQueen",
                "vehicle": [
                    { "make": "Powell Motors", "model": "Canyonero" },
                    { "make": "Miller-Meteor", "model": "Ecto-1" }
                ]
            }
        });

        // Use our new explode_document function
        let exploded_docs = explode(&[], big_json, Some(&driver_json_opts));

        writer.add_documents(
            exploded_docs
                .into_iter()
                .map(|value| {
                    TantivyDocument::from_json_object(&schema, value.as_object().unwrap().clone())
                        .unwrap()
                })
                .collect(),
        )?;
        writer.commit()?;

        // Build a NestedQuery
        let query_parser = QueryParser::for_index(&index, vec![driver_field]);
        let child_q = query_parser.parse_query("driver_json.vehicle.model:Canyonero")?;

        let nested_query = NestedQuery::new(
            vec!["driver_json".to_string()],
            Box::new(child_q),
            ScoreMode::Avg,
            false,
        );

        // Execute search
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;

        // We expect exactly 1 doc: The parent doc with child "Canyonero".
        assert_eq!(top_docs.len(), 1, "Expected exactly one matching doc.");
        Ok(())
    }

    #[test]
    fn test_nested_query_scenario_small() -> crate::Result<()> {
        let mut schema_builder = SchemaBuilder::new();

        let driver_json_opts = JsonObjectOptions::nested()
            .set_indexing_options(TextFieldIndexing::default())
            .add_subfield(
                "driver_json",
                JsonObjectOptions::nested().add_subfield("vehicle", JsonObjectOptions::nested()),
            );

        let driver_field =
            schema_builder.add_json_field_auto_nested("driver_json", driver_json_opts.clone());

        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut writer: IndexWriter<TantivyDocument> =
            index.writer_with_num_threads(2, 50_000_000)?;

        let big_json = json!({
        "driver_json": {
            "last_name": "McQueen",
            "vehicle": [
                { "make": "Powell", "model": "Canyonero" },
                { "make": "Miller-Meteor", "model": "Ecto" }
            ]
        }});

        let exploded_docs = explode(&[], big_json, Some(&driver_json_opts));
        assert_eq!(exploded_docs.len(), 4, "should be 4 exploded docs");

        writer.add_documents(
            exploded_docs
                .into_iter()
                .map(|value| {
                    TantivyDocument::from_json_object(&schema, value.as_object().unwrap().clone())
                        .unwrap()
                })
                .collect(),
        )?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&index, vec![driver_field]);

        // Check for the single parent doc that sets _is_parent_vehicle=true
        let nested_query = query_parser
            .parse_query("_is_parent_driver_json:true")
            .unwrap();
        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 1, "Expected one parent doc");

        // We can still find "driver_json.last_name:McQueen"
        let nested_query = query_parser
            .parse_query("driver_json.last_name:McQueen")
            .unwrap();
        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 1, "Expected to match only parent doc");
        assert_eq!(top_docs[0].1.doc_id, 3, "Returned doc is the parent");

        let nested_query = query_parser
            .parse_query("driver_json.vehicle.model:Canyonero")
            .unwrap();
        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(
            top_docs.len(),
            3,
            "Expected two docs (child + parent + grandparent) "
        );
        assert_eq!(top_docs[0].1.doc_id, 0,);
        assert_eq!(top_docs[1].1.doc_id, 2,);

        let nested_query = query_parser.parse_query("last_name:McQueen").unwrap();
        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 1, "Expected one doc with the parent data.");
        assert_eq!(
            top_docs[0].1.doc_id, 3,
            "Parent doc is the last doc (doc_id=3)"
        );

        let child_q = query_parser
            .parse_query("driver_json.vehicle.model:Canyonero")
            .unwrap();
        let inner_nested_q = NestedQuery::new(
            vec!["driver_json".into(), "vehicle".to_string()],
            Box::new(child_q),
            ScoreMode::Avg,
            false,
        );
        let nested_q = NestedQuery::new(
            vec!["driver_json".into()],
            Box::new(inner_nested_q),
            ScoreMode::Avg,
            false,
        );
        let top_docs = searcher.search(&nested_q, &TopDocs::with_limit(10))?;
        assert_eq!(
            top_docs.len(),
            1,
            "Expected one parent doc from the child match"
        );
        assert_eq!(top_docs[0].1.doc_id, 3, "Returned doc is the parent doc");

        let child_q = query_parser
            .parse_query("driver_json.vehicle.model:Canyonero AND driver_json.vehicle.make:Powell")
            .unwrap();
        let nested_q = NestedQuery::new(
            vec!["driver_json".to_string()],
            Box::new(child_q),
            ScoreMode::Avg,
            false,
        );
        let top_docs = searcher.search(&nested_q, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 1, "Still the same parent doc");
        assert_eq!(top_docs[0].1.doc_id, 3, "Parent doc is #2");

        let child_q = query_parser
            .parse_query("driver_json.vehicle.model:Ecto AND driver_json.vehicle.make:Powell")
            .unwrap();
        let inner_nested_q = NestedQuery::new(
            vec!["driver_json".into(), "vehicle".to_string()],
            Box::new(child_q),
            ScoreMode::Avg,
            false,
        );
        let nested_q = NestedQuery::new(
            vec!["driver_json".to_string()],
            Box::new(inner_nested_q),
            ScoreMode::Avg,
            false,
        );
        let top_docs = searcher.search(&nested_q, &TopDocs::with_limit(10))?;
        assert_eq!(
            top_docs.len(),
            0,
            "No doc matches model:Ecto & make:Powell in the same child"
        );

        Ok(())
    }

    #[test]
    fn test_nested_with_non_nested() -> crate::Result<()> {
        let mut schema_builder = SchemaBuilder::new();

        // The top-level "driver_json" is declared as nested:
        let mut driver_json_opts = JsonObjectOptions::default()
            .set_nested()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_index_option(IndexRecordOption::Basic),
            );
        // We'll define "vehicle" as a nested subfield, but we also have "bicycle" not declared nested
        driver_json_opts
            .subfields
            .insert("vehicle".to_string(), JsonObjectOptions::default());

        let driver_field = schema_builder.add_json_field("driver_json", driver_json_opts.clone());
        let _is_parent_vehicle = schema_builder.add_bool_field(
            "_is_parent_vehicle",
            NumericOptions::default().set_stored().set_indexed(),
        );

        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        index
            .tokenizers()
            .register("raw", SimpleTokenizer::default());
        let mut writer: IndexWriter<TantivyDocument> =
            index.writer_with_num_threads(2, 50_000_000)?;

        let big_json = json!({
            "last_name": "McQueen",
            "bicycle": [
                { "color": "red", "gears": 3 },
                { "color": "green", "gears": 1 }
            ],
            "vehicle": [
                { "make": "Powell", "model": "Canyonero" },
                { "make": "Miller-Meteor", "model": "Ecto" }
            ]
        });

        let exploded_docs = explode(&[], big_json, Some(&driver_json_opts));

        writer.add_documents(
            exploded_docs
                .into_iter()
                .map(|value| {
                    TantivyDocument::from_json_object(&schema, value.as_object().unwrap().clone())
                        .unwrap()
                })
                .collect(),
        )?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&index, vec![driver_field]);

        let query = query_parser
            .parse_query("driver_json.bicycle.color:red")
            .unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 1, "Parent doc with red bicycle");
        assert_eq!(top_docs[0].1.doc_id, 2, "Returned doc is the parent (#2)");

        let query = query_parser
            .parse_query("driver_json.bicycle.color:red AND driver_json.bicycle.gears:1")
            .unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        assert_eq!(
            top_docs.len(),
            0,
            "No single sub-doc has color:red and gears:1"
        );

        let bicycle_query = query_parser
            .parse_query("driver_json.bicycle.color:red")
            .unwrap();
        let child_q = query_parser
            .parse_query("driver_json.vehicle.model:Canyonero AND driver_json.vehicle.make:Ecto")
            .unwrap();

        // We'll combine them in a BooleanQuery
        let bool_query = BooleanQuery::intersection(vec![bicycle_query, Box::new(child_q)]);
        let top_docs = searcher.search(&bool_query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 0, "No single doc matches that combination");

        Ok(())
    }

    #[test]
    fn test_multi_level_nested_scenario() -> crate::Result<()> {
        use crate::query::nested_query::explode::explode;
        use crate::{
            collector::TopDocs,
            index::Index,
            query::{NestedQuery, QueryParser, ScoreMode},
            schema::{IndexRecordOption, JsonObjectOptions, SchemaBuilder, TextFieldIndexing},
            TantivyDocument,
        };
        use serde_json::json;

        // 1) Build a schema with top-level nested + sub-nested fields
        let mut schema_builder = SchemaBuilder::new();

        let indexing_opts = TextFieldIndexing::default()
            .set_tokenizer("raw")
            .set_index_option(IndexRecordOption::Basic);

        // top-level "driver_json" => nested
        let mut driver_json_opts =
            JsonObjectOptions::nested().set_indexing_options(indexing_opts.clone());

        // "crew" is nested
        let mut crew_opts = JsonObjectOptions::nested().set_indexing_options(indexing_opts.clone());

        // "kids" is nested inside "crew"
        let kids_opts = JsonObjectOptions::nested().set_indexing_options(indexing_opts.clone());
        crew_opts.subfields.insert("kids".to_string(), kids_opts);

        // attach "crew" subfield to top-level
        driver_json_opts
            .subfields
            .insert("crew".to_string(), crew_opts);

        // "vehicle" is also nested
        let vehicle_opts = JsonObjectOptions::nested().set_indexing_options(indexing_opts);
        driver_json_opts
            .subfields
            .insert("vehicle".to_string(), vehicle_opts);

        // Add it as a single field "driver_json", auto-nested
        let driver_field =
            schema_builder.add_json_field_auto_nested("driver_json", driver_json_opts.clone());

        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());

        let mut writer = index.writer_with_num_threads(2, 50_000_000)?;

        // 2) Our JSON includes `crew` => array of objects => each with possibly a `kids` array
        let big_json = json!({
            "last_name": "McQueen",
            "crew": [
                {
                    "role": "spotter",
                    "person": "Joe",
                    "kids": [
                        { "name": "Eve", "age": 3 },
                        { "name": "Sam", "age": 5 }
                    ]
                },
                {
                    "role": "mechanic",
                    "person": "Jim",
                    "kids": []
                }
            ],
            "vehicle": [
                { "make": "Powell", "model": "Canyonero" },
                { "make": "Miller-Meteor", "model": "Ecto-1" }
            ]
        });

        // 3) Explode the JSON
        let exploded_docs = explode(&[], big_json, Some(&driver_json_opts));
        let tantivy_docs: Vec<TantivyDocument> = exploded_docs
            .into_iter()
            .map(|val| {
                TantivyDocument::from_json_object(&schema, val.as_object().unwrap().clone())
                    .unwrap()
            })
            .collect();
        writer.add_documents(tantivy_docs)?;
        writer.commit()?;

        // 4) Query them
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&index, vec![driver_field]);

        // (a) Find parents that have a child with kids.name = Sam
        let child_q = query_parser.parse_query("driver_json.crew.kids.name:Sam")?;
        let nested_q = NestedQuery::new(
            vec![
                "driver_json".to_string(),
                "crew".to_string(),
                "kids".to_string(),
            ],
            Box::new(child_q),
            ScoreMode::Avg,
            false,
        );
        let top_docs = searcher.search(&nested_q, &TopDocs::with_limit(10))?;
        assert_eq!(
            1,
            top_docs.len(),
            "Should match the parent doc that has Sam"
        );

        // (b) vehicle model=Ecto-1
        let vehicle_q = query_parser.parse_query("driver_json.vehicle.model:Ecto-1")?;
        let nested_vehicle_q = NestedQuery::new(
            vec!["driver_json".to_string(), "vehicle".to_string()],
            Box::new(vehicle_q),
            ScoreMode::Avg,
            false,
        );
        let top_docs2 = searcher.search(&nested_vehicle_q, &TopDocs::with_limit(10))?;
        assert_eq!(1, top_docs2.len(), "Parent doc that has Ecto-1 vehicle");

        // (c) role=mechanic AND kids.name=Eve => that should not match a single crew child
        let child_q2 = query_parser
            .parse_query("driver_json.crew.role:mechanic AND driver_json.crew.kids.name:Eve")?;
        let nested_q2 = NestedQuery::new(
            vec![
                "driver_json".to_string(),
                "crew".to_string(),
                "kids".to_string(),
            ],
            Box::new(child_q2),
            ScoreMode::Avg,
            false,
        );
        let top_docs3 = searcher.search(&nested_q2, &TopDocs::with_limit(10))?;
        assert_eq!(
            0,
            top_docs3.len(),
            "No single crew entry has both mechanic + kid=Eve"
        );

        // (d) direct top-level last_name:McQueen => normal parse
        let plain_q = query_parser.parse_query("driver_json.last_name:McQueen")?;
        let top_docs_plain = searcher.search(&plain_q, &TopDocs::with_limit(10))?;
        assert_eq!(1, top_docs_plain.len(), "Should match on last_name=McQueen");

        Ok(())
    }
}

mod explode {
    use crate::schema::{JsonObjectOptions, ObjectMappingType};
    use common::JsonPathWriter;
    use serde_json::{Map, Value};

    /// Wrap `value` under the given `path`, producing exactly one doc.
    /// If `path` is empty, returns `[ value ]`.
    /// Otherwise nest `value` inside objects named by the path segments.
    fn wrap_in_path(path: &[&str], value: Value) -> Vec<Value> {
        if path.is_empty() {
            return vec![value];
        }
        let mut current = value;
        for seg in path.iter().rev() {
            let mut obj = Map::new();
            obj.insert(seg.to_string(), current);
            current = Value::Object(obj);
        }
        vec![current]
    }

    /// Create a doc with `_is_parent_<path> = true` and store `full_value` under that same path.
    fn make_parent_doc(path: &[&str], full_value: &Value) -> Value {
        // Build the `_is_parent_<path>` field
        let mut path_writer = JsonPathWriter::new();
        for seg in path {
            path_writer.push(seg);
        }
        let path_str = path_writer.as_str();
        let parent_flag = format!("_is_parent_{path_str}");

        let mut doc_map = Map::new();
        doc_map.insert(parent_flag, Value::Bool(true));

        // Now nest `full_value` under the path segments
        let mut current_map = match full_value {
            Value::Object(ref obj) => obj.clone(),
            other => {
                let mut tmp = Map::new();
                tmp.insert("".to_string(), other.clone());
                tmp
            }
        };
        for seg in path.iter().rev() {
            let mut new_map = Map::new();
            if current_map.len() == 1 && current_map.contains_key("") {
                // rename "" => seg
                if let Some(only_val) = current_map.remove("") {
                    new_map.insert(seg.to_string(), only_val);
                }
            } else {
                new_map.insert(seg.to_string(), Value::Object(current_map));
            }
            current_map = new_map;
        }
        // Merge it all
        for (k, v) in current_map {
            doc_map.insert(k, v);
        }
        Value::Object(doc_map)
    }

    /// Return the subset of subfields that are themselves `Nested`.
    fn nested_subfields<'a>(
        opts: &'a JsonObjectOptions,
        obj: &Map<String, Value>,
    ) -> Vec<(&'a String, &'a JsonObjectOptions)> {
        let mut results = Vec::new();
        for (child_key, child_opts) in &opts.subfields {
            if child_opts.object_mapping_type == ObjectMappingType::Nested {
                // Only relevant if the object actually has this child field
                if obj.contains_key(child_key) {
                    results.push((child_key, child_opts));
                }
            }
        }
        results
    }

    /// Explode the JSON `value` according to `opts` if it's nested.
    ///
    /// **Rules**:
    /// 1) If `opts` is missing or `object_mapping_type != Nested`, produce exactly **one** doc (via `wrap_in_path`).
    /// 2) **Nested array** => one child doc for each array item + one parent doc (unless `path.is_empty()`)  
    /// 3) **Nested object**:  
    ///    - If the object has subfields that are themselves nested, recursively explode them to produce child docs, then produce one parent doc with `_is_parent_<path> = true` for the entire object (unless `path.is_empty()`).  
    ///    - If the object does **not** contain any nested subfields, produce **only** one doc:
    ///       - if `path.is_empty()`, just the object,
    ///       - otherwise a single parent doc with `_is_parent_<path> = true`.
    /// 4) **Nested scalar** => exactly **one** doc (no `_is_parent_...`), even if `path` is non‐empty.
    ///
    pub fn explode(path: &[&str], value: Value, opts: Option<&JsonObjectOptions>) -> Vec<Value> {
        // If not nested => single doc
        let Some(my_opts) = opts else {
            return wrap_in_path(path, value);
        };
        if my_opts.object_mapping_type != ObjectMappingType::Nested {
            return wrap_in_path(path, value);
        }

        match value {
            Value::Array(arr) => {
                // Nested array => child doc per element, plus parent doc for entire array if path nonempty
                let mut docs = Vec::new();
                for elem in &arr {
                    // The user tests want each array item as a single doc, unless that item’s schema is also nested subfields.
                    // But typically "arr" corresponds to e.g. "j": [1,2,{k:v}] with no further subfields,
                    // so we just wrap each item.
                    docs.extend(wrap_in_path(path, elem.clone()));
                }
                if !path.is_empty() {
                    docs.push(make_parent_doc(path, &Value::Array(arr)));
                }
                docs
            }
            Value::Object(obj) => {
                // Possibly sub-nested
                let sub_nests = nested_subfields(my_opts, &obj);
                if sub_nests.is_empty() {
                    // No sub-nested => produce exactly 1 doc.
                    if path.is_empty() {
                        // top-level => just store the object
                        wrap_in_path(path, Value::Object(obj))
                    } else {
                        // produce a doc with `_is_parent_<path> = true`
                        vec![make_parent_doc(path, &Value::Object(obj))]
                    }
                } else {
                    // We do have sub-nested fields => produce child docs from each, then a parent doc
                    let mut docs = Vec::new();
                    for (child_key, child_opts) in sub_nests {
                        if let Some(subval) = obj.get(child_key) {
                            let mut new_path = path.to_vec();
                            new_path.push(child_key);
                            docs.extend(explode(&new_path, subval.clone(), Some(child_opts)));
                        }
                    }
                    // Then produce a parent doc if `path` is non-empty
                    if !path.is_empty() {
                        docs.push(make_parent_doc(path, &Value::Object(obj)));
                    }
                    docs
                }
            }
            scalar => {
                // Nested scalar => user tests want exactly one doc, no `_is_parent_`
                wrap_in_path(path, scalar)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use std::collections::BTreeMap;

        use super::*;
        use crate::schema::{JsonObjectOptions, ObjectMappingType};
        use serde_json::json;

        #[test]
        fn explode_non_nested_empty_object() {
            let path = vec![];
            let value = json!({});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Default,
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!({})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_non_nested_simple_object() {
            let path = vec![];
            let value = json!({"a": 1, "b": "two"});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Default,
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!({"a": 1, "b": "two"})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_non_nested_array() {
            let path = vec![];
            let value = json!([1, 2, 3]);
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Default,
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!([1, 2, 3])];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_empty_object() {
            let path = vec![];
            let value = json!({"root": {}});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: BTreeMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let parent_key = "_is_parent_root";
            let expected = vec![json!({parent_key: true, "root": {}})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_simple_scalar() {
            let path = vec!["field"];
            let value = json!("hello");
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            // No parent key should be added if its a scalar.
            let expected = vec![json!({"field": "hello"})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_simple_object() {
            let path = vec![];
            let value = json!({"root": {"a": 1, "b": "two"}});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: BTreeMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!({ "_is_parent_root": true, "root": {"a": 1, "b": "two"}})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_deep_object() {
            let path = vec![];
            let value = json!({"a": {"b": {"c": 42}}});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: BTreeMap::from_iter([(
                    "a".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!({"_is_parent_a": true, "a": {"b": {"c": 42}}})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_wide_object() {
            let value = serde_json::json!({
                "root": {
                    "a": 1,
                    "b": true,
                    "c": null,
                    "d": 3.14,
                    "e": "test",
                    "f": { "g": 99, "h": { "i": "deep" } },
                    "j": [1, 2, { "k": "v" }]
                }
            });

            // "root" is nested => we do block-join indexing for subfields.
            // Among them, "j" is also declared nested => we want to explode that array.
            let mut subfields_root = std::collections::BTreeMap::new();
            subfields_root.insert(
                "j".to_string(),
                JsonObjectOptions {
                    object_mapping_type: ObjectMappingType::Nested,
                    subfields: std::collections::BTreeMap::new(),
                    ..Default::default()
                },
            );

            let mut top_level_subfields = std::collections::BTreeMap::new();
            top_level_subfields.insert(
                "root".to_string(),
                JsonObjectOptions {
                    object_mapping_type: ObjectMappingType::Nested,
                    subfields: subfields_root,
                    ..Default::default()
                },
            );

            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: top_level_subfields,
                ..Default::default()
            };

            let path: Vec<&str> = vec![];
            let result = explode(&path, value, Some(&opts));

            // Then construct expected vector using these structs
            let expected = vec![
                serde_json::json!({ "root": { "j": 1 } }),
                serde_json::json!({ "root": { "j": 2 } }),
                serde_json::json!({ "root": { "j": { "k": "v" } } }),
                serde_json::json!({
                    "_is_parent_root\u{1}j": true,
                    "root": {
                        "j": [1, 2, { "k": "v" }]
                    }
                }),
                serde_json::json!({
                    "_is_parent_root": true,
                    "root": {
                        "a": 1,
                        "b": true,
                        "c": null,
                        "d": 3.14,
                        "e": "test",
                        "f": { "g": 99, "h": { "i": "deep" } },
                        "j": [1, 2, { "k": "v" }]
                    }
                }),
            ];

            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_simple_array() {
            let path = vec![];
            let value = json!([1, 2, 3]);
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            let expected = vec![json!(1), json!(2), json!(3)];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_array_of_objects() {
            let path = vec![];
            let value = json!({"root": [
                {"a": 1},
                {"b": 2},
                {"c": {"d": 3}}
            ]});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: BTreeMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));
            // We expect 3 child docs plus a parent doc
            let expected = vec![
                json!({"root": {"a": 1}}),
                json!({"root": {"b": 2}}),
                json!({"root": {"c": {"d": 3}}}),
                json!({
                    "_is_parent_root": true,
                    "root": [
                        {"a": 1},
                        {"b": 2},
                        {"c": {"d": 3}}
                    ]
                }),
            ];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_multi_dimensional_arrays() {
            let path = vec![];
            let value = json!({"root": [
                [1, 2],
                [3, [4, 5]],
                [6, {"x": [7, 8]}]
            ]});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: BTreeMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path, value, Some(&opts));

            // Because "root" is the nested subfield,
            // we expect a single parent doc with "_is_parent_root": true,
            // containing the entire array. (No recursion on multi-dimensional arrays.)
            // use serde_json::{json, Value};

            let expected = vec![
                json!({
                    "root": [1, 2]
                }),
                json!({
                    "root": [3, [4, 5]]
                }),
                json!({
                    "root": [6, { "x": [7, 8] }]
                }),
                json!({
                    "_is_parent_root": true,
                    "root": [
                        [1, 2],
                        [3, [4, 5]],
                        [6, { "x": [7, 8] }]
                    ]
                }),
            ];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_mixed_types() {
            // "mixed" is nested, and so are subfields "array" and "letters".
            let mut subfields_mixed = BTreeMap::new();
            subfields_mixed.insert(
                "array".to_string(),
                JsonObjectOptions {
                    object_mapping_type: ObjectMappingType::Nested,
                    ..Default::default()
                },
            );
            subfields_mixed.insert(
                "letters".to_string(),
                JsonObjectOptions {
                    object_mapping_type: ObjectMappingType::Nested,
                    ..Default::default()
                },
            );

            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: subfields_mixed,
                ..Default::default()
            };

            let path = vec!["mixed"];

            let value = serde_json::json!({
                "array": [1, "two", true, null, { "nested": [3, 4] }],
                "obj": { "a": 5, "b": { "c": 6 } },
                "scalar": 7,
                "string": "eight",
                "letters": [
                    { "a": 1 },
                    { "b": 2 },
                    { "c": { "d": 3 } }
                ],
                "bool": false
            });

            let result = explode(&path, value, Some(&opts));

            // Block‐join logic always introduces a parent doc for a “nested” array subfield.
            let expected = vec![
                // Child documents for "array"
                json!({ "mixed": { "array": 1 } }),
                json!({ "mixed": { "array": "two" } }),
                json!({ "mixed": { "array": true } }),
                json!({ "mixed": { "array": null } }),
                json!({ "mixed": { "array": { "nested": [3, 4] } } }),
                // **Additional Object for "array"**
                json!({
                    "_is_parent_mixed\u{1}array": true,
                    "mixed": {
                        "array": [
                            1,
                            "two",
                            true,
                            null,
                            { "nested": [3, 4] }
                        ]
                    }
                }),
                // Child documents for "letters"
                json!({ "mixed": { "letters": { "a": 1 } } }),
                json!({ "mixed": { "letters": { "b": 2 } } }),
                json!({ "mixed": { "letters": { "c": { "d": 3 } } } }),
                // **Additional Object for "letters"**
                json!({
                    "_is_parent_mixed\u{1}letters": true,
                    "mixed": {
                        "letters": [
                            { "a": 1 },
                            { "b": 2 },
                            { "c": { "d": 3 } }
                        ]
                    }
                }),
                // Final parent document
                json!({
                    "_is_parent_mixed": true,
                    "mixed": {
                        "array": [
                            1,
                            "two",
                            true,
                            null,
                            { "nested": [3, 4] }
                        ],
                        "bool": false,
                        "letters": [
                            { "a": 1 },
                            { "b": 2 },
                            { "c": { "d": 3 } }
                        ],
                        "obj": { "a": 5, "b": { "c": 6 } },
                        "scalar": 7,
                        "string": "eight"
                    }
                }),
            ];

            assert_eq!(result, expected);
        }

        #[test]
        fn test_nested_multi_level() {
            // "driver_json" is nested at top-level,
            // "vehicle" is nested subfield of "driver_json".
            let value = json!({
                "driver_json": {
                    "last_name": "McQueen",
                    "vehicle": [
                        {"make": "Powell", "model": "Canyonero"},
                        {"make": "Miller-Meteor", "model": "Ecto-1"}
                    ]
                }
            });

            let mut vehicle_opts = JsonObjectOptions::default();
            vehicle_opts.object_mapping_type = ObjectMappingType::Nested;

            let mut driver_json_opts = JsonObjectOptions::default();
            driver_json_opts.object_mapping_type = ObjectMappingType::Nested;
            driver_json_opts
                .subfields
                .insert("vehicle".to_string(), vehicle_opts);

            let mut top_opts = JsonObjectOptions::default();
            top_opts.object_mapping_type = ObjectMappingType::Nested;
            top_opts
                .subfields
                .insert("driver_json".to_string(), driver_json_opts);

            let docs = explode(&[], value.clone(), Some(&top_opts));

            // Explanation:
            //  - top-level is nested => subfield is "driver_json". The code sees "driver_json" is an object that has subfield "vehicle" also nested.
            //  - So we produce child docs for "driver_json.vehicle" => 2 array items => 2 child docs, then a parent doc for "driver_json.vehicle".
            //  - Then produce the parent doc for "driver_json" (since `path=["driver_json"]` is non-empty from the top-level perspective).
            //  - Because we are at top-level with path = [], there's no `_is_parent_` for that.
            //
            let child1 = json!({
                "driver_json": { "vehicle": { "make": "Powell", "model": "Canyonero" }}
            });
            let child2 = json!({
                "driver_json": { "vehicle": { "make": "Miller-Meteor", "model": "Ecto-1" }}
            });
            let vehicle_parent = json!({
                "_is_parent_driver_json\u{1}vehicle": true,
                "driver_json": {
                    "vehicle": [
                        { "make": "Powell", "model": "Canyonero" },
                        { "make": "Miller-Meteor", "model": "Ecto-1" }
                    ]
                }
            });
            let driver_json_parent = json!({
                "_is_parent_driver_json": true,
                "driver_json": {
                    "last_name": "McQueen",
                    "vehicle": [
                        { "make": "Powell", "model": "Canyonero" },
                        { "make": "Miller-Meteor", "model": "Ecto-1" }
                    ]
                }
            });

            let expected = vec![child1, child2, vehicle_parent, driver_json_parent];
            assert_eq!(docs, expected);
        }
    }
}
