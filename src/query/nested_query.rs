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
        println!("WEIGHT {parent_flag_name}");

        let schema = enable_scoring.schema();
        let parent_field = match schema.get_field(&parent_flag_name) {
            Ok(f) => f,
            Err(_) if self.ignore_unmapped => {
                return Ok(Box::new(NoMatchWeight));
            }
            Err(_) => {
                return Err(TantivyError::SchemaError(format!(
                    "NestedQuery path '{:?}' not mapped (no field '{}'), and ignore_unmapped=false",
                    self.path, parent_flag_name
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

// #[cfg(test)]
// mod tests {
//     use explode::explode_document;
//     use serde_json::json;

//     use super::*;
//     use crate::collector::TopDocs;
//     use crate::query::nested_query::ScoreMode;
//     use crate::query::QueryParser;
//     use crate::schema::{
//         IndexRecordOption, JsonObjectOptions, NumericOptions, Schema, SchemaBuilder,
//         TextFieldIndexing,
//     };
//     use crate::tokenizer::SimpleTokenizer;
//     use crate::{Index, IndexWriter, TantivyDocument};

//     #[test]
//     fn test_regular_json_indexing() -> crate::Result<()> {
//         let driver_json_options = JsonObjectOptions::default().set_indexing_options(
//             TextFieldIndexing::default().set_index_option(IndexRecordOption::WithFreqsAndPositions),
//         );

//         let mut schema_builder = SchemaBuilder::default();
//         let driver_field = schema_builder.add_json_field("driver", driver_json_options);

//         let schema: Schema = schema_builder.build();

//         let index = Index::create_in_ram(schema.clone());

//         let mut writer = index.writer(50_000_000)?;
//         let doc_json = json!({
//             "vehicle": [
//                 { "make": "Powell Motors", "model": "Canyonero" },
//                 { "make": "Miller-Meteor", "model": "Ecto-1" }
//             ],
//             "last_name": "McQueen"
//         });
//         writer.add_document(doc! {
//             driver_field => doc_json
//         })?;
//         writer.commit()?;

//         let reader = index.reader()?;
//         let searcher = reader.searcher();
//         let query_parser = QueryParser::for_index(&index, vec![]);
//         let query = query_parser.parse_query("driver.vehicle.make:Powell")?;
//         let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;

//         assert_eq!(1, top_docs.len(), "We expect exactly 1 doc to match.");

//         let query_parser = QueryParser::for_index(&index, vec![]);

//         let query = query_parser.parse_query("driver.vehicle.model:Canyonero")?;

//         let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;

//         assert_eq!(1, top_docs.len(), "We expect exactly 1 doc to match.");

//         let query2 = query_parser.parse_query("driver.vehicle.model:Canyonero")?;
//         let top_docs2 = searcher.search(&query2, &TopDocs::with_limit(10))?;
//         assert_eq!(
//             1,
//             top_docs2.len(),
//             "Should match the same doc via 'Canyonero'."
//         );

//         let query3 = query_parser.parse_query("driver.last_name:McQueen")?;
//         let top_docs3 = searcher.search(&query3, &TopDocs::with_limit(10))?;
//         assert_eq!(
//             1,
//             top_docs3.len(),
//             "Should match the doc via 'McQueen' too."
//         );

//         Ok(())
//     }

//     #[test]
//     fn test_multi_level_nested_query() -> crate::Result<()> {
//         let mut schema_builder = SchemaBuilder::default();

//         let driver_json_options = JsonObjectOptions::default()
//             .set_nested(true, false)
//             .set_indexing_options(TextFieldIndexing::default())
//             .add_subfield(
//                 "vehicle",
//                 JsonObjectOptions::default()
//                     .set_nested(true, false)
//                     .set_indexing_options(TextFieldIndexing::default()),
//             );

//         // We'll define them as normal boolean fields in Tantivy.
//         let bool_options = NumericOptions::default().set_stored().set_indexed();
//         let _is_parent_field = schema_builder.add_bool_field("_is_parent_vehicle", bool_options);

//         // Build final schema
//         let schema = schema_builder.build();

//         // 2) Create an in-memory index, plus an IndexWriter.
//         //
//         let index = Index::create_in_ram(schema.clone());
//         let mut writer: IndexWriter<TantivyDocument> = index.writer(50_000_000)?;
//         let big_json = json!({
//             "driver":  {
//                 "last_name": "McQueen",
//                 "vehicle": [
//                     { "make": "Powell Motors", "model": "Canyonero" },
//                     { "make": "Miller-Meteor", "model": "Ecto-1" }
//                 ]
//             },
//         });

//         Ok(())
//     }

//     /// Example usage of the new parameter in your test scenario.

//     #[test]
//     fn test_nested_query_scenario() -> crate::Result<()> {
//         use crate::query::nested_query::{NestedQuery, ScoreMode}; // your NestedQuery
//         use crate::schema::{JsonObjectOptions, NumericOptions, SchemaBuilder};
//         use crate::{collector::TopDocs, Index, IndexWriter};
//         use serde_json::json;

//         // 1) Build a schema with nested JSON.
//         let mut schema_builder = SchemaBuilder::new();

//         let mut driver_json_opts = JsonObjectOptions::default()
//             .set_nested(false, false)
//             .set_indexing_options(
//                 TextFieldIndexing::default()
//                     .set_tokenizer("raw")
//                     .set_index_option(IndexRecordOption::Basic),
//             );
//         driver_json_opts.subfields.insert(
//             "vehicle".to_string(),
//             JsonObjectOptions::default()
//                 .set_nested(false, false)
//                 .set_indexing_options(
//                     TextFieldIndexing::default()
//                         .set_tokenizer("raw")
//                         .set_index_option(IndexRecordOption::Basic),
//                 ),
//         );

//         let driver_field = schema_builder.add_json_field("driver_json", driver_json_opts.clone());

//         let bool_opts = NumericOptions::default().set_stored().set_indexed();
//         let _is_parent_driver_json_field =
//             schema_builder.add_bool_field("_is_parent_driver_json", bool_opts.clone());
//         let _is_parent_driver_json_vehicle_field =
//             schema_builder.add_bool_field("_is_parent_driver_json\u{1}vehicle", bool_opts);

//         let schema = schema_builder.build();

//         // 2) Create index + writer
//         let index = Index::create_in_ram(schema.clone());
//         let mut writer: IndexWriter<TantivyDocument> =
//             index.writer_with_num_threads(2, 50_000_000)?;

//         // 3) Sample nested JSON
//         let big_json = json!({
//             "driver_json": {
//                 "last_name": "McQueen",
//                 "vehicle": [
//                     { "make": "Powell Motors", "model": "Canyonero" },
//                     { "make": "Miller-Meteor", "model": "Ecto-1" }
//                 ]
//             }
//         });

//         // 4) Explode data. Pass `true` for is_top_level.
//         let exploded_docs = explode_document(
//             &big_json["driver_json"],
//             &["driver_json".into()],
//             driver_field,
//             &schema,
//             &driver_json_opts,
//         );

//         for doc in &exploded_docs {
//             println!("EXPLODED DOC: {doc:?}");
//         }

//         // 5) Index them
//         writer.add_documents(exploded_docs)?;
//         writer.commit()?;

//         // 6) Build a NestedQuery.
//         let query_parser = QueryParser::for_index(&index, vec![driver_field]);
//         let child_q = query_parser
//             .parse_query("driver_json.vehicle.model:Canyonero")
//             .unwrap();

//         let nested_query = NestedQuery::new(
//             vec!["driver_json".to_string()],
//             Box::new(child_q),
//             ScoreMode::Avg,
//             false,
//         );

//         // 7) Execute search
//         let reader = index.reader()?;
//         let searcher = reader.searcher();
//         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;

//         // We expect exactly 1 hit: The parent doc with a child "Canyonero".
//         assert_eq!(top_docs.len(), 1, "Expected exactly one matching doc.");

//         Ok(())
//     }

//     #[test]
//     fn test_nested_query_scenario_small() -> crate::Result<()> {
//         // 1) Build a schema with nested JSON.
//         let mut schema_builder = SchemaBuilder::new();

//         // The top-level "driver_json" is declared as nested:
//         let mut driver_json_opts = JsonObjectOptions::default()
//             .set_nested(true, false)
//             .set_indexing_options(
//                 TextFieldIndexing::default()
//                     .set_tokenizer("raw")
//                     .set_index_option(IndexRecordOption::Basic),
//             );

//         driver_json_opts
//             .subfields
//             .insert("vehicle".to_string(), JsonObjectOptions::default());

//         // The main nested-JSON field
//         let driver_field = schema_builder.add_json_field("driver_json", driver_json_opts.clone());

//         // We store two boolean fields: `_is_parent_driver_json` and `_is_parent_driver_json\u{1}vehicle`.
//         // Because our final doc is the "driver_json" parent, it will set `_is_parent_driver_json=true`.
//         // The array doc sets `_is_parent_driver_json\u{1}vehicle=true`.
//         let bool_opts = NumericOptions::default().set_stored().set_indexed();
//         let _is_parent_vehicle =
//             schema_builder.add_bool_field("_is_parent_vehicle", bool_opts.clone());

//         let schema = schema_builder.build();

//         // 2) Create index + writer
//         let index = Index::create_in_ram(schema.clone());
//         // register the "raw" tokenizer if you like:
//         index
//             .tokenizers()
//             .register("raw", SimpleTokenizer::default());
//         let mut writer: IndexWriter<TantivyDocument> =
//             index.writer_with_num_threads(2, 50_000_000)?;

//         // 3) Sample nested JSON
//         let big_json = json!(
//             {
//                 "last_name": "McQueen",
//                 "vehicle": [
//                     { "make": "Powell", "model": "Canyonero" },
//                     { "make": "Miller-Meteor", "model": "Ecto" }
//                 ]
//             }
//         );

//         // 4) Explode data. `true` => top-level => sets `_is_parent_driver_json=true` on the final doc.
//         let exploded_docs = explode_document(
//             &big_json,
//             &["vehicle".into()],
//             driver_field,
//             &schema,
//             &driver_json_opts,
//         );

//         assert_eq!(exploded_docs.len(), 3, "should be 3 exploded docs");

//         // Just to see what we produced:
//         for doc in &exploded_docs {
//             println!("EXPLODED DOC: {doc:?}");
//         }
//         // 5) Index them
//         writer.add_documents(exploded_docs)?;
//         writer.commit()?;

//         // Because the *final* doc sets `_is_parent_driver_json = true`,
//         // we only use a single-level path: ["driver_json"].
//         // 7) Execute search
//         let reader = index.reader()?;
//         let searcher = reader.searcher();

//         let query_parser = QueryParser::for_index(&index, vec![driver_field]);
//         let nested_query = query_parser.parse_query("_is_parent_vehicle:true").unwrap();

//         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
//         assert_eq!(top_docs.len(), 1, "Expected one parent doc");

//         let nested_query = query_parser
//             .parse_query("driver_json.last_name:McQueen")
//             .unwrap();
//         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
//         assert_eq!(top_docs.len(), 1, "Expected that full path matches parent");
//         assert_eq!(top_docs[0].1.doc_id, 2, "Returned doc is the parent");

//         let nested_query = query_parser
//             .parse_query("driver_json.vehicle.model:Canyonero")
//             .unwrap();
//         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
//         assert_eq!(top_docs.len(), 1, "Expected one doc (parent) ");
//         assert_eq!(
//             top_docs[0].1.doc_id, 0,
//             "Returned doc is the child doc without nested query"
//         );

//         let nested_query = query_parser.parse_query("last_name:McQueen").unwrap();
//         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
//         assert_eq!(top_docs.len(), 1, "Expected one doc with mode (parent).");
//         assert_eq!(top_docs[0].1.doc_id, 2, "Expect returned doc to be parent");

//         let child_q = query_parser
//             .parse_query("driver_json.vehicle.model:Canyonero")
//             .unwrap();
//         let nested_query = NestedQuery::new(
//             vec!["vehicle".to_string()],
//             Box::new(child_q),
//             ScoreMode::Avg,
//             false,
//         );
//         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
//         assert_eq!(top_docs.len(), 1, "Expected one child doc");
//         assert_eq!(top_docs[0].1.doc_id, 2, "Returned doc is the parent doc");

//         let child_q = query_parser
//             .parse_query("driver_json.vehicle.model:Canyonero AND driver_json.vehicle.make:Powell")
//             .unwrap();
//         let nested_query = NestedQuery::new(
//             vec!["vehicle".to_string()],
//             Box::new(child_q),
//             ScoreMode::Avg,
//             false,
//         );
//         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
//         assert_eq!(top_docs.len(), 1, "Expected one child doc");
//         assert_eq!(top_docs[0].1.doc_id, 2, "Returned doc is the parent doc");

//         let child_q = query_parser
//             .parse_query("driver_json.vehicle.model:Ecto AND driver_json.vehicle.make:Powell")
//             .unwrap();
//         let nested_query = NestedQuery::new(
//             vec!["vehicle".to_string()],
//             Box::new(child_q),
//             ScoreMode::Avg,
//             false,
//         );
//         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
//         assert_eq!(top_docs.len(), 0, "Expected no child doc");

//         // writer.delete_term(Term::from_field_bool(is_parent_vehicle, true));
//         // writer.commit()?;

//         // reader.reload()?;
//         // let results = reader.searcher().search(&AllQuery, &Count)?;

//         // assert_eq!(results, 0);
//         Ok(())
//     }

//     #[test]
//     fn test_nested_with_non_nested() -> crate::Result<()> {
//         // 1) Build a schema with nested JSON.
//         let mut schema_builder = SchemaBuilder::new();

//         // The top-level "driver_json" is declared as nested:
//         let mut driver_json_opts = JsonObjectOptions::default()
//             .set_nested(true, false)
//             .set_indexing_options(
//                 TextFieldIndexing::default()
//                     .set_tokenizer("raw")
//                     .set_index_option(IndexRecordOption::Basic),
//             );

//         driver_json_opts
//             .subfields
//             .insert("vehicle".to_string(), JsonObjectOptions::default());

//         // The main nested-JSON field
//         let driver_field = schema_builder.add_json_field("driver_json", driver_json_opts.clone());

//         let _is_parent_vehicle = schema_builder.add_bool_field(
//             "_is_parent_vehicle",
//             NumericOptions::default().set_stored().set_indexed(),
//         );

//         let schema = schema_builder.build();
//         let index = Index::create_in_ram(schema.clone());
//         index
//             .tokenizers()
//             .register("raw", SimpleTokenizer::default());
//         let mut writer: IndexWriter<TantivyDocument> =
//             index.writer_with_num_threads(2, 50_000_000)?;

//         let big_json = json!(
//             {
//                 "last_name": "McQueen",
//                 "bicycle": [
//                     { "color": "red", "gears": 3 },
//                     { "color": "green", "gears": 1 },
//                 ],
//                 "vehicle": [
//                     { "make": "Powell", "model": "Canyonero" },
//                     { "make": "Miller-Meteor", "model": "Ecto" }
//                 ]
//             }
//         );

//         // 4) Explode data. `true` => top-level => sets `_is_parent_driver_json=true` on the final doc.
//         let exploded_docs = explode_document(
//             &big_json,
//             &["vehicle".into()],
//             driver_field,
//             &schema,
//             &driver_json_opts,
//         );

//         writer.add_documents(exploded_docs)?;
//         writer.commit()?;

//         let reader = index.reader()?;
//         let searcher = reader.searcher();

//         let query_parser = QueryParser::for_index(&index, vec![driver_field]);

//         let query = query_parser
//             .parse_query("driver_json.bicycle.color:red")
//             .unwrap();
//         let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
//         assert_eq!(top_docs.len(), 1, "Expected that full path matches parent");
//         assert_eq!(top_docs[0].1.doc_id, 2, "Returned doc is the parent");

//         let query = query_parser
//             .parse_query("driver_json.bicycle.color:red AND driver_json.bicycle.gears:1")
//             .unwrap();
//         let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
//         assert_eq!(top_docs.len(), 0, "Expected no match in nested children");

//         let bicycle_query = query_parser
//             .parse_query("driver_json.bicycle.color:red")
//             .unwrap();
//         let nested_query = NestedQuery::new(
//             vec!["vehicle".into()],
//             query_parser
//                 .parse_query("driver_json.vehicle.make:Powell AND model:Ecto")
//                 .unwrap(),
//             ScoreMode::Avg,
//             false,
//         );
//         let bool_query = BooleanQuery::intersection(vec![bicycle_query, Box::new(nested_query)]);
//         let top_docs = searcher.search(&bool_query, &TopDocs::with_limit(10))?;
//         assert_eq!(top_docs.len(), 0, "No paths should match");

//         Ok(())
//     }
// }

// #[test]
// fn test_multi_level_nested_scenario() -> crate::Result<()> {
//     use crate::tokenizer::SimpleTokenizer;
//     use crate::{
//         collector::TopDocs,
//         index::Index,
//         query::{NestedQuery, QueryParser, ScoreMode},
//         schema::{
//             IndexRecordOption, JsonObjectOptions, NumericOptions, SchemaBuilder, TextFieldIndexing,
//         },
//     };
//     use serde_json::json;

//     // ----------------------------------------------------------
//     // 1) Build a schema with multi-level nested JSON: "driver_json" -> "crew" -> "crew.kids", and "vehicle".
//     // ----------------------------------------------------------
//     let mut schema_builder = SchemaBuilder::new();

//     // The top-level "driver_json" is declared as nested.
//     let mut driver_json_opts = JsonObjectOptions::default()
//         .set_nested(true, false)
//         .set_indexing_options(
//             TextFieldIndexing::default()
//                 .set_tokenizer("raw")
//                 .set_index_option(IndexRecordOption::Basic),
//         );

//     // 1a) Define "crew" as a nested subfield
//     let mut crew_opts = JsonObjectOptions::default()
//         .set_nested(true, false)
//         .set_indexing_options(
//             TextFieldIndexing::default()
//                 .set_tokenizer("raw")
//                 .set_index_option(IndexRecordOption::Basic),
//         );

//     // 1b) The "kids" subfield of "crew" is also nested
//     let kids_opts = JsonObjectOptions::default()
//         .set_nested(true, false)
//         .set_indexing_options(
//             TextFieldIndexing::default()
//                 .set_tokenizer("raw")
//                 .set_index_option(IndexRecordOption::Basic),
//         );
//     crew_opts.subfields.insert("kids".into(), kids_opts);

//     // Insert "crew" into "driver_json" subfields
//     driver_json_opts.subfields.insert("crew".into(), crew_opts);

//     // 1c) Define "vehicle" as nested
//     let vehicle_opts = JsonObjectOptions::default()
//         .set_nested(true, false)
//         .set_indexing_options(
//             TextFieldIndexing::default()
//                 .set_tokenizer("raw")
//                 .set_index_option(IndexRecordOption::Basic),
//         );
//     driver_json_opts
//         .subfields
//         .insert("vehicle".into(), vehicle_opts);

//     // Add main JSON field to the schema
//     let driver_field = schema_builder.add_json_field("driver_json", driver_json_opts.clone());

//     // 1d) Also define the boolean fields that label the parent docs.
//     // For multi-level, we might have:
//     //   _is_parent_driver_json
//     //   _is_parent_driver_json.crew
//     //   _is_parent_driver_json.crew.kids
//     //   _is_parent_driver_json.vehicle
//     //
//     // In this simple example, we can just define the fields we know we'll need:
//     let bool_opts = NumericOptions::default().set_stored().set_indexed();
//     let _is_parent_crew =
//         schema_builder.add_bool_field("_is_parent_driver_json.crew", bool_opts.clone());
//     let _is_parent_crew_kids =
//         schema_builder.add_bool_field("_is_parent_driver_json.crew.kids", bool_opts.clone());
//     let _is_parent_vehicle =
//         schema_builder.add_bool_field("_is_parent_driver_json.vehicle", bool_opts.clone());
//     // (We could also define `_is_parent_driver_json` if we wanted.)

//     let schema = schema_builder.build();

//     // ----------------------------------------------------------
//     // 2) Create index + writer
//     // ----------------------------------------------------------
//     let index = Index::create_in_ram(schema.clone());
//     // Register the "raw" tokenizer
//     index
//         .tokenizers()
//         .register("raw", SimpleTokenizer::default());
//     let mut writer = index.writer_with_num_threads(2, 50_000_000)?;

//     // ----------------------------------------------------------
//     // 3) Sample multi-level nested JSON
//     // ----------------------------------------------------------
//     let big_json = json!({
//         "last_name": "McQueen",
//         "crew": [
//           {
//             "role": "spotter",
//             "person": "Joe",
//             "kids": [
//               { "name": "Eve", "age": 3 },
//               { "name": "Sam", "age": 5 }
//             ]
//           },
//           {
//             "role": "mechanic",
//             "person": "Jim",
//             "kids": []
//           }
//         ],
//         "vehicle": [
//           { "make": "Powell", "model": "Canyonero" },
//           { "make": "Miller-Meteor", "model": "Ecto-1" }
//         ]
//     });

//     // ----------------------------------------------------------
//     // 4) Explode data.  This calls your explode_document or explode_document_multi
//     //    In your code, you'd recursively handle "crew" (nested) -> "kids" (nested),
//     //    and "vehicle" (nested) too.
//     // ----------------------------------------------------------
//     let exploded_docs = explode_document(
//         &big_json,
//         &["driver_json".into()], // "full_path"
//         driver_field,
//         &schema,
//         &driver_json_opts,
//     );

//     // Expect multiple children + parent. Let's suppose you produce 1 doc for each crew kid,
//     // plus 1 doc for the crew array, plus 1 doc for the vehicle array, plus final parent, etc.
//     // The exact number depends on your logic; let's just check it isn't empty:
//     assert!(
//         exploded_docs.len() >= 4,
//         "Expected multiple exploded docs for multi-level nested"
//     );

//     // Print for debug
//     for (i, doc) in exploded_docs.iter().enumerate() {
//         println!("Exploded doc #{i} => {doc:?}");
//     }

//     // 5) Index them
//     writer.add_documents(exploded_docs)?;
//     writer.commit()?;

//     // ----------------------------------------------------------
//     // 6) Now we can query them. Let's do a few sample queries:
//     // ----------------------------------------------------------
//     let reader = index.reader()?;
//     let searcher = reader.searcher();

//     // We'll parse queries using the `driver_field` as our default search field.
//     let query_parser = QueryParser::for_index(&index, vec![driver_field]);

//     // 6a) Query for kids named "Sam"
//     let child_q = query_parser.parse_query("driver_json.crew.kids.name:Sam")?;
//     let nested_query = NestedQuery::new(
//         vec!["crew".into(), "kids".into()],
//         Box::new(child_q),
//         ScoreMode::Avg,
//         false,
//     );
//     let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
//     // Expect exactly 1 parent doc (the parent that has a crew member with a kid named "Sam").
//     assert_eq!(top_docs.len(), 1, "Should match the parent with Sam kid");
//     println!("Query for kids.name:Sam => top_docs={top_docs:?}");

//     // 6b) Query for a vehicle with "model:Ecto-1"
//     let child_q2 = query_parser.parse_query("driver_json.vehicle.model:Ecto-1")?;
//     let nested_query2 = NestedQuery::new(
//         vec!["vehicle".into()],
//         Box::new(child_q2),
//         ScoreMode::Max,
//         false,
//     );
//     let top_docs2 = searcher.search(&nested_query2, &TopDocs::with_limit(10))?;
//     assert_eq!(top_docs2.len(), 1, "Parent doc that has Ecto-1 vehicle");
//     println!("Query for vehicle.model:Ecto-1 => top_docs2={top_docs2:?}");

//     // 6c) Query for crew.role:mechanic AND kids.name:Eve (should fail, those are different crew members)
//     let child_q3 = query_parser
//         .parse_query("driver_json.crew.role:mechanic AND driver_json.crew.kids.name:Eve")?;
//     let nested_query3 = NestedQuery::new(
//         vec!["crew".into(), "kids".into()],
//         Box::new(child_q3),
//         ScoreMode::Total,
//         false,
//     );
//     let top_docs3 = searcher.search(&nested_query3, &TopDocs::with_limit(10))?;
//     assert_eq!(
//         top_docs3.len(),
//         0,
//         "No single crew member has role=mechanic and kid=Eve"
//     );
//     println!("Query for role=mechanic & kids.name=Eve => top_docs3={top_docs3:?}");

//     // 6d) Basic test that top-level can still match last_name
//     let plain_query = query_parser.parse_query("driver_json.last_name:McQueen")?;
//     let top_docs_plain = searcher.search(&plain_query, &TopDocs::with_limit(10))?;
//     assert_eq!(
//         top_docs_plain.len(),
//         1,
//         "Should match the parent doc on last_name=McQueen"
//     );
//     println!("Query last_name:McQueen => top_docs_plain={top_docs_plain:?}");

//     Ok(())
// }

// mod explode {
//     use common::JsonPathWriter;
//     use serde_json::{Map, Value as JsonValue};

//     use crate::schema::{Field, JsonObjectOptions, ObjectMappingType, OwnedValue, Schema};
//     use crate::TantivyDocument;

//     /// A very simplified explode function for single-level nested arrays.
//     /// - If `path.is_empty()`, return a single doc containing the entire `json_val`.
//     /// - Otherwise, treat the top-level as an object, find the subfield at `path[0]`,
//     ///   and if it is a nested array, produce child docs for each item, plus one parent doc
//     ///   that has the other fields + an empty array for that subfield.
//     ///
//     /// This is enough to pass the `test_nested_query_scenario_small` test.
//     pub fn explode_document(
//         json_val: &JsonValue,
//         path: &[String],
//         json_field: Field,
//         schema: &Schema,
//         opts: &JsonObjectOptions,
//     ) -> Vec<TantivyDocument> {
//         // If no path: just store the JSON as a single doc. Not nested logic.
//         if path.is_empty() {
//             let mut doc = TantivyDocument::new();
//             doc.add_field_value(json_field, &OwnedValue::from(json_val.clone()));
//             return vec![doc];
//         }

//         // For simplicity, let's only handle the first path segment:
//         let subfield_name = &path[0];

//         // We expect the top-level `json_val` to be an object.
//         let top_obj = match json_val.as_object() {
//             Some(m) => m,
//             None => {
//                 // fallback: not an object => just store in one doc
//                 let mut doc = TantivyDocument::new();
//                 doc.add_field_value(json_field, &OwnedValue::from(json_val.clone()));
//                 return vec![doc];
//             }
//         };

//         // Attempt to find the subfield in this object that might be nested.
//         let maybe_subval = top_obj.get(subfield_name);

//         // 1) Gather child docs if this subfield is a nested array
//         let mut child_docs = Vec::new();

//         if let Some(subval) = maybe_subval {
//             // Check if the subfield is nested
//             if opts.object_mapping_type == ObjectMappingType::Nested {
//                 // If it's an array, produce child docs for each item
//                 if let Some(arr) = subval.as_array() {
//                     for item in arr {
//                         // Build a new JSON object that includes the path key
//                         // so that each child doc has { "vehicle": { ...child fields... } }
//                         // if `subfield_name` = "vehicle".
//                         let child_object = serde_json::json!({
//                             subfield_name: item
//                         });

//                         let mut child_doc = TantivyDocument::new();
//                         child_doc.add_field_value(json_field, &OwnedValue::from(child_object));
//                         // Children do NOT set the `_is_parent_...` flag
//                         child_docs.push(child_doc);
//                     }
//                 }
//             }
//         }

//         // 2) Now build the final parent doc. We copy all top-level fields except the nested array
//         //    is replaced with an empty array (if `include_in_parent=false`) or removed entirely.
//         let mut parent_map = Vec::new();
//         for (k, v) in top_obj {
//             if k == subfield_name {
//                 // We skip or replace with empty array if `include_in_parent=false`.
//                 if opts.nested_options.include_in_parent {
//                     // store an empty array in the parent
//                     let empty_array = JsonValue::Array(vec![]);
//                     parent_map.push((k.clone(), OwnedValue::from(empty_array)));
//                 } else {
//                     // removing it entirely
//                 }
//             } else {
//                 // Normal field => just copy it
//                 parent_map.push((k.clone(), OwnedValue::from(v.clone())));
//             }
//         }

//         let mut parent_doc = TantivyDocument::new();
//         parent_doc.add_field_value(json_field, &OwnedValue::Object(parent_map));

//         if opts.object_mapping_type == ObjectMappingType::Nested {
//             set_parent_flag(&mut parent_doc, path, schema);
//         }

//         // Final doc => appended after child docs
//         let mut docs = child_docs;
//         docs.push(parent_doc);
//         docs
//     }

//     /// Helper: sets `_is_parent_<path>` = true if that field exists in the schema.
//     fn set_parent_flag(doc: &mut TantivyDocument, path: &[String], schema: &Schema) {
//         let mut path_builder = JsonPathWriter::new();
//         for seg in path {
//             path_builder.push(seg);
//         }
//         let parent_flag_name = format!("_is_parent_{}", path_builder.as_str());

//         if let Ok(flag_field) = schema.get_field(&parent_flag_name) {
//             doc.set_is_parent(flag_field, true);
//         }
//     }

//     #[cfg(test)]
//     mod tests {
//         use super::*;
//         use crate::schema::{
//             IndexRecordOption, JsonObjectOptions, NumericOptions, Schema, SchemaBuilder,
//             TextFieldIndexing,
//         };
//         use crate::{doc, Document, TantivyDocument};
//         use serde_json::json;

//         /// Compare two `TantivyDocument` objects by converting both to `Document`,
//         /// then to a JSON string. If they differ, the assertion fails.
//         ///
//         /// This is often the easiest way to confirm that two docs are structurally equal.
//         fn assert_docs_eq(
//             schema: &Schema,
//             actual_doc: &TantivyDocument,
//             expected_doc: &TantivyDocument,
//             msg: &str,
//         ) {
//             let actual_json = actual_doc.to_json(schema);
//             let expected_json = expected_doc.to_json(schema);

//             assert_eq!(
//                 actual_json, expected_json,
//                 "{}\n\nActual doc: {}\nExpected doc: {}",
//                 msg, actual_json, expected_json
//             );
//         }

//         #[test]
//         fn test_nested_subfields_both_levels() {
//             // We'll illustrate a "driver" field that is NOT nested,
//             // but "crew" and "vehicle" subfields are nested.
//             // We'll set `include_in_parent=false` so the aggregator doc at that level
//             // has empty arrays, *plus* we produce aggregator docs for each subfield and
//             // each item in the sub-arrays.

//             let mut schema_builder = SchemaBuilder::default();

//             // "driver" is not nested at top-level
//             let mut driver_opts = JsonObjectOptions::default()
//                 .set_nested(false, false) // => object_mapping_type=Nested? Actually "false,false" => .object_mapping_type=Default
//                 .set_indexing_options(
//                     TextFieldIndexing::default()
//                         .set_tokenizer("raw")
//                         .set_index_option(IndexRecordOption::Basic),
//                 );

//             // "crew" is a nested subfield => aggregator doc for "crew," plus array item docs
//             // if "crew" is an array.
//             let mut crew_opts = JsonObjectOptions::default()
//                 .set_nested(false, false) // => object_mapping_type=Nested, include_in_parent=false
//                 .set_indexing_options(
//                     TextFieldIndexing::default()
//                         .set_tokenizer("raw")
//                         .set_index_option(IndexRecordOption::Basic),
//                 );

//             // "kids" is nested sub-subfield => aggregator doc for "kids," plus item docs
//             crew_opts.subfields.insert(
//                 "kids".into(),
//                 JsonObjectOptions::default()
//                     .set_nested(false, false)
//                     .set_indexing_options(
//                         TextFieldIndexing::default()
//                             .set_tokenizer("raw")
//                             .set_index_option(IndexRecordOption::Basic),
//                     ),
//             );

//             // "vehicle" is also nested
//             let vehicle_opts = JsonObjectOptions::default()
//                 .set_nested(false, false)
//                 .set_indexing_options(
//                     TextFieldIndexing::default()
//                         .set_tokenizer("raw")
//                         .set_index_option(IndexRecordOption::Basic),
//                 );

//             // Insert them into "driver"
//             driver_opts.subfields.insert("crew".into(), crew_opts);
//             driver_opts.subfields.insert("vehicle".into(), vehicle_opts);

//             let driver_field = schema_builder.add_json_field("driver", driver_opts.clone());
//             // If you want to see `_is_parent_driver`, you must do set_nested(true,false) for top-level,
//             // and also define `_is_parent_driver` as a stored bool.
//             // We'll define the subfields though:
//             let bool_opts = NumericOptions::default().set_indexed().set_stored();
//             let _ = schema_builder.add_bool_field("_is_parent_driver\u{1}crew", bool_opts.clone());
//             let _ = schema_builder
//                 .add_bool_field("_is_parent_driver\u{1}crew\u{1}kids", bool_opts.clone());
//             let _ =
//                 schema_builder.add_bool_field("_is_parent_driver\u{1}vehicle", bool_opts.clone());

//             let schema = schema_builder.build();

//             // Let's define some multi-level nested data:
//             let val = json!({
//                 "crew": [
//                    {
//                      "role": "spotter",
//                      "person": "Joe",
//                      "kids": [
//                        { "name": "Eve", "age": 3 },
//                        { "name": "Sam", "age": 5 }
//                      ]
//                    },
//                    {
//                      "role": "mechanic",
//                      "person": "Jim",
//                      "kids": []
//                    }
//                 ],
//                 "vehicle": [
//                    { "make": "Powell", "model": "Canyonero" },
//                    { "make": "Miller", "model": "Ecto-1" }
//                 ],
//                 "last_name": "McQueen"
//             });

//             // Now explode
//             let mut child_docs = explode_document(
//                 &val,
//                 &["driver".into()],
//                 driver_field,
//                 &schema,
//                 &driver_opts,
//             );

//             assert_eq!(child_docs.len(), 11);

//             let parent_doc = child_docs.pop().unwrap();

//             // Because "driver" is NOT nested at top level, we do NOT set `_is_parent_driver`
//             // in the parent doc.  The aggregator doc just has "crew": [], "vehicle": [], "last_name":"McQueen".
//             let expected_parent = doc!(
//                 driver_field => json!({
//                     "crew": [],
//                     "vehicle": [],
//                     "last_name": "McQueen"
//                 })
//             )
//             .into();

//             assert_docs_eq(
//                 &schema,
//                 &parent_doc,
//                 &expected_parent,
//                 "top-level aggregator doc mismatch",
//             );

//             // Let's do some rough checks on how many child docs we got:
//             //  1) aggregator doc for "crew" array
//             //  2) aggregator doc for item #0 => "crew": { "role":"spotter", "person":"Joe", "kids": ...}
//             //  3) aggregator doc for item #0's subfield "kids" array
//             //  4) aggregator doc for "kids" item #0 => "kids": { "name":"Eve","age":3}
//             //  5) aggregator doc for "kids" item #1 => ...
//             //  6) aggregator doc for item #1 => "crew": { "role":"mechanic","person":"Jim","kids":[] }
//             //
//             //  7) aggregator doc for "vehicle" array
//             //  8) aggregator doc for item #0 => "vehicle": { "make":"Powell","model":"Canyonero" }
//             //  9) aggregator doc for item #1 => "vehicle": { "make":"Miller","model":"Ecto-1" }
//             //
//             // So we might see 9 aggregator docs in total.
//             // The order might differ. We'll just check len >= 9:
//             assert!(
//                 child_docs.len() >= 9,
//                 "We expect multiple aggregator docs for crew + kids + vehicle"
//             );
//         }

//         // #[test]
//         // fn test_merge_two_docs() {
//         //     // We need a real schema with 2 fields: field1, field2
//         //     let mut builder = SchemaBuilder::default();
//         //     // Mark them stored so we can see them in .to_json(schema)
//         //     let json_opts = JsonObjectOptions::default().set_stored();

//         //     let field1 = builder.add_json_field("field1", json_opts.clone());
//         //     let field2 = builder.add_json_field("field2", json_opts.clone());
//         //     let schema = builder.build();

//         //     // docA => field1 = { "subA":1 }
//         //     // docB => field2 = { "subB":2 }
//         //     let doc_a = doc!(
//         //         field1 => json!({"subA": 1})
//         //     );
//         //     let doc_b = doc!(
//         //         field2 => json!({"subB": 2})
//         //     );

//         //     let doc_a_tantivy: TantivyDocument = doc_a.into();
//         //     let doc_b_tantivy: TantivyDocument = doc_b.into();

//         //     let merged = merge_two_docs(&doc_a_tantivy, &doc_b_tantivy);

//         //     // expected => doc!( field1 => json!({"subA":1}), field2 => json!({"subB":2}) )
//         //     let expected = doc!(
//         //         field1 => json!({"subA":1}),
//         //         field2 => json!({"subB":2})
//         //     )
//         //     .into();

//         //     assert_docs_eq(&schema, &merged, &expected, "merged doc mismatch");
//         // }
//     }
// }

mod splode {
    use crate::schema::{JsonObjectOptions, ObjectMappingType};
    use common::JsonPathWriter;
    use serde_json::{Map, Value};

    /// Explode (flatten) a JSON value into multiple documents for block-join ("nested") indexing.
    /// Wrap a scalar/object/array in the given path chain (turn path segments into nested objects).
    fn wrap_in_path(path: &[&String], value: Value) -> Vec<Value> {
        // If there's no path, it's just this Value
        if path.is_empty() {
            return vec![value];
        }

        // Otherwise, nest the value inside successive objects for each segment.
        let mut current = value;
        for segment in path.iter().rev() {
            let mut map = serde_json::Map::new();
            map.insert((*segment).clone(), current);
            current = Value::Object(map);
        }
        vec![current]
    }

    pub fn explode(path: &[&String], value: Value, opts: Option<&JsonObjectOptions>) -> Vec<Value> {
        // If we have no options or object_mapping_type != Nested, just one doc with `value`.
        if opts.map_or(true, |o| o.object_mapping_type != ObjectMappingType::Nested) {
            return wrap_in_path(path, value);
        }

        // We have an object/array in "nested" mode => block-join indexing
        match value {
            Value::Array(arr) => {
                // produce one child doc for each array element
                let mut docs = Vec::new();
                for item in arr {
                    // Recursively wrap or explode each array element
                    let wrapped = wrap_in_path(path, item);
                    docs.extend(wrapped);
                }
                docs
            }
            Value::Object(obj) => {
                let mut docs = Vec::new();
                let mut parent_map = serde_json::Map::new();

                // 1) collect fields
                for (k, v) in obj {
                    // if there's a nested subfield, we also explode it
                    if let Some(child_opts) = opts.and_then(|o| o.subfields.get(&k)) {
                        // keep the original value for parent
                        parent_map.insert(k.clone(), v.clone());

                        // if child is nested, recurse
                        if child_opts.object_mapping_type == ObjectMappingType::Nested {
                            let mut child_path = path.to_vec();
                            child_path.push(&k);
                            let child_docs = explode(&child_path, v, Some(child_opts));
                            docs.extend(child_docs);
                        }
                    } else {
                        // if not nested, just keep for parent
                        parent_map.insert(k, v);
                    }
                }

                // 2) produce the parent doc, but only if we are not at the top level
                //    (if path is empty, that means this object *is* the top level).
                if !path.is_empty() {
                    // Example: path = ["root"] => we generate _is_parent_root = true
                    let mut parent_doc = serde_json::Map::new();
                    let parent_field = format!(
                        "_is_parent_{}",
                        path.into_iter()
                            .map(|item| item.to_string())
                            .collect::<Vec<_>>()
                            .join("_")
                    );
                    parent_doc.insert(parent_field, Value::Bool(true));

                    // Wrap `parent_map` back under path
                    let mut current = parent_map;
                    for segment in path.iter().rev() {
                        let mut new_map = serde_json::Map::new();
                        new_map.insert(segment.to_string(), Value::Object(current));
                        current = new_map;
                    }

                    // Merge that path-wrapped map into `parent_doc`
                    for (k, v) in current {
                        parent_doc.insert(k, v);
                    }

                    // push the final parent doc
                    docs.push(Value::Object(parent_doc));
                } else {
                    // At top level, DO NOT push the doc here, because the recursion
                    // for the top-level subfield is responsible for generating the
                    // single parent doc. That way, we won't get an extra doc.
                }

                docs
            }
            // for scalars, just wrap
            scalar => wrap_in_path(path, scalar),
        }
    }

    #[cfg(test)]
    mod tests {
        use std::collections::HashMap;

        use super::*;
        use crate::schema::{JsonObjectOptions, ObjectMappingType};
        use serde_json::{json, Value};

        #[test]
        fn explode_non_nested_empty_object() {
            let path = vec![];
            let path_refs: Vec<&String> = path.iter().collect();
            let value = json!({});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Default,
                ..Default::default()
            };
            let result = explode(&path_refs, value, Some(&opts));
            let expected = vec![json!({})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_non_nested_simple_object() {
            let path = vec![];
            let path_refs: Vec<&String> = path.iter().collect();
            let value = json!({"a": 1, "b": "two"});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Default,
                ..Default::default()
            };
            let result = explode(&path_refs, value, Some(&opts));
            let expected = vec![json!({"a": 1, "b": "two"})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_non_nested_array() {
            let path = vec![];
            let path_refs: Vec<&String> = path.iter().collect();
            let value = json!([1, 2, 3]);
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Default,
                ..Default::default()
            };
            let result = explode(&path_refs, value, Some(&opts));
            let expected = vec![json!([1, 2, 3])];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_empty_object() {
            let path = vec![];
            let path_refs: Vec<&String> = path.iter().collect();
            let value = json!({"root": {}});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: HashMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path_refs, value, Some(&opts));
            let parent_key = "_is_parent_root";
            let expected = vec![json!({parent_key: true, "root": {}})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_simple_scalar() {
            let path = vec!["field".to_string()];
            let path_refs: Vec<&String> = path.iter().collect();
            let value = json!("hello");
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                ..Default::default()
            };
            let result = explode(&path_refs, value, Some(&opts));
            // No parent key should be added if its a scalar.
            let expected = vec![json!({"field": "hello"})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_simple_object() {
            let path = vec![];
            let path_refs: Vec<&String> = path.iter().collect();
            let value = json!({"root": {"a": 1, "b": "two"}});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: HashMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path_refs, value, Some(&opts));
            let expected = vec![json!({ "_is_parent_root": true, "root": {"a": 1, "b": "two"}})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_deep_object() {
            let path = vec![];
            let path_refs: Vec<&String> = path.iter().collect();
            let value = json!({"a": {"b": {"c": 42}}});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: HashMap::from_iter([(
                    "a".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path_refs, value, Some(&opts));
            let expected = vec![json!({"_is_parent_a": true, "a": {"b": {"c": 42}}})];
            assert_eq!(result, expected);
        }

        #[test]
        fn explode_nested_wide_object() {
            // Value has a "root" field that we treat as 'nested'.
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

            // "root" is nested => we do block-join indexing for its subfields.
            // Among them, only "j" is also declared nested => we want to explode that array.
            let mut subfields_root = std::collections::HashMap::new();
            subfields_root.insert(
                "j".to_string(),
                JsonObjectOptions {
                    object_mapping_type: ObjectMappingType::Nested,
                    subfields: std::collections::HashMap::new(),
                    ..Default::default()
                },
            );

            let mut top_level_subfields = std::collections::HashMap::new();
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

            let path: Vec<&String> = vec![];
            let result = explode(&path, value, Some(&opts));

            // We expect 'j' is exploded into 3 child docs, plus a single parent doc
            // that has _is_parent_root: true and the full object.
            let expected = vec![
                serde_json::json!({ "root": { "j": 1 } }),
                serde_json::json!({ "root": { "j": 2 } }),
                serde_json::json!({ "root": { "j": { "k": "v" } } }),
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
            let path_refs: Vec<&String> = path.iter().collect();
            let value = json!([1, 2, 3]);
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                ..Default::default()
            };
            let result = explode(&path_refs, value, Some(&opts));
            let expected = vec![json!(1), json!(2), json!(3)];
            assert_eq!(result, expected);
        }

        #[test]
        #[ignore]
        fn explode_nested_array_of_objects() {
            let path = vec![];
            let path_refs: Vec<&String> = path.iter().collect();
            let value = json!({"root": [
                {"a": 1},
                {"b": 2},
                {"c": {"d": 3}}
            ]});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: HashMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path_refs, value, Some(&opts));
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
        #[ignore]
        fn explode_nested_multi_dimensional_arrays() {
            let path = vec![];
            let path_refs: Vec<&String> = path.iter().collect();
            let value = json!({"root": [
                [1, 2],
                [3, [4, 5]],
                [6, {"x": [7, 8]}]
            ]});
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: HashMap::from_iter([(
                    "root".into(),
                    JsonObjectOptions {
                        object_mapping_type: ObjectMappingType::Nested,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            };
            let result = explode(&path_refs, value, Some(&opts));
            let mut expected_parent = serde_json::Map::new();
            expected_parent.insert("_is_parent_multi".to_string(), json!(true));
            let expected = vec![json!({
                "_is_parent_multi": true,
                "root": [
                    [1, 2],
                    [3, [4, 5]],
                    [6, {"x": [7, 8]}]
                ]
            })];
            assert_eq!(result, expected);
        }

        #[test]
        #[test]
        fn explode_nested_mixed_types() {
            use std::collections::HashMap;

            // Configure subfields of "mixed" so that "array" and "letters" are nested.
            let mut subfields_mixed = HashMap::new();
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

            // Top-level "mixed" is also nested
            let opts = JsonObjectOptions {
                object_mapping_type: ObjectMappingType::Nested,
                subfields: subfields_mixed,
                ..Default::default()
            };

            // Our path to the top-level object is just ["mixed"]
            let path = vec!["mixed".to_string()];
            let path_refs: Vec<&String> = path.iter().collect();

            // Here's the input JSON object
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

            // Run the explode function
            let result = explode(&path_refs, value, Some(&opts));

            // Because "array" and "letters" are marked nested, each item in those arrays
            // becomes its own child doc, plus a final parent doc.
            let expected = vec![
                // Child docs for array
                json!({ "mixed": { "array": 1 } }),
                json!({ "mixed": { "array": "two" } }),
                json!({ "mixed": { "array": true } }),
                json!({ "mixed": { "array": null } }),
                json!({ "mixed": { "array": { "nested": [3, 4] }}}),
                // Child docs for letters
                json!({ "mixed": { "letters": { "a": 1 }}}),
                json!({ "mixed": { "letters": { "b": 2 }}}),
                json!({ "mixed": { "letters": { "c": { "d": 3 }}}}),
                // Finally, the parent doc that contains the entire object
                json!({
                    "_is_parent_mixed": true,
                    "mixed": {
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
                    }
                }),
            ];

            assert_eq!(result, expected);
        }
    }
}
