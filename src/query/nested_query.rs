use std::fmt;
use std::sync::Arc;

use common::JsonPathWriter;

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

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use explode::explode_document;
    use serde_json::json;

    use super::*;
    use crate::collector::TopDocs;
    use crate::query::nested_query::ScoreMode;
    use crate::query::{AllQuery, QueryClone, QueryParser, TermQuery};
    // use crate::schema::document::parse_json_for_nested_sorted;
    use crate::schema::{
        DocParsingError, Field, IndexRecordOption, JsonObjectOptions, NumericOptions, Schema,
        SchemaBuilder, TextFieldIndexing, Value, STORED, STRING, TEXT,
    };
    use crate::tokenizer::SimpleTokenizer;
    use crate::{Index, IndexWriter, TantivyDocument, Term};

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

        let query_parser = QueryParser::for_index(&index, vec![]);

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
        let mut schema_builder = SchemaBuilder::default();

        let driver_json_options = JsonObjectOptions::default()
            .set_nested(true, false)
            .set_indexing_options(TextFieldIndexing::default())
            .add_subfield(
                "vehicle",
                JsonObjectOptions::default()
                    .set_nested(true, false)
                    .set_indexing_options(TextFieldIndexing::default()),
            );
        // let driver_last_name = schema_builder.add_text_field(
        //     "driver.last_name",
        //     TextFieldIndexing::default().
        // set_index_option(IndexRecordOption::WithFreqsAndPositions), );

        // let driver_vehicle_make = schema_builder.add_text_field(
        //     "driver.vehicle.make",
        //     TextFieldIndexing::default().
        // set_index_option(IndexRecordOption::WithFreqsAndPositions), );
        // let driver_vehicle_model =
        //     schema_builder.add_text_field("driver.vehicle.model", TextFieldIndexing::default());

        // We also define boolean fields that indicate whether each doc is the "parent" doc
        // for a given nested path.  The NestedQuery code looks for a field like "_is_parent_driver"
        // (and similarly "_is_parent_driver.vehicle") with "true" for the parent docs.
        // In practice, you can name these fields however you like, as long as they match the
        // "parent_flag_name" from NestedQuery’s logic:
        //    let parent_flag_name = format!("_is_parent_{}", path_str);
        //
        // For path = "driver", path_str is "driver", so the field is "_is_parent_driver".
        // For path = "driver.vehicle", path_str is "driver.vehicle", so the field is
        // "_is_parent_driver.vehicle".
        //
        // We'll define them as normal boolean fields in Tantivy.
        let bool_options = NumericOptions::default().set_stored().set_indexed();
        let is_parent_field = schema_builder.add_bool_field("_is_parent_vehicle", bool_options);

        // Build final schema
        let schema = schema_builder.build();

        // 2) Create an in-memory index, plus an IndexWriter.
        //
        let index = Index::create_in_ram(schema.clone());
        let mut writer: IndexWriter<TantivyDocument> = index.writer(50_000_000)?;
        let big_json = json!({
            "driver":  {
                "last_name": "McQueen",
                "vehicle": [
                    { "make": "Powell Motors", "model": "Canyonero" },
                    { "make": "Miller-Meteor", "model": "Ecto-1" }
                ]
            },
        });

        Ok(())
    }

    #[test]
    fn test_nested_query_scenario() -> crate::Result<()> {
        use crate::query::nested_query::{NestedQuery, ScoreMode}; // your NestedQuery
        use crate::query::TermQuery;
        use crate::schema::{JsonObjectOptions, NumericOptions, SchemaBuilder};
        use crate::{collector::TopDocs, Index, IndexWriter, Term};
        use serde_json::json;

        // 1) Build a schema with:
        //    - a JSON field "driver_json" that is nested
        //    - boolean fields named "_is_parent_driver_json" and "_is_parent_driver_json.vehicle"
        //      so that `NestedQuery` can find them.
        //    Adjust naming as needed to match your code.
        let mut schema_builder = SchemaBuilder::new();

        // Mark the top-level JSON as nested.

        let mut driver_json_opts = JsonObjectOptions::default()
            .set_nested(true, false)
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw") // or "keyword"
                    .set_index_option(IndexRecordOption::Basic),
            );
        // Add a subfield for "vehicle" also as nested

        driver_json_opts.subfields.insert(
            "vehicle".to_string(),
            JsonObjectOptions::default()
                .set_nested(true, false)
                .set_indexing_options(
                    TextFieldIndexing::default()
                        .set_tokenizer("raw") // or "keyword"
                        .set_index_option(IndexRecordOption::Basic),
                ),
        );

        let driver_field = schema_builder.add_json_field("driver_json", driver_json_opts.clone());

        // Now define the boolean fields for block-join queries. They must be indexed bools.
        // path_str for `driver_json` is just "driver_json", so the field is "_is_parent_driver_json"
        let bool_opts = NumericOptions::default().set_stored().set_indexed();
        let is_parent_driver_json_field =
            schema_builder.add_bool_field("_is_parent_driver_json", bool_opts.clone());

        // For path = ["driver_json","vehicle"], path_str is "driver_json.vehicle",
        // so we define "_is_parent_driver_json.vehicle"
        let is_parent_driver_json_vehicle_field =
            schema_builder.add_bool_field("_is_parent_driver_json\u{1}vehicle", bool_opts);

        let schema = schema_builder.build();

        // 2) Create an in‐memory index and an IndexWriter.
        let index = Index::create_in_ram(schema.clone());
        let mut writer: IndexWriter<TantivyDocument> =
            index.writer_with_num_threads(2, 50_000_000)?;

        // 3) Example nested JSON data
        let big_json = json!({
            "driver_json": {
                "last_name": "McQueen",
                "vehicle": [
                    { "make": "Powell Motors", "model": "Canyonero" },
                    { "make": "Miller-Meteor", "model": "Ecto-1" }
                ]
            }
        });

        // 4) Explode the JSON object into multiple docs
        // Adjust references to your actual `explode` module or method as needed.
        let exploded_docs = explode_document(
            &big_json["driver_json"],
            &["driver_json".into()],
            driver_field,
            &schema,
            &driver_json_opts,
        );

        println!("TEST SCENARIO: Got {} exploded docs", exploded_docs.len());

        // 5) Add all child + parent docs in one call
        writer.add_documents(exploded_docs)?;
        writer.commit()?; // flush & commit

        // 6) Now we can run a NestedQuery.
        //    e.g. childTerm => "Powell"
        // let child_term = Term::from_field_text(driver_field, "powell");
        // let child_q = TermQuery::new(child_term, IndexRecordOption::Basic);
        let query_parser = QueryParser::for_index(&index, vec![driver_field]);
        let child_q = query_parser
            .parse_query("driver_json.vehicle.model:Canyonero")
            .unwrap();

        // path = ["driver_json", "vehicle"]
        // ignore_unmapped = false
        let nested_query = NestedQuery::new(
            vec!["driver_json".to_string(), "vehicle".to_string()],
            Box::new(child_q),
            ScoreMode::Avg,
            false,
        );

        // 7) Execute search
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;

        // We expect exactly 1 hit: The parent doc that has a child "Powell"
        assert_eq!(top_docs.len(), 1, "Expected exactly one matching doc.");

        Ok(())
    }

    //     fn make_schema_for_eq_tests() -> (Schema, Field, Field) {
    //         let mut builder = SchemaBuilder::default();

    //         let group_f = builder.add_text_field("group", STRING | STORED);

    //         let nested_opts = NestedJsonObjectOptions::new()
    //             .set_include_in_parent(false)
    //             .set_store_parent_flag(true)
    //             .set_stored()
    //             .set_indexing_options(
    //                 TextFieldIndexing::default()
    //                     .set_tokenizer("default")
    //                     .set_index_option(IndexRecordOption::Basic),
    //             );
    //         let user_nested_f = builder.add_nested_json_field(vec!["user".into()], nested_opts);

    //         let schema = builder.build();
    //         (schema, user_nested_f, group_f)
    //     }

    //     fn index_doc_for_eq_tests(
    //         index_writer: &mut crate::indexer::IndexWriter,
    //         schema: &Schema,
    //         group_val: &str,
    //         user_array: serde_json::Value,
    //     ) {
    //         let top_obj = json!({
    //             "group": group_val,
    //             "user": user_array
    //         });
    //         let mut document = TantivyDocument::default();
    //         let expanded = parse_json_for_nested_sorted(schema, &mut document,
    // &top_obj).unwrap();

    //         index_writer.add_documents(expanded).unwrap();
    //     }

    //     #[test]
    //     fn test_ignore_unmapped_true() {
    //         let (schema, _user_nested_f, _group_f) = make_schema_for_eq_tests();
    //         let index = Index::create_in_ram(schema.clone());

    //         {
    //             let mut writer = index.writer_for_tests().unwrap();
    //             index_doc_for_eq_tests(
    //                 &mut writer,
    //                 &schema,
    //                 "someGroup",
    //                 json!([
    //                     { "first": "Bob", "last": "Smith" }
    //                 ]),
    //             );
    //             writer.commit().unwrap();
    //         }

    //         let reader = index.reader().unwrap();
    //         let searcher = reader.searcher();

    //         let query_parser = QueryParser::for_index(&index, vec![]);
    //         let child_q = query_parser.parse_query("user.first:Bob").unwrap();

    //         let nested_q = NestedQuery::new(
    //             vec!["unmapped".into()],
    //             Box::new(child_q),
    //             ScoreMode::None,
    //             true,
    //         );

    //         let top_docs = searcher
    //             .search(&nested_q, &TopDocs::with_limit(10))
    //             .unwrap();
    //         assert_eq!(
    //             top_docs.len(),
    //             0,
    //             "Expected zero hits for ignore_unmapped=true + unknown path"
    //         );
    //     }

    //     #[test]
    //     fn test_ignore_unmapped_false_error() {
    //         let (schema, _user_nested_f, _group_f) = make_schema_for_eq_tests();
    //         let index = Index::create_in_ram(schema.clone());

    //         let reader = index.reader().unwrap();
    //         let searcher = reader.searcher();

    //         let query_parser = QueryParser::for_index(&index, vec![]);
    //         let child_q = query_parser.parse_query("user.first:Anything").unwrap();
    //         let nested_q = NestedQuery::new(
    //             vec!["unmapped".into()],
    //             Box::new(child_q),
    //             ScoreMode::None,
    //             false,
    //         );

    //         let result = searcher.search(&nested_q, &TopDocs::with_limit(10));
    //         match result {
    //             Ok(_) => panic!("Expected an error for path=unmapped + ignore_unmapped=false"),
    //             Err(e) => {
    //                 let msg = format!("{:?}", e);
    //                 assert!(msg.contains("not mapped") && !msg.contains("ignore_unmapped=true"));
    //             }
    //         }
    //     }

    //     #[test]
    //     fn test_nested_query_some_match() -> crate::Result<()> {
    //         let (schema, user_nested_f, _group_f) = make_schema_for_eq_tests();
    //         let index = Index::create_in_ram(schema.clone());

    //         {
    //             let mut writer = index.writer_for_tests()?;
    //             index_doc_for_eq_tests(
    //                 &mut writer,
    //                 &schema,
    //                 "fans",
    //                 json!([
    //                     {"first":"Bob","last":"Smith"},
    //                     {"first":"Alice","last":"Branson"}
    //                 ]),
    //             );
    //             writer.commit()?;
    //         }

    //         let reader = index.reader()?;
    //         let searcher = reader.searcher();

    //         let query_parser = QueryParser::for_index(&index, vec![user_nested_f]);
    //         let child_q = query_parser.parse_query("first:Alice").unwrap();

    //         let nested_q = NestedQuery::new(
    //             vec!["user".into()],
    //             Box::new(child_q),
    //             ScoreMode::Avg,
    //             false,
    //         );

    //         let top_docs = searcher.search(&nested_q, &TopDocs::with_limit(10))?;
    //         assert_eq!(1, top_docs.len());
    //         Ok(())
    //     }

    //     #[test]
    //     fn test_nested_score_modes() -> crate::Result<()> {
    //         let (schema, nested_f, _group_f) = make_schema_for_eq_tests();
    //         let index = Index::create_in_ram(schema.clone());

    //         {
    //             let mut writer = index.writer_for_tests()?;
    //             index_doc_for_eq_tests(
    //                 &mut writer,
    //                 &schema,
    //                 "someGroup",
    //                 json!([
    //                     {"first":"java"},
    //                     {"first":"java"},
    //                     {"first":"rust"}
    //                 ]),
    //             );
    //             writer.commit()?;
    //         }

    //         let reader = index.reader()?;
    //         let searcher = reader.searcher();

    //         let query_parser = QueryParser::for_index(&index, vec![nested_f]);
    //         let child_q = query_parser.parse_query("first:java").unwrap();
    //         for &mode in &[
    //             ScoreMode::None,
    //             ScoreMode::Total,
    //             ScoreMode::Avg,
    //             ScoreMode::Max,
    //             ScoreMode::Min,
    //         ] {
    //             let nested_q = NestedQuery::new(
    //                 vec!["user".into()],
    //                 Box::new(child_q.box_clone()),
    //                 mode,
    //                 false,
    //             );
    //             let hits = searcher.search(&nested_q, &TopDocs::with_limit(10))?;
    //             assert_eq!(1, hits.len());
    //         }
    //         Ok(())
    //     }

    //     #[test]
    //     fn test_nested_score_mode_parsing() {
    //         assert_eq!(ScoreMode::from_str("none").unwrap(), ScoreMode::None);
    //         assert_eq!(ScoreMode::from_str("avg").unwrap(), ScoreMode::Avg);
    //         assert_eq!(ScoreMode::from_str("max").unwrap(), ScoreMode::Max);
    //         assert_eq!(ScoreMode::from_str("min").unwrap(), ScoreMode::Min);
    //         assert_eq!(ScoreMode::from_str("total").unwrap(), ScoreMode::Total);

    //         let err = ScoreMode::from_str("garbage").unwrap_err();
    //         assert!(err.to_string().contains("Unrecognized nested score_mode"));
    //     }

    //     fn make_multi_level_schema() -> (Schema, Field, Field, Field) {
    //         let mut builder = Schema::builder();

    //         let doc_tag_field = builder.add_text_field("doc_tag", STRING | STORED);

    //         let driver_nested_opts = NestedJsonObjectOptions::new()
    //             .set_include_in_parent(false)
    //             .set_store_parent_flag(true)
    //             .set_indexing_options(
    //                 TextFieldIndexing::default()
    //                     .set_tokenizer("raw")
    //                     .set_index_option(IndexRecordOption::Basic),
    //             );
    //         let driver_field = builder.add_nested_json_field(vec!["driver".into()],
    // driver_nested_opts);

    //         let vehicle_nested_opts = NestedJsonObjectOptions::new()
    //             .set_include_in_parent(false)
    //             .set_store_parent_flag(true)
    //             .set_indexing_options(
    //                 TextFieldIndexing::default()
    //                     .set_tokenizer("raw")
    //                     .set_index_option(IndexRecordOption::Basic),
    //             );
    //         let vehicle_field = builder
    //             .add_nested_json_field(vec!["driver".into(), "vehicle".into()],
    // vehicle_nested_opts);

    //         let schema = builder.build();
    //         (schema, doc_tag_field, driver_field, vehicle_field)
    //     }

    //     fn index_doc_multi_level(
    //         writer: &mut IndexWriter,
    //         schema: &Schema,
    //         doc_tag: &str,
    //         last_name: &str,
    //         vehicles: serde_json::Value,
    //     ) -> Result<(), DocParsingError> {
    //         let doc_obj = json!({
    //             "doc_tag": doc_tag,
    //             "driver": {
    //                 "last_name": last_name,
    //                 "vehicle": vehicles
    //             }
    //         });
    //         let json_doc = serde_json::to_string(&doc_obj).unwrap();
    //         let mut document = TantivyDocument::default();
    //         let expanded = parse_json_for_nested_sorted(
    //             schema,
    //             &mut document,
    //             &serde_json::from_str::<serde_json::Value>(&json_doc).unwrap(),
    //         )
    //         .unwrap();
    //         writer.add_documents(expanded).unwrap();
    //         Ok(())
    //     }

    //     #[test]
    //     fn test_multi_level_nested_query() -> crate::Result<()> {
    //         let (schema, _doc_tag_field, driver_field, vehicle_field) =
    // make_multi_level_schema();

    //         let index = Index::create_in_ram(schema.clone());
    //         {
    //             let mut writer = index.writer_for_tests()?;

    //             index_doc_multi_level(
    //                 &mut writer,
    //                 &schema,
    //                 "Doc1",
    //                 "McQueen",
    //                 json!([
    //                     { "make":"Powell Motors", "model":"Canyonero"},
    //                     { "make":"Miller-Meteor", "model":"Ecto-1"}
    //                 ]),
    //             )?;

    //             index_doc_multi_level(
    //                 &mut writer,
    //                 &schema,
    //                 "Doc2",
    //                 "Hudson",
    //                 json!([
    //                     { "make":"Mifune", "model":"Mach Five" },
    //                     { "make":"Miller-Meteor", "model":"Ecto-1" }
    //                 ]),
    //             )?;
    //             writer.commit()?;
    //         }

    //         let reader = index.reader()?;
    //         let searcher = reader.searcher();

    //         let query_parser = QueryParser::for_index(&index, vec![vehicle_field, driver_field]);
    //         let make_q = query_parser.parse_query("make:\"Powell Motors\"").unwrap();
    //         let model_q = query_parser.parse_query("model:Canyonero").unwrap();
    //         let bool_sub = BooleanQuery::new(vec![
    //             (Occur::Must, Box::new(make_q)),
    //             (Occur::Must, Box::new(model_q)),
    //         ]);

    //         let vehicle_nested = NestedQuery::new(
    //             vec!["driver".into(), "vehicle".into()],
    //             Box::new(bool_sub),
    //             ScoreMode::Avg,
    //             false,
    //         );
    //         let driver_nested = NestedQuery::new(
    //             vec!["driver".into()],
    //             Box::new(vehicle_nested),
    //             ScoreMode::Avg,
    //             false,
    //         );

    //         let hits = searcher.search(&driver_nested, &TopDocs::with_limit(10))?;
    //         assert_eq!(0, hits.len(), "multi-level nesting is still a todo!");

    //         Ok(())
    //     }

    //     fn make_comments_schema() -> (Schema, Field, Field, Field) {
    //         let mut builder = Schema::builder();

    //         let doc_num_field = builder.add_text_field("doc_num", STRING | STORED);

    //         let nested_opts = NestedJsonObjectOptions::new()
    //             .set_include_in_parent(false)
    //             .set_store_parent_flag(true);
    //         let comments_field = builder.add_nested_json_field(vec!["comments".into()],
    // nested_opts);

    //         let author_field = builder.add_text_field("comments.author", STRING);

    //         let schema = builder.build();
    //         (schema, doc_num_field, comments_field, author_field)
    //     }

    //     fn index_doc_with_comments(
    //         writer: &mut IndexWriter,
    //         schema: &Schema,
    //         doc_num: &str,
    //         comments: serde_json::Value,
    //     ) -> Result<(), DocParsingError> {
    //         let doc_obj = json!({
    //             "doc_num": doc_num,
    //             "comments": comments
    //         });
    //         let json_doc = serde_json::to_string(&doc_obj).unwrap();
    //         let mut document = TantivyDocument::default();
    //         let expanded = parse_json_for_nested_sorted(
    //             schema,
    //             &mut document,
    //             &serde_json::from_str::<serde_json::Value>(&json_doc).unwrap(),
    //         )
    //         .unwrap();
    //         writer.add_documents(expanded).unwrap();
    //         Ok(())
    //     }

    //     #[test]
    //     #[ignore = "must not queries not working to block join semantics"]
    //     fn test_comments_must_not_nested() -> crate::Result<()> {
    //         let (schema, doc_num_f, _comments_f, author_f) = make_comments_schema();
    //         let index = Index::create_in_ram(schema.clone());

    //         {
    //             let mut writer = index.writer_for_tests()?;
    //             index_doc_with_comments(
    //                 &mut writer,
    //                 &schema,
    //                 "1",
    //                 json!([
    //                     {"author":"kimchy"}
    //                 ]),
    //             )?;
    //             index_doc_with_comments(
    //                 &mut writer,
    //                 &schema,
    //                 "2",
    //                 json!([
    //                     {"author":"kimchy"},
    //                     {"author":"nik9000"}
    //                 ]),
    //             )?;
    //             index_doc_with_comments(
    //                 &mut writer,
    //                 &schema,
    //                 "3",
    //                 json!([
    //                     {"author":"nik9000"}
    //                 ]),
    //             )?;
    //             writer.commit()?;
    //         }

    //         let reader = index.reader()?;
    //         let searcher = reader.searcher();

    //         use crate::query::{BooleanQuery, Occur};

    //         let tq_nik = TermQuery::new(
    //             Term::from_field_text(author_f, "nik9000"),
    //             IndexRecordOption::Basic,
    //         );
    //         let must_not = BooleanQuery::new(vec![
    //             (Occur::Must, Box::new(AllQuery)),
    //             (Occur::MustNot, Box::new(tq_nik)),
    //         ]);
    //         let nested_query = NestedQuery::new(
    //             vec!["comments".into()],
    //             Box::new(must_not),
    //             ScoreMode::None,
    //             false,
    //         );

    //         let hits = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
    //         assert_eq!(
    //             1,
    //             hits.len(),
    //             "We get only doc #1 => doc #2 and doc #3 are excluded by must_not"
    //         );

    //         let (_score, addr) = hits[0];
    //         let stored_doc: TantivyDocument = searcher.doc(addr)?;
    //         let doc_num_val = stored_doc
    //             .get_first(doc_num_f)
    //             .unwrap()
    //             .as_str()
    //             .unwrap()
    //             .to_string();
    //         assert_eq!("1", doc_num_val);

    //         let tq_nik2 = TermQuery::new(
    //             Term::from_field_text(author_f, "nik9000"),
    //             IndexRecordOption::Basic,
    //         );
    //         let nested2 = NestedQuery::new(
    //             vec!["comments".into()],
    //             Box::new(tq_nik2),
    //             ScoreMode::None,
    //             false,
    //         );
    //         let bool_q = BooleanQuery::new(vec![
    //             (Occur::Must, Box::new(AllQuery)),
    //             (Occur::MustNot, Box::new(nested2)),
    //         ]);

    //         let hits2 = searcher.search(&bool_q, &TopDocs::with_limit(10))?;
    //         assert_eq!(
    //             1,
    //             hits2.len(),
    //             "Only doc #1 remains under an outer must_not"
    //         );
    //         let (_score2, addr2) = hits2[0];
    //         let doc_stored2: TantivyDocument = searcher.doc(addr2)?;
    //         let doc_num2 = doc_stored2
    //             .get_first(doc_num_f)
    //             .map(|v| v.as_str().unwrap().to_string())
    //             .unwrap();
    //         assert_eq!("1", doc_num2);

    //         Ok(())
    //     }

    //     #[test]
    //     fn test_nested_query_without_subfields() -> crate::Result<()> {
    //         use crate::collector::TopDocs;
    //         use crate::query::TermQuery;
    //         use crate::schema::{
    //             IndexRecordOption, SchemaBuilder, TantivyDocument, TextFieldIndexing, TEXT,
    //         };
    //         use crate::{Index, Term};

    //         let mut builder = SchemaBuilder::default();
    //         let nested_json_opts = NestedJsonObjectOptions::new()
    //             .set_include_in_parent(false)
    //             .set_store_parent_flag(true)
    //             .set_stored()
    //             .set_indexing_options(
    //                 TextFieldIndexing::default()
    //                     .set_tokenizer("default")
    //                     .set_index_option(IndexRecordOption::Basic),
    //             );
    //         let user_field = builder.add_nested_json_field(vec!["user".into()],
    // nested_json_opts);         builder.add_text_field("group", TEXT);

    //         let schema = builder.build();
    //         let index = Index::create_in_ram(schema.clone());

    //         {
    //             let mut writer: IndexWriter<TantivyDocument> = index.writer_for_tests()?;
    //             let json_doc = r#"{
    //             "group": "fans",
    //             "user": [
    //                 { "first":"John", "last":"Smith" },
    //                 { "first":"Alice", "last":"White" }
    //             ]
    //         }"#;

    //             let mut document = TantivyDocument::default();
    //             let expanded_docs = parse_json_for_nested_sorted(
    //                 &schema,
    //                 &mut document,
    //                 &serde_json::from_str::<serde_json::Value>(json_doc).unwrap(),
    //             )
    //             .expect("parse nested doc");
    //             let docs = expanded_docs.into_iter().collect::<Vec<_>>();
    //             writer.add_documents(docs)?;
    //             writer.commit()?;
    //         }

    //         let reader = index.reader()?;
    //         let searcher = reader.searcher();

    //         let mut child_term = Term::from_field_json_path(user_field, "first", false);
    //         child_term.append_type_and_str("alice");
    //         let child_query = TermQuery::new(child_term, IndexRecordOption::Basic);

    //         let nested_query = NestedQuery::new(
    //             vec!["user".to_string()],
    //             Box::new(child_query),
    //             ScoreMode::None,
    //             false,
    //         );

    //         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
    //         assert_eq!(
    //             top_docs.len(),
    //             1,
    //             "Should find parent doc with child 'alice'"
    //         );

    //         Ok(())
    //     }

    //     #[test]
    //     fn test_nested_query_parser_syntax() -> crate::Result<()> {
    //         let mut builder = SchemaBuilder::default();
    //         let nested_json_opts = NestedJsonObjectOptions::new()
    //             .set_include_in_parent(false)
    //             .set_store_parent_flag(true)
    //             .set_stored()
    //             .set_indexing_options(
    //                 TextFieldIndexing::default()
    //                     .set_tokenizer("default")
    //                     .set_index_option(IndexRecordOption::Basic),
    //             );
    //         let user_field = builder.add_nested_json_field(vec!["user".into()],
    // nested_json_opts);         let _group_field = builder.add_text_field("group", TEXT |
    // STORED);

    //         let schema = builder.build();
    //         let index = Index::create_in_ram(schema.clone());

    //         {
    //             let mut writer = index.writer_for_tests()?;
    //             let json_doc = r#"
    //         {
    //           "group": "complexFans",
    //           "user": [
    //             {
    //               "first": "Alice",
    //               "last": "Anderson",
    //               "hobbies": ["Chess", "Painting"],
    //               "kids": [
    //                 { "name": "Bob",   "age": 8 },
    //                 { "name": "Cathy", "age": 12 }
    //               ]
    //             },
    //             {
    //               "first": "Greg",
    //               "last":  "Johnson",
    //               "hobbies": ["Skiing", "Chess"],
    //               "kids": [
    //                 { "name": "Hank", "age": 3 }
    //               ]
    //             }
    //           ]
    //         }
    //         "#;

    //             let mut document = TantivyDocument::default();

    //             let expanded = parse_json_for_nested_sorted(
    //                 &schema,
    //                 &mut document,
    //                 &serde_json::from_str::<serde_json::Value>(json_doc).unwrap(),
    //             )
    //             .expect("parse nested doc");
    //             let docs: Vec<TantivyDocument> = expanded.into_iter().map(Into::into).collect();
    //             writer.add_documents(docs)?;
    //             writer.commit()?;
    //         }

    //         {
    //             let reader = index.reader()?;
    //             let searcher = reader.searcher();

    //             let query_parser = QueryParser::for_index(&index, vec![user_field]);

    //             let query_str = r#"
    //            user.first:Alice
    //            AND user.hobbies:Chess
    //            AND user.kids.age:[8 TO 9]
    //         "#;

    //             let query = query_parser.parse_query(query_str)?;

    //             let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;

    //             assert_eq!(
    //                 top_docs.len(),
    //                 1,
    //                 "Should find exactly one parent doc with child that meets all constraints."
    //             );

    //             let nested_query = NestedQuery::new(
    //                 vec!["user".into()],
    //                 Box::new(query_parser.parse_query("kids.name:Bob AND kids.age:8")?),
    //                 ScoreMode::None,
    //                 false,
    //             );

    //             let top_docs2 = searcher.search(&nested_query, &TopDocs::with_limit(10))?;

    //             assert_eq!(
    //                 top_docs2.len(),
    //                 1,
    //                 "Should match the same doc that has a kid named Bob and age 8"
    //             );

    //             let nested_query = NestedQuery::new(
    //                 vec!["user".into()],
    //                 Box::new(query_parser.parse_query("kids.name:Bob AND kids.age:3")?),
    //                 ScoreMode::None,
    //                 false,
    //             );

    //             let top_docs3 = searcher.search(&nested_query, &TopDocs::with_limit(10))?;

    //             assert_eq!(
    //                 top_docs3.len(),
    //                 0,
    //                 "Should not match two separate nested docs at the same level"
    //             );

    //             Ok(())
    //         }
    //     }

    // fn make_nested_schema() -> (Schema, Field, Field) {
    //     let mut builder = Schema::builder();

    //     let group_field = builder.add_text_field("group", STRING | STORED);

    //     let nested_opts = NestedJsonObjectOptions::new()
    //         .set_include_in_parent(false)
    //         .set_store_parent_flag(true)
    //         .set_stored()
    //         .set_indexing_options(
    //             TextFieldIndexing::default()
    //                 .set_tokenizer("default")
    //                 .set_index_option(IndexRecordOption::Basic),
    //         );
    //     let user_nested_field = builder.add_nested_json_field(vec!["user".into()], nested_opts);

    //     let schema = builder.build();
    //     (schema, user_nested_field, group_field)
    // }

    //     fn index_test_document(
    //         index_writer: &mut IndexWriter,
    //         schema: &Schema,
    //         group_val: &str,
    //         users: serde_json::Value,
    //     ) -> Result<(), DocParsingError> {
    //         let full_doc = json!({
    //             "group": group_val,
    //             "user": users,
    //         });

    //         let mut document = TantivyDocument::default();
    //         let expanded = parse_json_for_nested_sorted(schema, &mut document,
    // &full_doc).unwrap();

    //         index_writer.add_documents(expanded).unwrap();

    //         Ok(())
    //     }

    //     #[test]
    // fn test_nested_query_single_level() -> crate::Result<()> {
    //     let mut builder = Schema::builder();
    //     let group_field = builder.add_text_field("group", STRING | STORED);

    //     let (schema, user_nested_field, _group_field) = make_nested_schema();
    //     let index = Index::create_in_ram(schema.clone());

    //     {
    //         let mut writer = index.writer_for_tests()?;

    //         index_test_document(
    //             &mut writer,
    //             &schema,
    //             "fans",
    //             json!([
    //                 { "first": "John", "last": "Smith" },
    //                 { "first": "Alice", "last": "White" }
    //             ]),
    //         )?;

    //         index_test_document(
    //             &mut writer,
    //             &schema,
    //             "boring",
    //             json!([
    //                 { "first": "Bob", "last": "Marley" }
    //             ]),
    //         )?;
    //         writer.commit()?;
    //     }

    //     let reader = index.reader()?;
    //     let searcher = reader.searcher();

    //     let query_parser = QueryParser::for_index(&index, vec![user_nested_field]);

    //     let child_query = query_parser.parse_query("first:Alice").unwrap();

    //     let nested_query = NestedQuery::new(
    //         vec!["user".to_string()],
    //         Box::new(child_query),
    //         ScoreMode::Avg,
    //         false,
    //     );

    //     let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
    //     assert_eq!(
    //         top_docs.len(),
    //         1,
    //         "Should match exactly one parent doc with a nested child `first = Alice`."
    //     );

    //     Ok(())
    // }

    //     #[test]
    //     fn test_nested_query_no_match() -> crate::Result<()> {
    //         let (schema, user_nested_field, _group_field) = make_nested_schema();
    //         let index = Index::create_in_ram(schema.clone());

    //         {
    //             let mut writer = index.writer_for_tests()?;
    //             index_test_document(
    //                 &mut writer,
    //                 &schema,
    //                 "groupVal",
    //                 json!([
    //                     {"first":"John"},
    //                     {"first":"Alice"}
    //                 ]),
    //             )?;
    //             writer.commit()?;
    //         }

    //         let reader = index.reader()?;
    //         let searcher = reader.searcher();

    //         let query_parser = QueryParser::for_index(&index, vec![user_nested_field]);
    //         let child_query = query_parser.parse_query("first:NoSuchName").unwrap();
    //         let nested_query = NestedQuery::new(
    //             vec!["user".into()],
    //             Box::new(child_query),
    //             ScoreMode::None,
    //             false,
    //         );

    //         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
    //         assert_eq!(0, top_docs.len(), "No matches expected");

    //         Ok(())
    //     }

    //     #[test]
    //     fn test_nested_query_ignore_unmapped() -> crate::Result<()> {
    //         let (schema, ufield, _group_field) = make_nested_schema();
    //         let index = Index::create_in_ram(schema.clone());

    //         {
    //             let mut writer = index.writer_for_tests()?;
    //             index_test_document(
    //                 &mut writer,
    //                 &schema,
    //                 "unmappedTest",
    //                 json!([
    //                     {"first":"SomeName"}
    //                 ]),
    //             )?;
    //             writer.commit()?;
    //         }

    //         let reader = index.reader()?;
    //         let searcher = reader.searcher();

    //         let query_parser = QueryParser::for_index(&index, vec![ufield]);
    //         let child_query = query_parser.parse_query("first:SomeName").unwrap();
    //         let nested_query = NestedQuery::new(
    //             vec!["someUnknownPath".to_string()],
    //             Box::new(child_query),
    //             ScoreMode::Total,
    //             true,
    //         );

    //         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
    //         assert_eq!(0, top_docs.len(), "No docs returned, but no error either");

    //         Ok(())
    //     }

    //     #[test]
    //     #[ignore]
    //     fn test_nested_query_unmapped_error() {
    //         let (schema, _ufield, _group_field) = make_nested_schema();
    //         let index = Index::create_in_ram(schema.clone());

    //         let reader = index.reader().unwrap();
    //         let searcher = reader.searcher();

    //         let query_parser = QueryParser::for_index(&index, vec![]);
    //         let child_query = query_parser.parse_query("first:X").unwrap();
    //         let nested_query = NestedQuery::new(
    //             vec!["badPath".into()],
    //             Box::new(child_query),
    //             ScoreMode::None,
    //             false,
    //         );

    //         let res = searcher.search(&nested_query, &TopDocs::with_limit(10));
    //         match res {
    //             Err(e) => {
    //                 let msg = format!("{:?}", e);
    //                 assert!(
    //                     msg.contains("NestedQuery path 'badPath' not mapped")
    //                         && !msg.contains("ignore_unmapped=true"),
    //                     "Expected schema error complaining about unmapped path"
    //                 );
    //             }
    //             Ok(_) => panic!("Expected an error for unmapped path with
    // ignore_unmapped=false"),         }
    //     }

    //     #[test]
    //     #[ignore]
    //     fn test_nested_query_numeric_leaf() -> crate::Result<()> {
    //         let mut builder = SchemaBuilder::default();
    //         let nested_opts = NestedJsonObjectOptions::new()
    //             .set_include_in_parent(false)
    //             .set_store_parent_flag(true)
    //             .set_indexing_options(
    //                 TextFieldIndexing::default()
    //                     .set_tokenizer("default")
    //                     .set_index_option(IndexRecordOption::WithFreqs),
    //             );
    //         let user_field = builder.add_nested_json_field(vec!["user".into()], nested_opts);
    //         let schema = builder.build();

    //         let index = Index::create_in_ram(schema.clone());
    //         let mut writer = index.writer_for_tests()?;

    //         let doc_obj = json!({
    //             "user": [
    //                 { "age": 10 },
    //                 { "age": 20 }
    //             ]
    //         });

    //         {
    //             let mut document = TantivyDocument::default();
    //             let expanded_docs = parse_json_for_nested_sorted(&schema, &mut document,
    // &doc_obj)                 .expect("parse nested doc");
    //             writer.add_documents(expanded_docs)?;
    //         }
    //         writer.commit()?;

    //         let reader = index.reader()?;
    //         let searcher = reader.searcher();

    //         let query_parser = QueryParser::for_index(&index, vec![user_field]);
    //         let child_query = query_parser.parse_query(r#"age:20"#).unwrap();

    //         let nested_query = NestedQuery::new(
    //             vec!["user".into()],
    //             Box::new(child_query),
    //             ScoreMode::None,
    //             false,
    //         );

    //         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
    //         assert_eq!(
    //             top_docs.len(),
    //             1,
    //             "We should find exactly one parent doc that has a child with age=20"
    //         );

    //         Ok(())
    //     }

    //     fn make_nested_schema_complete() -> (Schema, Field, Field) {
    //         let mut builder = Schema::builder();

    //         let group_field = builder.add_text_field("group", STRING | STORED);

    //         let nested_opts = NestedJsonObjectOptions::new()
    //             .set_include_in_parent(false)
    //             .set_store_parent_flag(true)
    //             .set_stored()
    //             .set_indexing_options(
    //                 TextFieldIndexing::default()
    //                     .set_tokenizer("default")
    //                     .set_index_option(IndexRecordOption::Basic),
    //             );
    //         let user_nested_field = builder.add_nested_json_field(vec!["user".into()],
    // nested_opts);

    //         let schema = builder.build();
    //         (schema, user_nested_field, group_field)
    //     }

    //     fn index_test_document_complete(
    //         index_writer: &mut IndexWriter,
    //         schema: &Schema,
    //         group_val: &str,
    //         users: serde_json::Value,
    //     ) -> Result<(), DocParsingError> {
    //         let full_doc = json!({
    //             "group": group_val,
    //             "user": users,
    //         });

    //         let mut document = TantivyDocument::default();
    //         let expanded = parse_json_for_nested_sorted(schema, &mut document,
    // &full_doc).unwrap();

    //         index_writer.add_documents(expanded).unwrap();

    //         Ok(())
    //     }

    //     #[test]
    //     fn test_nested_query_single_level_complete() -> crate::Result<()> {
    //         let (schema, user_nested_field, _group_field) = make_nested_schema_complete();
    //         let index = Index::create_in_ram(schema.clone());

    //         {
    //             let mut writer = index.writer_for_tests()?;

    //             index_test_document_complete(
    //                 &mut writer,
    //                 &schema,
    //                 "fans",
    //                 json!([
    //                     { "first": "John", "last": "Smith" },
    //                     { "first": "Alice", "last": "White" }
    //                 ]),
    //             )?;

    //             index_test_document_complete(
    //                 &mut writer,
    //                 &schema,
    //                 "boring",
    //                 json!([
    //                     { "first": "Bob", "last": "Marley" }
    //                 ]),
    //             )?;
    //             writer.commit()?;
    //         }

    //         let reader = index.reader()?;
    //         let searcher = reader.searcher();

    //         let query_parser = QueryParser::for_index(&index, vec![user_nested_field]);

    //         let child_query = query_parser.parse_query("first:Alice").unwrap();

    //         let nested_query = NestedQuery::new(
    //             vec!["user".to_string()],
    //             Box::new(child_query),
    //             ScoreMode::Avg,
    //             false,
    //         );

    //         let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
    //         assert_eq!(
    //             top_docs.len(),
    //             1,
    //             "Should match exactly one parent doc with a nested child `first = Alice`."
    //         );

    //         Ok(())
    // }
}

mod explode {
    use common::JsonPathWriter;
    use serde_json::Value as JsonValue;

    use crate::schema::{Field, JsonObjectOptions, ObjectMappingType};
    use crate::TantivyDocument;

    /// Recursively explodes a JSON object (or array) into multiple `TantivyDocument`s,
    /// following Elasticsearch‐style nested semantics.
    ///
    /// Child docs come first, then the final doc for the parent object is last:
    ///
    ///  - If this level is `nested`, the final doc at this level is marked with
    ///    the corresponding `_is_parent_...` boolean field (if any) in your schema.
    ///    For example, if the current path is `["driver_json", "vehicle"]`, we look for
    ///    a boolean field named `"_is_parent_driver_json.vehicle"`.  If found, we set it
    ///    to `true` on that doc.  
    ///
    ///  - Any sub-objects or sub-arrays that are themselves `nested` become child docs.
    ///    Otherwise, they are stored in the parent's JSON as an ordinary non-nested value.
    ///
    ///  - All child docs appear before the final “parent” doc in the returned vector.
    ///
    /// # Arguments
    ///
    /// * `json_val` - The JSON value we’re expanding (object, array, string, number, bool, or null).
    /// * `path` - The current JSON path segments (e.g. `["driver_json", "vehicle"]`).
    /// * `json_field` - The `Field` where we store the JSON text for each doc.
    /// * `schema` - The schema from which we look up any `_is_parent_...` fields.
    /// * `opts` - The `JsonObjectOptions` for this level (indicates whether it’s `nested`, etc.).
    ///
    /// # Returns
    ///
    /// A vector of `TantivyDocument`s.  If `json_val` is a nested array or object, you can get multiple
    /// child docs plus one parent doc at this level.  Otherwise, you usually get a single doc.
    pub fn explode_document(
        json_val: &JsonValue,
        path: &[String],
        json_field: Field,
        schema: &crate::schema::Schema,
        opts: &JsonObjectOptions,
    ) -> Vec<TantivyDocument> {
        // Utility: sets the `_is_parent_...` bool field if `opts` is nested
        // and if such a field exists in the schema.
        fn maybe_mark_parent(
            doc: &mut TantivyDocument,
            path: &[String],
            schema: &crate::schema::Schema,
            opts: &JsonObjectOptions,
        ) {
            if opts.object_mapping_type == ObjectMappingType::Nested {
                // Convert path to "driver_json.vehicle" etc.
                let mut path_builder = JsonPathWriter::new();
                for seg in path {
                    path_builder.push(seg);
                }
                let parent_flag_name = format!("_is_parent_{}", path_builder.as_str());
                // If the field exists, set it to true.
                if let Ok(parent_flag_field) = schema.get_field(&parent_flag_name) {
                    doc.set_is_parent(parent_flag_field, true);
                }
            }
        }

        match json_val {
            //---------------------------------------------------------------------
            // CASE 1) JSON `null`, boolean, number, or string => simple scalar leaf
            //---------------------------------------------------------------------
            JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::String(_) => {
                // We produce exactly 1 doc.  If `opts` indicates `nested`, we mark
                // this doc as the parent for the current path (because there's no bigger “child” structure).
                let mut doc = TantivyDocument::new();
                doc.add_field_value(json_field, json_val);
                // Possibly set `_is_parent_...`
                maybe_mark_parent(&mut doc, path, schema, opts);

                vec![doc]
            }

            //---------------------------------------------------------------------
            // CASE 2) Arrays
            //---------------------------------------------------------------------
            JsonValue::Array(items) => {
                if opts.object_mapping_type == ObjectMappingType::Default {
                    // Not nested => flatten entire array into one doc
                    let mut doc = TantivyDocument::new();
                    doc.add_field_value(json_field, json_val);
                    // Not marking is_parent here, since object_mapping_type=Default
                    vec![doc]
                } else {
                    // This array is nested => produce child docs for each item, then
                    // produce a final doc for the array itself (the parent).
                    let mut docs_out = Vec::new();

                    // Child docs
                    for (i, element) in items.iter().enumerate() {
                        let subpath = path.to_vec();
                        // For clarity in debugging, we indicate array index
                        // subpath.push(format!("[{}]", i));

                        // These docs are children of this array => we do not “re-mark” them
                        // as the parent for *this* path, but they might get their own
                        // parent mark if they themselves are nested deeper.
                        let child_docs =
                            explode_document(element, &subpath, json_field, schema, opts);
                        docs_out.extend(child_docs);
                    }

                    // Now produce the doc for this array itself => marked as parent of `path`.
                    let mut array_doc = TantivyDocument::new();
                    array_doc.add_field_value(json_field, json_val);
                    // Possibly set `_is_parent_...`
                    maybe_mark_parent(&mut array_doc, path, schema, opts);

                    docs_out.push(array_doc);
                    docs_out
                }
            }

            //---------------------------------------------------------------------
            // CASE 3) Objects
            //---------------------------------------------------------------------
            JsonValue::Object(obj_map) => {
                if opts.object_mapping_type == ObjectMappingType::Default {
                    // Flatten entire object => produce 1 doc storing the entire JSON object
                    let mut doc = TantivyDocument::new();
                    doc.add_field_value(json_field, json_val);
                    // Not marking is_parent => not nested
                    vec![doc]
                } else {
                    // If we get here => this is a NESTED object => produce child docs
                    // for any subfields that are themselves marked nested in `opts.subfields`.
                    let mut docs_out = Vec::new();
                    // We'll build a new JSON object to store in our final "parent" doc.
                    // If a property is nested, we move it (or remove it) to child docs
                    // rather than keep it fully in the parent. If `include_in_parent=true`,
                    // we might store a stub like {}. If `include_in_parent=false`, we remove it.
                    let mut parent_obj = obj_map.clone();

                    for (prop_key, prop_val) in obj_map.iter() {
                        if let Some(child_opts) = opts.subfields.get(prop_key) {
                            if child_opts.object_mapping_type == ObjectMappingType::Nested {
                                // produce child docs
                                let mut subpath = path.to_vec();
                                subpath.push(prop_key.to_string());

                                let child_docs = explode_document(
                                    prop_val, &subpath, json_field, schema, child_opts,
                                );
                                docs_out.extend(child_docs);

                                // If we do NOT want to keep the nested property in the parent's JSON,
                                // remove it entirely.  If `include_in_parent=true`, we store a stub.
                                if !child_opts.is_include_in_parent() {
                                    parent_obj.remove(prop_key);
                                } else {
                                    // Replace with an empty object or some stub, so the parent's
                                    // JSON structure remains valid.
                                    parent_obj.insert(
                                        prop_key.clone(),
                                        JsonValue::Object(Default::default()),
                                    );
                                }
                            } else {
                                // This subfield is not nested => keep it in parent's JSON as-is.
                            }
                        } else {
                            // No special subfield config => treat it as non-nested => keep as-is.
                        }
                    }

                    // Now produce our final parent doc at this level => possibly flagged as `_is_parent_...`
                    let final_obj = JsonValue::Object(parent_obj);
                    let mut parent_doc = TantivyDocument::new();
                    parent_doc.add_field_value(json_field, &final_obj);
                    maybe_mark_parent(&mut parent_doc, path, schema, opts);

                    docs_out.push(parent_doc);
                    docs_out
                }
            }
        }
    }

    /// A convenience function to parse the given `json_val` into one or more
    /// `TantivyDocument`s for block‐join style nested indexing.
    ///
    /// - `json_field` is the field where we store the JSON text for each doc.
    /// - We look up the appropriate `_is_parent_...` field from `schema` based
    ///   on the current path.  For the top-level path = `[]`, we expect a field
    ///   named `_is_parent_` (with no suffix) only if that’s how your schema is set up.
    ///   Typically you have a nested path of length 1 or more, e.g. `["driver_json"]`.
    /// - `root_opts` is the `JsonObjectOptions` for the top-level object.
    ///   If `root_opts` is `nested`, the final doc here gets flagged `_is_parent_driver_json= true`
    ///   (assuming you have that field in the schema).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let schema = ...; // your tantivy Schema
    /// let json_field: Field = ...;
    /// let top_opts = JsonObjectOptions::default().set_nested(true, false);
    ///
    /// let doc_val = serde_json::json!({
    ///     "driver": {
    ///         "last_name": "McQueen",
    ///         "vehicle": [
    ///             { "make":"Powell Motors", "model":"Canyonero" },
    ///             { "make":"Miller-Meteor", "model":"Ecto-1" }
    ///         ]
    ///     }
    /// });
    ///
    /// let expanded_docs = parse_json_for_nested(&doc_val["driver"], json_field, &schema, &top_opts);
    /// // Now `expanded_docs` contains multiple child docs plus one final parent doc
    /// // for the "driver" path.  The parent doc is flagged `_is_parent_driver` = true
    /// // if your schema has that field.
    /// ```
    pub fn parse_json_for_nested(
        json_val: &serde_json::Value,
        json_field: Field,
        schema: &crate::schema::Schema,
        root_opts: &JsonObjectOptions,
    ) -> Vec<TantivyDocument> {
        let path = Vec::new(); // top-level path
        explode_document(json_val, &path, json_field, schema, root_opts)
    }

    #[cfg(test)]
    mod tests_explode {
        use super::*;
        use serde_json::json;

        use crate::schema::{
            JsonObjectOptions, NumericOptions, ObjectMappingType, OwnedValue, Schema, SchemaBuilder,
        };
        use crate::TantivyDocument;

        /// Helper to convert a single field's first value into a `serde_json::Value` for test checking.
        fn doc_json_value(
            doc: &TantivyDocument,
            field: crate::schema::Field,
        ) -> Option<serde_json::Value> {
            let comp_val = doc.get_first(field)?;
            let owned_val: OwnedValue = comp_val.into();
            Some(serde_json::to_value(&owned_val).unwrap())
        }

        #[test]
        fn test_explode_scalar_string() {
            let mut schema_builder = SchemaBuilder::default();
            // For path=[] (top-level) we'll define `_is_parent_` (uncommon, but okay).
            let bool_opts = NumericOptions::default().set_indexed().set_stored();
            schema_builder.add_bool_field("_is_parent_", bool_opts);
            let json_field = schema_builder.add_json_field("my_json", JsonObjectOptions::default());
            let schema = schema_builder.build();

            let val = json!("Hello World");
            // If top-level is nested:
            let root_opts = JsonObjectOptions::default().set_nested(true, false);

            let docs = explode_document(&val, &[], json_field, &schema, &root_opts);
            assert_eq!(docs.len(), 1);

            let stored_json = doc_json_value(&docs[0], json_field).unwrap();
            assert_eq!(stored_json, json!("Hello World"));

            // Because it's nested and `_is_parent_` exists, we set is_parent.
            // We can verify the doc has that bool, or just confirm we have no panics.
        }

        #[test]
        fn test_explode_scalar_number_not_nested() {
            // no parent flags at all
            let schema = Schema::builder().build();
            let val = json!(123);

            let root_opts = JsonObjectOptions::default().set_nested(false, false);
            let docs = explode_document(
                &val,
                &[],
                crate::schema::Field::from_field_id(0),
                &schema,
                &root_opts,
            );
            assert_eq!(docs.len(), 1);

            let doc = &docs[0];
            let stored: OwnedValue = doc
                .get_first(crate::schema::Field::from_field_id(0))
                .unwrap()
                .into();
            assert_eq!(stored, OwnedValue::I64(123));
        }

        #[test]
        fn test_explode_array_not_nested() {
            let schema = Schema::builder().build();
            let val = json!(["Alpha", "Bravo"]);

            let root_opts = JsonObjectOptions::default(); // not nested
            let docs = explode_document(
                &val,
                &[],
                crate::schema::Field::from_field_id(0),
                &schema,
                &root_opts,
            );

            // single doc => entire array
            assert_eq!(docs.len(), 1);

            let stored_json =
                doc_json_value(&docs[0], crate::schema::Field::from_field_id(0)).unwrap();
            assert_eq!(stored_json, json!(["Alpha", "Bravo"]));
        }

        #[test]
        fn test_explode_array_nested() {
            let mut sb = SchemaBuilder::default();
            // For path=["myArray"], define `_is_parent_myArray`
            let bool_opts = NumericOptions::default().set_stored().set_indexed();
            sb.add_bool_field("_is_parent_myArray", bool_opts);
            let json_field = sb.add_json_field("js", JsonObjectOptions::default());
            let schema = sb.build();

            let val = json!(["Alpha", "Bravo"]);
            let mut root_opts = JsonObjectOptions::default().set_nested(true, false);
            // The path is `["myArray"]`.
            let docs = explode_document(&val, &["myArray".into()], json_field, &schema, &root_opts);
            // Expect 3 docs: child(A), child(B), parent(array)
            assert_eq!(docs.len(), 3);

            // doc0 => "Alpha"
            let d0 = doc_json_value(&docs[0], json_field).unwrap();
            assert_eq!(d0, json!("Alpha"));

            // doc1 => "Bravo"
            let d1 = doc_json_value(&docs[1], json_field).unwrap();
            assert_eq!(d1, json!("Bravo"));

            // doc2 => entire array, flagged `_is_parent_myArray`
            let d2 = doc_json_value(&docs[2], json_field).unwrap();
            assert_eq!(d2, json!(["Alpha", "Bravo"]));
        }

        #[test]
        fn test_explode_object_not_nested() {
            let schema = Schema::builder().build();
            let val = json!({"key1": "Value1", "key2": 123});
            let root_opts = JsonObjectOptions::default();

            let docs = explode_document(
                &val,
                &[],
                crate::schema::Field::from_field_id(0),
                &schema,
                &root_opts,
            );
            // single doc => entire object
            assert_eq!(docs.len(), 1);

            let stored_json =
                doc_json_value(&docs[0], crate::schema::Field::from_field_id(0)).unwrap();
            assert_eq!(stored_json, val);
        }

        #[test]
        fn test_explode_object_nested_subfield() {
            // top-level path = "driver"; subfield path = "driver.vehicle"
            let mut sb = SchemaBuilder::default();
            let bool_opts = NumericOptions::default().set_stored().set_indexed();
            // We'll define these so the parent docs get flagged.
            sb.add_bool_field("_is_parent_driver", bool_opts.clone());
            sb.add_bool_field("_is_parent_driver.vehicle", bool_opts);

            let json_field = sb.add_json_field("js", JsonObjectOptions::default());
            let schema = sb.build();

            let mut driver_opts = JsonObjectOptions::default().set_nested(true, false);
            driver_opts.subfields.insert(
                "vehicle".to_string(),
                JsonObjectOptions::default().set_nested(true, false),
            );

            let val = json!({
                "last_name": "McQueen",
                "vehicle": [
                    {"make":"Powell Motors","model":"Canyonero"},
                    {"make":"Miller-Meteor","model":"Ecto-1"}
                ]
            });

            let docs =
                explode_document(&val, &["driver".into()], json_field, &schema, &driver_opts);

            // We expect:
            //  - child docs for each array item
            //  - doc for entire array => `_is_parent_driver.vehicle`
            //  - doc for the object => `_is_parent_driver`
            assert_eq!(docs.len(), 4);

            // doc0 => first child => make=Powell
            let doc0_json = doc_json_value(&docs[0], json_field).unwrap();
            assert_eq!(
                doc0_json,
                json!({"make":"Powell Motors","model":"Canyonero"})
            );

            // doc1 => second child => Ecto-1
            let doc1_json = doc_json_value(&docs[1], json_field).unwrap();
            assert_eq!(doc1_json, json!({"make":"Miller-Meteor","model":"Ecto-1"}));

            // doc2 => the array => `_is_parent_driver.vehicle`
            let doc2_json = doc_json_value(&docs[2], json_field).unwrap();
            assert_eq!(
                doc2_json,
                json!([
                    {"make":"Powell Motors","model":"Canyonero"},
                    {"make":"Miller-Meteor","model":"Ecto-1"}
                ])
            );

            // doc3 => the "driver" object => `_is_parent_driver`
            // Because `include_in_parent=true` by default, "vehicle" is replaced by {}
            let doc3_json = doc_json_value(&docs[3], json_field).unwrap();
            assert_eq!(doc3_json, json!({"last_name":"McQueen","vehicle":{}}));
        }

        #[test]
        fn test_explode_multi_level_nested() {
            // top-level path = ["outer"], but it’s not nested
            // inside that, path=["outer","driver"] => nested
            // inside that, path=["outer","driver","vehicle"] => also nested
            //
            // We'll define `_is_parent_outer.driver` and `_is_parent_outer.driver.vehicle`.
            let mut sb = SchemaBuilder::default();
            let bool_opts = NumericOptions::default().set_stored().set_indexed();
            sb.add_bool_field("_is_parent_outer.driver", bool_opts.clone());
            sb.add_bool_field("_is_parent_outer.driver.vehicle", bool_opts);

            let json_field = sb.add_json_field("json", JsonObjectOptions::default());
            let schema = sb.build();

            // `outer` is not nested => `outer` has a subfield "driver" which is nested.
            let mut outer_opts = JsonObjectOptions::default().set_nested(false, false);
            let mut driver_opts = JsonObjectOptions::default().set_nested(true, false);
            driver_opts.subfields.insert(
                "vehicle".into(),
                JsonObjectOptions::default().set_nested(true, false),
            );
            outer_opts.subfields.insert("driver".into(), driver_opts);

            let big_json = json!({
                "driver": {
                    "last_name": "McQueen",
                    "vehicle": [
                        {"make": "Powell Motors", "model": "Canyonero"},
                        {"make": "Miller-Meteor", "model": "Ecto-1"}
                    ]
                }
            });

            // Because top-level path is ["outer"] => we call explode_document
            let docs = explode_document(
                &big_json,
                &["outer".into()],
                json_field,
                &schema,
                &outer_opts,
            );
            // Explanation:
            //  - outer is not nested => it yields 1 doc (the entire object)
            //  - But subfield "driver" is nested => we produce child docs for driver.vehicle
            //    => 2 children + 1 array doc => `_is_parent_outer.driver.vehicle`
            //    => 1 doc for driver => `_is_parent_outer.driver`
            //  Finally, the top-level doc for "outer" remains unflagged.
            //
            // So we expect 1 (child doc for 1st car) + 1 (child doc for 2nd car)
            // + 1 (vehicle array) + 1 (driver object) + 1 (outer object)
            // = 5 docs total:
            assert_eq!(docs.len(), 5);

            // doc0 => first vehicle => "Canyonero"
            // doc1 => second vehicle => "Ecto-1"
            // doc2 => entire vehicle array => `_is_parent_outer.driver.vehicle`
            // doc3 => driver object => `_is_parent_outer.driver`
            // doc4 => outer object => no flags
            let d4 = doc_json_value(&docs[4], json_field).unwrap();
            assert_eq!(
                d4,
                json!({ "driver": {} }),
                "The final doc is the entire top-level object"
            );
        }

        #[test]
        fn test_parse_json_for_nested_example() {
            // Exactly the same logic, but using parse_json_for_nested at the top-level
            let mut sb = SchemaBuilder::default();
            let bool_opts = NumericOptions::default().set_stored().set_indexed();
            sb.add_bool_field("_is_parent_", bool_opts);
            let json_field = sb.add_json_field("j", JsonObjectOptions::default());
            let schema = sb.build();

            let top_opts = JsonObjectOptions::default().set_nested(true, false);
            let doc_val = json!({"hello": "world"});
            let expanded = parse_json_for_nested(&doc_val, json_field, &schema, &top_opts);
            assert_eq!(expanded.len(), 1);

            let stored_json = doc_json_value(&expanded[0], json_field).unwrap();
            assert_eq!(stored_json, json!({"hello":"world"}));
            // flagged `_is_parent_` presumably
        }

        #[test]
        fn test_nested_query_include_in_parent_false() {
            // We want to remove the nested field from the parent's JSON.
            let mut sb = SchemaBuilder::default();
            let bool_opts = NumericOptions::default().set_stored().set_indexed();
            sb.add_bool_field("_is_parent_stuff", bool_opts);
            let json_field = sb.add_json_field("root", JsonObjectOptions::default());
            let schema = sb.build();

            let mut root_opts = JsonObjectOptions::default().set_nested(false, false);
            let child_opts = JsonObjectOptions::default().set_nested(false, false);
            root_opts.subfields.insert("stuff".into(), child_opts);

            let val = json!({
                "stuff": ["A","B"],
                "other": 123
            });

            let docs = explode_document(&val, &[], json_field, &schema, &root_opts);
            // Explanation:
            //  - child doc => "A"
            //  - child doc => "B"
            //  - doc => entire array => `_is_parent_stuff`
            //  - final doc => top-level => removing "stuff", leaving {"other":123}
            assert_eq!(docs.len(), 4);

            let d0 = doc_json_value(&docs[0], json_field).unwrap();
            assert_eq!(d0, json!("A"));

            let d1 = doc_json_value(&docs[1], json_field).unwrap();
            assert_eq!(d1, json!("B"));

            let d2 = doc_json_value(&docs[2], json_field).unwrap();
            assert_eq!(d2, json!(["A", "B"]));

            let d3 = doc_json_value(&docs[3], json_field).unwrap();
            assert_eq!(d3, json!({"other":123}));
        }
    }
}
