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
    use crate::query::{AllQuery, BooleanQuery, NestedQuery, QueryClone, QueryParser, ScoreMode};
    use crate::schema::{
        IndexRecordOption, JsonObjectOptions, Schema, SchemaBuilder, TextFieldIndexing,
    };
    use crate::{doc, Index, IndexWriter, TantivyDocument};
    use query_grammar::Occur;
    use serde_json::json;

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
    fn test_nested_query_scenario_small() -> crate::Result<()> {
        let mut schema_builder = SchemaBuilder::new();

        let driver_json_opts = JsonObjectOptions::nested()
            .set_indexing_options(TextFieldIndexing::default())
            .add_subfield("vehicle", JsonObjectOptions::nested());

        let driver_field =
            schema_builder.add_nested_json_field("driver_json", driver_json_opts.clone());

        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut writer: IndexWriter<TantivyDocument> =
            index.writer_with_num_threads(2, 50_000_000)?;

        let big_json = json!({
            "last_name": "McQueen",
            "vehicle": [
                { "make": "Powell", "model": "Canyonero" },
                { "make": "Miller-Meteor", "model": "Ecto" }
            ]
        });

        let mut parent_doc = TantivyDocument::default();
        let mut child_docs = parent_doc
            .add_nested_object(&schema, driver_field, big_json, &driver_json_opts)
            .unwrap();

        child_docs.push(parent_doc);

        assert_eq!(child_docs.len(), 4, "should be 4 exploded docs");

        writer.add_documents(child_docs)?;
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

        let non_nested_query = query_parser
            .parse_query("driver_json.vehicle.model:Canyonero")
            .unwrap();
        let top_docs = searcher.search(&non_nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 1, "Expected 1 parent doc");
        assert_eq!(top_docs[0].1.doc_id, 3);

        let parent_query = query_parser.parse_query("last_name:McQueen").unwrap();
        let top_docs = searcher.search(&parent_query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 1, "Expected one doc with parent data");
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
        let driver_json_opts = JsonObjectOptions::nested()
            .set_indexing_options(TextFieldIndexing::default())
            .add_subfield("vehicle", JsonObjectOptions::nested());

        let driver_field =
            schema_builder.add_nested_json_field("driver_json", driver_json_opts.clone());

        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
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

        let mut parent_doc = TantivyDocument::default();
        let mut child_docs = parent_doc
            .add_nested_object(&schema, driver_field, big_json, &driver_json_opts)
            .unwrap();

        child_docs.push(parent_doc);

        writer.add_documents(child_docs)?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&index, vec![driver_field]);

        let query = query_parser
            .parse_query("driver_json.bicycle.color:red")
            .unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 1, "Parent doc with red bicycle");
        assert_eq!(top_docs[0].1.doc_id, 3, "Returned doc is the parent (#2)");

        let query = query_parser
            .parse_query("driver_json.bicycle.color:red AND driver_json.bicycle.gears:1")
            .unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 1, "Parent doc has color:red and gears:1");

        let bicycle_query = query_parser
            .parse_query("driver_json.bicycle.color:red")
            .unwrap();
        let child_q = query_parser
            .parse_query("driver_json.vehicle.model:Canyonero AND driver_json.vehicle.make:Ecto")
            .unwrap();

        let nested_query = NestedQuery::new(
            vec!["driver_json".into()],
            Box::new(NestedQuery::new(
                vec!["driver_json".into(), "vehicle".into()],
                Box::new(child_q),
                ScoreMode::Avg,
                false,
            )),
            ScoreMode::Avg,
            false,
        );
        // We'll combine them in a BooleanQuery
        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(top_docs.len(), 0, "No single doc matches that combination");

        // We'll combine them in a BooleanQuery
        let top_docs = searcher.search(
            &BooleanQuery::intersection(vec![Box::new(bicycle_query), Box::new(nested_query)]),
            &TopDocs::with_limit(10),
        )?;
        assert_eq!(top_docs.len(), 0, "No single doc matches that combination");

        Ok(())
    }

    #[test]
    fn test_multi_level_nested_scenario() -> crate::Result<()> {
        let driver_json_opts = JsonObjectOptions::nested()
            .set_indexing_options(TextFieldIndexing::default().set_tokenizer("raw"))
            .add_subfield("vehicle", JsonObjectOptions::nested())
            .add_subfield(
                "crew",
                JsonObjectOptions::nested().add_subfield("kids", JsonObjectOptions::nested()),
            );

        let mut schema_builder = SchemaBuilder::new();
        let driver_field =
            schema_builder.add_nested_json_field("driver_json", driver_json_opts.clone());

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
        let mut parent_doc = TantivyDocument::default();
        let mut child_docs = parent_doc
            .add_nested_object(&schema, driver_field, big_json, &driver_json_opts)
            .unwrap();

        child_docs.push(parent_doc);

        writer.add_documents(child_docs)?;
        writer.commit()?;

        // 4) Query them
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&index, vec![driver_field]);

        let child_q1 = query_parser
            .parse_query("driver_json.crew.role:mechanic AND driver_json.crew.kids.name:Eve")?;
        let top_docs1 = searcher.search(&child_q1, &TopDocs::with_limit(10))?;
        assert_eq!(1, top_docs1.len(), "Grandparent matches non-nested query");

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
        let top_docs2 = searcher.search(&nested_q2, &TopDocs::with_limit(10))?;
        assert_eq!(
            0,
            top_docs2.len(),
            "No single crew entry has both mechanic + kid=Eve"
        );

        let child_q3 = query_parser
            .parse_query("driver_json.crew.role:mechanic AND driver_json.crew.kids.name:Eve")?;
        let nested_q3 = NestedQuery::new(
            vec![
                "driver_json".to_string(),
                "crew".to_string(),
                "kids".to_string(),
            ],
            Box::new(child_q3),
            ScoreMode::Avg,
            false,
        );
        let top_docs3 = searcher.search(&nested_q3, &TopDocs::with_limit(10))?;
        assert_eq!(
            0,
            top_docs3.len(),
            "No single crew entry has both mechanic + kid=Eve"
        );

        let plain_q = query_parser.parse_query("driver_json.last_name:McQueen")?;
        let top_docs_plain = searcher.search(&plain_q, &TopDocs::with_limit(10))?;
        assert_eq!(1, top_docs_plain.len(), "Should match on last_name=McQueen");

        Ok(())
    }

    #[test]
    fn test_nested_query_two_docs() -> crate::Result<()> {
        let driver_json_opts = JsonObjectOptions::nested()
            .set_indexing_options(TextFieldIndexing::default())
            .add_subfield(
                "driver",
                JsonObjectOptions::nested().add_subfield("vehicle", JsonObjectOptions::nested()),
            );

        let mut schema_builder = SchemaBuilder::new();
        let driver_field =
            schema_builder.add_nested_json_field("nested_data", driver_json_opts.clone());

        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());

        let doc1 = json!({
            "driver": {
                "last_name": "McQueen",
                "vehicle": [
                    { "make": "Powell Motors", "model": "Canyonero" },
                    { "make": "Miller-Meteor", "model": "Ecto-1" }
                ]
            }
        });

        let doc2 = json!({
            "driver": {
                "last_name": "Hudson",
                "vehicle": [
                    { "make": "Mifune", "model": "Mach Five" },
                    { "make": "Miller-Meteor", "model": "Ecto-1" }
                ]
            }
        });

        let mut writer: IndexWriter<TantivyDocument> =
            index.writer_with_num_threads(2, 50_000_000)?;
        {
            let mut parent_doc1 = TantivyDocument::default();
            let mut exploded_children = parent_doc1
                .add_nested_object(&schema, driver_field, doc1, &driver_json_opts)
                .unwrap();
            exploded_children.push(parent_doc1);

            writer.add_documents(exploded_children)?;
        }

        {
            let mut parent_doc2 = TantivyDocument::default();
            let mut exploded_children = parent_doc2
                .add_nested_object(&schema, driver_field, doc2, &driver_json_opts)
                .unwrap();
            exploded_children.push(parent_doc2);

            writer.add_documents(exploded_children)?;
        }

        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![driver_field]);

        let child_query_a = query_parser.parse_query(
            "nested_data.driver.vehicle.make:powell AND nested_data.driver.vehicle.model:canyonero",
        )?;
        let nested_query_a = NestedQuery::new(
            vec!["nested_data".to_string()],
            Box::new(NestedQuery::new(
                vec!["nested_data".to_string(), "driver".to_string()],
                Box::new(NestedQuery::new(
                    vec![
                        "nested_data".to_string(),
                        "driver".to_string(),
                        "vehicle".to_string(),
                    ],
                    Box::new(child_query_a),
                    ScoreMode::Avg,
                    false,
                )),
                ScoreMode::Avg,
                false,
            )),
            ScoreMode::Avg,
            false,
        );

        let top_docs_a = searcher.search(&nested_query_a, &TopDocs::with_limit(10))?;

        assert_eq!(
            1,
            top_docs_a.len(),
            "Only doc #1 should match Powell + Canyonero"
        );

        let child_query_b = query_parser.parse_query(
            "nested_data.driver.vehicle.make:miller AND nested_data.driver.vehicle.model:ecto",
        )?;
        let nested_query_b = NestedQuery::new(
            vec!["nested_data".to_string()],
            Box::new(NestedQuery::new(
                vec!["nested_data".to_string(), "driver".to_string()],
                Box::new(NestedQuery::new(
                    vec![
                        "nested_data".to_string(),
                        "driver".to_string(),
                        "vehicle".to_string(),
                    ],
                    Box::new(child_query_b),
                    ScoreMode::Avg,
                    false,
                )),
                ScoreMode::Avg,
                false,
            )),
            ScoreMode::Avg,
            false,
        );

        let top_docs_b = searcher.search(&nested_query_b, &TopDocs::with_limit(10))?;
        assert_eq!(
            2,
            top_docs_b.len(),
            "Both doc #1 and doc #2 should match Miller-Meteor / Ecto-1"
        );

        let plain_query = query_parser.parse_query("driver.last_name:Hudson")?;
        let top_docs_plain = searcher.search(&plain_query, &TopDocs::with_limit(10))?;
        assert_eq!(1, top_docs_plain.len(), "One doc with last_name=Hudson");

        Ok(())
    }

    #[test]
    fn test_nested_must_not_clause() -> crate::Result<()> {
        let posts_opts = JsonObjectOptions::nested()
            .set_indexing_options(TextFieldIndexing::default().set_tokenizer("raw"))
            .add_subfield("comments", JsonObjectOptions::nested());

        let mut schema_builder = SchemaBuilder::new();
        let posts_field = schema_builder.add_nested_json_field("posts", posts_opts.clone());

        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut writer: IndexWriter<TantivyDocument> =
            index.writer_with_num_threads(1, 50_000_000)?;

        // --- Doc #1
        {
            let doc1_json = json!({
                "comments": [
                    {"author": "kimchy"}
                ]
            });
            let mut parent_doc1 = TantivyDocument::default();
            let mut exploded1 = parent_doc1
                .add_nested_object(&schema, posts_field, doc1_json, &posts_opts)
                .unwrap();
            exploded1.push(parent_doc1);
            writer.add_documents(exploded1)?;
        }

        // --- Doc #2
        {
            let doc2_json = json!({
                "comments": [
                    {"author": "kimchy"},
                    {"author": "nik9000"}
                ]
            });
            let mut parent_doc2 = TantivyDocument::default();
            let mut exploded2 = parent_doc2
                .add_nested_object(&schema, posts_field, doc2_json, &posts_opts)
                .unwrap();
            exploded2.push(parent_doc2);
            writer.add_documents(exploded2)?;
        }

        // --- Doc #3
        {
            let doc3_json = json!({
                "comments": [
                    {"author": "nik9000"}
                ]
            });
            let mut parent_doc3 = TantivyDocument::default();
            let mut exploded3 = parent_doc3
                .add_nested_object(&schema, posts_field, doc3_json, &posts_opts)
                .unwrap();
            exploded3.push(parent_doc3);
            writer.add_documents(exploded3)?;
        }

        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&index, vec![posts_field]);

        // ========================================================
        // (A) Nested "must_not => nik9000" at the child level
        //
        // This picks any doc that has at least one child subdoc
        // *not* matching "author=nik9000".
        //
        // We do that by building a BooleanQuery with:
        //   MUST: [AllQuery] (i.e., any subdoc)
        //   MUST_NOT: [nik9000]
        //
        // Then we wrap that in NestedQuery(path=["comments"]).
        // So a doc matches if it has at least one comment child where
        // "author != nik9000".
        //
        // Expect: Doc #1 (kimchy) and Doc #2 (kimchy + nik9000).
        //         Doc #2 qualifies via the "kimchy" subdocument.
        //         Doc #3 does NOT qualify because its only subdoc is "nik9000".
        // ========================================================

        let nested_inner = NestedQuery::new(
            vec!["posts".into()],
            Box::new(NestedQuery::new(
                vec!["posts".into(), "comments".into()],
                Box::new(BooleanQuery::new(vec![
                    (Occur::Must, AllQuery.box_clone()),
                    (
                        Occur::MustNot,
                        query_parser
                            .parse_query("posts.comments.author:nik9000")?
                            .box_clone(),
                    ),
                ])),
                ScoreMode::Avg,
                false,
            )),
            ScoreMode::Avg,
            false,
        );

        let top_docs_inner = searcher.search(&nested_inner, &TopDocs::with_limit(10))?;
        let doc_ids_inner: Vec<u32> = top_docs_inner
            .iter()
            .map(|(_, docaddr)| docaddr.doc_id)
            .collect();

        // We expect doc_ids_inner to have 2 hits (Doc #1 and #2).
        // The actual doc_id values in the segment can vary, so we check len().
        assert_eq!(
            2,
            doc_ids_inner.len(),
            "Docs #1 and #2 should match must_not => 'nik9000' at child-level"
        );

        // ========================================================
        // (B) Place the must_not at the *outer* level
        //
        // We want to exclude any doc that has ANY child subdoc with "author=nik9000".
        // So we do:
        //   MUST: [AllQuery]   // match all docs
        //   MUST_NOT: [ NestedQuery(path=["comments"], child=TermQuery(nik9000)) ]
        //
        // That means "throw away any doc that has a comment subdoc with 'nik9000'."
        //
        // Expect: Only Doc #1 remains. (Docs #2 and #3 both contain 'nik9000' subdocs.)
        // ========================================================
        let bool_outer = BooleanQuery::new(vec![
            (Occur::Must, AllQuery.box_clone()),
            (
                Occur::MustNot,
                Box::new(NestedQuery::new(
                    vec!["posts".into()],
                    NestedQuery::new(
                        vec!["posts".into(), "comments".into()],
                        query_parser
                            .parse_query("posts.comments.author:nik9000")?
                            .box_clone(),
                        ScoreMode::Avg,
                        false,
                    )
                    .box_clone(),
                    ScoreMode::Avg,
                    false,
                )),
            ),
        ]);

        let top_docs_outer = searcher.search(&bool_outer, &TopDocs::with_limit(10))?;
        let doc_ids_outer: Vec<u32> = top_docs_outer
            .iter()
            .map(|(_, docaddr)| docaddr.doc_id)
            .collect();

        // We expect only 1 doc to remain: the doc that has no "nik9000" at all.
        // That’s doc #1 in our data set.
        assert_eq!(
            1,
            doc_ids_outer.len(),
            "Only doc #1 remains, because docs #2 and #3 contain 'nik9000'"
        );

        Ok(())
    }
}
