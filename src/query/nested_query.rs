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

    use serde_json::json;

    use super::*;
    use crate::collector::TopDocs;
    use crate::query::nested_query::ScoreMode;
    use crate::query::{AllQuery, QueryClone, QueryParser, TermQuery};
    use crate::schema::document::parse_json_for_nested_sorted;
    use crate::schema::{
        DocParsingError, Field, IndexRecordOption, NestedJsonObjectOptions, Schema, SchemaBuilder,
        TextFieldIndexing, Value, STORED, STRING, TEXT,
    };
    use crate::{Index, IndexWriter, TantivyDocument, Term};

    fn make_schema_for_eq_tests() -> (Schema, Field, Field) {
        let mut builder = SchemaBuilder::default();

        let group_f = builder.add_text_field("group", STRING | STORED);

        let nested_opts = NestedJsonObjectOptions::new()
            .set_include_in_parent(false)
            .set_store_parent_flag(true)
            .set_stored()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::Basic),
            );
        let user_nested_f = builder.add_nested_json_field(vec!["user".into()], nested_opts);

        let schema = builder.build();
        (schema, user_nested_f, group_f)
    }

    fn index_doc_for_eq_tests(
        index_writer: &mut crate::indexer::IndexWriter,
        schema: &Schema,
        group_val: &str,
        user_array: serde_json::Value,
    ) {
        let top_obj = json!({
            "group": group_val,
            "user": user_array
        });
        let mut document = TantivyDocument::default();
        let expanded = parse_json_for_nested_sorted(schema, &mut document, &top_obj).unwrap();

        index_writer.add_documents(expanded).unwrap();
    }

    #[test]
    fn test_ignore_unmapped_true() {
        let (schema, _user_nested_f, _group_f) = make_schema_for_eq_tests();
        let index = Index::create_in_ram(schema.clone());

        {
            let mut writer = index.writer_for_tests().unwrap();
            index_doc_for_eq_tests(
                &mut writer,
                &schema,
                "someGroup",
                json!([
                    { "first": "Bob", "last": "Smith" }
                ]),
            );
            writer.commit().unwrap();
        }

        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![]);
        let child_q = query_parser.parse_query("user.first:Bob").unwrap();

        let nested_q = NestedQuery::new(
            vec!["unmapped".into()],
            Box::new(child_q),
            ScoreMode::None,
            true,
        );

        let top_docs = searcher
            .search(&nested_q, &TopDocs::with_limit(10))
            .unwrap();
        assert_eq!(
            top_docs.len(),
            0,
            "Expected zero hits for ignore_unmapped=true + unknown path"
        );
    }

    #[test]
    fn test_ignore_unmapped_false_error() {
        let (schema, _user_nested_f, _group_f) = make_schema_for_eq_tests();
        let index = Index::create_in_ram(schema.clone());

        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![]);
        let child_q = query_parser.parse_query("user.first:Anything").unwrap();
        let nested_q = NestedQuery::new(
            vec!["unmapped".into()],
            Box::new(child_q),
            ScoreMode::None,
            false,
        );

        let result = searcher.search(&nested_q, &TopDocs::with_limit(10));
        match result {
            Ok(_) => panic!("Expected an error for path=unmapped + ignore_unmapped=false"),
            Err(e) => {
                let msg = format!("{:?}", e);
                assert!(msg.contains("not mapped") && !msg.contains("ignore_unmapped=true"));
            }
        }
    }

    #[test]
    fn test_nested_query_some_match() -> crate::Result<()> {
        let (schema, user_nested_f, _group_f) = make_schema_for_eq_tests();
        let index = Index::create_in_ram(schema.clone());

        {
            let mut writer = index.writer_for_tests()?;
            index_doc_for_eq_tests(
                &mut writer,
                &schema,
                "fans",
                json!([
                    {"first":"Bob","last":"Smith"},
                    {"first":"Alice","last":"Branson"}
                ]),
            );
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![user_nested_f]);
        let child_q = query_parser.parse_query("first:Alice").unwrap();

        let nested_q = NestedQuery::new(
            vec!["user".into()],
            Box::new(child_q),
            ScoreMode::Avg,
            false,
        );

        let top_docs = searcher.search(&nested_q, &TopDocs::with_limit(10))?;
        assert_eq!(1, top_docs.len());
        Ok(())
    }

    #[test]
    fn test_nested_score_modes() -> crate::Result<()> {
        let (schema, nested_f, _group_f) = make_schema_for_eq_tests();
        let index = Index::create_in_ram(schema.clone());

        {
            let mut writer = index.writer_for_tests()?;
            index_doc_for_eq_tests(
                &mut writer,
                &schema,
                "someGroup",
                json!([
                    {"first":"java"},
                    {"first":"java"},
                    {"first":"rust"}
                ]),
            );
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![nested_f]);
        let child_q = query_parser.parse_query("first:java").unwrap();
        for &mode in &[
            ScoreMode::None,
            ScoreMode::Total,
            ScoreMode::Avg,
            ScoreMode::Max,
            ScoreMode::Min,
        ] {
            let nested_q = NestedQuery::new(
                vec!["user".into()],
                Box::new(child_q.box_clone()),
                mode,
                false,
            );
            let hits = searcher.search(&nested_q, &TopDocs::with_limit(10))?;
            assert_eq!(1, hits.len());
        }
        Ok(())
    }

    #[test]
    fn test_nested_score_mode_parsing() {
        assert_eq!(ScoreMode::from_str("none").unwrap(), ScoreMode::None);
        assert_eq!(ScoreMode::from_str("avg").unwrap(), ScoreMode::Avg);
        assert_eq!(ScoreMode::from_str("max").unwrap(), ScoreMode::Max);
        assert_eq!(ScoreMode::from_str("min").unwrap(), ScoreMode::Min);
        assert_eq!(ScoreMode::from_str("total").unwrap(), ScoreMode::Total);

        let err = ScoreMode::from_str("garbage").unwrap_err();
        assert!(err.to_string().contains("Unrecognized nested score_mode"));
    }

    fn make_multi_level_schema() -> (Schema, Field, Field, Field) {
        let mut builder = Schema::builder();

        let doc_tag_field = builder.add_text_field("doc_tag", STRING | STORED);

        let driver_nested_opts = NestedJsonObjectOptions::new()
            .set_include_in_parent(false)
            .set_store_parent_flag(true)
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_index_option(IndexRecordOption::Basic),
            );
        let driver_field = builder.add_nested_json_field(vec!["driver".into()], driver_nested_opts);

        let vehicle_nested_opts = NestedJsonObjectOptions::new()
            .set_include_in_parent(false)
            .set_store_parent_flag(true)
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_index_option(IndexRecordOption::Basic),
            );
        let vehicle_field = builder
            .add_nested_json_field(vec!["driver".into(), "vehicle".into()], vehicle_nested_opts);

        let schema = builder.build();
        (schema, doc_tag_field, driver_field, vehicle_field)
    }

    fn index_doc_multi_level(
        writer: &mut IndexWriter,
        schema: &Schema,
        doc_tag: &str,
        last_name: &str,
        vehicles: serde_json::Value,
    ) -> Result<(), DocParsingError> {
        let doc_obj = json!({
            "doc_tag": doc_tag,
            "driver": {
                "last_name": last_name,
                "vehicle": vehicles
            }
        });
        let json_doc = serde_json::to_string(&doc_obj).unwrap();
        let mut document = TantivyDocument::default();
        let expanded = parse_json_for_nested_sorted(
            schema,
            &mut document,
            &serde_json::from_str::<serde_json::Value>(&json_doc).unwrap(),
        )
        .unwrap();
        writer.add_documents(expanded).unwrap();
        Ok(())
    }

    #[test]
    fn test_multi_level_nested_query() -> crate::Result<()> {
        let (schema, _doc_tag_field, driver_field, vehicle_field) = make_multi_level_schema();

        let index = Index::create_in_ram(schema.clone());
        {
            let mut writer = index.writer_for_tests()?;

            index_doc_multi_level(
                &mut writer,
                &schema,
                "Doc1",
                "McQueen",
                json!([
                    { "make":"Powell Motors", "model":"Canyonero"},
                    { "make":"Miller-Meteor", "model":"Ecto-1"}
                ]),
            )?;

            index_doc_multi_level(
                &mut writer,
                &schema,
                "Doc2",
                "Hudson",
                json!([
                    { "make":"Mifune", "model":"Mach Five" },
                    { "make":"Miller-Meteor", "model":"Ecto-1" }
                ]),
            )?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![vehicle_field, driver_field]);
        let make_q = query_parser.parse_query("make:\"Powell Motors\"").unwrap();
        let model_q = query_parser.parse_query("model:Canyonero").unwrap();
        let bool_sub = BooleanQuery::new(vec![
            (Occur::Must, Box::new(make_q)),
            (Occur::Must, Box::new(model_q)),
        ]);

        let vehicle_nested = NestedQuery::new(
            vec!["driver".into(), "vehicle".into()],
            Box::new(bool_sub),
            ScoreMode::Avg,
            false,
        );
        let driver_nested = NestedQuery::new(
            vec!["driver".into()],
            Box::new(vehicle_nested),
            ScoreMode::Avg,
            false,
        );

        let hits = searcher.search(&driver_nested, &TopDocs::with_limit(10))?;
        assert_eq!(0, hits.len(), "multi-level nesting is still a todo!");

        Ok(())
    }

    fn make_comments_schema() -> (Schema, Field, Field, Field) {
        let mut builder = Schema::builder();

        let doc_num_field = builder.add_text_field("doc_num", STRING | STORED);

        let nested_opts = NestedJsonObjectOptions::new()
            .set_include_in_parent(false)
            .set_store_parent_flag(true);
        let comments_field = builder.add_nested_json_field(vec!["comments".into()], nested_opts);

        let author_field = builder.add_text_field("comments.author", STRING);

        let schema = builder.build();
        (schema, doc_num_field, comments_field, author_field)
    }

    fn index_doc_with_comments(
        writer: &mut IndexWriter,
        schema: &Schema,
        doc_num: &str,
        comments: serde_json::Value,
    ) -> Result<(), DocParsingError> {
        let doc_obj = json!({
            "doc_num": doc_num,
            "comments": comments
        });
        let json_doc = serde_json::to_string(&doc_obj).unwrap();
        let mut document = TantivyDocument::default();
        let expanded = parse_json_for_nested_sorted(
            schema,
            &mut document,
            &serde_json::from_str::<serde_json::Value>(&json_doc).unwrap(),
        )
        .unwrap();
        writer.add_documents(expanded).unwrap();
        Ok(())
    }

    #[test]
    #[ignore = "must not queries not working to block join semantics"]
    fn test_comments_must_not_nested() -> crate::Result<()> {
        let (schema, doc_num_f, _comments_f, author_f) = make_comments_schema();
        let index = Index::create_in_ram(schema.clone());

        {
            let mut writer = index.writer_for_tests()?;
            index_doc_with_comments(
                &mut writer,
                &schema,
                "1",
                json!([
                    {"author":"kimchy"}
                ]),
            )?;
            index_doc_with_comments(
                &mut writer,
                &schema,
                "2",
                json!([
                    {"author":"kimchy"},
                    {"author":"nik9000"}
                ]),
            )?;
            index_doc_with_comments(
                &mut writer,
                &schema,
                "3",
                json!([
                    {"author":"nik9000"}
                ]),
            )?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        use crate::query::{BooleanQuery, Occur};

        let tq_nik = TermQuery::new(
            Term::from_field_text(author_f, "nik9000"),
            IndexRecordOption::Basic,
        );
        let must_not = BooleanQuery::new(vec![
            (Occur::Must, Box::new(AllQuery)),
            (Occur::MustNot, Box::new(tq_nik)),
        ]);
        let nested_query = NestedQuery::new(
            vec!["comments".into()],
            Box::new(must_not),
            ScoreMode::None,
            false,
        );

        let hits = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(
            1,
            hits.len(),
            "We get only doc #1 => doc #2 and doc #3 are excluded by must_not"
        );

        let (_score, addr) = hits[0];
        let stored_doc: TantivyDocument = searcher.doc(addr)?;
        let doc_num_val = stored_doc
            .get_first(doc_num_f)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!("1", doc_num_val);

        let tq_nik2 = TermQuery::new(
            Term::from_field_text(author_f, "nik9000"),
            IndexRecordOption::Basic,
        );
        let nested2 = NestedQuery::new(
            vec!["comments".into()],
            Box::new(tq_nik2),
            ScoreMode::None,
            false,
        );
        let bool_q = BooleanQuery::new(vec![
            (Occur::Must, Box::new(AllQuery)),
            (Occur::MustNot, Box::new(nested2)),
        ]);

        let hits2 = searcher.search(&bool_q, &TopDocs::with_limit(10))?;
        assert_eq!(
            1,
            hits2.len(),
            "Only doc #1 remains under an outer must_not"
        );
        let (_score2, addr2) = hits2[0];
        let doc_stored2: TantivyDocument = searcher.doc(addr2)?;
        let doc_num2 = doc_stored2
            .get_first(doc_num_f)
            .map(|v| v.as_str().unwrap().to_string())
            .unwrap();
        assert_eq!("1", doc_num2);

        Ok(())
    }

    #[test]
    fn test_nested_query_without_subfields() -> crate::Result<()> {
        use crate::collector::TopDocs;
        use crate::query::TermQuery;
        use crate::schema::{
            IndexRecordOption, NestedJsonObjectOptions, SchemaBuilder, TantivyDocument,
            TextFieldIndexing, TEXT,
        };
        use crate::{Index, Term};

        let mut builder = SchemaBuilder::default();
        let nested_json_opts = NestedJsonObjectOptions::new()
            .set_include_in_parent(false)
            .set_store_parent_flag(true)
            .set_stored()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::Basic),
            );
        let user_field = builder.add_nested_json_field(vec!["user".into()], nested_json_opts);
        builder.add_text_field("group", TEXT);

        let schema = builder.build();
        let index = Index::create_in_ram(schema.clone());

        {
            let mut writer: IndexWriter<TantivyDocument> = index.writer_for_tests()?;
            let json_doc = r#"{
            "group": "fans",
            "user": [
                { "first":"John", "last":"Smith" },
                { "first":"Alice", "last":"White" }
            ]
        }"#;

            let mut document = TantivyDocument::default();
            let expanded_docs = parse_json_for_nested_sorted(
                &schema,
                &mut document,
                &serde_json::from_str::<serde_json::Value>(json_doc).unwrap(),
            )
            .expect("parse nested doc");
            let docs = expanded_docs.into_iter().collect::<Vec<_>>();
            writer.add_documents(docs)?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let mut child_term = Term::from_field_json_path(user_field, "first", false);
        child_term.append_type_and_str("alice");
        let child_query = TermQuery::new(child_term, IndexRecordOption::Basic);

        let nested_query = NestedQuery::new(
            vec!["user".to_string()],
            Box::new(child_query),
            ScoreMode::None,
            false,
        );

        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(
            top_docs.len(),
            1,
            "Should find parent doc with child 'alice'"
        );

        Ok(())
    }

    #[test]
    fn test_nested_query_parser_syntax() -> crate::Result<()> {
        let mut builder = SchemaBuilder::default();
        let nested_json_opts = NestedJsonObjectOptions::new()
            .set_include_in_parent(false)
            .set_store_parent_flag(true)
            .set_stored()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::Basic),
            );
        let user_field = builder.add_nested_json_field(vec!["user".into()], nested_json_opts);
        let _group_field = builder.add_text_field("group", TEXT | STORED);

        let schema = builder.build();
        let index = Index::create_in_ram(schema.clone());

        {
            let mut writer = index.writer_for_tests()?;
            let json_doc = r#"
        {
          "group": "complexFans",
          "user": [
            {
              "first": "Alice",
              "last": "Anderson",
              "hobbies": ["Chess", "Painting"],
              "kids": [
                { "name": "Bob",   "age": 8 },
                { "name": "Cathy", "age": 12 }
              ]
            },
            {
              "first": "Greg",
              "last":  "Johnson",
              "hobbies": ["Skiing", "Chess"],
              "kids": [
                { "name": "Hank", "age": 3 }
              ]
            }
          ]
        }
        "#;

            let mut document = TantivyDocument::default();

            let expanded = parse_json_for_nested_sorted(
                &schema,
                &mut document,
                &serde_json::from_str::<serde_json::Value>(json_doc).unwrap(),
            )
            .expect("parse nested doc");
            let docs: Vec<TantivyDocument> = expanded.into_iter().map(Into::into).collect();
            writer.add_documents(docs)?;
            writer.commit()?;
        }

        {
            let reader = index.reader()?;
            let searcher = reader.searcher();

            let query_parser = QueryParser::for_index(&index, vec![user_field]);

            let query_str = r#"
           user.first:Alice
           AND user.hobbies:Chess
           AND user.kids.age:[8 TO 9]
        "#;

            let query = query_parser.parse_query(query_str)?;

            let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;

            assert_eq!(
                top_docs.len(),
                1,
                "Should find exactly one parent doc with child that meets all constraints."
            );

            let nested_query = NestedQuery::new(
                vec!["user".into()],
                Box::new(query_parser.parse_query("kids.name:Bob AND kids.age:8")?),
                ScoreMode::None,
                false,
            );

            let top_docs2 = searcher.search(&nested_query, &TopDocs::with_limit(10))?;

            assert_eq!(
                top_docs2.len(),
                1,
                "Should match the same doc that has a kid named Bob and age 8"
            );

            let nested_query = NestedQuery::new(
                vec!["user".into()],
                Box::new(query_parser.parse_query("kids.name:Bob AND kids.age:3")?),
                ScoreMode::None,
                false,
            );

            let top_docs3 = searcher.search(&nested_query, &TopDocs::with_limit(10))?;

            assert_eq!(
                top_docs3.len(),
                0,
                "Should not match two separate nested docs at the same level"
            );

            Ok(())
        }
    }

    fn make_nested_schema() -> (Schema, Field, Field) {
        let mut builder = Schema::builder();

        let group_field = builder.add_text_field("group", STRING | STORED);

        let nested_opts = NestedJsonObjectOptions::new()
            .set_include_in_parent(false)
            .set_store_parent_flag(true)
            .set_stored()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::Basic),
            );
        let user_nested_field = builder.add_nested_json_field(vec!["user".into()], nested_opts);

        let schema = builder.build();
        (schema, user_nested_field, group_field)
    }

    fn index_test_document(
        index_writer: &mut IndexWriter,
        schema: &Schema,
        group_val: &str,
        users: serde_json::Value,
    ) -> Result<(), DocParsingError> {
        let full_doc = json!({
            "group": group_val,
            "user": users,
        });

        let mut document = TantivyDocument::default();
        let expanded = parse_json_for_nested_sorted(schema, &mut document, &full_doc).unwrap();

        index_writer.add_documents(expanded).unwrap();

        Ok(())
    }

    #[test]
    fn test_nested_query_single_level() -> crate::Result<()> {
        let (schema, user_nested_field, _group_field) = make_nested_schema();
        let index = Index::create_in_ram(schema.clone());

        {
            let mut writer = index.writer_for_tests()?;

            index_test_document(
                &mut writer,
                &schema,
                "fans",
                json!([
                    { "first": "John", "last": "Smith" },
                    { "first": "Alice", "last": "White" }
                ]),
            )?;

            index_test_document(
                &mut writer,
                &schema,
                "boring",
                json!([
                    { "first": "Bob", "last": "Marley" }
                ]),
            )?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![user_nested_field]);

        let child_query = query_parser.parse_query("first:Alice").unwrap();

        let nested_query = NestedQuery::new(
            vec!["user".to_string()],
            Box::new(child_query),
            ScoreMode::Avg,
            false,
        );

        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(
            top_docs.len(),
            1,
            "Should match exactly one parent doc with a nested child `first = Alice`."
        );

        Ok(())
    }

    #[test]
    fn test_nested_query_no_match() -> crate::Result<()> {
        let (schema, user_nested_field, _group_field) = make_nested_schema();
        let index = Index::create_in_ram(schema.clone());

        {
            let mut writer = index.writer_for_tests()?;
            index_test_document(
                &mut writer,
                &schema,
                "groupVal",
                json!([
                    {"first":"John"},
                    {"first":"Alice"}
                ]),
            )?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![user_nested_field]);
        let child_query = query_parser.parse_query("first:NoSuchName").unwrap();
        let nested_query = NestedQuery::new(
            vec!["user".into()],
            Box::new(child_query),
            ScoreMode::None,
            false,
        );

        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(0, top_docs.len(), "No matches expected");

        Ok(())
    }

    #[test]
    fn test_nested_query_ignore_unmapped() -> crate::Result<()> {
        let (schema, ufield, _group_field) = make_nested_schema();
        let index = Index::create_in_ram(schema.clone());

        {
            let mut writer = index.writer_for_tests()?;
            index_test_document(
                &mut writer,
                &schema,
                "unmappedTest",
                json!([
                    {"first":"SomeName"}
                ]),
            )?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![ufield]);
        let child_query = query_parser.parse_query("first:SomeName").unwrap();
        let nested_query = NestedQuery::new(
            vec!["someUnknownPath".to_string()],
            Box::new(child_query),
            ScoreMode::Total,
            true,
        );

        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(0, top_docs.len(), "No docs returned, but no error either");

        Ok(())
    }

    #[test]
    #[ignore]
    fn test_nested_query_unmapped_error() {
        let (schema, _ufield, _group_field) = make_nested_schema();
        let index = Index::create_in_ram(schema.clone());

        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![]);
        let child_query = query_parser.parse_query("first:X").unwrap();
        let nested_query = NestedQuery::new(
            vec!["badPath".into()],
            Box::new(child_query),
            ScoreMode::None,
            false,
        );

        let res = searcher.search(&nested_query, &TopDocs::with_limit(10));
        match res {
            Err(e) => {
                let msg = format!("{:?}", e);
                assert!(
                    msg.contains("NestedQuery path 'badPath' not mapped")
                        && !msg.contains("ignore_unmapped=true"),
                    "Expected schema error complaining about unmapped path"
                );
            }
            Ok(_) => panic!("Expected an error for unmapped path with ignore_unmapped=false"),
        }
    }

    #[test]
    #[ignore]
    fn test_nested_query_numeric_leaf() -> crate::Result<()> {
        let mut builder = SchemaBuilder::default();
        let nested_opts = NestedJsonObjectOptions::new()
            .set_include_in_parent(false)
            .set_store_parent_flag(true)
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::WithFreqs),
            );
        let user_field = builder.add_nested_json_field(vec!["user".into()], nested_opts);
        let schema = builder.build();

        let index = Index::create_in_ram(schema.clone());
        let mut writer = index.writer_for_tests()?;

        let doc_obj = json!({
            "user": [
                { "age": 10 },
                { "age": 20 }
            ]
        });

        {
            let mut document = TantivyDocument::default();
            let expanded_docs = parse_json_for_nested_sorted(&schema, &mut document, &doc_obj)
                .expect("parse nested doc");
            writer.add_documents(expanded_docs)?;
        }
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![user_field]);
        let child_query = query_parser.parse_query(r#"age:20"#).unwrap();

        let nested_query = NestedQuery::new(
            vec!["user".into()],
            Box::new(child_query),
            ScoreMode::None,
            false,
        );

        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(
            top_docs.len(),
            1,
            "We should find exactly one parent doc that has a child with age=20"
        );

        Ok(())
    }

    fn make_nested_schema_complete() -> (Schema, Field, Field) {
        let mut builder = Schema::builder();

        let group_field = builder.add_text_field("group", STRING | STORED);

        let nested_opts = NestedJsonObjectOptions::new()
            .set_include_in_parent(false)
            .set_store_parent_flag(true)
            .set_stored()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::Basic),
            );
        let user_nested_field = builder.add_nested_json_field(vec!["user".into()], nested_opts);

        let schema = builder.build();
        (schema, user_nested_field, group_field)
    }

    fn index_test_document_complete(
        index_writer: &mut IndexWriter,
        schema: &Schema,
        group_val: &str,
        users: serde_json::Value,
    ) -> Result<(), DocParsingError> {
        let full_doc = json!({
            "group": group_val,
            "user": users,
        });

        let mut document = TantivyDocument::default();
        let expanded = parse_json_for_nested_sorted(schema, &mut document, &full_doc).unwrap();

        index_writer.add_documents(expanded).unwrap();

        Ok(())
    }

    #[test]
    fn test_nested_query_single_level_complete() -> crate::Result<()> {
        let (schema, user_nested_field, _group_field) = make_nested_schema_complete();
        let index = Index::create_in_ram(schema.clone());

        {
            let mut writer = index.writer_for_tests()?;

            index_test_document_complete(
                &mut writer,
                &schema,
                "fans",
                json!([
                    { "first": "John", "last": "Smith" },
                    { "first": "Alice", "last": "White" }
                ]),
            )?;

            index_test_document_complete(
                &mut writer,
                &schema,
                "boring",
                json!([
                    { "first": "Bob", "last": "Marley" }
                ]),
            )?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&index, vec![user_nested_field]);

        let child_query = query_parser.parse_query("first:Alice").unwrap();

        let nested_query = NestedQuery::new(
            vec!["user".to_string()],
            Box::new(child_query),
            ScoreMode::Avg,
            false,
        );

        let top_docs = searcher.search(&nested_query, &TopDocs::with_limit(10))?;
        assert_eq!(
            top_docs.len(),
            1,
            "Should match exactly one parent doc with a nested child `first = Alice`."
        );

        Ok(())
    }
}
