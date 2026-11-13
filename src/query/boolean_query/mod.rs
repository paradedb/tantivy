mod block_maxscore;
mod block_wand_intersection;
mod block_wand_union;
mod boolean_query;
mod boolean_weight;

pub use self::block_wand_intersection::BlockWandIntersectionScorer;
pub use self::block_wand_union::{BlockWandSingleScorer, BlockWandUnionScorer};
pub use self::boolean_query::BooleanQuery;
pub use self::boolean_weight::BooleanWeight;

#[cfg(test)]
mod tests {

    use std::ops::Bound;

    use super::*;
    use crate::collector::tests::TEST_COLLECTOR_WITH_SCORE;
    use crate::collector::{Count, TopDocs};
    use crate::query::term_query::TermScorer;
    use crate::query::{
        AllScorer, EmptyScorer, EnableScoring, Intersection, Occur, Query, QueryParser, RangeQuery,
        RequiredOptionalScorer, Scorer, SumCombiner, TermQuery,
    };
    use crate::schema::*;
    use crate::{assert_nearly_equals, DocAddress, DocId, Index, IndexWriter, Score};

    fn aux_test_helper() -> crate::Result<(Index, Field)> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            // writing the segment
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field => "a b c"))?;
            index_writer.add_document(doc!(text_field => "a c"))?;
            index_writer.add_document(doc!(text_field => "b c"))?;
            index_writer.add_document(doc!(text_field => "a b c d"))?;
            index_writer.add_document(doc!(text_field => "d"))?;
            index_writer.commit()?;
        }
        Ok((index, text_field))
    }

    #[test]
    pub fn test_boolean_non_all_term_disjunction() -> crate::Result<()> {
        let (index, text_field) = aux_test_helper()?;
        let query_parser = QueryParser::for_index(&index, vec![text_field]);
        let query = query_parser.parse_query("(+a +b) d")?;
        let searcher = index.reader()?.searcher();
        assert_eq!(query.count(&searcher)?, 3);
        Ok(())
    }

    #[test]
    pub fn test_boolean_single_must_clause() -> crate::Result<()> {
        let (index, text_field) = aux_test_helper()?;
        let query_parser = QueryParser::for_index(&index, vec![text_field]);
        let query = query_parser.parse_query("+a")?;
        let searcher = index.reader()?.searcher();
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let scorer = weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert!(scorer.is::<TermScorer>());
        Ok(())
    }

    #[test]
    pub fn test_boolean_termonly_intersection() -> crate::Result<()> {
        let (index, text_field) = aux_test_helper()?;
        let query_parser = QueryParser::for_index(&index, vec![text_field]);
        let searcher = index.reader()?.searcher();
        {
            let query = query_parser.parse_query("+a +b +c")?;
            let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
            let scorer = weight.scorer(searcher.segment_reader(0u32), 1.0)?;
            assert!(scorer.is::<Intersection<TermScorer>>());
        }
        {
            let query = query_parser.parse_query("+a +(b c)")?;
            let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
            let scorer = weight.scorer(searcher.segment_reader(0u32), 1.0)?;
            assert!(scorer.is::<Intersection<Box<dyn Scorer>>>());
        }
        Ok(())
    }

    #[test]
    pub fn test_boolean_reqopt() -> crate::Result<()> {
        let (index, text_field) = aux_test_helper()?;
        let query_parser = QueryParser::for_index(&index, vec![text_field]);
        let searcher = index.reader()?.searcher();
        {
            let query = query_parser.parse_query("+a b")?;
            let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
            let scorer = weight.scorer(searcher.segment_reader(0u32), 1.0)?;
            assert!(scorer
                .is::<RequiredOptionalScorer<Box<dyn Scorer>, Box<dyn Scorer>, SumCombiner>>());
        }
        {
            let query = query_parser.parse_query("+a b")?;
            let weight = query.weight(EnableScoring::disabled_from_schema(searcher.schema()))?;
            let scorer = weight.scorer(searcher.segment_reader(0u32), 1.0)?;
            assert!(scorer.is::<TermScorer>());
        }
        Ok(())
    }

    #[test]
    pub fn test_boolean_query() -> crate::Result<()> {
        let (index, text_field) = aux_test_helper()?;

        let make_term_query = |text: &str| {
            let term_query = TermQuery::new(
                Term::from_field_text(text_field, text),
                IndexRecordOption::Basic,
            );
            let query: Box<dyn Query> = Box::new(term_query);
            query
        };

        let reader = index.reader()?;

        let matching_docs = |boolean_query: &dyn Query| {
            reader
                .searcher()
                .search(boolean_query, &TEST_COLLECTOR_WITH_SCORE)
                .unwrap()
                .docs()
                .iter()
                .cloned()
                .map(|doc| doc.doc_id)
                .collect::<Vec<DocId>>()
        };
        {
            let boolean_query = BooleanQuery::new(vec![(Occur::Must, make_term_query("a"))]);
            assert_eq!(matching_docs(&boolean_query), vec![0, 1, 3]);
        }
        {
            let boolean_query = BooleanQuery::new(vec![(Occur::Should, make_term_query("a"))]);
            assert_eq!(matching_docs(&boolean_query), vec![0, 1, 3]);
        }
        {
            let boolean_query = BooleanQuery::new(vec![
                (Occur::Should, make_term_query("a")),
                (Occur::Should, make_term_query("b")),
            ]);
            assert_eq!(matching_docs(&boolean_query), vec![0, 1, 2, 3]);
        }
        {
            let boolean_query = BooleanQuery::new(vec![
                (Occur::Must, make_term_query("a")),
                (Occur::Should, make_term_query("b")),
            ]);
            assert_eq!(matching_docs(&boolean_query), vec![0, 1, 3]);
        }
        {
            let boolean_query = BooleanQuery::new(vec![
                (Occur::Must, make_term_query("a")),
                (Occur::Should, make_term_query("b")),
                (Occur::MustNot, make_term_query("d")),
            ]);
            assert_eq!(matching_docs(&boolean_query), vec![0, 1]);
        }
        {
            let boolean_query = BooleanQuery::new(vec![(Occur::MustNot, make_term_query("d"))]);
            assert_eq!(matching_docs(&boolean_query), Vec::<u32>::new());
        }
        Ok(())
    }

    #[test]
    pub fn test_boolean_query_two_excluded() -> crate::Result<()> {
        let (index, text_field) = aux_test_helper()?;

        let make_term_query = |text: &str| {
            let term_query = TermQuery::new(
                Term::from_field_text(text_field, text),
                IndexRecordOption::Basic,
            );
            let query: Box<dyn Query> = Box::new(term_query);
            query
        };

        let reader = index.reader()?;

        let matching_topdocs = |query: &dyn Query| {
            reader
                .searcher()
                .search(query, &TopDocs::with_limit(3).order_by_score())
                .unwrap()
        };

        let score_doc_4: Score; // score of doc 4 should not be influenced by exclusion
        {
            let boolean_query_no_excluded =
                BooleanQuery::new(vec![(Occur::Must, make_term_query("d"))]);
            let topdocs_no_excluded = matching_topdocs(&boolean_query_no_excluded);
            assert_eq!(topdocs_no_excluded.len(), 2);
            let (top_score, top_doc) = topdocs_no_excluded[0];
            assert_eq!(top_doc, DocAddress::new(0, 4));
            assert_eq!(topdocs_no_excluded[1].1, DocAddress::new(0, 3)); // ignore score of doc 3.
            score_doc_4 = top_score;
        }

        {
            let boolean_query_two_excluded = BooleanQuery::new(vec![
                (Occur::Must, make_term_query("d")),
                (Occur::MustNot, make_term_query("a")),
                (Occur::MustNot, make_term_query("b")),
            ]);
            let topdocs_excluded = matching_topdocs(&boolean_query_two_excluded);
            assert_eq!(topdocs_excluded.len(), 1);
            let (top_score, top_doc) = topdocs_excluded[0];
            assert_eq!(top_doc, DocAddress::new(0, 4));
            assert_eq!(top_score, score_doc_4);
        }
        Ok(())
    }

    #[test]
    pub fn test_boolean_query_with_weight() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field => "a b c"))?;
            index_writer.add_document(doc!(text_field => "a c"))?;
            index_writer.add_document(doc!(text_field => "b c"))?;
            index_writer.commit()?;
        }
        let term_a: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text_field, "a"),
            IndexRecordOption::WithFreqs,
        ));
        let term_b: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text_field, "b"),
            IndexRecordOption::WithFreqs,
        ));
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let boolean_query =
            BooleanQuery::new(vec![(Occur::Should, term_a), (Occur::Should, term_b)]);
        let boolean_weight = boolean_query
            .weight(EnableScoring::enabled_from_searcher(&searcher))
            .unwrap();
        {
            let mut boolean_scorer = boolean_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
            assert_eq!(boolean_scorer.doc(), 0u32);
            assert_nearly_equals!(boolean_scorer.score(), 0.84163445);
        }
        {
            let mut boolean_scorer = boolean_weight.scorer(searcher.segment_reader(0u32), 2.0)?;
            assert_eq!(boolean_scorer.doc(), 0u32);
            assert_nearly_equals!(boolean_scorer.score(), 1.6832689);
        }
        Ok(())
    }

    #[test]
    pub fn test_intersection_score() -> crate::Result<()> {
        let (index, text_field) = aux_test_helper()?;

        let make_term_query = |text: &str| {
            let term_query = TermQuery::new(
                Term::from_field_text(text_field, text),
                IndexRecordOption::Basic,
            );
            let query: Box<dyn Query> = Box::new(term_query);
            query
        };
        let reader = index.reader()?;
        let score_docs = |boolean_query: &dyn Query| {
            let fruit = reader
                .searcher()
                .search(boolean_query, &TEST_COLLECTOR_WITH_SCORE)
                .unwrap();
            fruit.scores().to_vec()
        };

        {
            let boolean_query = BooleanQuery::new(vec![
                (Occur::Must, make_term_query("a")),
                (Occur::Must, make_term_query("b")),
            ]);
            let scores = score_docs(&boolean_query);
            assert_nearly_equals!(scores[0], 0.977973);
            assert_nearly_equals!(scores[1], 0.84699446);
        }
        Ok(())
    }

    #[test]
    pub fn test_explain() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", STRING);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(text=>"a"))?;
        index_writer.add_document(doc!(text=>"b"))?;
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();
        let term_a: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text, "a"),
            IndexRecordOption::Basic,
        ));
        let term_b: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text, "b"),
            IndexRecordOption::Basic,
        ));
        let query = BooleanQuery::from(vec![(Occur::Should, term_a), (Occur::Should, term_b)]);
        let explanation = query.explain(&searcher, DocAddress::new(0, 0u32))?;
        assert_nearly_equals!(explanation.value(), std::f32::consts::LN_2);
        Ok(())
    }

    #[test]
    pub fn test_boolean_weight_optimization() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(text_field=>"hello"))?;
        index_writer.add_document(doc!(text_field=>"hello happy"))?;
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();
        let term_match_all: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text_field, "hello"),
            IndexRecordOption::Basic,
        ));
        let term_match_some: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text_field, "happy"),
            IndexRecordOption::Basic,
        ));
        let term_match_none: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text_field, "tax"),
            IndexRecordOption::Basic,
        ));
        {
            let query = BooleanQuery::from(vec![
                (Occur::Must, term_match_all.box_clone()),
                (Occur::Must, term_match_some.box_clone()),
            ]);
            let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
            let scorer = weight.scorer(searcher.segment_reader(0u32), 1.0f32)?;
            assert!(scorer.is::<TermScorer>());
        }
        {
            let query = BooleanQuery::from(vec![
                (Occur::Must, term_match_all.box_clone()),
                (Occur::Must, term_match_some.box_clone()),
                (Occur::Must, term_match_none.box_clone()),
            ]);
            let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
            let scorer = weight.scorer(searcher.segment_reader(0u32), 1.0f32)?;
            assert!(scorer.is::<EmptyScorer>());
        }
        {
            let query = BooleanQuery::from(vec![
                (Occur::Should, term_match_all.box_clone()),
                (Occur::Should, term_match_none.box_clone()),
            ]);
            let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
            let scorer = weight.scorer(searcher.segment_reader(0u32), 1.0f32)?;
            assert!(scorer.is::<AllScorer>());
        }
        {
            let query = BooleanQuery::from(vec![
                (Occur::Should, term_match_some.box_clone()),
                (Occur::Should, term_match_none.box_clone()),
            ]);
            let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
            let scorer = weight.scorer(searcher.segment_reader(0u32), 1.0f32)?;
            assert!(scorer.is::<TermScorer>());
        }
        Ok(())
    }

    #[test]
    pub fn test_min_should_match_with_all_query() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let num_field =
            schema_builder.add_i64_field("num", NumericOptions::default().set_fast().set_indexed());
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;

        index_writer.add_document(doc!(text_field => "apple", num_field => 10i64))?;
        index_writer.add_document(doc!(text_field => "banana", num_field => 20i64))?;
        index_writer.commit()?;

        let searcher = index.reader()?.searcher();

        let effective_all_match_query: Box<dyn Query> = Box::new(RangeQuery::new(
            Bound::Excluded(Term::from_field_i64(num_field, 0)),
            Bound::Unbounded,
        ));
        let term_query: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text_field, "apple"),
            IndexRecordOption::Basic,
        ));

        // in some previous version, we would remove the 2 all_match, but then say we need *4*
        // matches out of the 3 term queries, which matches nothing.
        let mut bool_query = BooleanQuery::new(vec![
            (Occur::Should, effective_all_match_query.box_clone()),
            (Occur::Should, effective_all_match_query.box_clone()),
            (Occur::Should, term_query.box_clone()),
            (Occur::Should, term_query.box_clone()),
            (Occur::Should, term_query.box_clone()),
        ]);
        bool_query.set_minimum_number_should_match(4);
        let count = searcher.search(&bool_query, &Count)?;
        assert_eq!(count, 1);

        Ok(())
    }

    // =========================================================================
    // AllScorer Preservation Regression Tests
    // =========================================================================
    //
    // These tests verify the fix for a bug where AllScorer instances (produced by
    // queries matching all documents, such as range queries covering all values)
    // were incorrectly removed from Boolean query processing, causing documents
    // to be unexpectedly excluded from results.
    //
    // The bug manifested in several scenarios:
    // 1. SHOULD + SHOULD where one clause is AllScorer
    // 2. MUST (AllScorer) + SHOULD
    // 3. Range queries in Boolean clauses when all documents match the range

    /// Regression test: SHOULD clause with AllScorer combined with other SHOULD clauses.
    ///
    /// When a SHOULD clause produces an AllScorer (e.g., from a range query matching
    /// all documents), the Boolean query should still match all documents.
    ///
    /// Bug before fix: AllScorer was removed during optimization, leaving only the
    /// other SHOULD clauses, which incorrectly excluded documents.
    #[test]
    pub fn test_should_with_all_scorer_regression() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let num_field =
            schema_builder.add_i64_field("num", NumericOptions::default().set_fast().set_indexed());
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;

        // All docs have num > 0, so range query will return AllScorer
        index_writer.add_document(doc!(text_field => "hello", num_field => 10i64))?;
        index_writer.add_document(doc!(text_field => "world", num_field => 20i64))?;
        index_writer.add_document(doc!(text_field => "hello world", num_field => 30i64))?;
        index_writer.add_document(doc!(text_field => "foo", num_field => 40i64))?;
        index_writer.add_document(doc!(text_field => "bar", num_field => 50i64))?;
        index_writer.add_document(doc!(text_field => "baz", num_field => 60i64))?;
        index_writer.commit()?;

        let searcher = index.reader()?.searcher();

        // Range query matching all docs (returns AllScorer)
        let all_match_query: Box<dyn Query> = Box::new(RangeQuery::new(
            Bound::Excluded(Term::from_field_i64(num_field, 0)),
            Bound::Unbounded,
        ));
        let term_query: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text_field, "hello"),
            IndexRecordOption::Basic,
        ));

        // Verify range matches all 6 docs
        assert_eq!(searcher.search(all_match_query.as_ref(), &Count)?, 6);

        // RangeQuery(all) OR TermQuery should match all 6 docs
        let bool_query = BooleanQuery::new(vec![
            (Occur::Should, all_match_query.box_clone()),
            (Occur::Should, term_query.box_clone()),
        ]);
        let count = searcher.search(&bool_query, &Count)?;
        assert_eq!(count, 6, "SHOULD with AllScorer should match all docs");

        // Order should not matter
        let bool_query_reversed = BooleanQuery::new(vec![
            (Occur::Should, term_query.box_clone()),
            (Occur::Should, all_match_query.box_clone()),
        ]);
        let count_reversed = searcher.search(&bool_query_reversed, &Count)?;
        assert_eq!(
            count_reversed, 6,
            "Order of SHOULD clauses should not matter"
        );

        Ok(())
    }

    /// Regression test: MUST clause with AllScorer combined with SHOULD clause.
    ///
    /// When MUST contains an AllScorer, all documents satisfy the MUST constraint.
    /// The SHOULD clause should only affect scoring, not filtering.
    ///
    /// Bug before fix: AllScorer was removed, leaving an empty must_scorers vector.
    /// intersect_scorers([]) incorrectly returned EmptyScorer, matching 0 documents.
    #[test]
    pub fn test_must_all_with_should_regression() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let num_field =
            schema_builder.add_i64_field("num", NumericOptions::default().set_fast().set_indexed());
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;

        // All docs have num > 0, so range query will return AllScorer
        index_writer.add_document(doc!(text_field => "apple", num_field => 10i64))?;
        index_writer.add_document(doc!(text_field => "banana", num_field => 20i64))?;
        index_writer.add_document(doc!(text_field => "cherry", num_field => 30i64))?;
        index_writer.add_document(doc!(text_field => "date", num_field => 40i64))?;
        index_writer.commit()?;

        let searcher = index.reader()?.searcher();

        // Range query matching all docs (returns AllScorer)
        let all_match_query: Box<dyn Query> = Box::new(RangeQuery::new(
            Bound::Excluded(Term::from_field_i64(num_field, 0)),
            Bound::Unbounded,
        ));
        let term_query: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text_field, "apple"),
            IndexRecordOption::Basic,
        ));

        // Verify range matches all 4 docs
        assert_eq!(searcher.search(all_match_query.as_ref(), &Count)?, 4);

        // MUST(range matching all) AND SHOULD(term) should match all 4 docs
        let bool_query = BooleanQuery::new(vec![
            (Occur::Must, all_match_query.box_clone()),
            (Occur::Should, term_query.box_clone()),
        ]);
        let count = searcher.search(&bool_query, &Count)?;
        assert_eq!(count, 4, "MUST AllScorer + SHOULD should match all docs");

        Ok(())
    }

    /// Regression test: Range queries in Boolean clauses when all documents match.
    ///
    /// Range queries can return AllScorer as an optimization when all indexed values
    /// fall within the range. This test ensures such queries work correctly in
    /// Boolean combinations.
    ///
    /// This is the most common real-world manifestation of the bug, occurring in
    /// queries like: (age > 50 OR name = 'Alice') AND status = 'active'
    /// when all documents have age > 50.
    #[test]
    pub fn test_range_query_all_match_in_boolean() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let name_field = schema_builder.add_text_field("name", TEXT);
        let age_field =
            schema_builder.add_i64_field("age", NumericOptions::default().set_fast().set_indexed());
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;

        // All documents have age > 50, so range query will return AllScorer
        index_writer.add_document(doc!(name_field => "alice", age_field => 55_i64))?;
        index_writer.add_document(doc!(name_field => "bob", age_field => 60_i64))?;
        index_writer.add_document(doc!(name_field => "charlie", age_field => 70_i64))?;
        index_writer.add_document(doc!(name_field => "diana", age_field => 80_i64))?;
        index_writer.commit()?;

        let searcher = index.reader()?.searcher();

        let range_query: Box<dyn Query> = Box::new(RangeQuery::new(
            Bound::Excluded(Term::from_field_i64(age_field, 50)),
            Bound::Unbounded,
        ));
        let term_query: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(name_field, "alice"),
            IndexRecordOption::Basic,
        ));

        // Verify preconditions
        assert_eq!(searcher.search(range_query.as_ref(), &Count)?, 4);
        assert_eq!(searcher.search(term_query.as_ref(), &Count)?, 1);

        // SHOULD(range) OR SHOULD(term): range matches all, so result is 4
        let should_query = BooleanQuery::new(vec![
            (Occur::Should, range_query.box_clone()),
            (Occur::Should, term_query.box_clone()),
        ]);
        assert_eq!(
            searcher.search(&should_query, &Count)?,
            4,
            "SHOULD range OR term should match all"
        );

        // MUST(range) AND SHOULD(term): range matches all, term is optional
        let must_should_query = BooleanQuery::new(vec![
            (Occur::Must, range_query.box_clone()),
            (Occur::Should, term_query.box_clone()),
        ]);
        assert_eq!(
            searcher.search(&must_should_query, &Count)?,
            4,
            "MUST range + SHOULD term should match all"
        );

        Ok(())
    }

    /// Test multiple AllScorer instances in different clause types.
    ///
    /// Verifies correct behavior when AllScorers appear in multiple positions.
    #[test]
    pub fn test_multiple_all_scorers() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let num_field =
            schema_builder.add_i64_field("num", NumericOptions::default().set_fast().set_indexed());
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;

        // All docs have num > 0, so range queries will return AllScorer
        index_writer.add_document(doc!(text_field => "doc1", num_field => 10i64))?;
        index_writer.add_document(doc!(text_field => "doc2", num_field => 20i64))?;
        index_writer.add_document(doc!(text_field => "doc3", num_field => 30i64))?;
        index_writer.commit()?;

        let searcher = index.reader()?.searcher();

        // Two different range queries that both match all docs (return AllScorer)
        let all_query1: Box<dyn Query> = Box::new(RangeQuery::new(
            Bound::Excluded(Term::from_field_i64(num_field, 0)),
            Bound::Unbounded,
        ));
        let all_query2: Box<dyn Query> = Box::new(RangeQuery::new(
            Bound::Excluded(Term::from_field_i64(num_field, 5)),
            Bound::Unbounded,
        ));
        let term_query: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(text_field, "doc1"),
            IndexRecordOption::Basic,
        ));

        // Multiple AllScorers in SHOULD
        let multi_all_should = BooleanQuery::new(vec![
            (Occur::Should, all_query1.box_clone()),
            (Occur::Should, all_query2.box_clone()),
            (Occur::Should, term_query.box_clone()),
        ]);
        assert_eq!(
            searcher.search(&multi_all_should, &Count)?,
            3,
            "Multiple AllScorers in SHOULD"
        );

        // AllScorer in both MUST and SHOULD
        let all_must_and_should = BooleanQuery::new(vec![
            (Occur::Must, all_query1.box_clone()),
            (Occur::Should, all_query2.box_clone()),
        ]);
        assert_eq!(
            searcher.search(&all_must_and_should, &Count)?,
            3,
            "AllScorer in both MUST and SHOULD"
        );

        Ok(())
    }
}

/// A proptest which generates arbitrary permutations of a simple boolean AST, and then matches
/// the result against an index which contains all permutations of documents with N fields.
#[cfg(test)]
mod proptest_boolean_query {
    use std::collections::{BTreeMap, HashSet};
    use std::ops::{Bound, Range};

    use proptest::collection::vec;
    use proptest::prelude::*;

    use crate::collector::DocSetCollector;
    use crate::query::{AllQuery, BooleanQuery, Occur, Query, RangeQuery, TermQuery};
    use crate::schema::{Field, NumericOptions, OwnedValue, Schema, TEXT};
    use crate::{DocId, Index, Term};

    #[derive(Debug, Clone)]
    enum BooleanQueryAST {
        /// Matches all documents via AllQuery (wraps AllScorer in BoostScorer)
        All,
        /// Matches all documents via RangeQuery (returns bare AllScorer)
        /// This is the actual trigger for the AllScorer preservation bug
        RangeAll,
        /// Matches documents where the field has value "true"
        Leaf {
            field_idx: usize,
        },
        Union(Vec<BooleanQueryAST>),
        Intersection(Vec<BooleanQueryAST>),
    }

    impl BooleanQueryAST {
        fn matches(&self, doc_id: DocId) -> bool {
            match self {
                BooleanQueryAST::All => true,
                BooleanQueryAST::RangeAll => true,
                BooleanQueryAST::Leaf { field_idx } => Self::matches_field(doc_id, *field_idx),
                BooleanQueryAST::Union(children) => {
                    children.iter().any(|child| child.matches(doc_id))
                }
                BooleanQueryAST::Intersection(children) => {
                    children.iter().all(|child| child.matches(doc_id))
                }
            }
        }

        fn matches_field(doc_id: DocId, field_idx: usize) -> bool {
            ((doc_id as usize) >> field_idx) & 1 == 1
        }

        fn to_query(&self, fields: &[Field], range_field: Field) -> Box<dyn Query> {
            match self {
                BooleanQueryAST::All => Box::new(AllQuery),
                BooleanQueryAST::RangeAll => {
                    // Range query that matches all docs (all have value >= 0)
                    // This returns bare AllScorer, triggering the bug we fixed
                    Box::new(RangeQuery::new(
                        Bound::Included(Term::from_field_i64(range_field, 0)),
                        Bound::Unbounded,
                    ))
                }
                BooleanQueryAST::Leaf { field_idx } => Box::new(TermQuery::new(
                    Term::from_field_text(fields[*field_idx], "true"),
                    crate::schema::IndexRecordOption::Basic,
                )),
                BooleanQueryAST::Union(children) => {
                    let sub_queries = children
                        .iter()
                        .map(|child| (Occur::Should, child.to_query(fields, range_field)))
                        .collect();
                    Box::new(BooleanQuery::new(sub_queries))
                }
                BooleanQueryAST::Intersection(children) => {
                    let sub_queries = children
                        .iter()
                        .map(|child| (Occur::Must, child.to_query(fields, range_field)))
                        .collect();
                    Box::new(BooleanQuery::new(sub_queries))
                }
            }
        }
    }

    fn doc_ids(num_docs: usize, num_fields: usize) -> Range<DocId> {
        let permutations = 1 << num_fields;
        let copies = (num_docs as f32 / permutations as f32).ceil() as u32;
        0..(permutations * copies)
    }

    fn create_index_with_boolean_permutations(
        num_docs: usize,
        num_fields: usize,
    ) -> (Index, Vec<Field>, Field) {
        let mut schema_builder = Schema::builder();
        let fields: Vec<Field> = (0..num_fields)
            .map(|i| schema_builder.add_text_field(&format!("field_{}", i), TEXT))
            .collect();
        // Add a numeric field for RangeQuery tests - all docs have value = doc_id
        let range_field = schema_builder.add_i64_field(
            "range_field",
            NumericOptions::default().set_fast().set_indexed(),
        );
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests().unwrap();

        for doc_id in doc_ids(num_docs, num_fields) {
            let mut doc: BTreeMap<_, OwnedValue> = BTreeMap::default();
            for (field_idx, &field) in fields.iter().enumerate() {
                if (doc_id >> field_idx) & 1 == 1 {
                    doc.insert(field, "true".into());
                }
            }
            // All docs have non-negative values, so RangeQuery(>=0) matches all
            doc.insert(range_field, (doc_id as i64).into());
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
        (index, fields, range_field)
    }

    fn arb_boolean_query_ast(num_fields: usize) -> impl Strategy<Value = BooleanQueryAST> {
        // Leaf strategies: term queries, AllQuery, and RangeQuery matching all docs
        let leaf = prop_oneof![
            (0..num_fields).prop_map(|field_idx| BooleanQueryAST::Leaf { field_idx }),
            Just(BooleanQueryAST::All),
            Just(BooleanQueryAST::RangeAll),
        ];
        leaf.prop_recursive(
            8,   // 8 levels of recursion
            256, // 256 nodes max
            10,  // 10 items per collection
            |inner| {
                prop_oneof![
                    vec(inner.clone(), 1..10).prop_map(BooleanQueryAST::Union),
                    vec(inner, 1..10).prop_map(BooleanQueryAST::Intersection),
                ]
            },
        )
    }

    #[test]
    fn proptest_boolean_query() {
        // In the presence of optimizations around buffering, it can take large numbers of
        // documents to uncover some issues.
        let num_fields = 8;
        let num_docs = 1 << num_fields;
        let (index, fields, range_field) =
            create_index_with_boolean_permutations(num_docs, num_fields);
        let searcher = index.reader().unwrap().searcher();
        proptest!(|(ast in arb_boolean_query_ast(num_fields))| {
            let query = ast.to_query(&fields, range_field);

            let mut matching_docs = HashSet::new();
            for doc_id in doc_ids(num_docs, num_fields) {
                if ast.matches(doc_id as DocId) {
                    matching_docs.insert(doc_id as DocId);
                }
            }

            let doc_addresses = searcher.search(&*query, &DocSetCollector).unwrap();
            let result_docs: HashSet<DocId> =
                doc_addresses.into_iter().map(|doc_address| doc_address.doc_id).collect();
            prop_assert_eq!(result_docs, matching_docs);
        });
    }
}

#[cfg(test)]
mod pruning_and_equivalence_tests {
    use std::collections::{HashMap, HashSet};
    use std::ops::Bound;
    use std::sync::OnceLock;

    use proptest::prelude::*;

    use super::*;
    use crate::collector::tests::TEST_COLLECTOR_WITH_SCORE;
    use crate::collector::{Count, DocSetCollector, TopDocs};
    use crate::indexer::NoMergePolicy;
    use crate::query::scorer::BasicPruningScorer;
    use crate::query::{
        AllQuery, AllScorer, ConstScoreQuery, DisjunctionPruning, EmptyQuery, EmptyScorer,
        EnableScoring, Occur, PhraseQuery, Query, RangeQuery, Scorer, TermQuery, TermSetQuery,
    };
    use crate::schema::{Field, IndexRecordOption, NumericOptions, Schema, TEXT};
    use crate::{
        DocAddress, DocId, DocSet, Index, IndexWriter, Score, Searcher, SeekDangerResult,
        SegmentReader, Term, TERMINATED,
    };

    const DOCS_PER_SEGMENT: usize = 300;
    const SCORE_TOL: Score = 1e-4;

    #[derive(Clone, Copy, Debug)]
    enum Coverage {
        /// Segment `0..300`: `RangeFrom200` covers `200..300`.
        Partial,
        /// Segment `300..600`: `RangeFrom200` covers all docs (`AllScorer`).
        Full,
    }

    #[derive(Clone, Copy, Debug)]
    enum ScoringMode {
        Disabled,
        Enabled,
    }

    #[derive(Clone, Copy, Debug)]
    enum MinShould {
        Default,
        Explicit(usize),
    }

    /// Catalog of leaf queries exercising `EmptyScorer`, `AllScorer` (wrapped and bare, on one or
    /// both segments), constant-score vs dynamic filters, `TermSetQuery` candidate probing,
    /// non-freq vs freq `TermScorer`s, and mid-segment (`doc >= 150`) `seek_danger` invariants.
    macro_rules! leaves {
        ($($variant:ident),+ $(,)?) => {
            #[derive(Clone, Copy, Debug)]
            enum Leaf { $($variant),+ }
            impl Leaf {
                const ALL: &[Self] = &[$(Self::$variant),+];
            }
        };
    }

    leaves!(
        Empty,
        All,
        ConstAll,
        TermMissing,
        TermCommon,
        TermTarget,
        TermExtra,
        TermSparse,
        TermBasic,
        TermSet150,
        RangeAll,
        RangeFrom200,
        RangeAFrom100,
        RangeBFrom150,
        PhraseFrom150,
        ConstPhraseFrom150,
    );

    #[derive(Clone, Debug)]
    enum QueryAst {
        Leaf(Leaf),
        Boolean(Vec<(Occur, QueryAst)>, MinShould),
    }

    struct Fixture {
        _index: Index,
        searcher: Searcher,
        segments: [u32; 2],
        leaves: [Box<dyn Query>; Leaf::ALL.len()],
        leaf_scores: [[Vec<Option<Score>>; 2]; Leaf::ALL.len()],
    }

    impl Fixture {
        fn get() -> &'static Self {
            static FIXTURE: OnceLock<Fixture> = OnceLock::new();
            FIXTURE.get_or_init(|| {
                let mut sb = Schema::builder();
                let text = sb.add_text_field("text", TEXT);
                let tag = sb.add_text_field("tag", TEXT);
                let num_opts = NumericOptions::default().set_fast().set_indexed();
                let num_a = sb.add_i64_field("num_a", num_opts.clone());
                let num_b = sb.add_i64_field("num_b", num_opts);
                let index = Index::create_in_ram(sb.build());

                let mut writer: IndexWriter = index.writer_for_tests().unwrap();
                writer.set_merge_policy(Box::new(NoMergePolicy));
                for seg in 0..2 {
                    for i in (seg * DOCS_PER_SEGMENT)..((seg + 1) * DOCS_PER_SEGMENT) {
                        let mut words = vec!["common"];
                        if i % 3 == 0 {
                            words
                                .extend(std::iter::repeat_n("target", 1 + usize::from(i % 9 == 0)));
                        }
                        if i % 7 == 0 {
                            words
                                .extend(std::iter::repeat_n("extra", 1 + usize::from(i % 14 == 0)));
                        }
                        if matches!(i, 10 | 150 | 260) {
                            words.push("sparse");
                        }
                        let tag_val = if i >= 150 && i % 10 == 0 {
                            "red apple"
                        } else {
                            "apple red"
                        };
                        writer
                            .add_document(doc!(
                                text => words.join(" "),
                                tag => tag_val,
                                num_a => i as i64,
                                num_b => i as i64,
                            ))
                            .unwrap();
                    }
                    writer.commit().unwrap();
                }

                let searcher = index.reader().unwrap().searcher();
                assert_eq!(searcher.segment_readers().len(), 2);

                let term = |w: &str, opt: IndexRecordOption| -> Box<dyn Query> {
                    Box::new(TermQuery::new(Term::from_field_text(text, w), opt))
                };
                let range_from = |field: Field, lower: i64| -> Box<dyn Query> {
                    Box::new(RangeQuery::new(
                        Bound::Included(Term::from_field_i64(field, lower)),
                        Bound::Unbounded,
                    ))
                };
                let phrase = || {
                    PhraseQuery::new(vec![
                        Term::from_field_text(tag, "red"),
                        Term::from_field_text(tag, "apple"),
                    ])
                };
                let leaves = std::array::from_fn(|idx| -> Box<dyn Query> {
                    match Leaf::ALL[idx] {
                        Leaf::Empty => Box::new(EmptyQuery),
                        Leaf::All => Box::new(AllQuery),
                        Leaf::ConstAll => Box::new(ConstScoreQuery::new(AllQuery, 1.25)),
                        Leaf::TermMissing => term("missing", IndexRecordOption::WithFreqs),
                        Leaf::TermCommon => term("common", IndexRecordOption::WithFreqs),
                        Leaf::TermTarget => term("target", IndexRecordOption::WithFreqs),
                        Leaf::TermExtra => term("extra", IndexRecordOption::WithFreqs),
                        Leaf::TermSparse => {
                            term("sparse", IndexRecordOption::WithFreqsAndPositions)
                        }
                        Leaf::TermBasic => term("target", IndexRecordOption::Basic),
                        Leaf::TermSet150 => {
                            Box::new(TermSetQuery::new(vec![Term::from_field_i64(num_a, 150)]))
                        }
                        Leaf::RangeAll => range_from(num_a, 0),
                        Leaf::RangeFrom200 => range_from(num_a, 200),
                        Leaf::RangeAFrom100 => range_from(num_a, 100),
                        Leaf::RangeBFrom150 => range_from(num_b, 150),
                        Leaf::PhraseFrom150 => Box::new(phrase()),
                        Leaf::ConstPhraseFrom150 => Box::new(ConstScoreQuery::new(phrase(), 0.75)),
                    }
                });

                let leaf_scores = std::array::from_fn(|idx| {
                    let weight = leaves[idx]
                        .weight(EnableScoring::enabled_from_searcher(&searcher))
                        .unwrap();
                    std::array::from_fn(|seg_ord| {
                        let reader = searcher.segment_reader(seg_ord as u32);
                        let mut scores = vec![None; reader.max_doc() as usize];
                        for (doc, score) in collect_scorer(weight.scorer(reader, 1.0).unwrap()) {
                            scores[doc as usize] = Some(score);
                        }
                        scores
                    })
                });

                let seg0_min = searcher
                    .segment_reader(0)
                    .fast_fields()
                    .i64("num_a")
                    .unwrap()
                    .min_value();
                let segments = if seg0_min == 0 { [0, 1] } else { [1, 0] };
                Self {
                    _index: index,
                    searcher,
                    segments,
                    leaves,
                    leaf_scores,
                }
            })
        }

        fn leaf(&self, leaf: Leaf) -> Box<dyn Query> {
            self.leaves[leaf as usize].box_clone()
        }

        fn segment(&self, coverage: Coverage) -> &SegmentReader {
            self.searcher
                .segment_reader(self.segments[coverage as usize])
        }
    }

    impl QueryAst {
        fn to_boolean_query(&self, fixture: &Fixture) -> BooleanQuery {
            let Self::Boolean(clauses, min_should) = self else {
                unreachable!("expected Boolean root")
            };
            let subqueries = clauses
                .iter()
                .map(|(occur, child)| (*occur, child.to_query(fixture)))
                .collect();
            match *min_should {
                MinShould::Default => BooleanQuery::new(subqueries),
                MinShould::Explicit(n) => {
                    BooleanQuery::with_minimum_required_clauses(subqueries, n)
                }
            }
        }

        fn to_query(&self, fixture: &Fixture) -> Box<dyn Query> {
            match self {
                Self::Leaf(leaf) => fixture.leaf(*leaf),
                Self::Boolean(..) => Box::new(self.to_boolean_query(fixture)),
            }
        }

        fn eval(&self, fixture: &Fixture, addr: DocAddress) -> Option<Score> {
            let Self::Boolean(clauses, min_should) = self else {
                let Self::Leaf(leaf) = self else {
                    unreachable!()
                };
                return fixture.leaf_scores[*leaf as usize][addr.segment_ord as usize]
                    [addr.doc_id as usize];
            };
            let (mut num_must, mut num_should, mut matching_should) = (0usize, 0usize, 0usize);
            let mut total_score: Score = 0.0;
            for (occur, child) in clauses {
                let child_score = child.eval(fixture, addr);
                match occur {
                    Occur::Must => {
                        total_score += child_score?;
                        num_must += 1;
                    }
                    Occur::MustNot if child_score.is_some() => return None,
                    Occur::MustNot => {}
                    Occur::Should => {
                        num_should += 1;
                        if let Some(s) = child_score {
                            matching_should += 1;
                            total_score += s;
                        }
                    }
                }
            }
            if num_must == 0 && num_should == 0 {
                return None;
            }
            let specified = match *min_should {
                MinShould::Explicit(n) => n,
                MinShould::Default => {
                    let has_must_or_not = clauses
                        .iter()
                        .any(|(o, _)| matches!(o, Occur::Must | Occur::MustNot));
                    usize::from(!has_must_or_not && num_should > 0)
                }
            };
            let effective_min = if specified > 0 {
                specified
            } else {
                usize::from(num_must == 0)
            };
            (matching_should >= effective_min).then_some(total_score)
        }
    }

    #[derive(Clone, Copy)]
    struct ExpectedSegment<'a>(&'a [(DocId, Score)]);

    fn collect_scorer(mut scorer: impl Scorer) -> Vec<(DocId, Score)> {
        let mut out = Vec::new();
        while scorer.doc() != TERMINATED {
            out.push((scorer.doc(), scorer.score()));
            scorer.advance();
        }
        out
    }

    fn assert_segment_out(
        mode: ScoringMode,
        actual: &[(DocId, Score)],
        expected: ExpectedSegment<'_>,
        ctx: &dyn std::fmt::Debug,
    ) {
        assert_eq!(actual.len(), expected.0.len(), "{mode:?} {ctx:?}");
        for (&(doc, score), &(exp_doc, exp_score)) in actual.iter().zip(expected.0) {
            assert_eq!(doc, exp_doc, "{mode:?} {ctx:?}");
            if matches!(mode, ScoringMode::Enabled) {
                assert!(
                    (score - exp_score).abs() <= SCORE_TOL,
                    "doc {doc}: {score} vs {exp_score} for {ctx:?}"
                );
            }
        }
    }

    fn assert_pruned_subset(
        actual: &[(DocId, Score)],
        expected: ExpectedSegment<'_>,
        threshold: Score,
        ctx: &dyn std::fmt::Debug,
    ) {
        let exp_map: HashMap<DocId, Score> = expected.0.iter().copied().collect();
        let mut seen = HashSet::with_capacity(actual.len());
        for &(doc, score) in actual {
            let exp_score = exp_map[&doc];
            assert!(
                (score - exp_score).abs() <= SCORE_TOL && score > threshold - SCORE_TOL,
                "{ctx:?}"
            );
            assert!(seen.insert(doc), "duplicate doc {doc} for {ctx:?}");
        }
        for &(doc, exp_score) in expected.0 {
            if exp_score > threshold + SCORE_TOL {
                assert!(
                    seen.contains(&doc),
                    "missing doc {doc} ({exp_score} > {threshold}) {ctx:?}"
                );
            }
        }
    }

    /// Verifies agreement across `Searcher` collectors and `Weight` entry points against
    /// `QueryAst::eval` on both segments.
    fn assert_query_invariants(fixture: &Fixture, ast: &QueryAst) -> crate::Result<()> {
        let searcher = &fixture.searcher;
        let query = ast.to_boolean_query(fixture);
        let expected: [Vec<(DocId, Score)>; 2] = std::array::from_fn(|seg_ord| {
            (0..searcher.segment_reader(seg_ord as u32).max_doc())
                .filter_map(|doc| {
                    ast.eval(fixture, DocAddress::new(seg_ord as u32, doc))
                        .map(|score| (doc, score))
                })
                .collect()
        });
        let global_exp: HashMap<DocAddress, Score> = (0..2)
            .flat_map(|s| {
                expected[s]
                    .iter()
                    .map(move |&(d, sc)| (DocAddress::new(s as u32, d), sc))
            })
            .collect();

        // 1. Searcher-level collectors.
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, global_exp.len(), "{ast:?}");
        let doc_set = searcher.search(&query, &DocSetCollector)?;
        assert_eq!(doc_set, global_exp.keys().copied().collect(), "{ast:?}");

        let fruit = searcher.search(&query, &TEST_COLLECTOR_WITH_SCORE)?;
        assert_eq!(fruit.docs().len(), global_exp.len(), "{ast:?}");
        for (&addr, &score) in fruit.docs().iter().zip(fruit.scores()) {
            assert!((score - global_exp[&addr]).abs() <= SCORE_TOL, "{ast:?}");
        }

        let top5 = searcher.search(&query, &TopDocs::with_limit(5).order_by_score())?;
        assert_eq!(top5.len(), global_exp.len().min(5), "{ast:?}");
        let top5_addrs: HashSet<DocAddress> = top5.iter().map(|&(_, a)| a).collect();
        for &(score, addr) in &top5 {
            assert!((score - global_exp[&addr]).abs() <= SCORE_TOL, "{ast:?}");
        }
        if let Some(&(cutoff, _)) = top5.last().filter(|_| global_exp.len() > 5) {
            for (&addr, &exp_score) in &global_exp {
                if !top5_addrs.contains(&addr) {
                    assert!(exp_score <= cutoff + SCORE_TOL, "{ast:?}");
                }
            }
        }

        // 2. Segment-level Weight entry points across scoring and DisjunctionPruning modes.
        for (seg_ord, reader) in searcher.segment_readers().iter().enumerate() {
            let seg_exp = ExpectedSegment(&expected[seg_ord]);
            let exp_docs: Vec<DocId> = seg_exp.0.iter().map(|&(d, _)| d).collect();

            for mode in [ScoringMode::Disabled, ScoringMode::Enabled] {
                let weight = query.weight(match mode {
                    ScoringMode::Disabled => EnableScoring::disabled_from_searcher(searcher),
                    ScoringMode::Enabled => EnableScoring::enabled_from_searcher(searcher),
                })?;
                assert_segment_out(
                    mode,
                    &collect_scorer(weight.scorer(reader, 1.0)?),
                    seg_exp,
                    &ast,
                );
                assert_segment_out(
                    mode,
                    &collect_scorer(weight.pruning_scorer(reader, 1.0, Score::MIN)?),
                    seg_exp,
                    &ast,
                );

                let mut batch_docs = Vec::new();
                weight.for_each_no_score_batch(reader, &mut |batch| {
                    batch.for_each_doc_block(|docs| batch_docs.extend_from_slice(docs));
                })?;
                assert_eq!(batch_docs, exp_docs, "{mode:?} {ast:?}");

                let mut cand = weight.scorer_for_candidates_unpositioned(reader, 1.0, 8)?;
                let (mut cand_out, mut target) = (Vec::new(), cand.doc());
                while target < reader.max_doc() {
                    match cand.seek_danger(target) {
                        SeekDangerResult::Found => {
                            cand_out.push((target, cand.score()));
                            target += 1;
                        }
                        SeekDangerResult::SeekLowerBound(bound) => target = bound.max(target + 1),
                    }
                }
                assert_segment_out(mode, &cand_out, seg_exp, &ast);
            }

            if let Some(max_score) = seg_exp.0.iter().map(|&(_, s)| s).reduce(Score::max) {
                let scored_weight = query.weight(EnableScoring::enabled_from_searcher(searcher))?;
                for (&(doc, score), &(exp_doc, exp_score)) in
                    collect_scorer(scored_weight.scorer(reader, 2.0)?)
                        .iter()
                        .zip(seg_exp.0)
                {
                    assert_eq!(doc, exp_doc, "{ast:?}");
                    assert!(
                        (score - 2.0 * exp_score).abs() <= 2.0 * SCORE_TOL,
                        "{ast:?}"
                    );
                }

                // `max_score - 0.25` lies strictly between `term_score` and `term_score +
                // filter_boost` for constant filters (`>= 0.75`), catching un-subtracted
                // `filter_boost` thresholds.
                let threshold = max_score - 0.25;
                let pruned = collect_scorer(scored_weight.pruning_scorer(reader, 1.0, threshold)?);
                assert_pruned_subset(&pruned, seg_exp, threshold, &ast);

                for pruning_mode in [
                    DisjunctionPruning::Auto,
                    DisjunctionPruning::BlockWand,
                    DisjunctionPruning::BlockMaxScore,
                ] {
                    let mode_weight = query.weight(
                        EnableScoring::enabled_from_searcher(searcher)
                            .with_disjunction_pruning(pruning_mode),
                    )?;
                    for init in [Score::MIN, threshold] {
                        let mut out = Vec::new();
                        mode_weight.for_each_pruning(init, reader, &mut |doc, score| {
                            out.push((doc, score));
                            init
                        })?;
                        assert_pruned_subset(&out, seg_exp, init, &(&ast, pruning_mode));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn regression_and_single_clause_invariants() -> crate::Result<()> {
        use Leaf::*;
        let fixture = Fixture::get();
        let l = QueryAst::Leaf;
        let bq = |clauses: Vec<(Occur, QueryAst)>| QueryAst::Boolean(clauses, MinShould::Default);
        let and = |qs: &[QueryAst]| bq(qs.iter().cloned().map(|q| (Occur::Must, q)).collect());
        let must_not = |pos, neg| bq(vec![(Occur::Must, pos), (Occur::MustNot, neg)]);
        let or_leaves = |ls: &[Leaf]| bq(ls.iter().map(|&x| (Occur::Should, l(x))).collect());
        let filtered_or = |shoulds: &[Leaf], filters: &[Leaf]| {
            let clauses = shoulds
                .iter()
                .map(|&x| (Occur::Should, l(x)))
                .chain(filters.iter().map(|&x| (Occur::Must, l(x))))
                .collect();
            QueryAst::Boolean(clauses, MinShould::Explicit(1))
        };

        // Exhaustive 0-clause and 1-clause matrix across `Occur` and `minimum_number_should_match`.
        for min_should in [
            MinShould::Default,
            MinShould::Explicit(0),
            MinShould::Explicit(1),
            MinShould::Explicit(2),
            MinShould::Explicit(usize::MAX),
        ] {
            assert_query_invariants(fixture, &QueryAst::Boolean(vec![], min_should))?;
            for &single in Leaf::ALL {
                for occur in [Occur::Must, Occur::Should, Occur::MustNot] {
                    let node = QueryAst::Boolean(vec![(occur, l(single))], min_should);
                    assert_query_invariants(fixture, &node)?;
                }
            }
        }

        // Flattened `target AND (extra OR num_a >= 200)` keeps `AllScorer` in `Should`.
        let flattened = and(&[l(TermTarget), or_leaves(&[TermExtra, RangeFrom200])]);
        let (flat_clauses, flat_min) = flattened.to_boolean_query(fixture).flattened_clauses();
        assert_eq!(flat_min, 1);
        let should_count = flat_clauses
            .iter()
            .filter(|(o, _)| *o == Occur::Should)
            .count();
        assert_eq!(should_count, 2);

        let ranges_150 = and(&[l(RangeAFrom100), l(RangeBFrom150)]);
        let regressions = [
            flattened,
            and(&[l(TermTarget), l(RangeFrom200)]),
            must_not(l(TermTarget), l(RangeFrom200)),
            must_not(l(TermCommon), l(PhraseFrom150)),
            must_not(l(TermCommon), ranges_150),
            and(&[l(TermTarget), l(TermExtra), l(RangeFrom200)]),
            filtered_or(&[TermTarget, TermExtra], &[RangeFrom200]),
            and(&[or_leaves(&[TermTarget, TermExtra]), l(RangeFrom200)]),
            and(&[l(TermTarget), l(RangeAFrom100), l(RangeBFrom150)]),
            filtered_or(&[TermTarget, TermCommon], &[RangeAFrom100, RangeBFrom150]),
            and(&[l(TermTarget), l(TermCommon), l(ConstPhraseFrom150)]),
            filtered_or(&[TermTarget, TermCommon], &[ConstPhraseFrom150]),
        ];
        for ast in &regressions {
            assert_query_invariants(fixture, ast)?;
        }
        Ok(())
    }

    fn arb_boolean_ast() -> impl Strategy<Value = QueryAst> {
        let occur = prop_oneof![Just(Occur::Must), Just(Occur::Should), Just(Occur::MustNot)];
        let min_should = prop_oneof![
            3 => Just(MinShould::Default),
            1 => Just(MinShould::Explicit(0)),
            2 => Just(MinShould::Explicit(1)),
            2 => Just(MinShould::Explicit(2)),
            1 => Just(MinShould::Explicit(3)),
            1 => Just(MinShould::Explicit(usize::MAX)),
        ];
        let (root_occur, root_min) = (occur.clone(), min_should.clone());
        let child = prop::sample::select(Leaf::ALL)
            .prop_map(QueryAst::Leaf)
            .prop_recursive(3, 24, 5, move |inner| {
                (
                    prop::collection::vec((occur.clone(), inner), 0..=5),
                    min_should.clone(),
                )
                    .prop_map(|(c, m)| QueryAst::Boolean(c, m))
            });
        (prop::collection::vec((root_occur, child), 0..=5), root_min)
            .prop_map(|(c, m)| QueryAst::Boolean(c, m))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn proptest_boolean_query_invariants(ast in arb_boolean_ast()) {
            assert_query_invariants(Fixture::get(), &ast).unwrap();
        }
    }

    /// White-box checks for concrete `Scorer` and `PruningScorer` specializations.
    #[test]
    fn pruning_and_segment_scorer_specializations() -> crate::Result<()> {
        use Leaf::*;
        use Occur::{Must, MustNot, Should};
        let fixture = Fixture::get();
        let searcher = &fixture.searcher;
        let partial = fixture.segment(Coverage::Partial);
        let full = fixture.segment(Coverage::Full);
        let bq = |clauses: &[(Occur, Leaf)]| {
            BooleanQuery::new(clauses.iter().map(|&(o, l)| (o, fixture.leaf(l))).collect())
        };

        let range_weight = fixture
            .leaf(RangeFrom200)
            .weight(EnableScoring::enabled_from_searcher(searcher))?;
        assert!(range_weight.scorer(full, 1.0)?.is::<AllScorer>());
        assert!(!range_weight.scorer(partial, 1.0)?.is::<AllScorer>());

        let must_not_range = bq(&[(Must, TermTarget), (MustNot, RangeFrom200)]);
        for scoring in [
            EnableScoring::enabled_from_searcher(searcher),
            EnableScoring::disabled_from_searcher(searcher),
        ] {
            let weight = must_not_range.weight(scoring)?;
            assert!(weight.scorer(full, 1.0)?.is::<EmptyScorer>());
            assert!(!weight.scorer(partial, 1.0)?.is::<EmptyScorer>());
        }

        // Multi-block skipping: `sparse` matches {10, 150, 260}; `num_a >= 100` skips 10.
        let mut skip_pruning = bq(&[(Must, TermSparse), (Must, RangeAFrom100)])
            .weight(EnableScoring::enabled_from_searcher(searcher))?
            .pruning_scorer(partial, 1.0, 0.0)?;
        assert!(skip_pruning.is::<BlockWandSingleScorer>());
        assert_eq!(skip_pruning.doc(), 150);
        assert_eq!(skip_pruning.advance(), 260);
        assert_eq!(skip_pruning.advance(), TERMINATED);

        let filtered_shapes = |filter: Leaf| {
            let single = bq(&[(Must, TermTarget), (Must, filter)]);
            let union = BooleanQuery::with_minimum_required_clauses(
                vec![
                    (Should, fixture.leaf(TermTarget)),
                    (Should, fixture.leaf(TermExtra)),
                    (Must, fixture.leaf(filter)),
                ],
                1,
            );
            let intersection = bq(&[(Must, TermTarget), (Must, TermExtra), (Must, filter)]);
            (single, union, intersection)
        };

        for (name, reader, filter) in [
            ("TermSetQuery", partial, TermSet150),
            ("RangeQuery(partial)", partial, RangeFrom200),
            ("RangeQuery(AllScorer)", full, RangeFrom200),
            ("ConstScoreQuery", partial, ConstPhraseFrom150),
        ] {
            let (single, union, intersection) = filtered_shapes(filter);
            let enabled = |q: &BooleanQuery| {
                q.weight(EnableScoring::enabled_from_searcher(searcher))?
                    .pruning_scorer(reader, 1.0, 0.0)
            };
            assert!(
                enabled(&single)?.is::<BlockWandSingleScorer>(),
                "{name} single"
            );
            assert!(
                enabled(&union)?.is::<BlockWandUnionScorer>(),
                "{name} union"
            );
            assert!(
                enabled(&intersection)?.is::<BlockWandIntersectionScorer>(),
                "{name} intersection"
            );
            let disabled = single
                .weight(EnableScoring::disabled_from_searcher(searcher))?
                .pruning_scorer(reader, 1.0, 0.0)?;
            assert!(disabled.is::<BasicPruningScorer>(), "{name} disabled");
        }

        let (dyn_single, dyn_union, dyn_intersection) = filtered_shapes(PhraseFrom150);
        let phrase_main = bq(&[(Must, PhraseFrom150), (Must, TermSet150)]);
        for (label, query) in [
            ("dyn_single", &dyn_single),
            ("dyn_union", &dyn_union),
            ("dyn_intersection", &dyn_intersection),
            ("phrase_main", &phrase_main),
        ] {
            let s = query
                .weight(EnableScoring::enabled_from_searcher(searcher))?
                .pruning_scorer(partial, 1.0, 0.0)?;
            assert!(s.is::<BasicPruningScorer>(), "{label}");
        }
        Ok(())
    }
}
