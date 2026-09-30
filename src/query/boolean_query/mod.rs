mod block_maxscore;
mod block_wand_intersection;
mod block_wand_union;
mod boolean_query;
mod boolean_weight;
mod filtered_pruning_scorer;

pub use self::block_wand_intersection::BlockWandIntersectionScorer;
pub use self::block_wand_union::{BlockWandSingleScorer, BlockWandUnionScorer};
pub use self::boolean_query::BooleanQuery;
pub use self::boolean_weight::BooleanWeight;
pub use self::filtered_pruning_scorer::FilteredPruningScorer;

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
    use crate::{assert_nearly_equals, DocAddress, DocId, Index, IndexWriter, Score, TERMINATED};

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

    #[test]
    pub fn test_filtered_pruning_multi_block_skipping() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let num_field =
            schema_builder.add_i64_field("num", NumericOptions::default().set_fast().set_indexed());
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            // Create 300 docs across multiple 128-doc blocks
            for i in 0..300 {
                let text = if i == 10 || i == 150 || i == 260 {
                    "target"
                } else {
                    "other"
                };
                index_writer.add_document(doc!(text_field => text, num_field => i as i64))?;
            }
            index_writer.commit()?;
        }

        let searcher = index.reader()?.searcher();
        let term_query = TermQuery::new(
            Term::from_field_text(text_field, "target"),
            IndexRecordOption::WithFreqsAndPositions,
        );
        // Filter out doc 10, keeping only docs in [100, 300]
        let range_query = RangeQuery::new(
            Bound::Included(Term::from_field_i64(num_field, 100)),
            Bound::Included(Term::from_field_i64(num_field, 300)),
        );

        let query = BooleanQuery::new(vec![
            (Occur::Must, Box::new(term_query)),
            (Occur::Must, Box::new(range_query)),
        ]);

        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let mut pruning_scorer = weight
            .pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0)?
            .expect("should construct a dynamic pruning scorer");
        // First match must be doc 150 (doc 10 was skipped because filter doc was >= 100)
        assert_eq!(pruning_scorer.doc(), 150);
        assert_eq!(pruning_scorer.advance(), 260);
        assert_eq!(pruning_scorer.advance(), TERMINATED);

        let top_docs = searcher.search(&query, &TopDocs::with_limit(10).order_by_score())?;
        assert_eq!(top_docs.len(), 2);
        let doc_ids: Vec<DocId> = top_docs.iter().map(|(_, addr)| addr.doc_id).collect();
        assert!(doc_ids.contains(&150));
        assert!(doc_ids.contains(&260));

        Ok(())
    }

    #[test]
    pub fn test_filtered_pruning_initial_threshold() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let num_field =
            schema_builder.add_i64_field("num", NumericOptions::default().set_fast().set_indexed());
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field => "target other", num_field => 100i64))?;
            index_writer.add_document(doc!(text_field => "unrelated", num_field => 0i64))?;
            index_writer.commit()?;
        }

        let searcher = index.reader()?.searcher();
        let term_query1 = TermQuery::new(
            Term::from_field_text(text_field, "target"),
            IndexRecordOption::WithFreqsAndPositions,
        );
        let term_query2 = TermQuery::new(
            Term::from_field_text(text_field, "other"),
            IndexRecordOption::WithFreqsAndPositions,
        );
        let range_query = RangeQuery::new(
            Bound::Included(Term::from_field_i64(num_field, 50)),
            Bound::Included(Term::from_field_i64(num_field, 150)),
        );

        // Case 1: Single term + filter
        let single_query = BooleanQuery::new(vec![
            (Occur::Must, Box::new(term_query1.clone())),
            (Occur::Must, Box::new(range_query.clone())),
        ]);
        let weight = single_query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let mut baseline = weight
            .pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0)?
            .unwrap();
        assert_eq!(baseline.doc(), 0);
        let total_score = baseline.score();
        let init_threshold = total_score - 0.5;
        let pruning_scorer = weight
            .pruning_scorer(searcher.segment_reader(0u32), 1.0, init_threshold)?
            .unwrap();
        assert_eq!(pruning_scorer.doc(), 0);

        // Case 2: Multi-term intersection + filter (FilteredTermIntersection)
        let intersection_query = BooleanQuery::new(vec![
            (Occur::Must, Box::new(term_query1.clone())),
            (Occur::Must, Box::new(term_query2.clone())),
            (Occur::Must, Box::new(range_query.clone())),
        ]);
        let weight = intersection_query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let mut baseline = weight
            .pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0)?
            .unwrap();
        assert_eq!(baseline.doc(), 0);
        let total_score = baseline.score();
        let init_threshold = total_score - 0.5;
        let pruning_scorer = weight
            .pruning_scorer(searcher.segment_reader(0u32), 1.0, init_threshold)?
            .unwrap();
        assert_eq!(pruning_scorer.doc(), 0);

        // Case 3: Term union + filter (FilteredTermUnion)
        let mut union_query = BooleanQuery::new(vec![
            (Occur::Should, Box::new(term_query1.clone())),
            (Occur::Should, Box::new(term_query2.clone())),
            (Occur::Must, Box::new(range_query.clone())),
        ]);
        union_query.set_minimum_number_should_match(1);
        let weight = union_query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let mut baseline = weight
            .pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0)?
            .unwrap();
        assert_eq!(baseline.doc(), 0);
        let total_score = baseline.score();
        let init_threshold = total_score - 0.5;
        let pruning_scorer = weight
            .pruning_scorer(searcher.segment_reader(0u32), 1.0, init_threshold)?
            .unwrap();
        assert_eq!(pruning_scorer.doc(), 0);

        // Case 4: Nested BooleanQuery + filter (try_build_filtered_pruning)
        let inner_boolean = BooleanQuery::new(vec![
            (Occur::Should, Box::new(term_query1)),
            (Occur::Should, Box::new(term_query2)),
        ]);
        let nested_query = BooleanQuery::new(vec![
            (Occur::Must, Box::new(inner_boolean)),
            (Occur::Must, Box::new(range_query)),
        ]);
        let weight = nested_query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let mut baseline = weight
            .pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0)?
            .unwrap();
        assert_eq!(baseline.doc(), 0);
        let total_score = baseline.score();
        let init_threshold = total_score - 0.5;
        let pruning_scorer = weight
            .pruning_scorer(searcher.segment_reader(0u32), 1.0, init_threshold)?
            .unwrap();
        assert_eq!(pruning_scorer.doc(), 0);

        Ok(())
    }

    #[test]
    pub fn test_filtered_pruning_constant_vs_dynamic_filter() -> crate::Result<()> {
        use crate::query::{PhraseQuery, TermSetQuery};

        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let num_field =
            schema_builder.add_i64_field("num", NumericOptions::default().set_fast().set_indexed());
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field => "quick brown fox", num_field => 10i64))?;
            index_writer.add_document(doc!(text_field => "quick blue fox", num_field => 20i64))?;
            index_writer.add_document(doc!(text_field => "lazy brown dog", num_field => 10i64))?;
            index_writer.commit()?;
        }

        let searcher = index.reader()?.searcher();
        let term_query = TermQuery::new(
            Term::from_field_text(text_field, "quick"),
            IndexRecordOption::WithFreqsAndPositions,
        );

        // Case 1: Constant-scoring filter (TermSetQuery on fast field).
        // Should select FilteredPruningScorer.
        let term_set_query = TermSetQuery::new(vec![Term::from_field_i64(num_field, 10)]);
        let const_filter_query = BooleanQuery::new(vec![
            (Occur::Must, Box::new(term_query.clone())),
            (Occur::Must, Box::new(term_set_query.clone())),
        ]);
        let const_weight =
            const_filter_query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let const_pruning = const_weight.pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0)?;
        assert!(
            const_pruning.is_some(),
            "Constant-scoring filter should enable FilteredPruningScorer"
        );

        // Case 2: Dynamic-scoring filter (PhraseQuery).
        // Must return None when scoring is enabled to prevent false pruning.
        let phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "brown"),
            Term::from_field_text(text_field, "fox"),
        ]);
        let dynamic_filter_query = BooleanQuery::new(vec![
            (Occur::Must, Box::new(term_query.clone())),
            (Occur::Must, Box::new(phrase_query.clone())),
        ]);
        let dynamic_weight =
            dynamic_filter_query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let dynamic_pruning =
            dynamic_weight.pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0)?;
        assert!(
            dynamic_pruning.is_none(),
            "Dynamic-scoring filter must return None"
        );

        // Case 3: Scoring disabled (EnableScoring::disabled_from_searcher).
        // Must return None even with a constant-scoring filter.
        let disabled_weight =
            const_filter_query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let disabled_pruning =
            disabled_weight.pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0)?;
        assert!(
            disabled_pruning.is_none(),
            "Disabled scoring must return None"
        );

        // Case 4: Multi-term intersection with dynamic-scoring filter (FilteredTermIntersection).
        // Must return None.
        let term_query2 = TermQuery::new(
            Term::from_field_text(text_field, "brown"),
            IndexRecordOption::WithFreqsAndPositions,
        );
        let intersection_dynamic_query = BooleanQuery::new(vec![
            (Occur::Must, Box::new(term_query.clone())),
            (Occur::Must, Box::new(term_query2)),
            (Occur::Must, Box::new(phrase_query.clone())),
        ]);
        let intersection_dynamic_weight =
            intersection_dynamic_query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let intersection_dynamic_pruning =
            intersection_dynamic_weight.pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0)?;
        assert!(
            intersection_dynamic_pruning.is_none(),
            "FilteredTermIntersection with dynamic filter must return None"
        );

        // Case 5: PhraseQuery with constant-scoring filter.
        // PhraseQuery supports dynamic pruning, so try_build_filtered_pruning should
        // wrap it in FilteredPruningScorer without falling back.
        let phrase_filter_query = BooleanQuery::new(vec![
            (Occur::Must, Box::new(phrase_query.clone())),
            (Occur::Must, Box::new(term_set_query.clone())),
        ]);
        let phrase_filter_weight =
            phrase_filter_query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let phrase_filter_pruning =
            phrase_filter_weight.pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0)?;
        assert!(
            phrase_filter_pruning.is_some(),
            "PhraseQuery with constant-scoring filter should enable FilteredPruningScorer"
        );

        // Verify search results still match accurately.
        let top_docs = searcher.search(
            &dynamic_filter_query,
            &TopDocs::with_limit(10).order_by_score(),
        )?;
        assert_eq!(top_docs.len(), 1);
        assert_eq!(top_docs[0].1.doc_id, 0);

        let top_docs_phrase = searcher.search(
            &phrase_filter_query,
            &TopDocs::with_limit(10).order_by_score(),
        )?;
        assert_eq!(top_docs_phrase.len(), 1);
        assert_eq!(top_docs_phrase[0].1.doc_id, 0);

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

    use crate::collector::tests::TEST_COLLECTOR_WITH_SCORE;
    use crate::collector::{DocSetCollector, TopDocs};
    use crate::query::term_set_query::TermSetQuery;
    use crate::query::{
        AllQuery, BooleanQuery, EnableScoring, Occur, Query, RangeQuery, TermQuery,
    };
    use crate::schema::{Field, IndexRecordOption, NumericOptions, OwnedValue, Schema, TEXT};
    use crate::{DocAddress, DocId, Index, Score, Term};

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

    #[derive(Debug, Clone)]
    enum FilteredScoringClause {
        Term(usize),
        TermIntersection(Vec<usize>),
        TermUnion(Vec<usize>),
        NestedUnion(Vec<usize>),
        NestedIntersection(Vec<usize>),
    }

    #[derive(Debug, Clone)]
    enum FilteredFilterClause {
        Range(u32, u32),
        TermSet(Vec<usize>),
        Both(u32, u32, Vec<usize>),
    }

    fn arb_filtered_scoring_clause(
        num_fields: usize,
    ) -> impl Strategy<Value = FilteredScoringClause> {
        prop_oneof![
            (0..num_fields).prop_map(FilteredScoringClause::Term),
            proptest::collection::btree_set(0..num_fields, 2..=3)
                .prop_map(|s| FilteredScoringClause::TermIntersection(s.into_iter().collect())),
            proptest::collection::btree_set(0..num_fields, 2..=3)
                .prop_map(|s| FilteredScoringClause::TermUnion(s.into_iter().collect())),
            proptest::collection::btree_set(0..num_fields, 2..=3)
                .prop_map(|s| FilteredScoringClause::NestedUnion(s.into_iter().collect())),
            proptest::collection::btree_set(0..num_fields, 2..=3)
                .prop_map(|s| FilteredScoringClause::NestedIntersection(s.into_iter().collect())),
        ]
    }

    fn arb_filtered_filter_clause(
        num_fields: usize,
        num_docs: usize,
    ) -> impl Strategy<Value = FilteredFilterClause> {
        let range_strat = (0..num_docs as u32)
            .prop_flat_map(move |start| (start..num_docs as u32).prop_map(move |end| (start, end)));
        let term_set_strat = proptest::collection::btree_set(0..num_fields, 1..=3)
            .prop_map(|s| s.into_iter().collect());
        let both_strat = (0..num_docs as u32).prop_flat_map(move |start| {
            (
                start..num_docs as u32,
                proptest::collection::btree_set(0..num_fields, 1..=3),
            )
                .prop_map(move |(end, s)| (start, end, s.into_iter().collect()))
        });

        prop_oneof![
            range_strat.prop_map(|(start, end)| FilteredFilterClause::Range(start, end)),
            term_set_strat.prop_map(FilteredFilterClause::TermSet),
            both_strat
                .prop_map(|(start, end, terms)| FilteredFilterClause::Both(start, end, terms)),
        ]
    }

    fn scoring_and_filter_to_query(
        scoring: &FilteredScoringClause,
        filter: &FilteredFilterClause,
        fields: &[Field],
        range_field: Field,
    ) -> BooleanQuery {
        let filter_queries: Vec<Box<dyn Query>> = match filter {
            FilteredFilterClause::Range(start, end) => {
                vec![Box::new(RangeQuery::new(
                    Bound::Included(Term::from_field_i64(range_field, *start as i64)),
                    Bound::Included(Term::from_field_i64(range_field, *end as i64)),
                ))]
            }
            FilteredFilterClause::TermSet(indices) => {
                let terms: Vec<Term> = indices
                    .iter()
                    .map(|&idx| Term::from_field_text(fields[idx], "true"))
                    .collect();
                vec![Box::new(TermSetQuery::new(terms))]
            }
            FilteredFilterClause::Both(start, end, indices) => {
                let terms: Vec<Term> = indices
                    .iter()
                    .map(|&idx| Term::from_field_text(fields[idx], "true"))
                    .collect();
                vec![
                    Box::new(RangeQuery::new(
                        Bound::Included(Term::from_field_i64(range_field, *start as i64)),
                        Bound::Included(Term::from_field_i64(range_field, *end as i64)),
                    )),
                    Box::new(TermSetQuery::new(terms)),
                ]
            }
        };

        match scoring {
            FilteredScoringClause::Term(idx) => {
                let mut clauses: Vec<(Occur, Box<dyn Query>)> = vec![(
                    Occur::Must,
                    Box::new(TermQuery::new(
                        Term::from_field_text(fields[*idx], "true"),
                        IndexRecordOption::WithFreqsAndPositions,
                    )),
                )];
                for fq in filter_queries {
                    clauses.push((Occur::Must, fq));
                }
                BooleanQuery::new(clauses)
            }
            FilteredScoringClause::TermIntersection(indices) => {
                let mut clauses: Vec<(Occur, Box<dyn Query>)> = indices
                    .iter()
                    .map(|&idx| {
                        (
                            Occur::Must,
                            Box::new(TermQuery::new(
                                Term::from_field_text(fields[idx], "true"),
                                IndexRecordOption::WithFreqsAndPositions,
                            )) as Box<dyn Query>,
                        )
                    })
                    .collect();
                for fq in filter_queries {
                    clauses.push((Occur::Must, fq));
                }
                BooleanQuery::new(clauses)
            }
            FilteredScoringClause::TermUnion(indices) => {
                let mut clauses: Vec<(Occur, Box<dyn Query>)> = indices
                    .iter()
                    .map(|&idx| {
                        (
                            Occur::Should,
                            Box::new(TermQuery::new(
                                Term::from_field_text(fields[idx], "true"),
                                IndexRecordOption::WithFreqsAndPositions,
                            )) as Box<dyn Query>,
                        )
                    })
                    .collect();
                for fq in filter_queries {
                    clauses.push((Occur::Must, fq));
                }
                let mut bq = BooleanQuery::new(clauses);
                bq.set_minimum_number_should_match(1);
                bq
            }
            FilteredScoringClause::NestedUnion(indices) => {
                let inner_clauses: Vec<(Occur, Box<dyn Query>)> = indices
                    .iter()
                    .map(|&idx| {
                        (
                            Occur::Should,
                            Box::new(TermQuery::new(
                                Term::from_field_text(fields[idx], "true"),
                                IndexRecordOption::WithFreqsAndPositions,
                            )) as Box<dyn Query>,
                        )
                    })
                    .collect();
                let inner = BooleanQuery::new(inner_clauses);
                let mut clauses: Vec<(Occur, Box<dyn Query>)> =
                    vec![(Occur::Must, Box::new(inner))];
                for fq in filter_queries {
                    clauses.push((Occur::Must, fq));
                }
                BooleanQuery::new(clauses)
            }
            FilteredScoringClause::NestedIntersection(indices) => {
                let inner_clauses: Vec<(Occur, Box<dyn Query>)> = indices
                    .iter()
                    .map(|&idx| {
                        (
                            Occur::Must,
                            Box::new(TermQuery::new(
                                Term::from_field_text(fields[idx], "true"),
                                IndexRecordOption::WithFreqsAndPositions,
                            )) as Box<dyn Query>,
                        )
                    })
                    .collect();
                let inner = BooleanQuery::new(inner_clauses);
                let mut clauses: Vec<(Occur, Box<dyn Query>)> =
                    vec![(Occur::Must, Box::new(inner))];
                for fq in filter_queries {
                    clauses.push((Occur::Must, fq));
                }
                BooleanQuery::new(clauses)
            }
        }
    }

    #[test]
    fn proptest_filtered_pruning() {
        let num_fields = 8;
        let num_docs = 1 << num_fields;
        let (index, fields, range_field) =
            create_index_with_boolean_permutations(num_docs, num_fields);
        let searcher = index.reader().unwrap().searcher();

        let strategy = (
            arb_filtered_scoring_clause(num_fields),
            arb_filtered_filter_clause(num_fields, num_docs),
            1..=10usize,
        );

        proptest!(|(
            (scoring, filter, limit) in strategy
        )| {
            let query = scoring_and_filter_to_query(&scoring, &filter, &fields, range_field);

            // 1. Verify dynamic pruning is NOT disabled by the filter.
            let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher)).unwrap();
            let pruning = weight.pruning_scorer(searcher.segment_reader(0u32), 1.0, 0.0).unwrap();
            prop_assert!(
                pruning.is_some(),
                "query {:?} fell back to unpruned scoring",
                scoring
            );

            // 2. Compare pruned top-K against exhaustive (unpruned) top-K.
            let actual = searcher
                .search(&query, &TopDocs::with_limit(limit).order_by_score())
                .unwrap();
            let fruit = searcher.search(&query, &TEST_COLLECTOR_WITH_SCORE).unwrap();
            let mut expected: Vec<(Score, DocAddress)> = fruit
                .scores()
                .iter()
                .copied()
                .zip(fruit.docs().iter().copied())
                .collect();
            expected.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            expected.truncate(limit);

            prop_assert_eq!(actual.len(), expected.len());
            for (act, exp) in actual.iter().zip(expected.iter()) {
                prop_assert_eq!(act.1, exp.1);
                prop_assert!(
                    (act.0 - exp.0).abs() < 1e-4,
                    "score mismatch for doc {:?}: actual={}, expected={}",
                    act.1,
                    act.0,
                    exp.0
                );
            }
        });
    }
}
