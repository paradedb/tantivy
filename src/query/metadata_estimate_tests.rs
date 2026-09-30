use super::{
    AllQuery, BooleanQuery, BoostQuery, ConstScoreQuery, EmptyQuery, FuzzyTermQuery,
    MoreLikeThisQuery, Occur, PhrasePrefixQuery, PhraseQuery, Query, QueryClone, QueryEstimate,
    RegexPhraseQuery, RegexQuery, TermQuery,
};
use crate::merge_policy::NoMergePolicy;
use crate::schema::{Field, IndexRecordOption, Schema, FAST, TEXT};
use crate::{Index, IndexWriter, Term};

fn fixture() -> crate::Result<(Index, IndexWriter, Field, Field)> {
    let mut schema = Schema::builder();
    let text = schema.add_text_field("text", TEXT);
    let number = schema.add_u64_field("number", FAST);
    let index = Index::create_in_ram(schema.build());
    let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for id in 0..1000u64 {
        let mut body = String::from("all");
        if id < 600 {
            body.push_str(" common");
        }
        if id < 100 {
            body.push_str(" rare");
        }
        if id == 0 {
            body.push_str(" singleton unique");
        }
        body.push_str(if id < 500 { " left" } else { " right" });
        writer.add_document(doc!(text => body, number => id))?;
    }
    writer.commit()?;
    Ok((index, writer, text, number))
}

fn term(field: Field, text: &str) -> Box<dyn Query> {
    Box::new(TermQuery::new(
        Term::from_field_text(field, text),
        IndexRecordOption::WithFreqsAndPositions,
    ))
}

#[test]
fn metadata_estimates_terms_and_constants() -> crate::Result<()> {
    let (index, _writer, text, _) = fixture()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    for (word, count) in [("all", 1000), ("common", 600), ("rare", 100), ("absent", 0)] {
        assert_eq!(
            term(text, word).estimate_docs(reader)?,
            Some((count, u64::from(count)))
        );
    }
    assert_eq!(AllQuery.estimate_docs(reader)?, Some((1000, 1000)));
    assert_eq!(EmptyQuery.estimate_docs(reader)?, Some((0, 0)));
    Ok(())
}

#[test]
fn metadata_estimates_delegate_through_nested_wrappers() -> crate::Result<()> {
    let (index, _writer, text, number) = fixture()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    let wrap = |query: Box<dyn Query>, variant| -> Box<dyn Query> {
        match variant {
            0 => query,
            1 => Box::new(BoostQuery::new(query, 2.0)),
            2 => Box::new(ConstScoreQuery::new(query, 3.0)),
            3 => Box::new(BoostQuery::new(
                Box::new(ConstScoreQuery::new(query, 3.0)),
                2.0,
            )),
            4 => Box::new(ConstScoreQuery::new(
                Box::new(BoostQuery::new(query, 2.0)),
                3.0,
            )),
            _ => Box::new(query),
        }
    };
    for (query, expected) in [
        (term(text, "rare"), Some((100, 100))),
        (term(text, "absent"), Some((0, 0))),
        (Box::new(AllQuery) as Box<dyn Query>, Some((1000, 1000))),
        (Box::new(EmptyQuery), Some((0, 0))),
        (
            Box::new(TermQuery::new(
                Term::from_field_u64(number, 42),
                IndexRecordOption::Basic,
            )),
            None,
        ),
        (
            Box::new(BooleanQuery::intersection(vec![
                term(text, "rare"),
                term(text, "common"),
            ])),
            None,
        ),
    ] {
        for outer in 0..6 {
            for middle in 0..6 {
                for inner in 0..6 {
                    let wrapped = wrap(wrap(wrap(query.box_clone(), inner), middle), outer);
                    assert_eq!(wrapped.estimate_docs(reader)?, expected, "{wrapped:?}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn metadata_estimates_only_text_terms() -> crate::Result<()> {
    use crate::schema::INDEXED;

    let mut schema = Schema::builder();
    let json = schema.add_json_field("json", TEXT);
    let number = schema.add_u64_field("number", INDEXED | FAST);
    let unindexed = schema.add_text_field("unindexed", FAST);
    let index = Index::create_in_ram(schema.build());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(
        json => serde_json::json!({"word": "hello", "number": 42}),
        number => 42u64,
        unindexed => "hello"
    ))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    let mut json_text = Term::from_field_json_path(json, "word", false);
    json_text.append_type_and_str("hello");
    let mut json_number = Term::from_field_json_path(json, "number", false);
    json_number.append_type_and_fast_value(42i64);
    for (term, expected) in [
        (json_text, Some((1, 1))),
        (json_number, None),
        (Term::from_field_u64(number, 42), None),
        (Term::from_field_text(unindexed, "hello"), None),
    ] {
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        assert_eq!(query.estimate_docs(reader)?, expected, "{query:?}");
    }
    Ok(())
}

#[test]
fn metadata_estimates_keep_deleted_document_frequencies() -> crate::Result<()> {
    let (index, mut writer, text, _) = fixture()?;
    writer.delete_term(Term::from_field_text(text, "singleton"));
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    assert_eq!(reader.num_docs(), 999);
    assert_eq!(reader.max_doc(), 1000);
    assert_eq!(term(text, "singleton").estimate_docs(reader)?, Some((1, 1)));
    assert_eq!(term(text, "singleton").count(&searcher)?, 0);
    Ok(())
}

#[test]
fn metadata_estimates_leave_unsupported_queries_to_the_caller() -> crate::Result<()> {
    let (index, _writer, text, number) = fixture()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    let unsupported: Vec<Box<dyn Query>> = vec![
        Box::new(BooleanQuery::new(vec![])),
        Box::new(BooleanQuery::intersection(vec![term(text, "rare")])),
        Box::new(BooleanQuery::intersection(vec![
            term(text, "rare"),
            term(text, "common"),
        ])),
        Box::new(BooleanQuery::union(vec![
            term(text, "rare"),
            term(text, "common"),
        ])),
        Box::new(BooleanQuery::new(vec![
            (Occur::Must, term(text, "rare")),
            (Occur::Should, term(text, "common")),
        ])),
        Box::new(BooleanQuery::new(vec![(
            Occur::MustNot,
            term(text, "rare"),
        )])),
        Box::new(BooleanQuery::union_with_minimum_required_clauses(
            vec![term(text, "rare"), term(text, "common")],
            2,
        )),
        Box::new(TermQuery::new(
            Term::from_field_u64(number, 42),
            IndexRecordOption::Basic,
        )),
    ];
    for query in unsupported {
        assert_eq!(query.estimate_docs(reader)?, None, "{query:?}");
        let wrapped = BoostQuery::new(Box::new(ConstScoreQuery::new(query, 3.0)), 2.0);
        for occur in [Occur::Must, Occur::Should] {
            let nested = BooleanQuery::new(vec![
                (occur, term(text, "rare")),
                (occur, wrapped.box_clone()),
            ]);
            assert_eq!(nested.estimate_docs(reader)?, None, "{nested:?}");
        }
    }
    Ok(())
}

#[test]
fn metadata_estimates_text_expansions() -> crate::Result<()> {
    let (index, _writer, text, _) = fixture()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(RegexQuery::from_pattern("rar.*", text)?),
        Box::new(FuzzyTermQuery::new(
            Term::from_field_text(text, "raer"),
            1,
            true,
        )),
        Box::new(FuzzyTermQuery::new_prefix(
            Term::from_field_text(text, "rar"),
            0,
            false,
        )),
        Box::new(PhrasePrefixQuery::new(vec![Term::from_field_text(
            text, "rar",
        )])),
    ];
    for query in queries {
        assert_eq!(query.count(&searcher)?, 100);
        assert_eq!(query.estimate_docs(reader)?, Some((100, 100)), "{query:?}");
        let wrapped = BoostQuery::new(Box::new(ConstScoreQuery::new(query, 2.0)), 3.0);
        assert_eq!(wrapped.estimate_docs(reader)?, Some((100, 100)));
    }
    for query in [
        Box::new(RegexQuery::from_pattern("absent.*", text)?) as Box<dyn Query>,
        Box::new(FuzzyTermQuery::new(
            Term::from_field_text(text, "absent"),
            1,
            true,
        )),
        Box::new(FuzzyTermQuery::new(
            Term::from_field_text(text, "raer"),
            1,
            false,
        )),
    ] {
        assert_eq!(query.count(&searcher)?, 0);
        assert_eq!(query.estimate_docs(reader)?, Some((0, 0)));
    }
    let (count, cost) = RegexQuery::from_pattern("common|rare", text)?
        .estimate_docs(reader)?
        .unwrap();
    assert!((600..=700).contains(&count));
    assert_eq!(cost, 700);
    assert_eq!(
        RegexQuery::from_pattern(".*", text)?
            .estimate_docs(reader)?
            .unwrap()
            .0,
        1000
    );
    Ok(())
}

#[test]
fn metadata_estimates_phrases_and_slop() -> crate::Result<()> {
    let (index, _writer, text, _) = fixture()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    let mut phrase = PhraseQuery::new(vec![
        Term::from_field_text(text, "all"),
        Term::from_field_text(text, "common"),
    ]);
    let strict = phrase.estimate_docs(reader)?.unwrap();
    assert!(strict.0 > 0 && strict.0 < 600);
    assert!(strict.1 > u64::from(strict.0));
    phrase.set_slop(4);
    let relaxed = phrase.estimate_docs(reader)?.unwrap();
    assert!(relaxed.0 > strict.0 && relaxed.0 <= 600);
    phrase.set_slop(u32::MAX);
    assert_eq!(phrase.estimate_docs(reader)?.unwrap().0, 600);
    let prefix = PhrasePrefixQuery::new(vec![
        Term::from_field_text(text, "all"),
        Term::from_field_text(text, "comm"),
    ]);
    let regex = RegexPhraseQuery::new(text, vec!["all".into(), "comm.*".into()]);
    assert_eq!(prefix.estimate_docs(reader)?, Some(strict));
    assert_eq!(regex.estimate_docs(reader)?, Some(strict));
    for query in [
        Box::new(PhraseQuery::new(vec![
            Term::from_field_text(text, "all"),
            Term::from_field_text(text, "absent"),
        ])) as Box<dyn Query>,
        Box::new(PhrasePrefixQuery::new(vec![
            Term::from_field_text(text, "all"),
            Term::from_field_text(text, "absent"),
        ])),
        Box::new(RegexPhraseQuery::new(
            text,
            vec!["all".into(), "absent.*".into()],
        )),
    ] {
        assert_eq!(query.estimate_docs(reader)?, Some((0, 0)));
    }
    Ok(())
}

#[test]
fn metadata_estimates_fuzzy_json_path() -> crate::Result<()> {
    let mut schema = Schema::builder();
    let json = schema.add_json_field("json", TEXT);
    let index = Index::create_in_ram(schema.build());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(json => serde_json::json!({"a": "japan", "aa": "japan"})))?;
    writer.add_document(doc!(json => serde_json::json!({"aa": "japan"})))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    for (path, count) in [("a", 1), ("aa", 2), ("missing", 0)] {
        let mut term = Term::from_field_json_path(json, path, false);
        term.append_type_and_str("japam");
        let query = FuzzyTermQuery::new(term, 1, true);
        assert_eq!(query.count(&searcher)?, count as usize);
        assert_eq!(
            query.estimate_docs(searcher.segment_reader(0))?,
            Some((count, u64::from(count)))
        );
    }
    Ok(())
}

#[test]
fn metadata_estimates_limit_expansion_without_using_partial_counts() -> crate::Result<()> {
    use super::query_estimate::MAX_ESTIMATED_TERMS;

    let mut schema = Schema::builder();
    let text = schema.add_text_field("text", TEXT);
    let index = Index::create_in_ram(schema.build());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    let body = (0..=MAX_ESTIMATED_TERMS)
        .map(|id| format!("token{id:05}"))
        .collect::<Vec<_>>()
        .join(" ");
    writer.add_document(doc!(text => body))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    assert_eq!(
        RegexQuery::from_pattern("token.*", text)?.estimate_docs(reader)?,
        None
    );
    assert_eq!(
        RegexQuery::from_pattern(".*absent", text)?.estimate_docs(reader)?,
        None
    );
    assert_eq!(
        RegexQuery::from_pattern("token04096", text)?.estimate_docs(reader)?,
        Some((1, 1))
    );
    assert_eq!(
        FuzzyTermQuery::new(Term::from_field_text(text, "token04096"), 0, false)
            .estimate_docs(reader)?,
        Some((1, 1))
    );
    assert_eq!(
        FuzzyTermQuery::new_prefix(Term::from_field_text(text, "token"), 0, false)
            .estimate_docs(reader)?,
        None
    );
    let mut prefix = PhrasePrefixQuery::new(vec![Term::from_field_text(text, "token")]);
    prefix.set_max_expansions(MAX_ESTIMATED_TERMS as u32);
    assert_eq!(
        prefix.estimate_docs(reader)?,
        Some((1, MAX_ESTIMATED_TERMS as u64))
    );
    prefix.set_max_expansions(MAX_ESTIMATED_TERMS as u32 + 1);
    assert_eq!(prefix.estimate_docs(reader)?, None);
    prefix.set_max_expansions(0);
    assert_eq!(prefix.estimate_docs(reader)?, Some((0, 0)));
    let mut regex_phrase =
        RegexPhraseQuery::new(text, vec!["token00000".into(), "token00001".into()]);
    regex_phrase.set_max_expansions(1);
    assert_eq!(regex_phrase.estimate_docs(reader)?, None);
    regex_phrase.set_max_expansions(2);
    assert!(regex_phrase.estimate_docs(reader)?.is_some());
    Ok(())
}

#[test]
fn metadata_estimates_mlt_without_loading_source() -> crate::Result<()> {
    let (index, _writer, _, _) = fixture()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    let queries = [
        MoreLikeThisQuery::builder().with_document(crate::DocAddress::new(u32::MAX, u32::MAX)),
        MoreLikeThisQuery::builder().with_document_fields(vec![]),
    ];
    for query in queries {
        assert_eq!(query.estimate_docs(reader)?, Some((10, 1000)));
        let wrapped = ConstScoreQuery::new(Box::new(BoostQuery::new(Box::new(query), 5.0)), 3.0);
        assert_eq!(wrapped.estimate_docs(reader)?, Some((10, 1000)));
    }
    Ok(())
}
