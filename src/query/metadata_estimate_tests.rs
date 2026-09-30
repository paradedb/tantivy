use super::{
    AllQuery, BooleanQuery, BoostQuery, ConstScoreQuery, EmptyQuery, Occur, PhraseQuery, Query,
    QueryClone, QueryEstimate, TermQuery,
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
        Box::new(PhraseQuery::new(vec![
            Term::from_field_text(text, "all"),
            Term::from_field_text(text, "common"),
        ])),
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
