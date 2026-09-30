use super::{
    AllQuery, BooleanQuery, BoostQuery, ConstScoreQuery, EmptyQuery, Occur, PhraseQuery, Query,
    QueryClone, TermQuery,
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
fn metadata_estimates_terms_and_flat_booleans() -> crate::Result<()> {
    let (index, _writer, text, _) = fixture()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    for (word, count) in [("all", 1000), ("common", 600), ("rare", 100), ("absent", 0)] {
        assert_eq!(
            term(text, word).estimate_docs(reader)?,
            Some((count, u64::from(count)))
        );
    }
    for (words, and_count, and_cost, or_count, or_cost) in [
        (vec!["common", "rare"], 72, 100, 522, 700),
        (vec!["all", "rare"], 100, 100, 1000, 1000),
        (vec!["absent", "rare"], 0, 0, 100, 100),
        (vec!["absent", "absent"], 0, 0, 0, 0),
        (vec!["all", "all"], 1000, 1000, 1000, 1000),
        (vec!["rare"], 100, 100, 100, 100),
        (vec!["rare", "rare"], 12, 100, 154, 200),
        (vec!["singleton", "unique"], 0, 1, 2, 2),
        (vec!["left", "right"], 300, 500, 640, 1000),
    ] {
        for occur in [Occur::Must, Occur::Should] {
            let query =
                BooleanQuery::new(words.iter().map(|word| (occur, term(text, word))).collect());
            let expected = if occur == Occur::Must {
                (and_count, and_cost)
            } else {
                (or_count, or_cost)
            };
            assert_eq!(
                query.estimate_docs(reader)?,
                Some(expected),
                "{occur:?}: {words:?}"
            );
        }
    }
    let rounded_zero =
        BooleanQuery::intersection(vec![term(text, "singleton"), term(text, "unique")]);
    assert_eq!(rounded_zero.estimate_docs(reader)?, Some((0, 1)));
    assert_eq!(rounded_zero.count(&searcher)?, 1);
    assert_eq!(AllQuery.estimate_docs(reader)?, Some((1000, 1000)));
    assert_eq!(EmptyQuery.estimate_docs(reader)?, Some((0, 0)));
    assert_eq!(
        BooleanQuery::new(vec![]).estimate_docs(reader)?,
        Some((0, 0))
    );
    assert_eq!(
        BoostQuery::new(term(text, "rare"), 2.0).estimate_docs(reader)?,
        Some((100, 100))
    );
    assert_eq!(
        ConstScoreQuery::new(term(text, "rare"), 2.0).estimate_docs(reader)?,
        Some((100, 100))
    );
    Ok(())
}

#[test]
fn metadata_estimates_compose_wrappers_and_booleans() -> crate::Result<()> {
    let (index, _writer, text, _) = fixture()?;
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
    for (inner_occur, outer_occur, expected) in [
        (Occur::Must, Occur::Must, (52, 100)),
        (Occur::Must, Occur::Should, (510, 700)),
        (Occur::Should, Occur::Must, (376, 600)),
        (Occur::Should, Occur::Should, (697, 1300)),
    ] {
        for leaf_wrapper in 0..6 {
            for inner_wrapper in 0..6 {
                for outer_wrapper in 0..6 {
                    let inner = BooleanQuery::new(vec![
                        (inner_occur, wrap(term(text, "rare"), leaf_wrapper)),
                        (inner_occur, wrap(term(text, "common"), leaf_wrapper)),
                    ]);
                    let outer = BooleanQuery::new(vec![
                        (outer_occur, wrap(Box::new(inner), inner_wrapper)),
                        (outer_occur, wrap(term(text, "common"), leaf_wrapper)),
                    ]);
                    let query = wrap(Box::new(outer), outer_wrapper);
                    assert_eq!(query.estimate_docs(reader)?, Some(expected), "{query:?}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn metadata_estimates_preserve_nested_costs() -> crate::Result<()> {
    let (index, _writer, text, _) = fixture()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    let rounded_zero =
        BooleanQuery::intersection(vec![term(text, "singleton"), term(text, "unique")]);
    let nested = BooleanQuery::union(vec![term(text, "common"), term(text, "rare")]);
    let rounded_all = BooleanQuery::union((0..20).map(|_| term(text, "common")).collect());
    assert_eq!(rounded_all.estimate_docs(reader)?, Some((1000, 12000)));
    for child in [&rounded_zero, &nested, &rounded_all] {
        for (occur, identity) in [
            (Occur::Must, Box::new(AllQuery) as Box<dyn Query>),
            (Occur::Should, Box::new(EmptyQuery)),
        ] {
            for children in [
                vec![(occur, child.box_clone())],
                vec![(occur, child.box_clone()), (occur, identity)],
            ] {
                let query = BooleanQuery::new(children);
                assert_eq!(query.estimate_docs(reader)?, child.estimate_docs(reader)?);
            }
        }
    }
    let query = BooleanQuery::intersection(vec![rounded_zero.box_clone(), term(text, "rare")]);
    assert_eq!(query.estimate_docs(reader)?, Some((0, 1)));
    assert_eq!(query.count(&searcher)?, 1);
    let query = BooleanQuery::union(vec![rounded_zero.box_clone(), term(text, "unique")]);
    assert_eq!(query.estimate_docs(reader)?, Some((1, 2)));
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
    let query = BooleanQuery::intersection(vec![term(text, "all"), term(text, "rare")]);
    assert_eq!(query.estimate_docs(reader)?, Some((100, 100)));
    Ok(())
}

#[test]
fn metadata_estimates_leave_unsupported_queries_to_the_caller() -> crate::Result<()> {
    let (index, _writer, text, number) = fixture()?;
    let searcher = index.reader()?.searcher();
    let reader = searcher.segment_reader(0);
    let unsupported: Vec<Box<dyn Query>> = vec![
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
