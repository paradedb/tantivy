use tantivy::collector::Count;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::{
    BooleanQuery, BoostQuery, ConstScoreQuery, EnableScoring, Occur, Query, QueryParser,
};
use tantivy::schema::{Schema, INDEXED, TEXT};
use tantivy::{doc, Index, Searcher, Term};

fn assert_count(query: &dyn Query, ordinary: &Searcher, bitmap: &Searcher) -> tantivy::Result<()> {
    let weight = query.weight(EnableScoring::disabled_from_searcher(ordinary))?;
    let mut expected = 0usize;
    for reader in ordinary.segment_readers() {
        let mut scorer = weight.scorer(reader, 1.0)?;
        expected += if let Some(alive) = reader.alive_bitset() {
            scorer.count(alive)
        } else {
            scorer.count_including_deleted()
        } as usize;
    }
    let bitmap_weight = query.weight(
        EnableScoring::disabled_from_searcher(bitmap).with_bitmap_postings(true),
    )?;
    let optimized = bitmap.segment_readers().iter().try_fold(0usize, |total, reader| {
        bitmap_weight.count(reader).map(|count| total + count as usize)
    })?;
    assert_eq!(optimized, expected, "{query:?}");
    assert_eq!(bitmap.search(query, &Count)?, expected, "{query:?}");
    assert_eq!(
        bitmap.search(query, &(Count, Count))?,
        (expected, expected),
        "{query:?}"
    );
    Ok(())
}

#[test]
fn sparse_disjunction_counts_match_enumeration() -> tantivy::Result<()> {
    let mut schema = Schema::builder();
    let text = schema.add_text_field("text", TEXT);
    let other = schema.add_text_field("other", TEXT);
    let plain = schema.add_text_field(
        "plain",
        TEXT.set_indexing_options(
            TEXT.get_indexing_options()
                .unwrap()
                .clone()
                .set_bitmap_postings(false),
        ),
    );
    let id = schema.add_u64_field("id", INDEXED);
    let mut index = Index::create_in_ram(schema.build());
    let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for i in 0..131_079u64 {
        let mut terms = vec!["all".to_owned()];
        if i % 5 != 0 {
            terms.push("dense".into());
        }
        if i % 3 != 0 {
            terms.push("second".into());
        }
        if i % 101 == 0 {
            terms.push("medium".into());
        }
        if i % 17011 == 0 {
            terms.push("rare shared unusual phrase".into());
            terms.push(if i % 5 == 0 { "outside" } else { "inside" }.into());
        }
        if i % 2003 == 0 {
            terms.push(format!("r{}", (i / 2003) % 64));
        }
        let value = terms.join(" ");
        writer.add_document(
            doc!(text => value.as_str(), other => value.as_str(), plain => value.as_str(), id => i),
        )?;
        if i == 65_538 {
            writer.commit()?;
        }
    }
    writer.commit()?;
    writer.add_document(doc!(text => "rare shared", other => "dense", id => 200_000u64))?;
    writer.add_document(doc!(text => "all second", plain => "dense", id => 200_001u64))?;
    writer.commit()?;
    let parser = QueryParser::for_index(&index, vec![text]);
    for delete in [false, true] {
        if delete {
            for id_value in [0, 1, 63, 64, 1023, 1024, 17011, 65538, 65539, 131078] {
                writer.delete_term(Term::from_field_u64(id, id_value));
            }
            writer.commit()?;
        }
        index.settings_mut().bitmap_postings.use_for_queries = false;
        let ordinary = index.reader()?.searcher();
        index.settings_mut().bitmap_postings.use_for_queries = true;
        let bitmap = index.reader()?.searcher();
        assert_eq!(bitmap.segment_readers().len(), 3);
        for expression in [
            "dense OR rare",
            "dense OR outside",
            "dense OR inside",
            "dense OR rare OR shared",
            "dense OR dense OR rare",
            "dense OR rare OR rare",
            "(dense OR rare) OR (r0 OR r1)",
            "((rare OR dense) OR r0) OR r1",
            "dense OR \"unusual phrase\"",
            "dense OR (rare AND shared)",
            "dense OR other:rare",
            "dense^3 OR rare",
            "plain:dense OR rare",
            "+(dense OR rare OR r0) -shared",
            "+dense -shared -outside",
            "dense rare -shared",
            "+dense rare -shared",
            "+dense +second -rare",
            "+(dense OR rare) +medium -shared",
            "dense OR medium",
            "dense OR second OR rare",
            "rare OR shared",
            "all OR rare",
            "dense OR nonexistent",
            "nonexistent OR absent",
            "dense OR (rare OR nonexistent)",
            "+(dense OR rare) -all",
            "+all -rare",
            "+(rare OR shared) -dense",
        ] {
            let query = parser.parse_query(expression)?;
            assert_count(query.as_ref(), &ordinary, &bitmap)?;
            assert_count(&BoostQuery::new(query.clone(), 3.0), &ordinary, &bitmap)?;
            assert_count(&ConstScoreQuery::new(query, 7.0), &ordinary, &bitmap)?;
        }
        assert_count(
            &BooleanQuery::new(vec![(Occur::MustNot, parser.parse_query("rare")?)]),
            &ordinary,
            &bitmap,
        )?;
        for count in [1, 3, 15, 63] {
            let terms = std::iter::once("dense".to_owned())
                .chain((0..count).map(|i| format!("r{i}")))
                .collect::<Vec<_>>()
                .join(" OR ");
            assert_count(parser.parse_query(&terms)?.as_ref(), &ordinary, &bitmap)?;
        }
        let mut rng = fastrand::Rng::with_seed(295);
        let expressions = [
            "dense",
            "second",
            "rare",
            "shared",
            "nonexistent",
            "all",
            "other:rare",
            "rare AND shared",
            "dense OR r1",
            "\"unusual phrase\"",
        ];
        for _ in 0..200 {
            let clauses = (0..rng.usize(2..10))
                .map(|_| {
                    let occur = [Occur::Must, Occur::Should, Occur::Should, Occur::MustNot]
                        [rng.usize(0..4)];
                    Ok((
                        occur,
                        parser.parse_query(expressions[rng.usize(0..expressions.len())])?,
                    ))
                })
                .collect::<tantivy::Result<Vec<_>>>()?;
            let query = BooleanQuery::with_minimum_required_clauses(clauses, rng.usize(0..4));
            assert_count(&query, &ordinary, &bitmap)?;
        }
    }
    Ok(())
}
