use tantivy::collector::Count;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::{BooleanQuery, EnableScoring, Occur, Query, QueryParser};
use tantivy::schema::{Schema, INDEXED, TEXT};
use tantivy::{doc, DocSet, Index, Term};

#[test]
fn two_term_counts_match_union_enumeration() -> tantivy::Result<()> {
    for stored_bitmaps in [false, true] {
        let mut schema = Schema::builder();
        let options = TEXT.set_indexing_options(
            TEXT.get_indexing_options()
                .unwrap()
                .clone()
                .set_bitmap_postings(stored_bitmaps),
        );
        let text = schema.add_text_field("text", options.clone());
        let other = schema.add_text_field("other", options);
        let id = schema.add_u64_field("id", INDEXED);
        let mut index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for i in 0..4097u64 {
            let mut terms = vec!["all"];
            for (term, divisor, remainder) in [
                ("dense", 2, 0),
                ("second", 3, 0),
                ("medium", 17, 0),
                ("sparse", 101, 0),
                ("disjoint", 101, 1),
            ] {
                if i % divisor == remainder {
                    terms.push(term);
                }
            }
            if i < 256 {
                terms.push("prefix");
            }
            if i >= 3840 {
                terms.push("suffix");
            }
            if i % 11 == 0 {
                terms.push("repeat repeat repeat");
            }
            writer.add_document(doc!(text => terms.join(" "), other => if i % 7 == 0 { "rare" } else { "common" }, id => i))?;
            if i == 2047 {
                writer.commit()?;
            }
        }
        writer.commit()?;
        let parser = QueryParser::for_index(&index, vec![text]);
        let terms = [
            "all",
            "dense",
            "second",
            "medium",
            "sparse",
            "disjoint",
            "prefix",
            "suffix",
            "repeat",
            "missing",
            "other:rare",
        ];
        let mut queries = Vec::<Box<dyn Query>>::new();
        for left in terms {
            for right in terms {
                queries.push(parser.parse_query(&format!("{left} OR {right}"))?);
            }
        }
        for expression in [
            "dense OR sparse OR second",
            "dense OR \"all sparse\"",
            "dense OR (sparse AND medium)",
            "dense AND sparse",
            "dense -sparse",
        ] {
            queries.push(parser.parse_query(expression)?);
        }
        queries.push(Box::new(BooleanQuery::with_minimum_required_clauses(
            vec![
                (Occur::Should, parser.parse_query("dense")?),
                (Occur::Should, parser.parse_query("sparse")?),
            ],
            2,
        )));
        for deleted in [false, true] {
            if deleted {
                for value in [0, 1, 127, 128, 1023, 2047, 2048, 4096] {
                    writer.delete_term(Term::from_field_u64(id, value));
                }
                writer.commit()?;
            }
            for enabled in [false, true] {
                index.settings_mut().bitmap_postings.use_for_queries = enabled;
                let searcher = index.reader()?.searcher();
                assert_eq!(searcher.segment_readers().len(), 2);
                for query in &queries {
                    let ordinary =
                        query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
                    let expected = searcher
                        .segment_readers()
                        .iter()
                        .map(|reader| {
                            let mut scorer = ordinary.scorer(reader, 1.0).unwrap();
                            if let Some(alive) = reader.alive_bitset() {
                                scorer.count(alive)
                            } else {
                                scorer.count_including_deleted()
                            }
                        })
                        .sum::<u32>();
                    for opt_in in [false, true] {
                        let weight = query.weight(
                            EnableScoring::disabled_from_searcher(&searcher)
                                .with_bitmap_postings(opt_in),
                        )?;
                        let actual = searcher
                            .segment_readers()
                            .iter()
                            .map(|reader| weight.count(reader).unwrap())
                            .sum::<u32>();
                        assert_eq!(
                            actual, expected,
                            "{query:?}, stored={stored_bitmaps}, enabled={enabled}, \
                             opt_in={opt_in}, deleted={deleted}"
                        );
                    }
                    assert_eq!(searcher.search(query.as_ref(), &Count)?, expected as usize);
                }
            }
        }
    }
    Ok(())
}
