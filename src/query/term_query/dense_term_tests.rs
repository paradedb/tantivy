use crate::collector::{Count, TopDocs};
use crate::merge_policy::NoMergePolicy;
use crate::postings::{set_dense_term_ratio, set_norm_sidecar_provider};
use crate::query::{BooleanQuery, BoostQuery, EnableScoring, Query, QueryParser, TermQuery};
use crate::schema::{Field, IndexRecordOption, Schema, TEXT};
use crate::{DocAddress, DocSet, Executor, Index, Score, Searcher, Term, TERMINATED};

struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        set_dense_term_ratio(0.0);
        set_norm_sidecar_provider(None);
        crate::postings::set_union_deferred_seeks(false);
        crate::postings::set_postings_read_buffer_size(0);
        crate::postings::set_max_score_bound_mode(0);
    }
}

fn fixture(multiple_segments: bool, delete_positives: bool) -> crate::Result<(Index, Field)> {
    let mut builder = Schema::builder();
    let text = builder.add_text_field("text", TEXT);
    let index = Index::create_in_ram(builder.build());
    let mut writer = index.writer_for_tests()?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for doc in 0..200 {
        let mut body = match doc {
            0..15 => "common rare".to_owned(),
            15..160 => "common filler".to_owned(),
            _ => "filler".to_owned(),
        };
        for (end, word) in [
            (2, " few"),
            (20, " boundary"),
            (19, " below"),
            (1, " local"),
        ] {
            if doc < end {
                body.push_str(word);
            }
        }
        writer.add_document(crate::doc!(text => body))?;
        if multiple_segments && [9, 99].contains(&doc) {
            writer.commit()?;
        }
    }
    writer.commit()?;
    if delete_positives {
        writer.delete_term(Term::from_field_text(text, "few"));
        writer.commit()?;
    }
    Ok((index, text))
}

fn top(
    searcher: &Searcher,
    query: &dyn Query,
    limit: usize,
) -> crate::Result<Vec<(Score, DocAddress)>> {
    searcher.search_with_executor(
        query,
        &TopDocs::with_limit(limit).order_by_score(),
        &Executor::single_thread(),
        EnableScoring::enabled_from_searcher(searcher),
    )
}

fn exhaustive(searcher: &Searcher, query: &dyn Query) -> crate::Result<Vec<(Score, DocAddress)>> {
    let weight = query.weight(EnableScoring::enabled_from_searcher(searcher))?;
    let mut result = Vec::new();
    for (ordinal, segment) in searcher.segment_readers().iter().enumerate() {
        let mut scorer = weight.scorer(segment, 1.0)?;
        while scorer.doc() != TERMINATED {
            if segment
                .alive_bitset()
                .is_none_or(|alive| alive.is_alive(scorer.doc()))
            {
                result.push((
                    scorer.score(),
                    DocAddress::new(ordinal as u32, scorer.doc()),
                ));
            }
            scorer.advance();
        }
    }
    result.sort_unstable_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    Ok(result)
}

#[test]
fn dense_term_keeps_matching_and_zero_score_topk_fallback() -> crate::Result<()> {
    let _reset = Reset;
    for multiple_segments in [false, true] {
        for deleted in [false, true] {
            let (index, text) = fixture(multiple_segments, deleted)?;
            let searcher = index.reader()?.searcher();
            let parser = QueryParser::for_index(&index, vec![text]);
            set_dense_term_ratio(0.1);
            for (defer, mode, buffer) in [(false, 0, 0), (true, 2, 16)] {
                crate::postings::set_union_deferred_seeks(defer);
                crate::postings::set_max_score_bound_mode(mode);
                crate::postings::set_postings_read_buffer_size(buffer);
                for expression in [
                    "common OR rare",
                    "common OR few",
                    "common OR filler",
                    "common",
                    "common AND filler",
                    "common AND rare",
                    "+common rare",
                    "+filler rare",
                    "(common OR rare) -filler",
                    "\"common rare\" OR filler",
                ] {
                    let query = parser.parse_query(expression)?;
                    let all = exhaustive(&searcher, query.as_ref())?;
                    let count = searcher.search(query.as_ref(), &Count)?;
                    assert_eq!(all.len(), count, "{expression}");
                    for limit in [1, 10, 250] {
                        let expected = &all[..limit.min(all.len())];
                        assert_eq!(
                            top(&searcher, query.as_ref(), limit)?,
                            expected,
                            "{expression}, multiple={multiple_segments}, deleted={deleted}, \
                             limit={limit}, defer={defer}"
                        );
                    }
                    for offset in [1, 10, 20] {
                        let actual = searcher.search_with_executor(
                            query.as_ref(),
                            &TopDocs::with_limit(10).and_offset(offset).order_by_score(),
                            &Executor::single_thread(),
                            EnableScoring::enabled_from_searcher(&searcher),
                        )?;
                        assert_eq!(
                            actual,
                            all[offset.min(all.len())..(offset + 10).min(all.len())],
                            "{expression}, offset={offset}"
                        );
                    }
                }
            }
            let few = parser.parse_query("common OR few")?;
            let actual = top(&searcher, few.as_ref(), 10)?;
            assert_eq!(actual.len(), 10);
            assert_eq!(
                actual.iter().filter(|(score, _)| *score > 0.0).count(),
                if deleted { 0 } else { 2 }
            );
            let all_dense = parser.parse_query("common OR filler")?;
            assert!(top(&searcher, all_dense.as_ref(), 10)?
                .iter()
                .all(|(score, _)| *score == 0.0));
        }
    }
    Ok(())
}

#[test]
fn dense_term_matches_positive_cutoff_rewrite_and_preserves_phrases() -> crate::Result<()> {
    let _reset = Reset;
    let (index, text) = fixture(true, false)?;
    let searcher = index.reader()?.searcher();
    let parser = QueryParser::for_index(&index, vec![text]);
    let phrase = parser.parse_query("\"common rare\"")?;
    let full = parser.parse_query("common OR rare")?;
    let rare = parser.parse_query("rare")?;
    let common = parser.parse_query("common")?;
    let original_phrase = top(&searcher, phrase.as_ref(), 20)?;
    let original_common = top(&searcher, common.as_ref(), 20)?;
    set_dense_term_ratio(0.1);
    assert_eq!(
        top(&searcher, full.as_ref(), 10)?,
        top(&searcher, rare.as_ref(), 10)?
    );
    assert_eq!(top(&searcher, phrase.as_ref(), 20)?, original_phrase);
    assert!(top(&searcher, common.as_ref(), 20)?
        .iter()
        .all(|(score, _)| *score == 0.0));
    set_dense_term_ratio(0.0);
    assert_eq!(top(&searcher, common.as_ref(), 20)?, original_common);
    Ok(())
}

#[test]
fn dense_term_global_density_boundary_and_required_clauses() -> crate::Result<()> {
    let _reset = Reset;
    let (index, text) = fixture(true, false)?;
    let searcher = index.reader()?.searcher();
    assert_eq!(searcher.segment_readers().len(), 3);
    let term_query = |word| -> Box<dyn Query> {
        Box::new(TermQuery::new(
            Term::from_field_text(text, word),
            IndexRecordOption::WithFreqs,
        ))
    };
    set_dense_term_ratio(0.1);
    assert!(top(&searcher, term_query("boundary").as_ref(), 20)?
        .iter()
        .all(|(score, _)| *score == 0.0));
    for word in ["below", "local"] {
        assert!(top(&searcher, term_query(word).as_ref(), 20)?
            .iter()
            .all(|(score, _)| *score > 0.0));
    }
    for minimum in [1, 2, 3] {
        let query = BooleanQuery::union_with_minimum_required_clauses(
            vec![
                term_query("common"),
                term_query("filler"),
                term_query("rare"),
            ],
            minimum,
        );
        let all = exhaustive(&searcher, &query)?;
        assert_eq!(top(&searcher, &query, 10)?, &all[..10.min(all.len())]);
        assert_eq!(searcher.search(&query, &Count)?, all.len());
    }
    for boost in [-1.0, 0.0, 2.0] {
        let query = BoostQuery::new(term_query("common"), boost);
        let all = exhaustive(&searcher, &query)?;
        assert_eq!(top(&searcher, &query, 10)?, all[..10]);
        assert!(all.iter().all(|(score, _)| *score == 0.0));
    }
    Ok(())
}

#[test]
fn dense_term_never_opens_norm_sidecars() -> crate::Result<()> {
    let _reset = Reset;
    let (index, text) = fixture(false, false)?;
    let searcher = index.reader()?.searcher();
    let parser = QueryParser::for_index(&index, vec![text]);
    set_dense_term_ratio(0.1);
    set_norm_sidecar_provider(Some(|_, _| panic!("zero-score term requested a norm lane")));
    for expression in ["common", "common OR filler", "common AND filler"] {
        let query = parser.parse_query(expression)?;
        assert_eq!(top(&searcher, query.as_ref(), 10)?.len(), 10);
    }
    Ok(())
}

#[test]
fn dense_term_opens_zero_lanes_only_for_zero_score_fallback() -> crate::Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::query::boolean_query::BooleanWeight;
    use crate::query::score_combiner::SumCombiner;
    use crate::query::{Explanation, Occur, Scorer, Weight};
    use crate::{DocId, SegmentReader};

    struct CountedWeight {
        inner: Box<dyn Weight>,
        opens: Arc<AtomicUsize>,
    }

    impl Weight for CountedWeight {
        fn term_score_is_zero(&self) -> Option<bool> {
            self.inner.term_score_is_zero()
        }

        fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
            self.opens.fetch_add(1, Ordering::Relaxed);
            self.inner.scorer(reader, boost)
        }

        fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
            self.inner.explain(reader, doc)
        }
    }

    let _reset = Reset;
    let (index, text) = fixture(false, false)?;
    let searcher = index.reader()?.searcher();
    set_dense_term_ratio(0.1);
    let dense_opens = Arc::new(AtomicUsize::new(0));
    let positive_opens = Arc::new(AtomicUsize::new(0));
    let weights = [
        ("common", dense_opens.clone()),
        ("rare", positive_opens.clone()),
    ]
    .into_iter()
    .map(|(word, opens)| {
        let term = TermQuery::new(
            Term::from_field_text(text, word),
            IndexRecordOption::WithFreqs,
        );
        Ok((
            Occur::Should,
            Box::new(CountedWeight {
                inner: term.weight(EnableScoring::enabled_from_searcher(&searcher))?,
                opens,
            }) as Box<dyn Weight>,
        ))
    })
    .collect::<crate::Result<Vec<_>>>()?;
    let weight = BooleanWeight::new(weights, true, Box::new(SumCombiner::default));
    let segment = searcher.segment_reader(0);
    weight.for_each_pruning(Score::MIN, segment, &mut |_, score| {
        assert!(score > 0.0);
        0.001
    })?;
    assert_eq!(dense_opens.load(Ordering::Relaxed), 0);
    assert_eq!(positive_opens.load(Ordering::Relaxed), 1);

    let mut matches = Vec::new();
    weight.for_each_pruning(Score::MIN, segment, &mut |doc, score| {
        matches.push((doc, score));
        Score::MIN
    })?;
    assert_eq!(dense_opens.load(Ordering::Relaxed), 1);
    assert_eq!(positive_opens.load(Ordering::Relaxed), 3);
    assert_eq!(matches.len(), 160);
    matches.sort_unstable_by_key(|(doc, _)| *doc);
    assert!(matches.windows(2).all(|pair| pair[0].0 < pair[1].0));
    assert_eq!(matches.iter().filter(|(_, score)| *score > 0.0).count(), 15);
    Ok(())
}
