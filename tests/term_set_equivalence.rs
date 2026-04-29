//! End-to-end equivalence test for the gallop strategy.
//!
//! For the same `(corpus, term_set)`, the gallop path and the linear path
//! (today's `TermSetDocSet`) must return identical sorted DocId vectors. We
//! drive both via the public `FastFieldTermSetQuery` API: identical corpus,
//! identical query, only `TermSetStrategyConfig::gallop_enabled` differs.
//!
//! This is the strongest correctness invariant for #4895 — every other test
//! pins planner shape or algorithm details, but this one would catch any
//! semantic divergence between the two strategies before it lands.

use rand::prelude::*;
use rand::rngs::StdRng;
use tantivy::collector::DocSetCollector;
use tantivy::query::{FastFieldTermSetQuery, TermSetStrategyConfig};
use tantivy::schema::{NumericOptions, SchemaBuilder};
use tantivy::{
    doc, Index, IndexSettings, IndexSortByField, Order, ReloadPolicy, Searcher, Term,
};

#[derive(Clone, Copy, Debug)]
enum CorpusKind {
    /// D = 1, every value unique.
    PrimaryKey,
    /// D ≈ 100, foreign-key shape.
    ForeignKey,
}

fn value_for(doc_id: u64, kind: CorpusKind) -> u64 {
    match kind {
        CorpusKind::PrimaryKey => doc_id,
        CorpusKind::ForeignKey => doc_id / 100,
    }
}

fn distinct_count(n: u64, kind: CorpusKind) -> u64 {
    match kind {
        CorpusKind::PrimaryKey => n,
        CorpusKind::ForeignKey => n.div_ceil(100),
    }
}

fn build_sorted_corpus(
    n: u64,
    kind: CorpusKind,
    order: Order,
) -> (Searcher, tantivy::schema::Field) {
    let mut sb = SchemaBuilder::new();
    let field = sb.add_u64_field("fk", NumericOptions::default().set_fast().set_indexed());
    let schema = sb.build();
    let index = Index::builder()
        .schema(schema)
        .settings(IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "fk".to_string(),
                order,
            }),
            ..Default::default()
        })
        .create_in_ram()
        .unwrap();
    {
        let mut writer = index.writer_with_num_threads(1, 50_000_000).unwrap();
        for d in 0..n {
            writer.add_document(doc!(field => value_for(d, kind))).unwrap();
        }
        writer.commit().unwrap();
    }
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    (reader.searcher(), field)
}

fn sample_terms(distinct: u64, k: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut chosen: Vec<u64> = (0..distinct).collect();
    chosen.shuffle(&mut rng);
    chosen.truncate(k);
    chosen
}

fn run_query(
    searcher: &Searcher,
    field: tantivy::schema::Field,
    terms: &[u64],
    cfg: TermSetStrategyConfig,
) -> Vec<u32> {
    let q = FastFieldTermSetQuery::new(terms.iter().map(|v| Term::from_field_u64(field, *v)))
        .with_strategy_config(cfg);
    let result = searcher.search(&q, &DocSetCollector).unwrap();
    let mut docs: Vec<u32> = result.into_iter().map(|addr| addr.doc_id).collect();
    docs.sort_unstable();
    docs
}

fn cfg_gallop_on() -> TermSetStrategyConfig {
    TermSetStrategyConfig::default()
}

fn cfg_gallop_off() -> TermSetStrategyConfig {
    TermSetStrategyConfig {
        gallop_enabled: false,
        ..TermSetStrategyConfig::default()
    }
}

fn assert_equivalent(
    n: u64,
    kind: CorpusKind,
    order: Order,
    k_values: &[usize],
) {
    let (searcher, field) = build_sorted_corpus(n, kind, order);
    let distinct = distinct_count(n, kind);
    for &k in k_values {
        if (k as u64) > distinct {
            continue;
        }
        let terms = sample_terms(distinct, k, 7);
        let gallop_docs = run_query(&searcher, field, &terms, cfg_gallop_on());
        let linear_docs = run_query(&searcher, field, &terms, cfg_gallop_off());
        assert_eq!(
            gallop_docs, linear_docs,
            "strategy divergence: corpus={kind:?} order={order:?} N={n} K={k}",
        );
    }
}

#[test]
fn gallop_matches_linear_pk_asc() {
    assert_equivalent(
        10_000,
        CorpusKind::PrimaryKey,
        Order::Asc,
        &[1, 10, 100, 1_000],
    );
}

#[test]
fn gallop_matches_linear_pk_desc() {
    assert_equivalent(
        10_000,
        CorpusKind::PrimaryKey,
        Order::Desc,
        &[1, 10, 100, 1_000],
    );
}

#[test]
fn gallop_matches_linear_fk_asc() {
    assert_equivalent(
        10_000,
        CorpusKind::ForeignKey,
        Order::Asc,
        &[1, 5, 25, 100],
    );
}

#[test]
fn gallop_matches_linear_fk_desc() {
    assert_equivalent(
        10_000,
        CorpusKind::ForeignKey,
        Order::Desc,
        &[1, 5, 25, 100],
    );
}

/// Edge case: term set partially outside [min, max]. Pruning must agree
/// between the two strategies; both should produce the same DocIds.
#[test]
fn gallop_matches_linear_with_out_of_range_terms() {
    let (searcher, field) = build_sorted_corpus(10_000, CorpusKind::ForeignKey, Order::Asc);
    // Distinct values are [0, 100). Mix in some out-of-range terms (200, 999).
    let terms = vec![5u64, 17, 42, 99, 200, 999];
    let gallop_docs = run_query(&searcher, field, &terms, cfg_gallop_on());
    let linear_docs = run_query(&searcher, field, &terms, cfg_gallop_off());
    assert_eq!(gallop_docs, linear_docs);
    // And it should be non-empty (5, 17, 42, 99 are all in range with ~100 docs each).
    assert!(!gallop_docs.is_empty());
}

/// Edge case: term set entirely outside [min, max]. Both strategies must
/// return EmptyScorer / no docs.
#[test]
fn gallop_and_linear_both_empty_when_all_terms_pruned() {
    let (searcher, field) = build_sorted_corpus(10_000, CorpusKind::ForeignKey, Order::Asc);
    let terms = vec![10_000u64, 99_999, u64::MAX];
    let gallop_docs = run_query(&searcher, field, &terms, cfg_gallop_on());
    let linear_docs = run_query(&searcher, field, &terms, cfg_gallop_off());
    assert!(gallop_docs.is_empty());
    assert!(linear_docs.is_empty());
}
