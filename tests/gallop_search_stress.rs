//! Random integration sanity-check: the production gallop path
//! (`term_set_gallop::run`) and the linear path (`TermSetDocSet::advance`)
//! must produce identical sorted DocId sets across random sorted corpora
//! and random query terms. Any divergence between the gallop helper and
//! the linear ground truth surfaces as a symmetric-difference failure
//! here, so this test catches regressions in either path.
//!
//! Direct head-to-head comparison of `binary_search_sorted` vs
//! `gallop_search_sorted` (the helpers themselves) lives inside the crate
//! at `sorted_internals::gallop_tests`, where both `pub(crate)` helpers
//! are reachable.

use rand::prelude::*;
use rand::rngs::StdRng;
use tantivy::collector::DocSetCollector;
use tantivy::query::{FastFieldTermSetQuery, TermSetStrategyConfig};
use tantivy::schema::{NumericOptions, SchemaBuilder};
use tantivy::{Index, IndexSettings, IndexSortByField, Order, ReloadPolicy, Searcher, Term, doc};

fn cfg_gallop() -> TermSetStrategyConfig {
    TermSetStrategyConfig {
        gallop_enabled: true,
        gallop_max_density: 1.0,
        ..TermSetStrategyConfig::default()
    }
}

fn cfg_linear() -> TermSetStrategyConfig {
    TermSetStrategyConfig {
        gallop_enabled: false,
        gallop_max_density: 0.0,
        posting_max_density: 0.0,
        bitset_max_density: 0.0,
        hash_probe_max_density: 0.0,
        subsequent_bitset_max_density: 0.0,
        strategy_sink: None,
    }
}

fn build_sorted_index(
    n: usize,
    seed: u64,
    spread: u64,
    order: Order,
) -> (Searcher, tantivy::schema::Field) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut values: Vec<u64> = (0..n).map(|_| rng.random_range(0..spread)).collect();
    let mut sb = SchemaBuilder::new();
    let f = sb.add_u64_field("v", NumericOptions::default().set_fast().set_indexed());
    let schema = sb.build();
    let index = Index::builder()
        .schema(schema)
        .settings(IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "v".to_string(),
                order,
            }),
            ..Default::default()
        })
        .create_in_ram()
        .unwrap();
    let mut writer = index.writer_with_num_threads(1, 50_000_000).unwrap();
    // Avoid sorting locally — let the index sort it, matching production.
    values.shuffle(&mut rng);
    for v in values {
        writer.add_document(doc!(f => v)).unwrap();
    }
    writer.commit().unwrap();
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    (reader.searcher(), f)
}

fn collect(
    searcher: &Searcher,
    field: tantivy::schema::Field,
    term: u64,
    cfg: TermSetStrategyConfig,
) -> Vec<u32> {
    let q = FastFieldTermSetQuery::new(std::iter::once(Term::from_field_u64(field, term)))
        .with_strategy_config(cfg);
    let mut docs: Vec<u32> = searcher
        .search(&q, &DocSetCollector)
        .unwrap()
        .into_iter()
        .map(|a| a.doc_id)
        .collect();
    docs.sort_unstable();
    docs
}

#[test]
fn gallop_path_matches_linear_path_across_random_corpora() {
    // 4 (size, seed, spread, order) configurations. For each: 50 random
    // single-value queries (some in-range, some out). Total ~200
    // single-query comparisons; the gallop and linear strategies must
    // return identical DocId sets on each.
    let corpora: &[(usize, u64, u64, Order)] = &[
        (32, 1, 100, Order::Asc),
        (1024, 2, 1_000, Order::Asc),
        (10_000, 3, 5_000, Order::Asc),
        (100_000, 4, 50_000, Order::Asc),
        (32, 5, 100, Order::Desc),
        (1024, 6, 1_000, Order::Desc),
        (10_000, 7, 5_000, Order::Desc),
    ];

    for &(n, seed, spread, order) in corpora {
        let (searcher, f) = build_sorted_index(n, seed, spread, order);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(31));
        for _ in 0..50 {
            // Random target: half in-range, quarter below, quarter above.
            let target: u64 = match rng.random_range(0..4) {
                0 => 0,
                1 => spread.saturating_add(1000),
                _ => rng.random_range(0..spread),
            };
            let gallop_docs = collect(&searcher, f, target, cfg_gallop());
            let linear_docs = collect(&searcher, f, target, cfg_linear());
            assert_eq!(
                gallop_docs, linear_docs,
                "differential mismatch: corpus(n={n}, seed={seed}, spread={spread}, \
                 order={order:?}) target={target}",
            );
        }
    }
}

#[test]
fn gallop_path_matches_linear_path_on_multi_term_queries() {
    // Multi-term sweep: pick K=20 distinct values, run both strategies,
    // assert identical DocId vectors. Exercises the per-term loop in
    // term_set_gallop::run.
    let (searcher, f) = build_sorted_index(5_000, 42, 2_000, Order::Asc);
    let mut rng = StdRng::seed_from_u64(43);
    for trial in 0..10 {
        let mut terms: Vec<u64> = (0..20).map(|_| rng.random_range(0..2_000u64)).collect();
        terms.sort();
        terms.dedup();

        let q_gallop =
            FastFieldTermSetQuery::new(terms.iter().map(|v| Term::from_field_u64(f, *v)))
                .with_strategy_config(cfg_gallop());
        let q_linear =
            FastFieldTermSetQuery::new(terms.iter().map(|v| Term::from_field_u64(f, *v)))
                .with_strategy_config(cfg_linear());

        let mut g: Vec<u32> = searcher
            .search(&q_gallop, &DocSetCollector)
            .unwrap()
            .into_iter()
            .map(|a| a.doc_id)
            .collect();
        let mut l: Vec<u32> = searcher
            .search(&q_linear, &DocSetCollector)
            .unwrap()
            .into_iter()
            .map(|a| a.doc_id)
            .collect();
        g.sort_unstable();
        l.sort_unstable();
        assert_eq!(g, l, "multi-term mismatch trial={trial}");
    }
}
