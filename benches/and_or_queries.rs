// Benchmarks boolean conjunction queries using binggan.
//
// What’s measured:
// - Or and And queries with varying selectivity (only `Term` queries for now on leafs)
// - Nested AND/OR combinations (on multiple fields)
// - No-scoring path using the Count collector (focus on iterator/skip performance)
// - Top-K retrieval (k=10) using the TopDocs collector
//
// Corpus model:
// - Synthetic docs; each token a/b/c is independently included per doc
// - If none of a/b/c are included, emit a neutral filler token to keep doc length similar
//
// Notes:
// - After optimization, when scoring is disabled Tantivy reads doc-only postings
//   (IndexRecordOption::Basic), avoiding frequency decoding overhead.
// - This bench isolates boolean iteration speed and intersection/union cost.
// - Use `cargo bench --bench boolean_conjunction` to run.

use binggan::BenchRunner;
use rand::Rng;
use tantivy::collector::sort_key::SortByStaticFastValue;
use tantivy::collector::{Count, TopDocs};
use tantivy::Order;
mod bench_utils;
use bench_utils::*;

fn main() {
    // terms with varying selectivity, ordered from rarest to most common.
    // With 1M docs, we expect:
    // a: 0.01% (100), b: 1% (10k), c: 5% (50k), d: 15% (150k), e: 30% (300k)
    let num_docs = 1_000_000;
    let terms: &[(&str, f32)] = &[
        ("a", 0.0001),
        ("b", 0.01),
        ("c", 0.05),
        ("d", 0.15),
        ("e", 0.30),
    ];

    let queries: &[(&str, &[&str])] = &[
        (
            "only_union",
            &["c OR b", "c OR b OR d", "c OR e", "e OR a"] as &[&str],
        ),
        (
            "only_intersection",
            &["+c +b", "+c +b +d", "+c +e", "+e +a"] as &[&str],
        ),
        (
            "union_intersection",
            &["+c +(b OR d)", "+e +(c OR a)", "+(c OR b) +(d OR e)"] as &[&str],
        ),
    ];

    let mut runner = BenchRunner::new();
    let (only_title, title_and_body) = build_index(num_docs, 1, terms, |_, _, _| {
        (1, 0)
    });
    let term_pcts: Vec<(&str, String)> = terms
        .iter()
        .map(|&(term, p)| (term, format_pct(p)))
        .collect();

    for (view_name, bench_index) in [
        ("single_field", only_title),
        ("multi_field", title_and_body),
    ] {
        for (category_name, category_queries) in queries {
            for query_str in *category_queries {
                let mut group = runner.new_group();
                let query_label = query_label(query_str, &term_pcts);
                group.set_name(format!("{}_{}_{}", view_name, category_name, query_label));
                add_bench_task(&mut group, &bench_index, query_str, Count, "count");
                add_bench_task(
                    &mut group,
                    &bench_index,
                    query_str,
                    TopDocs::with_limit(10).order_by_score(),
                    "top10_inv_idx",
                );
                add_bench_task(
                    &mut group,
                    &bench_index,
                    query_str,
                    (Count, TopDocs::with_limit(10).order_by_score()),
                    "count+top10",
                );

                add_bench_task(
                    &mut group,
                    &bench_index,
                    query_str,
                    TopDocs::with_limit(10).order_by_fast_field::<u64>("score", Order::Asc),
                    "top10_by_ff",
                );
                add_bench_task(
                    &mut group,
                    &bench_index,
                    query_str,
                    TopDocs::with_limit(10).order_by((
                        SortByStaticFastValue::<u64>::for_field("score"),
                        SortByStaticFastValue::<u64>::for_field("score2"),
                    )),
                    "top10_by_2ff",
                );

                group.run();
            }
        }
    }
}
