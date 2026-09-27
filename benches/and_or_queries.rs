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

use binggan::{black_box, BenchRunner};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::sort_key::SortByStaticFastValue;
use tantivy::collector::{Collector, Count, TopDocs};
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, FAST, TEXT};
use tantivy::{doc, Index, Order, ReloadPolicy, Searcher};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PostingNorms {
    Disabled,
    Enabled,
}

#[derive(Clone)]
struct BenchIndex {
    #[allow(dead_code)]
    index: Index,
    searcher: Searcher,
    query_parser: QueryParser,
}

/// Build a single index containing both fields (title, body) and
/// return two BenchIndex views:
/// - single_field: QueryParser defaults to only "body"
/// - multi_field:  QueryParser defaults to ["title", "body"]
fn build_index(
    num_docs: usize,
    terms: &[(&str, f32)],
    pnorms: PostingNorms,
) -> (BenchIndex, BenchIndex) {
    // Unified schema (two text fields)
    let mut schema_builder = Schema::builder();
    let text_options = match pnorms {
        PostingNorms::Enabled => {
            let indexing = TEXT
                .get_indexing_options()
                .unwrap()
                .clone()
                .set_pnorms(true);
            TEXT.set_indexing_options(indexing)
        }
        PostingNorms::Disabled => TEXT,
    };
    let f_title = schema_builder.add_text_field("title", text_options.clone());
    let f_body = schema_builder.add_text_field("body", text_options);
    let f_score = schema_builder.add_u64_field("score", FAST);
    let f_score2 = schema_builder.add_u64_field("score2", FAST);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());

    // Populate index with stable RNG for reproducibility.
    let mut rng = StdRng::from_seed([7u8; 32]);

    // Populate: spread each present token 90/10 to body/title
    {
        let mut writer = index.writer_with_num_threads(1, 500_000_000).unwrap();
        for _ in 0..num_docs {
            let score = rng.random_range(0u64..100u64);
            let score2 = rng.random_range(0u64..100_000u64);
            let mut title_tokens: Vec<&str> = Vec::new();
            let mut body_tokens: Vec<&str> = Vec::new();
            for &(tok, prob) in terms {
                if rng.random_bool(prob as f64) {
                    if rng.random_bool(0.1) {
                        title_tokens.push(tok);
                    } else {
                        body_tokens.push(tok);
                    }
                }
            }
            if title_tokens.is_empty() && body_tokens.is_empty() {
                body_tokens.push("z");
            }
            writer
                .add_document(doc!(
                    f_title=>title_tokens.join(" "),
                    f_body=>body_tokens.join(" "),
                    f_score=>score,
                    f_score2=>score2,
                ))
                .unwrap();
        }
        writer.commit().unwrap();
    }

    // Prepare reader/searcher once.
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    let searcher = reader.searcher();

    // Build two query parsers with different default fields.
    let qp_single = QueryParser::for_index(&index, vec![f_body]);
    let qp_multi = QueryParser::for_index(&index, vec![f_title, f_body]);

    let only_title = BenchIndex {
        index: index.clone(),
        searcher: searcher.clone(),
        query_parser: qp_single,
    };
    let title_and_body = BenchIndex {
        index,
        searcher,
        query_parser: qp_multi,
    };
    (only_title, title_and_body)
}

fn format_pct(p: f32) -> String {
    let pct = (p as f64) * 100.0;
    let rounded = (pct * 1_000_000.0).round() / 1_000_000.0;
    if rounded.fract() <= 0.001 {
        format!("{}%", rounded as u64)
    } else {
        format!("{}%", rounded)
    }
}

fn query_label(query_str: &str, term_pcts: &[(&str, String)]) -> String {
    let mut label = query_str.to_string();
    for (term, pct) in term_pcts {
        label = label.replace(term, pct);
    }
    label.replace(' ', "_")
}

fn main() {
    println!("PID: {}", std::process::id());
    println!("Sleeping for 10 seconds before starting benchmark...");
    std::thread::sleep(std::time::Duration::from_secs(10));

    // Can be disabled via BENCH_BASELINE=0 / false, or SKIP_BASELINE=1 / true.
    let run_baseline = match std::env::var("BENCH_BASELINE") {
        Ok(v) => v != "0" && v.to_lowercase() != "false",
        Err(_) => std::env::var("SKIP_BASELINE")
            .map(|v| v == "0" || v.to_lowercase() == "false")
            .unwrap_or(true),
    };
    if !run_baseline {
        println!("Baseline variant disabled via environment variable.");
    }

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
    if let Some(num_iter) = std::env::var("NUM_ITER_GROUP")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        runner.config().set_num_iter_for_group(num_iter);
    }

    let bench_indices_baseline = if run_baseline {
        Some(build_index(num_docs, terms, PostingNorms::Disabled))
    } else {
        None
    };
    let bench_indices_pnorms = build_index(num_docs, terms, PostingNorms::Enabled);

    let term_pcts: Vec<(&str, String)> = terms
        .iter()
        .map(|&(term, p)| (term, format_pct(p)))
        .collect();

    let views = match &bench_indices_baseline {
        Some((b_single, b_multi)) => vec![
            ("single_field", Some(b_single), &bench_indices_pnorms.0),
            ("multi_field", Some(b_multi), &bench_indices_pnorms.1),
        ],
        None => vec![
            ("single_field", None, &bench_indices_pnorms.0),
            ("multi_field", None, &bench_indices_pnorms.1),
        ],
    };

    for (view_name, bench_index_baseline, bench_index_pnorms) in views {
        for (category_name, category_queries) in queries {
            for query_str in *category_queries {
                let query_label = query_label(query_str, &term_pcts);
                let base_name = format!("{}_{}_{}", view_name, category_name, query_label);

                add_bench_task(
                    &mut runner,
                    format!("{base_name} — count"),
                    bench_index_baseline,
                    bench_index_pnorms,
                    query_str,
                    || Count,
                );
                add_bench_task(
                    &mut runner,
                    format!("{base_name} — top10_inv_idx"),
                    bench_index_baseline,
                    bench_index_pnorms,
                    query_str,
                    || TopDocs::with_limit(10).order_by_score(),
                );
                add_bench_task(
                    &mut runner,
                    format!("{base_name} — count+top10"),
                    bench_index_baseline,
                    bench_index_pnorms,
                    query_str,
                    || (Count, TopDocs::with_limit(10).order_by_score()),
                );
                add_bench_task(
                    &mut runner,
                    format!("{base_name} — top10_by_ff"),
                    bench_index_baseline,
                    bench_index_pnorms,
                    query_str,
                    || TopDocs::with_limit(10).order_by_fast_field::<u64>("score", Order::Asc),
                );
                add_bench_task(
                    &mut runner,
                    format!("{base_name} — top10_by_2ff"),
                    bench_index_baseline,
                    bench_index_pnorms,
                    query_str,
                    || {
                        TopDocs::with_limit(10).order_by((
                            SortByStaticFastValue::<u64>::for_field("score"),
                            SortByStaticFastValue::<u64>::for_field("score2"),
                        ))
                    },
                );
            }
        }
    }
}

trait FruitCount {
    fn count(&self) -> usize;
}

impl FruitCount for usize {
    fn count(&self) -> usize {
        *self
    }
}

impl<T> FruitCount for Vec<T> {
    fn count(&self) -> usize {
        self.len()
    }
}

impl<A: FruitCount, B> FruitCount for (A, B) {
    fn count(&self) -> usize {
        self.0.count()
    }
}

fn add_bench_task<C: Collector + 'static, F: Fn() -> C + Copy + 'static>(
    runner: &mut BenchRunner,
    group_name: String,
    bench_index_baseline: Option<&BenchIndex>,
    bench_index_pnorms: &BenchIndex,
    query_str: &str,
    make_collector: F,
) where
    C::Fruit: FruitCount,
{
    let mut group = runner.new_group();
    group.set_name(group_name);

    if let Some(bench_index_baseline) = bench_index_baseline {
        let query = bench_index_baseline
            .query_parser
            .parse_query(query_str)
            .unwrap();
        let searcher = bench_index_baseline.searcher.clone();
        let collector = make_collector();
        group.register("baseline", move |_| {
            black_box(searcher.search(&query, &collector).unwrap().count())
        });
    }

    let query = bench_index_pnorms
        .query_parser
        .parse_query(query_str)
        .unwrap();
    let searcher = bench_index_pnorms.searcher.clone();
    let collector = make_collector();
    group.register("pnorms", move |_| {
        black_box(searcher.search(&query, &collector).unwrap().count())
    });

    group.run();
}
