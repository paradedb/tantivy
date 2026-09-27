// Benchmarks top-K intersection of term scorers (block_wand_intersection).
//
// What's measured:
// - Conjunctive queries (2, 3, 5, and 10 terms) with top-10 by score
// - Varying doc-frequency balance between terms (balanced, skewed, cascading)
// - Realistic term frequencies (geometric distribution, mostly low)
// - 1M-doc single segment
//
// Run with: cargo bench --bench intersection_bench
// Optional environment variables:
// - `BENCH_SCENARIO=<filter>`: only run scenarios matching the filter substring (e.g.
//   `BENCH_SCENARIO=ten`)
// - `BENCH_BASELINE=0` or `SKIP_BASELINE=1`: skip the baseline variant
// - `NUM_ITER_GROUP=<n>`: override iterations per group (default: 320)

use binggan::{black_box, BenchRunner};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, TEXT};
use tantivy::{doc, Index, ReloadPolicy, Searcher};

const NUM_DOCS: usize = 1_000_000;

const TERMS: [&str; 10] = [
    "aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh", "iii", "jjj",
];

#[derive(Clone, Copy, Debug)]
struct TermSpec {
    term: &'static str,
    doc_freq_prob: f64,
}

impl TermSpec {
    fn new(term: &'static str, doc_freq_prob: f64) -> Self {
        Self {
            term,
            doc_freq_prob,
        }
    }
}

fn term_spec_uniform(count: usize, prob: f64) -> Vec<TermSpec> {
    TERMS[..count]
        .iter()
        .map(|&term| TermSpec::new(term, prob))
        .collect()
}

fn conjunction_query(count: usize) -> String {
    TERMS[..count]
        .iter()
        .map(|term| format!("+{term}"))
        .collect::<Vec<_>>()
        .join(" ")
}

struct Scenario {
    label: &'static str,
    terms: Vec<TermSpec>,
    queries: Vec<String>,
}

struct BenchIndex {
    searcher: Searcher,
    query_parser: QueryParser,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PostingNorms {
    Disabled,
    Enabled,
}

/// Generate term frequency from a geometric-like distribution.
/// Most values are 1, a few are 2-3, rarely higher.
/// p controls the decay: higher p → more weight on tf=1.
fn random_term_freq(rng: &mut StdRng, p: f64) -> u32 {
    let mut tf = 1u32;
    while tf < 10 && rng.random_bool(1.0 - p) {
        tf += 1;
    }
    tf
}

/// Build an index with specified terms and their doc-frequency probabilities.
/// Each term occurrence has a realistic term frequency (geometric distribution).
/// Field length is padded with filler tokens to create varied fieldnorms.
fn build_index(terms: &[TermSpec], pnorms: PostingNorms) -> BenchIndex {
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
    let body = schema_builder.add_text_field("body", text_options);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);

    let mut rng = StdRng::from_seed([42u8; 32]);

    {
        let mut writer = index.writer_with_num_threads(1, 500_000_000).unwrap();
        let mut tokens: Vec<&'static str> = Vec::with_capacity(64);
        for _ in 0..NUM_DOCS {
            tokens.clear();

            for spec in terms {
                if rng.random_bool(spec.doc_freq_prob) {
                    let tf = random_term_freq(&mut rng, 0.7);
                    for _ in 0..tf {
                        tokens.push(spec.term);
                    }
                }
            }

            // Pad with filler to create varied field lengths (5-30 tokens).
            let filler_count = rng.random_range(5u32..30u32);
            for _ in 0..filler_count {
                tokens.push("filler");
            }

            let text = tokens.join(" ");
            writer.add_document(doc!(body => text)).unwrap();
        }
        writer.commit().unwrap();
    }

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(&index, vec![body]);

    BenchIndex {
        searcher,
        query_parser,
    }
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

    // Scenarios:
    //
    // Short conjunctions (2-3 terms):
    // - "balanced_10%_10%":            all terms ~10% -> intersection ~1% of docs
    // - "skewed_50%_2%":               one common (50%), one rare (2%) -> intersection ~1%
    // - "very_skewed_80%_0.5%":        one very common (80%), one very rare (0.5%) -> intersection
    //   ~0.4%
    // - "three_balanced_20%_20%_20%":  three terms ~20% each -> intersection ~0.8%
    // - "three_skewed_50%_10%_2%":     50% / 10% / 2% -> intersection ~0.1%
    //
    // Long conjunctions (5 and 10 terms):
    // - "five_balanced_30%":           5 terms ~30% each -> intersection ~0.24% of docs (~2,430
    //   matches)
    // - "five_skewed_1%_50%":          1 rare (1%) + 4 common (50%) -> intersection ~0.06% (~625
    //   matches)
    // - "ten_balanced_50%":            10 terms ~50% each -> intersection ~0.098% (~976 matches)
    // - "ten_skewed_1%_50%":           1 rare (1%) + 9 common (50%) -> intersection ~0.002% (~20
    //   matches)
    // - "ten_cascade_skewed":          10 terms with cascading selectivities (0.5% up to 80%)
    // - "ten_unmatched_20%":           10 terms ~20% each -> ~0 matching docs, stays in
    //   advance_without_pruning
    let mut scenarios = vec![
        Scenario {
            label: "balanced_10%_10%",
            terms: vec![TermSpec::new("aaa", 0.10), TermSpec::new("bbb", 0.10)],
            queries: vec!["+aaa +bbb".to_string()],
        },
        Scenario {
            label: "skewed_50%_2%",
            terms: vec![TermSpec::new("aaa", 0.50), TermSpec::new("bbb", 0.02)],
            queries: vec!["+aaa +bbb".to_string()],
        },
        Scenario {
            label: "very_skewed_80%_0.5%",
            terms: vec![TermSpec::new("aaa", 0.80), TermSpec::new("bbb", 0.005)],
            queries: vec!["+aaa +bbb".to_string()],
        },
        Scenario {
            label: "three_balanced_20%_20%_20%",
            terms: term_spec_uniform(3, 0.20),
            queries: vec!["+aaa +bbb".to_string(), "+aaa +bbb +ccc".to_string()],
        },
        Scenario {
            label: "three_skewed_50%_10%_2%",
            terms: vec![
                TermSpec::new("aaa", 0.50),
                TermSpec::new("bbb", 0.10),
                TermSpec::new("ccc", 0.02),
            ],
            queries: vec!["+aaa +bbb".to_string(), "+aaa +bbb +ccc".to_string()],
        },
        Scenario {
            label: "five_balanced_30%",
            terms: term_spec_uniform(5, 0.30),
            queries: vec![conjunction_query(5)],
        },
        Scenario {
            label: "five_skewed_1%_50%",
            terms: {
                let mut terms = vec![TermSpec::new("aaa", 0.01)];
                terms.extend(TERMS[1..5].iter().map(|&t| TermSpec::new(t, 0.50)));
                terms
            },
            queries: vec![conjunction_query(5)],
        },
        Scenario {
            label: "ten_balanced_50%",
            terms: term_spec_uniform(10, 0.50),
            queries: vec![conjunction_query(10)],
        },
        Scenario {
            label: "ten_skewed_1%_50%",
            terms: {
                let mut terms = vec![TermSpec::new("aaa", 0.01)];
                terms.extend(TERMS[1..10].iter().map(|&t| TermSpec::new(t, 0.50)));
                terms
            },
            queries: vec![conjunction_query(10)],
        },
        Scenario {
            label: "ten_cascade_skewed",
            terms: vec![
                TermSpec::new("aaa", 0.005),
                TermSpec::new("bbb", 0.02),
                TermSpec::new("ccc", 0.05),
                TermSpec::new("ddd", 0.10),
                TermSpec::new("eee", 0.15),
                TermSpec::new("fff", 0.20),
                TermSpec::new("ggg", 0.30),
                TermSpec::new("hhh", 0.40),
                TermSpec::new("iii", 0.60),
                TermSpec::new("jjj", 0.80),
            ],
            queries: vec![conjunction_query(10)],
        },
        Scenario {
            label: "ten_unmatched_20%",
            terms: term_spec_uniform(10, 0.20),
            queries: vec![conjunction_query(10)],
        },
    ];

    if let Ok(filter) = std::env::var("BENCH_SCENARIO") {
        scenarios.retain(|s| s.label.contains(&filter));
        println!(
            "Filtered to {} scenario(s) matching '{filter}'.",
            scenarios.len()
        );
    }

    let mut runner = BenchRunner::new();
    let num_iter = std::env::var("NUM_ITER_GROUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(320);
    runner.config().set_num_iter_for_group(num_iter);

    for scenario in &scenarios {
        let bench_index_baseline = if run_baseline {
            Some(build_index(&scenario.terms, PostingNorms::Disabled))
        } else {
            None
        };
        let bench_index_pnorms = build_index(&scenario.terms, PostingNorms::Enabled);

        for query_str in &scenario.queries {
            let mut group = runner.new_group();
            group.set_name(format!("intersection — {} — {query_str}", scenario.label));

            if let Some(bench_index_baseline) = bench_index_baseline.as_ref() {
                let query = bench_index_baseline
                    .query_parser
                    .parse_query(query_str)
                    .unwrap();
                let searcher = bench_index_baseline.searcher.clone();
                group.register("baseline", move |_| {
                    let collector = TopDocs::with_limit(10).order_by_score();
                    black_box(searcher.search(&query, &collector).unwrap());
                    1usize
                });
            }

            let query = bench_index_pnorms
                .query_parser
                .parse_query(query_str)
                .unwrap();
            let searcher = bench_index_pnorms.searcher.clone();
            group.register("pnorms", move |_| {
                let collector = TopDocs::with_limit(10).order_by_score();
                black_box(searcher.search(&query, &collector).unwrap());
                1usize
            });

            group.run();
        }
    }
}
