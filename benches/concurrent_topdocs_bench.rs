mod bench_utils;
use bench_utils::*;
use binggan::{black_box, BenchRunner};
use rand::Rng;
use tantivy::collector::TopDocs;

fn main() {
    println!("PID: {}", std::process::id());

    let num_docs = 50_000_000;
    let num_segments = 32;
    let terms: &[(&str, f32)] = &[
        ("a", 0.0001),
        ("b", 0.01),
        ("c", 0.05),
        ("d", 0.15),
        ("e", 0.30),
    ];

    let queries: &[&str] = &[
        "e",            // dense term
        "b",            // medium term
        "+c +(b OR d)", // intersection
    ];

    // Uniform Index: High BM25 scores are spread across all segments
    let (_uniform_single, uniform_multi) =
        build_index(num_docs, num_segments, terms, |_, _, rng| {
            if rng.random_bool(0.1) {
                (20, 0) // high TF, low doc length -> High BM25 score
            } else {
                (1, 20) // low TF, high doc length -> Low BM25 score
            }
        });

    // Bursty Index: High BM25 scores are distributed across ALL segments, but 
    // are spatially clustered at the beginning of each segment to test Block-WAND.
    let (_bursty_single, bursty_multi) =
        build_index(num_docs, num_segments, terms, |_, doc_id, rng| {
            let docs_per_segment = (num_docs / num_segments) as u32;
            if doc_id < docs_per_segment / 10 {
                (20, 0)
            } else {
                (1, 20)
            }
        });

    let mut runner = BenchRunner::new();

    let term_pcts: Vec<(&str, String)> = terms
        .iter()
        .map(|&(term, p)| (term, format_pct(p)))
        .collect();

    println!("Starting benchmark in 15 seconds.");
    std::thread::sleep(std::time::Duration::from_secs(15));

    for (dist_name, bench_index) in [("uniform", uniform_multi), ("bursty", bursty_multi)] {
        let mut index_multi_thread = bench_index.index.clone();
        index_multi_thread.set_multithread_executor(4).unwrap();
        let searcher_4t = index_multi_thread.reader().unwrap().searcher();
        let searcher_1t = bench_index.searcher.clone();

        for query_str in queries {
            let mut group = runner.new_group();
            let query_label = query_label(query_str, &term_pcts);
            group.set_name(format!("topdocs_score_{dist_name}_{query_label}"));

            let query = bench_index.query_parser.parse_query(query_str).unwrap();

            // 1 thread baseline
            let s1 = searcher_1t.clone();
            let q1 = query.box_clone();
            group.register("1_thread", move |_| {
                black_box(
                    s1.search(
                        &q1,
                        &TopDocs::with_limit(10).order_by_score(),
                    )
                    .unwrap(),
                )
            });

            // 4 threads (non-deterministic pruning)
            let s4 = searcher_4t.clone();
            let q4 = query.box_clone();
            group.register("4_threads", move |_| {
                black_box(
                    s4.search(
                        &q4,
                        &TopDocs::with_limit(10).order_by_score(),
                    )
                    .unwrap(),
                )
            });

            group.run();
        }
    }
}
