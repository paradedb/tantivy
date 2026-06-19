mod bench_utils;
use bench_utils::*;
use binggan::{black_box, BenchRunner};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal, Zipf};
use tantivy::collector::TopDocs;
use tantivy::schema::{Schema, FAST, TEXT};
use tantivy::{doc, Index, ReloadPolicy};
use tantivy::query::QueryParser;

fn main() {
    println!("PID: {}", std::process::id());

    // Using 1M docs for the benchmark to ensure it completes quickly
    // while still being large enough to show Block-WAND benefits.
    let num_docs = 1_000_000;
    let num_segments = 32;

    let vocab_size = 1000;
    let terms: Vec<String> = (1..=vocab_size).map(|i| format!("term_{i}")).collect();

    let queries: &[&str] = &[
        "term_2",                            // dense term
        "term_100",                          // rare term
        "+term_20 +(term_100 OR term_500)",  // intersection
    ];

    let term_pcts: Vec<(&str, String)> = vec![
        ("term_2", "dense".to_string()),
        ("term_20", "medium".to_string()),
        ("term_100", "rare".to_string()),
        ("term_500", "very_rare".to_string()),
    ];

    let lg_norm = LogNormal::new(2.996f64, 0.979f64).unwrap();
    let zipf = Zipf::new(vocab_size as f64, 1.1f64).unwrap();

    let build_realistic = |is_temporal: bool| -> BenchIndex {
        let mut schema_builder = Schema::builder();
        let f_title = schema_builder.add_text_field("title", TEXT);
        let f_body = schema_builder.add_text_field("body", TEXT);
        let f_score = schema_builder.add_u64_field("score", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        
        let mut writer = index.writer_with_num_threads(1, 500_000_000).unwrap();
        let docs_per_segment = num_docs / num_segments;
        
        let mut in_burst = false;
        let mut burst_term_idx = 0;
        let mut burst_remaining = 0;

        for _seg_ord in 0..num_segments {
            for _doc_id in 0..docs_per_segment {
                if is_temporal {
                    if burst_remaining == 0 {
                        // 1 in 10000 chance to start a burst
                        if rng.random_bool(0.0001) {
                            in_burst = true;
                            // Pick a medium/rare term to burst (e.g. rank 20 to 200)
                            burst_term_idx = rng.random_range(20..200);
                            // Burst lasts for 100 to 5000 docs
                            burst_remaining = rng.random_range(100..5000);
                        } else {
                            in_burst = false;
                        }
                    } else {
                        burst_remaining -= 1;
                    }
                }

                let doc_len = (lg_norm.sample(&mut rng) as usize).clamp(1, 5000);
                let mut body_tokens = Vec::with_capacity(doc_len);
                let mut title_tokens = Vec::new();
                
                for _ in 0..doc_len {
                    let mut term_idx = zipf.sample(&mut rng) as usize - 1;
                    
                    if in_burst && rng.random_bool(0.3) {
                        term_idx = burst_term_idx;
                    }
                    
                    if term_idx < vocab_size {
                        let term = terms[term_idx].as_str();
                        if rng.random_bool(0.1) {
                            title_tokens.push(term);
                        } else {
                            body_tokens.push(term);
                        }
                    }
                }
                
                if title_tokens.is_empty() && body_tokens.is_empty() {
                    body_tokens.push("z");
                }
                
                let score = rng.random_range(0u64..100_000u64);
                writer
                    .add_document(doc!(
                        f_title=>title_tokens.join(" "),
                        f_body=>body_tokens.join(" "),
                        f_score=>score,
                    ))
                    .unwrap();
            }
            writer.commit().unwrap();
        }

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .unwrap();
        let searcher = reader.searcher();
        let qp_multi = QueryParser::for_index(&index, vec![f_title, f_body]);

        BenchIndex {
            index,
            searcher,
            query_parser: qp_multi,
        }
    };

    println!("Building zipf_independent index...");
    let zipf_independent = build_realistic(false);
    println!("Building zipf_temporal index...");
    let zipf_temporal = build_realistic(true);

    let mut runner = BenchRunner::new();

    println!("Starting benchmark in 5 seconds.");
    std::thread::sleep(std::time::Duration::from_secs(5));

    for (dist_name, bench_index) in [("zipf_independent", zipf_independent), ("zipf_temporal", zipf_temporal)] {
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
