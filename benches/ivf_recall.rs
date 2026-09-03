//! Stacked-IVF recall harness: build an owned index and report recall@k
//! against exact ground truth as a function of `nprobe`.
//!
//! Usage:
//!   cargo bench --bench ivf_recall -- [n] [queries] [train_fraction]
//!
//! Loads `data/data_cohere_1m.bin` (or `$COHERE_PATH`): Cohere Embed-V3,
//! 1M × 1024 little-endian f32s. Download with superkmeans-rs
//! `scripts/download_cohere.py`. Falls back to synthetic blobs if the
//! dump is missing.

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use tantivy::schema::Metric;
use tantivy::vector::{IvfConfig, IvfIndexBuilder, SuperKMeansLevelClusterer, l2_squared};

const COHERE_N: usize = 1_000_000;
const COHERE_D: usize = 1024;
const TOP_K: usize = 10;
const BRANCHING_FACTOR: usize = 16;

fn cohere_path() -> PathBuf {
    env::var_os("COHERE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/data_cohere_1m.bin")
        })
}

/// Read the first `rows` vectors of the Cohere dump, or `None` if it is absent.
fn load_cohere(rows: usize) -> Option<(Vec<f32>, usize)> {
    let path = cohere_path();
    let meta = std::fs::metadata(&path).ok()?;
    if meta.len() as usize != COHERE_N * COHERE_D * size_of::<f32>() {
        eprintln!(
            "Cohere dump at {} has {} bytes, expected {}",
            path.display(),
            meta.len(),
            COHERE_N * COHERE_D * size_of::<f32>()
        );
        return None;
    }
    assert!(rows <= COHERE_N, "dataset has only {COHERE_N} rows");

    let mut file = File::open(&path).ok()?;
    let mut floats = vec![0.0f32; rows * COHERE_D];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            floats.as_mut_ptr().cast::<u8>(),
            rows * COHERE_D * size_of::<f32>(),
        )
    };
    file.read_exact(bytes).ok()?;
    Some((floats, COHERE_D))
}

fn make_blobs(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f32 / (1u64 << 53) as f32
    };
    let mut centers = vec![0.0f32; n_clusters * d];
    for c in &mut centers {
        *c = next() * 10.0;
    }
    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let cluster = i % n_clusters;
        for j in 0..d {
            let noise = (next() - 0.5) * 0.4;
            data[i * d + j] = centers[cluster * d + j] + noise;
        }
    }
    data
}

/// Exact top-`TOP_K` neighbors of each query, by brute force (L2).
fn ground_truth(
    base: &[f32],
    n: usize,
    queries: &[f32],
    n_queries: usize,
    d: usize,
) -> Vec<Vec<u32>> {
    (0..n_queries)
        .into_par_iter()
        .map(|q| {
            let query = &queries[q * d..(q + 1) * d];
            let mut scored: Vec<(f32, u32)> = (0..n)
                .map(|i| (l2_squared(query, &base[i * d..(i + 1) * d]), i as u32))
                .collect();
            scored.select_nth_unstable_by(TOP_K - 1, |a, b| a.0.total_cmp(&b.0));
            scored.truncate(TOP_K);
            scored.into_iter().map(|(_, i)| i).collect()
        })
        .collect()
}

fn invert_perm(perm: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; perm.len()];
    for (old, &new) in perm.iter().enumerate() {
        inv[new as usize] = old as u32;
    }
    inv
}

fn main() {
    let args: Vec<String> = env::args().collect();
    // `cargo bench` inserts the bench binary name at argv[1]; extra args
    // follow `--`. Accept either `ivf_recall [n] …` or `[n] …`.
    let positional: Vec<&str> = args
        .iter()
        .skip(1)
        .map(String::as_str)
        .filter(|a| !a.contains("ivf_recall") && *a != "--")
        .collect();
    let n: usize = positional
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let n_queries: usize = positional
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let train_fraction: f64 = positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    let (all, d) = match load_cohere(n + n_queries) {
        Some(v) => {
            println!(
                "dataset: Cohere 1M (first {} rows) from {}",
                n + n_queries,
                cohere_path().display()
            );
            v
        }
        None => {
            println!(
                "dataset: synthetic blobs (Cohere dump not found).\n\
                 Place 1M×1024 f32s at {} or set COHERE_PATH.\n\
                 Download: uv run --script scripts/download_cohere.py\n\
                 (from paradedb/superkmeans-rs)",
                cohere_path().display()
            );
            let d = 64;
            let n_syn = n.min(4_000);
            (make_blobs(n_syn + n_queries.min(50), d, 32, 42), d)
        }
    };
    let n = (all.len() / d).saturating_sub(n_queries).max(1);
    let n_queries = n_queries.min(all.len() / d - n);
    let (base, queries) = all.split_at(n * d);

    if (train_fraction - 1.0).abs() > f64::EPSILON {
        println!(
            "note: train_fraction={train_fraction} is ignored; \
             stacked IVF trains on all {n} members"
        );
    }

    println!("n={n} d={d} queries={n_queries} top_k={TOP_K} branching={BRANCHING_FACTOR}");

    print!("computing exact ground truth... ");
    let t0 = Instant::now();
    let truth = ground_truth(base, n, queries, n_queries, d);
    println!("{:.1}s", t0.elapsed().as_secs_f64());

    let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
    let config = IvfConfig::new(BRANCHING_FACTOR);
    println!("building stacked IVF...");
    let t0 = Instant::now();
    let (mut index, perm) = IvfIndexBuilder::new(base.to_vec(), n, d, &clusterer, config).build();
    let build_secs = t0.elapsed().as_secs_f64();
    let inv = invert_perm(&perm);
    println!(
        "built depth={} nlist={} in {:.2}s",
        index.depth(),
        index.nlist(),
        build_secs
    );

    let nlist = index.nlist();
    println!(
        "\n{:<8} {:<10} {:<12} {:<10} {:>6}",
        "nprobe",
        "lists",
        "%_of_base",
        format!("recall@{TOP_K}"),
        "ms/q"
    );
    for nprobe in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        if nprobe > nlist {
            break;
        }
        index.config.nprobe_fraction = nprobe as f32 / nlist as f32;

        let t0 = Instant::now();
        let (recall_sum, cand_sum, list_sum) = (0..n_queries)
            .into_par_iter()
            .map(|q| {
                let query = &queries[q * d..(q + 1) * d];
                let stats = index.search_with_stats(query, TOP_K, 1.0, Metric::L2);
                let found = stats
                    .hits
                    .iter()
                    .filter(|h| truth[q].contains(&inv[usize::from(h.node)]))
                    .count();
                (
                    found as f64 / TOP_K as f64,
                    stats.members_scored as f64,
                    stats.lists_scanned as f64,
                )
            })
            .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let elapsed = t0.elapsed().as_secs_f64();

        let mean_cand = cand_sum / n_queries as f64;
        let mean_lists = list_sum / n_queries as f64;
        println!(
            "{nprobe:<8} {mean_lists:<10.1} {:<12.2} {:<10.4} {:>6.2}",
            100.0 * mean_cand / n as f64,
            recall_sum / n_queries as f64,
            1e3 * elapsed / n_queries as f64,
        );
    }

    index.config.nprobe_fraction = 0.10;
    println!(
        "\n{:<8} {:<10} {:<12} {:<10} {:>6}",
        "target",
        "lists",
        "%_of_base",
        format!("recall@{TOP_K}"),
        "ms/q"
    );
    for target in [0.80f32, 0.90, 0.99] {
        let t0 = Instant::now();
        let (recall_sum, cand_sum, list_sum) = (0..n_queries)
            .into_par_iter()
            .map(|q| {
                let query = &queries[q * d..(q + 1) * d];
                let stats = index.search_with_stats(query, TOP_K, target, Metric::L2);
                let found = stats
                    .hits
                    .iter()
                    .filter(|h| truth[q].contains(&inv[usize::from(h.node)]))
                    .count();
                (
                    found as f64 / TOP_K as f64,
                    stats.members_scored as f64,
                    stats.lists_scanned as f64,
                )
            })
            .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let elapsed = t0.elapsed().as_secs_f64();
        let mean_cand = cand_sum / n_queries as f64;
        let mean_lists = list_sum / n_queries as f64;
        println!(
            "{target:<8.2} {mean_lists:<10.1} {:<12.2} {:<10.4} {:>6.2}",
            100.0 * mean_cand / n as f64,
            recall_sum / n_queries as f64,
            1e3 * elapsed / n_queries as f64,
        );
    }
}
