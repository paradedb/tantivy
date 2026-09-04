//! Stacked-IVF recall harness: build an owned index and report recall@k
//! against exact ground truth as a function of `nprobe`.
//!
//! Usage:
//!   [DATASET=cohere|sift] cargo bench --bench ivf_recall -- [n] [queries]
//!
//! `cohere` (default): `data/data_cohere_1m.bin` (or `$COHERE_PATH`),
//! Cohere Embed-V3, 1M × 1024 little-endian f32s. Download with
//! superkmeans-rs `scripts/download_cohere.py`. Queries are held-out
//! rows from the tail of the requested prefix.
//!
//! `sift`: `data/sift/` (or `$SIFT_DIR`) holding texmex `sift_base.fvecs`
//! and `sift_query.fvecs` (1M × 128). Queries come from the query file.
//! Download: `curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`.
//!
//! Falls back to synthetic blobs if the dataset is missing.

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use tantivy::schema::Metric;
use tantivy::vector::{l2_squared, IvfConfig, IvfIndexBuilder, SuperKMeansLevelClusterer};

const COHERE_N: usize = 1_000_000;
const COHERE_D: usize = 1024;
const TOP_K: usize = 10;
const BRANCHING_FACTOR: usize = 200;
const MAX_LEAF_SIZE: usize = 100;
/// Quake's `f_M`: initial candidate fraction at L0.
const APS_FRACTION: f32 = 0.02;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}

fn cohere_path() -> PathBuf {
    env::var_os("COHERE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir().join("data_cohere_1m.bin"))
}

fn sift_dir() -> PathBuf {
    env::var_os("SIFT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir().join("sift"))
}

/// Base rows, query rows, and dimension.
struct Dataset {
    name: String,
    base: Vec<f32>,
    queries: Vec<f32>,
    d: usize,
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

/// Read up to `rows` vectors from a texmex `.fvecs` file: each row is a
/// little-endian `i32` dimension followed by that many `f32`s.
fn load_fvecs(path: &PathBuf, rows: usize) -> Option<(Vec<f32>, usize)> {
    let mut file = File::open(path).ok()?;
    let mut header = [0u8; 4];
    file.read_exact(&mut header).ok()?;
    let d = i32::from_le_bytes(header) as usize;
    assert!(d > 0, "fvecs dim must be positive");
    let meta = std::fs::metadata(path).ok()?;
    let row_bytes = (d + 1) * size_of::<f32>();
    let available = meta.len() as usize / row_bytes;
    let rows = rows.min(available);

    let mut file = File::open(path).ok()?;
    let mut raw = vec![0u8; rows * row_bytes];
    file.read_exact(&mut raw).ok()?;
    let mut floats = Vec::with_capacity(rows * d);
    for row in raw.chunks_exact(row_bytes) {
        let dim = i32::from_le_bytes(row[..4].try_into().unwrap()) as usize;
        assert_eq!(dim, d, "fvecs rows must share one dimension");
        floats.extend(
            row[4..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
        );
    }
    Some((floats, d))
}

fn load_sift(n: usize, n_queries: usize) -> Option<Dataset> {
    let dir = sift_dir();
    let (base, d) = load_fvecs(&dir.join("sift_base.fvecs"), n)?;
    let (queries, dq) = load_fvecs(&dir.join("sift_query.fvecs"), n_queries)?;
    assert_eq!(d, dq, "SIFT base and query dims differ");
    Some(Dataset {
        name: format!(
            "SIFT1M (first {} base rows) from {}",
            base.len() / d,
            dir.display()
        ),
        base,
        queries,
        d,
    })
}

fn load_dataset(n: usize, n_queries: usize) -> Dataset {
    let which = env::var("DATASET").unwrap_or_else(|_| "cohere".to_string());
    match which.as_str() {
        "sift" => {
            if let Some(ds) = load_sift(n, n_queries) {
                return ds;
            }
            println!(
                "SIFT not found under {} (set SIFT_DIR). Download:\n  \
                 curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz",
                sift_dir().display()
            );
        }
        "cohere" => {
            if let Some((all, d)) = load_cohere(n + n_queries) {
                let name = format!(
                    "Cohere 1M (first {} rows) from {}",
                    n + n_queries,
                    cohere_path().display()
                );
                return split_tail(name, all, d, n_queries);
            }
            println!(
                "Cohere dump not found. Place 1M×1024 f32s at {} or set COHERE_PATH.\n\
                 Download: uv run --script scripts/download_cohere.py (paradedb/superkmeans-rs)",
                cohere_path().display()
            );
        }
        other => panic!("unknown DATASET={other}; expected cohere or sift"),
    }
    let d = 64;
    let n_syn = n.min(4_000);
    let n_queries = n_queries.min(50);
    let all = make_blobs(n_syn + n_queries, d, 32, 42);
    split_tail("synthetic blobs".to_string(), all, d, n_queries)
}

/// Hold out the last `n_queries` rows of `all` as queries.
fn split_tail(name: String, mut all: Vec<f32>, d: usize, n_queries: usize) -> Dataset {
    let total = all.len() / d;
    let n_queries = n_queries.min(total.saturating_sub(1));
    let queries = all.split_off((total - n_queries) * d);
    Dataset {
        name,
        base: all,
        queries,
        d,
    }
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

/// Exact top-`k` neighbors of each query, by brute force (L2), sorted
/// nearest-first so any prefix is the exact top-`k'` for `k' <= k`.
fn ground_truth(
    base: &[f32],
    n: usize,
    queries: &[f32],
    n_queries: usize,
    d: usize,
    k: usize,
) -> Vec<Vec<u32>> {
    (0..n_queries)
        .into_par_iter()
        .map(|q| {
            let query = &queries[q * d..(q + 1) * d];
            let mut scored: Vec<(f32, u32)> = (0..n)
                .map(|i| (l2_squared(query, &base[i * d..(i + 1) * d]), i as u32))
                .collect();
            scored.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
            scored.truncate(k);
            scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
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

/// Mean (recall@k, similarity computations, lists scanned, ms/query) over
/// all queries. Counts cover every level of the stack, so `%_sims` is total
/// query cost relative to brute force, not just L0 rows touched.
fn evaluate(
    index: &tantivy::vector::InMemoryStackedIvf,
    queries: &[f32],
    n_queries: usize,
    d: usize,
    recall_target: f32,
    truth: &[Vec<u32>],
    inv: &[u32],
) -> (f64, f64, f64, f64) {
    let t0 = Instant::now();
    let (recall_sum, cand_sum, list_sum) = (0..n_queries)
        .into_par_iter()
        .map(|q| {
            let query = &queries[q * d..(q + 1) * d];
            let (hits, stats) = index.search(query, TOP_K, recall_target, Metric::L2);
            let found = hits
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
    let nq = n_queries as f64;
    (
        recall_sum / nq,
        cand_sum / nq,
        list_sum / nq,
        1e3 * elapsed / nq,
    )
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

    let dataset = load_dataset(n, n_queries);
    println!("dataset: {}", dataset.name);
    let d = dataset.d;
    let base = dataset.base.as_slice();
    let queries = dataset.queries.as_slice();
    let n = base.len() / d;
    let n_queries = queries.len() / d;

    println!(
        "n={n} d={d} queries={n_queries} top_k={TOP_K} \
         branching={BRANCHING_FACTOR} max_leaf_size={MAX_LEAF_SIZE}"
    );

    print!("computing exact ground truth... ");
    let t0 = Instant::now();
    let truth = ground_truth(base, n, queries, n_queries, d, TOP_K);
    println!("{:.1}s", t0.elapsed().as_secs_f64());

    let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
    let config = IvfConfig {
        branching_factor: BRANCHING_FACTOR,
        max_leaf_size: MAX_LEAF_SIZE,
        ..Default::default()
    };
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

    // Fixed-nprobe baseline. With `recall=1.0` the top level ranks every
    // centroid, so routing is exact: this is flat IVF with an oracle nprobe.
    println!(
        "\n{:<8} {:<10} {:<12} {:<10} {:>6}",
        "nprobe",
        "lists",
        "%_sims",
        format!("recall@{TOP_K}"),
        "ms/q"
    );
    for nprobe in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        if nprobe > nlist {
            break;
        }
        index.config.nprobe_fraction = nprobe as f32 / nlist as f32;
        let (recall, cands, lists, ms) = evaluate(&index, queries, n_queries, d, 1.0, &truth, &inv);
        println!(
            "{nprobe:<8} {lists:<10.1} {:<12.2} {recall:<10.4} {ms:>6.2}",
            100.0 * cands / n as f64
        );
    }

    index.config.nprobe_fraction = APS_FRACTION;
    println!(
        "\nAPS: f_M={APS_FRACTION} -> {} candidate lists, parent recall {}\n\
         {:<8} {:<10} {:<12} {:<10} {:>6}",
        index.n_probe(),
        index.config.parent_recall_target,
        "target",
        "lists",
        "%_sims",
        format!("recall@{TOP_K}"),
        "ms/q"
    );
    for target in [0.80f32, 0.90, 0.95, 0.99] {
        let (recall, cands, lists, ms) =
            evaluate(&index, queries, n_queries, d, target, &truth, &inv);
        println!(
            "{target:<8.2} {lists:<10.1} {:<12.2} {recall:<10.4} {ms:>6.2}",
            100.0 * cands / n as f64
        );
    }
}
