//! Ordering quality of the routing index candidates (SPANN).
//!
//! The 200k Cohere vectors play the role of posting-list centroids in the
//! broader vector store: given a query, the routing index must order them so
//! the best posting lists are probed first. Each data structure is built
//! directly on the 200k vectors:
//!
//! - BKT: balanced k-means tree, greedy best-first descent (iterator)
//! - BKT+RNG: BKT seeds a beam search over a relative neighborhood graph
//!   built on the vectors (iterator)
//!
//! Per query we compute the exact top-k over the 200k, then report how deep
//! in each structure's output the true top-k sits (`*_depth`: position of
//! the last true top-k member in the emitted order) and how much work was
//! done to get there (`*_visited`).
//!
//! Usage:
//!   cargo bench --bench ivf_index_recall -- [n] [queries]

use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use superkmeans::{
    ClusterTree, HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, SuperKMeansConfig,
    TreeNode,
};
use tantivy::schema::{Metric, VectorOptions};
use tantivy::vector::ivf::{
    BKTree, BKTreeNode, IvfIndex, NeighborhoodGraphConfig, RelativeNeighborhoodGraph, Workspace,
};
use tantivy::vector::Similarity;

const COHERE_N: usize = 1_000_000;
const COHERE_D: usize = 1024;
const DEFAULT_N: usize = 200_000;
const DEFAULT_QUERIES: usize = 200;
/// BKT leaf width: the tree over the 200k has ~n/32 leaves, each holding its
/// assigned vectors as members.
const MAX_LEAF_SIZE: usize = 32;
/// Ground-truth neighborhood sizes on the 200k-vector set.
const EVAL_K: &[usize] = &[10, 100, 200, 400];

fn cohere_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/data_cohere_1m.bin")
}

fn load_cohere(rows: usize) -> Vec<f32> {
    let path = cohere_path();
    let expected_len = COHERE_N * COHERE_D * size_of::<f32>();
    let meta = std::fs::metadata(&path).unwrap_or_else(|_| {
        eprintln!(
            "missing Cohere dump at {}\n\
             download data_cohere_1m.bin (1_000_000 × 1024 f32 LE) into data/ first",
            path.display()
        );
        process::exit(1);
    });
    if meta.len() as usize != expected_len {
        eprintln!(
            "{} is {} bytes, expected {expected_len} (1_000_000 × 1024 f32)",
            path.display(),
            meta.len()
        );
        process::exit(1);
    }
    assert!(
        rows <= COHERE_N,
        "requested {rows} rows, Cohere dump has only {COHERE_N}"
    );

    let mut file = File::open(&path).unwrap_or_else(|err| {
        eprintln!("failed to open {}: {err}", path.display());
        process::exit(1);
    });
    let mut floats = vec![0.0f32; rows * COHERE_D];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            floats.as_mut_ptr().cast::<u8>(),
            rows * COHERE_D * size_of::<f32>(),
        )
    };
    if let Err(err) = file.read_exact(bytes) {
        eprintln!("failed to read {}: {err}", path.display());
        process::exit(1);
    }
    floats
}

/// BKT over the vectors: tree topology + per-node centers from the kmeans
/// tree, leaf members = the vector ids assigned to that leaf. Greedy descent
/// then emits vector ids nearest-leaf first.
///
/// `assignments[i]` maps vector `i` to a leaf ordinal — leaf ordinals count
/// leaves in node order, matching the centroid rows `train` returned.
fn bkt_from_cluster_tree(
    tree: &ClusterTree,
    assignments: &[u32],
    metric: Metric,
) -> BKTree<Vec<f32>> {
    assert_eq!(tree.root.0, 0, "BKTree search assumes the root is node 0");
    let dim = tree.dimensionality();
    let mut leaf_ord = vec![u32::MAX; tree.nodes.len()];
    let mut next = 0u32;
    for (i, node) in tree.nodes.iter().enumerate() {
        if node.is_leaf() {
            leaf_ord[i] = next;
            next += 1;
        }
    }
    assert_eq!(next as usize, tree.n_leaves);

    let mut leaf_members = vec![Vec::new(); tree.n_leaves];
    for (vid, &leaf) in assignments.iter().enumerate() {
        leaf_members[leaf as usize].push(vid as u32);
    }

    let mut members = Vec::with_capacity(assignments.len());
    let mut nodes = Vec::with_capacity(tree.nodes.len());
    for (i, node) in tree.nodes.iter().enumerate() {
        match node {
            TreeNode::Internal {
                centroid_offset,
                children_offset,
                children_size,
                ..
            } => nodes.push(BKTreeNode::Internal {
                centroid_id: *centroid_offset as u32,
                children_offset: *children_offset as u32,
                children_size: *children_size as u32,
            }),
            TreeNode::Leaf {
                centroid_offset, ..
            } => {
                let assigned = &leaf_members[leaf_ord[i] as usize];
                let members_offset = members.len() as u32;
                members.extend_from_slice(assigned);
                nodes.push(BKTreeNode::Leaf {
                    centroid_id: *centroid_offset as u32,
                    members_offset,
                    members_size: assigned.len() as u32,
                });
            }
        }
    }
    assert_eq!(members.len(), assignments.len());
    BKTree {
        dim,
        metric,
        nodes,
        members,
        centers: tree.centroids.clone(),
    }
}

/// Exact nearest `k` vector ids for `query` (higher [`Metric::similarity`]
/// first) — the ground truth every structure's ordering approximates.
fn exact_knn(data: &[f32], dim: usize, query: &[f32], k: usize, metric: Metric) -> Vec<u32> {
    let n = data.len() / dim;
    let take = k.min(n);
    let mut scored: Vec<(Similarity, u32)> = (0..n)
        .map(|i| {
            (
                metric.similarity(query, &data[i * dim..(i + 1) * dim]),
                i as u32,
            )
        })
        .collect();
    if take < n {
        scored.select_nth_unstable_by(take - 1, |a, b| b.0.cmp(&a.0));
        scored.truncate(take);
    }
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, id)| id).collect()
}

#[derive(Clone, Copy)]
struct Cover {
    /// Position (1-based) of the last true top-k member in the output order.
    depth: usize,
    /// Work done when that position was reached: routing nodes scored.
    visited: usize,
}

fn truth_sets(truth: &[u32], ks: &[usize]) -> Vec<HashSet<u32>> {
    ks.iter()
        .map(|&k| truth.iter().copied().take(k.min(truth.len())).collect())
        .collect()
}

/// Pull `rank_clusters` (vector ids, best-first per the structure) until every
/// truth set is covered or the iterator / `max_pulls` runs out.
fn cover_ranking(
    index: &IvfIndex,
    query: &[f32],
    truth: &[u32],
    ks: &[usize],
    max_pulls: usize,
) -> Vec<Option<Cover>> {
    let sets = truth_sets(truth, ks);
    let mut seen = vec![0usize; ks.len()];
    let mut out = vec![None; ks.len()];

    let mut ws = Workspace::new();
    let mut ranking = index.rank_clusters(&mut ws, query);
    let mut pulled = 0usize;
    while pulled < max_pulls {
        let Some(candidate) = ranking.next() else {
            break;
        };
        pulled += 1;
        let mut done = true;
        for (j, set) in sets.iter().enumerate() {
            if out[j].is_none() {
                if set.contains(&candidate.node) {
                    seen[j] += 1;
                    if seen[j] == set.len() {
                        out[j] = Some(Cover {
                            depth: pulled,
                            visited: ranking.metrics().visited_count,
                        });
                        continue;
                    }
                }
                done = false;
            }
        }
        if done {
            break;
        }
    }
    out
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let positional: Vec<&str> = args
        .iter()
        .skip(1)
        .map(String::as_str)
        .filter(|a| !a.starts_with('-') && *a != "--bench")
        .collect();
    let n: usize = positional
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N);
    let n_queries: usize = positional
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_QUERIES);

    let all = load_cohere(n + n_queries);
    let d = COHERE_D;
    let (base, queries) = all.split_at(n * d);
    println!("dataset: Cohere 1M  centroids={n} holdout={n_queries} d={d}");

    let gt_k = (*EVAL_K.iter().max().unwrap()).min(n);
    let gt_start = Instant::now();
    let truth: Vec<Vec<u32>> = (0..n_queries)
        .map(|q| {
            let query = &queries[q * d..(q + 1) * d];
            exact_knn(base, d, query, gt_k, Metric::L2)
        })
        .collect();
    println!(
        "exact holdout top-{gt_k} in {:.2}s",
        gt_start.elapsed().as_secs_f64()
    );

    // BKT: balanced k-means tree over the 200k vectors.
    let base_cfg = SuperKMeansConfig {
        data_already_rotated: true,
        ..Default::default()
    };
    let cfg = HierarchicalSuperKMeansConfig {
        base: base_cfg,
        max_leaf_size: MAX_LEAF_SIZE,
        ..Default::default()
    };
    let mut kmeans = HierarchicalSuperKMeans::with_config(d, cfg);
    let bkt_start = Instant::now();
    let leaf_centroids = kmeans.train(base, n);
    let leaf_assign = kmeans.assign(base, &leaf_centroids, n);
    let bkt = bkt_from_cluster_tree(&kmeans.tree, &leaf_assign, Metric::L2);
    println!(
        "BKT: {} leaves ({} tree nodes, max_leaf_size={MAX_LEAF_SIZE}) in {:.2}s",
        kmeans.tree.n_leaves,
        kmeans.tree.nodes.len(),
        bkt_start.elapsed().as_secs_f64()
    );

    // RNG over the same 200k vectors; BKT seeds its beam search.
    let rng_start = Instant::now();
    let rng = RelativeNeighborhoodGraph::build_from_centroids(
        base.to_vec(),
        d,
        Metric::L2,
        NeighborhoodGraphConfig::default(),
    )
    .expect("rng");
    println!(
        "RNG: {n} nodes in {:.2}s",
        rng_start.elapsed().as_secs_f64()
    );

    let options = VectorOptions::new(d, Metric::L2);
    let exact = IvfIndex::from_centroids(&options, base, None, None).unwrap();
    let bkt_only = IvfIndex::from_centroids(&options, base, Some(&bkt), None).unwrap();
    let bkt_rng = IvfIndex::from_centroids(&options, base, Some(&bkt), Some(&rng)).unwrap();

    println!("task: position of the true top-k in each structure's output order");

    println!(
        "\n{:<6} {:<12} {:<12} {:<12} {:<14} {:<14} {:<10} {:<10}",
        "k",
        "exact_depth",
        "bkt_depth",
        "rng_depth",
        "bkt_visited",
        "rng_visited",
        "bkt_ok",
        "rng_ok"
    );

    let ks: Vec<usize> = EVAL_K.iter().copied().filter(|&k| k <= n).collect();
    let mut exact_depths = vec![Vec::new(); ks.len()];
    let mut bkt_depths = vec![Vec::new(); ks.len()];
    let mut rng_depths = vec![Vec::new(); ks.len()];
    let mut bkt_visited = vec![Vec::new(); ks.len()];
    let mut rng_visited = vec![Vec::new(); ks.len()];
    let mut bkt_ok = vec![0usize; ks.len()];
    let mut rng_ok = vec![0usize; ks.len()];

    for q in 0..n_queries {
        let query = &queries[q * d..(q + 1) * d];
        let exact_c = cover_ranking(&exact, query, &truth[q], &ks, n);
        let bkt_c = cover_ranking(&bkt_only, query, &truth[q], &ks, n);
        let rng_c = cover_ranking(&bkt_rng, query, &truth[q], &ks, n);
        for i in 0..ks.len() {
            if let Some(c) = &exact_c[i] {
                exact_depths[i].push(c.depth as f64);
            }
            if let Some(c) = &bkt_c[i] {
                bkt_ok[i] += 1;
                bkt_depths[i].push(c.depth as f64);
                bkt_visited[i].push(c.visited as f64);
            }
            if let Some(c) = &rng_c[i] {
                rng_ok[i] += 1;
                rng_depths[i].push(c.depth as f64);
                rng_visited[i].push(c.visited as f64);
            }
        }
    }

    let nq = n_queries as f64;
    for (i, &k) in ks.iter().enumerate() {
        println!(
            "{k:<6} {:<12.1} {:<12.1} {:<12.1} {:<14.1} {:<14.1} {:<10.2} {:<10.2}",
            mean(&exact_depths[i]),
            mean(&bkt_depths[i]),
            mean(&rng_depths[i]),
            mean(&bkt_visited[i]),
            mean(&rng_visited[i]),
            bkt_ok[i] as f64 / nq,
            rng_ok[i] as f64 / nq,
        );
    }
}
