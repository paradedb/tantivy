//! Router quality audit: how many clusters each ranking of a segment's clusters needs to cover a
//! query's true neighbours, against an exact flat ranking of the same clusters, and what routing
//! costs. Diagnostics only; nothing here feeds a build or a query.
//!
//! Coverage recall treats probed clusters as scanned exactly: `cov(q, m)` is the share of the
//! query's ground-truth documents inside the first `m` clusters of a ranking. A router's ranking
//! at grid point `k` is its whole output for `router_k = k`, probed in full. Every router here is
//! opened from serialized bytes and walked by the production stacked walk, so reads are charged
//! as a query charges them.
//!
//! The grid only draws curves: `m_r` is the exact smallest request reaching a target. The order a
//! clusterer sees its rows in moves `m_r` by far more than a comparison between routers can
//! resolve, so each rebuilt structure is built [`REBUILDS`] times over shuffled rows and judged by
//! its median and range.

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde::Serialize;

use super::candidate::Candidate;
use super::search::{request, OutputPrecision, SearchPolicy, TopCandidates};
use super::source::VectorSource as _;
use super::stacked::levels::{
    InMemoryStore, LazyStore, LevelClusterer, StackedIvf, StackedSearchStats, LIST_OVERSCAN,
    PARENT_NPROBE_FRACTION,
};
use super::stacked::search::{RoutingOutput, StackedRouting};
use super::stacked::{router_config, StackedRouter};
use super::{exact, BuiltRouter, Router, RouterKind, RoutingParams};
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::clustering::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingBatch, IvfTrainingVectors,
    SuperKMeansClusterer, VectorSource,
};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::telemetry::{enter_vector_stage, measure, StageTotals};
use crate::vector::write::ivf::training_sample;
use crate::vector::VectorStage;
use crate::{DocId, TantivyError};

/// Recall targets every ranking is solved for.
pub const TARGETS: [f64; 3] = [0.90, 0.95, 0.99];

/// A rule's median `m_r` matches the flat ranking within this ratio of its probe count.
const GATE_RATIO: f64 = 1.02;

/// Builds of each rebuilt structure. Each shuffles the rows its clusterer sees, seeded by build.
pub const REBUILDS: u64 = 5;

/// Requests walked either side of a bisected `m_r`, since recall need not rise with the request.
const MONOTONE_SCAN: usize = 8;

/// Layer-0 bytes per posting row of a `[1,1]` cluster at d = 1024: the unit router work is priced
/// in.
const POSTING_ROW_BYTES: f64 = 140.0;

/// Rerank factors of the quantized rankings. Factor 1 is not a mode the audit measures.
const RERANK_FACTORS: [f32; 2] = [2.0, 4.0];

/// Router audit inputs read from one IVF segment.
pub struct RouterAuditExport {
    /// Vector dimension.
    pub dim: usize,
    /// The field's metric.
    pub metric: Metric,
    /// Centroid rows of slot 0, `f32` little-endian, in stored (router) order.
    pub centroid_rows: common::OwnedBytes,
    /// The router slot, its kind byte included.
    pub router_slot: common::OwnedBytes,
    /// Rows per cluster block.
    pub cluster_sizes: Vec<u32>,
    /// The primary cluster of each document with a vector.
    pub doc_clusters: Vec<(DocId, u32)>,
}

/// One audit run's inputs, all in memory.
pub struct AuditInput {
    /// Vector dimension.
    pub dim: usize,
    /// The field's metric.
    pub metric: Metric,
    /// Centroid rows, `dim` values each, in stored (router) order.
    pub centroids: Vec<f32>,
    /// The stored router slot, its kind byte included.
    pub router_slot: Vec<u8>,
    /// Rows per cluster.
    pub sizes: Vec<u32>,
    /// Query vectors.
    pub queries: Vec<Vec<f32>>,
    /// The cluster holding each ground-truth document of each query.
    pub gt_clusters: Vec<Vec<u32>>,
    /// The set each query is reported under.
    pub query_set: Vec<String>,
    /// Base grid of router requests; see [`default_ks`]. Each set adds its flat `m_r`s.
    pub ks: Vec<usize>,
    /// The index's clusterer, for the posting-shaped structure.
    pub clusterer: SuperKMeansClusterer,
}

/// Which phases to run.
#[derive(Clone, Copy, Debug)]
pub struct Phases {
    /// Flat baseline, today's routers, attribution and work.
    pub a: bool,
    /// Structures and the list-open rule sweep.
    pub b: bool,
}

/// One grid point of a recall curve.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct CurvePoint {
    /// Router request.
    pub k: usize,
    /// Mean coverage recall.
    pub recall: f64,
}

/// Percentiles of a per-query count, nearest rank.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Percentiles {
    /// Median.
    pub p50: u32,
    /// 95th percentile.
    pub p95: u32,
    /// 99th percentile.
    pub p99: u32,
}

/// The flat ranking at one recall target.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct FlatTarget {
    /// The target.
    pub target: f64,
    /// Smallest fixed prefix with mean recall at least the target.
    pub m_r: usize,
    /// Recall at `m_r`.
    pub recall: f64,
    /// Mean rows in the first `m_r` clusters.
    pub rows_r: f64,
}

/// The flat ranking over one query set.
#[derive(Clone, Debug, Serialize)]
pub struct FlatReport {
    /// Queries in the set.
    pub queries: usize,
    /// One row per target.
    pub targets: Vec<FlatTarget>,
    /// Clusters a query needs for nine of its ground-truth documents.
    pub need_9: Percentiles,
    /// Clusters a query needs for all ten.
    pub need_10: Percentiles,
}

/// Mean work per query of a ranking at one request.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Work {
    /// Lists opened per level, bottom first.
    pub lists_per_level: Vec<f64>,
    /// Members scored per level, bottom first, with the top's centroid scan last.
    pub scored_per_level: Vec<f64>,
    /// Vectors scored in all.
    pub scored: f64,
    /// Scored from one-bit codes.
    pub one_bit_scored: f64,
    /// Scored at full precision.
    pub exact_scored: f64,
    /// `one_bit_scored` and `exact_scored` priced by their row sizes.
    pub bytes: f64,
    /// Storage reads charged to routing and its exact rerank.
    pub reads: f64,
    /// Bytes charged to routing and its exact rerank.
    pub bytes_read: f64,
    /// Rows in the clusters the ranking probes.
    pub rows_r: f64,
    /// `bytes` in mean posting clusters.
    pub units_r: f64,
    /// Every vector scored at the layer-0 row size, in mean posting clusters.
    pub units_at_layer0: f64,
}

/// Where the ground-truth documents a flat prefix covers and a router's misses went.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Attribution {
    /// The shared prefix: `m_r` of the flat ranking.
    pub m: usize,
    /// Flat recall at `m`.
    pub flat_recall: f64,
    /// Router recall at request `m`.
    pub router_recall: f64,
    /// Documents the flat prefix covers and the router's misses, per query.
    pub missed: f64,
    /// Documents the router covers and the flat prefix misses, per query.
    pub gained: f64,
    /// Share of `missed` whose router list was never opened.
    pub unopened: f64,
    /// Share of `missed` scored one-bit and not retained for the rerank.
    pub dropped: f64,
    /// Share of `missed` kept and ranked past `m`.
    pub late: f64,
}

/// A request whose successor reached less recall.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct NonMonotoneStep {
    /// The request.
    pub k: usize,
    /// Recall at `k`.
    pub recall: f64,
    /// Recall at `k + 1`.
    pub next_recall: f64,
}

/// A ranking at one recall target.
#[derive(Clone, Debug, Serialize)]
pub struct RankingTarget {
    /// The target.
    pub target: f64,
    /// Exact smallest request reaching the target; `None` when the grid's largest does not.
    pub m_r: Option<usize>,
    /// Steps around `m_r` where a larger request lost recall.
    pub non_monotone: Vec<NonMonotoneStep>,
    /// The flat ranking's `m_r`.
    pub flat_m_r: usize,
    /// `m_r` over the flat ranking's.
    pub ratio: Option<f64>,
    /// Recall at `m_r`.
    pub recall: Option<f64>,
    /// Best recall on the grid.
    pub best_recall: f64,
    /// Grid point below `m_r`.
    pub below: Option<CurvePoint>,
    /// Grid point at or above `m_r`.
    pub above: Option<CurvePoint>,
    /// Work at `m_r`.
    pub work: Option<Work>,
    /// Attribution at the flat `m_r`.
    pub attribution: Option<Attribution>,
}

/// One list-size distribution of a stacked level.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct LevelShape {
    /// Lists in the level.
    pub lists: usize,
    /// Members over all lists.
    pub members: usize,
    /// Smallest list.
    pub min: u32,
    /// Median list.
    pub p50: u32,
    /// 99th percentile list.
    pub p99: u32,
    /// Largest list.
    pub max: u32,
}

/// One router under one policy, per query set.
#[derive(Clone, Debug, Serialize)]
pub struct RankingReport {
    /// Router name.
    pub name: String,
    /// Whether the patience stop is on.
    pub aps: bool,
    /// Member beam per requested member; rule sweeps only vary it.
    pub overscan: usize,
    /// List floor.
    pub min_fraction: f32,
    /// Rerank retention factor; 1 for full precision.
    pub rerank_factor: f32,
    /// Grid curves per set.
    pub curves: BTreeMap<String, Vec<CurvePoint>>,
    /// Target rows per set.
    pub targets: BTreeMap<String, Vec<RankingTarget>>,
}

/// One target over a structure's builds.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TargetSpread {
    /// The target.
    pub target: f64,
    /// The flat ranking's `m_r`.
    pub flat_m_r: usize,
    /// Median `m_r`; `None` when most builds never reach the target.
    pub median: Option<usize>,
    /// Smallest `m_r`.
    pub min: Option<usize>,
    /// Largest `m_r`; `None` when a build never reaches the target.
    pub max: Option<usize>,
    /// Builds that never reach the target.
    pub unreached: usize,
    /// Median vectors scored at `m_r`, over the builds that reach the target.
    pub median_scored: Option<f64>,
    /// Whether the range holds builds on both sides of `GATE_RATIO * flat_m_r`.
    pub straddles: bool,
}

/// One structure under one policy, over its builds.
#[derive(Clone, Debug, Serialize)]
pub struct RebuiltReport {
    /// Structure name.
    pub name: String,
    /// Whether the patience stop is on.
    pub aps: bool,
    /// Member beam per requested member.
    pub overscan: usize,
    /// List floor.
    pub min_fraction: f32,
    /// Rerank retention factor; 1 for full precision.
    pub rerank_factor: f32,
    /// Spread per set.
    pub spread: BTreeMap<String, Vec<TargetSpread>>,
    /// Each build's results, by seed.
    pub builds: Vec<RankingReport>,
}

/// The stored quantized router against its rebuilds.
#[derive(Clone, Debug, Serialize)]
pub struct RebuildCheck {
    /// Rerank factor.
    pub rerank_factor: f32,
    /// Patience on or off.
    pub aps: bool,
    /// Query set.
    pub query_set: String,
    /// Recall target.
    pub target: f64,
    /// The stored router's `m_r`.
    pub stored: Option<usize>,
    /// The rebuilds' median `m_r`.
    pub median: Option<usize>,
    /// The rebuilds' smallest `m_r`.
    pub min: Option<usize>,
    /// The rebuilds' largest `m_r`; `None` when a rebuild never reaches the target.
    pub max: Option<usize>,
    /// Whether the stored `m_r` lies inside the rebuilds' range.
    pub within: bool,
}

/// The gate's answer for one structure and query set.
#[derive(Clone, Debug, Serialize)]
pub struct GateResult {
    /// Structure name.
    pub structure: String,
    /// Query set.
    pub query_set: String,
    /// The cheapest passing rule, as an index into [`PhaseB::rules`].
    pub pick: Option<usize>,
    /// Without a pass, the rule with the smallest worst-target median ratio.
    pub closest: Option<usize>,
    /// The worst-target median ratio of `closest`.
    pub closest_ratio: Option<f64>,
    /// Rules whose range straddles the line at some target.
    pub straddling: Vec<usize>,
}

/// One build's shape.
#[derive(Clone, Debug, Serialize)]
pub struct BuildShape {
    /// Levels, bottom first.
    pub levels: Vec<LevelShape>,
    /// Centroids scored in full at the top.
    pub top: usize,
}

/// One structure's shapes, by build.
#[derive(Clone, Debug, Serialize)]
pub struct StructureReport {
    /// Structure name.
    pub name: String,
    /// One per build.
    pub builds: Vec<BuildShape>,
}

/// Phase B output.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PhaseB {
    /// Structures.
    pub structures: Vec<StructureReport>,
    /// Whether rule S(2, 0.25, on) reproduced the full-precision router's output on every build,
    /// query and grid point.
    pub s_reproduces_k1: bool,
    /// Every structure and rule.
    pub rules: Vec<RuleReport>,
    /// Per structure and set.
    pub gates: Vec<GateResult>,
}

/// One structure under one list-open rule.
#[derive(Clone, Debug, Serialize)]
pub struct RuleReport {
    /// Structure name.
    pub structure: String,
    /// The rule and its results.
    pub ranking: RebuiltReport,
}

/// The whole audit.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AuditReport {
    /// Clusters in the segment.
    pub clusters: usize,
    /// Rows over all clusters.
    pub rows: u64,
    /// Mean rows per cluster.
    pub mean_cluster_rows: f64,
    /// The stored router's shape.
    pub stored_router: Vec<LevelShape>,
    /// The flat ranking per set.
    pub flat: BTreeMap<String, FlatReport>,
    /// The stored router's rankings.
    pub rankings: Vec<RankingReport>,
    /// The rebuilt full-precision and quantized routers.
    pub rebuilt: Vec<RebuiltReport>,
    /// The stored quantized router against its rebuilds.
    pub rebuild_check: Vec<RebuildCheck>,
    /// Phase B.
    pub phase_b: Option<PhaseB>,
}

/// The stored router at one online operating point.
pub struct OnlinePoint {
    /// Rerank factor of the point.
    pub rerank_factor: f32,
    /// The `router_k` of each query.
    pub router_k: Vec<usize>,
    /// Clusters each query probed.
    pub clusters: Vec<usize>,
}

/// The geometric request grid: 8 to `min(clusters, 8192)`, ratio at most 2^(1/4), ends included.
pub fn default_ks(clusters: usize) -> Vec<usize> {
    let last = clusters.min(8192);
    let mut ks = Vec::new();
    let mut k = 8usize.min(last.max(1));
    while k < last {
        ks.push(k);
        k = ((k as f64 * 2f64.powf(0.25)).floor() as usize).max(k + 1);
    }
    ks.push(last.max(1));
    ks
}

// --- Walking ---

/// One walk's trail.
struct WalkTrace {
    /// Lists opened per level.
    opened: Vec<Vec<u32>>,
    /// Bottom members retained, before any rerank.
    retained: Vec<u32>,
}

struct Trace<'a>(&'a mut WalkTrace);

impl RoutingOutput for Trace<'_> {
    fn record(&mut self, _candidate: Candidate<(usize, u32)>) {}

    fn finish(self, best: TopCandidates<(usize, u32)>) -> Vec<Candidate<(usize, u32)>> {
        best.into_sorted()
    }

    fn opened(&mut self, level: usize, list: u32) {
        self.0.opened[level].push(list);
    }

    fn retained(&mut self, level: usize, best: &TopCandidates<(usize, u32)>) {
        if level == 0 {
            self.0.retained = best.iter().map(|candidate| candidate.node.1).collect();
        }
    }
}

/// One walk's output, in exported cluster ids, with its trail and cost.
struct Walk {
    output: Vec<u32>,
    trace: WalkTrace,
    stats: StackedSearchStats,
    totals: StageTotals,
}

/// A router opened from bytes, and how its members map to exported clusters.
struct Ranking {
    router: StackedRouter<LazyStore>,
    /// `to_exported[row]`; `None` when the router's rows are the exported clusters.
    to_exported: Option<Vec<u32>>,
    /// `to_row[cluster]`; `None` when the router's rows are the exported clusters.
    to_row: Option<Vec<u32>>,
    /// The bottom list of each member row.
    member_list: Vec<u32>,
}

impl Ranking {
    fn new(
        router: StackedRouter<LazyStore>,
        permutation: Option<Vec<u32>>,
    ) -> crate::Result<Ranking> {
        let rows = router.level.vectors.num_vectors(router.dim);
        let mut member_list = vec![u32::MAX; rows];
        for (list, &(start, end)) in router.level.offsets.iter().enumerate() {
            member_list[start as usize..end as usize].fill(list as u32);
        }
        if member_list.contains(&u32::MAX) {
            return Err(TantivyError::InvalidArgument(
                "router lists do not cover its members".into(),
            ));
        }
        let to_exported = permutation.as_ref().map(|permutation| {
            let mut to_exported = vec![0; permutation.len()];
            for (old, &new) in permutation.iter().enumerate() {
                to_exported[new as usize] = old as u32;
            }
            to_exported
        });
        Ok(Ranking {
            router,
            to_exported,
            to_row: permutation,
            member_list,
        })
    }

    fn opened(
        bytes: Vec<u8>,
        rows: Vec<u8>,
        permutation: Option<Vec<u32>>,
        options: &VectorOptions,
    ) -> crate::Result<Self> {
        match RouterKind::open(FileSlice::from(bytes), FileSlice::from(rows), options)? {
            Router::Stacked(router) => Self::new(router, permutation),
            Router::Exact(_) => Err(TantivyError::InvalidArgument(
                "the router audit needs a stacked router".into(),
            )),
        }
    }

    /// Serializes `built` and opens it again over `rows`, its members in router order.
    fn reopened(
        built: StackedRouter<InMemoryStore>,
        rows: Vec<u8>,
        permutation: Vec<u32>,
        options: &VectorOptions,
    ) -> crate::Result<Self> {
        let mut bytes = Vec::new();
        BuiltRouter::Stacked(built).serialize(&mut bytes)?;
        Self::opened(bytes, rows, Some(permutation), options)
    }

    fn compressed(&self) -> bool {
        self.router.level.vectors.compressed().is_some()
    }

    fn rows(&self) -> usize {
        self.router.level.vectors.num_vectors(self.router.dim)
    }

    fn levels(&self) -> usize {
        self.router.level.levels().count()
    }

    /// `params`' request of this router: its member count and policy.
    fn request(&self, params: RoutingParams) -> (usize, SearchPolicy) {
        request(self.compressed(), self.rows(), params)
    }

    fn exported(&self, row: u32) -> u32 {
        self.to_exported
            .as_ref()
            .map_or(row, |to_exported| to_exported[row as usize])
    }

    fn row(&self, cluster: u32) -> u32 {
        self.to_row
            .as_ref()
            .map_or(cluster, |to_row| to_row[cluster as usize])
    }

    /// The production walk under `policy`, traced.
    fn walk(&self, query: &VectorQuery<f32>, k: usize, policy: SearchPolicy) -> Walk {
        let mut trace = WalkTrace {
            opened: vec![Vec::new(); self.levels()],
            retained: Vec::new(),
        };
        let (ranked, totals) = measure(|| {
            let _routing = enter_vector_stage(VectorStage::Routing);
            StackedRouting::Segment(&self.router.level).search(
                query,
                k,
                policy,
                OutputPrecision::Exact,
                Trace(&mut trace),
            )
        });
        Walk {
            output: ranked
                .candidates
                .iter()
                .map(|candidate| self.exported(candidate.node.1))
                .collect(),
            trace,
            stats: ranked.stats[0],
            totals,
        }
    }

    /// The production entry point's output in exported cluster ids.
    fn ranked(&self, query: &VectorQuery<f32>, params: RoutingParams) -> Vec<u32> {
        self.router
            .rank(query, params)
            .map(|candidate| self.exported(candidate.node))
            .collect()
    }

    fn shape(&self) -> Vec<LevelShape> {
        self.router
            .level
            .levels()
            .map(|level| {
                let mut sizes: Vec<u32> = level
                    .offsets
                    .iter()
                    .map(|&(start, end)| (end - start) as u32)
                    .collect();
                sizes.sort_unstable();
                LevelShape {
                    lists: sizes.len(),
                    members: sizes.iter().map(|&size| size as usize).sum(),
                    min: sizes.first().copied().unwrap_or(0),
                    p50: nearest_rank(&sizes, 0.50),
                    p99: nearest_rank(&sizes, 0.99),
                    max: sizes.last().copied().unwrap_or(0),
                }
            })
            .collect()
    }

    fn top(&self) -> usize {
        self.router
            .level
            .levels()
            .last()
            .map_or(0, |level| level.nlist())
    }
}

// --- Harness ---

/// `f` over `0..n` on every core, results in order.
fn parallel<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |threads| threads.get())
        .min(n.max(1));
    let next = AtomicUsize::new(0);
    let mut pieces: Vec<Vec<(usize, T)>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut piece = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            return piece;
                        }
                        piece.push((i, f(i)));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("audit worker panicked"))
            .collect()
    });
    let mut all: Vec<(usize, T)> = pieces.drain(..).flatten().collect();
    all.sort_unstable_by_key(|&(i, _)| i);
    all.into_iter().map(|(_, value)| value).collect()
}

/// The value at nearest rank `q` of `sorted`, or 0 when empty.
fn nearest_rank(sorted: &[u32], q: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn percentiles(mut values: Vec<u32>) -> Percentiles {
    values.sort_unstable();
    Percentiles {
        p50: nearest_rank(&values, 0.50),
        p95: nearest_rank(&values, 0.95),
        p99: nearest_rank(&values, 0.99),
    }
}

/// The flat order of one query and its ground truth's positions in it.
struct FlatQuery {
    order: Vec<u32>,
    /// Position of each ground-truth document's cluster, ascending.
    gt_ranks: Vec<u32>,
    /// Position of every cluster.
    rank: Vec<u32>,
}

struct Audit<'a> {
    input: &'a AuditInput,
    options: VectorOptions,
    queries: Vec<VectorQuery<f32>>,
    flat: Vec<FlatQuery>,
    /// Query indices per set, in name order.
    sets: BTreeMap<String, Vec<usize>>,
    mean_rows: f64,
    /// Bytes of a one-bit row: code, coefficient and norm.
    one_bit_row: f64,
    /// Bytes of a full-precision row.
    exact_row: f64,
}

/// What one ranking resolves against a set's targets, before pricing.
struct Solved {
    m_r: Option<usize>,
    non_monotone: Vec<NonMonotoneStep>,
    recall: Option<f64>,
    best_recall: f64,
    below: Option<CurvePoint>,
    above: Option<CurvePoint>,
}

/// An exact `m_r`.
#[derive(Debug, PartialEq)]
struct Exact {
    m_r: usize,
    recall: f64,
    non_monotone: Vec<NonMonotoneStep>,
}

/// The smallest request reaching `target`, where `recall(lo)` falls short and `recall(hi)` does
/// not: bisects, then walks [`MONOTONE_SCAN`] requests either side and keeps the first that
/// reaches the target, because a non-monotone curve can hold one below the bisection's answer.
/// `last` is the largest request. A `lo` of 0 means no request below `hi` was tried.
fn exact_m_r(
    mut lo: usize,
    mut hi: usize,
    last: usize,
    target: f64,
    recall: &mut impl FnMut(usize) -> f64,
) -> Exact {
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if recall(mid) >= target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let window: Vec<CurvePoint> = (hi.saturating_sub(MONOTONE_SCAN).max(1)
        ..=(hi + MONOTONE_SCAN).min(last))
        .map(|k| CurvePoint {
            k,
            recall: recall(k),
        })
        .collect();
    let first = window
        .iter()
        .find(|point| point.recall >= target)
        .expect("the bisection's answer lies in the window");
    Exact {
        m_r: first.k,
        recall: first.recall,
        non_monotone: window
            .windows(2)
            .filter(|pair| pair[1].recall < pair[0].recall)
            .map(|pair| NonMonotoneStep {
                k: pair[0].k,
                recall: pair[0].recall,
                next_recall: pair[1].recall,
            })
            .collect(),
    }
}

/// The input order build `seed` shows its clusterer: a Fisher-Yates shuffle of `0..rows`.
fn shuffled_rows(rows: usize, seed: u64) -> Vec<u32> {
    let mut order: Vec<u32> = (0..rows as u32).collect();
    let mut state = seed;
    for last in (1..rows).rev() {
        // SplitMix64, so seed 0 shuffles like any other.
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        order.swap(last, (mixed % (last as u64 + 1)) as usize);
    }
    order
}

/// What `permutation`, over rows in `order`, does to the rows `order` shuffled: `result[row]` is
/// where row `row` of the unshuffled input landed.
fn unshuffled_permutation(order: &[u32], permutation: &[u32]) -> Vec<u32> {
    let mut result = vec![0; order.len()];
    for (&row, &new) in order.iter().zip(permutation) {
        result[row as usize] = new;
    }
    result
}

/// The median of `m_r`s, unreached ones last.
fn median_m_r(m_rs: &[Option<usize>]) -> Option<usize> {
    let mut sorted: Vec<usize> = m_rs.iter().map(|m| m.unwrap_or(usize::MAX)).collect();
    sorted.sort_unstable();
    Some(sorted[sorted.len() / 2]).filter(|&m| m != usize::MAX)
}

/// One target's spread over `rows`, a build each.
fn target_spread(rows: &[&RankingTarget]) -> TargetSpread {
    let m_rs: Vec<Option<usize>> = rows.iter().map(|row| row.m_r).collect();
    let reached: Vec<usize> = m_rs.iter().copied().flatten().collect();
    let unreached = m_rs.len() - reached.len();
    let min = reached.iter().copied().min();
    let max = reached.iter().copied().max().filter(|_| unreached == 0);
    let flat_m_r = rows[0].flat_m_r;
    let line = GATE_RATIO * flat_m_r as f64;
    let mut scored: Vec<f64> = rows
        .iter()
        .filter_map(|row| row.work.as_ref().map(|work| work.scored))
        .collect();
    scored.sort_unstable_by(f64::total_cmp);
    TargetSpread {
        target: rows[0].target,
        flat_m_r,
        median: median_m_r(&m_rs),
        min,
        max,
        unreached,
        median_scored: scored.get(scored.len() / 2).copied(),
        straddles: min.is_some_and(|min| min as f64 <= line)
            && max.is_none_or(|max| max as f64 > line),
    }
}

impl Audit<'_> {
    /// Mean coverage recall of `ranking` at `k` over `queries`.
    fn recall(&self, ranking: &Ranking, queries: &[usize], k: usize, policy: SearchPolicy) -> f64 {
        let covers = parallel(queries.len(), |i| {
            let q = queries[i];
            let walk = ranking.walk(&self.queries[q], k, policy);
            let output: HashSet<u32> = walk.output.into_iter().collect();
            self.input.gt_clusters[q]
                .iter()
                .filter(|cluster| output.contains(cluster))
                .count() as f64
                / self.input.gt_clusters[q].len() as f64
        });
        covers.iter().sum::<f64>() / covers.len().max(1) as f64
    }

    /// Draws the curve on the grid, then solves every target exactly inside the bracket the grid
    /// leaves.
    fn solve(
        &self,
        ranking: &Ranking,
        queries: &[usize],
        ks: &[usize],
        params: impl Fn(usize) -> (usize, SearchPolicy),
    ) -> (Vec<CurvePoint>, Vec<Solved>) {
        let mut memo = BTreeMap::new();
        let mut at = |k: usize| {
            *memo.entry(k).or_insert_with(|| {
                let (request, policy) = params(k);
                self.recall(ranking, queries, request, policy)
            })
        };
        let curve: Vec<CurvePoint> = ks
            .iter()
            .map(|&k| CurvePoint { k, recall: at(k) })
            .collect();
        let best_recall = curve.iter().map(|point| point.recall).fold(0.0, f64::max);
        let last = ks.last().copied().unwrap_or(1);
        let solved = TARGETS
            .iter()
            .map(|&target| {
                let Some(hit) = curve.iter().position(|point| point.recall >= target) else {
                    return Solved {
                        m_r: None,
                        non_monotone: Vec::new(),
                        recall: None,
                        best_recall,
                        below: curve.last().copied(),
                        above: None,
                    };
                };
                let below = hit.checked_sub(1).map(|below| curve[below]);
                let lo = below.map_or(0, |point| point.k);
                let exact = exact_m_r(lo, curve[hit].k, last, target, &mut at);
                Solved {
                    m_r: Some(exact.m_r),
                    non_monotone: exact.non_monotone,
                    recall: Some(exact.recall),
                    best_recall,
                    below,
                    above: Some(curve[hit]),
                }
            })
            .collect();
        (curve, solved)
    }

    /// Mean work of `ranking` at `k` over `queries`.
    fn work(&self, ranking: &Ranking, queries: &[usize], k: usize, policy: SearchPolicy) -> Work {
        let levels = ranking.levels();
        let walks = parallel(queries.len(), |i| {
            let walk = ranking.walk(&self.queries[queries[i]], k, policy);
            let mut lists = vec![0.0; levels];
            let mut scored = vec![0.0; levels + 1];
            for (level, stack) in ranking.router.level.levels().enumerate() {
                lists[level] = walk.trace.opened[level].len() as f64;
                scored[level] = walk.trace.opened[level]
                    .iter()
                    .map(|&list| {
                        let (start, end) = stack.offsets[list as usize];
                        (end - start) as f64
                    })
                    .sum();
            }
            scored[levels] = ranking.top() as f64;
            let routing = walk.totals.io(VectorStage::Routing);
            let exact = walk.totals.io(VectorStage::RoutingExact);
            let rows: f64 = walk
                .output
                .iter()
                .map(|&cluster| f64::from(self.input.sizes[cluster as usize]))
                .sum();
            (
                lists,
                scored,
                walk.stats.approximate_scored as f64,
                (walk.stats.members_scored - walk.stats.approximate_scored) as f64,
                (routing.reads + exact.reads) as f64,
                (routing.bytes_read + exact.bytes_read) as f64,
                rows,
            )
        });
        let count = walks.len().max(1) as f64;
        let mut work = Work {
            lists_per_level: vec![0.0; levels],
            scored_per_level: vec![0.0; levels + 1],
            ..Work::default()
        };
        for (lists, scored, one_bit, exact, reads, bytes_read, rows) in walks {
            for (total, value) in work.lists_per_level.iter_mut().zip(lists) {
                *total += value / count;
            }
            for (total, value) in work.scored_per_level.iter_mut().zip(scored) {
                *total += value / count;
            }
            work.one_bit_scored += one_bit / count;
            work.exact_scored += exact / count;
            work.reads += reads / count;
            work.bytes_read += bytes_read / count;
            work.rows_r += rows / count;
        }
        work.scored = work.one_bit_scored + work.exact_scored;
        work.bytes = work.one_bit_scored * self.one_bit_row + work.exact_scored * self.exact_row;
        work.units_r = work.bytes / (self.mean_rows * POSTING_ROW_BYTES);
        work.units_at_layer0 = work.scored / self.mean_rows;
        work
    }

    /// Classifies the ground truth the flat prefix `m` covers and `ranking` at request `m` misses.
    fn attribute(
        &self,
        ranking: &Ranking,
        queries: &[usize],
        m: usize,
        request: usize,
        policy: SearchPolicy,
    ) -> Attribution {
        let counts = parallel(queries.len(), |i| {
            let q = queries[i];
            let walk = ranking.walk(&self.queries[q], request, policy);
            let output: HashSet<u32> = walk.output.iter().copied().collect();
            let opened: HashSet<u32> = walk.trace.opened[0].iter().copied().collect();
            let retained: HashSet<u32> = walk.trace.retained.iter().copied().collect();
            let mut counts = [0usize; 6]; // flat, router, gained, unopened, dropped, late
            for &cluster in &self.input.gt_clusters[q] {
                let flat = (self.flat[q].rank[cluster as usize] as usize) < m;
                let router = output.contains(&cluster);
                counts[0] += usize::from(flat);
                counts[1] += usize::from(router);
                if router && !flat {
                    counts[2] += 1;
                } else if flat && !router {
                    let row = ranking.row(cluster);
                    if !opened.contains(&ranking.member_list[row as usize]) {
                        counts[3] += 1;
                    } else if ranking.compressed() && !retained.contains(&row) {
                        counts[4] += 1;
                    } else {
                        counts[5] += 1;
                    }
                }
            }
            counts
        });
        let mut totals = [0usize; 6];
        for count in counts {
            for (total, value) in totals.iter_mut().zip(count) {
                *total += value;
            }
        }
        let docs: usize = queries
            .iter()
            .map(|&q| self.input.gt_clusters[q].len())
            .sum();
        let missed = totals[3] + totals[4] + totals[5];
        let share = |count: usize| {
            if missed == 0 {
                0.0
            } else {
                count as f64 / missed as f64
            }
        };
        Attribution {
            m,
            flat_recall: totals[0] as f64 / docs.max(1) as f64,
            router_recall: totals[1] as f64 / docs.max(1) as f64,
            missed: missed as f64 / queries.len().max(1) as f64,
            gained: totals[2] as f64 / queries.len().max(1) as f64,
            unopened: share(totals[3]),
            dropped: share(totals[4]),
            late: share(totals[5]),
        }
    }

    /// Curves, targets, work and attribution of one ranking under one policy, per set. `params`
    /// turns a grid point into a request.
    fn report(
        &self,
        name: &str,
        ranking: &Ranking,
        ks: &BTreeMap<String, Vec<usize>>,
        sample: SearchPolicy,
        attribution: bool,
        params: impl Fn(usize) -> (usize, SearchPolicy) + Copy,
    ) -> RankingReport {
        let mut curves = BTreeMap::new();
        let mut targets = BTreeMap::new();
        for (set, queries) in &self.sets {
            let flat = self.flat_targets(queries);
            let (curve, solved) = self.solve(ranking, queries, &ks[set], params);
            let rows = solved
                .into_iter()
                .zip(&flat)
                .map(|(solved, flat)| {
                    let work = solved.m_r.map(|m_r| {
                        let (request, policy) = params(m_r);
                        self.work(ranking, queries, request, policy)
                    });
                    let attribution = attribution.then(|| {
                        let (request, policy) = params(flat.m_r);
                        self.attribute(ranking, queries, flat.m_r, request, policy)
                    });
                    RankingTarget {
                        target: flat.target,
                        m_r: solved.m_r,
                        non_monotone: solved.non_monotone,
                        flat_m_r: flat.m_r,
                        ratio: solved.m_r.map(|m_r| m_r as f64 / flat.m_r as f64),
                        recall: solved.recall,
                        best_recall: solved.best_recall,
                        below: solved.below,
                        above: solved.above,
                        work,
                        attribution,
                    }
                })
                .collect();
            curves.insert(set.clone(), curve);
            targets.insert(set.clone(), rows);
        }
        RankingReport {
            name: name.to_string(),
            aps: sample.aps,
            overscan: sample.list_overscan,
            min_fraction: sample.min_fraction,
            rerank_factor: sample.rerank_factor,
            curves,
            targets,
        }
    }

    /// [`Self::report`] over every build of a structure, and each target's spread.
    fn rebuilt_report(
        &self,
        name: &str,
        builds: &[Ranking],
        ks: &BTreeMap<String, Vec<usize>>,
        sample: SearchPolicy,
        attribution: bool,
        params: impl Fn(&Ranking, usize) -> (usize, SearchPolicy) + Copy,
    ) -> RebuiltReport {
        let reports: Vec<RankingReport> = builds
            .iter()
            .map(|ranking| {
                self.report(name, ranking, ks, sample, attribution, |k| {
                    params(ranking, k)
                })
            })
            .collect();
        let spread = self
            .sets
            .keys()
            .map(|set| {
                let rows = (0..TARGETS.len())
                    .map(|target| {
                        let rows: Vec<&RankingTarget> = reports
                            .iter()
                            .map(|report| &report.targets[set][target])
                            .collect();
                        target_spread(&rows)
                    })
                    .collect();
                (set.clone(), rows)
            })
            .collect();
        RebuiltReport {
            name: name.to_string(),
            aps: sample.aps,
            overscan: sample.list_overscan,
            min_fraction: sample.min_fraction,
            rerank_factor: sample.rerank_factor,
            spread,
            builds: reports,
        }
    }

    fn flat_targets(&self, queries: &[usize]) -> Vec<FlatTarget> {
        let clusters = self.input.sizes.len();
        let mut hits = vec![0u64; clusters];
        let mut docs = 0u64;
        for &q in queries {
            for &rank in &self.flat[q].gt_ranks {
                hits[rank as usize] += 1;
                docs += 1;
            }
        }
        TARGETS
            .iter()
            .map(|&target| {
                let mut covered = 0u64;
                let mut m_r = clusters;
                let mut recall = 1.0;
                for (m, &hit) in hits.iter().enumerate() {
                    covered += hit;
                    let at = covered as f64 / docs.max(1) as f64;
                    if at >= target {
                        m_r = m + 1;
                        recall = at;
                        break;
                    }
                }
                let rows: f64 = queries
                    .iter()
                    .map(|&q| {
                        self.flat[q].order[..m_r]
                            .iter()
                            .map(|&cluster| f64::from(self.input.sizes[cluster as usize]))
                            .sum::<f64>()
                    })
                    .sum();
                FlatTarget {
                    target,
                    m_r,
                    recall,
                    rows_r: rows / queries.len().max(1) as f64,
                }
            })
            .collect()
    }

    fn flat_report(&self, queries: &[usize]) -> FlatReport {
        let need = |docs: usize| {
            percentiles(
                queries
                    .iter()
                    .filter_map(|&q| self.flat[q].gt_ranks.get(docs - 1).map(|rank| rank + 1))
                    .collect(),
            )
        };
        FlatReport {
            queries: queries.len(),
            targets: self.flat_targets(queries),
            need_9: need(9),
            need_10: need(10),
        }
    }

    /// Each set's grid: the base grid and its flat `m_r`s.
    fn grids(&self) -> BTreeMap<String, Vec<usize>> {
        let clusters = self.input.sizes.len();
        self.sets
            .iter()
            .map(|(set, queries)| {
                let mut ks = self.input.ks.clone();
                for flat in self.flat_targets(queries) {
                    ks.push(flat.m_r);
                }
                ks.retain(|&k| k >= 1 && k <= clusters);
                ks.sort_unstable();
                ks.dedup();
                (set.clone(), ks)
            })
            .collect()
    }

    fn stored(&self) -> crate::Result<Ranking> {
        Ranking::opened(
            self.input.router_slot.clone(),
            le_bytes(&self.input.centroids),
            None,
            &self.options,
        )
    }

    /// The exported centroids in the order build `seed` sees them, and that order.
    fn shuffled(&self, seed: u64) -> (Vec<u32>, Vec<f32>) {
        let dim = self.input.dim;
        let order = shuffled_rows(self.input.sizes.len(), seed);
        let mut values = Vec::with_capacity(self.input.centroids.len());
        for &row in &order {
            let row = row as usize;
            values.extend_from_slice(&self.input.centroids[row * dim..(row + 1) * dim]);
        }
        (order, values)
    }

    /// `StackedRouter::build` over the shuffled centroids, with outputs in exported clusters.
    fn rebuilt(&self, compressed: bool, seed: u64) -> crate::Result<Ranking> {
        let (order, values) = self.shuffled(seed);
        let mut centroids = IvfCentroids::F32(IvfMatrix {
            values,
            rows: self.input.sizes.len(),
            dims: self.input.dim,
        });
        let (built, permutation) =
            StackedRouter::build_with_permutation(&self.options, &mut centroids, compressed)?;
        let IvfCentroids::F32(matrix) = centroids;
        Ranking::reopened(
            built,
            le_bytes(&matrix.values),
            unshuffled_permutation(&order, &permutation),
            &self.options,
        )
    }

    /// Every build of a rebuilt router.
    fn rebuilds(&self, compressed: bool) -> crate::Result<Vec<Ranking>> {
        (0..REBUILDS)
            .map(|seed| self.rebuilt(compressed, seed))
            .collect()
    }

    /// The posting-shaped structure over the shuffled centroids: levels from the segment's
    /// clusterer, trained as a segment is, until a bank's training sample fits one leaf. `None`
    /// when the centroids already do.
    fn posting_shaped(&self, seed: u64) -> crate::Result<Option<Ranking>> {
        let clusterer = PostingLevels {
            clusterer: &self.input.clusterer,
            options: self.options.clone(),
        };
        let clusters = self.input.sizes.len();
        if clusterer.fits_one_leaf(clusters) {
            return Ok(None);
        }
        let (order, values) = self.shuffled(seed);
        let (mut index, permutation) = StackedIvf::<InMemoryStore>::build(
            values,
            clusters,
            self.input.dim,
            &clusterer,
            router_config(),
        );
        let mut top = &mut index;
        while !clusterer.fits_one_leaf(top.nlist()) {
            top.add_level(&clusterer, router_config())
                .map_err(|error| TantivyError::InternalError(error.to_string()))?;
            top = top.parent.as_mut().expect("add_level hung a parent");
        }
        let rows = le_bytes(index.vectors.as_slice());
        let built = StackedRouter::from_index(index, &self.options, false)?;
        Ranking::reopened(
            built,
            rows,
            unshuffled_permutation(&order, &permutation),
            &self.options,
        )
        .map(Some)
    }

    /// Each stored quantized target against its rebuilds' range.
    fn rebuild_checks(
        &self,
        stored: &[RankingReport],
        rebuilt: &[RebuiltReport],
    ) -> Vec<RebuildCheck> {
        let mut checks = Vec::new();
        for rebuilds in rebuilt.iter().filter(|r| r.name.starts_with("K3r")) {
            let Some(stored) = stored.iter().find(|stored| {
                stored.rerank_factor == rebuilds.rerank_factor && stored.aps == rebuilds.aps
            }) else {
                continue;
            };
            for (set, spread) in &rebuilds.spread {
                for (spread, stored_row) in spread.iter().zip(&stored.targets[set]) {
                    let within = match stored_row.m_r {
                        Some(m_r) => {
                            spread.min.is_some_and(|min| min <= m_r)
                                && spread.max.is_none_or(|max| m_r <= max)
                        }
                        None => spread.unreached > 0,
                    };
                    checks.push(RebuildCheck {
                        rerank_factor: rebuilds.rerank_factor,
                        aps: rebuilds.aps,
                        query_set: set.clone(),
                        target: stored_row.target,
                        stored: stored_row.m_r,
                        median: spread.median,
                        min: spread.min,
                        max: spread.max,
                        within,
                    });
                }
            }
        }
        checks
    }

    /// Phase A: the stored router, and the rebuilt quantized and full-precision routers.
    fn phase_a(
        &self,
        ks: &BTreeMap<String, Vec<usize>>,
        full_precision: &[Ranking],
    ) -> crate::Result<(Vec<RankingReport>, Vec<RebuiltReport>)> {
        let stored = self.stored()?;
        let quantized = self.rebuilds(true)?;
        let mut rankings = Vec::new();
        let mut rebuilt = Vec::new();
        for aps in [true, false] {
            for &rerank_factor in &RERANK_FACTORS {
                let sample = quantized_params(0, rerank_factor, aps);
                rankings.push(self.report(
                    &format!("K3-{rerank_factor}"),
                    &stored,
                    ks,
                    stored.request(sample).1,
                    true,
                    |k| stored.request(quantized_params(k, rerank_factor, aps)),
                ));
                rebuilt.push(self.rebuilt_report(
                    &format!("K3r-{rerank_factor}"),
                    &quantized,
                    ks,
                    stored.request(sample).1,
                    true,
                    |ranking, k| ranking.request(quantized_params(k, rerank_factor, aps)),
                ));
            }
            rebuilt.push(self.rebuilt_report(
                "K1",
                full_precision,
                ks,
                full_precision[0].request(full_params(0, aps)).1,
                true,
                |ranking, k| ranking.request(full_params(k, aps)),
            ));
        }
        Ok((rankings, rebuilt))
    }

    /// Phase B's sweep over `(name, builds)`.
    fn phase_b(
        &self,
        ks: &BTreeMap<String, Vec<usize>>,
        structures: &[(&str, &[Ranking])],
    ) -> Vec<RuleReport> {
        let mut rules = Vec::new();
        for (structure, builds) in structures {
            for overscan in [1, 2, 4, 8] {
                for min_fraction in [0.0, PARENT_NPROBE_FRACTION] {
                    for aps in [true, false] {
                        let policy = rule(overscan, min_fraction, aps);
                        rules.push(RuleReport {
                            structure: (*structure).to_string(),
                            ranking: self.rebuilt_report(
                                &format!("{structure}(c={overscan}, f={min_fraction}, aps={aps})"),
                                builds,
                                ks,
                                policy,
                                false,
                                |_, k| (k, policy),
                            ),
                        });
                    }
                }
            }
        }
        rules
    }

    /// Whether S(2, 0.25, on) walks to the production entry point's output on every build.
    fn reproduces(&self, full_precision: &[Ranking], ks: &BTreeMap<String, Vec<usize>>) -> bool {
        let policy = rule(LIST_OVERSCAN, PARENT_NPROBE_FRACTION, true);
        full_precision.iter().all(|ranking| {
            self.sets.iter().all(|(set, queries)| {
                ks[set].iter().all(|&k| {
                    parallel(queries.len(), |i| {
                        let query = &self.queries[queries[i]];
                        ranking.walk(query, k, policy).output
                            == ranking.ranked(query, full_params(k, true))
                    })
                    .into_iter()
                    .all(|same| same)
                })
            })
        })
    }
}

/// For each structure and set: the cheapest rule, by median vectors scored summed over the
/// targets, whose median `m_r` is within [`GATE_RATIO`] of flat at every target.
fn gates(rules: &[RuleReport], structures: &[StructureReport]) -> Vec<GateResult> {
    let mut gates = Vec::new();
    let sets: Vec<String> = rules
        .first()
        .map(|rule| rule.ranking.spread.keys().cloned().collect())
        .unwrap_or_default();
    for structure in structures {
        for set in &sets {
            let of_structure = rules
                .iter()
                .enumerate()
                .filter(|(_, rule)| rule.structure == structure.name);
            let mut pick: Option<(usize, f64)> = None;
            let mut closest: Option<(usize, f64)> = None;
            let mut straddling = Vec::new();
            for (index, rule) in of_structure {
                let rows = &rule.ranking.spread[set];
                if rows.iter().any(|row| row.straddles) {
                    straddling.push(index);
                }
                let worst = rows
                    .iter()
                    .map(|row| {
                        row.median
                            .map_or(f64::INFINITY, |m| m as f64 / row.flat_m_r as f64)
                    })
                    .fold(0.0, f64::max);
                if worst.is_finite() && closest.is_none_or(|(_, best)| worst < best) {
                    closest = Some((index, worst));
                }
                if worst <= GATE_RATIO {
                    let scored: f64 = rows.iter().filter_map(|row| row.median_scored).sum();
                    if pick.is_none_or(|(_, best)| scored < best) {
                        pick = Some((index, scored));
                    }
                }
            }
            gates.push(GateResult {
                structure: structure.name.clone(),
                query_set: set.clone(),
                pick: pick.map(|(index, _)| index),
                closest: closest.map(|(index, _)| index),
                closest_ratio: closest.map(|(_, ratio)| ratio),
                straddling,
            });
        }
    }
    gates
}

fn quantized_params(k: usize, rerank_factor: f32, aps: bool) -> RoutingParams {
    RoutingParams {
        k,
        rerank_factor,
        min_beam_fraction: 0.5,
        aps,
    }
}

fn full_params(k: usize, aps: bool) -> RoutingParams {
    RoutingParams {
        k,
        aps,
        ..RoutingParams::default()
    }
}

fn rule(overscan: usize, min_fraction: f32, aps: bool) -> SearchPolicy {
    SearchPolicy {
        aps,
        min_fraction,
        rerank_factor: 1.0,
        list_overscan: overscan,
    }
}

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A bank of rows the clusterer reads as it reads a segment's sample.
struct Bank<'a> {
    values: &'a [f32],
    rows: Vec<u32>,
    dim: usize,
}

impl Bank<'_> {
    fn batch(&self, positions: impl Iterator<Item = usize>) -> IvfTrainingVectors {
        let mut values = Vec::new();
        let mut doc_ids = Vec::new();
        for position in positions {
            let row = self.rows[position] as usize;
            values.extend_from_slice(&self.values[row * self.dim..(row + 1) * self.dim]);
            doc_ids.push(row as DocId);
        }
        IvfTrainingVectors::F32(IvfTrainingBatch {
            matrix: IvfMatrix {
                rows: doc_ids.len(),
                dims: self.dim,
                values,
            },
            doc_ids,
        })
    }
}

impl VectorSource for Bank<'_> {
    fn num_rows(&self) -> usize {
        self.rows.len()
    }

    fn read_range(&mut self, range: Range<usize>) -> crate::Result<IvfTrainingVectors> {
        Ok(self.batch(range))
    }

    fn read_rows(&mut self, rows: &[u32]) -> crate::Result<IvfTrainingVectors> {
        Ok(self.batch(rows.iter().map(|&row| row as usize)))
    }
}

/// One level from the segment's clusterer: trained on a sample taken as a segment takes one, then
/// every member assigned.
struct PostingLevels<'a> {
    clusterer: &'a SuperKMeansClusterer,
    options: VectorOptions,
}

impl PostingLevels<'_> {
    /// Whether `n` members' training sample fits one leaf.
    fn fits_one_leaf(&self, n: usize) -> bool {
        let (size, _) = training_sample(n, self.clusterer.training_sample_ratio());
        size <= self.clusterer.max_leaf_size()
    }
}

impl LevelClusterer for PostingLevels<'_> {
    type Elem = f32;

    fn cluster(&self, data: &[f32], n: usize, dim: usize, _nlist: usize) -> (Vec<f32>, Vec<u32>) {
        let (size, interval) = training_sample(n, self.clusterer.training_sample_ratio());
        let mut sample = Bank {
            values: data,
            rows: (0..n)
                .step_by(interval)
                .take(size)
                .map(|row| row as u32)
                .collect(),
            dim,
        };
        let centroids = self
            .clusterer
            .train(&self.options, &mut sample)
            .expect("router level trains");
        let mut bank = Bank {
            values: data,
            rows: (0..n as u32).collect(),
            dim,
        };
        let assignments = self
            .clusterer
            .assign(&self.options, &mut bank, &centroids)
            .expect("router level assigns");
        let IvfCentroids::F32(matrix) = centroids;
        (matrix.values, assignments)
    }
}

fn prepare(input: &AuditInput) -> crate::Result<Audit<'_>> {
    let clusters = input.sizes.len();
    if clusters == 0
        || input.centroids.len() != clusters * input.dim
        || input.queries.len() != input.gt_clusters.len()
        || input.queries.len() != input.query_set.len()
    {
        return Err(TantivyError::InvalidArgument(
            "router audit inputs disagree on their shapes".into(),
        ));
    }
    if input
        .gt_clusters
        .iter()
        .flatten()
        .any(|&cluster| cluster as usize >= clusters)
    {
        return Err(TantivyError::InvalidArgument(
            "a ground-truth document names a cluster past the segment's".into(),
        ));
    }
    let options = VectorOptions::new(input.dim, input.metric);
    let queries: Vec<_> = input
        .queries
        .iter()
        .map(|query| VectorQuery::new(input.metric, Arc::new(query.clone())))
        .collect();
    let exact = exact::build(
        &options,
        &IvfCentroids::F32(IvfMatrix {
            values: input.centroids.clone(),
            rows: clusters,
            dims: input.dim,
        }),
    );
    let flat = parallel(queries.len(), |q| {
        let order: Vec<u32> = exact.rank(&queries[q]).map(|c| c.node).collect();
        let mut rank = vec![0u32; clusters];
        for (position, &cluster) in order.iter().enumerate() {
            rank[cluster as usize] = position as u32;
        }
        let mut gt_ranks: Vec<u32> = input.gt_clusters[q]
            .iter()
            .map(|&cluster| rank[cluster as usize])
            .collect();
        gt_ranks.sort_unstable();
        FlatQuery {
            order,
            gt_ranks,
            rank,
        }
    });
    let mut sets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (q, set) in input.query_set.iter().enumerate() {
        sets.entry(set.clone()).or_default().push(q);
    }
    let rows: u64 = input.sizes.iter().map(|&size| u64::from(size)).sum();
    Ok(Audit {
        input,
        options,
        queries,
        flat,
        sets,
        mean_rows: rows as f64 / clusters as f64,
        one_bit_row: (input.dim.div_ceil(64) * 8 + 8) as f64,
        exact_row: (input.dim * 4) as f64,
    })
}

/// Runs the requested phases.
pub fn run_audit(input: &AuditInput, phases: Phases) -> crate::Result<AuditReport> {
    let audit = prepare(input)?;
    let ks = audit.grids();
    let stored = audit.stored()?;
    let rows: u64 = input.sizes.iter().map(|&size| u64::from(size)).sum();
    let mut report = AuditReport {
        clusters: input.sizes.len(),
        rows,
        mean_cluster_rows: audit.mean_rows,
        stored_router: stored.shape(),
        flat: audit
            .sets
            .iter()
            .map(|(set, queries)| (set.clone(), audit.flat_report(queries)))
            .collect(),
        ..AuditReport::default()
    };
    let full_precision = audit.rebuilds(false)?;
    if phases.a {
        let (rankings, rebuilt) = audit.phase_a(&ks, &full_precision)?;
        report.rebuild_check = audit.rebuild_checks(&rankings, &rebuilt);
        report.rankings = rankings;
        report.rebuilt = rebuilt;
    }
    if phases.b {
        let posting_shaped: Vec<Ranking> = (0..REBUILDS)
            .map(|seed| audit.posting_shaped(seed))
            .collect::<crate::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        let mut structures: Vec<(&str, &[Ranking])> = vec![("S", &full_precision)];
        if !posting_shaped.is_empty() {
            structures.push(("L", &posting_shaped));
        }
        let shapes: Vec<StructureReport> = structures
            .iter()
            .map(|(name, builds)| StructureReport {
                name: (*name).to_string(),
                builds: builds
                    .iter()
                    .map(|ranking| BuildShape {
                        levels: ranking.shape(),
                        top: ranking.top(),
                    })
                    .collect(),
            })
            .collect();
        let rules = audit.phase_b(&ks, &structures);
        report.phase_b = Some(PhaseB {
            s_reproduces_k1: audit.reproduces(&full_precision, &ks),
            gates: gates(&rules, &shapes),
            structures: shapes,
            rules,
        });
    }
    Ok(report)
}

/// The kind `router_slot` opens to over `centroid_rows`, in stored order.
pub fn stored_router_kind(
    dim: usize,
    metric: Metric,
    router_slot: &[u8],
    centroid_rows: &[u8],
) -> crate::Result<RouterKind> {
    Ok(RouterKind::open(
        FileSlice::from(router_slot.to_vec()),
        FileSlice::from(centroid_rows.to_vec()),
        &VectorOptions::new(dim, metric),
    )?
    .kind())
}

/// The stored router's coverage recall at each online operating point: every query walks the
/// stored router at its `router_k` under the production policy, and its first `clusters` outputs
/// are scanned exactly.
pub fn offline_recall(input: &AuditInput, points: &[OnlinePoint]) -> crate::Result<Vec<f64>> {
    let audit = prepare(input)?;
    let stored = audit.stored()?;
    points
        .iter()
        .map(|point| {
            if point.router_k.len() != input.queries.len()
                || point.clusters.len() != input.queries.len()
            {
                return Err(TantivyError::InvalidArgument(
                    "an online point disagrees with the queries".into(),
                ));
            }
            let covers = parallel(input.queries.len(), |q| {
                let params = quantized_params(point.router_k[q], point.rerank_factor, true);
                let (k, policy) = stored.request(params);
                let walk = stored.walk(&audit.queries[q], k, policy);
                let probed: HashSet<u32> =
                    walk.output.into_iter().take(point.clusters[q]).collect();
                input.gt_clusters[q]
                    .iter()
                    .filter(|cluster| probed.contains(cluster))
                    .count() as f64
                    / input.gt_clusters[q].len() as f64
            });
            Ok(covers.iter().sum::<f64>() / covers.len().max(1) as f64)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 16;
    const CLUSTERS: usize = 300;
    const QUERIES: usize = 24;

    fn pseudo(i: usize) -> f32 {
        ((i * 2_654_435_761 % 1_000_003) as f32 / 500_001.0 - 1.0).sin()
    }

    fn centroids() -> Vec<f32> {
        (0..CLUSTERS * DIM).map(pseudo).collect()
    }

    /// Clusters nearest each query, ten per query, so coverage is known by construction.
    fn fixture(metric: Metric, clusterer: SuperKMeansClusterer) -> AuditInput {
        let centroids = centroids();
        let options = VectorOptions::new(DIM, metric);
        let mut matrix = IvfCentroids::F32(IvfMatrix {
            values: centroids,
            rows: CLUSTERS,
            dims: DIM,
        });
        // The stored router permutes its rows: the input holds them in router order.
        let built = RouterKind::QuantizedStacked
            .build(&options, &mut matrix)
            .expect("router builds");
        let mut router_slot = Vec::new();
        built
            .serialize(&mut router_slot)
            .expect("router serializes");
        let IvfCentroids::F32(matrix) = matrix;
        let queries: Vec<Vec<f32>> = (0..QUERIES)
            .map(|q| {
                (0..DIM)
                    .map(|d| pseudo(CLUSTERS * DIM + q * DIM + d))
                    .collect()
            })
            .collect();
        let gt_clusters = queries
            .iter()
            .map(|query| {
                let mut scored: Vec<(f32, u32)> = matrix
                    .values
                    .chunks_exact(DIM)
                    .enumerate()
                    .map(|(cluster, row)| {
                        (Metric::L2.similarity(query, row).score(), cluster as u32)
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                scored.iter().step_by(3).take(10).map(|s| s.1).collect()
            })
            .collect();
        AuditInput {
            dim: DIM,
            metric,
            centroids: matrix.values,
            router_slot,
            sizes: (0..CLUSTERS).map(|i| 1 + (i % 7) as u32).collect(),
            queries,
            gt_clusters,
            query_set: (0..QUERIES)
                .map(|q| if q < 8 { "a" } else { "b" }.to_string())
                .collect(),
            ks: default_ks(CLUSTERS),
            clusterer,
        }
    }

    fn leaf_clusterer() -> SuperKMeansClusterer {
        SuperKMeansClusterer::new()
            .with_max_leaf_size(16)
            .with_training_sample_ratio(0.5)
    }

    #[test]
    fn default_grid_stays_geometric_and_ends_at_the_cap() {
        for clusters in [8, 300, 2431, 25_000] {
            let ks = default_ks(clusters);
            assert_eq!(*ks.first().unwrap(), 8);
            assert_eq!(*ks.last().unwrap(), clusters.min(8192));
            for pair in ks.windows(2) {
                assert!(pair[1] > pair[0]);
                assert!(pair[1] as f64 / pair[0] as f64 <= 2f64.powf(0.25).max(9.0 / 8.0) + 1e-9);
            }
        }
    }

    /// Exact solves on a hand-made curve: the bisection's answer, then the smaller request a dip
    /// hides from it, and every dip inside the window.
    #[test]
    fn m_r_is_exact_and_reports_non_monotone_steps() {
        let mut steady = |k: usize| k as f64 / 100.0;
        let exact = exact_m_r(40, 80, 100, 0.5, &mut steady);
        assert_eq!(exact.m_r, 50);
        assert_eq!(exact.recall, 0.5);
        assert!(exact.non_monotone.is_empty());

        // 53 reaches 0.5, then 54 and 55 dip below it: bisection settles on 56.
        let mut dipping = |k: usize| match k {
            53 => 0.5,
            54 | 55 => 0.4,
            k if k >= 56 => 0.5 + k as f64 / 1000.0,
            k => k as f64 / 200.0,
        };
        let exact = exact_m_r(40, 80, 100, 0.5, &mut dipping);
        assert_eq!(exact.m_r, 53);
        assert_eq!(
            exact.non_monotone,
            vec![NonMonotoneStep {
                k: 53,
                recall: 0.5,
                next_recall: 0.4
            }]
        );

        // The window clips to the first and last requests.
        let mut rising = |k: usize| k as f64;
        assert_eq!(exact_m_r(0, 3, 4, 1.0, &mut rising).m_r, 1);
    }

    #[test]
    fn shuffles_are_permutations_and_seeds_differ() {
        let orders: Vec<Vec<u32>> = (0..REBUILDS).map(|seed| shuffled_rows(300, seed)).collect();
        for (seed, order) in orders.iter().enumerate() {
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..300).collect::<Vec<u32>>());
            assert_ne!(*order, (0..300).collect::<Vec<u32>>(), "seed {seed}");
        }
        for pair in orders.windows(2) {
            assert_ne!(pair[0], pair[1]);
        }
        assert_eq!(orders[3], shuffled_rows(300, 3));
    }

    /// Row 2 of the input is seen first and lands at row 7.
    #[test]
    fn permutations_map_back_through_the_shuffle() {
        assert_eq!(
            unshuffled_permutation(&[2, 0, 1], &[7, 8, 9]),
            vec![8, 9, 7]
        );
    }

    fn target_row(m_r: Option<usize>) -> RankingTarget {
        RankingTarget {
            target: 0.9,
            m_r,
            non_monotone: Vec::new(),
            flat_m_r: 100,
            ratio: None,
            recall: None,
            best_recall: 0.0,
            below: None,
            above: None,
            work: None,
            attribution: None,
        }
    }

    fn spread_of(m_rs: &[Option<usize>]) -> TargetSpread {
        let rows: Vec<RankingTarget> = m_rs.iter().map(|&m_r| target_row(m_r)).collect();
        target_spread(&rows.iter().collect::<Vec<_>>())
    }

    /// The line sits at 102 for a flat `m_r` of 100.
    #[test]
    fn spreads_take_medians_and_mark_the_gate_line() {
        let below = spread_of(&[Some(100), Some(99), Some(101), Some(102), Some(98)]);
        assert_eq!(
            (below.median, below.min, below.max, below.unreached),
            (Some(100), Some(98), Some(102), 0)
        );
        assert!(!below.straddles);
        let across = spread_of(&[Some(100), Some(99), Some(103), Some(102), Some(98)]);
        assert!(across.straddles);
        let above = spread_of(&[Some(103), Some(104), Some(105), Some(106), Some(107)]);
        assert!(!above.straddles);

        let unreached = spread_of(&[Some(100), None, None, None, Some(98)]);
        assert_eq!(
            (
                unreached.median,
                unreached.min,
                unreached.max,
                unreached.unreached
            ),
            (None, Some(98), None, 3)
        );
        assert!(unreached.straddles);
        assert_eq!(spread_of(&[Some(100), None, Some(98)]).median, Some(100));
    }

    #[test]
    fn flat_coverage_matches_a_hand_count() -> crate::Result<()> {
        let mut input = fixture(Metric::L2, leaf_clusterer());
        input.queries.truncate(1);
        input.query_set.truncate(1);
        input.gt_clusters.truncate(1);
        let audit = prepare(&input)?;
        let order = &audit.flat[0].order;
        // Three documents in the first cluster, one in the fourth, one in the tenth, five in the
        // twentieth: 0.6 of the documents by m = 1, 0.8 by m = 10, 1.0 by m = 20.
        let mut gt = vec![order[0]; 3];
        gt.push(order[3]);
        gt.push(order[9]);
        gt.extend(vec![order[19]; 5]);
        input.gt_clusters[0] = gt;
        let audit = prepare(&input)?;
        let flat = audit.flat_report(&[0]);
        assert_eq!(
            audit.flat[0].gt_ranks,
            vec![0, 0, 0, 3, 9, 19, 19, 19, 19, 19]
        );
        assert_eq!(flat.targets[0].m_r, 20);
        assert_eq!(flat.targets[0].recall, 1.0);
        assert_eq!(flat.need_9.p50, 20);
        assert_eq!(flat.need_10.p50, 20);
        let sizes = &input.sizes;
        let first_twenty: u32 = order[..20].iter().map(|&c| sizes[c as usize]).sum();
        assert_eq!(flat.targets[0].rows_r, f64::from(first_twenty));
        Ok(())
    }

    #[test]
    fn every_router_reaches_flat_recall_when_opened_in_full() -> crate::Result<()> {
        let input = fixture(Metric::Cosine, leaf_clusterer());
        let audit = prepare(&input)?;
        let queries: Vec<usize> = (0..QUERIES).collect();
        let exhaustive = rule(LIST_OVERSCAN, 1.0, false);
        for ranking in [
            audit.stored()?,
            audit.rebuilt(false, 0)?,
            audit.rebuilt(true, 1)?,
            audit
                .posting_shaped(2)?
                .expect("300 centroids need a level"),
        ] {
            assert_eq!(
                audit.recall(&ranking, &queries, CLUSTERS, exhaustive),
                1.0,
                "compressed: {}",
                ranking.compressed()
            );
        }
        Ok(())
    }

    /// The traced walk is the production walk, on every router and policy; the rule that matches
    /// the production request reproduces it.
    #[test]
    fn traced_walk_matches_rank_and_s_reproduces_k1() -> crate::Result<()> {
        let input = fixture(Metric::Cosine, leaf_clusterer());
        let audit = prepare(&input)?;
        let full_precision = audit.rebuilt(false, 0)?;
        let stored = audit.stored()?;
        for k in [8, 40, 150, CLUSTERS] {
            for query in &audit.queries {
                for aps in [true, false] {
                    let params = quantized_params(k, 2.0, aps);
                    let (request, policy) = stored.request(params);
                    assert_eq!(
                        stored.walk(query, request, policy).output,
                        stored.ranked(query, params)
                    );
                    let params = full_params(k, aps);
                    let (request, policy) = full_precision.request(params);
                    assert_eq!(
                        full_precision.walk(query, request, policy).output,
                        full_precision.ranked(query, params)
                    );
                }
            }
        }
        assert!(audit.reproduces(&[full_precision], &audit.grids()));
        Ok(())
    }

    /// Rebuilt routers report exported clusters: their outputs are a permutation away.
    #[test]
    fn rebuilt_outputs_are_exported_clusters() -> crate::Result<()> {
        let input = fixture(Metric::L2, leaf_clusterer());
        let audit = prepare(&input)?;
        for seed in 0..REBUILDS {
            for ranking in [
                audit.rebuilt(false, seed)?,
                audit
                    .posting_shaped(seed)?
                    .expect("300 centroids need a level"),
            ] {
                let walk = ranking.walk(&audit.queries[0], CLUSTERS, rule(2, 1.0, false));
                let mut output = walk.output.clone();
                output.sort_unstable();
                assert_eq!(output, (0..CLUSTERS as u32).collect::<Vec<_>>());
                assert_eq!(walk.output[0], audit.flat[0].order[0]);
                for row in 0..CLUSTERS as u32 {
                    assert_eq!(ranking.row(ranking.exported(row)), row);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn posting_levels_stop_when_a_sample_fits_one_leaf() -> crate::Result<()> {
        let input = fixture(Metric::L2, leaf_clusterer());
        let audit = prepare(&input)?;
        let ranking = audit
            .posting_shaped(0)?
            .expect("300 centroids need a level");
        let levels = PostingLevels {
            clusterer: &input.clusterer,
            options: audit.options.clone(),
        };
        let lists: Vec<usize> = ranking.shape().iter().map(|level| level.lists).collect();
        assert!(levels.fits_one_leaf(*lists.last().unwrap()));
        for &below_top in &lists[..lists.len() - 1] {
            assert!(!levels.fits_one_leaf(below_top));
        }
        assert_eq!(ranking.shape()[0].members, CLUSTERS);
        let (mut flat_input, leaf) = (input, usize::MAX);
        flat_input.clusterer = SuperKMeansClusterer::new().with_max_leaf_size(leaf);
        assert!(prepare(&flat_input)?.posting_shaped(0)?.is_none());
        Ok(())
    }

    #[test]
    fn attribution_classes_sum_to_one_and_split_by_trail() -> crate::Result<()> {
        let input = fixture(Metric::Cosine, leaf_clusterer());
        let audit = prepare(&input)?;
        let queries: Vec<usize> = (0..QUERIES).collect();
        let stored = audit.stored()?;
        let (request, policy) = stored.request(quantized_params(10, 2.0, false));
        let attribution = audit.attribute(&stored, &queries, 10, request, policy);
        if attribution.missed > 0.0 {
            let sum = attribution.unopened + attribution.dropped + attribution.late;
            assert!((sum - 1.0).abs() < 1e-9);
        }
        let exhaustive = rule(LIST_OVERSCAN, 1.0, false);
        assert_eq!(
            audit
                .attribute(&stored, &queries, CLUSTERS, CLUSTERS, exhaustive)
                .missed,
            0.0
        );
        Ok(())
    }

    #[test]
    fn run_audit_reports_every_ranking_and_rule() -> crate::Result<()> {
        let input = fixture(Metric::Cosine, leaf_clusterer());
        let report = run_audit(&input, Phases { a: true, b: true })?;
        assert_eq!(report.rankings.len(), 4);
        assert_eq!(report.rebuilt.len(), 6);
        assert!(report
            .rebuilt
            .iter()
            .all(|rebuilt| rebuilt.builds.len() == REBUILDS as usize));
        let phase_b = report.phase_b.as_ref().unwrap();
        assert_eq!(phase_b.structures.len(), 2);
        assert_eq!(phase_b.rules.len(), 32);
        assert!(phase_b.s_reproduces_k1);
        assert_eq!(phase_b.gates.len(), 4);
        assert_eq!(report.rebuild_check.len(), 24);
        serde_json::to_string(&report).expect("the report serializes");
        Ok(())
    }

    /// One-bit reads stay on `routing`; the rerank's raw rows land on `routing_exact`.
    #[test]
    fn the_rerank_reads_under_its_own_stage() -> crate::Result<()> {
        let input = fixture(Metric::Cosine, leaf_clusterer());
        let audit = prepare(&input)?;
        let stored = audit.stored()?;
        let (request, policy) = stored.request(quantized_params(40, 2.0, false));
        let walk = stored.walk(&audit.queries[0], request, policy);
        let exact = walk.totals.io(VectorStage::RoutingExact);
        assert!(walk.totals.io(VectorStage::Routing).bytes_read > 0);
        assert_eq!(
            exact.bytes_read,
            (walk.stats.leaf_exact_scored * DIM * size_of::<f32>()) as u64
        );
        Ok(())
    }

    /// The export reads every document's primary cluster and the slots a reader opened.
    #[test]
    fn export_matches_the_reader() -> crate::Result<()> {
        use crate::vector::test_support::{first_segment, FixtureSpec};

        let fixture = FixtureSpec::clustered(Metric::Cosine, 64, 8, 160)
            .router(RouterKind::QuantizedStacked)
            .schedule(&[1, 1])
            .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let (_, segment) = first_segment(&searcher);
        let reader = segment.vector_index(fixture.field)?;
        let export = reader.router_audit_export()?.expect("an IVF segment");
        let sizes = reader.cluster_sizes()?.expect("an IVF segment");
        assert_eq!(export.cluster_sizes, sizes);
        assert_eq!(
            export.centroid_rows.len(),
            sizes.len() * export.dim * size_of::<f32>()
        );
        assert_eq!(
            export.router_slot.as_slice()[0],
            reader.index().unwrap().router() as u8
        );
        let mut doc_clusters = Vec::new();
        for cluster in 0..sizes.len() {
            let docs = reader.cluster_doc_ids(cluster)?.expect("a valid cluster");
            doc_clusters.extend(docs.into_iter().map(|doc| (doc, cluster as u32)));
        }
        doc_clusters.sort_unstable();
        assert_eq!(export.doc_clusters, doc_clusters);
        Ok(())
    }

    #[test]
    fn offline_recall_truncates_to_the_probed_clusters() -> crate::Result<()> {
        let input = fixture(Metric::Cosine, leaf_clusterer());
        let all = OnlinePoint {
            rerank_factor: 4.0,
            router_k: vec![CLUSTERS; QUERIES],
            clusters: vec![CLUSTERS; QUERIES],
        };
        let none = OnlinePoint {
            rerank_factor: 4.0,
            router_k: vec![CLUSTERS; QUERIES],
            clusters: vec![0; QUERIES],
        };
        let recalls = offline_recall(&input, &[all, none])?;
        assert!(recalls[0] > 0.9);
        assert_eq!(recalls[1], 0.0);
        Ok(())
    }
}
