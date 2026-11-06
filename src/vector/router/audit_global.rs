//! Router quality audit across segments: an exact ranking of every segment's clusters against the
//! production global walk over the stored routers, with and without the partition layer.
//! Diagnostics only; nothing here feeds a build or a query.
//!
//! The probe unit is `(segment, cluster)`. Coverage, `m_r`, `need_9`, `need_10` and `rows_r` mean
//! what they mean in [`super::audit`], over the union of the segments. A ranking at request `k` is
//! the global walk's whole output for `router_k = k`, probed in full.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use serde::Serialize;

use super::audit::{
    flat_m_r, nearest_rank, parallel, percentiles, solve_curve, CurvePoint, FlatReport, FlatTarget,
    LevelShape, NonMonotoneStep, OnlinePoint, Work, POSTING_ROW_BYTES, TARGETS,
};
use super::candidate::Candidate;
use super::global::GlobalRanking;
use super::search::TopCandidates;
use super::stacked::search::{AllScored, RoutingOutput};
use super::{exact, OpenedRouter, Router, RouterKind, RouterMetrics, RoutingParams};
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::clustering::{IvfCentroids, IvfMatrix};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::telemetry::StageTotals;
use crate::vector::VectorStage;
use crate::TantivyError;

/// Rerank factors of the global rankings. Factor 1 is not a mode the audit measures.
const RERANK_FACTORS: [f32; 2] = [2.0, 4.0];

/// The list floor pg_search builds its routing parameters with.
const MIN_BEAM_FRACTION: f32 = 0.5;

/// A probe: a segment and one of its clusters.
type Probe = (u32, u32);

/// One segment's inputs.
pub struct GlobalSegment {
    /// The segment's id.
    pub segment_id: String,
    /// Centroid rows, `dim` values each, in stored (router) order.
    pub centroids: Vec<f32>,
    /// The stored router slot, its kind byte included.
    pub router_slot: Vec<u8>,
    /// Rows per cluster.
    pub sizes: Vec<u32>,
    /// The centroid stored for the segment's partition.
    pub partition_centroid: Option<Vec<f32>>,
}

/// One global audit run's inputs, all in memory.
pub struct GlobalAuditInput {
    /// Vector dimension.
    pub dim: usize,
    /// The field's metric.
    pub metric: Metric,
    /// The segments.
    pub segments: Vec<GlobalSegment>,
    /// Query vectors.
    pub queries: Vec<Vec<f32>>,
    /// The `(segment, cluster)` holding each ground-truth document of each query.
    pub gt_probes: Vec<Vec<(u32, u32)>>,
    /// The set each query is reported under.
    pub query_set: Vec<String>,
    /// Base grid of router requests. Each set adds its flat `m_r`s.
    pub ks: Vec<usize>,
}

/// One segment's shape.
#[derive(Clone, Debug, Serialize)]
pub struct SegmentShape {
    /// The segment's id.
    pub segment_id: String,
    /// Rows in the segment.
    pub rows: u64,
    /// Clusters in the segment.
    pub clusters: usize,
    /// Smallest cluster.
    pub min: u32,
    /// Median cluster.
    pub p50: u32,
    /// Largest cluster.
    pub max: u32,
    /// Norm of the stored partition centroid; `None` without one.
    pub partition_norm: Option<f64>,
    /// The stored router's levels, bottom first.
    pub router: Vec<LevelShape>,
}

/// Where the ground-truth documents the flat prefix covers and a router's misses went.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct GlobalAttribution {
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
    /// Share of `missed` whose segment the partition layer cut.
    pub partition: f64,
    /// Share of `missed` whose router list was never opened.
    pub unopened: f64,
    /// Share of `missed` scored one-bit and not retained for the rerank.
    pub dropped: f64,
    /// Share of `missed` kept and ranked past `m`.
    pub late: f64,
}

/// Segments a ranking reaches per query, at one request.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct SegmentCoverage {
    /// Segments whose partition survived the partition layer; all of them without one.
    pub partitions_kept: f64,
    /// Segments with a bottom list opened.
    pub opened: f64,
    /// Segments in the output.
    pub in_output: f64,
    /// Segments holding a ground-truth document.
    pub holding_gt: f64,
}

/// A ranking at one recall target.
#[derive(Clone, Debug, Serialize)]
pub struct GlobalTarget {
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
    /// Segments reached at `m_r`.
    pub segments: Option<SegmentCoverage>,
    /// Attribution at the flat `m_r`.
    pub attribution: GlobalAttribution,
}

/// One global ranking, per query set.
#[derive(Clone, Debug, Serialize)]
pub struct GlobalRankingReport {
    /// Ranking name.
    pub name: String,
    /// Whether the patience stop is on.
    pub aps: bool,
    /// Rerank retention factor.
    pub rerank_factor: f32,
    /// Whether partition centroids reach the walk.
    pub partitions: bool,
    /// Grid curves per set.
    pub curves: BTreeMap<String, Vec<CurvePoint>>,
    /// Target rows per set.
    pub targets: BTreeMap<String, Vec<GlobalTarget>>,
}

/// The whole global audit.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GlobalAuditReport {
    /// Clusters over all segments.
    pub clusters: usize,
    /// Rows over all segments.
    pub rows: u64,
    /// Mean rows per cluster.
    pub mean_cluster_rows: f64,
    /// One row per segment.
    pub segments: Vec<SegmentShape>,
    /// The flat ranking per set.
    pub flat: BTreeMap<String, FlatReport>,
    /// Whether FLAT-S, each segment ranked alone and merged, equals FLAT-G on every query.
    pub flat_s_matches_flat_g: bool,
    /// The global rankings.
    pub rankings: Vec<GlobalRankingReport>,
}

/// One walk's trail.
#[derive(Default)]
struct GlobalTrail {
    /// `(segment, list)`s opened per level.
    opened: Vec<Vec<Probe>>,
    /// Bottom members retained, before any rerank.
    retained: Vec<Probe>,
}

/// The production output, observed.
struct Observed<'a> {
    output: AllScored,
    trail: &'a mut GlobalTrail,
}

impl RoutingOutput for Observed<'_> {
    fn record(&mut self, candidate: Candidate<(usize, u32)>) {
        self.output.record(candidate);
    }

    fn finish(self, best: TopCandidates<(usize, u32)>) -> Vec<Candidate<(usize, u32)>> {
        self.output.finish(best)
    }

    fn opened(&mut self, level: usize, segment: usize, list: u32) {
        if self.trail.opened.len() <= level {
            self.trail.opened.resize_with(level + 1, Vec::new);
        }
        self.trail.opened[level].push((segment as u32, list));
    }

    fn retained(&mut self, level: usize, best: &TopCandidates<(usize, u32)>) {
        if level == 0 {
            self.trail.retained = best
                .iter()
                .map(|candidate| (candidate.node.0 as u32, candidate.node.1))
                .collect();
        }
    }
}

/// One walk's output, trail and cost.
struct GlobalWalk {
    output: Vec<Probe>,
    trail: GlobalTrail,
    metrics: Vec<RouterMetrics>,
    totals: Vec<StageTotals>,
}

/// The flat order of one query and its ground truth's positions in it.
struct FlatQuery {
    order: Vec<Probe>,
    /// Position of each ground-truth document's cluster, ascending.
    gt_ranks: Vec<u32>,
    /// Position of every probe, by global id.
    rank: Vec<u32>,
}

struct GlobalAudit<'a> {
    input: &'a GlobalAuditInput,
    routers: Vec<OpenedRouter>,
    partitions: Vec<Option<&'a [f32]>>,
    queries: Vec<VectorQuery<f32>>,
    flat: Vec<FlatQuery>,
    /// The global id of each segment's first cluster.
    first_id: Vec<usize>,
    /// The bottom list of each member, by segment and row.
    member_lists: Vec<Vec<u32>>,
    sets: BTreeMap<String, Vec<usize>>,
    flat_s_matches_flat_g: bool,
    mean_rows: f64,
    one_bit_row: f64,
    exact_row: f64,
}

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn stacked(
    router: &OpenedRouter,
) -> crate::Result<&super::stacked::StackedRouter<super::stacked::levels::LazyStore>> {
    match router {
        Router::Stacked(router) => Ok(router),
        Router::Exact(_) => Err(TantivyError::InvalidArgument(
            "the router audit needs stacked routers".into(),
        )),
    }
}

/// The bottom list of each member row of `router`.
fn member_lists(router: &OpenedRouter, rows: usize) -> crate::Result<Vec<u32>> {
    let mut lists = vec![u32::MAX; rows];
    for (list, &(start, end)) in stacked(router)?.level.offsets.iter().enumerate() {
        lists[start as usize..end as usize].fill(list as u32);
    }
    Ok(lists)
}

/// Ranks probes best first, then by segment and cluster: the order the global walk ties in.
fn sort_probes(ranked: &mut [(Candidate, Probe)]) {
    ranked.sort_unstable_by(|a, b| b.0.sim.cmp(&a.0.sim).then(a.1.cmp(&b.1)));
}

/// The segments' clusters ranked as one matrix: FLAT-G.
fn flat_global(
    query: &VectorQuery<f32>,
    router: &exact::ExactRouter<super::stacked::levels::InMemoryStore>,
    probes: &[Probe],
) -> Vec<Probe> {
    let mut ranked: Vec<_> = router
        .rank(query)
        .map(|candidate| (candidate, probes[candidate.node as usize]))
        .collect();
    sort_probes(&mut ranked);
    ranked.into_iter().map(|(_, probe)| probe).collect()
}

/// Each segment ranked alone, merged by score: FLAT-S.
fn flat_segments(
    query: &VectorQuery<f32>,
    routers: &[exact::ExactRouter<super::stacked::levels::InMemoryStore>],
) -> Vec<Probe> {
    let mut ranked: Vec<_> = routers
        .iter()
        .enumerate()
        .flat_map(|(segment, router)| {
            router
                .rank(query)
                .map(move |candidate| (candidate, (segment as u32, candidate.node)))
        })
        .collect();
    sort_probes(&mut ranked);
    ranked.into_iter().map(|(_, probe)| probe).collect()
}

fn prepare(input: &GlobalAuditInput) -> crate::Result<GlobalAudit<'_>> {
    let dim = input.dim;
    let shapes_agree = input.segments.iter().all(|segment| {
        segment.centroids.len() == segment.sizes.len() * dim
            && segment
                .partition_centroid
                .as_ref()
                .is_none_or(|centroid| centroid.len() == dim)
    });
    if input.segments.is_empty()
        || !shapes_agree
        || input.queries.len() != input.gt_probes.len()
        || input.queries.len() != input.query_set.len()
    {
        return Err(TantivyError::InvalidArgument(
            "global audit inputs disagree on their shapes".into(),
        ));
    }
    if input.gt_probes.iter().flatten().any(|&(segment, cluster)| {
        input
            .segments
            .get(segment as usize)
            .is_none_or(|stored| cluster as usize >= stored.sizes.len())
    }) {
        return Err(TantivyError::InvalidArgument(
            "a ground-truth document names a cluster past its segment's".into(),
        ));
    }
    let options = VectorOptions::new(dim, input.metric);
    let matrix = |values: Vec<f32>, rows: usize| {
        IvfCentroids::F32(IvfMatrix {
            values,
            rows,
            dims: dim,
        })
    };
    let routers = input
        .segments
        .iter()
        .map(|segment| {
            RouterKind::open(
                FileSlice::from(segment.router_slot.clone()),
                FileSlice::from(le_bytes(&segment.centroids)),
                &options,
            )
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let member_lists = routers
        .iter()
        .zip(&input.segments)
        .map(|(router, segment)| member_lists(router, segment.sizes.len()))
        .collect::<crate::Result<Vec<_>>>()?;
    let mut first_id = Vec::with_capacity(input.segments.len());
    let mut probes = Vec::new();
    let mut all_centroids = Vec::new();
    for (segment, stored) in input.segments.iter().enumerate() {
        first_id.push(probes.len());
        probes.extend((0..stored.sizes.len() as u32).map(|cluster| (segment as u32, cluster)));
        all_centroids.extend_from_slice(&stored.centroids);
    }
    let clusters = probes.len();
    let whole = exact::build(&options, &matrix(all_centroids, clusters));
    let each: Vec<_> = input
        .segments
        .iter()
        .map(|segment| {
            exact::build(
                &options,
                &matrix(segment.centroids.clone(), segment.sizes.len()),
            )
        })
        .collect();
    let queries: Vec<_> = input
        .queries
        .iter()
        .map(|query| VectorQuery::new(input.metric, Arc::new(query.clone())))
        .collect();
    let id_of = |first_id: &[usize], (segment, cluster): Probe| {
        first_id[segment as usize] + cluster as usize
    };
    let ranked = parallel(queries.len(), |q| {
        let order = flat_global(&queries[q], &whole, &probes);
        let matches = order == flat_segments(&queries[q], &each);
        let mut rank = vec![0u32; clusters];
        for (position, &probe) in order.iter().enumerate() {
            rank[id_of(&first_id, probe)] = position as u32;
        }
        let mut gt_ranks: Vec<u32> = input.gt_probes[q]
            .iter()
            .map(|&probe| rank[id_of(&first_id, probe)])
            .collect();
        gt_ranks.sort_unstable();
        (
            FlatQuery {
                order,
                gt_ranks,
                rank,
            },
            matches,
        )
    });
    let flat_s_matches_flat_g = ranked.iter().all(|(_, matches)| *matches);
    let flat = ranked.into_iter().map(|(flat, _)| flat).collect();
    let mut sets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (q, set) in input.query_set.iter().enumerate() {
        sets.entry(set.clone()).or_default().push(q);
    }
    let rows: u64 = input
        .segments
        .iter()
        .flat_map(|segment| &segment.sizes)
        .map(|&size| u64::from(size))
        .sum();
    Ok(GlobalAudit {
        input,
        routers,
        partitions: input
            .segments
            .iter()
            .map(|segment| segment.partition_centroid.as_deref())
            .collect(),
        queries,
        flat,
        first_id,
        member_lists,
        sets,
        flat_s_matches_flat_g,
        mean_rows: rows as f64 / clusters.max(1) as f64,
        one_bit_row: (dim.div_ceil(64) * 8 + 8) as f64,
        exact_row: (dim * 4) as f64,
    })
}

impl GlobalAudit<'_> {
    fn clusters(&self) -> usize {
        self.first_id.last().copied().unwrap_or(0)
            + self.input.segments.last().map_or(0, |s| s.sizes.len())
    }

    fn rank_of(&self, q: usize, (segment, cluster): Probe) -> usize {
        self.flat[q].rank[self.first_id[segment as usize] + cluster as usize] as usize
    }

    fn size_of(&self, (segment, cluster): Probe) -> f64 {
        f64::from(self.input.segments[segment as usize].sizes[cluster as usize])
    }

    /// The production global walk, observed. `partitions` off passes every centroid as `None`.
    fn walk(&self, q: usize, params: RoutingParams, partitions: bool) -> GlobalWalk {
        let routers: Vec<Option<&OpenedRouter>> = self.routers.iter().map(Some).collect();
        let none = vec![None; self.partitions.len()];
        let mut totals = vec![StageTotals::default(); routers.len()];
        let mut trail = GlobalTrail::default();
        let ranking = GlobalRanking::with_output(
            &routers,
            if partitions { &self.partitions } else { &none },
            &self.queries[q],
            params,
            &mut totals,
            &mut |_| {},
            Observed {
                output: AllScored::default(),
                trail: &mut trail,
            },
        );
        GlobalWalk {
            output: ranking
                .clusters
                .iter()
                .map(|(segment, candidate)| (*segment as u32, candidate.node))
                .collect(),
            trail,
            metrics: ranking.metrics,
            totals,
        }
    }

    fn recall(&self, queries: &[usize], params: RoutingParams, partitions: bool) -> f64 {
        let covers = parallel(queries.len(), |i| {
            let q = queries[i];
            let output: HashSet<Probe> = self
                .walk(q, params, partitions)
                .output
                .into_iter()
                .collect();
            self.input.gt_probes[q]
                .iter()
                .filter(|probe| output.contains(probe))
                .count() as f64
                / self.input.gt_probes[q].len() as f64
        });
        covers.iter().sum::<f64>() / covers.len().max(1) as f64
    }

    /// Mean work and segment reach at `params` over `queries`.
    fn work(
        &self,
        queries: &[usize],
        params: RoutingParams,
        partitions: bool,
    ) -> (Work, SegmentCoverage) {
        let levels = self
            .routers
            .iter()
            .filter_map(|router| stacked(router).ok())
            .map(|router| router.level.levels().count())
            .max()
            .unwrap_or(0);
        let has_partitions = partitions && self.partitions.iter().any(Option::is_some);
        let walks = parallel(queries.len(), |i| {
            let q = queries[i];
            let walk = self.walk(q, params, partitions);
            let mut lists = vec![0.0; levels + 1];
            let mut scored = vec![0.0; levels + 1];
            for (level, opened) in walk.trail.opened.iter().enumerate() {
                lists[level] = opened.len() as f64;
                if level == levels {
                    continue;
                }
                scored[level] = opened
                    .iter()
                    .filter_map(|&(segment, list)| {
                        let router = stacked(&self.routers[segment as usize]).ok()?;
                        let (start, end) = *router
                            .level
                            .levels()
                            .nth(level)?
                            .offsets
                            .get(list as usize)?;
                        Some((end - start) as f64)
                    })
                    .sum();
            }
            scored[levels] = if has_partitions {
                self.partitions.len() as f64
            } else {
                self.routers
                    .iter()
                    .filter_map(|router| stacked(router).ok())
                    .filter_map(|router| router.level.levels().last().map(|top| top.nlist() as f64))
                    .sum()
            };
            let mut one_bit = 0.0;
            let mut exact = 0.0;
            for metrics in &walk.metrics {
                match *metrics {
                    RouterMetrics::QuantizedStacked {
                        approximate_scored,
                        exact_scored,
                        ..
                    } => {
                        one_bit += approximate_scored as f64;
                        exact += exact_scored as f64;
                    }
                    RouterMetrics::Stacked { members_scored, .. } => exact += members_scored as f64,
                    RouterMetrics::Exact { visited_count } => exact += visited_count as f64,
                }
            }
            if has_partitions {
                exact += self.partitions.len() as f64;
            }
            let mut reads = 0.0;
            let mut bytes_read = 0.0;
            for totals in &walk.totals {
                for stage in [VectorStage::Routing, VectorStage::RoutingExact] {
                    let io = totals.io(stage);
                    reads += io.reads as f64;
                    bytes_read += io.bytes_read as f64;
                }
            }
            let rows: f64 = walk.output.iter().map(|&probe| self.size_of(probe)).sum();
            let distinct = |probes: &[Probe]| {
                probes
                    .iter()
                    .map(|&(segment, _)| segment)
                    .collect::<HashSet<_>>()
                    .len() as f64
            };
            let empty = Vec::new();
            let bottom = walk.trail.opened.first().unwrap_or(&empty);
            let kept = if has_partitions {
                walk.trail
                    .opened
                    .get(levels)
                    .map_or(0.0, |kept| distinct(kept))
            } else {
                self.partitions.len() as f64
            };
            let coverage = SegmentCoverage {
                partitions_kept: kept,
                opened: distinct(bottom),
                in_output: distinct(&walk.output),
                holding_gt: distinct(&self.input.gt_probes[q]),
            };
            (
                lists, scored, one_bit, exact, reads, bytes_read, rows, coverage,
            )
        });
        let count = walks.len().max(1) as f64;
        let mut work = Work {
            lists_per_level: vec![0.0; levels + 1],
            scored_per_level: vec![0.0; levels + 1],
            ..Work::default()
        };
        let mut reach = SegmentCoverage::default();
        for (lists, scored, one_bit, exact, reads, bytes_read, rows, coverage) in walks {
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
            reach.partitions_kept += coverage.partitions_kept / count;
            reach.opened += coverage.opened / count;
            reach.in_output += coverage.in_output / count;
            reach.holding_gt += coverage.holding_gt / count;
        }
        work.scored = work.one_bit_scored + work.exact_scored;
        work.bytes = work.one_bit_scored * self.one_bit_row + work.exact_scored * self.exact_row;
        work.units_r = work.bytes / (self.mean_rows * POSTING_ROW_BYTES);
        work.units_at_layer0 = work.scored / self.mean_rows;
        (work, reach)
    }

    /// Classifies the ground truth the flat prefix `m` covers and the walk at request `m` misses.
    fn attribute(
        &self,
        queries: &[usize],
        m: usize,
        params: RoutingParams,
        partitions: bool,
    ) -> GlobalAttribution {
        let levels = self
            .routers
            .iter()
            .filter_map(|router| stacked(router).ok())
            .map(|router| router.level.levels().count())
            .max()
            .unwrap_or(0);
        let has_partitions = partitions && self.partitions.iter().any(Option::is_some);
        let counts = parallel(queries.len(), |i| {
            let q = queries[i];
            let walk = self.walk(q, params, partitions);
            let output: HashSet<Probe> = walk.output.iter().copied().collect();
            let empty = Vec::new();
            let opened: HashSet<Probe> = walk
                .trail
                .opened
                .first()
                .unwrap_or(&empty)
                .iter()
                .copied()
                .collect();
            let kept: HashSet<u32> = walk
                .trail
                .opened
                .get(levels)
                .unwrap_or(&empty)
                .iter()
                .map(|&(segment, _)| segment)
                .collect();
            let retained: HashSet<Probe> = walk.trail.retained.iter().copied().collect();
            let mut counts = [0usize; 7]; // flat, router, gained, partition, unopened, dropped, late
            for &probe in &self.input.gt_probes[q] {
                let (segment, cluster) = probe;
                let flat = self.rank_of(q, probe) < m;
                let router = output.contains(&probe);
                counts[0] += usize::from(flat);
                counts[1] += usize::from(router);
                if router && !flat {
                    counts[2] += 1;
                } else if flat && !router {
                    let list = self.member_lists[segment as usize][cluster as usize];
                    if has_partitions && !kept.contains(&segment) {
                        counts[3] += 1;
                    } else if !opened.contains(&(segment, list)) {
                        counts[4] += 1;
                    } else if !retained.contains(&probe) {
                        counts[5] += 1;
                    } else {
                        counts[6] += 1;
                    }
                }
            }
            counts
        });
        let mut totals = [0usize; 7];
        for count in counts {
            for (total, value) in totals.iter_mut().zip(count) {
                *total += value;
            }
        }
        let docs: usize = queries.iter().map(|&q| self.input.gt_probes[q].len()).sum();
        let missed = totals[3] + totals[4] + totals[5] + totals[6];
        let share = |count: usize| {
            if missed == 0 {
                0.0
            } else {
                count as f64 / missed as f64
            }
        };
        GlobalAttribution {
            m,
            flat_recall: totals[0] as f64 / docs.max(1) as f64,
            router_recall: totals[1] as f64 / docs.max(1) as f64,
            missed: missed as f64 / queries.len().max(1) as f64,
            gained: totals[2] as f64 / queries.len().max(1) as f64,
            partition: share(totals[3]),
            unopened: share(totals[4]),
            dropped: share(totals[5]),
            late: share(totals[6]),
        }
    }

    fn flat_targets(&self, queries: &[usize]) -> Vec<FlatTarget> {
        let mut hits = vec![0u64; self.clusters()];
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
                let (m_r, recall) = flat_m_r(&hits, docs, target);
                let rows: f64 = queries
                    .iter()
                    .map(|&q| {
                        self.flat[q].order[..m_r]
                            .iter()
                            .map(|&probe| self.size_of(probe))
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

    /// One ranking over every set.
    fn report(
        &self,
        name: &str,
        rerank_factor: f32,
        aps: bool,
        partitions: bool,
    ) -> GlobalRankingReport {
        let params = |k| RoutingParams {
            k,
            rerank_factor,
            min_beam_fraction: MIN_BEAM_FRACTION,
            aps,
        };
        let mut curves = BTreeMap::new();
        let mut targets = BTreeMap::new();
        for (set, queries) in &self.sets {
            let flat = self.flat_targets(queries);
            let mut ks = self.input.ks.clone();
            ks.extend(flat.iter().map(|row| row.m_r));
            ks.retain(|&k| k >= 1 && k <= self.clusters());
            ks.sort_unstable();
            ks.dedup();
            let (curve, solved) = solve_curve(&ks, |k| self.recall(queries, params(k), partitions));
            let rows = solved
                .into_iter()
                .zip(&flat)
                .map(|(solved, flat)| {
                    let reach = solved
                        .m_r
                        .map(|m_r| self.work(queries, params(m_r), partitions));
                    let (work, segments) = reach.unzip();
                    GlobalTarget {
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
                        segments,
                        attribution: self.attribute(
                            queries,
                            flat.m_r,
                            params(flat.m_r),
                            partitions,
                        ),
                    }
                })
                .collect();
            curves.insert(set.clone(), curve);
            targets.insert(set.clone(), rows);
        }
        GlobalRankingReport {
            name: name.to_string(),
            aps,
            rerank_factor,
            partitions,
            curves,
            targets,
        }
    }

    fn segment_shapes(&self) -> crate::Result<Vec<SegmentShape>> {
        self.input
            .segments
            .iter()
            .zip(&self.routers)
            .map(|(segment, router)| {
                let mut sizes = segment.sizes.clone();
                sizes.sort_unstable();
                let router = stacked(router)?;
                let levels = router
                    .level
                    .levels()
                    .map(|level| {
                        let mut lists: Vec<u32> = level
                            .offsets
                            .iter()
                            .map(|&(start, end)| (end - start) as u32)
                            .collect();
                        lists.sort_unstable();
                        LevelShape {
                            lists: lists.len(),
                            members: lists.iter().map(|&size| size as usize).sum(),
                            min: lists.first().copied().unwrap_or(0),
                            p50: nearest_rank(&lists, 0.50),
                            p99: nearest_rank(&lists, 0.99),
                            max: lists.last().copied().unwrap_or(0),
                        }
                    })
                    .collect();
                Ok(SegmentShape {
                    segment_id: segment.segment_id.clone(),
                    rows: sizes.iter().map(|&size| u64::from(size)).sum(),
                    clusters: sizes.len(),
                    min: sizes.first().copied().unwrap_or(0),
                    p50: nearest_rank(&sizes, 0.50),
                    max: sizes.last().copied().unwrap_or(0),
                    partition_norm: segment.partition_centroid.as_ref().map(|centroid| {
                        centroid
                            .iter()
                            .map(|&value| f64::from(value) * f64::from(value))
                            .sum::<f64>()
                            .sqrt()
                    }),
                    router: levels,
                })
            })
            .collect()
    }
}

/// Runs the global audit over the stored routers.
pub fn run_global_audit(input: &GlobalAuditInput) -> crate::Result<GlobalAuditReport> {
    let audit = prepare(input)?;
    let mut rankings = Vec::new();
    for aps in [true, false] {
        for &rerank_factor in &RERANK_FACTORS {
            rankings.push(audit.report(&format!("G3-{rerank_factor}"), rerank_factor, aps, true));
        }
        rankings.push(audit.report("G3-4-nopart", 4.0, aps, false));
    }
    let rows: u64 = input
        .segments
        .iter()
        .flat_map(|segment| &segment.sizes)
        .map(|&size| u64::from(size))
        .sum();
    Ok(GlobalAuditReport {
        clusters: audit.clusters(),
        rows,
        mean_cluster_rows: audit.mean_rows,
        segments: audit.segment_shapes()?,
        flat: audit
            .sets
            .iter()
            .map(|(set, queries)| (set.clone(), audit.flat_report(queries)))
            .collect(),
        flat_s_matches_flat_g: audit.flat_s_matches_flat_g,
        rankings,
    })
}

/// The stored routers' coverage recall at each online operating point: every query walks the
/// global router at its `router_k`, and its first `clusters` outputs are scanned exactly.
pub fn global_offline_recall(
    input: &GlobalAuditInput,
    points: &[OnlinePoint],
) -> crate::Result<Vec<f64>> {
    let audit = prepare(input)?;
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
                let params = RoutingParams {
                    k: point.router_k[q],
                    rerank_factor: point.rerank_factor,
                    min_beam_fraction: MIN_BEAM_FRACTION,
                    aps: true,
                };
                let probed: HashSet<Probe> = audit
                    .walk(q, params, true)
                    .output
                    .into_iter()
                    .take(point.clusters[q])
                    .collect();
                input.gt_probes[q]
                    .iter()
                    .filter(|probe| probed.contains(probe))
                    .count() as f64
                    / input.gt_probes[q].len() as f64
            });
            Ok(covers.iter().sum::<f64>() / covers.len().max(1) as f64)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 16;
    const SEGMENTS: usize = 3;
    const CLUSTERS: usize = 60;
    const QUERIES: usize = 12;

    fn pseudo(i: usize) -> f32 {
        ((i * 2_654_435_761 % 1_000_003) as f32 / 500_001.0 - 1.0).sin()
    }

    /// Three segments around far-apart anchors, so the partition layer has something to cut.
    fn segment(index: usize, metric: Metric) -> GlobalSegment {
        let options = VectorOptions::new(DIM, metric);
        let anchor: Vec<f32> = (0..DIM)
            .map(|d| if d == index { 8.0 } else { 0.0 })
            .collect();
        let mut matrix = IvfCentroids::F32(IvfMatrix {
            values: (0..CLUSTERS * DIM)
                .map(|i| anchor[i % DIM] + pseudo(index * 7_919 + i) * 0.3)
                .collect(),
            rows: CLUSTERS,
            dims: DIM,
        });
        let built = RouterKind::QuantizedStacked
            .build(&options, &mut matrix)
            .expect("router builds");
        let mut router_slot = Vec::new();
        built
            .serialize(&mut router_slot)
            .expect("router serializes");
        let IvfCentroids::F32(matrix) = matrix;
        GlobalSegment {
            segment_id: format!("segment-{index}"),
            centroids: matrix.values,
            router_slot,
            sizes: (0..CLUSTERS).map(|i| 1 + (i % 5) as u32).collect(),
            partition_centroid: Some(anchor),
        }
    }

    fn fixture(metric: Metric) -> GlobalAuditInput {
        let queries: Vec<Vec<f32>> = (0..QUERIES)
            .map(|q| {
                (0..DIM)
                    .map(|d| {
                        let anchor = if d == q % SEGMENTS { 8.0 } else { 0.0 };
                        anchor + pseudo(1_000_000 + q * DIM + d) * 0.3
                    })
                    .collect()
            })
            .collect();
        GlobalAuditInput {
            dim: DIM,
            metric,
            segments: (0..SEGMENTS).map(|index| segment(index, metric)).collect(),
            gt_probes: queries
                .iter()
                .enumerate()
                .map(|(q, _)| {
                    (0..10)
                        .map(|doc| ((q % SEGMENTS) as u32, (doc * 3) as u32))
                        .collect()
                })
                .collect(),
            queries,
            query_set: vec!["held_out".to_string(); QUERIES],
            ks: vec![4, 8, 16, 32, 64, 128, SEGMENTS * CLUSTERS],
        }
    }

    fn params(k: usize, aps: bool) -> RoutingParams {
        RoutingParams {
            k,
            rerank_factor: 2.0,
            min_beam_fraction: MIN_BEAM_FRACTION,
            aps,
        }
    }

    #[test]
    fn flat_s_equals_flat_g_and_covers_every_probe() -> crate::Result<()> {
        let input = fixture(Metric::Cosine);
        let audit = prepare(&input)?;
        assert!(audit.flat_s_matches_flat_g);
        let mut order = audit.flat[0].order.clone();
        order.sort_unstable();
        order.dedup();
        assert_eq!(order.len(), SEGMENTS * CLUSTERS);
        Ok(())
    }

    #[test]
    fn flat_coverage_matches_a_hand_count() -> crate::Result<()> {
        let mut input = fixture(Metric::L2);
        input.queries.truncate(1);
        input.query_set.truncate(1);
        input.gt_probes.truncate(1);
        let order = prepare(&input)?.flat[0].order.clone();
        // Five documents in the first probe, four in the third, one in the seventh.
        let mut gt = vec![order[0]; 5];
        gt.extend(vec![order[2]; 4]);
        gt.push(order[6]);
        input.gt_probes[0] = gt;
        let audit = prepare(&input)?;
        let flat = audit.flat_report(&[0]);
        assert_eq!(flat.targets[0].m_r, 3);
        assert_eq!(flat.targets[1].m_r, 7);
        assert_eq!(flat.need_9.p50, 3);
        assert_eq!(flat.need_10.p50, 7);
        let rows: f64 = order[..7].iter().map(|&probe| audit.size_of(probe)).sum();
        assert_eq!(flat.targets[2].rows_r, rows);
        Ok(())
    }

    /// The observed walk is the production walk, with and without partitions.
    #[test]
    fn observed_walk_matches_production() -> crate::Result<()> {
        let input = fixture(Metric::Cosine);
        let audit = prepare(&input)?;
        let routers: Vec<Option<&OpenedRouter>> = audit.routers.iter().map(Some).collect();
        let none = vec![None; SEGMENTS];
        for (partitions, supplied) in [(true, &audit.partitions), (false, &none)] {
            for k in [4, 32, 128] {
                for aps in [true, false] {
                    for q in 0..QUERIES {
                        let mut totals = vec![StageTotals::default(); SEGMENTS];
                        let production = GlobalRanking::from(
                            &routers,
                            supplied,
                            &audit.queries[q],
                            params(k, aps),
                            &mut totals,
                            &mut |_| {},
                        );
                        let expected: Vec<Probe> = production
                            .clusters
                            .iter()
                            .map(|(segment, candidate)| (*segment as u32, candidate.node))
                            .collect();
                        assert_eq!(audit.walk(q, params(k, aps), partitions).output, expected);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn every_ranking_reaches_flat_recall_when_opened_in_full() -> crate::Result<()> {
        let input = fixture(Metric::Cosine);
        let audit = prepare(&input)?;
        let queries: Vec<usize> = (0..QUERIES).collect();
        let exhaustive = RoutingParams {
            k: SEGMENTS * CLUSTERS,
            rerank_factor: 4.0,
            min_beam_fraction: 1.0,
            aps: false,
        };
        assert_eq!(audit.recall(&queries, exhaustive, true), 1.0);
        assert_eq!(audit.recall(&queries, exhaustive, false), 1.0);
        Ok(())
    }

    /// With no list floor a small request keeps one partition: ground truth put in a far segment is
    /// cut there, and without the partition layer that class cannot occur. At the production floor
    /// the request is half the child lists, so the walk opens every partition.
    #[test]
    fn a_cut_partition_is_attributed_to_the_partition_layer() -> crate::Result<()> {
        let mut input = fixture(Metric::L2);
        for (q, gt) in input.gt_probes.iter_mut().enumerate() {
            let far = ((q + SEGMENTS - 1) % SEGMENTS) as u32;
            *gt = (0..10).map(|doc| (far, doc as u32)).collect();
        }
        let audit = prepare(&input)?;
        let queries: Vec<usize> = (0..QUERIES).collect();
        let m = SEGMENTS * CLUSTERS - 1;
        let tight = RoutingParams {
            min_beam_fraction: 0.0,
            ..params(2, false)
        };
        let cut = audit.attribute(&queries, m, tight, true);
        assert!(cut.partition > 0.0);
        assert!((cut.partition + cut.unopened + cut.dropped + cut.late - 1.0).abs() < 1e-9);
        assert_eq!(audit.attribute(&queries, m, tight, false).partition, 0.0);
        let (_, reach) = audit.work(&queries, params(2, false), true);
        assert_eq!(reach.partitions_kept, SEGMENTS as f64);
        Ok(())
    }

    #[test]
    fn work_counts_segments_and_the_partition_scan() -> crate::Result<()> {
        let input = fixture(Metric::Cosine);
        let audit = prepare(&input)?;
        let queries: Vec<usize> = (0..QUERIES).collect();
        let (work, reach) = audit.work(&queries, params(16, false), true);
        assert_eq!(reach.holding_gt, 1.0);
        assert!(reach.opened >= 1.0 && reach.opened <= SEGMENTS as f64);
        assert!(reach.partitions_kept <= SEGMENTS as f64);
        assert!(work.bytes_read > 0.0);
        let levels = work.lists_per_level.len();
        assert_eq!(work.scored_per_level[levels - 1], SEGMENTS as f64);
        let (_, without) = audit.work(&queries, params(16, false), false);
        assert_eq!(without.partitions_kept, SEGMENTS as f64);
        Ok(())
    }

    #[test]
    fn run_global_audit_reports_every_ranking() -> crate::Result<()> {
        let input = fixture(Metric::Cosine);
        let report = run_global_audit(&input)?;
        assert_eq!(report.rankings.len(), 6);
        assert!(report.flat_s_matches_flat_g);
        assert_eq!(report.segments.len(), SEGMENTS);
        assert_eq!(report.clusters, SEGMENTS * CLUSTERS);
        for segment in &report.segments {
            assert!((segment.partition_norm.unwrap() - 8.0).abs() < 1e-6);
        }
        let names: Vec<(&str, bool)> = report
            .rankings
            .iter()
            .map(|ranking| (ranking.name.as_str(), ranking.aps))
            .collect();
        assert!(names.contains(&("G3-2", false)));
        assert!(names.contains(&("G3-4-nopart", true)));
        serde_json::to_string(&report).expect("the report serializes");
        Ok(())
    }

    #[test]
    fn offline_recall_truncates_to_the_probed_clusters() -> crate::Result<()> {
        let input = fixture(Metric::Cosine);
        let all = SEGMENTS * CLUSTERS;
        let point = |clusters| OnlinePoint {
            rerank_factor: 4.0,
            router_k: vec![all; QUERIES],
            clusters: vec![clusters; QUERIES],
        };
        let recalls = global_offline_recall(&input, &[point(all), point(0)])?;
        assert!(recalls[0] > 0.9);
        assert_eq!(recalls[1], 0.0);
        Ok(())
    }
}
