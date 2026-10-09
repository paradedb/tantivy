//! Cluster sources for the IVF scan.
//!
//! A [`ClusterSource`] supplies a segment scan's layer-0 [`ClusterBatch`]es in its own order and
//! decides when to stop. It never scores: the scan admits each batch and reports back how many
//! rows it scored. [`RoutedClusters`] walks the router's ranking under the probe budget, the
//! bounds gate and the filter.

use std::ops::Range;
use std::time::Instant;

use common::BitSet;

use super::backend::{KthBound, ProbeController, ProbeStats, SegmentScan, UnitPricing};
use super::bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, to_bound_space, BoundStore, HeapPeek,
    QueryBound, QueryBoundTracker, Verdict,
};
use super::cluster_plan::{CentroidScore, ClusterBatch, SelectedDocs, Selection, Stage};
use super::index_reader::VectorIndexReader;
use super::ivf::{Candidate, IvfIndex};
use super::router::{RouterIter, RouterWorkspace, RoutingParams};
use super::{enter_vector_stage, Stage as IoStage, VectorElement};
use crate::collector::sort_key::Comparator;
use crate::collector::SegmentSortKeyComputer;
use crate::fastfield::AliveBitSet;
use crate::schema::Metric;
use crate::DocId;

/// Supplies layer-0 cluster batches in its own order and decides when to stop.
pub(super) trait ClusterSource {
    /// The next cluster to admit, or `None` when the source stops. `bound` is the scan's pruning
    /// bound as of the last admitted cluster.
    fn next(&mut self, bound: &KthBound) -> crate::Result<Option<ClusterBatch<'_>>>;
    /// Records that the batch last returned by [`Self::next`] was admitted with `rows_scored`
    /// rows; `bound` already includes them.
    fn admitted(&mut self, rows_scored: usize, bound: &KthBound) -> crate::Result<()>;
    /// Folds the source's counters into `stats`.
    fn finish(self, stats: &mut ProbeStats);
}

/// Admits every batch `source` supplies into `scan`, in the source's order.
pub(super) fn admit_all<S, T, K, CTail>(
    scan: &mut SegmentScan<'_, T, K, CTail>,
    source: &mut S,
    tie_break: &mut K,
    stats: &mut ProbeStats,
) -> crate::Result<()>
where
    S: ClusterSource,
    T: VectorElement,
    K: SegmentSortKeyComputer,
    CTail: Comparator<K::SegmentSortKey>,
{
    loop {
        let scored = match source.next(scan.bound())? {
            Some(batch) => scan.admit(&batch, tie_break, stats)?,
            None => return Ok(()),
        };
        source.admitted(scored, scan.bound())?;
    }
}

/// How a cluster row is tested before scoring: the filter's matches intersected with the alive
/// docs.
pub(super) enum RowGate<'a> {
    Open,
    AliveOnly(&'a AliveBitSet),
    FilterOnly(&'a BitSet),
    FilterAndAlive {
        filter: &'a BitSet,
        filter_and_alive: BitSet,
    },
}

impl<'a> RowGate<'a> {
    pub(super) fn new(filter: Option<&'a BitSet>, alive: Option<&'a AliveBitSet>) -> Self {
        match (filter, alive) {
            (None, None) => RowGate::Open,
            (None, Some(alive)) => RowGate::AliveOnly(alive),
            (Some(filter), None) => RowGate::FilterOnly(filter),
            (Some(filter), Some(alive)) => {
                let mut filter_and_alive = filter.clone();
                filter_and_alive.intersect_update(alive.bitset());
                RowGate::FilterAndAlive {
                    filter,
                    filter_and_alive,
                }
            }
        }
    }

    fn is_open(&self) -> bool {
        matches!(self, RowGate::Open)
    }
}

enum RowVerdict {
    Keep,
    Filtered,
    Dead,
}

/// Gates one cluster's rows. Returns `(all, visited, pruned_filter, pruned_dead)`: `all` selects
/// every row; otherwise `offsets` holds the kept cluster-local offsets, possibly none.
pub(super) fn select_cluster_rows(
    doc_ids: &[DocId],
    rows: Range<usize>,
    gate: &RowGate,
    offsets: &mut Vec<usize>,
) -> (bool, usize, usize, usize) {
    offsets.clear();
    let visited = rows.len();
    let (pruned_filter, pruned_dead) = match gate {
        RowGate::Open => return (true, visited, 0, 0),
        RowGate::AliveOnly(alive) => select_rows(doc_ids, rows, offsets, |doc| {
            if alive.is_alive(doc) {
                RowVerdict::Keep
            } else {
                RowVerdict::Dead
            }
        }),
        RowGate::FilterOnly(filter) => select_rows(doc_ids, rows, offsets, |doc| {
            if filter.contains(doc) {
                RowVerdict::Keep
            } else {
                RowVerdict::Filtered
            }
        }),
        RowGate::FilterAndAlive {
            filter,
            filter_and_alive,
        } => select_rows(doc_ids, rows, offsets, |doc| {
            if filter_and_alive.contains(doc) {
                RowVerdict::Keep
            } else if !filter.contains(doc) {
                RowVerdict::Filtered
            } else {
                RowVerdict::Dead
            }
        }),
    };
    (false, visited, pruned_filter, pruned_dead)
}

/// Returns `(pruned_filter, pruned_dead)`.
#[inline(always)]
fn select_rows(
    doc_ids: &[DocId],
    rows: Range<usize>,
    offsets: &mut Vec<usize>,
    verdict: impl Fn(DocId) -> RowVerdict,
) -> (usize, usize) {
    let mut pruned_filter = 0usize;
    let mut pruned_dead = 0usize;
    for (offset, _row) in rows.enumerate() {
        let doc = doc_ids[offset];
        match verdict(doc) {
            RowVerdict::Keep => offsets.push(offset),
            RowVerdict::Filtered => pruned_filter += 1,
            RowVerdict::Dead => pruned_dead += 1,
        }
    }
    (pruned_filter, pruned_dead)
}

/// Which [`ProbeStats`] counters a routed scan maintains beyond the shared ones: a quantized
/// segment also counts empty clusters and the eligible rows it charges; a full-precision segment
/// leaves those counters at zero.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RoutedAccounting {
    Quantized,
    FullPrecision,
}

/// The query-side inputs of a routed scan.
pub(super) struct RoutedQuery<'a> {
    /// The query in routing space, as the router and the centroid margins see it.
    pub(super) vector: &'a [f32],
    /// `||q||` for the dot margin's Cauchy-Schwarz term.
    pub(super) norm: f32,
    pub(super) metric: Metric,
}

/// Clusters in the router's order, admitted under the probe budget, the bounds gate and the
/// filter. Owns the ranked stream, the probe controller, the query bound, and the DocIds and
/// selection buffers each batch borrows.
pub(super) struct RoutedClusters<'a, 'w> {
    index: &'a IvfIndex,
    reader: &'a VectorIndexReader,
    query: RoutedQuery<'a>,
    ranked: RouterIter<'a, 'w>,
    controller: ProbeController<'a>,
    bounds: BoundStore<'a>,
    tracker: QueryBoundTracker,
    gate: &'a RowGate<'a>,
    accounting: RoutedAccounting,
    docs: Vec<DocId>,
    offsets: Vec<usize>,
    routing_ns: u64,
    postings_row: usize,
    postings_skipped: usize,
    clusters_skipped_empty: usize,
    bounds_skips: u32,
    vectors_visited: usize,
    pruned_filter: usize,
    pruned_dead: usize,
    eligible: usize,
}

impl<'a, 'w> RoutedClusters<'a, 'w> {
    /// Ranks the segment's clusters for `query` and prepares the probe controller. The ranking
    /// setup is charged to `stats.routing_ns`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        index: &'a IvfIndex,
        reader: &'a VectorIndexReader,
        query: RoutedQuery<'a>,
        workspace: &'w mut RouterWorkspace,
        routing: RoutingParams,
        pricing: UnitPricing,
        recall_target: f32,
        gate: &'a RowGate<'a>,
        accounting: RoutedAccounting,
        stats: &mut ProbeStats,
    ) -> Self {
        let routing_start = Instant::now();
        let (ranked, controller) = {
            let _routing_stage = enter_vector_stage(IoStage::Routing);
            let ranked = index.rank_clusters(workspace, query.vector, routing);
            let estimator = index.recall_estimator(&ranked, query.vector, recall_target);
            let controller =
                ProbeController::new(pricing, index.num_clusters(), estimator, recall_target);
            (ranked, controller)
        };
        stats.routing_ns += routing_start.elapsed().as_nanos() as u64;
        Self {
            index,
            reader,
            query,
            ranked,
            controller,
            bounds: index.bounds(),
            tracker: QueryBoundTracker::new(),
            gate,
            accounting,
            docs: Vec::new(),
            offsets: Vec::new(),
            routing_ns: 0,
            postings_row: 0,
            postings_skipped: 0,
            clusters_skipped_empty: 0,
            bounds_skips: 0,
            vectors_visited: 0,
            pruned_filter: 0,
            pruned_dead: 0,
            eligible: 0,
        }
    }

    /// The bounds gate's verdict for `cluster`, whose routing key is `sim`.
    fn verdict(&self, cluster: usize, sim: f32) -> Verdict {
        let query_bound = self.tracker.bound();
        bounds_verdict(query_bound, || {
            let QueryBound::Armed { t } = query_bound else {
                // `bounds_verdict` never calls the margin while Filling; +inf keeps even that
                // impossibility fail-open.
                return f32::INFINITY;
            };
            // Precondition of every margin: the stream key is the exact centroid similarity; an
            // approximate key makes a skip unsound.
            #[cfg(debug_assertions)]
            {
                let stride = self.reader.options().bytes_per_vector();
                let centroid_bytes = self.index.centroid_bytes().expect("readable centroid rows");
                let exact = self.query.metric.similarity_bytes::<f32>(
                    self.query.vector,
                    &centroid_bytes[cluster * stride..(cluster + 1) * stride],
                );
                debug_assert_eq!(
                    sim,
                    exact.score(),
                    "routing stream key must be the exact centroid similarity"
                );
            }
            let r = self.bounds.ball_r(cluster);
            match self.query.metric {
                Metric::L2 | Metric::Cosine => {
                    margin_ball_ball(t, r, to_bound_space(self.query.metric, sim))
                }
                Metric::Dot => margin_ball_halfspace(sim, self.query.norm, r, t),
            }
        })
    }

    /// Counts a pulled cluster that contributes no rows.
    fn skip_empty(&mut self, bound: &KthBound) -> crate::Result<()> {
        self.postings_skipped += 1;
        if self.accounting == RoutedAccounting::Quantized {
            self.clusters_skipped_empty += 1;
        }
        self.controller.cover(bound.running_estimate())
    }
}

impl ClusterSource for RoutedClusters<'_, '_> {
    fn next(&mut self, bound: &KthBound) -> crate::Result<Option<ClusterBatch<'_>>> {
        loop {
            let routing_start = Instant::now();
            let next = {
                let _routing_stage = enter_vector_stage(IoStage::Routing);
                self.ranked.next()
            };
            self.routing_ns += routing_start.elapsed().as_nanos() as u64;
            let Some(Candidate { sim, node }) = next else {
                return Ok(None);
            };
            if !self.controller.admit() {
                return Ok(None);
            }
            let cluster = node as usize;

            // A skip charges the open share and covers the cluster: no result inside the query
            // ball remains there. The bound is the pessimistic k-th, so a skip never drops a
            // true top-k row; APS covers with the point estimate.
            if self.verdict(cluster, sim.score()) == Verdict::Skip {
                self.controller.charge_open();
                self.controller.cover(bound.running_estimate())?;
                self.bounds_skips += 1;
                continue;
            }
            self.controller.charge_open();
            let rows = self.index.cluster_range(cluster);
            if rows.is_empty() {
                self.skip_empty(bound)?;
                continue;
            }

            // Filter before any payload read: DocIds alone, then the gate.
            if !self.gate.is_open() {
                self.reader.read_doc_ids(cluster, &mut self.docs)?;
            }
            let selection_start = Instant::now();
            let (all, visited, pruned_filter, pruned_dead) = {
                let _routing_stage = enter_vector_stage(IoStage::Routing);
                select_cluster_rows(&self.docs, rows.clone(), self.gate, &mut self.offsets)
            };
            self.routing_ns += selection_start.elapsed().as_nanos() as u64;
            self.vectors_visited += visited;
            self.pruned_filter += pruned_filter;
            self.pruned_dead += pruned_dead;
            if !all && self.offsets.is_empty() {
                self.skip_empty(bound)?;
                continue;
            }
            let selection = if all {
                Selection::All
            } else {
                Selection::Rows(&self.offsets)
            };
            let docs = if self.gate.is_open() {
                SelectedDocs::Deferred
            } else {
                SelectedDocs::ByClusterOffset(&self.docs)
            };
            return Ok(Some(ClusterBatch {
                stage: Stage::Layer(0),
                cluster,
                rows,
                selection,
                docs,
                centroid: CentroidScore::Known(sim),
            }));
        }
    }

    fn admitted(&mut self, rows_scored: usize, bound: &KthBound) -> crate::Result<()> {
        self.controller.charge_rows(rows_scored);
        if self.accounting == RoutedAccounting::Quantized {
            self.eligible += rows_scored;
        }
        self.postings_row += 1;
        let probe = (self.postings_row + self.postings_skipped - 1) as u32;
        self.tracker.observe(
            self.query.metric,
            HeapPeek::from_kth(bound.running_lower()),
            probe,
        );
        self.controller.cover(bound.running_estimate())
    }

    fn finish(self, stats: &mut ProbeStats) {
        // The armed index exists exactly when the bound armed.
        debug_assert!(
            self.tracker.armed_at_probe().is_some()
                == matches!(self.tracker.bound(), QueryBound::Armed { .. })
        );
        stats.record_routing(self.ranked.metrics());
        stats.routing_ns += self.routing_ns;
        stats.postings_row += self.postings_row;
        stats.postings_skipped += self.postings_skipped;
        stats.clusters_skipped_empty += self.clusters_skipped_empty;
        stats.bounds_skips += self.bounds_skips;
        stats.vectors_visited += self.vectors_visited;
        stats.pruned_filter += self.pruned_filter;
        stats.pruned_dead += self.pruned_dead;
        stats.layer0_eligible += self.eligible;
        stats.eligible_charged += self.eligible;
        stats.record_bound_armed(self.tracker.armed_at_probe());
        self.controller.finish(stats);
    }
}
