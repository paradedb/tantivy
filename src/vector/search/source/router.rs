//! The router cluster source: clusters in the router's order, admitted under the work budget,
//! the bounds gate and the filter.

use common::BitSet;

use crate::fastfield::AliveBitSet;
use crate::schema::Metric;
use crate::vector::params::ProbeParams;
use crate::vector::router::candidate::Candidate;
use crate::vector::router::{RouterIter, RouterMetrics, RoutingParams};
use crate::vector::search::cluster_source::{ClusterRows, ClusterSource, Feedback};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::search::source::controller::{
    ProbeController, ProbeOutcome, WorkBudget, WorkMeter,
};
use crate::vector::storage::bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, to_bound_space, BoundStore, HeapPeek,
    QueryBound, QueryBoundTracker, Verdict,
};
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::enter_vector_stage;
use crate::vector::{VectorElement, VectorStage};
use crate::{DocId, SegmentReader};

// --- Filter and gate ---

/// The filter's matches as the router source gates rows with them.
pub(crate) enum DocFilter {
    /// Every doc id below `max_doc` matches, deleted docs included; no bitset is built.
    All,
    /// The matching doc ids.
    Docs(BitSet),
}

impl DocFilter {
    pub(crate) fn is_empty(&self) -> bool {
        matches!(self, Self::Docs(docs) if docs.len() == 0)
    }

    /// The share of doc ids below `max_doc` that match.
    pub(crate) fn match_fraction(&self, max_doc: DocId) -> f64 {
        match self {
            Self::All => 1.0,
            Self::Docs(docs) => docs.len() as f64 / f64::from(max_doc.max(1)),
        }
    }
}

/// How a cluster row is tested before scoring: against the filter's matches and the alive docs.
pub(crate) enum RowGate<'a> {
    Open,
    AliveOnly(&'a AliveBitSet),
    FilterOnly(BitSet),
    FilterAndAlive {
        filter: BitSet,
        alive: &'a AliveBitSet,
    },
}

/// What a gate kept of one cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GatedRows {
    /// Every row is kept; the offsets are left empty.
    pub(crate) all: bool,
    pub(crate) pruned_filter: usize,
    pub(crate) pruned_dead: usize,
}

impl<'a> RowGate<'a> {
    pub(crate) fn new(filter: DocFilter, alive: Option<&'a AliveBitSet>) -> Self {
        match (filter, alive) {
            (DocFilter::All, None) => RowGate::Open,
            (DocFilter::All, Some(alive)) => RowGate::AliveOnly(alive),
            (DocFilter::Docs(filter), None) => RowGate::FilterOnly(filter),
            (DocFilter::Docs(filter), Some(alive)) => RowGate::FilterAndAlive { filter, alive },
        }
    }

    /// Whether every row passes without reading its doc id.
    pub(crate) fn is_open(&self) -> bool {
        matches!(self, RowGate::Open)
    }

    /// Gates a cluster whose DocIds column is `doc_ids`, writing the kept cluster-local offsets.
    pub(crate) fn select(&self, doc_ids: &[DocId], offsets: &mut Vec<u32>) -> GatedRows {
        offsets.clear();
        let (pruned_filter, pruned_dead) = match self {
            RowGate::Open => {
                return GatedRows {
                    all: true,
                    pruned_filter: 0,
                    pruned_dead: 0,
                }
            }
            RowGate::AliveOnly(alive) => gate(doc_ids, offsets, |doc| {
                if alive.is_alive(doc) {
                    RowVerdict::Keep
                } else {
                    RowVerdict::Dead
                }
            }),
            RowGate::FilterOnly(filter) => gate(doc_ids, offsets, |doc| {
                if filter.contains(doc) {
                    RowVerdict::Keep
                } else {
                    RowVerdict::Filtered
                }
            }),
            RowGate::FilterAndAlive { filter, alive } => gate(doc_ids, offsets, |doc| {
                if !filter.contains(doc) {
                    RowVerdict::Filtered
                } else if alive.is_alive(doc) {
                    RowVerdict::Keep
                } else {
                    RowVerdict::Dead
                }
            }),
        };
        GatedRows {
            all: false,
            pruned_filter,
            pruned_dead,
        }
    }
}

enum RowVerdict {
    Keep,
    Filtered,
    Dead,
}

/// Keeps the offsets `verdict` accepts; returns `(pruned_filter, pruned_dead)`.
#[inline(always)]
fn gate(
    doc_ids: &[DocId],
    offsets: &mut Vec<u32>,
    verdict: impl Fn(DocId) -> RowVerdict,
) -> (usize, usize) {
    let mut pruned_filter = 0usize;
    let mut pruned_dead = 0usize;
    for (offset, &doc) in (0u32..).zip(doc_ids) {
        match verdict(doc) {
            RowVerdict::Keep => offsets.push(offset),
            RowVerdict::Filtered => pruned_filter += 1,
            RowVerdict::Dead => pruned_dead += 1,
        }
    }
    (pruned_filter, pruned_dead)
}

// --- Source ---

/// What a router source counted, kept in place while it runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RouterCounters {
    /// Clusters admitted with at least one row.
    pub(crate) postings_row: usize,
    /// Clusters opened that contributed no row: empty, or every row gated out.
    pub(crate) postings_skipped: usize,
    /// Of `postings_skipped`, clusters with no eligible row.
    pub(crate) clusters_skipped_empty: usize,
    /// Clusters the bounds gate passed over.
    pub(crate) bounds_skips: u32,
    /// Rows the gate looked at.
    pub(crate) vectors_visited: usize,
    pub(crate) pruned_filter: usize,
    pub(crate) pruned_dead: usize,
    /// Rows admitted and charged.
    pub(crate) eligible: usize,
    /// The zero-based probe at which the query bound armed.
    pub(crate) bound_armed_at: Option<u32>,
}

/// A finished router source's report.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RouterReport {
    pub(crate) counters: RouterCounters,
    pub(crate) routing: RouterMetrics,
    pub(crate) probe: ProbeOutcome,
}

/// Clusters in the router's order, admitted under the work budget, the bounds gate and the
/// filter. Owns the ranked stream, the work meter, the probe controller, the query bound and the
/// buffers each cluster's rows borrow.
pub(crate) struct RouterClusters<'a, T: VectorElement> {
    index: &'a ClusterIndex,
    reader: &'a VectorIndexReader,
    query: &'a VectorQuery<T>,
    ranked: RouterIter,
    meter: WorkMeter,
    controller: ProbeController,
    bounds: BoundStore<'a>,
    tracker: QueryBoundTracker,
    gate: RowGate<'a>,
    /// The current cluster's DocIds column; read only when the gate is closed.
    docs: Vec<DocId>,
    offsets: Vec<u32>,
    selected_docs: Vec<DocId>,
    counters: RouterCounters,
}

impl<'a, T: VectorElement> RouterClusters<'a, T> {
    /// Ranks as many of `segment`'s clusters as `budget` buys at the filter's match fraction, and
    /// prepares the probe controller. `reader` is the segment's IVF reader.
    pub(crate) fn new(
        reader: &'a VectorIndexReader,
        query: &'a VectorQuery<T>,
        budget: WorkBudget,
        filter: DocFilter,
        segment: &'a SegmentReader,
        params: &'a ProbeParams,
    ) -> Self {
        let index = reader
            .index()
            .expect("the router source runs on an IVF segment");
        let clusters = index.num_clusters();
        let routing = RoutingParams {
            rerank_factor: params.router_rerank_factor,
            aps: params.router_aps,
            k: params.router_k(
                budget.total.get(),
                budget.per_cluster.get(),
                filter.match_fraction(segment.max_doc()),
                clusters,
            ),
        };
        let _routing = enter_vector_stage(VectorStage::Routing);
        let ranked = index.rank_clusters(query, routing);
        Self {
            index,
            reader,
            query,
            ranked,
            meter: WorkMeter::new(budget),
            controller: ProbeController::new(clusters),
            bounds: index.bounds(),
            tracker: QueryBoundTracker::new(),
            gate: RowGate::new(filter, segment.alive_bitset()),
            docs: Vec::new(),
            offsets: Vec::new(),
            selected_docs: Vec::new(),
            counters: RouterCounters::default(),
        }
    }

    /// The bounds gate's verdict for `cluster`, whose routing key is `sim`.
    fn verdict(&self, cluster: usize, sim: f32) -> Verdict {
        let query_bound = self.tracker.bound();
        bounds_verdict(query_bound, || {
            let QueryBound::Armed { t } = query_bound else {
                // Never called while filling; +inf keeps the impossible case fail-open.
                return f32::INFINITY;
            };
            // Every margin assumes the stream key is the exact centroid similarity. The row is
            // read unmetered, so debug builds report the same IO.
            #[cfg(debug_assertions)]
            {
                let row = self
                    .index
                    .centroid_row_slice(cluster)
                    .and_then(|row| Ok(row.read_bytes()?))
                    .expect("readable centroid row");
                let exact = self
                    .query
                    .metric()
                    .similarity_bytes::<f32>(self.query.widened(), &row);
                debug_assert_eq!(
                    sim,
                    exact.score(),
                    "routing stream key must be the exact centroid similarity"
                );
            }
            let r = self.bounds.ball_r(cluster);
            match self.query.metric() {
                Metric::L2 | Metric::Cosine => {
                    margin_ball_ball(t, r, to_bound_space(self.query.metric(), sim))
                }
                Metric::Dot => margin_ball_halfspace(sim, self.query.norm(), r, t),
            }
        })
    }

    /// Selects `cluster`'s rows into `offsets` and counts what the gate pruned.
    fn gate_cluster(&mut self, cluster: usize, rows: usize) -> crate::Result<GatedRows> {
        // DocIds alone, then the gate, before any payload read.
        if !self.gate.is_open() {
            self.reader.read_doc_ids(cluster, &mut self.docs)?;
            debug_assert_eq!(self.docs.len(), rows);
        }
        let gated = {
            let _routing = enter_vector_stage(VectorStage::Routing);
            self.gate.select(&self.docs, &mut self.offsets)
        };
        self.counters.vectors_visited += rows;
        self.counters.pruned_filter += gated.pruned_filter;
        self.counters.pruned_dead += gated.pruned_dead;
        Ok(gated)
    }

    /// Counts an opened cluster that contributes no rows.
    fn skip_empty(&mut self, _feedback: Feedback) -> crate::Result<()> {
        self.counters.postings_skipped += 1;
        self.counters.clusters_skipped_empty += 1;
        Ok(())
    }

    /// Ends the walk and reports what it counted.
    pub(crate) fn finish(mut self) -> RouterReport {
        debug_assert!(
            self.tracker.armed_at_probe().is_some()
                == matches!(self.tracker.bound(), QueryBound::Armed { .. })
        );
        self.counters.bound_armed_at = self.tracker.armed_at_probe();
        RouterReport {
            counters: self.counters,
            routing: self.ranked.metrics(),
            probe: self.controller.finish(&self.meter),
        }
    }
}

impl<T: VectorElement> ClusterSource for RouterClusters<'_, T> {
    fn next(&mut self, feedback: Feedback) -> crate::Result<Option<ClusterRows<'_>>> {
        loop {
            let next = {
                let _routing = enter_vector_stage(VectorStage::Routing);
                self.ranked.next()
            };
            let Some(Candidate { sim, node }) = next else {
                return Ok(None);
            };
            if !self.controller.admit(&self.meter) {
                return Ok(None);
            }
            let cluster = node as usize;

            // A skip charges the open share. The bound is the pessimistic k-th, so a skip
            // never drops a true top-k row.
            if self.verdict(cluster, sim.score()) == Verdict::Skip {
                self.meter.open();
                self.counters.bounds_skips += 1;
                continue;
            }
            self.meter.open();
            let rows = self.index.cluster_range(cluster);
            if rows.is_empty() {
                self.skip_empty(feedback)?;
                continue;
            }

            let gated = self.gate_cluster(cluster, rows.len())?;
            if !gated.all && self.offsets.is_empty() {
                self.skip_empty(feedback)?;
                continue;
            }
            let (offsets, docs) = if gated.all {
                (None, None)
            } else {
                self.selected_docs.clear();
                self.selected_docs.extend(
                    self.offsets
                        .iter()
                        .map(|&offset| self.docs[offset as usize]),
                );
                (Some(&self.offsets[..]), Some(&self.selected_docs[..]))
            };
            return Ok(Some(ClusterRows {
                cluster,
                rows,
                offsets,
                docs,
                centroid: Some(sim),
            }));
        }
    }

    fn admitted(&mut self, rows: usize, feedback: Feedback) -> crate::Result<()> {
        self.meter.rows(rows);
        self.counters.eligible += rows;
        self.counters.postings_row += 1;
        let probe = (self.counters.postings_row + self.counters.postings_skipped - 1) as u32;
        self.tracker.observe(
            self.query.metric(),
            HeapPeek::from_kth(feedback.kth_lower),
            probe,
        );
        Ok(())
    }
}
