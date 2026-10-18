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

struct AdaptiveProgress {
    weights: Box<[f64]>,
    sigma: f64,
    total: f64,
    remaining: f64,
    centroid: f64,
}

impl AdaptiveProgress {
    fn new(ranked: &RouterIter, clusters: usize, size: impl Fn(usize) -> usize) -> Option<Self> {
        let (mut count, mut mean, mut m2) = (0, 0.0, 0.0);
        let mut best = f64::NEG_INFINITY;
        ranked.for_each_candidate(|candidate| {
            let score = f64::from(candidate.sim.score());
            count += 1;
            let delta = score - mean;
            mean += delta / count as f64;
            m2 += delta * (score - mean);
            best = best.max(score);
        });
        let sigma = (m2 / count as f64).sqrt();
        if count != clusters || !sigma.is_finite() || sigma <= 0.0 {
            return None;
        }
        let mut weights = vec![0.0; clusters];
        ranked.for_each_candidate(|candidate| {
            let cluster = candidate.node as usize;
            weights[cluster] =
                size(cluster) as f64 * ((f64::from(candidate.sim.score()) - best) / sigma).exp();
        });
        let total: f64 = weights.iter().sum();
        (total.is_finite() && total > 0.0).then(|| Self {
            weights: weights.into_boxed_slice(),
            sigma,
            total,
            remaining: total,
            centroid: best,
        })
    }

    fn complete(&mut self, candidate: Candidate) {
        self.remaining = (self.remaining - self.weights[candidate.node as usize]).max(0.0);
        self.centroid = f64::from(candidate.sim.score());
    }

    fn score(&self, cutoff: f32, completed: usize) -> Option<f64> {
        (cutoff.is_finite() && completed > 0).then(|| {
            (completed as f64).ln() - (self.remaining / self.total).ln()
                + (f64::from(cutoff) - self.centroid) / self.sigma
        })
    }
}

#[cfg(test)]
mod adaptive_tests {
    use super::*;
    use crate::vector::distance::Similarity;

    #[test]
    fn remaining_weight_is_affine_invariant_and_keeps_unvisited_clusters() {
        let sizes = [100, 200, 50, 400];
        let mut reference: Option<f64> = None;
        for scale in [0.125_f32, 1.0, 1024.0] {
            for shift in [-16.0, 0.0, 32.0] {
                let candidates: Vec<_> = [4.0, 2.0, 0.0, -2.0]
                    .into_iter()
                    .enumerate()
                    .map(|(node, score)| Candidate {
                        node: node as u32,
                        sim: Similarity::new(scale * score + shift),
                    })
                    .collect();
                let ranked = RouterIter::Explicit(candidates.clone().into_iter());
                let mut progress = AdaptiveProgress::new(&ranked, 4, |i| sizes[i]).unwrap();
                progress.complete(candidates[0]);
                progress.complete(candidates[2]);
                assert!(
                    (progress.remaining - progress.weights[1] - progress.weights[3]).abs() < 1e-10
                );
                let score = progress.score(scale * 1.5 + shift, 2).unwrap();
                let sigma = 5.0_f64.sqrt();
                let remaining = 200.0 * (-2.0 / sigma).exp() + 400.0 * (-6.0 / sigma).exp();
                let total = 100.0 + remaining + 50.0 * (-4.0 / sigma).exp();
                let miss_fraction = remaining / total / 2.0 * (-1.5 / sigma).exp();
                assert!((score + miss_fraction.ln()).abs() < 1e-10);
                if let Some(previous) = reference {
                    assert!((score - previous).abs() < 1e-10);
                }
                reference = Some(score);
                assert!(progress.score(f32::NAN, 2).is_none());
                assert!(progress.score(0.0, 0).is_none());
            }
        }
    }

    #[test]
    fn unavailable_spread_or_incomplete_routing_disables_adaptive_progress() {
        for scores in [
            vec![],
            vec![1.0],
            vec![1.0, 1.0],
            vec![1.0, f32::NAN],
            vec![1.0, f32::INFINITY],
        ] {
            let count = scores.len();
            let ranked = RouterIter::Explicit(
                scores
                    .into_iter()
                    .enumerate()
                    .map(|(node, score)| Candidate {
                        node: node as u32,
                        sim: Similarity::new(score),
                    })
                    .collect::<Vec<_>>()
                    .into_iter(),
            );
            assert!(AdaptiveProgress::new(&ranked, count, |_| 100).is_none());
        }
        let ranked = RouterIter::Explicit(
            vec![
                Candidate {
                    node: 0,
                    sim: Similarity::new(2.0),
                },
                Candidate {
                    node: 1,
                    sim: Similarity::new(1.0),
                },
            ]
            .into_iter(),
        );
        assert!(AdaptiveProgress::new(&ranked, 3, |_| 100).is_none());
        assert!(AdaptiveProgress::new(&ranked, 2, |_| 0).is_none());
    }
}

/// Clusters in the router's order, admitted under the work budget, the bounds gate and the
/// filter. Owns the ranked stream, the work meter, the probe controller, the query bound and the
/// buffers each cluster's rows borrow.
pub(crate) struct RouterClusters<'a, T: VectorElement> {
    index: &'a ClusterIndex,
    reader: &'a VectorIndexReader,
    query: &'a VectorQuery<T>,
    ranked: RouterIter,
    adaptive: Option<AdaptiveProgress>,
    pending: Option<Candidate>,
    meter: WorkMeter,
    controller: ProbeController<'a>,
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
            k: params.router_k(
                budget.total.get(),
                budget.per_cluster.get(),
                filter.match_fraction(segment.max_doc()),
                clusters,
            ),
            recall: if params.quantized_recall.is_some() {
                1.0
            } else {
                params.router_recall_target
            },
        };
        let _routing = enter_vector_stage(VectorStage::Routing);
        let ranked = index.rank_clusters(query.widened(), routing);
        let adaptive = params
            .quantized_recall
            .filter(|p| p.target < 1.0)
            .and_then(|_| {
                AdaptiveProgress::new(&ranked, clusters, |cluster| {
                    index.cluster_range(cluster).len()
                })
            });
        let recall_target = if params.quantized_recall.is_some() {
            1.0
        } else {
            params.recall_target
        };
        let estimator = index.recall_estimator(&ranked, query.widened(), recall_target);
        Self {
            index,
            reader,
            query,
            ranked,
            adaptive,
            pending: None,
            meter: WorkMeter::new(budget),
            controller: ProbeController::new(
                clusters,
                estimator,
                recall_target,
                params.quantized_recall.is_none(),
            ),
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
    fn skip_empty(&mut self, feedback: Feedback) -> crate::Result<()> {
        self.counters.postings_skipped += 1;
        self.counters.clusters_skipped_empty += 1;
        self.controller.cover(feedback.kth_estimate)
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
    fn quantized_recall_score(&self, cutoff: f32, completed: usize) -> Option<f64> {
        self.adaptive.as_ref()?.score(cutoff, completed)
    }

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

            // A skip charges the open share and covers the cluster. The bound is the pessimistic
            // k-th, so a skip never drops a true top-k row; APS covers with the point estimate.
            if self.verdict(cluster, sim.score()) == Verdict::Skip {
                self.meter.open();
                self.controller.cover(feedback.kth_estimate)?;
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
            self.pending = Some(Candidate { sim, node });
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
        if let (Some(adaptive), Some(candidate)) = (&mut self.adaptive, self.pending.take()) {
            adaptive.complete(candidate);
        }
        self.meter.rows(rows);
        self.counters.eligible += rows;
        self.counters.postings_row += 1;
        let probe = (self.counters.postings_row + self.counters.postings_skipped - 1) as u32;
        self.tracker.observe(
            self.query.metric(),
            HeapPeek::from_kth(feedback.kth_lower),
            probe,
        );
        self.controller.cover(feedback.kth_estimate)
    }
}
