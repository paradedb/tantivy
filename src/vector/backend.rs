//! Per-segment vector search execution.
//!
//! Built once per segment by
//! [`TopDocsByVectorSimilarity`](super::collector::TopDocsByVectorSimilarity)
//! around the segment's cached [`VectorIndexReader`]. The search strategy
//! branches once, on whether the reader carries an [`IvfIndex`]: with it, the
//! filter is drained into a bitmap and the routed clusters are probed
//! adaptively; without it, the filter `Scorer` is iterated doc-by-doc and
//! every vector is scored exactly. Either way, every survivor's bytes are
//! fetched with one stride-sized read ([`VectorIndexReader::vector_bytes_for_row`])
//! — the unit the pg-backed `Directory` can serve zero-copy.

use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::time::Instant;

use common::BitSet;
use quant_model::f16::f16_to_f32;

use super::bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, to_bound_space, HeapPeek, QueryBound,
    QueryBoundTracker, Verdict,
};
use super::distance::norm_squared_wide;
use super::index_reader::{QuantizedLayerReader, VectorIndexReader};
use super::ivf::{AdaptiveProbeParams, Candidate, IvfIndex, IvfSearchMetrics, Workspace};
use super::prepared::{PreparedQuery, QuantizedQueryCache, QuantizedQueryCtx};
use super::tie_break::NoTieBreak;
use super::{enter_vector_stage, Stage, VectorElement};
use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, TopNComputer};
use crate::error::DataCorruption;
use crate::fastfield::AliveBitSet;
use crate::query::Weight;
use crate::schema::{Field, Metric};
use crate::{DocAddress, DocId, Score, SegmentOrdinal, SegmentReader, TantivyError};

/// The settled result.
type TieBreakHits<K> = Vec<(
    (Score, <K as SegmentSortKeyComputer>::SegmentSortKey),
    DocAddress,
)>;

/// The in-flight accumulator.
type TieBreakHeap<K, CTail> = TopNComputer<
    (Score, <K as SegmentSortKeyComputer>::SegmentSortKey),
    DocId,
    (NaturalComparator, CTail),
>;

/// Per-segment vector search: the segment's [`VectorIndexReader`] plus the
/// per-query state. Build via [`VectorBackend::for_segment`].
pub struct VectorBackend<T: VectorElement> {
    reader: Arc<VectorIndexReader>,
    query: Arc<PreparedQuery<T>>,
    quantized_query: Option<Arc<QuantizedQueryCtx>>,
    scan_init_ns: u64,
    query_prep_ns: u64,
    adaptive: AdaptiveProbeParams,
    segment_ord: SegmentOrdinal,
}

impl<T: VectorElement> VectorBackend<T> {
    /// Opens the segment's cached vector reader for `field` and prepares the
    /// query against the field's metric. A segment with no vector data gets
    /// the empty reader and yields no hits.
    pub(crate) fn for_segment(
        segment_reader: &SegmentReader,
        segment_ord: SegmentOrdinal,
        field: Field,
        query: Arc<Vec<T>>,
        quantized_queries: &QuantizedQueryCache,
        adaptive: AdaptiveProbeParams,
        max_scan_levels: usize,
    ) -> crate::Result<Self> {
        let reader = segment_reader.vector_index(field)?;
        let prep_start = Instant::now();
        let _query_prep_stage = enter_vector_stage(Stage::QueryPrep);
        let quantized_query = if max_scan_levels == 0 {
            None
        } else {
            reader.quantization().and_then(|quantized| {
                quantized.index_ctx().map(|index_ctx| {
                    let active_layers = max_scan_levels.min(index_ctx.specs.len());
                    quantized_queries.resolve(index_ctx, query.as_slice(), active_layers)
                })
            })
        };
        let query = Arc::new(PreparedQuery::<T>::new(reader.options().metric(), query));
        Ok(Self {
            reader,
            query,
            quantized_query,
            scan_init_ns: 0,
            query_prep_ns: prep_start.elapsed().as_nanos() as u64,
            adaptive,
            segment_ord,
        })
    }

    pub(crate) fn add_scan_init_ns(&mut self, elapsed_ns: u64) {
        self.scan_init_ns = self.scan_init_ns.saturating_add(elapsed_ns);
    }

    pub(crate) fn query_prep_ns(&self) -> u64 {
        self.query_prep_ns
    }

    /// Top-N within this segment: probe routed clusters when the reader has
    /// an index, exact-scan otherwise. Hits come back already tagged with
    /// `DocAddress`, so the collector doesn't need a second pass to attach
    /// the segment. The segment's [`ProbeStats`] ride along: the IVF path
    /// fills the probe-loop counters, the flat exact path only
    /// `exact_rows_read` plus exact-stage timings.
    pub fn top_n(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let (hits, stats) = self.top_n_by(
            weight,
            segment_reader,
            top_n,
            &mut NoTieBreak,
            NaturalComparator,
        )?;
        Ok((
            hits.into_iter()
                .map(|((score, ()), address)| (score, address))
                .collect(),
            stats,
        ))
    }

    /// [`Self::top_n`] ordered by `(similarity, tie_break)` rather than
    /// similarity alone.
    ///
    /// The tie-break participates in the heap's eviction decision, not just in
    /// the ordering of what survives: candidates that tie on similarity at the
    /// k/k+1 boundary are separated by `tie_break` before `DocId` is consulted.
    /// Applying a secondary key to the returned rows instead would be too late,
    /// since the losing ties are already gone.
    ///
    /// The similarity component is always compared with [`NaturalComparator`]
    /// ("higher is better"); `tie_comparator` orders the tail alone.
    pub fn top_n_by<K, CTail>(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
    ) -> crate::Result<(TieBreakHits<K>, ProbeStats)>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let mut stats = ProbeStats {
            scan_init_ns: self.scan_init_ns,
            query_prep_ns: self.query_prep_ns,
            ..Default::default()
        };
        let hits = match self.reader.index() {
            None => self.exact_top_n(
                weight,
                segment_reader,
                top_n,
                tie_break,
                tie_comparator,
                &mut stats,
            )?,
            Some(index) => match &self.quantized_query {
                Some(quantized_query) => self.quantized_top_n(
                    index,
                    quantized_query,
                    weight,
                    segment_reader,
                    top_n,
                    tie_break,
                    tie_comparator,
                    &mut stats,
                )?,
                None => self.approximate_top_n(
                    index,
                    weight,
                    segment_reader,
                    top_n,
                    tie_break,
                    tie_comparator,
                    &mut stats,
                )?,
            },
        };
        Ok((hits, stats))
    }

    /// Flat exact scan: drain the filter DocSet doc-by-doc, scoring each
    /// survivor from one stride-sized row read. Fills only the
    /// `exact_rows_read` stat.
    fn exact_top_n<K, CTail>(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let init_start = Instant::now();
        let init_stage = enter_vector_stage(Stage::ScanInit);
        // `for_each_no_score` walks the filter DocSet in ascending doc order,
        // which permits the fast `TopNComputer::push` path (valid only under
        // ascending-doc pushes).
        // `NaturalComparator` because similarity is "higher = better" — see
        // the note on `scan_clusters`.
        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        let alive = segment_reader.alive_bitset();
        let mut rows_read = 0usize;
        // Row reads are ranged and can fail; the `for_each` closure can't
        // return an error, so the first one is parked here and re-raised
        // after the walk.
        let mut read_err: Option<TantivyError> = None;
        drop(init_stage);
        stats.scan_init_ns = stats
            .scan_init_ns
            .saturating_add(init_start.elapsed().as_nanos() as u64);
        let scan_start = Instant::now();
        let exact_scan_stage = enter_vector_stage(Stage::ExactScan);
        weight.for_each_no_score(segment_reader, &mut |docs| {
            if read_err.is_some() {
                return;
            }
            for &doc in docs {
                if let Some(bs) = alive {
                    if !bs.is_alive(doc) {
                        continue;
                    }
                }
                let Some(row) = self.reader.row_id(doc) else {
                    continue;
                };
                match self.reader.vector_bytes_for_row(row) {
                    Ok(vbytes) => {
                        rows_read += 1;
                        let score = self.query.score_doc_bytes(&vbytes);
                        if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                            topn.push(key, doc);
                        }
                    }
                    Err(err) => {
                        read_err = Some(err);
                        return;
                    }
                }
            }
        })?;
        if let Some(err) = read_err {
            return Err(err);
        }
        drop(exact_scan_stage);
        stats.exact_scan_ns = Some(scan_start.elapsed().as_nanos() as u64);
        stats.exact_rows_read += rows_read;
        let segment_ord = self.segment_ord;
        let assembly_start = Instant::now();
        let _assembly_stage = enter_vector_stage(Stage::ResultAssembly);
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect();
        stats.result_assembly_ns = Some(assembly_start.elapsed().as_nanos() as u64);
        Ok(hits)
    }
}

/// A candidate's composite heap key, or `None` when its similarity alone
/// cannot beat the heap threshold, so the tie-break column is read only for
/// competitive candidates. The caller pushes the key itself, with whichever of
/// [`TopNComputer::push`]/`push_unordered` its doc arrival order permits.
///
/// The skip is exact rather than approximate: `(s, t) < (ts, tt)` requires
/// either `s < ts`, or `s == ts` with `t < tt`. So a candidate rejected here on
/// similarity alone could never have survived the full composite comparison,
/// and the tie-break lookup it would have cost is pure waste. Once the heap has
/// filled this is the common case.
#[inline(always)]
fn tie_break_key<K, CTail>(
    topn: &TieBreakHeap<K, CTail>,
    tie_break: &mut K,
    score: Score,
    doc: DocId,
) -> Option<(Score, K::SegmentSortKey)>
where
    K: SegmentSortKeyComputer,
    CTail: Comparator<K::SegmentSortKey>,
{
    if let Some(((threshold_score, _), _)) = &topn.threshold {
        if score < *threshold_score {
            return None;
        }
    }
    Some((score, tie_break.segment_sort_key(doc, score)))
}

/// How the probe loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
pub enum ProbeTermination {
    /// The work-unit probe budget was spent - the probe ceiling.
    Ceiling,
    /// The ranked centroids were exhausted before the ceiling bound. The
    /// bounds gate never terminates the scan - a skip is per-cluster and
    /// charges the open share; only the ceiling and the stream end it.
    #[default]
    Exhausted,
}

/// Per-segment probe-loop instrumentation: a prune breakdown of every
/// doc the inner loop touched, plus posting-fetch counters. Returned by
/// [`VectorBackend::top_n`] alongside the hits. The flat/exact path fills
/// `exact_rows_read` and exact-stage timings; every other funnel field is
/// IVF-probe-only.
#[derive(Debug, Default)]
pub struct LayerProbeStats {
    scan_ns: u64,
    boundary_ns: u64,
    scored: usize,
    survivors: usize,
    bias: Option<f32>,
    calibration: Option<f32>,
}

/// Candidate identities at the quantized stage boundaries, used only by the
/// deterministic end-to-end fixture to classify a mismatch as routing,
/// banding, or rerank. This never enters the production stats wire format.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct QuantizedStageTrace {
    pub(crate) scored_docs: Vec<DocId>,
    pub(crate) boundary_docs: Vec<Vec<DocId>>,
    pub(crate) rerank_docs: Vec<DocId>,
}

/// Sparse, zero-based per-layer instrumentation. A layer is inserted only
/// when its scorer executes, so omission remains the wire-level signal for a
/// truncated or shorter schedule.
#[derive(Debug, Default)]
pub struct LayerProbeStatsSet(Vec<LayerProbeStats>);

impl LayerProbeStatsSet {
    fn layer_mut(&mut self, layer: usize) -> &mut LayerProbeStats {
        if self.0.len() <= layer {
            self.0.resize_with(layer + 1, LayerProbeStats::default);
        }
        &mut self.0[layer]
    }

    pub fn get(&self, layer: usize) -> Option<&LayerProbeStats> {
        self.0.get(layer)
    }

    fn clear_timings(&mut self) {
        for layer in &mut self.0 {
            layer.scan_ns = 0;
            layer.boundary_ns = 0;
        }
    }
}

impl serde::Serialize for LayerProbeStatsSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(None)?;
        for (index, layer) in self.0.iter().enumerate() {
            map.serialize_entry(&format!("layer{index}_scan_ns"), &layer.scan_ns)?;
            map.serialize_entry(&format!("layer{index}_scored"), &layer.scored)?;
            map.serialize_entry(&format!("layer{index}_survivors"), &layer.survivors)?;
            if let Some(bias) = layer.bias {
                map.serialize_entry(&format!("layer{index}_bias"), &bias)?;
            }
            if let Some(calibration) = layer.calibration {
                map.serialize_entry(&format!("layer{index}_cal"), &calibration)?;
            }
            map.serialize_entry(&format!("boundary{index}_ns"), &layer.boundary_ns)?;
        }
        map.end()
    }
}

impl LayerProbeStats {
    pub fn scan_ns(&self) -> u64 {
        self.scan_ns
    }

    pub fn boundary_ns(&self) -> u64 {
        self.boundary_ns
    }

    pub fn scored(&self) -> usize {
        self.scored
    }

    pub fn survivors(&self) -> usize {
        self.survivors
    }

    pub fn calibration(&self) -> Option<f32> {
        self.calibration
    }
}

#[derive(Debug, Default, serde::Serialize)]
pub struct ProbeStats {
    /// Docs or posting memberships scored by the active scan path. On the
    /// quantized path this is the layer-0 membership count before κ bands.
    pub candidates_scored: usize,
    /// Sparse layer-indexed timing and funnel counters, flattened on the wire.
    #[serde(flatten)]
    pub layers: LayerProbeStatsSet,
    #[cfg(test)]
    #[serde(skip)]
    pub(crate) quantized_trace: QuantizedStageTrace,
    /// Distinct documents fetched for exact rerank.
    pub rerank_rows: usize,
    /// Segment reader, id-map, metadata/header, filter, and scan-state setup
    /// before routing or exact scoring begins. Query transformation and
    /// quantized scorer/CAL resolution live exclusively in `query_prep_ns`.
    pub scan_init_ns: u64,
    /// Segment-query rotation, bitplane, and LUT preparation time.
    pub query_prep_ns: u64,
    /// Lazy cluster-routing time, including every ranked-stream pull.
    pub routing_ns: u64,
    /// Exact scan time for an unquantized or explicitly exact path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_scan_ns: Option<u64>,
    /// Final heap-to-result assembly plus collector orchestration that wraps
    /// the individually timed scan stages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_assembly_ns: Option<u64>,
    /// Rerank dedup/storage ordering plus exact-row fetch time.
    pub rerank_fetch_ns: u64,
    /// Exact score, tie-break, and heap time for the rerank set.
    pub rerank_score_ns: u64,
    /// Every doc-id the inner loop touched, before any gate — the denominator
    /// for the prune breakdown.
    pub vectors_visited: usize,
    /// Touched docs rejected by `filter.contains`.
    pub pruned_filter: usize,
    /// Touched docs rejected by `is_alive`.
    pub pruned_dead: usize,
    /// Touched docs rejected by the replica `seen` dedup.
    pub pruned_seen: usize,
    /// Probed clusters whose surviving rows' posting bytes were fetched —
    /// one stride-sized ranged read per surviving row. Counts clusters,
    /// not rows.
    pub postings_row: usize,
    /// Probed clusters that fetched no posting bytes at all: the
    /// `filter → alive → seen` pre-pass left zero survivors (fully
    /// filtered / dead / already-seen, or the cluster is empty). The two
    /// `postings_*` counters partition the probed clusters:
    /// [`clusters_probed`](Self::clusters_probed) `== postings_row + postings_skipped`.
    pub postings_skipped: usize,
    /// Flat/exact-path stride-sized row reads — one per survivor scored.
    /// Filled only by the exact (non-IVF) path.
    pub exact_rows_read: usize,
    /// Routing cost of ranking the clusters to probe: centroids scored
    /// (`routing.visited_count`), plus the centroid-graph beam counters when
    /// routing went through the RNG. Ranking is lazy, so this covers only as
    /// much routing as the probe loop actually pulled. See
    /// [`IvfSearchMetrics`].
    #[serde(skip)]
    pub routing: IvfSearchMetrics,
    /// Centroids scored while producing routing order.
    pub routing_visited_count: usize,
    /// Segments routed through the centroid graph (0 or 1 per segment query).
    pub routing_graph_count: usize,
    /// Graph nodes visited and scored.
    pub routing_graph_visited_count: usize,
    /// Graph frontier candidates expanded.
    pub routing_graph_expanded_count: usize,
    /// Graph adjacency entries scanned.
    pub routing_graph_edges_scanned: usize,
    /// Graph result-set evictions.
    pub routing_graph_evictions: usize,
    /// Graph candidates returned before probing stopped.
    pub routing_graph_result_count: usize,
    /// Clusters the bounds gate passed over with a Skip verdict, without
    /// opening them: their margins proved they could not improve the
    /// armed result. Each charged the open share. Disjoint from the
    /// `postings_*` partition, which only counts opened clusters.
    pub bounds_skips: u32,
    /// Number of segment scans in which the query bound armed.
    pub bound_armed_count: u32,
    /// Sum of the zero-based probe index where the bound armed.
    pub bound_armed_probe_sum: u64,
    /// How the probe loop terminated. Per-segment; does not sum.
    pub termination: ProbeTermination,
    /// Work units this segment's probe loop charged against its resolved
    /// budget: opens at `x`, scored rows at `(1 - x)/n_avg`. The
    /// budget identity is per segment:
    /// `budget <= work_charged <= budget + last cluster's charge` on
    /// Ceiling terminations.
    pub work_charged: f32,
    /// IVF posting-membership rows in this segment, for deriving mean posting
    /// size without consulting config-side geometry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_rows: Option<usize>,
    /// IVF clusters in this segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_clusters: Option<usize>,
}

impl ProbeStats {
    pub(crate) fn stage_elapsed_ns(&self) -> u64 {
        let fixed = self
            .scan_init_ns
            .saturating_add(self.query_prep_ns)
            .saturating_add(self.routing_ns)
            .saturating_add(self.exact_scan_ns.unwrap_or_default())
            .saturating_add(self.result_assembly_ns.unwrap_or_default())
            .saturating_add(self.rerank_fetch_ns)
            .saturating_add(self.rerank_score_ns);
        self.layers.0.iter().fold(fixed, |total, layer| {
            total
                .saturating_add(layer.scan_ns)
                .saturating_add(layer.boundary_ns)
        })
    }

    fn record_routing(&mut self, routing: IvfSearchMetrics) {
        self.routing = routing;
        self.routing_visited_count += routing.visited_count;
        if let Some(graph) = routing.graph {
            self.routing_graph_count += 1;
            self.routing_graph_visited_count += graph.visited_count;
            self.routing_graph_expanded_count += graph.expanded_count;
            self.routing_graph_edges_scanned += graph.edges_scanned;
            self.routing_graph_evictions += graph.evictions;
            self.routing_graph_result_count += graph.result_count;
        }
    }

    fn record_bound_armed(&mut self, at_probe: Option<u32>) {
        if let Some(probe) = at_probe {
            self.bound_armed_count += 1;
            self.bound_armed_probe_sum += u64::from(probe);
        }
    }

    fn start_layer(&mut self, layer: usize, bias: f32, calibration: f32) {
        let layer = self.layers.layer_mut(layer);
        layer.bias = Some(bias);
        layer.calibration = Some(calibration);
    }

    fn record_layer_scan(&mut self, layer: usize, scored: usize, elapsed_ns: u64) {
        let stats = self.layers.layer_mut(layer);
        stats.scored += scored;
        stats.scan_ns += elapsed_ns;
    }

    fn record_boundary(&mut self, layer: usize, survivors: usize, elapsed_ns: u64) {
        let stats = self.layers.layer_mut(layer);
        stats.survivors += survivors;
        stats.boundary_ns += elapsed_ns;
    }

    pub(crate) fn clear_stage_timings(&mut self) {
        self.scan_init_ns = 0;
        self.query_prep_ns = 0;
        self.routing_ns = 0;
        self.layers.clear_timings();
        self.exact_scan_ns = self.exact_scan_ns.map(|_| 0);
        self.result_assembly_ns = self.result_assembly_ns.map(|_| 0);
        self.rerank_fetch_ns = 0;
        self.rerank_score_ns = 0;
    }

    /// Clusters the probe loop visited.
    ///
    /// Returns (`usize`): `postings_row + postings_skipped` — every probed
    /// cluster either fetched survivors or fetched nothing.
    #[inline]
    pub fn clusters_probed(&self) -> usize {
        self.postings_row + self.postings_skipped
    }
}

/// THE WORK-UNIT MODEL
///
/// The probe budget meters WORK: 1 unit = one average cluster of work,
/// with `n_avg = N / C` global across the index's IVF segments. Charging
/// is event-wise:
///
/// | event                    | charge          |
/// |--------------------------|-----------------|
/// | open a cluster           | `x`             |
/// | scored row               | `(1 - x)/n_avg` |
///
/// Only pre-pass survivors charge row work: filter/alive-rejected rows
/// and deduped replica re-encounters charge nothing (their buffer I/O
/// may still be paid), so a doc charges one row-deduction index-wide.
///
/// NORMALIZATION IDENTITY: an exhaustive, unfiltered, delete-free scan
/// charges `C*x + (1 - x)*N/n_avg = exactly C` units, so the probe
/// fraction keeps its scale across cluster granularities.
///
/// BOUNDARY RULE: the budget is inspected only at cluster boundaries -
/// open iff `remaining > 0`, deduct as-you-go, never truncate mid-cluster
/// (posting order is not distance order, so a partial scan is random loss
/// on a paid open). Overshoot is bounded by the last cluster's charge. No
/// pre-open cost knowledge is needed or used.
///
/// The bounds gate rides on this accounting: a skipped cluster charges
/// the open share (invariant: free skips break the normalization
/// identity), spends no row work, and never terminates the scan - the
/// budget and stream exhaustion are the only stops.
///
/// FIXED_PROBE_COST_ROWS is the fixed component of a probe — the cluster
/// OPEN — denominated in rows of full work, fitted on the reference
/// fixture; `x = fixed_probe_cost_rows() / (fixed_probe_cost_rows() +
/// n_avg)` self-calibrates to the index's granularity. Defaults to this
/// fitted value; runtime-settable via [`set_fixed_probe_cost_rows`] for
/// testing/calibration only. Despite "probe" in the name it covers ONLY
/// the open - routing/search cost is NOT modeled; removed once search is
/// costed.
pub const DEFAULT_FIXED_PROBE_COST_ROWS: f64 = 1.64;

/// Current FIXED_PROBE_COST_ROWS value, stored as f64 bits. See
/// [`DEFAULT_FIXED_PROBE_COST_ROWS`].
static FIXED_PROBE_COST_ROWS_BITS: AtomicU64 =
    AtomicU64::new(DEFAULT_FIXED_PROBE_COST_ROWS.to_bits());

/// Overrides the fixed per-probe cost (the cluster OPEN), in rows of full
/// work. Testing/calibration knob; non-finite or non-positive values reset
/// to [`DEFAULT_FIXED_PROBE_COST_ROWS`].
pub fn set_fixed_probe_cost_rows(v: f64) {
    let v = if v.is_finite() && v > 0.0 {
        v
    } else {
        DEFAULT_FIXED_PROBE_COST_ROWS
    };
    FIXED_PROBE_COST_ROWS_BITS.store(v.to_bits(), Relaxed);
}

/// The current fixed per-probe cost (the cluster OPEN), in rows of full
/// work. See [`DEFAULT_FIXED_PROBE_COST_ROWS`].
pub(crate) fn fixed_probe_cost_rows() -> f64 {
    f64::from_bits(FIXED_PROBE_COST_ROWS_BITS.load(Relaxed))
}

/// The per-index open share: what fraction of one average cluster's work
/// opening it costs. Covers the open only - routing/search cost is NOT
/// modeled (see [`DEFAULT_FIXED_PROBE_COST_ROWS`]).
///
/// * `n_avg` (`f64`) — native docs per cluster (see [`WorkModel`]).
///
/// Returns (`f64`): `fixed_probe_cost_rows() / (fixed_probe_cost_rows() +
/// n_avg)`, clamped to (0, 0.5] — a share above one half would mean opens
/// dominate rows, which only degenerate sub-2-row clusters produce.
pub(crate) fn open_share(n_avg: f64) -> f64 {
    let fixed = fixed_probe_cost_rows();
    (fixed / (fixed + n_avg.max(0.0))).min(0.5)
}

/// An amount of probe WORK, in the model's own unit: 1 unit is one
/// average cluster of work. Budgets, prices, and running spends share
/// this type so they compose only with each other; accumulation is f64.
///
/// NORMALIZATION IDENTITY: an exhaustive, unfiltered, delete-free scan of
/// a segment with `C` clusters charges exactly `C` units - the property
/// that lets the probe fraction keep its meaning across indexes with
/// different cluster granularity.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
pub struct WorkUnits(f64);

impl WorkUnits {
    /// No work.
    pub const ZERO: WorkUnits = WorkUnits(0.0);

    /// Wraps an amount already denominated in work units.
    ///
    /// * `units` (`f64`) — the amount, in work units.
    ///
    /// Returns (`WorkUnits`): the typed amount.
    #[inline]
    pub fn new(units: f64) -> WorkUnits {
        WorkUnits(units)
    }

    /// The raw amount, for arithmetic that genuinely leaves the unit.
    ///
    /// Returns (`f64`): the amount, in work units.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }

    /// The single narrowing point, for the telemetry fold.
    ///
    /// Returns (`f32`): the amount, narrowed once for `ProbeStats`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32
    }
}

impl std::ops::Add for WorkUnits {
    type Output = WorkUnits;
    #[inline]
    fn add(self, rhs: WorkUnits) -> WorkUnits {
        WorkUnits(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for WorkUnits {
    #[inline]
    fn add_assign(&mut self, rhs: WorkUnits) {
        self.0 += rhs.0;
    }
}

impl std::ops::Mul<f64> for WorkUnits {
    type Output = WorkUnits;
    /// Scaling by a COUNT (rows charged at one price) stays in the unit.
    #[inline]
    fn mul(self, rhs: f64) -> WorkUnits {
        WorkUnits(self.0 * rhs)
    }
}

/// The resolved per-segment prices the probe loop charges against its
/// budget: an open costs `open`, a scored row costs `row`. Built once
/// per segment from [`AdaptiveProbeParams::resolved_work_budget`]'s
/// `(budget, n_avg, x)`.
#[derive(Clone, Copy, Debug)]
struct UnitPricing {
    /// Work this segment may spend before the ceiling binds.
    budget: WorkUnits,
    /// The per-index open share `x`: what opening one cluster costs.
    open: WorkUnits,
    /// `(1 - x)/n_avg`: what one scored row costs.
    row: WorkUnits,
}

/// One gate survivor from the pre-pass over a cluster's rows: `row`
/// indexes into the segment-wide dense rows slot.
#[derive(Clone, Copy)]
struct Survivor {
    row: usize,
    doc: DocId,
}

/// A boundary survivor materialized only while compacting/reordering the SoA
/// scan buffers. The hot scan loops append primitive columns directly.
#[derive(Clone, Copy)]
struct QuantizedCandidate {
    row: usize,
    doc: DocId,
    score: f32,
    bias_correction: f32,
    sigma: f32,
}

/// Dense, row-parallel scan state. Keeping the numeric columns separate lets
/// the combine and uncertainty passes stream contiguous f32 slices without
/// constructing one candidate object per scored row.
struct QuantizedCandidates {
    rows: Vec<usize>,
    docs: Vec<DocId>,
    scores: Vec<f32>,
    bias_corrections: Vec<f32>,
    sigmas: Vec<f32>,
}

impl QuantizedCandidates {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            rows: Vec::with_capacity(capacity),
            docs: Vec::with_capacity(capacity),
            scores: Vec::with_capacity(capacity),
            bias_corrections: Vec::with_capacity(capacity),
            sigmas: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn push(&mut self, row: usize, doc: DocId, score: f32, bias_correction: f32, sigma: f32) {
        self.rows.push(row);
        self.docs.push(doc);
        self.scores.push(score);
        self.bias_corrections.push(bias_correction);
        self.sigmas.push(sigma);
    }

    /// The sole production site that applies calibration centering.
    #[inline(always)]
    fn estimate(&self, index: usize) -> f32 {
        self.scores[index] + self.bias_corrections[index]
    }

    fn materialize(&self, index: usize) -> QuantizedCandidate {
        QuantizedCandidate {
            row: self.rows[index],
            doc: self.docs[index],
            score: self.scores[index],
            bias_correction: self.bias_corrections[index],
            sigma: self.sigmas[index],
        }
    }

    fn replace_with_boundary_survivors(&mut self, survivors: &[QuantizedCandidate]) {
        self.rows.clear();
        self.docs.clear();
        self.scores.clear();
        self.bias_corrections.clear();
        self.sigmas.clear();
        self.rows.reserve(survivors.len());
        self.docs.reserve(survivors.len());
        self.scores.reserve(survivors.len());
        self.bias_corrections.reserve(survivors.len());
        self.sigmas.reserve(survivors.len());
        for survivor in survivors {
            self.push(
                survivor.row,
                survivor.doc,
                survivor.score,
                survivor.bias_correction,
                survivor.sigma,
            );
        }
    }
}

/// Pass A: decode the complete LE-f16 scale stream into reusable f32 scratch.
/// The loop body is branch-free; special-value handling lives in the inlined
/// bit-manipulation implementation of `f16_to_f32`.
#[inline(always)]
fn decode_scales(scales: &[u8], decoded: &mut Vec<f32>) {
    assert_eq!(scales.len() % std::mem::size_of::<u16>(), 0);
    decoded.resize(scales.len() / std::mem::size_of::<u16>(), 0.0);
    for (out, bytes) in decoded.iter_mut().zip(scales.chunks_exact(2)) {
        let bits = bytes[0] as u16 | (bytes[1] as u16) << 8;
        *out = f16_to_f32(bits);
    }
}

/// Decode a contiguous LE-f32 slot once, before the floating-point combine.
#[inline(always)]
fn decode_f32s(bytes: &[u8], decoded: &mut Vec<f32>) {
    assert_eq!(bytes.len() % std::mem::size_of::<f32>(), 0);
    decoded.resize(bytes.len() / std::mem::size_of::<f32>(), 0.0);
    for (out, bytes) in decoded.iter_mut().zip(bytes.chunks_exact(4)) {
        let bits = bytes[0] as u32
            | (bytes[1] as u32) << 8
            | (bytes[2] as u32) << 16
            | (bytes[3] as u32) << 24;
        *out = f32::from_bits(bits);
    }
}

/// Passes B and C for the cosine/dot plane-1 path. Keeping the score/bias
/// assembly and uncertainty scaling as separate f32-only loops gives LLVM a
/// straight-line vectorization unit for each output stream.
#[inline(always)]
fn combine_initial_dot_decoded(
    kernel_scores: &mut [f32],
    bias_scores: &mut [f32],
    sigma_scores: &mut [f32],
    decoded_scales: &[f32],
    cluster_score: f32,
    bias_factor: f32,
    sigma_factor: f32,
) {
    debug_assert_eq!(kernel_scores.len(), decoded_scales.len());
    debug_assert_eq!(bias_scores.len(), decoded_scales.len());
    debug_assert_eq!(sigma_scores.len(), decoded_scales.len());

    // Pass B: f32-only FMA assembly over contiguous arrays.
    for ((score, bias), &scale) in kernel_scores
        .iter_mut()
        .zip(bias_scores.iter_mut())
        .zip(decoded_scales)
    {
        *score = scale.mul_add(*score, cluster_score);
        *bias = scale * bias_factor;
    }

    // Pass C: f32-only uncertainty scaling.
    for (sigma, &scale) in sigma_scores.iter_mut().zip(decoded_scales) {
        *sigma = scale * sigma_factor;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn combine_initial_decoded(
    metric: Metric,
    kernel_scores: &mut [f32],
    bias_scores: &mut [f32],
    sigma_scores: &mut [f32],
    decoded_scales: &[f32],
    decoded_constants: &[f32],
    decoded_residual_norms: &[f32],
    cluster_score: f32,
    bias_factor: f32,
    sigma_factor: f32,
) {
    debug_assert_eq!(kernel_scores.len(), decoded_scales.len());
    debug_assert_eq!(bias_scores.len(), decoded_scales.len());
    debug_assert_eq!(sigma_scores.len(), decoded_scales.len());
    match metric {
        Metric::L2 => {
            debug_assert_eq!(decoded_constants.len(), decoded_scales.len());
            debug_assert_eq!(decoded_residual_norms.len(), decoded_scales.len());
            // Pass B: f32-only FMA assembly over contiguous arrays.
            for ((((score, bias), &scale), &constant), &residual_norm_sq) in kernel_scores
                .iter_mut()
                .zip(bias_scores.iter_mut())
                .zip(decoded_scales)
                .zip(decoded_constants)
                .zip(decoded_residual_norms)
            {
                *score = (2.0 * scale)
                    .mul_add(*score, cluster_score - 2.0 * constant - residual_norm_sq);
                *bias = scale * bias_factor;
            }
            // Pass C: f32-only uncertainty scaling.
            for (sigma, &scale) in sigma_scores.iter_mut().zip(decoded_scales) {
                *sigma = scale * sigma_factor;
            }
        }
        Metric::Dot | Metric::Cosine => combine_initial_dot_decoded(
            kernel_scores,
            bias_scores,
            sigma_scores,
            decoded_scales,
            cluster_score,
            bias_factor,
            sigma_factor,
        ),
    }
}

/// Criterion-only entry for the complete plane-1 cluster shape. The feature
/// gate keeps this receipt out of the default public API while the function
/// itself reuses the production kernel, scale decoder, and combine passes.
#[cfg(feature = "quantization-bench")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn quantization_bench_plane1_cosine_cluster(
    prepared: &cascade::PreparedSplitQuery,
    spec: cascade::LayerSpec,
    codes: &[u8],
    code_stride: usize,
    scales: &[u8],
    cluster_score: f32,
    bias_factor: f32,
    sigma_factor: f32,
    kernel_scores: &mut Vec<f32>,
    decoded_scales: &mut Vec<f32>,
    bias_scores: &mut Vec<f32>,
    sigma_scores: &mut Vec<f32>,
) -> f32 {
    let rows = scales.len() / std::mem::size_of::<u16>();
    assert_eq!(codes.len(), rows * code_stride);
    kernel_scores.resize(rows, 0.0);
    prepared.score_layer_batch_unscaled(0, codes, code_stride, spec, kernel_scores);
    decode_scales(scales, decoded_scales);
    bias_scores.resize(rows, 0.0);
    sigma_scores.resize(rows, 0.0);
    combine_initial_decoded(
        Metric::Cosine,
        kernel_scores,
        bias_scores,
        sigma_scores,
        decoded_scales,
        &[],
        &[],
        cluster_score,
        bias_factor,
        sigma_factor,
    );

    // One value from every output stream makes the writes observable without
    // adding a reduction to the measured per-row loop shape.
    kernel_scores[rows - 1] + bias_scores[rows - 1] + sigma_scores[rows - 1]
}

const COSINE_REFINEMENT_BATCH_ROWS: usize = 2_048;

fn cosine_refinement_batches(row_count: usize) -> impl Iterator<Item = Range<usize>> {
    (0..row_count)
        .step_by(COSINE_REFINEMENT_BATCH_ROWS)
        .map(move |start| start..(start + COSINE_REFINEMENT_BATCH_ROWS).min(row_count))
}

#[inline(always)]
fn combine_refinement_decoded(
    metric: Metric,
    candidates: &mut QuantizedCandidates,
    candidate_range: Range<usize>,
    kernel_scores: &[f32],
    decoded_scales: &[f32],
    decoded_constants: &[f32],
    sigma_factor: f32,
    bias_factor: f32,
) {
    let rows = candidate_range.len();
    debug_assert_eq!(kernel_scores.len(), rows);
    debug_assert_eq!(decoded_scales.len(), rows);
    let scores = &mut candidates.scores[candidate_range.clone()];
    let bias_corrections = &mut candidates.bias_corrections[candidate_range.clone()];
    match metric {
        Metric::L2 => {
            debug_assert_eq!(decoded_constants.len(), rows);
            for (((score, bias_correction), &kernel_score), (&scale, &constant)) in scores
                .iter_mut()
                .zip(bias_corrections.iter_mut())
                .zip(kernel_scores)
                .zip(decoded_scales.iter().zip(decoded_constants))
            {
                *score = (2.0 * scale).mul_add(kernel_score, *score - 2.0 * constant);
                *bias_correction = scale * bias_factor;
            }
        }
        Metric::Dot | Metric::Cosine => {
            for (((score, bias_correction), &kernel_score), &scale) in scores
                .iter_mut()
                .zip(bias_corrections.iter_mut())
                .zip(kernel_scores)
                .zip(decoded_scales)
            {
                *score = scale.mul_add(kernel_score, *score);
                *bias_correction = scale * bias_factor;
            }
        }
    }
    for (sigma, &scale) in candidates.sigmas[candidate_range]
        .iter_mut()
        .zip(decoded_scales)
    {
        *sigma = scale * sigma_factor;
    }
}

/// Score one storage-ordered survivor batch from independently planned SoA
/// pins. Packed codes are never gathered: every kernel call borrows one code
/// range and indexes rows relative to that pin. Scale/constant plans may have
/// different block geometry and fill reusable f32 scratch independently.
#[inline(always)]
fn apply_refinement_planned(
    query: &QuantizedQueryCtx,
    layer_idx: usize,
    metric: Metric,
    layer: &QuantizedLayerReader,
    candidates: &mut QuantizedCandidates,
    candidate_range: Range<usize>,
    available_rows: Range<usize>,
    query_norm: f32,
    kernel_scores: &mut Vec<f32>,
    decoded_scales: &mut Vec<f32>,
    decoded_constants: &mut Vec<f32>,
    read_ranges: &mut Vec<Range<usize>>,
    block_scratch: &mut Vec<(usize, usize)>,
    row_offsets: &mut Vec<usize>,
) -> crate::Result<()> {
    let rows = candidate_range.len();
    debug_assert!(rows > 0);
    debug_assert!(candidates.rows[candidate_range.clone()]
        .windows(2)
        .all(|pair| pair[0] < pair[1]));
    kernel_scores.resize(rows, 0.0);
    decoded_scales.resize(rows, 0.0);
    if metric == Metric::L2 {
        decoded_constants.resize(rows, 0.0);
    }

    // Codes: one indexed kernel entry per independently pinnable range.
    layer.plan_code_reads(
        available_rows.clone(),
        &candidates.rows[candidate_range.clone()],
        read_ranges,
        block_scratch,
    );
    let mut selected_start = 0usize;
    for read_range in read_ranges.iter().cloned() {
        let mut selected_end = selected_start;
        while selected_end < rows
            && candidates.rows[candidate_range.start + selected_end] < read_range.end
        {
            debug_assert!(
                candidates.rows[candidate_range.start + selected_end] >= read_range.start
            );
            selected_end += 1;
        }
        row_offsets.clear();
        row_offsets.extend(
            candidates.rows
                [candidate_range.start + selected_start..candidate_range.start + selected_end]
                .iter()
                .map(|&row| row - read_range.start),
        );
        let codes = layer.read_codes(read_range)?;
        query.score_layer_batch_unscaled_indexed(
            layer_idx,
            codes.as_slice(),
            layer.code_stride(),
            row_offsets,
            &mut kernel_scores[selected_start..selected_end],
        );
        selected_start = selected_end;
    }
    debug_assert_eq!(selected_start, rows);

    // Pass A: decode scales according to the scale slot's own page plan.
    layer.plan_scale_reads(
        available_rows.clone(),
        &candidates.rows[candidate_range.clone()],
        read_ranges,
        block_scratch,
    );
    selected_start = 0;
    for read_range in read_ranges.iter().cloned() {
        let scales = layer.read_scales(read_range.clone())?;
        while selected_start < rows
            && candidates.rows[candidate_range.start + selected_start] < read_range.end
        {
            let row = candidates.rows[candidate_range.start + selected_start];
            debug_assert!(row >= read_range.start);
            let offset = (row - read_range.start) * std::mem::size_of::<u16>();
            let bits = scales[offset] as u16 | (scales[offset + 1] as u16) << 8;
            decoded_scales[selected_start] = f16_to_f32(bits);
            selected_start += 1;
        }
    }
    debug_assert_eq!(selected_start, rows);

    if metric == Metric::L2 {
        layer.plan_constant_reads(
            available_rows,
            &candidates.rows[candidate_range.clone()],
            read_ranges,
            block_scratch,
        );
        selected_start = 0;
        for read_range in read_ranges.iter().cloned() {
            let constants = layer.read_constants(read_range.clone())?;
            while selected_start < rows
                && candidates.rows[candidate_range.start + selected_start] < read_range.end
            {
                let row = candidates.rows[candidate_range.start + selected_start];
                debug_assert!(row >= read_range.start);
                let offset = (row - read_range.start) * std::mem::size_of::<f32>();
                let bits = constants[offset] as u32
                    | (constants[offset + 1] as u32) << 8
                    | (constants[offset + 2] as u32) << 16
                    | (constants[offset + 3] as u32) << 24;
                decoded_constants[selected_start] = f32::from_bits(bits);
                selected_start += 1;
            }
        }
        debug_assert_eq!(selected_start, rows);
    }

    // Pass B: f32-only estimate/bias assembly across the complete logical
    // batch, including cross-cluster cosine batches.
    let sigma_factor = query.layer_sigma_factor(layer_idx, query_norm);
    let bias_factor = query.layer_bias_factor(layer_idx, query_norm);
    let decoded_constants = if metric == Metric::L2 {
        &decoded_constants[..rows]
    } else {
        &[]
    };
    combine_refinement_decoded(
        metric,
        candidates,
        candidate_range,
        &kernel_scores[..rows],
        &decoded_scales[..rows],
        decoded_constants,
        sigma_factor,
        bias_factor,
    );
    Ok(())
}

#[cfg(test)]
#[inline]
fn quantized_layer_constant(metric: Metric, stored: f32) -> f32 {
    if metric == Metric::L2 {
        stored
    } else {
        0.0
    }
}

struct QuantizedScanCtx {
    candidates: QuantizedCandidates,
    boundary_scratch: Vec<QuantizedCandidate>,
    /// One query-residual norm per admitted cluster, not one copy per row.
    cluster_query_norms: Vec<f32>,
    dedup_docs: bool,
    best_by_doc: HashMap<DocId, usize>,
    best_docs: Vec<DocId>,
    /// Best distinct-document rows seen so far, descending by estimate. This
    /// is updated once per completed cluster for the admission bound; the
    /// exact boundary still uses `kth_scratch` once after plane 1.
    bound_top: Vec<usize>,
    /// Reused cluster-local top-k, with the worst retained row at the root.
    /// Once full, a scored row performs one admission comparison rather than
    /// rescanning the retained rows for their minimum.
    local_top: BinaryHeap<LocalTopEntry>,
    local_top_n: usize,
    cluster_start: Option<usize>,
    /// Reused O(k) merge scratch. Replica matching happens here, once per
    /// cluster, never in the dense plane-1 append loop.
    bound_merge: Vec<usize>,
    kth_scratch: Vec<usize>,
    work_spent: WorkUnits,
}

impl QuantizedScanCtx {
    fn new(max_doc: DocId, candidate_capacity: usize, dedup_docs: bool) -> Self {
        let distinct_capacity = candidate_capacity.min(max_doc as usize);
        let dedup_capacity = if dedup_docs { distinct_capacity } else { 0 };
        Self {
            candidates: QuantizedCandidates::with_capacity(candidate_capacity),
            boundary_scratch: Vec::with_capacity(candidate_capacity),
            cluster_query_norms: Vec::new(),
            dedup_docs,
            best_by_doc: HashMap::with_capacity(dedup_capacity),
            best_docs: Vec::with_capacity(dedup_capacity),
            bound_top: Vec::new(),
            local_top: BinaryHeap::new(),
            local_top_n: 0,
            cluster_start: None,
            bound_merge: Vec::new(),
            kth_scratch: Vec::with_capacity(distinct_capacity),
            work_spent: WorkUnits::ZERO,
        }
    }

    fn begin_cluster(&mut self, top_n: usize) {
        debug_assert!(self.cluster_start.is_none());
        debug_assert!(self.local_top.is_empty());
        self.local_top.reserve(top_n);
        self.bound_top.reserve(top_n);
        self.bound_merge.reserve(top_n.saturating_mul(2));
        self.local_top_n = top_n;
        self.cluster_start = Some(self.candidates.len());
    }

    /// Append one row to the SoA buffers and maintain only the cluster-local
    /// running minimum. Document deduplication is deliberately absent here;
    /// it is performed at the cluster merge and κ boundaries.
    #[inline]
    fn push(&mut self, row: usize, doc: DocId, score: f32, bias_correction: f32, sigma: f32) {
        let index = self.candidates.len();
        self.candidates
            .push(row, doc, score, bias_correction, sigma);
        if self.local_top_n == 0 {
            return;
        }
        let entry = LocalTopEntry {
            estimate: self.candidates.estimate(index),
            row,
            index,
        };
        if self.local_top.len() < self.local_top_n {
            self.local_top.push(entry);
            return;
        }
        let mut worst = self
            .local_top
            .peek_mut()
            .expect("a full local top has a root");
        if entry.precedes(&worst) {
            *worst = entry;
        }
    }

    fn set_cluster_query_norm(&mut self, cluster: usize, query_norm: f32) {
        if self.cluster_query_norms.len() <= cluster {
            self.cluster_query_norms.resize(cluster + 1, f32::NAN);
        }
        self.cluster_query_norms[cluster] = query_norm;
    }

    fn cluster_query_norm(&self, cluster: usize) -> f32 {
        let query_norm = self.cluster_query_norms[cluster];
        debug_assert!(query_norm.is_finite());
        query_norm
    }

    /// Merge one completed cluster into the running admission top-k. The
    /// scorer writes the cluster densely first; this method derives a local-k
    /// and performs one small global merge, restoring the boundary design's
    /// separation between scoring and kth maintenance.
    fn finish_cluster_bound(&mut self) {
        let cluster_start = self
            .cluster_start
            .take()
            .expect("finish_cluster_bound requires begin_cluster");
        let top_n = std::mem::take(&mut self.local_top_n);
        if top_n == 0 || cluster_start >= self.candidates.len() {
            self.local_top.clear();
            return;
        }

        self.bound_merge.clear();
        self.bound_merge.extend_from_slice(&self.bound_top);
        while let Some(entry) = self.local_top.pop() {
            let index = entry.index;
            let doc = self.candidates.docs[index];
            if self.dedup_docs {
                if let Some(slot) = self
                    .bound_merge
                    .iter_mut()
                    .find(|slot| self.candidates.docs[**slot] == doc)
                {
                    if candidate_precedes(&self.candidates, index, *slot) {
                        *slot = index;
                    }
                    continue;
                }
            }
            self.bound_merge.push(index);
        }
        self.bound_merge
            .sort_unstable_by(|&a, &b| candidate_order(&self.candidates, a, b));
        self.bound_merge.truncate(top_n);
        std::mem::swap(&mut self.bound_top, &mut self.bound_merge);
    }

    fn running_pessimistic_kth(&self, top_n: usize, kappa: f32) -> Option<f32> {
        if top_n == 0 || self.bound_top.len() < top_n {
            return None;
        }
        let index = *self.bound_top.last().unwrap();
        Some(self.candidates.estimate(index) - kappa * self.candidates.sigmas[index])
    }

    fn rebuild_best_by_doc(&mut self) {
        if !self.dedup_docs {
            return;
        }
        self.best_by_doc.clear();
        self.best_docs.clear();
        for index in 0..self.candidates.len() {
            let doc = self.candidates.docs[index];
            let previous_best = self.best_by_doc.get(&doc).copied();
            if previous_best.is_none() {
                self.best_docs.push(doc);
            }
            if previous_best
                .is_none_or(|previous| candidate_precedes(&self.candidates, index, previous))
            {
                self.best_by_doc.insert(doc, index);
            }
        }
    }

    /// The distinct-document k-th estimate widened pessimistically by its σ.
    fn pessimistic_kth(&mut self, top_n: usize, kappa: f32) -> Option<f32> {
        debug_assert!(
            self.candidates
                .scores
                .iter()
                .zip(&self.candidates.bias_corrections)
                .zip(&self.candidates.sigmas)
                .all(|((&score, &bias), &sigma)| (score + bias).is_finite() && sigma.is_finite()),
            "quantized boundary inputs must be finite"
        );
        // Replica winner selection is boundary work. Plane-1 append and
        // refinement retain every membership without per-row hashing.
        self.rebuild_best_by_doc();
        let distinct_len = if self.dedup_docs {
            self.best_docs.len()
        } else {
            self.candidates.len()
        };
        if top_n == 0 || distinct_len < top_n {
            return None;
        }
        self.kth_scratch.clear();
        if self.dedup_docs {
            self.kth_scratch
                .extend(self.best_docs.iter().map(|doc| self.best_by_doc[doc]));
        } else {
            self.kth_scratch.extend(0..self.candidates.len());
        }
        let (_, selected, _) = self
            .kth_scratch
            .select_nth_unstable_by(top_n - 1, |&a, &b| candidate_order(&self.candidates, a, b));
        let index = *selected;
        Some(self.candidates.estimate(index) - kappa * self.candidates.sigmas[index])
    }

    fn band(&mut self, top_n: usize, kappa: f32) {
        let pessimistic_kth = self.pessimistic_kth(top_n, kappa);
        self.boundary_scratch.clear();
        for index in 0..self.candidates.len() {
            if pessimistic_kth.is_none_or(|kth| {
                self.candidates.estimate(index) + kappa * self.candidates.sigmas[index] >= kth
            }) {
                self.boundary_scratch
                    .push(self.candidates.materialize(index));
            }
        }
        self.boundary_scratch
            .sort_unstable_by_key(|candidate| candidate.row);
        self.candidates
            .replace_with_boundary_survivors(&self.boundary_scratch);
        self.best_by_doc.clear();
        self.best_docs.clear();
    }
}

/// A heap entry ordered with the worst retained candidate first. The total
/// key is the same estimate-descending, row-ascending key used at κ
/// boundaries, so equal estimates select the same σ-bearing row everywhere.
#[derive(Clone, Copy, Debug)]
struct LocalTopEntry {
    estimate: f32,
    row: usize,
    index: usize,
}

impl LocalTopEntry {
    #[inline]
    fn precedes(&self, other: &Self) -> bool {
        self.estimate
            .total_cmp(&other.estimate)
            .reverse()
            .then(self.row.cmp(&other.row))
            .is_lt()
    }
}

impl PartialEq for LocalTopEntry {
    fn eq(&self, other: &Self) -> bool {
        self.estimate.to_bits() == other.estimate.to_bits()
            && self.row == other.row
            && self.index == other.index
    }
}

impl Eq for LocalTopEntry {}

impl PartialOrd for LocalTopEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LocalTopEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .estimate
            .total_cmp(&self.estimate)
            .then(self.row.cmp(&other.row))
            .then(self.index.cmp(&other.index))
    }
}

#[cfg(test)]
fn distinct_candidate_docs(candidates: &QuantizedCandidates) -> Vec<DocId> {
    let mut docs = candidates.docs.clone();
    docs.sort_unstable();
    docs.dedup();
    docs
}

#[inline]
fn candidate_order(candidates: &QuantizedCandidates, a: usize, b: usize) -> std::cmp::Ordering {
    candidates
        .estimate(b)
        .total_cmp(&candidates.estimate(a))
        .then(candidates.rows[a].cmp(&candidates.rows[b]))
}

#[inline]
fn candidate_precedes(candidates: &QuantizedCandidates, a: usize, b: usize) -> bool {
    candidate_order(candidates, a, b).is_lt()
}

impl<T: VectorElement> VectorBackend<T> {
    #[allow(clippy::too_many_arguments)]
    fn quantized_top_n<K, CTail>(
        &self,
        index: &IvfIndex,
        query: &QuantizedQueryCtx,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        if top_n == 0 || segment_reader.max_doc() == 0 || index.num_clusters() == 0 {
            return Ok(Vec::new());
        }
        let init_start = Instant::now();
        let init_stage = enter_vector_stage(Stage::ScanInit);
        let max_doc = segment_reader.max_doc();
        let filter = build_filter_bitset(weight, segment_reader, max_doc)?;
        if filter.len() == 0 {
            return Ok(Vec::new());
        }
        let filter_is_all = filter.len() == max_doc as usize;
        let scan_levels = query.active_layers();
        let alive = segment_reader.alive_bitset();
        let quantized = self
            .reader
            .quantization()
            .expect("quantized query requires quantized slots");
        stats.segment_rows = Some(index.num_rows());
        stats.segment_clusters = Some(index.num_clusters());
        stats.start_layer(0, query.index.biases()[0], query.index.calibrations()[0]);
        let (work_budget, n_avg, x) = self
            .adaptive
            .resolved_work_budget(index.num_clusters(), index.num_docs())?;
        let pricing = UnitPricing {
            budget: WorkUnits::new(work_budget),
            open: WorkUnits::new(x),
            row: WorkUnits::new((1.0 - x) / n_avg),
        };
        let mut routing_ws = Workspace::new();
        let candidate_capacity =
            ((pricing.budget.get() / pricing.row.get()).ceil() as usize).min(index.num_rows());
        let mut scan = QuantizedScanCtx::new(
            max_doc,
            candidate_capacity,
            index.num_rows() != index.num_docs(),
        );
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut postings_row = 0usize;
        let mut postings_skipped = 0usize;
        let bounds = index.bounds();
        let metric = query.index.config.metric;
        let q_norm = norm_squared_wide(query.query()).sqrt() as f32;
        let mut bounds_skips = 0u32;
        let mut armed_probe = None;
        let mut cluster_rows = Vec::new();
        let mut kernel_scores = Vec::new();
        let mut decoded_scales = Vec::new();
        let mut decoded_constants = Vec::new();
        let mut decoded_residual_norms = Vec::new();
        let mut bias_scores = Vec::new();
        let mut sigma_scores = Vec::new();
        let mut survivor_rows = Vec::new();
        let mut survivor_read_ranges = Vec::new();
        let mut survivor_block_scratch = Vec::new();
        drop(init_stage);
        stats.scan_init_ns = stats
            .scan_init_ns
            .saturating_add(init_start.elapsed().as_nanos() as u64);

        let routing_start = Instant::now();
        let mut ranked = {
            let _routing_stage = enter_vector_stage(Stage::Routing);
            index.rank_clusters(&mut routing_ws, query.query())
        };
        let mut routing_ns = routing_start.elapsed().as_nanos() as u64;
        let routing_before_scan = routing_ns;
        let scan_start = Instant::now();
        let layer0_stage = enter_vector_stage(Stage::LayerScan(0));

        loop {
            let routing_start = Instant::now();
            let next = {
                let _routing_stage = enter_vector_stage(Stage::Routing);
                ranked.next()
            };
            routing_ns += routing_start.elapsed().as_nanos() as u64;
            let Some(Candidate { sim, node }) = next else {
                break;
            };
            if scan.work_spent >= pricing.budget {
                stats.termination = ProbeTermination::Ceiling;
                break;
            }
            let cluster = node as usize;
            let query_bound =
                scan.running_pessimistic_kth(top_n, 2.0)
                    .map_or(QueryBound::Filling, |score| QueryBound::Armed {
                        t: to_bound_space(metric, score),
                    });
            if armed_probe.is_none() && matches!(query_bound, QueryBound::Armed { .. }) {
                armed_probe = Some((postings_row + postings_skipped).saturating_sub(1) as u32);
            }
            let verdict = bounds_verdict(query_bound, || {
                let QueryBound::Armed { t } = query_bound else {
                    return f32::INFINITY;
                };
                let r = bounds.ball_r(cluster);
                match metric {
                    Metric::L2 | Metric::Cosine => {
                        margin_ball_ball(t, r, to_bound_space(metric, sim.score()))
                    }
                    Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t),
                }
            });
            if verdict == Verdict::Skip {
                scan.work_spent += pricing.open;
                bounds_skips += 1;
                continue;
            }
            scan.work_spent += pricing.open;
            let score_query_norm = query.score_query_norm(sim.score());
            let rows = index.cluster_range(cluster);
            cluster_rows.clear();
            let all_rows_eligible = filter_is_all && alive.is_none();
            if all_rows_eligible {
                visited += rows.len();
            } else {
                for row in rows.clone() {
                    visited += 1;
                    let doc = self.reader.doc_id_at(row);
                    if !filter.contains(doc) {
                        pruned_filter += 1;
                        continue;
                    }
                    if alive.is_some_and(|alive| !alive.is_alive(doc)) {
                        pruned_dead += 1;
                        continue;
                    }
                    cluster_rows.push((row, doc));
                }
            }
            if !all_rows_eligible && cluster_rows.is_empty() {
                postings_skipped += 1;
                continue;
            }
            let needs_l2_rows = metric == Metric::L2;
            let layer = quantized.layers()[0].read_batch(rows.clone(), needs_l2_rows)?;
            let residual_norms = if needs_l2_rows {
                quantized.read_residual_norm_batch(rows.clone())?
            } else {
                None
            };
            let batch_rows = layer.scales().len() / std::mem::size_of::<u16>();
            kernel_scores.resize(batch_rows, 0.0);
            query.score_layer_batch_unscaled(
                0,
                layer.codes(),
                layer.code_stride(),
                &mut kernel_scores[..batch_rows],
            );

            // Pass A: isolate all byte/integer decoding from the floating
            // combine loops and reuse the same scratch across clusters.
            decode_scales(layer.scales(), &mut decoded_scales);
            if needs_l2_rows {
                decode_f32s(layer.constants(), &mut decoded_constants);
                let residual_norms = residual_norms.as_ref().ok_or_else(|| {
                    TantivyError::DataCorruption(DataCorruption::comment_only(
                        "quantized L2 field is missing residual-norm slot 14",
                    ))
                })?;
                decode_f32s(residual_norms.as_bytes(), &mut decoded_residual_norms);
            }
            bias_scores.resize(batch_rows, 0.0);
            sigma_scores.resize(batch_rows, 0.0);
            let bias_factor = query.layer_bias_factor(0, score_query_norm);
            let cluster_score = sim.score();
            let sigma_factor = query.layer_sigma_factor(0, score_query_norm);
            combine_initial_decoded(
                metric,
                &mut kernel_scores,
                &mut bias_scores,
                &mut sigma_scores,
                &decoded_scales,
                &decoded_constants,
                &decoded_residual_norms,
                cluster_score,
                bias_factor,
                sigma_factor,
            );
            scan.set_cluster_query_norm(cluster, score_query_norm);
            scan.begin_cluster(top_n);
            if all_rows_eligible {
                for row in rows.clone() {
                    let local = row - rows.start;
                    scan.push(
                        row,
                        self.reader.doc_id_at(row),
                        kernel_scores[local],
                        bias_scores[local],
                        sigma_scores[local],
                    );
                }
            } else {
                for &(row, doc) in &cluster_rows {
                    let local = row - rows.start;
                    scan.push(
                        row,
                        doc,
                        kernel_scores[local],
                        bias_scores[local],
                        sigma_scores[local],
                    );
                }
            }
            scan.finish_cluster_bound();
            let cluster_scored = if all_rows_eligible {
                rows.len()
            } else {
                cluster_rows.len()
            };
            scan.work_spent += pricing.row * cluster_scored as f64;
            postings_row += 1;
        }
        stats.record_routing(ranked.metrics());
        stats.vectors_visited += visited;
        stats.pruned_filter += pruned_filter;
        stats.pruned_dead += pruned_dead;
        stats.postings_row += postings_row;
        stats.postings_skipped += postings_skipped;
        stats.candidates_scored += scan.candidates.len();
        let layer0_scored = scan.candidates.len();
        stats.bounds_skips += bounds_skips;
        stats.record_bound_armed(armed_probe);
        stats.work_charged += scan.work_spent.to_f32();
        drop(layer0_stage);
        let scan_ns = scan_start.elapsed().as_nanos() as u64;
        stats.routing_ns += routing_ns;
        stats.record_layer_scan(
            0,
            layer0_scored,
            scan_ns.saturating_sub(routing_ns.saturating_sub(routing_before_scan)),
        );

        #[cfg(test)]
        {
            stats.quantized_trace.scored_docs = distinct_candidate_docs(&scan.candidates);
        }

        let boundary_start = Instant::now();
        let boundary_stage = enter_vector_stage(Stage::Boundary(0));
        scan.band(top_n, 2.0);
        #[cfg(test)]
        stats
            .quantized_trace
            .boundary_docs
            .push(distinct_candidate_docs(&scan.candidates));
        drop(boundary_stage);
        stats.record_boundary(
            0,
            scan.candidates.len(),
            boundary_start.elapsed().as_nanos() as u64,
        );
        for layer_idx in 1..scan_levels {
            stats.start_layer(
                layer_idx,
                query.index.biases()[layer_idx],
                query.index.calibrations()[layer_idx],
            );
            let layer_start = Instant::now();
            let layer_stage = enter_vector_stage(Stage::LayerScan(layer_idx as u8));
            let layer = &quantized.layers()[layer_idx];
            let layer_scored = scan.candidates.len();
            if metric == Metric::Cosine {
                // Cosine has one query norm and no per-cluster refinement
                // term. Accumulate storage-ordered survivors across cluster
                // boundaries up to the explicitly authorized batch maximum.
                let query_norm = query.score_query_norm(0.0);
                for candidate_range in cosine_refinement_batches(scan.candidates.len()) {
                    let candidate_start = candidate_range.start;
                    let candidate_end = candidate_range.end;
                    let first_row = scan.candidates.rows[candidate_start];
                    let last_row = scan.candidates.rows[candidate_end - 1];
                    apply_refinement_planned(
                        query,
                        layer_idx,
                        metric,
                        layer,
                        &mut scan.candidates,
                        candidate_range,
                        first_row..last_row + 1,
                        query_norm,
                        &mut kernel_scores,
                        &mut decoded_scales,
                        &mut decoded_constants,
                        &mut survivor_read_ranges,
                        &mut survivor_block_scratch,
                        &mut survivor_rows,
                    )?;
                }
            } else {
                // L2 retains cluster-local query-residual norms. Dot keeps the
                // same cluster-local scheduling; only cosine is authorized to
                // cross cluster boundaries in v1.
                let mut candidate_start = 0;
                let mut cluster = 0;
                while candidate_start < scan.candidates.len() {
                    let first_row = scan.candidates.rows[candidate_start];
                    while cluster < index.num_clusters()
                        && index.cluster_range(cluster).end <= first_row
                    {
                        cluster += 1;
                    }
                    if cluster == index.num_clusters() {
                        return Err(TantivyError::DataCorruption(DataCorruption::comment_only(
                            format!(
                                "quantized survivor row {first_row} is outside IVF cluster ranges"
                            ),
                        )));
                    }
                    let cluster_rows = index.cluster_range(cluster);
                    if first_row < cluster_rows.start {
                        return Err(TantivyError::DataCorruption(DataCorruption::comment_only(
                            format!(
                                "quantized survivor row {first_row} precedes cluster {cluster} \
                                 range {cluster_rows:?}"
                            ),
                        )));
                    }
                    let mut candidate_end = candidate_start + 1;
                    while candidate_end < scan.candidates.len()
                        && scan.candidates.rows[candidate_end] < cluster_rows.end
                    {
                        candidate_end += 1;
                    }
                    let query_norm = scan.cluster_query_norm(cluster);
                    apply_refinement_planned(
                        query,
                        layer_idx,
                        metric,
                        layer,
                        &mut scan.candidates,
                        candidate_start..candidate_end,
                        cluster_rows,
                        query_norm,
                        &mut kernel_scores,
                        &mut decoded_scales,
                        &mut decoded_constants,
                        &mut survivor_read_ranges,
                        &mut survivor_block_scratch,
                        &mut survivor_rows,
                    )?;
                    candidate_start = candidate_end;
                }
            }
            drop(layer_stage);
            stats.record_layer_scan(
                layer_idx,
                layer_scored,
                layer_start.elapsed().as_nanos() as u64,
            );
            let final_sign = query.index.specs[layer_idx].bits == 1;
            let boundary_start = Instant::now();
            let kappa = if final_sign { 2.0 } else { 4.0 };
            let boundary_stage = enter_vector_stage(Stage::Boundary(layer_idx as u8));
            scan.band(top_n, kappa);
            #[cfg(test)]
            stats
                .quantized_trace
                .boundary_docs
                .push(distinct_candidate_docs(&scan.candidates));
            drop(boundary_stage);
            stats.record_boundary(
                layer_idx,
                scan.candidates.len(),
                boundary_start.elapsed().as_nanos() as u64,
            );
        }

        let rerank_fetch_start = Instant::now();
        let rerank_fetch_stage = enter_vector_stage(Stage::RerankFetch);
        // Refinement can change a replicated document's best membership. The
        // final physical dedup is performed once, immediately before fetch.
        scan.rebuild_best_by_doc();
        let rerank_capacity = if scan.dedup_docs {
            scan.best_docs.len()
        } else {
            scan.candidates.len()
        };
        let mut rerank = Vec::with_capacity(rerank_capacity);
        if scan.dedup_docs {
            // Replicated memberships survive independently through the
            // bands; `best_docs` names the best row per distinct document.
            for &doc in &scan.best_docs {
                let index = scan.best_by_doc[&doc];
                rerank.push((scan.candidates.rows[index], doc));
            }
        } else {
            rerank.extend(
                scan.candidates
                    .rows
                    .iter()
                    .copied()
                    .zip(scan.candidates.docs.iter().copied()),
            );
        }
        rerank.sort_unstable_by_key(|&(row, _)| row);
        #[cfg(test)]
        {
            stats.quantized_trace.rerank_docs = rerank.iter().map(|&(_, doc)| doc).collect();
            stats.quantized_trace.rerank_docs.sort_unstable();
            stats.quantized_trace.rerank_docs.dedup();
        }
        drop(rerank_fetch_stage);
        stats.rerank_fetch_ns += rerank_fetch_start.elapsed().as_nanos() as u64;
        stats.rerank_rows += rerank.len();

        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        for (row, doc) in rerank {
            let fetch_start = Instant::now();
            let bytes = {
                let _rerank_fetch_stage = enter_vector_stage(Stage::RerankFetch);
                self.reader.vector_bytes_for_row(row)?
            };
            stats.rerank_fetch_ns += fetch_start.elapsed().as_nanos() as u64;
            let score_start = Instant::now();
            {
                let _rerank_score_stage = enter_vector_stage(Stage::RerankScore);
                let score = self.query.score_doc_bytes(&bytes);
                if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                    topn.push_unordered(key, doc);
                }
            }
            stats.rerank_score_ns += score_start.elapsed().as_nanos() as u64;
            stats.exact_rows_read += 1;
        }
        let segment_ord = self.segment_ord;
        let assembly_start = Instant::now();
        let _assembly_stage = enter_vector_stage(Stage::ResultAssembly);
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| {
                (
                    candidate.sort_key,
                    DocAddress::new(segment_ord, candidate.doc),
                )
            })
            .collect();
        stats.result_assembly_ns = Some(assembly_start.elapsed().as_nanos() as u64);
        Ok(hits)
    }

    /// Top-N by IVF probe. Fills `stats` with this segment's probe-loop
    /// counters.
    #[allow(clippy::too_many_arguments)]
    fn approximate_top_n<K, CTail>(
        &self,
        index: &IvfIndex,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        if top_n == 0 {
            return Ok(Vec::new());
        }
        let init_start = Instant::now();
        let init_stage = enter_vector_stage(Stage::ScanInit);
        let max_doc = segment_reader.max_doc();
        if max_doc == 0 {
            return Ok(Vec::new());
        }

        let filter = build_filter_bitset(weight, segment_reader, max_doc)?;
        if filter.len() == 0 {
            return Ok(Vec::new());
        }
        let alive = segment_reader.alive_bitset();

        let num_centroids = index.num_clusters();
        if num_centroids == 0 {
            return Ok(Vec::new());
        }
        // Capacity counts native docs as WRITTEN (deleted rows still
        // charge on first touch - see `WorkModel::for_searcher`), and the
        // open share x is derived from the index's own n_avg at query
        // init - see `open_share`.
        let (work_budget, n_avg, x) = self
            .adaptive
            .resolved_work_budget(num_centroids, index.num_docs())?;
        debug_assert!(n_avg > 0.0);
        let pricing = UnitPricing {
            budget: WorkUnits::new(work_budget),
            open: WorkUnits::new(x),
            row: WorkUnits::new((1.0 - x) / n_avg),
        };

        // Phase 1: rank the clusters to probe, lazily — the scan below pulls
        // ranked clusters on demand, so routing cost is paid only as far as
        // probing actually reaches. The filter-effective budget can pull far
        // past its nominal cluster count on a selective filter (each passed-
        // over cluster streams few unseen rows), and lazy routing keeps that
        // cheap.
        // Routing operates in `f32` (centroid rows are `f32` today), so the
        // query is widened losslessly per element.
        let query_f32: Vec<f32> = self.query.query().iter().map(|e| e.to_f32()).collect();
        let mut routing_ws = Workspace::new();
        stats.segment_rows = Some(index.num_rows());
        stats.segment_clusters = Some(index.num_clusters());
        drop(init_stage);
        stats.scan_init_ns = stats
            .scan_init_ns
            .saturating_add(init_start.elapsed().as_nanos() as u64);
        let routing_start = Instant::now();
        let mut ranked = {
            let _routing_stage = enter_vector_stage(Stage::Routing);
            index.rank_clusters(&mut routing_ws, &query_f32)
        };
        let mut routing_ns = routing_start.elapsed().as_nanos() as u64;
        let routing_before_scan = routing_ns;

        let scan_start = Instant::now();
        let exact_scan_stage = enter_vector_stage(Stage::ExactScan);
        let topn = self.scan_clusters(
            index,
            &mut ranked,
            pricing,
            &filter,
            max_doc,
            alive,
            top_n,
            tie_break,
            tie_comparator,
            &query_f32,
            stats,
            &mut routing_ns,
        )?;
        drop(exact_scan_stage);
        stats.routing_ns += routing_ns;
        stats.exact_scan_ns = Some(
            scan_start
                .elapsed()
                .as_nanos()
                .saturating_sub(u128::from(routing_ns.saturating_sub(routing_before_scan)))
                as u64,
        );

        // The routing cost is only known once the scan stops pulling.
        stats.record_routing(ranked.metrics());

        let segment_ord = self.segment_ord;
        let assembly_start = Instant::now();
        let _assembly_stage = enter_vector_stage(Stage::ResultAssembly);
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect();
        stats.result_assembly_ns = Some(assembly_start.elapsed().as_nanos() as u64);
        Ok(hits)
    }

    /// Phase 2: the probe loop. Each ranked cluster first passes the
    /// bounds verdict — armed, the cluster's stored bound is collided
    /// with the query bound and a strict-negative margin skips it for
    /// the open share, without touching its rows. A probed cluster is
    /// then gated per row — [`Self::collect_cluster_survivors`] runs
    /// `filter → alive → seen` off the pinned id-map with no posting
    /// bytes in hand — and only the survivors' bytes are fetched, one
    /// stride-sized read per surviving row. Cluster-order arrival of
    /// survivors forbids the ascending-doc shortcut in `push`; use
    /// `push_unordered`.
    ///
    /// Note on `NaturalComparator` (vs the `TopNComputer::new` default):
    /// vector similarity is "higher = better", so we want top-N *largest*
    /// scores. The default `new()` wires `ReverseComparator`, which keeps
    /// top-N *smallest* — correct for ascending-distance metrics but inverted
    /// for our convention.
    ///
    /// `ranked` is pulled lazily, one cluster per probe: with graph routing,
    /// pulling past a converged batch resumes the beam search, so routing
    /// work interleaves with (and is bounded by) probing.
    ///
    /// `#[inline(never)]` so it forms its own flamegraph frame carrying its
    /// `score_doc_bytes` cost.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn scan_clusters<K, CTail>(
        &self,
        index: &IvfIndex,
        ranked: &mut impl Iterator<Item = Candidate>,
        pricing: UnitPricing,
        filter: &BitSet,
        max_doc: DocId,
        alive: Option<&AliveBitSet>,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        routing_query: &[f32],
        stats: &mut ProbeStats,
        routing_ns: &mut u64,
    ) -> crate::Result<TieBreakHeap<K, CTail>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        // `candidates` is the cumulative scored count that drives the gate; the
        // prune counters accumulate into locals and fold into `ProbeStats` once
        // after the loop, keeping the hot per-doc path free of indirection.
        let mut candidates = 0usize;
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut pruned_seen = 0usize;
        let mut postings_row = 0usize;
        let mut postings_skipped = 0usize;
        let mut bounds_skips = 0u32;
        let mut termination = ProbeTermination::Exhausted;
        // P2: the query bound, maintained at cluster boundaries. The
        // bound-space conversion runs on kth improvement only, inside the
        // tracker.
        let metric = self.query.metric();
        let mut bound_tracker = QueryBoundTracker::new();
        // P4: `||q||` for the dot margin's Cauchy-Schwarz term; once per
        // segment-query.
        let q_norm = norm_squared_wide(self.query.query()).sqrt() as f32;
        let bounds = index.bounds();
        // Replication can place the same doc in several probed clusters; dedup
        // by doc id so a vector is scored at most once.
        let mut seen = BitSet::with_max_value(max_doc);
        // The probed cluster's gate survivors; allocated once, reused
        // across clusters.
        let mut survivors: Vec<Survivor> = Vec::new();
        // f64 accumulation in the loop; f32 only at the telemetry fold.
        let mut work_spent = WorkUnits::ZERO;
        let work_budget = pricing.budget;

        loop {
            let routing_start = Instant::now();
            let next = {
                let _routing_stage = enter_vector_stage(Stage::Routing);
                ranked.next()
            };
            *routing_ns += routing_start.elapsed().as_nanos() as u64;
            let Some(Candidate { sim, node: cluster }) = next else {
                break;
            };
            // Boundary rule: open iff remaining > 0. The tripping pull
            // proves another ranked cluster existed, keeping `Ceiling`
            // distinct from `Exhausted`.
            if work_spent >= work_budget {
                termination = ProbeTermination::Ceiling;
                break;
            }
            let cluster = cluster as usize;

            // P5: the bounds verdict. The bound is consumed only through
            // `Armed` (the heap holds k results) — enforced by the enum;
            // Filling probes, and SATURATED probes arithmetically (+inf
            // margin). The margin closure runs on armed clusters only.
            let qb = bound_tracker.bound();
            let verdict = bounds_verdict(qb, || {
                let QueryBound::Armed { t } = qb else {
                    // `bounds_verdict` never calls the margin while
                    // Filling; +inf keeps even that impossibility
                    // fail-open.
                    return f32::INFINITY;
                };
                // The separation IS the routing key the ranked stream
                // already computed: `to_bound_space` maps the similarity
                // key into the metric's distance space for L2/cosine
                // (the heap-key and routing-key spaces coincide), and
                // dot consumes the raw `q . c` key directly.
                #[cfg(debug_assertions)]
                {
                    // Precondition of every margin: the stream key is the
                    // EXACT centroid similarity — an approximate key
                    // makes a skip unsound.
                    let stride = self.reader.options().bytes_per_vector();
                    let centroid_bytes = index.centroid_bytes().expect("readable centroid rows");
                    let exact = metric.similarity_bytes::<f32>(
                        routing_query,
                        &centroid_bytes[cluster * stride..(cluster + 1) * stride],
                    );
                    debug_assert_eq!(
                        sim, exact,
                        "routing stream key must be the exact centroid similarity"
                    );
                }
                let r = bounds.ball_r(cluster);
                match metric {
                    Metric::L2 | Metric::Cosine => {
                        margin_ball_ball(t, r, to_bound_space(metric, sim.score()))
                    }
                    Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t),
                }
            });
            if let Verdict::Skip = verdict {
                // A skip charges the open share: skips are search work,
                // and free skips break the work identity (validated to
                // +-0.03% in benchmarks). No row work is spent.
                work_spent += pricing.open;
                bounds_skips += 1;
                continue;
            }

            // Event-wise charging, part 1: the open.
            work_spent += pricing.open;

            let rows = index.cluster_range(cluster);

            // Pre-pass: gate off the pinned id-map BEFORE fetching any
            // posting bytes, so only rows that will be scored are read.
            let (v, pf, pd, ps, scored_rows) =
                self.collect_cluster_survivors(rows, filter, alive, &mut seen, &mut survivors);
            visited += v;
            pruned_filter += pf;
            pruned_dead += pd;
            pruned_seen += ps;

            // Event-wise charging, part 2: the rows that survive the
            // pre-pass — exactly the rows fetched and scored below.
            // Rejected and deduped rows charge nothing.
            work_spent += pricing.row * scored_rows as f64;

            if survivors.is_empty() {
                postings_skipped += 1;
            } else {
                postings_row += 1;
                // One stride-sized read per survivor — the unit the
                // pg-backed `Directory` serves zero-copy (see
                // `vector_bytes_for_row`).
                for &Survivor { row, doc } in &survivors {
                    let vbytes = self.reader.vector_bytes_for_row(row)?;
                    let score = self.query.score_doc_bytes(&vbytes);
                    if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                        topn.push_unordered(key, doc);
                    }
                }
            }
            candidates += survivors.len();

            // P2: fold the exact kth into the bound at the cluster
            // boundary. `kth_best` is O(buffer) and force-truncates —
            // results and every counter above are unaffected (truncation
            // only drops already-lost entries and tightens the push
            // threshold, which prunes pushes, not scoring).
            let probe_idx = (postings_row + postings_skipped - 1) as u32;
            let peek = HeapPeek::from_kth(topn.kth_best().map(|(score, _tie)| score));
            bound_tracker.observe(metric, peek, probe_idx);
        }
        // The armed index exists exactly when the bound armed.
        debug_assert!(
            bound_tracker.armed_at_probe().is_some()
                == matches!(bound_tracker.bound(), QueryBound::Armed { .. })
        );

        stats.vectors_visited += visited;
        stats.pruned_filter += pruned_filter;
        stats.pruned_dead += pruned_dead;
        stats.pruned_seen += pruned_seen;
        stats.postings_row += postings_row;
        stats.postings_skipped += postings_skipped;
        stats.candidates_scored += candidates;
        stats.bounds_skips += bounds_skips;
        stats.record_bound_armed(bound_tracker.armed_at_probe());
        stats.termination = termination;
        stats.work_charged += work_spent.to_f32();

        Ok(topn)
    }

    /// Phase 2 pre-pass: run one cluster's rows through the
    /// `filter → alive → seen` gate — off the pinned id-map alone, with no
    /// posting bytes fetched — collecting into `survivors` (cleared first)
    /// the rows to score.
    /// Returns `(visited, pruned_filter, pruned_dead, pruned_seen,
    /// scored_rows)` - the last being the survivor count, which is the
    /// work-unit row-charge basis and equals the partition identity's
    /// `scored` term.
    /// `#[inline(never)]` so per-cluster gate cost forms its own frame,
    /// while the per-row loop stays inlined inside it.
    #[inline(never)]
    fn collect_cluster_survivors(
        &self,
        rows: Range<usize>,
        filter: &BitSet,
        alive: Option<&AliveBitSet>,
        seen: &mut BitSet,
        survivors: &mut Vec<Survivor>,
    ) -> (usize, usize, usize, usize, usize) {
        survivors.clear();
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut pruned_seen = 0usize;
        let mut scored_rows = 0usize;
        for row in rows {
            let doc = self.reader.doc_id_at(row);
            visited += 1;
            // Dedup FIRST, marking on first encounter whatever the later
            // verdicts say, so a replica's second copy is never
            // re-checked (it counts as `pruned_seen`, not the original
            // verdict's bucket). The charge basis is `scored_rows` alone:
            // the partition identity
            // `visited == filter + dead + seen + scored` holds, and only
            // its `scored` term ever charges budget.
            if seen.contains(doc) {
                pruned_seen += 1;
                continue;
            }
            seen.insert(doc);
            if !filter.contains(doc) {
                pruned_filter += 1;
                continue;
            }
            if let Some(bs) = alive {
                if !bs.is_alive(doc) {
                    pruned_dead += 1;
                    continue;
                }
            }
            survivors.push(Survivor { row, doc });
            scored_rows += 1;
        }
        (
            visited,
            pruned_filter,
            pruned_dead,
            pruned_seen,
            scored_rows,
        )
    }
}

/// Drain the filter `DocSet` into a dense BitSet for O(1) random membership
/// testing per cluster doc. The BitSet allocates `max_doc / 8` bytes regardless
/// of filter selectivity — inherent to IVF needing membership tests on
/// out-of-order doc ids. `#[inline(never)]` so it forms its own flamegraph
/// frame; at low selectivity over a large segment this drain is real cost
/// otherwise hidden in the search entry.
#[inline(never)]
fn build_filter_bitset(
    weight: &dyn Weight,
    segment_reader: &SegmentReader,
    max_doc: DocId,
) -> crate::Result<BitSet> {
    let mut filter = BitSet::with_max_value(max_doc);
    weight.for_each_no_score(segment_reader, &mut |docs| {
        for &doc in docs {
            filter.insert(doc);
        }
    })?;
    Ok(filter)
}

#[cfg(test)]
mod tests {
    // ============================================================
    // IVF `top_n` test gate.
    //
    // Built on top of `crate::vector::tests::TestVectorIndex` (the
    // shared fixture) where the geometry fits — the 100-doc grid +
    // selectivity-based labels covers oracle / filter / delete /
    // overflow / zero-K. The handful of tests that need crafted point
    // geometry (the trap case + the result-level candidate-floor
    // demonstration) build a tiny IVF index inline via `build_inline_ivf`
    // and an `InlineClusterer` that's compatible with the batched
    // IvfClusterer trait.
    // ============================================================
    use std::cmp::Ordering;

    use super::*;
    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{
        AllQuery, BitSetDocSet, ConstScorer, EnableScoring, Explanation, Query, Scorer, TermQuery,
    };
    use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
    use crate::vector::tests::{exhaustive_params, TestVectorIndex};
    use crate::vector::{
        IvfCentroids, IvfClusterer, IvfMatrix, IvfMergeSettings, IvfTrainingVectors, IvfVectors,
        NeighborhoodGraphSearchMetrics, SearchTerminationReason, VectorClusterStats, VectorDType,
        VectorInfo, VectorOptions, VectorStorageFormat,
    };
    use crate::{Index, IndexWriter, TantivyDocument};

    const FIXTURE_NUM_DOCS: usize = 100;
    /// Number of centroids the shared fixture uses by default (the
    /// 3×3 `grid2d::centroids()` grid). Used by tests that need an
    /// "exhaustive" probe ceiling.
    const DEFAULT_NUM_CENTROIDS: usize = 9;

    /// Run the full collector path with the given filter and adaptive
    /// params. Returns the global top-K (already merged across
    /// segments) in descending-score / (seg_ord, doc_id) order — the
    /// same order `ground_truth::top_k` uses, so equality checks are
    /// well-defined.
    fn search(
        index: &Index,
        field: Field,
        filter: &dyn Query,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let collector = TopDocs::with_limit(k)
            .order_by_similarity(field, query)
            .with_adaptive_params(params);
        Ok(index
            .reader()?
            .searcher()
            .search(filter, &collector)?
            .results)
    }

    /// Probe-stat helper: run `VectorBackend::top_n` against
    /// the first segment of `index` and return (hits, stats).
    /// The contracts are per-segment, so collecting from segment 0 is
    /// what each assertion is talking about.
    fn run_top_n(
        index: &Index,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let weight = AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let quantized_queries = QuantizedQueryCache::default();
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            &quantized_queries,
            params,
            usize::MAX,
        )?;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_some(),
            "expected IVF storage"
        );
        backend.top_n(weight.as_ref(), segment_reader, k)
    }

    // ---- Inline IVF builder for crafted-geometry tests ----
    //
    // The shared fixture's `grid2d::vectors` lays 100 deterministic
    // points around a 3×3 grid; it doesn't expose a per-doc-vector
    // override. The trap-case and result-level candidate-floor tests
    // need points at specific coordinates, so they build a small IVF
    // index inline via the helper below.

    struct InlineClusterer {
        centroids: Vec<[f32; 2]>,
    }

    impl IvfClusterer for InlineClusterer {
        fn training_sample_ratio(&self) -> f32 {
            1.0
        }
        fn train(
            &self,
            options: &VectorOptions,
            _vectors: IvfTrainingVectors,
        ) -> crate::Result<IvfCentroids> {
            assert_eq!(options.dim(), 2);
            let num_centroids = self.centroids.len();
            Ok(IvfCentroids::F32(IvfMatrix {
                values: self
                    .centroids
                    .iter()
                    .flat_map(|c| c.iter().copied())
                    .collect(),
                rows: num_centroids,
                dims: 2,
            }))
        }
        fn assign(
            &self,
            options: &VectorOptions,
            vectors: IvfVectors<'_>,
            centroids: &IvfCentroids,
        ) -> crate::Result<Vec<u32>> {
            assert_eq!(options.dim(), 2);
            let IvfVectors::F32(vectors) = vectors;
            let IvfCentroids::F32(centroids) = centroids;
            Ok(vectors
                .matrix
                .values
                .chunks_exact(2)
                .map(|v| {
                    let mut best = 0u32;
                    let mut best_d2 = f32::INFINITY;
                    for (i, c) in centroids.values.chunks_exact(2).enumerate() {
                        let dx = v[0] - c[0];
                        let dy = v[1] - c[1];
                        let d2 = dx * dx + dy * dy;
                        if d2 < best_d2 {
                            best = i as u32;
                            best_d2 = d2;
                        }
                    }
                    best
                })
                .collect())
        }
    }

    /// Build a single-IVF-segment index with the supplied centroids and
    /// labelled docs. Splits docs across two commits so `merge_ivf`
    /// has ≥ 2 source segments to consume. Returns the index plus the
    /// `(embedding, label)` field handles.
    fn build_inline_ivf(
        metric: Metric,
        centroids: &[[f32; 2]],
        docs: &[(&str, [f32; 2])],
    ) -> crate::Result<(Index, Field, Field)> {
        assert!(docs.len() >= 2, "need ≥ 2 docs for ≥ 2 source segments");
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();

        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(InlineClusterer {
                centroids: centroids.to_vec(),
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        let mid = docs.len() / 2;
        for chunk in [&docs[..mid.max(1)], &docs[mid.max(1)..]] {
            for (label, v) in chunk {
                let mut doc = TantivyDocument::new();
                doc.add_text(label_field, label);
                doc.add_vector(embed_field, v.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }
        let segment_ids: Vec<_> = index.searchable_segment_ids()?.into_iter().collect();
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;
        Ok((index, embed_field, label_field))
    }

    /// Decode a stored little-endian `[f32; 2]` row.
    fn decode_2d(bytes: &[u8]) -> [f32; 2] {
        [
            f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        ]
    }

    /// L2-nearest centroid with first-wins tie-break on strict `<` — the
    /// same rule `InlineClusterer::assign` uses for the primary.
    fn nearest_centroid(p: [f32; 2], centroids: &[[f32; 2]]) -> usize {
        let mut best = 0;
        let mut best_d2 = f32::INFINITY;
        for (i, c) in centroids.iter().enumerate() {
            let dx = p[0] - c[0];
            let dy = p[1] - c[1];
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = i;
            }
        }
        best
    }

    /// Docs per centroid in the replication fixture.
    const REPLICATION_N_PER: usize = 6;

    /// Six well-separated centroids (3×2 grid, gap 10) and one label per
    /// doc. Docs sit tightly around their centroid (offsets ≤ 0.05
    /// against the grid gap of 10 — see [`replication_docs`]) so the
    /// primary and the next-nearest replica ranking are unambiguous.
    fn replication_fixture() -> (Vec<[f32; 2]>, Vec<String>) {
        let centroids = vec![
            [0.0f32, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 10.0],
        ];
        let labels = (0..centroids.len() * REPLICATION_N_PER)
            .map(|i| format!("d{i}"))
            .collect();
        (centroids, labels)
    }

    /// The replication fixture's docs: `REPLICATION_N_PER` per centroid,
    /// at offset `(i % REPLICATION_N_PER) * 0.01` along both axes.
    fn replication_docs<'a>(
        centroids: &[[f32; 2]],
        labels: &'a [String],
    ) -> Vec<(&'a str, [f32; 2])> {
        (0..labels.len())
            .map(|i| {
                let c = centroids[i / REPLICATION_N_PER];
                let off = (i % REPLICATION_N_PER) as f32 * 0.01;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect()
    }

    /// Merging flat segments with deletes past the clustering threshold: rows
    /// written for since-deleted docs still count toward the sources'
    /// `count()` (tombstones don't rewrite `.vec`), so the alive-doc merge
    /// iteration legitimately comes up short of `vector_count`. The merge
    /// must tolerate that, and the resulting IVF segment must hold — and
    /// count — the alive docs only.
    #[test]
    fn merge_flat_segments_with_deletes_into_ivf() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();

        // Same shape as `build_inline_ivf`, but the two flat source segments
        // stay unmerged so the deletes land BEFORE the clustering merge.
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(InlineClusterer {
                centroids: centroids.clone(),
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let mid = n / 2;
        for chunk in [&docs[..mid], &docs[mid..]] {
            for (label, v) in chunk {
                let mut doc = TantivyDocument::new();
                doc.add_text(label_field, label);
                doc.add_vector(embed_field, v.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }

        // Tombstone docs in BOTH flat sources (d0/d7 in the first commit,
        // d35 in the second), then merge everything into one IVF segment.
        let deleted = ["d0", "d7", "d35"];
        for label in deleted {
            writer.delete_term(Term::from_field_text(label_field, label));
        }
        writer.commit()?;
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        let alive = n - deleted.len();
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1, "one merged segment");
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        let info = vec_reader.info().expect("vector info");
        assert_eq!(info.format, VectorStorageFormat::Ivf, "merge must cluster");
        assert_eq!(info.num_vectors, alive, "deleted docs must not be counted");
        assert_eq!(vec_reader.num_vectors(), alive);

        // Every alive doc comes back exactly once; no deleted label survives.
        let hits = search(
            &index,
            embed_field,
            &AllQuery,
            vec![10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?;
        assert_eq!(hits.len(), alive, "exhaustive top-N must return alive docs");
        let mut seen_labels = std::collections::HashSet::new();
        for (_, addr) in &hits {
            let label = stored_label_at(&index, label_field, *addr)?;
            assert!(
                !deleted.contains(&label.as_str()),
                "deleted doc {label} surfaced in results"
            );
            assert!(seen_labels.insert(label), "duplicate doc in results");
        }
        Ok(())
    }

    /// Merging past the clustering threshold when every doc carrying a
    /// vector for ONE field is deleted, while another field keeps live
    /// vectors. The sources still report `vector_count > 0` for the emptied
    /// field (tombstones don't rewrite `.vec`), so it takes the training
    /// path, collects nothing — and used to `continue` without writing the
    /// field's `.vec`/`.centroids` slots. The live field still wrote, so the
    /// composites existed but the emptied field's slots were missing:
    /// `count()`, `open_column()` and `vector_info()` all failed with
    /// InternalError. The merge must instead write the same empty slots as
    /// the no-vectors-at-all fast path.
    #[test]
    fn merge_deleting_every_doc_of_one_field_writes_empty_ivf() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();

        let mut sb = Schema::builder();
        let doomed_field = sb.add_vector_field(
            "embedding_doomed",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let kept_field = sb.add_vector_field(
            "embedding_kept",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(InlineClusterer {
                centroids: centroids.clone(),
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Even docs carry the doomed field, odd docs the kept one, split
        // across two flat commits so BOTH sources hold doomed vectors.
        let mid = n / 2;
        for (i, (label, v)) in docs.iter().enumerate() {
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, label);
            let field = if i % 2 == 0 { doomed_field } else { kept_field };
            doc.add_vector(field, v.as_slice());
            writer.add_document(doc)?;
            if i + 1 == mid {
                writer.commit()?;
            }
        }
        writer.commit()?;

        // Tombstone every doomed-field doc, then merge everything into one
        // IVF segment.
        for (i, (label, _)) in docs.iter().enumerate() {
            if i % 2 == 0 {
                writer.delete_term(Term::from_field_text(label_field, label));
            }
        }
        writer.commit()?;
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1, "one merged segment");
        let segment_reader = &searcher.segment_readers()[0];

        // The emptied field reads back as a zeroed IVF field — not an error.
        let vec_reader = segment_reader.vector_index(doomed_field)?;
        assert_eq!(vec_reader.num_vectors(), 0);
        let info = vec_reader.info().expect("vector info");
        assert_eq!(
            info,
            VectorInfo {
                format: VectorStorageFormat::Ivf,
                num_vectors: 0,
                num_centroids: Some(0),
                cluster_stats: Some(VectorClusterStats {
                    min_cluster_size: 0,
                    max_cluster_size: 0,
                    avg_cluster_size: 0.0,
                    empty_clusters: 0,
                }),
            },
        );
        let ivf = vec_reader.index().expect("expected IVF storage");
        assert!(vec_reader.is_empty(), "no rows in the emptied field");
        assert_eq!(ivf.num_rows(), 0);
        assert_eq!(ivf.num_clusters(), 0);

        // The live field is untouched: every alive doc is counted and found.
        let kept_count = n / 2;
        assert_eq!(
            segment_reader.vector_index(kept_field)?.num_vectors(),
            kept_count
        );
        let hits = search(
            &index,
            kept_field,
            &AllQuery,
            vec![10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?;
        assert_eq!(hits.len(), kept_count, "kept field returns alive docs");
        Ok(())
    }

    /// Captures `paradedb::ivf_build` log records so a test can read back the
    /// timings line the merge emits.
    struct CaptureLogger;
    static CAPTURED_IVF_BUILD: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    impl log::Log for CaptureLogger {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.target() == "paradedb::ivf_build"
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                CAPTURED_IVF_BUILD
                    .lock()
                    .unwrap()
                    .push(format!("{}", r.args()));
            }
        }
        fn flush(&self) {}
    }
    static CAPTURE_LOGGER: CaptureLogger = CaptureLogger;

    /// The merge emits one parseable `ivf_build timings_ms ...` line per
    /// field. Builds a larger index so the phase timings are measurable,
    /// captures the line, and prints it (run with `--nocapture`) so we can
    /// see where build time goes.
    #[test]
    fn ivf_build_emits_timings_log() -> crate::Result<()> {
        let _ = log::set_logger(&CAPTURE_LOGGER);
        log::set_max_level(log::LevelFilter::Info);

        // 200 centroids on a 20×10 grid; ~5000 docs clustered around them.
        let mut centroids: Vec<[f32; 2]> = Vec::new();
        for x in 0..20 {
            for y in 0..10 {
                centroids.push([x as f32 * 10.0, y as f32 * 10.0]);
            }
        }
        let n_per = 25usize;
        let labels: Vec<String> = (0..centroids.len() * n_per)
            .map(|i| format!("d{i}"))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
            .map(|i| {
                let c = centroids[i / n_per];
                let off = (i % n_per) as f32 * 0.05;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect();

        let before = CAPTURED_IVF_BUILD.lock().unwrap().len();
        let _ = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let lines: Vec<String> = CAPTURED_IVF_BUILD.lock().unwrap()[before..].to_vec();
        let line = lines
            .iter()
            .find(|l| l.contains("ivf_build timings_ms") && l.contains("centroids=200"))
            .expect("expected an ivf_build timings line for the 200-centroid build");
        assert!(line.contains("train="));
        assert!(line.contains("posting_write="));
        eprintln!("IVF_BUILD_SAMPLE {line}");
        Ok(())
    }

    // ---- IVF top_n correctness tests ----

    /// Exhaustive probing on a multi-segment IVF index built by the
    /// shared fixture must match the brute-force oracle. Sweep over
    /// several queries and K values to cover ranking + drain edges.
    #[test]
    fn ivf_top_n_brute_force_oracle_l2() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
        for query in [[0.5_f32, 0.5], [9.5, 9.5], [5.0, 0.0], [3.7, 11.2]] {
            for k in [1usize, 3, 6, 10] {
                let expected = index.ground_truth(query, k)?;
                let actual = search(
                    &index.index,
                    index.embedding_field(),
                    &AllQuery,
                    query.to_vec(),
                    k,
                    params.clone(),
                )?;
                assert_eq!(actual, expected, "L2 exhaustive query={query:?} k={k}");
            }
        }
        Ok(())
    }

    /// Same exhaustive correctness, confirming the metric threads
    /// through generically.
    #[test]
    fn ivf_top_n_brute_force_oracle_cosine() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::Cosine)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
        for query in [[1.0_f32, 0.0], [0.0, 1.0], [0.7, 0.3]] {
            for k in [1usize, 3, 6] {
                let expected = index.ground_truth(query, k)?;
                let actual = search(
                    &index.index,
                    index.embedding_field(),
                    &AllQuery,
                    query.to_vec(),
                    k,
                    params.clone(),
                )?;
                assert_eq!(actual, expected, "Cosine exhaustive query={query:?} k={k}");
            }
        }
        Ok(())
    }

    /// Exhaustive-probe correctness for Dot. EXHAUSTIVE-PROBE ONLY by
    /// design: Dot isn't a metric (no triangle inequality), so the IVF
    /// cluster-locality assumption is heuristic for unnormalized dot
    /// and can break on high-magnitude vectors in a far cluster.
    /// Adaptive Dot recall is a benchmark question, deferred. This
    /// test confirms only that `Metric::Dot` threads through the
    /// backend's full top_n loop and matches brute force when every
    /// cluster is visited.
    #[test]
    fn ivf_top_n_brute_force_oracle_dot() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::Dot)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
        for query in [[1.0_f32, 0.0], [2.0, 0.0], [0.5, -0.5]] {
            for k in [1usize, 3, 6] {
                let expected = index.ground_truth(query, k)?;
                let actual = search(
                    &index.index,
                    index.embedding_field(),
                    &AllQuery,
                    query.to_vec(),
                    k,
                    params.clone(),
                )?;
                assert_eq!(actual, expected, "Dot exhaustive query={query:?} k={k}");
            }
        }
        Ok(())
    }

    /// The trap: query closest to centroid A, true NN in cluster B.
    /// Adaptive probing finds it; a 1-cluster probe ceiling must miss. Setup
    /// assertions confirm the geometry is genuinely a trap before
    /// the behavioral check — a slightly-off geometry could trivialize
    /// the test. INLINE because the shared fixture's 100-doc grid
    /// doesn't permit a single misplaced trap doc.
    #[test]
    fn ivf_top_n_trap_case() -> crate::Result<()> {
        let centroids = vec![[0.0_f32, 0.0], [10.0, 10.0]];
        // Two A-side docs far from the [1,1] query; a B-side trap
        // doc at [5, 5.01] just over the perpendicular bisector
        // (x+y=10) so it lands in cluster 1 yet is much closer to
        // the query than any A-side doc.
        let docs = [
            ("far_a", [0.0_f32, -10.0]),
            ("far_a", [-10.0, 0.0]),
            ("trap_b", [5.0, 5.01]),
            ("anchor_b", [10.0, 10.0]),
        ];
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let query = [1.0_f32, 1.0];

        // Setup assertions.
        //
        // (i) The trap doc is genuinely the true top-1 — without
        // this, "miss" and "find" would be indistinguishable.
        let oracle = ground_truth_top_k(&index, embed_field, Metric::L2, &query, 1)?;
        let trap_doc = stored_label_at(&index, label_field, oracle[0].1)?;
        assert_eq!(trap_doc, "trap_b", "true NN must be the trap doc");

        // (ii) Query's nearest centroid is A (the one at the origin).
        //
        // With the inline IVF building exactly one segment, segment 0
        // holds both centroids. We don't need to open the column
        // directly — the geometry says distance to A = √2 ≈ 1.41,
        // distance to B = √162 ≈ 12.73, so A wins decisively.

        // Behavioral check 1: a probe ceiling of 1 misses the trap.
        let one_probe = AdaptiveProbeParams {
            max_probe_fraction: 0.5,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let hits1 = search(&index, embed_field, &AllQuery, query.to_vec(), 1, one_probe)?;
        assert_eq!(hits1.len(), 1);
        assert_ne!(
            stored_label_at(&index, label_field, hits1[0].1)?,
            "trap_b",
            "a 1-cluster probe ceiling should miss the trap (probes only cluster A)",
        );

        // Behavioral check 2: exhaustive probing finds it.
        let hits2 = search(
            &index,
            embed_field,
            &AllQuery,
            query.to_vec(),
            1,
            exhaustive_params(2),
        )?;
        assert_eq!(hits2.len(), 1);
        assert_eq!(
            stored_label_at(&index, label_field, hits2[0].1)?,
            "trap_b",
            "exhaustive probing should find the trap doc",
        );
        Ok(())
    }

    /// Filter selectivity: only docs in the filter set surface, and
    /// the result equals the oracle restricted to that set. Uses the
    /// shared fixture's `.selectivities(..)` to drop a "selectivity_0.1"
    /// label on the first 10 of 100 docs.
    #[test]
    fn ivf_top_n_filter_selectivity() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .selectivities(&[0.1])
            .build()?;
        let filter = TermQuery::new(
            Term::from_field_text(index.label_field(), "selectivity_0.1"),
            IndexRecordOption::Basic,
        );
        let query = [0.5_f32, 0.5];
        let k = 5;
        // Oracle restricted to the filter set: brute force across the
        // whole index, then keep only the docs that carry the label.
        let filter_set = collect_filter_doc_set(&index.index, &filter)?;
        let mut restricted = ground_truth_top_k(
            &index.index,
            index.embedding_field(),
            Metric::L2,
            &query,
            FIXTURE_NUM_DOCS,
        )?;
        restricted.retain(|(_, addr)| filter_set.contains(addr));
        restricted.truncate(k);

        let actual = search(
            &index.index,
            index.embedding_field(),
            &filter,
            query.to_vec(),
            k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(actual, restricted);
        for (_, addr) in &actual {
            assert!(filter_set.contains(addr), "hit outside filter: {addr:?}");
        }
        Ok(())
    }

    /// Empty filter returns empty results, no panic.
    #[test]
    fn ivf_top_n_empty_filter() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        // No doc carries "absent" — the term query yields an empty
        // DocSet.
        let empty = TermQuery::new(
            Term::from_field_text(index.label_field(), "absent"),
            IndexRecordOption::Basic,
        );
        let hits = search(
            &index.index,
            index.embedding_field(),
            &empty,
            vec![0.0_f32, 0.0],
            5,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert!(hits.is_empty());
        Ok(())
    }

    /// K > total candidates: returns all docs in descending order,
    /// no panic.
    #[test]
    fn ivf_top_n_k_exceeds_candidates() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let query = [0.0_f32, 0.0];
        let big_k = FIXTURE_NUM_DOCS + 50;
        let expected = index.ground_truth(query, big_k)?;
        let actual = search(
            &index.index,
            index.embedding_field(),
            &AllQuery,
            query.to_vec(),
            big_k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(actual.len(), FIXTURE_NUM_DOCS);
        assert_eq!(actual, expected);
        Ok(())
    }

    /// Deletes: a doc marked deleted must never appear, even if it
    /// would otherwise rank top-K. Confirms the IVF backend's separate
    /// alive-check (the filter bitmap doesn't carry delete info).
    #[test]
    fn ivf_top_n_respects_deletes() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .selectivities(&[0.1])
            .build()?;
        // Delete every doc carrying the 0.1-selectivity label — the
        // 10 docs nearest to the grid's origin centroid by
        // construction (they're inserted first).
        {
            let mut writer: IndexWriter = index.index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            writer.delete_term(Term::from_field_text(
                index.label_field(),
                "selectivity_0.1",
            ));
            writer.commit()?;
        }

        // Oracle restricted to the surviving docs.
        let query = [0.0_f32, 0.0];
        let searcher = index.index.reader()?.searcher();
        let mut alive_addrs = std::collections::HashSet::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            let alive = segment_reader.alive_bitset();
            for doc in 0..segment_reader.max_doc() {
                let is_alive = alive.is_none_or(|bs| bs.is_alive(doc));
                if is_alive {
                    alive_addrs.insert(DocAddress::new(seg_ord as u32, doc));
                }
            }
        }
        assert!(
            alive_addrs.len() < FIXTURE_NUM_DOCS,
            "delete didn't remove anything (alive={})",
            alive_addrs.len(),
        );
        let k = 10;
        let mut expected = ground_truth_top_k(
            &index.index,
            index.embedding_field(),
            Metric::L2,
            &query,
            FIXTURE_NUM_DOCS,
        )?;
        expected.retain(|(_, addr)| alive_addrs.contains(addr));
        expected.truncate(k);

        let actual = search(
            &index.index,
            index.embedding_field(),
            &AllQuery,
            query.to_vec(),
            k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(actual, expected);
        for (_, addr) in &actual {
            assert!(
                alive_addrs.contains(addr),
                "deleted doc {addr:?} surfaced in results",
            );
        }
        Ok(())
    }
    /// `top_n == 0` returns empty without touching the column. The
    /// collector layer rejects `TopDocs::with_limit(0)` before it
    /// reaches the backend, so this test calls the backend directly
    /// via the instrumented seam — the short-circuit lives in
    /// `approximate_top_n`.
    #[test]
    fn ivf_top_n_zero_returns_empty() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let (hits, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            0,
            AdaptiveProbeParams::default(),
        )?;
        assert!(hits.is_empty());
        // Short-circuit fires before the probe loop, so no clusters
        // visited and no candidates scored.
        assert_eq!(stats.clusters_probed(), 0);
        assert_eq!(stats.candidates_scored, 0);
        Ok(())
    }

    /// Smoke for the instrumented seam: every centroid is probed under
    /// exhaustive params, and candidates_scored ≤ total docs in the
    /// inspected segment. Exhaustive params on a 9-centroid segment
    /// visit all 9.
    #[test]
    fn ivf_top_n_collects_probe_stats() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        // k = 64 exceeds any segment's doc count, so the query bound
        // never arms and the bounds gate skips nothing - every cluster
        // is probed and every counter equality below is exact.
        let (_, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            64,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(stats.clusters_probed(), DEFAULT_NUM_CENTROIDS);
        // The first segment has docs distributed across all 9 clusters;
        // candidates_scored equals the segment's doc count under
        // exhaustive probe + AllQuery.
        let segment_doc_count =
            index.index.reader()?.searcher().segment_readers()[0].max_doc() as usize;
        assert_eq!(stats.candidates_scored, segment_doc_count);

        // Counter invariant: every touched doc lands in exactly one bucket.
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
            "visited must equal filter+dead+seen+scored ({stats:?})"
        );
        // Navigation cost == the centroids ranked for this query.
        assert_eq!(stats.routing.visited_count, DEFAULT_NUM_CENTROIDS);
        // Exhaustive params (unclamped ceiling, unsatisfiable floor)
        // drain the ranked list.
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        Ok(())
    }

    /// A `max_probe_fraction` resolving below the cluster count forces the
    /// hard ceiling: the loop stops with `termination == Ceiling`, having
    /// probed exactly the cap, and the counter invariant still holds. Uses the
    /// deterministic `build_inline_ivf` fixture (fixed 6 centroids) so the
    /// cutoff is stable.
    #[test]
    fn ivf_probe_stats_termination_ceiling() -> crate::Result<()> {
        let centroids = [
            [0.0f32, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 10.0],
        ];
        let n_per = 6usize;
        let labels: Vec<String> = (0..centroids.len() * n_per)
            .map(|i| format!("d{i}"))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
            .map(|i| {
                let c = centroids[i / n_per];
                let off = (i % n_per) as f32 * 0.01;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

        // Cap 1 → ceiling at the first probe; an unsatisfiable survivor
        // floor keeps the gate from firing first.
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let (_, stats) = run_top_n(&index, embed_field, vec![10.0, 10.0], 3, params)?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        // Stopped at exactly the cap, short of the ranked list.
        assert_eq!(stats.clusters_probed(), 1);
        assert_eq!(stats.routing.visited_count, centroids.len());
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
            "visited must equal filter+dead+seen+scored ({stats:?})"
        );
        Ok(())
    }

    /// A single-centroid IVF merge skips the `.centroids` graph slot
    /// (nothing to route between), so the reader must take the linear
    /// fallback: rank the lone cluster without a graph and still return the
    /// exact top-K.
    #[test]
    fn ivf_single_centroid_routes_without_graph() -> crate::Result<()> {
        let centroids = [[0.0f32, 0.0]];
        let labels: Vec<String> = (0..5).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, [f32; 2])> = (0..5)
            .map(|i| (labels[i].as_str(), [i as f32 * 0.01, 0.0]))
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_some(),
            "expected IVF storage"
        );

        let query = [0.0f32, 0.0];
        let k = 3;
        let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, k)?;
        let (hits, stats) = run_top_n(
            &index,
            embed_field,
            query.to_vec(),
            k,
            AdaptiveProbeParams::default(),
        )?;
        assert_eq!(hits, expected, "linear fallback must match the oracle");
        assert_eq!(stats.clusters_probed(), 1, "one cluster, one probe");
        assert_eq!(stats.routing.visited_count, 1);
        Ok(())
    }

    /// When the probe ceiling is below the cluster count, cluster ranking
    /// routes via the persisted RNG instead of scanning every centroid. With
    /// 16 well-separated clusters and only 8 beam seeds, the router must
    /// still navigate to the true nearest cluster: the routed top-K equals
    /// the brute-force oracle, and the recorded navigation cost is the
    /// beam-visited count, not a full scan of the ranked list.
    #[test]
    fn ivf_routed_ranking_matches_oracle_on_separated_clusters() -> crate::Result<()> {
        // 4×4 grid of well-separated centroids (spacing 10), 4 docs each,
        // tightly packed around their centroid so each query's true top-K
        // lives entirely in one cluster.
        let side = 4usize;
        let centroids: Vec<[f32; 2]> = (0..side * side)
            .map(|i| [(i % side) as f32 * 10.0, (i / side) as f32 * 10.0])
            .collect();
        let n_per = 4usize;
        let labels: Vec<String> = (0..centroids.len() * n_per)
            .map(|i| format!("d{i}"))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
            .map(|i| {
                let c = centroids[i / n_per];
                let off = (i % n_per) as f32 * 0.01;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

        // The merged segment must carry the routing graph, and cap 2 (< 16
        // clusters) must engage it.
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        let ivf = vec_reader.index().expect("expected IVF segment");
        assert_eq!(ivf.num_clusters(), centroids.len());

        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let k = 3usize;
        for (ord, centroid) in centroids.iter().enumerate().step_by(3) {
            let query = [centroid[0] + 0.3, centroid[1] - 0.2];
            let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, k)?;
            let (hits, stats) = run_top_n(&index, embed_field, query.to_vec(), k, params.clone())?;
            assert_eq!(hits, expected, "routed top-{k} near centroid {ord}");
            assert!(
                stats.clusters_probed() <= 2,
                "cap 2 must bound the probes, got {}",
                stats.clusters_probed()
            );
            assert!(
                stats.routing.visited_count <= centroids.len(),
                "navigation cost is the beam-visited count"
            );
        }
        Ok(())
    }

    /// The raw per-cluster sizes from the reader's `cluster_sizes` must be
    /// exactly the un-collapsed array behind `info`'s aggregate cluster stats
    /// — the invariant `paradedb.ivf_cluster_sizes` relies on to reconcile
    /// with `paradedb.index_info`. Flat segments expose no sizes.
    #[test]
    fn ivf_cluster_sizes_match_vector_info() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let field = index.embedding_field();
        let searcher = index.index.reader()?.searcher();

        let mut segments_checked = 0;
        for segment_reader in searcher.segment_readers() {
            let vec_reader = segment_reader.vector_index(field)?;
            let sizes = vec_reader
                .cluster_sizes()
                .expect("ivf segment exposes cluster sizes");
            let info = vec_reader.info().expect("vector info");
            assert_eq!(info.format, VectorStorageFormat::Ivf);
            let stats = info.cluster_stats.expect("ivf cluster stats");

            // count == num_centroids, and every aggregate index_info reports is
            // reproducible from the raw array.
            assert_eq!(sizes.len(), info.num_centroids.expect("ivf centroids"));
            let sum: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
            let min = sizes.iter().copied().min().unwrap() as usize;
            let max = sizes.iter().copied().max().unwrap() as usize;
            let empty = sizes.iter().filter(|&&s| s == 0).count();
            let avg = sum as f64 / sizes.len() as f64;

            // Per-cluster sizes count posting rows (memberships)...
            let ivf = vec_reader.index().expect("expected IVF segment");
            assert_eq!(sum as usize, ivf.num_rows(), "sizes sum to rows");
            // ...and the shared fixture has no deletes, so the row total
            // coincides with the distinct-doc `num_vectors`.
            assert_eq!(sum as usize, info.num_vectors, "rows == docs");
            assert_eq!(min, stats.min_cluster_size, "min");
            assert_eq!(max, stats.max_cluster_size, "max");
            assert_eq!(empty, stats.empty_clusters, "empty");
            assert!(
                (avg - stats.avg_cluster_size).abs() < 1e-9,
                "avg {avg} vs {}",
                stats.avg_cluster_size
            );
            segments_checked += 1;
        }
        assert!(
            segments_checked > 0,
            "fixture must produce >= 1 IVF segment"
        );

        // Flat segments (no IVF data) yield None — the SRF emits no rows for them.
        let flat = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Flat)
            .build()?;
        let flat_field = flat.embedding_field();
        let flat_searcher = flat.index.reader()?.searcher();
        for segment_reader in flat_searcher.segment_readers() {
            assert!(
                segment_reader
                    .vector_index(flat_field)?
                    .cluster_sizes()
                    .is_none(),
                "flat segments expose no cluster sizes"
            );
        }
        Ok(())
    }

    // ============================================================
    // Adaptive-probing parameter contracts.
    //
    // The stop condition couples the three adaptive knobs. Each test
    // below holds the others permissive so one becomes the binding
    // constraint, then asserts the contract implied by the knob's
    // definition — exact only where the knob pins it (the absolute
    // ceiling), an inequality otherwise.
    // ============================================================

    // ============================================================
    // Work-unit budget properties. Unprimed (single-segment) runs use the
    // segment-local n_avg, under which units_seg is exactly C_seg - the
    // normalization identity, per segment. Every test here parks the
    // distance-ratio gate: K > N keeps the resolved floor
    // (top_n + overfetch_margin) unreachable, and a huge epsilon makes the
    // threshold vacuous - the stop point under test is the budget's alone.
    // ============================================================

    /// Full-budget params with the gate parked - the shared configuration
    /// for the budget properties.
    fn budget_only_params() -> AdaptiveProbeParams {
        AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 1,
            ..Default::default()
        }
    }

    /// An exhaustive scan charges `C*x + (1 - x)*N/n_avg = exactly C`
    /// units - capacity is the cluster count, whatever the size skew.
    #[test]
    fn unit_normalization_exact() -> crate::Result<()> {
        // Uneven sizes on purpose: [5, 2, 2, 1] docs across 4 clusters.
        let centroids = vec![[0.0_f32, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]];
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for (c, count) in [(0usize, 5usize), (1, 2), (2, 2), (3, 1)] {
            for i in 0..count {
                docs.push((
                    format!("d{c}_{i}"),
                    [centroids[c][0] + i as f32 * 0.01, 0.0],
                ));
            }
        }
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let (_, stats) = run_top_n(
            &index,
            embed_field,
            vec![0.0, 0.0],
            11,
            budget_only_params(),
        )?;
        let c = centroids.len() as f32;
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        assert!(
            (stats.work_charged - c).abs() <= 1e-6 * c,
            "an exhaustive scan must charge exactly C units: {stats:?}"
        );
        Ok(())
    }

    /// Skew charges proportionally: a 30-doc cluster consumes most of the
    /// budget a cluster-count budget would spread over five clusters, and
    /// the stop point shifts - hand-verified numbers. Also pins the
    /// overshoot bound: the final overrun is at most the last cluster's
    /// charge.
    #[test]
    fn imbalance_charges_proportionally() -> crate::Result<()> {
        // Sizes [30, 2, 2, 2, 2, 2]: C = 6, N = 40, n_avg = 20/3.
        // Per-index x = 1.64/(1.64 + 20/3) ~ 0.1975; row_charge =
        // (1 - x)*3/20 ~ 0.1204. units_seg = 6x + (1 - x)*40/(20/3) =
        // 1.185 + 4.815 = 6.0 (identity). The query sits on the big
        // centroid, so it opens first: charge 0.1975 + 30*0.1204 ~ 3.809.
        // Each small cluster charges 0.1975 + 2*0.1204 ~ 0.438. At
        // f = 0.8 (budget 4.8): big -> 3.809, small -> 4.247, small ->
        // 4.685, small -> 5.123 >= 4.8 at the next boundary, so 4 clusters
        // are probed. A cluster-count budget at f = 0.8 would probe
        // ceil(4.8) = 5.
        let centroids = vec![
            [0.0_f32, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
            [50.0, 0.0],
        ];
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for (c, count) in [(0usize, 30usize), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)] {
            for i in 0..count {
                docs.push((
                    format!("d{c}_{i}"),
                    [centroids[c][0] + i as f32 * 0.001, 0.0],
                ));
            }
        }
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.8,
            ..budget_only_params()
        };
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let (_, stats) = run_top_n(&index, embed_field, vec![0.0, 0.0], 41, params)?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling, "{stats:?}");
        assert_eq!(
            stats.clusters_probed(),
            4,
            "the big cluster eats the budget a count regime spreads over 5: {stats:?}"
        );
        assert!(
            (stats.work_charged - 5.123).abs() < 2e-3,
            "hand-computed spend: {stats:?}"
        );
        // Overshoot bound: the overrun is at most the last (small)
        // cluster's charge.
        let budget = 0.8f32 * 6.0;
        let last_charge = 0.1975 + 2.0 * 0.1204;
        assert!(
            stats.work_charged > budget && stats.work_charged - budget <= last_charge + 1e-4,
            "overshoot bounded by the last cluster's charge: {stats:?}"
        );
        Ok(())
    }

    /// A cluster's row charge tracks the rows it actually READ AND SCORED,
    /// not the rows it walked past. Same fixture, same opens, two runs:
    /// unfiltered (every row scored) and filtered to 3 of 22 docs. The
    /// filtered run must charge 19 row-shares less - the rows the filter
    /// rejected are never fetched, so they cost nothing. Under a
    /// first-seen basis the two runs would charge identically, since both
    /// walk all 22 rows and mark all 22 seen.
    #[test]
    fn filtered_rows_are_not_charged() -> crate::Result<()> {
        // Sizes [20, 2]: C = 2, N = 22, n_avg = 11.
        let centroids = vec![[0.0_f32, 0.0], [50.0, 0.0]];
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for (c, count) in [(0usize, 20usize), (1, 2)] {
            for i in 0..count {
                docs.push((
                    format!("d{c}_{i}"),
                    [centroids[c][0] + i as f32 * 0.001, 0.0],
                ));
            }
        }
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

        let n_avg = 22.0 / 2.0;
        let fixed = fixed_probe_cost_rows();
        let x = fixed / (fixed + n_avg);
        let row = (1.0 - x) / n_avg;

        // Unfiltered: every cluster opened, every row scored - the
        // identity's reference case, so exactly C = 2 units.
        let (_, full) = run_top_n(
            &index,
            embed_field,
            vec![0.0, 0.0],
            23,
            budget_only_params(),
        )?;
        assert_eq!(full.clusters_probed(), 2, "{full:?}");
        assert_eq!(full.candidates_scored, 22, "{full:?}");
        assert!(
            (full.work_charged as f64 - 2.0).abs() < 1e-5,
            "unfiltered exhaustive scan charges exactly C: {full:?}"
        );

        // Admit 3 docs of the big cluster; the other 19 rows are walked
        // and rejected, and cluster 1's 2 rows are walked and rejected.
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let admitted: Vec<DocId> = segment_reader
            .vector_index(embed_field)?
            .cluster_doc_ids(0)
            .expect("cluster 0 doc ids")
            .into_iter()
            .take(3)
            .collect();
        assert_eq!(admitted.len(), 3, "setup: need 3 admitted docs");
        let weight = FixedDocsWeight {
            max_doc: segment_reader.max_doc(),
            docs: admitted,
        };
        drop(searcher);

        let (_, filtered) = run_top_n_with_weight(
            &index,
            embed_field,
            vec![0.0, 0.0],
            23,
            budget_only_params(),
            &weight,
        )?;
        // Same opens, same rows walked - only the scored count differs.
        assert_eq!(filtered.clusters_probed(), 2, "{filtered:?}");
        assert_eq!(filtered.vectors_visited, 22, "{filtered:?}");
        assert_eq!(filtered.candidates_scored, 3, "{filtered:?}");

        let expected = 2.0 * x + 3.0 * row;
        assert!(
            (filtered.work_charged as f64 - expected).abs() < 1e-5,
            "charge must be 2 opens + 3 scored rows ({expected}): {filtered:?}"
        );
        // The 19 rejected rows cost exactly nothing.
        assert!(
            ((full.work_charged - filtered.work_charged) as f64 - 19.0 * row).abs() < 1e-5,
            "the filtered rows must account for the entire difference: full={full:?} \
             filtered={filtered:?}"
        );
        Ok(())
    }

    /// A budget below capacity binds, is attributed to the ceiling, and
    /// overshoots by at most one cluster's charge - the boundary rule on
    /// a real fixture rather than a hand-built one. The distance-ratio
    /// gate is parked (floor unreachable), so the stop point under test
    /// is the budget's alone.
    #[test]
    fn probe_stats_max_probe_fraction_ceiling() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.2,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let searcher = index.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(index.embedding_field())?;
        let ivf = vec_reader.index().expect("expected IVF storage");
        let (clusters, docs) = (ivf.num_clusters(), ivf.num_docs());
        let (budget, n_avg, x) = params.resolved_work_budget(clusters, docs)?;
        assert!(budget < clusters as f64, "setup: the budget must bind");
        drop(searcher);

        let (_, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            3,
            params,
        )?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert!(
            stats.clusters_probed() < clusters,
            "the budget must bind before exhaustion: {stats:?}"
        );
        assert!(
            stats.work_charged as f64 > budget,
            "the ceiling fires only once the budget is spent: {stats:?}"
        );
        // Overshoot bound: no single cluster can charge more than one
        // open plus every row in the segment.
        let max_cluster_charge = x + docs as f64 * (1.0 - x) / n_avg;
        assert!(
            stats.work_charged as f64 <= budget + max_cluster_charge + 1e-6,
            "overshoot is bounded by the last cluster's charge: {stats:?}"
        );
        Ok(())
    }

    /// `ProbeStats` (and nested routing / optional graph metrics) round-trip
    /// through `serde_json` with the field names callers rely on.
    #[test]
    fn probe_stats_serializes_to_json() {
        let mut stats = ProbeStats {
            candidates_scored: 10,
            rerank_rows: 4,
            scan_init_ns: 75,
            query_prep_ns: 125,
            routing_ns: 40,
            rerank_fetch_ns: 50,
            rerank_score_ns: 25,
            vectors_visited: 20,
            pruned_filter: 4,
            pruned_dead: 3,
            pruned_seen: 3,
            postings_row: 1,
            postings_skipped: 1,
            exact_rows_read: 0,
            bounds_skips: 2,
            termination: ProbeTermination::Ceiling,
            work_charged: 1.75,
            segment_rows: Some(100),
            segment_clusters: Some(5),
            ..Default::default()
        };
        stats.start_layer(0, -0.5, 2.25);
        stats.record_layer_scan(0, 10, 250);
        stats.record_boundary(0, 8, 50);
        stats.start_layer(1, 0.25, 2.5);
        stats.record_layer_scan(1, 8, 100);
        stats.record_boundary(1, 5, 25);
        stats.record_routing(IvfSearchMetrics {
            visited_count: 7,
            graph: Some(NeighborhoodGraphSearchMetrics {
                visited_count: 7,
                expanded_count: 4,
                edges_scanned: 12,
                evictions: 1,
                result_count: 3,
                termination_reason: SearchTerminationReason::SearchConverged,
            }),
        });
        stats.record_bound_armed(Some(1));

        let value = serde_json::to_value(&stats).expect("ProbeStats should serialize to JSON");
        assert_eq!(
            value,
            serde_json::json!({
                "candidates_scored": 10,
                "layer0_scan_ns": 250,
                "layer0_scored": 10,
                "layer0_survivors": 8,
                "layer0_bias": -0.5,
                "layer0_cal": 2.25,
                "boundary0_ns": 50,
                "layer1_scan_ns": 100,
                "layer1_scored": 8,
                "layer1_survivors": 5,
                "layer1_bias": 0.25,
                "layer1_cal": 2.5,
                "boundary1_ns": 25,
                "rerank_rows": 4,
                "scan_init_ns": 75,
                "query_prep_ns": 125,
                "routing_ns": 40,
                "rerank_fetch_ns": 50,
                "rerank_score_ns": 25,
                "vectors_visited": 20,
                "pruned_filter": 4,
                "pruned_dead": 3,
                "pruned_seen": 3,
                "postings_row": 1,
                "postings_skipped": 1,
                "exact_rows_read": 0,
                "routing_visited_count": 7,
                "routing_graph_count": 1,
                "routing_graph_visited_count": 7,
                "routing_graph_expanded_count": 4,
                "routing_graph_edges_scanned": 12,
                "routing_graph_evictions": 1,
                "routing_graph_result_count": 3,
                "bounds_skips": 2,
                "bound_armed_count": 1,
                "bound_armed_probe_sum": 1,
                "termination": "Ceiling",
                "work_charged": 1.75,
                "segment_rows": 100,
                "segment_clusters": 5
            })
        );
        assert_eq!(stats.clusters_probed(), 2);

        // Exact routing serializes only additive flat counters; nested routing
        // state is intentionally outside pdbench's numeric reach.
        let mut exact_routing = ProbeStats::default();
        exact_routing.record_routing(IvfSearchMetrics {
            visited_count: 7,
            graph: None,
        });
        let exact_value =
            serde_json::to_value(&exact_routing).expect("ProbeStats should serialize to JSON");
        assert_eq!(exact_value["routing_visited_count"], 7);
        assert_eq!(exact_value["routing_graph_count"], 0);
        assert!(exact_value.get("routing").is_none());
        assert!(exact_value.get("layer0_scan_ns").is_none());
    }

    #[test]
    fn quantized_boundary_counts_replicas_once() {
        let mut scan = QuantizedScanCtx::new(2, 2, true);
        for (row, doc, score) in [(0, 0, 100.0), (1, 0, 90.0), (2, 1, 0.0)] {
            scan.push(row, doc, score, 0.0, 0.0);
        }
        assert_eq!(scan.pessimistic_kth(2, 2.0), Some(0.0));
        scan.band(2, 2.0);
        assert!(
            scan.candidates.docs.contains(&1),
            "a replica of doc 0 must not displace the second distinct document"
        );
    }

    #[test]
    fn quantized_boundary_kth_uses_the_row_tie_for_sigma() {
        let mut scan = QuantizedScanCtx::new(3, 3, false);
        scan.begin_cluster(2);
        for (row, score, sigma) in [(0, 10.0, 0.0), (1, 9.0, 1.0), (2, 9.0, 100.0)] {
            scan.push(row, row as DocId, score, 0.0, sigma);
        }
        scan.finish_cluster_bound();

        // Equal estimates are ordered by the lower storage row. Admission
        // and the exact boundary must therefore widen row 1's estimate, not
        // choose row 2's much larger sigma arbitrarily.
        assert_eq!(scan.running_pessimistic_kth(2, 2.0), Some(7.0));
        assert_eq!(scan.pessimistic_kth(2, 2.0), Some(7.0));
    }

    #[test]
    fn cluster_local_admission_kth_matches_full_partition() {
        const TOP_N: usize = 4;
        let mut scan = QuantizedScanCtx::new(12, 24, true);
        for cluster in 0..6 {
            scan.begin_cluster(TOP_N);
            for row in cluster * 4..cluster * 4 + 4 {
                scan.push(
                    row,
                    (row % 12) as DocId,
                    3.0 - row as f32 * 0.071 + (row as f32 * 0.37).sin() * 0.2,
                    0.0,
                    0.01 + (row % 5) as f32 * 0.003,
                );
            }
            scan.finish_cluster_bound();
            assert_eq!(
                scan.running_pessimistic_kth(TOP_N, 2.0),
                scan.pessimistic_kth(TOP_N, 2.0),
                "cluster={cluster}"
            );
        }
    }

    fn independent_admission_top(
        scan: &QuantizedScanCtx,
        top_n: usize,
        dedup_docs: bool,
    ) -> Vec<usize> {
        fn oracle_precedes(scan: &QuantizedScanCtx, left: usize, right: usize) -> bool {
            let left_estimate =
                scan.candidates.scores[left] + scan.candidates.bias_corrections[left];
            let right_estimate =
                scan.candidates.scores[right] + scan.candidates.bias_corrections[right];
            right_estimate
                .total_cmp(&left_estimate)
                .then(scan.candidates.rows[left].cmp(&scan.candidates.rows[right]))
                .is_lt()
        }

        let mut indices = if dedup_docs {
            let mut best_by_doc = HashMap::<DocId, usize>::new();
            for index in 0..scan.candidates.len() {
                let doc = scan.candidates.docs[index];
                let best = best_by_doc.entry(doc).or_insert(index);
                if oracle_precedes(scan, index, *best) {
                    *best = index;
                }
            }
            best_by_doc.into_values().collect::<Vec<_>>()
        } else {
            (0..scan.candidates.len()).collect::<Vec<_>>()
        };
        indices.sort_unstable_by(|&left, &right| {
            let left_estimate =
                scan.candidates.scores[left] + scan.candidates.bias_corrections[left];
            let right_estimate =
                scan.candidates.scores[right] + scan.candidates.bias_corrections[right];
            right_estimate
                .total_cmp(&left_estimate)
                .then(scan.candidates.rows[left].cmp(&scan.candidates.rows[right]))
        });
        indices.truncate(top_n);
        indices
    }

    #[test]
    fn cluster_running_min_matches_independent_oracle() {
        // The clusters deliberately arrive fewer-than-k, descending, then
        // alternating. Doc 1 later improves; doc 2 later ties with a very
        // different sigma, for which the lower storage row must remain the
        // selected membership.
        let clusters: &[&[(usize, DocId, f32, f32)]] = &[
            &[(0, 0, 1.0, 0.01), (1, 1, 2.0, 0.02)],
            &[
                (2, 2, 9.0, 0.03),
                (3, 3, 8.0, 0.04),
                (4, 4, 7.0, 0.05),
                (5, 0, 6.0, 0.06),
                (6, 5, 5.0, 0.07),
            ],
            &[
                (7, 6, 0.0, 0.08),
                (8, 1, 10.0, 0.09),
                (9, 7, 1.0, 0.10),
                (10, 2, 9.0, 8.0),
                (11, 8, 2.0, 0.11),
            ],
        ];

        for dedup_docs in [false, true] {
            for top_n in [1, 3] {
                let mut scan = QuantizedScanCtx::new(16, 16, dedup_docs);
                for (cluster, rows) in clusters.iter().enumerate() {
                    scan.begin_cluster(top_n);
                    for &(row, doc, estimate, sigma) in *rows {
                        scan.push(row, doc, estimate, 0.0, sigma);
                    }
                    scan.finish_cluster_bound();

                    let expected = independent_admission_top(&scan, top_n, dedup_docs);
                    let actual_rows = scan
                        .bound_top
                        .iter()
                        .map(|&index| scan.candidates.rows[index])
                        .collect::<Vec<_>>();
                    let expected_rows = expected
                        .iter()
                        .map(|&index| scan.candidates.rows[index])
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual_rows, expected_rows,
                        "cluster={cluster}, dedup_docs={dedup_docs}, top_n={top_n}"
                    );

                    let expected_kth = (expected.len() == top_n).then(|| {
                        let index = expected[top_n - 1];
                        scan.candidates.scores[index] + scan.candidates.bias_corrections[index]
                            - 2.0 * scan.candidates.sigmas[index]
                    });
                    assert_eq!(
                        scan.running_pessimistic_kth(top_n, 2.0),
                        expected_kth,
                        "cluster={cluster}, dedup_docs={dedup_docs}, top_n={top_n}"
                    );
                }
            }
        }
    }

    #[test]
    fn replica_dedup_is_deferred_and_winner_can_flip() {
        let mut scan = QuantizedScanCtx::new(4, 4, true);
        scan.begin_cluster(2);
        for (row, doc, score, sigma) in [(10, 0, 10.0, 100.0), (11, 1, 8.0, 100.0)] {
            scan.push(row, doc, score, 0.0, sigma);
        }
        scan.finish_cluster_bound();
        scan.begin_cluster(2);
        scan.push(12, 0, 9.0, 0.0, 200.0);
        scan.finish_cluster_bound();

        assert_eq!(
            scan.candidates.docs.iter().filter(|&&doc| doc == 0).count(),
            2,
            "plane 1 must retain every replica membership"
        );
        assert!(
            scan.best_by_doc.is_empty() && scan.best_docs.is_empty(),
            "per-document hashing is deferred to a boundary"
        );

        scan.band(2, 2.0);
        scan.rebuild_best_by_doc();
        let winner_row = |scan: &QuantizedScanCtx, doc| {
            let index = scan.best_by_doc[&doc];
            scan.candidates.rows[index]
        };
        assert_eq!(winner_row(&scan, 0), 10);

        let row10 = scan
            .candidates
            .rows
            .iter()
            .position(|&row| row == 10)
            .unwrap();
        let row12 = scan
            .candidates
            .rows
            .iter()
            .position(|&row| row == 12)
            .unwrap();
        scan.candidates.scores[row10] = 5.0;
        scan.candidates.scores[row12] = 11.0;
        scan.band(2, 2.0);
        scan.rebuild_best_by_doc();
        assert_eq!(
            winner_row(&scan, 0),
            12,
            "refinement may flip the replica winner"
        );

        let row10 = scan
            .candidates
            .rows
            .iter()
            .position(|&row| row == 10)
            .unwrap();
        scan.candidates.scores[row10] = 11.0;
        scan.band(2, 2.0);
        scan.rebuild_best_by_doc();
        assert_eq!(
            winner_row(&scan, 0),
            10,
            "an equal estimate resolves to the lower storage row"
        );
    }

    #[test]
    fn quantized_split_constants_are_l2_only() {
        let stored = -0.375_f32;
        assert_eq!(quantized_layer_constant(Metric::L2, stored), stored);
        assert_eq!(quantized_layer_constant(Metric::Dot, stored), 0.0);
        assert_eq!(quantized_layer_constant(Metric::Cosine, stored), 0.0);
    }

    #[test]
    fn cosine_refinement_batches_cross_clusters_and_cap_at_2048() {
        let rows = (0..2_049).collect::<Vec<_>>();
        let clusters = [0..700, 700..1_400, 1_400..2_100];
        let batches = cosine_refinement_batches(rows.len()).collect::<Vec<_>>();
        assert_eq!(batches, [0..2_048, 2_048..2_049]);
        let first_rows = rows[batches[0].clone()].iter().copied();
        for cluster in clusters {
            assert!(
                first_rows.clone().any(|row| cluster.contains(&row)),
                "one logical cosine batch must cross all three clusters"
            );
        }
    }

    #[test]
    fn l2_refinement_keeps_cluster_norm_factors_and_constants_local() {
        let mut candidates = QuantizedCandidates::with_capacity(2);
        candidates.push(0, 10, 10.0, 0.0, 0.0);
        candidates.push(1, 11, 20.0, 0.0, 0.0);

        // Cluster 0: query-residual norm has already been folded into these
        // sigma/bias factors; its stored split constant is 5.
        combine_refinement_decoded(
            Metric::L2,
            &mut candidates,
            0..1,
            &[2.0],
            &[3.0],
            &[5.0],
            7.0,
            11.0,
        );
        // Cluster 1 deliberately uses different norm-derived factors and a
        // different stored constant. Neither may leak across the boundary.
        combine_refinement_decoded(
            Metric::L2,
            &mut candidates,
            1..2,
            &[4.0],
            &[2.0],
            &[1.0],
            13.0,
            17.0,
        );

        assert_eq!(candidates.scores, [12.0, 34.0]);
        assert_eq!(candidates.bias_corrections, [33.0, 34.0]);
        assert_eq!(candidates.sigmas, [21.0, 26.0]);
    }

    #[test]
    fn gate_b_recall_parity_survivor_sets() {
        const CANDIDATES: usize = 40;
        const TOP_N: usize = 10;
        const KAPPA_1: f32 = 2.0;
        const KAPPA_2: f32 = 4.0;

        let plane1_scores: Vec<f32> = (0..CANDIDATES)
            .map(|row| 2.0 - row as f32 * 0.09 + (row as f32 * 0.7).sin() * 0.08)
            .collect();
        let plane1_sigmas: Vec<f32> = (0..CANDIDATES)
            .map(|row| 0.03 + (row % 5) as f32 * 0.01)
            .collect();
        let refinements: Vec<f32> = (0..CANDIDATES)
            .map(|row| (row as f32 * 0.41).cos() * 0.06)
            .collect();
        let plane2_sigmas: Vec<f32> = (0..CANDIDATES)
            .map(|row| 0.012 + (row % 3) as f32 * 0.004)
            .collect();

        let (first_kth_index, first_kth) = cascade::kth(&plane1_scores, TOP_N);
        let harness_first = cascade::band_filter(
            &plane1_scores,
            &plane1_sigmas,
            KAPPA_1,
            first_kth - KAPPA_1 * plane1_sigmas[first_kth_index],
        );

        let mut scan = QuantizedScanCtx::new(CANDIDATES as DocId, CANDIDATES, false);
        for row in 0..CANDIDATES {
            scan.push(
                row,
                row as DocId,
                plane1_scores[row],
                0.0,
                plane1_sigmas[row],
            );
        }
        scan.band(TOP_N, KAPPA_1);
        let scan_first: Vec<u32> = scan.candidates.rows.iter().map(|&row| row as u32).collect();
        assert!(
            harness_first.len() < CANDIDATES,
            "GATE-B plane 1 must measurably filter"
        );
        assert_eq!(scan_first, harness_first, "plane-1 survivor set");

        let second_scores: Vec<f32> = harness_first
            .iter()
            .map(|&row| plane1_scores[row as usize] + refinements[row as usize])
            .collect();
        let second_sigmas: Vec<f32> = harness_first
            .iter()
            .map(|&row| plane2_sigmas[row as usize])
            .collect();
        let (second_kth_index, second_kth) = cascade::kth(&second_scores, TOP_N);
        let harness_second_local = cascade::band_filter(
            &second_scores,
            &second_sigmas,
            KAPPA_2,
            second_kth - KAPPA_2 * second_sigmas[second_kth_index],
        );
        let harness_second: Vec<u32> = harness_second_local
            .iter()
            .map(|&local| harness_first[local as usize])
            .collect();

        for index in 0..scan.candidates.len() {
            let row = scan.candidates.rows[index];
            scan.candidates.scores[index] += refinements[row];
            scan.candidates.sigmas[index] = plane2_sigmas[row];
        }
        scan.band(TOP_N, KAPPA_2);
        let scan_second: Vec<u32> = scan.candidates.rows.iter().map(|&row| row as u32).collect();
        assert!(
            harness_second.len() < harness_first.len(),
            "GATE-B plane 2 must measurably filter"
        );
        assert_eq!(scan_second, harness_second, "plane-2 survivor set");

        let mut exact_order: Vec<usize> = (0..CANDIDATES).collect();
        exact_order.sort_unstable_by(|&left, &right| {
            (plane1_scores[right] + refinements[right])
                .total_cmp(&(plane1_scores[left] + refinements[left]))
                .then_with(|| left.cmp(&right))
        });
        let recalled = exact_order[..TOP_N]
            .iter()
            .filter(|&&row| scan_second.contains(&(row as u32)))
            .count();
        assert_eq!(recalled, TOP_N, "candidate recall must match the harness");
    }

    #[test]
    fn quantized_boundary_filters_with_finite_sigmas() {
        let mut scan = QuantizedScanCtx::new(3, 3, false);
        for (row, score) in [(0, 100.0), (1, 99.0), (2, -100.0)] {
            scan.push(row, row as DocId, score, 0.0, 0.1);
        }
        scan.band(1, 2.0);
        assert_eq!(scan.candidates.len(), 1);
        assert_eq!(scan.candidates.rows[0], 0);
    }

    #[test]
    fn quantized_bias_is_applied_once_at_estimate_access() {
        let mut scan = QuantizedScanCtx::new(2, 2, false);
        scan.push(0, 0, 10.0, -100.0, 0.0);
        scan.push(1, 1, 9.0, 0.0, 0.0);
        assert_eq!(scan.candidates.estimate(0), -90.0);
        scan.band(1, 0.0);
        assert_eq!(scan.candidates.len(), 1);
        assert_eq!(scan.candidates.rows[0], 1);
    }

    #[test]
    fn quantized_depth_two_bias_replaces_depth_one_correction() {
        // Match the first sample in the independent calibration oracle:
        // depth-1 prefix 9 with correction +1, then depth-2 prefix 14 with
        // correction -4. The active-depth estimate remains 10; retaining
        // depth 1 as an additional correction would incorrectly produce 11.
        let mut candidates = QuantizedCandidates::with_capacity(1);
        candidates.push(0, 0, 9.0, 1.0, 1.0);
        assert_eq!(candidates.estimate(0).to_bits(), 10.0_f32.to_bits());

        candidates.scores[0] += 5.0;
        candidates.bias_corrections[0] = -4.0;
        assert_eq!(candidates.estimate(0).to_bits(), 10.0_f32.to_bits());
        assert_ne!(candidates.estimate(0).to_bits(), 11.0_f32.to_bits());
    }

    // ============================================================
    // Filter-aware posting fetches.
    //
    // The probe loop decides each cluster's survivors from the pinned
    // id-map BEFORE touching posting bytes, then fetches survivors with
    // one stride-sized read per row — or nothing at all when the gate
    // leaves no survivors. These tests pin the skip/fetch behavior and
    // that the two `postings_*` counters partition the probed clusters.
    // ============================================================

    /// A `Weight` over a fixed, hand-built doc-id set, so tests can hand
    /// the backend an exact filter BitSet without routing it through a
    /// real query.
    struct FixedDocsWeight {
        max_doc: DocId,
        docs: Vec<DocId>,
    }

    impl Weight for FixedDocsWeight {
        fn scorer(&self, _reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
            let mut bs = BitSet::with_max_value(self.max_doc);
            for &doc in &self.docs {
                bs.insert(doc);
            }
            Ok(Box::new(ConstScorer::new(BitSetDocSet::from(bs), boost)))
        }

        fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> crate::Result<Explanation> {
            unreachable!("the vector backend never explains filter docs")
        }
    }

    /// Like [`run_top_n`] but with a caller-supplied filter
    /// weight.
    fn run_top_n_with_weight(
        index: &Index,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
        weight: &dyn Weight,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let quantized_queries = QuantizedQueryCache::default();
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            &quantized_queries,
            params,
            usize::MAX,
        )?;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_some(),
            "expected IVF storage"
        );
        backend.top_n(weight, segment_reader, k)
    }

    /// Every touched row lands in exactly one prune bucket.
    fn assert_stats_identities(stats: &ProbeStats) {
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
            "visited must equal filter+dead+seen+scored ({stats:?})"
        );
    }

    /// Build a single-segment FLAT index (one commit, never merged past
    /// the clustering threshold): `docs` are `(label, Some(vector))`, or
    /// `(label, None)` for vectorless docs — mixing the two forces the
    /// `Bitmap` id-map.
    fn build_flat(
        dim: usize,
        docs: &[(&str, Option<Vec<f32>>)],
    ) -> crate::Result<(Index, Field, Field)> {
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(dim, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let index = Index::create_in_ram(sb.build());
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for (label, v) in docs {
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, label);
            if let Some(v) = v {
                doc.add_vector(embed_field, v.as_slice());
            }
            writer.add_document(doc)?;
        }
        writer.commit()?;
        Ok((index, embed_field, label_field))
    }

    /// Run the flat/exact path against `segment_reader` with a
    /// caller-supplied filter weight.
    fn run_exact_on_segment(
        segment_reader: &SegmentReader,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        weight: &dyn Weight,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let quantized_queries = QuantizedQueryCache::default();
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            &quantized_queries,
            AdaptiveProbeParams::default(),
            usize::MAX,
        )?;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_none(),
            "expected flat storage"
        );
        backend.top_n(weight, segment_reader, k)
    }

    /// Filter-aware fetches across selectivities: on the multi-cluster
    /// fixture, hand-built filters admitting {0, 1, 50, 100}% of docs
    /// return every admitted doc exactly once, with both partition
    /// identities intact. 0% admits nothing — the empty-filter
    /// short-circuit returns before the probe loop, so zero clusters
    /// probe and zero fetches happen (postings_skipped == clusters
    /// probed == 0).
    #[test]
    fn filter_aware_fetch_across_selectivities() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let params = exhaustive_params(centroids.len());

        for pct in [0usize, 1, 50, 100] {
            // Admit the first ceil(pct% · n) doc ids: doc · 100 < pct · n.
            let weight = FixedDocsWeight {
                max_doc: n as DocId,
                docs: (0..n as DocId)
                    .filter(|&doc| (doc as usize) * 100 < pct * n)
                    .collect(),
            };
            let admitted = weight.docs.len();
            let (hits, stats) = run_top_n_with_weight(
                &index,
                embed_field,
                vec![10.0, 10.0],
                n,
                params.clone(),
                &weight,
            )?;

            assert_eq!(
                hits.len(),
                admitted,
                "{pct}%: every admitted doc returns exactly once"
            );
            assert_stats_identities(&stats);

            match pct {
                0 => {
                    assert_eq!(stats.postings_row, 0, "0%: no fetches");
                    assert_eq!(
                        stats.postings_skipped,
                        stats.clusters_probed(),
                        "0%: every probed cluster skips its fetch"
                    );
                }
                1 => {
                    // One admitted doc → exactly one cluster fetches (its
                    // primary); every other probed cluster skips.
                    assert_eq!(stats.postings_row, 1, "{stats:?}");
                    assert_eq!(
                        stats.postings_skipped,
                        stats.clusters_probed() - 1,
                        "{stats:?}"
                    );
                }
                _ => {
                    assert!(stats.postings_row > 0, "{stats:?}");
                }
            }
        }
        Ok(())
    }

    /// Deletes are decided in the pre-pass: a cluster whose rows are all
    /// dead yields zero survivors and fetches nothing.
    #[test]
    fn filter_aware_fetch_skips_all_dead_clusters() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        // Cluster 0's rows are exactly its 6 primary docs — deleting
        // those leaves a fully-dead cluster.
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        {
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for i in 0..REPLICATION_N_PER {
                writer.delete_term(Term::from_field_text(label_field, &format!("d{i}")));
            }
            writer.commit()?;
        }

        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let segment_reader = &searcher.segment_readers()[0];
        // Setup: the tombstones landed, and cluster 0 is exactly the
        // dead docs.
        let alive = segment_reader.alive_bitset().expect("deletes must land");
        let cluster0 = segment_reader
            .vector_index(embed_field)?
            .cluster_doc_ids(0)
            .expect("ivf cluster 0");
        assert_eq!(cluster0.len(), REPLICATION_N_PER);
        assert!(
            cluster0.iter().all(|&doc| !alive.is_alive(doc)),
            "cluster 0 must be fully dead"
        );

        let max_doc = segment_reader.max_doc();
        let weight = FixedDocsWeight {
            max_doc,
            docs: (0..max_doc).collect(),
        };
        let params = exhaustive_params(centroids.len());
        let (hits, stats) =
            run_top_n_with_weight(&index, embed_field, vec![10.0, 10.0], n, params, &weight)?;

        assert_eq!(hits.len(), n - REPLICATION_N_PER, "only alive docs surface");
        assert_eq!(
            stats.pruned_dead, REPLICATION_N_PER,
            "every dead row prunes as dead: {stats:?}"
        );
        // The fully-dead cluster fetches nothing; the other five fetch.
        assert_eq!(stats.postings_skipped, 1, "{stats:?}");
        assert_eq!(stats.postings_row, centroids.len() - 1);
        assert_stats_identities(&stats);
        Ok(())
    }

    /// An empty cluster still counts as probed — the loop visits it and
    /// takes the skip path: `postings_skipped` increments, nothing is
    /// fetched, and the visited/prune counters don't move.
    #[test]
    fn empty_cluster_probed_but_fetch_skipped() -> crate::Result<()> {
        // No doc is nearest to the third centroid, so its cluster is
        // empty. The
        // empty centroid sits ON the query so its zero-radius bound can
        // never prove it useless - the bounds gate must not be what
        // skips it; the empty-fetch path is what's under test.
        let centroids = vec![[0.0f32, 0.0], [10.0, 0.0], [5.0, 0.1]];
        let labels: Vec<String> = (0..8).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, [f32; 2])> = (0..8)
            .map(|i| {
                let c = centroids[i % 2];
                (labels[i].as_str(), [c[0] + (i / 2) as f32 * 0.01, c[1]])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        assert_eq!(
            vec_reader.cluster_sizes(),
            Some(vec![4, 4, 0]),
            "setup: cluster 2 must be empty"
        );

        let max_doc = segment_reader.max_doc();
        let weight = FixedDocsWeight {
            max_doc,
            docs: (0..max_doc).collect(),
        };
        let (hits, stats) = run_top_n_with_weight(
            &index,
            embed_field,
            vec![5.0, 0.0],
            8,
            exhaustive_params(centroids.len()),
            &weight,
        )?;
        assert_eq!(hits.len(), 8);
        assert_eq!(
            stats.clusters_probed(),
            centroids.len(),
            "the empty cluster still counts as probed: {stats:?}"
        );
        assert_eq!(
            stats.postings_skipped, 1,
            "the empty cluster fetched nothing: {stats:?}"
        );
        assert_eq!(stats.postings_row, 2);
        assert_eq!(
            stats.vectors_visited, 8,
            "an empty cluster contributes no visited rows"
        );
        assert_stats_identities(&stats);
        Ok(())
    }

    /// Flat-path behavior across filter selectivities: hand-built filters
    /// admitting {0, 1, 50, 100}% of docs return every admitted doc
    /// exactly once, the probe-loop counters stay zeroed, and the path
    /// serves exactly one stride-sized row read per survivor.
    #[test]
    fn flat_exact_reads_one_row_per_survivor() -> crate::Result<()> {
        let n = 40usize;
        let labels: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, Option<Vec<f32>>)> = (0..n)
            .map(|i| (labels[i].as_str(), Some(vec![i as f32 * 0.1, 1.0])))
            .collect();
        let (index, embed_field, _label) = build_flat(2, &docs)?;
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let max_doc = segment_reader.max_doc();
        assert_eq!(max_doc as usize, n, "one segment holding every doc");

        for pct in [0usize, 1, 50, 100] {
            let weight = FixedDocsWeight {
                max_doc,
                docs: (0..max_doc)
                    .filter(|&doc| (doc as usize) * 100 < pct * n)
                    .collect(),
            };
            let admitted = weight.docs.len();

            let (hits, stats) =
                run_exact_on_segment(segment_reader, embed_field, vec![0.0, 0.0], n, &weight)?;

            assert_eq!(hits.len(), admitted, "{pct}%");
            // The exact path fills its row and stage instrumentation; the
            // probe-loop counters stay zeroed.
            assert_eq!(stats.vectors_visited, 0, "{pct}%: {stats:?}");
            assert_eq!(stats.candidates_scored, 0, "{pct}%: {stats:?}");
            assert_eq!(stats.clusters_probed(), 0, "{pct}%: {stats:?}");
            assert_eq!(
                stats.exact_rows_read, admitted,
                "{pct}%: one row read per survivor"
            );
            assert!(stats.exact_scan_ns.is_some(), "{pct}%: {stats:?}");
            assert!(stats.result_assembly_ns.is_some(), "{pct}%: {stats:?}");
        }
        Ok(())
    }

    /// Flat scan over a `Bitmap` id-map with tombstoned docs: vectorless
    /// and deleted docs leave holes between surviving rows; every alive
    /// doc with a vector is read and scored exactly once, matching the
    /// brute-force oracle.
    #[test]
    fn flat_exact_handles_bitmap_holes_and_deletes() -> crate::Result<()> {
        let n = 30usize;
        let labels: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
        // Every third doc carries no vector at all → the id-map is Bitmap
        // and those docs own no row.
        let docs: Vec<(&str, Option<Vec<f32>>)> = (0..n)
            .map(|i| {
                let v = (i % 3 != 2).then(|| vec![i as f32 * 0.1, 1.0]);
                (labels[i].as_str(), v)
            })
            .collect();
        let (index, embed_field, label_field) = build_flat(2, &docs)?;
        // Tombstone a few vectored docs → alive holes between rows that
        // do exist.
        {
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for i in [0usize, 6, 7, 12] {
                writer.delete_term(Term::from_field_text(label_field, &format!("d{i}")));
            }
            writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        // Setup: mixed vector coverage (Bitmap id-map: fewer rows than
        // docs) and the tombstones landed.
        assert!(vec_reader.num_vectors() < segment_reader.max_doc() as usize);
        assert!(segment_reader.alive_bitset().is_some());

        let max_doc = segment_reader.max_doc();
        let weight = FixedDocsWeight {
            max_doc,
            docs: (0..max_doc).collect(),
        };
        let query = vec![0.0f32, 0.0];
        let (hits, stats) =
            run_exact_on_segment(segment_reader, embed_field, query.clone(), n, &weight)?;

        // Oracle: every alive doc with a vector, exactly once.
        let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, n)?;
        assert_eq!(hits, expected);

        let survivors = hits.len();
        assert!(survivors > 0, "fixture must leave survivors");
        assert_eq!(
            stats.exact_rows_read, survivors,
            "one row read per survivor: {stats:?}"
        );
        Ok(())
    }

    // ---- Test-only helpers ----

    /// Compute a brute-force top-K with the same convention as the
    /// shared fixture's `ground_truth::top_k`, but accepting any
    /// `&Index` (the inline-built IVF index for crafted tests doesn't
    /// have a `TestVectorIndex` wrapper).
    fn ground_truth_top_k(
        index: &Index,
        vec_field: Field,
        metric: Metric,
        query: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let query = PreparedQuery::<f32>::new(metric, Arc::new(query.to_vec()));
        let searcher = index.reader()?.searcher();
        let mut scored = Vec::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            let vec_reader = segment_reader.vector_index(vec_field)?;
            let alive = segment_reader.alive_bitset();
            for doc in 0..segment_reader.max_doc() {
                if let Some(alive) = alive {
                    if !alive.is_alive(doc) {
                        continue;
                    }
                }
                if let Some(bytes) = vec_reader.vector_bytes(doc)? {
                    scored.push((
                        query.score_doc_bytes(&bytes),
                        DocAddress::new(seg_ord as u32, doc),
                    ));
                }
            }
        }
        scored.sort_by(|a: &(Score, DocAddress), b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.segment_ord.cmp(&b.1.segment_ord))
                .then(a.1.doc_id.cmp(&b.1.doc_id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Collect the set of `DocAddress`es that a `Query` admits, by
    /// walking the per-segment weight. Used by the filter selectivity
    /// test to build an oracle restricted to the filter set.
    fn collect_filter_doc_set(
        index: &Index,
        filter: &dyn Query,
    ) -> crate::Result<std::collections::HashSet<DocAddress>> {
        let searcher = index.reader()?.searcher();
        let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let mut admitted = std::collections::HashSet::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            weight.for_each_no_score(segment_reader, &mut |docs| {
                for &d in docs {
                    admitted.insert(DocAddress::new(seg_ord as u32, d));
                }
            })?;
        }
        Ok(admitted)
    }

    /// Read the stored label text at the given `DocAddress`.
    /// Used by the trap-case + floor tests to identify docs by name
    /// rather than relying on DocId (which the merger reassigns).
    fn stored_label_at(
        index: &Index,
        label_field: Field,
        addr: DocAddress,
    ) -> crate::Result<String> {
        use crate::schema::Value;
        let searcher = index.reader()?.searcher();
        let doc = searcher.doc::<TantivyDocument>(addr)?;
        Ok(doc
            .get_first(label_field)
            .and_then(|v| Value::as_str(&v))
            .expect("stored label")
            .to_string())
    }

    /// The shared fixture's first centroid (top-left of the 3×3 grid).
    fn grid2d_first_centroid() -> [f32; 2] {
        [0.0, 0.0]
    }

    // ==================================================================
    // P5: the bounds gate
    // ==================================================================

    mod bounds_gate_tests {
        use super::*;
        use crate::vector::bounds::{HeapPeek, QueryBound, QueryBoundTracker};
        use crate::vector::{margin_ball_ball, margin_ball_halfspace, to_bound_space};

        /// A single-segment IVF index over `docs` with fixed `centroids`:
        /// ONE commit, then a single-segment merge — a multi-segment
        /// merge's source order varies across processes and permutes
        /// target doc ids, and these tests assert doc-id-level results.
        pub(super) fn single_segment_fixture(
            metric: Metric,
            centroids: &[[f32; 2]],
            docs: &[[f32; 2]],
        ) -> crate::Result<(Index, Field)> {
            let mut sb = Schema::builder();
            let embed_field = sb.add_vector_field(
                "embedding",
                VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
            );
            sb.add_text_field("label", STRING | STORED);
            let schema = sb.build();
            let settings = IndexSettings {
                vector_clustering_threshold: 1,
                ..IndexSettings::default()
            };
            let index = Index::builder()
                .schema(schema)
                .settings(settings)
                .ivf_clusterer(Arc::new(InlineClusterer {
                    centroids: centroids.to_vec(),
                }))
                .create_in_ram()?;
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for (i, vector) in docs.iter().enumerate() {
                let mut doc = TantivyDocument::new();
                doc.add_text(index.schema().get_field("label").unwrap(), format!("d{i}"));
                doc.add_vector(embed_field, vector.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
            let segment_ids = index.searchable_segment_ids()?;
            assert_eq!(segment_ids.len(), 1, "single flat segment");
            writer.merge(&segment_ids).wait()?;
            writer.wait_merging_threads()?;
            Ok((index, embed_field))
        }

        /// Deterministic pseudo-random `f32` in `[-8, 8)` — a tiny LCG so
        /// the sweep needs no RNG dependency and every run replays.
        struct Lcg(u64);
        impl Lcg {
            fn next_f32(&mut self) -> f32 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 16.0
            }
            fn point(&mut self) -> [f32; 2] {
                loop {
                    let p = [self.next_f32(), self.next_f32()];
                    // Keep cosine's write normalization well-conditioned.
                    if p[0] * p[0] + p[1] * p[1] > 0.25 {
                        return p;
                    }
                }
            }
        }

        /// PROPERTY TEST — the home-cluster closure theorem. Random data x
        /// {L2, cosine, dot}; brute-force top-k vs gated top-k under a
        /// full budget. Asserts:
        ///
        /// (a) the result sets are identical — a wrong skip would lose a
        ///     true member;
        /// (b) no true member's HOME cluster is skippable even at the
        ///     FINAL (tightest) bound: `margin(t_final) >= 0`. The
        ///     running bound is never tighter than the final one (the kth
        ///     only improves), so this proves the home cluster probed at
        ///     every point of the scan — the bound fold covers every row,
        ///     so a qualifying row's cluster always fails the skip test.
        #[test]
        fn closure_no_true_member_skipped() -> crate::Result<()> {
            let k = 5;
            for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
                for seed in [11u64, 29, 47] {
                    let mut rng = Lcg(seed ^ (metric as u64) << 32);
                    let centroids: Vec<[f32; 2]> = (0..5).map(|_| rng.point()).collect();
                    let docs: Vec<[f32; 2]> = (0..40).map(|_| rng.point()).collect();
                    let (index, field) = single_segment_fixture(metric, &centroids, &docs)?;
                    let query: Vec<f32> = rng.point().to_vec();

                    let brute = ground_truth_top_k(&index, field, metric, &query, k)?;
                    let gated = search(
                        &index,
                        field,
                        &AllQuery,
                        query.clone(),
                        k,
                        exhaustive_params(centroids.len()),
                    )?;
                    assert_eq!(
                        gated, brute,
                        "{metric:?} seed {seed}: gated top-k must equal brute force"
                    );

                    // (b): the theorem, cluster by cluster. Homes are
                    // recomputed with the clusterer's own rule: stored
                    // (post-normalization) doc values against the RAW
                    // trained centroids - the values `assign` saw. The
                    // margin then runs against the STORED (normalized)
                    // centroid, exactly as the gate does; the fold covers
                    // members whatever rule assigned them, so the
                    // triangle argument is assignment-rule-agnostic.
                    let searcher = index.reader()?.searcher();
                    let segment_reader = &searcher.segment_readers()[0];
                    let vec_reader = segment_reader.vector_index(field)?;
                    let ivf = vec_reader.index().expect("IVF segment");
                    let bounds = ivf.bounds();
                    let centroid_bytes = ivf.centroid_bytes()?;
                    let stride = 2 * std::mem::size_of::<f32>();
                    let kth_key = brute[k - 1].0;
                    let t_final = to_bound_space(metric, kth_key);
                    let q_norm = (query[0] * query[0] + query[1] * query[1]).sqrt();
                    for &(_, addr) in &brute {
                        let stored = decode_2d(
                            &vec_reader
                                .vector_bytes(addr.doc_id)?
                                .expect("stored vector"),
                        );
                        let home = nearest_centroid(stored, &centroids);
                        let sim = Metric::similarity_bytes::<f32>(
                            metric,
                            &query,
                            &centroid_bytes[home * stride..(home + 1) * stride],
                        );
                        let r = bounds.ball_r(home);
                        let margin = match metric {
                            Metric::L2 | Metric::Cosine => {
                                margin_ball_ball(t_final, r, to_bound_space(metric, sim.score()))
                            }
                            Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t_final),
                        };
                        assert!(
                            margin >= 0.0,
                            "{metric:?} seed {seed}: true member {} home cluster {home} must \
                             never be skippable (margin {margin})",
                            addr.doc_id
                        );
                    }
                }
            }
            Ok(())
        }

        /// Boundary ties, L2: a cluster whose margin is EXACTLY zero (all
        /// values powers of two — the arithmetic is exact) holds a doc
        /// tying the kth at d == t. Exact touch must PROBE: the tie doc
        /// is scored and the doc-id tie-break decides, identically to
        /// brute force. The far cluster is provably useless and skipped.
        #[test]
        fn boundary_tie_probes_l2() -> crate::Result<()> {
            let centroids = [[0.0f32, 4.0], [4.0, 0.0], [8.0, 0.0]];
            // d0 home A (r 2), d1 home C at margin-zero touch (r 2),
            // d2 home B (r 1, disjoint by 5 - strictly nearest B, no
            // assignment tie with C).
            let docs = [[0.0f32, 2.0], [2.0, 0.0], [7.0, 0.0]];
            let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs)?;
            let query = vec![0.0f32, 0.0];

            let brute = ground_truth_top_k(&index, field, Metric::L2, &query, 1)?;
            assert_eq!(
                brute[0].1.doc_id, 0,
                "tie at d = 2 breaks to the lower doc id"
            );

            let (hits, stats) = run_top_n(&index, field, query, 1, exhaustive_params(3))?;
            assert_eq!(
                hits, brute,
                "exact-touch cluster must probe, preserving the tie"
            );
            assert_eq!(
                stats.candidates_scored, 2,
                "both d = 2 docs are scored - the margin == 0 cluster probed"
            );
            assert_eq!(
                stats.clusters_probed(),
                2,
                "the disjoint cluster is skipped, the touching one is not"
            );
            assert_eq!(stats.termination, ProbeTermination::Exhausted);
            Ok(())
        }

        /// Boundary ties, dot: the second cluster's best possible score
        /// (q.c + ||q||*r) EQUALS the kth score — margin exactly zero,
        /// integer arithmetic. Exact touch probes; the tied doc wins on
        /// doc id exactly as brute force says.
        #[test]
        fn boundary_tie_probes_dot() -> crate::Result<()> {
            let centroids = [[2.0f32, 0.0], [4.0, 4.0]];
            // d0 = (3, 0) home c0, score 3; d1 = (3, 4) home c1, score 3.
            let docs = [[3.0f32, 0.0], [3.0, 4.0]];
            let (index, field) = single_segment_fixture(Metric::Dot, &centroids, &docs)?;
            let query = vec![1.0f32, 0.0];

            let brute = ground_truth_top_k(&index, field, Metric::Dot, &query, 1)?;
            assert_eq!(
                brute[0].1.doc_id, 0,
                "score tie at 3 breaks to the lower doc id"
            );

            let (hits, stats) = run_top_n(&index, field, query, 1, exhaustive_params(2))?;
            assert_eq!(hits, brute);
            assert_eq!(
                stats.candidates_scored, 2,
                "the margin == 0 cluster probed and scored its tied doc"
            );
            Ok(())
        }

        /// Forced skips charge exactly the open share: on well-separated
        /// clusters the scan probes one cluster and proves the other five
        /// useless, and the work charge equals
        /// `probed*x + skipped*x + scored*(1 - x)/n_avg` — free skips
        /// would break this identity.
        #[test]
        fn skips_charge_open_share() -> crate::Result<()> {
            let centroids: Vec<[f32; 2]> = vec![
                [1.0, 1.0],
                [11.0, 1.0],
                [21.0, 1.0],
                [1.0, 11.0],
                [11.0, 11.0],
                [21.0, 11.0],
            ];
            let docs: Vec<[f32; 2]> = (0..36)
                .map(|i| {
                    let c = centroids[i / 6];
                    let off = (i % 6) as f32 * 0.01;
                    [c[0] + off, c[1] + off]
                })
                .collect();
            let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;

            assert_eq!(stats.termination, ProbeTermination::Exhausted);
            assert_eq!(
                stats.clusters_probed(),
                1,
                "only the home cluster survives the margins: {stats:?}"
            );
            assert_eq!(stats.candidates_scored, 6);
            // All six ranked clusters were pulled (Exhausted); five were
            // passed over by the gate.
            let skipped = 6 - stats.clusters_probed();
            let n_avg = 36.0f64 / 6.0;
            let x = open_share(n_avg);
            let row = (1.0 - x) / n_avg;
            let expected = (stats.clusters_probed() + skipped) as f64 * x
                + stats.candidates_scored as f64 * row;
            assert!(
                (stats.work_charged as f64 - expected).abs() < 1e-5,
                "skips must charge the open share: expected {expected}, got {}",
                stats.work_charged
            );
            Ok(())
        }

        /// With k unreachable the heap never fills, the bound never arms,
        /// and NOTHING is skipped — every cluster probes and the work
        /// identity has no skip term.
        #[test]
        fn unarmed_never_skips() -> crate::Result<()> {
            let centroids: Vec<[f32; 2]> = vec![[1.0, 1.0], [11.0, 1.0], [21.0, 1.0]];
            let docs: Vec<[f32; 2]> = (0..9)
                .map(|i| {
                    let c = centroids[i / 3];
                    let off = (i % 3) as f32 * 0.01;
                    [c[0] + off, c[1] + off]
                })
                .collect();
            let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs)?;
            // k = 100 > 9 docs: the heap can never hold k results.
            let (hits, stats) =
                run_top_n(&index, field, vec![0.2, 0.3], 100, exhaustive_params(3))?;
            assert_eq!(hits.len(), 9, "every doc is a hit at k > N");
            assert_eq!(
                stats.clusters_probed(),
                3,
                "unarmed, every cluster probes: {stats:?}"
            );
            assert_eq!(stats.candidates_scored, 9);
            let n_avg = 3.0f64;
            let x = open_share(n_avg);
            let row = (1.0 - x) / n_avg;
            let expected = 3.0 * x + 9.0 * row;
            assert!(
                (stats.work_charged as f64 - expected).abs() < 1e-5,
                "no skip term in the identity: expected {expected}, got {}",
                stats.work_charged
            );
            Ok(())
        }

        /// `t` tracks kth improvements in bound space per metric, and an
        /// unchanged kth leaves the bound untouched. The bound-space
        /// conversion runs inside the improvement branch only, so the
        /// cosine sqrt is paid per improvement — asserted here by value
        /// (the cached and recomputed paths are indistinguishable by
        /// construction when the key is unchanged).
        #[test]
        fn t_maintenance_per_metric() {
            // (metric, first kth key, expected t, improved key, expected t)
            let cases = [
                // L2 keys -d^2: d = 2, then d = 1.
                (Metric::L2, -4.0f32, 2.0f32, -1.0f32, 1.0f32),
                // Cosine keys cos: chord sqrt(2*(1-cos)).
                (Metric::Cosine, 0.5, 1.0, 0.875, 0.5),
                // Dot keys the score; identity.
                (Metric::Dot, 3.0, 3.0, 4.5, 4.5),
            ];
            for (metric, first, t_first, improved, t_improved) in cases {
                let mut tracker = QueryBoundTracker::new();
                assert_eq!(tracker.bound(), QueryBound::Filling);
                tracker.observe(metric, HeapPeek::Filling, 0);
                assert_eq!(
                    tracker.bound(),
                    QueryBound::Filling,
                    "{metric:?}: no arm on Filling"
                );

                tracker.observe(metric, HeapPeek::Full { kth_key: first }, 1);
                assert_eq!(
                    tracker.bound(),
                    QueryBound::Armed { t: t_first },
                    "{metric:?}: t from first kth"
                );
                // Unchanged kth: bound bit-identical.
                tracker.observe(metric, HeapPeek::Full { kth_key: first }, 2);
                assert_eq!(tracker.bound(), QueryBound::Armed { t: t_first });

                tracker.observe(metric, HeapPeek::Full { kth_key: improved }, 3);
                assert_eq!(
                    tracker.bound(),
                    QueryBound::Armed { t: t_improved },
                    "{metric:?}: t tracks the improvement"
                );
            }
        }

        /// The armed index is the first probe at which the heap held k
        /// results, and never moves after.
        #[test]
        fn armed_index_recorded() {
            let mut tracker = QueryBoundTracker::new();
            tracker.observe(Metric::L2, HeapPeek::Filling, 0);
            tracker.observe(Metric::L2, HeapPeek::Filling, 1);
            assert_eq!(tracker.armed_at_probe(), None, "unarmed while filling");
            tracker.observe(Metric::L2, HeapPeek::Full { kth_key: -1.0 }, 2);
            assert_eq!(tracker.armed_at_probe(), Some(2), "arms at the first Full");
            tracker.observe(Metric::L2, HeapPeek::Full { kth_key: -0.5 }, 3);
            assert_eq!(
                tracker.armed_at_probe(),
                Some(2),
                "later improvements never move the armed index"
            );
        }
    }

    // ==================================================================
    // P6: probe-stats telemetry
    // ==================================================================

    mod bounds_stats_tests {
        use super::bounds_gate_tests::single_segment_fixture;
        use super::*;

        /// The six-centroid separated fixture: probing the home cluster
        /// arms the bound and the other five clusters are provably
        /// useless.
        fn separated_fixture(metric: Metric) -> crate::Result<(Index, Field)> {
            let centroids: Vec<[f32; 2]> = vec![
                [1.0, 1.0],
                [11.0, 1.0],
                [21.0, 1.0],
                [1.0, 11.0],
                [11.0, 11.0],
                [21.0, 11.0],
            ];
            let docs: Vec<[f32; 2]> = (0..36)
                .map(|i| {
                    let c = centroids[i / 6];
                    let off = (i % 6) as f32 * 0.01;
                    [c[0] + off, c[1] + off]
                })
                .collect();
            single_segment_fixture(metric, &centroids, &docs)
        }

        /// `bounds_skips` counts exactly the clusters the gate passed
        /// over: all ranked clusters minus the probed ones on an
        /// exhausted stream.
        #[test]
        fn skip_count_matches() -> crate::Result<()> {
            let (index, field) = separated_fixture(Metric::L2)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;
            assert_eq!(stats.termination, ProbeTermination::Exhausted);
            assert_eq!(stats.clusters_probed(), 1, "{stats:?}");
            assert_eq!(
                stats.bounds_skips, 5,
                "every non-home cluster is a counted skip: {stats:?}"
            );
            Ok(())
        }

        /// Unarmed (k > N): the additive armed counters remain zero.
        #[test]
        fn armed_null_when_unarmed() -> crate::Result<()> {
            let (index, field) = separated_fixture(Metric::L2)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 100, exhaustive_params(6))?;
            assert_eq!(stats.bounds_skips, 0);
            let value = serde_json::to_value(&stats).expect("ProbeStats serializes");
            assert_eq!(value["bound_armed_count"], 0);
            assert_eq!(value["bound_armed_probe_sum"], 0);
            Ok(())
        }

        /// The armed index is the tracker's recorded value: the heap
        /// fills inside the first probed cluster at k <= its size
        /// (index 0), and spans into the second at larger k (index 1).
        #[test]
        fn armed_index_value() -> crate::Result<()> {
            let (index, field) = separated_fixture(Metric::L2)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;
            assert_eq!(stats.bound_armed_count, 1, "{stats:?}");
            assert_eq!(stats.bound_armed_probe_sum, 0, "{stats:?}");
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 10, exhaustive_params(6))?;
            assert_eq!(stats.bound_armed_count, 1, "{stats:?}");
            assert_eq!(
                stats.bound_armed_probe_sum, 1,
                "k = 10 needs the second probed cluster: {stats:?}"
            );
            assert!(
                stats.bounds_skips > 0,
                "armed late still skips the far tail"
            );
            Ok(())
        }
    }
}
