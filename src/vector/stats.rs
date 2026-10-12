//! Per-segment probe statistics and their serialization.

use crate::vector::router::RouterMetrics;
use crate::vector::source::controller::ProbeTermination;
#[cfg(test)]
use crate::DocId;

/// Per-segment probe instrumentation.
#[derive(Debug, Default)]
pub struct LayerProbeStats {
    /// Owned-word decoding fallbacks in sign-plane scoring.
    pub sign_word_fallbacks: u64,
    /// Actual read requests made by this layer.
    pub io: crate::vector::VectorIoStats,
    scan_ns: u64,
    boundary_ns: u64,
    scored: usize,
    survivors: usize,
}

/// Candidate identities at quantized stage boundaries.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct QuantizedStageTrace {
    pub(crate) scored_docs: Vec<DocId>,
    pub(crate) estimates: Vec<Vec<(usize, DocId, u32)>>,
    pub(crate) boundary_docs: Vec<Vec<DocId>>,
    pub(crate) rerank_docs: Vec<DocId>,
    pub(super) scored_rows: Vec<usize>,
    pub(super) boundary_rows: Vec<Vec<usize>>,
    pub(super) estimate_rows: Vec<Vec<(usize, u32)>>,
}

#[cfg(test)]
impl QuantizedStageTrace {
    /// Translates recorded rows after the probe, without storage reads in trace hooks.
    pub(super) fn translate(&mut self, docs: &[DocId]) {
        let translate = |rows: &[usize]| {
            let mut result: Vec<_> = rows.iter().map(|&row| docs[row]).collect();
            result.sort_unstable();
            result
        };
        self.scored_docs = translate(&self.scored_rows);
        self.boundary_docs = self
            .boundary_rows
            .iter()
            .map(|rows| translate(rows))
            .collect();
        self.estimates = self
            .estimate_rows
            .iter()
            .map(|trace| {
                trace
                    .iter()
                    .map(|&(row, bits)| (row, docs[row], bits))
                    .collect()
            })
            .collect();
    }
}

/// Sparse per-layer instrumentation.
#[derive(Debug, Default)]
pub struct LayerProbeStatsSet(pub(super) Vec<LayerProbeStats>);

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
            map.serialize_entry(
                &format!("layer{index}_sign_word_fallbacks"),
                &layer.sign_word_fallbacks,
            )?;
            serialize_prefixed(&layer.io, &format!("layer{index}"), &mut map)?;
            map.serialize_entry(&format!("layer{index}_scan_ns"), &layer.scan_ns)?;
            map.serialize_entry(&format!("layer{index}_scored"), &layer.scored)?;
            map.serialize_entry(&format!("layer{index}_survivors"), &layer.survivors)?;
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
}

fn serialize_prefixed<M>(
    io: &crate::vector::VectorIoStats,
    prefix: &str,
    map: &mut M,
) -> Result<(), M::Error>
where
    M: serde::ser::SerializeMap,
{
    map.serialize_entry(&format!("{prefix}_reads"), &io.reads)?;
    map.serialize_entry(&format!("{prefix}_bytes_read"), &io.bytes_read)?;
    map.serialize_entry(&format!("{prefix}_storage_blocks"), &io.storage_blocks)
}

fn serialize_rerank_io<S>(
    io: &crate::vector::VectorIoStats,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;

    let mut map = serializer.serialize_map(Some(3))?;
    serialize_prefixed(io, "rerank", &mut map)?;
    map.end()
}

/// Per-segment probe-loop instrumentation: a prune breakdown of every
/// doc the inner loop touched, plus posting-fetch counters. Returned by
/// [`VectorBackend::top_n`](crate::vector::backend::VectorBackend::top_n) alongside the hits. The
/// flat/exact path fills only `exact_rows_read`; every other field is IVF-probe-only.
#[derive(Debug, Default, serde::Serialize)]
pub struct ProbeStats {
    /// Actual row-fetch requests during exact reranking.
    #[serde(flatten, serialize_with = "serialize_rerank_io")]
    pub rerank_io: crate::vector::VectorIoStats,
    /// Docs that passed filter + alive + seen and were scored against the
    /// query. This stays the "scored" bucket and equals the final survivor
    /// `candidates`.
    pub candidates_scored: usize,
    /// Eligible layer-0 posting rows.
    pub layer0_eligible: usize,
    /// Routed clusters skipped because no row was eligible.
    pub clusters_skipped_empty: usize,
    /// Eligible rows charged against the probe budget.
    pub eligible_charged: usize,
    /// Sparse layer-indexed timing and funnel counters, flattened on the wire.
    #[serde(flatten)]
    pub layers: LayerProbeStatsSet,
    #[cfg(test)]
    #[serde(skip)]
    pub(crate) quantized_trace: QuantizedStageTrace,
    /// Distinct documents fetched for exact rerank.
    pub rerank_rows: usize,
    /// Scan initialization time.
    pub scan_init_ns: u64,
    /// Predicate evaluation and filter-bitset construction time.
    pub non_vector_search_ns: u64,
    /// Segment-query rotation, bitplane, and LUT preparation time.
    pub query_prep_ns: u64,
    /// Lazy cluster-routing time, including every ranked-stream pull.
    pub routing_ns: u64,
    /// Exact scan time for an unquantized or explicitly exact path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_scan_ns: Option<u64>,
    /// Result assembly time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_assembly_ns: Option<u64>,
    /// Rerank storage ordering plus exact-row fetch time.
    pub rerank_fetch_ns: u64,
    /// Exact score, tie-break, and heap time for the rerank set.
    pub rerank_score_ns: u64,
    /// Every doc-id the inner loop touched, before any gate — the denominator
    /// for the prune breakdown.
    /// Documents visited before pruning.
    pub vectors_visited: usize,
    /// Touched docs rejected by `filter.contains`.
    pub pruned_filter: usize,
    /// Touched docs rejected by `is_alive`.
    pub pruned_dead: usize,
    /// Probed clusters whose surviving rows' posting bytes were fetched —
    /// one stride-sized ranged read per surviving row. Counts clusters,
    /// not rows.
    /// Probed clusters with fetched posting rows.
    pub postings_row: usize,
    /// Probed clusters that fetched no posting bytes at all: the
    /// `filter → alive → seen` pre-pass left zero survivors (fully
    /// filtered / dead / already-seen, or the cluster is empty). The two
    /// `postings_*` counters partition the probed clusters:
    /// [`clusters_probed`](Self::clusters_probed) `== postings_row + postings_skipped`.
    /// Probed clusters without fetched posting rows.
    pub postings_skipped: usize,
    /// Flat/exact-path stride-sized row reads — one per survivor scored.
    /// Filled only by the exact (non-IVF) path.
    /// Full-precision row reads performed by the exact path.
    pub exact_rows_read: usize,
    /// Cluster-routing counters.
    #[serde(skip)]
    pub routing: Option<RouterMetrics>,
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
    /// Clusters rejected by the bounds gate.
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
    /// Work units charged by the probe loop.
    pub work_charged: f32,
    /// The resolved work budget the probe loop ran against: the
    /// `max_probe_fraction` ceiling in work units, floored by
    /// `min_probe_clusters`. `0` when no IVF probe loop ran.
    pub work_budget: f32,
    /// The APS recall estimate when the probe loop stopped; absent when
    /// APS was off or the heap never held `k` results. Per-segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_estimate: Option<f32>,
    /// Vector rows in the segment.
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
            .saturating_add(self.non_vector_search_ns)
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

    pub(super) fn record_routing(&mut self, routing: RouterMetrics) {
        self.routing = Some(routing);
        self.routing_visited_count += match routing {
            RouterMetrics::Rng(graph) => graph.visited_count,
            RouterMetrics::Stacked {
                candidate_count, ..
            } => candidate_count,
            RouterMetrics::Exact { visited_count } => visited_count,
        };
        if let RouterMetrics::Rng(graph) = routing {
            self.routing_graph_count += 1;
            self.routing_graph_visited_count += graph.visited_count;
            self.routing_graph_expanded_count += graph.expanded_count;
            self.routing_graph_edges_scanned += graph.edges_scanned;
            self.routing_graph_evictions += graph.evictions;
            self.routing_graph_result_count += graph.result_count;
        }
    }

    pub(super) fn record_bound_armed(&mut self, at_probe: Option<u32>) {
        if let Some(probe) = at_probe {
            self.bound_armed_count += 1;
            self.bound_armed_probe_sum += u64::from(probe);
        }
    }

    pub(super) fn start_layer(&mut self, layer: usize) {
        self.layers.layer_mut(layer);
    }

    pub(super) fn record_layer_scan(&mut self, layer: usize, scored: usize, elapsed_ns: u64) {
        let stats = self.layers.layer_mut(layer);
        stats.scored += scored;
        stats.scan_ns += elapsed_ns;
    }

    pub(super) fn record_boundary(&mut self, layer: usize, survivors: usize, elapsed_ns: u64) {
        let stats = self.layers.layer_mut(layer);
        stats.survivors += survivors;
        stats.boundary_ns += elapsed_ns;
    }

    pub(crate) fn clear_stage_timings(&mut self) {
        self.scan_init_ns = 0;
        self.non_vector_search_ns = 0;
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
    /// Returns the probed-cluster count.
    #[inline]
    pub fn clusters_probed(&self) -> usize {
        self.postings_row + self.postings_skipped
    }
}
