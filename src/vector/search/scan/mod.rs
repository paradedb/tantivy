//! The IVF segment scan: shortlists each cluster a source supplies, narrows the quantized
//! shortlist layer by layer under one pruning bound, and scores what is left exactly.

pub(crate) mod bound;
pub(crate) mod columns;
pub(crate) mod estimate;
mod exact;
pub(crate) mod heap;
pub(crate) mod layer_read;
pub(crate) mod plan;

use std::ops::Range;

use self::bound::KthBound;
use self::columns::QuantizedCandidates;
use self::estimate::{combine_initial, combine_refine, CombineCtx};
use self::exact::ExactScorer;
use self::heap::TieBreakHits;
use self::layer_read::{decode_layer, Decoded, LayerRead, LayerSpec};
#[cfg(test)]
use self::plan::{forced_plan, PlanOverride};
use self::plan::{price, Read, ReadCosts};
use crate::collector::sort_key::Comparator;
use crate::collector::SegmentSortKeyComputer;
use crate::error::DataCorruption;
use crate::schema::Metric;
use crate::vector::distance::Similarity;
use crate::vector::search::cluster_source::{ClusterRows, ClusterSource, Feedback};
use crate::vector::search::interval::{LowerEndpoint, Threshold};
use crate::vector::search::prepared::{QuantizedQueryCtx, ScoringQuery};
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::quantized::{QuantizedFieldReader, ReadScratch};
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::enter_vector_stage;
use crate::vector::{VectorElement, VectorStage};
use crate::SegmentOrdinal;

// --- Contracts ---

/// Sizes for one segment scan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanConfig {
    pub(crate) cluster_cascade: bool,
    pub(crate) cluster_stats: bool,
    pub(crate) deferred_rerank: bool,
    pub(crate) quantized_recall: Option<crate::vector::QuantizedRecallParams>,
    pub(crate) min_probe_clusters: usize,
    pub(crate) cluster_limit: usize,
    pub(crate) top_n: usize,
    /// Upper bound on rows admitted, for sizing the candidate columns.
    pub(crate) capacity: usize,
    /// Whether a cluster's rows may be scored exactly before the exact phase.
    pub(crate) exact_plan: bool,
    #[cfg(test)]
    pub(crate) plan_override: Option<PlanOverride>,
}

/// Per-layer counts of a quantized scan.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LayerReport {
    /// Candidates the layer scored.
    pub(crate) scored: usize,
    /// Candidates left after the layer's boundary.
    pub(crate) survivors: usize,
    /// Clusters read sparsely at this layer.
    pub(crate) sparse_clusters: usize,
    /// Clusters read as a whole band at this layer.
    pub(crate) full_clusters: usize,
    /// Clusters scored exactly, and finished, at this layer.
    pub(crate) exact_clusters: usize,
    /// Rows scored exactly, and finished, at this layer.
    pub(crate) exact_rows: usize,
}

impl LayerReport {
    /// Counts one cluster's `rows` read as `read`.
    fn count(&mut self, read: Read, rows: usize) {
        match read {
            Read::Exact => {
                self.exact_clusters += 1;
                self.exact_rows += rows;
            }
            Read::Layer(LayerRead::Sparse) => self.sparse_clusters += 1,
            Read::Layer(LayerRead::Band) => self.full_clusters += 1,
        }
    }
}

/// What a finished scan reports.
#[derive(Debug, Default)]
pub(crate) struct ScanReport {
    pub(crate) clusters: Vec<CompletedCluster>,
    pub(crate) quantized_recall_stop_at: Option<usize>,
    /// Rows shortlisted from the source.
    pub(crate) candidates_scored: usize,
    /// One per active layer on a quantized segment; empty at full precision.
    pub(crate) layers: Vec<LayerReport>,
    /// Rows scored at full precision.
    pub(crate) exact_rows: usize,
    /// Centroid rows read for shortlisted clusters without a centroid score.
    pub(crate) centroid_reads: usize,
    #[cfg(test)]
    pub(crate) trace: Box<ScanTrace>,
}

/// An online observation after finishing one nonempty quantized cluster.
#[derive(Debug, serde::Serialize)]
pub struct CompletedCluster {
    /// One-based ordinal among completed nonempty clusters.
    pub ordinal: usize,
    /// Segment-local storage cluster.
    pub cluster: usize,
    /// Rows admitted after filters and deletions.
    pub eligible: usize,
    /// Rows scored exactly, including cost-based direct reads.
    pub exact: usize,
    /// Rows scored exactly by the read planner before the final stage.
    pub planned_exact: usize,
    /// Running pruning cutoff after this cluster, absent while filling.
    pub kth: Option<f32>,
    /// Exact top-k document ids in result order.
    pub top_docs: Vec<crate::DocId>,
    pub layer_survivors: Vec<usize>,
    pub pending_docs: Vec<crate::DocId>,
    pub centroid: Option<f32>,
    pub kth_estimate: Option<f32>,
}

/// Rows each stage of a scan scored and kept, by global row.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct ScanTrace {
    /// Rows shortlisted at layer 0.
    pub(crate) scored_rows: Vec<usize>,
    pub(crate) boundary_rows: Vec<Vec<usize>>,
    /// `(row, estimate bits)` after the shortlist and after each refining layer.
    pub(crate) estimate_rows: Vec<Vec<(usize, u32)>>,
    /// `(row, [raw prefix, base, estimate, sigma] bits)` at the same points.
    pub(crate) layer_columns: Vec<Vec<(usize, [u32; 4])>>,
    /// Rows scored in the exact phase.
    pub(crate) exact_phase_rows: Vec<usize>,
}

// --- Quantized state ---

/// The quantized cascade's state: the candidate blocks and what reading and combining them needs.
struct QuantizedState<'a> {
    /// The active layers, in scan order.
    layers: Vec<LayerSpec<'a>>,
    /// The widened query: what the router scores centroids with.
    widened: &'a [f32],
    metric: Metric,
    dimension: usize,
    candidates: QuantizedCandidates,
    decoded: Decoded,
    /// The reader exact reads are priced through; `None` with the exact plan off.
    exact: Option<&'a VectorIndexReader>,
    index: &'a ClusterIndex,
    /// Centroid rows read for clusters without a centroid score.
    centroid_reads: usize,
    #[cfg(test)]
    plan_override: Option<PlanOverride>,
}

impl<'a> QuantizedState<'a> {
    fn new(
        query: &'a QuantizedQueryCtx,
        widened: &'a [f32],
        field: &'a QuantizedFieldReader,
        reader: &'a VectorIndexReader,
        index: &'a ClusterIndex,
        config: &ScanConfig,
    ) -> Self {
        let meta = query.index.meta.field();
        Self {
            layers: (0..query.active_layers())
                .map(|index| LayerSpec {
                    query,
                    reader: &field.layers()[index],
                    index,
                    query_error_squared: query.query_error_squared(index) as f32,
                })
                .collect(),
            widened,
            metric: meta.metric,
            dimension: meta.dim as usize,
            candidates: QuantizedCandidates::with_capacity(config.capacity),
            decoded: Decoded::default(),
            exact: config.exact_plan.then_some(reader),
            index,
            centroid_reads: 0,
            #[cfg(test)]
            plan_override: config.plan_override,
        }
    }

    /// The plan for `costs` of `selected` at `layer`, or a test's override.
    fn choose(&self, costs: &ReadCosts, selected: &ClusterRows<'_>, layer: usize) -> Read {
        #[cfg(test)]
        return forced_plan(costs, self.plan_override, selected.cluster, layer);
        #[cfg(not(test))]
        {
            let _ = (selected, layer);
            plan::plan(costs)
        }
    }

    /// Prices the rows of `selected` at `layer`, and exactly when the exact plan is on, and plans
    /// their read. A sparse plan leaves its rows and code runs in `scratch` for the read.
    fn read_plan(
        &self,
        selected: &ClusterRows<'_>,
        layer: usize,
        scratch: &mut ReadScratch,
    ) -> crate::Result<Read> {
        let costs = price(&self.layers[layer], self.exact, selected, scratch)?;
        Ok(self.choose(&costs, selected, layer))
    }

    /// How `layer`'s decoded rows combine into estimates around `centroid`, the query's centroid
    /// score.
    fn combine_ctx(&self, layer: usize, centroid: Similarity) -> CombineCtx {
        let spec = &self.layers[layer];
        let cluster_score = centroid.score();
        let query_norm = spec.query.score_query_norm(cluster_score);
        CombineCtx {
            metric: self.metric,
            dimension: self.dimension,
            query_norm_squared: query_norm * query_norm,
            query_error_squared: spec.query_error_squared,
            cluster_score,
        }
    }

    /// Scores a cluster's rows on layer 0, read as `read`, and appends them as a candidate block.
    fn admit(
        &mut self,
        rows: &ClusterRows<'_>,
        read: LayerRead,
        bound: &mut KthBound,
        scratch: &mut ReadScratch,
    ) -> crate::Result<()> {
        let centroid = match rows.centroid {
            Some(centroid) => centroid,
            None => self.read_centroid(rows.cluster)?,
        };
        let spec = self.layers[0];
        decode_layer(&spec, rows, read, &mut self.decoded, scratch)?;
        let ctx = self.combine_ctx(0, centroid);
        let at = self.candidates.push_block(rows, centroid);
        combine_initial(
            &ctx,
            &self.decoded,
            &mut self.candidates.columns,
            at.clone(),
        );
        self.candidates.finish_cluster(at, bound);
        Ok(())
    }

    /// Refines block `b`'s candidates with `layer`, read as `read`.
    fn refine(
        &mut self,
        layer: usize,
        b: usize,
        read: LayerRead,
        scratch: &mut ReadScratch,
    ) -> crate::Result<()> {
        let spec = self.layers[layer];
        decode_layer(
            &spec,
            &self.candidates.rows(b),
            read,
            &mut self.decoded,
            scratch,
        )?;
        let block = self.candidates.block(b);
        let (centroid, at) = (block.centroid, block.at.clone());
        let ctx = self.combine_ctx(layer, centroid);
        combine_refine(&ctx, &self.decoded, &mut self.candidates.columns, at);
        Ok(())
    }

    /// The query's similarity to `cluster`'s centroid, from its stored row, scored as the router
    /// scores it.
    fn read_centroid(&mut self, cluster: usize) -> crate::Result<Similarity> {
        let _centroid = enter_vector_stage(VectorStage::Centroid);
        let row = self.index.centroid_row(cluster)?;
        self.centroid_reads += 1;
        Ok(self.metric.similarity_bytes::<f32>(self.widened, &row))
    }

    /// Records the candidates `layer` scored and prunes them at its boundary.
    fn prune(&mut self, layer: usize, bound: &mut KthBound, report: &mut ScanReport) {
        report.layers[layer].scored += self.candidates.len();
        #[cfg(test)]
        {
            // Layer 0's candidates are listed in admission order, later layers' in row order.
            let row_order = layer > 0;
            report
                .trace
                .estimate_rows
                .push(self.candidates.estimate_trace(row_order));
            report
                .trace
                .layer_columns
                .push(self.candidates.column_trace(row_order));
        }
        let _boundary = enter_vector_stage(VectorStage::Boundary(layer as u8));
        self.candidates.band(bound);
        report.layers[layer].survivors += self.candidates.len();
        #[cfg(test)]
        report
            .trace
            .boundary_rows
            .push(self.candidates.global_rows());
    }
}

// --- Segment scan ---

/// One segment's IVF scan. A cluster source supplies each cluster's rows; the scan shortlists them
/// under the segment's quantization, keeps the pruning bound, and scores what is left exactly.
pub(crate) struct SegmentScan<'a, T, K, C>
where
    T: VectorElement,
    K: SegmentSortKeyComputer,
    C: Comparator<K::SegmentSortKey>,
{
    exact: ExactScorer<'a, T, K, C>,
    /// The quantized cascade; `None` scores every shortlisted row exactly.
    quantized: Option<QuantizedState<'a>>,
    scratch: ReadScratch,
    /// The candidate blocks by cluster id, which is row order; set when the shortlist closes.
    by_cluster: Vec<usize>,
    report: ScanReport,
    cluster_cascade: bool,
    cluster_stats: bool,
    deferred_rerank: bool,
    quantized_recall_gap: Option<f64>,
    min_probe_clusters: usize,
    cluster_limit: usize,
}

impl<'a, T, K, C> SegmentScan<'a, T, K, C>
where
    T: VectorElement,
    K: SegmentSortKeyComputer,
    C: Comparator<K::SegmentSortKey>,
{
    /// Fails with `DataCorruption` when a quantized query meets a segment without quantized slots.
    pub(crate) fn new(
        reader: &'a VectorIndexReader,
        index: &'a ClusterIndex,
        query: ScoringQuery<'a, T>,
        config: ScanConfig,
        tie_break: &'a mut K,
        tie_comparator: C,
    ) -> crate::Result<Self> {
        let quantized = match (query.quantized, reader.quantization()) {
            (Some(quantized), Some(field)) => Some(QuantizedState::new(
                quantized,
                query.query.widened(),
                field,
                reader,
                index,
                &config,
            )),
            (Some(_), None) => {
                return Err(DataCorruption::comment_only(
                    "quantized query requires quantized slots",
                )
                .into())
            }
            (None, _) => None,
        };
        let layers = quantized.as_ref().map_or(0, |state| state.layers.len());
        Ok(Self {
            cluster_cascade: config.cluster_cascade,
            cluster_stats: config.cluster_stats,
            deferred_rerank: config.deferred_rerank,
            quantized_recall_gap: config
                .quantized_recall
                .and_then(|p| p.gap_threshold(config.top_n)),
            min_probe_clusters: config.min_probe_clusters,
            cluster_limit: config.cluster_limit,
            exact: ExactScorer::new(reader, query.query, config.top_n, tie_break, tie_comparator),
            quantized,
            scratch: ReadScratch::default(),
            by_cluster: Vec::new(),
            report: ScanReport {
                layers: vec![LayerReport::default(); layers],
                ..ScanReport::default()
            },
        })
    }

    /// Scores one segment: shortlists each cluster `source` supplies, narrows the shortlist one
    /// quantized layer at a time, then scores what is left exactly.
    pub(crate) fn run(&mut self, source: &mut impl ClusterSource) -> crate::Result<()> {
        if self.deferred_rerank && self.quantized.is_some() {
            return self.run_deferred(source);
        }
        if self.cluster_cascade && self.quantized.is_some() {
            return self.run_clusters(source);
        }
        let stage = enter_vector_stage(self.shortlist_stage());
        while let Some(rows) = source.next(self.feedback())? {
            let scored = self.shortlist(&rows)?;
            source.admitted(scored, self.feedback())?;
        }
        drop(stage);
        self.close_shortlist();
        for layer in self.refining_layers() {
            self.narrow(layer)?;
        }
        self.score_shortlist()
    }

    fn run_deferred(&mut self, source: &mut impl ClusterSource) -> crate::Result<()> {
        let mut completed = 0;
        let mut completed_bound = KthBound::new(self.exact.bound.k);
        let mut survivors = Vec::new();
        while let Some(rows) = source.next(self.feedback())? {
            let scored = {
                let _stage = enter_vector_stage(self.shortlist_stage());
                self.shortlist(&rows)?
            };
            let state = self.quantized.as_mut().unwrap();
            let block = state.candidates.block_count() - 1;
            survivors.clear();
            for layer in 0..state.layers.len() {
                let _stage = enter_vector_stage(VectorStage::LayerScan(layer as u8));
                let count = state.candidates.rows(block).len();
                if layer > 0 && count > 0 {
                    let read =
                        state.read_plan(&state.candidates.rows(block), layer, &mut self.scratch)?;
                    self.report.layers[layer].count(read, count);
                    let Read::Layer(read) = read else {
                        unreachable!("deferred reranking disables direct exact reads");
                    };
                    state.refine(layer, block, read, &mut self.scratch)?;
                }
                if layer > 0 {
                    self.exact.bound.copy_running_from(&completed_bound);
                    let at = state.candidates.block(block).at.clone();
                    state.candidates.finish_cluster(at, &mut self.exact.bound);
                }
                self.report.layers[layer].scored += count;
                state.candidates.band_last(&self.exact.bound);
                let count = state.candidates.rows(block).len();
                self.report.layers[layer].survivors += count;
                if self.cluster_stats {
                    survivors.push(count);
                }
            }
            completed_bound.copy_running_from(&self.exact.bound);
            completed += 1;
            if self.cluster_stats {
                let mut pending_docs = Vec::new();
                let threshold = self
                    .exact
                    .bound
                    .running_lower()
                    .map(|kth| Threshold(LowerEndpoint(kth)));
                for b in 0..state.candidates.block_count() {
                    let selected = state.candidates.rows(b);
                    if selected.len() > 0 {
                        let docs = self.exact.selected_docs(&selected)?;
                        pending_docs.extend(
                            docs.into_iter()
                                .zip(state.candidates.block(b).at.clone())
                                .filter(|&(_, index)| {
                                    threshold
                                        .is_none_or(|kth| kth.admits(state.candidates.upper(index)))
                                })
                                .map(|(doc, _)| doc),
                        );
                    }
                }
                self.report.clusters.push(CompletedCluster {
                    ordinal: completed,
                    cluster: rows.cluster,
                    eligible: scored,
                    exact: 0,
                    planned_exact: 0,
                    kth: self.exact.bound.running_lower(),
                    kth_estimate: self.exact.bound.running_estimate(),
                    top_docs: Vec::new(),
                    pending_docs,
                    layer_survivors: survivors.clone(),
                    centroid: rows.centroid.map(|c| c.score()),
                });
            }
            source.admitted(scored, self.feedback())?;
            let recall_reached = completed >= self.min_probe_clusters
                && self.quantized_recall_gap.is_some_and(|threshold| {
                    self.exact
                        .bound
                        .running_estimate()
                        .and_then(|cutoff| source.quantized_recall_score(cutoff, completed))
                        .is_some_and(|score| score >= threshold)
                });
            if recall_reached {
                self.report.quantized_recall_stop_at = Some(completed);
                break;
            }
            if self.cluster_limit > 0 && completed >= self.cluster_limit {
                break;
            }
        }
        let state = self.quantized.as_mut().unwrap();
        state.candidates.band(&mut self.exact.bound);
        state.candidates.order_by_cluster(&mut self.by_cluster);
        self.score_shortlist()
    }

    fn run_clusters(&mut self, source: &mut impl ClusterSource) -> crate::Result<()> {
        while let Some(rows) = source.next(self.feedback())? {
            let cluster = rows.cluster;
            let exact_before = self.report.exact_rows;
            let planned_before: usize = self.report.layers.iter().map(|l| l.exact_rows).sum();
            let scored = {
                let _stage = enter_vector_stage(self.shortlist_stage());
                self.shortlist(&rows)?
            };
            self.close_shortlist();
            for layer in self.refining_layers() {
                self.narrow(layer)?;
            }
            self.score_shortlist()?;
            let exact = self.report.exact_rows - exact_before;
            let state = self.quantized.as_mut().unwrap();
            state.candidates.clear();
            self.exact.bound.boundary_lower(&mut []);
            let mut feedback = self.feedback();
            feedback.exact_rows = Some(exact);
            if self.cluster_stats {
                self.report.clusters.push(CompletedCluster {
                    ordinal: self.report.clusters.len() + 1,
                    cluster,
                    eligible: scored,
                    exact,
                    planned_exact: self
                        .report
                        .layers
                        .iter()
                        .map(|l| l.exact_rows)
                        .sum::<usize>()
                        - planned_before,
                    kth: feedback.kth_lower,
                    top_docs: self.exact.top_docs(),
                    layer_survivors: Vec::new(),
                    pending_docs: Vec::new(),
                    centroid: rows.centroid.map(|c| c.score()),
                    kth_estimate: feedback.kth_estimate,
                });
            }
            source.admitted(scored, feedback)?;
        }
        Ok(())
    }

    /// The stage the shortlist is built in: layer 0, or the exact scan at full precision.
    fn shortlist_stage(&self) -> VectorStage {
        match self.quantized {
            Some(_) => VectorStage::LayerScan(0),
            None => VectorStage::ExactScan,
        }
    }

    /// The pruning thresholds as of the last shortlisted cluster.
    fn feedback(&self) -> Feedback {
        Feedback {
            exact_rows: None,
            kth_lower: self.exact.bound.running_lower(),
            kth_estimate: self.exact.bound.running_estimate(),
        }
    }

    /// Orders the candidate blocks by cluster and prunes the shortlist at layer 0's boundary.
    fn close_shortlist(&mut self) {
        let Some(state) = &mut self.quantized else {
            return;
        };
        state.candidates.order_by_cluster(&mut self.by_cluster);
        state.prune(0, &mut self.exact.bound, &mut self.report);
    }

    /// The quantized layers after layer 0; none at full precision.
    fn refining_layers(&self) -> Range<usize> {
        1..self
            .quantized
            .as_ref()
            .map_or(1, |state| state.layers.len())
    }

    /// Adds a cluster's rows to the shortlist: scored immediately at full precision or when an
    /// exact read is strictly cheaper than the layer-0 read, otherwise as quantized candidates.
    /// Returns the rows scored.
    fn shortlist(&mut self, rows: &ClusterRows<'_>) -> crate::Result<usize> {
        let len = rows.len();
        debug_assert!(len > 0, "empty clusters never reach the scan");
        #[cfg(test)]
        self.report.trace.scored_rows.extend(rows.global_rows());
        match self.read_plan(rows, 0)? {
            Read::Exact => {
                self.report.exact_rows += self.exact.score(rows, &mut self.scratch)?;
            }
            Read::Layer(read) => {
                if let Some(state) = &mut self.quantized {
                    state.admit(rows, read, &mut self.exact.bound, &mut self.scratch)?;
                }
            }
        }
        self.report.candidates_scored += len;
        Ok(len)
    }

    /// How the rows of `selected` are read at `layer`: exactly at full precision, otherwise as the
    /// layer's costs plan, counted in the layer's report.
    fn read_plan(&mut self, selected: &ClusterRows<'_>, layer: usize) -> crate::Result<Read> {
        let Some(state) = &self.quantized else {
            return Ok(Read::Exact);
        };
        let read = state.read_plan(selected, layer, &mut self.scratch)?;
        self.report.layers[layer].count(read, selected.len());
        Ok(read)
    }

    /// Refines each block's surviving candidates with `layer`, cluster by cluster, then prunes them
    /// at the layer's boundary. A block whose survivors are cheaper to read exactly is scored
    /// immediately and retired before the boundary, so no row counts both as an interval and as
    /// an exact score.
    fn narrow(&mut self, layer: usize) -> crate::Result<()> {
        let Some(state) = &mut self.quantized else {
            return Ok(());
        };
        let stage = enter_vector_stage(VectorStage::LayerScan(layer as u8));
        for &b in &self.by_cluster {
            let group = state.candidates.rows(b);
            if group.len() == 0 {
                continue;
            }
            let read = state.read_plan(&group, layer, &mut self.scratch)?;
            self.report.layers[layer].count(read, group.len());
            match read {
                Read::Exact => {
                    self.report.exact_rows += self.exact.score(&group, &mut self.scratch)?;
                    state.candidates.retire(b);
                }
                Read::Layer(read) => state.refine(layer, b, read, &mut self.scratch)?,
            }
        }
        drop(stage);
        state.prune(layer, &mut self.exact.bound, &mut self.report);
        Ok(())
    }

    /// Scores the last boundary's survivors exactly, one block at a time in cluster order.
    fn score_shortlist(&mut self) -> crate::Result<()> {
        let Some(state) = &self.quantized else {
            return Ok(());
        };
        for &b in &self.by_cluster {
            let group = state.candidates.rows(b);
            if group.len() > 0 {
                self.report.exact_rows += self.exact.score_survivors(&group, &mut self.scratch)?;
            }
        }
        Ok(())
    }

    /// Returns the hits and the scan's report.
    pub(crate) fn finish(
        self,
        segment_ord: SegmentOrdinal,
    ) -> crate::Result<(TieBreakHits<K>, ScanReport)> {
        let _assembly = enter_vector_stage(VectorStage::ResultAssembly);
        let Self {
            exact,
            quantized,
            mut report,
            ..
        } = self;
        if let Some(state) = &quantized {
            report.centroid_reads = state.centroid_reads;
        }
        #[cfg(test)]
        {
            report.trace.exact_phase_rows = exact.exact_phase_rows.clone();
        }
        Ok((exact.into_hits(segment_ord), report))
    }
}
