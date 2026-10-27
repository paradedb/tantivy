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
use std::sync::Arc;

use self::bound::{PruningBound, TopK};
use self::columns::{Columns, QuantizedCandidates};
use self::estimate::{combine_initial, combine_refine, CombineCtx};
use self::exact::ExactScorer;
use self::heap::TieBreakHits;
use self::layer_read::{decode_layer, Decoded, LayerRead, LayerSpec};
#[cfg(test)]
use self::plan::{forced_plan, PlanOverride};
use self::plan::{price, Read, ReadCosts};
use crate::collector::sort_key::Comparator;
use crate::collector::{SegmentSortKeyComputer, SharedThreshold};
use crate::error::DataCorruption;
use crate::schema::Metric;
use crate::vector::distance::Similarity;
use crate::vector::search::cluster_source::{ClusterRows, ClusterSource, Feedback};
use crate::vector::search::interval::Threshold;
use crate::vector::search::prepared::{QuantizedQueryCtx, ScoringQuery};
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::quantized::{QuantizedFieldReader, ReadScratch};
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::enter_vector_stage;
use crate::vector::{VectorElement, VectorStage};
use crate::{Score, SegmentOrdinal};

// --- Contracts ---

/// Sizes for one segment scan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanConfig {
    pub(crate) top_n: usize,
    /// Upper bound on rows admitted, for sizing the candidate columns.
    pub(crate) capacity: usize,
    /// Whether a cluster's rows may be scored exactly before the exact phase.
    pub(crate) exact_plan: bool,
    #[cfg(test)]
    pub(crate) plan_override: Option<PlanOverride>,
    /// Disables layer-0 admission pruning, which never changes results; equivalence tests flip it.
    #[cfg(test)]
    pub(crate) admission_pruning_disabled: bool,
}

/// Per-layer counts of a quantized scan.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LayerReport {
    /// Candidates the layer scored.
    pub(crate) scored: usize,
    /// Rows dropped at admission because their interval could no longer reach the running top-k;
    /// layer 0 only.
    pub(crate) admission_pruned: usize,
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
    /// Survivors of the previous boundary dropped before this layer read them, because the
    /// pruning threshold rose past their upper endpoint; refining layers only.
    pub(crate) recheck_dropped: usize,
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
    /// Rows shortlisted from the source.
    pub(crate) candidates_scored: usize,
    /// One per active layer on a quantized segment; empty at full precision.
    pub(crate) layers: Vec<LayerReport>,
    /// Rows scored at full precision.
    pub(crate) exact_rows: usize,
    /// Survivors of the last boundary dropped before the exact phase read them.
    pub(crate) exact_recheck_dropped: usize,
    /// Thresholds the scan stored in the participants' slot.
    pub(crate) cross_bound_publishes: usize,
    /// Slot reads that raised the scan's threshold.
    pub(crate) cross_bound_raises: usize,
    /// Centroid rows read for shortlisted clusters without a centroid score.
    pub(crate) centroid_reads: usize,
    #[cfg(test)]
    pub(crate) trace: Box<ScanTrace>,
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
    /// Every exact score the scan computed, in order.
    pub(crate) exact_scores: Vec<f32>,
    /// The bound's known exact scores at the end of the scan.
    pub(crate) known_exact: Vec<f32>,
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
    /// Scratch for one cluster's combined columns while admission filters them.
    admission_columns: Columns,
    /// Scratch for the admitted row indices of one cluster.
    admission_keep: Vec<u32>,
    /// Rows the layer-0 kernel scored, before admission pruning.
    layer0_scored: usize,
    /// Rows admission pruning dropped.
    layer0_admission_pruned: usize,
    /// The reader exact reads are priced through; `None` with the exact plan off.
    exact: Option<&'a VectorIndexReader>,
    index: &'a ClusterIndex,
    /// Centroid rows read for clusters without a centroid score.
    centroid_reads: usize,
    #[cfg(test)]
    plan_override: Option<PlanOverride>,
    #[cfg(test)]
    admission_pruning_disabled: bool,
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
            admission_columns: Columns::default(),
            admission_keep: Vec::new(),
            layer0_scored: 0,
            layer0_admission_pruned: 0,
            exact: config.exact_plan.then_some(reader),
            index,
            centroid_reads: 0,
            #[cfg(test)]
            plan_override: config.plan_override,
            #[cfg(test)]
            admission_pruning_disabled: config.admission_pruning_disabled,
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

    /// Scores a cluster's rows on layer 0, read as `read`, and appends the rows whose interval
    /// can still reach the pruning threshold as a candidate block. The threshold only rises, so
    /// the rows dropped here are rows every later boundary would drop.
    fn admit(
        &mut self,
        rows: &ClusterRows<'_>,
        read: LayerRead,
        bound: &mut PruningBound,
        scratch: &mut ReadScratch,
    ) -> crate::Result<()> {
        let centroid = match rows.centroid {
            Some(centroid) => centroid,
            None => self.read_centroid(rows.cluster)?,
        };
        // The threshold as of the clusters admitted so far, before this cluster's rows enter.
        let prior = self.admission_threshold(bound);
        let spec = self.layers[0];
        decode_layer(&spec, rows, read, &mut self.decoded, scratch)?;
        self.layer0_scored += rows.len();
        let ctx = self.combine_ctx(0, centroid);
        match prior {
            Some(threshold) => {
                combine_initial(
                    &ctx,
                    &self.decoded,
                    &mut self.admission_columns,
                    0..rows.len(),
                );
                bound.admit(
                    rows,
                    &self.admission_columns,
                    self.candidates.kappa(),
                    threshold,
                    &mut self.admission_keep,
                );
                let at = self.candidates.push_selected(
                    rows,
                    centroid,
                    &self.admission_columns,
                    &self.admission_keep,
                );
                self.layer0_admission_pruned += rows.len() - at.len();
            }
            None => {
                let at = self.candidates.push_block(rows, centroid);
                combine_initial(
                    &ctx,
                    &self.decoded,
                    &mut self.candidates.columns,
                    at.clone(),
                );
                bound.merge_cluster(self.candidates.entries(at));
            }
        }
        Ok(())
    }

    /// The threshold admission prunes at; `None` while the bound's is unset or when a test
    /// disables pruning.
    fn admission_threshold(&self, bound: &mut PruningBound) -> Option<Threshold> {
        #[cfg(test)]
        if self.admission_pruning_disabled {
            return None;
        }
        bound.admission_threshold()
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

    /// Drops block `b`'s candidates whose upper endpoint fell below the bound's pruning threshold
    /// since the last boundary, before the block is read; returns the rows dropped.
    fn recheck(&mut self, b: usize, bound: &mut PruningBound) -> usize {
        match bound.recheck_threshold() {
            Some(threshold) => self.candidates.recheck(b, threshold),
            None => 0,
        }
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
    fn prune(&mut self, layer: usize, bound: &mut PruningBound, report: &mut ScanReport) {
        // Layer 0's kernel scores every admitted row; admission pruning drops rows before they
        // become candidates, so the scored count keeps its own counter.
        report.layers[layer].scored = if layer == 0 {
            report.layers[0].admission_pruned = self.layer0_admission_pruned;
            self.layer0_scored
        } else {
            self.candidates.len()
        };
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
        let threshold = bound.boundary(&mut self.candidates);
        self.candidates.band(threshold);
        report.layers[layer].survivors = self.candidates.len();
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
    /// The segment's pruning bound.
    bound: PruningBound,
    /// The quantized cascade; `None` scores every shortlisted row exactly.
    quantized: Option<QuantizedState<'a>>,
    scratch: ReadScratch,
    /// The candidate blocks by cluster id, which is row order; set when the shortlist closes.
    by_cluster: Vec<usize>,
    report: ScanReport,
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
        carried: TopK,
        global: Arc<dyn SharedThreshold<Score>>,
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
                .into());
            }
            (None, _) => None,
        };
        let layers = quantized.as_ref().map_or(0, |state| state.layers.len());
        Ok(Self {
            exact: ExactScorer::new(reader, query.query, config.top_n, tie_break, tie_comparator),
            bound: PruningBound::new(config.top_n, carried, global),
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

    /// The stage the shortlist is built in: layer 0, or the exact scan at full precision.
    fn shortlist_stage(&self) -> VectorStage {
        match self.quantized {
            Some(_) => VectorStage::LayerScan(0),
            None => VectorStage::ExactScan,
        }
    }

    /// The segment-only pruning thresholds as of the last shortlisted cluster.
    ///
    /// Carried exact scores and the shared threshold raise admission, but the router and APS keep
    /// reading this segment's own top-k. Folding the carried scores into either value would change
    /// which clusters open.
    fn feedback(&self) -> Feedback {
        Feedback {
            kth_lower: self.bound.own_lower(),
            kth_estimate: self.bound.own_estimate(),
        }
    }

    /// Orders the candidate blocks by cluster and prunes the shortlist at layer 0's boundary.
    fn close_shortlist(&mut self) {
        let Some(state) = &mut self.quantized else {
            return;
        };
        state.candidates.order_by_cluster(&mut self.by_cluster);
        state.prune(0, &mut self.bound, &mut self.report);
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
                self.report.exact_rows +=
                    self.exact.score(rows, &mut self.scratch, &mut self.bound)?;
            }
            Read::Layer(read) => {
                if let Some(state) = &mut self.quantized {
                    state.admit(rows, read, &mut self.bound, &mut self.scratch)?;
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
    /// at the layer's boundary. Each block first drops the candidates the pruning threshold has
    /// risen past, so its read is planned and priced on the rest. A block whose survivors are
    /// cheaper to read exactly is scored immediately and retired before the boundary, so no row
    /// counts both as an interval and as an exact score.
    fn narrow(&mut self, layer: usize) -> crate::Result<()> {
        let Some(state) = &mut self.quantized else {
            return Ok(());
        };
        let stage = enter_vector_stage(VectorStage::LayerScan(layer as u8));
        for &b in &self.by_cluster {
            self.report.layers[layer].recheck_dropped += state.recheck(b, &mut self.bound);
            let group = state.candidates.rows(b);
            if group.len() == 0 {
                continue;
            }
            let read = state.read_plan(&group, layer, &mut self.scratch)?;
            self.report.layers[layer].count(read, group.len());
            match read {
                Read::Exact => {
                    self.report.exact_rows +=
                        self.exact
                            .score(&group, &mut self.scratch, &mut self.bound)?;
                    state.candidates.retire(b);
                }
                Read::Layer(read) => state.refine(layer, b, read, &mut self.scratch)?,
            }
        }
        drop(stage);
        state.prune(layer, &mut self.bound, &mut self.report);
        Ok(())
    }

    /// Scores the last boundary's survivors exactly, one block at a time in cluster order, after
    /// dropping each block's candidates the pruning threshold has risen past.
    fn score_shortlist(&mut self) -> crate::Result<()> {
        let Some(state) = &mut self.quantized else {
            return Ok(());
        };
        for &b in &self.by_cluster {
            self.report.exact_recheck_dropped += state.recheck(b, &mut self.bound);
            let group = state.candidates.rows(b);
            if group.len() > 0 {
                self.report.exact_rows +=
                    self.exact
                        .score_survivors(&group, &mut self.scratch, &mut self.bound)?;
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
        let mut report = self.report;
        report.cross_bound_publishes = self.bound.publishes;
        report.cross_bound_raises = self.bound.raises;
        if let Some(state) = &self.quantized {
            report.centroid_reads = state.centroid_reads;
        }
        #[cfg(test)]
        {
            report.trace.exact_phase_rows = self.exact.exact_phase_rows.clone();
            report.trace.exact_scores = self.exact.exact_scores.clone();
            report.trace.known_exact = self.bound.known_exact();
        }
        Ok((self.exact.into_hits(segment_ord), report))
    }
}
