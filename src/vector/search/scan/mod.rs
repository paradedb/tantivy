//! The IVF segment scan: shortlists each cluster a source supplies, narrows the quantized
//! shortlist layer by layer under one pruning bound, and scores what is left exactly.

pub(crate) mod bound;
pub(crate) mod columns;
pub(crate) mod estimate;
mod exact;
pub(crate) mod heap;
pub(crate) mod histogram;
pub(crate) mod layer_read;
pub(crate) mod plan;
pub(crate) mod rate;

#[cfg(test)]
use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;

use common::BitSet;

use self::bound::{Candidate, PruningBound, TopK};
use self::columns::{Columns, QuantizedCandidates};
use self::estimate::{combine_initial, combine_refine, CombineCtx};
use self::exact::ExactScorer;
use self::heap::TieBreakHits;
use self::histogram::Fold;
use self::layer_read::{decode_layer, Decoded, LayerRead, LayerSpec};
#[cfg(test)]
use self::plan::{forced_plan, PlanOverride};
use self::plan::{price, Read, ReadCosts, SparsePricing};
use self::rate::AdmissionHistograms;
use crate::collector::sort_key::Comparator;
use crate::collector::SegmentSortKeyComputer;
use crate::error::DataCorruption;
use crate::query::{ExcludedDocSet, ExclusionSet, Weight};
use crate::schema::Metric;
use crate::vector::distance::Similarity;
use crate::vector::router::candidate::Candidate as RankedCandidate;
use crate::vector::search::cluster_source::{ClusterRows, ClusterSource};
use crate::vector::search::interval::{Estimate, Threshold, UpperEndpoint};
use crate::vector::search::prepared::{QuantizedQueryCtx, ScoringQuery};
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::quantized::{
    QuantizedClusterReader, QuantizedFieldReader, ReadScratch,
};
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::telemetry::enter_vector_stage;
use crate::vector::{BlockAddress, VectorElement, VectorStage};
use crate::{DocId, SegmentOrdinal, SegmentReader};

const FILTER_BATCH_HEADROOM: usize = 2;
const MAX_FILTER_BATCH_SIZE: usize = 64;

// --- Contracts ---

#[cfg(test)]
thread_local! {
    /// Scores mostly unseen replicated clusters row by row on this thread, so tests can compare
    /// that path with the whole-band one.
    pub(crate) static WHOLE_BAND_ADMISSION_OFF: Cell<bool> = const { Cell::new(false) };
}

/// Sizes for one segment scan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanConfig {
    pub(crate) top_n: usize,
    /// Upper bound on rows admitted, for sizing the candidate columns.
    pub(crate) capacity: usize,
    /// Whether a cluster's rows may be scored exactly before the exact phase.
    pub(crate) exact_plan: bool,
    pub(crate) adaptive_probing: Option<crate::vector::AdaptiveProbeParams>,
    /// Whether the source may hand over a document twice: a routed source on a replicated segment.
    /// A located batch holds one primary row per document.
    pub(crate) source_may_repeat_docs: bool,
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
    /// Clusters planned exact, and finished, at this layer.
    pub(crate) exact_clusters: usize,
    /// Rows planned exact at this layer, excluding copies dropped before scoring.
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
    pub(crate) late_filter: bool,
    pub(crate) filter_candidates: usize,
    pub(crate) filter_batches: usize,
    pub(crate) filter_checks: usize,
    pub(crate) filter_matches: usize,
    pub(crate) router_retries: usize,
    pub(crate) recall_reached: bool,
    /// One per active layer on a quantized segment; empty at full precision.
    pub(crate) layers: Vec<LayerReport>,
    /// Rows scored at full precision.
    pub(crate) exact_rows: usize,
    /// Survivors of the last boundary dropped before the exact phase read them.
    pub(crate) exact_recheck_dropped: usize,
    /// Rows dropped at admission as the second copy of a document an earlier cluster held.
    pub(crate) replication_dropped: usize,
    /// Thresholds the scan stored in the participants' slot.
    pub(crate) cross_bound_publishes: usize,
    /// Centroid rows read for shortlisted clusters without a centroid score.
    pub(crate) centroid_reads: usize,
    #[cfg(test)]
    pub(crate) trace: Box<ScanTrace>,
}

/// Rows each stage of a scan scored and kept, by (block, local row).
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct ScanTrace {
    /// Rows shortlisted at layer 0.
    pub(crate) scored_rows: Vec<BlockAddress>,
    pub(crate) boundary_rows: Vec<Vec<BlockAddress>>,
    /// `(row, estimate bits)` after the shortlist and after each refining layer.
    pub(crate) estimate_rows: Vec<Vec<(BlockAddress, u32)>>,
    /// `(row, [raw prefix, base, estimate, sigma] bits)` at the same points.
    pub(crate) layer_columns: Vec<Vec<(BlockAddress, [u32; 4])>>,
    /// Rows scored in the exact phase.
    pub(crate) exact_phase_rows: Vec<BlockAddress>,
    pub(crate) refined_topk_rows: Vec<BlockAddress>,
    /// Every exact score the scan computed, in order.
    pub(crate) exact_scores: Vec<f32>,
    /// Clusters whose unseen rows were scored from the whole band.
    pub(crate) whole_band_admissions: usize,
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
    /// Scratch for the upper endpoints of one cluster's unseen rows.
    admission_upper: Vec<f32>,
    /// Rows the layer-0 kernel scored, replicas included, before admission pruning.
    layer0_scored: usize,
    late_filter: bool,
    /// Rows admission pruning dropped.
    layer0_admission_pruned: usize,
    /// The reader exact reads are priced through; `None` with the exact plan off.
    exact: Option<&'a VectorIndexReader>,
    index: &'a ClusterIndex,
    segment_ord: SegmentOrdinal,
    /// Centroid rows read for clusters without a centroid score.
    centroid_reads: usize,
    #[cfg(test)]
    whole_band_admissions: usize,
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
        segment_ord: SegmentOrdinal,
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
            admission_upper: Vec::new(),
            layer0_scored: 0,
            late_filter: false,
            layer0_admission_pruned: 0,
            exact: config.exact_plan.then_some(reader),
            index,
            segment_ord,
            centroid_reads: 0,
            #[cfg(test)]
            whole_band_admissions: 0,
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
    /// their read. Returns the plan and the cluster resolved for it, which the read reuses. A
    /// sparse plan leaves its rows and code runs in `scratch` for the read.
    fn read_plan(
        &self,
        selected: &ClusterRows<'_>,
        layer: usize,
        scratch: &mut ReadScratch,
    ) -> crate::Result<(Read, QuantizedClusterReader<'a>)> {
        let cluster = self.layers[layer]
            .reader
            .cluster_in_block(selected.cluster)?;
        let costs = price(
            &cluster,
            self.exact,
            selected,
            scratch,
            self.sparse_pricing(),
        )?;
        Ok((self.choose(&costs, selected, layer), cluster))
    }

    /// How sparse reads are priced: in full when a test forces plans, so a forced sparse read
    /// finds its price.
    fn sparse_pricing(&self) -> SparsePricing {
        #[cfg(test)]
        if self.plan_override.is_some() {
            return SparsePricing::Full;
        }
        SparsePricing::Bounded
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
            width_unknown: spec.query.width_unknown(),
        }
    }

    /// Scores unseen rows on layer 0 and admits those that can reach the pruning threshold. Every
    /// row's survival score goes to `fold`, pruned or not. `handed` is the cluster as the source
    /// handed it, before a replicated segment's seen rows were dropped from `rows`.
    #[allow(clippy::too_many_arguments)]
    fn admit(
        &mut self,
        rows: &ClusterRows<'_>,
        handed: &ClusterRows<'_>,
        read: LayerRead,
        cluster: &QuantizedClusterReader<'_>,
        bound: &mut PruningBound,
        scratch: &mut ReadScratch,
        fold: Option<Fold<'_>>,
    ) -> crate::Result<()> {
        let centroid = match rows.centroid {
            Some(centroid) => centroid,
            None => self.read_centroid(rows.cluster)?,
        };
        // The threshold as of the clusters admitted so far, before this cluster's rows enter.
        let prior = self.admission_threshold(bound);
        if let Some(threshold) = prior.filter(|_| scores_whole_band(rows, handed, read)) {
            let unseen = UnseenRows {
                whole: handed,
                unseen: rows,
            };
            return self.admit_unseen_from_band(
                unseen, cluster, centroid, threshold, bound, scratch, fold,
            );
        }
        let spec = self.layers[0];
        decode_layer(&spec, rows, read, cluster, &mut self.decoded, scratch)?;
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
                if let Some(mut fold) = fold {
                    let columns = &self.admission_columns;
                    fold.upper_endpoints(
                        &columns.estimates[..rows.len()],
                        &columns.sigmas[..rows.len()],
                        self.candidates.kappa(),
                    );
                }
                self.select_admitted(rows, threshold);
                let at = self.candidates.push_selected(
                    rows,
                    centroid,
                    &self.admission_columns,
                    &self.admission_keep,
                );
                self.layer0_admission_pruned += rows.len() - at.len();
                if !self.late_filter {
                    bound.offer(self.candidates.entries(at, self.segment_ord));
                }
            }
            None => {
                let at = self.candidates.push_block(rows, centroid);
                combine_initial(
                    &ctx,
                    &self.decoded,
                    &mut self.candidates.columns,
                    at.clone(),
                );
                if let Some(mut fold) = fold {
                    let columns = &self.candidates.columns;
                    fold.upper_endpoints(
                        &columns.estimates[at.clone()],
                        &columns.sigmas[at.clone()],
                        self.candidates.kappa(),
                    );
                }
                if !self.late_filter {
                    bound.offer(self.candidates.entries(at, self.segment_ord));
                }
            }
        }
        Ok(())
    }

    /// Admits the unseen rows of a whole cluster read as a band: the band is decoded and combined
    /// whole through the contiguous kernel and batch decoders, and a row is kept when it is unseen
    /// and its upper endpoint reaches `threshold`. Each row's estimate, sigma, fold score and
    /// candidate entry are the ones the per-row path computes; the seen rows' sidecars are decoded
    /// and validated too.
    #[allow(clippy::too_many_arguments)]
    fn admit_unseen_from_band(
        &mut self,
        rows: UnseenRows<'_, '_>,
        cluster: &QuantizedClusterReader<'_>,
        centroid: Similarity,
        threshold: Threshold,
        bound: &mut PruningBound,
        scratch: &mut ReadScratch,
        fold: Option<Fold<'_>>,
    ) -> crate::Result<()> {
        let UnseenRows { whole, unseen } = rows;
        let offsets = unseen
            .offsets
            .expect("the unseen rows of a whole cluster carry offsets");
        let spec = self.layers[0];
        decode_layer(
            &spec,
            whole,
            LayerRead::Band,
            cluster,
            &mut self.decoded,
            scratch,
        )?;
        self.layer0_scored += unseen.len();
        let ctx = self.combine_ctx(0, centroid);
        combine_initial(
            &ctx,
            &self.decoded,
            &mut self.admission_columns,
            0..whole.len(),
        );
        let kappa = self.candidates.kappa();
        let columns = &self.admission_columns;
        // Unseen rows only, in row order: the scores and order the per-row path folds.
        self.admission_upper.clear();
        self.admission_upper.extend(offsets.iter().map(|&row| {
            let row = row as usize;
            Estimate(columns.estimates[row])
                .upper(columns.sigmas[row], kappa)
                .0
        }));
        if let Some(mut fold) = fold {
            fold.scores(&self.admission_upper);
        }
        self.admission_keep.clear();
        self.admission_keep.extend(
            offsets
                .iter()
                .zip(&self.admission_upper)
                .filter(|&(_, &upper)| threshold.admits(UpperEndpoint(upper)))
                .map(|(&row, _)| row),
        );
        let at = self.candidates.push_selected(
            whole,
            centroid,
            &self.admission_columns,
            &self.admission_keep,
        );
        self.layer0_admission_pruned += unseen.len() - at.len();
        if !self.late_filter {
            bound.offer(self.candidates.entries(at, self.segment_ord));
        }
        #[cfg(test)]
        {
            self.whole_band_admissions += 1;
        }
        Ok(())
    }

    /// Fills `admission_keep` with rows whose upper endpoint reaches `threshold`.
    fn select_admitted(&mut self, rows: &ClusterRows<'_>, threshold: Threshold) {
        debug_assert_eq!(self.admission_columns.estimates.len(), rows.len());
        let kappa = self.candidates.kappa();
        self.admission_keep.clear();
        for index in 0..rows.len() {
            let estimate = Estimate(self.admission_columns.estimates[index]);
            let sigma = self.admission_columns.sigmas[index];
            let upper = estimate.upper(sigma, kappa);
            if threshold.admits(upper) {
                self.admission_keep.push(index as u32);
            }
        }
    }

    /// The threshold admission prunes at: the shared slot, or `None` when unset / pruning is off.
    fn admission_threshold(&mut self, bound: &mut PruningBound) -> Option<Threshold> {
        #[cfg(test)]
        if self.admission_pruning_disabled {
            return None;
        }
        bound
            .shared_threshold()
            .map(|score| Threshold(crate::vector::search::interval::LowerEndpoint(score)))
    }

    /// Refines block `b`'s candidates with `layer`, read as `read` from `cluster`.
    fn refine(
        &mut self,
        layer: usize,
        b: usize,
        read: LayerRead,
        cluster: &QuantizedClusterReader<'_>,
        scratch: &mut ReadScratch,
    ) -> crate::Result<()> {
        let spec = self.layers[layer];
        decode_layer(
            &spec,
            &self.candidates.rows(b),
            read,
            cluster,
            &mut self.decoded,
            scratch,
        )?;
        let block = self.candidates.block(b);
        let (centroid, at) = (block.centroid, block.at.clone());
        let ctx = self.combine_ctx(layer, centroid);
        combine_refine(&ctx, &self.decoded, &mut self.candidates.columns, at);
        Ok(())
    }

    /// Drops block `b`'s candidates whose upper endpoint fell below the shared slot since the last
    /// boundary, before the block is read; returns the rows dropped.
    fn recheck(&mut self, b: usize, bound: &mut PruningBound) -> usize {
        match bound.shared_threshold() {
            Some(score) => self.candidates.recheck(
                b,
                Threshold(crate::vector::search::interval::LowerEndpoint(score)),
            ),
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
        if layer == 0 {
            report.layers[0].admission_pruned = self.layer0_admission_pruned;
            report.layers[0].scored = self.layer0_scored;
        } else {
            report.layers[layer].scored = self.candidates.len();
        }
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
        if !self.late_filter {
            bound.offer(self.candidates.live_candidates(self.segment_ord));
        }
        let threshold = bound
            .shared_threshold()
            .map(|score| Threshold(crate::vector::search::interval::LowerEndpoint(score)));
        self.candidates.band(threshold);
        report.layers[layer].survivors = self.candidates.len();
        #[cfg(test)]
        report
            .trace
            .boundary_rows
            .push(self.candidates.row_addresses());
    }
}

/// A whole cluster as the source handed it, and its rows not seen in earlier clusters.
#[derive(Clone, Copy)]
struct UnseenRows<'r, 'a> {
    whole: &'r ClusterRows<'a>,
    unseen: &'r ClusterRows<'a>,
}

/// Whether `rows`, the unseen rows of the whole cluster `handed`, are scored from the whole band:
/// the band is read anyway, and with at least three in four rows unseen the contiguous kernel and
/// batch decoders over every row cost less than gathering the unseen ones.
fn scores_whole_band(rows: &ClusterRows<'_>, handed: &ClusterRows<'_>, read: LayerRead) -> bool {
    #[cfg(test)]
    if WHOLE_BAND_ADMISSION_OFF.with(Cell::get) {
        return false;
    }
    read == LayerRead::Band
        && handed.offsets.is_none()
        && rows.offsets.is_some()
        && 4 * rows.len() >= 3 * handed.len()
}

// --- Segment scan ---

/// The admission-rate measurement, armed by the segment plan when the RateAPS is on.
#[derive(Default)]
pub(crate) struct RateApsState {
    /// Survival scores seen before the fill; they size the grid and nothing else.
    prefill: Vec<f32>,
    rows: usize,
    histograms: Option<AdmissionHistograms>,
    next_refinement: usize,
}

impl RateApsState {
    pub(crate) fn refinement_due(&mut self) -> bool {
        if self.next_refinement == 0 || self.rows < self.next_refinement {
            return false;
        }
        self.next_refinement = self.rows.saturating_mul(2);
        true
    }
}

/// The rows of `rows` whose document is not in `seen`, written to `offsets` and `docs`; every
/// row's document is marked in `seen`, and the rows dropped add to `dropped`.
fn unseen_rows<'s>(
    seen: &mut BitSet,
    rows: &ClusterRows<'_>,
    offsets: &'s mut Vec<u32>,
    docs: &'s mut Vec<DocId>,
    dropped: &mut usize,
) -> ClusterRows<'s> {
    let row_docs = rows
        .docs
        .expect("a replicated segment's rows carry their doc ids");
    debug_assert_eq!(row_docs.len(), rows.len());
    offsets.clear();
    docs.clear();
    offsets.resize(row_docs.len(), 0);
    docs.resize(row_docs.len(), 0);
    // Each row is written at the next free slot, which advances only for a document the test-and-
    // set finds unseen, so the loop has no branch on it.
    let mut kept = 0usize;
    let mut keep = |offset: u32, doc: DocId| {
        let fresh = seen.insert(doc);
        offsets[kept] = offset;
        docs[kept] = doc;
        kept += usize::from(fresh);
    };
    match rows.offsets {
        None => {
            for (offset, &doc) in (0u32..).zip(row_docs) {
                keep(offset, doc);
            }
        }
        Some(selected) => {
            for (&offset, &doc) in selected.iter().zip(row_docs) {
                keep(offset, doc);
            }
        }
    }
    *dropped += row_docs.len() - kept;
    offsets.truncate(kept);
    docs.truncate(kept);
    ClusterRows {
        cluster: rows.cluster,
        rows: rows.rows.clone(),
        offsets: Some(&offsets[..]),
        docs: Some(&docs[..]),
        centroid: rows.centroid,
    }
}

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
    /// The documents held by the clusters opened so far, on a routed scan of a replicated segment:
    /// a document's second copy is dropped before scoring, so each document is decided by the
    /// first opened cluster holding it and never reaches the bound or the heap twice.
    seen: Option<BitSet>,
    /// Rows dropped as a document's second copy.
    replication_dropped: usize,
    /// A replicated cluster's rows left after dropping previously encountered documents.
    unseen_offsets: Vec<u32>,
    unseen_docs: Vec<DocId>,
    /// The candidate blocks by cluster id, which is row order; set when the shortlist closes.
    by_cluster: Vec<usize>,
    /// The admission-rate measurement; `None` unless the RateAPS is on.
    rate: Option<RateApsState>,
    recall_fraction: Option<f64>,
    report: ScanReport,
    late_filter: Option<(&'a dyn Weight, &'a SegmentReader)>,
    filter_batch_size: usize,
    reader: &'a VectorIndexReader,
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
        bound: PruningBound,
        segment_ord: SegmentOrdinal,
        tie_break: &'a mut K,
        tie_comparator: C,
    ) -> crate::Result<Self> {
        let quantized = if let Some(quantized) = query.quantized {
            let field = reader.quantization()?.ok_or_else(|| {
                DataCorruption::comment_only("quantized query requires quantized slots")
            })?;
            Some(QuantizedState::new(
                quantized,
                query.query.widened(),
                field,
                reader,
                index,
                segment_ord,
                &config,
            ))
        } else {
            None
        };
        let layers = quantized.as_ref().map_or(0, |state| state.layers.len());
        let recall_fraction = config
            .adaptive_probing
            .filter(|_| config.top_n > 0)
            .map(|p| p.admission_fraction);
        Ok(Self {
            late_filter: None,
            filter_batch_size: 0,
            reader,
            exact: ExactScorer::new(
                reader,
                query.query,
                config.top_n,
                segment_ord,
                tie_break,
                tie_comparator,
            ),
            bound,
            quantized,
            scratch: ReadScratch::default(),
            seen: (reader.replicated() && config.source_may_repeat_docs)
                .then(|| BitSet::with_max_value(reader.max_doc())),
            replication_dropped: 0,
            unseen_offsets: Vec::new(),
            unseen_docs: Vec::new(),
            by_cluster: Vec::new(),
            recall_fraction,
            rate: recall_fraction.map(|_| RateApsState::default()),
            report: ScanReport {
                layers: vec![LayerReport::default(); layers],
                ..ScanReport::default()
            },
        })
    }

    pub(crate) fn with_post_filter(
        mut self,
        filter: Option<(&'a dyn Weight, &'a SegmentReader, f64)>,
    ) -> Self {
        if filter.is_some() {
            self.recall_fraction = None;
            self.rate = None;
        }
        if let Some(state) = &mut self.quantized {
            state.late_filter = filter.is_some();
            if state.late_filter {
                state.exact = None;
            }
        }
        self.report.late_filter = filter.is_some();
        self.late_filter = filter.map(|(weight, reader, selectivity)| {
            self.filter_batch_size = (self.bound.k().saturating_mul(FILTER_BATCH_HEADROOM) as f64
                / selectivity.clamp(f64::EPSILON, 1.0))
            .ceil() as usize;
            (weight, reader)
        });
        self
    }

    pub(crate) fn needs_more_matches(&self) -> bool {
        self.late_filter.is_some() && self.bound.shared_threshold().is_none()
    }

    pub(crate) fn retried_router(&mut self) {
        self.report.router_retries += 1;
    }

    pub(crate) fn swap_topk(&mut self, top: &mut TopK) {
        self.bound.swap_topk(top);
    }

    /// Scores one segment: shortlists each cluster `source` supplies, narrows the shortlist one
    /// quantized layer at a time, then scores what is left exactly.
    pub(crate) fn run(&mut self, source: &mut impl ClusterSource) -> crate::Result<()> {
        let stage = enter_vector_stage(self.shortlist_stage());
        while let Some(rows) = source.next()? {
            let scored = self.shortlist(&rows)?;
            source.admitted(scored)?;
            if self.recall_reached() {
                break;
            }
        }
        drop(stage);
        self.refine_shortlist()
    }

    pub(crate) fn swap_rate(&mut self, rate: &mut Option<RateApsState>) {
        std::mem::swap(&mut self.rate, rate);
    }

    pub(crate) fn recall_reached(&mut self) -> bool {
        let reached = self
            .recall_fraction
            .zip(self.bound.shared_threshold())
            .and_then(|(fraction, bar)| {
                self.rate
                    .as_ref()?
                    .histograms
                    .as_ref()?
                    .reading(bar)
                    .map(|reading| reading.reached(fraction))
            })
            .unwrap_or(false);
        self.report.recall_reached |= reached;
        reached
    }

    pub(crate) fn refine_shortlist(&mut self) -> crate::Result<()> {
        self.close_shortlist();
        if self.late_filter.is_some() {
            return self.filter_shortlist();
        }
        for layer in self.refining_layers() {
            self.narrow(layer)?;
        }
        self.score_shortlist()
    }

    pub(crate) fn refinement_candidates(&self, limit: usize) -> Vec<Candidate> {
        let Some(state) = &self.quantized else {
            return Vec::new();
        };
        if self.late_filter.is_some() {
            return Vec::new();
        }
        let mut candidates = state.candidates.live_candidates(state.segment_ord);
        if let Some(bar) = self.bound.shared_threshold() {
            candidates.retain(|candidate| candidate.bound.estimate() > bar);
        }
        if candidates.len() > limit {
            candidates.select_nth_unstable_by(limit, |a, b| {
                b.bound
                    .estimate()
                    .total_cmp(&a.bound.estimate())
                    .then_with(|| a.address.cmp(&b.address))
            });
            candidates.truncate(limit);
        }
        candidates
    }

    pub(crate) fn refine_topk(&mut self, top: &[Candidate]) -> crate::Result<()> {
        let Some(state) = &mut self.quantized else {
            return Ok(());
        };
        let mut rows: Vec<_> = top
            .iter()
            .filter(|candidate| candidate.address.segment_ord == state.segment_ord)
            .map(|candidate| candidate.address.block_address)
            .collect();
        if rows.is_empty() {
            return Ok(());
        }
        rows.sort_unstable();
        state.candidates.order_by_cluster(&mut self.by_cluster);
        for &b in &self.by_cluster {
            let group = state.candidates.rows(b);
            let offsets: Vec<_> = rows
                .iter()
                .filter(|row| {
                    row.block as usize == group.cluster && group.rows.contains(&(row.row as usize))
                })
                .map(|row| row.row - group.rows.start as u32)
                .filter(|offset| group.offsets.unwrap().binary_search(offset).is_ok())
                .collect();
            if offsets.is_empty() {
                continue;
            }
            let selected = ClusterRows {
                offsets: Some(&offsets),
                docs: None,
                ..group
            };
            let scored = self
                .exact
                .score(&selected, &mut self.scratch, &mut self.bound, None)?;
            self.report.exact_rows += scored;
            #[cfg(test)]
            self.report.trace.refined_topk_rows.extend(
                selected
                    .local_rows()
                    .map(|row| BlockAddress::new(selected.cluster as u32, row as u32)),
            );
            state.candidates.remove_rows(b, &offsets);
        }
        Ok(())
    }

    fn filter_shortlist(&mut self) -> crate::Result<()> {
        let _stage = enter_vector_stage(VectorStage::LateFilter);
        let (weight, reader) = self.late_filter.unwrap();
        let state = self
            .quantized
            .as_mut()
            .expect("post-filtering requires quantization");
        let selection = enter_vector_stage(VectorStage::LateFilterSelect);
        let threshold = self.bound.shared_threshold();
        let mut queue = BinaryHeap::from(
            state
                .candidates
                .candidates_with_upper(state.segment_ord)
                .filter(|(_, _, _, upper)| threshold.is_none_or(|bar| upper.0 >= bar))
                .map(|(block, index, _, upper)| RankedCandidate {
                    sim: Similarity::new(upper.0),
                    node: Reverse((block, index)),
                })
                .collect::<Vec<_>>(),
        );
        self.report.filter_candidates += queue.len();
        drop(selection);
        let mut batch = Vec::new();
        let mut indexed = Vec::new();
        let mut offsets = Vec::new();
        let mut indices = Vec::new();
        let mut doc_cache = HashMap::<usize, Vec<DocId>>::new();
        let batch_size = self.filter_batch_size.clamp(1, MAX_FILTER_BATCH_SIZE);
        while queue.peek().is_some_and(|next| {
            self.bound
                .shared_threshold()
                .is_none_or(|bar| next.sim.score() >= bar)
        }) {
            let selection = enter_vector_stage(VectorStage::LateFilterSelect);
            batch.clear();
            let threshold = self.bound.shared_threshold();
            while batch.len() < batch_size {
                let Some(next) = queue.peek() else {
                    break;
                };
                if threshold.is_some_and(|bar| next.sim.score() < bar) {
                    break;
                }
                batch.push(queue.pop().unwrap().node.0);
            }
            drop(selection);
            {
                let _sort = enter_vector_stage(VectorStage::LateFilterSort);
                batch.sort_unstable();
            }
            indexed.clear();
            for group in batch.chunk_by(|a, b| a.0 == b.0) {
                let b = group[0].0;
                let block = state.candidates.block(b);
                let cluster = block.cluster;
                let centroid = block.centroid;
                let first = block.at.start;
                let rows = state.candidates.rows(b);
                offsets.clear();
                offsets.extend(group.iter().map(|&(_, i)| rows.offsets.unwrap()[i - first]));
                let selected = ClusterRows {
                    offsets: Some(&offsets),
                    ..rows
                };
                state.admission_columns.truncate(0);
                indices.clear();
                indices.extend(group.iter().map(|&(_, i)| i as u32));
                state
                    .admission_columns
                    .extend_selected(&state.candidates.columns, &indices);
                for layer in 1..state.layers.len() {
                    let _layer = enter_vector_stage(VectorStage::LayerScan(layer as u8));
                    let (read, view) = state.read_plan(&selected, layer, &mut self.scratch)?;
                    let Read::Layer(read) = read else {
                        unreachable!("post-filtering defers exact scoring");
                    };
                    self.report.layers[layer].scored += selected.len();
                    self.report.layers[layer].count(Read::Layer(read), selected.len());
                    decode_layer(
                        &state.layers[layer],
                        &selected,
                        read,
                        &view,
                        &mut state.decoded,
                        &mut self.scratch,
                    )?;
                    let ctx = state.combine_ctx(layer, centroid);
                    combine_refine(
                        &ctx,
                        &state.decoded,
                        &mut state.admission_columns,
                        0..selected.len(),
                    );
                    if layer + 1 < state.layers.len() {
                        self.report.layers[layer].survivors += selected.len();
                    }
                }
                let doc_ids = enter_vector_stage(VectorStage::LateFilterDocIds);
                for (i, &offset) in offsets.iter().enumerate() {
                    let upper = Estimate(state.admission_columns.estimates[i])
                        .upper(state.admission_columns.sigmas[i], state.candidates.kappa());
                    if threshold.is_none_or(|bar| upper.0 >= bar) {
                        let docs = doc_cache.entry(cluster).or_default();
                        if docs.is_empty() {
                            self.reader.read_doc_ids(cluster, docs)?;
                        }
                        indexed.push((
                            docs[offset as usize],
                            BlockAddress::new(cluster as u32, offset),
                        ));
                    }
                }
                drop(doc_ids);
            }
            if self.report.layers.len() > 1 {
                self.report.layers.last_mut().unwrap().survivors += indexed.len();
            }
            if !indexed.is_empty() {
                self.report.filter_batches += 1;
                let sorting = enter_vector_stage(VectorStage::LateFilterSort);
                indexed.sort_unstable_by_key(|&(doc, _)| doc);
                drop(sorting);
                let mut scorer = {
                    let _setup = enter_vector_stage(VectorStage::LateFilterSetup);
                    ExcludedDocSet::new(weight.scorer_for_candidates_unpositioned(
                        reader,
                        1.0,
                        indexed.len() as u32,
                    )?)
                };
                {
                    let _predicate = enter_vector_stage(VectorStage::LateFilterPredicate);
                    self.report.filter_checks += indexed.len();
                    indexed.retain(|&(doc, _)| scorer.contains(doc));
                }
                self.report.filter_matches += indexed.len();
                self.report.exact_rows +=
                    self.exact
                        .score_candidates(&indexed, &mut self.scratch, &mut self.bound)?;
                #[cfg(test)]
                self.report
                    .trace
                    .refined_topk_rows
                    .extend(indexed.iter().map(|&(_, address)| address));
            }
        }
        for &b in &self.by_cluster {
            state.candidates.retire(b);
        }
        Ok(())
    }

    /// The stage the shortlist is built in: layer 0, or the exact scan at full precision.
    pub(crate) fn shortlist_stage(&self) -> VectorStage {
        match self.quantized {
            Some(_) => VectorStage::LayerScan(0),
            None => VectorStage::ExactScan,
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
    /// Returns eligible rows, including duplicate copies, for probe-budget accounting.
    pub(crate) fn shortlist(&mut self, rows: &ClusterRows<'_>) -> crate::Result<usize> {
        let _stage = enter_vector_stage(self.shortlist_stage());
        let len = rows.len();
        debug_assert!(len > 0, "empty clusters never reach the scan");
        #[cfg(test)]
        self.report.trace.scored_rows.extend(
            rows.local_rows()
                .map(|row| BlockAddress::new(rows.cluster as u32, row as u32)),
        );
        let unseen = self.seen.as_mut().map(|seen| {
            unseen_rows(
                seen,
                rows,
                &mut self.unseen_offsets,
                &mut self.unseen_docs,
                &mut self.replication_dropped,
            )
        });
        let selected = unseen
            .as_ref()
            .filter(|unseen| unseen.len() != len)
            .unwrap_or(rows);
        if selected.len() > 0 {
            // Planned and priced on the rows left to score; `len` stays the probe charge.
            let (read, cluster) = match &self.quantized {
                Some(state) => {
                    let (read, cluster) = state.read_plan(selected, 0, &mut self.scratch)?;
                    (read, Some(cluster))
                }
                None => (Read::Exact, None),
            };
            if self.quantized.is_some() {
                self.report.layers[0].count(read, selected.len());
            }
            let fold = self.rate.as_mut().map(|rate| match &mut rate.histograms {
                None => Fold::Prefill(&mut rate.prefill),
                Some(histograms) => {
                    let grid = histograms.grid();
                    Fold::Binned(histograms.cluster_mut(), grid)
                }
            });
            match read {
                Read::Exact => {
                    self.report.exact_rows +=
                        self.exact
                            .score(selected, &mut self.scratch, &mut self.bound, fold)?;
                }
                Read::Layer(read) => {
                    if let (Some(state), Some(cluster)) = (&mut self.quantized, &cluster) {
                        state.admit(
                            selected,
                            rows,
                            read,
                            cluster,
                            &mut self.bound,
                            &mut self.scratch,
                            fold,
                        )?;
                    }
                }
            }
        }
        self.report.candidates_scored += len;
        self.close_rate_cluster(len);
        Ok(len)
    }

    /// Ends the admission-rate measurement's cluster of `len` rows. Once the shared pruning
    /// threshold exists, the grid is set from it and every survival score so far.
    fn close_rate_cluster(&mut self, len: usize) {
        let Some(rate) = &mut self.rate else {
            return;
        };
        rate.rows += len;
        match &mut rate.histograms {
            Some(histograms) => histograms.finish_cluster(len),
            None => {
                if let Some(bar) = self.bound.shared_threshold() {
                    let w0 = rate.rows;
                    rate.histograms = Some(AdmissionHistograms::at_fill(bar, &rate.prefill, w0));
                    rate.next_refinement = w0.saturating_mul(2);
                    rate.prefill = Vec::new();
                }
            }
        }
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
            let (read, cluster) = state.read_plan(&group, layer, &mut self.scratch)?;
            self.report.layers[layer].count(read, group.len());
            match read {
                Read::Exact => {
                    self.report.exact_rows +=
                        self.exact
                            .score(&group, &mut self.scratch, &mut self.bound, None)?;
                    state.candidates.retire(b);
                }
                Read::Layer(read) => state.refine(layer, b, read, &cluster, &mut self.scratch)?,
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
        report.replication_dropped = self.replication_dropped;
        if let Some(state) = &self.quantized {
            report.centroid_reads = state.centroid_reads;
            #[cfg(test)]
            {
                report.trace.whole_band_admissions = state.whole_band_admissions;
            }
        }
        #[cfg(test)]
        {
            report.trace.exact_phase_rows = self.exact.exact_phase_rows.clone();
            report.trace.exact_scores = self.exact.exact_scores.clone();
        }
        Ok((self.exact.into_hits(segment_ord), report))
    }
}

#[cfg(test)]
mod tests {
    use common::BitSet;

    use super::unseen_rows;
    use crate::vector::search::cluster_source::ClusterRows;
    use crate::vector::test_support::Lcg;
    use crate::DocId;

    /// The replica drop keeps the rows whose document no earlier cluster held, in row order, marks
    /// every document seen and counts the rest, as a branching reference does: for whole clusters
    /// and gated selections over clusters sharing documents.
    #[test]
    fn unseen_rows_match_the_branching_reference() {
        let max_doc = 300;
        let mut rng = Lcg(11);
        let mut seen = BitSet::with_max_value(max_doc);
        let mut reference = BitSet::with_max_value(max_doc);
        let (mut offsets, mut docs) = (Vec::new(), Vec::new());
        let (mut dropped, mut reference_dropped) = (0, 0);
        for cluster in 0..40 {
            let cluster_docs: Vec<DocId> = (0..max_doc).filter(|_| rng.next_f32() > 0.8).collect();
            let rows = 0..cluster_docs.len();
            let selected: Vec<u32> = (0..cluster_docs.len() as u32)
                .filter(|_| rng.next_f32() > -0.5)
                .collect();
            let selected_docs: Vec<DocId> = selected
                .iter()
                .map(|&row| cluster_docs[row as usize])
                .collect();
            // Whole clusters and gated selections alternate.
            let (offsets_in, docs_in) = match cluster % 2 {
                0 => (None, &cluster_docs),
                _ => (Some(selected.as_slice()), &selected_docs),
            };
            let handed = ClusterRows {
                cluster,
                rows: rows.clone(),
                offsets: offsets_in,
                docs: Some(docs_in),
                centroid: None,
            };
            let mut want = (Vec::new(), Vec::new());
            for (position, &doc) in docs_in.iter().enumerate() {
                if reference.contains(doc) {
                    reference_dropped += 1;
                    continue;
                }
                reference.insert(doc);
                want.0
                    .push(offsets_in.map_or(position as u32, |o| o[position]));
                want.1.push(doc);
            }
            let got = unseen_rows(&mut seen, &handed, &mut offsets, &mut docs, &mut dropped);
            assert_eq!(got.offsets, Some(want.0.as_slice()), "cluster {cluster}");
            assert_eq!(got.docs, Some(want.1.as_slice()), "cluster {cluster}");
            assert_eq!(got.rows, rows);
            assert_eq!(dropped, reference_dropped, "cluster {cluster}");
            assert_eq!(seen.len(), reference.len(), "cluster {cluster}");
        }
        assert!(dropped > 0 && seen.len() > max_doc as usize / 2);
    }
}
