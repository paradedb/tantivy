//! The IVF segment scan: shortlists each cluster a source supplies, narrows the quantized
//! shortlist layer by layer under one pruning bound, and scores what is left exactly.

pub(crate) mod bound;
pub(crate) mod columns;
pub(crate) mod estimate;
mod exact;
pub(crate) mod heap;
pub(crate) mod layer_read;
#[cfg(test)]
mod quantized_boundary_tests;
#[cfg(test)]
mod tests;

use std::ops::Range;

use self::bound::KthBound;
use self::columns::QuantizedCandidates;
use self::estimate::{
    combine_initial_decoded, combine_refinement_decoded, cosine_refinement_batches,
};
use self::exact::ExactScorer;
use self::heap::TieBreakHits;
use self::layer_read::{candidate_offsets, score_layer};
use crate::collector::sort_key::Comparator;
use crate::collector::SegmentSortKeyComputer;
use crate::error::DataCorruption;
use crate::schema::Metric;
use crate::vector::cluster_source::{ClusterRows, ClusterSource, Feedback};
use crate::vector::index_reader::{QuantizedFieldReader, VectorIndexReader};
use crate::vector::ivf::IvfIndex;
use crate::vector::prepared::{ArithmeticError, QuantizedQueryCtx, ScoringQuery};
use crate::vector::{enter_vector_stage, Similarity, Stage, VectorElement};
use crate::SegmentOrdinal;

// --- Contracts ---

/// Sizes for one segment scan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanConfig {
    pub(crate) top_n: usize,
    /// Upper bound on rows admitted, for sizing the candidate columns.
    pub(crate) capacity: usize,
}

/// Per-layer counts of a quantized scan.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LayerReport {
    /// Candidates the layer scored.
    pub(crate) scored: usize,
    /// Candidates left after the layer's boundary.
    pub(crate) survivors: usize,
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
    #[cfg(test)]
    pub(crate) trace: ScanTrace,
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
    /// Rows scored in the exact phase.
    pub(crate) exact_phase_rows: Vec<usize>,
}

/// What the scan remembers about a cluster it admitted.
#[derive(Clone, Copy, Debug)]
struct AdmittedCluster {
    centroid: Similarity,
    /// Whether the candidates carry this cluster's doc ids.
    docs_in_candidates: bool,
}

// --- Quantized state ---

/// Per-segment buffers reused across clusters and layers.
#[derive(Default)]
struct ScanScratch {
    kernel_scores: Vec<f32>,
    decoded_scales: Vec<f32>,
    decoded_gammas: Vec<f32>,
    decoded_error_ratios: Vec<f32>,
    decoded_constants: Vec<f32>,
    decoded_residual_norms: Vec<f32>,
    base_scores: Vec<f32>,
    estimate_scores: Vec<f32>,
    sigma_scores: Vec<f32>,
    residual_norm_squared_scores: Vec<f32>,
    sign_query_error_terms: Vec<f32>,
    arithmetic_variances: Vec<ArithmeticError>,
    selection_offsets: Vec<usize>,
    selected_rows: Vec<usize>,
    row_offsets: Vec<usize>,
    read_ranges: Vec<Range<usize>>,
    /// A survivor group's cluster-local offsets.
    group_offsets: Vec<usize>,
}

/// The quantized cascade's state: candidate columns plus each admitted cluster's record.
struct QuantizedState<'a> {
    query: &'a QuantizedQueryCtx,
    field: &'a QuantizedFieldReader,
    metric: Metric,
    dimension: usize,
    candidates: QuantizedCandidates,
    /// By cluster id; `None` for clusters never admitted.
    admitted: Vec<Option<AdmittedCluster>>,
}

impl<'a> QuantizedState<'a> {
    fn new(query: &'a QuantizedQueryCtx, field: &'a QuantizedFieldReader, capacity: usize) -> Self {
        let meta = query.index.meta.field();
        Self {
            query,
            field,
            metric: meta.metric,
            dimension: meta.dim as usize,
            candidates: QuantizedCandidates::with_capacity(capacity),
            admitted: Vec::new(),
        }
    }

    fn record(&mut self, cluster: usize, admitted: AdmittedCluster) {
        if self.admitted.len() <= cluster {
            self.admitted.resize(cluster + 1, None);
        }
        self.admitted[cluster] = Some(admitted);
    }

    /// The admitted clusters in ascending id, which is ascending row order.
    fn admitted_clusters(&self) -> impl Iterator<Item = (usize, AdmittedCluster)> + '_ {
        self.admitted
            .iter()
            .enumerate()
            .filter_map(|(cluster, admitted)| admitted.map(|admitted| (cluster, admitted)))
    }

    /// Scores a cluster's rows on layer 0 from its whole band and appends them as quantized
    /// candidates.
    fn admit(
        &mut self,
        rows: &ClusterRows<'_>,
        bound: &mut KthBound,
        scratch: &mut ScanScratch,
    ) -> crate::Result<()> {
        let query = self.query;
        let selected_count = rows.len();
        let layer = &self.field.layers()[0];
        let cluster_score = rows.centroid.score();
        let band = layer.read_batch_in_block(rows.cluster, rows.rows.clone())?;
        let score_query_norm = query.score_query_norm(cluster_score);
        score_layer(
            query,
            0,
            layer,
            Some((&band, &mut scratch.decoded_residual_norms)),
            Some(rows.cluster),
            rows.rows.clone(),
            rows.offsets,
            &mut scratch.kernel_scores,
            &mut scratch.decoded_scales,
            &mut scratch.decoded_gammas,
            &mut scratch.decoded_error_ratios,
            &mut scratch.decoded_constants,
            &mut scratch.read_ranges,
            &mut scratch.selected_rows,
            &mut scratch.row_offsets,
        )?;
        scratch.base_scores.resize(selected_count, 0.0);
        scratch.estimate_scores.resize(selected_count, 0.0);
        scratch.sigma_scores.resize(selected_count, 0.0);
        scratch
            .residual_norm_squared_scores
            .resize(selected_count, 0.0);
        scratch.sign_query_error_terms.resize(selected_count, 0.0);
        scratch
            .arithmetic_variances
            .resize(selected_count, ArithmeticError::default());
        combine_initial_decoded(
            self.metric,
            self.dimension,
            &mut scratch.kernel_scores,
            &mut scratch.base_scores,
            &mut scratch.estimate_scores,
            &mut scratch.sigma_scores,
            &mut scratch.residual_norm_squared_scores,
            &mut scratch.sign_query_error_terms,
            &mut scratch.arithmetic_variances,
            &scratch.decoded_scales,
            &scratch.decoded_gammas,
            &scratch.decoded_error_ratios,
            &scratch.decoded_constants,
            &scratch.decoded_residual_norms,
            cluster_score,
            score_query_norm * score_query_norm,
            query.query_error_squared(0) as f32,
        );
        self.record(
            rows.cluster,
            AdmittedCluster {
                centroid: rows.centroid,
                docs_in_candidates: rows.docs.is_some(),
            },
        );
        let cluster_start = self.candidates.len();
        self.candidates.append_selected(
            rows.rows.clone(),
            rows.offsets,
            rows.docs,
            &scratch.base_scores[..selected_count],
            &scratch.kernel_scores[..selected_count],
            &scratch.estimate_scores[..selected_count],
            &scratch.sigma_scores[..selected_count],
            &scratch.residual_norm_squared_scores[..selected_count],
            &scratch.decoded_gammas[..selected_count],
            &scratch.sign_query_error_terms[..selected_count],
            &scratch.arithmetic_variances[..selected_count],
        );
        self.candidates.finish_cluster(cluster_start, bound);
        Ok(())
    }

    /// Records the candidates `layer` scored and prunes them at its boundary.
    fn prune(&mut self, layer: usize, bound: &mut KthBound, report: &mut ScanReport) {
        report.layers[layer].scored = self.candidates.len();
        #[cfg(test)]
        report
            .trace
            .estimate_rows
            .push(self.candidates.estimate_trace());
        let _boundary = enter_vector_stage(Stage::Boundary(layer as u8));
        self.candidates.band(bound);
        report.layers[layer].survivors = self.candidates.len();
        #[cfg(test)]
        report
            .trace
            .boundary_rows
            .push(self.candidates.rows.clone());
    }

    /// Refines every live candidate with `layer`, which follows at least one boundary.
    fn refine_layer(
        &mut self,
        layer_idx: usize,
        index: &IvfIndex,
        scratch: &mut ScanScratch,
    ) -> crate::Result<()> {
        let query = self.query;
        let metric = self.metric;
        let layer = &self.field.layers()[layer_idx];
        let sign_query_error_squared =
            if matches!(query.index.specs[layer_idx].kind, cascade::LayerKind::Sign) {
                query.query_error_squared(layer_idx) as f32
            } else {
                0.0
            };
        if metric == Metric::Cosine {
            let query_norm = query.score_query_norm(0.0);
            for candidate_range in cosine_refinement_batches(self.candidates.len()) {
                let first_row = self.candidates.rows[candidate_range.start];
                let last_row = self.candidates.rows[candidate_range.end - 1];
                let available_rows = first_row..last_row + 1;
                let offsets = candidate_offsets(
                    &self.candidates.rows[candidate_range.clone()],
                    &available_rows,
                    &mut scratch.selection_offsets,
                );
                let rows = score_layer(
                    query,
                    layer_idx,
                    layer,
                    None,
                    None,
                    available_rows,
                    Some(offsets),
                    &mut scratch.kernel_scores,
                    &mut scratch.decoded_scales,
                    &mut scratch.decoded_gammas,
                    &mut scratch.decoded_error_ratios,
                    &mut scratch.decoded_constants,
                    &mut scratch.read_ranges,
                    &mut scratch.selected_rows,
                    &mut scratch.row_offsets,
                )?;
                combine_refinement_decoded(
                    metric,
                    self.dimension,
                    &mut self.candidates,
                    candidate_range,
                    &scratch.kernel_scores[..rows],
                    &scratch.decoded_scales[..rows],
                    &scratch.decoded_gammas[..rows],
                    &scratch.decoded_error_ratios[..rows],
                    &[],
                    query_norm * query_norm,
                    sign_query_error_squared,
                );
            }
            return Ok(());
        }
        let groups = survivor_groups(&self.candidates.rows, self.admitted_clusters(), index);
        for (cluster, admitted, candidate_range) in groups {
            let cluster_rows = index.cluster_range(cluster);
            let query_norm = query.score_query_norm(admitted.centroid.score());
            let offsets = candidate_offsets(
                &self.candidates.rows[candidate_range.clone()],
                &cluster_rows,
                &mut scratch.selection_offsets,
            );
            let rows = score_layer(
                query,
                layer_idx,
                layer,
                None,
                Some(cluster),
                cluster_rows,
                Some(offsets),
                &mut scratch.kernel_scores,
                &mut scratch.decoded_scales,
                &mut scratch.decoded_gammas,
                &mut scratch.decoded_error_ratios,
                &mut scratch.decoded_constants,
                &mut scratch.read_ranges,
                &mut scratch.selected_rows,
                &mut scratch.row_offsets,
            )?;
            let decoded_constants = if metric == Metric::L2 {
                &scratch.decoded_constants[..rows]
            } else {
                &[]
            };
            combine_refinement_decoded(
                metric,
                self.dimension,
                &mut self.candidates,
                candidate_range,
                &scratch.kernel_scores[..rows],
                &scratch.decoded_scales[..rows],
                &scratch.decoded_gammas[..rows],
                &scratch.decoded_error_ratios[..rows],
                decoded_constants,
                query_norm * query_norm,
                sign_query_error_squared,
            );
        }
        Ok(())
    }
}

/// Splits row-sorted survivors into one candidate range per admitted cluster.
fn survivor_groups(
    rows: &[usize],
    admitted: impl Iterator<Item = (usize, AdmittedCluster)>,
    index: &IvfIndex,
) -> Vec<(usize, AdmittedCluster, Range<usize>)> {
    debug_assert!(
        rows.windows(2).all(|pair| pair[0] < pair[1]),
        "survivors are row-sorted"
    );
    let mut groups = Vec::new();
    let mut first = 0;
    for (cluster, record) in admitted {
        if first == rows.len() {
            break;
        }
        let cluster_rows = index.cluster_range(cluster);
        let end = first + rows[first..].partition_point(|&row| row < cluster_rows.end);
        if end > first {
            debug_assert!(rows[first] >= cluster_rows.start);
            groups.push((cluster, record, first..end));
        }
        first = end;
    }
    debug_assert_eq!(
        first,
        rows.len(),
        "every survivor belongs to an admitted cluster"
    );
    groups
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
    index: &'a IvfIndex,
    exact: ExactScorer<'a, T, K, C>,
    /// The quantized cascade; `None` scores every shortlisted row exactly.
    quantized: Option<QuantizedState<'a>>,
    scratch: ScanScratch,
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
        index: &'a IvfIndex,
        query: ScoringQuery<'a, T>,
        config: ScanConfig,
        tie_break: &'a mut K,
        tie_comparator: C,
    ) -> crate::Result<Self> {
        let quantized = match (query.quantized, reader.quantization()) {
            (Some(quantized), Some(field)) => {
                Some(QuantizedState::new(quantized, field, config.capacity))
            }
            (Some(_), None) => {
                return Err(DataCorruption::comment_only(
                    "quantized query requires quantized slots",
                )
                .into())
            }
            (None, _) => None,
        };
        let layers = quantized
            .as_ref()
            .map_or(0, |state| state.query.active_layers());
        Ok(Self {
            index,
            exact: ExactScorer::new(reader, query.query, config.top_n, tie_break, tie_comparator),
            quantized,
            scratch: ScanScratch::default(),
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
    fn shortlist_stage(&self) -> Stage {
        match self.quantized {
            Some(_) => Stage::LayerScan(0),
            None => Stage::ExactScan,
        }
    }

    /// The pruning thresholds as of the last shortlisted cluster.
    fn feedback(&self) -> Feedback {
        Feedback {
            kth_lower: self.exact.bound.running_lower(),
            kth_estimate: self.exact.bound.running_estimate(),
        }
    }

    /// Adds a cluster's rows to the shortlist: scored exactly at full precision, otherwise as
    /// quantized candidates. Returns the rows scored.
    fn shortlist(&mut self, rows: &ClusterRows<'_>) -> crate::Result<usize> {
        let len = rows.len();
        debug_assert!(len > 0, "empty clusters never reach the scan");
        #[cfg(test)]
        self.report.trace.scored_rows.extend(rows.global_rows());
        match &mut self.quantized {
            Some(state) => state.admit(rows, &mut self.exact.bound, &mut self.scratch)?,
            None => {
                self.report.exact_rows += self.exact.score(rows)?;
            }
        }
        self.report.candidates_scored += len;
        Ok(len)
    }

    /// Prunes the shortlist at layer 0's boundary.
    fn close_shortlist(&mut self) {
        if let Some(state) = &mut self.quantized {
            state.prune(0, &mut self.exact.bound, &mut self.report);
        }
    }

    /// The quantized layers after layer 0; none at full precision.
    fn refining_layers(&self) -> Range<usize> {
        1..self
            .quantized
            .as_ref()
            .map_or(1, |state| state.query.active_layers())
    }

    /// Refines the shortlist with `layer` and prunes it at that layer's boundary.
    fn narrow(&mut self, layer: usize) -> crate::Result<()> {
        let Some(state) = &mut self.quantized else {
            return Ok(());
        };
        {
            let _layer = enter_vector_stage(Stage::LayerScan(layer as u8));
            state.refine_layer(layer, self.index, &mut self.scratch)?;
        }
        state.prune(layer, &mut self.exact.bound, &mut self.report);
        Ok(())
    }

    /// Scores the last boundary's survivors exactly, one cluster group at a time.
    fn score_shortlist(&mut self) -> crate::Result<()> {
        let Some(state) = &self.quantized else {
            return Ok(());
        };
        let survivors = &state.candidates;
        for (cluster, admitted, group) in
            survivor_groups(&survivors.rows, state.admitted_clusters(), self.index)
        {
            let cluster_rows = self.index.cluster_range(cluster);
            self.scratch.group_offsets.clear();
            self.scratch.group_offsets.extend(
                survivors.rows[group.clone()]
                    .iter()
                    .map(|&row| row - cluster_rows.start),
            );
            let rows = ClusterRows {
                cluster,
                rows: cluster_rows,
                offsets: Some(&self.scratch.group_offsets),
                docs: admitted.docs_in_candidates.then(|| &survivors.docs[group]),
                centroid: admitted.centroid,
            };
            self.report.exact_rows += self.exact.score_survivors(&rows)?;
        }
        Ok(())
    }

    /// Returns the hits and the scan's report.
    pub(crate) fn finish(
        self,
        segment_ord: SegmentOrdinal,
    ) -> crate::Result<(TieBreakHits<K>, ScanReport)> {
        let _assembly = enter_vector_stage(Stage::ResultAssembly);
        let Self { exact, report, .. } = self;
        #[cfg(test)]
        let report = {
            let mut report = report;
            report.trace.exact_phase_rows = exact.exact_phase_rows.clone();
            report
        };
        Ok((exact.into_hits(segment_ord), report))
    }
}
