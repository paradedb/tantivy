//! Per-segment vector metadata, block columns and optional IVF routing.
//!
//! Metadata opens independently of search state. The first search-reader request
//! pins routing state and validates deferred row-group geometry. Clustered IdMap
//! entries open only for document lookups.
//! Cluster boundaries come from the centroid file. See FORMAT.md.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, OnceLock};

#[cfg(test)]
use common::HasLen;
use common::OwnedBytes;
use quant_model::f16::f16_to_f32;

use super::backend::{Estimate, Threshold};
use super::blocks::{BlockMetadata, Blocks};
use super::flat::IdMap;
use super::header::{read_centroid_header, read_vector_header, CentroidSlot, VectorEntry};
use super::ivf::{decode_row, IvfIndex, CENTROIDS_EXT};
use super::metadata::{SlotType, VectorColMetadata};
use super::prepared::{
    corrected_quantized_estimate, initial_dot_raw_prefix, initial_l2_raw_prefix,
    quantized_model_sigma, refine_dot_raw_prefix, refine_l2_raw_prefix, ArithmeticError,
    PreparedQuery, QuantizedIndexCtx, QuantizedQueryCtx,
};
use super::quantization::{
    quantized_code_tail_is_zero, QUANTIZED_BOUNDARY_KAPPA, QUANTIZED_CONSTANT_STRIDE,
    QUANTIZED_ERROR_RATIO_STRIDE, QUANTIZED_GAMMA_STRIDE, QUANTIZED_SCALE_STRIDE,
};
use super::storage_io::VectorRead;
use super::VEC_EXT;
use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::error::DataCorruption;
use crate::fastfield::AliveBitSet;
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType, Metric, VectorOptions};
use crate::{DocId, SegmentReader, TantivyError};

/// Which on-disk layout a segment's vector data uses, surfaced through
/// [`VectorInfo`] for tooling.
/// Vector row-storage layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorStorageFormat {
    /// Full-precision rows without cluster routing.
    Flat,
    /// Clustered inverted-file rows.
    Ivf,
}

/// Segment vector-storage metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorInfo {
    /// Row-storage layout.
    pub format: VectorStorageFormat,
    /// Distinct documents with a vector in this field.
    /// Distinct documents with vectors.
    pub num_vectors: usize,
    /// Number of IVF centroids.
    pub num_centroids: Option<usize>,
    /// IVF posting-size statistics.
    pub cluster_stats: Option<VectorClusterStats>,
}

/// IVF posting-size statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorClusterStats {
    /// Minimum posting size.
    pub min_cluster_size: usize,
    /// Maximum posting size.
    pub max_cluster_size: usize,
    /// Mean posting size.
    pub avg_cluster_size: f64,
    /// Empty posting count.
    pub empty_clusters: usize,
}

/// Moments of `(estimate - exact) / sigma` for estimator diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VectorEstimatorMoments {
    /// Observation count.
    pub sample_count: u64,
    /// Sum of normalized errors.
    pub normalized_error_sum: f64,
    /// Sum of squared normalized errors.
    pub normalized_error_squared_sum: f64,
}

impl VectorEstimatorMoments {
    fn observe(&mut self, value: f64) {
        self.sample_count += 1;
        self.normalized_error_sum += value;
        self.normalized_error_squared_sum += value * value;
    }

    /// Returns the mean normalized error.
    pub fn bias(&self) -> Option<f64> {
        (self.sample_count != 0).then(|| self.normalized_error_sum / self.sample_count as f64)
    }

    /// Returns the normalized-error standard deviation.
    pub fn spread(&self) -> Option<f64> {
        self.bias().map(|bias| {
            (self.normalized_error_squared_sum / self.sample_count as f64 - bias * bias)
                .max(0.0)
                .sqrt()
        })
    }
}

/// Estimator moments aggregated by depth and query.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorEstimatorMeasurements {
    source: VectorEstimatorSource,
    schedule: Vec<(&'static str, u8)>,
    aggregate: Vec<VectorEstimatorMoments>,
    per_query: Vec<Vec<VectorEstimatorMoments>>,
    sample_rows: u64,
    query_count: u32,
}

/// Query input for estimator diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorEstimatorQuery {
    /// Query coordinates.
    pub values: Vec<f32>,
    /// Document excluded from stored-row diagnostics.
    pub excluded_doc_id: Option<DocId>,
}

/// Query source for estimator diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorEstimatorSource {
    /// Caller-provided query vectors.
    Provided,
    /// Stored vectors sampled as pseudo-queries.
    HeldOut,
}

/// Mergeable scalar moments used by exact-E diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorAuditMoments {
    /// Observation count.
    pub sample_count: u64,
    /// Observation sum.
    pub sum: f64,
    /// Squared-observation sum.
    pub squared_sum: f64,
    /// Minimum observation.
    pub min: f64,
    /// Maximum observation.
    pub max: f64,
    samples: Vec<f64>,
}

impl Default for VectorAuditMoments {
    fn default() -> Self {
        Self {
            sample_count: 0,
            sum: 0.0,
            squared_sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            samples: Vec::new(),
        }
    }
}

impl VectorAuditMoments {
    fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.sample_count += 1;
        self.sum += value;
        self.squared_sum += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.samples.push(value);
    }

    fn merge(&mut self, other: &Self) {
        if other.sample_count == 0 {
            return;
        }
        self.sample_count += other.sample_count;
        self.sum += other.sum;
        self.squared_sum += other.squared_sum;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.samples.extend_from_slice(&other.samples);
    }

    /// Returns the observation mean.
    pub fn mean(&self) -> Option<f64> {
        (self.sample_count != 0).then(|| self.sum / self.sample_count as f64)
    }

    /// Returns the observation standard deviation.
    pub fn spread(&self) -> Option<f64> {
        self.mean().map(|mean| {
            (self.squared_sum / self.sample_count as f64 - mean * mean)
                .max(0.0)
                .sqrt()
        })
    }

    /// Exact nearest-rank quantile of the retained finite audit samples.
    pub fn quantile(&self, quantile: f64) -> Option<f64> {
        if self.samples.is_empty() || !(0.0..=1.0).contains(&quantile) {
            return None;
        }
        let mut samples = self.samples.clone();
        samples.sort_by(f64::total_cmp);
        let rank = ((quantile * samples.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(samples.len() - 1);
        Some(samples[rank])
    }

    /// Exact nearest-rank quantile of the retained absolute audit samples.
    pub fn quantile_abs(&self, quantile: f64) -> Option<f64> {
        if self.samples.is_empty() || !(0.0..=1.0).contains(&quantile) {
            return None;
        }
        let mut samples: Vec<f64> = self.samples.iter().map(|sample| sample.abs()).collect();
        samples.sort_by(f64::total_cmp);
        let rank = ((quantile * samples.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(samples.len() - 1);
        Some(samples[rank])
    }

    /// Returns the median.
    pub fn p50(&self) -> Option<f64> {
        self.quantile(0.50)
    }

    /// Returns the 95th percentile.
    pub fn p95(&self) -> Option<f64> {
        self.quantile(0.95)
    }

    /// Returns the 99th percentile.
    pub fn p99(&self) -> Option<f64> {
        self.quantile(0.99)
    }

    /// Returns the 99th percentile of absolute values.
    pub fn p99_abs(&self) -> Option<f64> {
        self.quantile_abs(0.99)
    }

    /// Returns the maximum absolute value.
    pub fn max_abs(&self) -> Option<f64> {
        (!self.samples.is_empty()).then(|| {
            self.samples
                .iter()
                .fold(0.0_f64, |max, value| max.max(value.abs()))
        })
    }
}

/// Corrected-error diagnostics for one prefix depth.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorErrorDepthMeasurements {
    /// Stored residual squared norm.
    pub residual_norm_squared: VectorAuditMoments,
    /// Decoded binary16 cumulative-prefix correction used by scoring.
    pub stored_gamma: VectorAuditMoments,
    /// Gamma before clamping and serialization.
    pub raw_gamma: VectorAuditMoments,
    /// Sampled rows whose stored per-layer scale is exactly zero.
    pub zero_scale_count: u64,
    /// Sampled rows whose regenerated gamma is below the declared lower clamp.
    pub gamma_lower_clamp_count: u64,
    /// Sampled rows whose regenerated gamma is above the declared upper clamp.
    pub gamma_upper_clamp_count: u64,
    /// Gamma serialization displacement in confidence-width units.
    pub gamma_round_trip_band_error: VectorAuditMoments,
    /// Stored corrected residual error ratio E.
    pub corrected_error_ratio: VectorAuditMoments,
    /// Serving uncertainty width.
    pub sigma: VectorAuditMoments,
}

impl VectorErrorDepthMeasurements {
    fn merge(&mut self, other: &Self) {
        self.residual_norm_squared
            .merge(&other.residual_norm_squared);
        self.stored_gamma.merge(&other.stored_gamma);
        self.raw_gamma.merge(&other.raw_gamma);
        self.zero_scale_count += other.zero_scale_count;
        self.gamma_lower_clamp_count += other.gamma_lower_clamp_count;
        self.gamma_upper_clamp_count += other.gamma_upper_clamp_count;
        self.gamma_round_trip_band_error
            .merge(&other.gamma_round_trip_band_error);
        self.corrected_error_ratio
            .merge(&other.corrected_error_ratio);
        self.sigma.merge(&other.sigma);
    }
}

/// Corrected-error audit measurements.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorErrorAuditMeasurements {
    /// Query source.
    pub source: VectorEstimatorSource,
    /// Estimator moments.
    pub estimator: VectorEstimatorMeasurements,
    /// Measurements by prefix depth.
    pub depths: Vec<VectorErrorDepthMeasurements>,
}

/// Confidence-cone measurements for one boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorErrorConeDepthMeasurements {
    /// Boundary confidence width.
    pub kappa: f32,
    /// Scored-row moments.
    pub scored_rows: VectorAuditMoments,
    /// Survivor-row moments.
    pub survivor_rows: VectorAuditMoments,
    /// Survivor-document moments.
    pub survivor_docs: VectorAuditMoments,
    /// Survivor-fraction moments.
    pub survivor_fraction: VectorAuditMoments,
    /// Candidate-recall moments.
    pub candidate_recall: VectorAuditMoments,
    /// Queries with at least one miss.
    pub queries_with_miss: u32,
}

impl VectorErrorConeDepthMeasurements {
    fn new(kappa: f32) -> Self {
        Self {
            kappa,
            scored_rows: VectorAuditMoments::default(),
            survivor_rows: VectorAuditMoments::default(),
            survivor_docs: VectorAuditMoments::default(),
            survivor_fraction: VectorAuditMoments::default(),
            candidate_recall: VectorAuditMoments::default(),
            queries_with_miss: 0,
        }
    }
}

/// Confidence-cone audit measurements.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorErrorConeAuditMeasurements {
    /// Query count.
    pub query_count: u32,
    /// Requested result count.
    pub top_k: u32,
    /// Measurements by prefix depth.
    pub depths: Vec<VectorErrorConeDepthMeasurements>,
}

const ERROR_CONE_QUERY_COUNT: usize = 100;
const ERROR_CONE_TOP_K: usize = 10;

impl VectorErrorAuditMeasurements {
    /// Ordered quantizer kinds and widths, independent of rotation seeds.
    pub fn schedule(&self) -> &[(&'static str, u8)] {
        self.estimator.schedule()
    }

    /// Merges compatible audit measurements.
    ///
    /// # Errors
    ///
    /// Returns an error when schedules, sources or depth counts differ.
    pub fn merge(&mut self, other: &Self) -> crate::Result<()> {
        self.estimator.check_schedule(&other.estimator)?;
        if self.source != other.source || self.depths.len() != other.depths.len() {
            return Err(TantivyError::InvalidArgument(
                "cannot merge exact-E audit measurements with different sources or depths"
                    .to_string(),
            ));
        }
        self.estimator.merge(&other.estimator)?;
        for (left, right) in self.depths.iter_mut().zip(&other.depths) {
            left.merge(right);
        }
        Ok(())
    }
}

#[inline]
fn diagnostic_query_norm(query: &QuantizedQueryCtx, metric: Metric, centroid_bytes: &[u8]) -> f32 {
    let routing_score = metric
        .similarity_bytes::<f32>(query.query(), centroid_bytes)
        .score();
    query.score_query_norm(routing_score)
}

#[inline]
fn observe_corrected_prefix(
    measurements: &mut VectorEstimatorMeasurements,
    query_idx: usize,
    depth: usize,
    exact_dot: f32,
    corrected_prefix_estimate: f32,
    model_sigma: f64,
) {
    if model_sigma > 0.0 && model_sigma.is_finite() {
        let error = (f64::from(corrected_prefix_estimate) - f64::from(exact_dot)) / model_sigma;
        if error.is_finite() {
            measurements.aggregate[depth].observe(error);
            measurements.per_query[query_idx][depth].observe(error);
        }
    }
}

pub(crate) fn diagnostic_advance_raw_prefix(
    query: &QuantizedQueryCtx,
    metric: Metric,
    depth: usize,
    codes: &[u8],
    scale: f32,
    constant: Option<f32>,
    raw_prefix: f32,
    cluster_score: f32,
    residual_norm_squared: f32,
    arithmetic: &mut ArithmeticError,
) -> crate::Result<f32> {
    let mut kernel_score = [0.0];
    query.score_layer_batch_unscaled(depth, codes, codes.len(), &mut kernel_score);
    if depth == 0 {
        *arithmetic = ArithmeticError::initial(
            metric,
            kernel_score[0],
            scale,
            constant.unwrap_or(0.0),
            cluster_score,
            residual_norm_squared,
        );
    } else {
        arithmetic.refine(
            metric,
            raw_prefix,
            kernel_score[0],
            scale,
            constant.unwrap_or(0.0),
        );
    }
    match (metric, depth, constant) {
        (Metric::L2, 0, Some(constant)) => {
            Ok(initial_l2_raw_prefix(kernel_score[0], scale, constant))
        }
        (Metric::L2, _, Some(constant)) => Ok(refine_l2_raw_prefix(
            raw_prefix,
            kernel_score[0],
            scale,
            constant,
        )),
        (Metric::Dot | Metric::Cosine, 0, None) => {
            Ok(initial_dot_raw_prefix(kernel_score[0], scale))
        }
        (Metric::Dot | Metric::Cosine, _, None) => {
            Ok(refine_dot_raw_prefix(raw_prefix, kernel_score[0], scale))
        }
        (Metric::L2, _, None) => Err(DataCorruption::comment_only(
            "vector estimator L2 layer is missing a split constant",
        )
        .into()),
        (Metric::Dot | Metric::Cosine, _, Some(_)) => Err(DataCorruption::comment_only(
            "vector estimator dot-like layer has an unexpected split constant",
        )
        .into()),
    }
}

/// SoA boundary state for corrected-error diagnostics.
#[derive(Default)]
struct ErrorConeCandidates {
    rows: Vec<usize>,
    docs: Vec<DocId>,
    raw_prefixes: Vec<f32>,
    sign_query_error_terms: Vec<f32>,
    estimates: Vec<f32>,
    sigmas: Vec<f32>,
    arithmetic_errors: Vec<ArithmeticError>,
}

impl ErrorConeCandidates {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn push(
        &mut self,
        row: usize,
        doc: DocId,
        raw_prefix: f32,
        sign_query_error_term: f32,
        estimate: f32,
        sigma: f32,
    ) {
        self.rows.push(row);
        self.docs.push(doc);
        self.raw_prefixes.push(raw_prefix);
        self.sign_query_error_terms.push(sign_query_error_term);
        self.estimates.push(estimate);
        self.sigmas.push(sigma);
        self.arithmetic_errors.push(ArithmeticError::default());
    }

    fn distinct_doc_count(&self) -> usize {
        self.docs
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    fn candidate_recall(&self, exact_top_docs: &[DocId]) -> f64 {
        let docs: std::collections::HashSet<DocId> = self.docs.iter().copied().collect();
        exact_top_docs
            .iter()
            .filter(|doc| docs.contains(doc))
            .count() as f64
            / exact_top_docs.len() as f64
    }

    /// Applies the production boundary rule and returns storage-ordered survivors.
    fn band(&mut self, top_k: usize, kappa: f32) {
        let mut best_by_doc: HashMap<DocId, usize> = HashMap::new();
        for index in 0..self.len() {
            let doc = self.docs[index];
            if best_by_doc.get(&doc).is_none_or(|&previous| {
                error_cone_candidate_order(self, index, previous, kappa).is_lt()
            }) {
                best_by_doc.insert(doc, index);
            }
        }
        let threshold = if top_k == 0 || best_by_doc.len() < top_k {
            None
        } else {
            let mut best: Vec<usize> = best_by_doc.into_values().collect();
            let (_, pivot, _) = best.select_nth_unstable_by(top_k - 1, |&left, &right| {
                error_cone_candidate_order(self, left, right, kappa)
            });
            let pivot = *pivot;
            Some(Threshold(
                Estimate(self.estimates[pivot]).lower(self.sigmas[pivot], kappa),
            ))
        };

        let mut survivors: Vec<usize> = (0..self.len())
            .filter(|&index| {
                threshold.is_none_or(|threshold| {
                    threshold
                        .admits(Estimate(self.estimates[index]).upper(self.sigmas[index], kappa))
                })
            })
            .collect();
        survivors.sort_unstable_by_key(|&index| self.rows[index]);

        let mut compacted = Self::default();
        compacted.rows.reserve(survivors.len());
        compacted.docs.reserve(survivors.len());
        compacted.raw_prefixes.reserve(survivors.len());
        compacted.sign_query_error_terms.reserve(survivors.len());
        compacted.estimates.reserve(survivors.len());
        compacted.sigmas.reserve(survivors.len());
        for index in survivors {
            compacted.push(
                self.rows[index],
                self.docs[index],
                self.raw_prefixes[index],
                self.sign_query_error_terms[index],
                self.estimates[index],
                self.sigmas[index],
            );
            *compacted.arithmetic_errors.last_mut().unwrap() = self.arithmetic_errors[index];
        }
        *self = compacted;
    }
}

fn error_cone_candidate_order(
    candidates: &ErrorConeCandidates,
    left: usize,
    right: usize,
    kappa: f32,
) -> Ordering {
    Estimate(candidates.estimates[right])
        .lower(candidates.sigmas[right], kappa)
        .0
        .total_cmp(
            &Estimate(candidates.estimates[left])
                .lower(candidates.sigmas[left], kappa)
                .0,
        )
        .then(candidates.rows[left].cmp(&candidates.rows[right]))
}

fn observe_error_cone_depth(
    measurements: &mut VectorErrorConeDepthMeasurements,
    scored: usize,
    candidates: &ErrorConeCandidates,
    exact_top_docs: &[DocId],
) {
    measurements.scored_rows.observe(scored as f64);
    measurements.survivor_rows.observe(candidates.len() as f64);
    measurements
        .survivor_docs
        .observe(candidates.distinct_doc_count() as f64);
    measurements.survivor_fraction.observe(if scored == 0 {
        0.0
    } else {
        candidates.len() as f64 / scored as f64
    });
    let recall = candidates.candidate_recall(exact_top_docs);
    measurements.candidate_recall.observe(recall);
    measurements.queries_with_miss += u32::from(recall < 1.0);
}

impl VectorEstimatorMeasurements {
    /// Ordered quantizer kinds and widths, independent of rotation seeds.
    pub fn schedule(&self) -> &[(&'static str, u8)] {
        &self.schedule
    }

    fn check_schedule(&self, other: &Self) -> crate::Result<()> {
        if self.schedule != other.schedule {
            return Err(TantivyError::InvalidArgument(format!(
                "cannot merge vector measurements with different schedules: {:?} and {:?}",
                self.schedule, other.schedule
            )));
        }
        Ok(())
    }

    /// Returns aggregate moments by prefix depth.
    pub fn aggregate(&self) -> &[VectorEstimatorMoments] {
        &self.aggregate
    }

    /// Returns moments by query and prefix depth.
    pub fn per_query(&self) -> &[Vec<VectorEstimatorMoments>] {
        &self.per_query
    }

    /// Returns the query source.
    pub fn source(&self) -> VectorEstimatorSource {
        self.source
    }

    /// Returns the number of sampled IVF rows.
    pub fn sample_rows(&self) -> u64 {
        self.sample_rows
    }

    /// Returns the number of prepared queries.
    pub fn query_count(&self) -> u32 {
        self.query_count
    }

    /// Merges compatible estimator measurements.
    ///
    /// # Errors
    ///
    /// Returns an error when schedules or measurement shapes differ.
    pub fn merge(&mut self, other: &Self) -> crate::Result<()> {
        self.check_schedule(other)?;
        if self.aggregate.is_empty() {
            *self = other.clone();
            return Ok(());
        }
        if self.aggregate.len() != other.aggregate.len()
            || self.per_query.len() != other.per_query.len()
            || self.query_count != other.query_count
            || self.source != other.source
        {
            return Err(TantivyError::InvalidArgument(
                "cannot merge vector estimator measurements with different shapes".to_string(),
            ));
        }
        self.sample_rows = self
            .sample_rows
            .checked_add(other.sample_rows)
            .ok_or_else(|| {
                TantivyError::InvalidArgument(
                    "vector estimator sample-row count exceeds u64".to_string(),
                )
            })?;
        for (left, right) in self.aggregate.iter_mut().zip(&other.aggregate) {
            left.sample_count += right.sample_count;
            left.normalized_error_sum += right.normalized_error_sum;
            left.normalized_error_squared_sum += right.normalized_error_squared_sum;
        }
        for (left_query, right_query) in self.per_query.iter_mut().zip(&other.per_query) {
            for (left, right) in left_query.iter_mut().zip(right_query) {
                left.sample_count += right.sample_count;
                left.normalized_error_sum += right.normalized_error_sum;
                left.normalized_error_squared_sum += right.normalized_error_squared_sum;
            }
        }
        Ok(())
    }
}

/// Deferred code, sidecar, and optional L2-constant slices for one residual layer.
pub(crate) struct QuantizedLayerReader {
    blocks: Arc<Blocks>,
    layer: usize,
    codes: usize,
    scales: usize,
    gammas: usize,
    errors: usize,
    constants: Option<usize>,
    code_stride: usize,
    dim: usize,
    bits: u8,
}

/// Pinned SoA ranges for one contiguous cluster posting.
pub(crate) struct QuantizedLayerBatch {
    pub(crate) residual_norms: Option<OwnedBytes>,
    codes: OwnedBytes,
    scales: OwnedBytes,
    gammas: OwnedBytes,
    error_ratios: OwnedBytes,
    constants: Option<OwnedBytes>,
    rows: Range<usize>,
    code_stride: usize,
}

/// Borrowed sidecar runs for one row range inside an IVF cluster.
pub(crate) struct QuantizedSidecarBatch {
    scales: OwnedBytes,
    gammas: OwnedBytes,
    error_ratios: OwnedBytes,
    constants: Option<OwnedBytes>,
    rows: Range<usize>,
}

impl QuantizedSidecarBatch {
    /// Decodes all selected sidecars together, preserving selected-row order.
    pub(crate) fn decode_selected(
        &self,
        rows: &[usize],
        scales: &mut [f32],
        gammas: &mut [f32],
        errors: &mut [f32],
        constants: &mut [f32],
    ) -> crate::Result<()> {
        if !constants.is_empty() && self.constants.is_none() {
            return Err(DataCorruption::comment_only(
                "quantized L2 field is missing a constants slot",
            )
            .into());
        }
        for (i, &row) in rows.iter().enumerate() {
            let local = self.local_row(row)?;
            let f32_offset = local * QUANTIZED_SCALE_STRIDE;
            let f16_offset = local * QUANTIZED_GAMMA_STRIDE;
            scales[i] =
                f32::from_le_bytes(self.scales[f32_offset..f32_offset + 4].try_into().unwrap());
            gammas[i] = f16_to_f32(u16::from_le_bytes(
                self.gammas[f16_offset..f16_offset + 2].try_into().unwrap(),
            ));
            errors[i] = f16_to_f32(u16::from_le_bytes(
                self.error_ratios[f16_offset..f16_offset + 2]
                    .try_into()
                    .unwrap(),
            ));
            if let Some(bytes) = &self.constants {
                constants[i] =
                    f32::from_le_bytes(bytes[f32_offset..f32_offset + 4].try_into().unwrap());
            }
        }
        Ok(())
    }

    fn local_row(&self, row: usize) -> crate::Result<usize> {
        if !self.rows.contains(&row) {
            return Err(TantivyError::InternalError(format!(
                "quantized row {row} is outside pinned sidecar range {:?}",
                self.rows
            )));
        }
        Ok(row - self.rows.start)
    }

    pub(crate) fn scale(&self, row: usize) -> crate::Result<f32> {
        let local = self.local_row(row)?;
        let start = local * QUANTIZED_SCALE_STRIDE;
        Ok(f32::from_le_bytes(
            self.scales[start..start + QUANTIZED_SCALE_STRIDE]
                .try_into()
                .unwrap(),
        ))
    }

    pub(crate) fn gamma(&self, row: usize) -> crate::Result<f32> {
        let local = self.local_row(row)?;
        let start = local * QUANTIZED_GAMMA_STRIDE;
        let bits = u16::from_le_bytes(
            self.gammas[start..start + QUANTIZED_GAMMA_STRIDE]
                .try_into()
                .unwrap(),
        );
        Ok(f16_to_f32(bits))
    }

    pub(crate) fn error_ratio(&self, row: usize) -> crate::Result<f32> {
        let local = self.local_row(row)?;
        let start = local * QUANTIZED_ERROR_RATIO_STRIDE;
        let bits = u16::from_le_bytes(
            self.error_ratios[start..start + QUANTIZED_ERROR_RATIO_STRIDE]
                .try_into()
                .unwrap(),
        );
        Ok(f16_to_f32(bits))
    }

    #[cfg(test)]
    pub(crate) fn scales(&self) -> &[u8] {
        &self.scales
    }

    #[cfg(test)]
    pub(crate) fn gammas(&self) -> &[u8] {
        &self.gammas
    }

    #[cfg(test)]
    pub(crate) fn error_ratios(&self) -> &[u8] {
        &self.error_ratios
    }
}

impl QuantizedLayerBatch {
    fn local_row(&self, row: usize) -> crate::Result<usize> {
        if !self.rows.contains(&row) {
            return Err(TantivyError::InternalError(format!(
                "quantized row {row} is outside pinned range {:?}",
                self.rows
            )));
        }
        Ok(row - self.rows.start)
    }

    pub(crate) fn code_bytes(&self, row: usize) -> crate::Result<&[u8]> {
        let local = self.local_row(row)?;
        let start = local * self.code_stride;
        Ok(&self.codes[start..start + self.code_stride])
    }

    pub(crate) fn scale(&self, row: usize) -> crate::Result<f32> {
        self.local_row(row)?;
        let local = row - self.rows.start;
        let start = local * QUANTIZED_SCALE_STRIDE;
        Ok(f32::from_le_bytes(
            self.scales[start..start + QUANTIZED_SCALE_STRIDE]
                .try_into()
                .unwrap(),
        ))
    }

    pub(crate) fn gamma(&self, row: usize) -> crate::Result<f32> {
        self.local_row(row)?;
        let local = row - self.rows.start;
        let start = local * QUANTIZED_GAMMA_STRIDE;
        let bits = u16::from_le_bytes(
            self.gammas[start..start + QUANTIZED_GAMMA_STRIDE]
                .try_into()
                .unwrap(),
        );
        Ok(f16_to_f32(bits))
    }

    pub(crate) fn error_ratio(&self, row: usize) -> crate::Result<f32> {
        self.local_row(row)?;
        let local = row - self.rows.start;
        let start = local * QUANTIZED_ERROR_RATIO_STRIDE;
        let bits = u16::from_le_bytes(
            self.error_ratios[start..start + QUANTIZED_ERROR_RATIO_STRIDE]
                .try_into()
                .unwrap(),
        );
        Ok(f16_to_f32(bits))
    }

    pub(crate) fn constant(&self, row: usize) -> crate::Result<Option<f32>> {
        let local = self.local_row(row)?;
        let Some(constants) = &self.constants else {
            return Ok(None);
        };
        let start = local * QUANTIZED_CONSTANT_STRIDE;
        Ok(Some(f32::from_le_bytes(
            constants[start..start + QUANTIZED_CONSTANT_STRIDE]
                .try_into()
                .unwrap(),
        )))
    }

    pub(crate) fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub(crate) fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub(crate) fn gammas(&self) -> &[u8] {
        &self.gammas
    }

    pub(crate) fn error_ratios(&self) -> &[u8] {
        &self.error_ratios
    }

    pub(crate) fn constants(&self) -> Option<&[u8]> {
        self.constants.as_deref()
    }

    pub(crate) fn code_stride(&self) -> usize {
        self.code_stride
    }
}

/// Rejects a decoded sidecar batch whose gammas leave the serialized clamp or
/// whose corrected-error ratios are not finite and non-negative. Runs once per
/// scored batch on the decoded values, over the selected rows only.
pub(crate) fn validate_decoded_sidecar(
    gammas: &[f32],
    error_ratios: &[f32],
    first_row: usize,
) -> crate::Result<()> {
    debug_assert_eq!(gammas.len(), error_ratios.len());
    for (index, &gamma) in gammas.iter().enumerate() {
        if !gamma.is_finite() || !(1.0..=4.0).contains(&gamma) {
            let row = first_row + index;
            return Err(DataCorruption::comment_only(format!(
                "quantized row {row} has invalid cumulative gamma {gamma}; expected finite [1,4]"
            ))
            .into());
        }
    }
    for (index, &error_ratio) in error_ratios.iter().enumerate() {
        if !error_ratio.is_finite() || error_ratio < 0.0 {
            let row = first_row + index;
            return Err(DataCorruption::comment_only(format!(
                "quantized row {row} has invalid corrected error ratio {error_ratio}; expected \
                 finite and non-negative"
            ))
            .into());
        }
    }
    Ok(())
}

/// Splits strictly increasing `rows` into maximal runs of consecutive rows.
fn push_consecutive_runs(rows: &[usize], read_ranges: &mut Vec<Range<usize>>) {
    debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
    let Some(&first) = rows.first() else { return };
    let mut start = first;
    let mut previous = first;
    for &row in &rows[1..] {
        if row != previous + 1 {
            read_ranges.push(start..previous + 1);
            start = row;
        }
        previous = row;
    }
    read_ranges.push(start..previous + 1);
}

fn storage_block_span(slot: &FileSlice, byte_range: Range<usize>) -> Option<(usize, usize)> {
    debug_assert!(byte_range.start < byte_range.end);
    let first = slot.storage_block_ord(byte_range.start)?;
    let last = slot.storage_block_ord(byte_range.end - 1)?;
    Some((first, last))
}

fn append_storage_block_span(
    slot: &FileSlice,
    byte_range: Range<usize>,
    spans: &mut Vec<(usize, usize)>,
) -> bool {
    let Some(span) = storage_block_span(slot, byte_range) else {
        return false;
    };
    spans.push(span);
    true
}

fn merged_storage_block_count(spans: &mut [(usize, usize)]) -> usize {
    spans.sort_unstable_by_key(|&(start, end)| (start, end));
    let mut count = 0usize;
    let mut current: Option<(usize, usize)> = None;
    for &(start, end) in spans.iter() {
        current = match current {
            None => Some((start, end)),
            Some((current_start, current_end)) if start <= current_end.saturating_add(1) => {
                Some((current_start, current_end.max(end)))
            }
            Some((current_start, current_end)) => {
                count += current_end - current_start + 1;
                Some((start, end))
            }
        };
    }
    if let Some((start, end)) = current {
        count += end - start + 1;
    }
    count
}

impl QuantizedLayerReader {
    pub(crate) fn code_stride(&self) -> usize {
        self.code_stride
    }
    fn validate_codes(&self, bytes: &OwnedBytes, rows: &Range<usize>) -> crate::Result<()> {
        for (local, code) in bytes.chunks_exact(self.code_stride).enumerate() {
            if !quantized_code_tail_is_zero(code, self.dim, self.bits) {
                return Err(DataCorruption::comment_only(format!(
                    "quantized row {} has non-zero padding bits for d={} b={}",
                    rows.start + local,
                    self.dim,
                    self.bits
                ))
                .into());
            }
        }
        Ok(())
    }
    pub(crate) fn read_codes(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        let bytes = self.blocks.read_column(self.codes, rows.clone())?;
        self.validate_codes(&bytes, &rows)?;
        Ok(bytes)
    }
    pub(crate) fn read_sidecar(&self, rows: Range<usize>) -> crate::Result<QuantizedSidecarBatch> {
        Ok(QuantizedSidecarBatch {
            scales: self.blocks.read_column(self.scales, rows.clone())?,
            gammas: self.blocks.read_column(self.gammas, rows.clone())?,
            error_ratios: self.blocks.read_column(self.errors, rows.clone())?,
            constants: None,
            rows,
        })
    }
    pub(crate) fn read_constants(&self, rows: Range<usize>) -> crate::Result<Option<OwnedBytes>> {
        self.constants
            .map(|idx| self.blocks.read_column(idx, rows))
            .transpose()
    }
    /// Pins one band's columns in a single request, then restricts views to the requested rows.
    pub(crate) fn read_batch(&self, rows: Range<usize>) -> crate::Result<QuantizedLayerBatch> {
        let b = self.blocks.block_for_range(&rows)?;
        self.read_batch_in_block(b, rows)
    }
    /// Uses a known block without resolving its global row range again.
    pub(crate) fn read_batch_in_block(
        &self,
        b: usize,
        rows: Range<usize>,
    ) -> crate::Result<QuantizedLayerBatch> {
        self.blocks.check_block_rows(b, &rows)?;
        let (span, bytes) = self.blocks.read_band(b, self.layer)?;
        let view = |idx: usize| -> OwnedBytes {
            let column =
                super::blocks::column_range(&self.blocks.slots, self.blocks.rows_in(b), idx);
            let start = self.blocks.block_start(b) + column.start - span.start;
            let stride = self.blocks.slots[idx].stride as usize;
            let first = self.blocks.block_rows[b];
            bytes.slice(start + (rows.start - first) * stride..start + (rows.end - first) * stride)
        };
        let codes = view(self.codes);
        self.validate_codes(&codes, &rows)?;
        Ok(QuantizedLayerBatch {
            codes,
            scales: view(self.scales),
            gammas: view(self.gammas),
            error_ratios: view(self.errors),
            constants: self.constants.map(view),
            residual_norms: (self.layer == 0).then(|| {
                view(
                    self.blocks
                        .slots
                        .iter()
                        .position(|slot| matches!(slot.slot_type, SlotType::ResidualNorms))
                        .expect("validated residual norms column"),
                )
            }),
            rows,
            code_stride: self.code_stride,
        })
    }
    /// Groups code rows only while their page spans overlap, trimming each range to its
    /// first and last selected row. Adjacent disjoint pages start a new group; a straddling row
    /// connects overlapping spans on both pages. Paged storage
    /// copies multi-page requests, so equal page counts do not make a wider available span free.
    /// Storage without page geometry uses consecutive selected rows.
    fn plan_code_slot_reads(
        slot: &FileSlice,
        stride: usize,
        row_origin: usize,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
    ) {
        let span = |row: usize| {
            storage_block_span(
                slot,
                (row - row_origin) * stride..(row + 1 - row_origin) * stride,
            )
        };
        let Some((_, mut end_page)) = span(rows[0]) else {
            push_consecutive_runs(rows, read_ranges);
            return;
        };
        let mut first = rows[0];
        let mut last = first;
        for &row in &rows[1..] {
            let (start, end) = span(row).expect("storage geometry was resolved above");
            if start <= end_page {
                end_page = end_page.max(end);
            } else {
                read_ranges.push(first..last + 1);
                first = row;
                end_page = end;
            }
            last = row;
        }
        read_ranges.push(first..last + 1);
    }

    fn plan_slot_reads(
        slot: &FileSlice,
        stride: usize,
        row_origin: usize,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        debug_assert!(!rows.is_empty());
        debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(rows.iter().all(|row| available_rows.contains(row)));
        let range_start = read_ranges.len();

        block_scratch.clear();
        for &row in rows {
            if !append_storage_block_span(
                slot,
                (row - row_origin) * stride..(row + 1 - row_origin) * stride,
                block_scratch,
            ) {
                push_consecutive_runs(rows, read_ranges);
                return;
            }
        }
        let touched_blocks = merged_storage_block_count(block_scratch);

        block_scratch.clear();
        block_scratch.push(
            storage_block_span(
                slot,
                (available_rows.start - row_origin) * stride
                    ..(available_rows.end - row_origin) * stride,
            )
            .expect("storage geometry was resolved above"),
        );
        let covered_blocks = merged_storage_block_count(block_scratch);
        debug_assert!(touched_blocks <= covered_blocks);
        if touched_blocks == covered_blocks {
            read_ranges.push(available_rows);
            return;
        }

        let mut first_row = rows[0];
        let mut previous_row = rows[0];
        let (_, mut group_end_block) = storage_block_span(
            slot,
            (rows[0] - row_origin) * stride..(rows[0] + 1 - row_origin) * stride,
        )
        .expect("storage geometry was resolved above");
        for &row in &rows[1..] {
            let (start_block, end_block) = storage_block_span(
                slot,
                (row - row_origin) * stride..(row + 1 - row_origin) * stride,
            )
            .expect("storage geometry was resolved above");
            if start_block <= group_end_block {
                group_end_block = group_end_block.max(end_block);
            } else {
                read_ranges.push(first_row..previous_row + 1);
                first_row = row;
                group_end_block = end_block;
            }
            previous_row = row;
        }
        read_ranges.push(first_row..previous_row + 1);

        debug_assert!(read_ranges[range_start..].windows(2).all(|pair| {
            let (_, left_end) = storage_block_span(
                slot,
                (pair[0].start - row_origin) * stride..(pair[0].end - row_origin) * stride,
            )
            .unwrap();
            let (right_start, _) = storage_block_span(
                slot,
                (pair[1].start - row_origin) * stride..(pair[1].end - row_origin) * stride,
            )
            .unwrap();
            left_end < right_start
        }));
    }

    /// Plans global row identifiers against a block-local column and restores global ranges.
    #[cfg(test)]
    pub(crate) fn plan_column_reads(
        &self,
        idx: usize,
        available: Range<usize>,
        rows: &[usize],
        ranges: &mut Vec<Range<usize>>,
        scratch: &mut Vec<(usize, usize)>,
    ) {
        plan_block_column(&self.blocks, idx, available, rows, ranges, scratch).unwrap();
    }
    #[cfg(test)]
    pub(crate) fn plan_code_reads(
        &self,
        available: Range<usize>,
        rows: &[usize],
        ranges: &mut Vec<Range<usize>>,
        scratch: &mut Vec<(usize, usize)>,
    ) {
        self.plan_column_reads(self.codes, available, rows, ranges, scratch);
    }
    /// Resolves a cluster once so ordered code reads need no repeated block lookup.
    pub(crate) fn cluster(&self, row: usize) -> crate::Result<QuantizedClusterReader<'_>> {
        self.cluster_in_block(self.blocks.block_of(row))
    }
    /// Preserves a cluster address already established by the scan.
    pub(crate) fn cluster_in_block(
        &self,
        block: usize,
    ) -> crate::Result<QuantizedClusterReader<'_>> {
        Ok(QuantizedClusterReader {
            layer: self,
            block,
            rows: self.blocks.block_rows[block]..self.blocks.block_rows[block + 1],
            codes: self.blocks.column(block, self.codes)?,
        })
    }
    #[cfg(test)]
    pub(crate) fn read_column(&self, idx: usize, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        self.blocks.read_column(idx, rows)
    }
    pub(crate) fn code_bytes(&self, row: usize) -> crate::Result<OwnedBytes> {
        self.read_codes(row..row + 1)
    }
    pub(crate) fn scale(&self, row: usize) -> crate::Result<f32> {
        self.read_sidecar(row..row + 1)?.scale(row)
    }
    pub(crate) fn gamma(&self, row: usize) -> crate::Result<f32> {
        self.read_sidecar(row..row + 1)?.gamma(row)
    }
    pub(crate) fn error_ratio(&self, row: usize) -> crate::Result<f32> {
        self.read_sidecar(row..row + 1)?.error_ratio(row)
    }
    pub(crate) fn constant(&self, row: usize) -> crate::Result<Option<f32>> {
        Ok(self
            .read_constants(row..row + 1)?
            .map(|b| f32::from_le_bytes(b.as_slice().try_into().unwrap())))
    }
}
/// Block-local code geometry and sidecar addressing for one selected cluster.
pub(crate) struct QuantizedClusterReader<'a> {
    layer: &'a QuantizedLayerReader,
    block: usize,
    pub(crate) rows: Range<usize>,
    codes: FileSlice,
}

impl QuantizedClusterReader<'_> {
    /// Emits code ranges in row order into reusable scratch, with overlap-only page grouping.
    pub(crate) fn plan_codes(&self, rows: &[usize], ranges: &mut Vec<Range<usize>>) {
        ranges.clear();
        QuantizedLayerReader::plan_code_slot_reads(
            &self.codes,
            self.layer.code_stride,
            self.rows.start,
            rows,
            ranges,
        );
    }

    /// Reads an ordered code range directly; code padding is checked before scoring.
    pub(crate) fn read_codes(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        let stride = self.layer.code_stride;
        let bytes = self
            .codes
            .slice((rows.start - self.rows.start) * stride..(rows.end - self.rows.start) * stride)
            .read_vector_bytes()?;
        self.layer.validate_codes(&bytes, &rows)?;
        Ok(bytes)
    }

    /// Pins the complete sidecar span once, from scales through errors or L2 constants.
    /// Small columns are contiguous within a band; reading the span avoids per-column plans
    /// and request sorting. Codes stay separate because multi-page requests copy in paged storage.
    pub(crate) fn read_sidecar(&self) -> crate::Result<QuantizedSidecarBatch> {
        use super::blocks::column_range;
        let layer = self.layer;
        let blocks = &layer.blocks;
        let n = blocks.rows_in(self.block);
        let first = column_range(&blocks.slots, n, layer.scales).start;
        let last = column_range(&blocks.slots, n, layer.constants.unwrap_or(layer.errors)).end;
        let bytes = blocks
            .block_slice(self.block, first..last)?
            .read_vector_bytes()?;
        let view = |idx| {
            let column = column_range(&blocks.slots, n, idx);
            bytes.slice(column.start - first..column.end - first)
        };
        Ok(QuantizedSidecarBatch {
            scales: view(layer.scales),
            gammas: view(layer.gammas),
            error_ratios: view(layer.errors),
            constants: layer.constants.map(view),
            rows: self.rows.clone(),
        })
    }
}

/// Splits increasing rows by block before consulting storage geometry.
#[cfg(test)]
fn plan_block_column(
    blocks: &Blocks,
    idx: usize,
    available: Range<usize>,
    rows: &[usize],
    ranges: &mut Vec<Range<usize>>,
    scratch: &mut Vec<(usize, usize)>,
) -> crate::Result<()> {
    ranges.clear();
    let mut selected = rows;
    while let Some(&row) = selected.first() {
        let b = blocks.block_of(row);
        let first = blocks.block_rows[b];
        let end = blocks.block_rows[b + 1];
        let count = selected.partition_point(|&r| r < end);
        if matches!(
            blocks.slots[idx].slot_type,
            SlotType::QuantLayerCodes { .. }
        ) {
            QuantizedLayerReader::plan_code_slot_reads(
                &blocks.column(b, idx)?,
                blocks.slots[idx].stride as usize,
                first,
                &selected[..count],
                ranges,
            );
        } else {
            QuantizedLayerReader::plan_slot_reads(
                &blocks.column(b, idx)?,
                blocks.slots[idx].stride as usize,
                first,
                available.start.max(first)..available.end.min(end),
                &selected[..count],
                ranges,
                scratch,
            );
        }
        selected = &selected[count..];
    }
    Ok(())
}

/// Stored layer readers and shared query-preparation metadata.
pub(crate) struct QuantizedFieldReader {
    index_ctx: Arc<QuantizedIndexCtx>,
    layers: Vec<QuantizedLayerReader>,
    blocks: Arc<Blocks>,
    norms: usize,
}

/// Storage-planned borrowed views of selected fp32 rows.
pub(crate) struct VectorRowBatch {
    selected_rows: Vec<usize>,
    chunks: Vec<VectorRowChunk>,
    stride: usize,
}

struct VectorRowChunk {
    rows: Range<usize>,
    bytes: OwnedBytes,
}

pub(crate) struct VectorRowBatchIter<'a> {
    batch: &'a VectorRowBatch,
    selected: usize,
    chunk: usize,
}

impl VectorRowBatch {
    pub(crate) fn iter(&self) -> VectorRowBatchIter<'_> {
        VectorRowBatchIter {
            batch: self,
            selected: 0,
            chunk: 0,
        }
    }

    #[cfg(test)]
    fn read_count(&self) -> usize {
        self.chunks.len()
    }
}

impl<'a> Iterator for VectorRowBatchIter<'a> {
    type Item = (usize, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let &row = self.batch.selected_rows.get(self.selected)?;
        while self.batch.chunks[self.chunk].rows.end <= row {
            self.chunk += 1;
        }
        let chunk = &self.batch.chunks[self.chunk];
        debug_assert!(chunk.rows.contains(&row));
        let local = row - chunk.rows.start;
        let start = local * self.batch.stride;
        self.selected += 1;
        Some((row, &chunk.bytes[start..start + self.batch.stride]))
    }
}

impl QuantizedFieldReader {
    pub(crate) fn layers(&self) -> &[QuantizedLayerReader] {
        &self.layers
    }
    pub(crate) fn residual_norm(&self, row: usize) -> crate::Result<f32> {
        let bytes = self.read_residual_norms(row..row + 1)?;
        Ok(f32::from_le_bytes(bytes.as_slice().try_into().unwrap()))
    }
    pub(crate) fn read_residual_norms(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        self.blocks.read_column(self.norms, rows)
    }
    pub(crate) fn index_ctx(&self) -> &Arc<QuantizedIndexCtx> {
        &self.index_ctx
    }
}

/// Every vector field declares exactly the two format entries.
fn validate_vector_entries(composite: &CompositeFile, field: Field) -> crate::Result<()> {
    if composite.field_indices().any(|(_, idx)| idx > 1)
        || !composite
            .field_indices()
            .any(|(f, idx)| f == field && idx == VectorEntry::IdMap.index())
        || composite
            .open_read_with_idx(field, VectorEntry::Data.index())
            .is_none()
    {
        return Err(DataCorruption::comment_only(
            "vector field requires exactly IdMap and Data entries",
        )
        .into());
    }
    Ok(())
}

/// Light field metadata with one shared, fallible search-state initialization.
pub(crate) struct VectorFieldReader {
    options: VectorOptions,
    source: Option<VectorSource>,
    search: OnceLock<crate::Result<Arc<VectorIndexReader>>>,
}

type CentroidSlices = (
    super::header::VectorFileVersion,
    FileSlice,
    FileSlice,
    FileSlice,
    FileSlice,
);

struct VectorSource {
    metadata: BlockMetadata,
    composite: CompositeFile,
    field: Field,
    max_doc: DocId,
    centroid_slots: Option<CentroidSlices>,
}

/// Opens an addressing entry only when a document lookup needs it; failures are shared.
struct DeferredIdMap {
    source: Option<(CompositeFile, Field, DocId)>,
    value: OnceLock<crate::Result<IdMap>>,
}
impl DeferredIdMap {
    /// Stores a map whose bytes need no deferred entry access.
    fn ready(map: IdMap) -> Self {
        Self {
            source: None,
            value: OnceLock::from(Ok(map)),
        }
    }
    /// Initializes the entry once and checks its variant against the stored partition.
    fn get(&self, clustered: bool) -> crate::Result<&IdMap> {
        self.value
            .get_or_init(|| {
                let (composite, field, max_doc) =
                    self.source.as_ref().expect("deferred entry source");
                let entry = composite
                    .open_read_with_idx(*field, VectorEntry::IdMap.index())
                    .unwrap();
                let map = IdMap::open(entry, *max_doc)
                    .map_err(|e| DataCorruption::comment_only(e.to_string()))?;
                if matches!(map, IdMap::DocLocations(_)) != clustered {
                    return Err(DataCorruption::comment_only("partition/id-map mismatch").into());
                }
                Ok(map)
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

/// Per-(segment, field) vector reader: the row store plus, for IVF segments,
/// the routing index. See the module docs for the layout and the
/// pinned-vs-deferred split.
pub struct VectorIndexReader {
    max_doc: DocId,
    options: VectorOptions,
    /// Distinct documents with a vector.
    num_vectors: usize,
    /// `false` for the placeholder built by [`Self::empty`] — the segment has
    /// no vector data for this field at all.
    present: bool,
    /// Document addressing, deferred until a lookup requires the entry.
    id_map: DeferredIdMap,
    /// Deferred row-group columns and their validated geometry.
    rows_slice: Arc<Blocks>,
    index: Option<IvfIndex>,
    quantization: Option<QuantizedFieldReader>,
}

impl VectorFieldReader {
    /// Opens `field`'s vector data in `segment_reader`'s segment. Returns the
    /// metadata placeholder when the segment has no `.vec` file. The vector header is
    /// validated before reading centroid data so unsupported formats have a typed error.
    pub(crate) fn open(segment_reader: &SegmentReader, field: Field) -> crate::Result<Self> {
        let entry = segment_reader.schema().get_field_entry(field);
        let options = match entry.field_type() {
            FieldType::Vector(opts) => opts.clone(),
            _ => {
                return Err(TantivyError::InvalidArgument(format!(
                    "field {:?} is not a vector field",
                    entry.name()
                )));
            }
        };

        let vec_file = match segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string()))
        {
            Ok(file) => file,
            Err(OpenReadError::FileDoesNotExist(_)) => {
                return Ok(Self {
                    options,
                    source: None,
                    search: OnceLock::new(),
                })
            }
            Err(err) => return Err(err.into()),
        };
        let (_, body) = read_vector_header(&vec_file)?;

        let centroid_slots =
            match segment_reader.open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string())) {
                Ok(file) => {
                    let (centroids_version, body) = read_centroid_header(&file)?;
                    let composite = CompositeFile::open(&body)?;
                    match (
                        composite.open_read_with_idx(field, CentroidSlot::Centroids.index()),
                        composite.open_read_with_idx(field, CentroidSlot::Offsets.index()),
                        composite.open_read_with_idx(field, CentroidSlot::Router.index()),
                        composite.open_read_with_idx(field, CentroidSlot::Bounds.index()),
                    ) {
                        (Some(centroids), Some(offsets), Some(router), Some(bounds)) => {
                            Some((centroids_version, centroids, offsets, router, bounds))
                        }
                        (Some(_), Some(_), None, _) => {
                            return Err(TantivyError::InternalError(format!(
                                "vector field {:?} has no router slot",
                                entry.name()
                            )));
                        }
                        (Some(_), Some(_), Some(_), None) => {
                            return Err(TantivyError::InternalError(format!(
                                "vector field {:?} has no bounds slot",
                                entry.name()
                            )));
                        }
                        _ => None,
                    }
                }
                Err(OpenReadError::FileDoesNotExist(_)) => None,
                Err(err) => return Err(err.into()),
            };

        let vec_composite = CompositeFile::open(&body)?;
        validate_vector_entries(&vec_composite, field)?;
        let data = vec_composite
            .open_read_with_idx(field, VectorEntry::Data.index())
            .unwrap();
        let metadata = BlockMetadata::open(data, &options, centroid_slots.is_some())?;
        Ok(Self {
            options,
            source: Some(VectorSource {
                metadata,
                composite: vec_composite,
                field,
                max_doc: segment_reader.max_doc(),
                centroid_slots,
            }),
            search: OnceLock::new(),
        })
    }

    pub(crate) fn metadata(&self) -> Option<Arc<VectorColMetadata>> {
        self.source
            .as_ref()
            .map(|source| Arc::clone(&source.metadata.meta))
    }

    /// Initializes and validates search bytes once, including failures shared by concurrent
    /// callers.
    pub(crate) fn search_reader(&self) -> crate::Result<Arc<VectorIndexReader>> {
        self.search
            .get_or_init(|| VectorIndexReader::open(self).map(Arc::new))
            .clone()
    }
}

impl VectorIndexReader {
    fn open(field: &VectorFieldReader) -> crate::Result<Self> {
        let options = field.options.clone();
        let Some(source) = &field.source else {
            return Ok(Self::empty(options));
        };
        let centroid_slots = source.centroid_slots.clone();
        let id_map = DeferredIdMap {
            source: Some((source.composite.clone(), source.field, source.max_doc)),
            value: OnceLock::new(),
        };
        let index = if let Some((version, centroids, offsets, router_slot, bounds)) = centroid_slots
        {
            Some(IvfIndex::open(
                version,
                &options,
                centroids,
                offsets,
                router_slot,
                bounds,
            )?)
        } else {
            None
        };
        let num_rows = match &index {
            Some(index) => index.num_rows(),
            None => id_map.get(false)?.num_rows() as usize,
        };
        let cluster_rows = index.as_ref().map(|ivf| {
            (0..ivf.num_clusters())
                .map(|b| ivf.cluster_range(b).start)
                .chain(std::iter::once(ivf.num_rows()))
                .collect()
        });
        let rows_slice = Arc::new(Blocks::from_metadata(
            source.metadata.clone(),
            num_rows,
            cluster_rows,
        )?);
        let quantization =
            if let VectorColMetadata::Quantized { layers: quants, .. } = rows_slice.meta.as_ref() {
                let mut layers = Vec::new();
                for (layer, quant) in quants.iter().enumerate() {
                    let mut codes = None;
                    let mut scales = None;
                    let mut gammas = None;
                    let mut errors = None;
                    let mut constants = None;
                    for (idx, slot) in rows_slice.slots.iter().enumerate() {
                        match &slot.slot_type {
                            SlotType::QuantLayerCodes { layer: l, .. } if *l as usize == layer => {
                                codes = Some(idx)
                            }
                            SlotType::QuantLayerScales { layer: l } if *l as usize == layer => {
                                scales = Some(idx)
                            }
                            SlotType::QuantLayerGammas { layer: l } if *l as usize == layer => {
                                gammas = Some(idx)
                            }
                            SlotType::QuantLayerErrors { layer: l } if *l as usize == layer => {
                                errors = Some(idx)
                            }
                            SlotType::QuantLayerConstants { layer: l } if *l as usize == layer => {
                                constants = Some(idx)
                            }
                            _ => (),
                        }
                    }
                    layers.push(QuantizedLayerReader {
                        blocks: Arc::clone(&rows_slice),
                        layer,
                        codes: codes.unwrap(),
                        scales: scales.unwrap(),
                        gammas: gammas.unwrap(),
                        errors: errors.unwrap(),
                        constants,
                        code_stride: quant.code_stride(options.dim()),
                        dim: options.dim(),
                        bits: quant.bits(),
                    });
                }
                Some(QuantizedFieldReader {
                    index_ctx: Arc::new(QuantizedIndexCtx::new(Arc::clone(&rows_slice.meta))?),
                    layers,
                    blocks: Arc::clone(&rows_slice),
                    norms: rows_slice
                        .slots
                        .iter()
                        .position(|slot| matches!(slot.slot_type, SlotType::ResidualNorms))
                        .expect("validated residual norms column"),
                })
            } else {
                None
            };

        let num_vectors = match &index {
            Some(index) => index.num_docs(),
            None => num_rows,
        };
        Ok(Self {
            max_doc: source.max_doc,
            options,
            num_vectors,
            present: true,
            rows_slice,
            id_map,
            index,
            quantization,
        })
    }

    /// The no-data placeholder: zero vectors, no index. Every accessor
    /// behaves as an empty column, so callers never branch on presence.
    /// Returns a vector reader with no rows or routing index.
    pub(crate) fn empty(options: VectorOptions) -> Self {
        let mut bytes = Vec::new();
        super::blocks::write_metadata(&mut bytes, &VectorColMetadata::build_flat(&options))
            .unwrap();
        let end = bytes.len();
        super::blocks::BlockDirectory::new(end as u64)
            .finish(&mut bytes)
            .unwrap();
        let rows_slice = Arc::new(Blocks::open(FileSlice::from(bytes), &options, 0, None).unwrap());
        Self {
            max_doc: 0,
            options,
            num_vectors: 0,
            present: false,
            rows_slice,
            id_map: DeferredIdMap::ready(IdMap::Identity { num_docs: 0 }),
            index: None,
            quantization: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn id_map_initialized(&self) -> bool {
        self.id_map.value.get().is_some()
    }

    /// Returns this segment's stored field metadata, independent of index build settings.
    /// `None` means the segment has no stored vector data; an empty stored field still
    /// returns its metadata.
    pub fn metadata(&self) -> Option<&VectorColMetadata> {
        self.present.then_some(self.rows_slice.meta.as_ref())
    }

    /// Returns the field options.
    pub fn options(&self) -> &VectorOptions {
        &self.options
    }

    /// Returns the vector dimension.
    pub fn dim(&self) -> usize {
        self.options.dim()
    }

    /// Number of distinct docs with a vector value.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Returns whether the field contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// The routing index, present iff the segment's rows are IVF-clustered.
    /// `None` means search must scan the rows exactly.
    /// Returns the optional IVF routing index.
    pub fn index(&self) -> Option<&IvfIndex> {
        self.index.as_ref()
    }

    pub(crate) fn quantization(&self) -> Option<&QuantizedFieldReader> {
        self.quantization.as_ref()
    }

    /// Whether this segment contains the field's quantized storage slots. Field
    /// policy can enable quantization while a small flat segment stores no codes.
    pub fn has_quantized_storage(&self) -> bool {
        self.quantization.is_some()
    }

    /// Storage info for tooling; `None` if the segment has no vector data for
    /// the field.
    /// Returns vector storage information when the field is present.
    pub fn info(&self) -> Option<VectorInfo> {
        if !self.present {
            return None;
        }
        let Some(index) = &self.index else {
            return Some(VectorInfo {
                format: VectorStorageFormat::Flat,
                num_vectors: self.num_vectors,
                num_centroids: None,
                cluster_stats: None,
            });
        };
        let mut empty_clusters = 0;
        let mut min_cluster_size = usize::MAX;
        let mut max_cluster_size = 0;
        let mut total_cluster_size = 0;
        for cluster_size in index.cluster_sizes() {
            empty_clusters += usize::from(cluster_size == 0);
            min_cluster_size = min_cluster_size.min(cluster_size);
            max_cluster_size = max_cluster_size.max(cluster_size);
            total_cluster_size += cluster_size;
        }
        let num_centroids = index.num_clusters();
        let avg_cluster_size = if num_centroids == 0 {
            0.0
        } else {
            total_cluster_size as f64 / num_centroids as f64
        };
        let min_cluster_size = if num_centroids == 0 {
            0
        } else {
            min_cluster_size
        };
        Some(VectorInfo {
            format: VectorStorageFormat::Ivf,
            num_vectors: self.num_vectors,
            num_centroids: Some(num_centroids),
            cluster_stats: Some(VectorClusterStats {
                min_cluster_size,
                max_cluster_size,
                avg_cluster_size,
                empty_clusters,
            }),
        })
    }

    /// Samples distinct live stored vectors for held-out estimator diagnostics.
    ///
    /// # Errors
    ///
    /// Returns an error when a sampled vector row cannot be read or decoded.
    pub fn sample_estimator_pseudo_queries(
        &self,
        count: usize,
        alive: Option<&AliveBitSet>,
    ) -> crate::Result<Option<Vec<VectorEstimatorQuery>>> {
        let (Some(index), Some(_quantization)) = (&self.index, &self.quantization) else {
            return Ok(None);
        };
        if count == 0 {
            return Ok(Some(Vec::new()));
        }

        let mut first_row_by_doc = BTreeMap::new();
        let mut docs = Vec::new();
        for cluster in 0..index.num_clusters() {
            let rows = index.cluster_range(cluster);
            if rows.is_empty() {
                continue;
            }
            self.read_doc_ids(cluster, &mut docs)?;
            for (row, &doc_id) in rows.zip(&docs) {
                if alive.is_some_and(|alive| !alive.is_alive(doc_id)) {
                    continue;
                }
                first_row_by_doc.entry(doc_id).or_insert(row);
            }
        }
        let target = count.min(first_row_by_doc.len());
        if target == 0 {
            return Ok(Some(Vec::new()));
        }
        let candidates: Vec<(DocId, usize)> = first_row_by_doc.into_iter().collect();
        let mut queries = Vec::with_capacity(target);
        for sample in 0..target {
            let candidate = sample * candidates.len() / target;
            let (doc_id, row) = candidates[candidate];
            let values = decode_row::<f32>(&self.vector_bytes_for_row(row)?, self.options.dim())?;
            queries.push(VectorEstimatorQuery {
                values,
                excluded_doc_id: Some(doc_id),
            });
        }
        Ok(Some(queries))
    }

    /// Counts distinct live IVF documents without decoding vector rows.
    pub fn live_distinct_vector_count(&self, alive: Option<&AliveBitSet>) -> crate::Result<usize> {
        self.live_posting_row_count(alive)
    }

    /// Measures normalized estimator errors over a deterministic posting-row sample.
    ///
    /// # Errors
    ///
    /// Returns an error when inputs or persisted quantization data are invalid.
    pub fn measure_estimator_queries(
        &self,
        source: VectorEstimatorSource,
        queries: &[VectorEstimatorQuery],
        sample_rows: usize,
        alive: Option<&AliveBitSet>,
    ) -> crate::Result<Option<VectorEstimatorMeasurements>> {
        Ok(self
            .audit_error_queries(source, queries, sample_rows, alive)?
            .map(|measurements| measurements.estimator))
    }

    /// Audits corrected-error details over a deterministic posting-row sample.
    ///
    /// # Errors
    ///
    /// Returns an error when inputs or persisted quantization data are invalid.
    pub fn audit_error_queries(
        &self,
        source: VectorEstimatorSource,
        queries: &[VectorEstimatorQuery],
        sample_rows: usize,
        alive: Option<&AliveBitSet>,
    ) -> crate::Result<Option<VectorErrorAuditMeasurements>> {
        let (Some(index), Some(quantization)) = (&self.index, &self.quantization) else {
            return Ok(None);
        };
        if sample_rows == 0 {
            return Err(TantivyError::InvalidArgument(
                "vector estimator sample_rows must be greater than zero".to_string(),
            ));
        }
        for query in queries {
            if query.values.len() != self.options.dim() {
                return Err(TantivyError::InvalidArgument(format!(
                    "vector estimator query has dimension {}; expected {}",
                    query.values.len(),
                    self.options.dim()
                )));
            }
        }

        let layer_count = quantization.layers().len();
        let mut measurements = VectorErrorAuditMeasurements {
            source,
            estimator: VectorEstimatorMeasurements {
                source,
                schedule: quantization
                    .index_ctx()
                    .meta
                    .layers()
                    .iter()
                    .map(|q| {
                        (
                            match q {
                                super::metadata::Quantizer::SignPlane { .. } => "SignPlane",
                                super::metadata::Quantizer::GridPlane { .. } => "GridPlane",
                            },
                            q.bits(),
                        )
                    })
                    .collect(),
                aggregate: vec![VectorEstimatorMoments::default(); layer_count],
                per_query: vec![
                    vec![VectorEstimatorMoments::default(); layer_count];
                    queries.len()
                ],
                sample_rows: 0,
                query_count: u32::try_from(queries.len()).map_err(|_| {
                    TantivyError::InvalidArgument(
                        "vector estimator query count exceeds u32".to_string(),
                    )
                })?,
            },
            depths: vec![VectorErrorDepthMeasurements::default(); layer_count],
        };
        if index.num_rows() == 0 || queries.is_empty() {
            return Ok(Some(measurements));
        }

        let measurement_ctx = Arc::clone(quantization.index_ctx());
        let prepared_queries: Vec<QuantizedQueryCtx> = queries
            .iter()
            .map(|query| QuantizedQueryCtx::new(Arc::clone(&measurement_ctx), query.values.clone()))
            .collect();
        let exact_queries: Vec<PreparedQuery<f32>> = queries
            .iter()
            .map(|query| PreparedQuery::new(self.options.metric(), Arc::new(query.values.clone())))
            .collect();
        let live_row_count = self.live_posting_row_count(alive)?;
        let target_rows = sample_rows.min(live_row_count);
        measurements.estimator.sample_rows = u64::try_from(target_rows).map_err(|_| {
            TantivyError::InvalidArgument(
                "vector estimator sample-row count exceeds u64".to_string(),
            )
        })?;
        if target_rows == 0 {
            return Ok(Some(measurements));
        }
        let centroid_stride = self.options.bytes_per_vector();
        let centroid_bytes = index.centroid_bytes()?;
        let mut sampled = 0usize;
        let mut live_rows_seen = 0usize;
        let mut next_sample_ordinal = 0usize;

        let mut docs = Vec::new();
        for cluster in 0..index.num_clusters() {
            let rows = index.cluster_range(cluster);
            if rows.is_empty() {
                continue;
            }
            self.read_doc_ids(cluster, &mut docs)?;
            let centroid_row = &centroid_bytes[cluster * centroid_stride..][..centroid_stride];
            let centroid = decode_row::<f32>(centroid_row, self.options.dim())?;
            for (row, &row_doc) in rows.zip(&docs) {
                if alive.is_some_and(|alive| !alive.is_alive(row_doc)) {
                    continue;
                }
                let live_ordinal = live_rows_seen;
                live_rows_seen += 1;
                if sampled >= target_rows || live_ordinal != next_sample_ordinal {
                    continue;
                }
                sampled += 1;
                next_sample_ordinal = sampled * live_row_count / target_rows;

                let vector_bytes = self.vector_bytes_for_row(row)?;
                let values = decode_row::<f32>(&vector_bytes, self.options.dim())?;
                let residual: Vec<f32> = values
                    .iter()
                    .zip(&centroid)
                    .map(|(&value, &center)| value - center)
                    .collect();
                let mut stored_layers = Vec::with_capacity(layer_count);
                let residual_norm_squared = quantization.residual_norm(row)?;
                if !residual_norm_squared.is_finite() || residual_norm_squared < 0.0 {
                    return Err(DataCorruption::comment_only(format!(
                        "exact-E audit row {row} has invalid stored residual squared norm \
                         {residual_norm_squared}"
                    ))
                    .into());
                }
                for layer in &quantization.layers {
                    let codes = layer.code_bytes(row)?;
                    let sidecar = layer.read_sidecar(row..row + 1)?;
                    let scale = sidecar.scale(row)?;
                    let stored_gamma = sidecar.gamma(row)?;
                    let corrected_error_ratio = sidecar.error_ratio(row)?;
                    let constant = layer.constant(row)?;
                    stored_layers.push((
                        codes,
                        scale,
                        constant,
                        stored_gamma,
                        corrected_error_ratio,
                    ));
                }

                let regenerated = cascade::audit_prefix_error_model(
                    &residual,
                    &measurement_ctx.specs,
                    &measurement_ctx.grids,
                );
                if regenerated.prefixes.len() != stored_layers.len() {
                    return Err(DataCorruption::comment_only(format!(
                        "exact-E audit row {row} regenerated {} prefixes; stored {}",
                        regenerated.prefixes.len(),
                        stored_layers.len()
                    ))
                    .into());
                }

                for (depth, ((codes, scale, _, stored_gamma, corrected_error_ratio), prefix)) in
                    stored_layers.iter().zip(&regenerated.prefixes).enumerate()
                {
                    if codes.as_slice() != prefix.codes.as_slice()
                        || scale.to_bits() != prefix.layer_scale.to_bits()
                        || stored_gamma.to_bits() != prefix.gamma.f16_value().to_bits()
                        || corrected_error_ratio.to_bits()
                            != prefix.corrected_error_ratio.f16_value().to_bits()
                    {
                        return Err(DataCorruption::comment_only(format!(
                            "exact-E audit row {row} depth {} does not reproduce its stored \
                             codes, scale, gamma, and corrected error",
                            depth + 1
                        ))
                        .into());
                    }
                    let depth_measurements = &mut measurements.depths[depth];
                    depth_measurements
                        .residual_norm_squared
                        .observe(f64::from(residual_norm_squared));
                    depth_measurements
                        .stored_gamma
                        .observe(f64::from(*stored_gamma));
                    depth_measurements.raw_gamma.observe(prefix.gamma.raw);
                    depth_measurements.zero_scale_count += u64::from(*scale == 0.0);
                    depth_measurements.gamma_lower_clamp_count += u64::from(prefix.gamma.raw < 1.0);
                    depth_measurements.gamma_upper_clamp_count += u64::from(prefix.gamma.raw > 4.0);
                    depth_measurements
                        .corrected_error_ratio
                        .observe(f64::from(*corrected_error_ratio));
                }

                for (query_idx, query) in prepared_queries.iter().enumerate() {
                    if queries[query_idx].excluded_doc_id == Some(row_doc) {
                        continue;
                    }
                    let cluster_score = self
                        .options
                        .metric()
                        .similarity_bytes::<f32>(query.query(), centroid_row)
                        .score();
                    let query_norm = query.score_query_norm(cluster_score);
                    let query_norm_squared = query_norm * query_norm;
                    let base = if self.options.metric() == Metric::L2 {
                        cluster_score - residual_norm_squared
                    } else {
                        cluster_score
                    };
                    let exact_score = exact_queries[query_idx].score_doc_bytes(&vector_bytes);
                    let mut raw_prefix_estimate = 0.0_f32;
                    let mut sign_query_error_term = 0.0_f32;
                    let mut arithmetic = ArithmeticError::default();
                    for (depth, (codes, scale, constant, stored_gamma, corrected_error_ratio)) in
                        stored_layers.iter().enumerate()
                    {
                        raw_prefix_estimate = diagnostic_advance_raw_prefix(
                            query,
                            self.options.metric(),
                            depth,
                            codes,
                            *scale,
                            *constant,
                            raw_prefix_estimate,
                            cluster_score,
                            residual_norm_squared,
                            &mut arithmetic,
                        )?;
                        if matches!(measurement_ctx.specs[depth].kind, cascade::LayerKind::Sign) {
                            sign_query_error_term +=
                                *scale * *scale * query.query_error_squared(depth) as f32;
                        }
                        let gamma = *stored_gamma;
                        let model_sigma = quantized_model_sigma(
                            self.options.metric(),
                            self.options.dim(),
                            residual_norm_squared,
                            *corrected_error_ratio,
                            gamma,
                            query_norm_squared,
                            sign_query_error_term,
                        );
                        let model_sigma = arithmetic.sigma(
                            self.options.metric(),
                            model_sigma,
                            gamma,
                            raw_prefix_estimate,
                            base,
                        );
                        measurements.depths[depth]
                            .sigma
                            .observe(f64::from(model_sigma));
                        let metric_factor = if self.options.metric() == Metric::L2 {
                            2.0
                        } else {
                            1.0
                        };
                        if model_sigma > 0.0 && model_sigma.is_finite() {
                            let gamma_round_trip_band_error = f64::from(metric_factor)
                                * (f64::from(gamma)
                                    - f64::from(regenerated.prefixes[depth].gamma.clamped))
                                * f64::from(raw_prefix_estimate)
                                / f64::from(model_sigma);
                            measurements.depths[depth]
                                .gamma_round_trip_band_error
                                .observe(gamma_round_trip_band_error);
                        }
                        let corrected_prefix = corrected_quantized_estimate(
                            self.options.metric(),
                            gamma,
                            raw_prefix_estimate,
                            base,
                        );
                        observe_corrected_prefix(
                            &mut measurements.estimator,
                            query_idx,
                            depth,
                            exact_score,
                            corrected_prefix,
                            f64::from(model_sigma),
                        );
                    }
                }
            }
        }
        Ok(Some(measurements))
    }

    /// Audits confidence-cone survivors across all clusters.
    ///
    /// # Errors
    ///
    /// Returns an error when audit inputs or persisted quantization data are invalid.
    pub fn audit_error_cone(
        &self,
        queries: &[VectorEstimatorQuery],
        alive: Option<&AliveBitSet>,
    ) -> crate::Result<Option<VectorErrorConeAuditMeasurements>> {
        let (Some(index), Some(quantization)) = (&self.index, &self.quantization) else {
            return Ok(None);
        };
        if queries.len() != ERROR_CONE_QUERY_COUNT {
            return Err(TantivyError::InvalidArgument(format!(
                "exact-E cone audit requires exactly {ERROR_CONE_QUERY_COUNT} external queries; \
                 received {}",
                queries.len()
            )));
        }
        for query in queries {
            if query.excluded_doc_id.is_some() {
                return Err(TantivyError::InvalidArgument(
                    "exact-E cone audit accepts external queries only".to_string(),
                ));
            }
            if query.values.len() != self.options.dim() {
                return Err(TantivyError::InvalidArgument(format!(
                    "exact-E cone audit query has dimension {}; expected {}",
                    query.values.len(),
                    self.options.dim()
                )));
            }
            if query.values.iter().any(|value| !value.is_finite()) {
                return Err(TantivyError::InvalidArgument(
                    "exact-E cone audit queries must contain only finite values".to_string(),
                ));
            }
        }

        let measurement_ctx = Arc::clone(quantization.index_ctx());
        let live_docs = self.live_distinct_vector_count(alive)?;
        if live_docs < ERROR_CONE_TOP_K {
            return Err(TantivyError::InvalidArgument(format!(
                "exact-E cone audit requires at least {ERROR_CONE_TOP_K} live documents; found \
                 {live_docs}"
            )));
        }

        let row_count = index.num_rows();
        let layer_count = measurement_ctx.specs.len();
        let mut gammas = vec![vec![f32::NAN; row_count]; layer_count];
        let mut stored_scales = vec![vec![f32::NAN; row_count]; layer_count];
        let mut corrected_error_ratios = vec![vec![f32::NAN; row_count]; layer_count];
        let mut residual_norms_squared = vec![f32::NAN; row_count];
        let centroid_stride = self.options.bytes_per_vector();
        let centroid_bytes = index.centroid_bytes()?;

        let mut docs = Vec::new();
        for cluster in 0..index.num_clusters() {
            let rows = index.cluster_range(cluster);
            if rows.is_empty() {
                continue;
            }
            self.read_doc_ids(cluster, &mut docs)?;
            for (row, &doc) in rows.zip(&docs) {
                if alive.is_some_and(|alive| !alive.is_alive(doc)) {
                    continue;
                }
                for (depth, layer) in quantization.layers.iter().enumerate() {
                    let sidecar = layer.read_sidecar(row..row + 1)?;
                    let scale = sidecar.scale(row)?;
                    let stored_gamma = sidecar.gamma(row)?;
                    stored_scales[depth][row] = scale;
                    gammas[depth][row] = stored_gamma;
                    corrected_error_ratios[depth][row] = sidecar.error_ratio(row)?;
                }
                residual_norms_squared[row] = quantization.residual_norm(row)?;
                if !residual_norms_squared[row].is_finite() || residual_norms_squared[row] < 0.0 {
                    return Err(DataCorruption::comment_only(format!(
                        "exact-E cone row {row} has invalid stored residual squared norm {}",
                        residual_norms_squared[row]
                    ))
                    .into());
                }
            }
        }

        let metric = self.options.metric();
        let mut measurements = VectorErrorConeAuditMeasurements {
            query_count: ERROR_CONE_QUERY_COUNT as u32,
            top_k: ERROR_CONE_TOP_K as u32,
            depths: (0..layer_count)
                .map(|_| QUANTIZED_BOUNDARY_KAPPA)
                .map(VectorErrorConeDepthMeasurements::new)
                .collect(),
        };
        for query_input in queries {
            let query =
                QuantizedQueryCtx::new(Arc::clone(&measurement_ctx), query_input.values.clone());
            let exact_query =
                PreparedQuery::<f32>::new(metric, Arc::new(query_input.values.clone()));
            let mut cluster_scores = Vec::with_capacity(index.num_clusters());
            let mut cluster_query_norms = Vec::with_capacity(index.num_clusters());
            for cluster in 0..index.num_clusters() {
                let centroid_row = &centroid_bytes[cluster * centroid_stride..][..centroid_stride];
                let score = metric
                    .similarity_bytes::<f32>(query.query(), centroid_row)
                    .score();
                cluster_scores.push(score);
                cluster_query_norms.push(query.score_query_norm(score));
            }

            let mut exact_by_doc: BTreeMap<DocId, f32> = BTreeMap::new();
            let mut candidates = ErrorConeCandidates::default();
            candidates.rows.reserve(row_count);
            candidates.docs.reserve(row_count);
            candidates.raw_prefixes.reserve(row_count);
            candidates.sign_query_error_terms.reserve(row_count);
            candidates.estimates.reserve(row_count);
            candidates.sigmas.reserve(row_count);
            for cluster in 0..index.num_clusters() {
                let rows = index.cluster_range(cluster);
                if rows.is_empty() {
                    continue;
                }
                self.read_doc_ids(cluster, &mut docs)?;
                let layer = quantization.layers()[0].read_batch_in_block(cluster, rows.clone())?;
                for (row, &doc) in rows.zip(&docs) {
                    if alive.is_some_and(|alive| !alive.is_alive(doc)) {
                        continue;
                    }
                    let bytes = self.vector_bytes_for_row(row)?;
                    let exact_score = exact_query.score_doc_bytes(&bytes);
                    match exact_by_doc.entry(doc) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(exact_score);
                        }
                        std::collections::btree_map::Entry::Occupied(entry) => {
                            if entry.get().to_bits() != exact_score.to_bits() {
                                return Err(DataCorruption::comment_only(format!(
                                    "error cone found duplicate rows with different scores for \
                                     doc {doc}"
                                ))
                                .into());
                            }
                        }
                    }
                    let scale = layer.scale(row)?;
                    let constant = layer.constant(row)?;
                    let gamma = gammas[0][row];
                    let residual_norm_squared = residual_norms_squared[row];
                    let mut arithmetic = ArithmeticError::default();
                    let raw_prefix = diagnostic_advance_raw_prefix(
                        &query,
                        metric,
                        0,
                        layer.code_bytes(row)?,
                        scale,
                        constant,
                        0.0,
                        cluster_scores[cluster],
                        residual_norm_squared,
                        &mut arithmetic,
                    )?;
                    let base = if metric == Metric::L2 {
                        cluster_scores[cluster] - residual_norm_squared
                    } else {
                        cluster_scores[cluster]
                    };
                    let estimate = corrected_quantized_estimate(metric, gamma, raw_prefix, base);
                    let sign_query_error_term =
                        if matches!(measurement_ctx.specs[0].kind, cascade::LayerKind::Sign) {
                            scale * scale * query.query_error_squared(0) as f32
                        } else {
                            0.0
                        };
                    let query_norm = cluster_query_norms[cluster];
                    let sigma = quantized_model_sigma(
                        metric,
                        self.options.dim(),
                        residual_norm_squared,
                        corrected_error_ratios[0][row],
                        gamma,
                        query_norm * query_norm,
                        sign_query_error_term,
                    );
                    let sigma = arithmetic.sigma(metric, sigma, gamma, raw_prefix, base);
                    if !estimate.is_finite() || !sigma.is_finite() {
                        return Err(DataCorruption::comment_only(format!(
                            "exact-E cone row {row} produced a non-finite depth-1 estimate or \
                             sigma"
                        ))
                        .into());
                    }
                    candidates.push(row, doc, raw_prefix, sign_query_error_term, estimate, sigma);
                    *candidates.arithmetic_errors.last_mut().unwrap() = arithmetic;
                }
            }

            let mut exact_docs: Vec<(DocId, f32)> = exact_by_doc.into_iter().collect();
            exact_docs.sort_unstable_by(|(left_doc, left_score), (right_doc, right_score)| {
                right_score
                    .total_cmp(left_score)
                    .then(left_doc.cmp(right_doc))
            });
            let exact_top_docs: Vec<DocId> = exact_docs
                .into_iter()
                .take(ERROR_CONE_TOP_K)
                .map(|(doc, _)| doc)
                .collect();
            debug_assert_eq!(exact_top_docs.len(), ERROR_CONE_TOP_K);

            for depth in 0..layer_count {
                if depth != 0 {
                    let mut cluster = 0usize;
                    for candidate in 0..candidates.len() {
                        let row = candidates.rows[candidate];
                        while cluster < index.num_clusters()
                            && index.cluster_range(cluster).end <= row
                        {
                            cluster += 1;
                        }
                        if cluster == index.num_clusters()
                            || !index.cluster_range(cluster).contains(&row)
                        {
                            return Err(DataCorruption::comment_only(format!(
                                "exact-E cone survivor row {row} is outside IVF cluster ranges"
                            ))
                            .into());
                        }
                        let layer = &quantization.layers()[depth];
                        let scale = stored_scales[depth][row];
                        let constant = layer.constant(row)?;
                        candidates.raw_prefixes[candidate] = diagnostic_advance_raw_prefix(
                            &query,
                            metric,
                            depth,
                            &layer.code_bytes(row)?,
                            scale,
                            constant,
                            candidates.raw_prefixes[candidate],
                            cluster_scores[cluster],
                            residual_norms_squared[row],
                            &mut candidates.arithmetic_errors[candidate],
                        )?;
                        if matches!(measurement_ctx.specs[depth].kind, cascade::LayerKind::Sign) {
                            candidates.sign_query_error_terms[candidate] +=
                                scale * scale * query.query_error_squared(depth) as f32;
                        }
                        let gamma = gammas[depth][row];
                        let residual_norm_squared = residual_norms_squared[row];
                        let base = if metric == Metric::L2 {
                            cluster_scores[cluster] - residual_norm_squared
                        } else {
                            cluster_scores[cluster]
                        };
                        candidates.estimates[candidate] = corrected_quantized_estimate(
                            metric,
                            gamma,
                            candidates.raw_prefixes[candidate],
                            base,
                        );
                        let query_norm = cluster_query_norms[cluster];
                        let sigma = quantized_model_sigma(
                            metric,
                            self.options.dim(),
                            residual_norm_squared,
                            corrected_error_ratios[depth][row],
                            gamma,
                            query_norm * query_norm,
                            candidates.sign_query_error_terms[candidate],
                        );
                        let sigma = candidates.arithmetic_errors[candidate].sigma(
                            metric,
                            sigma,
                            gamma,
                            candidates.raw_prefixes[candidate],
                            base,
                        );
                        candidates.sigmas[candidate] = sigma;
                        if !candidates.estimates[candidate].is_finite() || !sigma.is_finite() {
                            return Err(DataCorruption::comment_only(format!(
                                "exact-E cone row {row} produced a non-finite depth-{} estimate \
                                 or sigma",
                                depth + 1
                            ))
                            .into());
                        }
                    }
                }
                let scored = candidates.len();
                candidates.band(ERROR_CONE_TOP_K, QUANTIZED_BOUNDARY_KAPPA);
                observe_error_cone_depth(
                    &mut measurements.depths[depth],
                    scored,
                    &candidates,
                    &exact_top_docs,
                );
            }
        }
        Ok(Some(measurements))
    }

    /// Per-cluster posting-list sizes in cluster order — the distribution
    /// behind [`Self::info`]'s aggregate cluster stats. `None` when the
    /// field's storage is not IVF.
    /// Returns posting sizes in cluster order for IVF storage.
    pub fn cluster_sizes(&self) -> Option<Vec<u32>> {
        self.index
            .as_ref()
            .map(|index| index.cluster_sizes().map(|size| size as u32).collect())
    }

    /// Returns the number of live IVF posting rows.
    pub fn live_posting_row_count(&self, alive: Option<&AliveBitSet>) -> crate::Result<usize> {
        let Some(index) = &self.index else {
            return Ok(0);
        };
        let Some(alive) = alive else {
            return Ok(index.num_rows());
        };
        let mut docs = Vec::new();
        let mut count = 0;
        for cluster in 0..index.num_clusters() {
            self.read_doc_ids(cluster, &mut docs)?;
            count += docs.iter().filter(|&&doc| alive.is_alive(doc)).count();
        }
        Ok(count)
    }

    /// Returns whether the document has a vector, propagating storage errors.
    pub fn contains(&self, doc_id: DocId) -> crate::Result<bool> {
        Ok(self.row_id(doc_id)?.is_some())
    }

    /// Returns one document's raw little-endian vector bytes, or None if absent.
    ///
    /// # Errors
    ///
    /// Returns an error when the vector row cannot be read.
    pub fn vector_bytes(&self, doc_id: DocId) -> crate::Result<Option<OwnedBytes>> {
        if self.rows_slice.clustered() {
            let Some(location) = self
                .id_map
                .get(true)?
                .locate(doc_id, &self.rows_slice.block_rows)
                .map_err(|e| DataCorruption::comment_only(e.to_string()))?
            else {
                return Ok(None);
            };
            let stride = self.options.bytes_per_vector();
            let start = location.local as usize * stride;
            return Ok(Some(
                self.rows_slice
                    .column(location.cluster as usize, 0)?
                    .slice(start..start + stride)
                    .read_vector_bytes()?,
            ));
        }
        let Some(row) = self.row_id(doc_id)? else {
            return Ok(None);
        };
        self.vector_bytes_for_row(row).map(Some)
    }

    /// Returns one dense vector row by row index without a document lookup.
    ///
    /// # Errors
    ///
    /// Returns an error when `row` is out of bounds or cannot be read.
    pub fn vector_bytes_for_row(&self, row: usize) -> crate::Result<OwnedBytes> {
        if row >= *self.rows_slice.block_rows.last().unwrap() {
            return Err(TantivyError::InvalidArgument(format!(
                "vector row {row} is out of bounds"
            )));
        }
        self.rows_slice.read_column(0, row..row + 1)
    }

    /// Fetches increasing vector rows through a storage-aware range plan.
    pub(crate) fn read_vector_rows_planned(
        &self,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) -> crate::Result<VectorRowBatch> {
        let num_rows = *self.rows_slice.block_rows.last().unwrap();
        if rows.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(TantivyError::InvalidArgument(
                "planned vector rows must be strictly increasing".to_string(),
            ));
        }
        if let Some(&row) = rows.last() {
            if row >= num_rows {
                return Err(TantivyError::InvalidArgument(format!(
                    "vector row {row} is out of bounds"
                )));
            }
        } else {
            read_ranges.clear();
            block_scratch.clear();
            return Ok(VectorRowBatch {
                selected_rows: Vec::new(),
                chunks: Vec::new(),
                stride: self.options.bytes_per_vector(),
            });
        }

        let stride = self.options.bytes_per_vector();
        read_ranges.clear();
        let mut chunks = Vec::with_capacity(read_ranges.capacity().min(rows.len()));
        let mut selected = rows;
        while let Some(&row) = selected.first() {
            let block = self.rows_slice.block_of(row);
            let first = self.rows_slice.block_rows[block];
            let end = self.rows_slice.block_rows[block + 1];
            let count = selected.partition_point(|&r| r < end);
            let column = self.rows_slice.column(block, 0)?;
            let range_start = read_ranges.len();
            QuantizedLayerReader::plan_slot_reads(
                &column,
                stride,
                first,
                first..end,
                &selected[..count],
                read_ranges,
                block_scratch,
            );
            // The plan already establishes the block and row origin; reads keep that address.
            for row_range in read_ranges[range_start..].iter().cloned() {
                let bytes = column
                    .slice((row_range.start - first) * stride..(row_range.end - first) * stride)
                    .read_vector_bytes()?;
                chunks.push(VectorRowChunk {
                    rows: row_range,
                    bytes,
                });
            }
            selected = &selected[count..];
        }
        Ok(VectorRowBatch {
            selected_rows: rows.to_vec(),
            chunks,
            stride,
        })
    }

    /// Reads a clustered document id from its Data column, or selects a flat bitmap row.
    pub fn doc_id_at(&self, row: usize) -> crate::Result<DocId> {
        if row >= self.num_vectors() {
            return Err(TantivyError::InvalidArgument(format!(
                "vector row {row} is out of bounds"
            )));
        }
        Ok(self.row_doc_ids(row..row + 1)?[0])
    }
    /// Reads and validates ascending document ids without opening the clustered IdMap.
    pub(crate) fn row_doc_ids(&self, rows: Range<usize>) -> crate::Result<Vec<DocId>> {
        if !self.rows_slice.clustered() {
            let map = self.id_map.get(false)?;
            return Ok(rows.map(|row| map.doc_at(row as u32)).collect());
        }
        let mut result = Vec::with_capacity(rows.len());
        let mut docs = Vec::new();
        let mut row = rows.start;
        while row < rows.end {
            let cluster = self.rows_slice.block_of(row);
            self.read_doc_ids(cluster, &mut docs)?;
            let first = self.rows_slice.block_rows[cluster];
            let end = rows.end.min(self.rows_slice.block_rows[cluster + 1]);
            result.extend_from_slice(&docs[row - first..end - first]);
            row = end;
        }
        Ok(result)
    }
    /// Returns sorted document ids assigned to a cluster, or None for an invalid cluster.
    pub fn cluster_doc_ids(&self, cluster: usize) -> crate::Result<Option<Vec<DocId>>> {
        let Some(index) = &self.index else {
            return Ok(None);
        };
        if cluster >= index.num_clusters() {
            return Ok(None);
        }
        let mut docs = Vec::new();
        self.read_doc_ids(cluster, &mut docs)?;
        Ok(Some(docs))
    }
    /// Resolves one document by direct location or flat bitmap rank, validating stored coordinates.
    pub(crate) fn row_id(&self, doc_id: DocId) -> crate::Result<Option<usize>> {
        let clustered = self.rows_slice.clustered();
        let map = self.id_map.get(clustered)?;
        if clustered {
            let location = map
                .locate(doc_id, &self.rows_slice.block_rows)
                .map_err(|e| DataCorruption::comment_only(e.to_string()))?;
            Ok(location
                .map(|loc| self.rows_slice.block_rows[loc.cluster as usize] + loc.local as usize))
        } else {
            Ok(map.rank_if_exists(doc_id).map(|row| row as usize))
        }
    }
    /// Reads only a cluster's document column, validating bounds and ordering while decoding.
    pub(crate) fn read_doc_ids(&self, cluster: usize, out: &mut Vec<DocId>) -> crate::Result<()> {
        out.clear();
        let blocks = &self.rows_slice;
        if !blocks.clustered() || cluster >= blocks.block_rows.len() - 1 {
            return Err(DataCorruption::comment_only("invalid DocIds cluster").into());
        }
        if blocks.rows_in(cluster) == 0 {
            return Ok(());
        }
        let idx = blocks
            .slots
            .iter()
            .position(|slot| matches!(slot.slot_type, SlotType::DocIds))
            .ok_or_else(|| DataCorruption::comment_only("missing DocIds column"))?;
        let bytes = blocks.column(cluster, idx)?.read_vector_bytes()?;
        out.reserve(blocks.rows_in(cluster));
        let mut previous = None;
        for bytes in bytes.chunks_exact(std::mem::size_of::<DocId>()) {
            let doc = DocId::from_le_bytes(bytes.try_into().unwrap());
            if doc >= self.max_doc || previous.is_some_and(|prev| prev >= doc) {
                return Err(DataCorruption::comment_only(format!(
                    "cluster {cluster} DocIds must ascend and be below max_doc {}",
                    self.max_doc
                ))
                .into());
            }
            out.push(doc);
            previous = Some(doc);
        }
        Ok(())
    }

    /// Reads a cluster's full-precision rows without document ids or quantized columns.
    pub(crate) fn read_cluster_rows(&self, cluster: usize) -> crate::Result<OwnedBytes> {
        let rows = self.rows_slice.block_rows[cluster]..self.rows_slice.block_rows[cluster + 1];
        self.rows_slice.read_column(0, rows)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use quant_model::f16::f32_to_f16;

    use super::super::quantization::{quantized_code_stride, VectorQuantizationConfig};
    use super::*;
    use crate::directory::{CompositeWrite, FileHandle};

    type TrackedReads = Arc<Mutex<Vec<Range<usize>>>>;

    #[derive(Debug)]
    struct BlockTrackedBytes {
        bytes: Vec<u8>,
        reads: TrackedReads,
        block_len: usize,
    }

    impl HasLen for BlockTrackedBytes {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl FileHandle for BlockTrackedBytes {
        fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(self.bytes[range].to_vec()))
        }

        fn storage_block_len(&self) -> Option<usize> {
            Some(self.block_len)
        }
    }

    fn test_layer(
        codes: FileSlice,
        sidecar: FileSlice,
        constants: Option<FileSlice>,
        offsets: &[usize],
        dim: usize,
        bits: u8,
    ) -> QuantizedLayerReader {
        test_layer_tracked(codes, sidecar, constants, offsets, dim, bits, None)
    }
    fn test_layer_tracked(
        codes: FileSlice,
        sidecar: FileSlice,
        constants: Option<FileSlice>,
        offsets: &[usize],
        dim: usize,
        bits: u8,
        tracking: Option<(TrackedReads, usize)>,
    ) -> QuantizedLayerReader {
        use super::super::blocks::{block_len, column_range, pad, write_metadata};
        let metric = if constants.is_some() {
            Metric::L2
        } else {
            Metric::Dot
        };
        let opts = VectorOptions::new(dim, metric);
        let config = VectorQuantizationConfig::materialize(
            "v".into(),
            &opts,
            vec![super::super::VectorQuantizationLayer { bits, seed: 0 }],
        )
        .unwrap();
        let meta = VectorColMetadata::build_ivf(&opts, Some(&config)).unwrap();
        let slots = meta.slots();
        let mut out = Vec::new();
        write_metadata(&mut out, &meta).unwrap();
        let mut directory = super::super::blocks::BlockDirectory::new(out.len() as u64);
        let codes = codes.read_bytes().unwrap();
        let sidecar = sidecar.read_bytes().unwrap();
        let constants = constants.map(|s| s.read_bytes().unwrap());
        let stride = quantized_code_stride(dim, bits);
        for rows in offsets.windows(2) {
            let n = rows[1] - rows[0];
            let start = out.len();
            for (idx, slot) in slots.iter().enumerate() {
                let range = column_range(&slots, n, idx);
                let padding = start + range.start - out.len();
                pad(&mut out, padding).unwrap();
                let bytes: &[u8] = match &slot.slot_type {
                    SlotType::QuantLayerCodes { .. } if !codes.is_empty() => {
                        &codes[rows[0] * stride..rows[1] * stride]
                    }
                    SlotType::QuantLayerScales { .. } if !sidecar.is_empty() => {
                        &sidecar[rows[0] * 8..rows[0] * 8 + n * 4]
                    }
                    SlotType::QuantLayerGammas { .. } if !sidecar.is_empty() => {
                        &sidecar[rows[0] * 8 + n * 4..rows[0] * 8 + n * 6]
                    }
                    SlotType::QuantLayerErrors { .. } if !sidecar.is_empty() => {
                        &sidecar[rows[0] * 8 + n * 6..rows[1] * 8]
                    }
                    SlotType::QuantLayerConstants { .. } => {
                        &constants.as_ref().unwrap()[rows[0] * 4..rows[1] * 4]
                    }
                    _ => &[],
                };
                if matches!(slot.slot_type, SlotType::DocIds) {
                    for row in rows[0]..rows[1] {
                        out.extend_from_slice(&(row as u32).to_le_bytes());
                    }
                } else if bytes.is_empty() {
                    pad(&mut out, range.len()).unwrap();
                } else {
                    out.extend_from_slice(bytes);
                }
            }
            let padding = start + block_len(&slots, n) - out.len();
            pad(&mut out, padding).unwrap();
            directory.push(out.len() as u64, rows[1] as u32);
        }
        directory.finish(&mut out).unwrap();
        let entry = if let Some((reads, block_len)) = tracking {
            FileSlice::new(Arc::new(BlockTrackedBytes {
                bytes: out,
                reads,
                block_len,
            }))
        } else {
            FileSlice::from(out)
        };
        let blocks = Arc::new(
            Blocks::open(
                entry,
                &opts,
                *offsets.last().unwrap(),
                Some(offsets.to_vec()),
            )
            .unwrap(),
        );
        QuantizedLayerReader {
            blocks,
            layer: 0,
            codes: 3,
            scales: 4,
            gammas: 5,
            errors: 6,
            constants: constants.map(|_| 7),
            code_stride: stride,
            dim,
            bits,
        }
    }

    fn test_sidecar(scales: &[f32], gammas: &[f32]) -> Vec<u8> {
        test_sidecar_with_error_ratios(scales, gammas, &vec![0.25; scales.len()])
    }

    fn test_sidecar_with_error_ratios(
        scales: &[f32],
        gammas: &[f32],
        corrected_error_ratios: &[f32],
    ) -> Vec<u8> {
        assert_eq!(scales.len(), gammas.len());
        assert_eq!(scales.len(), corrected_error_ratios.len());
        scales
            .iter()
            .flat_map(|scale| scale.to_le_bytes())
            .chain(
                gammas
                    .iter()
                    .flat_map(|&gamma| f32_to_f16(gamma).to_le_bytes()),
            )
            .chain(
                corrected_error_ratios
                    .iter()
                    .flat_map(|&error_ratio| f32_to_f16(error_ratio).to_le_bytes()),
            )
            .collect()
    }

    fn tracked_id_composite(ids: Vec<u8>, reads: &Arc<Mutex<Vec<Range<usize>>>>) -> CompositeFile {
        let mut bytes = Vec::new();
        let mut writer = crate::directory::CompositeWrite::wrap(&mut bytes);
        writer
            .for_field_with_idx(Field::from_field_id(0), VectorEntry::IdMap.index())
            .write_all(&ids)
            .unwrap();
        writer.close().unwrap();
        let composite = CompositeFile::open(&FileSlice::new(Arc::new(BlockTrackedBytes {
            bytes,
            reads: Arc::clone(reads),
            block_len: 16,
        })))
        .unwrap();
        reads.lock().unwrap().clear();
        composite
    }

    // Opening reads only the tag; a lookup reads exactly its eight-byte record.
    #[test]
    fn location_lookup_reads_one_record_and_defers_entry() -> crate::Result<()> {
        use super::super::flat::id_map::DocLocation;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut bytes = Vec::new();
        IdMap::serialize_locations(
            &[
                DocLocation::ABSENT,
                DocLocation {
                    cluster: 0,
                    local: 0,
                },
                DocLocation {
                    cluster: 0,
                    local: 1,
                },
            ],
            &mut bytes,
        )?;
        let map = DeferredIdMap {
            source: Some((
                tracked_id_composite(bytes, &reads),
                Field::from_field_id(0),
                3,
            )),
            value: OnceLock::new(),
        };
        assert!(reads.lock().unwrap().is_empty());
        let opened = map.get(true)?;
        assert_eq!(&*reads.lock().unwrap(), &[0..1]);
        reads.lock().unwrap().clear();
        assert_eq!(
            opened.locate(2, &[0, 2])?,
            Some(DocLocation {
                cluster: 0,
                local: 1
            })
        );
        assert_eq!(&*reads.lock().unwrap(), &[17..25]);
        Ok(())
    }

    // Rows and layer bands each pin their own columns with one request.
    #[test]
    fn rows_and_layer_spans_read_once_per_block() -> crate::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let layer = test_layer_tracked(
            FileSlice::from(Vec::new()),
            FileSlice::from(Vec::new()),
            None,
            &[0, 3],
            64,
            1,
            Some((Arc::clone(&reads), 32)),
        );
        reads.lock().unwrap().clear();
        let rows = layer.blocks.read_column(0, 0..3)?;
        assert_eq!(rows.len(), 3 * 64 * 4);
        let start = layer.blocks.block_start(0);
        assert_eq!(&*reads.lock().unwrap(), &[start..start + rows.len()]);
        reads.lock().unwrap().clear();
        layer.read_batch(0..3)?;
        assert_eq!(&*reads.lock().unwrap(), &[layer.blocks.layer_span(0, 0)]);
        Ok(())
    }

    #[test]
    fn doc_ids_reject_out_of_bounds_and_nonascending_values() -> crate::Result<()> {
        use super::super::blocks::{block_len, column_range, pad, write_metadata, BlockDirectory};
        for docs in [[0u32, 1, 3], [0, 0, 2], [1, 0, 2]] {
            let options = VectorOptions::new(2, Metric::L2);
            let meta = VectorColMetadata::build_ivf(&options, None)?;
            let slots = meta.slots();
            let mut data = Vec::new();
            let first = write_metadata(&mut data, &meta)?;
            for (idx, slot) in slots.iter().enumerate() {
                let range = column_range(&slots, docs.len(), idx);
                let padding = first + range.start - data.len();
                pad(&mut data, padding)?;
                if matches!(slot.slot_type, SlotType::DocIds) {
                    for doc in docs {
                        data.extend_from_slice(&doc.to_le_bytes());
                    }
                } else {
                    pad(&mut data, range.len())?;
                }
            }
            let padding = first + block_len(&slots, docs.len()) - data.len();
            pad(&mut data, padding)?;
            let mut directory = BlockDirectory::new(first as u64);
            directory.push(data.len() as u64, docs.len() as u32);
            directory.finish(&mut data)?;
            let blocks = Arc::new(Blocks::open(
                FileSlice::from(data),
                &options,
                3,
                Some(vec![0, 3]),
            )?);
            let reader = VectorIndexReader {
                max_doc: 3,
                options,
                num_vectors: 3,
                present: true,
                rows_slice: blocks,
                id_map: DeferredIdMap::ready(IdMap::Identity { num_docs: 3 }),
                index: None,
                quantization: None,
            };
            let mut decoded = Vec::new();
            let error = reader.read_doc_ids(0, &mut decoded).unwrap_err();
            assert!(
                matches!(error, TantivyError::DataCorruption(_)),
                "column=DocIds values={docs:?}: {error}"
            );
            assert!(matches!(
                reader.doc_id_at(0),
                Err(TantivyError::DataCorruption(_))
            ));
            assert!(matches!(
                reader.doc_id_at(3),
                Err(TantivyError::InvalidArgument(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn metadata_defers_search_bytes_and_shares_initialization() -> crate::Result<()> {
        let options = VectorOptions::new(3, Metric::L2);
        let mut data = Vec::new();
        let prefix = super::super::blocks::write_metadata(
            &mut data,
            &VectorColMetadata::build_flat(&options),
        )?;
        data.extend([0; 12]);
        let end = data.len();
        let mut directory = super::super::blocks::BlockDirectory::new(prefix as u64);
        directory.push(end as u64, 1);
        directory.finish(&mut data)?;
        let footer = data.len() - 8;
        let directory_start = footer - 2 * 12;
        let mut ids = Vec::new();
        IdMap::serialize(&[1], 3, &mut ids)?;
        let data_reads = Arc::new(Mutex::new(Vec::new()));
        let id_reads = Arc::new(Mutex::new(Vec::new()));
        let metadata = BlockMetadata::open(
            FileSlice::new(Arc::new(BlockTrackedBytes {
                bytes: data,
                reads: Arc::clone(&data_reads),
                block_len: 16,
            })),
            &options,
            false,
        )?;
        let field = Arc::new(VectorFieldReader {
            options,
            source: Some(VectorSource {
                metadata,
                composite: tracked_id_composite(ids, &id_reads),
                field: Field::from_field_id(0),
                max_doc: 3,
                centroid_slots: None,
            }),
            search: OnceLock::new(),
        });
        assert!(field.metadata().is_some());
        assert!(field.search.get().is_none());
        assert!(id_reads.lock().unwrap().is_empty());
        assert!(data_reads.lock().unwrap().iter().all(|r| r.end <= prefix));
        let readers = std::thread::scope(|scope| {
            (0..8)
                .map(|_| scope.spawn(|| field.search_reader().unwrap()))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(readers.iter().all(|r| Arc::ptr_eq(r, &readers[0])));
        let reads = data_reads.lock().unwrap();
        assert_eq!(
            reads.iter().filter(|r| **r == (footer..footer + 8)).count(),
            1
        );
        assert_eq!(
            reads
                .iter()
                .filter(|r| **r == (directory_start..footer))
                .count(),
            1
        );
        drop(reads);
        assert_eq!(
            id_reads
                .lock()
                .unwrap()
                .iter()
                .filter(|r| **r == (0..1))
                .count(),
            1
        );
        assert_eq!(readers[0].num_vectors(), 1);
        assert_eq!(readers[0].row_id(1)?, Some(0));
        Ok(())
    }

    #[test]
    fn lazy_search_rejects_corrupt_geometry_and_caches_error() -> crate::Result<()> {
        let options = VectorOptions::new(3, Metric::L2);
        let mut data = Vec::new();
        super::super::blocks::write_metadata(&mut data, &VectorColMetadata::build_flat(&options))?;
        let end = data.len();
        super::super::blocks::BlockDirectory::new(end as u64).finish(&mut data)?;
        let id_reads = Arc::new(Mutex::new(Vec::new()));
        let field = VectorFieldReader {
            source: Some(VectorSource {
                metadata: BlockMetadata::open(FileSlice::from(data), &options, false)?,
                composite: tracked_id_composite(vec![0], &id_reads),
                field: Field::from_field_id(0),
                max_doc: 1,
                centroid_slots: None,
            }),
            options,
            search: OnceLock::new(),
        };
        assert!(field.metadata().is_some());
        for _ in 0..2 {
            assert!(matches!(
                field.search_reader(),
                Err(TantivyError::DataCorruption(_))
            ));
        }
        assert_eq!(id_reads.lock().unwrap().len(), 1);
        Ok(())
    }

    #[test]
    fn planned_vector_rows_coalesce_reads_and_preserve_exact_scores() -> crate::Result<()> {
        const DIM: usize = 2;
        const ROWS: usize = 16;
        let row_bytes = (0..ROWS)
            .flat_map(|row| {
                let values = [row as f32 + 0.25, row as f32 * -0.5 + 1.0];
                values.into_iter().flat_map(f32::to_le_bytes)
            })
            .collect::<Vec<_>>();
        let expected_bytes = row_bytes.clone();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let opts = VectorOptions::new(DIM, Metric::Dot);
        let mut data = Vec::new();
        let prefix =
            super::super::blocks::write_metadata(&mut data, &VectorColMetadata::build_flat(&opts))?;
        data.extend_from_slice(&expected_bytes);
        let mut directory = super::super::blocks::BlockDirectory::new(prefix as u64);
        directory.push(data.len() as u64, ROWS as u32);
        directory.finish(&mut data)?;
        let storage = Arc::new(BlockTrackedBytes {
            bytes: data,
            reads: Arc::clone(&reads),
            block_len: 32,
        });
        let blocks = Arc::new(Blocks::open(FileSlice::new(storage), &opts, ROWS, None)?);
        reads.lock().unwrap().clear();
        let reader = VectorIndexReader {
            max_doc: ROWS as DocId,
            options: VectorOptions::new(DIM, Metric::Dot),
            num_vectors: ROWS,
            present: true,
            id_map: DeferredIdMap::ready(IdMap::Identity {
                num_docs: ROWS as u32,
            }),
            rows_slice: blocks,
            index: None,
            quantization: None,
        };
        let selected = [1, 2, 10, 11];
        let mut ranges = Vec::new();
        let mut block_scratch = Vec::new();
        let batch = reader.read_vector_rows_planned(&selected, &mut ranges, &mut block_scratch)?;

        assert_eq!(batch.read_count(), 2);
        assert_eq!(ranges, [1..3, 10..12]);
        assert_eq!(
            &*reads.lock().unwrap(),
            &[prefix + 8..prefix + 24, prefix + 80..prefix + 96]
        );
        assert!(batch.read_count() < selected.len());

        let query = PreparedQuery::<f32>::new(Metric::Dot, Arc::new(vec![2.0, -0.5]));
        let actual = batch
            .iter()
            .map(|(row, bytes)| (row, query.score_doc_bytes(bytes).to_bits()))
            .collect::<Vec<_>>();
        let stride = DIM * std::mem::size_of::<f32>();
        let expected = selected
            .iter()
            .map(|&row| {
                let bytes = &expected_bytes[row * stride..(row + 1) * stride];
                (row, query.score_doc_bytes(bytes).to_bits())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn diagnostic_query_norm_matches_production_f32_chain() {
        const DIM: usize = 64;
        let source_query: Vec<f32> = (0..DIM)
            .map(|coordinate| 10_000.0 + coordinate as f32 * 0.375)
            .collect();
        let centroid: Vec<f32> = (0..DIM)
            .map(|coordinate| 9_997.0 - coordinate as f32 * 0.1875)
            .collect();
        let centroid_bytes = centroid
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut differs_from_f64_reference = false;

        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let config = VectorQuantizationConfig::materialize(
                "embedding".to_string(),
                &VectorOptions::new(DIM, metric),
                vec![super::super::quantization::VectorQuantizationLayer { bits: 1, seed: 7 }],
            )
            .unwrap();
            let query = QuantizedQueryCtx::new(
                Arc::new(QuantizedIndexCtx::from_config(config).unwrap()),
                source_query.clone(),
            );
            let routing_score = metric
                .similarity_bytes::<f32>(query.query(), &centroid_bytes)
                .score();
            let expected = query.score_query_norm(routing_score);
            let actual = diagnostic_query_norm(&query, metric, &centroid_bytes);
            assert_eq!(actual.to_bits(), expected.to_bits(), "metric={metric:?}");

            let f64_reference = query
                .query()
                .iter()
                .zip(&centroid)
                .map(|(&query, &centroid)| {
                    let value = if metric == Metric::L2 {
                        query - centroid
                    } else {
                        query
                    };
                    f64::from(value) * f64::from(value)
                })
                .sum::<f64>()
                .sqrt();
            differs_from_f64_reference |= f64::from(actual).to_bits() != f64_reference.to_bits();
        }
        assert!(differs_from_f64_reference);
    }

    #[test]
    fn exact_e_sigma_matches_the_serving_formula() {
        let residual_norm_squared = 9.0;
        let corrected_error_ratio = 0.25;
        let gamma = 2.0;
        let query_norm_squared = 3.0;
        let sign_query_error_term = 0.5;
        let dimension = 100;
        let variance =
            residual_norm_squared / dimension as f32 * corrected_error_ratio * query_norm_squared
                + gamma * gamma * sign_query_error_term;
        let dot = quantized_model_sigma(
            Metric::Dot,
            dimension,
            residual_norm_squared,
            corrected_error_ratio,
            gamma,
            query_norm_squared,
            sign_query_error_term,
        );
        let l2 = quantized_model_sigma(
            Metric::L2,
            dimension,
            residual_norm_squared,
            corrected_error_ratio,
            gamma,
            query_norm_squared,
            sign_query_error_term,
        );
        let expected = super::super::quantization::GAMMA_ANALYTICAL_SAFETY * variance.sqrt();
        assert_eq!(dot.to_bits(), expected.to_bits());
        assert_eq!(l2.to_bits(), (2.0 * expected).to_bits());
    }

    #[test]
    fn exact_e_sigma_supports_a_grid_first_prefix() {
        let sigma = quantized_model_sigma(Metric::Cosine, 64, 4.0, 0.5, 1.25, 2.0, 0.0);
        let expected = super::super::quantization::GAMMA_ANALYTICAL_SAFETY
            * (4.0_f32 / 64.0 * 0.5 * 2.0).sqrt();
        assert_eq!(sigma.to_bits(), expected.to_bits());
    }

    #[test]
    fn estimator_merge_sums_rows_and_preserves_query_count() {
        let measurement = |sample_rows, query_count, value| VectorEstimatorMeasurements {
            source: VectorEstimatorSource::Provided,
            schedule: vec![("SignPlane", 1)],
            aggregate: vec![VectorEstimatorMoments {
                sample_count: 1,
                normalized_error_sum: value,
                normalized_error_squared_sum: value * value,
            }],
            per_query: vec![
                vec![VectorEstimatorMoments {
                    sample_count: 1,
                    normalized_error_sum: value,
                    normalized_error_squared_sum: value * value,
                }];
                query_count as usize
            ],
            sample_rows,
            query_count,
        };
        let mut aggregate = measurement(7, 2, 1.0);
        aggregate.merge(&measurement(5, 2, 3.0)).unwrap();
        assert_eq!(aggregate.sample_rows(), 12);
        assert_eq!(aggregate.query_count(), 2);
        assert_eq!(aggregate.aggregate()[0].sample_count, 2);
        assert_eq!(aggregate.aggregate()[0].bias(), Some(2.0));
        assert!(aggregate.merge(&measurement(1, 3, 1.0)).is_err());
        let mut held_out = measurement(1, 2, 1.0);
        held_out.source = VectorEstimatorSource::HeldOut;
        assert!(aggregate.merge(&held_out).is_err());
    }

    #[test]
    fn measurement_merges_reject_different_schedules() {
        let measurement = |bits: &[u8]| VectorEstimatorMeasurements {
            source: VectorEstimatorSource::Provided,
            schedule: bits
                .iter()
                .map(|&bits| (if bits == 1 { "SignPlane" } else { "GridPlane" }, bits))
                .collect(),
            aggregate: vec![VectorEstimatorMoments::default(); bits.len()],
            per_query: vec![vec![VectorEstimatorMoments::default(); bits.len()]],
            sample_rows: 0,
            query_count: 1,
        };
        for (left, right) in [(&[1][..], &[1, 4][..]), (&[1, 4], &[2, 4])] {
            let a = measurement(left);
            let b = measurement(right);
            let expected = format!(
                "different schedules: {:?} and {:?}",
                a.schedule(),
                b.schedule()
            );
            assert!(a
                .clone()
                .merge(&b)
                .unwrap_err()
                .to_string()
                .contains(&expected));
            let mut audit = VectorErrorAuditMeasurements {
                source: VectorEstimatorSource::Provided,
                depths: vec![VectorErrorDepthMeasurements::default(); left.len()],
                estimator: a,
            };
            assert_eq!(audit.schedule(), audit.estimator.schedule());
            let other = VectorErrorAuditMeasurements {
                source: VectorEstimatorSource::Provided,
                depths: vec![VectorErrorDepthMeasurements::default(); right.len()],
                estimator: b,
            };
            assert!(audit
                .merge(&other)
                .unwrap_err()
                .to_string()
                .contains(&expected));
        }
    }

    #[test]
    fn estimator_error_sign_is_estimate_minus_exact() {
        let mut measurements = VectorEstimatorMeasurements {
            source: VectorEstimatorSource::Provided,
            schedule: vec![("SignPlane", 1)],
            aggregate: vec![VectorEstimatorMoments::default()],
            per_query: vec![vec![VectorEstimatorMoments::default()]],
            sample_rows: 1,
            query_count: 1,
        };
        observe_corrected_prefix(&mut measurements, 0, 0, 1.0, 3.0, 2.0);
        assert_eq!(measurements.aggregate()[0].bias(), Some(1.0));
    }

    #[test]
    fn audit_moments_retain_exact_mergeable_quantiles() {
        let mut left = VectorAuditMoments::default();
        for value in [-4.0, 1.0, 2.0] {
            left.observe(value);
        }
        let mut right = VectorAuditMoments::default();
        for value in [3.0, 5.0] {
            right.observe(value);
        }
        left.merge(&right);
        assert_eq!(left.p50(), Some(2.0));
        assert_eq!(left.p95(), Some(5.0));
        assert_eq!(left.p99(), Some(5.0));
        assert_eq!(left.p99_abs(), Some(5.0));
        assert_eq!(left.max_abs(), Some(5.0));
    }

    #[test]
    fn error_cone_boundary_includes_optimistic_equality() {
        let mut candidates = ErrorConeCandidates::default();
        candidates.push(0, 1, 0.0, 0.0, 10.0, 0.0);
        candidates.push(1, 2, 0.0, 0.0, 8.0, 1.0);
        candidates.push(2, 3, 0.0, 0.0, 4.0, 1.0);
        candidates.band(2, 2.0);
        assert_eq!(candidates.rows, vec![0, 1, 2]);
    }

    #[test]
    fn quantized_layer_reader_rejects_non_zero_tail() -> crate::Result<()> {
        let mut codes = vec![0_u8; quantized_code_stride(65, 1)];
        codes[8] = 1;
        let valid = test_layer(
            FileSlice::from(codes.clone()),
            FileSlice::empty(),
            None,
            &[0, 1],
            65,
            1,
        );
        assert_eq!(valid.code_bytes(0)?.as_slice(), codes);

        codes[8] |= 2;
        let corrupt = test_layer(
            FileSlice::from(codes.clone()),
            FileSlice::empty(),
            None,
            &[0, 1],
            65,
            1,
        );
        assert!(corrupt.code_bytes(0).is_err());
        Ok(())
    }

    /// Without storage geometry, the plan reads only selected runs, so an unselected
    /// row is neither read nor validated; corruption there is caught by whichever
    /// query selects it.
    #[test]
    fn no_geometry_plan_skips_unselected_gap_rows_and_validates_read_rows() {
        let stride = quantized_code_stride(65, 1);
        let mut codes = vec![0_u8; stride * 3];
        codes[8] = 1;
        codes[stride + 8] = 0b10;
        codes[stride * 2 + 8] = 1;
        let reader = test_layer(
            FileSlice::from(codes),
            FileSlice::from(test_sidecar(&[0.0; 3], &[1.0; 3])),
            None,
            &[0, 3],
            65,
            1,
        );
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();
        reader.plan_code_reads(0..3, &[0, 2], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..1, 2..3]);
        assert!(reader.read_codes(0..1).is_ok());
        assert!(reader.read_codes(2..3).is_ok());
        assert!(reader.read_codes(0..3).is_err());
    }

    #[test]
    fn quantized_layer_reader_pins_and_decodes_a_contiguous_batch() -> crate::Result<()> {
        let stride = quantized_code_stride(65, 1);
        let mut codes = vec![0_u8; stride * 2];
        codes[8] = 1;
        codes[stride + 8] = 1;
        let sidecar = test_sidecar(&[17.0, 29.0], &[1.0, 4.0]);
        let constants = [0.25_f32, -0.75_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let reader = test_layer(
            FileSlice::from(codes.clone()),
            FileSlice::from(sidecar),
            Some(FileSlice::from(constants)),
            &[0, 2],
            65,
            1,
        );
        let batch = reader.read_batch(0..2)?;
        assert_eq!(batch.code_bytes(0)?, &codes[..stride]);
        assert_eq!(batch.code_bytes(1)?, &codes[stride..]);
        assert_eq!(batch.scale(0)?, 17.0);
        assert_eq!(batch.scale(1)?, 29.0);
        assert_eq!(batch.gamma(0)?, 1.0);
        assert_eq!(batch.gamma(1)?, 4.0);
        assert_eq!(batch.error_ratio(0)?, 0.25);
        assert_eq!(batch.error_ratio(1)?, 0.25);
        assert_eq!(batch.scales().len(), 2 * QUANTIZED_SCALE_STRIDE);
        assert_eq!(batch.gammas().len(), 2 * QUANTIZED_GAMMA_STRIDE);
        assert_eq!(batch.error_ratios().len(), 2 * QUANTIZED_ERROR_RATIO_STRIDE);
        assert_eq!(batch.constant(0)?.unwrap().to_bits(), 0.25_f32.to_bits());
        assert_eq!(batch.constant(1)?.unwrap().to_bits(), (-0.75_f32).to_bits());

        codes[stride + 8] |= 2;
        let corrupt = test_layer(
            FileSlice::from(codes),
            FileSlice::from(test_sidecar(&[0.0, 0.0], &[1.0, 0.5])),
            Some(FileSlice::from(vec![0_u8; 8])),
            &[0, 2],
            65,
            1,
        );
        assert!(corrupt.read_batch(0..2).is_err());
        Ok(())
    }

    #[test]
    fn cluster_blocked_sidecar_maps_scales_and_gammas_by_cluster() -> crate::Result<()> {
        let mut sidecar = test_sidecar(&[11.0, 12.0], &[1.0, 2.0]);
        sidecar.extend(test_sidecar(&[13.0], &[3.0]));
        let reader = test_layer(
            FileSlice::empty(),
            FileSlice::from(sidecar),
            None,
            &[0, 2, 3],
            64,
            1,
        );

        let first = reader.read_sidecar(0..2)?;
        assert_eq!(first.scale(0)?, 11.0);
        assert_eq!(first.scale(1)?, 12.0);
        assert_eq!(first.gamma(0)?, 1.0);
        assert_eq!(first.gamma(1)?, 2.0);
        assert_eq!(first.error_ratio(0)?, 0.25);
        assert_eq!(first.error_ratio(1)?, 0.25);
        let second = reader.read_sidecar(2..3)?;
        assert_eq!(second.scale(2)?, 13.0);
        assert_eq!(second.gamma(2)?, 3.0);
        assert_eq!(second.error_ratio(2)?, 0.25);
        assert_eq!(reader.scale(2)?, 13.0);
        assert_eq!(reader.gamma(2)?, 3.0);
        Ok(())
    }

    #[test]
    fn sidecar_gamma_validation_is_range_scoped() {
        for gamma in [0.5, 5.0, f32::INFINITY, f32::NAN] {
            assert!(
                validate_decoded_sidecar(&[gamma], &[0.0], 0).is_err(),
                "gamma={gamma}"
            );
        }
        assert!(validate_decoded_sidecar(&[1.0, 4.0], &[0.0, 0.0], 0).is_ok());
    }

    #[test]
    fn sidecar_error_ratio_validation_is_range_scoped() {
        for error_ratio in [-0.5, f32::INFINITY, f32::NAN] {
            assert!(
                validate_decoded_sidecar(&[1.0], &[error_ratio], 0).is_err(),
                "error_ratio={error_ratio}"
            );
        }
        assert!(validate_decoded_sidecar(&[1.0, 1.0], &[0.0, 0.5], 0).is_ok());
    }

    // A whole band requires one read; sparse columns plan independently without crossing blocks.
    #[test]
    fn bands_and_sparse_columns_preserve_storage_geometry() -> crate::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let reader = test_layer_tracked(
            FileSlice::from(vec![0; 12 * 8]),
            FileSlice::from(
                [
                    test_sidecar(&[0.0; 8], &[1.0; 8]),
                    test_sidecar(&[0.0; 4], &[1.0; 4]),
                ]
                .concat(),
            ),
            None,
            &[0, 8, 8, 12],
            64,
            1,
            Some((Arc::clone(&reads), 16)),
        );
        reads.lock().unwrap().clear();
        let before = super::super::storage_io::snapshot();
        let stage = super::super::enter_vector_stage(super::super::Stage::LayerScan(0));
        let batch = reader.read_batch(0..8)?;
        drop(stage);
        assert_eq!(reads.lock().unwrap().len(), 1);
        let io = super::super::storage_io::snapshot()[0].since(before[0]);
        let range = reads.lock().unwrap()[0].clone();
        assert_eq!(io.reads, 1);
        assert_eq!(io.bytes_read, range.len() as u64);
        assert_eq!(
            io.storage_blocks,
            ((range.end - 1) / 16 - range.start / 16 + 1) as u64
        );
        assert!(batch.residual_norms.is_some());
        let sidecar = reader.read_sidecar(0..8)?;
        assert_eq!(batch.scales(), sidecar.scales());
        assert_eq!(batch.gammas(), sidecar.gammas());
        assert_eq!(batch.error_ratios(), sidecar.error_ratios());
        let mut ranges = Vec::new();
        let mut scratch = Vec::new();
        for column in [reader.codes, reader.scales, reader.gammas, reader.errors] {
            reader.plan_column_reads(column, 0..12, &[0, 7, 8, 11], &mut ranges, &mut scratch);
            assert!(ranges.iter().all(|r| r.end <= 8 || r.start >= 8));
            for r in ranges.clone() {
                reader.read_column(column, r)?;
            }
        }
        for column in [reader.codes, reader.scales, reader.gammas, reader.errors] {
            assert!(reader.read_column(column, 7..9).is_err());
        }
        Ok(())
    }

    #[test]
    fn code_page_groups_require_overlap_and_trim_selected_rows() {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let slot = FileSlice::new(Arc::new(BlockTrackedBytes {
            bytes: vec![0; 160],
            reads,
            block_len: 16,
        }));
        let mut ranges = Vec::new();
        // Adjacent disjoint pages 0 and 1 stay separate, as does page 4.
        QuantizedLayerReader::plan_code_slot_reads(&slot, 4, 100, &[101, 106, 118], &mut ranges);
        assert_eq!(ranges, [101..102, 106..107, 118..119]);
        ranges.clear();
        // Row 2 straddles pages 0/1, connecting row 0 to row 4; page 2 stays separate.
        QuantizedLayerReader::plan_code_slot_reads(&slot, 6, 0, &[0, 2, 4, 6], &mut ranges);
        assert_eq!(ranges, [0..5, 6..7]);
        ranges.clear();
        let unknown = FileSlice::from(vec![0; 160]);
        QuantizedLayerReader::plan_code_slot_reads(&unknown, 4, 0, &[1, 2, 6], &mut ranges);
        assert_eq!(ranges, [1..3, 6..7]);
    }

    #[test]
    fn sparse_cluster_reads_pin_one_sidecar_span() -> crate::Result<()> {
        use super::super::blocks::{block_len, column_range, write_metadata, BlockDirectory};
        use super::super::{enter_vector_stage, storage_io, Stage, VectorQuantizationLayer};
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let opts = VectorOptions::new(1024, metric);
            let config = VectorQuantizationConfig::materialize(
                "v".into(),
                &opts,
                vec![
                    VectorQuantizationLayer { bits: 1, seed: 7 },
                    VectorQuantizationLayer { bits: 4, seed: 11 },
                ],
            )?;
            let meta = VectorColMetadata::build_ivf(&opts, Some(&config))?;
            let slots = meta.slots();
            let mut data = Vec::new();
            write_metadata(&mut data, &meta)?;
            let mut directory = BlockDirectory::new(data.len() as u64);
            let boundaries = [0, 59, 59, 118];
            for rows in boundaries.windows(2) {
                let n = rows[1] - rows[0];
                let start = data.len();
                data.resize(start + block_len(&slots, n), 0);
                for (idx, slot) in slots.iter().enumerate() {
                    let range = column_range(&slots, n, idx);
                    match slot.slot_type {
                        SlotType::QuantLayerScales { .. }
                        | SlotType::QuantLayerConstants { .. } => {
                            for (row, bytes) in data[start + range.start..start + range.end]
                                .chunks_exact_mut(4)
                                .enumerate()
                            {
                                bytes.copy_from_slice(
                                    &((rows[0] + row + idx) as f32 / 16.0).to_le_bytes(),
                                );
                            }
                        }
                        SlotType::QuantLayerGammas { .. } | SlotType::QuantLayerErrors { .. } => {
                            for (row, bytes) in data[start + range.start..start + range.end]
                                .chunks_exact_mut(2)
                                .enumerate()
                            {
                                bytes.copy_from_slice(
                                    &f32_to_f16((rows[0] + row + idx) as f32 / 256.0).to_le_bytes(),
                                );
                            }
                        }
                        _ => {}
                    }
                }
                directory.push(data.len() as u64, rows[1] as u32);
            }
            directory.finish(&mut data)?;
            for block_len in [0, 16, 8160] {
                for prefix in [0, 3, 8171] {
                    let reads = Arc::new(Mutex::new(Vec::new()));
                    let mut parent = vec![0; prefix];
                    parent.extend_from_slice(&data);
                    let entry = if block_len == 0 {
                        FileSlice::from(parent).slice_from(prefix)
                    } else {
                        FileSlice::new(Arc::new(BlockTrackedBytes {
                            bytes: parent,
                            reads,
                            block_len,
                        }))
                        .slice_from(prefix)
                    };
                    let blocks =
                        Arc::new(Blocks::open(entry, &opts, 118, Some(boundaries.to_vec()))?);
                    let codes = slots
                        .iter()
                        .position(|s| {
                            matches!(s.slot_type, SlotType::QuantLayerCodes { layer: 1, .. })
                        })
                        .unwrap();
                    let reader = QuantizedLayerReader {
                        blocks,
                        layer: 1,
                        codes,
                        scales: codes + 1,
                        gammas: codes + 2,
                        errors: codes + 3,
                        constants: (metric == Metric::L2).then_some(codes + 4),
                        code_stride: quantized_code_stride(1024, 4),
                        dim: 1024,
                        bits: 4,
                    };
                    for selected in [vec![0, 58], vec![0, 58, 59, 117], vec![5, 6, 31, 54]] {
                        let mut ranges = Vec::new();
                        let mut start = 0;
                        while start < selected.len() {
                            let cluster = reader.cluster(selected[start])?;
                            let end = start
                                + selected[start..].partition_point(|&r| r < cluster.rows.end);
                            cluster.plan_codes(&selected[start..end], &mut ranges);
                            let mut expected = Vec::new();
                            reader.plan_column_reads(
                                reader.codes,
                                0..118,
                                &selected[start..end],
                                &mut expected,
                                &mut Vec::new(),
                            );
                            assert_eq!(ranges, expected);
                            let stage = enter_vector_stage(Stage::LayerScan(1));
                            let before = storage_io::snapshot()[1];
                            for rows in ranges.iter().cloned() {
                                cluster.read_codes(rows)?;
                            }
                            let code_io = storage_io::snapshot()[1].since(before);
                            let before = storage_io::snapshot()[1];
                            let sidecar = cluster.read_sidecar()?;
                            let sidecar_io = storage_io::snapshot()[1].since(before);
                            drop(stage);
                            assert_eq!(sidecar_io.reads, 1);
                            let first =
                                column_range(&slots, cluster.rows.len(), reader.scales).start;
                            let last = column_range(
                                &slots,
                                cluster.rows.len(),
                                reader.constants.unwrap_or(reader.errors),
                            )
                            .end;
                            assert_eq!(sidecar_io.bytes_read, (last - first) as u64);
                            let row_range = cluster.rows.clone();
                            assert_eq!(
                                sidecar.scales,
                                reader.read_column(reader.scales, row_range.clone())?
                            );
                            assert_eq!(
                                sidecar.gammas,
                                reader.read_column(reader.gammas, row_range.clone())?
                            );
                            assert_eq!(
                                sidecar.error_ratios,
                                reader.read_column(reader.errors, row_range.clone())?
                            );
                            assert_eq!(sidecar.constants, reader.read_constants(row_range)?);
                            let n = end - start;
                            let (mut scales, mut gammas, mut errors) =
                                (vec![0.; n], vec![0.; n], vec![0.; n]);
                            let mut constants = vec![0.; if metric == Metric::L2 { n } else { 0 }];
                            sidecar.decode_selected(
                                &selected[start..end],
                                &mut scales,
                                &mut gammas,
                                &mut errors,
                                &mut constants,
                            )?;
                            for (i, &row) in selected[start..end].iter().enumerate() {
                                assert_eq!(scales[i].to_bits(), reader.scale(row)?.to_bits());
                                assert_eq!(gammas[i].to_bits(), reader.gamma(row)?.to_bits());
                                assert_eq!(errors[i].to_bits(), reader.error_ratio(row)?.to_bits());
                                if metric == Metric::L2 {
                                    assert_eq!(
                                        constants[i].to_bits(),
                                        reader.constant(row)?.unwrap().to_bits()
                                    );
                                }
                            }
                            if block_len == 8160 && prefix == 0 && selected == [0, 58] {
                                assert_eq!(code_io.reads + sidecar_io.reads, 3);
                                assert_eq!(code_io.storage_blocks + sidecar_io.storage_blocks, 3);
                            }
                            start = end;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // Missing entries and unknown entry indices are corruption regardless of field order.
    #[test]
    fn vector_entries_require_exact_pair() -> crate::Result<()> {
        let field = Field::from_field_id(0);
        for entries in [vec![0], vec![1], vec![0, 1, 2], vec![1, 0]] {
            let mut bytes = Vec::new();
            let mut writer = CompositeWrite::wrap(&mut bytes);
            for &idx in &entries {
                writer.for_field_with_idx(field, idx).write_all(&[0])?;
            }
            writer.close()?;
            let composite = CompositeFile::open(&FileSlice::from(bytes))?;
            assert_eq!(
                validate_vector_entries(&composite, field).is_ok(),
                entries == [1, 0]
            );
        }
        Ok(())
    }
}
