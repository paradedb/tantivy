//! Per-(segment, field) vector reader, modeled on
//! [`InvertedIndexReader`](crate::index::InvertedIndexReader).
//!
//! One [`VectorIndexReader`] serves one vector field of one segment, opened
//! (and cached) via
//! [`SegmentReader::vector_index`](crate::SegmentReader::vector_index). Small
//! routing state is parsed once and pinned in memory, while the bulk payload
//! stays behind [`FileSlice`]s and is fetched with ranged reads at query time.
//!
//! The reader is "store + optional index":
//! - The **store** is the segment's `.vec` composite: slot `[0]` is the row→doc_id [`IdMap`], slot
//!   `[1]` the dense vector rows (deferred).
//! - The **index** ([`IvfIndex`]) exists if the segment was merged with IVF clustering, and is
//!   loaded from the sibling `.centroids` composite. It tells a query which clusters — contiguous
//!   row ranges of slot `[1]` — to probe. Without it, search falls back to an exact scan.
//!
//! The pairing is an invariant of the write path (`VectorPlugin`): the IVF
//! merge writes cluster-sorted rows + an `Explicit` id-map + `.centroids`;
//! every other writer produces doc-ordered rows + `Identity`/`Bitmap` and no
//! sidecar. [`VectorIndexReader::open`] validates the two signals agree.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use cascade::audit_prefix_gammas;
use common::{HasLen, OwnedBytes};
use quant_model::f16::f16_to_f32;

use super::flat::IdMap;
use super::header::{centroid_slot, read_header, vec_slot, VectorFileVersion};
use super::ivf::{decode_row, IvfIndex, CENTROIDS_EXT};
use super::prepared::{PreparedQuery, QuantizedIndexCtx, QuantizedQueryCtx};
use super::quantization::{
    quantized_code_stride, quantized_code_tail_is_zero, VectorQuantizationCalibrationSource,
    VectorQuantizationConfig, VectorQuantizationDepthCalibration, GAMMA_ANALYTICAL_SAFETY,
    QUANTIZED_CONSTANT_STRIDE, QUANTIZED_GAMMA_STRIDE, QUANTIZED_RESIDUAL_NORM_STRIDE,
    QUANTIZED_SCALE_GAMMA_STRIDE, QUANTIZED_SCALE_STRIDE,
};
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorStorageFormat {
    Flat,
    Ivf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorInfo {
    pub format: VectorStorageFormat,
    /// Distinct documents with a vector in this field. The per-cluster
    /// numbers (`cluster_stats`, [`VectorIndexReader::cluster_sizes`]) count
    /// posting rows, so with legacy V2 replication their sum could exceed
    /// `num_vectors`.
    pub num_vectors: usize,
    pub num_centroids: Option<usize>,
    pub cluster_stats: Option<VectorClusterStats>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorClusterStats {
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub avg_cluster_size: f64,
    pub empty_clusters: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VectorCalibrationMoments {
    pub sample_count: u64,
    pub normalized_error_sum: f64,
    pub normalized_error_squared_sum: f64,
}

impl VectorCalibrationMoments {
    fn observe(&mut self, value: f64) {
        self.sample_count += 1;
        self.normalized_error_sum += value;
        self.normalized_error_squared_sum += value * value;
    }

    pub fn bias(&self) -> Option<f64> {
        (self.sample_count != 0).then(|| self.normalized_error_sum / self.sample_count as f64)
    }

    pub fn spread(&self) -> Option<f64> {
        self.bias().map(|bias| {
            (self.normalized_error_squared_sum / self.sample_count as f64 - bias * bias)
                .max(0.0)
                .sqrt()
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorCalibrationMeasurements {
    aggregate: Vec<VectorCalibrationMoments>,
    per_query: Vec<Vec<VectorCalibrationMoments>>,
}

/// One production-query input to the non-persisting gamma audit.
///
/// External queries carry no exclusion. A stored pseudo-query carries its
/// source `doc_id` only while auditing the segment that owns it; the audit then
/// excludes every cluster containing any replica membership of that document.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorGammaAuditQuery {
    pub values: Vec<f32>,
    pub excluded_doc_id: Option<DocId>,
}

/// Mergeable scalar moments used by gamma diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorAuditMoments {
    pub sample_count: u64,
    pub sum: f64,
    pub squared_sum: f64,
    pub min: f64,
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

    pub fn mean(&self) -> Option<f64> {
        (self.sample_count != 0).then(|| self.sum / self.sample_count as f64)
    }

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

    pub fn p50(&self) -> Option<f64> {
        self.quantile(0.50)
    }

    pub fn p95(&self) -> Option<f64> {
        self.quantile(0.95)
    }

    pub fn p99(&self) -> Option<f64> {
        self.quantile(0.99)
    }

    pub fn p99_abs(&self) -> Option<f64> {
        self.quantile_abs(0.99)
    }

    pub fn max_abs(&self) -> Option<f64> {
        (!self.samples.is_empty()).then(|| {
            self.samples
                .iter()
                .fold(0.0_f64, |max, value| max.max(value.abs()))
        })
    }
}

/// Per-prefix gamma and model diagnostics. Gamma distributions are observed
/// once per sampled posting row; band-error distributions are observed once
/// per eligible query-row pair.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorGammaDepthMeasurements {
    pub gamma_raw: VectorAuditMoments,
    pub gamma_clamped: VectorAuditMoments,
    pub gamma_f16: VectorAuditMoments,
    /// Stored gamma minus its pre-f16 clamped value, once per sampled row.
    pub gamma_f16_roundtrip_error: VectorAuditMoments,
    /// `prefix_dot / d` from the pre-f16 audit reconstruction.
    pub raw_effective_scale_squared: VectorAuditMoments,
    /// Production-side effective scale derived only from decoded f16
    /// sidecars: `S1^2 * gamma1 / gamma_l`.
    pub stored_effective_scale_squared: VectorAuditMoments,
    /// The score-space effect of gamma f16 rounding, divided by model sigma.
    pub f16_band_error: VectorAuditMoments,
    /// The score-space effect of the `[1, 4]` gamma clamp, divided by model
    /// sigma. This is distinct from f16 rounding.
    pub clamp_band_error: VectorAuditMoments,
    /// `(||r_hat||^2 - <r,r_hat>) / <r,r_hat>` for the raw cumulative
    /// reconstruction. Zero is the projection-orthogonality identity.
    pub orthogonality_defect: VectorAuditMoments,
    pub zero_count: u64,
    pub clamp_count: u64,
}

impl VectorGammaDepthMeasurements {
    fn merge(&mut self, other: &Self) {
        self.gamma_raw.merge(&other.gamma_raw);
        self.gamma_clamped.merge(&other.gamma_clamped);
        self.gamma_f16.merge(&other.gamma_f16);
        self.gamma_f16_roundtrip_error
            .merge(&other.gamma_f16_roundtrip_error);
        self.raw_effective_scale_squared
            .merge(&other.raw_effective_scale_squared);
        self.stored_effective_scale_squared
            .merge(&other.stored_effective_scale_squared);
        self.f16_band_error.merge(&other.f16_band_error);
        self.clamp_band_error.merge(&other.clamp_band_error);
        self.orthogonality_defect.merge(&other.orthogonality_defect);
        self.zero_count += other.zero_count;
        self.clamp_count += other.clamp_count;
    }
}

/// Source-tagged, mergeable output of a read-only gamma audit.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorGammaAuditMeasurements {
    pub source: VectorQuantizationCalibrationSource,
    pub calibration: VectorCalibrationMeasurements,
    pub depths: Vec<VectorGammaDepthMeasurements>,
}

/// One boundary's read-only confidence-cone measurements. Counts and ratios
/// are retained per query so their means and exact quantiles remain available
/// without changing the production statistics channel.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorGammaConeDepthMeasurements {
    pub kappa: f32,
    pub scored_rows: VectorAuditMoments,
    pub survivor_rows: VectorAuditMoments,
    pub survivor_docs: VectorAuditMoments,
    pub survivor_fraction: VectorAuditMoments,
    pub candidate_recall: VectorAuditMoments,
    pub queries_with_miss: u32,
}

impl VectorGammaConeDepthMeasurements {
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

/// Segment-wide, all-clusters-admitted band audit for exactly 100 external
/// queries over a `[1,4]` schedule. No measured bias or spread participates:
/// this measures the gamma-corrected analytical mechanism itself.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorGammaConeAuditMeasurements {
    pub query_count: u32,
    pub top_k: u32,
    pub depths: Vec<VectorGammaConeDepthMeasurements>,
}

const GAMMA_CONE_QUERY_COUNT: usize = 100;
const GAMMA_CONE_TOP_K: usize = 10;
const GAMMA_CONE_KAPPAS: [f32; 2] = [2.0, 4.0];

impl VectorGammaAuditMeasurements {
    pub fn merge(&mut self, other: &Self) -> crate::Result<()> {
        if self.source != other.source || self.depths.len() != other.depths.len() {
            return Err(TantivyError::InvalidArgument(
                "cannot merge gamma audit measurements with different sources or depths"
                    .to_string(),
            ));
        }
        self.calibration.merge(&other.calibration)?;
        for (left, right) in self.depths.iter_mut().zip(&other.depths) {
            left.merge(right);
        }
        Ok(())
    }
}

#[inline]
fn calibration_query_norm(query: &QuantizedQueryCtx, metric: Metric, centroid_bytes: &[u8]) -> f32 {
    let routing_score = metric
        .similarity_bytes::<f32>(query.query(), centroid_bytes)
        .score();
    query.score_query_norm(routing_score)
}

#[inline]
fn observe_gamma_corrected_prefix(
    measurements: &mut VectorCalibrationMeasurements,
    query_idx: usize,
    depth: usize,
    exact_dot: f32,
    corrected_prefix_estimate: f32,
    model_sigma: f64,
) {
    if model_sigma > 0.0 && model_sigma.is_finite() {
        let error = (f64::from(exact_dot) - f64::from(corrected_prefix_estimate)) / model_sigma;
        if error.is_finite() {
            measurements.aggregate[depth].observe(error);
            measurements.per_query[query_idx][depth].observe(error);
        }
    }
}

#[inline]
fn gamma_model_variance(
    effective_scale_squared: f32,
    gamma: f32,
    score_query_norm_squared: f32,
    sign_query_error_term: f32,
) -> f32 {
    let data = effective_scale_squared * gamma * (gamma - 1.0) * score_query_norm_squared;
    let query = gamma * gamma * sign_query_error_term;
    (data + query).max(0.0)
}

#[inline]
fn gamma_production_sigma(
    effective_scale_squared: f32,
    gamma: f32,
    score_query_norm_squared: f32,
    sign_query_error_term: f32,
    metric: Metric,
) -> f32 {
    let metric_factor = if metric == Metric::L2 { 2.0 } else { 1.0 };
    metric_factor
        * GAMMA_ANALYTICAL_SAFETY
        * gamma_model_variance(
            effective_scale_squared,
            gamma,
            score_query_norm_squared,
            sign_query_error_term,
        )
        .sqrt()
}

#[inline]
fn gamma_effective_scale_squared(scale_one_squared: f32, gamma_one: f32, gamma: f32) -> f32 {
    scale_one_squared * gamma_one / gamma
}

fn verify_stored_gamma(
    row: usize,
    depth: usize,
    stored: f32,
    deterministic: f32,
) -> crate::Result<()> {
    if stored.to_bits() != deterministic.to_bits() {
        return Err(DataCorruption::comment_only(format!(
            "quantized row {row} layer {depth} stored gamma differs from the deterministic \
             encoder: stored={stored}, expected={deterministic}"
        ))
        .into());
    }
    Ok(())
}

fn excluded_membership_clusters<I>(
    queries: &[VectorGammaAuditQuery],
    memberships: I,
) -> Vec<Vec<usize>>
where
    I: IntoIterator<Item = (usize, DocId)>,
{
    let mut query_indices_by_doc: HashMap<DocId, Vec<usize>> = HashMap::new();
    for (query_idx, query) in queries.iter().enumerate() {
        if let Some(doc_id) = query.excluded_doc_id {
            query_indices_by_doc
                .entry(doc_id)
                .or_default()
                .push(query_idx);
        }
    }

    let mut excluded = vec![Vec::new(); queries.len()];
    for (cluster, doc_id) in memberships {
        let Some(query_indices) = query_indices_by_doc.get(&doc_id) else {
            continue;
        };
        for &query_idx in query_indices {
            if excluded[query_idx].last().copied() != Some(cluster) {
                excluded[query_idx].push(cluster);
            }
        }
    }
    excluded
}

/// Diagnostic-only SoA boundary state. Production candidate structs are not
/// materialized or reused by the read-only cone audit.
#[derive(Default)]
struct GammaConeCandidates {
    rows: Vec<usize>,
    docs: Vec<DocId>,
    raw_prefixes: Vec<f32>,
    estimates: Vec<f32>,
    sigmas: Vec<f32>,
}

impl GammaConeCandidates {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn push(&mut self, row: usize, doc: DocId, raw_prefix: f32, estimate: f32, sigma: f32) {
        self.rows.push(row);
        self.docs.push(doc);
        self.raw_prefixes.push(raw_prefix);
        self.estimates.push(estimate);
        self.sigmas.push(sigma);
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

    /// Apply the production boundary rule exactly: choose the k-th distinct
    /// document by estimate-descending/row-ascending order, widen only that
    /// pivot pessimistically, and retain every membership whose optimistic
    /// endpoint reaches the threshold. Memberships are sorted back into
    /// storage order for the next layer.
    fn band(&mut self, top_k: usize, kappa: f32) {
        let mut best_by_doc: HashMap<DocId, usize> = HashMap::new();
        for index in 0..self.len() {
            let doc = self.docs[index];
            if best_by_doc
                .get(&doc)
                .is_none_or(|&previous| gamma_cone_candidate_order(self, index, previous).is_lt())
            {
                best_by_doc.insert(doc, index);
            }
        }
        let threshold = if top_k == 0 || best_by_doc.len() < top_k {
            None
        } else {
            let mut best: Vec<usize> = best_by_doc.into_values().collect();
            let (_, pivot, _) = best.select_nth_unstable_by(top_k - 1, |&left, &right| {
                gamma_cone_candidate_order(self, left, right)
            });
            let pivot = *pivot;
            Some(self.estimates[pivot] - kappa * self.sigmas[pivot])
        };

        let mut survivors: Vec<usize> = (0..self.len())
            .filter(|&index| {
                threshold.is_none_or(|threshold| {
                    self.estimates[index] + kappa * self.sigmas[index] >= threshold
                })
            })
            .collect();
        survivors.sort_unstable_by_key(|&index| self.rows[index]);

        let mut compacted = Self::default();
        compacted.rows.reserve(survivors.len());
        compacted.docs.reserve(survivors.len());
        compacted.raw_prefixes.reserve(survivors.len());
        compacted.estimates.reserve(survivors.len());
        compacted.sigmas.reserve(survivors.len());
        for index in survivors {
            compacted.push(
                self.rows[index],
                self.docs[index],
                self.raw_prefixes[index],
                self.estimates[index],
                self.sigmas[index],
            );
        }
        *self = compacted;
    }
}

fn gamma_cone_candidate_order(
    candidates: &GammaConeCandidates,
    left: usize,
    right: usize,
) -> Ordering {
    candidates.estimates[right]
        .total_cmp(&candidates.estimates[left])
        .then(candidates.rows[left].cmp(&candidates.rows[right]))
}

fn observe_gamma_cone_depth(
    measurements: &mut VectorGammaConeDepthMeasurements,
    scored: usize,
    candidates: &GammaConeCandidates,
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

impl VectorCalibrationMeasurements {
    pub fn aggregate(&self) -> &[VectorCalibrationMoments] {
        &self.aggregate
    }

    pub fn per_query(&self) -> &[Vec<VectorCalibrationMoments>] {
        &self.per_query
    }

    pub fn merge(&mut self, other: &Self) -> crate::Result<()> {
        if self.aggregate.is_empty() {
            *self = other.clone();
            return Ok(());
        }
        if self.aggregate.len() != other.aggregate.len()
            || self.per_query.len() != other.per_query.len()
        {
            return Err(TantivyError::InvalidArgument(
                "cannot merge quantization calibration measurements with different shapes"
                    .to_string(),
            ));
        }
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

    pub fn finish(
        &self,
        source: VectorQuantizationCalibrationSource,
    ) -> crate::Result<Vec<VectorQuantizationDepthCalibration>> {
        self.aggregate
            .iter()
            .enumerate()
            .map(|(depth, moments)| {
                let Some(bias) = moments.bias() else {
                    return Err(TantivyError::InvalidArgument(format!(
                        "quantization calibration depth {depth} has no measurements"
                    )));
                };
                let spread = moments.spread().unwrap();
                if !bias.is_finite()
                    || !spread.is_finite()
                    || bias.abs() > f32::MAX as f64
                    || spread > f32::MAX as f64
                {
                    return Err(TantivyError::InvalidArgument(format!(
                        "quantization calibration depth {depth} produced non-finite f32 statistics"
                    )));
                }
                Ok(VectorQuantizationDepthCalibration {
                    bias: bias as f32,
                    spread: spread as f32,
                    sample_count: u32::try_from(moments.sample_count).map_err(|_| {
                        TantivyError::InvalidArgument(format!(
                            "quantization calibration depth {depth} sample count exceeds u32"
                        ))
                    })?,
                    source,
                })
            })
            .collect()
    }
}

/// Deferred slices for one residual plane. Codes/constants remain fixed-row
/// stride; the scale/gamma sidecar is blocked by IVF cluster ranges.
pub(crate) struct QuantizedLayerReader {
    codes: FileSlice,
    sidecar: FileSlice,
    constants: FileSlice,
    cluster_offsets: Arc<[usize]>,
    code_stride: usize,
    dim: usize,
    bits: u8,
}

/// Four pinned SoA ranges for one contiguous cluster posting.
pub(crate) struct QuantizedLayerBatch {
    codes: OwnedBytes,
    scales: OwnedBytes,
    gammas: OwnedBytes,
    constants: OwnedBytes,
    rows: Range<usize>,
    code_stride: usize,
}

/// Borrowed scale and cumulative-prefix-gamma runs for one row range inside
/// one cluster sidecar block.
pub(crate) struct QuantizedScaleGammaBatch {
    scales: OwnedBytes,
    gammas: OwnedBytes,
    rows: Range<usize>,
}

impl QuantizedScaleGammaBatch {
    fn local_row(&self, row: usize) -> crate::Result<usize> {
        if !self.rows.contains(&row) {
            return Err(TantivyError::InternalError(format!(
                "quantized row {row} is outside pinned sidecar range {:?}",
                self.rows
            )));
        }
        Ok(row - self.rows.start)
    }

    pub(crate) fn scale(&self, row: usize) -> crate::Result<u16> {
        let local = self.local_row(row)?;
        let start = local * QUANTIZED_SCALE_STRIDE;
        Ok(u16::from_le_bytes(
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

    pub(crate) fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub(crate) fn gammas(&self) -> &[u8] {
        &self.gammas
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

    pub(crate) fn scale(&self, row: usize) -> crate::Result<u16> {
        self.local_row(row)?;
        let local = row - self.rows.start;
        let start = local * QUANTIZED_SCALE_STRIDE;
        Ok(u16::from_le_bytes(
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

    pub(crate) fn constant(&self, row: usize) -> crate::Result<f32> {
        let local = self.local_row(row)?;
        let start = local * QUANTIZED_CONSTANT_STRIDE;
        Ok(f32::from_le_bytes(
            self.constants[start..start + QUANTIZED_CONSTANT_STRIDE]
                .try_into()
                .unwrap(),
        ))
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

    pub(crate) fn constants(&self) -> &[u8] {
        &self.constants
    }

    pub(crate) fn code_stride(&self) -> usize {
        self.code_stride
    }
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

    fn cluster_rows_containing(&self, rows: &Range<usize>) -> crate::Result<Range<usize>> {
        if rows.is_empty() {
            return Ok(rows.clone());
        }
        let Some(&total_rows) = self.cluster_offsets.last() else {
            return Err(DataCorruption::comment_only(
                "quantized sidecar has no IVF cluster offsets",
            )
            .into());
        };
        if rows.end > total_rows {
            return Err(DataCorruption::comment_only(format!(
                "quantized sidecar row range {rows:?} exceeds {total_rows} posting rows"
            ))
            .into());
        }
        let upper = self
            .cluster_offsets
            .partition_point(|&offset| offset <= rows.start);
        if upper == 0 || upper == self.cluster_offsets.len() {
            return Err(DataCorruption::comment_only(format!(
                "quantized sidecar row range {rows:?} is outside IVF cluster offsets"
            ))
            .into());
        }
        let cluster_rows = self.cluster_offsets[upper - 1]..self.cluster_offsets[upper];
        if rows.end > cluster_rows.end {
            return Err(TantivyError::InternalError(format!(
                "quantized sidecar row range {rows:?} crosses cluster boundary {cluster_rows:?}"
            )));
        }
        Ok(cluster_rows)
    }

    fn sidecar_byte_ranges(
        cluster_rows: &Range<usize>,
        rows: &Range<usize>,
    ) -> (Range<usize>, Range<usize>) {
        debug_assert!(cluster_rows.start <= rows.start && rows.end <= cluster_rows.end);
        let block_start = cluster_rows.start * QUANTIZED_SCALE_GAMMA_STRIDE;
        let scale_start = block_start + (rows.start - cluster_rows.start) * QUANTIZED_SCALE_STRIDE;
        let scale_end = block_start + (rows.end - cluster_rows.start) * QUANTIZED_SCALE_STRIDE;
        let gamma_block_start = block_start + cluster_rows.len() * QUANTIZED_SCALE_STRIDE;
        let gamma_start =
            gamma_block_start + (rows.start - cluster_rows.start) * QUANTIZED_GAMMA_STRIDE;
        let gamma_end =
            gamma_block_start + (rows.end - cluster_rows.start) * QUANTIZED_GAMMA_STRIDE;
        (scale_start..scale_end, gamma_start..gamma_end)
    }

    fn validate_gammas(gammas: &[u8], rows: &Range<usize>) -> crate::Result<()> {
        debug_assert_eq!(gammas.len(), rows.len() * QUANTIZED_GAMMA_STRIDE);
        for (local, bytes) in gammas.chunks_exact(QUANTIZED_GAMMA_STRIDE).enumerate() {
            let gamma = f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap()));
            if !gamma.is_finite() || !(1.0..=4.0).contains(&gamma) {
                return Err(DataCorruption::comment_only(format!(
                    "quantized row {} has invalid cumulative gamma {gamma}; expected finite [1,4]",
                    rows.start + local
                ))
                .into());
            }
        }
        Ok(())
    }

    /// Pin matching scale and gamma runs for `rows`, which must remain inside
    /// one IVF cluster. The sidecar's cluster-blocked layout is resolved from
    /// persisted cluster offsets; no global row stride is assumed.
    pub(crate) fn read_scale_gamma(
        &self,
        rows: Range<usize>,
    ) -> crate::Result<QuantizedScaleGammaBatch> {
        if rows.is_empty() {
            return Ok(QuantizedScaleGammaBatch {
                scales: OwnedBytes::empty(),
                gammas: OwnedBytes::empty(),
                rows,
            });
        }
        let cluster_rows = self.cluster_rows_containing(&rows)?;
        let (scale_range, gamma_range) = Self::sidecar_byte_ranges(&cluster_rows, &rows);

        let (scales, gammas) = if rows == cluster_rows {
            // A complete posting is one physical sidecar block [4a,4b), so
            // pin it once and split the two runs without copying.
            let scale_len = scale_range.len();
            self.sidecar
                .slice(scale_range.start..gamma_range.end)
                .read_bytes()?
                .split(scale_len)
        } else {
            let overlapping_blocks = storage_block_span(&self.sidecar, scale_range.clone())
                .zip(storage_block_span(&self.sidecar, gamma_range.clone()))
                .is_some_and(|((scale_start, scale_end), (gamma_start, gamma_end))| {
                    scale_start <= gamma_end && gamma_start <= scale_end
                });
            if overlapping_blocks {
                // Both runs touch one physical block. Pin their enclosing
                // range once, then borrow only the two requested subranges.
                let bytes = self
                    .sidecar
                    .slice(scale_range.start..gamma_range.end)
                    .read_bytes()?;
                let scales = bytes.slice(0..scale_range.len());
                let gamma_start = gamma_range.start - scale_range.start;
                let gammas = bytes.slice(gamma_start..gamma_start + gamma_range.len());
                (scales, gammas)
            } else {
                (
                    self.sidecar.slice(scale_range).read_bytes()?,
                    self.sidecar.slice(gamma_range).read_bytes()?,
                )
            }
        };
        Self::validate_gammas(gammas.as_slice(), &rows)?;
        Ok(QuantizedScaleGammaBatch {
            scales,
            gammas,
            rows,
        })
    }

    pub(crate) fn read_codes(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        let codes = self
            .codes
            .slice(rows.start * self.code_stride..rows.end * self.code_stride)
            .read_bytes()?;
        // The zero-tail invariant is checked once per pinned range, never in
        // an indexed survivor loop.
        for (local, code) in codes.chunks_exact(self.code_stride).enumerate() {
            if !quantized_code_tail_is_zero(code, self.dim, self.bits) {
                let row = rows.start + local;
                return Err(DataCorruption::comment_only(format!(
                    "quantized row {row} has non-zero padding bits for d={} b={}",
                    self.dim, self.bits
                ))
                .into());
            }
        }
        Ok(codes)
    }

    pub(crate) fn read_scales(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        Ok(self.read_scale_gamma(rows)?.scales)
    }

    pub(crate) fn read_constants(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        Ok(self
            .constants
            .slice(rows.start * QUANTIZED_CONSTANT_STRIDE..rows.end * QUANTIZED_CONSTANT_STRIDE)
            .read_bytes()?)
    }

    pub(crate) fn read_batch(
        &self,
        rows: Range<usize>,
        read_constants: bool,
    ) -> crate::Result<QuantizedLayerBatch> {
        if rows.start > rows.end {
            return Err(TantivyError::InternalError(format!(
                "invalid quantized row range {rows:?}"
            )));
        }
        let codes = self.read_codes(rows.clone())?;
        let sidecar = self.read_scale_gamma(rows.clone())?;
        let constants = if read_constants {
            self.read_constants(rows.clone())?
        } else {
            OwnedBytes::empty()
        };
        Ok(QuantizedLayerBatch {
            codes,
            scales: sidecar.scales,
            gammas: sidecar.gammas,
            constants,
            rows,
            code_stride: self.code_stride,
        })
    }

    fn plan_slot_reads(
        slot: &FileSlice,
        stride: usize,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        debug_assert!(!rows.is_empty());
        debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(rows.iter().all(|row| available_rows.contains(row)));
        read_ranges.clear();

        block_scratch.clear();
        for &row in rows {
            if !append_storage_block_span(slot, row * stride..(row + 1) * stride, block_scratch) {
                // In-memory and ordinary-file handles expose no page
                // geometry. Pin the available borrowed slice once; never
                // fabricate a block size.
                read_ranges.push(available_rows);
                return;
            }
        }
        let touched_blocks = merged_storage_block_count(block_scratch);

        block_scratch.clear();
        debug_assert!(append_storage_block_span(
            slot,
            available_rows.start * stride..available_rows.end * stride,
            block_scratch,
        ));
        let covered_blocks = merged_storage_block_count(block_scratch);
        debug_assert!(touched_blocks <= covered_blocks);
        if touched_blocks == covered_blocks {
            read_ranges.push(available_rows);
            return;
        }

        // Sparse path: merge only overlapping physical-block intervals.
        // Adjacent blocks stay as separate reads so page-backed handles never
        // materialize a multi-page OwnedBytes merely for call coalescing.
        let mut first_row = rows[0];
        let mut previous_row = rows[0];
        let (_, mut group_end_block) =
            storage_block_span(slot, rows[0] * stride..(rows[0] + 1) * stride)
                .expect("storage geometry was resolved above");
        for &row in &rows[1..] {
            let (start_block, end_block) =
                storage_block_span(slot, row * stride..(row + 1) * stride)
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

        debug_assert!(read_ranges.windows(2).all(|pair| {
            let (_, left_end) =
                storage_block_span(slot, pair[0].start * stride..pair[0].end * stride).unwrap();
            let (right_start, _) =
                storage_block_span(slot, pair[1].start * stride..pair[1].end * stride).unwrap();
            left_end < right_start
        }));
    }

    pub(crate) fn plan_code_reads(
        &self,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        Self::plan_slot_reads(
            &self.codes,
            self.code_stride,
            available_rows,
            rows,
            read_ranges,
            block_scratch,
        );
    }

    fn plan_sidecar_reads_in_cluster(
        &self,
        cluster_rows: Range<usize>,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        debug_assert!(!rows.is_empty());
        debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(rows.iter().all(|row| available_rows.contains(row)));
        debug_assert!(cluster_rows.start <= available_rows.start);
        debug_assert!(available_rows.end <= cluster_rows.end);
        let first_output = read_ranges.len();

        block_scratch.clear();
        for &row in rows {
            let (scale, gamma) = Self::sidecar_byte_ranges(&cluster_rows, &(row..row + 1));
            if !append_storage_block_span(&self.sidecar, scale, block_scratch)
                || !append_storage_block_span(&self.sidecar, gamma, block_scratch)
            {
                // Without storage geometry the only safe and useful borrowed
                // unit is the available range within this cluster block.
                read_ranges.push(available_rows);
                return;
            }
        }
        let touched_blocks = merged_storage_block_count(block_scratch);

        block_scratch.clear();
        let (available_scales, available_gammas) =
            Self::sidecar_byte_ranges(&cluster_rows, &available_rows);
        debug_assert!(append_storage_block_span(
            &self.sidecar,
            available_scales,
            block_scratch
        ));
        debug_assert!(append_storage_block_span(
            &self.sidecar,
            available_gammas,
            block_scratch
        ));
        let covered_blocks = merged_storage_block_count(block_scratch);
        debug_assert!(touched_blocks <= covered_blocks);
        if touched_blocks == covered_blocks {
            read_ranges.push(available_rows);
            return;
        }

        let block_spans = |rows: Range<usize>| {
            let (scale, gamma) = Self::sidecar_byte_ranges(&cluster_rows, &rows);
            (
                storage_block_span(&self.sidecar, scale)
                    .expect("sidecar storage geometry was resolved above"),
                storage_block_span(&self.sidecar, gamma)
                    .expect("sidecar storage geometry was resolved above"),
            )
        };
        let spans_overlap = |left: ((usize, usize), (usize, usize)),
                             right: ((usize, usize), (usize, usize))| {
            [left.0, left.1].into_iter().any(|(left_start, left_end)| {
                [right.0, right.1]
                    .into_iter()
                    .any(|(right_start, right_end)| {
                        left_start <= right_end && right_start <= left_end
                    })
            })
        };

        // Sparse path: group selected rows whenever either their scale pages
        // or gamma pages overlap. Each resulting pair of borrowed runs is
        // disjoint from every other pair at storage-block granularity.
        let mut first_row = rows[0];
        let mut previous_row = rows[0];
        for &row in &rows[1..] {
            let current = block_spans(first_row..previous_row + 1);
            let next = block_spans(row..row + 1);
            if spans_overlap(current, next) {
                previous_row = row;
            } else {
                read_ranges.push(first_row..previous_row + 1);
                first_row = row;
                previous_row = row;
            }
        }
        read_ranges.push(first_row..previous_row + 1);

        // Expanding from selected rows to page-aligned row groups can erase a
        // nominal sparse advantage. If it covers the whole block after all,
        // use one pin for the available range within this cluster.
        block_scratch.clear();
        for range in read_ranges[first_output..].iter().cloned() {
            let (scale, gamma) = Self::sidecar_byte_ranges(&cluster_rows, &range);
            debug_assert!(append_storage_block_span(
                &self.sidecar,
                scale,
                block_scratch
            ));
            debug_assert!(append_storage_block_span(
                &self.sidecar,
                gamma,
                block_scratch
            ));
        }
        if merged_storage_block_count(block_scratch) == covered_blocks {
            read_ranges.truncate(first_output);
            read_ranges.push(available_rows);
        }
    }

    pub(crate) fn plan_sidecar_reads(
        &self,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        debug_assert!(!rows.is_empty());
        debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(rows.iter().all(|row| available_rows.contains(row)));
        read_ranges.clear();

        // Cosine refinement may batch survivors across clusters. Split that
        // logical batch at persisted cluster boundaries before mapping either
        // sidecar run; a row range alone does not identify gamma's offset.
        let mut selected_start = 0usize;
        while selected_start < rows.len() {
            let first_row = rows[selected_start];
            let cluster_rows = self
                .cluster_rows_containing(&(first_row..first_row + 1))
                .expect("indexed sidecar row must belong to an IVF cluster");
            let selected_end = selected_start
                + rows[selected_start..].partition_point(|&row| row < cluster_rows.end);
            let cluster_available = available_rows.start.max(cluster_rows.start)
                ..available_rows.end.min(cluster_rows.end);
            self.plan_sidecar_reads_in_cluster(
                cluster_rows,
                cluster_available,
                &rows[selected_start..selected_end],
                read_ranges,
                block_scratch,
            );
            selected_start = selected_end;
        }
    }

    /// Compatibility name for the refinement path. Planning includes both
    /// scale and gamma runs; [`Self::read_scales`] validates the paired gamma
    /// range even until the scorer consumes gamma directly.
    pub(crate) fn plan_scale_reads(
        &self,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        self.plan_sidecar_reads(available_rows, rows, read_ranges, block_scratch);
    }

    pub(crate) fn plan_constant_reads(
        &self,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        Self::plan_slot_reads(
            &self.constants,
            QUANTIZED_CONSTANT_STRIDE,
            available_rows,
            rows,
            read_ranges,
            block_scratch,
        );
    }

    pub(crate) fn code_bytes(&self, row: usize) -> crate::Result<OwnedBytes> {
        let bytes = self
            .codes
            .slice(row * self.code_stride..(row + 1) * self.code_stride)
            .read_bytes()
            .map_err(TantivyError::from)?;
        if !quantized_code_tail_is_zero(bytes.as_slice(), self.dim, self.bits) {
            return Err(DataCorruption::comment_only(format!(
                "quantized row {row} has non-zero padding bits for d={} b={}",
                self.dim, self.bits
            ))
            .into());
        }
        Ok(bytes)
    }

    pub(crate) fn scale(&self, row: usize) -> crate::Result<u16> {
        self.read_scale_gamma(row..row + 1)?.scale(row)
    }

    pub(crate) fn gamma(&self, row: usize) -> crate::Result<f32> {
        self.read_scale_gamma(row..row + 1)?.gamma(row)
    }

    pub(crate) fn constant(&self, row: usize) -> crate::Result<f32> {
        let bytes = self
            .constants
            .slice(row * QUANTIZED_CONSTANT_STRIDE..(row + 1) * QUANTIZED_CONSTANT_STRIDE)
            .read_bytes()?;
        Ok(f32::from_le_bytes(bytes.as_slice().try_into().unwrap()))
    }
}

/// Field-keyed V3 quantized payloads resolved from immutable index metadata.
pub(crate) struct QuantizedFieldReader {
    config: VectorQuantizationConfig,
    // Positive-depth scans resolve this once for the segment/field. Level 0
    // never touches it, so unquantized IVF does not construct scorer state.
    index_ctx: OnceLock<Arc<QuantizedIndexCtx>>,
    layers: Vec<QuantizedLayerReader>,
    residual_norms: Option<FileSlice>,
}

pub(crate) struct QuantizedResidualNormBatch {
    bytes: OwnedBytes,
    rows: Range<usize>,
}

impl QuantizedResidualNormBatch {
    /// The complete fixed-stride LE-f32 range for one cluster pin. Callers
    /// decode this once as a contiguous pass rather than doing checked
    /// per-row lookups in the scoring loop.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        debug_assert_eq!(
            self.bytes.len(),
            self.rows.len() * QUANTIZED_RESIDUAL_NORM_STRIDE
        );
        &self.bytes
    }
}

impl QuantizedFieldReader {
    pub(crate) fn config(&self) -> &VectorQuantizationConfig {
        &self.config
    }

    pub(crate) fn layers(&self) -> &[QuantizedLayerReader] {
        &self.layers
    }

    pub(crate) fn residual_norm(&self, row: usize) -> crate::Result<Option<f32>> {
        let Some(residual_norms) = &self.residual_norms else {
            return Ok(None);
        };
        let bytes = residual_norms
            .slice(row * QUANTIZED_RESIDUAL_NORM_STRIDE..(row + 1) * QUANTIZED_RESIDUAL_NORM_STRIDE)
            .read_bytes()?;
        Ok(Some(f32::from_le_bytes(
            bytes.as_slice().try_into().unwrap(),
        )))
    }

    pub(crate) fn read_residual_norm_batch(
        &self,
        rows: Range<usize>,
    ) -> crate::Result<Option<QuantizedResidualNormBatch>> {
        let Some(residual_norms) = &self.residual_norms else {
            return Ok(None);
        };
        let bytes = residual_norms
            .slice(
                rows.start * QUANTIZED_RESIDUAL_NORM_STRIDE
                    ..rows.end * QUANTIZED_RESIDUAL_NORM_STRIDE,
            )
            .read_bytes()?;
        Ok(Some(QuantizedResidualNormBatch { bytes, rows }))
    }

    pub(crate) fn index_ctx(&self) -> crate::Result<Arc<QuantizedIndexCtx>> {
        if let Some(index_ctx) = self.index_ctx.get() {
            return Ok(Arc::clone(index_ctx));
        }
        let resolved = QuantizedIndexCtx::resolve_from_config(self.config.clone())?;
        // Concurrent segment users may race through the first resolution.
        // The process cache returns the same Arc identity, and OnceLock keeps
        // exactly one of them without caching a transient construction error.
        let _ = self.index_ctx.set(Arc::clone(&resolved));
        Ok(self.index_ctx.get().map(Arc::clone).unwrap_or(resolved))
    }

    #[cfg(test)]
    pub(crate) fn index_ctx_is_initialized(&self) -> bool {
        self.index_ctx.get().is_some()
    }
}

fn logical_slice(
    slice: FileSlice,
    logical_len: usize,
    description: &str,
) -> crate::Result<FileSlice> {
    let physical_len = slice.len();
    if physical_len < logical_len || physical_len - logical_len >= 64 {
        return Err(DataCorruption::comment_only(format!(
            "{description} physical length {physical_len} does not contain logical length \
             {logical_len} plus at most 63 alignment bytes"
        ))
        .into());
    }
    if physical_len != logical_len {
        let trailer = slice.slice(logical_len..physical_len).read_bytes()?;
        if trailer.iter().any(|&byte| byte != 0) {
            return Err(DataCorruption::comment_only(format!(
                "{description} has a non-zero alignment trailer"
            ))
            .into());
        }
    }
    Ok(slice.slice_to(logical_len))
}

/// Per-(segment, field) vector reader: the row store plus, for IVF segments,
/// the routing index. See the module docs for the layout and the
/// pinned-vs-deferred split.
pub struct VectorIndexReader {
    options: VectorOptions,
    /// Distinct docs with a vector (the IdMap row count for flat storage; the
    /// persisted doc count for IVF, whose row total legacy V2 replication
    /// could inflate).
    num_vectors: usize,
    /// `false` for the placeholder built by [`Self::empty`] — the segment has
    /// no vector data for this field at all.
    present: bool,
    /// `.vec` slot `[0]`
    id_map: IdMap,
    /// `.vec` slot `[1]`: the dense vector rows. Never materialized whole;
    /// queries fetch per-cluster (or per-doc) ranges.
    rows_slice: FileSlice,
    index: Option<IvfIndex>,
    quantization: Option<QuantizedFieldReader>,
}

impl VectorIndexReader {
    /// Opens `field`'s vector data in `segment_reader`'s segment. Returns the
    /// [`empty`](Self::empty) placeholder when the segment carries no vector
    /// data for the field (no `.vec` file, or the field has no slots in it),
    /// mirroring `SegmentReader::inverted_index`.
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
            Err(OpenReadError::FileDoesNotExist(_)) => return Ok(Self::empty(options)),
            Err(err) => return Err(err.into()),
        };
        let (version, body) = read_header(&vec_file)?;
        let vec_composite = CompositeFile::open(&body)?;
        let (Some(id_map_slice), Some(rows_slice)) = (
            vec_composite.open_read_with_idx(field, vec_slot::ID_MAP),
            vec_composite.open_read_with_idx(field, vec_slot::ROWS),
        ) else {
            return Ok(Self::empty(options));
        };
        let id_map = IdMap::open(id_map_slice, segment_reader.max_doc())?;
        let quantization_config = segment_reader
            .index_settings()
            .vector_quantization
            .iter()
            .find(|config| config.field == entry.name())
            .cloned();

        let centroid_slots = match segment_reader
            .open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string()))
        {
            Ok(file) => {
                let (centroids_version, body) = read_header(&file)?;
                let composite = CompositeFile::open(&body)?;
                match (
                    composite.open_read_with_idx(field, centroid_slot::CENTROIDS),
                    composite.open_read_with_idx(field, centroid_slot::OFFSETS),
                    composite.open_read_with_idx(field, centroid_slot::BOUNDS),
                ) {
                    // Slot [2] (the router) stays optional: the write
                    // side skips it for degenerate centroid counts. Slot
                    // [3] (bounds) is read whenever present; when absent,
                    // a V1 file simply predates it — `IvfIndex::open`
                    // synthesizes SATURATED bounds — while a V2+ file
                    // without it is corrupt, not old.
                    (Some(centroids), Some(offsets), bounds) => {
                        if bounds.is_none() && centroids_version >= VectorFileVersion::V2 {
                            return Err(TantivyError::InternalError(format!(
                                "vector field {:?} has a V2 `.centroids` file with no bounds slot",
                                entry.name()
                            )));
                        }
                        Some((
                            centroids_version,
                            centroids,
                            offsets,
                            composite.open_read_with_idx(field, centroid_slot::ROUTER),
                            bounds,
                        ))
                    }
                    _ => None,
                }
            }
            Err(OpenReadError::FileDoesNotExist(_)) => None,
            Err(err) => return Err(err.into()),
        };

        // The id-map variant and the `.centroids` sidecar are two signals of
        // one write-path decision; a mismatch means a corrupt segment, never a
        // fallback.
        let index = match (&id_map, centroid_slots) {
            (IdMap::Explicit(_), Some((version, centroids, offsets, router, bounds))) => Some(
                IvfIndex::open(version, &options, centroids, offsets, router, bounds)?,
            ),
            (IdMap::Explicit(_), None) => {
                return Err(TantivyError::InternalError(format!(
                    "vector field {:?} has cluster-sorted rows but no `.centroids` data",
                    entry.name()
                )));
            }
            (_, Some(_)) => {
                return Err(TantivyError::InternalError(format!(
                    "vector field {:?} has `.centroids` data but doc-ordered rows",
                    entry.name()
                )));
            }
            (_, None) => None,
        };

        let num_rows = id_map.num_rows() as usize;
        if let Some(index) = &index {
            if index.num_rows() != num_rows {
                return Err(TantivyError::InternalError(
                    "IVF id-map length does not match the cluster offsets".to_string(),
                ));
            }
        }
        let rows_slice = logical_slice(
            rows_slice,
            num_rows * options.bytes_per_vector(),
            &format!("vector field {:?} rows", entry.name()),
        )?;

        let first_quantized_slot = vec_composite
            .open_read_with_idx(field, vec_slot::quantized_codes(0))
            .is_some()
            || vec_composite
                .open_read_with_idx(field, vec_slot::QUANTIZED_RESIDUAL_NORMS)
                .is_some();
        let quantization = match (&index, quantization_config) {
            (Some(index), Some(config)) => {
                if version < VectorFileVersion::V3 {
                    return Err(DataCorruption::comment_only(format!(
                        "vector field {:?} enables quantization but its IVF rows predate V3",
                        entry.name()
                    ))
                    .into());
                }
                let cluster_offsets: Arc<[usize]> = (0..index.num_clusters())
                    .map(|cluster| index.cluster_range(cluster).start)
                    .chain(std::iter::once(index.num_rows()))
                    .collect::<Vec<_>>()
                    .into();
                let mut layers = Vec::with_capacity(config.layers.len());
                for (layer, spec) in config.layers.iter().enumerate() {
                    let code_stride = quantized_code_stride(options.dim(), spec.bits);
                    let (Some(codes), Some(sidecar), Some(constants)) = (
                        vec_composite.open_read_with_idx(field, vec_slot::quantized_codes(layer)),
                        vec_composite.open_read_with_idx(field, vec_slot::quantized_scales(layer)),
                        vec_composite
                            .open_read_with_idx(field, vec_slot::quantized_constants(layer)),
                    ) else {
                        return Err(DataCorruption::comment_only(format!(
                            "vector field {:?} has an incomplete configured quantization layer \
                             {layer}",
                            entry.name()
                        ))
                        .into());
                    };
                    layers.push(QuantizedLayerReader {
                        codes: logical_slice(
                            codes,
                            num_rows * code_stride,
                            &format!("vector field {:?} layer {layer} codes", entry.name()),
                        )?,
                        sidecar: logical_slice(
                            sidecar,
                            num_rows * QUANTIZED_SCALE_GAMMA_STRIDE,
                            &format!(
                                "vector field {:?} layer {layer} scale/gamma sidecar",
                                entry.name()
                            ),
                        )?,
                        constants: logical_slice(
                            constants,
                            num_rows * QUANTIZED_CONSTANT_STRIDE,
                            &format!("vector field {:?} layer {layer} constants", entry.name()),
                        )?,
                        cluster_offsets: Arc::clone(&cluster_offsets),
                        code_stride,
                        dim: options.dim(),
                        bits: spec.bits,
                    });
                }
                for layer in config.layers.len()..4 {
                    if vec_composite
                        .open_read_with_idx(field, vec_slot::quantized_codes(layer))
                        .is_some()
                        || vec_composite
                            .open_read_with_idx(field, vec_slot::quantized_scales(layer))
                            .is_some()
                        || vec_composite
                            .open_read_with_idx(field, vec_slot::quantized_constants(layer))
                            .is_some()
                    {
                        return Err(DataCorruption::comment_only(format!(
                            "vector field {:?} carries quantization layers beyond its configured \
                             prefix",
                            entry.name()
                        ))
                        .into());
                    }
                }
                let residual_norm_slot =
                    vec_composite.open_read_with_idx(field, vec_slot::QUANTIZED_RESIDUAL_NORMS);
                let residual_norms = match (config.needs_residual_norm(), residual_norm_slot) {
                    (true, Some(slice)) => Some(logical_slice(
                        slice,
                        num_rows * QUANTIZED_RESIDUAL_NORM_STRIDE,
                        &format!("vector field {:?} residual squared norms", entry.name()),
                    )?),
                    (true, None) => {
                        return Err(DataCorruption::comment_only(format!(
                            "L2 vector field {:?} is missing residual squared norm slot 14",
                            entry.name()
                        ))
                        .into());
                    }
                    (false, Some(_)) => {
                        return Err(DataCorruption::comment_only(format!(
                            "vector field {:?} carries residual norms for a metric that omits them",
                            entry.name()
                        ))
                        .into());
                    }
                    (false, None) => None,
                };
                // Slot 15 was retired before V3 shipped. Calibration lives in
                // index settings; any physical slot 15 bytes are ignored.
                Some(QuantizedFieldReader {
                    config,
                    index_ctx: OnceLock::new(),
                    layers,
                    residual_norms,
                })
            }
            (Some(_), None) if first_quantized_slot => {
                return Err(DataCorruption::comment_only(format!(
                    "vector field {:?} carries quantized slots without index metadata",
                    entry.name()
                ))
                .into());
            }
            (None, _) if first_quantized_slot => {
                return Err(DataCorruption::comment_only(format!(
                    "flat vector field {:?} unexpectedly carries quantized slots",
                    entry.name()
                ))
                .into());
            }
            _ => None,
        };

        let num_vectors = match &index {
            Some(index) => index.num_docs(),
            None => num_rows,
        };
        Ok(Self {
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
    pub(crate) fn empty(options: VectorOptions) -> Self {
        Self {
            options,
            num_vectors: 0,
            present: false,
            rows_slice: FileSlice::empty(),
            id_map: IdMap::Identity { num_docs: 0 },
            index: None,
            quantization: None,
        }
    }

    pub fn options(&self) -> &VectorOptions {
        &self.options
    }

    pub fn dim(&self) -> usize {
        self.options.dim()
    }

    /// Number of distinct docs with a vector value.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// The routing index, present iff the segment's rows are IVF-clustered.
    /// `None` means search must scan the rows exactly.
    pub fn index(&self) -> Option<&IvfIndex> {
        self.index.as_ref()
    }

    pub(crate) fn quantization(&self) -> Option<&QuantizedFieldReader> {
        self.quantization.as_ref()
    }

    /// Storage info for tooling; `None` if the segment has no vector data for
    /// the field.
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

    /// Deterministically sample distinct live stored vectors for a held-out
    /// gamma audit. Replica memberships collapse to the first posting row for
    /// each segment-local `doc_id`; interval sampling over sorted `doc_id`s
    /// returns exactly `count` queries when that many distinct live documents
    /// exist, and every returned query carries the id whose complete
    /// membership set must be excluded in its origin segment.
    pub fn sample_gamma_pseudo_queries(
        &self,
        count: usize,
        alive: Option<&AliveBitSet>,
    ) -> crate::Result<Option<Vec<VectorGammaAuditQuery>>> {
        let (Some(index), Some(_quantization)) = (&self.index, &self.quantization) else {
            return Ok(None);
        };
        if count == 0 {
            return Ok(Some(Vec::new()));
        }

        let mut first_row_by_doc = BTreeMap::new();
        for row in 0..index.num_rows() {
            let doc_id = self.doc_id_at(row);
            if alive.is_some_and(|alive| !alive.is_alive(doc_id)) {
                continue;
            }
            first_row_by_doc.entry(doc_id).or_insert(row);
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
            queries.push(VectorGammaAuditQuery {
                values,
                excluded_doc_id: Some(doc_id),
            });
        }
        Ok(Some(queries))
    }

    /// Count distinct live IVF documents without decoding vector rows. This
    /// lets a multi-segment coordinator apportion an exact global held-out
    /// query count before calling [`Self::sample_gamma_pseudo_queries`].
    pub fn live_distinct_vector_count(&self, alive: Option<&AliveBitSet>) -> usize {
        let Some(index) = &self.index else {
            return 0;
        };
        let mut docs = BTreeMap::new();
        for row in 0..index.num_rows() {
            let doc_id = self.doc_id_at(row);
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                docs.insert(doc_id, ());
            }
        }
        docs.len()
    }

    /// Run the non-persisting gamma audit over one deterministic interval
    /// sample of posting rows. Calling this twice with the same `sample_rows`
    /// and alive set gives the real-query and held-out protocols the exact
    /// same target rows. A held-out caller supplies `excluded_doc_id` only in
    /// the sampled query's origin segment; external queries and other
    /// segments leave it `None`.
    ///
    /// This deliberately resolves a calibration-independent production query
    /// context: rotations, bitplanes, LUTs, stored f16 scales and gammas,
    /// split constants, and metric-specific query norms match serving. The
    /// persisted diagnostic bias/spread never participates in the measurement.
    pub fn audit_gamma_queries(
        &self,
        source: VectorQuantizationCalibrationSource,
        queries: &[VectorGammaAuditQuery],
        sample_rows: usize,
        alive: Option<&AliveBitSet>,
    ) -> crate::Result<Option<VectorGammaAuditMeasurements>> {
        let (Some(index), Some(quantization)) = (&self.index, &self.quantization) else {
            return Ok(None);
        };
        if sample_rows == 0 {
            return Err(TantivyError::InvalidArgument(
                "gamma audit sample_rows must be greater than zero".to_string(),
            ));
        }
        for query in queries {
            if query.values.len() != self.options.dim() {
                return Err(TantivyError::InvalidArgument(format!(
                    "gamma audit query has dimension {}; expected {}",
                    query.values.len(),
                    self.options.dim()
                )));
            }
        }

        let layer_count = quantization.config.layers.len();
        let mut measurements = VectorGammaAuditMeasurements {
            source,
            calibration: VectorCalibrationMeasurements {
                aggregate: vec![VectorCalibrationMoments::default(); layer_count],
                per_query: vec![
                    vec![VectorCalibrationMoments::default(); layer_count];
                    queries.len()
                ],
            },
            depths: vec![VectorGammaDepthMeasurements::default(); layer_count],
        };
        if index.num_rows() == 0 || queries.is_empty() {
            return Ok(Some(measurements));
        }

        let measurement_ctx =
            QuantizedIndexCtx::for_calibration_measurement(quantization.config.clone())?;
        if measurement_ctx
            .specs
            .first()
            .is_none_or(|spec| spec.bits != 1)
        {
            return Err(TantivyError::InvalidArgument(
                "gamma audit requires a leading 1-bit sign layer".to_string(),
            ));
        }
        let prepared_queries: Vec<QuantizedQueryCtx> = queries
            .iter()
            .map(|query| QuantizedQueryCtx::new(Arc::clone(&measurement_ctx), query.values.clone()))
            .collect();
        let excluded_clusters = excluded_membership_clusters(
            queries,
            (0..index.num_clusters()).flat_map(|cluster| {
                index
                    .cluster_range(cluster)
                    .map(move |row| (cluster, self.doc_id_at(row)))
            }),
        );

        let live_row_count = self.live_posting_row_count(alive);
        let target_rows = sample_rows.min(live_row_count);
        if target_rows == 0 {
            return Ok(Some(measurements));
        }
        let centroid_stride = self.options.bytes_per_vector();
        let centroid_bytes = index.centroid_bytes()?;
        let mut sampled = 0usize;
        let mut live_rows_seen = 0usize;
        let mut next_sample_ordinal = 0usize;

        for cluster in 0..index.num_clusters() {
            let centroid_row = &centroid_bytes[cluster * centroid_stride..][..centroid_stride];
            let centroid = decode_row::<f32>(centroid_row, self.options.dim())?;
            for row in index.cluster_range(cluster) {
                if alive.is_some_and(|alive| !alive.is_alive(self.doc_id_at(row))) {
                    continue;
                }
                let live_ordinal = live_rows_seen;
                live_rows_seen += 1;
                if sampled >= target_rows || live_ordinal != next_sample_ordinal {
                    continue;
                }
                sampled += 1;
                next_sample_ordinal = sampled * live_row_count / target_rows;

                let values =
                    decode_row::<f32>(&self.vector_bytes_for_row(row)?, self.options.dim())?;
                let residual: Vec<f32> = values
                    .iter()
                    .zip(&centroid)
                    .map(|(&value, &center)| value - center)
                    .collect();
                let gamma_audit =
                    audit_prefix_gammas(&residual, &measurement_ctx.specs, &measurement_ctx.grids);
                debug_assert_eq!(gamma_audit.prefixes.len(), layer_count);

                let mut stored_layers = Vec::with_capacity(layer_count);
                for (depth, layer) in quantization.layers.iter().enumerate() {
                    let codes = layer.code_bytes(row)?;
                    let sidecar = layer.read_scale_gamma(row..row + 1)?;
                    let scale = sidecar.scale(row)?;
                    let stored_gamma = sidecar.gamma(row)?;
                    let constant = if self.options.metric() == Metric::L2 {
                        layer.constant(row)?
                    } else {
                        0.0
                    };
                    let expected = &gamma_audit.prefixes[depth];
                    if codes.as_slice() != expected.codes {
                        return Err(DataCorruption::comment_only(format!(
                            "gamma audit row {row} layer {depth} stored codes differ from the \
                             deterministic encoder"
                        ))
                        .into());
                    }
                    if scale != expected.layer_scale_f16 {
                        return Err(DataCorruption::comment_only(format!(
                            "gamma audit row {row} layer {depth} stored scale differs from the \
                             deterministic encoder"
                        ))
                        .into());
                    }
                    verify_stored_gamma(row, depth, stored_gamma, expected.gamma.f16_value())?;
                    stored_layers.push((codes, scale, constant, stored_gamma));
                }

                let stored_scale_one = f16_to_f32(stored_layers[0].1);
                let gamma_one = stored_layers[0].3;
                let stored_effective_scales_squared: Vec<f32> = stored_layers
                    .iter()
                    .map(|(_, _, _, gamma)| {
                        gamma_effective_scale_squared(
                            stored_scale_one * stored_scale_one,
                            gamma_one,
                            *gamma,
                        )
                    })
                    .collect();
                for (depth, prefix) in gamma_audit.prefixes.iter().enumerate() {
                    let stored_gamma = stored_layers[depth].3;
                    let depth_measurements = &mut measurements.depths[depth];
                    depth_measurements.gamma_raw.observe(prefix.gamma.raw);
                    depth_measurements
                        .gamma_clamped
                        .observe(f64::from(prefix.gamma.clamped));
                    depth_measurements
                        .gamma_f16
                        .observe(f64::from(stored_gamma));
                    depth_measurements
                        .gamma_f16_roundtrip_error
                        .observe(f64::from(stored_gamma) - f64::from(prefix.gamma.clamped));
                    depth_measurements
                        .raw_effective_scale_squared
                        .observe(prefix.s_eff_sq);
                    depth_measurements
                        .stored_effective_scale_squared
                        .observe(f64::from(stored_effective_scales_squared[depth]));
                    if gamma_audit.norm_sq == 0.0 && prefix.prefix_dot == 0.0 {
                        depth_measurements.zero_count += 1;
                    }
                    if !prefix.gamma.raw.is_finite()
                        || prefix.gamma.raw < 1.0
                        || prefix.gamma.raw > 4.0
                    {
                        depth_measurements.clamp_count += 1;
                    }
                    let reconstruction_norm_squared = prefix
                        .raw_prefix_reconstruction
                        .iter()
                        .map(|&value| f64::from(value).powi(2))
                        .sum::<f64>();
                    if prefix.prefix_dot != 0.0 && prefix.prefix_dot.is_finite() {
                        depth_measurements.orthogonality_defect.observe(
                            (reconstruction_norm_squared - prefix.prefix_dot) / prefix.prefix_dot,
                        );
                    }
                }

                for (query_idx, query) in prepared_queries.iter().enumerate() {
                    if excluded_clusters[query_idx].binary_search(&cluster).is_ok() {
                        continue;
                    }
                    let mut exact_dot = 0.0_f32;
                    for ((&residual_value, &center), &query_value) in
                        residual.iter().zip(&centroid).zip(query.query())
                    {
                        let score_query = if self.options.metric() == Metric::L2 {
                            query_value - center
                        } else {
                            query_value
                        };
                        exact_dot += residual_value * score_query;
                    }
                    let query_norm =
                        calibration_query_norm(query, self.options.metric(), centroid_row);
                    let query_norm_squared = query_norm * query_norm;
                    let mut raw_prefix_estimate = 0.0_f32;
                    let mut sign_query_error_term = 0.0_f32;
                    for (depth, ((codes, scale, constant, stored_gamma), prefix)) in
                        stored_layers.iter().zip(&gamma_audit.prefixes).enumerate()
                    {
                        raw_prefix_estimate += query.score_layer(depth, codes, *scale, *constant);
                        if measurement_ctx.specs[depth].bits == 1 {
                            let scale = f16_to_f32(*scale);
                            sign_query_error_term +=
                                scale * scale * query.query_error_squared(depth) as f32;
                        }
                        let gamma = *stored_gamma;
                        let model_sigma = gamma_model_variance(
                            stored_effective_scales_squared[depth],
                            gamma,
                            query_norm_squared,
                            sign_query_error_term,
                        )
                        .sqrt();
                        // H1/Q1 measures the formula-native S=1 spread. The
                        // serving safety multiplier is deliberately absent.
                        let corrected_prefix = *stored_gamma * raw_prefix_estimate;
                        observe_gamma_corrected_prefix(
                            &mut measurements.calibration,
                            query_idx,
                            depth,
                            exact_dot,
                            corrected_prefix,
                            f64::from(model_sigma),
                        );
                        if model_sigma > 0.0 && model_sigma.is_finite() {
                            let model_sigma = f64::from(model_sigma);
                            measurements.depths[depth].f16_band_error.observe(
                                (f64::from(gamma) - f64::from(prefix.gamma.clamped))
                                    * f64::from(raw_prefix_estimate)
                                    / model_sigma,
                            );
                            measurements.depths[depth].clamp_band_error.observe(
                                (f64::from(prefix.gamma.clamped) - prefix.gamma.raw)
                                    * f64::from(raw_prefix_estimate)
                                    / model_sigma,
                            );
                        }
                    }
                }
            }
        }
        Ok(Some(measurements))
    }

    /// Audit confidence-cone survivor economics with routing removed from the
    /// experiment. Every live posting membership in every cluster is scored;
    /// the two production boundary rules then run segment-wide at `k=10` and
    /// `kappa=(2,4)`. Exactly 100 external queries and a `[1,4]` schedule are
    /// required so no query/source/schedule fallback can silently change the
    /// ruled protocol.
    ///
    /// The estimates are gamma-corrected and deliberately uncentered. Measured
    /// calibration bias remains a separate regression tripwire and never
    /// enters this candidate-recall experiment.
    pub fn audit_gamma_cone(
        &self,
        queries: &[VectorGammaAuditQuery],
        alive: Option<&AliveBitSet>,
    ) -> crate::Result<Option<VectorGammaConeAuditMeasurements>> {
        let (Some(index), Some(quantization)) = (&self.index, &self.quantization) else {
            return Ok(None);
        };
        if queries.len() != GAMMA_CONE_QUERY_COUNT {
            return Err(TantivyError::InvalidArgument(format!(
                "gamma cone audit requires exactly {GAMMA_CONE_QUERY_COUNT} external queries; \
                 received {}",
                queries.len()
            )));
        }
        for query in queries {
            if query.excluded_doc_id.is_some() {
                return Err(TantivyError::InvalidArgument(
                    "gamma cone audit accepts external queries only".to_string(),
                ));
            }
            if query.values.len() != self.options.dim() {
                return Err(TantivyError::InvalidArgument(format!(
                    "gamma cone audit query has dimension {}; expected {}",
                    query.values.len(),
                    self.options.dim()
                )));
            }
            if query.values.iter().any(|value| !value.is_finite()) {
                return Err(TantivyError::InvalidArgument(
                    "gamma cone audit queries must contain only finite values".to_string(),
                ));
            }
        }

        let measurement_ctx =
            QuantizedIndexCtx::for_calibration_measurement(quantization.config.clone())?;
        if measurement_ctx.specs.len() != 2
            || measurement_ctx.specs[0].bits != 1
            || measurement_ctx.specs[1].bits != 4
        {
            return Err(TantivyError::InvalidArgument(
                "gamma cone audit requires the [1,4] schedule".to_string(),
            ));
        }
        let live_docs = self.live_distinct_vector_count(alive);
        if live_docs < GAMMA_CONE_TOP_K {
            return Err(TantivyError::InvalidArgument(format!(
                "gamma cone audit requires at least {GAMMA_CONE_TOP_K} live documents; found \
                 {live_docs}"
            )));
        }

        let row_count = index.num_rows();
        let layer_count = measurement_ctx.specs.len();
        let mut gammas = vec![vec![f32::NAN; row_count]; layer_count];
        let mut stored_scales = vec![vec![f32::NAN; row_count]; layer_count];
        let mut effective_scales_squared = vec![vec![f32::NAN; row_count]; layer_count];
        let mut residual_norms_squared =
            (self.options.metric() == Metric::L2).then(|| vec![f32::NAN; row_count]);
        let centroid_stride = self.options.bytes_per_vector();
        let centroid_bytes = index.centroid_bytes()?;

        // Gamma is a row property, independent of the query. Read the
        // persisted sidecar once for every live membership. Deterministic
        // reconstruction below is only an equality/corruption receipt; its
        // gamma never substitutes for the stored value.
        for cluster in 0..index.num_clusters() {
            let centroid_row = &centroid_bytes[cluster * centroid_stride..][..centroid_stride];
            let centroid = decode_row::<f32>(centroid_row, self.options.dim())?;
            for row in index.cluster_range(cluster) {
                let doc = self.doc_id_at(row);
                if alive.is_some_and(|alive| !alive.is_alive(doc)) {
                    continue;
                }
                let values =
                    decode_row::<f32>(&self.vector_bytes_for_row(row)?, self.options.dim())?;
                let residual: Vec<f32> = values
                    .iter()
                    .zip(&centroid)
                    .map(|(&value, &center)| value - center)
                    .collect();
                let audit =
                    audit_prefix_gammas(&residual, &measurement_ctx.specs, &measurement_ctx.grids);
                let mut decoded_scales = Vec::with_capacity(layer_count);
                for (depth, layer) in quantization.layers.iter().enumerate() {
                    let sidecar = layer.read_scale_gamma(row..row + 1)?;
                    let scale = sidecar.scale(row)?;
                    let stored_gamma = sidecar.gamma(row)?;
                    let expected = &audit.prefixes[depth];
                    if scale != expected.layer_scale_f16 {
                        return Err(DataCorruption::comment_only(format!(
                            "gamma cone row {row} layer {depth} stored scale differs from the \
                             deterministic encoder"
                        ))
                        .into());
                    }
                    let codes = layer.code_bytes(row)?;
                    if codes.as_slice() != expected.codes {
                        return Err(DataCorruption::comment_only(format!(
                            "gamma cone row {row} layer {depth} stored codes differ from the \
                             deterministic encoder"
                        ))
                        .into());
                    }
                    verify_stored_gamma(row, depth, stored_gamma, expected.gamma.f16_value())?;
                    let scale = f16_to_f32(scale);
                    stored_scales[depth][row] = scale;
                    gammas[depth][row] = stored_gamma;
                    decoded_scales.push(scale);
                }
                let scale_one_squared = decoded_scales[0] * decoded_scales[0];
                let gamma_one = gammas[0][row];
                for depth in 0..layer_count {
                    effective_scales_squared[depth][row] = gamma_effective_scale_squared(
                        scale_one_squared,
                        gamma_one,
                        gammas[depth][row],
                    );
                }
                if let Some(norms) = &mut residual_norms_squared {
                    norms[row] = quantization.residual_norm(row)?.ok_or_else(|| {
                        TantivyError::DataCorruption(DataCorruption::comment_only(
                            "quantized L2 field is missing residual-norm slot 14",
                        ))
                    })?;
                }
            }
        }

        let metric = self.options.metric();
        let mut measurements = VectorGammaConeAuditMeasurements {
            query_count: GAMMA_CONE_QUERY_COUNT as u32,
            top_k: GAMMA_CONE_TOP_K as u32,
            depths: GAMMA_CONE_KAPPAS
                .into_iter()
                .map(VectorGammaConeDepthMeasurements::new)
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
            let mut candidates = GammaConeCandidates::default();
            candidates.rows.reserve(row_count);
            candidates.docs.reserve(row_count);
            candidates.raw_prefixes.reserve(row_count);
            candidates.estimates.reserve(row_count);
            candidates.sigmas.reserve(row_count);
            for cluster in 0..index.num_clusters() {
                let rows = index.cluster_range(cluster);
                let layer =
                    quantization.layers()[0].read_batch(rows.clone(), metric == Metric::L2)?;
                for row in rows {
                    let doc = self.doc_id_at(row);
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
                                    "gamma cone replicas for doc {doc} have different exact scores"
                                ))
                                .into());
                            }
                        }
                    }
                    let scale = layer.scale(row)?;
                    let constant = if metric == Metric::L2 {
                        layer.constant(row)?
                    } else {
                        0.0
                    };
                    let raw_prefix = query.score_layer(0, layer.code_bytes(row)?, scale, constant);
                    let gamma = gammas[0][row];
                    let residual_norm_squared = residual_norms_squared
                        .as_ref()
                        .map_or(0.0, |norms| norms[row]);
                    let estimate = match metric {
                        Metric::L2 => (2.0 * gamma)
                            .mul_add(raw_prefix, cluster_scores[cluster] - residual_norm_squared),
                        Metric::Dot | Metric::Cosine => {
                            gamma.mul_add(raw_prefix, cluster_scores[cluster])
                        }
                    };
                    let sign_scale = stored_scales[0][row];
                    let sign_query_error_term =
                        sign_scale * sign_scale * query.query_error_squared(0) as f32;
                    let query_norm = cluster_query_norms[cluster];
                    let sigma = gamma_production_sigma(
                        effective_scales_squared[0][row],
                        gamma,
                        query_norm * query_norm,
                        sign_query_error_term,
                        metric,
                    );
                    if !estimate.is_finite() || !sigma.is_finite() {
                        return Err(DataCorruption::comment_only(format!(
                            "gamma cone row {row} produced a non-finite depth-1 estimate or sigma"
                        ))
                        .into());
                    }
                    candidates.push(row, doc, raw_prefix, estimate, sigma);
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
                .take(GAMMA_CONE_TOP_K)
                .map(|(doc, _)| doc)
                .collect();
            debug_assert_eq!(exact_top_docs.len(), GAMMA_CONE_TOP_K);

            let depth_zero_scored = candidates.len();
            candidates.band(GAMMA_CONE_TOP_K, GAMMA_CONE_KAPPAS[0]);
            observe_gamma_cone_depth(
                &mut measurements.depths[0],
                depth_zero_scored,
                &candidates,
                &exact_top_docs,
            );

            let depth_one_scored = candidates.len();
            let mut cluster = 0usize;
            for candidate in 0..candidates.len() {
                let row = candidates.rows[candidate];
                while cluster < index.num_clusters() && index.cluster_range(cluster).end <= row {
                    cluster += 1;
                }
                if cluster == index.num_clusters() || !index.cluster_range(cluster).contains(&row) {
                    return Err(DataCorruption::comment_only(format!(
                        "gamma cone survivor row {row} is outside IVF cluster ranges"
                    ))
                    .into());
                }
                let layer = &quantization.layers()[1];
                let scale = layer.scale(row)?;
                let constant = if metric == Metric::L2 {
                    layer.constant(row)?
                } else {
                    0.0
                };
                candidates.raw_prefixes[candidate] +=
                    query.score_layer(1, &layer.code_bytes(row)?, scale, constant);
                let gamma = gammas[1][row];
                let residual_norm_squared = residual_norms_squared
                    .as_ref()
                    .map_or(0.0, |norms| norms[row]);
                candidates.estimates[candidate] = match metric {
                    Metric::L2 => (2.0 * gamma).mul_add(
                        candidates.raw_prefixes[candidate],
                        cluster_scores[cluster] - residual_norm_squared,
                    ),
                    Metric::Dot | Metric::Cosine => {
                        gamma.mul_add(candidates.raw_prefixes[candidate], cluster_scores[cluster])
                    }
                };
                let sign_scale = stored_scales[0][row];
                let mut sign_query_error_term =
                    sign_scale * sign_scale * query.query_error_squared(0) as f32;
                if measurement_ctx.specs[1].bits == 1 {
                    let scale = stored_scales[1][row];
                    sign_query_error_term += scale * scale * query.query_error_squared(1) as f32;
                }
                let query_norm = cluster_query_norms[cluster];
                let sigma = gamma_production_sigma(
                    effective_scales_squared[1][row],
                    gamma,
                    query_norm * query_norm,
                    sign_query_error_term,
                    metric,
                );
                candidates.sigmas[candidate] = sigma;
                if !candidates.estimates[candidate].is_finite() || !sigma.is_finite() {
                    return Err(DataCorruption::comment_only(format!(
                        "gamma cone row {row} produced a non-finite depth-2 estimate or sigma"
                    ))
                    .into());
                }
            }
            candidates.band(GAMMA_CONE_TOP_K, GAMMA_CONE_KAPPAS[1]);
            observe_gamma_cone_depth(
                &mut measurements.depths[1],
                depth_one_scored,
                &candidates,
                &exact_top_docs,
            );
        }
        Ok(Some(measurements))
    }

    /// Per-cluster posting-list sizes in cluster order — the distribution
    /// behind [`Self::info`]'s aggregate cluster stats. `None` when the
    /// field's storage is not IVF.
    pub fn cluster_sizes(&self) -> Option<Vec<u32>> {
        self.index
            .as_ref()
            .map(|index| index.cluster_sizes().map(|size| size as u32).collect())
    }

    /// Number of live IVF posting memberships. Replicated documents count
    /// once per membership; deleted documents contribute no memberships.
    pub fn live_posting_row_count(&self, alive: Option<&AliveBitSet>) -> usize {
        let Some(index) = &self.index else {
            return 0;
        };
        (0..index.num_rows())
            .filter(|&row| alive.is_none_or(|alive| alive.is_alive(self.doc_id_at(row))))
            .count()
    }

    /// `true` if `doc_id` has a stored vector.
    pub fn contains(&self, doc_id: DocId) -> bool {
        self.row_id(doc_id).is_some()
    }

    /// The raw little-endian bytes of `doc_id`'s vector, fetched with one
    /// stride-sized ranged read; `None` if the doc has no vector.
    pub fn vector_bytes(&self, doc_id: DocId) -> crate::Result<Option<OwnedBytes>> {
        let Some(row) = self.row_id(doc_id) else {
            return Ok(None);
        };
        self.vector_bytes_for_row(row).map(Some)
    }

    /// The raw bytes of the single vector row at `row` of the dense rows
    /// slot, fetched with one stride-sized ranged read
    /// (`row * stride..(row + 1) * stride`). The caller resolves `row`
    /// beforehand (e.g. from a cluster's row range), so no doc→row lookup
    /// happens here.
    pub fn vector_bytes_for_row(&self, row: usize) -> crate::Result<OwnedBytes> {
        if row >= self.id_map.num_rows() as usize {
            return Err(TantivyError::InvalidArgument(format!(
                "vector row {row} is out of bounds"
            )));
        }
        let stride = self.options.bytes_per_vector();
        let bytes = self
            .rows_slice
            .slice(row * stride..(row + 1) * stride)
            .read_bytes()?;
        Ok(bytes)
    }

    /// The doc id stored at `row` of the cluster-sorted permutation, decoded
    /// on demand from the pinned `Explicit` id-map. IVF storage only.
    #[inline]
    pub fn doc_id_at(&self, row: usize) -> DocId {
        let IdMap::Explicit(bytes) = &self.id_map else {
            unreachable!("doc_id_at is only meaningful for cluster-sorted (IVF) storage");
        };
        let start = row * std::mem::size_of::<DocId>();
        DocId::from_le_bytes(
            bytes[start..start + std::mem::size_of::<DocId>()]
                .try_into()
                .unwrap(),
        )
    }

    /// The doc ids assigned to `cluster`, ascending; `None` if the storage is
    /// not IVF or `cluster` is out of bounds.
    pub fn cluster_doc_ids(&self, cluster: usize) -> Option<Vec<DocId>> {
        let index = self.index.as_ref()?;
        if cluster >= index.num_clusters() {
            return None;
        }
        Some(
            index
                .cluster_range(cluster)
                .map(|row| self.doc_id_at(row))
                .collect(),
        )
    }

    /// Doc → dense row. For clustered storage, rows are cluster-sorted and
    /// ascending by doc id within each cluster, so this scans clusters and
    /// binary-searches each one over the pinned id-map bytes. For the flat
    /// id-maps (`Identity`/`Bitmap`) the mapping is strictly ascending in
    /// doc id — the property the exact path's run builder leans on.
    pub(crate) fn row_id(&self, doc_id: DocId) -> Option<usize> {
        match &self.id_map {
            IdMap::Identity { num_docs } => (doc_id < *num_docs).then_some(doc_id as usize),
            IdMap::Bitmap(_) => self.id_map.rank_if_exists(doc_id).map(|row| row as usize),
            IdMap::Explicit(_) => {
                let index = self.index.as_ref()?;
                for cluster in 0..index.num_clusters() {
                    let rows = index.cluster_range(cluster);
                    let mut lo = rows.start;
                    let mut hi = rows.end;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        match self.doc_id_at(mid).cmp(&doc_id) {
                            Ordering::Less => lo = mid + 1,
                            Ordering::Greater => hi = mid,
                            Ordering::Equal => return Some(mid),
                        }
                    }
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use quant_model::f16::f32_to_f16;

    use super::*;
    use crate::directory::{CompositeWrite, FileHandle};
    use crate::vector::header::write_header;

    #[derive(Debug)]
    struct BlockTrackedBytes {
        bytes: Vec<u8>,
        reads: Arc<Mutex<Vec<Range<usize>>>>,
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

    fn test_cluster_offsets(offsets: &[usize]) -> Arc<[usize]> {
        Arc::from(offsets)
    }

    fn test_sidecar(scales: &[u16], gammas: &[f32]) -> Vec<u8> {
        assert_eq!(scales.len(), gammas.len());
        scales
            .iter()
            .flat_map(|scale| scale.to_le_bytes())
            .chain(
                gammas
                    .iter()
                    .flat_map(|&gamma| f32_to_f16(gamma).to_le_bytes()),
            )
            .collect()
    }

    #[test]
    fn calibration_query_norm_matches_production_f32_chain() {
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
        let mut differs_from_the_retired_f64_chain = false;

        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let config = VectorQuantizationConfig::materialize(
                "embedding".to_string(),
                &VectorOptions::new(DIM, metric),
                vec![super::super::quantization::VectorQuantizationLayer {
                    bits: 1,
                    quantizer: super::super::quantization::VectorQuantizer::RaBitQ,
                    seed: 7,
                }],
            )
            .unwrap();
            let query = QuantizedQueryCtx::new(
                QuantizedIndexCtx::for_calibration_measurement(config).unwrap(),
                source_query.clone(),
            );
            let routing_score = metric
                .similarity_bytes::<f32>(query.query(), &centroid_bytes)
                .score();
            let expected = query.score_query_norm(routing_score);
            let actual = calibration_query_norm(&query, metric, &centroid_bytes);
            assert_eq!(actual.to_bits(), expected.to_bits(), "metric={metric:?}");

            let retired_norm = query
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
            differs_from_the_retired_f64_chain |=
                f64::from(actual).to_bits() != retired_norm.to_bits();
        }
        assert!(differs_from_the_retired_f64_chain);
    }

    #[test]
    fn gamma_one_variance_is_formula_native() {
        // At gamma=1 the data-side projection term is exactly zero. Query
        // quantization remains exactly the supplied sum(s_j^2 * B_j).
        assert_eq!(gamma_model_variance(0.25, 1.0, 3.0, 0.0), 0.0);
        assert_eq!(gamma_model_variance(0.25, 1.0, 3.0, 0.5), 0.5);
    }

    #[test]
    fn production_gamma_sigma_applies_safety_and_l2_factor_once() {
        let variance = gamma_model_variance(0.25, 2.0, 3.0, 0.5);
        let dot = gamma_production_sigma(0.25, 2.0, 3.0, 0.5, Metric::Dot);
        let l2 = gamma_production_sigma(0.25, 2.0, 3.0, 0.5, Metric::L2);
        let expected = GAMMA_ANALYTICAL_SAFETY * variance.sqrt() as f32;
        assert_eq!(dot.to_bits(), expected.to_bits());
        assert_eq!(l2.to_bits(), (2.0 * expected).to_bits());
    }

    #[test]
    fn effective_scale_uses_first_scale_and_persisted_gamma_ratio() {
        assert_eq!(gamma_effective_scale_squared(0.25, 2.0, 4.0), 0.125);
        assert_eq!(gamma_effective_scale_squared(0.25, 2.0, 2.0), 0.25);
    }

    #[test]
    fn stored_gamma_must_equal_the_deterministic_encoder_receipt() {
        verify_stored_gamma(17, 1, 1.5, 1.5).unwrap();
        let error = verify_stored_gamma(17, 1, 1.5, 1.75).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("row 17 layer 1"));
        assert!(message.contains("stored gamma differs"));
    }

    #[test]
    fn held_out_exclusion_covers_every_replica_membership() {
        let queries = vec![
            VectorGammaAuditQuery {
                values: vec![1.0],
                excluded_doc_id: Some(7),
            },
            VectorGammaAuditQuery {
                values: vec![2.0],
                excluded_doc_id: None,
            },
            VectorGammaAuditQuery {
                values: vec![3.0],
                excluded_doc_id: Some(7),
            },
        ];
        let excluded = excluded_membership_clusters(
            &queries,
            [(0, 7), (0, 9), (1, 8), (2, 7), (2, 7), (3, 10)],
        );
        assert_eq!(excluded[0], vec![0, 2]);
        assert!(excluded[1].is_empty());
        assert_eq!(excluded[2], vec![0, 2]);
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
    fn gamma_cone_boundary_dedups_only_the_pivot_selection() {
        let mut candidates = GammaConeCandidates::default();
        // Doc 7's better replica selects its distinct-doc rank, but both
        // memberships remain independently eligible for the survivor set.
        candidates.push(0, 7, 0.0, 10.0, 0.0);
        candidates.push(1, 7, 0.0, 12.0, 0.0);
        candidates.push(2, 8, 0.0, 9.0, 0.0);
        candidates.push(3, 9, 0.0, 8.0, 0.0);
        candidates.band(2, 0.0);
        assert_eq!(candidates.rows, vec![0, 1, 2]);
        assert_eq!(candidates.docs, vec![7, 7, 8]);
    }

    #[test]
    fn gamma_cone_boundary_includes_optimistic_equality() {
        let mut candidates = GammaConeCandidates::default();
        candidates.push(0, 1, 0.0, 10.0, 0.0);
        candidates.push(1, 2, 0.0, 8.0, 1.0);
        candidates.push(2, 3, 0.0, 4.0, 1.0);
        // Pivot doc 2 gives T=8-2*1=6. Doc 3's upper endpoint equals 6,
        // and therefore survives the inclusive production comparison.
        candidates.band(2, 2.0);
        assert_eq!(candidates.rows, vec![0, 1, 2]);
    }

    #[test]
    fn quantized_layer_reader_rejects_non_zero_tail() -> crate::Result<()> {
        let mut codes = vec![0_u8; quantized_code_stride(65, 1)];
        codes[8] = 1;
        let valid = QuantizedLayerReader {
            codes: FileSlice::from(codes.clone()),
            sidecar: FileSlice::empty(),
            constants: FileSlice::empty(),
            cluster_offsets: test_cluster_offsets(&[0, 1]),
            code_stride: codes.len(),
            dim: 65,
            bits: 1,
        };
        assert_eq!(valid.code_bytes(0)?.as_slice(), codes);

        codes[8] |= 2;
        let corrupt = QuantizedLayerReader {
            codes: FileSlice::from(codes.clone()),
            sidecar: FileSlice::empty(),
            constants: FileSlice::empty(),
            cluster_offsets: test_cluster_offsets(&[0, 1]),
            code_stride: codes.len(),
            dim: 65,
            bits: 1,
        };
        assert!(corrupt.code_bytes(0).is_err());
        Ok(())
    }

    #[test]
    fn indexed_range_validates_unselected_gap_row_tail() {
        let stride = quantized_code_stride(65, 1);
        let mut codes = vec![0_u8; stride * 3];
        codes[8] = 1;
        codes[stride + 8] = 0b10;
        codes[stride * 2 + 8] = 1;
        let reader = QuantizedLayerReader {
            codes: FileSlice::from(codes),
            sidecar: FileSlice::from(test_sidecar(&[0; 3], &[1.0; 3])),
            constants: FileSlice::empty(),
            cluster_offsets: test_cluster_offsets(&[0, 3]),
            code_stride: stride,
            dim: 65,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();
        reader.plan_code_reads(0..3, &[0, 2], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..3]);
        assert!(
            reader.read_codes(ranges.pop().unwrap()).is_err(),
            "the unselected corrupt row inside the pinned range must be checked"
        );
    }

    #[test]
    fn quantized_layer_reader_pins_and_decodes_a_contiguous_batch() -> crate::Result<()> {
        let stride = quantized_code_stride(65, 1);
        let mut codes = vec![0_u8; stride * 2];
        codes[8] = 1;
        codes[stride + 8] = 1;
        let sidecar = test_sidecar(&[17, 29], &[1.0, 4.0]);
        let constants = [0.25_f32, -0.75_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let reader = QuantizedLayerReader {
            codes: FileSlice::from(codes.clone()),
            sidecar: FileSlice::from(sidecar),
            constants: FileSlice::from(constants),
            cluster_offsets: test_cluster_offsets(&[0, 2]),
            code_stride: stride,
            dim: 65,
            bits: 1,
        };
        let batch = reader.read_batch(0..2, true)?;
        assert_eq!(batch.code_bytes(0)?, &codes[..stride]);
        assert_eq!(batch.code_bytes(1)?, &codes[stride..]);
        assert_eq!(batch.scale(0)?, 17);
        assert_eq!(batch.scale(1)?, 29);
        assert_eq!(batch.gamma(0)?, 1.0);
        assert_eq!(batch.gamma(1)?, 4.0);
        assert_eq!(batch.scales().len(), 2 * QUANTIZED_SCALE_STRIDE);
        assert_eq!(batch.gammas().len(), 2 * QUANTIZED_GAMMA_STRIDE);
        assert_eq!(batch.constant(0)?.to_bits(), 0.25_f32.to_bits());
        assert_eq!(batch.constant(1)?.to_bits(), (-0.75_f32).to_bits());
        let dot_batch = reader.read_batch(0..2, false)?;
        assert!(dot_batch.constants().is_empty());

        codes[stride + 8] |= 2;
        let corrupt = QuantizedLayerReader {
            codes: FileSlice::from(codes),
            sidecar: FileSlice::from(test_sidecar(&[0, 0], &[1.0, 0.5])),
            constants: FileSlice::from(vec![0_u8; 8]),
            cluster_offsets: test_cluster_offsets(&[0, 2]),
            code_stride: stride,
            dim: 65,
            bits: 1,
        };
        assert!(corrupt.read_batch(0..2, true).is_err());
        Ok(())
    }

    #[test]
    fn cluster_blocked_sidecar_maps_scales_and_gammas_by_cluster() -> crate::Result<()> {
        let mut sidecar = test_sidecar(&[11, 12], &[1.0, 2.0]);
        sidecar.extend(test_sidecar(&[13], &[3.0]));
        let reader = QuantizedLayerReader {
            codes: FileSlice::empty(),
            sidecar: FileSlice::from(sidecar),
            constants: FileSlice::empty(),
            cluster_offsets: test_cluster_offsets(&[0, 2, 3]),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };

        let first = reader.read_scale_gamma(0..2)?;
        assert_eq!(first.scale(0)?, 11);
        assert_eq!(first.scale(1)?, 12);
        assert_eq!(first.gamma(0)?, 1.0);
        assert_eq!(first.gamma(1)?, 2.0);
        let second = reader.read_scale_gamma(2..3)?;
        assert_eq!(second.scale(2)?, 13);
        assert_eq!(second.gamma(2)?, 3.0);
        assert_eq!(reader.scale(2)?, 13);
        assert_eq!(reader.gamma(2)?, 3.0);
        Ok(())
    }

    #[test]
    fn sidecar_gamma_validation_is_range_scoped() {
        for gamma in [0.5, 5.0, f32::INFINITY, f32::NAN] {
            let reader = QuantizedLayerReader {
                codes: FileSlice::empty(),
                sidecar: FileSlice::from(test_sidecar(&[17], &[gamma])),
                constants: FileSlice::empty(),
                cluster_offsets: test_cluster_offsets(&[0, 1]),
                code_stride: 8,
                dim: 64,
                bits: 1,
            };
            assert!(reader.read_scale_gamma(0..1).is_err(), "gamma={gamma}");
        }
    }

    #[test]
    fn indexed_sidecar_plan_is_sparse_across_both_runs() -> crate::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let storage = Arc::new(BlockTrackedBytes {
            bytes: test_sidecar(&[0; 8], &[1.0; 8]),
            reads: Arc::clone(&reads),
            block_len: 8,
        });
        let reader = QuantizedLayerReader {
            codes: FileSlice::empty(),
            sidecar: FileSlice::new(storage),
            constants: FileSlice::empty(),
            cluster_offsets: test_cluster_offsets(&[0, 8]),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();

        reader.plan_sidecar_reads(0..8, &[1, 2], &mut ranges, &mut blocks);
        assert_eq!(ranges, [1..3]);
        let sparse = reader.read_scale_gamma(ranges.pop().unwrap())?;
        assert_eq!(sparse.scales().len(), 2 * QUANTIZED_SCALE_STRIDE);
        assert_eq!(sparse.gammas().len(), 2 * QUANTIZED_GAMMA_STRIDE);
        assert_eq!(&*reads.lock().unwrap(), &[2..6, 18..22]);

        reads.lock().unwrap().clear();
        reader.plan_sidecar_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        reader.read_scale_gamma(ranges.pop().unwrap())?;
        assert_eq!(&*reads.lock().unwrap(), &[0..32]);
        Ok(())
    }

    #[test]
    fn indexed_sidecar_plan_splits_cross_cluster_cosine_batches() -> crate::Result<()> {
        let mut sidecar = test_sidecar(&[10, 11, 12], &[1.0, 1.5, 2.0]);
        sidecar.extend(test_sidecar(&[20, 21, 22], &[2.5, 3.0, 3.5]));
        let reader = QuantizedLayerReader {
            codes: FileSlice::empty(),
            sidecar: FileSlice::from(sidecar),
            constants: FileSlice::empty(),
            cluster_offsets: test_cluster_offsets(&[0, 3, 6]),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();

        reader.plan_sidecar_reads(1..5, &[1, 4], &mut ranges, &mut blocks);
        assert_eq!(ranges, [1..3, 3..5]);
        let first = reader.read_scale_gamma(ranges[0].clone())?;
        let second = reader.read_scale_gamma(ranges[1].clone())?;
        assert_eq!(first.scale(1)?, 11);
        assert_eq!(first.gamma(1)?, 1.5);
        assert_eq!(second.scale(4)?, 21);
        assert_eq!(second.gamma(4)?, 3.0);
        Ok(())
    }

    #[test]
    fn indexed_read_plan_is_density_adaptive_across_soa_slots() -> crate::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut storage_bytes = vec![0_u8; 176];
        storage_bytes[80..112].copy_from_slice(&test_sidecar(&[0; 8], &[1.0; 8]));
        let storage = Arc::new(BlockTrackedBytes {
            bytes: storage_bytes,
            reads: Arc::clone(&reads),
            block_len: 16,
        });
        let codes = FileSlice::new(storage.clone()).slice(15..79);
        let sidecar = FileSlice::new(storage.clone()).slice(80..112);
        let constants = FileSlice::new(storage).slice(112..144);
        let reader = QuantizedLayerReader {
            codes,
            sidecar,
            constants,
            cluster_offsets: test_cluster_offsets(&[0, 8]),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();

        reader.plan_code_reads(0..8, &[1, 3], &mut ranges, &mut blocks);
        assert_eq!(ranges, [1..2, 3..4], "adjacent pages stay separate");
        reader.plan_code_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..1, 7..8]);
        for range in ranges.drain(..) {
            reader.read_codes(range)?;
        }
        reader.plan_scale_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        reader.read_scales(ranges.pop().unwrap())?;
        reader.plan_constant_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        reader.read_constants(ranges.pop().unwrap())?;
        let reads = reads.lock().unwrap();
        assert_eq!(&*reads, &[15..23, 71..79, 80..112, 112..144]);
        let mut pinned_blocks = reads
            .iter()
            .flat_map(|range| range.start / 16..=(range.end - 1) / 16)
            .collect::<Vec<_>>();
        let pin_count = pinned_blocks.len();
        pinned_blocks.sort_unstable();
        pinned_blocks.dedup();
        assert_eq!(
            pinned_blocks.len(),
            pin_count,
            "each physical block must be pinned once per indexed batch"
        );
        drop(reads);

        reader.plan_code_reads(0..8, &(0..8).collect::<Vec<_>>(), &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        Ok(())
    }

    #[test]
    fn indexed_read_plan_without_storage_geometry_pins_available_range() {
        let reader = QuantizedLayerReader {
            codes: FileSlice::from(vec![0_u8; 8 * 8]),
            sidecar: FileSlice::from(test_sidecar(&[0; 8], &[1.0; 8])),
            constants: FileSlice::empty(),
            cluster_offsets: test_cluster_offsets(&[0, 8]),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();
        reader.plan_code_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        reader.plan_scale_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
    }

    #[test]
    fn v3_maximal_trailers_are_trimmed_from_logical_arrays() -> crate::Result<()> {
        let field = Field::from_field_id(0);
        let mut bytes = Vec::new();
        write_header(&mut bytes)?;
        {
            let mut writer = CompositeWrite::wrap(&mut bytes);
            let rows = writer.for_field_with_idx(field, vec_slot::ROWS);
            rows.write_all(&[1, 2, 3, 4])?;
            rows.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(1), 0)
                .write_all(&[9])?;

            let sidecar = writer.for_field_with_idx(field, vec_slot::quantized_scales(0));
            sidecar.write_all(&17_u16.to_le_bytes())?;
            sidecar.write_all(&f32_to_f16(1.5).to_le_bytes())?;
            sidecar.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(2), 0)
                .write_all(&[9])?;

            let constants = writer.for_field_with_idx(field, vec_slot::quantized_constants(0));
            constants.write_all(&1.25_f32.to_le_bytes())?;
            constants.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(3), 0)
                .write_all(&[9])?;

            let norms = writer.for_field_with_idx(field, vec_slot::QUANTIZED_RESIDUAL_NORMS);
            norms.write_all(&2.5_f32.to_le_bytes())?;
            norms.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(4), 0)
                .write_all(&[9])?;
            writer.close()?;
        }

        let (version, body) = read_header(&FileSlice::from(bytes))?;
        assert_eq!(version, VectorFileVersion::V3);
        let composite = CompositeFile::open(&body)?;

        let rows = logical_slice(
            composite.open_read_with_idx(field, vec_slot::ROWS).unwrap(),
            4,
            "rows fixture",
        )?
        .read_bytes()?;
        let sidecar = logical_slice(
            composite
                .open_read_with_idx(field, vec_slot::quantized_scales(0))
                .unwrap(),
            QUANTIZED_SCALE_GAMMA_STRIDE,
            "scale/gamma fixture",
        )?
        .read_bytes()?;
        let constants = logical_slice(
            composite
                .open_read_with_idx(field, vec_slot::quantized_constants(0))
                .unwrap(),
            4,
            "constants fixture",
        )?
        .read_bytes()?;
        let norms = logical_slice(
            composite
                .open_read_with_idx(field, vec_slot::QUANTIZED_RESIDUAL_NORMS)
                .unwrap(),
            4,
            "residual norms fixture",
        )?
        .read_bytes()?;

        assert_eq!(rows.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(
            u16::from_le_bytes(sidecar[..QUANTIZED_SCALE_STRIDE].try_into().unwrap()),
            17
        );
        assert_eq!(
            f16_to_f32(u16::from_le_bytes(
                sidecar[QUANTIZED_SCALE_STRIDE..].try_into().unwrap()
            )),
            1.5
        );
        assert_eq!(
            f32::from_le_bytes(constants.as_slice().try_into().unwrap()),
            1.25
        );
        assert_eq!(
            f32::from_le_bytes(norms.as_slice().try_into().unwrap()),
            2.5
        );
        Ok(())
    }

    #[test]
    fn nonzero_alignment_trailer_is_corruption() {
        let slice = FileSlice::from(vec![1, 2, 3, 4, 0, 7]);
        assert!(logical_slice(slice, 4, "bad fixture").is_err());
    }
}
