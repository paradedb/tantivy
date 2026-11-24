//! Vector storage diagnostics: segment storage info and estimator and audit measurements.

use crate::vector::router::RouterKind;
use crate::vector::storage::metadata::{QuantizationSchedule, VectorColMetadata};
use crate::vector::storage::reader::VectorIndexReader;
use crate::{DocId, TantivyError};

pub(crate) mod audit;
#[cfg(test)]
mod tests;

/// The on-disk layout of a segment's vector rows, reported in [`VectorInfo`].
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
    pub num_vectors: usize,
    /// Number of IVF centroids.
    pub num_centroids: Option<usize>,
    /// IVF posting-size statistics.
    pub cluster_stats: Option<VectorClusterStats>,
    /// Router that ranks the IVF centroids.
    pub router: Option<RouterKind>,
    /// Centroids at each router level, bottom to top; empty for the exact router.
    pub router_levels: Option<Vec<usize>>,
    /// Bits per stored quantized layer, in scoring order; `None` when the rows are stored plain.
    pub quantized_layers: Option<Vec<u8>>,
    /// Quantized bytes stored per row, every layer's columns together; `None` when plain.
    pub quantized_bytes_per_row: Option<usize>,
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
    schedule: QuantizationSchedule,
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

impl VectorErrorAuditMeasurements {
    /// Ordered quantizer kinds and widths, independent of rotation seeds.
    #[cfg(test)]
    pub(crate) fn schedule(&self) -> &QuantizationSchedule {
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

impl VectorEstimatorMeasurements {
    /// Ordered quantizer kinds and widths, independent of rotation seeds.
    #[cfg(test)]
    pub(crate) fn schedule(&self) -> &QuantizationSchedule {
        &self.schedule
    }

    fn check_schedule(&self, other: &Self) -> crate::Result<()> {
        if self.schedule != other.schedule {
            return Err(TantivyError::InvalidArgument(format!(
                "cannot merge vector measurements with different schedules: {} and {}",
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

impl VectorIndexReader {
    /// Storage information for tooling; `None` when the segment has no vector data for the field.
    pub fn info(&self) -> Option<VectorInfo> {
        let meta = self.metadata()?;
        let (quantized_layers, quantized_bytes_per_row) = match meta {
            VectorColMetadata::Quantized { .. } => (
                Some(meta.layers().iter().map(|layer| layer.bits()).collect()),
                meta.quantized_bytes_per_row(),
            ),
            VectorColMetadata::Plain(_) => (None, None),
        };
        let Some(index) = &self.index else {
            return Some(VectorInfo {
                format: VectorStorageFormat::Flat,
                num_vectors: self.num_vectors,
                num_centroids: None,
                cluster_stats: None,
                router: None,
                router_levels: None,
                quantized_layers,
                quantized_bytes_per_row,
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
            router: Some(index.router()),
            router_levels: Some(index.router_levels()),
            quantized_layers,
            quantized_bytes_per_row,
        })
    }
}
