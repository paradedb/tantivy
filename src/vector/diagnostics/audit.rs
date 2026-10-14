//! Estimator and audit measurement over a segment's quantized storage.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::error::DataCorruption;
use crate::fastfield::AliveBitSet;
use crate::schema::Metric;
use crate::vector::clustering::{decode_row, decode_stored_row};
use crate::vector::diagnostics::{
    VectorErrorAuditMeasurements, VectorErrorConeAuditMeasurements,
    VectorErrorConeDepthMeasurements, VectorErrorDepthMeasurements, VectorEstimatorMeasurements,
    VectorEstimatorMoments, VectorEstimatorQuery, VectorEstimatorSource,
};
use crate::vector::quantization::{
    corrected_quantized_estimate, initial_dot_raw_prefix, initial_l2_raw_prefix,
    quantized_model_sigma, refine_dot_raw_prefix, refine_l2_raw_prefix, ArithmeticError,
    QUANTIZED_BOUNDARY_KAPPA,
};
use crate::vector::search::interval::{Estimate, Threshold};
use crate::vector::search::prepared::{QuantizedQueryCtx, VectorQuery};
use crate::vector::storage::metadata::QuantizationSchedule;
use crate::vector::storage::reader::VectorIndexReader;
use crate::{DocId, TantivyError};

const ERROR_CONE_QUERY_COUNT: usize = 100;
const ERROR_CONE_TOP_K: usize = 10;

#[inline]
fn diagnostic_query_norm(
    quantized: &QuantizedQueryCtx,
    query: &VectorQuery<f32>,
    metric: Metric,
    centroid_bytes: &[u8],
) -> f32 {
    let centroid_score = metric
        .similarity_bytes::<f32>(quantized.estimate_vector(query), centroid_bytes)
        .score();
    quantized.score_query_norm(centroid_score)
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

impl VectorIndexReader {
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
            let values = decode_stored_row(&self.vector_bytes_for_row(row)?, &self.options)?;
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
                schedule: QuantizationSchedule::new(
                    quantization
                        .index_ctx()
                        .meta
                        .layers()
                        .iter()
                        .map(|q| (q.kind(), q.bits()))
                        .collect(),
                ),
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
        let exact_queries: Vec<VectorQuery<f32>> = queries
            .iter()
            .map(|query| VectorQuery::for_options(&self.options, Arc::new(query.values.clone())))
            .collect::<crate::Result<_>>()?;
        let prepared_queries: Vec<QuantizedQueryCtx> = exact_queries
            .iter()
            .map(|query| QuantizedQueryCtx::new(Arc::clone(&measurement_ctx), query))
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
        let centroid_stride = self.options.dim() * std::mem::size_of::<f32>();
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
                let values = decode_stored_row(&vector_bytes, &self.options)?;
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

                let regenerated = quant::cascade::audit_prefix_error_model(
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
                        .similarity_bytes::<f32>(
                            query.estimate_vector(&exact_queries[query_idx]),
                            centroid_row,
                        )
                        .score();
                    let query_norm = query.score_query_norm(cluster_score);
                    let query_norm_squared = query_norm * query_norm;
                    let base = if self.options.metric() == Metric::L2 {
                        cluster_score - residual_norm_squared
                    } else {
                        cluster_score
                    };
                    let exact_score = exact_queries[query_idx].score_row(&vector_bytes).score();
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
                        if matches!(
                            measurement_ctx.specs[depth].kind,
                            quant::cascade::LayerKind::Sign
                        ) {
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
        let centroid_stride = self.options.dim() * std::mem::size_of::<f32>();
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
            let exact_query = VectorQuery::<f32>::for_options(
                &self.options,
                Arc::new(query_input.values.clone()),
            )?;
            let query = QuantizedQueryCtx::new(Arc::clone(&measurement_ctx), &exact_query);
            let mut cluster_scores = Vec::with_capacity(index.num_clusters());
            let mut cluster_query_norms = Vec::with_capacity(index.num_clusters());
            for cluster in 0..index.num_clusters() {
                let centroid_row = &centroid_bytes[cluster * centroid_stride..][..centroid_stride];
                let score = metric
                    .similarity_bytes::<f32>(query.estimate_vector(&exact_query), centroid_row)
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
                    let exact_score = exact_query.score_row(&bytes).score();
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
                    let sign_query_error_term = if matches!(
                        measurement_ctx.specs[0].kind,
                        quant::cascade::LayerKind::Sign
                    ) {
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
                        if matches!(
                            measurement_ctx.specs[depth].kind,
                            quant::cascade::LayerKind::Sign
                        ) {
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::schema::VectorOptions;
    use crate::vector::quantization::VectorQuantizationConfig;
    use crate::vector::storage::metadata::QuantizerKind;
    use crate::vector::storage::quantized::QuantizedIndexCtx;

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
                vec![crate::vector::quantization::VectorQuantizationLayer { bits: 1, seed: 7 }],
            )
            .unwrap();
            let vector_query = VectorQuery::new(metric, Arc::new(source_query.clone()));
            let query = QuantizedQueryCtx::new(
                Arc::new(QuantizedIndexCtx::from_config(config).unwrap()),
                &vector_query,
            );
            let centroid_score = metric
                .similarity_bytes::<f32>(query.estimate_vector(&vector_query), &centroid_bytes)
                .score();
            let expected = query.score_query_norm(centroid_score);
            let actual = diagnostic_query_norm(&query, &vector_query, metric, &centroid_bytes);
            assert_eq!(actual.to_bits(), expected.to_bits(), "metric={metric:?}");

            let f64_reference = query
                .estimate_vector(&vector_query)
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
        let expected = crate::vector::quantization::GAMMA_ANALYTICAL_SAFETY * variance.sqrt();
        assert_eq!(dot.to_bits(), expected.to_bits());
        assert_eq!(l2.to_bits(), (2.0 * expected).to_bits());
    }

    #[test]
    fn exact_e_sigma_supports_a_grid_first_prefix() {
        let sigma = quantized_model_sigma(Metric::Cosine, 64, 4.0, 0.5, 1.25, 2.0, 0.0);
        let expected = crate::vector::quantization::GAMMA_ANALYTICAL_SAFETY
            * (4.0_f32 / 64.0 * 0.5 * 2.0).sqrt();
        assert_eq!(sigma.to_bits(), expected.to_bits());
    }

    #[test]
    fn estimator_error_sign_is_estimate_minus_exact() {
        let mut measurements = VectorEstimatorMeasurements {
            source: VectorEstimatorSource::Provided,
            schedule: QuantizationSchedule::new(vec![(QuantizerKind::Sign, 1)]),
            aggregate: vec![VectorEstimatorMoments::default()],
            per_query: vec![vec![VectorEstimatorMoments::default()]],
            sample_rows: 1,
            query_count: 1,
        };
        observe_corrected_prefix(&mut measurements, 0, 0, 1.0, 3.0, 2.0);
        assert_eq!(measurements.aggregate()[0].bias(), Some(1.0));
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
}
