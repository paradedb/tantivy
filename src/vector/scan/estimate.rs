//! Combining decoded layers into estimates and sigmas.

use std::ops::Range;

use crate::schema::Metric;
use crate::vector::prepared::{
    corrected_quantized_estimate, initial_dot_raw_prefix, initial_l2_raw_prefix,
    quantized_model_sigma, refine_dot_raw_prefix, refine_l2_raw_prefix, ArithmeticError,
};
use crate::vector::scan::columns::QuantizedCandidates;
#[cfg(feature = "unstable")]
use crate::vector::scan::layer_read::{decode_f16s, decode_f32s};

/// Computes corrected-reconstruction uncertainty.
#[inline(always)]
fn fill_gamma_sigmas(
    sigmas: &mut [f32],
    residual_norms_squared: &[f32],
    gammas: &[f32],
    error_ratios: &[f32],
    sign_query_error_terms: &[f32],
    score_query_norm_squared: f32,
    dimension: usize,
    metric: Metric,
) {
    debug_assert_eq!(sigmas.len(), residual_norms_squared.len());
    debug_assert_eq!(sigmas.len(), gammas.len());
    debug_assert_eq!(sigmas.len(), error_ratios.len());
    debug_assert_eq!(sigmas.len(), sign_query_error_terms.len());
    for ((((sigma, &residual_norm_squared), &gamma), &error_ratio), &sign_query_error_term) in
        sigmas
            .iter_mut()
            .zip(residual_norms_squared)
            .zip(gammas)
            .zip(error_ratios)
            .zip(sign_query_error_terms)
    {
        *sigma = quantized_model_sigma(
            metric,
            dimension,
            residual_norm_squared,
            error_ratio,
            gamma,
            score_query_norm_squared,
            sign_query_error_term,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(in crate::vector) fn combine_initial_decoded(
    metric: Metric,
    dimension: usize,
    kernel_scores: &mut [f32],
    bases: &mut [f32],
    estimates: &mut [f32],
    sigmas: &mut [f32],
    residual_norms_squared: &mut [f32],
    sign_query_error_terms: &mut [f32],
    arithmetic_variances: &mut [ArithmeticError],
    decoded_scales: &[f32],
    decoded_gammas: &[f32],
    decoded_error_ratios: &[f32],
    decoded_constants: &[f32],
    decoded_residual_norms: &[f32],
    cluster_score: f32,
    score_query_norm_squared: f32,
    sign_query_error_squared: f32,
) {
    debug_assert_eq!(kernel_scores.len(), decoded_scales.len());
    debug_assert_eq!(kernel_scores.len(), decoded_gammas.len());
    debug_assert_eq!(kernel_scores.len(), decoded_error_ratios.len());
    debug_assert_eq!(kernel_scores.len(), decoded_residual_norms.len());
    debug_assert_eq!(bases.len(), decoded_scales.len());
    debug_assert_eq!(estimates.len(), decoded_scales.len());
    debug_assert_eq!(sigmas.len(), decoded_scales.len());
    debug_assert_eq!(residual_norms_squared.len(), decoded_scales.len());
    debug_assert_eq!(sign_query_error_terms.len(), decoded_scales.len());
    debug_assert_eq!(arithmetic_variances.len(), decoded_scales.len());
    for (index, error) in arithmetic_variances.iter_mut().enumerate() {
        *error = ArithmeticError::initial(
            metric,
            kernel_scores[index],
            decoded_scales[index],
            decoded_constants.get(index).copied().unwrap_or(0.0),
            cluster_score,
            decoded_residual_norms[index],
        );
    }
    match metric {
        Metric::L2 => {
            debug_assert_eq!(decoded_constants.len(), decoded_scales.len());
            for (
                (
                    (
                        (
                            (((raw_prefix, base), estimate), residual_norm_squared),
                            sign_query_error_term,
                        ),
                        &scale,
                    ),
                    &gamma,
                ),
                (&constant, &residual_norm_sq),
            ) in kernel_scores
                .iter_mut()
                .zip(bases.iter_mut())
                .zip(estimates.iter_mut())
                .zip(residual_norms_squared.iter_mut())
                .zip(sign_query_error_terms.iter_mut())
                .zip(decoded_scales)
                .zip(decoded_gammas)
                .zip(decoded_constants.iter().zip(decoded_residual_norms))
            {
                *raw_prefix = initial_l2_raw_prefix(*raw_prefix, scale, constant);
                *base = cluster_score - residual_norm_sq;
                *estimate = corrected_quantized_estimate(metric, gamma, *raw_prefix, *base);
                *residual_norm_squared = residual_norm_sq;
                *sign_query_error_term = scale * scale * sign_query_error_squared;
            }
        }
        Metric::Dot | Metric::Cosine => {
            for (
                (
                    (
                        (
                            (((raw_prefix, base), estimate), residual_norm_squared),
                            sign_query_error_term,
                        ),
                        &scale,
                    ),
                    &gamma,
                ),
                &residual_norm_sq,
            ) in kernel_scores
                .iter_mut()
                .zip(bases.iter_mut())
                .zip(estimates.iter_mut())
                .zip(residual_norms_squared.iter_mut())
                .zip(sign_query_error_terms.iter_mut())
                .zip(decoded_scales)
                .zip(decoded_gammas)
                .zip(decoded_residual_norms)
            {
                *raw_prefix = initial_dot_raw_prefix(*raw_prefix, scale);
                *base = cluster_score;
                *estimate = corrected_quantized_estimate(metric, gamma, *raw_prefix, *base);
                *residual_norm_squared = residual_norm_sq;
                *sign_query_error_term = scale * scale * sign_query_error_squared;
            }
        }
    }
    fill_gamma_sigmas(
        sigmas,
        residual_norms_squared,
        decoded_gammas,
        decoded_error_ratios,
        sign_query_error_terms,
        score_query_norm_squared,
        dimension,
        metric,
    );
    for (index, sigma) in sigmas.iter_mut().enumerate() {
        *sigma = arithmetic_variances[index].sigma(
            metric,
            *sigma,
            decoded_gammas[index],
            kernel_scores[index],
            bases[index],
        );
    }
    debug_assert!(estimates.iter().all(|e| e.is_finite()) && sigmas.iter().all(|s| s.is_finite()));
}

/// Runs the complete layer-0 cluster scoring shape.
#[cfg(feature = "unstable")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn quantization_bench_layer0_cosine_cluster(
    dimension: usize,
    prepared: &cascade::PreparedSplitQuery,
    spec: cascade::LayerSpec,
    codes: &[u8],
    code_stride: usize,
    scales: &[u8],
    gammas: &[u8],
    error_ratios: &[u8],
    residual_norms: &[u8],
    cluster_score: f32,
    score_query_norm_squared: f32,
    sign_query_error_squared: f32,
    kernel_scores: &mut Vec<f32>,
    decoded_scales: &mut Vec<f32>,
    decoded_gammas: &mut Vec<f32>,
    decoded_error_ratios: &mut Vec<f32>,
    decoded_residual_norms: &mut Vec<f32>,
    bases: &mut Vec<f32>,
    estimates: &mut Vec<f32>,
    sigmas: &mut Vec<f32>,
    residual_norms_squared: &mut Vec<f32>,
    sign_query_error_terms: &mut Vec<f32>,
) -> f32 {
    let rows = scales.len() / std::mem::size_of::<f32>();
    assert_eq!(codes.len(), rows * code_stride);
    assert_eq!(gammas.len(), rows * std::mem::size_of::<u16>());
    assert_eq!(error_ratios.len(), rows * std::mem::size_of::<u16>());
    assert_eq!(residual_norms.len(), rows * std::mem::size_of::<f32>());
    kernel_scores.resize(rows, 0.0);
    prepared.score_layer_batch_unscaled(0, codes, code_stride, spec, kernel_scores);
    decode_f32s(scales, decoded_scales);
    finish_quantization_bench_layer0_cosine_cluster(
        dimension,
        rows,
        gammas,
        error_ratios,
        residual_norms,
        cluster_score,
        score_query_norm_squared,
        sign_query_error_squared,
        kernel_scores,
        decoded_scales,
        decoded_gammas,
        decoded_error_ratios,
        decoded_residual_norms,
        bases,
        estimates,
        sigmas,
        residual_norms_squared,
        sign_query_error_terms,
    )
}

#[cfg(feature = "unstable")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn finish_quantization_bench_layer0_cosine_cluster(
    dimension: usize,
    rows: usize,
    gammas: &[u8],
    error_ratios: &[u8],
    residual_norms: &[u8],
    cluster_score: f32,
    score_query_norm_squared: f32,
    sign_query_error_squared: f32,
    kernel_scores: &mut [f32],
    decoded_scales: &mut [f32],
    decoded_gammas: &mut Vec<f32>,
    decoded_error_ratios: &mut Vec<f32>,
    decoded_residual_norms: &mut Vec<f32>,
    bases: &mut Vec<f32>,
    estimates: &mut Vec<f32>,
    sigmas: &mut Vec<f32>,
    residual_norms_squared: &mut Vec<f32>,
    sign_query_error_terms: &mut Vec<f32>,
) -> f32 {
    decode_f16s(gammas, decoded_gammas);
    decode_f16s(error_ratios, decoded_error_ratios);
    decode_f32s(residual_norms, decoded_residual_norms);
    bases.resize(rows, 0.0);
    estimates.resize(rows, 0.0);
    sigmas.resize(rows, 0.0);
    residual_norms_squared.resize(rows, 0.0);
    sign_query_error_terms.resize(rows, 0.0);
    let mut arithmetic_variances = vec![ArithmeticError::default(); rows];
    combine_initial_decoded(
        Metric::Cosine,
        dimension,
        kernel_scores,
        bases,
        estimates,
        sigmas,
        residual_norms_squared,
        sign_query_error_terms,
        &mut arithmetic_variances,
        decoded_scales,
        decoded_gammas,
        decoded_error_ratios,
        &[],
        decoded_residual_norms,
        cluster_score,
        score_query_norm_squared,
        sign_query_error_squared,
    );

    kernel_scores[rows - 1] + estimates[rows - 1] + sigmas[rows - 1]
}

const COSINE_REFINEMENT_BATCH_ROWS: usize = 2_048;

pub(in crate::vector) fn cosine_refinement_batches(
    row_count: usize,
) -> impl Iterator<Item = Range<usize>> {
    (0..row_count)
        .step_by(COSINE_REFINEMENT_BATCH_ROWS)
        .map(move |start| start..(start + COSINE_REFINEMENT_BATCH_ROWS).min(row_count))
}

#[inline(always)]
pub(in crate::vector) fn combine_refinement_decoded(
    metric: Metric,
    dimension: usize,
    candidates: &mut QuantizedCandidates,
    candidate_range: Range<usize>,
    kernel_scores: &[f32],
    decoded_scales: &[f32],
    decoded_gammas: &[f32],
    decoded_error_ratios: &[f32],
    decoded_constants: &[f32],
    score_query_norm_squared: f32,
    sign_query_error_squared: f32,
) {
    let rows = candidate_range.len();
    debug_assert_eq!(kernel_scores.len(), rows);
    debug_assert_eq!(decoded_scales.len(), rows);
    debug_assert_eq!(decoded_gammas.len(), rows);
    debug_assert_eq!(decoded_error_ratios.len(), rows);
    let bases = &candidates.bases[candidate_range.clone()];
    let raw_prefixes = &mut candidates.raw_prefixes[candidate_range.clone()];
    let estimates = &mut candidates.estimates[candidate_range.clone()];
    let residual_norms_squared = &candidates.residual_norm_squared[candidate_range.clone()];
    let current_gammas = &mut candidates.gammas[candidate_range.clone()];
    let sign_query_error_terms = &mut candidates.sign_query_error_terms[candidate_range.clone()];
    let arithmetic_variances = &mut candidates.arithmetic_variances[candidate_range.clone()];
    for (index, error) in arithmetic_variances.iter_mut().enumerate() {
        error.refine(
            metric,
            raw_prefixes[index],
            kernel_scores[index],
            decoded_scales[index],
            decoded_constants.get(index).copied().unwrap_or(0.0),
        );
    }
    match metric {
        Metric::L2 => {
            debug_assert_eq!(decoded_constants.len(), rows);
            for (
                (
                    (
                        ((((raw_prefix, estimate), current_gamma), sign_query_error_term), &base),
                        &kernel_score,
                    ),
                    (&scale, &gamma),
                ),
                &constant,
            ) in raw_prefixes
                .iter_mut()
                .zip(estimates.iter_mut())
                .zip(current_gammas.iter_mut())
                .zip(sign_query_error_terms.iter_mut())
                .zip(bases)
                .zip(kernel_scores)
                .zip(decoded_scales.iter().zip(decoded_gammas))
                .zip(decoded_constants)
            {
                *current_gamma = gamma;
                *sign_query_error_term += scale * scale * sign_query_error_squared;
                *raw_prefix = refine_l2_raw_prefix(*raw_prefix, kernel_score, scale, constant);
                *estimate = corrected_quantized_estimate(metric, gamma, *raw_prefix, base);
            }
        }
        Metric::Dot | Metric::Cosine => {
            for (
                (
                    ((((raw_prefix, estimate), current_gamma), sign_query_error_term), &base),
                    &kernel_score,
                ),
                (&scale, &gamma),
            ) in raw_prefixes
                .iter_mut()
                .zip(estimates.iter_mut())
                .zip(current_gammas.iter_mut())
                .zip(sign_query_error_terms.iter_mut())
                .zip(bases)
                .zip(kernel_scores)
                .zip(decoded_scales.iter().zip(decoded_gammas))
            {
                *current_gamma = gamma;
                *sign_query_error_term += scale * scale * sign_query_error_squared;
                *raw_prefix = refine_dot_raw_prefix(*raw_prefix, kernel_score, scale);
                *estimate = corrected_quantized_estimate(metric, gamma, *raw_prefix, base);
            }
        }
    }
    fill_gamma_sigmas(
        &mut candidates.sigmas[candidate_range.clone()],
        residual_norms_squared,
        current_gammas,
        decoded_error_ratios,
        sign_query_error_terms,
        score_query_norm_squared,
        dimension,
        metric,
    );
    for (index, sigma) in candidates.sigmas[candidate_range.clone()]
        .iter_mut()
        .enumerate()
    {
        *sigma = arithmetic_variances[index].sigma(
            metric,
            *sigma,
            current_gammas[index],
            raw_prefixes[index],
            bases[index],
        );
    }
    let sigmas = &candidates.sigmas[candidate_range];
    debug_assert!(estimates.iter().all(|e| e.is_finite()) && sigmas.iter().all(|s| s.is_finite()));
}
