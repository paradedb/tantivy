//! Combining a decoded layer into per-candidate estimates and sigmas.
//!
//! The float operations keep one order everywhere, so every estimate and sigma is bit-identical
//! across call sites.

use std::ops::Range;

use crate::schema::Metric;
use crate::vector::quantization::{
    corrected_quantized_estimate, initial_dot_raw_prefix, initial_l2_raw_prefix, quantized_sigma,
    refine_dot_raw_prefix, refine_l2_raw_prefix, ArithmeticError,
};
use crate::vector::search::scan::columns::Columns;
use crate::vector::search::scan::layer_read::Decoded;

/// What combining needs about the query and the cluster.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CombineCtx {
    pub(crate) metric: Metric,
    pub(crate) dimension: usize,
    /// The squared query norm the error model scores with.
    pub(crate) query_norm_squared: f32,
    /// The layer's own squared query-quantization error.
    pub(crate) query_error_squared: f32,
    /// The query's similarity to the cluster's centroid.
    pub(crate) cluster_score: f32,
}

// --- Combine ---

/// Turns a shortlisted cluster's decoded layer-0 rows into the candidates at `at`, growing `out` to
/// `at.end`.
#[inline(always)]
pub(crate) fn combine_initial(
    ctx: &CombineCtx,
    decoded: &Decoded,
    out: &mut Columns,
    at: Range<usize>,
) {
    let rows = decoded.kernel.len();
    debug_assert_eq!(decoded.norms.len(), rows);
    debug_assert_eq!(at.len(), rows);
    seed_initial(ctx, decoded, out, at.clone());
    let Columns {
        bases,
        raw_prefixes,
        estimates,
        residual_norm_squared,
        gammas,
        sign_query_error_terms,
        ..
    } = out;
    let bases = &mut bases[at.clone()];
    let raw_prefixes = &mut raw_prefixes[at.clone()];
    let estimates = &mut estimates[at.clone()];
    let residual_norm_squared = &mut residual_norm_squared[at.clone()];
    let gammas = &gammas[at.clone()];
    let sign_query_error_terms = &mut sign_query_error_terms[at.clone()];
    let query_error_squared = ctx.query_error_squared;
    match ctx.metric {
        Metric::L2 => {
            debug_assert_eq!(decoded.constants.len(), rows);
            for (index, raw_prefix) in raw_prefixes.iter_mut().enumerate() {
                let scale = decoded.scales[index];
                let norm = decoded.norms[index];
                *raw_prefix = initial_l2_raw_prefix(*raw_prefix, scale, decoded.constants[index]);
                bases[index] = ctx.cluster_score - norm;
                estimates[index] = corrected_quantized_estimate(
                    ctx.metric,
                    gammas[index],
                    *raw_prefix,
                    bases[index],
                );
                residual_norm_squared[index] = norm;
                sign_query_error_terms[index] = scale * scale * query_error_squared;
            }
        }
        Metric::Dot | Metric::Cosine => {
            for (index, raw_prefix) in raw_prefixes.iter_mut().enumerate() {
                let scale = decoded.scales[index];
                *raw_prefix = initial_dot_raw_prefix(*raw_prefix, scale);
                bases[index] = ctx.cluster_score;
                estimates[index] = corrected_quantized_estimate(
                    ctx.metric,
                    gammas[index],
                    *raw_prefix,
                    bases[index],
                );
                residual_norm_squared[index] = decoded.norms[index];
                sign_query_error_terms[index] = scale * scale * query_error_squared;
            }
        }
    }
    fill_sigmas(ctx, &decoded.errors, out, at);
}

/// Grows `out` to `at.end` and seeds the raw prefixes, gammas and arithmetic errors of a
/// cluster's layer-0 rows at `at`.
#[inline(always)]
fn seed_initial(ctx: &CombineCtx, decoded: &Decoded, out: &mut Columns, at: Range<usize>) {
    for column in [
        &mut out.bases,
        &mut out.estimates,
        &mut out.sigmas,
        &mut out.residual_norm_squared,
        &mut out.sign_query_error_terms,
    ] {
        column.resize(at.end, 0.0);
    }
    out.arithmetic_variances
        .resize(at.end, ArithmeticError::default());
    for (index, error) in out.arithmetic_variances[at.clone()].iter_mut().enumerate() {
        *error = ArithmeticError::initial(
            ctx.metric,
            decoded.kernel[index],
            decoded.scales[index],
            decoded.constants.get(index).copied().unwrap_or(0.0),
            ctx.cluster_score,
            decoded.norms[index],
        );
    }
    out.raw_prefixes.resize(at.end, 0.0);
    out.raw_prefixes[at.clone()].copy_from_slice(&decoded.kernel);
    out.gammas.resize(at.end, 0.0);
    out.gammas[at].copy_from_slice(&decoded.gammas);
}

/// Refines the candidates at `at` in place with a later layer's decoded rows.
#[inline(always)]
pub(crate) fn combine_refine(
    ctx: &CombineCtx,
    decoded: &Decoded,
    columns: &mut Columns,
    at: Range<usize>,
) {
    let rows = at.len();
    debug_assert_eq!(decoded.kernel.len(), rows);
    let bases = &columns.bases[at.clone()];
    let raw_prefixes = &mut columns.raw_prefixes[at.clone()];
    let estimates = &mut columns.estimates[at.clone()];
    let gammas = &mut columns.gammas[at.clone()];
    let sign_query_error_terms = &mut columns.sign_query_error_terms[at.clone()];
    let arithmetic_variances = &mut columns.arithmetic_variances[at.clone()];
    for (index, error) in arithmetic_variances.iter_mut().enumerate() {
        error.refine(
            ctx.metric,
            raw_prefixes[index],
            decoded.kernel[index],
            decoded.scales[index],
            decoded.constants.get(index).copied().unwrap_or(0.0),
        );
    }
    let query_error_squared = ctx.query_error_squared;
    match ctx.metric {
        Metric::L2 => {
            debug_assert_eq!(decoded.constants.len(), rows);
            for (index, raw_prefix) in raw_prefixes.iter_mut().enumerate() {
                let scale = decoded.scales[index];
                let gamma = decoded.gammas[index];
                gammas[index] = gamma;
                sign_query_error_terms[index] += scale * scale * query_error_squared;
                *raw_prefix = refine_l2_raw_prefix(
                    *raw_prefix,
                    decoded.kernel[index],
                    scale,
                    decoded.constants[index],
                );
                estimates[index] =
                    corrected_quantized_estimate(ctx.metric, gamma, *raw_prefix, bases[index]);
            }
        }
        Metric::Dot | Metric::Cosine => {
            for (index, raw_prefix) in raw_prefixes.iter_mut().enumerate() {
                let scale = decoded.scales[index];
                let gamma = decoded.gammas[index];
                gammas[index] = gamma;
                sign_query_error_terms[index] += scale * scale * query_error_squared;
                *raw_prefix = refine_dot_raw_prefix(*raw_prefix, decoded.kernel[index], scale);
                estimates[index] =
                    corrected_quantized_estimate(ctx.metric, gamma, *raw_prefix, bases[index]);
            }
        }
    }
    fill_sigmas(ctx, &decoded.errors, columns, at);
}

// --- Sigma ---

/// Sets the sigmas at `at`: each row's data-model and arithmetic variance under one square root.
/// The loop stays free of calls so it vectorizes.
#[inline(always)]
fn fill_sigmas(ctx: &CombineCtx, error_ratios: &[f32], columns: &mut Columns, at: Range<usize>) {
    let sigmas = &mut columns.sigmas[at.clone()];
    let residual_norms_squared = &columns.residual_norm_squared[at.clone()];
    let gammas = &columns.gammas[at.clone()];
    let sign_query_error_terms = &columns.sign_query_error_terms[at.clone()];
    let arithmetic_variances = &columns.arithmetic_variances[at.clone()];
    let raw_prefixes = &columns.raw_prefixes[at.clone()];
    let bases = &columns.bases[at.clone()];
    debug_assert_eq!(sigmas.len(), error_ratios.len());
    for (index, sigma) in sigmas.iter_mut().enumerate() {
        *sigma = quantized_sigma(
            ctx.metric,
            ctx.dimension,
            residual_norms_squared[index],
            error_ratios[index],
            gammas[index],
            ctx.query_norm_squared,
            sign_query_error_terms[index],
            arithmetic_variances[index],
            raw_prefixes[index],
            bases[index],
        );
    }
    debug_assert!(
        columns.estimates[at].iter().all(|e| e.is_finite()) && sigmas.iter().all(|s| s.is_finite())
    );
}

// --- Benchmark ---

/// Runs the complete layer-0 cluster scoring shape on raw column bytes.
#[cfg(feature = "unstable")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn quantization_bench_layer0_cosine_cluster(
    dimension: usize,
    prepared: &quant::cascade::PreparedSplitQuery,
    spec: quant::cascade::LayerSpec,
    codes: &[u8],
    code_stride: usize,
    scales: &[u8],
    gammas: &[u8],
    error_ratios: &[u8],
    residual_norms: &[u8],
    cluster_score: f32,
    query_norm_squared: f32,
    query_error_squared: f32,
    decoded: &mut Decoded,
    columns: &mut Columns,
) -> f32 {
    use crate::vector::search::scan::layer_read::{decode_f16s, decode_f32s};

    let rows = scales.len() / std::mem::size_of::<f32>();
    assert_eq!(codes.len(), rows * code_stride);
    assert_eq!(gammas.len(), rows * std::mem::size_of::<u16>());
    assert_eq!(error_ratios.len(), rows * std::mem::size_of::<u16>());
    assert_eq!(residual_norms.len(), rows * std::mem::size_of::<f32>());
    decoded.kernel.resize(rows, 0.0);
    prepared.score_layer_batch_unscaled(0, codes, code_stride, spec, &mut decoded.kernel);
    decode_f32s(scales, &mut decoded.scales);
    decode_f16s(gammas, &mut decoded.gammas);
    decode_f16s(error_ratios, &mut decoded.errors);
    decode_f32s(residual_norms, &mut decoded.norms);
    decoded.constants.clear();
    let ctx = CombineCtx {
        metric: Metric::Cosine,
        dimension,
        query_norm_squared,
        query_error_squared,
        cluster_score,
    };
    combine_initial(&ctx, decoded, columns, 0..rows);
    columns.raw_prefixes[rows - 1] + columns.estimates[rows - 1] + columns.sigmas[rows - 1]
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use quant::cascade::{encode_batch_in_place, prepare_centroid};
    use quant::model::f16::f16_to_f32;

    use super::*;
    use crate::schema::VectorOptions;
    use crate::vector::diagnostics::audit::diagnostic_advance_raw_prefix;
    use crate::vector::distance::Similarity;
    use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
    use crate::vector::search::cluster_source::ClusterRows;
    use crate::vector::search::prepared::{QuantizedQueryCtx, VectorQuery};
    use crate::vector::search::scan::columns::BoundaryHarness;
    use crate::vector::storage::quantized::QuantizedIndexCtx;
    use crate::vector::{VectorQuantizationConfig, VectorQuantizationLayer};

    fn ctx(
        metric: Metric,
        dimension: usize,
        query_norm_squared: f32,
        query_error_squared: f32,
        cluster_score: f32,
    ) -> CombineCtx {
        CombineCtx {
            metric,
            dimension,
            query_norm_squared,
            query_error_squared,
            cluster_score,
        }
    }

    fn decoded(
        kernel: &[f32],
        scales: &[f32],
        gammas: &[f32],
        errors: &[f32],
        constants: &[f32],
        norms: &[f32],
    ) -> Decoded {
        Decoded {
            kernel: kernel.to_vec(),
            scales: scales.to_vec(),
            gammas: gammas.to_vec(),
            errors: errors.to_vec(),
            constants: constants.to_vec(),
            norms: norms.to_vec(),
        }
    }

    /// At layer 0 an L2 gamma corrects the raw prefix around the exact base, which stays
    /// `cluster_score - ||r||^2`.
    #[test]
    fn initial_l2_gamma_leaves_exact_base_unscaled() {
        let mut columns = Columns::default();
        combine_initial(
            &ctx(Metric::L2, 1, 11.0, 13.0, 10.0),
            &decoded(&[2.0], &[3.0], &[2.0], &[0.5], &[5.0], &[7.0]),
            &mut columns,
            0..1,
        );
        assert_eq!(columns.bases, [3.0]);
        assert_eq!(columns.raw_prefixes, [1.0]);
        assert_eq!(columns.estimates, [7.0]);
        assert_eq!(columns.residual_norm_squared, [7.0]);
        assert_eq!(columns.sign_query_error_terms, [117.0]);
        let variance: f32 = 7.0 * 0.5 * 11.0 + 4.0 * 117.0;
        // The fused sigma adds its rounding pad and the arithmetic terms, all far below 1e-3.
        assert!((columns.sigmas[0] - 2.0 * 1.15 * variance.sqrt()).abs() < 1e-3);
    }

    /// A refining L2 gamma corrects only the raw prefix, and refining one candidate leaves the
    /// others untouched.
    #[test]
    fn l2_refinement_gamma_corrects_only_raw_prefix_and_keeps_state_local() {
        let mut columns = Columns {
            bases: vec![10.0, 20.0],
            raw_prefixes: vec![2.0, 1.0],
            estimates: vec![14.0, 22.0],
            sigmas: vec![0.0, 0.0],
            residual_norm_squared: vec![9.0, 4.0],
            gammas: vec![1.0, 1.0],
            sign_query_error_terms: vec![0.0, 1.0],
            arithmetic_variances: vec![ArithmeticError::default(); 2],
        };
        combine_refine(
            &ctx(Metric::L2, 1, 7.0, 11.0, 0.0),
            &decoded(&[2.0], &[3.0], &[2.0], &[0.5], &[5.0], &[]),
            &mut columns,
            0..1,
        );
        combine_refine(
            &ctx(Metric::L2, 1, 17.0, 13.0, 0.0),
            &decoded(&[4.0], &[2.0], &[1.5], &[0.25], &[1.0], &[]),
            &mut columns,
            1..2,
        );
        assert_eq!(columns.bases, [10.0, 20.0]);
        assert_eq!(columns.raw_prefixes, [3.0, 8.0]);
        assert_eq!(columns.estimates, [22.0, 44.0]);
        assert_eq!(columns.residual_norm_squared, [9.0, 4.0]);
        assert_eq!(columns.sign_query_error_terms, [99.0, 53.0]);
        // The fused sigma adds its rounding pad and the arithmetic terms, all far below 1e-3.
        assert!((columns.sigmas[0] - 2.0 * 1.15 * 427.5_f32.sqrt()).abs() < 1e-3);
        assert!((columns.sigmas[1] - 2.0 * 1.15 * 136.25_f32.sqrt()).abs() < 1e-3);
    }

    fn assert_diagnostic_prefix_matches_integrated(metric: Metric) {
        const DIM: usize = 64;
        let config = VectorQuantizationConfig::materialize(
            "embedding".to_string(),
            &VectorOptions::new(DIM, metric),
            vec![
                VectorQuantizationLayer { bits: 1, seed: 11 },
                VectorQuantizationLayer { bits: 4, seed: 22 },
            ],
        )
        .unwrap();
        let index = Arc::new(QuantizedIndexCtx::from_config(config).unwrap());
        let query = QuantizedQueryCtx::new(
            Arc::clone(&index),
            &VectorQuery::new(
                metric,
                Arc::new(
                    (0..DIM)
                        .map(|coordinate| ((coordinate as f32 + 0.25) * 0.173).sin())
                        .collect(),
                ),
            ),
        );
        let centroid: Vec<f32> = (0..DIM)
            .map(|coordinate| ((coordinate as f32 + 0.5) * 0.071).cos() * 0.1)
            .collect();
        let mut vector: Vec<f32> = centroid
            .iter()
            .enumerate()
            .map(|(coordinate, &center)| center + ((coordinate as f32 + 0.75) * 0.113).sin() * 0.2)
            .collect();
        let prepared_centroid = prepare_centroid(&centroid, &index.specs);
        let encoded = encode_batch_in_place(
            &mut vector,
            1,
            &prepared_centroid,
            &index.specs,
            &index.grids,
        );
        let residual_norm_squared = encoded.residual_norms_squared[0];
        let cluster_score = if metric == Metric::L2 { -1.75 } else { 0.25 };
        let score_query_norm = query.score_query_norm(cluster_score);
        let score_query_norm_squared = score_query_norm * score_query_norm;
        let base = if metric == Metric::L2 {
            cluster_score - residual_norm_squared
        } else {
            cluster_score
        };

        let first = &encoded.layers[0];
        let first_gamma = f16_to_f32(first.gammas[0]);
        let first_error_ratio = f16_to_f32(first.corrected_error_ratios[0]);
        let first_constant = (metric == Metric::L2).then_some(first.constants[0]);
        let mut diagnostic_arithmetic = ArithmeticError::default();
        let diagnostic_first_raw = diagnostic_advance_raw_prefix(
            &query,
            metric,
            0,
            &first.codes,
            first.scales[0],
            first_constant,
            0.0,
            cluster_score,
            residual_norm_squared,
            &mut diagnostic_arithmetic,
        )
        .unwrap();
        let diagnostic_sign_query_error_term =
            first.scales[0] * first.scales[0] * query.query_error_squared(0) as f32;
        let diagnostic_first_sigma = quantized_sigma(
            metric,
            DIM,
            residual_norm_squared,
            first_error_ratio,
            first_gamma,
            score_query_norm_squared,
            diagnostic_sign_query_error_term,
            diagnostic_arithmetic,
            diagnostic_first_raw,
            base,
        );

        let mut first_kernel = [0.0];
        query.score_layer_batch_unscaled(0, &first.codes, first.codes.len(), &mut first_kernel);
        let mut columns = Columns::default();
        combine_initial(
            &ctx(
                metric,
                DIM,
                score_query_norm_squared,
                query.query_error_squared(0) as f32,
                cluster_score,
            ),
            &decoded(
                &first_kernel,
                &[first.scales[0]],
                &[first_gamma],
                &[first_error_ratio],
                &first_constant.map_or_else(Vec::new, |constant| vec![constant]),
                &[residual_norm_squared],
            ),
            &mut columns,
            0..1,
        );
        assert_eq!(
            diagnostic_first_raw.to_bits(),
            columns.raw_prefixes[0].to_bits()
        );
        assert_eq!(
            corrected_quantized_estimate(metric, first_gamma, diagnostic_first_raw, base).to_bits(),
            columns.estimates[0].to_bits()
        );
        assert_eq!(
            diagnostic_first_sigma.to_bits(),
            columns.sigmas[0].to_bits()
        );

        let refinement = &encoded.layers[1];
        let refinement_gamma = f16_to_f32(refinement.gammas[0]);
        let refinement_error_ratio = f16_to_f32(refinement.corrected_error_ratios[0]);
        let refinement_constant = (metric == Metric::L2).then_some(refinement.constants[0]);
        let diagnostic_refined_raw = diagnostic_advance_raw_prefix(
            &query,
            metric,
            1,
            &refinement.codes,
            refinement.scales[0],
            refinement_constant,
            diagnostic_first_raw,
            cluster_score,
            residual_norm_squared,
            &mut diagnostic_arithmetic,
        )
        .unwrap();
        let diagnostic_refined_sigma = quantized_sigma(
            metric,
            DIM,
            residual_norm_squared,
            refinement_error_ratio,
            refinement_gamma,
            score_query_norm_squared,
            diagnostic_sign_query_error_term,
            diagnostic_arithmetic,
            diagnostic_refined_raw,
            base,
        );
        let mut refinement_kernel = [0.0];
        query.score_layer_batch_unscaled(
            1,
            &refinement.codes,
            refinement.codes.len(),
            &mut refinement_kernel,
        );
        combine_refine(
            &ctx(
                metric,
                DIM,
                score_query_norm_squared,
                query.query_error_squared(1) as f32,
                cluster_score,
            ),
            &decoded(
                &refinement_kernel,
                &[refinement.scales[0]],
                &[refinement_gamma],
                &[refinement_error_ratio],
                &refinement_constant.map_or_else(Vec::new, |constant| vec![constant]),
                &[],
            ),
            &mut columns,
            0..1,
        );
        assert_eq!(
            diagnostic_refined_raw.to_bits(),
            columns.raw_prefixes[0].to_bits()
        );
        assert_eq!(
            corrected_quantized_estimate(metric, refinement_gamma, diagnostic_refined_raw, base)
                .to_bits(),
            columns.estimates[0].to_bits()
        );
        assert_eq!(
            diagnostic_refined_sigma.to_bits(),
            columns.sigmas[0].to_bits()
        );
    }

    /// The diagnostic prefix path and the scan's combine produce bit-identical raw prefixes,
    /// estimates and sigmas on sign and grid layers.
    #[test]
    fn diagnostic_prefixes_match_integrated_sign_grid_combines() {
        assert_diagnostic_prefix_matches_integrated(Metric::Cosine);
        assert_diagnostic_prefix_matches_integrated(Metric::L2);
    }

    /// The Lloyd-Max grid for `(dimension, bits)`, built once per test binary.
    fn grid(dimension: usize, bits: u8) -> quant::model::Grid {
        static GRIDS: Mutex<Vec<(usize, u8, quant::model::Grid)>> = Mutex::new(Vec::new());
        let mut grids = GRIDS.lock().unwrap();
        if let Some((_, _, grid)) = grids
            .iter()
            .find(|entry| (entry.0, entry.1) == (dimension, bits))
        {
            return grid.clone();
        }
        let grid = quant::model::build_grid(dimension, bits);
        grids.push((dimension, bits, grid.clone()));
        grid
    }

    /// One L2 cluster scanned with the query at its centroid, through the serving kernels and
    /// combine; returns the top `k` rows, each boundary's survivors and the layer-0 sigmas.
    fn l2_cluster(
        centroid: &[f32],
        mut vectors: Vec<f32>,
        bits: &[u8],
        seed: u64,
        k: usize,
    ) -> (Vec<usize>, Vec<Vec<usize>>, Vec<f32>) {
        use quant::cascade::{prepare_split_query, LayerSpec};
        let dim = centroid.len();
        let n = vectors.len() / dim;
        let specs: Vec<_> = bits
            .iter()
            .enumerate()
            .map(|(layer, &bits)| LayerSpec {
                kind: if bits == 1 {
                    quant::cascade::LayerKind::Sign
                } else {
                    quant::cascade::LayerKind::Grid
                },
                bits,
                rotation: quant::cascade::Rotation::SeededFhtChaCha8 {
                    seed: seed + layer as u64,
                },
            })
            .collect();
        let grids: Vec<_> = bits.iter().map(|&bits| grid(dim, bits)).collect();
        let encoded = encode_batch_in_place(
            &mut vectors,
            n,
            &prepare_centroid(centroid, &specs),
            &specs,
            &grids,
        );
        let query = prepare_split_query(centroid, &specs, &grids, 4);
        let truth: Vec<f32> = encoded.residual_norms_squared.iter().map(|r| -r).collect();
        let mut exact: Vec<usize> = (0..n).collect();
        exact.sort_by(|&a, &b| truth[b].total_cmp(&truth[a]).then(a.cmp(&b)));
        let mut scan = BoundaryHarness::new(n);
        let mut boundaries = Vec::new();
        let mut initial_sigmas = Vec::new();
        let combine = ctx(Metric::L2, dim, 0.0, 0.0, 0.0);
        for (level, layer) in encoded.layers.iter().enumerate() {
            let mut scores = vec![0.0; n];
            query.score_layer_batch_unscaled(
                level,
                &layer.codes,
                layer.codes.len() / n,
                specs[level],
                &mut scores,
            );
            let gammas: Vec<_> = layer.gammas.iter().copied().map(f16_to_f32).collect();
            let errors: Vec<_> = layer
                .corrected_error_ratios
                .iter()
                .copied()
                .map(f16_to_f32)
                .collect();
            if level == 0 {
                let docs: Vec<_> = (0..n as u32).collect();
                let rows = ClusterRows {
                    cluster: 0,
                    rows: 0..n,
                    offsets: None,
                    docs: Some(&docs),
                    centroid: None,
                };
                let at = scan.candidates.push_block(&rows, Similarity::new(0.0));
                combine_initial(
                    &combine,
                    &decoded(
                        &scores,
                        &layer.scales,
                        &gammas,
                        &errors,
                        &layer.constants,
                        &encoded.residual_norms_squared,
                    ),
                    &mut scan.candidates.columns,
                    at.clone(),
                );
                initial_sigmas = scan.candidates.columns.sigmas[at].to_vec();
            } else {
                let rows = scan.candidates.global_rows();
                let select =
                    |values: &[f32]| -> Vec<f32> { rows.iter().map(|&row| values[row]).collect() };
                let refined = decoded(
                    &select(&scores),
                    &select(&layer.scales),
                    &select(&gammas),
                    &select(&errors),
                    &select(&layer.constants),
                    &[],
                );
                let count = scan.candidates.len();
                combine_refine(&combine, &refined, &mut scan.candidates.columns, 0..count);
            }
            scan.band(k, QUANTIZED_BOUNDARY_KAPPA);
            let survivors = scan.candidates.global_rows();
            assert!(
                exact[..k].iter().all(|row| survivors.contains(row)),
                "true top-k lost at layer {level}"
            );
            boundaries.push(survivors);
        }
        let mut result = scan.candidates.global_rows();
        result.sort_by(|&a, &b| truth[b].total_cmp(&truth[a]).then(a.cmp(&b)));
        result.truncate(k);
        assert_eq!(result, exact[..k]);
        (result, boundaries, initial_sigmas)
    }

    /// With the query at a far-from-origin centroid, L2 sigmas stay positive and the nearest row
    /// survives every boundary.
    #[test]
    fn l2_query_at_centroid_keeps_positive_sigma_and_nearest_row() {
        let centroid: Vec<f32> = (0..64).map(|i| 100_000.0 + (i % 13) as f32).collect();
        let vectors = (0..64)
            .flat_map(|row| {
                centroid
                    .iter()
                    .enumerate()
                    .map(move |(i, &c)| c + ((row * 64 + i) as f32 * 0.037).sin())
            })
            .collect();
        let (result, boundaries, sigmas) = l2_cluster(&centroid, vectors, &[4], 7, 1);
        assert!(sigmas.iter().all(|&sigma| sigma > 0.0));
        assert!(boundaries[0].contains(&34));
        assert_eq!(result, [34]);
    }

    /// Translating data and query together keeps the L2 ranking and only widens survivor sets.
    #[test]
    fn l2_translation_preserves_ranking_and_conservatively_widens_survivors() {
        // Dyadic inputs keep the translation exact in f32.
        for seed in 0..64 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let centroid: Vec<f32> = (0..64).map(|i| (i % 13) as f32).collect();
            let vectors: Vec<f32> = (0..64)
                .flat_map(|_| {
                    centroid
                        .iter()
                        .map(|&c| c + rng.i32(-128..129) as f32 / 128.0)
                        .collect::<Vec<_>>()
                })
                .collect();
            for bits in [&[4][..], &[2, 4][..], &[4, 4][..]] {
                let (original, boundaries, _) =
                    l2_cluster(&centroid, vectors.clone(), bits, seed, 3);
                let shifted_centroid: Vec<_> = centroid.iter().map(|c| c + 100_000.0).collect();
                let shifted_vectors = vectors.iter().map(|v| v + 100_000.0).collect();
                let (translated, shifted_boundaries, _) =
                    l2_cluster(&shifted_centroid, shifted_vectors, bits, seed, 3);
                assert_eq!(translated, original, "seed {seed}, schedule {bits:?}");
                for (before, after) in boundaries.iter().zip(&shifted_boundaries) {
                    assert!(
                        before.iter().all(|row| after.contains(row)),
                        "seed {seed}, schedule {bits:?}: translated survivors shrank"
                    );
                }
            }
        }
    }
}
