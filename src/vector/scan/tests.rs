//! Scan internals: the pruning bound, boundary selection and the combine math.

use std::sync::Arc;

use cascade::{encode_batch_in_place, prepare_centroid};
use quant_model::f16::f16_to_f32;

use super::columns::{BoundaryHarness, QuantizedCandidates};
use super::estimate::{
    combine_initial_decoded, combine_refinement_decoded, cosine_refinement_batches,
};
use crate::schema::Metric;
use crate::vector::interval::{LowerEndpoint, Threshold};
use crate::vector::prepared::{
    corrected_quantized_estimate, quantized_model_sigma, ArithmeticError, QuantizedIndexCtx,
    QuantizedQueryCtx, VectorQuery,
};
use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
use crate::vector::{VectorOptions, VectorQuantizationConfig, VectorQuantizationLayer};
use crate::DocId;

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
    let first_scale = first.scales[0];
    let first_gamma = f16_to_f32(first.gammas[0]);
    let first_error_ratio = f16_to_f32(first.corrected_error_ratios[0]);
    let first_constant = (metric == Metric::L2).then_some(first.constants[0]);
    let mut diagnostic_arithmetic = ArithmeticError::default();
    let diagnostic_first_raw = crate::vector::index_reader::diagnostic_advance_raw_prefix(
        &query,
        metric,
        0,
        &first.codes,
        first_scale,
        first_constant,
        0.0,
        cluster_score,
        residual_norm_squared,
        &mut diagnostic_arithmetic,
    )
    .unwrap();
    let diagnostic_first_estimate =
        corrected_quantized_estimate(metric, first_gamma, diagnostic_first_raw, base);
    let diagnostic_sign_query_error_term =
        first_scale * first_scale * query.query_error_squared(0) as f32;
    let diagnostic_first_sigma = quantized_model_sigma(
        metric,
        DIM,
        residual_norm_squared,
        first_error_ratio,
        first_gamma,
        score_query_norm_squared,
        diagnostic_sign_query_error_term,
    );
    let diagnostic_first_sigma = diagnostic_arithmetic.sigma(
        metric,
        diagnostic_first_sigma,
        first_gamma,
        diagnostic_first_raw,
        base,
    );

    let mut first_kernel = [0.0];
    query.score_layer_batch_unscaled(0, &first.codes, first.codes.len(), &mut first_kernel);
    let mut bases = [0.0];
    let mut estimates = [0.0];
    let mut sigmas = [0.0];
    let mut residual_norms = [0.0];
    let mut sign_query_error_terms = [0.0];
    let mut arithmetic = [ArithmeticError::default()];
    let first_constants = first_constant.map_or_else(Vec::new, |constant| vec![constant]);
    combine_initial_decoded(
        metric,
        DIM,
        &mut first_kernel,
        &mut bases,
        &mut estimates,
        &mut sigmas,
        &mut residual_norms,
        &mut sign_query_error_terms,
        &mut arithmetic,
        &[first_scale],
        &[first_gamma],
        &[first_error_ratio],
        &first_constants,
        &[residual_norm_squared],
        cluster_score,
        score_query_norm_squared,
        query.query_error_squared(0) as f32,
    );
    assert_eq!(diagnostic_first_raw.to_bits(), first_kernel[0].to_bits());
    assert_eq!(diagnostic_first_estimate.to_bits(), estimates[0].to_bits());
    assert_eq!(diagnostic_first_sigma.to_bits(), sigmas[0].to_bits());

    let mut candidates = QuantizedCandidates::with_capacity(1);
    candidates.push(
        0,
        0,
        bases[0],
        first_kernel[0],
        estimates[0],
        sigmas[0],
        residual_norms[0],
        first_gamma,
        sign_query_error_terms[0],
        arithmetic[0],
    );
    let refinement = &encoded.layers[1];
    let refinement_scale = refinement.scales[0];
    let refinement_gamma = f16_to_f32(refinement.gammas[0]);
    let refinement_error_ratio = f16_to_f32(refinement.corrected_error_ratios[0]);
    let refinement_constant = (metric == Metric::L2).then_some(refinement.constants[0]);
    let diagnostic_refined_raw = crate::vector::index_reader::diagnostic_advance_raw_prefix(
        &query,
        metric,
        1,
        &refinement.codes,
        refinement_scale,
        refinement_constant,
        diagnostic_first_raw,
        cluster_score,
        residual_norm_squared,
        &mut diagnostic_arithmetic,
    )
    .unwrap();
    let diagnostic_refined_estimate =
        corrected_quantized_estimate(metric, refinement_gamma, diagnostic_refined_raw, base);
    let diagnostic_refined_sigma = quantized_model_sigma(
        metric,
        DIM,
        residual_norm_squared,
        refinement_error_ratio,
        refinement_gamma,
        score_query_norm_squared,
        diagnostic_sign_query_error_term,
    );
    let diagnostic_refined_sigma = diagnostic_arithmetic.sigma(
        metric,
        diagnostic_refined_sigma,
        refinement_gamma,
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
    let refinement_constants = refinement_constant.map_or_else(Vec::new, |constant| vec![constant]);
    combine_refinement_decoded(
        metric,
        DIM,
        &mut candidates,
        0..1,
        &refinement_kernel,
        &[refinement_scale],
        &[refinement_gamma],
        &[refinement_error_ratio],
        &refinement_constants,
        score_query_norm_squared,
        query.query_error_squared(1) as f32,
    );
    assert_eq!(
        diagnostic_refined_raw.to_bits(),
        candidates.raw_prefixes[0].to_bits()
    );
    assert_eq!(
        diagnostic_refined_estimate.to_bits(),
        candidates.estimates[0].to_bits()
    );
    assert_eq!(
        diagnostic_refined_sigma.to_bits(),
        candidates.sigmas[0].to_bits()
    );
}

#[test]
fn diagnostic_prefixes_match_integrated_sign_grid_combines() {
    assert_diagnostic_prefix_matches_integrated(Metric::Cosine);
    assert_diagnostic_prefix_matches_integrated(Metric::L2);
}

fn push_test_candidate(
    scan: &mut BoundaryHarness,
    row: usize,
    doc: DocId,
    estimate: f32,
    sigma: f32,
) {
    scan.push(row, doc, 0.0, estimate, estimate, sigma, 1.0, 1.0, 0.0);
}

#[test]
fn quantized_boundary_kth_uses_the_row_tie_for_sigma() {
    let mut scan = BoundaryHarness::new(3);
    scan.begin_cluster(2);
    for (row, score, sigma) in [(0, 10.0, 0.0), (1, 9.0, 1.0), (2, 9.0, 100.0)] {
        push_test_candidate(&mut scan, row, row as DocId, score, sigma);
    }
    scan.finish_cluster_bound_with_kappa(2.0);

    assert_eq!(
        scan.running_pessimistic_kth(2, 2.0),
        Some(Threshold(LowerEndpoint(7.0)))
    );
    assert_eq!(
        scan.pessimistic_kth(2, 2.0),
        Some(Threshold(LowerEndpoint(7.0)))
    );
}

/// The APS kth is the lowest point estimate in the lower-endpoint
/// top-k: above the pessimistic kth, and never above the true k-th
/// best estimate.
#[test]
fn running_estimate_kth_bounds_the_kth_estimate() {
    let mut scan = BoundaryHarness::new(3);
    assert_eq!(scan.running_estimate_kth(2), None);
    scan.begin_cluster(2);
    // Lower endpoints at kappa 2: 10, 7, 4. The top-2 by lower endpoint
    // is rows 0 and 1; the top-2 by estimate is rows 2 and 0.
    for (row, score, sigma) in [(0, 10.0, 0.0), (1, 9.0, 1.0), (2, 12.0, 4.0)] {
        push_test_candidate(&mut scan, row, row as DocId, score, sigma);
    }
    scan.finish_cluster_bound_with_kappa(2.0);

    assert_eq!(
        scan.running_pessimistic_kth(2, 2.0),
        Some(Threshold(LowerEndpoint(7.0)))
    );
    assert_eq!(scan.running_estimate_kth(2), Some(9.0));
    assert_eq!(scan.running_estimate_kth(4), None);
}

#[test]
fn cluster_local_admission_kth_matches_full_partition() {
    const TOP_N: usize = 4;
    let mut scan = BoundaryHarness::new(24);
    for cluster in 0..6 {
        scan.begin_cluster(TOP_N);
        for row in cluster * 4..cluster * 4 + 4 {
            push_test_candidate(
                &mut scan,
                row,
                (row % 12) as DocId,
                3.0 - row as f32 * 0.071 + (row as f32 * 0.37).sin() * 0.2,
                0.01 + (row % 5) as f32 * 0.003,
            );
        }
        scan.finish_cluster_bound_with_kappa(2.0);
        assert_eq!(
            scan.running_pessimistic_kth(TOP_N, 2.0),
            scan.pessimistic_kth(TOP_N, 2.0),
            "cluster={cluster}"
        );
    }
}

fn independent_admission_top(scan: &BoundaryHarness, top_n: usize) -> Vec<usize> {
    let mut indices = (0..scan.candidates.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        let left_estimate = scan.candidates.estimates[left] - 2.0 * scan.candidates.sigmas[left];
        let right_estimate = scan.candidates.estimates[right] - 2.0 * scan.candidates.sigmas[right];
        right_estimate
            .total_cmp(&left_estimate)
            .then(scan.candidates.rows[left].cmp(&scan.candidates.rows[right]))
    });
    indices.truncate(top_n);
    indices
}

#[test]
fn cluster_rows_selection_matches_independent_oracle() {
    let clusters: &[&[(usize, DocId, f32, f32)]] = &[
        &[(0, 0, 1.0, 0.01), (1, 1, 2.0, 0.02)],
        &[
            (2, 2, 9.0, 0.03),
            (3, 3, 8.0, 0.04),
            (4, 4, 7.0, 0.05),
            (5, 0, 6.0, 0.06),
            (6, 5, 5.0, 0.07),
        ],
        &[
            (7, 6, 0.0, 0.08),
            (8, 1, 10.0, 0.09),
            (9, 7, 1.0, 0.10),
            (10, 2, 9.0, 8.0),
            (11, 8, 2.0, 0.11),
        ],
    ];

    for top_n in [1, 3] {
        let mut scan = BoundaryHarness::new(16);
        for (cluster, rows) in clusters.iter().enumerate() {
            scan.begin_cluster(top_n);
            for &(row, doc, estimate, sigma) in *rows {
                push_test_candidate(&mut scan, row, doc, estimate, sigma);
            }
            scan.finish_cluster_bound_with_kappa(2.0);

            let expected = independent_admission_top(&scan, top_n);
            let actual_rows = scan.running_rows();
            let expected_rows = expected
                .iter()
                .map(|&index| scan.candidates.rows[index])
                .collect::<Vec<_>>();
            assert_eq!(
                actual_rows, expected_rows,
                "cluster={cluster}, top_n={top_n}"
            );

            let expected_kth = (expected.len() == top_n).then(|| {
                let index = expected[top_n - 1];
                Threshold(LowerEndpoint(
                    scan.candidates.estimates[index] - 2.0 * scan.candidates.sigmas[index],
                ))
            });
            assert_eq!(
                scan.running_pessimistic_kth(top_n, 2.0),
                expected_kth,
                "cluster={cluster}, top_n={top_n}"
            );
        }
    }
}

#[test]
fn cosine_refinement_batches_cross_clusters_and_cap_at_2048() {
    let rows = (0..2_049).collect::<Vec<_>>();
    let clusters = [0..700, 700..1_400, 1_400..2_100];
    let batches = cosine_refinement_batches(rows.len()).collect::<Vec<_>>();
    assert_eq!(batches, [0..2_048, 2_048..2_049]);
    let first_rows = rows[batches[0].clone()].iter().copied();
    for cluster in clusters {
        assert!(
            first_rows.clone().any(|row| cluster.contains(&row)),
            "one logical cosine batch must cross all three clusters"
        );
    }
}

#[test]
fn initial_l2_gamma_leaves_exact_base_unscaled() {
    let mut raw_prefixes = [2.0];
    let mut bases = [0.0];
    let mut estimates = [0.0];
    let mut sigmas = [0.0];
    let mut residual_norms_squared = [0.0];
    let mut sign_query_error_terms = [0.0];
    combine_initial_decoded(
        Metric::L2,
        1,
        &mut raw_prefixes,
        &mut bases,
        &mut estimates,
        &mut sigmas,
        &mut residual_norms_squared,
        &mut sign_query_error_terms,
        &mut [ArithmeticError::default()],
        &[3.0],
        &[2.0],
        &[0.5],
        &[5.0],
        &[7.0],
        10.0,
        11.0,
        13.0,
    );

    assert_eq!(bases, [3.0]);
    assert_eq!(raw_prefixes, [1.0]);
    assert_eq!(estimates, [7.0]);
    assert_eq!(residual_norms_squared, [7.0]);
    assert_eq!(sign_query_error_terms, [117.0]);
    let variance: f32 = 7.0 * 0.5 * 11.0 + 4.0 * 117.0;
    let expected_sigma = 2.0 * 1.15 * variance.sqrt();
    assert!((sigmas[0] - expected_sigma).abs() < 1e-5);
}

#[test]
fn l2_refinement_gamma_corrects_only_raw_prefix_and_keeps_state_local() {
    let mut candidates = QuantizedCandidates::with_capacity(2);
    candidates.push(
        0,
        10,
        10.0,
        2.0,
        14.0,
        0.0,
        9.0,
        1.0,
        0.0,
        ArithmeticError::default(),
    );
    candidates.push(
        1,
        11,
        20.0,
        1.0,
        22.0,
        0.0,
        4.0,
        1.0,
        1.0,
        ArithmeticError::default(),
    );

    combine_refinement_decoded(
        Metric::L2,
        1,
        &mut candidates,
        0..1,
        &[2.0],
        &[3.0],
        &[2.0],
        &[0.5],
        &[5.0],
        7.0,
        11.0,
    );
    combine_refinement_decoded(
        Metric::L2,
        1,
        &mut candidates,
        1..2,
        &[4.0],
        &[2.0],
        &[1.5],
        &[0.25],
        &[1.0],
        17.0,
        13.0,
    );

    assert_eq!(candidates.bases, [10.0, 20.0]);
    assert_eq!(candidates.raw_prefixes, [3.0, 8.0]);
    assert_eq!(candidates.estimates, [22.0, 44.0]);
    assert_eq!(candidates.residual_norm_squared, [9.0, 4.0]);
    assert_eq!(candidates.sign_query_error_terms, [99.0, 53.0]);
    let expected0 = 2.0 * 1.15 * 427.5_f32.sqrt();
    let expected1 = 2.0 * 1.15 * 136.25_f32.sqrt();
    assert!((candidates.sigmas[0] - expected0).abs() < 1e-5);
    assert!((candidates.sigmas[1] - expected1).abs() < 1e-5);
}

#[test]
fn survivor_sets_match_kernel_harness() {
    const CANDIDATES: usize = 40;
    const TOP_N: usize = 10;
    const KAPPA: f32 = QUANTIZED_BOUNDARY_KAPPA;

    let layer0_scores: Vec<f32> = (0..CANDIDATES)
        .map(|row| 2.0 - row as f32 * 0.09 + (row as f32 * 0.7).sin() * 0.08)
        .collect();
    let layer0_sigmas: Vec<f32> = (0..CANDIDATES)
        .map(|row| 0.03 + (row % 5) as f32 * 0.01)
        .collect();
    let refinements: Vec<f32> = (0..CANDIDATES)
        .map(|row| (row as f32 * 0.41).cos() * 0.06)
        .collect();
    let layer1_sigmas: Vec<f32> = (0..CANDIDATES)
        .map(|row| 0.012 + (row % 3) as f32 * 0.004)
        .collect();

    let lower: Vec<f32> = layer0_scores
        .iter()
        .zip(&layer0_sigmas)
        .map(|(&estimate, &sigma)| estimate - KAPPA * sigma)
        .collect();
    let (_, first_kth) = cascade::kth(&lower, TOP_N);
    let harness_first = cascade::band_filter(&layer0_scores, &layer0_sigmas, KAPPA, first_kth);

    let mut scan = BoundaryHarness::new(CANDIDATES);
    for row in 0..CANDIDATES {
        push_test_candidate(
            &mut scan,
            row,
            row as DocId,
            layer0_scores[row],
            layer0_sigmas[row],
        );
    }
    scan.band(TOP_N, KAPPA);
    let scan_first: Vec<u32> = scan.candidates.rows.iter().map(|&row| row as u32).collect();
    assert!(
        harness_first.len() < CANDIDATES,
        "first boundary must measurably filter"
    );
    assert_eq!(scan_first, harness_first, "layer-0 survivor set");

    let second_scores: Vec<f32> = harness_first
        .iter()
        .map(|&row| layer0_scores[row as usize] + refinements[row as usize])
        .collect();
    let second_sigmas: Vec<f32> = harness_first
        .iter()
        .map(|&row| layer1_sigmas[row as usize])
        .collect();
    let lower: Vec<f32> = second_scores
        .iter()
        .zip(&second_sigmas)
        .map(|(&estimate, &sigma)| estimate - KAPPA * sigma)
        .collect();
    let (_, second_kth) = cascade::kth(&lower, TOP_N);
    let harness_second_local =
        cascade::band_filter(&second_scores, &second_sigmas, KAPPA, second_kth);
    let harness_second: Vec<u32> = harness_second_local
        .iter()
        .map(|&local| harness_first[local as usize])
        .collect();

    for index in 0..scan.candidates.len() {
        let row = scan.candidates.rows[index];
        scan.candidates.estimates[index] += refinements[row];
        scan.candidates.sigmas[index] = layer1_sigmas[row];
    }
    scan.band(TOP_N, KAPPA);
    let scan_second: Vec<u32> = scan.candidates.rows.iter().map(|&row| row as u32).collect();
    assert!(
        harness_second.len() < harness_first.len(),
        "second boundary must measurably filter"
    );
    assert_eq!(scan_second, harness_second, "layer-1 survivor set");

    let mut exact_order: Vec<usize> = (0..CANDIDATES).collect();
    exact_order.sort_unstable_by(|&left, &right| {
        (layer0_scores[right] + refinements[right])
            .total_cmp(&(layer0_scores[left] + refinements[left]))
            .then_with(|| left.cmp(&right))
    });
    let recalled = exact_order[..TOP_N]
        .iter()
        .filter(|&&row| scan_second.contains(&(row as u32)))
        .count();
    assert_eq!(recalled, TOP_N, "candidate recall must match the harness");
}

#[test]
fn quantized_boundary_filters_with_finite_sigmas() {
    let mut scan = BoundaryHarness::new(3);
    for (row, score) in [(0, 100.0), (1, 99.0), (2, -100.0)] {
        push_test_candidate(&mut scan, row, row as DocId, score, 0.1);
    }
    scan.band(1, 2.0);
    assert_eq!(scan.candidates.len(), 1);
    assert_eq!(scan.candidates.rows[0], 0);
}
