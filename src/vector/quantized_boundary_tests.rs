//! Regressions for the quantized boundary threshold.
//!
//! Two invariants, both found in review of the quantized scan:
//!
//! 1. The threshold at every checkpoint is the k-th largest *lower endpoint* `estimate - kappa *
//!    sigma`, not the lower endpoint of the k-th row by estimate. With the latter, a true top-k row
//!    can be pruned even when every interval encloses its exact score.
//! 2. L2 split-form scoring at `query ≈ centroid` must keep a positive sigma: the data-model width
//!    vanishes there, but the split subtraction's floating-point error does not, and with sigma = 0
//!    the nearest row is pruned by cancellation noise.
//!
//! The property tests below check enclosure, kappa/sigma monotonicity and
//! translation invariance across many seeds; the two named tests are the
//! minimal reproductions.

use super::*;

#[test]
fn enclosing_intervals_preserve_topk_and_running_threshold_many_seeds() {
    for seed in 0..512 {
        let mut rng = fastrand::Rng::with_seed(seed);
        let n = 97;
        let k = 1 + rng.usize(0..20);
        let mut truth = Vec::new();
        let mut scan = QuantizedScanCtx::new(n as u32, n);
        for cluster in (0..n).step_by(11) {
            scan.begin_cluster(k);
            for row in cluster..(cluster + 11).min(n) {
                let score = rng.f32() * 20.0 - 10.0;
                let sigma = 0.001 + rng.f32().powi(3) * 10.0;
                let estimate = score + (rng.f32() * 2.0 - 1.0) * QUANTIZED_BOUNDARY_KAPPA * sigma;
                truth.push(score);
                scan.push(
                    row, row as u32, 0.0, estimate, estimate, sigma, 1.0, 1.0, 0.0,
                );
            }
            scan.finish_cluster_bound();
            assert_eq!(
                scan.running_pessimistic_kth(k, QUANTIZED_BOUNDARY_KAPPA),
                scan.pessimistic_kth(k, QUANTIZED_BOUNDARY_KAPPA),
                "seed {seed}, cluster {cluster}"
            );
            if let Some(threshold) = scan.running_pessimistic_kth(k, QUANTIZED_BOUNDARY_KAPPA) {
                let mut exact = truth.clone();
                exact.sort_by(|a, b| b.total_cmp(a));
                assert!(threshold.0 .0 <= exact[k - 1] + 1e-5, "seed {seed}");
            }
        }
        let mut order = (0..n).collect::<Vec<_>>();
        order.sort_by(|&a, &b| truth[b].total_cmp(&truth[a]));
        scan.band(k, QUANTIZED_BOUNDARY_KAPPA);
        for row in &order[..k] {
            assert!(
                scan.candidates.rows.contains(row),
                "seed {seed}, true row {row}"
            );
        }
    }
}

fn survivors(estimates: &[f32], sigmas: &[f32], k: usize, kappa: f32) -> Vec<usize> {
    let mut scan = QuantizedScanCtx::new(estimates.len() as u32, estimates.len());
    for (row, (&estimate, &sigma)) in estimates.iter().zip(sigmas).enumerate() {
        scan.push(
            row, row as u32, 0.0, estimate, estimate, sigma, 1.0, 1.0, 0.0,
        );
    }
    scan.band(k, kappa);
    scan.candidates.rows
}

#[test]
fn widening_kappa_or_any_sigma_never_removes_a_survivor() {
    for seed in 0..256 {
        let mut rng = fastrand::Rng::with_seed(seed);
        let estimates: Vec<_> = (0..41).map(|_| rng.f32() * 20.0 - 10.0).collect();
        let sigmas: Vec<_> = (0..41).map(|_| rng.f32() * 3.0).collect();
        let k = 1 + rng.usize(0..10);
        let original = survivors(&estimates, &sigmas, k, 2.0);
        let wider = survivors(&estimates, &sigmas, k, 2.5);
        assert!(
            original.iter().all(|row| wider.contains(row)),
            "kappa seed {seed}"
        );
        for changed in 0..sigmas.len() {
            let mut enlarged = sigmas.clone();
            enlarged[changed] += 0.01 + rng.f32() * 10.0;
            let wider = survivors(&estimates, &enlarged, k, 2.0);
            assert!(
                original.iter().all(|row| wider.contains(row)),
                "sigma seed {seed}, row {changed}"
            );
        }
    }
}

/// One cluster, with q = centroid, through the same split-query kernels and
/// combine helpers as serving. Record each boundary before the exact rerank.
fn l2_cluster(
    centroid: &[f32],
    mut vectors: Vec<f32>,
    bits: &[u8],
    seed: u64,
    k: usize,
) -> (Vec<usize>, Vec<Vec<usize>>, Vec<f32>) {
    use cascade::{encode_batch_in_place, prepare_centroid, prepare_split_query, LayerSpec};
    use quant_model::build_grid;
    let dim = centroid.len();
    let n = vectors.len() / dim;
    let specs: Vec<_> = bits
        .iter()
        .enumerate()
        .map(|(layer, &bits)| LayerSpec {
            bits,
            seed: seed + layer as u64,
            rotate: true,
        })
        .collect();
    let grids: Vec<_> = bits.iter().map(|&bits| build_grid(dim, bits)).collect();
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
    let mut scan = QuantizedScanCtx::new(n as u32, n);
    let mut boundaries = Vec::new();
    let mut initial_sigmas = Vec::new();
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
            let mut bases = vec![0.0; n];
            let mut estimates = vec![0.0; n];
            let mut sigmas = vec![0.0; n];
            let mut norms = vec![0.0; n];
            let mut query_errors = vec![0.0; n];
            let mut arithmetic = vec![ArithmeticError::default(); n];
            combine_initial_decoded(
                Metric::L2,
                dim,
                &mut scores,
                &mut bases,
                &mut estimates,
                &mut sigmas,
                &mut norms,
                &mut query_errors,
                &mut arithmetic,
                &layer.scales,
                &gammas,
                &errors,
                &layer.constants,
                &encoded.residual_norms_squared,
                0.0,
                0.0,
                0.0,
            );
            initial_sigmas = sigmas.clone();
            let docs: Vec<_> = (0..n as u32).collect();
            scan.candidates.append_selected(
                0..n,
                &Selection::All,
                &docs,
                &bases,
                &scores,
                &estimates,
                &sigmas,
                &norms,
                &gammas,
                &query_errors,
                &arithmetic,
            );
        } else {
            let select = |values: &[f32]| {
                scan.candidates
                    .rows
                    .iter()
                    .map(|&row| values[row])
                    .collect::<Vec<_>>()
            };
            let (scores, scales, gammas, errors, constants) = (
                select(&scores),
                select(&layer.scales),
                select(&gammas),
                select(&errors),
                select(&layer.constants),
            );
            let count = scan.candidates.len();
            combine_refinement_decoded(
                Metric::L2,
                dim,
                &mut scan.candidates,
                0..count,
                &scores,
                &scales,
                &gammas,
                &errors,
                &constants,
                0.0,
                0.0,
            );
        }
        scan.band(k, QUANTIZED_BOUNDARY_KAPPA);
        assert!(
            exact[..k]
                .iter()
                .all(|row| scan.candidates.rows.contains(row)),
            "true top-k lost at layer {level}"
        );
        boundaries.push(scan.candidates.rows.clone());
    }
    let mut result = scan.candidates.rows;
    result.sort_by(|&a, &b| truth[b].total_cmp(&truth[a]).then(a.cmp(&b)));
    result.truncate(k);
    assert_eq!(result, exact[..k]);
    (result, boundaries, initial_sigmas)
}

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

#[test]
fn l2_translation_preserves_ranking_and_conservatively_widens_survivors() {
    // Dyadic inputs keep dataset/query translation exact in f32; otherwise the
    // input representation itself can change the true ranking before scoring.
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
            let (original, boundaries, _) = l2_cluster(&centroid, vectors.clone(), bits, seed, 3);
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

#[test]
fn boundary_preserves_topk_when_all_intervals_cover_truth() {
    let mut scan = QuantizedScanCtx::new(3, 3);
    let examples: [(f32, f32, f32); 3] = [
        (0.10, 0.015, 0.065),
        (0.09, 0.001, 0.090),
        (0.08, 0.001, 0.080),
    ];
    scan.begin_cluster(2);
    for (row, &(estimate, sigma, truth)) in examples.iter().enumerate() {
        assert!((truth - estimate).abs() <= QUANTIZED_BOUNDARY_KAPPA * sigma);
        scan.push(
            row, row as u32, 0.0, estimate, estimate, sigma, 1.0, 1.0, 0.0,
        );
    }
    scan.finish_cluster_bound();
    scan.band(2, QUANTIZED_BOUNDARY_KAPPA);
    assert!(
        scan.candidates.docs.contains(&2),
        "true runner-up was pruned despite valid intervals"
    );
}
