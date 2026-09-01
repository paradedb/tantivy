//! Residual quantization cascade and plane-boundary operations.

use fht::Rotation;
use grid_plane::{
    build_lut, decode as decode_grid, encode as encode_grid, estimate as estimate_grid,
};
use quant_model::f16::f16_to_f32;
use quant_model::Grid;
use sign_plane::{encode as encode_sign, estimate_fp as estimate_sign_fp, unpack as unpack_sign};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerSpec {
    pub bits: u8,
    pub seed: u64,
    pub rotate: bool,
}

#[derive(Clone, Debug)]
pub struct Encoded {
    pub codes: Vec<Vec<u8>>,
    pub scales: Vec<u16>,
    pub constants: Vec<f32>,
}

/// Cluster-scoped centroid state shared by every residual encoded in the cluster.
#[derive(Clone, Debug)]
pub struct PreparedCentroid {
    d: usize,
    specs: Vec<LayerSpec>,
    layers: Vec<Vec<f32>>,
    rotations: Vec<Option<Rotation>>,
}

#[derive(Clone, Debug)]
pub struct PreparedFpQuery {
    layers: Vec<Vec<f32>>,
}

pub fn prepare_centroid(centroid: &[f32], specs: &[LayerSpec]) -> PreparedCentroid {
    validate_specs(centroid.len(), specs);
    let mut current = centroid.to_vec();
    let mut layers = Vec::with_capacity(specs.len());
    let mut rotations = Vec::with_capacity(specs.len());
    for (layer, spec) in specs.iter().enumerate() {
        if layer == 0 || spec.rotate {
            let rotation = Rotation::new(centroid.len(), spec.seed);
            rotation.apply(&mut current);
            rotations.push(Some(rotation));
        } else {
            rotations.push(None);
        }
        layers.push(current.clone());
    }
    PreparedCentroid {
        d: centroid.len(),
        specs: specs.to_vec(),
        layers,
        rotations,
    }
}

pub fn encode_layers(
    r: &[f32],
    centroid: Option<&PreparedCentroid>,
    specs: &[LayerSpec],
    grids: &[Grid],
) -> Encoded {
    validate(r.len(), specs, grids);
    if let Some(centroid) = centroid {
        assert_eq!(centroid.d, r.len());
        assert_eq!(centroid.specs, specs);
    }
    let mut residual = r.to_vec();
    let mut all_codes = Vec::with_capacity(specs.len());
    let mut scales = Vec::with_capacity(specs.len());
    let mut constants = Vec::with_capacity(specs.len());

    for (layer, (spec, grid)) in specs.iter().zip(grids).enumerate() {
        if layer == 0 || spec.rotate {
            if let Some(rotation) = centroid.and_then(|context| context.rotations[layer].as_ref()) {
                rotation.apply(&mut residual);
            } else {
                Rotation::new(r.len(), spec.seed).apply(&mut residual);
            }
        }
        if spec.bits == 1 {
            let mut words = vec![0_u64; r.len() / 64];
            let scale = encode_sign(&residual, &mut words);
            let reconstruction_scale = f16_to_f32(scale);
            let signs = unpack_sign(&words, r.len());
            let mut constant = 0.0;
            for (i, (value, sign)) in residual.iter_mut().zip(signs).enumerate() {
                let reconstruction = reconstruction_scale * sign;
                if let Some(context) = centroid {
                    constant += context.layers[layer][i] * reconstruction;
                }
                *value -= reconstruction;
            }
            all_codes.push(words_to_bytes(&words));
            scales.push(scale);
            constants.push(constant);
        } else {
            let mut codes = vec![0_u8; grid_plane::packed_len(r.len(), spec.bits)];
            let scale = encode_grid(&residual, &grid.points, spec.bits, &mut codes);
            let reconstructed = decode_grid(&codes, &grid.points, r.len(), spec.bits, scale);
            let mut constant = 0.0;
            for (i, (value, reconstruction)) in residual.iter_mut().zip(reconstructed).enumerate() {
                if let Some(context) = centroid {
                    constant += context.layers[layer][i] * reconstruction;
                }
                *value -= reconstruction;
            }
            all_codes.push(codes);
            scales.push(scale);
            constants.push(constant);
        }
    }
    Encoded {
        codes: all_codes,
        scales,
        constants,
    }
}

pub fn prepare_fp_query(query: &[f32], specs: &[LayerSpec]) -> PreparedFpQuery {
    assert!(!query.is_empty() && query.len().is_multiple_of(64));
    assert!((1..=4).contains(&specs.len()));
    assert!(specs[0].rotate, "layer 0 must rotate");
    let mut current = query.to_vec();
    let mut layers = Vec::with_capacity(specs.len());
    for (layer, spec) in specs.iter().enumerate() {
        if layer == 0 || spec.rotate {
            Rotation::new(query.len(), spec.seed).apply(&mut current);
        }
        layers.push(current.clone());
    }
    PreparedFpQuery { layers }
}

pub fn estimate_prepared_fp(
    encoded: &Encoded,
    query: &PreparedFpQuery,
    specs: &[LayerSpec],
    grids: &[Grid],
    d: usize,
) -> f32 {
    estimate_prepared_fp_layers(encoded, query, specs, grids, d)
        .into_iter()
        .sum()
}

/// Score an encoded residual using a segment-wide query and stored centroid constants.
pub fn estimate_prepared_fp_split(
    encoded: &Encoded,
    query: &PreparedFpQuery,
    specs: &[LayerSpec],
    grids: &[Grid],
    d: usize,
) -> f32 {
    assert_eq!(encoded.constants.len(), specs.len());
    estimate_prepared_fp_layers(encoded, query, specs, grids, d)
        .into_iter()
        .zip(&encoded.constants)
        .map(|(score, constant)| score - constant)
        .sum()
}

fn estimate_prepared_fp_layers(
    encoded: &Encoded,
    query: &PreparedFpQuery,
    specs: &[LayerSpec],
    grids: &[Grid],
    d: usize,
) -> Vec<f32> {
    assert!(d > 0 && d.is_multiple_of(64));
    assert_eq!(encoded.codes.len(), specs.len());
    assert_eq!(encoded.scales.len(), specs.len());
    assert_eq!(query.layers.len(), specs.len());
    assert_eq!(grids.len(), specs.len());
    specs
        .iter()
        .zip(grids)
        .enumerate()
        .map(|(layer, (spec, grid))| {
            if spec.bits == 1 {
                let words = bytes_to_words(&encoded.codes[layer]);
                estimate_sign_fp(&words, encoded.scales[layer], &query.layers[layer])
            } else {
                let lut = build_lut(&query.layers[layer], &grid.points, spec.bits);
                estimate_grid(
                    &encoded.codes[layer],
                    encoded.scales[layer],
                    &lut,
                    d,
                    spec.bits,
                )
            }
        })
        .collect()
}

pub fn reconstruct_first_space(
    encoded: &Encoded,
    specs: &[LayerSpec],
    grids: &[Grid],
    d: usize,
) -> Vec<f32> {
    assert!(d > 0 && d.is_multiple_of(64));
    assert_eq!(encoded.codes.len(), specs.len());
    assert_eq!(encoded.scales.len(), specs.len());
    assert_eq!(grids.len(), specs.len());
    let mut total = vec![0.0; d];
    for layer in 0..specs.len() {
        let spec = specs[layer];
        let mut reconstruction = if spec.bits == 1 {
            let words = bytes_to_words(&encoded.codes[layer]);
            let scale = f16_to_f32(encoded.scales[layer]);
            unpack_sign(&words, d)
                .into_iter()
                .map(|sign| scale * sign)
                .collect()
        } else {
            decode_grid(
                &encoded.codes[layer],
                &grids[layer].points,
                d,
                spec.bits,
                encoded.scales[layer],
            )
        };
        for prior_layer in (1..=layer).rev() {
            if specs[prior_layer].rotate {
                Rotation::new(d, specs[prior_layer].seed).apply_inverse(&mut reconstruction);
            }
        }
        for (sum, value) in total.iter_mut().zip(reconstruction) {
            *sum += value;
        }
    }
    total
}

/// Select the k-th largest finite score, with `k` one-indexed.
pub fn kth(scores: &[f32], k: usize) -> (usize, f32) {
    assert!(k > 0 && k <= scores.len());
    debug_assert!(scores.iter().all(|score| score.is_finite()));
    assert!(u32::try_from(scores.len()).is_ok());
    let mut indices: Vec<u32> = (0..scores.len() as u32).collect();
    let (_, selected, _) = indices.select_nth_unstable_by(k - 1, |&a, &b| {
        scores[b as usize].total_cmp(&scores[a as usize])
    });
    let index = *selected as usize;
    (index, scores[index])
}

/// Retain candidates whose optimistic score reaches the supplied pessimistic threshold.
pub fn band_filter(scores: &[f32], sigmas: &[f32], kappa: f32, kth_pess: f32) -> Vec<u32> {
    assert_eq!(scores.len(), sigmas.len());
    assert!(u32::try_from(scores.len()).is_ok());
    debug_assert!(kappa.is_finite() && kth_pess.is_finite());
    debug_assert!(scores.iter().chain(sigmas).all(|value| value.is_finite()));
    scores
        .iter()
        .zip(sigmas)
        .enumerate()
        .filter_map(|(i, (&score, &sigma))| (score + kappa * sigma >= kth_pess).then_some(i as u32))
        .collect()
}

fn validate(d: usize, specs: &[LayerSpec], grids: &[Grid]) {
    validate_specs(d, specs);
    assert_eq!(specs.len(), grids.len());
    for (spec, grid) in specs.iter().zip(grids) {
        assert!((1..=8).contains(&spec.bits));
        assert_eq!(spec.bits, grid.bits);
        assert!(matches!(spec.bits, 1..=4));
    }
}

fn validate_specs(d: usize, specs: &[LayerSpec]) {
    assert!(d > 0 && d.is_multiple_of(64));
    assert!((1..=4).contains(&specs.len()));
    assert!(specs[0].rotate, "layer 0 must rotate");
}

fn words_to_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u64> {
    assert_eq!(bytes.len() % 8, 0);
    bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect()
}

#[cfg(test)]
mod tests {
    use quant_model::{build_grid, empirical_sigma, sigma_from_rho, DEFAULT_CAL};
    use rand_chacha::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;

    #[test]
    fn layered_space_bookkeeping_is_exact() {
        let d = 128;
        let rotation = Rotation::new(d, 22);
        let u1: Vec<f32> = (0..d).map(|i| (i as f32 * 0.021).sin()).collect();
        let y1: Vec<f32> = (0..d).map(|i| (i as f32 * 0.037).cos()).collect();
        let mut u2 = u1.clone();
        rotation.apply(&mut u2);
        let y2: Vec<f32> = (0..d).map(|i| (i as f32 * 0.053).sin()).collect();
        let layered = dot(&u1, &y1) + dot(&u2, &y2);
        let mut y2_in_first_space = y2.clone();
        rotation.apply_inverse(&mut y2_in_first_space);
        let combined: Vec<f32> = y1
            .iter()
            .zip(y2_in_first_space)
            .map(|(&a, b)| a + b)
            .collect();
        let direct = dot(&u1, &combined);
        assert!((layered - direct).abs() < 2e-4, "{layered} != {direct}");
    }

    #[test]
    fn exact_first_layer_produces_zero_second_layer() {
        let d = 64;
        let specs = [
            LayerSpec {
                bits: 1,
                seed: 17,
                rotate: true,
            },
            LayerSpec {
                bits: 4,
                seed: 23,
                rotate: true,
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 4)];
        let target = vec![0.5; d];
        let mut input = target.clone();
        Rotation::new(d, specs[0].seed).apply_inverse(&mut input);
        let mut rotated = input.clone();
        Rotation::new(d, specs[0].seed).apply(&mut rotated);
        assert_eq!(rotated, target);
        let encoded = encode_layers(&input, None, &specs, &grids);
        assert_eq!(encoded.scales[1], 0);
        assert!(encoded.codes[1].iter().all(|&byte| byte == 0));
    }

    #[test]
    #[should_panic(expected = "layer 0 must rotate")]
    fn layer_zero_rejects_rotation_ablation() {
        let specs = [LayerSpec {
            bits: 1,
            seed: 17,
            rotate: false,
        }];
        let grids = [build_grid(64, 1)];
        encode_layers(&[0.25; 64], None, &specs, &grids);
    }

    #[test]
    fn layered_golden_and_sigma() {
        let d = 768;
        let specs = [
            LayerSpec {
                bits: 1,
                seed: 11,
                rotate: true,
            },
            LayerSpec {
                bits: 4,
                seed: 22,
                rotate: true,
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 4)];
        let mut rng = ChaCha8Rng::seed_from_u64(0x0043_4153_4341_4445);
        let queries: Vec<Vec<f32>> = (0..32).map(|_| random_unit(&mut rng, d)).collect();
        let prepared: Vec<PreparedFpQuery> = queries
            .iter()
            .map(|query| prepare_fp_query(query, &specs))
            .collect();
        let mut error_energy = 0.0_f64;
        let mut signal_energy = 0.0_f64;
        let mut estimates = Vec::with_capacity(32_000);
        let mut truths = Vec::with_capacity(32_000);
        for _ in 0..1_000 {
            let vector = random_unit(&mut rng, d);
            let encoded = encode_layers(&vector, None, &specs, &grids);
            let mut first_space = vector.clone();
            Rotation::new(d, specs[0].seed).apply(&mut first_space);
            let reconstructed = reconstruct_first_space(&encoded, &specs, &grids, d);
            for (&actual, estimated) in first_space.iter().zip(reconstructed) {
                error_energy += f64::from(actual - estimated).powi(2);
                signal_energy += f64::from(actual).powi(2);
            }
            for (query, prepared) in queries.iter().zip(&prepared) {
                estimates.push(estimate_prepared_fp(&encoded, prepared, &specs, &grids, d));
                truths.push(dot(query, &vector));
            }
        }
        let rho = (error_energy / signal_energy).sqrt();
        assert!((rho - 0.0586).abs() <= 0.004, "rho={rho}");
        let empirical = empirical_sigma(&estimates, &truths);
        let modeled = sigma_from_rho(grids[0].rho_model * grids[1].rho_model, d, DEFAULT_CAL);
        assert!(
            (empirical / modeled - 1.0).abs() <= 0.05,
            "empirical={empirical}, model={modeled}"
        );
    }

    #[test]
    fn split_form_matches_direct_form_per_layer_and_summed() {
        let d = 768;
        let schedules = [
            [
                LayerSpec {
                    bits: 1,
                    seed: 11,
                    rotate: true,
                },
                LayerSpec {
                    bits: 4,
                    seed: 22,
                    rotate: true,
                },
            ],
            [
                LayerSpec {
                    bits: 1,
                    seed: 11,
                    rotate: true,
                },
                LayerSpec {
                    bits: 4,
                    seed: 22,
                    rotate: false,
                },
            ],
        ];
        let grids = [build_grid(d, 1), build_grid(d, 4)];
        let mut rng = ChaCha8Rng::seed_from_u64(0x0053_504c_4954_0001);

        for specs in schedules {
            for _ in 0..32 {
                let query = random_unit(&mut rng, d);
                let centroid: Vec<f32> = random_unit(&mut rng, d)
                    .into_iter()
                    .map(|value| value * 0.4)
                    .collect();
                let residual: Vec<f32> = random_unit(&mut rng, d)
                    .into_iter()
                    .map(|value| value * 0.8)
                    .collect();
                let direct_query: Vec<f32> =
                    query.iter().zip(&centroid).map(|(&q, &c)| q - c).collect();
                let context = prepare_centroid(&centroid, &specs);
                let encoded = encode_layers(&residual, Some(&context), &specs, &grids);
                let direct = prepare_fp_query(&direct_query, &specs);
                let split = prepare_fp_query(&query, &specs);
                let direct_layers =
                    estimate_prepared_fp_layers(&encoded, &direct, &specs, &grids, d);
                let split_layers = estimate_prepared_fp_layers(&encoded, &split, &specs, &grids, d);
                for layer in 0..specs.len() {
                    assert_relative_eq(
                        split_layers[layer] - encoded.constants[layer],
                        direct_layers[layer],
                    );
                }
                assert_relative_eq(
                    estimate_prepared_fp_split(&encoded, &split, &specs, &grids, d),
                    estimate_prepared_fp(&encoded, &direct, &specs, &grids, d),
                );
            }
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn centroid_rotations_are_cluster_scoped() {
        let d = 768;
        let specs = [
            LayerSpec {
                bits: 1,
                seed: 11,
                rotate: true,
            },
            LayerSpec {
                bits: 4,
                seed: 22,
                rotate: true,
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 4)];
        let mut rng = ChaCha8Rng::seed_from_u64(0x0052_4f54_434f_554e);
        let centroid = random_unit(&mut rng, d);
        fht::debug_reset_apply_count();
        let context = prepare_centroid(&centroid, &specs);
        for _ in 0..100 {
            let residual = random_unit(&mut rng, d);
            encode_layers(&residual, Some(&context), &specs, &grids);
        }
        let count = fht::debug_apply_count();
        assert!(count <= 2 * 100 + 2, "Rotation::apply count={count}");
    }

    #[test]
    fn kth_matches_sort_and_edges() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut scores: Vec<f32> = (0..1_001)
            .map(|i| {
                if i % 17 == 0 {
                    0.5
                } else {
                    rng.next_u32() as f32
                }
            })
            .collect();
        scores[3] = 12.0;
        scores[4] = -9.0;
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        for k in [1, 2, 17, 500, scores.len()] {
            let (index, value) = kth(&scores, k);
            assert_eq!(value, sorted[k - 1]);
            assert_eq!(scores[index], value);
        }
    }

    #[test]
    fn band_filter_retains_true_top_k_and_is_ascending() {
        let truths = [10.0_f32, 9.0, 8.0, 7.0, 6.0, 5.0];
        let scores = [8.5_f32, 10.0, 7.75, 7.1, 5.8, 5.2];
        let sigmas: Vec<f32> = scores
            .iter()
            .zip(truths)
            .map(|(&score, truth)| (score - truth).abs())
            .collect();
        let k = 3;
        let (index, value) = kth(&scores, k);
        let retained = band_filter(&scores, &sigmas, 1.0, value - sigmas[index]);
        assert!((0..k as u32).all(|top| retained.contains(&top)));
        assert!(retained.windows(2).all(|pair| pair[0] < pair[1]));
    }

    fn random_unit(rng: &mut ChaCha8Rng, d: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(d);
        while values.len() < d {
            let u1 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            let u2 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * u2;
            values.push((radius * angle.cos()) as f32);
            if values.len() < d {
                values.push((radius * angle.sin()) as f32);
            }
        }
        let norm = values
            .iter()
            .map(|&value| value * value)
            .sum::<f32>()
            .sqrt();
        for value in &mut values {
            *value /= norm;
        }
        values
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    fn assert_relative_eq(actual: f32, expected: f32) {
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
        assert!(
            absolute <= 1e-7 || relative <= 1e-5,
            "actual={actual}, expected={expected}, absolute={absolute}, relative={relative}"
        );
    }
}
