//! Audits of the error model: re-encoded prefix measurements and query-side quantization error.

use super::encode::{
    corrected_error_ratio_round_trip, gamma_round_trip, squared_norm_f32,
    CorrectedErrorRatioRoundTrip, GammaRoundTrip,
};
use super::query::{prepare_split_query_with_plan, QueryRotationPlan};
use super::{validate, words_to_bytes, LayerKind, LayerSpec};
use crate::grid::{encode_f32 as encode_grid, unpack as unpack_grid};
use crate::model::Grid;
use crate::sign::{encode_f32 as encode_sign, unpack as unpack_sign};

/// Error-model measurements for one encoded prefix.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixErrorAudit {
    /// Gamma serialization measurements.
    pub gamma: GammaRoundTrip,
    /// Corrected-error serialization measurements.
    pub corrected_error_ratio: CorrectedErrorRatioRoundTrip,
    /// Residual-to-reconstruction inner product.
    pub prefix_dot: f64,
    /// Current layer scale.
    pub layer_scale: f32,
    /// Current layer codes.
    pub codes: Vec<u8>,
    /// Cumulative reconstruction in residual coordinates.
    pub raw_prefix_reconstruction: Vec<f32>,
}

/// Non-persisting metadata audit for every active encoded prefix.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixErrorAuditResult {
    /// Original residual squared norm.
    pub norm_sq: f64,
    /// Measurements by prefix depth.
    pub prefixes: Vec<PrefixErrorAudit>,
}

/// Returns squared query-quantization error by layer.
pub fn audit_split_query_layer_error_squared(
    query: &[f32],
    specs: &[LayerSpec],
    grids: &[Grid],
    sign_query_bits: u8,
) -> Vec<f64> {
    validate(query.len(), specs, grids);
    let plan = QueryRotationPlan::new(query.len(), specs);
    audit_split_query_layer_error_squared_with_plan(query, &plan, grids, sign_query_bits)
}

/// Returns squared query-quantization error with an existing rotation plan.
pub fn audit_split_query_layer_error_squared_with_plan(
    query: &[f32],
    plan: &QueryRotationPlan,
    grids: &[Grid],
    sign_query_bits: u8,
) -> Vec<f64> {
    let prepared = prepare_split_query_with_plan(query, plan, grids, sign_query_bits);
    (0..prepared.layers.len())
        .map(|layer| prepared.query_error_squared(layer))
        .collect()
}

/// Re-encodes one residual and returns cumulative corrected-error measurements.
pub fn audit_prefix_error_model(
    r0: &[f32],
    specs: &[LayerSpec],
    grids: &[Grid],
) -> PrefixErrorAuditResult {
    validate(r0.len(), specs, grids);
    let d = r0.len();
    let plan = QueryRotationPlan::new(d, specs);
    let norm_sq = f64::from(squared_norm_f32(r0));
    let mut residual = r0.to_vec();
    let mut prefix_reconstruction = vec![0.0_f32; d];
    let mut raw_prefix_reconstruction = vec![0.0_f32; d];
    let mut prefixes = Vec::with_capacity(specs.len());

    for (layer, (spec, grid)) in specs.iter().zip(grids).enumerate() {
        if let Some(rotation) = &plan.rotations[layer] {
            rotation.apply(&mut residual);
            rotation.apply(&mut prefix_reconstruction);
        }

        let (codes, scale, reconstruction) = match spec.kind {
            LayerKind::Sign => {
                let mut words = vec![0_u64; d.div_ceil(64)];
                let scale = encode_sign(&residual, &mut words);
                let signs = unpack_sign(&words, d);
                let reconstruction: Vec<f32> = signs.into_iter().map(|sign| scale * sign).collect();
                (words_to_bytes(&words), scale, reconstruction)
            }
            LayerKind::Grid => {
                let mut codes = vec![0_u8; crate::grid::packed_len(d, spec.bits)];
                let scale = encode_grid(&residual, &grid.points, spec.bits, &mut codes);
                let unpacked = unpack_grid(&codes, d, spec.bits);
                let reconstruction: Vec<f32> = unpacked
                    .into_iter()
                    .map(|code| scale * grid.points[code as usize])
                    .collect();
                (codes, scale, reconstruction)
            }
        };

        let mut original_space_contribution = reconstruction.clone();
        for earlier_layer in (0..=layer).rev() {
            if let Some(rotation) = &plan.rotations[earlier_layer] {
                rotation.apply_inverse(&mut original_space_contribution);
            }
        }
        for (prefix, contribution) in raw_prefix_reconstruction
            .iter_mut()
            .zip(original_space_contribution)
        {
            *prefix += contribution;
        }
        for ((value, prefix), reconstruction) in residual
            .iter_mut()
            .zip(&mut prefix_reconstruction)
            .zip(reconstruction)
        {
            *prefix += reconstruction;
            *value -= reconstruction;
        }
        let prefix_dot = residual
            .iter()
            .zip(&prefix_reconstruction)
            .map(|(&residual, &reconstruction)| {
                let reconstruction = f64::from(reconstruction);
                (f64::from(residual) + reconstruction) * reconstruction
            })
            .sum::<f64>();
        let gamma = gamma_round_trip(norm_sq, prefix_dot);
        let stored_gamma = f64::from(gamma.f16_value());
        let corrected_error_norm_sq = residual
            .iter()
            .zip(&prefix_reconstruction)
            .map(|(&residual, &reconstruction)| {
                (f64::from(residual) + (1.0 - stored_gamma) * f64::from(reconstruction)).powi(2)
            })
            .sum::<f64>();
        prefixes.push(PrefixErrorAudit {
            gamma,
            corrected_error_ratio: corrected_error_ratio_round_trip(
                norm_sq,
                corrected_error_norm_sq,
            ),
            prefix_dot,
            layer_scale: scale,
            codes,
            raw_prefix_reconstruction: raw_prefix_reconstruction.clone(),
        });
    }

    PrefixErrorAuditResult { norm_sq, prefixes }
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::cascade::test_support::*;
    use crate::cascade::*;
    use crate::model::build_grid;
    use crate::model::f16::f32_to_f16;

    #[test]
    fn cluster_batch_prefix_metadata_matches_audit_bit_for_bit() {
        for d in [65, 100, 128] {
            let schedules = [
                vec![1_u8],
                vec![1, 1],
                vec![1, 4],
                vec![4],
                vec![2, 4],
                vec![3, 1, 2],
            ];
            for bits in schedules {
                let specs: Vec<LayerSpec> = bits
                    .iter()
                    .enumerate()
                    .map(|(layer, &bits)| LayerSpec {
                        kind: if bits == 1 {
                            LayerKind::Sign
                        } else {
                            LayerKind::Grid
                        },
                        bits,
                        rotation: Rotation::SeededFhtChaCha8 {
                            seed: 0x4741_4d4d_4100 + layer as u64,
                        },
                    })
                    .collect();
                let grids: Vec<Grid> = bits.iter().map(|&bits| build_grid(d, bits)).collect();
                let centroid: Vec<f32> = (0..d)
                    .map(|i| ((i as f32 + 0.5) * 0.013).sin() * 0.1)
                    .collect();
                let rows = 4;
                let vectors: Vec<Vec<f32>> = (0..rows)
                    .map(|row| {
                        centroid
                            .iter()
                            .enumerate()
                            .map(|(i, &center)| center + ((i + row * 7) as f32 * 0.031).cos() * 0.2)
                            .collect()
                    })
                    .collect();
                let expected: Vec<PrefixErrorAuditResult> = vectors
                    .iter()
                    .map(|vector| {
                        let residual: Vec<f32> = vector
                            .iter()
                            .zip(&centroid)
                            .map(|(&value, &center)| value - center)
                            .collect();
                        audit_prefix_error_model(&residual, &specs, &grids)
                    })
                    .collect();
                let mut row_major: Vec<f32> = vectors.into_iter().flatten().collect();
                let batch = encode_batch_in_place(
                    &mut row_major,
                    rows,
                    &prepare_centroid(&centroid, &specs),
                    &specs,
                    &grids,
                );

                for (layer, encoded) in batch.layers.iter().enumerate() {
                    let expected_gammas: Vec<u16> = expected
                        .iter()
                        .map(|audit| audit.prefixes[layer].gamma.f16)
                        .collect();
                    let expected_error_ratios: Vec<u16> = expected
                        .iter()
                        .map(|audit| audit.prefixes[layer].corrected_error_ratio.f16)
                        .collect();
                    assert_eq!(
                        encoded.gammas, expected_gammas,
                        "d={d} schedule={bits:?} layer={layer}"
                    );
                    assert_eq!(
                        encoded.corrected_error_ratios, expected_error_ratios,
                        "d={d} schedule={bits:?} layer={layer}"
                    );
                }
            }
        }
    }

    #[test]
    fn prefix_gamma_matches_rotated_odd_dimension_encoding() {
        let d = 100;
        let specs = [
            LayerSpec {
                kind: LayerKind::Sign,
                bits: 1,
                rotation: Rotation::SeededFhtChaCha8 {
                    seed: 0x4741_4d4d_4101,
                },
            },
            LayerSpec {
                kind: LayerKind::Grid,
                bits: 4,
                rotation: Rotation::SeededFhtChaCha8 {
                    seed: 0x4741_4d4d_4102,
                },
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 4)];
        let residual: Vec<f32> = (0..d)
            .map(|i| ((i as f32 + 0.25) * 0.071).sin() * (1.0 + i as f32 / d as f32))
            .collect();
        let audit = audit_prefix_error_model(&residual, &specs, &grids);
        let encoded = encode_layers(&residual, None, &specs, &grids);

        assert_eq!(audit.prefixes.len(), 2);
        for (prefix, ((&scale, codes), spec)) in audit
            .prefixes
            .iter()
            .zip(encoded.scales.iter().zip(&encoded.codes).zip(&specs))
        {
            assert_eq!(prefix.layer_scale, scale);
            assert_eq!(&prefix.codes, codes);
            let expected_stride = match spec.kind {
                LayerKind::Sign => d.div_ceil(64) * std::mem::size_of::<u64>(),
                LayerKind::Grid => crate::grid::packed_len(d, spec.bits),
            };
            assert_eq!(prefix.codes.len(), expected_stride);
            assert_eq!(prefix.gamma.f16, f32_to_f16(prefix.gamma.clamped));
            assert!((1.0..=4.0).contains(&prefix.gamma.f16_value()));
            assert_eq!(
                prefix.corrected_error_ratio.f16,
                f32_to_f16(prefix.corrected_error_ratio.raw as f32)
            );
            assert!(prefix.corrected_error_ratio.raw >= 0.0);
            assert_eq!(prefix.raw_prefix_reconstruction.len(), d);
        }
        assert!((audit.prefixes[0].gamma.raw - std::f64::consts::FRAC_PI_2).abs() < 0.25);
        assert!((audit.prefixes[1].gamma.raw - 1.0).abs() < 0.05);
    }

    #[test]
    fn prefix_gamma_gaussian_distribution_matches_sign_and_refined_limits() {
        let d = 769;
        let specs = [
            LayerSpec {
                kind: LayerKind::Sign,
                bits: 1,
                rotation: Rotation::SeededFhtChaCha8 { seed: 11 },
            },
            LayerSpec {
                kind: LayerKind::Grid,
                bits: 4,
                rotation: Rotation::SeededFhtChaCha8 { seed: 22 },
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 4)];
        let mut rng = ChaCha8Rng::seed_from_u64(0x4741_4d4d_4153_5441);
        let mut gamma_one = 0.0_f64;
        let mut gamma_two = 0.0_f64;
        const SAMPLES: usize = 256;
        for _ in 0..SAMPLES {
            let residual = random_unit(&mut rng, d);
            let audit = audit_prefix_error_model(&residual, &specs, &grids);
            gamma_one += audit.prefixes[0].gamma.raw;
            gamma_two += audit.prefixes[1].gamma.raw;
        }
        gamma_one /= SAMPLES as f64;
        gamma_two /= SAMPLES as f64;
        assert!(
            (gamma_one - std::f64::consts::FRAC_PI_2).abs() <= 0.03,
            "gamma1={gamma_one}"
        );
        assert!((gamma_two - 1.0).abs() <= 0.02, "gamma2={gamma_two}");
    }

    #[test]
    fn prefix_gamma_zero_residual_is_canonical() {
        let d = 65;
        let specs = [
            LayerSpec {
                kind: LayerKind::Sign,
                bits: 1,
                rotation: Rotation::SeededFhtChaCha8 { seed: 11 },
            },
            LayerSpec {
                kind: LayerKind::Grid,
                bits: 4,
                rotation: Rotation::SeededFhtChaCha8 { seed: 22 },
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 4)];
        let audit = audit_prefix_error_model(&vec![0.0; d], &specs, &grids);
        assert_eq!(audit.norm_sq, 0.0);
        for prefix in audit.prefixes {
            assert_eq!(prefix.prefix_dot, 0.0);
            assert_eq!(prefix.layer_scale, 0.0);
            assert_eq!(prefix.gamma.raw, 1.0);
            assert_eq!(prefix.gamma.f16_value(), 1.0);
            assert_eq!(prefix.corrected_error_ratio.raw, 0.0);
            assert_eq!(prefix.corrected_error_ratio.f16_value(), 0.0);
        }
    }

    #[test]
    fn split_query_error_audit_matches_sign_bitplanes_and_grid_is_exact() {
        let d = 100;
        let specs = [
            LayerSpec {
                kind: LayerKind::Sign,
                bits: 1,
                rotation: Rotation::SeededFhtChaCha8 { seed: 11 },
            },
            LayerSpec {
                kind: LayerKind::Grid,
                bits: 4,
                rotation: Rotation::SeededFhtChaCha8 { seed: 22 },
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 4)];
        let query: Vec<f32> = (0..d).map(|i| ((i as f32 + 0.5) * 0.037).cos()).collect();
        let errors = audit_split_query_layer_error_squared(&query, &specs, &grids, 4);
        assert_eq!(errors.len(), 2);
        assert!(errors[0] > 0.0);
        assert_eq!(errors[1], 0.0);

        let sign_specs = [
            specs[0],
            LayerSpec {
                kind: LayerKind::Sign,
                bits: 1,
                rotation: specs[1].rotation,
            },
        ];
        let sign_grids = [build_grid(d, 1), build_grid(d, 1)];
        let sign_errors =
            audit_split_query_layer_error_squared(&query, &sign_specs, &sign_grids, 4);
        assert_eq!(sign_errors.len(), 2);
        assert!(sign_errors.iter().all(|error| *error > 0.0));

        let constant = vec![0.25; d];
        let constant_errors =
            audit_split_query_layer_error_squared(&constant, &specs[..1], &grids[..1], 4);
        assert!(constant_errors[0] >= 0.0);
        assert!(constant_errors[0].is_finite());
    }
}
