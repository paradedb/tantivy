//! Encoding: a cluster tile or one residual through every layer, with the per-row scale, gamma,
//! corrected-error ratio and split-form constant the `.vec` format stores.

use std::sync::Arc;

use super::query::QueryRotationPlan;
use super::{
    validate, validate_specs, words_to_bytes, LayerKind, LayerSpec, Rotation, GAMMA_MAX, GAMMA_MIN,
};
use crate::grid::{
    encode_f32 as encode_grid, encode_f32_with_scratch as encode_grid_with_scratch,
    unpack as unpack_grid,
};
use crate::model::f16::{f16_to_f32, f32_to_f16};
use crate::model::Grid;
use crate::sign::{encode_f32 as encode_sign, pack as pack_sign, unpack as unpack_sign};

/// Encoded layers for one vector.
#[derive(Clone, Debug)]
pub struct Encoded {
    /// Packed code rows by layer.
    pub codes: Vec<Vec<u8>>,
    /// Row scales by layer.
    pub scales: Vec<f32>,
    /// Split-form constants by layer.
    pub constants: Vec<f32>,
}

/// One layer's row-parallel encoded output for a cluster batch.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedLayerBatch {
    /// Packed row codes, concatenated at the layer's fixed code stride.
    pub codes: Vec<u8>,
    /// One exact binary32 scale per row.
    pub scales: Vec<f32>,
    /// Binary16 cumulative-prefix gamma values.
    pub gammas: Vec<u16>,
    /// Binary16 corrected-error ratios.
    pub corrected_error_ratios: Vec<u16>,
    /// One binary32 split-form constant per row.
    pub constants: Vec<f32>,
}

/// SoA output from the cluster-scoped batch encoder.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedBatch {
    /// Encoded row count.
    pub rows: usize,
    /// Original residual squared norms.
    pub residual_norms_squared: Vec<f32>,
    /// Encoded output by layer.
    pub layers: Vec<EncodedLayerBatch>,
}

/// Reusable cluster encoding buffers.
#[derive(Debug)]
pub struct BatchEncodeWorkspace {
    prefix_reconstructions: Vec<f32>,
    prefix_dots: Vec<f64>,
    rotation_scratch: Vec<f32>,
    sign_words: Vec<u64>,
    grid_code_scratch: Vec<u8>,
    encoded: EncodedBatch,
}

impl BatchEncodeWorkspace {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self {
            prefix_reconstructions: Vec::new(),
            prefix_dots: Vec::new(),
            rotation_scratch: Vec::new(),
            sign_words: Vec::new(),
            grid_code_scratch: Vec::new(),
            encoded: EncodedBatch {
                rows: 0,
                residual_norms_squared: Vec::new(),
                layers: Vec::new(),
            },
        }
    }

    /// Creates a workspace sized for a cluster tile.
    pub fn with_capacity(d: usize, max_rows: usize, specs: &[LayerSpec]) -> Self {
        validate_specs(d, specs);
        let mut workspace = Self::new();
        workspace.reserve(d, max_rows, specs);
        workspace
    }

    fn reserve(&mut self, d: usize, rows: usize, specs: &[LayerSpec]) {
        self.prefix_reconstructions.reserve(rows * d);
        self.prefix_dots.reserve(rows);
        self.encoded.residual_norms_squared.reserve(rows);
        self.rotation_scratch.resize(d, 0.0);
        self.sign_words.resize(d.div_ceil(64), 0);
        self.grid_code_scratch.resize(d, 0);
        self.encoded
            .layers
            .resize_with(specs.len(), || EncodedLayerBatch {
                codes: Vec::new(),
                scales: Vec::new(),
                gammas: Vec::new(),
                corrected_error_ratios: Vec::new(),
                constants: Vec::new(),
            });
        for (spec, layer) in specs.iter().zip(&mut self.encoded.layers) {
            let code_stride = match spec.kind {
                LayerKind::Sign => d.div_ceil(64) * 8,
                LayerKind::Grid => crate::grid::packed_len(d, spec.bits),
            };
            layer.codes.reserve(rows * code_stride);
            layer.scales.reserve(rows);
            layer.gammas.reserve(rows);
            layer.corrected_error_ratios.reserve(rows);
            layer.constants.reserve(rows);
        }
    }
}

impl Default for BatchEncodeWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Gamma values around binary16 serialization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaRoundTrip {
    /// Unclamped gamma.
    pub raw: f64,
    /// Clamped binary32 gamma.
    pub clamped: f32,
    /// Serialized binary16 gamma.
    pub f16: u16,
}

impl GammaRoundTrip {
    /// Returns the decoded binary16 value.
    pub fn f16_value(self) -> f32 {
        f16_to_f32(self.f16)
    }
}

/// Corrected-error ratio around binary16 serialization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectedErrorRatioRoundTrip {
    /// Binary64 ratio before serialization.
    pub raw: f64,
    /// Serialized binary16 ratio.
    pub f16: u16,
}

impl CorrectedErrorRatioRoundTrip {
    /// Returns the decoded binary16 value.
    pub fn f16_value(self) -> f32 {
        f16_to_f32(self.f16)
    }
}

/// Cluster-scoped centroid state shared by every residual encoded in the cluster.
#[derive(Clone, Debug)]
pub struct PreparedCentroid {
    d: usize,
    specs: Vec<LayerSpec>,
    original: Vec<f32>,
    layers: Vec<Vec<f32>>,
    rotation_plan: Arc<QueryRotationPlan>,
}

/// Reusable centroid preparation buffers.
#[derive(Debug)]
pub struct PreparedCentroidWorkspace {
    current: Vec<f32>,
    rotation_scratch: Vec<f32>,
    prepared: PreparedCentroid,
}

impl PreparedCentroidWorkspace {
    /// Creates workspace for a rotation plan.
    pub fn new(plan: Arc<QueryRotationPlan>) -> Self {
        let d = plan.dimension();
        let layer_count = plan.specs.len();
        Self {
            current: Vec::with_capacity(d),
            rotation_scratch: vec![0.0; d],
            prepared: PreparedCentroid {
                d,
                specs: plan.specs.clone(),
                original: Vec::with_capacity(d),
                layers: (0..layer_count).map(|_| Vec::with_capacity(d)).collect(),
                rotation_plan: plan,
            },
        }
    }

    /// Prepares a centroid and returns its transformed layers.
    ///
    /// # Panics
    ///
    /// Panics when the centroid dimension does not match the rotation plan.
    pub fn prepare(&mut self, centroid: &[f32]) -> &PreparedCentroid {
        assert_eq!(centroid.len(), self.prepared.d);
        self.prepared.original.clear();
        self.prepared.original.extend_from_slice(centroid);
        self.current.clear();
        self.current.extend_from_slice(centroid);
        for (rotation, layer) in self
            .prepared
            .rotation_plan
            .rotations
            .iter()
            .zip(&mut self.prepared.layers)
        {
            if let Some(rotation) = rotation {
                rotation.apply_with_scratch(&mut self.current, &mut self.rotation_scratch);
            }
            layer.clear();
            layer.extend_from_slice(&self.current);
        }
        &self.prepared
    }
}

/// Prepares one centroid.
pub fn prepare_centroid(centroid: &[f32], specs: &[LayerSpec]) -> PreparedCentroid {
    let plan = Arc::new(QueryRotationPlan::new(centroid.len(), specs));
    prepare_centroid_with_plan(centroid, plan)
}

/// Prepares one centroid with an existing rotation plan.
pub fn prepare_centroid_with_plan(
    centroid: &[f32],
    plan: Arc<QueryRotationPlan>,
) -> PreparedCentroid {
    let mut workspace = PreparedCentroidWorkspace::new(plan);
    workspace.prepare(centroid);
    workspace.prepared
}

/// Encodes a row-major cluster tile in place.
pub fn encode_batch_in_place(
    vectors: &mut [f32],
    rows: usize,
    centroid: &PreparedCentroid,
    specs: &[LayerSpec],
    grids: &[Grid],
) -> EncodedBatch {
    let mut workspace = BatchEncodeWorkspace::new();
    encode_batch_in_place_reusing(
        vectors,
        rows,
        centroid,
        specs,
        grids,
        &mut workspace,
        true,
        |_, _, _| {},
    );
    workspace.encoded
}

/// Encodes a cluster tile with reusable workspace.
pub fn encode_batch_in_place_with_workspace<'workspace>(
    vectors: &mut [f32],
    rows: usize,
    centroid: &PreparedCentroid,
    specs: &[LayerSpec],
    grids: &[Grid],
    workspace: &'workspace mut BatchEncodeWorkspace,
    compute_constants: bool,
) -> &'workspace EncodedBatch {
    encode_batch_in_place_reusing(
        vectors,
        rows,
        centroid,
        specs,
        grids,
        workspace,
        compute_constants,
        |_, _, _| {},
    );
    &workspace.encoded
}

/// Encodes a cluster tile and observes residuals after each prefix.
pub fn encode_batch_in_place_with_residual_observer<F>(
    vectors: &mut [f32],
    rows: usize,
    centroid: &PreparedCentroid,
    specs: &[LayerSpec],
    grids: &[Grid],
    observer: F,
) -> EncodedBatch
where
    F: FnMut(usize, &[f32], &[f32]),
{
    let mut workspace = BatchEncodeWorkspace::new();
    encode_batch_in_place_reusing(
        vectors,
        rows,
        centroid,
        specs,
        grids,
        &mut workspace,
        true,
        observer,
    );
    workspace.encoded
}

#[allow(clippy::too_many_arguments)]
fn encode_batch_in_place_reusing<F>(
    vectors: &mut [f32],
    rows: usize,
    centroid: &PreparedCentroid,
    specs: &[LayerSpec],
    grids: &[Grid],
    workspace: &mut BatchEncodeWorkspace,
    compute_constants: bool,
    mut observer: F,
) where
    F: FnMut(usize, &[f32], &[f32]),
{
    validate(centroid.d, specs, grids);
    assert_eq!(centroid.specs, specs);
    assert_eq!(vectors.len(), rows * centroid.d);
    let d = centroid.d;
    let BatchEncodeWorkspace {
        prefix_reconstructions,
        prefix_dots,
        rotation_scratch,
        sign_words,
        grid_code_scratch,
        encoded,
    } = workspace;
    for row in vectors.chunks_exact_mut(d) {
        for (value, &center) in row.iter_mut().zip(&centroid.original) {
            *value -= center;
        }
    }

    prefix_reconstructions.clear();
    prefix_reconstructions.resize(rows * d, 0.0);
    encoded.residual_norms_squared.clear();
    encoded.residual_norms_squared.reserve(rows);
    for row in vectors.chunks_exact(d) {
        encoded.residual_norms_squared.push(squared_norm_f32(row));
    }
    prefix_dots.clear();
    prefix_dots.resize(rows, 0.0);
    rotation_scratch.resize(d, 0.0);
    sign_words.resize(d.div_ceil(64), 0);
    grid_code_scratch.resize(d, 0);
    encoded.rows = rows;
    encoded
        .layers
        .resize_with(specs.len(), || EncodedLayerBatch {
            codes: Vec::new(),
            scales: Vec::new(),
            gammas: Vec::new(),
            corrected_error_ratios: Vec::new(),
            constants: Vec::new(),
        });
    for (layer, (spec, grid)) in specs.iter().zip(grids).enumerate() {
        if spec.rotation != Rotation::None {
            let rotation = centroid.rotation_plan.rotations[layer]
                .as_ref()
                .expect("rotating layer must have a prepared rotation");
            for row in vectors.chunks_exact_mut(d) {
                rotation.apply_with_scratch(row, rotation_scratch);
            }
            for row in prefix_reconstructions.chunks_exact_mut(d) {
                rotation.apply_with_scratch(row, rotation_scratch);
            }
        }

        let code_stride = match spec.kind {
            LayerKind::Sign => d.div_ceil(64) * 8,
            LayerKind::Grid => crate::grid::packed_len(d, spec.bits),
        };
        let EncodedLayerBatch {
            codes,
            scales,
            gammas,
            corrected_error_ratios,
            constants,
        } = &mut encoded.layers[layer];
        codes.clear();
        codes.reserve(rows * code_stride);
        scales.clear();
        scales.reserve(rows);
        gammas.clear();
        gammas.reserve(rows);
        corrected_error_ratios.clear();
        corrected_error_ratios.reserve(rows);
        constants.clear();
        if compute_constants {
            constants.reserve(rows);
        }

        for (row_index, (residual, prefix_reconstruction)) in vectors
            .chunks_exact_mut(d)
            .zip(prefix_reconstructions.chunks_exact_mut(d))
            .enumerate()
        {
            match spec.kind {
                LayerKind::Sign => {
                    pack_sign(residual, sign_words);
                    let scale = residual.iter().map(|value| value.abs()).sum::<f32>() / d as f32;
                    let constant = if compute_constants {
                        let mut constant = 0.0_f32;
                        for i in 0..d {
                            let sign = if sign_words[i / 64] & (1_u64 << (i % 64)) != 0 {
                                1.0
                            } else {
                                -1.0
                            };
                            let reconstruction = scale * sign;
                            constant += centroid.layers[layer][i] * reconstruction;
                            prefix_reconstruction[i] += reconstruction;
                            residual[i] -= reconstruction;
                        }
                        Some(constant)
                    } else {
                        for i in 0..d {
                            let sign = if sign_words[i / 64] & (1_u64 << (i % 64)) != 0 {
                                1.0
                            } else {
                                -1.0
                            };
                            let reconstruction = scale * sign;
                            prefix_reconstruction[i] += reconstruction;
                            residual[i] -= reconstruction;
                        }
                        None
                    };
                    for &word in sign_words.iter() {
                        codes.extend_from_slice(&word.to_le_bytes());
                    }
                    scales.push(scale);
                    if let Some(constant) = constant {
                        constants.push(constant);
                    }
                }
                LayerKind::Grid => {
                    let code_start = codes.len();
                    codes.resize(code_start + code_stride, 0);
                    let scale = encode_grid_with_scratch(
                        residual,
                        &grid.points,
                        spec.bits,
                        &mut codes[code_start..],
                        grid_code_scratch,
                    );
                    let constant = if compute_constants {
                        let mut constant = 0.0_f32;
                        for i in 0..d {
                            let point = grid.points[grid_code_scratch[i] as usize];
                            let reconstruction = scale * point;
                            constant += centroid.layers[layer][i] * reconstruction;
                            prefix_reconstruction[i] += reconstruction;
                            residual[i] -= reconstruction;
                        }
                        Some(constant)
                    } else {
                        for i in 0..d {
                            let point = grid.points[grid_code_scratch[i] as usize];
                            let reconstruction = scale * point;
                            prefix_reconstruction[i] += reconstruction;
                            residual[i] -= reconstruction;
                        }
                        None
                    };
                    scales.push(scale);
                    if let Some(constant) = constant {
                        constants.push(constant);
                    }
                }
            }
            let prefix_dot = residual
                .iter()
                .zip(prefix_reconstruction.iter())
                .map(|(&residual, &reconstruction)| {
                    let reconstruction = f64::from(reconstruction);
                    (f64::from(residual) + reconstruction) * reconstruction
                })
                .sum::<f64>();
            prefix_dots[row_index] = prefix_dot;
            let radius_squared = f64::from(encoded.residual_norms_squared[row_index]);
            let gamma = gamma_round_trip(radius_squared, prefix_dot);
            gammas.push(gamma.f16);
            let stored_gamma = f64::from(gamma.f16_value());
            let corrected_error_norm_sq = residual
                .iter()
                .zip(prefix_reconstruction.iter())
                .map(|(&residual, &reconstruction)| {
                    (f64::from(residual) + (1.0 - stored_gamma) * f64::from(reconstruction)).powi(2)
                })
                .sum::<f64>();
            corrected_error_ratios.push(
                corrected_error_ratio_round_trip(radius_squared, corrected_error_norm_sq).f16,
            );
        }

        observer(layer, vectors, scales);
    }
}

/// Encodes one residual through every layer.
///
/// # Panics
///
/// Panics when residual, centroid, schedule, or grid configuration is invalid.
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
        if spec.rotation != Rotation::None {
            if let Some(rotation) =
                centroid.and_then(|context| context.rotation_plan.rotations[layer].as_ref())
            {
                rotation.apply(&mut residual);
            } else {
                spec.rotation.build(r.len()).unwrap().apply(&mut residual);
            }
        }
        match spec.kind {
            LayerKind::Sign => {
                let mut words = vec![0_u64; r.len().div_ceil(64)];
                let scale = encode_sign(&residual, &mut words);
                let signs = unpack_sign(&words, r.len());
                let mut constant = 0.0;
                for (i, (value, sign)) in residual.iter_mut().zip(signs).enumerate() {
                    let reconstruction = scale * sign;
                    if let Some(context) = centroid {
                        constant += context.layers[layer][i] * reconstruction;
                    }
                    *value -= reconstruction;
                }
                all_codes.push(words_to_bytes(&words));
                scales.push(scale);
                constants.push(constant);
            }
            LayerKind::Grid => {
                let mut codes = vec![0_u8; crate::grid::packed_len(r.len(), spec.bits)];
                let scale = encode_grid(&residual, &grid.points, spec.bits, &mut codes);
                let reconstructed = unpack_grid(&codes, r.len(), spec.bits)
                    .into_iter()
                    .map(|code| scale * grid.points[code as usize]);
                let mut constant = 0.0;
                for (i, (value, reconstruction)) in
                    residual.iter_mut().zip(reconstructed).enumerate()
                {
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
    }
    Encoded {
        codes: all_codes,
        scales,
        constants,
    }
}

pub(super) fn gamma_round_trip(norm_sq: f64, prefix_dot: f64) -> GammaRoundTrip {
    let raw = if norm_sq == 0.0 && prefix_dot == 0.0 {
        1.0
    } else {
        norm_sq / prefix_dot
    };
    let clamped = raw.clamp(f64::from(GAMMA_MIN), f64::from(GAMMA_MAX)) as f32;
    GammaRoundTrip {
        raw,
        clamped,
        f16: f32_to_f16(clamped),
    }
}

/// Computes a storage-shaped squared norm.
#[inline]
#[allow(unknown_lints)]
#[allow(clippy::chunks_exact_to_as_chunks)]
pub(super) fn squared_norm_f32(values: &[f32]) -> f32 {
    const LANES: usize = 16;
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    let mut sums = [0.0_f32; LANES];
    for chunk in chunks {
        for lane in 0..LANES {
            sums[lane] += chunk[lane] * chunk[lane];
        }
    }
    let mut total = sums.iter().sum::<f32>();
    for &value in tail {
        total += value * value;
    }
    total
}

pub(super) fn corrected_error_ratio_round_trip(
    original_norm_sq: f64,
    corrected_error_norm_sq: f64,
) -> CorrectedErrorRatioRoundTrip {
    let raw = if original_norm_sq == 0.0 {
        debug_assert_eq!(corrected_error_norm_sq, 0.0);
        0.0
    } else {
        corrected_error_norm_sq / original_norm_sq
    };
    CorrectedErrorRatioRoundTrip {
        raw,
        f16: f32_to_f16(raw as f32),
    }
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::cascade::test_support::*;
    use crate::cascade::*;
    use crate::model::build_grid;

    #[cfg(debug_assertions)]
    #[test]
    fn centroid_preparation_shares_one_expanded_plan_and_rotates_once_per_cluster() {
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
        let plan = Arc::new(QueryRotationPlan::new(d, &specs));
        let mut workspace = PreparedCentroidWorkspace::new(Arc::clone(&plan));
        let first_values: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013).sin()).collect();
        let second_values: Vec<f32> = (0..d).map(|i| (i as f32 * 0.017).cos()).collect();
        let storage_signature = |workspace: &PreparedCentroidWorkspace| {
            let mut storage = vec![
                (
                    workspace.current.as_ptr() as usize,
                    workspace.current.capacity(),
                ),
                (
                    workspace.rotation_scratch.as_ptr() as usize,
                    workspace.rotation_scratch.capacity(),
                ),
                (
                    workspace.prepared.original.as_ptr() as usize,
                    workspace.prepared.original.capacity(),
                ),
                (
                    workspace.prepared.layers.as_ptr() as usize,
                    workspace.prepared.layers.capacity(),
                ),
            ];
            storage.extend(
                workspace
                    .prepared
                    .layers
                    .iter()
                    .map(|layer| (layer.as_ptr() as usize, layer.capacity())),
            );
            storage
        };
        let allocated = storage_signature(&workspace);
        assert!(allocated.iter().all(|(_, capacity)| *capacity > 0));

        crate::rotation::debug_reset_apply_count();
        let first_layers = {
            let first = workspace.prepare(&first_values);
            assert!(Arc::ptr_eq(&first.rotation_plan, &plan));
            let first_clone = first.clone();
            assert!(Arc::ptr_eq(
                &first.rotation_plan,
                &first_clone.rotation_plan
            ));
            first.layers.clone()
        };
        assert_eq!(storage_signature(&workspace), allocated);
        let second = workspace.prepare(&second_values);
        assert!(Arc::ptr_eq(&second.rotation_plan, &plan));
        assert_eq!(crate::rotation::debug_apply_count(), 2 * specs.len());
        assert_eq!(storage_signature(&workspace), allocated);

        let direct = prepare_centroid(&first_values, &specs);
        assert_eq!(first_values, direct.original);
        assert_eq!(first_layers, direct.layers);
    }

    #[test]
    fn exact_first_layer_produces_zero_second_layer() {
        let d = 64;
        let specs = [
            LayerSpec {
                kind: LayerKind::Sign,
                bits: 1,
                rotation: Rotation::SeededFhtChaCha8 { seed: 17 },
            },
            LayerSpec {
                kind: LayerKind::Grid,
                bits: 4,
                rotation: Rotation::SeededFhtChaCha8 { seed: 23 },
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 4)];
        let target = vec![0.5; d];
        let mut input = target.clone();
        specs[0]
            .rotation
            .build(d)
            .unwrap()
            .apply_inverse(&mut input);
        let mut rotated = input.clone();
        specs[0].rotation.build(d).unwrap().apply(&mut rotated);
        assert_eq!(rotated, target);
        let encoded = encode_layers(&input, None, &specs, &grids);
        assert_eq!(encoded.scales[1], 0.0);
        assert!(encoded.codes[1].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn cluster_batch_matches_row_encoder_byte_for_byte() {
        let d = 128;
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
        let centroid: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013).sin() * 0.1).collect();
        let vectors: Vec<Vec<f32>> = (0..7)
            .map(|row| {
                centroid
                    .iter()
                    .enumerate()
                    .map(|(i, &center)| center + ((i + row) as f32 * 0.031).cos() * 0.2)
                    .collect()
            })
            .collect();
        let prepared = prepare_centroid(&centroid, &specs);
        let expected: Vec<Encoded> = vectors
            .iter()
            .map(|vector| {
                let residual: Vec<f32> = vector
                    .iter()
                    .zip(&centroid)
                    .map(|(&value, &center)| value - center)
                    .collect();
                encode_layers(&residual, Some(&prepared), &specs, &grids)
            })
            .collect();
        let mut row_major: Vec<f32> = vectors.iter().flatten().copied().collect();
        let actual =
            encode_batch_in_place(&mut row_major, expected.len(), &prepared, &specs, &grids);

        let mut without_constant_values: Vec<f32> = vectors.iter().flatten().copied().collect();
        let mut workspace = BatchEncodeWorkspace::with_capacity(d, expected.len(), &specs);
        let without_constants = encode_batch_in_place_with_workspace(
            &mut without_constant_values,
            expected.len(),
            &prepared,
            &specs,
            &grids,
            &mut workspace,
            false,
        );

        assert_eq!(actual.rows, expected.len());
        for (layer, batch) in actual.layers.iter().enumerate() {
            let expected_codes: Vec<u8> = expected
                .iter()
                .flat_map(|encoded| encoded.codes[layer].iter().copied())
                .collect();
            let expected_scales: Vec<f32> = expected
                .iter()
                .map(|encoded| encoded.scales[layer])
                .collect();
            let expected_constants: Vec<f32> = expected
                .iter()
                .map(|encoded| encoded.constants[layer])
                .collect();
            assert_eq!(batch.codes, expected_codes);
            assert_eq!(batch.scales, expected_scales);
            assert_eq!(batch.constants, expected_constants);
            assert_eq!(batch.gammas.len(), expected.len());
            assert_eq!(batch.corrected_error_ratios.len(), expected.len());
            assert_eq!(without_constants.layers[layer].codes, batch.codes);
            assert_eq!(without_constants.layers[layer].scales, batch.scales);
            assert_eq!(without_constants.layers[layer].gammas, batch.gammas);
            assert_eq!(
                without_constants.layers[layer].corrected_error_ratios,
                batch.corrected_error_ratios
            );
            assert!(without_constants.layers[layer].constants.is_empty());
        }
    }

    fn batch_workspace_storage_signature(workspace: &BatchEncodeWorkspace) -> Vec<(usize, usize)> {
        let mut storage = vec![
            (
                workspace.prefix_reconstructions.as_ptr() as usize,
                workspace.prefix_reconstructions.capacity(),
            ),
            (
                workspace.prefix_dots.as_ptr() as usize,
                workspace.prefix_dots.capacity(),
            ),
            (
                workspace.rotation_scratch.as_ptr() as usize,
                workspace.rotation_scratch.capacity(),
            ),
            (
                workspace.sign_words.as_ptr() as usize,
                workspace.sign_words.capacity(),
            ),
            (
                workspace.grid_code_scratch.as_ptr() as usize,
                workspace.grid_code_scratch.capacity(),
            ),
            (
                workspace.encoded.layers.as_ptr() as usize,
                workspace.encoded.layers.capacity(),
            ),
            (
                workspace.encoded.residual_norms_squared.as_ptr() as usize,
                workspace.encoded.residual_norms_squared.capacity(),
            ),
        ];
        for layer in &workspace.encoded.layers {
            storage.extend([
                (layer.codes.as_ptr() as usize, layer.codes.capacity()),
                (layer.scales.as_ptr() as usize, layer.scales.capacity()),
                (layer.gammas.as_ptr() as usize, layer.gammas.capacity()),
                (
                    layer.corrected_error_ratios.as_ptr() as usize,
                    layer.corrected_error_ratios.capacity(),
                ),
                (
                    layer.constants.as_ptr() as usize,
                    layer.constants.capacity(),
                ),
            ]);
        }
        storage
    }

    #[test]
    fn batch_workspace_reuses_every_allocation_across_tiles() {
        let d = 100;
        let max_rows = 8;
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
        let centroid: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013).sin() * 0.1).collect();
        let make_vectors = |rows| {
            (0..rows)
                .flat_map(|row| {
                    centroid
                        .iter()
                        .enumerate()
                        .map(move |(i, &center)| center + ((i + row) as f32 * 0.031).cos() * 0.2)
                })
                .collect::<Vec<f32>>()
        };
        let prepared = prepare_centroid(&centroid, &specs);
        let mut workspace = BatchEncodeWorkspace::with_capacity(d, max_rows, &specs);
        let allocated = batch_workspace_storage_signature(&workspace);
        assert!(allocated.iter().all(|(_, capacity)| *capacity > 0));

        let mut first_vectors = make_vectors(max_rows);
        let first = encode_batch_in_place_with_workspace(
            &mut first_vectors,
            max_rows,
            &prepared,
            &specs,
            &grids,
            &mut workspace,
            true,
        )
        .clone();
        assert_eq!(batch_workspace_storage_signature(&workspace), allocated);

        let small_rows = 3;
        let mut small_vectors = make_vectors(small_rows);
        encode_batch_in_place_with_workspace(
            &mut small_vectors,
            small_rows,
            &prepared,
            &specs,
            &grids,
            &mut workspace,
            true,
        );
        assert_eq!(batch_workspace_storage_signature(&workspace), allocated);

        let mut repeated_vectors = make_vectors(max_rows);
        let repeated = encode_batch_in_place_with_workspace(
            &mut repeated_vectors,
            max_rows,
            &prepared,
            &specs,
            &grids,
            &mut workspace,
            true,
        );
        assert_eq!(repeated, &first);
        assert_eq!(batch_workspace_storage_signature(&workspace), allocated);
    }

    #[test]
    fn serialized_prefix_decode_reconstructs_gamma_error_and_shared_radius() {
        const D: usize = 100;
        const ROWS: usize = 3;
        for bits in [[1_u8, 4_u8], [2_u8, 4_u8]] {
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
                        seed: 0xDEC0_DE00 + layer as u64,
                    },
                })
                .collect();
            let grids: Vec<Grid> = bits.iter().map(|&bits| build_grid(D, bits)).collect();
            let centroid: Vec<f32> = (0..D)
                .map(|i| ((i as f32 + 0.75) * 0.017).sin() * 0.13)
                .collect();
            let vectors: Vec<Vec<f32>> = (0..ROWS)
                .map(|row| {
                    centroid
                        .iter()
                        .enumerate()
                        .map(|(i, &center)| {
                            center
                                + ((i * 11 + row * 7) as f32 * 0.029).cos()
                                    * (0.2 + row as f32 * 0.03)
                        })
                        .collect()
                })
                .collect();
            let mut row_major: Vec<f32> = vectors.iter().flatten().copied().collect();
            let batch = encode_batch_in_place(
                &mut row_major,
                ROWS,
                &prepare_centroid(&centroid, &specs),
                &specs,
                &grids,
            );

            for (row_index, vector) in vectors.iter().enumerate() {
                let mut residual: Vec<f32> = vector
                    .iter()
                    .zip(&centroid)
                    .map(|(&value, &center)| value - center)
                    .collect();

                let chunks = residual.chunks_exact(16);
                let tail = chunks.remainder();
                let mut lanes = [0.0_f32; 16];
                for chunk in chunks {
                    for lane in 0..16 {
                        lanes[lane] += chunk[lane] * chunk[lane];
                    }
                }
                let mut radius_squared = lanes.iter().sum::<f32>();
                for &value in tail {
                    radius_squared += value * value;
                }
                assert_eq!(
                    batch.residual_norms_squared[row_index].to_bits(),
                    radius_squared.to_bits(),
                    "schedule={bits:?} row={row_index}"
                );

                let mut prefix_reconstruction = vec![0.0_f32; D];
                for (layer, (spec, grid)) in specs.iter().zip(&grids).enumerate() {
                    let rotation = spec.rotation.build(D).unwrap();
                    rotation.apply(&mut residual);
                    rotation.apply(&mut prefix_reconstruction);

                    let stride = match spec.kind {
                        LayerKind::Sign => D.div_ceil(64) * std::mem::size_of::<u64>(),
                        LayerKind::Grid => crate::grid::packed_len(D, spec.bits),
                    };
                    let codes =
                        &batch.layers[layer].codes[row_index * stride..(row_index + 1) * stride];
                    let scale = batch.layers[layer].scales[row_index];
                    let contribution: Vec<f32> = match spec.kind {
                        LayerKind::Sign => (0..D)
                            .map(|i| {
                                let byte = codes[i / 8];
                                let sign = if byte & (1 << (i % 8)) != 0 {
                                    1.0
                                } else {
                                    -1.0
                                };
                                scale * sign
                            })
                            .collect(),
                        LayerKind::Grid => unpack_grid(codes, D, spec.bits)
                            .into_iter()
                            .map(|code| scale * grid.points[code as usize])
                            .collect(),
                    };
                    for ((residual, prefix), reconstruction) in residual
                        .iter_mut()
                        .zip(&mut prefix_reconstruction)
                        .zip(contribution)
                    {
                        *prefix += reconstruction;
                        *residual -= reconstruction;
                    }

                    let prefix_dot = residual
                        .iter()
                        .zip(&prefix_reconstruction)
                        .map(|(&residual, &reconstruction)| {
                            let reconstruction = f64::from(reconstruction);
                            (f64::from(residual) + reconstruction) * reconstruction
                        })
                        .sum::<f64>();
                    let raw_gamma = if radius_squared == 0.0 && prefix_dot == 0.0 {
                        1.0
                    } else {
                        f64::from(radius_squared) / prefix_dot
                    };
                    let expected_gamma = f32_to_f16(
                        raw_gamma.clamp(f64::from(GAMMA_MIN), f64::from(GAMMA_MAX)) as f32,
                    );
                    assert_eq!(
                        batch.layers[layer].gammas[row_index], expected_gamma,
                        "schedule={bits:?} row={row_index} layer={layer}"
                    );

                    let stored_gamma = f64::from(f16_to_f32(expected_gamma));
                    let corrected_error_norm_squared = residual
                        .iter()
                        .zip(&prefix_reconstruction)
                        .map(|(&residual, &reconstruction)| {
                            (f64::from(residual) + (1.0 - stored_gamma) * f64::from(reconstruction))
                                .powi(2)
                        })
                        .sum::<f64>();
                    let expected_error_ratio = f32_to_f16(
                        (corrected_error_norm_squared / f64::from(radius_squared)) as f32,
                    );
                    assert_eq!(
                        batch.layers[layer].corrected_error_ratios[row_index], expected_error_ratio,
                        "schedule={bits:?} row={row_index} layer={layer}"
                    );
                }
            }
        }
    }

    #[test]
    fn cluster_batch_zero_residual_metadata_is_canonical() {
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
        let centroid = vec![0.25_f32; d];
        let mut vectors = centroid.clone();
        let batch = encode_batch_in_place(
            &mut vectors,
            1,
            &prepare_centroid(&centroid, &specs),
            &specs,
            &grids,
        );
        assert_eq!(batch.residual_norms_squared, [0.0]);
        for layer in batch.layers {
            assert_eq!(layer.gammas, [f32_to_f16(1.0)]);
            assert_eq!(layer.corrected_error_ratios, [f32_to_f16(0.0)]);
        }
    }

    #[test]
    fn residual_observer_reports_each_exact_prefix_error() {
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
        let centroid: Vec<f32> = (0..d).map(|i| (i as f32 * 0.017).sin() * 0.1).collect();
        let original: Vec<f32> = centroid
            .iter()
            .enumerate()
            .map(|(i, &center)| center + (i as f32 * 0.029).cos() * 0.2)
            .collect();
        let prepared = prepare_centroid(&centroid, &specs);
        let mut observed = Vec::new();
        let mut full = original.clone();
        encode_batch_in_place_with_residual_observer(
            &mut full,
            1,
            &prepared,
            &specs,
            &grids,
            |_, residual, _| observed.push(residual.to_vec()),
        );
        let prefix_prepared = prepare_centroid(&centroid, &specs[..1]);
        let mut prefix = original;
        encode_batch_in_place(&mut prefix, 1, &prefix_prepared, &specs[..1], &grids[..1]);
        assert_eq!(observed, vec![prefix, full]);
    }

    #[test]
    fn gamma_round_trip_handles_zero_and_both_clamps() {
        let zero = gamma_round_trip(0.0, 0.0);
        assert_eq!(zero.raw, 1.0);
        assert_eq!(zero.clamped, 1.0);
        assert_eq!(zero.f16_value(), 1.0);

        let lower = gamma_round_trip(1.0, 2.0);
        assert_eq!(lower.raw, 0.5);
        assert_eq!(lower.clamped, 1.0);
        assert_eq!(lower.f16_value(), 1.0);

        let upper = gamma_round_trip(8.0, 1.0);
        assert_eq!(upper.raw, 8.0);
        assert_eq!(upper.clamped, 4.0);
        assert_eq!(upper.f16_value(), 4.0);
    }

    #[test]
    fn corrected_error_ratio_round_trip_is_unclamped() {
        let zero = corrected_error_ratio_round_trip(0.0, 0.0);
        assert_eq!(zero.raw, 0.0);
        assert_eq!(zero.f16_value(), 0.0);

        let fraction = corrected_error_ratio_round_trip(8.0, 3.0);
        assert_eq!(fraction.raw, 0.375);
        assert_eq!(fraction.f16_value(), 0.375);

        let above_one = corrected_error_ratio_round_trip(1.0, 2.0);
        assert_eq!(above_one.raw, 2.0);
        assert_eq!(above_one.f16_value(), 2.0);
    }

    // An explicit identity transform is valid even for the first layer.
    #[test]
    fn layer_zero_supports_identity_rotation() {
        let specs = [LayerSpec {
            bits: 1,
            kind: LayerKind::Sign,
            rotation: Rotation::None,
        }];
        let grids = [build_grid(64, 1)];
        let encoded = encode_layers(&[0.25; 64], None, &specs, &grids);
        assert_eq!(encoded.scales, [0.25]);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn cumulative_gamma_rotation_count_is_cluster_scoped_and_explicit() {
        let d = 768;
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
        let mut rng = ChaCha8Rng::seed_from_u64(0x0052_4f54_434f_554e);
        let centroid = random_unit(&mut rng, d);
        let rows = 100;
        let mut vectors: Vec<f32> = (0..rows)
            .flat_map(|_| {
                let residual = random_unit(&mut rng, d);
                centroid
                    .iter()
                    .zip(residual)
                    .map(|(&center, residual)| center + residual)
                    .collect::<Vec<_>>()
            })
            .collect();
        crate::rotation::debug_reset_apply_count();
        let context = prepare_centroid(&centroid, &specs);
        encode_batch_in_place(&mut vectors, rows, &context, &specs, &grids);
        let count = crate::rotation::debug_apply_count();
        assert_eq!(count, 2 * 2 * rows + 2, "Rotation::apply count={count}");
    }
}

#[cfg(test)]
mod storage_contract_tests {
    use super::*;
    use crate::cascade::*;

    /// Each quantizer/rotation pair has literal code, scale and transform expectations.
    #[test]
    fn tagged_encoder_literal_cases() {
        let input = [-0.5, 0.25, 0.75, -0.125, 0.375, -0.625, 0.0, 0.5];
        let transformed = [
            0.30935916,
            0.044194173,
            0.0,
            0.44194174,
            -0.26516503,
            0.08838839,
            -0.92807764,
            0.6629126,
        ];
        let cases = [
            (
                LayerKind::Sign,
                Rotation::None,
                [150, 0, 0, 0, 0, 0, 0, 0],
                0.390625_f32,
            ),
            (
                LayerKind::Sign,
                Rotation::SeededFhtChaCha8 { seed: 7 },
                [169, 0, 0, 0, 0, 0, 0, 0],
                0.37565047,
            ),
            (
                LayerKind::Grid,
                Rotation::None,
                [120, 146, 0, 0, 0, 0, 0, 0],
                0.45285556,
            ),
            (
                LayerKind::Grid,
                Rotation::SeededFhtChaCha8 { seed: 7 },
                [151, 201, 0, 0, 0, 0, 0, 0],
                0.45285556,
            ),
        ];
        for (kind, rotation, codes, scale) in cases {
            let bits = if kind == LayerKind::Sign { 1 } else { 2 };
            let specs = [LayerSpec {
                kind,
                bits,
                rotation,
            }];
            let grid = Grid {
                bits,
                points: if bits == 1 {
                    vec![-1.0, 1.0]
                } else {
                    vec![-1.5, -0.5, 0.5, 1.5]
                },
                rho_model: 0.5,
            };
            let mut rotated = input;
            if let Some(transform) = rotation.build(input.len()) {
                transform.apply(&mut rotated);
            }
            let expected = if rotation == Rotation::None {
                input
            } else {
                transformed
            };
            assert_eq!(
                rotated.map(f32::to_bits),
                expected.map(f32::to_bits),
                "{kind:?} {rotation:?} transform"
            );
            let mut values = input;
            let centroid = prepare_centroid(&[0.125; 8], &specs);
            let encoded = encode_batch_in_place(&mut values, 1, &centroid, &specs, &[grid]);
            assert_eq!(
                encoded.layers[0].codes, codes,
                "{kind:?} {rotation:?} codes"
            );
            assert_eq!(
                encoded.layers[0].scales[0].to_bits(),
                scale.to_bits(),
                "{kind:?} {rotation:?} scale"
            );
        }
    }
}
