//! Query preparation and scoring: the rotation plan, full-precision and split-form queries, and
//! the per-layer kernels they dispatch to.

use std::borrow::Cow;

use super::encode::Encoded;
use super::{aligned_le_words, validate, validate_specs, LayerKind, LayerSpec, Rotation};
use crate::grid::{
    build_lut, build_packed_lut_4, score as score_grid, score_batch as score_grid_batch,
    score_batch_indexed as score_grid_batch_indexed,
    score_batch_packed_4 as score_grid_batch_packed_4,
    score_batch_packed_4_indexed as score_grid_batch_packed_4_indexed, unpack as unpack_grid,
};
use crate::model::Grid;
use crate::rotation::Rotation as FhtRotation;
use crate::sign::{
    estimate_asym_batch_unscaled as estimate_sign_batch,
    estimate_asym_batch_unscaled_indexed as estimate_sign_batch_indexed,
    estimate_asym_unscaled as estimate_sign_asym, estimate_fp_unscaled as estimate_sign_fp,
    prepare_query, unpack as unpack_sign, QueryPlanes,
};

#[derive(Clone, Debug)]
/// Full-precision query transformed for every layer.
pub struct PreparedFpQuery {
    layers: Vec<Vec<f32>>,
}

/// Reusable rotations for a quantization schedule.
#[derive(Clone, Debug)]
pub struct QueryRotationPlan {
    d: usize,
    pub(super) specs: Vec<LayerSpec>,
    pub(super) rotations: Vec<Option<FhtRotation>>,
}

impl QueryRotationPlan {
    /// Expands layer seeds into rotations.
    pub fn new(d: usize, specs: &[LayerSpec]) -> Self {
        validate_specs(d, specs);
        let rotations = specs.iter().map(|spec| spec.rotation.build(d)).collect();
        Self {
            d,
            specs: specs.to_vec(),
            rotations,
        }
    }

    /// Returns the vector dimension.
    pub fn dimension(&self) -> usize {
        self.d
    }

    /// Returns the layer specifications.
    pub fn specs(&self) -> &[LayerSpec] {
        &self.specs
    }

    fn active_specs(&self, active_layers: usize) -> &[LayerSpec] {
        assert!((1..=self.specs.len()).contains(&active_layers));
        &self.specs[..active_layers]
    }

    fn prepare_layers(&self, query: &[f32], active_layers: usize) -> Vec<Vec<f32>> {
        assert_eq!(query.len(), self.d);
        self.active_specs(active_layers);
        let mut current = query.to_vec();
        let mut layers = Vec::with_capacity(active_layers);
        for layer in 0..active_layers {
            if let Some(rotation) = &self.rotations[layer] {
                rotation.apply(&mut current);
            }
            layers.push(current.clone());
        }
        layers
    }
}

impl PreparedFpQuery {
    /// Query in the coordinate space of one encoded layer.
    pub fn layer(&self, layer: usize) -> &[f32] {
        &self.layers[layer]
    }

    /// Query in the coordinate space of the final encoded layer.
    ///
    /// # Panics
    ///
    /// Panics when the prepared query contains no layers.
    pub fn final_layer(&self) -> &[f32] {
        self.layers
            .last()
            .expect("validated quantization schedule is non-empty")
    }
}

pub(super) enum PreparedSplitLayer {
    Sign(QueryPlanes),
    Grid {
        lut: Vec<f32>,
        packed_lut_4: Option<Vec<f32>>,
    },
}

/// Prepared sign bitplanes and grid tables for a query.
pub struct PreparedSplitQuery {
    d: usize,
    pub(super) layers: Vec<PreparedSplitLayer>,
}

/// Prepares a split-form query.
pub fn prepare_split_query(
    query: &[f32],
    specs: &[LayerSpec],
    grids: &[Grid],
    sign_query_bits: u8,
) -> PreparedSplitQuery {
    validate(query.len(), specs, grids);
    let plan = QueryRotationPlan::new(query.len(), specs);
    prepare_split_query_with_plan(query, &plan, grids, sign_query_bits)
}

/// Prepares a split-form query with an existing rotation plan.
///
/// # Panics
///
/// Panics when query, plan, grids, or bit-width configuration is invalid.
pub fn prepare_split_query_with_plan(
    query: &[f32],
    plan: &QueryRotationPlan,
    grids: &[Grid],
    sign_query_bits: u8,
) -> PreparedSplitQuery {
    let specs = plan.active_specs(grids.len());
    validate(query.len(), specs, grids);
    assert_eq!(query.len(), plan.dimension());
    assert!((1..=8).contains(&sign_query_bits));
    let rotated_layers = plan.prepare_layers(query, specs.len());
    let mut layers = Vec::with_capacity(specs.len());
    for ((spec, grid), current) in specs.iter().zip(grids).zip(rotated_layers) {
        match spec.kind {
            LayerKind::Sign => {
                layers.push(PreparedSplitLayer::Sign(prepare_query(
                    &current,
                    sign_query_bits,
                )));
            }
            LayerKind::Grid => {
                let lut = build_lut(&current, &grid.points, spec.bits);
                let packed_lut_4 = (spec.bits == 4).then(|| build_packed_lut_4(&lut, query.len()));
                layers.push(PreparedSplitLayer::Grid { lut, packed_lut_4 });
            }
        }
    }
    PreparedSplitQuery {
        d: query.len(),
        layers,
    }
}

impl PreparedSplitQuery {
    /// Returns squared query-quantization error for one layer.
    pub fn query_error_squared(&self, layer: usize) -> f64 {
        match &self.layers[layer] {
            PreparedSplitLayer::Sign(query) => query.error_squared(),
            PreparedSplitLayer::Grid { .. } => 0.0,
        }
    }

    /// Scores one layer with its split-form constant.
    pub fn score_layer(
        &self,
        layer: usize,
        codes: &[u8],
        scale: f32,
        constant: f32,
        spec: LayerSpec,
    ) -> f32 {
        self.score_layer_without_constant(layer, codes, scale, spec) - constant
    }

    /// Scores one layer without a split-form constant.
    ///
    /// # Panics
    ///
    /// Panics when the layer, codes, or specification does not match the prepared query.
    pub fn score_layer_without_constant(
        &self,
        layer: usize,
        codes: &[u8],
        scale: f32,
        spec: LayerSpec,
    ) -> f32 {
        match &self.layers[layer] {
            PreparedSplitLayer::Sign(query) => {
                assert_eq!(spec.bits, 1);
                let words = aligned_le_words(codes);
                scale * estimate_sign_asym(words.as_ref(), query)
            }
            PreparedSplitLayer::Grid { lut, .. } => {
                assert!(spec.bits > 1);
                scale * score_grid(codes, lut, self.d, spec.bits)
            }
        }
    }

    /// Scores a contiguous batch without scales or constants. Returns its sign-word fallbacks: 1
    /// when sign codes aren't `u64`-aligned and are decoded into owned words, else 0.
    ///
    /// # Panics
    ///
    /// Panics when the layer, codes, stride, specification, or output shape is invalid.
    #[inline(always)]
    pub fn score_layer_batch_unscaled(
        &self,
        layer: usize,
        codes: &[u8],
        code_stride: usize,
        spec: LayerSpec,
        out: &mut [f32],
    ) -> u64 {
        assert_eq!(codes.len(), out.len() * code_stride);
        match &self.layers[layer] {
            PreparedSplitLayer::Sign(query) => {
                assert_eq!(spec.bits, 1);
                assert_eq!(code_stride % std::mem::size_of::<u64>(), 0);
                let words = aligned_le_words(codes);
                estimate_sign_batch(
                    words.as_ref(),
                    code_stride / std::mem::size_of::<u64>(),
                    query,
                    out,
                );
                u64::from(matches!(words, Cow::Owned(_)))
            }
            PreparedSplitLayer::Grid { lut, packed_lut_4 } => {
                assert!(spec.bits > 1);
                if let Some(packed_lut_4) = packed_lut_4 {
                    score_grid_batch_packed_4(codes, code_stride, packed_lut_4, self.d, out);
                } else {
                    score_grid_batch(codes, code_stride, lut, self.d, spec.bits, out);
                }
                0
            }
        }
    }

    /// Scores selected rows without scales or constants. Returns its sign-word fallbacks, as
    /// [`Self::score_layer_batch_unscaled`] does.
    ///
    /// # Panics
    ///
    /// Panics when the layer, codes, indices, specification, or output shape is invalid.
    #[inline(always)]
    pub fn score_layer_batch_unscaled_indexed(
        &self,
        layer: usize,
        codes: &[u8],
        code_stride: usize,
        row_offsets: &[usize],
        spec: LayerSpec,
        out: &mut [f32],
    ) -> u64 {
        assert_eq!(row_offsets.len(), out.len());
        match &self.layers[layer] {
            PreparedSplitLayer::Sign(query) => {
                assert_eq!(spec.bits, 1);
                assert_eq!(code_stride % std::mem::size_of::<u64>(), 0);
                let words = aligned_le_words(codes);
                estimate_sign_batch_indexed(
                    words.as_ref(),
                    code_stride / std::mem::size_of::<u64>(),
                    row_offsets,
                    query,
                    out,
                );
                u64::from(matches!(words, Cow::Owned(_)))
            }
            PreparedSplitLayer::Grid {
                packed_lut_4: Some(packed_lut_4),
                ..
            } => {
                assert_eq!(spec.bits, 4);
                score_grid_batch_packed_4_indexed(
                    codes,
                    code_stride,
                    row_offsets,
                    packed_lut_4,
                    self.d,
                    out,
                );
                0
            }
            PreparedSplitLayer::Grid {
                lut,
                packed_lut_4: None,
            } => {
                assert!(matches!(spec.bits, 2 | 3));
                score_grid_batch_indexed(
                    codes,
                    code_stride,
                    row_offsets,
                    lut,
                    self.d,
                    spec.bits,
                    out,
                );
                0
            }
        }
    }
}

/// Prepares a full-precision query for every layer.
pub fn prepare_fp_query(query: &[f32], specs: &[LayerSpec]) -> PreparedFpQuery {
    let plan = QueryRotationPlan::new(query.len(), specs);
    prepare_fp_query_with_plan(query, &plan, specs.len())
}

/// Prepares full-precision query coordinates for one schedule prefix.
pub fn prepare_fp_query_with_plan(
    query: &[f32],
    plan: &QueryRotationPlan,
    active_layers: usize,
) -> PreparedFpQuery {
    PreparedFpQuery {
        layers: plan.prepare_layers(query, active_layers),
    }
}

/// Scores encoded layers with a prepared full-precision query.
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
///
/// # Panics
///
/// Panics when encoded, query, schedule, grid, or dimension shapes disagree.
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
    assert!(d > 0);
    assert_eq!(encoded.codes.len(), specs.len());
    assert_eq!(encoded.scales.len(), specs.len());
    assert_eq!(query.layers.len(), specs.len());
    assert_eq!(grids.len(), specs.len());
    specs
        .iter()
        .zip(grids)
        .enumerate()
        .map(|(layer, (spec, grid))| match spec.kind {
            LayerKind::Sign => {
                let words = aligned_le_words(&encoded.codes[layer]);
                let estimate =
                    encoded.scales[layer] * estimate_sign_fp(words.as_ref(), &query.layers[layer]);
                estimate
            }
            LayerKind::Grid => {
                let lut = build_lut(&query.layers[layer], &grid.points, spec.bits);
                encoded.scales[layer] * score_grid(&encoded.codes[layer], &lut, d, spec.bits)
            }
        })
        .collect()
}

/// Reconstructs the encoded prefix in the original coordinate space.
///
/// # Panics
///
/// Panics when encoded, schedule, grid, or dimension shapes disagree.
pub fn reconstruct_first_space(
    encoded: &Encoded,
    specs: &[LayerSpec],
    grids: &[Grid],
    d: usize,
) -> Vec<f32> {
    assert!(d > 0);
    assert_eq!(encoded.codes.len(), specs.len());
    assert_eq!(encoded.scales.len(), specs.len());
    assert_eq!(grids.len(), specs.len());
    let mut total = vec![0.0; d];
    for layer in 0..specs.len() {
        let spec = specs[layer];
        let mut reconstruction: Vec<f32> = match spec.kind {
            LayerKind::Sign => {
                let words = aligned_le_words(&encoded.codes[layer]);
                let signs = unpack_sign(words.as_ref(), d);
                signs
                    .into_iter()
                    .map(|sign| encoded.scales[layer] * sign)
                    .collect()
            }
            LayerKind::Grid => unpack_grid(&encoded.codes[layer], d, spec.bits)
                .into_iter()
                .map(|code| encoded.scales[layer] * grids[layer].points[code as usize])
                .collect(),
        };
        for earlier_layer in (1..=layer).rev() {
            if specs[earlier_layer].rotation != Rotation::None {
                specs[earlier_layer]
                    .rotation
                    .build(d)
                    .unwrap()
                    .apply_inverse(&mut reconstruction);
            }
        }
        for (sum, value) in total.iter_mut().zip(reconstruction) {
            *sum += value;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::cascade::test_support::*;
    use crate::cascade::*;
    use crate::grid::encode_f32 as encode_grid;
    use crate::model::{build_grid, empirical_sigma, isotropic_sigma};
    use crate::sign::encode_f32 as encode_sign;

    #[test]
    fn reusable_rotation_plan_matches_direct_query_preparation() {
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
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.019).cos()).collect();
        let plan = QueryRotationPlan::new(d, &specs);

        let mut direct_current = query.clone();
        let mut direct_layers = Vec::new();
        for spec in &specs {
            if spec.rotation != Rotation::None {
                spec.rotation.build(d).unwrap().apply(&mut direct_current);
            }
            direct_layers.push(direct_current.clone());
        }
        let planned_fp = prepare_fp_query_with_plan(&query, &plan, specs.len());
        assert_eq!(planned_fp.layers, direct_layers);
        assert_eq!(prepare_fp_query(&query, &specs).layers, planned_fp.layers);

        let encoded = encode_layers(&query, None, &specs, &grids);
        let direct_split = prepare_split_query(&query, &specs, &grids, 4);
        let planned_split = prepare_split_query_with_plan(&query, &plan, &grids, 4);
        for layer in 0..specs.len() {
            assert_eq!(
                direct_split.score_layer(
                    layer,
                    &encoded.codes[layer],
                    encoded.scales[layer],
                    0.0,
                    specs[layer],
                ),
                planned_split.score_layer(
                    layer,
                    &encoded.codes[layer],
                    encoded.scales[layer],
                    0.0,
                    specs[layer],
                ),
            );
        }
    }

    #[test]
    fn rotation_plan_prepares_only_active_prefix() {
        let d = 100;
        let specs = [
            LayerSpec {
                kind: LayerKind::Sign,
                bits: 1,
                rotation: Rotation::SeededFhtChaCha8 { seed: 11 },
            },
            LayerSpec {
                kind: LayerKind::Grid,
                bits: 2,
                rotation: Rotation::SeededFhtChaCha8 { seed: 22 },
            },
            LayerSpec {
                kind: LayerKind::Grid,
                bits: 4,
                rotation: Rotation::SeededFhtChaCha8 { seed: 33 },
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 2), build_grid(d, 4)];
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.023).sin()).collect();
        let plan = QueryRotationPlan::new(d, &specs);

        #[cfg(debug_assertions)]
        crate::rotation::debug_reset_apply_count();
        let fp = prepare_fp_query_with_plan(&query, &plan, 1);
        let split = prepare_split_query_with_plan(&query, &plan, &grids[..1], 4);

        assert_eq!(fp.layers.len(), 1);
        assert_eq!(split.layers.len(), 1);
        #[cfg(debug_assertions)]
        assert_eq!(crate::rotation::debug_apply_count(), 2);
        let mut expected = query;
        specs[0].rotation.build(d).unwrap().apply(&mut expected);
        assert_eq!(fp.layers[0], expected);
    }

    #[test]
    fn rotation_plan_preserves_skipped_layers_and_prefix_split_scores() {
        let d = 100;
        let specs = [
            LayerSpec {
                kind: LayerKind::Sign,
                bits: 1,
                rotation: Rotation::SeededFhtChaCha8 { seed: 11 },
            },
            LayerSpec {
                kind: LayerKind::Grid,
                bits: 2,
                rotation: Rotation::None,
            },
            LayerSpec {
                kind: LayerKind::Grid,
                bits: 4,
                rotation: Rotation::SeededFhtChaCha8 { seed: 33 },
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 2), build_grid(d, 4)];
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.029).cos()).collect();
        let plan = QueryRotationPlan::new(d, &specs);

        let mut layer_0 = query.clone();
        specs[0].rotation.build(d).unwrap().apply(&mut layer_0);
        let layer_1 = layer_0.clone();
        let mut layer_2 = layer_1.clone();
        specs[2].rotation.build(d).unwrap().apply(&mut layer_2);

        let prefix_fp = prepare_fp_query_with_plan(&query, &plan, 2);
        assert_eq!(prefix_fp.layers, [layer_0.clone(), layer_1.clone()]);
        assert_eq!(prefix_fp.layers[0], prefix_fp.layers[1]);

        let full_fp = prepare_fp_query_with_plan(&query, &plan, specs.len());
        assert_eq!(full_fp.layers, [layer_0, layer_1, layer_2]);

        let encoded = encode_layers(&query, None, &specs, &grids);
        let prefix_split = prepare_split_query_with_plan(&query, &plan, &grids[..2], 4);
        let full_split = prepare_split_query_with_plan(&query, &plan, &grids, 4);
        for layer in 0..2 {
            assert_eq!(
                prefix_split.score_layer(
                    layer,
                    &encoded.codes[layer],
                    encoded.scales[layer],
                    0.0,
                    specs[layer],
                ),
                full_split.score_layer(
                    layer,
                    &encoded.codes[layer],
                    encoded.scales[layer],
                    0.0,
                    specs[layer],
                ),
            );
        }
    }

    #[test]
    #[should_panic]
    fn split_query_rejects_mismatched_schedule_lengths() {
        let d = 64;
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
        prepare_split_query(&vec![0.25; d], &specs, &grids[..1], 4);
    }

    #[test]
    fn layered_space_bookkeeping_is_exact() {
        let d = 128;
        let rotation = FhtRotation::new(d, 22);
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

    /// Unaligned sign codes are decoded into owned words and counted; aligned ones are borrowed.
    #[test]
    fn batch_scoring_returns_sign_word_fallbacks() {
        let d = 128;
        let spec = LayerSpec {
            kind: LayerKind::Sign,
            bits: 1,
            rotation: Rotation::SeededFhtChaCha8 { seed: 11 },
        };
        let prepared = prepare_split_query(
            &(0..d).map(|i| (i as f32 * 0.019).cos()).collect::<Vec<_>>(),
            &[spec],
            &[build_grid(d, 1)],
            4,
        );
        let (rows, stride) = (3, d / 8);
        let codes: Vec<u8> = (0..rows * stride).map(|i| (i * 37 % 251) as u8).collect();
        let mut storage = vec![0_u8; codes.len() + 8];
        let mut scores = Vec::new();
        for aligned in [true, false] {
            let offset = (0..8)
                .find(|offset| ((storage.as_ptr() as usize + offset) % 8 == 0) == aligned)
                .expect("eight consecutive offsets hold both aligned and unaligned starts");
            storage[offset..offset + codes.len()].copy_from_slice(&codes);
            let placed = &storage[offset..offset + codes.len()];
            let mut dense = vec![0.0; rows];
            let fallbacks =
                prepared.score_layer_batch_unscaled(0, placed, stride, spec, &mut dense);
            assert_eq!(fallbacks, u64::from(!aligned));
            let mut indexed = vec![0.0; 2];
            let fallbacks = prepared.score_layer_batch_unscaled_indexed(
                0,
                placed,
                stride,
                &[0, 2],
                spec,
                &mut indexed,
            );
            assert_eq!(fallbacks, u64::from(!aligned));
            assert_eq!(indexed, [dense[0], dense[2]]);
            scores.push(dense);
        }
        assert_eq!(scores[0], scores[1]);
    }

    #[test]
    fn batch_scoring_matches_one_row_calls_for_sign_and_grid() {
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
        let centroid: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013).sin() * 0.1).collect();
        let rows = 7;
        let mut vectors: Vec<f32> = (0..rows)
            .flat_map(|row| {
                centroid
                    .iter()
                    .enumerate()
                    .map(move |(i, &center)| center + ((i + row) as f32 * 0.031).cos() * 0.2)
            })
            .collect();
        let encoded = encode_batch_in_place(
            &mut vectors,
            rows,
            &prepare_centroid(&centroid, &specs),
            &specs,
            &grids,
        );
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.019).cos()).collect();
        let prepared = prepare_split_query(&query, &specs, &grids, 4);
        let unit_scale = 1.0;

        for (layer, encoded) in encoded.layers.iter().enumerate() {
            let stride = encoded.codes.len() / rows;
            let mut batch = vec![0.0; rows];
            prepared.score_layer_batch_unscaled(
                layer,
                &encoded.codes,
                stride,
                specs[layer],
                &mut batch,
            );
            for (row, &actual) in batch.iter().enumerate() {
                let expected = prepared.score_layer(
                    layer,
                    &encoded.codes[row * stride..(row + 1) * stride],
                    unit_scale,
                    0.0,
                    specs[layer],
                );
                assert_relative_eq(actual, expected);
            }
        }
    }

    #[test]
    fn indexed_batch_scoring_matches_dense_for_all_widths_and_dimensions() {
        const ROWS: usize = 11;
        let row_offsets = [10, 2, 7, 2, 0, 9, 4, 6, 1, 8];

        for d in [64, 65] {
            let query: Vec<f32> = (0..d).map(|i| ((i as f32 + 0.25) * 0.019).cos()).collect();
            for bits in 1..=4 {
                let spec = LayerSpec {
                    kind: if bits == 1 {
                        LayerKind::Sign
                    } else {
                        LayerKind::Grid
                    },
                    bits,
                    rotation: Rotation::SeededFhtChaCha8 {
                        seed: 0x00b2_1d00 + u64::from(bits),
                    },
                };
                let grid = build_grid(d, bits);
                let prepared = prepare_split_query(&query, &[spec], &[grid.clone()], 4);
                if bits == 4 {
                    assert!(matches!(
                        &prepared.layers[0],
                        PreparedSplitLayer::Grid {
                            packed_lut_4: Some(_),
                            ..
                        }
                    ));
                }

                let code_stride = if bits == 1 {
                    d.div_ceil(64) * std::mem::size_of::<u64>()
                } else {
                    crate::grid::packed_len(d, bits)
                };
                let mut codes = Vec::with_capacity(ROWS * code_stride);
                for row in 0..ROWS {
                    let values: Vec<f32> = (0..d)
                        .map(|i| {
                            let phase = (i * 13 + row * 29 + bits as usize * 7) as f32;
                            (phase * 0.017).sin() + (phase * 0.031).cos() * 0.25
                        })
                        .collect();
                    if bits == 1 {
                        let mut words = vec![0_u64; d.div_ceil(64)];
                        encode_sign(&values, &mut words);
                        codes.extend_from_slice(&words_to_bytes(&words));
                    } else {
                        let start = codes.len();
                        codes.resize(start + code_stride, 0);
                        encode_grid(
                            &values,
                            &grid.points,
                            bits,
                            &mut codes[start..start + code_stride],
                        );
                    }
                }

                let mut dense = vec![0.0; ROWS];
                prepared.score_layer_batch_unscaled(0, &codes, code_stride, spec, &mut dense);
                let mut indexed = vec![0.0; row_offsets.len()];
                prepared.score_layer_batch_unscaled_indexed(
                    0,
                    &codes,
                    code_stride,
                    &row_offsets,
                    spec,
                    &mut indexed,
                );
                for (&row, &actual) in row_offsets.iter().zip(&indexed) {
                    assert_relative_eq(actual, dense[row]);
                }
            }
        }
    }

    #[test]
    fn layered_golden_and_sigma() {
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
            specs[0].rotation.build(d).unwrap().apply(&mut first_space);
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
        let modeled = isotropic_sigma(grids[0].rho_model * grids[1].rho_model, d);
        assert!(
            (empirical / modeled - 1.0).abs() <= 0.05,
            "empirical={empirical}, model={modeled}"
        );
    }

    #[test]
    fn odd_dimension_layered_rho_goldens() {
        for (d, rho_golden) in [
            (64, 0.057_061_499),
            (65, 0.056_939_250),
            (100, 0.057_902_796),
            (300, 0.058_633_208),
            (769, 0.058_493_834),
        ] {
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
            let model_rho = grids[0].rho_model * grids[1].rho_model;
            let mut rng = ChaCha8Rng::seed_from_u64(0x004f_4444_4449_4d00 ^ d as u64);
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
                specs[0].rotation.build(d).unwrap().apply(&mut first_space);
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
            let measured_rho = (error_energy / signal_energy).sqrt();
            let sigma_ratio = empirical_sigma(&estimates, &truths) / isotropic_sigma(model_rho, d);
            println!(
                "d={d} measured_rho={measured_rho:.9} model_rho={model_rho:.9} \
                 sigma_ratio={sigma_ratio:.9}"
            );
            assert!(
                (measured_rho / model_rho - 1.0).abs() <= 0.08,
                "d={d}: measured rho {measured_rho}, model rho {model_rho}"
            );
            assert!(
                (measured_rho - rho_golden).abs() <= 0.004,
                "d={d}: measured rho {measured_rho}, golden {rho_golden}"
            );
            assert!(
                (sigma_ratio - 1.0).abs() <= 0.05,
                "d={d}: sigma ratio {sigma_ratio}"
            );
        }
    }

    #[test]
    fn split_form_matches_direct_form_per_layer_and_summed() {
        let d = 768;
        let schedules = [
            [
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
            ],
            [
                LayerSpec {
                    kind: LayerKind::Sign,
                    bits: 1,
                    rotation: Rotation::SeededFhtChaCha8 { seed: 11 },
                },
                LayerSpec {
                    kind: LayerKind::Grid,
                    bits: 4,
                    rotation: Rotation::None,
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
}
