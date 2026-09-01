//! Residual quantization cascade and plane-boundary operations.

use std::borrow::Cow;

use fht::Rotation;
use grid_plane::{
    build_lut, build_packed_lut_4, decode as decode_grid, encode as encode_grid,
    estimate as estimate_grid, score_batch as score_grid_batch,
    score_batch_indexed as score_grid_batch_indexed,
    score_batch_packed_4 as score_grid_batch_packed_4,
    score_batch_packed_4_indexed as score_grid_batch_packed_4_indexed,
};
use quant_model::f16::f16_to_f32;
use quant_model::Grid;
use sign_plane::{
    encode as encode_sign, estimate_asym as estimate_sign_asym,
    estimate_asym_batch_unscaled as estimate_sign_batch,
    estimate_asym_batch_unscaled_indexed as estimate_sign_batch_indexed,
    estimate_fp as estimate_sign_fp, prepare_query, unpack as unpack_sign, QueryPlanes,
};

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

/// One layer's row-parallel encoded output for a cluster batch.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedLayerBatch {
    /// Packed row codes, concatenated at the layer's fixed code stride.
    pub codes: Vec<u8>,
    /// One binary16 scale per row.
    pub scales: Vec<u16>,
    /// One binary32 split-form constant per row.
    pub constants: Vec<f32>,
}

/// SoA output from the cluster-scoped batch encoder.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedBatch {
    pub rows: usize,
    pub layers: Vec<EncodedLayerBatch>,
}

/// Cluster-scoped centroid state shared by every residual encoded in the cluster.
#[derive(Clone, Debug)]
pub struct PreparedCentroid {
    d: usize,
    specs: Vec<LayerSpec>,
    original: Vec<f32>,
    layers: Vec<Vec<f32>>,
    rotations: Vec<Option<Rotation>>,
}

#[derive(Clone, Debug)]
pub struct PreparedFpQuery {
    layers: Vec<Vec<f32>>,
}

/// Immutable, reusable rotation state for preparing queries against one
/// quantization schedule.
///
/// Construct this once with the persisted layer seeds, then reuse it for
/// every query. Preparing a prefix applies only the rotations in that prefix.
#[derive(Clone, Debug)]
pub struct QueryRotationPlan {
    d: usize,
    specs: Vec<LayerSpec>,
    rotations: Vec<Option<Rotation>>,
}

impl QueryRotationPlan {
    /// Expand every rotating layer's seed into its format-stable rotation.
    pub fn new(d: usize, specs: &[LayerSpec]) -> Self {
        validate_specs(d, specs);
        let rotations = specs
            .iter()
            .enumerate()
            .map(|(layer, spec)| (layer == 0 || spec.rotate).then(|| Rotation::new(d, spec.seed)))
            .collect();
        Self {
            d,
            specs: specs.to_vec(),
            rotations,
        }
    }

    pub fn dimension(&self) -> usize {
        self.d
    }

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
    pub fn final_layer(&self) -> &[f32] {
        self.layers
            .last()
            .expect("validated quantization schedule is non-empty")
    }
}

enum PreparedSplitLayer {
    Sign(QueryPlanes),
    Grid {
        lut: Vec<f32>,
        packed_lut_4: Option<Vec<f32>>,
    },
}

/// Segment-query state with all rotations, sign bitplanes, and grid LUTs hoisted.
pub struct PreparedSplitQuery {
    d: usize,
    layers: Vec<PreparedSplitLayer>,
}

pub fn prepare_split_query(
    query: &[f32],
    specs: &[LayerSpec],
    grids: &[Grid],
    sign_query_bits: u8,
) -> PreparedSplitQuery {
    // Preserve the original harness API contract: only the explicit
    // `_with_plan` entry accepts a grid slice as an active prefix.
    validate(query.len(), specs, grids);
    let plan = QueryRotationPlan::new(query.len(), specs);
    prepare_split_query_with_plan(query, &plan, grids, sign_query_bits)
}

/// Prepare a query using pre-expanded rotations from `plan`.
///
/// `grids` selects the active schedule prefix: passing `&all_grids[..depth]`
/// prepares exactly `depth` layers.
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
        if spec.bits == 1 {
            layers.push(PreparedSplitLayer::Sign(prepare_query(
                &current,
                sign_query_bits,
            )));
        } else {
            let lut = build_lut(&current, &grid.points, spec.bits);
            let packed_lut_4 = (spec.bits == 4).then(|| build_packed_lut_4(&lut, query.len()));
            layers.push(PreparedSplitLayer::Grid { lut, packed_lut_4 });
        }
    }
    PreparedSplitQuery {
        d: query.len(),
        layers,
    }
}

impl PreparedSplitQuery {
    /// Score one stored layer as `kernel * scale - split_constant`.
    pub fn score_layer(
        &self,
        layer: usize,
        codes: &[u8],
        scale: u16,
        constant: f32,
        spec: LayerSpec,
    ) -> f32 {
        match &self.layers[layer] {
            PreparedSplitLayer::Sign(query) => {
                assert_eq!(spec.bits, 1);
                let words = aligned_le_words(codes);
                let estimate = estimate_sign_asym(words.as_ref(), scale, query);
                estimate - constant
            }
            PreparedSplitLayer::Grid { lut, .. } => {
                assert!(spec.bits > 1);
                estimate_grid(codes, scale, lut, self.d, spec.bits) - constant
            }
        }
    }

    /// Score a fixed-stride batch without scales or split constants. This is
    /// the scan-path kernel boundary: one call covers one cluster or one
    /// densely gathered survivor stream.
    #[inline(always)]
    pub fn score_layer_batch_unscaled(
        &self,
        layer: usize,
        codes: &[u8],
        code_stride: usize,
        spec: LayerSpec,
        out: &mut [f32],
    ) {
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
            }
            PreparedSplitLayer::Grid { lut, packed_lut_4 } => {
                assert!(spec.bits > 1);
                if let Some(packed_lut_4) = packed_lut_4 {
                    score_grid_batch_packed_4(codes, code_stride, packed_lut_4, self.d, out);
                } else {
                    score_grid_batch(codes, code_stride, lut, self.d, spec.bits, out);
                }
            }
        }
    }

    /// Score selected rows from one borrowed contiguous posting range. This
    /// is the sparse survivor-stream equivalent of the fixed-stride batch
    /// entry. Every supported layer width remains borrowed in place; b=4
    /// uses the packed LUT fast path while sign and b=2/b=3 use their indexed
    /// kernels directly.
    #[inline(always)]
    pub fn score_layer_batch_unscaled_indexed(
        &self,
        layer: usize,
        codes: &[u8],
        code_stride: usize,
        row_offsets: &[usize],
        spec: LayerSpec,
        out: &mut [f32],
    ) {
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
            }
        }
    }
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
        original: centroid.to_vec(),
        layers,
        rotations,
    }
}

/// Encode a row-major cluster tile using at most two `rows * d` f32 buffers.
///
/// `vectors` is consumed as scratch: the function first subtracts the prepared
/// centroid, then carries the residual through the rotated layer chain. The
/// output is layer-separated SoA in the same row order as the input.
pub fn encode_batch_in_place(
    vectors: &mut [f32],
    rows: usize,
    centroid: &PreparedCentroid,
    specs: &[LayerSpec],
    grids: &[Grid],
) -> EncodedBatch {
    encode_batch_in_place_with_residual_observer(
        vectors,
        rows,
        centroid,
        specs,
        grids,
        |_, _, _| {},
    )
}

/// Encode a row-major cluster tile and expose the remaining residual after
/// each prefix to a non-owning observer. The observer runs before the next
/// layer's rotation, so its residual and scale stream are in that prefix's
/// exact scoring coordinate space.
pub fn encode_batch_in_place_with_residual_observer<F>(
    vectors: &mut [f32],
    rows: usize,
    centroid: &PreparedCentroid,
    specs: &[LayerSpec],
    grids: &[Grid],
    mut observer: F,
) -> EncodedBatch
where
    F: FnMut(usize, &[f32], &[u16]),
{
    validate(centroid.d, specs, grids);
    assert_eq!(centroid.specs, specs);
    assert_eq!(vectors.len(), rows * centroid.d);

    let d = centroid.d;
    for row in vectors.chunks_exact_mut(d) {
        for (value, &center) in row.iter_mut().zip(&centroid.original) {
            *value -= center;
        }
    }

    let mut reconstruction = vec![0.0_f32; vectors.len()];
    let mut layers = Vec::with_capacity(specs.len());
    for (layer, (spec, grid)) in specs.iter().zip(grids).enumerate() {
        if layer == 0 || spec.rotate {
            let rotation = centroid.rotations[layer]
                .as_ref()
                .expect("rotating layer must have a prepared rotation");
            for row in vectors.chunks_exact_mut(d) {
                rotation.apply(row);
            }
        }

        reconstruction.fill(0.0);
        let code_stride = if spec.bits == 1 {
            d.div_ceil(64) * 8
        } else {
            grid_plane::packed_len(d, spec.bits)
        };
        let mut codes = Vec::with_capacity(rows * code_stride);
        let mut scales = Vec::with_capacity(rows);

        for (residual, reconstructed) in vectors
            .chunks_exact(d)
            .zip(reconstruction.chunks_exact_mut(d))
        {
            if spec.bits == 1 {
                let mut words = vec![0_u64; d.div_ceil(64)];
                let scale = encode_sign(residual, &mut words);
                let reconstruction_scale = f16_to_f32(scale);
                for (i, value) in reconstructed.iter_mut().enumerate() {
                    let sign = if words[i / 64] & (1_u64 << (i % 64)) != 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    *value = reconstruction_scale * sign;
                }
                for word in words {
                    codes.extend_from_slice(&word.to_le_bytes());
                }
                scales.push(scale);
            } else {
                let code_start = codes.len();
                codes.resize(code_start + code_stride, 0);
                let scale =
                    encode_grid(residual, &grid.points, spec.bits, &mut codes[code_start..]);
                reconstructed.copy_from_slice(&decode_grid(
                    &codes[code_start..],
                    &grid.points,
                    d,
                    spec.bits,
                    scale,
                ));
                scales.push(scale);
            }
        }

        let constants = reconstruction
            .chunks_exact(d)
            .map(|row| {
                centroid.layers[layer]
                    .iter()
                    .zip(row)
                    .map(|(&center, &value)| center * value)
                    .sum()
            })
            .collect();

        for (residual, reconstructed) in vectors
            .chunks_exact_mut(d)
            .zip(reconstruction.chunks_exact(d))
        {
            for (value, &encoded) in residual.iter_mut().zip(reconstructed) {
                *value -= encoded;
            }
        }

        observer(layer, vectors, &scales);

        layers.push(EncodedLayerBatch {
            codes,
            scales,
            constants,
        });
    }

    EncodedBatch { rows, layers }
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
            let mut words = vec![0_u64; r.len().div_ceil(64)];
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
    let plan = QueryRotationPlan::new(query.len(), specs);
    prepare_fp_query_with_plan(query, &plan, specs.len())
}

/// Prepare full-precision query coordinates for one active schedule prefix
/// using pre-expanded rotations from `plan`.
pub fn prepare_fp_query_with_plan(
    query: &[f32],
    plan: &QueryRotationPlan,
    active_layers: usize,
) -> PreparedFpQuery {
    PreparedFpQuery {
        layers: plan.prepare_layers(query, active_layers),
    }
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
    assert!(d > 0);
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
                let words = aligned_le_words(&encoded.codes[layer]);
                let estimate =
                    estimate_sign_fp(words.as_ref(), encoded.scales[layer], &query.layers[layer]);
                estimate
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
    assert!(d > 0);
    assert_eq!(encoded.codes.len(), specs.len());
    assert_eq!(encoded.scales.len(), specs.len());
    assert_eq!(grids.len(), specs.len());
    let mut total = vec![0.0; d];
    for layer in 0..specs.len() {
        let spec = specs[layer];
        let mut reconstruction = if spec.bits == 1 {
            let scale = f16_to_f32(encoded.scales[layer]);
            let words = aligned_le_words(&encoded.codes[layer]);
            let signs = unpack_sign(words.as_ref(), d);
            signs.into_iter().map(|sign| scale * sign).collect()
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
    assert!(d > 0);
    assert!((1..=4).contains(&specs.len()));
    assert!(specs[0].rotate, "layer 0 must rotate");
}

fn words_to_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[inline]
fn aligned_le_words(bytes: &[u8]) -> Cow<'_, [u64]> {
    assert_eq!(bytes.len() % std::mem::size_of::<u64>(), 0);

    #[cfg(target_endian = "little")]
    {
        // SAFETY: every bit pattern is a valid `u64`. `align_to` returns only
        // views within `bytes`, and borrowing is allowed only when the entire
        // input is represented by exactly the expected number of words.
        let (prefix, words, suffix) = unsafe { bytes.align_to::<u64>() };
        if prefix.is_empty()
            && suffix.is_empty()
            && words.len() == bytes.len() / std::mem::size_of::<u64>()
        {
            return Cow::Borrowed(words);
        }
    }

    Cow::Owned(decode_le_words(bytes))
}

fn decode_le_words(bytes: &[u8]) -> Vec<u64> {
    assert_eq!(bytes.len() % 8, 0);
    let chunks = bytes.chunks_exact(8);
    assert!(chunks.remainder().is_empty());
    chunks
        .map(|chunk| {
            let bytes: [u8; 8] = chunk
                .try_into()
                .expect("chunk size is fixed at eight bytes");
            u64::from_le_bytes(bytes)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use quant_model::{build_grid, empirical_sigma, sigma_from_rho, DEFAULT_CAL};
    use rand_chacha::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;

    #[cfg(target_endian = "little")]
    #[test]
    fn aligned_le_words_borrows_aligned_input() {
        let expected = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
        // SAFETY: every `u64` byte is a valid `u8`, and `u8` has alignment one.
        let (prefix, bytes, suffix) = unsafe { expected.as_slice().align_to::<u8>() };
        assert!(prefix.is_empty() && suffix.is_empty());

        let words = aligned_le_words(bytes);
        assert_eq!(words.as_ref(), expected);
        assert!(matches!(words, Cow::Borrowed(_)));
    }

    #[test]
    fn aligned_le_words_decodes_deliberately_unaligned_input() {
        let expected = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
        let encoded = words_to_bytes(&expected);
        let mut storage = vec![0_u8; encoded.len() + 8];
        let offset = (0..8)
            .find(|offset| (storage.as_ptr() as usize + offset) % std::mem::align_of::<u64>() != 0)
            .expect("at least seven of eight consecutive byte offsets are unaligned");
        storage[offset..offset + encoded.len()].copy_from_slice(&encoded);

        let words = aligned_le_words(&storage[offset..offset + encoded.len()]);
        assert_eq!(words.as_ref(), expected);
        assert!(matches!(words, Cow::Owned(_)));
    }

    #[test]
    fn reusable_rotation_plan_matches_legacy_query_preparation() {
        let d = 100;
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
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.019).cos()).collect();
        let plan = QueryRotationPlan::new(d, &specs);

        let mut legacy_current = query.clone();
        let mut legacy_layers = Vec::new();
        for (layer, spec) in specs.iter().enumerate() {
            if layer == 0 || spec.rotate {
                Rotation::new(d, spec.seed).apply(&mut legacy_current);
            }
            legacy_layers.push(legacy_current.clone());
        }
        let planned_fp = prepare_fp_query_with_plan(&query, &plan, specs.len());
        assert_eq!(planned_fp.layers, legacy_layers);
        assert_eq!(prepare_fp_query(&query, &specs).layers, planned_fp.layers);

        let encoded = encode_layers(&query, None, &specs, &grids);
        let legacy_split = prepare_split_query(&query, &specs, &grids, 4);
        let planned_split = prepare_split_query_with_plan(&query, &plan, &grids, 4);
        for layer in 0..specs.len() {
            assert_eq!(
                legacy_split.score_layer(
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
                bits: 1,
                seed: 11,
                rotate: true,
            },
            LayerSpec {
                bits: 2,
                seed: 22,
                rotate: true,
            },
            LayerSpec {
                bits: 4,
                seed: 33,
                rotate: true,
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 2), build_grid(d, 4)];
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.023).sin()).collect();
        let plan = QueryRotationPlan::new(d, &specs);

        #[cfg(debug_assertions)]
        fht::debug_reset_apply_count();
        let fp = prepare_fp_query_with_plan(&query, &plan, 1);
        let split = prepare_split_query_with_plan(&query, &plan, &grids[..1], 4);

        assert_eq!(fp.layers.len(), 1);
        assert_eq!(split.layers.len(), 1);
        #[cfg(debug_assertions)]
        assert_eq!(fht::debug_apply_count(), 2);
        let mut expected = query;
        Rotation::new(d, specs[0].seed).apply(&mut expected);
        assert_eq!(fp.layers[0], expected);
    }

    #[test]
    fn rotation_plan_preserves_skipped_layers_and_prefix_split_scores() {
        let d = 100;
        let specs = [
            LayerSpec {
                bits: 1,
                seed: 11,
                rotate: true,
            },
            LayerSpec {
                bits: 2,
                seed: 22,
                rotate: false,
            },
            LayerSpec {
                bits: 4,
                seed: 33,
                rotate: true,
            },
        ];
        let grids = [build_grid(d, 1), build_grid(d, 2), build_grid(d, 4)];
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.029).cos()).collect();
        let plan = QueryRotationPlan::new(d, &specs);

        let mut layer_0 = query.clone();
        Rotation::new(d, specs[0].seed).apply(&mut layer_0);
        let layer_1 = layer_0.clone();
        let mut layer_2 = layer_1.clone();
        Rotation::new(d, specs[2].seed).apply(&mut layer_2);

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
    fn legacy_split_query_rejects_an_implicit_grid_prefix() {
        let d = 64;
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
        prepare_split_query(&vec![0.25; d], &specs, &grids[..1], 4);
    }

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
    fn cluster_batch_matches_row_encoder_byte_for_byte() {
        let d = 128;
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
        let mut row_major: Vec<f32> = vectors.into_iter().flatten().collect();
        let actual =
            encode_batch_in_place(&mut row_major, expected.len(), &prepared, &specs, &grids);

        assert_eq!(actual.rows, expected.len());
        for (layer, batch) in actual.layers.iter().enumerate() {
            let expected_codes: Vec<u8> = expected
                .iter()
                .flat_map(|encoded| encoded.codes[layer].iter().copied())
                .collect();
            let expected_scales: Vec<u16> = expected
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
        }
    }

    #[test]
    fn residual_observer_reports_each_exact_prefix_error() {
        let d = 100;
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
    fn batch_scoring_matches_one_row_calls_for_sign_and_grid() {
        let d = 100;
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
        let unit_scale = quant_model::f16::f32_to_f16(1.0);

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
                    bits,
                    seed: 0x00b2_1d00 + u64::from(bits),
                    rotate: true,
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
                    grid_plane::packed_len(d, bits)
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
            let measured_rho = (error_energy / signal_energy).sqrt();
            let sigma_ratio =
                empirical_sigma(&estimates, &truths) / sigma_from_rho(model_rho, d, DEFAULT_CAL);
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
