//! Residual quantization cascade and layer-boundary operations.

use std::borrow::Cow;
use std::sync::Arc;

use fht::Rotation;
use grid_plane::{
    build_lut, build_packed_lut_4, encode_f32 as encode_grid,
    encode_f32_with_scratch as encode_grid_with_scratch, score as score_grid,
    score_batch as score_grid_batch, score_batch_indexed as score_grid_batch_indexed,
    score_batch_packed_4 as score_grid_batch_packed_4,
    score_batch_packed_4_indexed as score_grid_batch_packed_4_indexed, unpack as unpack_grid,
};
use quant_model::f16::{f16_to_f32, f32_to_f16};
use quant_model::Grid;
use sign_plane::{
    encode_f32 as encode_sign, estimate_asym_batch_unscaled as estimate_sign_batch,
    estimate_asym_batch_unscaled_indexed as estimate_sign_batch_indexed,
    estimate_asym_unscaled as estimate_sign_asym, estimate_fp_unscaled as estimate_sign_fp,
    pack as pack_sign, prepare_query, unpack as unpack_sign, QueryPlanes,
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
    pub scales: Vec<f32>,
    pub constants: Vec<f32>,
}

/// One layer's row-parallel encoded output for a cluster batch.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedLayerBatch {
    /// Packed row codes, concatenated at the layer's fixed code stride.
    pub codes: Vec<u8>,
    /// One exact binary32 scale per row.
    pub scales: Vec<f32>,
    /// One binary16 cumulative-prefix gamma per row.
    ///
    /// Gamma is exact encode-side metadata for an encoded prefix:
    /// `||r0||^2 / <r0, rhat_prefix>`, clamped to `[1, 4]` before the f16
    /// round trip.
    pub gammas: Vec<u16>,
    /// One binary16 corrected-estimator error ratio per row.
    ///
    /// `E = ||r0 - gamma_stored * rhat_prefix||^2 / ||r0||^2`, where
    /// `gamma_stored` is the decoded post-clamp binary16 coefficient used by
    /// scoring. A zero original residual has the canonical value zero.
    /// Nonzero values are converted directly to binary16 without clamping.
    pub corrected_error_ratios: Vec<u16>,
    /// One binary32 split-form constant per row.
    pub constants: Vec<f32>,
}

/// SoA output from the cluster-scoped batch encoder.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedBatch {
    pub rows: usize,
    /// One shared binary32 `R0²` value per row.
    ///
    /// This is both the value serialized in the radius slot and the numerator
    /// and denominator anchor used for cumulative gamma and corrected error.
    pub residual_norms_squared: Vec<f32>,
    pub layers: Vec<EncodedLayerBatch>,
}

/// Reusable storage for cluster-tile encoding.
///
/// Construct one workspace for a merge worker and reuse it across tiles and
/// clusters with the same dimension. The encoder clears lengths between calls
/// while retaining every scratch and output allocation.
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

    /// Preallocate every tile-sized scratch and output buffer.
    ///
    /// Encoding up to `max_rows` with this dimension and schedule does not
    /// grow any workspace allocation.
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
            let code_stride = if spec.bits == 1 {
                d.div_ceil(64) * 8
            } else {
                grid_plane::packed_len(d, spec.bits)
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

/// One cumulative-prefix correction before and after the storage-shaped f16
/// round trip. This is audit output only; no production encoder consumes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaRoundTrip {
    pub raw: f64,
    pub clamped: f32,
    pub f16: u16,
}

impl GammaRoundTrip {
    pub fn f16_value(self) -> f32 {
        f16_to_f32(self.f16)
    }
}

/// One corrected-estimator error ratio before and after binary16 storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectedErrorRatioRoundTrip {
    pub raw: f64,
    pub f16: u16,
}

impl CorrectedErrorRatioRoundTrip {
    pub fn f16_value(self) -> f32 {
        f16_to_f32(self.f16)
    }
}

/// Audit data for one cumulative encoded prefix. Reconstructions and inner
/// products are expressed in the original, unrotated residual coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixErrorAudit {
    pub gamma: GammaRoundTrip,
    pub corrected_error_ratio: CorrectedErrorRatioRoundTrip,
    pub prefix_dot: f64,
    pub layer_scale: f32,
    pub codes: Vec<u8>,
    pub raw_prefix_reconstruction: Vec<f32>,
}

/// Non-persisting metadata audit for every active encoded prefix.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixErrorAuditResult {
    pub norm_sq: f64,
    pub prefixes: Vec<PrefixErrorAudit>,
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

/// Reusable cluster-centroid transform and output storage.
///
/// A prepared centroid is immutable while its cluster's tiles are encoded.
/// Calling [`Self::prepare`] for the next cluster overwrites the same buffers
/// after the previous borrowed view has gone out of scope.
#[derive(Debug)]
pub struct PreparedCentroidWorkspace {
    current: Vec<f32>,
    rotation_scratch: Vec<f32>,
    prepared: PreparedCentroid,
}

impl PreparedCentroidWorkspace {
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

#[derive(Clone, Debug)]
pub struct PreparedFpQuery {
    layers: Vec<Vec<f32>>,
}

/// Immutable, reusable rotation state for preparing queries and centroids
/// against one quantization schedule.
///
/// Construct this once with the persisted layer seeds, then reuse it for every
/// query and cluster centroid. Preparing a prefix applies only its rotations.
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
    // The direct entry prepares the complete supplied schedule.
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

/// Audit-only exact squared query-quantization error for every prepared layer.
/// Sign layers report the error accumulated while producing the affine
/// bitplanes consumed by the scoring kernel; grid layers use the rotated f32
/// query directly and report zero.
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

/// [`audit_split_query_layer_error_squared`] using already expanded rotations.
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

impl PreparedSplitQuery {
    /// Exact squared query-quantization error `B_j` for one prepared layer.
    /// Grid layers score the f32 query directly and therefore return zero.
    pub fn query_error_squared(&self, layer: usize) -> f64 {
        match &self.layers[layer] {
            PreparedSplitLayer::Sign(query) => query.error_squared(),
            PreparedSplitLayer::Grid { .. } => 0.0,
        }
    }

    /// Score one stored layer as `kernel * scale - split_constant`.
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

    /// Score one stored layer for metrics whose format omits split constants.
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
    let plan = Arc::new(QueryRotationPlan::new(centroid.len(), specs));
    prepare_centroid_with_plan(centroid, plan)
}

/// Prepare one cluster centroid using schedule rotations expanded once by the
/// caller. Every returned centroid shares the plan's permutation and sign
/// buffers through `Arc`.
pub fn prepare_centroid_with_plan(
    centroid: &[f32],
    plan: Arc<QueryRotationPlan>,
) -> PreparedCentroid {
    let mut workspace = PreparedCentroidWorkspace::new(plan);
    workspace.prepare(centroid);
    workspace.prepared
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

/// Encode a row-major cluster tile while retaining all scratch and output
/// allocations in `workspace` for the next tile or cluster.
///
/// The returned batch borrows the workspace and remains valid until the next
/// call using that workspace. Callers that stream the layer slices before the
/// next encode avoid allocating an owned batch per tile.
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

    // Carry the cumulative reconstruction itself through the same rotation
    // chain as the residual. This buffer is assembled only from the codes and
    // binary32 scales that are serialized, so gamma and E describe the
    // reconstruction the scorer actually consumes. The encoder uses exactly
    // two `rows * d` f32 buffers.
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
        if layer == 0 || spec.rotate {
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

        let code_stride = if spec.bits == 1 {
            d.div_ceil(64) * 8
        } else {
            grid_plane::packed_len(d, spec.bits)
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
            if spec.bits == 1 {
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
            } else {
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
            if let Some(rotation) =
                centroid.and_then(|context| context.rotation_plan.rotations[layer].as_ref())
            {
                rotation.apply(&mut residual);
            } else {
                Rotation::new(r.len(), spec.seed).apply(&mut residual);
            }
        }
        if spec.bits == 1 {
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
        } else {
            let mut codes = vec![0_u8; grid_plane::packed_len(r.len(), spec.bits)];
            let scale = encode_grid(&residual, &grid.points, spec.bits, &mut codes);
            let reconstructed = unpack_grid(&codes, r.len(), spec.bits)
                .into_iter()
                .map(|code| scale * grid.points[code as usize]);
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

fn gamma_round_trip(norm_sq: f64, prefix_dot: f64) -> GammaRoundTrip {
    let raw = if norm_sq == 0.0 && prefix_dot == 0.0 {
        1.0
    } else {
        norm_sq / prefix_dot
    };
    let clamped = raw.clamp(1.0, 4.0) as f32;
    GammaRoundTrip {
        raw,
        clamped,
        f16: f32_to_f16(clamped),
    }
}

/// Storage-shaped squared norm. Sixteen independent binary32 accumulators
/// match Tantivy's vector-distance kernel while keeping this crate standalone.
#[inline]
#[allow(unknown_lints)]
#[allow(clippy::chunks_exact_to_as_chunks)]
fn squared_norm_f32(values: &[f32]) -> f32 {
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

fn corrected_error_ratio_round_trip(
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

/// Re-encode one residual without persisting anything and expose the
/// cumulative gamma measurements used by the external audit harness.
///
/// Each layer advances its residual with the exact binary32 scale used by
/// production encoding. Every contribution is inverse-rotated into `r0`'s
/// original coordinates for reporting. Gamma and the corrected error ratio
/// are measured in the encoder's equivalent cumulatively rotated coordinates.
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

        let (codes, scale, reconstruction) = if spec.bits == 1 {
            let mut words = vec![0_u64; d.div_ceil(64)];
            let scale = encode_sign(&residual, &mut words);
            let signs = unpack_sign(&words, d);
            let reconstruction: Vec<f32> = signs.into_iter().map(|sign| scale * sign).collect();
            (words_to_bytes(&words), scale, reconstruction)
        } else {
            let mut codes = vec![0_u8; grid_plane::packed_len(d, spec.bits)];
            let scale = encode_grid(&residual, &grid.points, spec.bits, &mut codes);
            let unpacked = unpack_grid(&codes, d, spec.bits);
            let reconstruction: Vec<f32> = unpacked
                .into_iter()
                .map(|code| scale * grid.points[code as usize])
                .collect();
            (codes, scale, reconstruction)
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
                    encoded.scales[layer] * estimate_sign_fp(words.as_ref(), &query.layers[layer]);
                estimate
            } else {
                let lut = build_lut(&query.layers[layer], &grid.points, spec.bits);
                encoded.scales[layer] * score_grid(&encoded.codes[layer], &lut, d, spec.bits)
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
        let mut reconstruction: Vec<f32> = if spec.bits == 1 {
            let words = aligned_le_words(&encoded.codes[layer]);
            let signs = unpack_sign(words.as_ref(), d);
            signs
                .into_iter()
                .map(|sign| encoded.scales[layer] * sign)
                .collect()
        } else {
            unpack_grid(&encoded.codes[layer], d, spec.bits)
                .into_iter()
                .map(|code| encoded.scales[layer] * grids[layer].points[code as usize])
                .collect()
        };
        for earlier_layer in (1..=layer).rev() {
            if specs[earlier_layer].rotate {
                Rotation::new(d, specs[earlier_layer].seed).apply_inverse(&mut reconstruction);
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
        assert_eq!(spec.bits, grid.bits);
        assert!(matches!(spec.bits, 1..=4));
    }
}

fn validate_specs(d: usize, specs: &[LayerSpec]) {
    assert!(d > 0);
    assert!((1..=3).contains(&specs.len()));
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
    bytes
        .chunks(8)
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
    use quant_model::{build_grid, empirical_sigma, isotropic_sigma};
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
    fn reusable_rotation_plan_matches_direct_query_preparation() {
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

        let mut direct_current = query.clone();
        let mut direct_layers = Vec::new();
        for (layer, spec) in specs.iter().enumerate() {
            if layer == 0 || spec.rotate {
                Rotation::new(d, spec.seed).apply(&mut direct_current);
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

    #[cfg(debug_assertions)]
    #[test]
    fn centroid_preparation_shares_one_expanded_plan_and_rotates_once_per_cluster() {
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

        fht::debug_reset_apply_count();
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
        assert_eq!(fht::debug_apply_count(), 2 * specs.len());
        assert_eq!(storage_signature(&workspace), allocated);

        let direct = prepare_centroid(&first_values, &specs);
        assert_eq!(first_values, direct.original);
        assert_eq!(first_layers, direct.layers);
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
    fn split_query_rejects_mismatched_schedule_lengths() {
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
        assert_eq!(encoded.scales[1], 0.0);
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
                        bits,
                        seed: 0x4741_4d4d_4100 + layer as u64,
                        rotate: true,
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
    fn serialized_prefix_decode_reconstructs_gamma_error_and_shared_radius() {
        // This test deliberately does not call the metadata audit or the row
        // encoder. It starts from packed batch bytes plus stored f32 scales,
        // decodes every contribution, and independently rebuilds each prefix.
        // The grid-first case proves that neither the identity nor its odd-d
        // tail handling depends on a leading sign layer.
        const D: usize = 100;
        const ROWS: usize = 3;
        for bits in [[1_u8, 4_u8], [2_u8, 4_u8]] {
            let specs: Vec<LayerSpec> = bits
                .iter()
                .enumerate()
                .map(|(layer, &bits)| LayerSpec {
                    bits,
                    seed: 0xDEC0_DE00 + layer as u64,
                    rotate: true,
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

                // Independent copy of the format's sixteen-lane binary32
                // radius accumulation, matching the public vector kernel.
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
                    let rotation = Rotation::new(D, spec.seed);
                    rotation.apply(&mut residual);
                    rotation.apply(&mut prefix_reconstruction);

                    let stride = if spec.bits == 1 {
                        D.div_ceil(64) * std::mem::size_of::<u64>()
                    } else {
                        grid_plane::packed_len(D, spec.bits)
                    };
                    let codes =
                        &batch.layers[layer].codes[row_index * stride..(row_index + 1) * stride];
                    let scale = batch.layers[layer].scales[row_index];
                    let contribution: Vec<f32> = if spec.bits == 1 {
                        (0..D)
                            .map(|i| {
                                let byte = codes[i / 8];
                                let sign = if byte & (1 << (i % 8)) != 0 {
                                    1.0
                                } else {
                                    -1.0
                                };
                                scale * sign
                            })
                            .collect()
                    } else {
                        unpack_grid(codes, D, spec.bits)
                            .into_iter()
                            .map(|code| scale * grid.points[code as usize])
                            .collect()
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
                    let expected_gamma = f32_to_f16(raw_gamma.clamp(1.0, 4.0) as f32);
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

    #[test]
    fn prefix_gamma_matches_rotated_odd_dimension_encoding() {
        let d = 100;
        let specs = [
            LayerSpec {
                bits: 1,
                seed: 0x4741_4d4d_4101,
                rotate: true,
            },
            LayerSpec {
                bits: 4,
                seed: 0x4741_4d4d_4102,
                rotate: true,
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
            let expected_stride = if spec.bits == 1 {
                d.div_ceil(64) * std::mem::size_of::<u64>()
            } else {
                grid_plane::packed_len(d, spec.bits)
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
        let query: Vec<f32> = (0..d).map(|i| ((i as f32 + 0.5) * 0.037).cos()).collect();
        let errors = audit_split_query_layer_error_squared(&query, &specs, &grids, 4);
        assert_eq!(errors.len(), 2);
        assert!(errors[0] > 0.0);
        assert_eq!(errors[1], 0.0);

        let sign_specs = [
            specs[0],
            LayerSpec {
                bits: 1,
                seed: specs[1].seed,
                rotate: true,
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
    fn cumulative_gamma_rotation_count_is_cluster_scoped_and_explicit() {
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
        fht::debug_reset_apply_count();
        let context = prepare_centroid(&centroid, &specs);
        encode_batch_in_place(&mut vectors, rows, &context, &specs, &grids);
        let count = fht::debug_apply_count();
        // Two centroid rotations per cluster plus, for each row and layer,
        // one residual rotation and one original-direction rotation used by
        // cumulative gamma. This is the complete production
        // `2 * rows + 2` bound without hiding the added exact metadata work.
        assert_eq!(count, 2 * 2 * rows + 2, "Rotation::apply count={count}");
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
