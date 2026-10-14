//! Field-keyed vector quantization settings, row-layout constants, and the arithmetic of quantized
//! estimates.

use std::collections::BTreeSet;

use quant_model::build_grid;
use serde::{Deserialize, Serialize};

use crate::schema::{FieldType, Metric, Schema, VectorDType, VectorOptions};
use crate::TantivyError;

// --- Validation ---

/// Validates model values shared by settings and stored segment metadata.
pub(crate) fn validate_quantization_values<'a>(
    dim: usize,
    models: impl IntoIterator<Item = (f64, Option<&'a [f32]>)>,
) -> Result<(), String> {
    if dim < 64 {
        return Err(
            "quantization requires dimension ≥ 64; the quantization error model is not validated \
             below this"
                .into(),
        );
    }
    for (rho, points) in models {
        if !rho.is_finite() || rho < 0.0 {
            return Err("rho_model must be finite and non-negative".into());
        }
        if points.is_some_and(|points| {
            points.iter().any(|p| !p.is_finite()) || points.windows(2).any(|p| p[0] >= p[1])
        }) {
            return Err("grid points must be finite and strictly increasing".into());
        }
    }
    Ok(())
}

// --- Constants ---

/// Settings identifier for the vector quantization format.
pub const VECTOR_QUANTIZATION_FORMAT_VERSION: u32 = 3;
/// Maximum number of residual layers stored by the vector format.
pub const MAX_QUANTIZATION_LAYERS: usize = 3;
/// Code sections begin at a 64-byte-aligned file offset.
pub const QUANTIZED_CODE_ALIGNMENT: usize = 64;
/// One little-endian binary32 scale per vector row and layer.
pub const QUANTIZED_SCALE_STRIDE: usize = 4;
/// One binary16 cumulative-prefix gamma per vector row and layer.
pub const QUANTIZED_GAMMA_STRIDE: usize = 2;
/// One binary16 corrected-error ratio per vector row and layer.
pub const QUANTIZED_ERROR_RATIO_STRIDE: usize = 2;
/// Combined per-layer sidecar bytes per vector row.
pub const QUANTIZED_SIDECAR_STRIDE: usize =
    QUANTIZED_SCALE_STRIDE + QUANTIZED_GAMMA_STRIDE + QUANTIZED_ERROR_RATIO_STRIDE;
/// One binary32 split-form constant per vector row and layer.
pub const QUANTIZED_CONSTANT_STRIDE: usize = 4;
/// One binary32 residual squared norm per vector row.
pub const QUANTIZED_RESIDUAL_NORM_STRIDE: usize = 4;
/// Closed-form uncertainty safety multiplier declared by the scoring policy.
pub(crate) const GAMMA_ANALYTICAL_SAFETY: f32 = 1.15;
/// Uniform confidence width at every quantized layer boundary.
pub(crate) const QUANTIZED_BOUNDARY_KAPPA: f32 = 2.5;
/// Query-side bits used by every sign-layer estimator.
pub(crate) const SIGN_QUERY_BITS: u8 = 4;

// --- Configuration ---

/// Normalization applied to fp32 rows before residual encoding and exact scoring.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VectorNormPolicy {
    /// Preserve input values.
    None,
    /// Normalize vectors to unit L2 norm.
    UnitL2,
}

impl VectorNormPolicy {
    /// Resolves the normalization policy for vector options.
    pub fn for_options(options: &VectorOptions) -> Self {
        if options.needs_normalization() {
            Self::UnitL2
        } else {
            Self::None
        }
    }
}

/// Format-stable validity tuple for one residual layer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct VectorQuantizationLayer {
    /// Code width.
    pub bits: u8,
    /// Rotation seed.
    pub seed: u64,
}

/// Exact-density grid persisted as part of the index configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorQuantizationGrid {
    /// Code width.
    pub bits: u8,
    /// Reconstruction points.
    pub points: Vec<f32>,
    /// Exact-density normalized RMSE resolved when the grid is materialized.
    pub rho_model: f64,
}

impl PartialEq for VectorQuantizationGrid {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
            && self.rho_model.to_bits() == other.rho_model.to_bits()
            && self.points.len() == other.points.len()
            && self
                .points
                .iter()
                .zip(&other.points)
                .all(|(left, right)| left.to_bits() == right.to_bits())
    }
}

impl Eq for VectorQuantizationGrid {}

/// One vector field's quantization configuration, persisted per index.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct VectorQuantizationConfig {
    /// Schema field name.
    pub field: String,
    /// Quantization representation identifier.
    pub format_version: u32,
    /// Vector dimension.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Vector normalization policy.
    pub norm_policy: VectorNormPolicy,
    /// Residual layers in scoring order.
    pub layers: Vec<VectorQuantizationLayer>,
    /// Persisted grids by code width.
    pub grids: Vec<VectorQuantizationGrid>,
}

impl VectorQuantizationConfig {
    /// Materializes one field's quantization configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the dimension, schedule, or generated grids are invalid.
    pub fn materialize(
        field: String,
        options: &VectorOptions,
        layers: Vec<VectorQuantizationLayer>,
    ) -> crate::Result<Self> {
        if options.dim() < 64 {
            return Err(TantivyError::InvalidArgument(
                "quantization requires dimension ≥ 64; the quantization error model is not \
                 validated below this"
                    .to_string(),
            ));
        }
        if !(1..=MAX_QUANTIZATION_LAYERS).contains(&layers.len()) {
            return Err(TantivyError::InvalidArgument(format!(
                "quantization layer count {} must be in 1..={MAX_QUANTIZATION_LAYERS}",
                layers.len()
            )));
        }
        let mut grid_widths = BTreeSet::new();
        for (layer, spec) in layers.iter().enumerate() {
            if !(1..=4).contains(&spec.bits) {
                return Err(TantivyError::InvalidArgument(format!(
                    "quantization layer {layer} has {} bits; supported range is 1..=4",
                    spec.bits
                )));
            }
            grid_widths.insert(spec.bits);
        }

        let grids = grid_widths
            .into_iter()
            .map(|bits| {
                let grid = build_grid(options.dim(), bits);
                VectorQuantizationGrid {
                    bits,
                    points: grid.points,
                    rho_model: grid.rho_model,
                }
            })
            .collect();
        let config = Self {
            field,
            format_version: VECTOR_QUANTIZATION_FORMAT_VERSION,
            dim: options.dim(),
            metric: options.metric(),
            norm_policy: VectorNormPolicy::for_options(options),
            layers,
            grids,
        };
        config.validate(options)?;
        Ok(config)
    }

    /// Materializes one field's configuration with one layer per entry of `bits`, in scoring
    /// order, each with a fresh random rotation seed.
    ///
    /// # Errors
    ///
    /// Returns an error when the dimension, schedule, or generated grids are invalid.
    pub fn with_random_seeds(
        field: String,
        options: &VectorOptions,
        bits: &[u8],
    ) -> crate::Result<Self> {
        let layers = bits
            .iter()
            .map(|&bits| VectorQuantizationLayer {
                bits,
                seed: fastrand::u64(..),
            })
            .collect();
        Self::materialize(field, options, layers)
    }

    /// Validate the persisted configuration against its schema field.
    ///
    /// # Errors
    ///
    /// Returns an error when persisted settings do not match the schema or format contract.
    pub fn validate(&self, options: &VectorOptions) -> crate::Result<()> {
        let invalid = |message: String| {
            TantivyError::InvalidArgument(format!(
                "invalid vector quantization configuration for field {:?}: {message}",
                self.field
            ))
        };

        if self.format_version != VECTOR_QUANTIZATION_FORMAT_VERSION {
            return Err(invalid(format!(
                "format version {} is unsupported; rebuild required",
                self.format_version
            )));
        }
        validate_quantization_values(self.dim, std::iter::empty()).map_err(invalid)?;
        if self.dim != options.dim() {
            return Err(invalid(format!(
                "dimension {} does not match schema dimension {}",
                self.dim,
                options.dim()
            )));
        }
        if options.dtype() != VectorDType::F32 {
            return Err(invalid(format!(
                "dtype {:?} is unsupported; expected f32",
                options.dtype()
            )));
        }
        if self.metric != options.metric() {
            return Err(invalid(format!(
                "metric {:?} does not match schema metric {:?}",
                self.metric,
                options.metric()
            )));
        }
        let expected_norm = VectorNormPolicy::for_options(options);
        if self.norm_policy != expected_norm {
            return Err(invalid(format!(
                "norm policy {:?} does not match schema policy {:?}",
                self.norm_policy, expected_norm
            )));
        }
        if !(1..=MAX_QUANTIZATION_LAYERS).contains(&self.layers.len()) {
            return Err(invalid(format!(
                "layer count {} must be in 1..={MAX_QUANTIZATION_LAYERS}",
                self.layers.len()
            )));
        }
        let mut model_widths = BTreeSet::new();
        let mut required_point_grids = BTreeSet::new();
        for (layer, spec) in self.layers.iter().enumerate() {
            if !(1..=4).contains(&spec.bits) {
                return Err(invalid(format!(
                    "layer {layer} has {} bits; supported range is 1..=4",
                    spec.bits
                )));
            }
            if spec.bits > 1 {
                required_point_grids.insert(spec.bits);
            }
            model_widths.insert(spec.bits);
        }

        let mut present_grids = BTreeSet::new();
        for grid in &self.grids {
            if !present_grids.insert(grid.bits) {
                return Err(invalid(format!(
                    "grid width {} is present more than once",
                    grid.bits
                )));
            }
            if !(1..=4).contains(&grid.bits) {
                return Err(invalid(format!(
                    "grid width {} is outside the supported range 1..=4",
                    grid.bits
                )));
            }
            let expected_points = 1usize << grid.bits;
            if grid.points.len() != expected_points {
                return Err(invalid(format!(
                    "grid width {} has {} points; expected {expected_points}",
                    grid.bits,
                    grid.points.len()
                )));
            }
            validate_quantization_values(
                self.dim,
                std::iter::once((grid.rho_model, Some(grid.points.as_slice()))),
            )
            .map_err(invalid)?;
        }
        if present_grids != model_widths || !required_point_grids.is_subset(&present_grids) {
            return Err(invalid(format!(
                "grid widths {present_grids:?} must exactly equal model widths {model_widths:?} \
                 and contain TurboQuant widths {required_point_grids:?}"
            )));
        }
        Ok(())
    }

    /// Logical quantized bytes stored for one vector row.
    pub fn bytes_per_row(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(|layer| {
                quantized_code_stride(self.dim, layer.bits)
                    + QUANTIZED_SIDECAR_STRIDE
                    + usize::from(self.needs_constants()) * QUANTIZED_CONSTANT_STRIDE
            })
            .sum();
        layer_bytes + QUANTIZED_RESIDUAL_NORM_STRIDE
    }

    /// Whether split-form L2 assembly needs per-layer constants.
    pub fn needs_constants(&self) -> bool {
        self.metric == Metric::L2
    }
}

/// Validate all field-keyed configurations against an index schema.
pub(crate) fn validate_quantization_configs(
    configs: &[VectorQuantizationConfig],
    schema: &Schema,
) -> crate::Result<()> {
    let mut fields = BTreeSet::new();
    for config in configs {
        if !fields.insert(config.field.as_str()) {
            return Err(TantivyError::InvalidArgument(format!(
                "vector quantization configuration for field {:?} is duplicated",
                config.field
            )));
        }
        let field = schema.get_field(&config.field)?;
        let options = match schema.get_field_entry(field).field_type() {
            FieldType::Vector(options) => options,
            _ => {
                return Err(TantivyError::InvalidArgument(format!(
                    "vector quantization configuration targets non-vector field {:?}",
                    config.field
                )));
            }
        };
        config.validate(options)?;
    }
    Ok(())
}

/// Packed-code stride for one vector row.
///
/// # Panics
///
/// Panics when `dim` is zero, `bits` is outside `1..=4`, or multiplication overflows.
pub fn quantized_code_stride(dim: usize, bits: u8) -> usize {
    assert!(dim > 0);
    assert!((1..=4).contains(&bits));
    dim.checked_mul(usize::from(bits))
        .expect("quantized code stride overflow")
        .div_ceil(64)
        * 8
}

/// Whether all padding bits after the row's `dim * bits` payload are zero.
pub(crate) fn quantized_code_tail_is_zero(codes: &[u8], dim: usize, bits: u8) -> bool {
    if dim == 0 || !(1..=4).contains(&bits) || codes.len() != quantized_code_stride(dim, bits) {
        return false;
    }
    let used_bits = dim * usize::from(bits);
    let full_bytes = used_bits / 8;
    let tail_bits = used_bits % 8;
    if tail_bits == 0 {
        codes[full_bytes..].iter().all(|&byte| byte == 0)
    } else {
        let used_mask = (1_u8 << tail_bits) - 1;
        codes[full_bytes] & !used_mask == 0 && codes[full_bytes + 1..].iter().all(|&byte| byte == 0)
    }
}

// --- Estimates ---

/// Applies the metric-specific correction to a cumulative quantized estimate.
#[inline(always)]
pub(crate) fn corrected_quantized_estimate(
    metric: Metric,
    gamma: f32,
    raw_prefix: f32,
    base: f32,
) -> f32 {
    let metric_factor = if metric == Metric::L2 { 2.0 } else { 1.0 };
    (metric_factor * gamma).mul_add(raw_prefix, base)
}

/// Combines the first L2 layer's unscaled kernel output.
#[inline(always)]
pub(crate) fn initial_l2_raw_prefix(kernel_score: f32, scale: f32, constant: f32) -> f32 {
    scale.mul_add(kernel_score, -constant)
}

/// Combines the first dot-like layer's unscaled kernel output.
#[inline(always)]
pub(crate) fn initial_dot_raw_prefix(kernel_score: f32, scale: f32) -> f32 {
    kernel_score * scale
}

/// Adds an L2 refinement to the cumulative raw prefix.
#[inline(always)]
pub(crate) fn refine_l2_raw_prefix(
    raw_prefix: f32,
    kernel_score: f32,
    scale: f32,
    constant: f32,
) -> f32 {
    scale.mul_add(kernel_score, raw_prefix - constant)
}

/// Squared rounding error of an L2 raw prefix, in raw-prefix units.
///
/// The split operands were rounded independently, so cancellation does not remove their error; a
/// refinement adds the `previous_prefix - constant` subtraction.
#[inline(always)]
pub(crate) fn l2_arithmetic_variance(
    kernel_score: f32,
    scale: f32,
    constant: f32,
    previous_prefix: Option<f32>,
) -> f32 {
    // c = 2: independently rounded split operands, then the fused subtraction.
    let subtraction = rounding_error(scale * kernel_score, constant, 2.0);
    // c = 1: the separate `previous_prefix - constant` subtraction.
    let refinement = previous_prefix.map_or(0.0, |prefix| rounding_error(prefix, constant, 1.0));
    subtraction * subtraction + refinement * refinement
}

/// The f32 error of `rounding_count` roundings over operands `a` and `b`: machine epsilon times
/// the sum of their magnitudes.
#[inline(always)]
pub(crate) fn rounding_error(a: f32, b: f32, rounding_count: f32) -> f32 {
    rounding_count * f32::EPSILON * (a.abs() + b.abs())
}

/// An estimate's arithmetic rounding error, kept apart from gamma so a refinement can change gamma
/// without losing or double-scaling the error already in the raw prefix.
#[derive(Clone, Copy, Default)]
pub(crate) struct ArithmeticError {
    pub(crate) raw_variance: f32,
    pub(crate) base_variance: f32,
}

impl ArithmeticError {
    /// The error of a cluster's first layer.
    pub(crate) fn initial(
        metric: Metric,
        kernel: f32,
        scale: f32,
        constant: f32,
        cluster_score: f32,
        residual_norm_sq: f32,
    ) -> Self {
        let raw_variance = if metric == Metric::L2 {
            l2_arithmetic_variance(kernel, scale, constant, None)
        } else {
            // c = 1: the initial scale * kernel multiplication.
            rounding_error(scale * kernel, 0.0, 1.0).powi(2)
        };
        let base_variance = if metric == Metric::L2 {
            // c = 1: cluster_score - residual_norm_sq.
            rounding_error(cluster_score, residual_norm_sq, 1.0).powi(2)
        } else {
            0.0
        };
        Self {
            raw_variance,
            base_variance,
        }
    }

    /// Adds one refining layer's error.
    pub(crate) fn refine(
        &mut self,
        metric: Metric,
        prefix: f32,
        kernel: f32,
        scale: f32,
        constant: f32,
    ) {
        self.raw_variance += if metric == Metric::L2 {
            l2_arithmetic_variance(kernel, scale, constant, Some(prefix))
        } else {
            // c = 2: rounded layer contribution plus the refinement addition.
            rounding_error(scale * kernel, prefix, 2.0).powi(2)
        };
    }

    /// The estimate's width: `model_sigma` combined in quadrature with the arithmetic error under
    /// the metric's correction.
    pub(crate) fn sigma(
        self,
        metric: Metric,
        model_sigma: f32,
        gamma: f32,
        prefix: f32,
        base: f32,
    ) -> f32 {
        let factor = if metric == Metric::L2 {
            2.0 * gamma
        } else {
            gamma
        };
        // c = 1: the final fused base-plus-corrected-residual expression. This
        // includes cosine's base-plus-residual sum even if its model width is zero.
        let final_error = rounding_error(factor * prefix, base, 1.0);
        model_sigma.hypot(
            (factor * factor * self.raw_variance + self.base_variance + final_error * final_error)
                .sqrt(),
        )
    }
}

/// Adds a dot-like refinement to the cumulative raw prefix.
#[inline(always)]
pub(crate) fn refine_dot_raw_prefix(raw_prefix: f32, kernel_score: f32, scale: f32) -> f32 {
    scale.mul_add(kernel_score, raw_prefix)
}

/// The statistical width of a corrected estimate: its data and query quantization variance,
/// scaled by the policy's safety factor.
#[inline(always)]
pub(crate) fn quantized_model_sigma(
    metric: Metric,
    dimension: usize,
    residual_norm_squared: f32,
    corrected_error_ratio: f32,
    gamma: f32,
    score_query_norm_squared: f32,
    sign_query_error_term: f32,
) -> f32 {
    debug_assert_ne!(dimension, 0);
    let data_variance = residual_norm_squared
        * (1.0 / dimension as f32)
        * corrected_error_ratio
        * score_query_norm_squared;
    let query_variance = gamma * gamma * sign_query_error_term;
    let metric_factor = if metric == Metric::L2 { 2.0 } else { 1.0 };
    metric_factor * GAMMA_ANALYTICAL_SAFETY * (data_variance + query_variance).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(bits: u8) -> VectorQuantizationGrid {
        let count = 1usize << bits;
        VectorQuantizationGrid {
            bits,
            points: (0..count).map(|point| point as f32).collect(),
            rho_model: 0.25,
        }
    }

    fn config(bits: &[u8]) -> VectorQuantizationConfig {
        let grid_bits: BTreeSet<u8> = bits.iter().copied().collect();
        VectorQuantizationConfig {
            field: "embedding".to_string(),
            format_version: VECTOR_QUANTIZATION_FORMAT_VERSION,
            dim: 768,
            metric: Metric::Dot,
            norm_policy: VectorNormPolicy::None,
            layers: bits
                .iter()
                .enumerate()
                .map(|(layer, &bits)| VectorQuantizationLayer {
                    bits,
                    seed: 11 * (layer as u64 + 1),
                })
                .collect(),
            grids: grid_bits.into_iter().map(grid).collect(),
        }
    }

    #[test]
    fn settings_ignore_unknown_grid_fields() {
        let settings = crate::index::IndexSettings {
            vector_quantization: vec![config(&[1, 4])],
            ..Default::default()
        };
        let mut json = serde_json::to_value(&settings).unwrap();
        for grid in json["vector_quantization"][0]["grids"]
            .as_array_mut()
            .unwrap()
        {
            grid["version"] = serde_json::json!(1);
        }
        let decoded: crate::index::IndexSettings = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.vector_quantization, settings.vector_quantization);
        decoded.vector_quantization[0]
            .validate(&VectorOptions::new(768, Metric::Dot))
            .unwrap();
        let encoded = serde_json::to_value(decoded).unwrap();
        for grid in encoded["vector_quantization"][0]["grids"]
            .as_array()
            .unwrap()
        {
            assert!(grid.get("version").is_none());
        }
    }

    #[test]
    fn one_plus_four_is_500_bytes_per_dot_row() {
        assert_eq!(config(&[1, 4]).bytes_per_row(), 500);
    }

    #[test]
    fn one_plus_four_is_508_bytes_per_l2_row() {
        let mut config = config(&[1, 4]);
        config.metric = Metric::L2;
        assert_eq!(config.bytes_per_row(), 508);
    }

    #[test]
    fn validates_against_the_field_schema() {
        let mut builder = Schema::builder();
        builder.add_vector_field("embedding", VectorOptions::new(768, Metric::Dot));
        let schema = builder.build();
        validate_quantization_configs(&[config(&[1, 4])], &schema).unwrap();
    }

    #[test]
    fn rejects_width_above_four() {
        let options = VectorOptions::new(768, Metric::Dot);
        let mut invalid = config(&[1, 4]);
        invalid.layers[1].bits = 5;
        let err = invalid.validate(&options).unwrap_err();
        assert!(err.to_string().contains("supported range is 1..=4"));
    }

    #[test]
    fn accepts_grid_first_schedules() {
        let options = VectorOptions::new(768, Metric::Dot);
        config(&[4]).validate(&options).unwrap();
        config(&[2, 4]).validate(&options).unwrap();
    }

    #[test]
    fn rejects_quantized_dimensions_below_model_floor() {
        let options = VectorOptions::new(63, Metric::Dot);
        let mut invalid = config(&[1, 4]);
        invalid.dim = 63;
        let err = invalid.validate(&options).unwrap_err();
        assert!(err.to_string().contains(
            "quantization requires dimension ≥ 64; the quantization error model is not validated \
             below this"
        ));
    }

    #[test]
    fn materializes_exact_density_grids_once_per_width() {
        let options = VectorOptions::new(64, Metric::Dot);
        let layers = [1, 4, 4]
            .into_iter()
            .enumerate()
            .map(|(layer, bits)| VectorQuantizationLayer {
                bits,
                seed: layer as u64 + 1,
            })
            .collect();
        let materialized =
            VectorQuantizationConfig::materialize("embedding".to_string(), &options, layers)
                .unwrap();
        assert_eq!(materialized.dim, 64);
        assert_eq!(materialized.grids.len(), 2);
        for persisted in &materialized.grids {
            let recomputed = build_grid(materialized.dim, persisted.bits);
            assert_eq!(persisted.points.len(), 1usize << persisted.bits);
            assert_eq!(persisted.points, recomputed.points);
            assert_eq!(persisted.rho_model, recomputed.rho_model);
        }
    }

    #[test]
    fn accepts_general_dimension_and_uses_word_rounded_strides() {
        let options = VectorOptions::new(769, Metric::Dot);
        let mut general = config(&[1, 4]);
        general.dim = 769;
        general.validate(&options).unwrap();
        assert_eq!(quantized_code_stride(65, 1), 16);
        assert_eq!(quantized_code_stride(100, 4), 56);
        assert_eq!(quantized_code_stride(769, 4), 392);
    }

    #[test]
    fn divisible_dimensions_use_exact_byte_strides() {
        for dim in [64, 128, 768, 1536] {
            for bits in 1..=4 {
                assert_eq!(quantized_code_stride(dim, bits), dim * bits as usize / 8);
            }
        }
    }

    #[test]
    fn validates_zero_tail_bits() {
        let mut codes = vec![0_u8; quantized_code_stride(65, 4)];
        codes[32] = 0x0f;
        assert!(quantized_code_tail_is_zero(&codes, 65, 4));
        codes[32] |= 0x10;
        assert!(!quantized_code_tail_is_zero(&codes, 65, 4));
    }

    #[test]
    fn grids_are_field_scoped_for_mixed_dimensions() {
        let mut builder = Schema::builder();
        builder.add_vector_field("embedding_768", VectorOptions::new(768, Metric::Dot));
        builder.add_vector_field("embedding_1536", VectorOptions::new(1536, Metric::Cosine));
        let schema = builder.build();

        let mut first = config(&[1, 4]);
        first.field = "embedding_768".to_string();
        let mut second = config(&[1, 4]);
        second.field = "embedding_1536".to_string();
        second.dim = 1536;
        second.metric = Metric::Cosine;
        second.norm_policy = VectorNormPolicy::UnitL2;
        validate_quantization_configs(&[first, second], &schema).unwrap();
    }

    #[test]
    fn index_settings_round_trip_field_scoped_metadata() {
        let settings = crate::IndexSettings {
            vector_quantization: vec![config(&[1, 4])],
            ..crate::IndexSettings::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        let decoded: crate::IndexSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, settings);
    }

    #[test]
    fn index_builder_accepts_general_dimension_quantization() {
        let mut builder = Schema::builder();
        builder.add_vector_field("embedding", VectorOptions::new(769, Metric::Dot));
        let schema = builder.build();
        let mut general = config(&[1, 4]);
        general.dim = 769;
        let settings = crate::IndexSettings {
            vector_quantization: vec![general],
            ..crate::IndexSettings::default()
        };

        crate::Index::builder()
            .schema(schema)
            .settings(settings)
            .create_in_ram()
            .unwrap();
    }
}
