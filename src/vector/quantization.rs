//! Index-level configuration and fixed-stride layout for vector quantization.
//!
//! Configuration is persisted once in [`IndexSettings`](crate::IndexSettings),
//! keyed by vector field name, and reused by every segment. Field scoping is
//! necessary because an index may contain vector fields with different
//! dimensions and metrics; exact-density grids depend on the dimension.

use std::collections::BTreeSet;

use quant_model::build_grid;
use serde::{Deserialize, Serialize};

use crate::schema::{FieldType, Metric, Schema, VectorDType, VectorOptions};
use crate::TantivyError;

/// Settings schema for the vector quantization format.
pub const VECTOR_QUANTIZATION_FORMAT_VERSION: u32 = 4;
/// Version of the persisted exact-density Lloyd-Max grid representation.
pub const GRID_FORMAT_VERSION: u32 = 1;
/// Maximum number of residual layers stored by the vector format.
pub const MAX_QUANTIZATION_LAYERS: usize = 3;
/// Code sections begin at a 64-byte-aligned file offset.
pub const QUANTIZED_CODE_ALIGNMENT: usize = 64;
/// One little-endian binary32 scale per posting-membership row and layer.
pub const QUANTIZED_SCALE_STRIDE: usize = 4;
/// One binary16 cumulative-prefix gamma per posting-membership row and layer.
pub const QUANTIZED_GAMMA_STRIDE: usize = 2;
/// One binary16 corrected-error ratio per posting-membership row and layer.
pub const QUANTIZED_ERROR_RATIO_STRIDE: usize = 2;
/// Combined per-layer sidecar bytes per posting-membership row.
pub const QUANTIZED_SIDECAR_STRIDE: usize =
    QUANTIZED_SCALE_STRIDE + QUANTIZED_GAMMA_STRIDE + QUANTIZED_ERROR_RATIO_STRIDE;
/// One binary32 split-form constant per posting-membership row and layer.
pub const QUANTIZED_CONSTANT_STRIDE: usize = 4;
/// One binary32 residual squared norm per posting-membership row.
pub const QUANTIZED_RESIDUAL_NORM_STRIDE: usize = 4;
/// Closed-form uncertainty safety multiplier declared by the scoring policy.
pub(crate) const GAMMA_ANALYTICAL_SAFETY: f32 = 1.15;
/// Uniform confidence width at every quantized layer boundary.
pub(crate) const QUANTIZED_BOUNDARY_KAPPA: f32 = 2.0;
/// Query-side bits used by every sign-layer estimator.
pub(crate) const SIGN_QUERY_BITS: u8 = 4;
/// Origin of one settings-backed calibration measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum VectorQuantizationCalibrationSource {
    HeldOut = 0,
    RealQuery = 1,
}

impl Serialize for VectorQuantizationCalibrationSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        serializer.serialize_u8(*self as u8)
    }
}

impl<'de> Deserialize<'de> for VectorQuantizationCalibrationSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        match u8::deserialize(deserializer)? {
            0 => Ok(Self::HeldOut),
            1 => Ok(Self::RealQuery),
            value => Err(serde::de::Error::custom(format!(
                "quantization calibration source {value} is unsupported; expected 0 or 1"
            ))),
        }
    }
}

/// Diagnostic bias and spread for one active scorer prefix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorQuantizationDepthCalibration {
    pub bias: f32,
    pub spread: f32,
    pub sample_count: u32,
    pub source: VectorQuantizationCalibrationSource,
    pub protocol: String,
}

impl PartialEq for VectorQuantizationDepthCalibration {
    fn eq(&self, other: &Self) -> bool {
        self.bias.to_bits() == other.bias.to_bits()
            && self.spread.to_bits() == other.spread.to_bits()
            && self.sample_count == other.sample_count
            && self.source == other.source
            && self.protocol == other.protocol
    }
}

impl Eq for VectorQuantizationDepthCalibration {}

/// Normalization applied to fp32 rows before residual encoding and rerank.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VectorNormPolicy {
    None,
    UnitL2,
}

impl VectorNormPolicy {
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
    pub bits: u8,
    pub seed: u64,
}

/// Exact-density grid persisted as part of the index configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorQuantizationGrid {
    pub bits: u8,
    pub version: u32,
    pub points: Vec<f32>,
    /// Exact-density normalized RMSE resolved when the grid is materialized.
    pub rho_model: f64,
}

// Grid points are finite after validation. Bit equality also distinguishes
// -0.0 and makes Eq sound for IndexSettings without weakening its contract.
impl PartialEq for VectorQuantizationGrid {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
            && self.version == other.version
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
    pub field: String,
    pub format_version: u32,
    pub dim: usize,
    pub metric: Metric,
    pub norm_policy: VectorNormPolicy,
    pub layers: Vec<VectorQuantizationLayer>,
    pub grids: Vec<VectorQuantizationGrid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    calibration: Option<Vec<VectorQuantizationDepthCalibration>>,
}

impl VectorQuantizationConfig {
    /// Materialize one field's format-stable configuration, including every
    /// exact-density TurboQuant grid required by `layers`.
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
            // Persist one complete model entry for every width, including the
            // sign quantizer. Construction resolves these entries instead of
            // rerunning the solver; scoring uncertainty comes from the stored
            // exact-error ratio rather than a model rho.
            grid_widths.insert(spec.bits);
        }

        let grids = grid_widths
            .into_iter()
            .map(|bits| {
                let grid = build_grid(options.dim(), bits);
                VectorQuantizationGrid {
                    bits,
                    version: GRID_FORMAT_VERSION,
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
            calibration: None,
        };
        config.validate(options)?;
        Ok(config)
    }

    /// Validate the persisted configuration against its schema field.
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
        if self.dim < 64 {
            return Err(invalid(
                "quantization requires dimension ≥ 64; the quantization error model is not \
                 validated below this"
                    .to_string(),
            ));
        }
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
            if grid.version != GRID_FORMAT_VERSION {
                return Err(invalid(format!(
                    "grid width {} has version {}; expected {GRID_FORMAT_VERSION}",
                    grid.bits, grid.version
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
            if grid.points.iter().any(|point| !point.is_finite())
                || grid.points.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(invalid(format!(
                    "grid width {} points must be finite and strictly increasing",
                    grid.bits
                )));
            }
            if !grid.rho_model.is_finite() || grid.rho_model < 0.0 {
                return Err(invalid(format!(
                    "grid width {} rho_model must be finite and non-negative",
                    grid.bits
                )));
            }
        }
        if present_grids != model_widths || !required_point_grids.is_subset(&present_grids) {
            return Err(invalid(format!(
                "grid widths {present_grids:?} must exactly equal model widths {model_widths:?} \
                 and contain TurboQuant widths {required_point_grids:?}"
            )));
        }
        if let Some(calibration) = &self.calibration {
            validate_calibration(calibration, self.layers.len()).map_err(invalid)?;
        }
        Ok(())
    }

    /// Returns settings-backed diagnostic calibration when one was recorded.
    /// Production scoring never consumes this metadata.
    pub fn calibration(&self) -> Option<&[VectorQuantizationDepthCalibration]> {
        self.calibration.as_deref()
    }

    /// Install caller-query diagnostic measurements.
    pub fn install_real_query_calibration(
        &mut self,
        calibration: Vec<VectorQuantizationDepthCalibration>,
    ) -> crate::Result<()> {
        validate_calibration(&calibration, self.layers.len())
            .map_err(TantivyError::InvalidArgument)?;
        if calibration
            .iter()
            .any(|depth| depth.source != VectorQuantizationCalibrationSource::RealQuery)
        {
            return Err(TantivyError::InvalidArgument(format!(
                "caller-query diagnostic entries must have source RealQuery"
            )));
        }
        self.calibration = Some(calibration);
        Ok(())
    }

    /// Logical quantized bytes stored for one posting-membership row.
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

fn validate_calibration(
    calibration: &[VectorQuantizationDepthCalibration],
    layer_count: usize,
) -> Result<(), String> {
    if calibration.len() != layer_count {
        return Err(format!(
            "calibration has {} depths; expected {layer_count}",
            calibration.len()
        ));
    }
    for (depth, value) in calibration.iter().enumerate() {
        if !value.bias.is_finite() {
            return Err(format!(
                "calibration depth {depth} bias must be finite, got {}",
                value.bias
            ));
        }
        if !value.spread.is_finite() || value.spread < 0.0 {
            return Err(format!(
                "calibration depth {depth} spread must be finite and non-negative, got {}",
                value.spread
            ));
        }
        if value.sample_count == 0 {
            return Err(format!(
                "calibration depth {depth} sample_count must be greater than zero"
            ));
        }
        if value.protocol.trim().is_empty() {
            return Err(format!(
                "calibration depth {depth} protocol must be non-empty"
            ));
        }
        if depth != 0 && value.source != calibration[0].source {
            return Err(
                "quantization calibration source must be uniform across all depths".to_string(),
            );
        }
        if depth != 0 && value.protocol != calibration[0].protocol {
            return Err(
                "quantization calibration protocol must be uniform across all depths".to_string(),
            );
        }
    }
    Ok(())
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

/// Packed-code stride for one posting-membership row.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(bits: u8) -> VectorQuantizationGrid {
        let count = 1usize << bits;
        VectorQuantizationGrid {
            bits,
            version: GRID_FORMAT_VERSION,
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
            calibration: None,
        }
    }

    #[test]
    fn one_plus_four_is_500_bytes_per_dot_row() {
        assert_eq!(config(&[1, 4]).bytes_per_row(), 500);
    }

    #[test]
    fn settings_calibration_round_trips_and_enforces_precedence() {
        let mut config = config(&[1, 4]);
        let diagnostic = vec![
            VectorQuantizationDepthCalibration {
                bias: -0.5,
                spread: 2.25,
                sample_count: 1_024,
                source: VectorQuantizationCalibrationSource::RealQuery,
                protocol: "REAL_QUERY_EXACT_E_BQ4".to_string(),
            };
            2
        ];
        config
            .install_real_query_calibration(diagnostic.clone())
            .unwrap();
        let replacement = diagnostic
            .iter()
            .map(|depth| VectorQuantizationDepthCalibration {
                bias: depth.bias + 0.25,
                ..depth.clone()
            })
            .collect::<Vec<_>>();
        config
            .install_real_query_calibration(replacement.clone())
            .unwrap();
        let json = serde_json::to_string(&config).unwrap();
        let decoded: VectorQuantizationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.calibration(), Some(replacement.as_slice()));

        let mut zero_samples = replacement.clone();
        zero_samples[0].sample_count = 0;
        assert!(config.install_real_query_calibration(zero_samples).is_err());

        let mut empty_protocol = replacement.clone();
        empty_protocol[0].protocol.clear();
        assert!(config
            .install_real_query_calibration(empty_protocol)
            .is_err());

        let mut mixed_protocol = replacement.clone();
        mixed_protocol[1].protocol = "HELD_OUT_EXACT_E_BQ4".to_string();
        assert!(config
            .install_real_query_calibration(mixed_protocol)
            .is_err());

        let mut mixed = config.clone();
        let mut mixed_depths = replacement;
        mixed_depths[0].source = VectorQuantizationCalibrationSource::HeldOut;
        mixed.calibration = Some(mixed_depths);
        assert!(mixed
            .validate(&VectorOptions::new(768, Metric::Dot))
            .is_err());
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
            assert_eq!(persisted.version, GRID_FORMAT_VERSION);
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
