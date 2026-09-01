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

/// Quantized payloads first appear in V3 vector files.
pub const VECTOR_QUANTIZATION_FORMAT_VERSION: u32 = 3;
/// Version of the persisted exact-density Lloyd-Max grid representation.
pub const GRID_FORMAT_VERSION: u32 = 1;
/// V1 supports at most four residual layers.
pub const MAX_QUANTIZATION_LAYERS: usize = 4;
/// V1 code sections begin at a 64-byte-aligned file offset.
pub const QUANTIZED_CODE_ALIGNMENT: usize = 64;
/// One binary16 scale per posting-membership row and layer.
pub const QUANTIZED_SCALE_STRIDE: usize = 2;
/// One binary32 split-form constant per posting-membership row and layer.
pub const QUANTIZED_CONSTANT_STRIDE: usize = 4;
/// One binary32 residual squared norm per posting-membership row when needed.
pub const QUANTIZED_RESIDUAL_NORM_STRIDE: usize = 4;
/// Version of the per-segment, per-field measured calibration payload.
pub const QUANTIZED_CALIBRATION_VERSION: u32 = 3;
const LEGACY_QUANTIZED_CALIBRATION_VERSION: u32 = 1;
const LEGACY_PER_DEPTH_CALIBRATION_VERSION: u32 = 2;
const LEGACY_QUANTIZED_CALIBRATION_METADATA_LEN: usize = 12;
const QUANTIZED_CALIBRATION_HEADER_LEN: usize = 8;
const LEGACY_QUANTIZED_CALIBRATION_DEPTH_LEN: usize = 8;
const QUANTIZED_CALIBRATION_DEPTH_LEN: usize = 12;

pub(crate) fn quantized_calibration_metadata_len(layer_count: usize) -> usize {
    QUANTIZED_CALIBRATION_HEADER_LEN + layer_count * QUANTIZED_CALIBRATION_DEPTH_LEN
}

/// Build-measured uncertainty calibration for one scorer prefix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VectorQuantizationDepthCalibration {
    /// Signed normalized error center. The reader adds
    /// `bias * scale * rho * ||q||` to the estimate.
    pub(crate) bias: f32,
    pub(crate) cal: f32,
    pub(crate) sample_count: u32,
}

/// One calibration entry for every active prefix depth in a quantized field.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VectorQuantizationCalibration {
    pub(crate) depths: Vec<VectorQuantizationDepthCalibration>,
}

impl VectorQuantizationCalibration {
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(quantized_calibration_metadata_len(self.depths.len()));
        bytes.extend_from_slice(&QUANTIZED_CALIBRATION_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.depths.len() as u32).to_le_bytes());
        for depth in &self.depths {
            bytes.extend_from_slice(&depth.bias.to_le_bytes());
            bytes.extend_from_slice(&depth.cal.to_le_bytes());
            bytes.extend_from_slice(&depth.sample_count.to_le_bytes());
        }
        bytes
    }

    pub(crate) fn decode(bytes: &[u8], layer_count: usize) -> crate::Result<Self> {
        if bytes.len() < std::mem::size_of::<u32>() {
            return Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only(format!(
                    "quantization calibration metadata has {} bytes; expected a version word",
                    bytes.len()
                )),
            ));
        }
        let version = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        if version == LEGACY_QUANTIZED_CALIBRATION_VERSION {
            if bytes.len() != LEGACY_QUANTIZED_CALIBRATION_METADATA_LEN {
                return Err(TantivyError::DataCorruption(
                    crate::error::DataCorruption::comment_only(format!(
                        "legacy quantization calibration metadata has {} bytes; expected {}",
                        bytes.len(),
                        LEGACY_QUANTIZED_CALIBRATION_METADATA_LEN
                    )),
                ));
            }
            let depth = VectorQuantizationDepthCalibration {
                bias: 0.0,
                cal: f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                sample_count: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            };
            validate_calibration_depth(depth)?;
            return Ok(Self {
                depths: vec![depth; layer_count],
            });
        }
        if version == LEGACY_PER_DEPTH_CALIBRATION_VERSION {
            let expected_len = QUANTIZED_CALIBRATION_HEADER_LEN
                + layer_count * LEGACY_QUANTIZED_CALIBRATION_DEPTH_LEN;
            if bytes.len() != expected_len {
                return Err(TantivyError::DataCorruption(
                    crate::error::DataCorruption::comment_only(format!(
                        "legacy per-depth quantization calibration metadata has {} bytes; \
                         expected {expected_len}",
                        bytes.len()
                    )),
                ));
            }
            let stored_layer_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
            if stored_layer_count != layer_count {
                return Err(TantivyError::DataCorruption(
                    crate::error::DataCorruption::comment_only(format!(
                        "quantization calibration has {stored_layer_count} depths; expected \
                         {layer_count}"
                    )),
                ));
            }
            let mut depths = Vec::with_capacity(layer_count);
            for encoded in bytes[QUANTIZED_CALIBRATION_HEADER_LEN..]
                .chunks_exact(LEGACY_QUANTIZED_CALIBRATION_DEPTH_LEN)
            {
                let depth = VectorQuantizationDepthCalibration {
                    bias: 0.0,
                    cal: f32::from_le_bytes(encoded[..4].try_into().unwrap()),
                    sample_count: u32::from_le_bytes(encoded[4..8].try_into().unwrap()),
                };
                validate_calibration_depth(depth)?;
                depths.push(depth);
            }
            return Ok(Self { depths });
        }
        if version != QUANTIZED_CALIBRATION_VERSION {
            return Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only(format!(
                    "quantization calibration metadata version {version} is unsupported; expected \
                     {QUANTIZED_CALIBRATION_VERSION}"
                )),
            ));
        }
        let expected_len = quantized_calibration_metadata_len(layer_count);
        if bytes.len() != expected_len {
            return Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only(format!(
                    "quantization calibration metadata has {} bytes; expected {expected_len}",
                    bytes.len()
                )),
            ));
        }
        let stored_layer_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if stored_layer_count != layer_count {
            return Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only(format!(
                    "quantization calibration has {stored_layer_count} depths; expected \
                     {layer_count}"
                )),
            ));
        }
        let mut depths = Vec::with_capacity(layer_count);
        for encoded in
            bytes[QUANTIZED_CALIBRATION_HEADER_LEN..].chunks_exact(QUANTIZED_CALIBRATION_DEPTH_LEN)
        {
            let depth = VectorQuantizationDepthCalibration {
                bias: f32::from_le_bytes(encoded[..4].try_into().unwrap()),
                cal: f32::from_le_bytes(encoded[4..8].try_into().unwrap()),
                sample_count: u32::from_le_bytes(encoded[8..12].try_into().unwrap()),
            };
            validate_calibration_depth(depth)?;
            depths.push(depth);
        }
        Ok(Self { depths })
    }
}

fn validate_calibration_depth(depth: VectorQuantizationDepthCalibration) -> crate::Result<()> {
    if !depth.bias.is_finite() {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only(format!(
                "quantization calibration bias must be finite, got {}",
                depth.bias
            )),
        ));
    }
    if !depth.cal.is_finite() || depth.cal < 0.0 {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only(format!(
                "quantization calibration must be finite and non-negative, got {}",
                depth.cal
            )),
        ));
    }
    Ok(())
}

/// The stored-code construction and corresponding scoring kernel.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VectorQuantizer {
    /// Single-bit RaBitQ sign snap and popcount scoring.
    RaBitQ,
    /// Exact-density TurboQuant grid and LUT scoring.
    TurboQuant,
}

impl VectorQuantizer {
    /// Default quantizer inferred from a layer's stored width.
    pub fn inferred(bits: u8) -> Self {
        if bits == 1 {
            Self::RaBitQ
        } else {
            Self::TurboQuant
        }
    }
}

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
    pub quantizer: VectorQuantizer,
    pub seed: u64,
}

/// Exact-density grid persisted as part of the index configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorQuantizationGrid {
    pub bits: u8,
    pub version: u32,
    pub points: Vec<f32>,
    /// Exact-density normalized RMSE resolved when the grid is materialized.
    /// `None` reads pre-amendment V3 metadata; the segment-open resolver has
    /// a compatibility path that derives it from the persisted points once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rho_model: Option<f64>,
}

// Grid points are finite after validation. Bit equality also distinguishes
// -0.0 and makes Eq sound for IndexSettings without weakening its contract.
impl PartialEq for VectorQuantizationGrid {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
            && self.version == other.version
            && self.rho_model.map(f64::to_bits) == other.rho_model.map(f64::to_bits)
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
            if spec.quantizer == VectorQuantizer::RaBitQ && spec.bits != 1 {
                return Err(TantivyError::InvalidArgument(format!(
                    "quantization layer {layer} selects RaBitQ with {} bits; RaBitQ requires 1 bit",
                    spec.bits
                )));
            }
            // Persist one complete model entry for every width, including
            // the sign plane. Sign scoring does not consume the points, but
            // its sigma chain consumes the persisted rho.
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
                    rho_model: Some(grid.rho_model),
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
                "format version {} is unsupported; expected {}",
                self.format_version, VECTOR_QUANTIZATION_FORMAT_VERSION
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
            match spec.quantizer {
                VectorQuantizer::RaBitQ if spec.bits != 1 => {
                    return Err(invalid(format!(
                        "layer {layer} selects RaBitQ with {} bits; RaBitQ requires 1 bit",
                        spec.bits
                    )));
                }
                VectorQuantizer::TurboQuant => {
                    required_point_grids.insert(spec.bits);
                }
                VectorQuantizer::RaBitQ => {}
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
            if grid
                .rho_model
                .is_some_and(|rho| !rho.is_finite() || rho < 0.0)
            {
                return Err(invalid(format!(
                    "grid width {} rho_model must be finite and non-negative",
                    grid.bits
                )));
            }
        }
        // Pre-amendment V3 persisted point grids only for TurboQuant widths.
        // New metadata persists every model width so sign rho never needs to
        // be resolved again, while the legacy subset remains readable until
        // its next merge/REINDEX.
        if !required_point_grids.is_subset(&present_grids)
            || !present_grids.is_subset(&model_widths)
        {
            return Err(invalid(format!(
                "grid widths {present_grids:?} must contain TurboQuant widths \
                 {required_point_grids:?} and stay within model widths {model_widths:?}"
            )));
        }
        Ok(())
    }

    /// Logical quantized bytes stored for one posting-membership row.
    pub fn bytes_per_row(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(|layer| {
                quantized_code_stride(self.dim, layer.bits)
                    + QUANTIZED_SCALE_STRIDE
                    + QUANTIZED_CONSTANT_STRIDE
            })
            .sum();
        layer_bytes + usize::from(self.needs_residual_norm()) * QUANTIZED_RESIDUAL_NORM_STRIDE
    }

    /// Whether split-form metric assembly needs the exact residual norm.
    pub fn needs_residual_norm(&self) -> bool {
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
            rho_model: Some(0.25),
        }
    }

    fn config(bits: &[u8]) -> VectorQuantizationConfig {
        let grid_bits: BTreeSet<u8> = bits.iter().copied().filter(|bits| *bits > 1).collect();
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
                    quantizer: VectorQuantizer::inferred(bits),
                    seed: 11 * (layer as u64 + 1),
                })
                .collect(),
            grids: grid_bits.into_iter().map(grid).collect(),
        }
    }

    #[test]
    fn one_plus_four_is_492_bytes_per_dot_row() {
        assert_eq!(config(&[1, 4]).bytes_per_row(), 492);
    }

    #[test]
    fn calibration_metadata_round_trips_and_rejects_invalid_values() {
        let metadata = VectorQuantizationCalibration {
            depths: vec![
                VectorQuantizationDepthCalibration {
                    bias: -0.5,
                    cal: 2.25,
                    sample_count: 1_024,
                },
                VectorQuantizationDepthCalibration {
                    bias: 0.25,
                    cal: 1.75,
                    sample_count: 1_000,
                },
            ],
        };
        assert_eq!(
            VectorQuantizationCalibration::decode(&metadata.encode(), 2).unwrap(),
            metadata
        );

        let mut invalid = metadata.encode();
        invalid[8..12].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(VectorQuantizationCalibration::decode(&invalid, 2).is_err());

        let mut legacy = [0_u8; LEGACY_QUANTIZED_CALIBRATION_METADATA_LEN];
        legacy[..4].copy_from_slice(&LEGACY_QUANTIZED_CALIBRATION_VERSION.to_le_bytes());
        legacy[4..8].copy_from_slice(&3.25_f32.to_le_bytes());
        legacy[8..12].copy_from_slice(&777_u32.to_le_bytes());
        let decoded = VectorQuantizationCalibration::decode(&legacy, 2).unwrap();
        assert_eq!(
            decoded.depths,
            vec![
                VectorQuantizationDepthCalibration {
                    bias: 0.0,
                    cal: 3.25,
                    sample_count: 777,
                };
                2
            ]
        );

        let mut legacy_per_depth = vec![0_u8; 8 + 2 * 8];
        legacy_per_depth[..4].copy_from_slice(&LEGACY_PER_DEPTH_CALIBRATION_VERSION.to_le_bytes());
        legacy_per_depth[4..8].copy_from_slice(&2_u32.to_le_bytes());
        legacy_per_depth[8..12].copy_from_slice(&3.0_f32.to_le_bytes());
        legacy_per_depth[12..16].copy_from_slice(&500_u32.to_le_bytes());
        legacy_per_depth[16..20].copy_from_slice(&2.0_f32.to_le_bytes());
        legacy_per_depth[20..24].copy_from_slice(&600_u32.to_le_bytes());
        let decoded = VectorQuantizationCalibration::decode(&legacy_per_depth, 2).unwrap();
        assert_eq!(decoded.depths[0].bias, 0.0);
        assert_eq!(decoded.depths[0].cal, 3.0);
        assert_eq!(decoded.depths[1].bias, 0.0);
        assert_eq!(decoded.depths[1].cal, 2.0);
    }

    #[test]
    fn one_plus_four_is_496_bytes_per_l2_row() {
        let mut config = config(&[1, 4]);
        config.metric = Metric::L2;
        assert_eq!(config.bytes_per_row(), 496);
        assert!(config.needs_residual_norm());
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
                quantizer: VectorQuantizer::inferred(bits),
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
            assert_eq!(persisted.rho_model, Some(recomputed.rho_model));
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
    fn divisible_dimensions_retain_the_original_v3_stride() {
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
