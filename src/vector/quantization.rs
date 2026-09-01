//! Index-level configuration and fixed-stride layout for vector quantization.
//!
//! Configuration is persisted once in [`IndexSettings`](crate::IndexSettings),
//! keyed by vector field name, and reused by every segment. Field scoping is
//! necessary because an index may contain vector fields with different
//! dimensions and metrics; exact-density grids depend on the dimension.

use std::collections::BTreeSet;

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
}

// Grid points are finite after validation. Bit equality also distinguishes
// -0.0 and makes Eq sound for IndexSettings without weakening its contract.
impl PartialEq for VectorQuantizationGrid {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
            && self.version == other.version
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
        if self.dim == 0 || !self.dim.is_multiple_of(64) {
            return Err(invalid(format!(
                "dimension {} must be non-zero and divisible by 64",
                self.dim
            )));
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

        let mut required_grids = BTreeSet::new();
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
                    required_grids.insert(spec.bits);
                }
                VectorQuantizer::RaBitQ => {}
            }
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
        }
        if present_grids != required_grids {
            return Err(invalid(format!(
                "grid widths {present_grids:?} do not match TurboQuant widths {required_grids:?}"
            )));
        }
        Ok(())
    }

    /// Logical quantized bytes stored for one posting-membership row.
    pub fn bytes_per_row(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                quantized_code_stride(self.dim, layer.bits)
                    + QUANTIZED_SCALE_STRIDE
                    + QUANTIZED_CONSTANT_STRIDE
            })
            .sum()
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
    assert!(dim > 0 && dim.is_multiple_of(64));
    assert!((1..=4).contains(&bits));
    dim * usize::from(bits) / 8
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
    fn one_plus_four_is_492_bytes_per_posting_row() {
        assert_eq!(config(&[1, 4]).bytes_per_row(), 492);
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
    fn rejects_dimension_not_divisible_by_64() {
        let options = VectorOptions::new(769, Metric::Dot);
        let mut invalid = config(&[1, 4]);
        invalid.dim = 769;
        let err = invalid.validate(&options).unwrap_err();
        assert!(err.to_string().contains("divisible by 64"));
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
    fn index_builder_rejects_invalid_quantization_before_writing() {
        let mut builder = Schema::builder();
        builder.add_vector_field("embedding", VectorOptions::new(769, Metric::Dot));
        let schema = builder.build();
        let mut invalid = config(&[1, 4]);
        invalid.dim = 769;
        let settings = crate::IndexSettings {
            vector_quantization: vec![invalid],
            ..crate::IndexSettings::default()
        };

        let err = crate::Index::builder()
            .schema(schema)
            .settings(settings)
            .create_in_ram()
            .unwrap_err();
        assert!(err.to_string().contains("divisible by 64"));
    }
}
