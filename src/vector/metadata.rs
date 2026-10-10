//! Stored field descriptions and the column contract. See FORMAT.md.
use std::hash::{Hash, Hasher};
use std::{fmt, io};

use common::BinarySerializable;

use super::element::ElemType;
use super::quantization::{VectorNormPolicy, VectorQuantizationConfig, MAX_QUANTIZATION_LAYERS};
use crate::error::DataCorruption;
use crate::schema::{Metric, VectorDType, VectorOptions};

/// Default flat row-group size; the actual size is persisted with each field.
pub(crate) const FLAT_ROWS_PER_BLOCK: u32 = 16_384;

/// Per-field storage and decoding contract.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum VectorColMetadata {
    /// Full-precision rows without quantized columns.
    Plain(VectorFieldMeta),
    /// Full-precision rows plus ordered residual quantization layers.
    Quantized {
        /// Schema contract and row grouping.
        field: VectorFieldMeta,
        /// Ordered encoder and scorer contracts.
        layers: Vec<Quantizer>,
    },
}
/// Schema and block geometry stored in every Data entry.
#[derive(Clone, Debug)]
pub struct VectorFieldMeta {
    pub(crate) dim: u32,
    pub(crate) dtype: VectorDType,
    pub(crate) metric: Metric,
    pub(crate) norm_policy: VectorNormPolicy,
    pub(crate) partition: Partition,
}
/// Source of block row boundaries; cluster offsets reside in the centroid file.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Partition {
    /// One block per IVF cluster, with offsets stored in the centroid file.
    Clusters,
    /// Fixed-size row groups, with a possibly shorter final block.
    Uniform {
        /// Maximum number of rows in a block.
        rows_per_block: u32,
    },
}

/// The stable public name of a quantizer family; adding a quantizer adds a variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum QuantizerKind {
    /// One-bit signs scored as popcounted words.
    Sign,
    /// Packed scalar codes scored against reconstruction points.
    Grid,
}

impl QuantizerKind {
    /// Returns the stable public name of this quantizer family.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sign => "sign",
            Self::Grid => "grid",
        }
    }
}

impl fmt::Display for QuantizerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Ordered quantizer families and code widths, independent of rotation seeds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizationSchedule(Vec<(QuantizerKind, u8)>);

impl QuantizationSchedule {
    /// Constructs a schedule from its ordered layers.
    pub fn new(layers: Vec<(QuantizerKind, u8)>) -> Self {
        Self(layers)
    }

    /// Returns the quantizer family and code width of each layer in order.
    pub fn layers(&self) -> &[(QuantizerKind, u8)] {
        &self.0
    }
}

impl fmt::Display for QuantizationSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (index, (kind, bits)) in self.layers().iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{kind}:{bits}")?;
        }
        f.write_str("]")
    }
}
/// Tagged encode/decode contract; a semantic change requires a new variant.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Quantizer {
    /// One-bit signs scored as popcounted words.
    SignPlane {
        /// Transform applied before encoding and query preparation.
        rotation: Rotation,
        /// Exact model parameter bits.
        rho_model: F64Bits,
    },
    /// Packed scalar codes scored against persisted reconstruction points.
    GridPlane {
        /// Code width in bits.
        bits: u8,
        /// Transform applied before encoding and query preparation.
        rotation: Rotation,
        /// Persisted reconstruction and error model.
        grid: Grid,
    },
}
/// Pins the transform, random generator and seed expansion semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Rotation {
    /// Coordinates are used directly.
    None,
    /// Seeded FHT with ChaCha8 and the pinned seed expansion contract.
    SeededFhtChaCha8 {
        /// Seed used to construct the transform.
        seed: u64,
    },
}
/// Exact binary64 identity for persisted model parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct F64Bits(pub(crate) u64);
impl F64Bits {
    /// Exact stored bits, preserving signed zero and NaN payloads.
    pub fn to_bits(self) -> u64 {
        self.0
    }
    /// Stored model parameter interpreted as binary64.
    pub fn value(self) -> f64 {
        f64::from_bits(self.0)
    }
}
/// Persisted reconstruction points and error-model parameter.
#[derive(Clone, Debug)]
pub struct Grid {
    pub(crate) points: Vec<f32>,
    pub(crate) rho_model: f64,
}
impl VectorFieldMeta {
    /// Number of coordinates per vector.
    pub fn dim(&self) -> u32 {
        self.dim
    }
    /// Stored full-precision element representation.
    pub fn dtype(&self) -> VectorDType {
        self.dtype
    }
    /// Similarity metric used when the segment was built.
    pub fn metric(&self) -> Metric {
        self.metric
    }
    /// Normalization applied to stored rows.
    pub fn norm_policy(&self) -> VectorNormPolicy {
        self.norm_policy
    }
    /// Source of the segment's block row boundaries.
    pub fn partition(&self) -> &Partition {
        &self.partition
    }
}
impl Grid {
    /// Persisted reconstruction points in code order.
    pub fn points(&self) -> &[f32] {
        &self.points
    }
    /// Persisted error-model parameter.
    pub fn rho_model(&self) -> f64 {
        self.rho_model
    }
}
/// Compares every reconstruction point and model parameter by its exact stored bits.
impl PartialEq for Grid {
    fn eq(&self, other: &Self) -> bool {
        let Self { points, rho_model } = self;
        let Self {
            points: other_points,
            rho_model: other_rho,
        } = other;
        rho_model.to_bits() == other_rho.to_bits()
            && points.len() == other_points.len()
            && points
                .iter()
                .zip(other_points)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}
impl Eq for Grid {}
impl Hash for Grid {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Self { points, rho_model } = self;
        rho_model.to_bits().hash(state);
        points.len().hash(state);
        for point in points {
            point.to_bits().hash(state);
        }
    }
}

/// Derived block column; neither this record nor its stride is stored.
#[derive(Clone, Debug)]
pub(crate) struct Slot {
    pub slot_type: SlotType,
    pub elem: ElemType,
    pub stride: u32,
}
impl Slot {
    /// Width of the element consumed by the decoder, also the column alignment.
    pub(crate) fn type_bytes(&self) -> usize {
        debug_assert_eq!(self.elem, self.slot_type.elem());
        self.elem.size()
    }
}
/// Column semantics and ownership; the column list is part of the format.
#[derive(Clone, Debug)]
pub(crate) enum SlotType {
    Rows { dtype: VectorDType },
    ResidualNorms,
    QuantLayerCodes { layer: u8, quant: Quantizer },
    QuantLayerScales { layer: u8 },
    QuantLayerGammas { layer: u8 },
    QuantLayerErrors { layer: u8 },
    QuantLayerConstants { layer: u8 },
}
impl SlotType {
    /// Element required by this column's decoder contract.
    fn elem(&self) -> ElemType {
        match self {
            Self::Rows {
                dtype: VectorDType::F32,
            } => ElemType::F32,
            Self::QuantLayerCodes { quant, .. } => quant.codes_elem(),
            Self::QuantLayerGammas { .. } | Self::QuantLayerErrors { .. } => ElemType::F16,
            Self::ResidualNorms
            | Self::QuantLayerScales { .. }
            | Self::QuantLayerConstants { .. } => ElemType::F32,
        }
    }

    /// Norms share band zero so a first-layer scan needs one contiguous read.
    pub(crate) fn band(&self) -> Option<u8> {
        match self {
            Self::Rows { .. } => None,
            Self::ResidualNorms => Some(0),
            Self::QuantLayerCodes { layer, .. }
            | Self::QuantLayerScales { layer }
            | Self::QuantLayerGammas { layer }
            | Self::QuantLayerErrors { layer }
            | Self::QuantLayerConstants { layer } => Some(*layer),
        }
    }
}
impl Quantizer {
    /// Returns the quantizer family independently of its width and rotation.
    pub const fn kind(&self) -> QuantizerKind {
        match self {
            Self::SignPlane { .. } => QuantizerKind::Sign,
            Self::GridPlane { .. } => QuantizerKind::Grid,
        }
    }

    /// Sign codes are decoded as words; grid codes are decoded as bytes.
    pub(crate) const fn codes_elem(&self) -> ElemType {
        match self {
            Self::SignPlane { .. } => ElemType::U64,
            Self::GridPlane { .. } => ElemType::U8,
        }
    }

    /// Returns the numeric width without using it to infer the quantizer kind.
    pub fn bits(&self) -> u8 {
        match self {
            Self::SignPlane { .. } => 1,
            Self::GridPlane { bits, .. } => *bits,
        }
    }
    /// Returns the tagged transform for this encoding contract.
    pub fn rotation(&self) -> Rotation {
        match self {
            Self::SignPlane { rotation, .. } | Self::GridPlane { rotation, .. } => *rotation,
        }
    }
    /// Bytes per row of codes under this quantizer's packing contract.
    pub(crate) fn code_stride(&self, dim: usize) -> usize {
        match self {
            Self::SignPlane { .. } => dim.div_ceil(u64::BITS as usize) * ElemType::U64.size(),
            Self::GridPlane { bits, .. } => cascade::grid_code_stride(dim, *bits),
        }
    }
    /// Ordered owned columns, frozen by the quantizer tag; see FORMAT.md.
    pub(crate) fn layer_slots(&self, layer: u8, dim: u32, metric: Metric) -> Vec<Slot> {
        let mut slots = vec![
            Slot {
                slot_type: SlotType::QuantLayerCodes {
                    layer,
                    quant: self.clone(),
                },
                elem: self.codes_elem(),
                stride: self.code_stride(dim as usize) as u32,
            },
            Slot {
                slot_type: SlotType::QuantLayerScales { layer },
                elem: ElemType::F32,
                stride: ElemType::F32.size() as u32,
            },
            Slot {
                slot_type: SlotType::QuantLayerGammas { layer },
                elem: ElemType::F16,
                stride: ElemType::F16.size() as u32,
            },
            Slot {
                slot_type: SlotType::QuantLayerErrors { layer },
                elem: ElemType::F16,
                stride: ElemType::F16.size() as u32,
            },
        ];
        if metric == Metric::L2 {
            slots.push(Slot {
                slot_type: SlotType::QuantLayerConstants { layer },
                elem: ElemType::F32,
                stride: ElemType::F32.size() as u32,
            });
        }
        slots
    }
}
fn metric_tag(metric: Metric) -> u8 {
    match metric {
        Metric::L2 => 0,
        Metric::Dot => 1,
        Metric::Cosine => 2,
    }
}
impl VectorColMetadata {
    /// Resolves the field contract without consulting mutable index settings.
    pub fn field(&self) -> &VectorFieldMeta {
        match self {
            Self::Plain(field) | Self::Quantized { field, .. } => field,
        }
    }
    /// Returns the ordered quantizers, or no layers for plain row storage.
    pub fn layers(&self) -> &[Quantizer] {
        match self {
            Self::Plain(_) => &[],
            Self::Quantized { layers, .. } => layers,
        }
    }
    /// Logical quantized bytes per row, including residual norms and layer columns.
    /// Excludes full-precision rows and alignment padding; plain storage returns `None`.
    pub fn quantized_bytes_per_row(&self) -> Option<usize> {
        match self {
            Self::Plain(_) => None,
            Self::Quantized { .. } => Some(
                self.slots()
                    .iter()
                    .filter(|slot| slot.slot_type.band().is_some())
                    .map(|slot| slot.stride as usize)
                    .sum(),
            ),
        }
    }
    /// Rows precede residual norms and ordered layer bands.
    pub(crate) fn slots(&self) -> Vec<Slot> {
        let field = self.field();
        let mut slots = vec![Slot {
            slot_type: SlotType::Rows { dtype: field.dtype },
            elem: ElemType::F32,
            stride: field.dim * ElemType::F32.size() as u32,
        }];
        if let Self::Quantized { layers, .. } = self {
            slots.push(Slot {
                slot_type: SlotType::ResidualNorms,
                elem: ElemType::F32,
                stride: ElemType::F32.size() as u32,
            });
            for (layer, quant) in layers.iter().enumerate() {
                slots.extend(quant.layer_slots(layer as u8, field.dim, field.metric));
            }
        }
        slots
    }
    fn base(opts: &VectorOptions, partition: Partition) -> VectorFieldMeta {
        VectorFieldMeta {
            dim: opts.dim() as u32,
            dtype: opts.dtype(),
            metric: opts.metric(),
            norm_policy: VectorNormPolicy::for_options(opts),
            partition,
        }
    }
    /// Builds clustered metadata from the index's write target, persisting all models.
    pub(crate) fn build_ivf(
        opts: &VectorOptions,
        config: Option<&VectorQuantizationConfig>,
    ) -> crate::Result<Self> {
        let field = Self::base(opts, Partition::Clusters);
        let Some(config) = config else {
            return Ok(Self::Plain(field));
        };
        config.validate(opts)?;
        let layers = config
            .layers
            .iter()
            .map(|layer| {
                let grid = config
                    .grids
                    .iter()
                    .find(|grid| grid.bits == layer.bits)
                    .ok_or_else(|| {
                        crate::TantivyError::DataCorruption(DataCorruption::comment_only(
                            "invalid vector metadata: missing persisted grid/model",
                        ))
                    })?;
                let rotation = Rotation::SeededFhtChaCha8 { seed: layer.seed };
                Ok(if layer.bits == 1 {
                    Quantizer::SignPlane {
                        rotation,
                        rho_model: F64Bits(grid.rho_model.to_bits()),
                    }
                } else {
                    Quantizer::GridPlane {
                        bits: layer.bits,
                        rotation,
                        grid: Grid {
                            points: grid.points.clone(),
                            rho_model: grid.rho_model,
                        },
                    }
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Self::Quantized { field, layers })
    }
    /// Flat blocks contain only full-precision rows.
    pub(crate) fn build_flat(opts: &VectorOptions) -> Self {
        Self::Plain(Self::base(
            opts,
            Partition::Uniform {
                rows_per_block: FLAT_ROWS_PER_BLOCK,
            },
        ))
    }
    /// Validates stored values before deriving strides or allocating block tables.
    pub(crate) fn validate(&self, opts: &VectorOptions, clustered: bool) -> crate::Result<()> {
        let f = self.field();
        if f.dim as usize != opts.dim()
            || f.dim == 0
            || f.dim > u32::MAX / 4
            || f.dtype != opts.dtype()
            || f.metric != opts.metric()
            || f.norm_policy != VectorNormPolicy::for_options(opts)
        {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only("invalid vector metadata: schema mismatch"),
            ));
        }
        match f.partition {
            Partition::Clusters if clustered => (),
            Partition::Uniform { rows_per_block } if !clustered && rows_per_block > 0 => (),
            _ => {
                return Err(crate::TantivyError::DataCorruption(
                    DataCorruption::comment_only(
                        "invalid vector metadata: partition/backend mismatch or zero block size",
                    ),
                ))
            }
        }
        if let Self::Quantized { layers, .. } = self {
            if !(1..=MAX_QUANTIZATION_LAYERS).contains(&layers.len()) {
                return Err(crate::TantivyError::DataCorruption(
                    DataCorruption::comment_only("invalid vector metadata: layer count"),
                ));
            }
            super::quantization::validate_quantization_values(
                f.dim as usize,
                layers.iter().map(|q| match q {
                    Quantizer::SignPlane { rho_model, .. } => (rho_model.value(), None),
                    Quantizer::GridPlane { grid, .. } => {
                        (grid.rho_model, Some(grid.points.as_slice()))
                    }
                }),
            )
            .map_err(|message| {
                crate::TantivyError::DataCorruption(DataCorruption::comment_only(format!(
                    "invalid vector metadata: {message}"
                )))
            })?;
            for quant in layers {
                if let Quantizer::GridPlane { bits, grid, .. } = quant {
                    if !(2..=4).contains(bits) || grid.points.len() != 1 << bits {
                        return Err(crate::TantivyError::DataCorruption(
                            DataCorruption::comment_only(
                                "invalid vector metadata: grid width or point count",
                            ),
                        ));
                    }
                }
            }
        }
        if self
            .slots()
            .iter()
            .any(|slot| slot.stride as usize % slot.type_bytes() != 0)
        {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only(
                    "invalid vector metadata: stride is not a whole number of decoder elements",
                ),
            ));
        }
        Ok(())
    }
    /// Resolves tagged kernel contracts and persisted models for encoding or query preparation.
    pub(crate) fn runtime(&self) -> (Vec<cascade::LayerSpec>, Vec<quant_model::Grid>) {
        self.layers()
            .iter()
            .map(|q| {
                let rotation = match q.rotation() {
                    Rotation::None => cascade::Rotation::None,
                    Rotation::SeededFhtChaCha8 { seed } => {
                        cascade::Rotation::SeededFhtChaCha8 { seed }
                    }
                };
                match q {
                    Quantizer::SignPlane { rho_model, .. } => (
                        cascade::LayerSpec {
                            kind: cascade::LayerKind::Sign,
                            bits: 1,
                            rotation,
                        },
                        quant_model::Grid {
                            bits: 1,
                            points: Vec::new(),
                            rho_model: f64::from_bits(rho_model.0),
                        },
                    ),
                    Quantizer::GridPlane { bits, grid, .. } => (
                        cascade::LayerSpec {
                            kind: cascade::LayerKind::Grid,
                            bits: *bits,
                            rotation,
                        },
                        quant_model::Grid {
                            bits: *bits,
                            points: grid.points.clone(),
                            rho_model: grid.rho_model,
                        },
                    ),
                }
            })
            .unzip()
    }
    /// Serializes the metadata grammar using explicit tag maps, never Rust discriminants.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(u8::from(matches!(self, Self::Quantized { .. })));
        let f = self.field();
        out.extend(f.dim.to_le_bytes());
        out.push(match f.dtype {
            VectorDType::F32 => 0,
        });
        out.push(metric_tag(f.metric));
        out.push(match f.norm_policy {
            VectorNormPolicy::None => 0,
            VectorNormPolicy::UnitL2 => 1,
        });
        match f.partition {
            Partition::Clusters => out.push(0),
            Partition::Uniform { rows_per_block } => {
                out.push(1);
                out.extend(rows_per_block.to_le_bytes());
            }
        }
        if let Self::Quantized { layers, .. } = self {
            out.push(layers.len() as u8);
            for quant in layers {
                match quant {
                    Quantizer::SignPlane {
                        rotation,
                        rho_model,
                    } => {
                        out.push(0);
                        rotation.write(&mut out);
                        out.extend(rho_model.0.to_le_bytes());
                    }
                    Quantizer::GridPlane {
                        bits,
                        rotation,
                        grid,
                    } => {
                        out.extend([1, *bits]);
                        rotation.write(&mut out);
                        out.extend(grid.rho_model.to_le_bytes());
                        out.extend((grid.points.len() as u16).to_le_bytes());
                        for p in &grid.points {
                            out.extend(p.to_le_bytes());
                        }
                    }
                }
            }
        }
        out
    }
    /// Parses a bounded metadata record, rejecting unknown tags and trailing bytes.
    pub(crate) fn from_bytes(mut input: &[u8]) -> crate::Result<Self> {
        fn parse(input: &mut &[u8]) -> io::Result<VectorColMetadata> {
            let repr = u8::deserialize(input)?;
            let dim = u32::deserialize(input)?;
            let dtype = match u8::deserialize(input)? {
                0 => VectorDType::F32,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unknown metadata tag; rebuild required",
                    ))
                }
            };
            let metric = match u8::deserialize(input)? {
                0 => Metric::L2,
                1 => Metric::Dot,
                2 => Metric::Cosine,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unknown metadata tag; rebuild required",
                    ))
                }
            };
            let norm_policy = match u8::deserialize(input)? {
                0 => VectorNormPolicy::None,
                1 => VectorNormPolicy::UnitL2,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unknown metadata tag; rebuild required",
                    ))
                }
            };
            let partition = match u8::deserialize(input)? {
                0 => Partition::Clusters,
                1 => Partition::Uniform {
                    rows_per_block: u32::deserialize(input)?,
                },
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unknown metadata tag; rebuild required",
                    ))
                }
            };
            let field = VectorFieldMeta {
                dim,
                dtype,
                metric,
                norm_policy,
                partition,
            };
            Ok(match repr {
                0 => VectorColMetadata::Plain(field),
                1 => {
                    let count = u8::deserialize(input)? as usize;
                    if !(1..=MAX_QUANTIZATION_LAYERS).contains(&count) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "unknown metadata tag; rebuild required",
                        ));
                    }
                    let mut layers = Vec::with_capacity(count);
                    for _ in 0..count {
                        layers.push(match u8::deserialize(input)? {
                            0 => Quantizer::SignPlane {
                                rotation: Rotation::read(input)?,
                                rho_model: F64Bits(u64::deserialize(input)?),
                            },
                            1 => {
                                let bits = u8::deserialize(input)?;
                                let rotation = Rotation::read(input)?;
                                let rho_model = f64::from_bits(u64::deserialize(input)?);
                                let count = u16::deserialize(input)? as usize;
                                if !(2..=4).contains(&bits) || count != 1 << bits {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "unknown metadata tag; rebuild required",
                                    ));
                                }
                                let points = (0..count)
                                    .map(|_| u32::deserialize(input).map(f32::from_bits))
                                    .collect::<io::Result<Vec<_>>>()?;
                                Quantizer::GridPlane {
                                    bits,
                                    rotation,
                                    grid: Grid { rho_model, points },
                                }
                            }
                            _ => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "unknown metadata tag; rebuild required",
                                ))
                            }
                        });
                    }
                    VectorColMetadata::Quantized { field, layers }
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unknown metadata tag; rebuild required",
                    ))
                }
            })
        }
        let meta = parse(&mut input).map_err(|e| {
            crate::TantivyError::DataCorruption(DataCorruption::comment_only(format!(
                "invalid vector metadata: {e}"
            )))
        })?;
        if !input.is_empty() {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only("invalid vector metadata: trailing metadata bytes"),
            ));
        }
        Ok(meta)
    }
}
impl Rotation {
    fn write(self, out: &mut Vec<u8>) {
        match self {
            Self::None => out.push(0),
            Self::SeededFhtChaCha8 { seed } => {
                out.push(1);
                out.extend(seed.to_le_bytes());
            }
        }
    }
    fn read(input: &mut &[u8]) -> io::Result<Self> {
        match u8::deserialize(input)? {
            0 => Ok(Self::None),
            1 => Ok(Self::SeededFhtChaCha8 {
                seed: u64::deserialize(input)?,
            }),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown rotation tag; rebuild required",
            )),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::super::quantization::VectorQuantizationLayer;
    use super::*;
    fn metadata(metric: Metric, schedule: &[u8]) -> VectorColMetadata {
        let opts = VectorOptions::new(100, metric);
        let config = VectorQuantizationConfig::materialize(
            "v".into(),
            &opts,
            schedule
                .iter()
                .map(|&bits| VectorQuantizationLayer { bits, seed: 17 })
                .collect(),
        )
        .unwrap();
        VectorColMetadata::build_ivf(&opts, Some(&config)).unwrap()
    }

    #[test]
    fn quantizer_kind_and_schedule_names() {
        for (kind, name) in [(QuantizerKind::Sign, "sign"), (QuantizerKind::Grid, "grid")] {
            assert_eq!(kind.name(), name);
            assert_eq!(kind.to_string(), name);
        }
        let meta = metadata(Metric::L2, &[1, 4]);
        assert_eq!(meta.layers()[0].kind(), QuantizerKind::Sign);
        assert_eq!(meta.layers()[1].kind(), QuantizerKind::Grid);
        let schedule =
            QuantizationSchedule::new(vec![(QuantizerKind::Sign, 1), (QuantizerKind::Grid, 4)]);
        assert_eq!(schedule.to_string(), "[sign:1, grid:4]");
    }
    // Pins ordered slot semantics, widths, strides, and scan-band ownership.
    #[test]
    fn golden_slot_contract() {
        for clustered in [false, true] {
            for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
                for schedule in [&[1][..], &[2], &[3], &[4], &[1, 4], &[1, 2, 4]] {
                    let mut meta = metadata(metric, schedule);
                    if !clustered {
                        if let VectorColMetadata::Quantized { field, .. } = &mut meta {
                            field.partition = Partition::Uniform {
                                rows_per_block: FLAT_ROWS_PER_BLOCK,
                            };
                        }
                    }
                    let norms = 1;
                    let slots = meta.slots();
                    assert!(matches!(
                        slots[0].slot_type,
                        SlotType::Rows {
                            dtype: VectorDType::F32
                        }
                    ));
                    assert_eq!(
                        (slots[0].elem, slots[0].stride, slots[0].slot_type.band()),
                        (ElemType::F32, 400, None)
                    );
                    assert!(matches!(slots[norms].slot_type, SlotType::ResidualNorms));
                    assert_eq!(
                        (
                            slots[norms].elem,
                            slots[norms].stride,
                            slots[norms].slot_type.band()
                        ),
                        (ElemType::F32, 4, Some(0))
                    );
                    let per_layer = if metric == Metric::L2 { 5 } else { 4 };
                    assert_eq!(slots.len(), norms + 1 + schedule.len() * per_layer);
                    for (l, &bits) in schedule.iter().enumerate() {
                        let layer = l as u8;
                        let cols =
                            &slots[norms + 1 + l * per_layer..norms + 1 + (l + 1) * per_layer];
                        assert!(
                            matches!(&cols[0].slot_type, SlotType::QuantLayerCodes { layer: n, quant } if *n == layer && quant.bits() == bits)
                        );
                        assert!(
                            matches!(cols[1].slot_type, SlotType::QuantLayerScales { layer: n } if n == layer)
                        );
                        assert!(
                            matches!(cols[2].slot_type, SlotType::QuantLayerGammas { layer: n } if n == layer)
                        );
                        assert!(
                            matches!(cols[3].slot_type, SlotType::QuantLayerErrors { layer: n } if n == layer)
                        );
                        let mut expected = vec![
                            (
                                if bits == 1 {
                                    ElemType::U64
                                } else {
                                    ElemType::U8
                                },
                                match bits {
                                    1 => 16,
                                    2 => 32,
                                    3 => 40,
                                    4 => 56,
                                    _ => unreachable!(),
                                },
                            ),
                            (ElemType::F32, 4),
                            (ElemType::F16, 2),
                            (ElemType::F16, 2),
                        ];
                        if metric == Metric::L2 {
                            assert!(
                                matches!(cols[4].slot_type, SlotType::QuantLayerConstants { layer: n } if n == layer)
                            );
                            expected.push((ElemType::F32, 4));
                        }
                        assert_eq!(
                            cols.iter().map(|c| (c.elem, c.stride)).collect::<Vec<_>>(),
                            expected
                        );
                        assert!(cols.iter().all(|c| c.slot_type.band() == Some(layer)));
                    }
                    assert_eq!(
                        VectorColMetadata::from_bytes(&meta.to_bytes())
                            .unwrap()
                            .to_bytes(),
                        meta.to_bytes()
                    );
                }
                let opts = VectorOptions::new(100, metric);
                let plain = if clustered {
                    VectorColMetadata::build_ivf(&opts, None).unwrap()
                } else {
                    VectorColMetadata::build_flat(&opts)
                };
                assert_eq!(plain.slots().len(), 1);
                assert!(matches!(
                    plain.slots()[0].slot_type,
                    SlotType::Rows {
                        dtype: VectorDType::F32
                    }
                ));
                assert_eq!(plain.slots()[0].slot_type.band(), None);
                assert_eq!(plain.slots()[0].stride, 400);
                assert_eq!(plain.slots()[0].elem, ElemType::F32);
                assert_eq!(
                    VectorColMetadata::from_bytes(&plain.to_bytes())
                        .unwrap()
                        .to_bytes(),
                    plain.to_bytes()
                );
            }
        }
    }
    #[test]
    fn invalid_model_values_fail_at_open() {
        use crate::directory::FileSlice;
        use crate::vector::blocks::{write_metadata, BlockMetadata};
        for case in 0..7 {
            let mut meta = metadata(Metric::L2, &[1, 4]);
            let VectorColMetadata::Quantized { field, layers } = &mut meta else {
                unreachable!()
            };
            match case {
                0..=3 => {
                    let Quantizer::GridPlane { grid, .. } = &mut layers[1] else {
                        unreachable!()
                    };
                    match case {
                        0 => grid.points[0] = f32::NAN,
                        1 => grid.points[1] = grid.points[0],
                        2 => grid.rho_model = -1.0,
                        _ => grid.rho_model = f64::NAN,
                    }
                }
                4 | 5 => {
                    let Quantizer::SignPlane { rho_model, .. } = &mut layers[0] else {
                        unreachable!()
                    };
                    rho_model.0 = if case == 4 {
                        (-1.0_f64).to_bits()
                    } else {
                        f64::NAN.to_bits()
                    };
                }
                _ => field.dim = 63,
            }
            let opts = VectorOptions::new(field.dim as usize, Metric::L2);
            let mut bytes = Vec::new();
            write_metadata(&mut bytes, &meta).unwrap();
            assert!(
                matches!(
                    BlockMetadata::open(FileSlice::from(bytes), &opts, true),
                    Err(crate::TantivyError::DataCorruption(_))
                ),
                "model corruption case {case}"
            );
        }
    }

    // Unknown tags and malformed field contracts must fail before opening payloads.
    #[test]
    fn metadata_rejects_corruption() {
        let meta = metadata(Metric::L2, &[1]);
        for offset in [0, 5, 6, 7, 8, 10, 11] {
            let mut bytes = meta.to_bytes();
            bytes[offset] = 255;
            assert!(
                VectorColMetadata::from_bytes(&bytes).is_err(),
                "offset {offset}"
            );
        }
        let opts = VectorOptions::new(100, Metric::L2);
        for change in 0..5 {
            let mut broken = meta.clone();
            if let VectorColMetadata::Quantized { field, layers } = &mut broken {
                match change {
                    0 => field.dim = 0,
                    1 => field.metric = Metric::Dot,
                    2 => field.norm_policy = VectorNormPolicy::UnitL2,
                    3 => field.partition = Partition::Uniform { rows_per_block: 0 },
                    _ => layers.clear(),
                }
            }
            assert!(matches!(
                broken.validate(&opts, true),
                Err(crate::TantivyError::DataCorruption(_))
            ));
        }
        assert!(meta.validate(&opts, false).is_err());
        assert!(VectorColMetadata::build_flat(&opts)
            .validate(&opts, true)
            .is_err());
    }
    // Exact byte sequences freeze metadata framing and the explicit tag assignments.
    #[test]
    fn quantizer_equality_and_hash_cover_every_field() {
        fn hash(value: &impl Hash) -> u64 {
            let mut state = std::collections::hash_map::DefaultHasher::new();
            value.hash(&mut state);
            state.finish()
        }
        let sign = Quantizer::SignPlane {
            rotation: Rotation::SeededFhtChaCha8 { seed: 7 },
            rho_model: F64Bits(0.25f64.to_bits()),
        };
        assert_eq!(sign, sign.clone());
        assert_eq!(hash(&sign), hash(&sign.clone()));
        for change in 0..3 {
            let mut other = sign.clone();
            let Quantizer::SignPlane {
                rotation,
                rho_model,
            } = &mut other
            else {
                unreachable!()
            };
            match change {
                0 => *rotation = Rotation::None,
                1 => *rotation = Rotation::SeededFhtChaCha8 { seed: 11 },
                _ => rho_model.0 ^= 1,
            }
            assert_ne!(sign, other, "sign field change {change}");
            assert_ne!(hash(&sign), hash(&other), "sign hash field change {change}");
        }
        let grid = Quantizer::GridPlane {
            bits: 2,
            rotation: Rotation::SeededFhtChaCha8 { seed: 7 },
            grid: Grid {
                points: vec![-1.0, -0.5, 0.5, 1.0],
                rho_model: 0.25,
            },
        };
        assert_ne!(sign, grid);
        assert_eq!(grid, grid.clone());
        assert_eq!(hash(&grid), hash(&grid.clone()));
        for change in 0..7 {
            let mut other = grid.clone();
            let Quantizer::GridPlane {
                bits,
                rotation,
                grid: values,
            } = &mut other
            else {
                unreachable!()
            };
            match change {
                0 => *bits = 3,
                1 => *rotation = Rotation::None,
                2 => *rotation = Rotation::SeededFhtChaCha8 { seed: 11 },
                3 => values.rho_model = f64::from_bits(values.rho_model.to_bits() ^ 1),
                4 => values.points[1] = f32::from_bits(values.points[1].to_bits() ^ 1),
                5 => {
                    values.points.pop();
                }
                _ => values.points.swap(0, 1),
            }
            assert_ne!(grid, other, "grid field change {change}");
            assert_ne!(hash(&grid), hash(&other), "grid hash field change {change}");
        }
        for (point, rho) in [
            (0.0, 0.0),
            (
                f32::from_bits(0x7fc00001),
                f64::from_bits(0x7ff8000000000001),
            ),
        ] {
            let original = Grid {
                points: vec![point],
                rho_model: rho,
            };
            assert_eq!(original, original.clone());
            assert_eq!(hash(&original), hash(&original.clone()));
            let mut changed = original.clone();
            changed.points[0] = f32::from_bits(point.to_bits() ^ 0x80000000);
            assert_ne!(original, changed);
            assert_ne!(hash(&original), hash(&changed));
            let mut changed = original.clone();
            changed.rho_model = f64::from_bits(rho.to_bits() ^ 0x8000000000000000);
            assert_ne!(original, changed);
            assert_ne!(hash(&original), hash(&changed));
        }
    }

    #[test]
    fn grid_metadata_grammar_literal() {
        let mut meta = metadata(Metric::Dot, &[2]);
        let VectorColMetadata::Quantized { layers, .. } = &mut meta else {
            unreachable!()
        };
        layers[0] = Quantizer::GridPlane {
            bits: 2,
            rotation: Rotation::None,
            grid: Grid {
                points: vec![-1.0, -0.5, 0.5, 1.0],
                rho_model: 0.25,
            },
        };
        let expected = [
            1, 100, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 208, 63, 4, 0, 0, 0, 128,
            191, 0, 0, 0, 191, 0, 0, 0, 63, 0, 0, 128, 63,
        ];
        assert_eq!(meta.to_bytes(), expected);
        let decoded = VectorColMetadata::from_bytes(&expected).unwrap();
        assert_eq!(meta.layers(), decoded.layers());
        assert_eq!(decoded.to_bytes(), expected);
    }

    #[test]
    fn metadata_grammar_golden_bytes() {
        let opts = VectorOptions::new(100, Metric::Dot);
        let plain = VectorColMetadata::build_flat(&opts);
        assert_eq!(plain.to_bytes(), [0, 100, 0, 0, 0, 0, 1, 0, 1, 0, 64, 0, 0]);
        let mut quant = metadata(Metric::Dot, &[1]);
        if let VectorColMetadata::Quantized { layers, .. } = &mut quant {
            layers[0] = Quantizer::SignPlane {
                rotation: Rotation::None,
                rho_model: F64Bits(1.0f64.to_bits()),
            };
        }
        let mut expected = vec![1, 100, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0];
        expected.extend(1.0f64.to_le_bytes());
        assert_eq!(quant.to_bytes(), expected);
        assert_eq!(
            VectorColMetadata::from_bytes(&expected).unwrap().to_bytes(),
            expected
        );
        assert!(VectorColMetadata::from_bytes(&expected[..expected.len() - 1]).is_err());
        expected.push(0);
        assert!(VectorColMetadata::from_bytes(&expected).is_err());
    }

    // Invalid layer count, grid width and grid cardinality cannot reach kernel dispatch.
    #[test]
    fn metadata_rejects_invalid_layer_contracts() {
        let opts = VectorOptions::new(100, Metric::L2);
        for change in 0..3 {
            let mut meta = metadata(Metric::L2, &[4]);
            if let VectorColMetadata::Quantized { layers, .. } = &mut meta {
                match change {
                    0 => layers.resize(4, layers[0].clone()),
                    1 => {
                        if let Quantizer::GridPlane { bits, .. } = &mut layers[0] {
                            *bits = 1;
                        }
                    }
                    _ => {
                        if let Quantizer::GridPlane { grid, .. } = &mut layers[0] {
                            grid.points.pop();
                        }
                    }
                }
            }
            assert!(meta.validate(&opts, true).is_err());
        }
    }
}
