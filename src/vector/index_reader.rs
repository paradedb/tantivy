//! Per-(segment, field) vector reader, modeled on
//! [`InvertedIndexReader`](crate::index::InvertedIndexReader).
//!
//! One [`VectorIndexReader`] serves one vector field of one segment, opened
//! (and cached) via
//! [`SegmentReader::vector_index`](crate::SegmentReader::vector_index). Small
//! routing state is parsed once and pinned in memory, while the bulk payload
//! stays behind [`FileSlice`]s and is fetched with ranged reads at query time.
//!
//! The reader is "store + optional index":
//! - The **store** is the segment's `.vec` composite: slot `[0]` is the row→doc_id [`IdMap`], slot
//!   `[1]` the dense vector rows (deferred).
//! - The **index** ([`IvfIndex`]) exists if the segment was merged with IVF clustering, and is
//!   loaded from the sibling `.centroids` composite. It tells a query which clusters — contiguous
//!   row ranges of slot `[1]` — to probe. Without it, search falls back to an exact scan.
//!
//! The pairing is an invariant of the write path (`VectorPlugin`): the IVF
//! merge writes cluster-sorted rows + an `Explicit` id-map + `.centroids`;
//! every other writer produces doc-ordered rows + `Identity`/`Bitmap` and no
//! sidecar. [`VectorIndexReader::open`] validates the two signals agree.

use std::cmp::Ordering;
use std::ops::Range;
use std::sync::Arc;

#[cfg(test)]
use cascade::{encode_batch_in_place, prepare_centroid, prepare_fp_query};
use common::{HasLen, OwnedBytes};
#[cfg(test)]
use quant_model::f16::f16_to_f32;

use super::flat::IdMap;
use super::header::{centroid_slot, read_header, vec_slot, VectorFileVersion};
#[cfg(test)]
use super::ivf::decode_row;
use super::ivf::{IvfIndex, CENTROIDS_EXT};
use super::prepared::QuantizedIndexCtx;
use super::quantization::{
    quantized_calibration_metadata_len, quantized_code_stride, quantized_code_tail_is_zero,
    VectorQuantizationCalibration, VectorQuantizationConfig, QUANTIZED_CALIBRATION_VERSION,
    QUANTIZED_CONSTANT_STRIDE, QUANTIZED_RESIDUAL_NORM_STRIDE, QUANTIZED_SCALE_STRIDE,
};
use super::VEC_EXT;
use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::error::DataCorruption;
use crate::index::SegmentComponent;
#[cfg(test)]
use crate::schema::Metric;
use crate::schema::{Field, FieldType, VectorOptions};
use crate::{DocId, SegmentReader, TantivyError};

/// Which on-disk layout a segment's vector data uses, surfaced through
/// [`VectorInfo`] for tooling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorStorageFormat {
    Flat,
    Ivf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorInfo {
    pub format: VectorStorageFormat,
    /// Distinct documents with a vector in this field. The per-cluster
    /// numbers (`cluster_stats`, [`VectorIndexReader::cluster_sizes`]) count
    /// posting rows, so with legacy V2 replication their sum could exceed
    /// `num_vectors`.
    pub num_vectors: usize,
    pub num_centroids: Option<usize>,
    pub cluster_stats: Option<VectorClusterStats>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorClusterStats {
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub avg_cluster_size: f64,
    pub empty_clusters: usize,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizationCalibrationAudit {
    pub depth: usize,
    pub sample_count: usize,
    pub cal: f64,
    pub signed_mean: f64,
    pub signed_stddev: f64,
    pub signed_skewness: f64,
    pub signed_excess_kurtosis: f64,
    pub abs_p50: f64,
    pub abs_p90: f64,
    pub abs_p95: f64,
    pub abs_p99: f64,
    pub abs_max: f64,
}

/// Deferred fixed-stride slices for one residual plane.
pub(crate) struct QuantizedLayerReader {
    codes: FileSlice,
    scales: FileSlice,
    constants: FileSlice,
    code_stride: usize,
    dim: usize,
    bits: u8,
}

/// Three pinned SoA ranges for one contiguous cluster posting.
pub(crate) struct QuantizedLayerBatch {
    codes: OwnedBytes,
    scales: OwnedBytes,
    constants: OwnedBytes,
    rows: Range<usize>,
    code_stride: usize,
    dim: usize,
    bits: u8,
}

/// Reusable dense SoA stream gathered from ascending survivor row ids. The
/// gatherer pins each touched code block once; the cascade kernel then sees
/// one fixed-stride batch rather than one call per survivor.
pub(crate) struct QuantizedLayerDenseBatch {
    codes: Vec<u8>,
    scales: Vec<u8>,
    constants: Vec<u8>,
    code_stride: usize,
}

impl QuantizedLayerDenseBatch {
    pub(crate) fn new() -> Self {
        Self {
            codes: Vec::new(),
            scales: Vec::new(),
            constants: Vec::new(),
            code_stride: 0,
        }
    }

    pub(crate) fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub(crate) fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub(crate) fn constants(&self) -> &[u8] {
        &self.constants
    }

    pub(crate) fn code_stride(&self) -> usize {
        self.code_stride
    }
}

impl QuantizedLayerBatch {
    fn local_row(&self, row: usize) -> crate::Result<usize> {
        if !self.rows.contains(&row) {
            return Err(TantivyError::InternalError(format!(
                "quantized row {row} is outside pinned range {:?}",
                self.rows
            )));
        }
        Ok(row - self.rows.start)
    }

    pub(crate) fn code_bytes(&self, row: usize) -> crate::Result<&[u8]> {
        let local = self.local_row(row)?;
        let start = local * self.code_stride;
        let bytes = &self.codes[start..start + self.code_stride];
        if !quantized_code_tail_is_zero(bytes, self.dim, self.bits) {
            return Err(DataCorruption::comment_only(format!(
                "quantized row {row} has non-zero padding bits for d={} b={}",
                self.dim, self.bits
            ))
            .into());
        }
        Ok(bytes)
    }

    pub(crate) fn scale(&self, row: usize) -> crate::Result<u16> {
        let local = self.local_row(row)?;
        let start = local * QUANTIZED_SCALE_STRIDE;
        Ok(u16::from_le_bytes(
            self.scales[start..start + QUANTIZED_SCALE_STRIDE]
                .try_into()
                .unwrap(),
        ))
    }

    pub(crate) fn constant(&self, row: usize) -> crate::Result<f32> {
        let local = self.local_row(row)?;
        let start = local * QUANTIZED_CONSTANT_STRIDE;
        Ok(f32::from_le_bytes(
            self.constants[start..start + QUANTIZED_CONSTANT_STRIDE]
                .try_into()
                .unwrap(),
        ))
    }

    pub(crate) fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub(crate) fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub(crate) fn constants(&self) -> &[u8] {
        &self.constants
    }

    pub(crate) fn code_stride(&self) -> usize {
        self.code_stride
    }
}

impl QuantizedLayerReader {
    pub(crate) fn read_batch(
        &self,
        rows: Range<usize>,
        read_constants: bool,
    ) -> crate::Result<QuantizedLayerBatch> {
        if rows.start > rows.end {
            return Err(TantivyError::InternalError(format!(
                "invalid quantized row range {rows:?}"
            )));
        }
        let codes = self
            .codes
            .slice(rows.start * self.code_stride..rows.end * self.code_stride)
            .read_bytes()?;
        let scales = self
            .scales
            .slice(rows.start * QUANTIZED_SCALE_STRIDE..rows.end * QUANTIZED_SCALE_STRIDE)
            .read_bytes()?;
        let constants = if read_constants {
            self.constants
                .slice(rows.start * QUANTIZED_CONSTANT_STRIDE..rows.end * QUANTIZED_CONSTANT_STRIDE)
                .read_bytes()?
        } else {
            OwnedBytes::empty()
        };
        Ok(QuantizedLayerBatch {
            codes,
            scales,
            constants,
            rows,
            code_stride: self.code_stride,
            dim: self.dim,
            bits: self.bits,
        })
    }

    pub(crate) fn gather_rows(
        &self,
        rows: &[usize],
        read_constants: bool,
        dense: &mut QuantizedLayerDenseBatch,
    ) -> crate::Result<()> {
        debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        dense.codes.clear();
        dense.scales.clear();
        dense.constants.clear();
        dense.code_stride = self.code_stride;
        dense.codes.reserve(rows.len() * self.code_stride);
        dense.scales.reserve(rows.len() * QUANTIZED_SCALE_STRIDE);
        if read_constants {
            dense
                .constants
                .reserve(rows.len() * QUANTIZED_CONSTANT_STRIDE);
        }

        const FALLBACK_READ_BLOCK_BYTES: usize = 8 * 1024;
        let mut first = 0;
        while first < rows.len() {
            let code_offset = rows[first] * self.code_stride;
            let code_block = self
                .codes
                .storage_block_ord(code_offset)
                .unwrap_or(code_offset / FALLBACK_READ_BLOCK_BYTES);
            let mut end = first + 1;
            while end < rows.len() {
                let next_offset = rows[end] * self.code_stride;
                let next_block = self
                    .codes
                    .storage_block_ord(next_offset)
                    .unwrap_or(next_offset / FALLBACK_READ_BLOCK_BYTES);
                if next_block != code_block {
                    break;
                }
                end += 1;
            }
            let pinned = self.read_batch(rows[first]..rows[end - 1] + 1, read_constants)?;
            for &row in &rows[first..end] {
                let local = row - pinned.rows.start;
                let code_start = local * self.code_stride;
                let code = &pinned.codes[code_start..code_start + self.code_stride];
                if !quantized_code_tail_is_zero(code, self.dim, self.bits) {
                    return Err(DataCorruption::comment_only(format!(
                        "quantized row {row} has non-zero padding bits for d={} b={}",
                        self.dim, self.bits
                    ))
                    .into());
                }
                dense.codes.extend_from_slice(code);

                let scale_start = local * QUANTIZED_SCALE_STRIDE;
                dense.scales.extend_from_slice(
                    &pinned.scales[scale_start..scale_start + QUANTIZED_SCALE_STRIDE],
                );
                if read_constants {
                    let constant_start = local * QUANTIZED_CONSTANT_STRIDE;
                    dense.constants.extend_from_slice(
                        &pinned.constants
                            [constant_start..constant_start + QUANTIZED_CONSTANT_STRIDE],
                    );
                }
            }
            first = end;
        }
        Ok(())
    }

    pub(crate) fn code_bytes(&self, row: usize) -> crate::Result<OwnedBytes> {
        let bytes = self
            .codes
            .slice(row * self.code_stride..(row + 1) * self.code_stride)
            .read_bytes()
            .map_err(TantivyError::from)?;
        if !quantized_code_tail_is_zero(bytes.as_slice(), self.dim, self.bits) {
            return Err(DataCorruption::comment_only(format!(
                "quantized row {row} has non-zero padding bits for d={} b={}",
                self.dim, self.bits
            ))
            .into());
        }
        Ok(bytes)
    }

    pub(crate) fn scale(&self, row: usize) -> crate::Result<u16> {
        let bytes = self
            .scales
            .slice(row * QUANTIZED_SCALE_STRIDE..(row + 1) * QUANTIZED_SCALE_STRIDE)
            .read_bytes()?;
        Ok(u16::from_le_bytes(bytes.as_slice().try_into().unwrap()))
    }

    pub(crate) fn constant(&self, row: usize) -> crate::Result<f32> {
        let bytes = self
            .constants
            .slice(row * QUANTIZED_CONSTANT_STRIDE..(row + 1) * QUANTIZED_CONSTANT_STRIDE)
            .read_bytes()?;
        Ok(f32::from_le_bytes(bytes.as_slice().try_into().unwrap()))
    }
}

/// Field-keyed V3 quantized payloads resolved from immutable index metadata.
pub(crate) struct QuantizedFieldReader {
    config: VectorQuantizationConfig,
    layers: Vec<QuantizedLayerReader>,
    residual_norms: Option<FileSlice>,
    index_ctx: Arc<QuantizedIndexCtx>,
}

pub(crate) struct QuantizedResidualNormBatch {
    bytes: OwnedBytes,
    rows: Range<usize>,
}

impl QuantizedResidualNormBatch {
    pub(crate) fn get(&self, row: usize) -> crate::Result<f32> {
        if !self.rows.contains(&row) {
            return Err(TantivyError::InternalError(format!(
                "residual-norm row {row} is outside pinned range {:?}",
                self.rows
            )));
        }
        let start = (row - self.rows.start) * QUANTIZED_RESIDUAL_NORM_STRIDE;
        Ok(f32::from_le_bytes(
            self.bytes[start..start + QUANTIZED_RESIDUAL_NORM_STRIDE]
                .try_into()
                .unwrap(),
        ))
    }
}

impl QuantizedFieldReader {
    pub(crate) fn config(&self) -> &VectorQuantizationConfig {
        &self.config
    }

    pub(crate) fn layers(&self) -> &[QuantizedLayerReader] {
        &self.layers
    }

    pub(crate) fn residual_norm(&self, row: usize) -> crate::Result<Option<f32>> {
        let Some(residual_norms) = &self.residual_norms else {
            return Ok(None);
        };
        let bytes = residual_norms
            .slice(row * QUANTIZED_RESIDUAL_NORM_STRIDE..(row + 1) * QUANTIZED_RESIDUAL_NORM_STRIDE)
            .read_bytes()?;
        Ok(Some(f32::from_le_bytes(
            bytes.as_slice().try_into().unwrap(),
        )))
    }

    pub(crate) fn read_residual_norm_batch(
        &self,
        rows: Range<usize>,
    ) -> crate::Result<Option<QuantizedResidualNormBatch>> {
        let Some(residual_norms) = &self.residual_norms else {
            return Ok(None);
        };
        let bytes = residual_norms
            .slice(
                rows.start * QUANTIZED_RESIDUAL_NORM_STRIDE
                    ..rows.end * QUANTIZED_RESIDUAL_NORM_STRIDE,
            )
            .read_bytes()?;
        Ok(Some(QuantizedResidualNormBatch { bytes, rows }))
    }

    pub(crate) fn index_ctx(&self) -> Arc<QuantizedIndexCtx> {
        Arc::clone(&self.index_ctx)
    }
}

fn logical_slice(
    slice: FileSlice,
    logical_len: usize,
    description: &str,
) -> crate::Result<FileSlice> {
    let physical_len = slice.len();
    if physical_len < logical_len || physical_len - logical_len >= 64 {
        return Err(DataCorruption::comment_only(format!(
            "{description} physical length {physical_len} does not contain logical length \
             {logical_len} plus at most 63 alignment bytes"
        ))
        .into());
    }
    if physical_len != logical_len {
        let trailer = slice.slice(logical_len..physical_len).read_bytes()?;
        if trailer.iter().any(|&byte| byte != 0) {
            return Err(DataCorruption::comment_only(format!(
                "{description} has a non-zero alignment trailer"
            ))
            .into());
        }
    }
    Ok(slice.slice_to(logical_len))
}

/// Per-(segment, field) vector reader: the row store plus, for IVF segments,
/// the routing index. See the module docs for the layout and the
/// pinned-vs-deferred split.
pub struct VectorIndexReader {
    options: VectorOptions,
    /// Distinct docs with a vector (the IdMap row count for flat storage; the
    /// persisted doc count for IVF, whose row total legacy V2 replication
    /// could inflate).
    num_vectors: usize,
    /// `false` for the placeholder built by [`Self::empty`] — the segment has
    /// no vector data for this field at all.
    present: bool,
    /// `.vec` slot `[0]`
    id_map: IdMap,
    /// `.vec` slot `[1]`: the dense vector rows. Never materialized whole;
    /// queries fetch per-cluster (or per-doc) ranges.
    rows_slice: FileSlice,
    index: Option<IvfIndex>,
    quantization: Option<QuantizedFieldReader>,
}

impl VectorIndexReader {
    /// Opens `field`'s vector data in `segment_reader`'s segment. Returns the
    /// [`empty`](Self::empty) placeholder when the segment carries no vector
    /// data for the field (no `.vec` file, or the field has no slots in it),
    /// mirroring `SegmentReader::inverted_index`.
    pub(crate) fn open(segment_reader: &SegmentReader, field: Field) -> crate::Result<Self> {
        let entry = segment_reader.schema().get_field_entry(field);
        let options = match entry.field_type() {
            FieldType::Vector(opts) => opts.clone(),
            _ => {
                return Err(TantivyError::InvalidArgument(format!(
                    "field {:?} is not a vector field",
                    entry.name()
                )));
            }
        };

        let vec_file = match segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string()))
        {
            Ok(file) => file,
            Err(OpenReadError::FileDoesNotExist(_)) => return Ok(Self::empty(options)),
            Err(err) => return Err(err.into()),
        };
        let (version, body) = read_header(&vec_file)?;
        let vec_composite = CompositeFile::open(&body)?;
        let (Some(id_map_slice), Some(rows_slice)) = (
            vec_composite.open_read_with_idx(field, vec_slot::ID_MAP),
            vec_composite.open_read_with_idx(field, vec_slot::ROWS),
        ) else {
            return Ok(Self::empty(options));
        };
        let id_map = IdMap::open(id_map_slice, segment_reader.max_doc())?;
        let quantization_config = segment_reader
            .index_settings()
            .vector_quantization
            .iter()
            .find(|config| config.field == entry.name())
            .cloned();

        let centroid_slots = match segment_reader
            .open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string()))
        {
            Ok(file) => {
                let (centroids_version, body) = read_header(&file)?;
                let composite = CompositeFile::open(&body)?;
                match (
                    composite.open_read_with_idx(field, centroid_slot::CENTROIDS),
                    composite.open_read_with_idx(field, centroid_slot::OFFSETS),
                    composite.open_read_with_idx(field, centroid_slot::BOUNDS),
                ) {
                    // Slot [2] (the router) stays optional: the write
                    // side skips it for degenerate centroid counts. Slot
                    // [3] (bounds) is read whenever present; when absent,
                    // a V1 file simply predates it — `IvfIndex::open`
                    // synthesizes SATURATED bounds — while a V2+ file
                    // without it is corrupt, not old.
                    (Some(centroids), Some(offsets), bounds) => {
                        if bounds.is_none() && centroids_version >= VectorFileVersion::V2 {
                            return Err(TantivyError::InternalError(format!(
                                "vector field {:?} has a V2 `.centroids` file with no bounds slot",
                                entry.name()
                            )));
                        }
                        Some((
                            centroids_version,
                            centroids,
                            offsets,
                            composite.open_read_with_idx(field, centroid_slot::ROUTER),
                            bounds,
                        ))
                    }
                    _ => None,
                }
            }
            Err(OpenReadError::FileDoesNotExist(_)) => None,
            Err(err) => return Err(err.into()),
        };

        // The id-map variant and the `.centroids` sidecar are two signals of
        // one write-path decision; a mismatch means a corrupt segment, never a
        // fallback.
        let index = match (&id_map, centroid_slots) {
            (IdMap::Explicit(_), Some((version, centroids, offsets, router, bounds))) => Some(
                IvfIndex::open(version, &options, centroids, offsets, router, bounds)?,
            ),
            (IdMap::Explicit(_), None) => {
                return Err(TantivyError::InternalError(format!(
                    "vector field {:?} has cluster-sorted rows but no `.centroids` data",
                    entry.name()
                )));
            }
            (_, Some(_)) => {
                return Err(TantivyError::InternalError(format!(
                    "vector field {:?} has `.centroids` data but doc-ordered rows",
                    entry.name()
                )));
            }
            (_, None) => None,
        };

        let num_rows = id_map.num_rows() as usize;
        if let Some(index) = &index {
            if index.num_rows() != num_rows {
                return Err(TantivyError::InternalError(
                    "IVF id-map length does not match the cluster offsets".to_string(),
                ));
            }
        }
        let rows_slice = logical_slice(
            rows_slice,
            num_rows * options.bytes_per_vector(),
            &format!("vector field {:?} rows", entry.name()),
        )?;

        let first_quantized_slot = vec_composite
            .open_read_with_idx(field, vec_slot::quantized_codes(0))
            .is_some()
            || vec_composite
                .open_read_with_idx(field, vec_slot::QUANTIZED_RESIDUAL_NORMS)
                .is_some()
            || vec_composite
                .open_read_with_idx(field, vec_slot::QUANTIZED_CALIBRATION)
                .is_some();
        let quantization = match (&index, quantization_config) {
            (Some(_), Some(config)) => {
                if version < VectorFileVersion::V3 {
                    return Err(DataCorruption::comment_only(format!(
                        "vector field {:?} enables quantization but its IVF rows predate V3",
                        entry.name()
                    ))
                    .into());
                }
                let mut layers = Vec::with_capacity(config.layers.len());
                for (layer, spec) in config.layers.iter().enumerate() {
                    let code_stride = quantized_code_stride(options.dim(), spec.bits);
                    let (Some(codes), Some(scales), Some(constants)) = (
                        vec_composite.open_read_with_idx(field, vec_slot::quantized_codes(layer)),
                        vec_composite.open_read_with_idx(field, vec_slot::quantized_scales(layer)),
                        vec_composite
                            .open_read_with_idx(field, vec_slot::quantized_constants(layer)),
                    ) else {
                        return Err(DataCorruption::comment_only(format!(
                            "vector field {:?} has an incomplete configured quantization layer \
                             {layer}",
                            entry.name()
                        ))
                        .into());
                    };
                    layers.push(QuantizedLayerReader {
                        codes: logical_slice(
                            codes,
                            num_rows * code_stride,
                            &format!("vector field {:?} layer {layer} codes", entry.name()),
                        )?,
                        scales: logical_slice(
                            scales,
                            num_rows * QUANTIZED_SCALE_STRIDE,
                            &format!("vector field {:?} layer {layer} scales", entry.name()),
                        )?,
                        constants: logical_slice(
                            constants,
                            num_rows * QUANTIZED_CONSTANT_STRIDE,
                            &format!("vector field {:?} layer {layer} constants", entry.name()),
                        )?,
                        code_stride,
                        dim: options.dim(),
                        bits: spec.bits,
                    });
                }
                for layer in config.layers.len()..4 {
                    if vec_composite
                        .open_read_with_idx(field, vec_slot::quantized_codes(layer))
                        .is_some()
                        || vec_composite
                            .open_read_with_idx(field, vec_slot::quantized_scales(layer))
                            .is_some()
                        || vec_composite
                            .open_read_with_idx(field, vec_slot::quantized_constants(layer))
                            .is_some()
                    {
                        return Err(DataCorruption::comment_only(format!(
                            "vector field {:?} carries quantization layers beyond its configured \
                             prefix",
                            entry.name()
                        ))
                        .into());
                    }
                }
                let residual_norm_slot =
                    vec_composite.open_read_with_idx(field, vec_slot::QUANTIZED_RESIDUAL_NORMS);
                let residual_norms = match (config.needs_residual_norm(), residual_norm_slot) {
                    (true, Some(slice)) => Some(logical_slice(
                        slice,
                        num_rows * QUANTIZED_RESIDUAL_NORM_STRIDE,
                        &format!("vector field {:?} residual squared norms", entry.name()),
                    )?),
                    (true, None) => {
                        return Err(DataCorruption::comment_only(format!(
                            "L2 vector field {:?} is missing residual squared norm slot 14",
                            entry.name()
                        ))
                        .into());
                    }
                    (false, Some(_)) => {
                        return Err(DataCorruption::comment_only(format!(
                            "vector field {:?} carries residual norms for a metric that omits them",
                            entry.name()
                        ))
                        .into());
                    }
                    (false, None) => None,
                };
                let calibration = vec_composite
                    .open_read_with_idx(field, vec_slot::QUANTIZED_CALIBRATION)
                    .ok_or_else(|| {
                        DataCorruption::comment_only(format!(
                            "quantized vector field {:?} is missing calibration metadata slot 15",
                            entry.name()
                        ))
                    })?;
                if calibration.len() < std::mem::size_of::<u32>() {
                    return Err(DataCorruption::comment_only(format!(
                        "vector field {:?} quantization calibration has {} bytes; expected a \
                         version word",
                        entry.name(),
                        calibration.len()
                    ))
                    .into());
                }
                let calibration_version = calibration.slice_to(4).read_bytes()?;
                let calibration_version =
                    u32::from_le_bytes(calibration_version.as_slice().try_into().unwrap());
                let calibration_len = match calibration_version {
                    1 => 12,
                    QUANTIZED_CALIBRATION_VERSION => {
                        quantized_calibration_metadata_len(config.layers.len())
                    }
                    version => {
                        return Err(DataCorruption::comment_only(format!(
                            "quantization calibration metadata version {version} is unsupported; \
                             expected {QUANTIZED_CALIBRATION_VERSION}"
                        ))
                        .into());
                    }
                };
                let calibration = logical_slice(
                    calibration,
                    calibration_len,
                    &format!("vector field {:?} quantization calibration", entry.name()),
                )?
                .read_bytes()?;
                let calibration =
                    VectorQuantizationCalibration::decode(&calibration, config.layers.len())?;
                let cals = calibration.depths.iter().map(|depth| depth.cal).collect();
                let index_ctx = QuantizedIndexCtx::resolve(config.clone(), cals);
                Some(QuantizedFieldReader {
                    config,
                    layers,
                    residual_norms,
                    index_ctx,
                })
            }
            (Some(_), None) if first_quantized_slot => {
                return Err(DataCorruption::comment_only(format!(
                    "vector field {:?} carries quantized slots without index metadata",
                    entry.name()
                ))
                .into());
            }
            (None, _) if first_quantized_slot => {
                return Err(DataCorruption::comment_only(format!(
                    "flat vector field {:?} unexpectedly carries quantized slots",
                    entry.name()
                ))
                .into());
            }
            _ => None,
        };

        let num_vectors = match &index {
            Some(index) => index.num_docs(),
            None => num_rows,
        };
        Ok(Self {
            options,
            num_vectors,
            present: true,
            rows_slice,
            id_map,
            index,
            quantization,
        })
    }

    /// The no-data placeholder: zero vectors, no index. Every accessor
    /// behaves as an empty column, so callers never branch on presence.
    pub(crate) fn empty(options: VectorOptions) -> Self {
        Self {
            options,
            num_vectors: 0,
            present: false,
            rows_slice: FileSlice::empty(),
            id_map: IdMap::Identity { num_docs: 0 },
            index: None,
            quantization: None,
        }
    }

    pub fn options(&self) -> &VectorOptions {
        &self.options
    }

    pub fn dim(&self) -> usize {
        self.options.dim()
    }

    /// Number of distinct docs with a vector value.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// The routing index, present iff the segment's rows are IVF-clustered.
    /// `None` means search must scan the rows exactly.
    pub fn index(&self) -> Option<&IvfIndex> {
        self.index.as_ref()
    }

    pub(crate) fn quantization(&self) -> Option<&QuantizedFieldReader> {
        self.quantization.as_ref()
    }

    /// Storage info for tooling; `None` if the segment has no vector data for
    /// the field.
    pub fn info(&self) -> Option<VectorInfo> {
        if !self.present {
            return None;
        }
        let Some(index) = &self.index else {
            return Some(VectorInfo {
                format: VectorStorageFormat::Flat,
                num_vectors: self.num_vectors,
                num_centroids: None,
                cluster_stats: None,
            });
        };
        let mut empty_clusters = 0;
        let mut min_cluster_size = usize::MAX;
        let mut max_cluster_size = 0;
        let mut total_cluster_size = 0;
        for cluster_size in index.cluster_sizes() {
            empty_clusters += usize::from(cluster_size == 0);
            min_cluster_size = min_cluster_size.min(cluster_size);
            max_cluster_size = max_cluster_size.max(cluster_size);
            total_cluster_size += cluster_size;
        }
        let num_centroids = index.num_clusters();
        let avg_cluster_size = if num_centroids == 0 {
            0.0
        } else {
            total_cluster_size as f64 / num_centroids as f64
        };
        let min_cluster_size = if num_centroids == 0 {
            0
        } else {
            min_cluster_size
        };
        Some(VectorInfo {
            format: VectorStorageFormat::Ivf,
            num_vectors: self.num_vectors,
            num_centroids: Some(num_centroids),
            cluster_stats: Some(VectorClusterStats {
                min_cluster_size,
                max_cluster_size,
                avg_cluster_size,
                empty_clusters,
            }),
        })
    }

    #[cfg(test)]
    /// Exact calibration value decoded from quantization metadata slot 15.
    pub fn quantization_calibrations(&self) -> Option<&[f32]> {
        self.quantization
            .as_ref()
            .map(|quantization| quantization.index_ctx.calibrations())
    }

    #[cfg(test)]
    /// Replays the merge-time held-out calibration protocol against the
    /// already-persisted exact rows, using only the requested scorer prefix.
    /// This is an audit path: it never mutates slots or participates in search.
    pub fn quantization_calibration_audit(
        &self,
        depth: usize,
    ) -> crate::Result<Option<QuantizationCalibrationAudit>> {
        const SAMPLE_ROWS: usize = 1_024;
        const PSEUDO_QUERIES: usize = 64;

        let (Some(index), Some(quantization)) = (&self.index, &self.quantization) else {
            return Ok(None);
        };
        if depth == 0 || depth > quantization.index_ctx.specs.len() {
            return Err(TantivyError::InvalidArgument(format!(
                "quantization calibration audit depth {depth} is outside 1..={}",
                quantization.index_ctx.specs.len()
            )));
        }
        let specs = &quantization.index_ctx.specs[..depth];
        let rho = quantization.index_ctx.grids[depth - 1].rho_model;
        let num_rows = index.num_rows();
        if num_rows == 0 {
            return Ok(Some(QuantizationCalibrationAudit {
                depth,
                sample_count: 0,
                cal: 1.0,
                signed_mean: 0.0,
                signed_stddev: 0.0,
                signed_skewness: 0.0,
                signed_excess_kurtosis: 0.0,
                abs_p50: 0.0,
                abs_p90: 0.0,
                abs_p95: 0.0,
                abs_p99: 0.0,
                abs_max: 0.0,
            }));
        }

        let pseudo_query_count = num_rows.min(PSEUDO_QUERIES);
        let mut pseudo_queries = Vec::with_capacity(pseudo_query_count);
        for sample in 0..pseudo_query_count {
            let row = sample * num_rows / pseudo_query_count;
            let cluster = (0..index.num_clusters())
                .find(|&cluster| index.cluster_range(cluster).contains(&row))
                .expect("persisted row must belong to one IVF cluster");
            pseudo_queries.push((
                cluster,
                decode_row::<f32>(&self.vector_bytes_for_row(row)?, self.options.dim())?,
            ));
        }

        let centroid_stride = self.options.bytes_per_vector();
        let centroid_bytes = index.centroid_bytes()?;
        let interval = num_rows.div_ceil(SAMPLE_ROWS).max(1);
        let mut empirical_variance_sum = 0.0;
        let mut model_variance_sum = 0.0;
        let mut normalized_errors = Vec::with_capacity(SAMPLE_ROWS.min(num_rows));

        for cluster in 0..index.num_clusters() {
            let rows = index.cluster_range(cluster);
            if rows.is_empty() {
                continue;
            }
            let centroid = decode_row::<f32>(
                &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
                self.options.dim(),
            )?;
            let prepared_centroid = prepare_centroid(&centroid, specs);
            let (_, source_query) = (0..pseudo_queries.len())
                .map(|offset| (cluster + offset) % pseudo_queries.len())
                .find_map(|query_idx| {
                    let (query_cluster, query) = &pseudo_queries[query_idx];
                    (*query_cluster != cluster).then_some((*query_cluster, query))
                })
                .expect("held-out calibration requires a query from another cluster");
            let mut score_query = source_query.clone();
            if self.options.metric() == Metric::L2 {
                for (value, &center) in score_query.iter_mut().zip(&centroid) {
                    *value -= center;
                }
            }
            let query_norm = score_query
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let prepared_query = prepare_fp_query(&score_query, specs);
            let final_query = prepared_query.final_layer();

            for row in rows {
                if !row.is_multiple_of(interval) || normalized_errors.len() >= SAMPLE_ROWS {
                    continue;
                }
                let mut values =
                    decode_row::<f32>(&self.vector_bytes_for_row(row)?, self.options.dim())?;
                let encoded = encode_batch_in_place(
                    &mut values,
                    1,
                    &prepared_centroid,
                    specs,
                    &quantization.index_ctx.grids[..depth],
                );
                let scale = encoded.layers[depth - 1].scales[0];
                let dot_error = values
                    .iter()
                    .zip(final_query)
                    .map(|(&error, &query)| f64::from(error) * f64::from(query))
                    .sum::<f64>();
                let model_sigma = f64::from(f16_to_f32(scale)) * rho * f64::from(query_norm);
                empirical_variance_sum += dot_error.powi(2);
                model_variance_sum += model_sigma.powi(2);
                if model_sigma > 0.0 {
                    normalized_errors.push(dot_error / model_sigma);
                }
            }
        }

        let cal = if model_variance_sum == 0.0 {
            1.0
        } else {
            (empirical_variance_sum / model_variance_sum).sqrt()
        };
        let sample_count = normalized_errors.len();
        let signed_mean = normalized_errors.iter().sum::<f64>() / sample_count as f64;
        let centered: Vec<f64> = normalized_errors
            .iter()
            .map(|value| value - signed_mean)
            .collect();
        let variance =
            centered.iter().map(|value| value.powi(2)).sum::<f64>() / sample_count as f64;
        let signed_stddev = variance.sqrt();
        let signed_skewness = if signed_stddev == 0.0 {
            0.0
        } else {
            centered.iter().map(|value| value.powi(3)).sum::<f64>()
                / sample_count as f64
                / signed_stddev.powi(3)
        };
        let signed_excess_kurtosis = if signed_stddev == 0.0 {
            0.0
        } else {
            centered.iter().map(|value| value.powi(4)).sum::<f64>()
                / sample_count as f64
                / signed_stddev.powi(4)
                - 3.0
        };
        let mut absolute_errors: Vec<f64> =
            normalized_errors.iter().map(|value| value.abs()).collect();
        absolute_errors.sort_by(f64::total_cmp);
        let percentile = |fraction: f64| {
            let index = ((absolute_errors.len() - 1) as f64 * fraction).round() as usize;
            absolute_errors[index]
        };

        Ok(Some(QuantizationCalibrationAudit {
            depth,
            sample_count,
            cal,
            signed_mean,
            signed_stddev,
            signed_skewness,
            signed_excess_kurtosis,
            abs_p50: percentile(0.50),
            abs_p90: percentile(0.90),
            abs_p95: percentile(0.95),
            abs_p99: percentile(0.99),
            abs_max: *absolute_errors.last().unwrap(),
        }))
    }

    /// Per-cluster posting-list sizes in cluster order — the distribution
    /// behind [`Self::info`]'s aggregate cluster stats. `None` when the
    /// field's storage is not IVF.
    pub fn cluster_sizes(&self) -> Option<Vec<u32>> {
        self.index
            .as_ref()
            .map(|index| index.cluster_sizes().map(|size| size as u32).collect())
    }

    /// `true` if `doc_id` has a stored vector.
    pub fn contains(&self, doc_id: DocId) -> bool {
        self.row_id(doc_id).is_some()
    }

    /// The raw little-endian bytes of `doc_id`'s vector, fetched with one
    /// stride-sized ranged read; `None` if the doc has no vector.
    pub fn vector_bytes(&self, doc_id: DocId) -> crate::Result<Option<OwnedBytes>> {
        let Some(row) = self.row_id(doc_id) else {
            return Ok(None);
        };
        self.vector_bytes_for_row(row).map(Some)
    }

    /// The raw bytes of the single vector row at `row` of the dense rows
    /// slot, fetched with one stride-sized ranged read
    /// (`row * stride..(row + 1) * stride`). The caller resolves `row`
    /// beforehand (e.g. from a cluster's row range), so no doc→row lookup
    /// happens here.
    pub fn vector_bytes_for_row(&self, row: usize) -> crate::Result<OwnedBytes> {
        if row >= self.id_map.num_rows() as usize {
            return Err(TantivyError::InvalidArgument(format!(
                "vector row {row} is out of bounds"
            )));
        }
        let stride = self.options.bytes_per_vector();
        let bytes = self
            .rows_slice
            .slice(row * stride..(row + 1) * stride)
            .read_bytes()?;
        Ok(bytes)
    }

    /// The doc id stored at `row` of the cluster-sorted permutation, decoded
    /// on demand from the pinned `Explicit` id-map. IVF storage only.
    #[inline]
    pub fn doc_id_at(&self, row: usize) -> DocId {
        let IdMap::Explicit(bytes) = &self.id_map else {
            unreachable!("doc_id_at is only meaningful for cluster-sorted (IVF) storage");
        };
        let start = row * std::mem::size_of::<DocId>();
        DocId::from_le_bytes(
            bytes[start..start + std::mem::size_of::<DocId>()]
                .try_into()
                .unwrap(),
        )
    }

    /// The doc ids assigned to `cluster`, ascending; `None` if the storage is
    /// not IVF or `cluster` is out of bounds.
    pub fn cluster_doc_ids(&self, cluster: usize) -> Option<Vec<DocId>> {
        let index = self.index.as_ref()?;
        if cluster >= index.num_clusters() {
            return None;
        }
        Some(
            index
                .cluster_range(cluster)
                .map(|row| self.doc_id_at(row))
                .collect(),
        )
    }

    /// Doc → dense row. For clustered storage, rows are cluster-sorted and
    /// ascending by doc id within each cluster, so this scans clusters and
    /// binary-searches each one over the pinned id-map bytes. For the flat
    /// id-maps (`Identity`/`Bitmap`) the mapping is strictly ascending in
    /// doc id — the property the exact path's run builder leans on.
    pub(crate) fn row_id(&self, doc_id: DocId) -> Option<usize> {
        match &self.id_map {
            IdMap::Identity { num_docs } => (doc_id < *num_docs).then_some(doc_id as usize),
            IdMap::Bitmap(_) => self.id_map.rank_if_exists(doc_id).map(|row| row as usize),
            IdMap::Explicit(_) => {
                let index = self.index.as_ref()?;
                for cluster in 0..index.num_clusters() {
                    let rows = index.cluster_range(cluster);
                    let mut lo = rows.start;
                    let mut hi = rows.end;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        match self.doc_id_at(mid).cmp(&doc_id) {
                            Ordering::Less => lo = mid + 1,
                            Ordering::Greater => hi = mid,
                            Ordering::Equal => return Some(mid),
                        }
                    }
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::directory::{CompositeWrite, FileHandle};
    use crate::vector::header::write_header;

    #[derive(Debug)]
    struct BlockTrackedBytes {
        bytes: Vec<u8>,
        reads: Arc<Mutex<Vec<Range<usize>>>>,
        block_len: usize,
    }

    impl HasLen for BlockTrackedBytes {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl FileHandle for BlockTrackedBytes {
        fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(self.bytes[range].to_vec()))
        }

        fn storage_block_len(&self) -> Option<usize> {
            Some(self.block_len)
        }
    }

    #[test]
    fn quantized_layer_reader_rejects_non_zero_tail() -> crate::Result<()> {
        let mut codes = vec![0_u8; quantized_code_stride(65, 1)];
        codes[8] = 1;
        let valid = QuantizedLayerReader {
            codes: FileSlice::from(codes.clone()),
            scales: FileSlice::empty(),
            constants: FileSlice::empty(),
            code_stride: codes.len(),
            dim: 65,
            bits: 1,
        };
        assert_eq!(valid.code_bytes(0)?.as_slice(), codes);

        codes[8] |= 2;
        let corrupt = QuantizedLayerReader {
            codes: FileSlice::from(codes.clone()),
            scales: FileSlice::empty(),
            constants: FileSlice::empty(),
            code_stride: codes.len(),
            dim: 65,
            bits: 1,
        };
        assert!(corrupt.code_bytes(0).is_err());
        Ok(())
    }

    #[test]
    fn quantized_layer_reader_pins_and_decodes_a_contiguous_batch() -> crate::Result<()> {
        let stride = quantized_code_stride(65, 1);
        let mut codes = vec![0_u8; stride * 2];
        codes[8] = 1;
        codes[stride + 8] = 1;
        let scales = [17_u16, 29_u16]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let constants = [0.25_f32, -0.75_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let reader = QuantizedLayerReader {
            codes: FileSlice::from(codes.clone()),
            scales: FileSlice::from(scales),
            constants: FileSlice::from(constants),
            code_stride: stride,
            dim: 65,
            bits: 1,
        };
        let batch = reader.read_batch(0..2, true)?;
        assert_eq!(batch.code_bytes(0)?, &codes[..stride]);
        assert_eq!(batch.code_bytes(1)?, &codes[stride..]);
        assert_eq!(batch.scale(0)?, 17);
        assert_eq!(batch.scale(1)?, 29);
        assert_eq!(batch.constant(0)?.to_bits(), 0.25_f32.to_bits());
        assert_eq!(batch.constant(1)?.to_bits(), (-0.75_f32).to_bits());
        let dot_batch = reader.read_batch(0..2, false)?;
        assert!(dot_batch.constants().is_empty());

        codes[stride + 8] |= 2;
        let corrupt = QuantizedLayerReader {
            codes: FileSlice::from(codes),
            scales: FileSlice::from(vec![0_u8; 4]),
            constants: FileSlice::from(vec![0_u8; 8]),
            code_stride: stride,
            dim: 65,
            bits: 1,
        };
        assert!(corrupt.read_batch(0..2, true)?.code_bytes(1).is_err());
        Ok(())
    }

    #[test]
    fn sparse_gather_groups_by_absolute_storage_block() -> crate::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let codes = FileSlice::new(Arc::new(BlockTrackedBytes {
            bytes: vec![0_u8; 64],
            reads: Arc::clone(&reads),
            block_len: 16,
        }))
        .slice(15..31);
        let reader = QuantizedLayerReader {
            codes,
            scales: FileSlice::from(vec![0_u8; 4]),
            constants: FileSlice::empty(),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };
        let mut dense = QuantizedLayerDenseBatch::new();
        reader.gather_rows(&[0, 1], false, &mut dense)?;
        assert_eq!(&*reads.lock().unwrap(), &[15..23, 23..31]);
        assert_eq!(dense.codes(), &[0_u8; 16]);
        Ok(())
    }

    #[test]
    fn v3_maximal_trailers_are_trimmed_from_logical_arrays() -> crate::Result<()> {
        let field = Field::from_field_id(0);
        let mut bytes = Vec::new();
        write_header(&mut bytes)?;
        {
            let mut writer = CompositeWrite::wrap(&mut bytes);
            let rows = writer.for_field_with_idx(field, vec_slot::ROWS);
            rows.write_all(&[1, 2, 3, 4])?;
            rows.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(1), 0)
                .write_all(&[9])?;

            let scales = writer.for_field_with_idx(field, vec_slot::quantized_scales(0));
            scales.write_all(&17_u16.to_le_bytes())?;
            scales.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(2), 0)
                .write_all(&[9])?;

            let constants = writer.for_field_with_idx(field, vec_slot::quantized_constants(0));
            constants.write_all(&1.25_f32.to_le_bytes())?;
            constants.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(3), 0)
                .write_all(&[9])?;

            let norms = writer.for_field_with_idx(field, vec_slot::QUANTIZED_RESIDUAL_NORMS);
            norms.write_all(&2.5_f32.to_le_bytes())?;
            norms.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(4), 0)
                .write_all(&[9])?;
            writer.close()?;
        }

        let (version, body) = read_header(&FileSlice::from(bytes))?;
        assert_eq!(version, VectorFileVersion::V3);
        let composite = CompositeFile::open(&body)?;

        let rows = logical_slice(
            composite.open_read_with_idx(field, vec_slot::ROWS).unwrap(),
            4,
            "rows fixture",
        )?
        .read_bytes()?;
        let scales = logical_slice(
            composite
                .open_read_with_idx(field, vec_slot::quantized_scales(0))
                .unwrap(),
            2,
            "scales fixture",
        )?
        .read_bytes()?;
        let constants = logical_slice(
            composite
                .open_read_with_idx(field, vec_slot::quantized_constants(0))
                .unwrap(),
            4,
            "constants fixture",
        )?
        .read_bytes()?;
        let norms = logical_slice(
            composite
                .open_read_with_idx(field, vec_slot::QUANTIZED_RESIDUAL_NORMS)
                .unwrap(),
            4,
            "residual norms fixture",
        )?
        .read_bytes()?;

        assert_eq!(rows.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(
            u16::from_le_bytes(scales.as_slice().try_into().unwrap()),
            17
        );
        assert_eq!(
            f32::from_le_bytes(constants.as_slice().try_into().unwrap()),
            1.25
        );
        assert_eq!(
            f32::from_le_bytes(norms.as_slice().try_into().unwrap()),
            2.5
        );
        Ok(())
    }

    #[test]
    fn nonzero_alignment_trailer_is_corruption() {
        let slice = FileSlice::from(vec![1, 2, 3, 4, 0, 7]);
        assert!(logical_slice(slice, 4, "bad fixture").is_err());
    }
}
