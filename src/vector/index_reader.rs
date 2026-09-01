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
use std::sync::{Arc, OnceLock};

use common::{HasLen, OwnedBytes};
use quant_model::f16::f16_to_f32;

use super::flat::IdMap;
use super::header::{centroid_slot, read_header, vec_slot, VectorFileVersion};
use super::ivf::{decode_row, IvfIndex, CENTROIDS_EXT};
use super::prepared::{QuantizedIndexCtx, QuantizedQueryCtx};
use super::quantization::{
    quantized_code_stride, quantized_code_tail_is_zero, VectorQuantizationCalibrationSource,
    VectorQuantizationConfig, VectorQuantizationDepthCalibration, QUANTIZED_CONSTANT_STRIDE,
    QUANTIZED_RESIDUAL_NORM_STRIDE, QUANTIZED_SCALE_STRIDE,
};
use super::VEC_EXT;
use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::error::DataCorruption;
use crate::fastfield::AliveBitSet;
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType, Metric, VectorOptions};
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VectorCalibrationMoments {
    pub sample_count: u64,
    pub normalized_error_sum: f64,
    pub normalized_error_squared_sum: f64,
}

impl VectorCalibrationMoments {
    fn observe(&mut self, value: f64) {
        self.sample_count += 1;
        self.normalized_error_sum += value;
        self.normalized_error_squared_sum += value * value;
    }

    pub fn bias(&self) -> Option<f64> {
        (self.sample_count != 0).then(|| self.normalized_error_sum / self.sample_count as f64)
    }

    pub fn spread(&self) -> Option<f64> {
        self.bias().map(|bias| {
            (self.normalized_error_squared_sum / self.sample_count as f64 - bias * bias)
                .max(0.0)
                .sqrt()
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorCalibrationMeasurements {
    aggregate: Vec<VectorCalibrationMoments>,
    per_query: Vec<Vec<VectorCalibrationMoments>>,
}

#[inline]
fn calibration_query_norm(query: &QuantizedQueryCtx, metric: Metric, centroid_bytes: &[u8]) -> f32 {
    let routing_score = metric
        .similarity_bytes::<f32>(query.query(), centroid_bytes)
        .score();
    query.score_query_norm(routing_score)
}

#[inline]
fn observe_calibration_prefix(
    measurements: &mut VectorCalibrationMeasurements,
    query_idx: usize,
    depth: usize,
    exact_dot: f32,
    prefix_estimate: &mut f32,
    layer_estimate: f32,
    model_sigma: f32,
) {
    // Production accumulates residual-plane estimates in f32. Keep the
    // calibration prefix bit-for-bit in that arithmetic domain and widen
    // only the final normalized error for the f64 moment accumulators.
    *prefix_estimate += layer_estimate;
    if model_sigma > 0.0 && model_sigma.is_finite() {
        let error = (exact_dot - *prefix_estimate) / model_sigma;
        if error.is_finite() {
            let error = f64::from(error);
            measurements.aggregate[depth].observe(error);
            measurements.per_query[query_idx][depth].observe(error);
        }
    }
}

impl VectorCalibrationMeasurements {
    pub fn aggregate(&self) -> &[VectorCalibrationMoments] {
        &self.aggregate
    }

    pub fn per_query(&self) -> &[Vec<VectorCalibrationMoments>] {
        &self.per_query
    }

    pub fn merge(&mut self, other: &Self) -> crate::Result<()> {
        if self.aggregate.is_empty() {
            *self = other.clone();
            return Ok(());
        }
        if self.aggregate.len() != other.aggregate.len()
            || self.per_query.len() != other.per_query.len()
        {
            return Err(TantivyError::InvalidArgument(
                "cannot merge quantization calibration measurements with different shapes"
                    .to_string(),
            ));
        }
        for (left, right) in self.aggregate.iter_mut().zip(&other.aggregate) {
            left.sample_count += right.sample_count;
            left.normalized_error_sum += right.normalized_error_sum;
            left.normalized_error_squared_sum += right.normalized_error_squared_sum;
        }
        for (left_query, right_query) in self.per_query.iter_mut().zip(&other.per_query) {
            for (left, right) in left_query.iter_mut().zip(right_query) {
                left.sample_count += right.sample_count;
                left.normalized_error_sum += right.normalized_error_sum;
                left.normalized_error_squared_sum += right.normalized_error_squared_sum;
            }
        }
        Ok(())
    }

    pub fn finish(
        &self,
        source: VectorQuantizationCalibrationSource,
    ) -> crate::Result<Vec<VectorQuantizationDepthCalibration>> {
        self.aggregate
            .iter()
            .enumerate()
            .map(|(depth, moments)| {
                let Some(bias) = moments.bias() else {
                    return Err(TantivyError::InvalidArgument(format!(
                        "quantization calibration depth {depth} has no measurements"
                    )));
                };
                let spread = moments.spread().unwrap();
                if !bias.is_finite()
                    || !spread.is_finite()
                    || bias.abs() > f32::MAX as f64
                    || spread > f32::MAX as f64
                {
                    return Err(TantivyError::InvalidArgument(format!(
                        "quantization calibration depth {depth} produced non-finite f32 statistics"
                    )));
                }
                Ok(VectorQuantizationDepthCalibration {
                    bias: bias as f32,
                    spread: spread as f32,
                    sample_count: u32::try_from(moments.sample_count).map_err(|_| {
                        TantivyError::InvalidArgument(format!(
                            "quantization calibration depth {depth} sample count exceeds u32"
                        ))
                    })?,
                    source,
                })
            })
            .collect()
    }
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
        Ok(&self.codes[start..start + self.code_stride])
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

fn storage_block_span(slot: &FileSlice, byte_range: Range<usize>) -> Option<(usize, usize)> {
    debug_assert!(byte_range.start < byte_range.end);
    let first = slot.storage_block_ord(byte_range.start)?;
    let last = slot.storage_block_ord(byte_range.end - 1)?;
    Some((first, last))
}

fn append_storage_block_span(
    slot: &FileSlice,
    byte_range: Range<usize>,
    spans: &mut Vec<(usize, usize)>,
) -> bool {
    let Some(span) = storage_block_span(slot, byte_range) else {
        return false;
    };
    spans.push(span);
    true
}

fn merged_storage_block_count(spans: &mut [(usize, usize)]) -> usize {
    spans.sort_unstable_by_key(|&(start, end)| (start, end));
    let mut count = 0usize;
    let mut current: Option<(usize, usize)> = None;
    for &(start, end) in spans.iter() {
        current = match current {
            None => Some((start, end)),
            Some((current_start, current_end)) if start <= current_end.saturating_add(1) => {
                Some((current_start, current_end.max(end)))
            }
            Some((current_start, current_end)) => {
                count += current_end - current_start + 1;
                Some((start, end))
            }
        };
    }
    if let Some((start, end)) = current {
        count += end - start + 1;
    }
    count
}

impl QuantizedLayerReader {
    pub(crate) fn code_stride(&self) -> usize {
        self.code_stride
    }

    pub(crate) fn read_codes(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        let codes = self
            .codes
            .slice(rows.start * self.code_stride..rows.end * self.code_stride)
            .read_bytes()?;
        // The zero-tail invariant is checked once per pinned range, never in
        // an indexed survivor loop.
        for (local, code) in codes.chunks_exact(self.code_stride).enumerate() {
            if !quantized_code_tail_is_zero(code, self.dim, self.bits) {
                let row = rows.start + local;
                return Err(DataCorruption::comment_only(format!(
                    "quantized row {row} has non-zero padding bits for d={} b={}",
                    self.dim, self.bits
                ))
                .into());
            }
        }
        Ok(codes)
    }

    pub(crate) fn read_scales(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        Ok(self
            .scales
            .slice(rows.start * QUANTIZED_SCALE_STRIDE..rows.end * QUANTIZED_SCALE_STRIDE)
            .read_bytes()?)
    }

    pub(crate) fn read_constants(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        Ok(self
            .constants
            .slice(rows.start * QUANTIZED_CONSTANT_STRIDE..rows.end * QUANTIZED_CONSTANT_STRIDE)
            .read_bytes()?)
    }

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
        let codes = self.read_codes(rows.clone())?;
        let scales = self.read_scales(rows.clone())?;
        let constants = if read_constants {
            self.read_constants(rows.clone())?
        } else {
            OwnedBytes::empty()
        };
        Ok(QuantizedLayerBatch {
            codes,
            scales,
            constants,
            rows,
            code_stride: self.code_stride,
        })
    }

    fn plan_slot_reads(
        slot: &FileSlice,
        stride: usize,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        debug_assert!(!rows.is_empty());
        debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(rows.iter().all(|row| available_rows.contains(row)));
        read_ranges.clear();

        block_scratch.clear();
        for &row in rows {
            if !append_storage_block_span(slot, row * stride..(row + 1) * stride, block_scratch) {
                // In-memory and ordinary-file handles expose no page
                // geometry. Pin the available borrowed slice once; never
                // fabricate a block size.
                read_ranges.push(available_rows);
                return;
            }
        }
        let touched_blocks = merged_storage_block_count(block_scratch);

        block_scratch.clear();
        debug_assert!(append_storage_block_span(
            slot,
            available_rows.start * stride..available_rows.end * stride,
            block_scratch,
        ));
        let covered_blocks = merged_storage_block_count(block_scratch);
        debug_assert!(touched_blocks <= covered_blocks);
        if touched_blocks == covered_blocks {
            read_ranges.push(available_rows);
            return;
        }

        // Sparse path: merge only overlapping physical-block intervals.
        // Adjacent blocks stay as separate reads so page-backed handles never
        // materialize a multi-page OwnedBytes merely for call coalescing.
        let mut first_row = rows[0];
        let mut previous_row = rows[0];
        let (_, mut group_end_block) =
            storage_block_span(slot, rows[0] * stride..(rows[0] + 1) * stride)
                .expect("storage geometry was resolved above");
        for &row in &rows[1..] {
            let (start_block, end_block) =
                storage_block_span(slot, row * stride..(row + 1) * stride)
                    .expect("storage geometry was resolved above");
            if start_block <= group_end_block {
                group_end_block = group_end_block.max(end_block);
            } else {
                read_ranges.push(first_row..previous_row + 1);
                first_row = row;
                group_end_block = end_block;
            }
            previous_row = row;
        }
        read_ranges.push(first_row..previous_row + 1);

        debug_assert!(read_ranges.windows(2).all(|pair| {
            let (_, left_end) =
                storage_block_span(slot, pair[0].start * stride..pair[0].end * stride).unwrap();
            let (right_start, _) =
                storage_block_span(slot, pair[1].start * stride..pair[1].end * stride).unwrap();
            left_end < right_start
        }));
    }

    pub(crate) fn plan_code_reads(
        &self,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        Self::plan_slot_reads(
            &self.codes,
            self.code_stride,
            available_rows,
            rows,
            read_ranges,
            block_scratch,
        );
    }

    pub(crate) fn plan_scale_reads(
        &self,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        Self::plan_slot_reads(
            &self.scales,
            QUANTIZED_SCALE_STRIDE,
            available_rows,
            rows,
            read_ranges,
            block_scratch,
        );
    }

    pub(crate) fn plan_constant_reads(
        &self,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        Self::plan_slot_reads(
            &self.constants,
            QUANTIZED_CONSTANT_STRIDE,
            available_rows,
            rows,
            read_ranges,
            block_scratch,
        );
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
    // Positive-depth scans resolve this once for the segment/field. Level 0
    // never touches it, so unquantized IVF does not construct scorer state.
    index_ctx: OnceLock<Option<Arc<QuantizedIndexCtx>>>,
    layers: Vec<QuantizedLayerReader>,
    residual_norms: Option<FileSlice>,
}

pub(crate) struct QuantizedResidualNormBatch {
    bytes: OwnedBytes,
    rows: Range<usize>,
}

impl QuantizedResidualNormBatch {
    /// The complete fixed-stride LE-f32 range for one cluster pin. Callers
    /// decode this once as a contiguous pass rather than doing checked
    /// per-row lookups in the scoring loop.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        debug_assert_eq!(
            self.bytes.len(),
            self.rows.len() * QUANTIZED_RESIDUAL_NORM_STRIDE
        );
        &self.bytes
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

    pub(crate) fn index_ctx(&self) -> Option<Arc<QuantizedIndexCtx>> {
        self.index_ctx
            .get_or_init(|| QuantizedIndexCtx::resolve_from_config(self.config.clone()))
            .as_ref()
            .map(Arc::clone)
    }

    #[cfg(test)]
    pub(crate) fn index_ctx_is_initialized(&self) -> bool {
        self.index_ctx.get().is_some()
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
                // Slot 15 was retired before V3 shipped. Calibration lives in
                // index settings; any physical slot 15 bytes are ignored.
                Some(QuantizedFieldReader {
                    config,
                    index_ctx: OnceLock::new(),
                    layers,
                    residual_norms,
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
        self.quantization
            .as_ref()
            .filter(|quantization| quantization.config.calibration().is_some())
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

    /// Measure per-prefix normalized estimator errors for caller-supplied
    /// queries using the same quantized query preparation and split-form
    /// kernels as production scanning. Raw moments make results mergeable
    /// across segment readers before settings are updated. Callers must pass
    /// the owning [`SegmentReader::alive_bitset`] so deleted documents never
    /// enter the target-row sample.
    pub fn calibrate_external_queries(
        &self,
        queries: &[Vec<f32>],
        sample_rows: usize,
        alive: Option<&AliveBitSet>,
    ) -> crate::Result<Option<VectorCalibrationMeasurements>> {
        let (Some(index), Some(quantization)) = (&self.index, &self.quantization) else {
            return Ok(None);
        };
        if sample_rows == 0 {
            return Err(TantivyError::InvalidArgument(
                "quantization calibration sample_rows must be greater than zero".to_string(),
            ));
        }
        for query in queries {
            if query.len() != self.options.dim() {
                return Err(TantivyError::InvalidArgument(format!(
                    "quantization calibration query has dimension {}; expected {}",
                    query.len(),
                    self.options.dim()
                )));
            }
        }

        let layer_count = quantization.config.layers.len();
        let mut measurements = VectorCalibrationMeasurements {
            aggregate: vec![VectorCalibrationMoments::default(); layer_count],
            per_query: vec![vec![VectorCalibrationMoments::default(); layer_count]; queries.len()],
        };
        if index.num_rows() == 0 || queries.is_empty() {
            return Ok(Some(measurements));
        }

        let measurement_ctx =
            QuantizedIndexCtx::for_calibration_measurement(quantization.config.clone());
        let prepared_queries: Vec<QuantizedQueryCtx> = queries
            .iter()
            .cloned()
            .map(|query| QuantizedQueryCtx::new(Arc::clone(&measurement_ctx), query))
            .collect();
        let live_row_count = self.live_posting_row_count(alive);
        let target_rows = sample_rows.min(live_row_count);
        if target_rows == 0 {
            return Ok(Some(measurements));
        }
        let centroid_stride = self.options.bytes_per_vector();
        let centroid_bytes = index.centroid_bytes()?;
        let mut sampled = 0usize;
        let mut live_rows_seen = 0usize;
        let mut next_sample_ordinal = 0usize;

        for cluster in 0..index.num_clusters() {
            let rows = index.cluster_range(cluster);
            let centroid_row = &centroid_bytes[cluster * centroid_stride..][..centroid_stride];
            let centroid = decode_row::<f32>(centroid_row, self.options.dim())?;
            for row in rows {
                if alive.is_some_and(|alive| !alive.is_alive(self.doc_id_at(row))) {
                    continue;
                }
                let live_ordinal = live_rows_seen;
                live_rows_seen += 1;
                if sampled >= target_rows || live_ordinal != next_sample_ordinal {
                    continue;
                }
                sampled += 1;
                next_sample_ordinal = sampled * live_row_count / target_rows;
                let values =
                    decode_row::<f32>(&self.vector_bytes_for_row(row)?, self.options.dim())?;
                for (query_idx, query) in prepared_queries.iter().enumerate() {
                    let mut exact_dot = 0.0f32;
                    for ((&value, &center), &query_value) in
                        values.iter().zip(&centroid).zip(query.query())
                    {
                        let residual = value - center;
                        let score_query = if self.options.metric() == Metric::L2 {
                            query_value - center
                        } else {
                            query_value
                        };
                        exact_dot += residual * score_query;
                    }
                    let query_norm =
                        calibration_query_norm(query, self.options.metric(), centroid_row);
                    let mut prefix_estimate = 0.0f32;
                    for (depth, layer) in quantization.layers.iter().enumerate() {
                        let scale = layer.scale(row)?;
                        let constant = if self.options.metric() == Metric::L2 {
                            layer.constant(row)?
                        } else {
                            0.0
                        };
                        let layer_estimate =
                            query.score_layer(depth, &layer.code_bytes(row)?, scale, constant);
                        let model_sigma = f16_to_f32(scale)
                            * measurement_ctx.grids[depth].rho_model as f32
                            * query_norm;
                        observe_calibration_prefix(
                            &mut measurements,
                            query_idx,
                            depth,
                            exact_dot,
                            &mut prefix_estimate,
                            layer_estimate,
                            model_sigma,
                        );
                    }
                }
            }
        }
        Ok(Some(measurements))
    }

    /// Per-cluster posting-list sizes in cluster order — the distribution
    /// behind [`Self::info`]'s aggregate cluster stats. `None` when the
    /// field's storage is not IVF.
    pub fn cluster_sizes(&self) -> Option<Vec<u32>> {
        self.index
            .as_ref()
            .map(|index| index.cluster_sizes().map(|size| size as u32).collect())
    }

    /// Number of live IVF posting memberships. Replicated documents count
    /// once per membership; deleted documents contribute no memberships.
    pub fn live_posting_row_count(&self, alive: Option<&AliveBitSet>) -> usize {
        let Some(index) = &self.index else {
            return 0;
        };
        (0..index.num_rows())
            .filter(|&row| alive.is_none_or(|alive| alive.is_alive(self.doc_id_at(row))))
            .count()
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
    fn calibration_query_norm_matches_production_f32_chain() {
        const DIM: usize = 64;
        let source_query: Vec<f32> = (0..DIM)
            .map(|coordinate| 10_000.0 + coordinate as f32 * 0.375)
            .collect();
        let centroid: Vec<f32> = (0..DIM)
            .map(|coordinate| 9_997.0 - coordinate as f32 * 0.1875)
            .collect();
        let centroid_bytes = centroid
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut differs_from_the_retired_f64_chain = false;

        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let config = VectorQuantizationConfig::materialize(
                "embedding".to_string(),
                &VectorOptions::new(DIM, metric),
                vec![super::super::quantization::VectorQuantizationLayer {
                    bits: 1,
                    quantizer: super::super::quantization::VectorQuantizer::RaBitQ,
                    seed: 7,
                }],
            )
            .unwrap();
            let query = QuantizedQueryCtx::new(
                QuantizedIndexCtx::for_calibration_measurement(config),
                source_query.clone(),
            );
            let routing_score = metric
                .similarity_bytes::<f32>(query.query(), &centroid_bytes)
                .score();
            let expected = query.score_query_norm(routing_score);
            let actual = calibration_query_norm(&query, metric, &centroid_bytes);
            assert_eq!(actual.to_bits(), expected.to_bits(), "metric={metric:?}");

            let retired_norm = query
                .query()
                .iter()
                .zip(&centroid)
                .map(|(&query, &centroid)| {
                    let value = if metric == Metric::L2 {
                        query - centroid
                    } else {
                        query
                    };
                    f64::from(value) * f64::from(value)
                })
                .sum::<f64>()
                .sqrt();
            differs_from_the_retired_f64_chain |=
                f64::from(actual).to_bits() != retired_norm.to_bits();
        }
        assert!(differs_from_the_retired_f64_chain);
    }

    #[test]
    fn depth_two_calibration_oracle_tracks_signed_prefix_errors() {
        // Two deterministic samples with unit model sigma:
        // depth 1 errors = [10 - 9, 10 - 7] = [1, 3]
        // depth 2 errors = [10 - (9 + 5), 10 - (7 + 5)] = [-4, -2].
        // The aggregate oracle is therefore (+2, 1) then (-3, 1) for
        // (bias, spread). This catches both sign loss and treating layer 2
        // as a standalone estimate instead of an f32 prefix refinement.
        let mut measurements = VectorCalibrationMeasurements {
            aggregate: vec![VectorCalibrationMoments::default(); 2],
            per_query: vec![vec![VectorCalibrationMoments::default(); 2]; 2],
        };
        for (query_idx, layer_estimates) in [[9.0_f32, 5.0], [7.0, 5.0]].into_iter().enumerate() {
            let mut prefix_estimate = 0.0_f32;
            for (depth, layer_estimate) in layer_estimates.into_iter().enumerate() {
                observe_calibration_prefix(
                    &mut measurements,
                    query_idx,
                    depth,
                    10.0,
                    &mut prefix_estimate,
                    layer_estimate,
                    1.0,
                );
            }
        }

        let calibration = measurements
            .finish(VectorQuantizationCalibrationSource::RealQuery)
            .unwrap();
        assert_eq!(calibration[0].bias.to_bits(), 2.0_f32.to_bits());
        assert_eq!(calibration[0].spread.to_bits(), 1.0_f32.to_bits());
        assert_eq!(calibration[1].bias.to_bits(), (-3.0_f32).to_bits());
        assert_eq!(calibration[1].spread.to_bits(), 1.0_f32.to_bits());
        assert!(calibration[0].bias.is_sign_positive());
        assert!(calibration[1].bias.is_sign_negative());
        assert_eq!(measurements.per_query()[0][0].bias(), Some(1.0));
        assert_eq!(measurements.per_query()[0][1].bias(), Some(-4.0));
        assert_eq!(measurements.per_query()[1][0].bias(), Some(3.0));
        assert_eq!(measurements.per_query()[1][1].bias(), Some(-2.0));
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
    fn indexed_range_validates_unselected_gap_row_tail() {
        let stride = quantized_code_stride(65, 1);
        let mut codes = vec![0_u8; stride * 3];
        codes[8] = 1;
        codes[stride + 8] = 0b10;
        codes[stride * 2 + 8] = 1;
        let reader = QuantizedLayerReader {
            codes: FileSlice::from(codes),
            scales: FileSlice::from(vec![0_u8; 3 * QUANTIZED_SCALE_STRIDE]),
            constants: FileSlice::empty(),
            code_stride: stride,
            dim: 65,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();
        reader.plan_code_reads(0..3, &[0, 2], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..3]);
        assert!(
            reader.read_codes(ranges.pop().unwrap()).is_err(),
            "the unselected corrupt row inside the pinned range must be checked"
        );
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
        assert!(corrupt.read_batch(0..2, true).is_err());
        Ok(())
    }

    #[test]
    fn indexed_read_plan_is_density_adaptive_across_soa_slots() -> crate::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let storage = Arc::new(BlockTrackedBytes {
            bytes: vec![0_u8; 160],
            reads: Arc::clone(&reads),
            block_len: 16,
        });
        let codes = FileSlice::new(storage.clone()).slice(15..79);
        let scales = FileSlice::new(storage.clone()).slice(80..96);
        let constants = FileSlice::new(storage).slice(96..128);
        let reader = QuantizedLayerReader {
            codes,
            scales,
            constants,
            code_stride: 8,
            dim: 64,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();

        reader.plan_code_reads(0..8, &[1, 3], &mut ranges, &mut blocks);
        assert_eq!(ranges, [1..2, 3..4], "adjacent pages stay separate");
        reader.plan_code_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..1, 7..8]);
        for range in ranges.drain(..) {
            reader.read_codes(range)?;
        }
        reader.plan_scale_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        reader.read_scales(ranges.pop().unwrap())?;
        reader.plan_constant_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        reader.read_constants(ranges.pop().unwrap())?;
        let reads = reads.lock().unwrap();
        assert_eq!(&*reads, &[15..23, 71..79, 80..96, 96..128]);
        let mut pinned_blocks = reads
            .iter()
            .flat_map(|range| range.start / 16..=(range.end - 1) / 16)
            .collect::<Vec<_>>();
        let pin_count = pinned_blocks.len();
        pinned_blocks.sort_unstable();
        pinned_blocks.dedup();
        assert_eq!(
            pinned_blocks.len(),
            pin_count,
            "each physical block must be pinned once per indexed batch"
        );
        drop(reads);

        reader.plan_code_reads(0..8, &(0..8).collect::<Vec<_>>(), &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        Ok(())
    }

    #[test]
    fn indexed_read_plan_without_storage_geometry_pins_available_range() {
        let reader = QuantizedLayerReader {
            codes: FileSlice::from(vec![0_u8; 8 * 8]),
            scales: FileSlice::from(vec![0_u8; 8 * 2]),
            constants: FileSlice::empty(),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();
        reader.plan_code_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        reader.plan_scale_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
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
