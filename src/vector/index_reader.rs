//! Per-segment vector row storage, IVF routing, and quantized-layer access.

use std::cmp::Ordering;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use common::{HasLen, OwnedBytes};
use quant_model::f16::f16_to_f32;

use super::flat::IdMap;
use super::header::{
    read_centroid_header, read_vector_header, CentroidSlot, VectorFileVersion, VectorSlot,
};
use super::ivf::{IvfIndex, CENTROIDS_EXT};
use super::prepared::{PreparedQuery, QuantizedIndexCtx};
use super::quantization::{
    quantized_code_stride, quantized_code_tail_is_zero, VectorQuantizationConfig,
    MAX_QUANTIZATION_LAYERS, QUANTIZED_CONSTANT_STRIDE, QUANTIZED_ERROR_RATIO_STRIDE,
    QUANTIZED_GAMMA_STRIDE, QUANTIZED_RESIDUAL_NORM_STRIDE, QUANTIZED_SCALE_STRIDE,
    QUANTIZED_SIDECAR_STRIDE,
};
use super::VEC_EXT;
use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::error::DataCorruption;
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType, Metric, VectorOptions};
use crate::{DocId, SegmentReader, TantivyError};

/// Vector row-storage layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorStorageFormat {
    /// Full-precision rows without cluster routing.
    Flat,
    /// Clustered inverted-file rows.
    Ivf,
}

/// Segment vector-storage metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorInfo {
    /// Row-storage layout.
    pub format: VectorStorageFormat,
    /// Distinct documents with vectors.
    pub num_vectors: usize,
    /// Number of IVF centroids.
    pub num_centroids: Option<usize>,
    /// IVF posting-size statistics.
    pub cluster_stats: Option<VectorClusterStats>,
}

/// IVF posting-size statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorClusterStats {
    /// Minimum posting size.
    pub min_cluster_size: usize,
    /// Maximum posting size.
    pub max_cluster_size: usize,
    /// Mean posting size.
    pub avg_cluster_size: f64,
    /// Empty posting count.
    pub empty_clusters: usize,
}

/// Deferred code, sidecar, and optional L2-constant slices for one residual layer.
pub(crate) struct QuantizedLayerReader {
    codes: FileSlice,
    sidecar: FileSlice,
    constants: Option<FileSlice>,
    cluster_offsets: Arc<[usize]>,
    code_stride: usize,
    dim: usize,
    bits: u8,
}

/// Pinned SoA ranges for one contiguous cluster posting.
pub(crate) struct QuantizedLayerBatch {
    codes: OwnedBytes,
    scales: OwnedBytes,
    gammas: OwnedBytes,
    error_ratios: OwnedBytes,
    constants: Option<OwnedBytes>,
    rows: Range<usize>,
    code_stride: usize,
}

/// Borrowed sidecar runs for one row range inside an IVF cluster.
pub(crate) struct QuantizedSidecarBatch {
    scales: OwnedBytes,
    gammas: OwnedBytes,
    error_ratios: OwnedBytes,
    rows: Range<usize>,
}

impl QuantizedSidecarBatch {
    fn local_row(&self, row: usize) -> crate::Result<usize> {
        if !self.rows.contains(&row) {
            return Err(TantivyError::InternalError(format!(
                "quantized row {row} is outside pinned sidecar range {:?}",
                self.rows
            )));
        }
        Ok(row - self.rows.start)
    }

    pub(crate) fn scale(&self, row: usize) -> crate::Result<f32> {
        let local = self.local_row(row)?;
        let start = local * QUANTIZED_SCALE_STRIDE;
        Ok(f32::from_le_bytes(
            self.scales[start..start + QUANTIZED_SCALE_STRIDE]
                .try_into()
                .unwrap(),
        ))
    }

    pub(crate) fn gamma(&self, row: usize) -> crate::Result<f32> {
        let local = self.local_row(row)?;
        let start = local * QUANTIZED_GAMMA_STRIDE;
        let bits = u16::from_le_bytes(
            self.gammas[start..start + QUANTIZED_GAMMA_STRIDE]
                .try_into()
                .unwrap(),
        );
        Ok(f16_to_f32(bits))
    }

    pub(crate) fn error_ratio(&self, row: usize) -> crate::Result<f32> {
        let local = self.local_row(row)?;
        let start = local * QUANTIZED_ERROR_RATIO_STRIDE;
        let bits = u16::from_le_bytes(
            self.error_ratios[start..start + QUANTIZED_ERROR_RATIO_STRIDE]
                .try_into()
                .unwrap(),
        );
        Ok(f16_to_f32(bits))
    }

    pub(crate) fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub(crate) fn gammas(&self) -> &[u8] {
        &self.gammas
    }

    pub(crate) fn error_ratios(&self) -> &[u8] {
        &self.error_ratios
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
        Ok(&self.codes[start..start + self.code_stride])
    }

    pub(crate) fn scale(&self, row: usize) -> crate::Result<f32> {
        self.local_row(row)?;
        let local = row - self.rows.start;
        let start = local * QUANTIZED_SCALE_STRIDE;
        Ok(f32::from_le_bytes(
            self.scales[start..start + QUANTIZED_SCALE_STRIDE]
                .try_into()
                .unwrap(),
        ))
    }

    pub(crate) fn gamma(&self, row: usize) -> crate::Result<f32> {
        self.local_row(row)?;
        let local = row - self.rows.start;
        let start = local * QUANTIZED_GAMMA_STRIDE;
        let bits = u16::from_le_bytes(
            self.gammas[start..start + QUANTIZED_GAMMA_STRIDE]
                .try_into()
                .unwrap(),
        );
        Ok(f16_to_f32(bits))
    }

    pub(crate) fn error_ratio(&self, row: usize) -> crate::Result<f32> {
        self.local_row(row)?;
        let local = row - self.rows.start;
        let start = local * QUANTIZED_ERROR_RATIO_STRIDE;
        let bits = u16::from_le_bytes(
            self.error_ratios[start..start + QUANTIZED_ERROR_RATIO_STRIDE]
                .try_into()
                .unwrap(),
        );
        Ok(f16_to_f32(bits))
    }

    pub(crate) fn constant(&self, row: usize) -> crate::Result<Option<f32>> {
        let local = self.local_row(row)?;
        let Some(constants) = &self.constants else {
            return Ok(None);
        };
        let start = local * QUANTIZED_CONSTANT_STRIDE;
        Ok(Some(f32::from_le_bytes(
            constants[start..start + QUANTIZED_CONSTANT_STRIDE]
                .try_into()
                .unwrap(),
        )))
    }

    pub(crate) fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub(crate) fn scales(&self) -> &[u8] {
        &self.scales
    }

    pub(crate) fn gammas(&self) -> &[u8] {
        &self.gammas
    }

    pub(crate) fn error_ratios(&self) -> &[u8] {
        &self.error_ratios
    }

    pub(crate) fn constants(&self) -> Option<&[u8]> {
        self.constants.as_deref()
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

    fn cluster_rows_containing(&self, rows: &Range<usize>) -> crate::Result<Range<usize>> {
        if rows.is_empty() {
            return Ok(rows.clone());
        }
        let Some(&total_rows) = self.cluster_offsets.last() else {
            return Err(DataCorruption::comment_only(
                "quantized sidecar has no IVF cluster offsets",
            )
            .into());
        };
        if rows.end > total_rows {
            return Err(DataCorruption::comment_only(format!(
                "quantized sidecar row range {rows:?} exceeds {total_rows} posting rows"
            ))
            .into());
        }
        let upper = self
            .cluster_offsets
            .partition_point(|&offset| offset <= rows.start);
        if upper == 0 || upper == self.cluster_offsets.len() {
            return Err(DataCorruption::comment_only(format!(
                "quantized sidecar row range {rows:?} is outside IVF cluster offsets"
            ))
            .into());
        }
        let cluster_rows = self.cluster_offsets[upper - 1]..self.cluster_offsets[upper];
        if rows.end > cluster_rows.end {
            return Err(TantivyError::InternalError(format!(
                "quantized sidecar row range {rows:?} crosses cluster boundary {cluster_rows:?}"
            )));
        }
        Ok(cluster_rows)
    }

    fn sidecar_byte_ranges(
        cluster_rows: &Range<usize>,
        rows: &Range<usize>,
    ) -> (Range<usize>, Range<usize>, Range<usize>) {
        debug_assert!(cluster_rows.start <= rows.start && rows.end <= cluster_rows.end);
        let block_start = cluster_rows.start * QUANTIZED_SIDECAR_STRIDE;
        let scale_start = block_start + (rows.start - cluster_rows.start) * QUANTIZED_SCALE_STRIDE;
        let scale_end = block_start + (rows.end - cluster_rows.start) * QUANTIZED_SCALE_STRIDE;
        let gamma_block_start = block_start + cluster_rows.len() * QUANTIZED_SCALE_STRIDE;
        let gamma_start =
            gamma_block_start + (rows.start - cluster_rows.start) * QUANTIZED_GAMMA_STRIDE;
        let gamma_end =
            gamma_block_start + (rows.end - cluster_rows.start) * QUANTIZED_GAMMA_STRIDE;
        let error_ratio_block_start =
            gamma_block_start + cluster_rows.len() * QUANTIZED_GAMMA_STRIDE;
        let error_ratio_start = error_ratio_block_start
            + (rows.start - cluster_rows.start) * QUANTIZED_ERROR_RATIO_STRIDE;
        let error_ratio_end = error_ratio_block_start
            + (rows.end - cluster_rows.start) * QUANTIZED_ERROR_RATIO_STRIDE;
        (
            scale_start..scale_end,
            gamma_start..gamma_end,
            error_ratio_start..error_ratio_end,
        )
    }

    fn validate_gammas(gammas: &[u8], rows: &Range<usize>) -> crate::Result<()> {
        debug_assert_eq!(gammas.len(), rows.len() * QUANTIZED_GAMMA_STRIDE);
        for (local, bytes) in gammas.chunks_exact(QUANTIZED_GAMMA_STRIDE).enumerate() {
            let gamma = f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap()));
            if !gamma.is_finite() || !(1.0..=4.0).contains(&gamma) {
                return Err(DataCorruption::comment_only(format!(
                    "quantized row {} has invalid cumulative gamma {gamma}; expected finite [1,4]",
                    rows.start + local
                ))
                .into());
            }
        }
        Ok(())
    }

    fn validate_error_ratios(error_ratios: &[u8], rows: &Range<usize>) -> crate::Result<()> {
        debug_assert_eq!(
            error_ratios.len(),
            rows.len() * QUANTIZED_ERROR_RATIO_STRIDE
        );
        for (local, bytes) in error_ratios
            .chunks_exact(QUANTIZED_ERROR_RATIO_STRIDE)
            .enumerate()
        {
            let error_ratio = f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap()));
            if !error_ratio.is_finite() || error_ratio < 0.0 {
                return Err(DataCorruption::comment_only(format!(
                    "quantized row {} has invalid corrected error ratio {error_ratio}; expected \
                     finite and non-negative",
                    rows.start + local
                ))
                .into());
            }
        }
        Ok(())
    }

    /// Pins the scale, gamma, and corrected-error-ratio runs for an in-cluster row range.
    pub(crate) fn read_sidecar(&self, rows: Range<usize>) -> crate::Result<QuantizedSidecarBatch> {
        if rows.is_empty() {
            return Ok(QuantizedSidecarBatch {
                scales: OwnedBytes::empty(),
                gammas: OwnedBytes::empty(),
                error_ratios: OwnedBytes::empty(),
                rows,
            });
        }
        let cluster_rows = self.cluster_rows_containing(&rows)?;
        let (scale_range, gamma_range, error_ratio_range) =
            Self::sidecar_byte_ranges(&cluster_rows, &rows);

        let (scales, gammas, error_ratios) = if rows == cluster_rows {
            let scale_len = scale_range.len();
            let gamma_len = gamma_range.len();
            let (scales, remainder) = self
                .sidecar
                .slice(scale_range.start..error_ratio_range.end)
                .read_bytes()?
                .split(scale_len);
            let (gammas, error_ratios) = remainder.split(gamma_len);
            (scales, gammas, error_ratios)
        } else {
            let spans = [
                storage_block_span(&self.sidecar, scale_range.clone()),
                storage_block_span(&self.sidecar, gamma_range.clone()),
                storage_block_span(&self.sidecar, error_ratio_range.clone()),
            ];
            let one_block_group = spans.iter().flatten().all(|&(start, end)| {
                spans
                    .iter()
                    .flatten()
                    .all(|&(other_start, other_end)| start <= other_end && other_start <= end)
            });
            if one_block_group {
                let bytes = self
                    .sidecar
                    .slice(scale_range.start..error_ratio_range.end)
                    .read_bytes()?;
                let scales = bytes.slice(0..scale_range.len());
                let gamma_start = gamma_range.start - scale_range.start;
                let gammas = bytes.slice(gamma_start..gamma_start + gamma_range.len());
                let error_ratio_start = error_ratio_range.start - scale_range.start;
                let error_ratios =
                    bytes.slice(error_ratio_start..error_ratio_start + error_ratio_range.len());
                (scales, gammas, error_ratios)
            } else {
                (
                    self.sidecar.slice(scale_range).read_bytes()?,
                    self.sidecar.slice(gamma_range).read_bytes()?,
                    self.sidecar.slice(error_ratio_range).read_bytes()?,
                )
            }
        };
        Self::validate_gammas(gammas.as_slice(), &rows)?;
        Self::validate_error_ratios(error_ratios.as_slice(), &rows)?;
        Ok(QuantizedSidecarBatch {
            scales,
            gammas,
            error_ratios,
            rows,
        })
    }

    pub(crate) fn read_codes(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        let codes = self
            .codes
            .slice(rows.start * self.code_stride..rows.end * self.code_stride)
            .read_bytes()?;
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

    pub(crate) fn read_constants(&self, rows: Range<usize>) -> crate::Result<Option<OwnedBytes>> {
        let Some(constants) = &self.constants else {
            return Ok(None);
        };
        Ok(Some(
            constants
                .slice(rows.start * QUANTIZED_CONSTANT_STRIDE..rows.end * QUANTIZED_CONSTANT_STRIDE)
                .read_bytes()?,
        ))
    }

    pub(crate) fn read_batch(&self, rows: Range<usize>) -> crate::Result<QuantizedLayerBatch> {
        if rows.start > rows.end {
            return Err(TantivyError::InternalError(format!(
                "invalid quantized row range {rows:?}"
            )));
        }
        let codes = self.read_codes(rows.clone())?;
        let sidecar = self.read_sidecar(rows.clone())?;
        let constants = self.read_constants(rows.clone())?;
        Ok(QuantizedLayerBatch {
            codes,
            scales: sidecar.scales,
            gammas: sidecar.gammas,
            error_ratios: sidecar.error_ratios,
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
                read_ranges.push(available_rows);
                return;
            }
        }
        let touched_blocks = merged_storage_block_count(block_scratch);

        block_scratch.clear();
        block_scratch.push(
            storage_block_span(
                slot,
                available_rows.start * stride..available_rows.end * stride,
            )
            .expect("storage geometry was resolved above"),
        );
        let covered_blocks = merged_storage_block_count(block_scratch);
        debug_assert!(touched_blocks <= covered_blocks);
        if touched_blocks == covered_blocks {
            read_ranges.push(available_rows);
            return;
        }

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

    fn plan_sidecar_reads_in_cluster(
        &self,
        cluster_rows: Range<usize>,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        debug_assert!(!rows.is_empty());
        debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(rows.iter().all(|row| available_rows.contains(row)));
        debug_assert!(cluster_rows.start <= available_rows.start);
        debug_assert!(available_rows.end <= cluster_rows.end);
        let first_output = read_ranges.len();

        block_scratch.clear();
        for &row in rows {
            let (scale, gamma, error_ratio) =
                Self::sidecar_byte_ranges(&cluster_rows, &(row..row + 1));
            if !append_storage_block_span(&self.sidecar, scale, block_scratch)
                || !append_storage_block_span(&self.sidecar, gamma, block_scratch)
                || !append_storage_block_span(&self.sidecar, error_ratio, block_scratch)
            {
                read_ranges.push(available_rows);
                return;
            }
        }
        let touched_blocks = merged_storage_block_count(block_scratch);

        block_scratch.clear();
        let (available_scales, available_gammas, available_error_ratios) =
            Self::sidecar_byte_ranges(&cluster_rows, &available_rows);
        for range in [available_scales, available_gammas, available_error_ratios] {
            block_scratch.push(
                storage_block_span(&self.sidecar, range)
                    .expect("sidecar storage geometry was resolved above"),
            );
        }
        let covered_blocks = merged_storage_block_count(block_scratch);
        debug_assert!(touched_blocks <= covered_blocks);
        if touched_blocks == covered_blocks {
            read_ranges.push(available_rows);
            return;
        }

        let block_spans = |rows: Range<usize>| {
            let (scale, gamma, error_ratio) = Self::sidecar_byte_ranges(&cluster_rows, &rows);
            (
                storage_block_span(&self.sidecar, scale)
                    .expect("sidecar storage geometry was resolved above"),
                storage_block_span(&self.sidecar, gamma)
                    .expect("sidecar storage geometry was resolved above"),
                storage_block_span(&self.sidecar, error_ratio)
                    .expect("sidecar storage geometry was resolved above"),
            )
        };
        let spans_overlap =
            |left: ((usize, usize), (usize, usize), (usize, usize)),
             right: ((usize, usize), (usize, usize), (usize, usize))| {
                [left.0, left.1, left.2]
                    .into_iter()
                    .any(|(left_start, left_end)| {
                        [right.0, right.1, right.2]
                            .into_iter()
                            .any(|(right_start, right_end)| {
                                left_start <= right_end && right_start <= left_end
                            })
                    })
            };

        let mut first_row = rows[0];
        let mut previous_row = rows[0];
        for &row in &rows[1..] {
            let current = block_spans(first_row..previous_row + 1);
            let next = block_spans(row..row + 1);
            if spans_overlap(current, next) {
                previous_row = row;
            } else {
                read_ranges.push(first_row..previous_row + 1);
                first_row = row;
                previous_row = row;
            }
        }
        read_ranges.push(first_row..previous_row + 1);

        block_scratch.clear();
        for range in read_ranges[first_output..].iter().cloned() {
            let (scale, gamma, error_ratio) = Self::sidecar_byte_ranges(&cluster_rows, &range);
            for range in [scale, gamma, error_ratio] {
                block_scratch.push(
                    storage_block_span(&self.sidecar, range)
                        .expect("sidecar storage geometry was resolved above"),
                );
            }
        }
        if merged_storage_block_count(block_scratch) == covered_blocks {
            read_ranges.truncate(first_output);
            read_ranges.push(available_rows);
        }
    }

    pub(crate) fn plan_sidecar_reads(
        &self,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        debug_assert!(!rows.is_empty());
        debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(rows.iter().all(|row| available_rows.contains(row)));
        read_ranges.clear();

        let mut selected_start = 0usize;
        while selected_start < rows.len() {
            let first_row = rows[selected_start];
            let cluster_rows = self
                .cluster_rows_containing(&(first_row..first_row + 1))
                .expect("indexed sidecar row must belong to an IVF cluster");
            let selected_end = selected_start
                + rows[selected_start..].partition_point(|&row| row < cluster_rows.end);
            let cluster_available = available_rows.start.max(cluster_rows.start)
                ..available_rows.end.min(cluster_rows.end);
            self.plan_sidecar_reads_in_cluster(
                cluster_rows,
                cluster_available,
                &rows[selected_start..selected_end],
                read_ranges,
                block_scratch,
            );
            selected_start = selected_end;
        }
    }

    pub(crate) fn plan_constant_reads(
        &self,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) -> crate::Result<()> {
        let Some(constants) = &self.constants else {
            return Err(TantivyError::InternalError(
                "L2 quantized scoring requires a constants slot".to_string(),
            ));
        };
        Self::plan_slot_reads(
            constants,
            QUANTIZED_CONSTANT_STRIDE,
            available_rows,
            rows,
            read_ranges,
            block_scratch,
        );
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

    pub(crate) fn scale(&self, row: usize) -> crate::Result<f32> {
        self.read_sidecar(row..row + 1)?.scale(row)
    }

    pub(crate) fn gamma(&self, row: usize) -> crate::Result<f32> {
        self.read_sidecar(row..row + 1)?.gamma(row)
    }

    pub(crate) fn error_ratio(&self, row: usize) -> crate::Result<f32> {
        self.read_sidecar(row..row + 1)?.error_ratio(row)
    }

    pub(crate) fn constant(&self, row: usize) -> crate::Result<Option<f32>> {
        let Some(constants) = &self.constants else {
            return Ok(None);
        };
        let bytes = constants
            .slice(row * QUANTIZED_CONSTANT_STRIDE..(row + 1) * QUANTIZED_CONSTANT_STRIDE)
            .read_bytes()?;
        Ok(Some(f32::from_le_bytes(
            bytes.as_slice().try_into().unwrap(),
        )))
    }
}

/// Field-keyed quantized payloads resolved from immutable index metadata.
pub(crate) struct QuantizedFieldReader {
    config: VectorQuantizationConfig,
    index_ctx: OnceLock<Arc<QuantizedIndexCtx>>,
    layers: Vec<QuantizedLayerReader>,
    residual_norms: FileSlice,
}

pub(crate) struct QuantizedResidualNormBatch {
    bytes: OwnedBytes,
    rows: Range<usize>,
}

/// Storage-planned borrowed views of selected fp32 rows.
pub(crate) struct VectorRowBatch {
    selected_rows: Vec<usize>,
    chunks: Vec<VectorRowChunk>,
    stride: usize,
}

struct VectorRowChunk {
    rows: Range<usize>,
    bytes: OwnedBytes,
}

pub(crate) struct VectorRowBatchIter<'a> {
    batch: &'a VectorRowBatch,
    selected: usize,
    chunk: usize,
}

impl VectorRowBatch {
    pub(crate) fn iter(&self) -> VectorRowBatchIter<'_> {
        VectorRowBatchIter {
            batch: self,
            selected: 0,
            chunk: 0,
        }
    }

    #[cfg(test)]
    fn read_count(&self) -> usize {
        self.chunks.len()
    }
}

impl<'a> Iterator for VectorRowBatchIter<'a> {
    type Item = (usize, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let &row = self.batch.selected_rows.get(self.selected)?;
        while self.batch.chunks[self.chunk].rows.end <= row {
            self.chunk += 1;
        }
        let chunk = &self.batch.chunks[self.chunk];
        debug_assert!(chunk.rows.contains(&row));
        let local = row - chunk.rows.start;
        let start = local * self.batch.stride;
        self.selected += 1;
        Some((row, &chunk.bytes[start..start + self.batch.stride]))
    }
}

impl QuantizedResidualNormBatch {
    /// Returns the fixed-stride little-endian f32 range for one cluster pin.
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

    pub(crate) fn residual_norm(&self, row: usize) -> crate::Result<f32> {
        let bytes = self
            .residual_norms
            .slice(row * QUANTIZED_RESIDUAL_NORM_STRIDE..(row + 1) * QUANTIZED_RESIDUAL_NORM_STRIDE)
            .read_bytes()?;
        Ok(f32::from_le_bytes(bytes.as_slice().try_into().unwrap()))
    }

    pub(crate) fn read_residual_norm_batch(
        &self,
        rows: Range<usize>,
    ) -> crate::Result<QuantizedResidualNormBatch> {
        let bytes = self
            .residual_norms
            .slice(
                rows.start * QUANTIZED_RESIDUAL_NORM_STRIDE
                    ..rows.end * QUANTIZED_RESIDUAL_NORM_STRIDE,
            )
            .read_bytes()?;
        Ok(QuantizedResidualNormBatch { bytes, rows })
    }

    pub(crate) fn plan_residual_norm_reads(
        &self,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        QuantizedLayerReader::plan_slot_reads(
            &self.residual_norms,
            QUANTIZED_RESIDUAL_NORM_STRIDE,
            available_rows,
            rows,
            read_ranges,
            block_scratch,
        );
    }

    pub(crate) fn read_residual_norms(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        Ok(self
            .residual_norms
            .slice(
                rows.start * QUANTIZED_RESIDUAL_NORM_STRIDE
                    ..rows.end * QUANTIZED_RESIDUAL_NORM_STRIDE,
            )
            .read_bytes()?)
    }

    pub(crate) fn index_ctx(&self) -> crate::Result<Arc<QuantizedIndexCtx>> {
        if let Some(index_ctx) = self.index_ctx.get() {
            return Ok(Arc::clone(index_ctx));
        }
        let resolved = QuantizedIndexCtx::resolve_from_config(self.config.clone())?;
        let _ = self.index_ctx.set(Arc::clone(&resolved));
        Ok(self.index_ctx.get().map(Arc::clone).unwrap_or(resolved))
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

fn validate_vector_slot_map(composite: &CompositeFile) -> crate::Result<()> {
    if let Some((field, slot)) = composite
        .field_indices()
        .find(|&(_, slot)| slot >= VectorSlot::COUNT)
    {
        return Err(DataCorruption::comment_only(format!(
            "vector composite field {field:?} declares slot {slot}; format {} permits only slots \
             0..{}",
            super::header::VECTOR_FILE_FORMAT_VERSION,
            VectorSlot::COUNT - 1
        ))
        .into());
    }
    Ok(())
}

fn validate_quantization_file_format(
    config: Option<&VectorQuantizationConfig>,
    vector_file_format: VectorFileVersion,
    field_name: &str,
) -> crate::Result<()> {
    if let Some(config) = config {
        if config.format_version != vector_file_format as u32 {
            return Err(DataCorruption::comment_only(format!(
                "vector field {field_name:?} settings format version {} does not match `.vec` \
                 format version {}; rebuild required",
                config.format_version, vector_file_format as u32
            ))
            .into());
        }
    }
    Ok(())
}

/// Per-segment vector row reader with optional IVF routing.
pub struct VectorIndexReader {
    options: VectorOptions,
    /// Distinct documents with a vector.
    num_vectors: usize,
    /// Whether the segment contains vector data for this field.
    present: bool,
    /// `.vec` slot `[0]`
    id_map: IdMap,
    /// Deferred full-precision vector rows.
    rows_slice: FileSlice,
    index: Option<IvfIndex>,
    quantization: Option<QuantizedFieldReader>,
}

impl VectorIndexReader {
    /// Opens a segment's vector data for one field.
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

        let centroid_slots = match segment_reader
            .open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string()))
        {
            Ok(file) => {
                let (centroids_version, body) = read_centroid_header(&file)?;
                let composite = CompositeFile::open(&body)?;
                match (
                    composite.open_read_with_idx(field, CentroidSlot::Centroids.index()),
                    composite.open_read_with_idx(field, CentroidSlot::Offsets.index()),
                    composite.open_read_with_idx(field, CentroidSlot::Bounds.index()),
                ) {
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
                            composite.open_read_with_idx(field, CentroidSlot::Router.index()),
                            bounds,
                        ))
                    }
                    _ => None,
                }
            }
            Err(OpenReadError::FileDoesNotExist(_)) => None,
            Err(err) => return Err(err.into()),
        };

        let vec_file = match segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string()))
        {
            Ok(file) => file,
            Err(OpenReadError::FileDoesNotExist(_)) => return Ok(Self::empty(options)),
            Err(err) => return Err(err.into()),
        };
        let (vector_file_format, body) = read_vector_header(&vec_file)?;
        let vec_composite = CompositeFile::open(&body)?;
        validate_vector_slot_map(&vec_composite)?;
        let (Some(id_map_slice), Some(rows_slice)) = (
            vec_composite.open_read_with_idx(field, VectorSlot::IdMap.index()),
            vec_composite.open_read_with_idx(field, VectorSlot::Rows.index()),
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
        validate_quantization_file_format(
            quantization_config.as_ref(),
            vector_file_format,
            entry.name(),
        )?;

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

        let quantized_slot_present = vec_composite
            .open_read_with_idx(field, VectorSlot::ResidualNorms.index())
            .is_some()
            || (0..MAX_QUANTIZATION_LAYERS).any(|layer| {
                vec_composite
                    .open_read_with_idx(field, VectorSlot::codes(layer).index())
                    .is_some()
                    || vec_composite
                        .open_read_with_idx(field, VectorSlot::sidecar(layer).index())
                        .is_some()
                    || vec_composite
                        .open_read_with_idx(field, VectorSlot::constants(layer).index())
                        .is_some()
            });
        let quantization = match (&index, quantization_config) {
            (Some(index), Some(config)) => {
                let cluster_offsets: Arc<[usize]> = (0..index.num_clusters())
                    .map(|cluster| index.cluster_range(cluster).start)
                    .chain(std::iter::once(index.num_rows()))
                    .collect::<Vec<_>>()
                    .into();
                let mut layers = Vec::with_capacity(config.layers.len());
                for (layer, spec) in config.layers.iter().enumerate() {
                    let code_stride = quantized_code_stride(options.dim(), spec.bits);
                    let (Some(codes), Some(sidecar)) = (
                        vec_composite.open_read_with_idx(field, VectorSlot::codes(layer).index()),
                        vec_composite.open_read_with_idx(field, VectorSlot::sidecar(layer).index()),
                    ) else {
                        return Err(DataCorruption::comment_only(format!(
                            "vector field {:?} has an incomplete configured quantization layer \
                             {layer}",
                            entry.name()
                        ))
                        .into());
                    };
                    let constant_slot = vec_composite
                        .open_read_with_idx(field, VectorSlot::constants(layer).index());
                    let constants = match (config.needs_constants(), constant_slot) {
                        (true, Some(constants)) => Some(logical_slice(
                            constants,
                            num_rows * QUANTIZED_CONSTANT_STRIDE,
                            &format!("vector field {:?} layer {layer} constants", entry.name()),
                        )?),
                        (true, None) => {
                            return Err(DataCorruption::comment_only(format!(
                                "L2 vector field {:?} is missing layer {layer} constants",
                                entry.name()
                            ))
                            .into());
                        }
                        (false, Some(_)) => {
                            return Err(DataCorruption::comment_only(format!(
                                "vector field {:?} carries layer {layer} constants for a metric \
                                 that omits them",
                                entry.name()
                            ))
                            .into());
                        }
                        (false, None) => None,
                    };
                    layers.push(QuantizedLayerReader {
                        codes: logical_slice(
                            codes,
                            num_rows * code_stride,
                            &format!("vector field {:?} layer {layer} codes", entry.name()),
                        )?,
                        sidecar: logical_slice(
                            sidecar,
                            num_rows * QUANTIZED_SIDECAR_STRIDE,
                            &format!(
                                "vector field {:?} layer {layer} \
                                 scale/gamma/corrected-error-ratio sidecar",
                                entry.name()
                            ),
                        )?,
                        constants,
                        cluster_offsets: Arc::clone(&cluster_offsets),
                        code_stride,
                        dim: options.dim(),
                        bits: spec.bits,
                    });
                }
                for layer in config.layers.len()..MAX_QUANTIZATION_LAYERS {
                    if vec_composite
                        .open_read_with_idx(field, VectorSlot::codes(layer).index())
                        .is_some()
                        || vec_composite
                            .open_read_with_idx(field, VectorSlot::sidecar(layer).index())
                            .is_some()
                        || vec_composite
                            .open_read_with_idx(field, VectorSlot::constants(layer).index())
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
                let residual_norms = match vec_composite
                    .open_read_with_idx(field, VectorSlot::ResidualNorms.index())
                {
                    Some(slice) => logical_slice(
                        slice,
                        num_rows * QUANTIZED_RESIDUAL_NORM_STRIDE,
                        &format!("vector field {:?} residual squared norms", entry.name()),
                    )?,
                    None => {
                        return Err(DataCorruption::comment_only(format!(
                            "quantized vector field {:?} is missing residual squared norm slot 2",
                            entry.name()
                        ))
                        .into());
                    }
                };
                Some(QuantizedFieldReader {
                    config,
                    index_ctx: OnceLock::new(),
                    layers,
                    residual_norms,
                })
            }
            (Some(_), None) if quantized_slot_present => {
                return Err(DataCorruption::comment_only(format!(
                    "vector field {:?} carries quantized slots without index metadata",
                    entry.name()
                ))
                .into());
            }
            (None, _) if quantized_slot_present => {
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

    /// Returns a vector reader with no rows or routing index.
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

    /// Returns the field options.
    pub fn options(&self) -> &VectorOptions {
        &self.options
    }

    /// Returns the vector dimension.
    pub fn dim(&self) -> usize {
        self.options.dim()
    }

    /// Number of distinct docs with a vector value.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Returns whether the field contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Returns the optional IVF routing index.
    pub fn index(&self) -> Option<&IvfIndex> {
        self.index.as_ref()
    }

    pub(crate) fn quantization(&self) -> Option<&QuantizedFieldReader> {
        self.quantization.as_ref()
    }

    /// Returns vector storage information when the field is present.
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

    /// Returns posting sizes in cluster order for IVF storage.
    pub fn cluster_sizes(&self) -> Option<Vec<u32>> {
        self.index
            .as_ref()
            .map(|index| index.cluster_sizes().map(|size| size as u32).collect())
    }

    /// `true` if `doc_id` has a stored vector.
    pub fn contains(&self, doc_id: DocId) -> bool {
        self.row_id(doc_id).is_some()
    }

    /// Returns one document's raw little-endian vector bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when the vector row cannot be read.
    pub fn vector_bytes(&self, doc_id: DocId) -> crate::Result<Option<OwnedBytes>> {
        let Some(row) = self.row_id(doc_id) else {
            return Ok(None);
        };
        self.vector_bytes_for_row(row).map(Some)
    }

    /// Returns one dense vector row by row index.
    ///
    /// # Errors
    ///
    /// Returns an error when `row` is out of bounds or cannot be read.
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

    /// Fetches increasing vector rows through a storage-aware range plan.
    pub(crate) fn read_vector_rows_planned(
        &self,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) -> crate::Result<VectorRowBatch> {
        let num_rows = self.id_map.num_rows() as usize;
        if rows.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(TantivyError::InvalidArgument(
                "planned vector rows must be strictly increasing".to_string(),
            ));
        }
        if let Some(&row) = rows.last() {
            if row >= num_rows {
                return Err(TantivyError::InvalidArgument(format!(
                    "vector row {row} is out of bounds"
                )));
            }
        } else {
            read_ranges.clear();
            block_scratch.clear();
            return Ok(VectorRowBatch {
                selected_rows: Vec::new(),
                chunks: Vec::new(),
                stride: self.options.bytes_per_vector(),
            });
        }

        let stride = self.options.bytes_per_vector();
        if storage_block_span(&self.rows_slice, rows[0] * stride..(rows[0] + 1) * stride).is_some()
        {
            QuantizedLayerReader::plan_slot_reads(
                &self.rows_slice,
                stride,
                0..num_rows,
                rows,
                read_ranges,
                block_scratch,
            );
        } else {
            read_ranges.clear();
            block_scratch.clear();
            let mut start = rows[0];
            let mut previous = rows[0];
            for &row in &rows[1..] {
                if row != previous + 1 {
                    read_ranges.push(start..previous + 1);
                    start = row;
                }
                previous = row;
            }
            read_ranges.push(start..previous + 1);
        }

        let mut chunks = Vec::with_capacity(read_ranges.len());
        for row_range in read_ranges.iter().cloned() {
            let bytes = self
                .rows_slice
                .slice(row_range.start * stride..row_range.end * stride)
                .read_bytes()?;
            debug_assert_eq!(bytes.len(), row_range.len() * stride);
            chunks.push(VectorRowChunk {
                rows: row_range,
                bytes,
            });
        }
        Ok(VectorRowBatch {
            selected_rows: rows.to_vec(),
            chunks,
            stride,
        })
    }

    /// Returns the document id at one IVF row.
    ///
    /// # Panics
    ///
    /// Panics when the reader is not IVF-backed or `row` is out of bounds.
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

    /// Returns the sorted document ids assigned to an IVF cluster.
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

    /// Returns the dense row containing a document's vector.
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

    use quant_model::f16::f32_to_f16;

    use super::*;
    use crate::directory::{CompositeWrite, FileHandle};
    use crate::vector::header::{read_vector_header, write_vector_header};

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

    fn test_cluster_offsets(offsets: &[usize]) -> Arc<[usize]> {
        Arc::from(offsets)
    }

    fn test_sidecar(scales: &[f32], gammas: &[f32]) -> Vec<u8> {
        test_sidecar_with_error_ratios(scales, gammas, &vec![0.25; scales.len()])
    }

    fn test_sidecar_with_error_ratios(
        scales: &[f32],
        gammas: &[f32],
        corrected_error_ratios: &[f32],
    ) -> Vec<u8> {
        assert_eq!(scales.len(), gammas.len());
        assert_eq!(scales.len(), corrected_error_ratios.len());
        scales
            .iter()
            .flat_map(|scale| scale.to_le_bytes())
            .chain(
                gammas
                    .iter()
                    .flat_map(|&gamma| f32_to_f16(gamma).to_le_bytes()),
            )
            .chain(
                corrected_error_ratios
                    .iter()
                    .flat_map(|&error_ratio| f32_to_f16(error_ratio).to_le_bytes()),
            )
            .collect()
    }

    #[test]
    fn planned_vector_rows_coalesce_reads_and_preserve_exact_scores() -> crate::Result<()> {
        const DIM: usize = 2;
        const ROWS: usize = 16;
        let row_bytes = (0..ROWS)
            .flat_map(|row| {
                let values = [row as f32 + 0.25, row as f32 * -0.5 + 1.0];
                values.into_iter().flat_map(f32::to_le_bytes)
            })
            .collect::<Vec<_>>();
        let expected_bytes = row_bytes.clone();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let storage = Arc::new(BlockTrackedBytes {
            bytes: row_bytes,
            reads: Arc::clone(&reads),
            block_len: 32,
        });
        let reader = VectorIndexReader {
            options: VectorOptions::new(DIM, Metric::Dot),
            num_vectors: ROWS,
            present: true,
            id_map: IdMap::Identity {
                num_docs: ROWS as u32,
            },
            rows_slice: FileSlice::new(storage),
            index: None,
            quantization: None,
        };
        let selected = [1, 2, 10, 11];
        let mut ranges = Vec::new();
        let mut block_scratch = Vec::new();
        let batch = reader.read_vector_rows_planned(&selected, &mut ranges, &mut block_scratch)?;

        assert_eq!(batch.read_count(), 2);
        assert_eq!(ranges, [1..3, 10..12]);
        assert_eq!(&*reads.lock().unwrap(), &[8..24, 80..96]);
        assert!(batch.read_count() < selected.len());

        let query = PreparedQuery::<f32>::new(Metric::Dot, Arc::new(vec![2.0, -0.5]));
        let actual = batch
            .iter()
            .map(|(row, bytes)| (row, query.score_doc_bytes(bytes).to_bits()))
            .collect::<Vec<_>>();
        let stride = DIM * std::mem::size_of::<f32>();
        let expected = selected
            .iter()
            .map(|&row| {
                let bytes = &expected_bytes[row * stride..(row + 1) * stride];
                (row, query.score_doc_bytes(bytes).to_bits())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn quantized_layer_reader_rejects_non_zero_tail() -> crate::Result<()> {
        let mut codes = vec![0_u8; quantized_code_stride(65, 1)];
        codes[8] = 1;
        let valid = QuantizedLayerReader {
            codes: FileSlice::from(codes.clone()),
            sidecar: FileSlice::empty(),
            constants: None,
            cluster_offsets: test_cluster_offsets(&[0, 1]),
            code_stride: codes.len(),
            dim: 65,
            bits: 1,
        };
        assert_eq!(valid.code_bytes(0)?.as_slice(), codes);

        codes[8] |= 2;
        let corrupt = QuantizedLayerReader {
            codes: FileSlice::from(codes.clone()),
            sidecar: FileSlice::empty(),
            constants: None,
            cluster_offsets: test_cluster_offsets(&[0, 1]),
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
            sidecar: FileSlice::from(test_sidecar(&[0.0; 3], &[1.0; 3])),
            constants: None,
            cluster_offsets: test_cluster_offsets(&[0, 3]),
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
        let sidecar = test_sidecar(&[17.0, 29.0], &[1.0, 4.0]);
        let constants = [0.25_f32, -0.75_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let reader = QuantizedLayerReader {
            codes: FileSlice::from(codes.clone()),
            sidecar: FileSlice::from(sidecar),
            constants: Some(FileSlice::from(constants)),
            cluster_offsets: test_cluster_offsets(&[0, 2]),
            code_stride: stride,
            dim: 65,
            bits: 1,
        };
        let batch = reader.read_batch(0..2)?;
        assert_eq!(batch.code_bytes(0)?, &codes[..stride]);
        assert_eq!(batch.code_bytes(1)?, &codes[stride..]);
        assert_eq!(batch.scale(0)?, 17.0);
        assert_eq!(batch.scale(1)?, 29.0);
        assert_eq!(batch.gamma(0)?, 1.0);
        assert_eq!(batch.gamma(1)?, 4.0);
        assert_eq!(batch.error_ratio(0)?, 0.25);
        assert_eq!(batch.error_ratio(1)?, 0.25);
        assert_eq!(batch.scales().len(), 2 * QUANTIZED_SCALE_STRIDE);
        assert_eq!(batch.gammas().len(), 2 * QUANTIZED_GAMMA_STRIDE);
        assert_eq!(batch.error_ratios().len(), 2 * QUANTIZED_ERROR_RATIO_STRIDE);
        assert_eq!(batch.constant(0)?.unwrap().to_bits(), 0.25_f32.to_bits());
        assert_eq!(batch.constant(1)?.unwrap().to_bits(), (-0.75_f32).to_bits());

        codes[stride + 8] |= 2;
        let corrupt = QuantizedLayerReader {
            codes: FileSlice::from(codes),
            sidecar: FileSlice::from(test_sidecar(&[0.0, 0.0], &[1.0, 0.5])),
            constants: Some(FileSlice::from(vec![0_u8; 8])),
            cluster_offsets: test_cluster_offsets(&[0, 2]),
            code_stride: stride,
            dim: 65,
            bits: 1,
        };
        assert!(corrupt.read_batch(0..2).is_err());
        Ok(())
    }

    #[test]
    fn cluster_blocked_sidecar_maps_scales_and_gammas_by_cluster() -> crate::Result<()> {
        let mut sidecar = test_sidecar(&[11.0, 12.0], &[1.0, 2.0]);
        sidecar.extend(test_sidecar(&[13.0], &[3.0]));
        let reader = QuantizedLayerReader {
            codes: FileSlice::empty(),
            sidecar: FileSlice::from(sidecar),
            constants: None,
            cluster_offsets: test_cluster_offsets(&[0, 2, 3]),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };

        let first = reader.read_sidecar(0..2)?;
        assert_eq!(first.scale(0)?, 11.0);
        assert_eq!(first.scale(1)?, 12.0);
        assert_eq!(first.gamma(0)?, 1.0);
        assert_eq!(first.gamma(1)?, 2.0);
        assert_eq!(first.error_ratio(0)?, 0.25);
        assert_eq!(first.error_ratio(1)?, 0.25);
        let second = reader.read_sidecar(2..3)?;
        assert_eq!(second.scale(2)?, 13.0);
        assert_eq!(second.gamma(2)?, 3.0);
        assert_eq!(second.error_ratio(2)?, 0.25);
        assert_eq!(reader.scale(2)?, 13.0);
        assert_eq!(reader.gamma(2)?, 3.0);
        Ok(())
    }

    #[test]
    fn sidecar_gamma_validation_is_range_scoped() {
        for gamma in [0.5, 5.0, f32::INFINITY, f32::NAN] {
            let reader = QuantizedLayerReader {
                codes: FileSlice::empty(),
                sidecar: FileSlice::from(test_sidecar(&[17.0], &[gamma])),
                constants: None,
                cluster_offsets: test_cluster_offsets(&[0, 1]),
                code_stride: 8,
                dim: 64,
                bits: 1,
            };
            assert!(reader.read_sidecar(0..1).is_err(), "gamma={gamma}");
        }
    }

    #[test]
    fn sidecar_error_ratio_validation_is_range_scoped() {
        for error_ratio in [-0.5, f32::INFINITY, f32::NAN] {
            let reader = QuantizedLayerReader {
                codes: FileSlice::empty(),
                sidecar: FileSlice::from(test_sidecar_with_error_ratios(
                    &[17.0],
                    &[1.0],
                    &[error_ratio],
                )),
                constants: None,
                cluster_offsets: test_cluster_offsets(&[0, 1]),
                code_stride: 8,
                dim: 64,
                bits: 1,
            };
            assert!(
                reader.read_sidecar(0..1).is_err(),
                "error_ratio={error_ratio}"
            );
        }
    }

    #[test]
    fn indexed_sidecar_plan_is_sparse_across_all_three_runs() -> crate::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let storage = Arc::new(BlockTrackedBytes {
            bytes: test_sidecar(&[0.0; 8], &[1.0; 8]),
            reads: Arc::clone(&reads),
            block_len: 8,
        });
        let reader = QuantizedLayerReader {
            codes: FileSlice::empty(),
            sidecar: FileSlice::new(storage),
            constants: None,
            cluster_offsets: test_cluster_offsets(&[0, 8]),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();

        reader.plan_sidecar_reads(0..8, &[1, 2], &mut ranges, &mut blocks);
        assert_eq!(ranges, [1..3]);
        let sparse = reader.read_sidecar(ranges.pop().unwrap())?;
        assert_eq!(sparse.scales().len(), 2 * QUANTIZED_SCALE_STRIDE);
        assert_eq!(sparse.gammas().len(), 2 * QUANTIZED_GAMMA_STRIDE);
        assert_eq!(
            sparse.error_ratios().len(),
            2 * QUANTIZED_ERROR_RATIO_STRIDE
        );
        assert_eq!(&*reads.lock().unwrap(), &[4..12, 34..38, 50..54]);

        reads.lock().unwrap().clear();
        reader.plan_sidecar_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..1, 7..8]);
        for range in ranges.drain(..) {
            reader.read_sidecar(range)?;
        }
        assert_eq!(
            &*reads.lock().unwrap(),
            &[0..4, 32..34, 48..50, 28..32, 46..48, 62..64]
        );
        Ok(())
    }

    #[test]
    fn indexed_sidecar_plan_splits_cross_cluster_cosine_batches() -> crate::Result<()> {
        let mut sidecar = test_sidecar(&[10.0, 11.0, 12.0], &[1.0, 1.5, 2.0]);
        sidecar.extend(test_sidecar(&[20.0, 21.0, 22.0], &[2.5, 3.0, 3.5]));
        let reader = QuantizedLayerReader {
            codes: FileSlice::empty(),
            sidecar: FileSlice::from(sidecar),
            constants: None,
            cluster_offsets: test_cluster_offsets(&[0, 3, 6]),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();

        reader.plan_sidecar_reads(1..5, &[1, 4], &mut ranges, &mut blocks);
        assert_eq!(ranges, [1..3, 3..5]);
        let first = reader.read_sidecar(ranges[0].clone())?;
        let second = reader.read_sidecar(ranges[1].clone())?;
        assert_eq!(first.scale(1)?, 11.0);
        assert_eq!(first.gamma(1)?, 1.5);
        assert_eq!(first.error_ratio(1)?, 0.25);
        assert_eq!(second.scale(4)?, 21.0);
        assert_eq!(second.gamma(4)?, 3.0);
        assert_eq!(second.error_ratio(4)?, 0.25);
        Ok(())
    }

    #[test]
    fn indexed_read_plan_is_density_adaptive_across_soa_slots() -> crate::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut storage_bytes = vec![0_u8; 192];
        storage_bytes[80..144].copy_from_slice(&test_sidecar(&[0.0; 8], &[1.0; 8]));
        let storage = Arc::new(BlockTrackedBytes {
            bytes: storage_bytes,
            reads: Arc::clone(&reads),
            block_len: 16,
        });
        let codes = FileSlice::new(storage.clone()).slice(15..79);
        let sidecar = FileSlice::new(storage.clone()).slice(80..144);
        let constants = FileSlice::new(storage).slice(144..176);
        let reader = QuantizedLayerReader {
            codes,
            sidecar,
            constants: Some(constants),
            cluster_offsets: test_cluster_offsets(&[0, 8]),
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
        reader.plan_sidecar_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        reader.read_sidecar(ranges.pop().unwrap())?;
        reader.plan_constant_reads(0..8, &[0, 7], &mut ranges, &mut blocks)?;
        assert_eq!(ranges, [0..8]);
        reader.read_constants(ranges.pop().unwrap())?;
        let reads = reads.lock().unwrap();
        assert_eq!(&*reads, &[15..23, 71..79, 80..144, 144..176]);
        let mut pinned_blocks = reads
            .iter()
            .flat_map(|range| range.start / 16..=(range.end - 1) / 16)
            .collect::<Vec<_>>();
        let pin_count = pinned_blocks.len();
        pinned_blocks.sort_unstable();
        let mut previous = None;
        pinned_blocks.retain(|block| {
            let keep = previous != Some(*block);
            previous = Some(*block);
            keep
        });
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
            sidecar: FileSlice::from(test_sidecar(&[0.0; 8], &[1.0; 8])),
            constants: None,
            cluster_offsets: test_cluster_offsets(&[0, 8]),
            code_stride: 8,
            dim: 64,
            bits: 1,
        };
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();
        reader.plan_code_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
        reader.plan_sidecar_reads(0..8, &[0, 7], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..8]);
    }

    #[test]
    fn maximal_trailers_are_trimmed_from_logical_arrays() -> crate::Result<()> {
        let field = Field::from_field_id(0);
        let mut bytes = Vec::new();
        write_vector_header(&mut bytes)?;
        {
            let mut writer = CompositeWrite::wrap(&mut bytes);
            let rows = writer.for_field_with_idx(field, VectorSlot::Rows.index());
            rows.write_all(&[1, 2, 3, 4])?;
            rows.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(1), 0)
                .write_all(&[9])?;

            let sidecar = writer.for_field_with_idx(field, VectorSlot::sidecar(0).index());
            sidecar.write_all(&17_f32.to_le_bytes())?;
            sidecar.write_all(&f32_to_f16(1.5).to_le_bytes())?;
            sidecar.write_all(&f32_to_f16(0.75).to_le_bytes())?;
            sidecar.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(2), 0)
                .write_all(&[9])?;

            let constants = writer.for_field_with_idx(field, VectorSlot::constants(0).index());
            constants.write_all(&1.25_f32.to_le_bytes())?;
            constants.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(3), 0)
                .write_all(&[9])?;

            let norms = writer.for_field_with_idx(field, VectorSlot::ResidualNorms.index());
            norms.write_all(&2.5_f32.to_le_bytes())?;
            norms.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(4), 0)
                .write_all(&[9])?;
            writer.close()?;
        }

        let (_, body) = read_vector_header(&FileSlice::from(bytes))?;
        let composite = CompositeFile::open(&body)?;

        let rows = logical_slice(
            composite
                .open_read_with_idx(field, VectorSlot::Rows.index())
                .unwrap(),
            4,
            "rows fixture",
        )?
        .read_bytes()?;
        let sidecar = logical_slice(
            composite
                .open_read_with_idx(field, VectorSlot::sidecar(0).index())
                .unwrap(),
            QUANTIZED_SIDECAR_STRIDE,
            "scale/gamma/corrected-error-ratio fixture",
        )?
        .read_bytes()?;
        let constants = logical_slice(
            composite
                .open_read_with_idx(field, VectorSlot::constants(0).index())
                .unwrap(),
            4,
            "constants fixture",
        )?
        .read_bytes()?;
        let norms = logical_slice(
            composite
                .open_read_with_idx(field, VectorSlot::ResidualNorms.index())
                .unwrap(),
            4,
            "residual norms fixture",
        )?
        .read_bytes()?;

        assert_eq!(rows.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(
            f32::from_le_bytes(sidecar[..QUANTIZED_SCALE_STRIDE].try_into().unwrap()),
            17.0
        );
        assert_eq!(
            f16_to_f32(u16::from_le_bytes(
                sidecar[QUANTIZED_SCALE_STRIDE..QUANTIZED_SCALE_STRIDE + QUANTIZED_GAMMA_STRIDE]
                    .try_into()
                    .unwrap()
            )),
            1.5
        );
        assert_eq!(
            f16_to_f32(u16::from_le_bytes(
                sidecar[QUANTIZED_SCALE_STRIDE + QUANTIZED_GAMMA_STRIDE..]
                    [..QUANTIZED_ERROR_RATIO_STRIDE]
                    .try_into()
                    .unwrap()
            )),
            0.75
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
    fn vector_format_rejects_slots_outside_the_fixed_map() -> crate::Result<()> {
        let field = Field::from_field_id(0);
        let mut bytes = Vec::new();
        write_vector_header(&mut bytes)?;
        {
            let mut writer = CompositeWrite::wrap(&mut bytes);
            writer
                .for_field_with_idx(field, VectorSlot::COUNT)
                .write_all(&[1])?;
            writer.close()?;
        }

        let (_, body) = read_vector_header(&FileSlice::from(bytes))?;
        let composite = CompositeFile::open(&body)?;
        let error = validate_vector_slot_map(&composite).unwrap_err();
        assert!(error.to_string().contains("permits only slots 0..11"));
        Ok(())
    }

    #[test]
    fn nonzero_alignment_trailer_is_corruption() {
        let slice = FileSlice::from(vec![1, 2, 3, 4, 0, 7]);
        assert!(logical_slice(slice, 4, "bad fixture").is_err());
    }
}
