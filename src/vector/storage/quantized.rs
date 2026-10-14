//! Quantized layer storage: layer and cluster readers, read costs and decoded batches.

use std::ops::Range;
use std::sync::Arc;

use common::{HasLen, OwnedBytes};
use quant::cascade::{LayerSpec, QueryRotationPlan};
use quant::model::f16::f16_to_f32;
use quant::model::Grid;

use crate::directory::FileSlice;
use crate::error::DataCorruption;
#[cfg(test)]
use crate::vector::quantization::VectorQuantizationConfig;
use crate::vector::quantization::{
    quantized_code_tail_is_zero, QUANTIZED_CONSTANT_STRIDE, QUANTIZED_ERROR_RATIO_STRIDE,
    QUANTIZED_GAMMA_STRIDE, QUANTIZED_SCALE_STRIDE,
};
use crate::vector::storage::blocks::Blocks;
use crate::vector::storage::io::VectorRead;
use crate::vector::storage::metadata::{SlotType, VectorColMetadata};
use crate::vector::storage::reader::VectorRowChunk;
use crate::TantivyError;

/// Deferred code, sidecar, and optional L2-constant slices for one residual layer.
pub(crate) struct QuantizedLayerReader {
    pub(crate) blocks: Arc<Blocks>,
    pub(crate) layer: usize,
    pub(crate) codes: usize,
    pub(crate) scales: usize,
    pub(crate) gammas: usize,
    pub(crate) errors: usize,
    pub(crate) constants: Option<usize>,
    pub(crate) code_stride: usize,
    pub(crate) dim: usize,
    pub(crate) bits: u8,
}

// --- Read costs ---

/// A read's cost: the storage blocks it spans with block geometry, its bytes without.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ReadCost(pub(crate) usize);

impl std::ops::Add for ReadCost {
    type Output = ReadCost;

    fn add(self, rhs: ReadCost) -> ReadCost {
        ReadCost(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for ReadCost {
    fn add_assign(&mut self, rhs: ReadCost) {
        self.0 += rhs.0;
    }
}

/// The cost of reading `range` of `slice` in the unit of storage with `block_len`-byte blocks:
/// the blocks from its first to its last byte, by the slice's own geometry or, without one, as if
/// the range started on a block boundary. Bytes when `block_len` is `None`.
pub(crate) fn range_cost(
    slice: &FileSlice,
    range: Range<usize>,
    block_len: Option<usize>,
) -> ReadCost {
    if range.is_empty() {
        return ReadCost(0);
    }
    let Some(block_len) = block_len else {
        return ReadCost(range.len());
    };
    match storage_block_span(slice, range.clone()) {
        Some((first, last)) => ReadCost(last - first + 1),
        None => ReadCost(range.len().div_ceil(block_len)),
    }
}

/// The cost of reading full-precision rows of `stride` bytes from one cluster's `column`: the
/// distinct blocks the rows at ascending `offsets` touch, or every row's span for `None`; bytes
/// without block geometry.
pub(crate) fn exact_read_cost(
    column: &FileSlice,
    stride: usize,
    offsets: Option<&[u32]>,
    rows: usize,
) -> ReadCost {
    let Some(offsets) = offsets else {
        return range_cost(column, 0..rows * stride, column.storage_block_len());
    };
    if column.storage_block_len().is_none() {
        return ReadCost(offsets.len() * stride);
    }
    // Ascending rows: a block counts once, when it lies past the last block counted.
    let mut blocks = 0usize;
    let mut next = 0usize;
    for &offset in offsets {
        let offset = offset as usize;
        let Some((first, last)) =
            storage_block_span(column, offset * stride..(offset + 1) * stride)
        else {
            continue;
        };
        let first = first.max(next);
        if first <= last {
            blocks += last - first + 1;
            next = last + 1;
        }
    }
    ReadCost(blocks)
}

/// Reusable buffers for planning and reading one batch's rows.
#[derive(Default)]
pub(crate) struct ReadScratch {
    /// Global rows of the batch's selection, ascending.
    pub(crate) rows: Vec<usize>,
    /// Planned row runs, in row order: code runs of a sparse layer read, row runs of an exact
    /// read.
    pub(crate) runs: Vec<Range<usize>>,
    /// Offsets of the selected rows within one run or band, as the scoring kernels index them.
    pub(crate) offsets: Vec<usize>,
    /// Storage-block runs of an exact read.
    pub(crate) blocks: Vec<(usize, usize)>,
    /// The row chunks of an exact read, borrowed by its batch.
    pub(crate) chunks: Vec<VectorRowChunk>,
}

/// Pinned SoA ranges for one contiguous cluster posting.
pub(crate) struct QuantizedLayerBatch {
    pub(crate) residual_norms: Option<OwnedBytes>,
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
    constants: Option<OwnedBytes>,
    rows: Range<usize>,
}

impl QuantizedSidecarBatch {
    /// Decodes all selected sidecars together, preserving selected-row order.
    pub(crate) fn decode_selected(
        &self,
        rows: &[usize],
        scales: &mut [f32],
        gammas: &mut [f32],
        errors: &mut [f32],
        constants: &mut [f32],
    ) -> crate::Result<()> {
        if !constants.is_empty() && self.constants.is_none() {
            return Err(DataCorruption::comment_only(
                "quantized L2 field is missing a constants slot",
            )
            .into());
        }
        for (i, &row) in rows.iter().enumerate() {
            let local = self.local_row(row)?;
            let f32_offset = local * QUANTIZED_SCALE_STRIDE;
            let f16_offset = local * QUANTIZED_GAMMA_STRIDE;
            scales[i] =
                f32::from_le_bytes(self.scales[f32_offset..f32_offset + 4].try_into().unwrap());
            gammas[i] = f16_to_f32(u16::from_le_bytes(
                self.gammas[f16_offset..f16_offset + 2].try_into().unwrap(),
            ));
            errors[i] = f16_to_f32(u16::from_le_bytes(
                self.error_ratios[f16_offset..f16_offset + 2]
                    .try_into()
                    .unwrap(),
            ));
            if let Some(bytes) = &self.constants {
                constants[i] =
                    f32::from_le_bytes(bytes[f32_offset..f32_offset + 4].try_into().unwrap());
            }
        }
        Ok(())
    }

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

    #[cfg(test)]
    pub(crate) fn scales(&self) -> &[u8] {
        &self.scales
    }

    #[cfg(test)]
    pub(crate) fn gammas(&self) -> &[u8] {
        &self.gammas
    }

    #[cfg(test)]
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

/// Rejects a decoded sidecar batch whose gammas leave the serialized clamp or
/// whose corrected-error ratios are not finite and non-negative. Runs once per
/// scored batch on the decoded values, over the selected rows only.
pub(crate) fn validate_decoded_sidecar(
    gammas: &[f32],
    error_ratios: &[f32],
    first_row: usize,
) -> crate::Result<()> {
    debug_assert_eq!(gammas.len(), error_ratios.len());
    for (index, &gamma) in gammas.iter().enumerate() {
        if !gamma.is_finite() || !(1.0..=4.0).contains(&gamma) {
            let row = first_row + index;
            return Err(DataCorruption::comment_only(format!(
                "quantized row {row} has invalid cumulative gamma {gamma}; expected finite [1,4]"
            ))
            .into());
        }
    }
    for (index, &error_ratio) in error_ratios.iter().enumerate() {
        if !error_ratio.is_finite() || error_ratio < 0.0 {
            let row = first_row + index;
            return Err(DataCorruption::comment_only(format!(
                "quantized row {row} has invalid corrected error ratio {error_ratio}; expected \
                 finite and non-negative"
            ))
            .into());
        }
    }
    Ok(())
}

/// Splits strictly increasing `rows` into maximal runs of consecutive rows.
fn push_consecutive_runs(rows: &[usize], read_ranges: &mut Vec<Range<usize>>) {
    debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
    let Some(&first) = rows.first() else { return };
    let mut start = first;
    let mut previous = first;
    for &row in &rows[1..] {
        if row != previous + 1 {
            read_ranges.push(start..previous + 1);
            start = row;
        }
        previous = row;
    }
    read_ranges.push(start..previous + 1);
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
    /// Whether this layer's band carries the rows' residual norms: layer 0 only.
    pub(crate) fn has_norms(&self) -> bool {
        self.layer == 0
    }
    /// The cost of reading `cluster`'s whole band in one request.
    pub(crate) fn band_cost(&self, cluster: usize) -> crate::Result<ReadCost> {
        let band = self.blocks.band_slice(cluster, self.layer)?;
        Ok(range_cost(&band, 0..band.len(), band.storage_block_len()))
    }
    /// The cost of reading `cluster`'s rows at `offsets` sparsely: their planned code runs, the
    /// sidecar span and, with norms, the norms column. Leaves the rows and runs in `scratch`.
    pub(crate) fn sparse_cost(
        &self,
        cluster: usize,
        offsets: &[u32],
        scratch: &mut ReadScratch,
    ) -> crate::Result<ReadCost> {
        let reader = self.cluster_in_block(cluster)?;
        let origin = reader.rows.start;
        scratch.rows.clear();
        scratch
            .rows
            .extend(offsets.iter().map(|&offset| origin + offset as usize));
        reader.plan_codes(&scratch.rows, &mut scratch.runs);
        let stride = self.code_stride;
        let unit = reader.codes.storage_block_len();
        let mut cost = ReadCost(0);
        for run in &scratch.runs {
            cost += range_cost(
                &reader.codes,
                (run.start - origin) * stride..(run.end - origin) * stride,
                unit,
            );
        }
        let sidecar = reader.sidecar_slice()?;
        cost += range_cost(&sidecar, 0..sidecar.len(), unit);
        if self.has_norms() {
            let norms = reader.norms_slice()?;
            cost += range_cost(&norms, 0..norms.len(), unit);
        }
        Ok(cost)
    }
    fn validate_codes(&self, bytes: &OwnedBytes, rows: &Range<usize>) -> crate::Result<()> {
        for (local, code) in bytes.chunks_exact(self.code_stride).enumerate() {
            if !quantized_code_tail_is_zero(code, self.dim, self.bits) {
                return Err(DataCorruption::comment_only(format!(
                    "quantized row {} has non-zero padding bits for d={} b={}",
                    rows.start + local,
                    self.dim,
                    self.bits
                ))
                .into());
            }
        }
        Ok(())
    }
    pub(crate) fn read_codes(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        let bytes = self.blocks.read_column(self.codes, rows.clone())?;
        self.validate_codes(&bytes, &rows)?;
        Ok(bytes)
    }
    pub(crate) fn read_sidecar(&self, rows: Range<usize>) -> crate::Result<QuantizedSidecarBatch> {
        Ok(QuantizedSidecarBatch {
            scales: self.blocks.read_column(self.scales, rows.clone())?,
            gammas: self.blocks.read_column(self.gammas, rows.clone())?,
            error_ratios: self.blocks.read_column(self.errors, rows.clone())?,
            constants: None,
            rows,
        })
    }
    pub(crate) fn read_constants(&self, rows: Range<usize>) -> crate::Result<Option<OwnedBytes>> {
        self.constants
            .map(|idx| self.blocks.read_column(idx, rows))
            .transpose()
    }
    /// Pins one band's columns in a single request, then restricts views to the requested rows.
    #[cfg(test)]
    pub(crate) fn read_batch(&self, rows: Range<usize>) -> crate::Result<QuantizedLayerBatch> {
        let b = self.blocks.block_for_range(&rows)?;
        self.read_batch_in_block(b, rows)
    }
    /// Uses a known block without resolving its global row range again.
    pub(crate) fn read_batch_in_block(
        &self,
        b: usize,
        rows: Range<usize>,
    ) -> crate::Result<QuantizedLayerBatch> {
        self.blocks.check_block_rows(b, &rows)?;
        let (span, bytes) = self.blocks.read_band(b, self.layer)?;
        let view = |idx: usize| -> OwnedBytes {
            let column = crate::vector::storage::blocks::column_range(
                &self.blocks.slots,
                self.blocks.rows_in(b),
                idx,
            );
            let start = self.blocks.block_start(b) + column.start - span.start;
            let stride = self.blocks.slots[idx].stride as usize;
            let first = self.blocks.block_rows[b];
            bytes.slice(start + (rows.start - first) * stride..start + (rows.end - first) * stride)
        };
        let codes = view(self.codes);
        self.validate_codes(&codes, &rows)?;
        Ok(QuantizedLayerBatch {
            codes,
            scales: view(self.scales),
            gammas: view(self.gammas),
            error_ratios: view(self.errors),
            constants: self.constants.map(view),
            residual_norms: self.has_norms().then(|| {
                view(
                    self.blocks
                        .slots
                        .iter()
                        .position(|slot| matches!(slot.slot_type, SlotType::ResidualNorms))
                        .expect("validated residual norms column"),
                )
            }),
            rows,
            code_stride: self.code_stride,
        })
    }
    /// Groups code rows only while their page spans overlap, trimming each range to its
    /// first and last selected row. Adjacent disjoint pages start a new group; a straddling row
    /// connects overlapping spans on both pages. Paged storage
    /// copies multi-page requests, so equal page counts do not make a wider available span free.
    /// Storage without page geometry uses consecutive selected rows.
    fn plan_code_slot_reads(
        slot: &FileSlice,
        stride: usize,
        row_origin: usize,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
    ) {
        let span = |row: usize| {
            storage_block_span(
                slot,
                (row - row_origin) * stride..(row + 1 - row_origin) * stride,
            )
        };
        let Some((_, mut end_page)) = span(rows[0]) else {
            push_consecutive_runs(rows, read_ranges);
            return;
        };
        let mut first = rows[0];
        let mut last = first;
        for &row in &rows[1..] {
            let (start, end) = span(row).expect("storage geometry was resolved above");
            if start <= end_page {
                end_page = end_page.max(end);
            } else {
                read_ranges.push(first..last + 1);
                first = row;
                end_page = end;
            }
            last = row;
        }
        read_ranges.push(first..last + 1);
    }

    pub(crate) fn plan_slot_reads(
        slot: &FileSlice,
        stride: usize,
        row_origin: usize,
        available_rows: Range<usize>,
        rows: &[usize],
        read_ranges: &mut Vec<Range<usize>>,
        block_scratch: &mut Vec<(usize, usize)>,
    ) {
        debug_assert!(!rows.is_empty());
        debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(rows.iter().all(|row| available_rows.contains(row)));
        let range_start = read_ranges.len();

        block_scratch.clear();
        for &row in rows {
            if !append_storage_block_span(
                slot,
                (row - row_origin) * stride..(row + 1 - row_origin) * stride,
                block_scratch,
            ) {
                push_consecutive_runs(rows, read_ranges);
                return;
            }
        }
        let touched_blocks = merged_storage_block_count(block_scratch);

        block_scratch.clear();
        block_scratch.push(
            storage_block_span(
                slot,
                (available_rows.start - row_origin) * stride
                    ..(available_rows.end - row_origin) * stride,
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
        let (_, mut group_end_block) = storage_block_span(
            slot,
            (rows[0] - row_origin) * stride..(rows[0] + 1 - row_origin) * stride,
        )
        .expect("storage geometry was resolved above");
        for &row in &rows[1..] {
            let (start_block, end_block) = storage_block_span(
                slot,
                (row - row_origin) * stride..(row + 1 - row_origin) * stride,
            )
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

        debug_assert!(read_ranges[range_start..].windows(2).all(|pair| {
            let (_, left_end) = storage_block_span(
                slot,
                (pair[0].start - row_origin) * stride..(pair[0].end - row_origin) * stride,
            )
            .unwrap();
            let (right_start, _) = storage_block_span(
                slot,
                (pair[1].start - row_origin) * stride..(pair[1].end - row_origin) * stride,
            )
            .unwrap();
            left_end < right_start
        }));
    }

    /// Plans global row identifiers against a block-local column and restores global ranges.
    #[cfg(test)]
    pub(crate) fn plan_column_reads(
        &self,
        idx: usize,
        available: Range<usize>,
        rows: &[usize],
        ranges: &mut Vec<Range<usize>>,
        scratch: &mut Vec<(usize, usize)>,
    ) {
        plan_block_column(&self.blocks, idx, available, rows, ranges, scratch).unwrap();
    }
    #[cfg(test)]
    pub(crate) fn plan_code_reads(
        &self,
        available: Range<usize>,
        rows: &[usize],
        ranges: &mut Vec<Range<usize>>,
        scratch: &mut Vec<(usize, usize)>,
    ) {
        self.plan_column_reads(self.codes, available, rows, ranges, scratch);
    }
    /// Resolves a cluster once so ordered code reads need no repeated block lookup.
    #[cfg(test)]
    pub(crate) fn cluster(&self, row: usize) -> crate::Result<QuantizedClusterReader<'_>> {
        self.cluster_in_block(self.blocks.block_of(row))
    }
    /// Preserves a cluster address already established by the scan.
    pub(crate) fn cluster_in_block(
        &self,
        block: usize,
    ) -> crate::Result<QuantizedClusterReader<'_>> {
        Ok(QuantizedClusterReader {
            layer: self,
            block,
            rows: self.blocks.block_rows[block]..self.blocks.block_rows[block + 1],
            codes: self.blocks.column(block, self.codes)?,
        })
    }
    #[cfg(test)]
    pub(crate) fn read_column(&self, idx: usize, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        self.blocks.read_column(idx, rows)
    }
    pub(crate) fn code_bytes(&self, row: usize) -> crate::Result<OwnedBytes> {
        self.read_codes(row..row + 1)
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
        Ok(self
            .read_constants(row..row + 1)?
            .map(|b| f32::from_le_bytes(b.as_slice().try_into().unwrap())))
    }
}
/// Block-local code geometry and sidecar addressing for one selected cluster.
pub(crate) struct QuantizedClusterReader<'a> {
    layer: &'a QuantizedLayerReader,
    block: usize,
    pub(crate) rows: Range<usize>,
    codes: FileSlice,
}

impl QuantizedClusterReader<'_> {
    /// Emits code ranges in row order into reusable scratch, with overlap-only page grouping.
    pub(crate) fn plan_codes(&self, rows: &[usize], ranges: &mut Vec<Range<usize>>) {
        ranges.clear();
        QuantizedLayerReader::plan_code_slot_reads(
            &self.codes,
            self.layer.code_stride,
            self.rows.start,
            rows,
            ranges,
        );
    }

    /// Reads an ordered code range directly; code padding is checked before scoring.
    pub(crate) fn read_codes(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        let stride = self.layer.code_stride;
        let bytes = self
            .codes
            .slice((rows.start - self.rows.start) * stride..(rows.end - self.rows.start) * stride)
            .read_vector_bytes()?;
        self.layer.validate_codes(&bytes, &rows)?;
        Ok(bytes)
    }

    /// Pins the complete sidecar span once, from scales through errors or L2 constants.
    /// Small columns are contiguous within a band; reading the span avoids per-column plans
    /// and request sorting. Codes stay separate because multi-page requests copy in paged storage.
    pub(crate) fn read_sidecar(&self) -> crate::Result<QuantizedSidecarBatch> {
        use crate::vector::storage::blocks::column_range;
        let layer = self.layer;
        let blocks = &layer.blocks;
        let n = blocks.rows_in(self.block);
        let first = column_range(&blocks.slots, n, layer.scales).start;
        let bytes = self.sidecar_slice()?.read_vector_bytes()?;
        let view = |idx| {
            let column = column_range(&blocks.slots, n, idx);
            bytes.slice(column.start - first..column.end - first)
        };
        Ok(QuantizedSidecarBatch {
            scales: view(layer.scales),
            gammas: view(layer.gammas),
            error_ratios: view(layer.errors),
            constants: layer.constants.map(view),
            rows: self.rows.clone(),
        })
    }

    /// The sidecar span: scales through error ratios, or through L2 constants.
    fn sidecar_slice(&self) -> crate::Result<FileSlice> {
        use crate::vector::storage::blocks::column_range;
        let layer = self.layer;
        let blocks = &layer.blocks;
        let n = blocks.rows_in(self.block);
        let first = column_range(&blocks.slots, n, layer.scales).start;
        let last = column_range(&blocks.slots, n, layer.constants.unwrap_or(layer.errors)).end;
        blocks.block_slice(self.block, first..last)
    }

    /// The cluster's residual-norms column, stored in layer 0's band.
    fn norms_slice(&self) -> crate::Result<FileSlice> {
        let blocks = &self.layer.blocks;
        let idx = blocks
            .slots
            .iter()
            .position(|slot| matches!(slot.slot_type, SlotType::ResidualNorms))
            .ok_or_else(|| DataCorruption::comment_only("missing residual norms column"))?;
        blocks.column(self.block, idx)
    }

    /// Reads the cluster's residual norms in one request.
    pub(crate) fn read_norms(&self) -> crate::Result<OwnedBytes> {
        Ok(self.norms_slice()?.read_vector_bytes()?)
    }
}

/// Splits increasing rows by block before consulting storage geometry.
#[cfg(test)]
fn plan_block_column(
    blocks: &Blocks,
    idx: usize,
    available: Range<usize>,
    rows: &[usize],
    ranges: &mut Vec<Range<usize>>,
    scratch: &mut Vec<(usize, usize)>,
) -> crate::Result<()> {
    ranges.clear();
    let mut selected = rows;
    while let Some(&row) = selected.first() {
        let b = blocks.block_of(row);
        let first = blocks.block_rows[b];
        let end = blocks.block_rows[b + 1];
        let count = selected.partition_point(|&r| r < end);
        if matches!(
            blocks.slots[idx].slot_type,
            SlotType::QuantLayerCodes { .. }
        ) {
            QuantizedLayerReader::plan_code_slot_reads(
                &blocks.column(b, idx)?,
                blocks.slots[idx].stride as usize,
                first,
                &selected[..count],
                ranges,
            );
        } else {
            QuantizedLayerReader::plan_slot_reads(
                &blocks.column(b, idx)?,
                blocks.slots[idx].stride as usize,
                first,
                available.start.max(first)..available.end.min(end),
                &selected[..count],
                ranges,
                scratch,
            );
        }
        selected = &selected[count..];
    }
    Ok(())
}

/// Stored layer readers and shared query-preparation metadata.
pub(crate) struct QuantizedFieldReader {
    pub(crate) index_ctx: Arc<QuantizedIndexCtx>,
    pub(crate) layers: Vec<QuantizedLayerReader>,
    pub(crate) blocks: Arc<Blocks>,
    pub(crate) norms: usize,
}

impl QuantizedFieldReader {
    pub(crate) fn layers(&self) -> &[QuantizedLayerReader] {
        &self.layers
    }
    pub(crate) fn residual_norm(&self, row: usize) -> crate::Result<f32> {
        let bytes = self.read_residual_norms(row..row + 1)?;
        Ok(f32::from_le_bytes(bytes.as_slice().try_into().unwrap()))
    }
    pub(crate) fn read_residual_norms(&self, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        self.blocks.read_column(self.norms, rows)
    }
    pub(crate) fn index_ctx(&self) -> &Arc<QuantizedIndexCtx> {
        &self.index_ctx
    }
}

// --- Query preparation inputs ---

/// Immutable quantization state for one segment's field, built when the
/// segment's vector reader opens.
pub(crate) struct QuantizedIndexCtx {
    pub(crate) meta: Arc<VectorColMetadata>,
    pub(crate) specs: Vec<LayerSpec>,
    pub(crate) grids: Vec<Grid>,
    pub(crate) rotation_plan: QueryRotationPlan,
}

impl QuantizedIndexCtx {
    /// Builds query inputs only from the immutable segment metadata.
    pub(crate) fn new(meta: Arc<VectorColMetadata>) -> crate::Result<Self> {
        if !matches!(meta.as_ref(), VectorColMetadata::Quantized { .. }) {
            return Err(TantivyError::InvalidArgument(
                "prepared queries require quantized metadata".into(),
            ));
        }
        let (specs, grids) = meta.runtime();
        let rotation_plan = QueryRotationPlan::new(meta.field().dim as usize, &specs);
        Ok(Self {
            meta,
            specs,
            grids,
            rotation_plan,
        })
    }
    #[cfg(test)]
    pub(crate) fn from_config(config: VectorQuantizationConfig) -> crate::Result<Self> {
        let opts = crate::schema::VectorOptions::new(config.dim, config.metric);
        Self::new(Arc::new(VectorColMetadata::build_ivf(
            &opts,
            Some(&config),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use quant::model::f16::f32_to_f16;

    use super::*;
    use crate::schema::{Metric, VectorOptions};
    use crate::vector::quantization::{quantized_code_stride, VectorQuantizationConfig};
    use crate::vector::storage::io::test_support::{BlockTrackedBytes, TrackedReads};

    fn test_layer(
        codes: FileSlice,
        sidecar: FileSlice,
        constants: Option<FileSlice>,
        offsets: &[usize],
        dim: usize,
        bits: u8,
    ) -> QuantizedLayerReader {
        test_layer_tracked(codes, sidecar, constants, offsets, dim, bits, None)
    }
    fn test_layer_tracked(
        codes: FileSlice,
        sidecar: FileSlice,
        constants: Option<FileSlice>,
        offsets: &[usize],
        dim: usize,
        bits: u8,
        tracking: Option<(TrackedReads, usize)>,
    ) -> QuantizedLayerReader {
        use crate::vector::storage::blocks::{block_len, column_range, pad, write_metadata};
        let metric = if constants.is_some() {
            Metric::L2
        } else {
            Metric::Dot
        };
        let opts = VectorOptions::new(dim, metric);
        let config = VectorQuantizationConfig::materialize(
            "v".into(),
            &opts,
            vec![crate::vector::VectorQuantizationLayer { bits, seed: 0 }],
        )
        .unwrap();
        let meta = VectorColMetadata::build_ivf(&opts, Some(&config)).unwrap();
        let slots = meta.slots();
        let mut out = Vec::new();
        write_metadata(&mut out, &meta).unwrap();
        let mut directory = crate::vector::storage::blocks::BlockDirectory::new(out.len() as u64);
        let codes = codes.read_bytes().unwrap();
        let sidecar = sidecar.read_bytes().unwrap();
        let constants = constants.map(|s| s.read_bytes().unwrap());
        let stride = quantized_code_stride(dim, bits);
        for rows in offsets.windows(2) {
            let n = rows[1] - rows[0];
            let start = out.len();
            for (idx, slot) in slots.iter().enumerate() {
                let range = column_range(&slots, n, idx);
                let padding = start + range.start - out.len();
                pad(&mut out, padding).unwrap();
                let bytes: &[u8] = match &slot.slot_type {
                    SlotType::QuantLayerCodes { .. } if !codes.is_empty() => {
                        &codes[rows[0] * stride..rows[1] * stride]
                    }
                    SlotType::QuantLayerScales { .. } if !sidecar.is_empty() => {
                        &sidecar[rows[0] * 8..rows[0] * 8 + n * 4]
                    }
                    SlotType::QuantLayerGammas { .. } if !sidecar.is_empty() => {
                        &sidecar[rows[0] * 8 + n * 4..rows[0] * 8 + n * 6]
                    }
                    SlotType::QuantLayerErrors { .. } if !sidecar.is_empty() => {
                        &sidecar[rows[0] * 8 + n * 6..rows[1] * 8]
                    }
                    SlotType::QuantLayerConstants { .. } => {
                        &constants.as_ref().unwrap()[rows[0] * 4..rows[1] * 4]
                    }
                    _ => &[],
                };
                if matches!(slot.slot_type, SlotType::DocIds) {
                    for row in rows[0]..rows[1] {
                        out.extend_from_slice(&(row as u32).to_le_bytes());
                    }
                } else if bytes.is_empty() {
                    pad(&mut out, range.len()).unwrap();
                } else {
                    out.extend_from_slice(bytes);
                }
            }
            let padding = start + block_len(&slots, n) - out.len();
            pad(&mut out, padding).unwrap();
            directory.push(out.len() as u64, rows[1] as u32);
        }
        directory.finish(&mut out).unwrap();
        let entry = if let Some((reads, block_len)) = tracking {
            FileSlice::new(Arc::new(BlockTrackedBytes {
                bytes: out,
                reads,
                block_len,
            }))
        } else {
            FileSlice::from(out)
        };
        let blocks = Arc::new(
            Blocks::open(
                entry,
                &opts,
                *offsets.last().unwrap(),
                Some(offsets.to_vec()),
            )
            .unwrap(),
        );
        QuantizedLayerReader {
            blocks,
            layer: 0,
            codes: 3,
            scales: 4,
            gammas: 5,
            errors: 6,
            constants: constants.map(|_| 7),
            code_stride: stride,
            dim,
            bits,
        }
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

    // Rows and layer bands each pin their own columns with one request.
    #[test]
    fn rows_and_layer_spans_read_once_per_block() -> crate::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let layer = test_layer_tracked(
            FileSlice::from(Vec::new()),
            FileSlice::from(Vec::new()),
            None,
            &[0, 3],
            64,
            1,
            Some((Arc::clone(&reads), 32)),
        );
        reads.lock().unwrap().clear();
        let rows = layer.blocks.read_column(0, 0..3)?;
        assert_eq!(rows.len(), 3 * 64 * 4);
        let start = layer.blocks.block_start(0);
        assert_eq!(&*reads.lock().unwrap(), &[start..start + rows.len()]);
        reads.lock().unwrap().clear();
        layer.read_batch(0..3)?;
        assert_eq!(&*reads.lock().unwrap(), &[layer.blocks.layer_span(0, 0)]);
        Ok(())
    }

    #[test]
    fn quantized_layer_reader_rejects_non_zero_tail() -> crate::Result<()> {
        let mut codes = vec![0_u8; quantized_code_stride(65, 1)];
        codes[8] = 1;
        let valid = test_layer(
            FileSlice::from(codes.clone()),
            FileSlice::empty(),
            None,
            &[0, 1],
            65,
            1,
        );
        assert_eq!(valid.code_bytes(0)?.as_slice(), codes);

        codes[8] |= 2;
        let corrupt = test_layer(
            FileSlice::from(codes.clone()),
            FileSlice::empty(),
            None,
            &[0, 1],
            65,
            1,
        );
        assert!(corrupt.code_bytes(0).is_err());
        Ok(())
    }

    /// Without storage geometry, the plan reads only selected runs, so an unselected
    /// row is neither read nor validated; corruption there is caught by whichever
    /// query selects it.
    #[test]
    fn no_geometry_plan_skips_unselected_gap_rows_and_validates_read_rows() {
        let stride = quantized_code_stride(65, 1);
        let mut codes = vec![0_u8; stride * 3];
        codes[8] = 1;
        codes[stride + 8] = 0b10;
        codes[stride * 2 + 8] = 1;
        let reader = test_layer(
            FileSlice::from(codes),
            FileSlice::from(test_sidecar(&[0.0; 3], &[1.0; 3])),
            None,
            &[0, 3],
            65,
            1,
        );
        let mut ranges = Vec::new();
        let mut blocks = Vec::new();
        reader.plan_code_reads(0..3, &[0, 2], &mut ranges, &mut blocks);
        assert_eq!(ranges, [0..1, 2..3]);
        assert!(reader.read_codes(0..1).is_ok());
        assert!(reader.read_codes(2..3).is_ok());
        assert!(reader.read_codes(0..3).is_err());
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
        let reader = test_layer(
            FileSlice::from(codes.clone()),
            FileSlice::from(sidecar),
            Some(FileSlice::from(constants)),
            &[0, 2],
            65,
            1,
        );
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
        let corrupt = test_layer(
            FileSlice::from(codes),
            FileSlice::from(test_sidecar(&[0.0, 0.0], &[1.0, 0.5])),
            Some(FileSlice::from(vec![0_u8; 8])),
            &[0, 2],
            65,
            1,
        );
        assert!(corrupt.read_batch(0..2).is_err());
        Ok(())
    }

    #[test]
    fn cluster_blocked_sidecar_maps_scales_and_gammas_by_cluster() -> crate::Result<()> {
        let mut sidecar = test_sidecar(&[11.0, 12.0], &[1.0, 2.0]);
        sidecar.extend(test_sidecar(&[13.0], &[3.0]));
        let reader = test_layer(
            FileSlice::empty(),
            FileSlice::from(sidecar),
            None,
            &[0, 2, 3],
            64,
            1,
        );

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
            assert!(
                validate_decoded_sidecar(&[gamma], &[0.0], 0).is_err(),
                "gamma={gamma}"
            );
        }
        assert!(validate_decoded_sidecar(&[1.0, 4.0], &[0.0, 0.0], 0).is_ok());
    }

    #[test]
    fn sidecar_error_ratio_validation_is_range_scoped() {
        for error_ratio in [-0.5, f32::INFINITY, f32::NAN] {
            assert!(
                validate_decoded_sidecar(&[1.0], &[error_ratio], 0).is_err(),
                "error_ratio={error_ratio}"
            );
        }
        assert!(validate_decoded_sidecar(&[1.0, 1.0], &[0.0, 0.5], 0).is_ok());
    }

    // A whole band requires one read; sparse columns plan independently without crossing blocks.
    #[test]
    fn bands_and_sparse_columns_preserve_storage_geometry() -> crate::Result<()> {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let reader = test_layer_tracked(
            FileSlice::from(vec![0; 12 * 8]),
            FileSlice::from(
                [
                    test_sidecar(&[0.0; 8], &[1.0; 8]),
                    test_sidecar(&[0.0; 4], &[1.0; 4]),
                ]
                .concat(),
            ),
            None,
            &[0, 8, 8, 12],
            64,
            1,
            Some((Arc::clone(&reads), 16)),
        );
        reads.lock().unwrap().clear();
        let (batch, totals) = crate::vector::telemetry::measure(|| {
            let _stage = crate::vector::telemetry::enter_vector_stage(
                crate::vector::VectorStage::LayerScan(0),
            );
            reader.read_batch(0..8)
        });
        let batch = batch?;
        assert_eq!(reads.lock().unwrap().len(), 1);
        let io = totals.io(crate::vector::VectorStage::LayerScan(0));
        let range = reads.lock().unwrap()[0].clone();
        assert_eq!(io.reads, 1);
        assert_eq!(io.bytes_read, range.len() as u64);
        assert_eq!(
            io.storage_blocks,
            ((range.end - 1) / 16 - range.start / 16 + 1) as u64
        );
        assert!(batch.residual_norms.is_some());
        let sidecar = reader.read_sidecar(0..8)?;
        assert_eq!(batch.scales(), sidecar.scales());
        assert_eq!(batch.gammas(), sidecar.gammas());
        assert_eq!(batch.error_ratios(), sidecar.error_ratios());
        let mut ranges = Vec::new();
        let mut scratch = Vec::new();
        for column in [reader.codes, reader.scales, reader.gammas, reader.errors] {
            reader.plan_column_reads(column, 0..12, &[0, 7, 8, 11], &mut ranges, &mut scratch);
            assert!(ranges.iter().all(|r| r.end <= 8 || r.start >= 8));
            for r in ranges.clone() {
                reader.read_column(column, r)?;
            }
        }
        for column in [reader.codes, reader.scales, reader.gammas, reader.errors] {
            assert!(reader.read_column(column, 7..9).is_err());
        }
        Ok(())
    }

    #[test]
    fn code_page_groups_require_overlap_and_trim_selected_rows() {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let slot = FileSlice::new(Arc::new(BlockTrackedBytes {
            bytes: vec![0; 160],
            reads,
            block_len: 16,
        }));
        let mut ranges = Vec::new();
        // Adjacent disjoint pages 0 and 1 stay separate, as does page 4.
        QuantizedLayerReader::plan_code_slot_reads(&slot, 4, 100, &[101, 106, 118], &mut ranges);
        assert_eq!(ranges, [101..102, 106..107, 118..119]);
        ranges.clear();
        // Row 2 straddles pages 0/1, connecting row 0 to row 4; page 2 stays separate.
        QuantizedLayerReader::plan_code_slot_reads(&slot, 6, 0, &[0, 2, 4, 6], &mut ranges);
        assert_eq!(ranges, [0..5, 6..7]);
        ranges.clear();
        let unknown = FileSlice::from(vec![0; 160]);
        QuantizedLayerReader::plan_code_slot_reads(&unknown, 4, 0, &[1, 2, 6], &mut ranges);
        assert_eq!(ranges, [1..3, 6..7]);
    }

    #[test]
    fn sparse_cluster_reads_pin_one_sidecar_span() -> crate::Result<()> {
        use crate::vector::storage::blocks::{
            block_len, column_range, write_metadata, BlockDirectory,
        };
        use crate::vector::telemetry::{enter_vector_stage, measure};
        use crate::vector::{VectorQuantizationLayer, VectorStage};
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let opts = VectorOptions::new(1024, metric);
            let config = VectorQuantizationConfig::materialize(
                "v".into(),
                &opts,
                vec![
                    VectorQuantizationLayer { bits: 1, seed: 7 },
                    VectorQuantizationLayer { bits: 4, seed: 11 },
                ],
            )?;
            let meta = VectorColMetadata::build_ivf(&opts, Some(&config))?;
            let slots = meta.slots();
            let mut data = Vec::new();
            write_metadata(&mut data, &meta)?;
            let mut directory = BlockDirectory::new(data.len() as u64);
            let boundaries = [0, 59, 59, 118];
            for rows in boundaries.windows(2) {
                let n = rows[1] - rows[0];
                let start = data.len();
                data.resize(start + block_len(&slots, n), 0);
                for (idx, slot) in slots.iter().enumerate() {
                    let range = column_range(&slots, n, idx);
                    match slot.slot_type {
                        SlotType::QuantLayerScales { .. }
                        | SlotType::QuantLayerConstants { .. } => {
                            for (row, bytes) in data[start + range.start..start + range.end]
                                .chunks_exact_mut(4)
                                .enumerate()
                            {
                                bytes.copy_from_slice(
                                    &((rows[0] + row + idx) as f32 / 16.0).to_le_bytes(),
                                );
                            }
                        }
                        SlotType::QuantLayerGammas { .. } | SlotType::QuantLayerErrors { .. } => {
                            for (row, bytes) in data[start + range.start..start + range.end]
                                .chunks_exact_mut(2)
                                .enumerate()
                            {
                                bytes.copy_from_slice(
                                    &f32_to_f16((rows[0] + row + idx) as f32 / 256.0).to_le_bytes(),
                                );
                            }
                        }
                        _ => {}
                    }
                }
                directory.push(data.len() as u64, rows[1] as u32);
            }
            directory.finish(&mut data)?;
            for block_len in [0, 16, 8160] {
                for prefix in [0, 3, 8171] {
                    let reads = Arc::new(Mutex::new(Vec::new()));
                    let mut parent = vec![0; prefix];
                    parent.extend_from_slice(&data);
                    let entry = if block_len == 0 {
                        FileSlice::from(parent).slice_from(prefix)
                    } else {
                        FileSlice::new(Arc::new(BlockTrackedBytes {
                            bytes: parent,
                            reads,
                            block_len,
                        }))
                        .slice_from(prefix)
                    };
                    let blocks =
                        Arc::new(Blocks::open(entry, &opts, 118, Some(boundaries.to_vec()))?);
                    let codes = slots
                        .iter()
                        .position(|s| {
                            matches!(s.slot_type, SlotType::QuantLayerCodes { layer: 1, .. })
                        })
                        .unwrap();
                    let reader = QuantizedLayerReader {
                        blocks,
                        layer: 1,
                        codes,
                        scales: codes + 1,
                        gammas: codes + 2,
                        errors: codes + 3,
                        constants: (metric == Metric::L2).then_some(codes + 4),
                        code_stride: quantized_code_stride(1024, 4),
                        dim: 1024,
                        bits: 4,
                    };
                    for selected in [vec![0, 58], vec![0, 58, 59, 117], vec![5, 6, 31, 54]] {
                        let mut ranges = Vec::new();
                        let mut start = 0;
                        while start < selected.len() {
                            let cluster = reader.cluster(selected[start])?;
                            let end = start
                                + selected[start..].partition_point(|&r| r < cluster.rows.end);
                            cluster.plan_codes(&selected[start..end], &mut ranges);
                            let mut expected = Vec::new();
                            reader.plan_column_reads(
                                reader.codes,
                                0..118,
                                &selected[start..end],
                                &mut expected,
                                &mut Vec::new(),
                            );
                            assert_eq!(ranges, expected);
                            let (codes, code_totals) = measure(|| {
                                let _stage = enter_vector_stage(VectorStage::LayerScan(1));
                                ranges
                                    .iter()
                                    .try_for_each(|rows| cluster.read_codes(rows.clone()).map(drop))
                            });
                            codes?;
                            let code_io = code_totals.io(VectorStage::LayerScan(1));
                            let (sidecar, sidecar_totals) = measure(|| {
                                let _stage = enter_vector_stage(VectorStage::LayerScan(1));
                                cluster.read_sidecar()
                            });
                            let sidecar = sidecar?;
                            let sidecar_io = sidecar_totals.io(VectorStage::LayerScan(1));
                            assert_eq!(sidecar_io.reads, 1);
                            let first =
                                column_range(&slots, cluster.rows.len(), reader.scales).start;
                            let last = column_range(
                                &slots,
                                cluster.rows.len(),
                                reader.constants.unwrap_or(reader.errors),
                            )
                            .end;
                            assert_eq!(sidecar_io.bytes_read, (last - first) as u64);
                            let row_range = cluster.rows.clone();
                            assert_eq!(
                                sidecar.scales,
                                reader.read_column(reader.scales, row_range.clone())?
                            );
                            assert_eq!(
                                sidecar.gammas,
                                reader.read_column(reader.gammas, row_range.clone())?
                            );
                            assert_eq!(
                                sidecar.error_ratios,
                                reader.read_column(reader.errors, row_range.clone())?
                            );
                            assert_eq!(sidecar.constants, reader.read_constants(row_range)?);
                            let n = end - start;
                            let (mut scales, mut gammas, mut errors) =
                                (vec![0.; n], vec![0.; n], vec![0.; n]);
                            let mut constants = vec![0.; if metric == Metric::L2 { n } else { 0 }];
                            sidecar.decode_selected(
                                &selected[start..end],
                                &mut scales,
                                &mut gammas,
                                &mut errors,
                                &mut constants,
                            )?;
                            for (i, &row) in selected[start..end].iter().enumerate() {
                                assert_eq!(scales[i].to_bits(), reader.scale(row)?.to_bits());
                                assert_eq!(gammas[i].to_bits(), reader.gamma(row)?.to_bits());
                                assert_eq!(errors[i].to_bits(), reader.error_ratio(row)?.to_bits());
                                if metric == Metric::L2 {
                                    assert_eq!(
                                        constants[i].to_bits(),
                                        reader.constant(row)?.unwrap().to_bits()
                                    );
                                }
                            }
                            if block_len == 8160 && prefix == 0 && selected == [0, 58] {
                                assert_eq!(code_io.reads + sidecar_io.reads, 3);
                                assert_eq!(code_io.storage_blocks + sidecar_io.storage_blocks, 3);
                            }
                            start = end;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Exact reads cost the distinct storage blocks the selected rows touch, a whole cluster its
    /// span; without block geometry, bytes.
    #[test]
    fn exact_read_cost_counts_distinct_blocks_then_bytes() {
        // 100-byte blocks, 40-byte rows from offset 0: rows 0-1 lie in block 0, row 2 straddles
        // blocks 0-1, rows 3-4 lie in block 1, row 5 in block 2, row 7 straddles blocks 2-3.
        let file = FileSlice::new(Arc::new(BlockTrackedBytes {
            bytes: vec![0; 1000],
            reads: Default::default(),
            block_len: 100,
        }));
        let column = file.slice(0..400);
        let cost = |offsets: Option<&[u32]>| exact_read_cost(&column, 40, offsets, 10).0;
        assert_eq!(cost(Some(&[0, 1])), 1);
        assert_eq!(cost(Some(&[0, 3])), 2);
        assert_eq!(cost(Some(&[2])), 2, "a straddling row counts both blocks");
        assert_eq!(
            cost(Some(&[1, 2, 3])),
            2,
            "blocks shared by rows count once"
        );
        assert_eq!(cost(Some(&[0, 3, 5, 7])), 4);
        assert_eq!(cost(None), 4, "all rows span blocks 0-3");

        let plain = FileSlice::from(vec![0u8; 1000]);
        let column = plain.slice(0..400);
        let cost = |offsets: Option<&[u32]>| exact_read_cost(&column, 40, offsets, 10).0;
        assert_eq!(cost(Some(&[0, 9])), 80);
        assert_eq!(cost(None), 400);
    }
}
