//! Block addressing and entry framing. See FORMAT.md for the column contract.
use std::io::{self, Write};
use std::ops::Range;
use std::sync::Arc;

use common::{HasLen, OwnedBytes};

use super::metadata::{Partition, Slot, VectorColMetadata};
use super::storage_io::VectorRead;
use super::ENTRY_ALIGN;
use crate::directory::FileSlice;
use crate::error::DataCorruption;
use crate::schema::VectorOptions;

/// Rounds a validated file offset up to a power-of-two alignment.
pub(crate) fn align_up(pos: usize, align: usize) -> usize {
    (pos + align - 1) & !(align - 1)
}
/// Column bytes relative to a block; empty blocks consume no bytes.
pub(crate) fn column_range(slots: &[Slot], n: usize, idx: usize) -> Range<usize> {
    let mut pos = 0;
    for (i, slot) in slots.iter().enumerate() {
        pos = align_up(pos, slot.type_bytes());
        let end = pos + n * slot.stride as usize;
        if i == idx {
            return pos..end;
        }
        pos = end;
    }
    panic!("column index out of bounds")
}
/// Total aligned block length, derived from the format's ordered column list.
pub(crate) fn block_len(slots: &[Slot], n: usize) -> usize {
    align_up(
        column_range(slots, n, slots.len() - 1).end,
        block_align(slots),
    )
}
/// Maximum decoder element width in a field, also its block alignment.
pub(crate) fn block_align(slots: &[Slot]) -> usize {
    slots.iter().map(Slot::type_bytes).max().expect("Rows slot")
}
/// Logical Data entry length, including its directory and terminal block count.
pub(crate) fn data_entry_len(blocks_end: usize, num_blocks: usize) -> usize {
    let directory_start = align_up(blocks_end, ENTRY_ALIGN);
    align_up(directory_start + (num_blocks + 1) * 12, ENTRY_ALIGN) + 8
}
/// Records actual entry-relative positions while blocks stream to storage.
pub(crate) struct BlockDirectory {
    byte_starts: Vec<u64>,
    row_starts: Vec<u32>,
}
impl BlockDirectory {
    pub(crate) fn new(first: u64) -> Self {
        Self {
            byte_starts: vec![first],
            row_starts: vec![0],
        }
    }
    pub(crate) fn push(&mut self, end: u64, rows: u32) {
        self.byte_starts.push(end);
        self.row_starts.push(rows);
    }
    /// Aligns the directory, writes both arrays, then pads the terminal count to eight bytes.
    pub(crate) fn finish(mut self, writer: &mut impl Write) -> io::Result<usize> {
        let blocks_end = *self.byte_starts.last().unwrap() as usize;
        let directory_start = align_up(blocks_end, ENTRY_ALIGN);
        pad(writer, directory_start - blocks_end)?;
        *self.byte_starts.last_mut().unwrap() = directory_start as u64;
        for offset in &self.byte_starts {
            writer.write_all(&offset.to_le_bytes())?;
        }
        for row in &self.row_starts {
            writer.write_all(&row.to_le_bytes())?;
        }
        let num_blocks = self.row_starts.len() - 1;
        let arrays_end = directory_start + self.row_starts.len() * 12;
        let len = data_entry_len(blocks_end, num_blocks);
        pad(writer, len - 8 - arrays_end)?;
        writer.write_all(&(num_blocks as u64).to_le_bytes())?;
        assert_eq!(len % ENTRY_ALIGN, 0);
        Ok(len)
    }
}
/// Writes zero padding without allocating in proportion to the alignment.
pub(crate) fn pad(writer: &mut impl Write, bytes: usize) -> io::Result<()> {
    let zeroes = [0; 4096];
    let mut remaining = bytes;
    while remaining > 0 {
        let n = remaining.min(zeroes.len());
        writer.write_all(&zeroes[..n])?;
        remaining -= n;
    }
    Ok(())
}
/// Writes the length-prefixed metadata and returns the first block's relative offset.
pub(crate) fn write_metadata(
    writer: &mut impl Write,
    meta: &VectorColMetadata,
) -> io::Result<usize> {
    let bytes = meta.to_bytes();
    writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&bytes)?;
    let start = align_up(4 + bytes.len(), block_align(&meta.slots()));
    pad(writer, start - 4 - bytes.len())?;
    Ok(start)
}
/// Validated field metadata and the deferred Data entry.
#[derive(Clone)]
pub(crate) struct BlockMetadata {
    entry: FileSlice,
    pub(crate) meta: Arc<VectorColMetadata>,
    metadata_end: usize,
}
impl BlockMetadata {
    pub(crate) fn open(
        entry: FileSlice,
        opts: &VectorOptions,
        clustered: bool,
    ) -> crate::Result<Self> {
        if entry.len() < 4 {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only("invalid vector blocks: missing metadata length"),
            ));
        }
        let len = u32::from_le_bytes(
            entry
                .slice_to(4)
                .read_bytes()?
                .as_slice()
                .try_into()
                .unwrap(),
        ) as usize;
        if len > entry.len() - 4 {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only("invalid vector blocks: truncated metadata"),
            ));
        }
        let meta = Arc::new(VectorColMetadata::from_bytes(
            &entry.slice(4..4 + len).read_bytes()?,
        )?);
        meta.validate(opts, clustered)?;
        Ok(Self {
            entry,
            meta,
            metadata_end: 4 + len,
        })
    }
}
/// Data entry plus its validated stored directory. Per-column offsets are recomputed, never cached
/// per block.
pub(crate) struct Blocks {
    entry: FileSlice,
    pub(crate) meta: Arc<VectorColMetadata>,
    pub(crate) slots: Vec<Slot>,
    bands: Vec<Range<usize>>,
    pub(crate) block_rows: Arc<[usize]>,
    pub(crate) block_starts: Arc<[u64]>,
}
impl Blocks {
    /// Opens metadata and validates the stored directory against the field row counts.
    pub(crate) fn open(
        entry: FileSlice,
        opts: &VectorOptions,
        num_rows: usize,
        clusters: Option<Vec<usize>>,
    ) -> crate::Result<Self> {
        Self::from_metadata(
            BlockMetadata::open(entry, opts, clusters.is_some())?,
            num_rows,
            clusters,
        )
    }

    /// Validates row geometry before exposing any column bytes.
    pub(crate) fn from_metadata(
        metadata: BlockMetadata,
        num_rows: usize,
        clusters: Option<Vec<usize>>,
    ) -> crate::Result<Self> {
        let BlockMetadata {
            entry,
            meta,
            metadata_end,
        } = metadata;
        if entry.len() < 8 || entry.len() % ENTRY_ALIGN != 0 {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only(
                    "invalid vector blocks: missing or misaligned directory footer",
                ),
            ));
        }
        let footer = entry.len() - 8;
        let count = u64::from_le_bytes(
            entry
                .slice_from(footer)
                .read_bytes()?
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let count = usize::try_from(count).map_err(|_| {
            crate::TantivyError::DataCorruption(DataCorruption::comment_only(
                "invalid vector blocks: block count overflow",
            ))
        })?;
        let entries = count.checked_add(1).ok_or_else(|| {
            crate::TantivyError::DataCorruption(DataCorruption::comment_only(
                "invalid vector blocks: block count overflow",
            ))
        })?;
        let array_bytes = entries.checked_mul(12).ok_or_else(|| {
            crate::TantivyError::DataCorruption(DataCorruption::comment_only(
                "invalid vector blocks: directory size overflow",
            ))
        })?;
        let row_bytes = entries.checked_mul(4).ok_or_else(|| {
            crate::TantivyError::DataCorruption(DataCorruption::comment_only(
                "invalid vector blocks: directory size overflow",
            ))
        })?;
        let padded_rows = row_bytes
            .checked_add(ENTRY_ALIGN - 1)
            .map(|n| n & !(ENTRY_ALIGN - 1))
            .ok_or_else(|| {
                crate::TantivyError::DataCorruption(DataCorruption::comment_only(
                    "invalid vector blocks: directory padding overflow",
                ))
            })?;
        let padded_bytes = entries
            .checked_mul(8)
            .and_then(|n| n.checked_add(padded_rows))
            .ok_or_else(|| {
                crate::TantivyError::DataCorruption(DataCorruption::comment_only(
                    "invalid vector blocks: directory size overflow",
                ))
            })?;
        let directory_start = footer
            .checked_sub(padded_bytes)
            .filter(|&start| start >= metadata_end)
            .ok_or_else(|| {
                crate::TantivyError::DataCorruption(DataCorruption::comment_only(
                    "invalid vector blocks: truncated directory",
                ))
            })?;
        let directory = entry.slice(directory_start..footer).read_bytes()?;
        let block_starts: Vec<u64> = directory[..entries * 8]
            .chunks_exact(8)
            .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let block_rows: Vec<usize> = directory[entries * 8..array_bytes]
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()) as usize)
            .collect();
        if directory[array_bytes..].iter().any(|&v| v != 0) {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only("invalid vector blocks: nonzero directory padding"),
            ));
        }
        let slots = meta.slots();
        let align = block_align(&slots);
        // With no blocks, the first boundary is the entry-aligned directory itself.
        if block_starts[0] < metadata_end as u64
            || block_starts[0] - metadata_end as u64
                >= if count == 0 { ENTRY_ALIGN } else { align } as u64
            || block_starts.last() != Some(&(directory_start as u64))
            || block_starts.iter().any(|&v| v % align as u64 != 0)
            || block_starts.windows(2).any(|v| v[0] > v[1])
        {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only("invalid vector blocks: invalid byte boundaries"),
            ));
        }
        if block_rows[0] != 0
            || block_rows.last() != Some(&num_rows)
            || block_rows.windows(2).any(|v| v[0] > v[1])
        {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only("invalid vector blocks: invalid row boundaries"),
            ));
        }
        match (&meta.field().partition, clusters) {
            (Partition::Clusters, Some(rows)) if block_rows == rows => {}
            (Partition::Uniform { rows_per_block }, None)
                if count == num_rows.div_ceil(*rows_per_block as usize)
                    && block_rows.iter().enumerate().all(|(b, &row)| {
                        row == b.saturating_mul(*rows_per_block as usize).min(num_rows)
                    }) => {}
            _ => {
                return Err(crate::TantivyError::DataCorruption(
                    DataCorruption::comment_only(
                        "invalid vector blocks: directory/partition row boundaries mismatch",
                    ),
                ))
            }
        }
        // Bound row arithmetic before any column layout is evaluated. Individual column
        // spans are checked against the stored next-block boundary before bytes are exposed.
        let row_bytes: u64 = slots.iter().map(|slot| u64::from(slot.stride)).sum();
        for (rows, bytes) in block_rows.windows(2).zip(block_starts.windows(2)) {
            if ((rows[1] - rows[0]) as u64)
                .checked_mul(row_bytes)
                .is_none_or(|size| size > bytes[1] - bytes[0])
            {
                return Err(crate::TantivyError::DataCorruption(
                    DataCorruption::comment_only("invalid vector blocks: truncated block"),
                ));
            }
        }
        let payload_end = if count == 0 {
            metadata_end
        } else {
            let last = count - 1;
            let column_end = column_range(
                &slots,
                block_rows[count] - block_rows[last],
                slots.len() - 1,
            )
            .end;
            (block_starts[last] as usize)
                .checked_add(column_end)
                .ok_or_else(|| {
                    crate::TantivyError::DataCorruption(DataCorruption::comment_only(
                        "invalid vector blocks: last column overflow",
                    ))
                })?
        };
        if payload_end > directory_start || directory_start - payload_end >= ENTRY_ALIGN {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only(
                    "invalid vector blocks: invalid block-area padding length",
                ),
            ));
        }
        if entry
            .slice(payload_end..directory_start)
            .read_bytes()?
            .iter()
            .any(|&v| v != 0)
        {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only("invalid vector blocks: nonzero block-area padding"),
            ));
        }
        let mut bands: Vec<Range<usize>> = Vec::new();
        for (i, slot) in slots.iter().enumerate() {
            if let Some(band) = slot.slot_type.band() {
                if bands.len() == band as usize {
                    bands.push(i..i + 1);
                } else {
                    bands[band as usize].end = i + 1;
                }
            }
        }
        Ok(Self {
            entry,
            meta,
            slots,
            bands,
            block_rows: block_rows.into(),
            block_starts: block_starts.into(),
        })
    }
    /// Relative start of a block, including the metadata prefix.
    pub(crate) fn block_start(&self, b: usize) -> usize {
        self.block_starts[b] as usize
    }
    /// Number of posting rows in one block; empty clusters have zero rows.
    pub(crate) fn rows_in(&self, b: usize) -> usize {
        self.block_rows[b + 1] - self.block_rows[b]
    }
    /// Finds a nonempty block containing an in-range row, skipping empty clusters.
    pub(crate) fn block_of(&self, row: usize) -> usize {
        assert!(row < *self.block_rows.last().unwrap());
        match self.meta.field().partition {
            Partition::Uniform { rows_per_block } => row / rows_per_block as usize,
            Partition::Clusters => self.block_rows.partition_point(|&start| start <= row) - 1,
        }
    }
    /// Resolves one complete column as a file slice preserving parent storage geometry.
    pub(crate) fn column(&self, b: usize, idx: usize) -> crate::Result<FileSlice> {
        let range = column_range(&self.slots, self.rows_in(b), idx);
        self.block_slice(b, range)
    }
    /// Resolves a byte span relative to a validated block while preserving storage geometry.
    pub(crate) fn block_slice(&self, b: usize, range: Range<usize>) -> crate::Result<FileSlice> {
        let start = self.block_start(b);
        if range.start > range.end || range.end > self.block_start(b + 1) - start {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only(
                    "invalid vector blocks: column crosses stored block boundary",
                ),
            ));
        }
        Ok(self.entry.slice(start + range.start..start + range.end))
    }
    /// Rejects row spans crossing a block, including malformed and out-of-range spans.
    pub(crate) fn block_for_range(&self, rows: &Range<usize>) -> crate::Result<usize> {
        if rows.start >= rows.end || rows.end > *self.block_rows.last().unwrap() {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only("invalid vector blocks: invalid row range"),
            ));
        }
        let b = self.block_of(rows.start);
        if rows.end > self.block_rows[b + 1] {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only(
                    "invalid vector blocks: row range crosses block boundary",
                ),
            ));
        }
        Ok(b)
    }
    /// Checks a known block-local address without searching the row directory.
    pub(crate) fn check_block_rows(&self, b: usize, rows: &Range<usize>) -> crate::Result<()> {
        if b + 1 >= self.block_rows.len()
            || rows.start < self.block_rows[b]
            || rows.start >= rows.end
            || rows.end > self.block_rows[b + 1]
        {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only(
                    "invalid vector blocks: invalid block-local row range",
                ),
            ));
        }
        Ok(())
    }
    /// Reads a block-local row span from one column.
    pub(crate) fn read_column(&self, idx: usize, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        if rows.start == rows.end && rows.end <= *self.block_rows.last().unwrap() {
            return Ok(OwnedBytes::empty());
        }
        let b = self.block_for_range(&rows)?;
        let stride = self.slots[idx].stride as usize;
        let first = self.block_rows[b];
        Ok(self
            .column(b, idx)?
            .slice((rows.start - first) * stride..(rows.end - first) * stride)
            .read_vector_bytes()?)
    }
    /// Whether blocks carry a document-id column and use centroid row boundaries.
    pub(crate) fn clustered(&self) -> bool {
        matches!(self.meta.field().partition, Partition::Clusters)
    }
    /// Layer zero spans residual norms through its last sidecar; document ids are read separately.
    pub(crate) fn layer_span(&self, b: usize, layer: usize) -> Range<usize> {
        self.band_range(b, layer)
    }
    /// Range of one scan band within the Data entry; band zero includes residual norms.
    pub(crate) fn band_range(&self, b: usize, layer: usize) -> Range<usize> {
        let band = &self.bands[layer];
        let n = self.rows_in(b);
        let start = column_range(&self.slots, n, band.start).start;
        let end = column_range(&self.slots, n, band.end - 1).end;
        self.block_start(b) + start..self.block_start(b) + end
    }
    /// Resolves one scan band as a file slice preserving storage geometry.
    pub(crate) fn band_slice(&self, b: usize, layer: usize) -> crate::Result<FileSlice> {
        let range = self.layer_span(b, layer);
        let start = self.block_start(b);
        self.block_slice(b, range.start - start..range.end - start)
    }
    /// Pins a band with exactly one read. Column views borrow this span without a heap list.
    pub(crate) fn read_band(
        &self,
        b: usize,
        layer: usize,
    ) -> crate::Result<(Range<usize>, OwnedBytes)> {
        let range = self.layer_span(b, layer);
        let bytes = self.band_slice(b, layer)?.read_vector_bytes()?;
        Ok((range, bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;
    use crate::vector::{VectorQuantizationConfig, VectorQuantizationLayer};
    // Every flat boundary maps to the correct row, including empty and partial blocks.
    #[test]
    fn uniform_boundaries_and_rows() {
        let opts = VectorOptions::new(3, Metric::L2);
        for r in [7, super::super::metadata::FLAT_ROWS_PER_BLOCK as usize] {
            for n in [0, 1, r - 1, r, r + 1, 3 * r] {
                let mut meta = VectorColMetadata::build_flat(&opts);
                let VectorColMetadata::Plain(field) = &mut meta else {
                    unreachable!()
                };
                field.partition = Partition::Uniform {
                    rows_per_block: r as u32,
                };
                let mut bytes = Vec::new();
                write_metadata(&mut bytes, &meta).unwrap();
                let mut directory = BlockDirectory::new(bytes.len() as u64);
                for start in (0..n).step_by(r) {
                    let end = (start + r).min(n);
                    for row in start..end {
                        bytes.extend(vec![row as u8; 12]);
                    }
                    pad(
                        &mut bytes,
                        block_len(&meta.slots(), end - start) - (end - start) * 12,
                    )
                    .unwrap();
                    directory.push(bytes.len() as u64, end as u32);
                }
                directory.finish(&mut bytes).unwrap();
                let blocks = Blocks::open(FileSlice::from(bytes.clone()), &opts, n, None).unwrap();
                assert_eq!(
                    data_entry_len(
                        *blocks.block_starts.last().unwrap() as usize,
                        blocks.block_rows.len() - 1
                    ),
                    bytes.len()
                );
                for row in 0..n {
                    assert_eq!(blocks.block_of(row), row / r);
                    assert_eq!(
                        &*blocks.read_column(0, row..row + 1).unwrap(),
                        vec![row as u8; 12]
                    );
                }
                if n > r {
                    assert!(blocks.read_column(0, r - 1..r + 1).is_err());
                }
                bytes.push(0);
                assert!(Blocks::open(FileSlice::from(bytes), &opts, n, None).is_err());
            }
        }
    }
    // Footer framing, both arrays, and every padding region have corruption checks.
    #[test]
    fn directory_round_trip_and_corruption() {
        let opts = VectorOptions::new(3, Metric::L2);
        let meta = VectorColMetadata::build_flat(&opts);
        for n in [0, 1, 2] {
            let mut bytes = Vec::new();
            write_metadata(&mut bytes, &meta).unwrap();
            let first = bytes.len();
            let mut directory = BlockDirectory::new(first as u64);
            bytes.extend(vec![0; n * 12]);
            let block_end = bytes.len();
            if n != 0 {
                directory.push(block_end as u64, n as u32);
            }
            directory.finish(&mut bytes).unwrap();
            let blocks = Blocks::open(FileSlice::from(bytes.clone()), &opts, n, None).unwrap();
            let b = usize::from(n != 0);
            let start = align_up(block_end, ENTRY_ALIGN);
            assert_eq!(blocks.block_starts[b], start as u64);
            assert_eq!(blocks.block_rows[b], n);
            assert_eq!(
                u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap()),
                b as u64
            );
            let reject = |corrupt: Vec<u8>| {
                assert!(matches!(
                    Blocks::open(FileSlice::from(corrupt), &opts, n, None),
                    Err(crate::TantivyError::DataCorruption(_))
                ))
            };
            for offset in block_end..start {
                let mut corrupt = bytes.clone();
                corrupt[offset] = 1;
                reject(corrupt);
            }
            for offset in start + (b + 1) * 12..bytes.len() - 8 {
                let mut corrupt = bytes.clone();
                corrupt[offset] = 1;
                reject(corrupt);
            }
            let mut corrupt = bytes.clone();
            corrupt[start + b * 8] ^= 1;
            reject(corrupt);
            let mut corrupt = bytes.clone();
            corrupt[start + (b + 1) * 8 + b * 4] ^= 1;
            reject(corrupt);
            let mut corrupt = bytes.clone();
            let footer = corrupt.len() - 8;
            corrupt[footer..].fill(255);
            reject(corrupt);
            let mut corrupt = bytes.clone();
            corrupt.pop();
            reject(corrupt);
            if n != 0 {
                let mut corrupt = bytes.clone();
                corrupt[start..start + 8].copy_from_slice(&(start as u64 + 8).to_le_bytes());
                reject(corrupt);
                let mut corrupt = bytes.clone();
                corrupt[start..start + 8].copy_from_slice(&((first + 1) as u64).to_le_bytes());
                reject(corrupt);
            }
        }
    }

    // Band views agree with columns, alignment is absolute within blocks, and padding is zero.
    #[test]
    fn clustered_layout_bands_and_zero_padding() {
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            for schedule in [&[1][..], &[1, 4], &[1, 2, 4]] {
                let opts = VectorOptions::new(100, metric);
                let config = VectorQuantizationConfig::materialize(
                    "v".into(),
                    &opts,
                    schedule
                        .iter()
                        .map(|&bits| VectorQuantizationLayer { bits, seed: 9 })
                        .collect(),
                )
                .unwrap();
                let meta = VectorColMetadata::build_ivf(&opts, Some(&config)).unwrap();
                let slots = meta.slots();
                let mut bytes = Vec::new();
                write_metadata(&mut bytes, &meta).unwrap();
                let mut directory = BlockDirectory::new(bytes.len() as u64);
                let mut rows = 0;
                for n in [0, 3, 0, 5, 0] {
                    let start = bytes.len();
                    for idx in 0..slots.len() {
                        let col = column_range(&slots, n, idx);
                        assert_eq!(col.start % slots[idx].type_bytes(), 0);
                        let padding = start + col.start - bytes.len();
                        pad(&mut bytes, padding).unwrap();
                        bytes.extend(std::iter::repeat_n((idx + 1) as u8, col.len()));
                    }
                    let padding = start + block_len(&slots, n) - bytes.len();
                    pad(&mut bytes, padding).unwrap();
                    assert_eq!(bytes.len() - start, block_len(&slots, n));
                    rows += n as u32;
                    directory.push(bytes.len() as u64, rows);
                }
                directory.finish(&mut bytes).unwrap();
                assert!(Blocks::open(
                    FileSlice::from(bytes.clone()),
                    &opts,
                    8,
                    Some(vec![0, 0, 2, 3, 8, 8])
                )
                .is_err());
                let blocks = Blocks::open(
                    FileSlice::from(bytes.clone()),
                    &opts,
                    8,
                    Some(vec![0, 0, 3, 3, 8, 8]),
                )
                .unwrap();
                assert_eq!(blocks.block_of(0), 1);
                assert_eq!(blocks.block_of(3), 3);
                for b in [1, 3] {
                    let mut previous = blocks.block_start(b);
                    for idx in 0..slots.len() {
                        let col = column_range(&slots, blocks.rows_in(b), idx);
                        assert!(bytes[previous..blocks.block_start(b) + col.start]
                            .iter()
                            .all(|&v| v == 0));
                        previous = blocks.block_start(b) + col.end;
                    }
                    assert!(bytes[previous..blocks.block_starts[b + 1] as usize]
                        .iter()
                        .all(|&v| v == 0));
                    for l in 0..schedule.len() {
                        let (span, pinned) = blocks.read_band(b, l).unwrap();
                        let first = blocks.bands[l].start;
                        for idx in first..blocks.bands[l].end {
                            let col = column_range(&slots, blocks.rows_in(b), idx);
                            let start = blocks.block_start(b) + col.start - span.start;
                            let view = pinned.slice(start..start + col.len());
                            assert_eq!(
                                view.as_slice(),
                                blocks
                                    .column(b, idx)
                                    .unwrap()
                                    .read_bytes()
                                    .unwrap()
                                    .as_slice()
                            );
                        }
                    }
                }
                assert!(blocks.read_column(0, 2..4).is_err());
            }
        }
    }
}
