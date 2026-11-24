//! Block addressing and entry framing. See FORMAT.md for the column contract.
use std::io::{self, Write};
use std::ops::Range;
use std::sync::Arc;

use common::{HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::error::DataCorruption;
use crate::schema::VectorOptions;
use crate::vector::storage::io::{VectorRead, VectorReadKind};
use crate::vector::storage::metadata::{Partition, Slot, SlotType, VectorColMetadata};
use crate::vector::ENTRY_ALIGN;

/// A vector's row position within one physical block of a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockAddress {
    pub block: u32,
    pub row: u32,
}
impl BlockAddress {
    pub fn new(block: u32, row: u32) -> Self {
        Self { block, row }
    }
}

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
    slots
        .iter()
        .map(Slot::type_bytes)
        .max()
        .expect("every field has a column")
}
/// Logical Data entry length, including its directory and footer.
pub(crate) fn data_entry_len(blocks_end: usize, num_blocks: usize) -> usize {
    let directory_start = align_up(blocks_end, ENTRY_ALIGN);
    directory_start + (num_blocks + 1) * 16 + 16
}
/// Records actual entry-relative positions while blocks stream to storage.
pub(crate) struct BlockOffsetsWriter {
    entries: Vec<(u64, u64)>,
    num_rows: u64,
}
impl BlockOffsetsWriter {
    pub(crate) fn new(first: u64) -> Self {
        Self {
            entries: vec![(first, 0)],
            num_rows: 0,
        }
    }
    pub(crate) fn push(&mut self, end: u64, num_rows: usize) {
        self.entries.last_mut().unwrap().1 = num_rows as u64;
        self.entries.push((end, 0));
        self.num_rows += num_rows as u64;
    }
    pub(crate) fn finish(mut self, writer: &mut impl Write) -> io::Result<usize> {
        let blocks_end = self.entries.last().unwrap().0;
        let offsets_start = align_up(blocks_end as usize, ENTRY_ALIGN);
        pad(writer, offsets_start - blocks_end as usize)?;
        self.entries.last_mut().unwrap().0 = offsets_start as u64;
        for (offset, count) in &self.entries {
            writer.write_all(&offset.to_le_bytes())?;
            writer.write_all(&count.to_le_bytes())?;
        }
        let num_blocks = self.entries.len() - 1;
        if self.num_rows > u32::MAX as u64 || num_blocks > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "vector block/row count exceeds u32",
            ));
        }
        writer.write_all(&self.num_rows.to_le_bytes())?;
        writer.write_all(&(num_blocks as u64).to_le_bytes())?;
        Ok(data_entry_len(blocks_end as usize, num_blocks))
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
/// Lazy `(byte_offset, num_rows)` entries shared by routing and column readers.
#[derive(Clone)]
pub(crate) struct BlockOffsets {
    offsets_slice: FileSlice,
    offsets_start: usize,
    num_blocks: usize,
    num_rows: usize,
    metadata_end: usize,
}
impl BlockOffsets {
    fn open(data_slice: &FileSlice, metadata_end: usize) -> crate::Result<Self> {
        if data_slice.len() < 32 || data_slice.len() % ENTRY_ALIGN != 0 {
            return Err(DataCorruption::comment_only("invalid vector block footer").into());
        }
        let tail = data_slice.read_bytes_slice(data_slice.len() - 32..data_slice.len())?;
        let word = |i: usize| u64::from_le_bytes(tail[i..i + 8].try_into().unwrap());
        let num_rows = word(16);
        let num_blocks = word(24);
        if num_rows > u32::MAX as u64
            || num_blocks > u32::MAX as u64
            || (num_blocks == 0 && num_rows != 0)
        {
            return Err(DataCorruption::comment_only("invalid vector block/row count").into());
        }
        let offsets_len = (num_blocks as usize)
            .checked_add(1)
            .and_then(|count| count.checked_mul(16))
            .ok_or_else(|| DataCorruption::comment_only("block offsets size overflow"))?;
        let start = data_slice
            .len()
            .checked_sub(16)
            .and_then(|end| end.checked_sub(offsets_len))
            .filter(|&start| start >= metadata_end && start % ENTRY_ALIGN == 0)
            .ok_or_else(|| DataCorruption::comment_only("truncated block offsets"))?;
        if word(0) != start as u64 || word(8) != 0 {
            return Err(DataCorruption::comment_only("invalid block offsets sentinel").into());
        }
        if num_blocks == 0
            && (start - metadata_end >= ENTRY_ALIGN
                || data_slice
                    .read_bytes_slice(metadata_end..start)?
                    .iter()
                    .any(|&v| v != 0))
        {
            return Err(DataCorruption::comment_only("invalid empty vector blocks").into());
        }
        Ok(Self {
            offsets_slice: data_slice.slice(start..start + offsets_len),
            offsets_start: start,
            num_blocks: num_blocks as usize,
            num_rows: num_rows as usize,
            metadata_end,
        })
    }
    fn start(&self) -> usize {
        self.offsets_start
    }
    pub(crate) fn num_blocks(&self) -> usize {
        self.num_blocks
    }
    pub(crate) fn num_rows(&self) -> usize {
        self.num_rows
    }
    fn read(&self, block: usize) -> crate::Result<(Range<usize>, usize)> {
        if block >= self.num_blocks {
            return Err(DataCorruption::comment_only("vector block out of range").into());
        }
        let at = block * 16;
        // Current tuple and the next byte offset, in one range read.
        let bytes = self
            .offsets_slice
            .read_vector_bytes_slice(at..at + 24, VectorReadKind::Other)?;
        let word = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        let (start, count, end) = (word(0), word(8), word(16));
        if start < self.metadata_end as u64
            || start > end
            || end > self.start() as u64
            || count > self.num_rows as u64
        {
            return Err(DataCorruption::comment_only("invalid vector block offsets").into());
        }
        Ok((start as usize..end as usize, count as usize))
    }
    pub(crate) fn num_rows_in(&self, block: usize) -> crate::Result<usize> {
        Ok(self.read(block)?.1)
    }
}

/// All blocks for one field, with eager metadata and lazy byte and row offsets.
pub(crate) struct Blocks {
    data_slice: FileSlice,
    offsets: BlockOffsets,
    pub(crate) metadata: Arc<VectorColMetadata>,
    pub(crate) schema: Vec<Slot>,
    bands: Vec<Range<usize>>,
    /// A uniform partition's rows as one contiguous span, row `r` at `r * row_bytes`, confirmed
    /// at open against the last block's tuple. Clustered blocks have none.
    flat_rows: Option<FileSlice>,
}

/// The span of a uniform partition's rows. Block `b` starts `b * rows_per_block * row_bytes`
/// past block 0, which follows the metadata at the block alignment, so a row's bytes are
/// arithmetic on its index; the last block's tuple, the one offsets read of the open, confirms
/// the layout.
fn flat_rows_span(
    entry: &FileSlice,
    offsets: &BlockOffsets,
    slots: &[Slot],
    metadata_end: usize,
    rows_per_block: usize,
) -> crate::Result<FileSlice> {
    let [slot] = slots else {
        return Err(DataCorruption::comment_only("flat vector blocks carry one column").into());
    };
    let row_bytes = slot.stride as usize;
    let start = align_up(metadata_end, block_align(slots));
    let num_blocks = offsets.num_blocks();
    let num_rows = offsets.num_rows();
    let end = num_rows
        .checked_mul(row_bytes)
        .and_then(|bytes| start.checked_add(bytes))
        .filter(|&end| end <= offsets.start())
        .ok_or_else(|| DataCorruption::comment_only("flat vector rows overrun their entry"))?;
    if num_blocks > 0 {
        let full = rows_per_block * row_bytes;
        let (last, last_rows) = offsets.read(num_blocks - 1)?;
        if block_len(slots, rows_per_block) != full
            || last.start != start + (num_blocks - 1) * full
            || last_rows != num_rows - (num_blocks - 1) * rows_per_block
            || last.end < last.start + last_rows * row_bytes
        {
            return Err(
                DataCorruption::comment_only("flat vector blocks are not contiguous").into(),
            );
        }
    }
    Ok(entry.slice(start..end))
}

/// One block's validated boundaries, reused locally while accessing its columns.
pub(crate) struct Block<'a> {
    blocks: &'a Blocks,
    bytes: Range<usize>,
    rows: Range<usize>,
}

impl Blocks {
    pub(crate) fn open(
        entry: FileSlice,
        options: &VectorOptions,
        clustered: bool,
    ) -> crate::Result<Self> {
        Self::from_metadata(BlockMetadata::open(entry, options, clustered)?)
    }
    pub(crate) fn from_metadata(metadata: BlockMetadata) -> crate::Result<Self> {
        let BlockMetadata {
            entry,
            meta,
            metadata_end,
        } = metadata;
        let offsets = BlockOffsets::open(&entry, metadata_end)?;
        if let Partition::Uniform { rows_per_block } = meta.field().partition {
            if offsets.num_blocks != offsets.num_rows.div_ceil(rows_per_block as usize) {
                return Err(DataCorruption::comment_only("invalid uniform block count").into());
            }
        }
        let slots = meta.slots();
        let flat_rows = match meta.field().partition {
            Partition::Uniform { rows_per_block } => Some(flat_rows_span(
                &entry,
                &offsets,
                &slots,
                metadata_end,
                rows_per_block as usize,
            )?),
            Partition::Clusters => None,
        };
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
            data_slice: entry,
            offsets,
            metadata: meta,
            schema: slots,
            bands,
            flat_rows,
        })
    }
    pub(crate) fn num_blocks(&self) -> usize {
        self.offsets.num_blocks
    }

    /// A uniform partition's rows as one contiguous span; `None` for clustered blocks.
    pub(crate) fn flat_rows(&self) -> Option<&FileSlice> {
        self.flat_rows.as_ref()
    }
    pub(crate) fn num_rows(&self) -> usize {
        self.offsets.num_rows()
    }

    pub(crate) fn block(&self, b: usize) -> crate::Result<Block<'_>> {
        let (bytes, n) = self.offsets.read(b)?;
        let (start, end) = (bytes.start, bytes.end);
        let offsets_start = self.offsets.start();
        let align = block_align(&self.schema);
        if start % align != 0
            || end % align != 0
            || (b == 0 && start - self.offsets.metadata_end >= align)
            || (b + 1 == self.num_blocks() && end != offsets_start)
        {
            return Err(DataCorruption::comment_only("invalid vector block boundaries").into());
        }
        if let Partition::Uniform { rows_per_block } = self.metadata.field().partition {
            let expected =
                (self.num_rows() - b * rows_per_block as usize).min(rows_per_block as usize);
            if n != expected {
                return Err(DataCorruption::comment_only("invalid uniform block row count").into());
            }
        }
        let row_bytes: usize = self.schema.iter().map(|s| s.stride as usize).sum();
        if n.checked_mul(row_bytes)
            .is_none_or(|bytes| bytes > end - start)
        {
            return Err(DataCorruption::comment_only("truncated vector block").into());
        }
        let payload_end = start + block_len(&self.schema, n);
        if payload_end > end
            || (b + 1 < self.num_blocks() && payload_end != end)
            || end - payload_end >= ENTRY_ALIGN
        {
            return Err(DataCorruption::comment_only("invalid vector block size").into());
        }
        if payload_end < end
            && self
                .data_slice
                .read_vector_bytes_slice(payload_end..end, VectorReadKind::Other)?
                .iter()
                .any(|&v| v != 0)
        {
            return Err(DataCorruption::comment_only("nonzero vector block padding").into());
        }
        Ok(Block {
            blocks: self,
            bytes: start..end,
            rows: 0..n,
        })
    }
    pub(crate) fn block_offsets(&self) -> BlockOffsets {
        self.offsets.clone()
    }
    pub(crate) fn row_range(&self, block: usize) -> crate::Result<Range<usize>> {
        Ok(0..self.offsets.num_rows_in(block)?)
    }
    pub(crate) fn block_for_range(
        &self,
        block: usize,
        rows: &Range<usize>,
    ) -> crate::Result<Block<'_>> {
        let block = self.block(block)?;
        block.check_rows(rows)?;
        Ok(block)
    }
    /// Reads block-local rows from one column; the range must fit within a single block.
    pub(crate) fn read_column(
        &self,
        block: usize,
        idx: usize,
        rows: Range<usize>,
    ) -> crate::Result<OwnedBytes> {
        if rows.start == rows.end && rows.end <= self.num_rows() {
            return Ok(OwnedBytes::empty());
        }
        self.block_for_range(block, &rows)?.read_column(idx, rows)
    }

    pub(crate) fn clustered(&self) -> bool {
        matches!(self.metadata.field().partition, Partition::Clusters)
    }

    /// The storage block length the entry's bytes report, when its storage has one.
    pub(crate) fn storage_block_len(&self) -> Option<usize> {
        self.data_slice.storage_block_len()
    }
}

impl Block<'_> {
    pub(crate) fn num_rows(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn row_range(&self) -> Range<usize> {
        self.rows.clone()
    }

    /// Column byte range relative to the Data entry, calculated without storage reads.
    pub(crate) fn column_range(&self, idx: usize) -> Range<usize> {
        let range = column_range(&self.blocks.schema, self.num_rows(), idx);
        self.bytes.start + range.start..self.bytes.start + range.end
    }

    /// A complete column as a lazy slice preserving the underlying storage geometry.
    pub(crate) fn column(&self, idx: usize) -> FileSlice {
        self.blocks.data_slice.slice(self.column_range(idx))
    }

    /// A byte range relative to the block's start.
    pub(crate) fn byte_slice(&self, range: Range<usize>) -> crate::Result<FileSlice> {
        if range.start > range.end || range.end > self.bytes.len() {
            return Err(DataCorruption::comment_only("byte range crosses block boundary").into());
        }
        Ok(self
            .blocks
            .data_slice
            .slice(self.bytes.start + range.start..self.bytes.start + range.end))
    }

    /// Validates a nonempty range of block-local row numbers against this block.
    pub(crate) fn check_rows(&self, rows: &Range<usize>) -> crate::Result<()> {
        if rows.start < self.rows.start || rows.start >= rows.end || rows.end > self.rows.end {
            return Err(DataCorruption::comment_only("row range crosses block boundary").into());
        }
        Ok(())
    }

    /// Reads block-local rows directly from a column, without creating a temporary FileSlice.
    pub(crate) fn read_column(&self, idx: usize, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        if rows.is_empty() && rows == self.rows {
            return Ok(OwnedBytes::empty());
        }
        self.check_rows(&rows)?;
        let slot = &self.blocks.schema[idx];
        let stride = slot.stride as usize;
        let start = self.column_range(idx).start;
        let kind = match slot.slot_type {
            SlotType::Rows { .. } => VectorReadKind::FullPrecision,
            ref slot => slot
                .band()
                .map(VectorReadKind::QuantizedLayer)
                .unwrap_or_default(),
        };
        Ok(self.blocks.data_slice.read_vector_bytes_slice(
            start + (rows.start - self.rows.start) * stride
                ..start + (rows.end - self.rows.start) * stride,
            kind,
        )?)
    }

    /// One quantization layer's columns within the Data entry; layer zero includes norms.
    pub(crate) fn band_range(&self, layer: usize) -> Range<usize> {
        let band = &self.blocks.bands[layer];
        self.column_range(band.start).start..self.column_range(band.end - 1).end
    }

    pub(crate) fn band_slice(&self, layer: usize) -> FileSlice {
        self.blocks.data_slice.slice(self.band_range(layer))
    }

    /// Pins a band's payload in one read, sharing it among the decoded column views.
    pub(crate) fn read_band(&self, layer: usize) -> crate::Result<(Range<usize>, OwnedBytes)> {
        let range = self.band_range(layer);
        let bytes = self
            .blocks
            .data_slice
            .read_vector_bytes_slice(range.clone(), VectorReadKind::QuantizedLayer(layer as u8))?;
        Ok((range, bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;
    use crate::vector::{VectorQuantizationConfig, VectorQuantizationLayer};

    fn encoded(opts: &VectorOptions, counts: &[usize], clustered: bool) -> Vec<u8> {
        let meta = if clustered {
            VectorColMetadata::build_ivf(opts, None).unwrap()
        } else {
            VectorColMetadata::build_flat(opts)
        };
        let mut data = Vec::new();
        write_metadata(&mut data, &meta).unwrap();
        let mut offsets = BlockOffsetsWriter::new(data.len() as u64);
        for &n in counts {
            pad(&mut data, block_len(&meta.slots(), n)).unwrap();
            offsets.push(data.len() as u64, n);
        }
        let len = offsets.finish(&mut data).unwrap();
        assert_eq!(data.len(), len);
        data
    }

    #[test]
    fn tuple_reads_are_lazy_and_reused_by_a_block() -> crate::Result<()> {
        use crate::vector::storage::io::test_support::BlockTrackedBytes;
        for n in [32, 10_000] {
            let opts = VectorOptions::new(2, Metric::L2);
            let data = encoded(&opts, &vec![1; n], true);
            let reads = Arc::new(std::sync::Mutex::new(Vec::new()));
            let blocks = Blocks::open(
                FileSlice::new(Arc::new(BlockTrackedBytes {
                    bytes: data,
                    reads: reads.clone(),
                    block_len: 8192,
                })),
                &opts,
                true,
            )?;
            let start = blocks.offsets.start();
            assert!(reads
                .lock()
                .unwrap()
                .iter()
                .all(|r| r.end <= start || r.start >= start + n * 16));
            reads.lock().unwrap().clear();
            let block = blocks.block(17)?;
            assert_eq!(block.row_range(), 0..1);
            assert_eq!(
                *reads.lock().unwrap(),
                [start + 17 * 16..start + 17 * 16 + 24]
            );
            reads.lock().unwrap().clear();
            block.read_column(0, 0..1)?;
            assert!(reads.lock().unwrap().iter().all(|r| r.end <= start));
            reads.lock().unwrap().clear();
            assert_eq!(blocks.block_offsets().num_rows_in(17)?, 1);
            assert_eq!(
                *reads.lock().unwrap(),
                [start + 17 * 16..start + 17 * 16 + 24]
            );
        }
        Ok(())
    }

    #[test]
    fn tuple_corruption_is_validated_without_loading_other_blocks() -> crate::Result<()> {
        let opts = VectorOptions::new(2, Metric::L2);
        let data = encoded(&opts, &[3, 5, 7], true);
        let start = Blocks::open(FileSlice::from(data.clone()), &opts, true)?
            .offsets
            .start();
        for at in [start + 16, start + 24] {
            let mut corrupt = data.clone();
            corrupt[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            let blocks = Blocks::open(FileSlice::from(corrupt), &opts, true)?;
            assert_eq!(blocks.block(2)?.num_rows(), 7);
            assert!(blocks.block(1).is_err());
        }
        let mut corrupt = data.clone();
        corrupt[start + 24..start + 32].copy_from_slice(&4u64.to_le_bytes());
        assert!(Blocks::open(FileSlice::from(corrupt), &opts, true)?
            .block(1)
            .is_err());
        for at in [
            data.len() - 32,
            data.len() - 24,
            data.len() - 16,
            data.len() - 8,
        ] {
            let mut corrupt = data.clone();
            corrupt[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(Blocks::open(FileSlice::from(corrupt), &opts, true).is_err());
        }
        Ok(())
    }

    #[test]
    fn local_row_bounds_and_empty_blocks() -> crate::Result<()> {
        let opts = VectorOptions::new(2, Metric::L2);
        for counts in [&[][..], &[0], &[0, 3, 0, 5, 0]] {
            let blocks = Blocks::open(FileSlice::from(encoded(&opts, counts, true)), &opts, true)?;
            assert_eq!(blocks.num_rows(), counts.iter().sum::<usize>());
            assert_eq!(blocks.num_blocks(), counts.len());
            for (b, &n) in counts.iter().enumerate() {
                assert_eq!(blocks.row_range(b)?, 0..n);
                assert_eq!(blocks.block(b)?.num_rows(), n);
                if n > 0 {
                    assert!(blocks.read_column(b, 0, n - 1..n).is_ok());
                }
                assert!(blocks.read_column(b, 0, n..n + 1).is_err());
            }
            assert!(blocks.block(counts.len()).is_err());
        }
        Ok(())
    }

    #[test]
    fn flat_blocks_store_full_and_partial_counts() -> crate::Result<()> {
        let opts = VectorOptions::new(2, Metric::L2);
        let n = crate::vector::storage::metadata::FLAT_ROWS_PER_BLOCK as usize;
        let blocks = Blocks::open(
            FileSlice::from(encoded(&opts, &[n, n, 3], false)),
            &opts,
            false,
        )?;
        assert_eq!(blocks.num_rows(), 2 * n + 3);
        assert_eq!(blocks.block(2)?.row_range(), 0..3);
        // A wrong count in the last tuple breaks the contiguous layout the open confirms.
        let mut data = encoded(&opts, &[n, n, 3], false);
        let at = blocks.offsets.start() + 2 * 16 + 8;
        data[at..at + 8].copy_from_slice(&2u64.to_le_bytes());
        assert!(Blocks::open(FileSlice::from(data), &opts, false).is_err());
        Ok(())
    }

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
                let mut directory = BlockOffsetsWriter::new(bytes.len() as u64);
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
                    directory.push(bytes.len() as u64, n);
                }
                directory.finish(&mut bytes).unwrap();
                let blocks = Blocks::open(FileSlice::from(bytes.clone()), &opts, true).unwrap();
                for b in [1, 3] {
                    let block = blocks.block(b).unwrap();
                    let mut previous = block.bytes.start;
                    for idx in 0..slots.len() {
                        let col = column_range(&slots, block.num_rows(), idx);
                        assert!(bytes[previous..block.bytes.start + col.start]
                            .iter()
                            .all(|&v| v == 0));
                        previous = block.bytes.start + col.end;
                    }
                    assert!(bytes[previous..block.bytes.end].iter().all(|&v| v == 0));
                    for l in 0..schedule.len() {
                        let (span, pinned) = block.read_band(l).unwrap();
                        let first = blocks.bands[l].start;
                        for idx in first..blocks.bands[l].end {
                            let col = column_range(&slots, block.num_rows(), idx);
                            let start = block.bytes.start + col.start - span.start;
                            let view = pinned.slice(start..start + col.len());
                            assert_eq!(
                                view.as_slice(),
                                block.column(idx).read_bytes().unwrap().as_slice()
                            );
                        }
                    }
                }
                assert!(blocks.read_column(1, 0, 2..4).is_err());
            }
        }
    }
}
