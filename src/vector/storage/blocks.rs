//! Block addressing and entry framing. See FORMAT.md for the column contract.
use std::io::{self, Write};
use std::ops::Range;
use std::sync::Arc;

use common::{HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::error::DataCorruption;
use crate::schema::VectorOptions;
use crate::vector::storage::centroids::ClusterOffsets;
use crate::vector::storage::io::{VectorRead, VectorReadKind};
use crate::vector::storage::metadata::{Partition, Slot, SlotType, VectorColMetadata};
use crate::vector::ENTRY_ALIGN;

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
/// Logical Data entry length, including its directory and footer.
pub(crate) fn data_entry_len(blocks_end: usize, num_blocks: usize) -> usize {
    let directory_start = align_up(blocks_end, ENTRY_ALIGN);
    directory_start + (num_blocks + 1) * 8 + 8
}
/// Records actual entry-relative positions while blocks stream to storage.
pub(crate) struct BlockDirectory {
    byte_offsets: Vec<u64>,
}
impl BlockDirectory {
    pub(crate) fn new(first: u64) -> Self {
        Self {
            byte_offsets: vec![first],
        }
    }
    pub(crate) fn push(&mut self, end: u64) {
        self.byte_offsets.push(end);
    }
    pub(crate) fn finish(mut self, writer: &mut impl Write) -> io::Result<usize> {
        let blocks_end = *self.byte_offsets.last().unwrap();
        let directory_start = align_up(blocks_end as usize, ENTRY_ALIGN);
        pad(writer, directory_start - blocks_end as usize)?;
        *self.byte_offsets.last_mut().unwrap() = directory_start as u64;
        for offset in &self.byte_offsets {
            writer.write_all(&offset.to_le_bytes())?;
        }
        let num_blocks = self.byte_offsets.len() - 1;
        let len = data_entry_len(blocks_end as usize, num_blocks);
        writer.write_all(&(num_blocks as u64).to_le_bytes())?;
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
/// Row geometry: uniform blocks use the IdMap count; clusters use `.centroids` offsets.
pub(crate) enum RowOffsets {
    Uniform { num_rows: usize },
    Clustered(ClusterOffsets),
}

impl RowOffsets {
    fn num_rows(&self) -> usize {
        match self {
            Self::Uniform { num_rows } => *num_rows,
            Self::Clustered(offsets) => offsets.num_rows(),
        }
    }

    #[cfg(test)]
    pub(crate) fn clusters(offsets: &[u64]) -> Self {
        let bytes: Vec<u8> = offsets.iter().flat_map(|n| n.to_le_bytes()).collect();
        Self::Clustered(ClusterOffsets::open(FileSlice::from(bytes)).unwrap())
    }
}

/// All blocks for one field, with eager metadata and lazy byte and row offsets.
pub(crate) struct Blocks {
    data_slice: FileSlice,
    pub(crate) metadata: Arc<VectorColMetadata>,
    pub(crate) schema: Vec<Slot>,
    bands: Vec<Range<usize>>,
    num_blocks: usize,
    metadata_end: usize,
    row_offsets: RowOffsets,
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
        row_offsets: RowOffsets,
    ) -> crate::Result<Self> {
        let clustered = matches!(row_offsets, RowOffsets::Clustered(_));
        Self::from_metadata(BlockMetadata::open(entry, options, clustered)?, row_offsets)
    }
    pub(crate) fn from_metadata(
        metadata: BlockMetadata,
        row_offsets: RowOffsets,
    ) -> crate::Result<Self> {
        let BlockMetadata {
            entry,
            meta,
            metadata_end,
        } = metadata;
        if entry.len() < 8 || entry.len() % ENTRY_ALIGN != 0 {
            return Err(DataCorruption::comment_only("invalid vector block-count footer").into());
        }
        let footer = entry.len() - 8;
        let bytes = entry.read_bytes_slice(footer..entry.len())?;
        let count = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
            .map_err(|_| DataCorruption::comment_only("vector block count overflow"))?;
        let num_rows = row_offsets.num_rows();
        if num_rows > u32::MAX as usize || (count == 0 && num_rows != 0) {
            return Err(DataCorruption::comment_only("invalid vector row count").into());
        }
        let block_offsets_len = count
            .checked_add(1)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| DataCorruption::comment_only("vector block offsets size overflow"))?;
        let block_offsets_start = footer
            .checked_sub(block_offsets_len)
            .filter(|&start| start >= metadata_end && start % ENTRY_ALIGN == 0)
            .ok_or_else(|| DataCorruption::comment_only("truncated vector block offsets"))?;
        match (&row_offsets, &meta.field().partition) {
            (RowOffsets::Uniform { .. }, Partition::Uniform { rows_per_block }) => {
                if count != num_rows.div_ceil(*rows_per_block as usize) {
                    return Err(DataCorruption::comment_only("invalid uniform block count").into());
                }
            }
            (RowOffsets::Clustered(offsets), Partition::Clusters) => {
                if count != offsets.num_clusters() {
                    return Err(
                        DataCorruption::comment_only("cluster offset count mismatch").into(),
                    );
                }
            }
            _ => {
                return Err(DataCorruption::comment_only("partition/row geometry mismatch").into());
            }
        }
        if count == 0 {
            let sentinel = entry.read_bytes_slice(block_offsets_start..footer)?;
            if u64::from_le_bytes(sentinel[..8].try_into().unwrap()) != block_offsets_start as u64
                || block_offsets_start - metadata_end >= ENTRY_ALIGN
                || entry
                    .read_bytes_slice(metadata_end..block_offsets_start)?
                    .iter()
                    .any(|&v| v != 0)
            {
                return Err(
                    DataCorruption::comment_only("invalid empty vector block offsets").into(),
                );
            }
        }
        let slots = meta.slots();
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
            metadata: meta,
            schema: slots,
            bands,
            metadata_end,
            num_blocks: count,
            row_offsets,
        })
    }
    pub(crate) fn num_blocks(&self) -> usize {
        self.num_blocks
    }
    pub(crate) fn num_rows(&self) -> usize {
        self.row_offsets.num_rows()
    }

    pub(crate) fn block(&self, b: usize) -> crate::Result<Block<'_>> {
        if b >= self.num_blocks {
            return Err(DataCorruption::comment_only("vector block out of range").into());
        }
        let offsets_start = self.block_offsets_start();
        let offset = offsets_start + b * 8;
        let bytes = self
            .data_slice
            .read_vector_bytes_slice(offset..offset + 16, VectorReadKind::Other)?;
        let decode = |at: usize| -> crate::Result<usize> {
            Ok(
                usize::try_from(u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()))
                    .map_err(|_| DataCorruption::comment_only("vector byte offset overflow"))?,
            )
        };
        let start = decode(0)?;
        let end = decode(8)?;
        let rows = self.block_row_range(b)?;
        let num_rows = self.num_rows();
        let (first, last) = (rows.start, rows.end);
        let align = block_align(&self.schema);
        if start < self.metadata_end
            || end > offsets_start
            || start > end
            || start % align != 0
            || end % align != 0
            || first > last
            || last > num_rows
            || (b == 0 && (first != 0 || start - self.metadata_end >= align))
            || (b + 1 == self.num_blocks && (end != offsets_start || last != num_rows))
        {
            return Err(DataCorruption::comment_only("invalid vector block boundaries").into());
        }
        let n = last - first;
        let row_bytes: usize = self.schema.iter().map(|s| s.stride as usize).sum();
        if n.checked_mul(row_bytes)
            .is_none_or(|bytes| bytes > end - start)
        {
            return Err(DataCorruption::comment_only("truncated vector block").into());
        }
        let payload_end = start + block_len(&self.schema, n);
        if payload_end > end
            || (b + 1 < self.num_blocks && payload_end != end)
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
            rows: first..last,
        })
    }
    // Construction checks the arithmetic and framing before any block access.
    fn block_offsets_start(&self) -> usize {
        self.data_slice.len() - 8 - (self.num_blocks + 1) * 8
    }

    fn block_row_range(&self, b: usize) -> crate::Result<Range<usize>> {
        match &self.row_offsets {
            RowOffsets::Uniform { num_rows } => {
                let Partition::Uniform { rows_per_block } = self.metadata.field().partition else {
                    unreachable!("validated row offsets")
                };
                let size = rows_per_block as usize;
                Ok(b * size..((b + 1) * size).min(*num_rows))
            }
            RowOffsets::Clustered(offsets) => offsets.range(b),
        }
    }

    pub(crate) fn row_range(&self, b: usize) -> crate::Result<Range<usize>> {
        Ok(self.block(b)?.row_range())
    }

    /// Finds a block by row, reading only row offsets during clustered binary search.
    pub(crate) fn block_of(&self, row: usize) -> crate::Result<usize> {
        if row >= self.num_rows() {
            return Err(DataCorruption::comment_only("vector row out of range").into());
        }
        if let Partition::Uniform { rows_per_block } = self.metadata.field().partition {
            return Ok(row / rows_per_block as usize);
        }
        let (mut lo, mut hi) = (0, self.num_blocks);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let rows = self.block_row_range(mid)?;
            if row < rows.start {
                hi = mid;
            } else if row >= rows.end {
                lo = mid + 1;
            } else {
                return Ok(mid);
            }
        }
        Err(DataCorruption::comment_only("vector row missing from row offsets").into())
    }

    /// Resolves a nonempty field-wide row span contained in one block.
    pub(crate) fn block_for_range(&self, rows: &Range<usize>) -> crate::Result<Block<'_>> {
        if rows.start >= rows.end || rows.end > self.num_rows() {
            return Err(DataCorruption::comment_only("invalid vector row range").into());
        }
        let block = self.block(self.block_of(rows.start)?)?;
        block.check_rows(rows)?;
        Ok(block)
    }

    /// Reads field-wide rows from one column; the range must fit within a single block.
    pub(crate) fn read_column(&self, idx: usize, rows: Range<usize>) -> crate::Result<OwnedBytes> {
        if rows.start == rows.end && rows.end <= self.num_rows() {
            return Ok(OwnedBytes::empty());
        }
        self.block_for_range(&rows)?.read_column(idx, rows)
    }

    pub(crate) fn clustered(&self) -> bool {
        matches!(self.row_offsets, RowOffsets::Clustered(_))
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

    /// Validates a nonempty range of field-wide row numbers against this block.
    pub(crate) fn check_rows(&self, rows: &Range<usize>) -> crate::Result<()> {
        if rows.start < self.rows.start || rows.start >= rows.end || rows.end > self.rows.end {
            return Err(DataCorruption::comment_only("row range crosses block boundary").into());
        }
        Ok(())
    }

    /// Reads field-wide rows directly from a column, without creating a temporary FileSlice.
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
    #[test]
    fn directory_reads_only_the_probed_cluster() -> crate::Result<()> {
        use crate::vector::storage::io::test_support::BlockTrackedBytes;
        for count in [32, 10_000] {
            let opts = VectorOptions::new(2, Metric::L2);
            let meta = VectorColMetadata::build_ivf(&opts, None)?;
            let mut data = Vec::new();
            write_metadata(&mut data, &meta)?;
            let mut writer = BlockDirectory::new(data.len() as u64);
            for _ in 0..count {
                pad(&mut data, block_len(&meta.slots(), 1))?;
                writer.push(data.len() as u64);
            }
            writer.finish(&mut data)?;
            let offsets: Vec<u8> = (0..=count as u64).flat_map(u64::to_le_bytes).collect();
            let reads = Arc::new(std::sync::Mutex::new(Vec::new()));
            let offset_reads = Arc::new(std::sync::Mutex::new(Vec::new()));
            let cluster_offsets =
                ClusterOffsets::open(FileSlice::new(Arc::new(BlockTrackedBytes {
                    bytes: offsets.clone(),
                    reads: offset_reads.clone(),
                    block_len: 8192,
                })))?;
            let blocks = Blocks::open(
                FileSlice::new(Arc::new(BlockTrackedBytes {
                    bytes: data.clone(),
                    reads: reads.clone(),
                    block_len: 8192,
                })),
                &opts,
                RowOffsets::Clustered(cluster_offsets),
            )?;
            let directory =
                blocks.block_offsets_start()..blocks.block_offsets_start() + (count + 1) * 8;
            assert!(reads
                .lock()
                .unwrap()
                .iter()
                .all(|r| r.end <= directory.start || r.start >= directory.end));
            // Opening reads only the row array's first and final sentinel, independent of K.
            assert_eq!(
                *offset_reads.lock().unwrap(),
                [0..8, count * 8..(count + 1) * 8]
            );
            reads.lock().unwrap().clear();
            offset_reads.lock().unwrap().clear();
            let block = blocks.block(17)?;
            assert_eq!(block.row_range(), 17..18);
            assert_eq!(
                *reads.lock().unwrap(),
                [directory.start + 17 * 8..directory.start + 19 * 8]
            );
            assert_eq!(*offset_reads.lock().unwrap(), [17 * 8..19 * 8]);
            reads.lock().unwrap().clear();
            offset_reads.lock().unwrap().clear();
            block.column(0);
            block.read_column(0, 17..18)?;
            assert!(reads
                .lock()
                .unwrap()
                .iter()
                .all(|r| r.end <= directory.start));
            assert!(offset_reads.lock().unwrap().is_empty());
            // A fresh Block reads its boundaries again; the collection retains no addresses.
            blocks.block(17)?;
            assert_eq!(*offset_reads.lock().unwrap(), [17 * 8..19 * 8]);

            // Corruption in either file is deferred until its cluster is accessed.
            for corrupt_rows in [false, true] {
                let mut data = data.clone();
                let mut offsets = offsets.clone();
                if corrupt_rows {
                    offsets[25 * 8..26 * 8].copy_from_slice(&u64::MAX.to_le_bytes());
                } else {
                    data[directory.start + 25 * 8..directory.start + 26 * 8]
                        .copy_from_slice(&u64::MAX.to_le_bytes());
                }
                let rows = RowOffsets::Clustered(ClusterOffsets::open(FileSlice::from(offsets))?);
                let blocks = Blocks::open(FileSlice::from(data), &opts, rows)?;
                assert_eq!(blocks.row_range(17)?, 17..18);
                assert!(matches!(
                    blocks.block(25),
                    Err(crate::TantivyError::DataCorruption(_))
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn byte_directory_requires_matching_cluster_rows() -> crate::Result<()> {
        let opts = VectorOptions::new(2, Metric::L2);
        let meta = VectorColMetadata::build_ivf(&opts, None)?;
        let mut data = Vec::new();
        write_metadata(&mut data, &meta)?;
        let mut writer = BlockDirectory::new(data.len() as u64);
        pad(&mut data, block_len(&meta.slots(), 3))?;
        writer.push(data.len() as u64);
        writer.finish(&mut data)?;

        for offsets in [&[0][..], &[0, 1, 3]] {
            assert!(Blocks::open(
                FileSlice::from(data.clone()),
                &opts,
                RowOffsets::clusters(offsets)
            )
            .is_err());
        }
        for offsets in [&[0, 0][..], &[0, 2], &[0, 4]] {
            let blocks = Blocks::open(
                FileSlice::from(data.clone()),
                &opts,
                RowOffsets::clusters(offsets),
            )?;
            assert!(blocks.block(0).is_err());
        }
        let blocks = Blocks::open(FileSlice::from(data), &opts, RowOffsets::clusters(&[0, 3]))?;
        assert_eq!(blocks.row_range(0)?, 0..3);
        Ok(())
    }

    // Every flat boundary maps to the correct row, including empty and partial blocks.
    #[test]
    fn uniform_boundaries_and_rows() {
        let opts = VectorOptions::new(3, Metric::L2);
        for r in [
            7,
            crate::vector::storage::metadata::FLAT_ROWS_PER_BLOCK as usize,
        ] {
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
                    directory.push(bytes.len() as u64);
                }
                directory.finish(&mut bytes).unwrap();
                let blocks = Blocks::open(
                    FileSlice::from(bytes.clone()),
                    &opts,
                    RowOffsets::Uniform { num_rows: n },
                )
                .unwrap();
                assert_eq!(
                    data_entry_len(blocks.block_offsets_start(), blocks.num_blocks()),
                    bytes.len()
                );
                for row in 0..n {
                    assert_eq!(blocks.block_of(row).unwrap(), row / r);
                    assert_eq!(
                        &*blocks.read_column(0, row..row + 1).unwrap(),
                        vec![row as u8; 12]
                    );
                }
                if n > r {
                    assert!(blocks.read_column(0, r - 1..r + 1).is_err());
                }
                bytes.push(0);
                assert!(Blocks::open(
                    FileSlice::from(bytes),
                    &opts,
                    RowOffsets::Uniform { num_rows: n }
                )
                .is_err());
            }
        }
    }
    // Footer framing, directory entries, and padding have corruption checks.
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
                directory.push(block_end as u64);
            }
            directory.finish(&mut bytes).unwrap();
            let blocks = Blocks::open(
                FileSlice::from(bytes.clone()),
                &opts,
                RowOffsets::Uniform { num_rows: n },
            )
            .unwrap();
            let b = usize::from(n != 0);
            let start = align_up(block_end, ENTRY_ALIGN);
            assert_eq!(blocks.block_offsets_start(), start);
            assert_eq!(blocks.num_rows(), n);
            assert_eq!(
                u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap()),
                b as u64
            );
            let reject = |corrupt: Vec<u8>| {
                assert!(matches!(
                    Blocks::open(
                        FileSlice::from(corrupt),
                        &opts,
                        RowOffsets::Uniform { num_rows: n }
                    )
                    .and_then(|blocks| {
                        for b in 0..blocks.num_blocks() {
                            blocks.block(b)?;
                        }
                        Ok(())
                    }),
                    Err(crate::TantivyError::DataCorruption(_))
                ))
            };
            for offset in block_end..start {
                let mut corrupt = bytes.clone();
                corrupt[offset] = 1;
                reject(corrupt);
            }
            let mut corrupt = bytes.clone();
            corrupt[start + b * 8] ^= 1;
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
                    directory.push(bytes.len() as u64);
                }
                directory.finish(&mut bytes).unwrap();
                let blocks = Blocks::open(
                    FileSlice::from(bytes.clone()),
                    &opts,
                    RowOffsets::clusters(&[0, 0, 3, 3, 8, 8]),
                )
                .unwrap();
                assert_eq!(blocks.block_of(0).unwrap(), 1);
                assert_eq!(blocks.block_of(3).unwrap(), 3);
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
                assert!(blocks.read_column(0, 2..4).is_err());
            }
        }
    }
}
