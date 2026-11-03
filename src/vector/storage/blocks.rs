//! Block addressing and entry framing. See FORMAT.md for the column contract.
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::ops::Range;
use std::sync::Arc;

use common::{HasLen, OwnedBytes};
use parking_lot::Mutex;

use crate::directory::FileSlice;
use crate::error::DataCorruption;
use crate::schema::VectorOptions;
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
    align_up(directory_start + (num_blocks + 1) * 12, ENTRY_ALIGN) + 16
}
/// Records actual entry-relative positions while blocks stream to storage.
pub(crate) struct BlockDirectory {
    entries: Vec<(u64, u32)>,
}
impl BlockDirectory {
    pub(crate) fn new(first: u64) -> Self {
        Self {
            entries: vec![(first, 0)],
        }
    }
    pub(crate) fn push(&mut self, end: u64, rows: u32) {
        self.entries.push((end, rows));
    }
    pub(crate) fn finish(mut self, writer: &mut impl Write) -> io::Result<usize> {
        let (blocks_end, num_rows) = *self.entries.last().unwrap();
        let directory_start = align_up(blocks_end as usize, ENTRY_ALIGN);
        pad(writer, directory_start - blocks_end as usize)?;
        self.entries.last_mut().unwrap().0 = directory_start as u64;
        for (offset, row) in &self.entries {
            writer.write_all(&offset.to_le_bytes())?;
            writer.write_all(&row.to_le_bytes())?;
        }
        let num_blocks = self.entries.len() - 1;
        let arrays_end = directory_start + self.entries.len() * 12;
        let len = data_entry_len(blocks_end as usize, num_blocks);
        pad(writer, len - 16 - arrays_end)?;
        writer.write_all(&(num_blocks as u64).to_le_bytes())?;
        writer.write_all(&u64::from(num_rows).to_le_bytes())?;
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
/// Data entry with a lazy directory. Only accessed block addresses are retained.
pub(crate) struct Blocks {
    entry: FileSlice,
    pub(crate) meta: Arc<VectorColMetadata>,
    pub(crate) slots: Vec<Slot>,
    bands: Vec<Range<usize>>,
    directory: FileSlice,
    directory_start: usize,
    metadata_end: usize,
    num_blocks: usize,
    num_rows: usize,
    accessed: Mutex<BTreeMap<usize, BlockAddress>>,
}
#[derive(Clone, Debug)]
pub(crate) struct BlockAddress {
    pub(crate) bytes: Range<usize>,
    pub(crate) rows: Range<usize>,
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
        if entry.len() < 16 || entry.len() % ENTRY_ALIGN != 0 {
            return Err(DataCorruption::comment_only("invalid vector directory footer").into());
        }
        let footer = entry.len() - 16;
        let bytes = entry.slice_from(footer).read_bytes()?;
        let count = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
            .map_err(|_| DataCorruption::comment_only("vector block count overflow"))?;
        let num_rows = usize::try_from(u64::from_le_bytes(bytes[8..].try_into().unwrap()))
            .map_err(|_| DataCorruption::comment_only("vector row count overflow"))?;
        if num_rows > u32::MAX as usize || (count == 0 && num_rows != 0) {
            return Err(DataCorruption::comment_only("invalid vector row count").into());
        }
        let directory_len = count
            .checked_add(1)
            .and_then(|n| n.checked_mul(12))
            .ok_or_else(|| DataCorruption::comment_only("vector directory size overflow"))?;
        let padded_len = directory_len
            .checked_add(ENTRY_ALIGN - 1)
            .map(|n| n & !(ENTRY_ALIGN - 1))
            .ok_or_else(|| DataCorruption::comment_only("vector directory size overflow"))?;
        let directory_start = footer
            .checked_sub(padded_len)
            .filter(|&start| start >= metadata_end && start % ENTRY_ALIGN == 0)
            .ok_or_else(|| DataCorruption::comment_only("truncated vector directory"))?;
        if entry
            .slice(directory_start + directory_len..footer)
            .read_bytes()?
            .iter()
            .any(|&b| b != 0)
        {
            return Err(DataCorruption::comment_only("nonzero vector directory padding").into());
        }
        if let Partition::Uniform { rows_per_block } = meta.field().partition {
            if count != num_rows.div_ceil(rows_per_block as usize) {
                return Err(DataCorruption::comment_only("invalid uniform block count").into());
            }
        }
        let directory = entry.slice(directory_start..directory_start + directory_len);
        if count == 0 {
            let sentinel = directory.read_bytes()?;
            if u64::from_le_bytes(sentinel[..8].try_into().unwrap()) != directory_start as u64
                || u32::from_le_bytes(sentinel[8..].try_into().unwrap()) != 0
                || directory_start - metadata_end >= ENTRY_ALIGN
                || entry
                    .slice(metadata_end..directory_start)
                    .read_bytes()?
                    .iter()
                    .any(|&v| v != 0)
            {
                return Err(DataCorruption::comment_only("invalid empty vector directory").into());
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
            entry,
            meta,
            slots,
            bands,
            directory,
            directory_start,
            metadata_end,
            num_blocks: count,
            num_rows,
            accessed: Mutex::new(BTreeMap::new()),
        })
    }
    #[cfg(test)]
    pub(crate) fn accessed_clusters(&self) -> Vec<usize> {
        self.accessed.lock().keys().copied().collect()
    }

    pub(crate) fn num_blocks(&self) -> usize {
        self.num_blocks
    }
    pub(crate) fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub(crate) fn address(&self, b: usize) -> crate::Result<BlockAddress> {
        if b >= self.num_blocks {
            return Err(DataCorruption::comment_only("vector block out of range").into());
        }
        if let Some(address) = self.accessed.lock().get(&b) {
            return Ok(address.clone());
        }
        let bytes = self
            .directory
            .slice(b * 12..(b + 2) * 12)
            .read_vector_bytes(VectorReadKind::Other)?;
        let decode = |at: usize| -> crate::Result<(usize, usize)> {
            let offset = usize::try_from(u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()))
                .map_err(|_| DataCorruption::comment_only("vector byte offset overflow"))?;
            Ok((
                offset,
                u32::from_le_bytes(bytes[at + 8..at + 12].try_into().unwrap()) as usize,
            ))
        };
        let (start, first) = decode(0)?;
        let (end, last) = decode(12)?;
        let align = block_align(&self.slots);
        if start < self.metadata_end
            || end > self.directory_start
            || start > end
            || start % align != 0
            || end % align != 0
            || first > last
            || last > self.num_rows
            || (b == 0 && (first != 0 || start - self.metadata_end >= align))
            || (b + 1 == self.num_blocks && (end != self.directory_start || last != self.num_rows))
        {
            return Err(DataCorruption::comment_only("invalid vector block boundaries").into());
        }
        if let Partition::Uniform { rows_per_block } = self.meta.field().partition {
            let rows = rows_per_block as usize;
            if first != b * rows || last != ((b + 1) * rows).min(self.num_rows) {
                return Err(DataCorruption::comment_only("invalid uniform row boundaries").into());
            }
        }
        let n = last - first;
        let row_bytes: usize = self.slots.iter().map(|s| s.stride as usize).sum();
        if n.checked_mul(row_bytes)
            .is_none_or(|bytes| bytes > end - start)
        {
            return Err(DataCorruption::comment_only("truncated vector block").into());
        }
        let payload_end = start + block_len(&self.slots, n);
        if payload_end > end
            || (b + 1 < self.num_blocks && payload_end != end)
            || end - payload_end >= ENTRY_ALIGN
        {
            return Err(DataCorruption::comment_only("invalid vector block size").into());
        }
        if payload_end < end
            && self
                .entry
                .slice(payload_end..end)
                .read_vector_bytes(VectorReadKind::Other)?
                .iter()
                .any(|&v| v != 0)
        {
            return Err(DataCorruption::comment_only("nonzero vector block padding").into());
        }
        let address = BlockAddress {
            bytes: start..end,
            rows: first..last,
        };
        self.accessed.lock().insert(b, address.clone());
        Ok(address)
    }
    #[cfg(test)]
    pub(crate) fn block_start(&self, b: usize) -> crate::Result<usize> {
        Ok(self.address(b)?.bytes.start)
    }
    pub(crate) fn rows_in(&self, b: usize) -> crate::Result<usize> {
        Ok(self.address(b)?.rows.len())
    }
    pub(crate) fn row_range(&self, b: usize) -> crate::Result<Range<usize>> {
        Ok(self.address(b)?.rows)
    }
    pub(crate) fn block_of(&self, row: usize) -> crate::Result<usize> {
        if row >= self.num_rows {
            return Err(DataCorruption::comment_only("vector row out of range").into());
        }
        if let Partition::Uniform { rows_per_block } = self.meta.field().partition {
            return Ok(row / rows_per_block as usize);
        }
        let (mut lo, mut hi) = (0, self.num_blocks);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let address = self.address(mid)?;
            if row < address.rows.start {
                hi = mid;
            } else if row >= address.rows.end {
                lo = mid + 1;
            } else {
                return Ok(mid);
            }
        }
        Err(DataCorruption::comment_only("vector row missing from directory").into())
    }
    /// Resolves one complete column as a file slice preserving parent storage geometry.
    pub(crate) fn column(&self, b: usize, idx: usize) -> crate::Result<FileSlice> {
        let range = column_range(&self.slots, self.rows_in(b)?, idx);
        self.block_slice(b, range)
    }
    /// Resolves a byte span relative to a validated block while preserving storage geometry.
    pub(crate) fn block_slice(&self, b: usize, range: Range<usize>) -> crate::Result<FileSlice> {
        let address = self.address(b)?;
        let start = address.bytes.start;
        if range.start > range.end || range.end > address.bytes.end - start {
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
        if rows.start >= rows.end || rows.end > self.num_rows {
            return Err(crate::TantivyError::DataCorruption(
                DataCorruption::comment_only("invalid vector blocks: invalid row range"),
            ));
        }
        let b = self.block_of(rows.start)?;
        if rows.end > self.row_range(b)?.end {
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
        let block_rows = self.row_range(b)?;
        if rows.start < block_rows.start || rows.start >= rows.end || rows.end > block_rows.end {
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
        if rows.start == rows.end && rows.end <= self.num_rows {
            return Ok(OwnedBytes::empty());
        }
        let b = self.block_for_range(&rows)?;
        self.read_column_in_block(b, idx, rows)
    }
    pub(crate) fn read_column_in_block(
        &self,
        b: usize,
        idx: usize,
        rows: Range<usize>,
    ) -> crate::Result<OwnedBytes> {
        if rows.is_empty() && rows == self.row_range(b)? {
            return Ok(OwnedBytes::empty());
        }
        self.check_block_rows(b, &rows)?;
        let stride = self.slots[idx].stride as usize;
        let first = self.row_range(b)?.start;
        let kind = match self.slots[idx].slot_type {
            SlotType::Rows { .. } => VectorReadKind::FullPrecision,
            ref slot => slot
                .band()
                .map(VectorReadKind::QuantizedLayer)
                .unwrap_or_default(),
        };
        Ok(self
            .column(b, idx)?
            .slice((rows.start - first) * stride..(rows.end - first) * stride)
            .read_vector_bytes(kind)?)
    }
    /// Whether blocks carry a document-id column and variable cluster boundaries.
    pub(crate) fn clustered(&self) -> bool {
        matches!(self.meta.field().partition, Partition::Clusters)
    }
    /// Layer zero spans residual norms through its last sidecar; document ids are read separately.
    pub(crate) fn layer_span(&self, b: usize, layer: usize) -> crate::Result<Range<usize>> {
        self.band_range(b, layer)
    }
    /// Range of one scan band within the Data entry; band zero includes residual norms.
    pub(crate) fn band_range(&self, b: usize, layer: usize) -> crate::Result<Range<usize>> {
        let band = &self.bands[layer];
        let address = self.address(b)?;
        let n = address.rows.len();
        let start = column_range(&self.slots, n, band.start).start;
        let end = column_range(&self.slots, n, band.end - 1).end;
        Ok(address.bytes.start + start..address.bytes.start + end)
    }
    /// One scan band of block `b` as a file slice.
    pub(crate) fn band_slice(&self, b: usize, layer: usize) -> crate::Result<FileSlice> {
        let range = self.layer_span(b, layer)?;
        let address = self.address(b)?;
        let start = address.bytes.start;
        self.block_slice(b, range.start - start..range.end - start)
    }
    /// Pins a band with exactly one read. Column views borrow this span without a heap list.
    pub(crate) fn read_band(
        &self,
        b: usize,
        layer: usize,
    ) -> crate::Result<(Range<usize>, OwnedBytes)> {
        let range = self.layer_span(b, layer)?;
        let bytes = self
            .band_slice(b, layer)?
            .read_vector_bytes(VectorReadKind::QuantizedLayer(layer as u8))?;
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
            for row in 0..count {
                pad(&mut data, block_len(&meta.slots(), 1))?;
                writer.push(data.len() as u64, row + 1);
            }
            writer.finish(&mut data)?;
            let reads = Arc::new(std::sync::Mutex::new(Vec::new()));
            let blocks = Blocks::open(
                FileSlice::new(Arc::new(BlockTrackedBytes {
                    bytes: data.clone(),
                    reads: reads.clone(),
                    block_len: 8192,
                })),
                &opts,
                true,
            )?;
            let directory =
                blocks.directory_start..blocks.directory_start + (count as usize + 1) * 12;
            assert!(reads
                .lock()
                .unwrap()
                .iter()
                .all(|r| r.end <= directory.start || r.start >= directory.end));
            assert!(blocks.accessed_clusters().is_empty());
            reads.lock().unwrap().clear();
            assert_eq!(blocks.row_range(17)?, 17..18);
            assert_eq!(
                *reads.lock().unwrap(),
                [directory.start + 17 * 12..directory.start + 19 * 12]
            );
            reads.lock().unwrap().clear();
            blocks.column(17, 0)?;
            blocks.read_column_in_block(17, 0, 17..18)?;
            assert!(reads
                .lock()
                .unwrap()
                .iter()
                .all(|r| r.end <= directory.start));
            assert_eq!(blocks.accessed_clusters(), [17]);

            data[directory.start + 25 * 12 + 8..directory.start + 25 * 12 + 12]
                .copy_from_slice(&u32::MAX.to_le_bytes());
            let blocks = Blocks::open(FileSlice::from(data), &opts, true)?;
            assert_eq!(blocks.row_range(17)?, 17..18);
            assert!(matches!(
                blocks.address(25),
                Err(crate::TantivyError::DataCorruption(_))
            ));
        }
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
                    directory.push(bytes.len() as u64, end as u32);
                }
                directory.finish(&mut bytes).unwrap();
                let blocks = Blocks::open(FileSlice::from(bytes.clone()), &opts, false).unwrap();
                assert_eq!(
                    data_entry_len(blocks.directory_start, blocks.num_blocks()),
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
                assert!(Blocks::open(FileSlice::from(bytes), &opts, false).is_err());
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
                directory.push(block_end as u64, n as u32);
            }
            directory.finish(&mut bytes).unwrap();
            let blocks = Blocks::open(FileSlice::from(bytes.clone()), &opts, false).unwrap();
            let b = usize::from(n != 0);
            let start = align_up(block_end, ENTRY_ALIGN);
            assert_eq!(blocks.directory_start, start);
            assert_eq!(blocks.num_rows(), n);
            assert_eq!(
                u64::from_le_bytes(bytes[bytes.len() - 16..bytes.len() - 8].try_into().unwrap()),
                b as u64
            );
            let reject = |corrupt: Vec<u8>| {
                assert!(matches!(
                    Blocks::open(FileSlice::from(corrupt), &opts, false).and_then(|blocks| {
                        for b in 0..blocks.num_blocks() {
                            blocks.address(b)?;
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
            for offset in start + (b + 1) * 12..bytes.len() - 16 {
                let mut corrupt = bytes.clone();
                corrupt[offset] = 1;
                reject(corrupt);
            }
            let mut corrupt = bytes.clone();
            corrupt[start + b * 12] ^= 1;
            reject(corrupt);
            let mut corrupt = bytes.clone();
            corrupt[start + b * 12 + 8] ^= 1;
            reject(corrupt);
            let mut corrupt = bytes.clone();
            let footer = corrupt.len() - 16;
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
                let blocks = Blocks::open(FileSlice::from(bytes.clone()), &opts, true).unwrap();
                assert_eq!(blocks.block_of(0).unwrap(), 1);
                assert_eq!(blocks.block_of(3).unwrap(), 3);
                for b in [1, 3] {
                    let mut previous = blocks.block_start(b).unwrap();
                    for idx in 0..slots.len() {
                        let col = column_range(&slots, blocks.rows_in(b).unwrap(), idx);
                        assert!(bytes[previous..blocks.block_start(b).unwrap() + col.start]
                            .iter()
                            .all(|&v| v == 0));
                        previous = blocks.block_start(b).unwrap() + col.end;
                    }
                    assert!(bytes[previous..blocks.address(b).unwrap().bytes.end]
                        .iter()
                        .all(|&v| v == 0));
                    for l in 0..schedule.len() {
                        let (span, pinned) = blocks.read_band(b, l).unwrap();
                        let first = blocks.bands[l].start;
                        for idx in first..blocks.bands[l].end {
                            let col = column_range(&slots, blocks.rows_in(b).unwrap(), idx);
                            let start = blocks.block_start(b).unwrap() + col.start - span.start;
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
