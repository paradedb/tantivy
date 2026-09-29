//! Block addressing and entry framing. See FORMAT.md for the column contract.
use std::io::{self, Write};
use std::ops::Range;
use std::sync::Arc;

use common::{HasLen, OwnedBytes};

use super::metadata::{Partition, Slot, VectorColMetadata};
use super::storage_io::VectorRead;
use super::MAX_ELEM_BYTES;
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
/// Logical Data entry length, including its element-aligned trailer.
pub(crate) fn data_entry_len(blocks_end: usize) -> usize {
    align_up(blocks_end, MAX_ELEM_BYTES)
}
/// Completes a Data entry with zero bytes so the next Data entry needs no inter-entry gap.
pub(crate) fn finish_data(writer: &mut impl Write, blocks_end: usize) -> io::Result<usize> {
    let len = data_entry_len(blocks_end);
    pad(writer, len - blocks_end)?;
    assert_eq!(len % MAX_ELEM_BYTES, 0);
    Ok(len)
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
    header_len: usize,
}
impl BlockMetadata {
    pub(crate) fn open(
        entry: FileSlice,
        opts: &VectorOptions,
        clustered: bool,
    ) -> crate::Result<Self> {
        if entry.len() < 4 {
            return Err(bad("missing metadata length"));
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
            return Err(bad("truncated metadata"));
        }
        let meta = Arc::new(VectorColMetadata::from_bytes(
            &entry.slice(4..4 + len).read_bytes()?,
        )?);
        meta.validate(opts, clustered)?;
        let header_len = align_up(4 + len, block_align(&meta.slots()));
        if header_len > entry.len() {
            return Err(bad("truncated metadata padding"));
        }
        Ok(Self {
            entry,
            meta,
            header_len,
        })
    }
}
/// Data entry plus derived geometry. Per-column offsets are recomputed, never cached per block.
pub(crate) struct Blocks {
    entry: FileSlice,
    pub(crate) meta: Arc<VectorColMetadata>,
    pub(crate) slots: Vec<Slot>,
    bands: Vec<Range<usize>>,
    pub(crate) block_rows: Arc<[usize]>,
    pub(crate) block_starts: Arc<[u64]>,
}
fn bad(message: &str) -> crate::TantivyError {
    DataCorruption::comment_only(format!("invalid vector blocks: {message}")).into()
}
impl Blocks {
    /// Opens validated metadata and checks the exact logical entry length using row counts.
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
            header_len,
        } = metadata;
        let block_rows: Vec<usize> = match (&meta.field().partition, clusters) {
            (Partition::Clusters, Some(rows)) => rows,
            (Partition::Uniform { rows_per_block }, None) => (0..num_rows)
                .step_by(*rows_per_block as usize)
                .chain(std::iter::once(num_rows))
                .collect(),
            _ => return Err(bad("partition/backend mismatch")),
        };
        if block_rows.first() != Some(&0)
            || block_rows.last() != Some(&num_rows)
            || block_rows.windows(2).any(|r| r[0] > r[1])
        {
            return Err(bad("invalid row boundaries"));
        }
        let slots = meta.slots();
        let align = block_align(&slots);
        let mut position = header_len;
        let mut block_starts = vec![position as u64];
        let max_rows = block_rows
            .windows(2)
            .map(|r| r[1] - r[0])
            .max()
            .unwrap_or(0);
        let row_bytes: usize = slots.iter().map(|slot| slot.stride as usize).sum();
        if max_rows
            .checked_mul(row_bytes)
            .is_none_or(|size| size > entry.len())
        {
            return Err(bad("truncated block"));
        }
        // Each row count has one layout; repeated cluster sizes share its checked length.
        let mut lengths = vec![None; max_rows + 1];
        for rows in block_rows.windows(2) {
            let n = rows[1] - rows[0];
            let size = if let Some(size) = lengths[n] {
                size
            } else {
                let mut size = 0usize;
                for slot in &slots {
                    size = size
                        .checked_add(slot.type_bytes() - 1)
                        .map(|s| s & !(slot.type_bytes() - 1))
                        .and_then(|s| {
                            n.checked_mul(slot.stride as usize)
                                .and_then(|len| s.checked_add(len))
                        })
                        .ok_or_else(|| bad("block length overflow"))?;
                }
                size = size
                    .checked_add(align - 1)
                    .map(|s| s & !(align - 1))
                    .ok_or_else(|| bad("block padding overflow"))?;
                lengths[n] = Some(size);
                size
            };
            position = position
                .checked_add(size)
                .ok_or_else(|| bad("entry length overflow"))?;
            if position > entry.len() {
                return Err(bad("truncated block"));
            }
            block_starts.push(position as u64);
        }
        let entry_len = position
            .checked_add(MAX_ELEM_BYTES - 1)
            .map(|p| p & !(MAX_ELEM_BYTES - 1))
            .ok_or_else(|| bad("entry padding overflow"))?;
        if entry_len != entry.len() {
            return Err(bad("entry length mismatch"));
        }
        if entry
            .slice(position..entry_len)
            .read_bytes()?
            .iter()
            .any(|&byte| byte != 0)
        {
            return Err(bad("nonzero entry trailer"));
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
    pub(crate) fn column(&self, b: usize, idx: usize) -> FileSlice {
        let range = column_range(&self.slots, self.rows_in(b), idx);
        self.entry
            .slice(self.block_start(b) + range.start..self.block_start(b) + range.end)
    }
    /// Rejects row spans crossing a block, including malformed and out-of-range spans.
    pub(crate) fn block_for_range(&self, rows: &Range<usize>) -> crate::Result<usize> {
        if rows.start >= rows.end || rows.end > *self.block_rows.last().unwrap() {
            return Err(bad("invalid row range"));
        }
        let b = self.block_of(rows.start);
        if rows.end > self.block_rows[b + 1] {
            return Err(bad("row range crosses block boundary"));
        }
        Ok(b)
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
            .column(b, idx)
            .slice((rows.start - first) * stride..(rows.end - first) * stride)
            .read_vector_bytes()?)
    }
    /// Range of one scan band within the Data entry; band zero includes residual norms.
    pub(crate) fn band_range(&self, b: usize, layer: usize) -> Range<usize> {
        let band = &self.bands[layer];
        let n = self.rows_in(b);
        let start = column_range(&self.slots, n, band.start).start;
        let end = column_range(&self.slots, n, band.end - 1).end;
        self.block_start(b) + start..self.block_start(b) + end
    }
    /// Pins a band with exactly one read and returns zero-copy views in column order.
    pub(crate) fn read_band(
        &self,
        b: usize,
        layer: usize,
    ) -> crate::Result<Vec<(usize, OwnedBytes)>> {
        let range = self.band_range(b, layer);
        let bytes = self.entry.slice(range.clone()).read_vector_bytes()?;
        let mut views = Vec::new();
        for idx in self.bands[layer].clone() {
            let col = column_range(&self.slots, self.rows_in(b), idx);
            let start = self.block_start(b) + col.start - range.start;
            views.push((idx, bytes.slice(start..start + col.len())));
        }
        Ok(views)
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
                }
                let end = bytes.len();
                finish_data(&mut bytes, end).unwrap();
                let blocks = Blocks::open(FileSlice::from(bytes.clone()), &opts, n, None).unwrap();
                assert_eq!(
                    data_entry_len(*blocks.block_starts.last().unwrap() as usize),
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
    // The entry trailer has an exact bounded length and must contain only zero bytes.
    #[test]
    fn rejects_nonzero_or_wrong_length_entry_trailer() {
        let opts = VectorOptions::new(3, Metric::L2);
        let meta = VectorColMetadata::build_flat(&opts);
        let mut bytes = Vec::new();
        write_metadata(&mut bytes, &meta).unwrap();
        bytes.extend([0; 24]);
        let end = bytes.len();
        assert_ne!(data_entry_len(end), end);
        finish_data(&mut bytes, end).unwrap();
        assert!(Blocks::open(FileSlice::from(bytes.clone()), &opts, 2, None).is_ok());
        bytes[end] = 1;
        assert!(matches!(
            Blocks::open(FileSlice::from(bytes.clone()), &opts, 2, None),
            Err(crate::TantivyError::DataCorruption(_))
        ));
        bytes[end] = 0;
        bytes.pop();
        assert!(Blocks::open(FileSlice::from(bytes), &opts, 2, None).is_err());
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
                }
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
                        for (idx, view) in blocks.read_band(b, l).unwrap() {
                            assert_eq!(
                                view.as_slice(),
                                blocks.column(b, idx).read_bytes().unwrap().as_slice()
                            );
                        }
                    }
                }
                assert!(blocks.read_column(0, 2..4).is_err());
            }
        }
    }
}
