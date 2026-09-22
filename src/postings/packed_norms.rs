use std::cell::RefCell;
use std::io::{self, Write};
use std::sync::{Arc, OnceLock};

use common::file_slice::DeferredFileSlice;
use common::HasLen;
use tantivy_bitpacker::{compute_num_bits, minmax, BitPacker, BitUnpacker};

use crate::directory::{BufferedFileSlice, FileSlice, OwnedBytes};

const BLOCK_LEN: usize = 128;
const FOOTER_LEN: usize = 21;
const MAGIC: &[u8; 4] = b"PNB1";

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct PackedNormStats {
    pub raw_bytes: u64,
    pub packed_bytes: u64,
    pub constant_blocks: u64,
    pub packed_blocks: u64,
    pub raw_blocks: u64,
}

pub struct PackedNormWriter<W: Write> {
    writer: W,
    pending: [u8; BLOCK_LEN],
    pending_len: usize,
    offsets: Vec<u64>,
    stats: PackedNormStats,
}

impl<W: Write> PackedNormWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            pending: [0; BLOCK_LEN],
            pending_len: 0,
            offsets: Vec::new(),
            stats: PackedNormStats::default(),
        }
    }

    fn flush_block(&mut self) -> io::Result<()> {
        let values = &self.pending[..self.pending_len];
        let Some((mut minimum, maximum)) = minmax(values.iter().copied()) else {
            return Ok(());
        };
        let width = compute_num_bits(u64::from(maximum - minimum));
        self.offsets.push(self.stats.packed_bytes);
        if width == 8 {
            minimum = 0;
            self.stats.raw_blocks += 1;
        } else if width == 0 {
            self.stats.constant_blocks += 1;
        } else {
            self.stats.packed_blocks += 1;
        }
        self.writer.write_all(&[width, minimum])?;
        if width == 8 {
            self.writer.write_all(values)?;
        } else if width != 0 {
            let mut packer = BitPacker::new();
            for value in values {
                packer.write(u64::from(*value - minimum), width, &mut self.writer)?;
            }
            packer.close(&mut self.writer)?;
        }
        self.stats.packed_bytes += 2 + (values.len() as u64 * u64::from(width)).div_ceil(8);
        self.pending_len = 0;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<PackedNormStats> {
        self.flush_block()?;
        let directory_start = self.stats.packed_bytes;
        let offset_width = if directory_start <= u64::from(u32::MAX) {
            4
        } else {
            8
        };
        for offset in &self.offsets {
            self.writer
                .write_all(&offset.to_le_bytes()[..offset_width])?;
        }
        self.writer.write_all(&self.stats.raw_bytes.to_le_bytes())?;
        self.writer.write_all(&directory_start.to_le_bytes())?;
        self.writer.write_all(&[offset_width as u8])?;
        self.writer.write_all(MAGIC)?;
        self.stats.packed_bytes += (self.offsets.len() * offset_width + FOOTER_LEN) as u64;
        Ok(self.stats)
    }
}

impl<W: Write> Write for PackedNormWriter<W> {
    fn write(&mut self, mut values: &[u8]) -> io::Result<usize> {
        let len = values.len();
        while !values.is_empty() {
            let count = values.len().min(BLOCK_LEN - self.pending_len);
            self.pending[self.pending_len..self.pending_len + count]
                .copy_from_slice(&values[..count]);
            self.pending_len += count;
            self.stats.raw_bytes += count as u64;
            values = &values[count..];
            if self.pending_len == BLOCK_LEN {
                self.flush_block()?;
            }
        }
        Ok(len)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

struct Metadata {
    source: FileSlice,
    len: usize,
    directory_start: usize,
    offset_width: usize,
}

pub(crate) struct PackedNormSource {
    source: DeferredFileSlice,
    metadata: OnceLock<io::Result<Option<Metadata>>>,
}

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid packed norm stream")
}

impl PackedNormSource {
    pub(crate) fn new(source: DeferredFileSlice) -> Self {
        Self {
            source,
            metadata: OnceLock::new(),
        }
    }

    fn metadata(&self) -> io::Result<Option<&Metadata>> {
        self.metadata
            .get_or_init(|| {
                let source = self.source.open()?;
                if source.is_empty() {
                    return Ok(None);
                }
                let footer_start = source.len().checked_sub(FOOTER_LEN).ok_or_else(invalid)?;
                let footer = source.slice_from(footer_start).read_bytes()?;
                if &footer[17..] != MAGIC {
                    return Err(invalid());
                }
                let len = usize::try_from(u64::from_le_bytes(footer[..8].try_into().unwrap()))
                    .map_err(|_| invalid())?;
                let directory_start =
                    usize::try_from(u64::from_le_bytes(footer[8..16].try_into().unwrap()))
                        .map_err(|_| invalid())?;
                let offset_width = footer[16] as usize;
                if !matches!(offset_width, 4 | 8)
                    || len
                        .div_ceil(BLOCK_LEN)
                        .checked_mul(offset_width)
                        .and_then(|n| directory_start.checked_add(n))
                        != Some(footer_start)
                {
                    return Err(invalid());
                }
                Ok(Some(Metadata {
                    source: source.clone(),
                    len,
                    directory_start,
                    offset_width,
                }))
            })
            .as_ref()
            .map(|metadata| metadata.as_ref())
            .map_err(|e| io::Error::new(e.kind(), e.to_string()))
    }
}

#[derive(Clone)]
pub(crate) struct PackedNormReader {
    len: usize,
    offset_width: usize,
    payload: BufferedFileSlice,
    directory: BufferedFileSlice,
    payload_len: usize,
    first_block: usize,
    range: std::ops::Range<usize>,
    block: RefCell<(usize, u8, BitUnpacker, OwnedBytes)>,
}

impl PackedNormReader {
    pub(crate) fn open(
        source: &Arc<PackedNormSource>,
        range: Option<std::ops::Range<usize>>,
    ) -> io::Result<Option<Self>> {
        let Some(meta) = source.metadata()? else {
            return Ok(None);
        };
        let range = range.unwrap_or(0..meta.len);
        if range.start > range.end || range.end > meta.len {
            return Err(invalid());
        }
        let first_block = range.start / BLOCK_LEN;
        let after_block = range.end.div_ceil(BLOCK_LEN);
        let has_end_offset = after_block < meta.len.div_ceil(BLOCK_LEN);
        let directory_start = meta.directory_start + first_block * meta.offset_width;
        let directory_end =
            meta.directory_start + (after_block + usize::from(has_end_offset)) * meta.offset_width;
        let directory =
            BufferedFileSlice::new(meta.source.slice(directory_start..directory_end), 8192);
        let payload_len = if has_end_offset {
            let offset = (after_block - first_block) * meta.offset_width;
            let data = directory.get_bytes(offset as u64..(offset + meta.offset_width) as u64)?;
            let mut bytes = [0; 8];
            bytes[..meta.offset_width].copy_from_slice(&data);
            usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| invalid())?
        } else {
            meta.directory_start
        };
        if payload_len > meta.directory_start {
            return Err(invalid());
        }
        Ok(Some(Self {
            len: meta.len,
            offset_width: meta.offset_width,
            payload: BufferedFileSlice::new(meta.source.slice_to(payload_len), 8192),
            directory,
            payload_len,
            first_block,
            range,
            block: RefCell::new((usize::MAX, 0, BitUnpacker::new(0), OwnedBytes::empty())),
        }))
    }

    pub(crate) fn read(&self, ordinal: usize) -> io::Result<u8> {
        if !self.range.contains(&ordinal) {
            return Err(invalid());
        }
        let block_id = ordinal / BLOCK_LEN;
        let mut block = self.block.borrow_mut();
        if block.0 != block_id {
            let last = block_id + 1 == self.len.div_ceil(BLOCK_LEN);
            let start = (block_id - self.first_block) * self.offset_width;
            let offsets = self.directory.get_bytes(
                start as u64..(start + self.offset_width * if last { 1 } else { 2 }) as u64,
            )?;
            let mut bytes = [0; 8];
            bytes[..self.offset_width].copy_from_slice(&offsets[..self.offset_width]);
            let from = usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| invalid())?;
            let to = if last {
                self.payload_len
            } else {
                bytes[..self.offset_width].copy_from_slice(&offsets[self.offset_width..]);
                usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| invalid())?
            };
            if to > self.payload_len || to.checked_sub(from).is_none_or(|n| n < 2) {
                return Err(invalid());
            }
            let data = self.payload.get_bytes(from as u64..to as u64)?;
            let width = data[0];
            let count = (self.len - block_id * BLOCK_LEN).min(BLOCK_LEN);
            if width > 8 || data.len() != 2 + (count * width as usize).div_ceil(8) {
                return Err(invalid());
            }
            *block = (
                block_id,
                data[1],
                BitUnpacker::new(width),
                data.slice(2..data.len()),
            );
        }
        let delta = block.2.get((ordinal % BLOCK_LEN) as u32, &block.3) as u8;
        block.1.checked_add(delta).ok_or_else(invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn all_widths_tails_and_random_access() {
        for len in [0, 1, 127, 128, 129, 8193, 20000] {
            for width in 0..=8 {
                let mask = (1u16 << width) - 1;
                let base = if width == 8 { 0 } else { 19 };
                let values: Vec<u8> = (0..len)
                    .map(|i| base + ((i as u16).wrapping_mul(73) & mask) as u8)
                    .collect();
                let mut bytes = Vec::new();
                let mut writer = PackedNormWriter::new(&mut bytes);
                for chunk in values.chunks(91) {
                    writer.write_all(chunk).unwrap();
                }
                let stats = writer.finish().unwrap();
                assert_eq!(stats.raw_bytes, len as u64);
                assert_eq!(stats.packed_bytes, bytes.len() as u64);
                let source = FileSlice::from(bytes);
                let opens = Arc::new(AtomicUsize::new(0));
                let counter = opens.clone();
                let source = Arc::new(PackedNormSource::new(DeferredFileSlice::new(move || {
                    counter.fetch_add(1, Ordering::Relaxed);
                    Ok(source.clone())
                })));
                assert_eq!(opens.load(Ordering::Relaxed), 0);
                let reader = PackedNormReader::open(&source, None).unwrap().unwrap();
                for i in (0..len).rev() {
                    assert_eq!(reader.read(i).unwrap(), values[i]);
                }
                let clone = reader.clone();
                for i in (0..len).step_by(137) {
                    assert_eq!(clone.read(i).unwrap(), values[i]);
                }
                assert!(reader.read(len).is_err());
                assert_eq!(opens.load(Ordering::Relaxed), 1);
                assert!(PackedNormReader::open(&source, None).unwrap().is_some());
                assert_eq!(opens.load(Ordering::Relaxed), 1);
                if len > 2 {
                    let clipped = PackedNormReader::open(&source, Some(1..len - 1))
                        .unwrap()
                        .unwrap();
                    assert_eq!(clipped.read(1).unwrap(), values[1]);
                    assert_eq!(clipped.read(len - 2).unwrap(), values[len - 2]);
                    assert!(clipped.read(0).is_err());
                    assert!(clipped.read(len - 1).is_err());
                }
            }
        }
    }

    #[test]
    fn malformed_streams() {
        for bytes in [vec![0], vec![0; FOOTER_LEN]] {
            let source = FileSlice::from(bytes);
            let source = Arc::new(PackedNormSource::new(DeferredFileSlice::new(move || {
                Ok(source.clone())
            })));
            assert!(PackedNormReader::open(&source, None).is_err());
        }
        let mut bytes = Vec::new();
        let mut writer = PackedNormWriter::new(&mut bytes);
        writer.write_all(&[1; 129]).unwrap();
        writer.finish().unwrap();
        bytes[0] = 9;
        let source = FileSlice::from(bytes);
        let source = Arc::new(PackedNormSource::new(DeferredFileSlice::new(move || {
            Ok(source.clone())
        })));
        assert!(PackedNormReader::open(&source, None)
            .unwrap()
            .unwrap()
            .read(0)
            .is_err());
    }
}
