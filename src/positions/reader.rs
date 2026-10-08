use std::io;
use std::ops::Range;

use common::{BinarySerializable, HasLen, VInt};

use crate::directory::{FileSlice, OwnedBytes};
use crate::positions::COMPRESSION_BLOCK_SIZE;
use crate::postings::compression::BlockDecoder;

/// When accessing the positions of a term, we get a positions_idx from the `Terminfo`.
/// This means we need to skip to the `nth` position efficiently.
///
/// Blocks are compressed using bitpacking, so `skip_read` contains the number of bits
/// (values can go from 0 to 32 bits) required to decompress every block.
///
/// A given block obviously takes `(128 x  num_bit_for_the_block / num_bits_in_a_byte)`,
/// so skipping a block without decompressing it is just a matter of advancing that many
/// bytes.

#[derive(Clone)]
pub struct PositionReader {
    bit_widths: OwnedBytes,
    positions: PositionData,
    positions_byte_offset: usize,

    // Wrapping prefix sums let documents borrow positions from a decoded block.
    // Subtracting the preceding value recovers each document's starting point.
    block_decoder: BlockDecoder,

    // offset, expressed in positions, for the first position of the block currently loaded
    // block_offset is a multiple of COMPRESSION_BLOCK_SIZE.
    block_offset: u64,
    // offset, expressed in positions, for the position of the first block encoded
    // in `positions`, and if bitpacked, compressed using the bitwidth in `bit_widths`.
    //
    // As we advance, anchor increases simultaneously with bit_widths and positions get consumed.
    anchor_offset: u64,

    // These are copies used for .reset().
    original_bit_widths: OwnedBytes,
    original_positions_byte_offset: usize,
}

#[derive(Clone)]
enum PositionData {
    Eager(OwnedBytes),
    Lazy {
        file: FileSlice,
        buffer: OwnedBytes,
        buffer_start: usize,
    },
}

impl PositionData {
    fn with_bytes<T>(
        &mut self,
        range: Range<usize>,
        consume: impl FnOnce(&[u8]) -> T,
    ) -> io::Result<T> {
        if range.is_empty() {
            return Ok(consume(&[]));
        }
        match self {
            PositionData::Eager(bytes) => Ok(consume(&bytes.as_slice()[range])),
            PositionData::Lazy {
                file,
                buffer,
                buffer_start,
            } => {
                let buffer_end = *buffer_start + buffer.len();
                if range.start < *buffer_start || range.end > buffer_end {
                    let start = file.storage_block_range(range.start).unwrap().start;
                    let end = file.storage_block_range(range.end - 1).unwrap().end;
                    let overlap_start = start.max(*buffer_start);
                    let overlap_end = end.min(buffer_end);
                    *buffer = if overlap_start < overlap_end {
                        let mut bytes = Vec::with_capacity(end - start);
                        if start < overlap_start {
                            bytes.extend_from_slice(&file.read_bytes_slice(start..overlap_start)?);
                        }
                        bytes.extend_from_slice(
                            &buffer[overlap_start - *buffer_start..overlap_end - *buffer_start],
                        );
                        if overlap_end < end {
                            bytes.extend_from_slice(&file.read_bytes_slice(overlap_end..end)?);
                        }
                        OwnedBytes::new(bytes)
                    } else {
                        file.read_bytes_slice(start..end)?
                    };
                    *buffer_start = start;
                }
                Ok(consume(
                    &buffer[range.start - *buffer_start..range.end - *buffer_start],
                ))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            PositionData::Eager(bytes) => bytes.len(),
            PositionData::Lazy { file, .. } => file.len(),
        }
    }
}

impl PositionReader {
    /// Opens term positions already loaded into contiguous memory.
    pub fn open(mut positions_data: OwnedBytes) -> io::Result<PositionReader> {
        let num_positions_bitpacked_blocks = VInt::deserialize(&mut positions_data)?.0 as usize;
        let (bit_widths, positions) = positions_data.split(num_positions_bitpacked_blocks);
        Ok(Self::from_parts(
            bit_widths,
            PositionData::Eager(positions),
            0,
        ))
    }

    /// Opens a term position stream, loading compressed blocks on demand for block-backed files.
    pub(crate) fn open_file_slice(positions_data: FileSlice) -> io::Result<PositionReader> {
        if positions_data.storage_block_len().is_none() {
            return Self::open(positions_data.read_bytes()?);
        }
        let mut positions = PositionData::Lazy {
            file: positions_data,
            buffer: OwnedBytes::empty(),
            buffer_start: 0,
        };
        let (num_positions_bitpacked_blocks, header_len) = positions
            .with_bytes(0..positions.len().min(10), |mut bytes| {
                VInt::deserialize_with_size(&mut bytes)
            })??;
        let num_positions_bitpacked_blocks = usize::try_from(num_positions_bitpacked_blocks.0)
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "position block count is too large",
                )
            })?;
        let positions_start = header_len
            .checked_add(num_positions_bitpacked_blocks)
            .filter(|&end| end <= positions.len())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "truncated position bit widths")
            })?;
        let bit_widths = positions.with_bytes(header_len..positions_start, |bytes| {
            OwnedBytes::new(bytes.to_vec())
        })?;
        Ok(Self::from_parts(bit_widths, positions, positions_start))
    }

    fn from_parts(
        bit_widths: OwnedBytes,
        positions: PositionData,
        positions_byte_offset: usize,
    ) -> PositionReader {
        PositionReader {
            bit_widths: bit_widths.clone(),
            positions,
            positions_byte_offset,
            block_decoder: BlockDecoder::default(),
            block_offset: i64::MAX as u64,
            anchor_offset: 0u64,
            original_bit_widths: bit_widths,
            original_positions_byte_offset: positions_byte_offset,
        }
    }

    fn reset(&mut self) {
        self.bit_widths = self.original_bit_widths.clone();
        self.positions_byte_offset = self.original_positions_byte_offset;
        self.block_offset = i64::MAX as u64;
        self.anchor_offset = 0u64;
    }

    /// Advance from num_blocks bitpacked blocks.
    ///
    /// Panics if there are not that many remaining blocks.
    fn advance_num_blocks(&mut self, num_blocks: usize) {
        let num_bits: usize = self.bit_widths.as_ref()[..num_blocks]
            .iter()
            .cloned()
            .map(|num_bits| num_bits as usize)
            .sum();
        let num_bytes_to_skip = num_bits * COMPRESSION_BLOCK_SIZE / 8;
        self.bit_widths.advance(num_blocks);
        self.positions_byte_offset += num_bytes_to_skip;
        self.anchor_offset += (num_blocks * COMPRESSION_BLOCK_SIZE) as u64;
    }

    /// block_rel_id is counted relatively to the anchor.
    /// block_rel_id = 0 means the anchor block.
    /// block_rel_id = i means the ith block after the anchor block.
    fn load_block(&mut self, block_rel_id: usize) {
        let bit_widths = self.bit_widths.as_slice();
        let byte_offset: usize = bit_widths[0..block_rel_id]
            .iter()
            .map(|&b| b as usize)
            .sum::<usize>()
            * COMPRESSION_BLOCK_SIZE
            / 8;
        let start = self.positions_byte_offset + byte_offset;
        let is_bitpacked = bit_widths.len() > block_rel_id;
        let end = if is_bitpacked {
            start + bit_widths[block_rel_id] as usize * COMPRESSION_BLOCK_SIZE / 8
        } else {
            self.positions.len()
        };
        let block_decoder = &mut self.block_decoder;
        self.positions
            .with_bytes(start..end, |compressed_data| {
                if is_bitpacked {
                    block_decoder.uncompress_block_sorted(
                        compressed_data,
                        0,
                        bit_widths[block_rel_id],
                        false,
                    );
                } else {
                    block_decoder.uncompress_vint_sorted_until_end(compressed_data);
                }
            })
            .expect("position data became unreadable after the reader was opened");
        self.block_offset = self.anchor_offset + (block_rel_id * COMPRESSION_BLOCK_SIZE) as u64;
    }

    /// Fills a buffer with the positions `[offset..offset+output.len())` integers.
    ///
    /// This function is optimized to be called with increasing values of `offset`.
    pub fn read(&mut self, offset: u64, output: &mut [u32]) {
        let mut cursor = 0;
        self.read_blocks(offset, output.len(), |block, mut previous| {
            for &value in block {
                output[cursor] = value.wrapping_sub(previous);
                previous = value;
                cursor += 1;
            }
        });
    }

    pub(crate) fn append_positions_with_offset(
        &mut self,
        offset: u64,
        count: usize,
        mut position_offset: u32,
        output: &mut Vec<u32>,
    ) {
        output.reserve(count);
        self.read_blocks(offset, count, |block, previous| {
            let base = position_offset.wrapping_sub(previous);
            output.extend(block.iter().map(|&value| value.wrapping_add(base)));
            if let Some(&last) = block.last() {
                position_offset = last.wrapping_add(base);
            }
        });
    }

    fn read_blocks(
        &mut self,
        mut offset: u64,
        mut count: usize,
        mut consume: impl FnMut(&[u32], u32),
    ) {
        if offset < self.anchor_offset {
            self.reset();
        }
        let delta_to_block_offset = offset as i64 - self.block_offset as i64;
        if !(0..128).contains(&delta_to_block_offset) {
            // The first position is not within the first block.
            // (Note that it could be before or after)
            // We need to possibly skip a few blocks, and decompress the first relevant block.
            let delta_to_anchor_offset = offset - self.anchor_offset;
            let num_blocks_to_skip =
                (delta_to_anchor_offset / (COMPRESSION_BLOCK_SIZE as u64)) as usize;
            self.advance_num_blocks(num_blocks_to_skip);
            self.load_block(0);
        } else {
            // The request offset is within the loaded block.
            // We still need to advance anchor_offset to our current block.
            let num_blocks_to_skip =
                ((self.block_offset - self.anchor_offset) / COMPRESSION_BLOCK_SIZE as u64) as usize;
            self.advance_num_blocks(num_blocks_to_skip);
        }

        // At this point, the block containing offset is loaded, and anchor has
        // been updated to point to it as well.
        for i in 1.. {
            let offset_in_block = (offset as usize) % COMPRESSION_BLOCK_SIZE;
            let remaining_in_block = COMPRESSION_BLOCK_SIZE - offset_in_block;
            let len = count.min(remaining_in_block);
            let positions = self.block_decoder.output_array();
            let previous = offset_in_block.checked_sub(1).map_or(0, |i| positions[i]);
            consume(&positions[offset_in_block..][..len], previous);
            count -= len;
            if count == 0 {
                break;
            }
            // we load block #i if necessary.
            offset += remaining_in_block as u64;
            self.load_block(i);
        }
    }
}
