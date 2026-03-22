use std::io;
use std::io::Write;
use std::sync::{Arc, OnceLock};

use common::file_slice::FileSlice;
use common::{BinarySerializable, CountingWriter, DeserializeFrom, HasLen, OwnedBytes};
use fastdivide::DividerU64;
use tantivy_bitpacker::{BitPacker, BitUnpacker, compute_num_bits};

use crate::MonotonicallyMappableToU64;
use crate::column_values::u64_based::line::Line;
use crate::column_values::u64_based::{ColumnCodec, ColumnCodecEstimator, ColumnStats};
use crate::column_values::{ColumnValues, VecColumn};

const BLOCK_SIZE: u32 = 512u32;

#[derive(Debug, Default)]
struct Block {
    line: Line,
    bit_unpacker: BitUnpacker,
}

impl BinarySerializable for Block {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        self.line.serialize(writer)?;
        self.bit_unpacker.bit_width().serialize(writer)?;
        Ok(())
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let line = Line::deserialize(reader)?;
        let bit_width = u8::deserialize(reader)?;
        Ok(Block {
            line,
            bit_unpacker: BitUnpacker::new(bit_width),
        })
    }
}

fn compute_num_blocks(num_vals: u32) -> u32 {
    num_vals.div_ceil(BLOCK_SIZE)
}

pub struct BlockwiseLinearEstimator {
    block: Vec<u64>,
    values_num_bytes: u64,
    meta_num_bytes: u64,
}

impl Default for BlockwiseLinearEstimator {
    fn default() -> Self {
        Self {
            block: Vec::with_capacity(BLOCK_SIZE as usize),
            values_num_bytes: 0u64,
            meta_num_bytes: 0u64,
        }
    }
}

impl BlockwiseLinearEstimator {
    fn flush_block_estimate(&mut self) {
        if self.block.is_empty() {
            return;
        }
        let column = VecColumn::from(std::mem::take(&mut self.block));
        let line = Line::train(&column);
        self.block = column.into();

        let mut max_value = 0u64;
        for (i, buffer_val) in self.block.iter().enumerate() {
            let interpolated_val = line.eval(i as u32);
            let val = buffer_val.wrapping_sub(interpolated_val);
            max_value = val.max(max_value);
        }
        let bit_width = compute_num_bits(max_value) as usize;
        self.values_num_bytes += (bit_width * self.block.len() + 7) as u64 / 8;
        self.meta_num_bytes += 1 + line.num_bytes();
    }
}

impl ColumnCodecEstimator for BlockwiseLinearEstimator {
    fn collect(&mut self, value: u64) {
        self.block.push(value);
        if self.block.len() == BLOCK_SIZE as usize {
            self.flush_block_estimate();
            self.block.clear();
        }
    }
    fn estimate(&self, stats: &ColumnStats) -> Option<u64> {
        let mut estimate = 4 + stats.num_bytes() + self.meta_num_bytes + self.values_num_bytes;
        if stats.gcd.get() > 1 {
            let estimate_gain_from_gcd =
                (stats.gcd.get() as f32).log2().floor() * stats.num_rows as f32 / 8.0f32;
            estimate = estimate.saturating_sub(estimate_gain_from_gcd as u64);
        }
        Some(estimate)
    }

    fn finalize(&mut self) {
        self.flush_block_estimate();
    }

    fn serialize(
        &self,
        stats: &ColumnStats,
        mut vals: &mut dyn Iterator<Item = u64>,
        wrt: &mut dyn Write,
    ) -> io::Result<()> {
        stats.serialize(wrt)?;
        let mut buffer = Vec::with_capacity(BLOCK_SIZE as usize);
        let num_blocks = compute_num_blocks(stats.num_rows) as usize;
        let mut blocks = Vec::with_capacity(num_blocks);

        let mut bit_packer = BitPacker::new();

        let gcd_divider = DividerU64::divide_by(stats.gcd.get());

        for _ in 0..num_blocks {
            buffer.clear();
            buffer.extend(
                (&mut vals)
                    .map(MonotonicallyMappableToU64::to_u64)
                    .take(BLOCK_SIZE as usize),
            );

            for buffer_val in buffer.iter_mut() {
                *buffer_val = gcd_divider.divide(*buffer_val - stats.min_value);
            }

            let line = Line::train(&VecColumn::from(buffer.to_vec()));

            assert!(!buffer.is_empty());

            for (i, buffer_val) in buffer.iter_mut().enumerate() {
                let interpolated_val = line.eval(i as u32);
                *buffer_val = buffer_val.wrapping_sub(interpolated_val);
            }

            let bit_width = buffer.iter().copied().map(compute_num_bits).max().unwrap();

            for &buffer_val in &buffer {
                bit_packer.write(buffer_val, bit_width, wrt)?;
            }

            blocks.push(Block {
                line,
                bit_unpacker: BitUnpacker::new(bit_width),
            });
        }

        bit_packer.close(wrt)?;

        assert_eq!(blocks.len(), num_blocks);

        let mut counting_wrt = CountingWriter::wrap(wrt);
        for block in &blocks {
            block.serialize(&mut counting_wrt)?;
        }
        let footer_len = counting_wrt.written_bytes();
        (footer_len as u32).serialize(&mut counting_wrt)?;

        Ok(())
    }
}

pub struct BlockwiseLinearCodec;

impl ColumnCodec<u64> for BlockwiseLinearCodec {
    type ColumnValues = BlockwiseLinearReader;

    type Estimator = BlockwiseLinearEstimator;

    fn load(file_slice: FileSlice) -> io::Result<Self::ColumnValues> {
        let (stats, body) = ColumnStats::deserialize_from_tail(file_slice)?;

        let (_, footer_len_bytes) = body.clone().split_from_end(4);

        let footer_len: u32 = footer_len_bytes.read_bytes()?.as_slice().deserialize()?;
        let (data, footer_file) = body.split_from_end(footer_len as usize + 4);

        let footer_bytes = footer_file.read_bytes()?;
        let mut footer_cursor: &[u8] = footer_bytes.as_slice();
        let num_blocks = compute_num_blocks(stats.num_rows) as usize;

        let mut lines = Vec::with_capacity(num_blocks);
        let mut bit_unpackers = Vec::with_capacity(num_blocks);
        let mut data_file_slices: Vec<(FileSlice, OnceLock<OwnedBytes>)> =
            Vec::with_capacity(num_blocks);
        let mut start_offset = 0usize;

        for _ in 0..num_blocks {
            let line = Line::deserialize(&mut footer_cursor)?;
            let bit_width = u8::deserialize(&mut footer_cursor)?;
            let len = (bit_width as usize) * BLOCK_SIZE as usize / 8;

            lines.push(line);
            bit_unpackers.push(BitUnpacker::new(bit_width));
            data_file_slices.push((
                data.slice(start_offset..(start_offset + len).min(data.len())),
                OnceLock::new(),
            ));

            start_offset += len;
        }

        Ok(BlockwiseLinearReader {
            lines: lines.into(),
            bit_unpackers: bit_unpackers.into(),
            data_file_slices: data_file_slices.into(),
            stats,
        })
    }
}

pub struct BlockwiseLinearReader {
    lines: Arc<[Line]>,
    bit_unpackers: Arc<[BitUnpacker]>,
    data_file_slices: Arc<[(FileSlice, OnceLock<OwnedBytes>)]>,
    stats: ColumnStats,
}

impl Clone for BlockwiseLinearReader {
    fn clone(&self) -> Self {
        BlockwiseLinearReader {
            lines: self.lines.clone(),
            bit_unpackers: self.bit_unpackers.clone(),
            data_file_slices: self.data_file_slices.clone(),
            stats: self.stats.clone(),
        }
    }
}

impl ColumnValues for BlockwiseLinearReader {
    #[inline(always)]
    fn get_val(&self, idx: u32) -> u64 {
        let block_id = (idx / BLOCK_SIZE) as usize;
        let idx_within_block = idx % BLOCK_SIZE;
        let interpoled_val: u64 = self.lines[block_id].eval(idx_within_block);
        let (file_slice, data) = &self.data_file_slices[block_id];
        let block_bytes = data.get_or_init(|| file_slice.read_bytes().unwrap());
        let bitpacked_diff = self.bit_unpackers[block_id].get(idx_within_block, block_bytes);
        self.stats.min_value
            + self
                .stats
                .gcd
                .get()
                .wrapping_mul(interpoled_val.wrapping_add(bitpacked_diff))
    }

    #[inline(always)]
    fn min_value(&self) -> u64 {
        self.stats.min_value
    }

    #[inline(always)]
    fn max_value(&self) -> u64 {
        self.stats.max_value
    }

    #[inline(always)]
    fn num_vals(&self) -> u32 {
        self.stats.num_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column_values::u64_based::tests::create_and_validate;

    #[test]
    fn test_with_codec_data_sets_simple() {
        create_and_validate::<BlockwiseLinearCodec>(
            &[11, 20, 40, 20, 10, 10, 10, 10, 10, 10],
            "simple test",
        )
        .unwrap();
    }

    #[test]
    fn test_with_codec_data_sets_simple_gcd() {
        let (_, actual_compression_rate) = create_and_validate::<BlockwiseLinearCodec>(
            &[10, 20, 40, 20, 10, 10, 10, 10, 10, 10],
            "name",
        )
        .unwrap();
        assert_eq!(actual_compression_rate, 0.175);
    }

    #[test]
    fn test_with_codec_data_sets() {
        let data_sets = crate::column_values::u64_based::tests::get_codec_test_datasets();
        for (mut data, name) in data_sets {
            create_and_validate::<BlockwiseLinearCodec>(&data, name);
            data.reverse();
            create_and_validate::<BlockwiseLinearCodec>(&data, name);
        }
    }

    #[test]
    fn test_blockwise_linear_fast_field_rand() {
        for _ in 0..500 {
            let mut data = (0..1 + rand::random::<u8>() as usize)
                .map(|_| rand::random::<i64>() as u64 / 2)
                .collect::<Vec<_>>();
            create_and_validate::<BlockwiseLinearCodec>(&data, "rand");
            data.reverse();
            create_and_validate::<BlockwiseLinearCodec>(&data, "rand");
        }
    }
}
