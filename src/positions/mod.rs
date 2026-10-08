//! Tantivy can (if instructed to do so in the schema) store the term positions in a given field.
//!
//! This position is expressed as token ordinal. For instance,
//! In "The beauty and the beast", the term "the" appears in position 0 and position 3.
//! This information is useful to run phrase queries.
//!
//! The [position](crate::index::SegmentComponent::Positions) file contains all of the
//! bitpacked positions delta, for all terms of a given field, one term after the other.
//!
//! Each term is encoded independently.
//! Like for posting lists, tantivy relies on simd bitpacking to encode the positions delta in
//! blocks of 128 deltas. Because we rarely have a multiple of 128, the final block encodes
//! the remaining values with variable int encoding.
//!
//! In order to make reading possible, the term delta positions first encode the number of
//! bitpacked blocks, then the bitwidth for each block, then the actual bitpacked blocks and finally
//! the final variable int encoded block.
//!
//! Contrary to postings list, the reader does not have access on the number of positions that is
//! encoded, and instead stops decoding the last block when its byte slice has been entirely read.
//!
//! More formally:
//! * *Positions* := *NumBitPackedBlocks* *BitPackedPositionBlock*^(P/128)
//!   *BitPackedPositionsDeltaBitWidth* *VIntPosDeltas*?
//! * *NumBitPackedBlocks**: := *P* / 128 encoded as a variable byte integer.
//! * *BitPackedPositionBlock* := bit width encoded block of 128 positions delta
//! * *BitPackedPositionsDeltaBitWidth* := (*BitWidth*: u8)^*NumBitPackedBlocks*
//! * *VIntPosDeltas* := *VIntPosDelta*^(*P* % 128).
//!
//! The skip widths encoded separately makes it easy and fast to rapidly skip over n positions.

mod reader;
mod serializer;

use bitpacking::{BitPacker, BitPacker4x};

pub use self::reader::PositionReader;
pub use self::serializer::PositionSerializer;

const COMPRESSION_BLOCK_SIZE: usize = BitPacker4x::BLOCK_LEN;

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashSet;
    use std::io;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use common::HasLen;
    use proptest::prelude::*;
    use proptest::sample::select;

    use super::PositionSerializer;
    use crate::directory::{FileHandle, FileSlice, OwnedBytes};
    use crate::positions::reader::PositionReader;

    fn create_positions_data(vals: &[u32]) -> crate::Result<OwnedBytes> {
        let mut positions_buffer = vec![];
        let mut serializer = PositionSerializer::new(&mut positions_buffer);
        serializer.write_positions_delta(vals);
        serializer.close_term()?;
        serializer.close()?;
        Ok(OwnedBytes::new(positions_buffer))
    }

    #[derive(Debug)]
    struct BlockBackedFile {
        data: Vec<u8>,
        reads: Arc<Mutex<Vec<Range<usize>>>>,
        block_len: usize,
    }

    impl HasLen for BlockBackedFile {
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    impl FileHandle for BlockBackedFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(OwnedBytes::new(self.data[range].to_vec()))
        }

        fn storage_block_len(&self) -> Option<usize> {
            Some(self.block_len)
        }
    }

    fn gen_delta_positions() -> BoxedStrategy<Vec<u32>> {
        select(&[0, 1, 70, 127, 128, 129, 200, 255, 256, 257, 270][..])
            .prop_flat_map(|num_delta_positions| {
                proptest::collection::vec(
                    select(&[1u32, 2u32, 4u32, 8u32, 16u32][..]),
                    num_delta_positions,
                )
            })
            .boxed()
    }

    proptest! {
        #[test]
        fn test_position_delta(delta_positions in gen_delta_positions()) {
            let delta_positions_data = create_positions_data(&delta_positions).unwrap();
            let mut position_reader = PositionReader::open(delta_positions_data).unwrap();
            let mut minibuf = [0u32; 1];
            for (offset, &delta_position) in delta_positions.iter().enumerate() {
                position_reader.read(offset as u64, &mut minibuf[..]);
                assert_eq!(delta_position, minibuf[0]);
            }
        }
    }

    proptest! {
        #[test]
        fn test_append_positions(delta_positions in gen_delta_positions(), position_offset in 0u32..100) {
            let data = create_positions_data(&delta_positions).unwrap();
            let mut reader = PositionReader::open(data).unwrap();
            let mut actual = vec![19, 41];
            reader.append_positions_with_offset(0, delta_positions.len(), position_offset, &mut actual);
            let mut expected = vec![19, 41];
            let mut position = position_offset;
            for delta in delta_positions {
                position += delta;
                expected.push(position);
            }
            prop_assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_append_positions_with_offsets() -> crate::Result<()> {
        let deltas: Vec<u32> = (0..2_049).map(|position| position % 257).collect();
        let data = create_positions_data(&deltas)?;
        let file = FileSlice::new(Arc::new(BlockBackedFile {
            data: data.as_slice().to_vec(),
            reads: Arc::new(Mutex::new(Vec::new())),
            block_len: 17,
        }));
        for mut reader in [
            PositionReader::open(data)?,
            PositionReader::open_file_slice(file)?,
        ] {
            let mut actual = vec![19, 41];
            for (offset, count) in [
                (0, 0),
                (0, 257),
                (127, 130),
                (1_900, 149),
                (512, 400),
                (129, 0),
                (31, 129),
                (2_048, 1),
                (2_048, 1),
                (0, 2_049),
            ] {
                actual.truncate(2);
                reader.append_positions_with_offset(offset as u64, count, 17, &mut actual);
                let mut expected = vec![19, 41];
                let mut position = 17;
                for delta in &deltas[offset..offset + count] {
                    position += delta;
                    expected.push(position);
                }
                assert_eq!(actual, expected, "offset={offset}, count={count}");
            }
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn test_streamed_position_intersection(
            deltas in proptest::collection::vec(0u32..32, 0..600),
            mut candidates in proptest::collection::vec(0u32..10_000, 0..100),
            start_hint in 0usize..1_000,
            count_hint in 0usize..1_000,
            position_offset in 0u32..100,
        ) {
            candidates.sort_unstable();
            let offset = start_hint % (deltas.len() + 1);
            let count = count_hint % (deltas.len() - offset + 1);
            let mut available = std::collections::BTreeMap::<u32, usize>::new();
            let mut position = position_offset;
            for delta in &deltas[offset..offset + count] {
                position += delta;
                *available.entry(position).or_default() += 1;
            }
            let mut expected = Vec::new();
            for &candidate in &candidates {
                if let Some(count) = available.get_mut(&candidate) {
                    if *count > 0 {
                        expected.push(candidate);
                        *count -= 1;
                    }
                }
            }
            let data = create_positions_data(&deltas).unwrap();
            let mut reader = PositionReader::open(data).unwrap();
            for stop_at_first in [false, true, false] {
                let mut actual = candidates.clone();
                let matches = reader.intersect_positions_with_offset(
                    offset as u64, count, position_offset, &mut actual, stop_at_first,
                );
                let expected = if stop_at_first { &expected[..expected.len().min(1)] } else { &expected };
                prop_assert_eq!(matches, expected.len());
                prop_assert_eq!(actual.as_slice(), expected);
            }
        }
    }

    #[test]
    fn test_streamed_positions_stop_reading() -> crate::Result<()> {
        let data = create_positions_data(&vec![1; 4_097])?;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let file = FileSlice::new(Arc::new(BlockBackedFile {
            data: data.as_slice().to_vec(),
            reads: reads.clone(),
            block_len: 17,
        }));
        let mut reader = PositionReader::open_file_slice(file)?;
        let mut candidates = vec![1, 4_096];
        assert_eq!(
            reader.intersect_positions_with_offset(0, 4_097, 0, &mut candidates, true),
            1
        );
        assert_eq!(candidates, [1]);
        let early_bytes: usize = reads.lock().unwrap().iter().map(Range::len).sum();
        for (offset, count, target) in [
            (2_000, 1_000, 257),
            (127, 270, 129),
            (4_096, 1, 1),
            (0, 4_097, 4_097),
        ] {
            let mut candidates = vec![target];
            assert_eq!(
                reader.intersect_positions_with_offset(offset, count, 0, &mut candidates, false),
                1
            );
            assert_eq!(candidates, [target]);
        }
        assert!(early_bytes < data.len());
        Ok(())
    }

    #[test]
    fn test_position_block_prefix_sums_wrap() -> crate::Result<()> {
        let deltas: Vec<_> = [u32::MAX - 7, 0, 4, 5, 7, 11]
            .into_iter()
            .cycle()
            .take(600)
            .collect();
        let data = create_positions_data(&deltas)?;
        let mut reader = PositionReader::open(data)?;
        let mut raw = vec![0; deltas.len()];
        reader.read(0, &mut raw);
        assert_eq!(raw, deltas);
        for offset in (0..deltas.len()).step_by(3) {
            let mut expected = Vec::new();
            let mut position = 2;
            for &delta in &deltas[offset..offset + 3] {
                position += delta;
                expected.push(position);
            }
            let mut actual = Vec::new();
            reader.append_positions_with_offset(offset as u64, 3, 2, &mut actual);
            assert_eq!(actual, expected);
            let mut candidates = expected.clone();
            assert_eq!(
                reader.intersect_positions_with_offset(offset as u64, 3, 2, &mut candidates, false),
                3
            );
            assert_eq!(candidates, expected);
        }
        Ok(())
    }

    #[test]
    fn test_position_read() -> crate::Result<()> {
        let position_deltas: Vec<u32> = (0..1000).collect();
        let positions_data = create_positions_data(&position_deltas[..])?;
        assert_eq!(positions_data.len(), 1224);
        let mut position_reader = PositionReader::open(positions_data)?;
        for &n in &[1, 10, 127, 128, 130, 312] {
            let mut v = vec![0u32; n];
            position_reader.read(0, &mut v[..]);
            for i in 0..n {
                assert_eq!(position_deltas[i], i as u32);
            }
        }
        Ok(())
    }

    #[test]
    fn test_empty_position() -> crate::Result<()> {
        let mut positions_buffer = vec![];
        let mut serializer = PositionSerializer::new(&mut positions_buffer);
        serializer.close_term()?;
        serializer.close()?;
        let position_delta = OwnedBytes::new(positions_buffer);
        assert!(PositionReader::open(position_delta).is_ok());
        Ok(())
    }

    #[test]
    fn test_multiple_write_positions() -> crate::Result<()> {
        let mut positions_buffer = vec![];
        let mut serializer = PositionSerializer::new(&mut positions_buffer);
        serializer.write_positions_delta(&[1u32, 12u32]);
        serializer.write_positions_delta(&[4u32, 17u32]);
        serializer.write_positions_delta(&[443u32]);
        serializer.close_term()?;
        serializer.close()?;
        let position_delta = OwnedBytes::new(positions_buffer);
        let mut output_delta_pos_buffer = [0u32; 5];
        let mut position_reader = PositionReader::open(position_delta)?;
        position_reader.read(0, &mut output_delta_pos_buffer[..]);
        assert_eq!(
            &output_delta_pos_buffer[..],
            &[1u32, 12u32, 4u32, 17u32, 443u32]
        );
        Ok(())
    }

    #[test]
    fn test_position_read_with_offset() -> crate::Result<()> {
        let position_deltas: Vec<u32> = (0..1000).collect();
        let positions_data = create_positions_data(&position_deltas[..])?;
        assert_eq!(positions_data.len(), 1224);
        let mut position_reader = PositionReader::open(positions_data)?;
        for &offset in &[1u64, 10u64, 127u64, 128u64, 130u64, 312u64] {
            for &len in &[1, 10, 130, 500] {
                let mut v = vec![0u32; len];
                position_reader.read(offset, &mut v[..]);
                for i in 0..len {
                    assert_eq!(v[i], i as u32 + offset as u32);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_position_blocks_are_loaded_lazily() -> crate::Result<()> {
        let position_deltas: Vec<u32> = (0..2_000).map(|position| position % 257).collect();
        let positions_data = create_positions_data(&position_deltas)?;
        let positions_len = positions_data.len();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let file = FileSlice::new(Arc::new(BlockBackedFile {
            data: positions_data.as_slice().to_vec(),
            reads: reads.clone(),
            block_len: 17,
        }));
        let mut position_reader = PositionReader::open_file_slice(file)?;

        assert!(reads
            .lock()
            .unwrap()
            .iter()
            .all(|range| range.len() < positions_len));

        for &(offset, len) in &[(0, 300), (127, 257), (900, 600), (31, 129)] {
            let mut output = vec![0; len];
            position_reader.read(offset, &mut output);
            assert_eq!(
                output,
                position_deltas[offset as usize..offset as usize + len]
            );
        }
        assert!(reads
            .lock()
            .unwrap()
            .iter()
            .all(|range| range.len() < positions_len));
        Ok(())
    }

    #[test]
    fn test_position_storage_pages_are_reused() -> crate::Result<()> {
        let mut repeated_pages = 0;
        for (name, num_positions, step, block_len, term_offset, value) in [
            ("tiny vint", 3, 1, 8192, 123, 255),
            ("tiny bitpacked", 256, 128, 8192, 123, 255),
            ("dense aligned", 262_161, 128, 8192, 0, 255),
            ("dense unaligned", 262_161, 128, 8192, 123, 255),
            (
                "dense non-power-of-two pages",
                262_161,
                128,
                8136,
                16_395,
                255,
            ),
            ("term straddling pages", 256, 128, 8192, 8191, 255),
            ("sparse", 262_161, 16_384, 8192, 123, 255),
            ("multi-page metadata", 1_049_233, 128, 8192, 8191, 255),
            ("small pages", 2_049, 128, 17, 15, 255),
            ("zero bit widths", 32_785, 128, 8192, 123, 0),
        ] {
            let position_deltas = vec![value; num_positions];
            let positions_data = create_positions_data(&position_deltas)?;
            let term_end = term_offset + positions_data.len();
            let mut data = vec![0; term_offset];
            data.extend_from_slice(&positions_data);
            data.extend_from_slice(&[0; 32]);
            let reads = Arc::new(Mutex::new(Vec::new()));
            let file = FileSlice::new(Arc::new(BlockBackedFile {
                data,
                reads: reads.clone(),
                block_len,
            }))
            .slice(term_offset / 2..term_end)
            .slice_from(term_offset - term_offset / 2);
            let mut reader = PositionReader::open_file_slice(file)?;
            for offset in (0..num_positions).step_by(step) {
                let mut output = vec![0; 128.min(num_positions - offset)];
                reader.read(offset as u64, &mut output);
                assert_eq!(output, position_deltas[offset..offset + output.len()]);
            }
            let reads = reads.lock().unwrap();
            let mut pages = HashSet::new();
            let mut page_accesses = 0;
            for range in reads.iter().filter(|range| !range.is_empty()) {
                assert!(range.start >= term_offset && range.end <= term_end);
                assert!(range.start == term_offset || range.start % block_len == 0);
                assert!(range.end == term_end || range.end % block_len == 0);
                for page in range.start / block_len..=(range.end - 1) / block_len {
                    page_accesses += 1;
                    pages.insert(page);
                }
            }
            eprintln!(
                "{name}: {} reads, {page_accesses} page accesses, {} unique pages, {} bytes",
                reads.len(),
                pages.len(),
                reads.iter().map(|range| range.len()).sum::<usize>()
            );
            repeated_pages += page_accesses - pages.len();
            if step <= 128 {
                assert_eq!(
                    pages.len(),
                    (term_end - 1) / block_len - term_offset / block_len + 1
                );
            } else {
                assert!(pages.len() < (term_end - term_offset) / block_len);
            }
        }
        assert_eq!(repeated_pages, 0);
        Ok(())
    }

    #[test]
    fn test_position_page_cache_survives_clone_and_reset() -> crate::Result<()> {
        let position_deltas: Vec<u32> = (0..2_049).map(|position| position % 257).collect();
        let positions_data = create_positions_data(&position_deltas)?;
        let mut data = vec![0; 123];
        data.extend_from_slice(&positions_data);
        let reads = Arc::new(Mutex::new(Vec::new()));
        let file = FileSlice::new(Arc::new(BlockBackedFile {
            data,
            reads: reads.clone(),
            block_len: 8192,
        }))
        .slice_from(123);
        let mut reader = PositionReader::open_file_slice(file)?;
        reader.read(1_024, &mut [0; 128]);
        for mut reader in [reader.clone(), reader] {
            for (offset, len) in [(0, 256), (1_000, 500), (127, 1_922), (256, 1)] {
                let mut output = vec![0; len];
                reader.read(offset as u64, &mut output);
                assert_eq!(output, position_deltas[offset..offset + len]);
            }
        }
        assert_eq!(reads.lock().unwrap().len(), 1);
        Ok(())
    }

    #[test]
    fn test_position_read_after_skip() -> crate::Result<()> {
        let position_deltas: Vec<u32> = (0..1_000).collect();
        let positions_data = create_positions_data(&position_deltas[..])?;
        assert_eq!(positions_data.len(), 1224);

        let mut position_reader = PositionReader::open(positions_data)?;
        let mut buf = [0u32; 7];
        let mut c = 0;

        let mut offset = 0;
        for _ in 0..100 {
            position_reader.read(offset, &mut buf);
            position_reader.read(offset, &mut buf);
            offset += 7;
            for &el in &buf {
                assert_eq!(c, el);
                c += 1;
            }
        }
        Ok(())
    }

    #[test]
    fn test_position_reread_anchor_different_than_block() -> crate::Result<()> {
        let positions_delta: Vec<u32> = (0..2_000_000).collect();
        let positions_data = create_positions_data(&positions_delta[..])?;
        assert_eq!(positions_data.len(), 5003499);
        let mut position_reader = PositionReader::open(positions_data)?;
        let mut buf = [0u32; 256];
        position_reader.read(128, &mut buf);
        for i in 0..256 {
            assert_eq!(buf[i], (128 + i) as u32);
        }
        position_reader.read(128, &mut buf);
        for i in 0..256 {
            assert_eq!(buf[i], (128 + i) as u32);
        }
        Ok(())
    }

    #[test]
    fn test_position_requesting_passed_block() -> crate::Result<()> {
        let positions_delta: Vec<u32> = (0..512).collect();
        let positions_data = create_positions_data(&positions_delta[..])?;
        assert_eq!(positions_data.len(), 533);
        let mut buf = [0u32; 1];
        let mut position_reader = PositionReader::open(positions_data)?;
        position_reader.read(230, &mut buf);
        assert_eq!(buf[0], 230);
        position_reader.read(9, &mut buf);
        assert_eq!(buf[0], 9);
        Ok(())
    }

    #[test]
    fn test_position() -> crate::Result<()> {
        const CONST_VAL: u32 = 9u32;
        let positions_delta: Vec<u32> = std::iter::repeat_n(CONST_VAL, 2_000_000).collect();
        let positions_data = create_positions_data(&positions_delta[..])?;
        assert_eq!(positions_data.len(), 1_015_627);
        let mut position_reader = PositionReader::open(positions_data)?;
        let mut buf = [0u32; 1];
        position_reader.read(0, &mut buf);
        assert_eq!(buf[0], CONST_VAL);
        Ok(())
    }

    #[test]
    fn test_position_advance() -> crate::Result<()> {
        let positions_delta: Vec<u32> = (0..2_000_000).collect();
        let positions_data = create_positions_data(&positions_delta[..])?;
        assert_eq!(positions_data.len(), 5_003_499);
        for &offset in &[
            10,
            128 * 1024,
            128 * 1024 - 1,
            128 * 1024 + 7,
            128 * 10 * 1024 + 10,
        ] {
            let mut position_reader = PositionReader::open(positions_data.clone())?;
            let mut buf = [0u32; 1];
            position_reader.read(offset, &mut buf);
            assert_eq!(buf[0], offset as u32);
        }
        Ok(())
    }
}
