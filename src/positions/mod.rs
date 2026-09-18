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

thread_local! {
    pub(crate) static LAZY_POSITION_READS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn set_lazy_position_reads(enabled: bool) {
    LAZY_POSITION_READS.set(enabled);
}

use bitpacking::{BitPacker, BitPacker4x};

pub use self::reader::PositionReader;
pub use self::serializer::PositionSerializer;

const COMPRESSION_BLOCK_SIZE: usize = BitPacker4x::BLOCK_LEN;

#[cfg(test)]
pub(crate) mod tests {

    use proptest::prelude::*;
    use proptest::sample::select;

    use super::PositionSerializer;
    use crate::directory::OwnedBytes;
    use crate::positions::reader::PositionReader;

    fn create_positions_data(vals: &[u32]) -> crate::Result<OwnedBytes> {
        let mut positions_buffer = vec![];
        let mut serializer = PositionSerializer::new(&mut positions_buffer);
        serializer.write_positions_delta(vals);
        serializer.close_term()?;
        serializer.close()?;
        Ok(OwnedBytes::new(positions_buffer))
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

    #[test]
    fn test_lazy_position_reads_match_eager() -> crate::Result<()> {
        use std::ops::Range;
        use std::sync::{Arc, Mutex};

        use common::HasLen;

        use crate::directory::{FileHandle, FileSlice};

        #[derive(Debug)]
        struct TrackedFile {
            bytes: OwnedBytes,
            reads: Arc<Mutex<Vec<Range<usize>>>>,
        }
        impl HasLen for TrackedFile {
            fn len(&self) -> usize {
                self.bytes.len()
            }
        }
        impl FileHandle for TrackedFile {
            fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
                assert!(
                    !range.is_empty(),
                    "empty reads are not supported by every FileHandle"
                );
                self.reads.lock().unwrap().push(range.clone());
                Ok(self.bytes.slice(range))
            }
        }

        for count in [0, 1, 127, 128, 129, 255, 256, 257, 10_003] {
            for pattern in 0..3 {
                let values: Vec<_> = (0..count)
                    .map(|i| match pattern {
                        0 => 0,
                        1 => (i * 37 % 4096) as u32,
                        _ if i / 128 % 2 == 0 => 0,
                        _ => [1, 127, 128, 65535, u32::MAX][i % 5],
                    })
                    .collect();
                let bytes = create_positions_data(&values)?;
                let reads = Arc::new(Mutex::new(Vec::new()));
                let file = FileSlice::new(Arc::new(TrackedFile {
                    bytes: bytes.clone(),
                    reads: reads.clone(),
                }));
                let mut lazy = PositionReader::open_from_file(file)?;
                let mut eager = PositionReader::open(bytes.clone())?;
                if count == 10_003 && pattern == 1 {
                    let opened: usize = reads.lock().unwrap().iter().map(|range| range.len()).sum();
                    assert!(opened < bytes.len() / 20);
                    reads.lock().unwrap().clear();
                    let mut one = [0];
                    lazy.read(8192, &mut one);
                    assert_eq!(one[0], values[8192]);
                    let fetched: usize =
                        reads.lock().unwrap().iter().map(|range| range.len()).sum();
                    assert!(fetched < bytes.len() / 20);
                    let read_count = reads.lock().unwrap().len();
                    lazy.read(8193, &mut one);
                    assert_eq!(one[0], values[8193]);
                    assert_eq!(reads.lock().unwrap().len(), read_count);
                }

                let mut offsets = vec![0, count / 2, count.saturating_sub(1)];
                offsets.extend([127, 128, 129, 255, 256]);
                offsets.retain(|&offset| offset < count);
                offsets.extend(offsets.clone().into_iter().rev());
                for offset in offsets {
                    for requested in [0, 1, 127, 128, 129, 399, count] {
                        let len = requested.min(count - offset);
                        let mut actual = vec![0; len];
                        let mut expected = vec![0; len];
                        lazy.read(offset as u64, &mut actual);
                        eager.read(offset as u64, &mut expected);
                        assert_eq!(actual, expected);
                        assert_eq!(actual, values[offset..offset + len]);

                        let mut cloned = lazy.clone();
                        let clone_offset = count - len;
                        cloned.read(clone_offset as u64, &mut actual);
                        assert_eq!(actual, values[clone_offset..]);
                        cloned.read(offset as u64, &mut actual);
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
        Ok(())
    }
}
