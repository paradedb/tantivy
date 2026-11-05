pub(crate) mod v2;
pub(crate) mod v3;

use std::io::{self, Read, Write};
use std::ops::Range;
use std::sync::Arc;

use common::file_slice::FileSlice;
use common::{BinarySerializable, FixedSize};
use tantivy_fst::{Automaton, MapBuilder};

use crate::{TermOrdinal, common_prefix_len};

#[derive(Debug, Clone)]
pub enum SSTableIndex {
    V2(v2::SSTableIndex),
    V3(v3::SSTableIndexV3),
    V3Empty(v3::SSTableIndexV3Empty),
}

impl SSTableIndex {
    pub(crate) fn open(
        version: u32,
        index_offset: u64,
        index_bytes: FileSlice,
    ) -> io::Result<Self> {
        // Keep bulk reads for backends without page-granular access.
        let index_bytes =
            if index_bytes.num_bytes() > 0 && index_bytes.storage_block_ord(0).is_none() {
                FileSlice::new(Arc::new(index_bytes.read_bytes()?))
            } else {
                index_bytes
            };
        let index =
            match version {
                2 => SSTableIndex::V2(v2::SSTableIndex::load(index_bytes.read_bytes()?).map_err(
                    |_| io::Error::new(io::ErrorKind::InvalidData, "SSTable corruption"),
                )?),
                3 => {
                    let (index_bytes, footerv3_len_bytes) = index_bytes.split_from_end(8);
                    let store_offset = u64::deserialize(&mut footerv3_len_bytes.read_bytes()?)?;
                    if store_offset != 0 {
                        SSTableIndex::V3(v3::SSTableIndexV3::load(index_bytes, store_offset)?)
                    } else {
                        // if store_offset is zero, there is no index, so we build a pseudo-index
                        // assuming a single block of sstable covering everything.
                        SSTableIndex::V3Empty(v3::SSTableIndexV3Empty::load(index_offset as usize))
                    }
                }
                _ => {
                    return Err(io::Error::other(format!(
                        "Unsupported sstable version, expected one of [2, 3], found {version}"
                    )));
                }
            };
        Ok(index)
    }

    pub(crate) fn get_block(&self, block_id: u64) -> io::Result<Option<BlockAddr>> {
        match self {
            Self::V2(index) => Ok(index.get_block(block_id as usize)),
            Self::V3(index) => index.get_block(block_id),
            Self::V3Empty(index) => Ok(index.get_block(block_id)),
        }
    }

    pub(crate) fn locate_with_key(&self, key: &[u8]) -> io::Result<Option<u64>> {
        match self {
            Self::V2(index) => Ok(index.locate_with_key(key).map(|id| id as u64)),
            Self::V3(index) => index.locate_with_key(key),
            Self::V3Empty(index) => Ok(index.locate_with_key(key)),
        }
    }

    pub fn get_block_with_key(&self, key: &[u8]) -> io::Result<Option<BlockAddr>> {
        match self.locate_with_key(key)? {
            Some(id) => self.get_block(id),
            None => Ok(None),
        }
    }

    pub(crate) fn locate_with_ord(&self, ord: TermOrdinal) -> io::Result<u64> {
        match self {
            Self::V2(index) => Ok(index.locate_with_ord(ord) as u64),
            Self::V3(index) => index.locate_with_ord(ord),
            Self::V3Empty(index) => Ok(index.locate_with_ord(ord)),
        }
    }

    pub fn get_block_with_ord(&self, ord: TermOrdinal) -> io::Result<BlockAddr> {
        match self {
            Self::V2(index) => Ok(index.get_block_with_ord(ord)),
            Self::V3(index) => index.get_block_with_ord(ord),
            Self::V3Empty(index) => Ok(index.get_block_with_ord(ord)),
        }
    }

    pub(crate) fn get_and_locate_with_ord(&self, ord: TermOrdinal) -> io::Result<(BlockAddr, u64)> {
        match self {
            Self::V2(index) => Ok(index.get_and_locate_with_ord(ord)),
            Self::V3(index) => index.get_and_locate_with_ord(ord),
            Self::V3Empty(index) => Ok(index.get_and_locate_with_ord(ord)),
        }
    }

    pub fn get_block_for_automaton<'a>(
        &'a self,
        automaton: &'a impl Automaton,
    ) -> io::Result<impl Iterator<Item = io::Result<(u64, BlockAddr)>> + 'a> {
        Ok(match self {
            Self::V2(index) => BlockIter::V2(index.get_block_for_automaton(automaton).map(Ok)),
            Self::V3(index) => BlockIter::V3(index.get_block_for_automaton(automaton)?),
            Self::V3Empty(index) => {
                BlockIter::V3Empty(std::iter::once(Ok((0, index.block_addr.clone()))))
            }
        })
    }
}

enum BlockIter<V2, V3, T> {
    V2(V2),
    V3(V3),
    V3Empty(std::iter::Once<T>),
}

impl<V2: Iterator<Item = T>, V3: Iterator<Item = T>, T> Iterator for BlockIter<V2, V3, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            BlockIter::V2(v2) => v2.next(),
            BlockIter::V3(v3) => v3.next(),
            BlockIter::V3Empty(once) => once.next(),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct BlockAddr {
    pub first_ordinal: u64,
    pub byte_range: Range<usize>,
}

impl BlockAddr {
    fn to_block_start(&self) -> BlockStartAddr {
        BlockStartAddr {
            first_ordinal: self.first_ordinal,
            byte_range_start: self.byte_range.start,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BlockStartAddr {
    first_ordinal: u64,
    byte_range_start: usize,
}

impl BlockStartAddr {
    fn to_block_addr(&self, byte_range_end: usize) -> BlockAddr {
        BlockAddr {
            first_ordinal: self.first_ordinal,
            byte_range: self.byte_range_start..byte_range_end,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BlockMeta {
    /// Any byte string that is lexicographically greater or equal to
    /// the last key in the block,
    /// and yet strictly smaller than the first key in the next block.
    pub last_key_or_greater: Vec<u8>,
    pub block_addr: BlockAddr,
}

impl BinarySerializable for BlockStartAddr {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        let start = self.byte_range_start as u64;
        start.serialize(writer)?;
        self.first_ordinal.serialize(writer)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let byte_range_start = u64::deserialize(reader)? as usize;
        let first_ordinal = u64::deserialize(reader)?;
        Ok(BlockStartAddr {
            first_ordinal,
            byte_range_start,
        })
    }

    // Provided method
    fn num_bytes(&self) -> u64 {
        BlockStartAddr::SIZE_IN_BYTES as u64
    }
}

impl FixedSize for BlockStartAddr {
    const SIZE_IN_BYTES: usize = 2 * u64::SIZE_IN_BYTES;
}

/// Given that left < right,
/// mutates `left into a shorter byte string left'` that
/// matches `left <= left' < right`.
fn find_shorter_str_in_between(left: &mut Vec<u8>, right: &[u8]) {
    assert!(&left[..] < right);
    let common_len = common_prefix_len(left, right);
    if left.len() == common_len {
        return;
    }
    // It is possible to do one character shorter in some case,
    // but it is not worth the extra complexity
    for pos in (common_len + 1)..left.len() {
        if left[pos] != u8::MAX {
            left[pos] += 1;
            left.truncate(pos + 1);
            return;
        }
    }
}

#[derive(Default)]
pub struct SSTableIndexBuilder {
    blocks: Vec<BlockMeta>,
}

impl SSTableIndexBuilder {
    /// In order to make the index as light as possible, we
    /// try to find a shorter alternative to the last key of the last block
    /// that is still smaller than the next key.
    pub(crate) fn shorten_last_block_key_given_next_key(&mut self, next_key: &[u8]) {
        if let Some(last_block) = self.blocks.last_mut() {
            find_shorter_str_in_between(&mut last_block.last_key_or_greater, next_key);
        }
    }

    pub fn add_block(&mut self, last_key: &[u8], byte_range: Range<usize>, first_ordinal: u64) {
        self.blocks.push(BlockMeta {
            last_key_or_greater: last_key.to_vec(),
            block_addr: BlockAddr {
                byte_range,
                first_ordinal,
            },
        })
    }

    pub fn serialize<W: std::io::Write>(&self, wrt: W) -> io::Result<u64> {
        if self.blocks.len() <= 1 {
            return Ok(0);
        }
        let counting_writer = common::CountingWriter::wrap(wrt);
        let mut map_builder = MapBuilder::new(counting_writer).map_err(fst_error_to_io_error)?;
        for (i, block) in self.blocks.iter().enumerate() {
            map_builder
                .insert(&block.last_key_or_greater, i as u64)
                .map_err(fst_error_to_io_error)?;
        }
        let counting_writer = map_builder.into_inner().map_err(fst_error_to_io_error)?;
        let written_bytes = counting_writer.written_bytes();
        let mut wrt = counting_writer.finish();

        let mut block_store_writer = v3::BlockAddrStoreWriter::new();
        for block in &self.blocks {
            block_store_writer.write_block_meta(block.block_addr.clone())?;
        }
        block_store_writer.serialize(&mut wrt)?;

        Ok(written_bytes)
    }
}

fn fst_error_to_io_error(error: tantivy_fst::Error) -> io::Error {
    match error {
        tantivy_fst::Error::Fst(fst_error) => io::Error::other(fst_error),
        tantivy_fst::Error::Io(ioerror) => ioerror,
    }
}

#[cfg(test)]
mod tests {
    #[track_caller]
    fn test_find_shorter_str_in_between_aux(left: &[u8], right: &[u8]) {
        let mut left_buf = left.to_vec();
        super::find_shorter_str_in_between(&mut left_buf, right);
        assert!(left_buf.len() <= left.len());
        assert!(left <= &left_buf);
        assert!(&left_buf[..] < right);
    }

    #[test]
    fn test_find_shorter_str_in_between() {
        test_find_shorter_str_in_between_aux(b"", b"hello");
        test_find_shorter_str_in_between_aux(b"abc", b"abcd");
        test_find_shorter_str_in_between_aux(b"abcd", b"abd");
        test_find_shorter_str_in_between_aux(&[0, 0, 0], &[1]);
        test_find_shorter_str_in_between_aux(&[0, 0, 0], &[0, 0, 1]);
        test_find_shorter_str_in_between_aux(&[0, 0, 255, 255, 255, 0u8], &[0, 1]);
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn test_proptest_find_shorter_str(left in any::<Vec<u8>>(), right in any::<Vec<u8>>()) {
            if left < right {
                test_find_shorter_str_in_between_aux(&left, &right);
            }
        }
    }
}
