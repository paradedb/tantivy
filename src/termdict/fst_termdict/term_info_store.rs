use std::cmp;
use std::io::{self, Read, Write};

use byteorder::{ByteOrder, LittleEndian};
use common::{BinarySerializable, FixedSize};
use tantivy_bitpacker::{compute_num_bits, BitPacker};

use crate::directory::{FileSlice, OwnedBytes};
use crate::postings::{TermInfo, TermInfoVersion};
use crate::termdict::TermOrdinal;

const BLOCK_LEN: usize = 256;

#[derive(Debug, Eq, PartialEq, Default)]
struct TermInfoBlockMeta {
    offset: u64,
    ref_term_info: TermInfo,
    doc_freq_nbits: u8,
    postings_offset_nbits: u8,
    positions_offset_nbits: u8,
    pnorms_offset_nbits: u8,
    bitmap_offset_nbits: u8,
}

impl TermInfoBlockMeta {
    fn serialized_size(version: TermInfoVersion) -> usize {
        u64::SIZE_IN_BYTES
            + version.serialized_size()
            + 3
            + usize::from(version != TermInfoVersion::V1)
            + usize::from(version == TermInfoVersion::V3)
    }

    fn serialize<W: Write + ?Sized>(
        &self,
        write: &mut W,
        version: TermInfoVersion,
    ) -> io::Result<()> {
        self.offset.serialize(write)?;
        self.ref_term_info.serialize_versioned(write, version)?;
        write.write_all(&[
            self.doc_freq_nbits,
            self.postings_offset_nbits,
            self.positions_offset_nbits,
        ])?;
        if version != TermInfoVersion::V1 {
            self.pnorms_offset_nbits.serialize(write)?;
        }
        if version == TermInfoVersion::V3 {
            self.bitmap_offset_nbits.serialize(write)?;
        }
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R, version: TermInfoVersion) -> io::Result<Self> {
        let offset = u64::deserialize(reader)?;
        let ref_term_info = TermInfo::deserialize_versioned(reader, version)?;
        let mut buffer = [0u8; 3];
        reader.read_exact(&mut buffer)?;
        Ok(TermInfoBlockMeta {
            offset,
            ref_term_info,
            doc_freq_nbits: buffer[0],
            postings_offset_nbits: buffer[1],
            positions_offset_nbits: buffer[2],
            pnorms_offset_nbits: match version {
                TermInfoVersion::V1 => 0,
                TermInfoVersion::V2 | TermInfoVersion::V3 => u8::deserialize(reader)?,
            },
            bitmap_offset_nbits: if version == TermInfoVersion::V3 {
                u8::deserialize(reader)?
            } else {
                0
            },
        })
    }

    fn num_bits(&self) -> usize {
        usize::from(self.doc_freq_nbits)
            + usize::from(self.postings_offset_nbits)
            + usize::from(self.positions_offset_nbits)
            + usize::from(self.pnorms_offset_nbits)
            + usize::from(self.bitmap_offset_nbits)
    }

    // Here inner_offset is the offset within the block, WITHOUT the first term_info.
    // In other word, term_info #1,#2,#3 gets inner_offset 0,1,2... While term_info #0
    // is encoded without bitpacking.
    fn deserialize_term_info(&self, data: &[u8], inner_offset: usize) -> TermInfo {
        assert!(inner_offset < BLOCK_LEN - 1);
        let num_bits = self.num_bits();

        let posting_start_addr = num_bits * inner_offset;
        // the posting_start is the posting_start of the next term info.
        let posting_end_addr = posting_start_addr + num_bits;
        let positions_start_addr = posting_start_addr + self.postings_offset_nbits as usize;
        // the position_end is the positions_start of the next term info.
        let positions_end_addr = positions_start_addr + num_bits;

        let doc_freq_addr = positions_start_addr + self.positions_offset_nbits as usize;

        let postings_start_offset = self.ref_term_info.postings_range.start
            + extract_bits(data, posting_start_addr, self.postings_offset_nbits) as usize;
        let postings_end_offset = self.ref_term_info.postings_range.start
            + extract_bits(data, posting_end_addr, self.postings_offset_nbits) as usize;

        let positions_start_offset = self.ref_term_info.positions_range.start
            + extract_bits(data, positions_start_addr, self.positions_offset_nbits) as usize;
        let positions_end_offset = self.ref_term_info.positions_range.start
            + extract_bits(data, positions_end_addr, self.positions_offset_nbits) as usize;

        let doc_freq = extract_bits(data, doc_freq_addr, self.doc_freq_nbits) as u32;

        TermInfo {
            doc_freq,
            postings_range: postings_start_offset..postings_end_offset,
            positions_range: positions_start_offset..positions_end_offset,
            pnorms_offset: self.ref_term_info.pnorms_offset.map(|offset| {
                offset
                    + extract_bits(
                        data,
                        doc_freq_addr + usize::from(self.doc_freq_nbits),
                        self.pnorms_offset_nbits,
                    )
            }),
            bitmap_offset: {
                let encoded = extract_bits(
                    data,
                    doc_freq_addr
                        + usize::from(self.doc_freq_nbits)
                        + usize::from(self.pnorms_offset_nbits),
                    self.bitmap_offset_nbits,
                );
                encoded.checked_sub(1)
            },
        }
    }
}

#[derive(Clone)]
pub struct TermInfoStore {
    num_terms: usize,
    block_meta_bytes: OwnedBytes,
    term_info_bytes: OwnedBytes,
    version: TermInfoVersion,
}

fn extract_bits(data: &[u8], addr_bits: usize, num_bits: u8) -> u64 {
    assert!(num_bits <= 56);
    let addr_byte = addr_bits / 8;
    let bit_shift = (addr_bits % 8) as u64;
    let val_unshifted_unmasked: u64 = if data.len() >= addr_byte + 8 {
        LittleEndian::read_u64(&data[addr_byte..][..8])
    } else {
        // the buffer is not large enough.
        // Let's copy the few remaining bytes to a 8 byte buffer
        // padded with 0s.
        let mut buf = [0u8; 8];
        let data_to_copy = &data[addr_byte..];
        let nbytes = data_to_copy.len();
        buf[..nbytes].copy_from_slice(data_to_copy);
        LittleEndian::read_u64(&buf)
    };
    let val_shifted_unmasked = val_unshifted_unmasked >> bit_shift;
    let mask = (1u64 << u64::from(num_bits)) - 1;
    val_shifted_unmasked & mask
}

impl TermInfoStore {
    pub fn open(
        term_info_store_file: FileSlice,
        version: TermInfoVersion,
    ) -> io::Result<TermInfoStore> {
        let (len_slice, main_slice) = term_info_store_file.split(16);
        let mut bytes = len_slice.read_bytes()?;
        let len = u64::deserialize(&mut bytes)? as usize;
        let num_terms = u64::deserialize(&mut bytes)? as usize;
        let (block_meta_file, term_info_file) = main_slice.split(len);
        let term_info_bytes = term_info_file.read_bytes()?;
        Ok(TermInfoStore {
            num_terms,
            block_meta_bytes: block_meta_file.read_bytes()?,
            term_info_bytes,
            version,
        })
    }

    pub fn get(&self, term_ord: TermOrdinal) -> TermInfo {
        let block_id = (term_ord as usize) / BLOCK_LEN;
        let buffer = self.block_meta_bytes.as_slice();
        let mut block_data: &[u8] =
            &buffer[block_id * TermInfoBlockMeta::serialized_size(self.version)..];
        let term_info_block_data = TermInfoBlockMeta::deserialize(&mut block_data, self.version)
            .expect("Failed to deserialize terminfoblockmeta");
        let inner_offset = (term_ord as usize) % BLOCK_LEN;
        if inner_offset == 0 {
            term_info_block_data.ref_term_info
        } else {
            term_info_block_data.deserialize_term_info(
                &self.term_info_bytes[term_info_block_data.offset as usize..],
                inner_offset - 1,
            )
        }
    }

    pub fn num_terms(&self) -> usize {
        self.num_terms
    }
}

pub struct TermInfoStoreWriter {
    buffer_block_metas: Vec<TermInfoBlockMeta>,
    buffer_term_infos: Vec<u8>,
    term_infos: Vec<TermInfo>,
    num_terms: u64,
    has_pnorms: bool,
    has_bitmaps: bool,
}

fn bitpack_serialize<W: Write>(
    write: &mut W,
    bit_packer: &mut BitPacker,
    term_info_block_meta: &TermInfoBlockMeta,
    term_info: &TermInfo,
) -> io::Result<()> {
    bit_packer.write(
        term_info.postings_range.start as u64,
        term_info_block_meta.postings_offset_nbits,
        write,
    )?;
    bit_packer.write(
        term_info.positions_range.start as u64,
        term_info_block_meta.positions_offset_nbits,
        write,
    )?;
    bit_packer.write(
        u64::from(term_info.doc_freq),
        term_info_block_meta.doc_freq_nbits,
        write,
    )?;
    if let Some(offset) = term_info.pnorms_offset {
        bit_packer.write(offset, term_info_block_meta.pnorms_offset_nbits, write)?;
    }
    bit_packer.write(
        term_info.bitmap_offset.map_or(0, |offset| offset + 1),
        term_info_block_meta.bitmap_offset_nbits,
        write,
    )?;
    Ok(())
}

impl TermInfoStoreWriter {
    pub fn new() -> TermInfoStoreWriter {
        TermInfoStoreWriter {
            buffer_block_metas: Vec::new(),
            buffer_term_infos: Vec::new(),
            term_infos: Vec::with_capacity(BLOCK_LEN),
            num_terms: 0u64,
            has_pnorms: false,
            has_bitmaps: false,
        }
    }

    pub fn has_pnorms(&self) -> bool {
        self.has_pnorms
    }

    pub(crate) fn version(&self) -> TermInfoVersion {
        if self.has_bitmaps {
            TermInfoVersion::V3
        } else if self.has_pnorms {
            TermInfoVersion::V2
        } else {
            TermInfoVersion::V1
        }
    }

    fn flush_block(&mut self) -> io::Result<()> {
        let mut bit_packer = BitPacker::new();
        let last_term_info = if let Some(last_term_info) = self.term_infos.last().cloned() {
            last_term_info
        } else {
            return Ok(());
        };
        let ref_term_info = self.term_infos[0].clone();
        let bitmap_offset_nbits = compute_num_bits(
            self.term_infos
                .iter()
                .filter_map(|info| info.bitmap_offset)
                .max()
                .map_or(0, |offset| offset + 1),
        );
        if bitmap_offset_nbits > 56 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "bitmap offset exceeds the term dictionary limit",
            ));
        }
        let pnorms_offset_nbits = ref_term_info.pnorms_offset.map_or(0, |base| {
            compute_num_bits(last_term_info.pnorms_offset.unwrap() - base)
        });
        let postings_end_offset =
            last_term_info.postings_range.end - ref_term_info.postings_range.start;
        let positions_end_offset =
            last_term_info.positions_range.end - ref_term_info.positions_range.start;
        for term_info in &mut self.term_infos[1..] {
            term_info.postings_range.start -= ref_term_info.postings_range.start;
            term_info.positions_range.start -= ref_term_info.positions_range.start;
            if let Some(offset) = &mut term_info.pnorms_offset {
                *offset -= ref_term_info.pnorms_offset.unwrap();
            }
        }

        let mut max_doc_freq: u32 = 0u32;

        for term_info in &self.term_infos[1..] {
            max_doc_freq = cmp::max(max_doc_freq, term_info.doc_freq);
        }

        let max_doc_freq_nbits: u8 = compute_num_bits(u64::from(max_doc_freq));
        let max_postings_offset_nbits = compute_num_bits(postings_end_offset as u64);
        let max_positions_offset_nbits = compute_num_bits(positions_end_offset as u64);

        let term_info_block_meta = TermInfoBlockMeta {
            offset: self.buffer_term_infos.len() as u64,
            ref_term_info,
            doc_freq_nbits: max_doc_freq_nbits,
            postings_offset_nbits: max_postings_offset_nbits,
            positions_offset_nbits: max_positions_offset_nbits,
            pnorms_offset_nbits,
            bitmap_offset_nbits,
        };

        for term_info in &self.term_infos[1..] {
            bitpack_serialize(
                &mut self.buffer_term_infos,
                &mut bit_packer,
                &term_info_block_meta,
                term_info,
            )?;
        }

        // We still need to serialize the end offset for postings & positions.
        bit_packer.write(
            postings_end_offset as u64,
            term_info_block_meta.postings_offset_nbits,
            &mut self.buffer_term_infos,
        )?;
        bit_packer.write(
            positions_end_offset as u64,
            term_info_block_meta.positions_offset_nbits,
            &mut self.buffer_term_infos,
        )?;

        // Block need end up at the end of a byte.
        bit_packer.flush(&mut self.buffer_term_infos)?;
        self.buffer_block_metas.push(term_info_block_meta);
        self.term_infos.clear();

        Ok(())
    }

    pub fn write_term_info(&mut self, term_info: &TermInfo) -> io::Result<()> {
        if self.num_terms == 0 {
            self.has_pnorms = term_info.pnorms_offset.is_some();
        }
        assert_eq!(term_info.pnorms_offset.is_some(), self.has_pnorms);
        self.has_bitmaps |= term_info.bitmap_offset.is_some();
        self.num_terms += 1u64;
        self.term_infos.push(term_info.clone());
        if self.term_infos.len() >= BLOCK_LEN {
            self.flush_block()?;
        }
        Ok(())
    }

    pub fn serialize<W: io::Write + ?Sized>(&mut self, write: &mut W) -> io::Result<()> {
        if !self.term_infos.is_empty() {
            self.flush_block()?;
        }
        let version = self.version();
        let len =
            (self.buffer_block_metas.len() * TermInfoBlockMeta::serialized_size(version)) as u64;
        len.serialize(write)?;
        self.num_terms.serialize(write)?;
        for meta in &self.buffer_block_metas {
            meta.serialize(write, version)?;
        }
        write.write_all(&self.buffer_term_infos)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use tantivy_bitpacker::{compute_num_bits, BitPacker};

    use super::{extract_bits, TermInfoBlockMeta, TermInfoStore, TermInfoStoreWriter};
    use crate::directory::FileSlice;
    use crate::postings::{TermInfo, TermInfoVersion};

    #[test]
    fn test_bitpacked() {
        let mut buffer = Vec::new();
        let mut bitpack = BitPacker::new();
        bitpack.write(321u64, 9, &mut buffer).unwrap();
        assert_eq!(compute_num_bits(321u64), 9);
        bitpack.write(2u64, 2, &mut buffer).unwrap();
        assert_eq!(compute_num_bits(2u64), 2);
        bitpack.write(51, 6, &mut buffer).unwrap();
        assert_eq!(compute_num_bits(51), 6);
        bitpack.close(&mut buffer).unwrap();
        assert_eq!(buffer.len(), 3);
        assert_eq!(extract_bits(&buffer[..], 0, 9), 321u64);
        assert_eq!(extract_bits(&buffer[..], 9, 2), 2u64);
        assert_eq!(extract_bits(&buffer[..], 11, 6), 51u64);
    }

    #[test]
    fn test_term_info_block_meta_serialization() {
        for version in [TermInfoVersion::V1, TermInfoVersion::V2] {
            let term_info_block_meta = TermInfoBlockMeta {
                offset: 2009u64,
                ref_term_info: TermInfo {
                    doc_freq: 512,
                    postings_range: 51..57,
                    positions_range: 110..134,
                    pnorms_offset: (version == TermInfoVersion::V2).then_some(1 << 40),
                    bitmap_offset: None,
                },
                bitmap_offset_nbits: 0,
                doc_freq_nbits: 10,
                postings_offset_nbits: 5,
                positions_offset_nbits: 8,
                pnorms_offset_nbits: if version == TermInfoVersion::V2 {
                    11
                } else {
                    0
                },
            };
            let mut buffer = Vec::new();
            term_info_block_meta
                .serialize(&mut buffer, version)
                .unwrap();
            assert_eq!(buffer.len(), TermInfoBlockMeta::serialized_size(version));
            let mut cursor = buffer.as_slice();
            assert_eq!(
                TermInfoBlockMeta::deserialize(&mut cursor, version).unwrap(),
                term_info_block_meta
            );
            assert!(cursor.is_empty());
        }
    }

    #[test]
    fn test_pack() -> crate::Result<()> {
        for count in [0, 1, 255, 256, 257, 1_000] {
            let mut plain_len = 0;
            for initial_norm_offset in [None, Some(0), Some(1 << 40)] {
                let mut store_writer = TermInfoStoreWriter::new();
                let mut term_infos = vec![];
                let offset = |i| i * 13 + i * i;
                let mut norm_offset = initial_norm_offset;
                for i in 0..count {
                    let term_info = TermInfo {
                        doc_freq: i as u32,
                        postings_range: offset(i)..offset(i + 1),
                        positions_range: offset(i) * 3..offset(i + 1) * 3,
                        pnorms_offset: norm_offset,
                        bitmap_offset: None,
                    };
                    if let Some(offset) = &mut norm_offset {
                        *offset += u64::from(term_info.doc_freq);
                    }
                    store_writer.write_term_info(&term_info)?;
                    term_infos.push(term_info);
                }
                let mut buffer = Vec::new();
                store_writer.serialize(&mut buffer)?;
                let version = if store_writer.has_pnorms() {
                    TermInfoVersion::V2
                } else {
                    TermInfoVersion::V1
                };
                if initial_norm_offset.is_none() {
                    plain_len = buffer.len();
                } else if count >= 256 {
                    assert!(buffer.len() - plain_len < count * 3);
                }
                let term_info_store = TermInfoStore::open(FileSlice::from(buffer), version)?;
                assert_eq!(term_info_store.num_terms(), count);
                for i in (0..count).rev() {
                    assert_eq!(
                        term_info_store.get(i as u64),
                        term_infos[i],
                        "term info {i}"
                    );
                }
            }
        }
        Ok(())
    }
}
