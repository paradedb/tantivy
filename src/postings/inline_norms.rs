use std::io::{self, Write};

use common::{BinarySerializable, VInt};
use tantivy_bitpacker::{compute_num_bits, minmax, BitPacker, BitUnpacker};

use crate::directory::{FileSlice, OwnedBytes};
use common::HasLen;

const MAGIC: [u8; 10] = [127, 127, 127, 127, 127, 127, 127, 127, 127, 132];
const BLOCK_LEN: usize = 128;
const RAW: u8 = 9;
const BLOCKS: u8 = 10;

pub(crate) fn read_without_norms(source: FileSlice) -> io::Result<(OwnedBytes, Option<FileSlice>)> {
    let probe_len = source.len().min(20);
    let mut header = source.slice_to(probe_len).read_bytes()?;
    if !header.starts_with(&MAGIC) {
        return Ok((source.read_bytes()?, None));
    }
    header.advance(MAGIC.len());
    let norm_len = usize::try_from(VInt::deserialize_u64(&mut header)?).map_err(|_| invalid())?;
    let end = (probe_len - header.len())
        .checked_add(norm_len)
        .filter(|&end| end <= source.len())
        .ok_or_else(invalid)?;
    Ok((
        source.slice_from(end).read_bytes()?,
        Some(source.slice_to(end)),
    ))
}

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid inline norms")
}

fn frame(values: &[u8]) -> (u8, u8, usize) {
    let (minimum, maximum) = minmax(values.iter().copied()).unwrap();
    let width = compute_num_bits(u64::from(maximum - minimum));
    (
        minimum,
        width,
        2 + (values.len() * width as usize).div_ceil(8),
    )
}

fn write_frame(values: &[u8], output: &mut Vec<u8>) -> io::Result<()> {
    let (minimum, width, _) = frame(values);
    output.extend_from_slice(&[width, minimum]);
    let mut packer = BitPacker::new();
    for &value in values {
        packer.write(u64::from(value - minimum), width, output)?;
    }
    packer.close(output)
}

pub(crate) fn write_header(values: &[u8], output: &mut impl Write) -> io::Result<()> {
    if values.is_empty() {
        return Err(invalid());
    }
    let whole_size = frame(values).2;
    let raw_size = values.len() + 1;
    let count = values.len().div_ceil(BLOCK_LEN);
    let blocks_size =
        1 + (count - 1) * 4 + values.chunks(BLOCK_LEN).map(|v| frame(v).2).sum::<usize>();
    let mut data = Vec::with_capacity(whole_size.min(raw_size).min(blocks_size));
    if blocks_size < whole_size.min(raw_size) {
        data.push(BLOCKS);
        let mut offset = 0usize;
        for (i, block) in values.chunks(BLOCK_LEN).enumerate() {
            if i != 0 {
                u32::try_from(offset)
                    .map_err(|_| invalid())?
                    .serialize(&mut data)?;
            }
            offset += frame(block).2;
        }
        for block in values.chunks(BLOCK_LEN) {
            write_frame(block, &mut data)?;
        }
    } else if whole_size <= raw_size {
        write_frame(values, &mut data)?;
    } else {
        data.push(RAW);
        data.extend_from_slice(values);
    }
    output.write_all(&MAGIC)?;
    VInt(data.len() as u64).serialize(output)?;
    output.write_all(&data)
}

#[derive(Clone)]
pub(crate) struct InlineNorms {
    data: OwnedBytes,
    len: usize,
}

impl InlineNorms {
    pub(crate) fn read_header(
        doc_freq: u32,
        mut bytes: OwnedBytes,
    ) -> io::Result<(Option<Self>, OwnedBytes)> {
        if !bytes.starts_with(&MAGIC) {
            return Ok((None, bytes));
        }
        bytes.advance(MAGIC.len());
        let byte_len =
            usize::try_from(VInt::deserialize_u64(&mut bytes)?).map_err(|_| invalid())?;
        if byte_len > bytes.len() || byte_len == 0 || doc_freq == 0 {
            return Err(invalid());
        }
        let (data, rest) = bytes.split(byte_len);
        let reader = Self {
            data,
            len: doc_freq as usize,
        };
        if reader.data[0] > BLOCKS {
            return Err(invalid());
        }
        Ok((Some(reader), rest))
    }

    pub(crate) fn read(&self, ordinal: usize) -> io::Result<u8> {
        if ordinal >= self.len {
            return Err(invalid());
        }
        if self.data[0] == RAW {
            if self.data.len() != self.len + 1 {
                return Err(invalid());
            }
            return Ok(self.data[ordinal + 1]);
        }
        let (data, count, index) = if self.data[0] == BLOCKS {
            let blocks = self.len.div_ceil(BLOCK_LEN);
            let base = 1 + (blocks - 1) * 4;
            if base > self.data.len() {
                return Err(invalid());
            }
            let block = ordinal / BLOCK_LEN;
            let offset = |i: usize| -> usize {
                if i == 0 {
                    0
                } else {
                    u32::from_le_bytes(self.data[1 + (i - 1) * 4..1 + i * 4].try_into().unwrap())
                        as usize
                }
            };
            let from = offset(block);
            let to = if block + 1 == blocks {
                self.data.len() - base
            } else {
                offset(block + 1)
            };
            if from > to || to > self.data.len() - base {
                return Err(invalid());
            }
            (
                &self.data[base + from..base + to],
                (self.len - block * BLOCK_LEN).min(BLOCK_LEN),
                ordinal % BLOCK_LEN,
            )
        } else {
            (&self.data[..], self.len, ordinal)
        };
        if data.len() < 2 || data[0] > 8 || data.len() != 2 + (count * data[0] as usize).div_ceil(8)
        {
            return Err(invalid());
        }
        let delta = BitUnpacker::new(data[0]).get(index as u32, &data[2..]) as u8;
        data[1].checked_add(delta).ok_or_else(invalid)
    }
}

pub fn rewrite_inline_posting_norms(
    bytes: OwnedBytes,
    doc_freq: u32,
    norms: &[u8],
) -> io::Result<Vec<u8>> {
    if bytes.starts_with(&MAGIC) {
        return Ok(bytes.to_vec());
    }
    let (offset, rest) = super::term_norms::read_header(bytes.clone())?;
    let Some(offset) = offset else {
        return Ok(bytes.to_vec());
    };
    let (_, rest) = super::packed_norms::EmbeddedNormDirectory::read_header(rest)?;
    let offset = usize::try_from(offset).map_err(|_| invalid())?;
    let end = offset.checked_add(doc_freq as usize).ok_or_else(invalid)?;
    let values = norms.get(offset..end).ok_or_else(invalid)?;
    let mut output = Vec::new();
    write_header(values, &mut output)?;
    output.extend_from_slice(&rest);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_only_reads_skip_norms_and_scoring_loads_them_once() {
        use crate::directory::FileHandle;
        use common::file_slice::DeferredFileSlice;
        use std::ops::Range;
        use std::sync::{Arc, Mutex};

        #[derive(Debug)]
        struct Tracked {
            data: OwnedBytes,
            reads: Arc<Mutex<Vec<Range<usize>>>>,
        }
        impl HasLen for Tracked {
            fn len(&self) -> usize {
                self.data.len()
            }
        }
        impl FileHandle for Tracked {
            fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
                self.reads.lock().unwrap().push(range.clone());
                Ok(self.data.slice(range))
            }
        }
        let values: Vec<u8> = (0..20000).map(|i| (i % 251) as u8).collect();
        let mut bytes = Vec::new();
        write_header(&values, &mut bytes).unwrap();
        let end = bytes.len();
        bytes.extend_from_slice(b"postings body");
        let reads = Arc::new(Mutex::new(Vec::new()));
        let file = FileSlice::new(Arc::new(Tracked {
            data: OwnedBytes::new(bytes),
            reads: reads.clone(),
        }));
        let (body, source) = read_without_norms(file).unwrap();
        assert_eq!(&*body, b"postings body");
        assert_eq!(*reads.lock().unwrap(), vec![0..20, end..end + 13]);
        super::super::term_norms::set_posting_norms_enabled(true);
        let mut reader = super::super::term_norms::TermNormReader::new(
            Arc::new(DeferredFileSlice::new(|| panic!("sidecar opened"))),
            0,
            values.len() as u32,
        )
        .unwrap();
        reader.lazy_inline_norms = source;
        assert_eq!(reads.lock().unwrap().len(), 2);
        for i in (0..values.len()).rev() {
            assert_eq!(reader.read(i).unwrap(), values[i]);
        }
        assert_eq!(*reads.lock().unwrap(), vec![0..20, end..end + 13, 0..end]);
    }

    #[test]
    fn roundtrip_modes_widths_tails_and_random_access() {
        let mut modes = std::collections::HashSet::new();
        for len in [1, 2, 127, 128, 129, 8193, 20000] {
            for width in 0..=8 {
                for local in [false, true] {
                    let mask = (1u16 << width) - 1;
                    let values: Vec<u8> = (0..len)
                        .map(|i| {
                            let n = ((i as u16).wrapping_mul(73) & mask) as u8;
                            if local {
                                ((i / 128) % 16) as u8 * 16 + n % 16
                            } else {
                                n
                            }
                        })
                        .collect();
                    let mut encoded = Vec::new();
                    write_header(&values, &mut encoded).unwrap();
                    assert!(encoded.len() <= 18 + values.len());
                    let (reader, rest) =
                        InlineNorms::read_header(len as u32, OwnedBytes::new(encoded.clone()))
                            .unwrap();
                    assert!(rest.is_empty());
                    let reader = reader.unwrap();
                    modes.insert(reader.data[0]);
                    for i in (0..len).rev() {
                        assert_eq!(reader.read(i).unwrap(), values[i]);
                    }
                    for i in (0..len).step_by(137) {
                        assert_eq!(reader.clone().read(i).unwrap(), values[i]);
                    }
                    assert!(reader.read(len).is_err());
                    for tail in 1..=encoded.len().min(8) {
                        if encoded.len() - tail >= MAGIC.len() {
                            assert!(InlineNorms::read_header(
                                len as u32,
                                OwnedBytes::new(encoded[..encoded.len() - tail].to_vec())
                            )
                            .is_err());
                        }
                    }
                }
            }
        }
        assert!(modes.contains(&RAW));
        assert!(modes.contains(&BLOCKS));
        assert!(modes.contains(&0));
    }

    #[test]
    fn rewrite_removes_old_offset_and_preserves_body() {
        let values: Vec<u8> = (0..300).map(|i| (i % 31) as u8).collect();
        let mut old = super::super::term_norms::MAGIC.to_vec();
        37u64.serialize(&mut old).unwrap();
        old.extend_from_slice(b"postings body");
        let rewritten = rewrite_inline_posting_norms(OwnedBytes::new(old), 129, &values).unwrap();
        let (norms, body) =
            InlineNorms::read_header(129, OwnedBytes::new(rewritten.clone())).unwrap();
        assert_eq!(&*body, b"postings body");
        for i in 0..129 {
            assert_eq!(norms.as_ref().unwrap().read(i).unwrap(), values[37 + i]);
        }
        assert_eq!(
            rewrite_inline_posting_norms(OwnedBytes::new(rewritten.clone()), 129, &values).unwrap(),
            rewritten
        );
    }
}
