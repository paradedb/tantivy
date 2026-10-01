use std::io;

mod merger;

use std::iter::ExactSizeIterator;

use common::{BinarySerializable, VInt};
use sstable::streamer::StreamerWithState;
use sstable::value::{ValueReader, ValueWriter};
use sstable::SSTable;
use tantivy_fst::automaton::AlwaysMatch;
use tantivy_fst::Automaton;

pub use self::merger::TermMerger;
use crate::postings::{TermInfo, TermInfoVersion};

pub struct TermWithStateStreamerBuilder<'a, A>
where
    A: Automaton,
    A::State: Clone,
{
    streamer_builder: TermStreamerBuilder<'a, A>,
}

pub trait TermDictionaryExt {
    fn search_with_state<'a, A>(&'a self, automaton: A) -> TermWithStateStreamerBuilder<'a, A>
    where
        A: Automaton + 'a,
        A::State: Clone;
}

impl TermDictionaryExt for TermDictionary {
    fn search_with_state<'a, A>(&'a self, automaton: A) -> TermWithStateStreamerBuilder<'a, A>
    where
        A: Automaton + 'a,
        A::State: Clone,
    {
        let streamer_builder = self.search(automaton);
        TermWithStateStreamerBuilder::new(streamer_builder)
    }
}

impl<'a, A> TermWithStateStreamerBuilder<'a, A>
where
    A: Automaton,
    A::State: Clone,
{
    pub fn new(streamer_builder: TermStreamerBuilder<'a, A>) -> Self {
        TermWithStateStreamerBuilder { streamer_builder }
    }

    pub fn ge<T: AsRef<[u8]>>(mut self, bound: T) -> Self {
        self.streamer_builder = self.streamer_builder.ge(bound);
        self
    }

    pub fn gt<T: AsRef<[u8]>>(mut self, bound: T) -> Self {
        self.streamer_builder = self.streamer_builder.gt(bound);
        self
    }

    pub fn le<T: AsRef<[u8]>>(mut self, bound: T) -> Self {
        self.streamer_builder = self.streamer_builder.le(bound);
        self
    }

    pub fn lt<T: AsRef<[u8]>>(mut self, bound: T) -> Self {
        self.streamer_builder = self.streamer_builder.lt(bound);
        self
    }

    pub fn backward(mut self) -> Self {
        self.streamer_builder = self.streamer_builder.backward();
        self
    }

    pub fn into_stream(self) -> io::Result<TermWithStateStreamer<'a, A>> {
        let streamer_with_state = self.streamer_builder.into_stream_with_state()?;
        Ok(TermWithStateStreamer {
            streamer_with_state,
        })
    }
}

pub struct TermWithStateStreamer<'a, A>
where
    A: Automaton,
    A::State: Clone,
{
    streamer_with_state: StreamerWithState<'a, TermSSTable, A>,
}

impl<'a, A> TermWithStateStreamer<'a, A>
where
    A: Automaton,
    A::State: Clone,
{
    pub fn advance(&mut self) -> bool {
        self.streamer_with_state.advance()
    }

    pub fn next(&mut self) -> Option<(&[u8], &TermInfo, A::State)> {
        self.streamer_with_state.next()
    }

    pub fn key(&self) -> &[u8] {
        self.streamer_with_state.key()
    }

    pub fn value(&self) -> &TermInfo {
        self.streamer_with_state.value()
    }

    pub fn state(&self) -> Option<&A::State> {
        self.streamer_with_state.state()
    }
}

/// The term dictionary contains all of the terms in
/// `tantivy index` in a sorted manner.
///
/// The `Fst` crate is used to associate terms to their
/// respective `TermOrdinal`. The `TermInfoStore` then makes it
/// possible to fetch the associated `TermInfo`.
pub type TermDictionary = sstable::Dictionary<TermSSTable>;

/// Builder for the new term dictionary.
pub type TermDictionaryBuilder<W> = sstable::Writer<W, TermInfoValueWriter>;

/// `TermStreamer` acts as a cursor over a range of terms of a segment.
/// Terms are guaranteed to be sorted.
pub type TermStreamer<'a, A = AlwaysMatch> = sstable::Streamer<'a, TermSSTable, A>;

/// SSTable used to store TermInfo objects.
#[derive(Clone)]
pub struct TermSSTable;

pub type TermStreamerBuilder<'a, A = AlwaysMatch> = sstable::StreamerBuilder<'a, TermSSTable, A>;

impl SSTable for TermSSTable {
    type Value = TermInfo;
    type ValueReader = TermInfoValueReader;
    type ValueWriter = TermInfoValueWriter;
}

// V1 blocks start with their term count; reserve an impossible count for versioned blocks.
const VERSIONED_BLOCK: u64 = u64::MAX;

#[derive(Default)]
pub struct TermInfoValueReader {
    term_infos: Vec<TermInfo>,
}

impl ValueReader for TermInfoValueReader {
    type Value = TermInfo;

    #[inline(always)]
    fn value(&self, idx: usize) -> &TermInfo {
        &self.term_infos[idx]
    }

    fn load(&mut self, mut data: &[u8]) -> io::Result<usize> {
        let len_before = data.len();
        let header = VInt::deserialize_u64(&mut data)?;
        let (version, num_els) = if header == VERSIONED_BLOCK {
            let version = TermInfoVersion::deserialize(&mut data)?;
            (version, VInt::deserialize_u64(&mut data)?)
        } else {
            (TermInfoVersion::V1, header)
        };
        self.term_infos.clear();
        if num_els == 0 {
            return Ok(len_before - data.len());
        }
        let mut postings_start = VInt::deserialize_u64(&mut data)? as usize;
        let mut positions_start = VInt::deserialize_u64(&mut data)? as usize;
        let mut pnorms_offset = match version {
            TermInfoVersion::V1 => None,
            TermInfoVersion::V2 => Some(VInt::deserialize_u64(&mut data)?),
            TermInfoVersion::V3 => {
                let offset = VInt::deserialize_u64(&mut data)?;
                (offset != u64::MAX).then_some(offset)
            }
        };

        let mut freqs_start = if version == TermInfoVersion::V3 {
            Some(VInt::deserialize_u64(&mut data)? as usize)
        } else {
            None
        };
        self.term_infos.reserve_exact(num_els as usize);
        for _ in 0..num_els {
            let doc_freq = VInt::deserialize_u64(&mut data)? as u32;
            let postings_num_bytes = VInt::deserialize_u64(&mut data)?;
            let positions_num_bytes = VInt::deserialize_u64(&mut data)?;
            let postings_end = postings_start + postings_num_bytes as usize;
            let positions_end = positions_start + positions_num_bytes as usize;
            let freqs_range = if let Some(start) = &mut freqs_start {
                let end = *start + VInt::deserialize_u64(&mut data)? as usize;
                let range = *start..end;
                *start = end;
                Some(range)
            } else {
                None
            };
            let term_info = TermInfo {
                doc_freq,
                postings_range: postings_start..postings_end,
                positions_range: positions_start..positions_end,
                pnorms_offset,
                freqs_range,
            };
            self.term_infos.push(term_info);
            postings_start = postings_end;
            positions_start = positions_end;
            if let Some(offset) = &mut pnorms_offset {
                *offset += u64::from(doc_freq);
            }
        }
        let consumed_len = len_before - data.len();
        Ok(consumed_len)
    }
}

#[derive(Default)]
pub struct TermInfoValueWriter {
    term_infos: Vec<TermInfo>,
}

impl ValueWriter for TermInfoValueWriter {
    type Value = TermInfo;

    fn write(&mut self, term_info: &TermInfo) {
        self.term_infos.push(term_info.clone());
    }

    fn serialize_block(&self, buffer: &mut Vec<u8>) {
        let has_pnorms = self
            .term_infos
            .iter()
            .any(|info| info.pnorms_offset.is_some());
        let has_freqs = self
            .term_infos
            .iter()
            .any(|info| info.freqs_range.is_some());
        if has_pnorms || has_freqs {
            VInt(VERSIONED_BLOCK).serialize_into_vec(buffer);
            let version = if has_freqs {
                TermInfoVersion::V3
            } else {
                TermInfoVersion::V2
            };
            version.serialize(buffer).unwrap();
        }
        VInt(self.term_infos.len() as u64).serialize_into_vec(buffer);
        if self.term_infos.is_empty() {
            return;
        }
        VInt(self.term_infos[0].postings_range.start as u64).serialize_into_vec(buffer);
        VInt(self.term_infos[0].positions_range.start as u64).serialize_into_vec(buffer);
        let mut pnorms_offset = self.term_infos[0].pnorms_offset;
        if has_pnorms || has_freqs {
            VInt(pnorms_offset.unwrap_or(u64::MAX)).serialize_into_vec(buffer);
        }
        let mut freqs_start = self.term_infos[0]
            .freqs_range
            .as_ref()
            .map(|range| range.start);
        if let Some(start) = freqs_start {
            VInt(start as u64).serialize_into_vec(buffer);
        }
        for term_info in &self.term_infos {
            VInt(term_info.doc_freq as u64).serialize_into_vec(buffer);
            VInt(term_info.postings_range.len() as u64).serialize_into_vec(buffer);
            VInt(term_info.positions_range.len() as u64).serialize_into_vec(buffer);
            assert_eq!(
                term_info.freqs_range.as_ref().map(|range| range.start),
                freqs_start
            );
            if let Some(range) = &term_info.freqs_range {
                VInt(term_info.freqs_num_bytes() as u64).serialize_into_vec(buffer);
                freqs_start = Some(range.end);
            }
            assert_eq!(term_info.pnorms_offset, pnorms_offset);
            if let Some(offset) = &mut pnorms_offset {
                *offset += u64::from(term_info.doc_freq);
            }
        }
    }

    fn clear(&mut self) {
        self.term_infos.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use common::{BinarySerializable, VInt};
    use sstable::value::{ValueReader, ValueWriter};

    use crate::postings::TermInfo;
    use crate::termdict::sstable_termdict::TermInfoValueReader;

    #[test]
    fn empty_block_clears_previous_values() {
        let mut writer = super::TermInfoValueWriter::default();
        let mut reader = TermInfoValueReader::default();
        writer.write(&TermInfo::default());
        let mut bytes = Vec::new();
        writer.serialize_block(&mut bytes);
        reader.load(&bytes).unwrap();
        assert_eq!(reader.term_infos.len(), 1);
        writer.clear();
        bytes.clear();
        writer.serialize_block(&mut bytes);
        assert_eq!(reader.load(&bytes).unwrap(), bytes.len());
        assert!(reader.term_infos.is_empty());
    }

    #[test]
    fn rejects_unknown_block_version() {
        let mut bytes = Vec::new();
        VInt(super::VERSIONED_BLOCK).serialize_into_vec(&mut bytes);
        4u32.serialize(&mut bytes).unwrap();
        let error = TermInfoValueReader::default().load(&bytes).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("version 4"));
    }

    #[test]
    fn norm_offset_overhead_is_constant_per_block() {
        let mut overhead = None;
        for count in [1, 64, 1_024] {
            let mut plain = super::TermInfoValueWriter::default();
            let mut norms = super::TermInfoValueWriter::default();
            for i in 0..count {
                let mut info = TermInfo {
                    doc_freq: 1,
                    postings_range: i * 4..(i + 1) * 4,
                    positions_range: i..i + 1,
                    pnorms_offset: None,
                    freqs_range: None,
                };
                plain.write(&info);
                info.pnorms_offset = Some((1 << 40) + i as u64);
                norms.write(&info);
            }
            let mut plain_bytes = Vec::new();
            let mut norm_bytes = Vec::new();
            plain.serialize_block(&mut plain_bytes);
            norms.serialize_block(&mut norm_bytes);
            let extra_bytes = norm_bytes.len() - plain_bytes.len();
            assert_eq!(*overhead.get_or_insert(extra_bytes), extra_bytes);
        }
    }

    #[test]
    fn test_block_terminfos() {
        let mut term_info_writer = super::TermInfoValueWriter::default();
        term_info_writer.write(&TermInfo {
            doc_freq: 120u32,
            postings_range: 17..45,
            positions_range: 10..122,
            pnorms_offset: None,
            freqs_range: None,
        });
        term_info_writer.write(&TermInfo {
            doc_freq: 10u32,
            postings_range: 45..450,
            positions_range: 122..1100,
            pnorms_offset: None,
            freqs_range: None,
        });
        term_info_writer.write(&TermInfo {
            doc_freq: 17u32,
            postings_range: 450..462,
            positions_range: 1100..1302,
            pnorms_offset: None,
            freqs_range: None,
        });
        let mut buffer = Vec::new();
        term_info_writer.serialize_block(&mut buffer);
        let mut term_info_reader = TermInfoValueReader::default();
        let num_bytes: usize = term_info_reader.load(&buffer[..]).unwrap();
        assert_eq!(
            term_info_reader.value(0),
            &TermInfo {
                pnorms_offset: None,
                freqs_range: None,
                doc_freq: 120u32,
                postings_range: 17..45,
                positions_range: 10..122
            }
        );
        assert_eq!(buffer.len(), num_bytes);
    }
}
