use std::io;

mod merger;

use std::iter::ExactSizeIterator;

use common::VInt;
use sstable::streamer::StreamerWithState;
use sstable::value::{ValueReader, ValueWriter};
use sstable::SSTable;
use tantivy_fst::automaton::AlwaysMatch;
use tantivy_fst::Automaton;

pub use self::merger::TermMerger;
use crate::postings::TermInfo;

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
        let num_els = VInt::deserialize_u64(&mut data)?;
        let mut postings_start = VInt::deserialize_u64(&mut data)? as usize;
        let mut positions_start = VInt::deserialize_u64(&mut data)? as usize;

        self.term_infos.clear();
        self.term_infos.reserve_exact(num_els as usize);
        for _ in 0..num_els {
            let doc_freq = VInt::deserialize_u64(&mut data)? as u32;
            let postings_num_bytes = VInt::deserialize_u64(&mut data)?;
            let positions_num_bytes = VInt::deserialize_u64(&mut data)?;
            let postings_end = postings_start + postings_num_bytes as usize;
            let positions_end = positions_start + positions_num_bytes as usize;
            let term_info = TermInfo {
                doc_freq,
                postings_range: postings_start..postings_end,
                positions_range: positions_start..positions_end,
            };
            self.term_infos.push(term_info);
            postings_start = postings_end;
            positions_start = positions_end;
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
        VInt(self.term_infos.len() as u64).serialize_into_vec(buffer);
        if self.term_infos.is_empty() {
            return;
        }
        VInt(self.term_infos[0].postings_range.start as u64).serialize_into_vec(buffer);
        VInt(self.term_infos[0].positions_range.start as u64).serialize_into_vec(buffer);
        for term_info in &self.term_infos {
            VInt(term_info.doc_freq as u64).serialize_into_vec(buffer);
            VInt(term_info.postings_range.len() as u64).serialize_into_vec(buffer);
            VInt(term_info.positions_range.len() as u64).serialize_into_vec(buffer);
        }
    }

    fn clear(&mut self) {
        self.term_infos.clear();
    }
}

#[cfg(test)]
mod tests {
    use sstable::value::{ValueReader, ValueWriter};

    use crate::postings::TermInfo;
    use crate::termdict::sstable_termdict::TermInfoValueReader;

    #[test]
    fn test_block_terminfos() {
        let mut term_info_writer = super::TermInfoValueWriter::default();
        term_info_writer.write(&TermInfo {
            doc_freq: 120u32,
            postings_range: 17..45,
            positions_range: 10..122,
        });
        term_info_writer.write(&TermInfo {
            doc_freq: 10u32,
            postings_range: 45..450,
            positions_range: 122..1100,
        });
        term_info_writer.write(&TermInfo {
            doc_freq: 17u32,
            postings_range: 450..462,
            positions_range: 1100..1302,
        });
        let mut buffer = Vec::new();
        term_info_writer.serialize_block(&mut buffer);
        let mut term_info_reader = TermInfoValueReader::default();
        let num_bytes: usize = term_info_reader.load(&buffer[..]).unwrap();
        assert_eq!(
            term_info_reader.value(0),
            &TermInfo {
                doc_freq: 120u32,
                postings_range: 17..45,
                positions_range: 10..122
            }
        );
        assert_eq!(buffer.len(), num_bytes);
    }
}
