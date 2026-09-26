use std::io;
use std::ops::Range;
use std::sync::Arc;

use common::{BinarySerializable, FixedSize};
use rustc_hash::FxHashMap;

use super::SegmentPostings;
use crate::index::{SegmentId, SegmentReader};
use crate::schema::IndexRecordOption;
use crate::Term;

/// `TermInfo` wraps the metadata associated with a Term.
/// It is segment-local.
#[derive(Debug, Default, Eq, PartialEq, Clone)]
pub struct TermInfo {
    /// Number of documents in the segment containing the term
    pub doc_freq: u32,
    /// Byte range of the posting list within the postings (`.idx`) file.
    pub postings_range: Range<usize>,
    /// Byte range of the positions of this terms in the positions (`.pos`) file.
    pub positions_range: Range<usize>,
}

#[derive(Clone, Default)]
pub(crate) struct ResolvedTermInfo {
    pub doc_freq: u64,
    pub segments: Option<Arc<FxHashMap<SegmentId, Option<TermInfo>>>>,
}

impl ResolvedTermInfo {
    pub fn get(&self, reader: &SegmentReader, term: &Term) -> crate::Result<Option<TermInfo>> {
        match self
            .segments
            .as_ref()
            .and_then(|segments| segments.get(&reader.segment_id()))
        {
            Some(info) => Ok(info.clone()),
            None => Ok(reader.inverted_index(term.field())?.get_term_info(term)?),
        }
    }

    pub fn read_postings(
        &self,
        reader: &SegmentReader,
        term: &Term,
        option: IndexRecordOption,
    ) -> crate::Result<Option<SegmentPostings>> {
        let Some(info) = self.get(reader, term)? else {
            return Ok(None);
        };
        Ok(Some(
            reader
                .inverted_index(term.field())?
                .read_postings_from_terminfo(&info, option)?,
        ))
    }
}

impl TermInfo {
    pub(crate) fn posting_num_bytes(&self) -> u32 {
        let num_bytes = self.postings_range.len();
        assert!(num_bytes <= u32::MAX as usize);
        num_bytes as u32
    }

    pub(crate) fn positions_num_bytes(&self) -> u32 {
        let num_bytes = self.positions_range.len();
        assert!(num_bytes <= u32::MAX as usize);
        num_bytes as u32
    }
}

impl FixedSize for TermInfo {
    /// Size required for the binary serialization of a `TermInfo` object.
    /// This is large, but in practise, `TermInfo` are encoded in blocks and
    /// only the first `TermInfo` of a block is serialized uncompressed.
    /// The subsequent `TermInfo` are delta encoded and bitpacked.
    const SIZE_IN_BYTES: usize = 3 * u32::SIZE_IN_BYTES + 2 * u64::SIZE_IN_BYTES;
}

impl BinarySerializable for TermInfo {
    fn serialize<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        self.doc_freq.serialize(writer)?;
        (self.postings_range.start as u64).serialize(writer)?;
        self.posting_num_bytes().serialize(writer)?;
        (self.positions_range.start as u64).serialize(writer)?;
        self.positions_num_bytes().serialize(writer)?;
        Ok(())
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let doc_freq = u32::deserialize(reader)?;
        let postings_start_offset = u64::deserialize(reader)? as usize;
        let postings_num_bytes = u32::deserialize(reader)? as usize;
        let postings_end_offset = postings_start_offset + postings_num_bytes;
        let positions_start_offset = u64::deserialize(reader)? as usize;
        let positions_num_bytes = u32::deserialize(reader)? as usize;
        let positions_end_offset = positions_start_offset + positions_num_bytes;
        Ok(TermInfo {
            doc_freq,
            postings_range: postings_start_offset..postings_end_offset,
            positions_range: positions_start_offset..positions_end_offset,
        })
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::indexer::NoMergePolicy;
    use crate::schema::{Schema, TEXT};
    use crate::tests::fixed_size_test;
    use crate::{DocSet, Index, IndexWriter};

    // TODO add serialize/deserialize test for terminfo

    #[test]
    fn test_fixed_size() {
        fixed_size_test::<TermInfo>();
    }

    #[test]
    fn resolved_term_info_reuses_metadata_and_falls_back() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let field = schema.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema.build());
        let mut writer: IndexWriter = index.writer_for_tests()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        writer.add_document(doc!(field => "rust memory"))?;
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        let segment = searcher.segment_reader(0);
        let term = Term::from_field_text(field, "rust");
        let missing = Term::from_field_text(field, "missing");
        let info = segment.inverted_index(field)?.get_term_info(&term)?;
        let resolved = ResolvedTermInfo {
            doc_freq: 1,
            segments: Some(Arc::new(
                [(segment.segment_id(), info.clone())].into_iter().collect(),
            )),
        };
        let absent = ResolvedTermInfo {
            doc_freq: 0,
            segments: Some(Arc::new(
                [(segment.segment_id(), None)].into_iter().collect(),
            )),
        };
        let reopened = index.reader()?.searcher();
        let same_segment = reopened.segment_reader(0);
        // Mismatched lookup keys prove the cached entry wins over the dictionary.
        assert_eq!(resolved.get(same_segment, &missing)?, info);
        assert_eq!(
            resolved
                .read_postings(same_segment, &missing, IndexRecordOption::Basic)?
                .unwrap()
                .doc(),
            0
        );
        assert!(absent
            .read_postings(same_segment, &term, IndexRecordOption::Basic)?
            .is_none());
        assert_eq!(ResolvedTermInfo::default().get(segment, &term)?, info);

        writer.add_document(doc!(field => "rust rust"))?;
        writer.commit()?;
        let updated = index.reader()?.searcher();
        let new_segment = updated
            .segment_readers()
            .iter()
            .find(|other| other.segment_id() != segment.segment_id())
            .unwrap();
        assert_eq!(
            resolved.get(new_segment, &term)?,
            new_segment.inverted_index(field)?.get_term_info(&term)?
        );
        assert!(resolved.get(new_segment, &missing)?.is_none());
        Ok(())
    }
}
