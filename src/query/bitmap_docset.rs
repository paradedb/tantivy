use std::io;

use common::buffered_file_slice::BufferedFileSlice;
use common::{HasLen, TinySet};

use crate::directory::FileSlice;
use crate::docset::{SeekDangerResult, BLOCK_NUM_TINYBITSETS, BLOCK_WINDOW};
use crate::postings::term_bitmaps::bitmap_num_bytes;
use crate::{DocId, DocSet, TERMINATED};

/// A buffered, on-disk membership bitmap over segment-local document IDs.
pub struct BitmapDocSet {
    data: BufferedFileSlice,
    max_doc: DocId,
    doc_freq: u32,
    doc: DocId,
}

impl BitmapDocSet {
    /// Opens a term's bitmap, validating the payload bounds.
    pub fn open(source: FileSlice, offset: u64, max_doc: DocId, doc_freq: u32) -> io::Result<Self> {
        let end = offset.checked_add(bitmap_num_bytes(max_doc));
        if max_doc > TERMINATED || end.is_none_or(|end| end > source.len() as u64) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid posting bitmap bounds",
            ));
        }
        let mut docset = Self {
            data: BufferedFileSlice::new_block_aligned(
                source.slice(offset as usize..end.unwrap() as usize),
                8192,
            ),
            max_doc,
            doc_freq,
            doc: TERMINATED,
        };
        docset.doc = docset.next_doc(0);
        Ok(docset)
    }

    fn next_doc(&self, target: DocId) -> DocId {
        if target >= self.max_doc {
            return TERMINATED;
        }
        let mut offset = u64::from(target / 64) * 8;
        let end = bitmap_num_bytes(self.max_doc);
        let mut first_mask = u64::MAX << (target % 64);
        while offset < end {
            let bytes = self
                .data
                .read_chunk(offset, 8)
                .expect("failed to read posting bitmap");
            for bytes in bytes.as_chunks::<8>().0 {
                let word = u64::from_le_bytes(*bytes) & first_mask;
                if word != 0 {
                    let doc = offset as u32 * 8 + word.trailing_zeros();
                    return if doc < self.max_doc { doc } else { TERMINATED };
                }
                first_mask = u64::MAX;
                offset += 8;
            }
        }
        TERMINATED
    }
}

impl DocSet for BitmapDocSet {
    fn advance(&mut self) -> DocId {
        self.seek(self.doc.saturating_add(1))
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if target >= self.doc {
            self.doc = self.next_doc(target);
        }
        self.doc
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        if target >= self.max_doc {
            self.doc = TERMINATED;
            return SeekDangerResult::SeekLowerBound(TERMINATED);
        }
        if target < self.doc {
            return SeekDangerResult::SeekLowerBound(self.doc);
        }
        let byte = self
            .data
            .read_byte(u64::from(target / 8))
            .expect("failed to read posting bitmap");
        self.doc = target;
        if byte & (1 << (target % 8)) != 0 {
            SeekDangerResult::Found
        } else {
            SeekDangerResult::SeekLowerBound(target + 1)
        }
    }

    fn doc(&self) -> DocId {
        self.doc
    }

    fn size_hint(&self) -> u32 {
        self.doc_freq
    }

    fn has_fast_bitset(&self) -> bool {
        true
    }

    fn fill_bitset_block(
        &mut self,
        base: DocId,
        mask: &mut [TinySet; BLOCK_NUM_TINYBITSETS],
    ) -> DocId {
        let horizon = base.saturating_add(BLOCK_WINDOW).min(self.max_doc);
        let start = base.max(self.doc);
        if start >= horizon {
            return self.doc;
        }
        let offset = u64::from(base / 64) * 8;
        let bytes = self
            .data
            .get_bytes(offset..bitmap_num_bytes(horizon))
            .expect("failed to read posting bitmap");
        let shift = base % 64;
        if shift == 0 {
            let mut block = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
            for (word, bytes) in block.iter_mut().zip(bytes.as_chunks::<8>().0) {
                *word = TinySet::deserialize(*bytes);
            }
            crate::docset::retain_bitset_range(&mut block, base, start, horizon);
            crate::docset::union_bitset_blocks(mask, &block);
            self.doc = self.next_doc(horizon);
            return self.doc;
        }
        let word_at = |i: usize| {
            bytes
                .get(i * 8..i * 8 + 8)
                .map_or(0, |word| u64::from_le_bytes(word.try_into().unwrap()))
        };
        for (i, bucket) in mask.iter_mut().enumerate() {
            let word_base = base + i as u32 * 64;
            if word_base >= horizon || word_base + 64 <= start {
                continue;
            }
            let mut word = word_at(i) >> shift;
            if shift != 0 {
                word |= word_at(i + 1) << (64 - shift);
            }
            if start > word_base {
                word &= u64::MAX << (start - word_base);
            }
            if horizon - word_base < 64 {
                word &= (1u64 << (horizon - word_base)) - 1;
            }
            *bucket = bucket.union(TinySet::deserialize(word.to_le_bytes()));
        }
        self.doc = self.next_doc(horizon);
        self.doc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bitmap(docs: &[DocId], max_doc: DocId) -> BitmapDocSet {
        let mut data = vec![0u8; bitmap_num_bytes(max_doc) as usize];
        for &doc in docs {
            data[doc as usize / 8] |= 1 << (doc % 8);
        }
        BitmapDocSet::open(FileSlice::from(data), 0, max_doc, docs.len() as u32).unwrap()
    }

    #[derive(Debug)]
    struct TrackedBitmapFile {
        data: Vec<u8>,
        reads: std::sync::Arc<std::sync::Mutex<Vec<std::ops::Range<usize>>>>,
        block_len: Option<usize>,
    }

    impl HasLen for TrackedBitmapFile {
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    impl crate::directory::FileHandle for TrackedBitmapFile {
        fn read_bytes(&self, range: std::ops::Range<usize>) -> io::Result<common::OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            Ok(common::OwnedBytes::new(self.data[range].to_vec()))
        }

        fn storage_block_len(&self) -> Option<usize> {
            self.block_len
        }
    }

    #[test]
    fn bitmap_jump_does_not_reread_the_same_buffer() {
        use std::sync::{Arc, Mutex};

        for probe_first in [false, true] {
            let reads = Arc::new(Mutex::new(Vec::new()));
            let file = FileSlice::new(Arc::new(TrackedBitmapFile {
                data: vec![255; 32768],
                reads: reads.clone(),
                block_len: None,
            }));
            let mut bitmap = BitmapDocSet::open(file, 0, 262144, 262144).unwrap();
            if probe_first {
                assert_eq!(bitmap.seek_danger(66560), SeekDangerResult::Found);
            }
            let mut mask = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
            assert_eq!(bitmap.fill_bitset_block(66560, &mut mask), 67584);
            assert!(mask.iter().all(|word| word.len() == 64));
            assert_eq!(*reads.lock().unwrap(), vec![0..8192, 8192..16384]);
        }
    }

    #[test]
    fn bitmap_reads_respect_storage_blocks_and_nested_slice_offsets() {
        use std::sync::{Arc, Mutex};

        for prefix in [1, 151, 8155] {
            let reads = Arc::new(Mutex::new(Vec::new()));
            let max_doc = 300003;
            let len = bitmap_num_bytes(max_doc) as usize;
            let file = FileSlice::new(Arc::new(TrackedBitmapFile {
                data: vec![255; prefix + 37 + len + 19],
                reads: reads.clone(),
                block_len: Some(8156),
            }))
            .slice(37..);
            let mut bitmap = BitmapDocSet::open(file, prefix as u64, max_doc, max_doc).unwrap();
            for base in (0..max_doc).step_by(BLOCK_WINDOW as usize) {
                let mut mask = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
                let next = bitmap.fill_bitset_block(base, &mut mask);
                let expected = (max_doc - base).min(BLOCK_WINDOW);
                assert_eq!(mask.iter().map(|word| word.len()).sum::<u32>(), expected);
                assert_eq!(
                    next,
                    if base + expected == max_doc {
                        TERMINATED
                    } else {
                        base + expected
                    }
                );
            }
            let reads = reads.lock().unwrap();
            let mut end = prefix + 37;
            for range in reads.iter() {
                assert_eq!(range.start, end, "overlapping or skipped read: {reads:?}");
                assert!(range.end % 8156 == 0 || range.end == prefix + 37 + len);
                end = range.end;
            }
            assert_eq!(end, prefix + 37 + len);
        }
    }

    #[test]
    fn bitmap_empty_gaps_cross_storage_boundaries_without_rereads() {
        use std::sync::{Arc, Mutex};

        for block_len in [None, Some(8156)] {
            for prefix in [0, 1, 151, 8155] {
                let max_doc = 1_048_573;
                let docs = [0, 131_071, 800_003, max_doc - 1];
                let len = bitmap_num_bytes(max_doc) as usize;
                let mut data = vec![0; prefix + len];
                for doc in docs {
                    data[prefix + doc as usize / 8] |= 1 << (doc % 8);
                }
                let reads = Arc::new(Mutex::new(Vec::new()));
                let file = FileSlice::new(Arc::new(TrackedBitmapFile {
                    data,
                    reads: reads.clone(),
                    block_len,
                }));
                let mut bitmap = BitmapDocSet::open(file, prefix as u64, max_doc, 4).unwrap();
                for doc in docs {
                    assert_eq!(bitmap.doc(), doc);
                    bitmap.advance();
                }
                assert_eq!(bitmap.doc(), TERMINATED);
                let reads = reads.lock().unwrap();
                let mut end = prefix;
                for range in reads.iter() {
                    assert_eq!(range.start, end, "overlapping or skipped read: {reads:?}");
                    end = range.end;
                }
                assert_eq!(end, prefix + len);
                assert!(reads.len() <= 18, "{reads:?}");
            }
        }
        let mut tail = bitmap(&[0, 131_071], 1_048_576);
        assert_eq!(tail.seek(131_072), TERMINATED);
    }

    proptest::proptest! {
        #[test]
        fn bitmap_cursor_operations_match_postings(
            mut docs in proptest::collection::vec(0u32..5137, 0..900),
            actions in proptest::collection::vec((0u8..3, 0u32..300), 0..100),
        ) {
            docs.sort_unstable(); docs.dedup();
            let mut bitmap = bitmap(&docs, 5137);
            let mut reference = crate::query::VecDocSet::from(docs);
            for (action, gap) in actions {
                assert_eq!(bitmap.doc(), reference.doc());
                if reference.doc() == TERMINATED { break; }
                match action {
                    0 => { assert_eq!(bitmap.advance(), reference.advance()); }
                    1 => {
                        let target = reference.doc() + gap;
                        assert_eq!(bitmap.seek(target), reference.seek(target));
                    }
                    _ => {
                        let base = reference.doc().saturating_sub(gap);
                        let mut actual = [TinySet::singleton(7); BLOCK_NUM_TINYBITSETS];
                        let mut expected = actual;
                        assert_eq!(bitmap.fill_bitset_block(base, &mut actual), reference.fill_bitset_block(base, &mut expected));
                        assert_eq!(actual, expected);
                    }
                }
            }
            bitmap.seek(TERMINATED);
            assert_eq!(bitmap.seek_danger(0), SeekDangerResult::SeekLowerBound(TERMINATED));
        }
    }

    #[test]
    fn bitmap_selection_preserves_scored_postings_and_read_fallback() -> crate::Result<()> {
        use crate::query::{EnableScoring, Query, TermQuery, TermScorer};
        use crate::schema::{IndexRecordOption, Schema, TEXT};
        use crate::{Index, Term};
        let mut schema = Schema::builder();
        let field = schema.add_text_field(
            "text",
            TEXT.set_indexing_options(
                TEXT.get_indexing_options()
                    .unwrap()
                    .clone()
                    .set_bitmap_postings(true),
            ),
        );
        let mut index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        for doc in 0..1024 {
            writer.add_document(doc!(field => if doc % 2 == 0 { "common" } else { "other" }))?;
        }
        writer.commit()?;
        for enabled in [true, false] {
            index.settings_mut().bitmap_postings.use_for_queries = enabled;
            let searcher = index.reader()?.searcher();
            let query = TermQuery::new(
                Term::from_field_text(field, "common"),
                IndexRecordOption::WithFreqs,
            );
            for scoring in [false, true] {
                let weight = query.weight(if scoring {
                    EnableScoring::enabled_from_searcher(&searcher)
                } else {
                    EnableScoring::disabled_from_searcher(&searcher)
                })?;
                let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
                assert_eq!(scorer.has_fast_bitset(), enabled && !scoring);
                assert_eq!(scorer.is::<TermScorer>(), !enabled || scoring);
                for doc in (0..1024).step_by(2) {
                    assert_eq!(scorer.doc(), doc);
                    scorer.advance();
                }
                assert_eq!(scorer.doc(), TERMINATED);
            }
        }
        Ok(())
    }

    #[test]
    fn bitmap_iteration_probes_and_windows() {
        let max_doc = 70_013;
        let docs: Vec<_> = (0..max_doc)
            .filter(|doc| doc % 7 == 3 || *doc == max_doc - 1)
            .collect();
        let mut iter = bitmap(&docs, max_doc);
        for &doc in &docs {
            assert_eq!(iter.doc(), doc);
            iter.advance();
        }
        assert_eq!(iter.advance(), TERMINATED);
        assert_eq!(iter.seek(TERMINATED), TERMINATED);
        let mut iter = bitmap(&docs, max_doc);
        for doc in 0..max_doc {
            assert_eq!(
                iter.seek_danger(doc) == SeekDangerResult::Found,
                docs.binary_search(&doc).is_ok()
            );
        }
        for base in [0, 1, 63, 64, 1019, 65_533, 69_977, max_doc] {
            for skip in [0, 1, 63, 517, 1025] {
                let mut iter = bitmap(&docs, max_doc);
                let start = iter.seek(base + skip);
                let mut mask = [TinySet::empty(); BLOCK_NUM_TINYBITSETS];
                let next = iter.fill_bitset_block(base, &mut mask);
                let result: Vec<_> = mask
                    .into_iter()
                    .enumerate()
                    .flat_map(|(i, word)| {
                        word.into_iter().map(move |bit| base + i as u32 * 64 + bit)
                    })
                    .collect();
                let expected: Vec<_> = docs
                    .iter()
                    .copied()
                    .filter(|&doc| doc >= start && doc >= base && doc < base + BLOCK_WINDOW)
                    .collect();
                assert_eq!(result, expected, "base {base}, skip {skip}");
                assert_eq!(
                    next,
                    docs.iter()
                        .copied()
                        .find(|&doc| doc >= (base + BLOCK_WINDOW).max(start))
                        .unwrap_or(TERMINATED)
                );
            }
        }
        assert!(BitmapDocSet::open(FileSlice::empty(), 0, 65, 0).is_err());
        assert_eq!(bitmap(&[], 0).doc(), TERMINATED);
    }
}
