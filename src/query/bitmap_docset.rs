//! Reads the on-disk bitmap for one dense term. Each set bit identifies a document
//! containing that term. It supports individual document probes and reading
//! 8,192-document masks for bitmap combinations and count collection.
//!
//! Ordinary postings remain available for scoring and positions. For legacy flat
//! bitmaps, they are opened lazily to seek across long empty regions.

use std::io;

use common::buffered_file_slice::BufferedFileSlice;
use common::{HasLen, TinySet};

use crate::directory::FileSlice;
use crate::docset::{SeekDangerResult, BLOCK_NUM_TINYBITSETS, BLOCK_WINDOW};
use crate::postings::term_bitmaps::{
    bitmap_block_count, bitmap_summary_num_bytes, flat_bitmap_num_bytes,
    HIERARCHICAL_BITMAP_ENTRY_LEN, HIERARCHICAL_BITMAP_HEADER_LEN, HIERARCHICAL_BITMAP_MAGIC,
    SPARSE_BLOCK_MAX_DOCS,
};
use crate::postings::SegmentPostings;
use crate::{DocId, DocSet, TERMINATED};

#[derive(Clone, Copy)]
struct HierarchicalEntry {
    count: u16,
    payload_offset: u32,
}

struct HierarchicalIndex {
    summary: Vec<u64>,
    summary_prefix: Vec<u32>,
    entries: Vec<HierarchicalEntry>,
    payload_base: u64,
}

enum BitmapStorage {
    Flat,
    Hierarchical(HierarchicalIndex),
}

/// A buffered, on-disk membership bitmap over segment-local document IDs.
pub struct BitmapDocSet {
    data: BufferedFileSlice,
    storage: BitmapStorage,
    max_doc: DocId,
    doc_freq: u32,
    doc: DocId,
    postings: Option<Box<SegmentPostings>>,
    open_postings: Option<Box<dyn FnOnce() -> io::Result<SegmentPostings> + Send>>,
}

impl BitmapDocSet {
    /// Opens a term's bitmap, validating the payload bounds.
    /// Ordinary postings are opened lazily to navigate long empty regions in legacy flat bitmaps.
    pub fn open(
        source: FileSlice,
        offset: u64,
        max_doc: DocId,
        doc_freq: u32,
        open_postings: impl FnOnce() -> io::Result<SegmentPostings> + Send + 'static,
    ) -> io::Result<Self> {
        if max_doc > TERMINATED || offset > source.len() as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid posting bitmap bounds",
            ));
        }
        let offset = offset as usize;
        let available = source.len() - offset;
        let data = BufferedFileSlice::new_block_aligned(source.slice(offset..), 8192);
        let probe_len = HIERARCHICAL_BITMAP_HEADER_LEN.min(available);
        let probe = data.get_bytes(0..probe_len as u64)?;
        let storage = if probe.starts_with(&HIERARCHICAL_BITMAP_MAGIC) {
            if probe.len() < HIERARCHICAL_BITMAP_HEADER_LEN {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated hierarchical posting bitmap header",
                ));
            }
            let present_count = u32::from_le_bytes(probe[8..12].try_into().unwrap()) as usize;
            let payload_len = u32::from_le_bytes(probe[12..16].try_into().unwrap()) as usize;
            let summary_len = bitmap_summary_num_bytes(max_doc);
            let directory_len = present_count
                .checked_mul(HIERARCHICAL_BITMAP_ENTRY_LEN)
                .ok_or_else(|| invalid_bitmap("hierarchical posting bitmap directory overflow"))?;
            let metadata_len = HIERARCHICAL_BITMAP_HEADER_LEN
                .checked_add(summary_len)
                .and_then(|len| len.checked_add(directory_len))
                .ok_or_else(|| invalid_bitmap("hierarchical posting bitmap metadata overflow"))?;
            let total_len = metadata_len
                .checked_add(payload_len)
                .ok_or_else(|| invalid_bitmap("hierarchical posting bitmap payload overflow"))?;
            if total_len > available {
                return Err(invalid_bitmap("truncated hierarchical posting bitmap"));
            }
            let metadata = data.get_bytes(0..metadata_len as u64)?;
            let mut summary = Vec::with_capacity(summary_len / 8);
            for bytes in metadata
                [HIERARCHICAL_BITMAP_HEADER_LEN..HIERARCHICAL_BITMAP_HEADER_LEN + summary_len]
                .chunks_exact(8)
            {
                summary.push(u64::from_le_bytes(bytes.try_into().unwrap()));
            }
            if summary.last().is_some_and(|last| {
                let valid = bitmap_block_count(max_doc) % 64;
                valid != 0 && *last & (u64::MAX << valid) != 0
            }) {
                return Err(invalid_bitmap(
                    "posting bitmap summary contains out-of-range blocks",
                ));
            }
            let summary_count: usize = summary.iter().map(|word| word.count_ones() as usize).sum();
            if summary_count != present_count {
                return Err(invalid_bitmap(
                    "posting bitmap summary and directory disagree",
                ));
            }
            let mut summary_prefix = Vec::with_capacity(summary.len() + 1);
            summary_prefix.push(0);
            for word in &summary {
                summary_prefix.push(summary_prefix.last().unwrap() + word.count_ones());
            }
            let directory = &metadata[HIERARCHICAL_BITMAP_HEADER_LEN + summary_len..metadata_len];
            let mut entries = Vec::with_capacity(present_count);
            let mut expected_offset = 0usize;
            for bytes in directory.chunks_exact(HIERARCHICAL_BITMAP_ENTRY_LEN) {
                let count = u16::from_le_bytes(bytes[0..2].try_into().unwrap());
                let payload_offset = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                if payload_offset as usize != expected_offset || count == 0 {
                    return Err(invalid_bitmap("invalid hierarchical posting bitmap entry"));
                }
                expected_offset = expected_offset
                    .checked_add(if count as usize <= SPARSE_BLOCK_MAX_DOCS {
                        count as usize * 2
                    } else {
                        BLOCK_NUM_TINYBITSETS * 8
                    })
                    .ok_or_else(|| invalid_bitmap("posting bitmap entry overflow"))?;
                entries.push(HierarchicalEntry {
                    count,
                    payload_offset,
                });
            }
            if expected_offset != payload_len {
                return Err(invalid_bitmap("posting bitmap payload length mismatch"));
            }
            BitmapStorage::Hierarchical(HierarchicalIndex {
                summary,
                summary_prefix,
                entries,
                payload_base: metadata_len as u64,
            })
        } else {
            let flat_len = flat_bitmap_num_bytes(max_doc) as usize;
            if flat_len > available {
                return Err(invalid_bitmap("invalid posting bitmap bounds"));
            }
            BitmapStorage::Flat
        };
        let mut docset = Self {
            data,
            storage,
            max_doc,
            doc_freq,
            doc: TERMINATED,
            postings: None,
            open_postings: Some(Box::new(open_postings)),
        };
        docset.doc = docset.next_doc(0);
        Ok(docset)
    }

    #[inline]
    fn next_doc(&mut self, target: DocId) -> DocId {
        match &self.storage {
            BitmapStorage::Flat => self.next_flat_doc(target),
            BitmapStorage::Hierarchical(index) => self.next_hierarchical_doc(index, target),
        }
    }

    fn next_flat_doc(&mut self, target: DocId) -> DocId {
        if target >= self.max_doc {
            return TERMINATED;
        }
        let mut offset = u64::from(target / 64) * 8;
        let start_offset = offset;
        let end = flat_bitmap_num_bytes(self.max_doc);
        let mut first_mask = u64::MAX << (target % 64);
        'scan: while offset < end {
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
                if offset - start_offset >= 8192 {
                    break 'scan;
                }
            }
        }
        if offset == end {
            return TERMINATED;
        }
        // Keep local scans cheap; use the existing postings skips across long empty regions.
        self.seek_postings((offset * 8) as DocId)
    }

    #[cold]
    fn seek_postings(&mut self, target: DocId) -> DocId {
        let postings = self.postings.get_or_insert_with(|| {
            Box::new(
                self.open_postings.take().unwrap()()
                    .expect("failed to open bitmap navigation postings"),
            )
        });
        if postings.doc() < target {
            postings.seek(target)
        } else {
            postings.doc()
        }
    }

    fn next_hierarchical_doc(&self, index: &HierarchicalIndex, target: DocId) -> DocId {
        if target >= self.max_doc {
            return TERMINATED;
        }
        let mut block = (target / BLOCK_WINDOW) as usize;
        while let Some((next_block, entry_ord)) = index.next_present_block(block) {
            let entry = index.entries[entry_ord];
            let block_base = next_block as u32 * BLOCK_WINDOW;
            let relative_target = target.saturating_sub(block_base);
            let payload_start = index.payload_base + u64::from(entry.payload_offset);
            if entry.count as usize <= SPARSE_BLOCK_MAX_DOCS {
                let payload_len = u64::from(entry.count) * 2;
                let bytes = self
                    .data
                    .get_bytes(payload_start..payload_start + payload_len)
                    .expect("failed to read sparse posting bitmap block");
                for raw in bytes.chunks_exact(2) {
                    let relative = u16::from_le_bytes(raw.try_into().unwrap()) as u32;
                    if relative >= relative_target {
                        return block_base + relative;
                    }
                }
            } else {
                let word_start = (relative_target / 64) as usize;
                let payload_len = (BLOCK_NUM_TINYBITSETS * 8) as u64;
                let bytes = self
                    .data
                    .get_bytes(payload_start..payload_start + payload_len)
                    .expect("failed to read dense posting bitmap block");
                let mut first_mask = u64::MAX << (relative_target % 64);
                for (word_ord, raw) in bytes.chunks_exact(8).enumerate().skip(word_start) {
                    let word = u64::from_le_bytes(raw.try_into().unwrap()) & first_mask;
                    if word != 0 {
                        let doc = block_base + word_ord as u32 * 64 + word.trailing_zeros();
                        return if doc < self.max_doc { doc } else { TERMINATED };
                    }
                    first_mask = u64::MAX;
                }
            }
            block = next_block + 1;
        }
        TERMINATED
    }

    fn fill_hierarchical_block(
        &self,
        index: &HierarchicalIndex,
        base: DocId,
        start: DocId,
        horizon: DocId,
        mask: &mut [TinySet; BLOCK_NUM_TINYBITSETS],
    ) {
        let first_block = (start / BLOCK_WINDOW) as usize;
        let last_block = ((horizon - 1) / BLOCK_WINDOW) as usize;
        let mut block = first_block;
        let mut result = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
        while let Some((present_block, entry_ord)) = index.next_present_block(block) {
            if present_block > last_block {
                break;
            }
            let entry = index.entries[entry_ord];
            let block_base = present_block as u32 * BLOCK_WINDOW;
            let payload_start = index.payload_base + u64::from(entry.payload_offset);
            if entry.count as usize <= SPARSE_BLOCK_MAX_DOCS {
                let payload_len = u64::from(entry.count) * 2;
                let bytes = self
                    .data
                    .get_bytes(payload_start..payload_start + payload_len)
                    .expect("failed to read sparse posting bitmap block");
                for raw in bytes.chunks_exact(2) {
                    let doc = block_base + u16::from_le_bytes(raw.try_into().unwrap()) as u32;
                    if doc >= start && doc < horizon {
                        let relative = doc - base;
                        result[(relative / 64) as usize].insert_mut(relative % 64);
                    }
                }
            } else {
                let payload_len = (BLOCK_NUM_TINYBITSETS * 8) as u64;
                let bytes = self
                    .data
                    .get_bytes(payload_start..payload_start + payload_len)
                    .expect("failed to read dense posting bitmap block");
                if block_base == base {
                    for (word, raw) in result.iter_mut().zip(bytes.chunks_exact(8)) {
                        *word = TinySet::deserialize(raw.try_into().unwrap());
                    }
                } else {
                    for (word_ord, raw) in bytes.chunks_exact(8).enumerate() {
                        let word_base = block_base + word_ord as u32 * 64;
                        if word_base >= horizon || word_base + 64 <= base {
                            continue;
                        }
                        let mut bits = u64::from_le_bytes(raw.try_into().unwrap());
                        if word_base < base {
                            bits >>= base - word_base;
                            result[0] = result[0].union(TinySet::deserialize(bits.to_le_bytes()));
                        } else {
                            let delta = word_base - base;
                            let bucket = delta as usize / 64;
                            let shift = delta % 64;
                            result[bucket] = result[bucket]
                                .union(TinySet::deserialize((bits << shift).to_le_bytes()));
                            if shift != 0 && bucket + 1 < result.len() {
                                result[bucket + 1] = result[bucket + 1].union(
                                    TinySet::deserialize((bits >> (64 - shift)).to_le_bytes()),
                                );
                            }
                        }
                    }
                }
            }
            block = present_block + 1;
        }
        crate::docset::retain_bitset_range(&mut result, base, start, horizon);
        crate::docset::union_bitset_blocks(mask, &result);
    }
}

impl HierarchicalIndex {
    fn next_present_block(&self, block: usize) -> Option<(usize, usize)> {
        let mut word_ord = block / 64;
        let mut word = *self.summary.get(word_ord)? & (u64::MAX << (block % 64));
        while word == 0 {
            word_ord += 1;
            word = *self.summary.get(word_ord)?;
        }
        let block = word_ord * 64 + word.trailing_zeros() as usize;
        let lower = self.summary[word_ord] & ((1u64 << (block % 64)).wrapping_sub(1));
        let entry_ord = self.summary_prefix[word_ord] as usize + lower.count_ones() as usize;
        Some((block, entry_ord))
    }
}

fn invalid_bitmap(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
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
        self.doc = self.next_doc(target);
        if self.doc == target {
            SeekDangerResult::Found
        } else {
            SeekDangerResult::SeekLowerBound(self.doc)
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
        if let BitmapStorage::Hierarchical(index) = &self.storage {
            self.fill_hierarchical_block(index, base, start, horizon, mask);
            self.doc = self.next_doc(horizon);
            return self.doc;
        }
        let offset = u64::from(base / 64) * 8;
        let bytes = self
            .data
            .get_bytes(offset..flat_bitmap_num_bytes(horizon))
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

    fn hierarchical_bitmap(docs: &[DocId], max_doc: DocId) -> BitmapDocSet {
        let data = crate::postings::term_bitmaps::encode_hierarchical_bitmap(max_doc, docs, &[]);
        BitmapDocSet::open(FileSlice::from(data), 0, max_doc, docs.len() as u32, || {
            panic!("hierarchical bitmap navigation should not open postings")
        })
        .unwrap()
    }

    fn bitmap(docs: &[DocId], max_doc: DocId) -> BitmapDocSet {
        let mut data = vec![0u8; flat_bitmap_num_bytes(max_doc) as usize];
        for &doc in docs {
            data[doc as usize / 8] |= 1 << (doc % 8);
        }
        let docs = docs.to_vec();
        BitmapDocSet::open(
            FileSlice::from(data),
            0,
            max_doc,
            docs.len() as u32,
            move || Ok(SegmentPostings::create_from_docs(&docs)),
        )
        .unwrap()
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
            let mut bitmap = BitmapDocSet::open(file, 0, 262144, 262144, || {
                panic!("dense bitmap should not open postings")
            })
            .unwrap();
            if probe_first {
                assert_eq!(bitmap.seek_danger(66560), SeekDangerResult::Found);
            }
            let mut mask = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
            assert_eq!(bitmap.fill_bitset_block(66560, &mut mask), 74752);
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
            let len = flat_bitmap_num_bytes(max_doc) as usize;
            let file = FileSlice::new(Arc::new(TrackedBitmapFile {
                data: vec![255; prefix + 37 + len + 19],
                reads: reads.clone(),
                block_len: Some(8156),
            }))
            .slice(37..);
            let mut bitmap = BitmapDocSet::open(file, prefix as u64, max_doc, max_doc, || {
                panic!("dense bitmap should not open postings")
            })
            .unwrap();
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
            let source_end = prefix + 37 + len + 19;
            for range in reads.iter() {
                assert_eq!(range.start, end, "overlapping or skipped read: {reads:?}");
                assert!(range.end % 8156 == 0 || range.end == source_end);
                end = range.end;
            }
            assert!(end >= prefix + 37 + len);
            assert!(end <= source_end);
        }
    }

    #[test]
    fn bitmap_navigation_skips_large_empty_prefix_gaps_and_tail() {
        use std::sync::{Arc, Mutex};

        let max_doc = 100_000_000;
        let docs = [90_000_000, 90_000_002, 99_000_000];
        let mut data = vec![0; flat_bitmap_num_bytes(max_doc) as usize];
        for doc in docs {
            data[doc as usize / 8] |= 1 << (doc % 8);
        }
        let reads = Arc::new(Mutex::new(Vec::new()));
        let file = FileSlice::new(Arc::new(TrackedBitmapFile {
            data,
            reads: reads.clone(),
            block_len: Some(8156),
        }));
        let mut bitmap = BitmapDocSet::open(file, 0, max_doc, 3, move || {
            Ok(SegmentPostings::create_from_docs(&docs))
        })
        .unwrap();
        assert_eq!(bitmap.doc(), docs[0]);
        assert!(reads.lock().unwrap().iter().map(|r| r.len()).sum::<usize>() <= 16312);
        assert_eq!(bitmap.advance(), docs[1]);
        assert_eq!(bitmap.seek(95_000_000), docs[2]);
        assert_eq!(bitmap.advance(), TERMINATED);
        assert!(reads.lock().unwrap().iter().map(|r| r.len()).sum::<usize>() < 65536);
    }

    #[test]
    fn bitmap_empty_gaps_cross_storage_boundaries_without_rereads() {
        use std::sync::{Arc, Mutex};

        for block_len in [None, Some(8156)] {
            for prefix in [0, 1, 151, 8155] {
                let max_doc = 1_048_573;
                let docs = [0, 131_071, 800_003, max_doc - 1];
                let len = flat_bitmap_num_bytes(max_doc) as usize;
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
                let mut bitmap = BitmapDocSet::open(file, prefix as u64, max_doc, 4, move || {
                    Ok(SegmentPostings::create_from_docs(&docs))
                })
                .unwrap();
                for doc in docs {
                    assert_eq!(bitmap.doc(), doc);
                    bitmap.advance();
                }
                assert_eq!(bitmap.doc(), TERMINATED);
                let reads = reads.lock().unwrap();
                let mut end = prefix;
                for range in reads.iter() {
                    assert!(range.start >= end, "overlapping read: {reads:?}");
                    end = range.end;
                }
                assert!(end <= prefix + len);
                assert!(reads.len() <= 8, "{reads:?}");
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
            scale in proptest::sample::select(vec![1u32, 4096]),
        ) {
            for doc in &mut docs { *doc *= scale; }
            docs.sort_unstable(); docs.dedup();
            let mut bitmap = bitmap(&docs, 5137 * scale);
            let mut reference = crate::query::VecDocSet::from(docs);
            for (action, gap) in actions {
                assert_eq!(bitmap.doc(), reference.doc());
                if reference.doc() == TERMINATED { break; }
                match action {
                    0 => { assert_eq!(bitmap.advance(), reference.advance()); }
                    1 => {
                        let target = reference.doc() + gap * scale;
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
        use crate::query::{
            BoostQuery, ConstScoreQuery, EnableScoring, FuzzyTermQuery, Query, RegexQuery,
            TermQuery, TermScorer,
        };
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
            let term = Term::from_field_text(field, "common");
            let query = TermQuery::new(term.clone(), IndexRecordOption::WithFreqs);
            let queries: Vec<Box<dyn Query>> = vec![
                Box::new(query.clone()),
                Box::new(RegexQuery::from_pattern("common", field)?),
                Box::new(FuzzyTermQuery::new(term, 0, false)),
                Box::new(BoostQuery::new(query.clone(), 2.0)),
                Box::new(ConstScoreQuery::new(query, 2.0)),
            ];
            for query in queries {
                for (scoring, opt_in) in
                    [(false, false), (false, true), (true, false), (true, true)]
                {
                    let weight = query.weight(
                        (if scoring {
                            EnableScoring::enabled_from_searcher(&searcher)
                        } else {
                            EnableScoring::disabled_from_searcher(&searcher)
                        })
                        .with_bitmap_postings(opt_in),
                    )?;
                    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
                    assert_eq!(scorer.has_fast_bitset(), enabled && opt_in && !scoring);
                    if query.is::<TermQuery>() {
                        assert_eq!(scorer.is::<TermScorer>(), !enabled || !opt_in || scoring);
                    }
                    for doc in (0..1024).step_by(2) {
                        assert_eq!(scorer.doc(), doc);
                        scorer.advance();
                    }
                    assert_eq!(scorer.doc(), TERMINATED);
                }
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
        assert!(BitmapDocSet::open(
            FileSlice::empty(),
            0,
            65,
            0,
            || Ok(SegmentPostings::empty())
        )
        .is_err());
        assert_eq!(bitmap(&[], 0).doc(), TERMINATED);
    }

    #[test]
    fn hierarchical_bitmap_skips_empty_blocks_and_decodes_sparse_and_dense_blocks() {
        let max_doc = BLOCK_WINDOW * 4 + 123;
        let mut docs = vec![3, 70, 500];
        docs.extend((BLOCK_WINDOW * 2..BLOCK_WINDOW * 3).step_by(2));
        docs.extend([BLOCK_WINDOW * 3 + 1, BLOCK_WINDOW * 4 - 1]);

        let mut iter = hierarchical_bitmap(&docs, max_doc);
        for &doc in &docs {
            assert_eq!(iter.doc(), doc);
            iter.advance();
        }
        assert_eq!(iter.doc(), TERMINATED);

        for base in [
            0,
            1,
            BLOCK_WINDOW - 63,
            BLOCK_WINDOW,
            BLOCK_WINDOW + 17,
            BLOCK_WINDOW * 2,
            BLOCK_WINDOW * 2 + 11,
            BLOCK_WINDOW * 3 - 31,
        ] {
            let mut actual = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
            let mut expected = actual;
            let mut hierarchical = hierarchical_bitmap(&docs, max_doc);
            let mut reference = crate::query::VecDocSet::from(docs.clone());
            assert_eq!(
                hierarchical.fill_bitset_block(base, &mut actual),
                reference.fill_bitset_block(base, &mut expected),
                "base {base}",
            );
            assert_eq!(actual, expected, "base {base}");
        }
    }
}
