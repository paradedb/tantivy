use common::{BitSet, TinySet};

use crate::docset::{DocSet, BLOCK_NUM_TINYBITSETS, BLOCK_WINDOW, TERMINATED};
use crate::DocId;

/// A `BitSetDocSet` makes it possible to iterate through a bitset as if it was a `DocSet`.
///
/// # Implementation detail
///
/// Skipping is relatively fast here as we can directly point to the
/// right tiny bitset bucket.
///
/// TODO: Consider implementing a `BitTreeSet` in order to advance faster
/// when the bitset is sparse
pub struct BitSetDocSet {
    docs: BitSet,
    cursor_bucket: u32, //< index associated with the current tiny bitset
    cursor_tinybitset: TinySet,
    doc: u32,
}

impl BitSetDocSet {
    fn go_to_bucket(&mut self, bucket_addr: u32) {
        self.cursor_bucket = bucket_addr;
        self.cursor_tinybitset = self.docs.tinyset(bucket_addr);
    }
}

impl From<BitSet> for BitSetDocSet {
    fn from(docs: BitSet) -> BitSetDocSet {
        let first_tiny_bitset = if docs.max_value() == 0 {
            TinySet::empty()
        } else {
            docs.tinyset(0)
        };
        let mut docset = BitSetDocSet {
            docs,
            cursor_bucket: 0,
            cursor_tinybitset: first_tiny_bitset,
            doc: 0u32,
        };
        docset.advance();
        docset
    }
}

impl DocSet for BitSetDocSet {
    #[inline]
    fn advance(&mut self) -> DocId {
        if let Some(lower) = self.cursor_tinybitset.pop_lowest() {
            self.doc = (self.cursor_bucket * 64u32) | lower;
            return self.doc;
        }
        if let Some(cursor_bucket) = self.docs.first_non_empty_bucket(self.cursor_bucket + 1) {
            self.go_to_bucket(cursor_bucket);
            let lower = self.cursor_tinybitset.pop_lowest().unwrap();
            self.doc = (cursor_bucket * 64u32) | lower;
            self.doc
        } else {
            self.doc = TERMINATED;
            TERMINATED
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        // DocSet contract: seek targets are monotonically non-decreasing.
        // If target is at or before our current doc, we're already past
        // it — return as-is. Also covers the post-TERMINATED case, and
        // the seek-to-current case that mask+advance below wouldn't:
        // advance() has already popped self.doc's bit out of
        // cursor_tinybitset, so masking with `>= target` would skip it.
        if target <= self.doc {
            return self.doc;
        }

        // Out-of-range target: nothing past `max_value` can match.
        // Terminates and keeps the bucket math below within the
        // tinysets array.
        if target >= self.docs.max_value() {
            self.doc = TERMINATED;
            return TERMINATED;
        }

        // Jump to target_bucket if it's ahead of where we are. Past the
        // guard above we know target > self.doc, so target_bucket is
        // always >= cursor_bucket.
        let target_bucket = target / 64u32;
        if target_bucket > self.cursor_bucket {
            self.go_to_bucket(target_bucket);
        }

        // Drop any bits in the current word that are below target. If
        // we just jumped buckets, this masks the freshly loaded word.
        // If we stayed in cursor_bucket, this masks against whatever
        // advance() left of the word.
        self.cursor_tinybitset = self
            .cursor_tinybitset
            .intersect(TinySet::range_greater_or_equal(target));

        // advance() pops the next set bit via trailing_zeros, skips
        // zero words via first_non_empty_bucket, and terminates
        // cleanly when nothing remains.
        self.advance()
    }

    fn has_fast_bitset(&self) -> bool {
        true
    }

    fn fill_bitset_block(
        &mut self,
        base: DocId,
        mask: &mut [TinySet; BLOCK_NUM_TINYBITSETS],
    ) -> DocId {
        let start = base.max(self.doc);
        let end = base.saturating_add(BLOCK_WINDOW).min(self.docs.max_value());
        let shift = base % 64;
        let word_at = |bucket: u32| {
            if bucket * 64 < self.docs.max_value() {
                self.docs.tinyset(bucket).into_u64()
            } else {
                0
            }
        };
        for (i, word) in mask.iter_mut().enumerate() {
            let word_base = base + i as u32 * 64;
            if word_base >= end || word_base + 64 <= start {
                continue;
            }
            let bucket = base / 64 + i as u32;
            let mut bits = word_at(bucket) >> shift;
            if shift != 0 {
                bits |= word_at(bucket + 1) << (64 - shift);
            }
            bits &= u64::MAX << start.saturating_sub(word_base);
            if end - word_base < 64 {
                bits &= (1u64 << (end - word_base)) - 1;
            }
            *word = word.union(TinySet::deserialize(bits.to_le_bytes()));
        }
        if end > self.doc {
            self.seek(end);
        }
        self.doc
    }

    /// Returns the current document
    fn doc(&self) -> DocId {
        self.doc
    }

    /// Returns the number of values set in the underlying bitset.
    fn size_hint(&self) -> u32 {
        self.docs.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use common::BitSet;

    use super::BitSetDocSet;
    use crate::docset::{DocSet, BLOCK_NUM_TINYBITSETS, BLOCK_WINDOW, TERMINATED};
    use crate::tests::generate_nonunique_unsorted;
    use crate::DocId;

    fn create_docbitset(docs: &[DocId], max_doc: DocId) -> BitSetDocSet {
        let mut docset = BitSet::with_max_value(max_doc);
        for &doc in docs {
            docset.insert(doc);
        }
        BitSetDocSet::from(docset)
    }

    #[test]
    fn test_bitset_large() {
        let arr = generate_nonunique_unsorted(100_000, 5_000);
        let mut btreeset: BTreeSet<u32> = BTreeSet::new();
        let mut bitset = BitSet::with_max_value(100_000);
        for el in arr {
            btreeset.insert(el);
            bitset.insert(el);
        }
        for i in 0..100_000 {
            assert_eq!(btreeset.contains(&i), bitset.contains(i));
        }
        assert_eq!(btreeset.len(), bitset.len());
        let mut bitset_docset = BitSetDocSet::from(bitset);
        let mut remaining = true;
        for el in btreeset.into_iter() {
            assert!(remaining);
            assert_eq!(bitset_docset.doc(), el);
            remaining = bitset_docset.advance() != TERMINATED;
        }
        assert!(!remaining);
    }

    #[test]
    fn test_empty() {
        let bitset = BitSet::with_max_value(1000);
        let mut empty = BitSetDocSet::from(bitset);
        assert_eq!(empty.advance(), TERMINATED)
    }

    #[test]
    fn test_seek_terminated() {
        let bitset = BitSet::with_max_value(1000);
        let mut empty = BitSetDocSet::from(bitset);
        assert_eq!(empty.seek(TERMINATED), TERMINATED)
    }

    fn test_go_through_sequential(docs: &[DocId]) {
        let mut docset = create_docbitset(docs, 1_000u32);
        for &doc in docs {
            assert_eq!(doc, docset.doc());
            docset.advance();
        }
        assert_eq!(docset.advance(), TERMINATED);
    }

    #[test]
    fn test_docbitset_sequential() {
        test_go_through_sequential(&[1, 2, 3]);
        test_go_through_sequential(&[1, 2, 3, 4, 5, 63, 64, 65]);
        test_go_through_sequential(&[63, 64, 65]);
        test_go_through_sequential(&[1, 2, 3, 4, 95, 96, 97, 98, 99]);
    }

    #[test]
    fn test_docbitset_seek_same_bucket() {
        // All three bits live in bucket 0 (positions 10, 30, 50), so
        // every seek below stays inside the current cursor word —
        // exercises the same-bucket path of the rewritten `seek`.
        let mut docset = create_docbitset(&[10, 30, 50], 1_000);
        assert_eq!(docset.doc(), 10);
        // target > self.doc, target in same bucket, skips past 10.
        assert_eq!(docset.seek(20), 30);
        // target > self.doc again, same bucket, lands on the next bit.
        assert_eq!(docset.seek(40), 50);
        // target past the last bit in the bucket: mask empties the
        // word, advance() finds no later non-empty bucket → terminate.
        assert_eq!(docset.seek(60), TERMINATED);
        // Idempotent after termination.
        assert_eq!(docset.advance(), TERMINATED);
    }

    #[test]
    fn test_docbitset_seek_to_current() {
        // seek(target) where target == self.doc must return self.doc,
        // not advance past it. After construction, advance() has
        // already popped self.doc's bit out of the cursor word, so a
        // mask-and-advance without the `target <= self.doc` guard
        // would skip the current doc.
        let mut docset = create_docbitset(&[10, 30], 1_000);
        assert_eq!(docset.doc(), 10);
        assert_eq!(docset.seek(10), 10);
        assert_eq!(docset.doc(), 10);
        assert_eq!(docset.advance(), 30);
    }

    #[test]
    fn test_docbitset_skip() {
        {
            let mut docset = create_docbitset(&[1, 5, 6, 7, 5112], 10_000);
            assert_eq!(docset.seek(7), 7);
            assert_eq!(docset.doc(), 7);
            assert_eq!(docset.advance(), 5112);
            assert_eq!(docset.doc(), 5112);
            assert_eq!(docset.advance(), TERMINATED);
        }
        {
            let mut docset = create_docbitset(&[1, 5, 6, 7, 5112], 10_000);
            assert_eq!(docset.seek(3), 5);
            assert_eq!(docset.doc(), 5);
            assert_eq!(docset.advance(), 6);
        }
        {
            let mut docset = create_docbitset(&[5112], 10_000);
            assert_eq!(docset.seek(5112), 5112);
            assert_eq!(docset.doc(), 5112);
            assert_eq!(docset.advance(), TERMINATED);
        }
        {
            let mut docset = create_docbitset(&[5112], 10_000);
            assert_eq!(docset.seek(5113), TERMINATED);
            assert_eq!(docset.advance(), TERMINATED);
        }
        {
            let mut docset = create_docbitset(&[5112], 10_000);
            assert_eq!(docset.seek(5111), 5112);
            assert_eq!(docset.doc(), 5112);
            assert_eq!(docset.advance(), TERMINATED);
        }
        {
            let mut docset = create_docbitset(&[1, 5, 6, 7, 5112, 5500, 6666], 10_000);
            assert_eq!(docset.seek(5112), 5112);
            assert_eq!(docset.doc(), 5112);
            assert_eq!(docset.advance(), 5500);
            assert_eq!(docset.doc(), 5500);
            assert_eq!(docset.advance(), 6666);
            assert_eq!(docset.doc(), 6666);
            assert_eq!(docset.advance(), TERMINATED);
        }
        {
            let mut docset = create_docbitset(&[1, 5, 6, 7, 5112, 5500, 6666], 10_000);
            assert_eq!(docset.seek(5111), 5112);
            assert_eq!(docset.doc(), 5112);
            assert_eq!(docset.advance(), 5500);
            assert_eq!(docset.doc(), 5500);
            assert_eq!(docset.advance(), 6666);
            assert_eq!(docset.doc(), 6666);
            assert_eq!(docset.advance(), TERMINATED);
        }
        {
            let mut docset = create_docbitset(&[1, 5, 6, 7, 5112, 5513, 6666], 10_000);
            assert_eq!(docset.seek(5111), 5112);
            assert_eq!(docset.doc(), 5112);
            assert_eq!(docset.advance(), 5513);
            assert_eq!(docset.doc(), 5513);
            assert_eq!(docset.advance(), 6666);
            assert_eq!(docset.doc(), 6666);
            assert_eq!(docset.advance(), TERMINATED);
        }
    }

    #[test]
    fn test_bitset_is_empty_in_range() {
        let docs = vec![1, 5, 63, 64, 65, 127, 200];
        let mut docset = create_docbitset(&docs, 300);

        assert!(!docset.is_empty_in_range(0, 10)); // doc 1, 5 in range
        assert!(docset.is_empty_in_range(6, 62)); // empty in [6, 62]
        assert!(!docset.is_empty_in_range(63, 63)); // doc 63 matches
        assert!(docset.is_empty_in_range(128, 199)); // empty in [128, 199]
        assert!(!docset.is_empty_in_range(128, 200)); // doc 200 matches
        assert!(docset.is_empty_in_range(201, 500)); // past last doc
        assert!(docset.is_empty_in_range(10, 5)); // start > end
    }
}

#[cfg(all(test, feature = "unstable"))]
mod bench {

    use super::{BitSet, BitSetDocSet};
    use crate::docset::TERMINATED;
    use crate::{test, tests, DocSet};

    #[bench]
    fn bench_bitset_1pct_insert(b: &mut test::Bencher) {
        let els = tests::generate_nonunique_unsorted(1_000_000u32, 10_000);
        b.iter(|| {
            let mut bitset = BitSet::with_max_value(1_000_000);
            for el in els.iter().cloned() {
                bitset.insert(el);
            }
        });
    }

    #[bench]
    fn bench_bitset_1pct_clone(b: &mut test::Bencher) {
        let els = tests::generate_nonunique_unsorted(1_000_000u32, 10_000);
        let mut bitset = BitSet::with_max_value(1_000_000);
        for el in els {
            bitset.insert(el);
        }
        b.iter(|| bitset.clone());
    }

    #[bench]
    fn bench_bitset_1pct_clone_iterate(b: &mut test::Bencher) {
        let els = tests::sample(1_000_000u32, 0.01);
        let mut bitset = BitSet::with_max_value(1_000_000);
        for el in els {
            bitset.insert(el);
        }
        b.iter(|| {
            let mut docset = BitSetDocSet::from(bitset.clone());
            while docset.advance() != TERMINATED {}
        });
    }
}
