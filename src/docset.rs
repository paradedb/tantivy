use std::borrow::{Borrow, BorrowMut};

use common::TinySet;

use crate::fastfield::AliveBitSet;
use crate::DocId;

/// Estimated total cost to consume a [`DocSet`] via `advance()` until [`TERMINATED`],
/// in units of one `TermScorer::advance()` (`1` per matching document).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DriveCost(pub u64);

impl DriveCost {
    /// Zero drive cost.
    pub const ZERO: Self = Self(0);

    /// Saturating addition of drive costs.
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating multiplication of drive cost by `factor`.
    #[inline]
    pub const fn saturating_mul(self, factor: u64) -> Self {
        Self(self.0.saturating_mul(factor))
    }
}

/// Fixed per-call setup cost of a `seek_danger` (or batched probe) invocation,
/// in units of one `TermScorer::advance()`.
///
/// Zero for in-memory docsets (`TermScorer`, `BitSetDocSet`, `RangeDocSet`, `PhraseScorer`);
/// non-zero for lazy SSTable dictionary block decoding or cross-boundary vectorized predicates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProbeSetupCost(pub u64);

impl ProbeSetupCost {
    /// Zero setup cost.
    pub const ZERO: Self = Self(0);

    /// Saturating addition of setup costs.
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
}

/// Marginal per-document cost of testing membership in a [`DocSet`],
/// in units of one `TermScorer::advance()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProbePerDocCost(pub u64);

impl ProbePerDocCost {
    /// Zero per-document probe cost.
    pub const ZERO: Self = Self(0);

    /// Saturating addition of per-document probe costs.
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating multiplication of per-document probe cost by `factor`.
    #[inline]
    pub const fn saturating_mul(self, factor: u64) -> Self {
        Self(self.0.saturating_mul(factor))
    }
}

/// Estimated cost of probing membership of externally supplied documents (`seek_danger`).
///
/// Modeled as `setup + batch_len * per_doc`, separating fixed per-call overhead (amortized across
/// a candidate batch) from marginal per-document work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProbeCost {
    /// Fixed per-call overhead (`0` for in-memory docsets).
    pub setup: ProbeSetupCost,
    /// Marginal cost per probed document.
    pub per_doc: ProbePerDocCost,
}

impl ProbeCost {
    /// Zero probe cost.
    pub const ZERO: Self = Self {
        setup: ProbeSetupCost::ZERO,
        per_doc: ProbePerDocCost::ZERO,
    };

    /// Bitset or contiguous range membership test (`O(1)` in-memory lookup).
    pub const BITSET: Self = Self::per_doc(1);

    /// Default postings seek/probe cost (`TermScorer` block skip + search).
    pub const TERM: Self = Self::per_doc(3);

    /// In-memory fast-field value decode + comparison (`RangeDocSet`).
    pub const FAST_FIELD: Self = Self::per_doc(5);

    /// Binary search / gallop on a sorted fast-field column.
    pub const GALLOP: Self = Self::per_doc(8);

    /// In-memory fast-field decode + hash/ordinal set lookup.
    pub const FAST_FIELD_LOOKUP: Self = Self::per_doc(10);

    /// Lazy SSTable dictionary block decode + lookup:
    /// fixed block read/decompression setup plus ordinal lookup per probed document.
    pub const DICT_BLOCK: Self = Self {
        setup: ProbeSetupCost(64),
        per_doc: ProbePerDocCost(10),
    };

    /// Minimum `setup` above which amortizing over a candidate batch beats buffering overhead
    /// in streaming operators (`Intersection`, `Exclude`).
    pub const BATCH_SETUP_THRESHOLD: ProbeSetupCost = ProbeSetupCost(32);

    /// Creates a `ProbeCost` with zero fixed setup overhead.
    #[inline]
    pub const fn per_doc(per_doc: u64) -> Self {
        Self {
            setup: ProbeSetupCost::ZERO,
            per_doc: ProbePerDocCost(per_doc),
        }
    }

    /// Creates a `ProbeCost` with both fixed per-call setup and marginal per-doc cost.
    #[inline]
    pub const fn new(setup: ProbeSetupCost, per_doc: ProbePerDocCost) -> Self {
        Self { setup, per_doc }
    }

    /// Cost of a single scalar `seek_danger(target)` probe.
    #[inline]
    pub const fn scalar(self) -> u64 {
        self.setup.0.saturating_add(self.per_doc.0)
    }

    /// Total cost of probing `batch_len` candidates in a single batched call.
    #[inline]
    pub const fn for_batch(self, batch_len: u64) -> u64 {
        if batch_len == 0 {
            return 0;
        }
        self.setup
            .0
            .saturating_add(self.per_doc.0.saturating_mul(batch_len))
    }

    /// Whether amortizing fixed per-call setup across a candidate batch beats buffering overhead.
    #[inline]
    pub const fn prefers_batching(self) -> bool {
        self.setup.0 >= Self::BATCH_SETUP_THRESHOLD.0
    }

    /// Combines two sequentially evaluated probes conservatively.
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self {
            setup: self.setup.saturating_add(rhs.setup),
            per_doc: self.per_doc.saturating_add(rhs.per_doc),
        }
    }

    /// Scales marginal per-doc cost by a selectivity fraction in `[0.0, 1.0]` while preserving
    /// fixed per-call setup (which is paid whenever at least one candidate reaches this probe).
    #[inline]
    pub fn scale_per_doc(self, selectivity: f64) -> Self {
        let scaled = (self.per_doc.0 as f64 * selectivity.clamp(0.0, 1.0)).ceil() as u64;
        Self {
            setup: self.setup,
            per_doc: ProbePerDocCost(scaled),
        }
    }
}

/// Estimated execution costs for a [`DocSet`], in units of one `TermScorer::advance()`.
///
/// Separates full-scan drive work ([`DriveCost`]) from point/batched membership probing
/// ([`ProbeCost`]), which are largely independent: for example, a dense bitset has high drive
/// cost with cheap `O(1)` probes, whereas a rare phrase or vectorized predicate has low drive
/// cost or an unbounded `size_hint()` with expensive probes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DocSetCost {
    /// Total cost to consume the entire `DocSet` via `advance()` until [`TERMINATED`].
    pub drive: DriveCost,
    /// Cost to test membership of externally supplied documents (`seek_danger`).
    pub probe: ProbeCost,
}

impl DocSetCost {
    /// Zero drive and probe cost.
    pub const ZERO: Self = Self {
        drive: DriveCost::ZERO,
        probe: ProbeCost::ZERO,
    };

    /// Creates a new `DocSetCost` from drive and probe components.
    #[inline]
    pub const fn new(drive: DriveCost, probe: ProbeCost) -> Self {
        Self { drive, probe }
    }
}

/// Sentinel value returned when a [`DocSet`] has been entirely consumed.
///
/// This is not `u32::MAX` as one would have expected, due to the lack of SSE2 instructions
/// to compare `[u32; 4]`.
pub const TERMINATED: DocId = i32::MAX as u32;

/// The collect_block method on `SegmentCollector` uses a buffer of this size.
/// Passed results to `collect_block` will not exceed this size and will be
/// exactly this size as long as we can fill the buffer.
pub const COLLECT_BLOCK_BUFFER_LEN: usize = 64;

/// Number of `TinySet` (64-bit) buckets in a block used by [`DocSet::fill_bitset_block`].
pub const BLOCK_NUM_TINYBITSETS: usize = 16;

/// Number of doc IDs covered by one block: `BLOCK_NUM_TINYBITSETS * 64 = 1024`.
pub const BLOCK_WINDOW: u32 = BLOCK_NUM_TINYBITSETS as u32 * 64;

/// Initializes `mask[..len.div_ceil(64)]` with bits `0..len` set (all candidates active).
#[inline]
pub fn fill_all_candidates_mask(len: usize, mask: &mut [TinySet]) {
    let words = len.div_ceil(64);
    debug_assert!(mask.len() >= words);
    let full_words = len / 64;
    let rem = len % 64;
    mask[..full_words].fill(TinySet::full());
    if rem > 0 {
        mask[full_words] = TinySet::range_lower(rem as u32);
    }
}

/// Represents an iterable set of sorted doc ids.
pub trait DocSet: Send {
    /// Goes to the next element.
    ///
    /// The DocId of the next element is returned.
    /// In other words we should always have :
    /// ```compile_fail
    /// let doc = docset.advance();
    /// assert_eq!(doc, docset.doc());
    /// ```
    ///
    /// If we reached the end of the `DocSet`, [`TERMINATED`] should be returned.
    ///
    /// Calling `.advance()` on a terminated `DocSet` should be supported, and [`TERMINATED`] should
    /// be returned.
    fn advance(&mut self) -> DocId;

    /// Advances the `DocSet` forward until reaching the target, or going to the
    /// lowest [`DocId`] greater than the target.
    ///
    /// If the end of the `DocSet` is reached, [`TERMINATED`] is returned.
    ///
    /// Calling `.seek(target)` on a terminated `DocSet` is legal. Implementation
    /// of `DocSet` should support it.
    ///
    /// Calling `seek(TERMINATED)` is also legal and is the normal way to consume a `DocSet`.
    ///
    /// `target` has to be larger or equal to `.doc()` when calling `seek`.
    fn seek(&mut self, target: DocId) -> DocId {
        let mut doc = self.doc();
        debug_assert!(doc <= target);
        while doc < target {
            doc = self.advance();
        }
        doc
    }

    /// !!!Dragons ahead!!!
    /// In spirit, this is an approximate and dangerous version of `seek`.
    ///
    /// It can leave the DocSet in an `invalid` state and might return a
    /// lower bound of what the result of Seek would have been.
    ///
    ///
    /// More accurately it returns either:
    /// - Found if the target is in the docset. In that case, the DocSet is left in a valid state.
    /// - SeekLowerBound(seek_lower_bound) if the target is not in the docset. In that case, The
    ///   DocSet can be the left in a invalid state. The DocSet should then only receives call to
    ///   `seek_danger(..)` until it returns `Found`, and get back to a valid state.
    ///
    /// `seek_lower_bound` can be any `DocId` (in the docset or not) as long as it is in
    /// `(target .. seek_result] U {TERMINATED}` where `seek_result` is the first document in the
    /// docset greater than to `target`.
    ///
    /// `seek_danger` may return `SeekLowerBound(TERMINATED)`.
    ///
    /// Calling `seek_danger` with TERMINATED as a target is allowed,
    /// and should always return NewTarget(TERMINATED) or anything larger as TERMINATED is NOT in
    /// the DocSet.
    ///
    /// DocSets that already have an efficient `seek` method don't need to implement
    /// `seek_danger`.
    ///
    /// Consecutive calls to seek_danger are guaranteed to have strictly increasing `target`
    /// values.
    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        if target >= TERMINATED {
            debug_assert!(target == TERMINATED);
            // No need to advance.
            return SeekDangerResult::SeekLowerBound(target);
        }

        // The default implementation does not include any
        // `danger zone` behavior.
        //
        // It does not leave the scorer in an invalid state.
        // For this reason, we can safely call `self.doc()`.
        let mut doc = self.doc();
        if doc < target {
            doc = self.seek(target);
        }
        if doc == target {
            SeekDangerResult::Found
        } else {
            SeekDangerResult::SeekLowerBound(doc)
        }
    }

    /// Intersects `mask` in place with membership of a strictly increasing slice of candidate
    /// document IDs, returning a lower bound on the next matching document strictly after
    /// `docs.last()` (capped at [`TERMINATED`], or `0` if `docs` is empty).
    ///
    /// Complements [`DocSet::fill_bitset_block`] (which evaluates contiguous 1,024-doc windows):
    /// `seek_danger_batch` probes sparse externally supplied candidates without evaluating
    /// unprobed neighbors, amortizing [`ProbeSetupCost`] across the batch. Using an in-place
    /// AND mask lets conjunctions reuse a single `mask` without scratch buffers, while returning
    /// the skip horizon preserves `SeekLowerBound` skipping across batches.
    ///
    /// `mask` must have length `>= docs.len().div_ceil(64)` with no bits set at positions
    /// `>= docs.len()` (see [`fill_all_candidates_mask`]). Only candidates `i` whose bit is set
    /// in `mask` need to be tested; implementations clear bit `(i % 64)` of `mask[i / 64]` for
    /// any active candidate not present in this `DocSet`, and leave already-cleared bits cleared.
    ///
    /// Follows the same monotonicity and danger-zone contract as [`DocSet::seek_danger`]: `docs`
    /// must be strictly increasing (`docs[0] < docs[1] < ... < TERMINATED`) and monotonically
    /// increasing relative to prior `seek_danger` / `seek_danger_batch` calls, and the `DocSet`
    /// may be left in an invalid state afterwards.
    fn seek_danger_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) -> DocId {
        if docs.is_empty() {
            return 0;
        }
        let words = docs.len().div_ceil(64);
        debug_assert!(mask.len() >= words);
        debug_assert!(
            docs.windows(2).all(|w| w[0] < w[1]),
            "seek_danger_batch requires strictly increasing docs"
        );
        debug_assert!(
            *docs.last().unwrap() < TERMINATED,
            "seek_danger_batch docs must be < TERMINATED"
        );
        debug_assert!(
            docs.len() % 64 == 0
                || mask[words - 1]
                    .intersect(TinySet::range_greater_or_equal((docs.len() % 64) as u32))
                    .is_empty(),
            "seek_danger_batch mask must not set bits >= docs.len()"
        );
        let last = *docs.last().unwrap();
        let mut lower_bound = 0;
        for (word_idx, word) in mask[..words].iter_mut().enumerate() {
            if word.is_empty() {
                continue;
            }
            let base = word_idx * 64;
            let word_last = docs[(base + 63).min(docs.len() - 1)];
            if word_last < lower_bound {
                *word = TinySet::EMPTY;
                continue;
            }
            let mut keep = TinySet::EMPTY;
            for bit in *word {
                let doc = docs[base + bit as usize];
                if doc < lower_bound {
                    continue;
                }
                match self.seek_danger(doc) {
                    SeekDangerResult::Found => {
                        keep.insert_mut(bit);
                    }
                    SeekDangerResult::SeekLowerBound(bound) => lower_bound = bound,
                }
            }
            *word = keep;
        }
        lower_bound.max(last.saturating_add(1)).min(TERMINATED)
    }

    /// Fills a given mutable buffer with the next doc ids from the
    /// `DocSet`
    ///
    /// If that many `DocId`s are available, the method should
    /// fill the entire buffer and return the length of the buffer.
    ///
    /// If we reach the end of the `DocSet` before filling
    /// it entirely, then the buffer is filled up to this point, and
    /// return value is the number of elements that were filled.
    ///
    /// # Warning
    ///
    /// This method is only here for specific high-performance
    /// use case where batching. The normal way to
    /// go through the `DocId`'s is to call `.advance()`.
    fn fill_buffer(&mut self, buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN]) -> usize {
        if self.doc() == TERMINATED {
            return 0;
        }
        for (i, buffer_val) in buffer.iter_mut().enumerate() {
            *buffer_val = self.doc();
            if self.advance() == TERMINATED {
                return i + 1;
            }
        }
        buffer.len()
    }

    /// Returns the current document
    /// Right after creating a new `DocSet`, the docset points to the first document.
    ///
    /// If the `DocSet` is empty, `.doc()` should return [`TERMINATED`].
    fn doc(&self) -> DocId;

    /// Returns a best-effort hint of the
    /// length of the docset.
    fn size_hint(&self) -> u32;

    /// Returns best-effort cost estimates for driving (`advance()`) and probing (`seek_danger()`)
    /// this `DocSet`, in units of one `TermScorer::advance()`.
    ///
    /// By default `drive` is `size_hint()` and `probe` is [`ProbeCost::TERM`].
    fn cost(&self) -> DocSetCost {
        DocSetCost::new(DriveCost(self.size_hint() as u64), ProbeCost::TERM)
    }

    /// Fills a bitmask representing which documents in `[min_doc, min_doc + BLOCK_WINDOW)` are
    /// present in this docset.
    ///
    /// The window is divided into `BLOCK_NUM_TINYBITSETS` buckets of 64 docs each.
    /// Returns the next doc `>= min_doc + BLOCK_WINDOW`, or `TERMINATED` if exhausted.
    fn fill_bitset_block(
        &mut self,
        min_doc: DocId,
        mask: &mut [TinySet; BLOCK_NUM_TINYBITSETS],
    ) -> DocId {
        self.seek(min_doc);
        let horizon = min_doc + BLOCK_WINDOW;
        loop {
            let doc = self.doc();
            if doc >= horizon {
                return doc;
            }
            let delta = doc - min_doc;
            mask[(delta / 64) as usize].insert_mut(delta % 64);
            if self.advance() == TERMINATED {
                return TERMINATED;
            }
        }
    }

    /// Returns `true` if this `DocSet` is guaranteed to contain no documents in the inclusive
    /// range `[start, end]`.
    ///
    /// Returning `false` is always conservative and safe (callers will still test individual
    /// candidates). Like [`Self::seek_danger`], this may leave the `DocSet` in the danger zone
    /// and expects non-decreasing `start` values across consecutive calls.
    fn is_empty_in_range(&mut self, start: DocId, end: DocId) -> bool {
        if start > end {
            return true;
        }
        match self.seek_danger(start) {
            SeekDangerResult::Found => false,
            SeekDangerResult::SeekLowerBound(bound) => bound > end,
        }
    }

    /// Returns the number documents matching.
    /// Calling this method consumes the `DocSet`.
    fn count(&mut self, alive_bitset: &AliveBitSet) -> u32 {
        let mut count = 0u32;
        let mut doc = self.doc();
        while doc != TERMINATED {
            if alive_bitset.is_alive(doc) {
                count += 1u32;
            }
            doc = self.advance();
        }
        count
    }

    /// Returns the count of documents, deleted or not.
    /// Calling this method consumes the `DocSet`.
    ///
    /// Of course, the result is an upper bound of the result
    /// given by `count()`.
    fn count_including_deleted(&mut self) -> u32 {
        let mut count = 0u32;
        let mut doc = self.doc();
        while doc != TERMINATED {
            count += 1u32;
            doc = self.advance();
        }
        count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeekDangerResult {
    /// The target was found in the DocSet.
    Found,
    /// The target was not found in the DocSet.
    /// We return a range in which the value could be.
    /// The given target can be any DocId, that is <= than the first document
    /// in the docset after the target.
    SeekLowerBound(DocId),
}

impl DocSet for &mut dyn DocSet {
    fn advance(&mut self) -> u32 {
        (**self).advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        (**self).seek(target)
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        (**self).seek_danger(target)
    }

    fn seek_danger_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) -> DocId {
        (**self).seek_danger_batch(docs, mask)
    }

    fn fill_buffer(&mut self, buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN]) -> usize {
        (**self).fill_buffer(buffer)
    }

    fn fill_bitset_block(
        &mut self,
        min_doc: DocId,
        mask: &mut [TinySet; BLOCK_NUM_TINYBITSETS],
    ) -> DocId {
        (**self).fill_bitset_block(min_doc, mask)
    }

    fn is_empty_in_range(&mut self, start: DocId, end: DocId) -> bool {
        (**self).is_empty_in_range(start, end)
    }

    fn doc(&self) -> u32 {
        (**self).doc()
    }

    fn size_hint(&self) -> u32 {
        (**self).size_hint()
    }

    fn cost(&self) -> DocSetCost {
        (**self).cost()
    }

    fn count(&mut self, alive_bitset: &AliveBitSet) -> u32 {
        (**self).count(alive_bitset)
    }

    fn count_including_deleted(&mut self) -> u32 {
        (**self).count_including_deleted()
    }
}

impl<TDocSet: DocSet + ?Sized> DocSet for Box<TDocSet> {
    fn advance(&mut self) -> DocId {
        let unboxed: &mut TDocSet = self.borrow_mut();
        unboxed.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let unboxed: &mut TDocSet = self.borrow_mut();
        unboxed.seek(target)
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        let unboxed: &mut TDocSet = self.borrow_mut();
        unboxed.seek_danger(target)
    }

    fn seek_danger_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) -> DocId {
        let unboxed: &mut TDocSet = self.borrow_mut();
        unboxed.seek_danger_batch(docs, mask)
    }

    fn fill_buffer(&mut self, buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN]) -> usize {
        let unboxed: &mut TDocSet = self.borrow_mut();
        unboxed.fill_buffer(buffer)
    }

    fn fill_bitset_block(
        &mut self,
        min_doc: DocId,
        mask: &mut [TinySet; BLOCK_NUM_TINYBITSETS],
    ) -> DocId {
        let unboxed: &mut TDocSet = self.borrow_mut();
        unboxed.fill_bitset_block(min_doc, mask)
    }

    fn is_empty_in_range(&mut self, start: DocId, end: DocId) -> bool {
        let unboxed: &mut TDocSet = self.borrow_mut();
        unboxed.is_empty_in_range(start, end)
    }

    fn doc(&self) -> DocId {
        let unboxed: &TDocSet = self.borrow();
        unboxed.doc()
    }

    fn size_hint(&self) -> u32 {
        let unboxed: &TDocSet = self.borrow();
        unboxed.size_hint()
    }

    fn cost(&self) -> DocSetCost {
        let unboxed: &TDocSet = self.borrow();
        unboxed.cost()
    }

    fn count(&mut self, alive_bitset: &AliveBitSet) -> u32 {
        let unboxed: &mut TDocSet = self.borrow_mut();
        unboxed.count(alive_bitset)
    }

    fn count_including_deleted(&mut self) -> u32 {
        let unboxed: &mut TDocSet = self.borrow_mut();
        unboxed.count_including_deleted()
    }
}
