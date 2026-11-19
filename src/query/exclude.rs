use common::TinySet;

use crate::docset::{
    fill_all_candidates_mask, DocSet, DocSetCost, DriveCost, ProbeCost, SeekDangerResult,
    BLOCK_NUM_TINYBITSETS, BLOCK_WINDOW, TERMINATED,
};
use crate::query::Scorer;
use crate::{DocId, Score};

/// An exclusion set is a set of documents
/// that should be excluded from a given DocSet.
///
/// It can be a single DocSet, or a Vec of DocSets.
pub trait ExclusionSet: Send {
    /// Returns `true` if the given `doc` is in the exclusion set.
    fn contains(&mut self, doc: DocId) -> bool;

    /// Clears bit `i` of `mask` for every active candidate `docs[i]` present in this exclusion set.
    ///
    /// `docs` must be strictly increasing (`< TERMINATED`) and monotonic across calls, and
    /// `mask` must have length `>= docs.len().div_ceil(64)` with no bits set at positions
    /// `>= docs.len()`.
    fn remove_excluded_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) {
        let words = docs.len().div_ceil(64);
        debug_assert!(mask.len() >= words);
        for (word_idx, word) in mask[..words].iter_mut().enumerate() {
            if word.is_empty() {
                continue;
            }
            let base = word_idx * 64;
            let mut keep = TinySet::EMPTY;
            for bit in *word {
                if !self.contains(docs[base + bit as usize]) {
                    keep.insert_mut(bit);
                }
            }
            *word = keep;
        }
    }

    /// Estimated cost to probe membership in this exclusion set.
    fn probe_cost(&self) -> ProbeCost {
        ProbeCost::TERM
    }
}

/// A `DocSet` used as an exclusion set.
///
/// It remembers the lowest doc the `DocSet` can still contain, starting from the doc the `DocSet`
/// is on, and answers lower docs without calling `seek_danger`. `seek_danger` targets then never
/// go below the `DocSet`'s position, which `PhraseScorer` asserts and `RangeDocSet` relies on.
pub struct ExcludedDocSet<TDocSet> {
    docset: TDocSet,
    lower_bound: DocId,
}

impl<TDocSet: DocSet> ExcludedDocSet<TDocSet> {
    /// Wraps a `DocSet` that has not been probed with `seek_danger` yet.
    pub fn new(docset: TDocSet) -> Self {
        let lower_bound = docset.doc();
        ExcludedDocSet {
            docset,
            lower_bound,
        }
    }
}

impl<TDocSet: DocSet> ExclusionSet for ExcludedDocSet<TDocSet> {
    #[inline]
    fn contains(&mut self, doc: DocId) -> bool {
        if doc < self.lower_bound {
            return false;
        }
        match self.docset.seek_danger(doc) {
            SeekDangerResult::Found => true,
            SeekDangerResult::SeekLowerBound(lower_bound) => {
                self.lower_bound = lower_bound;
                false
            }
        }
    }

    fn remove_excluded_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) {
        let words = docs.len().div_ceil(64);
        if words == 0 {
            return;
        }
        debug_assert!(mask.len() >= words);
        let mut stack_buf = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
        let mut heap_buf = Vec::new();
        let excl: &mut [TinySet] = if words <= stack_buf.len() {
            &mut stack_buf[..words]
        } else {
            heap_buf.resize(words, TinySet::EMPTY);
            &mut heap_buf
        };
        let mut any = false;
        for (word_idx, &word) in mask[..words].iter().enumerate() {
            if word.is_empty() {
                continue;
            }
            let base = word_idx * 64;
            let word_last = docs[(base + 63).min(docs.len() - 1)];
            if word_last < self.lower_bound {
                continue;
            }
            if docs[base] >= self.lower_bound {
                excl[word_idx] = word;
                any = true;
            } else {
                let mut active = TinySet::EMPTY;
                for bit in word {
                    if docs[base + bit as usize] >= self.lower_bound {
                        active.insert_mut(bit);
                    }
                }
                excl[word_idx] = active;
                any |= !active.is_empty();
            }
        }
        if !any {
            return;
        }
        let next_bound = self.docset.seek_danger_batch(docs, excl);
        self.lower_bound = self.lower_bound.max(next_bound);
        for (m, &e) in mask[..words].iter_mut().zip(excl.iter()) {
            let keep = m.into_u64() & !e.into_u64();
            *m = TinySet::deserialize(keep.to_le_bytes());
        }
    }

    fn probe_cost(&self) -> ProbeCost {
        self.docset.cost().probe
    }
}

impl<TDocSet: DocSet> ExclusionSet for Vec<ExcludedDocSet<TDocSet>> {
    #[inline]
    fn contains(&mut self, doc: DocId) -> bool {
        self.iter_mut().any(|docset| docset.contains(doc))
    }

    fn remove_excluded_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) {
        let words = docs.len().div_ceil(64);
        for docset in self {
            if mask[..words].iter().all(|w| w.is_empty()) {
                break;
            }
            docset.remove_excluded_batch(docs, mask);
        }
    }

    fn probe_cost(&self) -> ProbeCost {
        let mut probe = ProbeCost::ZERO;
        for docset in self {
            probe = probe.saturating_add(docset.probe_cost());
        }
        probe
    }
}

/// Filters a given `DocSet` by removing the docs from an exclusion set.
///
/// The excluding docsets have no impact on scoring.
pub struct Exclude<TDocSet, TExclusionSet> {
    underlying_docset: TDocSet,
    exclusion_set: TExclusionSet,
}

impl<TDocSet, TExclusionSet> Exclude<TDocSet, TExclusionSet>
where
    TDocSet: DocSet,
    TExclusionSet: ExclusionSet,
{
    /// Creates a new `ExcludeScorer`
    pub fn new(
        mut underlying_docset: TDocSet,
        mut exclusion_set: TExclusionSet,
    ) -> Exclude<TDocSet, TExclusionSet> {
        while underlying_docset.doc() != TERMINATED {
            let target = underlying_docset.doc();
            if !exclusion_set.contains(target) {
                break;
            }
            underlying_docset.advance();
        }
        Exclude {
            underlying_docset,
            exclusion_set,
        }
    }
}

impl<TDocSet, TExclusionSet> DocSet for Exclude<TDocSet, TExclusionSet>
where
    TDocSet: DocSet,
    TExclusionSet: ExclusionSet,
{
    fn advance(&mut self) -> DocId {
        loop {
            let candidate = self.underlying_docset.advance();
            if candidate == TERMINATED {
                return TERMINATED;
            }
            if !self.exclusion_set.contains(candidate) {
                return candidate;
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let candidate = self.underlying_docset.seek(target);
        if candidate == TERMINATED {
            return TERMINATED;
        }
        if !self.exclusion_set.contains(candidate) {
            return candidate;
        }
        self.advance()
    }

    /// One lookup in the exclusion set settles the target. `seek` would walk the underlying
    /// docset through every excluded doc, and a filter under block-WAND is probed once per
    /// candidate.
    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        if target >= TERMINATED {
            debug_assert!(target == TERMINATED);
            return SeekDangerResult::SeekLowerBound(target);
        }
        match self.underlying_docset.seek_danger(target) {
            SeekDangerResult::Found if self.exclusion_set.contains(target) => {
                // The next doc we can accept is strictly after the excluded target.
                SeekDangerResult::SeekLowerBound(target + 1)
            }
            // Excluding docs never adds any, so the underlying lower bound holds for us too.
            result => result,
        }
    }

    fn seek_danger_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) -> DocId {
        let next_bound = self.underlying_docset.seek_danger_batch(docs, mask);
        let words = docs.len().div_ceil(64);
        if !mask[..words].iter().all(|w| w.is_empty()) {
            self.exclusion_set.remove_excluded_batch(docs, mask);
        }
        next_bound
    }

    /// With no underlying docs in the range there is nothing to exclude. With some, the answer
    /// needs the walk this type avoids, so it stays unknown.
    fn is_empty_in_range(&mut self, start: DocId, end: DocId) -> bool {
        self.underlying_docset.is_empty_in_range(start, end)
    }

    fn doc(&self) -> DocId {
        self.underlying_docset.doc()
    }

    /// `.size_hint()` directly returns the size
    /// of the underlying docset without taking in account
    /// the fact that docs might be deleted.
    fn size_hint(&self) -> u32 {
        self.underlying_docset.size_hint()
    }

    fn cost(&self) -> DocSetCost {
        let underlying = self.underlying_docset.cost();
        let excl_probe = self.exclusion_set.probe_cost();
        let probe_work = DriveCost(
            u64::from(self.underlying_docset.size_hint()).saturating_mul(excl_probe.scalar()),
        );
        DocSetCost::new(
            underlying.drive.saturating_add(probe_work),
            underlying.probe.saturating_add(excl_probe),
        )
    }
}

impl<TScorer, TExclusionSet> Scorer for Exclude<TScorer, TExclusionSet>
where
    TScorer: Scorer,
    TExclusionSet: ExclusionSet + 'static,
{
    #[inline]
    fn score(&mut self) -> Score {
        self.underlying_docset.score()
    }

    #[inline]
    fn constant_score(&self) -> Option<Score> {
        self.underlying_docset.constant_score()
    }
}

/// Progressive candidate batch size for streaming batched filters (`64 -> 256 -> 1024`),
/// bounding `LIMIT k` overshoot on the first refill while amortizing fixed setup on full scans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BatchSizeRamp {
    Small,
    Medium,
    Full,
}

impl BatchSizeRamp {
    #[inline]
    pub(crate) const fn limit(self) -> usize {
        match self {
            Self::Small => 64,
            Self::Medium => 256,
            Self::Full => BLOCK_WINDOW as usize,
        }
    }

    #[inline]
    pub(crate) fn promote(&mut self) {
        *self = match *self {
            Self::Small => Self::Medium,
            Self::Medium | Self::Full => Self::Full,
        };
    }
}

/// Buffered surviving candidates covering `[current_doc, end)` for progressive batch scorers.
#[derive(Default)]
pub(crate) struct CandidateBuffer {
    pub(crate) docs: Vec<DocId>,
    pub(crate) scores: Vec<Score>,
    pub(crate) idx: usize,
    /// Exclusive upper bound of the doc ID range whose matches are materialized in `docs[idx..]`.
    /// `0` when no buffered range is active (unpositioned or invalidated after probing past
    /// `end`).
    pub(crate) end: DocId,
}

impl CandidateBuffer {
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.docs.clear();
        self.scores.clear();
        self.idx = 0;
        self.end = 0;
    }

    #[inline]
    pub(crate) fn remaining(&self) -> &[DocId] {
        &self.docs[self.idx..]
    }

    /// Probes `target` inside `[current_doc, self.end)`. Returns `None` if `target >= self.end`.
    #[inline]
    pub(crate) fn seek_danger_within(
        &mut self,
        target: DocId,
    ) -> Option<(SeekDangerResult, Option<Score>)> {
        if target >= self.end {
            return None;
        }
        let rem = &self.docs[self.idx..];
        let rel = rem.partition_point(|&d| d < target);
        let idx = self.idx + rel;
        if idx < self.docs.len() && self.docs[idx] == target {
            self.idx = idx;
            let score = self.scores.get(idx).copied();
            return Some((SeekDangerResult::Found, score));
        }
        let next = if idx < self.docs.len() {
            self.docs[idx]
        } else {
            self.end
        };
        Some((SeekDangerResult::SeekLowerBound(next), None))
    }

    /// Filters `docs` / `mask` against buffered matches `< self.end` and delegates `>= self.end`
    /// to `probe_suffix`.
    pub(crate) fn seek_danger_batch_with_fallback(
        &mut self,
        docs: &[DocId],
        mask: &mut [TinySet],
        mut probe_suffix: impl FnMut(&[DocId], &mut [TinySet]) -> DocId,
    ) -> DocId {
        if docs.is_empty() {
            return 0;
        }
        let split = docs.partition_point(|&d| d < self.end);
        if split == 0 {
            self.clear();
            return probe_suffix(docs, mask);
        }
        let rem = &self.docs[self.idx..];
        if split == docs.len() {
            let mut s_idx = 0usize;
            for (word_idx, word) in mask[..docs.len().div_ceil(64)].iter_mut().enumerate() {
                if word.is_empty() {
                    continue;
                }
                let base = word_idx * 64;
                let mut keep = 0u64;
                for bit in *word {
                    let doc = docs[base + bit as usize];
                    while s_idx < rem.len() && rem[s_idx] < doc {
                        s_idx += 1;
                    }
                    if s_idx < rem.len() && rem[s_idx] == doc {
                        keep |= 1u64 << bit;
                        s_idx += 1;
                    }
                }
                *word = TinySet::deserialize(keep.to_le_bytes());
            }
            let last = docs[docs.len() - 1];
            let pos = rem.partition_point(|&d| d <= last);
            return if pos < rem.len() { rem[pos] } else { self.end };
        }

        // Straddling batch: match `docs[..split]` against `rem` and delegate `docs[split..]`.
        let prefix_words = split.div_ceil(64);
        let mut s_idx = 0usize;
        for (word_idx, word) in mask[..prefix_words].iter_mut().enumerate() {
            if word.is_empty() {
                continue;
            }
            let base = word_idx * 64;
            let mut keep = 0u64;
            for bit in *word {
                let i = base + bit as usize;
                if i >= split {
                    keep |= 1u64 << bit;
                    continue;
                }
                let doc = docs[i];
                while s_idx < rem.len() && rem[s_idx] < doc {
                    s_idx += 1;
                }
                if s_idx < rem.len() && rem[s_idx] == doc {
                    keep |= 1u64 << bit;
                    s_idx += 1;
                }
            }
            *word = TinySet::deserialize(keep.to_le_bytes());
        }
        let suffix = &docs[split..];
        let mut sub_mask = vec![TinySet::EMPTY; suffix.len().div_ceil(64)];
        for j in 0..suffix.len() {
            let src_idx = split + j;
            if mask[src_idx / 64].contains((src_idx % 64) as u32) {
                sub_mask[j / 64].insert_mut((j % 64) as u32);
            }
        }
        let bound = probe_suffix(suffix, &mut sub_mask);
        for j in 0..suffix.len() {
            if !sub_mask[j / 64].contains((j % 64) as u32) {
                let src_idx = split + j;
                mask[src_idx / 64].remove_mut((src_idx % 64) as u32);
            }
        }
        self.clear();
        bound
    }

    /// Resolves `is_empty_in_range(start, end)` against buffered matches `< self.end`.
    /// Returns `Some(is_empty)` if resolved, or `None` if the caller must also check
    /// `[start.max(self.end), end]`.
    #[inline]
    pub(crate) fn is_empty_in_range_within(&self, start: DocId, end: DocId) -> Option<bool> {
        if start > end {
            return Some(true);
        }
        if start >= self.end {
            return None;
        }
        let rem = self.remaining();
        let pos = rem.partition_point(|&d| d < start);
        if pos < rem.len() && rem[pos] <= end {
            return Some(false);
        }
        if end < self.end {
            return Some(true);
        }
        None
    }
}

/// Batched counterpart to [`Exclude`] used when `exclusion_set` has high fixed per-call probe
/// setup (`ProbeCost::prefers_batching`). Buffers candidates from `underlying` and probes
/// `exclusion_set` via [`ExclusionSet::remove_excluded_batch`].
pub(crate) struct BatchExcludeScorer<TScorer, TExclusionSet> {
    underlying: TScorer,
    exclusion_set: TExclusionSet,
    underlying_const_score: Option<Score>,
    ramp: BatchSizeRamp,
    buf: CandidateBuffer,
    current: (DocId, Score),
}

impl<TScorer, TExclusionSet> BatchExcludeScorer<TScorer, TExclusionSet>
where
    TScorer: Scorer,
    TExclusionSet: ExclusionSet,
{
    pub(crate) fn new(underlying: TScorer, exclusion_set: TExclusionSet) -> Self {
        let underlying_const_score = underlying.constant_score();
        let mut scorer = Self {
            underlying,
            exclusion_set,
            underlying_const_score,
            ramp: BatchSizeRamp::Small,
            buf: CandidateBuffer::default(),
            current: (TERMINATED, underlying_const_score.unwrap_or(0.0)),
        };
        scorer.refill();
        scorer
    }

    fn refill(&mut self) {
        self.buf.clear();
        let has_var_score = self.underlying_const_score.is_none();
        let mut mask = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];

        loop {
            let mut doc = self.underlying.doc();
            if doc == TERMINATED {
                self.buf.end = TERMINATED;
                self.current.0 = TERMINATED;
                return;
            }
            let target_len = self.ramp.limit();
            self.ramp.promote();
            self.buf.docs.clear();
            if has_var_score {
                self.buf.scores.clear();
                while doc != TERMINATED && self.buf.docs.len() < target_len {
                    self.buf.docs.push(doc);
                    self.buf.scores.push(self.underlying.score());
                    doc = self.underlying.advance();
                }
            } else {
                while doc != TERMINATED && self.buf.docs.len() < target_len {
                    self.buf.docs.push(doc);
                    doc = self.underlying.advance();
                }
            }
            let n = self.buf.docs.len();
            let words = n.div_ceil(64);
            fill_all_candidates_mask(n, &mut mask);
            self.exclusion_set
                .remove_excluded_batch(&self.buf.docs, &mut mask[..words]);

            let mut write = 0usize;
            for (word_idx, &word) in mask[..words].iter().enumerate() {
                let base = word_idx * 64;
                for bit in word {
                    let i = base + bit as usize;
                    self.buf.docs[write] = self.buf.docs[i];
                    if has_var_score {
                        self.buf.scores[write] = self.buf.scores[i];
                    }
                    write += 1;
                }
            }
            self.buf.docs.truncate(write);
            if has_var_score {
                self.buf.scores.truncate(write);
            }
            if write > 0 {
                self.buf.end = self.underlying.doc();
                self.current = (
                    self.buf.docs[0],
                    self.underlying_const_score
                        .unwrap_or_else(|| self.buf.scores[0]),
                );
                return;
            }
        }
    }
}

impl<TScorer, TExclusionSet> DocSet for BatchExcludeScorer<TScorer, TExclusionSet>
where
    TScorer: Scorer,
    TExclusionSet: ExclusionSet,
{
    #[inline]
    fn advance(&mut self) -> DocId {
        if self.current.0 == TERMINATED {
            return TERMINATED;
        }
        self.buf.idx += 1;
        if self.buf.idx < self.buf.docs.len() {
            self.current = (
                self.buf.docs[self.buf.idx],
                self.underlying_const_score
                    .unwrap_or_else(|| self.buf.scores[self.buf.idx]),
            );
            return self.current.0;
        }
        self.refill();
        self.current.0
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if target <= self.current.0 {
            return self.current.0;
        }
        if target >= TERMINATED {
            self.buf.clear();
            self.buf.end = TERMINATED;
            self.underlying.seek(TERMINATED);
            self.current.0 = TERMINATED;
            return TERMINATED;
        }
        if target < self.buf.end {
            if let Some(&last_buf) = self.buf.docs.last() {
                if target <= last_buf {
                    let rel = self.buf.remaining().partition_point(|&d| d < target);
                    self.buf.idx += rel;
                    self.current = (
                        self.buf.docs[self.buf.idx],
                        self.underlying_const_score
                            .unwrap_or_else(|| self.buf.scores[self.buf.idx]),
                    );
                    return self.current.0;
                }
            }
        }
        if self.underlying.doc() < target {
            self.underlying.seek(target);
        }
        self.refill();
        self.current.0
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        if target >= TERMINATED || self.current.0 == TERMINATED {
            return SeekDangerResult::SeekLowerBound(TERMINATED);
        }
        if let Some((result, score)) = self.buf.seek_danger_within(target) {
            if result == SeekDangerResult::Found {
                self.current = (
                    target,
                    self.underlying_const_score
                        .unwrap_or_else(|| score.unwrap_or(0.0)),
                );
            }
            return result;
        }
        self.buf.clear();
        match self.underlying.seek_danger(target) {
            SeekDangerResult::Found if self.exclusion_set.contains(target) => {
                SeekDangerResult::SeekLowerBound(target + 1)
            }
            SeekDangerResult::Found => {
                let score = self
                    .underlying_const_score
                    .unwrap_or_else(|| self.underlying.score());
                self.current = (target, score);
                self.underlying.advance();
                self.buf.docs.push(target);
                if self.underlying_const_score.is_none() {
                    self.buf.scores.push(score);
                }
                self.buf.idx = 0;
                self.buf.end = self.underlying.doc();
                SeekDangerResult::Found
            }
            SeekDangerResult::SeekLowerBound(lb) => {
                if lb == TERMINATED {
                    self.current.0 = TERMINATED;
                }
                SeekDangerResult::SeekLowerBound(lb)
            }
        }
    }

    fn seek_danger_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) -> DocId {
        if docs.is_empty() {
            return 0;
        }
        if self.current.0 == TERMINATED {
            mask[..docs.len().div_ceil(64)].fill(TinySet::EMPTY);
            return TERMINATED;
        }
        let underlying = &mut self.underlying;
        let exclusion_set = &mut self.exclusion_set;
        self.buf
            .seek_danger_batch_with_fallback(docs, mask, |suffix, sub_mask| {
                let next_bound = underlying.seek_danger_batch(suffix, sub_mask);
                let words = suffix.len().div_ceil(64);
                if !sub_mask[..words].iter().all(|w| w.is_empty()) {
                    exclusion_set.remove_excluded_batch(suffix, sub_mask);
                }
                next_bound
            })
    }

    fn is_empty_in_range(&mut self, start: DocId, end: DocId) -> bool {
        if self.current.0 == TERMINATED {
            return true;
        }
        if let Some(empty) = self.buf.is_empty_in_range_within(start, end) {
            return empty;
        }
        self.underlying
            .is_empty_in_range(start.max(self.buf.end), end)
    }

    #[inline]
    fn doc(&self) -> DocId {
        self.current.0
    }

    fn size_hint(&self) -> u32 {
        self.underlying.size_hint()
    }

    fn cost(&self) -> DocSetCost {
        let underlying = self.underlying.cost();
        let excl_probe = self.exclusion_set.probe_cost();
        let hint = u64::from(self.underlying.size_hint());
        let batches = hint.div_ceil(u64::from(BLOCK_WINDOW));
        let batch_work = DriveCost(
            batches
                .saturating_mul(excl_probe.setup.0)
                .saturating_add(hint.saturating_mul(excl_probe.per_doc.0)),
        );
        DocSetCost::new(
            underlying.drive.saturating_add(batch_work),
            underlying.probe.saturating_add(excl_probe),
        )
    }
}

impl<TScorer, TExclusionSet> Scorer for BatchExcludeScorer<TScorer, TExclusionSet>
where
    TScorer: Scorer + 'static,
    TExclusionSet: ExclusionSet + 'static,
{
    #[inline]
    fn score(&mut self) -> Score {
        self.current.1
    }

    #[inline]
    fn constant_score(&self) -> Option<Score> {
        self.underlying_const_score
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::postings::tests::test_skip_against_unoptimized;
    use crate::query::VecDocSet;
    use crate::tests::sample_with_seed;

    #[test]
    fn test_exclude() {
        let mut exclude_scorer = Exclude::new(
            VecDocSet::from(vec![1, 2, 5, 8, 10, 15, 24]),
            ExcludedDocSet::new(VecDocSet::from(vec![1, 2, 3, 10, 16, 24])),
        );
        let mut els = vec![];
        while exclude_scorer.doc() != TERMINATED {
            els.push(exclude_scorer.doc());
            exclude_scorer.advance();
        }
        assert_eq!(els, vec![5, 8, 15]);
    }

    #[test]
    fn test_exclude_skip() {
        test_skip_against_unoptimized(
            || {
                Box::new(Exclude::new(
                    VecDocSet::from(vec![1, 2, 5, 8, 10, 15, 24]),
                    ExcludedDocSet::new(VecDocSet::from(vec![1, 2, 3, 10, 16, 24])),
                ))
            },
            vec![5, 8, 10, 15, 24],
        );
    }

    #[test]
    fn test_exclude_skip_random() {
        let sample_include = sample_with_seed(10_000, 0.1, 1);
        let sample_exclude = sample_with_seed(10_000, 0.05, 2);
        let sample_skip = sample_with_seed(10_000, 0.005, 3);
        test_skip_against_unoptimized(
            || {
                Box::new(Exclude::new(
                    VecDocSet::from(sample_include.clone()),
                    ExcludedDocSet::new(VecDocSet::from(sample_exclude.clone())),
                ))
            },
            sample_skip,
        );
    }

    /// Counts the `advance` calls on a wrapped docset, to show when a walk happens.
    struct CountingDocSet {
        inner: VecDocSet,
        advances: usize,
    }

    impl DocSet for CountingDocSet {
        fn advance(&mut self) -> DocId {
            self.advances += 1;
            self.inner.advance()
        }

        fn seek(&mut self, target: DocId) -> DocId {
            self.inner.seek(target)
        }

        fn doc(&self) -> DocId {
            self.inner.doc()
        }

        fn size_hint(&self) -> u32 {
            self.inner.size_hint()
        }
    }

    fn exclude_with_counts(
        include: Vec<DocId>,
        exclude: Vec<DocId>,
    ) -> Exclude<CountingDocSet, ExcludedDocSet<VecDocSet>> {
        Exclude::new(
            CountingDocSet {
                inner: VecDocSet::from(include),
                advances: 0,
            },
            ExcludedDocSet::new(VecDocSet::from(exclude)),
        )
    }

    #[test]
    fn seek_danger_found_leaves_valid_state() {
        let mut exclude =
            exclude_with_counts(vec![1, 2, 5, 8, 10, 15, 24], vec![1, 2, 3, 10, 16, 24]);
        assert_eq!(exclude.seek_danger(5), SeekDangerResult::Found);
        assert_eq!(exclude.doc(), 5);
        assert_eq!(exclude.advance(), 8);
    }

    #[test]
    fn seek_danger_on_an_excluded_doc_does_not_walk() {
        // Docs 10 to 19 are all excluded, so a seek to 10 would step through every one of them.
        let include: Vec<DocId> = (0..30).collect();
        let exclude: Vec<DocId> = (10..20).collect();
        let mut exclude = exclude_with_counts(include, exclude);
        let advances_before = exclude.underlying_docset.advances;

        assert_eq!(
            exclude.seek_danger(10),
            SeekDangerResult::SeekLowerBound(11)
        );
        assert_eq!(exclude.underlying_docset.advances, advances_before);

        // Later probes recover. The first kept doc is found in a valid state.
        assert_eq!(
            exclude.seek_danger(19),
            SeekDangerResult::SeekLowerBound(20)
        );
        assert_eq!(exclude.seek_danger(20), SeekDangerResult::Found);
        assert_eq!(exclude.doc(), 20);
        assert_eq!(exclude.advance(), 21);
    }

    #[test]
    fn seek_danger_passes_the_underlying_lower_bound_through() {
        let mut exclude =
            exclude_with_counts(vec![1, 2, 5, 8, 10, 15, 24], vec![1, 2, 3, 10, 16, 24]);
        // 6 is not in the underlying docset, whose own lower bound is its next doc, 8.
        assert_eq!(exclude.seek_danger(6), SeekDangerResult::SeekLowerBound(8));
        assert_eq!(
            exclude.seek_danger(TERMINATED),
            SeekDangerResult::SeekLowerBound(TERMINATED)
        );
    }

    #[test]
    fn seek_danger_matches_seek() {
        let include = vec![1, 2, 5, 8, 10, 15, 24];
        let exclude = vec![1, 2, 3, 10, 16, 24];
        for target in 0..30 {
            let mut fresh = Exclude::new(
                VecDocSet::from(include.clone()),
                ExcludedDocSet::new(VecDocSet::from(exclude.clone())),
            );
            // `seek` must not move backwards, and the leading excluded docs are already skipped.
            if target < fresh.doc() {
                continue;
            }
            let next = fresh.seek(target);
            let mut probed = Exclude::new(
                VecDocSet::from(include.clone()),
                ExcludedDocSet::new(VecDocSet::from(exclude.clone())),
            );
            match probed.seek_danger(target) {
                SeekDangerResult::Found => assert_eq!(next, target, "target {target}"),
                SeekDangerResult::SeekLowerBound(bound) => {
                    assert_ne!(next, target, "target {target}");
                    assert!(
                        bound > target && bound <= next,
                        "target {target}: bound {bound}, next {next}"
                    );
                }
            }
        }
    }

    #[test]
    fn is_empty_in_range_follows_the_underlying_docset() {
        // A fresh docset per probe, since `seek_danger` targets must increase within one docset.
        let fresh = || exclude_with_counts(vec![0, 5, 8], vec![5]);
        assert!(fresh().is_empty_in_range(6, 7));
        assert!(fresh().is_empty_in_range(9, 20));
        // 5 is excluded, but telling needs the walk, so the range is not known to be empty.
        assert!(!fresh().is_empty_in_range(5, 5));
        assert!(!fresh().is_empty_in_range(0, 8));
    }
}
