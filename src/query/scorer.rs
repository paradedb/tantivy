use std::ops::DerefMut;

use common::TinySet;
use downcast_rs::impl_downcast;

use crate::docset::{DocSet, DocSetCost, SeekDangerResult};
use crate::{DocId, Score, TERMINATED};

/// Scored set of documents matching a query within a specific segment.
///
/// See [`Query`](crate::query::Query).
pub trait Scorer: downcast_rs::Downcast + DocSet + 'static {
    /// Returns the score.
    ///
    /// This method will perform a bit of computation and is not cached.
    fn score(&mut self) -> Score;

    /// Returns the constant score produced for every document by this scorer,
    /// or `None` if the scorer produces variable scores per document.
    fn constant_score(&self) -> Option<Score> {
        None
    }
}

impl_downcast!(Scorer);

impl<TScorer: Scorer + ?Sized> Scorer for Box<TScorer> {
    #[inline]
    fn score(&mut self) -> Score {
        self.deref_mut().score()
    }

    #[inline]
    fn constant_score(&self) -> Option<Score> {
        (**self).constant_score()
    }
}

/// A [`Scorer`] that may not yet be positioned on its first matching document.
///
/// Returned by
/// [`Weight::scorer_for_candidates_unpositioned`](crate::query::Weight::scorer_for_candidates_unpositioned)
/// when a weight defers full-segment bitset construction or scanning to the first match until
/// an upstream driver supplies candidate documents.
///
/// Unlike [`Scorer`] / [`DocSet`], `UnpositionedScorer` does not expose `doc()`, `score()`, or
/// `advance()`, preventing callers from treating an unpositioned doc ID as a match. Callers can
/// probe it directly via [`Self::seek_danger`] / [`Self::seek_danger_batch`] /
/// [`Self::is_empty_in_range`], wrap it in
/// [`ExcludedDocSet::from_unpositioned`](crate::query::ExcludedDocSet::from_unpositioned), or
/// explicitly position it via [`Self::into_positioned`].
pub struct UnpositionedScorer {
    scorer: Box<dyn Scorer>,
    positioned: bool,
}

impl UnpositionedScorer {
    /// Wraps a [`Scorer`] that is already positioned on its first matching document (or
    /// [`TERMINATED`]).
    #[inline]
    pub fn from_positioned(scorer: Box<dyn Scorer>) -> Self {
        Self {
            scorer,
            positioned: true,
        }
    }

    /// Wraps a [`Scorer`] whose initial position has not yet been computed.
    ///
    /// Immediately after construction, `scorer.doc()` must return [`TERMINATED`] if known empty,
    /// and a lower bound (`<=` first match) otherwise. `scorer.seek(target)` for any
    /// `target >= scorer.doc()` must position `scorer` on its first matching document `>= target`
    /// (or [`TERMINATED`]).
    #[inline]
    pub fn from_unpositioned(scorer: Box<dyn Scorer>) -> Self {
        let positioned = scorer.doc() == TERMINATED;
        Self { scorer, positioned }
    }

    /// Transforms the inner scorer while preserving its positioning state (for example wrapping
    /// in [`ConstScorer`](crate::query::ConstScorer)).
    #[inline]
    pub fn map_scorer(self, f: impl FnOnce(Box<dyn Scorer>) -> Box<dyn Scorer>) -> Self {
        Self {
            scorer: f(self.scorer),
            positioned: self.positioned,
        }
    }

    /// Returns `true` if the underlying scorer is known to be positioned on a matching document
    /// (or [`TERMINATED`]).
    #[inline]
    pub fn is_positioned(&self) -> bool {
        self.positioned
    }

    /// Returns a lower bound on the first matching document, or [`TERMINATED`] if known empty.
    #[inline]
    pub fn lower_bound(&self) -> DocId {
        self.scorer.doc()
    }

    /// Returns `true` if the scorer is known to match no documents.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.scorer.doc() == TERMINATED
    }

    /// Probes whether `target` matches, following the [`DocSet::seek_danger`] contract.
    #[inline]
    pub fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        let result = self.scorer.seek_danger(target);
        self.positioned = result == SeekDangerResult::Found;
        result
    }

    /// Probes a sorted batch of candidates in place, following the [`DocSet::seek_danger_batch`]
    /// contract.
    #[inline]
    pub fn seek_danger_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) -> DocId {
        if !docs.is_empty() {
            self.positioned = false;
        }
        self.scorer.seek_danger_batch(docs, mask)
    }

    /// Returns `true` if the scorer is known to have no matches in `[start, end]`, following the
    /// [`DocSet::is_empty_in_range`] contract.
    #[inline]
    pub fn is_empty_in_range(&mut self, start: DocId, end: DocId) -> bool {
        self.positioned = false;
        self.scorer.is_empty_in_range(start, end)
    }

    #[inline]
    pub fn size_hint(&self) -> u32 {
        self.scorer.size_hint()
    }

    #[inline]
    pub fn cost(&self) -> DocSetCost {
        self.scorer.cost()
    }

    #[inline]
    pub fn constant_score(&self) -> Option<Score> {
        self.scorer.constant_score()
    }

    #[inline]
    pub fn is<T: Scorer>(&self) -> bool {
        self.scorer.is::<T>()
    }

    /// Positions the underlying scorer on its first matching document (or [`TERMINATED`]) if not
    /// already positioned, and returns it as a [`Scorer`].
    ///
    /// Must not be called after a `seek_danger` / `seek_danger_batch` call that left the scorer
    /// in the danger zone.
    #[inline]
    pub fn into_positioned(mut self) -> Box<dyn Scorer> {
        self.position_at(0);
        self.scorer
    }

    /// Seeks the underlying scorer to at least `target`, ensuring it is positioned even when
    /// `target == self.lower_bound()`.
    #[inline]
    pub(crate) fn position_at(&mut self, target: DocId) -> DocId {
        let current = self.scorer.doc();
        if current == TERMINATED {
            self.positioned = true;
            return TERMINATED;
        }
        if !self.positioned || current < target {
            self.positioned = true;
            self.scorer.seek(target.max(current))
        } else {
            current
        }
    }

    /// Returns a mutable reference to the underlying scorer once positioned.
    #[inline]
    pub(crate) fn positioned_mut(&mut self) -> &mut dyn Scorer {
        debug_assert!(self.positioned);
        self.scorer.as_mut()
    }

    /// Returns a reference to the underlying concrete scorer when already positioned.
    #[inline]
    pub(crate) fn downcast_ref_positioned<T: Scorer>(&self) -> Option<&T> {
        self.positioned
            .then(|| self.scorer.downcast_ref::<T>())
            .flatten()
    }

    /// Unwraps the inner `Box<dyn Scorer>` without positioning it.
    ///
    /// Callers must only invoke probe-safe methods (`doc()` as lower bound, `seek_danger`,
    /// `seek_danger_batch`, `is_empty_in_range`, `cost`, `size_hint`, `constant_score`) or
    /// `seek(target >= doc())` on the returned scorer.
    #[inline]
    pub(crate) fn into_raw_unpositioned(self) -> Box<dyn Scorer> {
        self.scorer
    }
}

pub trait PruningScorer: Scorer {
    fn set_threshold(&mut self, score: Score);
}

impl_downcast!(PruningScorer);

impl<TScorer: PruningScorer + ?Sized> PruningScorer for Box<TScorer> {
    #[inline]
    fn set_threshold(&mut self, score: Score) {
        self.deref_mut().set_threshold(score);
    }
}

pub struct BasicPruningScorer {
    scorer: Box<dyn Scorer>,
    threshold: Score,
    current: (DocId, Score),
}
impl BasicPruningScorer {
    pub fn new(scorer: Box<dyn Scorer>, threshold: Score) -> Self {
        let mut pruning = Self {
            scorer,
            threshold,
            current: (0, Score::MIN),
        };
        pruning.advance();
        pruning
    }
}
impl Scorer for BasicPruningScorer {
    #[inline]
    fn score(&mut self) -> Score {
        self.current.1
    }

    #[inline]
    fn constant_score(&self) -> Option<Score> {
        self.scorer.constant_score()
    }
}
impl PruningScorer for BasicPruningScorer {
    #[inline]
    fn set_threshold(&mut self, score: Score) {
        self.threshold = score;
    }
}
impl DocSet for BasicPruningScorer {
    #[inline]
    fn doc(&self) -> crate::DocId {
        self.current.0
    }

    fn advance(&mut self) -> crate::DocId {
        let mut doc_id = self.scorer.doc();
        while doc_id != TERMINATED {
            let score = self.scorer.score();
            if score > self.threshold {
                self.current = (doc_id, score);
                self.scorer.advance();
                return doc_id;
            }
            doc_id = self.scorer.advance();
        }
        self.current = (TERMINATED, Score::MIN);
        TERMINATED
    }

    /// The number of elements yielded by a PruningScorer depends on the threshold and cannot be
    /// computed ahead of time, so just defer to the inner scorer.
    fn size_hint(&self) -> u32 {
        self.scorer.size_hint()
    }

    fn cost(&self) -> crate::DocSetCost {
        self.scorer.cost()
    }
}
