use std::ops::DerefMut;

use downcast_rs::impl_downcast;

use crate::docset::DocSet;
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

pub trait PruningScorer: Scorer {
    fn set_threshold(&mut self, score: Score);
}

impl_downcast!(PruningScorer);

/// A scorer that bounds its score per posting block, so Block-WAND can skip the blocks that
/// cannot beat the current threshold. A term scorer reads the bound from its skip data, and a
/// scorer over several posting lists sums the bounds of its members.
pub trait BlockMaxScorer: Scorer {
    /// An upper bound of `score()` for every doc in the current block.
    fn block_max_score(&mut self) -> Score;

    /// The last doc of the current block, or `TERMINATED` in the last block.
    fn last_doc_in_block(&self) -> DocId;

    /// Moves to the block that holds `target` without reading it.
    fn seek_block(&mut self, target: DocId);

    /// An upper bound of `score()` over every doc.
    fn max_score(&self) -> Score;
}

impl<TScorer: BlockMaxScorer + ?Sized> BlockMaxScorer for Box<TScorer> {
    #[inline]
    fn block_max_score(&mut self) -> Score {
        self.deref_mut().block_max_score()
    }

    #[inline]
    fn last_doc_in_block(&self) -> DocId {
        (**self).last_doc_in_block()
    }

    #[inline]
    fn seek_block(&mut self, target: DocId) {
        self.deref_mut().seek_block(target)
    }

    #[inline]
    fn max_score(&self) -> Score {
        (**self).max_score()
    }
}

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
}
