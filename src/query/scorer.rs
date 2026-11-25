use std::ops::DerefMut;

use downcast_rs::impl_downcast;

use crate::docset::DocSet;
use crate::Score;

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
