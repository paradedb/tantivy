use crate::query::Scorer;
use crate::{Ctid, Score, INVALID_CTID};

/// The `ScoreCombiner` trait defines how to compute
/// an overall score given a list of scores.
pub trait ScoreCombiner: Default + Clone + Send + Copy + 'static {
    /// Aggregates the score combiner with the given scorer.
    ///
    /// The `ScoreCombiner` may decide to call `.scorer.score()`
    /// or not.
    #[must_use]
    fn update<TScorer: Scorer>(&mut self, scorer: &mut TScorer) -> Ctid;

    /// Clears the score combiner state back to its initial state.
    fn clear(&mut self);

    /// Returns the aggregate score.
    fn score(&self) -> (Score, Ctid);
}

/// Just ignores scores. The `DoNothingCombiner` does not
/// even call the scorers `.score()` function.
///
/// It is useful to optimize the case when scoring is disabled.
#[derive(Clone, Copy)] //< these should not be too much work :)
pub struct DoNothingCombiner {
    ctid: Ctid,
}

impl Default for DoNothingCombiner {
    fn default() -> Self {
        Self { ctid: INVALID_CTID }
    }
}

impl ScoreCombiner for DoNothingCombiner {
    fn update<TScorer: Scorer>(&mut self, scorer: &mut TScorer) -> Ctid {
        let (_, ctid) = scorer.score();
        self.ctid = ctid;
        ctid
    }

    fn clear(&mut self) {}

    fn score(&self) -> (Score, Ctid) {
        (1.0, self.ctid)
    }
}

/// Sums the score of different scorers.
#[derive(Default, Clone, Copy)]
pub struct SumCombiner {
    score: Score,
    ctid: Ctid,
}

impl ScoreCombiner for SumCombiner {
    fn update<TScorer: Scorer>(&mut self, scorer: &mut TScorer) -> Ctid {
        let (score, ctid) = scorer.score();
        self.score += score;
        self.ctid = ctid;
        ctid
    }

    fn clear(&mut self) {
        self.score = 0.0;
    }

    fn score(&self) -> (Score, Ctid) {
        (self.score, self.ctid)
    }
}

/// Take max score of different scorers
/// and optionally sum it with other matches multiplied by `tie_breaker`
#[derive(Clone, Copy)]
pub struct DisjunctionMaxCombiner {
    max: Score,
    sum: Score,
    tie_breaker: Score,
    ctid: Ctid,
}

impl Default for DisjunctionMaxCombiner {
    fn default() -> Self {
        Self {
            max: 0.0,
            sum: 0.0,
            tie_breaker: 0.0,
            ctid: INVALID_CTID,
        }
    }
}

impl DisjunctionMaxCombiner {
    /// Creates `DisjunctionMaxCombiner` with tie breaker
    pub fn with_tie_breaker(tie_breaker: Score) -> DisjunctionMaxCombiner {
        DisjunctionMaxCombiner {
            max: 0.0,
            sum: 0.0,
            tie_breaker,
            ctid: INVALID_CTID,
        }
    }
}

impl ScoreCombiner for DisjunctionMaxCombiner {
    fn update<TScorer: Scorer>(&mut self, scorer: &mut TScorer) -> Ctid {
        let (score, ctid) = scorer.score();
        self.max = Score::max(score, self.max);
        self.sum += score;
        self.ctid = ctid;
        ctid
    }

    fn clear(&mut self) {
        self.max = 0.0;
        self.sum = 0.0;
    }

    fn score(&self) -> (Score, Ctid) {
        let score = self.max + (self.sum - self.max) * self.tie_breaker;
        (score, self.ctid)
    }
}
