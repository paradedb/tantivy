use crate::query::Scorer;
use crate::Score;

/// The `ScoreCombiner` trait defines how to compute
/// an overall score given a list of scores.
pub trait ScoreCombiner: Default + Clone + Send + Copy + 'static {
    /// Whether Block-WAND may drive this combiner.
    ///
    /// Block-WAND prunes blocks by comparing the *sum* of the per-term block
    /// maxima against the threshold, and scores the surviving documents the
    /// same way. That is only the right answer for a combiner that sums. A
    /// combiner aggregating any other way (dis_max takes the maximum) has to
    /// see every matching term, so it must not be pruned this way.
    const SUPPORTS_BLOCK_WAND: bool = false;

    /// Aggregates the score combiner with the given scorer.
    ///
    /// The `ScoreCombiner` may decide to call `.scorer.score()`
    /// or not.
    fn update<TScorer: Scorer>(&mut self, scorer: &mut TScorer);

    /// Clears the score combiner state back to its initial state.
    fn clear(&mut self);

    /// Returns the aggregate score.
    fn score(&self) -> Score;

    /// Combines an array of upper bounds into an overall upper bound for this combiner.
    fn combine_max_scores(&self, max_scores: &[Score]) -> Score {
        max_scores.iter().copied().sum()
    }
}

/// Just ignores scores. The `DoNothingCombiner` does not
/// even call the scorers `.score()` function.
///
/// It is useful to optimize the case when scoring is disabled.
#[derive(Default, Clone, Copy)] //< these should not be too much work :)
pub struct DoNothingCombiner;

impl ScoreCombiner for DoNothingCombiner {
    fn update<TScorer: Scorer>(&mut self, _scorer: &mut TScorer) {}

    fn clear(&mut self) {}

    #[inline]
    fn score(&self) -> Score {
        1.0
    }

    fn combine_max_scores(&self, _max_scores: &[Score]) -> Score {
        1.0
    }
}

/// Sums the score of different scorers.
#[derive(Default, Clone, Copy)]
pub struct SumCombiner {
    score: Score,
}

impl ScoreCombiner for SumCombiner {
    const SUPPORTS_BLOCK_WAND: bool = true;

    fn update<TScorer: Scorer>(&mut self, scorer: &mut TScorer) {
        self.score += scorer.score();
    }

    fn clear(&mut self) {
        self.score = 0.0;
    }

    #[inline]
    fn score(&self) -> Score {
        self.score
    }
}

/// Take max score of different scorers
/// and optionally sum it with other matches multiplied by `tie_breaker`
#[derive(Default, Clone, Copy)]
pub struct DisjunctionMaxCombiner {
    max: Score,
    sum: Score,
    tie_breaker: Score,
}

impl DisjunctionMaxCombiner {
    /// Creates `DisjunctionMaxCombiner` with tie breaker
    pub fn with_tie_breaker(tie_breaker: Score) -> DisjunctionMaxCombiner {
        DisjunctionMaxCombiner {
            max: 0.0,
            sum: 0.0,
            tie_breaker,
        }
    }
}

impl ScoreCombiner for DisjunctionMaxCombiner {
    fn update<TScorer: Scorer>(&mut self, scorer: &mut TScorer) {
        let score = scorer.score();
        self.max = Score::max(score, self.max);
        self.sum += score;
    }

    fn clear(&mut self) {
        self.max = 0.0;
        self.sum = 0.0;
    }

    #[inline]
    fn score(&self) -> Score {
        self.max + (self.sum - self.max) * self.tie_breaker
    }

    fn combine_max_scores(&self, max_scores: &[Score]) -> Score {
        if max_scores.is_empty() {
            return 0.0;
        }
        let mut max = 0.0;
        let mut sum = 0.0;
        for &s in max_scores {
            max = Score::max(s, max);
            sum += s;
        }
        max + (sum - max) * self.tie_breaker
    }
}
