use crate::query::Bm25Weight;
use crate::Score;

pub(super) fn optimistic_score(weight: &Bm25Weight, term_freq: u32) -> Score {
    let score = weight.score(0, term_freq);
    if score >= 0.0 {
        score
    } else {
        Score::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::optimistic_score;
    use crate::query::Bm25Weight;
    use crate::{Bm25Params, Score};

    #[test]
    fn test_candidate_bound_dominates_all_fieldnorms() {
        for k1 in [0.0, 0.5, 1.2, 2.0, 10.0] {
            for b in [0.0, 0.25, 0.75, 1.0] {
                for average in [1.0, 8.0, 100.0, 1_000_000.0] {
                    let weight =
                        Bm25Weight::for_one_term(10, 1_000, average, Bm25Params::new(k1, b));
                    for freq in [1, 2, 16, 127, 1_000, u32::MAX] {
                        let upper = optimistic_score(&weight, freq);
                        for norm in 0..=255 {
                            assert!(upper >= weight.score(norm, freq));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_candidate_bound_does_not_prune_negative_or_nan_weights() {
        let weight = Bm25Weight::for_one_term(10, 1_000, 100.0, Bm25Params::default());
        assert_eq!(optimistic_score(&weight.boost_by(-1.0), 1), Score::INFINITY);
        assert_eq!(
            optimistic_score(&weight.boost_by(Score::NAN), 1),
            Score::INFINITY
        );
    }
}
