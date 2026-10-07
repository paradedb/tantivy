use std::collections::HashMap;

use crate::docset::{DocSet, COLLECT_BLOCK_BUFFER_LEN};
use crate::index::SegmentReader;
use crate::postings::{FreqReadingOption, SegmentPostings};
use crate::query::bitmap_combination::{BitmapCombination, BitmapOperation};
use crate::query::boolean_query::{
    BlockWandIntersectionScorer, BlockWandSingleScorer, BlockWandUnionScorer,
};
use crate::query::disjunction::Disjunction;
use crate::query::explanation::does_not_match;
use crate::query::phrase_query::PhraseScorer;
use crate::query::score_combiner::{DoNothingCombiner, ScoreCombiner};
use crate::query::scorer::BasicPruningScorer;
use crate::query::term_query::TermScorer;
use crate::query::weight::{for_each_docset_buffered, for_each_pruning_scorer, for_each_scorer};
use crate::query::{
    intersect_scorers as intersect_scored_scorers, AllScorer, BufferedUnionScorer,
    DisjunctionPruning, EmptyScorer, Exclude, Explanation, Occur, RequiredOptionalScorer, Scorer,
    Weight,
};
use crate::{DocId, Score, TERMINATED};

fn intersect_scorers(
    scorers: Vec<Box<dyn Scorer>>,
    num_docs: u32,
    scoring_enabled: bool,
    bitmap_enabled: bool,
) -> Box<dyn Scorer> {
    if bitmap_enabled
        && !scoring_enabled
        && scorers.len() > 1
        && scorers.iter().any(|scorer| scorer.has_fast_bitset())
        && scorers
            .iter()
            .all(|scorer| scorer.size_hint().saturating_mul(32) >= num_docs)
        && !scorers.iter().any(|phrase| {
            phrase.is::<PhraseScorer<SegmentPostings>>()
                && scorers.iter().any(|driver| {
                    !driver.is::<PhraseScorer<SegmentPostings>>()
                        && driver.cost() < phrase.cost()
                        && u64::from(driver.size_hint()) * 2 <= u64::from(num_docs)
                })
        })
    {
        Box::new(BitmapCombination::new(
            scorers,
            BitmapOperation::Intersection,
            num_docs,
        ))
    } else {
        intersect_scored_scorers(scorers, num_docs)
    }
}

pub(crate) enum SpecializedScorer {
    TermUnion(Vec<TermScorer>),
    TermIntersection(Vec<TermScorer>),
    FilteredTermUnion {
        terms: Vec<TermScorer>,
        filter: Box<dyn Scorer>,
    },
    FilteredTermIntersection {
        terms: Vec<TermScorer>,
        filter: Box<dyn Scorer>,
    },
    Other(Box<dyn Scorer>),
}

fn scorer_disjunction<TScoreCombiner>(
    scorers: Vec<Box<dyn Scorer>>,
    score_combiner: TScoreCombiner,
    minimum_match_required: usize,
) -> Box<dyn Scorer>
where
    TScoreCombiner: ScoreCombiner,
{
    debug_assert!(!scorers.is_empty());
    debug_assert!(minimum_match_required > 1);
    if scorers.len() == 1 {
        return scorers.into_iter().next().unwrap(); // Safe unwrap.
    }
    Box::new(Disjunction::new(
        scorers,
        score_combiner,
        minimum_match_required,
    ))
}

/// num_docs is the number of documents in the segment.
fn scorer_union<TScoreCombiner>(
    scorers: Vec<Box<dyn Scorer>>,
    score_combiner_fn: impl Fn() -> TScoreCombiner,
    num_docs: u32,
    bitmap_enabled: bool,
) -> SpecializedScorer
where
    TScoreCombiner: ScoreCombiner,
{
    assert!(!scorers.is_empty());
    if scorers.len() == 1 && !scorers[0].is::<TermScorer>() {
        return SpecializedScorer::Other(scorers.into_iter().next().unwrap()); //< we checked the size beforehand
    }
    if bitmap_enabled
        && TScoreCombiner::constant_score() == Some(1.0)
        && scorers.iter().any(|scorer| scorer.has_fast_bitset())
    {
        return SpecializedScorer::Other(Box::new(BitmapCombination::new(
            scorers,
            BitmapOperation::Union,
            num_docs,
        )));
    }
    {
        let is_all_term_queries = scorers.iter().all(|scorer| scorer.is::<TermScorer>());
        if is_all_term_queries {
            let scorers: Vec<TermScorer> = scorers
                .into_iter()
                .map(|scorer| *(scorer.downcast::<TermScorer>().map_err(|_| ()).unwrap()))
                .collect();
            if scorers
                .iter()
                .all(|scorer| scorer.freq_reading_option() == FreqReadingOption::ReadFreq)
            {
                // Block wand is only available if we read frequencies.
                return SpecializedScorer::TermUnion(scorers);
            } else if scorers.len() == 1 {
                // Single TermScorer without freq reading — unwrap directly.
                return SpecializedScorer::Other(Box::new(scorers.into_iter().next().unwrap()));
            } else {
                return SpecializedScorer::Other(Box::new(BufferedUnionScorer::build(
                    scorers,
                    score_combiner_fn,
                    num_docs,
                )));
            }
        }
    }
    SpecializedScorer::Other(Box::new(BufferedUnionScorer::build(
        scorers,
        score_combiner_fn,
        num_docs,
    )))
}

fn into_box_scorer<TScoreCombiner: ScoreCombiner>(
    scorer: SpecializedScorer,
    score_combiner_fn: impl Fn() -> TScoreCombiner,
    num_docs: u32,
    bitmap_enabled: bool,
) -> Box<dyn Scorer> {
    match scorer {
        SpecializedScorer::TermUnion(mut term_scorers) => {
            if term_scorers.len() == 1 {
                Box::new(term_scorers.pop().unwrap())
            } else {
                let union_scorer =
                    BufferedUnionScorer::build(term_scorers, score_combiner_fn, num_docs);
                Box::new(union_scorer)
            }
        }
        SpecializedScorer::TermIntersection(term_scorers) => {
            let boxed_scorers: Vec<Box<dyn Scorer>> = term_scorers
                .into_iter()
                .map(|s| Box::new(s) as Box<dyn Scorer>)
                .collect();
            intersect_scorers(
                boxed_scorers,
                num_docs,
                TScoreCombiner::constant_score().is_none(),
                bitmap_enabled,
            )
        }
        SpecializedScorer::FilteredTermUnion { mut terms, filter } => {
            let term_scorer: Box<dyn Scorer> = if terms.len() == 1 {
                Box::new(terms.pop().unwrap())
            } else {
                Box::new(BufferedUnionScorer::build(
                    terms,
                    score_combiner_fn,
                    num_docs,
                ))
            };
            intersect_scorers(
                vec![term_scorer, filter],
                num_docs,
                TScoreCombiner::constant_score().is_none(),
                bitmap_enabled,
            )
        }
        SpecializedScorer::FilteredTermIntersection { terms, filter } => {
            let mut boxed_scorers: Vec<Box<dyn Scorer>> = terms
                .into_iter()
                .map(|s| Box::new(s) as Box<dyn Scorer>)
                .collect();
            boxed_scorers.push(filter);
            intersect_scorers(
                boxed_scorers,
                num_docs,
                TScoreCombiner::constant_score().is_none(),
                bitmap_enabled,
            )
        }
        SpecializedScorer::Other(scorer) => scorer,
    }
}

/// Returns the effective MUST scorer, accounting for removed AllScorers.
///
/// When AllScorer instances are removed from must_scorers as an optimization,
/// we must restore the "match all" semantics if the list becomes empty.
fn effective_must_scorer(
    must_scorers: Vec<Box<dyn Scorer>>,
    removed_all_scorer_count: usize,
    max_doc: DocId,
    num_docs: u32,
    scoring_enabled: bool,
    bitmap_enabled: bool,
) -> Option<Box<dyn Scorer>> {
    if must_scorers.is_empty() {
        if removed_all_scorer_count > 0 {
            // Had AllScorer(s) only - all docs match
            Some(Box::new(AllScorer::new(max_doc)))
        } else {
            // No MUST constraint at all
            None
        }
    } else {
        Some(intersect_scorers(
            must_scorers,
            num_docs,
            scoring_enabled,
            bitmap_enabled,
        ))
    }
}

/// Returns a SHOULD scorer with AllScorer union if any were removed.
///
/// For union semantics (OR): if any SHOULD clause was an AllScorer, the result
/// should include all documents. We restore this by unioning with AllScorer.
///
/// When `scoring_enabled` is false, we can just return AllScorer alone since
/// we don't need score contributions from the should_scorer.
fn effective_should_scorer_for_union<TScoreCombiner: ScoreCombiner>(
    should_scorer: SpecializedScorer,
    removed_all_scorer_count: usize,
    max_doc: DocId,
    num_docs: u32,
    score_combiner_fn: impl Fn() -> TScoreCombiner,
    scoring_enabled: bool,
    bitmap_enabled: bool,
) -> SpecializedScorer {
    if removed_all_scorer_count > 0 {
        if scoring_enabled {
            // Need to union to get score contributions from both
            let all_scorers: Vec<Box<dyn Scorer>> = vec![
                into_box_scorer(should_scorer, &score_combiner_fn, num_docs, bitmap_enabled),
                Box::new(AllScorer::new(max_doc)),
            ];
            SpecializedScorer::Other(Box::new(BufferedUnionScorer::build(
                all_scorers,
                score_combiner_fn,
                num_docs,
            )))
        } else {
            // Scoring disabled - AllScorer alone is sufficient
            SpecializedScorer::Other(Box::new(AllScorer::new(max_doc)))
        }
    } else {
        should_scorer
    }
}

enum ShouldScorersCombinationMethod {
    // Should scorers are irrelevant.
    Ignored,
    // Only contributes to final score.
    Optional(SpecializedScorer),
    // Regardless of score, the should scorers may impact whether a document is matching or not.
    Required(SpecializedScorer),
}

/// Weight associated to the `BoolQuery`.
pub struct BooleanWeight<TScoreCombiner: ScoreCombiner> {
    weights: Vec<(Occur, Box<dyn Weight>)>,
    minimum_number_should_match: usize,
    scoring_enabled: bool,
    disjunction_pruning: DisjunctionPruning,
    score_combiner_fn: Box<dyn Fn() -> TScoreCombiner + Sync + Send>,
}

impl<TScoreCombiner: ScoreCombiner> BooleanWeight<TScoreCombiner> {
    /// Creates a new boolean weight.
    pub fn new(
        weights: Vec<(Occur, Box<dyn Weight>)>,
        scoring_enabled: bool,
        score_combiner_fn: Box<dyn Fn() -> TScoreCombiner + Sync + Send + 'static>,
    ) -> BooleanWeight<TScoreCombiner> {
        BooleanWeight {
            weights,
            scoring_enabled,
            score_combiner_fn,
            minimum_number_should_match: 1,
            disjunction_pruning: DisjunctionPruning::Auto,
        }
    }

    /// Create a new boolean weight with minimum number of required should clauses specified.
    pub fn with_minimum_number_should_match(
        weights: Vec<(Occur, Box<dyn Weight>)>,
        minimum_number_should_match: usize,
        scoring_enabled: bool,
        score_combiner_fn: Box<dyn Fn() -> TScoreCombiner + Sync + Send + 'static>,
    ) -> BooleanWeight<TScoreCombiner> {
        BooleanWeight {
            weights,
            minimum_number_should_match,
            scoring_enabled,
            score_combiner_fn,
            disjunction_pruning: DisjunctionPruning::Auto,
        }
    }

    pub(crate) fn with_disjunction_pruning(mut self, pruning: DisjunctionPruning) -> Self {
        self.disjunction_pruning = pruning;
        self
    }

    fn should_use_block_maxscore(&self, scorers: &[TermScorer], max_doc: DocId) -> bool {
        match self.disjunction_pruning {
            DisjunctionPruning::Auto => {
                super::block_maxscore::should_use_block_maxscore(scorers, max_doc)
            }
            DisjunctionPruning::BlockWand => false,
            DisjunctionPruning::BlockMaxScore => true,
        }
    }

    fn per_occur_scorers(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<HashMap<Occur, Vec<Box<dyn Scorer>>>> {
        let mut per_occur_scorers: HashMap<Occur, Vec<Box<dyn Scorer>>> = HashMap::new();
        for (occur, subweight) in &self.weights {
            let sub_scorer: Box<dyn Scorer> = subweight.scorer(reader, boost)?;
            per_occur_scorers
                .entry(*occur)
                .or_default()
                .push(sub_scorer);
        }
        Ok(per_occur_scorers)
    }

    fn complex_scorer<TComplexScoreCombiner: ScoreCombiner>(
        &self,
        reader: &SegmentReader,
        boost: Score,
        score_combiner_fn: impl Fn() -> TComplexScoreCombiner,
    ) -> crate::Result<SpecializedScorer> {
        let num_docs = reader.num_docs();
        let mut per_occur_scorers = self.per_occur_scorers(reader, boost)?;

        // Indicate how should clauses are combined with must clauses.
        let mut must_scorers: Vec<Box<dyn Scorer>> =
            per_occur_scorers.remove(&Occur::Must).unwrap_or_default();
        let must_special_scorer_counts = remove_and_count_all_and_empty_scorers(&mut must_scorers);

        if must_special_scorer_counts.num_empty_scorers > 0 {
            return Ok(SpecializedScorer::Other(Box::new(EmptyScorer)));
        }

        let mut should_scorers = per_occur_scorers.remove(&Occur::Should).unwrap_or_default();
        let should_special_scorer_counts =
            remove_and_count_all_and_empty_scorers(&mut should_scorers);

        let mut exclude_scorers: Vec<Box<dyn Scorer>> = per_occur_scorers
            .remove(&Occur::MustNot)
            .unwrap_or_default();
        let exclude_special_scorer_counts =
            remove_and_count_all_and_empty_scorers(&mut exclude_scorers);

        if exclude_special_scorer_counts.num_all_scorers > 0 {
            // We exclude all documents at one point.
            return Ok(SpecializedScorer::Other(Box::new(EmptyScorer)));
        }

        let effective_minimum_number_should_match = self
            .minimum_number_should_match
            .saturating_sub(should_special_scorer_counts.num_all_scorers);

        let should_scorers: ShouldScorersCombinationMethod = {
            let num_of_should_scorers = should_scorers.len();
            if effective_minimum_number_should_match > num_of_should_scorers {
                // We don't have enough scorers to satisfy the minimum number of should matches.
                // The request will match no documents.
                return Ok(SpecializedScorer::Other(Box::new(EmptyScorer)));
            }
            match effective_minimum_number_should_match {
                0 if num_of_should_scorers == 0 => ShouldScorersCombinationMethod::Ignored,
                0 => ShouldScorersCombinationMethod::Optional(scorer_union(
                    should_scorers,
                    &score_combiner_fn,
                    num_docs,
                    reader.bitmap_postings_enabled,
                )),
                1 => ShouldScorersCombinationMethod::Required(scorer_union(
                    should_scorers,
                    &score_combiner_fn,
                    num_docs,
                    reader.bitmap_postings_enabled,
                )),
                n if num_of_should_scorers == n => {
                    // When num_of_should_scorers equals the number of should clauses,
                    // they are no different from must clauses.
                    must_scorers.append(&mut should_scorers);
                    ShouldScorersCombinationMethod::Ignored
                }
                _ => ShouldScorersCombinationMethod::Required(SpecializedScorer::Other(
                    scorer_disjunction(
                        should_scorers,
                        score_combiner_fn(),
                        effective_minimum_number_should_match,
                    ),
                )),
            }
        };

        let include_scorer = match (should_scorers, must_scorers) {
            (ShouldScorersCombinationMethod::Ignored, must_scorers) => {
                // No SHOULD clauses (or they were absorbed into MUST).
                // Result depends entirely on MUST + any removed AllScorers.
                let combined_all_scorer_count = must_special_scorer_counts.num_all_scorers
                    + should_special_scorer_counts.num_all_scorers;

                let mut term_scorers = Vec::new();
                let mut other_scorers = Vec::new();
                for scorer in must_scorers {
                    if scorer.is::<TermScorer>() {
                        let term_scorer =
                            *(scorer.downcast::<TermScorer>().map_err(|_| ()).unwrap());
                        if term_scorer.freq_reading_option() == FreqReadingOption::ReadFreq {
                            term_scorers.push(term_scorer);
                        } else {
                            other_scorers.push(Box::new(term_scorer) as Box<dyn Scorer>);
                        }
                    } else {
                        other_scorers.push(scorer);
                    }
                }

                if combined_all_scorer_count == 0 && other_scorers.is_empty() {
                    if term_scorers.len() >= 2 {
                        SpecializedScorer::TermIntersection(term_scorers)
                    } else if term_scorers.len() == 1 {
                        SpecializedScorer::TermUnion(term_scorers)
                    } else {
                        SpecializedScorer::Other(Box::new(EmptyScorer))
                    }
                } else if combined_all_scorer_count == 0
                    && !term_scorers.is_empty()
                    && !other_scorers.is_empty()
                {
                    let filter = intersect_scorers(
                        other_scorers,
                        num_docs,
                        self.scoring_enabled,
                        reader.bitmap_postings_enabled,
                    );
                    if term_scorers.len() >= 2 {
                        SpecializedScorer::FilteredTermIntersection {
                            terms: term_scorers,
                            filter,
                        }
                    } else {
                        SpecializedScorer::FilteredTermUnion {
                            terms: term_scorers,
                            filter,
                        }
                    }
                } else {
                    let filter: Box<dyn Scorer> =
                        if term_scorers.is_empty() && other_scorers.is_empty() {
                            if combined_all_scorer_count > 0 {
                                Box::new(AllScorer::new(reader.max_doc()))
                            } else {
                                Box::new(EmptyScorer)
                            }
                        } else {
                            let mut all_scorers: Vec<Box<dyn Scorer>> = term_scorers
                                .into_iter()
                                .map(|s| Box::new(s) as Box<dyn Scorer>)
                                .collect();
                            all_scorers.extend(other_scorers);
                            effective_must_scorer(
                                all_scorers,
                                combined_all_scorer_count,
                                reader.max_doc(),
                                num_docs,
                                self.scoring_enabled,
                                reader.bitmap_postings_enabled,
                            )
                            .unwrap_or_else(|| Box::new(EmptyScorer))
                        };
                    SpecializedScorer::Other(filter)
                }
            }
            (ShouldScorersCombinationMethod::Optional(should_scorer), must_scorers) => {
                // Optional SHOULD: contributes to scoring but not required for matching.
                match effective_must_scorer(
                    must_scorers,
                    must_special_scorer_counts.num_all_scorers,
                    reader.max_doc(),
                    num_docs,
                    self.scoring_enabled,
                    reader.bitmap_postings_enabled,
                ) {
                    None => {
                        // No MUST constraint: promote SHOULD to required.
                        // Must preserve any removed AllScorers from SHOULD via union.
                        effective_should_scorer_for_union(
                            should_scorer,
                            should_special_scorer_counts.num_all_scorers,
                            reader.max_doc(),
                            num_docs,
                            &score_combiner_fn,
                            self.scoring_enabled,
                            reader.bitmap_postings_enabled,
                        )
                    }
                    Some(must_scorer) => {
                        // Has MUST constraint: SHOULD only affects scoring.
                        if self.scoring_enabled {
                            SpecializedScorer::Other(Box::new(RequiredOptionalScorer::<
                                _,
                                _,
                                TScoreCombiner,
                            >::new(
                                must_scorer,
                                into_box_scorer(
                                    should_scorer,
                                    &score_combiner_fn,
                                    num_docs,
                                    reader.bitmap_postings_enabled,
                                ),
                            )))
                        } else {
                            SpecializedScorer::Other(must_scorer)
                        }
                    }
                }
            }
            (ShouldScorersCombinationMethod::Required(should_scorer), must_scorers) => {
                // Required SHOULD: at least `minimum_number_should_match` must match.
                // Semantics: (MUST constraint) AND (SHOULD constraint)
                match effective_must_scorer(
                    must_scorers,
                    must_special_scorer_counts.num_all_scorers,
                    reader.max_doc(),
                    num_docs,
                    self.scoring_enabled,
                    reader.bitmap_postings_enabled,
                ) {
                    None => {
                        // No MUST constraint: SHOULD alone determines matching.
                        should_scorer
                    }
                    Some(must_scorer) => match should_scorer {
                        SpecializedScorer::TermUnion(terms) => {
                            SpecializedScorer::FilteredTermUnion {
                                terms,
                                filter: must_scorer,
                            }
                        }
                        SpecializedScorer::TermIntersection(terms) => {
                            SpecializedScorer::FilteredTermIntersection {
                                terms,
                                filter: must_scorer,
                            }
                        }
                        _ => {
                            let should_boxed = into_box_scorer(
                                should_scorer,
                                &score_combiner_fn,
                                num_docs,
                                reader.bitmap_postings_enabled,
                            );
                            SpecializedScorer::Other(intersect_scorers(
                                vec![must_scorer, should_boxed],
                                num_docs,
                                self.scoring_enabled,
                                reader.bitmap_postings_enabled,
                            ))
                        }
                    },
                }
            }
        };
        if exclude_scorers.is_empty() {
            return Ok(include_scorer);
        }

        let include_scorer_boxed = into_box_scorer(
            include_scorer,
            &score_combiner_fn,
            num_docs,
            reader.bitmap_postings_enabled,
        );
        if reader.bitmap_postings_enabled
            && !self.scoring_enabled
            && include_scorer_boxed.has_fast_bitset()
        {
            let mut children = vec![include_scorer_boxed];
            children.extend(exclude_scorers);
            return Ok(SpecializedScorer::Other(Box::new(BitmapCombination::new(
                children,
                BitmapOperation::Exclude,
                num_docs,
            ))));
        }
        let scorer: Box<dyn Scorer> = if exclude_scorers.len() == 1 {
            let exclude_scorer = exclude_scorers.pop().unwrap();
            match exclude_scorer.downcast::<TermScorer>() {
                // Cast to TermScorer succeeded
                Ok(exclude_scorer) => Box::new(Exclude::new(include_scorer_boxed, *exclude_scorer)),
                // We get back the original Box<dyn Scorer>
                Err(exclude_scorer) => Box::new(Exclude::new(include_scorer_boxed, exclude_scorer)),
            }
        } else {
            Box::new(Exclude::new(include_scorer_boxed, exclude_scorers))
        };
        Ok(SpecializedScorer::Other(scorer))
    }
}

#[derive(Default, Copy, Clone, Debug)]
struct AllAndEmptyScorerCounts {
    num_all_scorers: usize,
    num_empty_scorers: usize,
}

fn remove_and_count_all_and_empty_scorers(
    scorers: &mut Vec<Box<dyn Scorer>>,
) -> AllAndEmptyScorerCounts {
    let mut counts = AllAndEmptyScorerCounts::default();
    scorers.retain(|scorer| {
        if scorer.is::<AllScorer>() {
            counts.num_all_scorers += 1;
            false
        } else if scorer.is::<EmptyScorer>() {
            counts.num_empty_scorers += 1;
            false
        } else {
            true
        }
    });
    counts
}

impl<TScoreCombiner: ScoreCombiner + Sync> Weight for BooleanWeight<TScoreCombiner> {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let num_docs = reader.num_docs();
        if self.weights.is_empty() {
            Ok(Box::new(EmptyScorer))
        } else if self.weights.len() == 1 {
            let &(occur, ref weight) = &self.weights[0];
            if occur == Occur::MustNot {
                Ok(Box::new(EmptyScorer))
            } else {
                weight.scorer(reader, boost)
            }
        } else if self.scoring_enabled {
            self.complex_scorer(reader, boost, &self.score_combiner_fn)
                .map(|specialized_scorer| {
                    into_box_scorer(
                        specialized_scorer,
                        &self.score_combiner_fn,
                        num_docs,
                        reader.bitmap_postings_enabled,
                    )
                })
        } else {
            self.complex_scorer(reader, boost, DoNothingCombiner::default)
                .map(|specialized_scorer| {
                    into_box_scorer(
                        specialized_scorer,
                        DoNothingCombiner::default,
                        num_docs,
                        reader.bitmap_postings_enabled,
                    )
                })
        }
    }

    fn pruning_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
        init_threshold: Score,
    ) -> crate::Result<Box<dyn crate::query::scorer::PruningScorer>> {
        let scorer = self.complex_scorer(reader, boost, &self.score_combiner_fn)?;
        match scorer {
            // Block-WAND scores by summing the matching terms, so it may only
            // drive a combiner that sums. Anything else (dis_max) still needs
            // every matching term and falls back to the plain union.
            SpecializedScorer::TermUnion(scorers) if !TScoreCombiner::SUPPORTS_BLOCK_WAND => {
                let union_scorer =
                    BufferedUnionScorer::build(scorers, &self.score_combiner_fn, reader.num_docs());
                Ok(Box::new(BasicPruningScorer::new(
                    Box::new(union_scorer),
                    init_threshold,
                )))
            }
            SpecializedScorer::TermUnion(mut scorers) => {
                // Drop already-exhausted scorers so a lone survivor uses the
                // (~3x faster) single-scorer specialization
                scorers.retain(|scorer| scorer.doc() < TERMINATED);
                match scorers.len() {
                    0 => Ok(Box::new(EmptyScorer)),
                    1 => Ok(Box::new(BlockWandSingleScorer::new(
                        scorers.pop().unwrap(),
                        init_threshold,
                    ))),
                    _ => Ok(Box::new(BlockWandUnionScorer::new(scorers, init_threshold))),
                }
            }
            SpecializedScorer::TermIntersection(scorers) => Ok(Box::new(
                BlockWandIntersectionScorer::new(scorers, init_threshold),
            )),
            SpecializedScorer::FilteredTermUnion { terms, filter } => {
                if !TScoreCombiner::SUPPORTS_BLOCK_WAND
                    || !self.scoring_enabled
                    || filter.constant_score().is_none()
                {
                    let union_scorer = into_box_scorer(
                        SpecializedScorer::FilteredTermUnion { terms, filter },
                        &self.score_combiner_fn,
                        reader.num_docs(),
                        reader.bitmap_postings_enabled,
                    );
                    return Ok(Box::new(BasicPruningScorer::new(
                        union_scorer,
                        init_threshold,
                    )));
                }
                let filter_boost = filter.constant_score().unwrap();
                let mut terms = terms;
                terms.retain(|scorer| scorer.doc() < TERMINATED);
                match terms.len() {
                    0 => Ok(Box::new(EmptyScorer)),
                    1 => Ok(Box::new(BlockWandSingleScorer::with_filter(
                        terms.pop().unwrap(),
                        init_threshold,
                        filter,
                        filter_boost,
                    ))),
                    _ => Ok(Box::new(BlockWandUnionScorer::with_filter(
                        terms,
                        init_threshold,
                        filter,
                        filter_boost,
                    ))),
                }
            }
            SpecializedScorer::FilteredTermIntersection { terms, filter } => {
                if !self.scoring_enabled || filter.constant_score().is_none() {
                    let intersection_scorer = into_box_scorer(
                        SpecializedScorer::FilteredTermIntersection { terms, filter },
                        &self.score_combiner_fn,
                        reader.num_docs(),
                        reader.bitmap_postings_enabled,
                    );
                    return Ok(Box::new(BasicPruningScorer::new(
                        intersection_scorer,
                        init_threshold,
                    )));
                }
                if terms.iter().any(|s| s.doc() == TERMINATED) {
                    return Ok(Box::new(EmptyScorer));
                }
                let filter_boost = filter.constant_score().unwrap();
                match terms.len() {
                    0 => Ok(Box::new(EmptyScorer)),
                    1 => {
                        let mut terms = terms;
                        Ok(Box::new(BlockWandSingleScorer::with_filter(
                            terms.pop().unwrap(),
                            init_threshold,
                            filter,
                            filter_boost,
                        )))
                    }
                    _ => Ok(Box::new(BlockWandIntersectionScorer::with_filter(
                        terms,
                        init_threshold,
                        filter,
                        filter_boost,
                    ))),
                }
            }
            SpecializedScorer::Other(scorer) => {
                Ok(Box::new(BasicPruningScorer::new(scorer, init_threshold)))
            }
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        if !self.scoring_enabled {
            return Ok(Explanation::new("BooleanQuery with no scoring", 1.0));
        }

        let mut explanation = Explanation::new("BooleanClause. sum of ...", scorer.score());
        for (occur, subweight) in &self.weights {
            if is_include_occur(*occur) {
                if let Ok(child_explanation) = subweight.explain(reader, doc) {
                    explanation.add_detail(child_explanation);
                }
            }
        }
        Ok(explanation)
    }

    fn for_each(
        &self,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score),
    ) -> crate::Result<()> {
        let scorer = self.complex_scorer(reader, 1.0, &self.score_combiner_fn)?;
        let num_docs = reader.num_docs();
        match scorer {
            SpecializedScorer::TermUnion(mut term_scorers) => {
                if term_scorers.len() == 1 {
                    let mut term_scorer = term_scorers.pop().unwrap();
                    for_each_scorer(&mut term_scorer, callback);
                } else {
                    let mut union_scorer =
                        BufferedUnionScorer::build(term_scorers, &self.score_combiner_fn, num_docs);
                    for_each_scorer(&mut union_scorer, callback);
                }
            }
            SpecializedScorer::TermIntersection(term_scorers) => {
                let boxed_scorers: Vec<Box<dyn Scorer>> = term_scorers
                    .into_iter()
                    .map(|term_scorer| Box::new(term_scorer) as Box<dyn Scorer>)
                    .collect();
                let mut intersection = intersect_scorers(
                    boxed_scorers,
                    num_docs,
                    self.scoring_enabled,
                    reader.bitmap_postings_enabled,
                );
                for_each_scorer(intersection.as_mut(), callback);
            }
            SpecializedScorer::FilteredTermUnion { .. }
            | SpecializedScorer::FilteredTermIntersection { .. } => {
                let mut scorer = into_box_scorer(
                    scorer,
                    &self.score_combiner_fn,
                    num_docs,
                    reader.bitmap_postings_enabled,
                );
                for_each_scorer(scorer.as_mut(), callback);
            }
            SpecializedScorer::Other(mut scorer) => {
                for_each_scorer(scorer.as_mut(), callback);
            }
        }
        Ok(())
    }

    /// Calls `callback` with all of the `(doc, score)` for which score
    /// is exceeding a given threshold.
    ///
    /// This method is useful for the TopDocs collector.
    /// For all docsets, the blanket implementation has the benefit
    /// of prefiltering (doc, score) pairs, avoiding the
    /// virtual dispatch cost.
    ///
    /// More importantly, it makes it possible for scorers to implement
    /// important optimization (e.g. BlockWAND).
    ///
    /// Overrides the blanket implementation to drive the concrete pruning scorer
    /// directly, rather than through a `Box<dyn PruningScorer>`. Monomorphizing
    /// `for_each_pruning_scorer` over the concrete scorer type keeps
    /// `advance`/`score`/`set_threshold` statically dispatched (and inlinable) in
    /// the hot loop, so the callback is the only dynamic call
    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> crate::Result<()> {
        let scorer = self.complex_scorer(reader, 1.0, &self.score_combiner_fn)?;
        match scorer {
            // Block-WAND scores by summing the matching terms, so it may only
            // drive a combiner that sums. Anything else (dis_max) still needs
            // every matching term and falls back to the plain union.
            SpecializedScorer::TermUnion(scorers) if !TScoreCombiner::SUPPORTS_BLOCK_WAND => {
                let union_scorer =
                    BufferedUnionScorer::build(scorers, &self.score_combiner_fn, reader.num_docs());
                let mut scorer = BasicPruningScorer::new(Box::new(union_scorer), threshold);
                for_each_pruning_scorer(&mut scorer, callback);
            }
            SpecializedScorer::TermUnion(mut scorers) => {
                scorers.retain(|scorer| scorer.doc() < TERMINATED);
                match scorers.len() {
                    0 => {}
                    1 => {
                        let mut scorer =
                            BlockWandSingleScorer::new(scorers.pop().unwrap(), threshold);
                        for_each_pruning_scorer(&mut scorer, callback);
                    }
                    _ if self.should_use_block_maxscore(&scorers, reader.max_doc()) => {
                        super::block_maxscore::block_maxscore(
                            scorers,
                            threshold,
                            super::block_maxscore::MIN_BOUND_WINDOW,
                            callback,
                        );
                    }
                    _ => {
                        let mut scorer = BlockWandUnionScorer::new(scorers, threshold);
                        for_each_pruning_scorer(&mut scorer, callback);
                    }
                }
            }
            SpecializedScorer::TermIntersection(scorers) => {
                let mut scorer = BlockWandIntersectionScorer::new(scorers, threshold);
                for_each_pruning_scorer(&mut scorer, callback);
            }
            SpecializedScorer::FilteredTermUnion { terms, mut filter } => {
                if !TScoreCombiner::SUPPORTS_BLOCK_WAND
                    || !self.scoring_enabled
                    || filter.constant_score().is_none()
                {
                    let union_scorer = into_box_scorer(
                        SpecializedScorer::FilteredTermUnion { terms, filter },
                        &self.score_combiner_fn,
                        reader.num_docs(),
                        reader.bitmap_postings_enabled,
                    );
                    let mut scorer = BasicPruningScorer::new(union_scorer, threshold);
                    for_each_pruning_scorer(&mut scorer, callback);
                    return Ok(());
                }
                let filter_boost = filter.constant_score().unwrap();
                let mut terms = terms;
                terms.retain(|scorer| scorer.doc() < TERMINATED);
                match terms.len() {
                    0 => {}
                    1 => {
                        let mut scorer = BlockWandSingleScorer::with_filter(
                            terms.pop().unwrap(),
                            threshold,
                            filter,
                            filter_boost,
                        );
                        for_each_pruning_scorer(&mut scorer, callback);
                    }
                    _ if self.should_use_block_maxscore(&terms, reader.max_doc()) => {
                        super::block_maxscore::block_maxscore_filtered(
                            terms,
                            threshold,
                            Some(&mut *filter),
                            filter_boost,
                            super::block_maxscore::MIN_BOUND_WINDOW,
                            callback,
                        );
                    }
                    _ => {
                        let mut scorer = BlockWandUnionScorer::with_filter(
                            terms,
                            threshold,
                            filter,
                            filter_boost,
                        );
                        for_each_pruning_scorer(&mut scorer, callback);
                    }
                }
            }
            SpecializedScorer::FilteredTermIntersection { terms, filter } => {
                if !self.scoring_enabled || filter.constant_score().is_none() {
                    let intersection_scorer = into_box_scorer(
                        SpecializedScorer::FilteredTermIntersection { terms, filter },
                        &self.score_combiner_fn,
                        reader.num_docs(),
                        reader.bitmap_postings_enabled,
                    );
                    let mut scorer = BasicPruningScorer::new(intersection_scorer, threshold);
                    for_each_pruning_scorer(&mut scorer, callback);
                    return Ok(());
                }
                if terms.iter().any(|s| s.doc() == TERMINATED) {
                    return Ok(());
                }
                let filter_boost = filter.constant_score().unwrap();
                match terms.len() {
                    0 => {}
                    1 => {
                        let mut terms = terms;
                        let mut scorer = BlockWandSingleScorer::with_filter(
                            terms.pop().unwrap(),
                            threshold,
                            filter,
                            filter_boost,
                        );
                        for_each_pruning_scorer(&mut scorer, callback);
                    }
                    _ => {
                        let mut scorer = BlockWandIntersectionScorer::with_filter(
                            terms,
                            threshold,
                            filter,
                            filter_boost,
                        );
                        for_each_pruning_scorer(&mut scorer, callback);
                    }
                }
            }
            SpecializedScorer::Other(scorer) => {
                let mut scorer = BasicPruningScorer::new(scorer, threshold);
                for_each_pruning_scorer(&mut scorer, callback);
            }
        }
        Ok(())
    }

    fn for_each_no_score(
        &self,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(&[DocId]),
    ) -> crate::Result<()> {
        let scorer = self.complex_scorer(reader, 1.0, DoNothingCombiner::default)?;
        let num_docs = reader.num_docs();
        let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];

        match scorer {
            SpecializedScorer::TermUnion(mut term_scorers) => {
                if term_scorers.len() == 1 {
                    let mut term_scorer = term_scorers.pop().unwrap();
                    for_each_docset_buffered(&mut term_scorer, &mut buffer, callback);
                } else {
                    let mut union_scorer =
                        BufferedUnionScorer::build(term_scorers, &self.score_combiner_fn, num_docs);
                    for_each_docset_buffered(&mut union_scorer, &mut buffer, callback);
                }
            }
            SpecializedScorer::TermIntersection(term_scorers) => {
                let boxed_scorers: Vec<Box<dyn Scorer>> = term_scorers
                    .into_iter()
                    .map(|term_scorer| Box::new(term_scorer) as Box<dyn Scorer>)
                    .collect();
                let mut intersection = intersect_scorers(
                    boxed_scorers,
                    num_docs,
                    self.scoring_enabled,
                    reader.bitmap_postings_enabled,
                );
                for_each_docset_buffered(intersection.as_mut(), &mut buffer, callback);
            }
            SpecializedScorer::FilteredTermUnion { .. }
            | SpecializedScorer::FilteredTermIntersection { .. } => {
                let mut scorer = into_box_scorer(
                    scorer,
                    DoNothingCombiner::default,
                    num_docs,
                    reader.bitmap_postings_enabled,
                );
                for_each_docset_buffered(scorer.as_mut(), &mut buffer, callback);
            }
            SpecializedScorer::Other(mut scorer) => {
                for_each_docset_buffered(scorer.as_mut(), &mut buffer, callback);
            }
        }
        Ok(())
    }

    fn for_each_no_score_batch(
        &self,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(crate::DocSetBatch<'_>),
    ) -> crate::Result<()> {
        let scorer = self.complex_scorer(reader, 1.0, DoNothingCombiner::default)?;
        let num_docs = reader.num_docs();

        match scorer {
            SpecializedScorer::TermUnion(mut term_scorers) => {
                if term_scorers.len() == 1 {
                    let mut term_scorer = term_scorers.pop().unwrap();
                    crate::query::weight::for_each_docset_batch(&mut term_scorer, callback);
                } else {
                    let mut union_scorer =
                        BufferedUnionScorer::build(term_scorers, &self.score_combiner_fn, num_docs);
                    crate::query::weight::for_each_docset_batch(&mut union_scorer, callback);
                }
            }
            SpecializedScorer::TermIntersection(term_scorers) => {
                let boxed_scorers: Vec<Box<dyn Scorer>> = term_scorers
                    .into_iter()
                    .map(|term_scorer| Box::new(term_scorer) as Box<dyn Scorer>)
                    .collect();
                let mut intersection = intersect_scorers(
                    boxed_scorers,
                    num_docs,
                    self.scoring_enabled,
                    reader.bitmap_postings_enabled,
                );
                crate::query::weight::for_each_docset_batch(intersection.as_mut(), callback);
            }
            SpecializedScorer::FilteredTermUnion { .. }
            | SpecializedScorer::FilteredTermIntersection { .. } => {
                let mut scorer = into_box_scorer(
                    scorer,
                    DoNothingCombiner::default,
                    num_docs,
                    reader.bitmap_postings_enabled,
                );
                crate::query::weight::for_each_docset_batch(scorer.as_mut(), callback);
            }
            SpecializedScorer::Other(mut scorer) => {
                crate::query::weight::for_each_docset_batch(scorer.as_mut(), callback);
            }
        }
        Ok(())
    }
}

fn is_include_occur(occur: Occur) -> bool {
    match occur {
        Occur::Must | Occur::Should => true,
        Occur::MustNot => false,
    }
}

#[cfg(test)]
mod tests {
    use super::BooleanWeight;
    use crate::query::{Bm25Weight, DisjunctionPruning, SumCombiner, TermScorer};
    use crate::Bm25Params;

    #[test]
    fn bitmap_phrase_intersections_use_selective_candidates() -> crate::Result<()> {
        use crate::query::bitmap_combination::BitmapCombination;
        use crate::query::{EnableScoring, QueryParser};
        use crate::schema::{Schema, FAST, TEXT};
        use crate::{Index, TERMINATED};

        let mut schema = Schema::builder();
        let text = schema.add_text_field(
            "text",
            TEXT.set_indexing_options(
                TEXT.get_indexing_options()
                    .unwrap()
                    .clone()
                    .set_bitmap_postings(true),
            ),
        );
        let number = schema.add_u64_field("number", FAST);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        for doc in 0..4096u32 {
            let mut terms = vec![if doc % 3 == 0 { "of the" } else { "of gap the" }];
            for (term, frequency) in [
                ("selective", 512),
                ("half", 2048),
                ("abovehalf", 2049),
                ("dense", 3072),
            ] {
                if doc < frequency {
                    terms.push(term);
                }
            }
            writer.add_document(doc!(text => terms.join(" "), number => u64::from(doc)))?;
        }
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let reader = searcher.segment_reader(0);
        let mut ordinary_reader = reader.clone();
        ordinary_reader.bitmap_postings_enabled = false;
        let parser = QueryParser::for_index(&index, vec![text]);
        for (expression, bitmap) in [
            (r#"selective AND "of the""#, false),
            (r#""of the" AND selective"#, false),
            (r#"half AND "of the""#, false),
            (r#"abovehalf AND "of the""#, true),
            (r#"dense AND "of the""#, true),
            (r#"dense AND "of the" AND selective"#, false),
            (r#"selective AND (dense OR "of the")"#, true),
            (r#"selective OR "of the""#, true),
            ("selective AND dense", true),
            ("dense AND number:[0 TO 511]", true),
        ] {
            let query = parser.parse_query(expression)?;
            let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
            let mut scorer = weight.scorer(reader, 1.0)?;
            assert_eq!(scorer.is::<BitmapCombination>(), bitmap, "{expression}");
            let mut expected_scorer = weight.scorer(&ordinary_reader, 1.0)?;
            let mut expected = Vec::new();
            while expected_scorer.doc() != TERMINATED {
                expected.push(expected_scorer.doc());
                expected_scorer.advance();
            }
            let mut actual = Vec::new();
            crate::query::for_each_docset_batch(scorer.as_mut(), &mut |batch| {
                batch.for_each_doc_block(|docs| actual.extend_from_slice(docs));
            });
            assert_eq!(actual, expected, "{expression}");
            assert!(!actual.is_empty(), "{expression}");
            let scored = query
                .weight(EnableScoring::enabled_from_searcher(&searcher))?
                .scorer(reader, 1.0)?;
            assert!(!scored.is::<BitmapCombination>(), "{expression}");
        }
        Ok(())
    }

    #[test]
    fn test_shared_conjunction_norms_match_exhaustive() -> crate::Result<()> {
        use crate::query::{BoostQuery, EnableScoring, Query, QueryParser};
        use crate::schema::{Schema, TEXT};
        use crate::{Index, Score, TERMINATED};

        for pnorms in [false, true] {
            let mut schema = Schema::builder();
            let options = TEXT.set_indexing_options(
                TEXT.get_indexing_options()
                    .unwrap()
                    .clone()
                    .set_pnorms(pnorms),
            );
            let field = schema.add_text_field("text", options.clone());
            let other = schema.add_text_field("other", options);
            let index = Index::create_in_ram(schema.build());
            let mut writer = index.writer_for_tests()?;
            writer.set_merge_policy(Box::new(crate::merge_policy::NoMergePolicy));
            let mut seed = 71u32;
            for ordinal in 0..10000 {
                let len = if ordinal < 32 { 4 } else { 1 + ordinal % 160 };
                let mut text = String::new();
                for _ in 0..len {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    text.push_str(["a ", "b ", "c ", "x ", "1 ", "2 "][(seed >> 24) as usize % 6]);
                }
                writer.add_document(
                    doc!(field => text, other => if ordinal % 3 == 0 { "a" } else { "x a b" }),
                )?;
            }
            writer.commit()?;
            drop(writer);
            let searcher = index.reader()?.searcher();
            let parser = QueryParser::for_index(&index, vec![field]);
            let reader = searcher.segment_reader(0);
            let term_query = parser.parse_query("a")?;
            let scored = term_query
                .weight(EnableScoring::enabled_from_searcher(&searcher))?
                .scorer(reader, 1.0)?;
            let unscored = term_query
                .weight(EnableScoring::disabled_from_searcher(&searcher))?
                .scorer(reader, 1.0)?;
            let other_field = parser
                .parse_query("other:a")?
                .weight(EnableScoring::enabled_from_searcher(&searcher))?
                .scorer(reader, 1.0)?;
            let scored = scored.downcast_ref::<TermScorer>().unwrap();
            assert!(scored.shares_fieldnorms_with(scored));
            assert!(!scored.shares_fieldnorms_with(unscored.downcast_ref::<TermScorer>().unwrap()));
            assert!(
                !scored.shares_fieldnorms_with(other_field.downcast_ref::<TermScorer>().unwrap())
            );
            let queries = [
                "a AND b",
                "a AND b AND c",
                "a AND other:b",
                "a AND other:b AND c",
                "a^2.5 AND b^0.5",
            ]
            .into_iter()
            .map(|expression| Ok((expression, parser.parse_query(expression)?)))
            .collect::<crate::Result<Vec<_>>>()?;
            for (expression, parsed) in queries {
                for boost in [0.0, 1.0, 2.5] {
                    let query = BoostQuery::new(parsed.box_clone(), boost);
                    for mode in [
                        DisjunctionPruning::Auto,
                        DisjunctionPruning::BlockMaxScore,
                        DisjunctionPruning::BlockWand,
                    ] {
                        let weight = query.weight(
                            EnableScoring::enabled_from_searcher(&searcher)
                                .with_disjunction_pruning(mode),
                        )?;
                        for reader in searcher.segment_readers() {
                            let mut baseline = weight.scorer(reader, 1.0)?;
                            let mut expected = Vec::new();
                            while baseline.doc() != TERMINATED {
                                expected.push((baseline.doc(), baseline.score()));
                                baseline.advance();
                            }
                            if boost == 1.0 && mode == DisjunctionPruning::Auto {
                                let disabled = query
                                    .weight(EnableScoring::disabled_from_searcher(&searcher))?;
                                let mut docs = Vec::new();
                                disabled.for_each_no_score(reader, &mut |block| {
                                    docs.extend_from_slice(block);
                                })?;
                                assert_eq!(docs, expected.iter().map(|v| v.0).collect::<Vec<_>>());
                            }
                            let all = expected
                                .iter()
                                .copied()
                                .collect::<std::collections::HashMap<_, _>>();
                            expected
                                .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                            for top_k in [1, 3, 10, 50] {
                                let mut actual: Vec<(u32, Score)> = Vec::new();
                                weight.for_each_pruning(
                                    Score::MIN,
                                    reader,
                                    &mut |doc, score| {
                                        assert!((score - all[&doc]).abs() <= 1e-5);
                                        actual.push((doc, score));
                                        actual.sort_unstable_by(|a, b| {
                                            b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
                                        });
                                        actual.truncate(top_k);
                                        if actual.len() == top_k {
                                            actual.last().unwrap().1
                                        } else {
                                            Score::MIN
                                        }
                                    },
                                )?;
                                assert_eq!(actual.len(), expected.len().min(top_k));
                                for (actual, expected) in actual.iter().zip(&expected) {
                                    assert!(
                                        (actual.1 - expected.1).abs() <= 1e-5,
                                        "{expression}, boost={boost}, {mode:?}, pnorms={pnorms}, \
                                         k={top_k}: {actual:?} != {expected:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_disjunction_pruning_overrides_cutoffs() {
        for (term_count, doc_freq, max_doc, auto_maxscore) in [
            (2, 1, 1_024, false),
            (3, 128, 1_024, true),
            (3, 128, 1_048_576, false),
        ] {
            let docs: Vec<_> = (0..doc_freq).map(|doc| (doc, 1)).collect();
            let norms = vec![1; doc_freq as usize];
            let scorers: Vec<_> = (0..term_count)
                .map(|_| {
                    TermScorer::create_for_test(
                        &docs,
                        &norms,
                        Bm25Weight::for_one_term(
                            doc_freq as u64,
                            max_doc as u64,
                            1.0,
                            Bm25Params::default(),
                        ),
                    )
                })
                .collect();
            for weight in [
                BooleanWeight::new(Vec::new(), true, Box::new(SumCombiner::default)),
                BooleanWeight::with_minimum_number_should_match(
                    Vec::new(),
                    1,
                    true,
                    Box::new(SumCombiner::default),
                ),
            ] {
                assert_eq!(
                    weight.should_use_block_maxscore(&scorers, max_doc),
                    auto_maxscore
                );
                let weight = weight.with_disjunction_pruning(DisjunctionPruning::BlockWand);
                assert!(!weight.should_use_block_maxscore(&scorers, max_doc));
                let weight = weight.with_disjunction_pruning(DisjunctionPruning::BlockMaxScore);
                assert!(weight.should_use_block_maxscore(&scorers, max_doc));
            }
        }
    }

    #[test]
    fn test_specialized_scorer_unboxed_filters() -> crate::Result<()> {
        use super::SpecializedScorer;
        use crate::query::boolean_query::BlockWandUnionScorer;
        use crate::query::scorer::BasicPruningScorer;
        use crate::query::term_query::TermScorer;
        use crate::query::{
            ConstScoreQuery, EnableScoring, Occur, Query, SumCombiner, TermQuery, Weight,
        };
        use crate::schema::{IndexRecordOption, Schema, TEXT};
        use crate::{Index, Term};

        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer_for_tests()?;
        writer.add_document(doc!(field => "a b c"))?;
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        let reader = searcher.segment_reader(0);

        let term_a = TermQuery::new(
            Term::from_field_text(field, "a"),
            IndexRecordOption::WithFreqsAndPositions,
        );
        let term_b = TermQuery::new(
            Term::from_field_text(field, "b"),
            IndexRecordOption::WithFreqsAndPositions,
        );
        let term_c = TermQuery::new(
            Term::from_field_text(field, "c"),
            IndexRecordOption::WithFreqsAndPositions,
        );
        let weight_a = term_a.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let weight_b = term_b.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let weight_c = term_c.weight(EnableScoring::enabled_from_searcher(&searcher))?;

        // Case 1: Should "a", "b" with Must "c" -> FilteredTermUnion with filter containing
        // TermScorer
        let bool_weight = BooleanWeight::new(
            vec![
                (Occur::Should, weight_a),
                (Occur::Should, weight_b),
                (Occur::Must, weight_c),
            ],
            true,
            Box::new(SumCombiner::default),
        );
        let specialized =
            bool_weight.complex_scorer(reader, 1.0, &bool_weight.score_combiner_fn)?;
        match specialized {
            SpecializedScorer::FilteredTermUnion { terms, filter } => {
                assert_eq!(terms.len(), 2);
                assert!(filter.is::<TermScorer>());
            }
            _ => panic!("Expected FilteredTermUnion with TermScorer filter"),
        }
        // Dynamic-score filter falls back to non-BMW pruning in pruning_scorer:
        let scorer = bool_weight.pruning_scorer(reader, 1.0, 0.0)?;
        assert!(scorer.is::<BasicPruningScorer>());
        let mut count = 0;
        bool_weight.for_each_pruning(0.0, reader, &mut |_doc, _score| {
            count += 1;
            0.0
        })?;
        assert_eq!(count, 1);

        // Case 2: Must "a", "b", "c" -> SpecializedScorer::TermIntersection (3 terms)
        let weight_a = term_a.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let weight_b = term_b.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let weight_c = term_c.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let bool_weight = BooleanWeight::with_minimum_number_should_match(
            vec![
                (Occur::Must, weight_a),
                (Occur::Must, weight_b),
                (Occur::Must, weight_c),
            ],
            0,
            true,
            Box::new(SumCombiner::default),
        );
        let specialized =
            bool_weight.complex_scorer(reader, 1.0, &bool_weight.score_combiner_fn)?;
        match specialized {
            SpecializedScorer::TermIntersection(terms) => {
                assert_eq!(terms.len(), 3);
            }
            _ => panic!("Expected TermIntersection"),
        }

        // Case 3: Must "a", "b" with Must basic (non-freq) "c" -> FilteredTermIntersection with
        // filter containing TermScorer
        let term_basic_c =
            TermQuery::new(Term::from_field_text(field, "c"), IndexRecordOption::Basic);
        let weight_a = term_a.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let weight_b = term_b.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let weight_basic_c =
            term_basic_c.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let bool_weight = BooleanWeight::with_minimum_number_should_match(
            vec![
                (Occur::Must, weight_a),
                (Occur::Must, weight_b),
                (Occur::Must, weight_basic_c),
            ],
            0,
            true,
            Box::new(SumCombiner::default),
        );
        let specialized =
            bool_weight.complex_scorer(reader, 1.0, &bool_weight.score_combiner_fn)?;
        match specialized {
            SpecializedScorer::FilteredTermIntersection { terms, filter } => {
                assert_eq!(terms.len(), 2);
                assert!(filter.is::<TermScorer>());
            }
            _ => panic!("Expected FilteredTermIntersection with TermScorer filter"),
        }
        let scorer = bool_weight.pruning_scorer(reader, 1.0, 0.0)?;
        assert!(scorer.is::<BasicPruningScorer>());
        let mut count = 0;
        bool_weight.for_each_pruning(0.0, reader, &mut |_doc, _score| {
            count += 1;
            0.0
        })?;
        assert_eq!(count, 1);

        // Case 4: Filter with constant score enables dynamic BlockWandUnionScorer
        let weight_a = term_a.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let weight_b = term_b.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let const_c = ConstScoreQuery::new(term_c, 1.0);
        let weight_const_c = const_c.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let bool_weight = BooleanWeight::new(
            vec![
                (Occur::Should, weight_a),
                (Occur::Should, weight_b),
                (Occur::Must, weight_const_c),
            ],
            true,
            Box::new(SumCombiner::default),
        );
        let scorer = bool_weight.pruning_scorer(reader, 1.0, 0.0)?;
        assert!(scorer.is::<BlockWandUnionScorer>());
        let mut count = 0;
        bool_weight.for_each_pruning(0.0, reader, &mut |_doc, _score| {
            count += 1;
            0.0
        })?;
        assert_eq!(count, 1);

        Ok(())
    }
}
