use crate::docset::{DocSet, DriveCost, COLLECT_BLOCK_BUFFER_LEN};
use crate::index::SegmentReader;
use crate::postings::FreqReadingOption;
use crate::query::boolean_query::{
    BlockWandIntersectionScorer, BlockWandSingleScorer, BlockWandUnionScorer,
};
use crate::query::disjunction::Disjunction;
use crate::query::exclude::BatchExcludeScorer;
use crate::query::explanation::does_not_match;
use crate::query::intersection::{intersect_candidate_scorers, intersect_scorers_unpositioned};
use crate::query::score_combiner::{DoNothingCombiner, ScoreCombiner};
use crate::query::scorer::BasicPruningScorer;
use crate::query::term_query::TermScorer;
use crate::query::weight::{for_each_docset_buffered, for_each_pruning_scorer, for_each_scorer};
use crate::query::{
    AllScorer, BufferedUnionScorer, CandidateHint, DisjunctionPruning, EmptyScorer, Exclude,
    ExcludedDocSet, ExclusionSet, Explanation, Occur, RequiredOptionalScorer, Scorer,
    UnpositionedScorer, Weight,
};
use crate::{DocId, Score, TERMINATED};

pub(crate) enum SpecializedScorer {
    TermUnion(Vec<TermScorer>),
    TermIntersection(Vec<TermScorer>),
    FilteredTermUnion {
        terms: Vec<TermScorer>,
        filter: UnpositionedScorer,
    },
    FilteredTermIntersection {
        terms: Vec<TermScorer>,
        filter: UnpositionedScorer,
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
) -> SpecializedScorer
where
    TScoreCombiner: ScoreCombiner,
{
    assert!(!scorers.is_empty());
    if scorers.len() == 1 && !scorers[0].is::<TermScorer>() {
        return SpecializedScorer::Other(scorers.into_iter().next().unwrap()); //< we checked the size beforehand
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

fn is_freq_term_scorer(scorer: &dyn Scorer) -> bool {
    scorer
        .downcast_ref::<TermScorer>()
        .is_some_and(|scorer| scorer.freq_reading_option() == FreqReadingOption::ReadFreq)
}

fn is_freq_term_unpositioned(scorer: &UnpositionedScorer) -> bool {
    scorer
        .downcast_ref_positioned::<TermScorer>()
        .is_some_and(|scorer| scorer.freq_reading_option() == FreqReadingOption::ReadFreq)
}

fn into_box_scorer<TScoreCombiner: ScoreCombiner>(
    scorer: SpecializedScorer,
    score_combiner_fn: impl Fn() -> TScoreCombiner,
    num_docs: u32,
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
            let boxed_scorers: Vec<UnpositionedScorer> = term_scorers
                .into_iter()
                .map(|s| UnpositionedScorer::from_positioned(Box::new(s) as Box<dyn Scorer>))
                .collect();
            intersect_candidate_scorers(boxed_scorers, num_docs)
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
            intersect_candidate_scorers(
                vec![UnpositionedScorer::from_positioned(term_scorer), filter],
                num_docs,
            )
        }
        SpecializedScorer::FilteredTermIntersection { terms, filter } => {
            let mut boxed_scorers: Vec<UnpositionedScorer> = terms
                .into_iter()
                .map(|s| UnpositionedScorer::from_positioned(Box::new(s) as Box<dyn Scorer>))
                .collect();
            boxed_scorers.push(filter);
            intersect_candidate_scorers(boxed_scorers, num_docs)
        }
        SpecializedScorer::Other(scorer) => scorer,
    }
}

/// Returns the effective MUST scorer, accounting for removed AllScorers.
///
/// When AllScorer instances are removed from must_scorers as an optimization,
/// we must restore the "match all" semantics if the list becomes empty.
fn effective_must_scorer(
    must_scorers: Vec<UnpositionedScorer>,
    removed_all_scorer_count: usize,
    max_doc: DocId,
    num_docs: u32,
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
        Some(intersect_candidate_scorers(must_scorers, num_docs))
    }
}

fn effective_must_scorer_unpositioned(
    must_scorers: Vec<UnpositionedScorer>,
    removed_all_scorer_count: usize,
    max_doc: DocId,
    num_docs: u32,
) -> Option<UnpositionedScorer> {
    if must_scorers.is_empty() {
        if removed_all_scorer_count > 0 {
            Some(UnpositionedScorer::from_positioned(Box::new(
                AllScorer::new(max_doc),
            )))
        } else {
            None
        }
    } else {
        Some(intersect_scorers_unpositioned(must_scorers, num_docs))
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
) -> SpecializedScorer {
    if removed_all_scorer_count > 0 {
        if scoring_enabled {
            // Need to union to get score contributions from both
            let all_scorers: Vec<Box<dyn Scorer>> = vec![
                into_box_scorer(should_scorer, &score_combiner_fn, num_docs),
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ExecutionMode {
    Exhaustive,
    Pruning,
}

enum ShouldScorersCombinationMethod {
    // Should scorers are irrelevant.
    Ignored,
    // Only contributes to final score.
    Optional(SpecializedScorer),
    // Regardless of score, the should scorers may impact whether a document is matching or not.
    Required(SpecializedScorer),
}

#[derive(Default)]
struct PerOccurScorers {
    must: Vec<UnpositionedScorer>,
    should: Vec<Box<dyn Scorer>>,
    must_not: Vec<UnpositionedScorer>,
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

    /// Returns true when the non-deferred clauses will enter a filtered Block-WAND /
    /// Block-MaxScore scorer (`FilteredTermUnion` or `FilteredTermIntersection`) under
    /// `ExecutionMode::Pruning`, so deferred fast-field `Occur::Must` clauses should be
    /// constructed lazily with `CandidateHint::Pruned`.
    ///
    /// Queries without frequency-bearing WAND drivers (for example `AllQuery AND
    /// TermPredicateQuery`), with active `Occur::MustNot` exclusions, or with non-constant-score
    /// auxiliary filters return `false` so exhaustive postings/bitset strategies are preserved.
    fn can_wand_prune_deferred_musts(
        &self,
        per_occur_scorers: &PerOccurScorers,
        has_deferred_must_not: bool,
        execution_mode: ExecutionMode,
    ) -> bool {
        if execution_mode != ExecutionMode::Pruning
            || !self.scoring_enabled
            || has_deferred_must_not
        {
            return false;
        }
        let has_active_exclude = per_occur_scorers
            .must_not
            .iter()
            .any(|s| !s.is::<EmptyScorer>());
        if has_active_exclude {
            return false;
        }
        let musts = per_occur_scorers.must.as_slice();
        let non_empty_shoulds: Vec<&Box<dyn Scorer>> = per_occur_scorers
            .should
            .iter()
            .filter(|s| !s.is::<EmptyScorer>())
            .collect();

        if self.minimum_number_should_match == 1 && !non_empty_shoulds.is_empty() {
            TScoreCombiner::SUPPORTS_BLOCK_WAND
                && non_empty_shoulds
                    .iter()
                    .all(|s| is_freq_term_scorer(s.as_ref()))
                && musts.iter().all(|s| s.constant_score().is_some())
        } else if non_empty_shoulds.is_empty()
            || (self.minimum_number_should_match >= 1
                && self.minimum_number_should_match == non_empty_shoulds.len())
        {
            let mut freq_term_count = 0usize;
            for s in musts {
                if is_freq_term_unpositioned(s) {
                    freq_term_count += 1;
                } else if s.constant_score().is_none() {
                    return false;
                }
            }
            for s in &non_empty_shoulds {
                if is_freq_term_scorer(s.as_ref()) {
                    freq_term_count += 1;
                } else if s.constant_score().is_none() {
                    return false;
                }
            }
            freq_term_count >= 2 || (freq_term_count == 1 && TScoreCombiner::SUPPORTS_BLOCK_WAND)
        } else {
            false
        }
    }

    fn per_occur_scorers(
        &self,
        reader: &SegmentReader,
        boost: Score,
        execution_mode: ExecutionMode,
    ) -> crate::Result<PerOccurScorers> {
        let mut per_occur_scorers = PerOccurScorers::default();
        let mut deferred_fast_clauses: Vec<(Occur, &dyn Weight)> = Vec::new();

        for (occur, subweight) in &self.weights {
            if matches!(*occur, Occur::Must | Occur::MustNot) && subweight.uses_fast_fields() {
                deferred_fast_clauses.push((*occur, subweight.as_ref()));
                continue;
            }
            let sub_scorer: Box<dyn Scorer> = subweight.scorer(reader, boost)?;
            match *occur {
                Occur::Must => {
                    let is_empty = sub_scorer.is::<EmptyScorer>();
                    per_occur_scorers
                        .must
                        .push(UnpositionedScorer::from_positioned(sub_scorer));
                    if is_empty {
                        return Ok(per_occur_scorers);
                    }
                }
                Occur::Should => per_occur_scorers.should.push(sub_scorer),
                Occur::MustNot => per_occur_scorers
                    .must_not
                    .push(UnpositionedScorer::from_positioned(sub_scorer)),
            }
        }

        if !deferred_fast_clauses.is_empty() {
            let num_docs = reader.num_docs();
            let max_doc = reader.max_doc();
            let max_driver_cost = DriveCost((f64::from(max_doc) * 0.8) as u64);

            let mut candidate_bound: Option<u32> = None;
            let mut min_driver_cost: Option<DriveCost> = None;

            if !per_occur_scorers.must.is_empty() {
                let mut hints = Vec::with_capacity(per_occur_scorers.must.len());
                for s in &per_occur_scorers.must {
                    if !self.scoring_enabled && s.is::<AllScorer>() {
                        continue;
                    }
                    hints.push(s.size_hint());
                    let c = s.cost().drive;
                    min_driver_cost = Some(min_driver_cost.map_or(c, |prev| prev.min(c)));
                }
                if !hints.is_empty() {
                    candidate_bound = Some(crate::query::size_hint::estimate_intersection(
                        hints.into_iter(),
                        num_docs,
                    ));
                }
            }

            if self.minimum_number_should_match >= 1 {
                if !per_occur_scorers.should.is_empty() {
                    let shoulds = &per_occur_scorers.should;
                    let has_removed_all =
                        !self.scoring_enabled && shoulds.iter().any(|s| s.is::<AllScorer>());
                    if !has_removed_all {
                        let non_empty: Vec<&Box<dyn Scorer>> =
                            shoulds.iter().filter(|s| !s.is::<EmptyScorer>()).collect();
                        if non_empty.len() < self.minimum_number_should_match {
                            // Required SHOULD cannot be satisfied; complex_scorer will
                            // short-circuit.
                            return Ok(per_occur_scorers);
                        }
                        let union_hint = non_empty
                            .iter()
                            .map(|s| u64::from(s.size_hint()))
                            .sum::<u64>()
                            .min(u64::from(max_doc))
                            as u32;
                        let union_cost = non_empty
                            .iter()
                            .map(|s| s.cost().drive)
                            .fold(DriveCost::ZERO, DriveCost::saturating_add);
                        candidate_bound =
                            Some(candidate_bound.map_or(union_hint, |prev| prev.min(union_hint)));
                        min_driver_cost =
                            Some(min_driver_cost.map_or(union_cost, |prev| prev.min(union_cost)));
                    }
                } else {
                    return Ok(per_occur_scorers);
                }
            }

            let has_deferred_must_not = deferred_fast_clauses
                .iter()
                .any(|(occur, _)| *occur == Occur::MustNot);
            let candidate_hint = if self.can_wand_prune_deferred_musts(
                &per_occur_scorers,
                has_deferred_must_not,
                execution_mode,
            ) {
                Some(CandidateHint::Pruned {
                    upper_bound: candidate_bound.unwrap_or(num_docs).max(1),
                })
            } else {
                match (candidate_bound, min_driver_cost) {
                    (Some(c), Some(cost))
                        if c.saturating_mul(32) < num_docs && cost < max_driver_cost =>
                    {
                        Some(CandidateHint::Estimated(c.max(1)))
                    }
                    _ => None,
                }
            };

            for (occur, subweight) in deferred_fast_clauses {
                let sub_scorer = match candidate_hint {
                    Some(hint) => {
                        subweight.scorer_for_candidates_unpositioned(reader, boost, hint)?
                    }
                    None => UnpositionedScorer::from_positioned(subweight.scorer(reader, boost)?),
                };
                let is_empty = sub_scorer.is::<EmptyScorer>();
                match occur {
                    Occur::Must => {
                        per_occur_scorers.must.push(sub_scorer);
                        if is_empty {
                            return Ok(per_occur_scorers);
                        }
                    }
                    Occur::MustNot => per_occur_scorers.must_not.push(sub_scorer),
                    Occur::Should => unreachable!("SHOULD clauses are never deferred"),
                }
            }
        }

        Ok(per_occur_scorers)
    }

    fn complex_scorer<TComplexScoreCombiner: ScoreCombiner>(
        &self,
        reader: &SegmentReader,
        boost: Score,
        score_combiner_fn: impl Fn() -> TComplexScoreCombiner,
        execution_mode: ExecutionMode,
    ) -> crate::Result<SpecializedScorer> {
        let num_docs = reader.num_docs();
        let PerOccurScorers {
            must: mut must_scorers,
            should: mut should_scorers,
            must_not: mut exclude_scorers,
        } = self.per_occur_scorers(reader, boost, execution_mode)?;

        // Indicate how should clauses are combined with must clauses.
        let must_special_scorer_counts = remove_and_count_all_and_empty_scorers(
            &mut must_scorers,
            self.scoring_enabled,
            |s| s.is::<AllScorer>(),
            |s| s.is::<EmptyScorer>(),
        );

        if must_special_scorer_counts.num_empty_scorers > 0 {
            return Ok(SpecializedScorer::Other(Box::new(EmptyScorer)));
        }

        let should_special_scorer_counts = remove_and_count_all_and_empty_scorers(
            &mut should_scorers,
            self.scoring_enabled,
            |s| s.is::<AllScorer>(),
            |s| s.is::<EmptyScorer>(),
        );

        // A MUST_NOT clause never contributes to the score, so its `AllScorer` is
        // always dropped and counted; the check below then empties the result.
        let exclude_special_scorer_counts = remove_and_count_all_and_empty_scorers(
            &mut exclude_scorers,
            false,
            |s| s.is::<AllScorer>(),
            |s| s.is::<EmptyScorer>(),
        );

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
                )),
                1 => ShouldScorersCombinationMethod::Required(scorer_union(
                    should_scorers,
                    &score_combiner_fn,
                    num_docs,
                )),
                n if num_of_should_scorers == n => {
                    // When num_of_should_scorers equals the number of should clauses,
                    // they are no different from must clauses.
                    must_scorers.extend(
                        should_scorers
                            .drain(..)
                            .map(UnpositionedScorer::from_positioned),
                    );
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
                        let term_scorer = *(scorer
                            .into_positioned()
                            .downcast::<TermScorer>()
                            .map_err(|_| ())
                            .unwrap());
                        if term_scorer.freq_reading_option() == FreqReadingOption::ReadFreq {
                            term_scorers.push(term_scorer);
                        } else {
                            other_scorers
                                .push(UnpositionedScorer::from_positioned(
                                    Box::new(term_scorer) as Box<dyn Scorer>
                                ));
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
                    let filter = intersect_scorers_unpositioned(other_scorers, num_docs);
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
                    let filter: Box<dyn Scorer> = if term_scorers.is_empty()
                        && other_scorers.is_empty()
                    {
                        if combined_all_scorer_count > 0 {
                            Box::new(AllScorer::new(reader.max_doc()))
                        } else {
                            Box::new(EmptyScorer)
                        }
                    } else {
                        let mut all_scorers: Vec<UnpositionedScorer> = term_scorers
                            .into_iter()
                            .map(|s| {
                                UnpositionedScorer::from_positioned(Box::new(s) as Box<dyn Scorer>)
                            })
                            .collect();
                        all_scorers.extend(other_scorers);
                        effective_must_scorer(
                            all_scorers,
                            combined_all_scorer_count,
                            reader.max_doc(),
                            num_docs,
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
                                into_box_scorer(should_scorer, &score_combiner_fn, num_docs),
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
                match should_scorer {
                    SpecializedScorer::TermUnion(_) | SpecializedScorer::TermIntersection(_) => {
                        match effective_must_scorer_unpositioned(
                            must_scorers,
                            must_special_scorer_counts.num_all_scorers,
                            reader.max_doc(),
                            num_docs,
                        ) {
                            None => should_scorer,
                            Some(filter) => match should_scorer {
                                SpecializedScorer::TermUnion(terms) => {
                                    SpecializedScorer::FilteredTermUnion { terms, filter }
                                }
                                SpecializedScorer::TermIntersection(terms) => {
                                    SpecializedScorer::FilteredTermIntersection { terms, filter }
                                }
                                _ => unreachable!(),
                            },
                        }
                    }
                    _ => match effective_must_scorer(
                        must_scorers,
                        must_special_scorer_counts.num_all_scorers,
                        reader.max_doc(),
                        num_docs,
                    ) {
                        None => should_scorer,
                        Some(must_scorer) => {
                            let should_boxed =
                                into_box_scorer(should_scorer, &score_combiner_fn, num_docs);
                            SpecializedScorer::Other(intersect_candidate_scorers(
                                vec![
                                    UnpositionedScorer::from_positioned(must_scorer),
                                    UnpositionedScorer::from_positioned(should_boxed),
                                ],
                                num_docs,
                            ))
                        }
                    },
                }
            }
        };
        if exclude_scorers.is_empty() {
            return Ok(include_scorer);
        }

        let include_scorer_boxed = into_box_scorer(include_scorer, &score_combiner_fn, num_docs);
        let scorer: Box<dyn Scorer> = if exclude_scorers.len() == 1 {
            let exclude_scorer = exclude_scorers.pop().unwrap();
            if exclude_scorer.is::<TermScorer>() {
                let term_scorer = *(exclude_scorer
                    .into_positioned()
                    .downcast::<TermScorer>()
                    .map_err(|_| ())
                    .unwrap());
                Box::new(Exclude::new(
                    include_scorer_boxed,
                    ExcludedDocSet::new(term_scorer),
                ))
            } else {
                let excl = ExcludedDocSet::from_unpositioned(exclude_scorer);
                if excl.probe_cost().prefers_batching() {
                    Box::new(BatchExcludeScorer::new(include_scorer_boxed, excl))
                } else {
                    Box::new(Exclude::new(include_scorer_boxed, excl))
                }
            }
        } else {
            let exclude_scorers: Vec<_> = exclude_scorers
                .into_iter()
                .map(ExcludedDocSet::from_unpositioned)
                .collect();
            if exclude_scorers.probe_cost().prefers_batching() {
                Box::new(BatchExcludeScorer::new(
                    include_scorer_boxed,
                    exclude_scorers,
                ))
            } else {
                Box::new(Exclude::new(include_scorer_boxed, exclude_scorers))
            }
        };
        Ok(SpecializedScorer::Other(scorer))
    }
}

#[derive(Default, Copy, Clone, Debug)]
struct AllAndEmptyScorerCounts {
    num_all_scorers: usize,
    num_empty_scorers: usize,
}

/// Removes `EmptyScorer`s, and `AllScorer`s unless `scoring_enabled`: an
/// `AllScorer` scores 1.0 per document, so it only drops out when scores are
/// not needed.
fn remove_and_count_all_and_empty_scorers<T>(
    scorers: &mut Vec<T>,
    scoring_enabled: bool,
    is_all: impl Fn(&T) -> bool,
    is_empty: impl Fn(&T) -> bool,
) -> AllAndEmptyScorerCounts {
    let mut counts = AllAndEmptyScorerCounts::default();
    scorers.retain(|scorer| {
        if !scoring_enabled && is_all(scorer) {
            counts.num_all_scorers += 1;
            false
        } else if is_empty(scorer) {
            counts.num_empty_scorers += 1;
            false
        } else {
            true
        }
    });
    counts
}

impl<TScoreCombiner: ScoreCombiner + Sync> Weight for BooleanWeight<TScoreCombiner> {
    fn uses_fast_fields(&self) -> bool {
        self.weights
            .iter()
            .any(|(_, weight)| weight.uses_fast_fields())
    }

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
            self.complex_scorer(
                reader,
                boost,
                &self.score_combiner_fn,
                ExecutionMode::Exhaustive,
            )
            .map(|specialized_scorer| {
                into_box_scorer(specialized_scorer, &self.score_combiner_fn, num_docs)
            })
        } else {
            self.complex_scorer(
                reader,
                boost,
                DoNothingCombiner::default,
                ExecutionMode::Exhaustive,
            )
            .map(|specialized_scorer| {
                into_box_scorer(specialized_scorer, DoNothingCombiner::default, num_docs)
            })
        }
    }

    fn scorer_for_candidates_unpositioned(
        &self,
        reader: &SegmentReader,
        boost: Score,
        candidate_hint: CandidateHint,
    ) -> crate::Result<UnpositionedScorer> {
        if self.weights.len() == 1 {
            let &(occur, ref weight) = &self.weights[0];
            if occur != Occur::MustNot {
                return weight.scorer_for_candidates_unpositioned(reader, boost, candidate_hint);
            }
        }
        if !self.scoring_enabled
            && self.minimum_number_should_match == 0
            && self.weights.iter().all(|(occur, _)| *occur == Occur::Must)
        {
            let mut fast_fields = self
                .weights
                .iter()
                .filter(|(_, weight)| weight.uses_fast_fields());
            if let (Some((_, candidate_weight)), None) = (fast_fields.next(), fast_fields.next()) {
                for (_, weight) in &self.weights {
                    if !weight.uses_fast_fields()
                        && !weight.scorer(reader, boost)?.is::<AllScorer>()
                    {
                        return self
                            .scorer(reader, boost)
                            .map(UnpositionedScorer::from_positioned);
                    }
                }
                return candidate_weight.scorer_for_candidates_unpositioned(
                    reader,
                    boost,
                    candidate_hint,
                );
            }
        }
        self.scorer(reader, boost)
            .map(UnpositionedScorer::from_positioned)
    }

    fn pruning_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
        init_threshold: Score,
    ) -> crate::Result<Box<dyn crate::query::scorer::PruningScorer>> {
        let scorer = self.complex_scorer(
            reader,
            boost,
            &self.score_combiner_fn,
            ExecutionMode::Pruning,
        )?;
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
                if !(TScoreCombiner::SUPPORTS_BLOCK_WAND
                    && self.scoring_enabled
                    && filter.constant_score().is_some())
                {
                    let union_scorer = into_box_scorer(
                        SpecializedScorer::FilteredTermUnion { terms, filter },
                        &self.score_combiner_fn,
                        reader.num_docs(),
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
                        filter.into_raw_unpositioned(),
                        filter_boost,
                    ))),
                    _ => Ok(Box::new(BlockWandUnionScorer::with_filter(
                        terms,
                        init_threshold,
                        filter.into_raw_unpositioned(),
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
                            filter.into_raw_unpositioned(),
                            filter_boost,
                        )))
                    }
                    _ => Ok(Box::new(BlockWandIntersectionScorer::with_filter(
                        terms,
                        init_threshold,
                        filter.into_raw_unpositioned(),
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
        let scorer = self.complex_scorer(
            reader,
            1.0,
            &self.score_combiner_fn,
            ExecutionMode::Exhaustive,
        )?;
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
                let boxed_scorers: Vec<UnpositionedScorer> =
                    term_scorers
                        .into_iter()
                        .map(|term_scorer| {
                            UnpositionedScorer::from_positioned(
                                Box::new(term_scorer) as Box<dyn Scorer>
                            )
                        })
                        .collect();
                let mut intersection = intersect_candidate_scorers(boxed_scorers, num_docs);
                for_each_scorer(intersection.as_mut(), callback);
            }
            SpecializedScorer::FilteredTermUnion { .. }
            | SpecializedScorer::FilteredTermIntersection { .. } => {
                let mut scorer = into_box_scorer(scorer, &self.score_combiner_fn, num_docs);
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
        let scorer =
            self.complex_scorer(reader, 1.0, &self.score_combiner_fn, ExecutionMode::Pruning)?;
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
            SpecializedScorer::FilteredTermUnion { terms, filter } => {
                if !(TScoreCombiner::SUPPORTS_BLOCK_WAND
                    && self.scoring_enabled
                    && filter.constant_score().is_some())
                {
                    let union_scorer = into_box_scorer(
                        SpecializedScorer::FilteredTermUnion { terms, filter },
                        &self.score_combiner_fn,
                        reader.num_docs(),
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
                            filter.into_raw_unpositioned(),
                            filter_boost,
                        );
                        for_each_pruning_scorer(&mut scorer, callback);
                    }
                    _ if self.should_use_block_maxscore(&terms, reader.max_doc()) => {
                        let mut raw_filter = filter.into_raw_unpositioned();
                        super::block_maxscore::block_maxscore_filtered(
                            terms,
                            threshold,
                            Some(&mut *raw_filter),
                            filter_boost,
                            super::block_maxscore::MIN_BOUND_WINDOW,
                            callback,
                        );
                    }
                    _ => {
                        let mut scorer = BlockWandUnionScorer::with_filter(
                            terms,
                            threshold,
                            filter.into_raw_unpositioned(),
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
                            filter.into_raw_unpositioned(),
                            filter_boost,
                        );
                        for_each_pruning_scorer(&mut scorer, callback);
                    }
                    _ => {
                        let mut scorer = BlockWandIntersectionScorer::with_filter(
                            terms,
                            threshold,
                            filter.into_raw_unpositioned(),
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
        let scorer = self.complex_scorer(
            reader,
            1.0,
            DoNothingCombiner::default,
            ExecutionMode::Exhaustive,
        )?;
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
                let boxed_scorers: Vec<UnpositionedScorer> =
                    term_scorers
                        .into_iter()
                        .map(|term_scorer| {
                            UnpositionedScorer::from_positioned(
                                Box::new(term_scorer) as Box<dyn Scorer>
                            )
                        })
                        .collect();
                let mut intersection = intersect_candidate_scorers(boxed_scorers, num_docs);
                for_each_docset_buffered(intersection.as_mut(), &mut buffer, callback);
            }
            SpecializedScorer::FilteredTermUnion { .. }
            | SpecializedScorer::FilteredTermIntersection { .. } => {
                let mut scorer = into_box_scorer(scorer, DoNothingCombiner::default, num_docs);
                for_each_docset_buffered(scorer.as_mut(), &mut buffer, callback);
            }
            SpecializedScorer::Other(mut scorer) => {
                for_each_docset_buffered(scorer.as_mut(), &mut buffer, callback);
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
    use std::ops::Bound;
    use std::sync::Arc;

    use columnar::ColumnType;
    use common::TinySet;
    use proptest::prelude::*;

    use super::{BooleanWeight, ExecutionMode};
    use crate::query::{
        AllQuery, Bm25Weight, BooleanQuery, CandidateHint, DisjunctionPruning, EnableScoring,
        ExactPostings, ExcludedDocSet, ExclusionSet, FastFieldTermSetQuery, Occur, Query,
        RangeQuery, SumCombiner, TermPredicate, TermPredicateQuery, TermQuery, TermScorer,
    };
    use crate::schema::{IndexRecordOption, Schema, TextOptions, FAST, TEXT};
    use crate::{Bm25Params, DocId, Index, Score, Term, TERMINATED};

    #[derive(Debug)]
    struct ModPredicate {
        modulus: u32,
        rem: u32,
    }

    impl ModPredicate {
        fn matches_doc(&self, doc: DocId) -> bool {
            self.rem < self.modulus && doc % self.modulus == self.rem
        }
    }

    impl TermPredicate for ModPredicate {
        fn filter_bytes(&self, data: &[u8], offsets: &[u32], out: &mut [TinySet]) {
            for i in 0..offsets.len().saturating_sub(1) {
                let slice = &data[offsets[i] as usize..offsets[i + 1] as usize];
                if let Some(num) = std::str::from_utf8(slice)
                    .ok()
                    .and_then(|s| s.strip_prefix("k-"))
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    if self.matches_doc(num) {
                        out[i / 64].insert_mut((i % 64) as u32);
                    }
                }
            }
        }

        fn filter_monotonic_u64s(
            &self,
            _col_type: ColumnType,
            values: &[u64],
            out: &mut [TinySet],
        ) {
            for (i, &v) in values.iter().enumerate() {
                if let Ok(doc) = u32::try_from(v) {
                    if self.matches_doc(doc) {
                        out[i / 64].insert_mut((i % 64) as u32);
                    }
                }
            }
        }

        fn probe_cost(&self) -> crate::ProbeCost {
            crate::ProbeCost::per_doc(5)
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum TermKind {
        Half,
        Div7,
        Div8,
        Div9,
        Offset5,
        Rare,
    }

    impl TermKind {
        fn token(self) -> &'static str {
            match self {
                TermKind::Half => "half",
                TermKind::Div7 => "d7",
                TermKind::Div8 => "d8",
                TermKind::Div9 => "d9",
                TermKind::Offset5 => "off5",
                TermKind::Rare => "rare",
            }
        }

        fn matches(self, doc: DocId) -> bool {
            match self {
                TermKind::Half => doc % 2 == 0,
                TermKind::Div7 => doc % 7 == 0,
                TermKind::Div8 => doc % 8 == 0,
                TermKind::Div9 => doc % 9 == 0,
                TermKind::Offset5 => doc % 5 == 3,
                TermKind::Rare => doc % 64 == 0,
            }
        }
    }

    #[derive(Debug, Clone)]
    enum ClauseLeaf {
        All,
        Term { kind: TermKind, with_freqs: bool },
        Range { lo: u32, hi: u32 },
        NumTermSet { start: u32, step: u32, count: u32 },
        NumPredicate { modulus: u32, rem: u32 },
        DictPredicate { modulus: u32, rem: u32 },
    }

    impl ClauseLeaf {
        fn is_fast_field(&self) -> bool {
            !matches!(self, ClauseLeaf::All | ClauseLeaf::Term { .. })
        }

        fn matches(&self, doc: DocId) -> bool {
            match *self {
                ClauseLeaf::All => true,
                ClauseLeaf::Term { kind, .. } => kind.matches(doc),
                ClauseLeaf::Range { lo, hi } => doc >= lo && doc < hi,
                ClauseLeaf::NumTermSet { start, step, count } => {
                    doc >= start && (doc - start) % step == 0 && (doc - start) / step < count
                }
                ClauseLeaf::NumPredicate { modulus, rem }
                | ClauseLeaf::DictPredicate { modulus, rem } => {
                    rem < modulus && doc % modulus == rem
                }
            }
        }

        fn to_query(
            &self,
            text: crate::schema::Field,
            value: crate::schema::Field,
            tag: crate::schema::Field,
        ) -> Box<dyn Query> {
            match *self {
                ClauseLeaf::All => Box::new(AllQuery),
                ClauseLeaf::Term { kind, with_freqs } => Box::new(TermQuery::new(
                    Term::from_field_text(text, kind.token()),
                    if with_freqs {
                        IndexRecordOption::WithFreqs
                    } else {
                        IndexRecordOption::Basic
                    },
                )),
                ClauseLeaf::Range { lo, hi } => Box::new(RangeQuery::new(
                    Bound::Included(Term::from_field_u64(value, u64::from(lo))),
                    Bound::Excluded(Term::from_field_u64(value, u64::from(hi))),
                )),
                ClauseLeaf::NumTermSet { start, step, count } => {
                    let terms = (0..count)
                        .map(|i| Term::from_field_u64(value, u64::from(start + i * step)));
                    Box::new(FastFieldTermSetQuery::new(terms))
                }
                ClauseLeaf::NumPredicate { modulus, rem } => Box::new(
                    TermPredicateQuery::new(value, Arc::new(ModPredicate { modulus, rem }))
                        .with_exact_postings(ExactPostings::Unavailable),
                ),
                ClauseLeaf::DictPredicate { modulus, rem } => Box::new(
                    TermPredicateQuery::new(tag, Arc::new(ModPredicate { modulus, rem }))
                        .with_exact_postings(ExactPostings::Unavailable),
                ),
            }
        }
    }

    fn arb_clause(num_docs: u32) -> impl Strategy<Value = (Occur, ClauseLeaf)> {
        let occur = prop_oneof![
            5 => Just(Occur::Must),
            2 => Just(Occur::Should),
            1 => Just(Occur::MustNot),
        ];
        let term = prop_oneof![
            Just(TermKind::Half),
            Just(TermKind::Div7),
            Just(TermKind::Div8),
            Just(TermKind::Div9),
            Just(TermKind::Offset5),
            Just(TermKind::Rare),
        ];
        let leaf = prop_oneof![
            1 => Just(ClauseLeaf::All),
            5 => (term, any::<bool>()).prop_map(|(kind, with_freqs)| {
                ClauseLeaf::Term { kind, with_freqs }
            }),
            2 => (0..=num_docs, 0..=num_docs).prop_map(|(a, b)| ClauseLeaf::Range {
                lo: a.min(b),
                hi: a.max(b),
            }),
            2 => (0u32..16, 1u32..16, 0u32..40).prop_map(|(start, step, count)| {
                ClauseLeaf::NumTermSet { start, step, count }
            }),
            2 => (2u32..12).prop_flat_map(|modulus| {
                (0..=modulus).prop_map(move |rem| ClauseLeaf::NumPredicate { modulus, rem })
            }),
            2 => (2u32..12).prop_flat_map(|modulus| {
                (0..=modulus).prop_map(move |rem| ClauseLeaf::DictPredicate { modulus, rem })
            }),
        ];
        (occur, leaf)
    }

    fn expected_boolean_matches(
        clauses: &[(Occur, ClauseLeaf)],
        min_should: usize,
        num_docs: u32,
    ) -> Vec<DocId> {
        let has_must = clauses.iter().any(|(o, _)| *o == Occur::Must);
        let should_count = clauses.iter().filter(|(o, _)| *o == Occur::Should).count();
        let required_should = if min_should == 0 && !has_must && should_count > 0 {
            1
        } else {
            min_should
        };
        if !has_must && should_count == 0 {
            return Vec::new();
        }
        (0..num_docs)
            .filter(|&doc| {
                let mut matched_should = 0usize;
                for (occur, leaf) in clauses {
                    let hit = leaf.matches(doc);
                    match occur {
                        Occur::Must if !hit => return false,
                        Occur::MustNot if hit => return false,
                        Occur::Should if hit => matched_should += 1,
                        _ => {}
                    }
                }
                matched_should >= required_should
            })
            .collect()
    }

    #[test]
    fn candidate_scorer_preserves_boolean_matches_and_scores() -> crate::Result<()> {
        const DOCUMENTS: u32 = 1200;
        let mut schema = Schema::builder();
        let value = schema.add_u64_field("value", FAST);
        let tag = schema.add_text_field("tag", TextOptions::default().set_fast("raw"));
        let text = schema.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        for doc in 0..DOCUMENTS {
            let mut tokens = Vec::new();
            for kind in [
                TermKind::Half,
                TermKind::Div7,
                TermKind::Div8,
                TermKind::Div9,
                TermKind::Offset5,
                TermKind::Rare,
            ] {
                if kind.matches(doc) {
                    tokens.push(kind.token());
                }
            }
            writer.add_document(doc!(
                value => u64::from(doc),
                tag => format!("k-{doc:04}"),
                text => tokens.join(" "),
            ))?;
        }
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        let reader = searcher.segment_reader(0);

        proptest!(ProptestConfig::with_cases(128), |(
            clauses in prop::collection::vec(arb_clause(DOCUMENTS), 2..=5),
            min_should in 0usize..=2,
        )| {
            let query = BooleanQuery::with_minimum_required_clauses(
                clauses
                    .iter()
                    .map(|(occur, leaf)| (*occur, leaf.to_query(text, value, tag)))
                    .collect(),
                min_should,
            );
            let expected_docs = expected_boolean_matches(&clauses, min_should, DOCUMENTS);

            for scoring in [
                EnableScoring::disabled_from_searcher(&searcher),
                EnableScoring::enabled_from_searcher(&searcher),
            ] {
                let weight = query.weight(scoring).unwrap();
                let mut ordinary = weight.scorer(reader, 1.0).unwrap();
                let mut ordinary_results = Vec::new();
                while ordinary.doc() != TERMINATED {
                    ordinary_results.push((ordinary.doc(), ordinary.score()));
                    ordinary.advance();
                }
                let ordinary_docs: Vec<_> = ordinary_results.iter().map(|&(doc, _)| doc).collect();
                prop_assert_eq!(&ordinary_docs, &expected_docs);

                if scoring.is_scoring_enabled() {
                    let mut pruned_results = Vec::new();
                    weight
                        .for_each_pruning(Score::MIN, reader, &mut |doc, score| {
                            pruned_results.push((doc, score));
                            Score::MIN
                        })
                        .unwrap();
                    prop_assert_eq!(pruned_results.len(), ordinary_results.len());
                    for (&(p_doc, p_score), &(o_doc, o_score)) in
                        pruned_results.iter().zip(&ordinary_results)
                    {
                        prop_assert_eq!(p_doc, o_doc);
                        prop_assert!(
                            (p_score - o_score).abs() <= 1e-5,
                            "doc {p_doc}: pruned score {p_score} != ordinary score {o_score}"
                        );
                    }
                }

                let mut candidates = ExcludedDocSet::from_unpositioned(
                    weight
                        .scorer_for_candidates_unpositioned(
                            reader,
                            1.0,
                            CandidateHint::Estimated(reader.max_doc()),
                        )
                        .unwrap(),
                );
                let actual_docs: Vec<_> = (0..reader.max_doc())
                    .filter(|&doc| candidates.contains(doc))
                    .collect();
                prop_assert_eq!(&actual_docs, &expected_docs);

                let mut candidate = weight
                    .scorer_for_candidates_unpositioned(
                        reader,
                        1.0,
                        CandidateHint::Estimated(reader.max_doc()),
                    )
                    .unwrap()
                    .into_positioned();
                for (doc, score) in ordinary_results {
                    prop_assert_eq!(candidate.seek(doc), doc);
                    prop_assert_eq!(candidate.score(), score);
                }
            }
        });

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
        let specialized = bool_weight.complex_scorer(
            reader,
            1.0,
            &bool_weight.score_combiner_fn,
            ExecutionMode::Exhaustive,
        )?;
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
        let specialized = bool_weight.complex_scorer(
            reader,
            1.0,
            &bool_weight.score_combiner_fn,
            ExecutionMode::Exhaustive,
        )?;
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
        let specialized = bool_weight.complex_scorer(
            reader,
            1.0,
            &bool_weight.score_combiner_fn,
            ExecutionMode::Exhaustive,
        )?;
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
