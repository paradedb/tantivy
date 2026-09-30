use std::collections::HashMap;

use crate::docset::{DocSet, COLLECT_BLOCK_BUFFER_LEN};
use crate::index::SegmentReader;
use crate::postings::FreqReadingOption;
use crate::query::boolean_query::{
    BlockWandIntersectionScorer, BlockWandSingleScorer, BlockWandUnionScorer, FilteredPruningScorer,
};
use crate::query::disjunction::Disjunction;
use crate::query::explanation::does_not_match;
use crate::query::score_combiner::{DoNothingCombiner, ScoreCombiner};
use crate::query::scorer::{BasicPruningScorer, PruningScorer};
use crate::query::term_query::TermScorer;
use crate::query::weight::{for_each_docset_buffered, for_each_pruning_scorer, for_each_scorer};
use crate::query::{
    intersect_scorers, AllScorer, BufferedUnionScorer, DisjunctionPruning, EmptyScorer, Exclude,
    Explanation, Intersection, Occur, RequiredOptionalScorer, Scorer, Weight,
};
use crate::{DocId, Score, TERMINATED};

enum SpecializedScorer<TOther = Box<dyn Scorer>> {
    TermUnion(Vec<TermScorer>),
    TermIntersection(Vec<TermScorer>),
    FilteredTermUnion {
        terms: Vec<TermScorer>,
        filter: TOther,
    },
    FilteredTermIntersection {
        terms: Vec<TermScorer>,
        filter: TOther,
    },
    Other(TOther),
}

enum FilteredTermsPruningScorer {
    Empty,
    Single(FilteredPruningScorer<BlockWandSingleScorer, Box<dyn Scorer>>),
    Union(FilteredPruningScorer<BlockWandUnionScorer, Box<dyn Scorer>>),
    Intersection(FilteredPruningScorer<BlockWandIntersectionScorer, Box<dyn Scorer>>),
}

impl FilteredTermsPruningScorer {
    fn into_pruning_scorer(self) -> Option<Box<dyn PruningScorer>> {
        match self {
            FilteredTermsPruningScorer::Empty => Some(Box::new(EmptyScorer)),
            FilteredTermsPruningScorer::Single(s) => Some(Box::new(s)),
            FilteredTermsPruningScorer::Union(s) => Some(Box::new(s)),
            FilteredTermsPruningScorer::Intersection(s) => Some(Box::new(s)),
        }
    }

    fn for_each_pruning(self, callback: &mut dyn FnMut(DocId, Score) -> Score) {
        match self {
            FilteredTermsPruningScorer::Empty => {}
            FilteredTermsPruningScorer::Single(mut s) => for_each_pruning_scorer(&mut s, callback),
            FilteredTermsPruningScorer::Union(mut s) => for_each_pruning_scorer(&mut s, callback),
            FilteredTermsPruningScorer::Intersection(mut s) => {
                for_each_pruning_scorer(&mut s, callback)
            }
        }
    }
}

#[inline]
fn inner_pruning_threshold(threshold: Score, filter_boost: Option<Score>) -> Score {
    if threshold == Score::MIN {
        Score::MIN
    } else {
        threshold - filter_boost.unwrap_or(0.0)
    }
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
    mut scorers: Vec<Box<dyn Scorer>>,
    score_combiner_fn: impl Fn() -> TScoreCombiner,
    num_docs: u32,
) -> SpecializedScorer
where
    TScoreCombiner: ScoreCombiner,
{
    assert!(!scorers.is_empty());
    if scorers.len() == 1 {
        let single = scorers.pop().unwrap();
        if single.is::<TermScorer>() {
            let term_scorer = *(single.downcast::<TermScorer>().map_err(|_| ()).unwrap());
            if term_scorer.freq_reading_option() == FreqReadingOption::ReadFreq {
                return SpecializedScorer::TermUnion(vec![term_scorer]);
            } else {
                return SpecializedScorer::Other(Box::new(term_scorer));
            }
        } else {
            return SpecializedScorer::Other(single);
        }
    }
    if scorers.iter().all(|scorer| scorer.is::<TermScorer>()) {
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
        } else {
            return SpecializedScorer::Other(Box::new(BufferedUnionScorer::build(
                scorers,
                score_combiner_fn,
                num_docs,
            )));
        }
    }
    SpecializedScorer::Other(Box::new(BufferedUnionScorer::build(
        scorers,
        score_combiner_fn,
        num_docs,
    )))
}

fn into_box_scorer<TScoreCombiner: ScoreCombiner, TOther: Into<Box<dyn Scorer>>>(
    scorer: SpecializedScorer<TOther>,
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
            let intersection = Intersection::new(term_scorers, num_docs);
            if intersection.doc() == TERMINATED {
                Box::new(EmptyScorer)
            } else {
                Box::new(intersection)
            }
        }
        SpecializedScorer::FilteredTermUnion { terms, filter } => {
            let term_scorer = into_box_scorer(
                SpecializedScorer::<TOther>::TermUnion(terms),
                score_combiner_fn,
                num_docs,
            );
            intersect_scorers(vec![term_scorer, filter.into()], num_docs)
        }
        SpecializedScorer::FilteredTermIntersection { terms, filter } => {
            let term_scorer = into_box_scorer(
                SpecializedScorer::<TOther>::TermIntersection(terms),
                score_combiner_fn,
                num_docs,
            );
            intersect_scorers(vec![term_scorer, filter.into()], num_docs)
        }
        SpecializedScorer::Other(scorer) => scorer.into(),
    }
}

impl<TOther: Scorer + Into<Box<dyn Scorer>>> SpecializedScorer<TOther> {
    fn for_each<TScoreCombiner: ScoreCombiner>(
        self,
        num_docs: u32,
        score_combiner_fn: impl Fn() -> TScoreCombiner,
        callback: &mut dyn FnMut(DocId, Score),
    ) {
        match self {
            SpecializedScorer::TermUnion(mut term_scorers) => {
                if term_scorers.len() == 1 {
                    let mut term_scorer = term_scorers.pop().unwrap();
                    for_each_scorer(&mut term_scorer, callback);
                } else {
                    let mut union_scorer =
                        BufferedUnionScorer::build(term_scorers, score_combiner_fn, num_docs);
                    for_each_scorer(&mut union_scorer, callback);
                }
            }
            SpecializedScorer::TermIntersection(term_scorers) => {
                let mut intersection = Intersection::new(term_scorers, num_docs);
                for_each_scorer(&mut intersection, callback);
            }
            SpecializedScorer::FilteredTermUnion { .. }
            | SpecializedScorer::FilteredTermIntersection { .. } => {
                let mut scorer = into_box_scorer(self, score_combiner_fn, num_docs);
                for_each_scorer(scorer.as_mut(), callback);
            }
            SpecializedScorer::Other(mut scorer) => {
                for_each_scorer(&mut scorer, callback);
            }
        }
    }
}

impl<TOther: DocSet + Into<Box<dyn Scorer>>> SpecializedScorer<TOther> {
    fn for_each_no_score<TScoreCombiner: ScoreCombiner>(
        self,
        num_docs: u32,
        score_combiner_fn: impl Fn() -> TScoreCombiner,
        buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN],
        callback: &mut dyn FnMut(&[DocId]),
    ) {
        match self {
            SpecializedScorer::TermUnion(mut term_scorers) => {
                if term_scorers.len() == 1 {
                    let mut term_scorer = term_scorers.pop().unwrap();
                    for_each_docset_buffered(&mut term_scorer, buffer, callback);
                } else {
                    let mut union_scorer =
                        BufferedUnionScorer::build(term_scorers, score_combiner_fn, num_docs);
                    for_each_docset_buffered(&mut union_scorer, buffer, callback);
                }
            }
            SpecializedScorer::TermIntersection(term_scorers) => {
                let mut intersection = Intersection::new(term_scorers, num_docs);
                for_each_docset_buffered(&mut intersection, buffer, callback);
            }
            SpecializedScorer::FilteredTermUnion { .. }
            | SpecializedScorer::FilteredTermIntersection { .. } => {
                let mut scorer = into_box_scorer(self, DoNothingCombiner::default, num_docs);
                for_each_docset_buffered(scorer.as_mut(), buffer, callback);
            }
            SpecializedScorer::Other(mut scorer) => {
                for_each_docset_buffered(&mut scorer, buffer, callback);
            }
        }
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
        Some(intersect_scorers(must_scorers, num_docs))
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

enum ShouldScorersCombinationMethod<TOther = Box<dyn Scorer>> {
    // Should scorers are irrelevant.
    Ignored,
    // Only contributes to final score.
    Optional(SpecializedScorer<TOther>),
    // Regardless of score, the should scorers may impact whether a document is matching or not.
    Required(SpecializedScorer<TOther>),
}

/// Structural classification of a [`BooleanWeight`] for dynamic pruning (e.g. BlockWAND).
///
/// Unifies the static query capability check ([`BooleanWeight::is_pruning_supported`])
/// with the filtered pruning planner ([`BooleanWeight::try_build_filtered_pruning`]).
/// Inspecting clause structure in a single place ensures the capability check and the
/// pruning scorer construction logic remain consistent.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PruningShape {
    /// Not eligible for dynamic pruning (e.g. scoring disabled, combiner does not support
    /// BlockWAND, contains `MustNot` clauses, or clauses lack pruning support).
    NotSupported,
    /// Disjunction of scoring clauses (all `Occur::Should`).
    PureDisjunction,
    /// Flat filtered disjunction: `Occur::Should` scoring clauses filtered by `Occur::Must`
    /// clauses.
    FilteredDisjunction,
    /// Conjunction where all `Occur::Must` clauses support dynamic pruning.
    PureConjunction,
    /// Filtered conjunction: exactly one `Occur::Must` clause is the dynamic pruning driver,
    /// and the remaining `Occur::Must` clauses are filters.
    FilteredConjunction { scoring_driver_idx: usize },
    /// Conjunction with multiple scoring drivers and filter clauses.
    FilteredConjunctionMultipleDrivers,
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

    fn pruning_shape(&self) -> PruningShape {
        if !self.scoring_enabled || !TScoreCombiner::SUPPORTS_BLOCK_WAND || self.weights.is_empty()
        {
            return PruningShape::NotSupported;
        }
        if self
            .weights
            .iter()
            .any(|(occur, _)| *occur == Occur::MustNot)
        {
            return PruningShape::NotSupported;
        }
        let has_must = self.weights.iter().any(|(occur, _)| *occur == Occur::Must);
        let has_should = self
            .weights
            .iter()
            .any(|(occur, _)| *occur == Occur::Should);

        if has_should && has_must {
            if self.minimum_number_should_match != 1 {
                return PruningShape::NotSupported;
            }
            let all_should_prune = self
                .weights
                .iter()
                .filter(|(occur, _)| *occur == Occur::Should)
                .all(|(_, weight)| weight.is_pruning_supported());
            if all_should_prune {
                PruningShape::FilteredDisjunction
            } else {
                PruningShape::NotSupported
            }
        } else if has_should {
            if self.minimum_number_should_match > 1 {
                return PruningShape::NotSupported;
            }
            let all_should_prune = self
                .weights
                .iter()
                .all(|(_, weight)| weight.is_pruning_supported());
            if all_should_prune {
                PruningShape::PureDisjunction
            } else {
                PruningShape::NotSupported
            }
        } else if has_must {
            let bmw_count = self
                .weights
                .iter()
                .filter(|(_, weight)| weight.is_pruning_supported())
                .count();
            if bmw_count == self.weights.len() {
                PruningShape::PureConjunction
            } else if bmw_count == 1 && self.weights.len() >= 2 {
                let scoring_driver_idx = self
                    .weights
                    .iter()
                    .position(|(_, weight)| weight.is_pruning_supported())
                    .unwrap();
                PruningShape::FilteredConjunction { scoring_driver_idx }
            } else if bmw_count >= 2 {
                PruningShape::FilteredConjunctionMultipleDrivers
            } else {
                PruningShape::NotSupported
            }
        } else {
            PruningShape::NotSupported
        }
    }

    fn try_build_filtered_pruning(
        &self,
        reader: &SegmentReader,
        boost: Score,
        threshold: Score,
    ) -> crate::Result<Option<Box<dyn PruningScorer>>> {
        let PruningShape::FilteredConjunction {
            scoring_driver_idx: bmw_idx,
        } = self.pruning_shape()
        else {
            return Ok(None);
        };

        let mut filter_scorers = Vec::new();
        for (idx, (_, weight)) in self.weights.iter().enumerate() {
            if idx != bmw_idx {
                filter_scorers.push(weight.scorer(reader, boost)?);
            }
        }
        let filter = intersect_scorers(filter_scorers, reader.num_docs());

        // BMW requires a constant score contribution from the filter to prune blocks
        // without false negatives (subtracting the filter boost from the threshold).
        // If the filter produces document-dependent dynamic scores, we must fall back
        // to non-BMW pruning.
        let filter_boost = filter.constant_score();
        if self.scoring_enabled && filter_boost.is_none() {
            return Ok(None);
        }
        let filter_boost = filter_boost.unwrap_or(0.0);

        let inner_threshold = inner_pruning_threshold(threshold, Some(filter_boost));
        let Some(bmw_scorer) =
            self.weights[bmw_idx]
                .1
                .pruning_scorer(reader, boost, inner_threshold)?
        else {
            return Ok(None);
        };

        Ok(Some(Box::new(
            FilteredPruningScorer::new_with_filter_boost(bmw_scorer, filter, filter_boost),
        )))
    }

    fn build_filtered_term_union_pruning_scorer(
        &self,
        mut terms: Vec<TermScorer>,
        filter: Box<dyn Scorer>,
        threshold: Score,
    ) -> FilteredTermsPruningScorer {
        let filter_boost = filter.constant_score().unwrap_or(0.0);
        let inner_threshold = inner_pruning_threshold(threshold, Some(filter_boost));
        terms.retain(|scorer| scorer.doc() < TERMINATED);
        match terms.len() {
            0 => FilteredTermsPruningScorer::Empty,
            1 => {
                let single = BlockWandSingleScorer::new(terms.pop().unwrap(), inner_threshold);
                FilteredTermsPruningScorer::Single(FilteredPruningScorer::new_with_filter_boost(
                    single,
                    filter,
                    filter_boost,
                ))
            }
            _ => {
                let union_scorer = BlockWandUnionScorer::new(terms, inner_threshold);
                FilteredTermsPruningScorer::Union(FilteredPruningScorer::new_with_filter_boost(
                    union_scorer,
                    filter,
                    filter_boost,
                ))
            }
        }
    }

    fn build_filtered_term_intersection_pruning_scorer(
        &self,
        mut terms: Vec<TermScorer>,
        filter: Box<dyn Scorer>,
        threshold: Score,
    ) -> FilteredTermsPruningScorer {
        if terms.iter().any(|s| s.doc() == TERMINATED) {
            FilteredTermsPruningScorer::Empty
        } else if terms.len() < 2 {
            let filter_boost = filter.constant_score().unwrap_or(0.0);
            let inner_threshold = inner_pruning_threshold(threshold, Some(filter_boost));
            if let Some(term) = terms.pop() {
                let single = BlockWandSingleScorer::new(term, inner_threshold);
                FilteredTermsPruningScorer::Single(FilteredPruningScorer::new_with_filter_boost(
                    single,
                    filter,
                    filter_boost,
                ))
            } else {
                FilteredTermsPruningScorer::Empty
            }
        } else {
            let filter_boost = filter.constant_score().unwrap_or(0.0);
            let inner_threshold = inner_pruning_threshold(threshold, Some(filter_boost));
            let intersection_scorer = BlockWandIntersectionScorer::new(terms, inner_threshold);
            FilteredTermsPruningScorer::Intersection(FilteredPruningScorer::new_with_filter_boost(
                intersection_scorer,
                filter,
                filter_boost,
            ))
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
            return Ok(SpecializedScorer::Other(
                Box::new(EmptyScorer) as Box<dyn Scorer>
            ));
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
            return Ok(SpecializedScorer::Other(
                Box::new(EmptyScorer) as Box<dyn Scorer>
            ));
        }

        let effective_minimum_number_should_match = self
            .minimum_number_should_match
            .saturating_sub(should_special_scorer_counts.num_all_scorers);

        let should_scorers: ShouldScorersCombinationMethod = {
            let num_of_should_scorers = should_scorers.len();
            if effective_minimum_number_should_match > num_of_should_scorers {
                // We don't have enough scorers to satisfy the minimum number of should matches.
                // The request will match no documents.
                return Ok(SpecializedScorer::Other(
                    Box::new(EmptyScorer) as Box<dyn Scorer>
                ));
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
                        SpecializedScorer::Other(Box::new(EmptyScorer) as Box<dyn Scorer>)
                    }
                } else if combined_all_scorer_count == 0
                    && !term_scorers.is_empty()
                    && !other_scorers.is_empty()
                {
                    let filter = intersect_scorers(other_scorers, num_docs);
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
                        } else if other_scorers.is_empty() {
                            if term_scorers.len() == 1 {
                                Box::new(term_scorers.pop().unwrap())
                            } else {
                                let intersection = Intersection::new(term_scorers, num_docs);
                                if intersection.doc() == TERMINATED {
                                    Box::new(EmptyScorer)
                                } else {
                                    Box::new(intersection)
                                }
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
                                Box<dyn Scorer>,
                                Box<dyn Scorer>,
                                TScoreCombiner,
                            >::new(
                                must_scorer,
                                into_box_scorer(should_scorer, &score_combiner_fn, num_docs),
                            ))
                                as Box<dyn Scorer>)
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
                            let should_boxed =
                                into_box_scorer(should_scorer, &score_combiner_fn, num_docs);
                            SpecializedScorer::Other(intersect_scorers(
                                vec![must_scorer, should_boxed],
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
                    into_box_scorer(specialized_scorer, &self.score_combiner_fn, num_docs)
                })
        } else {
            self.complex_scorer(reader, boost, DoNothingCombiner::default)
                .map(|specialized_scorer| {
                    into_box_scorer(specialized_scorer, DoNothingCombiner::default, num_docs)
                })
        }
    }

    fn pruning_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
        init_threshold: Score,
    ) -> crate::Result<Option<Box<dyn PruningScorer>>> {
        if self.pruning_shape() == PruningShape::NotSupported {
            return Ok(None);
        }
        let scorer = self.complex_scorer(reader, boost, &self.score_combiner_fn)?;
        match scorer {
            // Block-WAND scores by summing the matching terms, so it may only
            // drive a combiner that sums. Anything else (dis_max) still needs
            // every matching term and falls back to the plain union.
            SpecializedScorer::TermUnion(_) if !TScoreCombiner::SUPPORTS_BLOCK_WAND => Ok(None),
            SpecializedScorer::TermUnion(mut scorers) => {
                // Drop already-exhausted scorers so a lone survivor uses the
                // (~3x faster) single-scorer specialization
                scorers.retain(|scorer| scorer.doc() < TERMINATED);
                match scorers.len() {
                    0 => Ok(Some(Box::new(EmptyScorer))),
                    1 => Ok(Some(Box::new(BlockWandSingleScorer::new(
                        scorers.pop().unwrap(),
                        init_threshold,
                    )))),
                    _ => Ok(Some(Box::new(BlockWandUnionScorer::new(
                        scorers,
                        init_threshold,
                    )))),
                }
            }
            SpecializedScorer::TermIntersection(scorers) => Ok(Some(Box::new(
                BlockWandIntersectionScorer::new(scorers, init_threshold),
            ))),
            SpecializedScorer::FilteredTermUnion { terms, filter } => {
                if !TScoreCombiner::SUPPORTS_BLOCK_WAND
                    || !self.scoring_enabled
                    || filter.constant_score().is_none()
                {
                    return Ok(None);
                }
                let scorer = self
                    .build_filtered_term_union_pruning_scorer(terms, filter, init_threshold)
                    .into_pruning_scorer();
                Ok(scorer)
            }
            SpecializedScorer::FilteredTermIntersection { terms, filter } => {
                if !self.scoring_enabled
                    || !TScoreCombiner::SUPPORTS_BLOCK_WAND
                    || filter.constant_score().is_none()
                {
                    return Ok(None);
                }
                let scorer = self
                    .build_filtered_term_intersection_pruning_scorer(terms, filter, init_threshold)
                    .into_pruning_scorer();
                Ok(scorer)
            }
            SpecializedScorer::Other(_) => {
                self.try_build_filtered_pruning(reader, boost, init_threshold)
            }
        }
    }

    fn is_pruning_supported(&self) -> bool {
        !matches!(self.pruning_shape(), PruningShape::NotSupported)
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
        scorer.for_each(reader.num_docs(), &self.score_combiner_fn, callback);
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
                let mut scorer = BasicPruningScorer::new(union_scorer, threshold);
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
                if !TScoreCombiner::SUPPORTS_BLOCK_WAND
                    || !self.scoring_enabled
                    || filter.constant_score().is_none()
                {
                    let union_scorer = BufferedUnionScorer::build(
                        terms,
                        &self.score_combiner_fn,
                        reader.num_docs(),
                    );
                    let combined =
                        intersect_scorers(vec![Box::new(union_scorer), filter], reader.num_docs());
                    let mut scorer = BasicPruningScorer::new(combined, threshold);
                    for_each_pruning_scorer(&mut scorer, callback);
                } else {
                    self.build_filtered_term_union_pruning_scorer(terms, filter, threshold)
                        .for_each_pruning(callback);
                }
            }
            SpecializedScorer::FilteredTermIntersection { mut terms, filter } => {
                if terms.iter().any(|s| s.doc() == TERMINATED) {
                    return Ok(());
                }
                if !self.scoring_enabled
                    || !TScoreCombiner::SUPPORTS_BLOCK_WAND
                    || filter.constant_score().is_none()
                {
                    let term_intersection: Box<dyn Scorer> = if terms.len() == 1 {
                        Box::new(terms.pop().unwrap())
                    } else {
                        let intersection = Intersection::new(terms, reader.num_docs());
                        if intersection.doc() == TERMINATED {
                            Box::new(EmptyScorer)
                        } else {
                            Box::new(intersection)
                        }
                    };
                    let combined =
                        intersect_scorers(vec![term_intersection, filter], reader.num_docs());
                    let mut scorer = BasicPruningScorer::new(combined, threshold);
                    for_each_pruning_scorer(&mut scorer, callback);
                } else {
                    self.build_filtered_term_intersection_pruning_scorer(terms, filter, threshold)
                        .for_each_pruning(callback);
                }
            }
            SpecializedScorer::Other(scorer) => {
                if let Some(mut filtered) =
                    self.try_build_filtered_pruning(reader, 1.0, threshold)?
                {
                    for_each_pruning_scorer(filtered.as_mut(), callback);
                } else {
                    let mut scorer = BasicPruningScorer::new(scorer, threshold);
                    for_each_pruning_scorer(&mut scorer, callback);
                }
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
        let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];
        scorer.for_each_no_score(
            reader.num_docs(),
            DoNothingCombiner::default,
            &mut buffer,
            callback,
        );
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
        use crate::query::term_query::TermScorer;
        use crate::query::{EnableScoring, Occur, Query, TermQuery, Weight};
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
        assert!(bool_weight.pruning_scorer(reader, 1.0, 0.0)?.is_none());
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
        assert!(bool_weight.pruning_scorer(reader, 1.0, 0.0)?.is_none());
        let mut count = 0;
        bool_weight.for_each_pruning(0.0, reader, &mut |_doc, _score| {
            count += 1;
            0.0
        })?;
        assert_eq!(count, 1);

        // Case 4: Filter with constant score enables FilteredPruningScorer for BMW
        use crate::query::ConstScoreQuery;
        let weight_a = term_a.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let weight_b = term_b.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let const_c = ConstScoreQuery::new(Box::new(term_c), 1.0);
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
        assert!(bool_weight.pruning_scorer(reader, 1.0, 0.0)?.is_some());
        let mut count = 0;
        bool_weight.for_each_pruning(0.0, reader, &mut |_doc, _score| {
            count += 1;
            0.0
        })?;
        assert_eq!(count, 1);

        Ok(())
    }
}
