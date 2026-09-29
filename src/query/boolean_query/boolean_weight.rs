use std::collections::HashMap;

use crate::docset::{DocSet, COLLECT_BLOCK_BUFFER_LEN};
use crate::index::SegmentReader;
use crate::postings::{FreqReadingOption, SegmentPostings};
use crate::query::boolean_query::mixed_scorer::MixedScorer;
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
    intersect_scorers, AllScorer, BufferedUnionScorer, DisjunctionPruning, EmptyScorer, Exclude,
    Explanation, Occur, RequiredOptionalScorer, Scorer, Weight,
};
use crate::{DocId, Score, TERMINATED};

enum SpecializedScorer {
    TermUnion(Vec<TermScorer>),
    MixedUnion(Vec<MixedScorer>),
    TermIntersection(Vec<TermScorer>),
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
    if TScoreCombiner::SUPPORTS_BLOCK_WAND
        && scorers.iter().all(|scorer| {
            if let Some(term) = scorer.downcast_ref::<TermScorer>() {
                term.freq_reading_option() == FreqReadingOption::ReadFreq
                    && term.bm25_weight().global_score_bound().is_some()
            } else {
                scorer
                    .downcast_ref::<PhraseScorer<SegmentPostings>>()
                    .and_then(PhraseScorer::global_score_bound)
                    .is_some()
            }
        })
    {
        let scorers = scorers
            .into_iter()
            .map(|scorer| match scorer.downcast::<TermScorer>() {
                Ok(term) => MixedScorer::Term(*term),
                Err(scorer) => {
                    let phrase = scorer
                        .downcast::<PhraseScorer<SegmentPostings>>()
                        .map_err(|_| ())
                        .unwrap();
                    let bound = phrase.global_score_bound().unwrap();
                    MixedScorer::Phrase(*phrase, bound)
                }
            })
            .collect();
        return SpecializedScorer::MixedUnion(scorers);
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
        SpecializedScorer::MixedUnion(scorers) => Box::new(BufferedUnionScorer::build(
            scorers,
            score_combiner_fn,
            num_docs,
        )),
        SpecializedScorer::TermIntersection(term_scorers) => {
            let boxed_scorers: Vec<Box<dyn Scorer>> = term_scorers
                .into_iter()
                .map(|s| Box::new(s) as Box<dyn Scorer>)
                .collect();
            intersect_scorers(boxed_scorers, num_docs)
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
        if scorers
            .iter()
            .any(|scorer| !scorer.bm25_weight().supports_pruning(1.0))
        {
            return false;
        }
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

                // Try to detect a pure TermScorer intersection for block-max optimization.
                // Preconditions: no removed AllScorers, at least 2 scorers, all TermScorer
                // with frequency reading enabled.
                if combined_all_scorer_count == 0
                    && must_scorers.len() >= 2
                    && must_scorers.iter().all(|s| s.is::<TermScorer>())
                {
                    let term_scorers: Vec<TermScorer> = must_scorers
                        .into_iter()
                        .map(|s| *(s.downcast::<TermScorer>().map_err(|_| ()).unwrap()))
                        .collect();
                    if term_scorers
                        .iter()
                        .all(|s| s.freq_reading_option() == FreqReadingOption::ReadFreq)
                    {
                        SpecializedScorer::TermIntersection(term_scorers)
                    } else {
                        let must_scorers: Vec<Box<dyn Scorer>> = term_scorers
                            .into_iter()
                            .map(|s| Box::new(s) as Box<dyn Scorer>)
                            .collect();
                        let boxed_scorer: Box<dyn Scorer> =
                            effective_must_scorer(must_scorers, 0, reader.max_doc(), num_docs)
                                .unwrap_or_else(|| Box::new(EmptyScorer));
                        SpecializedScorer::Other(boxed_scorer)
                    }
                } else {
                    let boxed_scorer: Box<dyn Scorer> = effective_must_scorer(
                        must_scorers,
                        combined_all_scorer_count,
                        reader.max_doc(),
                        num_docs,
                    )
                    .unwrap_or_else(|| Box::new(EmptyScorer));
                    SpecializedScorer::Other(boxed_scorer)
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
                    Some(must_scorer) => {
                        // Has MUST constraint: intersect MUST with SHOULD.
                        let should_boxed =
                            into_box_scorer(should_scorer, &score_combiner_fn, num_docs);
                        SpecializedScorer::Other(intersect_scorers(
                            vec![must_scorer, should_boxed],
                            num_docs,
                        ))
                    }
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
            SpecializedScorer::MixedUnion(scorers) => {
                let union =
                    BufferedUnionScorer::build(scorers, &self.score_combiner_fn, reader.num_docs());
                Ok(Box::new(BasicPruningScorer::new(
                    Box::new(union),
                    init_threshold,
                )))
            }
            SpecializedScorer::TermIntersection(scorers) => Ok(Box::new(
                BlockWandIntersectionScorer::new(scorers, init_threshold),
            )),
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
            SpecializedScorer::MixedUnion(scorers) => {
                let mut union =
                    BufferedUnionScorer::build(scorers, &self.score_combiner_fn, num_docs);
                for_each_scorer(&mut union, callback);
            }
            SpecializedScorer::TermIntersection(term_scorers) => {
                let boxed_scorers: Vec<Box<dyn Scorer>> = term_scorers
                    .into_iter()
                    .map(|term_scorer| Box::new(term_scorer) as Box<dyn Scorer>)
                    .collect();
                let mut intersection = intersect_scorers(boxed_scorers, num_docs);
                for_each_scorer(intersection.as_mut(), callback);
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
                        let min_window = if scorers.len() == 2 {
                            0
                        } else {
                            super::block_maxscore::MIN_BOUND_WINDOW
                        };
                        super::block_maxscore::block_maxscore(
                            scorers, threshold, min_window, callback,
                        );
                    }
                    _ => {
                        let mut scorer = BlockWandUnionScorer::new(scorers, threshold);
                        for_each_pruning_scorer(&mut scorer, callback);
                    }
                }
            }
            SpecializedScorer::MixedUnion(scorers) => {
                if self.disjunction_pruning == DisjunctionPruning::BlockWand {
                    let union = BufferedUnionScorer::build(
                        scorers,
                        &self.score_combiner_fn,
                        reader.num_docs(),
                    );
                    let mut scorer = BasicPruningScorer::new(Box::new(union), threshold);
                    for_each_pruning_scorer(&mut scorer, callback);
                } else {
                    super::block_maxscore::block_maxscore(
                        scorers,
                        threshold,
                        super::block_maxscore::MIN_BOUND_WINDOW,
                        callback,
                    );
                }
            }
            SpecializedScorer::TermIntersection(scorers) => {
                let mut scorer = BlockWandIntersectionScorer::new(scorers, threshold);
                for_each_pruning_scorer(&mut scorer, callback);
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
            SpecializedScorer::MixedUnion(scorers) => {
                let mut union =
                    BufferedUnionScorer::build(scorers, &self.score_combiner_fn, num_docs);
                for_each_docset_buffered(&mut union, &mut buffer, callback);
            }
            SpecializedScorer::TermIntersection(term_scorers) => {
                let boxed_scorers: Vec<Box<dyn Scorer>> = term_scorers
                    .into_iter()
                    .map(|term_scorer| Box::new(term_scorer) as Box<dyn Scorer>)
                    .collect();
                let mut intersection = intersect_scorers(boxed_scorers, num_docs);
                for_each_docset_buffered(intersection.as_mut(), &mut buffer, callback);
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
    use super::BooleanWeight;
    use crate::query::{Bm25Weight, DisjunctionPruning, SumCombiner, TermScorer};
    use crate::Bm25Params;

    #[test]
    fn test_mixed_phrase_union_matches_exhaustive() -> crate::Result<()> {
        use crate::query::{
            BooleanQuery, BoostQuery, DisjunctionMaxQuery, EnableScoring, Occur, Query, QueryParser,
        };
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
            let field = schema.add_text_field("text", options);
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
                writer.add_document(doc!(field => text))?;
                if ordinal == 31 {
                    writer.commit()?;
                }
            }
            writer.commit()?;
            drop(writer);
            let searcher = index.reader()?.searcher();
            let parser = QueryParser::for_index(&index, vec![field]);
            let reader = searcher.segment_reader(0);
            let mut scorers = Vec::new();
            for expression in ["a", "\"a b\""] {
                scorers.push(
                    parser
                        .parse_query(expression)?
                        .weight(EnableScoring::enabled_from_searcher(&searcher))?
                        .scorer(reader, 1.0)?,
                );
            }
            assert!(matches!(
                super::scorer_union(scorers, SumCombiner::default, reader.num_docs()),
                super::SpecializedScorer::MixedUnion(_)
            ));
            let mut queries = Vec::new();
            for expression in [
                "a OR \"a b\"",
                "a OR a OR \"a b\"",
                "\"a b\" OR \"b c\"",
                "a OR \"a a\" OR \"a b\"",
                "a OR \"missing b\"",
                "a OR \"a b\" OR missing",
                "a OR \"a b\"^2.5",
                "a OR \"a b\"~2",
                "(a OR \"a b\") AND c",
                "a OR b OR c OR 1,2 OR x",
            ] {
                queries.push((expression, parser.parse_query(expression)?));
            }
            queries.push((
                "negative phrase",
                Box::new(BooleanQuery::union(vec![
                    parser.parse_query("a")?,
                    Box::new(BoostQuery::new(parser.parse_query("\"a b\"")?, -1.0)),
                ])),
            ));
            queries.push((
                "minimum two",
                Box::new(BooleanQuery::union_with_minimum_required_clauses(
                    vec![
                        parser.parse_query("a")?,
                        parser.parse_query("\"a b\"")?,
                        parser.parse_query("\"b c\"")?,
                    ],
                    2,
                )),
            ));
            queries.push((
                "dismax",
                Box::new(DisjunctionMaxQuery::new(vec![
                    parser.parse_query("a")?,
                    parser.parse_query("\"a b\"")?,
                ])),
            ));
            queries.push((
                "exclusion",
                Box::new(BooleanQuery::new(vec![
                    (Occur::Should, parser.parse_query("a")?),
                    (Occur::Should, parser.parse_query("\"a b\"")?),
                    (Occur::MustNot, parser.parse_query("\"x x x\"")?),
                ])),
            ));
            for (expression, parsed) in queries {
                for boost in [0.0, 1.0, 2.5, -1.0] {
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
    fn test_negative_term_weight_disables_maxscore() {
        let weight = Bm25Weight::for_one_term(256, 1024, 1.0, Bm25Params::default());
        let docs: Vec<_> = (0..256).map(|doc| (doc, 1)).collect();
        let norms = vec![1; 256];
        let scorers = vec![
            TermScorer::create_for_test(&docs, &norms, weight.clone()),
            TermScorer::create_for_test(&docs, &norms, weight.boost_by(-1.0)),
        ];
        for mode in [DisjunctionPruning::Auto, DisjunctionPruning::BlockMaxScore] {
            let boolean = BooleanWeight::new(Vec::new(), true, Box::new(SumCombiner::default))
                .with_disjunction_pruning(mode);
            assert!(!boolean.should_use_block_maxscore(&scorers, 1024));
        }
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
}
