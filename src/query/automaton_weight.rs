use std::any::{Any, TypeId};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io;
use std::sync::Arc;

use tantivy_fst::Automaton;

use super::phrase_prefix_query::prefix_end;
use super::BufferedUnionScorer;
use crate::index::SegmentReader;
use crate::postings::TermInfo;
use crate::query::fuzzy_query::DfaWrapper;
use crate::query::score_combiner::SumCombiner;
use crate::query::{ConstScorer, Explanation, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption};
use crate::termdict::{TermDictionary, TermWithStateStreamer};
use crate::{DocId, Score, TantivyError};

/// A weight struct for Fuzzy Term and Regex Queries
pub struct AutomatonWeight<A> {
    field: Field,
    scoring_enabled: bool,
    bitmap_enabled: bool,
    automaton: Arc<A>,
    // For JSON fields, the term dictionary include terms from all paths.
    // We apply additional filtering based on the given JSON path, when searching within the term
    // dictionary. This prevents terms from unrelated paths from matching the search criteria.
    json_path_bytes: Option<Box<[u8]>>,
    // Maximum number of matching terms to expand to, per segment. `None` keeps every match.
    max_expansions: Option<usize>,
}

impl<A> AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    /// Create a new AutomationWeight
    pub fn new<IntoArcA: Into<Arc<A>>>(field: Field, automaton: IntoArcA) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            scoring_enabled: true,
            bitmap_enabled: false,
            automaton: automaton.into(),
            json_path_bytes: None,
            max_expansions: None,
        }
    }

    /// Create a new AutomationWeight for a json path
    pub fn new_for_json_path<IntoArcA: Into<Arc<A>>>(
        field: Field,
        automaton: IntoArcA,
        json_path_bytes: &[u8],
    ) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            scoring_enabled: true,
            bitmap_enabled: false,
            automaton: automaton.into(),
            json_path_bytes: Some(json_path_bytes.to_vec().into_boxed_slice()),
            max_expansions: None,
        }
    }

    /// Keeps at most `max_expansions` matching terms per segment. `None` (the default) keeps
    /// every match.
    ///
    /// For a fuzzy query (a Levenshtein DFA), the terms closest to the query term are kept, and
    /// terms at the same edit distance are kept in term order. Any other automaton carries no
    /// distance, so the first terms in term order are kept.
    pub fn with_max_expansions(mut self, max_expansions: Option<usize>) -> Self {
        self.max_expansions = max_expansions;
        self
    }

    pub(crate) fn with_bitmap_postings(mut self, enabled: bool) -> Self {
        self.bitmap_enabled = enabled;
        self
    }

    /// Enables score computation; disabling it permits membership-only posting readers.
    pub fn with_scoring_enabled(mut self, enabled: bool) -> Self {
        self.scoring_enabled = enabled;
        self
    }

    fn automaton_stream<'a>(
        &'a self,
        term_dict: &'a TermDictionary,
    ) -> io::Result<TermWithStateStreamer<'a, &'a A>> {
        let automaton: &A = &self.automaton;
        let mut term_stream_builder = term_dict.search_with_state(automaton);

        if let Some(json_path_bytes) = &self.json_path_bytes {
            term_stream_builder = term_stream_builder.ge(json_path_bytes);
            if let Some(end) = prefix_end(json_path_bytes) {
                term_stream_builder = term_stream_builder.lt(&end);
            }
        }

        term_stream_builder.interruptible().into_stream()
    }

    /// Returns the terms of `term_dict` that match the automaton, with the automaton state each
    /// one ended in, capped at `max_expansions`. The result is in term order.
    fn matching_terms(&self, term_dict: &TermDictionary) -> io::Result<Vec<(TermInfo, A::State)>> {
        let mut term_stream = self.automaton_stream(term_dict)?;

        let Some(max_expansions) = self.max_expansions else {
            let mut matches = Vec::new();
            while let Some((_term, term_info, state)) = term_stream.next() {
                matches.push((term_info.clone(), state));
            }
            return Ok(matches);
        };
        if max_expansions == 0 {
            return Ok(Vec::new());
        }

        // A max-heap ordered by (distance, term ordinal): its top is the worst match kept so far,
        // and a better candidate replaces it. Memory stays O(max_expansions) however many terms
        // the automaton matches, and no posting list is opened for a term that gets dropped.
        let mut kept: BinaryHeap<RankedMatch<A::State>> = BinaryHeap::with_capacity(max_expansions);
        let mut term_ord = 0;
        while let Some((_term, term_info, state)) = term_stream.next() {
            let distance = automaton_distance(self.automaton.as_ref(), &state).unwrap_or(0);
            let candidate = RankedMatch {
                distance,
                term_ord,
                term_info: term_info.clone(),
                state,
            };
            term_ord += 1;

            if kept.len() < max_expansions {
                kept.push(candidate);
            } else if let Some(mut worst) = kept.peek_mut() {
                // Terms arrive in term order, so a candidate at the same distance as the worst
                // match has a larger ordinal and loses the tie: the earlier term is kept.
                if candidate < *worst {
                    // Dropping `worst` (a `PeekMut`) restores the heap order.
                    *worst = candidate;
                }
            }
            // Once every kept match is at distance 0, no later term can beat one. For a
            // non-fuzzy automaton every match is distance 0, so this stops after the first
            // `max_expansions` terms.
            if kept.len() == max_expansions && kept.peek().is_some_and(|worst| worst.distance == 0)
            {
                break;
            }
        }

        let mut kept = kept.into_vec();
        kept.sort_unstable_by_key(|ranked| ranked.term_ord);
        Ok(kept
            .into_iter()
            .map(|ranked| (ranked.term_info, ranked.state))
            .collect())
    }

    /// Returns the term infos that match the automaton, capped at `max_expansions`.
    pub fn get_match_term_infos(&self, reader: &SegmentReader) -> crate::Result<Vec<TermInfo>> {
        let inverted_index = reader.inverted_index(self.field)?;
        let term_infos = self
            .matching_terms(inverted_index.terms())?
            .into_iter()
            .map(|(term_info, _state)| term_info)
            .collect();
        Ok(term_infos)
    }
}

/// A matching term competing for one of the `max_expansions` slots. Ordered by edit distance,
/// then by position in the term dictionary: the smaller, the better.
struct RankedMatch<S> {
    distance: u8,
    term_ord: usize,
    term_info: TermInfo,
    state: S,
}

impl<S> RankedMatch<S> {
    fn rank(&self) -> (u8, usize) {
        (self.distance, self.term_ord)
    }
}

impl<S> PartialEq for RankedMatch<S> {
    fn eq(&self, other: &Self) -> bool {
        self.rank() == other.rank()
    }
}

impl<S> Eq for RankedMatch<S> {}

impl<S> PartialOrd for RankedMatch<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S> Ord for RankedMatch<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank())
    }
}

impl<A> Weight for AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let inverted_index = reader.inverted_index(self.field)?;

        let mut scorers = vec![];
        let mut bitmaps: Vec<Box<dyn Scorer>> = Vec::new();
        for (term_info, state) in self.matching_terms(inverted_index.terms())? {
            if self.bitmap_enabled && !self.scoring_enabled && reader.bitmap_postings_enabled {
                if let Some(bitmap) =
                    inverted_index.read_bitmap_from_terminfo(&term_info, reader.max_doc())?
                {
                    bitmaps.push(Box::new(ConstScorer::new(bitmap, boost)));
                    continue;
                }
            }
            let score = automaton_score(self.automaton.as_ref(), &state);
            let segment_postings =
                inverted_index.read_postings_from_terminfo(&term_info, IndexRecordOption::Basic)?;
            let scorer = ConstScorer::new(segment_postings, boost * score);
            scorers.push(scorer);
        }

        if !bitmaps.is_empty() {
            bitmaps.extend(
                scorers
                    .into_iter()
                    .map(|scorer| Box::new(scorer) as Box<dyn Scorer>),
            );
            return Ok(Box::new(super::bitmap_combination::BitmapCombination::new(
                bitmaps,
                super::bitmap_combination::BitmapOperation::Union,
                reader.max_doc(),
            )));
        }

        let scorer = BufferedUnionScorer::build(scorers, SumCombiner::default, reader.max_doc());
        Ok(Box::new(scorer))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) == doc {
            Ok(Explanation::new("AutomatonScorer", scorer.score()))
        } else {
            Err(TantivyError::InvalidArgument(
                "Document does not exist".to_string(),
            ))
        }
    }
}

/// The edit distance a matching term ended at, when `automaton` is a Levenshtein DFA (a fuzzy
/// query). `None` for any other automaton, whose matches carry no notion of distance.
fn automaton_distance<A>(automaton: &A, state: &A::State) -> Option<u8>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    if TypeId::of::<DfaWrapper>() == <A as Any>::type_id(automaton)
        && TypeId::of::<u32>() == <A::State as Any>::type_id(state)
    {
        // SAFETY: the `TypeId` checks above prove that `A` is `DfaWrapper` and `A::State` is
        // `u32`, so both casts are to the value's own type.
        let dfa = unsafe { &*(automaton as *const A as *const DfaWrapper) };
        let id = unsafe { *(state as *const A::State as *const u32) };
        Some(dfa.0.distance(id).to_u8())
    } else {
        None
    }
}

fn automaton_score<A>(automaton: &A, state: &A::State) -> f32
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    automaton_distance(automaton, state).map_or(1.0, |dist| 1.0 / (1.0 + dist as f32))
}

#[cfg(test)]
mod tests {
    use tantivy_fst::Automaton;

    use super::AutomatonWeight;
    use crate::docset::TERMINATED;
    use crate::query::Weight;
    use crate::schema::{Schema, STRING};
    use crate::{Index, IndexWriter};

    fn create_index() -> crate::Result<Index> {
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(title=>"abc"))?;
        index_writer.add_document(doc!(title=>"bcd"))?;
        index_writer.add_document(doc!(title=>"abcd"))?;
        index_writer.commit()?;
        Ok(index)
    }

    #[derive(Clone, Copy)]
    enum State {
        Start,
        NotMatching,
        AfterA,
    }

    struct PrefixedByA;

    impl Automaton for PrefixedByA {
        type State = State;

        fn start(&self) -> Self::State {
            State::Start
        }

        fn is_match(&self, state: &Self::State) -> bool {
            matches!(*state, State::AfterA)
        }

        fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
            match *state {
                State::Start => {
                    if byte == b'a' {
                        State::AfterA
                    } else {
                        State::NotMatching
                    }
                }
                State::AfterA => State::AfterA,
                State::NotMatching => State::NotMatching,
            }
        }
    }

    #[test]
    fn test_automaton_weight() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), 2u32);
        assert_eq!(scorer.doc(), 2u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_boost() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.32)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.32);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_max_expansions_keeps_first_terms() -> crate::Result<()> {
        // `PrefixedByA` matches `abc` (doc 0) and `abcd` (doc 2), and carries no edit distance,
        // so a cap keeps the first terms in term order.
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0u32);

        let uncapped = AutomatonWeight::new(field, PrefixedByA);
        assert_eq!(uncapped.get_match_term_infos(segment_reader)?.len(), 2);

        let capped = AutomatonWeight::new(field, PrefixedByA).with_max_expansions(Some(1));
        assert_eq!(capped.get_match_term_infos(segment_reader)?.len(), 1);
        let mut scorer = capped.scorer(segment_reader, 1.0)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_max_expansions_zero() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0u32);

        let capped = AutomatonWeight::new(field, PrefixedByA).with_max_expansions(Some(0));
        assert!(capped.get_match_term_infos(segment_reader)?.is_empty());
        let scorer = capped.scorer(segment_reader, 1.0)?;
        assert_eq!(scorer.doc(), TERMINATED);
        Ok(())
    }
}
