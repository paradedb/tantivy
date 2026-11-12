use std::any::{Any, TypeId};
use std::io;
use std::sync::Arc;

use common::BitSet;
use tantivy_fst::Automaton;

use super::bitset::or_term_info_into_bitset;
use super::phrase_prefix_query::prefix_end;
use super::{BitSetDocSet, BufferedUnionScorer};
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
        }
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

    /// Returns the term infos that match the automaton
    pub fn get_match_term_infos(&self, reader: &SegmentReader) -> crate::Result<Vec<TermInfo>> {
        let inverted_index = reader.inverted_index(self.field)?;
        let term_dict = inverted_index.terms();
        let mut term_stream = self.automaton_stream(term_dict)?;
        let mut term_infos = Vec::new();
        while term_stream.advance() {
            term_infos.push(term_stream.value().clone());
        }
        Ok(term_infos)
    }

    // only fuzzy, distance-based scores need per-term scores
    fn needs_per_term_scores(&self) -> bool {
        self.scoring_enabled && TypeId::of::<A>() == TypeId::of::<DfaWrapper>()
    }
}

impl<A> Weight for AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let max_doc = reader.max_doc();
        let inverted_index = reader.inverted_index(self.field)?;
        let term_dict = inverted_index.terms();
        let mut term_stream = self.automaton_stream(term_dict)?;
        if self.needs_per_term_scores() {
            // union of scorer-per-term
            let mut term_scorers = Vec::new();
            while let Some((_term, term_info, state)) = term_stream.next() {
                let term_score = automaton_score(self.automaton.as_ref(), state);
                let term_segment_postings = inverted_index
                    .read_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
                let term_scorer = ConstScorer::new(term_segment_postings, boost * term_score);
                term_scorers.push(term_scorer);
            }
            let scorer = BufferedUnionScorer::build(term_scorers, SumCombiner::default, max_doc);
            Ok(Box::new(scorer))
        } else {
            // bitmap of matched docs with const score
            let mut doc_bitset = BitSet::with_max_value(max_doc);
            let use_bmap = self.bitmap_enabled && reader.bitmap_postings_enabled;
            while term_stream.advance() {
                let term_info = term_stream.value();
                or_term_info_into_bitset(&inverted_index, term_info, &mut doc_bitset, use_bmap)?;
            }
            let doc_bitset = BitSetDocSet::from(doc_bitset);
            let scorer = ConstScorer::new(doc_bitset, boost);
            Ok(Box::new(scorer))
        }
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

fn automaton_score<A>(automaton: &A, state: A::State) -> f32
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    if TypeId::of::<DfaWrapper>() == automaton.type_id() && TypeId::of::<u32>() == state.type_id() {
        let dfa = automaton as *const A as *const DfaWrapper;
        let dfa = unsafe { &*dfa };

        let id = &state as *const A::State as *const u32;
        let id = unsafe { *id };

        let dist = dfa.0.distance(id).to_u8() as f32;
        1.0 / (1.0 + dist)
    } else {
        1.0
    }
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
}
