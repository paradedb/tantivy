use std::any::{Any, TypeId};
use std::io;
use std::sync::Arc;

use tantivy_fst::Automaton;

use super::phrase_prefix_query::prefix_end;
use super::BufferedUnionScorer;
use crate::index::SegmentReader;
use crate::postings::TermInfo;
use crate::query::fuzzy_query::DfaWrapper;
use crate::query::query_estimate::{bounded_prefix_stream, estimate_term_union, EstimationBudget};
use crate::query::score_combiner::SumCombiner;
use crate::query::{ConstScorer, Explanation, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption};
use crate::termdict::{TermDictionary, TermWithStateStreamer};
use crate::{DocId, Score, TantivyError};

/// A weight struct for Fuzzy Term and Regex Queries
pub struct AutomatonWeight<A> {
    field: Field,
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
            automaton: automaton.into(),
            json_path_bytes: Some(json_path_bytes.to_vec().into_boxed_slice()),
        }
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

        term_stream_builder.into_stream()
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

    /// Estimates the union of matching term frequencies within fixed candidate and byte budgets;
    /// cost sums their frequencies without opening postings or automaton-filtered streams.
    pub(crate) fn estimate_docs(
        &self,
        reader: &SegmentReader,
        remaining_terms: &mut usize,
        budget: &mut EstimationBudget,
    ) -> crate::Result<Option<(u32, u64)>> {
        let inverted_index = reader.inverted_index(self.field)?;
        self.estimate_dictionary(
            inverted_index.terms(),
            reader.max_doc(),
            remaining_terms,
            budget,
        )
    }

    fn estimate_dictionary(
        &self,
        dictionary: &TermDictionary,
        max_doc: u32,
        remaining_terms: &mut usize,
        budget: &mut EstimationBudget,
    ) -> crate::Result<Option<(u32, u64)>> {
        let mut prefix = automaton_prefix(self.automaton.as_ref());
        if let Some(json_path) = &self.json_path_bytes {
            if json_path.starts_with(&prefix) {
                prefix = json_path.to_vec();
            } else if !prefix.starts_with(json_path) {
                return Ok(Some((0, 0)));
            }
        }
        let Some(mut stream) =
            bounded_prefix_stream(dictionary, &prefix, budget.remaining_terms + 1, budget)?
        else {
            // The dictionary range exceeds the payload read budget.
            return Ok(None);
        };
        let mut frequencies = Vec::new();
        while stream.advance() {
            if !budget.consume(stream.key()) {
                // Unexamined candidates may match, even if every examined term was a nonmatch.
                return Ok(None);
            }
            let mut state = self.automaton.start();
            for &byte in stream.key() {
                state = self.automaton.accept(&state, byte);
                if !self.automaton.can_match(&state) {
                    break;
                }
            }
            if !self.automaton.is_match(&state) {
                continue;
            }
            if *remaining_terms == 0 {
                // The query's expansion limit was exceeded.
                return Ok(None);
            }
            *remaining_terms -= 1;
            frequencies.push(stream.value().doc_freq);
        }
        Ok(Some(estimate_term_union(&frequencies, max_doc)))
    }
}

/// Extracts up to 64 mandatory prefix bytes using only the automaton, independent of index size.
fn automaton_prefix(automaton: &impl Automaton) -> Vec<u8> {
    let mut prefix = Vec::new();
    let mut state = automaton.start();
    while prefix.len() < 64 && !automaton.is_match(&state) {
        let mut next = None;
        for byte in 0..=u8::MAX {
            let next_state = automaton.accept(&state, byte);
            if automaton.can_match(&next_state) {
                if next.is_some() {
                    return prefix;
                }
                next = Some((byte, next_state));
            }
        }
        let Some((byte, next_state)) = next else {
            break;
        };
        prefix.push(byte);
        state = next_state;
    }
    prefix
}

impl<A> Weight for AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let inverted_index = reader.inverted_index(self.field)?;
        let term_dict = inverted_index.terms();
        let mut term_stream = self.automaton_stream(term_dict)?;

        let mut scorers = vec![];
        while let Some((_term, term_info, state)) = term_stream.next() {
            let score = automaton_score(self.automaton.as_ref(), state);
            let segment_postings =
                inverted_index.read_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
            let scorer = ConstScorer::new(segment_postings, boost * score);
            scorers.push(scorer);
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
    use std::ops::Range;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use common::{HasLen, OwnedBytes};
    use tantivy_fst::Automaton;

    use super::AutomatonWeight;
    use crate::directory::{FileHandle, FileSlice};
    use crate::docset::TERMINATED;
    use crate::postings::TermInfo;
    use crate::query::query_estimate::{EstimationBudget, MAX_ESTIMATED_TERMS};
    use crate::query::Weight;
    use crate::schema::{Schema, STRING};
    use crate::termdict::{TermDictionary, TermDictionaryBuilder};
    use crate::{Index, IndexWriter};

    #[derive(Debug)]
    struct CountReads {
        data: OwnedBytes,
        bytes: Arc<AtomicUsize>,
    }

    impl HasLen for CountReads {
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    impl FileHandle for CountReads {
        fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
            self.bytes.fetch_add(range.len(), Ordering::Relaxed);
            Ok(self.data.slice(range))
        }
    }

    struct CountTransitions<A> {
        inner: A,
        transitions: Arc<AtomicUsize>,
    }

    impl<A: Automaton> Automaton for CountTransitions<A> {
        type State = A::State;
        fn start(&self) -> Self::State {
            self.inner.start()
        }
        fn is_match(&self, state: &Self::State) -> bool {
            self.inner.is_match(state)
        }
        fn can_match(&self, state: &Self::State) -> bool {
            self.inner.can_match(state)
        }
        fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
            self.transitions.fetch_add(1, Ordering::Relaxed);
            self.inner.accept(state, byte)
        }
    }

    #[test]
    fn metadata_estimates_bound_nonmatches_and_dictionary_reads() -> crate::Result<()> {
        let field = Schema::builder().add_text_field("text", STRING);
        let mut measurements = Vec::new();
        for num_terms in [2_000, 20_000] {
            let mut builder = TermDictionaryBuilder::create(Vec::new())?;
            for id in 0..num_terms {
                builder.insert(
                    format!("token{id:08}"),
                    &TermInfo {
                        doc_freq: 1,
                        postings_range: id * 100..(id + 1) * 100,
                        positions_range: id * 200..(id + 1) * 200,
                        pnorms_offset: None,
                    },
                )?;
            }
            let bytes = Arc::new(AtomicUsize::new(0));
            let data = builder.finish()?;
            let dictionary_len = data.len();
            let dictionary = TermDictionary::open(FileSlice::new(Arc::new(CountReads {
                data: OwnedBytes::new(data),
                bytes: bytes.clone(),
            })))?;
            let mut results = Vec::new();
            for pattern in [".*", ".*absent"] {
                bytes.store(0, Ordering::Relaxed);
                let transitions = Arc::new(AtomicUsize::new(0));
                let weight = AutomatonWeight::new(
                    field,
                    CountTransitions {
                        inner: tantivy_fst::Regex::new(pattern).unwrap(),
                        transitions: transitions.clone(),
                    },
                );
                let mut budget = EstimationBudget::default();
                budget.remaining_terms = 32;
                let mut expansions = MAX_ESTIMATED_TERMS;
                assert_eq!(
                    weight.estimate_dictionary(&dictionary, 1, &mut expansions, &mut budget)?,
                    None
                );
                let work = transitions.load(Ordering::Relaxed);
                assert!(work <= 32 * 13 + 256);
                let read_bytes = bytes.load(Ordering::Relaxed);
                if num_terms == 20_000 {
                    assert!(read_bytes < dictionary_len / 2);
                }
                results.push((work, read_bytes));
            }
            measurements.push(results);
        }
        for (small, large) in measurements[0].iter().zip(&measurements[1]) {
            assert_eq!(small.0, large.0);
            assert!(large.1 <= small.1 + 16_384);
        }
        Ok(())
    }

    #[test]
    fn metadata_estimates_bound_long_term_processing() -> crate::Result<()> {
        let field = Schema::builder().add_text_field("text", STRING);
        let mut builder = TermDictionaryBuilder::create(Vec::new())?;
        for id in 0..32 {
            builder.insert(
                format!("{id:02}{}", "x".repeat(65_000)),
                &TermInfo {
                    doc_freq: 1,
                    ..TermInfo::default()
                },
            )?;
        }
        let dictionary = TermDictionary::open(FileSlice::from(builder.finish()?))?;
        let transitions = Arc::new(AtomicUsize::new(0));
        let weight = AutomatonWeight::new(
            field,
            CountTransitions {
                inner: tantivy_fst::Regex::new(".*").unwrap(),
                transitions: transitions.clone(),
            },
        );
        let mut expansions = MAX_ESTIMATED_TERMS;
        assert_eq!(
            weight.estimate_dictionary(
                &dictionary,
                1,
                &mut expansions,
                &mut EstimationBudget::default()
            )?,
            None
        );
        assert!(transitions.load(Ordering::Relaxed) <= 1 << 20);
        Ok(())
    }

    #[cfg(feature = "quickwit")]
    #[test]
    fn metadata_estimates_reject_large_payload_before_reading() -> crate::Result<()> {
        use rand::{RngCore, SeedableRng};

        let field = Schema::builder().add_text_field("text", STRING);
        let mut random = rand::rngs::StdRng::seed_from_u64(42);
        let mut builder = TermDictionaryBuilder::create(Vec::new())?;
        for id in 0..32u8 {
            let mut key = vec![0u8; 65_000];
            random.fill_bytes(&mut key);
            key[0] = id;
            builder.insert(
                key,
                &TermInfo {
                    doc_freq: 1,
                    ..TermInfo::default()
                },
            )?;
        }
        let bytes = Arc::new(AtomicUsize::new(0));
        let dictionary = TermDictionary::open(FileSlice::new(Arc::new(CountReads {
            data: OwnedBytes::new(builder.finish()?),
            bytes: bytes.clone(),
        })))?;
        bytes.store(0, Ordering::Relaxed);
        let transitions = Arc::new(AtomicUsize::new(0));
        let weight = AutomatonWeight::new(
            field,
            CountTransitions {
                inner: tantivy_fst::Regex::new(".*").unwrap(),
                transitions: transitions.clone(),
            },
        );
        let mut expansions = MAX_ESTIMATED_TERMS;
        assert_eq!(
            weight.estimate_dictionary(
                &dictionary,
                1,
                &mut expansions,
                &mut EstimationBudget::default()
            )?,
            None
        );
        assert_eq!(transitions.load(Ordering::Relaxed), 0);
        assert!(bytes.load(Ordering::Relaxed) < 1 << 20);
        Ok(())
    }

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
