use levenshtein_automata::{Distance, LevenshteinAutomatonBuilder, DFA};
use once_cell::sync::OnceCell;
use tantivy_fst::Automaton;

use crate::query::{AutomatonWeight, EnableScoring, Query, Weight};
use crate::schema::{Term, Type};
use crate::TantivyError::InvalidArgument;

pub(crate) struct DfaWrapper(pub DFA);

impl Automaton for DfaWrapper {
    type State = u32;

    fn start(&self) -> Self::State {
        self.0.initial_state()
    }

    fn is_match(&self, state: &Self::State) -> bool {
        match self.0.distance(*state) {
            Distance::Exact(_) => true,
            Distance::AtLeast(_) => false,
        }
    }

    fn can_match(&self, state: &u32) -> bool {
        *state != levenshtein_automata::SINK_STATE
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        self.0.transition(*state, byte)
    }
}

/// A Fuzzy Query matches all of the documents
/// containing a specific term that is within
/// Levenshtein distance
/// ```rust
/// use tantivy::collector::{Count, TopDocs};
/// use tantivy::query::FuzzyTermQuery;
/// use tantivy::schema::{Schema, TEXT};
/// use tantivy::{doc, Index, IndexWriter, Term};
///
/// fn example() -> tantivy::Result<()> {
///     let mut schema_builder = Schema::builder();
///     let title = schema_builder.add_text_field("title", TEXT);
///     let schema = schema_builder.build();
///     let index = Index::create_in_ram(schema);
///     {
///         let mut index_writer: IndexWriter = index.writer(15_000_000)?;
///         index_writer.add_document(doc!(
///             title => "The Name of the Wind",
///         ))?;
///         index_writer.add_document(doc!(
///             title => "The Diary of Muadib",
///         ))?;
///         index_writer.add_document(doc!(
///             title => "A Dairy Cow",
///         ))?;
///         index_writer.add_document(doc!(
///             title => "The Diary of a Young Girl",
///         ))?;
///         index_writer.commit()?;
///     }
///     let reader = index.reader()?;
///     let searcher = reader.searcher();
///
///     {
///         let term = Term::from_field_text(title, "Diary");
///         let query = FuzzyTermQuery::new(term, 1, true);
///         let (top_docs, count) = searcher.search(&query, &(TopDocs::with_limit(2).order_by_score(), Count)).unwrap();
///         assert_eq!(count, 2);
///         assert_eq!(top_docs.len(), 2);
///     }
///
///     Ok(())
/// }
/// # assert!(example().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct FuzzyTermQuery {
    /// What term are we searching
    term: Term,
    /// How many changes are we going to allow
    distance: u8,
    /// Should a transposition cost 1 or 2?
    transposition_cost_one: bool,
    /// is a starts with query
    prefix: bool,
    /// Maximum number of matching terms to expand to, per segment. `None` means unlimited.
    max_expansions: Option<u32>,
}

impl FuzzyTermQuery {
    /// Creates a new Fuzzy Query
    pub fn new(term: Term, distance: u8, transposition_cost_one: bool) -> FuzzyTermQuery {
        FuzzyTermQuery {
            term,
            distance,
            transposition_cost_one,
            prefix: false,
            max_expansions: None,
        }
    }

    /// Creates a new Fuzzy Query of the Term prefix
    pub fn new_prefix(term: Term, distance: u8, transposition_cost_one: bool) -> FuzzyTermQuery {
        FuzzyTermQuery {
            term,
            distance,
            transposition_cost_one,
            prefix: true,
            max_expansions: None,
        }
    }

    /// Caps the number of terms this query expands to.
    ///
    /// A fuzzy query matches every term of the term dictionary within the Levenshtein distance,
    /// and opens one posting list per matching term. On a large dictionary that can be thousands
    /// of terms. With a cap, only the `max_expansions` terms closest to the query term are kept:
    /// smaller edit distance first, and terms at the same distance in term order.
    ///
    /// The cap applies per segment, as each segment has its own term dictionary. A cap of `0`
    /// matches nothing.
    pub fn set_max_expansions(&mut self, max_expansions: u32) {
        self.max_expansions = Some(max_expansions);
    }

    fn specialized_weight(&self) -> crate::Result<AutomatonWeight<DfaWrapper>> {
        static AUTOMATON_BUILDER: [[OnceCell<LevenshteinAutomatonBuilder>; 2]; 3] = [
            [OnceCell::new(), OnceCell::new()],
            [OnceCell::new(), OnceCell::new()],
            [OnceCell::new(), OnceCell::new()],
        ];

        let automaton_builder = AUTOMATON_BUILDER
            .get(self.distance as usize)
            .ok_or_else(|| {
                InvalidArgument(format!(
                    "Levenshtein distance of {} is not allowed. Choose a value less than {}",
                    self.distance,
                    AUTOMATON_BUILDER.len()
                ))
            })?
            .get(self.transposition_cost_one as usize)
            .unwrap()
            .get_or_init(|| {
                LevenshteinAutomatonBuilder::new(self.distance, self.transposition_cost_one)
            });

        let term_value = self.term.value();

        let term_text = if term_value.typ() == Type::Json {
            if let Some(json_path_type) = term_value.json_path_type() {
                if json_path_type != Type::Str {
                    return Err(InvalidArgument(format!(
                        "The fuzzy term query requires a string path type for a json term. Found \
                         {json_path_type:?}"
                    )));
                }
            }

            std::str::from_utf8(self.term.serialized_value_bytes()).map_err(|_| {
                InvalidArgument(
                    "Failed to convert json term value bytes to utf8 string.".to_string(),
                )
            })?
        } else {
            term_value.as_str().ok_or_else(|| {
                InvalidArgument("The fuzzy term query requires a string term.".to_string())
            })?
        };
        let automaton = if self.prefix {
            automaton_builder.build_prefix_dfa(term_text)
        } else {
            automaton_builder.build_dfa(term_text)
        };

        let weight = if let Some((json_path_bytes, _)) = term_value.as_json() {
            AutomatonWeight::new_for_json_path(
                self.term.field(),
                DfaWrapper(automaton),
                json_path_bytes,
            )
        } else {
            AutomatonWeight::new(self.term.field(), DfaWrapper(automaton))
        };
        Ok(weight.with_max_expansions(self.max_expansions.map(|n| n as usize)))
    }
}

impl Query for FuzzyTermQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(
            self.specialized_weight()?
                .with_scoring_enabled(enable_scoring.is_scoring_enabled())
                .with_bitmap_postings(enable_scoring.bitmap_postings_enabled()),
        ))
    }
}

#[cfg(test)]
mod test {
    use super::FuzzyTermQuery;
    use crate::collector::{Count, DocSetCollector, TopDocs};
    use crate::indexer::NoMergePolicy;
    use crate::query::QueryParser;
    use crate::schema::{Schema, STORED, TEXT};
    use crate::{assert_nearly_equals, Index, IndexWriter, TantivyDocument, Term};

    #[test]
    pub fn test_fuzzy_json_path() -> crate::Result<()> {
        // # Defining the schema
        let mut schema_builder = Schema::builder();
        let attributes = schema_builder.add_json_field("attributes", TEXT | STORED);
        let schema = schema_builder.build();

        // # Indexing documents
        let index = Index::create_in_ram(schema.clone());

        let mut index_writer = index.writer_for_tests()?;
        index_writer.set_merge_policy(Box::new(NoMergePolicy));
        let doc = TantivyDocument::parse_json(
            &schema,
            r#"{
            "attributes": {
                "a": "japan"
            }
        }"#,
        )?;
        index_writer.add_document(doc)?;
        let doc = TantivyDocument::parse_json(
            &schema,
            r#"{
            "attributes": {
                "aa": "japan"
            }
        }"#,
        )?;
        index_writer.add_document(doc)?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // # Fuzzy search
        let query_parser = QueryParser::for_index(&index, vec![attributes]);

        let get_json_path_term = |query: &str| -> crate::Result<Term> {
            let query = query_parser.parse_query(query)?;
            let mut terms = Vec::new();
            query.query_terms(attributes, searcher.segment_reader(0), &mut |term, _| {
                terms.push(term.clone());
            });

            Ok(terms[0].clone())
        };

        // shall not match the first document due to json path mismatch
        {
            let term = get_json_path_term("attributes.aa:japan")?;
            let fuzzy_query = FuzzyTermQuery::new(term, 2, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            assert_eq!(top_docs[0].1.doc_id, 1, "Expected the second document");
        }

        // shall match the first document because Levenshtein distance is 1 (substitute 'o' with
        // 'a')
        {
            let term = get_json_path_term("attributes.a:japon")?;

            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            assert_eq!(top_docs[0].1.doc_id, 0, "Expected the first document");
        }

        // shall not match because non-prefix Levenshtein distance is more than 1 (add 'a' and 'n')
        {
            let term = get_json_path_term("attributes.a:jap")?;

            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 0, "Expected no document");
        }

        Ok(())
    }

    #[test]
    pub fn test_fuzzy_term() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(
                country_field => "japan",
            ))?;
            index_writer.add_document(doc!(
                country_field => "korea",
            ))?;
            index_writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();

        // passes because Levenshtein distance is 1 (substitute 'o' with 'a')
        {
            let term = Term::from_field_text(country_field, "japon");
            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            let (score, _) = top_docs[0];
            assert_nearly_equals!(0.5, score);
        }

        // fails because non-prefix Levenshtein distance is more than 1 (add 'a' and 'n')
        {
            let term = Term::from_field_text(country_field, "jap");

            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 0, "Expected no document");
        }

        // passes because prefix Levenshtein distance is 0
        {
            let term = Term::from_field_text(country_field, "jap");
            let fuzzy_query = FuzzyTermQuery::new_prefix(term, 1, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            let (score, _) = top_docs[0];
            assert_nearly_equals!(1.0, score);
        }
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_term_transposition_cost_one() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(country_field => "japan"))?;
        index_writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let term_jaapn = Term::from_field_text(country_field, "jaapn");
        {
            let fuzzy_query_transposition = FuzzyTermQuery::new(term_jaapn.clone(), 1, true);
            let count = searcher.search(&fuzzy_query_transposition, &Count)?;
            assert_eq!(count, 1);
        }
        {
            let fuzzy_query_transposition = FuzzyTermQuery::new(term_jaapn, 1, false);
            let count = searcher.search(&fuzzy_query_transposition, &Count)?;
            assert_eq!(count, 0);
        }
        Ok(())
    }

    /// Returns the sorted ids of the documents a fuzzy query for `japan` (distance 2) matches,
    /// with the given `max_expansions`, over one segment holding one term per document:
    ///
    /// | doc | term    | distance to `japan` |
    /// |-----|---------|---------------------|
    /// | 0   | `japan` | 0                   |
    /// | 1   | `japon` | 1                   |
    /// | 2   | `jaban` | 1                   |
    /// | 3   | `jason` | 2                   |
    /// | 4   | `korea` | no match            |
    fn fuzzy_japan_doc_ids(max_expansions: Option<u32>) -> crate::Result<Vec<u32>> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        for country in ["japan", "japon", "jaban", "jason", "korea"] {
            index_writer.add_document(doc!(country_field => country))?;
        }
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);

        let mut fuzzy_query =
            FuzzyTermQuery::new(Term::from_field_text(country_field, "japan"), 2, true);
        if let Some(max_expansions) = max_expansions {
            fuzzy_query.set_max_expansions(max_expansions);
        }
        let mut doc_ids: Vec<u32> = searcher
            .search(&fuzzy_query, &DocSetCollector)?
            .into_iter()
            .map(|doc_address| doc_address.doc_id)
            .collect();
        doc_ids.sort_unstable();
        Ok(doc_ids)
    }

    #[test]
    pub fn test_fuzzy_max_expansions_unset_keeps_every_match() -> crate::Result<()> {
        assert_eq!(fuzzy_japan_doc_ids(None)?, vec![0, 1, 2, 3]);
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_max_expansions_keeps_closest_terms() -> crate::Result<()> {
        // Only the exact term.
        assert_eq!(fuzzy_japan_doc_ids(Some(1))?, vec![0]);
        // Both distance-1 terms, but not the distance-2 `jason`.
        assert_eq!(fuzzy_japan_doc_ids(Some(3))?, vec![0, 1, 2]);
        // A cap above the match count keeps every match.
        assert_eq!(fuzzy_japan_doc_ids(Some(50))?, vec![0, 1, 2, 3]);
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_max_expansions_breaks_ties_in_term_order() -> crate::Result<()> {
        // `jaban` and `japon` are both at distance 1, and only one slot is left after `japan`:
        // `jaban` comes first in term order, so it is kept.
        assert_eq!(fuzzy_japan_doc_ids(Some(2))?, vec![0, 2]);
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_max_expansions_zero_matches_nothing() -> crate::Result<()> {
        assert_eq!(fuzzy_japan_doc_ids(Some(0))?, Vec::<u32>::new());
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_max_expansions_applies_per_segment() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.set_merge_policy(Box::new(NoMergePolicy));
        // Two segments: `japan` in the first, `japon` in the second.
        index_writer.add_document(doc!(country_field => "japan"))?;
        index_writer.commit()?;
        index_writer.add_document(doc!(country_field => "japon"))?;
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 2);

        // Each segment keeps its own closest term, so both documents match even with a cap of 1.
        let mut fuzzy_query =
            FuzzyTermQuery::new(Term::from_field_text(country_field, "japan"), 1, true);
        fuzzy_query.set_max_expansions(1);
        assert_eq!(searcher.search(&fuzzy_query, &Count)?, 2);
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_max_expansions_keeps_scores() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(country_field => "japan"))?;
        index_writer.add_document(doc!(country_field => "japon"))?;
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();

        // A capped query scores the terms it keeps exactly as an uncapped one: 1 / (1 + distance).
        let mut fuzzy_query =
            FuzzyTermQuery::new(Term::from_field_text(country_field, "japan"), 1, true);
        fuzzy_query.set_max_expansions(2);
        let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
        assert_eq!(top_docs.len(), 2);
        assert_nearly_equals!(1.0, top_docs[0].0);
        assert_nearly_equals!(0.5, top_docs[1].0);
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_max_expansions_prefix() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        // `jab...` is 1 edit away from the prefix `jap`, `jap...` is 0 edits away.
        index_writer.add_document(doc!(country_field => "jabber"))?;
        index_writer.add_document(doc!(country_field => "japan"))?;
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();

        let mut fuzzy_query =
            FuzzyTermQuery::new_prefix(Term::from_field_text(country_field, "jap"), 1, true);
        assert_eq!(searcher.search(&fuzzy_query, &Count)?, 2);
        fuzzy_query.set_max_expansions(1);
        let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
        assert_eq!(top_docs.len(), 1);
        assert_eq!(
            top_docs[0].1.doc_id, 1,
            "expected `japan`, the closer prefix match"
        );
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_max_expansions_json_path() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let attributes = schema_builder.add_json_field("attributes", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        for json in [
            r#"{"attributes": {"a": "japon"}}"#, // doc 0: path `a`, distance 1
            r#"{"attributes": {"b": "japan"}}"#, // doc 1: path `b`, out of the query's path
            r#"{"attributes": {"a": "japan"}}"#, // doc 2: path `a`, distance 0
        ] {
            index_writer.add_document(TantivyDocument::parse_json(&schema, json)?)?;
        }
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();

        let query =
            QueryParser::for_index(&index, vec![attributes]).parse_query("attributes.a:japan")?;
        let mut term = None;
        query.query_terms(attributes, searcher.segment_reader(0), &mut |t, _| {
            term = Some(t.clone())
        });
        // Only terms under path `a` compete for the slot: `japan` under `b` must not take it.
        let mut fuzzy_query = FuzzyTermQuery::new(term.unwrap(), 1, true);
        assert_eq!(
            searcher.search(&fuzzy_query, &Count)?,
            2,
            "uncapped: both path `a` docs"
        );
        fuzzy_query.set_max_expansions(1);
        let doc_ids: Vec<u32> = searcher
            .search(&fuzzy_query, &DocSetCollector)?
            .into_iter()
            .map(|doc_address| doc_address.doc_id)
            .collect();
        assert_eq!(doc_ids, vec![2]);
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_max_expansions_bitmap_postings() -> crate::Result<()> {
        use crate::query::{EnableScoring, Query};

        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field(
            "country",
            TEXT.set_indexing_options(
                TEXT.get_indexing_options()
                    .unwrap()
                    .clone()
                    .set_bitmap_postings(true),
            ),
        );
        let index = Index::create_in_ram(schema_builder.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        // 200 documents per term: dense enough to be stored as bitmaps.
        for doc in 0..800 {
            let country = ["japan", "japon", "jaban", "jason"][doc % 4];
            index_writer.add_document(doc!(country_field => country))?;
        }
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();
        let segment_reader = searcher.segment_reader(0);

        let count_with_cap = |max_expansions: Option<u32>| -> crate::Result<(u32, bool)> {
            let mut fuzzy_query =
                FuzzyTermQuery::new(Term::from_field_text(country_field, "japan"), 2, true);
            if let Some(max_expansions) = max_expansions {
                fuzzy_query.set_max_expansions(max_expansions);
            }
            // Scoring disabled with bitmap postings on: the path `Count` takes.
            let weight = fuzzy_query.weight(
                EnableScoring::disabled_from_searcher(&searcher).with_bitmap_postings(true),
            )?;
            let uses_bitmaps = weight.scorer(segment_reader, 1.0)?.has_fast_bitset();
            Ok((weight.count(segment_reader)?, uses_bitmaps))
        };

        assert_eq!(count_with_cap(None)?, (800, true));
        assert_eq!(count_with_cap(Some(1))?, (200, true));
        assert_eq!(count_with_cap(Some(3))?, (600, true));
        Ok(())
    }

    /// Plain Levenshtein distance (a transposition costs 2), the reference for the DFA.
    fn levenshtein(a: &str, b: &str) -> u8 {
        let b: Vec<char> = b.chars().collect();
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        for (i, ca) in a.chars().enumerate() {
            let mut cur = vec![i + 1; b.len() + 1];
            for (j, cb) in b.iter().enumerate() {
                let substitution = prev[j] + usize::from(ca != *cb);
                cur[j + 1] = substitution.min(prev[j + 1] + 1).min(cur[j] + 1);
            }
            prev = cur;
        }
        prev[b.len()] as u8
    }

    /// Compares capped fuzzy queries on random dictionaries against a brute-force reference: the
    /// terms within the distance, sorted by (distance, term), truncated to `max_expansions`.
    #[test]
    pub fn test_fuzzy_max_expansions_matches_reference() -> crate::Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let random_word = |rng: &mut StdRng| -> String {
            let len = rng.random_range(1..=5);
            (0..len)
                .map(|_| ['a', 'b', 'c'][rng.random_range(0..3)])
                .collect()
        };

        for seed in 0..100u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let words: Vec<String> = (0..40).map(|_| random_word(&mut rng)).collect();

            let mut schema_builder = Schema::builder();
            let field = schema_builder.add_text_field("word", TEXT);
            let index = Index::create_in_ram(schema_builder.build());
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            for word in &words {
                index_writer.add_document(doc!(field => word.as_str()))?;
            }
            index_writer.commit()?;
            let searcher = index.reader()?.searcher();

            for _ in 0..10 {
                let query_word = random_word(&mut rng);
                let distance = rng.random_range(0..=2u8);
                let max_expansions = rng.random_range(0..=6u32);

                // The reference: distinct matching terms, closest first, then in term order.
                let mut matching: Vec<(u8, &str)> = words
                    .iter()
                    .map(|word| (levenshtein(&query_word, word), word.as_str()))
                    .filter(|(dist, _)| *dist <= distance)
                    .collect();
                matching.sort_unstable();
                matching.dedup();
                matching.truncate(max_expansions as usize);
                let kept: Vec<&str> = matching.iter().map(|(_, word)| *word).collect();
                let expected: Vec<u32> = (0..words.len() as u32)
                    .filter(|doc| kept.contains(&words[*doc as usize].as_str()))
                    .collect();

                let mut fuzzy_query =
                    FuzzyTermQuery::new(Term::from_field_text(field, &query_word), distance, false);
                fuzzy_query.set_max_expansions(max_expansions);
                let mut actual: Vec<u32> = searcher
                    .search(&fuzzy_query, &DocSetCollector)?
                    .into_iter()
                    .map(|doc_address| doc_address.doc_id)
                    .collect();
                actual.sort_unstable();

                assert_eq!(
                    actual, expected,
                    "seed {seed}: query {query_word:?}, distance {distance}, max_expansions \
                     {max_expansions}, words {words:?}"
                );
            }
        }
        Ok(())
    }
}
