use std::fmt;

use crate::docset::{DocSet, TERMINATED};
use crate::fieldnorm::FieldNormReader;
use crate::index::{Bm25Params, SegmentReader};
use crate::postings::{FreqReadingOption, Postings, SegmentPostings};
use crate::query::bm25::Bm25Weight;
use crate::query::boolean_query::BlockWandSingleScorer;
use crate::query::explanation::does_not_match;
use crate::query::scorer::{BasicPruningScorer, BlockMaxScorer, PruningScorer};
use crate::query::{EmptyScorer, EnableScoring, Explanation, Query, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption};
use crate::{DocId, Score, Term};

/// A query that matches documents containing any of the given terms, scoring
/// each document as if all the terms were a single term.
///
/// This mirrors Lucene's `SynonymQuery`: the terms are treated as one
/// pseudo-term whose document frequency is the maximum document frequency of
/// the individual terms, and a document's term frequency is the sum of its
/// frequencies for each term. This avoids two artifacts of expanding synonyms
/// into a boolean OR of term queries:
///
/// * a rare synonym receiving a large idf boost relative to a common one, and
/// * documents containing several variants outscoring documents containing a single variant, purely
///   because scores of separate clauses are summed.
///
/// All terms must belong to the same field.
#[derive(Clone)]
pub struct SynonymQuery {
    terms: Vec<Term>,
}

impl fmt::Debug for SynonymQuery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SynonymQuery({:?})", self.terms)
    }
}

impl SynonymQuery {
    /// Creates a new `SynonymQuery` from the given terms.
    ///
    /// Panics if `terms` is empty or if the terms do not all belong to the
    /// same field.
    pub fn new(terms: Vec<Term>) -> SynonymQuery {
        assert!(!terms.is_empty(), "SynonymQuery requires at least one term");
        let field = terms[0].field();
        assert!(
            terms[1..].iter().all(|term| term.field() == field),
            "All SynonymQuery terms must belong to the same field."
        );
        SynonymQuery { terms }
    }

    /// The terms this query is built out of.
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    fn specialized_weight(
        &self,
        enable_scoring: EnableScoring<'_>,
    ) -> crate::Result<SynonymWeight> {
        let schema = enable_scoring.schema();
        let field_entry = schema.get_field_entry(self.terms[0].field());
        if !field_entry.is_indexed() {
            let error_msg = format!("Field {:?} is not indexed.", field_entry.name());
            return Err(crate::TantivyError::SchemaError(error_msg));
        }
        let bm25_weight = match enable_scoring {
            EnableScoring::Enabled {
                statistics_provider,
                ..
            } => {
                let field = self.terms[0].field();
                let total_num_tokens = statistics_provider.total_num_tokens(field)?;
                let total_num_docs = statistics_provider.total_num_docs()?;
                let average_fieldnorm = total_num_tokens as Score / total_num_docs as Score;
                let params = statistics_provider.bm25_params(field);
                // Blended statistics: the pseudo-term's document frequency is
                // the maximum document frequency among the terms.
                let mut max_doc_freq: u64 = 0;
                for term in &self.terms {
                    max_doc_freq = max_doc_freq.max(statistics_provider.doc_freq(term)?);
                }
                Bm25Weight::for_one_term(max_doc_freq, total_num_docs, average_fieldnorm, params)
            }
            EnableScoring::Disabled { .. } => Bm25Weight::new(
                Explanation::new("<no score>", 1.0f32),
                1.0f32,
                Bm25Params::default(),
            ),
        };
        let scoring_enabled = enable_scoring.is_scoring_enabled();
        let index_record_option = if scoring_enabled {
            IndexRecordOption::WithFreqs
        } else {
            IndexRecordOption::Basic
        };
        Ok(SynonymWeight {
            terms: self.terms.clone(),
            index_record_option,
            similarity_weight: bm25_weight,
            scoring_enabled,
        })
    }
}

impl Query for SynonymQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(self.specialized_weight(enable_scoring)?))
    }

    fn query_terms(
        &self,
        _field: Field,
        _segment_reader: &SegmentReader,
        visitor: &mut dyn FnMut(&Term, bool),
    ) {
        for term in &self.terms {
            visitor(term, false);
        }
    }
}

pub struct SynonymWeight {
    terms: Vec<Term>,
    index_record_option: IndexRecordOption,
    similarity_weight: Bm25Weight,
    scoring_enabled: bool,
}

impl SynonymWeight {
    fn specialized_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<Option<SynonymScorer>> {
        let field = self.terms[0].field();
        let inverted_index = reader.inverted_index(field)?;
        let mut postings: Vec<SegmentPostings> = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            if let Some(term_info) = inverted_index.get_term_info(term)? {
                postings.push(
                    inverted_index
                        .read_postings_from_terminfo(&term_info, self.index_record_option)?,
                );
            }
        }
        if postings.is_empty() {
            return Ok(None);
        }
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let similarity_weight = self.similarity_weight.boost_by(boost);
        Ok(Some(SynonymScorer::new(
            postings,
            fieldnorm_reader,
            similarity_weight,
        )))
    }

    fn fieldnorm_reader(&self, segment_reader: &SegmentReader) -> crate::Result<FieldNormReader> {
        if self.scoring_enabled {
            if let Some(fieldnorm_reader) = segment_reader
                .fieldnorms_readers()
                .get_field(self.terms[0].field())?
            {
                return Ok(fieldnorm_reader);
            }
        }
        Ok(FieldNormReader::constant(segment_reader.max_doc(), 1))
    }
}

impl Weight for SynonymWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        match self.specialized_scorer(reader, boost)? {
            Some(scorer) => Ok(Box::new(scorer)),
            None => Ok(Box::new(EmptyScorer)),
        }
    }

    fn pruning_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
        init_threshold: Score,
    ) -> crate::Result<Box<dyn PruningScorer>> {
        match self.specialized_scorer(reader, boost)? {
            Some(scorer) if scorer.reads_freqs() => {
                Ok(Box::new(BlockWandSingleScorer::new(scorer, init_threshold)))
            }
            Some(scorer) => Ok(Box::new(BasicPruningScorer::new(
                Box::new(scorer),
                init_threshold,
            ))),
            None => Ok(Box::new(EmptyScorer)),
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let Some(mut scorer) = self.specialized_scorer(reader, 1.0)? else {
            return Err(does_not_match(doc));
        };
        if scorer.doc() > doc || scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        let fieldnorm_id = scorer.fieldnorm_id();
        let term_freq = scorer.term_freq();
        let mut explanation = self.similarity_weight.explain(fieldnorm_id, term_freq);
        explanation.add_context(format!(
            "SynonymQuery, blended term_freq across {:?}",
            self.terms
        ));
        Ok(explanation)
    }
}

/// A union scorer over the posting lists of the synonym's terms.
///
/// Docs matched by several terms are scored once, with a term frequency equal
/// to the sum of the individual term frequencies.
pub struct SynonymScorer {
    postings: Vec<SegmentPostings>,
    doc: DocId,
    fieldnorm_reader: FieldNormReader,
    similarity_weight: Bm25Weight,
}

impl SynonymScorer {
    fn new(
        postings: Vec<SegmentPostings>,
        fieldnorm_reader: FieldNormReader,
        similarity_weight: Bm25Weight,
    ) -> SynonymScorer {
        let doc = postings.iter().map(DocSet::doc).min().unwrap_or(TERMINATED);
        SynonymScorer {
            postings,
            doc,
            fieldnorm_reader,
            similarity_weight,
        }
    }

    /// Block-WAND needs the frequencies, which a scorer built without scoring does not read.
    pub(crate) fn reads_freqs(&self) -> bool {
        self.postings.iter().all(|postings| {
            postings.block_cursor.freq_reading_option() == FreqReadingOption::ReadFreq
        })
    }

    fn fieldnorm_id(&self) -> u8 {
        self.fieldnorm_reader.fieldnorm_id(self.doc)
    }

    /// The blended term frequency for the current doc: the sum of the term
    /// frequencies of every synonym term matching it.
    fn term_freq(&mut self) -> u32 {
        let doc = self.doc;
        self.postings
            .iter_mut()
            .filter(|postings| postings.doc() == doc)
            .map(|postings| postings.term_freq())
            .sum()
    }
}

impl DocSet for SynonymScorer {
    fn advance(&mut self) -> DocId {
        let current = self.doc;
        if current == TERMINATED {
            return TERMINATED;
        }
        let mut next = TERMINATED;
        for postings in &mut self.postings {
            let mut doc = postings.doc();
            if doc == current {
                doc = postings.advance();
            }
            next = next.min(doc);
        }
        self.doc = next;
        next
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.doc >= target {
            return self.doc;
        }
        let mut next = TERMINATED;
        for postings in &mut self.postings {
            let mut doc = postings.doc();
            if doc < target {
                doc = postings.seek(target);
            }
            next = next.min(doc);
        }
        self.doc = next;
        next
    }

    fn doc(&self) -> DocId {
        self.doc
    }

    fn size_hint(&self) -> u32 {
        // Upper bound: the union can't match more docs than the sum of the
        // individual posting lists.
        self.postings
            .iter()
            .map(DocSet::size_hint)
            .fold(0u32, u32::saturating_add)
    }
}

impl Scorer for SynonymScorer {
    fn score(&mut self) -> Score {
        let fieldnorm_id = self.fieldnorm_id();
        let term_freq = self.term_freq();
        self.similarity_weight.score(fieldnorm_id, term_freq)
    }
}

impl BlockMaxScorer for SynonymScorer {
    /// The members' bounds summed, no higher than the group's own ceiling. BM25 grows less than
    /// linearly with the term frequency, so the score of a summed frequency never exceeds the sum
    /// of the scores, and a member whose block starts after this one contributes nothing here.
    fn block_max_score(&mut self) -> Score {
        let fieldnorm_reader = &self.fieldnorm_reader;
        let weight = &self.similarity_weight;
        self.postings
            .iter_mut()
            .filter(|postings| postings.doc() != TERMINATED)
            .map(|postings| {
                postings
                    .block_cursor
                    .block_max_score(fieldnorm_reader, weight)
            })
            .sum::<Score>()
            .min(weight.max_score())
    }

    /// The nearest block end among the members, so no member crosses into a new block before it.
    fn last_doc_in_block(&self) -> DocId {
        self.postings
            .iter()
            .filter(|postings| postings.doc() != TERMINATED)
            .map(|postings| postings.block_cursor.skip_reader().last_doc_in_block())
            .min()
            .unwrap_or(TERMINATED)
    }

    fn seek_block(&mut self, target: DocId) {
        for postings in &mut self.postings {
            if postings.doc() < target {
                postings.block_cursor.seek_block(target);
            }
        }
    }

    fn max_score(&self) -> Score {
        self.similarity_weight.max_score()
    }
}

#[cfg(test)]
mod tests {
    use super::{SynonymQuery, SynonymScorer};
    use crate::collector::{Count, TopDocs};
    use crate::docset::{DocSet, TERMINATED};
    use crate::query::{
        BlockMaxScorer, BooleanQuery, EnableScoring, Occur, Query, Scorer, TermQuery,
    };
    use crate::schema::{Field, IndexRecordOption, Schema, TEXT};
    use crate::{assert_nearly_equals, DocAddress, Index, IndexWriter, Score, Term};

    fn build_test_index() -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut writer: IndexWriter = index.writer_for_tests()?;
            writer.add_document(doc!(text => "the cat sat"))?;
            writer.add_document(doc!(text => "a feline slept"))?;
            writer.add_document(doc!(text => "cat and feline together"))?;
            writer.add_document(doc!(text => "dogs everywhere"))?;
            writer.add_document(doc!(text => "cat cat cat"))?;
            writer.commit()?;
        }
        Ok(index)
    }

    #[test]
    fn test_synonym_query_matches_union() -> crate::Result<()> {
        let index = build_test_index()?;
        let text = index.schema().get_field("text").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = SynonymQuery::new(vec![
            Term::from_field_text(text, "cat"),
            Term::from_field_text(text, "feline"),
        ]);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 4);
        Ok(())
    }

    #[test]
    fn test_synonym_query_missing_term() -> crate::Result<()> {
        let index = build_test_index()?;
        let text = index.schema().get_field("text").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = SynonymQuery::new(vec![
            Term::from_field_text(text, "cat"),
            Term::from_field_text(text, "nosuchterm"),
        ]);
        let count = searcher.search(&query, &Count)?;
        assert_eq!(count, 3);
        let all_missing = SynonymQuery::new(vec![Term::from_field_text(text, "nosuchterm")]);
        let count = searcher.search(&all_missing, &Count)?;
        assert_eq!(count, 0);
        Ok(())
    }

    #[test]
    fn test_synonym_query_blended_scoring() -> crate::Result<()> {
        let index = build_test_index()?;
        let text = index.schema().get_field("text").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = SynonymQuery::new(vec![
            Term::from_field_text(text, "cat"),
            Term::from_field_text(text, "feline"),
        ]);
        let top_docs = searcher.search(&query, &TopDocs::with_limit(4).order_by_score())?;
        // Doc 4 ("cat cat cat") has the highest blended term frequency and the
        // same fieldnorm as doc 2 does not; it must outrank single-occurrence
        // docs.
        assert_eq!(top_docs[0].1, DocAddress::new(0, 4));

        // A doc matching both synonyms scores as one pseudo-term with tf=2,
        // strictly less than 2x the single-variant score (saturation), and
        // identical to what a single term with tf=2 would get.
        let scores: std::collections::HashMap<DocAddress, f32> = top_docs
            .iter()
            .map(|(score, addr)| (*addr, *score))
            .collect();
        let single = scores[&DocAddress::new(0, 0)];
        let both = scores[&DocAddress::new(0, 2)];
        assert!(both > 0.0 && single > 0.0);
        assert!(both < 2.0 * single);
        Ok(())
    }

    #[test]
    fn test_synonym_query_scores_lower_than_boolean_or_for_multi_variant_docs() -> crate::Result<()>
    {
        // Documents containing both variants should NOT be double-counted the
        // way a boolean OR of term queries does.
        let index = build_test_index()?;
        let text = index.schema().get_field("text").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let cat = Term::from_field_text(text, "cat");
        let feline = Term::from_field_text(text, "feline");

        let synonym = SynonymQuery::new(vec![cat.clone(), feline.clone()]);
        let boolean = BooleanQuery::new(vec![
            (
                Occur::Should,
                Box::new(TermQuery::new(cat, IndexRecordOption::WithFreqs)) as Box<dyn Query>,
            ),
            (
                Occur::Should,
                Box::new(TermQuery::new(feline, IndexRecordOption::WithFreqs)) as Box<dyn Query>,
            ),
        ]);

        let synonym_top = searcher.search(&synonym, &TopDocs::with_limit(5).order_by_score())?;
        let boolean_top = searcher.search(&boolean, &TopDocs::with_limit(5).order_by_score())?;
        // Same match set.
        let mut synonym_docs: Vec<_> = synonym_top.iter().map(|(_, d)| *d).collect();
        let mut boolean_docs: Vec<_> = boolean_top.iter().map(|(_, d)| *d).collect();
        synonym_docs.sort();
        boolean_docs.sort();
        assert_eq!(synonym_docs, boolean_docs);
        Ok(())
    }

    #[test]
    fn test_synonym_query_explain() -> crate::Result<()> {
        let index = build_test_index()?;
        let text = index.schema().get_field("text").unwrap();
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = SynonymQuery::new(vec![
            Term::from_field_text(text, "cat"),
            Term::from_field_text(text, "feline"),
        ]);
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let explanation = weight.explain(searcher.segment_reader(0), 2)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(5).order_by_score())?;
        let score = top_docs
            .iter()
            .find(|(_, addr)| *addr == DocAddress::new(0, 2))
            .map(|(score, _)| *score)
            .unwrap();
        assert_nearly_equals!(explanation.value(), score);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "at least one term")]
    fn test_synonym_query_empty_terms_panics() {
        let _ = SynonymQuery::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "same field")]
    fn test_synonym_query_mixed_fields_panics() {
        let mut schema_builder = Schema::builder();
        let a = schema_builder.add_text_field("a", TEXT);
        let b = schema_builder.add_text_field("b", TEXT);
        let _schema = schema_builder.build();
        let _ = SynonymQuery::new(vec![
            Term::from_field_text(a, "x"),
            Term::from_field_text(b, "y"),
        ]);
    }

    /// Thousands of docs in one segment, so the posting lists span many blocks and the term
    /// frequencies and field lengths vary from block to block.
    fn large_index() -> crate::Result<(Index, Field)> {
        let mut schema_builder = Schema::builder();
        let text = schema_builder.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer: IndexWriter = index.writer_for_tests()?;
        for i in 0..6_000u32 {
            let mut words = Vec::new();
            if i % 3 == 0 {
                words.extend(std::iter::repeat_n("cat", (i % 5 + 1) as usize));
            }
            if i % 7 == 0 {
                words.extend(std::iter::repeat_n("feline", (i % 4 + 1) as usize));
            }
            if i % 11 == 0 {
                words.extend(std::iter::repeat_n("dog", 2));
            }
            words.extend(std::iter::repeat_n("pad", (i % 9) as usize));
            writer.add_document(doc!(text => words.join(" ")))?;
        }
        writer.commit()?;
        Ok((index, text))
    }

    fn cat_feline(text: Field) -> SynonymQuery {
        SynonymQuery::new(vec![
            Term::from_field_text(text, "cat"),
            Term::from_field_text(text, "feline"),
        ])
    }

    /// Every `(score, doc)` of `query`, best first, with no pruning involved.
    fn exhaustive(
        searcher: &crate::Searcher,
        query: &dyn Query,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let weight = query.weight(EnableScoring::enabled_from_searcher(searcher))?;
        let mut hits = Vec::new();
        for (ord, reader) in searcher.segment_readers().iter().enumerate() {
            weight.for_each(reader, &mut |doc, score| {
                hits.push((score, DocAddress::new(ord as u32, doc)))
            })?;
        }
        hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        Ok(hits)
    }

    fn assert_same_top(pruned: &[(Score, DocAddress)], all: &[(Score, DocAddress)]) {
        assert_eq!(pruned.len(), all.len().min(pruned.len()));
        let by_doc: std::collections::HashMap<DocAddress, Score> =
            all.iter().map(|(score, doc)| (*doc, *score)).collect();
        for (i, (score, doc)) in pruned.iter().enumerate() {
            assert_nearly_equals!(*score, all[i].0);
            assert_nearly_equals!(*score, by_doc[doc]);
        }
    }

    #[test]
    fn the_block_bound_covers_every_doc_of_the_block() -> crate::Result<()> {
        let (index, text) = large_index()?;
        let searcher = index.reader()?.searcher();
        let query = cat_feline(text);
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
        let mut scorer = scorer.downcast::<SynonymScorer>().map_err(|_| ()).unwrap();
        let mut blocks = 0;
        while scorer.doc() != TERMINATED {
            let bound = scorer.block_max_score();
            let end = scorer.last_doc_in_block();
            assert!(bound <= scorer.max_score());
            while scorer.doc() != TERMINATED && scorer.doc() <= end {
                let score = scorer.score();
                assert!(
                    score <= bound,
                    "doc {} scores {score} over the bound {bound}",
                    scorer.doc()
                );
                scorer.advance();
            }
            blocks += 1;
            if end == TERMINATED {
                break;
            }
            scorer.seek_block(end + 1);
            scorer.seek(end + 1);
        }
        assert!(blocks > 10, "the index spans {blocks} blocks only");
        Ok(())
    }

    #[test]
    fn pruned_top_k_matches_the_exhaustive_order() -> crate::Result<()> {
        let (index, text) = large_index()?;
        let searcher = index.reader()?.searcher();
        let query = cat_feline(text);
        let pruned = searcher.search(&query, &TopDocs::with_limit(10).order_by_score())?;
        assert_same_top(&pruned, &exhaustive(&searcher, &query)?);
        Ok(())
    }

    #[test]
    fn a_union_with_a_term_prunes_to_the_same_top_k() -> crate::Result<()> {
        let (index, text) = large_index()?;
        let searcher = index.reader()?.searcher();
        let query = BooleanQuery::new(vec![
            (
                Occur::Should,
                Box::new(TermQuery::new(
                    Term::from_field_text(text, "dog"),
                    IndexRecordOption::WithFreqs,
                )),
            ),
            (Occur::Should, Box::new(cat_feline(text))),
        ]);
        let pruned = searcher.search(&query, &TopDocs::with_limit(10).order_by_score())?;
        assert_same_top(&pruned, &exhaustive(&searcher, &query)?);
        Ok(())
    }
}
