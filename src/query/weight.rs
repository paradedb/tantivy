use super::scorer::{BasicPruningScorer, PruningScorer};
use super::Scorer;
use crate::docset::COLLECT_BLOCK_BUFFER_LEN;
use crate::index::SegmentReader;
use crate::query::Explanation;
use crate::{DocId, DocSet, Score, TERMINATED};

/// Iterates through all of the documents and scores matched by the DocSet
/// `DocSet`.
pub(crate) fn for_each_scorer<TScorer: Scorer + ?Sized>(
    scorer: &mut TScorer,
    callback: &mut dyn FnMut(DocId, Score),
) {
    let mut doc = scorer.doc();
    while doc != TERMINATED {
        callback(doc, scorer.score());
        doc = scorer.advance();
    }
}

/// Iterates through all of the documents matched by the DocSet
/// `DocSet`.
#[inline]
pub(crate) fn for_each_docset_buffered<T: DocSet + ?Sized>(
    docset: &mut T,
    buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN],
    mut callback: impl FnMut(&[DocId]),
) {
    loop {
        let num_items = docset.fill_buffer(buffer);
        callback(&buffer[..num_items]);
        if num_items != buffer.len() {
            break;
        }
    }
}

/// Iterates through all of the `(doc, score)` produced by a pruning scorer.
///
/// `callback` returns the new threshold after each call, which is fed back into
/// the scorer via [`PruningScorer::set_threshold`] so it can keep pruning.
pub(crate) fn for_each_pruning_scorer<TScorer: PruningScorer + ?Sized>(
    scorer: &mut TScorer,
    callback: &mut dyn FnMut(DocId, Score) -> Score,
) {
    let mut doc = scorer.doc();
    while doc != TERMINATED {
        let new_threshold = callback(doc, scorer.score());
        scorer.set_threshold(new_threshold);
        doc = scorer.advance();
    }
}

/// A Weight is the specialization of a `Query`
/// for a given set of segments.
///
/// See [`Query`](crate::query::Query).
pub trait Weight: Send + Sync + 'static {
    /// Returns the scorer for the given segment.
    ///
    /// `boost` is a multiplier to apply to the score.
    ///
    /// See [`Query`](crate::query::Query).
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>>;

    /// Returns a pruning scorer for the given segment.
    ///
    /// `boost` is a multiplier to apply to the score. `init_threshold` is the
    /// initial score threshold below which documents may be pruned.
    ///
    /// The default implementation wraps [`Weight::scorer`] in a
    /// [`BasicPruningScorer`], which simply filters out `(doc, score)` pairs
    /// below the current threshold. Scorers that can prune more aggressively
    /// (e.g. BlockWAND over a union or intersection) override this.
    fn pruning_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
        init_threshold: Score,
    ) -> crate::Result<Box<dyn PruningScorer>> {
        Ok(Box::new(BasicPruningScorer::new(
            self.scorer(reader, boost)?,
            init_threshold,
        )))
    }

    /// Returns an [`Explanation`] for the given document.
    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation>;

    /// Returns the number documents within the given [`SegmentReader`].
    fn count(&self, reader: &SegmentReader) -> crate::Result<u32> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if let Some(alive_bitset) = reader.alive_bitset() {
            Ok(scorer.count(alive_bitset))
        } else {
            Ok(scorer.count_including_deleted())
        }
    }

    /// Iterates through all of the document matched by the DocSet
    /// `DocSet` and push the scored documents to the collector.
    fn for_each(
        &self,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score),
    ) -> crate::Result<()> {
        let mut scorer = self.scorer(reader, 1.0)?;
        for_each_scorer(scorer.as_mut(), callback);
        Ok(())
    }

    /// Iterates through all of the document matched by the DocSet
    /// `DocSet` and push the scored documents to the collector.
    fn for_each_no_score(
        &self,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(&[DocId]),
    ) -> crate::Result<()> {
        let mut docset = self.scorer(reader, 1.0)?;

        let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];
        for_each_docset_buffered(&mut docset, &mut buffer, callback);
        Ok(())
    }

    /// Calls `callback` with all of the `(doc, score)` for which score
    /// is exceeding a given threshold.
    ///
    /// This method is useful for the [`TopDocs`](crate::collector::TopDocs) collector.
    /// For all docsets, the blanket implementation has the benefit
    /// of prefiltering (doc, score) pairs, avoiding the
    /// virtual dispatch cost.
    ///
    /// More importantly, it makes it possible for scorers to implement
    /// important optimization (e.g. BlockWAND for union).
    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> crate::Result<()> {
        let mut scorer = self.pruning_scorer(reader, 1.0, threshold)?;
        for_each_pruning_scorer(scorer.as_mut(), callback);
        Ok(())
    }

    /// Returns an upper bound on the maximum score any document in the given segment
    /// can achieve for this query/weight and boost factor.
    ///
    /// The default implementation delegates to [`Weight::index_max_score`].
    fn max_score(&self, _reader: &SegmentReader, boost: Score) -> crate::Result<Score> {
        self.index_max_score(boost)
    }

    /// Returns an upper bound on the maximum score across all segments in the index.
    ///
    /// The default implementation returns `Score::MAX` as a sound, conservative fallback.
    fn index_max_score(&self, _boost: Score) -> crate::Result<Score> {
        Ok(Score::MAX)
    }
}

impl Weight for Box<dyn Weight> {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        self.as_ref().scorer(reader, boost)
    }

    fn pruning_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
        init_threshold: Score,
    ) -> crate::Result<Box<dyn PruningScorer>> {
        self.as_ref().pruning_scorer(reader, boost, init_threshold)
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        self.as_ref().explain(reader, doc)
    }

    fn count(&self, reader: &SegmentReader) -> crate::Result<u32> {
        self.as_ref().count(reader)
    }

    fn for_each(
        &self,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score),
    ) -> crate::Result<()> {
        self.as_ref().for_each(reader, callback)
    }

    fn for_each_no_score(
        &self,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(&[DocId]),
    ) -> crate::Result<()> {
        self.as_ref().for_each_no_score(reader, callback)
    }

    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> crate::Result<()> {
        self.as_ref().for_each_pruning(threshold, reader, callback)
    }

    fn max_score(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Score> {
        self.as_ref().max_score(reader, boost)
    }

    fn index_max_score(&self, boost: Score) -> crate::Result<Score> {
        self.as_ref().index_max_score(boost)
    }
}

#[cfg(test)]
mod tests {
    use crate::index::Index;
    use crate::query::{
        AllQuery, BooleanQuery, BoostQuery, ConstScoreQuery, DisjunctionMaxQuery, EmptyQuery,
        EnableScoring, Occur, PhraseQuery, Query, TermQuery,
    };
    use crate::schema::{IndexRecordOption, Schema, TEXT};
    use crate::{DocSet, IndexWriter, Term, TERMINATED};

    #[test]
    fn test_term_weight_max_score() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        {
            let mut writer: IndexWriter = index.writer_for_tests()?;
            // Segment 0: contains "apple", "banana"
            writer.add_document(doc!(text_field => "apple apple banana"))?;
            writer.commit()?;

            // Segment 1: contains only "banana"
            writer.add_document(doc!(text_field => "banana"))?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 2);

        let term_apple = Term::from_field_text(text_field, "apple");
        let term_cherry = Term::from_field_text(text_field, "cherry");

        let query_apple =
            TermQuery::new(term_apple.clone(), IndexRecordOption::WithFreqsAndPositions);
        let weight_apple = query_apple.weight(EnableScoring::enabled_from_searcher(&searcher))?;

        let index_max = weight_apple.index_max_score(1.0)?;
        assert!(index_max > 0.0);

        let seg0_has_apple = searcher
            .segment_reader(0)
            .inverted_index(text_field)?
            .get_term_info(&term_apple)?
            .is_some();
        let (apple_seg, no_apple_seg) = if seg0_has_apple {
            (searcher.segment_reader(0), searcher.segment_reader(1))
        } else {
            (searcher.segment_reader(1), searcher.segment_reader(0))
        };

        let apple_seg_max = weight_apple.max_score(apple_seg, 1.0)?;
        assert_eq!(apple_seg_max, index_max);

        // Verify that all scored docs in the matching segment have score <= apple_seg_max
        let mut scorer0 = weight_apple.scorer(apple_seg, 1.0)?;
        while scorer0.doc() != TERMINATED {
            assert!(scorer0.score() <= apple_seg_max);
            scorer0.advance();
        }

        // Segment without "apple" -> max_score == 0.0
        let no_apple_seg_max = weight_apple.max_score(no_apple_seg, 1.0)?;
        assert_eq!(no_apple_seg_max, 0.0);

        // Query for term not present anywhere in index ("cherry")
        let query_cherry = TermQuery::new(term_cherry, IndexRecordOption::WithFreqsAndPositions);
        let weight_cherry = query_cherry.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        assert_eq!(
            weight_cherry.max_score(searcher.segment_reader(0), 1.0)?,
            0.0
        );
        assert_eq!(
            weight_cherry.max_score(searcher.segment_reader(1), 1.0)?,
            0.0
        );

        // Boost propagation
        let boosted_max = weight_apple.index_max_score(2.5)?;
        assert_eq!(boosted_max, index_max * 2.5);

        Ok(())
    }

    #[test]
    fn test_boolean_and_dismax_max_score() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        {
            let mut writer: IndexWriter = index.writer_for_tests()?;
            writer.add_document(doc!(text_field => "apple banana"))?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let seg_reader = searcher.segment_reader(0);

        let term_apple = Term::from_field_text(text_field, "apple");
        let term_banana = Term::from_field_text(text_field, "banana");

        let q_apple = Box::new(TermQuery::new(
            term_apple,
            IndexRecordOption::WithFreqsAndPositions,
        )) as Box<dyn Query>;
        let q_banana = Box::new(TermQuery::new(
            term_banana,
            IndexRecordOption::WithFreqsAndPositions,
        )) as Box<dyn Query>;

        let w_apple = q_apple.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let w_banana = q_banana.weight(EnableScoring::enabled_from_searcher(&searcher))?;

        let max_a = w_apple.index_max_score(1.0)?;
        let max_b = w_banana.index_max_score(1.0)?;

        // BooleanQuery with Should clauses (sum)
        let bool_should = BooleanQuery::new(vec![
            (Occur::Should, q_apple.box_clone()),
            (Occur::Should, q_banana.box_clone()),
        ]);
        let w_bool_should = bool_should.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        assert_eq!(w_bool_should.index_max_score(1.0)?, max_a + max_b);
        assert_eq!(w_bool_should.max_score(seg_reader, 1.0)?, max_a + max_b);

        // BooleanQuery with Filter and MustNot (they contribute 0 to score)
        let bool_filter = BooleanQuery::new(vec![
            (Occur::Must, q_apple.box_clone()),
            (Occur::MustNot, q_banana.box_clone()),
        ]);
        let w_bool_filter = bool_filter.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        assert_eq!(w_bool_filter.index_max_score(1.0)?, max_a);
        assert_eq!(w_bool_filter.max_score(seg_reader, 1.0)?, max_a);

        // DisjunctionMaxQuery with tie_breaker = 0.5
        let dismax = DisjunctionMaxQuery::with_tie_breaker(
            vec![q_apple.box_clone(), q_banana.box_clone()],
            0.5,
        );
        let w_dismax = dismax.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let expected_dismax = max_a.max(max_b) + (max_a + max_b - max_a.max(max_b)) * 0.5;
        assert_eq!(w_dismax.index_max_score(1.0)?, expected_dismax);
        assert_eq!(w_dismax.max_score(seg_reader, 1.0)?, expected_dismax);

        // BoostQuery
        let boost_q = BoostQuery::new(q_apple.box_clone(), 3.0);
        let w_boost = boost_q.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        assert_eq!(w_boost.index_max_score(1.0)?, max_a * 3.0);
        assert_eq!(w_boost.max_score(seg_reader, 1.0)?, max_a * 3.0);

        Ok(())
    }

    #[test]
    fn test_const_all_empty_phrase_max_score() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        {
            let mut writer: IndexWriter = index.writer_for_tests()?;
            writer.add_document(doc!(text_field => "quick brown fox"))?;
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let seg_reader = searcher.segment_reader(0);

        // ConstScoreQuery
        let const_q = ConstScoreQuery::new(Box::new(AllQuery), 4.2);
        let w_const = const_q.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        assert_eq!(w_const.index_max_score(1.0)?, 4.2);
        assert_eq!(w_const.max_score(seg_reader, 2.0)?, 8.4);

        // AllQuery
        let all_q = AllQuery;
        let w_all = all_q.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        assert_eq!(w_all.index_max_score(1.0)?, 1.0);
        assert_eq!(w_all.max_score(seg_reader, 1.5)?, 1.5);

        // EmptyQuery
        let empty_q = EmptyQuery;
        let w_empty = empty_q.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        assert_eq!(w_empty.index_max_score(1.0)?, 0.0);
        assert_eq!(w_empty.max_score(seg_reader, 1.0)?, 0.0);

        // PhraseQuery with existing terms
        let phrase_q = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "quick"),
            Term::from_field_text(text_field, "brown"),
        ]);
        let w_phrase = phrase_q.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        assert!(w_phrase.index_max_score(1.0)? > 0.0);
        assert_eq!(
            w_phrase.max_score(seg_reader, 1.0)?,
            w_phrase.index_max_score(1.0)?
        );

        // PhraseQuery with missing term
        let phrase_missing = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "quick"),
            Term::from_field_text(text_field, "missing"),
        ]);
        let w_phrase_missing =
            phrase_missing.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        assert_eq!(w_phrase_missing.max_score(seg_reader, 1.0)?, 0.0);

        // Query helper methods on Query trait
        assert_eq!(const_q.index_max_score(&searcher)?, 4.2);
        assert_eq!(const_q.max_score(&searcher, seg_reader)?, 4.2);

        Ok(())
    }
}
