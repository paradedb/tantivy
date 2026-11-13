use std::sync::Mutex;

use super::scorer::{BasicPruningScorer, PruningScorer};
use super::Scorer;
use crate::docset::{SeekDangerResult, COLLECT_BLOCK_BUFFER_LEN};
use crate::index::SegmentReader;
use crate::query::Explanation;
use crate::{DocId, DocSet, Score, TERMINATED};

/// A prepared membership check over sorted candidate document IDs.
#[derive(Clone)]
pub struct CandidateFilter(std::sync::Arc<CandidateCheck>);

type CandidateCheck = dyn Fn(&[DocId]) -> crate::Result<Vec<bool>> + Send + Sync;

impl CandidateFilter {
    /// Builds a reusable predicate over batches sorted by document ID.
    pub fn new(
        check: impl Fn(&[DocId]) -> crate::Result<Vec<bool>> + Send + Sync + 'static,
    ) -> Self {
        Self(std::sync::Arc::new(check))
    }

    pub(crate) fn from_weight<W: Weight + Clone>(weight: W, reader: &SegmentReader) -> Self {
        let reader = reader.clone();
        Self::new(move |docs| weight.matches_candidates(&reader, docs))
    }

    pub(crate) fn from_cloned_scorer<S: Scorer + Clone>(scorer: &dyn Scorer) -> Option<Self> {
        let scorer = Mutex::new(scorer.downcast_ref::<S>()?.clone());
        Some(Self::new(move |docs| {
            let prepared = scorer.lock().unwrap();
            let mut scorer = prepared.clone();
            Ok(docs
                .iter()
                .map(|&doc| scorer.seek_danger(doc) == SeekDangerResult::Found)
                .collect())
        }))
    }

    /// Tests membership without advancing through intervening matching documents.
    pub fn matches(&self, docs: &[DocId]) -> crate::Result<Vec<bool>> {
        (self.0)(docs)
    }
}

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

pub(crate) fn for_each_docset_batch<T: DocSet + ?Sized>(
    docset: &mut T,
    callback: &mut dyn FnMut(crate::DocSetBatch<'_>),
) {
    if docset.has_fast_bitset() {
        while docset.doc() < TERMINATED {
            let base = docset.doc() / crate::BLOCK_WINDOW * crate::BLOCK_WINDOW;
            let mut mask = [common::TinySet::EMPTY; crate::BLOCK_NUM_TINYBITSETS];
            docset.fill_bitset_block(base, &mut mask);
            callback(crate::DocSetBatch::Bitmap(base, &mask));
        }
    } else {
        let mut buffer = [0; COLLECT_BLOCK_BUFFER_LEN];
        for_each_docset_buffered(docset, &mut buffer, |docs| {
            callback(crate::DocSetBatch::Docs(docs))
        });
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
    /// Whether evaluating this filter reads fast-field values rather than only postings.
    fn uses_fast_fields(&self) -> bool {
        false
    }

    /// Prepares reusable candidate checks. Posting-only leaves may materialize their matches.
    fn candidate_filter(&self, reader: &SegmentReader) -> crate::Result<CandidateFilter> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.is::<crate::query::AllScorer>() {
            let max_doc = reader.max_doc();
            return Ok(CandidateFilter::new(move |docs| {
                Ok(docs.iter().map(|&doc| doc < max_doc).collect())
            }));
        }
        let mut matched = common::BitSet::with_max_value(reader.max_doc());
        let mut buffer = [0; COLLECT_BLOCK_BUFFER_LEN];
        for_each_docset_buffered(scorer.as_mut(), &mut buffer, |docs| {
            for &doc in docs {
                matched.insert(doc);
            }
        });
        Ok(CandidateFilter::new(move |docs| {
            Ok(docs.iter().map(|&doc| matched.contains(doc)).collect())
        }))
    }

    /// Tests sorted candidate document IDs without enumerating all matching documents.
    fn matches_candidates(
        &self,
        reader: &SegmentReader,
        docs: &[DocId],
    ) -> crate::Result<Vec<bool>> {
        let mut scorer = self.scorer_for_candidates_unpositioned(reader, 1.0, docs.len() as u32)?;
        Ok(docs
            .iter()
            .map(|&doc| scorer.seek_danger(doc) == crate::docset::SeekDangerResult::Found)
            .collect())
    }

    /// Returns the scorer for the given segment.
    ///
    /// `boost` is a multiplier to apply to the score.
    ///
    /// See [`Query`](crate::query::Query).
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>>;

    /// Returns an unpositioned scorer for the given segment when the caller will probe it against
    /// an upstream candidate set of approximately `candidate_size` documents.
    ///
    /// Weights whose strategy selection depends on the upstream candidate count (such as
    /// `TermSetWeight`) override this to pick candidate-probing strategies (e.g. fast-field point
    /// lookups via `seek_danger`) instead of full-segment bitset construction or an eager scan to
    /// the first matching document.
    ///
    /// Immediately after construction, `scorer.doc()` returns [`TERMINATED`](crate::TERMINATED) if
    /// the segment is known to have no matches, and `0` (or the first match) otherwise. Callers
    /// must position the scorer via `seek_danger` (until `Found` is returned) or `seek` before
    /// relying on `doc()`, `score()`, or `advance()`.
    fn scorer_for_candidates_unpositioned(
        &self,
        reader: &SegmentReader,
        boost: Score,
        candidate_size: u32,
    ) -> crate::Result<Box<dyn Scorer>> {
        let _ = candidate_size;
        self.scorer(reader, boost)
    }

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

    /// Visits unscored matches, preserving membership bitmaps for capable collectors.
    fn for_each_no_score_batch(
        &self,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(crate::DocSetBatch<'_>),
    ) -> crate::Result<()> {
        let mut scorer = self.scorer(reader, 1.0)?;
        for_each_docset_batch(scorer.as_mut(), callback);
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
}

#[cfg(test)]
mod candidate_tests {
    use std::ops::Bound;

    use super::*;
    use crate::query::{
        BooleanQuery, EnableScoring, ExistsQuery, FastFieldRangeQuery, Occur, Query, TermQuery,
    };
    use crate::schema::{IndexRecordOption, Schema, FAST, INDEXED, STRING};
    use crate::{Index, TantivyDocument, Term};

    #[test]
    fn candidate_filter_matches_scorer_for_mixed_queries() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let number = schema.add_u64_field("number", FAST | INDEXED);
        let text = schema.add_text_field("text", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_with_num_threads::<TantivyDocument>(1, 15_000_000)?;
        for i in 0..300u64 {
            let mut doc = TantivyDocument::new();
            if i % 7 != 0 {
                doc.add_u64(number, i);
            }
            if i % 11 == 0 {
                doc.add_u64(number, 1000 - i);
            }
            doc.add_text(text, if i % 3 == 0 { "yes" } else { "no" });
            writer.add_document(doc)?;
        }
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        let reader = searcher.segment_reader(0);
        let range = || {
            Box::new(FastFieldRangeQuery::new(
                Bound::Included(Term::from_field_u64(number, 50)),
                Bound::Excluded(Term::from_field_u64(number, 150)),
            )) as Box<dyn Query>
        };
        let term = || {
            Box::new(TermQuery::new(
                Term::from_field_text(text, "yes"),
                IndexRecordOption::Basic,
            )) as Box<dyn Query>
        };
        let mut queries: Vec<Box<dyn Query>> = vec![
            range(),
            Box::new(ExistsQuery::new("number".into(), false)),
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, range()),
                (Occur::Must, term()),
            ])),
            Box::new(BooleanQuery::new(vec![
                (Occur::Should, range()),
                (Occur::Should, term()),
            ])),
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, range()),
                (Occur::MustNot, term()),
            ])),
            Box::new(BooleanQuery::new(vec![(Occur::MustNot, range())])),
        ];
        let mut minimum =
            BooleanQuery::new(vec![(Occur::Should, range()), (Occur::Should, term())]);
        minimum.set_minimum_number_should_match(2);
        queries.push(Box::new(minimum));
        for query in queries {
            let weight = query.weight(EnableScoring::disabled_from_schema(searcher.schema()))?;
            assert!(weight.uses_fast_fields());
            let prepared = weight.candidate_filter(reader)?;
            let mut scorer = weight.scorer(reader, 1.0)?;
            let mut expected = vec![false; reader.max_doc() as usize];
            while scorer.doc() != TERMINATED {
                expected[scorer.doc() as usize] = true;
                scorer.advance();
            }
            let mut scorer = weight.scorer_for_candidates_unpositioned(reader, 1.0, 10)?;
            let lower_bound = scorer.doc();
            for doc in 0..reader.max_doc() {
                let matched =
                    doc >= lower_bound && scorer.seek_danger(doc) == SeekDangerResult::Found;
                assert_eq!(matched, expected[doc as usize], "{query:?}, doc={doc}");
            }
            // Each batch is sorted; successive batches intentionally move backwards.
            for docs in [
                vec![250, 299],
                vec![0, 0, 3, 49, 50, 77, 149, 150, 200],
                (0..reader.max_doc()).collect(),
            ] {
                let wanted: Vec<_> = docs.iter().map(|&doc| expected[doc as usize]).collect();
                assert_eq!(prepared.matches(&docs)?, wanted, "{query:?}");
            }
        }
        Ok(())
    }
}
