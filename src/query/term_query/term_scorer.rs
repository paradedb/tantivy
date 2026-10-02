use crate::docset::DocSet;
use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentId;
use crate::postings::compression::COMPRESSION_BLOCK_SIZE;
use crate::postings::{BlockSegmentPostings, FreqReadingOption, Postings, SegmentPostings};
use crate::query::bm25::Bm25Weight;
use crate::query::{Explanation, Scorer};
use crate::schema::Field;
use crate::{DocId, Score};

#[derive(Clone)]
pub struct TermScorer {
    postings: SegmentPostings,
    fieldnorm_reader: FieldNormReader,
    similarity_weight: Bm25Weight,
    fieldnorm_source: Option<(SegmentId, Field)>,
    block_max_weight_override: Option<Bm25Weight>,
}

impl TermScorer {
    pub fn new(
        postings: SegmentPostings,
        fieldnorm_reader: FieldNormReader,
        similarity_weight: Bm25Weight,
    ) -> TermScorer {
        TermScorer {
            postings,
            fieldnorm_reader,
            similarity_weight,
            fieldnorm_source: None,
            block_max_weight_override: None,
        }
    }

    pub(crate) fn with_block_max_weight_override(mut self, weight: Bm25Weight) -> Self {
        self.block_max_weight_override = Some(weight);
        self
    }

    pub(crate) fn with_fieldnorm_source(mut self, segment: SegmentId, field: Field) -> Self {
        self.fieldnorm_source = Some((segment, field));
        self
    }

    pub(crate) fn shares_fieldnorms_with(&self, other: &Self) -> bool {
        self.fieldnorm_source.is_some() && self.fieldnorm_source == other.fieldnorm_source
    }

    #[inline]
    pub(crate) fn has_term_norms(&self) -> bool {
        self.postings.block_cursor.has_term_norms()
    }

    /// If term norms are enabled, batch-evaluates BM25 scores for documents in the current block
    /// strictly smaller than `window_end` and invokes `f(docs, scores)`.
    ///
    /// This auto-vectorizes the score calculation using SIMD across contiguous frequency and norm
    /// buffers. Advances the underlying posting cursor across the consumed slice.
    ///
    /// Returns `true` if handled in batch, or `false` if term norms are not present.
    pub(crate) fn for_each_block_slice_with_term_norms<F>(
        &mut self,
        window_end: DocId,
        mut f: F,
    ) -> bool
    where
        F: FnMut(&[DocId], &[Score]),
    {
        if !self.has_term_norms() {
            return false;
        }

        let norm_const = self.similarity_weight.norm_const();
        let norm_factor = self.similarity_weight.norm_factor();
        let weight = self.similarity_weight.weight();
        let read_freq = matches!(
            self.postings.block_cursor.freq_reading_option(),
            FreqReadingOption::ReadFreq
        );

        while self.doc() < window_end {
            let start = self.postings.block_offset();
            let block_docs = self.postings.block_cursor.docs();
            let block_len = self.postings.block_cursor.block_len();
            if start >= block_len {
                break;
            }
            let slice_docs = &block_docs[start..block_len];
            let count = slice_docs.partition_point(|&d| d < window_end);
            if count == 0 {
                break;
            }
            let end = start + count;

            let mut batch_scores = [0.0f32; COMPRESSION_BLOCK_SIZE];
            let sc = &mut batch_scores[..count];

            {
                let norms_decoder = self.postings.block_cursor.fieldnorm_decoder();
                let norms = &norms_decoder.output_array()[start..end];

                if read_freq {
                    let freqs = &self.postings.block_cursor.freq_output_array()[start..end];
                    for i in 0..count {
                        let tf = freqs[i] as f32;
                        let norm = norm_const + norm_factor * (norms[i] as f32);
                        sc[i] = weight * (tf / (tf + norm));
                    }
                } else {
                    for i in 0..count {
                        let norm = norm_const + norm_factor * (norms[i] as f32);
                        sc[i] = weight * (1.0 / (1.0 + norm));
                    }
                }
            }

            let docs = &self.postings.block_cursor.docs()[start..end];
            f(docs, sc);

            self.postings.advance_within_block_by(count);
        }

        true
    }

    pub(crate) fn seek_block(&mut self, target_doc: DocId) {
        self.postings.block_cursor.seek_block(target_doc);
    }

    #[cfg(test)]
    pub fn create_for_test(
        doc_and_tfs: &[(DocId, u32)],
        fieldnorms: &[u32],
        similarity_weight: Bm25Weight,
    ) -> TermScorer {
        assert!(!doc_and_tfs.is_empty());
        assert!(
            doc_and_tfs
                .iter()
                .map(|(doc, _tf)| *doc)
                .max()
                .unwrap_or(0u32)
                < fieldnorms.len() as u32
        );
        let segment_postings =
            SegmentPostings::create_from_docs_and_tfs(doc_and_tfs, Some(fieldnorms));
        let fieldnorm_reader = FieldNormReader::for_test(fieldnorms);
        TermScorer::new(segment_postings, fieldnorm_reader, similarity_weight)
    }

    /// See `FreqReadingOption`.
    pub(crate) fn freq_reading_option(&self) -> FreqReadingOption {
        self.postings.block_cursor.freq_reading_option()
    }

    /// Returns the maximum score for the current block.
    ///
    /// In some rare case, the result may not be exact. In this case a lower value is returned,
    /// (and may lead us to return a lesser document).
    ///
    /// At index time, we store the (fieldnorm_id, term frequency) pair that maximizes the
    /// score assuming the average fieldnorm computed on this segment.
    ///
    /// Though extremely rare, it is theoretically possible that the actual average fieldnorm
    /// is different enough from the current segment average fieldnorm that the maximum over a
    /// specific is achieved on a different document.
    ///
    /// (The result is on the other hand guaranteed to be correct if there is only one segment).
    pub fn block_max_score(&mut self) -> Score {
        if let Some(weight) = &self.block_max_weight_override {
            let block = &mut self.postings.block_cursor;
            if !block.block_is_loaded()
                && matches!(
                    block.skip_reader().block_info(),
                    crate::postings::BlockInfo::VInt { .. }
                )
            {
                return Score::INFINITY;
            }
            block.block_max_score(&self.fieldnorm_reader, weight)
        } else {
            self.postings
                .block_cursor
                .block_max_score(&self.fieldnorm_reader, &self.similarity_weight)
        }
    }

    pub(crate) fn block_max_score_up_to(&mut self, target: DocId) -> (Score, DocId) {
        self.postings.block_cursor.block_max_score_up_to(
            target,
            &self.fieldnorm_reader,
            &self.similarity_weight,
        )
    }

    pub fn term_freq(&self) -> u32 {
        self.postings.term_freq()
    }

    pub fn fieldnorm_id(&self) -> u8 {
        self.postings
            .block_cursor
            .fieldnorm_id_at(self.postings.block_offset(), &self.fieldnorm_reader)
    }

    pub fn fieldnorm(&self) -> u32 {
        self.postings
            .block_cursor
            .fieldnorm_at(self.postings.block_offset(), &self.fieldnorm_reader)
    }

    pub fn explain(&self) -> Explanation {
        let fieldnorm_id = self.fieldnorm_id();
        let term_freq = self.term_freq();
        self.similarity_weight.explain(fieldnorm_id, term_freq)
    }

    pub fn max_score(&self) -> Score {
        self.similarity_weight.max_score()
    }

    pub fn last_doc_in_block(&self) -> DocId {
        self.postings.block_cursor.skip_reader().last_doc_in_block()
    }

    /// Returns a mutable reference to the underlying block cursor.
    pub(crate) fn block_cursor(&mut self) -> &mut BlockSegmentPostings {
        &mut self.postings.block_cursor
    }

    /// Returns a reference to the fieldnorm reader for batch lookups.
    pub(crate) fn fieldnorm_reader(&self) -> &FieldNormReader {
        &self.fieldnorm_reader
    }

    /// Returns a reference to the BM25 weight for batch score computation.
    pub(crate) fn bm25_weight(&self) -> &Bm25Weight {
        &self.similarity_weight
    }
}

impl DocSet for TermScorer {
    #[inline]
    fn advance(&mut self) -> DocId {
        self.postings.advance()
    }

    #[inline]
    fn seek(&mut self, target: DocId) -> DocId {
        debug_assert!(target >= self.doc());
        self.postings.seek(target)
    }

    #[inline]
    fn doc(&self) -> DocId {
        self.postings.doc()
    }

    fn size_hint(&self) -> u32 {
        self.postings.size_hint()
    }

    // TODO
    // It is probably possible to optimize fill_bitset_block for TermScorer,
    // working directly with the blocks, enabling vectorization.
    // I did not manage to get a performance improvement on Mac ARM,
    // and do not have access to x86 to investigate.
}

impl Scorer for TermScorer {
    #[inline]
    fn score(&mut self) -> Score {
        let term_freq = self.term_freq();
        if self.postings.block_cursor.has_term_norms() {
            let fieldnorm = self.fieldnorm();
            self.similarity_weight.score_fieldnorm(fieldnorm, term_freq)
        } else {
            let fieldnorm_id = self.fieldnorm_id();
            self.similarity_weight.score(fieldnorm_id, term_freq)
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::index::{Bm25Params, SegmentId};
    use crate::indexer::index_writer::MEMORY_BUDGET_NUM_BYTES_MIN;
    use crate::merge_policy::NoMergePolicy;
    use crate::postings::compression::COMPRESSION_BLOCK_SIZE;
    use crate::query::term_query::TermScorer;
    use crate::query::{Bm25Weight, EnableScoring, Scorer, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, TEXT};
    use crate::{
        assert_nearly_equals, DocId, DocSet, Index, IndexWriter, Score, Searcher, Term, TERMINATED,
    };

    #[test]
    fn test_term_scorer_max_score() -> crate::Result<()> {
        let bm25_weight = Bm25Weight::for_one_term(3, 6, 10.0, Bm25Params::default());
        let mut term_scorer = TermScorer::create_for_test(
            &[(2, 3), (3, 12), (7, 8)],
            &[0, 0, 10, 12, 0, 0, 0, 100],
            bm25_weight,
        );
        let max_scorer = term_scorer.max_score();
        crate::assert_nearly_equals!(max_scorer, 1.3990127);
        assert_eq!(term_scorer.doc(), 2);
        assert_eq!(term_scorer.term_freq(), 3);
        assert_nearly_equals!(term_scorer.block_max_score(), 1.3676447);
        assert_nearly_equals!(term_scorer.score(), 1.0892314);
        assert_eq!(term_scorer.advance(), 3);
        assert_eq!(term_scorer.doc(), 3);
        assert_eq!(term_scorer.term_freq(), 12);
        assert_nearly_equals!(term_scorer.score(), 1.3676447);
        assert_eq!(term_scorer.advance(), 7);
        assert_eq!(term_scorer.doc(), 7);
        assert_eq!(term_scorer.term_freq(), 8);
        assert_nearly_equals!(term_scorer.score(), 0.72015285);
        assert_eq!(term_scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_term_scorer_shallow_advance() -> crate::Result<()> {
        let bm25_weight = Bm25Weight::for_one_term(300, 1024, 10.0, Bm25Params::default());
        let mut doc_and_tfs = vec![];
        for i in 0u32..300u32 {
            let doc = i * 10;
            doc_and_tfs.push((doc, 1u32 + doc % 3u32));
        }
        let fieldnorms: Vec<u32> = std::iter::repeat_n(10u32, 3_000).collect();
        let mut term_scorer = TermScorer::create_for_test(&doc_and_tfs, &fieldnorms, bm25_weight);
        assert_eq!(term_scorer.doc(), 0u32);
        term_scorer.seek_block(1289);
        assert_eq!(term_scorer.doc(), 0u32);
        term_scorer.seek(1289);
        assert_eq!(term_scorer.doc(), 1290);
        Ok(())
    }

    proptest! {
        #[test]
        fn test_term_scorer_block_max_score(term_freqs_fieldnorms in proptest::collection::vec((1u32..10u32, 0u32..100u32), 80..300)) {
        let term_doc_freq = term_freqs_fieldnorms.len();
         let doc_tfs: Vec<(u32, u32)> = term_freqs_fieldnorms.iter()
                   .cloned()
                  .enumerate()
                  .map(|(doc, (tf, _))| (doc as u32, tf))
                  .collect();

         let mut fieldnorms: Vec<u32> = vec![];
         for item in term_freqs_fieldnorms.iter().take(term_doc_freq) {
             let (tf, num_extra_terms) = item;
             fieldnorms.push(tf + num_extra_terms);
         }
         let average_fieldnorm = fieldnorms
             .iter()
             .cloned()
             .sum::<u32>() as Score / term_doc_freq as Score;
             // Average fieldnorm is over the entire index,
             // not necessarily the docs that are in the posting list.
             // For this reason we multiply by 1.1 to make a realistic value.
         let bm25_weight = Bm25Weight::for_one_term(term_doc_freq as u64,
            term_doc_freq as u64 * 10u64,
            average_fieldnorm,
            Bm25Params::default());

         let mut term_scorer =
              TermScorer::create_for_test(&doc_tfs[..], &fieldnorms[..], bm25_weight);

         let docs: Vec<DocId> = (0..term_doc_freq).map(|doc| doc as DocId).collect();
         for block in docs.chunks(COMPRESSION_BLOCK_SIZE) {
             let block_max_score: Score = term_scorer.block_max_score();
             let mut block_max_score_computed: Score = 0.0;
             for &doc in block {
                assert_eq!(term_scorer.doc(), doc);
                block_max_score_computed = block_max_score_computed.max(term_scorer.score());
                term_scorer.advance();
             }
             assert_nearly_equals!(block_max_score_computed, block_max_score);
         }
        }
    }

    #[test]
    fn test_block_wand() {
        let mut doc_tfs: Vec<(u32, u32)> = vec![];
        for doc in 0u32..128u32 {
            doc_tfs.push((doc, 1u32));
        }
        for doc in 128u32..256u32 {
            doc_tfs.push((doc, if doc == 200 { 2u32 } else { 1u32 }));
        }
        doc_tfs.push((256, 1u32));
        doc_tfs.push((257, 3u32));
        doc_tfs.push((258, 1u32));

        let fieldnorms: Vec<u32> = std::iter::repeat_n(20u32, 300).collect();
        let bm25_weight = Bm25Weight::for_one_term(10, 129, 20.0, Bm25Params::default());
        let mut docs = TermScorer::create_for_test(&doc_tfs[..], &fieldnorms[..], bm25_weight);
        assert_nearly_equals!(docs.block_max_score(), 2.5161593);
        docs.seek_block(135);
        assert_nearly_equals!(docs.block_max_score(), 3.4597192);
        docs.seek_block(256);
        // the block is not loaded yet.
        assert_nearly_equals!(docs.block_max_score(), 5.2971773);
        assert_eq!(256, docs.seek(256));
        assert_nearly_equals!(docs.block_max_score(), 3.9539647);
    }

    fn test_block_wand_aux(term_query: &TermQuery, searcher: &Searcher) -> crate::Result<()> {
        let term_weight =
            term_query.specialized_weight(EnableScoring::enabled_from_searcher(searcher))?;
        for reader in searcher.segment_readers() {
            let mut block_max_scores = vec![];
            let mut block_max_scores_b = vec![];
            let mut docs = vec![];
            {
                let mut term_scorer = term_weight.term_scorer_for_test(reader, 1.0)?.unwrap();
                while term_scorer.doc() != TERMINATED {
                    let mut score = term_scorer.score();
                    docs.push(term_scorer.doc());
                    for _ in 0..128 {
                        score = score.max(term_scorer.score());
                        if term_scorer.advance() == TERMINATED {
                            break;
                        }
                    }
                    block_max_scores.push(score);
                }
            }
            {
                let mut term_scorer = term_weight.term_scorer_for_test(reader, 1.0)?.unwrap();
                for d in docs {
                    term_scorer.seek_block(d);
                    block_max_scores_b.push(term_scorer.block_max_score());
                }
            }
            for (l, r) in block_max_scores
                .iter()
                .cloned()
                .zip(block_max_scores_b.iter().cloned())
            {
                assert_nearly_equals!(l, r);
            }
        }
        Ok(())
    }

    #[ignore]
    #[test]
    fn test_block_wand_long_test() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter =
            index.writer_with_num_threads(3, 3 * MEMORY_BUDGET_NUM_BYTES_MIN)?;
        use rand::Rng;
        let mut rng = rand::rng();
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for _ in 0..3_000 {
            let term_freq = rng.random_range(1..10000);
            let words: Vec<&str> = std::iter::repeat_n("bbbb", term_freq).collect();
            let text = words.join(" ");
            writer.add_document(doc!(text_field=>text))?;
        }
        writer.commit()?;
        let term_query = TermQuery::new(
            Term::from_field_text(text_field, "bbbb"),
            IndexRecordOption::WithFreqs,
        );
        let segment_ids: Vec<SegmentId>;
        let reader = index.reader()?;
        {
            let searcher = reader.searcher();
            segment_ids = searcher
                .segment_readers()
                .iter()
                .map(|segment| segment.segment_id())
                .collect();
            test_block_wand_aux(&term_query, &searcher)?;
        }
        writer.merge(&segment_ids[..]).wait().unwrap();
        {
            reader.reload()?;
            let searcher = reader.searcher();
            assert_eq!(searcher.segment_readers().len(), 1);
            test_block_wand_aux(&term_query, &searcher)?;
        }
        Ok(())
    }

    #[test]
    fn test_for_each_block_slice_with_term_norms() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let text = schema.add_text_field(
            "text",
            TEXT.set_indexing_options(
                TEXT.get_indexing_options()
                    .unwrap()
                    .clone()
                    .set_pnorms(true),
            ),
        );
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        for id in 0..250 {
            let padding = "hello ".repeat((id % 13) + 1);
            writer.add_document(doc!(text => format!("rust {}", padding)))?;
        }
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let term = Term::from_field_text(text, "rust");
        let term_query = TermQuery::new(term, IndexRecordOption::WithFreqs);
        let term_weight =
            term_query.specialized_weight(EnableScoring::enabled_from_searcher(&searcher))?;

        // 1. Collect reference doc IDs and scores via scalar advance() + score()
        let mut ref_docs = Vec::new();
        let mut ref_scores = Vec::new();
        {
            let mut scalar_scorer = term_weight
                .term_scorer_for_test(segment_reader, 1.0)?
                .unwrap();
            while scalar_scorer.doc() != TERMINATED {
                ref_docs.push(scalar_scorer.doc());
                ref_scores.push(scalar_scorer.score());
                scalar_scorer.advance();
            }
        }
        assert_eq!(ref_docs.len(), 250);

        // 2. Collect doc IDs and scores via for_each_block_slice_with_term_norms in chunks
        let mut batch_docs = Vec::new();
        let mut batch_scores = Vec::new();
        {
            let mut batch_scorer = term_weight
                .term_scorer_for_test(segment_reader, 1.0)?
                .unwrap();
            assert!(batch_scorer.has_term_norms());

            let mut window_end = 50;
            while batch_scorer.doc() != TERMINATED {
                let handled =
                    batch_scorer.for_each_block_slice_with_term_norms(window_end, |docs, sc| {
                        batch_docs.extend_from_slice(docs);
                        batch_scores.extend_from_slice(sc);
                    });
                assert!(handled);
                window_end += 50;
            }
        }

        assert_eq!(batch_docs, ref_docs);
        assert_eq!(batch_scores.len(), ref_scores.len());
        for (b, r) in batch_scores.iter().zip(&ref_scores) {
            assert_nearly_equals!(*b, *r);
        }

        Ok(())
    }
}
