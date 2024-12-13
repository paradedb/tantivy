use crate::docset::{DocSet, COLLECT_BLOCK_BUFFER_LEN, TERMINATED};
use crate::index::SegmentReader;
use crate::query::boost_query::BoostScorer;
use crate::query::explanation::does_not_match;
use crate::query::{EnableScoring, Explanation, InvertedIndexRangeWeight, Query, Scorer, Weight};
use crate::{Ctid, DocId, Score, INVALID_CTID};
use std::ops::Bound;

/// Query that matches all of the documents.
///
/// All of the document get the score 1.0.
#[derive(Clone, Debug)]
pub struct AllQuery;

impl Query for AllQuery {
    fn weight(&self, scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        match scoring
            .schema()
            .fields()
            .find(|(_, field)| field.is_key_field())
        {
            Some((key_field, _)) => Ok(Box::new(InvertedIndexRangeWeight::new(
                key_field,
                &Bound::Unbounded,
                &Bound::Unbounded,
                None,
            ))),
            None => {
                if cfg!(test) {
                    // only used for testing
                    Ok(Box::new(AllWeight))
                } else {
                    panic!("`AllQuery` requires a designated key field in the schema")
                }
            }
        }
    }
}

/// Weight associated with the `AllQuery` query.
pub struct AllWeight;

impl Weight for AllWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let all_scorer = AllScorer::new(reader.max_doc());
        Ok(Box::new(BoostScorer::new(all_scorer, boost)))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        if doc >= reader.max_doc() {
            return Err(does_not_match(doc));
        }
        Ok(Explanation::new("AllQuery", 1.0))
    }
}

/// Scorer associated with the `AllQuery` query.
pub struct AllScorer {
    doc: DocId,
    max_doc: DocId,
}

impl AllScorer {
    /// Creates a new AllScorer with `max_doc` docs.
    pub fn new(max_doc: DocId) -> AllScorer {
        AllScorer { doc: 0u32, max_doc }
    }
}

impl DocSet for AllScorer {
    #[inline(always)]
    fn advance(&mut self) -> DocId {
        if self.doc + 1 >= self.max_doc {
            self.doc = TERMINATED;
            return TERMINATED;
        }
        self.doc += 1;
        self.doc
    }

    fn fill_buffer(
        &mut self,
        buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN],
        ctid_buffer: &mut [Ctid; COLLECT_BLOCK_BUFFER_LEN],
    ) -> usize {
        if self.doc() == TERMINATED {
            return 0;
        }
        let is_safe_distance = self.doc() + (buffer.len() as u32) < self.max_doc;
        if is_safe_distance {
            let num_items = buffer.len();
            for (buffer_val, ctid_val) in buffer.iter_mut().zip(ctid_buffer.iter_mut()) {
                *buffer_val = self.doc();
                *ctid_val = INVALID_CTID;
                
                self.doc += 1;
            }
            num_items
        } else {
            for (i, (buffer_val, ctid_val)) in buffer.iter_mut().zip(ctid_buffer.iter_mut()).enumerate() {
                *buffer_val = self.doc();
                *ctid_val = INVALID_CTID;
                if self.advance() == TERMINATED {
                    return i + 1;
                }
            }
            buffer.len()
        }
    }

    #[inline(always)]
    fn doc(&self) -> DocId {
        self.doc
    }

    fn ctid(&self) -> Ctid {
        // hardcoded for tests
        INVALID_CTID
    }

    fn size_hint(&self) -> u32 {
        self.max_doc
    }
}

impl Scorer for AllScorer {
    fn score(&mut self) -> (Score, Ctid) {
        if cfg!(test) {
            (1.0, INVALID_CTID)
        } else {
            unreachable!("AllScorer is not directly supported anymore and should not be called")
        }
    }
}

pub mod postings_all_scorer {
    use crate::fastfield::AliveBitSet;
    use crate::postings::SegmentPostings;
    use crate::query::Scorer;
    use crate::{Ctid, DocId, DocSet, Score, COLLECT_BLOCK_BUFFER_LEN};

    pub struct AllScorer {
        postings: SegmentPostings,
    }

    impl AllScorer {
        pub fn new(postings: SegmentPostings) -> AllScorer {
            Self { postings }
        }
    }

    impl DocSet for AllScorer {
        fn advance(&mut self) -> DocId {
            self.postings.advance()
        }

        fn seek(&mut self, target: DocId) -> DocId {
            self.postings.seek(target)
        }

        fn fill_buffer(
            &mut self,
            buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN],
            ctid_buffer: &mut [Ctid; COLLECT_BLOCK_BUFFER_LEN],
        ) -> usize {
            self.postings.fill_buffer(buffer, ctid_buffer)
        }

        fn doc(&self) -> DocId {
            self.postings.doc()
        }

        fn ctid(&self) -> Ctid {
            self.postings.ctid()
        }

        fn size_hint(&self) -> u32 {
            self.postings.size_hint()
        }

        fn count(&mut self, alive_bitset: &AliveBitSet) -> u32 {
            self.postings.count(alive_bitset)
        }

        fn count_including_deleted(&mut self) -> u32 {
            self.postings.count_including_deleted()
        }
    }

    impl Scorer for AllScorer {
        fn score(&mut self) -> (Score, Ctid) {
            (1.0, self.postings.ctid())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AllQuery;
    use crate::docset::{DocSet, COLLECT_BLOCK_BUFFER_LEN, TERMINATED};
    use crate::query::{AllScorer, EnableScoring, Query};
    use crate::schema::{Schema, TEXT};
    use crate::{Index, IndexWriter};

    fn create_test_index() -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(field=>"aaa"))?;
        index_writer.add_document(doc!(field=>"bbb"))?;
        index_writer.commit()?;
        index_writer.add_document(doc!(field=>"ccc"))?;
        index_writer.commit()?;
        Ok(index)
    }

    #[test]
    fn test_all_query() -> crate::Result<()> {
        let index = create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let weight = AllQuery.weight(EnableScoring::disabled_from_schema(&index.schema()))?;
        {
            let reader = searcher.segment_reader(0);
            let mut scorer = weight.scorer(reader, 1.0)?;
            assert_eq!(scorer.doc(), 0u32);
            assert_eq!(scorer.advance(), 1u32);
            assert_eq!(scorer.doc(), 1u32);
            assert_eq!(scorer.advance(), TERMINATED);
        }
        {
            let reader = searcher.segment_reader(1);
            let mut scorer = weight.scorer(reader, 1.0)?;
            assert_eq!(scorer.doc(), 0u32);
            assert_eq!(scorer.advance(), TERMINATED);
        }
        Ok(())
    }

    #[test]
    fn test_all_query_with_boost() -> crate::Result<()> {
        let index = create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let weight = AllQuery.weight(EnableScoring::disabled_from_schema(searcher.schema()))?;
        let reader = searcher.segment_reader(0);
        {
            let mut scorer = weight.scorer(reader, 2.0)?;
            assert_eq!(scorer.doc(), 0u32);
            assert_eq!(scorer.score().0, 2.0);
        }
        {
            let mut scorer = weight.scorer(reader, 1.5)?;
            assert_eq!(scorer.doc(), 0u32);
            assert_eq!(scorer.score().0, 1.5);
        }
        Ok(())
    }

    #[test]
    pub fn test_fill_buffer() {
        let mut postings = AllScorer {
            doc: 0u32,
            max_doc: COLLECT_BLOCK_BUFFER_LEN as u32 * 2 + 9,
        };
        let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];
        let mut ctid_buffer = [(0, 0); COLLECT_BLOCK_BUFFER_LEN];
        assert_eq!(
            postings.fill_buffer(&mut buffer, &mut ctid_buffer),
            COLLECT_BLOCK_BUFFER_LEN
        );
        for i in 0u32..COLLECT_BLOCK_BUFFER_LEN as u32 {
            assert_eq!(buffer[i as usize], i);
        }
        assert_eq!(
            postings.fill_buffer(&mut buffer, &mut ctid_buffer),
            COLLECT_BLOCK_BUFFER_LEN
        );
        for i in 0u32..COLLECT_BLOCK_BUFFER_LEN as u32 {
            assert_eq!(buffer[i as usize], i + COLLECT_BLOCK_BUFFER_LEN as u32);
        }
        assert_eq!(postings.fill_buffer(&mut buffer, &mut ctid_buffer), 9);
    }
}
