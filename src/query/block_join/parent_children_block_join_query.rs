use std::fmt;
use std::sync::Arc;

use super::ParentBitSetProducer;
use crate::core::searcher::Searcher;
use crate::index::SegmentId;
use crate::query::{EmptyScorer, EnableScoring, Explanation, Query, QueryClone, Scorer, Weight};
use crate::{DocId, DocSet, Result, Score, SegmentReader, TERMINATED};

/// A query that returns all matching child documents for one specific
/// parent document (by parent segment ID + doc_id).
///
/// The parent doc must be indexed in "block" form with its children
/// preceding it, and parent_filter must mark docs that are parents.
pub struct ParentChildrenBlockJoinQuery {
    parent_filter: Arc<dyn ParentBitSetProducer>,
    child_query: Box<dyn Query>,
    /// The segment ID of the parent doc's segment.
    parent_segment_id: SegmentId,
    /// The doc_id of the parent within that segment.
    parent_doc_id: DocId,
}

#[allow(unused)]
impl ParentChildrenBlockJoinQuery {
    /// Create a new parent->children block-join query.
    ///
    /// - parent_filter: marks which docs are parents
    /// - child_query:   the underlying child query
    /// - parent_segment_id: which segment the parent doc is in
    /// - parent_doc_id: the doc ID of that parent in that segment
    pub fn new(
        parent_filter: Arc<dyn ParentBitSetProducer>,
        child_query: Box<dyn Query>,
        parent_segment_id: SegmentId,
        parent_doc_id: DocId,
    ) -> Self {
        ParentChildrenBlockJoinQuery {
            parent_filter,
            child_query,
            parent_segment_id,
            parent_doc_id,
        }
    }
}

impl Clone for ParentChildrenBlockJoinQuery {
    fn clone(&self) -> Self {
        Self {
            parent_filter: Arc::clone(&self.parent_filter),
            child_query: self.child_query.box_clone(),
            parent_segment_id: self.parent_segment_id,
            parent_doc_id: self.parent_doc_id,
        }
    }
}

impl fmt::Debug for ParentChildrenBlockJoinQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParentChildrenBlockJoinQuery(segment_id={:?}, parent_doc={}, ...)",
            self.parent_segment_id, self.parent_doc_id
        )
    }
}

impl Query for ParentChildrenBlockJoinQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> Result<Box<dyn Weight>> {
        // Build the child weight right here:
        let child_weight = self.child_query.weight(enable_scoring)?;
        Ok(Box::new(ParentChildrenBlockJoinWeight {
            parent_filter: Arc::clone(&self.parent_filter),
            child_weight,
            parent_segment_id: self.parent_segment_id,
            parent_doc_id: self.parent_doc_id,
        }))
    }

    fn explain(&self, _searcher: &Searcher, _doc_id: crate::DocAddress) -> Result<Explanation> {
        // In Lucene's version, it says "Not implemented". We'll do the same.
        Ok(Explanation::new(
            "Not implemented in ParentChildrenBlockJoinQuery",
            0.0,
        ))
    }

    fn count(&self, searcher: &Searcher) -> Result<usize> {
        // Only the single segment that matches parent_segment_id can have child matches
        let seg_reader_opt = searcher
            .segment_readers()
            .iter()
            .find(|sr| sr.segment_id() == self.parent_segment_id);

        if let Some(reader) = seg_reader_opt {
            let w = self.weight(EnableScoring::disabled_from_searcher(searcher))?;
            let c = w.count(reader)?;
            Ok(c as usize)
        } else {
            Ok(0)
        }
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a crate::schema::Term, bool)) {
        // Forward down to child query
        self.child_query.query_terms(visitor);
    }
}

struct ParentChildrenBlockJoinWeight {
    parent_filter: Arc<dyn ParentBitSetProducer>,
    child_weight: Box<dyn Weight>,
    parent_segment_id: SegmentId,
    parent_doc_id: DocId,
}

impl Weight for ParentChildrenBlockJoinWeight {
    fn scorer(&self, reader: &SegmentReader, boost: f32) -> Result<Box<dyn Scorer>> {
        // If this segment doesn't match the parent's segment_id, return empty
        if reader.segment_id() != self.parent_segment_id {
            return Ok(Box::new(EmptyScorer));
        }

        // If doc=0, no children can precede
        if self.parent_doc_id == 0 {
            return Ok(Box::new(EmptyScorer));
        }

        // Confirm that doc=parent_doc_id is actually a parent
        let parent_bits = self.parent_filter.produce(reader)?;
        if !parent_bits.contains(self.parent_doc_id) {
            // Means user gave an invalid parent doc
            return Ok(Box::new(EmptyScorer));
        }

        // The preceding parent doc is found via prev_set_bit(parent_doc_id - 1).
        // Then children are from that doc+1 up to parent_doc_id-1.
        let prev_parent = parent_bits.prev_set_bit(self.parent_doc_id - 1);
        let first_child = if prev_parent == u32::MAX {
            0
        } else {
            prev_parent + 1
        };

        // If the range is empty => no child docs
        if first_child >= self.parent_doc_id {
            return Ok(Box::new(EmptyScorer));
        }

        // Build the underlying child scorer
        let child_scorer = self.child_weight.scorer(reader, boost)?;
        if child_scorer.doc() == TERMINATED {
            return Ok(Box::new(EmptyScorer));
        }

        // Wrap in a bounding scorer that only returns docs in [first_child .. parent_doc_id)
        Ok(Box::new(ParentChildrenBlockJoinScorer {
            inner: child_scorer,
            bound_start: first_child,
            bound_end: self.parent_doc_id,
            done: false,
        }))
    }

    fn explain(&self, _reader: &SegmentReader, _doc_id: DocId) -> Result<Explanation> {
        // For child doc explanations, you can implement or do no-match
        Ok(Explanation::new("No explanation implemented", 0.0))
    }

    fn count(&self, reader: &SegmentReader) -> Result<u32> {
        let mut sc = self.scorer(reader, 1.0)?;
        let mut cnt = 0u32;
        let mut doc_id = sc.advance();
        while doc_id != TERMINATED {
            cnt += 1;
            doc_id = sc.advance();
        }
        Ok(cnt)
    }

    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> Result<()> {
        let mut sc = self.scorer(reader, 1.0)?;
        let mut _current_threshold = threshold;
        let mut doc_id = sc.advance();
        while doc_id != TERMINATED {
            let s = sc.score();
            _current_threshold = callback(doc_id, s);
            doc_id = sc.advance();
        }
        Ok(())
    }
}
struct ParentChildrenBlockJoinScorer {
    inner: Box<dyn Scorer>,
    bound_start: DocId,
    bound_end: DocId,
    done: bool,
}
impl DocSet for ParentChildrenBlockJoinScorer {
    fn advance(&mut self) -> DocId {
        if self.done {
            return TERMINATED;
        }
        let mut d = self.inner.advance();
        // Skip child docs below bound_start
        while d != TERMINATED && d < self.bound_start {
            d = self.inner.advance();
        }
        // If we exceed bound_end, we're done
        if d == TERMINATED || d >= self.bound_end {
            self.done = true;
            TERMINATED
        } else {
            d
        }
    }

    fn doc(&self) -> DocId {
        if self.done {
            TERMINATED
        } else {
            let d = self.inner.doc();
            if d >= self.bound_end {
                TERMINATED
            } else {
                d
            }
        }
    }

    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl Scorer for ParentChildrenBlockJoinScorer {
    fn score(&mut self) -> Score {
        if self.done || self.doc() == TERMINATED {
            0.0
        } else {
            self.inner.score()
        }
    }
}
