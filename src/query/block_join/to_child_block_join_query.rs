use std::cell::{Cell, RefCell};
use std::fmt;
use std::sync::Arc;

use common::BitSet;

use crate::core::searcher::Searcher;
use crate::query::{EmptyScorer, EnableScoring, Explanation, Query, QueryClone, Scorer, Weight};
use crate::schema::Term;
use crate::{DocAddress, DocId, DocSet, Result, Score, SegmentReader, TERMINATED};

use super::ParentBitSetProducer;

pub struct ToChildBlockJoinQuery {
    parent_query: Box<dyn Query>,
    parent_bitset_producer: Arc<dyn ParentBitSetProducer>,
}

impl Clone for ToChildBlockJoinQuery {
    fn clone(&self) -> Self {
        ToChildBlockJoinQuery {
            parent_query: self.parent_query.box_clone(),
            parent_bitset_producer: Arc::clone(&self.parent_bitset_producer),
        }
    }
}

impl fmt::Debug for ToChildBlockJoinQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ToChildBlockJoinQuery(...)")
    }
}

#[allow(unused)]
impl ToChildBlockJoinQuery {
    pub fn new(
        parent_query: Box<dyn Query>,
        parent_bitset_producer: Arc<dyn ParentBitSetProducer>,
    ) -> Self {
        Self {
            parent_query,
            parent_bitset_producer,
        }
    }
}

struct ToChildBlockJoinWeight {
    parent_weight: Box<dyn Weight>,
    parent_bits: Arc<dyn ParentBitSetProducer>,
}

struct ToChildBlockJoinScorer {
    // Must be wrapped in RefCell if you call .advance() from a &self context
    parent_scorer: RefCell<Box<dyn Scorer>>,
    bits: BitSet,
    boost: f32,

    // Fields we mutate from &self => store in Cell
    doc_done: Cell<bool>,
    init: Cell<bool>,
    current_doc: Cell<DocId>,
    current_parent: Cell<DocId>,
}

impl Query for ToChildBlockJoinQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> Result<Box<dyn Weight>> {
        let pw = self.parent_query.weight(enable_scoring)?;
        Ok(Box::new(ToChildBlockJoinWeight {
            parent_weight: pw,
            parent_bits: Arc::clone(&self.parent_bitset_producer),
        }))
    }

    fn explain(&self, searcher: &Searcher, doc_addr: DocAddress) -> Result<Explanation> {
        let sr = searcher.segment_reader(doc_addr.segment_ord);
        let w = self.weight(EnableScoring::enabled_from_searcher(searcher))?;
        w.explain(sr, doc_addr.doc_id)
    }

    fn count(&self, searcher: &Searcher) -> Result<usize> {
        let w = self.weight(EnableScoring::disabled_from_searcher(searcher))?;
        let mut c = 0usize;
        for seg in searcher.segment_readers().iter() {
            let sub_count = w.count(seg)? as usize;
            c += sub_count;
        }
        Ok(c)
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        self.parent_query.query_terms(visitor);
    }
}
impl Weight for ToChildBlockJoinWeight {
    fn scorer(&self, reader: &SegmentReader, boost: f32) -> Result<Box<dyn Scorer>> {
        let ps = self.parent_weight.scorer(reader, boost)?;
        let bits = self.parent_bits.produce(reader)?;
        if bits.is_empty() {
            return Ok(Box::new(EmptyScorer));
        }
        Ok(Box::new(ToChildBlockJoinScorer {
            parent_scorer: RefCell::new(ps), // <-- wrap in RefCell
            bits,
            boost,
            doc_done: Cell::new(false),            // <-- wrap bool in Cell
            init: Cell::new(false),                // <-- wrap bool in Cell
            current_doc: Cell::new(TERMINATED),    // <-- wrap u32 in Cell
            current_parent: Cell::new(TERMINATED), // <-- wrap u32 in Cell
        }))
    }

    fn explain(&self, reader: &SegmentReader, doc_id: DocId) -> Result<Explanation> {
        let mut sc = self.scorer(reader, 1.0)?;

        // "Advance first" approach
        let mut current = sc.advance();
        while current < doc_id && current != TERMINATED {
            current = sc.advance();
        }
        if current != doc_id {
            return Ok(Explanation::new("Not a match", 0.0));
        }
        let val = sc.score();
        let mut ex = Explanation::new_with_string("ToChildBlockJoin".to_string(), val);
        ex.add_detail(Explanation::new_with_string(
            "child doc matched".to_string(),
            val,
        ));
        Ok(ex)
    }

    fn count(&self, reader: &SegmentReader) -> Result<u32> {
        let mut sc = self.scorer(reader, 1.0)?;
        let mut c = 0;

        // Advance first, then loop
        let mut doc_id = sc.advance();
        while doc_id != TERMINATED {
            c += 1;
            doc_id = sc.advance();
        }
        Ok(c)
    }

    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> Result<()> {
        let mut scorer = self.scorer(reader, 1.0)?;
        let mut _current_threshold = threshold;

        // Advance first, then loop
        let mut doc_id = scorer.advance();
        while doc_id != TERMINATED {
            let score = scorer.score();
            _current_threshold = callback(doc_id, score);
            doc_id = scorer.advance();
        }
        Ok(())
    }
}
impl DocSet for ToChildBlockJoinScorer {
    fn advance(&mut self) -> DocId {
        self.advance_doc()
    }

    fn doc(&self) -> DocId {
        if self.doc_done.get() {
            return TERMINATED;
        }
        if !self.init.get() {
            // First invocation => advance once
            let first_doc = self.advance_doc();
            if first_doc == TERMINATED {
                return TERMINATED;
            }
        }
        self.current_doc.get()
    }

    fn size_hint(&self) -> u32 {
        self.bits.len() as u32
    }
}
impl Scorer for ToChildBlockJoinScorer {
    fn score(&mut self) -> Score {
        if self.doc_done.get() || self.current_parent.get() == TERMINATED {
            0.0
        } else {
            // Score is simply the parent's score * boost
            let pscore = self.parent_scorer.borrow_mut().score();
            pscore * self.boost
        }
    }
}
impl ToChildBlockJoinScorer {
    fn advance_doc(&self) -> DocId {
        // If done, stop
        if self.doc_done.get() {
            return TERMINATED;
        }
        // First time => set init + read the parent's doc
        if !self.init.get() {
            self.init.set(true);
            let parent_doc = self.parent_scorer.borrow().doc();
            self.current_parent.set(parent_doc);

            if parent_doc == TERMINATED {
                self.doc_done.set(true);
                return TERMINATED;
            }
            // Move to that parent's first child
            self.advance_to_first_child_of_parent()
        } else {
            // Normal “go to next child doc”
            let next_child = self.current_doc.get().saturating_add(1);
            // If we reached or passed the parent doc, move to next parent
            if next_child >= self.current_parent.get() {
                let mut ps = self.parent_scorer.borrow_mut();
                let next_parent = ps.advance();
                self.current_parent.set(next_parent);
                if next_parent == TERMINATED {
                    self.doc_done.set(true);
                    return TERMINATED;
                }
                // Advance to the child range for that new parent
                return self.advance_to_first_child_of_parent();
            }
            self.current_doc.set(next_child);
            next_child
        }
    }

    fn advance_to_first_child_of_parent(&self) -> DocId {
        loop {
            let p = self.current_parent.get();
            if p == TERMINATED {
                // No more parents at all
                self.doc_done.set(true);
                return TERMINATED;
            }

            // If parent is doc=0, it has no preceding doc range => no children
            if p == 0 {
                // Move to next parent
                let mut ps = self.parent_scorer.borrow_mut();
                let next_parent = ps.advance();
                self.current_parent.set(next_parent);
                continue;
            }

            // Find the previous parent's position
            let prev_parent = self.bits.prev_set_bit(p - 1);
            // Start children just after that, or at doc=0 if none
            let first_child = if prev_parent == u32::MAX {
                0
            } else {
                prev_parent + 1
            };

            // If there's no space for children in [first_child..p),
            // skip to the next parent
            if first_child >= p {
                let mut ps = self.parent_scorer.borrow_mut();
                let next_parent = ps.advance();
                self.current_parent.set(next_parent);
                continue;
            }

            // Found a valid child range => set current_doc to that start
            self.current_doc.set(first_child);
            return first_child;
        }
    }
}
