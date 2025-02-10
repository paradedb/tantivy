use std::cell::{Cell, RefCell};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use common::BitSet;

use crate::core::searcher::Searcher;
use crate::query::{EnableScoring, Explanation, Query, QueryClone, Scorer, Weight};
use crate::schema::Term;
use crate::{DocAddress, DocId, DocSet, Result, Score, SegmentReader, TantivyError, TERMINATED};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScoreMode {
    Total,
    #[default]
    Avg,
    Max,
    Min,
    None,
}

impl FromStr for ScoreMode {
    type Err = TantivyError;

    fn from_str(mode: &str) -> std::result::Result<Self, Self::Err> {
        match mode.trim().to_lowercase().as_str() {
            "none" => Ok(ScoreMode::None),
            "avg" => Ok(ScoreMode::Avg),
            "max" => Ok(ScoreMode::Max),
            "min" => Ok(ScoreMode::Min),
            "total" => Ok(ScoreMode::Total),
            other => Err(TantivyError::InvalidArgument(format!(
                "Unrecognized nested score_mode: '{}'",
                other
            ))),
        }
    }
}

impl ScoreMode {
    fn combine(&self, child_score: f32, accum: f32, _count: u32) -> f32 {
        

        match self {
            ScoreMode::None => 0.0,
            ScoreMode::Total => accum + child_score,
            ScoreMode::Avg => accum + child_score,
            ScoreMode::Max => accum.max(child_score),
            ScoreMode::Min => accum.min(child_score),
        }
    }

    fn finalize_score(&self, sumval: f32, count: u32) -> f32 {
        

        match self {
            ScoreMode::None => 0.0,
            ScoreMode::Total => sumval,
            ScoreMode::Avg => {
                if count == 0 {
                    0.0
                } else {
                    

                    sumval / count as f32
                }
            }
            ScoreMode::Max => sumval,
            ScoreMode::Min => sumval,
        }
    }
}

// ParentBitSetProducer trait
///////////////////////////////////////////////////////////////////////////////

pub trait ParentBitSetProducer: Send + Sync + 'static {
    fn produce(&self, reader: &SegmentReader) -> Result<BitSet>;
}

// EmptyScorer struct
///////////////////////////////////////////////////////////////////////////////

struct EmptyScorer;

impl DocSet for EmptyScorer {
    fn advance(&mut self) -> DocId {
        TERMINATED
    }
    fn doc(&self) -> DocId {
        TERMINATED
    }
    fn size_hint(&self) -> u32 {
        0
    }
}

impl Scorer for EmptyScorer {
    fn score(&mut self) -> Score {
        0.0
    }
}

// ToParentBlockJoinQuery struct
///////////////////////////////////////////////////////////////////////////////

pub struct ToParentBlockJoinQuery {
    child_query: Box<dyn Query>,
    parent_bitset_producer: Arc<dyn ParentBitSetProducer>,
    score_mode: ScoreMode,
}

impl Clone for ToParentBlockJoinQuery {
    fn clone(&self) -> Self {
        Self {
            child_query: self.child_query.box_clone(),
            parent_bitset_producer: Arc::clone(&self.parent_bitset_producer),
            score_mode: self.score_mode,
        }
    }
}

impl fmt::Debug for ToParentBlockJoinQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ToParentBlockJoinQuery(...)")
    }
}

impl ToParentBlockJoinQuery {
    pub fn new(
        child_query: Box<dyn Query>,
        parent_bitset_producer: Arc<dyn ParentBitSetProducer>,
        score_mode: ScoreMode,
    ) -> Self {
        Self {
            child_query,
            parent_bitset_producer,
            score_mode,
        }
    }
}

struct ToParentBlockJoinWeight {
    child_weight: Box<dyn Weight>,
    parent_bits: Arc<dyn ParentBitSetProducer>,
    score_mode: ScoreMode,
}

/// The scorer that lazily initializes on doc().
struct ToParentBlockJoinScorer {
    child_scorer: RefCell<Box<dyn Scorer>>,
    parents: BitSet,
    score_mode: ScoreMode,
    boost: f32,

    doc_done: Cell<bool>,
    init: Cell<bool>,
    current_parent: Cell<DocId>,
    current_score: Cell<f32>,
    child_count: Cell<u32>,
}

impl Query for ToParentBlockJoinQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> Result<Box<dyn Weight>> {
        let child_w = self.child_query.weight(enable_scoring)?;

        Ok(Box::new(ToParentBlockJoinWeight {
            child_weight: child_w,
            parent_bits: Arc::clone(&self.parent_bitset_producer),
            score_mode: self.score_mode,
        }))
    }

    fn explain(&self, searcher: &Searcher, doc_addr: DocAddress) -> Result<Explanation> {
        let seg_reader = searcher.segment_reader(doc_addr.segment_ord);
        let w = self.weight(EnableScoring::enabled_from_searcher(searcher))?;
        w.explain(seg_reader, doc_addr.doc_id)
    }

    fn count(&self, searcher: &Searcher) -> Result<usize> {
        let w = self.weight(EnableScoring::disabled_from_searcher(searcher))?;
        let mut total = 0usize;
        for sr in searcher.segment_readers() {
            let seg_count = w.count(sr)? as usize;

            total += seg_count;
        }

        Ok(total)
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        self.child_query.query_terms(visitor);
    }
}

impl Weight for ToParentBlockJoinWeight {
    fn scorer(&self, reader: &SegmentReader, boost: f32) -> Result<Box<dyn Scorer>> {
        let child_sc = self.child_weight.scorer(reader, boost)?;

        let bitset = self.parent_bits.produce(reader)?;

        if bitset.is_empty() {
            return Ok(Box::new(EmptyScorer));
        }
        let scorer = ToParentBlockJoinScorer {
            child_scorer: RefCell::new(child_sc),
            parents: bitset,
            score_mode: self.score_mode,
            boost,
            doc_done: Cell::new(false),
            init: Cell::new(false),
            current_parent: Cell::new(TERMINATED),
            current_score: Cell::new(0.0),
            child_count: Cell::new(0),
        };

        Ok(Box::new(scorer))
    }

    fn explain(&self, reader: &SegmentReader, doc_id: DocId) -> Result<Explanation> {
        let mut sc = self.scorer(reader, 1.0)?;

        let mut current = sc.advance();

        while current < doc_id && current != TERMINATED {
            current = sc.advance();
        }
        if current != doc_id {
            return Ok(Explanation::new("Not a match", 0.0));
        }
        let val = sc.score();

        let mut ex = Explanation::new_with_string("ToParentBlockJoin aggregator".to_string(), val);
        ex.add_detail(Explanation::new_with_string(
            format!("score_mode={:?}", self.score_mode),
            val,
        ));
        Ok(ex)
    }

    fn count(&self, reader: &SegmentReader) -> Result<u32> {
        let mut sc = self.scorer(reader, 1.0)?;

        let mut count = 0u32;
        let mut doc = sc.advance();

        while doc != TERMINATED {
            count += 1;

            doc = sc.advance();
        }

        Ok(count)
    }

    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> Result<()> {
        let mut scorer = self.scorer(reader, 1.0)?;

        let mut _current_threshold = threshold;
        let mut doc = scorer.advance();

        while doc != TERMINATED {
            let score = scorer.score();

            _current_threshold = callback(doc, score);

            doc = scorer.advance();
        }

        Ok(())
    }
}

impl DocSet for ToParentBlockJoinScorer {
    fn advance(&mut self) -> DocId {
        self.advance_doc()
    }

    fn doc(&self) -> DocId {
        if self.doc_done.get() {
            TERMINATED
        } else if !self.init.get() {
            // We do lazy init here:
            self.advance_doc()
        } else {
            

            self.current_parent.get()
        }
    }

    fn size_hint(&self) -> u32 {
        

        self.parents.len() as u32
    }
}

impl Scorer for ToParentBlockJoinScorer {
    fn score(&mut self) -> Score {
        if self.doc_done.get() || self.current_parent.get() == TERMINATED {
            0.0
        } else {
            let sumval = self.current_score.get();
            let cnt = self.child_count.get();

            let final_score = self.score_mode.finalize_score(sumval, cnt);

            

            final_score * self.boost
        }
    }
}

impl ToParentBlockJoinScorer {
    fn advance_doc(&self) -> DocId {
        if self.doc_done.get() {
            return TERMINATED;
        }

        if !self.init.get() {
            self.init.set(true);
        }

        let next_parent = self.find_next_parent();

        if next_parent == TERMINATED {
            self.doc_done.set(true);
            self.current_parent.set(TERMINATED);
            return TERMINATED;
        }
        self.current_parent.set(next_parent);

        next_parent
    }

    fn find_next_parent(&self) -> DocId {
        let mut child_scorer = self.child_scorer.borrow_mut();

        // Get current child doc
        let mut child_doc = child_scorer.doc();

        // Find the next parent bit
        let parent_doc = self.parents.next_set_bit(child_doc);

        if parent_doc == u32::MAX {
            return TERMINATED;
        }

        // Reset accumulators for this parent
        self.current_score.set(0.0);
        self.child_count.set(0);

        // If we have no children, just return the parent with score 0
        if child_doc == TERMINATED {
            return parent_doc;
        }

        // Process all children up until this parent
        while child_doc != TERMINATED && child_doc < parent_doc {
            let cscore = child_scorer.score();

            // Combine child score
            let new_sum =
                self.score_mode
                    .combine(cscore, self.current_score.get(), self.child_count.get());

            self.current_score.set(new_sum);
            self.child_count
                .set(self.child_count.get().saturating_add(1));

            // Advance child
            child_doc = child_scorer.advance();
        }

        parent_doc
    }
}
