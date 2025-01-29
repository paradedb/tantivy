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
        println!("Parsing ScoreMode from string: '{}'", mode);
        match mode.trim().to_lowercase().as_str() {
            "none" => {
                println!("Parsed ScoreMode::None");
                Ok(ScoreMode::None)
            }
            "avg" => {
                println!("Parsed ScoreMode::Avg");
                Ok(ScoreMode::Avg)
            }
            "max" => {
                println!("Parsed ScoreMode::Max");
                Ok(ScoreMode::Max)
            }
            "min" => {
                println!("Parsed ScoreMode::Min");
                Ok(ScoreMode::Min)
            }
            "total" => {
                println!("Parsed ScoreMode::Total");
                Ok(ScoreMode::Total)
            }
            other => {
                println!("Failed to parse ScoreMode. Unrecognized mode: '{}'", other);
                Err(TantivyError::InvalidArgument(format!(
                    "Unrecognized nested score_mode: '{}'",
                    other
                )))
            }
        }
    }
}

impl ScoreMode {
    fn combine(&self, child_score: f32, accum: f32, _count: u32) -> f32 {
        println!(
            "Combining scores with ScoreMode::{:?}: child_score = {}, accum = {}, count = {}",
            self, child_score, accum, _count
        );
        let result = match self {
            ScoreMode::None => 0.0,
            ScoreMode::Total => accum + child_score,
            ScoreMode::Avg => accum + child_score,
            ScoreMode::Max => accum.max(child_score),
            ScoreMode::Min => accum.min(child_score),
        };
        println!("Result after combine: {}", result);
        result
    }

    fn finalize_score(&self, sumval: f32, count: u32) -> f32 {
        println!(
            "Finalizing score with ScoreMode::{:?}: sumval = {}, count = {}",
            self, sumval, count
        );
        let final_score = match self {
            ScoreMode::None => 0.0,
            ScoreMode::Total => sumval,
            ScoreMode::Avg => {
                if count == 0 {
                    println!("Count is 0, returning 0.0 for Avg score");
                    0.0
                } else {
                    let avg = sumval / count as f32;
                    println!("Calculated Avg score: {}", avg);
                    avg
                }
            }
            ScoreMode::Max => sumval,
            ScoreMode::Min => sumval,
        };
        println!("Final score: {}", final_score);
        final_score
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
        println!("EmptyScorer::advance called. Returning TERMINATED.");
        TERMINATED
    }
    fn doc(&self) -> DocId {
        println!("EmptyScorer::doc called. Returning TERMINATED.");
        TERMINATED
    }
    fn size_hint(&self) -> u32 {
        println!("EmptyScorer::size_hint called. Returning 0.");
        0
    }
}

impl Scorer for EmptyScorer {
    fn score(&mut self) -> Score {
        println!("EmptyScorer::score called. Returning 0.0.");
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
        println!("Cloning ToParentBlockJoinQuery.");
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
        println!(
            "Creating new ToParentBlockJoinQuery with score_mode: {:?}",
            score_mode
        );
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
        println!("ToParentBlockJoinQuery::weight called.");
        let child_w = self.child_query.weight(enable_scoring)?;
        println!("Obtained child_weight.");
        Ok(Box::new(ToParentBlockJoinWeight {
            child_weight: child_w,
            parent_bits: Arc::clone(&self.parent_bitset_producer),
            score_mode: self.score_mode,
        }))
    }

    fn explain(&self, searcher: &Searcher, doc_addr: DocAddress) -> Result<Explanation> {
        println!(
            "ToParentBlockJoinQuery::explain called for DocAddress: {:?}",
            doc_addr
        );
        let seg_reader = searcher.segment_reader(doc_addr.segment_ord);
        let w = self.weight(EnableScoring::enabled_from_searcher(searcher))?;
        w.explain(seg_reader, doc_addr.doc_id)
    }

    fn count(&self, searcher: &Searcher) -> Result<usize> {
        println!("ToParentBlockJoinQuery::count called.");
        let w = self.weight(EnableScoring::disabled_from_searcher(searcher))?;
        let mut total = 0usize;
        for sr in searcher.segment_readers() {
            println!("Counting documents in segment: {:?}", sr);
            let seg_count = w.count(sr)? as usize;
            println!("Segment count: {}", seg_count);
            total += seg_count;
        }
        println!("Total count: {}", total);
        Ok(total)
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        println!("ToParentBlockJoinQuery::query_terms called.");
        self.child_query.query_terms(visitor);
    }
}

impl Weight for ToParentBlockJoinWeight {
    fn scorer(&self, reader: &SegmentReader, boost: f32) -> Result<Box<dyn Scorer>> {
        println!(
            "ToParentBlockJoinWeight::scorer called with boost: {}",
            boost
        );
        let child_sc = self.child_weight.scorer(reader, boost)?;
        println!("Obtained child_scorer.");
        let bitset = self.parent_bits.produce(reader)?;
        println!("Produced parent_bitset with {} bits set.", bitset.len());
        if bitset.is_empty() {
            println!("Parent bitset is empty. Returning EmptyScorer.");
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
        println!("Created ToParentBlockJoinScorer.");
        Ok(Box::new(scorer))
    }

    fn explain(&self, reader: &SegmentReader, doc_id: DocId) -> Result<Explanation> {
        println!(
            "ToParentBlockJoinWeight::explain called for doc_id: {}",
            doc_id
        );
        let mut sc = self.scorer(reader, 1.0)?;
        println!("Obtained scorer for explanation.");
        let mut current = sc.advance();
        println!("Advanced scorer to first document: {}", current);
        while current < doc_id && current != TERMINATED {
            println!(
                "Current doc {} is less than target doc_id {}. Advancing.",
                current, doc_id
            );
            current = sc.advance();
            println!("Advanced to document: {}", current);
        }
        if current != doc_id {
            println!(
                "Document {} not found. Returning Explanation with score 0.0.",
                doc_id
            );
            return Ok(Explanation::new("Not a match", 0.0));
        }
        let val = sc.score();
        println!(
            "Document {} found. Score: {}. Creating Explanation.",
            doc_id, val
        );
        let mut ex = Explanation::new_with_string("ToParentBlockJoin aggregator".to_string(), val);
        ex.add_detail(Explanation::new_with_string(
            format!("score_mode={:?}", self.score_mode),
            val,
        ));
        Ok(ex)
    }

    fn count(&self, reader: &SegmentReader) -> Result<u32> {
        println!("ToParentBlockJoinWeight::count called.");
        let mut sc = self.scorer(reader, 1.0)?;
        println!("Obtained scorer for count.");
        let mut count = 0u32;
        let mut doc = sc.advance();
        println!("Starting document iteration for count. First doc: {}", doc);
        while doc != TERMINATED {
            count += 1;
            println!("Counting doc: {}. Current count: {}", doc, count);
            doc = sc.advance();
            println!("Advanced to next doc: {}", doc);
        }
        println!("Final count: {}", count);
        Ok(count)
    }

    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> Result<()> {
        println!(
            "ToParentBlockJoinWeight::for_each_pruning called with threshold: {}",
            threshold
        );
        let mut scorer = self.scorer(reader, 1.0)?;
        println!("Obtained scorer for pruning.");
        let mut _current_threshold = threshold;
        let mut doc = scorer.advance();
        println!("Starting pruning iteration. First doc: {}", doc);
        while doc != TERMINATED {
            let score = scorer.score();
            println!("Doc: {}, Score: {}", doc, score);
            _current_threshold = callback(doc, score);
            println!(
                "Callback applied. Updated threshold: {}",
                _current_threshold
            );
            doc = scorer.advance();
            println!("Advanced to next doc: {}", doc);
        }
        println!("Completed pruning iteration.");
        Ok(())
    }
}

impl DocSet for ToParentBlockJoinScorer {
    fn advance(&mut self) -> DocId {
        println!("ToParentBlockJoinScorer::advance called.");
        self.advance_doc()
    }

    fn doc(&self) -> DocId {
        if self.doc_done.get() {
            println!(
                "ToParentBlockJoinScorer::doc called. doc_done is true. Returning TERMINATED."
            );
            TERMINATED
        } else if !self.init.get() {
            println!("ToParentBlockJoinScorer::doc called. Initializing.");
            // We do lazy init here:
            self.advance_doc()
        } else {
            let current = self.current_parent.get();
            println!(
                "ToParentBlockJoinScorer::doc called. Current parent: {}",
                current
            );
            current
        }
    }

    fn size_hint(&self) -> u32 {
        let size = self.parents.len() as u32;
        println!("ToParentBlockJoinScorer::size_hint called. Size: {}", size);
        size
    }
}

impl Scorer for ToParentBlockJoinScorer {
    fn score(&mut self) -> Score {
        if self.doc_done.get() || self.current_parent.get() == TERMINATED {
            println!(
                "ToParentBlockJoinScorer::score called. doc_done: {}, current_parent: TERMINATED. Returning 0.0.",
                self.doc_done.get()
            );
            0.0
        } else {
            let sumval = self.current_score.get();
            let cnt = self.child_count.get();
            println!(
                "ToParentBlockJoinScorer::score called. sumval: {}, count: {}",
                sumval, cnt
            );
            let final_score = self.score_mode.finalize_score(sumval, cnt);
            println!("Final score after finalize_score: {}", final_score);
            let boosted_score = final_score * self.boost;
            println!(
                "Boosted score ({} * {}): {}",
                final_score, self.boost, boosted_score
            );
            boosted_score
        }
    }
}

impl ToParentBlockJoinScorer {
    fn advance_doc(&self) -> DocId {
        if self.doc_done.get() {
            println!("ToParentBlockJoinScorer::advance_doc called. doc_done is true. Returning TERMINATED.");
            return TERMINATED;
        }

        if !self.init.get() {
            println!("ToParentBlockJoinScorer::advance_doc called. Initializing.");
            self.init.set(true);
        }

        let next_parent = self.find_next_parent();
        println!("Next parent found: {}", next_parent);
        if next_parent == TERMINATED {
            println!("No more parents. Setting doc_done to true.");
            self.doc_done.set(true);
            self.current_parent.set(TERMINATED);
            return TERMINATED;
        }
        self.current_parent.set(next_parent);
        println!("Set current_parent to {}", next_parent);
        next_parent
    }

    fn find_next_parent(&self) -> DocId {
        println!("ToParentBlockJoinScorer::find_next_parent called.");
        let mut child_scorer = self.child_scorer.borrow_mut();

        // Get current child doc
        let mut child_doc = child_scorer.doc();
        println!("Current child_doc: {}", child_doc);

        // Find the next parent bit
        let parent_doc = self.parents.next_set_bit(child_doc);
        println!(
            "Next set parent bit after child_doc {}: {}",
            child_doc, parent_doc
        );
        if parent_doc == u32::MAX {
            println!("No next parent bit found. Returning TERMINATED.");
            return TERMINATED;
        }

        // Reset accumulators for this parent
        self.current_score.set(0.0);
        self.child_count.set(0);
        println!(
            "Reset accumulators for parent_doc {}: current_score = 0.0, child_count = 0",
            parent_doc
        );

        // If we have no children, just return the parent with score 0
        if child_doc == TERMINATED {
            println!(
                "No children for parent_doc {}. Returning parent_doc with score 0.",
                parent_doc
            );
            return parent_doc;
        }

        // Process all children up until this parent
        while child_doc != TERMINATED && child_doc < parent_doc {
            let cscore = child_scorer.score();
            println!(
                "Processing child_doc {} with score {} for parent_doc {}",
                child_doc, cscore, parent_doc
            );

            // Combine child score
            let new_sum =
                self.score_mode
                    .combine(cscore, self.current_score.get(), self.child_count.get());
            println!("New sum after combining: {}", new_sum);
            self.current_score.set(new_sum);
            self.child_count
                .set(self.child_count.get().saturating_add(1));
            println!("Updated child_count: {}", self.child_count.get());

            // Advance child
            child_doc = child_scorer.advance();
            println!("Advanced child_scorer to doc: {}", child_doc);
        }

        // Skip if child is exactly at parent
        if child_doc == parent_doc {
            println!(
                "Child_doc {} is exactly at parent_doc {}. Advancing child_scorer.",
                child_doc, parent_doc
            );
            child_doc = child_scorer.advance();
            println!("Advanced child_scorer to doc: {}", child_doc);
        }

        parent_doc
    }
}
