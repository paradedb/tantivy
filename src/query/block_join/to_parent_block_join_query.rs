use std::cell::{Cell, RefCell};
use std::fmt;
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

impl ScoreMode {
    /// Convert the user input like "none", "min", etc. into `ScoreMode`.
    pub fn from_str(mode: &str) -> crate::Result<ScoreMode> {
        println!("ScoreMode::from_str: Parsing mode '{}'", mode);
        match mode.to_lowercase().as_str() {
            "none" => Ok(ScoreMode::None),
            "avg" => Ok(ScoreMode::Avg),
            "max" => Ok(ScoreMode::Max),
            "min" => Ok(ScoreMode::Min),
            "total" => Ok(ScoreMode::Total),
            other => Err(TantivyError::InvalidArgument(format!(
                "Unrecognized nested score_mode: {}",
                other
            ))),
        }
    }

    fn combine(&self, child_score: f32, accum: f32, count: u32) -> f32 {
        let result = match self {
            ScoreMode::None => 0.0,
            ScoreMode::Total => accum + child_score,
            ScoreMode::Avg => accum + child_score,
            ScoreMode::Max => accum.max(child_score),
            ScoreMode::Min => accum.min(child_score),
        };
        println!(
            "[ScoreMode::combine] child_score={}, accum={}, count={}, new_accum={}",
            child_score, accum, count, result
        );
        result
    }

    fn finalize_score(&self, sumval: f32, count: u32) -> f32 {
        let final_val = match self {
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
        };
        println!(
            "[ScoreMode::finalize_score] mode={:?}, sumval={}, count={}, final_val={}",
            self, sumval, count, final_val
        );
        final_val
    }
}

///////////////////////////////////////////////////////////////////////////////
// ParentBitSetProducer
///////////////////////////////////////////////////////////////////////////////

pub trait ParentBitSetProducer: Send + Sync + 'static {
    fn produce(&self, reader: &SegmentReader) -> Result<BitSet>;
}

///////////////////////////////////////////////////////////////////////////////
// EmptyScorer
///////////////////////////////////////////////////////////////////////////////

struct EmptyScorer;

impl DocSet for EmptyScorer {
    fn advance(&mut self) -> DocId {
        println!("[EmptyScorer::advance] => TERMINATED");
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

///////////////////////////////////////////////////////////////////////////////
// ToParentBlockJoinQuery
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
        println!("[ToParentBlockJoinQuery::weight] Building child weight...");
        let child_w = self.child_query.weight(enable_scoring)?;
        println!(
            "[ToParentBlockJoinQuery::weight] Child weight built. score_mode={:?}",
            self.score_mode
        );
        Ok(Box::new(ToParentBlockJoinWeight {
            child_weight: child_w,
            parent_bits: Arc::clone(&self.parent_bitset_producer),
            score_mode: self.score_mode,
        }))
    }

    fn explain(&self, searcher: &Searcher, doc_addr: DocAddress) -> Result<Explanation> {
        println!(
            "[ToParentBlockJoinQuery::explain] doc_addr={:?}, score_mode={:?}",
            doc_addr, self.score_mode
        );
        let seg_reader = searcher.segment_reader(doc_addr.segment_ord);
        let w = self.weight(EnableScoring::enabled_from_searcher(searcher))?;
        w.explain(seg_reader, doc_addr.doc_id)
    }

    fn count(&self, searcher: &Searcher) -> Result<usize> {
        println!("[ToParentBlockJoinQuery::count] Starting count...");
        let w = self.weight(EnableScoring::disabled_from_searcher(searcher))?;
        let mut total = 0usize;
        for sr in searcher.segment_readers() {
            let seg_count = w.count(sr)? as usize;
            println!(
                "  [ToParentBlockJoinQuery::count] segment_count={} => accumulate",
                seg_count
            );
            total += seg_count;
        }
        println!("[ToParentBlockJoinQuery::count] done => total={}", total);
        Ok(total)
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        println!("[ToParentBlockJoinQuery::query_terms] visiting child_query");
        self.child_query.query_terms(visitor);
    }
}

impl Weight for ToParentBlockJoinWeight {
    fn scorer(&self, reader: &SegmentReader, boost: f32) -> Result<Box<dyn Scorer>> {
        println!(
            "[ToParentBlockJoinWeight::scorer] start => segment={}, boost={}",
            reader.segment_id(),
            boost
        );
        let child_sc = self.child_weight.scorer(reader, boost)?;
        println!("[ToParentBlockJoinWeight::scorer] child_scorer built, now produce parent_bits");
        let bitset = self.parent_bits.produce(reader)?;
        println!(
            "[ToParentBlockJoinWeight::scorer] parent_bitset len={} for segment (max_doc={})",
            bitset.len(),
            reader.max_doc()
        );
        if bitset.is_empty() {
            println!("[ToParentBlockJoinWeight::scorer] bitset is empty => return EmptyScorer");
            return Ok(Box::new(EmptyScorer));
        }
        println!(
            "[ToParentBlockJoinWeight::scorer] Building ToParentBlockJoinScorer with score_mode={:?}",
            self.score_mode
        );
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
        println!(
            "[ToParentBlockJoinWeight::explain] doc_id={}, segment={}",
            doc_id,
            reader.segment_id()
        );
        let mut sc = self.scorer(reader, 1.0)?;
        let mut current = sc.advance();
        while current < doc_id && current != TERMINATED {
            current = sc.advance();
        }
        if current != doc_id {
            println!("  => Not a match => Explanation(0.0)");
            return Ok(Explanation::new("Not a match", 0.0));
        }
        let val = sc.score();
        println!(
            "  => matched doc_id={} => final parent score={}",
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
        println!(
            "[ToParentBlockJoinWeight::count] => building scorer for segment={}, then doc loop",
            reader.segment_id()
        );
        let mut sc = self.scorer(reader, 1.0)?;
        let mut count = 0u32;
        let mut doc = sc.advance();
        while doc != TERMINATED {
            count += 1;
            doc = sc.advance();
        }
        println!(
            "[ToParentBlockJoinWeight::count] segment={} => counted {} parents",
            reader.segment_id(),
            count
        );
        Ok(count)
    }

    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> Result<()> {
        println!(
            "[ToParentBlockJoinWeight::for_each_pruning] => segment={}, threshold={}",
            reader.segment_id(),
            threshold
        );
        let mut scorer = self.scorer(reader, 1.0)?;
        let mut current_threshold = threshold;
        let mut doc = scorer.advance();
        while doc != TERMINATED {
            let score = scorer.score();
            println!(
                "  => doc={}, parent_score={}, threshold(before)={}",
                doc, score, current_threshold
            );
            current_threshold = callback(doc, score);
            doc = scorer.advance();
        }
        println!("  => done for segment => Ok(())");
        Ok(())
    }
}

impl DocSet for ToParentBlockJoinScorer {
    fn advance(&mut self) -> DocId {
        let doc_id = self.advance_doc();
        println!(
            "[ToParentBlockJoinScorer::advance] => next parent doc={}",
            doc_id
        );
        doc_id
    }

    fn doc(&self) -> DocId {
        let d = if self.doc_done.get() {
            TERMINATED
        } else if !self.init.get() {
            // We do lazy init here:
            let doc_id = self.advance_doc();
            println!(
                "[ToParentBlockJoinScorer::doc lazy-init] advanced => doc_id={}",
                doc_id
            );
            doc_id
        } else {
            self.current_parent.get()
        };
        println!("[ToParentBlockJoinScorer::doc] => {}", d);
        d
    }

    fn size_hint(&self) -> u32 {
        self.parents.len() as u32
    }
}

impl Scorer for ToParentBlockJoinScorer {
    fn score(&mut self) -> Score {
        if self.doc_done.get() || self.current_parent.get() == TERMINATED {
            println!("[ToParentBlockJoinScorer::score] doc_done or TERMINATED => 0.0");
            0.0
        } else {
            let sum_val = self.current_score.get();
            let cnt = self.child_count.get();
            let final_score = self.score_mode.finalize_score(sum_val, cnt);
            let final_boosted = final_score * self.boost;
            println!(
                "[ToParentBlockJoinScorer::score] sum_val={}, count={}, mode=>score={}, boost={}, final_boosted={}",
                sum_val, cnt, final_score, self.boost, final_boosted
            );
            final_boosted
        }
    }
}

impl ToParentBlockJoinScorer {
    fn advance_doc(&self) -> DocId {
        println!(
            "[ToParentBlockJoinScorer::advance_doc] doc_done={}, init={}, current_parent={}",
            self.doc_done.get(),
            self.init.get(),
            self.current_parent.get()
        );
        if self.doc_done.get() {
            println!("  => doc_done => return TERMINATED");
            return TERMINATED;
        }

        if !self.init.get() {
            self.init.set(true);
            println!("  => first-time init done, not bailing out yet...");
        }

        let next_parent = self.find_next_parent();
        println!(
            "  => find_next_parent => returns parent_doc={}",
            next_parent
        );
        if next_parent == TERMINATED {
            println!("  => no more parents => doc_done=true => return TERMINATED");
            self.doc_done.set(true);
            self.current_parent.set(TERMINATED);
            return TERMINATED;
        }
        self.current_parent.set(next_parent);
        next_parent
    }

    fn find_next_parent(&self) -> DocId {
        println!("[ToParentBlockJoinScorer::find_next_parent] ENTER");
        let mut child_scorer = self.child_scorer.borrow_mut();

        // Get current child doc
        let mut child_doc = child_scorer.doc();
        println!("  => child_doc={}", child_doc);

        // Find the next parent bit
        let parent_doc = self.parents.next_set_bit(child_doc);
        println!(
            "  => next_set_bit({}) => parent_doc={}",
            child_doc, parent_doc
        );

        if parent_doc == u32::MAX {
            println!("  => no more parent bits => return TERMINATED");
            return TERMINATED;
        }

        // Reset accumulators for this parent
        self.current_score.set(0.0);
        self.child_count.set(0);

        // If we have no children, just return the parent with score 0
        if child_doc == TERMINATED {
            println!("  => no children => return parent_doc with score 0");
            return parent_doc;
        }

        // Process all children up until this parent
        while child_doc != TERMINATED && child_doc < parent_doc {
            let cscore = child_scorer.score();
            println!(
                "    => child_doc={}, child_score={}, accum_so_far={}, child_count={}",
                child_doc,
                cscore,
                self.current_score.get(),
                self.child_count.get()
            );

            // Combine child score
            let new_sum =
                self.score_mode
                    .combine(cscore, self.current_score.get(), self.child_count.get());
            self.current_score.set(new_sum);
            self.child_count
                .set(self.child_count.get().saturating_add(1));

            // Advance child
            child_doc = child_scorer.advance();
            println!("    => advanced child => {}", child_doc);
        }

        // Skip if child is exactly at parent
        if child_doc == parent_doc {
            child_scorer.advance();
        }

        println!("  => returning parent_doc={}", parent_doc);
        parent_doc
    }
}
