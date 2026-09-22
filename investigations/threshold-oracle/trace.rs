use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use crate::{Score, TERMINATED};

thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
    static BOUND: RefCell<bool> = const { RefCell::new(false) };
}

#[derive(Default)]
struct Trace {
    disable_tf: bool,
    segment: usize,
    threshold: Score,
    candidates: u64,
    tf_rejected: u64,
    norm_reads: [u64; 2],
    pages: [BTreeSet<(usize, usize)>; 2],
    norm_docs: [BTreeSet<(usize, u32)>; 2],
    decoded_full: BTreeSet<(usize, u32)>,
    decoded_tails: BTreeSet<usize>,
    skipped_full: BTreeSet<(usize, u32)>,
    visited_full: BTreeSet<(usize, u32)>,
    heap_updates: usize,
    timeline: Vec<Value>,
    bounds: BTreeMap<(usize, u32), Score>,
}

impl Trace {
    fn snapshot(&mut self, event: &str) {
        self.timeline.push(json!({
            "event": event,
            "segment": self.segment,
            "threshold": (self.threshold >= 0.0).then_some(self.threshold),
            "candidates": self.candidates,
            "tf_rejected": self.tf_rejected,
            "norm_reads": self.norm_reads.iter().sum::<u64>(),
            "tail_bound_reads": self.norm_reads[1],
            "norm_pages": self.pages[0].union(&self.pages[1]).count(),
        }));
    }
}

pub(crate) fn start(disable_tf: bool, threshold: Score) {
    TRACE.with(|cell| {
        assert!(cell.borrow().is_none());
        *cell.borrow_mut() = Some(Trace {
            disable_tf,
            threshold,
            ..Trace::default()
        });
    });
}

pub(crate) fn segment(segment: usize, threshold: Score) {
    TRACE.with_borrow_mut(|state| {
        if let Some(trace) = state {
            trace.segment = segment;
            trace.threshold = threshold;
            trace.snapshot("segment_start");
        }
    });
}

pub(crate) fn disable_tf_filter() -> bool {
    TRACE.with_borrow(|state| state.as_ref().is_some_and(|trace| trace.disable_tf))
}

pub(crate) fn set_bounds(bounds: BTreeMap<(usize, u32), Score>) {
    TRACE.with_borrow_mut(|state| state.as_mut().unwrap().bounds = bounds);
}

pub(crate) fn block_bound(end: u32) -> Option<Score> {
    TRACE.with_borrow(|state| {
        let trace = state.as_ref()?;
        trace.bounds.get(&(trace.segment, end)).copied()
    })
}

pub(crate) struct BoundPhase(bool);

pub(crate) fn bound_phase() -> BoundPhase {
    BoundPhase(BOUND.replace(true))
}

impl Drop for BoundPhase {
    fn drop(&mut self) {
        BOUND.set(self.0);
    }
}

pub(crate) fn norm_read(segment: usize, doc: u32, page: usize) {
    TRACE.with_borrow_mut(|state| {
        if let Some(trace) = state {
            let phase = usize::from(BOUND.with_borrow(|bound| *bound));
            trace.norm_reads[phase] += 1;
            trace.pages[phase].insert((segment, page));
            trace.norm_docs[phase].insert((segment, doc));
        }
    });
}

pub(crate) fn decoded(end: u32, len: usize) {
    TRACE.with_borrow_mut(|state| {
        if let Some(trace) = state {
            if len == 128 {
                trace.decoded_full.insert((trace.segment, end));
            } else if len > 0 {
                trace.decoded_tails.insert(trace.segment);
            }
        }
    });
}

pub(crate) fn skipped(end: u32) {
    TRACE.with_borrow_mut(|state| {
        if let Some(trace) = state {
            if end != TERMINATED {
                trace.skipped_full.insert((trace.segment, end));
            }
        }
    });
}

pub(crate) fn candidate(_doc: u32, end: u32, threshold: Score, scores: bool) {
    TRACE.with_borrow_mut(|state| {
        if let Some(trace) = state {
            trace.candidates += 1;
            trace.tf_rejected += u64::from(!scores);
            trace.threshold = threshold;
            if end != TERMINATED {
                trace.visited_full.insert((trace.segment, end));
            }
            if trace.candidates % 128 == 0 {
                trace.snapshot("sample");
            }
        }
    });
}

pub(crate) fn heap_update(threshold: Score) {
    TRACE.with_borrow_mut(|state| {
        if let Some(trace) = state {
            trace.heap_updates += 1;
            trace.threshold = threshold;
            trace.snapshot("heap_update");
        }
    });
}

pub(crate) fn finish() -> Value {
    TRACE.with(|cell| {
        let mut trace = cell.take().unwrap();
        trace.snapshot("end");
        assert_eq!(trace.candidates - trace.tf_rejected, trace.norm_reads[0]);
        json!({
            "candidates": trace.candidates,
            "tf_rejected": trace.tf_rejected,
            "norm_reads": trace.norm_reads.iter().sum::<u64>(),
            "scoring_norm_reads": trace.norm_reads[0],
            "tail_bound_norm_reads": trace.norm_reads[1],
            "norm_pages": trace.pages[0].union(&trace.pages[1]).count(),
            "scoring_norm_pages": trace.pages[0].len(),
            "tail_bound_norm_pages": trace.pages[1].len(),
            "tail_only_norm_pages": trace.pages[1].difference(&trace.pages[0]).count(),
            "unique_scored_docs": trace.norm_docs[0].len(),
            "unique_tail_bound_docs": trace.norm_docs[1].len(),
            "full_blocks_decoded": trace.decoded_full.len(),
            "partial_blocks_decoded": trace.decoded_tails.len(),
            "full_blocks_visited": trace.visited_full.len(),
            "full_blocks_skipped_without_candidates": trace.skipped_full.difference(&trace.visited_full).count(),
            "full_blocks_pruned_after_candidates": trace.skipped_full.intersection(&trace.visited_full).count(),
            "heap_updates": trace.heap_updates,
            "timeline": trace.timeline,
        })
    })
}
