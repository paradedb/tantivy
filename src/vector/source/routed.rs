//! Row gating for routed clusters.

use std::ops::Range;

use common::BitSet;

use crate::fastfield::AliveBitSet;
use crate::vector::scan::layer_read::Selection;
use crate::vector::segment::plan::SegmentFilter;
use crate::DocId;

/// How a cluster row is tested before scoring: the filter's matches
/// intersected with the alive docs.
pub(in crate::vector) enum RowGate<'a> {
    Open,
    AliveOnly(&'a AliveBitSet),
    FilterOnly(&'a BitSet),
    FilterAndAlive {
        filter: &'a BitSet,
        filter_and_alive: BitSet,
    },
}

impl<'a> RowGate<'a> {
    pub(in crate::vector) fn new(
        filter: &'a SegmentFilter,
        alive: Option<&'a AliveBitSet>,
    ) -> Self {
        match (filter.docs(), alive) {
            (None, None) => RowGate::Open,
            (None, Some(alive)) => RowGate::AliveOnly(alive),
            (Some(filter), None) => RowGate::FilterOnly(filter),
            (Some(filter), Some(alive)) => {
                let mut filter_and_alive = filter.clone();
                filter_and_alive.intersect_update(alive.bitset());
                RowGate::FilterAndAlive {
                    filter,
                    filter_and_alive,
                }
            }
        }
    }
}

enum RowVerdict {
    Keep,
    Filtered,
    Dead,
}

pub(in crate::vector) fn select_cluster_rows<'a>(
    doc_ids: &[DocId],
    rows: Range<usize>,
    gate: &RowGate,
    offsets: &'a mut Vec<usize>,
) -> (Selection<'a>, usize, usize, usize) {
    offsets.clear();
    let visited = rows.len();
    let (pruned_filter, pruned_dead) = match gate {
        RowGate::Open => {
            return (Selection::All, visited, 0, 0);
        }
        RowGate::AliveOnly(alive) => select_rows(doc_ids, rows, offsets, |doc| {
            if alive.is_alive(doc) {
                RowVerdict::Keep
            } else {
                RowVerdict::Dead
            }
        }),
        RowGate::FilterOnly(filter) => select_rows(doc_ids, rows, offsets, |doc| {
            if filter.contains(doc) {
                RowVerdict::Keep
            } else {
                RowVerdict::Filtered
            }
        }),
        RowGate::FilterAndAlive {
            filter,
            filter_and_alive,
        } => select_rows(doc_ids, rows, offsets, |doc| {
            if filter_and_alive.contains(doc) {
                RowVerdict::Keep
            } else if !filter.contains(doc) {
                RowVerdict::Filtered
            } else {
                RowVerdict::Dead
            }
        }),
    };
    if offsets.is_empty() {
        (Selection::None, visited, pruned_filter, pruned_dead)
    } else {
        (
            Selection::Rows(offsets),
            visited,
            pruned_filter,
            pruned_dead,
        )
    }
}

/// Returns `(pruned_filter, pruned_dead)`.
#[inline(always)]
fn select_rows(
    doc_ids: &[DocId],
    rows: Range<usize>,
    offsets: &mut Vec<usize>,
    verdict: impl Fn(DocId) -> RowVerdict,
) -> (usize, usize) {
    let mut pruned_filter = 0usize;
    let mut pruned_dead = 0usize;
    for (offset, _row) in rows.enumerate() {
        let doc = doc_ids[offset];
        match verdict(doc) {
            RowVerdict::Keep => {
                offsets.push(offset);
            }
            RowVerdict::Filtered => pruned_filter += 1,
            RowVerdict::Dead => pruned_dead += 1,
        }
    }
    (pruned_filter, pruned_dead)
}
