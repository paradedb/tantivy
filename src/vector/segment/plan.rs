//! Collecting a segment filter for the IVF scan.

use common::BitSet;

use crate::docset::COLLECT_BLOCK_BUFFER_LEN;
use crate::query::{for_each_docset_buffered, AllScorer, Weight};
use crate::{DocId, SegmentReader};

/// A segment's filter matches, as consumed by the IVF probes.
pub(in crate::vector) enum SegmentFilter {
    /// Every doc id below `max_doc` matches (deleted docs included); no
    /// bitset is materialized.
    All,
    /// The matching doc ids.
    Docs(BitSet),
}

impl SegmentFilter {
    pub(in crate::vector) fn is_empty(&self) -> bool {
        matches!(self, SegmentFilter::Docs(filter) if filter.len() == 0)
    }

    pub(in crate::vector) fn docs(&self) -> Option<&BitSet> {
        match self {
            SegmentFilter::All => None,
            SegmentFilter::Docs(filter) => Some(filter),
        }
    }

    /// The share of doc ids below `max_doc` that match.
    pub(super) fn match_fraction(&self, max_doc: DocId) -> f64 {
        match self {
            SegmentFilter::All => 1.0,
            SegmentFilter::Docs(filter) => filter.len() as f64 / f64::from(max_doc.max(1)),
        }
    }
}

/// Drain the filter `DocSet` into a dense BitSet for O(1) random membership
/// testing per cluster doc. The BitSet allocates `max_doc / 8` bytes regardless
/// of filter selectivity — inherent to IVF needing membership tests on
/// out-of-order doc ids. A filter whose scorer is an [`AllScorer`] (match-all,
/// including boolean queries that collapse to one) skips the drain entirely,
/// as does a drained bitset that turns out full. `#[inline(never)]` so it
/// forms its own flamegraph frame; at low selectivity over a large segment
/// this drain is real cost otherwise hidden in the search entry.
/// Materializes a filter doc set as a dense bitset unless it matches all docs.
#[inline(never)]
pub(in crate::vector) fn build_segment_filter(
    weight: &dyn Weight,
    segment_reader: &SegmentReader,
    max_doc: DocId,
) -> crate::Result<SegmentFilter> {
    let mut scorer = weight.scorer(segment_reader, 1.0)?;
    if scorer.is::<AllScorer>() {
        return Ok(SegmentFilter::All);
    }
    let mut filter = BitSet::with_max_value(max_doc);
    let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];
    for_each_docset_buffered(scorer.as_mut(), &mut buffer, |docs| {
        for &doc in docs {
            filter.insert(doc);
        }
    });
    if filter.len() == max_doc as usize {
        return Ok(SegmentFilter::All);
    }
    Ok(SegmentFilter::Docs(filter))
}
