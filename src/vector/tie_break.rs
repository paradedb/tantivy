//! The default, empty tie-break for the vector top-N heap.
//!
//! [`TopDocsByVectorSimilarity`](crate::vector::TopDocsByVectorSimilarity) sorts by a composite
//! `(similarity, tie_break)` key so a secondary `ORDER BY` column takes part in heap eviction.
//! Without one, the tail is this zero-sized key: `(Score, ())` has the layout of `Score` and
//! compares by similarity alone.

use std::cmp::Ordering;

use crate::collector::sort_key::NaturalComparator;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer};
use crate::{DocId, Score, SegmentOrdinal, SegmentReader};

/// A [`SortKeyComputer`] that contributes nothing to the ordering: a vector search without a
/// tie-break runs the same composite path with this key.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTieBreak;

impl SortKeyComputer for NoTieBreak {
    type SortKey = ();
    type Child = NoTieBreak;
    type Comparator = NaturalComparator;

    fn segment_sort_key_computer(&self, _segment_reader: &SegmentReader) -> crate::Result<Self> {
        Ok(NoTieBreak)
    }
}

impl SegmentSortKeyComputer for NoTieBreak {
    type SortKey = ();
    type SegmentSortKey = ();
    type SegmentComparator = NaturalComparator;

    #[inline(always)]
    fn segment_sort_key(&mut self, _doc: DocId, _score: Score) {}

    #[inline(always)]
    fn compare_segment_sort_key(&self, _left: &(), _right: &()) -> Ordering {
        Ordering::Equal
    }

    fn convert_segment_sort_key(&self, _sort_key: ()) {}

    fn supports_bm25_pruning(&self) -> bool {
        false
    }

    fn bm25_pruning_threshold(
        &self,
        _threshold: &(),
        _segment_ord: SegmentOrdinal,
        _threshold_ord: SegmentOrdinal,
    ) -> Option<Score> {
        None
    }
}
