//! Tie-broken top-k hits and their ordering key.

use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, TopNComputer};
use crate::{DocAddress, DocId, Score};

/// The settled result.
pub(in crate::vector) type TieBreakHits<K> = Vec<(
    (Score, <K as SegmentSortKeyComputer>::SegmentSortKey),
    DocAddress,
)>;

/// The in-flight accumulator.
pub(in crate::vector) type TieBreakHeap<K, CTail> = TopNComputer<
    (Score, <K as SegmentSortKeyComputer>::SegmentSortKey),
    DocId,
    (NaturalComparator, CTail),
>;

/// A candidate's composite heap key, or `None` when its similarity alone loses to the threshold.
///
/// The skip is exact: `(s, t) < (ts, tt)` needs `s < ts` or `s == ts`, so a rejected candidate
/// could never survive the full comparison. The caller pushes the key.
#[inline(always)]
pub(in crate::vector) fn tie_break_key<K, CTail>(
    topn: &TieBreakHeap<K, CTail>,
    tie_break: &mut K,
    score: Score,
    doc: DocId,
) -> Option<(Score, K::SegmentSortKey)>
where
    K: SegmentSortKeyComputer,
    CTail: Comparator<K::SegmentSortKey>,
{
    if let Some(((threshold_score, _), _)) = &topn.threshold {
        if score < *threshold_score {
            return None;
        }
    }
    Some((score, tie_break.segment_sort_key(doc, score)))
}
