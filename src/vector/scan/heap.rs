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

/// A candidate's composite heap key, or `None` when its similarity alone
/// cannot beat the heap threshold, so the tie-break column is read only for
/// competitive candidates. The caller pushes the key itself, with whichever of
/// [`TopNComputer::push`]/`push_unordered` its doc arrival order permits.
///
/// The skip is exact rather than approximate: `(s, t) < (ts, tt)` requires
/// either `s < ts`, or `s == ts` with `t < tt`. So a candidate rejected here on
/// similarity alone could never have survived the full composite comparison,
/// and the tie-break lookup it would have cost is pure waste. Once the heap has
/// filled this is the common case.
/// Builds a competitive candidate's composite heap key.
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
