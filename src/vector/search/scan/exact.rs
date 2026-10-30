//! Exact scoring: full-precision reads of a cluster's selected rows, scored into the pruning bound
//! and the result heap.

use common::OwnedBytes;

use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, TopNComputer};
use crate::vector::search::cluster_source::ClusterRows;
use crate::vector::search::prepared::VectorQuery;
use crate::vector::search::scan::bound::{Candidate, PruningBound};
use crate::vector::search::scan::heap::{tie_break_key, TieBreakHeap, TieBreakHits};
use crate::vector::search::scan::histogram::Fold;
use crate::vector::storage::quantized::ReadScratch;
use crate::vector::storage::reader::{VectorIndexReader, VectorRowBatch};
use crate::vector::telemetry::enter_vector_stage;
use crate::vector::write::RowAddress;
use crate::vector::{VectorElement, VectorStage};
use crate::{DocAddress, DocId, SegmentOrdinal};

/// Full-precision rows of one cluster's selection.
enum ExactRows<'s> {
    /// The cluster's whole full-precision span, in one request.
    Span(OwnedBytes),
    /// The selected rows, through a storage-aware read plan into the scan's scratch.
    Planned(VectorRowBatch<'s>),
}

/// Scores rows exactly into the pruning bound and the result heap; the only way rows reach the
/// heap.
pub(super) struct ExactScorer<'a, T, K, C>
where
    T: VectorElement,
    K: SegmentSortKeyComputer,
    C: Comparator<K::SegmentSortKey>,
{
    reader: &'a VectorIndexReader,
    query: &'a VectorQuery<T>,
    topn: TieBreakHeap<K, C>,
    tie_break: &'a mut K,
    segment_ord: SegmentOrdinal,
    /// A cluster's DocIds column, read when its selected rows carry no doc ids.
    cluster_docs: Vec<DocId>,
    /// Global rows scored in the exact phase, in order; the scan copies them into its trace.
    #[cfg(test)]
    pub(super) exact_phase_rows: Vec<usize>,
    /// Every score computed, in order; the scan copies them into its trace.
    #[cfg(test)]
    pub(super) exact_scores: Vec<f32>,
}

impl<'a, T, K, C> ExactScorer<'a, T, K, C>
where
    T: VectorElement,
    K: SegmentSortKeyComputer,
    C: Comparator<K::SegmentSortKey>,
{
    pub(super) fn new(
        reader: &'a VectorIndexReader,
        query: &'a VectorQuery<T>,
        top_n: usize,
        segment_ord: SegmentOrdinal,
        tie_break: &'a mut K,
        tie_comparator: C,
    ) -> Self {
        Self {
            reader,
            query,
            topn: TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator)),
            tie_break,
            segment_ord,
            cluster_docs: Vec::new(),
            #[cfg(test)]
            exact_phase_rows: Vec::new(),
            #[cfg(test)]
            exact_scores: Vec::new(),
        }
    }

    // --- Scoring ---

    /// Scores every row `selected` selects into `bound` and the heap, reading through `scratch`;
    /// returns the rows scored. A row without a doc id reads the cluster's DocIds column when it
    /// first needs one. Each exact is `offer`ed to the bound (which CASes the shared slot).
    pub(super) fn score(
        &mut self,
        selected: &ClusterRows<'_>,
        scratch: &mut ReadScratch,
        bound: &mut PruningBound,
        mut fold: Option<Fold<'_>>,
    ) -> crate::Result<usize> {
        self.score_rows(selected, scratch, bound, &mut fold)
    }

    /// Scores the exact phase's survivors in `selected`, as [`ExactScorer::score`] does.
    pub(super) fn score_survivors(
        &mut self,
        selected: &ClusterRows<'_>,
        scratch: &mut ReadScratch,
        bound: &mut PruningBound,
    ) -> crate::Result<usize> {
        #[cfg(test)]
        self.exact_phase_rows.extend(selected.global_rows());
        self.score(selected, scratch, bound, None)
    }

    /// Reads `selected`'s rows in `ExactFetch` and scores them in `ExactScore`.
    fn score_rows(
        &mut self,
        selected: &ClusterRows<'_>,
        scratch: &mut ReadScratch,
        bound: &mut PruningBound,
        fold: &mut Option<Fold<'_>>,
    ) -> crate::Result<usize> {
        let mut column_read = false;
        let rows = {
            let _fetch = enter_vector_stage(VectorStage::ExactFetch);
            self.read(selected, scratch)?
        };
        let _score = enter_vector_stage(VectorStage::ExactScore);

        let scored = match &rows {
            ExactRows::Span(bytes) => {
                let stride = self.reader.options().bytes_per_vector();
                for (local, row_bytes) in bytes.chunks_exact(stride).enumerate() {
                    self.offer(
                        selected,
                        bound,
                        fold,
                        &mut column_read,
                        local,
                        local,
                        selected.rows.start + local,
                        row_bytes,
                    )?;
                }
                bytes.len() / stride
            }
            ExactRows::Planned(planned) => {
                let mut scored = 0;
                for (position, (row, row_bytes)) in planned.iter().enumerate() {
                    self.offer(
                        selected,
                        bound,
                        fold,
                        &mut column_read,
                        position,
                        row - selected.rows.start,
                        row,
                        row_bytes,
                    )?;
                    scored += 1;
                }
                scored
            }
        };
        debug_assert_eq!(scored, selected.len());
        Ok(scored)
    }

    /// Scores one row into `bound`, `fold` and, when it can enter the top k, the heap.
    ///
    /// `position` indexes the selection's doc ids and `offset` the cluster's DocIds column, read
    /// once per selection on first use. The bound is keyed by [`RowAddress`] so doc lookup stays
    /// deferred until the score can enter the result heap.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn offer(
        &mut self,
        selected: &ClusterRows<'_>,
        bound: &mut PruningBound,
        fold: &mut Option<Fold<'_>>,
        column_read: &mut bool,
        position: usize,
        offset: usize,
        global_row: usize,
        bytes: &[u8],
    ) -> crate::Result<()> {
        let score = self.query.score_row(bytes).score();
        #[cfg(test)]
        self.exact_scores.push(score);
        bound.offer([Candidate::exact(
            RowAddress {
                segment_ord: self.segment_ord,
                row_id: global_row,
            },
            score,
        )]);
        if let Some(fold) = fold {
            fold.score(score);
        }
        // Rows that cannot enter the top k skip the tie-break and the doc id lookup: below the
        // shared floor, or below the result heap's own threshold.
        if bound.shared_threshold().is_some_and(|kth| score < kth)
            || self
                .topn
                .threshold
                .as_ref()
                .is_some_and(|((threshold, _), _)| score < *threshold)
        {
            return Ok(());
        }
        let doc = match selected.docs {
            Some(docs) => docs[position],
            None => {
                if !*column_read {
                    let _fetch = enter_vector_stage(VectorStage::ExactFetch);
                    self.reader
                        .read_doc_ids(selected.cluster, &mut self.cluster_docs)?;
                    *column_read = true;
                }
                self.cluster_docs[offset]
            }
        };
        if let Some(key) = tie_break_key(&self.topn, &mut *self.tie_break, score, doc) {
            self.topn.push_unordered(key, doc);
        }
        Ok(())
    }

    // --- Reads ---

    /// Reads the selected full-precision rows: the cluster's span, or the rows planned into
    /// `scratch`.
    fn read<'s>(
        &self,
        selected: &ClusterRows<'_>,
        scratch: &'s mut ReadScratch,
    ) -> crate::Result<ExactRows<'s>> {
        Ok(match selected.offsets {
            None => ExactRows::Span(self.reader.read_cluster_rows(selected.cluster)?),
            Some(_) => {
                scratch.rows.clear();
                scratch.rows.extend(selected.global_rows());
                ExactRows::Planned(self.reader.read_vector_rows_planned(scratch)?)
            }
        })
    }

    /// The heap's hits, best first, tagged with `segment_ord`.
    pub(super) fn into_hits(self, segment_ord: SegmentOrdinal) -> TieBreakHits<K> {
        self.topn
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| {
                (
                    candidate.sort_key,
                    DocAddress::new(segment_ord, candidate.doc),
                )
            })
            .collect()
    }
}
