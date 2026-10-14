//! Exact scoring: full-precision reads of a cluster's selected rows, scored into the pruning bound
//! and the result heap.

use std::ops::Range;

use common::OwnedBytes;

use super::bound::KthBound;
use super::heap::{tie_break_key, TieBreakHeap, TieBreakHits};
use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, TopNComputer};
use crate::vector::cluster_source::ClusterRows;
use crate::vector::index_reader::{VectorIndexReader, VectorRowBatch};
use crate::vector::prepared::VectorQuery;
use crate::vector::{enter_vector_stage, Stage, VectorElement};
use crate::{DocAddress, DocId, SegmentOrdinal};

/// Full-precision rows of one cluster's selection.
enum ExactRows {
    /// The cluster's whole full-precision span, in one request.
    Span(OwnedBytes),
    /// The selected rows, through a storage-aware read plan.
    Planned(VectorRowBatch),
}

/// Scores rows exactly into the bound and the result heap; the only way rows reach the heap.
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
    pub(super) bound: KthBound,
    rows: Vec<usize>,
    read_ranges: Vec<Range<usize>>,
    block_scratch: Vec<(usize, usize)>,
    /// A cluster's DocIds column, read when its selected rows carry no doc ids.
    cluster_docs: Vec<DocId>,
    /// Global rows scored in the exact phase, in order; the scan copies them into its trace.
    #[cfg(test)]
    pub(super) exact_phase_rows: Vec<usize>,
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
        tie_break: &'a mut K,
        tie_comparator: C,
    ) -> Self {
        Self {
            reader,
            query,
            topn: TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator)),
            tie_break,
            bound: KthBound::new(top_n),
            rows: Vec::new(),
            read_ranges: Vec::new(),
            block_scratch: Vec::new(),
            cluster_docs: Vec::new(),
            #[cfg(test)]
            exact_phase_rows: Vec::new(),
        }
    }

    // --- Scoring ---

    /// Scores every row `selected` selects; returns the rows scored. A row without a doc id reads
    /// the cluster's DocIds column when it first needs one.
    pub(super) fn score(&mut self, selected: &ClusterRows<'_>) -> crate::Result<usize> {
        self.score_rows(selected, false)
    }

    /// Scores the exact phase's survivors in `selected`; returns the rows scored. Without doc ids
    /// in hand, the cluster's DocIds column is read up front with the rows.
    pub(super) fn score_survivors(&mut self, selected: &ClusterRows<'_>) -> crate::Result<usize> {
        #[cfg(test)]
        self.exact_phase_rows.extend(selected.global_rows());
        self.score_rows(selected, selected.docs.is_none())
    }

    /// Reads `selected`'s rows in `ExactFetch`, with the DocIds column first when `docs_first`, and
    /// scores them in `ExactScore`.
    fn score_rows(&mut self, selected: &ClusterRows<'_>, docs_first: bool) -> crate::Result<usize> {
        let mut column_read = false;
        let rows = {
            let _fetch = enter_vector_stage(Stage::ExactFetch);
            if docs_first {
                self.reader
                    .read_doc_ids(selected.cluster, &mut self.cluster_docs)?;
                column_read = true;
            }
            self.read(selected)?
        };
        let _score = enter_vector_stage(Stage::ExactScore);

        let scored = match &rows {
            ExactRows::Span(bytes) => {
                let stride = self.reader.options().bytes_per_vector();
                for (local, row_bytes) in bytes.chunks_exact(stride).enumerate() {
                    self.offer(selected, &mut column_read, local, local, row_bytes)?;
                }
                bytes.len() / stride
            }
            ExactRows::Planned(planned) => {
                let mut scored = 0;
                for (position, (row, row_bytes)) in planned.iter().enumerate() {
                    self.offer(
                        selected,
                        &mut column_read,
                        position,
                        row - selected.rows.start,
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

    /// Scores one row into the bound and, when it can enter the top k, the heap.
    ///
    /// `position` indexes the selection's doc ids and `offset` the cluster's DocIds column, read
    /// once per selection on first use.
    #[inline]
    fn offer(
        &mut self,
        selected: &ClusterRows<'_>,
        column_read: &mut bool,
        position: usize,
        offset: usize,
        bytes: &[u8],
    ) -> crate::Result<()> {
        let score = self.query.score_row(bytes).score();
        self.bound.push_exact(score);
        // Rows that cannot enter the top k skip the tie-break and the doc id lookup.
        if self.bound.exact_kth().is_some_and(|kth| score < kth)
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

    /// Reads the selected full-precision rows: the cluster's span, or the rows planned.
    fn read(&mut self, selected: &ClusterRows<'_>) -> crate::Result<ExactRows> {
        Ok(match selected.offsets {
            None => ExactRows::Span(self.reader.read_cluster_rows(selected.cluster)?),
            Some(_) => {
                self.rows.clear();
                self.rows.extend(selected.global_rows());
                ExactRows::Planned(self.reader.read_vector_rows_planned(
                    &self.rows,
                    &mut self.read_ranges,
                    &mut self.block_scratch,
                )?)
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
