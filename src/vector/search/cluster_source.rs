//! The vocabulary shared by cluster sources and the segment scan: one cluster's selected rows,
//! the scan's feedback to its source, and the source contract.

use std::ops::Range;

use crate::vector::distance::Similarity;
use crate::vector::search::scan::rate::RateReading;
use crate::DocId;

/// One cluster's selected rows, with what is already known about them.
pub(crate) struct ClusterRows<'a> {
    pub(crate) cluster: usize,
    /// The cluster's full global row range.
    pub(crate) rows: Range<usize>,
    /// Strictly ascending cluster-local offsets of the selected rows, at least one; `None`
    /// selects every row.
    pub(crate) offsets: Option<&'a [u32]>,
    /// Doc ids aligned with `offsets`, when already in hand; otherwise the scan reads the DocIds
    /// column for the rows that need one.
    pub(crate) docs: Option<&'a [DocId]>,
    /// The query's similarity to the cluster's centroid under the field metric, when already
    /// computed; otherwise the scan computes it if a quantized read needs it, and the plan prices
    /// that read.
    pub(crate) centroid: Option<Similarity>,
}

impl ClusterRows<'_> {
    /// Number of selected rows.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.offsets.map_or(self.rows.len(), <[u32]>::len)
    }

    /// The selected rows' global row numbers, ascending.
    pub(crate) fn global_rows(&self) -> impl Iterator<Item = usize> + '_ {
        let (span, offsets) = match self.offsets {
            None => (self.rows.clone(), &[][..]),
            Some(offsets) => (0..0, offsets),
        };
        span.chain(
            offsets
                .iter()
                .map(|&offset| self.rows.start + offset as usize),
        )
    }
}

/// What the scan reports to a source after each cluster.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Feedback {
    /// The pruning threshold: the k-th largest lower endpoint, once `k` rows are in.
    pub(crate) kth_lower: Option<f32>,
    /// The APS cover threshold: never above the k-th best estimate, so it never prunes.
    pub(crate) kth_estimate: Option<f32>,
    /// The admission rate under the pruning threshold; `None` with the RateAPS off or before the
    /// opening exists.
    pub(crate) rate: Option<RateReading>,
}

/// Supplies a segment scan's clusters in its own order and decides when to stop.
pub(crate) trait ClusterSource {
    /// The next cluster's rows, or `None` when the source stops.
    fn next(&mut self, feedback: Feedback) -> crate::Result<Option<ClusterRows<'_>>>;
    /// Records that the last cluster was shortlisted with `rows` scored; `feedback` includes them.
    fn admitted(&mut self, rows: usize, feedback: Feedback) -> crate::Result<()>;
}
