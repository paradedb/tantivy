//! Per-cluster planning vocabulary for the IVF scan.
//!
//! A [`ClusterBatch`] is one cluster's rows entering one cascade [`Stage`]: the rows it selects,
//! the documents they belong to (or how to find them), and what is known about the query's
//! similarity to the cluster's centroid. Cluster sources produce layer-0 batches; the segment
//! scan regroups the survivors of each later stage into batches of its own.

use std::ops::Range;

use super::Similarity;
use crate::DocId;

/// A cascade stage for one cluster's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Quantized layer `l`. Layer 0 admits rows from a cluster source; later layers refine the
    /// survivors of the previous boundary.
    Layer(usize),
    /// Exact scoring of every row still live after the last layer.
    Final,
}

/// The query-centroid similarity, when the source has it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CentroidScore {
    /// The exact similarity the router keys on.
    Known(Similarity),
}

/// The rows of one cluster a batch selects.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Selection<'a> {
    /// Every row of the cluster.
    All,
    /// Strictly ascending cluster-local row offsets, at least one.
    Rows(&'a [usize]),
}

impl Selection<'_> {
    /// Number of selected rows in a cluster spanning `rows`.
    #[inline]
    pub(crate) fn len(&self, rows: &Range<usize>) -> usize {
        match self {
            Self::All => rows.len(),
            Self::Rows(offsets) => offsets.len(),
        }
    }
}

/// Document ids for a batch's rows. Within one cluster they are all resolved or all deferred.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SelectedDocs<'a> {
    /// The cluster's DocIds column, one entry per cluster row.
    ByClusterOffset(&'a [DocId]),
    /// One document per selected row, aligned with [`Selection::Rows`].
    BySelection(&'a [DocId]),
    /// Unread; resolved from the DocIds column only for rows that need a document.
    Deferred,
}

/// One cluster's rows entering a stage.
pub(crate) struct ClusterBatch<'a> {
    pub(crate) stage: Stage,
    pub(crate) cluster: usize,
    /// The cluster's full global row range.
    pub(crate) rows: Range<usize>,
    pub(crate) selection: Selection<'a>,
    pub(crate) docs: SelectedDocs<'a>,
    pub(crate) centroid: CentroidScore,
}

impl ClusterBatch<'_> {
    /// Number of selected rows.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.selection.len(&self.rows)
    }
}
