//! Per-cluster planning vocabulary for the IVF scan.
//!
//! A [`ClusterBatch`] is one cluster's rows entering one cascade [`Stage`]: the rows it selects,
//! the documents they belong to (or how to find them), and what is known about the query's
//! similarity to the cluster's centroid. Cluster sources produce layer-0 batches; the segment
//! scan regroups the survivors of each later stage into batches of its own.
//!
//! Each batch is read the cheapest way its storage layout allows ([`ReadPlan`]), judged by the
//! storage blocks each read would touch ([`ReadCost`]). The decision covers a whole cluster at a
//! stage boundary and changes only reads, never scores.

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
    /// Not computed: a quantized layer-0 read must first read the centroid row.
    Unknown,
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

    /// Selected cluster-local offsets; `None` selects every row.
    #[inline]
    pub(crate) fn offsets(&self) -> Option<&[usize]> {
        match self {
            Self::All => None,
            Self::Rows(offsets) => Some(offsets),
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

/// How a batch's rows are read at its stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadPlan {
    /// The selected rows at full precision, scored exactly: the rows finish here.
    Exact,
    /// The selected rows' code runs, the layer's sidecar span, and at layer 0 the residual
    /// norms.
    Sparse,
    /// The layer's whole band in one request.
    Full,
}

/// A read's cost in the segment's one unit: storage blocks when the storage has block geometry,
/// bytes otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ReadCost(pub(crate) usize);

impl std::ops::Add for ReadCost {
    type Output = ReadCost;

    fn add(self, rhs: ReadCost) -> ReadCost {
        ReadCost(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for ReadCost {
    fn add_assign(&mut self, rhs: ReadCost) {
        self.0 += rhs.0;
    }
}

/// The two ways to read one batch at a quantized layer, priced from slice geometry alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayerCosts {
    /// The sparse read; `None` when every row is selected.
    pub(crate) sparse: Option<ReadCost>,
    /// The whole band.
    pub(crate) full: ReadCost,
    /// The centroid row either quantized read needs first: non-zero only at layer 0 with an
    /// unknown centroid score.
    pub(crate) centroid: ReadCost,
}

impl LayerCosts {
    /// Sparse only when strictly cheaper than the whole band; a tie reads the band in one
    /// request.
    pub(crate) fn plan(&self) -> ReadPlan {
        match self.sparse {
            Some(sparse) if sparse < self.full => ReadPlan::Sparse,
            _ => ReadPlan::Full,
        }
    }

    /// The cheaper of the two quantized reads.
    pub(crate) fn cheapest(&self) -> ReadCost {
        self.sparse
            .map_or(self.full, |sparse| sparse.min(self.full))
    }
}

/// Every way to read one batch at its stage, priced from slice geometry alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BatchCosts {
    /// The selected rows' full-precision reads: the distinct storage blocks they touch.
    pub(crate) exact: ReadCost,
    /// The quantized reads; `None` at `Final`, where only exact scoring remains.
    pub(crate) layer: Option<LayerCosts>,
}

impl BatchCosts {
    /// Exact only when strictly cheaper than the cheaper quantized read, so ties stay
    /// quantized; otherwise the layer's own plan. The rerank of quantized rows is not priced,
    /// which keeps the rule conservative toward quantized reads.
    pub(crate) fn plan(&self, exact_enabled: bool) -> ReadPlan {
        let Some(layer) = self.layer else {
            return ReadPlan::Exact;
        };
        if exact_enabled && self.exact < layer.cheapest() + layer.centroid {
            return ReadPlan::Exact;
        }
        layer.plan()
    }
}
