//! Inverted-file vector storage and cluster routing.

pub(crate) mod bkt;
pub(crate) mod graph;
mod index;
mod ivf;
mod params;
mod partition;
mod plugin;
mod training;

/// The IVF cluster-routing file. Written per field, only for IVF segments.
pub(crate) const CENTROIDS_EXT: &str = "centroids";

pub use bkt::{BKTree, BKTreeNode, BKTreeSearchIterator, NodeId as BktNodeId};
pub use graph::{
    Candidate, Graph, NeighborhoodGraphConfig, NeighborhoodGraphSearchMetrics, NodeId,
    RelativeNeighborhoodGraph, ResumableSearchIterator, SearchIterator, SearchTerminationReason,
    Workspace,
};
pub use index::{IvfIndex, IvfSearchMetrics, RoutingIndex};
pub use ivf::{
    AddLevelError, ClusterId, FlatStore, IvfConfig, IvfIndex as MultiLevelIvf, IvfIndexBuilder,
    IvfLevelClusterer, PersistedStackedIvf, SliceStore, StackedIvfIndex, SuperKMeansLevelClusterer,
};
pub use params::{AdaptiveProbeParams, WorkModel};
pub(crate) use plugin::merge_ivf;
pub(crate) use training::{decode_row, decode_row_append, encode_vector};
pub use training::{
    BuiltRouter, IvfCentroids, IvfClusterer, IvfMatrix, IvfMatrixView, IvfMergeSettings,
    IvfTrainingBatch, IvfTrainingVectors, IvfVectorBatch, IvfVectors,
};
