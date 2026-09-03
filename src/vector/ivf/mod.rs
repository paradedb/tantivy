//! IVF (inverted-file) vector storage format.
//!
//! The unified [`VectorPlugin`](crate::vector::VectorPlugin) routes to
//! this module when the merge target meets
//! [`IndexSettings::vector_clustering_threshold`](crate::index::IndexSettings::vector_clustering_threshold),
//! which defaults to 10k docs.

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
pub use index::IvfIndex;
pub use ivf::{
    AddLevelError, ClusterId, InMemoryStackedIvf, InMemoryStore, IvfConfig,
    IvfIndex as MultiLevelIvf, IvfIndexBuilder, IvfLevelClusterer, LazyStackedIvf, LazyStore,
    SuperKMeansLevelClusterer,
};
pub use params::{AdaptiveProbeParams, WorkModel};
pub(crate) use plugin::merge_ivf;
pub(crate) use training::{decode_row, encode_vector};
pub use training::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfMatrixView, IvfMergeSettings, IvfTrainingBatch,
    IvfTrainingVectors, IvfVectorBatch, IvfVectors,
};
