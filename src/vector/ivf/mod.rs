//! Inverted-file vector storage and cluster routing.

mod aps;
mod assignments;
pub(crate) mod bkt;
pub(crate) mod centroid_index;
pub(crate) mod graph;
mod index;
mod ivf;
mod params;
mod partition;
#[cfg(test)]
mod plugin;
mod training;
mod writer;

/// The IVF cluster-routing file. Written per field, only for IVF segments.
pub(crate) const CENTROIDS_EXT: &str = "centroids";

pub use aps::APS_MAX_DIM;
pub(crate) use aps::{supports_metric as aps_supports_metric, CandidateRows, RecallEstimator};
pub use bkt::{BKTree, BKTreeNode, BKTreeSearchIterator, NodeId as BktNodeId};
pub use centroid_index::CentroidProducer;
pub use graph::{
    Candidate, Graph, NeighborhoodGraphConfig, NeighborhoodGraphSearchMetrics, NodeId,
    RelativeNeighborhoodGraph, ResumableSearchIterator, SearchIterator, SearchTerminationReason,
    Workspace,
};
pub use index::IvfIndex;
pub(crate) use index::RouterIndex;
pub use ivf::{
    AddLevelError, ClusterId, InMemoryStackedIvf, InMemoryStore, IvfConfig,
    IvfIndex as MultiLevelIvf, IvfIndexBuilder, IvfLevelClusterer, LazyStackedIvf, LazyStore,
    StackedSearchStats, SuperKMeansLevelClusterer, PARENT_NPROBE_FRACTION,
};
pub use params::{AdaptiveProbeParams, WorkModel, DEFAULT_ROUTER_RECALL};
pub(crate) use training::{decode_row, decode_row_append, encode_vector};
pub use training::{IvfCentroids, IvfMatrix};
pub(crate) use writer::{merge_shared, IvfVecWriter, SharedSegmentMeta};
