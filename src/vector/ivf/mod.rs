//! IVF (inverted-file) vector storage — the clustered layout.
//!
//! Every write path here assigns vectors against the index-level centroid
//! index (see [`crate::vector::ivf::centroid_index`]) and stores cluster-sorted
//! rows; the per-segment remainder — offsets, bounds, IVF meta — is read
//! back through [`SegmentClusters`]. (Indexes without a centroid index store the
//! flat layout instead, written inline by
//! [`VecWriter`](crate::vector::VecWriter).)

pub(crate) mod assign;
pub(crate) mod bounds;
pub(crate) mod centroid_index;
pub(crate) mod graph;
mod params;
mod partition;
mod plugin;
mod segment_clusters;
mod types;

pub use graph::{
    Candidate, Graph, NeighborhoodGraphConfig, NeighborhoodGraphSearchMetrics, NodeId,
    RelativeNeighborhoodGraph, ResumableSearchIterator, SearchIterator, SearchTerminationReason,
    Workspace,
};
pub use params::AdaptiveProbeParams;
pub(crate) use plugin::{merge_ivf, write_ivf_field, IvfFieldWriteParams};
pub use segment_clusters::{IvfSearchMetrics, SegmentClusters};
pub(crate) use types::{decode_row, encode_vector};
pub use types::{IvfCentroids, IvfMatrix};
