//! IVF (inverted-file) vector storage.
//!
//! From format V3 on this is the ONLY per-segment layout: every write path
//! assigns vectors against the index-level centroid index (see
//! [`crate::vector::centroid_index`]) and stores cluster-sorted rows. The
//! per-segment remainder — offsets, bounds, IVF meta — is read back through
//! [`IvfIndex`].

pub(crate) mod assign;
pub(crate) mod graph;
mod index;
mod params;
mod partition;
mod plugin;
mod types;

pub use graph::{
    Candidate, Graph, NeighborhoodGraphConfig, NeighborhoodGraphSearchMetrics, NodeId,
    RelativeNeighborhoodGraph, ResumableSearchIterator, SearchIterator, SearchTerminationReason,
    Workspace,
};
pub use index::{IvfIndex, IvfSearchMetrics};
pub use params::AdaptiveProbeParams;
pub(crate) use plugin::{merge_ivf, write_ivf_field, IvfFieldWriteParams};
pub(crate) use types::{decode_row, encode_vector};
pub use types::{IvfCentroids, IvfMatrix};
