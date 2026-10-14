//! Inverted-file vector storage and cluster routing.

mod aps;
#[cfg(test)]
mod build_tests;
mod candidate;
mod clusterer;
mod index;
mod ivf;
mod params;
mod plugin;
mod training;

/// The IVF cluster-routing file. Written per field, only for IVF segments.
pub(crate) const CENTROIDS_EXT: &str = "centroids";

pub use aps::APS_MAX_DIM;
pub(crate) use aps::{supports_metric as aps_supports_metric, CandidateRows, RecallEstimator};
pub use candidate::Candidate;
pub use clusterer::{SuperKMeansClusterer, DEFAULT_MAX_LEAF_SIZE, DEFAULT_TRAINING_SAMPLE_RATIO};
pub use index::IvfIndex;
pub use ivf::{
    AddLevelError, ClusterId, InMemoryStackedIvf, InMemoryStore, IvfConfig,
    IvfIndex as MultiLevelIvf, IvfIndexBuilder, IvfLevelClusterer, LazyStackedIvf, LazyStore,
    StackedSearchStats, SuperKMeansLevelClusterer, PARENT_NPROBE_FRACTION,
};
pub use params::{AdaptiveProbeParams, LocatedPath, Selectivity, WorkModel, DEFAULT_ROUTER_RECALL};
pub(crate) use plugin::merge_ivf;
pub(crate) use training::{decode_row, decode_row_append, encode_vector};
pub use training::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfMatrixView, IvfMergeSettings, IvfTrainingBatch,
    IvfTrainingVectors, IvfVectorBatch, IvfVectors,
};
