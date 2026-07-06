//! IVF (inverted-file) vector storage format.
//!
//! The unified [`VectorPlugin`](crate::vector::VectorPlugin) routes to
//! this module when the merge target meets
//! [`IndexSettings::vector_clustering_threshold`](crate::index::IndexSettings::vector_clustering_threshold),
//! which defaults to 10k docs.

mod centroids;
mod plugin;
mod reader;
mod training;

/// The IVF cluster-routing file. Written per field, only for IVF segments.
pub(crate) const CENTROIDS_EXT: &str = "centroids";

pub(crate) use centroids::CentroidsMeta;
pub(crate) use plugin::merge_ivf;
pub use reader::{IvfVecReader, IvfVectorColumn};
pub(crate) use training::{decode_row, encode_vector};
pub use training::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfMatrixView, IvfMergeSettings, IvfVectorBatch,
    IvfVectors,
};
