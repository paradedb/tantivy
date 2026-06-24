//! Clustering abstraction for IVF (inverted-file) vector storage.
//!
//! This module currently exposes only the [`IvfClusterer`] trait and its
//! data types. An index registers a clusterer via
//! [`IndexBuilder::ivf_clusterer`](crate::IndexBuilder::ivf_clusterer); the
//! IVF on-disk format and the merge that consumes the clusterer land in a
//! follow-up.

mod training;

pub use training::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfMatrixView, IvfMergeSettings, IvfVectorBatch,
    IvfVectors,
};
