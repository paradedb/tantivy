//! IVF (inverted-file) vector storage format.
//!
//! Status: scaffolding only. The [`merge_ivf`] body and the
//! [`IvfVecReader`]'s column-open path are both `todo!()` / `None`
//! until the clustering algorithm lands. Until then, the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) only routes to this
//! module when the merge target meets
//! [`IndexSettings::vector_clustering_threshold`](crate::index::IndexSettings::vector_clustering_threshold),
//! which defaults to 10k docs.

mod params;
mod plugin;
mod reader;
mod training;

pub use params::AdaptiveProbeParams;
pub(crate) use plugin::merge_ivf;
pub use reader::{IvfVecReader, IvfVectorColumn};
pub(crate) use training::{decode_row, encode_vector};
pub use training::{
    IvfCentroids, IvfClusterer, IvfMergeSettings, IvfTypedVector, IvfVector, IvfVectors,
};
