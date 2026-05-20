//! IVF (inverted-file) vector storage format.
//!
//! The unified [`VectorPlugin`](crate::vector::VectorPlugin) routes to
//! this module when the merge target meets
//! [`IndexSettings::vector_clustering_threshold`](crate::index::IndexSettings::vector_clustering_threshold),
//! which defaults to `usize::MAX` (effectively off).

mod params;
mod plugin;
mod reader;
mod training;

#[cfg(test)]
mod tests;

pub use params::AdaptiveProbeParams;
pub(crate) const ASSIGNMENTS_EXT: &str = "assignments";
pub(crate) const IVFVEC_EXT: &str = "vec";
pub(crate) use super::meta::IvfFieldMeta;
pub(crate) use plugin::merge_ivf;
pub use reader::{IvfVecReader, IvfVectorColumn};
pub(crate) use training::{decode_row, encode_vector};
pub use training::{
    IvfCentroids, IvfClusterer, IvfMergeSettings, IvfTypedVector, IvfVector, IvfVectors,
};
