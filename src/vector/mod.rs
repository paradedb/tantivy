//! Vector field type, distance kernels, and the per-segment storage plugin.
//!
//! Schema-level concepts ([`VectorOptions`], [`Metric`], [`VectorElement`])
//! and the distance kernels live at this level. The on-disk formats
//! live in submodules: [`flat`] for the dense full-precision layout
//! and [`ivf`] for the partitioned/clustered accelerator.
//! Both formats are owned by a single [`VectorPlugin`] which picks
//! between them per merge based on
//! [`IndexSettings::vector_clustering_threshold`](crate::index::IndexSettings::vector_clustering_threshold).
//! Top-N vector queries dispatch over them via [`VectorBackend`].

mod backend;
mod collector;
mod distance;
mod meta;
mod options;
mod plugin;
mod reader;

pub mod flat;
pub mod ivf;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use backend::ProbeStats;
pub use backend::VectorBackend;
pub use collector::TopDocsByVectorSimilarity;
pub use distance::{cosine, cosine_bytes, dot, dot_bytes, l2_squared, l2_squared_bytes};
pub use flat::{FlatVecReader, FlatVecWriter, FlatVectorColumn};
pub use ivf::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfMatrixView, IvfMergeSettings, IvfVectorBatch,
    IvfVectors,
};
pub use options::{Metric, VectorDType, VectorElement, VectorOptions};
pub use plugin::VectorPlugin;
pub(crate) use plugin::VECTOR_PLUGIN_NAME;
pub use reader::{VectorColumn, VectorColumnReader, VectorReader};
