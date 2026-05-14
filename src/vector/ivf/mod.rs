//! IVF (inverted-file) vector index plugin.
//!
//! Status: plugin registered, merge hook present, clustering algorithm
//! pending. The plugin participates in the merge lifecycle and
//! short-circuits below
//! [`IndexSettings::vector_clustering_threshold`](crate::index::IndexSettings).
//! At/above the threshold the body is a TODO — when it lands, every
//! qualifying merge writes a `.ivfvec` file. Until then, both the
//! reader's `open_column` and the search-side `IvfBackend::top_n` stay
//! `None`/`todo!()` and queries fall through to the flat backend.

mod params;
mod plugin;
mod reader;
mod writer;

pub use params::AdaptiveProbeParams;
pub use plugin::IvfVecPlugin;
pub use reader::{IvfVecReader, IvfVectorColumn};
pub use writer::IvfVecPluginWriter;
