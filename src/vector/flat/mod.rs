//! Flat vector index plugin.
//!
//! Stores full-precision vectors contiguously per segment, multiplexed across
//! fields via a [`CompositeFile`](crate::directory::CompositeFile). The segment
//! file extension is `.flatvec`. Lookup is `slice_start + doc_id * dim * 4`.

mod plugin;
mod presence;
mod reader;
mod writer;

pub use plugin::FlatVecPlugin;
pub use reader::{FlatVecReader, VectorColumn};
pub use writer::FlatVecPluginWriter;

#[cfg(test)]
mod tests;
