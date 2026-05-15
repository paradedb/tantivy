//! Flat vector storage format.
//!
//! Stores full-precision vectors contiguously per segment, multiplexed
//! across fields via a [`CompositeFile`](crate::directory::CompositeFile).
//! The segment file extension is `.flatvec`. Lookup is
//! `slice_start + doc_id * dim * 4`.
//!
//! Exposed as a format inside the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) — the
//! [`FlatVecWriter`] is the per-doc accumulator (every initial segment
//! write produces flatvec, since clustering is merge-time only) and
//! [`merge_flat`] is the merge routine the parent plugin calls below
//! the clustering threshold.

mod plugin;
mod presence;
mod reader;
mod writer;

/// Segment file extension for flat-format vector storage. Used both
/// as the `SegmentComponent::Custom` tag for open_write/open_read and
/// as the extension reported by [`VectorPlugin::extensions`](crate::vector::VectorPlugin)
/// for GC accounting.
pub(crate) const FLATVEC_EXT: &str = "flatvec";

pub(crate) use plugin::merge_flat;
pub use reader::{FlatVecReader, VectorColumn};
pub use writer::FlatVecWriter;

#[cfg(test)]
mod tests;
