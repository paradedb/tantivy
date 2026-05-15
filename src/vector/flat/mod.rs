//! Flat vector storage format.
//!
//! Status: scaffolding only. The on-disk format and the actual
//! reader/writer code lands in the next PR. The types below exist so
//! that the unified [`VectorPlugin`](crate::vector::VectorPlugin) and
//! the [`VectorBackend`](crate::vector::VectorBackend) dispatch
//! compile end-to-end and the public surface is stable; every method
//! body is `todo!()` and the plugin isn't registered yet, so nothing
//! actually fires.

mod plugin;
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
