//! Flat vector index plugin.
//!
//! Status: skeleton. The type surface exists so that
//! [`VectorBackend`](crate::vector::backend::VectorBackend) compiles and
//! callers can wire up [`TopDocsByVectorSimilarity`](crate::vector::TopDocsByVectorSimilarity)
//! end-to-end, but no plugin is actually registered yet and the runtime
//! methods are `todo!()`. The on-disk format (dense full-precision rows
//! multiplexed across fields via a
//! [`CompositeFile`](crate::directory::CompositeFile)) lands with the
//! real writer/reader.

mod plugin;
mod reader;
mod writer;

pub use plugin::FlatVecPlugin;
pub use reader::{FlatVecReader, VectorColumn};
pub use writer::FlatVecPluginWriter;
