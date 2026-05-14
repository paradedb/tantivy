//! IVF (inverted-file) vector index plugin.
//!
//! Status: stub. The type surface exists so that the dispatch in
//! [`VectorBackend`](crate::vector::backend::VectorBackend) compiles and
//! callers can wire up `TopDocsByVectorSimilarity` end-to-end, but no
//! plugin is actually registered yet and the runtime methods are
//! `todo!()`. Until the writer/reader land, every segment falls through
//! to the flat backend.

mod params;
mod reader;

pub use params::AdaptiveProbeParams;
pub use reader::{IvfVecReader, IvfVectorColumn};
