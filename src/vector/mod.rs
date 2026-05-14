//! Vector field type, distance kernels, and per-format index plugins.
//!
//! Schema-level concepts ([`VectorOptions`], [`Metric`], [`VectorElement`])
//! and the distance kernels live at this level. The on-disk formats
//! live in submodules: [`flat`] for the dense full-precision layout
//! (the default and always-available raw store), [`ivf`] for the
//! partitioned/clustered accelerator (stub). Top-N vector queries
//! dispatch over them via [`VectorBackend`].

mod backend;
mod collector;
mod distance;
mod options;

pub mod flat;
pub mod ivf;

pub use backend::VectorBackend;
pub use collector::TopDocsByVectorSimilarity;
pub use distance::{cosine, cosine_bytes, dot, dot_bytes, l2_squared, l2_squared_bytes};
pub use flat::{FlatVecPlugin, FlatVecPluginWriter, FlatVecReader, VectorColumn};
pub use ivf::{IvfVecPlugin, IvfVecPluginWriter};
pub use options::{Metric, VectorDType, VectorElement, VectorOptions};
