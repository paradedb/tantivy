//! Vector field type, distance kernels, and the per-segment storage plugin.
//!
//! Schema-level concepts ([`VectorOptions`], [`Metric`], [`VectorElement`])
//! and the distance kernels live at this level. The on-disk format lives
//! in the [`flat`] submodule (dense full-precision layout), owned by the
//! [`VectorPlugin`]. Top-N vector queries dispatch over it via
//! [`VectorBackend`].

mod backend;
mod collector;
mod distance;
mod meta;
mod options;
mod plugin;
mod reader;

pub mod flat;

pub use backend::VectorBackend;
pub use collector::TopDocsByVectorSimilarity;
pub use distance::{cosine, cosine_bytes, dot, dot_bytes, l2_squared, l2_squared_bytes};
pub use flat::{FlatVecReader, FlatVecWriter, FlatVectorColumn};
pub use options::{Metric, VectorDType, VectorElement, VectorOptions};
pub use plugin::VectorPlugin;
pub use reader::{VectorColumn, VectorColumnReader, VectorReader};
