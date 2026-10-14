//! Distance kernels, scan hooks and the stacked IVF, for the benches only; not a stable API.

pub use crate::vector::distance::{
    cosine, cosine_bytes, dot, dot_bytes, l2_squared, l2_squared_bytes,
    quantization_bench_dot_bytes_f32,
};
pub use crate::vector::router::candidate::Candidate;
pub use crate::vector::router::stacked::levels::{
    ClusterId, InMemoryStore, LevelClusterer, StackedIvf, StackedIvfBuilder, StackedIvfConfig,
    StackedSearchStats, SuperKMeansLevelClusterer,
};
pub use crate::vector::search::prepared::VectorQuery as PreparedQuery;
pub use crate::vector::search::scan::columns::Columns;
pub use crate::vector::search::scan::estimate::quantization_bench_layer0_cosine_cluster;
pub use crate::vector::search::scan::layer_read::Decoded;
