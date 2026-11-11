//! Distance kernels, scan hooks and the stacked IVF, for the benches only; not a stable API.

use crate::vector::distance::{admit, normalize_row_with_norm_squared, row_norm_squared};
pub use crate::vector::distance::{
    cosine, cosine_bytes, dot, dot_bytes, l2_squared, l2_squared_bytes,
    quantization_bench_dot_bytes_f32,
};
use crate::vector::quantization::VectorNormPolicy;

/// The add path's per-row work: validation, and normalization when the metric requires it.
/// Returns whether the row is accepted.
pub fn add_row_check(opts: &crate::schema::VectorOptions, row: &mut [u8]) -> bool {
    let norm_policy = VectorNormPolicy::for_options(opts);
    let norm_squared = row_norm_squared(opts.dtype(), row);
    if admit(norm_squared, norm_policy).is_err() {
        return false;
    }
    match norm_policy {
        VectorNormPolicy::UnitL2 => {
            normalize_row_with_norm_squared(opts.dtype(), row, norm_squared);
        }
        VectorNormPolicy::None => {}
    }
    true
}
pub use crate::vector::router::candidate::Candidate;
pub use crate::vector::router::stacked::levels::{
    ClusterId, InMemoryStore, LevelClusterer, StackedIvf, StackedIvfBuilder, StackedIvfConfig,
    StackedSearchStats, SuperKMeansLevelClusterer,
};
pub use crate::vector::search::scan::columns::Columns;
pub use crate::vector::search::scan::estimate::quantization_bench_layer0_cosine_cluster;
pub use crate::vector::search::scan::layer_read::Decoded;
