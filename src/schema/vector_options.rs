use serde::{Deserialize, Serialize};

/// Distance metric used to compare vectors at query time.
///
/// Stored in the schema so plugins (clustering, quantization) can be configured
/// consistently with how the field will be searched. `Cosine` is treated as
/// `InnerProduct` over L2-normalized vectors at the plugin layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VectorMetric {
    /// Euclidean (L2) distance.
    L2,
    /// Cosine distance (1 - cosine similarity).
    Cosine,
    /// Negative inner product (maximum similarity).
    InnerProduct,
}

/// Configuration for a dense vector field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorOptions {
    /// Number of dimensions in the vector.
    pub dimensions: usize,
    /// Distance metric used at query time. No default — schemas must
    /// pick one explicitly so the plugin layer ranks consistently with
    /// how the caller intends to query.
    pub metric: VectorMetric,
}

impl VectorOptions {
    /// Create vector options with the given dimensions and metric.
    pub fn new(dimensions: usize, metric: VectorMetric) -> Self {
        Self { dimensions, metric }
    }
}
