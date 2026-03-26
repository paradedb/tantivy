use serde::{Deserialize, Serialize};

/// Configuration for a dense vector field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorOptions {
    /// Number of dimensions in the vector.
    pub dimensions: usize,
}

impl VectorOptions {
    /// Create vector options with the given number of dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}
