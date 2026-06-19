use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorDType {
    F32,
}

impl VectorDType {
    pub fn size_bytes(self) -> usize {
        match self {
            VectorDType::F32 => 4,
        }
    }
}

/// Distance / similarity metric used when ranking vector field values.
///
/// All metrics are presented to callers in a "higher is better" orientation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    L2,
    Cosine,
    Dot,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorOptions {
    dim: usize,
    dtype: VectorDType,
    metric: Metric,
}

impl VectorOptions {
    pub fn new(dim: usize, metric: Metric) -> VectorOptions {
        VectorOptions {
            dim,
            dtype: VectorDType::F32,
            metric,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn dtype(&self) -> VectorDType {
        self.dtype
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn with_dtype(mut self, dtype: VectorDType) -> VectorOptions {
        self.dtype = dtype;
        self
    }

    pub fn bytes_per_vector(&self) -> usize {
        self.dim * self.dtype.size_bytes()
    }
}
