use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorDType {
    F32,
    F16,
}

impl VectorDType {
    pub fn size_bytes(self) -> usize {
        match self {
            VectorDType::F32 => 4,
            VectorDType::F16 => 2,
        }
    }
}

/// Distance / similarity metric used when ranking vector field values.
///
/// All metrics are presented to callers in a "higher is better" orientation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
            + if self.dtype == VectorDType::F16 && self.metric == Metric::Cosine {
                4
            } else {
                0
            }
    }

    /// Whether this field's `(metric, dtype)` requires write-time
    /// unit-normalization (see `vector::distance::maybe_normalize_bytes`).
    /// Cosine rows carry unit-normalization in both storage formats.
    pub fn needs_normalization(&self) -> bool {
        self.metric == Metric::Cosine
    }
}

#[cfg(test)]
mod tests {
    use crate::schema::{Metric, Schema, VectorOptions};

    #[test]
    fn test_vector_field_schema_round_trip() {
        let mut schema_builder = Schema::builder();
        schema_builder.add_vector_field("embedding", VectorOptions::new(128, Metric::Cosine));
        let schema = schema_builder.build();

        let schema_json = serde_json::to_string_pretty(&schema).unwrap();
        let expected = r#"[
  {
    "name": "embedding",
    "type": "vector",
    "options": {
      "dim": 128,
      "dtype": "f32",
      "metric": "cosine"
    }
  }
]"#;
        assert_eq!(schema_json, expected);

        let deserialized: Schema = serde_json::from_str(expected).unwrap();
        assert_eq!(schema, deserialized);
    }
}
