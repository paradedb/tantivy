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

    /// L2-normalize `row` in place if this field's `(metric, dtype)`
    /// combination requires write-time unit-normalization for the
    /// search-time fast path. Currently only `Cosine + F32` triggers;
    /// every other combination is a no-op.
    ///
    /// Pre-normalizing at write time lets
    /// [`PreparedQuery::score_doc_bytes`](crate::vector::PreparedQuery::score_doc_bytes)
    /// reduce per-doc cosine work to `dot * inv_norm_q` — no per-doc
    /// `norm_squared_bytes` pass.
    pub fn maybe_normalize_bytes(&self, row: &mut [u8]) {
        debug_assert_eq!(row.len(), self.bytes_per_vector());
        if let (Metric::Cosine, VectorDType::F32) = (self.metric, self.dtype) {
            normalize_f32_inplace(row)
        }
    }
}

fn normalize_f32_inplace(row: &mut [u8]) {
    debug_assert_eq!(row.len() % 4, 0);
    let n = row.len() / 4;

    let mut sum_sq: f32 = 0.0;
    for i in 0..n {
        let off = i * 4;
        let v = f32::from_le_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]]);
        sum_sq += v * v;
    }
    let norm = sum_sq.sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return;
    }
    let inv = 1.0 / norm;
    for i in 0..n {
        let off = i * 4;
        let v = f32::from_le_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]]);
        let nv = v * inv;
        row[off..off + 4].copy_from_slice(&nv.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_f32_inplace;
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

    fn bytes(vec: &[f32]) -> Vec<u8> {
        vec.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn floats(buf: &[u8]) -> Vec<f32> {
        buf.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn normalize_scales_to_unit_norm() {
        let mut buf = bytes(&[3.0_f32, 0.0, 4.0]);
        normalize_f32_inplace(&mut buf);
        let out = floats(&buf);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6, "norm={n}, out={out:?}");
        // Direction preserved (dot with input ⇒ original L2 norm).
        let dot = 3.0 * out[0] + 0.0 * out[1] + 4.0 * out[2];
        assert!((dot - 5.0).abs() < 1e-5, "dot={dot}");
    }

    #[test]
    fn normalize_zero_vector_is_unchanged() {
        let mut buf = bytes(&[0.0_f32, 0.0, 0.0]);
        normalize_f32_inplace(&mut buf);
        assert_eq!(floats(&buf), vec![0.0_f32, 0.0, 0.0]);
    }

    #[test]
    fn normalize_already_unit_is_idempotent() {
        let unit = [1.0_f32 / 2.0_f32.sqrt(), 1.0 / 2.0_f32.sqrt()];
        let mut buf = bytes(&unit);
        normalize_f32_inplace(&mut buf);
        let out = floats(&buf);
        for (a, b) in unit.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "drift: {a} -> {b}");
        }
    }

    #[test]
    fn maybe_normalize_routes_only_cosine_f32() {
        let opts = VectorOptions::new(3, Metric::Cosine);
        let mut buf = bytes(&[3.0_f32, 0.0, 4.0]);
        opts.maybe_normalize_bytes(&mut buf);
        let out = floats(&buf);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (n - 1.0).abs() < 1e-6,
            "Cosine+F32 should normalize, norm={n}"
        );
    }

    #[test]
    fn maybe_normalize_is_noop_for_l2() {
        let opts = VectorOptions::new(3, Metric::L2);
        let input = [3.0_f32, 0.0, 4.0];
        let mut buf = bytes(&input);
        opts.maybe_normalize_bytes(&mut buf);
        assert_eq!(
            floats(&buf),
            input.to_vec(),
            "L2 must not mutate stored rows"
        );
    }

    #[test]
    fn maybe_normalize_is_noop_for_dot() {
        let opts = VectorOptions::new(3, Metric::Dot);
        let input = [3.0_f32, 0.0, 4.0];
        let mut buf = bytes(&input);
        opts.maybe_normalize_bytes(&mut buf);
        assert_eq!(
            floats(&buf),
            input.to_vec(),
            "Dot must not mutate stored rows"
        );
    }
}
