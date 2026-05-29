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

impl Metric {
    /// Compute a "higher is better" similarity score between two vectors.
    ///
    /// L2 distance is negated (squared, then sign-flipped) so all metrics
    /// share the same ranking convention. Magnitude differences across
    /// metrics are the caller's problem.
    #[inline]
    pub fn similarity<T: VectorElement>(self, query: &[T], doc: &[T]) -> f32 {
        use crate::vector::distance::{cosine, dot, l2_squared};
        match self {
            Metric::L2 => -l2_squared(query, doc),
            Metric::Cosine => cosine(query, doc),
            Metric::Dot => dot(query, doc),
        }
    }

    /// Like [`similarity`](Self::similarity), but the doc side is
    /// little-endian bytes — typically a borrowed slice straight out
    /// of the segment's file.
    #[inline]
    pub fn similarity_bytes<T: VectorElement>(self, query: &[T], doc_bytes: &[u8]) -> f32 {
        use crate::vector::distance::{cosine_bytes, dot_bytes, l2_squared_bytes};
        match self {
            Metric::L2 => -l2_squared_bytes(query, doc_bytes),
            Metric::Cosine => cosine_bytes(query, doc_bytes),
            Metric::Dot => dot_bytes(query, doc_bytes),
        }
    }
}

/// A vector element type with the primitives needed by the storage
/// layer and the distance kernels.
///
/// Implemented for the element types supported by [`VectorDType`]. The
/// `DTYPE` associated constant lets callers reject mismatches between
/// the declared schema dtype and the type passed at runtime. The
/// arithmetic methods (`squared_diff`, `product`) return `f32` so that
/// kernels can use a uniform accumulator type across dtypes.
pub trait VectorElement: Copy + Send + Sync + 'static {
    const DTYPE: VectorDType;
    const SIZE_BYTES: usize;

    fn encode_le(&self, buf: &mut Vec<u8>);

    /// Decode one element from its little-endian byte representation.
    /// `bytes.len()` must be `SIZE_BYTES`.
    fn decode_le(bytes: &[u8]) -> Self;

    /// `(a - b)^2` promoted to `f32` for accumulator-friendly distance
    /// computation. For `f32` this is the obvious arithmetic; for
    /// quantized types it may promote through a wider integer first.
    fn squared_diff(a: Self, b: Self) -> f32;

    /// `a * b` promoted to `f32`. Same rationale as `squared_diff`.
    fn product(a: Self, b: Self) -> f32;
}

impl VectorElement for f32 {
    const DTYPE: VectorDType = VectorDType::F32;
    const SIZE_BYTES: usize = 4;

    #[inline(always)]
    fn encode_le(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }

    #[inline(always)]
    fn decode_le(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[inline(always)]
    fn squared_diff(a: Self, b: Self) -> f32 {
        let d = a - b;
        d * d
    }

    #[inline(always)]
    fn product(a: Self, b: Self) -> f32 {
        a * b
    }
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
    /// [`PreparedQuery::score_doc_bytes`](super::prepared::PreparedQuery::score_doc_bytes)
    /// reduce per-doc cosine work to `dot * inv_norm_q` — no per-doc
    /// `norm_squared_bytes` pass.
    pub fn maybe_normalize_bytes(&self, row: &mut [u8]) {
        debug_assert_eq!(row.len(), self.bytes_per_vector());
        match (self.metric, self.dtype) {
            (Metric::Cosine, VectorDType::F32) => normalize_f32_inplace(row),
            _ => {}
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
    use super::*;

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
