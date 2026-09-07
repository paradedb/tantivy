//! Vector distance and similarity kernels.

use std::cmp::Ordering;

use crate::schema::{Metric, VectorDType, VectorOptions};
use crate::vector::{Accumulator, VectorElement};

/// Totally ordered higher-is-better similarity score.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Similarity(f32);

impl Similarity {
    /// Similarity below every finite score.
    pub const WORST: Similarity = Similarity(f32::NEG_INFINITY);

    /// Wraps a higher-is-better score.
    #[inline]
    pub fn new(score: f32) -> Self {
        Similarity(score)
    }

    /// Returns the raw score.
    #[inline]
    pub fn score(self) -> f32 {
        self.0
    }
}

impl Eq for Similarity {}

impl Ord for Similarity {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for Similarity {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Number of independent reduction accumulators.
const LANES: usize = 16;

/// Squared Euclidean distance.
#[inline]
pub fn l2_squared<T: VectorElement>(a: &[T], b: &[T]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let a_chunks = a.chunks_exact(LANES);
    let b_chunks = b.chunks_exact(LANES);
    let a_tail = a_chunks.remainder();
    let b_tail = b_chunks.remainder();

    let mut sums = [0f32; LANES];
    for (ac, bc) in a_chunks.zip(b_chunks) {
        for i in 0..LANES {
            sums[i] += T::squared_diff(ac[i], bc[i]);
        }
    }
    let mut acc: f32 = sums.iter().sum();
    for (&x, &y) in a_tail.iter().zip(b_tail.iter()) {
        acc += T::squared_diff(x, y);
    }
    acc
}

/// Dot product.
#[inline]
pub fn dot<T: VectorElement>(a: &[T], b: &[T]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let a_chunks = a.chunks_exact(LANES);
    let b_chunks = b.chunks_exact(LANES);
    let a_tail = a_chunks.remainder();
    let b_tail = b_chunks.remainder();

    let mut sums = [0f32; LANES];
    for (ac, bc) in a_chunks.zip(b_chunks) {
        for i in 0..LANES {
            sums[i] += T::product(ac[i], bc[i]);
        }
    }
    let mut acc: f32 = sums.iter().sum();
    for (&x, &y) in a_tail.iter().zip(b_tail.iter()) {
        acc += T::product(x, y);
    }
    acc
}

/// Sum of squares (squared L2 norm).
#[inline]
pub fn norm_squared<T: VectorElement>(a: &[T]) -> f32 {
    norm_squared_wide(a) as f32
}

/// Sum of squares using widened accumulation.
#[inline]
pub(crate) fn norm_squared_wide<T: VectorElement>(a: &[T]) -> f64 {
    let chunks = a.chunks_exact(LANES);
    let tail = chunks.remainder();

    let mut sums = [T::Acc::ZERO; LANES];
    for c in chunks {
        for i in 0..LANES {
            sums[i] = sums[i].add(T::mul_wide(c[i], c[i]));
        }
    }
    let mut acc = T::Acc::ZERO;
    for s in sums {
        acc = acc.add(s);
    }
    for &x in tail {
        acc = acc.add(T::mul_wide(x, x));
    }
    acc.to_f64()
}

/// Cosine similarity, or zero when either vector has zero norm.
#[inline]
pub fn cosine<T: VectorElement>(a: &[T], b: &[T]) -> f32 {
    let na = norm_squared(a).sqrt();
    let nb = norm_squared(b).sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// `l2_squared` where the doc side is little-endian bytes encoding `T`.
#[inline]
pub fn l2_squared_bytes<T: VectorElement>(query: &[T], doc_bytes: &[u8]) -> f32 {
    debug_assert_eq!(doc_bytes.len(), query.len() * T::SIZE_BYTES);
    let q_chunks = query.chunks_exact(LANES);
    let b_chunks = doc_bytes.chunks_exact(LANES * T::SIZE_BYTES);
    let q_tail = q_chunks.remainder();
    let b_tail = b_chunks.remainder();

    let mut sums = [0f32; LANES];
    for (qc, bc) in q_chunks.zip(b_chunks) {
        for i in 0..LANES {
            let v = T::decode_le(&bc[i * T::SIZE_BYTES..(i + 1) * T::SIZE_BYTES]);
            sums[i] += T::squared_diff(qc[i], v);
        }
    }
    let mut acc: f32 = sums.iter().sum();
    for (i, &q) in q_tail.iter().enumerate() {
        let v = T::decode_le(&b_tail[i * T::SIZE_BYTES..(i + 1) * T::SIZE_BYTES]);
        acc += T::squared_diff(q, v);
    }
    acc
}

/// `dot` where the doc side is little-endian bytes encoding `T`.
#[inline]
pub fn dot_bytes<T: VectorElement>(query: &[T], doc_bytes: &[u8]) -> f32 {
    debug_assert_eq!(doc_bytes.len(), query.len() * T::SIZE_BYTES);
    let q_chunks = query.chunks_exact(LANES);
    let b_chunks = doc_bytes.chunks_exact(LANES * T::SIZE_BYTES);
    let q_tail = q_chunks.remainder();
    let b_tail = b_chunks.remainder();

    let mut sums = [0f32; LANES];
    for (qc, bc) in q_chunks.zip(b_chunks) {
        for i in 0..LANES {
            let v = T::decode_le(&bc[i * T::SIZE_BYTES..(i + 1) * T::SIZE_BYTES]);
            sums[i] += T::product(qc[i], v);
        }
    }
    let mut acc: f32 = sums.iter().sum();
    for (i, &q) in q_tail.iter().enumerate() {
        let v = T::decode_le(&b_tail[i * T::SIZE_BYTES..(i + 1) * T::SIZE_BYTES]);
        acc += T::product(q, v);
    }
    acc
}

/// Runs the binary32 byte-backed dot kernel.
#[cfg(feature = "quantization-bench")]
#[doc(hidden)]
#[inline(never)]
pub fn quantization_bench_dot_bytes_f32(query: &[f32], doc_bytes: &[u8]) -> f32 {
    dot_bytes::<f32>(query, doc_bytes)
}

/// `norm_squared` over little-endian bytes encoding `T`.
#[inline]
pub fn norm_squared_bytes<T: VectorElement>(doc_bytes: &[u8]) -> f32 {
    norm_squared_bytes_wide::<T>(doc_bytes) as f32
}

/// [`norm_squared_wide`] over little-endian bytes encoding `T`.
#[inline]
pub(crate) fn norm_squared_bytes_wide<T: VectorElement>(doc_bytes: &[u8]) -> f64 {
    debug_assert_eq!(doc_bytes.len() % T::SIZE_BYTES, 0);
    let b_chunks = doc_bytes.chunks_exact(LANES * T::SIZE_BYTES);
    let b_tail = b_chunks.remainder();

    let mut sums = [T::Acc::ZERO; LANES];
    for bc in b_chunks {
        for i in 0..LANES {
            let v = T::decode_le(&bc[i * T::SIZE_BYTES..(i + 1) * T::SIZE_BYTES]);
            sums[i] = sums[i].add(T::mul_wide(v, v));
        }
    }
    let mut acc = T::Acc::ZERO;
    for s in sums {
        acc = acc.add(s);
    }
    let tail_dim = b_tail.len() / T::SIZE_BYTES;
    for i in 0..tail_dim {
        let v = T::decode_le(&b_tail[i * T::SIZE_BYTES..(i + 1) * T::SIZE_BYTES]);
        acc = acc.add(T::mul_wide(v, v));
    }
    acc.to_f64()
}

/// Outcome of an in-place normalization attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormalizeOutcome {
    /// Row rescaled to unit norm.
    Normalized,
    /// Zero vector left unchanged.
    ZeroSkipped,
    /// Non-finite row left unchanged.
    NonFinite,
}

/// Normalizes a little-endian row in place using widened arithmetic.
pub(crate) fn normalize_bytes_inplace<T: VectorElement>(row: &mut [u8]) -> NormalizeOutcome {
    debug_assert_eq!(row.len() % T::SIZE_BYTES, 0);
    let norm = norm_squared_bytes_wide::<T>(row).sqrt();
    if norm == 0.0 {
        return NormalizeOutcome::ZeroSkipped;
    }
    if !norm.is_finite() {
        return NormalizeOutcome::NonFinite;
    }
    let inv = 1.0 / norm;
    for chunk in row.chunks_exact_mut(T::SIZE_BYTES) {
        let scaled = (T::decode_le(chunk).to_f32() as f64 * inv) as f32;
        let mut sink: &mut [u8] = chunk;
        T::from_f32(scaled)
            .encode_le(&mut sink)
            .expect("row chunk is exactly element-sized");
    }
    NormalizeOutcome::Normalized
}

/// Normalizes a row when required by its vector options.
pub(crate) fn maybe_normalize_bytes(opts: &VectorOptions, row: &mut [u8]) -> NormalizeOutcome {
    debug_assert_eq!(row.len(), opts.bytes_per_vector());
    if !opts.needs_normalization() {
        return NormalizeOutcome::Normalized; // no-op metrics count as fine
    }
    match opts.dtype() {
        VectorDType::F32 => normalize_bytes_inplace::<f32>(row),
    } // exhaustive: a new dtype variant must decide its policy here
}

/// `cosine` where the doc side is little-endian bytes encoding `T`.
#[inline]
pub fn cosine_bytes<T: VectorElement>(query: &[T], doc_bytes: &[u8]) -> f32 {
    let nq = norm_squared(query).sqrt();
    let nd = norm_squared_bytes::<T>(doc_bytes).sqrt();
    if nq == 0.0 || nd == 0.0 {
        return 0.0;
    }
    dot_bytes(query, doc_bytes) / (nq * nd)
}

impl Metric {
    /// Computes vector similarity.
    #[inline]
    pub fn similarity<T: VectorElement>(self, query: &[T], doc: &[T]) -> Similarity {
        Similarity(match self {
            Metric::L2 => -l2_squared(query, doc),
            Metric::Cosine => cosine(query, doc),
            Metric::Dot => dot(query, doc),
        })
    }

    /// Computes similarity with a little-endian document row.
    #[inline]
    pub fn similarity_bytes<T: VectorElement>(self, query: &[T], doc_bytes: &[u8]) -> Similarity {
        Similarity(match self {
            Metric::L2 => -l2_squared_bytes(query, doc_bytes),
            Metric::Cosine => cosine_bytes(query, doc_bytes),
            Metric::Dot => dot_bytes(query, doc_bytes),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Metric, VectorOptions};

    #[test]
    fn test_l2_squared() {
        let a: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let b: [f32; 4] = [4.0, 3.0, 2.0, 1.0];
        assert!((l2_squared(&a, &b) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot() {
        let a: [f32; 3] = [1.0, 2.0, 3.0];
        let b: [f32; 3] = [4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_identical() {
        let a: [f32; 2] = [3.0, 4.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a: [f32; 2] = [1.0, 0.0];
        let b: [f32; 2] = [0.0, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a: [f32; 2] = [1.0, 0.0];
        let b: [f32; 2] = [-1.0, 0.0];
        assert!((cosine(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_zero_norm() {
        let a: [f32; 2] = [0.0, 0.0];
        let b: [f32; 2] = [1.0, 0.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn test_higher_is_better_ranking() {
        let query: [f32; 3] = [1.0, 0.0, 0.0];
        let near: [f32; 3] = [1.0, 0.1, 0.0];
        let far: [f32; 3] = [-1.0, 0.0, 0.0];
        for m in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let s_near = m.similarity(&query, &near);
            let s_far = m.similarity(&query, &far);
            assert!(s_near > s_far, "metric {m:?}: {s_near:?} vs {s_far:?}");
        }
    }

    #[test]
    fn test_byte_kernels_match_f32_kernels() {
        let dim = LANES + 5;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.137 - 1.3).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();

        let eps = 1e-5;
        assert!((l2_squared(&a, &b) - l2_squared_bytes::<f32>(&a, &b_bytes)).abs() < eps);
        assert!((dot(&a, &b) - dot_bytes::<f32>(&a, &b_bytes)).abs() < eps);
        assert!((cosine(&a, &b) - cosine_bytes::<f32>(&a, &b_bytes)).abs() < eps);
        assert!((norm_squared(&b) - norm_squared_bytes::<f32>(&b_bytes)).abs() < eps);
    }

    #[test]
    fn test_long_vector_chunking() {
        let dim = LANES * 2 + 3;
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i + 1) as f32 * 0.1).collect();
        let expected = dim as f32 * 0.01;
        assert!((l2_squared(&a, &b) - expected).abs() < 1e-4);
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
        assert_eq!(
            normalize_bytes_inplace::<f32>(&mut buf),
            NormalizeOutcome::Normalized
        );
        let out = floats(&buf);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6, "norm={n}, out={out:?}");
        let dot = 3.0 * out[0] + 0.0 * out[1] + 4.0 * out[2];
        assert!((dot - 5.0).abs() < 1e-5, "dot={dot}");
    }

    #[test]
    fn normalize_zero_vector_is_unchanged() {
        let mut buf = bytes(&[0.0_f32, 0.0, 0.0]);
        assert_eq!(
            normalize_bytes_inplace::<f32>(&mut buf),
            NormalizeOutcome::ZeroSkipped
        );
        assert_eq!(floats(&buf), vec![0.0_f32, 0.0, 0.0]);
    }

    #[test]
    fn normalize_already_unit_is_idempotent() {
        let unit = [1.0_f32 / 2.0_f32.sqrt(), 1.0 / 2.0_f32.sqrt()];
        let mut buf = bytes(&unit);
        assert_eq!(
            normalize_bytes_inplace::<f32>(&mut buf),
            NormalizeOutcome::Normalized
        );
        let out = floats(&buf);
        for (a, b) in unit.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "drift: {a} -> {b}");
        }
    }

    #[test]
    fn maybe_normalize_routes_only_cosine_f32() {
        let opts = VectorOptions::new(3, Metric::Cosine);
        let mut buf = bytes(&[3.0_f32, 0.0, 4.0]);
        assert_eq!(
            maybe_normalize_bytes(&opts, &mut buf),
            NormalizeOutcome::Normalized
        );
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
        maybe_normalize_bytes(&opts, &mut buf);
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
        maybe_normalize_bytes(&opts, &mut buf);
        assert_eq!(
            floats(&buf),
            input.to_vec(),
            "Dot must not mutate stored rows"
        );
    }

    #[test]
    fn wide_accumulation_single_square() {
        let mut buf = bytes(&[1e20_f32, 1.0]);
        let n2 = norm_squared_bytes_wide::<f32>(&buf);
        assert!(n2.is_finite(), "n2={n2}");
        assert!((n2 - 1e40).abs() / 1e40 < 1e-3, "n2={n2}");
        assert_eq!(
            normalize_bytes_inplace::<f32>(&mut buf),
            NormalizeOutcome::Normalized
        );
        let out = floats(&buf);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-3, "norm={n}, out={out:?}");
        assert!((out[0] - 1.0).abs() < 1e-3, "out={out:?}");
        assert!(out[1] >= 0.0, "out={out:?}");
    }

    #[test]
    fn wide_accumulation_sum_across_dims() {
        let v = vec![5e17_f32; 1536];
        let mut buf = bytes(&v);
        assert!(norm_squared_bytes_wide::<f32>(&buf).is_finite());
        assert_eq!(
            normalize_bytes_inplace::<f32>(&mut buf),
            NormalizeOutcome::Normalized
        );
        let out = floats(&buf);
        let n: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-3, "norm={n}");
    }

    #[test]
    fn normalize_extreme_norm_exceeding_f32_max() {
        let mut buf = bytes(&[3e38_f32, 3e38, 3e38]);
        assert_eq!(
            normalize_bytes_inplace::<f32>(&mut buf),
            NormalizeOutcome::Normalized
        );
        let out = floats(&buf);
        let n: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-3, "norm={n}");
        let expected = 1.0 / 3.0_f32.sqrt();
        for v in &out {
            assert!((v - expected).abs() < 1e-3, "out={out:?}");
        }
    }

    #[test]
    fn normalize_classifies_nan_as_non_finite() {
        let original = bytes(&[1.0_f32, f32::NAN, 2.0]);
        let mut buf = original.clone();
        assert_eq!(
            normalize_bytes_inplace::<f32>(&mut buf),
            NormalizeOutcome::NonFinite
        );
        assert_eq!(buf, original, "row must be left byte-identical");
    }

    #[test]
    fn norm_squared_consistency() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![3.0, 4.0],
            vec![0.1; 33],
            (0..40).map(|i| (i as f32).sin()).collect(),
            vec![-2.5, 7.25, 0.0, 1e-3],
        ];
        for v in vectors {
            let expected: f32 = v.iter().map(|x| x * x).sum();
            let typed = norm_squared(&v);
            let from_bytes = norm_squared_bytes::<f32>(&bytes(&v));
            for got in [typed, from_bytes] {
                assert!(
                    (got - expected).abs() <= expected.abs() * 1e-6,
                    "got={got}, expected={expected}, v={v:?}"
                );
            }
        }
    }
}
