//! Distance kernels for the flat vector index.
//!
//! Each kernel uses a chunked-scalar accumulator pattern: an array of
//! `LANES` independent f32 accumulators with no loop-carried dependency.
//! LLVM's autovectorizer turns this into AVX2 / AVX-512 / NEON FMA loops
//! on the platforms that support them, without any explicit `std::arch`
//! intrinsics.
//!
//! `LANES = 16` matches the f32 width of an AVX-512 register
//! (`512 bits / sizeof::<f32>() == 16`), and is a multiple of the AVX2
//! (8) and NEON (4) widths.
//!
//! Kernels are generic over [`VectorElement`]. For `f32` the arithmetic
//! methods compile to plain `fsub` / `fmul` / `fadd`; quantized dtypes
//! plug in their own decode + arithmetic via the trait.

use std::cmp::Ordering;

use crate::schema::{Metric, VectorDType, VectorOptions};
use crate::vector::{Accumulator, VectorElement};

/// A "higher is better" similarity score — the one ranking convention of the
/// whole vector module.
///
/// The raw kernels below ([`l2_squared`], [`dot`], …) return bare `f32`s in
/// whatever space the math lives in; [`Metric::similarity`] is the boundary
/// that folds them all into similarity space (negating L2) and wraps the
/// result. Downstream code — edge ordering, beam search, RNG occlusion —
/// compares `Similarity` values directly and never re-negates, so a distance
/// can't be confused for a similarity without an explicit
/// [`Similarity::new`].
///
/// Ordered totally via [`f32::total_cmp`], so it works directly in heaps and
/// sorts: greater means more similar, and "best-first" always means
/// descending `Similarity`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Similarity(f32);

impl Similarity {
    /// Less similar than any real score (`-∞`); the empty-slot sentinel.
    pub const WORST: Similarity = Similarity(f32::NEG_INFINITY);

    /// More similar than any real score (`+inf`); the saturation value for
    /// a bound that cannot be computed. See [`Metric::best_possible`].
    pub const BEST: Similarity = Similarity(f32::INFINITY);

    /// Wraps a raw score that is *already* in similarity space (higher is
    /// better). Callers converting from a distance must negate first — that
    /// negation is exactly what this type exists to make explicit.
    #[inline]
    pub fn new(score: f32) -> Self {
        Similarity(score)
    }

    /// The raw score, e.g. to hand off as a document [`Score`](crate::Score).
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

/// 16 = 512 (avx512 register width) / 32 (sizeof::<f32>() in bits).
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

/// `norm_squared` with wide per-lane accumulation: elements widen to
/// [`VectorElement::Acc`] *before* squaring ([`VectorElement::mul_wide`]),
/// so no finite input can overflow the narrow element type — for f32,
/// any sum of finite squares is finite in f64.
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

/// Cosine similarity: `dot(a, b) / (||a|| * ||b||)`. Returns 0.0 if either
/// vector has zero norm — avoids NaN propagating into top-K heaps.
#[inline]
pub fn cosine<T: VectorElement>(a: &[T], b: &[T]) -> f32 {
    let na = norm_squared(a).sqrt();
    let nb = norm_squared(b).sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

// =====================================================================
// Byte-input variants: avoid materializing the doc-side vector into an
// intermediate scratch buffer. The doc bytes are decoded inline inside
// the chunked accumulator, so the segment scan touches each byte
// exactly once regardless of directory backend (mmap, RAM, or custom).
// =====================================================================

/// `l2_squared` where the doc side is little-endian bytes encoding `T`.
///
/// Uses `chunks_exact` so LLVM sees fixed-length inner slices and can
/// elide bounds checks + autovectorize the chunked accumulator into
/// 4-wide NEON / 8-wide AVX2 / 16-wide AVX-512 SIMD.
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

/// `norm_squared` over little-endian bytes encoding `T`.
#[inline]
pub fn norm_squared_bytes<T: VectorElement>(doc_bytes: &[u8]) -> f32 {
    norm_squared_bytes_wide::<T>(doc_bytes) as f32
}

/// [`l2_squared_bytes`] with the accumulation widened to
/// [`VectorElement::Acc`]. Squaring happens before the caller's `sqrt`, so
/// an f32 accumulator saturates at a coordinate magnitude whose DISTANCE
/// would still fit in f32 comfortably; accumulating wide moves the limit
/// out to the accumulator's range. Off the query-scoring hot path on
/// purpose - see [`l2_squared_bytes`], which stays narrow.
#[inline]
pub(crate) fn l2_squared_bytes_wide<T: VectorElement>(query: &[T], doc_bytes: &[u8]) -> f64 {
    debug_assert_eq!(doc_bytes.len(), query.len() * T::SIZE_BYTES);
    let q_chunks = query.chunks_exact(LANES);
    let b_chunks = doc_bytes.chunks_exact(LANES * T::SIZE_BYTES);
    let q_tail = q_chunks.remainder();
    let b_tail = b_chunks.remainder();

    let mut sums = [T::Acc::ZERO; LANES];
    for (qc, bc) in q_chunks.zip(b_chunks) {
        for i in 0..LANES {
            let v = T::decode_le(&bc[i * T::SIZE_BYTES..(i + 1) * T::SIZE_BYTES]);
            sums[i] = sums[i].add(T::squared_diff_wide(qc[i], v));
        }
    }
    let mut acc = T::Acc::ZERO;
    for s in sums {
        acc = acc.add(s);
    }
    for (i, &q) in q_tail.iter().enumerate() {
        let v = T::decode_le(&b_tail[i * T::SIZE_BYTES..(i + 1) * T::SIZE_BYTES]);
        acc = acc.add(T::squared_diff_wide(q, v));
    }
    acc.to_f64()
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

/// A cluster's native radius: the greatest displacement of any native
/// member from its centroid, in the stored representation's L2 space.
///
/// INVARIANT: non-negative, and either finite or [`SATURATED`](Self::SATURATED).
/// Both halves matter. A negative radius would make the ball's near surface
/// `d_c + |r|` instead of `d_c - r`, pushing the bound the WRONG way and
/// sorting the cluster behind where it belongs; the type makes that
/// unrepresentable rather than guarded against.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Radius(f32);

impl Radius {
    /// Every native member sits exactly on the centroid. A real value, not
    /// a disabled bound: it makes the bound exact.
    pub const ZERO: Radius = Radius(0.0);

    /// The cluster could not be bounded - a centroid that is not a usable
    /// origin, or a member whose displacement is not finite. Saturating
    /// rather than flagging keeps it out of every branch: the bound it
    /// produces is the metric's maximum, so the cluster always qualifies
    /// and can never be the basis of a termination proof.
    pub const SATURATED: Radius = Radius(f32::INFINITY);

    /// The single validating boundary, for values read off disk.
    ///
    /// FAIL OPEN: anything that is not a trustworthy non-negative
    /// magnitude - negative, or NaN - becomes [`SATURATED`](Self::SATURATED)
    /// rather than an error or a clamp. Same reasoning as a gate resolving
    /// ambiguity to probe: a bound we cannot trust must never be allowed to
    /// exclude anything.
    #[inline]
    pub fn from_stored(raw: f32) -> Radius {
        if raw >= 0.0 {
            Radius(raw)
        } else {
            // Catches negatives AND NaN, since every NaN comparison is
            // false.
            Radius::SATURATED
        }
    }

    /// Whether the ball is a point, which is what makes the bound exact.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    /// The magnitude, for storage and for the telemetry/SRF path.
    #[inline]
    pub fn get(self) -> f32 {
        self.0
    }
}

/// Outcome of an in-place normalization attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormalizeOutcome {
    /// Row rescaled to unit norm.
    Normalized,
    /// Zero vector: normalization is undefined but the data is honest;
    /// row left unchanged. `dot(q, 0) = 0`, so it scores 0 everywhere.
    ZeroSkipped,
    /// With wide accumulation, a non-finite norm occurs IF AND ONLY IF
    /// the input contains a NaN or ±inf element (finite f32 elements
    /// sum to at most ~1.8e80, always finite in f64). Row left
    /// unchanged; the caller decides policy.
    NonFinite,
}

/// L2-normalize a little-endian `T` row in place, with all internal
/// arithmetic in f64 (see [`norm_squared_bytes_wide`]). Elements narrow
/// back to `T` only at write-back, where every value is `<= 1`.
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

/// L2-normalize `row` in place if `opts` requires write-time
/// unit-normalization (see [`VectorOptions::needs_normalization`]).
///
/// Pre-normalizing at write time lets
/// [`PreparedQuery::score_doc_bytes`](crate::vector::PreparedQuery::score_doc_bytes)
/// reduce per-doc cosine work to `dot * inv_norm_q` — no per-doc
/// `norm_squared_bytes` pass.
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

/// A metric that HAS a distance space: the similarity is a monotone
/// transform of a distance, so a bound can be taken there and brought
/// back. `Dot` deliberately has no entry - an inner product is not a
/// distance, and a uniform conversion that lies for one metric is exactly
/// how a sign error gets in.
#[derive(Clone, Copy)]
enum BallSpace {
    /// `sim = -d^2`.
    NegativeSquared,
    /// `sim = 1 - d^2/2`, the chord of a unit-norm pair.
    Chord,
}

impl BallSpace {
    #[inline]
    fn to_distance(self, sim: Similarity) -> f32 {
        match self {
            BallSpace::NegativeSquared => (-sim.score()).max(0.0).sqrt(),
            BallSpace::Chord => (2.0 * (1.0 - sim.score())).max(0.0).sqrt(),
        }
    }

    #[inline]
    fn to_similarity(self, distance: f32) -> Similarity {
        match self {
            BallSpace::NegativeSquared => Similarity(-(distance * distance)),
            BallSpace::Chord => Similarity(1.0 - distance * distance / 2.0),
        }
    }
}

impl Metric {
    /// The distance space this metric's similarities live in, if it has
    /// one.
    #[inline]
    fn ball_space(self) -> Option<BallSpace> {
        match self {
            Metric::L2 => Some(BallSpace::NegativeSquared),
            Metric::Cosine => Some(BallSpace::Chord),
            Metric::Dot => None,
        }
    }

    /// The best score any point inside the cluster's ball could achieve:
    /// the centroid's similarity relaxed by the cluster's radius.
    ///
    /// NAMING: higher is better, so this is an UPPER bound on achievable
    /// similarity. It is never a "lower bound" - the conversion passes
    /// through distance space, where the sign inverts.
    ///
    /// Two exactness rules, both early returns rather than arithmetic
    /// coincidences:
    ///
    /// * A [`SATURATED`](Radius::SATURATED) radius, or a non-finite `centroid` or `query_norm`,
    ///   yields [`Similarity::BEST`]. A bound that cannot be computed must never exclude anything.
    /// * A [`ZERO`](Radius::ZERO) radius returns `centroid` UNCHANGED. The ball is a point, so its
    ///   best member is the centroid itself. Round-tripping through `sqrt` and back would not be
    ///   bit-exact for most values, and this key is a sort order.
    #[inline]
    pub fn best_possible(
        self,
        centroid: Similarity,
        radius: Radius,
        query_norm: f32,
    ) -> Similarity {
        if radius == Radius::SATURATED || !centroid.score().is_finite() || !query_norm.is_finite() {
            return Similarity::BEST;
        }
        if radius.is_zero() {
            return centroid;
        }
        match self.ball_space() {
            Some(space) => {
                let surface = (space.to_distance(centroid) - radius.get()).max(0.0);
                space.to_similarity(surface)
            }
            // Cauchy-Schwarz: `<q, p> <= <q, mu> + ||q|| * r`.
            None => Similarity(centroid.score() + query_norm * radius.get()),
        }
    }

    /// Compute the [`Similarity`] of two vectors.
    ///
    /// L2 distance is negated (squared, then sign-flipped) here, and only
    /// here, so all metrics share the same "higher is better" convention.
    /// Magnitude differences across metrics are the caller's problem.
    #[inline]
    pub fn similarity<T: VectorElement>(self, query: &[T], doc: &[T]) -> Similarity {
        Similarity(match self {
            Metric::L2 => -l2_squared(query, doc),
            Metric::Cosine => cosine(query, doc),
            Metric::Dot => dot(query, doc),
        })
    }

    /// Like [`similarity`](Self::similarity), but the doc side is
    /// little-endian bytes — typically a borrowed slice straight out
    /// of the segment's file.
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

    /// Hand-computed `best_possible` per metric: the surface is
    /// `max(0, d_c - r)`, converted back to similarity space.
    #[test]
    fn best_possible_hand_computed_per_metric() {
        // L2: sim = -d^2. sim = -25 -> d_c = 5. r = 2 -> surface 3 -> -9.
        assert_eq!(
            Metric::L2.best_possible(Similarity::new(-25.0), Radius::from_stored(2.0), 0.0),
            Similarity::new(-9.0)
        );
        // Cosine: chord d_c = sqrt(2*(1 - s)). s = 0.5 -> d_c = 1.
        // r = 0.25 -> surface 0.75 -> 1 - 0.75^2/2 = 0.71875.
        assert_eq!(
            Metric::Cosine.best_possible(Similarity::new(0.5), Radius::from_stored(0.25), 0.0),
            Similarity::new(0.71875)
        );
        // Dot: Cauchy-Schwarz, sim + ||q||*r = 10 + 2*3 = 16.
        assert_eq!(
            Metric::Dot.best_possible(Similarity::new(10.0), Radius::from_stored(3.0), 2.0),
            Similarity::new(16.0)
        );
    }

    /// A zero radius returns the centroid similarity UNCHANGED, bit for
    /// bit. The ball is a point, so its best member is the centroid.
    ///
    /// The values here are chosen to DRIFT through the conversion, which
    /// is what makes this test load-bearing: `sqrt` then square fails to
    /// round-trip for about half of L2 similarities and two thirds of
    /// cosine ones. `-25` and `0.5` are perfect squares and would pass
    /// either way. Delete the `is_zero` early return and this fails:
    /// L2 `-0.2` comes back `-0.19999999`, cosine `0.3` comes back
    /// `0.29999995`.
    #[test]
    fn zero_radius_is_the_centroid_similarity() {
        for (metric, sim, query_norm) in [
            (Metric::L2, Similarity::new(-0.2), 0.0),
            (Metric::Cosine, Similarity::new(0.3), 0.0),
            (Metric::Dot, Similarity::new(10.0), 2.0),
        ] {
            assert_eq!(
                metric.best_possible(sim, Radius::ZERO, query_norm),
                sim,
                "{metric:?}: a zero radius must return the centroid exactly"
            );
        }
    }

    /// A radius that reaches past the query clamps the surface to zero, so
    /// the key saturates at the metric's maximum. Every such cluster ties
    /// there; the ranking breaks those ties by node id rather than
    /// inventing an ordering the geometry does not have.
    #[test]
    fn radius_past_the_query_clamps_to_the_metric_maximum() {
        // L2: d_c = 5, r = 7 -> surface 0 -> 0.0, the largest L2 score.
        assert_eq!(
            Metric::L2.best_possible(Similarity::new(-25.0), Radius::from_stored(7.0), 0.0),
            Similarity::new(0.0)
        );
        // Cosine: d_c = 1, r = 2 -> surface 0 -> 1.0, a perfect cosine.
        assert_eq!(
            Metric::Cosine.best_possible(Similarity::new(0.5), Radius::from_stored(2.0), 0.0),
            Similarity::new(1.0)
        );
        // Two clusters at different distances both clamp: they tie.
        assert_eq!(
            Metric::L2.best_possible(Similarity::new(-100.0), Radius::from_stored(30.0), 0.0),
            Metric::L2.best_possible(Similarity::new(-25.0), Radius::from_stored(7.0), 0.0),
        );
    }

    /// A radius we could not compute saturates the key on EVERY metric, so
    /// the cluster sorts ahead of every bounded one and can never be the
    /// basis of a termination proof. Dot needs the guard rather than the
    /// arithmetic: `0.0 * inf` is NaN, which would sort as garbage.
    #[test]
    fn saturated_radius_yields_the_best_possible_similarity() {
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            assert_eq!(
                metric.best_possible(Similarity::new(-25.0), Radius::SATURATED, 0.0),
                Similarity::BEST,
                "{metric:?}: an unbounded cluster must sort first"
            );
            assert_eq!(
                metric.best_possible(Similarity::new(-25.0), Radius::SATURATED, 1.0),
                Similarity::BEST,
                "{metric:?}: with a non-zero query norm too"
            );
            // A negative radius on disk arrives here as SATURATED, so the
            // "surface = d_c + |r|" inversion is unrepresentable.
            assert_eq!(
                metric.best_possible(Similarity::new(-25.0), Radius::from_stored(-3.0), 1.0),
                Similarity::BEST,
                "{metric:?}: a negative stored radius fails open"
            );
            assert_eq!(
                metric.best_possible(Similarity::new(f32::NAN), Radius::from_stored(1.0), 1.0),
                Similarity::BEST,
                "{metric:?}: an uncomputable bound saturates"
            );
        }
        assert!(Similarity::BEST > Similarity::new(f32::MAX));
    }

    /// `from_stored` is the only way a radius enters from disk, and it
    /// FAILS OPEN: anything that is not a trustworthy non-negative
    /// magnitude becomes SATURATED rather than being clamped or trusted.
    /// A negative radius would otherwise put the ball's near surface at
    /// `d_c + |r|` - the wrong direction - and sort the cluster behind
    /// where it belongs.
    #[test]
    fn from_stored_fails_open_on_untrustworthy_values() {
        assert_eq!(Radius::from_stored(-0.0), Radius::ZERO, "-0.0 is zero");
        for bad in [-1.0f32, -1e-30, f32::NAN, f32::NEG_INFINITY, -f32::MAX] {
            assert_eq!(
                Radius::from_stored(bad),
                Radius::SATURATED,
                "{bad} must not be trusted as a magnitude"
            );
        }
        for good in [0.0f32, 1e-30, 2.5, f32::MAX] {
            assert_eq!(
                Radius::from_stored(good).get(),
                good,
                "{good} passes through"
            );
        }
        assert_eq!(Radius::from_stored(f32::INFINITY), Radius::SATURATED);
        assert!(Radius::ZERO.is_zero());
        assert!(!Radius::SATURATED.is_zero());
        assert!(!Radius::from_stored(1e-30).is_zero());
    }
    use super::*;
    use crate::schema::{Metric, VectorOptions};

    #[test]
    fn test_l2_squared() {
        let a: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let b: [f32; 4] = [4.0, 3.0, 2.0, 1.0];
        // (1-4)^2 + (2-3)^2 + (3-2)^2 + (4-1)^2 = 9+1+1+9 = 20
        assert!((l2_squared(&a, &b) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot() {
        let a: [f32; 3] = [1.0, 2.0, 3.0];
        let b: [f32; 3] = [4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
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
        // Random-ish data that exercises both the chunked path and the tail.
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
        // Exercise both the chunked path and the tail.
        let dim = LANES * 2 + 3;
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i + 1) as f32 * 0.1).collect();
        // L2² = sum (-0.1)^2 = dim * 0.01
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
        // Direction preserved (dot with input ⇒ original L2 norm).
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
        // 1e20² = 1e40 overflows f32 (max ~3.4e38); the old narrow kernel
        // returned inf and the guard left the row raw.
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
        // Direction preserved: dominated by the first component.
        assert!((out[0] - 1.0).abs() < 1e-3, "out={out:?}");
        assert!(out[1] >= 0.0, "out={out:?}");
    }

    #[test]
    fn wide_accumulation_sum_across_dims() {
        // Each square (2.5e35) is a finite f32, but 1536 of them sum to
        // ~3.8e38 > f32::MAX — the old fold overflowed across dimensions.
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
        // Norm ≈ 5.2e38 exceeds f32::MAX: the f64 inv multiply is load-
        // bearing all the way to the final narrowing.
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
        // The wide-backed wrappers must agree with plain f32 arithmetic
        // on ordinary data.
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
