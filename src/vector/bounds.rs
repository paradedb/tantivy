//! Centroid bounds: the stored cluster-side extents the probe gate
//! certifies skips against.
//!
//! Model: cluster = centroid + stored shape (grows: Ball today, Aabb/...
//! later). Query = position in ctx + the metric's sublevel set at the kth
//! threshold — fully determined by (metric, t), never stored, never
//! shaped. Collision matrix is therefore 2 x K:
//!
//! ```text
//!                 | Ball(r)              | Aabb(h[dim])   (future)
//!   BallRegion    | margin_ball_ball     | margin_ball_box
//!   Halfspace     | margin_ball_hspace   | margin_box_hspace
//! ```
//!
//! Ball-column margins are scalar-only (separation = the routing key,
//! already computed). Box-column margins need the residual / query vector
//! — one pass over dims per check. That is the price of AABB; it is not
//! paid today.
//!
//! This module owns the semantic types; the `.centroids` slot byte format
//! lives with the other slot serializers in [`ivf::index`](super::ivf).

// ======================================================================
// P1: storage
// ======================================================================

use std::io;

use crate::schema::Metric;
use crate::vector::VectorElement;

/// Segment-level bound shape, captured in the `.centroids` V2 bounds slot
/// at build time — one kind for every cluster of a field's segment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BoundKind {
    /// One `f32` per cluster: max `||x - c||` over members, in the
    /// stored representation. Metric-uniform by the cosine
    /// write-normalization invariant (unit members + renormalized
    /// centroid make the residual norm the chord).
    /// `f32::INFINITY` = SATURATED: every margin comes out +inf, the
    /// cluster probes. Fail-open is arithmetic.
    ///
    /// Quantized elements (i8) are a rule, not code, until the dtype
    /// exists: a bound folded over quantized rows must inflate by the
    /// quantization error at write, or a true member can sit outside it.
    Ball = 0,
    // Aabb = 1,   // dim f32 half-extents about the centroid
}

impl BoundKind {
    /// Decodes the on-disk kind byte.
    ///
    /// * `code` (`u8`) — the kind byte leading a bounds slot.
    ///
    /// Returns (`io::Result<BoundKind>`): the kind, or `InvalidData` for a
    /// byte no known kind uses — a corrupt or future-format slot.
    pub fn from_code(code: u8) -> io::Result<BoundKind> {
        match code {
            0 => Ok(BoundKind::Ball),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown centroid bound kind: {other}"),
            )),
        }
    }

    /// Per-cluster payload stride of this kind, in `f32`s.
    ///
    /// * `_dim` (`usize`) — the field's vector dimensionality; unused by `Ball`, consumed by the
    ///   future per-dimension kinds.
    ///
    /// Returns (`usize`): how many `f32`s one cluster's bound occupies.
    pub fn stride(self, _dim: usize) -> usize {
        match self {
            BoundKind::Ball => 1,
            // BoundKind::Aabb => dim,
        }
    }
}

/// Read view over the bounds payload of one segment's `.centroids` field
/// slot.
pub struct BoundStore<'a> {
    kind: BoundKind,
    data: &'a [f32],
}

impl<'a> BoundStore<'a> {
    /// Wraps a decoded bounds payload.
    ///
    /// * `kind` (`BoundKind`) — the slot's segment-level kind byte.
    /// * `data` (`&[f32]`) — the payload, `num_clusters * kind.stride(dim)` values in cluster
    ///   order.
    ///
    /// Returns (`BoundStore`): the view; per-kind accessors validate the
    /// kind on read.
    pub fn new(kind: BoundKind, data: &'a [f32]) -> Self {
        Self { kind, data }
    }

    /// The segment-level bound kind of this store.
    ///
    /// Returns (`BoundKind`): the kind every cluster in the store uses.
    pub fn kind(&self) -> BoundKind {
        self.kind
    }

    /// The ball bound of `cluster`.
    ///
    /// * `cluster` (`usize`) — dense cluster id, `0..num_clusters`.
    ///
    /// Returns (`f32`): max native-member residual norm against the stored
    /// centroid; `f32::INFINITY` = SATURATED (always probes).
    #[inline]
    pub fn ball_r(&self, cluster: usize) -> f32 {
        debug_assert_eq!(self.kind, BoundKind::Ball);
        self.data[cluster]
    }

    /// The raw payload, for serialization and tests.
    ///
    /// Returns (`&[f32]`): the values [`Self::new`] was built over.
    pub fn values(&self) -> &'a [f32] {
        self.data
    }
}

// ---- build ------------------------------------------------------------
//
// The ONLY producer of bounds. The merge path runs this same fold over its
// re-assignment output against the NEW centroids; no bound-combining API
// exists, by design -- folded input radii under merge are unsound.

/// Accumulates one segment's per-cluster ball bounds during a merge.
pub struct BoundsBuilder {
    r: Vec<f32>,
}

impl BoundsBuilder {
    /// Starts a fold with every cluster's bound at `0.0` — exact for
    /// clusters that end up empty or whose members all sit on the
    /// centroid.
    ///
    /// * `num_clusters` (`usize`) — cluster count of the segment being written.
    ///
    /// Returns (`BoundsBuilder`): the zeroed fold state.
    pub fn new(num_clusters: usize) -> Self {
        Self {
            r: vec![0.0; num_clusters],
        }
    }

    /// Folds one native member assigned to `cluster`.
    ///
    /// * `cluster` (`usize`) — the member's HOME (primary) cluster.
    /// * `residual_norm` (`f32`) — `||x - c||` of the member's STORED row against the STORED
    ///   centroid (post-renormalization for cosine). A non-finite residual saturates the cluster.
    pub fn add_native(&mut self, cluster: usize, residual_norm: f32) {
        let slot = &mut self.r[cluster];
        if !residual_norm.is_finite() {
            *slot = f32::INFINITY; // SATURATED
        } else if residual_norm > *slot {
            *slot = residual_norm;
        }
    }

    /// Marks `cluster` SATURATED — its centroid is degenerate (non-finite,
    /// or zero-norm under cosine renormalization), so no residual against
    /// it can certify a skip.
    ///
    /// * `cluster` (`usize`) — the cluster to saturate.
    pub fn saturate(&mut self, cluster: usize) {
        self.r[cluster] = f32::INFINITY;
    }

    /// Finishes the fold.
    ///
    /// Returns (`Vec<f32>`): the per-cluster ball bounds in cluster order —
    /// the [`BoundKind::Ball`] slot payload.
    pub fn finish(self) -> Vec<f32> {
        self.r
    }
}

/// `||x - c||` of one stored row against its stored centroid — the
/// residual the write fold hands to [`BoundsBuilder::add_native`]. One
/// kernel for producer and verifier, so a recomputed fold is bit-equal.
///
/// * `row_bytes` (`&[u8]`) — the member's STORED row, little-endian `f32`s exactly as written
///   (post-renormalization for cosine).
/// * `c` (`&[f32]`) — the STORED centroid values; `row_bytes.len() == 4 * c.len()`.
///
/// Returns (`f32`): the L2 residual norm, accumulated in `f64` so no
/// finite input overflows; non-finite inputs propagate to a non-finite
/// result (which saturates at the builder).
pub fn residual_norm<T: VectorElement>(row_bytes: &[u8], c: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len(), c.len() * T::SIZE_BYTES);
    let mut acc = 0.0f64;
    for (chunk, &ci) in row_bytes.chunks_exact(T::SIZE_BYTES).zip(c) {
        let xi = T::decode_le(chunk.try_into().unwrap()).to_f32();
        let d = xi as f64 - ci as f64;
        acc += d * d;
    }
    acc.sqrt() as f32
}

// ======================================================================
// P2: query bound
// ======================================================================

/// Collector-side peek: the kth ordering key iff the heap holds k results.
///
/// The key is in NATIVE heap space — the similarity the heap actually
/// ranks by (`-d^2` for L2, cosine for cosine, the raw score for dot).
/// Conversion into bound space is the query bound's job, downstream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeapPeek {
    /// Fewer than k results collected; no kth key exists.
    Filling,
    /// Exactly k results held; `kth_key` (`f32`) is the exact kth-best
    /// heap key.
    Full { kth_key: f32 },
}

impl HeapPeek {
    /// Wraps a [`TopNComputer::kth_best`] snapshot of the similarity
    /// component.
    ///
    /// * `kth` (`Option<f32>`) — the exact kth-best heap key; `None` while the heap holds fewer
    ///   than k results.
    ///
    /// Returns (`HeapPeek`): `Full` iff the heap holds k results.
    ///
    /// [`TopNComputer::kth_best`]: crate::collector::TopNComputer::kth_best
    pub fn from_kth(kth: Option<f32>) -> HeapPeek {
        match kth {
            Some(kth_key) => HeapPeek::Full { kth_key },
            None => HeapPeek::Filling,
        }
    }
}

/// The armed kth threshold in BOUND space. Lives next to the collector
/// heap; the conversion runs only when the kth result improves, so the
/// cosine sqrt is paid per heap update, not per cluster. `Filling` makes
/// armed-only structural: no threshold exists before k results.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QueryBound {
    /// Fewer than k results; no threshold exists.
    Filling,
    /// The heap holds k results; `t` (`f32`) is the kth threshold in
    /// bound space — the query region's extent.
    Armed { t: f32 },
}

/// Heap key -> bound space, per metric.
///
/// * `metric` (`Metric`) — the field's metric, which fixes the heap's key space.
/// * `heap_key` (`f32`) — the exact kth heap key ([`HeapPeek::Full`]): `-d^2` for L2, cosine for
///   cosine, the raw score for dot.
///
/// Returns (`f32`): the threshold in bound space — the L2 distance
/// `sqrt(-key)` for L2, the chord `sqrt(2 * (1 - key))` for cosine (unit
/// vectors: `||q - x||^2 = 2 * (1 - cos)`), the key itself for dot (the
/// halfspace offset `s_k`). Float-noise excursions past the metric's key
/// range clamp to `0.0`; a NaN key propagates and fails open downstream.
#[inline]
pub fn to_bound_space(metric: Metric, heap_key: f32) -> f32 {
    match metric {
        Metric::L2 => (-heap_key).max(0.0).sqrt(),
        Metric::Cosine => (2.0 * (1.0 - heap_key).max(0.0)).sqrt(),
        Metric::Dot => heap_key,
    }
}

/// Maintains the [`QueryBound`] beside the probe loop: converts heap
/// peeks into bound space on kth improvement only, and records the probe
/// index at which the bound first armed.
pub(crate) struct QueryBoundTracker {
    bound: QueryBound,
    /// The raw kth heap key behind the current `t` — the improvement
    /// detector that keeps [`to_bound_space`] off the per-cluster path.
    raw_kth: Option<f32>,
    armed_at_probe: Option<u32>,
}

impl QueryBoundTracker {
    /// Starts unarmed.
    ///
    /// Returns (`QueryBoundTracker`): `Filling`, no kth key, no armed
    /// index.
    pub(crate) fn new() -> Self {
        Self {
            bound: QueryBound::Filling,
            raw_kth: None,
            armed_at_probe: None,
        }
    }

    /// Folds one cluster-boundary peek into the bound.
    ///
    /// * `metric` (`Metric`) — the field's metric, for the key conversion.
    /// * `peek` (`HeapPeek`) — the heap's exact kth snapshot after the cluster's rows were scored.
    /// * `probe_idx` (`u32`) — 0-based index of the cluster just probed; captured once, at the
    ///   arming transition.
    pub(crate) fn observe(&mut self, metric: Metric, peek: HeapPeek, probe_idx: u32) {
        let HeapPeek::Full { kth_key } = peek else {
            return;
        };
        if self.raw_kth != Some(kth_key) {
            self.raw_kth = Some(kth_key);
            self.bound = QueryBound::Armed {
                t: to_bound_space(metric, kth_key),
            };
        }
        if self.armed_at_probe.is_none() {
            self.armed_at_probe = Some(probe_idx);
        }
    }

    /// The current bound.
    ///
    /// Returns (`QueryBound`): `Armed` iff a `Full` peek has been
    /// observed.
    pub(crate) fn bound(&self) -> QueryBound {
        self.bound
    }

    /// The probe index at which the bound armed.
    ///
    /// Returns (`Option<u32>`): 0-based probed-cluster index of the first
    /// `Full` peek; `None` if the heap never filled.
    pub(crate) fn armed_at_probe(&self) -> Option<u32> {
        self.armed_at_probe
    }
}

// ======================================================================
// P4: collision
// ======================================================================
// Ball column: scalar-only. `separation` / `q_dot_c` is the routing key of
// the ranked stream. Precondition (debug_assert at the call site): the
// stream key is the EXACT centroid distance / dot -- an approximate key
// makes a skip unsound.

/// Signed probe margin for a ball cluster bound vs the armed ball region.
/// Sphere-sphere: L2 (`separation = ||q - c||`) and cosine
/// (`separation = chord(q_hat, c_hat)`).
///
/// * `t` (`f32`) — armed kth threshold in bound space (the query's extent).
/// * `r` (`f32`) — cluster bound radius; `f32::INFINITY` = SATURATED (always probes).
/// * `separation` (`f32`) — exact centroid distance from the routing stream key.
///
/// Returns (`f32`): the signed margin — `>= 0` probe, `< 0` skip-safe
/// (strict). Magnitude = overlap quality; never an ordering key.
#[inline]
pub fn margin_ball_ball(t: f32, r: f32, separation: f32) -> f32 {
    (t + r) - separation
}

/// Signed probe margin for a ball cluster bound vs the armed halfspace.
/// Sphere-halfspace via Cauchy-Schwarz on the residual: the cluster's
/// best possible score vs the kth score. Dot only.
///
/// * `q_dot_c` (`f32`) — exact `q . c` from the routing stream key.
/// * `q_norm` (`f32`) — `||q||`, computed once per query.
/// * `r` (`f32`) — cluster bound radius; `f32::INFINITY` = SATURATED (always probes).
/// * `s_k` (`f32`) — armed kth score (the halfspace offset).
///
/// Returns (`f32`): the signed margin — `>= 0` probe, `< 0` skip-safe
/// (strict). Magnitude = overlap quality; never an ordering key.
#[inline]
pub fn margin_ball_halfspace(q_dot_c: f32, q_norm: f32, r: f32, s_k: f32) -> f32 {
    (q_dot_c + q_norm * r) - s_k
}

// ---- future box column (formulas fixed now, code lands with the kind) --
//
// BallRegion x Aabb: t - dist(q, box(c, h)); the box distance needs the
// residual q - c:
//   d2 = sum_i max(|q_i - c_i| - h_i, 0)^2;  margin = t - sqrt(d2)
//
// Halfspace x Aabb: support of the box in direction q:
//   margin = (q . c + sum_i |q_i| * h_i) - s_k
//
// Both walk the dims per check: the box column does not get the
// routing-key-free scalar path.

// ======================================================================
// P5: gate verdict
// ======================================================================

/// The gate's decision for one ranked cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Open and scan the cluster.
    Probe,
    /// Pass the cluster over — its bound proves it cannot improve the
    /// armed result.
    Skip,
}

/// The entire consumption surface of the bound.
///
/// * `qb` (`QueryBound`) — the current query bound; the bound is consumed only through `Armed` (the
///   heap holds k results) — enforced by the enum, not a runtime check.
/// * `margin` (`impl FnOnce() -> f32`) — lazily computes the cluster's signed margin; never called
///   while `Filling`.
///
/// Returns (`Verdict`): skip only on STRICT negative — `margin == 0`
/// (exact touch) probes, or a boundary tie breaks exactness. A NaN margin
/// compares false and probes: fail-open is arithmetic. Margin magnitude
/// is the overlap-quality number and is NEVER an ordering key.
#[inline]
pub fn bounds_verdict(qb: QueryBound, margin: impl FnOnce() -> f32) -> Verdict {
    match qb {
        QueryBound::Filling => Verdict::Probe,
        QueryBound::Armed { .. } => {
            if margin() < 0.0 {
                Verdict::Skip
            } else {
                Verdict::Probe
            }
        }
    }
}

#[cfg(test)]
mod bounds_storage_tests {
    use super::*;

    /// Ball stride is 1 f32/cluster whatever the dimensionality — the
    /// stride table consults the kind, not the field.
    #[test]
    fn stride_table() {
        for dim in [1usize, 2, 128, 1536] {
            assert_eq!(BoundKind::Ball.stride(dim), 1);
        }
    }

    #[test]
    fn kind_byte_round_trips() {
        assert_eq!(
            BoundKind::from_code(BoundKind::Ball as u8).unwrap(),
            BoundKind::Ball
        );
        assert!(BoundKind::from_code(7).is_err());
    }

    /// The builder folds a max, starts exact at 0.0, and saturates on
    /// non-finite residuals and degenerate centroids.
    #[test]
    fn builder_folds_max_and_saturates() {
        let mut builder = BoundsBuilder::new(3);
        builder.add_native(0, 1.0);
        builder.add_native(0, 0.5);
        builder.add_native(0, 2.0);
        builder.add_native(1, f32::NAN); // non-finite member residual
        builder.saturate(2); // degenerate centroid
        let r = builder.finish();
        assert_eq!(r[0], 2.0);
        assert_eq!(r[1], f32::INFINITY);
        assert_eq!(r[2], f32::INFINITY);
    }

    /// A saturated cluster stays saturated: later finite members must not
    /// un-saturate it (max against +inf).
    #[test]
    fn saturation_is_sticky() {
        let mut builder = BoundsBuilder::new(1);
        builder.add_native(0, f32::INFINITY);
        builder.add_native(0, 0.25);
        assert_eq!(builder.finish()[0], f32::INFINITY);
    }

    /// Little-endian `f32` row bytes, as the row store holds them.
    fn row(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn residual_norm_matches_hand_values() {
        assert_eq!(residual_norm::<f32>(&row(&[3.0, 4.0]), &[0.0, 0.0]), 5.0);
        assert_eq!(residual_norm::<f32>(&row(&[1.0, 1.0]), &[1.0, 1.0]), 0.0);
        assert!(residual_norm::<f32>(&row(&[f32::NAN, 0.0]), &[0.0, 0.0]).is_nan());
        assert_eq!(
            residual_norm::<f32>(&row(&[f32::INFINITY, 0.0]), &[0.0, 0.0]),
            f32::INFINITY
        );
    }
}

#[cfg(test)]
mod bounds_peek_tests {
    use std::sync::Arc;

    use super::*;
    use crate::collector::sort_key::NaturalComparator;
    use crate::collector::TopNComputer;
    use crate::schema::Metric;
    use crate::vector::PreparedQuery;

    /// The vector heap: f32 similarity keys, "higher is better",
    /// out-of-order pushes — the `scan_clusters` shape.
    fn heap(k: usize) -> TopNComputer<f32, u32, NaturalComparator> {
        TopNComputer::new_with_comparator(k, NaturalComparator)
    }

    /// Little-endian row bytes for `values`, as the row store holds them.
    fn row(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// k-1 results peek `Filling`; the kth flips to `Full` with the exact
    /// kth key; `Full` never appears earlier.
    #[test]
    fn filling_until_k() {
        let k = 3;
        let mut topn = heap(k);
        for (doc, score) in [(0u32, 0.9f32), (1, 0.5)] {
            topn.push_unordered(score, doc);
            assert_eq!(
                HeapPeek::from_kth(topn.kth_best()),
                HeapPeek::Filling,
                "no kth key below k results"
            );
        }
        topn.push_unordered(0.7, 2);
        assert_eq!(
            HeapPeek::from_kth(topn.kth_best()),
            HeapPeek::Full { kth_key: 0.5 },
            "exactly k results arm the peek with the kth key"
        );
    }

    /// The kth key is in NATIVE heap space per metric: `-d^2` for L2,
    /// cosine similarity for cosine, the raw dot score for dot — the
    /// values `PreparedQuery::score_doc_bytes` produces over known
    /// vectors.
    #[test]
    fn kth_key_per_metric() {
        let query = vec![1.0f32, 0.0];
        // Stored rows: for cosine these must be unit-norm (the write path
        // normalizes at ingest); L2/Dot store raw values.
        let docs: [(&[f32], u32); 3] = [
            (&[1.0, 0.0], 0),  // d^2 = 0,   cos = 1.0,       dot = 1.0
            (&[0.0, 1.0], 1),  // d^2 = 2,   cos = 0.0,       dot = 0.0
            (&[-1.0, 0.0], 2), // d^2 = 4,   cos = -1.0,      dot = -1.0
        ];
        let expected_kth = [
            (Metric::L2, -2.0f32), // 2nd best of {0, -2, -4}
            (Metric::Cosine, 0.0), // 2nd best of {1, 0, -1}
            (Metric::Dot, 0.0),    // 2nd best of {1, 0, -1}
        ];
        for (metric, expected) in expected_kth {
            let prepared = PreparedQuery::<f32>::new(metric, Arc::new(query.clone()));
            let mut topn = heap(2);
            for (values, doc) in docs {
                topn.push_unordered(prepared.score_doc_bytes(&row(values)), doc);
            }
            let peek = HeapPeek::from_kth(topn.kth_best());
            assert_eq!(
                peek,
                HeapPeek::Full { kth_key: expected },
                "{metric:?}: kth key must be the native-space similarity"
            );
        }
    }

    /// A better result tightens the kth key; a worse one leaves it alone.
    #[test]
    fn improvement_updates_key() {
        let mut topn = heap(2);
        topn.push_unordered(0.2, 0);
        topn.push_unordered(0.4, 1);
        assert_eq!(
            HeapPeek::from_kth(topn.kth_best()),
            HeapPeek::Full { kth_key: 0.2 }
        );
        topn.push_unordered(0.9, 2);
        assert_eq!(
            HeapPeek::from_kth(topn.kth_best()),
            HeapPeek::Full { kth_key: 0.4 },
            "an improving push must raise the kth key"
        );
        topn.push_unordered(0.1, 3);
        assert_eq!(
            HeapPeek::from_kth(topn.kth_best()),
            HeapPeek::Full { kth_key: 0.4 },
            "a non-improving push must not move the kth key"
        );
    }
}

#[cfg(test)]
mod bounds_margin_tests {
    use super::*;

    /// Sphere-sphere: disjoint is negative, exact touch is zero,
    /// overlapping is positive.
    #[test]
    fn sphere_sphere_cases() {
        // t = 1, r = 2: extents cover 3. Separation 5 -> disjoint by 2.
        assert_eq!(margin_ball_ball(1.0, 2.0, 5.0), -2.0);
        // Separation exactly 3 -> touching.
        assert_eq!(margin_ball_ball(1.0, 2.0, 3.0), 0.0);
        // Separation 2 -> overlap by 1.
        assert_eq!(margin_ball_ball(1.0, 2.0, 2.0), 1.0);
    }

    /// Sphere-halfspace: the same trio through the dot margin — best
    /// possible score below / at / above the kth score.
    #[test]
    fn halfspace_cases() {
        // q.c = 1, ||q|| = 2, r = 0.5: best possible 2. s_k = 3 -> short
        // by 1.
        assert_eq!(margin_ball_halfspace(1.0, 2.0, 0.5, 3.0), -1.0);
        // s_k = 2 -> exact touch.
        assert_eq!(margin_ball_halfspace(1.0, 2.0, 0.5, 2.0), 0.0);
        // s_k = 1.5 -> the cluster can beat the kth by 0.5.
        assert_eq!(margin_ball_halfspace(1.0, 2.0, 0.5, 1.5), 0.5);
    }

    /// SATURATED (`r = +inf`) forces a `+inf` margin in both collision
    /// functions, for any finite inputs — fail-open is arithmetic.
    #[test]
    fn saturated_probes() {
        assert_eq!(margin_ball_ball(0.0, f32::INFINITY, 1.0e30), f32::INFINITY);
        assert_eq!(
            margin_ball_halfspace(-1.0e30, 1.0, f32::INFINITY, 1.0e30),
            f32::INFINITY
        );
        assert_eq!(
            bounds_verdict(QueryBound::Armed { t: 0.0 }, || margin_ball_ball(
                0.0,
                f32::INFINITY,
                1.0e30
            )),
            Verdict::Probe
        );
    }

    /// The verdict skips on STRICT negative only: an exact-touch margin
    /// of zero probes, and NaN (degenerate arithmetic) probes.
    #[test]
    fn boundary_tie_probes() {
        let armed = QueryBound::Armed { t: 1.0 };
        assert_eq!(bounds_verdict(armed, || 0.0), Verdict::Probe);
        assert_eq!(bounds_verdict(armed, || -0.0), Verdict::Probe);
        assert_eq!(bounds_verdict(armed, || f32::NAN), Verdict::Probe);
        assert_eq!(bounds_verdict(armed, || -1.0e-30), Verdict::Skip);
        assert_eq!(bounds_verdict(armed, || 1.0e-30), Verdict::Probe);
    }

    /// While `Filling`, the margin closure is never even called — the
    /// bound is consumed only through `Armed`.
    #[test]
    fn filling_never_consumes_margin() {
        let verdict = bounds_verdict(QueryBound::Filling, || {
            panic!("margin must not be computed while filling")
        });
        assert_eq!(verdict, Verdict::Probe);
    }
}
