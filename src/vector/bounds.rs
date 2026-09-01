//! Cluster-bound construction and query-time pruning.

use std::io;

use crate::schema::Metric;
use crate::vector::VectorElement;

/// Identifies a stored cluster-bound shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BoundKind {
    /// Maximum member-to-centroid distance per cluster.
    Ball = 0,
}

impl BoundKind {
    /// Decodes a bound-kind byte.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown kind byte.
    pub fn from_code(code: u8) -> io::Result<BoundKind> {
        match code {
            0 => Ok(BoundKind::Ball),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown centroid bound kind: {other}"),
            )),
        }
    }

    /// Returns the number of binary32 values per cluster.
    pub fn stride(self, _dim: usize) -> usize {
        match self {
            BoundKind::Ball => 1,
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
    pub fn new(kind: BoundKind, data: &'a [f32]) -> Self {
        Self { kind, data }
    }

    /// Returns the stored bound kind.
    pub fn kind(&self) -> BoundKind {
        self.kind
    }

    /// Returns one cluster's ball radius.
    #[inline]
    pub fn ball_r(&self, cluster: usize) -> f32 {
        debug_assert_eq!(self.kind, BoundKind::Ball);
        self.data[cluster]
    }

    /// Returns the raw bounds payload.
    pub fn values(&self) -> &'a [f32] {
        self.data
    }
}

/// Accumulates one segment's per-cluster ball bounds during a merge.
pub struct BoundsBuilder {
    r: Vec<f32>,
}

impl BoundsBuilder {
    /// Creates zeroed cluster bounds.
    pub fn new(num_clusters: usize) -> Self {
        Self {
            r: vec![0.0; num_clusters],
        }
    }

    /// Adds one primary member's residual norm.
    pub fn add_native(&mut self, cluster: usize, residual_norm: f32) {
        let slot = &mut self.r[cluster];
        if !residual_norm.is_finite() {
            *slot = f32::INFINITY; // SATURATED
        } else if residual_norm > *slot {
            *slot = residual_norm;
        }
    }

    /// Saturates one cluster bound.
    pub fn saturate(&mut self, cluster: usize) {
        self.r[cluster] = f32::INFINITY;
    }

    /// Returns the accumulated cluster radii.
    pub fn finish(self) -> Vec<f32> {
        self.r
    }
}

/// Computes a stored row's residual norm from its centroid.
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

/// Represents whether the result heap has a kth key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeapPeek {
    /// Fewer than `k` results are present.
    Filling,
    /// The heap holds a kth key.
    Full {
        /// Kth key.
        kth_key: f32,
    },
}

impl HeapPeek {
    /// Converts an optional kth key into a heap state.
    pub fn from_kth(kth: Option<f32>) -> HeapPeek {
        match kth {
            Some(kth_key) => HeapPeek::Full { kth_key },
            None => HeapPeek::Filling,
        }
    }
}

/// Stores the kth threshold in bound space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QueryBound {
    /// Fewer than `k` results are present.
    Filling,
    /// The heap holds a bound threshold.
    Armed {
        /// Kth bound threshold.
        t: f32,
    },
}

/// Converts a metric heap key into bound space.
#[inline]
pub fn to_bound_space(metric: Metric, heap_key: f32) -> f32 {
    match metric {
        Metric::L2 => (-heap_key).max(0.0).sqrt(),
        Metric::Cosine => (2.0 * (1.0 - heap_key).max(0.0)).sqrt(),
        Metric::Dot => heap_key,
    }
}

/// Tracks the current query bound and its first armed probe.
pub(crate) struct QueryBoundTracker {
    bound: QueryBound,
    /// Current kth heap key.
    raw_kth: Option<f32>,
    armed_at_probe: Option<u32>,
}

impl QueryBoundTracker {
    /// Creates an unarmed tracker.
    pub(crate) fn new() -> Self {
        Self {
            bound: QueryBound::Filling,
            raw_kth: None,
            armed_at_probe: None,
        }
    }

    /// Updates the bound from one cluster-boundary heap state.
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

    /// Returns the current query bound.
    pub(crate) fn bound(&self) -> QueryBound {
        self.bound
    }

    /// Returns the probe index that armed the bound.
    pub(crate) fn armed_at_probe(&self) -> Option<u32> {
        self.armed_at_probe
    }
}

/// Computes a ball-to-ball signed probe margin.
#[inline]
pub fn margin_ball_ball(t: f32, r: f32, separation: f32) -> f32 {
    (t + r) - separation
}

/// Computes a ball-to-halfspace signed probe margin.
#[inline]
pub fn margin_ball_halfspace(q_dot_c: f32, q_norm: f32, r: f32, s_k: f32) -> f32 {
    (q_dot_c + q_norm * r) - s_k
}

/// Gate decision for one ranked cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Scan the cluster.
    Probe,
    /// Skip the cluster.
    Skip,
}

/// Converts a query bound and signed margin into a probe verdict.
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

    #[test]
    fn builder_folds_max_and_saturates() {
        let mut builder = BoundsBuilder::new(3);
        builder.add_native(0, 1.0);
        builder.add_native(0, 0.5);
        builder.add_native(0, 2.0);
        builder.add_native(1, f32::NAN);
        builder.saturate(2);
        let r = builder.finish();
        assert_eq!(r[0], 2.0);
        assert_eq!(r[1], f32::INFINITY);
        assert_eq!(r[2], f32::INFINITY);
    }

    #[test]
    fn saturation_is_sticky() {
        let mut builder = BoundsBuilder::new(1);
        builder.add_native(0, f32::INFINITY);
        builder.add_native(0, 0.25);
        assert_eq!(builder.finish()[0], f32::INFINITY);
    }

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

    fn heap(k: usize) -> TopNComputer<f32, u32, NaturalComparator> {
        TopNComputer::with_comparator(k, NaturalComparator)
    }

    fn row(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

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

    #[test]
    fn kth_key_per_metric() {
        let query = vec![1.0f32, 0.0];
        let docs: [(&[f32], u32); 3] = [(&[1.0, 0.0], 0), (&[0.0, 1.0], 1), (&[-1.0, 0.0], 2)];
        let expected_kth = [
            (Metric::L2, -2.0f32),
            (Metric::Cosine, 0.0),
            (Metric::Dot, 0.0),
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

    #[test]
    fn sphere_sphere_cases() {
        assert_eq!(margin_ball_ball(1.0, 2.0, 5.0), -2.0);
        assert_eq!(margin_ball_ball(1.0, 2.0, 3.0), 0.0);
        assert_eq!(margin_ball_ball(1.0, 2.0, 2.0), 1.0);
    }

    #[test]
    fn halfspace_cases() {
        assert_eq!(margin_ball_halfspace(1.0, 2.0, 0.5, 3.0), -1.0);
        assert_eq!(margin_ball_halfspace(1.0, 2.0, 0.5, 2.0), 0.0);
        assert_eq!(margin_ball_halfspace(1.0, 2.0, 0.5, 1.5), 0.5);
    }

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

    #[test]
    fn boundary_tie_probes() {
        let armed = QueryBound::Armed { t: 1.0 };
        assert_eq!(bounds_verdict(armed, || 0.0), Verdict::Probe);
        assert_eq!(bounds_verdict(armed, || -0.0), Verdict::Probe);
        assert_eq!(bounds_verdict(armed, || f32::NAN), Verdict::Probe);
        assert_eq!(bounds_verdict(armed, || -1.0e-30), Verdict::Skip);
        assert_eq!(bounds_verdict(armed, || 1.0e-30), Verdict::Probe);
    }

    #[test]
    fn filling_never_consumes_margin() {
        let verdict = bounds_verdict(QueryBound::Filling, || {
            panic!("margin must not be computed while filling")
        });
        assert_eq!(verdict, Verdict::Probe);
    }
}
