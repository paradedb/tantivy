//! Per-query precomputation hoisted out of the per-doc scoring loop.
//!
//! Built once per [`VectorBackend::for_segment`] and held by
//! [`FlatBackend`] / [`IvfBackend`]. Hides the metric match and any
//! metric-specific precomputed scalars (currently only `1/||q||` for
//! cosine) behind two named entry points:
//!
//! * [`PreparedQuery::score_doc_bytes`] — score against a stored
//!   document. For `Cosine`, the document must be unit-normalized at
//!   write time (`FlatVecWriter` enforces this).
//! * [`PreparedQuery::score_centroid_bytes`] — score against an IVF
//!   centroid (which is *not* unit-normalized — k-means cluster mean).
//!
//! New per-query precomputation for a future metric adds an enum
//! variant here, not a field on every backend.
//!
//! [`VectorBackend::for_segment`]: super::backend::VectorBackend::for_segment
//! [`FlatBackend`]: super::backend::FlatBackend
//! [`IvfBackend`]: super::backend::IvfBackend

use std::sync::Arc;

use super::distance::{cosine_bytes, dot_bytes, l2_squared_bytes, norm_squared};
use super::options::{Metric, VectorElement};

/// A search-ready bundle of `(metric, query, …precomputed scalars)`.
///
/// Cheap to construct (one `norm_squared` pass over the query for
/// cosine, no allocations) and `Arc`-shared across per-segment backends
/// so the precomputation runs once per top-level query.
pub struct PreparedQuery<T: VectorElement> {
    query: Arc<Vec<T>>,
    kind: QueryKind,
}

/// Metric-specific per-query state. Each variant carries only what
/// that metric actually needs — no dead fields for L2 / Dot.
enum QueryKind {
    L2,
    Dot,
    Cosine {
        /// `1.0 / ||q||`, used to turn the per-doc
        /// `dot(q, d_unit) / ||q||` into a single multiply. Set to
        /// `0.0` for zero / non-finite query norm so a degenerate query
        /// scores `0.0` against every doc — preserves the
        /// `nq == 0.0 -> 0.0` short-circuit the old `cosine_bytes`
        /// branch enforced.
        inv_norm_q: f32,
    },
}

impl<T: VectorElement> PreparedQuery<T> {
    pub fn new(metric: Metric, query: Arc<Vec<T>>) -> Self {
        let kind = match metric {
            Metric::L2 => QueryKind::L2,
            Metric::Dot => QueryKind::Dot,
            Metric::Cosine => {
                let nq = norm_squared::<T>(&query).sqrt();
                let inv_norm_q = if nq == 0.0 || !nq.is_finite() {
                    0.0
                } else {
                    1.0 / nq
                };
                QueryKind::Cosine { inv_norm_q }
            }
        };
        Self { query, kind }
    }

    pub fn metric(&self) -> Metric {
        match self.kind {
            QueryKind::L2 => Metric::L2,
            QueryKind::Dot => Metric::Dot,
            QueryKind::Cosine { .. } => Metric::Cosine,
        }
    }

    pub fn query(&self) -> &[T] {
        &self.query
    }

    /// Score a stored document. For `Cosine` the doc must be unit-
    /// normalized at write time — `FlatVecWriter::push_bytes` does that
    /// — letting us replace `dot(q, d) / (||q|| * ||d||)` with a single
    /// `dot(q, d_unit) * inv_norm_q`.
    #[inline]
    pub fn score_doc_bytes(&self, doc_bytes: &[u8]) -> f32 {
        match self.kind {
            QueryKind::L2 => -l2_squared_bytes::<T>(&self.query, doc_bytes),
            QueryKind::Dot => dot_bytes::<T>(&self.query, doc_bytes),
            QueryKind::Cosine { inv_norm_q } => {
                dot_bytes::<T>(&self.query, doc_bytes) * inv_norm_q
            }
        }
    }

    /// Score an IVF centroid (a k-means cluster mean, *not* unit-
    /// normalized). Cosine takes the full `cosine_bytes` path here so
    /// the centroid's own norm is honored — only the per-doc cluster
    /// scan exploits the unit-norm invariant.
    #[inline]
    pub fn score_centroid_bytes(&self, centroid_bytes: &[u8]) -> f32 {
        match self.kind {
            QueryKind::L2 => -l2_squared_bytes::<T>(&self.query, centroid_bytes),
            QueryKind::Dot => dot_bytes::<T>(&self.query, centroid_bytes),
            QueryKind::Cosine { .. } => cosine_bytes::<T>(&self.query, centroid_bytes),
        }
    }
}
