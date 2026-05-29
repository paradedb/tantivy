//! Per-query precomputation hoisted out of the per-doc scoring loop.
//!
//! Built once per [`VectorBackend::for_segment`] and held by
//! [`FlatBackend`] / [`IvfBackend`]. Hides the metric match and any
//! metric-specific precomputed scalars (currently only `1/||q||` for
//! cosine) behind [`PreparedQuery::score_doc_bytes`].
//!
//! Stored vectors — including IVF centroids — are unit-normalized at
//! write time for `Cosine + F32` (see
//! [`VectorOptions::maybe_normalize_bytes`](super::options::VectorOptions::maybe_normalize_bytes)),
//! so a single scoring entry point covers both per-doc and centroid
//! scans.
//!
//! [`VectorBackend::for_segment`]: super::backend::VectorBackend::for_segment
//! [`FlatBackend`]: super::backend::FlatBackend
//! [`IvfBackend`]: super::backend::IvfBackend

use std::sync::Arc;

use super::distance::{dot_bytes, l2_squared_bytes, norm_squared};
use super::options::{Metric, VectorElement};

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
        /// `1.0 / ||q||`. `0.0` for a zero / non-finite query norm so a
        /// degenerate query scores `0.0` against every doc.
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

    /// Score a stored vector — either a document or an IVF centroid.
    /// Both are unit-normalized at write time for `Cosine + F32`, so
    /// the cosine branch collapses to `dot * inv_norm_q`.
    #[inline]
    pub fn score_doc_bytes(&self, doc_bytes: &[u8]) -> f32 {
        match self.kind {
            QueryKind::L2 => -l2_squared_bytes::<T>(&self.query, doc_bytes),
            QueryKind::Dot => dot_bytes::<T>(&self.query, doc_bytes),
            QueryKind::Cosine { inv_norm_q } => dot_bytes::<T>(&self.query, doc_bytes) * inv_norm_q,
        }
    }
}
