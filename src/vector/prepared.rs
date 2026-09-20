//! Per-query precomputation hoisted out of the per-doc scoring loop.
//!
//! Built once per [`VectorBackend::for_segment`] and held by the backend.
//! Hides the metric match and any metric-specific precomputed scalars
//! (currently only `1/||q||` for cosine) behind
//! [`PreparedQuery::score_doc_bytes`].
//!
//! Stored vectors — including IVF centroids — are unit-normalized at
//! write time for `Cosine + F32` (see
//! [`maybe_normalize_bytes`](super::distance::maybe_normalize_bytes)),
//! so a single scoring entry point covers both per-doc and centroid
//! scans.
//!
//! [`VectorBackend::for_segment`]: super::backend::VectorBackend::for_segment

use std::sync::Arc;

use super::distance::{dot_bytes, l2_squared_bytes, norm_squared_wide, DotAccumulator};
use super::VectorElement;
use crate::schema::{Metric, VectorDType};

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
                // Wide accumulation, so a huge-but-finite query norm stays
                // finite. The degenerate guard remains load-bearing: queries
                // are user input at search time, not ingest-validated.
                let nq = norm_squared_wide::<T>(&query).sqrt();
                let inv_norm_q = if nq == 0.0 || !nq.is_finite() {
                    0.0
                } else {
                    (1.0 / nq) as f32
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
    pub(crate) fn dot_accumulator(&self) -> Option<DotAccumulator> {
        (T::DTYPE == VectorDType::F32 && T::SIZE_BYTES == 4 && !matches!(self.kind, QueryKind::L2))
            .then(DotAccumulator::new)
    }

    #[inline]
    pub(crate) fn score_doc_fragment(
        &self,
        accumulator: &mut DotAccumulator,
        bytes: &[u8],
        complete: bool,
    ) -> Option<f32> {
        accumulator.push::<T>(&self.query, bytes);
        complete.then(|| {
            let dot = accumulator.finish::<T>(&self.query);
            match self.kind {
                QueryKind::Dot => dot,
                QueryKind::Cosine { inv_norm_q } => dot * inv_norm_q,
                QueryKind::L2 => unreachable!("L2 uses the contiguous row scorer"),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn fragmented_dot_scores_match_contiguous_bits() {
        for dim in [1, 15, 16, 17, 31, 33, 1024] {
            for profile in 0..3 {
                let query: Vec<f32> = (0..dim)
                    .map(|i| match profile {
                        0 => ((i * 17 % 31) as f32 - 15.0) * 0.03137,
                        1 => [1.0e20, -1.0e20, 1.0e-20, -1.0e-20][i % 4],
                        _ => {
                            if i % 2 == 0 {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                    })
                    .collect();
                let doc: Vec<f32> = (0..dim)
                    .map(|i| ((i * 13 % 43) as f32 - 21.0) * 0.06257)
                    .collect();
                let doc = bytes(&doc);
                for metric in [Metric::Dot, Metric::Cosine] {
                    let prepared = PreparedQuery::new(metric, Arc::new(query.clone()));
                    let expected = prepared.score_doc_bytes(&doc).to_bits();
                    let mut accumulator = prepared.dot_accumulator().unwrap();
                    for split in 0..=doc.len() {
                        assert!(prepared
                            .score_doc_fragment(&mut accumulator, &doc[..split], false)
                            .is_none());
                        let score = prepared
                            .score_doc_fragment(&mut accumulator, &doc[split..], true)
                            .unwrap();
                        assert_eq!(
                            score.to_bits(),
                            expected,
                            "{metric:?}, dim={dim}, split={split}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fragmented_dot_scores_handle_many_chunks_and_row_reuse() {
        for dim in [3, 17, 1024, 4099] {
            for chunk_size in [1, 3, 7, 63, 64, 65, 8160] {
                for metric in [Metric::Dot, Metric::Cosine] {
                    let query = (0..dim).map(|i| i as f32 * 0.003 - 0.2).collect();
                    let prepared = PreparedQuery::new(metric, Arc::new(query));
                    let mut accumulator = prepared.dot_accumulator().unwrap();
                    for row in 0..3 {
                        let doc = bytes(
                            &(0..dim)
                                .map(|i| ((i + row * 7) % 23) as f32 * 0.125 - 1.0)
                                .collect::<Vec<_>>(),
                        );
                        let mut actual = None;
                        for (i, fragment) in doc.chunks(chunk_size).enumerate() {
                            assert!(actual.is_none());
                            actual = prepared.score_doc_fragment(
                                &mut accumulator,
                                fragment,
                                (i + 1) * chunk_size >= doc.len(),
                            );
                        }
                        assert_eq!(
                            actual.unwrap().to_bits(),
                            prepared.score_doc_bytes(&doc).to_bits(),
                            "{metric:?}, dim={dim}, chunks={chunk_size}",
                        );
                    }
                }
            }
        }
        let prepared = PreparedQuery::new(Metric::L2, Arc::new(vec![1.0f32; 17]));
        assert!(prepared.dot_accumulator().is_none());
    }
}
