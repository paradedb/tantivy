//! The query, prepared once per search, and the quantized query context built from it for each
//! segment encoding.
//!
//! Document scoring uses write-time cosine normalization. Routing also handles unnormalized
//! centroids and shares the query's cached norm across every level and segment.

use std::sync::Arc;

use quant::cascade::{prepare_split_query_with_plan, PreparedSplitQuery};

use crate::schema::Metric;
use crate::vector::distance::{dot_bytes, l2_squared_bytes, norm_squared_wide, Similarity};
use crate::vector::quantization::SIGN_QUERY_BITS;
use crate::vector::storage::quantized::QuantizedIndexCtx;
use crate::vector::telemetry::charge_fallbacks;
use crate::vector::VectorElement;
use crate::TantivyError;

// --- Query ---

/// Shared query values, norms, and cosine normalization, prepared once per search.
pub(crate) struct VectorQuery<T: VectorElement> {
    metric: Metric,
    raw: Arc<Vec<T>>,
    /// The values widened losslessly to `f32`: what routers rank with.
    widened: Arc<[f32]>,
    /// `||q||`, for the dot margin's Cauchy-Schwarz term.
    norm: f32,
    routing_norm: f32,
    estimate_norm: f32,
    kind: QueryKind,
}

/// Metric-specific per-query state.
enum QueryKind {
    L2,
    Dot,
    Cosine {
        /// Reciprocal query norm, or zero for a degenerate query, which then scores zero.
        inv_norm_q: f32,
        normalized: Vec<f32>,
    },
}

/// The query as one vector set scores it: the query plus that set's quantization, if any.
#[derive(Clone, Copy)]
pub(crate) struct ScoringQuery<'q, T: VectorElement> {
    pub(crate) query: &'q VectorQuery<T>,
    pub(crate) quantized: Option<&'q QuantizedQueryCtx>,
}

impl<T: VectorElement> VectorQuery<T> {
    /// Prepares `raw` under `metric`.
    pub(crate) fn new(metric: Metric, raw: Arc<Vec<T>>) -> Self {
        // Wide accumulation keeps a huge but finite norm finite. Queries are user input, so the
        // degenerate guard stays.
        let norm_squared = norm_squared_wide::<T>(&raw);
        let norm = norm_squared.sqrt();
        let widened: Arc<[f32]> = raw.iter().map(|element| element.to_f32()).collect();
        // Routing preserves the scalar kernels' narrowing before the square root.
        let routing_norm = (norm_squared as f32).sqrt();
        let (kind, estimate_norm) = match metric {
            Metric::L2 => (QueryKind::L2, routing_norm),
            Metric::Dot => (QueryKind::Dot, routing_norm),
            Metric::Cosine => {
                let inv_norm_q = if norm == 0.0 || !norm.is_finite() {
                    0.0
                } else {
                    (1.0 / norm) as f32
                };
                let normalized: Vec<_> = if norm == 0.0 || !norm.is_finite() {
                    vec![0.0; widened.len()]
                } else {
                    widened.iter().map(|value| value * inv_norm_q).collect()
                };
                let estimate_norm = (norm_squared_wide(&normalized) as f32).sqrt();
                (
                    QueryKind::Cosine {
                        inv_norm_q,
                        normalized,
                    },
                    estimate_norm,
                )
            }
        };
        Self {
            metric,
            widened,
            norm: norm as f32,
            routing_norm,
            estimate_norm,
            kind,
            raw,
        }
    }

    /// Similarity to one stored row under the metric.
    #[inline]
    pub(crate) fn score_row(&self, row: &[u8]) -> Similarity {
        Similarity::new(match self.kind {
            QueryKind::L2 => -l2_squared_bytes::<T>(&self.raw, row),
            QueryKind::Dot => dot_bytes::<T>(&self.raw, row),
            QueryKind::Cosine { inv_norm_q, .. } => dot_bytes::<T>(&self.raw, row) * inv_norm_q,
        })
    }

    /// The values widened to `f32`: what routers rank with.
    pub(crate) fn widened(&self) -> &[f32] {
        &self.widened
    }

    /// `||q||`, for the dot margin's Cauchy-Schwarz term.
    pub(crate) fn norm(&self) -> f32 {
        self.norm
    }

    pub(crate) fn routing_norm(&self) -> f32 {
        self.routing_norm
    }

    pub(crate) fn estimate_vector(&self) -> &[f32] {
        match &self.kind {
            QueryKind::Cosine { normalized, .. } => normalized,
            _ => self.widened(),
        }
    }

    pub(crate) fn metric(&self) -> Metric {
        self.metric
    }
}

// --- Quantized query ---

/// Immutable query rotations, sign bitplanes, and LUTs shared across segments.
pub(crate) struct QuantizedQueryCtx {
    pub(crate) index: Arc<QuantizedIndexCtx>,
    prepared: PreparedSplitQuery,
    query_norm: f32,
    active_layers: usize,
}

impl QuantizedQueryCtx {
    /// Prepares `query` for every layer of `index`'s encoding.
    pub(crate) fn new<T: VectorElement>(
        index: Arc<QuantizedIndexCtx>,
        query: &VectorQuery<T>,
    ) -> Self {
        let active_layers = index.specs.len();
        Self::with_depth(index, query, active_layers)
    }

    /// Prepares `query` for the first `active_layers` layers of `index`'s encoding.
    pub(crate) fn with_depth<T: VectorElement>(
        index: Arc<QuantizedIndexCtx>,
        query: &VectorQuery<T>,
        active_layers: usize,
    ) -> Self {
        assert!((1..=index.specs.len()).contains(&active_layers));
        debug_assert_eq!(index.meta.field().metric, query.metric());
        let vector = query.estimate_vector();
        let prepared = prepare_split_query_with_plan(
            vector,
            &index.rotation_plan,
            &index.grids[..active_layers],
            SIGN_QUERY_BITS,
        );
        Self {
            index,
            prepared,
            query_norm: query.estimate_norm,
            active_layers,
        }
    }

    pub(crate) fn active_layers(&self) -> usize {
        self.active_layers
    }

    /// Squared query-quantization error contributed by this layer alone: the
    /// sign-plane error for a 1-bit layer, zero for a grid layer. Callers that
    /// need the cumulative term sum it themselves (see `combine_refinement_decoded`).
    pub(crate) fn query_error_squared(&self, layer: usize) -> f64 {
        self.prepared.query_error_squared(layer)
    }

    pub(crate) fn score_layer(
        &self,
        layer: usize,
        codes: &[u8],
        scale: f32,
        constant: Option<f32>,
    ) -> crate::Result<f32> {
        match (self.index.meta.field().metric, constant) {
            (Metric::L2, Some(constant)) => Ok(self.prepared.score_layer(
                layer,
                codes,
                scale,
                constant,
                self.index.specs[layer],
            )),
            (Metric::L2, None) => Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only(
                    "quantized L2 scoring requires a split constant",
                ),
            )),
            (Metric::Dot | Metric::Cosine, None) => Ok(self.prepared.score_layer_without_constant(
                layer,
                codes,
                scale,
                self.index.specs[layer],
            )),
            (Metric::Dot | Metric::Cosine, Some(_)) => Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only(
                    "quantized dot and cosine scoring omit split constants",
                ),
            )),
        }
    }

    /// Scores a fixed-stride code batch with the resolved layer kernel, charging its sign-word
    /// fallbacks to the measure in scope.
    #[inline(always)]
    pub(crate) fn score_layer_batch_unscaled(
        &self,
        layer: usize,
        codes: &[u8],
        code_stride: usize,
        out: &mut [f32],
    ) {
        let fallbacks = self.prepared.score_layer_batch_unscaled(
            layer,
            codes,
            code_stride,
            self.index.specs[layer],
            out,
        );
        charge_fallbacks(layer, fallbacks);
    }

    /// Scores the rows at `row_offsets` of a code batch, charging its sign-word fallbacks to the
    /// measure in scope.
    #[inline(always)]
    pub(crate) fn score_layer_batch_unscaled_indexed(
        &self,
        layer: usize,
        codes: &[u8],
        code_stride: usize,
        row_offsets: &[usize],
        out: &mut [f32],
    ) {
        let fallbacks = self.prepared.score_layer_batch_unscaled_indexed(
            layer,
            codes,
            code_stride,
            row_offsets,
            self.index.specs[layer],
            out,
        );
        charge_fallbacks(layer, fallbacks);
    }

    /// The norm the layer error model scales by: the query-to-centroid distance under L2, otherwise
    /// the norm of the vector the estimates score with.
    pub(crate) fn score_query_norm(&self, centroid_score: f32) -> f32 {
        if self.index.meta.field().metric == Metric::L2 {
            (-centroid_score).max(0.0).sqrt()
        } else {
            self.query_norm
        }
    }

    /// The vector the estimates score with: the normalized query under cosine, otherwise
    /// `query`'s widened values. `query` is the query this context was prepared from.
    pub(crate) fn estimate_vector<'q, T: VectorElement>(
        &'q self,
        query: &'q VectorQuery<T>,
    ) -> &'q [f32] {
        query.estimate_vector()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{QuantizedIndexCtx, QuantizedQueryCtx, VectorQuery};
    use crate::schema::{Metric, VectorOptions};
    use crate::vector::distance::{norm_squared, norm_squared_wide};
    use crate::vector::{VectorQuantizationConfig, VectorQuantizationLayer};

    #[test]
    fn cosine_normalization_is_shared_across_encodings() {
        let mut config = VectorQuantizationConfig::materialize(
            "shared_cosine_query".to_string(),
            &VectorOptions::new(100, Metric::Cosine),
            vec![VectorQuantizationLayer { bits: 1, seed: 17 }],
        )
        .unwrap();
        let first_index = Arc::new(QuantizedIndexCtx::from_config(config.clone()).unwrap());
        config.layers[0].seed = 23;
        let second_index = Arc::new(QuantizedIndexCtx::from_config(config).unwrap());

        for scale in [0.0, 1.0, 1e-30, 1e20] {
            let raw: Vec<f32> = (0..100).map(|i| (i as f32 - 50.5) * scale).collect();
            let norm = norm_squared_wide(&raw).sqrt();
            let expected: Vec<_> = if norm == 0.0 {
                vec![0.0; raw.len()]
            } else {
                raw.iter()
                    .map(|value| value * (1.0 / norm) as f32)
                    .collect()
            };
            let query = VectorQuery::new(Metric::Cosine, Arc::new(raw.clone()));
            assert_eq!(query.norm(), norm as f32);
            assert_eq!(query.routing_norm(), norm_squared(&raw).sqrt());
            assert_eq!(query.estimate_vector(), expected);

            for index in [&first_index, &second_index] {
                let encoded = QuantizedQueryCtx::new(Arc::clone(index), &query);
                assert!(std::ptr::eq(
                    encoded.estimate_vector(&query),
                    query.estimate_vector()
                ));
                assert_eq!(
                    encoded.score_query_norm(0.0),
                    norm_squared(&expected).sqrt()
                );
            }
        }
    }

    #[test]
    fn quantized_query_is_prepared_only_for_matching_index_configs() {
        let mut config = VectorQuantizationConfig::materialize(
            "prepared_for_embedding".to_string(),
            &VectorOptions::new(100, Metric::Dot),
            vec![VectorQuantizationLayer {
                bits: 1,
                seed: 0xfeed_2001,
            }],
        )
        .unwrap();
        let first_index = Arc::new(QuantizedIndexCtx::from_config(config.clone()).unwrap());
        let query = QuantizedQueryCtx::new(
            Arc::clone(&first_index),
            &VectorQuery::new(Metric::Dot, Arc::new(vec![0.5_f32; 100])),
        );
        assert!(Arc::ptr_eq(&query.index.meta, &first_index.meta));

        let other_segment = Arc::new(QuantizedIndexCtx::from_config(config.clone()).unwrap());
        assert_eq!(query.index.meta.to_bytes(), other_segment.meta.to_bytes());

        config.layers[0].seed = 0xfeed_2002;
        let reseeded_index = Arc::new(QuantizedIndexCtx::from_config(config).unwrap());
        assert_ne!(query.index.meta.to_bytes(), reseeded_index.meta.to_bytes());
    }

    #[test]
    fn quantized_index_context_requires_persisted_grid_and_rho() {
        let config = VectorQuantizationConfig::materialize(
            "strict_persisted_model".to_string(),
            &VectorOptions::new(100, Metric::Dot),
            vec![VectorQuantizationLayer {
                bits: 1,
                seed: 0xfeed_4001,
            }],
        )
        .unwrap();

        let mut missing_grid = config.clone();
        missing_grid.grids.clear();
        let error = QuantizedIndexCtx::from_config(missing_grid)
            .err()
            .expect("missing grid must be rejected");
        assert!(error.to_string().contains("grid"));
    }

    #[test]
    fn query_error_is_exact_for_sign_layers_and_zero_for_grid_luts() {
        let config = VectorQuantizationConfig::materialize(
            "query_error_by_kernel".to_string(),
            &VectorOptions::new(100, Metric::Dot),
            vec![
                VectorQuantizationLayer { bits: 1, seed: 1 },
                VectorQuantizationLayer { bits: 4, seed: 2 },
            ],
        )
        .unwrap();
        let query_values = (0..100)
            .map(|coordinate| ((coordinate as f32 + 0.25) * 0.173).sin())
            .collect::<Vec<_>>();
        let index = Arc::new(QuantizedIndexCtx::from_config(config).unwrap());
        let expected = quant::cascade::audit_split_query_layer_error_squared_with_plan(
            &query_values,
            &index.rotation_plan,
            &index.grids,
            4,
        );
        let query = QuantizedQueryCtx::new(
            index,
            &VectorQuery::new(Metric::Dot, Arc::new(query_values)),
        );

        assert_eq!(
            query.query_error_squared(0).to_bits(),
            expected[0].to_bits()
        );
        assert!(query.query_error_squared(0) > 0.0);
        assert_eq!(query.query_error_squared(1), 0.0);
    }
}
