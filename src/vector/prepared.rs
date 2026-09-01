//! Prepared exact and quantized query state.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cascade::{prepare_split_query_with_plan, LayerSpec, PreparedSplitQuery, QueryRotationPlan};
use quant_model::Grid;

use super::distance::{dot_bytes, l2_squared_bytes, norm_squared_wide};
use super::quantization::{VectorQuantizationConfig, GAMMA_ANALYTICAL_SAFETY, SIGN_QUERY_BITS};
use super::VectorElement;
use crate::schema::Metric;
use crate::TantivyError;

/// Metric-specific prepared vector query.
pub struct PreparedQuery<T: VectorElement> {
    query: Arc<Vec<T>>,
    kind: QueryKind,
}

/// Metric-specific per-query state.
enum QueryKind {
    L2,
    Dot,
    Cosine {
        /// Reciprocal query norm, or zero for a degenerate query.
        inv_norm_q: f32,
    },
}

/// Immutable field/segment quantization state resolved once before query prep.
pub(crate) struct QuantizedIndexCtx {
    pub(crate) config: VectorQuantizationConfig,
    pub(crate) specs: Vec<LayerSpec>,
    pub(crate) grids: Vec<Grid>,
    rotation_plan: QueryRotationPlan,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct QuantizedIndexCacheKey {
    config_json: String,
}

static QUANTIZED_INDEX_CACHE: OnceLock<
    Mutex<HashMap<QuantizedIndexCacheKey, Arc<QuantizedIndexCtx>>>,
> = OnceLock::new();

/// Applies the metric-specific correction to a cumulative quantized estimate.
#[inline(always)]
pub(crate) fn corrected_quantized_estimate(
    metric: Metric,
    gamma: f32,
    raw_prefix: f32,
    base: f32,
) -> f32 {
    let metric_factor = if metric == Metric::L2 { 2.0 } else { 1.0 };
    (metric_factor * gamma).mul_add(raw_prefix, base)
}

/// Combines the first L2 layer's unscaled kernel output.
#[inline(always)]
pub(crate) fn initial_l2_raw_prefix(kernel_score: f32, scale: f32, constant: f32) -> f32 {
    scale.mul_add(kernel_score, -constant)
}

/// Combines the first dot-like layer's unscaled kernel output.
#[inline(always)]
pub(crate) fn initial_dot_raw_prefix(kernel_score: f32, scale: f32) -> f32 {
    kernel_score * scale
}

/// Adds an L2 refinement to the cumulative raw prefix.
#[inline(always)]
pub(crate) fn refine_l2_raw_prefix(
    raw_prefix: f32,
    kernel_score: f32,
    scale: f32,
    constant: f32,
) -> f32 {
    scale.mul_add(kernel_score, raw_prefix - constant)
}

/// Adds a dot-like refinement to the cumulative raw prefix.
#[inline(always)]
pub(crate) fn refine_dot_raw_prefix(raw_prefix: f32, kernel_score: f32, scale: f32) -> f32 {
    scale.mul_add(kernel_score, raw_prefix)
}

/// Computes the production uncertainty width for a corrected estimate.
#[inline(always)]
pub(crate) fn quantized_model_sigma(
    metric: Metric,
    dimension: usize,
    residual_norm_squared: f32,
    corrected_error_ratio: f32,
    gamma: f32,
    score_query_norm_squared: f32,
    sign_query_error_term: f32,
) -> f32 {
    debug_assert_ne!(dimension, 0);
    let data_variance = residual_norm_squared
        * (1.0 / dimension as f32)
        * corrected_error_ratio
        * score_query_norm_squared;
    let query_variance = gamma * gamma * sign_query_error_term;
    let metric_factor = if metric == Metric::L2 { 2.0 } else { 1.0 };
    metric_factor * GAMMA_ANALYTICAL_SAFETY * (data_variance + query_variance).sqrt()
}

impl QuantizedIndexCtx {
    fn new(config: VectorQuantizationConfig) -> crate::Result<Self> {
        let specs: Vec<LayerSpec> = config
            .layers
            .iter()
            .map(|layer| LayerSpec {
                bits: layer.bits,
                seed: layer.seed,
                rotate: true,
            })
            .collect();
        let grids: Vec<Grid> = config
            .layers
            .iter()
            .map(|layer| {
                let stored = config
                    .grids
                    .iter()
                    .find(|grid| grid.bits == layer.bits)
                    .ok_or_else(|| {
                        TantivyError::InvalidArgument(format!(
                            "quantization field {:?} layer width {} has no persisted grid/model \
                             entry; rebuild required",
                            config.field, layer.bits
                        ))
                    })?;
                Ok(Grid {
                    bits: layer.bits,
                    points: stored.points.clone(),
                    rho_model: stored.rho_model,
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let rotation_plan = QueryRotationPlan::new(config.dim, &specs);
        Ok(Self {
            config,
            specs,
            grids,
            rotation_plan,
        })
    }

    /// Resolves process-cached scorer state for one persisted configuration.
    pub(crate) fn resolve(config: VectorQuantizationConfig) -> crate::Result<Arc<Self>> {
        let runtime_config = (
            config.field.as_str(),
            config.format_version,
            config.dim,
            config.metric,
            config.norm_policy,
            config.layers.as_slice(),
            config.grids.as_slice(),
        );
        let key = QuantizedIndexCacheKey {
            config_json: serde_json::to_string(&runtime_config).map_err(|error| {
                TantivyError::InternalError(format!(
                    "vector quantization config failed to serialize for context caching: {error}"
                ))
            })?,
        };
        let cache = QUANTIZED_INDEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().expect("quantized index cache lock poisoned");
        if let Some(resolved) = cache.get(&key) {
            return Ok(Arc::clone(resolved));
        }
        let resolved = Arc::new(Self::new(config)?);
        cache.insert(key, Arc::clone(&resolved));
        Ok(resolved)
    }

    pub(crate) fn resolve_from_config(
        config: VectorQuantizationConfig,
    ) -> crate::Result<Arc<Self>> {
        Self::resolve(config)
    }
}

/// Collector-scoped cache of prepared quantized queries.
#[derive(Default)]
pub(crate) struct QuantizedQueryCache {
    queries: Mutex<HashMap<(usize, usize), Arc<QuantizedQueryCtx>>>,
}

impl QuantizedQueryCache {
    pub(crate) fn resolve<T: VectorElement>(
        &self,
        index: Arc<QuantizedIndexCtx>,
        query: &[T],
        active_layers: usize,
    ) -> Arc<QuantizedQueryCtx> {
        assert!((1..=index.specs.len()).contains(&active_layers));
        let index_identity = Arc::as_ptr(&index) as usize;
        let key = (index_identity, active_layers);
        let mut queries = self
            .queries
            .lock()
            .expect("quantized query cache lock poisoned");
        if let Some(prepared) = queries.get(&key) {
            return Arc::clone(prepared);
        }
        let query_f32 = query.iter().map(|value| value.to_f32()).collect();
        let prepared = Arc::new(QuantizedQueryCtx::with_depth(
            index,
            query_f32,
            active_layers,
        ));
        queries.insert(key, Arc::clone(&prepared));
        prepared
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.queries.lock().unwrap().len()
    }
}

/// Immutable query rotations, sign bitplanes, and LUTs shared across segments.
pub(crate) struct QuantizedQueryCtx {
    pub(crate) index: Arc<QuantizedIndexCtx>,
    prepared: PreparedSplitQuery,
    query: Vec<f32>,
    query_norm_sq: f32,
    active_layers: usize,
}

impl QuantizedQueryCtx {
    pub(crate) fn new(index: Arc<QuantizedIndexCtx>, query: Vec<f32>) -> Self {
        let active_layers = index.specs.len();
        Self::with_depth(index, query, active_layers)
    }

    pub(crate) fn with_depth(
        index: Arc<QuantizedIndexCtx>,
        mut query: Vec<f32>,
        active_layers: usize,
    ) -> Self {
        assert!((1..=index.specs.len()).contains(&active_layers));
        if index.config.metric == Metric::Cosine {
            let norm = norm_squared_wide(&query).sqrt();
            if norm != 0.0 && norm.is_finite() {
                let inv = (1.0 / norm) as f32;
                for value in &mut query {
                    *value *= inv;
                }
            } else {
                query.fill(0.0);
            }
        }
        let query_norm_sq = norm_squared_wide(&query) as f32;
        let prepared = prepare_split_query_with_plan(
            &query,
            &index.rotation_plan,
            &index.grids[..active_layers],
            SIGN_QUERY_BITS,
        );
        Self {
            index,
            prepared,
            query,
            query_norm_sq,
            active_layers,
        }
    }

    pub(crate) fn active_layers(&self) -> usize {
        self.active_layers
    }

    /// Squared query-quantization error accumulated through this layer.
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
        match (self.index.config.metric, constant) {
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

    /// Scores a fixed-stride code batch with the resolved layer kernel.
    #[inline(always)]
    pub(crate) fn score_layer_batch_unscaled(
        &self,
        layer: usize,
        codes: &[u8],
        code_stride: usize,
        out: &mut [f32],
    ) {
        self.prepared.score_layer_batch_unscaled(
            layer,
            codes,
            code_stride,
            self.index.specs[layer],
            out,
        );
    }

    #[inline(always)]
    pub(crate) fn score_layer_batch_unscaled_indexed(
        &self,
        layer: usize,
        codes: &[u8],
        code_stride: usize,
        row_offsets: &[usize],
        out: &mut [f32],
    ) {
        self.prepared.score_layer_batch_unscaled_indexed(
            layer,
            codes,
            code_stride,
            row_offsets,
            self.index.specs[layer],
            out,
        );
    }

    /// Returns the query-vector norm used by the layer error model.
    pub(crate) fn score_query_norm(&self, routing_score: f32) -> f32 {
        if self.index.config.metric == Metric::L2 {
            (-routing_score).max(0.0).sqrt()
        } else {
            self.query_norm_sq.sqrt()
        }
    }

    pub(crate) fn query(&self) -> &[f32] {
        &self.query
    }
}

impl<T: VectorElement> PreparedQuery<T> {
    /// Prepares a query for one metric.
    pub fn new(metric: Metric, query: Arc<Vec<T>>) -> Self {
        let kind = match metric {
            Metric::L2 => QueryKind::L2,
            Metric::Dot => QueryKind::Dot,
            Metric::Cosine => {
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

    /// Returns the query metric.
    pub fn metric(&self) -> Metric {
        match self.kind {
            QueryKind::L2 => Metric::L2,
            QueryKind::Dot => Metric::Dot,
            QueryKind::Cosine { .. } => Metric::Cosine,
        }
    }

    /// Returns the query coordinates.
    pub fn query(&self) -> &[T] {
        &self.query
    }

    /// Scores a stored vector row.
    #[inline]
    pub fn score_doc_bytes(&self, doc_bytes: &[u8]) -> f32 {
        match self.kind {
            QueryKind::L2 => -l2_squared_bytes::<T>(&self.query, doc_bytes),
            QueryKind::Dot => dot_bytes::<T>(&self.query, doc_bytes),
            QueryKind::Cosine { inv_norm_q } => dot_bytes::<T>(&self.query, doc_bytes) * inv_norm_q,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{QuantizedIndexCtx, QuantizedQueryCache, QuantizedQueryCtx};
    use crate::schema::{Metric, VectorOptions};
    use crate::vector::{VectorQuantizationConfig, VectorQuantizationLayer};

    #[test]
    fn resolved_quantized_index_context_is_reused_across_segment_opens() {
        let config = VectorQuantizationConfig::materialize(
            "cache_reuse_embedding".to_string(),
            &VectorOptions::new(100, Metric::Dot),
            vec![
                VectorQuantizationLayer {
                    bits: 1,
                    seed: 0xfeed_0001,
                },
                VectorQuantizationLayer {
                    bits: 4,
                    seed: 0xfeed_0002,
                },
            ],
        )
        .unwrap();
        let first = QuantizedIndexCtx::resolve(config.clone()).unwrap();
        let reopened = QuantizedIndexCtx::resolve(config).unwrap();
        assert!(Arc::ptr_eq(&first, &reopened));
    }

    #[test]
    fn quantized_query_context_is_shared_by_index_and_prefix() {
        let config = VectorQuantizationConfig::materialize(
            "shared_query_embedding".to_string(),
            &VectorOptions::new(100, Metric::Dot),
            vec![
                VectorQuantizationLayer {
                    bits: 1,
                    seed: 0xfeed_1001,
                },
                VectorQuantizationLayer {
                    bits: 4,
                    seed: 0xfeed_1002,
                },
            ],
        )
        .unwrap();
        let index = QuantizedIndexCtx::resolve(config).unwrap();
        let cache = QuantizedQueryCache::default();
        let query = vec![0.25_f32; 100];

        let first_segment = cache.resolve(Arc::clone(&index), &query, 1);
        let second_segment = cache.resolve(Arc::clone(&index), &query, 1);
        assert!(Arc::ptr_eq(&first_segment, &second_segment));
        assert_eq!(first_segment.active_layers(), 1);

        let full_prefix = cache.resolve(index, &query, 2);
        assert!(!Arc::ptr_eq(&first_segment, &full_prefix));
        assert_eq!(full_prefix.active_layers(), 2);
        assert_eq!(cache.queries.lock().unwrap().len(), 2);
    }

    #[test]
    fn reused_collector_cache_does_not_cross_index_contexts() {
        let mut config = VectorQuantizationConfig::materialize(
            "first_query_index".to_string(),
            &VectorOptions::new(100, Metric::Dot),
            vec![VectorQuantizationLayer {
                bits: 1,
                seed: 0xfeed_2001,
            }],
        )
        .unwrap();
        let first_index = QuantizedIndexCtx::resolve(config.clone()).unwrap();
        config.field = "second_query_index".to_string();
        let second_index = QuantizedIndexCtx::resolve(config).unwrap();
        assert!(!Arc::ptr_eq(&first_index, &second_index));

        let cache = QuantizedQueryCache::default();
        let query = vec![0.5_f32; 100];
        let first = cache.resolve(first_index, &query, 1);
        let second = cache.resolve(second_index, &query, 1);
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(cache.queries.lock().unwrap().len(), 2);
    }

    #[test]
    fn concurrent_segments_share_one_quantized_query_context() {
        let config = VectorQuantizationConfig::materialize(
            "concurrent_query_embedding".to_string(),
            &VectorOptions::new(100, Metric::Dot),
            vec![
                VectorQuantizationLayer {
                    bits: 1,
                    seed: 0xfeed_3001,
                },
                VectorQuantizationLayer {
                    bits: 4,
                    seed: 0xfeed_3002,
                },
            ],
        )
        .unwrap();
        let index = QuantizedIndexCtx::resolve(config).unwrap();
        let cache = QuantizedQueryCache::default();
        let query = vec![0.75_f32; 100];

        let prepared = std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|_| {
                    let index = Arc::clone(&index);
                    let cache = &cache;
                    let query = &query;
                    scope.spawn(move || cache.resolve(index, query, 2))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert!(prepared
            .iter()
            .skip(1)
            .all(|other| Arc::ptr_eq(&prepared[0], other)));
        assert_eq!(cache.len(), 1);
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
        let error = QuantizedIndexCtx::resolve(missing_grid)
            .err()
            .expect("missing grid must be rejected");
        assert!(error.to_string().contains("no persisted grid/model entry"));
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
        let index = QuantizedIndexCtx::resolve(config).unwrap();
        let expected = cascade::audit_split_query_layer_error_squared_with_plan(
            &query_values,
            &index.rotation_plan,
            &index.grids,
            4,
        );
        let query = QuantizedQueryCtx::new(index, query_values);

        assert_eq!(
            query.query_error_squared(0).to_bits(),
            expected[0].to_bits()
        );
        assert!(query.query_error_squared(0) > 0.0);
        assert_eq!(query.query_error_squared(1), 0.0);
    }
}
