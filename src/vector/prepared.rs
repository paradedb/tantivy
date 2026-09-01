//! Per-query precomputation hoisted out of the per-doc scoring loop.
//!
//! The exact-query state is built per segment backend. Quantized rotations are
//! process-cached with the index configuration, while bitplanes and LUTs are
//! built once per collector query and shared by every segment backend.
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

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cascade::{prepare_split_query_with_plan, LayerSpec, PreparedSplitQuery, QueryRotationPlan};
use quant_model::Grid;

use super::distance::{dot_bytes, l2_squared_bytes, norm_squared_wide};
use super::quantization::{VectorQuantizationConfig, SIGN_QUERY_BITS};
use super::VectorElement;
use crate::schema::Metric;
use crate::TantivyError;

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
        // Seed expansion and permutation construction are index-scoped. The
        // process cache retains this plan across both segments and queries.
        let rotation_plan = QueryRotationPlan::new(config.dim, &specs);
        Ok(Self {
            config,
            specs,
            grids,
            rotation_plan,
        })
    }

    /// Resolve immutable scorer state once per persisted configuration and
    /// reuse it across SegmentReader lifetimes in the same backend process.
    /// SegmentReader's field cache provides the first level; this strong
    /// process cache closes the pg_search query boundary, which reopens
    /// segment readers. Entries intentionally live for the process lifetime.
    pub(crate) fn resolve(config: VectorQuantizationConfig) -> crate::Result<Arc<Self>> {
        // Calibration is diagnostic metadata, not scorer state. Keep it out
        // of both context construction and cache identity.
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

/// Collector-scoped cache of prepared quantized queries. A collector may be
/// reused against another index with the same schema, so the process-cached
/// index-context identity and active prefix are both part of the key. Keeping
/// the context in the value makes the pointer identity stable for the entry's
/// lifetime.
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

/// Immutable query rotations, sign bitplanes, and LUTs shared by every
/// segment using the same collector, resolved index context, and prefix.
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

    /// Exact squared query-quantization error `B_j` already accumulated while
    /// preparing this layer's scoring state.
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

    /// Enter the resolved monomorphic kernel once for a fixed-stride batch.
    /// Scale, sigma, and accumulated-score assembly stay in the caller so the
    /// survivor loop can fuse those operations directly into its candidate
    /// buffer.
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

    /// Norm of the query vector used by a layer's dot-product error model.
    /// L2 estimates `<q-c,r>`; Dot and Cosine estimate `<q,r>` directly.
    /// The routing score already carries exact `-||q-c||²` for L2, so no
    /// centroid row is needed on the scan path for any metric.
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{QuantizedIndexCtx, QuantizedQueryCache, QuantizedQueryCtx};
    use crate::schema::{Metric, VectorOptions};
    use crate::vector::{
        VectorQuantizationCalibrationSource, VectorQuantizationConfig,
        VectorQuantizationDepthCalibration, VectorQuantizationLayer,
    };

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
        assert!(config.calibration().is_none());
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
    fn diagnostic_calibration_does_not_change_query_scoring() {
        let config = VectorQuantizationConfig::materialize(
            "diagnostic_calibration".to_string(),
            &VectorOptions::new(100, Metric::Dot),
            vec![VectorQuantizationLayer {
                bits: 1,
                seed: 0xfeed_5001,
            }],
        )
        .unwrap();
        let mut diagnostic = config.clone();
        diagnostic
            .install_real_query_calibration(vec![VectorQuantizationDepthCalibration {
                bias: -7.5,
                spread: 11.25,
                sample_count: 1_000,
                source: VectorQuantizationCalibrationSource::RealQuery,
                protocol: "REAL_QUERY_EXACT_E_BQ4".to_string(),
            }])
            .unwrap();
        let query_values = (0..100)
            .map(|coordinate| ((coordinate as f32 + 0.75) * 0.113).cos())
            .collect::<Vec<_>>();
        let uncalibrated_index = QuantizedIndexCtx::resolve(config).unwrap();
        let diagnostic_index = QuantizedIndexCtx::resolve(diagnostic).unwrap();
        assert!(Arc::ptr_eq(&uncalibrated_index, &diagnostic_index));
        let uncalibrated = QuantizedQueryCtx::new(uncalibrated_index, query_values.clone());
        let calibrated = QuantizedQueryCtx::new(diagnostic_index, query_values);
        let mut codes = vec![0; 16];
        codes[..8].fill(0xa5);

        assert_eq!(
            uncalibrated
                .score_layer(0, &codes, 1.0, None)
                .unwrap()
                .to_bits(),
            calibrated
                .score_layer(0, &codes, 1.0, None)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            uncalibrated.query_error_squared(0).to_bits(),
            calibrated.query_error_squared(0).to_bits()
        );
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
