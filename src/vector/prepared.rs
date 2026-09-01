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
#[cfg(test)]
use quant_model::f16::f16_to_f32;
use quant_model::{exact_sign_grid, rho_model_for_points, Grid};

use super::distance::{dot_bytes, l2_squared_bytes, norm_squared_wide};
use super::quantization::{VectorQuantizationConfig, VectorQuantizer};
use super::VectorElement;
use crate::schema::Metric;

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
    biases: Vec<f32>,
    cals: Vec<f32>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct QuantizedIndexCacheKey {
    config_json: String,
    bias_bits: Vec<u32>,
    cal_bits: Vec<u32>,
}

static QUANTIZED_INDEX_CACHE: OnceLock<
    Mutex<HashMap<QuantizedIndexCacheKey, Arc<QuantizedIndexCtx>>>,
> = OnceLock::new();

impl QuantizedIndexCtx {
    pub(crate) fn calibrations(&self) -> &[f32] {
        &self.cals
    }

    pub(crate) fn biases(&self) -> &[f32] {
        &self.biases
    }

    pub(crate) fn new_with_biases_and_cals(
        config: VectorQuantizationConfig,
        biases: Vec<f32>,
        cals: Vec<f32>,
    ) -> Self {
        assert!(
            (biases.is_empty() && cals.is_empty())
                || (biases.len() == config.layers.len() && cals.len() == config.layers.len())
        );
        assert!(biases.iter().all(|bias| bias.is_finite()));
        assert!(cals.iter().all(|cal| cal.is_finite() && *cal >= 0.0));
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
                if let Some(stored) = config.grids.iter().find(|grid| grid.bits == layer.bits) {
                    Grid {
                        bits: layer.bits,
                        points: stored.points.clone(),
                        rho_model: stored.rho_model.unwrap_or_else(|| {
                            // Compatibility for pre-amendment V3 metadata:
                            // evaluate the persisted points once, without a
                            // Lloyd-Max solve. The resolved context is then
                            // cached process-wide across segment reopens.
                            rho_model_for_points(config.dim, &stored.points)
                        }),
                    }
                } else {
                    debug_assert_eq!(layer.quantizer, VectorQuantizer::RaBitQ);
                    debug_assert_eq!(layer.bits, 1);
                    exact_sign_grid(config.dim)
                }
            })
            .collect();
        // Seed expansion and permutation construction are index-scoped. The
        // process cache retains this plan across both segments and queries.
        let rotation_plan = QueryRotationPlan::new(config.dim, &specs);
        Self {
            config,
            specs,
            grids,
            rotation_plan,
            biases,
            cals,
        }
    }

    /// Resolve immutable scorer state once per persisted configuration and
    /// reuse it across SegmentReader lifetimes in the same backend process.
    /// SegmentReader's field cache provides the first level; this strong
    /// process cache closes the pg_search query boundary, which reopens
    /// segment readers. Entries intentionally live for the process lifetime.
    pub(crate) fn resolve(
        config: VectorQuantizationConfig,
        biases: Vec<f32>,
        cals: Vec<f32>,
    ) -> Arc<Self> {
        let key = QuantizedIndexCacheKey {
            config_json: serde_json::to_string(&config)
                .expect("vector quantization config must serialize"),
            bias_bits: biases.iter().map(|bias| bias.to_bits()).collect(),
            cal_bits: cals.iter().map(|cal| cal.to_bits()).collect(),
        };
        let cache = QUANTIZED_INDEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().expect("quantized index cache lock poisoned");
        if let Some(resolved) = cache.get(&key) {
            return Arc::clone(resolved);
        }
        let resolved = Arc::new(Self::new_with_biases_and_cals(config, biases, cals));
        cache.insert(key, Arc::clone(&resolved));
        resolved
    }

    pub(crate) fn resolve_from_config(config: VectorQuantizationConfig) -> Option<Arc<Self>> {
        let calibration = config.calibration()?;
        let biases = calibration.iter().map(|depth| depth.bias).collect();
        let spreads = calibration.iter().map(|depth| depth.spread).collect();
        Some(Self::resolve(config, biases, spreads))
    }

    /// An uncentered context used only to measure settings calibration. It is
    /// never exposed to the production scan fallback.
    pub(crate) fn for_calibration_measurement(config: VectorQuantizationConfig) -> Arc<Self> {
        Arc::new(Self::new_with_biases_and_cals(
            config,
            Vec::new(),
            Vec::new(),
        ))
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
        let prepared = Arc::new(QuantizedQueryCtx::new_with_depth(
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
        Self::new_with_depth(index, query, active_layers)
    }

    pub(crate) fn new_with_depth(
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
            4,
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

    pub(crate) fn score_layer(&self, layer: usize, codes: &[u8], scale: u16, constant: f32) -> f32 {
        self.prepared
            .score_layer(layer, codes, scale, constant, self.index.specs[layer])
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

    #[inline(always)]
    pub(crate) fn layer_sigma_factor(&self, layer: usize, query_norm: f32) -> f32 {
        self.index.grids[layer].rho_model as f32
            * self.index.cals[layer]
            * query_norm
            * if self.index.config.metric == Metric::L2 {
                2.0
            } else {
                1.0
            }
    }

    #[inline(always)]
    pub(crate) fn layer_bias_factor(&self, layer: usize, query_norm: f32) -> f32 {
        self.index.grids[layer].rho_model as f32
            * self.index.biases[layer]
            * query_norm
            * if self.index.config.metric == Metric::L2 {
                2.0
            } else {
                1.0
            }
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

/// Convert one layer's stored reconstruction scale into score-space uncertainty.
/// The scale belongs to the residual entering that layer, so the layer-local rho
/// already represents all error left after that boundary.
#[cfg(test)]
fn score_sigma_from_scale(
    metric: Metric,
    rho: f64,
    cal: f32,
    query_residual_norm: f32,
    scale: u16,
) -> f32 {
    let dot_sigma = f16_to_f32(scale) * rho as f32 * cal * query_residual_norm;
    if metric == Metric::L2 {
        2.0 * dot_sigma
    } else {
        dot_sigma
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

    use quant_model::f16::{f16_to_f32, f32_to_f16};
    use quant_model::DEFAULT_CAL;

    use super::{score_sigma_from_scale, QuantizedIndexCtx, QuantizedQueryCache};
    use crate::schema::{Metric, VectorOptions};
    use crate::vector::{VectorQuantizationConfig, VectorQuantizationLayer, VectorQuantizer};

    #[test]
    fn resolved_quantized_index_context_is_reused_across_segment_opens() {
        let config = VectorQuantizationConfig::materialize(
            "cache_reuse_embedding".to_string(),
            &VectorOptions::new(100, Metric::Dot),
            vec![
                VectorQuantizationLayer {
                    bits: 1,
                    quantizer: VectorQuantizer::RaBitQ,
                    seed: 0xfeed_0001,
                },
                VectorQuantizationLayer {
                    bits: 4,
                    quantizer: VectorQuantizer::TurboQuant,
                    seed: 0xfeed_0002,
                },
            ],
        )
        .unwrap();
        let first = QuantizedIndexCtx::resolve(config.clone(), vec![0.0, 0.0], vec![2.27, 2.31]);
        let reopened = QuantizedIndexCtx::resolve(config, vec![0.0, 0.0], vec![2.27, 2.31]);
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
                    quantizer: VectorQuantizer::RaBitQ,
                    seed: 0xfeed_1001,
                },
                VectorQuantizationLayer {
                    bits: 4,
                    quantizer: VectorQuantizer::TurboQuant,
                    seed: 0xfeed_1002,
                },
            ],
        )
        .unwrap();
        let index = QuantizedIndexCtx::resolve(config, vec![0.0, 0.0], vec![2.27, 2.31]);
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
                quantizer: VectorQuantizer::RaBitQ,
                seed: 0xfeed_2001,
            }],
        )
        .unwrap();
        let first_index = QuantizedIndexCtx::resolve(config.clone(), vec![0.0], vec![2.27]);
        config.field = "second_query_index".to_string();
        let second_index = QuantizedIndexCtx::resolve(config, vec![0.0], vec![2.27]);
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
                    quantizer: VectorQuantizer::RaBitQ,
                    seed: 0xfeed_3001,
                },
                VectorQuantizationLayer {
                    bits: 4,
                    quantizer: VectorQuantizer::TurboQuant,
                    seed: 0xfeed_3002,
                },
            ],
        )
        .unwrap();
        let index = QuantizedIndexCtx::resolve(config, vec![0.0, 0.0], vec![2.27, 2.31]);
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
    fn quantized_sigma_decodes_f16_scale_and_applies_rho_and_cal() {
        let scale_bits = f32_to_f16(0.037_531);
        let rho = 0.097_3;
        let query_residual_norm = 0.812_5;
        let expected =
            f16_to_f32(scale_bits) * rho as f32 * DEFAULT_CAL as f32 * query_residual_norm;

        let dot = score_sigma_from_scale(
            Metric::Dot,
            rho,
            DEFAULT_CAL as f32,
            query_residual_norm,
            scale_bits,
        );
        let cosine = score_sigma_from_scale(
            Metric::Cosine,
            rho,
            DEFAULT_CAL as f32,
            query_residual_norm,
            scale_bits,
        );
        let l2 = score_sigma_from_scale(
            Metric::L2,
            rho,
            DEFAULT_CAL as f32,
            query_residual_norm,
            scale_bits,
        );

        assert_eq!(dot.to_bits(), expected.to_bits());
        assert_eq!(cosine.to_bits(), expected.to_bits());
        assert_eq!(l2.to_bits(), (2.0 * expected).to_bits());
        assert!(cosine.is_finite());
    }

    #[test]
    fn quantized_sigma_selects_calibration_by_prefix_depth() {
        let config = VectorQuantizationConfig::materialize(
            "per_depth_calibration".to_string(),
            &VectorOptions::new(100, Metric::Dot),
            vec![
                VectorQuantizationLayer {
                    bits: 1,
                    quantizer: VectorQuantizer::RaBitQ,
                    seed: 1,
                },
                VectorQuantizationLayer {
                    bits: 4,
                    quantizer: VectorQuantizer::TurboQuant,
                    seed: 2,
                },
            ],
        )
        .unwrap();
        let query = super::QuantizedQueryCtx::new(
            Arc::new(QuantizedIndexCtx::new_with_biases_and_cals(
                config,
                vec![-1.5, 0.25],
                vec![3.5, 2.25],
            )),
            vec![0.1; 100],
        );
        assert_eq!(
            query.layer_sigma_factor(0, 1.0).to_bits(),
            (query.index.grids[0].rho_model as f32 * 3.5).to_bits()
        );
        assert_eq!(
            query.layer_sigma_factor(1, 1.0).to_bits(),
            (query.index.grids[1].rho_model as f32 * 2.25).to_bits()
        );
        assert_eq!(
            query.layer_bias_factor(0, 1.0).to_bits(),
            (query.index.grids[0].rho_model as f32 * -1.5).to_bits()
        );
        assert_eq!(
            query.layer_bias_factor(1, 1.0).to_bits(),
            (query.index.grids[1].rho_model as f32 * 0.25).to_bits()
        );
    }
}
