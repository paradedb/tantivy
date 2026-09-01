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

use cascade::{prepare_split_query, LayerSpec, PreparedSplitQuery};
use quant_model::f16::f16_to_f32;
use quant_model::{build_grid, Grid, DEFAULT_CAL};

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
}

impl QuantizedIndexCtx {
    pub(crate) fn new(config: VectorQuantizationConfig) -> Self {
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
                let modeled = build_grid(config.dim, layer.bits);
                match layer.quantizer {
                    VectorQuantizer::RaBitQ => modeled,
                    VectorQuantizer::TurboQuant => {
                        let stored = config
                            .grids
                            .iter()
                            .find(|grid| grid.bits == layer.bits)
                            .expect("validated grid must exist");
                        Grid {
                            bits: layer.bits,
                            points: stored.points.clone(),
                            rho_model: modeled.rho_model,
                        }
                    }
                }
            })
            .collect();
        Self {
            config,
            specs,
            grids,
        }
    }
}

/// Immutable per-segment-query rotations, sign bitplanes, and LUTs.
pub(crate) struct QuantizedQueryCtx {
    pub(crate) index: Arc<QuantizedIndexCtx>,
    prepared: PreparedSplitQuery,
    query: Vec<f32>,
    query_norm_sq: f32,
}

impl QuantizedQueryCtx {
    pub(crate) fn new(index: Arc<QuantizedIndexCtx>, mut query: Vec<f32>) -> Self {
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
        let prepared = prepare_split_query(&query, &index.specs, &index.grids, 4);
        Self {
            index,
            prepared,
            query,
            query_norm_sq,
        }
    }

    pub(crate) fn score_layer(&self, layer: usize, codes: &[u8], scale: u16, constant: f32) -> f32 {
        self.prepared
            .score_layer(layer, codes, scale, constant, self.index.specs[layer])
    }

    /// `||q-c||`, using the routing score where its metric makes that exact.
    pub(crate) fn query_residual_norm(&self, routing_score: f32, centroid: &[f32]) -> f32 {
        match self.index.config.metric {
            Metric::L2 => (-routing_score).max(0.0).sqrt(),
            Metric::Cosine => {
                let centroid_norm_sq = norm_squared_wide(centroid) as f32;
                (self.query_norm_sq + centroid_norm_sq - 2.0 * routing_score)
                    .max(0.0)
                    .sqrt()
            }
            Metric::Dot => {
                let centroid_norm_sq = norm_squared_wide(centroid) as f32;
                (self.query_norm_sq + centroid_norm_sq - 2.0 * routing_score)
                    .max(0.0)
                    .sqrt()
            }
        }
    }

    /// Norm of the query vector used by a layer's dot-product error model.
    /// L2 estimates `<q-c,r>`; Dot and Cosine estimate `<q,r>` directly.
    pub(crate) fn score_query_norm(&self, routing_score: f32, centroid: &[f32]) -> f32 {
        if self.index.config.metric == Metric::L2 {
            self.query_residual_norm(routing_score, centroid)
        } else {
            self.query_norm_sq.sqrt()
        }
    }

    /// Score-space uncertainty. L2's score contains `2 * est`, hence `2σ_dot`.
    pub(crate) fn score_sigma(&self, layer: usize, query_residual_norm: f32, scale: u16) -> f32 {
        score_sigma_from_scale(
            self.index.config.metric,
            self.index.grids[layer].rho_model,
            query_residual_norm,
            scale,
        )
    }

    pub(crate) fn query(&self) -> &[f32] {
        &self.query
    }
}

/// Convert one layer's stored reconstruction scale into score-space uncertainty.
/// The scale belongs to the residual entering that layer, so the layer-local rho
/// already represents all error left after that boundary.
fn score_sigma_from_scale(metric: Metric, rho: f64, query_residual_norm: f32, scale: u16) -> f32 {
    let dot_sigma = f16_to_f32(scale) * rho as f32 * DEFAULT_CAL as f32 * query_residual_norm;
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
    use quant_model::f16::{f16_to_f32, f32_to_f16};
    use quant_model::DEFAULT_CAL;

    use super::score_sigma_from_scale;
    use crate::schema::Metric;

    #[test]
    fn quantized_sigma_decodes_f16_scale_and_applies_rho_and_cal() {
        let scale_bits = f32_to_f16(0.037_531);
        let rho = 0.097_3;
        let query_residual_norm = 0.812_5;
        let expected =
            f16_to_f32(scale_bits) * rho as f32 * DEFAULT_CAL as f32 * query_residual_norm;

        let dot = score_sigma_from_scale(Metric::Dot, rho, query_residual_norm, scale_bits);
        let cosine = score_sigma_from_scale(Metric::Cosine, rho, query_residual_norm, scale_bits);
        let l2 = score_sigma_from_scale(Metric::L2, rho, query_residual_norm, scale_bits);

        assert_eq!(dot.to_bits(), expected.to_bits());
        assert_eq!(cosine.to_bits(), expected.to_bits());
        assert_eq!(l2.to_bits(), (2.0 * expected).to_bits());
        assert!(cosine.is_finite());
    }
}
