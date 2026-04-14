//! Query-time distance estimation using RaBitQ quantized records.
//!
//! Precompute query constants once, then estimate distance per document
//! using only the quantized record (binary code + correction scalars).

use super::quantizer::QuantizedVector;
use super::rotation::DynamicRotator;
use super::{simd, Metric};

/// Precomputed query state for efficient distance estimation.
///
/// Build once per query via [`RaBitQQuery::new`], then call
/// [`estimate_distance`](Self::estimate_distance) per candidate document.
pub struct RaBitQQuery {
    /// Rotated query vector (full precision).
    rotated_query: Vec<f32>,
    /// Precomputed: c1 * sum(rotated_query), where c1 = -0.5.
    k1x_sum_q: f32,
    /// Precomputed: cb * sum(rotated_query), where cb = -(2^ex_bits - 0.5).
    kbx_sum_q: f32,
    /// 2^ex_bits.
    binary_scale: f32,
    /// ex_bits configuration.
    ex_bits: usize,
    /// The distance metric.
    metric: Metric,
}

impl RaBitQQuery {
    /// Prepare a query for distance estimation.
    ///
    /// Rotates the query vector and precomputes constants.
    pub fn new(query: &[f32], rotator: &DynamicRotator, ex_bits: usize, metric: Metric) -> Self {
        let rotated_query = rotator.rotate(query);
        let sum_query: f32 = rotated_query.iter().sum();
        let c1 = -0.5f32;
        let cb = -((1u32 << ex_bits) as f32 - 0.5);

        Self {
            rotated_query,
            k1x_sum_q: c1 * sum_query,
            kbx_sum_q: cb * sum_query,
            binary_scale: (1u32 << ex_bits) as f32,
            ex_bits,
            metric,
        }
    }

    /// Estimate the distance between the query and a quantized document.
    ///
    /// For L2 metric, returns approximate squared L2 distance (lower = closer).
    /// For InnerProduct, returns negative inner product (lower = more similar).
    ///
    /// `g_add` is the centroid correction term. For brute-force (zero centroid),
    /// pass `0.0`.
    pub fn estimate_distance(&self, qv: &QuantizedVector, g_add: f32) -> f32 {
        // Stage 1: binary code distance
        let binary_code = qv.unpack_binary_code();
        let mut binary_dot = 0.0f32;
        for (&bit, &q_val) in binary_code.iter().zip(self.rotated_query.iter()) {
            binary_dot += (bit as f32) * q_val;
        }
        let binary_term = binary_dot + self.k1x_sum_q;
        let mut distance = qv.f_add + g_add + qv.f_rescale * binary_term;

        // Stage 2: extended code refinement (if ex_bits > 0)
        if self.ex_bits > 0 && !qv.ex_code_packed.is_empty() {
            let ex_code = qv.unpack_ex_code();
            let mut ex_dot = 0.0f32;
            for (&code, &q_val) in ex_code.iter().zip(self.rotated_query.iter()) {
                ex_dot += (code as f32) * q_val;
            }
            let total_term = self.binary_scale * binary_dot + ex_dot + self.kbx_sum_q;
            distance = qv.f_add_ex + g_add + qv.f_rescale_ex * total_term;
        }

        match self.metric {
            Metric::L2 => distance,
            Metric::InnerProduct => -distance,
        }
    }

    /// Estimate distance from a packed byte record.
    ///
    /// `g_add` is the centroid correction term (query-to-centroid distance for
    /// centroid-residual quantization, or `0.0` for zero-centroid encoding).
    pub fn estimate_distance_from_record(
        &self,
        record: &[u8],
        padded_dims: usize,
        g_add: f32,
    ) -> f32 {
        let qv = super::record::unpack(record, padded_dims, self.ex_bits);
        self.estimate_distance(&qv, g_add)
    }

    /// Two-stage distance estimation with lower-bound pruning.
    ///
    /// Stage 1: compute distance using only 1-bit binary codes (cheap).
    /// If `stage1_distance - f_error >= threshold`, the vector can't beat
    /// the current k-th best, so return `None` (pruned).
    /// Stage 2: refine with extended codes (expensive, only for survivors).
    ///
    /// `threshold` is the raw distance threshold (not negated). For L2,
    /// a vector is pruned if its lower bound >= threshold.
    pub fn estimate_distance_pruned(
        &self,
        record: &[u8],
        padded_dims: usize,
        g_add: f32,
        threshold: f32,
    ) -> Option<f32> {
        let binary_bytes = padded_dims.div_ceil(8);
        let ex_bytes = super::record::ex_bytes(padded_dims, self.ex_bits);

        // Read binary code + scalars (skip extended codes for now)
        let binary_code_packed = &record[..binary_bytes];
        let scalar_offset = binary_bytes + ex_bytes;
        let read_f32 = |off: usize| -> f32 {
            f32::from_le_bytes([
                record[scalar_offset + off],
                record[scalar_offset + off + 1],
                record[scalar_offset + off + 2],
                record[scalar_offset + off + 3],
            ])
        };
        let f_add = read_f32(8);     // 3rd scalar (after delta, vl)
        let f_rescale = read_f32(12); // 4th scalar
        let f_error = read_f32(16);   // 5th scalar

        // Stage 1: binary-only distance
        let binary_code = super::simd::unpack_binary_code_from_packed(binary_code_packed, padded_dims);
        let mut binary_dot = 0.0f32;
        for (&bit, &q_val) in binary_code.iter().zip(self.rotated_query.iter()) {
            binary_dot += (bit as f32) * q_val;
        }
        let binary_term = binary_dot + self.k1x_sum_q;
        let stage1_distance = f_add + g_add + f_rescale * binary_term;

        // Lower-bound check: if lower bound >= threshold, prune
        let lower_bound = stage1_distance - f_error.abs();
        if lower_bound >= threshold {
            return None;
        }

        // Stage 2: full estimation with extended codes
        if self.ex_bits > 0 {
            let qv = super::record::unpack(record, padded_dims, self.ex_bits);
            let distance = self.estimate_distance(&qv, g_add);
            Some(distance)
        } else {
            Some(stage1_distance)
        }
    }

    /// Access the rotated query vector.
    pub fn rotated_query(&self) -> &[f32] {
        &self.rotated_query
    }
}

// Suppress unused import warning — simd is used indirectly via QuantizedVector::unpack_*
#[allow(unused_imports)]
use super::simd as _simd_used;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::rabitq::quantizer::{quantize_with_centroid, RabitqConfig};
    use crate::vector::rabitq::rotation::DynamicRotator;
    use crate::vector::rabitq::RotatorType;

    #[test]
    fn test_distance_estimation_self_is_minimal() {
        // A vector's estimated distance to its own quantized form should be
        // the smallest (most negative) among all candidates.
        let dims = 64;
        let rotator = DynamicRotator::new(dims, RotatorType::MatrixRotator, 42);
        let padded = rotator.padded_dim();
        let config = RabitqConfig::new(1);
        let zero_centroid = vec![0.0f32; padded];

        let vector: Vec<f32> = (0..dims).map(|i| (i as f32) / dims as f32).collect();
        let other: Vec<f32> = (0..dims).map(|i| -((i as f32) / dims as f32)).collect();

        let qv_self = quantize_with_centroid(
            &rotator.rotate(&vector),
            &zero_centroid,
            &config,
            Metric::L2,
        );
        let qv_other =
            quantize_with_centroid(&rotator.rotate(&other), &zero_centroid, &config, Metric::L2);

        let query = RaBitQQuery::new(&vector, &rotator, 0, Metric::L2);
        let dist_self = query.estimate_distance(&qv_self, 0.0);
        let dist_other = query.estimate_distance(&qv_other, 0.0);

        assert!(
            dist_self < dist_other,
            "self should be closer: {dist_self} vs {dist_other}"
        );
    }

    #[test]
    fn test_distance_ordering() {
        // A query should be closer to a similar vector than a dissimilar one
        let dims = 64;
        let rotator = DynamicRotator::new(dims, RotatorType::MatrixRotator, 42);
        let padded = rotator.padded_dim();
        let config = RabitqConfig::new(1);
        let zero_centroid = vec![0.0f32; padded];

        let query_vec: Vec<f32> = (0..dims).map(|i| (i as f32) / dims as f32).collect();
        let similar_vec: Vec<f32> = (0..dims).map(|i| (i as f32) / dims as f32 + 0.01).collect();
        let dissimilar_vec: Vec<f32> = (0..dims).map(|i| -((i as f32) / dims as f32)).collect();

        let qv_similar = quantize_with_centroid(
            &rotator.rotate(&similar_vec),
            &zero_centroid,
            &config,
            Metric::L2,
        );
        let qv_dissimilar = quantize_with_centroid(
            &rotator.rotate(&dissimilar_vec),
            &zero_centroid,
            &config,
            Metric::L2,
        );

        let query = RaBitQQuery::new(&query_vec, &rotator, 0, Metric::L2);
        let dist_similar = query.estimate_distance(&qv_similar, 0.0);
        let dist_dissimilar = query.estimate_distance(&qv_dissimilar, 0.0);

        assert!(
            dist_similar < dist_dissimilar,
            "similar vector should be closer: {dist_similar} vs {dist_dissimilar}"
        );
    }
}
