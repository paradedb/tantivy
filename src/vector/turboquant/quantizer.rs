//! TurboQuant encoder.
//!
//! Two-stage data-oblivious vector quantization. Encodes one vector at
//! a time into a fixed-size record; needs no centroid, no training, no
//! per-dataset state beyond a shared rotator + codebook.
//!
//! ## Stages
//!
//! **Stage 1 — MSE.** Rotate the vector with a Haar-like random
//! rotation (we reuse `DynamicRotator::FhtKacRotator` for O(d log d)
//! speed), then scalar-quantize each coordinate with a Lloyd-Max
//! codebook tuned for the coordinate's Beta marginal. Uses `bit_width
//! - 1` bits per coordinate when `bit_width > 1`, else Stage 1 is
//! skipped.
//!
//! **Stage 2 — QJL residual.** Compute the residual `r = rotated -
//! dequant(stage1)`, project it via a second independent random
//! orthogonal rotation, and store the sign bits together with the
//! residual norm γ. The query-time estimator recovers an unbiased
//! inner product using `sqrt(π/2)·γ·<signs, R_qjl·rotated_query>/d`.
//! 1 bit per coordinate.
//!
//! The paper uses a Gaussian matrix for the QJL projection. We use
//! the same FhtKacRotator family with a distinct seed: it is
//! orthogonal (not Gaussian) but the preceding Stage 1 rotation has
//! already concentrated each coordinate onto a Gaussian-like marginal,
//! so SRHT here behaves well in practice. Swap-in of a true Gaussian
//! projection is a one-type change in `QjlProjection` if precision
//! concerns arise later.
//!
//! ## Input contract
//!
//! The codebook is tuned for **unit-norm** input — callers are
//! responsible for normalising vectors before `encode` when that
//! matches their metric (e.g. cosine). Non-unit inputs still encode
//! cleanly but waste bits; scalar indices saturate at the extreme
//! centroids and Stage 2 compensates with a larger γ.

use std::sync::Arc;

use super::bitpack;
use super::codebook::{get_or_generate_cached, Codebook};
use super::record::{bytes_per_record, norm_offset, stage2_offset, write_norm};
use crate::vector::rotation::{DynamicRotator, RotatorType};

/// A configured TurboQuant encoder/decoder.
///
/// Cheap to clone (all state is behind `Arc`).
#[derive(Clone)]
pub struct TurboQuantizer {
    /// Input dimensionality.
    pub dim: usize,
    /// Rotator-padded dimensionality (always `>= dim`).
    pub padded_dim: usize,
    /// Total bits per coordinate: Stage 1 uses `bit_width - 1`, Stage 2 uses 1.
    pub bit_width: u8,
    /// Stage 1 Haar-like rotation (applied to the raw input vector).
    rotator: Arc<DynamicRotator>,
    /// Stage 2 QJL projection (applied to the Stage 1 residual).
    qjl_rotator: Arc<DynamicRotator>,
    /// Scalar codebook for Stage 1.
    codebook: Arc<Codebook>,
}

/// Default Stage-1 + Stage-2 bit width when callers don't specify.
/// Stage 1 uses `DEFAULT_BIT_WIDTH - 1` bits (= 3) and Stage 2 uses 1.
pub const DEFAULT_BIT_WIDTH: u8 = 4;

/// Default rotator seed when callers don't specify. The Stage 2 QJL
/// rotation derives its own seed from this via `wrapping_add(GOLDEN)`.
pub const DEFAULT_ROTATOR_SEED: u64 = 42;

impl TurboQuantizer {
    /// Build a quantizer for `dim`-dimensional vectors. `bit_width`
    /// (`DEFAULT_BIT_WIDTH` when `None`, valid range 1..=8) sets the
    /// total bits per coordinate (Stage 1 = `bit_width - 1`,
    /// Stage 2 = 1). `rotator_seed` (`DEFAULT_ROTATOR_SEED` when
    /// `None`) seeds the Stage 1 rotation; the Stage 2 QJL rotation
    /// is seeded with `rotator_seed.wrapping_add(0x9E37_79B9_7F4A_7C15)`
    /// (splittable, so both seeds are independent even when the caller
    /// passes sequential seeds).
    pub fn new(dim: usize, bit_width: Option<u8>, rotator_seed: Option<u64>) -> Self {
        let bit_width = bit_width.unwrap_or(DEFAULT_BIT_WIDTH);
        let rotator_seed = rotator_seed.unwrap_or(DEFAULT_ROTATOR_SEED);
        assert!((1..=8).contains(&bit_width), "bit_width must be 1..=8");
        assert!(dim > 0, "dim must be > 0");

        let rotator = DynamicRotator::new(dim, RotatorType::FhtKacRotator, rotator_seed);
        let padded_dim = rotator.padded_dim();

        // Stage 2 operates on already-rotated (padded) vectors of length
        // `padded_dim`, so it must be constructed at that size.
        let qjl_seed = rotator_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let qjl_rotator = DynamicRotator::new(padded_dim, RotatorType::FhtKacRotator, qjl_seed);

        // Codebook tuned for the Beta marginal of the padded_dim-sphere.
        // Stage 1 uses bit_width - 1 bits; with bit_width == 1 there is
        // no Stage 1 at all so the codebook is unused (we still build
        // one at b=1 to keep the struct non-optional).
        let s1_bits = bit_width.saturating_sub(1).max(1);
        let codebook = get_or_generate_cached(padded_dim, s1_bits);

        Self {
            dim,
            padded_dim,
            bit_width,
            rotator: Arc::new(rotator),
            qjl_rotator: Arc::new(qjl_rotator),
            codebook: Arc::new(codebook),
        }
    }

    /// Bytes per encoded record.
    #[inline]
    pub fn bytes_per_record(&self) -> usize {
        bytes_per_record(self.padded_dim, self.bit_width)
    }

    pub(crate) fn rotator(&self) -> &DynamicRotator {
        &self.rotator
    }
    pub(crate) fn qjl_rotator(&self) -> &DynamicRotator {
        &self.qjl_rotator
    }
    pub(crate) fn codebook(&self) -> &Codebook {
        &self.codebook
    }

    /// Encode `vec` into a fresh record byte vector.
    ///
    /// Panics if `vec.len() != self.dim`.
    pub fn encode(&self, vec: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; self.bytes_per_record()];
        self.encode_into(vec, &mut out);
        out
    }

    /// Reconstruct an approximate rotated-space vector from `record`.
    ///
    /// Writes `padded_dim` floats to `out`. Uses Stage 1 only (the
    /// scalar dequantized indices); Stage 2 sign bits cannot be
    /// reconstructed per-coordinate. The result lives in the rotated
    /// space (apply `rotator().inverse_rotate` to get back to input
    /// space — usually unnecessary, since callers that want to compare
    /// against centroids can rotate the centroids once instead).
    ///
    /// For `bit_width == 1` (Stage 1 absent) the output is all zeros.
    pub fn dequantize_into(&self, record: &[u8], out: &mut [f32]) {
        assert!(
            out.len() >= self.padded_dim,
            "dequantize_into: out buffer too small ({} < {})",
            out.len(),
            self.padded_dim
        );
        for x in out[..self.padded_dim].iter_mut() {
            *x = 0.0;
        }
        if self.bit_width <= 1 {
            return;
        }
        use super::bitpack;
        use super::record::stage1_bytes;
        let s1_bits = self.bit_width - 1;
        let s1 = stage1_bytes(record, self.padded_dim, self.bit_width);
        let mut indices = vec![0u8; self.padded_dim];
        bitpack::unpack_into(s1, self.padded_dim, s1_bits, &mut indices);
        for i in 0..self.padded_dim {
            out[i] = self.codebook.dequantize_scalar(indices[i]);
        }
    }

    /// Encode `vec` into `out` (which must be at least `bytes_per_record()`
    /// bytes). Reuses the caller's buffer to avoid a per-doc allocation
    /// on the hot path.
    pub fn encode_into(&self, vec: &[f32], out: &mut [u8]) {
        assert_eq!(vec.len(), self.dim, "TurboQuant encode: dim mismatch");
        assert!(out.len() >= self.bytes_per_record());

        // Stage 1 — rotate, scalar-quantize, reconstruct in rotated space
        let z = self.rotator.rotate(vec);
        let mut s1_indices = vec![0u8; self.padded_dim];
        let mut residual = vec![0.0f32; self.padded_dim];
        let mut gamma_sq = 0.0f64;

        if self.bit_width > 1 {
            for i in 0..self.padded_dim {
                let idx = self.codebook.quantize_scalar(z[i]);
                s1_indices[i] = idx;
                let r = z[i] - self.codebook.dequantize_scalar(idx);
                residual[i] = r;
                gamma_sq += (r as f64) * (r as f64);
            }
        } else {
            // Stage 1 skipped (b=1): residual is the rotated vector itself.
            for i in 0..self.padded_dim {
                residual[i] = z[i];
                gamma_sq += (z[i] as f64) * (z[i] as f64);
            }
        }
        let gamma = gamma_sq.sqrt() as f32;

        // Zero the packed region (indices / signs) before OR-style packing.
        let norm_off = norm_offset(self.padded_dim, self.bit_width);
        for b in &mut out[..norm_off] {
            *b = 0;
        }

        // Stage 1 packed indices
        let s1_end = stage2_offset(self.padded_dim, self.bit_width);
        if self.bit_width > 1 {
            let s1_bits = self.bit_width - 1;
            bitpack::pack_into(&s1_indices, s1_bits, &mut out[..s1_end]);
        }

        // Stage 2 — project residual, pack sign bits
        let sr = self.qjl_rotator.rotate(&residual);
        let signs: Vec<bool> = sr.iter().map(|&v| v >= 0.0).collect();
        bitpack::pack_signs_into(&signs, &mut out[s1_end..norm_off]);

        // γ (residual norm)
        write_norm(out, self.padded_dim, self.bit_width, gamma);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a random unit-norm vector.
    fn unit_rand(d: usize, seed: u64) -> Vec<f32> {
        use rand::prelude::*;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n = Normal::new(0.0_f32, 1.0).unwrap();
        let mut v: Vec<f32> = (0..d).map(|_| n.sample(&mut rng)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn encode_record_size_matches_layout() {
        let tq = TurboQuantizer::new(768, Some(3), Some(42));
        let v = unit_rand(768, 1);
        let rec = tq.encode(&v);
        assert_eq!(rec.len(), tq.bytes_per_record());
        // For d=768, b=3 → 192 (stage1) + 96 (stage2) + 4 (norm) = 292
        assert_eq!(rec.len(), 292);
    }

    #[test]
    fn encode_is_deterministic_for_same_seed() {
        let tq_a = TurboQuantizer::new(256, Some(3), Some(42));
        let tq_b = TurboQuantizer::new(256, Some(3), Some(42));
        let v = unit_rand(256, 7);
        assert_eq!(tq_a.encode(&v), tq_b.encode(&v));
    }

    #[test]
    fn different_seeds_give_different_records() {
        let tq_a = TurboQuantizer::new(256, Some(3), Some(42));
        let tq_b = TurboQuantizer::new(256, Some(3), Some(43));
        let v = unit_rand(256, 7);
        assert_ne!(tq_a.encode(&v), tq_b.encode(&v));
    }

    #[test]
    fn encode_bit_width_1_is_just_signs_plus_norm() {
        let tq = TurboQuantizer::new(128, Some(1), Some(42));
        let v = unit_rand(128, 3);
        let rec = tq.encode(&v);
        // b=1: 0 stage1 bytes + 128/8=16 stage2 bytes + 4 norm = 20
        assert_eq!(rec.len(), 20);
    }

    #[test]
    fn encode_into_matches_encode() {
        let tq = TurboQuantizer::new(256, Some(3), Some(42));
        let v = unit_rand(256, 11);
        let out1 = tq.encode(&v);
        let mut out2 = vec![0u8; tq.bytes_per_record()];
        tq.encode_into(&v, &mut out2);
        assert_eq!(out1, out2);
    }
}
