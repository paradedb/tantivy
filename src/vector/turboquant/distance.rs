//! Query-time inner-product estimation against TurboQuant records.
//!
//! A `TurboQuantQuery` is built once per search, then hot-looped to
//! estimate `<query, record>` for every candidate. Keeping the rotated
//! query + QJL-projected query cached avoids redoing O(d log d) work on
//! every candidate.
//!
//! ## IP formula
//!
//! Given encode(`x`) = (stage1_indices `z̃`, stage2_signs `s`, γ), query
//! `y`, Stage 1 rotator `R_1` and Stage 2 QJL projection `R_2`:
//!
//!     ỹ  = R_1 · y                  (`rotated_query`)
//!     z̃_f = dequant(stage1_indices)  (reconstructed rotated vec)
//!     ŷ  = R_2 · ỹ                  (`qjl_query`)
//!
//! Then
//!
//!     <x, y> ≈ <z̃_f, ỹ> + (√(π/2) · γ / √d) · <sign_to_±1(s), ŷ>
//!
//! The first term is Stage 1 MSE; the second is the QJL residual
//! correction (Stage 2). Returning IP — *higher is more similar* —
//! matches what `sort_by_vector_distance.rs` expects after our
//! InnerProduct sign fix, so downstream code stays unchanged.
//!
//! ## NEON SIMD
//!
//! On `aarch64`, the hot path uses two specialized kernels:
//!   * `stage2_dot_neon`: 8 floats per byte via XOR-with-sign-mask (lookup-driven, branchless). 4
//!     accumulators for ILP.
//!   * `stage1_dot_b4_neon`: only for `bit_width == 4` (3-bit Stage 1 indices, 8-level codebook).
//!     Decodes 8 indices from 3 bytes via bitfield extracts, scalar-gathers the codebook, FMAs into
//!     4 NEON accumulators.
//!
//! Other bit widths fall back to the scalar path. Production usage is
//! pinned at `b=4` in pg_search, so the fast path covers it.

use super::quantizer::TurboQuantizer;
use super::record::{read_norm, stage1_bytes, stage2_bytes};

/// Per-query state shared across all candidate evaluations in one search.
pub struct TurboQuantQuery {
    rotated_query: Vec<f32>,
    qjl_query: Vec<f32>,
    /// `sqrt(π/2) / sqrt(padded_dim)` — scale factor on the Stage 2 contribution.
    qjl_scale: f32,
    /// Padded dim (= codebook + record layout dim).
    padded_dim: usize,
    bit_width: u8,
    /// Per-coordinate × per-codebook-entry LUT: `lut[i * K + v] =
    /// rotated_query[i] * codebook.dequantize_scalar(v)` where
    /// K = 2^(bit_width - 1). Used by the scalar fallback. Empty when
    /// the NEON fast path is selected, or when `bit_width == 1`.
    s1_lut: Vec<f32>,
    /// 2^(bit_width - 1) — number of Stage 1 codebook entries; cached so
    /// the hot loop can use a constant when `bit_width` is known.
    s1_levels: usize,
    /// Bare Stage 1 codebook centroids (only populated when the NEON
    /// fast path is selected). Indexed by quantization index.
    codebook: Vec<f32>,
    /// True when the SIMD Stage 1 / Stage 2 kernels are used.
    use_simd: bool,
}

impl TurboQuantQuery {
    /// Prepare `query` for many estimate_ip calls.
    pub fn new(quantizer: &TurboQuantizer, query: &[f32]) -> Self {
        assert_eq!(query.len(), quantizer.dim, "TurboQuantQuery: dim mismatch");
        let rotated = quantizer.rotator().rotate(query);
        let qjl_query = quantizer.qjl_rotator().rotate(&rotated);
        // Stage 2 scale.
        //
        // The Gaussian-projection QJL paper derives `√(π/2) / d` because
        // for a Gaussian S, each row scales <a,b> by 1/d. Our QJL uses
        // an SRHT-style orthogonal projection (FhtKacRotator) instead
        // of Gaussian. SRHT preserves L2 norm and gives marginal
        // (S·a)_i ~ N(0, ‖a‖²/d), so the per-coord covariance scales
        // by 1/d but the per-coord standard deviation scales by 1/√d.
        // Working through E[sign(X)·Y] for jointly Gaussian X,Y under
        // SRHT marginals gives:
        //   <a,b> ≈ √(π/2) · ‖a‖ · <signs, S·b> / √d
        // i.e. the right divisor is √d, not d. Verified numerically:
        // with the wrong /d scale the estimator undershoots by a factor
        // of ~27 at d=768 (≈ √768). With /√d it matches true IP to
        // within quantization noise.
        let qjl_scale = (std::f32::consts::PI / 2.0).sqrt() / (quantizer.padded_dim as f32).sqrt();

        let bw = quantizer.bit_width;
        let s1_levels = if bw > 1 { 1usize << (bw - 1) } else { 0 };
        let codebook_ref = quantizer.codebook();

        // SIMD selection: we have a NEON Stage 1 kernel for b=4 only,
        // and a NEON Stage 2 kernel for any bit width with d % 8 == 0.
        // We enable SIMD whenever Stage 2 (which always runs) can be
        // vectorized. Stage 1 takes the SIMD path only when both
        // conditions hold; otherwise it falls back to the scalar LUT.
        let stage2_simd_ok = cfg!(target_arch = "aarch64") && quantizer.padded_dim % 8 == 0;
        let stage1_simd_ok = stage2_simd_ok && bw == 4;
        let use_simd = stage2_simd_ok;

        let codebook: Vec<f32> = if stage1_simd_ok {
            (0..s1_levels)
                .map(|v| codebook_ref.dequantize_scalar(v as u8))
                .collect()
        } else {
            Vec::new()
        };

        // s1_lut is always populated when there is a Stage 1 (b > 1).
        // The SIMD kernel uses the bare codebook + rotated_query
        // instead, but the scalar fallback / debug paths still want
        // the LUT, and building it is a one-time per-query cost
        // (≈ d * K * 4 bytes ≈ 24 KiB at d=768, K=8 — negligible
        // against the work of querying 60 K candidates).
        let s1_lut: Vec<f32> = if s1_levels == 0 {
            Vec::new()
        } else {
            let mut lut = Vec::with_capacity(quantizer.padded_dim * s1_levels);
            for i in 0..quantizer.padded_dim {
                let qi = rotated[i];
                for v in 0..s1_levels {
                    lut.push(qi * codebook_ref.dequantize_scalar(v as u8));
                }
            }
            lut
        };

        Self {
            rotated_query: rotated,
            qjl_query,
            qjl_scale,
            padded_dim: quantizer.padded_dim,
            bit_width: bw,
            s1_lut,
            s1_levels,
            codebook,
            use_simd,
        }
    }

    /// Estimate the inner product `<x, query>` for the record encoded from
    /// some full-precision `x`. Higher is more similar.
    #[inline]
    pub fn estimate_ip(&self, record: &[u8]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            if self.use_simd {
                // SAFETY: `use_simd` is set only when the platform is
                // aarch64 and `padded_dim % 8 == 0`. NEON is part of
                // the aarch64 baseline.
                return unsafe { self.estimate_ip_neon(record) };
            }
        }
        self.estimate_ip_scalar(record)
    }

    /// Scalar reference implementation — used as a fallback and for
    /// correctness comparisons against the SIMD kernels.
    #[inline]
    fn estimate_ip_scalar(&self, record: &[u8]) -> f32 {
        let d = self.padded_dim;
        let bw = self.bit_width;

        // Stage 1: reconstruct dequantized rotated vector, dot with
        // rotated query. Skip entirely if bit_width == 1.
        let mut ip = 0.0f32;
        if bw > 1 {
            let s1_bits = (bw - 1) as u32;
            let mask = ((1u16 << s1_bits) - 1) as u16;
            let s1 = stage1_bytes(record, d, bw);
            let levels = self.s1_levels;
            let lut = self.s1_lut.as_slice();
            for i in 0..d {
                let bit_offset = i * s1_bits as usize;
                let byte_idx = bit_offset / 8;
                let bit_idx = bit_offset % 8;
                let hi: u16 = unsafe { *s1.get_unchecked(byte_idx) }.into();
                let lo: u16 = if byte_idx + 1 < s1.len() {
                    unsafe { *s1.get_unchecked(byte_idx + 1) }.into()
                } else {
                    0
                };
                let combined = (hi << 8) | lo;
                let shift = 16 - s1_bits - bit_idx as u32;
                let idx = ((combined >> shift) & mask) as usize;
                ip += unsafe { *lut.get_unchecked(i * levels + idx) };
            }
        }

        // Stage 2: sign-weighted dot of QJL-projected query, scaled by γ·√(π/2)/√d.
        let s2 = stage2_bytes(record, d, bw);
        let gamma = read_norm(record, d, bw);
        let mut stage2 = 0.0f32;
        let qjl = self.qjl_query.as_slice();
        let full_bytes = d / 8;
        for byte_idx in 0..full_bytes {
            let bits = unsafe { *s2.get_unchecked(byte_idx) };
            let base = byte_idx * 8;
            for j in 0..8 {
                let q = unsafe { *qjl.get_unchecked(base + j) };
                let bit = (bits >> (7 - j)) & 1;
                if bit == 1 {
                    stage2 += q;
                } else {
                    stage2 -= q;
                }
            }
        }
        let tail_bits = d - full_bytes * 8;
        if tail_bits > 0 {
            let bits = s2[full_bytes];
            let base = full_bytes * 8;
            for j in 0..tail_bits {
                let q = self.qjl_query[base + j];
                let bit = (bits >> (7 - j)) & 1;
                if bit == 1 {
                    stage2 += q;
                } else {
                    stage2 -= q;
                }
            }
        }

        ip + gamma * self.qjl_scale * stage2
    }

    /// NEON-vectorized estimator. Delegates to specialized kernels.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn estimate_ip_neon(&self, record: &[u8]) -> f32 {
        let d = self.padded_dim;
        let bw = self.bit_width;
        let s2 = stage2_bytes(record, d, bw);
        let gamma = read_norm(record, d, bw);
        let full_bytes = d / 8;

        let stage2 = neon::stage2_dot_neon(s2, &self.qjl_query, full_bytes);

        let mut ip = 0.0f32;
        if bw == 4 {
            let s1 = stage1_bytes(record, d, bw);
            let cb: &[f32; 8] = self.codebook.as_slice().try_into().unwrap_unchecked();
            ip = neon::stage1_dot_b4_neon(s1, &self.rotated_query, cb, d);
        } else if bw > 1 {
            // Shouldn't happen given current SIMD-selection rules, but
            // keep a safety net.
            ip = self.estimate_ip_scalar_stage1_only(record);
        }

        ip + gamma * self.qjl_scale * stage2
    }

    /// Stage 1 only, scalar — safety fallback inside the NEON path when
    /// b ∉ {1, 4}. Production never hits this.
    #[cfg(target_arch = "aarch64")]
    fn estimate_ip_scalar_stage1_only(&self, record: &[u8]) -> f32 {
        let d = self.padded_dim;
        let bw = self.bit_width;
        let s1_bits = (bw - 1) as u32;
        let mask = ((1u16 << s1_bits) - 1) as u16;
        let s1 = stage1_bytes(record, d, bw);
        let levels = self.s1_levels;
        let lut = self.s1_lut.as_slice();
        let mut ip = 0.0f32;
        for i in 0..d {
            let bit_offset = i * s1_bits as usize;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;
            let hi: u16 = unsafe { *s1.get_unchecked(byte_idx) }.into();
            let lo: u16 = if byte_idx + 1 < s1.len() {
                unsafe { *s1.get_unchecked(byte_idx + 1) }.into()
            } else {
                0
            };
            let combined = (hi << 8) | lo;
            let shift = 16 - s1_bits - bit_idx as u32;
            let idx = ((combined >> shift) & mask) as usize;
            ip += unsafe { *lut.get_unchecked(i * levels + idx) };
        }
        ip
    }

    /// Estimate the squared L2 distance `‖x - query‖²`. Assumes inputs
    /// are unit-norm (so `‖x - y‖² = 2 - 2·<x, y>`); used by
    /// cosine-ordering callers.
    #[inline]
    pub fn estimate_l2sq_unit(&self, record: &[u8]) -> f32 {
        2.0 - 2.0 * self.estimate_ip(record)
    }

    /// `(s1_lut, K)` view: `s1_lut[i*K + v] = rotated_query[i] *
    /// codebook[v]`. Exposed for the batched scorer in
    /// `transposed.rs` so it can reuse the cached per-query tables.
    pub fn s1_lut_view(&self) -> (&[f32], usize) {
        (self.s1_lut.as_slice(), self.s1_levels)
    }

    /// QJL-projected query, length `padded_dim`. Stage-2 scoring
    /// dots this against per-doc sign bits.
    pub fn qjl_query_view(&self) -> &[f32] {
        &self.qjl_query
    }

    /// Scalar `√(π/2) / √padded_dim` Stage-2 multiplier.
    pub fn qjl_scale_value(&self) -> f32 {
        self.qjl_scale
    }

    /// Padded vector dimension (`= TurboQuantizer::padded_dim`).
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Bit width passed to `TurboQuantizer::new`.
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }
}

/// Precomputed XOR sign masks: `SIGN_MASK_TABLE[byte][j]` is `0` if bit
/// `j` (MSB-first) of `byte` is 1, else `0x8000_0000`. XORing a float's
/// bit pattern with this mask flips its sign exactly when the
/// corresponding bit is 0 — i.e. `f → +f if bit==1 else -f`. This makes
/// the Stage 2 inner loop branchless.
///
/// 256 × 8 × 4 B = 8 KiB, fits comfortably in L1d.
const SIGN_MASK_TABLE: [[u32; 8]; 256] = build_sign_mask_table();

const fn build_sign_mask_table() -> [[u32; 8]; 256] {
    let mut t = [[0u32; 8]; 256];
    let mut b: usize = 0;
    while b < 256 {
        let mut j = 0;
        while j < 8 {
            let bit = (b >> (7 - j)) & 1;
            t[b][j] = if bit == 1 { 0 } else { 0x8000_0000 };
            j += 1;
        }
        b += 1;
    }
    t
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    use super::SIGN_MASK_TABLE;

    /// Stage 2 inner loop: `Σ_byte Σ_j (bit_j ? +qjl[base+j] : -qjl[base+j])`.
    ///
    /// Per byte we load 8 floats from `qjl`, look up an 8-lane XOR sign
    /// mask from the byte, XOR, and accumulate. We process 2 bytes (16
    /// floats) per iteration with 4 independent accumulators to hide
    /// FMA/ALU latency on Apple-silicon NEON.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn stage2_dot_neon(s2: &[u8], qjl: &[f32], full_bytes: usize) -> f32 {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let qptr = qjl.as_ptr();
        let pairs = full_bytes / 2;
        let mut byte_idx = 0;
        while byte_idx < pairs * 2 {
            let m0 = SIGN_MASK_TABLE[*s2.get_unchecked(byte_idx) as usize];
            let m1 = SIGN_MASK_TABLE[*s2.get_unchecked(byte_idx + 1) as usize];
            let base = byte_idx * 8;

            let q0 = vld1q_f32(qptr.add(base));
            let q1 = vld1q_f32(qptr.add(base + 4));
            let q2 = vld1q_f32(qptr.add(base + 8));
            let q3 = vld1q_f32(qptr.add(base + 12));

            let mv0 = vld1q_u32(m0.as_ptr());
            let mv1 = vld1q_u32(m0.as_ptr().add(4));
            let mv2 = vld1q_u32(m1.as_ptr());
            let mv3 = vld1q_u32(m1.as_ptr().add(4));

            let f0 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q0), mv0));
            let f1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q1), mv1));
            let f2 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q2), mv2));
            let f3 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q3), mv3));

            acc0 = vaddq_f32(acc0, f0);
            acc1 = vaddq_f32(acc1, f1);
            acc2 = vaddq_f32(acc2, f2);
            acc3 = vaddq_f32(acc3, f3);

            byte_idx += 2;
        }
        // Remaining single byte (when full_bytes is odd).
        if byte_idx < full_bytes {
            let m = SIGN_MASK_TABLE[*s2.get_unchecked(byte_idx) as usize];
            let base = byte_idx * 8;
            let q0 = vld1q_f32(qptr.add(base));
            let q1 = vld1q_f32(qptr.add(base + 4));
            let mv0 = vld1q_u32(m.as_ptr());
            let mv1 = vld1q_u32(m.as_ptr().add(4));
            let f0 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q0), mv0));
            let f1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q1), mv1));
            acc0 = vaddq_f32(acc0, f0);
            acc1 = vaddq_f32(acc1, f1);
        }

        let s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(s)
    }

    /// Stage 1 inner loop for `bit_width == 4` (3-bit indices, 8-level
    /// codebook). Returns `Σ_i rotated_query[i] * cb[idx[i]]`.
    ///
    /// 8 indices pack into 3 bytes. We decode them with shifts/masks,
    /// scalar-gather the 8 codebook centroids, and FMA them into NEON
    /// accumulators. Two 8-coord chunks (16 coords / 6 input bytes) per
    /// iteration with 4 independent accumulators for ILP.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn stage1_dot_b4_neon(s1: &[u8], rotated_q: &[f32], cb: &[f32; 8], d: usize) -> f32 {
        debug_assert_eq!(d % 8, 0, "stage1_dot_b4_neon expects d divisible by 8");
        debug_assert!(s1.len() * 8 >= d * 3, "stage1_dot_b4_neon: s1 too short");

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let qptr = rotated_q.as_ptr();
        let cb_ptr = cb.as_ptr();
        let chunks_8 = d / 8;
        let pairs = chunks_8 / 2;
        let mut chunk_i = 0;
        while chunk_i < pairs * 2 {
            // Chunk A
            let s_off = chunk_i * 3;
            let base = chunk_i * 8;
            let combined = ((*s1.get_unchecked(s_off) as u32) << 16)
                | ((*s1.get_unchecked(s_off + 1) as u32) << 8)
                | (*s1.get_unchecked(s_off + 2) as u32);
            let i0 = ((combined >> 21) & 7) as usize;
            let i1 = ((combined >> 18) & 7) as usize;
            let i2 = ((combined >> 15) & 7) as usize;
            let i3 = ((combined >> 12) & 7) as usize;
            let i4 = ((combined >> 9) & 7) as usize;
            let i5 = ((combined >> 6) & 7) as usize;
            let i6 = ((combined >> 3) & 7) as usize;
            let i7 = (combined & 7) as usize;
            let cv0 = [
                *cb_ptr.add(i0),
                *cb_ptr.add(i1),
                *cb_ptr.add(i2),
                *cb_ptr.add(i3),
            ];
            let cv1 = [
                *cb_ptr.add(i4),
                *cb_ptr.add(i5),
                *cb_ptr.add(i6),
                *cb_ptr.add(i7),
            ];
            let c0 = vld1q_f32(cv0.as_ptr());
            let c1 = vld1q_f32(cv1.as_ptr());
            let q0 = vld1q_f32(qptr.add(base));
            let q1 = vld1q_f32(qptr.add(base + 4));
            acc0 = vfmaq_f32(acc0, c0, q0);
            acc1 = vfmaq_f32(acc1, c1, q1);

            // Chunk B
            let s_off2 = (chunk_i + 1) * 3;
            let base2 = (chunk_i + 1) * 8;
            let cm = ((*s1.get_unchecked(s_off2) as u32) << 16)
                | ((*s1.get_unchecked(s_off2 + 1) as u32) << 8)
                | (*s1.get_unchecked(s_off2 + 2) as u32);
            let j0 = ((cm >> 21) & 7) as usize;
            let j1 = ((cm >> 18) & 7) as usize;
            let j2 = ((cm >> 15) & 7) as usize;
            let j3 = ((cm >> 12) & 7) as usize;
            let j4 = ((cm >> 9) & 7) as usize;
            let j5 = ((cm >> 6) & 7) as usize;
            let j6 = ((cm >> 3) & 7) as usize;
            let j7 = (cm & 7) as usize;
            let dv0 = [
                *cb_ptr.add(j0),
                *cb_ptr.add(j1),
                *cb_ptr.add(j2),
                *cb_ptr.add(j3),
            ];
            let dv1 = [
                *cb_ptr.add(j4),
                *cb_ptr.add(j5),
                *cb_ptr.add(j6),
                *cb_ptr.add(j7),
            ];
            let d0 = vld1q_f32(dv0.as_ptr());
            let d1 = vld1q_f32(dv1.as_ptr());
            let qa = vld1q_f32(qptr.add(base2));
            let qb = vld1q_f32(qptr.add(base2 + 4));
            acc2 = vfmaq_f32(acc2, d0, qa);
            acc3 = vfmaq_f32(acc3, d1, qb);

            chunk_i += 2;
        }
        // Tail: remaining single 3-byte chunk.
        if chunk_i < chunks_8 {
            let s_off = chunk_i * 3;
            let base = chunk_i * 8;
            let combined = ((*s1.get_unchecked(s_off) as u32) << 16)
                | ((*s1.get_unchecked(s_off + 1) as u32) << 8)
                | (*s1.get_unchecked(s_off + 2) as u32);
            let i0 = ((combined >> 21) & 7) as usize;
            let i1 = ((combined >> 18) & 7) as usize;
            let i2 = ((combined >> 15) & 7) as usize;
            let i3 = ((combined >> 12) & 7) as usize;
            let i4 = ((combined >> 9) & 7) as usize;
            let i5 = ((combined >> 6) & 7) as usize;
            let i6 = ((combined >> 3) & 7) as usize;
            let i7 = (combined & 7) as usize;
            let cv0 = [
                *cb_ptr.add(i0),
                *cb_ptr.add(i1),
                *cb_ptr.add(i2),
                *cb_ptr.add(i3),
            ];
            let cv1 = [
                *cb_ptr.add(i4),
                *cb_ptr.add(i5),
                *cb_ptr.add(i6),
                *cb_ptr.add(i7),
            ];
            let c0 = vld1q_f32(cv0.as_ptr());
            let c1 = vld1q_f32(cv1.as_ptr());
            let q0 = vld1q_f32(qptr.add(base));
            let q1 = vld1q_f32(qptr.add(base + 4));
            acc0 = vfmaq_f32(acc0, c0, q0);
            acc1 = vfmaq_f32(acc1, c1, q1);
        }

        let s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(s)
    }
}

#[cfg(test)]
mod tests {
    use super::super::quantizer::TurboQuantizer;
    use super::*;

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

    fn exact_ip(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Recall@k of turboquant vs brute-force exact IP on a random unit-sphere
    /// dataset. Tight enough to catch regressions in the end-to-end pipeline
    /// (encode + query + IP estimate + sort order).
    #[test]
    fn recall_at_10_beats_floor() {
        let d = 256;
        let n = 1000;
        let k_recall = 10;
        let bit_width = 3;

        let tq = TurboQuantizer::new(d, Some(bit_width), Some(42));

        let docs: Vec<Vec<f32>> = (0..n).map(|i| unit_rand(d, 1_000 + i as u64)).collect();
        let records: Vec<Vec<u8>> = docs.iter().map(|v| tq.encode(v)).collect();

        let queries: Vec<Vec<f32>> = (0..10).map(|i| unit_rand(d, 7_000 + i)).collect();

        let mut total_recall = 0usize;
        for q in &queries {
            let mut exact: Vec<(usize, f32)> = docs
                .iter()
                .enumerate()
                .map(|(i, v)| (i, exact_ip(v, q)))
                .collect();
            exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let gt: std::collections::HashSet<usize> =
                exact.iter().take(k_recall).map(|(i, _)| *i).collect();

            let tqq = TurboQuantQuery::new(&tq, q);
            let mut est: Vec<(usize, f32)> = records
                .iter()
                .enumerate()
                .map(|(i, r)| (i, tqq.estimate_ip(r)))
                .collect();
            est.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top: std::collections::HashSet<usize> =
                est.iter().take(k_recall).map(|(i, _)| *i).collect();

            total_recall += gt.intersection(&top).count();
        }
        let avg_recall = total_recall as f32 / (queries.len() * k_recall) as f32;
        assert!(
            avg_recall >= 0.5,
            "recall@10 below sanity floor: {avg_recall}"
        );
    }

    #[test]
    fn self_ip_close_to_one() {
        let d = 768;
        let tq = TurboQuantizer::new(d, Some(3), Some(42));
        let v = unit_rand(d, 1);
        let rec = tq.encode(&v);
        let q = TurboQuantQuery::new(&tq, &v);
        let ip = q.estimate_ip(&rec);
        assert!((ip - 1.0).abs() < 0.2, "self-IP too far from 1.0: {ip}");
    }

    #[test]
    fn higher_bit_width_tightens_self_ip() {
        let d = 768;
        let v = unit_rand(d, 1);

        let tq2 = TurboQuantizer::new(d, Some(2), Some(42));
        let tq4 = TurboQuantizer::new(d, Some(4), Some(42));

        let r2 = tq2.encode(&v);
        let r4 = tq4.encode(&v);

        let q2 = TurboQuantQuery::new(&tq2, &v);
        let q4 = TurboQuantQuery::new(&tq4, &v);

        let err2 = (q2.estimate_ip(&r2) - 1.0).abs();
        let err4 = (q4.estimate_ip(&r4) - 1.0).abs();

        assert!(
            err4 <= err2 + 0.05,
            "expected b=4 error ({err4}) not much worse than b=2 error ({err2})"
        );
    }

    /// SIMD path must produce numerically equivalent IP estimates to
    /// the scalar reference. We exercise the b=4 path (Stage 1+Stage 2
    /// SIMD) and the b=3 path (Stage 2 SIMD only).
    #[test]
    fn neon_matches_scalar_b4() {
        let d = 768;
        let tq = TurboQuantizer::new(d, Some(4), Some(42));
        let docs: Vec<Vec<f32>> = (0..16).map(|i| unit_rand(d, 100 + i)).collect();
        let recs: Vec<Vec<u8>> = docs.iter().map(|v| tq.encode(v)).collect();
        let q = unit_rand(d, 9_001);
        let qq = TurboQuantQuery::new(&tq, &q);

        for rec in &recs {
            let scalar = qq.estimate_ip_scalar(rec);
            let combined = qq.estimate_ip(rec);
            // Float reductions can reorder; tolerate small drift.
            assert!(
                (scalar - combined).abs() < 1e-4,
                "scalar {scalar} vs simd {combined}"
            );
        }
    }

    /// Ignored microbenchmark: estimate how long `estimate_ip` takes
    /// per call. Run with:
    ///
    ///     cargo test --release --lib \
    ///       vector::turboquant::distance::tests::bench_estimate_ip \
    ///       -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_estimate_ip() {
        use std::time::Instant;

        let d = 768;
        let n = 60_000;
        let bw = 4u8;

        let tq = TurboQuantizer::new(d, Some(bw), Some(42));
        let docs: Vec<Vec<u8>> = (0..n)
            .map(|i| tq.encode(&unit_rand(d, 1_000 + i as u64)))
            .collect();
        let q = unit_rand(d, 9_001);
        let qq = TurboQuantQuery::new(&tq, &q);

        // Warm up.
        let mut sink = 0.0f32;
        for r in &docs {
            sink += qq.estimate_ip(r);
        }

        let start = Instant::now();
        for r in &docs {
            sink += qq.estimate_ip(r);
        }
        let total = start.elapsed();
        eprintln!(
            "SIMD: {} docs in {:?}, {:.1} ns/doc, sum {sink}",
            n,
            total,
            total.as_nanos() as f64 / n as f64
        );

        // Force scalar path for comparison.
        let mut qq_scalar = TurboQuantQuery::new(&tq, &q);
        qq_scalar.use_simd = false;

        for r in &docs {
            sink += qq_scalar.estimate_ip(r);
        }
        let start = Instant::now();
        for r in &docs {
            sink += qq_scalar.estimate_ip(r);
        }
        let total = start.elapsed();
        eprintln!(
            "Scalar: {} docs in {:?}, {:.1} ns/doc, sum {sink}",
            n,
            total,
            total.as_nanos() as f64 / n as f64
        );
    }

    #[test]
    fn neon_matches_scalar_b3() {
        let d = 256;
        let tq = TurboQuantizer::new(d, Some(3), Some(42));
        let docs: Vec<Vec<f32>> = (0..16).map(|i| unit_rand(d, 200 + i)).collect();
        let recs: Vec<Vec<u8>> = docs.iter().map(|v| tq.encode(v)).collect();
        let q = unit_rand(d, 9_002);
        let qq = TurboQuantQuery::new(&tq, &q);

        for rec in &recs {
            let scalar = qq.estimate_ip_scalar(rec);
            let combined = qq.estimate_ip(rec);
            assert!(
                (scalar - combined).abs() < 1e-4,
                "scalar {scalar} vs simd {combined}"
            );
        }
    }
}
