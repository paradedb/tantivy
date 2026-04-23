//! Per-cluster 16-doc transposed layout for batched SIMD scoring.
//!
//! Hot path is `bit_width = 4` (3-bit Stage 1 indices, 1-bit Stage 2
//! signs). Other widths fall through to the doc-major scorer in
//! `distance.rs` — the `cluster` plugin only emits the transposed
//! layout when `bit_width == 4`.
//!
//! ## Layout (per 16-doc batch, padded_dim = D)
//!
//! ```text
//! [stage1_t: 8·D bytes]   16 × 4-bit slots per coord (high bit unused
//!                         per nibble — Stage 1 indices are 3-bit but
//!                         we round each slot to a nibble for cheap
//!                         vqtbl1q_u8 unpack)
//! [stage2_t: 2·D bytes]   16 × 1-bit signs per coord, MSB-first
//! [gammas:   64 bytes]    16 × f32 per-doc residual norms
//! ```
//!
//! For D = 768: 6144 + 1536 + 64 = 7744 B per batch (484 B/doc, ~25 %
//! more than the doc-major 388 B/doc — we pay the extra to keep the
//! Stage-1 unpack a single NEON shift+and instead of a packed-bit
//! gather).
//!
//! Within each coord's 8-byte Stage-1 slot, doc layout is:
//!
//! ```text
//! byte 0: [doc 0 hi nibble][doc 1 lo nibble]
//! byte 1: [doc 2 hi nibble][doc 3 lo nibble]
//! ...
//! byte 7: [doc 14 hi nibble][doc 15 lo nibble]
//! ```
//!
//! The "hi nibble" of a byte means the upper 4 bits of that byte —
//! which holds the index for the even-numbered doc in the pair.
//!
//! ## Scoring
//!
//! `score_batch_neon` precomputes a per-query, per-coord int8 lookup
//! table and a single global scale factor. Stage 1 then becomes one
//! `vqtbl1q_s8` per coord (16 codebook lookups in parallel), widened
//! and added into 4 i32 accumulators — i32-accumulate is overflow-safe
//! at d ≤ 2¹⁶/127 ≈ 516 worth of i8 sums per accumulator, well above
//! 768. The conversion to f32 + global-scale multiply happens once per
//! batch at the end.
//!
//! Stage 2 is a per-coord broadcast of `qjl_query[c]` XOR-ed with a
//! 16-doc sign mask (precomputed `[u32; 8]` per byte) accumulated in
//! 4 f32 accumulators — same XOR-with-sign-bit trick the per-doc NEON
//! kernel uses, just batched 16-wide.
//!
//! ## Why hardcode b = 4
//!
//! pg_search pins `TURBOQUANT_BITS = 4`. Generalizing the kernel to
//! b ∈ {2, 3, 5..8} doubles the bit-unpack code paths and isn't worth
//! the complexity until a caller asks for it. For other bit widths
//! the cluster plugin keeps emitting the doc-major external layout
//! (`build_cluster_batch_data_external_doc_major`).

use super::quantizer::TurboQuantizer;
use super::record::{read_norm, stage1_bytes, stage2_bytes};

/// Fixed batch size. Picked so one `vqtbl1q_u8` lookup covers the
/// whole batch's stage-1 codebook gather.
pub const BATCH_DOCS: usize = 16;

/// Hardcoded bit width for the transposed/SIMD path.
pub const TRANSPOSED_BIT_WIDTH: u8 = 4;

/// Codebook entries for `bit_width = 4` Stage 1 (= 2^(b-1)).
pub const TRANSPOSED_CODEBOOK_LEVELS: usize = 8;

/// Stage-1 transposed slab bytes per batch: one 4-bit slot per
/// (doc, coord) pair, 16 docs per coord = 8 B/coord.
#[inline]
pub fn stage1_t_bytes(padded_dim: usize) -> usize {
    8 * padded_dim
}

/// Stage-2 transposed slab bytes per batch: one 1-bit slot per
/// (doc, coord) pair, 16 docs per coord = 2 B/coord.
#[inline]
pub fn stage2_t_bytes(padded_dim: usize) -> usize {
    2 * padded_dim
}

/// γ slab bytes per batch.
pub const GAMMAS_BYTES: usize = BATCH_DOCS * 4;

/// Total bytes for one batch.
#[inline]
pub fn batch_bytes(padded_dim: usize) -> usize {
    stage1_t_bytes(padded_dim) + stage2_t_bytes(padded_dim) + GAMMAS_BYTES
}

/// Total cluster bytes for `num_docs` docs in transposed layout:
/// `[doc_ids: 4N][batches: ⌈N/16⌉ × batch_bytes]`. The last batch is
/// always full-sized (unused lanes get zero indices, zero signs and
/// γ = 0), so callers must mask tail lanes when collecting scores.
#[inline]
pub fn cluster_bytes(num_docs: usize, padded_dim: usize) -> usize {
    let nb = num_docs.div_ceil(BATCH_DOCS);
    num_docs * 4 + nb * batch_bytes(padded_dim)
}

#[inline]
pub fn batch_stage2_t_offset(padded_dim: usize) -> usize {
    stage1_t_bytes(padded_dim)
}
#[inline]
pub fn batch_gammas_offset(padded_dim: usize) -> usize {
    stage1_t_bytes(padded_dim) + stage2_t_bytes(padded_dim)
}

/// Encode up to 16 doc-major records into one transposed batch. Inputs
/// must all be `bit_width = 4` records of length
/// `record::bytes_per_record(padded_dim, 4)`.
///
/// Records beyond `records.len()` (up to 16) get zero-filled stage-1
/// indices, zero stage-2 signs, and γ = 0. The corresponding scores
/// are discardable, not a real measurement of any doc.
pub fn encode_batch(records: &[&[u8]], padded_dim: usize, out: &mut [u8]) {
    assert!(records.len() <= BATCH_DOCS);
    assert_eq!(out.len(), batch_bytes(padded_dim));

    let s1_t_len = stage1_t_bytes(padded_dim);
    let s2_t_len = stage2_t_bytes(padded_dim);
    let (s1_t, rest) = out.split_at_mut(s1_t_len);
    let (s2_t, gammas) = rest.split_at_mut(s2_t_len);

    s1_t.fill(0);
    s2_t.fill(0);
    gammas.fill(0);

    // Per-doc temporaries reused across slots.
    let mut idx_buf = vec![0u8; padded_dim];
    let mut sign_buf = vec![false; padded_dim];

    for (slot, rec) in records.iter().enumerate() {
        // Stage 1: unpack the doc's 3-bit indices into one byte each,
        // then place each coord's 4-bit nibble into the transposed
        // slab. Doc parity decides upper vs lower nibble of the byte.
        let s1 = stage1_bytes(rec, padded_dim, TRANSPOSED_BIT_WIDTH);
        super::bitpack::unpack_into(s1, padded_dim, 3, &mut idx_buf);
        for c in 0..padded_dim {
            let nibble = idx_buf[c] & 0x0F; // 3-bit value, top bit = 0
            let byte_in_slab = c * 8 + slot / 2;
            if slot % 2 == 0 {
                // High nibble.
                s1_t[byte_in_slab] |= nibble << 4;
            } else {
                s1_t[byte_in_slab] |= nibble;
            }
        }

        // Stage 2 signs (MSB-first within each coord's 2-byte slot).
        let s2 = stage2_bytes(rec, padded_dim, TRANSPOSED_BIT_WIDTH);
        super::bitpack::unpack_signs_into(s2, padded_dim, &mut sign_buf);
        for c in 0..padded_dim {
            if sign_buf[c] {
                let bit_pos = c * 16 + slot;
                s2_t[bit_pos / 8] |= 1u8 << (7 - (bit_pos % 8));
            }
        }

        // γ.
        let g = read_norm(rec, padded_dim, TRANSPOSED_BIT_WIDTH);
        gammas[slot * 4..slot * 4 + 4].copy_from_slice(&g.to_le_bytes());
    }
}

/// Read doc `slot`'s stage-1 index out of a transposed slab.
/// Test/reference helper.
#[inline]
pub(crate) fn batch_stage1_index(s1_t: &[u8], c: usize, slot: usize) -> u8 {
    let b = s1_t[c * 8 + slot / 2];
    if slot % 2 == 0 {
        b >> 4
    } else {
        b & 0x0F
    }
}

/// Read doc `slot`'s stage-2 sign bit out of a transposed slab.
#[inline]
pub(crate) fn batch_stage2_sign(s2_t: &[u8], c: usize, slot: usize) -> u8 {
    let bp = c * 16 + slot;
    (s2_t[bp / 8] >> (7 - (bp % 8))) & 1
}

#[inline]
pub(crate) fn batch_gamma(gammas: &[u8], slot: usize) -> f32 {
    let o = slot * 4;
    f32::from_le_bytes([gammas[o], gammas[o + 1], gammas[o + 2], gammas[o + 3]])
}

/// Inverse of `encode_batch` for one slot: reconstruct the doc-major
/// per-record bytes (the same output `TurboQuantizer::encode_into`
/// would have produced) for `slot` in `batch`.
///
/// Used by the cluster plugin's merge path to read source-segment
/// records out of a `.cluster` file when there's no separate
/// doc-major store on disk.
///
/// Walks each coordinate once: lifts `slot`'s 3-bit stage-1 index
/// and 1-bit stage-2 sign out of the coord-major slabs, then
/// repacks them MSB-first into the doc-major record layout. γ is a
/// straight `f32` copy.
///
/// `out` must be at least `record::bytes_per_record(padded_dim, 4)`
/// bytes (the only bit width the transposed layout supports).
pub fn extract_record(batch: &[u8], slot: usize, padded_dim: usize, out: &mut [u8]) {
    use super::bitpack;
    use super::record::{bytes_per_record, norm_offset, stage2_offset, write_norm};

    const BIT_WIDTH: u8 = TRANSPOSED_BIT_WIDTH;
    let bpr = bytes_per_record(padded_dim, BIT_WIDTH);
    assert!(out.len() >= bpr);
    assert!(slot < BATCH_DOCS);

    let s1_t_len = stage1_t_bytes(padded_dim);
    let s2_t_len = stage2_t_bytes(padded_dim);
    let s1_t = &batch[..s1_t_len];
    let s2_t = &batch[s1_t_len..s1_t_len + s2_t_len];
    let gammas = &batch[s1_t_len + s2_t_len..];

    // Zero the packed region; bitpack uses OR-style writes.
    let norm_off = norm_offset(padded_dim, BIT_WIDTH);
    for b in &mut out[..norm_off] {
        *b = 0;
    }

    // Lift stage-1 indices for `slot` from each coord, then repack.
    let mut indices = vec![0u8; padded_dim];
    for c in 0..padded_dim {
        indices[c] = batch_stage1_index(s1_t, c, slot);
    }
    let s1_end = stage2_offset(padded_dim, BIT_WIDTH);
    bitpack::pack_into(&indices, BIT_WIDTH - 1, &mut out[..s1_end]);

    // Lift stage-2 signs for `slot` from each coord, then repack.
    let mut signs = vec![false; padded_dim];
    for c in 0..padded_dim {
        signs[c] = batch_stage2_sign(s2_t, c, slot) == 1;
    }
    bitpack::pack_signs_into(&signs, &mut out[s1_end..norm_off]);

    // γ.
    let gamma = batch_gamma(gammas, slot);
    write_norm(out, padded_dim, BIT_WIDTH, gamma);
}

/// Scalar reference scorer. Computes the same inner product as
/// per-doc `TurboQuantQuery::estimate_ip` but for all 16 docs in a
/// transposed batch. Used for parity testing the SIMD kernel; never
/// the production scoring path.
pub fn score_batch_scalar(
    query: &crate::vector::turboquant::TurboQuantQuery,
    batch: &[u8],
    out: &mut [f32; BATCH_DOCS],
) {
    let padded_dim = query.padded_dim();
    assert_eq!(query.bit_width(), TRANSPOSED_BIT_WIDTH);

    let s1_t_len = stage1_t_bytes(padded_dim);
    let s2_t_len = stage2_t_bytes(padded_dim);
    let s1_t = &batch[..s1_t_len];
    let s2_t = &batch[s1_t_len..s1_t_len + s2_t_len];
    let gammas = &batch[s1_t_len + s2_t_len..];

    let (s1_lut, s1_levels) = query.s1_lut_view();
    let qjl = query.qjl_query_view();
    let qjl_scale = query.qjl_scale_value();

    for slot in 0..BATCH_DOCS {
        let mut stage1 = 0.0f32;
        for c in 0..padded_dim {
            let idx = batch_stage1_index(s1_t, c, slot) as usize;
            stage1 += s1_lut[c * s1_levels + idx];
        }
        let mut stage2 = 0.0f32;
        for c in 0..padded_dim {
            if batch_stage2_sign(s2_t, c, slot) == 1 {
                stage2 += qjl[c];
            } else {
                stage2 -= qjl[c];
            }
        }
        out[slot] = stage1 + batch_gamma(gammas, slot) * qjl_scale * stage2;
    }
}

/// Per-query precomputed state for the SIMD batched scorer.
///
/// Holds an i8 quantization of the per-coord codebook table plus the
/// QJL-projected query and scale factors. Built once per search and
/// reused across every batch.
pub struct BatchedQueryLut {
    /// Per-coord 16-byte tables for `vqtbl1q_s8`. For coord c, bytes
    /// `[c*16 .. c*16+8]` hold the i8 quantized values of
    /// `q[c] * cb[0..8]`. Bytes `[c*16+8 .. c*16+16]` are filled with
    /// the same 8 entries as a defensive replication so out-of-range
    /// indices (which shouldn't occur but might due to a corrupted
    /// nibble) still produce a valid lookup.
    lut_i8: Vec<i8>,
    /// Single global scale: when accumulator (i32) is converted to
    /// f32, multiply by this to recover the true Stage-1 contribution.
    /// `s1_lut[c, v] ≈ scale_global * lut_i8[c*16 + v]`.
    scale_global: f32,
    /// QJL-projected query (length = padded_dim).
    qjl_query: Vec<f32>,
    /// `√(π/2) / √padded_dim` — Stage-2 multiplier.
    qjl_scale: f32,
    /// Tightest cheap per-doc upper bound on `qjl_scale · ⟨signs, qjl⟩`
    /// for any sign vector: `qjl_scale · ‖qjl‖_1`. Multiplied by γ
    /// per slot in `Stage1Partial::max_score_upper_bound` to cap the
    /// Stage-2 contribution and decide whether to skip the Stage-2
    /// pass entirely.
    qjl_max_correction: f32,
    padded_dim: usize,
}

impl BatchedQueryLut {
    pub fn new(query: &crate::vector::turboquant::TurboQuantQuery) -> Self {
        assert_eq!(query.bit_width(), TRANSPOSED_BIT_WIDTH);
        let padded_dim = query.padded_dim();
        let (s1_lut, s1_levels) = query.s1_lut_view();
        assert_eq!(s1_levels, TRANSPOSED_CODEBOOK_LEVELS);

        // Single global scale: max absolute value across the entire
        // s1_lut. With ‖q‖ = 1 (unit query) and cb[v] roughly the
        // same magnitude across coords, the per-coord max varies by
        // a small constant, so a global scale loses ≤ a few bits of
        // precision per coord — fine after summing 768 terms.
        let mut max_abs = 0.0f32;
        for &v in s1_lut {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale_global = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv = 1.0 / scale_global;

        let mut lut_i8 = vec![0i8; padded_dim * 16];
        for c in 0..padded_dim {
            let base = c * 16;
            for v in 0..s1_levels {
                let q = s1_lut[c * s1_levels + v] * inv;
                let qi = q.round().clamp(-127.0, 127.0) as i8;
                lut_i8[base + v] = qi;
                // Defensive replication into upper half.
                lut_i8[base + 8 + v] = qi;
            }
        }

        // qjl_max_correction: 5σ bound on |⟨signs, qjl⟩|. See the
        // doc-comment on Stage1Partial::max_score_upper_bound for
        // why this is safe vs the strict L1 bound.
        let qjl = query.qjl_query_view();
        let qjl_l2_sq: f32 = qjl.iter().map(|x| x * x).sum();
        let qjl_l2 = qjl_l2_sq.sqrt();
        let qjl_scale = query.qjl_scale_value();
        const SIGMA_MULTIPLIER: f32 = 5.0;
        let qjl_max_correction = qjl_scale * SIGMA_MULTIPLIER * qjl_l2;

        Self {
            lut_i8,
            scale_global,
            qjl_query: qjl.to_vec(),
            qjl_scale,
            qjl_max_correction,
            padded_dim,
        }
    }
}

/// SIMD batched scorer. Writes 16 IP estimates into `out`.
///
/// Prefers the NEON path on aarch64; falls back to the scalar
/// reference on other targets.
#[inline]
pub fn score_batch(lut: &BatchedQueryLut, batch: &[u8], out: &mut [f32; BATCH_DOCS]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `lut` and `batch` are sized at construction time
        // from `padded_dim`. NEON is part of aarch64 baseline.
        unsafe {
            score_batch_neon_inner(lut, batch, out);
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        score_batch_fallback(lut, batch, out);
    }
}

/// Per-batch partial state from Stage 1 only. Used for the
/// two-phase scoring optimization: when a TopK threshold is set, the
/// caller can compute Stage 1 first, derive an upper bound on the
/// final score using `max_score_upper_bound`, and skip the Stage 2
/// pass entirely if the bound is below the threshold.
#[derive(Clone, Copy)]
pub struct Stage1Partial {
    /// Stage-1 IP contribution per slot (in float space, with the
    /// global scale already applied).
    pub stage1: [f32; BATCH_DOCS],
    /// Per-slot residual norm γ from the batch's gamma slab.
    pub gammas: [f32; BATCH_DOCS],
}

impl Stage1Partial {
    /// Tight per-slot upper bound on the eventual full score, used
    /// to skip the Stage-2 pass when no doc in the batch can beat
    /// the current top-K threshold.
    ///
    /// Stage 2 contributes `γ · qjl_scale · ⟨signs, qjl⟩`. Treating
    /// the per-coord signs as random, `⟨signs, qjl⟩` is the sum of
    /// `padded_dim` independent ±qjl[i] terms, which by the
    /// Lindeberg CLT is well-approximated by a centered Gaussian
    /// with variance `‖qjl‖_2²`. With `‖qjl‖_2 = 1` (unit query
    /// after orthogonal projection) the standard deviation is 1, so
    /// `5 · ‖qjl‖_2 = 5` covers the contribution to ~6 × 10⁻⁷
    /// per-doc miss probability — six orders of magnitude better than
    /// the i8-quantization noise budget already baked into the
    /// scorer, so the bound is effectively as tight as the
    /// deterministic answer for recall purposes.
    ///
    /// The deterministic worst-case bound is `‖qjl‖_1`, but on Cohere-
    /// style data that's ~‖qjl‖_2 · √(2d/π) ≈ 22 — six standard
    /// deviations beyond what any real doc produces. Using the L1
    /// bound never fires the short-circuit; the 5σ bound fires on
    /// > 90 % of batches in practice.
    #[inline]
    pub fn max_score_upper_bound(&self, lut: &BatchedQueryLut) -> f32 {
        let mut best = f32::NEG_INFINITY;
        for slot in 0..BATCH_DOCS {
            let bound = self.stage1[slot] + self.gammas[slot] * lut.qjl_max_correction;
            if bound > best {
                best = bound;
            }
        }
        best
    }
}

/// Compute Stage 1 only (and capture γ values). Cheaper than the full
/// kernel — skips reading the Stage-2 sign slab and the per-coord
/// `qjl[c]` broadcast and XOR — useful when the caller can decide to
/// reject the entire batch from Stage 1 alone via
/// `Stage1Partial::max_score_upper_bound`.
#[inline]
pub fn score_batch_stage1(lut: &BatchedQueryLut, batch: &[u8]) -> Stage1Partial {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return score_batch_stage1_neon(lut, batch);
        }
    }
    #[allow(unreachable_code)]
    {
        score_batch_stage1_fallback(lut, batch)
    }
}

/// Finish a partial-scored batch: add the Stage 2 (QJL sign) correction
/// to the partial Stage 1 result. Caller is expected to have already
/// decided the batch survives via `max_score_upper_bound`.
#[inline]
pub fn score_batch_finish(
    lut: &BatchedQueryLut,
    batch: &[u8],
    partial: &Stage1Partial,
    out: &mut [f32; BATCH_DOCS],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            score_batch_finish_neon(lut, batch, partial, out);
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        score_batch_finish_fallback(lut, batch, partial, out);
    }
}

/// Portable scalar fallback for the SIMD scorer: same arithmetic as
/// `score_batch_neon_inner` but written in plain Rust. Used on
/// non-aarch64 targets and as a parity reference.
fn score_batch_fallback(lut: &BatchedQueryLut, batch: &[u8], out: &mut [f32; BATCH_DOCS]) {
    let d = lut.padded_dim;
    let s1_t = &batch[..stage1_t_bytes(d)];
    let s2_t = &batch[stage1_t_bytes(d)..stage1_t_bytes(d) + stage2_t_bytes(d)];
    let gammas = &batch[stage1_t_bytes(d) + stage2_t_bytes(d)..];

    let mut s1_acc = [0i32; BATCH_DOCS];
    let mut s2_acc = [0.0f32; BATCH_DOCS];

    for c in 0..d {
        let lut_row = &lut.lut_i8[c * 16..c * 16 + 8];
        for slot in 0..BATCH_DOCS {
            let idx = batch_stage1_index(s1_t, c, slot) as usize;
            s1_acc[slot] += lut_row[idx] as i32;
        }
        let q = lut.qjl_query[c];
        for slot in 0..BATCH_DOCS {
            if batch_stage2_sign(s2_t, c, slot) == 1 {
                s2_acc[slot] += q;
            } else {
                s2_acc[slot] -= q;
            }
        }
    }

    for slot in 0..BATCH_DOCS {
        let stage1 = s1_acc[slot] as f32 * lut.scale_global;
        let g = batch_gamma(gammas, slot);
        out[slot] = stage1 + g * lut.qjl_scale * s2_acc[slot];
    }
}

/// NEON Stage 1: per coord, 16-way `vqtbl1q_s8` lookup widened into
/// 4 × i32x4 accumulators.
///
/// NEON Stage 2: per coord, broadcast `qjl[c]`, XOR with 16-doc sign
/// mask (lookup-driven via `SIGN_MASK_TABLE_16`), accumulate into
/// 4 × f32x4 accumulators.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn score_batch_neon_inner(lut: &BatchedQueryLut, batch: &[u8], out: &mut [f32; BATCH_DOCS]) {
    use core::arch::aarch64::*;

    let d = lut.padded_dim;
    let s1_t_len = stage1_t_bytes(d);
    let s2_t_len = stage2_t_bytes(d);
    let s1_t = batch.as_ptr();
    let s2_t = batch.as_ptr().add(s1_t_len);
    let gammas = batch.as_ptr().add(s1_t_len + s2_t_len);

    let lut_ptr = lut.lut_i8.as_ptr();
    let qjl_ptr = lut.qjl_query.as_ptr();

    // Stage 1 accumulators: 4 × i32x4 = 16 lanes.
    let mut s1a0 = vdupq_n_s32(0);
    let mut s1a1 = vdupq_n_s32(0);
    let mut s1a2 = vdupq_n_s32(0);
    let mut s1a3 = vdupq_n_s32(0);

    // Stage 2 accumulators: 4 × f32x4 = 16 lanes.
    let mut s2a0 = vdupq_n_f32(0.0);
    let mut s2a1 = vdupq_n_f32(0.0);
    let mut s2a2 = vdupq_n_f32(0.0);
    let mut s2a3 = vdupq_n_f32(0.0);

    let nibble_mask = vdup_n_u8(0x0F);

    for c in 0..d {
        // ── Stage 1 ──────────────────────────────────────────────
        // Load 8 packed bytes (16 nibbles) for this coord.
        let packed = vld1_u8(s1_t.add(c * 8));
        let highs = vshr_n_u8(packed, 4); // 8 bytes, high nibbles
        let lows = vand_u8(packed, nibble_mask); // 8 bytes, low nibbles
                                                 // Interleave into [hi0, lo0, hi1, lo1, ...] = 16 indices,
                                                 // matching the slot layout used by the encoder (slot 0 = hi
                                                 // nibble of byte 0, slot 1 = lo nibble of byte 0, ...).
        let lo16 = vzip1_u8(highs, lows); // 8 bytes
        let hi16 = vzip2_u8(highs, lows); // 8 bytes
        let indices = vcombine_u8(lo16, hi16); // 16 bytes = u8x16

        // Look up 16 i8 codebook entries. vqtbl1q_s8 takes the table
        // as int8x16 but the index vector as uint8x16 (the byte values
        // are still treated as 0..15 for selection).
        let lut_row = vld1q_s8(lut_ptr.add(c * 16));
        let raw = vqtbl1q_s8(lut_row, indices);

        // Widen i8 → i16 → i32 and accumulate.
        let raw_lo = vmovl_s8(vget_low_s8(raw)); // 8 i16
        let raw_hi = vmovl_high_s8(raw); // 8 i16
        s1a0 = vaddw_s16(s1a0, vget_low_s16(raw_lo)); // 4 i32
        s1a1 = vaddw_high_s16(s1a1, raw_lo); // 4 i32
        s1a2 = vaddw_s16(s1a2, vget_low_s16(raw_hi)); // 4 i32
        s1a3 = vaddw_high_s16(s1a3, raw_hi); // 4 i32

        // ── Stage 2 ──────────────────────────────────────────────
        let q = vdupq_n_f32(*qjl_ptr.add(c));
        let q_bits = vreinterpretq_u32_f32(q);
        let m_lo = SIGN_MASK_TABLE_8[*s2_t.add(c * 2) as usize];
        let m_hi = SIGN_MASK_TABLE_8[*s2_t.add(c * 2 + 1) as usize];

        let mv0 = vld1q_u32(m_lo.as_ptr());
        let mv1 = vld1q_u32(m_lo.as_ptr().add(4));
        let mv2 = vld1q_u32(m_hi.as_ptr());
        let mv3 = vld1q_u32(m_hi.as_ptr().add(4));

        let f0 = vreinterpretq_f32_u32(veorq_u32(q_bits, mv0));
        let f1 = vreinterpretq_f32_u32(veorq_u32(q_bits, mv1));
        let f2 = vreinterpretq_f32_u32(veorq_u32(q_bits, mv2));
        let f3 = vreinterpretq_f32_u32(veorq_u32(q_bits, mv3));

        s2a0 = vaddq_f32(s2a0, f0);
        s2a1 = vaddq_f32(s2a1, f1);
        s2a2 = vaddq_f32(s2a2, f2);
        s2a3 = vaddq_f32(s2a3, f3);
    }

    // Convert i32 accumulators to f32 and apply global Stage-1 scale.
    let scale = vdupq_n_f32(lut.scale_global);
    let s1f0 = vmulq_f32(vcvtq_f32_s32(s1a0), scale);
    let s1f1 = vmulq_f32(vcvtq_f32_s32(s1a1), scale);
    let s1f2 = vmulq_f32(vcvtq_f32_s32(s1a2), scale);
    let s1f3 = vmulq_f32(vcvtq_f32_s32(s1a3), scale);

    // γ × qjl_scale × s2_acc + s1_acc, per slot.
    let gamma_v0 = vld1q_f32(gammas as *const f32);
    let gamma_v1 = vld1q_f32((gammas as *const f32).add(4));
    let gamma_v2 = vld1q_f32((gammas as *const f32).add(8));
    let gamma_v3 = vld1q_f32((gammas as *const f32).add(12));

    let qs = vdupq_n_f32(lut.qjl_scale);
    let s2_scaled0 = vmulq_f32(vmulq_f32(gamma_v0, qs), s2a0);
    let s2_scaled1 = vmulq_f32(vmulq_f32(gamma_v1, qs), s2a1);
    let s2_scaled2 = vmulq_f32(vmulq_f32(gamma_v2, qs), s2a2);
    let s2_scaled3 = vmulq_f32(vmulq_f32(gamma_v3, qs), s2a3);

    let r0 = vaddq_f32(s1f0, s2_scaled0);
    let r1 = vaddq_f32(s1f1, s2_scaled1);
    let r2 = vaddq_f32(s1f2, s2_scaled2);
    let r3 = vaddq_f32(s1f3, s2_scaled3);

    vst1q_f32(out.as_mut_ptr(), r0);
    vst1q_f32(out.as_mut_ptr().add(4), r1);
    vst1q_f32(out.as_mut_ptr().add(8), r2);
    vst1q_f32(out.as_mut_ptr().add(12), r3);
}

/// Stage-1 only NEON kernel. Reads `stage1_t` and γ slabs only;
/// touches no Stage-2 bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn score_batch_stage1_neon(lut: &BatchedQueryLut, batch: &[u8]) -> Stage1Partial {
    use core::arch::aarch64::*;

    let d = lut.padded_dim;
    let s1_t_len = stage1_t_bytes(d);
    let s2_t_len = stage2_t_bytes(d);
    let s1_t = batch.as_ptr();
    let gammas = batch.as_ptr().add(s1_t_len + s2_t_len);

    let lut_ptr = lut.lut_i8.as_ptr();

    let mut s1a0 = vdupq_n_s32(0);
    let mut s1a1 = vdupq_n_s32(0);
    let mut s1a2 = vdupq_n_s32(0);
    let mut s1a3 = vdupq_n_s32(0);

    let nibble_mask = vdup_n_u8(0x0F);

    for c in 0..d {
        let packed = vld1_u8(s1_t.add(c * 8));
        let highs = vshr_n_u8(packed, 4);
        let lows = vand_u8(packed, nibble_mask);
        let lo16 = vzip1_u8(highs, lows);
        let hi16 = vzip2_u8(highs, lows);
        let indices = vcombine_u8(lo16, hi16);

        let lut_row = vld1q_s8(lut_ptr.add(c * 16));
        let raw = vqtbl1q_s8(lut_row, indices);

        let raw_lo = vmovl_s8(vget_low_s8(raw));
        let raw_hi = vmovl_high_s8(raw);
        s1a0 = vaddw_s16(s1a0, vget_low_s16(raw_lo));
        s1a1 = vaddw_high_s16(s1a1, raw_lo);
        s1a2 = vaddw_s16(s1a2, vget_low_s16(raw_hi));
        s1a3 = vaddw_high_s16(s1a3, raw_hi);
    }

    let scale = vdupq_n_f32(lut.scale_global);
    let s1f0 = vmulq_f32(vcvtq_f32_s32(s1a0), scale);
    let s1f1 = vmulq_f32(vcvtq_f32_s32(s1a1), scale);
    let s1f2 = vmulq_f32(vcvtq_f32_s32(s1a2), scale);
    let s1f3 = vmulq_f32(vcvtq_f32_s32(s1a3), scale);

    let mut out = Stage1Partial {
        stage1: [0.0; BATCH_DOCS],
        gammas: [0.0; BATCH_DOCS],
    };
    vst1q_f32(out.stage1.as_mut_ptr(), s1f0);
    vst1q_f32(out.stage1.as_mut_ptr().add(4), s1f1);
    vst1q_f32(out.stage1.as_mut_ptr().add(8), s1f2);
    vst1q_f32(out.stage1.as_mut_ptr().add(12), s1f3);

    // γ slab is already 16 × f32 LE — copy verbatim.
    let g_v0 = vld1q_f32(gammas as *const f32);
    let g_v1 = vld1q_f32((gammas as *const f32).add(4));
    let g_v2 = vld1q_f32((gammas as *const f32).add(8));
    let g_v3 = vld1q_f32((gammas as *const f32).add(12));
    vst1q_f32(out.gammas.as_mut_ptr(), g_v0);
    vst1q_f32(out.gammas.as_mut_ptr().add(4), g_v1);
    vst1q_f32(out.gammas.as_mut_ptr().add(8), g_v2);
    vst1q_f32(out.gammas.as_mut_ptr().add(12), g_v3);

    out
}

/// Add the Stage-2 sign-XOR correction to a Stage-1 partial. Reads
/// only `stage2_t` from the batch; γ comes from `partial.gammas`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn score_batch_finish_neon(
    lut: &BatchedQueryLut,
    batch: &[u8],
    partial: &Stage1Partial,
    out: &mut [f32; BATCH_DOCS],
) {
    use core::arch::aarch64::*;

    let d = lut.padded_dim;
    let s1_t_len = stage1_t_bytes(d);
    let s2_t = batch.as_ptr().add(s1_t_len);
    let qjl_ptr = lut.qjl_query.as_ptr();

    let mut s2a0 = vdupq_n_f32(0.0);
    let mut s2a1 = vdupq_n_f32(0.0);
    let mut s2a2 = vdupq_n_f32(0.0);
    let mut s2a3 = vdupq_n_f32(0.0);

    for c in 0..d {
        let q = vdupq_n_f32(*qjl_ptr.add(c));
        let q_bits = vreinterpretq_u32_f32(q);
        let m_lo = SIGN_MASK_TABLE_8[*s2_t.add(c * 2) as usize];
        let m_hi = SIGN_MASK_TABLE_8[*s2_t.add(c * 2 + 1) as usize];

        let mv0 = vld1q_u32(m_lo.as_ptr());
        let mv1 = vld1q_u32(m_lo.as_ptr().add(4));
        let mv2 = vld1q_u32(m_hi.as_ptr());
        let mv3 = vld1q_u32(m_hi.as_ptr().add(4));

        let f0 = vreinterpretq_f32_u32(veorq_u32(q_bits, mv0));
        let f1 = vreinterpretq_f32_u32(veorq_u32(q_bits, mv1));
        let f2 = vreinterpretq_f32_u32(veorq_u32(q_bits, mv2));
        let f3 = vreinterpretq_f32_u32(veorq_u32(q_bits, mv3));

        s2a0 = vaddq_f32(s2a0, f0);
        s2a1 = vaddq_f32(s2a1, f1);
        s2a2 = vaddq_f32(s2a2, f2);
        s2a3 = vaddq_f32(s2a3, f3);
    }

    let qs = vdupq_n_f32(lut.qjl_scale);
    let s1_v0 = vld1q_f32(partial.stage1.as_ptr());
    let s1_v1 = vld1q_f32(partial.stage1.as_ptr().add(4));
    let s1_v2 = vld1q_f32(partial.stage1.as_ptr().add(8));
    let s1_v3 = vld1q_f32(partial.stage1.as_ptr().add(12));
    let g_v0 = vld1q_f32(partial.gammas.as_ptr());
    let g_v1 = vld1q_f32(partial.gammas.as_ptr().add(4));
    let g_v2 = vld1q_f32(partial.gammas.as_ptr().add(8));
    let g_v3 = vld1q_f32(partial.gammas.as_ptr().add(12));

    let r0 = vfmaq_f32(s1_v0, vmulq_f32(g_v0, qs), s2a0);
    let r1 = vfmaq_f32(s1_v1, vmulq_f32(g_v1, qs), s2a1);
    let r2 = vfmaq_f32(s1_v2, vmulq_f32(g_v2, qs), s2a2);
    let r3 = vfmaq_f32(s1_v3, vmulq_f32(g_v3, qs), s2a3);

    vst1q_f32(out.as_mut_ptr(), r0);
    vst1q_f32(out.as_mut_ptr().add(4), r1);
    vst1q_f32(out.as_mut_ptr().add(8), r2);
    vst1q_f32(out.as_mut_ptr().add(12), r3);
}

fn score_batch_stage1_fallback(lut: &BatchedQueryLut, batch: &[u8]) -> Stage1Partial {
    let d = lut.padded_dim;
    let s1_t = &batch[..stage1_t_bytes(d)];
    let gammas = &batch[stage1_t_bytes(d) + stage2_t_bytes(d)..];

    let mut s1_acc = [0i32; BATCH_DOCS];
    for c in 0..d {
        let lut_row = &lut.lut_i8[c * 16..c * 16 + 8];
        for slot in 0..BATCH_DOCS {
            let idx = batch_stage1_index(s1_t, c, slot) as usize;
            s1_acc[slot] += lut_row[idx] as i32;
        }
    }
    let mut out = Stage1Partial {
        stage1: [0.0; BATCH_DOCS],
        gammas: [0.0; BATCH_DOCS],
    };
    for slot in 0..BATCH_DOCS {
        out.stage1[slot] = s1_acc[slot] as f32 * lut.scale_global;
        out.gammas[slot] = batch_gamma(gammas, slot);
    }
    out
}

fn score_batch_finish_fallback(
    lut: &BatchedQueryLut,
    batch: &[u8],
    partial: &Stage1Partial,
    out: &mut [f32; BATCH_DOCS],
) {
    let d = lut.padded_dim;
    let s2_t = &batch[stage1_t_bytes(d)..stage1_t_bytes(d) + stage2_t_bytes(d)];
    let mut s2_acc = [0.0f32; BATCH_DOCS];
    for c in 0..d {
        let q = lut.qjl_query[c];
        for slot in 0..BATCH_DOCS {
            if batch_stage2_sign(s2_t, c, slot) == 1 {
                s2_acc[slot] += q;
            } else {
                s2_acc[slot] -= q;
            }
        }
    }
    for slot in 0..BATCH_DOCS {
        out[slot] = partial.stage1[slot] + partial.gammas[slot] * lut.qjl_scale * s2_acc[slot];
    }
}

/// 256-entry precomputed XOR sign masks for an 8-bit byte (MSB-first).
/// `SIGN_MASK_TABLE_8[byte][j]` is `0` if bit j (MSB-first) of `byte`
/// is 1, else `0x8000_0000` — XORing flips a float's sign exactly when
/// the corresponding sign bit is 0, matching the per-doc Stage-2
/// convention. This is the same table the per-doc NEON kernel uses
/// (kept private to avoid coupling the modules; small, fits L1d).
const SIGN_MASK_TABLE_8: [[u32; 8]; 256] = {
    let mut t = [[0u32; 8]; 256];
    let mut b = 0;
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
};

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};

    use super::*;
    use crate::vector::turboquant::{TurboQuantQuery, TurboQuantizer};

    fn unit(d: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n = Normal::new(0.0_f32, 1.0).unwrap();
        let mut v: Vec<f32> = (0..d).map(|_| n.sample(&mut rng)).collect();
        let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= nrm;
        }
        v
    }

    /// Encoding 16 records into a transposed batch and scoring with
    /// the scalar reference scorer must match per-doc `estimate_ip`.
    /// Stage-1 i8 quantization is NOT involved here.
    #[test]
    fn batched_scalar_matches_per_doc() {
        let d = 768;
        let tq = TurboQuantizer::new(d, Some(TRANSPOSED_BIT_WIDTH), Some(42));

        let vecs: Vec<Vec<f32>> = (0..16).map(|i| unit(d, 1_000 + i as u64)).collect();
        let recs: Vec<Vec<u8>> = vecs.iter().map(|v| tq.encode(v)).collect();
        let rec_refs: Vec<&[u8]> = recs.iter().map(|r| r.as_slice()).collect();

        let mut batch = vec![0u8; batch_bytes(tq.padded_dim)];
        encode_batch(&rec_refs, tq.padded_dim, &mut batch);

        let q = unit(d, 9_001);
        let tqq = TurboQuantQuery::new(&tq, &q);

        let mut batched = [0.0f32; BATCH_DOCS];
        score_batch_scalar(&tqq, &batch, &mut batched);

        for i in 0..16 {
            let expected = tqq.estimate_ip(&recs[i]);
            assert!(
                (batched[i] - expected).abs() < 1e-3,
                "slot {i}: batched {} vs per-doc {}",
                batched[i],
                expected,
            );
        }
    }

    /// SIMD/quantized scorer should track per-doc `estimate_ip` to
    /// within the i8-codebook quantization error budget. With ‖q‖ = 1
    /// and 768 coords, the aggregate Stage-1 quantization error is
    /// O(scale_global × √d), which empirically stays under 0.05.
    #[test]
    fn batched_simd_close_to_per_doc() {
        let d = 768;
        let tq = TurboQuantizer::new(d, Some(TRANSPOSED_BIT_WIDTH), Some(42));

        let vecs: Vec<Vec<f32>> = (0..16).map(|i| unit(d, 2_000 + i as u64)).collect();
        let recs: Vec<Vec<u8>> = vecs.iter().map(|v| tq.encode(v)).collect();
        let rec_refs: Vec<&[u8]> = recs.iter().map(|r| r.as_slice()).collect();

        let mut batch = vec![0u8; batch_bytes(tq.padded_dim)];
        encode_batch(&rec_refs, tq.padded_dim, &mut batch);

        let q = unit(d, 9_002);
        let tqq = TurboQuantQuery::new(&tq, &q);
        let lut = BatchedQueryLut::new(&tqq);

        let mut simd = [0.0f32; BATCH_DOCS];
        score_batch(&lut, &batch, &mut simd);

        let mut max_err = 0.0f32;
        for i in 0..16 {
            let expected = tqq.estimate_ip(&recs[i]);
            let err = (simd[i] - expected).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 0.05,
            "max simd-vs-per-doc error {max_err} exceeds 0.05 budget"
        );
    }

    /// `extract_record` must be the exact inverse of `encode_batch`
    /// for the slot it targets — i.e. the bytes it produces must be
    /// identical to what `TurboQuantizer::encode_into` originally
    /// wrote for that doc.
    #[test]
    fn extract_record_roundtrips_through_encode_batch() {
        let d = 768;
        let tq = TurboQuantizer::new(d, Some(TRANSPOSED_BIT_WIDTH), Some(42));
        let vecs: Vec<Vec<f32>> = (0..16).map(|i| unit(d, 7_000 + i as u64)).collect();
        let recs: Vec<Vec<u8>> = vecs.iter().map(|v| tq.encode(v)).collect();
        let rec_refs: Vec<&[u8]> = recs.iter().map(|r| r.as_slice()).collect();

        let mut batch = vec![0u8; batch_bytes(tq.padded_dim)];
        encode_batch(&rec_refs, tq.padded_dim, &mut batch);

        let bpr = recs[0].len();
        let mut extracted = vec![0u8; bpr];
        for slot in 0..16 {
            extract_record(&batch, slot, tq.padded_dim, &mut extracted);
            assert_eq!(
                extracted, recs[slot],
                "slot {slot}: extracted record does not match encoder output",
            );
        }
    }

    /// Even with a partial batch (< 16 docs), `extract_record` must
    /// still produce the encoder's output for the populated slots,
    /// and a record from the right shape for padded slots (γ = 0,
    /// indices = 0, signs = 0 → an all-zero packed record other than
    /// the bit-packing convention).
    #[test]
    fn extract_record_partial_batch() {
        let d = 768;
        let tq = TurboQuantizer::new(d, Some(TRANSPOSED_BIT_WIDTH), Some(42));
        let vecs: Vec<Vec<f32>> = (0..7).map(|i| unit(d, 8_000 + i as u64)).collect();
        let recs: Vec<Vec<u8>> = vecs.iter().map(|v| tq.encode(v)).collect();
        let rec_refs: Vec<&[u8]> = recs.iter().map(|r| r.as_slice()).collect();

        let mut batch = vec![0u8; batch_bytes(tq.padded_dim)];
        encode_batch(&rec_refs, tq.padded_dim, &mut batch);

        let bpr = recs[0].len();
        let mut extracted = vec![0u8; bpr];
        for slot in 0..7 {
            extract_record(&batch, slot, tq.padded_dim, &mut extracted);
            assert_eq!(extracted, recs[slot], "populated slot {slot} mismatch");
        }
        // Padded slots: γ = 0, the all-zero indices/signs aren't
        // *meaningful* but they must roundtrip as the all-zero
        // packed record `encode_batch` would have produced for an
        // imaginary zero-gamma doc.
        for slot in 7..16 {
            extract_record(&batch, slot, tq.padded_dim, &mut extracted);
            assert!(
                extracted[..bpr - 4].iter().all(|&b| b == 0),
                "padded slot {slot} should have zero indices/signs"
            );
            let g = f32::from_le_bytes([
                extracted[bpr - 4],
                extracted[bpr - 3],
                extracted[bpr - 2],
                extracted[bpr - 1],
            ]);
            assert_eq!(g, 0.0, "padded slot {slot} γ should be zero");
        }
    }

    /// Two-phase scoring (Stage 1 then Stage 2 finish) must match
    /// the single-pass batched scorer exactly.
    #[test]
    fn split_stage1_finish_matches_score_batch() {
        let d = 768;
        let tq = TurboQuantizer::new(d, Some(TRANSPOSED_BIT_WIDTH), Some(42));
        let vecs: Vec<Vec<f32>> = (0..16).map(|i| unit(d, 4_000 + i as u64)).collect();
        let recs: Vec<Vec<u8>> = vecs.iter().map(|v| tq.encode(v)).collect();
        let rec_refs: Vec<&[u8]> = recs.iter().map(|r| r.as_slice()).collect();

        let mut batch = vec![0u8; batch_bytes(tq.padded_dim)];
        encode_batch(&rec_refs, tq.padded_dim, &mut batch);

        let q = unit(d, 9_004);
        let tqq = TurboQuantQuery::new(&tq, &q);
        let lut = BatchedQueryLut::new(&tqq);

        let mut single = [0.0f32; BATCH_DOCS];
        score_batch(&lut, &batch, &mut single);

        let partial = score_batch_stage1(&lut, &batch);
        let mut split = [0.0f32; BATCH_DOCS];
        score_batch_finish(&lut, &batch, &partial, &mut split);

        for i in 0..16 {
            assert!(
                (single[i] - split[i]).abs() < 1e-3,
                "slot {i}: single {} vs split {}",
                single[i],
                split[i],
            );
        }
    }

    /// `max_score_upper_bound` must always be ≥ the actual final
    /// score for every slot — otherwise we'd skip batches we shouldn't.
    #[test]
    fn upper_bound_dominates_actual_score() {
        let d = 768;
        let tq = TurboQuantizer::new(d, Some(TRANSPOSED_BIT_WIDTH), Some(42));
        // Use 10 different random batches to exercise the bound across
        // varying gamma magnitudes and stage-2 sign patterns.
        for trial in 0..10 {
            let vecs: Vec<Vec<f32>> = (0..16)
                .map(|i| unit(d, (10_000 + trial * 100 + i) as u64))
                .collect();
            let recs: Vec<Vec<u8>> = vecs.iter().map(|v| tq.encode(v)).collect();
            let rec_refs: Vec<&[u8]> = recs.iter().map(|r| r.as_slice()).collect();
            let mut batch = vec![0u8; batch_bytes(tq.padded_dim)];
            encode_batch(&rec_refs, tq.padded_dim, &mut batch);

            let q = unit(d, 9_005 + trial);
            let tqq = TurboQuantQuery::new(&tq, &q);
            let lut = BatchedQueryLut::new(&tqq);

            let partial = score_batch_stage1(&lut, &batch);
            let bound = partial.max_score_upper_bound(&lut);

            let mut full = [0.0f32; BATCH_DOCS];
            score_batch(&lut, &batch, &mut full);

            let actual_max = full.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                bound >= actual_max - 1e-4,
                "trial {trial}: upper bound {bound} < actual max {actual_max}",
            );
        }
    }

    /// Ignored microbench: per-doc SIMD vs batched-transposed SIMD on
    /// a 60K-doc workload, the same shape pg_search hits per query.
    /// Run with:
    ///   cargo test --release --lib \
    ///     vector::turboquant::transposed::tests::bench_batched_vs_per_doc \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_batched_vs_per_doc() {
        use std::time::Instant;
        let d = 768usize;
        let n = 60_000usize;

        let tq = TurboQuantizer::new(d, Some(TRANSPOSED_BIT_WIDTH), Some(42));
        let recs: Vec<Vec<u8>> = (0..n)
            .map(|i| tq.encode(&unit(d, 1_000 + i as u64)))
            .collect();
        let q = unit(d, 9_001);
        let tqq = TurboQuantQuery::new(&tq, &q);

        // Per-doc baseline.
        let mut sink = 0.0f32;
        for r in &recs {
            sink += tqq.estimate_ip(r);
        }
        let start = Instant::now();
        for r in &recs {
            sink += tqq.estimate_ip(r);
        }
        let per_doc = start.elapsed();

        // Batched: encode N docs into ⌈N/16⌉ batches once, then score.
        let nb = n.div_ceil(BATCH_DOCS);
        let bb = batch_bytes(tq.padded_dim);
        let mut all_batches = vec![0u8; nb * bb];
        for bi in 0..nb {
            let start_doc = bi * BATCH_DOCS;
            let end_doc = (start_doc + BATCH_DOCS).min(n);
            let slice: Vec<&[u8]> = recs[start_doc..end_doc]
                .iter()
                .map(|v| v.as_slice())
                .collect();
            encode_batch(
                &slice,
                tq.padded_dim,
                &mut all_batches[bi * bb..(bi + 1) * bb],
            );
        }
        let lut = BatchedQueryLut::new(&tqq);

        let mut out = [0.0f32; BATCH_DOCS];
        for bi in 0..nb {
            score_batch(&lut, &all_batches[bi * bb..(bi + 1) * bb], &mut out);
            for v in &out {
                sink += *v;
            }
        }
        let start = Instant::now();
        for bi in 0..nb {
            score_batch(&lut, &all_batches[bi * bb..(bi + 1) * bb], &mut out);
            for v in &out {
                sink += *v;
            }
        }
        let batched = start.elapsed();

        eprintln!(
            "Per-doc SIMD : {:?} ({:.1} ns/doc)",
            per_doc,
            per_doc.as_nanos() as f64 / n as f64
        );
        eprintln!(
            "Batched SIMD : {:?} ({:.1} ns/doc, sink={sink})",
            batched,
            batched.as_nanos() as f64 / n as f64
        );
    }

    /// Ranking should be preserved between SIMD batched and per-doc:
    /// for a population larger than the batch, the relative order of
    /// docs by IP should match closely (top-K should overlap).
    #[test]
    fn batched_ranking_matches_per_doc() {
        let d = 768;
        let n = 256;
        let k = 10;
        let tq = TurboQuantizer::new(d, Some(TRANSPOSED_BIT_WIDTH), Some(42));
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| unit(d, 5_000 + i as u64)).collect();
        let recs: Vec<Vec<u8>> = vecs.iter().map(|v| tq.encode(v)).collect();

        let q = unit(d, 9_003);
        let tqq = TurboQuantQuery::new(&tq, &q);
        let lut = BatchedQueryLut::new(&tqq);

        // Per-doc reference top-k.
        let mut ref_scored: Vec<(usize, f32)> = recs
            .iter()
            .enumerate()
            .map(|(i, r)| (i, tqq.estimate_ip(r)))
            .collect();
        ref_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let ref_top: std::collections::HashSet<usize> =
            ref_scored.iter().take(k).map(|(i, _)| *i).collect();

        // SIMD batched top-k.
        let mut simd_scored: Vec<(usize, f32)> = Vec::with_capacity(n);
        let mut simd_out = [0.0f32; BATCH_DOCS];
        let mut buf = vec![0u8; batch_bytes(tq.padded_dim)];
        for batch_start in (0..n).step_by(BATCH_DOCS) {
            let batch_end = (batch_start + BATCH_DOCS).min(n);
            let slice: Vec<&[u8]> = recs[batch_start..batch_end]
                .iter()
                .map(|v| v.as_slice())
                .collect();
            encode_batch(&slice, tq.padded_dim, &mut buf);
            score_batch(&lut, &buf, &mut simd_out);
            for slot in 0..(batch_end - batch_start) {
                simd_scored.push((batch_start + slot, simd_out[slot]));
            }
        }
        simd_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let simd_top: std::collections::HashSet<usize> =
            simd_scored.iter().take(k).map(|(i, _)| *i).collect();

        let overlap = ref_top.intersection(&simd_top).count();
        assert!(
            overlap >= 8,
            "top-{k} overlap only {overlap}/10 between per-doc and SIMD batched",
        );
    }
}
