//! Sign-plane encoding and popcount scoring kernels.

use crate::model::f16::{f16_to_f32, f32_to_f16};

#[derive(Clone, Debug)]
/// Quantized query bitplanes and affine reconstruction parameters.
pub struct QueryPlanes {
    /// Query-code bitplanes.
    pub planes: Vec<Vec<u64>>,
    /// Minimum query value.
    pub lo: f32,
    /// Query-code step.
    pub delta: f32,
    /// Sum of query codes.
    pub sum_codes: u64,
    error_squared: f64,
    d: usize,
}

impl QueryPlanes {
    /// Returns the query reconstruction's squared error.
    pub fn error_squared(&self) -> f64 {
        self.error_squared
    }
}

fn packed_words(d: usize) -> usize {
    assert!(d > 0);
    d.div_ceil(64)
}

fn tail_is_zero(words: &[u64], d: usize) -> bool {
    if d == 0 || words.len() != packed_words(d) {
        return false;
    }
    let tail = d % 64;
    tail == 0 || words.last().is_some_and(|word| word >> tail == 0)
}

/// Packs vector signs into words.
///
/// # Panics
///
/// Panics when the input is empty or the output word count is invalid.
pub fn pack(y: &[f32], out: &mut [u64]) {
    assert!(!y.is_empty());
    assert_eq!(out.len(), packed_words(y.len()));
    out.fill(0);
    for (i, &value) in y.iter().enumerate() {
        if value > 0.0 {
            out[i / 64] |= 1_u64 << (i % 64);
        }
    }
    debug_assert!(tail_is_zero(out, y.len()));
}

/// Unpacks words into unit signs.
///
/// # Panics
///
/// Panics when `d` is zero or the packed word count is invalid.
pub fn unpack(bits: &[u64], d: usize) -> Vec<f32> {
    assert!(d > 0);
    assert_eq!(bits.len(), packed_words(d));
    debug_assert!(tail_is_zero(bits, d));
    (0..d)
        .map(|i| {
            if bits[i / 64] & (1_u64 << (i % 64)) != 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Encode signs and return the mean-absolute-value scale after f16 rounding.
///
/// # Panics
///
/// Panics when the input is empty or the output word count is invalid.
pub fn encode(y: &[f32], out_bits: &mut [u64]) -> u16 {
    f32_to_f16(encode_f32(y, out_bits))
}

/// Encode signs and return the exact mean-absolute-value scale.
///
/// # Panics
///
/// Panics when the input is empty or the output word count is invalid.
pub fn encode_f32(y: &[f32], out_bits: &mut [u64]) -> f32 {
    assert!(!y.is_empty());
    assert_eq!(out_bits.len(), packed_words(y.len()));
    pack(y, out_bits);
    if y.iter().all(|&value| value == 0.0) {
        return 0.0;
    }
    y.iter().map(|value| value.abs()).sum::<f32>() / y.len() as f32
}

/// Hamming distance between two sign-code vectors.
///
/// # Panics
///
/// Panics when either input is empty or their word counts differ.
pub fn score_sym(x: &[u64], q: &[u64]) -> u32 {
    assert!(!x.is_empty());
    assert_eq!(x.len(), q.len());
    x.iter().zip(q).map(|(&a, &b)| (a ^ b).count_ones()).sum()
}

/// Quantizes a query into affine bitplanes.
///
/// # Panics
///
/// Panics when the query is empty or `bq` is outside `1..=8`.
pub fn prepare_query(u: &[f32], bq: u8) -> QueryPlanes {
    assert!(!u.is_empty());
    assert!((1..=8).contains(&bq));
    debug_assert!(u.iter().all(|value| value.is_finite()));

    let lo = u.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = u.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let levels = (1_u16 << bq) - 1;
    let delta = if hi == lo {
        0.0
    } else {
        (hi - lo) / f32::from(levels)
    };
    let mut planes = vec![vec![0_u64; packed_words(u.len())]; bq as usize];
    let mut sum_codes = 0_u64;
    let mut error_squared = 0.0_f64;

    if delta != 0.0 {
        for (i, &value) in u.iter().enumerate() {
            let code = (((value - lo) / delta).round() as u16).min(levels);
            sum_codes += u64::from(code);
            let reconstructed = lo + delta * f32::from(code);
            error_squared += f64::from(value - reconstructed).powi(2);
            for bit in 0..bq {
                if code & (1_u16 << bit) != 0 {
                    planes[bit as usize][i / 64] |= 1_u64 << (i % 64);
                }
            }
        }
    }

    debug_assert!(planes.iter().all(|plane| tail_is_zero(plane, u.len())));
    QueryPlanes {
        planes,
        lo,
        delta,
        sum_codes,
        error_squared,
        d: u.len(),
    }
}

/// Return positive-sign count `P` and query-code sum over positive signs `S`.
///
/// # Panics
///
/// Panics when code and query bitplane shapes disagree.
pub fn score_asym(x: &[u64], q: &QueryPlanes) -> (u32, u64) {
    assert!(!x.is_empty());
    assert!(!q.planes.is_empty());
    assert!(q.planes.iter().all(|plane| plane.len() == x.len()));
    debug_assert!(tail_is_zero(x, q.d));
    debug_assert!(q.planes.iter().all(|plane| tail_is_zero(plane, q.d)));
    let positives = x.iter().map(|word| word.count_ones()).sum();
    let weighted_sum = q
        .planes
        .iter()
        .enumerate()
        .map(|(bit, plane)| {
            let intersections: u64 = x
                .iter()
                .zip(plane)
                .map(|(&signs, &query)| u64::from((signs & query).count_ones()))
                .sum();
            intersections << bit
        })
        .sum();
    (positives, weighted_sum)
}

/// Scores one sign row with a binary16 scale.
///
/// # Panics
///
/// Panics when code and query bitplane shapes disagree.
pub fn estimate_asym(x: &[u64], scale: u16, q: &QueryPlanes) -> f32 {
    f16_to_f32(scale) * estimate_asym_unscaled(x, q)
}

/// Scores one sign row with a binary32 scale.
///
/// # Panics
///
/// Panics when code and query bitplane shapes disagree.
pub fn estimate_asym_f32(x: &[u64], scale: f32, q: &QueryPlanes) -> f32 {
    scale * estimate_asym_unscaled(x, q)
}

/// Score one sign row without applying its stored scale.
///
/// # Panics
///
/// Panics when code and query bitplane shapes disagree.
#[inline]
pub fn estimate_asym_unscaled(x: &[u64], q: &QueryPlanes) -> f32 {
    let (positives, weighted_sum) = score_asym(x, q);
    finish_asym(positives, weighted_sum, q)
}

/// Affine epilogue shared by every asymmetric scorer; a single body keeps the integer-to-f32
/// rounding identical across the scalar and NEON kernels.
#[inline(always)]
fn finish_asym(positives: u32, weighted_sum: u64, q: &QueryPlanes) -> f32 {
    let sign_sum = i64::from(positives) * 2 - q.d as i64;
    let code_sum = i128::from(weighted_sum) * 2 - i128::from(q.sum_codes);
    q.lo * sign_sum as f32 + q.delta * code_sum as f32
}

/// Scalar four-plane row kernel; the portable counterpart of the [`neon`] kernels.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline(always)]
fn score_row4(row: &[u64], planes: [&[u64]; 4]) -> (u32, u64) {
    let [p0, p1, p2, p3] = planes;
    let mut positives = 0_u32;
    let mut weighted_sum = 0_u64;
    for ((((signs, &q0), &q1), &q2), &q3) in row.iter().copied().zip(p0).zip(p1).zip(p2).zip(p3) {
        positives += signs.count_ones();
        weighted_sum += u64::from((signs & q0).count_ones());
        weighted_sum += u64::from((signs & q1).count_ones()) << 1;
        weighted_sum += u64::from((signs & q2).count_ones()) << 2;
        weighted_sum += u64::from((signs & q3).count_ones()) << 3;
    }
    (positives, weighted_sum)
}

/// Scores a contiguous fixed-stride sign batch without row scales.
///
/// # Panics
///
/// Panics when row, bitplane, stride, or output shapes disagree.
#[inline(always)]
pub fn estimate_asym_batch_unscaled(
    rows: &[u64],
    words_per_row: usize,
    q: &QueryPlanes,
    out: &mut [f32],
) {
    assert!(words_per_row > 0);
    assert_eq!(rows.len(), out.len() * words_per_row);
    assert!(!q.planes.is_empty());
    assert!(q.planes.iter().all(|plane| plane.len() == words_per_row));

    if let [p0, p1, p2, p3] = q.planes.as_slice() {
        let planes = [p0.as_slice(), p1.as_slice(), p2.as_slice(), p3.as_slice()];
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        neon::batch(neon::accumulator(), rows, words_per_row, planes, q, out);
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        for (row, score) in rows.chunks_exact(words_per_row).zip(out) {
            let (positives, weighted_sum) = score_row4(row, planes);
            *score = finish_asym(positives, weighted_sum, q);
        }
        return;
    }

    for (row, score) in rows.chunks_exact(words_per_row).zip(out) {
        let mut positives = 0_u32;
        let mut weighted_sum = 0_u64;
        for (word_idx, &signs) in row.iter().enumerate() {
            positives += signs.count_ones();
            for (bit, plane) in q.planes.iter().enumerate() {
                weighted_sum += u64::from((signs & plane[word_idx]).count_ones()) << bit;
            }
        }
        *score = finish_asym(positives, weighted_sum, q);
    }
}

/// Scores selected sign rows from a fixed-stride posting range.
///
/// # Panics
///
/// Panics when row, index, bitplane, stride, or output shapes disagree.
#[inline(always)]
pub fn estimate_asym_batch_unscaled_indexed(
    rows: &[u64],
    words_per_row: usize,
    row_offsets: &[usize],
    q: &QueryPlanes,
    out: &mut [f32],
) {
    assert!(words_per_row > 0);
    assert_eq!(rows.len() % words_per_row, 0);
    assert_eq!(row_offsets.len(), out.len());
    assert!(row_offsets
        .iter()
        .all(|&row| row < rows.len() / words_per_row));
    assert!(!q.planes.is_empty());
    assert!(q.planes.iter().all(|plane| plane.len() == words_per_row));

    if let [p0, p1, p2, p3] = q.planes.as_slice() {
        let planes = [p0.as_slice(), p1.as_slice(), p2.as_slice(), p3.as_slice()];
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        neon::indexed(
            neon::accumulator(),
            rows,
            words_per_row,
            row_offsets,
            planes,
            q,
            out,
        );
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        for (&row_offset, score) in row_offsets.iter().zip(out) {
            let row_base = row_offset * words_per_row;
            let row = &rows[row_base..row_base + words_per_row];
            let (positives, weighted_sum) = score_row4(row, planes);
            *score = finish_asym(positives, weighted_sum, q);
        }
        return;
    }

    for (&row_offset, score) in row_offsets.iter().zip(out) {
        let row_base = row_offset * words_per_row;
        let row = &rows[row_base..row_base + words_per_row];
        let mut positives = 0_u32;
        let mut weighted_sum = 0_u64;
        for (word_idx, &signs) in row.iter().enumerate() {
            positives += signs.count_ones();
            for (bit, plane) in q.planes.iter().enumerate() {
                weighted_sum += u64::from((signs & plane[word_idx]).count_ones()) << bit;
            }
        }
        *score = finish_asym(positives, weighted_sum, q);
    }
}

/// NEON popcount kernels for the four-plane batch scorers.
///
/// Per 16-byte chunk per row: 4 AND, 5 CNT, 5 accumulates. With FEAT_DotProd each count feeds its
/// own u32-lane accumulator through UDOT against an all-ones vector; the baseline fallback
/// pairwise-accumulates into u16 lanes (UADALP) and flushes them before they can overflow. Blocks
/// of two rows share each plane load, an odd trailing word is counted in scalar code, and the
/// exact integer totals feed [`finish_asym`], so scores are bit-identical to the scalar kernels.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use std::arch::aarch64::{
        uint32x4_t, uint8x16_t, vaddlvq_u16, vaddvq_u32, vandq_u8, vcntq_u8, vdupq_n_u16,
        vdupq_n_u32, vdupq_n_u8, vld1q_u8, vpadalq_u8,
    };

    use super::{finish_asym, QueryPlanes};

    /// UADALP adds at most 16 to a u16 lane per chunk, so this many chunks stay under 65536.
    const WIDEN_FLUSH_CHUNKS: usize = 4000;

    /// Count-reduction instruction, picked once per batch from CPU features.
    #[derive(Clone, Copy)]
    pub(super) enum Accumulator {
        /// UDOT straight into u32 lanes; requires FEAT_DotProd.
        Dot,
        /// UADALP into u16 lanes with periodic flushes; baseline NEON.
        Widen,
    }

    pub(super) fn accumulator() -> Accumulator {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            Accumulator::Dot
        } else {
            Accumulator::Widen
        }
    }

    /// Scores a contiguous fixed-stride batch.
    pub(super) fn batch(
        acc: Accumulator,
        rows: &[u64],
        words_per_row: usize,
        planes: [&[u64]; 4],
        q: &QueryPlanes,
        out: &mut [f32],
    ) {
        assert_eq!(rows.len(), out.len() * words_per_row);
        assert_shapes(words_per_row, planes);
        match acc {
            // SAFETY: the assertions above keep every row index below `out.len()` in bounds, and
            // `Dot` is only constructed behind FEAT_DotProd detection.
            Accumulator::Dot => unsafe { drive_dot(rows, words_per_row, |i| i, planes, q, out) },
            // SAFETY: bounds as above.
            Accumulator::Widen => unsafe {
                drive_widen(rows, words_per_row, |i| i, planes, q, out)
            },
        }
    }

    /// Scores selected rows of a fixed-stride posting range.
    pub(super) fn indexed(
        acc: Accumulator,
        rows: &[u64],
        words_per_row: usize,
        row_offsets: &[usize],
        planes: [&[u64]; 4],
        q: &QueryPlanes,
        out: &mut [f32],
    ) {
        assert_eq!(rows.len() % words_per_row, 0);
        assert_eq!(row_offsets.len(), out.len());
        assert!(row_offsets
            .iter()
            .all(|&row| row < rows.len() / words_per_row));
        assert_shapes(words_per_row, planes);
        match acc {
            // SAFETY: the assertions above keep every selected row in bounds, and `Dot` is only
            // constructed behind FEAT_DotProd detection.
            Accumulator::Dot => unsafe {
                drive_dot(rows, words_per_row, |i| row_offsets[i], planes, q, out)
            },
            // SAFETY: bounds as above.
            Accumulator::Widen => unsafe {
                drive_widen(rows, words_per_row, |i| row_offsets[i], planes, q, out)
            },
        }
    }

    fn assert_shapes(words_per_row: usize, planes: [&[u64]; 4]) {
        assert!(words_per_row > 0);
        assert!(planes.iter().all(|plane| plane.len() == words_per_row));
    }

    /// Pairs rows so [`block2_dot`] shares each plane load, with a single-row cleanup call.
    ///
    /// # Safety
    ///
    /// FEAT_DotProd must be present, every plane must hold `words` words, and
    /// `row_index(i) * words + words <= rows.len()` for every `i < out.len()`.
    #[target_feature(enable = "dotprod")]
    unsafe fn drive_dot(
        rows: &[u64],
        words: usize,
        row_index: impl Fn(usize) -> usize,
        planes: [&[u64]; 4],
        q: &QueryPlanes,
        out: &mut [f32],
    ) {
        let base = rows.as_ptr();
        for pair in 0..out.len() / 2 {
            let rows2 = [
                base.add(row_index(pair * 2) * words),
                base.add(row_index(pair * 2 + 1) * words),
            ];
            let [(pos_a, sum_a), (pos_b, sum_b)] = block2_dot(rows2, words, planes);
            out[pair * 2] = finish_asym(pos_a, sum_a, q);
            out[pair * 2 + 1] = finish_asym(pos_b, sum_b, q);
        }
        if out.len() % 2 == 1 {
            let last = out.len() - 1;
            let (positives, weighted) = row_dot(base.add(row_index(last) * words), words, planes);
            out[last] = finish_asym(positives, weighted, q);
        }
    }

    /// Pairs rows so [`block2_widen`] shares each plane load, with a single-row cleanup call.
    ///
    /// # Safety
    ///
    /// Every plane must hold `words` words, and `row_index(i) * words + words <= rows.len()` for
    /// every `i < out.len()`.
    unsafe fn drive_widen(
        rows: &[u64],
        words: usize,
        row_index: impl Fn(usize) -> usize,
        planes: [&[u64]; 4],
        q: &QueryPlanes,
        out: &mut [f32],
    ) {
        let base = rows.as_ptr();
        for pair in 0..out.len() / 2 {
            let rows2 = [
                base.add(row_index(pair * 2) * words),
                base.add(row_index(pair * 2 + 1) * words),
            ];
            let [(pos_a, sum_a), (pos_b, sum_b)] = block2_widen(rows2, words, planes);
            out[pair * 2] = finish_asym(pos_a, sum_a, q);
            out[pair * 2 + 1] = finish_asym(pos_b, sum_b, q);
        }
        if out.len() % 2 == 1 {
            let last = out.len() - 1;
            let (positives, weighted) = row_widen(base.add(row_index(last) * words), words, planes);
            out[last] = finish_asym(positives, weighted, q);
        }
    }

    /// Two-row UDOT block; each plane chunk loads once and feeds both rows.
    ///
    /// # Safety
    ///
    /// FEAT_DotProd must be present, each row pointer must be readable for `words` words, and
    /// every plane must hold `words` words.
    #[target_feature(enable = "dotprod")]
    #[inline]
    unsafe fn block2_dot(
        rows2: [*const u64; 2],
        words: usize,
        planes: [&[u64]; 4],
    ) -> [(u32, u64); 2] {
        let row_bytes = rows2.map(|row| row.cast::<u8>());
        let plane_bytes = planes.map(|plane| plane.as_ptr().cast::<u8>());
        let ones = vdupq_n_u8(1);
        let mut acc_p = [vdupq_n_u32(0); 2];
        let mut acc = [[vdupq_n_u32(0); 4]; 2];
        for chunk in 0..words / 2 {
            let off = chunk * 16;
            let v0 = vld1q_u8(plane_bytes[0].add(off));
            let v1 = vld1q_u8(plane_bytes[1].add(off));
            let v2 = vld1q_u8(plane_bytes[2].add(off));
            let v3 = vld1q_u8(plane_bytes[3].add(off));
            for r in 0..2 {
                let s = vld1q_u8(row_bytes[r].add(off));
                acc_p[r] = udot_ones(acc_p[r], vcntq_u8(s), ones);
                acc[r][0] = udot_ones(acc[r][0], vcntq_u8(vandq_u8(s, v0)), ones);
                acc[r][1] = udot_ones(acc[r][1], vcntq_u8(vandq_u8(s, v1)), ones);
                acc[r][2] = udot_ones(acc[r][2], vcntq_u8(vandq_u8(s, v2)), ones);
                acc[r][3] = udot_ones(acc[r][3], vcntq_u8(vandq_u8(s, v3)), ones);
            }
        }
        let mut totals = [(0_u32, 0_u64); 2];
        for r in 0..2 {
            let mut positives = vaddvq_u32(acc_p[r]);
            let mut sums = [
                u64::from(vaddvq_u32(acc[r][0])),
                u64::from(vaddvq_u32(acc[r][1])),
                u64::from(vaddvq_u32(acc[r][2])),
                u64::from(vaddvq_u32(acc[r][3])),
            ];
            tail_word(rows2[r], words, planes, &mut positives, &mut sums);
            totals[r] = (positives, weighted_total(sums));
        }
        totals
    }

    /// Single-row UDOT kernel for a trailing row.
    ///
    /// # Safety
    ///
    /// FEAT_DotProd must be present, `row` must be readable for `words` words, and every plane
    /// must hold `words` words.
    #[target_feature(enable = "dotprod")]
    #[inline]
    unsafe fn row_dot(row: *const u64, words: usize, planes: [&[u64]; 4]) -> (u32, u64) {
        let row_bytes = row.cast::<u8>();
        let plane_bytes = planes.map(|plane| plane.as_ptr().cast::<u8>());
        let ones = vdupq_n_u8(1);
        let mut acc_p = vdupq_n_u32(0);
        let mut acc = [vdupq_n_u32(0); 4];
        for chunk in 0..words / 2 {
            let off = chunk * 16;
            let s = vld1q_u8(row_bytes.add(off));
            acc_p = udot_ones(acc_p, vcntq_u8(s), ones);
            for b in 0..4 {
                let masked = vandq_u8(s, vld1q_u8(plane_bytes[b].add(off)));
                acc[b] = udot_ones(acc[b], vcntq_u8(masked), ones);
            }
        }
        let mut positives = vaddvq_u32(acc_p);
        let mut sums = [
            u64::from(vaddvq_u32(acc[0])),
            u64::from(vaddvq_u32(acc[1])),
            u64::from(vaddvq_u32(acc[2])),
            u64::from(vaddvq_u32(acc[3])),
        ];
        tail_word(row, words, planes, &mut positives, &mut sums);
        (positives, weighted_total(sums))
    }

    /// Two-row UADALP block; each plane chunk loads once and feeds both rows.
    ///
    /// # Safety
    ///
    /// Each row pointer must be readable for `words` words and every plane must hold `words`
    /// words.
    #[inline]
    unsafe fn block2_widen(
        rows2: [*const u64; 2],
        words: usize,
        planes: [&[u64]; 4],
    ) -> [(u32, u64); 2] {
        let row_bytes = rows2.map(|row| row.cast::<u8>());
        let plane_bytes = planes.map(|plane| plane.as_ptr().cast::<u8>());
        let chunks = words / 2;
        let mut positives = [0_u32; 2];
        let mut sums = [[0_u64; 4]; 2];
        let mut chunk = 0;
        while chunk < chunks {
            let flush_end = (chunk + WIDEN_FLUSH_CHUNKS).min(chunks);
            let mut acc_p = [vdupq_n_u16(0); 2];
            let mut acc = [[vdupq_n_u16(0); 4]; 2];
            while chunk < flush_end {
                let off = chunk * 16;
                let v0 = vld1q_u8(plane_bytes[0].add(off));
                let v1 = vld1q_u8(plane_bytes[1].add(off));
                let v2 = vld1q_u8(plane_bytes[2].add(off));
                let v3 = vld1q_u8(plane_bytes[3].add(off));
                for r in 0..2 {
                    let s = vld1q_u8(row_bytes[r].add(off));
                    acc_p[r] = vpadalq_u8(acc_p[r], vcntq_u8(s));
                    acc[r][0] = vpadalq_u8(acc[r][0], vcntq_u8(vandq_u8(s, v0)));
                    acc[r][1] = vpadalq_u8(acc[r][1], vcntq_u8(vandq_u8(s, v1)));
                    acc[r][2] = vpadalq_u8(acc[r][2], vcntq_u8(vandq_u8(s, v2)));
                    acc[r][3] = vpadalq_u8(acc[r][3], vcntq_u8(vandq_u8(s, v3)));
                }
                chunk += 1;
            }
            for r in 0..2 {
                positives[r] += vaddlvq_u16(acc_p[r]);
                for b in 0..4 {
                    sums[r][b] += u64::from(vaddlvq_u16(acc[r][b]));
                }
            }
        }
        let mut totals = [(0_u32, 0_u64); 2];
        for r in 0..2 {
            tail_word(rows2[r], words, planes, &mut positives[r], &mut sums[r]);
            totals[r] = (positives[r], weighted_total(sums[r]));
        }
        totals
    }

    /// Single-row UADALP kernel for a trailing row.
    ///
    /// # Safety
    ///
    /// `row` must be readable for `words` words and every plane must hold `words` words.
    #[inline]
    unsafe fn row_widen(row: *const u64, words: usize, planes: [&[u64]; 4]) -> (u32, u64) {
        let row_bytes = row.cast::<u8>();
        let plane_bytes = planes.map(|plane| plane.as_ptr().cast::<u8>());
        let chunks = words / 2;
        let mut positives = 0_u32;
        let mut sums = [0_u64; 4];
        let mut chunk = 0;
        while chunk < chunks {
            let flush_end = (chunk + WIDEN_FLUSH_CHUNKS).min(chunks);
            let mut acc_p = vdupq_n_u16(0);
            let mut acc = [vdupq_n_u16(0); 4];
            while chunk < flush_end {
                let off = chunk * 16;
                let s = vld1q_u8(row_bytes.add(off));
                acc_p = vpadalq_u8(acc_p, vcntq_u8(s));
                for b in 0..4 {
                    let masked = vandq_u8(s, vld1q_u8(plane_bytes[b].add(off)));
                    acc[b] = vpadalq_u8(acc[b], vcntq_u8(masked));
                }
                chunk += 1;
            }
            positives += vaddlvq_u16(acc_p);
            for b in 0..4 {
                sums[b] += u64::from(vaddlvq_u16(acc[b]));
            }
        }
        tail_word(row, words, planes, &mut positives, &mut sums);
        (positives, weighted_total(sums))
    }

    /// Adds an odd trailing word's counts in scalar code.
    ///
    /// # Safety
    ///
    /// `row` must be readable for `words` words and every plane must hold `words` words.
    #[inline(always)]
    unsafe fn tail_word(
        row: *const u64,
        words: usize,
        planes: [&[u64]; 4],
        positives: &mut u32,
        sums: &mut [u64; 4],
    ) {
        if words % 2 == 1 {
            let last = words - 1;
            let signs = *row.add(last);
            *positives += signs.count_ones();
            for (sum, plane) in sums.iter_mut().zip(planes) {
                *sum += u64::from((signs & plane[last]).count_ones());
            }
        }
    }

    /// Weighted plane total; equals the scalar kernels' per-word shifted sums because shifting
    /// distributes over addition.
    #[inline(always)]
    fn weighted_total(sums: [u64; 4]) -> u64 {
        sums[0] + (sums[1] << 1) + (sums[2] << 2) + (sums[3] << 3)
    }

    /// Accumulates the byte sums of `counts` into the u32 lanes of `acc`: UDOT against all-ones.
    /// `vdotq_u32` is unstable, so the instruction is inline asm.
    ///
    /// # Safety
    ///
    /// FEAT_DotProd must be present.
    #[target_feature(enable = "dotprod")]
    #[inline]
    unsafe fn udot_ones(acc: uint32x4_t, counts: uint8x16_t, ones: uint8x16_t) -> uint32x4_t {
        let mut out = acc;
        // SAFETY: UDOT touches only its three register operands.
        core::arch::asm!(
            "udot {acc:v}.4s, {counts:v}.16b, {ones:v}.16b",
            acc = inout(vreg) out,
            counts = in(vreg) counts,
            ones = in(vreg) ones,
            options(pure, nomem, nostack),
        );
        out
    }
}

/// Scores one sign row against a full-precision query with a binary16 scale.
///
/// # Panics
///
/// Panics when the code and query dimensions disagree.
pub fn estimate_fp(x: &[u64], scale: u16, query: &[f32]) -> f32 {
    f16_to_f32(scale) * estimate_fp_unscaled(x, query)
}

/// Scores one sign row against a full-precision query with a binary32 scale.
///
/// # Panics
///
/// Panics when the code and query dimensions disagree.
pub fn estimate_fp_f32(x: &[u64], scale: f32, query: &[f32]) -> f32 {
    scale * estimate_fp_unscaled(x, query)
}

/// Score a sign row against a full-precision query without applying its scale.
///
/// # Panics
///
/// Panics when the query is empty or the code word count is invalid.
pub fn estimate_fp_unscaled(x: &[u64], query: &[f32]) -> f32 {
    assert!(!query.is_empty());
    assert_eq!(x.len(), packed_words(query.len()));
    debug_assert!(tail_is_zero(x, query.len()));
    let signed_dot: f32 = query
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            if x[i / 64] & (1_u64 << (i % 64)) != 0 {
                value
            } else {
                -value
            }
        })
        .sum();
    signed_dot
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;

    #[test]
    fn packing_round_trip() {
        let values: Vec<f32> = (0..128)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut packed = [0_u64; 2];
        pack(&values, &mut packed);
        assert_eq!(unpack(&packed, 128), values);
    }

    #[test]
    fn odd_dimension_round_trip_and_zero_tail() {
        for d in [65, 100, 300, 769] {
            let values: Vec<f32> = (0..d)
                .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
                .collect();
            let mut packed = vec![u64::MAX; packed_words(d)];
            pack(&values, &mut packed);
            assert!(tail_is_zero(&packed, d), "d={d}: {packed:x?}");
            assert_eq!(unpack(&packed, d), values);

            let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.13).sin()).collect();
            let prepared = prepare_query(&query, 4);
            assert!(
                prepared.planes.iter().all(|plane| tail_is_zero(plane, d)),
                "d={d}"
            );
            let scale = f32_to_f16(0.25);
            let direct = estimate_fp(&packed, scale, &query);
            let asymmetric = estimate_asym(&packed, scale, &prepared);
            let quantized_query: Vec<f32> = (0..d)
                .map(|i| {
                    let code: u16 = prepared
                        .planes
                        .iter()
                        .enumerate()
                        .map(|(bit, plane)| (((plane[i / 64] >> (i % 64)) & 1) as u16) << bit)
                        .sum();
                    prepared.lo + prepared.delta * f32::from(code)
                })
                .collect();
            let quantized_direct = estimate_fp(&packed, scale, &quantized_query);
            assert!((asymmetric - quantized_direct).abs() < 2e-4, "d={d}");
            let expected_error_squared = query
                .iter()
                .zip(&quantized_query)
                .map(|(&value, &reconstructed)| f64::from(value - reconstructed).powi(2))
                .sum::<f64>();
            assert_eq!(
                prepared.error_squared().to_bits(),
                expected_error_squared.to_bits(),
                "d={d}"
            );
            assert!(direct.is_finite());

            let mut query_signs = vec![0_u64; packed_words(d)];
            let query_scale = encode(&query, &mut query_signs);
            let decoded_dot = dot(
                &unpack(&packed, d)
                    .into_iter()
                    .map(|sign| sign * f16_to_f32(scale))
                    .collect::<Vec<_>>(),
                &unpack(&query_signs, d)
                    .into_iter()
                    .map(|sign| sign * f16_to_f32(query_scale))
                    .collect::<Vec<_>>(),
            );
            let sign_sum = d as i64 - 2 * i64::from(score_sym(&packed, &query_signs));
            let popcount = f16_to_f32(scale) * f16_to_f32(query_scale) * sign_sum as f32;
            assert!((decoded_dot - popcount).abs() < 1e-5, "d={d}");
        }
    }

    #[test]
    fn symmetric_hamming_score() {
        assert_eq!(score_sym(&[0, u64::MAX], &[u64::MAX, u64::MAX]), 64);
    }

    #[test]
    fn decode_then_dot_matches_symmetric_estimate() {
        let d = 768;
        let mut rng = ChaCha8Rng::seed_from_u64(0x0044_4543_4f44_4501);
        for _ in 0..100 {
            let data = random_unit(&mut rng, d);
            let query = random_unit(&mut rng, d);
            let mut data_bits = vec![0_u64; packed_words(d)];
            let mut query_bits = vec![0_u64; packed_words(d)];
            let data_scale = f16_to_f32(encode(&data, &mut data_bits));
            let query_scale = f16_to_f32(encode(&query, &mut query_bits));
            let decoded_data: Vec<f32> = unpack(&data_bits, d)
                .into_iter()
                .map(|sign| sign * data_scale)
                .collect();
            let decoded_query: Vec<f32> = unpack(&query_bits, d)
                .into_iter()
                .map(|sign| sign * query_scale)
                .collect();
            let decoded_dot = dot(&decoded_data, &decoded_query);
            let sign_sum = d as i64 - 2 * i64::from(score_sym(&data_bits, &query_bits));
            let popcount_estimate = data_scale * query_scale * sign_sum as f32;
            let absolute = (decoded_dot - popcount_estimate).abs();
            let relative = absolute / popcount_estimate.abs().max(f32::MIN_POSITIVE);
            assert!(
                absolute <= 1e-7 || relative <= 1e-5,
                "decoded={decoded_dot}, popcount={popcount_estimate}, absolute={absolute}, \
                 relative={relative}"
            );
        }
    }

    #[test]
    fn indexed_batch_matches_contiguous_and_scalar_for_sparse_rows() {
        let row_offsets = [17, 0, 8, 22, 3, 11, 11, 5];
        for d in [65, 128] {
            let words_per_row = packed_words(d);
            let rows = 23;
            let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.071).cos()).collect();
            for bq in [1, 4] {
                let prepared = prepare_query(&query, bq);
                let mut packed = vec![0_u64; rows * words_per_row];
                for row in 0..rows {
                    let values: Vec<f32> = (0..d)
                        .map(|i| {
                            if (i * 7 + row * 5) % 11 < 6 {
                                1.0
                            } else {
                                -1.0
                            }
                        })
                        .collect();
                    pack(
                        &values,
                        &mut packed[row * words_per_row..(row + 1) * words_per_row],
                    );
                }

                let expected: Vec<f32> = row_offsets
                    .iter()
                    .map(|&row| {
                        estimate_asym_unscaled(
                            &packed[row * words_per_row..(row + 1) * words_per_row],
                            &prepared,
                        )
                    })
                    .collect();
                let mut contiguous = vec![0.0; rows];
                estimate_asym_batch_unscaled(&packed, words_per_row, &prepared, &mut contiguous);
                let mut indexed = vec![0.0; row_offsets.len()];
                estimate_asym_batch_unscaled_indexed(
                    &packed,
                    words_per_row,
                    &row_offsets,
                    &prepared,
                    &mut indexed,
                );

                for ((&row, expected), actual) in row_offsets.iter().zip(expected).zip(indexed) {
                    assert_eq!(actual, expected, "d={d}, bq={bq}, row={row}");
                    assert_eq!(actual, contiguous[row], "d={d}, bq={bq}, row={row}");
                }
            }
        }
    }

    #[test]
    fn batch_scorers_match_scalar_bit_for_bit() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5349_474e_4245_4e43);
        for d in [64, 768, 1000, 1024, 1088] {
            let words = packed_words(d);
            let tail = d % 64;
            let mask = if tail == 0 {
                u64::MAX
            } else {
                (1_u64 << tail) - 1
            };
            let query = random_unit(&mut rng, d);
            let q = prepare_query(&query, 4);
            for rows in [1_usize, 2, 3, 200] {
                let mut packed = vec![0_u64; rows * words];
                for word in packed.iter_mut() {
                    *word = rng.next_u64();
                }
                for row in packed.chunks_exact_mut(words) {
                    *row.last_mut().unwrap() &= mask;
                }
                let expected: Vec<f32> = packed
                    .chunks_exact(words)
                    .map(|row| estimate_asym_unscaled(row, &q))
                    .collect();
                // Unordered offsets with duplicates and an odd count exercise pairing and the
                // trailing-row path.
                let row_offsets: Vec<usize> = (0..rows).rev().chain([0, rows - 1, 0]).collect();

                let mut contiguous = vec![0.0_f32; rows];
                estimate_asym_batch_unscaled(&packed, words, &q, &mut contiguous);
                for row in 0..rows {
                    assert_eq!(
                        contiguous[row].to_bits(),
                        expected[row].to_bits(),
                        "contiguous d={d}, rows={rows}, row={row}"
                    );
                }
                let mut indexed = vec![0.0_f32; row_offsets.len()];
                estimate_asym_batch_unscaled_indexed(
                    &packed,
                    words,
                    &row_offsets,
                    &q,
                    &mut indexed,
                );
                for (i, &offset) in row_offsets.iter().enumerate() {
                    assert_eq!(
                        indexed[i].to_bits(),
                        expected[offset].to_bits(),
                        "indexed d={d}, rows={rows}, i={i}"
                    );
                }

                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    let [p0, p1, p2, p3] = q.planes.as_slice() else {
                        unreachable!()
                    };
                    let planes = [p0.as_slice(), p1.as_slice(), p2.as_slice(), p3.as_slice()];
                    let mut accumulators = vec![neon::Accumulator::Widen];
                    if std::arch::is_aarch64_feature_detected!("dotprod") {
                        accumulators.push(neon::Accumulator::Dot);
                    }
                    for acc in accumulators {
                        let mut got = vec![0.0_f32; rows];
                        neon::batch(acc, &packed, words, planes, &q, &mut got);
                        for row in 0..rows {
                            assert_eq!(
                                got[row].to_bits(),
                                expected[row].to_bits(),
                                "neon batch d={d}, rows={rows}, row={row}"
                            );
                        }
                        let mut got_indexed = vec![0.0_f32; row_offsets.len()];
                        neon::indexed(
                            acc,
                            &packed,
                            words,
                            &row_offsets,
                            planes,
                            &q,
                            &mut got_indexed,
                        );
                        for (i, &offset) in row_offsets.iter().enumerate() {
                            assert_eq!(
                                got_indexed[i].to_bits(),
                                expected[offset].to_bits(),
                                "neon indexed d={d}, rows={rows}, i={i}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn formula_exactness_and_constant_query() {
        let signs: Vec<f32> = (0..128)
            .map(|i| if i % 5 < 2 { 1.0 } else { -1.0 })
            .collect();
        let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut bits = [0_u64; 2];
        pack(&signs, &mut bits);
        let prepared = prepare_query(&query, 4);
        let (positives, sum) = score_asym(&bits, &prepared);
        let combined = prepared.lo * (2_i64 * i64::from(positives) - 128) as f32
            + prepared.delta * (2_i128 * i128::from(sum) - i128::from(prepared.sum_codes)) as f32;
        let direct: f32 = signs
            .iter()
            .enumerate()
            .map(|(i, &sign)| {
                let code: u16 = prepared
                    .planes
                    .iter()
                    .enumerate()
                    .map(|(bit, plane)| (((plane[i / 64] >> (i % 64)) & 1) as u16) << bit)
                    .sum();
                sign * (prepared.lo + prepared.delta * f32::from(code))
            })
            .sum();
        assert!((combined - direct).abs() < 2e-4, "{combined} != {direct}");

        let constant = vec![0.375; 128];
        let prepared = prepare_query(&constant, 4);
        assert_eq!(prepared.delta, 0.0);
        assert_eq!(prepared.sum_codes, 0);
        assert_eq!(prepared.error_squared(), 0.0);
        assert!(prepared.planes.iter().flatten().all(|&word| word == 0));
        let scale = f32_to_f16(0.25);
        let estimated = estimate_asym(&bits, scale, &prepared);
        let direct: f32 = signs.iter().map(|&sign| sign * 0.375 * 0.25).sum();
        assert!((estimated - direct).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_is_canonical() {
        let mut bits = [u64::MAX; 2];
        let scale = encode(&[0.0; 128], &mut bits);
        assert_eq!(scale, 0);
        assert_eq!(bits, [0, 0]);
        assert_eq!(estimate_fp(&bits, scale, &[0.25; 128]), 0.0);
        assert_eq!(
            estimate_asym(&bits, scale, &prepare_query(&[0.25; 128], 4)),
            0.0
        );
    }

    #[test]
    fn sign_scale_matches_reference_reconstruction() {
        let d = 768;
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let mut encoded_error = 0.0_f64;
        let mut reference_error = 0.0_f64;
        let mut energy = 0.0_f64;
        for _ in 0..1_000 {
            let vector = random_unit(&mut rng, d);
            let mut bits = vec![0_u64; packed_words(d)];
            let scale = f16_to_f32(encode(&vector, &mut bits));
            let reference_scale = f16_to_f32(f32_to_f16(
                vector.iter().map(|value| value.abs()).sum::<f32>() / d as f32,
            ));
            for &value in &vector {
                let sign = if value > 0.0 { 1.0 } else { -1.0 };
                encoded_error += f64::from(value - scale * sign).powi(2);
                reference_error += f64::from(value - reference_scale * sign).powi(2);
                energy += f64::from(value).powi(2);
            }
        }
        let encoded_rho = (encoded_error / energy).sqrt();
        let reference_rho = (reference_error / energy).sqrt();
        assert!((encoded_rho - reference_rho).abs() < 1e-6);
    }

    #[test]
    fn statistical_estimators() {
        let d = 768;
        let mut rng = ChaCha8Rng::seed_from_u64(0x51_47_4e);
        let queries: Vec<Vec<f32>> = (0..100).map(|_| random_unit(&mut rng, d)).collect();
        let q4: Vec<QueryPlanes> = queries
            .iter()
            .map(|query| prepare_query(query, 4))
            .collect();
        let q1: Vec<QueryPlanes> = queries
            .iter()
            .map(|query| prepare_query(query, 1))
            .collect();
        let mut truth_energy = 0.0_f64;
        let mut fp_error = 0.0_f64;
        let mut q4_error = 0.0_f64;
        let mut q1_error = 0.0_f64;
        for _ in 0..1_000 {
            let vector = random_unit(&mut rng, d);
            let mut bits = vec![0_u64; packed_words(d)];
            let scale = encode(&vector, &mut bits);
            for ((query, q4), q1) in queries.iter().zip(&q4).zip(&q1) {
                let truth = dot(query, &vector);
                truth_energy += f64::from(truth).powi(2);
                fp_error += f64::from(estimate_fp(&bits, scale, query) - truth).powi(2);
                q4_error += f64::from(estimate_asym(&bits, scale, q4) - truth).powi(2);
                q1_error += f64::from(estimate_asym(&bits, scale, q1) - truth).powi(2);
            }
        }
        let fp_rho = (fp_error / truth_energy).sqrt();
        let q4_rho = (q4_error / truth_energy).sqrt();
        let q1_rho = (q1_error / truth_energy).sqrt();
        assert!(
            (fp_rho - 0.6025).abs() <= 0.010,
            "fp={fp_rho}, q4={q4_rho}, q1={q1_rho}"
        );
        assert!(q4_rho <= fp_rho * 1.05, "fp={fp_rho}, q4={q4_rho}");
        assert!(q1_rho > q4_rho, "q1={q1_rho}, q4={q4_rho}");
    }

    fn random_unit(rng: &mut ChaCha8Rng, d: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(d);
        while values.len() < d {
            let u1 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            let u2 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * u2;
            values.push((radius * angle.cos()) as f32);
            if values.len() < d {
                values.push((radius * angle.sin()) as f32);
            }
        }
        let norm = values
            .iter()
            .map(|&value| value * value)
            .sum::<f32>()
            .sqrt();
        for value in &mut values {
            *value /= norm;
        }
        values
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }
}
