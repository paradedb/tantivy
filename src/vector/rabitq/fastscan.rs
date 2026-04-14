//! FastScan LUT-based distance computation.
//!
//! Replaces float dot products with precomputed lookup table lookups.
//! For each group of 4 dimensions, we precompute all 2^4=16 possible
//! dot products. Then each 4-bit nibble of the packed binary code
//! becomes a direct index into the LUT — no multiplications needed.
//!
//! Batch mode: process 32 vectors at once with column-major binary codes
//! and SIMD shuffle-based accumulation (NEON vqtbl1q / AVX2 vpshufb).

pub const BATCH_SIZE: usize = 32;

const KPOS: [usize; 16] = [3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3];

#[inline]
fn lowbit(x: usize) -> usize {
    x & x.wrapping_neg()
}

pub struct QueryLut {
    lut_f32: Vec<f32>,
    lut_u8: Vec<u8>,
    delta: f32,
    sum_vl: f32,
    num_codebooks: usize,
}

impl QueryLut {
    pub fn new(rotated_query: &[f32]) -> Self {
        let dim = rotated_query.len();
        assert!(dim % 4 == 0);
        let num_codebooks = dim / 4;
        let mut lut_f32 = vec![0.0f32; num_codebooks * 16];

        for i in 0..num_codebooks {
            let q_offset = i * 4;
            let lut_offset = i * 16;
            lut_f32[lut_offset] = 0.0;
            for j in 1..16 {
                let prev_idx = j - lowbit(j);
                lut_f32[lut_offset + j] =
                    lut_f32[lut_offset + prev_idx] + rotated_query[q_offset + KPOS[j]];
            }
        }

        let vl = lut_f32
            .iter()
            .copied()
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);
        let vr = lut_f32
            .iter()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);
        let delta = (vr - vl) / 255.0;

        let lut_u8 = if delta > 0.0 {
            lut_f32
                .iter()
                .map(|&v| ((v - vl) / delta).round().clamp(0.0, 255.0) as u8)
                .collect()
        } else {
            vec![0u8; lut_f32.len()]
        };

        let sum_vl = vl * (num_codebooks as f32);

        Self {
            lut_f32,
            lut_u8,
            delta,
            sum_vl,
            num_codebooks,
        }
    }

    pub fn binary_dot(&self, packed_binary: &[u8]) -> f32 {
        let mut result = 0.0f32;
        for byte_idx in 0..packed_binary.len() {
            let byte = packed_binary[byte_idx];
            let codebook_base = byte_idx * 2;
            if codebook_base >= self.num_codebooks {
                break;
            }
            let hi = ((byte >> 4) & 0x0F) as usize;
            result += self.lut_f32[codebook_base * 16 + hi];
            if codebook_base + 1 < self.num_codebooks {
                let lo = (byte & 0x0F) as usize;
                result += self.lut_f32[(codebook_base + 1) * 16 + lo];
            }
        }
        result
    }

    pub fn delta(&self) -> f32 {
        self.delta
    }

    pub fn sum_vl(&self) -> f32 {
        self.sum_vl
    }

    pub fn lut_u8(&self) -> &[u8] {
        &self.lut_u8
    }
}

/// Transpose 32 vectors' binary codes from row-major to column-major.
///
/// Input: `codes[vec_idx * dim_bytes + col]` — row-major, each vector contiguous.
/// Output: for each column (byte), 32 vectors' bytes are contiguous.
///
/// The output is NOT permuted by KPERM0 — this is a simple transpose
/// suitable for scalar batch accumulation. SIMD paths may need additional
/// permutation.
pub fn pack_batch_simple(
    codes: &[&[u8]],
    dim_bytes: usize,
    output: &mut [u8],
) {
    assert!(codes.len() <= BATCH_SIZE);
    assert!(output.len() >= dim_bytes * BATCH_SIZE);

    for col in 0..dim_bytes {
        for (vec_idx, code) in codes.iter().enumerate() {
            output[col * BATCH_SIZE + vec_idx] = code[col];
        }
        for vec_idx in codes.len()..BATCH_SIZE {
            output[col * BATCH_SIZE + vec_idx] = 0;
        }
    }
}

/// Compute binary dot products for a batch of 32 vectors using the
/// column-major layout produced by `pack_batch_simple`.
///
/// Returns 32 u32 accumulators (one per vector).
pub fn accumulate_batch(
    packed_codes: &[u8],
    lut_u8: &[u8],
    dim_bytes: usize,
    results: &mut [u32; BATCH_SIZE],
) {
    results.fill(0);

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                return accumulate_batch_neon(packed_codes, lut_u8, dim_bytes, results);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return accumulate_batch_avx2(packed_codes, lut_u8, dim_bytes, results);
            }
        }
    }

    accumulate_batch_scalar(packed_codes, lut_u8, dim_bytes, results);
}

fn accumulate_batch_scalar(
    packed_codes: &[u8],
    lut_u8: &[u8],
    dim_bytes: usize,
    results: &mut [u32; BATCH_SIZE],
) {
    for col in 0..dim_bytes {
        let codebook_hi = col * 2;
        let codebook_lo = col * 2 + 1;
        let lut_hi_base = codebook_hi * 16;
        let lut_lo_base = codebook_lo * 16;

        let col_offset = col * BATCH_SIZE;
        for vec_idx in 0..BATCH_SIZE {
            let byte = packed_codes[col_offset + vec_idx];
            let hi = ((byte >> 4) & 0x0F) as usize;
            let lo = (byte & 0x0F) as usize;
            results[vec_idx] += lut_u8[lut_hi_base + hi] as u32;
            if lut_lo_base + 15 < lut_u8.len() {
                results[vec_idx] += lut_u8[lut_lo_base + lo] as u32;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_batch_neon(
    packed_codes: &[u8],
    lut_u8: &[u8],
    dim_bytes: usize,
    results: &mut [u32; BATCH_SIZE],
) {
    use std::arch::aarch64::*;

    let mask_lo = vdupq_n_u8(0x0F);

    // 4 NEON registers × 16 bytes = 32 vectors × u16 accumulators
    let mut accu0 = vdupq_n_u16(0); // vecs 0-7
    let mut accu1 = vdupq_n_u16(0); // vecs 8-15
    let mut accu2 = vdupq_n_u16(0); // vecs 16-23
    let mut accu3 = vdupq_n_u16(0); // vecs 24-31

    for col in 0..dim_bytes {
        let codebook_hi = col * 2;
        let codebook_lo = col * 2 + 1;

        let lut_hi = vld1q_u8(lut_u8.as_ptr().add(codebook_hi * 16));

        let col_offset = col * BATCH_SIZE;

        // Load 32 bytes (one byte per vector for this column)
        let codes0 = vld1q_u8(packed_codes.as_ptr().add(col_offset));
        let codes1 = vld1q_u8(packed_codes.as_ptr().add(col_offset + 16));

        // Extract high nibbles
        let hi0 = vshrq_n_u8(codes0, 4);
        let hi1 = vshrq_n_u8(codes1, 4);

        // Lookup high nibble values
        let val_hi0 = vqtbl1q_u8(lut_hi, hi0);
        let val_hi1 = vqtbl1q_u8(lut_hi, hi1);

        // Accumulate (widen u8 → u16)
        accu0 = vaddw_u8(accu0, vget_low_u8(val_hi0));
        accu1 = vaddw_u8(accu1, vget_high_u8(val_hi0));
        accu2 = vaddw_u8(accu2, vget_low_u8(val_hi1));
        accu3 = vaddw_u8(accu3, vget_high_u8(val_hi1));

        // Low nibble (second codebook)
        if codebook_lo * 16 + 15 < lut_u8.len() {
            let lut_lo = vld1q_u8(lut_u8.as_ptr().add(codebook_lo * 16));
            let lo0 = vandq_u8(codes0, mask_lo);
            let lo1 = vandq_u8(codes1, mask_lo);
            let val_lo0 = vqtbl1q_u8(lut_lo, lo0);
            let val_lo1 = vqtbl1q_u8(lut_lo, lo1);
            accu0 = vaddw_u8(accu0, vget_low_u8(val_lo0));
            accu1 = vaddw_u8(accu1, vget_high_u8(val_lo0));
            accu2 = vaddw_u8(accu2, vget_low_u8(val_lo1));
            accu3 = vaddw_u8(accu3, vget_high_u8(val_lo1));
        }
    }

    // Store results: extract u16 lanes via vst1q then widen to u32
    let mut out = [0u32; BATCH_SIZE];
    let mut tmp = [0u16; 8];
    vst1q_u16(tmp.as_mut_ptr(), accu0);
    for i in 0..8 { out[i] = tmp[i] as u32; }
    vst1q_u16(tmp.as_mut_ptr(), accu1);
    for i in 0..8 { out[8 + i] = tmp[i] as u32; }
    vst1q_u16(tmp.as_mut_ptr(), accu2);
    for i in 0..8 { out[16 + i] = tmp[i] as u32; }
    vst1q_u16(tmp.as_mut_ptr(), accu3);
    for i in 0..8 { out[24 + i] = tmp[i] as u32; }

    results.copy_from_slice(&out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_batch_avx2(
    packed_codes: &[u8],
    lut_u8: &[u8],
    dim_bytes: usize,
    results: &mut [u32; BATCH_SIZE],
) {
    use std::arch::x86_64::*;

    let mask_lo = _mm256_set1_epi8(0x0F);

    // 2 AVX2 registers × 32 bytes = 32 vectors × u16 accumulators
    // Each 256-bit register holds 16 u16 values
    let mut accu0 = _mm256_setzero_si256(); // vecs 0-15
    let mut accu1 = _mm256_setzero_si256(); // vecs 16-31

    for col in 0..dim_bytes {
        let codebook_hi = col * 2;
        let codebook_lo = col * 2 + 1;

        // AVX2 vpshufb operates on two 128-bit lanes independently,
        // so we need to broadcast the 16-byte LUT to both lanes
        let lut_hi_128 = _mm_loadu_si128(lut_u8.as_ptr().add(codebook_hi * 16) as *const _);
        let lut_hi = _mm256_broadcastsi128_si256(lut_hi_128);

        let col_offset = col * BATCH_SIZE;

        // Load 32 bytes (one per vector for this column)
        let codes = _mm256_loadu_si256(packed_codes.as_ptr().add(col_offset) as *const _);

        // Extract high nibbles: shift right 4, mask
        let hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), mask_lo);

        // Shuffle lookup for high nibble
        let val_hi = _mm256_shuffle_epi8(lut_hi, hi);

        // Accumulate: widen u8 → u16 by adding to zero-extended values
        // Split into low/high 128-bit halves, zero-extend u8→u16, accumulate
        let val_hi_lo = _mm256_unpacklo_epi8(val_hi, _mm256_setzero_si256());
        let val_hi_hi = _mm256_unpackhi_epi8(val_hi, _mm256_setzero_si256());
        accu0 = _mm256_add_epi16(accu0, val_hi_lo);
        accu1 = _mm256_add_epi16(accu1, val_hi_hi);

        // Low nibble (second codebook)
        if codebook_lo * 16 + 15 < lut_u8.len() {
            let lut_lo_128 = _mm_loadu_si128(lut_u8.as_ptr().add(codebook_lo * 16) as *const _);
            let lut_lo = _mm256_broadcastsi128_si256(lut_lo_128);
            let lo = _mm256_and_si256(codes, mask_lo);
            let val_lo = _mm256_shuffle_epi8(lut_lo, lo);
            let val_lo_lo = _mm256_unpacklo_epi8(val_lo, _mm256_setzero_si256());
            let val_lo_hi = _mm256_unpackhi_epi8(val_lo, _mm256_setzero_si256());
            accu0 = _mm256_add_epi16(accu0, val_lo_lo);
            accu1 = _mm256_add_epi16(accu1, val_lo_hi);
        }
    }

    // Store results: extract u16 lanes to u32
    // accu0 holds: lane0=[v0..v7 as u16], lane1=[v16..v23 as u16]
    // accu1 holds: lane0=[v8..v15 as u16], lane1=[v24..v31 as u16]
    let mut out = [0u32; BATCH_SIZE];
    let mut tmp0 = [0u16; 16];
    let mut tmp1 = [0u16; 16];
    _mm256_storeu_si256(tmp0.as_mut_ptr() as *mut _, accu0);
    _mm256_storeu_si256(tmp1.as_mut_ptr() as *mut _, accu1);
    // lane0 of accu0: vecs 0-7
    for i in 0..8 { out[i] = tmp0[i] as u32; }
    // lane1 of accu0: vecs 16-23
    for i in 0..8 { out[16 + i] = tmp0[8 + i] as u32; }
    // lane0 of accu1: vecs 8-15
    for i in 0..8 { out[8 + i] = tmp1[i] as u32; }
    // lane1 of accu1: vecs 24-31
    for i in 0..8 { out[24 + i] = tmp1[8 + i] as u32; }

    results.copy_from_slice(&out);
}

/// Convert batch accumulator results back to float binary_dot values.
///
/// `accu[i]` is the quantized LUT accumulation for vector i.
/// Returns `binary_dot[i] ≈ delta * accu[i] + sum_vl`.
pub fn denormalize_batch(
    accu: &[u32; BATCH_SIZE],
    delta: f32,
    sum_vl: f32,
    binary_dots: &mut [f32; BATCH_SIZE],
) {
    for i in 0..BATCH_SIZE {
        binary_dots[i] = delta * (accu[i] as f32) + sum_vl;
    }
}

/// Compute batch distances from binary dot products.
///
/// For each of 32 vectors: `distance[i] = f_add[i] + g_add + f_rescale[i] * (binary_dot[i] + k1x_sum_q)`
/// Also computes lower bounds: `lower_bound[i] = distance[i] - |f_error[i]|`
pub fn compute_batch_distances(
    binary_dots: &[f32; BATCH_SIZE],
    f_add: &[f32],
    f_rescale: &[f32],
    f_error: &[f32],
    g_add: f32,
    k1x_sum_q: f32,
    distances: &mut [f32; BATCH_SIZE],
    lower_bounds: &mut [f32; BATCH_SIZE],
) {
    for i in 0..BATCH_SIZE {
        let binary_term = binary_dots[i] + k1x_sum_q;
        distances[i] = f_add[i] + g_add + f_rescale[i] * binary_term;
        lower_bounds[i] = distances[i] - f_error[i].abs();
    }
}

/// Per-batch data stored in the cluster file for FastScan.
///
/// Each batch holds 32 vectors' binary codes in column-major layout
/// plus per-vector scalars.
pub fn batch_data_size(dim_bytes: usize) -> usize {
    let binary_bytes = dim_bytes * BATCH_SIZE;
    let scalar_bytes = BATCH_SIZE * 3 * 4; // f_add, f_rescale, f_error as f32
    binary_bytes + scalar_bytes
}

/// Compute dot product of query with packed extended codes directly,
/// without unpacking to u16 first. Works on the C++-compatible packed format.
///
/// For ex_bits=2: each 4 bytes encode 16 values (2 bits each).
/// For ex_bits=6: each 12 bytes encode 16 values (6 bits each).
pub fn ip_packed_ex_f32(
    query: &[f32],
    packed_ex_code: &[u8],
    padded_dim: usize,
    ex_bits: usize,
) -> f32 {
    match ex_bits {
        2 => ip_packed_ex2_f32(query, packed_ex_code, padded_dim),
        6 => ip_packed_ex6_f32_scalar(query, packed_ex_code, padded_dim),
        _ => {
            let ex_code = super::simd::unpack_ex_code_from_packed(packed_ex_code, padded_dim, ex_bits);
            ex_code.iter().zip(query.iter()).map(|(&c, &q)| (c as f32) * q).sum()
        }
    }
}

fn ip_packed_ex2_f32(query: &[f32], packed_ex_code: &[u8], padded_dim: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                return ip_packed_ex2_f32_neon(query, packed_ex_code, padded_dim);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return ip_packed_ex2_f32_avx2(query, packed_ex_code, padded_dim);
            }
        }
    }

    ip_packed_ex2_f32_scalar(query, packed_ex_code, padded_dim)
}

#[cfg(target_arch = "aarch64")]
unsafe fn ip_packed_ex2_f32_neon(
    query: &[f32],
    packed_ex_code: &[u8],
    padded_dim: usize,
) -> f32 {
    use std::arch::aarch64::*;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let mut query_ptr = query.as_ptr();
    let mut code_ptr = packed_ex_code.as_ptr();

    // Process 16 elements per iteration (4 bytes of 2-bit codes)
    for _ in 0..(padded_dim / 16) {
        let compact = std::ptr::read_unaligned(code_ptr as *const u32);

        // Extract 4 groups of 4 codes (shift by 0,2,4,6 bits, mask with 0x03)
        // Group 0: codes 0,1,2,3
        let g0 = [
            (compact & 0x3) as f32,
            ((compact >> 8) & 0x3) as f32,
            ((compact >> 16) & 0x3) as f32,
            ((compact >> 24) & 0x3) as f32,
        ];
        // Group 1: codes 4,5,6,7
        let g1 = [
            ((compact >> 2) & 0x3) as f32,
            ((compact >> 10) & 0x3) as f32,
            ((compact >> 18) & 0x3) as f32,
            ((compact >> 26) & 0x3) as f32,
        ];
        // Group 2: codes 8,9,10,11
        let g2 = [
            ((compact >> 4) & 0x3) as f32,
            ((compact >> 12) & 0x3) as f32,
            ((compact >> 20) & 0x3) as f32,
            ((compact >> 28) & 0x3) as f32,
        ];
        // Group 3: codes 12,13,14,15
        let g3 = [
            ((compact >> 6) & 0x3) as f32,
            ((compact >> 14) & 0x3) as f32,
            ((compact >> 22) & 0x3) as f32,
            ((compact >> 30) & 0x3) as f32,
        ];

        let cv0 = vld1q_f32(g0.as_ptr());
        let cv1 = vld1q_f32(g1.as_ptr());
        let cv2 = vld1q_f32(g2.as_ptr());
        let cv3 = vld1q_f32(g3.as_ptr());

        let q0 = vld1q_f32(query_ptr);
        let q1 = vld1q_f32(query_ptr.add(4));
        let q2 = vld1q_f32(query_ptr.add(8));
        let q3 = vld1q_f32(query_ptr.add(12));

        sum0 = vfmaq_f32(sum0, cv0, q0);
        sum1 = vfmaq_f32(sum1, cv1, q1);
        sum2 = vfmaq_f32(sum2, cv2, q2);
        sum3 = vfmaq_f32(sum3, cv3, q3);

        query_ptr = query_ptr.add(16);
        code_ptr = code_ptr.add(4);
    }

    let total = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    vaddvq_f32(total)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn ip_packed_ex2_f32_avx2(
    query: &[f32],
    packed_ex_code: &[u8],
    padded_dim: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mask = _mm_set1_epi8(0b00000011);

    let mut query_ptr = query.as_ptr();
    let mut code_ptr = packed_ex_code.as_ptr();

    for _ in 0..(padded_dim / 16) {
        let compact = std::ptr::read_unaligned(code_ptr as *const i32);

        let code_i32 = _mm_set_epi32(compact >> 6, compact >> 4, compact >> 2, compact);
        let code_masked = _mm_and_si128(code_i32, mask);

        // Lower 8 codes → f32, FMA with query
        let code_f32_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(code_masked));
        let query_lo = _mm256_loadu_ps(query_ptr);
        sum = _mm256_fmadd_ps(code_f32_lo, query_lo, sum);

        // Upper 8 codes → f32, FMA with query
        let code_masked_hi = _mm_unpackhi_epi64(code_masked, code_masked);
        let code_f32_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(code_masked_hi));
        let query_hi = _mm256_loadu_ps(query_ptr.add(8));
        sum = _mm256_fmadd_ps(code_f32_hi, query_hi, sum);

        query_ptr = query_ptr.add(16);
        code_ptr = code_ptr.add(4);
    }

    let sum_hi = _mm256_extractf128_ps(sum, 1);
    let sum_lo = _mm256_castps256_ps128(sum);
    let sum_128 = _mm_add_ps(sum_lo, sum_hi);
    let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
    let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 0x55));
    _mm_cvtss_f32(sum_32)
}

fn ip_packed_ex2_f32_scalar(query: &[f32], packed_ex_code: &[u8], padded_dim: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut code_idx = 0;

    for chunk in 0..(padded_dim / 16) {
        let base = chunk * 4;
        let compact = u32::from_le_bytes([
            packed_ex_code[base],
            packed_ex_code[base + 1],
            packed_ex_code[base + 2],
            packed_ex_code[base + 3],
        ]);

        for i in 0..4 {
            let byte_offset = i * 8;
            let c0 = ((compact >> byte_offset) & 0x3) as f32;
            let c1 = ((compact >> (byte_offset + 2)) & 0x3) as f32;
            let c2 = ((compact >> (byte_offset + 4)) & 0x3) as f32;
            let c3 = ((compact >> (byte_offset + 6)) & 0x3) as f32;

            sum += c0 * query[code_idx];
            sum += c1 * query[code_idx + 4];
            sum += c2 * query[code_idx + 8];
            sum += c3 * query[code_idx + 12];
            code_idx += 1;
        }
        code_idx += 12;
    }
    sum
}

fn ip_packed_ex6_f32_scalar(query: &[f32], packed_ex_code: &[u8], padded_dim: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut code_idx = 0;

    for chunk in 0..(padded_dim / 16) {
        let base = chunk * 12;

        let compact4 = u64::from_le_bytes([
            packed_ex_code[base],
            packed_ex_code[base + 1],
            packed_ex_code[base + 2],
            packed_ex_code[base + 3],
            packed_ex_code[base + 4],
            packed_ex_code[base + 5],
            packed_ex_code[base + 6],
            packed_ex_code[base + 7],
        ]);

        let compact2 = u32::from_le_bytes([
            packed_ex_code[base + 8],
            packed_ex_code[base + 9],
            packed_ex_code[base + 10],
            packed_ex_code[base + 11],
        ]);

        for i in 0..4 {
            let lo_offset = i * 16;
            let hi_offset = i * 8;

            let lo0 = ((compact4 >> lo_offset) & 0xF) as u16;
            let lo1 = ((compact4 >> (lo_offset + 4)) & 0xF) as u16;
            let lo2 = ((compact4 >> (lo_offset + 8)) & 0xF) as u16;
            let lo3 = ((compact4 >> (lo_offset + 12)) & 0xF) as u16;

            let hi0 = ((compact2 >> hi_offset) & 0x3) as u16;
            let hi1 = ((compact2 >> (hi_offset + 2)) & 0x3) as u16;
            let hi2 = ((compact2 >> (hi_offset + 4)) & 0x3) as u16;
            let hi3 = ((compact2 >> (hi_offset + 6)) & 0x3) as u16;

            let c0 = (lo0 | (hi0 << 4)) as f32;
            let c1 = (lo1 | (hi1 << 4)) as f32;
            let c2 = (lo2 | (hi2 << 4)) as f32;
            let c3 = (lo3 | (hi3 << 4)) as f32;

            sum += c0 * query[code_idx];
            sum += c1 * query[code_idx + 4];
            sum += c2 * query[code_idx + 8];
            sum += c3 * query[code_idx + 12];
            code_idx += 1;
        }
        code_idx += 12;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_matches_scalar() {
        let dim = 64;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) / dim as f32 - 0.5).collect();
        let lut = QueryLut::new(&query);

        let binary_code: Vec<u8> = (0..dim).map(|i| (i % 2) as u8).collect();
        let scalar_dot: f32 = binary_code
            .iter()
            .zip(query.iter())
            .map(|(&b, &q)| (b as f32) * q)
            .sum();

        let packed_len = dim / 8;
        let mut packed = vec![0u8; packed_len];
        for i in 0..dim {
            if binary_code[i] == 1 {
                packed[i / 8] |= 1 << (7 - (i % 8));
            }
        }

        let lut_dot = lut.binary_dot(&packed);
        assert!(
            (scalar_dot - lut_dot).abs() < 1e-5,
            "scalar={scalar_dot} lut={lut_dot}"
        );
    }

    #[test]
    fn test_lut_all_zeros() {
        let dim = 32;
        let query: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let lut = QueryLut::new(&query);
        let packed = vec![0u8; dim / 8];
        assert_eq!(lut.binary_dot(&packed), 0.0);
    }

    #[test]
    fn test_lut_all_ones() {
        let dim = 32;
        let query: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let lut = QueryLut::new(&query);
        let packed = vec![0xFFu8; dim / 8];
        let expected: f32 = query.iter().sum();
        let result = lut.binary_dot(&packed);
        assert!(
            (expected - result).abs() < 1e-4,
            "expected={expected} got={result}"
        );
    }

    #[test]
    fn test_batch_accumulate_matches_scalar() {
        let dim = 64;
        let dim_bytes = dim / 8;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) / dim as f32 - 0.5).collect();
        let lut = QueryLut::new(&query);

        // Create 32 random-ish packed binary codes
        let codes: Vec<Vec<u8>> = (0..BATCH_SIZE)
            .map(|v| {
                (0..dim_bytes)
                    .map(|b| ((v * 7 + b * 13) & 0xFF) as u8)
                    .collect()
            })
            .collect();

        // Per-vector scalar LUT dot products
        let scalar_dots: Vec<f32> = codes.iter().map(|c| lut.binary_dot(c)).collect();

        // Batch: transpose + accumulate
        let code_refs: Vec<&[u8]> = codes.iter().map(|c| c.as_slice()).collect();
        let mut transposed = vec![0u8; dim_bytes * BATCH_SIZE];
        pack_batch_simple(&code_refs, dim_bytes, &mut transposed);

        let mut accu = [0u32; BATCH_SIZE];
        accumulate_batch(&transposed, lut.lut_u8(), dim_bytes, &mut accu);

        let mut batch_dots = [0.0f32; BATCH_SIZE];
        denormalize_batch(&accu, lut.delta(), lut.sum_vl(), &mut batch_dots);

        // Compare: batch results should approximate scalar results
        for i in 0..BATCH_SIZE {
            let err = (scalar_dots[i] - batch_dots[i]).abs();
            assert!(
                err < 0.5,
                "vec {i}: scalar={} batch={} err={err}",
                scalar_dots[i],
                batch_dots[i]
            );
        }
    }
}
