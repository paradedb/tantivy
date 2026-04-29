/// Compute the dot product between two vectors.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked that AVX2 is available on this CPU.
            return unsafe { x86::dot_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: We just checked that SSE2 is available on this CPU.
            return unsafe { x86::dot_sse2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    let result = unsafe { neon::dot_neon(a, b) };

    #[cfg(not(target_arch = "aarch64"))]
    let result = dot_scalar(a, b);

    result
}

/// Compute the squared L2 norm of a vector.
#[inline]
pub fn l2_norm_sqr(v: &[f32]) -> f32 {
    dot(v, v)
}

/// Compute the squared Euclidean distance between two vectors.
#[inline]
pub fn l2_distance_sqr(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked that AVX2 is available on this CPU.
            return unsafe { x86::l2_distance_sqr_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: We just checked that SSE2 is available on this CPU.
            return unsafe { x86::l2_distance_sqr_sse2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    let result = unsafe { neon::l2_distance_sqr_neon(a, b) };

    #[cfg(not(target_arch = "aarch64"))]
    let result = l2_distance_sqr_scalar(a, b);

    result
}

/// Normalize a vector in-place. Returns the original norm.
#[inline]
pub fn normalize(v: &mut [f32]) -> f32 {
    let norm = l2_norm_sqr(v).sqrt();
    if norm <= f32::EPSILON {
        return 0.0;
    }
    for value in v.iter_mut() {
        *value /= norm;
    }
    norm
}

/// Compute `a - b` element-wise.
#[inline]
pub fn subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f32; len];

    if len == 0 {
        return out;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked that AVX2 is available on this CPU.
            unsafe {
                x86::subtract_avx2(a, b, &mut out);
            }
            return out;
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: We just checked that SSE2 is available on this CPU.
            unsafe {
                x86::subtract_sse2(a, b, &mut out);
            }
            return out;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::subtract_neon(a, b, &mut out);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        subtract_scalar_into(a, b, &mut out);
    }

    out
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn l2_distance_sqr_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn subtract_scalar_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((dst, x), y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *dst = *x - *y;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        let chunks = len / 8;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        while i < chunks * 8 {
            let va = _mm256_loadu_ps(a_ptr.add(i));
            let vb = _mm256_loadu_ps(b_ptr.add(i));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
            i += 8;
        }

        let mut sum = 0.0f32;
        if chunks > 0 {
            let mut buf = [0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), acc);
            sum = buf.iter().copied().sum();
        }

        while i < len {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn dot_sse2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut acc = _mm_setzero_ps();
        let mut i = 0usize;
        let chunks = len / 4;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        while i < chunks * 4 {
            let va = _mm_loadu_ps(a_ptr.add(i));
            let vb = _mm_loadu_ps(b_ptr.add(i));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
            i += 4;
        }

        let mut sum = 0.0f32;
        if chunks > 0 {
            let mut buf = [0f32; 4];
            _mm_storeu_ps(buf.as_mut_ptr(), acc);
            sum = buf.iter().copied().sum();
        }

        while i < len {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn l2_distance_sqr_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        let chunks = len / 8;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        while i < chunks * 8 {
            let va = _mm256_loadu_ps(a_ptr.add(i));
            let vb = _mm256_loadu_ps(b_ptr.add(i));
            let diff = _mm256_sub_ps(va, vb);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
            i += 8;
        }

        let mut sum = 0.0f32;
        if chunks > 0 {
            let mut buf = [0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), acc);
            sum = buf.iter().copied().sum();
        }

        while i < len {
            let diff = *a_ptr.add(i) - *b_ptr.add(i);
            sum += diff * diff;
            i += 1;
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn l2_distance_sqr_sse2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut acc = _mm_setzero_ps();
        let mut i = 0usize;
        let chunks = len / 4;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        while i < chunks * 4 {
            let va = _mm_loadu_ps(a_ptr.add(i));
            let vb = _mm_loadu_ps(b_ptr.add(i));
            let diff = _mm_sub_ps(va, vb);
            acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
            i += 4;
        }

        let mut sum = 0.0f32;
        if chunks > 0 {
            let mut buf = [0f32; 4];
            _mm_storeu_ps(buf.as_mut_ptr(), acc);
            sum = buf.iter().copied().sum();
        }

        while i < len {
            let diff = *a_ptr.add(i) - *b_ptr.add(i);
            sum += diff * diff;
            i += 1;
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn subtract_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = out.len();
        let mut i = 0usize;
        let chunks = len / 8;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i < chunks * 8 {
            let va = _mm256_loadu_ps(a_ptr.add(i));
            let vb = _mm256_loadu_ps(b_ptr.add(i));
            let diff = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(out_ptr.add(i), diff);
            i += 8;
        }

        while i < len {
            *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn subtract_sse2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = out.len();
        let mut i = 0usize;
        let chunks = len / 4;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i < chunks * 4 {
            let va = _mm_loadu_ps(a_ptr.add(i));
            let vb = _mm_loadu_ps(b_ptr.add(i));
            let diff = _mm_sub_ps(va, vb);
            _mm_storeu_ps(out_ptr.add(i), diff);
            i += 4;
        }

        while i < len {
            *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        let chunks = len / 4;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        while i < chunks * 4 {
            let va = vld1q_f32(a_ptr.add(i));
            let vb = vld1q_f32(b_ptr.add(i));
            acc = vaddq_f32(acc, vmulq_f32(va, vb));
            i += 4;
        }

        let mut sum = if chunks > 0 { vaddvq_f32(acc) } else { 0.0f32 };
        while i < len {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn l2_distance_sqr_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Four independent accumulators to hide FMA's ~3-cycle latency
        // on Apple-silicon NEON. With one accumulator the loop is
        // latency-bound (~1 fmadd / 3 cycles); with four it's
        // throughput-bound (~4 fmadds / 3 cycles), ~3-4× faster on
        // dense d=768 distances. This shows up directly in the cluster
        // plugin's per-doc nearest-centroid pass which is bottlenecked
        // here.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let chunks16 = len / 16;
        let mut i = 0usize;
        while i < chunks16 * 16 {
            let va0 = vld1q_f32(a_ptr.add(i));
            let vb0 = vld1q_f32(b_ptr.add(i));
            let d0 = vsubq_f32(va0, vb0);
            acc0 = vfmaq_f32(acc0, d0, d0);

            let va1 = vld1q_f32(a_ptr.add(i + 4));
            let vb1 = vld1q_f32(b_ptr.add(i + 4));
            let d1 = vsubq_f32(va1, vb1);
            acc1 = vfmaq_f32(acc1, d1, d1);

            let va2 = vld1q_f32(a_ptr.add(i + 8));
            let vb2 = vld1q_f32(b_ptr.add(i + 8));
            let d2 = vsubq_f32(va2, vb2);
            acc2 = vfmaq_f32(acc2, d2, d2);

            let va3 = vld1q_f32(a_ptr.add(i + 12));
            let vb3 = vld1q_f32(b_ptr.add(i + 12));
            let d3 = vsubq_f32(va3, vb3);
            acc3 = vfmaq_f32(acc3, d3, d3);

            i += 16;
        }

        // 4-wide tail.
        while i + 4 <= len {
            let va = vld1q_f32(a_ptr.add(i));
            let vb = vld1q_f32(b_ptr.add(i));
            let d = vsubq_f32(va, vb);
            acc0 = vfmaq_f32(acc0, d, d);
            i += 4;
        }

        let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut sum = vaddvq_f32(acc);

        // Scalar tail.
        while i < len {
            let diff = *a_ptr.add(i) - *b_ptr.add(i);
            sum += diff * diff;
            i += 1;
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn subtract_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = out.len();
        let mut i = 0usize;
        let chunks = len / 4;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let out_ptr = out.as_mut_ptr();

        while i < chunks * 4 {
            let va = vld1q_f32(a_ptr.add(i));
            let vb = vld1q_f32(b_ptr.add(i));
            let diff = vsubq_f32(va, vb);
            vst1q_f32(out_ptr.add(i), diff);
            i += 4;
        }

        while i < len {
            *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
            i += 1;
        }
    }
}
