use std::sync::OnceLock;

use quant::model::f16::f16_to_f32;

pub(crate) type Kernel = fn(&[f32], &[u8]) -> f32;
pub(crate) type PackedKernel = fn(&[u16], &[u8]) -> f32;

pub(crate) fn packed_dot_kernel() -> Option<PackedKernel> {
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if std::arch::is_aarch64_feature_detected!("fhm") {
        return Some(neon_dot_fhm);
    }
    None
}

#[cfg(target_arch = "aarch64")]
fn neon_dot_fhm(query: &[u16], bytes: &[u8]) -> f32 {
    assert_eq!(bytes.len(), query.len() * 2);
    unsafe { neon_dot_fhm_inner(query, bytes) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fhm")]
unsafe fn neon_dot_fhm_inner(query: &[u16], bytes: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let end = query.len() / 16 * 16;
    let mut sums = [vdupq_n_f32(0.0); 4];
    for offset in (0..end).step_by(16) {
        for (lane, sum) in sums.iter_mut().enumerate() {
            let offset = offset + lane * 4;
            let q = vld1_u16(query.as_ptr().add(offset));
            let v = vld1_u16(bytes.as_ptr().add(offset * 2).cast());
            std::arch::asm!(
                "fmlal {sum:v}.4s, {q:v}.4h, {v:v}.4h",
                sum = inout(vreg) *sum,
                q = in(vreg) q,
                v = in(vreg) v,
                options(pure, nomem, nostack),
            );
        }
    }
    let mut sum = vaddvq_f32(vaddq_f32(
        vaddq_f32(sums[0], sums[1]),
        vaddq_f32(sums[2], sums[3]),
    ));
    for (&q, v) in query[end..].iter().zip(bytes[end * 2..].chunks_exact(2)) {
        sum = f16_to_f32(q).mul_add(f16_to_f32(u16::from_le_bytes([v[0], v[1]])), sum);
    }
    sum
}

pub(crate) fn kernels() -> (Kernel, Kernel) {
    static KERNELS: OnceLock<(Kernel, Kernel)> = OnceLock::new();
    *KERNELS.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        {
            return (neon::<false>, neon::<true>);
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("f16c")
            && std::is_x86_feature_detected!("fma")
        {
            return (avx::<false>, avx::<true>);
        }
        #[allow(unreachable_code)]
        (scalar::<false>, scalar::<true>)
    })
}

/// Dot product against packed little-endian binary16 coordinates.
pub fn dot_f16_bytes(query: &[f32], bytes: &[u8]) -> f32 {
    kernels().0(query, bytes)
}

/// Squared L2 distance against packed little-endian binary16 coordinates.
pub fn l2_squared_f16_bytes(query: &[f32], bytes: &[u8]) -> f32 {
    kernels().1(query, bytes)
}

fn scalar<const L2: bool>(query: &[f32], bytes: &[u8]) -> f32 {
    assert_eq!(bytes.len(), query.len() * 2);
    query
        .iter()
        .zip(bytes.chunks_exact(2))
        .fold(0.0, |sum, (&q, v)| {
            let v = f16_to_f32(u16::from_le_bytes([v[0], v[1]]));
            if L2 {
                (q - v).mul_add(q - v, sum)
            } else {
                q.mul_add(v, sum)
            }
        })
}

#[cfg(target_arch = "aarch64")]
fn neon<const L2: bool>(query: &[f32], bytes: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    assert_eq!(bytes.len(), query.len() * 2);
    let end = query.len() / 16 * 16;
    unsafe {
        let mut sums = [vdupq_n_f32(0.0); 4];
        for offset in (0..end).step_by(16) {
            for (lane, sum) in sums.iter_mut().enumerate() {
                let offset = offset + lane * 4;
                let packed = vld1_u16(bytes.as_ptr().add(offset * 2).cast());
                let value: float32x4_t;
                std::arch::asm!(
                    "fcvtl {value:v}.4s, {packed:v}.4h",
                    packed = in(vreg) packed,
                    value = out(vreg) value,
                    options(pure, nomem, nostack),
                );
                let q = vld1q_f32(query.as_ptr().add(offset));
                *sum = if L2 {
                    let delta = vsubq_f32(q, value);
                    vfmaq_f32(*sum, delta, delta)
                } else {
                    vfmaq_f32(*sum, q, value)
                };
            }
        }
        vaddvq_f32(vaddq_f32(
            vaddq_f32(sums[0], sums[1]),
            vaddq_f32(sums[2], sums[3]),
        )) + scalar::<L2>(&query[end..], &bytes[end * 2..])
    }
}

#[cfg(target_arch = "x86_64")]
fn avx<const L2: bool>(query: &[f32], bytes: &[u8]) -> f32 {
    assert_eq!(bytes.len(), query.len() * 2);
    unsafe { avx_inner::<L2>(query, bytes) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn avx_inner<const L2: bool>(query: &[f32], bytes: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let end = query.len() / 32 * 32;
    let mut sums = [_mm256_setzero_ps(); 4];
    for offset in (0..end).step_by(32) {
        for (lane, sum) in sums.iter_mut().enumerate() {
            let offset = offset + lane * 8;
            let value = _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add(offset * 2).cast()));
            let q = _mm256_loadu_ps(query.as_ptr().add(offset));
            *sum = if L2 {
                let delta = _mm256_sub_ps(q, value);
                _mm256_fmadd_ps(delta, delta, *sum)
            } else {
                _mm256_fmadd_ps(q, value, *sum)
            };
        }
    }
    let sum = _mm256_add_ps(
        _mm256_add_ps(sums[0], sums[1]),
        _mm256_add_ps(sums[2], sums[3]),
    );
    let mut lanes = [0.0; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    lanes.iter().sum::<f32>() + scalar::<L2>(&query[end..], &bytes[end * 2..])
}

#[cfg(test)]
mod tests {
    use quant::model::f16::f32_to_f16;

    use super::*;

    #[test]
    fn kernels_match_widened_reference_for_unaligned_rows_and_tails() {
        for dim in [0, 1, 3, 15, 16, 17, 31, 32, 33, 65, 1024, 1536] {
            let query: Vec<_> = (0..dim).map(|i| ((i as f32 + 1.0) * 0.17).sin()).collect();
            let mut bytes = vec![0];
            for i in 0..dim {
                bytes.extend_from_slice(&f32_to_f16((i as f32 * 0.31).cos()).to_le_bytes());
            }
            for (actual, expected) in [
                (
                    dot_f16_bytes(&query, &bytes[1..]),
                    scalar::<false>(&query, &bytes[1..]),
                ),
                (
                    l2_squared_f16_bytes(&query, &bytes[1..]),
                    scalar::<true>(&query, &bytes[1..]),
                ),
            ] {
                assert!(
                    (actual - expected).abs() <= 2e-5 * expected.abs().max(1.0),
                    "dim={dim}: {actual} != {expected}"
                );
            }
            if let Some(kernel) = packed_dot_kernel() {
                let packed: Vec<_> = query.iter().copied().map(f32_to_f16).collect();
                let rounded: Vec<_> = packed.iter().copied().map(f16_to_f32).collect();
                let expected = scalar::<false>(&rounded, &bytes[1..]);
                assert!(
                    (kernel(&packed, &bytes[1..]) - expected).abs()
                        <= 2e-5 * expected.abs().max(1.0)
                );
            }
        }
    }

    #[test]
    fn subnormal_and_maximum_half_inputs_accumulate_in_f32() {
        let bits = [1u16, 0x8001, 0x7bff, 0xfbff, 0, 0x8000];
        let packed: Vec<_> = bits.into_iter().cycle().take(33).collect();
        let query: Vec<_> = packed.iter().copied().map(f16_to_f32).collect();
        let bytes: Vec<_> = packed.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        let expected = scalar::<false>(&query, &bytes);
        assert!(expected.is_finite());
        assert!((dot_f16_bytes(&query, &bytes) - expected).abs() <= expected * 2e-6);
        assert_eq!(l2_squared_f16_bytes(&query, &bytes), 0.0);
        if let Some(kernel) = packed_dot_kernel() {
            assert!((kernel(&packed, &bytes) - expected).abs() <= expected * 2e-6);
        }
    }
}
