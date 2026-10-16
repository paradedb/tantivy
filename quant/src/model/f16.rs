//! IEEE-754 binary16 conversion with round-to-nearest, ties-to-even.

#[inline(always)]
/// Converts binary16 bits to binary32.
pub fn f16_to_f32(value: u16) -> f32 {
    let word = (value as u32) << 16;
    let sign = word & 0x8000_0000;
    let nonsign = word & 0x7fff_ffff;
    let renorm_shift = nonsign.leading_zeros().saturating_sub(5);
    let inf_nan_mask = ((nonsign.wrapping_add(0x0400_0000) as i32 >> 8) as u32) & 0x7f80_0000;
    let zero_mask = ((nonsign.wrapping_sub(1) as i32) >> 31) as u32;
    let magnitude = (((nonsign << renorm_shift) >> 3)
        .wrapping_add((0x70_u32.wrapping_sub(renorm_shift)) << 23)
        | inf_nan_mask)
        & !zero_mask;
    f32::from_bits(sign | magnitude)
}

/// Converts binary32 to binary16 bits.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007f_ffff;

    if exponent == 0xff {
        let payload = (fraction >> 13) as u16;
        return sign
            | 0x7c00
            | if payload == 0 && fraction != 0 {
                1
            } else {
                payload
            };
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = fraction | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let mut rounded = mantissa >> shift;
        let remainder = mantissa & ((1_u32 << shift) - 1);
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }

    let mut result = sign | ((half_exponent as u16) << 10) | (fraction >> 13) as u16;
    let remainder = fraction & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && result & 1 != 0) {
        result = result.wrapping_add(1);
    }
    result
}

/// Decodes a run of little-endian binary16 values into binary32.
///
/// binary16 to binary32 widening is exact, and the hardware converters (NEON FCVTL, F16C behind
/// the x86 dispatch) agree with the portable bit manipulation on every finite and infinite
/// input, so swapping them never changes results; NaNs stay NaNs either way and sidecar decoding
/// validates finiteness.
///
/// # Panics
///
/// Panics when the byte length is odd or the output length disagrees.
pub fn decode_f16s_to_f32s(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(bytes.len() % 2, 0);
    assert_eq!(out.len() * 2, bytes.len());
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    neon::decode(bytes, out);
    #[cfg(target_arch = "x86_64")]
    x86::decode(bytes, out);
    #[cfg(not(any(
        all(target_arch = "aarch64", target_endian = "little"),
        target_arch = "x86_64"
    )))]
    decode_scalar(bytes, out);
}

/// Portable body of [`decode_f16s_to_f32s`].
#[inline(always)]
fn decode_scalar(bytes: &[u8], out: &mut [f32]) {
    for (out, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *out = f16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
    }
}

/// NEON decode: FCVTL widens eight values per iteration. The conversion intrinsics are
/// unstable, so the instruction pair is inline asm. The vector load reads the u16 lanes
/// straight from little-endian bytes, so the path exists only on little-endian targets;
/// big-endian aarch64 falls back to the portable conversion.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon {
    use std::arch::aarch64::{float32x4_t, uint16x8_t, vld1q_u16, vst1q_f32};

    pub(super) fn decode(bytes: &[u8], out: &mut [f32]) {
        let blocks = out.len() / 8;
        // SAFETY: the caller asserts `bytes` holds `2 * out.len()` bytes, so each block of
        // eight u16 loads and eight f32 stores is in bounds; the load and store allow
        // unaligned addresses.
        unsafe {
            for block in 0..blocks {
                let half = vld1q_u16(bytes.as_ptr().add(block * 16).cast());
                vst1q_f32(out.as_mut_ptr().add(block * 8), fcvtl(half));
                vst1q_f32(out.as_mut_ptr().add(block * 8 + 4), fcvtl2(half));
            }
        }
        super::decode_scalar(&bytes[blocks * 16..], &mut out[blocks * 8..]);
    }

    /// Widens the low four binary16 lanes.
    #[inline(always)]
    fn fcvtl(half: uint16x8_t) -> float32x4_t {
        let wide;
        // SAFETY: FCVTL touches only its register operands.
        unsafe {
            core::arch::asm!(
                "fcvtl {wide:v}.4s, {half:v}.4h",
                wide = out(vreg) wide,
                half = in(vreg) half,
                options(pure, nomem, nostack),
            );
        }
        wide
    }

    /// Widens the high four binary16 lanes.
    #[inline(always)]
    fn fcvtl2(half: uint16x8_t) -> float32x4_t {
        let wide;
        // SAFETY: FCVTL2 touches only its register operands.
        unsafe {
            core::arch::asm!(
                "fcvtl2 {wide:v}.4s, {half:v}.8h",
                wide = out(vreg) wide,
                half = in(vreg) half,
                options(pure, nomem, nostack),
            );
        }
        wide
    }
}

/// x86 decode: VCVTPH2PS widens eight values per iteration behind the AVX2 tier; other tiers
/// keep the portable bit manipulation.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use crate::cpu::{tier, Tier};

    pub(super) fn decode(bytes: &[u8], out: &mut [f32]) {
        match tier() {
            // SAFETY: the tier is only reported when the CPU has avx2, popcnt and f16c.
            Tier::Avx2 => unsafe { decode_f16c(bytes, out) },
            _ => super::decode_scalar(bytes, out),
        }
    }

    /// # Safety
    ///
    /// The CPU must have AVX and F16C.
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn decode_f16c(bytes: &[u8], out: &mut [f32]) {
        use std::arch::x86_64::{_mm256_cvtph_ps, _mm256_storeu_ps, _mm_loadu_si128};
        let blocks = out.len() / 8;
        for block in 0..blocks {
            // SAFETY: the public caller asserts `bytes` holds `2 * out.len()` bytes, so each
            // block of eight loads and stores is in bounds; both allow unaligned addresses.
            let half = _mm_loadu_si128(bytes.as_ptr().add(block * 16).cast());
            _mm256_storeu_ps(out.as_mut_ptr().add(block * 8), _mm256_cvtph_ps(half));
        }
        super::decode_scalar(&bytes[blocks * 16..], &mut out[blocks * 8..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_against_half() {
        for bits in 0_u16..=u16::MAX {
            let ours = f16_to_f32(bits);
            let oracle = half::f16::from_bits(bits).to_f32();
            if oracle.is_nan() {
                assert!(ours.is_nan(), "{bits:#06x}");
            } else {
                assert_eq!(ours.to_bits(), oracle.to_bits(), "{bits:#06x}");
            }
            let ours_back = f32_to_f16(oracle);
            let oracle_back = half::f16::from_f32(oracle).to_bits();
            if oracle.is_nan() {
                assert_eq!(ours_back & 0x7c00, 0x7c00);
                assert_ne!(ours_back & 0x03ff, 0);
            } else {
                assert_eq!(ours_back, oracle_back, "{bits:#06x}");
            }
        }
    }

    #[test]
    fn batch_decode_matches_scalar_for_every_pattern() {
        let bytes: Vec<u8> = (0_u32..65_536)
            .flat_map(|bits| (bits as u16).to_le_bytes())
            .collect();
        let mut decoded = vec![0.0_f32; 65_536];
        decode_f16s_to_f32s(&bytes, &mut decoded);
        for bits in 0_u16..=u16::MAX {
            let want = f16_to_f32(bits);
            let got = decoded[bits as usize];
            if want.is_nan() {
                assert!(got.is_nan(), "{bits:#06x}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "{bits:#06x}");
            }
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("f16c")
        {
            let mut forced = vec![0.0_f32; 65_536];
            // SAFETY: avx and f16c were just detected.
            unsafe { x86::decode_f16c(&bytes, &mut forced) };
            for bits in 0_u16..=u16::MAX {
                let want = f16_to_f32(bits);
                let got = forced[bits as usize];
                if want.is_nan() {
                    assert!(got.is_nan(), "f16c {bits:#06x}");
                } else {
                    assert_eq!(got.to_bits(), want.to_bits(), "f16c {bits:#06x}");
                }
            }
        }
        // Lengths around the eight-lane block size exercise the scalar remainder.
        for len in [0_usize, 1, 7, 8, 9, 15, 16, 17] {
            let mut tail = vec![0.0_f32; len];
            decode_f16s_to_f32s(&bytes[..len * 2], &mut tail);
            for (i, value) in tail.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    f16_to_f32(i as u16).to_bits(),
                    "len={len}, i={i}"
                );
            }
        }
    }

    #[test]
    fn monotonic_on_positives() {
        let mut previous = 0.0;
        for bits in 0_u16..0x7c00 {
            let value = f16_to_f32(bits);
            assert!(value >= previous);
            previous = value;
        }
    }
}
