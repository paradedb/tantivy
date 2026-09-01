//! IEEE-754 binary16 conversion with round-to-nearest, ties-to-even.

#[inline(always)]
pub fn f16_to_f32(value: u16) -> f32 {
    // Branch-free binary16 expansion. `renorm_shift` normalizes subnormals;
    // `inf_nan_mask` restores the all-ones exponent; `zero_mask` clears the
    // synthetic exponent that normalization would otherwise give ±0.
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
    fn monotonic_on_positives() {
        let mut previous = 0.0;
        for bits in 0_u16..0x7c00 {
            let value = f16_to_f32(bits);
            assert!(value >= previous);
            previous = value;
        }
    }
}
