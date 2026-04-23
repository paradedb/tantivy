//! MSB-first bit-packed storage for `bit_width`-bit codebook indices.
//!
//! For `bit_width` ∈ {1..=8} we pack `count` indices into `ceil(count *
//! bit_width / 8)` bytes. Bits are MSB-first within each byte and indices
//! cross byte boundaries when needed.
//!
//! Adapted from https://github.com/abdelstark/turboquant (MIT). The
//! abdelstark crate exposes a `BitPackedVector` with serde state; we
//! only need free functions over slices since the layout is owned by
//! the on-disk record format in `record.rs`.

/// Bytes needed to hold `count` indices at `bit_width` bits each.
#[inline]
pub fn packed_byte_size(count: usize, bit_width: u8) -> usize {
    (count * bit_width as usize).div_ceil(8)
}

/// Pack `indices` (each ≤ 2^bit_width - 1) into `out`.
///
/// Panics if `out.len() < packed_byte_size(indices.len(), bit_width)` or
/// if `bit_width` is outside 1..=8. Input indices that exceed the bit
/// width are masked silently to make the hot path branchless; callers
/// should validate upstream when correctness matters.
pub fn pack_into(indices: &[u8], bit_width: u8, out: &mut [u8]) {
    assert!((1..=8).contains(&bit_width));
    let need = packed_byte_size(indices.len(), bit_width);
    assert!(
        out.len() >= need,
        "bitpack: out too small ({} < {})",
        out.len(),
        need
    );

    // Zero only the region we'll write into.
    for b in &mut out[..need] {
        *b = 0;
    }

    let bw = bit_width as u32;
    let mask = ((1u16 << bw) - 1) as u16;

    for (i, &idx) in indices.iter().enumerate() {
        let val = (idx as u16) & mask;
        let bit_offset = i * bit_width as usize;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;

        // Shift so the bit_width bits land at the top of a u16, then OR
        // into two consecutive bytes (the second one only if it exists).
        let shifted = val << (16 - bw - bit_idx as u32);
        let hi = (shifted >> 8) as u8;
        let lo = (shifted & 0xFF) as u8;

        out[byte_idx] |= hi;
        if byte_idx + 1 < need {
            out[byte_idx + 1] |= lo;
        }
    }
}

/// Allocating wrapper around [`pack_into`].
pub fn pack(indices: &[u8], bit_width: u8) -> Vec<u8> {
    let mut out = vec![0u8; packed_byte_size(indices.len(), bit_width)];
    pack_into(indices, bit_width, &mut out);
    out
}

/// Unpack `count` indices of `bit_width` bits each from `data` into `out`.
///
/// Panics if `data` is too short or `bit_width` is outside 1..=8.
pub fn unpack_into(data: &[u8], count: usize, bit_width: u8, out: &mut [u8]) {
    assert!((1..=8).contains(&bit_width));
    let need = packed_byte_size(count, bit_width);
    assert!(data.len() >= need, "bitpack: data too short");
    assert!(out.len() >= count, "bitpack: out too small");

    let bw = bit_width as u32;
    let mask = ((1u16 << bw) - 1) as u16;

    for i in 0..count {
        let bit_offset = i * bit_width as usize;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;

        let hi = data[byte_idx] as u16;
        let lo = if byte_idx + 1 < data.len() {
            data[byte_idx + 1] as u16
        } else {
            0
        };
        let combined = (hi << 8) | lo;
        let shift = 16 - bw - bit_idx as u32;
        out[i] = ((combined >> shift) & mask) as u8;
    }
}

/// Allocating wrapper around [`unpack_into`].
pub fn unpack(data: &[u8], count: usize, bit_width: u8) -> Vec<u8> {
    let mut out = vec![0u8; count];
    unpack_into(data, count, bit_width, &mut out);
    out
}

/// Pack `signs` (one bit each) MSB-first into bytes. `len(signs)` bits
/// fit in `ceil(len/8)` bytes.
pub fn pack_signs_into(signs: &[bool], out: &mut [u8]) {
    let need = signs.len().div_ceil(8);
    assert!(out.len() >= need);
    for b in &mut out[..need] {
        *b = 0;
    }
    for (i, &s) in signs.iter().enumerate() {
        if s {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8);
            out[byte_idx] |= 1 << bit_idx;
        }
    }
}

pub fn pack_signs(signs: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; signs.len().div_ceil(8)];
    pack_signs_into(signs, &mut out);
    out
}

/// Unpack `count` signs from `data` into `out` (true = 1, false = 0).
pub fn unpack_signs_into(data: &[u8], count: usize, out: &mut [bool]) {
    assert!(data.len() >= count.div_ceil(8));
    assert!(out.len() >= count);
    for i in 0..count {
        let byte_idx = i / 8;
        let bit_idx = 7 - (i % 8);
        out[i] = (data[byte_idx] >> bit_idx) & 1 == 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(indices: &[u8], bit_width: u8) {
        let packed = pack(indices, bit_width);
        assert_eq!(packed.len(), packed_byte_size(indices.len(), bit_width));
        let unpacked = unpack(&packed, indices.len(), bit_width);
        assert_eq!(
            unpacked, indices,
            "roundtrip failed at bit_width={bit_width}"
        );
    }

    #[test]
    fn pack_b2_roundtrip() {
        let indices: Vec<u8> = (0..200).map(|i| (i as u8) & 0b11).collect();
        roundtrip(&indices, 2);
    }

    #[test]
    fn pack_b3_roundtrip() {
        let indices: Vec<u8> = (0..768).map(|i| (i as u8) & 0b111).collect();
        roundtrip(&indices, 3);
    }

    #[test]
    fn pack_b4_roundtrip() {
        let indices: Vec<u8> = (0..768).map(|i| (i as u8) & 0b1111).collect();
        roundtrip(&indices, 4);
    }

    #[test]
    fn pack_b1_alternating() {
        let indices = vec![1u8, 0, 1, 0, 1, 0, 1, 0];
        let packed = pack(&indices, 1);
        assert_eq!(packed, vec![0b10101010]);
        assert_eq!(unpack(&packed, 8, 1), indices);
    }

    #[test]
    fn pack_b8_passthrough() {
        let indices: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let packed = pack(&indices, 8);
        assert_eq!(packed, indices);
        assert_eq!(unpack(&packed, 256, 8), indices);
    }

    #[test]
    fn signs_roundtrip() {
        let signs: Vec<bool> = (0..768).map(|i| i % 3 == 0).collect();
        let packed = pack_signs(&signs);
        assert_eq!(packed.len(), 96);
        let mut out = vec![false; signs.len()];
        unpack_signs_into(&packed, signs.len(), &mut out);
        assert_eq!(out, signs);
    }
}
