//! Packed exact-density grid encoding and LUT scoring kernels.

use quant_model::f16::{f16_to_f32, f32_to_f16};

pub fn packed_len(d: usize, bits: u8) -> usize {
    assert!(d > 0);
    assert!(matches!(bits, 2..=4));
    d.checked_mul(bits as usize)
        .expect("packed grid length overflow")
        .div_ceil(64)
        * 8
}

fn tail_is_zero(packed: &[u8], d: usize, bits: u8) -> bool {
    if d == 0 || !matches!(bits, 2..=4) || packed.len() != packed_len(d, bits) {
        return false;
    }
    let used_bits = d * bits as usize;
    let full_bytes = used_bits / 8;
    let tail_bits = used_bits % 8;
    if tail_bits == 0 {
        packed[full_bytes..].iter().all(|&byte| byte == 0)
    } else {
        let used_mask = (1_u8 << tail_bits) - 1;
        packed[full_bytes] & !used_mask == 0
            && packed[full_bytes + 1..].iter().all(|&byte| byte == 0)
    }
}

pub fn pack(codes: &[u8], bits: u8, out: &mut [u8]) {
    assert!(!codes.is_empty());
    assert_eq!(out.len(), packed_len(codes.len(), bits));
    out.fill(0);
    let mask = (1_u8 << bits) - 1;
    if bits == 3 {
        for (i, &code) in codes.iter().enumerate() {
            assert!(code <= mask);
            let bit_offset = i * 3;
            let byte = bit_offset / 8;
            let shift = bit_offset % 8;
            out[byte] |= code << shift;
            if shift > 5 {
                out[byte + 1] |= code >> (8 - shift);
            }
        }
        debug_assert!(tail_is_zero(out, codes.len(), bits));
        return;
    }
    let per_byte = 8 / bits as usize;
    for (i, &code) in codes.iter().enumerate() {
        assert!(code <= mask);
        out[i / per_byte] |= code << (bits as usize * (i % per_byte));
    }
    debug_assert!(tail_is_zero(out, codes.len(), bits));
}

pub fn unpack(packed: &[u8], d: usize, bits: u8) -> Vec<u8> {
    assert!(d > 0);
    assert_eq!(packed.len(), packed_len(d, bits));
    debug_assert!(tail_is_zero(packed, d, bits));
    let mask = (1_u8 << bits) - 1;
    if bits == 3 {
        return (0..d).map(|i| code_at_3(packed, i)).collect();
    }
    let per_byte = 8 / bits as usize;
    (0..d)
        .map(|i| (packed[i / per_byte] >> (bits as usize * (i % per_byte))) & mask)
        .collect()
}

/// Encode a vector and return its RMS scale after f16 rounding.
pub fn encode(y: &[f32], grid: &[f32], bits: u8, out: &mut [u8]) -> u16 {
    let mut code_scratch = vec![0_u8; y.len()];
    encode_with_scratch(y, grid, bits, out, &mut code_scratch)
}

/// Encode a vector using caller-owned one-byte-per-coordinate scratch.
///
/// The scratch form is byte-identical to [`encode`] and lets a cluster batch
/// reuse one allocation across every row.
pub fn encode_with_scratch(
    y: &[f32],
    grid: &[f32],
    bits: u8,
    out: &mut [u8],
    code_scratch: &mut [u8],
) -> u16 {
    assert!(!y.is_empty());
    validate_grid(grid, bits);
    assert_eq!(out.len(), packed_len(y.len(), bits));
    assert_eq!(code_scratch.len(), y.len());
    let norm_squared = y.iter().map(|&value| value * value).sum::<f32>();
    if norm_squared == 0.0 {
        out.fill(0);
        return 0;
    }

    let scale = norm_squared.sqrt() / (y.len() as f32).sqrt();
    let mut boundaries = [0.0_f32; 15];
    for (boundary, pair) in boundaries.iter_mut().zip(grid.windows(2)) {
        *boundary = (pair[0] + pair[1]) * 0.5;
    }
    let boundaries = &boundaries[..grid.len() - 1];
    for (code, &value) in code_scratch.iter_mut().zip(y) {
        *code = boundaries.partition_point(|&boundary| value / scale > boundary) as u8;
    }
    pack(code_scratch, bits, out);
    f32_to_f16(scale)
}

pub fn decode(codes: &[u8], grid: &[f32], d: usize, bits: u8, scale: u16) -> Vec<f32> {
    let mut decoded = vec![0.0_f32; d];
    decode_into(codes, grid, bits, scale, &mut decoded);
    decoded
}

/// Decode a packed row into caller-owned output storage.
pub fn decode_into(codes: &[u8], grid: &[f32], bits: u8, scale: u16, out: &mut [f32]) {
    assert!(!out.is_empty());
    assert_eq!(codes.len(), packed_len(out.len(), bits));
    validate_grid(grid, bits);
    debug_assert!(tail_is_zero(codes, out.len(), bits));
    let scale = f16_to_f32(scale);
    for (i, value) in out.iter_mut().enumerate() {
        *value = scale * grid[code_at(codes, i, bits) as usize];
    }
}

pub fn build_lut(u: &[f32], grid: &[f32], bits: u8) -> Vec<f32> {
    assert!(!u.is_empty());
    validate_grid(grid, bits);
    let mut lut = Vec::with_capacity(u.len() * grid.len());
    for &value in u {
        lut.extend(grid.iter().map(|&point| value * point));
    }
    lut
}

/// Fold each pair of b=4 coordinate LUTs into the packed-byte domain. The
/// result is query-scoped and lets a batch scorer perform one lookup/add per
/// two coordinates.
pub fn build_packed_lut_4(lut: &[f32], d: usize) -> Vec<f32> {
    assert_eq!(lut.len(), d * 16);
    let pairs = d / 2;
    let mut packed_lut = Vec::with_capacity(packed_lut_len_4(d));
    for pair in 0..pairs {
        let low_lut = pair * 32;
        let high_lut = low_lut + 16;
        for packed in 0_u16..=255 {
            packed_lut.push(
                lut[low_lut + (packed as u8 & 0x0f) as usize]
                    + lut[high_lut + (packed as u8 >> 4) as usize],
            );
        }
    }
    if !d.is_multiple_of(2) {
        let tail_lut = pairs * 32;
        packed_lut.extend_from_slice(&lut[tail_lut..tail_lut + 16]);
    }
    debug_assert_eq!(packed_lut.len(), packed_lut_len_4(d));
    packed_lut
}

fn packed_lut_len_4(d: usize) -> usize {
    (d / 2) * 256 + (d % 2) * 16
}

pub fn score_batch_packed_4(
    codes: &[u8],
    code_stride: usize,
    packed_lut: &[f32],
    d: usize,
    out: &mut [f32],
) {
    assert_eq!(code_stride, packed_len(d, 4));
    assert_eq!(codes.len(), out.len() * code_stride);
    assert_eq!(packed_lut.len(), packed_lut_len_4(d));
    let pairs = d / 2;
    let mut row = 0;
    while row + 8 <= out.len() {
        let offsets = [
            row * code_stride,
            (row + 1) * code_stride,
            (row + 2) * code_stride,
            (row + 3) * code_stride,
            (row + 4) * code_stride,
            (row + 5) * code_stride,
            (row + 6) * code_stride,
            (row + 7) * code_stride,
        ];
        let mut sums = [0.0_f32; 8];
        for pair in 0..pairs {
            let lut_base = pair * 256;
            for lane in 0..8 {
                // SAFETY: lengths and strides are validated above; a packed
                // byte is exactly the 0..256 table index.
                unsafe {
                    let packed = *codes.get_unchecked(offsets[lane] + pair);
                    sums[lane] += *packed_lut.get_unchecked(lut_base + packed as usize);
                }
            }
        }
        if !d.is_multiple_of(2) {
            let lut_base = pairs * 256;
            for lane in 0..8 {
                // SAFETY: an odd dimension leaves one valid low nibble in
                // byte `pairs` and one trailing 16-entry LUT.
                unsafe {
                    let packed = *codes.get_unchecked(offsets[lane] + pairs);
                    sums[lane] += *packed_lut.get_unchecked(lut_base + (packed & 0x0f) as usize);
                }
            }
        }
        out[row..row + 8].copy_from_slice(&sums);
        row += 8;
    }
    for (local, score_out) in out[row..].iter_mut().enumerate() {
        let packed_row = &codes[(row + local) * code_stride..(row + local + 1) * code_stride];
        let mut sum = 0.0;
        for pair in 0..pairs {
            unsafe {
                sum += *packed_lut
                    .get_unchecked(pair * 256 + *packed_row.get_unchecked(pair) as usize);
            }
        }
        if !d.is_multiple_of(2) {
            unsafe {
                sum += *packed_lut.get_unchecked(
                    pairs * 256 + (*packed_row.get_unchecked(pairs) & 0x0f) as usize,
                );
            }
        }
        *score_out = sum;
    }
}

/// Score selected rows from one borrowed contiguous posting range. Row
/// offsets are local to `codes`; no survivor code bytes are gathered or
/// copied before entering this batch kernel.
pub fn score_batch_packed_4_indexed(
    codes: &[u8],
    code_stride: usize,
    row_offsets: &[usize],
    packed_lut: &[f32],
    d: usize,
    out: &mut [f32],
) {
    assert_eq!(code_stride, packed_len(d, 4));
    assert_eq!(codes.len() % code_stride, 0);
    assert_eq!(row_offsets.len(), out.len());
    assert!(row_offsets
        .iter()
        .all(|&row| row < codes.len() / code_stride));
    assert_eq!(packed_lut.len(), packed_lut_len_4(d));
    let pairs = d / 2;
    let mut row = 0;
    while row + 8 <= out.len() {
        let offsets = [
            row_offsets[row] * code_stride,
            row_offsets[row + 1] * code_stride,
            row_offsets[row + 2] * code_stride,
            row_offsets[row + 3] * code_stride,
            row_offsets[row + 4] * code_stride,
            row_offsets[row + 5] * code_stride,
            row_offsets[row + 6] * code_stride,
            row_offsets[row + 7] * code_stride,
        ];
        let mut sums = [0.0_f32; 8];
        for pair in 0..pairs {
            let lut_base = pair * 256;
            for lane in 0..8 {
                // SAFETY: row offsets, strides, and LUT dimensions are
                // validated above; each packed byte is a 0..256 index.
                unsafe {
                    let packed = *codes.get_unchecked(offsets[lane] + pair);
                    sums[lane] += *packed_lut.get_unchecked(lut_base + packed as usize);
                }
            }
        }
        if !d.is_multiple_of(2) {
            let lut_base = pairs * 256;
            for lane in 0..8 {
                unsafe {
                    let packed = *codes.get_unchecked(offsets[lane] + pairs);
                    sums[lane] += *packed_lut.get_unchecked(lut_base + (packed & 0x0f) as usize);
                }
            }
        }
        out[row..row + 8].copy_from_slice(&sums);
        row += 8;
    }
    for (local, score_out) in out[row..].iter_mut().enumerate() {
        let offset = row_offsets[row + local] * code_stride;
        let mut sum = 0.0;
        for pair in 0..pairs {
            unsafe {
                sum += *packed_lut
                    .get_unchecked(pair * 256 + *codes.get_unchecked(offset + pair) as usize);
            }
        }
        if !d.is_multiple_of(2) {
            unsafe {
                sum += *packed_lut.get_unchecked(
                    pairs * 256 + (*codes.get_unchecked(offset + pairs) & 0x0f) as usize,
                );
            }
        }
        *score_out = sum;
    }
}

pub fn score(codes: &[u8], lut: &[f32], d: usize, bits: u8) -> f32 {
    assert!(d > 0);
    assert_eq!(codes.len(), packed_len(d, bits));
    debug_assert!(tail_is_zero(codes, d, bits));
    let levels = 1usize << bits;
    assert_eq!(lut.len(), d * levels);
    let mask = (1_u8 << bits) - 1;
    if bits == 3 {
        return (0..d)
            .map(|i| lut[i * levels + code_at_3(codes, i) as usize])
            .sum();
    }
    let per_byte = 8 / bits as usize;
    (0..d)
        .map(|i| {
            let code = (codes[i / per_byte] >> (bits as usize * (i % per_byte))) & mask;
            lut[i * levels + code as usize]
        })
        .sum()
}

pub fn estimate(codes: &[u8], scale: u16, lut: &[f32], d: usize, bits: u8) -> f32 {
    f16_to_f32(scale) * score(codes, lut, d, bits)
}

/// Score a fixed-stride row batch in one kernel call, leaving scale and
/// split-form constant application to the caller's separate SoA pass.
pub fn score_batch(
    codes: &[u8],
    code_stride: usize,
    lut: &[f32],
    d: usize,
    bits: u8,
    out: &mut [f32],
) {
    assert_eq!(code_stride, packed_len(d, bits));
    assert_eq!(codes.len(), out.len() * code_stride);
    let levels = 1usize << bits;
    assert_eq!(lut.len(), d * levels);
    match bits {
        4 => score_batch_4(codes, code_stride, lut, d, out),
        2 => score_batch_2(codes, code_stride, lut, d, out),
        3 => {
            for (row, score_out) in codes.chunks_exact(code_stride).zip(out) {
                let mut sum = 0.0;
                for i in 0..d {
                    sum += lut[i * levels + code_at_3(row, i) as usize];
                }
                *score_out = sum;
            }
        }
        _ => unreachable!("grid widths are validated as 2..=4"),
    }
}

/// Score selected rows from one borrowed fixed-stride posting range without
/// gathering or copying their packed code bytes. `row_offsets` are local row
/// indices into `codes` and may be sparse or repeated.
pub fn score_batch_indexed(
    codes: &[u8],
    code_stride: usize,
    row_offsets: &[usize],
    lut: &[f32],
    d: usize,
    bits: u8,
    out: &mut [f32],
) {
    assert_eq!(code_stride, packed_len(d, bits));
    assert_eq!(codes.len() % code_stride, 0);
    assert_eq!(row_offsets.len(), out.len());
    assert!(row_offsets
        .iter()
        .all(|&row| row < codes.len() / code_stride));
    let levels = 1usize << bits;
    assert_eq!(lut.len(), d * levels);
    match bits {
        4 => score_batch_4_indexed(codes, code_stride, row_offsets, lut, d, out),
        2 => score_batch_2_indexed(codes, code_stride, row_offsets, lut, d, out),
        3 => {
            for (&row_offset, score_out) in row_offsets.iter().zip(out) {
                let base = row_offset * code_stride;
                let packed_row = &codes[base..base + code_stride];
                let mut sum = 0.0;
                for i in 0..d {
                    sum += lut[i * levels + code_at_3(packed_row, i) as usize];
                }
                *score_out = sum;
            }
        }
        _ => unreachable!("grid widths are validated as 2..=4"),
    }
}

#[inline(always)]
fn score_batch_4_indexed(
    codes: &[u8],
    stride: usize,
    row_offsets: &[usize],
    lut: &[f32],
    d: usize,
    out: &mut [f32],
) {
    let pairs = d / 2;
    let mut row = 0;
    while row + 8 <= out.len() {
        let offsets = [
            row_offsets[row] * stride,
            row_offsets[row + 1] * stride,
            row_offsets[row + 2] * stride,
            row_offsets[row + 3] * stride,
            row_offsets[row + 4] * stride,
            row_offsets[row + 5] * stride,
            row_offsets[row + 6] * stride,
            row_offsets[row + 7] * stride,
        ];
        let mut sums = [0.0_f32; 8];
        for pair in 0..pairs {
            let low_lut = pair * 32;
            let high_lut = low_lut + 16;
            for lane in 0..8 {
                unsafe {
                    let packed = *codes.get_unchecked(offsets[lane] + pair);
                    sums[lane] += *lut.get_unchecked(low_lut + (packed & 0x0f) as usize);
                    sums[lane] += *lut.get_unchecked(high_lut + (packed >> 4) as usize);
                }
            }
        }
        if !d.is_multiple_of(2) {
            let low_lut = pairs * 32;
            for lane in 0..8 {
                unsafe {
                    let packed = *codes.get_unchecked(offsets[lane] + pairs);
                    sums[lane] += *lut.get_unchecked(low_lut + (packed & 0x0f) as usize);
                }
            }
        }
        out[row..row + 8].copy_from_slice(&sums);
        row += 8;
    }
    for (local, score_out) in out[row..].iter_mut().enumerate() {
        let offset = row_offsets[row + local] * stride;
        let mut sum = 0.0;
        for pair in 0..pairs {
            unsafe {
                let packed = *codes.get_unchecked(offset + pair);
                let low_lut = pair * 32;
                sum += *lut.get_unchecked(low_lut + (packed & 0x0f) as usize);
                sum += *lut.get_unchecked(low_lut + 16 + (packed >> 4) as usize);
            }
        }
        if !d.is_multiple_of(2) {
            unsafe {
                sum += *lut.get_unchecked(
                    pairs * 32 + (*codes.get_unchecked(offset + pairs) & 0x0f) as usize,
                );
            }
        }
        *score_out = sum;
    }
}

#[inline(always)]
fn score_batch_2_indexed(
    codes: &[u8],
    stride: usize,
    row_offsets: &[usize],
    lut: &[f32],
    d: usize,
    out: &mut [f32],
) {
    for (&row_offset, score_out) in row_offsets.iter().zip(out) {
        let base = row_offset * stride;
        let packed_row = &codes[base..base + stride];
        let mut sum = 0.0;
        let full_bytes = d / 4;
        for (byte_index, &packed) in packed_row[..full_bytes].iter().enumerate() {
            let lut_base = byte_index * 16;
            sum += lut[lut_base + (packed & 0x03) as usize];
            sum += lut[lut_base + 4 + ((packed >> 2) & 0x03) as usize];
            sum += lut[lut_base + 8 + ((packed >> 4) & 0x03) as usize];
            sum += lut[lut_base + 12 + (packed >> 6) as usize];
        }
        for i in full_bytes * 4..d {
            let packed = packed_row[i / 4];
            let code = (packed >> (2 * (i % 4))) & 0x03;
            sum += lut[i * 4 + code as usize];
        }
        *score_out = sum;
    }
}

/// Score eight b=4 rows together so each pair of 16-entry coordinate LUTs is
/// hot while the batch consumes it. This is the refinement-path shape: one
/// kernel entry per survivor batch, with no scalar-row scorer calls.
#[inline(always)]
fn score_batch_4(codes: &[u8], stride: usize, lut: &[f32], d: usize, out: &mut [f32]) {
    let pairs = d / 2;
    let mut row = 0;
    while row + 8 <= out.len() {
        let offsets = [
            row * stride,
            (row + 1) * stride,
            (row + 2) * stride,
            (row + 3) * stride,
            (row + 4) * stride,
            (row + 5) * stride,
            (row + 6) * stride,
            (row + 7) * stride,
        ];
        let mut sums = [0.0_f32; 8];
        for pair in 0..pairs {
            let low_lut = pair * 32;
            let high_lut = low_lut + 16;
            for lane in 0..8 {
                // SAFETY: `score_batch` validates `codes == rows * stride`,
                // `stride == packed_len(d, 4)`, and `lut == d * 16` above.
                // `row + lane` is an output row, `pair < d / 2`, and each
                // nibble is in 0..16.
                let packed = unsafe { *codes.get_unchecked(offsets[lane] + pair) };
                unsafe {
                    sums[lane] += *lut.get_unchecked(low_lut + (packed & 0x0f) as usize);
                    sums[lane] += *lut.get_unchecked(high_lut + (packed >> 4) as usize);
                }
            }
        }
        if !d.is_multiple_of(2) {
            let low_lut = pairs * 32;
            for lane in 0..8 {
                // SAFETY: an odd `d` leaves one valid low nibble at `pairs`.
                let packed = unsafe { *codes.get_unchecked(offsets[lane] + pairs) };
                unsafe {
                    sums[lane] += *lut.get_unchecked(low_lut + (packed & 0x0f) as usize);
                }
            }
        }
        out[row..row + 8].copy_from_slice(&sums);
        row += 8;
    }
    for (local, score_out) in out[row..].iter_mut().enumerate() {
        let packed_row = &codes[(row + local) * stride..(row + local + 1) * stride];
        let mut sum = 0.0;
        for pair in 0..pairs {
            // SAFETY: `pair < d / 2 <= packed_len(d, 4)`.
            let packed = unsafe { *packed_row.get_unchecked(pair) };
            let low_lut = pair * 32;
            unsafe {
                sum += *lut.get_unchecked(low_lut + (packed & 0x0f) as usize);
                sum += *lut.get_unchecked(low_lut + 16 + (packed >> 4) as usize);
            }
        }
        if !d.is_multiple_of(2) {
            // SAFETY: the odd coordinate and its 16-entry LUT are present.
            unsafe {
                sum += *lut
                    .get_unchecked(pairs * 32 + (*packed_row.get_unchecked(pairs) & 0x0f) as usize);
            }
        }
        *score_out = sum;
    }
}

#[inline(always)]
fn score_batch_2(codes: &[u8], stride: usize, lut: &[f32], d: usize, out: &mut [f32]) {
    for (packed_row, score_out) in codes.chunks_exact(stride).zip(out) {
        let mut sum = 0.0;
        let full_bytes = d / 4;
        for (byte_index, &packed) in packed_row[..full_bytes].iter().enumerate() {
            let lut_base = byte_index * 16;
            sum += lut[lut_base + (packed & 0x03) as usize];
            sum += lut[lut_base + 4 + ((packed >> 2) & 0x03) as usize];
            sum += lut[lut_base + 8 + ((packed >> 4) & 0x03) as usize];
            sum += lut[lut_base + 12 + (packed >> 6) as usize];
        }
        for i in full_bytes * 4..d {
            let packed = packed_row[i / 4];
            let code = (packed >> (2 * (i % 4))) & 0x03;
            sum += lut[i * 4 + code as usize];
        }
        *score_out = sum;
    }
}

fn validate_grid(grid: &[f32], bits: u8) {
    assert!(matches!(bits, 2..=4));
    assert_eq!(grid.len(), 1usize << bits);
    debug_assert!(grid.windows(2).all(|pair| pair[0] < pair[1]));
}

fn code_at_3(packed: &[u8], i: usize) -> u8 {
    let bit_offset = i * 3;
    let byte = bit_offset / 8;
    let shift = bit_offset % 8;
    let low = packed[byte] >> shift;
    let high = if shift > 5 {
        packed[byte + 1] << (8 - shift)
    } else {
        0
    };
    (low | high) & 0b111
}

#[inline]
fn code_at(packed: &[u8], i: usize, bits: u8) -> u8 {
    if bits == 3 {
        return code_at_3(packed, i);
    }
    let per_byte = 8 / bits as usize;
    (packed[i / per_byte] >> (bits as usize * (i % per_byte))) & ((1_u8 << bits) - 1)
}

#[cfg(test)]
mod tests {
    use quant_model::build_grid;
    use rand_chacha::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;

    #[test]
    fn packing_round_trips_and_orders_bits() {
        let codes4: Vec<u8> = (0..128).map(|i| (i % 16) as u8).collect();
        let mut packed4 = vec![0; packed_len(128, 4)];
        pack(&codes4, 4, &mut packed4);
        assert_eq!(packed4[0], 0x10);
        assert_eq!(packed4[1], 0x32);
        assert_eq!(unpack(&packed4, 128, 4), codes4);

        let codes2: Vec<u8> = (0..128).map(|i| (i % 4) as u8).collect();
        let mut packed2 = vec![0; packed_len(128, 2)];
        pack(&codes2, 2, &mut packed2);
        assert_eq!(packed2[0], 0b11_10_01_00);
        assert_eq!(unpack(&packed2, 128, 2), codes2);

        let codes3: Vec<u8> = (0..128).map(|i| (i % 8) as u8).collect();
        let mut packed3 = vec![0; packed_len(128, 3)];
        pack(&codes3, 3, &mut packed3);
        assert_eq!(unpack(&packed3, 128, 3), codes3);
    }

    #[test]
    fn caller_scratch_encode_and_decode_match_wrappers() {
        for (d, bits) in [(65, 2), (100, 3), (769, 4)] {
            let grid = build_grid(d, bits);
            let values: Vec<f32> = (0..d).map(|i| ((i as f32 + 0.25) * 0.071).sin()).collect();
            let mut expected_codes = vec![0_u8; packed_len(d, bits)];
            let expected_scale = encode(&values, &grid.points, bits, &mut expected_codes);
            let expected_values = decode(&expected_codes, &grid.points, d, bits, expected_scale);

            let mut actual_codes = vec![0_u8; packed_len(d, bits)];
            let mut code_scratch = vec![0_u8; d];
            let actual_scale = encode_with_scratch(
                &values,
                &grid.points,
                bits,
                &mut actual_codes,
                &mut code_scratch,
            );
            let mut actual_values = vec![f32::NAN; d];
            decode_into(
                &actual_codes,
                &grid.points,
                bits,
                actual_scale,
                &mut actual_values,
            );
            assert_eq!(actual_scale, expected_scale);
            assert_eq!(actual_codes, expected_codes);
            assert_eq!(actual_values, expected_values);
        }
    }

    #[test]
    fn odd_dimension_round_trips_score_and_zero_tail() {
        for d in [65, 100, 300, 769] {
            for bits in 2..=4 {
                let codes: Vec<u8> = (0..d).map(|i| (i % (1usize << bits)) as u8).collect();
                let mut packed = vec![0xff; packed_len(d, bits)];
                pack(&codes, bits, &mut packed);
                assert!(tail_is_zero(&packed, d, bits), "d={d}, b={bits}");
                assert_eq!(unpack(&packed, d, bits), codes, "d={d}, b={bits}");

                let grid = build_grid(d, bits);
                let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.043).cos()).collect();
                let lut = build_lut(&query, &grid.points, bits);
                let actual = score(&packed, &lut, d, bits);
                let direct: f32 = query
                    .iter()
                    .zip(&codes)
                    .map(|(&q, &code)| q * grid.points[code as usize])
                    .sum();
                assert!((actual - direct).abs() < 1e-5, "d={d}, b={bits}");
            }
        }
    }

    #[test]
    fn batch_scoring_matches_scalar_for_all_widths_and_odd_tail() {
        for d in [100, 128, 769] {
            for bits in 2..=4 {
                let grid = build_grid(d, bits);
                let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.043).cos()).collect();
                let lut = build_lut(&query, &grid.points, bits);
                let stride = packed_len(d, bits);
                let rows = 19;
                let mut packed = vec![0_u8; rows * stride];
                for row in 0..rows {
                    let codes: Vec<u8> = (0..d)
                        .map(|i| ((i * 7 + row * 3) % (1usize << bits)) as u8)
                        .collect();
                    pack(&codes, bits, &mut packed[row * stride..(row + 1) * stride]);
                }
                let expected: Vec<f32> = packed
                    .chunks_exact(stride)
                    .map(|row| score(row, &lut, d, bits))
                    .collect();
                let mut actual = vec![0.0; rows];
                score_batch(&packed, stride, &lut, d, bits, &mut actual);
                for (expected, actual) in expected.into_iter().zip(actual) {
                    let tolerance = expected.abs().max(1.0) * 1e-5;
                    assert!((expected - actual).abs() <= tolerance, "d={d}, b={bits}");
                }
            }
        }
    }

    #[test]
    fn indexed_scoring_matches_contiguous_and_scalar_for_sparse_rows() {
        let row_offsets = [17, 0, 8, 22, 3, 11, 11, 5, 19];
        for d in [65, 100, 128] {
            for bits in 2..=4 {
                let grid = build_grid(d, bits);
                let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.043).cos()).collect();
                let lut = build_lut(&query, &grid.points, bits);
                let stride = packed_len(d, bits);
                let rows = 23;
                let mut packed = vec![0_u8; rows * stride];
                for row in 0..rows {
                    let codes: Vec<u8> = (0..d)
                        .map(|i| ((i * 7 + row * 3) % (1usize << bits)) as u8)
                        .collect();
                    pack(&codes, bits, &mut packed[row * stride..(row + 1) * stride]);
                }

                let expected: Vec<f32> = row_offsets
                    .iter()
                    .map(|&row| score(&packed[row * stride..(row + 1) * stride], &lut, d, bits))
                    .collect();
                let mut contiguous = vec![0.0; rows];
                score_batch(&packed, stride, &lut, d, bits, &mut contiguous);
                let mut indexed = vec![0.0; row_offsets.len()];
                score_batch_indexed(&packed, stride, &row_offsets, &lut, d, bits, &mut indexed);

                for ((&row, expected), actual) in row_offsets.iter().zip(expected).zip(indexed) {
                    let tolerance = expected.abs().max(1.0) * 1e-5;
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "indexed: d={d}, b={bits}, row={row}, actual={actual}, expected={expected}"
                    );
                    assert!(
                        (actual - contiguous[row]).abs() <= tolerance,
                        "contiguous: d={d}, b={bits}, row={row}"
                    );
                }
            }
        }
    }

    #[test]
    fn packed_b4_indexed_scoring_supports_odd_dimensions() {
        let row_offsets = [18, 2, 9, 0, 14, 6, 6];
        for d in [65, 100, 128, 769] {
            let bits = 4;
            let grid = build_grid(d, bits);
            let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.031).sin()).collect();
            let lut = build_lut(&query, &grid.points, bits);
            let packed_lut = build_packed_lut_4(&lut, d);
            assert_eq!(packed_lut.len(), packed_lut_len_4(d));
            let stride = packed_len(d, bits);
            let rows = 19;
            let mut packed = vec![0_u8; rows * stride];
            for row in 0..rows {
                let codes: Vec<u8> = (0..d).map(|i| ((i * 5 + row * 7) % 16) as u8).collect();
                pack(&codes, bits, &mut packed[row * stride..(row + 1) * stride]);
            }

            let mut contiguous = vec![0.0; rows];
            score_batch_packed_4(&packed, stride, &packed_lut, d, &mut contiguous);
            let mut indexed = vec![0.0; row_offsets.len()];
            score_batch_packed_4_indexed(
                &packed,
                stride,
                &row_offsets,
                &packed_lut,
                d,
                &mut indexed,
            );
            for (&row, actual) in row_offsets.iter().zip(indexed) {
                let expected = score(&packed[row * stride..(row + 1) * stride], &lut, d, bits);
                let tolerance = expected.abs().max(1.0) * 1e-5;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "indexed: d={d}, row={row}, actual={actual}, expected={expected}"
                );
                assert!(
                    (actual - contiguous[row]).abs() <= tolerance,
                    "contiguous: d={d}, row={row}"
                );
            }
        }
    }

    #[test]
    fn zero_vector_is_canonical() {
        let grid = build_grid(128, 4);
        let mut codes = vec![0xff; packed_len(128, 4)];
        let scale = encode(&[0.0; 128], &grid.points, 4, &mut codes);
        assert_eq!(scale, 0);
        assert!(codes.iter().all(|&byte| byte == 0));
        let lut = build_lut(&[0.25; 128], &grid.points, 4);
        assert_eq!(estimate(&codes, scale, &lut, 128, 4), 0.0);
    }

    #[test]
    fn lut_score_matches_direct_sum() {
        let d = 128;
        let grid = build_grid(d, 4);
        let vector: Vec<f32> = (0..d).map(|i| (i as f32 * 0.071).sin()).collect();
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.043).cos()).collect();
        let mut codes = vec![0; packed_len(d, 4)];
        let scale = encode(&vector, &grid.points, 4, &mut codes);
        let lut = build_lut(&query, &grid.points, 4);
        let actual = score(&codes, &lut, d, 4);
        let unpacked = unpack(&codes, d, 4);
        let direct: f32 = query
            .iter()
            .zip(unpacked)
            .map(|(&q, code)| q * grid.points[code as usize])
            .sum();
        assert!(
            (actual - direct).abs() < 1e-5,
            "{actual} != {direct}, scale={scale}"
        );
    }

    #[test]
    fn b3_lut_score_matches_direct_sum() {
        let d = 128;
        let grid = build_grid(d, 3);
        let vector: Vec<f32> = (0..d).map(|i| (i as f32 * 0.071).sin()).collect();
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.043).cos()).collect();
        let mut codes = vec![0; packed_len(d, 3)];
        encode(&vector, &grid.points, 3, &mut codes);
        let lut = build_lut(&query, &grid.points, 3);
        let actual = score(&codes, &lut, d, 3);
        let direct: f32 = query
            .iter()
            .zip(unpack(&codes, d, 3))
            .map(|(&q, code)| q * grid.points[code as usize])
            .sum();
        assert!((actual - direct).abs() < 1e-5, "{actual} != {direct}");
    }

    #[test]
    fn packed_pipeline_matches_reference_and_goldens() {
        let d = 768;
        for (bits, target, tolerance) in [(4, 0.0973, 0.004), (2, 0.3424, 0.010)] {
            let grid = build_grid(d, bits);
            let mut rng = ChaCha8Rng::seed_from_u64(0x4752_4944 ^ u64::from(bits));
            let mut packed_error = 0.0_f64;
            let mut reference_error = 0.0_f64;
            let mut energy = 0.0_f64;
            for _ in 0..1_000 {
                let vector = random_unit(&mut rng, d);
                let mut packed = vec![0; packed_len(d, bits)];
                let scale = encode(&vector, &grid.points, bits, &mut packed);
                let decoded = decode(&packed, &grid.points, d, bits, scale);
                let raw_scale = vector
                    .iter()
                    .map(|&value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    / (d as f32).sqrt();
                let stored_scale = f16_to_f32(f32_to_f16(raw_scale));
                let boundaries: Vec<f32> = grid
                    .points
                    .windows(2)
                    .map(|pair| (pair[0] + pair[1]) * 0.5)
                    .collect();
                for (&value, reconstructed) in vector.iter().zip(decoded) {
                    let code = boundaries.partition_point(|&boundary| value / raw_scale > boundary);
                    let reference = stored_scale * grid.points[code];
                    packed_error += f64::from(value - reconstructed).powi(2);
                    reference_error += f64::from(value - reference).powi(2);
                    energy += f64::from(value).powi(2);
                }
            }
            let packed_rho = (packed_error / energy).sqrt();
            let reference_rho = (reference_error / energy).sqrt();
            assert!((packed_rho - reference_rho).abs() < 1e-6);
            assert!(
                (packed_rho - target).abs() <= tolerance,
                "b={bits}: {packed_rho}"
            );
        }
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
}
