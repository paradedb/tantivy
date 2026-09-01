//! Packed exact-density grid encoding and LUT scoring kernels.

use quant_model::f16::{f16_to_f32, f32_to_f16};

pub fn packed_len(d: usize, bits: u8) -> usize {
    assert!(d > 0 && d.is_multiple_of(64));
    assert!(matches!(bits, 2..=4));
    (d * bits as usize).div_ceil(8)
}

pub fn pack(codes: &[u8], bits: u8, out: &mut [u8]) {
    assert!(!codes.is_empty() && codes.len().is_multiple_of(64));
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
        return;
    }
    let per_byte = 8 / bits as usize;
    for (i, &code) in codes.iter().enumerate() {
        assert!(code <= mask);
        out[i / per_byte] |= code << (bits as usize * (i % per_byte));
    }
}

pub fn unpack(packed: &[u8], d: usize, bits: u8) -> Vec<u8> {
    assert!(d > 0 && d.is_multiple_of(64));
    assert_eq!(packed.len(), packed_len(d, bits));
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
    assert!(!y.is_empty() && y.len().is_multiple_of(64));
    validate_grid(grid, bits);
    assert_eq!(out.len(), packed_len(y.len(), bits));
    let norm_squared = y.iter().map(|&value| value * value).sum::<f32>();
    if norm_squared == 0.0 {
        out.fill(0);
        return 0;
    }

    let scale = norm_squared.sqrt() / (y.len() as f32).sqrt();
    let boundaries: Vec<f32> = grid
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect();
    let codes: Vec<u8> = y
        .iter()
        .map(|&value| boundaries.partition_point(|&boundary| value / scale > boundary) as u8)
        .collect();
    pack(&codes, bits, out);
    f32_to_f16(scale)
}

pub fn decode(codes: &[u8], grid: &[f32], d: usize, bits: u8, scale: u16) -> Vec<f32> {
    assert!(d > 0 && d.is_multiple_of(64));
    validate_grid(grid, bits);
    let scale = f16_to_f32(scale);
    unpack(codes, d, bits)
        .into_iter()
        .map(|code| scale * grid[code as usize])
        .collect()
}

pub fn build_lut(u: &[f32], grid: &[f32], bits: u8) -> Vec<f32> {
    assert!(!u.is_empty() && u.len().is_multiple_of(64));
    validate_grid(grid, bits);
    let mut lut = Vec::with_capacity(u.len() * grid.len());
    for &value in u {
        lut.extend(grid.iter().map(|&point| value * point));
    }
    lut
}

pub fn score(codes: &[u8], lut: &[f32], d: usize, bits: u8) -> f32 {
    assert!(d > 0 && d.is_multiple_of(64));
    assert_eq!(codes.len(), packed_len(d, bits));
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
