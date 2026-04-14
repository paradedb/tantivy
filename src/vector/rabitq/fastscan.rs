//! FastScan LUT-based distance computation.
//!
//! Replaces float dot products with precomputed lookup table lookups.
//! For each group of 4 dimensions, we precompute all 2^4=16 possible
//! dot products. Then each 4-bit nibble of the packed binary code
//! becomes a direct index into the LUT — no multiplications needed.

pub const BATCH_SIZE: usize = 32;

const KPOS: [usize; 16] = [3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3];

#[inline]
fn lowbit(x: usize) -> usize {
    x & x.wrapping_neg()
}

pub struct QueryLut {
    lut: Vec<f32>,
    num_codebooks: usize,
}

impl QueryLut {
    pub fn new(rotated_query: &[f32]) -> Self {
        let dim = rotated_query.len();
        assert!(dim % 4 == 0);
        let num_codebooks = dim / 4;
        let mut lut = vec![0.0f32; num_codebooks * 16];

        for i in 0..num_codebooks {
            let q_offset = i * 4;
            let lut_offset = i * 16;

            lut[lut_offset] = 0.0;
            for j in 1..16 {
                let prev_idx = j - lowbit(j);
                lut[lut_offset + j] =
                    lut[lut_offset + prev_idx] + rotated_query[q_offset + KPOS[j]];
            }
        }

        Self { lut, num_codebooks }
    }

    /// Compute the binary dot product for a single vector using LUT lookups.
    ///
    /// `packed_binary` is the packed binary code (1 bit per dim, 8 dims per byte).
    /// Returns the equivalent of `sum(binary_code[i] * rotated_query[i])`.
    pub fn binary_dot(&self, packed_binary: &[u8]) -> f32 {
        let mut result = 0.0f32;

        // Each byte encodes 8 dimensions = 2 codebooks (2 nibbles)
        for byte_idx in 0..packed_binary.len() {
            let byte = packed_binary[byte_idx];
            let codebook_base = byte_idx * 2;

            if codebook_base >= self.num_codebooks {
                break;
            }

            // High nibble (byte >> 4): dimensions byte_idx*8 .. byte_idx*8+3
            let hi = ((byte >> 4) & 0x0F) as usize;
            result += self.lut[codebook_base * 16 + hi];

            // Low nibble (byte & 0xF): dimensions byte_idx*8+4 .. byte_idx*8+7
            if codebook_base + 1 < self.num_codebooks {
                let lo = (byte & 0x0F) as usize;
                result += self.lut[(codebook_base + 1) * 16 + lo];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_matches_scalar() {
        let dim = 64;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) / dim as f32 - 0.5).collect();
        let lut = QueryLut::new(&query);

        // Create a binary code: alternating 0/1
        let binary_code: Vec<u8> = (0..dim).map(|i| (i % 2) as u8).collect();

        // Scalar dot product
        let scalar_dot: f32 = binary_code
            .iter()
            .zip(query.iter())
            .map(|(&b, &q)| (b as f32) * q)
            .sum();

        // Pack binary code using the same MSB-first packing as simd.rs
        let packed_len = dim / 8;
        let mut packed = vec![0u8; packed_len];
        for i in 0..dim {
            if binary_code[i] == 1 {
                packed[i / 8] |= 1 << (7 - (i % 8));
            }
        }

        // LUT dot product
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
}
