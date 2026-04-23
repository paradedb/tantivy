//! On-disk record layout for TurboQuant-encoded vectors.
//!
//! Each record is fixed-size and self-describing (given the shared
//! configuration: padded dimension + bit width). Layout:
//!
//! ```text
//! ┌─────────────────────────┬──────────────────────┬─────────────┐
//! │ stage1 indices          │ stage2 QJL sign bits │ γ (f32 LE)  │
//! │ packed_byte_size(       │ ceil(padded_dim/8)   │ 4 bytes     │
//! │   padded_dim, s1_bits)  │                      │             │
//! └─────────────────────────┴──────────────────────┴─────────────┘
//! ```
//!
//! where
//!   - `s1_bits = bit_width - 1`  (Stage 1 MSE code)
//!   - `padded_dim` is the rotator's padded dimension (the vector is zero-padded to that size
//!     before rotation).
//!   - `γ` is the L2 norm of the Stage 1 residual (in rotated space), used by the QJL estimator.
//!
//! For `bit_width = 1` the record has Stage 2 only (just QJL signs +
//! norm) — no Stage 1 codes.

/// Packed byte offset at which the Stage 2 sign bits begin.
#[inline]
pub fn stage2_offset(padded_dim: usize, bit_width: u8) -> usize {
    let s1_bits = bit_width.saturating_sub(1) as usize;
    (padded_dim * s1_bits).div_ceil(8)
}

/// Packed byte offset at which the residual-norm scalar (γ) begins.
#[inline]
pub fn norm_offset(padded_dim: usize, bit_width: u8) -> usize {
    stage2_offset(padded_dim, bit_width) + padded_dim.div_ceil(8)
}

/// Total bytes per record.
#[inline]
pub fn bytes_per_record(padded_dim: usize, bit_width: u8) -> usize {
    norm_offset(padded_dim, bit_width) + 4
}

/// Read the residual norm γ from a record.
#[inline]
pub fn read_norm(record: &[u8], padded_dim: usize, bit_width: u8) -> f32 {
    let off = norm_offset(padded_dim, bit_width);
    f32::from_le_bytes([
        record[off],
        record[off + 1],
        record[off + 2],
        record[off + 3],
    ])
}

/// Write γ into a record buffer.
#[inline]
pub fn write_norm(record: &mut [u8], padded_dim: usize, bit_width: u8, gamma: f32) {
    let off = norm_offset(padded_dim, bit_width);
    record[off..off + 4].copy_from_slice(&gamma.to_le_bytes());
}

/// Slice of a record holding the Stage 1 bit-packed indices.
#[inline]
pub fn stage1_bytes(record: &[u8], padded_dim: usize, bit_width: u8) -> &[u8] {
    &record[..stage2_offset(padded_dim, bit_width)]
}

/// Slice of a record holding the Stage 2 packed sign bits.
#[inline]
pub fn stage2_bytes(record: &[u8], padded_dim: usize, bit_width: u8) -> &[u8] {
    let start = stage2_offset(padded_dim, bit_width);
    let end = norm_offset(padded_dim, bit_width);
    &record[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_layout_b3_d768() {
        let d = 768;
        let bw = 3u8;
        // stage1: 768 * 2 bits = 1536 bits = 192 bytes
        // stage2: 768 bits = 96 bytes
        // norm: 4 bytes
        // total: 292 bytes
        assert_eq!(stage2_offset(d, bw), 192);
        assert_eq!(norm_offset(d, bw), 192 + 96);
        assert_eq!(bytes_per_record(d, bw), 192 + 96 + 4);
    }

    #[test]
    fn record_layout_b1_has_only_stage2() {
        // bit_width=1 means just QJL signs + norm; no Stage 1 bits.
        let d = 128;
        assert_eq!(stage2_offset(d, 1), 0);
        assert_eq!(norm_offset(d, 1), 16);
        assert_eq!(bytes_per_record(d, 1), 20);
    }

    #[test]
    fn norm_roundtrip() {
        let d = 768;
        let bw = 3;
        let mut rec = vec![0u8; bytes_per_record(d, bw)];
        write_norm(&mut rec, d, bw, 1.234_f32);
        assert!((read_norm(&rec, d, bw) - 1.234_f32).abs() < 1e-6);
    }
}
