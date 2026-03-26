//! Pack/unpack [`QuantizedVector`] to/from flat byte records for BqVecPlugin storage.
//!
//! Record layout (all little-endian):
//! ```text
//! [binary_code_packed: padded_dims/8 bytes]
//! [ex_code_packed:     (padded_dims * ex_bits) / 8 bytes]  (0 if ex_bits == 0)
//! [delta:          f32]
//! [vl:             f32]
//! [f_add:          f32]
//! [f_rescale:      f32]
//! [f_error:        f32]
//! [residual_norm:  f32]
//! [f_add_ex:       f32]
//! [f_rescale_ex:   f32]
//! ```

use super::quantizer::QuantizedVector;

const NUM_SCALARS: usize = 8;
const SCALAR_BYTES: usize = NUM_SCALARS * 4; // 32 bytes

/// Total bytes per record for the given padded dimensionality and extended bits.
pub fn bytes_per_record(padded_dims: usize, ex_bits: usize) -> usize {
    let binary_bytes = padded_dims.div_ceil(8);
    let ex_bytes = if ex_bits > 0 {
        (padded_dims * ex_bits).div_ceil(8)
    } else {
        0
    };
    binary_bytes + ex_bytes + SCALAR_BYTES
}

/// Pack a [`QuantizedVector`] into a flat byte record.
pub fn pack(qv: &QuantizedVector) -> Vec<u8> {
    let total = qv.binary_code_packed.len() + qv.ex_code_packed.len() + SCALAR_BYTES;
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&qv.binary_code_packed);
    buf.extend_from_slice(&qv.ex_code_packed);
    buf.extend_from_slice(&qv.delta.to_le_bytes());
    buf.extend_from_slice(&qv.vl.to_le_bytes());
    buf.extend_from_slice(&qv.f_add.to_le_bytes());
    buf.extend_from_slice(&qv.f_rescale.to_le_bytes());
    buf.extend_from_slice(&qv.f_error.to_le_bytes());
    buf.extend_from_slice(&qv.residual_norm.to_le_bytes());
    buf.extend_from_slice(&qv.f_add_ex.to_le_bytes());
    buf.extend_from_slice(&qv.f_rescale_ex.to_le_bytes());
    buf
}

/// Unpack a flat byte record into a [`QuantizedVector`].
pub fn unpack(bytes: &[u8], padded_dims: usize, ex_bits: usize) -> QuantizedVector {
    let binary_bytes = padded_dims.div_ceil(8);
    let ex_bytes = if ex_bits > 0 {
        (padded_dims * ex_bits).div_ceil(8)
    } else {
        0
    };

    let mut pos = 0;
    let binary_code_packed = bytes[pos..pos + binary_bytes].to_vec();
    pos += binary_bytes;

    let ex_code_packed = bytes[pos..pos + ex_bytes].to_vec();
    pos += ex_bytes;

    let read_f32 = |pos: &mut usize| -> f32 {
        let val = f32::from_le_bytes([bytes[*pos], bytes[*pos + 1], bytes[*pos + 2], bytes[*pos + 3]]);
        *pos += 4;
        val
    };

    let delta = read_f32(&mut pos);
    let vl = read_f32(&mut pos);
    let f_add = read_f32(&mut pos);
    let f_rescale = read_f32(&mut pos);
    let f_error = read_f32(&mut pos);
    let residual_norm = read_f32(&mut pos);
    let f_add_ex = read_f32(&mut pos);
    let f_rescale_ex = read_f32(&mut pos);

    QuantizedVector {
        binary_code_packed,
        ex_code_packed,
        ex_bits: ex_bits as u8,
        dim: padded_dims,
        delta,
        vl,
        f_add,
        f_rescale,
        f_error,
        residual_norm,
        f_add_ex,
        f_rescale_ex,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let qv = QuantizedVector {
            binary_code_packed: vec![0xAA, 0x55],
            ex_code_packed: vec![],
            ex_bits: 0,
            dim: 16,
            delta: 1.5,
            vl: -0.3,
            f_add: 2.0,
            f_rescale: -1.0,
            f_error: 0.01,
            residual_norm: 3.14,
            f_add_ex: 0.0,
            f_rescale_ex: 0.0,
        };

        let packed = pack(&qv);
        assert_eq!(packed.len(), bytes_per_record(16, 0));

        let unpacked = unpack(&packed, 16, 0);
        assert_eq!(unpacked.binary_code_packed, qv.binary_code_packed);
        assert_eq!(unpacked.delta, qv.delta);
        assert_eq!(unpacked.vl, qv.vl);
        assert_eq!(unpacked.f_add, qv.f_add);
        assert_eq!(unpacked.f_rescale, qv.f_rescale);
        assert_eq!(unpacked.f_error, qv.f_error);
        assert_eq!(unpacked.residual_norm, qv.residual_norm);
    }

    #[test]
    fn test_pack_unpack_with_ex_bits() {
        let qv = QuantizedVector {
            binary_code_packed: vec![0xFF; 4],   // 32 dims / 8
            ex_code_packed: vec![0x12; 24],       // 32 * 6 / 8 = 24 bytes
            ex_bits: 6,
            dim: 32,
            delta: 1.0,
            vl: 2.0,
            f_add: 3.0,
            f_rescale: 4.0,
            f_error: 5.0,
            residual_norm: 6.0,
            f_add_ex: 7.0,
            f_rescale_ex: 8.0,
        };

        let packed = pack(&qv);
        assert_eq!(packed.len(), bytes_per_record(32, 6));
        assert_eq!(packed.len(), 4 + 24 + 32);

        let unpacked = unpack(&packed, 32, 6);
        assert_eq!(unpacked.binary_code_packed, qv.binary_code_packed);
        assert_eq!(unpacked.ex_code_packed, qv.ex_code_packed);
        assert_eq!(unpacked.f_add_ex, 7.0);
        assert_eq!(unpacked.f_rescale_ex, 8.0);
    }
}
