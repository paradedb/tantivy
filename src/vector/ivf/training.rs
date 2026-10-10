use crate::vector::VectorElement;
use crate::TantivyError;

#[derive(Clone, Debug)]
/// Trained centroid matrix.
pub enum IvfCentroids {
    /// Binary32 centroids.
    F32(IvfMatrix<f32>),
}

#[derive(Clone, Debug)]
/// Owned row-major matrix.
pub struct IvfMatrix<T> {
    /// Row-major values.
    pub values: Vec<T>,
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub dims: usize,
}

pub(crate) fn decode_row<T: VectorElement>(bytes: &[u8], dim: usize) -> crate::Result<Vec<T>> {
    let mut decoded = Vec::with_capacity(dim);
    decode_row_append(bytes, dim, &mut decoded)?;
    Ok(decoded)
}

/// Decodes a row into caller-owned storage.
pub(crate) fn decode_row_append<T: VectorElement>(
    bytes: &[u8],
    dim: usize,
    decoded: &mut Vec<T>,
) -> crate::Result<()> {
    let expected = dim * T::SIZE_BYTES;
    if bytes.len() != expected {
        return Err(TantivyError::InvalidArgument(format!(
            "vector byte length mismatch: expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    decoded.extend(bytes.chunks_exact(T::SIZE_BYTES).map(T::decode_le));
    Ok(())
}

pub(crate) fn encode_vector<T: VectorElement>(vector: &[T], dim: usize) -> crate::Result<Vec<u8>> {
    if vector.len() != dim {
        return Err(TantivyError::InvalidArgument(format!(
            "centroid length mismatch: expected {dim} elements, got {}",
            vector.len()
        )));
    }
    let mut bytes = Vec::with_capacity(dim * T::SIZE_BYTES);
    for element in vector {
        element.encode_le(&mut bytes)?;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_row_append_reuses_the_batch_allocation() {
        let values = [1.25_f32, -2.5, 3.75];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut decoded = Vec::with_capacity(8);
        decoded.push(99.0_f32);
        let allocation = decoded.as_ptr();
        decode_row_append::<f32>(&bytes, values.len(), &mut decoded).unwrap();
        assert_eq!(decoded.as_ptr(), allocation);
        assert_eq!(decoded, [99.0, 1.25, -2.5, 3.75]);
    }

    #[test]
    fn decode_row_append_rejects_shape_before_mutating_batch() {
        let mut decoded = vec![7.0_f32];
        assert!(decode_row_append::<f32>(&[0; 3], 1, &mut decoded).is_err());
        assert_eq!(decoded, [7.0]);
    }
}
