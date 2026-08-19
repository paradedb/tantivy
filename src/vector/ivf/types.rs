//! Matrix types crossing the [`CentroidIndex`](crate::vector::CentroidIndex)
//! boundary, plus the row byte codecs shared by the write paths.

use crate::vector::VectorElement;
use crate::TantivyError;

/// A consumer-provided centroid matrix, one variant per supported dtype.
#[derive(Clone, Debug)]
pub enum IvfCentroids {
    F32(IvfMatrix<f32>),
}

/// A dense row-major matrix: `rows` vectors of `dims` elements.
#[derive(Clone, Debug)]
pub struct IvfMatrix<T> {
    pub values: Vec<T>,
    pub rows: usize,
    pub dims: usize,
}

pub(crate) fn decode_row<T: VectorElement>(bytes: &[u8], dim: usize) -> crate::Result<Vec<T>> {
    let expected = dim * T::SIZE_BYTES;
    if bytes.len() != expected {
        return Err(TantivyError::InvalidArgument(format!(
            "vector byte length mismatch: expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(T::SIZE_BYTES)
        .map(T::decode_le)
        .collect())
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
