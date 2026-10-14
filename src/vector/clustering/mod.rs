//! Custom clustering: the [`IvfClusterer`] a merge trains and assigns with, its inputs and
//! outputs, and the built-in [`SuperKMeansClusterer`].

use crate::schema::{VectorDType, VectorOptions};
use crate::vector::VectorElement;
use crate::{DocId, TantivyError};

mod superkmeans;

pub use superkmeans::{SuperKMeansClusterer, DEFAULT_MAX_LEAF_SIZE, DEFAULT_TRAINING_SAMPLE_RATIO};

pub trait IvfClusterer: Send + Sync + 'static {
    /// Fraction of vectors sampled for training, in `(0, 1]`.
    fn training_sample_ratio(&self) -> f32;

    /// Trains centroids.
    fn train(
        &self,
        options: &VectorOptions,
        vectors: IvfTrainingVectors,
    ) -> crate::Result<IvfCentroids>;

    /// Assigns vectors to centroids.
    fn assign(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<u32>>;

    fn assign_batch_size(&self) -> usize {
        2048
    }

    fn merge_settings(&self, _total_target_docs: usize) -> crate::Result<IvfMergeSettings> {
        let training_sample_ratio = self.training_sample_ratio();
        let assign_batch_size = self.assign_batch_size();

        assert!(
            training_sample_ratio > 0.0 && training_sample_ratio <= 1.0,
            "IvfClusterer training_sample_ratio must be greater than 0 and less than or equal to \
             1, got {training_sample_ratio}"
        );
        assert!(
            assign_batch_size > 0,
            "IvfClusterer assign_batch_size must be greater than 0, got {assign_batch_size}"
        );

        Ok(IvfMergeSettings {
            training_sample_ratio,
            assign_batch_size,
        })
    }
}

#[derive(Clone, Copy, Debug)]
/// Merge-time IVF sizes.
pub struct IvfMergeSettings {
    /// Fraction of vectors sampled for training, in `(0, 1]`.
    pub training_sample_ratio: f32,
    pub assign_batch_size: usize,
}

#[derive(Clone, Debug)]
/// Trained centroid matrix.
pub enum IvfCentroids {
    /// Binary32 centroids.
    F32(IvfMatrix<f32>),
}

#[derive(Clone, Copy, Debug)]
/// Borrowed vector batch.
pub enum IvfVectors<'a> {
    /// Binary32 vector batch.
    F32(IvfVectorBatch<'a, f32>),
}

/// Owned vector training input.
#[derive(Clone, Debug)]
pub enum IvfTrainingVectors {
    /// Binary32 training batch.
    F32(IvfTrainingBatch<f32>),
}

#[derive(Clone, Debug)]
/// Owned training rows and document identifiers.
pub struct IvfTrainingBatch<T> {
    /// Document identifiers.
    pub doc_ids: Vec<DocId>,
    /// Training matrix.
    pub matrix: IvfMatrix<T>,
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

#[derive(Clone, Copy, Debug)]
/// Borrowed row-major matrix.
pub struct IvfMatrixView<'a, T> {
    /// Row-major values.
    pub values: &'a [T],
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub dims: usize,
}

#[derive(Clone, Copy, Debug)]
/// Borrowed vectors and document identifiers.
pub struct IvfVectorBatch<'a, T> {
    /// Document identifiers.
    pub doc_ids: &'a [DocId],
    /// Vector matrix.
    pub matrix: IvfMatrixView<'a, T>,
}

pub(crate) fn decode_row<T: VectorElement>(bytes: &[u8], dim: usize) -> crate::Result<Vec<T>> {
    let mut decoded = Vec::with_capacity(dim);
    decode_row_append(bytes, dim, &mut decoded)?;
    Ok(decoded)
}

pub(crate) fn decode_stored_row(bytes: &[u8], options: &VectorOptions) -> crate::Result<Vec<f32>> {
    let mut decoded = Vec::with_capacity(options.dim());
    decode_stored_row_append(bytes, options, &mut decoded)?;
    Ok(decoded)
}

pub(crate) fn decode_stored_row_append(
    bytes: &[u8],
    options: &VectorOptions,
    decoded: &mut Vec<f32>,
) -> crate::Result<()> {
    if bytes.len() != options.bytes_per_vector() {
        return Err(TantivyError::InvalidArgument(
            "stored vector byte length mismatch".into(),
        ));
    }
    match options.dtype() {
        VectorDType::F32 => decode_row_append(bytes, options.dim(), decoded),
        VectorDType::F16 => {
            let coords = options.dim() * 2;
            let inverse = if options.needs_normalization() {
                f32::from_le_bytes(bytes[coords..].try_into().unwrap())
            } else {
                1.0
            };
            decoded.extend(bytes[..coords].chunks_exact(2).map(|v| {
                quant::model::f16::f16_to_f32(u16::from_le_bytes([v[0], v[1]])) * inverse
            }));
            Ok(())
        }
    }
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
