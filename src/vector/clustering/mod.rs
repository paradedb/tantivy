//! Custom clustering: the [`IvfClusterer`] a merge trains and assigns with, its inputs and
//! outputs, and the built-in [`SuperKMeansClusterer`].

use std::ops::Range;

use crate::schema::VectorOptions;
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
        source: &mut dyn VectorSource,
    ) -> crate::Result<IvfCentroids>;

    /// Assigns vectors to centroids.
    fn assign(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<u32>>;

    /// Assigns one primary and at most one replica per source vector.
    fn assign_with_replicas(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<ClusterAssignment>> {
        Ok(self
            .assign(options, source, centroids)?
            .into_iter()
            .map(|primary| ClusterAssignment {
                primary,
                secondary: None,
            })
            .collect())
    }
}

/// A vector's primary cluster and optional boundary replica.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterAssignment {
    pub primary: u32,
    pub secondary: Option<u32>,
}

/// Replayable, read-only view of vectors for IVF [`IvfClusterer::train`] or
/// [`IvfClusterer::assign`].
///
/// Rows are identified by their 0-based position in a fixed scan order, and
/// every read returns the same vectors for the same positions. Reads hand the
/// decoded rows to the caller and keep none, so what the caller holds is all
/// the vector memory a train or assign uses.
///
/// Merge builds a sample for training and the full set for assignment. Reads
/// fail with [`TantivyError::Cancelled`] if the merge is cancelled.
pub trait VectorSource {
    /// Number of rows in scan order.
    fn num_rows(&self) -> usize;

    /// Loads the rows in `range` of scan order.
    fn read_range(&mut self, range: Range<usize>) -> crate::Result<IvfTrainingVectors>;

    /// Loads the rows at the given scan positions, returned in the given order.
    ///
    /// Positions may be scattered and unsorted; implementations should fetch
    /// them in scan order, so one large request is cheaper than many small
    /// ones.
    fn read_rows(&mut self, rows: &[u32]) -> crate::Result<IvfTrainingVectors>;

    /// Polled between compute steps. Once the training sample fits in memory the clusterer
    /// stops reading, so the reads' own cancel checks never run again.
    fn wants_cancel(&self) -> bool {
        false
    }
}

/// Reads every row of `source` in scan order, at most `batch_rows` at a time, and calls `f`
/// with each batch. A batch is dropped once `f` returns unless `f` keeps it.
pub fn for_each_batch(
    source: &mut dyn VectorSource,
    batch_rows: usize,
    mut f: impl FnMut(IvfTrainingVectors),
) -> crate::Result<()> {
    let num_rows = source.num_rows();
    for start in (0..num_rows).step_by(batch_rows.max(1)) {
        let end = start.saturating_add(batch_rows.max(1)).min(num_rows);
        f(source.read_range(start..end)?);
    }
    Ok(())
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

/// Decodes a row into a caller-owned slot of exactly one row.
pub(crate) fn decode_row_into<T: VectorElement>(
    bytes: &[u8],
    decoded: &mut [T],
) -> crate::Result<()> {
    let expected = decoded.len() * T::SIZE_BYTES;
    if bytes.len() != expected {
        return Err(TantivyError::InvalidArgument(format!(
            "vector byte length mismatch: expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    for (value, chunk) in decoded.iter_mut().zip(bytes.chunks_exact(T::SIZE_BYTES)) {
        *value = T::decode_le(chunk);
    }
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
