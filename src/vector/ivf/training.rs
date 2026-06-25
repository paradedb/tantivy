use crate::vector::options::{VectorElement, VectorOptions};
use crate::{DocId, TantivyError};

pub trait IvfClusterer: Send + Sync + 'static {
    fn centroid_ratio(&self) -> f32;

    fn training_samples_per_centroid(&self) -> usize;

    fn train(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        num_centroids: usize,
    ) -> crate::Result<IvfCentroids>;

    fn assign(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<u32>>;

    fn assign_batch_size(&self) -> usize {
        2048
    }

    fn merge_settings(&self, total_target_docs: usize) -> crate::Result<IvfMergeSettings> {
        let centroid_ratio = self.centroid_ratio();
        let training_samples_per_centroid = self.training_samples_per_centroid();
        let assign_batch_size = self.assign_batch_size();

        assert!(
            centroid_ratio > 0.0 && centroid_ratio <= 1.0,
            "IvfClusterer centroid_ratio must be greater than 0 and less than or equal to 1, got \
             {centroid_ratio}"
        );
        assert!(
            training_samples_per_centroid > 1,
            "IvfClusterer training_samples_per_centroid must be greater than 1, got \
             {training_samples_per_centroid}"
        );
        assert!(
            assign_batch_size > 0,
            "IvfClusterer assign_batch_size must be greater than 0, got {assign_batch_size}"
        );

        let num_centroids =
            ((total_target_docs as f64) * f64::from(centroid_ratio)).ceil() as usize;
        let num_centroids = num_centroids.clamp(1, total_target_docs);

        // Default size bounds the merge driver enforces on PRIMARY cluster
        // membership: split clusters above `max_posting_len`, dissolve those
        // below `min_posting_len`. Both are anchored to the mean posting
        // length (≈ `1 / centroid_ratio`). Provisional factors — tune
        // against real-data size sweeps. A clusterer that wants to opt out
        // of balancing can override `merge_settings` and pass
        // `max_posting_len = usize::MAX`, `min_posting_len = 0`.
        let mean_posting_len = (total_target_docs / num_centroids).max(1);
        let max_posting_len = mean_posting_len.saturating_mul(MAX_POSTING_FACTOR);
        let min_posting_len = (mean_posting_len / MIN_POSTING_DIVISOR).max(1);

        Ok(IvfMergeSettings {
            num_centroids,
            training_samples_per_centroid,
            assign_batch_size,
            max_posting_len,
            min_posting_len,
        })
    }
}

/// Split clusters whose primary membership exceeds `MAX_POSTING_FACTOR ×`
/// the mean posting length. Provisional — see [`IvfClusterer::merge_settings`].
const MAX_POSTING_FACTOR: usize = 2;
/// Dissolve clusters whose primary membership falls below `1 /
/// MIN_POSTING_DIVISOR ×` the mean posting length. Provisional.
const MIN_POSTING_DIVISOR: usize = 2;

#[derive(Clone, Copy, Debug)]
pub struct IvfMergeSettings {
    pub num_centroids: usize,
    pub training_samples_per_centroid: usize,
    pub assign_batch_size: usize,
    /// Hard upper bound on a primary cluster's size. The merge driver
    /// splits any cluster above this into sub-clusters. `usize::MAX`
    /// disables splitting.
    pub max_posting_len: usize,
    /// Lower bound on a primary cluster's size. The merge driver dissolves
    /// any cluster below this and reassigns its members to the nearest
    /// surviving centroid. `0` disables merging.
    pub min_posting_len: usize,
}

#[derive(Clone, Debug)]
pub enum IvfCentroids {
    F32(IvfMatrix<f32>),
}

#[derive(Clone, Copy, Debug)]
pub enum IvfVectors<'a> {
    F32(IvfVectorBatch<'a, f32>),
}

#[derive(Clone, Debug)]
pub struct IvfMatrix<T> {
    pub values: Vec<T>,
    pub rows: usize,
    pub dims: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct IvfMatrixView<'a, T> {
    pub values: &'a [T],
    pub rows: usize,
    pub dims: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct IvfVectorBatch<'a, T> {
    pub doc_ids: &'a [DocId],
    pub matrix: IvfMatrixView<'a, T>,
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
        element.encode_le(&mut bytes);
    }
    Ok(bytes)
}
