use super::BKTree;
use crate::schema::VectorOptions;
use crate::vector::VectorElement;
use crate::{DocId, TantivyError};

pub trait IvfClusterer: Send + Sync + 'static {
    fn centroid_ratio(&self) -> f32;

    /// Fraction of vectors sampled for training, in `(0, 1]`.
    fn training_sample_ratio(&self) -> f32;

    fn train(
        &self,
        options: &VectorOptions,
        vectors: IvfTrainingVectors,
    ) -> crate::Result<IvfCentroids>;

    fn assign(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<u32>>;

    /// Optional BKT over `centroids` for RNG seeding.
    ///
    /// Default `None` omits `.centroids` slot `[4]`. When present, leaf
    /// `members` are IVF / RNG [`NodeId`](super::NodeId)s.
    fn build_bkt(
        &self,
        options: &VectorOptions,
        centroids: &IvfCentroids,
    ) -> crate::Result<Option<BKTree<Vec<f32>>>> {
        let _ = (options, centroids);
        Ok(None)
    }

    fn assign_batch_size(&self) -> usize {
        2048
    }

    fn merge_settings(&self, _total_target_docs: usize) -> crate::Result<IvfMergeSettings> {
        let centroid_ratio = self.centroid_ratio();
        let training_sample_ratio = self.training_sample_ratio();
        let assign_batch_size = self.assign_batch_size();

        assert!(
            centroid_ratio > 0.0 && centroid_ratio <= 1.0,
            "IvfClusterer centroid_ratio must be greater than 0 and less than or equal to 1, got \
             {centroid_ratio}"
        );
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
            // Replication off by default (primary-only layout).
            replicas: 1,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IvfMergeSettings {
    /// Fraction of vectors sampled for training, in `(0, 1]`.
    pub training_sample_ratio: f32,
    pub assign_batch_size: usize,
    /// Total number of cells a vector is written into (SPANN `ReplicaCount`):
    /// the primary plus up to `replicas - 1` additional cells taken from the
    /// nearest centroids — selected exactly for small centroid sets, via a
    /// transient build-time neighborhood graph for large ones. `1` (the
    /// default) disables replication entirely — no selector is built and the
    /// output is the primary-only layout.
    pub replicas: usize,
}

#[derive(Clone, Debug)]
pub enum IvfCentroids {
    F32(IvfMatrix<f32>),
}

#[derive(Clone, Copy, Debug)]
pub enum IvfVectors<'a> {
    F32(IvfVectorBatch<'a, f32>),
}

/// Owned training input: the merge hands its sampled buffers to the clusterer,
/// which may consume them in place instead of copying.
#[derive(Clone, Debug)]
pub enum IvfTrainingVectors {
    F32(IvfTrainingBatch<f32>),
}

#[derive(Clone, Debug)]
pub struct IvfTrainingBatch<T> {
    pub doc_ids: Vec<DocId>,
    pub matrix: IvfMatrix<T>,
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
        element.encode_le(&mut bytes)?;
    }
    Ok(bytes)
}
