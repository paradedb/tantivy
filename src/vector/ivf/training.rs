use crate::schema::VectorOptions;
use crate::vector::{sq4_stride, Sq4Params, VectorElement};
use crate::{DocId, TantivyError};

pub trait IvfClusterer: Send + Sync + 'static {
    fn centroid_ratio(&self) -> f32;

    fn training_samples_per_centroid(&self) -> usize;

    fn train(
        &self,
        options: &VectorOptions,
        vectors: IvfTrainingVectors,
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
        Ok(IvfMergeSettings {
            num_centroids,
            training_samples_per_centroid,
            assign_batch_size,
            // Replication off by default (primary-only layout).
            replicas: 1,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IvfMergeSettings {
    pub num_centroids: usize,
    pub training_samples_per_centroid: usize,
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
    /// Full-precision centroids — what [`IvfClusterer::train`] returns.
    F32(IvfMatrix<f32>),
    /// SQ4-quantized centroids — what the merge hands to
    /// [`IvfClusterer::assign`] after compressing the trained set, so the
    /// full f32 matrix never persists past training. Score against them
    /// with the `sq4` kernels (e.g. via
    /// [`Sq4Arena`](crate::vector::Sq4Arena)) or decode rows on demand
    /// with [`Sq4Params::decode_row_into`].
    Sq4(Sq4Centroids),
}

impl IvfCentroids {
    /// The number of centroid rows.
    pub fn rows(&self) -> usize {
        match self {
            IvfCentroids::F32(matrix) => matrix.rows,
            IvfCentroids::Sq4(sq4) => sq4.rows,
        }
    }

    /// The vector dimensionality.
    pub fn dims(&self) -> usize {
        match self {
            IvfCentroids::F32(matrix) => matrix.dims,
            IvfCentroids::Sq4(sq4) => sq4.params.dim(),
        }
    }

    /// Decodes centroid `row` into `out` (`dims` f32s) — a copy for `F32`,
    /// the reconstruction for `Sq4`. Codec-agnostic scoring for `assign`
    /// implementations that don't use the SQ4 kernels directly.
    pub fn decode_row_into(&self, row: usize, out: &mut [f32]) {
        match self {
            IvfCentroids::F32(matrix) => {
                out.copy_from_slice(&matrix.values[row * matrix.dims..][..matrix.dims])
            }
            IvfCentroids::Sq4(sq4) => sq4.params.decode_row_into(sq4.row(row), out),
        }
    }
}

/// Packed SQ4 centroid rows plus their quantization params; row `i` is
/// `codes[i * sq4_stride(dim)..][..sq4_stride(dim)]`.
#[derive(Clone, Debug)]
pub struct Sq4Centroids {
    pub codes: Vec<u8>,
    pub params: Sq4Params,
    pub rows: usize,
}

impl Sq4Centroids {
    /// The packed codes of centroid `row`.
    pub fn row(&self, row: usize) -> &[u8] {
        let stride = sq4_stride(self.params.dim());
        &self.codes[row * stride..][..stride]
    }
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
