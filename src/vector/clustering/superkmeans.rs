//! The built-in [`IvfClusterer`]: hierarchical SuperKMeans.

use std::fmt;
use std::sync::{Arc, Mutex};

use superkmeans::{Dataset, HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, Matrix};

use crate::schema::{Metric, VectorOptions};
use crate::vector::clustering::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingVectors, IvfVectors, TrainingSource,
};
use crate::TantivyError;

/// Default [`SuperKMeansClusterer::with_max_leaf_size`].
pub const DEFAULT_MAX_LEAF_SIZE: usize = 200;
/// Default [`SuperKMeansClusterer::with_training_sample_ratio`].
pub const DEFAULT_TRAINING_SAMPLE_RATIO: f64 = 0.32;

const TRAINING_BATCH_SIZE: usize = 4_096;
const ASSIGN_BATCH_SIZE: usize = 40_960;

/// Replays a [`TrainingSource`] as a SuperKMeans [`Dataset`].
struct SourceDataset<'a>(&'a mut dyn TrainingSource);

impl Dataset for SourceDataset<'_> {
    type Error = TantivyError;

    fn for_each_batch(&mut self, f: &mut dyn FnMut(Matrix<'_>)) -> crate::Result<()> {
        self.0
            .for_each_batch(TRAINING_BATCH_SIZE, &mut |IvfVectors::F32(batch)| {
                f(Matrix::new(
                    batch.matrix.values,
                    batch.matrix.rows,
                    batch.matrix.dims,
                ))
            })
    }
}

/// Trains centroids with hierarchical SuperKMeans and assigns each vector to its nearest one.
/// Cosine and Dot cluster angularly.
#[derive(Clone)]
pub struct SuperKMeansClusterer {
    config: HierarchicalSuperKMeansConfig,
    training_sample_ratio: f32,
    /// The assigner for the last `(dim, angular)`, reused across a merge's batches.
    assigner: Arc<Mutex<Option<Assigner>>>,
}

struct Assigner {
    dim: usize,
    angular: bool,
    kmeans: Arc<HierarchicalSuperKMeans>,
}

impl Default for SuperKMeansClusterer {
    fn default() -> Self {
        let mut config = HierarchicalSuperKMeansConfig::default();
        config.base.suppress_warnings = true;
        config.max_leaf_size = DEFAULT_MAX_LEAF_SIZE;
        Self {
            config,
            training_sample_ratio: DEFAULT_TRAINING_SAMPLE_RATIO as f32,
            assigner: Arc::new(Mutex::new(None)),
        }
    }
}

impl fmt::Debug for SuperKMeansClusterer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SuperKMeansClusterer")
            .field("config", &self.config)
            .field("training_sample_ratio", &self.training_sample_ratio)
            .finish_non_exhaustive()
    }
}

impl SuperKMeansClusterer {
    /// At most `max_leaf_size` training vectors per clustering leaf.
    #[must_use]
    pub fn with_max_leaf_size(mut self, max_leaf_size: usize) -> Self {
        self.config.max_leaf_size = max_leaf_size;
        self
    }

    /// Trains on `ratio` of the vectors, in `(0, 1]`.
    #[must_use]
    pub fn with_training_sample_ratio(mut self, ratio: f32) -> Self {
        self.training_sample_ratio = ratio;
        self
    }

    /// The assigner for `(dim, angular)`, built on first use and kept for the next batch.
    fn assigner(&self, dim: usize, angular: bool) -> Arc<HierarchicalSuperKMeans> {
        let mut cached = self
            .assigner
            .lock()
            .expect("SuperKMeans assigner mutex poisoned");
        match cached.as_ref() {
            Some(assigner) if assigner.dim == dim && assigner.angular == angular => {
                Arc::clone(&assigner.kmeans)
            }
            _ => {
                let mut config = self.config.clone();
                config.base.angular = angular;
                let kmeans = Arc::new(HierarchicalSuperKMeans::with_config(dim, config));
                *cached = Some(Assigner {
                    dim,
                    angular,
                    kmeans: Arc::clone(&kmeans),
                });
                kmeans
            }
        }
    }
}

/// Whether `metric` clusters by angle.
fn is_angular(metric: Metric) -> bool {
    matches!(metric, Metric::Cosine | Metric::Dot)
}

impl IvfClusterer for SuperKMeansClusterer {
    fn training_sample_ratio(&self) -> f32 {
        self.training_sample_ratio
    }

    fn assign_batch_size(&self) -> usize {
        ASSIGN_BATCH_SIZE
    }

    fn train(
        &self,
        options: &VectorOptions,
        source: &mut dyn TrainingSource,
    ) -> crate::Result<IvfCentroids> {
        let dim = options.dim();
        let num_rows = source.num_rows();
        if num_rows == 0 {
            return Err(TantivyError::InvalidArgument(
                "cannot train IVF centroids on zero vectors".to_string(),
            ));
        }
        let mut config = self.config.clone();
        if is_angular(options.metric()) {
            config.base.angular = true;
        }
        let kmeans = HierarchicalSuperKMeans::with_config(dim, config);

        // Only the meso step scans every row; each meso cluster is then loaded and split on its
        // own, so peak memory is one meso cluster.
        let meso = kmeans.train_meso(&mut SourceDataset(&mut *source), num_rows)?;
        let mut centroids = Vec::new();
        for m in 0..meso.n_meso() {
            let IvfTrainingVectors::F32(vectors) = source.read_rows(meso.meso_rows(m))?;
            if vectors.matrix.dims != dim {
                return Err(TantivyError::InvalidArgument(format!(
                    "vector dimensionality mismatch: expected {dim}, got {}",
                    vectors.matrix.dims
                )));
            }
            let fine = kmeans.train_meso_cluster(&vectors.matrix.values, vectors.matrix.rows);
            centroids.extend_from_slice(&fine.centroids);
        }
        if centroids.is_empty() || centroids.len() % dim != 0 {
            return Err(TantivyError::InternalError(format!(
                "SuperKMeans returned an invalid centroid matrix with {} floats for dimension \
                 {dim}",
                centroids.len()
            )));
        }
        Ok(IvfCentroids::F32(IvfMatrix {
            rows: centroids.len() / dim,
            values: centroids,
            dims: dim,
        }))
    }

    fn assign(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<u32>> {
        let IvfVectors::F32(vectors) = vectors;
        let IvfCentroids::F32(centroids) = centroids;
        if vectors.matrix.rows == 0 {
            return Ok(Vec::new());
        }
        let kmeans = self.assigner(options.dim(), is_angular(options.metric()));
        Ok(kmeans.assign(
            vectors.matrix.values,
            centroids.values.as_slice(),
            vectors.matrix.rows,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::clustering::{IvfMatrixView, IvfTrainingBatch, IvfVectorBatch};

    /// In-memory [`TrainingSource`] that counts full scans.
    struct VecSource {
        values: Vec<f32>,
        dim: usize,
        scans: usize,
    }

    impl TrainingSource for VecSource {
        fn num_rows(&self) -> usize {
            self.values.len() / self.dim
        }

        fn for_each_batch(
            &mut self,
            batch_size: usize,
            f: &mut dyn FnMut(IvfVectors<'_>),
        ) -> crate::Result<()> {
            self.scans += 1;
            for (i, chunk) in self.values.chunks(batch_size * self.dim).enumerate() {
                let rows = chunk.len() / self.dim;
                let first = (i * batch_size) as u32;
                let doc_ids: Vec<u32> = (first..first + rows as u32).collect();
                f(IvfVectors::F32(IvfVectorBatch {
                    doc_ids: &doc_ids,
                    matrix: IvfMatrixView {
                        values: chunk,
                        rows,
                        dims: self.dim,
                    },
                }));
            }
            Ok(())
        }

        fn read_rows(&mut self, rows: &[u32]) -> crate::Result<IvfTrainingVectors> {
            let values = rows
                .iter()
                .flat_map(|&row| {
                    self.values[row as usize * self.dim..][..self.dim]
                        .iter()
                        .copied()
                })
                .collect();
            Ok(IvfTrainingVectors::F32(IvfTrainingBatch {
                doc_ids: rows.to_vec(),
                matrix: IvfMatrix {
                    values,
                    rows: rows.len(),
                    dims: self.dim,
                },
            }))
        }
    }

    #[test]
    fn trains_by_streaming_the_source() {
        let (n, dim) = (5_000, 16);
        let mut source = VecSource {
            values: superkmeans::make_blobs(n, dim, 25, false, 1.0, 10.0, 42),
            dim,
            scans: 0,
        };
        let clusterer = SuperKMeansClusterer::default().with_max_leaf_size(100);
        let options = VectorOptions::new(dim, Metric::L2);

        let IvfCentroids::F32(centroids) = clusterer.train(&options, &mut source).unwrap();
        assert_eq!(centroids.dims, dim);
        assert_eq!(centroids.values.len(), centroids.rows * dim);
        assert!(centroids.rows >= n / 100, "{} centroids", centroids.rows);
        assert!(source.scans > 1, "meso step should replay the source");

        let assignments = clusterer
            .assign(
                &options,
                IvfVectors::F32(IvfVectorBatch {
                    doc_ids: &(0..n as u32).collect::<Vec<_>>(),
                    matrix: IvfMatrixView {
                        values: &source.values,
                        rows: n,
                        dims: dim,
                    },
                }),
                &IvfCentroids::F32(centroids.clone()),
            )
            .unwrap();
        assert!(assignments.iter().all(|&c| (c as usize) < centroids.rows));
    }

    #[test]
    fn rejects_an_empty_source() {
        let mut source = VecSource {
            values: Vec::new(),
            dim: 4,
            scans: 0,
        };
        let options = VectorOptions::new(4, Metric::L2);
        assert!(SuperKMeansClusterer::default()
            .train(&options, &mut source)
            .is_err());
        assert_eq!(source.scans, 0);
    }

    /// The defaults are the documented constants, and the two setters are independent.
    #[test]
    fn settings_default_and_set_independently() {
        let clusterer = SuperKMeansClusterer::default();
        assert_eq!(clusterer.training_sample_ratio(), 0.32);
        assert_eq!(clusterer.assign_batch_size(), ASSIGN_BATCH_SIZE);
        assert_eq!(clusterer.config.max_leaf_size, DEFAULT_MAX_LEAF_SIZE);

        let clusterer = clusterer
            .with_max_leaf_size(20)
            .with_training_sample_ratio(0.25);
        assert_eq!(clusterer.training_sample_ratio(), 0.25);
        assert_eq!(clusterer.config.max_leaf_size, 20);
        let clusterer = clusterer.with_training_sample_ratio(1.0);
        assert_eq!(clusterer.training_sample_ratio(), 1.0);
        assert_eq!(clusterer.config.max_leaf_size, 20);
    }
}
