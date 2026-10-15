//! The built-in [`IvfClusterer`]: hierarchical SuperKMeans.

use std::fmt;
use std::sync::{Arc, Mutex};

use superkmeans::{Dataset, HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, Matrix};

use crate::schema::{Metric, VectorOptions};
use crate::vector::clustering::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingVectors, IvfVectors, VectorSource,
};
use crate::TantivyError;

/// Default [`SuperKMeansClusterer::with_max_leaf_size`].
pub const DEFAULT_MAX_LEAF_SIZE: usize = 200;
/// Default [`SuperKMeansClusterer::with_training_sample_ratio`].
pub const DEFAULT_TRAINING_SAMPLE_RATIO: f64 = 0.32;

const DEFAULT_BATCH_SIZE: usize = 40_960;

/// Replays a [`VectorSource`] as a SuperKMeans [`Dataset`].
struct SourceDataset<'a> {
    source: &'a mut dyn VectorSource,
    batch_size: usize,
}

impl Dataset for SourceDataset<'_> {
    type Error = TantivyError;

    fn for_each_batch(&mut self, f: &mut dyn FnMut(Matrix<'_>)) -> crate::Result<()> {
        self.source
            .for_each_batch(self.batch_size, &mut |IvfVectors::F32(batch)| {
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
    memory_limit: Option<usize>,
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
        Self::new()
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
    /// Defaults: [`DEFAULT_MAX_LEAF_SIZE`], [`DEFAULT_TRAINING_SAMPLE_RATIO`], no memory limit.
    pub fn new() -> Self {
        let mut config = HierarchicalSuperKMeansConfig::default();
        config.base.suppress_warnings = true;
        config.max_leaf_size = DEFAULT_MAX_LEAF_SIZE;
        Self {
            config,
            training_sample_ratio: DEFAULT_TRAINING_SAMPLE_RATIO as f32,
            memory_limit: None,
            assigner: Arc::new(Mutex::new(None)),
        }
    }

    /// At most `max_leaf_size` training vectors per clustering leaf.
    pub fn with_max_leaf_size(mut self, max_leaf_size: usize) -> Self {
        self.config.max_leaf_size = max_leaf_size;
        self
    }

    /// Caps the vectors held at once (meso batches, fine-step groups, assign batches) at about
    /// `memory_limit` bytes.
    pub fn with_memory_limit(mut self, memory_limit: usize) -> Self {
        self.memory_limit = Some(memory_limit);
        self
    }

    /// Trains on `ratio` of the vectors, in `(0, 1]`.
    pub fn with_training_sample_ratio(mut self, ratio: f32) -> Self {
        self.training_sample_ratio = ratio;
        self
    }

    fn batch_size(&self, vector_options: &VectorOptions) -> usize {
        let batch_size = match self.memory_limit {
            Some(byte_limit) => byte_limit.saturating_div(vector_options.bytes_per_vector()),
            None => DEFAULT_BATCH_SIZE,
        };
        batch_size.max(1)
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

/// The end of the fine-step group of meso clusters starting at `start`, given the partition's
/// `offsets`. Clusters join while the group plus the fine step's working copy of its largest
/// cluster stays within `capacity` rows; a group always holds at least one cluster.
fn fine_group_end(offsets: &[usize], start: usize, capacity: usize) -> usize {
    let n_meso = offsets.len() - 1;
    let mut end = start + 1;
    let mut largest = offsets[end] - offsets[start];
    while end < n_meso {
        let next_largest = largest.max(offsets[end + 1] - offsets[end]);
        if offsets[end + 1] - offsets[start] + next_largest > capacity {
            break;
        }
        largest = next_largest;
        end += 1;
    }
    end
}

/// Whether `metric` clusters by angle.
fn is_angular(metric: Metric) -> bool {
    matches!(metric, Metric::Cosine | Metric::Dot)
}

impl IvfClusterer for SuperKMeansClusterer {
    fn training_sample_ratio(&self) -> f32 {
        self.training_sample_ratio
    }

    fn train(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
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

        // Only the meso step scans every row. The fine step then loads consecutive meso clusters
        // as one group per `read_rows`, so the source can fetch them in a single ordered pass.
        let batch_size = self.batch_size(options);
        let meso = kmeans.train_meso(
            &mut SourceDataset {
                source: &mut *source,
                batch_size,
            },
            num_rows,
        )?;
        let mut centroids = Vec::new();
        let mut start = 0;
        while start < meso.n_meso() {
            let end = fine_group_end(&meso.offsets, start, batch_size);
            let group_start = meso.offsets[start];
            let group_rows = &meso.rows[group_start..meso.offsets[end]];
            let IvfTrainingVectors::F32(group) = source.read_rows(group_rows)?;
            if group.matrix.dims != dim || group.matrix.rows != group_rows.len() {
                return Err(TantivyError::InvalidArgument(format!(
                    "vector source returned {} rows of dimension {}, expected {} of dimension \
                     {dim}",
                    group.matrix.rows,
                    group.matrix.dims,
                    group_rows.len()
                )));
            }
            for m in start..end {
                let rows = meso.offsets[m] - group_start..meso.offsets[m + 1] - group_start;
                let fine = kmeans.train_meso_cluster(
                    &group.matrix.values[rows.start * dim..rows.end * dim],
                    rows.len(),
                );
                centroids.extend_from_slice(&fine.centroids);
            }
            start = end;
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
        source: &mut dyn VectorSource,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<u32>> {
        let mut assignments = Vec::with_capacity(source.num_rows());
        let kmeans = self.assigner(options.dim(), is_angular(options.metric()));
        let IvfCentroids::F32(centroids) = centroids;
        source.for_each_batch(self.batch_size(options), &mut |vectors| {
            let IvfVectors::F32(batch) = vectors;
            assignments.extend(kmeans.assign(
                batch.matrix.values,
                centroids.values.as_slice(),
                batch.matrix.rows,
            ));
        })?;
        Ok(assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::clustering::{IvfMatrixView, IvfTrainingBatch, IvfVectorBatch};

    /// In-memory [`VectorSource`] that counts full scans and `read_rows` calls.
    struct VecSource {
        values: Vec<f32>,
        dim: usize,
        scans: usize,
        row_reads: usize,
    }

    impl VecSource {
        fn new(values: Vec<f32>, dim: usize) -> Self {
            VecSource {
                values,
                dim,
                scans: 0,
                row_reads: 0,
            }
        }
    }

    impl VectorSource for VecSource {
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
            self.row_reads += 1;
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
        let mut source = VecSource::new(
            superkmeans::make_blobs(n, dim, 25, false, 1.0, 10.0, 42),
            dim,
        );
        let clusterer = SuperKMeansClusterer::new().with_max_leaf_size(100);
        let options = VectorOptions::new(dim, Metric::L2);

        let IvfCentroids::F32(centroids) = clusterer.train(&options, &mut source).unwrap();
        assert_eq!(centroids.dims, dim);
        assert_eq!(centroids.values.len(), centroids.rows * dim);
        assert!(centroids.rows >= n / 100, "{} centroids", centroids.rows);
        assert!(source.scans > 1, "meso step should replay the source");

        let assignments = clusterer
            .assign(&options, &mut source, &IvfCentroids::F32(centroids.clone()))
            .unwrap();
        assert_eq!(assignments.len(), n);
        assert!(assignments.iter().all(|&c| (c as usize) < centroids.rows));
    }

    #[test]
    fn rejects_an_empty_source() {
        let mut source = VecSource::new(Vec::new(), 4);
        let options = VectorOptions::new(4, Metric::L2);
        assert!(SuperKMeansClusterer::new()
            .train(&options, &mut source)
            .is_err());
        assert_eq!(source.scans, 0);
    }

    /// Meso clusters of 4, 3, 5 and 2 rows: a group plus its largest cluster fits the capacity,
    /// and an oversized cluster still forms a group of one.
    #[test]
    fn fine_groups_fit_the_capacity() {
        let offsets = [0, 4, 7, 12, 14];
        assert_eq!(fine_group_end(&offsets, 0, 100), 4);
        assert_eq!(fine_group_end(&offsets, 0, 11), 2);
        assert_eq!(fine_group_end(&offsets, 0, 10), 1);
        assert_eq!(fine_group_end(&offsets, 2, 10), 3);
        assert_eq!(fine_group_end(&offsets, 3, 4), 4);
        assert_eq!(fine_group_end(&offsets, 0, 1), 1);
    }

    /// Without a tight limit the fine step reads every meso cluster in one call; a tight limit
    /// splits it into more calls, each covering whole meso clusters.
    #[test]
    fn fine_step_reads_meso_clusters_in_groups() {
        let (n, dim) = (5_000, 16);
        let values = superkmeans::make_blobs(n, dim, 25, false, 1.0, 10.0, 42);
        let options = VectorOptions::new(dim, Metric::L2);

        let mut source = VecSource::new(values.clone(), dim);
        let clusterer = SuperKMeansClusterer::new().with_max_leaf_size(100);
        clusterer.train(&options, &mut source).unwrap();
        assert_eq!(source.row_reads, 1);

        let mut source = VecSource::new(values, dim);
        let clusterer = clusterer.with_memory_limit(1_000 * options.bytes_per_vector());
        let IvfCentroids::F32(centroids) = clusterer.train(&options, &mut source).unwrap();
        assert!(source.row_reads > 1, "{} row reads", source.row_reads);
        assert!(centroids.rows >= n / 100, "{} centroids", centroids.rows);
    }

    /// The defaults are the documented constants, and the two setters are independent.
    #[test]
    fn settings_default_and_set_independently() {
        let clusterer = SuperKMeansClusterer::new();
        assert_eq!(clusterer.training_sample_ratio(), 0.32);
        assert_eq!(clusterer.config.max_leaf_size, DEFAULT_MAX_LEAF_SIZE);

        let clusterer = SuperKMeansClusterer::new()
            .with_max_leaf_size(20)
            .with_training_sample_ratio(0.25);
        assert_eq!(clusterer.training_sample_ratio(), 0.25);
        assert_eq!(clusterer.config.max_leaf_size, 20);
        let clusterer = clusterer.with_training_sample_ratio(1.0);
        assert_eq!(clusterer.training_sample_ratio(), 1.0);
        assert_eq!(clusterer.config.max_leaf_size, 20);
    }
}
