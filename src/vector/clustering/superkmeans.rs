//! The built-in [`IvfClusterer`]: hierarchical SuperKMeans.

use std::fmt;

use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig};

use crate::schema::{Metric, VectorOptions};
use crate::vector::clustering::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingVectors, IvfVectors,
};
use crate::TantivyError;

/// Default [`SuperKMeansClusterer::with_max_leaf_size`].
pub const DEFAULT_MAX_LEAF_SIZE: usize = 200;
/// Default [`SuperKMeansClusterer::with_training_sample_ratio`].
pub const DEFAULT_TRAINING_SAMPLE_RATIO: f64 = 0.32;
const ASSIGN_BATCH_SIZE: usize = 40_960;

/// Trains centroids with hierarchical SuperKMeans and assigns each vector to its nearest one.
/// Cosine and Dot cluster angularly.
#[derive(Clone)]
pub struct SuperKMeansClusterer {
    config: HierarchicalSuperKMeansConfig,
    training_sample_ratio: f32,
}

impl Default for SuperKMeansClusterer {
    fn default() -> Self {
        let mut config = HierarchicalSuperKMeansConfig::default();
        config.base.suppress_warnings = true;
        config.max_leaf_size = DEFAULT_MAX_LEAF_SIZE;
        Self {
            config,
            training_sample_ratio: DEFAULT_TRAINING_SAMPLE_RATIO as f32,
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
        vectors: IvfTrainingVectors,
    ) -> crate::Result<IvfCentroids> {
        let IvfTrainingVectors::F32(vectors) = vectors;
        let dim = options.dim();
        let mut config = self.config.clone();
        if is_angular(options.metric()) {
            config.base.angular = true;
        }
        let mut kmeans = HierarchicalSuperKMeans::with_config(dim, config);
        let rows = vectors.matrix.rows;
        let centroids = kmeans.train_owned(vectors.matrix.values, rows);
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
        Ok(superkmeans::assign(
            vectors.matrix.values,
            &centroids.values,
            vectors.matrix.rows,
            options.dim(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::clustering::{IvfMatrixView, IvfVectorBatch};

    /// The defaults are the documented constants, and the two setters are independent.
    #[test]
    fn settings_default_and_set_independently() {
        let clusterer = SuperKMeansClusterer::default();
        let settings = clusterer.merge_settings(10_000).unwrap();
        assert_eq!(settings.training_sample_ratio, 0.32);
        assert_eq!(settings.assign_batch_size, ASSIGN_BATCH_SIZE);
        assert_eq!(clusterer.config.max_leaf_size, DEFAULT_MAX_LEAF_SIZE);

        let clusterer = clusterer
            .with_max_leaf_size(20)
            .with_training_sample_ratio(0.25);
        assert_eq!(clusterer.training_sample_ratio(), 0.25);
        assert_eq!(clusterer.config.max_leaf_size, 20);
        let clusterer = clusterer.with_training_sample_ratio(1.0);
        assert_eq!(
            clusterer.merge_settings(100).unwrap().training_sample_ratio,
            1.0
        );
        assert_eq!(clusterer.config.max_leaf_size, 20);
    }

    #[test]
    fn assignment_matches_brute_force() -> crate::Result<()> {
        let clusterer = SuperKMeansClusterer::default();
        for dim in [3, 64, 1024] {
            let mut values = superkmeans::generate_random_vectors(1031, dim, -1.0, 1.0, 42);
            let mut centers = superkmeans::generate_random_vectors(19, dim, -1.0, 1.0, 17);
            centers.extend_from_within(..dim);
            values[..dim].copy_from_slice(&centers[..dim]);
            let centroids = IvfCentroids::F32(IvfMatrix {
                values: centers.clone(),
                rows: 20,
                dims: dim,
            });
            for rows in [0, 1, 1031] {
                let doc_ids = (0..rows as u32).collect::<Vec<_>>();
                let values = &values[..rows * dim];
                let mut expected = vec![0; rows];
                let mut distances = vec![0.0; rows];
                superkmeans::find_nearest_neighbor_brute_force(
                    values,
                    &centers,
                    rows,
                    20,
                    dim,
                    &mut expected,
                    &mut distances,
                );
                for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
                    let actual = clusterer.assign(
                        &VectorOptions::new(dim, metric),
                        IvfVectors::F32(IvfVectorBatch {
                            doc_ids: &doc_ids,
                            matrix: IvfMatrixView {
                                values,
                                rows,
                                dims: dim,
                            },
                        }),
                        &centroids,
                    )?;
                    assert_eq!(
                        actual, expected,
                        "dim={dim}, rows={rows}, metric={metric:?}"
                    );
                }
            }
        }
        Ok(())
    }
}
