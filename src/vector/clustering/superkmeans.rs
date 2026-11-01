//! The built-in [`IvfClusterer`]: hierarchical SuperKMeans.

use std::fmt;
use std::sync::{Arc, Mutex};

use superkmeans::batch::find_k_nearest_neighbors;
use superkmeans::utils::{normalize_rows_l2, squared_norms};
use superkmeans::{Dataset, HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, Matrix};

use crate::schema::{Metric, VectorOptions};
use crate::vector::clustering::{
    for_each_batch, ClusterAssignment, IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingVectors,
    VectorSource,
};
use crate::vector::distance::{cosine, dot, l2_squared};
use crate::TantivyError;

/// Default [`SuperKMeansClusterer::with_max_leaf_size`].
pub const DEFAULT_MAX_LEAF_SIZE: usize = 200;
/// Default [`SuperKMeansClusterer::with_training_sample_ratio`].
pub const DEFAULT_TRAINING_SAMPLE_RATIO: f64 = 0.32;

/// Rows held at once without a memory limit.
const DEFAULT_BUDGET_ROWS: usize = 40_960;

/// Replays a [`VectorSource`] as a SuperKMeans [`Dataset`]. The first pass keeps rows
/// `0..kept_rows` decoded so later passes skip reading them; the rest stream through batches of
/// `batch_rows`.
struct SourceDataset<'a> {
    source: &'a mut dyn VectorSource,
    dim: usize,
    kept_rows: usize,
    batch_rows: usize,
    kept: Option<IvfMatrix<f32>>,
}

impl Dataset for SourceDataset<'_> {
    type Error = TantivyError;

    fn for_each_batch(&mut self, f: &mut dyn FnMut(Matrix<'_>)) -> crate::Result<()> {
        let num_rows = self.source.num_rows();
        if self.kept.is_none() && self.kept_rows > 0 {
            let IvfTrainingVectors::F32(batch) = self.source.read_range(0..self.kept_rows)?;
            self.kept = Some(checked_matrix(batch.matrix, self.kept_rows, self.dim)?);
        }
        if let Some(kept) = &self.kept {
            if self.source.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            f(Matrix::new(&kept.values, kept.rows, kept.dims));
        }
        for start in (self.kept_rows..num_rows).step_by(self.batch_rows) {
            let end = (start + self.batch_rows).min(num_rows);
            let IvfTrainingVectors::F32(batch) = self.source.read_range(start..end)?;
            let batch = checked_matrix(batch.matrix, end - start, self.dim)?;
            f(Matrix::new(&batch.values, batch.rows, batch.dims));
        }
        Ok(())
    }
}

/// `matrix`, if it holds `rows` vectors of dimension `dim`.
fn checked_matrix(
    matrix: IvfMatrix<f32>,
    rows: usize,
    dim: usize,
) -> crate::Result<IvfMatrix<f32>> {
    if matrix.dims != dim || matrix.rows != rows || matrix.values.len() != rows * dim {
        return Err(TantivyError::InvalidArgument(format!(
            "vector source returned {} rows of dimension {}, expected {rows} of dimension {dim}",
            matrix.rows, matrix.dims
        )));
    }
    Ok(matrix)
}

/// Reorders the `dim`-wide rows of `values` in place so that row `i` becomes the old row
/// `order[i]`, following each cycle of the permutation with one held row.
fn permute_rows(values: &mut [f32], dim: usize, order: &[u32]) {
    let mut placed = vec![false; order.len()];
    let mut held = vec![0.0; dim];
    for start in 0..order.len() {
        if placed[start] {
            continue;
        }
        held.copy_from_slice(&values[start * dim..(start + 1) * dim]);
        let mut slot = start;
        loop {
            placed[slot] = true;
            let from = order[slot] as usize;
            if from == start {
                values[slot * dim..(slot + 1) * dim].copy_from_slice(&held);
                break;
            }
            values.copy_within(from * dim..(from + 1) * dim, slot * dim);
            slot = from;
        }
    }
}

/// Trains centroids with hierarchical SuperKMeans and assigns each vector to its nearest one.
/// Cosine and Dot cluster angularly.
#[derive(Clone)]
pub struct SuperKMeansClusterer {
    config: HierarchicalSuperKMeansConfig,
    training_sample_ratio: f32,
    memory_limit: Option<usize>,
    boundary_epsilon: f32,
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
            .field("boundary_epsilon", &self.boundary_epsilon)
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
            boundary_epsilon: 0.0,
            assigner: Arc::new(Mutex::new(None)),
        }
    }

    /// At most `max_leaf_size` training vectors per clustering leaf.
    pub fn with_max_leaf_size(mut self, max_leaf_size: usize) -> Self {
        self.config.max_leaf_size = max_leaf_size;
        self
    }

    /// Caps the vectors held at once (the meso step's kept and streamed rows, fine-step groups,
    /// assign batches) at about `memory_limit` bytes.
    pub fn with_memory_limit(mut self, memory_limit: usize) -> Self {
        self.memory_limit = Some(memory_limit);
        self
    }

    /// Trains on `ratio` of the vectors, in `(0, 1]`.
    pub fn with_training_sample_ratio(mut self, ratio: f32) -> Self {
        self.training_sample_ratio = ratio;
        self
    }

    /// Adds one runner-up posting when `(d2 - d1) / abs(d1) <= epsilon`.
    /// L2 and cosine use squared distances; Dot uses negative inner products.
    /// Zero disables replication; a zero primary distance never replicates.
    pub fn with_boundary_epsilon(mut self, epsilon: f32) -> Self {
        self.boundary_epsilon = epsilon;
        self
    }

    /// Vectors the memory limit holds at once.
    fn budget_rows(&self, vector_options: &VectorOptions) -> usize {
        let budget_rows = match self.memory_limit {
            Some(byte_limit) => byte_limit.saturating_div(vector_options.bytes_per_vector()),
            None => DEFAULT_BUDGET_ROWS,
        };
        budget_rows.max(1)
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

        // The meso step keeps as many rows decoded as the budget allows, leaving room for one
        // streamed batch when the sample does not fit.
        let budget_rows = self.budget_rows(options);
        let (kept_rows, batch_rows) = if num_rows <= budget_rows {
            (num_rows, budget_rows)
        } else {
            let batch_rows = (budget_rows / 8).clamp(1, DEFAULT_BUDGET_ROWS);
            (budget_rows - batch_rows, batch_rows)
        };
        let mut dataset = SourceDataset {
            source: &mut *source,
            dim,
            kept_rows,
            batch_rows,
            kept: None,
        };
        let meso = kmeans.train_meso(&mut dataset, num_rows)?;
        let kept = dataset.kept.take();

        let largest_meso = (0..meso.n_meso())
            .map(|m| meso.meso_rows(m).len())
            .max()
            .unwrap_or(0);
        let mut centroids = Vec::new();
        match kept {
            // Every row is still decoded: regroup them in place by meso cluster, no reads.
            Some(mut kept) if kept.rows == num_rows && num_rows + largest_meso <= budget_rows => {
                permute_rows(&mut kept.values, dim, &meso.rows);
                for m in 0..meso.n_meso() {
                    if source.wants_cancel() {
                        return Err(TantivyError::Cancelled);
                    }
                    let rows = meso.offsets[m]..meso.offsets[m + 1];
                    let fine = kmeans.train_meso_cluster(
                        &kept.values[rows.start * dim..rows.end * dim],
                        rows.len(),
                    );
                    centroids.extend_from_slice(&fine.centroids);
                }
            }
            // Otherwise free the kept rows and load consecutive meso clusters as one group per
            // `read_rows`, so the source can fetch each group in a single ordered pass.
            kept => {
                drop(kept);
                let mut start = 0;
                while start < meso.n_meso() {
                    let end = fine_group_end(&meso.offsets, start, budget_rows);
                    let group_start = meso.offsets[start];
                    let group_rows = &meso.rows[group_start..meso.offsets[end]];
                    let IvfTrainingVectors::F32(group) = source.read_rows(group_rows)?;
                    let group = checked_matrix(group.matrix, group_rows.len(), dim)?;
                    for m in start..end {
                        if source.wants_cancel() {
                            return Err(TantivyError::Cancelled);
                        }
                        let rows = meso.offsets[m] - group_start..meso.offsets[m + 1] - group_start;
                        let fine = kmeans.train_meso_cluster(
                            &group.values[rows.start * dim..rows.end * dim],
                            rows.len(),
                        );
                        centroids.extend_from_slice(&fine.centroids);
                    }
                    start = end;
                }
            }
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
        for_each_batch(
            source,
            self.budget_rows(options),
            |IvfTrainingVectors::F32(batch)| {
                assignments.extend(kmeans.assign(
                    &batch.matrix.values,
                    centroids.values.as_slice(),
                    batch.matrix.rows,
                ));
            },
        )?;
        Ok(assignments)
    }

    fn assign_with_replicas(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<ClusterAssignment>> {
        if !self.boundary_epsilon.is_finite() || self.boundary_epsilon < 0.0 {
            return Err(TantivyError::InvalidArgument(
                "boundary_epsilon must be finite and non-negative".into(),
            ));
        }
        let IvfCentroids::F32(matrix) = centroids;
        let dim = options.dim();
        if matrix.rows == 0 || matrix.dims != dim || matrix.values.len() != matrix.rows * dim {
            return Err(TantivyError::InvalidArgument(
                "invalid assignment centroids".into(),
            ));
        }
        if self.boundary_epsilon == 0.0 || matrix.rows == 1 {
            return Ok(self
                .assign(options, source, centroids)?
                .into_iter()
                .map(|primary| ClusterAssignment {
                    primary,
                    secondary: None,
                })
                .collect());
        }

        let mut normalized_centroids;
        let centers = if options.metric() == Metric::Cosine {
            normalized_centroids = matrix.values.clone();
            normalize_rows_l2(&mut normalized_centroids, matrix.rows, dim);
            &normalized_centroids[..]
        } else {
            &matrix.values[..]
        };
        let mut centroid_norms = squared_norms(centers, matrix.rows, dim);
        let max_centroid_norm = centroid_norms.iter().copied().fold(0.0f32, f32::max).sqrt();
        if options.metric() == Metric::Dot {
            centroid_norms.fill(0.0);
        }
        let mut assignments = Vec::with_capacity(source.num_rows());
        let batch_rows = self.budget_rows(options);
        for start in (0..source.num_rows()).step_by(batch_rows) {
            if source.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            let end = (start + batch_rows).min(source.num_rows());
            let IvfTrainingVectors::F32(batch) = source.read_range(start..end)?;
            let mut batch = checked_matrix(batch.matrix, end - start, dim)?;
            if options.metric() == Metric::Cosine {
                normalize_rows_l2(&mut batch.values, batch.rows, dim);
            }
            let mut norms = squared_norms(&batch.values, batch.rows, dim);
            if options.metric() == Metric::Dot {
                // A query-constant shift keeps the helper's distances non-negative without
                // changing dot-product ranking or normalizing the indexed vectors.
                for norm in &mut norms {
                    *norm = 2.0 * norm.sqrt() * max_centroid_norm;
                }
            }
            let mut ids = vec![0; batch.rows * 2];
            let mut distances = vec![0.0; batch.rows * 2];
            find_k_nearest_neighbors(
                &batch.values,
                centers,
                batch.rows,
                matrix.rows,
                dim,
                &norms,
                &centroid_norms,
                2,
                &mut ids,
                &mut distances,
            );
            for (vector, pair) in batch.values.chunks_exact(dim).zip(ids.chunks_exact(2)) {
                let primary = pair[0];
                let mut assignment = ClusterAssignment {
                    primary,
                    secondary: None,
                };
                if pair[1] < matrix.rows as u32 && pair[1] != primary {
                    let distance = |cluster: u32| {
                        let center = &centers[cluster as usize * dim..][..dim];
                        match options.metric() {
                            Metric::L2 => l2_squared(vector, center),
                            Metric::Cosine => 2.0 * (1.0 - cosine(vector, center)).max(0.0),
                            Metric::Dot => -dot(vector, center),
                        }
                    };
                    let d1 = f64::from(distance(primary));
                    let d2 = f64::from(distance(pair[1]));
                    let valid_cosine =
                        options.metric() != Metric::Cosine || vector.iter().any(|&v| v != 0.0);
                    if valid_cosine
                        && d1.is_finite()
                        && d2.is_finite()
                        && d1 != 0.0
                        && (d2 - d1) / d1.abs() <= f64::from(self.boundary_epsilon)
                    {
                        assignment.secondary = Some(pair[1]);
                    }
                }
                assignments.push(assignment);
            }
        }
        Ok(assignments)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::ops::Range;

    use super::*;
    use crate::vector::clustering::IvfTrainingBatch;

    /// In-memory [`VectorSource`] that records every range read and counts `read_rows` calls.
    struct VecSource {
        values: Vec<f32>,
        dim: usize,
        ranges: Vec<Range<usize>>,
        row_reads: usize,
    }

    impl VecSource {
        fn new(values: Vec<f32>, dim: usize) -> Self {
            VecSource {
                values,
                dim,
                ranges: Vec::new(),
                row_reads: 0,
            }
        }

        fn batch(&self, rows: impl Iterator<Item = usize>) -> IvfTrainingVectors {
            let rows: Vec<usize> = rows.collect();
            let values = rows
                .iter()
                .flat_map(|&row| self.values[row * self.dim..][..self.dim].iter().copied())
                .collect();
            IvfTrainingVectors::F32(IvfTrainingBatch {
                doc_ids: rows.iter().map(|&row| row as u32).collect(),
                matrix: IvfMatrix {
                    values,
                    rows: rows.len(),
                    dims: self.dim,
                },
            })
        }
    }

    impl VectorSource for VecSource {
        fn num_rows(&self) -> usize {
            self.values.len() / self.dim
        }

        fn read_range(&mut self, range: Range<usize>) -> crate::Result<IvfTrainingVectors> {
            self.ranges.push(range.clone());
            Ok(self.batch(range))
        }

        fn read_rows(&mut self, rows: &[u32]) -> crate::Result<IvfTrainingVectors> {
            self.row_reads += 1;
            Ok(self.batch(rows.iter().map(|&row| row as usize)))
        }
    }

    /// Reports a cancel once it has been polled `polls_before_cancel` times, like a backend
    /// that receives the request part way through training.
    struct CancellingSource {
        inner: VecSource,
        polls_before_cancel: usize,
        polls: Cell<usize>,
    }

    impl VectorSource for CancellingSource {
        fn num_rows(&self) -> usize {
            self.inner.num_rows()
        }

        fn read_range(&mut self, range: Range<usize>) -> crate::Result<IvfTrainingVectors> {
            self.inner.read_range(range)
        }

        fn read_rows(&mut self, rows: &[u32]) -> crate::Result<IvfTrainingVectors> {
            self.inner.read_rows(rows)
        }

        fn wants_cancel(&self) -> bool {
            self.polls.set(self.polls.get() + 1);
            self.polls.get() > self.polls_before_cancel
        }
    }

    fn blobs(n: usize, dim: usize) -> Vec<f32> {
        superkmeans::make_blobs(n, dim, 25, false, 1.0, 10.0, 42)
    }

    fn train(clusterer: &SuperKMeansClusterer, source: &mut VecSource) -> IvfMatrix<f32> {
        let options = VectorOptions::new(source.dim, Metric::L2);
        let IvfCentroids::F32(centroids) = clusterer.train(&options, source).unwrap();
        centroids
    }

    #[test]
    fn boundary_assignments_choose_only_the_runner_up() -> crate::Result<()> {
        let options = VectorOptions::new(2, Metric::L2);
        let centers = IvfCentroids::F32(IvfMatrix {
            values: vec![0.0, 0.0, 10.0, 0.0, 5.0, 20.0],
            rows: 3,
            dims: 2,
        });
        let vectors = vec![4.9, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0];
        for (epsilon, replicas) in [
            (0.0, vec![None; 4]),
            (0.05, vec![None, None, Some(1), None]),
            (0.1, vec![Some(1), None, Some(1), None]),
        ] {
            let clusterer = SuperKMeansClusterer::new()
                .with_boundary_epsilon(epsilon)
                .with_memory_limit(16);
            let mut source = VecSource::new(vectors.clone(), 2);
            let assignments = clusterer.assign_with_replicas(&options, &mut source, &centers)?;
            assert_eq!(
                assignments.iter().map(|a| a.primary).collect::<Vec<_>>(),
                vec![0; 4]
            );
            assert_eq!(
                assignments.iter().map(|a| a.secondary).collect::<Vec<_>>(),
                replicas
            );
            assert!(source.ranges.iter().all(|range| range.len() <= 2));
        }
        Ok(())
    }

    #[test]
    fn boundary_assignment_handles_angular_metrics_and_zero_vectors() -> crate::Result<()> {
        let centers = IvfCentroids::F32(IvfMatrix {
            values: vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
            rows: 3,
            dims: 2,
        });
        for metric in [Metric::Cosine, Metric::Dot] {
            let options = VectorOptions::new(2, metric);
            for scale in [0.1, 1.0, 10.0] {
                let mut source = VecSource::new(vec![scale, scale * 0.9, 0.0, 0.0, scale, 0.0], 2);
                let assignments = SuperKMeansClusterer::new()
                    .with_boundary_epsilon(0.35)
                    .assign_with_replicas(&options, &mut source, &centers)?;
                assert_eq!(
                    assignments[0],
                    ClusterAssignment {
                        primary: 0,
                        secondary: Some(1)
                    }
                );
                assert!(assignments[1..].iter().all(|a| a.secondary.is_none()));
            }
        }
        Ok(())
    }

    #[test]
    fn boundary_assignment_rejects_invalid_epsilon_and_stops_on_cancel() {
        let options = VectorOptions::new(2, Metric::L2);
        let centers = IvfCentroids::F32(IvfMatrix {
            values: vec![0.0, 0.0, 1.0, 1.0],
            rows: 2,
            dims: 2,
        });
        for epsilon in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(SuperKMeansClusterer::new()
                .with_boundary_epsilon(epsilon)
                .assign_with_replicas(&options, &mut VecSource::new(vec![0.5, 0.5], 2), &centers)
                .is_err());
        }
        let mut source = CancellingSource {
            inner: VecSource::new(vec![0.5; 8], 2),
            polls_before_cancel: 1,
            polls: Cell::new(0),
        };
        let result = SuperKMeansClusterer::new()
            .with_boundary_epsilon(0.35)
            .with_memory_limit(8)
            .assign_with_replicas(&options, &mut source, &centers);
        assert!(matches!(result, Err(TantivyError::Cancelled)));
        assert_eq!(source.inner.ranges, vec![0..1]);
    }

    #[test]
    fn trains_and_assigns_from_the_source() {
        let (n, dim) = (5_000, 16);
        let mut source = VecSource::new(blobs(n, dim), dim);
        let clusterer = SuperKMeansClusterer::new().with_max_leaf_size(100);
        let options = VectorOptions::new(dim, Metric::L2);

        let centroids = train(&clusterer, &mut source);
        assert_eq!(centroids.dims, dim);
        assert_eq!(centroids.values.len(), centroids.rows * dim);
        assert!(centroids.rows >= n / 100, "{} centroids", centroids.rows);

        let assignments = clusterer
            .assign(&options, &mut source, &IvfCentroids::F32(centroids.clone()))
            .unwrap();
        assert_eq!(assignments.len(), n);
        assert!(assignments.iter().all(|&c| (c as usize) < centroids.rows));
    }

    /// The sample fits in memory, so without the polls nothing after the single read could
    /// notice a cancel. One poll per meso pass and one per fine cluster both stop training.
    #[test]
    fn a_cancel_stops_training_in_the_meso_and_fine_steps() {
        let (n, dim) = (5_000, 16);
        let clusterer = SuperKMeansClusterer::new().with_max_leaf_size(100);
        let options = VectorOptions::new(dim, Metric::L2);

        let mut source = CancellingSource {
            inner: VecSource::new(blobs(n, dim), dim),
            polls_before_cancel: usize::MAX,
            polls: Cell::new(0),
        };
        clusterer.train(&options, &mut source).unwrap();
        let polls_in_full_run = source.polls.get();
        assert!(polls_in_full_run > 2, "{polls_in_full_run} polls");

        for polls_before_cancel in [1, polls_in_full_run - 1] {
            let mut source = CancellingSource {
                inner: VecSource::new(blobs(n, dim), dim),
                polls_before_cancel,
                polls: Cell::new(0),
            };
            let err = clusterer.train(&options, &mut source).unwrap_err();
            assert!(matches!(err, TantivyError::Cancelled), "{err:?}");
            assert_eq!(source.polls.get(), polls_before_cancel + 1);
            assert_eq!(source.inner.ranges, vec![0..n]);
        }
    }

    #[test]
    fn rejects_an_empty_source() {
        let mut source = VecSource::new(Vec::new(), 4);
        let options = VectorOptions::new(4, Metric::L2);
        assert!(SuperKMeansClusterer::new()
            .train(&options, &mut source)
            .is_err());
        assert!(source.ranges.is_empty());
    }

    /// A sample that fits is read once: the meso passes replay the kept rows and the fine step
    /// regroups them in place.
    #[test]
    fn a_sample_that_fits_is_read_once() {
        let (n, dim) = (5_000, 16);
        let mut source = VecSource::new(blobs(n, dim), dim);
        train(
            &SuperKMeansClusterer::new().with_max_leaf_size(100),
            &mut source,
        );
        assert_eq!(source.ranges.len(), 1);
        assert_eq!(source.ranges[0], 0..n);
        assert_eq!(source.row_reads, 0);
    }

    /// A sample larger than the budget reads its kept prefix once and streams only the rest on
    /// every pass, then reads the fine step's meso clusters in groups.
    #[test]
    fn a_larger_sample_streams_only_the_rows_past_the_kept_prefix() {
        let (n, dim) = (5_000, 16);
        let mut source = VecSource::new(blobs(n, dim), dim);
        let options = VectorOptions::new(dim, Metric::L2);
        let clusterer = SuperKMeansClusterer::new()
            .with_max_leaf_size(100)
            .with_memory_limit(1_000 * options.bytes_per_vector());
        train(&clusterer, &mut source);

        let (kept, batch) = (875, 125);
        assert_eq!(source.ranges[0], 0..kept);
        assert!(source.ranges[1..]
            .iter()
            .all(|range| range.start >= kept && range.len() <= batch));
        let passes = source
            .ranges
            .iter()
            .filter(|range| range.start == kept)
            .count();
        assert!(passes > 1, "meso step should replay the streamed rows");
        assert!(source.row_reads > 1, "{} row reads", source.row_reads);
    }

    /// The memory limit changes how the sample is read, never the centroids: keeping every row
    /// and regrouping in place, keeping every row but reading the fine step back in groups, and
    /// streaming most rows all train the same model.
    #[test]
    fn the_memory_limit_does_not_change_the_centroids() {
        let (n, dim) = (5_000, 16);
        let options = VectorOptions::new(dim, Metric::L2);
        let clusterer = SuperKMeansClusterer::new().with_max_leaf_size(100);
        let reference = train(&clusterer, &mut VecSource::new(blobs(n, dim), dim));
        for limit_rows in [n, 1_000] {
            let mut source = VecSource::new(blobs(n, dim), dim);
            let limited = clusterer
                .clone()
                .with_memory_limit(limit_rows * options.bytes_per_vector());
            let centroids = train(&limited, &mut source);
            assert!(
                source.row_reads > 0,
                "{limit_rows} rows should read groups back"
            );
            assert_eq!(centroids.rows, reference.rows, "{limit_rows} rows");
            assert_eq!(centroids.values, reference.values, "{limit_rows} rows");
        }
    }

    #[test]
    fn permute_rows_moves_each_row_to_its_slot() {
        let mut values = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
        permute_rows(&mut values, 2, &[3, 0, 4, 1, 2]);
        assert_eq!(values, [3.0, 3.5, 0.0, 0.5, 4.0, 4.5, 1.0, 1.5, 2.0, 2.5]);
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
