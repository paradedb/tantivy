//! The built-in [`IvfClusterer`]: hierarchical SuperKMeans.

use std::fmt;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use superkmeans::{
    Dataset, HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, Matrix, SoarAssigner,
};

use crate::schema::VectorOptions;
use crate::vector::clustering::soar::SOAR_CANDIDATES;
use crate::vector::clustering::{
    for_each_batch, IvfAssignmentCandidate, IvfCentroids, IvfClusterer, IvfMatrix,
    IvfTrainingVectors, VectorSource,
};
use crate::vector::distance::dot;
use crate::vector::router::ASSIGNMENT_CANDIDATES;
use crate::{Result, TantivyError};

/// Default [`SuperKMeansClusterer::with_target_leaf_size`].
pub const DEFAULT_TARGET_LEAF_SIZE: usize = 200;
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

/// Trains hierarchical SuperKMeans centroids, with exhaustive or tree-shortlisted assignment.
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
    /// Defaults: [`DEFAULT_TARGET_LEAF_SIZE`], [`DEFAULT_TRAINING_SAMPLE_RATIO`], no memory limit.
    pub fn new() -> Self {
        let mut config = HierarchicalSuperKMeansConfig::default();
        config.base.suppress_warnings = true;
        config.target_leaf_size = DEFAULT_TARGET_LEAF_SIZE;
        Self {
            config,
            training_sample_ratio: DEFAULT_TRAINING_SAMPLE_RATIO as f32,
            memory_limit: None,
            assigner: Arc::new(Mutex::new(None)),
        }
    }

    /// Target vectors per clustering leaf on the full dataset. Training scales this by
    /// [`Self::with_training_sample_ratio`] so sampled leaves stay proportional and final
    /// clusters approximate this size. Splits aim for the scaled size; actual leaves may be
    /// somewhat larger (about √2× the target).
    pub fn with_target_leaf_size(mut self, target_leaf_size: usize) -> Self {
        self.config.target_leaf_size = target_leaf_size;
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

    /// Leaf size passed to SuperKMeans for the training sample.
    ///
    /// For example, with defaults (`target_leaf_size=200`, `training_sample_ratio=0.32`)
    /// this returns `64`.
    fn training_target_leaf_size(&self) -> usize {
        ((self.config.target_leaf_size as f64) * f64::from(self.training_sample_ratio))
            .round()
            .max(1.0) as usize
    }

    /// Config for HierarchicalSuperKMeans, with leaf size scaled to the training sample.
    fn training_config(&self, angular: bool) -> HierarchicalSuperKMeansConfig {
        let mut config = self.config.clone();
        config.base.angular = angular;
        config.target_leaf_size = self.training_target_leaf_size();
        config
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
                let kmeans = Arc::new(HierarchicalSuperKMeans::with_config(
                    dim,
                    self.training_config(angular),
                ));
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
        let kmeans = HierarchicalSuperKMeans::with_config(
            dim,
            self.training_config(options.metric().clusters_by_angle()),
        );

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
        let kmeans = self.assigner(options.dim(), options.metric().clusters_by_angle());
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

    /// The nearest centroid and the SOAR secondary among the [`SOAR_CANDIDATES`] centroids
    /// nearest to each vector after it, from one assigner prepared for the whole source; `None`
    /// with a single centroid. Batches are as large as the memory budget allows.
    fn assign_pairs(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
        centroids: &IvfCentroids,
        lambda: f32,
    ) -> crate::Result<Vec<(u32, Option<u32>)>> {
        let mut pairs = Vec::with_capacity(source.num_rows());
        let IvfCentroids::F32(centroids) = centroids;
        let soar = SoarAssigner::with_candidates(
            Matrix::new(&centroids.values, centroids.rows, centroids.dims),
            lambda,
            2,
            SOAR_CANDIDATES,
        );
        let batch_rows = self.budget_rows(options);
        for_each_batch(source, batch_rows, |IvfTrainingVectors::F32(batch)| {
            let (primary, secondary) = soar.assign(&batch.matrix.values, batch.matrix.rows);
            pairs.extend(
                primary
                    .into_iter()
                    .zip(secondary)
                    .map(|(first, second)| (first, (second != u32::MAX).then_some(second))),
            );
        })?;
        Ok(pairs)
    }

    fn assign_routed(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
        centroids: &IvfCentroids,
        lambda: Option<f32>,
        candidates: &(dyn Fn(&[f32]) -> Result<Vec<IvfAssignmentCandidate>> + Sync),
    ) -> Result<Vec<(u32, Option<u32>)>> {
        let IvfCentroids::F32(matrix) = centroids;
        if matrix.rows <= ASSIGNMENT_CANDIDATES {
            return match lambda {
                Some(lambda) => self.assign_pairs(options, source, centroids, lambda),
                None => Ok(self
                    .assign(options, source, centroids)?
                    .into_iter()
                    .map(|cluster| (cluster, None))
                    .collect()),
            };
        }
        let mut pairs = Vec::with_capacity(source.num_rows());
        let batch_rows = self.budget_rows(options);
        for start in (0..source.num_rows()).step_by(batch_rows) {
            if source.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            let end = start.saturating_add(batch_rows).min(source.num_rows());
            let IvfTrainingVectors::F32(batch) = source.read_range(start..end)?;
            let batch = checked_matrix(batch.matrix, end - start, options.dim())?;
            let assigned: Result<Vec<_>> = batch
                .values
                .par_chunks_exact(options.dim())
                .map_init(
                    || vec![0.0; options.dim()],
                    |residual, vector| {
                        let mut candidates = candidates(vector)?;
                        assign_candidates(vector, matrix, &mut candidates, lambda, residual)
                    },
                )
                .collect();
            pairs.extend(assigned?);
        }
        Ok(pairs)
    }
}

fn assign_candidates(
    vector: &[f32],
    centroids: &IvfMatrix<f32>,
    candidates: &mut [IvfAssignmentCandidate],
    lambda: Option<f32>,
    residual: &mut [f32],
) -> Result<(u32, Option<u32>)> {
    if candidates.is_empty()
        || candidates.iter().any(|candidate| {
            candidate.centroid as usize >= centroids.rows
                || !candidate.distance.is_finite()
                || candidate.distance < 0.0
        })
    {
        return Err(TantivyError::InvalidArgument(
            "invalid centroid assignment shortlist".into(),
        ));
    }
    candidates.sort_unstable_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then(a.centroid.cmp(&b.centroid))
    });
    let primary = candidates[0];
    let Some(lambda) = lambda else {
        return Ok((primary.centroid, None));
    };
    let centroid = |id: u32| &centroids.values[id as usize * centroids.dims..][..centroids.dims];
    let norm = primary.distance.sqrt();
    if norm > 0.0 && norm.is_finite() {
        let inv_norm = norm.recip();
        for ((value, x), c) in residual
            .iter_mut()
            .zip(vector)
            .zip(centroid(primary.centroid))
        {
            *value = (x - c) * inv_norm;
        }
    } else {
        residual.fill(0.0);
    }
    let along = dot(vector, residual);
    let secondary = candidates
        .iter()
        .filter(|candidate| candidate.centroid != primary.centroid)
        .take(SOAR_CANDIDATES)
        .map(|candidate| {
            let gap = along - dot(residual, centroid(candidate.centroid));
            (candidate.distance + lambda * gap * gap, candidate.centroid)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
        .map(|(_, id)| id);
    Ok((primary.centroid, secondary))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::env;
    use std::fs::File;
    use std::io::Read;
    use std::ops::Range;
    use std::path::PathBuf;
    use std::time::Instant;

    use rayon::current_num_threads;
    use serde_json::json;

    use super::*;
    use crate::schema::Metric;
    use crate::vector::clustering::IvfTrainingBatch;
    use crate::vector::distance::l2_squared;
    use crate::vector::router::{RouterKind, ASSIGNMENT_BEAM_FRACTION, ASSIGNMENT_RERANK_FACTOR};

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

    #[test]
    #[ignore = "requires IVF_ASSIGNMENT_FIXTURES containing cached Cohere f32 rows and centroids"]
    fn profile_routed_assignment() {
        const DIM: usize = 1024;
        const ROWS: usize = 8192;
        const ITERATIONS: usize = 3;
        const MEMORY_BYTES: usize = 256 * 1024 * 1024;
        const WARMUP_ROWS: usize = 256;
        const MAX_FIXTURE_CENTROIDS: usize = 5000;
        let fixtures = PathBuf::from(env::var("IVF_ASSIGNMENT_FIXTURES").unwrap());
        let read = |name: &str, elements: usize| {
            let mut bytes = Vec::new();
            File::open(fixtures.join(name))
                .unwrap()
                .take((elements * size_of::<f32>()) as u64)
                .read_to_end(&mut bytes)
                .unwrap();
            bytes
                .chunks_exact(size_of::<f32>())
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let vectors = read("segment.f32", ROWS * DIM);
        let mut values = Vec::new();
        for segment in [7, 6, 5] {
            values.extend(read(
                &format!("centroids-{segment}.f32"),
                MAX_FIXTURE_CENTROIDS * DIM,
            ));
        }
        let candidates: usize = env::var("IVF_ASSIGNMENT_CANDIDATES")
            .map(|s| s.parse().unwrap())
            .unwrap_or(ASSIGNMENT_CANDIDATES);
        let beam: f32 = env::var("IVF_ASSIGNMENT_BEAM")
            .map(|s| s.parse().unwrap())
            .unwrap_or(ASSIGNMENT_BEAM_FRACTION);
        let rerank: f32 = env::var("IVF_ASSIGNMENT_RERANK")
            .map(|s| s.parse().unwrap())
            .unwrap_or(ASSIGNMENT_RERANK_FACTOR);
        let options = VectorOptions::new(DIM, Metric::Cosine);
        let clusterer = SuperKMeansClusterer::new().with_memory_limit(MEMORY_BYTES);
        for count in [4096, 9000, 14000] {
            let mut centroids = IvfCentroids::F32(IvfMatrix {
                values: values[..count * DIM].to_vec(),
                rows: count,
                dims: DIM,
            });
            let router = RouterKind::QuantizedStacked
                .build(&options, &mut centroids)
                .unwrap();
            let IvfCentroids::F32(matrix) = &centroids;
            let mut source = VecSource::new(vectors.clone(), DIM);
            let mut warmup = VecSource::new(vectors[..WARMUP_ROWS * DIM].to_vec(), DIM);
            clusterer
                .assign_pairs(&options, &mut warmup, &centroids, 2.0)
                .unwrap();
            clusterer
                .assign_routed(&options, &mut warmup, &centroids, Some(2.0), &|x| {
                    router.assignment_candidates_with(x, candidates, beam, rerank)
                })
                .unwrap();
            for iteration in 0..ITERATIONS {
                let exact = |source: &mut VecSource| {
                    let start = Instant::now();
                    let pairs = clusterer
                        .assign_pairs(&options, source, &centroids, 2.0)
                        .unwrap();
                    (start.elapsed().as_secs_f64(), pairs)
                };
                let routed = |source: &mut VecSource| {
                    let start = Instant::now();
                    let pairs = clusterer
                        .assign_routed(&options, source, &centroids, Some(2.0), &|x| {
                            router.assignment_candidates_with(x, candidates, beam, rerank)
                        })
                        .unwrap();
                    (start.elapsed().as_secs_f64(), pairs)
                };
                let (exact, routed) = if iteration % 2 == 0 {
                    (exact(&mut source), routed(&mut source))
                } else {
                    let routed = routed(&mut source);
                    (exact(&mut source), routed)
                };
                let (mut primary_agree, mut secondary_agree) = (0, 0);
                let (mut exact_loss, mut routed_loss) = (0.0_f64, 0.0_f64);
                let (mut exact_soar, mut routed_soar) = (0.0_f64, 0.0_f64);
                for ((vector, expected), actual) in
                    vectors.chunks_exact(DIM).zip(&exact.1).zip(&routed.1)
                {
                    primary_agree += usize::from(expected.0 == actual.0);
                    secondary_agree += usize::from(expected.1 == actual.1);
                    let score = |&(primary, secondary): &(u32, Option<u32>)| {
                        let centroid = |id: u32| &matrix.values[id as usize * DIM..][..DIM];
                        let residual: Vec<_> = vector
                            .iter()
                            .zip(centroid(primary))
                            .map(|(x, c)| x - c)
                            .collect();
                        let squared = dot(&residual, &residual);
                        let other = centroid(secondary.unwrap());
                        let gap = dot(vector, &residual) - dot(other, &residual);
                        (
                            f64::from(squared),
                            f64::from(
                                l2_squared(vector, other)
                                    + if squared > 0.0 {
                                        2.0 * gap * gap / squared
                                    } else {
                                        0.0
                                    },
                            ),
                        )
                    };
                    let (loss, cost) = score(expected);
                    exact_loss += loss;
                    exact_soar += cost;
                    let (loss, cost) = score(actual);
                    routed_loss += loss;
                    routed_soar += cost;
                }
                println!(
                    "{}",
                    json!({"centroids":count,"rows":ROWS,"dimension":DIM,"iteration":iteration,
                    "candidates":candidates,"beam":beam,"rerank":rerank,"threads":current_num_threads(),"exact_seconds":exact.0,"routed_seconds":routed.0,
                    "speedup":exact.0/routed.0,"primary_agreement":primary_agree as f64/ROWS as f64,
                    "secondary_agreement":secondary_agree as f64/ROWS as f64,
                    "primary_loss_ratio":routed_loss/exact_loss,"soar_cost_ratio":routed_soar/exact_soar})
                );
            }
        }
    }

    #[test]
    fn routed_assignment_matches_exhaustive_with_all_candidates_and_bounds_batches() {
        let (n, dim, batch_rows) = (3_000, 16, 17);
        let options = VectorOptions::new(dim, Metric::L2);
        let mut source = VecSource::new(blobs(n, dim), dim);
        let clusterer = SuperKMeansClusterer::new()
            .with_target_leaf_size(20)
            .with_memory_limit(batch_rows * options.bytes_per_vector());
        let centroids = clusterer.train(&options, &mut source).unwrap();
        let IvfCentroids::F32(matrix) = &centroids;
        assert!(matrix.rows > ASSIGNMENT_CANDIDATES);
        for (i, value) in source.values.iter_mut().enumerate() {
            *value += ((i * 7 % 31) as f32 - 15.0) * 0.03;
        }
        let exact = clusterer
            .assign_pairs(&options, &mut source, &centroids, 2.0)
            .unwrap();
        source.ranges.clear();
        let routed = clusterer
            .assign_routed(&options, &mut source, &centroids, Some(2.0), &|vector| {
                Ok((0..matrix.rows as u32)
                    .rev()
                    .map(|id| IvfAssignmentCandidate {
                        centroid: id,
                        distance: l2_squared(vector, &matrix.values[id as usize * dim..][..dim]),
                    })
                    .collect())
            })
            .unwrap();
        for (row, (expected, actual)) in exact.iter().zip(&routed).enumerate() {
            assert_eq!(expected, actual, "row {row}");
        }
        assert_eq!(source.ranges[0], 0..batch_rows);
        assert!(source.ranges.iter().all(|r| r.len() <= batch_rows));
    }

    #[test]
    fn tree_assignment_handles_permutations_metrics_and_replication() {
        let (n, dim) = (3_000, 16);
        for metric in Metric::ALL {
            let options = VectorOptions::new(dim, metric);
            for kind in [RouterKind::Stacked, RouterKind::QuantizedStacked] {
                let clusterer = SuperKMeansClusterer::new().with_target_leaf_size(20);
                let mut source = VecSource::new(blobs(n, dim), dim);
                let mut centroids = clusterer.train(&options, &mut source).unwrap();
                let router = kind.build(&options, &mut centroids).unwrap();
                let IvfCentroids::F32(matrix) = &centroids;
                for (i, value) in source.values.iter_mut().enumerate() {
                    *value += ((i * 7 % 31) as f32 - 15.0) * 0.03;
                }
                let exact = clusterer
                    .assign_pairs(&options, &mut source, &centroids, 2.0)
                    .unwrap();
                let routed = clusterer
                    .assign_routed(&options, &mut source, &centroids, Some(2.0), &|v| {
                        router.assignment_candidates_with(v, matrix.rows, 1.0, 1.0)
                    })
                    .unwrap();
                const RELATIVE_COST_TOLERANCE: f64 = 1e-6;
                for (row, (expected, actual)) in exact.iter().zip(&routed).enumerate() {
                    let vector = &source.values[row * dim..][..dim];
                    let centroid = |id: u32| &matrix.values[id as usize * dim..][..dim];
                    let costs = |&(primary, secondary): &(u32, Option<u32>)| {
                        let residual: Vec<_> = vector
                            .iter()
                            .zip(centroid(primary))
                            .map(|(&x, &c)| f64::from(x) - f64::from(c))
                            .collect();
                        let distance: f64 = residual.iter().map(|r| r * r).sum();
                        let secondary = centroid(secondary.unwrap());
                        let secondary_distance: f64 = vector
                            .iter()
                            .zip(secondary)
                            .map(|(&x, &c)| (f64::from(x) - f64::from(c)).powi(2))
                            .sum();
                        let gap: f64 = vector
                            .iter()
                            .zip(secondary)
                            .zip(&residual)
                            .map(|((&x, &c), r)| (f64::from(x) - f64::from(c)) * r)
                            .sum();
                        [
                            distance,
                            secondary_distance
                                + if distance > 0.0 {
                                    2.0 * gap * gap / distance
                                } else {
                                    0.0
                                },
                        ]
                    };
                    // GEMM and direct distances can choose different centroids at near ties.
                    for (expected_cost, actual_cost) in
                        costs(expected).into_iter().zip(costs(actual))
                    {
                        assert!(
                            (expected_cost - actual_cost).abs()
                                <= RELATIVE_COST_TOLERANCE * expected_cost.abs().max(1.0),
                            "{metric:?} {kind:?} row {row}: {expected:?} ({expected_cost}) vs \
                             {actual:?} ({actual_cost})"
                        );
                    }
                }
                let primary = clusterer
                    .assign_routed(&options, &mut source, &centroids, None, &|v| {
                        router.assignment_candidates(v)
                    })
                    .unwrap();
                assert!(primary
                    .iter()
                    .all(|&(id, second)| (id as usize) < matrix.rows && second.is_none()));
            }
        }
    }

    #[test]
    fn shortlisted_soar_handles_zero_residual_ties_and_invalid_ids() {
        let centroids = IvfMatrix {
            values: vec![0.0, 1.0, -1.0],
            rows: 3,
            dims: 1,
        };
        let assign = |ids: &[u32]| {
            let mut candidates: Vec<_> = ids
                .iter()
                .map(|&id| IvfAssignmentCandidate {
                    centroid: id,
                    distance: if id == 0 { 0.0 } else { 1.0 },
                })
                .collect();
            assign_candidates(&[0.0], &centroids, &mut candidates, Some(2.0), &mut [0.0])
        };
        assert_eq!(assign(&[2, 0, 1]).unwrap(), (0, Some(1)));
        assert_eq!(assign(&[0]).unwrap(), (0, None));
        assert!(assign(&[]).is_err());
        assert!(assign(&[3]).is_err());
    }

    #[test]
    fn routed_assignment_preserves_small_fallback_and_cancellation() {
        let options = VectorOptions::new(1, Metric::L2);
        let clusterer =
            SuperKMeansClusterer::new().with_memory_limit(3 * options.bytes_per_vector());
        let mut source = VecSource::new(vec![0.0; 10], 1);
        let one = IvfCentroids::F32(IvfMatrix {
            values: vec![0.0],
            rows: 1,
            dims: 1,
        });
        let pairs = clusterer
            .assign_routed(&options, &mut source, &one, Some(2.0), &|_| {
                panic!("small centroid sets remain exhaustive")
            })
            .unwrap();
        assert_eq!(pairs, vec![(0, None); 10]);

        let count = ASSIGNMENT_CANDIDATES + 1;
        let centroids = IvfCentroids::F32(IvfMatrix {
            values: (0..count).map(|i| i as f32).collect(),
            rows: count,
            dims: 1,
        });
        let mut cancelling = CancellingSource {
            inner: source,
            polls_before_cancel: 1,
            polls: Cell::new(0),
        };
        cancelling.inner.ranges.clear();
        let error = clusterer
            .assign_routed(&options, &mut cancelling, &centroids, Some(2.0), &|_| {
                Ok(vec![
                    IvfAssignmentCandidate {
                        centroid: 0,
                        distance: 0.0,
                    },
                    IvfAssignmentCandidate {
                        centroid: 1,
                        distance: 1.0,
                    },
                ])
            })
            .unwrap_err();
        assert!(matches!(error, TantivyError::Cancelled));
        assert_eq!(cancelling.inner.ranges.len(), 1);
        assert_eq!(cancelling.inner.ranges[0], 0..3);
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
    fn trains_and_assigns_from_the_source() {
        let (n, dim) = (5_000, 16);
        let mut source = VecSource::new(blobs(n, dim), dim);
        let clusterer = SuperKMeansClusterer::new().with_target_leaf_size(100);
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
        let clusterer = SuperKMeansClusterer::new().with_target_leaf_size(100);
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

    /// Every vector gets a secondary distinct from its primary; with a large `lambda` the
    /// secondary's residual lies further off the primary residual than the second-nearest
    /// centroid's does for most vectors; one centroid gives no secondary.
    #[test]
    fn assigns_pairs_off_the_primary_residual() {
        let (n, dim) = (3_000, 16);
        let mut source = VecSource::new(blobs(n, dim), dim);
        let clusterer = SuperKMeansClusterer::new().with_target_leaf_size(100);
        let options = VectorOptions::new(dim, Metric::L2);
        let centroids = train(&clusterer, &mut source);
        let trained = IvfCentroids::F32(centroids.clone());
        let nearest = clusterer.assign(&options, &mut source, &trained).unwrap();
        let pairs = clusterer
            .assign_pairs(&options, &mut source, &trained, 50.0)
            .unwrap();
        assert_eq!(pairs.len(), n);
        let projection = |x: &[f32], primary: u32, cluster: u32| -> f32 {
            let c1 = &centroids.values[primary as usize * dim..][..dim];
            let c = &centroids.values[cluster as usize * dim..][..dim];
            let residual: Vec<f32> = x.iter().zip(c1).map(|(a, b)| a - b).collect();
            let norm = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
            x.iter()
                .zip(c)
                .zip(&residual)
                .map(|((a, b), r)| (a - b) * r / norm)
                .sum::<f32>()
                .abs()
        };
        let mut off_residual = 0;
        for (row, &(primary, second)) in pairs.iter().enumerate() {
            assert_eq!(primary, nearest[row]);
            let second = second.expect("more than one centroid");
            assert_ne!(second, primary, "row {row}");
            let x = &source.values[row * dim..(row + 1) * dim];
            let mut by_distance: Vec<(f32, u32)> = (0..centroids.rows as u32)
                .map(|c| {
                    let centroid = &centroids.values[c as usize * dim..][..dim];
                    (
                        x.iter().zip(centroid).map(|(a, b)| (a - b) * (a - b)).sum(),
                        c,
                    )
                })
                .collect();
            by_distance.sort_by(|a, b| a.0.total_cmp(&b.0));
            let second_nearest = by_distance[1].1;
            if projection(x, primary, second) <= projection(x, primary, second_nearest) {
                off_residual += 1;
            }
        }
        assert!(
            off_residual * 10 >= n * 9,
            "{off_residual} of {n} secondaries lie at least as far off the residual as nearest-2"
        );

        let one = IvfCentroids::F32(IvfMatrix {
            values: centroids.values[..dim].to_vec(),
            rows: 1,
            dims: dim,
        });
        let pairs = clusterer
            .assign_pairs(&options, &mut source, &one, 1.0)
            .unwrap();
        assert!(pairs
            .iter()
            .all(|&(primary, second)| primary == 0 && second.is_none()));
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
            &SuperKMeansClusterer::new().with_target_leaf_size(100),
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
            .with_target_leaf_size(100)
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
        let clusterer = SuperKMeansClusterer::new().with_target_leaf_size(100);
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

    /// Stored settings are independent; training leaf size is scaled by the sample ratio.
    #[test]
    fn settings_default_and_scale_training_leaf_size() {
        let clusterer = SuperKMeansClusterer::new();
        assert_eq!(clusterer.training_sample_ratio(), 0.32);
        assert_eq!(clusterer.config.target_leaf_size, DEFAULT_TARGET_LEAF_SIZE);
        assert_eq!(clusterer.training_target_leaf_size(), 64);

        let clusterer = SuperKMeansClusterer::new()
            .with_target_leaf_size(20)
            .with_training_sample_ratio(0.25);
        assert_eq!(clusterer.training_sample_ratio(), 0.25);
        assert_eq!(clusterer.config.target_leaf_size, 20);
        assert_eq!(clusterer.training_target_leaf_size(), 5);
        let clusterer = clusterer.with_training_sample_ratio(1.0);
        assert_eq!(clusterer.training_sample_ratio(), 1.0);
        assert_eq!(clusterer.config.target_leaf_size, 20);
        assert_eq!(clusterer.training_target_leaf_size(), 20);
    }
}
