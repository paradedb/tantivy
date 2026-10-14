//! The train step: a sample of a field's vectors streamed to the clusterer in source storage
//! order, the clusterer's centroids, and the router over them.

use std::sync::Arc;
use std::time::Instant;

use super::{FieldMerge, IvfBuildTimings};
use crate::indexer::segment_updater::CancelSentinel;
use crate::schema::VectorOptions;
use crate::vector::clustering::{
    decode_row_append, IvfCentroids, IvfClusterer, IvfMatrix, IvfMatrixView, IvfTrainingBatch,
    IvfTrainingVectors, IvfVectorBatch, IvfVectors, TrainingSource,
};
use crate::vector::router::{BuiltRouter, RouterKind};
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::write::RowAddress;
use crate::{DocId, TantivyError};

/// A field's trained centroids.
pub(super) struct Trained {
    pub(super) centroids: IvfCentroids,
    pub(super) num_centroids: usize,
    /// Alive documents with a vector, counted while sampling.
    pub(super) present_vectors: usize,
}

/// Samples the field's vectors in target-doc order and trains centroids on the sample, streamed
/// in source storage order; `None` when every vector-bearing document was deleted.
pub(super) fn train(
    merge: &FieldMerge,
    clusterer: &dyn IvfClusterer,
    source_rows: &[Option<RowAddress>],
    vector_count: usize,
    timings: &mut IvfBuildTimings,
) -> crate::Result<Option<Trained>> {
    let opts = merge.opts;
    let training_sample_size = {
        let ratio = f64::from(clusterer.training_sample_ratio()).clamp(f64::MIN_POSITIVE, 1.0);
        let target = ((vector_count as f64) * ratio).ceil() as usize;
        target.clamp(1, vector_count)
    };
    let training_sample_interval = (vector_count / training_sample_size).max(1);

    // The sample is picked in target-doc order, then scanned in source storage order so every
    // training pass reads sequentially.
    let mut training_rows: Vec<(RowAddress, DocId)> = Vec::with_capacity(training_sample_size);
    let mut present_vector_ord = 0usize;
    for (target_doc_id, source) in source_rows
        .iter()
        .enumerate()
        .filter_map(|(doc_id, row)| row.map(|row| (doc_id as DocId, row)))
    {
        if training_rows.len() < training_sample_size
            && present_vector_ord % training_sample_interval == 0
        {
            training_rows.push((source, target_doc_id));
        }
        present_vector_ord += 1;
    }
    debug_assert_eq!(source_rows.len(), merge.num_target_docs as usize);
    training_rows.sort_unstable();
    debug_assert!(
        if merge.ctx.readers.iter().any(|reader| reader.has_deletes()) {
            present_vector_ord <= vector_count
        } else {
            present_vector_ord == vector_count
        },
        "{present_vector_ord} alive docs with vectors vs {vector_count} reported by source count()"
    );
    if training_rows.is_empty() {
        return Ok(None);
    }

    let mut training_source =
        MergeTrainingSource::new(&merge.readers, training_rows, opts.dim(), merge.ctx.cancel);
    let train_start = Instant::now();
    let centroids = clusterer.train(opts, &mut training_source)?;
    timings.train = train_start.elapsed();
    timings.source_reads += training_source.reads;

    if merge.ctx.cancel.wants_cancel() {
        return Err(TantivyError::Cancelled);
    }

    let num_centroids = validate_centroids(opts, &centroids)?;
    Ok(Some(Trained {
        centroids,
        num_centroids,
        present_vectors: present_vector_ord,
    }))
}

/// Sampled merge rows, scanned in source storage order.
struct MergeTrainingSource<'a> {
    readers: &'a [Arc<VectorIndexReader>],
    rows: Vec<(RowAddress, DocId)>,
    dim: usize,
    cancel: &'a dyn CancelSentinel,
    reads: usize,
    batch_values: Vec<f32>,
    batch_doc_ids: Vec<DocId>,
}

impl<'a> MergeTrainingSource<'a> {
    fn new(
        readers: &'a [Arc<VectorIndexReader>],
        rows: Vec<(RowAddress, DocId)>,
        dim: usize,
        cancel: &'a dyn CancelSentinel,
    ) -> Self {
        MergeTrainingSource {
            readers,
            rows,
            dim,
            cancel,
            reads: 0,
            batch_values: Vec::new(),
            batch_doc_ids: Vec::new(),
        }
    }
}

fn decode_training_rows(
    readers: &[Arc<VectorIndexReader>],
    dim: usize,
    rows: impl Iterator<Item = (RowAddress, DocId)>,
    values: &mut Vec<f32>,
    doc_ids: &mut Vec<DocId>,
) -> crate::Result<()> {
    for (source, target_doc_id) in rows {
        let bytes = readers[source.segment_ord as usize].vector_bytes_for_row(source.row_id)?;
        decode_row_append::<f32>(&bytes, dim, values)?;
        doc_ids.push(target_doc_id);
    }
    Ok(())
}

impl TrainingSource for MergeTrainingSource<'_> {
    fn num_rows(&self) -> usize {
        self.rows.len()
    }

    fn for_each_batch(
        &mut self,
        batch_size: usize,
        f: &mut dyn FnMut(IvfVectors<'_>),
    ) -> crate::Result<()> {
        for batch in self.rows.chunks(batch_size.max(1)) {
            if self.cancel.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            self.batch_values.clear();
            self.batch_doc_ids.clear();
            decode_training_rows(
                self.readers,
                self.dim,
                batch.iter().copied(),
                &mut self.batch_values,
                &mut self.batch_doc_ids,
            )?;
            self.reads += batch.len();
            f(IvfVectors::F32(IvfVectorBatch {
                doc_ids: &self.batch_doc_ids,
                matrix: IvfMatrixView {
                    values: &self.batch_values,
                    rows: batch.len(),
                    dims: self.dim,
                },
            }));
        }
        Ok(())
    }

    fn read_rows(&mut self, rows: &[u32]) -> crate::Result<IvfTrainingVectors> {
        if self.cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
        if let Some(&row) = rows.iter().find(|&&row| row as usize >= self.rows.len()) {
            return Err(TantivyError::InvalidArgument(format!(
                "training row {row} is out of bounds for {} rows",
                self.rows.len()
            )));
        }
        let mut values = Vec::with_capacity(rows.len() * self.dim);
        let mut doc_ids = Vec::with_capacity(rows.len());
        decode_training_rows(
            self.readers,
            self.dim,
            rows.iter().map(|&row| self.rows[row as usize]),
            &mut values,
            &mut doc_ids,
        )?;
        self.reads += rows.len();
        Ok(IvfTrainingVectors::F32(IvfTrainingBatch {
            doc_ids,
            matrix: IvfMatrix {
                values,
                rows: rows.len(),
                dims: self.dim,
            },
        }))
    }
}

/// The number of centroids a clusterer produced, checked against `opts`'s dimension.
fn validate_centroids(opts: &VectorOptions, centroids: &IvfCentroids) -> crate::Result<usize> {
    let IvfCentroids::F32(centroid_matrix) = centroids;
    if centroid_matrix.dims != opts.dim() {
        return Err(TantivyError::InvalidArgument(format!(
            "IvfClusterer produced centroids with {} dimensions, expected {}",
            centroid_matrix.dims,
            opts.dim()
        )));
    }
    if centroid_matrix.values.len() != centroid_matrix.rows * centroid_matrix.dims {
        return Err(TantivyError::InvalidArgument(format!(
            "IvfClusterer produced {} centroid values for {} rows x {} dimensions",
            centroid_matrix.values.len(),
            centroid_matrix.rows,
            centroid_matrix.dims
        )));
    }
    if centroid_matrix.rows == 0 {
        return Err(TantivyError::InvalidArgument(
            "IvfClusterer produced zero centroids".to_string(),
        ));
    }
    Ok(centroid_matrix.rows)
}

/// Builds `router` over `centroids`, which it may reorder but not reshape.
pub(super) fn build_router(
    router: RouterKind,
    opts: &VectorOptions,
    centroids: &mut IvfCentroids,
) -> crate::Result<BuiltRouter> {
    let IvfCentroids::F32(matrix) = &*centroids;
    let shape = (matrix.rows, matrix.dims, matrix.values.len());
    let router = router.build(opts, centroids)?;
    let IvfCentroids::F32(matrix) = &*centroids;
    if (matrix.rows, matrix.dims, matrix.values.len()) != shape {
        return Err(TantivyError::InvalidArgument(
            "Router changed the centroid matrix shape while building".to_string(),
        ));
    }
    Ok(router)
}
