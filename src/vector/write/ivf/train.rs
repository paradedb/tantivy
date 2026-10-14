//! The train step: a target-ordered sample of a field's vectors, the clusterer's centroids, and
//! the router over them.

use std::time::Instant;

use super::{FieldMerge, IvfBuildTimings};
use crate::schema::VectorOptions;
use crate::vector::clustering::{
    decode_row_append, IvfCentroids, IvfClusterer, IvfMatrix, IvfMergeSettings, IvfTrainingBatch,
    IvfTrainingVectors,
};
use crate::vector::router::{BuiltRouter, RouterKind};
use crate::vector::write::RowAddress;
use crate::{DocId, TantivyError};

/// A field's trained centroids.
pub(super) struct Trained {
    pub(super) centroids: IvfCentroids,
    pub(super) num_centroids: usize,
    /// Alive documents with a vector, counted while sampling.
    pub(super) present_vectors: usize,
}

/// Samples the field's vectors in target-doc order and trains centroids on the sample; `None`
/// when every vector-bearing document was deleted.
pub(super) fn train(
    merge: &FieldMerge,
    clusterer: &dyn IvfClusterer,
    settings: &IvfMergeSettings,
    source_rows: &[Option<RowAddress>],
    vector_count: usize,
    timings: &mut IvfBuildTimings,
) -> crate::Result<Option<Trained>> {
    let opts = merge.opts;
    let training_sample_size = {
        let ratio = f64::from(settings.training_sample_ratio).clamp(f64::MIN_POSITIVE, 1.0);
        let target = ((vector_count as f64) * ratio).ceil() as usize;
        target.clamp(1, vector_count)
    };
    let training_sample_interval = (vector_count / training_sample_size).max(1);
    let dim = opts.dim();

    // Each sampled row paired with its position in target-doc order.
    let mut training_doc_ids = Vec::with_capacity(training_sample_size);
    let mut training_sources: Vec<(RowAddress, usize)> = Vec::with_capacity(training_sample_size);
    let mut present_vector_ord = 0usize;
    for (target_doc_id, source) in source_rows
        .iter()
        .enumerate()
        .filter_map(|(doc_id, row)| row.map(|row| (doc_id as DocId, row)))
    {
        if training_sources.len() < training_sample_size
            && present_vector_ord % training_sample_interval == 0
        {
            training_sources.push((source, training_doc_ids.len()));
            training_doc_ids.push(target_doc_id);
        }
        present_vector_ord += 1;
    }
    debug_assert_eq!(source_rows.len(), merge.num_target_docs as usize);

    // Read in source storage order, but keep the training matrix in target-doc
    // order so the trainer's input does not depend on source clustering.
    training_sources.sort_unstable();
    let mut training_values = vec![0.0f32; training_sources.len() * dim];
    let mut decode_buffer = Vec::with_capacity(dim);
    for (source, sample_idx) in training_sources {
        timings.source_reads += 1;
        let bytes =
            merge.readers[source.segment_ord as usize].vector_bytes_for_row(source.row_id)?;
        decode_buffer.clear();
        decode_row_append::<f32>(&bytes, dim, &mut decode_buffer)?;
        training_values[sample_idx * dim..][..dim].copy_from_slice(&decode_buffer);
    }
    debug_assert!(
        if merge.ctx.readers.iter().any(|reader| reader.has_deletes()) {
            present_vector_ord <= vector_count
        } else {
            present_vector_ord == vector_count
        },
        "{present_vector_ord} alive docs with vectors vs {vector_count} reported by source count()"
    );
    if training_doc_ids.is_empty() {
        return Ok(None);
    }

    let training_rows = training_doc_ids.len();
    let training_vectors = IvfTrainingVectors::F32(IvfTrainingBatch {
        doc_ids: training_doc_ids,
        matrix: IvfMatrix {
            values: training_values,
            rows: training_rows,
            dims: opts.dim(),
        },
    });
    let train_start = Instant::now();
    let centroids = clusterer.train(opts, training_vectors)?;

    timings.train = train_start.elapsed();

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
