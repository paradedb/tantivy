//! The assign step: every vector to its nearest centroid, and the clusters' row ranges.

use std::time::Instant;

use itertools::Itertools;

use super::{AssignedVector, FieldMerge, IvfBuildTimings};
use crate::vector::clustering::{
    decode_stored_row_append, IvfCentroids, IvfClusterer, IvfMatrixView, IvfMergeSettings,
    IvfVectorBatch, IvfVectors,
};
use crate::vector::write::RowAddress;
use crate::{DocId, TantivyError};

/// Assigns each of the field's vectors to one of `centroids`, reading the sources in storage
/// order and batching by the clusterer's batch size.
pub(super) fn assign(
    merge: &FieldMerge,
    clusterer: &dyn IvfClusterer,
    settings: &IvfMergeSettings,
    source_rows: Vec<Option<RowAddress>>,
    centroids: &IvfCentroids,
    vector_count: usize,
    timings: &mut IvfBuildTimings,
) -> crate::Result<Vec<AssignedVector>> {
    let opts = merge.opts;
    let dim = opts.dim();
    let IvfCentroids::F32(centroid_matrix) = centroids;
    let num_centroids = centroid_matrix.rows;
    let mut assigned_vectors = Vec::with_capacity(vector_count);
    // Assignment is per vector, so batches can follow source storage order;
    // `assigned_vectors` is re-sorted by (cluster, target doc) afterwards.
    let assign_sources: Vec<(RowAddress, DocId)> = source_rows
        .into_iter()
        .enumerate()
        .filter_map(|(doc_id, row)| row.map(|row| (row, doc_id as DocId)))
        .sorted()
        .collect();

    let batch_capacity = settings.assign_batch_size.min(assign_sources.len());
    let mut batch_values = Vec::with_capacity(batch_capacity * dim);
    let mut batch_doc_ids = Vec::with_capacity(batch_capacity);
    for batch in assign_sources.chunks(settings.assign_batch_size) {
        if merge.ctx.cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
        batch_values.clear();
        batch_doc_ids.clear();
        for &(source, target_doc_id) in batch {
            timings.source_reads += 1;
            let bytes =
                merge.readers[source.segment_ord as usize].vector_bytes_for_row(source.row_id)?;
            decode_stored_row_append(&bytes, opts, &mut batch_values)?;
            batch_doc_ids.push(target_doc_id);
        }
        let assign_start = Instant::now();
        let clusters = clusterer.assign(
            opts,
            IvfVectors::F32(IvfVectorBatch {
                doc_ids: batch_doc_ids.as_slice(),
                matrix: IvfMatrixView {
                    values: batch_values.as_slice(),
                    rows: batch.len(),
                    dims: dim,
                },
            }),
            centroids,
        )?;
        timings.assign += assign_start.elapsed();
        if clusters.len() != batch.len() {
            return Err(TantivyError::InvalidArgument(format!(
                "IvfClusterer assigned {} clusters for {} vectors",
                clusters.len(),
                batch.len()
            )));
        }
        for (cluster, &(source, target_doc_id)) in clusters.into_iter().zip(batch) {
            let cluster = cluster as usize;
            if cluster >= num_centroids {
                return Err(TantivyError::InvalidArgument(format!(
                    "IvfClusterer assigned vector to cluster {cluster}, but only {num_centroids} \
                     centroids were trained"
                )));
            }
            assigned_vectors.push(AssignedVector {
                cluster,
                target_doc_id,
                source,
            });
        }
    }
    Ok(assigned_vectors)
}

/// Sorts `assigned` by (cluster, target doc) and returns the clusters' row offsets: a prefix sum
/// of `num_centroids + 1` entries.
pub(super) fn cluster_offsets(assigned: &mut [AssignedVector], num_centroids: usize) -> Vec<u64> {
    let mut cluster_counts = vec![0usize; num_centroids];
    for assigned_vector in assigned.iter() {
        cluster_counts[assigned_vector.cluster] += 1;
    }

    assigned.sort_unstable_by_key(|vector| (vector.cluster, vector.target_doc_id));

    let mut cluster_offsets: Vec<u64> = Vec::with_capacity(num_centroids + 1);
    let mut next_offset = 0u64;
    cluster_offsets.push(next_offset);
    for cluster_count in cluster_counts {
        next_offset += cluster_count as u64;
        cluster_offsets.push(next_offset);
    }
    cluster_offsets
}
