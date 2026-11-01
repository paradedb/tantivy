//! Primary and optional boundary assignments, and the clusters' row ranges.

use std::time::Instant;

use itertools::Itertools;

use super::source::MergeVectorSource;
use super::{AssignedVector, FieldMerge, IvfBuildTimings};
use crate::vector::clustering::{IvfCentroids, IvfClusterer, VectorSource};
use crate::vector::write::RowAddress;
use crate::{DocId, TantivyError};

/// Assigns each vector to its primary and optional secondary, reading the sources in storage
/// order through a [`MergeVectorSource`].
pub(super) fn assign(
    merge: &FieldMerge,
    clusterer: &dyn IvfClusterer,
    source_rows: Vec<Option<RowAddress>>,
    centroids: &IvfCentroids,
    vector_count: usize,
    timings: &mut IvfBuildTimings,
) -> crate::Result<Vec<AssignedVector>> {
    let opts = merge.opts;
    let IvfCentroids::F32(centroid_matrix) = centroids;
    let num_centroids = centroid_matrix.rows;
    // Assignment is per vector, so the source follows storage order;
    // `assigned_vectors` is re-sorted by (cluster, target doc) afterwards.
    let assign_sources: Vec<(RowAddress, DocId)> = source_rows
        .into_iter()
        .enumerate()
        .filter_map(|(doc_id, row)| row.map(|row| (row, doc_id as DocId)))
        .sorted()
        .collect();
    debug_assert!(assign_sources.len() <= vector_count);

    let mut source =
        MergeVectorSource::new(&merge.readers, assign_sources, opts.dim(), merge.ctx.cancel);
    let assign_start = Instant::now();
    let clusters = clusterer.assign_with_replicas(opts, &mut source, centroids)?;
    timings.assign += assign_start.elapsed();
    timings.assign_reads += source.reads;

    if clusters.len() != source.num_rows() {
        return Err(TantivyError::InvalidArgument(format!(
            "IvfClusterer assigned {} clusters for {} vectors",
            clusters.len(),
            source.num_rows()
        )));
    }

    let mut assigned_vectors = Vec::with_capacity(clusters.len());
    for (assignment, &(row, target_doc_id)) in clusters.into_iter().zip(source.rows()) {
        if assignment.secondary == Some(assignment.primary) {
            return Err(TantivyError::InvalidArgument(format!(
                "IvfClusterer duplicated primary cluster {}",
                assignment.primary
            )));
        }
        for (cluster, primary) in std::iter::once((assignment.primary, true))
            .chain(assignment.secondary.map(|cluster| (cluster, false)))
        {
            let cluster = cluster as usize;
            if cluster >= num_centroids {
                return Err(TantivyError::InvalidArgument(format!(
                    "IvfClusterer assigned vector to cluster {cluster}, but only {num_centroids} \
                     centroids were trained"
                )));
            }
            assigned_vectors.push(AssignedVector {
                cluster,
                primary,
                target_doc_id,
                source: row,
            });
        }
    }
    if assigned_vectors.len() > u32::MAX as usize {
        return Err(TantivyError::InvalidArgument(
            "too many vector postings".into(),
        ));
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
