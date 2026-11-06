//! The assign step: every vector to its nearest centroid and, when the field spills, to a second
//! one; and the clusters' row ranges.

use std::time::Instant;

use itertools::Itertools;

use super::source::MergeVectorSource;
use super::{AssignedVector, FieldMerge, IvfBuildTimings};
use crate::vector::clustering::soar::SOAR_LAMBDA;
use crate::vector::clustering::{IvfCentroids, IvfClusterer, VectorSource};
use crate::vector::write::RowAddress;
use crate::{DocId, TantivyError};

/// Assigns each of the field's vectors to one of `centroids`, and to a second one when the field
/// spills, reading the sources in storage order through a [`MergeVectorSource`]. A spilled vector
/// yields two entries with the same source and document, the primary first.
pub(super) fn assign(
    merge: &FieldMerge,
    clusterer: &dyn IvfClusterer,
    source_rows: &[Option<RowAddress>],
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
        .iter()
        .enumerate()
        .filter_map(|(doc_id, row)| row.map(|row| (row, doc_id as DocId)))
        .sorted()
        .collect();
    debug_assert!(assign_sources.len() <= vector_count);

    let mut source =
        MergeVectorSource::new(&merge.readers, assign_sources, opts.dim(), merge.ctx.cancel);
    let assign_start = Instant::now();
    let pairs: Vec<(u32, Option<u32>)> = if merge.spill {
        clusterer.assign_pairs(opts, &mut source, centroids, SOAR_LAMBDA)?
    } else {
        clusterer
            .assign(opts, &mut source, centroids)?
            .into_iter()
            .map(|cluster| (cluster, None))
            .collect()
    };
    timings.assign += assign_start.elapsed();
    timings.assign_reads += source.reads;

    if pairs.len() != source.num_rows() {
        return Err(TantivyError::InvalidArgument(format!(
            "IvfClusterer assigned {} clusters for {} vectors",
            pairs.len(),
            source.num_rows()
        )));
    }

    let check = |cluster: u32| -> crate::Result<usize> {
        let cluster = cluster as usize;
        if cluster >= num_centroids {
            return Err(TantivyError::InvalidArgument(format!(
                "IvfClusterer assigned vector to cluster {cluster}, but only {num_centroids} \
                 centroids were trained"
            )));
        }
        Ok(cluster)
    };
    let mut assigned_vectors = Vec::with_capacity(pairs.len());
    for ((primary, secondary), &(row, target_doc_id)) in pairs.into_iter().zip(source.rows()) {
        let cluster = check(primary)?;
        assigned_vectors.push(AssignedVector {
            cluster,
            target_doc_id,
            source: row,
            primary: true,
        });
        if let Some(secondary) = secondary {
            let secondary = check(secondary)?;
            if secondary == cluster {
                return Err(TantivyError::InvalidArgument(format!(
                    "IvfClusterer spilled a vector into its own cluster {cluster}"
                )));
            }
            assigned_vectors.push(AssignedVector {
                cluster: secondary,
                target_doc_id,
                source: row,
                primary: false,
            });
        }
    }
    Ok(assigned_vectors)
}

/// Sorts `assigned` by (cluster, target doc) and returns the clusters' row offsets: a prefix sum
/// of `num_centroids + 1` entries. A document appears at most once per cluster, so each block's
/// documents stay strictly ascending.
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
