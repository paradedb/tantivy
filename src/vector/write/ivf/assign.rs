//! The assign step: every vector to its nearest centroid and, when the field spills, to additional
//! cells; and the clusters' row ranges.

use std::time::Instant;

use itertools::Itertools;

use super::source::MergeVectorSource;
use super::{AssignedVector, FieldMerge, IvfBuildTimings};
use crate::vector::clustering::soar::{SOAR_LAMBDA, SOAR_REPLICAS};
use crate::vector::clustering::{IvfCentroids, IvfClusterer, VectorSource};
use crate::vector::write::RowAddress;
use crate::{DocId, TantivyError};

/// Assigns vectors in storage order, with the primary followed by distinct spill cells.
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
    let replicas = if merge.spill { SOAR_REPLICAS } else { 1 };
    let (primary, secondary) = if merge.spill {
        clusterer.assign_replicas(opts, &mut source, centroids, SOAR_LAMBDA, replicas)?
    } else {
        (clusterer.assign(opts, &mut source, centroids)?, Vec::new())
    };
    timings.assign += assign_start.elapsed();
    timings.assign_reads += source.reads;

    if primary.len() != source.num_rows() || secondary.len() != primary.len() * (replicas - 1) {
        return Err(TantivyError::InvalidArgument(format!(
            "IvfClusterer returned {} primaries and {} secondary slots for {} vectors",
            primary.len(),
            secondary.len(),
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
    let mut assigned_vectors = Vec::with_capacity(primary.len() * replicas);
    for (index, (primary, &(row, target_doc_id))) in
        primary.into_iter().zip(source.rows()).enumerate()
    {
        let cluster = check(primary)?;
        assigned_vectors.push(AssignedVector {
            cluster,
            target_doc_id,
            source: row,
            primary: true,
        });
        let secondaries = &secondary[index * (replicas - 1)..(index + 1) * (replicas - 1)];
        for (slot, &secondary) in secondaries.iter().enumerate() {
            if secondary == u32::MAX {
                continue;
            }
            let secondary_cluster = check(secondary)?;
            if secondary == primary || secondaries[..slot].contains(&secondary) {
                return Err(TantivyError::InvalidArgument(format!(
                    "IvfClusterer repeated cluster {secondary} for one vector"
                )));
            }
            assigned_vectors.push(AssignedVector {
                cluster: secondary_cluster,
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
