//! IVF-format merge routine.
//!
//! The IVF format is one of two storage modes the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) can produce per merge.
//! This module exposes the merge body so the parent plugin can call it
//! after the threshold check.

use std::io::Write;

use common::{BinarySerializable, OwnedBytes};

use super::{
    decode_row, encode_vector, IvfCentroids, IvfClusterer, IvfFieldMeta, IvfMatrix, IvfMatrixView,
    IvfVectorBatch, IvfVectors, ASSIGNMENTS_EXT, IVFVEC_EXT,
};
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::indexer::segment_updater::CancelSentinel;
use crate::plugin::PluginMergeContext;
use crate::schema::FieldType;
use crate::vector::meta::{VectorStorageFormat, VECMETA_EXT};
use crate::vector::reader::{VectorColumn, VectorColumnReader, VectorReader};
use crate::vector::{Metric, VectorDType, VectorOptions};
use crate::{DocId, TantivyError};

struct AssignedVector {
    cluster: usize,
    target_doc_id: DocId,
    source_segment_ord: usize,
    source_doc_id: DocId,
}

/// A proposed boundary-replication of a vector into a non-primary cluster
/// (Phase 2). `primary_idx` indexes the vector's primary [`AssignedVector`]
/// (its source bytes/doc-id are reused for the replica entry); `cluster` is
/// the trained-centroid id of the replication target, remapped to final ids
/// after rebalance and then subjected to the per-cluster budget.
struct ReplicaCandidate {
    primary_idx: usize,
    cluster: usize,
}

/// Safety cap on rebalance passes. Each pass dissolves undersized clusters
/// then splits oversized ones; a handful of passes converges the size
/// distribution into `[min_posting_len, max_posting_len]` in practice. The
/// cap bounds worst-case merge/split ping-pong rather than being expected to
/// be hit.
const MAX_REBALANCE_PASSES: usize = 4;

/// Fetch and decode a single source vector for an assigned entry.
fn decode_source_vector(
    columns: &[VectorColumn],
    entry: &AssignedVector,
    dim: usize,
) -> crate::Result<Vec<f32>> {
    let bytes = columns[entry.source_segment_ord]
        .vector_bytes_at(entry.source_doc_id)
        .ok_or_else(|| {
            TantivyError::InternalError(format!(
                "missing source vector for doc {:?} during cluster rebalance",
                entry.source_doc_id
            ))
        })?;
    decode_row::<f32>(bytes, dim)
}

/// Enforce the primary-membership size bounds produced by
/// [`IvfClusterer::merge_settings`]. Dissolves clusters below
/// `min_posting_len` (reassigning their members to the nearest surviving
/// centroid) and splits clusters above `max_posting_len` into sub-clusters,
/// appending the new centroids. On return, every entry in `assigned` points
/// at a densely re-indexed surviving centroid in `centroids`.
///
/// Operates on PRIMARY membership only — one entry per vector. Boundary
/// replication (Phase 2) appends additional entries afterwards.
#[allow(clippy::too_many_arguments)]
fn rebalance_clusters(
    cancel: &dyn CancelSentinel,
    opts: &VectorOptions,
    clusterer: &dyn IvfClusterer,
    columns: &[VectorColumn],
    assigned: &mut [AssignedVector],
    centroids: &mut Vec<Vec<f32>>,
    assign_batch_size: usize,
    max_posting_len: usize,
    min_posting_len: usize,
    target_posting_len: usize,
) -> crate::Result<Vec<usize>> {
    let dim = opts.dim();
    let mut alive = vec![true; centroids.len()];

    for _pass in 0..MAX_REBALANCE_PASSES {
        if cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
        let mut changed = merge_undersized(
            opts,
            clusterer,
            columns,
            assigned,
            centroids.as_slice(),
            &mut alive,
            assign_batch_size,
            min_posting_len,
            dim,
        )?;
        if cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
        changed |= split_oversized(
            opts,
            clusterer,
            columns,
            assigned,
            centroids,
            &mut alive,
            max_posting_len,
            target_posting_len,
            min_posting_len,
            dim,
        )?;
        if !changed {
            break;
        }
    }

    // Compact: drop dissolved (dead) centroids and densely re-index the
    // survivors. Every live entry already points at a surviving centroid, so
    // the remap is total.
    let mut remap = vec![usize::MAX; centroids.len()];
    let mut compacted: Vec<Vec<f32>> = Vec::with_capacity(centroids.len());
    for (c, centroid) in centroids.drain(..).enumerate() {
        if alive[c] {
            remap[c] = compacted.len();
            compacted.push(centroid);
        }
    }
    for entry in assigned.iter_mut() {
        let new_cluster = remap[entry.cluster];
        debug_assert_ne!(
            new_cluster,
            usize::MAX,
            "entry remained assigned to a dissolved cluster"
        );
        entry.cluster = new_cluster;
    }
    *centroids = compacted;
    // `remap[c]` maps a pre-rebalance centroid id (including the trained-id
    // prefix that replica candidates reference) to its final compacted id, or
    // `usize::MAX` if the cluster was dissolved. The driver uses it to
    // re-point replica candidates onto the final centroid set.
    Ok(remap)
}

/// Dissolve clusters with fewer than `min_posting_len` members, reassigning
/// each member to the nearest surviving centroid (one whose count is already
/// `>= min_posting_len`). Returns whether anything changed.
#[allow(clippy::too_many_arguments)]
fn merge_undersized(
    opts: &VectorOptions,
    clusterer: &dyn IvfClusterer,
    columns: &[VectorColumn],
    assigned: &mut [AssignedVector],
    centroids: &[Vec<f32>],
    alive: &mut [bool],
    assign_batch_size: usize,
    min_posting_len: usize,
    dim: usize,
) -> crate::Result<bool> {
    if min_posting_len == 0 {
        return Ok(false);
    }
    let n = centroids.len();
    let mut counts = vec![0usize; n];
    for entry in assigned.iter() {
        counts[entry.cluster] += 1;
    }
    let survivors: Vec<usize> = (0..n)
        .filter(|&c| alive[c] && counts[c] >= min_posting_len)
        .collect();
    let dissolved: Vec<usize> = (0..n)
        .filter(|&c| alive[c] && counts[c] < min_posting_len)
        .collect();
    // Nothing to dissolve, or nowhere to send the members.
    if dissolved.is_empty() || survivors.is_empty() {
        return Ok(false);
    }

    let mut survivor_values = Vec::with_capacity(survivors.len() * dim);
    for &c in &survivors {
        survivor_values.extend_from_slice(&centroids[c]);
    }
    let survivor_centroids = IvfCentroids::F32(IvfMatrix {
        values: survivor_values,
        rows: survivors.len(),
        dims: dim,
    });

    let mut is_dissolved = vec![false; n];
    for &c in &dissolved {
        is_dissolved[c] = true;
    }
    let members: Vec<usize> = (0..assigned.len())
        .filter(|&i| is_dissolved[assigned[i].cluster])
        .collect();

    let batch_size = assign_batch_size.max(1);
    let mut batch_values = Vec::with_capacity(batch_size * dim);
    let mut batch_doc_ids = Vec::with_capacity(batch_size);
    for chunk in members.chunks(batch_size) {
        batch_values.clear();
        batch_doc_ids.clear();
        for &i in chunk {
            let vector = decode_source_vector(columns, &assigned[i], dim)?;
            batch_values.extend_from_slice(&vector);
            batch_doc_ids.push(assigned[i].target_doc_id);
        }
        let local = clusterer.assign(
            opts,
            IvfVectors::F32(IvfVectorBatch {
                doc_ids: &batch_doc_ids,
                matrix: IvfMatrixView {
                    values: &batch_values,
                    rows: chunk.len(),
                    dims: dim,
                },
            }),
            &survivor_centroids,
        )?;
        if local.len() != chunk.len() {
            return Err(TantivyError::InvalidArgument(format!(
                "IvfClusterer reassigned {} clusters for {} vectors during merge",
                local.len(),
                chunk.len()
            )));
        }
        for (&i, assignment) in chunk.iter().zip(local.iter()) {
            // Rebalance reassigns primaries only; replicas are ignored here.
            let survivor_ord = assignment.primary as usize;
            if survivor_ord >= survivors.len() {
                return Err(TantivyError::InvalidArgument(format!(
                    "IvfClusterer reassigned to survivor {survivor_ord}, but only {} survive",
                    survivors.len()
                )));
            }
            assigned[i].cluster = survivors[survivor_ord];
        }
    }

    for &c in &dissolved {
        alive[c] = false;
    }
    Ok(true)
}

/// Split clusters with more than `max_posting_len` members into sub-clusters
/// (re-trained on just those members), reusing the original centroid id for
/// the first sub-cluster and appending the rest. Returns whether anything
/// changed. Only clusters present at entry are considered; sub-clusters that
/// are themselves still oversized are handled by the next rebalance pass.
#[allow(clippy::too_many_arguments)]
fn split_oversized(
    opts: &VectorOptions,
    clusterer: &dyn IvfClusterer,
    columns: &[VectorColumn],
    assigned: &mut [AssignedVector],
    centroids: &mut Vec<Vec<f32>>,
    alive: &mut Vec<bool>,
    max_posting_len: usize,
    target_posting_len: usize,
    min_posting_len: usize,
    dim: usize,
) -> crate::Result<bool> {
    if max_posting_len == usize::MAX {
        return Ok(false);
    }
    let n = centroids.len();
    let mut counts = vec![0usize; n];
    for entry in assigned.iter() {
        counts[entry.cluster] += 1;
    }
    let mut changed = false;
    for c in 0..n {
        if !alive[c] || counts[c] <= max_posting_len {
            continue;
        }
        let members: Vec<usize> = (0..assigned.len())
            .filter(|&i| assigned[i].cluster == c)
            .collect();
        let count = members.len();
        debug_assert_eq!(count, counts[c]);

        // Aim sub-clusters at the target posting length, but never produce so
        // many that the average sub-cluster would fall below `min_posting_len`
        // (which would only feed the next merge pass).
        let target = target_posting_len.max(1);
        let mut n_sub = count.div_ceil(target).max(2);
        if min_posting_len >= 1 {
            n_sub = n_sub.min((count / min_posting_len).max(2));
        }
        n_sub = n_sub.min(count);
        if n_sub < 2 {
            continue;
        }

        let mut values = Vec::with_capacity(count * dim);
        let mut doc_ids = Vec::with_capacity(count);
        for &i in &members {
            let vector = decode_source_vector(columns, &assigned[i], dim)?;
            values.extend_from_slice(&vector);
            doc_ids.push(assigned[i].target_doc_id);
        }
        let batch = IvfVectors::F32(IvfVectorBatch {
            doc_ids: &doc_ids,
            matrix: IvfMatrixView {
                values: &values,
                rows: count,
                dims: dim,
            },
        });
        let sub = clusterer.train(opts, batch, n_sub)?;
        let IvfCentroids::F32(sub_matrix) = &sub;
        if sub_matrix.dims != dim
            || sub_matrix.rows != n_sub
            || sub_matrix.values.len() != n_sub * dim
        {
            return Err(TantivyError::InvalidArgument(format!(
                "IvfClusterer split produced {} centroids ({} values, {} dims) for {n_sub} \
                 requested sub-clusters of dim {dim}",
                sub_matrix.rows,
                sub_matrix.values.len(),
                sub_matrix.dims
            )));
        }
        let local = clusterer.assign(opts, batch, &sub)?;
        if local.len() != count {
            return Err(TantivyError::InvalidArgument(format!(
                "IvfClusterer assigned {} sub-clusters for {count} split members",
                local.len()
            )));
        }

        // Sub-cluster 0 reuses centroid id `c`; the rest are appended.
        let mut sub_global = Vec::with_capacity(n_sub);
        sub_global.push(c);
        centroids[c].copy_from_slice(&sub_matrix.values[0..dim]);
        for j in 1..n_sub {
            sub_global.push(centroids.len());
            centroids.push(sub_matrix.values[j * dim..(j + 1) * dim].to_vec());
            alive.push(true);
        }
        for (&i, assignment) in members.iter().zip(local.iter()) {
            // Splitting reassigns primaries only; replicas are ignored here.
            let sub_ord = assignment.primary as usize;
            if sub_ord >= n_sub {
                return Err(TantivyError::InvalidArgument(format!(
                    "IvfClusterer split assigned to sub-cluster {sub_ord}, but only {n_sub} exist"
                )));
            }
            assigned[i].cluster = sub_global[sub_ord];
        }
        changed = true;
    }
    Ok(changed)
}

/// Phase 2 boundary replication, post-rebalance: re-point each replica
/// candidate onto the final centroid set, enforce the per-cluster budget, and
/// append the accepted replicas to `assigned` as extra entries (same source
/// bytes + doc-id as their primary, but a different target cluster).
///
/// A candidate is dropped if its target cluster was dissolved during rebalance
/// or if it maps to the vector's own primary cluster (which would duplicate
/// the doc within a single posting and break the reader's within-cluster
/// binary search). Per target cluster, when candidates exceed
/// `max_replicas_per_cluster` the nearest (highest-similarity) ones are kept.
fn append_replicas(
    opts: &VectorOptions,
    columns: &[VectorColumn],
    centroids: &[Vec<f32>],
    assigned: &mut Vec<AssignedVector>,
    candidates: &[ReplicaCandidate],
    remap: &[usize],
    max_replicas_per_cluster: usize,
) -> crate::Result<()> {
    let dim = opts.dim();
    let metric = opts.metric();
    // Per final cluster: (primary_idx, similarity-to-this-cluster). Higher
    // similarity == nearer (Metric is "higher is better").
    let mut by_cluster: Vec<Vec<(usize, f32)>> = vec![Vec::new(); centroids.len()];
    for cand in candidates {
        let final_cluster = remap[cand.cluster];
        if final_cluster == usize::MAX {
            continue; // replication target dissolved during rebalance
        }
        let primary = &assigned[cand.primary_idx];
        if final_cluster == primary.cluster {
            continue; // would duplicate the doc inside its own primary cluster
        }
        let vector = decode_source_vector(columns, primary, dim)?;
        let similarity = metric.similarity(&vector, &centroids[final_cluster]);
        by_cluster[final_cluster].push((cand.primary_idx, similarity));
    }

    for (final_cluster, mut cands) in by_cluster.into_iter().enumerate() {
        if cands.is_empty() {
            continue;
        }
        if max_replicas_per_cluster > 0 && cands.len() > max_replicas_per_cluster {
            // Distance-ranked acceptance: keep the nearest budget-many.
            cands.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            cands.truncate(max_replicas_per_cluster);
        }
        for (primary_idx, _similarity) in cands {
            let (target_doc_id, source_segment_ord, source_doc_id) = {
                let primary = &assigned[primary_idx];
                (
                    primary.target_doc_id,
                    primary.source_segment_ord,
                    primary.source_doc_id,
                )
            };
            assigned.push(AssignedVector {
                cluster: final_cluster,
                target_doc_id,
                source_segment_ord,
                source_doc_id,
            });
        }
    }
    Ok(())
}

pub(crate) fn merge_ivf(
    ctx: &PluginMergeContext,
    clusterer: Option<&dyn IvfClusterer>,
) -> crate::Result<()> {
    if ctx.cancel.wants_cancel() {
        return Err(TantivyError::Cancelled);
    }

    let has_vector_field = ctx
        .schema
        .fields()
        .any(|(_, entry)| matches!(entry.field_type(), FieldType::Vector(_)));
    if !has_vector_field {
        return Ok(());
    }

    let clusterer = clusterer.ok_or_else(|| {
        TantivyError::InvalidArgument(
            "vector_clustering_threshold selected IVF merge, but no IvfClusterer is configured"
                .to_string(),
        )
    })?;

    let num_target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
    if num_target_docs == 0 {
        return Ok(());
    }

    let settings = clusterer.merge_settings(num_target_docs as usize)?;
    let source_readers: Vec<VectorReader> = ctx
        .readers
        .iter()
        .map(VectorReader::open)
        .collect::<crate::Result<Vec<_>>>()?;

    let directory = ctx.target_segment.index().directory();
    let meta_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(VECMETA_EXT.to_string()));
    let assignments_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(ASSIGNMENTS_EXT.to_string()));
    let vec_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(IVFVEC_EXT.to_string()));
    let mut meta_file = directory.open_write(&meta_path)?;
    VectorStorageFormat::Ivf.serialize(&mut meta_file)?;
    let mut meta_write = CompositeWrite::wrap(meta_file);
    let mut assignments_write = CompositeWrite::wrap(directory.open_write(&assignments_path)?);
    let mut vec_write = CompositeWrite::wrap(directory.open_write(&vec_path)?);

    for (field, entry) in ctx.schema.fields() {
        let opts = match entry.field_type() {
            FieldType::Vector(opts) => opts,
            _ => continue,
        };
        let vector_count = source_readers
            .iter()
            .map(|reader| reader.count(field))
            .sum::<crate::Result<usize>>()?;
        if vector_count == 0 {
            {
                let assignments_w = assignments_write.for_field(field);
                assignments_w.flush()?;
            }
            {
                let vec_w = vec_write.for_field(field);
                vec_w.flush()?;
            }
            let mut cluster_offsets = Vec::with_capacity(8);
            0u64.serialize(&mut cluster_offsets)?;
            let meta = IvfFieldMeta {
                num_centroids: 0,
                centroid_bytes: OwnedBytes::new(Vec::new()),
                cluster_offsets: OwnedBytes::new(cluster_offsets),
            };
            {
                let meta_w = meta_write.for_field(field);
                meta.serialize(meta_w, opts)?;
                meta_w.flush()?;
            }
            continue;
        }
        let columns: Vec<_> = source_readers
            .iter()
            .map(|reader| reader.open_column(field))
            .collect::<crate::Result<Vec<_>>>()?;
        let num_centroids = settings.num_centroids.min(vector_count);
        let training_sample_size =
            vector_count.min(num_centroids.saturating_mul(settings.training_samples_per_centroid));
        let training_sample_interval = (vector_count / training_sample_size).max(1);
        match opts.dtype() {
            VectorDType::F32 => {
                let mut training_values = Vec::with_capacity(training_sample_size * opts.dim());
                let mut training_doc_ids = Vec::with_capacity(training_sample_size);
                let mut target_doc_id: DocId = 0;
                let mut present_vector_ord = 0usize;
                let mut sampled_count = 0usize;
                for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                    let column = &columns[old_doc_addr.segment_ord as usize];
                    if let Some(bytes) = column.vector_bytes_at(old_doc_addr.doc_id) {
                        let should_sample = sampled_count < training_sample_size
                            && present_vector_ord % training_sample_interval == 0;
                        if should_sample {
                            training_doc_ids.push(target_doc_id);
                            training_values
                                .extend_from_slice(&decode_row::<f32>(bytes, opts.dim())?);
                            sampled_count += 1;
                        }
                        present_vector_ord += 1;
                    }
                    target_doc_id += 1;
                }
                debug_assert_eq!(target_doc_id, num_target_docs);
                debug_assert_eq!(present_vector_ord, vector_count);
                if training_doc_ids.is_empty() {
                    continue;
                }

                let training_vectors = IvfVectors::F32(IvfVectorBatch {
                    doc_ids: &training_doc_ids,
                    matrix: IvfMatrixView {
                        values: &training_values,
                        rows: training_doc_ids.len(),
                        dims: opts.dim(),
                    },
                });
                let centroids = clusterer.train(opts, training_vectors, num_centroids)?;

                if ctx.cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }

                let IvfCentroids::F32(centroid_matrix) = &centroids;
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
                if centroid_matrix.rows != num_centroids {
                    return Err(TantivyError::InvalidArgument(format!(
                        "IvfClusterer produced {} centroids, but {num_centroids} were requested",
                        centroid_matrix.rows
                    )));
                }
                // Mutable working copy of the trained centroids. Encoding +
                // Cosine normalization is deferred until after rebalance,
                // which can split/merge clusters and append new centroids.
                let mut centroid_rows: Vec<Vec<f32>> = centroid_matrix
                    .values
                    .chunks_exact(opts.dim())
                    .map(|centroid| centroid.to_vec())
                    .collect();

                let mut assigned_vectors = Vec::with_capacity(vector_count);
                // Phase 2 replica candidates accumulated across assign batches
                // (empty unless the clusterer proposes replicas, i.e. unless
                // max_replicas_per_vector > 0). Empty => byte-identical to Phase 1.
                let mut replica_candidates: Vec<ReplicaCandidate> = Vec::new();
                let mut target_doc_id: DocId = 0;
                {
                    let mut batch_values = Vec::with_capacity(
                        settings.assign_batch_size.min(vector_count) * opts.dim(),
                    );
                    let mut batch_doc_ids =
                        Vec::with_capacity(settings.assign_batch_size.min(vector_count));
                    let mut batch_sources =
                        Vec::with_capacity(settings.assign_batch_size.min(vector_count));
                    let mut flush_assign_batch =
                        |batch_values: &mut Vec<f32>,
                         batch_doc_ids: &mut Vec<DocId>,
                         batch_sources: &mut Vec<(DocId, usize, DocId)>|
                         -> crate::Result<()> {
                            if batch_doc_ids.is_empty() {
                                return Ok(());
                            }
                            let batch_len = batch_doc_ids.len();
                            let clusters = clusterer.assign(
                                opts,
                                IvfVectors::F32(IvfVectorBatch {
                                    doc_ids: batch_doc_ids.as_slice(),
                                    matrix: IvfMatrixView {
                                        values: batch_values.as_slice(),
                                        rows: batch_len,
                                        dims: opts.dim(),
                                    },
                                }),
                                &centroids,
                            )?;
                            if clusters.len() != batch_len {
                                return Err(TantivyError::InvalidArgument(format!(
                                    "IvfClusterer assigned {} clusters for {} vectors",
                                    clusters.len(),
                                    batch_len
                                )));
                            }
                            for (assignment, (target_doc_id, source_segment_ord, source_doc_id)) in
                                clusters.into_iter().zip(batch_sources.drain(..))
                            {
                                let cluster = assignment.primary as usize;
                                if cluster >= num_centroids {
                                    return Err(TantivyError::InvalidArgument(format!(
                                        "IvfClusterer assigned vector to cluster {cluster}, but \
                                         only {num_centroids} centroids were trained"
                                    )));
                                }
                                let primary_idx = assigned_vectors.len();
                                assigned_vectors.push(AssignedVector {
                                    cluster,
                                    target_doc_id,
                                    source_segment_ord,
                                    source_doc_id,
                                });
                                // Phase 2: stash this vector's replica candidates
                                // (trained-centroid ids) against the primary entry;
                                // the per-cluster budget is enforced after rebalance.
                                for &replica_cluster in &assignment.replicas {
                                    replica_candidates.push(ReplicaCandidate {
                                        primary_idx,
                                        cluster: replica_cluster as usize,
                                    });
                                }
                            }
                            batch_values.clear();
                            batch_doc_ids.clear();
                            Ok(())
                        };
                    for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                        let column = &columns[old_doc_addr.segment_ord as usize];
                        if let Some(bytes) = column.vector_bytes_at(old_doc_addr.doc_id) {
                            batch_doc_ids.push(target_doc_id);
                            batch_values.extend_from_slice(&decode_row::<f32>(bytes, opts.dim())?);
                            batch_sources.push((
                                target_doc_id,
                                old_doc_addr.segment_ord as usize,
                                old_doc_addr.doc_id,
                            ));
                            if batch_doc_ids.len() == settings.assign_batch_size {
                                flush_assign_batch(
                                    &mut batch_values,
                                    &mut batch_doc_ids,
                                    &mut batch_sources,
                                )?;
                            }
                        }
                        target_doc_id += 1;
                    }
                    flush_assign_batch(&mut batch_values, &mut batch_doc_ids, &mut batch_sources)?;
                }
                debug_assert_eq!(target_doc_id, num_target_docs);
                debug_assert_eq!(assigned_vectors.len(), vector_count);

                // ---- Phase 1: balance enforcement on PRIMARY membership ----
                // Dissolve undersized clusters (lifts the min off 1) and split
                // oversized ones (caps the max). Operates on PRIMARY membership
                // only; boundary replicas (Phase 2) are appended afterwards and
                // never count against the split threshold. `rebalance_clusters`
                // returns the centroid-id remap so the replica candidates
                // (captured against trained ids) can be re-pointed onto the
                // final centroid set.
                let remap: Vec<usize> = if settings.max_posting_len != usize::MAX
                    || settings.min_posting_len != 0
                {
                    // Natural target posting length ≈ mean cluster size
                    // (≈ 1/centroid_ratio); splits aim sub-clusters here.
                    let target_posting_len = (vector_count / num_centroids)
                        .max(1)
                        .min(settings.max_posting_len.max(1));
                    if matches!(opts.metric(), Metric::Dot) {
                        // NOTE(metric soundness): Dot has no triangle
                        // inequality and is clustered from raw, unnormalized
                        // magnitudes, so the Euclidean balanced split, the
                        // nearest-centroid reassignment, AND the Phase-2 ε₁
                        // replica gate are all geometrically questionable for
                        // Dot. The clusterer sets `angular=true` for Dot;
                        // replicas follow that same treatment. NOT silently
                        // treated as L2 — flagged, still unresolved.
                    }
                    rebalance_clusters(
                        ctx.cancel,
                        opts,
                        clusterer,
                        &columns,
                        &mut assigned_vectors,
                        &mut centroid_rows,
                        settings.assign_batch_size,
                        settings.max_posting_len,
                        settings.min_posting_len,
                        target_posting_len,
                    )?
                } else {
                    // Balancing disabled: centroids unchanged, identity remap.
                    (0..centroid_rows.len()).collect()
                };

                // Rebalance may have changed the centroid count.
                let num_centroids = centroid_rows.len();

                // ---- Phase 2: boundary replication ----
                // Re-point each replica candidate onto the final centroid set,
                // enforce the per-cluster budget (distance-ranked acceptance),
                // and append the survivors as extra `(vector, cluster)` entries.
                // Skipped entirely when replication is off (no candidates), so
                // the write below stays byte-identical to Phase 1.
                if settings.max_replicas_per_vector > 0 && !replica_candidates.is_empty() {
                    append_replicas(
                        opts,
                        &columns,
                        &centroid_rows,
                        &mut assigned_vectors,
                        &replica_candidates,
                        &remap,
                        settings.max_replicas_per_cluster,
                    )?;
                }

                // radius: per-cluster radius is computed HERE — AFTER replication,
                // over FINAL membership (primaries + accepted replicas). Boundary
                // replicas are the farthest members and define r_c, so computing
                // before replication would under-cover stored vectors. No meta
                // field persists radius yet; this is the persist point that will
                // feed the future radius-bound termination. (stub)

                let mut cluster_counts = vec![0usize; num_centroids];
                for assigned_vector in &assigned_vectors {
                    cluster_counts[assigned_vector.cluster] += 1;
                }

                assigned_vectors
                    .sort_unstable_by_key(|vector| (vector.cluster, vector.target_doc_id));

                let mut cluster_offsets = Vec::with_capacity((num_centroids + 1) * 8);
                let mut next_offset = 0u64;
                next_offset.serialize(&mut cluster_offsets)?;
                for cluster_count in cluster_counts {
                    next_offset += cluster_count as u64;
                    next_offset.serialize(&mut cluster_offsets)?;
                }

                {
                    let assignments_w = assignments_write.for_field(field);
                    for assigned_vector in &assigned_vectors {
                        assigned_vector.target_doc_id.serialize(assignments_w)?;
                    }
                    assignments_w.flush()?;
                }

                {
                    let vec_w = vec_write.for_field(field);
                    for assigned_vector in &assigned_vectors {
                        let column = &columns[assigned_vector.source_segment_ord];
                        let bytes = column
                            .vector_bytes_at(assigned_vector.source_doc_id)
                            .ok_or_else(|| {
                                TantivyError::InternalError(format!(
                                    "missing source vector for doc {:?}",
                                    assigned_vector.source_doc_id
                                ))
                            })?;
                        vec_w.write_all(bytes)?;
                    }
                    vec_w.flush()?;
                }

                // K-means cluster means are not unit-norm; for Cosine+F32
                // normalize each centroid here so the search path can score
                // both docs and centroids with the same `dot * inv_norm_q`
                // fast kernel.
                let mut centroid_bytes =
                    Vec::with_capacity(num_centroids * opts.bytes_per_vector());
                for centroid in &centroid_rows {
                    let mut bytes = encode_vector(centroid, opts.dim())?;
                    opts.maybe_normalize_bytes(&mut bytes);
                    centroid_bytes.extend_from_slice(&bytes);
                }
                let meta = IvfFieldMeta {
                    num_centroids,
                    centroid_bytes: OwnedBytes::new(centroid_bytes),
                    cluster_offsets: OwnedBytes::new(cluster_offsets),
                };
                {
                    let meta_w = meta_write.for_field(field);
                    meta.serialize(meta_w, opts)?;
                    meta_w.flush()?;
                }
            }
        }
    }

    meta_write.close()?;
    assignments_write.close()?;
    vec_write.close()?;
    Ok(())
}
