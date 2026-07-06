//! IVF-format merge routine.
//!
//! The IVF format is one of two storage modes the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) can produce per merge.
//! This module exposes the merge body so the parent plugin can call it
//! after the threshold check.

use std::io::Write;

use super::{
    decode_row, encode_vector, CentroidsMeta, IvfCentroids, IvfClusterer, IvfMatrixView,
    IvfVectorBatch, IvfVectors, CENTROIDS_EXT,
};
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
use crate::schema::{FieldType, VectorDType};
use crate::vector::flat::IdMap;
use crate::vector::header::write_header;
use crate::vector::reader::{VectorColumnReader, VectorReader};
use crate::vector::VEC_EXT;
use crate::{DocId, TantivyError};

struct AssignedVector {
    cluster: usize,
    target_doc_id: DocId,
    source_segment_ord: usize,
    source_doc_id: DocId,
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
    let vec_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
    let centroids_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(CENTROIDS_EXT.to_string()));
    let mut vec_file = directory.open_write(&vec_path)?;
    write_header(&mut vec_file)?;
    let mut vec_write = CompositeWrite::wrap(vec_file);
    let mut centroids_write = CompositeWrite::wrap(directory.open_write(&centroids_path)?);

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
            // `.vec`: empty Explicit id-map + empty rows.
            {
                let id_map_w = vec_write.for_field_with_idx(field, 0);
                IdMap::serialize_explicit(&[], id_map_w)?;
                id_map_w.flush()?;
            }
            {
                let rows_w = vec_write.for_field_with_idx(field, 1);
                rows_w.flush()?;
            }
            // `.centroids`: zero centroids, single zero offset.
            {
                let centroids_w = centroids_write.for_field_with_idx(field, 0);
                CentroidsMeta::serialize_centroids(0, &[], opts, centroids_w)?;
                centroids_w.flush()?;
            }
            {
                let offsets_w = centroids_write.for_field_with_idx(field, 1);
                CentroidsMeta::serialize_offsets(&[0u64], offsets_w)?;
                offsets_w.flush()?;
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
                let encoded_centroids = centroid_matrix
                    .values
                    .chunks_exact(opts.dim())
                    .map(|centroid| encode_vector(centroid, opts.dim()))
                    .collect::<crate::Result<Vec<_>>>()?;

                let mut assigned_vectors = Vec::with_capacity(vector_count);
                let mut cluster_counts = vec![0usize; num_centroids];
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
                            for (cluster, (target_doc_id, source_segment_ord, source_doc_id)) in
                                clusters.into_iter().zip(batch_sources.drain(..))
                            {
                                let cluster = cluster as usize;
                                if cluster >= num_centroids {
                                    return Err(TantivyError::InvalidArgument(format!(
                                        "IvfClusterer assigned vector to cluster {cluster}, but \
                                         only {num_centroids} centroids were trained"
                                    )));
                                }
                                assigned_vectors.push(AssignedVector {
                                    cluster,
                                    target_doc_id,
                                    source_segment_ord,
                                    source_doc_id,
                                });
                                cluster_counts[cluster] += 1;
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
                assigned_vectors
                    .sort_unstable_by_key(|vector| (vector.cluster, vector.target_doc_id));

                let mut cluster_offsets: Vec<u64> = Vec::with_capacity(num_centroids + 1);
                let mut next_offset = 0u64;
                cluster_offsets.push(next_offset);
                for cluster_count in cluster_counts {
                    next_offset += cluster_count as u64;
                    cluster_offsets.push(next_offset);
                }

                // `.vec` slot [0]: the row→doc_id permutation (Explicit), in
                // cluster-sorted row order — parallel to the rows in slot [1].
                {
                    let id_map_w = vec_write.for_field_with_idx(field, 0);
                    let row_doc_ids: Vec<DocId> = assigned_vectors
                        .iter()
                        .map(|assigned_vector| assigned_vector.target_doc_id)
                        .collect();
                    IdMap::serialize_explicit(&row_doc_ids, id_map_w)?;
                    id_map_w.flush()?;
                }

                // `.vec` slot [1]: the cluster-sorted vector rows.
                {
                    let rows_w = vec_write.for_field_with_idx(field, 1);
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
                        rows_w.write_all(bytes)?;
                    }
                    rows_w.flush()?;
                }

                // `.centroids`: routing — centroids in slot [0], cluster
                // offsets in slot [1].
                let centroid_bytes: Vec<u8> = encoded_centroids.into_iter().flatten().collect();
                {
                    let centroids_w = centroids_write.for_field_with_idx(field, 0);
                    CentroidsMeta::serialize_centroids(
                        num_centroids,
                        &centroid_bytes,
                        opts,
                        centroids_w,
                    )?;
                    centroids_w.flush()?;
                }
                {
                    let offsets_w = centroids_write.for_field_with_idx(field, 1);
                    CentroidsMeta::serialize_offsets(&cluster_offsets, offsets_w)?;
                    offsets_w.flush()?;
                }
            }
        }
    }

    vec_write.close()?;
    centroids_write.close()?;
    Ok(())
}
