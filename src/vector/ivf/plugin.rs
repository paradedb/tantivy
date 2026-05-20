//! IVF-format merge routine.
//!
//! The IVF format is one of two storage modes the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) can produce per merge.
//! This module exposes the merge body — clustering source vectors
//! into an `.ivfvec` file — so the parent plugin can call it after the
//! threshold check.
//!
//! Status: algorithm pending. The parent plugin only routes here when
//! the target segment's doc count meets the clustering threshold; that
//! branch is unreachable under the default `usize::MAX` threshold, so
//! existing tests are unaffected.

use std::sync::Arc;

use super::{
    decode_row, encode_vector, IvfCentroids, IvfClusterer, IvfTypedVector, IvfVector, IvfVectors,
};
use crate::plugin::PluginMergeContext;
use crate::schema::FieldType;
use crate::vector::reader::{VectorColumnReader, VectorReader};
use crate::vector::{VectorDType, VECTOR_PLUGIN_NAME};
use crate::{DocId, TantivyError};

/// Cluster source vectors and write the target segment's `.ivfvec`.
///
/// Caller (`VectorPlugin::merge`) has already verified that the target
/// segment's doc count meets the clustering threshold. This routine is
/// unconditional — when called, it owns producing the target's IVF
/// output (centroid table, per-cluster doc-id postings, per-cluster
/// vector blob, cluster offset table) from whatever the source
/// segments expose (flat columns or IVF columns).
///
/// Sketch of the implementation:
///   1. Read source vectors. For each source, prefer the flat column
///      if present; otherwise reconstruct from the source's own IVF
///      data (vectors live inside the per-cluster blob).
///   2. Ask the configured clusterer for centroids.
///   3. Ask the configured clusterer for vector-to-centroid assignments.
///   4. Serialize the IVF layout
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
    let source_readers: Vec<Option<Arc<VectorReader>>> = ctx
        .readers
        .iter()
        .map(|reader| reader.plugin_reader::<VectorReader>(VECTOR_PLUGIN_NAME))
        .collect::<crate::Result<Vec<_>>>()?;

    for (field, entry) in ctx.schema.fields() {
        let opts = match entry.field_type() {
            FieldType::Vector(opts) => opts,
            _ => continue,
        };
        let vector_count = source_readers
            .iter()
            .map(|reader_opt| match reader_opt {
                Some(reader) => reader.count(field),
                None => Err(TantivyError::InternalError(
                    "vectors plugin reader missing during IVF vector merge".to_string(),
                )),
            })
            .sum::<crate::Result<usize>>()?;
        if vector_count == 0 {
            continue;
        }
        let columns: Vec<_> = source_readers
            .iter()
            .map(|reader_opt| match reader_opt {
                Some(reader) => reader.open_column(field),
                None => Err(TantivyError::InternalError(
                    "vectors plugin reader missing during IVF vector merge".to_string(),
                )),
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let num_centroids = settings.num_centroids.min(vector_count);
        let training_sample_size =
            vector_count.min(num_centroids.saturating_mul(settings.training_samples_per_centroid));
        let training_sample_interval = (vector_count / training_sample_size).max(1);
        match opts.dtype() {
            VectorDType::F32 => {
                let mut training_vectors = Vec::with_capacity(training_sample_size);
                let mut new_doc_id: DocId = 0;
                let mut present_vector_ord = 0usize;
                let mut sampled_count = 0usize;
                for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                    let column = &columns[old_doc_addr.segment_ord as usize];
                    if let Some(bytes) = column.vector_bytes_at(old_doc_addr.doc_id) {
                        let should_sample = sampled_count < training_sample_size
                            && present_vector_ord % training_sample_interval == 0;
                        if should_sample {
                            training_vectors.push(IvfTypedVector {
                                doc_id: new_doc_id,
                                vector: decode_row::<f32>(bytes, opts.dim())?,
                            });
                            sampled_count += 1;
                        }
                        present_vector_ord += 1;
                    }
                    new_doc_id += 1;
                }
                debug_assert_eq!(new_doc_id, num_target_docs);
                debug_assert_eq!(present_vector_ord, vector_count);
                if training_vectors.is_empty() {
                    continue;
                }

                let centroids = clusterer.train(
                    opts,
                    IvfVectors::F32(&training_vectors),
                    num_centroids,
                )?;

                if ctx.cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }

                let encoded_centroids = match &centroids {
                    IvfCentroids::F32(centroids) => centroids
                        .iter()
                        .map(|centroid| encode_vector(centroid, opts.dim()))
                        .collect::<crate::Result<Vec<_>>>()?,
                };

                let mut assignments = Vec::new();
                let mut new_doc_id: DocId = 0;
                for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                    let column = &columns[old_doc_addr.segment_ord as usize];
                    if let Some(bytes) = column.vector_bytes_at(old_doc_addr.doc_id) {
                        let vector = decode_row::<f32>(bytes, opts.dim())?;
                        let vector = IvfTypedVector {
                            doc_id: new_doc_id,
                            vector,
                        };
                        assignments.push(clusterer.assign(
                            opts,
                            IvfVector::F32(vector),
                            &centroids,
                        )?);
                    }
                    new_doc_id += 1;
                }
                debug_assert_eq!(new_doc_id, num_target_docs);
                let _ = (encoded_centroids, assignments);
            }
        }
    }

    todo!("IVF serialization at merge time")
}
