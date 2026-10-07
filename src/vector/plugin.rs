//! Unified vector storage plugin.
//!
//! [`VectorPlugin`] owns per-segment vector storage end-to-end:
//! - With shared centroids, flushes assign vectors to the persisted centroid order; merges preserve
//!   clustered memberships and assign flat inputs. Both write clustered `.vec` blocks plus
//!   segment-specific `.centroids` metadata.
//! - Without shared centroids, flushes and merges write flat `.vec` files.
//! - During reads, [`VectorIndexReader`](super::VectorIndexReader) opens the field's `.vec` slots
//!   (and the `.centroids` sidecar when present) via
//!   [`SegmentReader::vector_index`](crate::SegmentReader::vector_index).
use super::flat::{merge_flat, FlatVecWriter};
use super::ivf::{merge_shared, IvfVecWriter, CENTROIDS_EXT};
use super::VEC_EXT;
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};

pub struct VectorPlugin;

impl SegmentPlugin for VectorPlugin {
    fn extensions(&self) -> &[&str] {
        &[VEC_EXT, CENTROIDS_EXT]
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        let schema = ctx.segment.schema();
        if ctx.segment.index().centroid_index_meta().is_some() {
            Ok(Box::new(IvfVecWriter::for_schema(&schema)))
        } else {
            Ok(Box::new(FlatVecWriter::for_schema(&schema)))
        }
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        if ctx.target_segment.index().centroid_index_meta().is_some() {
            return merge_shared(&ctx);
        }
        merge_flat(&ctx)
    }
}

/// Resolves target-document order to source rows using document columns or flat maps.
/// Target order is independent of source clustering.
pub(crate) fn merge_source_rows(
    ctx: &PluginMergeContext,
    readers: &[std::sync::Arc<super::VectorIndexReader>],
) -> crate::Result<Vec<Option<(usize, usize)>>> {
    let mut target_docs: Vec<Vec<Option<crate::DocId>>> = ctx
        .readers
        .iter()
        .map(|reader| vec![None; reader.max_doc() as usize])
        .collect();
    let mut count = 0;
    for (new_doc, source) in ctx.doc_id_mapping.iter_source_doc_addrs().enumerate() {
        target_docs[source.segment_ord as usize][source.doc_id as usize] =
            Some(new_doc as crate::DocId);
        count += 1;
    }
    let mut source_rows = vec![None; count];
    for (segment, reader) in readers.iter().enumerate() {
        let mut record = |row, doc| -> crate::Result<()> {
            if ctx.cancel.wants_cancel() {
                return Err(crate::TantivyError::Cancelled);
            }
            let new_doc = target_docs[segment].get(doc as usize).ok_or_else(|| {
                crate::error::DataCorruption::comment_only(
                    "DocIds contains a document outside the segment",
                )
            })?;
            if let Some(new_doc) = new_doc {
                source_rows[*new_doc as usize] = Some((segment, row));
            }
            Ok(())
        };
        if let Some(index) = reader.index() {
            let mut docs = Vec::new();
            for cluster in 0..index.num_clusters() {
                reader.read_doc_ids(cluster, &mut docs)?;
                for (row, &doc) in index.cluster_range(cluster).zip(&docs) {
                    record(row, doc)?;
                }
            }
        } else {
            for (row, doc) in reader
                .row_doc_ids(0..reader.num_vectors())?
                .into_iter()
                .enumerate()
            {
                record(row, doc)?;
            }
        }
    }
    Ok(source_rows)
}
