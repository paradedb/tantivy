//! Unified vector storage plugin.
//!
//! [`VectorPlugin`] owns per-segment vector storage end-to-end:
//! - During indexing, accumulates raw vector bytes per doc and writes a single `.vec` file at
//!   segment finalize (always flat — clustering is a merge-time transform).
//! - During merge, picks one of two output formats by target doc count: below
//!   [`IndexSettings::vector_clustering_threshold`](crate::index::IndexSettings::vector_clustering_threshold)
//!   it copies vectors forward into a flat `.vec`; at or above the threshold it writes an IVF
//!   `.vec` (with `IdMap::DocLocations`) plus a `.centroids` file.
//! - During reads, [`VectorIndexReader`](super::VectorIndexReader) opens the field's `.vec` slots
//!   (and the `.centroids` sidecar when present) via
//!   [`SegmentReader::vector_index`](crate::SegmentReader::vector_index).
//!
//! Owning both flat and IVF extensions on one plugin keeps the "exactly one
//! format per segment" invariant right by construction: the dispatch is one
//! `if` inside one `merge()` method, not a cross-plugin coordination problem.

use super::flat::{merge_flat, FlatVecWriter};
use super::ivf::{merge_ivf, CENTROIDS_EXT};
use super::VEC_EXT;
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};

pub struct VectorPlugin;

impl SegmentPlugin for VectorPlugin {
    fn extensions(&self) -> &[&str] {
        &[VEC_EXT, CENTROIDS_EXT]
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        // Per-doc indexing only ever produces flatvec — clustering
        // exists exclusively as a merge-time transformation.
        Ok(Box::new(FlatVecWriter::for_schema(&ctx.segment.schema())))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        // Target cardinality selects uniform or clustered storage.
        let target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
        let threshold = ctx.settings.vector_clustering_threshold();
        if (target_docs as usize) < threshold {
            merge_flat(&ctx)
        } else {
            merge_ivf(
                &ctx,
                ctx.target_segment.index().ivf_clusterer(),
                ctx.target_segment.index().ivf_router(),
            )
        }
    }
}

/// Resolves target-document order to source rows using document columns or flat maps.
/// Target order fixes training samples and assignment batches independently of source clustering.
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
