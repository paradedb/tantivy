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

use std::sync::Arc;

use super::flat::{merge_flat, FlatVecWriter};
use super::ivf::{merge_ivf, CENTROIDS_EXT};
use super::{VectorIndexReader, VEC_EXT};
use crate::error::DataCorruption;
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};
use crate::{DocAddress, DocId, SegmentOrdinal, TantivyError};

pub struct VectorPlugin;

/// Physical position of a vector in a segment's vector storage.
///
/// Not a `DocId`: docs without a vector have no row, and clustered segments order rows by
/// cluster.
pub(crate) type RowId = usize;

/// A vector row within a set of segments, the row counterpart of [`DocAddress`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RowAddress {
    pub segment_ord: SegmentOrdinal,
    pub row_id: RowId,
}

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
///
/// Returns one entry per target `DocId`, `None` for docs without a vector.
pub(crate) fn merge_source_rows(
    ctx: &PluginMergeContext,
    readers: &[Arc<VectorIndexReader>],
) -> crate::Result<Vec<Option<RowAddress>>> {
    // Indexed by source `DocAddress`; `None` for deleted docs.
    let mut target_doc_ids: Vec<Vec<Option<DocId>>> = ctx
        .readers
        .iter()
        .map(|reader| vec![None; reader.max_doc() as usize])
        .collect();
    let mut num_target_docs = 0;
    for (target_doc_id, source) in ctx.doc_id_mapping.iter_source_doc_addrs().enumerate() {
        target_doc_ids[source.segment_ord as usize][source.doc_id as usize] =
            Some(target_doc_id as DocId);
        num_target_docs += 1;
    }

    let mut source_rows = vec![None; num_target_docs];
    for (segment_ord, reader) in readers.iter().enumerate() {
        let segment_ord = segment_ord as SegmentOrdinal;
        for_each_row(reader, |row_id, doc_id| {
            if ctx.cancel.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            let source = DocAddress::new(segment_ord, doc_id);
            let target_doc_id = target_doc_ids[source.segment_ord as usize]
                .get(source.doc_id as usize)
                .ok_or_else(|| {
                    DataCorruption::comment_only("DocIds contains a document outside the segment")
                })?;
            if let Some(target_doc_id) = *target_doc_id {
                source_rows[target_doc_id as usize] = Some(RowAddress {
                    segment_ord,
                    row_id,
                });
            }
            Ok(())
        })?;
    }
    Ok(source_rows)
}

/// Calls `f` with every vector row in `reader` and the source `DocId` stored in it.
fn for_each_row(
    reader: &VectorIndexReader,
    mut f: impl FnMut(RowId, DocId) -> crate::Result<()>,
) -> crate::Result<()> {
    if let Some(index) = reader.index() {
        let mut doc_ids = Vec::new();
        for cluster in 0..index.num_clusters() {
            reader.read_doc_ids(cluster, &mut doc_ids)?;
            for (row_id, &doc_id) in index.cluster_range(cluster).zip(&doc_ids) {
                f(row_id, doc_id)?;
            }
        }
    } else {
        let doc_ids = reader.row_doc_ids(0..reader.num_vectors())?;
        for (row_id, doc_id) in doc_ids.into_iter().enumerate() {
            f(row_id, doc_id)?;
        }
    }
    Ok(())
}
