//! Unified vector storage plugin.
//!
//! [`VectorPlugin`] owns per-segment vector storage end-to-end:
//! - During indexing, accumulates raw vector bytes per doc and writes flat `.vecmeta` and
//!   `.flatvec` files at segment finalize.
//! - During merge, picks exactly one of two output formats by target
//!   doc count: below
//!   [`IndexSettings::vector_clustering_threshold`](crate::index::IndexSettings::vector_clustering_threshold)
//!   it copies vectors forward into flat `.vecmeta` and `.flatvec`; at or
//!   above the threshold it writes IVF `.vecmeta`, `.assignments`, and
//!   `.vec` files.
//! - During reads, [`VectorReader`](super::reader::VectorReader) uses
//!   the segment-level `.vecmeta` marker to open the selected storage format.
//!
//! Owning both flat and IVF extensions on one plugin keeps
//! the "exactly one format per segment" invariant right by construction:
//! the dispatch is one `if` inside one `merge()` method, not a
//! cross-plugin coordination problem.

use super::flat::{merge_flat, FlatVecWriter, FLATVEC_EXT};
use super::ivf::{merge_ivf, ASSIGNMENTS_EXT, IVFVEC_EXT};
use super::meta::VECMETA_EXT;
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};

pub struct VectorPlugin;

impl SegmentPlugin for VectorPlugin {
    fn extensions(&self) -> &[&str] {
        &[FLATVEC_EXT, VECMETA_EXT, ASSIGNMENTS_EXT, IVFVEC_EXT]
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        // Per-doc indexing only ever produces flatvec — clustering
        // exists exclusively as a merge-time transformation.
        Ok(Box::new(FlatVecWriter::for_schema(&ctx.segment.schema())))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        // simple merge strategy, may change later
        // do clustering only if the target segment has more than the threshold number of docs
        let target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
        let threshold = ctx.settings.vector_clustering_threshold();
        if (target_docs as usize) < threshold {
            merge_flat(&ctx)
        } else {
            merge_ivf(&ctx, ctx.target_segment.index().ivf_clusterer())
        }
    }
}
