//! Unified vector storage plugin.
//!
//! [`VectorPlugin`] owns per-segment vector storage end-to-end:
//! - During indexing, accumulates raw vector bytes per doc and writes
//!   a `.flatvec` file at segment finalize.
//! - During merge, picks exactly one of two output formats by target
//!   doc count: below
//!   [`IndexSettings::vector_clustering_threshold`](crate::index::IndexSettings::vector_clustering_threshold)
//!   it copies vectors forward into a new `.flatvec`; at or above the
//!   threshold it clusters everything into a new `.ivfvec` (currently
//!   a `todo!()` body).
//! - During reads, exposes both flat and IVF views via a single
//!   [`VectorReader`](super::reader::VectorReader) — the search-side
//!   [`VectorBackend`](super::backend::VectorBackend) picks IVF when
//!   present and falls back to flat.
//!
//! Owning both `flatvec` and `ivfvec` extensions on one plugin keeps
//! the "exactly one format per segment" invariant right by construction:
//! the dispatch is one `if` inside one `merge()` method, not a
//! cross-plugin coordination problem.

use std::sync::Arc;

use super::flat::{merge_flat, FlatVecWriter, FLATVEC_EXT};
use super::ivf::merge_ivf;
use super::reader::VectorReader;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};

pub struct VectorPlugin;

impl SegmentPlugin for VectorPlugin {
    fn name(&self) -> &str {
        "vectors"
    }

    fn extensions(&self) -> Vec<&str> {
        // todo: add ivf extensions
        vec![FLATVEC_EXT]
    }

    fn write_phase(&self) -> u32 {
        2
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        // Per-doc indexing only ever produces flatvec — clustering
        // exists exclusively as a merge-time transformation.
        Ok(Box::new(FlatVecWriter::for_schema(ctx.schema)))
    }

    fn open_reader(&self, ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        Ok(Arc::new(VectorReader::open(ctx)?))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        // simple merge strategy, may change later
        // do clustering only if the target segment has more than the threshold number of docs
        let target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
        let threshold = ctx.settings.vector_clustering_threshold();
        if (target_docs as usize) < threshold {
            merge_flat(&ctx)
        } else {
            merge_ivf(&ctx)
        }
    }
}
