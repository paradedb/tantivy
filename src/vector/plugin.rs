//! Unified vector storage plugin.
//!
//! [`VectorPlugin`] owns per-segment vector storage end-to-end:
//! - During indexing, accumulates raw vector bytes per doc; at segment finalize the buffered rows
//!   are assigned against the index-level centroid set and written as a clustered `.vec`.
//! - During merge, source rows are streamed out and reassigned against the same set.
//! - During reads, [`VectorIndexReader`](super::VectorIndexReader) opens the field's `.vec` slots
//!   via [`SegmentReader::vector_index`](crate::SegmentReader::vector_index).
//!
//! There is exactly one layout — IVF against the index-level set — so
//! there is no per-merge format decision left to make.

use super::ivf::merge_ivf;
use super::writer::VecWriter;
use super::VEC_EXT;
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};

pub struct VectorPlugin;

impl SegmentPlugin for VectorPlugin {
    fn extensions(&self) -> &[&str] {
        &[VEC_EXT]
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(VecWriter::for_schema(&ctx.segment.schema())))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        merge_ivf(&ctx)
    }
}
