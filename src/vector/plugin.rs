//! Vector storage plugin.
//!
//! [`VectorPlugin`] owns per-segment vector storage end-to-end:
//! - During indexing, accumulates raw vector bytes per doc and writes a single `.vec` file at
//!   segment finalize.
//! - During merge, copies vectors forward into a new `.vec` file.
//! - During reads, [`VectorReader`](super::reader::VectorReader) opens `.vec` and learns the
//!   storage mode from its self-describing `IdMap` header.

use super::flat::{merge_flat, FlatVecWriter};
use super::VEC_EXT;
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};

pub struct VectorPlugin;

impl SegmentPlugin for VectorPlugin {
    fn extensions(&self) -> &[&str] {
        &[VEC_EXT]
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(FlatVecWriter::for_schema(&ctx.segment.schema())))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        merge_flat(&ctx)
    }
}
