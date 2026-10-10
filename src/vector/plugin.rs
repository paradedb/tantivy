//! Unified vector storage plugin.
//!
//! Commits and merges assign vectors against the index-level centroids when
//! present, or write flat vectors otherwise. The segment reader opens the
//! resulting `.vec` file; centroids and routing are shared by the index.

use super::flat::VecWriter;
use super::ivf::merge_ivf;
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
