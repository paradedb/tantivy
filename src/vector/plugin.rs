//! Vector segment writer and merge dispatcher.

use super::flat::{merge_flat, FlatVecWriter};
use super::ivf::{merge_ivf, CENTROIDS_EXT};
use super::VEC_EXT;
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};

/// Stores flat or inverted-file vector segments.
pub struct VectorPlugin;

impl SegmentPlugin for VectorPlugin {
    fn extensions(&self) -> &[&str] {
        &[VEC_EXT, CENTROIDS_EXT]
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(FlatVecWriter::for_schema(&ctx.segment.schema())))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        let target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
        let threshold = ctx.settings.vector_clustering_threshold();
        if (target_docs as usize) < threshold {
            merge_flat(&ctx)
        } else {
            merge_ivf(&ctx, ctx.target_segment.index().ivf_clusterer())
        }
    }
}
