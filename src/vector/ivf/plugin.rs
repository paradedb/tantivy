//! IVF vector index plugin.
//!
//! Status: hook in place, algorithm pending. The plugin participates in
//! the merge lifecycle and short-circuits below the clustering
//! threshold (see [`IndexSettings::vector_clustering_threshold`]). The
//! clustering body itself is a TODO — when it lands, every merge whose
//! target meets the threshold will produce a `.ivfvec` file.

use std::sync::Arc;

use super::reader::IvfVecReader;
use super::writer::IvfVecPluginWriter;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};

/// Built-in plugin for clustered (IVF) vector storage.
///
/// During normal indexing this plugin is a no-op — it owns no per-doc
/// state. The actual work happens in [`SegmentPlugin::merge`], which
/// inspects the target segment's doc count and either skips (leaving
/// the segment as flat) or clusters the source vectors into the
/// inverted-file layout.
pub struct IvfVecPlugin;

impl SegmentPlugin for IvfVecPlugin {
    fn name(&self) -> &str {
        "ivf_vec"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["ivfvec"]
    }

    /// Phase 3 — runs after the basic components (fieldnorms, postings,
    /// fast fields, store at phases 0-2) and after flat vectors. Phase
    /// order doesn't affect merge correctness since each plugin reads
    /// from source readers and writes to the target independently, but
    /// keeping IVF after flat documents the conceptual layering.
    fn write_phase(&self) -> u32 {
        3
    }

    fn create_writer(&self, _ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(IvfVecPluginWriter::new()))
    }

    fn open_reader(&self, _ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        Ok(Arc::new(IvfVecReader::stub()))
    }

    /// Symmetric counterpart to [`FlatVecPlugin::merge`](crate::vector::flat::FlatVecPlugin):
    /// exactly one of the two writes a vector file per merge.
    ///
    /// - Below threshold → `Ok(())`. `FlatVecPlugin::merge` writes
    ///   `.flatvec` and this method writes nothing.
    /// - At/above threshold → clusters every source vector into an
    ///   `.ivfvec` file in the target segment. `FlatVecPlugin::merge`
    ///   has already short-circuited and written nothing.
    ///
    /// The TODO body is the clustering algorithm itself:
    ///   1. Read source vectors. For each source, prefer the flat
    ///      column if present; otherwise reconstruct from the source's
    ///      own IVF data (vectors live inside the per-cluster blob).
    ///   2. Train centroids — e.g., k-means over a sample of the
    ///      assembled vectors. K, sample size, iteration cap, and seed
    ///      live on a forthcoming `IvfVecConfig`.
    ///   3. Assign every vector to its nearest centroid.
    ///   4. Serialize the IVF layout to the target segment under
    ///      `SegmentComponent::Custom("ivfvec")`: centroid table,
    ///      per-cluster doc-id postings, per-cluster vector blob, and
    ///      a cluster-offset table.
    ///
    /// Note: this skeleton does NOT delete the source flatvec rows
    /// produced for sub-threshold ancestors — those naturally fall
    /// away because their parent segments are replaced by this merged
    /// segment, which from this point on has only the ivfvec file.
    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        let target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
        let threshold = ctx.settings.vector_clustering_threshold();
        if (target_docs as usize) < threshold {
            // FlatVecPlugin handles this merge.
            return Ok(());
        }
        // At/above threshold the IVF path is responsible for writing
        // every source vector into the target. Until the clustering
        // body lands, returning Ok(()) here would silently lose
        // vectors (FlatVecPlugin has already short-circuited above
        // the threshold). `todo!()` ensures any test that opts into
        // clustering by lowering the threshold fails loudly rather
        // than producing a corrupt index.
        //
        // The default threshold is `usize::MAX`, so this branch is
        // unreachable under default settings — existing tests are
        // unaffected.
        todo!("IVF clustering at merge time — see docstring for algorithm sketch")
    }
}
