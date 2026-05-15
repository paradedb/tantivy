//! Flat-format merge routine.
//!
//! Status: skeleton. The flat merge body — copying raw vector bytes
//! from source segments into a single `.flatvec` composite file —
//! lands with the real flat reader/writer. The signature exists so
//! that the unified [`VectorPlugin`](crate::vector::VectorPlugin) can
//! call it from its threshold dispatch.

use crate::plugin::PluginMergeContext;

/// Merge source vectors into the target segment's `.flatvec` file.
///
/// Caller (`VectorPlugin::merge`) routes here when the target
/// segment's doc count is below the clustering threshold.
pub(crate) fn merge_flat(_ctx: &PluginMergeContext) -> crate::Result<()> {
    todo!("flat vector merge")
}
