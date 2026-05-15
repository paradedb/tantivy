//! IVF-format merge routine.
//!
//! The IVF format is one of two storage modes the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) can produce per merge.
//! This module exposes the merge body — clustering source vectors
//! into an `.ivfvec` file — so the parent plugin can call it after the
//! threshold check.
//!
//! Status: algorithm pending. The parent plugin only routes here when
//! the target segment's doc count meets the clustering threshold; that
//! branch is unreachable under the default `usize::MAX` threshold, so
//! existing tests are unaffected.

use crate::plugin::PluginMergeContext;

/// Cluster source vectors and write the target segment's `.ivfvec`.
///
/// Caller (`VectorPlugin::merge`) has already verified that the target
/// segment's doc count meets the clustering threshold. This routine is
/// unconditional — when called, it owns producing the target's IVF
/// output (centroid table, per-cluster doc-id postings, per-cluster
/// vector blob, cluster offset table) from whatever the source
/// segments expose (flat columns or IVF columns).
///
/// Sketch of the implementation:
///   1. Read source vectors. For each source, prefer the flat column
///      if present; otherwise reconstruct from the source's own IVF
///      data (vectors live inside the per-cluster blob).
///   2. Train centroids — e.g., k-means over a sample of the
///      assembled vectors. K, sample size, iteration cap, and seed
///      live on a forthcoming `IvfVecConfig`.
///   3. Assign every vector to its nearest centroid.
///   4. Serialize the IVF layout
pub(crate) fn merge_ivf(_ctx: &PluginMergeContext) -> crate::Result<()> {
    todo!("IVF clustering at merge time")
}
