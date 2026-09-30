use crate::SegmentReader;

/// Explicit metadata-estimation support required by every [`super::Query`].
pub trait QueryEstimate {
    /// Estimates `(matching_docs, traversal_cost)` without decoding posting lists.
    /// Returns `None` when the caller must estimate this query itself or use a fallback.
    /// Counts may include deleted documents and are not bounds for filtering results.
    fn estimate_docs(&self, reader: &SegmentReader) -> crate::Result<Option<(u32, u64)>>;
}
