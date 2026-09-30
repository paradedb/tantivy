use super::size_hint::{estimate_intersection, estimate_union};
use crate::SegmentReader;

pub(super) const MAX_ESTIMATED_TERMS: usize = 4096;

/// Explicit metadata-estimation support required by every [`super::Query`].
pub trait QueryEstimate {
    /// Estimates `(matching_docs, traversal_cost)` without decoding posting lists.
    /// Returns `None` when the caller must estimate this query itself or use a fallback.
    /// Counts may include deleted documents and are not bounds for filtering results.
    /// Text expansions can return `None` after 4,096 matching terms. This limits expansion,
    /// not dictionary traversal time; selective automata may still visit many dictionary entries.
    fn estimate_docs(&self, reader: &SegmentReader) -> crate::Result<Option<(u32, u64)>>;
}

pub(super) fn estimate_term_union(frequencies: &[u32], max_doc: u32) -> (u32, u64) {
    if max_doc == 0 {
        return (0, 0);
    }
    let cost = frequencies.iter().map(|&freq| u64::from(freq)).sum();
    let largest = frequencies.iter().copied().max().unwrap_or(0);
    let count = estimate_union(frequencies.iter().copied(), max_doc)
        .max(largest)
        .min(max_doc);
    (count, cost)
}

pub(super) fn estimate_phrase(terms: &[(u32, u64)], max_doc: u32, slop: u32) -> (u32, u64) {
    if terms.len() == 1 {
        return terms[0];
    }
    if terms.is_empty() || terms.iter().any(|&(count, _)| count == 0) {
        return (0, 0);
    }
    let mut frequencies: Vec<_> = terms.iter().map(|&(count, _)| count).collect();
    frequencies.sort_unstable();
    let candidates = estimate_intersection(frequencies.into_iter(), max_doc);
    // Reuse PhraseScorer's positional discount, relaxing it toward conjunction for larger slop.
    let positional_cost = 10 * terms.len() as u64;
    let fraction = ((f64::from(slop) + 1.0) / positional_cost as f64).min(1.0);
    let count = (f64::from(candidates) * fraction).ceil() as u32;
    let cost = terms
        .iter()
        .fold(0u64, |sum, &(_, cost)| sum.saturating_add(cost))
        .saturating_add(u64::from(candidates).saturating_mul(positional_cost));
    (count, cost)
}
