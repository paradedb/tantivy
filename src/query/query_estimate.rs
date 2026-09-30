use super::phrase_prefix_query::prefix_end;
use super::size_hint::{estimate_intersection, estimate_union};
use crate::termdict::{TermDictionary, TermStreamer};
use crate::SegmentReader;

pub(super) const MAX_ESTIMATED_TERMS: usize = 4096;
const MAX_ESTIMATED_TERM_BYTES: usize = 1 << 20;

pub(super) struct EstimationBudget {
    pub remaining_terms: usize,
    remaining_bytes: usize,
    #[cfg(feature = "quickwit")]
    remaining_read_bytes: usize,
}

impl Default for EstimationBudget {
    fn default() -> Self {
        Self {
            remaining_terms: MAX_ESTIMATED_TERMS,
            remaining_bytes: MAX_ESTIMATED_TERM_BYTES,
            #[cfg(feature = "quickwit")]
            remaining_read_bytes: MAX_ESTIMATED_TERM_BYTES,
        }
    }
}

impl EstimationBudget {
    pub fn consume(&mut self, term: &[u8]) -> bool {
        if self.remaining_terms == 0 || term.len() > self.remaining_bytes {
            return false;
        }
        self.remaining_terms -= 1;
        self.remaining_bytes -= term.len();
        true
    }
}

/// Explicit metadata-estimation support required by every [`super::Query`].
pub trait QueryEstimate {
    /// Estimates `(matching_docs, traversal_cost)` without decoding posting lists.
    /// Returns `None` when the caller must estimate this query itself or use a fallback.
    /// Counts may include deleted documents and are not bounds for filtering results.
    /// Expansions inspect at most 4,096 candidate terms (including nonmatches) and 1 MiB of
    /// term bytes, plus one lookahead term, before returning `None`. SSTable payload reads are
    /// also capped at 1 MiB per expansion estimate. They never scan documents.
    /// Work also depends on query size, dictionary lookups and cold metadata initialization;
    /// opening an uncached FST dictionary can load the dictionary into memory.
    fn estimate_docs(&self, reader: &SegmentReader) -> crate::Result<Option<(u32, u64)>>;
}

/// Seeks to a prefix and bounds SSTable reads before opening the unfiltered term stream.
pub(super) fn bounded_prefix_stream<'a>(
    dictionary: &'a TermDictionary,
    prefix: &[u8],
    limit: usize,
    budget: &mut EstimationBudget,
) -> std::io::Result<Option<TermStreamer<'a>>> {
    let mut stream = dictionary.range().ge(prefix);
    let end = prefix_end(prefix);
    if let Some(end) = &end {
        stream = stream.lt(end);
    }
    #[cfg(feature = "quickwit")]
    {
        use std::ops::Bound;

        use common::HasLen;

        let upper = end.as_deref().map_or(Bound::Unbounded, Bound::Excluded);
        let bytes = dictionary
            .file_slice_for_range((Bound::Included(prefix), upper), Some(limit as u64))?
            .len();
        if bytes > budget.remaining_read_bytes {
            // Opening this range would exceed the dictionary payload read budget.
            return Ok(None);
        }
        budget.remaining_read_bytes -= bytes;
        stream = stream.limit(limit as u64);
    }
    #[cfg(not(feature = "quickwit"))]
    let _ = (limit, budget);
    stream.into_stream().map(Some)
}

/// Uses the existing overlap-adjusted union heuristic, bounded below by the largest frequency;
/// traversal cost is the sum of frequencies, including overlap.
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

/// Discounts the estimated term intersection by `(slop + 1) / (10 * terms)`, capped at one;
/// cost includes term traversal and positional checks on intersection candidates.
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
