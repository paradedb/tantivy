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

/// Each [`super::Query`] must provide an estimate or explicitly return `None`.
pub trait QueryEstimate {
    /// Estimate how many documents match and how much work the query would do, using stored
    /// index statistics. Returns `(matching_docs, traversal_cost)`; cost is a rough work estimate,
    /// not a time measurement. Returns `None` when the caller must supply an estimate or fallback.
    /// Counts can include deleted documents and must not be used to exclude search results.
    ///
    /// Finding words for regex, fuzzy, and prefix queries stops at 4,096 terms (even nonmatches)
    /// or 1 MiB of term bytes, plus one extra term to check whether we finished. Reading the
    /// SSTable blocks containing those terms is also limited to 1 MiB. If we cannot finish
    /// within these limits, we return `None`.
    /// We never scan documents. Larger queries and dictionary lookups can still take more time;
    /// opening an FST dictionary for the first time can load the whole dictionary into memory.
    fn estimate_docs(&self, reader: &SegmentReader) -> crate::Result<Option<(u32, u64)>>;
}

/// Start reading at the prefix. For SSTables, check how much data we would load before reading it.
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

/// Estimate how many documents contain at least one term. Treat each term's share of documents
/// as a chance of matching, reduced by 20% to allow for terms that often occur together.
/// Multiply the chances of missing each term to estimate how many miss them all; the rest match.
/// Keep the result between the largest term count and the total document count.
/// Work adds the individual counts because each term has its own document list to visit.
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

/// Estimate documents containing every term, assuming mostly independent occurrences with a
/// small increase for words that occur together. Assume 1 in `10 * terms.len()` has the right
/// word positions. Allowing gaps multiplies that fraction by `slop + 1`, up to 100%.
/// Work adds the term counts and `10 * terms.len()` per document expected to contain every term,
/// to account for checking word positions. These factors are guesses, not measured rates.
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
