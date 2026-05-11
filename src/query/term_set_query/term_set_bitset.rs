//! `BitsetFromPostings` strategy for `FastFieldTermSetQuery`.
//!
//! Performs K direct `TermDictionary::get(key)` lookups for the sorted
//! query terms and OR's each matching posting list into a single
//! segment-sized `BitSet`. The result is wrapped in `BitSetDocSet` and
//! returned as a scorer.
//!
//! Cost is approximately `K × per_lookup + bitset_iter`. `LinearScan` is
//! preferred when K is a sizable fraction of N — at that point the K
//! lookups outweigh a single O(N) hashset-probing scan. The current
//! planner gate (`bitset_max_density = 1/4`) doesn't enforce that bound
//! tightly; a follow-up will calibrate it with a wider bench sweep.
//!
//! ## Why direct lookups, not streaming-with-automaton
//!
//! An earlier version of this strategy streamed the term dictionary with
//! an FST set-membership automaton (one scan touches all K matches plus
//! every other dictionary entry). Phase 5c microbenchmarks (May 2026)
//! measured the streaming variant against direct lookups across a wide
//! K/D/N matrix and found direct lookups won every cell — frequently by
//! 10–25× on low-D columns, where the streaming walk's per-entry FST
//! traversal scales with `dict_size` (≈ N/D), not K. The streaming
//! variant's bottleneck was the lack of within-entry `can_match` early
//! termination on the SSTable streamer — bytes feed through the
//! automaton even into dead states. Direct lookups avoid the walk
//! entirely.
//!
//! See `design-doc.md` for the cost analysis. The bench captures live in
//! `tantivy/benches/term_set_queries.phase5c-rerun.txt` and
//! `tantivy/benches/term_set_queries.phase5c-20M.txt`.

use common::BitSet;

use crate::index::SegmentReader;
use crate::query::{BitSetDocSet, ConstScorer, EmptyScorer, Scorer};
use crate::schema::{Field, IndexRecordOption};
use crate::Score;

/// Build a `Scorer` that OR's the posting lists of every input value into
/// a per-segment `BitSet`. Values are interpreted via `u64::to_be_bytes`,
/// which matches the encoding `Term::serialized_value_bytes` produces for
/// numeric Terms (the same encoding the term dictionary keys on).
///
/// Caller must guarantee the field has an inverted index. The dispatch
/// site in `FastFieldTermSetWeight::scorer` enforces this; the function
/// returns an error if `reader.inverted_index(field)` fails.
pub(crate) fn bitset_from_postings_scorer(
    reader: &SegmentReader,
    field: Field,
    values: &[u64],
    boost: Score,
) -> crate::Result<Box<dyn Scorer>> {
    if values.is_empty() || reader.max_doc() == 0 {
        return Ok(Box::new(EmptyScorer));
    }

    let inverted_index = reader.inverted_index(field)?;
    let term_dict = inverted_index.terms();
    let mut bitset = BitSet::with_max_value(reader.max_doc());

    // K direct lookups. Duplicate values in `values` issue duplicate
    // lookups against the term dictionary, but each duplicate lookup
    // returns the same `TermInfo` and produces a no-op set of bitset
    // inserts (bits are already set). Pre-deduping would cost an
    // allocation + sort of K bytes; per the Phase 5c bench, that cost
    // outweighs the duplicate-lookup savings in the typical case where
    // input is already unique. Callers that want to skip duplicates
    // should dedupe upstream.
    for &v in values {
        let key = v.to_be_bytes();
        let Some(term_info) = term_dict.get(key.as_slice())? else {
            continue;
        };
        let mut block_postings = inverted_index
            .read_block_postings_from_terminfo(&term_info, IndexRecordOption::Basic)?;
        loop {
            let docs = block_postings.docs();
            if docs.is_empty() {
                break;
            }
            for &doc in docs {
                bitset.insert(doc);
            }
            block_postings.advance();
        }
    }

    Ok(Box::new(ConstScorer::new(
        BitSetDocSet::from(bitset),
        boost,
    )))
}
