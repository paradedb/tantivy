//! Predicate-backed set-membership API (`TermPredicate`, `TermPredicateQuery`),
//! streaming batch evaluators, and lazy fast-field `DocSet`s shared with
//! [`TermSetWeight`].
//!
//! [`TermSetWeight`]: super::TermSetWeight

use std::ops::{Bound, Range, RangeInclusive};
use std::sync::{Arc, Mutex};
use std::{fmt, io};

use columnar::{BlockAddr, BytesColumn, Cardinality, Column, ColumnType, SortedTermSlice};
use common::bounds::BoundsRange;
use common::{BitSet, TinySet};

use super::term_set_strategy::TermSetStrategyConfig;
use super::term_set_weight::TermSetWeight;
use crate::docset::{ProbeCost, SeekDangerResult};
use crate::query::{EnableScoring, Query, Weight};
use crate::schema::Field;
use crate::{DocId, DocSet, Term, TERMINATED};

/// Number of dictionary entries evaluated per `TermPredicate` batch call.
const DICT_BATCH_LEN: usize = 4096;

/// Maximum number of documents evaluated per `PredicateColumnDocSet` window refill (`16 * 64`).
const COLUMN_WINDOW_WORDS: usize = 16;
const COLUMN_WINDOW_DOCS: u32 = (COLUMN_WINDOW_WORDS * 64) as u32;

/// Whether the segment's inverted index stores exact whole-value posting lists for this field.
///
/// Separated from `FieldType::is_indexed()` so consumers with splitting or normalizing
/// tokenizers on `columnar=true` text fields can disable postings while still executing
/// against the raw-value `StrColumn` fast field.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ExactPostings {
    Available,
    Unavailable,
}

/// Whether a `Str` field's fast-field dictionary stores exact terms matching the query's
/// value domain when an inverted index is also present.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ExactFastField {
    /// Infer from schema: allow `FieldType::Str` fast-field dictionary only when the
    /// field is not indexed or the fast-field and indexing tokenizer names match.
    #[default]
    Infer,
    /// Caller guarantees the fast-field dictionary stores exact values for the query
    /// (for example aliased identity tokenizers like `"keyword"` indexing + `"raw"` fast field,
    /// or a raw fast field on a tokenized `TEXT | FAST` field when postings are disabled via
    /// [`ExactPostings::Unavailable`]).
    Available,
}

/// Vectorized membership oracle evaluated over dictionary slices or `Column<u64>` batches.
pub trait TermPredicate: fmt::Debug + Send + Sync + 'static {
    /// Evaluate membership on `offsets.len() - 1` byte slices packed contiguously in `data`,
    /// where item `i` is `&data[offsets[i] as usize..offsets[i + 1] as usize]`.
    ///
    /// `out` is pre-sized to `num_items.div_ceil(64)` and cleared to `TinySet::EMPTY`;
    /// implementations set bit `i % 64` of `out[i / 64]` for each matching item `i`.
    fn filter_bytes(&self, data: &[u8], offsets: &[u32], out: &mut [TinySet]);

    /// Evaluate membership on `values.len()` `MonotonicallyMappableToU64`-encoded values
    /// of `col_type` (`I64`, `U64`, `F64`, `Bool`, `DateTime`).
    ///
    /// `out` is pre-sized to `values.len().div_ceil(64)` and cleared to `TinySet::EMPTY`;
    /// implementations set bit `i % 64` of `out[i / 64]` for each matching item `i`.
    fn filter_monotonic_u64s(&self, col_type: ColumnType, values: &[u64], out: &mut [TinySet]);

    /// Optional cardinality hint (`K`) when the caller knows the build-side distinct key count.
    fn estimated_distinct_keys(&self) -> Option<u64> {
        None
    }

    /// Estimated cost of evaluating this predicate on a batch of candidate values, in units of
    /// one `TermScorer::advance()`.
    ///
    /// Modeled as `setup + batch_len * per_doc`, covering only this predicate's own evaluation
    /// work (excluding Tantivy's fast-field column or dictionary-block decode cost, which
    /// `PredicateColumnDocSet` and `CachedDictPredicateDocSet` add on top):
    ///
    /// - `setup` is the fixed per-call overhead of invoking `filter_bytes` or
    ///   `filter_monotonic_u64s` (amortized across a batch). Use `ProbeSetupCost::ZERO` for
    ///   in-process Rust lookups (for example `ProbeCost::per_doc(5)` for a `HashSet`), or a value
    ///   `>= ProbeCost::BATCH_SETUP_THRESHOLD` for cross-boundary / FFI / I/O-backed predicates so
    ///   streaming operators batch probes and WAND/MaxScore defer filter evaluation past score
    ///   pruning.
    /// - `per_doc` is the marginal cost per value evaluated in the batch (relative to `1` for a
    ///   `TermScorer::advance()`, `3` for a term postings seek, and `5` for a fast-field column
    ///   read).
    fn probe_cost(&self) -> ProbeCost;
}

/// A Tantivy `Query` that filters a single `Field` through a [`TermPredicate`],
/// executing via [`TermSetWeight`].
#[derive(Clone, Debug)]
pub struct TermPredicateQuery {
    field: Field,
    predicate: Arc<dyn TermPredicate>,
    bounds: BoundsRange<Term>,
    exact_postings: ExactPostings,
    exact_fast_field: ExactFastField,
    strategy_config: TermSetStrategyConfig,
}

impl TermPredicateQuery {
    /// Create a new `TermPredicateQuery` on `field`.
    pub fn new(field: Field, predicate: Arc<dyn TermPredicate>) -> Self {
        Self {
            field,
            predicate,
            bounds: BoundsRange::new(Bound::Unbounded, Bound::Unbounded),
            exact_postings: ExactPostings::Available,
            exact_fast_field: ExactFastField::default(),
            strategy_config: TermSetStrategyConfig::default(),
        }
    }

    /// Restrict candidate values/ordinals to `bounds` before evaluating `predicate`.
    pub fn with_bounds(mut self, bounds: BoundsRange<Term>) -> Self {
        self.bounds = bounds;
        self
    }

    /// Specify whether the field's inverted index carries exact whole-value postings.
    pub fn with_exact_postings(mut self, exact_postings: ExactPostings) -> Self {
        self.exact_postings = exact_postings;
        self
    }

    /// Specify whether a `Str` fast-field dictionary stores exact values for this predicate
    /// (for example aliased identity tokenizers or a raw fast field when tokenized postings are
    /// disabled via [`ExactPostings::Unavailable`]).
    pub fn with_exact_fast_field(mut self, exact_fast_field: ExactFastField) -> Self {
        self.exact_fast_field = exact_fast_field;
        self
    }

    /// Override the strategy thresholds and `strategy_sink`.
    pub fn with_strategy_config(mut self, cfg: TermSetStrategyConfig) -> Self {
        self.strategy_config = cfg;
        self
    }
}

impl Query for TermPredicateQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let field_entry = enable_scoring.schema().get_field_entry(self.field);
        let weight = TermSetWeight::from_predicate(
            self.field,
            field_entry.field_type(),
            Arc::clone(&self.predicate),
            self.bounds.clone(),
            self.exact_postings,
            self.exact_fast_field,
            self.strategy_config.clone(),
        )?;
        Ok(Box::new(weight))
    }
}

// ---------------------------------------------------------------------------
// Shared batch evaluation & dictionary ordinal resolution helpers
// ---------------------------------------------------------------------------

fn prepare_tinyset_buf(buf: &mut Vec<TinySet>, num_items: usize) {
    buf.clear();
    buf.resize(num_items.div_ceil(64), TinySet::EMPTY);
}

#[inline]
fn try_for_each_set_bit(
    words: &[TinySet],
    len: usize,
    mut f: impl FnMut(usize) -> crate::Result<()>,
) -> crate::Result<()> {
    for (word_idx, &word) in words.iter().enumerate() {
        for bit in word {
            let idx = word_idx * 64 + bit as usize;
            if idx >= len {
                return Ok(());
            }
            f(idx)?;
        }
    }
    Ok(())
}

/// Packs streamed byte keys into `DICT_BATCH_LEN` chunks and invokes `TermPredicate::filter_bytes`.
pub(crate) struct BytesBatcher<M> {
    data: Vec<u8>,
    offsets: Vec<u32>,
    meta: Vec<M>,
    out: Vec<TinySet>,
}

impl<M> BytesBatcher<M> {
    pub(crate) fn new() -> Self {
        let mut offsets = Vec::with_capacity(DICT_BATCH_LEN + 1);
        offsets.push(0);
        Self {
            data: Vec::with_capacity(DICT_BATCH_LEN * 16),
            offsets,
            meta: Vec::with_capacity(DICT_BATCH_LEN),
            out: Vec::new(),
        }
    }

    pub(crate) fn push(
        &mut self,
        meta: M,
        key: &[u8],
        predicate: &dyn TermPredicate,
        on_match: &mut impl FnMut(&M, &[u8]) -> crate::Result<()>,
    ) -> crate::Result<()> {
        if self.meta.len() == DICT_BATCH_LEN
            || (!self.meta.is_empty()
                && self.data.len().saturating_add(key.len()) > i32::MAX as usize)
        {
            self.flush(predicate, on_match)?;
        }
        self.data.extend_from_slice(key);
        self.offsets.push(self.data.len() as u32);
        self.meta.push(meta);
        Ok(())
    }

    pub(crate) fn finish(
        &mut self,
        predicate: &dyn TermPredicate,
        on_match: &mut impl FnMut(&M, &[u8]) -> crate::Result<()>,
    ) -> crate::Result<()> {
        if !self.meta.is_empty() {
            self.flush(predicate, on_match)?;
        }
        Ok(())
    }

    fn flush(
        &mut self,
        predicate: &dyn TermPredicate,
        on_match: &mut impl FnMut(&M, &[u8]) -> crate::Result<()>,
    ) -> crate::Result<()> {
        prepare_tinyset_buf(&mut self.out, self.meta.len());
        predicate.filter_bytes(&self.data, &self.offsets, &mut self.out);
        let data = &self.data;
        let offsets = &self.offsets;
        let meta = &self.meta;
        try_for_each_set_bit(&self.out, meta.len(), |idx| {
            let slice = &data[offsets[idx] as usize..offsets[idx + 1] as usize];
            on_match(&meta[idx], slice)
        })?;
        self.data.clear();
        self.offsets.clear();
        self.offsets.push(0);
        self.meta.clear();
        Ok(())
    }
}

/// Packs streamed `u64` keys into `DICT_BATCH_LEN` chunks and invokes
/// `TermPredicate::filter_monotonic_u64s`.
pub(crate) struct U64Batcher<M> {
    vals: Vec<u64>,
    meta: Vec<M>,
    out: Vec<TinySet>,
}

impl<M> U64Batcher<M> {
    pub(crate) fn new() -> Self {
        Self {
            vals: Vec::with_capacity(DICT_BATCH_LEN),
            meta: Vec::with_capacity(DICT_BATCH_LEN),
            out: Vec::new(),
        }
    }

    pub(crate) fn push(
        &mut self,
        meta: M,
        val: u64,
        col_type: ColumnType,
        predicate: &dyn TermPredicate,
        on_match: &mut impl FnMut(&M, u64) -> crate::Result<()>,
    ) -> crate::Result<()> {
        self.vals.push(val);
        self.meta.push(meta);
        if self.vals.len() == DICT_BATCH_LEN {
            self.flush(col_type, predicate, on_match)?;
        }
        Ok(())
    }

    pub(crate) fn finish(
        &mut self,
        col_type: ColumnType,
        predicate: &dyn TermPredicate,
        on_match: &mut impl FnMut(&M, u64) -> crate::Result<()>,
    ) -> crate::Result<()> {
        if !self.vals.is_empty() {
            self.flush(col_type, predicate, on_match)?;
        }
        Ok(())
    }

    fn flush(
        &mut self,
        col_type: ColumnType,
        predicate: &dyn TermPredicate,
        on_match: &mut impl FnMut(&M, u64) -> crate::Result<()>,
    ) -> crate::Result<()> {
        prepare_tinyset_buf(&mut self.out, self.vals.len());
        predicate.filter_monotonic_u64s(col_type, &self.vals, &mut self.out);
        let vals = &self.vals;
        let meta = &self.meta;
        try_for_each_set_bit(&self.out, vals.len(), |idx| on_match(&meta[idx], vals[idx]))?;
        self.vals.clear();
        self.meta.clear();
        Ok(())
    }
}

/// Resolves a sorted slice of explicit byte keys against `bytes_col.dictionary()`,
/// decoding each touched SSTable block at most once.
pub(crate) fn resolve_explicit_dict_ordinals(
    bytes_col: &BytesColumn,
    ord_range: Range<u64>,
    sorted_keys: &[Vec<u8>],
) -> io::Result<BitSet> {
    let num_terms = bytes_col.num_terms() as u32;
    let mut matched = BitSet::with_max_value(num_terms);
    if ord_range.is_empty() || sorted_keys.is_empty() {
        return Ok(matched);
    }
    let slice = SortedTermSlice::new_assume_sorted(sorted_keys);
    for res in bytes_col.dictionary().batch_term_ord_exact(slice) {
        let (_idx, ord) = res?;
        if ord_range.contains(&ord) {
            if let Ok(ord_u32) = u32::try_from(ord) {
                matched.insert(ord_u32);
            }
        }
    }
    Ok(matched)
}

/// Streams `ord_range` of `bytes_col.dictionary()` in batches through `predicate`,
/// returning the `BitSet` of matching ordinals plus up to `bitset_key_cap` raw keys
/// (`None` once matched keys exceed `bitset_key_cap`).
///
/// Uses single-page coalesced block reads via `visit_blocks_coalesced` so page-backed
/// storage (e.g. Postgres shared buffers) stays zero-copy on interior blocks without
/// copying the entire dictionary across page boundaries.
pub(crate) fn stream_predicate_dict_ordinals(
    bytes_col: &BytesColumn,
    ord_range: Range<u64>,
    predicate: &dyn TermPredicate,
    bitset_key_cap: usize,
) -> crate::Result<(BitSet, Option<Vec<Vec<u8>>>)> {
    let dict = bytes_col.dictionary();
    let num_terms = bytes_col.num_terms() as u32;
    let mut matched_ords = BitSet::with_max_value(num_terms);
    let mut retained_keys: Option<Vec<Vec<u8>>> =
        (bitset_key_cap > 0).then(|| Vec::with_capacity(bitset_key_cap.min(256)));
    let Some(block_ids) = dict.block_ids_for_ord_range(ord_range.clone())? else {
        return Ok((matched_ords, retained_keys));
    };

    let num_blocks = block_ids.end().saturating_sub(*block_ids.start()) + 1;
    let mut blocks = Vec::with_capacity(num_blocks as usize);
    dict.visit_blocks_in_range(block_ids, |addr| blocks.push(addr))?;

    let mut batcher = BytesBatcher::new();
    let mut on_match = |&ord: &u32, key: &[u8]| -> crate::Result<()> {
        matched_ords.insert(ord);
        if let Some(keys) = retained_keys.as_mut() {
            if keys.len() < bitset_key_cap {
                keys.push(key.to_vec());
            } else {
                retained_keys = None;
            }
        }
        Ok(())
    };
    dict.visit_blocks_coalesced(&blocks, |ord, key| -> crate::Result<()> {
        if ord < ord_range.start || ord >= ord_range.end {
            return Ok(());
        }
        let Ok(ord_u32) = u32::try_from(ord) else {
            return Ok(());
        };
        batcher.push(ord_u32, key, predicate, &mut on_match)
    })?;
    batcher.finish(predicate, &mut on_match)?;
    Ok((matched_ords, retained_keys))
}

// ---------------------------------------------------------------------------
// CachedDictPredicateDocSet: candidate-probing DocSet over a dictionary fast
// field that evaluates and caches whole SSTable blocks on demand (shared
// across a query's `TermSetWeight`). It stays block-cached across probes,
// seals to `FullyPopulated` once every block in `ord_range` is decoded, and
// promotes remaining untested blocks via skip-tested single-page coalesced
// reads only when full resolution is required (`ensure_populated` on
// `seek`/`advance` or an exhaustive `TermSetWeight` call).
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum DictKeyProbe {
    SortedTerms(Arc<[Vec<u8>]>),
    Predicate(Arc<dyn TermPredicate>),
}

impl DictKeyProbe {
    /// Evaluates a batch of byte slices packed contiguously in `(data, offsets)`.
    ///
    /// Keys in `(data, offsets)` must be sorted ascending (which holds for every decoded
    /// SSTable block and ascending sequence of blocks), enabling a bounded forward scan for
    /// `SortedTerms` and a single vectorized call for `Predicate`.
    fn matches_sorted_batch(&self, data: &[u8], offsets: &[u32], out: &mut Vec<TinySet>) {
        let count = offsets.len().saturating_sub(1);
        prepare_tinyset_buf(out, count);
        if count == 0 {
            return;
        }
        match self {
            DictKeyProbe::SortedTerms(terms) => {
                if terms.is_empty() {
                    return;
                }
                let first = &data[offsets[0] as usize..offsets[1] as usize];
                let last = &data[offsets[count - 1] as usize..offsets[count] as usize];
                let lo = terms.partition_point(|t| t.as_slice() < first);
                if lo >= terms.len() {
                    return;
                }
                let hi = lo + terms[lo..].partition_point(|t| t.as_slice() <= last);
                let slice = &terms[lo..hi];
                if slice.is_empty() {
                    return;
                }
                let mut cursor = 0usize;
                for i in 0..count {
                    if cursor >= slice.len() {
                        break;
                    }
                    let key = &data[offsets[i] as usize..offsets[i + 1] as usize];
                    match slice[cursor].as_slice().cmp(key) {
                        std::cmp::Ordering::Equal => {
                            out[i / 64].insert_mut((i % 64) as u32);
                            cursor += 1;
                        }
                        std::cmp::Ordering::Greater => {}
                        std::cmp::Ordering::Less => {
                            match slice[cursor..].binary_search_by(|t| t.as_slice().cmp(key)) {
                                Ok(idx) => {
                                    out[i / 64].insert_mut((i % 64) as u32);
                                    cursor += idx + 1;
                                }
                                Err(idx) => {
                                    cursor += idx;
                                }
                            }
                        }
                    }
                }
            }
            DictKeyProbe::Predicate(pred) => pred.filter_bytes(data, offsets, out),
        }
    }

    fn resolve_all(&self, bytes_col: &BytesColumn, ord_range: Range<u64>) -> BitSet {
        match self {
            DictKeyProbe::SortedTerms(terms) => {
                resolve_explicit_dict_ordinals(bytes_col, ord_range, terms)
                    .expect("dictionary ordinal lookup should succeed")
            }
            DictKeyProbe::Predicate(pred) => {
                stream_predicate_dict_ordinals(bytes_col, ord_range, pred.as_ref(), 0)
                    .expect("dictionary predicate evaluation should succeed")
                    .0
            }
        }
    }
}

/// Reusable scratch buffers for `evaluate_blocks_into_cache`, avoiding repeated `Vec` growth
/// across batches on `CachedDictPredicateDocSet`.
#[derive(Default)]
pub(crate) struct DictEvalScratch {
    data: Vec<u8>,
    offsets: Vec<u32>,
    batch_ords: Vec<u32>,
    skipped_ords: Vec<u32>,
    out: Vec<TinySet>,
}

/// Decodes each SSTable block in `block_addrs` (must be in ascending `byte_range.start` order),
/// coalescing blocks strictly within the same single storage page on page-backed storage,
/// evaluates all in-range terms against `probe` in `DICT_BATCH_LEN`-capped batches, marks every
/// decoded ordinal in `tested_ords`, and records matching ordinals in `matched_ords`.
fn evaluate_blocks_into_cache(
    bytes_col: &BytesColumn,
    ord_range: &Range<u64>,
    probe: &DictKeyProbe,
    block_addrs: &[BlockAddr],
    tested_ords: &mut BitSet,
    matched_ords: &mut BitSet,
    scratch: &mut DictEvalScratch,
) -> io::Result<()> {
    if block_addrs.is_empty() {
        return Ok(());
    }
    let dict = bytes_col.dictionary();
    let max_ord = tested_ords.max_value();

    scratch.data.clear();
    scratch.offsets.clear();
    scratch.batch_ords.clear();
    scratch.skipped_ords.clear();
    if scratch.data.capacity() == 0 {
        scratch.data.reserve(
            block_addrs
                .len()
                .saturating_mul(4_096)
                .min(DICT_BATCH_LEN * 40),
        );
    }
    if scratch.offsets.capacity() == 0 {
        let cap = block_addrs
            .len()
            .saturating_mul(128)
            .clamp(256, DICT_BATCH_LEN);
        scratch.offsets.reserve(cap + 1);
        scratch.batch_ords.reserve(cap);
    }
    scratch.offsets.push(0u32);

    let flush =
        |scratch: &mut DictEvalScratch, tested_ords: &mut BitSet, matched_ords: &mut BitSet| {
            if !scratch.batch_ords.is_empty() {
                probe.matches_sorted_batch(&scratch.data, &scratch.offsets, &mut scratch.out);
                let _ = try_for_each_set_bit(&scratch.out, scratch.batch_ords.len(), |idx| {
                    matched_ords.insert(scratch.batch_ords[idx]);
                    Ok(())
                });
                for &ord in scratch.batch_ords.iter() {
                    tested_ords.insert(ord);
                }
                scratch.data.clear();
                scratch.offsets.clear();
                scratch.offsets.push(0u32);
                scratch.batch_ords.clear();
            }
            for &ord in scratch.skipped_ords.iter() {
                tested_ords.insert(ord);
            }
            scratch.skipped_ords.clear();
        };

    dict.visit_blocks_coalesced(block_addrs, |ord, key| -> io::Result<()> {
        let Ok(ord_u32) = u32::try_from(ord) else {
            return Ok(());
        };
        if ord_u32 >= max_ord {
            return Ok(());
        }
        if !ord_range.contains(&ord) {
            scratch.skipped_ords.push(ord_u32);
            return Ok(());
        }
        if scratch.batch_ords.len() == DICT_BATCH_LEN
            || (!scratch.batch_ords.is_empty()
                && scratch.data.len().saturating_add(key.len()) > i32::MAX as usize)
        {
            flush(scratch, tested_ords, matched_ords);
        }
        scratch.data.extend_from_slice(key);
        scratch.offsets.push(scratch.data.len() as u32);
        scratch.batch_ords.push(ord_u32);
        Ok(())
    })?;
    flush(scratch, tested_ords, matched_ords);
    Ok(())
}

/// Completes a partially populated `BlockCached` entry into a full `matched_ords` `BitSet`
/// without re-decoding any SSTable block whose `first_ordinal` is already recorded in
/// `tested_ords`.
///
/// For small explicit term sets (`terms.len() < remaining_blocks`), delegates to
/// `resolve_explicit_dict_ordinals` instead of scanning every remaining dictionary block.
pub(crate) fn resolve_remaining_into_cache(
    bytes_col: &BytesColumn,
    ord_range: &Range<u64>,
    block_ids: &RangeInclusive<u64>,
    probe: &DictKeyProbe,
    tested_ords: &mut BitSet,
    matched_ords: &mut BitSet,
    decoded_blocks: u64,
    scratch: &mut DictEvalScratch,
) -> crate::Result<BitSet> {
    let total_blocks = block_ids.end().saturating_sub(*block_ids.start()) + 1;
    let remaining_count = total_blocks.saturating_sub(decoded_blocks);
    if let DictKeyProbe::SortedTerms(terms) = probe {
        if (terms.len() as u64) < remaining_count {
            return Ok(resolve_explicit_dict_ordinals(
                bytes_col,
                ord_range.clone(),
                terms,
            )?);
        }
    }
    if remaining_count > 0 {
        let max_ord = tested_ords.max_value();
        let mut remaining_blocks = Vec::with_capacity(remaining_count as usize);
        bytes_col
            .dictionary()
            .visit_blocks_in_range(block_ids.clone(), |addr| {
                let already_tested = u32::try_from(addr.first_ordinal)
                    .is_ok_and(|first| first < max_ord && tested_ords.contains(first));
                if !already_tested {
                    remaining_blocks.push(addr);
                }
            })?;
        evaluate_blocks_into_cache(
            bytes_col,
            ord_range,
            probe,
            &remaining_blocks,
            tested_ords,
            matched_ords,
            scratch,
        )?;
    }
    Ok(std::mem::replace(matched_ords, BitSet::with_max_value(0)))
}

/// Per-segment dictionary resolution state shared across scorers created from the same
/// [`TermSetWeight`].
pub(crate) enum SegmentDictState {
    Unpopulated,
    BlockCached {
        tested_ords: BitSet,
        matched_ords: BitSet,
        /// Distinct SSTable blocks within `SegmentDictEntry::block_ids` decoded so far.
        decoded_blocks: u64,
    },
    FullyPopulated(Arc<BitSet>),
}

/// Cached per-segment metadata and ordinal resolution state on [`TermSetWeight`].
pub(crate) struct SegmentDictEntry {
    pub(crate) ord_range: Range<u64>,
    pub(crate) block_ids: Option<RangeInclusive<u64>>,
    pub(crate) state: SegmentDictState,
}

impl SegmentDictEntry {
    /// Number of SSTable blocks spanned by `ord_range`.
    #[inline]
    pub(crate) fn num_sstable_blocks(&self) -> u64 {
        let Some(block_ids) = &self.block_ids else {
            return 0;
        };
        block_ids.end().saturating_sub(*block_ids.start()) + 1
    }
}

#[inline]
fn scale_size_hint(remaining: u32, matching_keys: u64, distinct_keys: u64) -> u32 {
    if remaining == 0 || matching_keys == 0 {
        return 0;
    }
    let denom = distinct_keys.max(1);
    let numer = matching_keys.min(denom);
    let est = (u128::from(remaining) * u128::from(numer)).div_ceil(u128::from(denom));
    est.min(u128::from(remaining)) as u32
}

pub(crate) struct CachedDictPredicateDocSet {
    bytes_col: BytesColumn,
    ord_range: Range<u64>,
    probe: DictKeyProbe,
    shared: Arc<Mutex<SegmentDictEntry>>,
    populated: Option<Arc<BitSet>>,
    eval_scratch: DictEvalScratch,
    scratch_uncached_ords: Vec<u32>,
    scratch_touched_blocks: Vec<BlockAddr>,
    doc_id: DocId,
    positioned: bool,
}

impl CachedDictPredicateDocSet {
    pub(crate) fn new_unpositioned(
        bytes_col: BytesColumn,
        ord_range: Range<u64>,
        probe: DictKeyProbe,
        shared: Arc<Mutex<SegmentDictEntry>>,
    ) -> Self {
        let empty = bytes_col.num_rows() == 0 || ord_range.is_empty();
        Self {
            bytes_col,
            ord_range,
            probe,
            shared,
            populated: None,
            eval_scratch: DictEvalScratch::default(),
            scratch_uncached_ords: Vec::new(),
            scratch_touched_blocks: Vec::new(),
            doc_id: if empty { TERMINATED } else { 0 },
            positioned: empty,
        }
    }

    fn ensure_populated(&mut self) -> Arc<BitSet> {
        if let Some(matched) = &self.populated {
            return Arc::clone(matched);
        }
        let mut entry = self.shared.lock().unwrap();
        let SegmentDictEntry {
            block_ids, state, ..
        } = &mut *entry;
        let arc = match state {
            SegmentDictState::FullyPopulated(arc) => Arc::clone(arc),
            SegmentDictState::BlockCached {
                tested_ords,
                matched_ords,
                decoded_blocks,
            } => {
                let matched = if let Some(block_ids) = block_ids.as_ref() {
                    resolve_remaining_into_cache(
                        &self.bytes_col,
                        &self.ord_range,
                        block_ids,
                        &self.probe,
                        tested_ords,
                        matched_ords,
                        *decoded_blocks,
                        &mut self.eval_scratch,
                    )
                    .expect("dictionary promotion should succeed")
                } else {
                    std::mem::replace(matched_ords, BitSet::with_max_value(0))
                };
                let arc = Arc::new(matched);
                *state = SegmentDictState::FullyPopulated(Arc::clone(&arc));
                arc
            }
            SegmentDictState::Unpopulated => {
                let arc = Arc::new(
                    self.probe
                        .resolve_all(&self.bytes_col, self.ord_range.clone()),
                );
                *state = SegmentDictState::FullyPopulated(Arc::clone(&arc));
                arc
            }
        };
        drop(entry);
        self.populated = Some(Arc::clone(&arc));
        arc
    }

    #[inline]
    fn check_ord(&mut self, ord: u64) -> bool {
        if !self.ord_range.contains(&ord) {
            return false;
        }
        let Ok(ord_u32) = u32::try_from(ord) else {
            return false;
        };
        if let Some(matched) = &self.populated {
            return ord_u32 < matched.max_value() && matched.contains(ord_u32);
        }
        let mut entry = self.shared.lock().unwrap();
        let num_sstable_blocks = entry.num_sstable_blocks();
        let SegmentDictEntry { state, .. } = &mut *entry;
        match state {
            SegmentDictState::FullyPopulated(arc) => {
                let hit = ord_u32 < arc.max_value() && arc.contains(ord_u32);
                self.populated = Some(Arc::clone(arc));
                hit
            }
            SegmentDictState::BlockCached {
                tested_ords,
                matched_ords,
                decoded_blocks,
            } => {
                if tested_ords.contains(ord_u32) {
                    return matched_ords.contains(ord_u32);
                }
                let dict = self.bytes_col.dictionary();
                if let Some((block_addr, _end_ord)) = dict
                    .block_addr_and_end_ord(ord)
                    .expect("dictionary block lookup should succeed")
                {
                    evaluate_blocks_into_cache(
                        &self.bytes_col,
                        &self.ord_range,
                        &self.probe,
                        std::slice::from_ref(&block_addr),
                        tested_ords,
                        matched_ords,
                        &mut self.eval_scratch,
                    )
                    .expect("dictionary block evaluation should succeed");
                    *decoded_blocks += 1;
                    if *decoded_blocks >= num_sstable_blocks {
                        let arc =
                            Arc::new(std::mem::replace(matched_ords, BitSet::with_max_value(0)));
                        let hit = ord_u32 < arc.max_value() && arc.contains(ord_u32);
                        *state = SegmentDictState::FullyPopulated(Arc::clone(&arc));
                        self.populated = Some(arc);
                        return hit;
                    }
                } else {
                    tested_ords.insert(ord_u32);
                }
                matched_ords.contains(ord_u32)
            }
            SegmentDictState::Unpopulated => {
                let arc = Arc::new(
                    self.probe
                        .resolve_all(&self.bytes_col, self.ord_range.clone()),
                );
                let hit = ord_u32 < arc.max_value() && arc.contains(ord_u32);
                *state = SegmentDictState::FullyPopulated(Arc::clone(&arc));
                self.populated = Some(arc);
                hit
            }
        }
    }

    #[inline]
    fn check_doc(&mut self, doc: DocId) -> bool {
        let (first_ord, rest) = {
            let mut iter = self.bytes_col.ords().values_for_doc(doc);
            let Some(first) = iter.next() else {
                return false;
            };
            let rest: smallvec::SmallVec<[u64; 4]> = iter.collect();
            (first, rest)
        };
        self.check_ord(first_ord) || rest.into_iter().any(|ord| self.check_ord(ord))
    }
}

impl DocSet for CachedDictPredicateDocSet {
    fn advance(&mut self) -> DocId {
        debug_assert!(self.positioned);
        if self.doc_id == TERMINATED {
            return TERMINATED;
        }
        self.seek(self.doc_id + 1)
    }

    fn seek(&mut self, target: DocId) -> DocId {
        debug_assert!(target >= self.doc_id || self.doc_id == TERMINATED);
        self.positioned = true;
        let max_doc = self.bytes_col.num_rows();
        if target >= max_doc {
            self.doc_id = TERMINATED;
            return TERMINATED;
        }
        let matched_ords = self.ensure_populated();
        if matched_ords.len() == 0 {
            self.doc_id = TERMINATED;
            return TERMINATED;
        }
        let ords = self.bytes_col.ords();
        for next_doc in target..max_doc {
            for ord in ords.values_for_doc(next_doc) {
                if let Ok(ord_u32) = u32::try_from(ord) {
                    if ord_u32 < matched_ords.max_value() && matched_ords.contains(ord_u32) {
                        self.doc_id = next_doc;
                        return next_doc;
                    }
                }
            }
        }
        self.doc_id = TERMINATED;
        TERMINATED
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        let max_doc = self.bytes_col.num_rows();
        if target >= max_doc {
            self.doc_id = TERMINATED;
            self.positioned = true;
            return SeekDangerResult::SeekLowerBound(TERMINATED);
        }
        self.positioned = self.check_doc(target);
        if self.positioned {
            self.doc_id = target;
            SeekDangerResult::Found
        } else {
            SeekDangerResult::SeekLowerBound(target + 1)
        }
    }

    fn seek_danger_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) -> DocId {
        if docs.is_empty() {
            return 0;
        }
        let words = docs.len().div_ceil(64);
        debug_assert!(mask.len() >= words);
        let max_doc = self.bytes_col.num_rows();
        let last = *docs.last().unwrap();
        let next_bound = if last.saturating_add(1) >= max_doc {
            TERMINATED
        } else {
            last.saturating_add(1).min(TERMINATED)
        };
        self.positioned = false;

        let bytes_col = &self.bytes_col;
        let ord_range = &self.ord_range;
        let filter_mask = |matched_ords: &BitSet, mask: &mut [TinySet]| -> DocId {
            let ords = bytes_col.ords();
            for (word_idx, word) in mask[..words].iter_mut().enumerate() {
                if word.is_empty() {
                    continue;
                }
                let base = word_idx * 64;
                let mut keep = TinySet::EMPTY;
                for bit in *word {
                    let doc = docs[base + bit as usize];
                    if doc >= max_doc {
                        break;
                    }
                    let matched = ords.values_for_doc(doc).any(|ord| {
                        ord_range.contains(&ord)
                            && u32::try_from(ord).is_ok_and(|o| {
                                o < matched_ords.max_value() && matched_ords.contains(o)
                            })
                    });
                    if matched {
                        keep.insert_mut(bit);
                    }
                }
                *word = keep;
            }
            next_bound
        };

        if let Some(matched) = &self.populated {
            if matched.len() == 0 {
                mask[..words].fill(TinySet::EMPTY);
                return TERMINATED;
            }
            return filter_mask(matched, mask);
        }

        let mut entry = self.shared.lock().unwrap();
        let num_sstable_blocks = entry.num_sstable_blocks();
        let SegmentDictEntry { state, .. } = &mut *entry;
        let promoted_arc: Option<Arc<BitSet>> = match state {
            SegmentDictState::FullyPopulated(arc) => Some(Arc::clone(arc)),
            SegmentDictState::Unpopulated => {
                let arc = Arc::new(self.probe.resolve_all(bytes_col, ord_range.clone()));
                *state = SegmentDictState::FullyPopulated(Arc::clone(&arc));
                Some(arc)
            }
            SegmentDictState::BlockCached {
                tested_ords,
                matched_ords,
                decoded_blocks,
            } => {
                let uncached_ords = &mut self.scratch_uncached_ords;
                uncached_ords.clear();
                'collect: for (word_idx, &word) in mask[..words].iter().enumerate() {
                    if word.is_empty() {
                        continue;
                    }
                    let base = word_idx * 64;
                    for bit in word {
                        let doc = docs[base + bit as usize];
                        if doc >= max_doc {
                            break 'collect;
                        }
                        for ord in bytes_col.ords().values_for_doc(doc) {
                            if !ord_range.contains(&ord) {
                                continue;
                            }
                            let Ok(ord_u32) = u32::try_from(ord) else {
                                continue;
                            };
                            if !tested_ords.contains(ord_u32) {
                                uncached_ords.push(ord_u32);
                            }
                        }
                    }
                }
                if uncached_ords.is_empty() {
                    return filter_mask(matched_ords, mask);
                }
                uncached_ords.sort_unstable();
                uncached_ords.dedup();
                let dict = bytes_col.dictionary();
                let touched_blocks = &mut self.scratch_touched_blocks;
                touched_blocks.clear();
                let mut idx = 0usize;
                while idx < uncached_ords.len() {
                    let ord = u64::from(uncached_ords[idx]);
                    if let Some((block_addr, end_ord)) = dict
                        .block_addr_and_end_ord(ord)
                        .expect("dictionary block lookup should succeed")
                    {
                        touched_blocks.push(block_addr);
                        idx += 1;
                        while idx < uncached_ords.len() && u64::from(uncached_ords[idx]) < end_ord {
                            idx += 1;
                        }
                    } else {
                        tested_ords.insert(uncached_ords[idx]);
                        idx += 1;
                    }
                }
                if touched_blocks.is_empty() {
                    return filter_mask(matched_ords, mask);
                }
                evaluate_blocks_into_cache(
                    bytes_col,
                    ord_range,
                    &self.probe,
                    touched_blocks,
                    tested_ords,
                    matched_ords,
                    &mut self.eval_scratch,
                )
                .expect("dictionary block evaluation should succeed");
                *decoded_blocks += touched_blocks.len() as u64;
                if *decoded_blocks >= num_sstable_blocks {
                    let arc = Arc::new(std::mem::replace(matched_ords, BitSet::with_max_value(0)));
                    *state = SegmentDictState::FullyPopulated(Arc::clone(&arc));
                    Some(arc)
                } else {
                    return filter_mask(matched_ords, mask);
                }
            }
        };
        drop(entry);

        let arc = promoted_arc.expect("lock path either returned early or populated");
        self.populated = Some(Arc::clone(&arc));
        if arc.len() == 0 {
            mask[..words].fill(TinySet::EMPTY);
            return TERMINATED;
        }
        filter_mask(&arc, mask)
    }

    fn is_empty_in_range(&mut self, start: DocId, end: DocId) -> bool {
        start > end || start >= self.bytes_col.num_rows()
    }

    fn doc(&self) -> DocId {
        self.doc_id
    }

    fn size_hint(&self) -> u32 {
        if self.doc_id == TERMINATED {
            return 0;
        }
        let remaining = if self.positioned {
            self.bytes_col.num_rows().saturating_sub(self.doc_id)
        } else {
            self.bytes_col.num_rows()
        };
        let distinct = self.bytes_col.num_terms() as u64;
        let range_ords = self.ord_range.end.saturating_sub(self.ord_range.start);
        let matching_ords = if let Some(matched) = &self.populated {
            matched.len() as u64
        } else {
            match &self.probe {
                DictKeyProbe::SortedTerms(terms) => (terms.len() as u64).min(range_ords),
                DictKeyProbe::Predicate(pred) => pred
                    .estimated_distinct_keys()
                    .map_or(range_ords, |k| k.min(range_ords)),
            }
        };
        scale_size_hint(remaining, matching_ords, distinct)
    }

    fn cost(&self) -> crate::DocSetCost {
        let probe = if self.populated.is_some() {
            crate::ProbeCost::FAST_FIELD_LOOKUP
        } else {
            match &self.probe {
                DictKeyProbe::SortedTerms(_) => crate::ProbeCost::DICT_BLOCK,
                DictKeyProbe::Predicate(pred) => {
                    crate::ProbeCost::DICT_BLOCK.saturating_add(pred.probe_cost())
                }
            }
        };
        let drive =
            crate::DriveCost(u64::from(self.bytes_col.num_rows()).saturating_mul(probe.per_doc.0));
        crate::DocSetCost::new(drive, probe)
    }
}

// ---------------------------------------------------------------------------
// PredicateColumnDocSet — vectorized window scan + candidate probing over
// a numeric `Column<u64>`
// ---------------------------------------------------------------------------

pub(crate) struct PredicateColumnDocSet {
    column: Column<u64>,
    col_type: ColumnType,
    value_range: RangeInclusive<u64>,
    doc_bounds: Range<DocId>,
    predicate: Arc<dyn TermPredicate>,
    window: Range<DocId>,
    window_mask: [TinySet; COLUMN_WINDOW_WORDS],
    scratch_doc_ids: Vec<DocId>,
    scratch_opt_vals: Vec<Option<u64>>,
    scratch_vals: Vec<u64>,
    scratch_val_docs: Vec<u32>,
    scratch_out: Vec<TinySet>,
    doc_id: DocId,
    positioned: bool,
}

impl PredicateColumnDocSet {
    pub(crate) fn new(
        column: Column<u64>,
        col_type: ColumnType,
        value_range: RangeInclusive<u64>,
        doc_bounds: Range<DocId>,
        predicate: Arc<dyn TermPredicate>,
    ) -> Self {
        let mut s = Self::new_unpositioned(column, col_type, value_range, doc_bounds, predicate);
        if s.doc_id != TERMINATED {
            let start = s.doc_bounds.start;
            s.seek(start);
        }
        s
    }

    pub(crate) fn new_unpositioned(
        column: Column<u64>,
        col_type: ColumnType,
        value_range: RangeInclusive<u64>,
        doc_bounds: Range<DocId>,
        predicate: Arc<dyn TermPredicate>,
    ) -> Self {
        let empty = doc_bounds.is_empty() || column.num_docs() == 0;
        Self {
            column,
            col_type,
            value_range,
            doc_bounds,
            predicate,
            window: 0..0,
            window_mask: [TinySet::EMPTY; COLUMN_WINDOW_WORDS],
            scratch_doc_ids: Vec::new(),
            scratch_opt_vals: Vec::new(),
            scratch_vals: Vec::new(),
            scratch_val_docs: Vec::new(),
            scratch_out: Vec::new(),
            doc_id: if empty { TERMINATED } else { 0 },
            positioned: empty,
        }
    }

    fn fill_window(&mut self, base: DocId) {
        let end = base
            .saturating_add(COLUMN_WINDOW_DOCS)
            .min(self.doc_bounds.end);
        self.window = base..end;
        if base >= end {
            return;
        }
        let len = (end - base) as usize;
        let num_words = len.div_ceil(64);
        self.window_mask[..num_words].fill(TinySet::EMPTY);

        let unconstrained = *self.value_range.start() <= self.column.min_value()
            && *self.value_range.end() >= self.column.max_value();

        match self.column.get_cardinality() {
            Cardinality::Full if unconstrained => {
                // Direct 1-to-1 contiguous decode into `scratch_vals` and `window_mask` with no
                // `DocId` buffer, `Option<u64>` wrapper, range check, or per-bit scatter loop.
                self.scratch_vals.resize(len, 0);
                self.column
                    .values
                    .get_range(u64::from(base), &mut self.scratch_vals);
                self.predicate.filter_monotonic_u64s(
                    self.col_type,
                    &self.scratch_vals,
                    &mut self.window_mask[..num_words],
                );
                return;
            }
            Cardinality::Full => {
                self.scratch_opt_vals.clear();
                self.scratch_vals.resize(len, 0);
                self.column
                    .values
                    .get_range(u64::from(base), &mut self.scratch_vals);
                self.scratch_val_docs.clear();
                let mut compact_len = 0usize;
                for offset in 0..len {
                    let v = self.scratch_vals[offset];
                    if self.value_range.contains(&v) {
                        self.scratch_vals[compact_len] = v;
                        self.scratch_val_docs.push(offset as u32);
                        compact_len += 1;
                    }
                }
                self.scratch_vals.truncate(compact_len);
            }
            Cardinality::Optional => {
                self.scratch_vals.clear();
                self.scratch_val_docs.clear();
                self.scratch_doc_ids.clear();
                self.scratch_doc_ids.extend(base..end);
                self.scratch_opt_vals.resize(len, None);
                self.column
                    .first_vals(&self.scratch_doc_ids, &mut self.scratch_opt_vals);
                for (offset, maybe_val) in self.scratch_opt_vals.iter().copied().enumerate() {
                    if let Some(v) = maybe_val {
                        if self.value_range.contains(&v) {
                            self.scratch_vals.push(v);
                            self.scratch_val_docs.push(offset as u32);
                        }
                    }
                }
            }
            Cardinality::Multivalued => {
                self.scratch_vals.clear();
                self.scratch_val_docs.clear();
                for offset in 0..len {
                    for v in self.column.values_for_doc(base + offset as u32) {
                        if self.value_range.contains(&v) {
                            self.scratch_vals.push(v);
                            self.scratch_val_docs.push(offset as u32);
                        }
                    }
                }
            }
        }

        if self.scratch_vals.is_empty() {
            return;
        }
        prepare_tinyset_buf(&mut self.scratch_out, self.scratch_vals.len());
        self.predicate.filter_monotonic_u64s(
            self.col_type,
            &self.scratch_vals,
            &mut self.scratch_out,
        );
        let val_docs = &self.scratch_val_docs;
        let window_mask = &mut self.window_mask;
        let _ = try_for_each_set_bit(&self.scratch_out, val_docs.len(), |idx| {
            let doc_offset = val_docs[idx];
            window_mask[(doc_offset / 64) as usize].insert_mut(doc_offset % 64);
            Ok(())
        });
    }

    #[inline]
    fn find_in_window(&self, from_doc: DocId) -> Option<DocId> {
        if from_doc >= self.window.end || from_doc < self.window.start {
            return None;
        }
        let num_words = (self.window.end - self.window.start).div_ceil(64) as usize;
        let rel = from_doc - self.window.start;
        let first_word = (rel / 64) as usize;
        let masked_first =
            self.window_mask[first_word].intersect(TinySet::range_greater_or_equal(rel % 64));
        if let Some(bit) = masked_first.into_iter().next() {
            let doc = self.window.start + (first_word as u32) * 64 + bit;
            return (doc < self.window.end).then_some(doc);
        }
        for (word_idx, &word) in self.window_mask[..num_words]
            .iter()
            .enumerate()
            .skip(first_word + 1)
        {
            if let Some(bit) = word.into_iter().next() {
                let doc = self.window.start + (word_idx as u32) * 64 + bit;
                return (doc < self.window.end).then_some(doc);
            }
        }
        None
    }
}

impl DocSet for PredicateColumnDocSet {
    #[inline]
    fn advance(&mut self) -> DocId {
        debug_assert!(self.positioned);
        if self.doc_id == TERMINATED {
            return TERMINATED;
        }
        let next = self.doc_id + 1;
        if next >= self.window.start && next < self.window.end {
            if let Some(hit) = self.find_in_window(next) {
                self.doc_id = hit;
                return hit;
            }
            return self.seek(self.window.end);
        }
        self.seek(next)
    }

    fn seek(&mut self, target: DocId) -> DocId {
        debug_assert!(target >= self.doc_id || self.doc_id == TERMINATED);
        self.positioned = true;
        let mut cursor = target.max(self.doc_bounds.start);
        if cursor >= self.doc_bounds.end {
            self.doc_id = TERMINATED;
            return TERMINATED;
        }
        if cursor >= self.window.start && cursor < self.window.end {
            if let Some(hit) = self.find_in_window(cursor) {
                self.doc_id = hit;
                return hit;
            }
            cursor = self.window.end;
        }
        while cursor < self.doc_bounds.end {
            self.fill_window(cursor);
            if let Some(hit) = self.find_in_window(cursor) {
                self.doc_id = hit;
                return hit;
            }
            cursor = self.window.end;
        }
        self.doc_id = TERMINATED;
        TERMINATED
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        if target >= self.doc_bounds.end {
            self.doc_id = TERMINATED;
            self.positioned = true;
            return SeekDangerResult::SeekLowerBound(TERMINATED);
        }
        if target < self.doc_bounds.start {
            return SeekDangerResult::SeekLowerBound(self.doc_bounds.start);
        }
        if target >= self.window.start && target < self.window.end {
            let rel = target - self.window.start;
            let hit = self.window_mask[(rel / 64) as usize].contains(rel % 64);
            self.positioned = hit;
            return if hit {
                self.doc_id = target;
                SeekDangerResult::Found
            } else {
                let next_bound = self.find_in_window(target + 1).unwrap_or(self.window.end);
                if next_bound >= self.doc_bounds.end {
                    self.doc_id = TERMINATED;
                    self.positioned = true;
                    SeekDangerResult::SeekLowerBound(TERMINATED)
                } else {
                    SeekDangerResult::SeekLowerBound(next_bound)
                }
            };
        }

        self.window = 0..0;
        let hit = match self.column.get_cardinality() {
            Cardinality::Full | Cardinality::Optional => {
                self.column.first(target).is_some_and(|v| {
                    if !self.value_range.contains(&v) {
                        return false;
                    }
                    let mut out = [TinySet::EMPTY; 1];
                    self.predicate.filter_monotonic_u64s(
                        self.col_type,
                        std::slice::from_ref(&v),
                        &mut out,
                    );
                    out[0].contains(0)
                })
            }
            Cardinality::Multivalued => {
                self.scratch_vals.clear();
                for v in self.column.values_for_doc(target) {
                    if self.value_range.contains(&v) {
                        self.scratch_vals.push(v);
                    }
                }
                if self.scratch_vals.is_empty() {
                    false
                } else {
                    prepare_tinyset_buf(&mut self.scratch_out, self.scratch_vals.len());
                    self.predicate.filter_monotonic_u64s(
                        self.col_type,
                        &self.scratch_vals,
                        &mut self.scratch_out,
                    );
                    self.scratch_out.iter().any(|w| !w.is_empty())
                }
            }
        };
        self.positioned = hit;
        if hit {
            self.doc_id = target;
            SeekDangerResult::Found
        } else {
            let next_bound = target.saturating_add(1);
            if next_bound >= self.doc_bounds.end {
                self.doc_id = TERMINATED;
                self.positioned = true;
                SeekDangerResult::SeekLowerBound(TERMINATED)
            } else {
                SeekDangerResult::SeekLowerBound(next_bound)
            }
        }
    }

    fn seek_danger_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) -> DocId {
        if docs.is_empty() {
            return 0;
        }
        let words = docs.len().div_ceil(64);
        debug_assert!(mask.len() >= words);
        let first = docs[0];
        let last = *docs.last().unwrap();
        let next_bound =
            if self.doc_bounds.is_empty() || last.saturating_add(1) >= self.doc_bounds.end {
                TERMINATED
            } else if last.saturating_add(1) < self.doc_bounds.start {
                self.doc_bounds.start
            } else {
                last.saturating_add(1).min(TERMINATED)
            };
        self.positioned = false;

        if self.doc_bounds.is_empty()
            || last < self.doc_bounds.start
            || first >= self.doc_bounds.end
        {
            mask[..words].fill(TinySet::EMPTY);
            return next_bound;
        }

        let active_count: usize = mask[..words].iter().map(|w| w.len() as usize).sum();
        if active_count == 0 {
            return next_bound;
        }

        let all_active_and_in_bounds = active_count == docs.len()
            && first >= self.doc_bounds.start
            && last < self.doc_bounds.end;
        let unconstrained = *self.value_range.start() <= self.column.min_value()
            && *self.value_range.end() >= self.column.max_value();

        match self.column.get_cardinality() {
            Cardinality::Full if all_active_and_in_bounds && unconstrained => {
                self.scratch_vals.resize(docs.len(), 0);
                if last - first + 1 == docs.len() as u32 {
                    self.column
                        .values
                        .get_range(u64::from(first), &mut self.scratch_vals);
                } else {
                    self.column.values.get_vals(docs, &mut self.scratch_vals);
                }
                mask[..words].fill(TinySet::EMPTY);
                self.predicate.filter_monotonic_u64s(
                    self.col_type,
                    &self.scratch_vals,
                    &mut mask[..words],
                );
                return next_bound;
            }
            Cardinality::Full if all_active_and_in_bounds => {
                self.scratch_vals.resize(docs.len(), 0);
                if last - first + 1 == docs.len() as u32 {
                    self.column
                        .values
                        .get_range(u64::from(first), &mut self.scratch_vals);
                } else {
                    self.column.values.get_vals(docs, &mut self.scratch_vals);
                }
                self.scratch_val_docs.clear();
                let mut compact = 0usize;
                for pos in 0..docs.len() {
                    let v = self.scratch_vals[pos];
                    if self.value_range.contains(&v) {
                        self.scratch_vals[compact] = v;
                        self.scratch_val_docs.push(pos as u32);
                        compact += 1;
                    }
                }
                self.scratch_vals.truncate(compact);
            }
            Cardinality::Full => {
                self.scratch_doc_ids.clear();
                self.scratch_val_docs.clear();
                for (word_idx, &word) in mask[..words].iter().enumerate() {
                    if word.is_empty() {
                        continue;
                    }
                    let base = word_idx * 64;
                    for bit in word {
                        let pos = base + bit as usize;
                        let doc = docs[pos];
                        if doc >= self.doc_bounds.start && doc < self.doc_bounds.end {
                            self.scratch_doc_ids.push(doc);
                            self.scratch_val_docs.push(pos as u32);
                        }
                    }
                }
                if self.scratch_doc_ids.is_empty() {
                    mask[..words].fill(TinySet::EMPTY);
                    return next_bound;
                }
                self.scratch_vals.resize(self.scratch_doc_ids.len(), 0);
                self.column
                    .values
                    .get_vals(&self.scratch_doc_ids, &mut self.scratch_vals);
                if !unconstrained {
                    let mut compact = 0usize;
                    for idx in 0..self.scratch_vals.len() {
                        let v = self.scratch_vals[idx];
                        if self.value_range.contains(&v) {
                            self.scratch_vals[compact] = v;
                            self.scratch_val_docs[compact] = self.scratch_val_docs[idx];
                            compact += 1;
                        }
                    }
                    self.scratch_vals.truncate(compact);
                    self.scratch_val_docs.truncate(compact);
                }
            }
            Cardinality::Optional if all_active_and_in_bounds => {
                self.scratch_opt_vals.resize(docs.len(), None);
                self.column.first_vals(docs, &mut self.scratch_opt_vals);
                self.scratch_vals.clear();
                self.scratch_val_docs.clear();
                for (pos, maybe_val) in self.scratch_opt_vals.iter().copied().enumerate() {
                    if let Some(v) = maybe_val {
                        if self.value_range.contains(&v) {
                            self.scratch_vals.push(v);
                            self.scratch_val_docs.push(pos as u32);
                        }
                    }
                }
            }
            Cardinality::Optional => {
                self.scratch_doc_ids.clear();
                self.scratch_val_docs.clear();
                for (word_idx, &word) in mask[..words].iter().enumerate() {
                    if word.is_empty() {
                        continue;
                    }
                    let base = word_idx * 64;
                    for bit in word {
                        let pos = base + bit as usize;
                        let doc = docs[pos];
                        if doc >= self.doc_bounds.start && doc < self.doc_bounds.end {
                            self.scratch_doc_ids.push(doc);
                            self.scratch_val_docs.push(pos as u32);
                        }
                    }
                }
                if self.scratch_doc_ids.is_empty() {
                    mask[..words].fill(TinySet::EMPTY);
                    return next_bound;
                }
                self.scratch_opt_vals
                    .resize(self.scratch_doc_ids.len(), None);
                self.column
                    .first_vals(&self.scratch_doc_ids, &mut self.scratch_opt_vals);
                self.scratch_vals.clear();
                let mut compact = 0usize;
                for (idx, maybe_val) in self.scratch_opt_vals.iter().copied().enumerate() {
                    if let Some(v) = maybe_val {
                        if self.value_range.contains(&v) {
                            self.scratch_vals.push(v);
                            self.scratch_val_docs[compact] = self.scratch_val_docs[idx];
                            compact += 1;
                        }
                    }
                }
                self.scratch_val_docs.truncate(compact);
            }
            Cardinality::Multivalued => {
                self.scratch_vals.clear();
                self.scratch_val_docs.clear();
                for (word_idx, &word) in mask[..words].iter().enumerate() {
                    if word.is_empty() {
                        continue;
                    }
                    let base = word_idx * 64;
                    for bit in word {
                        let pos = base + bit as usize;
                        let doc = docs[pos];
                        if doc >= self.doc_bounds.start && doc < self.doc_bounds.end {
                            for v in self.column.values_for_doc(doc) {
                                if self.value_range.contains(&v) {
                                    self.scratch_vals.push(v);
                                    self.scratch_val_docs.push(pos as u32);
                                }
                            }
                        }
                    }
                }
            }
        }

        mask[..words].fill(TinySet::EMPTY);
        if self.scratch_vals.is_empty() {
            return next_bound;
        }
        prepare_tinyset_buf(&mut self.scratch_out, self.scratch_vals.len());
        self.predicate.filter_monotonic_u64s(
            self.col_type,
            &self.scratch_vals,
            &mut self.scratch_out,
        );
        let val_docs = &self.scratch_val_docs;
        let _ = try_for_each_set_bit(&self.scratch_out, val_docs.len(), |idx| {
            let pos = val_docs[idx];
            mask[(pos / 64) as usize].insert_mut(pos % 64);
            Ok(())
        });
        next_bound
    }

    fn is_empty_in_range(&mut self, start: DocId, end: DocId) -> bool {
        if start > end || start >= self.doc_bounds.end || end < self.doc_bounds.start {
            return true;
        }
        if start >= self.window.start && end < self.window.end {
            return self.find_in_window(start).is_none_or(|hit| hit > end);
        }
        false
    }

    fn doc(&self) -> DocId {
        self.doc_id
    }

    fn size_hint(&self) -> u32 {
        if self.doc_id == TERMINATED {
            return 0;
        }
        let remaining = if self.positioned {
            self.doc_bounds
                .end
                .saturating_sub(self.doc_id.max(self.doc_bounds.start))
        } else {
            self.doc_bounds.end.saturating_sub(self.doc_bounds.start)
        };
        let Some(est_keys) = self.predicate.estimated_distinct_keys() else {
            return remaining;
        };
        let window_docs = u64::from(self.doc_bounds.end.saturating_sub(self.doc_bounds.start));
        let distinct_upper = if self.col_type == ColumnType::F64 {
            window_docs
        } else {
            let span = self
                .value_range
                .end()
                .saturating_sub(*self.value_range.start())
                .saturating_add(1);
            window_docs.min(span)
        };
        scale_size_hint(remaining, est_keys, distinct_upper)
    }

    fn cost(&self) -> crate::DocSetCost {
        let window_docs = u64::from(self.doc_bounds.end.saturating_sub(self.doc_bounds.start));
        let probe = crate::ProbeCost::FAST_FIELD.saturating_add(self.predicate.probe_cost());
        crate::DocSetCost::new(
            crate::DriveCost(window_docs.saturating_mul(probe.per_doc.0)),
            probe,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::ops::{Bound, Range};
    use std::path::Path;
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use columnar::{ColumnType, MonotonicallyMappableToU64};
    use common::bounds::BoundsRange;
    use common::{HasLen, OwnedBytes, TinySet};
    use proptest::prelude::*;
    use rustc_hash::FxHashSet;

    use super::{ExactFastField, ExactPostings, TermPredicate, TermPredicateQuery};
    use crate::collector::{DocSetCollector, TopDocs};
    use crate::directory::error::{DeleteError, OpenReadError, OpenWriteError};
    use crate::directory::{
        Directory, FileHandle, RamDirectory, WatchCallback, WatchHandle, WritePtr,
    };
    use crate::query::{
        AllQuery, BooleanQuery, CandidateHint, ConstScoreQuery, EnableScoring,
        FastFieldTermSetQuery, Occur, Query, RangeQuery, StrategyTag, TermQuery, TermSetQuery,
        TermSetStrategyConfig,
    };
    use crate::schema::{
        IndexRecordOption, NumericOptions, SchemaBuilder, TextFieldIndexing, TextOptions, FAST,
        STRING, TEXT,
    };
    use crate::{DocId, Index, IndexSettings, IndexSortByField, Order, Term, TERMINATED};

    #[derive(Debug, Default)]
    struct EvalTelemetry {
        calls: AtomicUsize,
        items: AtomicUsize,
    }

    #[derive(Debug)]
    struct SetPredicate {
        bytes_set: FxHashSet<Vec<u8>>,
        i64_set: FxHashSet<i64>,
        estimated_keys: Option<u64>,
        probe_cost: crate::ProbeCost,
        bytes_telemetry: EvalTelemetry,
        u64_telemetry: EvalTelemetry,
    }

    impl SetPredicate {
        fn from_strings<'a>(items: impl IntoIterator<Item = &'a str>) -> Self {
            Self {
                bytes_set: items.into_iter().map(|s| s.as_bytes().to_vec()).collect(),
                i64_set: FxHashSet::default(),
                estimated_keys: None,
                probe_cost: crate::ProbeCost::per_doc(5),
                bytes_telemetry: EvalTelemetry::default(),
                u64_telemetry: EvalTelemetry::default(),
            }
        }

        fn from_i64s(items: impl IntoIterator<Item = i64>) -> Self {
            Self {
                bytes_set: FxHashSet::default(),
                i64_set: items.into_iter().collect(),
                estimated_keys: None,
                probe_cost: crate::ProbeCost::per_doc(5),
                bytes_telemetry: EvalTelemetry::default(),
                u64_telemetry: EvalTelemetry::default(),
            }
        }

        fn with_estimated_distinct_keys(mut self, count: u64) -> Self {
            self.estimated_keys = Some(count);
            self
        }

        fn with_probe_cost(mut self, probe_cost: crate::ProbeCost) -> Self {
            self.probe_cost = probe_cost;
            self
        }

        fn total_calls(&self) -> usize {
            self.bytes_telemetry.calls.load(Ordering::Relaxed)
                + self.u64_telemetry.calls.load(Ordering::Relaxed)
        }
    }

    impl TermPredicate for SetPredicate {
        fn filter_bytes(&self, data: &[u8], offsets: &[u32], out: &mut [TinySet]) {
            let count = offsets.len().saturating_sub(1);
            self.bytes_telemetry.calls.fetch_add(1, Ordering::Relaxed);
            self.bytes_telemetry
                .items
                .fetch_add(count, Ordering::Relaxed);
            for i in 0..count {
                let slice = &data[offsets[i] as usize..offsets[i + 1] as usize];
                if self.bytes_set.contains(slice) {
                    out[i / 64].insert_mut((i % 64) as u32);
                }
            }
        }

        fn filter_monotonic_u64s(&self, col_type: ColumnType, values: &[u64], out: &mut [TinySet]) {
            assert_eq!(col_type, ColumnType::I64);
            self.u64_telemetry.calls.fetch_add(1, Ordering::Relaxed);
            self.u64_telemetry
                .items
                .fetch_add(values.len(), Ordering::Relaxed);
            for (i, &raw) in values.iter().enumerate() {
                if self.i64_set.contains(&i64::from_u64(raw)) {
                    out[i / 64].insert_mut((i % 64) as u32);
                }
            }
        }

        fn estimated_distinct_keys(&self) -> Option<u64> {
            self.estimated_keys
        }

        fn probe_cost(&self) -> crate::ProbeCost {
            self.probe_cost
        }
    }

    fn assert_narrow_runs_before_broad(
        searcher: &crate::Searcher,
        field: crate::schema::Field,
        driver_docs: &[DocId],
        make_pair: impl Fn() -> (Arc<SetPredicate>, Arc<SetPredicate>),
        expected_matches: &[DocId],
    ) -> crate::Result<()> {
        let segment = searcher.segment_reader(0);
        let scoring = EnableScoring::disabled_from_searcher(searcher);
        let hint = CandidateHint::Estimated(driver_docs.len().max(1) as u32);
        let build = |pred: Arc<SetPredicate>| {
            TermPredicateQuery::new(field, pred)
                .with_exact_postings(ExactPostings::Unavailable)
                .weight(scoring)?
                .scorer_for_candidates_unpositioned(segment, 1.0, hint)
        };
        for reverse in [false, true] {
            let (narrow, broad) = make_pair();
            let (narrow_s, broad_s) = (build(narrow)?, build(broad.clone())?);
            assert!(narrow_s.size_hint() < broad_s.size_hint());
            let driver = crate::query::UnpositionedScorer::from_positioned(Box::new(
                crate::query::ConstScorer::new(
                    crate::query::VecDocSet::from(driver_docs.to_vec()),
                    1.0,
                ),
            ));
            let scorers = if reverse {
                vec![broad_s, narrow_s, driver]
            } else {
                vec![driver, narrow_s, broad_s]
            };
            let mut inter = crate::query::intersection::intersect_candidate_scorers(
                scorers,
                segment.num_docs(),
            );
            let mut matched = Vec::new();
            while inter.doc() != TERMINATED {
                matched.push(inter.doc());
                inter.advance();
            }
            assert_eq!(matched, expected_matches);
            assert!(broad.total_calls() <= expected_matches.len());
        }
        Ok(())
    }

    #[test]
    fn dictionary_strategies_and_selective_candidate_bypass() -> crate::Result<()> {
        let mut sb = SchemaBuilder::new();
        let tag_field = sb.add_text_field("tag", STRING);
        let str_fast_idx = sb.add_text_field("str_both", STRING | FAST);
        let str_fast_only = sb.add_text_field("str_fast_only", FAST);
        let str_idx_only = sb.add_text_field(
            "str_idx_only",
            TextOptions::default().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_index_option(IndexRecordOption::Basic),
            ),
        );
        let index = Index::create_in_ram(sb.build());
        {
            let mut writer = index.writer_for_tests()?;
            for i in 0..3000u32 {
                let s = format!("key-{i:06}-padding-to-grow-dictionary-blocks");
                let tag = if i == 42 || i == 777 {
                    "rare"
                } else {
                    "common"
                };
                writer.add_document(doc!(
                    tag_field => tag,
                    str_fast_idx => s.as_str(),
                    str_fast_only => s.as_str(),
                    str_idx_only => s.as_str(),
                ))?;
            }
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // 1. Selective match on `STRING | FAST` (`1/3000 <= 1/200`) -> Bitset
        let sink = Arc::new(AtomicU8::new(0));
        let pred = Arc::new(SetPredicate::from_strings([
            "key-000042-padding-to-grow-dictionary-blocks",
        ]));
        let q = TermPredicateQuery::new(str_fast_idx, pred).with_strategy_config(
            TermSetStrategyConfig {
                strategy_sink: Some(sink.clone()),
                ..Default::default()
            },
        );
        assert_eq!(searcher.search(&q, &DocSetCollector)?.len(), 1);
        assert_eq!(sink.load(Ordering::Relaxed), StrategyTag::Bitset as u8);

        // 2. `FAST`-only string column works for both `TermPredicateQuery` and `TermSetQuery`
        let targets: Vec<String> = (100..200)
            .map(|i| format!("key-{i:06}-padding-to-grow-dictionary-blocks"))
            .collect();
        let pred_dense = Arc::new(SetPredicate::from_strings(
            targets.iter().map(String::as_str),
        ));
        let hits_both = searcher.search(
            &TermPredicateQuery::new(str_fast_idx, pred_dense.clone()),
            &DocSetCollector,
        )?;
        let hits_fast_pred = searcher.search(
            &TermPredicateQuery::new(str_fast_only, pred_dense.clone())
                .with_exact_postings(ExactPostings::Unavailable),
            &DocSetCollector,
        )?;
        let hits_fast_termset = searcher.search(
            &FastFieldTermSetQuery::new(
                targets
                    .iter()
                    .map(|s| Term::from_field_text(str_fast_only, s)),
            ),
            &DocSetCollector,
        )?;
        let hits_idx_only = searcher.search(
            &TermPredicateQuery::new(str_idx_only, pred_dense),
            &DocSetCollector,
        )?;
        assert_eq!(hits_both.len(), 100);
        assert_eq!(hits_both, hits_fast_pred);
        assert_eq!(hits_both, hits_fast_termset);
        assert_eq!(hits_both, hits_idx_only);

        // 3. Combined with a very selective `TermQuery("rare")` (`C = 2 << num_blocks`):
        // decodes at most the 2 touched SSTable blocks rather than streaming all 3,000
        // dictionary entries.
        let selective_pred = Arc::new(SetPredicate::from_strings([
            "key-000777-padding-to-grow-dictionary-blocks",
        ]));
        let bool_q = BooleanQuery::new(vec![
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    Term::from_field_text(tag_field, "rare"),
                    IndexRecordOption::Basic,
                )),
            ),
            (
                Occur::Must,
                Box::new(TermPredicateQuery::new(
                    str_fast_idx,
                    selective_pred.clone(),
                )),
            ),
        ]);
        assert_eq!(searcher.search(&bool_q, &DocSetCollector)?.len(), 1);
        assert!(selective_pred.bytes_telemetry.calls.load(Ordering::Relaxed) <= 2);
        assert!(selective_pred.bytes_telemetry.items.load(Ordering::Relaxed) < 1_000);

        // 4. Selective dictionary predicate sorts ahead of a broad one regardless of order.
        let broad_keys: Vec<String> = (0..2_400u32)
            .map(|i| format!("key-{i:06}-padding-to-grow-dictionary-blocks"))
            .collect();
        assert_narrow_runs_before_broad(
            &searcher,
            str_fast_only,
            &[42, 777],
            || {
                (
                    Arc::new(
                        SetPredicate::from_strings([
                            "key-000777-padding-to-grow-dictionary-blocks",
                        ])
                        .with_estimated_distinct_keys(1),
                    ),
                    Arc::new(
                        SetPredicate::from_strings(broad_keys.iter().map(String::as_str))
                            .with_estimated_distinct_keys(2_400),
                    ),
                )
            },
            &[777],
        )?;
        Ok(())
    }

    #[test]
    fn sorted_string_gallop_for_both_term_set_and_predicate_plus_bounds() -> crate::Result<()> {
        let mut sb = SchemaBuilder::new();
        let str_field = sb.add_text_field("k_str", STRING | FAST);
        let num_field =
            sb.add_i64_field("k_num", NumericOptions::default().set_fast().set_indexed());
        let index = Index::builder()
            .schema(sb.build())
            .settings(IndexSettings {
                sort_by_field: Some(IndexSortByField {
                    field: "k_str".to_string(),
                    order: Order::Asc,
                }),
                ..Default::default()
            })
            .create_in_ram()?;
        {
            let mut writer = index.writer_for_tests()?;
            for i in 0..2000i64 {
                writer.add_document(doc!(str_field => format!("s-{i:05}"), num_field => i * 10))?;
            }
            writer.commit()?;
        }

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Both `TermPredicateQuery` and `TermSetQuery` commit to `Gallop` on sorted `STRING |
        // FAST`.
        let pred_sink = Arc::new(AtomicU8::new(0));
        let str_pred = Arc::new(SetPredicate::from_strings(["s-00010", "s-01500"]));
        let pred_hits = searcher.search(
            &TermPredicateQuery::new(str_field, str_pred).with_strategy_config(
                TermSetStrategyConfig {
                    strategy_sink: Some(pred_sink.clone()),
                    ..Default::default()
                },
            ),
            &DocSetCollector,
        )?;
        assert_eq!(pred_hits.len(), 2);
        assert_eq!(pred_sink.load(Ordering::Relaxed), StrategyTag::Gallop as u8);

        let set_sink = Arc::new(AtomicU8::new(0));
        let set_hits = searcher.search(
            &TermSetQuery::new([
                Term::from_field_text(str_field, "s-00010"),
                Term::from_field_text(str_field, "s-01500"),
            ])
            .with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(set_sink.clone()),
                ..Default::default()
            }),
            &DocSetCollector,
        )?;
        assert_eq!(set_hits, pred_hits);
        assert_eq!(set_sink.load(Ordering::Relaxed), StrategyTag::Gallop as u8);

        // Numeric column with narrowing bounds + out-of-range Empty short-circuit
        let num_pred = Arc::new(SetPredicate::from_i64s([50, 150, 99999]));
        let q_num =
            TermPredicateQuery::new(num_field, num_pred.clone()).with_bounds(BoundsRange::new(
                Bound::Included(Term::from_field_i64(num_field, 100)),
                Bound::Included(Term::from_field_i64(num_field, 200)),
            ));
        assert_eq!(searcher.search(&q_num, &DocSetCollector)?.len(), 1);

        let empty_sink = Arc::new(AtomicU8::new(0));
        let q_empty = TermPredicateQuery::new(num_field, num_pred)
            .with_bounds(BoundsRange::new(
                Bound::Included(Term::from_field_i64(num_field, 50_000)),
                Bound::Included(Term::from_field_i64(num_field, 60_000)),
            ))
            .with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(empty_sink.clone()),
                ..Default::default()
            });
        assert!(searcher.search(&q_empty, &DocSetCollector)?.is_empty());
        assert_eq!(empty_sink.load(Ordering::Relaxed), StrategyTag::Empty as u8);

        // Nested `((t1 OR t2 OR t3) AND TermPredicate) AND RangeQuery` must flatten to
        // `FilteredTermUnion` and match `TermSetQuery` under `TopDocs`.
        let mut wsb = SchemaBuilder::new();
        let body = wsb.add_text_field("body", TEXT);
        let owner = wsb.add_i64_field("owner", NumericOptions::default().set_fast().set_indexed());
        let widx = Index::create_in_ram(wsb.build());
        {
            let mut w = widx.writer_for_tests()?;
            for i in 0..8_192i64 {
                let mut d = crate::TantivyDocument::new();
                let txt = match i % 4 {
                    0 => "alpha beta gamma",
                    1 => "alpha beta",
                    2 => "beta gamma",
                    _ => "delta",
                };
                d.add_text(body, txt);
                d.add_i64(owner, i);
                w.add_document(d)?;
            }
            w.commit()?;
        }
        let wsearcher = widx.reader()?.searcher();
        let keys: Vec<i64> = (0..8_192i64).step_by(7).collect();
        let wpred = Arc::new(SetPredicate::from_i64s(keys.iter().copied()));
        let range_q = RangeQuery::new(
            Bound::Included(Term::from_field_i64(owner, 1_000)),
            Bound::Excluded(Term::from_field_i64(owner, 6_000)),
        );
        let disjunct = BooleanQuery::union(vec![
            Box::new(TermQuery::new(
                Term::from_field_text(body, "alpha"),
                IndexRecordOption::WithFreqs,
            )),
            Box::new(TermQuery::new(
                Term::from_field_text(body, "beta"),
                IndexRecordOption::WithFreqs,
            )),
            Box::new(TermQuery::new(
                Term::from_field_text(body, "gamma"),
                IndexRecordOption::WithFreqs,
            )),
        ]);
        let make_nested_scored = |filter_q: Box<dyn crate::query::Query>| {
            let inner = BooleanQuery::new(vec![
                (Occur::Must, Box::new(disjunct.clone())),
                (Occur::Must, Box::new(ConstScoreQuery::new(filter_q, 0.0))),
            ]);
            BooleanQuery::new(vec![
                (Occur::Must, Box::new(inner)),
                (
                    Occur::Must,
                    Box::new(ConstScoreQuery::new(Box::new(range_q.clone()), 0.0)),
                ),
            ])
        };
        let nested_pred =
            make_nested_scored(Box::new(TermPredicateQuery::new(owner, wpred.clone())));

        let explicit_terms: Vec<Term> = keys
            .iter()
            .map(|&k| Term::from_field_i64(owner, k))
            .collect();
        let nested_set = make_nested_scored(Box::new(TermSetQuery::new(explicit_terms)));
        let got_top = wsearcher.search(&nested_pred, &TopDocs::with_limit(15).order_by_score())?;
        let exp_top = wsearcher.search(&nested_set, &TopDocs::with_limit(15).order_by_score())?;
        assert_eq!(got_top, exp_top);

        // Selective numeric predicate sorts ahead of a broad one regardless of clause order.
        let driver_docs: Vec<DocId> = (0..100).collect();
        assert_narrow_runs_before_broad(
            &wsearcher,
            owner,
            &driver_docs,
            || {
                (
                    Arc::new(SetPredicate::from_i64s([0, 4, 8]).with_estimated_distinct_keys(3)),
                    Arc::new(
                        SetPredicate::from_i64s(0..6_000i64).with_estimated_distinct_keys(6_000),
                    ),
                )
            },
            &[0, 4, 8],
        )?;
        Ok(())
    }

    #[test]
    fn batch_probing_telemetry_and_semantics() -> crate::Result<()> {
        use crate::query::Query;
        use crate::TERMINATED;

        let mut sb = SchemaBuilder::new();
        let body = sb.add_text_field("body", TEXT);
        let full_col = sb.add_i64_field(
            "full_col",
            NumericOptions::default().set_fast().set_indexed(),
        );
        let opt_col = sb.add_i64_field(
            "opt_col",
            NumericOptions::default().set_fast().set_indexed(),
        );
        let multi_col = sb.add_i64_field(
            "multi_col",
            NumericOptions::default().set_fast().set_indexed(),
        );
        let sorted_owner = sb.add_i64_field(
            "sorted_owner",
            NumericOptions::default().set_fast().set_indexed(),
        );
        let str_col = sb.add_text_field("str_col", FAST);
        let index = Index::builder()
            .schema(sb.build())
            .settings(IndexSettings {
                sort_by_field: Some(IndexSortByField {
                    field: "sorted_owner".to_string(),
                    order: Order::Asc,
                }),
                ..Default::default()
            })
            .create_in_ram()?;
        {
            let mut w = index.writer_for_tests()?;
            for i in 0..8_192i64 {
                let mut d = crate::TantivyDocument::new();
                let mut tokens = Vec::new();
                let tf = (i % 4) + 1;
                if i % 40 == 0 {
                    for _ in 0..tf {
                        tokens.push("stride40");
                    }
                }
                if i % 3 == 0 {
                    for _ in 0..tf {
                        tokens.push("hit1");
                    }
                }
                if i % 5 == 0 {
                    for _ in 0..((i / 3) % 4 + 1) {
                        tokens.push("hit2");
                    }
                }
                if i % 7 == 0 {
                    for _ in 0..((i / 5) % 4 + 1) {
                        tokens.push("hit3");
                    }
                }
                if tokens.is_empty() {
                    tokens.push("filler");
                }
                d.add_text(body, &tokens.join(" "));
                d.add_i64(full_col, i);
                if i % 3 != 0 {
                    d.add_i64(opt_col, i);
                }
                match i % 4 {
                    0 | 2 => {
                        d.add_i64(multi_col, i);
                        d.add_i64(multi_col, i + 10_000);
                    }
                    1 => d.add_i64(multi_col, i),
                    _ => {}
                }
                d.add_i64(sorted_owner, i);
                d.add_text(str_col, &format!("key-{i:06}-padding-block"));
                w.add_document(d)?;
            }
            w.commit()?;
        }
        let searcher = index.reader()?.searcher();
        let keys: Vec<i64> = (0..18_192i64).step_by(3).collect();
        let pred = Arc::new(
            SetPredicate::from_i64s(keys.iter().copied()).with_probe_cost(crate::ProbeCost::new(
                crate::ProbeSetupCost(300),
                crate::ProbePerDocCost(10),
            )),
        );
        let reset = |p: &SetPredicate| {
            p.u64_telemetry.calls.store(0, Ordering::Relaxed);
            p.u64_telemetry.items.store(0, Ordering::Relaxed);
            p.bytes_telemetry.calls.store(0, Ordering::Relaxed);
            p.bytes_telemetry.items.store(0, Ordering::Relaxed);
        };
        let term_q = |tok: &str| {
            Box::new(TermQuery::new(
                Term::from_field_text(body, tok),
                IndexRecordOption::WithFreqs,
            )) as Box<dyn crate::query::Query>
        };

        // 1. Medium-stride (40) scoring `BatchFilteredScorer` across Full / Optional / Multivalued
        //    (both unconstrained and value-range constrained) and sorted `doc_bounds`.
        let unbounded = BoundsRange::new(Bound::Unbounded, Bound::Unbounded);
        for (field, bounds) in [
            (full_col, unbounded.clone()),
            (
                full_col,
                BoundsRange::new(
                    Bound::Included(Term::from_field_i64(full_col, 1_000)),
                    Bound::Excluded(Term::from_field_i64(full_col, 5_000)),
                ),
            ),
            (opt_col, unbounded.clone()),
            (multi_col, unbounded),
            (
                sorted_owner,
                BoundsRange::new(
                    Bound::Included(Term::from_field_i64(sorted_owner, 4_000)),
                    Bound::Excluded(Term::from_field_i64(sorted_owner, 4_500)),
                ),
            ),
        ] {
            reset(&pred);
            let pred_q = BooleanQuery::new(vec![
                (Occur::Must, term_q("stride40")),
                (
                    Occur::Must,
                    Box::new(
                        TermPredicateQuery::new(field, pred.clone()).with_bounds(bounds.clone()),
                    ),
                ),
            ]);
            let ref_terms: Vec<Term> = keys
                .iter()
                .copied()
                .filter(|&k| {
                    let lo_ok = match &bounds.lower_bound {
                        Bound::Included(t) => k >= t.value().as_i64().unwrap(),
                        Bound::Excluded(t) => k > t.value().as_i64().unwrap(),
                        Bound::Unbounded => true,
                    };
                    let hi_ok = match &bounds.upper_bound {
                        Bound::Included(t) => k <= t.value().as_i64().unwrap(),
                        Bound::Excluded(t) => k < t.value().as_i64().unwrap(),
                        Bound::Unbounded => true,
                    };
                    lo_ok && hi_ok
                })
                .map(|k| Term::from_field_i64(field, k))
                .collect();
            let ref_q = BooleanQuery::new(vec![
                (Occur::Must, term_q("stride40")),
                (Occur::Must, Box::new(TermSetQuery::new(ref_terms))),
            ]);
            // Exercise `BatchFilteredScorer` directly via `weight.scorer()` (draining docs+scores).
            let enable = crate::query::EnableScoring::enabled_from_searcher(&searcher);
            let mut got_scored = Vec::new();
            let mut s = pred_q
                .weight(enable)?
                .scorer(searcher.segment_reader(0), 1.0)?;
            while s.doc() != TERMINATED {
                got_scored.push((s.doc(), s.score()));
                s.advance();
            }
            let mut exp_scored = Vec::new();
            let mut ref_s = ref_q
                .weight(enable)?
                .scorer(searcher.segment_reader(0), 1.0)?;
            while ref_s.doc() != TERMINATED {
                exp_scored.push((ref_s.doc(), ref_s.score()));
                ref_s.advance();
            }
            assert_eq!(
                got_scored, exp_scored,
                "score/doc mismatch for field {field:?}"
            );
            let calls = pred.u64_telemetry.calls.load(Ordering::Relaxed);
            let items = pred.u64_telemetry.items.load(Ordering::Relaxed);
            assert!(
                calls <= 2,
                "expected <= 2 batched calls (64 + 141) for stride40 on {field:?}, got {calls}"
            );
            if field == sorted_owner {
                assert!(
                    items <= 20,
                    "doc_bounds skip horizon should skip docs outside [4000..4500), got {items} \
                     items"
                );
            } else {
                assert!(
                    items <= 410,
                    "batched seek_danger_batch should not decode skipped neighbors, got {items} \
                     items"
                );
            }
            // Also verify `TopDocs` (`BlockWandSingleScorer` path) matches exact scores & batches.
            reset(&pred);
            let got = searcher.search(&pred_q, &TopDocs::with_limit(256).order_by_score())?;
            let exp = searcher.search(&ref_q, &TopDocs::with_limit(256).order_by_score())?;
            assert_eq!(got, exp, "TopDocs score/doc mismatch for field {field:?}");
            assert!(
                pred.u64_telemetry.calls.load(Ordering::Relaxed) <= 4,
                "expected <= 4 BlockWandSingleScorer warmup chunks on {field:?}, got {}",
                pred.u64_telemetry.calls.load(Ordering::Relaxed)
            );
        }

        // 2. Progressive 64-doc ramp on early stop + `seek()` inside and past buffer.
        reset(&pred);
        let limit_q = BooleanQuery::new(vec![
            (Occur::Must, term_q("stride40")),
            (
                Occur::Must,
                Box::new(TermPredicateQuery::new(full_col, pred.clone())),
            ),
        ]);
        let weight = limit_q.weight(crate::query::EnableScoring::enabled_from_searcher(
            &searcher,
        ))?;
        let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
        let mut first_ten = Vec::new();
        while first_ten.len() < 10 && scorer.doc() != TERMINATED {
            first_ten.push((scorer.doc(), scorer.score()));
            scorer.advance();
        }
        assert_eq!(first_ten.len(), 10);
        assert_eq!(
            pred.u64_telemetry.calls.load(Ordering::Relaxed),
            1,
            "LIMIT 10 should only trigger the initial 64-doc batch"
        );
        assert_eq!(
            pred.u64_telemetry.items.load(Ordering::Relaxed),
            64,
            "first batch should be 64 docs, not 1024"
        );
        // Seek inside current buffer, then seek past buffer.
        let target_in_buf = scorer.doc() + 40;
        let d1 = scorer.seek(target_in_buf);
        assert!(d1 >= target_in_buf && d1 != TERMINATED);
        let d2 = scorer.seek(5_000);
        assert!(d2 >= 5_000 && d2 != TERMINATED);

        // 3. `CachedDictPredicateDocSet` batching via `seek_danger_batch`.
        let str_targets: Vec<String> = (0..8_192i64)
            .step_by(40)
            .filter(|i| i % 3 == 0)
            .map(|i| format!("key-{i:06}-padding-block"))
            .collect();
        let str_pred = Arc::new(SetPredicate::from_strings(
            str_targets.iter().map(String::as_str),
        ));
        let dict_pred_q = BooleanQuery::new(vec![
            (Occur::Must, term_q("stride40")),
            (
                Occur::Must,
                Box::new(
                    TermPredicateQuery::new(str_col, str_pred.clone())
                        .with_exact_postings(ExactPostings::Unavailable),
                ),
            ),
        ]);
        let dict_ref_q = BooleanQuery::new(vec![
            (Occur::Must, term_q("stride40")),
            (
                Occur::Must,
                Box::new(FastFieldTermSetQuery::new(
                    str_targets
                        .iter()
                        .map(|s| Term::from_field_text(str_col, s)),
                )),
            ),
        ]);
        assert_eq!(
            searcher.search(&dict_pred_q, &DocSetCollector)?,
            searcher.search(&dict_ref_q, &DocSetCollector)?
        );
        assert!(
            str_pred.bytes_telemetry.calls.load(Ordering::Relaxed) <= 3,
            "CachedDictPredicateDocSet should batch filter_bytes calls, got {}",
            str_pred.bytes_telemetry.calls.load(Ordering::Relaxed)
        );

        // 4. WAND callsites: `BlockWandSingleScorer`, `BlockWandIntersectionScorer`,
        //    `block_maxscore_filtered`, and `BlockWandUnionScorer` via `Weight::pruning_scorer`.
        let full_ref_terms: Vec<Term> = (0..8_192i64)
            .step_by(3)
            .map(|k| Term::from_field_i64(full_col, k))
            .collect();
        for (name, terms_q) in [
            ("single", term_q("hit1")),
            (
                "intersection",
                Box::new(BooleanQuery::intersection(vec![
                    term_q("hit1"),
                    term_q("hit2"),
                ])) as Box<dyn crate::query::Query>,
            ),
            (
                "maxscore",
                Box::new(BooleanQuery::union(vec![
                    term_q("hit1"),
                    term_q("hit2"),
                    term_q("hit3"),
                ])) as Box<dyn crate::query::Query>,
            ),
        ] {
            reset(&pred);
            let q_pred = BooleanQuery::new(vec![
                (Occur::Must, terms_q.box_clone()),
                (
                    Occur::Must,
                    Box::new(ConstScoreQuery::new(
                        Box::new(TermPredicateQuery::new(full_col, pred.clone())),
                        0.0,
                    )),
                ),
            ]);
            let q_ref = BooleanQuery::new(vec![
                (Occur::Must, terms_q),
                (
                    Occur::Must,
                    Box::new(ConstScoreQuery::new(
                        Box::new(TermSetQuery::new(full_ref_terms.clone())),
                        0.0,
                    )),
                ),
            ]);
            let got = searcher.search(&q_pred, &TopDocs::with_limit(10).order_by_score())?;
            let exp = searcher.search(&q_ref, &TopDocs::with_limit(10).order_by_score())?;
            assert_eq!(got, exp, "WAND top-10 mismatch for {name}");
            let calls = pred.u64_telemetry.calls.load(Ordering::Relaxed);
            assert!(
                calls <= 64,
                "WAND {name} should batch filter probes, got {calls} calls"
            );
        }

        // 5. `BatchExcludeScorer`: batched `MUST_NOT` with exact scores.
        reset(&pred);
        let excl_pred_q = BooleanQuery::new(vec![
            (Occur::Must, term_q("stride40")),
            (
                Occur::MustNot,
                Box::new(TermPredicateQuery::new(full_col, pred.clone())),
            ),
        ]);
        let excl_ref_q = BooleanQuery::new(vec![
            (Occur::Must, term_q("stride40")),
            (Occur::MustNot, Box::new(TermSetQuery::new(full_ref_terms))),
        ]);
        let got_excl = searcher.search(&excl_pred_q, &TopDocs::with_limit(256).order_by_score())?;
        let exp_excl = searcher.search(&excl_ref_q, &TopDocs::with_limit(256).order_by_score())?;
        assert_eq!(got_excl, exp_excl);
        assert!(
            pred.u64_telemetry.calls.load(Ordering::Relaxed) <= 3,
            "BatchExcludeScorer should batch MUST_NOT probes, got {} calls",
            pred.u64_telemetry.calls.load(Ordering::Relaxed)
        );
        assert_eq!(pred.u64_telemetry.items.load(Ordering::Relaxed), 205);
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        #[test]
        fn predicate_matches_term_set_across_substrates_and_sorts(
            sort_by_str in any::<bool>(),
            sort_dir in prop_oneof![Just(None), Just(Some(Order::Asc)), Just(Some(Order::Desc))],
            doc_vals in prop::collection::vec(prop::option::of(0i64..40), 1..300),
            query_vals in prop::collection::vec(0i64..50, 0..30),
            bound_pair in prop::option::of((0i64..50, 0i64..50)),
            with_rare_filter in any::<bool>(),
            hint_keys in any::<bool>(),
            high_setup in any::<bool>(),
        ) {
            let mut sb = SchemaBuilder::new();
            let rare_tag = sb.add_text_field("rare_tag", STRING);
            let str_both = sb.add_text_field("s_both", STRING | FAST);
            let str_fast_only = sb.add_text_field("s_fast", FAST);
            let str_idx_only = sb.add_text_field(
                "s_idx",
                TextOptions::default().set_indexing_options(
                    TextFieldIndexing::default()
                        .set_tokenizer("raw")
                        .set_index_option(IndexRecordOption::Basic),
                ),
            );
            let num_both = sb.add_i64_field("n_both", NumericOptions::default().set_fast().set_indexed());
            let num_fast_only = sb.add_i64_field("n_fast", NumericOptions::default().set_fast());
            let num_idx_only = sb.add_i64_field("n_idx", NumericOptions::default().set_indexed());

            let mut idx_builder = Index::builder().schema(sb.build());
            if let Some(order) = sort_dir {
                idx_builder = idx_builder.settings(IndexSettings {
                    sort_by_field: Some(IndexSortByField {
                        field: if sort_by_str { "s_both" } else { "n_both" }.to_string(),
                        order,
                    }),
                    ..Default::default()
                });
            }
            let index = idx_builder.create_in_ram().unwrap();
            {
                let mut writer = index.writer_for_tests().unwrap();
                for (doc_idx, maybe_v) in doc_vals.iter().copied().enumerate() {
                    let mut d = crate::TantivyDocument::new();
                    d.add_text(rare_tag, if doc_idx % 97 == 0 { "hit" } else { "miss" });
                    if let Some(v) = maybe_v {
                        let s = format!("val-{v:04}-block-pad-0123456789abcdef");
                        d.add_text(str_both, &s);
                        d.add_text(str_fast_only, &s);
                        d.add_text(str_idx_only, &s);
                        d.add_i64(num_both, v);
                        d.add_i64(num_fast_only, v);
                        d.add_i64(num_idx_only, v);
                    }
                    writer.add_document(d).unwrap();
                }
                writer.commit().unwrap();
            }

            let reader = index.reader().unwrap();
            let searcher = reader.searcher();

            let (lo_opt, hi_opt) = match bound_pair {
                Some((a, b)) => (Some(a.min(b)), Some(a.max(b))),
                None => (None, None),
            };
            let effective_vals: Vec<i64> = query_vals
                .iter()
                .copied()
                .filter(|v| lo_opt.is_none_or(|lo| *v >= lo) && hi_opt.is_none_or(|hi| *v <= hi))
                .collect();

            let wrap_rare = |q: Box<dyn crate::query::Query>| -> Box<dyn crate::query::Query> {
                if with_rare_filter {
                    Box::new(BooleanQuery::new(vec![
                        (
                            Occur::Must,
                            Box::new(TermQuery::new(
                                Term::from_field_text(rare_tag, "hit"),
                                IndexRecordOption::Basic,
                            )),
                        ),
                        (Occur::Must, q),
                    ]))
                } else {
                    q
                }
            };

            let ref_terms: Vec<Term> = effective_vals
                .iter()
                .map(|&v| Term::from_field_i64(num_both, v))
                .collect();
            let expected = searcher
                .search(&wrap_rare(Box::new(TermSetQuery::new(ref_terms))), &DocSetCollector)
                .unwrap();

            // Also verify `TermSetQuery` / `FastFieldTermSetQuery` on `str_both` and `str_fast_only`.
            let str_ref_both: Vec<Term> = effective_vals
                .iter()
                .map(|&v| Term::from_field_text(str_both, &format!("val-{v:04}-block-pad-0123456789abcdef")))
                .collect();
            prop_assert_eq!(
                searcher.search(&wrap_rare(Box::new(TermSetQuery::new(str_ref_both))), &DocSetCollector).unwrap(),
                expected.clone()
            );
            let str_ref_fast: Vec<Term> = effective_vals
                .iter()
                .map(|&v| Term::from_field_text(str_fast_only, &format!("val-{v:04}-block-pad-0123456789abcdef")))
                .collect();
            prop_assert_eq!(
                searcher.search(&wrap_rare(Box::new(FastFieldTermSetQuery::new(str_ref_fast))), &DocSetCollector).unwrap(),
                expected.clone()
            );

            let str_strings: Vec<String> = query_vals
                .iter()
                .map(|&v| format!("val-{v:04}-block-pad-0123456789abcdef"))
                .collect();
            let i64_set: FxHashSet<i64> = query_vals.iter().copied().collect();
            let pred = Arc::new(SetPredicate {
                bytes_set: str_strings.iter().map(|s| s.as_bytes().to_vec()).collect(),
                estimated_keys: hint_keys.then_some(i64_set.len() as u64),
                i64_set,
                probe_cost: if high_setup {
                    crate::ProbeCost::new(crate::ProbeSetupCost(300), crate::ProbePerDocCost(10))
                } else {
                    crate::ProbeCost::per_doc(5)
                },
                bytes_telemetry: EvalTelemetry::default(),
                u64_telemetry: EvalTelemetry::default(),
            });

            let str_bounds = |f| {
                BoundsRange::new(
                    lo_opt.map_or(Bound::Unbounded, |lo| {
                        Bound::Included(Term::from_field_text(
                            f,
                            &format!("val-{lo:04}-block-pad-0123456789abcdef"),
                        ))
                    }),
                    hi_opt.map_or(Bound::Unbounded, |hi| {
                        Bound::Included(Term::from_field_text(
                            f,
                            &format!("val-{hi:04}-block-pad-0123456789abcdef"),
                        ))
                    }),
                )
            };
            let num_bounds = |f| {
                BoundsRange::new(
                    lo_opt.map_or(Bound::Unbounded, |lo| Bound::Included(Term::from_field_i64(f, lo))),
                    hi_opt.map_or(Bound::Unbounded, |hi| Bound::Included(Term::from_field_i64(f, hi))),
                )
            };

            for candidate_q in [
                TermPredicateQuery::new(str_both, pred.clone()).with_bounds(str_bounds(str_both)),
                TermPredicateQuery::new(str_fast_only, pred.clone())
                    .with_bounds(str_bounds(str_fast_only))
                    .with_exact_postings(ExactPostings::Unavailable),
                TermPredicateQuery::new(str_idx_only, pred.clone()).with_bounds(str_bounds(str_idx_only)),
                TermPredicateQuery::new(num_both, pred.clone()).with_bounds(num_bounds(num_both)),
                TermPredicateQuery::new(num_fast_only, pred.clone())
                    .with_bounds(num_bounds(num_fast_only))
                    .with_exact_postings(ExactPostings::Unavailable),
                TermPredicateQuery::new(num_idx_only, pred.clone()).with_bounds(num_bounds(num_idx_only)),
            ] {
                let got = searcher
                    .search(&wrap_rare(Box::new(candidate_q)), &DocSetCollector)
                    .unwrap();
                prop_assert_eq!(got, expected.clone());
            }

            // Verify `PredicateColumnDocSet` propagates `probe_cost()` and bounds `size_hint()`.
            let num_s = TermPredicateQuery::new(num_fast_only, pred.clone())
                .with_bounds(num_bounds(num_fast_only))
                .with_exact_postings(ExactPostings::Unavailable)
                .weight(EnableScoring::disabled_from_searcher(&searcher))
                .unwrap()
                .scorer_for_candidates_unpositioned(
                    searcher.segment_reader(0),
                    1.0,
                    CandidateHint::Estimated(1),
                )
                .unwrap();
            if !num_s.is_empty() {
                prop_assert_eq!(
                    num_s.cost().probe,
                    crate::ProbeCost::FAST_FIELD.saturating_add(pred.probe_cost())
                );
                prop_assert!(num_s.size_hint() <= searcher.segment_reader(0).num_docs());
            }
        }
    }

    const TEST_STORAGE_PAGE_LEN: usize = 8192;

    #[derive(Clone, Debug)]
    struct PagedFastDirectory {
        inner: RamDirectory,
        block_len: usize,
        fast_reads: Option<Arc<Mutex<Vec<Range<usize>>>>>,
    }

    impl PagedFastDirectory {
        fn new(inner: RamDirectory, block_len: usize, record_reads: bool) -> Self {
            Self {
                inner,
                block_len,
                fast_reads: record_reads.then(|| Arc::new(Mutex::new(Vec::new()))),
            }
        }
    }

    #[derive(Debug)]
    struct PagedFastFile {
        inner: Arc<dyn FileHandle>,
        block_len: usize,
        reads: Option<Arc<Mutex<Vec<Range<usize>>>>>,
    }

    impl HasLen for PagedFastFile {
        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    impl FileHandle for PagedFastFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            if let Some(reads) = &self.reads {
                reads.lock().unwrap().push(range.clone());
            }
            self.inner.read_bytes(range)
        }

        fn storage_block_len(&self) -> Option<usize> {
            Some(self.block_len)
        }
    }

    impl Directory for PagedFastDirectory {
        fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
            let inner = self.inner.get_file_handle(path)?;
            if path.extension().is_some_and(|ext| ext == "fast") {
                Ok(Arc::new(PagedFastFile {
                    inner,
                    block_len: self.block_len,
                    reads: self.fast_reads.clone(),
                }))
            } else {
                Ok(inner)
            }
        }
        fn delete(&self, path: &Path) -> Result<(), DeleteError> {
            self.inner.delete(path)
        }
        fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
            self.inner.exists(path)
        }
        fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
            self.inner.open_write(path)
        }
        fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
            self.inner.atomic_read(path)
        }
        fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
            self.inner.atomic_write(path, data)
        }
        fn sync_directory(&self) -> io::Result<()> {
            self.inner.sync_directory()
        }
        fn watch(&self, callback: WatchCallback) -> crate::Result<WatchHandle> {
            self.inner.watch(callback)
        }
    }

    #[allow(dead_code)]
    struct DictCacheFixture {
        ram_dir: RamDirectory,
        _index: Index,
        reader: crate::IndexReader,
        _paged_index: Index,
        paged_reader: crate::IndexReader,
        field: crate::schema::Field,
        title_field: crate::schema::Field,
        field_type: crate::schema::FieldType,
        doc_keys: Vec<String>,
        doc_matched: Vec<bool>,
        matching_terms: Vec<Term>,
        matching_strings: Vec<String>,
        doc_block_first_ord: Vec<u64>,
        sorted_terms: Vec<String>,
        max_terms_per_block: usize,
    }

    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    fn dict_cache_fixture() -> &'static DictCacheFixture {
        static FIXTURE: std::sync::OnceLock<DictCacheFixture> = std::sync::OnceLock::new();
        FIXTURE.get_or_init(|| {
            const NUM_UNIQUE: usize = 18_000;
            const NUM_DOCS: usize = 20_000;

            let mut sb = SchemaBuilder::new();
            let field = sb.add_text_field("dict_key", STRING | FAST);
            let title_field = sb.add_text_field("title", TEXT);
            let schema = sb.build();
            let field_type = schema.get_field_entry(field).field_type().clone();
            let ram_dir = RamDirectory::create();
            let index = Index::create(ram_dir.clone(), schema, IndexSettings::default()).unwrap();

            let unique_keys: Vec<String> = (0..NUM_UNIQUE)
                .map(|i| {
                    let a = splitmix64(i as u64);
                    let b = splitmix64(a);
                    let c = splitmix64(b);
                    let d = splitmix64(c);
                    format!("{a:016x}-{b:016x}-{c:016x}-{d:016x}-{i:06}")
                })
                .collect();

            let mut doc_keys = Vec::with_capacity(NUM_DOCS);
            let mut doc_matched = Vec::with_capacity(NUM_DOCS);
            {
                let mut writer = index.writer_with_num_threads(1, 50_000_000).unwrap();
                for doc_id in 0..NUM_DOCS {
                    let val_idx = doc_id % NUM_UNIQUE;
                    let key = &unique_keys[val_idx];
                    doc_keys.push(key.clone());
                    doc_matched.push(val_idx % 2 == 0);
                    // ~10% of documents match the 4-term BM25 disjunction (well above the 3.125%
                    // exhaustive candidate threshold), with a sparse set of high-scoring 4-term
                    // documents so WAND / Block-MaxScore can prune the remaining ~95% of blocks.
                    let title = if doc_id < 1_000 && doc_id % 20 == 0 {
                        "how how using using get get create create"
                    } else if doc_id % 10 == 0 {
                        match (doc_id / 10) % 4 {
                            0 => "how filler filler filler filler filler filler filler",
                            1 => "using filler filler filler filler filler filler filler",
                            2 => "get filler filler filler filler filler filler filler",
                            _ => "create filler filler filler filler filler filler filler",
                        }
                    } else {
                        "unrelated body text for non matching document"
                    };
                    writer
                        .add_document(doc!(field => key.as_str(), title_field => title))
                        .unwrap();
                }
                writer.commit().unwrap();
            }

            let reader = index.reader().unwrap();
            let paged_index = Index::open(PagedFastDirectory::new(
                ram_dir.clone(),
                TEST_STORAGE_PAGE_LEN,
                false,
            ))
            .unwrap();
            let paged_reader = paged_index.reader().unwrap();

            let searcher = reader.searcher();
            assert_eq!(searcher.segment_readers().len(), 1);
            let segment = searcher.segment_reader(0);
            let bytes_col = columnar::BytesColumn::from(
                segment.fast_fields().str("dict_key").unwrap().unwrap(),
            );
            let dict = bytes_col.dictionary();
            let num_terms = bytes_col.num_terms() as u64;
            let block_ids = dict.block_ids_for_ord_range(0..num_terms).unwrap().unwrap();
            let num_blocks = block_ids.end().saturating_sub(*block_ids.start()) + 1;
            assert!(
                num_blocks >= 256,
                "fixture must span >= 256 SSTable blocks, got {num_blocks}"
            );
            let paged_searcher = paged_reader.searcher();
            let paged_segment = paged_searcher.segment_reader(0);
            let _paged_col = columnar::BytesColumn::from(
                paged_segment
                    .fast_fields()
                    .str("dict_key")
                    .unwrap()
                    .unwrap(),
            );

            let mut max_terms_per_block = 0usize;
            let mut cur_ord = 0u64;
            while cur_ord < num_terms {
                let (addr, end_ord) = dict.block_addr_and_end_ord(cur_ord).unwrap().unwrap();
                max_terms_per_block =
                    max_terms_per_block.max((end_ord - addr.first_ordinal) as usize);
                cur_ord = end_ord;
            }

            let mut doc_block_first_ord = Vec::with_capacity(NUM_DOCS);
            for doc_id in 0..NUM_DOCS as u32 {
                let ord = bytes_col.ords().values_for_doc(doc_id).next().unwrap();
                let (addr, _) = dict.block_addr_and_end_ord(ord).unwrap().unwrap();
                doc_block_first_ord.push(addr.first_ordinal);
            }

            let mut sorted_terms = unique_keys.clone();
            sorted_terms.sort_unstable();
            let matching_strings: Vec<String> = unique_keys
                .iter()
                .enumerate()
                .filter(|(idx, _)| idx % 2 == 0)
                .map(|(_, s)| s.clone())
                .collect();
            let matching_terms: Vec<Term> = matching_strings
                .iter()
                .map(|s| Term::from_field_text(field, s))
                .collect();

            DictCacheFixture {
                ram_dir,
                _index: index,
                reader,
                _paged_index: paged_index,
                paged_reader,
                field,
                title_field,
                field_type,
                doc_keys,
                doc_matched,
                matching_terms,
                matching_strings,
                doc_block_first_ord,
                sorted_terms,
                max_terms_per_block,
            }
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Verifies query-scoped per-segment dictionary caching, SSTable block budgeting, and
        /// skip-tested promotion across repeated `scorer_for_candidates_unpositioned` chunks
        /// (both mmap/in-memory and 8 KB page-backed storage), covering bounded ranges,
        /// `TermSetWeight::clone` sharing, `TermPredicate`, and explicit `TermSetQuery`.
        #[test]
        fn dict_block_cache_and_weight_reuse_property(
            chunk_sizes in proptest::collection::vec(
                prop_oneof![1usize..=16, 17usize..=64, 300usize..=420],
                2..=12,
            ),
            start_doc in 0usize..20_000,
            stride in prop_oneof![Just(1usize), Just(2usize), 7usize..=503usize],
            bounded in any::<bool>(),
            explicit_terms in any::<bool>(),
            paged in any::<bool>(),
            hint_keys in any::<bool>(),
            high_setup in any::<bool>(),
            use_batch in any::<bool>(),
        ) {
            let f = dict_cache_fixture();
            let searcher = if paged {
                f.paged_reader.searcher()
            } else {
                f.reader.searcher()
            };
            let segment = searcher.segment_reader(0);
            let bytes_col =
                columnar::BytesColumn::from(segment.fast_fields().str("dict_key").unwrap().unwrap());
            let dict = bytes_col.dictionary();

            let (bounds, lo_str, hi_str, bounded_num_terms, bounded_num_blocks) = if bounded {
                let lo_idx = f.sorted_terms.len() / 6;
                let hi_idx = f.sorted_terms.len() * 5 / 6;
                let lo = f.sorted_terms[lo_idx].as_str();
                let hi = f.sorted_terms[hi_idx].as_str();
                let (lo_b, hi_b) = dict
                    .term_bounds_to_ord(Bound::Included(lo.as_bytes()), Bound::Included(hi.as_bytes()))
                    .unwrap();
                let lo_ord = match lo_b {
                    Bound::Included(o) => o,
                    Bound::Excluded(o) => o + 1,
                    Bound::Unbounded => 0,
                };
                let hi_excl = match hi_b {
                    Bound::Included(o) => o + 1,
                    Bound::Excluded(o) => o,
                    Bound::Unbounded => bytes_col.num_terms() as u64,
                };
                let block_ids = dict.block_ids_for_ord_range(lo_ord..hi_excl).unwrap().unwrap();
                let num_blocks = (block_ids.end().saturating_sub(*block_ids.start()) + 1) as usize;
                (
                    BoundsRange::new(
                        Bound::Included(Term::from_field_text(f.field, lo)),
                        Bound::Included(Term::from_field_text(f.field, hi)),
                    ),
                    Some(lo),
                    Some(hi),
                    (hi_excl - lo_ord) as usize,
                    num_blocks,
                )
            } else {
                let block_ids = dict
                    .block_ids_for_ord_range(0..bytes_col.num_terms() as u64)
                    .unwrap()
                    .unwrap();
                let num_blocks = (block_ids.end().saturating_sub(*block_ids.start()) + 1) as usize;
                (
                    BoundsRange::new(Bound::Unbounded, Bound::Unbounded),
                    None,
                    None,
                    bytes_col.num_terms(),
                    num_blocks,
                )
            };

            let mut pred_val =
                SetPredicate::from_strings(f.matching_strings.iter().map(String::as_str));
            if hint_keys {
                pred_val = pred_val.with_estimated_distinct_keys(f.matching_strings.len() as u64);
            }
            if high_setup {
                pred_val = pred_val.with_probe_cost(crate::ProbeCost::new(
                    crate::ProbeSetupCost(300),
                    crate::ProbePerDocCost(10),
                ));
            }
            let pred = Arc::new(pred_val);
            let weight = if explicit_terms && !bounded {
                super::TermSetWeight::from_terms(
                    f.field,
                    &f.matching_terms,
                    &f.field_type,
                    ExactFastField::Infer,
                    TermSetStrategyConfig::default(),
                )
                .unwrap()
            } else {
                super::TermSetWeight::from_predicate(
                    f.field,
                    &f.field_type,
                    pred.clone(),
                    bounds,
                    ExactPostings::Unavailable,
                    ExactFastField::Infer,
                    TermSetStrategyConfig::default(),
                )
                .unwrap()
            };
            let weight_clone = weight.clone();

            let mut touched_blocks: FxHashSet<u64> = FxHashSet::default();
            let mut cursor = start_doc;
            let mut all_chunks_under_budget = true;

            for (chunk_idx, &chunk_len) in chunk_sizes.iter().enumerate() {
                let mut docs: Vec<u32> = Vec::with_capacity(chunk_len);
                for _ in 0..chunk_len {
                    docs.push((cursor % f.doc_keys.len()) as u32);
                    cursor = cursor.wrapping_add(stride);
                }
                docs.sort_unstable();
                docs.dedup();
                if docs.len() >= bounded_num_blocks {
                    all_chunks_under_budget = false;
                }

                let w = if chunk_idx % 2 == 0 {
                    &weight
                } else {
                    &weight_clone
                };
                let mut scorer = super::Weight::scorer_for_candidates_unpositioned(
                    w,
                    segment,
                    1.0,
                    CandidateHint::Estimated(docs.len() as u32),
                )
                .unwrap();

                if chunk_idx == 0 && docs.len() < bounded_num_blocks {
                    let expected_probe = if explicit_terms && !bounded {
                        crate::ProbeCost::DICT_BLOCK
                    } else {
                        crate::ProbeCost::DICT_BLOCK.saturating_add(pred.probe_cost())
                    };
                    prop_assert_eq!(scorer.cost().probe, expected_probe);
                    let est_ords = if explicit_terms && !bounded {
                        (f.matching_terms.len() as u64).min(bounded_num_terms as u64)
                    } else {
                        pred.estimated_distinct_keys()
                            .map_or(bounded_num_terms as u64, |k| k.min(bounded_num_terms as u64))
                    };
                    prop_assert_eq!(
                        scorer.size_hint(),
                        super::scale_size_hint(
                            bytes_col.num_rows(),
                            est_ords,
                            bytes_col.num_terms() as u64,
                        )
                    );
                }

                if use_batch {
                    let words = docs.len().div_ceil(64);
                    let mut mask = vec![TinySet::EMPTY; words];
                    crate::docset::fill_all_candidates_mask(docs.len(), &mut mask);
                    scorer.seek_danger_batch(&docs, &mut mask);
                    for (i, &doc_id) in docs.iter().enumerate() {
                        let key = f.doc_keys[doc_id as usize].as_str();
                        let in_bounds = lo_str.is_none_or(|lo| key >= lo)
                            && hi_str.is_none_or(|hi| key <= hi);
                        if in_bounds {
                            touched_blocks.insert(f.doc_block_first_ord[doc_id as usize]);
                        }
                        let expected = f.doc_matched[doc_id as usize] && in_bounds;
                        let actual = mask[i / 64].contains((i % 64) as u32);
                        prop_assert_eq!(actual, expected, "batch mismatch at doc_id={}", doc_id);
                    }
                } else {
                    for &doc_id in &docs {
                        let key = f.doc_keys[doc_id as usize].as_str();
                        let in_bounds = lo_str.is_none_or(|lo| key >= lo)
                            && hi_str.is_none_or(|hi| key <= hi);
                        if in_bounds {
                            touched_blocks.insert(f.doc_block_first_ord[doc_id as usize]);
                        }
                        let expected = f.doc_matched[doc_id as usize] && in_bounds;
                        let actual = matches!(
                            scorer.seek_danger(doc_id),
                            crate::docset::SeekDangerResult::Found
                        );
                        prop_assert_eq!(actual, expected, "mismatch at doc_id={}", doc_id);
                    }
                }
            }

            if !(explicit_terms && !bounded) {
                let calls = pred.bytes_telemetry.calls.load(Ordering::Relaxed);
                let items = pred.bytes_telemetry.items.load(Ordering::Relaxed);
                // Skip-tested promotion guarantees every dictionary entry in `ord_range` is
                // evaluated by `filter_bytes` at most once (<= 1.0x), whether mmap or paged.
                prop_assert!(
                    items <= bounded_num_terms,
                    "expected <= 1.0x dict evaluation ({} items), got {}",
                    bounded_num_terms,
                    items
                );
                // When every chunk stays strictly under `bounded_num_blocks`, no full promotion
                // occurs (neither on mmap nor on paged storage) and only touched blocks are decoded.
                if all_chunks_under_budget {
                    prop_assert!(
                        calls <= touched_blocks.len(),
                        "expected <= {} filter_bytes calls (unique blocks), got {}",
                        touched_blocks.len(),
                        calls
                    );
                    prop_assert!(
                        items <= touched_blocks.len() * f.max_terms_per_block,
                        "expected <= {} items across {} blocks, got {}",
                        touched_blocks.len() * f.max_terms_per_block,
                        touched_blocks.len(),
                        items
                    );
                }
            }
        }
    }

    #[test]
    fn single_page_reads_and_pruned_candidate_hint() -> crate::Result<()> {
        let f = dict_cache_fixture();

        // 1. Verify single-page coalescing and zero cross-page coalescing on 8 KB paged storage.
        let mut dict_buf = Vec::new();
        let mut dict_builder = <columnar::Dictionary>::builder(&mut dict_buf)?;
        for term in &f.sorted_terms {
            dict_builder.insert(term.as_bytes(), &())?;
        }
        dict_builder.finish()?;
        let fast_reads = Arc::new(Mutex::new(Vec::new()));
        let paged_handle: Arc<dyn FileHandle> = Arc::new(PagedFastFile {
            inner: Arc::new(OwnedBytes::new(dict_buf)),
            block_len: TEST_STORAGE_PAGE_LEN,
            reads: Some(fast_reads.clone()),
        });
        let tracked_dict =
            <columnar::Dictionary>::open(common::file_slice::FileSlice::new(paged_handle))?;
        let block_ids = tracked_dict
            .block_ids_for_ord_range(0..tracked_dict.num_terms() as u64)?
            .unwrap();
        let num_blocks = block_ids.end().saturating_sub(*block_ids.start()) + 1;
        let mut all_blocks = Vec::with_capacity(num_blocks as usize);
        tracked_dict.visit_blocks_in_range(block_ids, |addr| all_blocks.push(addr))?;
        let straddling_ranges: FxHashSet<Range<usize>> = all_blocks
            .iter()
            .filter(|addr| {
                let s = tracked_dict
                    .sstable_slice
                    .storage_block_ord(addr.byte_range.start);
                let e = tracked_dict
                    .sstable_slice
                    .storage_block_ord(addr.byte_range.end - 1);
                s != e
            })
            .map(|addr| addr.byte_range.clone())
            .collect();

        fast_reads.lock().unwrap().clear();
        let mut visited_terms = 0usize;
        tracked_dict.visit_blocks_coalesced(&all_blocks, |_ord, _key| -> io::Result<()> {
            visited_terms += 1;
            Ok(())
        })?;
        assert_eq!(visited_terms, tracked_dict.num_terms());
        let recorded_reads = fast_reads.lock().unwrap().clone();
        assert!(
            recorded_reads.len() < all_blocks.len(),
            "expected single-page coalescing to issue fewer reads ({}) than SSTable blocks ({})",
            recorded_reads.len(),
            all_blocks.len()
        );
        for range in &recorded_reads {
            let start_page = range.start / TEST_STORAGE_PAGE_LEN;
            let end_page = (range.end - 1) / TEST_STORAGE_PAGE_LEN;
            if start_page != end_page {
                let sstable_rel_matches_single_block =
                    straddling_ranges.iter().any(|b| b.len() == range.len());
                assert!(
                    sstable_rel_matches_single_block,
                    "unexpected cross-page coalesced read {range:?} (len={})",
                    range.len()
                );
            }
        }

        // 2. Under `CandidateHint::Pruned`, large `TermSetQuery` and `TermPredicateQuery` over
        // high-cardinality `STRING | FAST` choose lazy `CachedDictPredicateDocSet` (`Linear`) and
        // do zero dictionary I/O when never probed, while non-prunable `AllQuery AND
        // TermPredicateQuery` stays on `dispatch_postings`.
        let searcher = f.reader.searcher();
        let segment = searcher.segment_reader(0);
        let enable = EnableScoring::disabled_from_searcher(&searcher);

        let termset_sink = Arc::new(AtomicU8::new(0));
        let termset_w = TermSetQuery::new(f.matching_terms.clone())
            .with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(termset_sink.clone()),
                ..Default::default()
            })
            .weight(enable)?;
        let _ts_unpos = termset_w.scorer_for_candidates_unpositioned(
            segment,
            1.0,
            CandidateHint::Pruned { upper_bound: 2_000 },
        )?;
        assert_eq!(
            termset_sink.load(Ordering::Relaxed),
            StrategyTag::Linear as u8,
            "large TermSetQuery under CandidateHint::Pruned should use lazy Linear block cache"
        );

        let pred_warm = Arc::new(SetPredicate::from_strings(
            f.matching_strings.iter().map(String::as_str),
        ));
        let warm_sink = Arc::new(AtomicU8::new(0));
        let pred_w = TermPredicateQuery::new(f.field, pred_warm.clone())
            .with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(warm_sink.clone()),
                ..Default::default()
            })
            .weight(enable)?;
        let _pred_unpos = pred_w.scorer_for_candidates_unpositioned(
            segment,
            1.0,
            CandidateHint::Pruned { upper_bound: 2_000 },
        )?;
        assert_eq!(
            warm_sink.load(Ordering::Relaxed),
            StrategyTag::Linear as u8,
            "TermPredicateQuery under CandidateHint::Pruned must choose lazy Linear cache"
        );
        assert_eq!(
            pred_warm.bytes_telemetry.items.load(Ordering::Relaxed),
            0,
            "unprobed CachedDictPredicateDocSet should decode 0 dictionary terms"
        );

        let all_pred = Arc::new(SetPredicate::from_strings(
            f.matching_strings.iter().map(String::as_str),
        ));
        let all_sink = Arc::new(AtomicU8::new(0));
        let all_q = BooleanQuery::new(vec![
            (Occur::Must, Box::new(AllQuery)),
            (
                Occur::Must,
                Box::new(ConstScoreQuery::new(
                    Box::new(TermPredicateQuery::new(f.field, all_pred).with_strategy_config(
                        TermSetStrategyConfig {
                            strategy_sink: Some(all_sink.clone()),
                            ..Default::default()
                        },
                    )),
                    0.0,
                )),
            ),
        ]);
        let all_docs = searcher.search(&all_q, &DocSetCollector)?;
        assert_eq!(all_docs.len(), f.doc_matched.iter().filter(|&&m| m).count());
        let all_tag = all_sink.load(Ordering::Relaxed);
        assert!(
            all_tag == StrategyTag::Automaton as u8 || all_tag == StrategyTag::Bitset as u8,
            "AllQuery AND TermPredicateQuery should stay on dispatch_postings, got tag={all_tag}"
        );
        Ok(())
    }

    #[test]
    fn aliased_tokenizer_exact_fast_field_opt_in() -> crate::Result<()> {
        use crate::tokenizer::RawTokenizer;

        // 1. Unsorted `Str` field with aliased `"keyword"` indexing tokenizer (`RawTokenizer`) and
        //    `"raw"` fast-field tokenizer under WAND TopDocs(10) (`for_each_pruning`).
        let mut sb = SchemaBuilder::new();
        let kw_opts = TextOptions::default().set_fast("raw").set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("keyword")
                .set_index_option(IndexRecordOption::Basic),
        );
        let kw_field = sb.add_text_field("kw", kw_opts.clone());
        let title_field = sb.add_text_field("title", TEXT | FAST);
        let index = Index::create_in_ram(sb.build());
        index
            .tokenizers()
            .register("keyword", RawTokenizer::default());

        let unique_keys: Vec<String> = (0..4_000u64)
            .map(|i| {
                let a = splitmix64(i);
                let b = splitmix64(a);
                format!("{a:016x}-{b:016x}-{i:05}")
            })
            .collect();
        {
            let mut writer = index.writer_for_tests()?;
            for doc_id in 0..4_000usize {
                let title = if doc_id == 0 {
                    "Hello World"
                } else if doc_id < 400 && doc_id % 20 == 0 {
                    "how how using using get get create create"
                } else if doc_id % 10 == 0 {
                    match (doc_id / 10) % 4 {
                        0 => "how filler filler filler filler filler filler filler",
                        1 => "using filler filler filler filler filler filler filler",
                        2 => "get filler filler filler filler filler filler filler",
                        _ => "create filler filler filler filler filler filler filler",
                    }
                } else {
                    "unrelated body text"
                };
                writer.add_document(doc!(
                    kw_field => unique_keys[doc_id].as_str(),
                    title_field => title,
                ))?;
            }
            writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        let large_strings: Vec<String> = unique_keys
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, s)| s.clone())
            .collect();
        let large_terms: Vec<Term> = large_strings
            .iter()
            .map(|s| Term::from_field_text(kw_field, s))
            .collect();
        let large_pred = Arc::new(SetPredicate::from_strings(
            large_strings.iter().map(String::as_str),
        ));

        let segment = searcher.segment_reader(0);
        let enable = EnableScoring::disabled_from_searcher(&searcher);
        let hint = CandidateHint::Pruned { upper_bound: 400 };

        // Default `ExactFastField::Infer` locks aliased `"keyword"`/`"raw"` out of the fast field.
        let infer_sink = Arc::new(AtomicU8::new(0));
        let infer_w = TermSetQuery::new(large_terms.clone())
            .with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(infer_sink.clone()),
                ..Default::default()
            })
            .weight(enable)?;
        let _ = infer_w.scorer_for_candidates_unpositioned(segment, 1.0, hint)?;
        assert_eq!(
            infer_sink.load(Ordering::Relaxed),
            StrategyTag::Automaton as u8
        );

        let pred_infer_sink = Arc::new(AtomicU8::new(0));
        let pred_infer_w = TermPredicateQuery::new(kw_field, large_pred.clone())
            .with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(pred_infer_sink.clone()),
                ..Default::default()
            })
            .weight(enable)?;
        let _ = pred_infer_w.scorer_for_candidates_unpositioned(segment, 1.0, hint)?;
        assert_eq!(
            pred_infer_sink.load(Ordering::Relaxed),
            StrategyTag::Automaton as u8
        );

        // `ExactFastField::Available` unlocks `CachedDictPredicateDocSet` (`Linear`) for large K
        // under pruned candidate probing, while keeping `ExactPostings::Available` (`Bitset`) for
        // small K.
        let avail_sink = Arc::new(AtomicU8::new(0));
        let avail_w = TermSetQuery::new(large_terms)
            .with_exact_fast_field(ExactFastField::Available)
            .with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(avail_sink.clone()),
                ..Default::default()
            })
            .weight(enable)?;
        let _ = avail_w.scorer_for_candidates_unpositioned(segment, 1.0, hint)?;
        assert_eq!(
            avail_sink.load(Ordering::Relaxed),
            StrategyTag::Linear as u8
        );

        let pred_avail_sink = Arc::new(AtomicU8::new(0));
        let pred_avail_w = TermPredicateQuery::new(kw_field, large_pred)
            .with_exact_fast_field(ExactFastField::Available)
            .with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(pred_avail_sink.clone()),
                ..Default::default()
            })
            .weight(enable)?;
        let _ = pred_avail_w.scorer_for_candidates_unpositioned(segment, 1.0, hint)?;
        assert_eq!(
            pred_avail_sink.load(Ordering::Relaxed),
            StrategyTag::Linear as u8
        );

        let small_terms = vec![
            Term::from_field_text(kw_field, &unique_keys[0]),
            Term::from_field_text(kw_field, &unique_keys[20]),
        ];
        let small_sink = Arc::new(AtomicU8::new(0));
        let small_q = TermSetQuery::new(small_terms)
            .with_exact_fast_field(ExactFastField::Available)
            .with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(small_sink.clone()),
                ..Default::default()
            });
        let small_docs = searcher.search(&small_q, &DocSetCollector)?;
        assert_eq!(small_docs.len(), 2);
        assert_eq!(
            small_sink.load(Ordering::Relaxed),
            StrategyTag::Bitset as u8
        );

        // `TEXT | FAST` (`"default"` indexing vs `"raw"` fast field):
        // - Default `(ExactPostings::Available, ExactFastField::Infer)` locks out the raw fast
        //   field so it stays on tokenized postings without mixing domains.
        // - Disabling postings without opting into `ExactFastField::Available` fails closed.
        // - Opting into `ExactFastField::Available` + `ExactPostings::Unavailable` queries raw
        //   fast-field values directly.
        let raw_title_pred = Arc::new(SetPredicate::from_strings(["Hello World"]));
        let token_title_pred = Arc::new(SetPredicate::from_strings(["hello"]));
        assert_eq!(
            searcher
                .search(
                    &TermPredicateQuery::new(title_field, raw_title_pred.clone()),
                    &DocSetCollector,
                )?
                .len(),
            0
        );
        assert_eq!(
            searcher
                .search(
                    &TermPredicateQuery::new(title_field, token_title_pred),
                    &DocSetCollector,
                )?
                .len(),
            1
        );
        assert!(matches!(
            searcher.search(
                &TermPredicateQuery::new(title_field, raw_title_pred.clone())
                    .with_exact_postings(ExactPostings::Unavailable),
                &DocSetCollector,
            ),
            Err(crate::TantivyError::SchemaError(_))
        ));
        assert_eq!(
            searcher
                .search(
                    &TermPredicateQuery::new(title_field, raw_title_pred)
                        .with_exact_postings(ExactPostings::Unavailable)
                        .with_exact_fast_field(ExactFastField::Available),
                    &DocSetCollector,
                )?
                .len(),
            1
        );

        // 2. Sorted `"keyword"` + `set_fast("raw")`: `Infer` stays on postings while `Available`
        //    unlocks `Gallop` with identical matches.
        let mut ssb = SchemaBuilder::new();
        let sorted_kw = ssb.add_text_field("kw", kw_opts);
        let sorted_index = Index::builder()
            .schema(ssb.build())
            .settings(IndexSettings {
                sort_by_field: Some(IndexSortByField {
                    field: "kw".to_string(),
                    order: Order::Asc,
                }),
                ..Default::default()
            })
            .create_in_ram()?;
        sorted_index
            .tokenizers()
            .register("keyword", RawTokenizer::default());
        {
            let mut writer = sorted_index.writer_for_tests()?;
            for i in 0..2_000u32 {
                writer.add_document(doc!(sorted_kw => format!("k-{i:05}")))?;
            }
            writer.commit()?;
        }
        let sorted_searcher = sorted_index.reader()?.searcher();
        let probe_terms = vec![
            Term::from_field_text(sorted_kw, "k-00010"),
            Term::from_field_text(sorted_kw, "k-01500"),
        ];
        let probe_pred = Arc::new(SetPredicate::from_strings(["k-00010", "k-01500"]));

        let sorted_infer_sink = Arc::new(AtomicU8::new(0));
        let sorted_infer_hits = sorted_searcher.search(
            &TermSetQuery::new(probe_terms.clone()).with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(sorted_infer_sink.clone()),
                ..Default::default()
            }),
            &DocSetCollector,
        )?;
        assert_eq!(sorted_infer_hits.len(), 2);
        assert_eq!(
            sorted_infer_sink.load(Ordering::Relaxed),
            StrategyTag::Bitset as u8
        );

        let sorted_pred_infer_sink = Arc::new(AtomicU8::new(0));
        let sorted_pred_infer_hits = sorted_searcher.search(
            &TermPredicateQuery::new(sorted_kw, probe_pred.clone()).with_strategy_config(
                TermSetStrategyConfig {
                    strategy_sink: Some(sorted_pred_infer_sink.clone()),
                    ..Default::default()
                },
            ),
            &DocSetCollector,
        )?;
        assert_eq!(sorted_pred_infer_hits, sorted_infer_hits);
        assert_eq!(
            sorted_pred_infer_sink.load(Ordering::Relaxed),
            StrategyTag::Bitset as u8
        );

        let sorted_avail_sink = Arc::new(AtomicU8::new(0));
        let sorted_avail_hits = sorted_searcher.search(
            &TermSetQuery::new(probe_terms)
                .with_exact_fast_field(ExactFastField::Available)
                .with_strategy_config(TermSetStrategyConfig {
                    strategy_sink: Some(sorted_avail_sink.clone()),
                    ..Default::default()
                }),
            &DocSetCollector,
        )?;
        assert_eq!(sorted_avail_hits, sorted_infer_hits);
        assert_eq!(
            sorted_avail_sink.load(Ordering::Relaxed),
            StrategyTag::Gallop as u8
        );

        let sorted_pred_avail_sink = Arc::new(AtomicU8::new(0));
        let sorted_pred_avail_hits = sorted_searcher.search(
            &TermPredicateQuery::new(sorted_kw, probe_pred)
                .with_exact_fast_field(ExactFastField::Available)
                .with_strategy_config(TermSetStrategyConfig {
                    strategy_sink: Some(sorted_pred_avail_sink.clone()),
                    ..Default::default()
                }),
            &DocSetCollector,
        )?;
        assert_eq!(sorted_pred_avail_hits, sorted_infer_hits);
        assert_eq!(
            sorted_pred_avail_sink.load(Ordering::Relaxed),
            StrategyTag::Gallop as u8
        );
        Ok(())
    }
}
