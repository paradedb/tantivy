//! Predicate-backed set-membership API (`TermPredicate`, `TermPredicateQuery`),
//! streaming batch evaluators, and lazy fast-field `DocSet`s shared with
//! [`TermSetWeight`].
//!
//! [`TermSetWeight`]: super::TermSetWeight

use std::ops::{Bound, Range, RangeInclusive};
use std::sync::Arc;
use std::{fmt, io};

use columnar::{BytesColumn, Cardinality, Column, ColumnType, SortedTermSlice};
use common::bounds::BoundsRange;
use common::{BitSet, TinySet};

use super::term_set_strategy::TermSetStrategyConfig;
use super::term_set_weight::TermSetWeight;
use crate::docset::SeekDangerResult;
use crate::query::{EnableScoring, Query, Weight};
use crate::schema::Field;
use crate::{DocId, DocSet, Term, TERMINATED};

/// Number of dictionary entries evaluated per `TermPredicate` batch call.
pub(crate) const DICT_BATCH_LEN: usize = 4096;

/// Maximum number of documents evaluated per `PredicateColumnDocSet` window refill (`16 * 64`).
const COLUMN_WINDOW_WORDS: usize = 16;
const COLUMN_WINDOW_DOCS: u32 = (COLUMN_WINDOW_WORDS * 64) as u32;

/// Minimum SSTable blocks in the bounded dictionary slice before cached
/// point lookups are considered over a single sequential dictionary pass.
pub(crate) const MIN_BLOCKS_FOR_POINT_LOOKUP: u64 = 16;

/// Upper cap on unique `ord_to_bytes` point lookups before promoting to a
/// dictionary pass, bounding worst-case block-decompression overhead when
/// upstream `size_hint()` underestimates candidate cardinality.
pub(crate) const MAX_POINT_LOOKUP_BUDGET: usize = 32;

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

/// Vectorized membership oracle evaluated over dictionary slices or `Column<u64>` batches.
pub trait TermPredicate: fmt::Debug + Send + Sync + 'static {
    /// Evaluate membership on `offsets.len() - 1` byte slices packed contiguously in `data`,
    /// where item `i` is `&data[offsets[i] as usize..offsets[i + 1] as usize]`.
    ///
    /// `out` is pre-sized to `num_items.div_ceil(64)` and cleared to `TinySet::EMPTY`;
    /// implementations set bit `i % 64` of `out[i / 64]` for each matching item `i`.
    fn filter_bytes(&self, data: &[u8], offsets: &[u32], out: &mut [TinySet]) -> crate::Result<()>;

    /// Evaluate membership on `values.len()` `MonotonicallyMappableToU64`-encoded values
    /// of `col_type` (`I64`, `U64`, `F64`, `Bool`, `DateTime`).
    ///
    /// `out` is pre-sized to `values.len().div_ceil(64)` and cleared to `TinySet::EMPTY`;
    /// implementations set bit `i % 64` of `out[i / 64]` for each matching item `i`.
    fn filter_monotonic_u64s(
        &self,
        col_type: ColumnType,
        values: &[u64],
        out: &mut [TinySet],
    ) -> crate::Result<()>;

    /// Optional cardinality hint (`K`) when the caller knows the build-side distinct key count.
    fn estimated_distinct_keys(&self) -> Option<u64> {
        None
    }
}

/// A Tantivy `Query` that filters a single `Field` through a [`TermPredicate`],
/// executing via [`TermSetWeight`].
#[derive(Clone, Debug)]
pub struct TermPredicateQuery {
    field: Field,
    predicate: Arc<dyn TermPredicate>,
    bounds: BoundsRange<Term>,
    exact_postings: ExactPostings,
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
            self.strategy_config.clone(),
        )?
        .with_bitmap_postings(enable_scoring.bitmap_postings_enabled());
        Ok(Box::new(weight))
    }
}

// ---------------------------------------------------------------------------
// Shared batch evaluation & dictionary ordinal resolution helpers
// ---------------------------------------------------------------------------

pub(crate) fn prepare_tinyset_buf(buf: &mut Vec<TinySet>, num_items: usize) {
    buf.clear();
    buf.resize(num_items.div_ceil(64), TinySet::EMPTY);
}

#[inline]
pub(crate) fn try_for_each_set_bit(
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
        predicate.filter_bytes(&self.data, &self.offsets, &mut self.out)?;
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
        predicate.filter_monotonic_u64s(col_type, &self.vals, &mut self.out)?;
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
    if ord_range.is_empty() {
        return Ok((matched_ords, retained_keys));
    }

    let mut lo_key = Vec::new();
    let mut hi_key = Vec::new();
    let mut builder = dict.range().interruptible();
    if ord_range.start > 0 && dict.ord_to_term(ord_range.start, &mut lo_key)? {
        builder = builder.ge(&lo_key);
    }
    if ord_range.end < u64::from(num_terms) && dict.ord_to_term(ord_range.end - 1, &mut hi_key)? {
        builder = builder.le(&hi_key);
    }
    let mut stream = builder.into_stream()?;

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
    while stream.advance() {
        let ord = stream.term_ord();
        if ord < ord_range.start {
            continue;
        }
        if ord >= ord_range.end {
            break;
        }
        let Ok(ord_u32) = u32::try_from(ord) else {
            break;
        };
        batcher.push(ord_u32, stream.key(), predicate, &mut on_match)?;
    }
    batcher.finish(predicate, &mut on_match)?;
    Ok((matched_ords, retained_keys))
}

// ---------------------------------------------------------------------------
// CachedDictPredicateDocSet — candidate-probing DocSet over a dictionary fast
// field that caches tested ordinals and auto-promotes to a full dictionary
// pass if unique point lookups exceed `remaining_lookups`.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum DictKeyProbe {
    SortedTerms(Arc<[Vec<u8>]>),
    Predicate(Arc<dyn TermPredicate>),
}

impl DictKeyProbe {
    fn matches_single(&self, key: &[u8], scratch_out: &mut [TinySet; 1]) -> bool {
        match self {
            DictKeyProbe::SortedTerms(terms) => {
                terms.binary_search_by(|k| k.as_slice().cmp(key)).is_ok()
            }
            DictKeyProbe::Predicate(pred) => {
                scratch_out[0] = TinySet::EMPTY;
                let offsets = [0u32, key.len() as u32];
                pred.filter_bytes(key, &offsets, scratch_out)
                    .expect("predicate evaluation should succeed");
                scratch_out[0].contains(0)
            }
        }
    }

    fn matches_batch(&self, data: &[u8], offsets: &[u32], out: &mut Vec<TinySet>) {
        let count = offsets.len().saturating_sub(1);
        prepare_tinyset_buf(out, count);
        if count == 0 {
            return;
        }
        match self {
            DictKeyProbe::SortedTerms(terms) => {
                for i in 0..count {
                    let key = &data[offsets[i] as usize..offsets[i + 1] as usize];
                    if terms.binary_search_by(|k| k.as_slice().cmp(key)).is_ok() {
                        out[i / 64].insert_mut((i % 64) as u32);
                    }
                }
            }
            DictKeyProbe::Predicate(pred) => {
                pred.filter_bytes(data, offsets, out)
                    .expect("predicate evaluation should succeed");
            }
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

#[derive(Clone, Copy, Debug)]
enum DictLookupMode {
    PointWithFallback { remaining_lookups: usize },
    FullyPopulated,
}

pub(crate) struct CachedDictPredicateDocSet {
    bytes_col: BytesColumn,
    ord_range: Range<u64>,
    probe: DictKeyProbe,
    tested_ords: BitSet,
    matched_ords: BitSet,
    mode: DictLookupMode,
    scratch_bytes: Vec<u8>,
    scratch_out: [TinySet; 1],
    doc_id: DocId,
    positioned: bool,
}

impl CachedDictPredicateDocSet {
    pub(crate) fn new_unpositioned(
        bytes_col: BytesColumn,
        ord_range: Range<u64>,
        probe: DictKeyProbe,
        point_lookup_budget: usize,
    ) -> Self {
        let num_terms = bytes_col.num_terms() as u32;
        let empty = bytes_col.num_rows() == 0 || ord_range.is_empty();
        Self {
            bytes_col,
            ord_range,
            probe,
            tested_ords: BitSet::with_max_value(num_terms),
            matched_ords: BitSet::with_max_value(num_terms),
            mode: DictLookupMode::PointWithFallback {
                remaining_lookups: point_lookup_budget,
            },
            scratch_bytes: Vec::new(),
            scratch_out: [TinySet::EMPTY; 1],
            doc_id: if empty { TERMINATED } else { 0 },
            positioned: empty,
        }
    }

    fn populate_all(&mut self) {
        if !matches!(self.mode, DictLookupMode::FullyPopulated) {
            self.matched_ords = self
                .probe
                .resolve_all(&self.bytes_col, self.ord_range.clone());
            self.mode = DictLookupMode::FullyPopulated;
        }
    }

    #[inline]
    fn check_ord(&mut self, ord: u64) -> bool {
        if !self.ord_range.contains(&ord) {
            return false;
        }
        let Ok(ord_u32) = u32::try_from(ord) else {
            return false;
        };
        match &mut self.mode {
            DictLookupMode::FullyPopulated => self.matched_ords.contains(ord_u32),
            DictLookupMode::PointWithFallback { remaining_lookups } => {
                if self.tested_ords.contains(ord_u32) {
                    return self.matched_ords.contains(ord_u32);
                }
                if *remaining_lookups == 0 {
                    self.populate_all();
                    return self.matched_ords.contains(ord_u32);
                }
                *remaining_lookups -= 1;
                self.scratch_bytes.clear();
                let found = self
                    .bytes_col
                    .ord_to_bytes(ord, &mut self.scratch_bytes)
                    .expect("dictionary ordinal lookup should succeed");
                let is_match = found
                    && self
                        .probe
                        .matches_single(&self.scratch_bytes, &mut self.scratch_out);
                self.tested_ords.insert(ord_u32);
                if is_match {
                    self.matched_ords.insert(ord_u32);
                }
                is_match
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
        self.populate_all();
        if self.matched_ords.len() == 0 {
            self.doc_id = TERMINATED;
            return TERMINATED;
        }
        let ords = self.bytes_col.ords();
        for next_doc in target..max_doc {
            for ord in ords.values_for_doc(next_doc) {
                if let Ok(ord_u32) = u32::try_from(ord) {
                    if ord_u32 < self.matched_ords.max_value()
                        && self.matched_ords.contains(ord_u32)
                    {
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

        if matches!(self.mode, DictLookupMode::FullyPopulated) && self.matched_ords.len() == 0 {
            mask[..words].fill(TinySet::EMPTY);
            return TERMINATED;
        }

        if let DictLookupMode::PointWithFallback { remaining_lookups } = self.mode {
            let mut uncached_ords: Vec<u32> = Vec::new();
            let mut promoted = false;
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
                    for ord in self.bytes_col.ords().values_for_doc(doc) {
                        if !self.ord_range.contains(&ord) {
                            continue;
                        }
                        let Ok(ord_u32) = u32::try_from(ord) else {
                            continue;
                        };
                        if !self.tested_ords.contains(ord_u32) {
                            if uncached_ords.len() >= remaining_lookups {
                                promoted = true;
                                break 'collect;
                            }
                            self.tested_ords.insert(ord_u32);
                            uncached_ords.push(ord_u32);
                        }
                    }
                }
            }
            if promoted {
                self.populate_all();
            } else if !uncached_ords.is_empty() {
                if let DictLookupMode::PointWithFallback {
                    ref mut remaining_lookups,
                } = self.mode
                {
                    *remaining_lookups = remaining_lookups.saturating_sub(uncached_ords.len());
                }
                let mut data = Vec::new();
                let mut offsets = Vec::with_capacity(uncached_ords.len() + 1);
                offsets.push(0u32);
                let mut valid_ords = Vec::with_capacity(uncached_ords.len());
                for ord_u32 in uncached_ords {
                    let start = data.len();
                    let found = self
                        .bytes_col
                        .ord_to_bytes(u64::from(ord_u32), &mut data)
                        .expect("dictionary ordinal lookup should succeed");
                    if found {
                        offsets.push(data.len() as u32);
                        valid_ords.push(ord_u32);
                    } else {
                        data.truncate(start);
                    }
                }
                let mut batch_out = Vec::new();
                self.probe.matches_batch(&data, &offsets, &mut batch_out);
                let matched_ords = &mut self.matched_ords;
                let _ = try_for_each_set_bit(&batch_out, valid_ords.len(), |idx| {
                    matched_ords.insert(valid_ords[idx]);
                    Ok(())
                });
            }
        }

        if matches!(self.mode, DictLookupMode::FullyPopulated) && self.matched_ords.len() == 0 {
            mask[..words].fill(TinySet::EMPTY);
            return TERMINATED;
        }

        let ords = self.bytes_col.ords();
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
                    self.ord_range.contains(&ord)
                        && u32::try_from(ord).is_ok_and(|o| {
                            o < self.matched_ords.max_value() && self.matched_ords.contains(o)
                        })
                });
                if matched {
                    keep.insert_mut(bit);
                }
            }
            *word = keep;
        }
        next_bound
    }

    fn is_empty_in_range(&mut self, start: DocId, end: DocId) -> bool {
        start > end || start >= self.bytes_col.num_rows()
    }

    fn doc(&self) -> DocId {
        self.doc_id
    }

    fn size_hint(&self) -> u32 {
        self.bytes_col
            .num_rows()
            .saturating_sub(self.doc_id.saturating_add(1))
    }

    fn cost(&self) -> crate::DocSetCost {
        let probe = if matches!(self.mode, DictLookupMode::FullyPopulated) {
            crate::ProbeCost::FAST_FIELD_LOOKUP
        } else {
            crate::ProbeCost::VECTORIZED_PREDICATE
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
                self.predicate
                    .filter_monotonic_u64s(
                        self.col_type,
                        &self.scratch_vals,
                        &mut self.window_mask[..num_words],
                    )
                    .expect("numeric predicate evaluation should succeed");
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
        self.predicate
            .filter_monotonic_u64s(self.col_type, &self.scratch_vals, &mut self.scratch_out)
            .expect("numeric predicate evaluation should succeed");
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
                    self.predicate
                        .filter_monotonic_u64s(self.col_type, std::slice::from_ref(&v), &mut out)
                        .expect("numeric predicate evaluation should succeed");
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
                    self.predicate
                        .filter_monotonic_u64s(
                            self.col_type,
                            &self.scratch_vals,
                            &mut self.scratch_out,
                        )
                        .expect("numeric predicate evaluation should succeed");
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
                self.predicate
                    .filter_monotonic_u64s(self.col_type, &self.scratch_vals, &mut mask[..words])
                    .expect("numeric predicate evaluation should succeed");
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
        self.predicate
            .filter_monotonic_u64s(self.col_type, &self.scratch_vals, &mut self.scratch_out)
            .expect("numeric predicate evaluation should succeed");
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
        self.doc_bounds
            .end
            .saturating_sub(self.doc_id.max(self.doc_bounds.start).saturating_add(1))
    }

    fn cost(&self) -> crate::DocSetCost {
        let window_docs = u64::from(self.doc_bounds.end.saturating_sub(self.doc_bounds.start));
        let probe = crate::ProbeCost::VECTORIZED_PREDICATE;
        crate::DocSetCost::new(
            crate::DriveCost(window_docs.saturating_mul(probe.per_doc.0)),
            probe,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Bound;
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
    use std::sync::Arc;

    use columnar::{ColumnType, MonotonicallyMappableToU64};
    use common::bounds::BoundsRange;
    use common::TinySet;
    use proptest::prelude::*;
    use rustc_hash::FxHashSet;

    use super::{ExactPostings, TermPredicate, TermPredicateQuery};
    use crate::collector::{DocSetCollector, TopDocs};
    use crate::query::{
        BooleanQuery, ConstScoreQuery, FastFieldTermSetQuery, Occur, RangeQuery, StrategyTag,
        TermQuery, TermSetQuery, TermSetStrategyConfig,
    };
    use crate::schema::{
        IndexRecordOption, NumericOptions, SchemaBuilder, TextFieldIndexing, TextOptions, FAST,
        STRING, TEXT,
    };
    use crate::{Index, IndexSettings, IndexSortByField, Order, Term};

    #[derive(Debug, Default)]
    struct EvalTelemetry {
        calls: AtomicUsize,
        items: AtomicUsize,
    }

    #[derive(Debug)]
    struct SetPredicate {
        bytes_set: FxHashSet<Vec<u8>>,
        i64_set: FxHashSet<i64>,
        bytes_calls: AtomicUsize,
        bytes_items: AtomicUsize,
        u64_telemetry: EvalTelemetry,
    }

    impl SetPredicate {
        fn from_strings<'a>(items: impl IntoIterator<Item = &'a str>) -> Self {
            Self {
                bytes_set: items.into_iter().map(|s| s.as_bytes().to_vec()).collect(),
                i64_set: FxHashSet::default(),
                bytes_calls: AtomicUsize::new(0),
                bytes_items: AtomicUsize::new(0),
                u64_telemetry: EvalTelemetry::default(),
            }
        }

        fn from_i64s(items: impl IntoIterator<Item = i64>) -> Self {
            Self {
                bytes_set: FxHashSet::default(),
                i64_set: items.into_iter().collect(),
                bytes_calls: AtomicUsize::new(0),
                bytes_items: AtomicUsize::new(0),
                u64_telemetry: EvalTelemetry::default(),
            }
        }
    }

    impl TermPredicate for SetPredicate {
        fn filter_bytes(
            &self,
            data: &[u8],
            offsets: &[u32],
            out: &mut [TinySet],
        ) -> crate::Result<()> {
            let count = offsets.len().saturating_sub(1);
            self.bytes_calls.fetch_add(1, Ordering::Relaxed);
            self.bytes_items.fetch_add(count, Ordering::Relaxed);
            for i in 0..count {
                let slice = &data[offsets[i] as usize..offsets[i + 1] as usize];
                if self.bytes_set.contains(slice) {
                    out[i / 64].insert_mut((i % 64) as u32);
                }
            }
            Ok(())
        }

        fn filter_monotonic_u64s(
            &self,
            col_type: ColumnType,
            values: &[u64],
            out: &mut [TinySet],
        ) -> crate::Result<()> {
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
            Ok(())
        }
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
        // evaluates only the 2 candidate ordinals rather than decoding all 3,000 dictionary
        // entries.
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
        assert_eq!(selective_pred.bytes_items.load(Ordering::Relaxed), 2);
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
                    Box::new(ConstScoreQuery::new(range_q.clone(), 0.0)),
                ),
            ])
        };
        let nested_pred =
            make_nested_scored(Box::new(TermPredicateQuery::new(owner, wpred.clone())));
        let (_, flat_min_should) = nested_pred.flattened_clauses();
        assert_eq!(
            flat_min_should, 1,
            "nested range-constrained disjunction should flatten"
        );

        let explicit_terms: Vec<Term> = keys
            .iter()
            .map(|&k| Term::from_field_i64(owner, k))
            .collect();
        let nested_set = make_nested_scored(Box::new(TermSetQuery::new(explicit_terms)));
        let got_top = wsearcher.search(&nested_pred, &TopDocs::with_limit(15).order_by_score())?;
        let exp_top = wsearcher.search(&nested_set, &TopDocs::with_limit(15).order_by_score())?;
        assert_eq!(got_top, exp_top);
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
        let pred = Arc::new(SetPredicate::from_i64s(keys.iter().copied()));
        let reset = |p: &SetPredicate| {
            p.u64_telemetry.calls.store(0, Ordering::Relaxed);
            p.u64_telemetry.items.store(0, Ordering::Relaxed);
            p.bytes_calls.store(0, Ordering::Relaxed);
            p.bytes_items.store(0, Ordering::Relaxed);
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
            str_pred.bytes_calls.load(Ordering::Relaxed) <= 3,
            "CachedDictPredicateDocSet should batch filter_bytes calls, got {}",
            str_pred.bytes_calls.load(Ordering::Relaxed)
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
                        TermPredicateQuery::new(full_col, pred.clone()),
                        0.0,
                    )),
                ),
            ]);
            let q_ref = BooleanQuery::new(vec![
                (Occur::Must, terms_q),
                (
                    Occur::Must,
                    Box::new(ConstScoreQuery::new(
                        TermSetQuery::new(full_ref_terms.clone()),
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

        // 4b. `BlockWandUnionScorer` via `Weight::pruning_scorer` (streaming full-scan path used
        //     by ParadeDB `Scanner`): verify both direct `PredicateColumnDocSet` and combined
        //     `BatchFilteredScorer(RangeDocSet, PredicateColumnDocSet)` evaluate only exact
        //     surviving candidates (zero speculative contiguous window probes) in ramped batches.
        let enable = crate::query::EnableScoring::enabled_from_searcher(&searcher);
        let seg_reader = searcher.segment_reader(0);
        let expected_union_all = (0..8_192i64).filter(|i| i % 5 == 0 || i % 7 == 0).count();
        let expected_union_range = (2_000..4_000i64)
            .filter(|i| i % 5 == 0 || i % 7 == 0)
            .count();
        for (name, range_filter, expected_items, max_calls) in [
            ("union_direct", None, expected_union_all, 6usize),
            (
                "union_with_range_batch_filtered",
                Some(crate::query::RangeQuery::new(
                    Bound::Included(Term::from_field_i64(full_col, 2_000)),
                    Bound::Excluded(Term::from_field_i64(full_col, 4_000)),
                )),
                expected_union_range,
                6usize,
            ),
        ] {
            reset(&pred);
            let mut pred_clauses: Vec<(Occur, Box<dyn crate::query::Query>)> = vec![
                (Occur::Should, term_q("hit2")),
                (Occur::Should, term_q("hit3")),
                (
                    Occur::Must,
                    Box::new(ConstScoreQuery::new(
                        TermPredicateQuery::new(full_col, pred.clone()),
                        0.0,
                    )),
                ),
            ];
            let mut ref_clauses: Vec<(Occur, Box<dyn crate::query::Query>)> = vec![
                (Occur::Should, term_q("hit2")),
                (Occur::Should, term_q("hit3")),
                (
                    Occur::Must,
                    Box::new(ConstScoreQuery::new(
                        TermSetQuery::new(full_ref_terms.clone()),
                        0.0,
                    )),
                ),
            ];
            if let Some(rq) = range_filter {
                pred_clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(rq.clone(), 0.0))));
                ref_clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(rq, 0.0))));
            }
            let mut pred_q = BooleanQuery::new(pred_clauses);
            pred_q.set_minimum_number_should_match(1);
            let mut ref_q = BooleanQuery::new(ref_clauses);
            ref_q.set_minimum_number_should_match(1);

            let mut pred_scorer =
                pred_q
                    .weight(enable)?
                    .pruning_scorer(seg_reader, 1.0, crate::Score::MIN)?;
            let mut got_docs = Vec::new();
            while pred_scorer.doc() != TERMINATED {
                got_docs.push((pred_scorer.doc(), pred_scorer.score()));
                pred_scorer.advance();
            }
            let mut ref_scorer =
                ref_q
                    .weight(enable)?
                    .pruning_scorer(seg_reader, 1.0, crate::Score::MIN)?;
            let mut exp_docs = Vec::new();
            while ref_scorer.doc() != TERMINATED {
                exp_docs.push((ref_scorer.doc(), ref_scorer.score()));
                ref_scorer.advance();
            }
            assert_eq!(got_docs, exp_docs, "pruning_scorer mismatch for {name}");
            let calls = pred.u64_telemetry.calls.load(Ordering::Relaxed);
            let items = pred.u64_telemetry.items.load(Ordering::Relaxed);
            // `PredicateColumnDocSet::new` runs one eager 1,024-doc positioning window when
            // `union_hint >= num_docs / 32`; every subsequent probe must go through
            // `seek_danger_batch` on exact surviving candidates (zero speculative windows).
            assert!(
                calls > 0 && calls <= max_calls + 1,
                "{name}: expected <= {} batched calls, got {calls}",
                max_calls + 1
            );
            assert_eq!(
                items,
                expected_items + super::COLUMN_WINDOW_DOCS as usize,
                "{name}: should evaluate only surviving candidates plus initial positioning window"
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
            let pred = Arc::new(SetPredicate {
                bytes_set: str_strings.iter().map(|s| s.as_bytes().to_vec()).collect(),
                i64_set: query_vals.iter().copied().collect(),
                bytes_calls: AtomicUsize::new(0),
                bytes_items: AtomicUsize::new(0),
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
        }
    }
}
