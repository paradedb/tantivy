//! Unified `TermSetWeight` executing both explicit `TermSetQuery` key sets and
//! vectorized `TermPredicateQuery` filters across dictionary fast fields,
//! numeric/IP fast fields, and inverted-index posting lists.
//!
//! Strategy selection separates two orthogonal costs:
//! 1. Choosing the execution substrate (`BytesColumn` / `Column<u64>` fast field vs
//!    `InvertedIndexReader` posting lists) based on segment sort order, upstream candidate
//!    selectivity `C`, dictionary size `V`, and expected match count `K`.
//! 2. Resolving matching dictionary entries on whichever dictionary was chosen (batched
//!    block-skipping lookup for explicit sorted keys vs single-pass batch streaming for opaque
//!    `TermPredicate`s).

use std::net::Ipv6Addr;
use std::ops::{Bound, RangeInclusive};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::{fmt, io};

use columnar::{BytesColumn, Column, ColumnType};
use common::bounds::BoundsRange;
use common::BitSet;
use rustc_hash::{FxHashMap, FxHashSet};
use tantivy_fst::raw::CompiledAddr;
use tantivy_fst::{Automaton, Map};

use super::term_predicate::{
    resolve_explicit_dict_ordinals, resolve_remaining_into_cache, stream_predicate_dict_ordinals,
    BytesBatcher, CachedDictPredicateDocSet, DictEvalScratch, DictKeyProbe, ExactFastField,
    ExactPostings, PredicateColumnDocSet, SegmentDictEntry, SegmentDictState, TermPredicate,
    U64Batcher,
};
use super::term_set_bitset::bitset_from_postings_scorer;
use super::term_set_gallop::TermSetGallopDocSet;
use super::term_set_strategy::{
    max_admissible_bitset_keys, select_strategy, sorted_range_in_column, try_gallop_order,
    PlannerInputs, StrategyChoice, StrategyTag, TermSetStrategyConfig,
};
use crate::docset::SeekDangerResult;
use crate::index::{SegmentId, SegmentReader};
use crate::postings::TermInfo;
use crate::query::bitset::or_term_info_into_bitset;
use crate::query::range_query::sorted_internals::sorted_doc_range_for_bounds;
use crate::query::{
    AutomatonWeight, BitSetDocSet, CandidateHint, ConstScorer, EmptyScorer, Explanation, Scorer,
    UnpositionedScorer, Weight,
};
use crate::schema::{Field, FieldType, DATE_TIME_PRECISION_INDEXED};
use crate::termdict::TermStreamer;
use crate::{DocId, DocSet, Score, TantivyError, Term, TERMINATED};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FastSubstrate {
    None,
    Dictionary,
    Numeric,
    Ipv6,
}

#[derive(Clone, Debug)]
enum ExplicitTyped {
    Numeric(Vec<u64>),
    Ipv6(Vec<Ipv6Addr>),
    Bytes,
}

#[derive(Debug)]
struct ExplicitTerms {
    terms_bytes: Arc<[Vec<u8>]>,
    typed: ExplicitTyped,
    terms_fst: OnceLock<Arc<Map<Vec<u8>>>>,
}

impl ExplicitTerms {
    fn fst(&self) -> crate::Result<Arc<Map<Vec<u8>>>> {
        if let Some(fst) = self.terms_fst.get() {
            return Ok(Arc::clone(fst));
        }
        let map = Map::from_iter(self.terms_bytes.iter().map(|k| (k.as_slice(), 0u64)))
            .map_err(io::Error::other)?;
        let arc = Arc::new(map);
        let _ = self.terms_fst.set(Arc::clone(&arc));
        Ok(arc)
    }
}

#[derive(Clone, Debug)]
enum KeyFilter {
    Terms(Arc<ExplicitTerms>),
    Predicate {
        predicate: Arc<dyn TermPredicate>,
        bounds: BoundsRange<Term>,
    },
}

impl KeyFilter {
    fn is_definitely_empty(&self) -> bool {
        match self {
            KeyFilter::Terms(t) => t.terms_bytes.is_empty(),
            KeyFilter::Predicate { predicate, .. } => {
                predicate.estimated_distinct_keys() == Some(0)
            }
        }
    }

    fn known_key_count(&self) -> Option<u64> {
        match self {
            KeyFilter::Terms(t) => Some(t.terms_bytes.len() as u64),
            KeyFilter::Predicate { predicate, .. } => predicate.estimated_distinct_keys(),
        }
    }

    fn byte_bounds(&self) -> (Bound<&[u8]>, Bound<&[u8]>) {
        match self {
            KeyFilter::Terms(t) => match (t.terms_bytes.first(), t.terms_bytes.last()) {
                (Some(lo), Some(hi)) => (
                    Bound::Included(lo.as_slice()),
                    Bound::Included(hi.as_slice()),
                ),
                _ => (Bound::Unbounded, Bound::Unbounded),
            },
            KeyFilter::Predicate { bounds, .. } => (
                term_bound_bytes(&bounds.lower_bound),
                term_bound_bytes(&bounds.upper_bound),
            ),
        }
    }
}

fn classify_substrates(
    field_type: &FieldType,
    requested_postings: ExactPostings,
    exact_fast_field: ExactFastField,
    for_explicit_terms: bool,
) -> (FastSubstrate, ExactPostings) {
    let fast_substrate = match field_type {
        FieldType::IpAddr(opts) if opts.is_fast() && for_explicit_terms => FastSubstrate::Ipv6,
        FieldType::U64(opts)
        | FieldType::I64(opts)
        | FieldType::F64(opts)
        | FieldType::Bool(opts)
            if opts.is_fast() =>
        {
            FastSubstrate::Numeric
        }
        FieldType::Date(opts)
            if opts.is_fast()
                && (!for_explicit_terms
                    || !opts.is_indexed()
                    || opts.get_precision() == DATE_TIME_PRECISION_INDEXED) =>
        {
            FastSubstrate::Numeric
        }
        FieldType::Bytes(opts) if opts.is_fast() => FastSubstrate::Dictionary,
        #[allow(deprecated)]
        FieldType::Str(opts)
            if opts.is_fast()
                && (exact_fast_field == ExactFastField::Available
                    || opts.get_indexing_options().is_none()
                    || opts.get_fast_field_tokenizer_name()
                        == opts.get_indexing_options().map(|i| i.tokenizer())) =>
        {
            FastSubstrate::Dictionary
        }
        _ => FastSubstrate::None,
    };

    let postings_compatible = field_type.is_indexed()
        && !matches!(
            field_type,
            FieldType::Date(opts)
                if !for_explicit_terms && opts.get_precision() != DATE_TIME_PRECISION_INDEXED
        );
    let exact_postings = if requested_postings == ExactPostings::Available && postings_compatible {
        ExactPostings::Available
    } else {
        ExactPostings::Unavailable
    };

    (fast_substrate, exact_postings)
}

/// Query-scoped per-segment dictionary resolution cache shared across clones of a
/// [`TermSetWeight`], preventing repeated candidate-probing calls (such as vector shortlist
/// chunks) from re-scanning or re-decompressing the same segment dictionary.
#[derive(Clone, Default)]
struct SegmentDictCache(Arc<RwLock<FxHashMap<SegmentId, Arc<Mutex<SegmentDictEntry>>>>>);

impl fmt::Debug for SegmentDictCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentDictCache").finish_non_exhaustive()
    }
}

/// Unified `Weight` for single-field set-membership queries (`TermSetQuery`,
/// `FastFieldTermSetQuery`, and `TermPredicateQuery`).
#[derive(Clone, Debug)]
pub struct TermSetWeight {
    field: Field,
    filter: KeyFilter,
    fast_substrate: FastSubstrate,
    exact_postings: ExactPostings,
    strategy_config: TermSetStrategyConfig,
    bitmap_enabled: bool,
    dict_cache: SegmentDictCache,
}

impl TermSetWeight {
    /// Construct a `TermSetWeight` from an explicit slice of `Term`s.
    pub fn from_terms(
        field: Field,
        terms: &[Term],
        field_type: &FieldType,
        exact_fast_field: ExactFastField,
        strategy_config: TermSetStrategyConfig,
    ) -> crate::Result<Self> {
        let (fast_substrate, exact_postings) =
            classify_substrates(field_type, ExactPostings::Available, exact_fast_field, true);
        if fast_substrate == FastSubstrate::None && exact_postings == ExactPostings::Unavailable {
            return Err(TantivyError::SchemaError(format!(
                "Field {field:?} is neither fast nor indexed; TermSetQuery has no execution path"
            )));
        }

        let mut terms_bytes: Vec<Vec<u8>> = terms
            .iter()
            .map(|t| t.serialized_value_bytes().to_vec())
            .collect();
        terms_bytes.sort_unstable();
        terms_bytes.dedup();

        let typed = if let Some(first) = terms.first() {
            let first_val = first.value();
            if first_val.as_ip_addr().is_some() {
                if !field_type.is_ip_addr() && !field_type.is_json() {
                    return Err(TantivyError::InvalidArgument(format!(
                        "TermSet for field {field:?} contains IP addresses, but field type is \
                         {field_type:?}"
                    )));
                }
                let mut values = Vec::with_capacity(terms.len());
                for term in terms {
                    let Some(v) = term.value().as_ip_addr() else {
                        return Err(TantivyError::InvalidArgument(format!(
                            "Expected term with ip address, but got {term:?}"
                        )));
                    };
                    values.push(v);
                }
                ExplicitTyped::Ipv6(values)
            } else if first_val.as_u64_lenient().is_some() && !field_type.is_json() {
                if field_type.is_ip_addr() {
                    return Err(TantivyError::InvalidArgument(format!(
                        "TermSet for field {field:?} contains numeric values, but field type is \
                         {field_type:?}"
                    )));
                }
                let mut values = Vec::with_capacity(terms.len());
                for term in terms {
                    let Some(v) = term.value().as_u64_lenient() else {
                        return Err(TantivyError::InvalidArgument(format!(
                            "Expected term with u64, i64, f64, bool, or date, but got {term:?}"
                        )));
                    };
                    values.push(v);
                }
                values.sort_unstable();
                values.dedup();
                ExplicitTyped::Numeric(values)
            } else {
                ExplicitTyped::Bytes
            }
        } else {
            ExplicitTyped::Bytes
        };

        Ok(Self {
            field,
            filter: KeyFilter::Terms(Arc::new(ExplicitTerms {
                terms_bytes: terms_bytes.into(),
                typed,
                terms_fst: OnceLock::new(),
            })),
            fast_substrate,
            exact_postings,
            strategy_config,
            bitmap_enabled: false,
            dict_cache: SegmentDictCache::default(),
        })
    }

    /// Construct a `TermSetWeight` from a vectorized [`TermPredicate`].
    pub fn from_predicate(
        field: Field,
        field_type: &FieldType,
        predicate: Arc<dyn TermPredicate>,
        bounds: BoundsRange<Term>,
        requested_postings: ExactPostings,
        exact_fast_field: ExactFastField,
        strategy_config: TermSetStrategyConfig,
    ) -> crate::Result<Self> {
        if matches!(
            field_type,
            FieldType::JsonObject(_) | FieldType::Facet(_) | FieldType::IpAddr(_)
        ) {
            return Err(TantivyError::SchemaError(format!(
                "TermPredicateQuery does not support field {field:?} of type {field_type:?}"
            )));
        }
        let (fast_substrate, exact_postings) =
            classify_substrates(field_type, requested_postings, exact_fast_field, false);
        if fast_substrate == FastSubstrate::None && exact_postings == ExactPostings::Unavailable {
            let hint = if matches!(field_type, FieldType::Str(opts) if opts.is_fast()) {
                " (fast-field and indexing tokenizers differ under ExactFastField::Infer; pass \
                 ExactFastField::Available if the fast field stores exact predicate values)"
            } else {
                ""
            };
            return Err(TantivyError::SchemaError(format!(
                "Field {field:?} has neither a supported fast-field column nor exact \
                 postings{hint}"
            )));
        }
        Ok(Self {
            field,
            filter: KeyFilter::Predicate { predicate, bounds },
            fast_substrate,
            exact_postings,
            strategy_config,
            bitmap_enabled: false,
            dict_cache: SegmentDictCache::default(),
        })
    }

    pub(crate) fn with_bitmap_postings(mut self, enabled: bool) -> Self {
        self.bitmap_enabled = enabled;
        self
    }

    fn record_strategy_tag(&self, tag: StrategyTag) {
        if let Some(sink) = self.strategy_config.strategy_sink.as_ref() {
            sink.store(tag as u8, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn automaton_scorer(
        &self,
        reader: &SegmentReader,
        explicit: &ExplicitTerms,
        boost: Score,
    ) -> crate::Result<Box<dyn Scorer>> {
        if explicit.terms_bytes.is_empty() {
            return Ok(Box::new(EmptyScorer));
        }
        let weight = AutomatonWeight::new(self.field, SetDfaWrapper(explicit.fst()?))
            .with_bitmap_postings(self.bitmap_enabled);
        weight.scorer(reader, boost)
    }

    fn scorer_with_candidates(
        &self,
        reader: &SegmentReader,
        boost: Score,
        candidate_hint: Option<CandidateHint>,
    ) -> crate::Result<UnpositionedScorer> {
        if reader.max_doc() == 0 || self.filter.is_definitely_empty() {
            if matches!(self.filter, KeyFilter::Predicate { .. }) {
                self.record_strategy_tag(StrategyTag::Empty);
            }
            return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
        }

        let field_entry = reader.schema().get_field_entry(self.field);
        let field_name = field_entry.name();
        let field_type = field_entry.field_type();

        match self.fast_substrate {
            FastSubstrate::Ipv6 => {
                self.dispatch_ipv6_fast(reader, field_name, boost, candidate_hint)
            }
            FastSubstrate::Dictionary => {
                self.dispatch_dictionary_fast(reader, field_name, field_type, boost, candidate_hint)
            }
            FastSubstrate::Numeric => {
                self.dispatch_numeric_fast(reader, field_name, field_type, boost, candidate_hint)
            }
            FastSubstrate::None => self
                .dispatch_postings(reader, field_type, boost)
                .map(UnpositionedScorer::from_positioned),
        }
    }

    fn dispatch_ipv6_fast(
        &self,
        reader: &SegmentReader,
        field_name: &str,
        boost: Score,
        candidate_hint: Option<CandidateHint>,
    ) -> crate::Result<UnpositionedScorer> {
        let KeyFilter::Terms(explicit) = &self.filter else {
            return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
        };
        let ExplicitTyped::Ipv6(values) = &explicit.typed else {
            return self
                .dispatch_postings(
                    reader,
                    reader.schema().get_field_entry(self.field).field_type(),
                    boost,
                )
                .map(UnpositionedScorer::from_positioned);
        };
        let Some(ip_addr_column): Option<Column<Ipv6Addr>> =
            reader.fast_fields().column_opt(field_name)?
        else {
            return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
        };
        self.record_strategy_tag(StrategyTag::Linear);
        let term_set: FxHashSet<Ipv6Addr> = values.iter().copied().collect();
        if candidate_hint.is_some() {
            let docset = TermSetDocSet::new_unpositioned(ip_addr_column, term_set);
            Ok(UnpositionedScorer::from_unpositioned(Box::new(
                ConstScorer::new(docset, boost),
            )))
        } else {
            let docset = TermSetDocSet::new(ip_addr_column, term_set);
            Ok(UnpositionedScorer::from_positioned(Box::new(
                ConstScorer::new(docset, boost),
            )))
        }
    }

    fn get_or_init_dict_entry(
        &self,
        segment_id: SegmentId,
        bytes_col: &BytesColumn,
    ) -> crate::Result<Arc<Mutex<SegmentDictEntry>>> {
        if let Some(entry) = self.dict_cache.0.read().unwrap().get(&segment_id) {
            return Ok(Arc::clone(entry));
        }
        let dict = bytes_col.dictionary();
        let ords_col = bytes_col.ords();
        let num_terms = bytes_col.num_terms();
        let (lo_bytes, hi_bytes) = self.filter.byte_bounds();
        let (lo_ord_bound, hi_ord_bound) = dict.term_bounds_to_ord(lo_bytes, hi_bytes)?;
        let lo_ord = match lo_ord_bound {
            Bound::Included(o) => o,
            Bound::Excluded(o) => o.saturating_add(1),
            Bound::Unbounded => 0,
        }
        .max(ords_col.min_value());
        let hi_ord_excl = match hi_ord_bound {
            Bound::Included(o) => o.saturating_add(1),
            Bound::Excluded(o) => o,
            Bound::Unbounded => num_terms as u64,
        }
        .min(ords_col.max_value().saturating_add(1))
        .min(num_terms as u64);

        let entry = if lo_ord >= hi_ord_excl {
            SegmentDictEntry {
                ord_range: 0..0,
                block_ids: None,
                state: SegmentDictState::FullyPopulated(Arc::new(BitSet::with_max_value(0))),
            }
        } else if let Some(block_ids) = dict.block_ids_for_ord_range(lo_ord..hi_ord_excl)? {
            SegmentDictEntry {
                ord_range: lo_ord..hi_ord_excl,
                block_ids: Some(block_ids),
                state: SegmentDictState::Unpopulated,
            }
        } else {
            SegmentDictEntry {
                ord_range: 0..0,
                block_ids: None,
                state: SegmentDictState::FullyPopulated(Arc::new(BitSet::with_max_value(0))),
            }
        };
        let mut cache = self.dict_cache.0.write().unwrap();
        Ok(Arc::clone(
            cache
                .entry(segment_id)
                .or_insert_with(|| Arc::new(Mutex::new(entry))),
        ))
    }

    fn dispatch_dictionary_fast(
        &self,
        reader: &SegmentReader,
        field_name: &str,
        field_type: &FieldType,
        boost: Score,
        candidate_hint: Option<CandidateHint>,
    ) -> crate::Result<UnpositionedScorer> {
        let bytes_col_opt: Option<BytesColumn> =
            if let Some(str_col) = reader.fast_fields().str(field_name)? {
                Some(BytesColumn::from(str_col))
            } else {
                reader.fast_fields().bytes(field_name)?
            };
        let Some(bytes_col) = bytes_col_opt else {
            self.record_strategy_tag(StrategyTag::Empty);
            return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
        };

        let n = bytes_col.num_rows();
        let num_terms = bytes_col.num_terms();
        if n == 0 || num_terms == 0 {
            self.record_strategy_tag(StrategyTag::Empty);
            return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
        }

        let ords_col = bytes_col.ords();
        let d = ((n as u64) / (num_terms as u64).max(1))
            .max(1)
            .min(u32::MAX as u64) as u32;
        let candidate_size = candidate_hint.map(CandidateHint::upper_bound);
        let planner_inputs = PlannerInputs {
            field_name,
            candidate_size,
            avg_docs_per_term: Some(d),
        };
        let bitset_key_cap = if self.exact_postings == ExactPostings::Available {
            max_admissible_bitset_keys(reader, ords_col, &planner_inputs, &self.strategy_config)
        } else {
            0
        };

        // Choose between the fast-field `BytesColumn` and `TermDictionary` postings.
        let prefer_fast_column = if self.exact_postings == ExactPostings::Unavailable {
            true
        } else if try_gallop_order(reader, ords_col, &planner_inputs, &self.strategy_config)
            .is_some()
        {
            true
        } else if candidate_hint.is_some() {
            self.filter
                .known_key_count()
                .is_none_or(|k| k > bitset_key_cap as u64)
        } else {
            self.filter.known_key_count().is_some_and(|k| {
                k > bitset_key_cap as u64 && k.saturating_mul(2) >= num_terms as u64
            })
        };

        if candidate_hint.is_none() && !prefer_fast_column {
            return self
                .dispatch_postings(reader, field_type, boost)
                .map(UnpositionedScorer::from_positioned);
        }

        let entry_arc = self.get_or_init_dict_entry(reader.segment_id(), &bytes_col)?;
        let mut entry = entry_arc.lock().unwrap();
        let num_sstable_blocks = entry.num_sstable_blocks();
        let ord_range = entry.ord_range.clone();
        let SegmentDictEntry {
            block_ids, state, ..
        } = &mut *entry;
        let mut retained_keys: Option<Vec<Vec<u8>>> = None;
        let ords_arc = match state {
            SegmentDictState::FullyPopulated(arc) => Arc::clone(arc),
            state => {
                let probe_opt = if num_sstable_blocks > 2 {
                    match (candidate_hint, &self.filter) {
                        (
                            Some(CandidateHint::Pruned { .. }),
                            KeyFilter::Predicate { predicate, .. },
                        ) => Some(DictKeyProbe::Predicate(Arc::clone(predicate))),
                        (Some(CandidateHint::Pruned { .. }), KeyFilter::Terms(explicit))
                            if (explicit.terms_bytes.len() as u64) > bitset_key_cap as u64 =>
                        {
                            Some(DictKeyProbe::SortedTerms(Arc::clone(&explicit.terms_bytes)))
                        }
                        (
                            Some(CandidateHint::Estimated(c)),
                            KeyFilter::Predicate { predicate, .. },
                        ) if u64::from(c) < num_sstable_blocks => {
                            Some(DictKeyProbe::Predicate(Arc::clone(predicate)))
                        }
                        (Some(CandidateHint::Estimated(c)), KeyFilter::Terms(explicit))
                            if u64::from(c) < num_sstable_blocks
                                && (u64::from(c) * 4) <= explicit.terms_bytes.len() as u64 =>
                        {
                            Some(DictKeyProbe::SortedTerms(Arc::clone(&explicit.terms_bytes)))
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some(probe) = probe_opt {
                    if matches!(*state, SegmentDictState::Unpopulated) {
                        *state = SegmentDictState::BlockCached {
                            tested_ords: BitSet::with_max_value(num_terms as u32),
                            matched_ords: BitSet::with_max_value(num_terms as u32),
                            decoded_blocks: 0,
                        };
                    }
                    drop(entry);
                    self.record_strategy_tag(StrategyTag::Linear);
                    let docset = CachedDictPredicateDocSet::new_unpositioned(
                        bytes_col, ord_range, probe, entry_arc,
                    );
                    return Ok(UnpositionedScorer::from_unpositioned(Box::new(
                        ConstScorer::new(docset, boost),
                    )));
                }

                if !prefer_fast_column {
                    drop(entry);
                    return self
                        .dispatch_postings(reader, field_type, boost)
                        .map(UnpositionedScorer::from_positioned);
                }

                let (matched_ords, keys) = match state {
                    SegmentDictState::BlockCached {
                        tested_ords,
                        matched_ords,
                        decoded_blocks,
                    } => {
                        let probe = match &self.filter {
                            KeyFilter::Terms(explicit) => {
                                DictKeyProbe::SortedTerms(Arc::clone(&explicit.terms_bytes))
                            }
                            KeyFilter::Predicate { predicate, .. } => {
                                DictKeyProbe::Predicate(Arc::clone(predicate))
                            }
                        };
                        let ords = if let Some(block_ids) = block_ids.as_ref() {
                            let mut scratch = DictEvalScratch::default();
                            resolve_remaining_into_cache(
                                &bytes_col,
                                &ord_range,
                                block_ids,
                                &probe,
                                tested_ords,
                                matched_ords,
                                *decoded_blocks,
                                &mut scratch,
                            )?
                        } else {
                            std::mem::replace(matched_ords, BitSet::with_max_value(0))
                        };
                        (ords, None)
                    }
                    SegmentDictState::Unpopulated => match &self.filter {
                        KeyFilter::Terms(explicit) => {
                            let ords = resolve_explicit_dict_ordinals(
                                &bytes_col,
                                ord_range,
                                &explicit.terms_bytes,
                            )?;
                            let keys = (self.exact_postings == ExactPostings::Available
                                && ords.len() <= bitset_key_cap)
                                .then(|| explicit.terms_bytes.to_vec());
                            (ords, keys)
                        }
                        KeyFilter::Predicate { predicate, .. } => stream_predicate_dict_ordinals(
                            &bytes_col,
                            ord_range,
                            predicate.as_ref(),
                            bitset_key_cap,
                        )?,
                    },
                    SegmentDictState::FullyPopulated(_) => unreachable!(),
                };
                retained_keys = keys;
                let arc = Arc::new(matched_ords);
                *state = SegmentDictState::FullyPopulated(Arc::clone(&arc));
                arc
            }
        };
        drop(entry);

        if ords_arc.len() == 0 {
            self.record_strategy_tag(StrategyTag::Empty);
            return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
        }

        let retained_keys = if let Some(keys) = retained_keys {
            Some(keys)
        } else if self.exact_postings == ExactPostings::Available
            && ords_arc.len() <= bitset_key_cap
        {
            match &self.filter {
                KeyFilter::Terms(explicit) => Some(explicit.terms_bytes.to_vec()),
                KeyFilter::Predicate { .. } => {
                    let matched_u64s: Vec<u64> =
                        matched_ords_iter(&ords_arc).map(u64::from).collect();
                    let mut keys = Vec::with_capacity(matched_u64s.len());
                    bytes_col
                        .dictionary()
                        .sorted_ords_to_term_cb(&matched_u64s, |k| keys.push(k.to_vec()))?;
                    Some(keys)
                }
            }
        } else {
            None
        };

        let k_prime = ords_arc.len() as u32;
        let sorted_ords = Arc::clone(&ords_arc);
        let postings_cb = retained_keys.map(|keys| {
            Box::new(move || {
                bitset_from_postings_scorer(reader, self.field, &keys, boost, self.bitmap_enabled)
            }) as Box<dyn FnOnce() -> crate::Result<Box<dyn Scorer>> + '_>
        });

        dispatch_u64_column(
            reader,
            ords_col.clone(),
            planner_inputs,
            k_prime,
            &self.strategy_config,
            boost,
            move || matched_ords_iter(&sorted_ords).map(u64::from).collect(),
            move || ords_arc,
            postings_cb,
        )
    }

    fn dispatch_numeric_fast(
        &self,
        reader: &SegmentReader,
        field_name: &str,
        field_type: &FieldType,
        boost: Score,
        candidate_hint: Option<CandidateHint>,
    ) -> crate::Result<UnpositionedScorer> {
        let candidate_size = candidate_hint.map(CandidateHint::upper_bound);
        let Some((column, col_type)) = reader.fast_fields().u64_lenient_for_type(
            Some(&[
                ColumnType::U64,
                ColumnType::I64,
                ColumnType::F64,
                ColumnType::Bool,
                ColumnType::DateTime,
            ]),
            field_name,
        )?
        else {
            self.record_strategy_tag(StrategyTag::Empty);
            return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
        };

        let n = column.num_docs();
        if n == 0 {
            self.record_strategy_tag(StrategyTag::Empty);
            return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
        }

        match &self.filter {
            KeyFilter::Terms(explicit) => {
                let ExplicitTyped::Numeric(values) = &explicit.typed else {
                    return self
                        .dispatch_postings(reader, field_type, boost)
                        .map(UnpositionedScorer::from_positioned);
                };
                let range = sorted_range_in_column(values, &column);
                if range.is_empty() {
                    self.record_strategy_tag(StrategyTag::Empty);
                    return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
                }
                let pruned_values = &values[range.clone()];
                let pruned_bytes = &explicit.terms_bytes[range];
                let k_prime = pruned_values.len() as u32;

                let bitset_could_admit = if let Some(c) = candidate_size {
                    let c = c.clamp(1, n) as f64;
                    (k_prime as f64) / c <= self.strategy_config.subsequent_bitset_max_density
                } else {
                    let loosest = self
                        .strategy_config
                        .bitset_max_density_unique
                        .max(self.strategy_config.bitset_max_density_multi);
                    (k_prime as f64) / (n as f64) <= loosest
                };
                let d = if bitset_could_admit && self.exact_postings == ExactPostings::Available {
                    let dict_size = reader
                        .inverted_index(self.field)?
                        .terms()
                        .num_terms()
                        .max(1) as u64;
                    ((u64::from(n) / dict_size).max(1)).min(u32::MAX as u64) as u32
                } else {
                    1
                };

                let planner_inputs = PlannerInputs {
                    field_name,
                    candidate_size,
                    avg_docs_per_term: Some(d),
                };
                let postings_cb = (self.exact_postings == ExactPostings::Available).then(|| {
                    Box::new(move || {
                        bitset_from_postings_scorer(
                            reader,
                            self.field,
                            pruned_bytes,
                            boost,
                            self.bitmap_enabled,
                        )
                    })
                        as Box<dyn FnOnce() -> crate::Result<Box<dyn Scorer>> + '_>
                });

                dispatch_u64_column(
                    reader,
                    column,
                    planner_inputs,
                    k_prime,
                    &self.strategy_config,
                    boost,
                    || pruned_values.to_vec(),
                    || pruned_values.iter().copied().collect::<FxHashSet<u64>>(),
                    postings_cb,
                )
            }
            KeyFilter::Predicate { predicate, bounds } => {
                let Some(value_range) = intersect_u64_bounds(bounds, &column) else {
                    self.record_strategy_tag(StrategyTag::Empty);
                    return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
                };

                let doc_bounds = reader
                    .sort_by_field()
                    .filter(|sbf| sbf.field == field_name)
                    .and_then(|sbf| sorted_doc_range_for_bounds(&column, &value_range, sbf.order))
                    .unwrap_or(0..n);
                if doc_bounds.is_empty() {
                    self.record_strategy_tag(StrategyTag::Empty);
                    return Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer)));
                }

                self.record_strategy_tag(StrategyTag::Linear);
                if candidate_size.is_some() {
                    let docset = PredicateColumnDocSet::new_unpositioned(
                        column,
                        col_type,
                        value_range,
                        doc_bounds,
                        Arc::clone(predicate),
                    );
                    Ok(UnpositionedScorer::from_unpositioned(Box::new(
                        ConstScorer::new(docset, boost),
                    )))
                } else {
                    let docset = PredicateColumnDocSet::new(
                        column,
                        col_type,
                        value_range,
                        doc_bounds,
                        Arc::clone(predicate),
                    );
                    Ok(UnpositionedScorer::from_positioned(Box::new(
                        ConstScorer::new(docset, boost),
                    )))
                }
            }
        }
    }

    fn dispatch_postings(
        &self,
        reader: &SegmentReader,
        field_type: &FieldType,
        boost: Score,
    ) -> crate::Result<Box<dyn Scorer>> {
        let max_doc = reader.max_doc();
        if max_doc == 0 || self.exact_postings == ExactPostings::Unavailable {
            self.record_strategy_tag(StrategyTag::Empty);
            return Ok(Box::new(EmptyScorer));
        }

        match &self.filter {
            KeyFilter::Terms(explicit) => {
                let k = explicit.terms_bytes.len() as u64;
                if k == 0 {
                    self.record_strategy_tag(StrategyTag::Empty);
                    return Ok(Box::new(EmptyScorer));
                }
                if (k as f64) / (max_doc as f64) <= self.strategy_config.bitset_max_density_multi {
                    self.record_strategy_tag(StrategyTag::Bitset);
                    bitset_from_postings_scorer(
                        reader,
                        self.field,
                        &explicit.terms_bytes,
                        boost,
                        self.bitmap_enabled,
                    )
                } else {
                    self.record_strategy_tag(StrategyTag::Automaton);
                    self.automaton_scorer(reader, explicit, boost)
                }
            }
            KeyFilter::Predicate { predicate, bounds } => {
                let inverted_index = reader.inverted_index(self.field)?;
                let mut builder = inverted_index.terms().range().interruptible();
                match &bounds.lower_bound {
                    Bound::Included(t) => builder = builder.ge(t.serialized_value_bytes()),
                    Bound::Excluded(t) => builder = builder.gt(t.serialized_value_bytes()),
                    Bound::Unbounded => {}
                }
                match &bounds.upper_bound {
                    Bound::Included(t) => builder = builder.le(t.serialized_value_bytes()),
                    Bound::Excluded(t) => builder = builder.lt(t.serialized_value_bytes()),
                    Bound::Unbounded => {}
                }
                let mut stream: TermStreamer<'_> = builder.into_stream()?;
                let use_bmap = self.bitmap_enabled && reader.bitmap_postings_enabled;
                let mut doc_bitset = BitSet::with_max_value(max_doc);
                let mut matched_terms: u64 = 0;

                if let Some(col_type) = numeric_column_type_for_field(field_type) {
                    let mut batcher = U64Batcher::new();
                    let mut on_match = |info: &TermInfo, _val: u64| -> crate::Result<()> {
                        matched_terms += 1;
                        or_term_info_into_bitset(&inverted_index, info, &mut doc_bitset, use_bmap)
                    };
                    while stream.advance() {
                        if let Ok(bytes) = <[u8; 8]>::try_from(stream.key()) {
                            batcher.push(
                                stream.value().clone(),
                                u64::from_be_bytes(bytes),
                                col_type,
                                predicate.as_ref(),
                                &mut on_match,
                            )?;
                        }
                    }
                    batcher.finish(col_type, predicate.as_ref(), &mut on_match)?;
                } else {
                    let mut batcher = BytesBatcher::new();
                    let mut on_match = |info: &TermInfo, _key: &[u8]| -> crate::Result<()> {
                        matched_terms += 1;
                        or_term_info_into_bitset(&inverted_index, info, &mut doc_bitset, use_bmap)
                    };
                    while stream.advance() {
                        batcher.push(
                            stream.value().clone(),
                            stream.key(),
                            predicate.as_ref(),
                            &mut on_match,
                        )?;
                    }
                    batcher.finish(predicate.as_ref(), &mut on_match)?;
                }

                if matched_terms == 0 {
                    self.record_strategy_tag(StrategyTag::Empty);
                    return Ok(Box::new(EmptyScorer));
                }
                let tag = if (matched_terms as f64) / (max_doc as f64)
                    <= self.strategy_config.bitset_max_density_multi
                {
                    StrategyTag::Bitset
                } else {
                    StrategyTag::Automaton
                };
                self.record_strategy_tag(tag);
                Ok(Box::new(ConstScorer::new(
                    BitSetDocSet::from(doc_bitset),
                    boost,
                )))
            }
        }
    }
}

impl Weight for TermSetWeight {
    fn uses_fast_fields(&self) -> bool {
        self.fast_substrate != FastSubstrate::None && !self.filter.is_definitely_empty()
    }

    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        self.scorer_with_candidates(reader, boost, None)
            .map(UnpositionedScorer::into_positioned)
    }

    fn scorer_for_candidates_unpositioned(
        &self,
        reader: &SegmentReader,
        boost: Score,
        candidate_hint: CandidateHint,
    ) -> crate::Result<UnpositionedScorer> {
        self.scorer_with_candidates(reader, boost, Some(candidate_hint))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.doc() > doc || scorer.seek(doc) != doc {
            return Err(TantivyError::InvalidArgument(format!(
                "Document #({doc}) does not match"
            )));
        }
        Ok(Explanation::new("TermSetScorer", scorer.score()))
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_u64_column<M: ValueMembership<u64>>(
    reader: &SegmentReader,
    column: Column<u64>,
    planner_inputs: PlannerInputs<'_>,
    k_prime: u32,
    cfg: &TermSetStrategyConfig,
    boost: Score,
    make_sorted_u64s: impl FnOnce() -> Vec<u64>,
    make_membership: impl FnOnce() -> M,
    try_make_postings_bitset: Option<Box<dyn FnOnce() -> crate::Result<Box<dyn Scorer>> + '_>>,
) -> crate::Result<UnpositionedScorer> {
    let candidate_size = planner_inputs.candidate_size;
    match select_strategy(reader, &column, planner_inputs, k_prime, cfg) {
        StrategyChoice::Empty => Ok(UnpositionedScorer::from_positioned(Box::new(EmptyScorer))),
        StrategyChoice::Gallop(sort_order) => {
            let cardinality = column.get_cardinality();
            let docset =
                TermSetGallopDocSet::new(column, sort_order, make_sorted_u64s(), cardinality);
            Ok(UnpositionedScorer::from_positioned(Box::new(
                ConstScorer::new(docset, boost),
            )))
        }
        StrategyChoice::BitsetFromPostings if try_make_postings_bitset.is_some() => {
            (try_make_postings_bitset.expect("checked Some"))()
                .map(UnpositionedScorer::from_positioned)
        }
        StrategyChoice::LinearScan | StrategyChoice::BitsetFromPostings => {
            let membership = make_membership();
            if candidate_size.is_some() {
                let docset = TermSetDocSet::new_unpositioned(column, membership);
                Ok(UnpositionedScorer::from_unpositioned(Box::new(
                    ConstScorer::new(docset, boost),
                )))
            } else {
                let docset = TermSetDocSet::new(column, membership);
                Ok(UnpositionedScorer::from_positioned(Box::new(
                    ConstScorer::new(docset, boost),
                )))
            }
        }
    }
}

fn term_bound_bytes(bound: &Bound<Term>) -> Bound<&[u8]> {
    match bound {
        Bound::Included(t) => Bound::Included(t.serialized_value_bytes()),
        Bound::Excluded(t) => Bound::Excluded(t.serialized_value_bytes()),
        Bound::Unbounded => Bound::Unbounded,
    }
}

fn numeric_column_type_for_field(field_type: &FieldType) -> Option<ColumnType> {
    match field_type {
        FieldType::U64(_) => Some(ColumnType::U64),
        FieldType::I64(_) => Some(ColumnType::I64),
        FieldType::F64(_) => Some(ColumnType::F64),
        FieldType::Bool(_) => Some(ColumnType::Bool),
        FieldType::Date(_) => Some(ColumnType::DateTime),
        _ => None,
    }
}

fn intersect_u64_bounds(
    bounds: &BoundsRange<Term>,
    column: &Column<u64>,
) -> Option<RangeInclusive<u64>> {
    let mut lo = column.min_value();
    let mut hi = column.max_value();
    if lo > hi {
        return None;
    }
    match &bounds.lower_bound {
        Bound::Included(t) => {
            if let Some(v) = t.value().as_u64_lenient() {
                lo = lo.max(v);
            }
        }
        Bound::Excluded(t) => {
            if let Some(v) = t.value().as_u64_lenient() {
                lo = lo.max(v.checked_add(1)?);
            }
        }
        Bound::Unbounded => {}
    }
    match &bounds.upper_bound {
        Bound::Included(t) => {
            if let Some(v) = t.value().as_u64_lenient() {
                hi = hi.min(v);
            }
        }
        Bound::Excluded(t) => {
            if let Some(v) = t.value().as_u64_lenient() {
                hi = hi.min(v.checked_sub(1)?);
            }
        }
        Bound::Unbounded => {}
    }
    (lo <= hi).then_some(lo..=hi)
}

fn matched_ords_iter(bitset: &BitSet) -> impl Iterator<Item = u32> + '_ {
    let num_buckets = bitset.max_value().div_ceil(64);
    (0..num_buckets).flat_map(move |bucket| {
        let base = bucket * 64;
        bitset
            .tinyset(bucket)
            .into_iter()
            .map(move |bit| base + bit)
    })
}

// ---------------------------------------------------------------------------
// SetDfaWrapper — FST-as-set automaton driving the `Automaton` dispatch
// ---------------------------------------------------------------------------

pub(crate) struct SetDfaWrapper(pub(crate) Arc<Map<Vec<u8>>>);

impl Automaton for SetDfaWrapper {
    type State = Option<CompiledAddr>;

    fn start(&self) -> Option<CompiledAddr> {
        Some((*self.0).as_ref().root().addr())
    }

    fn is_match(&self, state_opt: &Option<CompiledAddr>) -> bool {
        state_opt.is_some_and(|state| (*self.0).as_ref().node(state).is_final())
    }

    fn accept(&self, state_opt: &Option<CompiledAddr>, byte: u8) -> Option<CompiledAddr> {
        let node = (*self.0).as_ref().node((*state_opt)?);
        Some(node.transition_addr(node.find_input(byte)?))
    }

    fn can_match(&self, state: &Self::State) -> bool {
        state.is_some()
    }
}

// ---------------------------------------------------------------------------
// TermSetDocSet — fast-field linear scan over a `Column<T>` filtered by an
// in-memory membership structure (`FxHashSet<T>` or ordinal `Arc<BitSet>`)
// ---------------------------------------------------------------------------

pub(crate) trait ValueMembership<T>: Clone + Send + Sync + 'static {
    fn is_empty(&self) -> bool;
    fn contains(&self, value: T) -> bool;
}

impl<T: Copy + Eq + std::hash::Hash + Send + Sync + 'static> ValueMembership<T> for FxHashSet<T> {
    #[inline]
    fn is_empty(&self) -> bool {
        FxHashSet::is_empty(self)
    }

    #[inline]
    fn contains(&self, value: T) -> bool {
        FxHashSet::contains(self, &value)
    }
}

impl ValueMembership<u64> for Arc<BitSet> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn contains(&self, value: u64) -> bool {
        u32::try_from(value)
            .ok()
            .is_some_and(|ord| ord < self.max_value() && BitSet::contains(self, ord))
    }
}

#[derive(Clone)]
pub(crate) struct TermSetDocSet<T: Copy, S: ValueMembership<T> = FxHashSet<T>> {
    column: Column<T>,
    values: S,
    doc_id: DocId,
    max_doc: DocId,
    positioned: bool,
}

impl<T: Copy + PartialOrd + std::fmt::Debug + Send + Sync + 'static, S: ValueMembership<T>>
    TermSetDocSet<T, S>
{
    pub fn new(column: Column<T>, values: S) -> Self {
        let mut doc_set = Self::new_unpositioned(column, values);
        if doc_set.doc_id != TERMINATED {
            doc_set.seek(0);
        }
        doc_set
    }

    pub fn new_unpositioned(column: Column<T>, values: S) -> Self {
        let max_doc = column.num_docs();
        let doc_id = if max_doc == 0 || values.is_empty() {
            TERMINATED
        } else {
            0
        };
        Self {
            column,
            values,
            doc_id,
            max_doc,
            positioned: doc_id == TERMINATED,
        }
    }
}

impl<T: Copy + PartialOrd + std::fmt::Debug + Send + Sync + 'static, S: ValueMembership<T>> DocSet
    for TermSetDocSet<T, S>
{
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
        if target >= self.max_doc {
            self.doc_id = TERMINATED;
            return TERMINATED;
        }
        for next_doc_id in target..self.max_doc {
            if self
                .column
                .values_for_doc(next_doc_id)
                .any(|v| self.values.contains(v))
            {
                self.doc_id = next_doc_id;
                return next_doc_id;
            }
        }
        self.doc_id = TERMINATED;
        TERMINATED
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        if target >= self.max_doc {
            self.doc_id = TERMINATED;
            self.positioned = true;
            return SeekDangerResult::SeekLowerBound(TERMINATED);
        }
        self.positioned = self
            .column
            .values_for_doc(target)
            .any(|value| self.values.contains(value));
        if self.positioned {
            self.doc_id = target;
            SeekDangerResult::Found
        } else {
            SeekDangerResult::SeekLowerBound(target + 1)
        }
    }

    fn doc(&self) -> DocId {
        self.doc_id
    }

    fn size_hint(&self) -> u32 {
        self.max_doc.saturating_sub(self.doc_id.saturating_add(1))
    }

    fn cost(&self) -> crate::DocSetCost {
        crate::DocSetCost::new(
            crate::DriveCost(u64::from(self.size_hint())),
            crate::ProbeCost::FAST_FIELD_LOOKUP,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::net::IpAddr;
    use std::str::FromStr;
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::Arc;

    use super::{StrategyTag, TermSetDocSet, TermSetStrategyConfig};
    use crate::collector::{Count, TopDocs};
    use crate::query::{
        BooleanQuery, CandidateHint, ConstScoreQuery, EnableScoring, FastFieldTermSetQuery, Query,
        QueryParser, TermSetQuery,
    };
    use crate::schema::{IntoIpv6Addr, Schema, FAST, INDEXED, STRING, TEXT};
    use crate::{DocSet, Index, IndexWriter, Term, TERMINATED};

    fn create_test_index() -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let text_field_fast = schema_builder.add_text_field("text_fast", STRING | FAST);
        let u64_field_fast = schema_builder.add_u64_field("u64_fast", FAST | INDEXED);
        let ip_field_fast = schema_builder.add_ip_addr_field("ip_fast", FAST | INDEXED);
        let bool_field_fast = schema_builder.add_bool_field("bool_fast", FAST | INDEXED);
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(
                text_field => "doc1",
                text_field_fast => "doc1",
                u64_field_fast => 1u64,
                ip_field_fast => IpAddr::from_str("127.0.0.1").unwrap().into_ipv6_addr(),
                bool_field_fast => true,
            ))?;
            index_writer.add_document(doc!(
                text_field => "doc2",
                text_field_fast => "doc2",
                u64_field_fast => 2u64,
                ip_field_fast => IpAddr::from_str("127.0.0.2").unwrap().into_ipv6_addr(),
                bool_field_fast => false,
            ))?;
            index_writer.add_document(doc!(
                text_field => "doc3",
                text_field_fast => "doc3",
                u64_field_fast => 3u64,
                ip_field_fast => IpAddr::from_str("::ffff:127.0.0.3").unwrap().into_ipv6_addr(),
                bool_field_fast => true,
            ))?;
            index_writer.commit()?;
        }
        Ok(index)
    }

    #[test]
    pub fn test_term_set_query_fast_fields_and_parser() -> crate::Result<()> {
        let index = create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let schema = index.schema();
        let u64_field = schema.get_field("u64_fast").unwrap();
        let text_field = schema.get_field("text_fast").unwrap();
        let ip_field = schema.get_field("ip_fast").unwrap();

        assert_eq!(
            searcher.search(
                &FastFieldTermSetQuery::new([
                    Term::from_field_u64(u64_field, 1),
                    Term::from_field_u64(u64_field, 3),
                ]),
                &Count
            )?,
            2
        );
        assert_eq!(
            searcher.search(
                &FastFieldTermSetQuery::new([
                    Term::from_field_u64(u64_field, 4),
                    Term::from_field_u64(u64_field, 5),
                ]),
                &Count
            )?,
            0
        );
        assert_eq!(
            searcher.search(&FastFieldTermSetQuery::new(Vec::<Term>::new()), &Count)?,
            0
        );
        assert_eq!(
            searcher.search(
                &FastFieldTermSetQuery::new([
                    Term::from_field_text(text_field, "doc1"),
                    Term::from_field_text(text_field, "doc3"),
                ]),
                &Count
            )?,
            2
        );
        assert_eq!(
            searcher.search(
                &FastFieldTermSetQuery::new([
                    Term::from_field_ip_addr(
                        ip_field,
                        IpAddr::from_str("127.0.0.1").unwrap().into_ipv6_addr(),
                    ),
                    Term::from_field_ip_addr(
                        ip_field,
                        IpAddr::from_str("127.0.0.3").unwrap().into_ipv6_addr(),
                    ),
                ]),
                &Count
            )?,
            2
        );

        // Empty IP `TermSetDocSet` inside `BooleanQuery::intersection` does not underflow
        // `size_hint`.
        let intersection = BooleanQuery::intersection(vec![
            Box::new(FastFieldTermSetQuery::new([Term::from_field_ip_addr(
                ip_field,
                IpAddr::from_str("10.0.0.1").unwrap().into_ipv6_addr(),
            )])),
            Box::new(FastFieldTermSetQuery::new([Term::from_field_ip_addr(
                ip_field,
                IpAddr::from_str("127.0.0.1").unwrap().into_ipv6_addr(),
            )])),
        ]);
        assert_eq!(searcher.search(&intersection, &Count)?, 0);

        let qp = QueryParser::for_index(&index, vec![]);
        for (q, expected) in [
            ("u64_fast: IN [1 3]", 2),
            ("text_fast: IN [doc1 doc3]", 2),
            ("ip_fast: IN [127.0.0.1 127.0.0.3]", 2),
            ("bool_fast: IN [true]", 2),
        ] {
            assert_eq!(searcher.search(&qp.parse_query(q)?, &Count)?, expected);
        }
        Ok(())
    }

    #[test]
    pub fn test_term_set_query_fast_field_missing_value_and_seek_danger() -> crate::Result<()> {
        let mut sb = Schema::builder();
        let text_field = sb.add_text_field("text", TEXT);
        let u64_field = sb.add_u64_field("u64_fast", FAST | INDEXED);
        let index = Index::create_in_ram(sb.build());
        {
            let mut writer = index.writer_for_tests()?;
            writer.add_document(doc!(u64_field => 1u64))?;
            writer.add_document(doc!(text_field => "doc2"))?;
            writer.add_document(doc!(u64_field => 3u64))?;
            writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert!(searcher
            .search(
                &FastFieldTermSetQuery::new([Term::from_field_u64(u64_field, 0)]),
                &TopDocs::with_limit(10).order_by_score()
            )?
            .is_empty());
        let top_docs = searcher.search(
            &FastFieldTermSetQuery::new([Term::from_field_u64(u64_field, 1)]),
            &TopDocs::with_limit(10).order_by_score(),
        )?;
        assert_eq!(top_docs.len(), 1);
        assert_eq!(top_docs[0].1.doc_id, 0);

        let full_idx = create_test_index()?;
        let full_reader = full_idx.reader()?;
        let seg_reader = full_reader.searcher().segment_reader(0).clone();
        let column: columnar::Column<u64> = seg_reader.fast_fields().u64("u64_fast")?;
        let mut values = rustc_hash::FxHashSet::default();
        values.insert(2u64);
        let mut docset = TermSetDocSet::new(column, values);
        assert_eq!(
            docset.seek_danger(0),
            crate::docset::SeekDangerResult::SeekLowerBound(1)
        );
        assert_eq!(
            docset.seek_danger(1),
            crate::docset::SeekDangerResult::Found
        );
        assert_eq!(docset.doc(), 1);
        assert_eq!(
            docset.seek_danger(2),
            crate::docset::SeekDangerResult::SeekLowerBound(3)
        );
        assert_eq!(
            docset.seek_danger(10),
            crate::docset::SeekDangerResult::SeekLowerBound(TERMINATED)
        );
        assert_eq!(docset.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    pub fn test_scorer_for_candidates_and_range_pruning() -> crate::Result<()> {
        let index = create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0);
        let u64_field_fast = index.schema().get_field("u64_fast").unwrap();

        let sink = Arc::new(AtomicU8::new(0));
        let query = ConstScoreQuery::new(
            TermSetQuery::new([
                Term::from_field_u64(u64_field_fast, 100),
                Term::from_field_u64(u64_field_fast, 0),
                Term::from_field_u64(u64_field_fast, 2),
            ])
            .with_strategy_config(TermSetStrategyConfig {
                strategy_sink: Some(sink.clone()),
                ..Default::default()
            }),
            0.0,
        );
        let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let mut scorer = weight.scorer_for_candidates_unpositioned(
            segment_reader,
            1.0,
            CandidateHint::Estimated(1),
        )?;
        assert_eq!(sink.load(Ordering::Relaxed), StrategyTag::Linear as u8);
        assert!(!scorer.is_positioned());
        assert_eq!(scorer.lower_bound(), 0);
        assert_eq!(
            scorer.seek_danger(0),
            crate::docset::SeekDangerResult::SeekLowerBound(1)
        );
        assert_eq!(
            scorer.seek_danger(1),
            crate::docset::SeekDangerResult::Found
        );
        assert_eq!(
            scorer.seek_danger(2),
            crate::docset::SeekDangerResult::SeekLowerBound(3)
        );

        let empty_sink = Arc::new(AtomicU8::new(0));
        let empty_weight = TermSetQuery::new([
            Term::from_field_u64(u64_field_fast, 0),
            Term::from_field_u64(u64_field_fast, 10),
        ])
        .with_strategy_config(TermSetStrategyConfig {
            strategy_sink: Some(empty_sink.clone()),
            ..Default::default()
        })
        .weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let empty_scorer = empty_weight.scorer_for_candidates_unpositioned(
            segment_reader,
            1.0,
            CandidateHint::Estimated(1),
        )?;
        assert_eq!(empty_sink.load(Ordering::Relaxed), StrategyTag::Empty as u8);
        assert!(empty_scorer.is_empty());
        Ok(())
    }
}
