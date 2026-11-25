//! Strategy selection for `TermSetWeight`.
//!
//! Mirrors the structure of `range_query::range_query_fastfield`:
//! `Weight::scorer()` consults [`select_strategy`] for `Column<u64>`-backed
//! fast fields (numeric fast fields and dictionary fast fields via
//! `BytesColumn::ords()`) to decide between `Gallop`, `BitsetFromPostings`,
//! and `LinearScan`, then constructs the right `DocSet`. Fields without a
//! fast-field column (or dictionary fast fields where postings beat a full
//! dictionary scan) take `TermSetWeight::dispatch_postings`, which applies a
//! direct `BitsetFromPostings`-vs-`Automaton` density gate.
//!
//! ## `inputs.avg_docs_per_term` (`D`)
//!
//! Computing `D` precisely from a numeric column alone would require
//! posting-list lookups (which is some of the work
//! `BitsetFromPostings` does anyway). `select_strategy` accepts
//! `None` and treats it as `D = 1`. The first-column dispatch uses
//! `D` to pick between two `BitsetFromPostings`-vs-`LinearScan`
//! thresholds; a missing estimate degrades to the tighter `D = 1` threshold.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use columnar::{Cardinality, Column};

use crate::index::SegmentReader;
use crate::Order;

/// Numeric tag for the chosen strategy. Used by the optional
/// `TermSetStrategyConfig::strategy_sink` so consumers (paradedb) can surface
/// the per-segment dispatch decision in `EXPLAIN ANALYZE` without depending
/// on `StrategyChoice` directly.
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StrategyTag {
    None = 0,
    Gallop = 1,
    Linear = 2,
    Bitset = 3,
    Empty = 4,
    /// Streaming dictionary walk over inverted-index postings (via an FST
    /// set-membership automaton for explicit terms, or batched predicate
    /// evaluation over `TermDictionary`) when density exceeds
    /// `bitset_max_density_multi`.
    Automaton = 5,
}

impl TryFrom<u8> for StrategyTag {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(StrategyTag::None),
            1 => Ok(StrategyTag::Gallop),
            2 => Ok(StrategyTag::Linear),
            3 => Ok(StrategyTag::Bitset),
            4 => Ok(StrategyTag::Empty),
            5 => Ok(StrategyTag::Automaton),
            _ => Err("unknown StrategyTag value"),
        }
    }
}

/// User-tunable density thresholds for `BitsetFromPostings`. Defaults are bench-calibrated
/// against the SSTable backend and overridable via the
/// `paradedb.term_set_*_max_density` GUCs at the consumer side.
///
/// "Density" is a unitless ratio of *matching count over corpus
/// count*. For bitset on the first column it reduces to `K' / N` (D
/// is used only to pick which threshold applies, not multiplied into
/// the gate variable); for the subsequent-column branch, `K' · D / C`.
/// Bitset fires when its density is at or below the corresponding
/// `_max_density` threshold. Gallop has no density gate — see
/// `gallop_enabled`.
///
/// We use `f64` rather than integer denominators so bench-derived
/// thresholds don't have to round to `1/N` for small `N`.
#[derive(Clone, Debug)]
pub struct TermSetStrategyConfig {
    /// Kill-switch: when `false`, the planner never returns `Gallop` even if
    /// every other gate passes.
    pub gallop_enabled: bool,
    /// `BitsetFromPostings` first-column threshold for *unique-valued
    /// columns* (`D = 1`, i.e. `dict_size = N`, e.g., primary keys).
    /// The strategy fires when `K' / N <= bitset_max_density_unique`.
    /// Default `1/2000 = 0.0005` — calibrated against the SSTable
    /// backend where per-`dict.get` zstd block decompress dominates
    /// batched cost on unique columns; the empirical crossover with
    /// `LinearScan` lands at `K'/N ≈ 1/1000` and this default sits at
    /// half that to give linear the boundary cells.
    pub bitset_max_density_unique: f64,
    /// `BitsetFromPostings` first-column threshold for *non-unique
    /// columns* (`D >= 2`). The strategy fires when
    /// `K' / N <= bitset_max_density_multi`. Default `1/200 = 0.005` —
    /// the `cost_model` bench tier pinned the batched-vs-linear
    /// crossover to K/N ∈ [0.0055, 0.0088] across D ∈ {2, 5, 20, 50}
    /// (spread 1.59×), and `1/200` sits just below the tightest
    /// measured crossover (D=2). Exactly 10× looser than the unique
    /// threshold, matching the ~10× per-K cost gap: batched on a
    /// non-unique column amortizes the zstd block decompress across
    /// multiple keys per block, dropping per-K dict cost relative to
    /// the D=1 case where every key maps to a distinct block.
    pub bitset_max_density_multi: f64,
    /// Subsequent-column `BitsetFromPostings` threshold: `K' · D / C` cutoff.
    /// Default `1/2000 = 0.0005` — in the subsequent-column branch (`C < N`),
    /// `BitsetFromPostings` still pays the full SSTable `dict.get` zstd block
    /// decompression and postings decode across `K'` keys regardless of how small
    /// `C` is, whereas `TermSetDocSet::new_unpositioned` only performs `C`
    /// in-memory fast-field point probes (`column.values_for_doc` + `FxHashSet`
    /// or ordinal `BitSet` lookup). Matching `bitset_max_density_unique`
    /// (`1/2000`) ensures bitset is only chosen when `K' · D` is at least
    /// ~2000× smaller than `C`.
    pub subsequent_bitset_max_density: f64,
    /// Optional sink for the per-segment strategy choice. When `Some`,
    /// `select_strategy` writes the chosen `StrategyTag` (as `u8`) on its way
    /// out — last-segment-wins is fine because `EXPLAIN` only asks "did any
    /// segment use it?". When `None`, no atomic store happens and the
    /// hot-path cost is one `Option::is_some` check.
    pub strategy_sink: Option<Arc<AtomicU8>>,
}

impl Default for TermSetStrategyConfig {
    fn default() -> Self {
        Self {
            gallop_enabled: true,
            // Defaults are bench- and profile-derived. See per-field doc-comments
            // for the calibration story.
            bitset_max_density_unique: 1.0 / 2000.0,
            bitset_max_density_multi: 1.0 / 200.0,
            subsequent_bitset_max_density: 1.0 / 2000.0,
            strategy_sink: None,
        }
    }
}

/// Selected execution strategy on a `Column<u64>` (either a numeric fast-field
/// column or a dictionary fast-field's `ords` column), without allocating
/// `Gallop`'s `sorted_terms` vector up front unless `Gallop` is chosen.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum StrategyChoice {
    Gallop(Order),
    LinearScan,
    BitsetFromPostings,
    Empty,
}

impl StrategyChoice {
    pub(crate) fn tag(self) -> StrategyTag {
        match self {
            StrategyChoice::Gallop(_) => StrategyTag::Gallop,
            StrategyChoice::LinearScan => StrategyTag::Linear,
            StrategyChoice::BitsetFromPostings => StrategyTag::Bitset,
            StrategyChoice::Empty => StrategyTag::Empty,
        }
    }
}

/// Inputs to the planner that aren't already on `Column`.
pub(crate) struct PlannerInputs<'a> {
    pub field_name: &'a str,
    /// Upstream candidate-set size. `None` ≡ full segment scan (first column).
    pub candidate_size: Option<u32>,
    /// Average documents per term in this column. `None` is treated as `D = 1`.
    pub avg_docs_per_term: Option<u32>,
}

/// Maximum `k_prime` for which [`select_strategy`] could return
/// `BitsetFromPostings` (assuming `Gallop` did not already claim the segment).
pub(crate) fn max_admissible_bitset_keys(
    reader: &SegmentReader,
    column: &Column<u64>,
    inputs: &PlannerInputs<'_>,
    cfg: &TermSetStrategyConfig,
) -> usize {
    let n = column.num_docs();
    if n == 0 {
        return 0;
    }
    if try_gallop_order(reader, column, inputs, cfg).is_some() {
        return 0;
    }
    if let Some(c) = inputs.candidate_size {
        let c = c.clamp(1, n) as f64;
        let d = inputs.avg_docs_per_term.unwrap_or(1).max(1) as f64;
        let max_kd = (cfg.subsequent_bitset_max_density * c).floor();
        if max_kd <= 0.0 {
            0
        } else {
            (max_kd / d).floor().max(0.0) as usize
        }
    } else {
        let d = inputs.avg_docs_per_term.unwrap_or(1);
        let threshold = if d >= 2 {
            cfg.bitset_max_density_multi
        } else {
            cfg.bitset_max_density_unique
        };
        (threshold * (n as f64)).floor().max(0.0) as usize
    }
}

/// Strategy dispatch for `TermSetWeight` over a `Column<u64>` given the count
/// `k_prime` of matching keys/ordinals within the column's `[min, max]` range.
///
/// Decision tree:
/// 1. `n == 0 || k_prime == 0` -> `Empty`.
/// 2. Sorted by `field_name` + `gallop_enabled` + `candidate_size.is_none()` -> `Gallop(order)`.
/// 3. `candidate_size` is `Some(c)`: `K' * D / C <= subsequent_bitset_max_density` ->
///    `BitsetFromPostings`, else `LinearScan`.
/// 4. Full segment: `K' / N <= threshold(D)` -> `BitsetFromPostings`, else `LinearScan`.
pub(crate) fn select_strategy(
    reader: &SegmentReader,
    column: &Column<u64>,
    inputs: PlannerInputs<'_>,
    k_prime: u32,
    cfg: &TermSetStrategyConfig,
) -> StrategyChoice {
    let choice = select_strategy_inner(reader, column, inputs, k_prime, cfg);
    if let Some(sink) = cfg.strategy_sink.as_ref() {
        sink.store(choice.tag() as u8, Ordering::Relaxed);
    }
    choice
}

fn select_strategy_inner(
    reader: &SegmentReader,
    column: &Column<u64>,
    inputs: PlannerInputs<'_>,
    k_prime: u32,
    cfg: &TermSetStrategyConfig,
) -> StrategyChoice {
    let n = column.num_docs();
    if n == 0 || k_prime == 0 {
        return StrategyChoice::Empty;
    }

    if let Some(order) = try_gallop_order(reader, column, &inputs, cfg) {
        return StrategyChoice::Gallop(order);
    }

    if let Some(c) = inputs.candidate_size {
        let c = c.clamp(1, n);
        let d = inputs.avg_docs_per_term.unwrap_or(1) as u64;
        let kd = (k_prime as u64).saturating_mul(d) as f64;
        if kd / (c as f64) <= cfg.subsequent_bitset_max_density {
            return StrategyChoice::BitsetFromPostings;
        }
        return StrategyChoice::LinearScan;
    }

    let d = inputs.avg_docs_per_term.unwrap_or(1);
    let density = (k_prime as f64) / (n as f64);
    let bitset_threshold = if d >= 2 {
        cfg.bitset_max_density_multi
    } else {
        cfg.bitset_max_density_unique
    };
    if density <= bitset_threshold {
        StrategyChoice::BitsetFromPostings
    } else {
        StrategyChoice::LinearScan
    }
}

/// Return the index range `lo_idx..hi_idx` of `sorted_terms` (assumed sorted
/// ascending) whose elements fall within `[column.min_value(), column.max_value()]`.
pub(crate) fn sorted_range_in_column(
    sorted_terms: &[u64],
    column: &Column<u64>,
) -> std::ops::Range<usize> {
    let lo = column.min_value();
    let hi = column.max_value();
    if lo > hi || sorted_terms.is_empty() {
        return 0..0;
    }
    let lo_idx = sorted_terms.partition_point(|&v| v < lo);
    let hi_idx = lo_idx + sorted_terms[lo_idx..].partition_point(|&v| v <= hi);
    lo_idx..hi_idx
}

/// Gallop admissibility: returns `Some(order)` when the segment is
/// sorted by `field_name`, the column shape supports gallop, and the caller is
/// driving a full-segment scan (`candidate_size.is_none()`).
pub(crate) fn try_gallop_order(
    reader: &SegmentReader,
    column: &Column<u64>,
    inputs: &PlannerInputs<'_>,
    cfg: &TermSetStrategyConfig,
) -> Option<Order> {
    if !cfg.gallop_enabled || inputs.candidate_size.is_some() {
        return None;
    }
    if !matches!(
        column.get_cardinality(),
        Cardinality::Full | Cardinality::Optional
    ) {
        return None;
    }
    reader
        .sort_by_field()
        .filter(|sbf| sbf.field == inputs.field_name)
        .map(|sbf| sbf.order)
}

#[cfg(test)]
mod tests {
    //! Unit tests pin the planner's *output shape* on cases it can decide
    //! today. Tests that need to exercise gate semantics on specific K/N
    //! cells pass `avg_docs_per_term` and density-threshold overrides
    //! directly so they remain stable across future calibration changes.
    use super::*;
    use crate::schema::{NumericOptions, SchemaBuilder};
    use crate::{Index, IndexSettings, IndexSortByField};

    fn build_index(
        n: u64,
        sort: Option<IndexSortByField>,
        value_for_doc: impl Fn(u64) -> u64,
    ) -> (Index, crate::schema::Field, String) {
        let mut sb = SchemaBuilder::new();
        let field = sb.add_u64_field("fk", NumericOptions::default().set_fast().set_indexed());
        let schema = sb.build();
        let mut builder = Index::builder().schema(schema);
        if let Some(sbf) = sort {
            builder = builder.settings(IndexSettings {
                sort_by_field: Some(sbf),
                ..Default::default()
            });
        }
        let index = builder.create_in_ram().unwrap();
        let mut writer = index.writer_for_tests().unwrap();
        for d_idx in 0..n {
            writer
                .add_document(doc!(field => value_for_doc(d_idx)))
                .unwrap();
        }
        writer.commit().unwrap();
        (index, field, "fk".to_string())
    }

    fn open_column(index: &Index, field_name: &str) -> (SegmentReader, Column<u64>) {
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0).clone();
        let (column, _) = segment_reader
            .fast_fields()
            .u64_lenient_for_type(None, field_name)
            .unwrap()
            .unwrap();
        (segment_reader, column)
    }

    fn term_set(values: impl IntoIterator<Item = u64>) -> Vec<u64> {
        values.into_iter().collect()
    }

    fn plan_slice(
        reader: &SegmentReader,
        column: &Column<u64>,
        inputs: PlannerInputs<'_>,
        terms: &[u64],
        cfg: &TermSetStrategyConfig,
    ) -> StrategyChoice {
        let range = sorted_range_in_column(terms, column);
        select_strategy(reader, column, inputs, range.len() as u32, cfg)
    }

    #[test]
    fn select_handles_min_max_pruning_dropping_all_terms() {
        let (index, _field, name) = build_index(1024, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: None,
            },
            &term_set([10_000_000u64, 99_999_999, u64::MAX]),
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, StrategyChoice::Empty);
    }

    #[test]
    fn select_returns_bitset_from_postings_when_gallop_kill_switch_engaged() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(
            n,
            Some(IndexSortByField {
                field: "fk".to_string(),
                order: Order::Asc,
            }),
            |i| i,
        );
        let (reader, column) = open_column(&index, &name);
        let cfg = TermSetStrategyConfig {
            gallop_enabled: false,
            bitset_max_density_multi: 1.0 / 50.0,
            ..TermSetStrategyConfig::default()
        };
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: Some(2),
            },
            &term_set(0..60),
            &cfg,
        );
        assert_eq!(strat, StrategyChoice::BitsetFromPostings);
    }

    #[test]
    fn select_returns_bitset_when_first_column_highly_selective() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let cfg = TermSetStrategyConfig {
            bitset_max_density_unique: 1.0 / 50.0,
            ..TermSetStrategyConfig::default()
        };
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: Some(1),
            },
            &term_set(0..8),
            &cfg,
        );
        assert_eq!(strat, StrategyChoice::BitsetFromPostings);
    }

    #[test]
    fn select_falls_back_to_linear_scan_when_density_exceeds_bitset_threshold() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: Some(1),
            },
            &term_set(0..2000),
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, StrategyChoice::LinearScan);
    }

    #[test]
    fn select_returns_linear_scan_for_subsequent_column_when_kd_exceeds_threshold() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: Some(100),
                avg_docs_per_term: None,
            },
            &term_set(0..50),
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, StrategyChoice::LinearScan);
    }

    #[test]
    fn select_returns_bitset_from_postings_when_subsequent_column_kd_below_threshold() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: Some(2048),
                avg_docs_per_term: None,
            },
            &term_set(0..1),
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, StrategyChoice::BitsetFromPostings);
    }

    #[test]
    fn select_skips_gallop_when_unsorted() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: None,
            },
            &term_set(0..8),
            &TermSetStrategyConfig::default(),
        );
        assert!(!matches!(strat, StrategyChoice::Gallop(_)));
    }

    #[test]
    fn select_skips_gallop_when_field_mismatches_sort_field() {
        let n: u64 = 4096;
        let mut sb = SchemaBuilder::new();
        let fk = sb.add_u64_field("fk", NumericOptions::default().set_fast().set_indexed());
        let other = sb.add_u64_field("other", NumericOptions::default().set_fast().set_indexed());
        let schema = sb.build();
        let index = Index::builder()
            .schema(schema)
            .settings(IndexSettings {
                sort_by_field: Some(IndexSortByField {
                    field: "other".to_string(),
                    order: Order::Asc,
                }),
                ..Default::default()
            })
            .create_in_ram()
            .unwrap();
        {
            let mut writer = index.writer_for_tests().unwrap();
            for i in 0..n {
                writer.add_document(doc!(fk => i, other => n - i)).unwrap();
            }
            writer.commit().unwrap();
        }
        let (reader, column) = open_column(&index, "fk");
        let cfg = TermSetStrategyConfig {
            bitset_max_density_unique: 1.0 / 50.0,
            ..TermSetStrategyConfig::default()
        };
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: "fk",
                candidate_size: None,
                avg_docs_per_term: Some(1),
            },
            &term_set(0..8),
            &cfg,
        );
        assert_eq!(strat, StrategyChoice::BitsetFromPostings);
    }

    #[test]
    fn select_returns_gallop_when_sorted_and_small_k() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(
            n,
            Some(IndexSortByField {
                field: "fk".to_string(),
                order: Order::Asc,
            }),
            |i| i,
        );
        let (reader, column) = open_column(&index, &name);
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: None,
            },
            &term_set(0..8),
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, StrategyChoice::Gallop(Order::Asc));
    }

    #[test]
    fn select_writes_chosen_tag_to_strategy_sink() {
        let n: u64 = 4096;
        let sink = Arc::new(AtomicU8::new(StrategyTag::None as u8));
        let cfg = TermSetStrategyConfig {
            strategy_sink: Some(sink.clone()),
            ..TermSetStrategyConfig::default()
        };

        let (sorted_idx, _f, name_s) = build_index(
            n,
            Some(IndexSortByField {
                field: "fk".to_string(),
                order: Order::Asc,
            }),
            |i| i,
        );
        let (reader_s, col_s) = open_column(&sorted_idx, &name_s);
        let _ = plan_slice(
            &reader_s,
            &col_s,
            PlannerInputs {
                field_name: &name_s,
                candidate_size: None,
                avg_docs_per_term: None,
            },
            &term_set(0..8),
            &cfg,
        );
        assert_eq!(
            StrategyTag::try_from(sink.load(Ordering::Relaxed)).unwrap(),
            StrategyTag::Gallop
        );

        let (unsorted_idx, _f, name_u) = build_index(n, None, |i| i);
        let (reader_u, col_u) = open_column(&unsorted_idx, &name_u);
        let _ = plan_slice(
            &reader_u,
            &col_u,
            PlannerInputs {
                field_name: &name_u,
                candidate_size: None,
                avg_docs_per_term: None,
            },
            &term_set(0..2000),
            &cfg,
        );
        assert_eq!(
            StrategyTag::try_from(sink.load(Ordering::Relaxed)).unwrap(),
            StrategyTag::Linear
        );
    }

    #[test]
    fn select_respects_gallop_enabled_kill_switch() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(
            n,
            Some(IndexSortByField {
                field: "fk".to_string(),
                order: Order::Asc,
            }),
            |i| i,
        );
        let (reader, column) = open_column(&index, &name);
        let cfg = TermSetStrategyConfig {
            gallop_enabled: false,
            bitset_max_density_unique: 1.0 / 50.0,
            ..TermSetStrategyConfig::default()
        };
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: Some(1),
            },
            &term_set(0..8),
            &cfg,
        );
        assert_eq!(strat, StrategyChoice::BitsetFromPostings);
    }

    fn run_dispatch_at(d: u32, k: usize) -> StrategyChoice {
        let n: u64 = 20_000;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: Some(d),
            },
            &term_set(0..k as u64),
            &TermSetStrategyConfig::default(),
        )
    }

    #[test]
    fn d1_gate_admits_bitset_just_below_threshold() {
        assert_eq!(run_dispatch_at(1, 10), StrategyChoice::BitsetFromPostings);
    }

    #[test]
    fn d1_gate_rejects_bitset_just_above_threshold() {
        assert_eq!(run_dispatch_at(1, 11), StrategyChoice::LinearScan);
    }

    #[test]
    fn d2_gate_admits_bitset_just_below_threshold() {
        assert_eq!(run_dispatch_at(2, 100), StrategyChoice::BitsetFromPostings);
    }

    #[test]
    fn d2_gate_rejects_bitset_just_above_threshold() {
        assert_eq!(run_dispatch_at(2, 101), StrategyChoice::LinearScan);
    }

    #[test]
    fn d50_uses_the_multi_value_gate() {
        assert_eq!(run_dispatch_at(50, 20), StrategyChoice::BitsetFromPostings);
    }

    #[test]
    fn gallop_admits_at_any_density_when_segment_is_sorted() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(
            n,
            Some(IndexSortByField {
                field: "fk".to_string(),
                order: Order::Asc,
            }),
            |i| i,
        );
        let (reader, column) = open_column(&index, &name);
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: Some(1),
            },
            &term_set(0..50),
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, StrategyChoice::Gallop(Order::Asc));
    }

    #[test]
    fn empty_inputs_route_to_empty_strategy() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = plan_slice(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: Some(1),
            },
            &[],
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, StrategyChoice::Empty);
    }
}
