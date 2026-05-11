//! Strategy selection for `FastFieldTermSetQuery`.
//!
//! Mirrors the structure of `range_query::range_query_fastfield`:
//! `Weight::scorer()` consults this module to decide between `Gallop`,
//! `BitsetFromPostings`, and `LinearScan`, then constructs the right `DocSet`.
//!
//! See `design-doc.md` for the decision tree and Phase 5c bench data for the
//! `BitsetFromPostings` vs `LinearScan` cost model.
//!
//! ## Note on `inputs.avg_docs_per_term` (`D`)
//!
//! Computing `D` precisely from the column alone would require posting-list
//! lookups (which is some of the work `BitsetFromPostings` does anyway). We
//! accept `None`, which `select_strategy` treats as `D = 1`. The first-column
//! dispatch uses `D` only to bias the `BitsetFromPostings` vs `LinearScan`
//! cutoff via `bitset_max_density`; a missing estimate degrades to the
//! conservative `D = 1` reading, which broadens the bitset-eligible region.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use columnar::{Cardinality, Column};

use crate::index::SegmentReader;
use crate::Order;

/// Numeric tag for the chosen strategy. Used by the optional
/// `TermSetStrategyConfig::strategy_sink` so consumers (paradedb) can surface
/// the per-segment dispatch decision in `EXPLAIN ANALYZE` without depending
/// on `TermSetStrategy` directly.
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StrategyTag {
    None = 0,
    Gallop = 1,
    Linear = 2,
    Bitset = 3,
    Empty = 4,
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
            _ => Err("unknown StrategyTag value"),
        }
    }
}

/// User-tunable density thresholds. Defaults match the starting estimates in
/// `design.md` §4 and are overridden via the `paradedb.term_set_*_max_density`
/// GUCs at the consumer side.
///
/// "Density" is a unitless ratio of *matching count over corpus count*. For
/// gallop, that's `K' / N`; for bitset on the first column, it's `K' · D / N`;
/// for the subsequent-column branch, `K' · D / C`. Each strategy fires when
/// its density is *below* the corresponding `_max_density` threshold.
///
/// We use `f64` rather than integer denominators so bench-derived
/// thresholds don't have to round to `1/N` for small `N`.
#[derive(Clone, Debug)]
pub struct TermSetStrategyConfig {
    /// Kill-switch: when `false`, the planner never returns `Gallop` even if
    /// every other gate would pass.
    pub gallop_enabled: bool,
    /// Gallop fires when `K' / N < gallop_max_density` on a sorted segment.
    /// Default `1/100 = 0.01`. At K/N = 0.01 gallop wins ~2.6× over linear
    /// on LowFk-shaped corpora with consistent behavior across N values.
    /// The corpus generator caps `distinct = N/100`, so the empirical
    /// crossover above 0.01 isn't pinned for LowFk; PK-shaped corpora
    /// show a different (D-dependent) crossover — see Follow-up H.
    pub gallop_max_density: f64,
    /// First-column `BitsetFromPostings` threshold: `K' · D / N` cutoff. The
    /// strategy fires when `K' · D / N < bitset_max_density`; otherwise the
    /// planner falls through to `LinearScan`.
    pub bitset_max_density: f64,
    /// Subsequent-column `BitsetFromPostings` threshold: `K' · D / C` cutoff.
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
            // gallop_max_density: 1/100 — bench-tuned (see field doc).
            // bitset_max_density: 1/50 — calibrated Phase 5e from N ∈ {1M, 20M,
            //   50M} sweeps on PK-shape (D=1). Direct-lookup `BitsetFromPostings`
            //   loses to `LinearScan` once K·D/N crosses ~1/50; the strict
            //   less-than gate keeps bitset firing below that ratio.
            // subsequent_bitset_max_density: 1/4 — unchanged from Phase 5c
            //   starting estimate; subsequent-column bench coverage is thinner,
            //   defer tightening until we have data.
            gallop_max_density: 1.0 / 100.0,
            bitset_max_density: 1.0 / 50.0,
            subsequent_bitset_max_density: 1.0 / 4.0,
            strategy_sink: None,
        }
    }
}

/// Selected execution strategy. Set of variants that can be dispatched today.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TermSetStrategy {
    /// Sorted-segment fast path. Carries the pruned, ascending-sorted term list
    /// because the planner already had to compute it for min/max pruning.
    Gallop {
        sort_order: Order,
        sorted_terms: Vec<u64>,
    },
    /// `TermSetDocSet` over the fast field. Terminal fallback when no other
    /// strategy qualifies.
    LinearScan,
    /// Direct `TermDictionary::get(key)` lookups + bitset OR. The K-bounded
    /// inverted-index path; see `term_set_bitset.rs`.
    BitsetFromPostings,
    /// The result is definitively empty; no posting list reads or column
    /// scans needed. Returned when the planner can prove no docs can match:
    ///   - `n == 0` (empty segment)
    ///   - `term_set.is_empty()` (no query terms)
    ///   - `k_prime == 0` (all query terms pruned outside `[min, max]`)
    ///   - `candidate_size == 0` (subsequent-column branch with empty upstream)
    /// The dispatch site emits an `EmptyScorer` for this variant. Closing
    /// this gap was Phase 5i — pre-fix the planner returned `LinearScan`
    /// here and the dispatch site walked every doc in the segment doing
    /// hashset probes that all failed by construction.
    Empty,
}

/// Inputs to the planner that aren't already on `Column`.
pub struct PlannerInputs<'a> {
    pub field_name: &'a str,
    /// Upstream candidate-set size. `None` ≡ full segment scan (first column).
    pub candidate_size: Option<u32>,
    /// Average documents per term in this column. `None` is treated as `D = 1`.
    pub avg_docs_per_term: Option<u32>,
}

/// Strategy dispatch for `FastFieldTermSetQuery`.
///
/// # Decision tree (evaluated in order)
///
/// ## Tier 1 — upstream: `TermSetQuery::specialized_weight`
///
/// `TermSetQuery` itself doesn't reach this function unless tier 1 routes
/// to `FastFieldTermSetWeight`. The tier-1 fork is on field type only:
///
///   - field is `is_fast()` AND value type is U64/I64/F64/Date/IpAddr → `FastFieldTermSetWeight`
///     (this dispatch table runs for every K).
///   - Otherwise (non-fast field, or `Bool/Json/Str/...` value type) → `AutomatonWeight` over the
///     inverted index (streaming dictionary walk + `BufferedUnionScorer` heap merge of K posting
///     lists). The strategy framework below never sees the query in this branch.
///
/// Phase 5h removed the prior `terms.len() > 1024` short-circuit that
/// forced small-K queries on fast-supported types into `AutomatonWeight`.
/// Tier 2 below handles every K cleanly: small K on a sorted column hits
/// `Gallop`; small K on an unsorted column hits `BitsetFromPostings` with
/// direct dictionary lookups — both much faster than the streaming
/// `AutomatonWeight` walk on a large dictionary. `AutomatonWeight` is now
/// reserved exclusively for fields where the fast-field path doesn't
/// apply.
///
/// ## Tier 2 — this function (`select_strategy` over a `Column<u64>`)
///
/// 1. **Defensive empty checks** — if `n == 0`, the input term set is empty, or all terms prune
///    outside `[column.min, column.max]`, return `LinearScan`. Upstream callers handle the
///    truly-empty result via `EmptyScorer`; this arm just keeps the planner total over its
///    well-defined domain.
///
/// 2. **Gallop** — sort-dependent fast path. Fires when all of:
///      - `cfg.gallop_enabled` (kill switch, default true)
///      - `inputs.candidate_size.is_none_or(|c| c == n)`  (first column; candidate-set filtering
///        hasn't already reduced N)
///      - column cardinality is `Full` or `Optional` (gallop's range walk assumes one value per
///        doc)
///      - the segment has a `sort_by_field` matching `inputs.field_name`
///      - `K' / N < cfg.gallop_max_density`  (default 1/100 = 0.01)
///    `Gallop` reads the fast-field column directly with exponential jumps;
///    cost is `O(K' · log(N/K'))`. Preferred over all inverted-index
///    strategies when the column is sort-compatible with the filter, since
///    no other strategy exploits sortedness.
///
/// 3. **Subsequent-column branch** — when an upstream candidate set has already narrowed work to
///    `C` candidates (`inputs.candidate_size < n`):
///      - `c > 0` AND `K' · D / C < cfg.subsequent_bitset_max_density` (default 1/4) →
///        `BitsetFromPostings`
///      - Otherwise → `LinearScan`
///    The bitset path here still streams K direct dictionary lookups (the
///    `BitsetFromPostings` algorithm is the same as the first-column
///    branch); it's the eligibility threshold that differs.
///
/// 4. **First-column branch** — full segment, no upstream filter:
///      - `K' · D / N < cfg.bitset_max_density` (default 1/50 = 0.02) → `BitsetFromPostings`.
///        Direct dictionary lookups + bitset OR. Wins 5–250× over `LinearScan` in this band per
///        Phase 5c sweeps; see PR #<NN>.
///      - Otherwise → `LinearScan`. Single O(N) pass with `HashSet` probe per doc; wins when K is a
///        large enough fraction of N that K discrete dictionary lookups exceed one full scan.
///
/// # Constants
///
/// All "density" values are unitless ratios; the strict less-than gate
/// means the threshold value itself routes to the next-tier strategy.
///
///   - `cfg.gallop_max_density`              default 1/100 = 0.01
///   - `cfg.bitset_max_density`              default 1/50  = 0.02
///   - `cfg.subsequent_bitset_max_density`   default 1/4   = 0.25
///   - `cfg.gallop_enabled`                  default true
///
/// `D` (`avg_docs_per_term`) defaults to 1 when the caller doesn't supply
/// it. The dispatch site in `FastFieldTermSetWeight::scorer` currently
/// always passes `None`; the threshold gates therefore reduce to `K' / N`
/// and `K' / C` in practice. Plumbing a real per-column D estimator is a
/// separate follow-up.
///
/// # Strategy sink for EXPLAIN
///
/// When `cfg.strategy_sink` is `Some`, the chosen `StrategyTag` is stored
/// (as `u8`) on the sink before returning so consumers can surface it in
/// `EXPLAIN ANALYZE`. The thin wrapper around the inner planner keeps the
/// per-segment hot path uncluttered.
pub fn select_strategy(
    reader: &SegmentReader,
    column: &Column<u64>,
    inputs: PlannerInputs<'_>,
    term_set: &[u64],
    cfg: &TermSetStrategyConfig,
) -> TermSetStrategy {
    let strat = select_strategy_inner(reader, column, inputs, term_set, cfg);
    if let Some(sink) = cfg.strategy_sink.as_ref() {
        let tag = match &strat {
            TermSetStrategy::Gallop { .. } => StrategyTag::Gallop,
            TermSetStrategy::LinearScan => StrategyTag::Linear,
            TermSetStrategy::BitsetFromPostings => StrategyTag::Bitset,
            TermSetStrategy::Empty => StrategyTag::Empty,
        };
        sink.store(tag as u8, Ordering::Relaxed);
    }
    strat
}

fn select_strategy_inner(
    reader: &SegmentReader,
    column: &Column<u64>,
    inputs: PlannerInputs<'_>,
    term_set: &[u64],
    cfg: &TermSetStrategyConfig,
) -> TermSetStrategy {
    let n = column.num_docs();
    if n == 0 || term_set.is_empty() {
        // Empty segment or empty query → no docs can match.
        return TermSetStrategy::Empty;
    }

    // Count terms surviving min/max pruning. Non-gallop branches only need
    // K' to evaluate density gates; allocating a `Vec<u64>` here just to
    // discard it on those branches is wasted work. The Gallop branch below
    // does the allocation only when it commits to that strategy.
    let lo = column.min_value();
    let hi = column.max_value();
    let in_range = |v: u64| v >= lo && v <= hi;
    let k_prime = term_set.iter().copied().filter(|&v| in_range(v)).count() as u32;
    if k_prime == 0 {
        // All query terms fall outside the column's [min, max] — no doc
        // can match. Returning `LinearScan` here would let the dispatch
        // site walk every doc in the segment doing hashset probes that
        // we already know will fail; returning `Empty` lets it short
        // circuit to `EmptyScorer`. (Phase 5i fix.)
        return TermSetStrategy::Empty;
    }

    let candidate_eq_n = inputs.candidate_size.is_none_or(|c| c == n);
    let cardinality = column.get_cardinality();
    let n_f = n as f64;

    // Try gallop (sort-dependent fast path).
    if cfg.gallop_enabled
        && candidate_eq_n
        && matches!(cardinality, Cardinality::Full | Cardinality::Optional)
    {
        if let Some(order) = reader
            .sort_by_field()
            .filter(|sbf| sbf.field == inputs.field_name)
            .map(|sbf| sbf.order)
        {
            // K' / N < gallop_max_density. Strict less-than matches the
            // pre-refactor integer form `K' < N / R_GALLOP` at every
            // exactly-representable density (the defaults are 1/2^k).
            if (k_prime as f64) / n_f < cfg.gallop_max_density {
                // Gallop is the only branch that needs the pruned Vec
                // (sorted + deduped + handed to `TermSetGallopDocSet`), so
                // allocate it only here.
                let mut pruned: Vec<u64> =
                    term_set.iter().copied().filter(|&v| in_range(v)).collect();
                pruned.sort_unstable();
                // Dedup duplicate input terms. `TermSetGallopDocSet`
                // tolerates duplicates (the second occurrence finds an
                // empty range and skips), but each duplicate still pays
                // two `gallop_search_sorted` probes — cheap insurance on
                // the hot path. Production input today flows through
                // HashSet-derived sources so this is a no-op there;
                // `TermSetQuery::new` doesn't enforce uniqueness on its
                // input, hence the belt-and-suspenders dedup.
                pruned.dedup();
                return TermSetStrategy::Gallop {
                    sort_order: order,
                    sorted_terms: pruned,
                };
            }
        }
    }

    // Sort-agnostic selection.
    let d = inputs.avg_docs_per_term.unwrap_or(1) as u64;
    let kd = (k_prime as u64).saturating_mul(d) as f64;

    if let Some(c) = inputs.candidate_size.filter(|&c| c < n) {
        // Subsequent column. `c == 0` means the upstream filter produced no
        // candidates — no docs can match this filter either. For `c > 0`,
        // fire `BitsetFromPostings` when `K·D/C` falls below the threshold;
        // otherwise default to `LinearScan`.
        if c == 0 {
            return TermSetStrategy::Empty;
        }
        if kd / (c as f64) < cfg.subsequent_bitset_max_density {
            return TermSetStrategy::BitsetFromPostings;
        }
        TermSetStrategy::LinearScan
    } else {
        // First column, post-gallop.
        let kd_density = kd / n_f;
        if kd_density < cfg.bitset_max_density {
            return TermSetStrategy::BitsetFromPostings;
        }
        TermSetStrategy::LinearScan
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests pin the planner's *output shape* on cases it can decide
    //! today. Once `inputs.avg_docs_per_term` is populated from a real column
    //! estimator, the `K' · D < threshold` arithmetic will shift and these
    //! tests may need to be revisited.
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

    #[test]
    fn select_handles_min_max_pruning_dropping_all_terms() {
        // Column values are 0..N, so terms entirely outside [0, N-1] all prune.
        let (index, _field, name) = build_index(1024, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = select_strategy(
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
        // All query terms fall outside the column's [min, max]. The planner
        // proves no docs can match and returns `Empty` so the dispatch site
        // emits `EmptyScorer` instead of walking the segment with a hashset
        // of guaranteed-to-miss values. (Phase 5i.)
        assert_eq!(strat, TermSetStrategy::Empty);
    }

    #[test]
    fn select_returns_bitset_from_postings_when_gallop_rejected_due_to_high_k_planner_shape_only() {
        // Sorted ASC, but K is *too big* for gallop: K'/N >= gallop_max_density.
        // With D = 1, we want K'·D / N < bitset_max_density so the
        // sort-agnostic branch lands on BitsetFromPostings.
        // Defaults (Phase 5e): gallop=1/100=0.01, bitset=1/50=0.02.
        // N = 4096, K = 60 → K/N = 60/4096 ≈ 0.0146:
        //   0.0146 ≥ 0.01  → gallop rejected
        //   0.0146 < 0.02  → bitset accepted
        let n: u64 = 4096;
        let (index, _field, name) = build_index(
            n,
            Some(IndexSortByField {
                field: "fk".to_string(),
                order: Order::Asc,
            }),
            |i| i, // every value distinct, all in range
        );
        let (reader, column) = open_column(&index, &name);
        let strat = select_strategy(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: None,
            },
            &term_set(0..60), // K' = 60 (all in [0, 4095])
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, TermSetStrategy::BitsetFromPostings);
    }

    #[test]
    fn select_returns_bitset_when_first_column_highly_selective() {
        // Unsorted, K well below `bitset_max_density`.
        // N = 4096, K = 8, D = 1 → K/N = 0.00195 < 0.25.
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = select_strategy(
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
        assert_eq!(strat, TermSetStrategy::BitsetFromPostings);
    }

    #[test]
    fn select_falls_back_to_linear_scan_when_kd_exceeds_bitset_threshold() {
        // Unsorted, K large enough that K·D / N >= bitset_max_density.
        // With D = 1 and bitset_max_density = 1/4 = 0.25,
        // N = 4096 and K = 2000 → K/N ≈ 0.488 > 0.25 → LinearScan.
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = select_strategy(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: None,
            },
            &term_set(0..2000),
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, TermSetStrategy::LinearScan);
    }

    #[test]
    fn select_returns_linear_scan_for_subsequent_column_when_kd_exceeds_threshold() {
        // Subsequent column with c=100, K·D/c = 50/100 = 0.5 > 0.25 →
        // bitset rejected, falls through to LinearScan.
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = select_strategy(
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
        assert_eq!(strat, TermSetStrategy::LinearScan);
    }

    #[test]
    fn select_returns_bitset_from_postings_when_subsequent_column_kd_below_threshold() {
        // Subsequent column: K·D / C < subsequent_bitset_max_density.
        // K·D / C = 50 / 1024 ≈ 0.0488 < 1/4 = 0.25 → BitsetFromPostings.
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = select_strategy(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: Some(1024),
                avg_docs_per_term: None,
            },
            &term_set(0..50),
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, TermSetStrategy::BitsetFromPostings);
    }

    /// Gallop precondition: only fires on sorted segments. The decision
    /// tree must still produce *some* sensible non-Gallop variant on
    /// unsorted input.
    #[test]
    fn select_skips_gallop_when_unsorted() {
        let n: u64 = 4096;
        let (index, _field, name) = build_index(n, None, |i| i);
        let (reader, column) = open_column(&index, &name);
        let strat = select_strategy(
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
        assert!(!matches!(strat, TermSetStrategy::Gallop { .. }));
    }

    /// Gallop precondition: even with sort_by, a *different* sort field
    /// disables gallop. We build an index with two fast fields, sort by the
    /// second, and query the first — so
    /// `sort_by_field().field != inputs.field_name` fires.
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
        let strat = select_strategy(
            &reader,
            &column,
            PlannerInputs {
                field_name: "fk",
                candidate_size: None,
                avg_docs_per_term: None,
            },
            &term_set(0..8),
            &TermSetStrategyConfig::default(),
        );
        assert!(!matches!(strat, TermSetStrategy::Gallop { .. }));
        // With K=8 on N=4096, K/N = 0.00195 < bitset_max_density (0.25), so
        // the first-column branch picks BitsetFromPostings.
        assert_eq!(strat, TermSetStrategy::BitsetFromPostings);
    }

    /// Positive gallop assertion. Sorted ASC, K small, K'/N below the
    /// gallop_max_density threshold.
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
        let strat = select_strategy(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: None,
            },
            &term_set(0..8), // K' = 8 < 4096/64 = 64
            &TermSetStrategyConfig::default(),
        );
        match strat {
            TermSetStrategy::Gallop {
                sort_order,
                sorted_terms,
            } => {
                assert!(matches!(sort_order, Order::Asc));
                // Pruned + sorted ascending; pruning preserves all 8 (all in [0, 4095]).
                assert_eq!(sorted_terms, vec![0, 1, 2, 3, 4, 5, 6, 7]);
            }
            other => panic!("expected Gallop, got {other:?}"),
        }
    }

    /// `strategy_sink` records the chosen variant on every call.
    /// `select_strategy` is invoked twice against differently-shaped corpora;
    /// the sink should reflect the *latest* strategy on each call (last-write
    /// semantics).
    #[test]
    fn select_writes_chosen_tag_to_strategy_sink() {
        use std::sync::atomic::{AtomicU8, Ordering};
        use std::sync::Arc;

        let n: u64 = 4096;
        let sink = Arc::new(AtomicU8::new(StrategyTag::None as u8));
        let cfg = TermSetStrategyConfig {
            strategy_sink: Some(sink.clone()),
            ..TermSetStrategyConfig::default()
        };

        // Sorted ASC, K=8 → Gallop.
        let (sorted_idx, _f, name_s) = build_index(
            n,
            Some(IndexSortByField {
                field: "fk".to_string(),
                order: Order::Asc,
            }),
            |i| i,
        );
        let (reader_s, col_s) = open_column(&sorted_idx, &name_s);
        let _ = select_strategy(
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

        // Unsorted, K=2000 → LinearScan (terminal fallback). Last-write-wins
        // overwrites the GALLOP tag from the previous call.
        let (unsorted_idx, _f, name_u) = build_index(n, None, |i| i);
        let (reader_u, col_u) = open_column(&unsorted_idx, &name_u);
        let _ = select_strategy(
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

    /// Kill-switch: gallop_enabled=false forces a non-gallop strategy
    /// even when the segment is sorted and K is tiny.
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
            ..TermSetStrategyConfig::default()
        };
        let strat = select_strategy(
            &reader,
            &column,
            PlannerInputs {
                field_name: &name,
                candidate_size: None,
                avg_docs_per_term: None,
            },
            &term_set(0..8),
            &cfg,
        );
        assert!(!matches!(strat, TermSetStrategy::Gallop { .. }));
        // K=8 on N=4096, K·D/N = 0.00195 < bitset_max_density (0.25), so the
        // first-column branch picks BitsetFromPostings.
        assert_eq!(strat, TermSetStrategy::BitsetFromPostings);
    }
}
