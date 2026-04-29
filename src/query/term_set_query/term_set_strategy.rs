//! Strategy selection for `FastFieldTermSetQuery`.
//!
//! Mirrors the structure of `range_query::range_query_fastfield`:
//! `Weight::scorer()` consults this module to decide between gallop, linear, or
//! future strategies, then constructs the right `DocSet`.
//!
//! See `design.md` §4 (decision tree) and `implementation.md` §2.1 for the
//! background. For #4895 the planner returns its full set of variants from day
//! one — the dispatch in `FastFieldTermSetWeight::scorer()` will route the
//! non-`Gallop` / non-`LinearScan` variants to the existing `TermSetDocSet`
//! until follow-ups A and B fill them in.
//!
//! ## Note on `inputs.avg_docs_per_term` (`D`)
//!
//! Computing `D` precisely from the column alone would require posting-list
//! lookups (which are exactly the work Strategy 4 wants to amortize). For
//! Step 2 we accept `None`, which `select_strategy` treats as `D = 1`. That
//! biases Step 2's first-column branch toward `PostingListDirect`, but since
//! the dispatch site stubs both `PostingListDirect` and `BitsetFromPostings`
//! to `LinearScan` until follow-up A/B land, the bias is invisible at runtime
//! and only shows up in unit tests on the planner.

use columnar::{Cardinality, Column};
use rustc_hash::FxHashSet;

use crate::index::SegmentReader;
use crate::Order;

/// User-tunable thresholds. Defaults match the starting estimates in
/// `design.md` §4 and are overridden via the `paradedb.term_set_*_ratio` GUCs
/// at the consumer side.
#[derive(Clone, Debug)]
pub struct TermSetStrategyConfig {
    /// Kill-switch: when `false`, the planner never returns `Gallop` even if
    /// every other gate would pass.
    pub gallop_enabled: bool,
    /// `R_GALLOP`: gallop is taken when `K' < N / R_GALLOP` on a sorted segment.
    pub gallop_ratio: u32,
    /// `R_POSTING`: first-column posting-list direct selection threshold.
    pub posting_ratio: u32,
    /// `R_BITSET`: first-column bitset-from-postings selection threshold.
    pub bitset_ratio: u32,
    /// `R_HASH_PROBE`: subsequent-column hash-probe selection threshold.
    pub hash_probe_ratio: u32,
    /// `R_SUBSEQUENT_BITSET`: subsequent-column bitset-from-postings threshold.
    pub subsequent_bitset_ratio: u32,
}

impl Default for TermSetStrategyConfig {
    fn default() -> Self {
        Self {
            gallop_enabled: true,
            gallop_ratio: 64,
            posting_ratio: 256,
            bitset_ratio: 4,
            hash_probe_ratio: 16,
            subsequent_bitset_ratio: 4,
        }
    }
}

/// Selected execution strategy. The non-`Gallop` / non-`LinearScan` variants
/// are reserved for follow-ups A and B; `select_strategy` returns them today
/// so unit tests can pin the planner shape, but `scorer()` routes them through
/// the `TermSetDocSet` linear path until those strategies are implemented.
// Step 2 stages the planner without wiring the dispatch in `scorer()`; Step 3
// activates the call site, at which point these items become live.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TermSetStrategy {
    /// Sorted-segment fast path. Carries the pruned, ascending-sorted term list
    /// because the planner already had to compute it for min/max pruning.
    Gallop {
        sort_order: Order,
        sorted_terms: Vec<u64>,
    },
    /// Today's `TermSetDocSet`. Also the terminal fallback when no other
    /// strategy qualifies.
    LinearScan,
    /// Reserved (follow-up B).
    BitsetFromPostings,
    /// Reserved (follow-up A).
    PostingListDirect,
    /// Reserved (subsequent-column work, post-#4895).
    HashProbe,
}

/// Inputs to the planner that aren't already on `Column`.
#[allow(dead_code)]
pub struct PlannerInputs<'a> {
    pub field_name: &'a str,
    /// Upstream candidate-set size. `None` ≡ full segment scan (first column).
    pub candidate_size: Option<u32>,
    /// Average documents per term in this column. `None` is treated as `D = 1`.
    pub avg_docs_per_term: Option<u32>,
}

/// Run the decision tree from `design.md` §4.
///
/// Step 0 prunes the term set against the column's `[min, max]`. Step 1 is the
/// sort-dependent gallop fast path. Step 2 is the sort-agnostic dispatch
/// branching on `C < N` (subsequent column) vs `C == N` (first column).
#[allow(dead_code)]
pub fn select_strategy(
    reader: &SegmentReader,
    column: &Column<u64>,
    inputs: PlannerInputs<'_>,
    term_set: &FxHashSet<u64>,
    cfg: &TermSetStrategyConfig,
) -> TermSetStrategy {
    let n = column.num_docs();
    if n == 0 || term_set.is_empty() {
        // The empty-input case is handled upstream as `EmptyScorer`; falling
        // through to `LinearScan` here is a no-op for correctness.
        return TermSetStrategy::LinearScan;
    }

    // Step 0: prune to [min, max].
    let lo = column.min_value();
    let hi = column.max_value();
    let mut pruned: Vec<u64> = term_set
        .iter()
        .copied()
        .filter(|v| *v >= lo && *v <= hi)
        .collect();
    let k_prime = pruned.len() as u32;
    if k_prime == 0 {
        // All terms pruned — `scorer()` will turn the empty pruned set into
        // `EmptyScorer`. We still return `LinearScan` here because Step 2's
        // implementation of "no docs match anything" is the natural fallback.
        return TermSetStrategy::LinearScan;
    }

    let candidate_eq_n = inputs.candidate_size.is_none_or(|c| c == n);
    let cardinality = column.get_cardinality();

    // Step 1: try gallop (sort-dependent fast path).
    if cfg.gallop_enabled
        && candidate_eq_n
        && matches!(cardinality, Cardinality::Full | Cardinality::Optional)
    {
        if let Some(order) = reader
            .sort_by_field()
            .filter(|sbf| sbf.field == inputs.field_name)
            .map(|sbf| sbf.order)
        {
            let r = cfg.gallop_ratio.max(1);
            if k_prime < n / r {
                pruned.sort_unstable();
                return TermSetStrategy::Gallop {
                    sort_order: order,
                    sorted_terms: pruned,
                };
            }
        }
    }

    // Step 2: sort-agnostic selection.
    let d = inputs.avg_docs_per_term.unwrap_or(1) as u64;
    let kd = (k_prime as u64).saturating_mul(d);

    if let Some(c) = inputs.candidate_size.filter(|&c| c < n) {
        // Subsequent column.
        let hash_threshold = n / cfg.hash_probe_ratio.max(1);
        if c < hash_threshold {
            return TermSetStrategy::HashProbe;
        }
        let bitset_threshold = (c as u64) / cfg.subsequent_bitset_ratio.max(1) as u64;
        if kd < bitset_threshold {
            return TermSetStrategy::BitsetFromPostings;
        }
        TermSetStrategy::HashProbe
    } else {
        // First column, post-gallop.
        let posting_threshold = (n as u64) / cfg.posting_ratio.max(1) as u64;
        if kd < posting_threshold {
            return TermSetStrategy::PostingListDirect;
        }
        let bitset_threshold = (n as u64) / cfg.bitset_ratio.max(1) as u64;
        if kd < bitset_threshold {
            return TermSetStrategy::BitsetFromPostings;
        }
        TermSetStrategy::LinearScan
    }
}

#[cfg(test)]
mod tests {
    //! Step 2 unit tests pin the planner's *output shape* on the cases it can
    //! decide today. Tests asserting `Gallop` returns are deferred to Step 3,
    //! where the gallop dispatch is wired and end-to-end behavior is observable.
    //!
    //! The `_planner_shape_only` suffix is a deliberate signal: today the
    //! dispatch maps every non-Gallop / non-LinearScan variant to
    //! `TermSetDocSet`. Once Follow-up A populates `inputs.avg_docs_per_term`
    //! properly, the `K' · D < threshold` arithmetic will shift and these
    //! tests will need to be revisited.
    use rustc_hash::FxHashSet;

    use super::*;
    use crate::schema::{NumericOptions, SchemaBuilder};
    use crate::{doc, Index, IndexSettings, IndexSortByField};

    fn build_index(
        n: u64,
        sort: Option<IndexSortByField>,
        value_for_doc: impl Fn(u64) -> u64,
    ) -> (Index, crate::schema::Field, String) {
        let mut sb = SchemaBuilder::new();
        let field = sb.add_u64_field(
            "fk",
            NumericOptions::default().set_fast().set_indexed(),
        );
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

    fn term_set(values: impl IntoIterator<Item = u64>) -> FxHashSet<u64> {
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
        // All terms pruned → LinearScan (scorer() will turn it into EmptyScorer).
        assert_eq!(strat, TermSetStrategy::LinearScan);
    }

    #[test]
    fn select_returns_bitset_from_postings_when_gallop_rejected_due_to_high_k_planner_shape_only() {
        // Sorted ASC, but K is *too big* for gallop: K >= N / R_GALLOP.
        // With D = 1, we want K · D < N / R_BITSET as well so Step 2 lands on
        // BitsetFromPostings (and *not* PostingListDirect, which requires
        // K · D < N / R_POSTING — a tighter threshold).
        // Pick N = 4096, R_GALLOP = 64, R_POSTING = 256, R_BITSET = 4.
        //   gallop crossover  = 4096 / 64  = 64
        //   posting crossover = 4096 / 256 = 16
        //   bitset crossover  = 4096 / 4   = 1024
        // K = 200 satisfies 64 ≤ K (gallop rejected), 200 ≥ 16 (posting rejected),
        // 200 < 1024 (bitset accepted). N rounded up to 4096 docs.
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
            &term_set(0..200), // K' = 200 (all in [0, 4095])
            &TermSetStrategyConfig::default(),
        );
        assert_eq!(strat, TermSetStrategy::BitsetFromPostings);
    }

    #[test]
    fn select_returns_posting_direct_when_first_column_highly_selective_planner_shape_only() {
        // Unsorted, K small enough that K · D < N / R_POSTING.
        // N = 4096, R_POSTING = 256 → posting threshold = 16. K = 8 < 16.
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
        assert_eq!(strat, TermSetStrategy::PostingListDirect);
    }

    #[test]
    fn select_falls_back_to_linear_scan_only_when_kd_exceeds_all_thresholds() {
        // Unsorted, K large enough that K · D >= every threshold in Step 2's
        // first-column branch. With D = 1 and R_BITSET = 4 → bitset threshold
        // = N / 4 = 1024. K = 2000 > 1024 → LinearScan.
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
    fn select_returns_hash_probe_when_subsequent_column_with_small_c_planner_shape_only() {
        // Subsequent column: candidate_size = Some(c) where c < N / R_HASH_PROBE.
        // N = 4096, R_HASH_PROBE = 16 → hash threshold = 256. C = 100 < 256.
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
        assert_eq!(strat, TermSetStrategy::HashProbe);
    }

    #[test]
    fn select_returns_bitset_from_postings_when_subsequent_column_kd_below_threshold_planner_shape_only(
    ) {
        // Subsequent column branch: K · D < C / R_SUBSEQUENT_BITSET.
        // N = 4096, hash threshold = N / 16 = 256.
        // C = 1024 ≥ 256 → past the HashProbe gate, falling into the bitset arm.
        // bitset threshold = C / R_SUBSEQUENT_BITSET = 1024 / 4 = 256.
        // K · D = 50 < 256 → BitsetFromPostings.
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

    /// Step 1 gate: gallop only fires on sorted segments. The decision tree
    /// must still produce *some* sensible non-Gallop variant on unsorted input.
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

    /// Step 1 gate: even with sort_by, a *different* sort field disables gallop.
    /// We build an index with two fast fields, sort by the second, and query
    /// the first — so `sort_by_field().field != inputs.field_name` fires.
    #[test]
    fn select_skips_gallop_when_field_mismatches_sort_field() {
        let n: u64 = 4096;
        let mut sb = SchemaBuilder::new();
        let fk = sb.add_u64_field("fk", NumericOptions::default().set_fast().set_indexed());
        let other =
            sb.add_u64_field("other", NumericOptions::default().set_fast().set_indexed());
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
        // With K=8, Step 2 first-column branch picks PostingListDirect.
        assert_eq!(strat, TermSetStrategy::PostingListDirect);
    }

    // Note: `select_skips_gallop_when_multivalued` and the positive
    // `select_returns_gallop_*` assertions live in Step 3, where the gallop
    // dispatch is wired and we can drive multi-valued and end-to-end equivalence
    // corpora without overloading this unit-test fixture.
}
