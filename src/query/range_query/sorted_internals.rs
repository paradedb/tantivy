//! Binary-search primitives shared by sorted-segment fast paths.
//!
//! Extracted from `range_query_fastfield.rs` so the term-set gallop strategy
//! (paradedb/paradedb#4895) can reuse them without duplicating numeric logic.
//! Behavior is unchanged from the original site; this is a pure relocation.
//!
//! The DESC monotonicity invariant matters for callers that use a forward-only
//! cursor across multiple searches: `binary_search_sorted` always returns a
//! position in `[lo, hi]`, never below `lo`. Preserving this lets
//! `term_set_gallop`'s shrinking-window cursor advance monotonically through
//! DocId space.

use columnar::Column;

use crate::Order;

/// Binary search for the boundary between NULLs and non-NULLs.
///
/// This is separated from value search because NULL docs have no stored value —
/// `column.first(doc)` returns `None`. We can only test presence (`is_some()`),
/// not compare against a target value. Once the NULL boundary is known, the
/// non-NULL range is passed to `binary_search_sorted` which can safely `.expect()`
/// on every lookup.
///
/// - `Order::Asc`: NULLs are at the start. Returns the first DocId with a value.
/// - `Order::Desc`: NULLs are at the end. Returns the first DocId without a value (i.e., past all
///   valued docs).
pub(crate) fn binary_search_null_boundary(
    column: &Column<u64>,
    lo: u32,
    hi: u32,
    order: Order,
) -> u32 {
    let mut lo = lo;
    let mut hi = hi;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let has_value = column.first(mid).is_some();
        match order {
            Order::Asc => {
                // NULLs at start. Looking for first doc WITH a value.
                if has_value {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            Order::Desc => {
                // NULLs at end. Looking for first doc WITHOUT a value.
                if has_value {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
        }
    }
    lo
}

/// Binary search on a sorted column for the boundary of a value range.
///
/// Returns a DocId forming one side of the half-open range `[start, end)`:
/// - `strict=false` (inclusive): first doc whose value is at or past `target` — used for `start`.
/// - `strict=true` (exclusive): first doc whose value is strictly past `target` — used for `end`.
///
/// The caller guarantees that `[lo, hi)` contains only non-NULL docs
/// (the NULL boundary was already computed by `binary_search_null_boundary`).
pub(crate) fn binary_search_sorted(
    column: &Column<u64>,
    lo: u32,
    hi: u32,
    target: u64,
    order: Order,
    strict: bool,
) -> u32 {
    let mut lo = lo;
    let mut hi = hi;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        // Safe: caller guarantees [lo, hi) is non-NULL (see binary_search_null_boundary).
        let val = column
            .first(mid)
            .expect("doc in non-NULL range has no value");
        let go_right = match (order, strict) {
            (Order::Asc, false) => val < target,
            (Order::Asc, true) => val <= target,
            (Order::Desc, false) => val > target,
            (Order::Desc, true) => val >= target,
        };
        if go_right {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Whether the search predicate at `val` says "advance further forward in the
/// window". `binary_search_sorted` and `gallop_search_sorted` share this
/// monotonicity: across any sorted column window, `go_right` is true for a
/// prefix `[lo, k)` and false for the suffix `[k, hi)`. The answer is `k`.
#[inline]
fn go_right(val: u64, target: u64, order: Order, strict: bool) -> bool {
    match (order, strict) {
        (Order::Asc, false) => val < target,
        (Order::Asc, true) => val <= target,
        (Order::Desc, false) => val > target,
        (Order::Desc, true) => val >= target,
    }
}

/// Galloping (exponential probe + bounded binary search) over a sorted column.
///
/// Same semantics as [`binary_search_sorted`]: returns the first DocId in
/// `[lo, hi]` where the `(order, strict)` predicate stops advancing forward.
/// Cost is `O(log d)` where `d` is the distance from `lo` to the answer, vs
/// `binary_search_sorted`'s `O(log W)` over the full window.
///
/// Used by `term_set_gallop::run` for per-term column searches. Empirically
/// 1.65–3.50× faster than `binary_search_sorted` on the term-set forward-
/// cursor pattern; the win is dominated by cache locality on the early
/// probes (`lo+1`, `lo+3`, `lo+7` land in the same cache line as `lo`)
/// rather than the asymptotic `log d` vs `log W` difference. Whether
/// galloping wins for other access patterns (e.g. `RangeQuery`'s sorted
/// path, which does two unrelated searches at arbitrary bounds) is a
/// separate question and would need its own bench gate.
///
/// Falls back to `binary_search_sorted` for tiny windows (`hi - lo < 16`)
/// where the per-step probe overhead dominates.
pub(crate) fn gallop_search_sorted(
    column: &Column<u64>,
    lo: u32,
    hi: u32,
    target: u64,
    order: Order,
    strict: bool,
) -> u32 {
    // Small-window fallback. With < 16 elements binary search does at most
    // 4 comparisons; galloping pays one column read + branch per probe and
    // can easily exceed that on tiny ranges.
    if hi.saturating_sub(lo) < 16 {
        return binary_search_sorted(column, lo, hi, target, order, strict);
    }

    // Phase 1 — exponential probe. Maintain `prev` (last index known to be in
    // the go_right region) and double `step` each iteration. Stop when a
    // probe lands on a not-go_right value (the bracket is `[prev, probe]`),
    // or when probe reaches `hi - 1` while still in the go_right region (in
    // which case the answer is `hi`).
    let mut prev = lo;
    let mut step: u32 = 1;
    loop {
        let probe = lo.saturating_add(step).min(hi - 1);
        // Safe: `probe < hi` and the caller guarantees `[lo, hi)` is non-NULL.
        let val = column
            .first(probe)
            .expect("doc in non-NULL range has no value");
        if !go_right(val, target, order, strict) {
            // Bracket found: answer is in [prev, probe]. Phase 2 binary-searches
            // the half-open range [prev, probe + 1).
            return binary_search_sorted(column, prev, probe + 1, target, order, strict);
        }
        if probe == hi - 1 {
            // Reached the end of the window in the go_right region: answer is
            // hi (matches binary_search_sorted's contract for "predicate true
            // everywhere in [lo, hi)").
            return hi;
        }
        prev = probe;
        step = step.saturating_mul(2);
    }
}

#[cfg(test)]
mod gallop_tests {
    //! Hand-written coverage of `gallop_search_sorted` against
    //! `binary_search_sorted` on the same fixtures. The randomized stress
    //! test in `tests/gallop_search_stress.rs` exercises the helper against
    //! 1000s of random inputs; the cases here pin specific edge-cases so a
    //! regression has a clear failure point.
    use super::*;
    use crate::schema::{NumericOptions, SchemaBuilder};
    use crate::{Index, IndexSettings, IndexSortByField};

    fn build_sorted_column(values: &[u64], order: Order) -> Column<u64> {
        let mut sb = SchemaBuilder::new();
        let f = sb.add_u64_field("v", NumericOptions::default().set_fast().set_indexed());
        let schema = sb.build();
        let index = Index::builder()
            .schema(schema)
            .settings(IndexSettings {
                sort_by_field: Some(IndexSortByField {
                    field: "v".to_string(),
                    order,
                }),
                ..Default::default()
            })
            .create_in_ram()
            .unwrap();
        let mut writer = index.writer_for_tests().unwrap();
        for &v in values {
            writer.add_document(doc!(f => v)).unwrap();
        }
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let segment = reader.searcher().segment_reader(0).clone();
        let (column, _) = segment
            .fast_fields()
            .u64_lenient_for_type(None, "v")
            .unwrap()
            .unwrap();
        column
    }

    /// Helper: assert gallop and binary search agree on this input.
    #[track_caller]
    fn assert_agrees(
        column: &Column<u64>,
        lo: u32,
        hi: u32,
        target: u64,
        order: Order,
        strict: bool,
    ) {
        let bin = binary_search_sorted(column, lo, hi, target, order, strict);
        let gal = gallop_search_sorted(column, lo, hi, target, order, strict);
        assert_eq!(
            gal, bin,
            "gallop({lo},{hi},target={target},order={order:?},strict={strict}) \
             returned {gal} but binary returned {bin}",
        );
    }

    #[test]
    fn asc_target_near_lo() {
        // Sorted ASC: 1..=64. Target=2 sits near lo.
        let vals: Vec<u64> = (1..=64).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        assert_agrees(&col, 0, 64, 2, Order::Asc, false);
        assert_agrees(&col, 0, 64, 2, Order::Asc, true);
    }

    #[test]
    fn asc_target_near_hi() {
        let vals: Vec<u64> = (1..=64).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        assert_agrees(&col, 0, 64, 63, Order::Asc, false);
        assert_agrees(&col, 0, 64, 63, Order::Asc, true);
    }

    #[test]
    fn asc_target_at_exact_lo_value() {
        let vals: Vec<u64> = (1..=64).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        // target=1 is the lo value. strict=false → 0 (the doc at lo).
        // strict=true → 1 (first doc strictly past 1).
        assert_agrees(&col, 0, 64, 1, Order::Asc, false);
        assert_agrees(&col, 0, 64, 1, Order::Asc, true);
    }

    #[test]
    fn asc_target_at_exact_hi_value() {
        let vals: Vec<u64> = (1..=64).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        // target=64 is the last value. strict=false → 63. strict=true → 64.
        assert_agrees(&col, 0, 64, 64, Order::Asc, false);
        assert_agrees(&col, 0, 64, 64, Order::Asc, true);
    }

    #[test]
    fn asc_target_between_elements() {
        // Skip even values; target is odd. Insertion point should be the
        // index of the next even-valued doc in both predicates.
        let vals: Vec<u64> = (0..64).map(|i| 2 * i).collect(); // 0, 2, 4, …, 126
        let col = build_sorted_column(&vals, Order::Asc);
        // target=11 sits between 10 (at index 5) and 12 (at index 6).
        // strict=false → 6. strict=true → 6.
        assert_agrees(&col, 0, 64, 11, Order::Asc, false);
        assert_agrees(&col, 0, 64, 11, Order::Asc, true);
    }

    #[test]
    fn asc_target_past_window() {
        let vals: Vec<u64> = (1..=64).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        assert_agrees(&col, 0, 64, 9999, Order::Asc, false);
        assert_agrees(&col, 0, 64, 9999, Order::Asc, true);
    }

    #[test]
    fn asc_target_before_window() {
        let vals: Vec<u64> = (1..=64).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        // target=0 < every value: both predicates return lo.
        assert_agrees(&col, 0, 64, 0, Order::Asc, false);
        assert_agrees(&col, 0, 64, 0, Order::Asc, true);
    }

    #[test]
    fn desc_basic() {
        let vals: Vec<u64> = (1..=64).rev().collect(); // 64, 63, …, 1
        let col = build_sorted_column(&vals, Order::Desc);
        for &target in &[64, 50, 32, 10, 1, 0, 99] {
            assert_agrees(&col, 0, 64, target, Order::Desc, false);
            assert_agrees(&col, 0, 64, target, Order::Desc, true);
        }
    }

    #[test]
    fn empty_window() {
        let vals: Vec<u64> = (1..=16).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        // lo == hi: binary_search_sorted returns lo. Gallop falls back to
        // binary_search_sorted via the small-window guard.
        assert_agrees(&col, 5, 5, 7, Order::Asc, false);
        assert_agrees(&col, 5, 5, 7, Order::Asc, true);
    }

    #[test]
    fn single_element_window() {
        let vals: Vec<u64> = (1..=16).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        // [3, 4): one element at index 3 with value 4.
        assert_agrees(&col, 3, 4, 3, Order::Asc, false);
        assert_agrees(&col, 3, 4, 4, Order::Asc, false);
        assert_agrees(&col, 3, 4, 5, Order::Asc, false);
        assert_agrees(&col, 3, 4, 4, Order::Asc, true);
    }

    #[test]
    fn fallback_boundary_window_15_vs_16() {
        // Window of size 15 → falls back to binary_search_sorted.
        // Window of size 16 → exercises the gallop loop. Both must agree
        // with the binary-search ground truth.
        let vals: Vec<u64> = (0..32).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        for hi in &[15u32, 16, 17, 32] {
            for &target in &[0u64, 5, 10, 14, 15, 16, 31, 99] {
                assert_agrees(&col, 0, *hi, target, Order::Asc, false);
                assert_agrees(&col, 0, *hi, target, Order::Asc, true);
            }
        }
    }

    #[test]
    fn large_window_multiple_gallop_steps() {
        // 1000 elements forces phase 1 to execute multiple doublings before
        // bracketing. Spot-check several targets.
        let vals: Vec<u64> = (0..1000).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        for &target in &[0u64, 1, 2, 16, 17, 100, 500, 800, 999, 1000, 9999] {
            assert_agrees(&col, 0, 1000, target, Order::Asc, false);
            assert_agrees(&col, 0, 1000, target, Order::Asc, true);
        }
    }

    /// Randomized differential: drive both helpers with the same inputs
    /// across many random configurations and assert agreement.
    /// Complements the hand-written cases above — they pin specific
    /// edge-cases; this one catches regressions in unanticipated regions
    /// of the (lo, hi, target, order, strict) input space.
    #[test]
    fn randomized_differential_against_binary_search() {
        use rand::prelude::*;
        use rand::rngs::StdRng;

        // Build several sorted corpora of varying size and value spread.
        let configs: &[(u32, u64, Order)] = &[
            (32, 100, Order::Asc),
            (256, 500, Order::Asc),
            (1024, 1000, Order::Asc),
            (4096, 2000, Order::Asc),
            (256, 500, Order::Desc),
            (1024, 1000, Order::Desc),
        ];

        for &(n, spread, order) in configs {
            // Generate a sorted column.
            let mut rng = StdRng::seed_from_u64(u64::from(n).wrapping_mul(spread));
            let mut values: Vec<u64> = (0..n).map(|_| rng.random_range(0..spread)).collect();
            // Index sorts internally; just feed the unsorted vec.
            let column = build_sorted_column(&values, order);

            // Sort our local copy so we can pick a representative target.
            match order {
                Order::Asc => values.sort_unstable(),
                Order::Desc => values.sort_unstable_by(|a, b| b.cmp(a)),
            }

            // Drive the helpers with random (lo, hi, target, strict) tuples.
            for _ in 0..200 {
                let mut lo = rng.random_range(0..n);
                let mut hi = rng.random_range(0..=n);
                if lo > hi {
                    std::mem::swap(&mut lo, &mut hi);
                }
                // Mix of in-range, out-of-range, and boundary targets.
                let target: u64 = match rng.random_range(0..5) {
                    0 => 0,                              // before
                    1 => spread.saturating_add(100),     // after
                    2 => values[rng.random_range(0..values.len())], // present
                    _ => rng.random_range(0..spread),    // probably present
                };
                for &strict in &[false, true] {
                    let bin = binary_search_sorted(&column, lo, hi, target, order, strict);
                    let gal = gallop_search_sorted(&column, lo, hi, target, order, strict);
                    assert_eq!(
                        gal, bin,
                        "differential mismatch: n={n} spread={spread} order={order:?} \
                         lo={lo} hi={hi} target={target} strict={strict}",
                    );
                }
            }
        }
    }

    #[test]
    fn shrinking_window_lo_advances() {
        // Mirrors term_set_gallop's usage: successive calls with a shrinking
        // window (lo advances forward). Galloping must work correctly when
        // lo > 0 and the answer is close to the current lo.
        let vals: Vec<u64> = (0..1000).collect();
        let col = build_sorted_column(&vals, Order::Asc);
        // Successive [lo, hi) ranges: [0, 1000), [100, 1000), [500, 1000), …
        for &lo in &[0u32, 100, 500, 800, 950] {
            for &target in &[lo as u64, lo as u64 + 1, lo as u64 + 50, 999] {
                assert_agrees(&col, lo, 1000, target, Order::Asc, false);
                assert_agrees(&col, lo, 1000, target, Order::Asc, true);
            }
        }
    }
}
