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
