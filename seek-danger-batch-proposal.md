# Proposal: Batched Candidate Probing (`DocSet::seek_danger_batch`) and `BatchFilteredScorer`

## Motivation

`DocSet` currently covers three of the four access quadrants:

| | Drive (enumerate matches) | Probe (test externally supplied `DocId`s) |
|---|---|---|
| Scalar | `advance()`, `seek(target)` | `seek_danger(target) -> SeekDangerResult` |
| Batched | `fill_buffer(&mut [DocId; 64])`, `fill_bitset_block(base, &mut [TinySet; 16])` | missing |

`fill_bitset_block` does not cover batched probing because it evaluates every document in a contiguous `1,024`-doc window `[base, base + 1024)`. When a driver yields sparse or medium-density candidates (for example `1%..5%` of a segment, average stride `20..100` docs), evaluating `1,024` consecutive documents wastes `20..100x` work on non-candidates.

Conversely, scalar `seek_danger(target)` forces a bad tradeoff for cross-boundary or vectorized `DocSet`s whose `ProbeCost` has large fixed per-call setup (`ProbeSetupCost`):
* `PredicateColumnDocSet` (and external consumer `DocSet`s such as ParadeDB's `HeapFilterScorer` crossing into Postgres heap fetches) pay ~`200..500+ ns` of per-call batch/RPC setup on every single-doc probe.
* Speculative contiguous-window buffering inside `PredicateColumnDocSet::seek_danger` helps only on dense drivers (`gap <= 16`); on medium-stride (`20..100`) drivers it must fall back to 1-doc probes (paying full per-call setup on every match) or waste `20..100x` evaluations on unprobed neighbors.
* On partial range-partitioned segments (`Intersection(RangeDocSet, PredicateColumnDocSet)`), occasional gaps `> 16` reset adaptive windowing back to 1-doc probes.
* Existing Tantivy callsites that already hold a sorted batch of candidates (`block_maxscore_filtered`, `BlockWandIntersectionScorer`, `vector/filter.rs::filter_shortlist`, `vector/filter.rs::filter_post_clusters`) still loop calling scalar `seek_danger(doc)` one document at a time.

## Proposed API

Add one method with a default implementation (plus a mask-initialization helper) to `src/docset.rs`:

```rust
/// Initializes `mask[..len.div_ceil(64)]` with bits `0..len` set (all candidates active).
#[inline]
pub fn fill_all_candidates_mask(len: usize, mask: &mut [TinySet]) {
    let full_words = len / 64;
    let rem = len % 64;
    debug_assert!(mask.len() >= len.div_ceil(64));
    mask[..full_words].fill(TinySet::full());
    if rem > 0 {
        mask[full_words] = TinySet::range_lower(rem as u32);
    }
}

/// Intersects `mask` in place with membership of a strictly increasing slice of candidate
/// document IDs, returning a lower bound on the next matching document strictly after
/// `docs.last()` (capped at [`TERMINATED`], or `0` if `docs` is empty).
///
/// `mask` must have length `>= docs.len().div_ceil(64)` with no bits set at positions
/// `>= docs.len()`. Only candidates `i` whose bit is set in `mask` need to be tested;
/// implementations clear bit `(i % 64)` of `mask[i / 64]` for any active candidate not
/// present in this `DocSet`, and leave already-cleared bits cleared.
///
/// Follows the same monotonicity and danger-zone contract as [`DocSet::seek_danger`]:
/// `docs` must be strictly increasing (`docs[0] < docs[1] < ... < TERMINATED`) and
/// monotonically increasing relative to prior `seek_danger` / `seek_danger_batch` calls,
/// and the `DocSet` may be left in an invalid state afterwards.
fn seek_danger_batch(&mut self, docs: &[DocId], mask: &mut [TinySet]) -> DocId {
    if docs.is_empty() {
        return 0;
    }
    let last = *docs.last().unwrap();
    let words = docs.len().div_ceil(64);
    debug_assert!(mask.len() >= words);
    let mut lower_bound = 0;
    for (word_idx, word) in mask[..words].iter_mut().enumerate() {
        if word.is_empty() {
            continue;
        }
        let base = word_idx * 64;
        let word_last = docs[(base + 63).min(docs.len() - 1)];
        if word_last < lower_bound {
            *word = TinySet::EMPTY;
            continue;
        }
        let mut keep = TinySet::EMPTY;
        for bit in *word {
            let doc = docs[base + bit as usize];
            if doc < lower_bound {
                continue;
            }
            match self.seek_danger(doc) {
                SeekDangerResult::Found => {
                    keep.insert_mut(bit);
                }
                SeekDangerResult::SeekLowerBound(bound) => lower_bound = bound,
            }
        }
        *word = keep;
    }
    lower_bound.max(last.saturating_add(1)).min(TERMINATED)
}
```

Why in-place AND on `&mut [TinySet]` plus returning the skip-horizon `DocId`:
* Polarity symmetry with `fill_bitset_block`: `fill_bitset_block` is a drive operation (ORing match producers), while `seek_danger_batch` is a probe/filter operation (ANDing active candidates, analogous to Velox's `SelectivityVector`).
* Zero scratch buffers in conjunctions: `Intersection::seek_danger_batch` and `BatchFilteredScorer::seek_danger_batch` reuse the exact same `docs` and `mask` across all children, short-circuiting as soon as `mask` becomes empty, with zero `sub_docs` / `sub_pos` scratch arrays, zero stack bloat, and zero heap allocations.
* Inter-batch skip horizon (`-> DocId`): preserves `SeekLowerBound` propagation across batches. When `PredicateColumnDocSet` has narrowed `doc_bounds` (sorted segments) or reaches `doc_bounds.end`, returning `doc_bounds.start` or `TERMINATED` lets `BatchFilteredScorer` call `inner.seek(next_bound)` or stop immediately rather than draining and discarding batches from `inner`.
* Compact and stack-friendly: `[TinySet; 1]` covers 64 shortlist docs, `[TinySet; 2]` covers 128-doc WAND blocks, `[TinySet; 16]` covers 1,024-doc `BatchFilteredScorer` chunks. Callers compact parallel `(doc, score)` or `(doc, BlockAddress)` buffers via `for bit in word` (`trailing_zeros()`).
* Clean `Exclude` (AND-NOT) and `Union` (OR): `Exclude` copies `mask` (`<= 16` `u64`s) into `excl_mask`, probes `exclusion_set` only on candidates that survived `underlying`, and clears `mask[w] &= !excl_mask[w]`.

## How Callers Choose `seek_danger_batch`: Callsite vs `DocSetCost`

Caller selection splits into two rules:

1. Category A: callsites that already hold a sorted batch of candidates call `seek_danger_batch` unconditionally (purely callsite-driven).
   * `src/query/boolean_query/block_maxscore.rs`: `matches[..matches_len]` (`<= 4,096` sorted `(DocId, Score)` pairs). While `threshold == Score::MIN` (initial top-k warmup), `AfterWeakAndThreshold` probes in `64`-doc chunks so `threshold` warms up quickly instead of probing an entire `4,096`-doc warmup window; once `threshold > Score::MIN`, all remaining `final_score > threshold` survivors are probed in one `seek_danger_batch` call.
   * `src/query/boolean_query/block_wand_intersection.rs`: when the filter prefers batching (`AfterScoreBatch` / `EarlyBatch`), batches `<= 128`-doc block candidates (and `64`-doc warmup chunks before the top-k threshold initializes) into a single `seek_danger_batch` call and drains surviving `(doc, score)` pairs while honoring live `threshold` updates.
   * `src/query/boolean_query/block_wand_union.rs` (`BlockWandSingleScorer` and `BlockWandUnionScorer`): when the filter prefers batching (`AfterScoreBatch`), collects `score > inner_threshold` survivors (`BlockWandSingleScorer`: capped at `64` during `Score::MIN` warmup and `128` once a threshold is active; `BlockWandUnionScorer`: ramping `64 -> 256 -> 1024` via `BatchSizeRamp` while `threshold == Score::MIN` and capped at `128` once a threshold is active), probes once with `seek_danger_batch`, propagates `next_bound` back into the underlying scorers, and yields surviving `(doc, score)` pairs while honoring live `threshold` updates.
   * `src/vector/filter.rs`: `filter_shortlist` (already sorted `candidates: Vec<(DocId, BlockAddress)>`) and `filter_post_clusters` (chunked `DocId`s drained from the routed cluster `BitSet`).
   * Calling `seek_danger_batch` unconditionally here introduces zero extra buffering: for scalar docsets, the default loop runs the same `seek_danger` checks with `SeekLowerBound` skip-ahead under a single virtual call, while immediately vectorizing `PredicateColumnDocSet`, `CachedDictPredicateDocSet`, and future batch-capable docsets.
   * `DocSetCost` remains responsible for pipeline ordering at these callsites (whether the filter runs before or after weak terms / secondary terms / score-threshold checks).

2. Category B: streaming callsites (`Intersection`, `Exclude`) introduce buffering only when gated by `DocSetCost::probe.prefers_batching()`.
   * Buffering candidates in `Intersection` or `Exclude` disables per-doc `SeekLowerBound` leapfrog skipping from the follower back into the driver, copies `(doc, score)` into a buffer, and can overshoot `LIMIT k` on the first batch.
   * For in-memory docsets (`TermScorer`, `BitSetDocSet`, `RangeDocSet`, `TermSetDocSet`, `PhraseScorer`), `ProbeSetupCost` is `0`, so `prefers_batching()` is `false` and they remain on the untouched zero-overhead scalar `Intersection` / `Exclude`.
   * Only followers with `cost().probe.prefers_batching()` (i.e. `setup >= ProbeCost::BATCH_SETUP_THRESHOLD`) and `constant_score().is_some()` (since batch probing does not position the follower per doc for `follower.score()`) are split out into `BatchFilteredScorer`.

## `BatchFilteredScorer` Design

Added alongside `Intersection` in `src/query/intersection.rs` and constructed by `intersect_scorers` when any follower satisfies `cost().probe.prefers_batching() && constant_score().is_some()`:

1. Partitioning in `intersect_scorers`:
   * Select the lowest-`DriveCost` scorer (tie-breaking by `size_hint()`) first so a narrow batchable scorer (for example a sorted-segment `PredicateColumnDocSet` with small `doc_bounds` or a nested `BatchFilteredScorer`) can still drive when cheaper.
   * Partition the remaining followers: those matching `cost().probe.prefers_batching() && constant_score().is_some()` go into `batch_filters`; the rest stay in `scalar_scorers`.
   * Build `inner` using the existing scalar `Intersection` (preserving full leapfrog skipping among all postings, bitsets, ranges, and phrases).
   * Order `batch_filters` by `(probe.for_batch(expected_batch) / expected_batch) / (1 - selectivity)` (with `expected_batch = 256`) so lower-setup / cheaper / more-selective batch filters run first and can empty `mask` before higher-setup filters run.
2. Progressive batch refill (`64 -> 256 -> 1024`) with inter-batch skip horizon:
   * First refill pulls at most `64` candidates from `inner` (bounding `LIMIT k` overshoot), ramping to `256` on the second refill and `1,024` thereafter to amortize `ProbeSetupCost` below `1` unit/doc on full scans.
   * Stores `buf_scores: Vec<Score>` only when `inner.constant_score().is_none()`; when `inner` is constant-score, fills `buf_docs` via `inner.fill_buffer` and adds `inner_constant_score + filter_score`.
   * Initializes `mask` via `fill_all_candidates_mask`, runs all `batch_filters` in place against the same `buf_docs` and `mask` (short-circuiting on empty `mask`), and compacts `buf_docs` (and `buf_scores`) once at the end.
   * Tracks `next_inner_seek` from the maximum returned `next_bound` (initialized from the maximum initial `doc()` across `inner` and `batch_filters`): if `next_bound == TERMINATED`, stops refilling once the current batch's survivors are drained; if `next_bound > last_doc + 1`, calls `inner.seek(next_bound)` before the next refill.
3. `seek`, `seek_danger`, `seek_danger_batch`, `is_empty_in_range`, and `cost`:
   * `seek(target)`: if `target` lies within the remaining buffered survivors (`target <= last_buffered_doc`), advances `buf.idx` via binary search; otherwise drops the buffer, seeks `inner.seek(target.max(next_inner_seek))`, and refills.
   * `seek_danger(target)` and `seek_danger_batch(docs, mask)`: resolve targets strictly below the active buffered horizon against `CandidateBuffer` (so an already-positioned scorer whose `inner` has advanced ahead of `buf.docs` never misses buffered matches) and delegate targets at or above the buffered horizon directly to `inner` and `batch_filters` in place.
   * `is_empty_in_range(start, end)`: checks any active buffered range `< buf.end` first, then delegates `[start.max(buf.end), end]` to `inner` and `batch_filters`.
   * `cost()`: computes amortized batched drive cost (`inner.drive + ceil(inner_hint / B) * setup + inner_hint * per_doc`, chained by selectivity) and propagates `probe` (`inner.probe` + selectivity-scaled `batch_filters.probe`, retaining `setup`).
4. Avoiding eager upfront refill for WAND filters:
   * When `BooleanWeight` builds `filter` for `FilteredTermUnion` / `FilteredBlockMaxUnion` / `FilteredTermIntersection`, constructing via `intersect_scorers_unpositioned` (`BatchFilteredScorer::new_unpositioned`) avoids an eager 64-doc refill before WAND can prune blocks with `is_empty_in_range` or high thresholds.

## Leaf and Composite `seek_danger_batch` Implementations

1. `PredicateColumnDocSet` (`src/query/term_set_query/term_predicate.rs`):
   * Returns `TERMINATED` when `doc_bounds` is empty or `last + 1 >= doc_bounds.end`, `doc_bounds.start` when `last + 1 < doc_bounds.start`, else `last + 1`.
   * When all candidates in `docs` are active in `mask` and lie within `doc_bounds`:
     * `Cardinality::Full`: uses `column.values.get_range(u64::from(first), &mut scratch_vals)` if contiguous, else `column.values.get_vals(docs, &mut scratch_vals)`, clears `mask` and calls `predicate.filter_monotonic_u64s` directly into `mask`, then masks out values outside `value_range` if bounded.
     * `Cardinality::Optional`: calls `column.first_vals(docs, &mut scratch_opt_vals)`, packs non-null in-range `(pos, val)` pairs, calls `predicate.filter_monotonic_u64s` once, and scatters set bits into `mask`.
     * `Cardinality::Multivalued`: gathers `(pos, val)` across `column.values_for_doc(doc)`, calls `predicate.filter_monotonic_u64s` once, and sets matching `pos` bits in `mask`.
   * When `mask` is already sparse (for example after `RangeDocSet` in `Intersection`) or `doc_bounds` trims `docs`: iterates only set bits inside `doc_bounds` using `PredicateColumnDocSet`'s existing `scratch_doc_ids` / `scratch_val_docs` buffers, decodes only those active docs, runs one `predicate.filter_monotonic_u64s` call, and writes back matching bits into `mask`.
   * Zero speculative neighbor decoding; scalar `seek_danger` keeps its existing adaptive windowing for scalar callers (`BlockWandUnionScorer`).
2. `CachedDictPredicateDocSet` (`src/query/term_set_query/term_predicate.rs`):
   * Iterates only set bits in `mask`.
   * When `FullyPopulated`: checks `bytes_col.ords()` against `matched_ords` directly in memory.
   * When `PointWithFallback`: collects unique uncached ordinals across active candidates (promoting via `populate_all()` if unique misses exceed `remaining_lookups`), resolves all uncached ordinals in a single `predicate.filter_bytes` call, updates caches, and clears non-matching bits in `mask`.
3. `ConstScorer`, `BoostScorer`, `RequiredOptionalScorer`, `Box<TDocSet>`, `&mut dyn DocSet`: forward `seek_danger_batch`.
4. `Intersection`: calls `self.left.seek_danger_batch(docs, mask)`, then `self.right` and each clause in `self.others` on the same `docs` and `mask`, short-circuiting as soon as `mask` is empty, returning `max` of returned skip bounds.
5. `Exclude` / `ExclusionSet` / `BatchExcludeScorer`: adds `ExclusionSet::remove_excluded_batch(&mut self, docs: &[DocId], mask: &mut [TinySet])` (implemented on `ExcludedDocSet` and `Vec<ExcludedDocSet>`), forwards `Exclude::seek_danger_batch` (testing `exclusion_set` only on candidates that survived `underlying`), and routes `MUST_NOT` clauses with `exclusion_set.probe_cost().prefers_batching()` onto `BatchExcludeScorer`.

## Scope of Commit 2

1. `src/docset.rs`, `src/lib.rs`: add `fill_all_candidates_mask`, `DocSet::seek_danger_batch` default method, and forwarding on `&mut dyn DocSet` and `Box<TDocSet>`.
2. `src/query/term_set_query/term_predicate.rs`: implement `seek_danger_batch` on `PredicateColumnDocSet` and `CachedDictPredicateDocSet`.
3. `src/query/intersection.rs`: implement `Intersection::seek_danger_batch`, add `BatchFilteredScorer` (with `new` and `new_unpositioned`), and split `prefers_batching()` constant-score followers in `intersect_scorers`.
4. `src/query/exclude.rs`, `src/query/const_score_query.rs`, `src/query/boost_query.rs`, `src/query/reqopt_scorer.rs`: forward/compose `seek_danger_batch` and add `BatchExcludeScorer` / `BatchSizeRamp` / `CandidateBuffer`.
5. `src/query/boolean_query/boolean_weight.rs`, `block_maxscore.rs`, `block_wand_intersection.rs`, `block_wand_union.rs` (`BlockWandSingleScorer`): wire Category A callsites to `seek_danger_batch`, use `intersect_scorers_unpositioned` for WAND filters, defer fast-field `Occur::MustNot` clauses to `scorer_for_candidates_unpositioned` on selective drivers, and wire `BatchExcludeScorer` for batchable `MUST_NOT` clauses.
6. `src/vector/filter.rs`: wire `filter_shortlist` and `filter_post_clusters` to `seek_danger_batch`.
7. Tests:
   * Shared property tests in `src/query/seek_danger_tests.rs` verifying `seek_danger` and `seek_danger_batch` (under both all-active and sparse pre-filtered masks, multi-chunk monotonic probing, partially-advanced buffered scorers, and skip-horizon validity) across every registered `DocSet` / `Scorer`, plus a `seek_danger_batch_implementation_inventory` guard.
   * Parameterized/telemetry test in `src/query/term_set_query/term_predicate.rs` verifying `BatchFilteredScorer` across `Full` / `Optional` / `Multivalued` + bounded `value_range` + sorted `doc_bounds` skip horizons, progressive `64`-doc ramp on early `LIMIT`, `CachedDictPredicateDocSet` batched `filter_bytes`, WAND batched probe counts (`BlockWandSingleScorer`, `BlockWandIntersectionScorer`, `block_maxscore_filtered`), and `BatchExcludeScorer`.
