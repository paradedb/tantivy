# Proposal: Split `DocSet::cost()` into Drive Cost and Probe Cost

## Motivation

`DocSet::cost(&self) -> u64` currently models a single quantity: the estimated total work to drive (consume) an entire `DocSet` via `advance()` until `TERMINATED`, relative to driving a `TermScorer` with `size_hint()` matches (`1` unit per matching document).

`src/docset.rs` already notes this limitation as future work:

> We may want to differentiate `DocSet` costs more granular, e.g. `creation_cost`, `advance_cost`, `seek_cost` to get a good estimation what query types to choose.

In practice, query execution needs two orthogonal cost axes alongside selectivity (`s = size_hint() / segment_num_docs`):

1. `DriveCost(u64)`: total work to enumerate all matches via `advance()`. Scales with `O(s * N)` for indexed and bitset docsets (`TermScorer`, `BitSetDocSet`, `ContiguousDocSet`, `TermSetGallopDocSet`), `O(N)` for fast-field column scans (`RangeDocSet`, `TermSetDocSet`, `PredicateColumnDocSet`, `CachedDictPredicateDocSet`), and `O(co_occurrences * pos_cost)` for `PhraseScorer`.
2. `ProbeCost(u64)`: work for a single `seek_danger(target)` point probe, in the same units as one `TermScorer::advance()`. Largely independent of `size_hint()`.

Conflating these two axes causes four concrete problems:

1. Overlapping `cost()` values across vastly different `seek_danger` latencies:
   * A 50%-dense `BitSetDocSet` has `cost() = 0.5 * N` (`500K`), yet `seek_danger(doc)` is a `~1..2 ns` L2 bit test with tight `SeekLowerBound` skips.
   * A rare `PhraseScorer` can have low `cost()` (`2,000`), yet `seek_danger(doc)` decodes positions.
   * `TermSetDocSet` inherits default `cost() = size_hint() = N` even though `seek_danger(doc)` performs random fast-field decompression plus a hash-set probe (`~30..60 ns`) and returns `target + 1`.
   * External/vectorized filters (`PredicateColumnDocSet`, `CachedDictPredicateDocSet`, or consumer filters such as heap-fetching `DocSet`s) incur `~150..1,000+ ns` per single-doc `seek_danger(doc)` call and currently have to resort to sentinel values like `u64::MAX / 2` to stay last in `Intersection`.
2. `Intersection::cost()` hides expensive non-driver clauses:
   * `Intersection::cost()` returns `self.left.cost()` (the driver's drive cost). When an `Intersection` combines a cheaper driver with an expensive probe clause (e.g. `Intersection(RangeDocSet, PredicateColumnDocSet)`), the expensive probe cost is invisible to callers such as WAND or an outer `Intersection`.
   * Changing `Intersection::cost()` to `max(left, right)` or `left + right` would break driver selection, because `Intersection(rare_term[100], PredicateColumnDocSet)` genuinely is cheap to drive (`rare_term` only probes the predicate 100 times).
3. `cost() / size_hint()` does not recover probe cost:
   * `cost() / size_hint()` measures drive work per surviving match, which blows up on selective cheap filters like `Intersection(BitSet[1%], BitSet[1%])` (`ratio = 100`) ... the exact case where WAND most benefits from early filtering.
4. WAND filter placement cannot distinguish cheap bitset/posting filters from expensive fast-field/phrase/external filters:
   * In `block_maxscore_filtered`, `BlockWandUnionScorer`, `BlockWandSingleScorer`, and `BlockWandIntersectionScorer`, the scoring terms already drive the scan; `filter` is only ever called via `seek_danger(doc)` (and block-level `is_empty_in_range`).
   * Testing `filter.seek_danger(doc)` before `weak` terms (`block_maxscore_filtered`), `align_scorers` + `score > inner_threshold` (`BlockWandUnionScorer`), or `secondaries` (`BlockWandIntersectionScorer`) is a large win for cheap selective filters (`BitSetDocSet`, selective `TermScorer`), but a large regression for high-probe-cost filters (or ultra-dense filters) where posting alignment and score pruning reject `~99%` of candidates cheaply in L1.

## Proposed API

Refine `DocSet::cost()` in `src/docset.rs` rather than adding a separate boolean or side-channel trait method:

```rust
/// Total cost to consume a `DocSet` via `advance()`, in units of one `TermScorer::advance()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DriveCost(pub u64);

/// Fixed per-call setup cost of a `seek_danger` / `seek_danger_batch` invocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProbeSetupCost(pub u64);

/// Marginal per-document cost of testing membership in a `DocSet`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProbePerDocCost(pub u64);

/// Cost of probing membership: `setup + batch_len * per_doc`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProbeCost {
    pub setup: ProbeSetupCost,
    pub per_doc: ProbePerDocCost,
}

impl ProbeCost {
    pub const BITSET: Self = Self::per_doc(1);
    pub const TERM: Self = Self::per_doc(3);
    pub const FAST_FIELD: Self = Self::per_doc(5);
    pub const GALLOP: Self = Self::per_doc(8);
    pub const FAST_FIELD_LOOKUP: Self = Self::per_doc(10);
    pub const VECTORIZED_PREDICATE: Self = Self {
        setup: ProbeSetupCost(300),
        per_doc: ProbePerDocCost(15),
    };

    pub const fn scalar(self) -> u64 { ... }
    pub const fn for_batch(self, batch_len: u64) -> u64 { ... }
    pub const fn prefers_batching(self) -> bool { ... }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DocSetCost {
    pub drive: DriveCost,
    pub probe: ProbeCost,
}
```

Default implementation on `DocSet`:

```rust
fn cost(&self) -> DocSetCost {
    DocSetCost::new(DriveCost(self.size_hint() as u64), ProbeCost::TERM)
}
```

Separating `ProbeCost` into `setup` and `per_doc` lets follow-up batched candidate probing (`DocSet::seek_danger_batch`) distinguish high-per-doc in-memory docsets that leapfrog-skip (`PhraseScorer`: `setup = 0, per_doc = 10 * num_terms`) from cross-boundary vectorized predicates that benefit from buffering candidates (`PredicateColumnDocSet`: `setup = 300, per_doc = 15`).

### Concrete `DocSet::cost()` Assignments

* `BitSetDocSet`, `BitmapDocSet`, `AllScorer`, `EmptyScorer`, `ContiguousDocSet`: `drive = size_hint`, `probe = ProbeCost::BITSET`.
* `TermScorer`, `SegmentPostings`, `LoadedPostings`, `TermSetGallopDocSet`: `drive = size_hint`, `probe = ProbeCost::POSTING`.
* `RangeDocSet`: `drive = (0.8 * N) as u64`, `probe = ProbeCost::FAST_FIELD`.
* `TermSetDocSet`: `drive = 2 * N`, `probe = ProbeCost::FAST_FIELD_HASH`.
* `PhraseScorer` / `PhrasePrefixScorer`: `drive = intersection.size_hint * 10 * num_terms`, `probe = ProbeCost(10 * num_terms)`.
* `PredicateColumnDocSet` / `CachedDictPredicateDocSet`: `drive = 4 * N` (removing the `u64::MAX / 2` sentinel), `probe = ProbeCost::EXTERNAL_PREDICATE`.
* `Union` (`BufferedUnionScorer`, `SimpleUnion`, `Disjunction`): `drive = sum(child.drive)`, `probe = sum(child.probe)` (capped/saturated).
* `ConstScorer`, `BoostScorer`, `Exclude`, `ReqOptScorer`: forward underlying `DocSetCost`.
* `Intersection` and `BitmapCombination`: compose both `drive` and `probe` from children (see below).

## Combining Cost and Selectivity

Let `s_i = size_hint_i / N` be a clause's estimated selectivity (pass probability) and `1 - s_i` its rejection probability.

Because `size_hint()` is accurate for indexed/bitset docsets (`TermScorer`, `BitSetDocSet`, `ContiguousDocSet`) but hardcoded or conservative for fast-field docsets (`RangeDocSet` uses `N / 10`; `TermSetDocSet` and `PredicateColumnDocSet` return `doc_bounds.len()` up to `N`; `ExistsDocSet` returns `0`), selectivity estimates should clamp safely:
* When `0 < size_hint < N`: `s = (size_hint as f64 / N as f64).min(0.95)`.
* When `size_hint >= N` (unknown/unbounded fast-field docsets): use a neutral fallback (e.g. `s = 0.5`) so `1 - s` never divides by zero and ordering degrades cleanly to `probe`.

Two standard conjunction rules follow from `(drive, probe, s)`:

1. Probe ordering (`Intersection` non-drivers and WAND early vs late filter):
   Order independent probe clauses `P_i` by ascending `probe_i / (1 - s_i)` (expected probe cost per rejected candidate).
   * Minimizes expected probe work `probe_1 + s_1 * probe_2 + s_1 * s_2 * probe_3 + ...`.
   * Example (`Intersection`): `BitSetDocSet(s=0.5, probe=1)` (`1 / 0.5 = 2`) is probed ahead of `TermScorer(s=0.1, probe=6)` (`6 / 0.9 = 6.7`), while `RangeDocSet(s=0.1, probe=15)` (`15 / 0.9 = 16.7`) is probed ahead of a dense `TermScorer(s=0.9, probe=6)` (`6 / 0.1 = 60`).
   * Example (WAND): compare `probe_F / (1 - s_F)` against the cost of evaluating `k` deferred posting lists (`k * ProbeCost::POSTING`, where post-threshold pruning rejects `~99%`, so `1 - s_W ~ 1`). Selective `BitSetDocSet` (`~1..2`) and selective `TermScorer` (`~6..10`) run early; high-probe-cost filters (`TermSetDocSet`, `PredicateColumnDocSet`, `PhraseScorer`, heap filters) and ultra-dense filters (`s_F >= 0.9`) defer until after posting alignment and `score > threshold`.
2. Driver selection and composite `Intersection::cost()`:
   Given driver `D` followed by probe-ordered clauses `P_1, P_2, ...`, total expected intersection drive work is:
   `drive(D) = D.drive + D.size_hint * chained_probe(P_1, P_2, ...)`
   where `chained_probe(P_1, P_2, ...) = P_1.probe + s_1 * (P_2.probe + s_2 * ...)`.
   * `Intersection::cost().drive` reports `drive(self.left)` (accurately reflecting that driving a 100-doc `TermScorer` and probing `PredicateColumnDocSet` costs `100 + 100 * 150`, not `4 * N`).
   * `Intersection::cost().probe` reports `chained_probe(self.left, self.right, self.others...)`, so `Intersection(RangeDocSet, PredicateColumnDocSet)` surfaces its full probe cost to WAND without inflating drive cost.

## Callsites Impacted (Isolated Tantivy Commit)

1. `src/docset.rs`: define `DriveCost`, `ProbeCost`, `DocSetCost`, update `DocSet::cost(&self) -> DocSetCost` and forwarding impls.
2. `src/query/intersection.rs`: pick `self.left` using total intersection drive cost, order `self.right` / `self.others` by `probe / (1 - s)`, and implement composite `Intersection::cost()`.
3. `src/query/boolean_query/boolean_weight.rs`:
   * `per_occur_scorers`: use `cost().drive` for `min_driver_cost`.
   * `intersect_scorers`: generalize the hardcoded `PhraseScorer` downcast (`lines 38-45`) that guards `BitmapCombination` by comparing intersection drive cost (`driver.drive + driver.size_hint * child.probe`) against full-segment `child.drive`.
   * Pass `reader.num_docs()` (or filter selectivity) alongside `filter` into WAND constructors.
4. `src/query/bitmap_combination.rs`: sort `Intersection` children by `(cost().drive, size_hint())` and compute composite `DocSetCost`.
5. WAND scorers:
   * `src/query/boolean_query/block_wand_union.rs`: check `self.scorer.score() > inner_threshold` before `filter.seek_danger(doc)` unconditionally in `BlockWandSingleScorer`; in `BlockWandUnionScorer`, defer `filter.seek_danger(pivot_doc)` after `align_scorers` and `score > inner_threshold` when `probe_F / (1 - s_F)` exceeds posting-alignment cost.
   * `src/query/boolean_query/block_maxscore.rs`: check `scorer.score()` bound before `f.seek_danger(doc)` unconditionally in `strong.len() == 1`; defer `f.seek_danger(doc)` past the `weak`-term scoring loop (testing only `final_score > threshold` survivors) when `probe_F / (1 - s_F)` exceeds `weak.len() * ProbeCost::POSTING`.
   * `src/query/boolean_query/block_wand_intersection.rs`: defer `filter.seek_danger(candidate)` past `secondaries` intersection/scoring when `probe_F / (1 - s_F)` exceeds `secondaries.len() * ProbeCost::POSTING`.
6. Unit and property tests covering driver selection, probe ordering, composite `Intersection::cost()`, and WAND early-vs-late equivalence.
