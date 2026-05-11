//! Microbenchmarks for `FastFieldTermSetQuery` (paradedb/paradedb#4895).
//!
//! # Tiers
//!
//! Three tiers driven by the `TERM_SET_BENCH_TIER` environment variable:
//!
//!   - `smoke` (default, target < 60s wall-clock): N ∈ {1M}, K ∈ {100, 10_000}, two corpus kinds ×
//!     two sort orders × four strategies plus the multi-column AND-intersection panel. Catches
//!     regressions; not for threshold derivation.
//!   - `full` (manual, ~30min): N ∈ {1M, 10M, 50M}, K ∈ {10, 100, 1_000, 10_000, 100_000}. 360-cell
//!     matrix used to derive `TermSetStrategyConfig::default()` densities.
//!   - `threshold` (~1min): targeted LowFk panel through K/N ∈ [0.002, 0.01] for fine-grained
//!     crossover characterization between the full tier's 10× geometric K spacing.
//!
//! # Corpus shapes
//!
//! The bench varies `D` (average documents per distinct value) across three
//! shapes that span the customer workloads we care about:
//!
//!   - `PrimaryKey` (`D = 1`): `value_for_doc(d) = d`, so every doc has a unique value. `distinct =
//!     N`. Models hash-join build sides on unique keys (UUIDs, surrogate IDs).
//!   - `LowFk` (`D ≈ 100`): `value_for_doc(d) = d / 100`, so each value appears in ~100 contiguous
//!     docs after sorting. `distinct = N/100`. Models foreign-key joins on moderate-cardinality
//!     columns — the typical paradedb hash-join pattern.
//!   - `HighFk` (`D ≈ 100_000`): `value_for_doc(d) = d / 100_000`. `distinct = N/100_000`. Models
//!     very-low-cardinality columns (status enums, region codes). Only present in the full tier at
//!     `N ≥ 10M` where there are enough distinct values to sample from.
//!
//! D shape matters for strategy choice: with `LowFk`, gallop emits ~D docs
//! per term through `RangeUnionDocSet`, which adds linear-equivalent cost
//! on top of the per-term search. PK doesn't pay that emission cost
//! (single-doc ranges). The `gallop_max_density` default is tuned to
//! LowFk because that's the dominant customer shape; PK queries with
//! K/N just above the threshold underutilize gallop slightly as a
//! tradeoff.
//!
//! # Strategy mapping
//!
//!   - `Gallop`: planner forced via `gallop_max_density = 1.0` so any K < N qualifies.
//!   - `Linear`: planner forced to terminal `LinearScan` via `gallop_enabled = false` + zero
//!     densities.
//!   - `BitsetFromPostings` (real, exercised as `bitset_from_postings_real`): planner forced via
//!     `bitset_max_density = 1.0`. This is the production strategy as of Phase 5d.
//!   - `DirectBitset`: bench-only verbatim copy of the production `bitset_from_postings_scorer`,
//!     kept for comparison against pre-5d numbers.
//!   - `PostingDirect`: synthetic baseline — `BooleanQuery` of `TermQuery::Should` over the same
//!     terms. Exercises `BufferedUnionScorer`. Retained to characterize the union scaling cost.
//!   - `BitsetFromPostings` (synthetic, exercised as `bitset_from_postings`): synthetic baseline —
//!     `BooleanQuery::Should` union materialized into a `BitSet` via a custom collector. Retained
//!     to characterize bitset construction + iteration cost on top of the union scorer.
//!
//! # Timing boundaries
//!
//! Inside the timed closure (work `searcher.search` does end-to-end per call):
//! `Term::from_field_u64` for each term; `FastFieldTermSetQuery::new` +
//! `with_strategy_config`; the inner `query.weight()` build; per-segment
//! `weight.scorer()` build (which runs `select_strategy`); the collector walk.
//! Multi-column `and_intersect` cells additionally pay one `BooleanQuery::new`
//! per iteration.
//!
//! Outside the timed closure (paid once per cell as setup): corpus build
//! (schema, index, writer, all `add_document` calls, commit), reader and
//! `Searcher` instantiation, and the deterministic `sample_terms` shuffle.
//! Only the raw `Vec<u64>` of sampled values is captured into the closure;
//! `Term::from_field_u64` runs per iteration.
//!
//! For cells with K ≥ ~1_000 the per-iteration construction cost is a
//! small fraction of total work (≤ ~5–10%), so the reported throughput
//! reflects scoring-loop performance. For very small K (10, 100) the
//! construction overhead is a larger fraction and binggan's
//! input-size-divided-by-time formula reaches artifact territory when
//! iterations finish in microseconds — relative ratios between strategies
//! at the same cell remain meaningful but absolute throughput numbers
//! below ~10ms are not reliable.
//!
//! # Captured outputs
//!
//! Captured outputs from all tiers live in `benches/term_set_queries-captures/`.
//! The full and threshold tiers in particular are persisted as
//! `full-tier.txt` and `threshold-tier.txt` in that directory. Re-run with
//!
//! ```text
//! TERM_SET_BENCH_TIER=full      cargo bench --bench term_set_queries 2>&1 \
//!     | tee benches/term_set_queries-captures/full-tier.txt
//! TERM_SET_BENCH_TIER=threshold cargo bench --bench term_set_queries 2>&1 \
//!     | tee benches/term_set_queries-captures/threshold-tier.txt
//! ```
//!
//! when the gallop algorithm or `TermSetStrategyConfig::default()`
//! changes meaningfully. Comment-only and test-only changes don't
//! warrant a refresh.

use binggan::{black_box, BenchRunner};
use common::BitSet;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::{Collector, Count, DocSetCollector, SegmentCollector};
use tantivy::query::{
    BitSetDocSet, BooleanQuery, ConstScorer, EmptyScorer, EnableScoring, Explanation,
    FastFieldTermSetQuery, Occur, Query, Scorer, TermQuery, TermSetQuery, TermSetStrategyConfig,
    Weight,
};
use tantivy::schema::{Field, IndexRecordOption, NumericOptions, SchemaBuilder};
use tantivy::{
    doc, DocId, DocSet, Index, IndexSettings, IndexSortByField, Order, ReloadPolicy, Score,
    Searcher, SegmentOrdinal, SegmentReader, TantivyError, Term,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CorpusKind {
    /// D = 1, every value unique.
    PrimaryKey,
    /// D ≈ 100, foreign-key shape.
    LowFk,
    /// D ≈ 100_000.
    HighFk,
}

impl CorpusKind {
    fn label(self) -> &'static str {
        match self {
            CorpusKind::PrimaryKey => "pk",
            CorpusKind::LowFk => "lowfk",
            CorpusKind::HighFk => "highfk",
        }
    }

    fn distinct_count(self, n: u64) -> u64 {
        match self {
            CorpusKind::PrimaryKey => n,
            CorpusKind::LowFk => n.div_ceil(100),
            CorpusKind::HighFk => n.div_ceil(100_000).max(1),
        }
    }

    fn value_for_doc(self, doc_id: u64) -> u64 {
        match self {
            CorpusKind::PrimaryKey => doc_id,
            CorpusKind::LowFk => doc_id / 100,
            CorpusKind::HighFk => doc_id / 100_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sort {
    Asc,
    None,
}

impl Sort {
    fn label(self) -> &'static str {
        match self {
            Sort::Asc => "asc",
            Sort::None => "unsorted",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Strat {
    Gallop,
    Linear,
    PostingDirect,
    BitsetFromPostings,
    /// Real planner-driven `BitsetFromPostings` strategy. Forced via
    /// `bitset_max_density = 1.0` so the dispatch always picks it on
    /// indexed numeric fast fields. The other `BitsetFromPostings`
    /// variant is the synthetic posting-union-into-bitset baseline that
    /// retains `BufferedUnionScorer`; this one exercises the real
    /// strategy that bypasses the union scorer entirely.
    BitsetFromPostingsReal,
    /// Phase 5c fourth variant: K independent `TermDictionary::get(key)`
    /// lookups (no streaming automaton) + OR each posting list into one
    /// `BitSet`. Bench-only — lives in this file, not in production code.
    /// Measures whether the streaming dict-walk in `bitset_real` is the
    /// avoidable overhead on low-D columns, and whether bitset-OR is a
    /// strict improvement over `BufferedUnionScorer` at our K range.
    DirectBitset,
}

impl Strat {
    fn label(self) -> &'static str {
        match self {
            Strat::Gallop => "gallop",
            Strat::Linear => "linear",
            Strat::PostingDirect => "posting_direct",
            Strat::BitsetFromPostings => "bitset_from_postings",
            Strat::BitsetFromPostingsReal => "bitset_from_postings_real",
            Strat::DirectBitset => "direct_bitset",
        }
    }
}

fn applicable(sort: Sort, strat: Strat) -> bool {
    !(sort == Sort::None && strat == Strat::Gallop)
}

fn bench_tier() -> &'static str {
    match std::env::var("TERM_SET_BENCH_TIER").as_deref() {
        Ok("full") => "full",
        Ok("threshold") => "threshold",
        Ok("phase5b") => "phase5b",
        Ok("profile5b") => "profile5b",
        Ok("phase5c20m") => "phase5c20m",
        Ok("phase5e_calib") => "phase5e_calib",
        Ok("phase5e_validate") => "phase5e_validate",
        Ok("phase5h_smallK") => "phase5h_smallK",
        Ok("profile5e") => "profile5e",
        _ => "smoke",
    }
}

fn build_corpus(n: u64, kind: CorpusKind, sort: Sort) -> (Searcher, tantivy::schema::Field) {
    let mut sb = SchemaBuilder::new();
    let field = sb.add_u64_field("fk", NumericOptions::default().set_fast().set_indexed());
    let schema = sb.build();
    let mut builder = Index::builder().schema(schema);
    if let Sort::Asc = sort {
        builder = builder.settings(IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "fk".to_string(),
                order: Order::Asc,
            }),
            ..Default::default()
        });
    }
    let index = builder.create_in_ram().unwrap();

    {
        let writer_mem = (200_000_000u64).max(n * 32);
        let mut writer = index
            .writer_with_num_threads(1, writer_mem as usize)
            .unwrap();
        for d in 0..n {
            writer
                .add_document(doc!(field => kind.value_for_doc(d)))
                .unwrap();
        }
        writer.commit().unwrap();
    }
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    (reader.searcher(), field)
}

fn sample_terms(distinct: u64, k: usize, seed: u64) -> Vec<u64> {
    let k = k.min(distinct as usize);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut chosen: Vec<u64> = (0..distinct).collect();
    chosen.shuffle(&mut rng);
    chosen.truncate(k);
    chosen
}

fn cfg_force_gallop() -> TermSetStrategyConfig {
    TermSetStrategyConfig {
        gallop_enabled: true,
        // Strict less-than: K/N < 1.0 admits any K < N.
        gallop_max_density: 1.0,
        // The other thresholds don't matter for sorted+small-K cases —
        // once gallop is taken, the sort-agnostic branch isn't reached.
        ..TermSetStrategyConfig::default()
    }
}

fn cfg_force_bitset_real() -> TermSetStrategyConfig {
    TermSetStrategyConfig {
        gallop_enabled: false,
        gallop_max_density: 0.0,
        bitset_max_density: 1.0,
        subsequent_bitset_max_density: 1.0,
        strategy_sink: None,
    }
}

fn cfg_force_linear() -> TermSetStrategyConfig {
    TermSetStrategyConfig {
        gallop_enabled: false,
        gallop_max_density: 0.0,
        // Strict less-than: nothing is < 0.0, so the sort-agnostic bitset
        // arms are rejected and we land on the LinearScan terminal.
        bitset_max_density: 0.0,
        subsequent_bitset_max_density: 0.0,
        strategy_sink: None,
    }
}

/// Run the planner-driven path with the given config. `K` distinct terms
/// from `[0, distinct_count)` are emitted as `Term::from_field_u64`.
fn run_planner_path(
    searcher: &Searcher,
    field: tantivy::schema::Field,
    terms: &[u64],
    cfg: TermSetStrategyConfig,
) -> usize {
    let q = FastFieldTermSetQuery::new(terms.iter().map(|v| Term::from_field_u64(field, *v)))
        .with_strategy_config(cfg);
    searcher.search(&q, &Count).unwrap()
}

/// Synthetic Strategy 4: posting-list union via `BooleanQuery` of
/// `TermQuery::Should`. Each `TermQuery` walks its posting list; the union
/// scorer interleaves them. No bitset materialization.
fn run_posting_direct_synthetic(
    searcher: &Searcher,
    field: tantivy::schema::Field,
    terms: &[u64],
) -> usize {
    let subqueries: Vec<(Occur, Box<dyn Query>)> = terms
        .iter()
        .map(|&v| {
            (
                Occur::Should,
                Box::new(TermQuery::new(
                    Term::from_field_u64(field, v),
                    IndexRecordOption::Basic,
                )) as Box<dyn Query>,
            )
        })
        .collect();
    let q = BooleanQuery::new(subqueries);
    searcher.search(&q, &Count).unwrap()
}

/// Synthetic Strategy 3: posting-list union into a `BitSet`, then iterate
/// the bitset. Captures bitset construction + iteration cost on top of
/// posting-list iteration. We use a custom `Collector` to write each
/// matching DocId into a per-segment `BitSet`, then merge by iteration.
fn run_bitset_from_postings_synthetic(
    searcher: &Searcher,
    field: tantivy::schema::Field,
    terms: &[u64],
) -> usize {
    let subqueries: Vec<(Occur, Box<dyn Query>)> = terms
        .iter()
        .map(|&v| {
            (
                Occur::Should,
                Box::new(TermQuery::new(
                    Term::from_field_u64(field, v),
                    IndexRecordOption::Basic,
                )) as Box<dyn Query>,
            )
        })
        .collect();
    let q = BooleanQuery::new(subqueries);
    searcher.search(&q, &BitSetCounter).unwrap()
}

// --- Phase 5c fourth variant: direct lookups + bitset OR --------------------

/// `direct_bitset_scorer`: K independent `TermDictionary::get(key)` lookups,
/// each opening a `BlockSegmentPostings` and OR'ing its docs into a single
/// per-segment `BitSet`. No streaming automaton, no `BufferedUnionScorer`.
///
/// Bench-only — lives in this file, not in `term_set_bitset.rs`.
fn direct_bitset_scorer(
    reader: &SegmentReader,
    field: Field,
    values: &[u64],
    boost: Score,
) -> tantivy::Result<Box<dyn Scorer>> {
    if values.is_empty() || reader.max_doc() == 0 {
        return Ok(Box::new(EmptyScorer));
    }
    let inverted_index = reader.inverted_index(field)?;
    let term_dict = inverted_index.terms();
    let mut bitset = BitSet::with_max_value(reader.max_doc());
    for &v in values {
        let key_buf = v.to_be_bytes();
        let term_info = term_dict.get(&key_buf[..])?;
        let Some(term_info) = term_info else { continue };
        let mut block_postings = inverted_index
            .read_block_postings_from_terminfo(&term_info, IndexRecordOption::Basic)?;
        loop {
            let docs = block_postings.docs();
            if docs.is_empty() {
                break;
            }
            for &doc in docs {
                bitset.insert(doc);
            }
            block_postings.advance();
        }
    }
    let docset = BitSetDocSet::from(bitset);
    Ok(Box::new(ConstScorer::new(docset, boost)))
}

/// Minimal `Query` wrapper so `Searcher::search` can drive the fourth
/// variant the same way it drives the other strategies — apples-to-apples
/// on `Searcher::search` overhead.
struct DirectBitsetQuery {
    field: Field,
    values: Vec<u64>,
}

impl std::fmt::Debug for DirectBitsetQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectBitsetQuery")
            .field("field", &self.field)
            .field("values_len", &self.values.len())
            .finish()
    }
}

impl Clone for DirectBitsetQuery {
    fn clone(&self) -> Self {
        Self {
            field: self.field,
            values: self.values.clone(),
        }
    }
}

impl Query for DirectBitsetQuery {
    fn weight(&self, _enable_scoring: EnableScoring<'_>) -> tantivy::Result<Box<dyn Weight>> {
        Ok(Box::new(DirectBitsetWeight {
            field: self.field,
            values: self.values.clone(),
        }))
    }
}

struct DirectBitsetWeight {
    field: Field,
    values: Vec<u64>,
}

impl Weight for DirectBitsetWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> tantivy::Result<Box<dyn Scorer>> {
        direct_bitset_scorer(reader, self.field, &self.values, boost)
    }
    fn explain(&self, reader: &SegmentReader, doc: DocId) -> tantivy::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(TantivyError::InvalidArgument(format!(
                "doc {doc} does not match"
            )));
        }
        Ok(Explanation::new("DirectBitsetScorer", scorer.score()))
    }
}

fn run_direct_bitset(searcher: &Searcher, field: tantivy::schema::Field, terms: &[u64]) -> usize {
    let q = DirectBitsetQuery {
        field,
        values: terms.to_vec(),
    };
    searcher.search(&q, &Count).unwrap()
}

/// Drive `TermSetQuery` rather than `FastFieldTermSetQuery` so tier-1 dispatch
/// (`TermSetQuery::specialized_weight`) runs. Used by the Phase 5h cells that
/// validate the K ≤ 1024 short-circuit removal: pre-change captures show
/// `AutomatonWeight`, post-change captures show the tier-2 dispatch result.
fn run_term_set_query(searcher: &Searcher, field: tantivy::schema::Field, terms: &[u64]) -> usize {
    let q = TermSetQuery::new(terms.iter().map(|v| Term::from_field_u64(field, *v)));
    searcher.search(&q, &Count).unwrap()
}

/// Collector that, per segment, materializes matches into a `BitSet` sized
/// to the segment's `max_doc`, then walks the bitset to count.
struct BitSetCounter;

impl Collector for BitSetCounter {
    type Fruit = usize;
    type Child = BitSetSegmentCollector;

    fn for_segment(
        &self,
        _segment_local_id: SegmentOrdinal,
        segment_reader: &SegmentReader,
    ) -> tantivy::Result<Self::Child> {
        let max_doc = segment_reader.max_doc();
        Ok(BitSetSegmentCollector {
            bitset: BitSet::with_max_value(max_doc),
        })
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn merge_fruits(&self, segment_fruits: Vec<usize>) -> tantivy::Result<usize> {
        Ok(segment_fruits.into_iter().sum())
    }
}

struct BitSetSegmentCollector {
    bitset: BitSet,
}

impl SegmentCollector for BitSetSegmentCollector {
    type Fruit = usize;

    fn collect(&mut self, doc: DocId, _score: f32) {
        self.bitset.insert(doc);
    }

    fn harvest(self) -> usize {
        // Iterate the bitset to simulate the "walk bitset to emit DocIds"
        // step that Strategy 3 performs after the per-term phase.
        let mut docset: tantivy::query::BitSetDocSet = self.bitset.into();
        let mut count = 0usize;
        while docset.doc() != tantivy::TERMINATED {
            count += 1;
            docset.advance();
        }
        count
    }
}

#[allow(dead_code)]
fn _docset_collector_smoke(searcher: &Searcher) {
    // Reserved: anchor the DocSetCollector import for cells that may want
    // sorted-DocId outputs. Currently unused; left here so adding such
    // cells doesn't require import gymnastics.
    let _ = searcher.search(
        &FastFieldTermSetQuery::new(Vec::<Term>::new()),
        &DocSetCollector,
    );
}
#[allow(dead_code)]
fn _weight_smoke<W: Weight>(_w: &W) {}

fn matrix_for_tier(tier: &str) -> (Vec<u64>, Vec<usize>, Vec<CorpusKind>) {
    match tier {
        "full" => (
            vec![1_000_000, 10_000_000, 50_000_000],
            vec![10, 100, 1_000, 10_000, 100_000],
            vec![
                CorpusKind::PrimaryKey,
                CorpusKind::LowFk,
                CorpusKind::HighFk,
            ],
        ),
        // Smoke skips HighFk at N=1M because its `distinct = 10` excludes
        // every smoke K level — building its corpus would be wasted work.
        // HighFk is exercised only in the full tier at N >= 10M where larger
        // K values can apply.
        _ => (
            vec![1_000_000],
            vec![100, 10_000],
            vec![CorpusKind::PrimaryKey, CorpusKind::LowFk],
        ),
    }
}

/// Build a corpus with a parametric `D` (average docs per distinct value).
/// `value_for_doc(d_idx) = d_idx / d`, so `distinct = ceil(n / d)`. Sort is
/// always `None` for the Phase 5b matrices (the PK-shape regression case
/// shows up identically on sorted and unsorted segments — gallop is gated
/// off by `cfg_force_bitset_real` regardless).
fn build_corpus_parametric_d(n: u64, d: u64) -> (Searcher, tantivy::schema::Field) {
    let mut sb = SchemaBuilder::new();
    let field = sb.add_u64_field("fk", NumericOptions::default().set_fast().set_indexed());
    let schema = sb.build();
    let index = Index::builder().schema(schema).create_in_ram().unwrap();
    {
        let writer_mem = (200_000_000u64).max(n * 32);
        let mut writer = index
            .writer_with_num_threads(1, writer_mem as usize)
            .unwrap();
        for doc_id in 0..n {
            writer.add_document(doc!(field => doc_id / d)).unwrap();
        }
        writer.commit().unwrap();
    }
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    (reader.searcher(), field)
}

/// Sorted variant of `build_corpus_parametric_d`. `sort_by_field` is set to
/// the parametric-D column ascending, so the resulting segment has
/// `reader.sort_by_field()` matching `"fk"` — the precondition tier-2
/// dispatch checks to admit `Gallop`. Used by Phase 5h Sorted_PK cells.
fn build_corpus_parametric_d_sorted(n: u64, d: u64) -> (Searcher, tantivy::schema::Field) {
    let mut sb = SchemaBuilder::new();
    let field = sb.add_u64_field("fk", NumericOptions::default().set_fast().set_indexed());
    let schema = sb.build();
    let index = Index::builder()
        .schema(schema)
        .settings(IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "fk".to_string(),
                order: Order::Asc,
            }),
            ..Default::default()
        })
        .create_in_ram()
        .unwrap();
    {
        let writer_mem = (200_000_000u64).max(n * 32);
        let mut writer = index
            .writer_with_num_threads(1, writer_mem as usize)
            .unwrap();
        for doc_id in 0..n {
            writer.add_document(doc!(field => doc_id / d)).unwrap();
        }
        writer.commit().unwrap();
    }
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    (reader.searcher(), field)
}

/// Phase 5b measurement-only tier: K-scaling, D-scaling, N-scaling sweeps
/// to confirm (or refute) the per-term-setup-cost hypothesis. Three
/// strategies measured: `bitset_real`, `linear`, `posting_direct`.
/// All cells use the same dispatch path (`run_planner_path` with forced
/// configs); no synthetic baselines.
fn run_phase5b_tier() {
    let mut runner = BenchRunner::new();

    // --- K-scaling sweep: D=1 (PK), N=1M, K logarithmically from 500 to 100K ---
    {
        let n: u64 = 1_000_000;
        let (searcher, field) = build_corpus_parametric_d(n, 1);
        let mut group = runner.new_group();
        group.set_name(format!("phase5b k_scaling n={n} D=1 (pk)"));
        group.set_input_size(n as usize);
        let k_levels: &[usize] = &[
            500, 1_000, 2_000, 3_000, 5_000, 7_000, 10_000, 15_000, 20_000, 30_000, 50_000, 100_000,
        ];
        for &k in k_levels {
            let terms = sample_terms(n, k, 7);
            register_four_strategies(&mut group, &searcher, field, k, terms);
        }
        group.run();
    }

    // --- D-scaling sweep: K=10K, N=1M, D ∈ {1, 10, 100, 1000} ---
    // At D=1000 with N=1M, distinct = 1000 and K_clamped = 1000 (sample_terms
    // clamps K to distinct). The strategy's per-term-setup cost is driven by
    // K' (post-dedup distinct count), so the D=1000 row directly probes the
    // K=1000 cost at high-D; results should be read as "what does the bitset
    // strategy cost at K' = 1000 on a high-D column?", not as "K=10K at
    // D=1000". Documented in the report.
    {
        let n: u64 = 1_000_000;
        let k_requested: usize = 10_000;
        let d_levels: &[u64] = &[1, 10, 100, 1_000];
        for &d in d_levels {
            let (searcher, field) = build_corpus_parametric_d(n, d);
            let mut group = runner.new_group();
            group.set_name(format!("phase5b d_scaling n={n} K_req={k_requested} D={d}"));
            group.set_input_size(n as usize);
            // Honest K' shown in the cell name: sample_terms clamps to distinct.
            let distinct = n.div_ceil(d).max(1);
            let k_effective = k_requested.min(distinct as usize);
            let terms = sample_terms(distinct, k_effective, 7);
            register_four_strategies(&mut group, &searcher, field, k_effective, terms);
            group.run();
        }
    }

    // --- N-scaling sweep: vary (K, N) along the K-floor crossover curve ---
    {
        let cells: &[(usize, u64)] = &[
            (1_000, 100_000),
            (1_000, 1_000_000),
            (1_000, 10_000_000),
            (10_000, 1_000_000),
            (10_000, 10_000_000),
        ];
        for &(k, n) in cells {
            let (searcher, field) = build_corpus_parametric_d(n, 1);
            let mut group = runner.new_group();
            group.set_name(format!("phase5b n_scaling n={n} D=1 (pk)"));
            group.set_input_size(n as usize);
            let terms = sample_terms(n, k, 7);
            register_four_strategies(&mut group, &searcher, field, k, terms);
            group.run();
        }
    }
}

/// Register the four strategies under measurement (`bitset_real`,
/// `direct_bitset`, `linear`, `posting_direct`) on a binggan group. Used by
/// the Phase 5b sub-sweeps and the Phase 5c N=20M cells so a label change
/// applies uniformly.
fn register_four_strategies(
    group: &mut binggan::BenchGroup<'_, '_>,
    searcher: &Searcher,
    field: tantivy::schema::Field,
    k: usize,
    terms: Vec<u64>,
) {
    let s = searcher.clone();
    let t = terms.clone();
    group.register(format!("k={k} strat=bitset_real"), move |_| {
        black_box(run_planner_path(&s, field, &t, cfg_force_bitset_real()))
    });
    let s = searcher.clone();
    let t = terms.clone();
    group.register(format!("k={k} strat=direct_bitset"), move |_| {
        black_box(run_direct_bitset(&s, field, &t))
    });
    let s = searcher.clone();
    let t = terms.clone();
    group.register(format!("k={k} strat=linear"), move |_| {
        black_box(run_planner_path(&s, field, &t, cfg_force_linear()))
    });
    let s = searcher.clone();
    let t = terms;
    group.register(format!("k={k} strat=posting_direct"), move |_| {
        black_box(run_posting_direct_synthetic(&s, field, &t))
    });
}

/// Phase 5c N=20M tier: scale the K-sweep to production-relevant segment
/// sizes (10–100M docs) and exercise all four variants — `bitset_real`,
/// `direct_bitset`, `linear`, `posting_direct` — across three D shapes:
///   - PK_20M:    D=1,   dict_size=20M   (primary key)
///   - MedFk_20M: D=10,  dict_size=2M    (foreign key, modest cardinality)
///   - LowFk_20M: D=100, dict_size=200K  (low-cardinality FK)
fn run_phase5c20m_tier() {
    let mut runner = BenchRunner::new();
    let n: u64 = 20_000_000;

    // PK_20M (D=1) — K ∈ {1K, 10K, 100K, 500K, 1M}
    {
        let d: u64 = 1;
        let (searcher, field) = build_corpus_parametric_d(n, d);
        let mut group = runner.new_group();
        group.set_name(format!("phase5c20m n={n} D={d} (pk_20m)"));
        group.set_input_size(n as usize);
        let distinct = n.div_ceil(d).max(1);
        for &k in &[1_000usize, 10_000, 100_000, 500_000, 1_000_000] {
            let k_eff = k.min(distinct as usize);
            let terms = sample_terms(distinct, k_eff, 7);
            register_four_strategies(&mut group, &searcher, field, k_eff, terms);
        }
        group.run();
    }

    // MedFk_20M (D=10) — K ∈ {1K, 10K, 100K, 500K}
    {
        let d: u64 = 10;
        let (searcher, field) = build_corpus_parametric_d(n, d);
        let mut group = runner.new_group();
        group.set_name(format!("phase5c20m n={n} D={d} (medfk_20m)"));
        group.set_input_size(n as usize);
        let distinct = n.div_ceil(d).max(1);
        for &k in &[1_000usize, 10_000, 100_000, 500_000] {
            let k_eff = k.min(distinct as usize);
            let terms = sample_terms(distinct, k_eff, 7);
            register_four_strategies(&mut group, &searcher, field, k_eff, terms);
        }
        group.run();
    }

    // LowFk_20M (D=100) — K ∈ {100, 1K, 10K, 100K}
    {
        let d: u64 = 100;
        let (searcher, field) = build_corpus_parametric_d(n, d);
        let mut group = runner.new_group();
        group.set_name(format!("phase5c20m n={n} D={d} (lowfk_20m)"));
        group.set_input_size(n as usize);
        let distinct = n.div_ceil(d).max(1);
        for &k in &[100usize, 1_000, 10_000, 100_000] {
            let k_eff = k.min(distinct as usize);
            let terms = sample_terms(distinct, k_eff, 7);
            register_four_strategies(&mut group, &searcher, field, k_eff, terms);
        }
        group.run();
    }
}

/// Phase 5e-planner calibration: locate `K_crossover` (where direct-lookup
/// `BitsetFromPostings` starts losing to `LinearScan`) at N=50M on PK shape.
/// Measures only `bitset_real` (post-5d production) vs `linear` — the two
/// strategies the planner picks between in the regression band.
fn run_phase5e_calib_tier() {
    let mut runner = BenchRunner::new();
    let n: u64 = 50_000_000;
    let d: u64 = 1;
    let (searcher, field) = build_corpus_parametric_d(n, d);
    let mut group = runner.new_group();
    group.set_name(format!("phase5e_calib n={n} D={d} (pk_50m)"));
    group.set_input_size(n as usize);
    let distinct = n.div_ceil(d).max(1);
    for &k in &[10_000usize, 100_000, 500_000, 1_000_000, 2_000_000] {
        let k_eff = k.min(distinct as usize);
        let terms = sample_terms(distinct, k_eff, 7);
        // Only the two strategies the planner toggles between in this band.
        let s = searcher.clone();
        let t = terms.clone();
        group.register(format!("k={k_eff} strat=bitset_real"), move |_| {
            black_box(run_planner_path(&s, field, &t, cfg_force_bitset_real()))
        });
        let s = searcher.clone();
        let t = terms;
        group.register(format!("k={k_eff} strat=linear"), move |_| {
            black_box(run_planner_path(&s, field, &t, cfg_force_linear()))
        });
    }
    group.run();
}

/// Phase 5e-planner validation: re-run a focused subset of N=1M and N=20M
/// cells to verify that the tightened `bitset_max_density` correctly routes
/// the regression cells to `LinearScan` without losing wins. Exercises the
/// **default** TermSetStrategyConfig (no force-bitset / force-linear) so the
/// planner's dispatch is what's under test, not the strategy in isolation.
fn run_phase5e_validate_tier() {
    let mut runner = BenchRunner::new();

    // PK D=1 at N=1M — K spanning the win region into the regression band.
    {
        let n: u64 = 1_000_000;
        let (searcher, field) = build_corpus_parametric_d(n, 1);
        let mut group = runner.new_group();
        group.set_name(format!("phase5e_validate n={n} D=1 (pk)"));
        group.set_input_size(n as usize);
        for &k in &[500usize, 10_000, 100_000, 500_000] {
            let terms = sample_terms(n, k, 7);
            // Planner-driven (default config) — the dispatch is the test.
            let s = searcher.clone();
            let t = terms.clone();
            group.register(format!("k={k} planner=default"), move |_| {
                black_box(run_planner_path(
                    &s,
                    field,
                    &t,
                    TermSetStrategyConfig::default(),
                ))
            });
            // For comparison: forced LinearScan, to verify the planner picked
            // the cheaper of (default-dispatch, LinearScan) in each cell.
            let s = searcher.clone();
            let t = terms;
            group.register(format!("k={k} forced=linear"), move |_| {
                black_box(run_planner_path(&s, field, &t, cfg_force_linear()))
            });
        }
        group.run();
    }

    // PK_20M D=1 — covers the high-K regression band cells.
    {
        let n: u64 = 20_000_000;
        let (searcher, field) = build_corpus_parametric_d(n, 1);
        let mut group = runner.new_group();
        group.set_name(format!("phase5e_validate n={n} D=1 (pk_20m)"));
        group.set_input_size(n as usize);
        for &k in &[1_000usize, 10_000, 100_000, 500_000, 1_000_000] {
            let terms = sample_terms(n, k, 7);
            let s = searcher.clone();
            let t = terms.clone();
            group.register(format!("k={k} planner=default"), move |_| {
                black_box(run_planner_path(
                    &s,
                    field,
                    &t,
                    TermSetStrategyConfig::default(),
                ))
            });
            let s = searcher.clone();
            let t = terms;
            group.register(format!("k={k} forced=linear"), move |_| {
                black_box(run_planner_path(&s, field, &t, cfg_force_linear()))
            });
        }
        group.run();
    }

    // MedFk_20M (D=10) — covers the K=500K regression cell + a win cell.
    {
        let n: u64 = 20_000_000;
        let d: u64 = 10;
        let (searcher, field) = build_corpus_parametric_d(n, d);
        let mut group = runner.new_group();
        group.set_name(format!("phase5e_validate n={n} D={d} (medfk_20m)"));
        group.set_input_size(n as usize);
        let distinct = n.div_ceil(d).max(1);
        for &k in &[10_000usize, 100_000, 500_000] {
            let k_eff = k.min(distinct as usize);
            let terms = sample_terms(distinct, k_eff, 7);
            let s = searcher.clone();
            let t = terms.clone();
            group.register(format!("k={k_eff} planner=default"), move |_| {
                black_box(run_planner_path(
                    &s,
                    field,
                    &t,
                    TermSetStrategyConfig::default(),
                ))
            });
            let s = searcher.clone();
            let t = terms;
            group.register(format!("k={k_eff} forced=linear"), move |_| {
                black_box(run_planner_path(&s, field, &t, cfg_force_linear()))
            });
        }
        group.run();
    }
}

/// Phase 5h validation: small-K cells driven through `TermSetQuery::specialized_weight`
/// (tier-1 dispatch). Captured PRE-change to record AutomatonWeight numbers
/// and POST-change to record the new tier-2 dispatch numbers; the file
/// `term_set_queries-captures/phase5h-smallk.txt` holds the post-change
/// reading (the pre-change baseline lives in the same file via run name).
///
/// We measure `term_set_query` (drives `TermSetQuery` → tier-1 dispatch) and
/// `posting_direct` (BooleanQuery::Should over TermQuery — proxy for the
/// BufferedUnionScorer compose step that both AutomatonWeight and any
/// K-direct-lookup alternative pay). The cell-by-cell delta on the
/// `term_set_query` row tells the story.
fn run_phase5h_smallk_tier() {
    let mut runner = BenchRunner::new();
    let small_ks: &[usize] = &[10, 100, 500, 1000];

    // PK_1M (D=1, dict_size=1M).
    {
        let n: u64 = 1_000_000;
        let d: u64 = 1;
        let (searcher, field) = build_corpus_parametric_d(n, d);
        let mut group = runner.new_group();
        group.set_name(format!("phase5h_smallK n={n} D={d} (pk_1m)"));
        group.set_input_size(n as usize);
        let distinct = n.div_ceil(d).max(1);
        for &k in small_ks {
            let k_eff = k.min(distinct as usize);
            let terms = sample_terms(distinct, k_eff, 7);
            let s = searcher.clone();
            let t = terms.clone();
            group.register(format!("k={k_eff} run=term_set_query"), move |_| {
                black_box(run_term_set_query(&s, field, &t))
            });
            let s = searcher.clone();
            let t = terms;
            group.register(format!("k={k_eff} run=posting_direct"), move |_| {
                black_box(run_posting_direct_synthetic(&s, field, &t))
            });
        }
        group.run();
    }

    // PK_20M (D=1, dict_size=20M).
    {
        let n: u64 = 20_000_000;
        let d: u64 = 1;
        let (searcher, field) = build_corpus_parametric_d(n, d);
        let mut group = runner.new_group();
        group.set_name(format!("phase5h_smallK n={n} D={d} (pk_20m)"));
        group.set_input_size(n as usize);
        let distinct = n.div_ceil(d).max(1);
        for &k in small_ks {
            let k_eff = k.min(distinct as usize);
            let terms = sample_terms(distinct, k_eff, 7);
            let s = searcher.clone();
            let t = terms.clone();
            group.register(format!("k={k_eff} run=term_set_query"), move |_| {
                black_box(run_term_set_query(&s, field, &t))
            });
            let s = searcher.clone();
            let t = terms;
            group.register(format!("k={k_eff} run=posting_direct"), move |_| {
                black_box(run_posting_direct_synthetic(&s, field, &t))
            });
        }
        group.run();
    }

    // LowFk_20M (D=100, dict_size=200K).
    {
        let n: u64 = 20_000_000;
        let d: u64 = 100;
        let (searcher, field) = build_corpus_parametric_d(n, d);
        let mut group = runner.new_group();
        group.set_name(format!("phase5h_smallK n={n} D={d} (lowfk_20m)"));
        group.set_input_size(n as usize);
        let distinct = n.div_ceil(d).max(1);
        for &k in small_ks {
            let k_eff = k.min(distinct as usize);
            let terms = sample_terms(distinct, k_eff, 7);
            let s = searcher.clone();
            let t = terms.clone();
            group.register(format!("k={k_eff} run=term_set_query"), move |_| {
                black_box(run_term_set_query(&s, field, &t))
            });
            let s = searcher.clone();
            let t = terms;
            group.register(format!("k={k_eff} run=posting_direct"), move |_| {
                black_box(run_posting_direct_synthetic(&s, field, &t))
            });
        }
        group.run();
    }

    // Sorted_PK_20M (D=1, dict_size=20M, segment sorted ASC by `fk`). This is the
    // headline cell for tier-1 removal: pre-change it routed to AutomatonWeight
    // ignoring the sort; post-change it should hit Gallop via tier-2.
    {
        let n: u64 = 20_000_000;
        let d: u64 = 1;
        let (searcher, field) = build_corpus_parametric_d_sorted(n, d);
        let mut group = runner.new_group();
        group.set_name(format!("phase5h_smallK n={n} D={d} (sorted_pk_20m)"));
        group.set_input_size(n as usize);
        let distinct = n.div_ceil(d).max(1);
        for &k in small_ks {
            let k_eff = k.min(distinct as usize);
            let terms = sample_terms(distinct, k_eff, 7);
            let s = searcher.clone();
            let t = terms.clone();
            group.register(format!("k={k_eff} run=term_set_query"), move |_| {
                black_box(run_term_set_query(&s, field, &t))
            });
            let s = searcher.clone();
            let t = terms;
            group.register(format!("k={k_eff} run=posting_direct"), move |_| {
                black_box(run_posting_direct_synthetic(&s, field, &t))
            });
        }
        group.run();
    }
}

/// Phase 5b.2 profiling mode: run the regression cell (PK D=1 K=10K
/// Phase 5e-verify profiling mode: run the bench-only `direct_bitset`
/// helper (Phase 5c — algorithmically identical to the production
/// `bitset_real` post-5d but skips planner overhead) at one of two
/// regression cells, picked by `PROFILE5E_CELL`:
///
///   - `5e1` (default): PK D=1 K=100K N=1M    — smaller regression (~1.26× linear)
///   - `5e2`:           PK D=1 K=1M   N=20M   — extreme regression  (~2.07× linear)
///
/// Same tight-loop / wall-clock-budget shape as `run_profile5b_tier`.
fn run_profile5e_tier() {
    let cell = std::env::var("PROFILE5E_CELL").unwrap_or_else(|_| "5e1".to_string());
    let (n, k, label) = match cell.as_str() {
        "5e2" => (20_000_000u64, 1_000_000usize, "PK D=1 K=1M N=20M"),
        _ => (1_000_000u64, 100_000usize, "PK D=1 K=100K N=1M"),
    };
    let (searcher, field) = build_corpus_parametric_d(n, 1);
    let terms = sample_terms(n, k, 7);

    // Warm OS page cache the same way a binggan cell would see it.
    for _ in 0..3 {
        std::hint::black_box(run_direct_bitset(&searcher, field, &terms));
    }

    let budget = std::time::Duration::from_secs(
        std::env::var("PROFILE5E_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(15),
    );
    let start = std::time::Instant::now();
    let mut iters = 0u64;
    while start.elapsed() < budget {
        std::hint::black_box(run_direct_bitset(&searcher, field, &terms));
        iters += 1;
    }
    let elapsed = start.elapsed();
    let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    eprintln!(
        "profile5e[{cell}]: {label} direct_bitset | iters={iters} elapsed={:.2}s per_iter={:.2}ms",
        elapsed.as_secs_f64(),
        per_iter_ms,
    );
}

/// bitset_real on N=1M) repeatedly for a fixed wall-clock budget so
/// `samply` has a stable target to sample. Not a binggan group — just a
/// raw loop, so the profile isn't diluted by binggan's stat machinery.
fn run_profile5b_tier() {
    let n: u64 = 1_000_000;
    let k: usize = 10_000;
    let (searcher, field) = build_corpus_parametric_d(n, 1);
    let terms = sample_terms(n, k, 7);
    let cfg = cfg_force_bitset_real();

    // Warmup so the OS page cache is hot — same as a binggan cell would see.
    for _ in 0..3 {
        std::hint::black_box(run_planner_path(&searcher, field, &terms, cfg.clone()));
    }

    let budget = std::time::Duration::from_secs(
        std::env::var("PROFILE5B_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10),
    );
    let start = std::time::Instant::now();
    let mut iters = 0u64;
    while start.elapsed() < budget {
        std::hint::black_box(run_planner_path(&searcher, field, &terms, cfg.clone()));
        iters += 1;
    }
    let elapsed = start.elapsed();
    let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    eprintln!(
        "profile5b: PK D=1 K={k} N={n} bitset_real | iters={iters} elapsed={:.2}s per_iter={:.2}ms",
        elapsed.as_secs_f64(),
        per_iter_ms,
    );
}

fn main() {
    let tier = bench_tier();

    if tier == "phase5b" {
        run_phase5b_tier();
        return;
    }
    if tier == "profile5b" {
        run_profile5b_tier();
        return;
    }
    if tier == "phase5c20m" {
        run_phase5c20m_tier();
        return;
    }
    if tier == "phase5e_calib" {
        run_phase5e_calib_tier();
        return;
    }
    if tier == "phase5e_validate" {
        run_phase5e_validate_tier();
        return;
    }
    if tier == "phase5h_smallK" {
        run_phase5h_smallk_tier();
        return;
    }
    if tier == "profile5e" {
        run_profile5e_tier();
        return;
    }

    // Threshold tier: targeted measurements to close the LowFk gallop-vs-linear
    // crossover gap left by the full tier's 10x geometric K spacing. Runs only
    // a focused (LowFk, ASC, gallop+linear) panel at K values distributed
    // log-uniformly through K/N ∈ (0.001, 0.01). Bypasses the matrix loop
    // entirely; doesn't touch smoke or full tier code paths.
    if tier == "threshold" {
        run_threshold_tier();
        return;
    }

    let (n_levels, k_levels, kinds) = matrix_for_tier(tier);
    let sorts = [Sort::Asc, Sort::None];
    let strategies = [
        Strat::Gallop,
        Strat::Linear,
        Strat::PostingDirect,
        Strat::BitsetFromPostings,
        Strat::BitsetFromPostingsReal,
        Strat::DirectBitset,
    ];

    let mut runner = BenchRunner::new();

    for &n in &n_levels {
        for &kind in kinds.iter() {
            for &sort in &sorts {
                let (searcher, field) = build_corpus(n, kind, sort);
                let mut group = runner.new_group();
                group.set_name(format!("n={n} kind={} sort={}", kind.label(), sort.label()));
                group.set_input_size(n as usize);
                let distinct = kind.distinct_count(n);
                for &k in &k_levels {
                    if (k as u64) > distinct {
                        continue;
                    }
                    // Smoke trim: K=10K cells on the unsorted groups duplicate
                    // information already in the corresponding asc groups
                    // (linear/posting/bitset are sort-insensitive). Drop them
                    // in smoke to keep wall-clock under 90s after the
                    // and_intersect cells were added. Full tier still has them.
                    if tier != "full" && sort == Sort::None && k == 10_000 {
                        continue;
                    }
                    let terms = sample_terms(distinct, k, 7);
                    for &strat in &strategies {
                        if !applicable(sort, strat) {
                            continue;
                        }
                        // Force-gallop requires K < N strictly; with K = N
                        // the strict-less-than would still fail. Skip the
                        // degenerate cell so timings aren't reported as
                        // gallop misses.
                        if strat == Strat::Gallop && (k as u64) >= n {
                            continue;
                        }
                        let name = format!("k={k} strat={}", strat.label());
                        let searcher = searcher.clone();
                        let terms_v = terms.clone();
                        group.register(name, move |_| match strat {
                            Strat::Gallop => black_box(run_planner_path(
                                &searcher,
                                field,
                                &terms_v,
                                cfg_force_gallop(),
                            )),
                            Strat::Linear => black_box(run_planner_path(
                                &searcher,
                                field,
                                &terms_v,
                                cfg_force_linear(),
                            )),
                            Strat::PostingDirect => {
                                black_box(run_posting_direct_synthetic(&searcher, field, &terms_v))
                            }
                            Strat::BitsetFromPostings => black_box(
                                run_bitset_from_postings_synthetic(&searcher, field, &terms_v),
                            ),
                            Strat::BitsetFromPostingsReal => black_box(run_planner_path(
                                &searcher,
                                field,
                                &terms_v,
                                cfg_force_bitset_real(),
                            )),
                            Strat::DirectBitset => {
                                black_box(run_direct_bitset(&searcher, field, &terms_v))
                            }
                        });
                    }
                }
                group.run();
            }
        }
    }

    // Multi-column AND-intersection cells. Probes the smart-seek win on the
    // linear-scan path: column A is sorted (gallop), column B is unsorted
    // (linear) — BooleanQuery::Must of two FastFieldTermSetQueries. With the
    // trait-default seek, column B's per-`seek(target)` walk dilutes
    // column A's gallop selectivity; with the smart-seek override on
    // TermSetDocSet, column B jumps directly to each target.
    //
    // Two corpus shapes are exercised:
    //   - dense FK (D ≈ 100): gallop output is ~1500 contiguous ranges of ~100 docs each. B's seeks
    //     rarely cross range boundaries, so the smart-vs-default delta is bounded — the lower-bound
    //     case.
    //   - sparse PK (D = 1): gallop output is 1500 *isolated* DocIds spread across the segment.
    //     Every gallop emit forces B to skip a large gap, which is exactly where smart seek pays
    //     off — the upper-bound case for typical hash-join build sides.
    run_and_intersect_cells(&mut runner);
}

/// Targeted threshold-tier panel: closes the LowFk K/N ∈ (0.001, 0.01) gap
/// the full-tier matrix skipped. Eight (K, N) cells × two strategies
/// (gallop, linear), all on LowFk + sorted ASC. Reuses the existing
/// build_corpus / sample_terms / cfg_force_* primitives so per-cell
/// behavior is identical to the matrix cells — just different K values.
fn run_threshold_tier() {
    // (K, N) pairs spread log-uniformly through K/N ∈ (0.001, 0.01).
    // Five at N=1M (K/N = 0.002, 0.003, 0.005, 0.007, 0.010) and three at
    // N=10M (K/N = 0.003, 0.005, 0.007) so we can also check whether the
    // crossover is N-dependent.
    let cells: &[(usize, u64)] = &[
        (2_000, 1_000_000),
        (3_000, 1_000_000),
        (5_000, 1_000_000),
        (7_000, 1_000_000),
        (10_000, 1_000_000),
        (30_000, 10_000_000),
        (50_000, 10_000_000),
        (70_000, 10_000_000),
    ];

    // Build one corpus per N (LowFk distinct = N/100, comfortably above max
    // K=70K at N=10M).
    let mut runner = BenchRunner::new();
    for &n in &[1_000_000u64, 10_000_000u64] {
        let (searcher, field) = build_corpus(n, CorpusKind::LowFk, Sort::Asc);
        let mut group = runner.new_group();
        group.set_name(format!("threshold n={n} kind=lowfk sort=asc"));
        group.set_input_size(n as usize);

        let distinct = CorpusKind::LowFk.distinct_count(n);
        for &(k, n_cell) in cells {
            if n_cell != n {
                continue;
            }
            assert!(
                (k as u64) <= distinct,
                "k={k} exceeds LowFk distinct={distinct} at n={n}",
            );
            let terms = sample_terms(distinct, k, 7);
            for &strat in &[Strat::Gallop, Strat::Linear] {
                let s = searcher.clone();
                let terms_v = terms.clone();
                let cell_name = format!("k={k} strat={}", strat.label());
                group.register(cell_name, move |_| match strat {
                    Strat::Gallop => {
                        black_box(run_planner_path(&s, field, &terms_v, cfg_force_gallop()))
                    }
                    Strat::Linear => {
                        black_box(run_planner_path(&s, field, &terms_v, cfg_force_linear()))
                    }
                    _ => unreachable!(),
                });
            }
        }
        group.run();
    }
}

/// Build the (a sorted, b unsorted) two-column AND-intersection bench cells.
/// Extracted so the corpus-build cost is paid once per (kind) shape.
fn run_and_intersect_cells(runner: &mut BenchRunner) {
    for &(label, density_label) in &[("dense_fk", "D=100"), ("sparse_pk", "D=1")] {
        let (searcher, a, b, n, a_distinct, b_distinct) = build_and_intersect_corpus(label);

        let mut group = runner.new_group();
        group.set_name(format!("and_intersect n=1M kind={label} ({density_label})"));
        group.set_input_size(n as usize);

        // Sample terms uniformly across each column's distinct value space
        // (rather than using the contiguous prefix `0..1500`). This is the
        // load-bearing detail for the sparse_pk cell: with a = doc_id (D=1),
        // a contiguous prefix `0..1500` would map gallop's output to a SINGLE
        // contiguous DocId range [0, 1500), defeating the very thing we're
        // trying to measure (sparse, scattered gallop hits forcing many big
        // seeks into B). Random sampling spreads the 1500 hits across the
        // full segment so each forces a real ~N/K-doc gap-skip.
        let a_terms: Vec<u64> = sample_terms(a_distinct, 1500, 7);
        let b_terms: Vec<u64> = sample_terms(b_distinct, 1500, 11);

        let s = searcher.clone();
        let at = a_terms.clone();
        let bt = b_terms.clone();
        group.register("a=gallop b=linear", move |_| {
            let qa = Box::new(
                FastFieldTermSetQuery::new(at.iter().map(|v| Term::from_field_u64(a, *v)))
                    .with_strategy_config(cfg_force_gallop()),
            ) as Box<dyn Query>;
            let qb = Box::new(
                FastFieldTermSetQuery::new(bt.iter().map(|v| Term::from_field_u64(b, *v)))
                    .with_strategy_config(cfg_force_linear()),
            ) as Box<dyn Query>;
            let bq = BooleanQuery::new(vec![(Occur::Must, qa), (Occur::Must, qb)]);
            black_box(s.search(&bq, &Count).unwrap())
        });

        let s = searcher.clone();
        let at = a_terms.clone();
        let bt = b_terms.clone();
        group.register("a=linear b=linear", move |_| {
            let qa = Box::new(
                FastFieldTermSetQuery::new(at.iter().map(|v| Term::from_field_u64(a, *v)))
                    .with_strategy_config(cfg_force_linear()),
            ) as Box<dyn Query>;
            let qb = Box::new(
                FastFieldTermSetQuery::new(bt.iter().map(|v| Term::from_field_u64(b, *v)))
                    .with_strategy_config(cfg_force_linear()),
            ) as Box<dyn Query>;
            let bq = BooleanQuery::new(vec![(Occur::Must, qa), (Occur::Must, qb)]);
            black_box(s.search(&bq, &Count).unwrap())
        });

        group.run();
    }
}

fn build_and_intersect_corpus(
    kind: &str,
) -> (
    Searcher,
    tantivy::schema::Field,
    tantivy::schema::Field,
    u64,
    u64, // a_distinct
    u64, // b_distinct
) {
    let n: u64 = 1_000_000;
    let mut sb = SchemaBuilder::new();
    let a = sb.add_u64_field("a", NumericOptions::default().set_fast().set_indexed());
    let b = sb.add_u64_field("b", NumericOptions::default().set_fast().set_indexed());
    let schema = sb.build();
    let index = Index::builder()
        .schema(schema)
        .settings(IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "a".to_string(),
                order: Order::Asc,
            }),
            ..Default::default()
        })
        .create_in_ram()
        .unwrap();
    {
        let mut writer = index.writer_with_num_threads(1, 200_000_000).unwrap();
        for d in 0..n {
            // dense_fk:  a = d / 100   → ~10K distinct values, ~100 docs each (D≈100).
            // sparse_pk: a = d         → 1M distinct values, 1 doc each (D=1).
            // b is the same in both: an unsorted spread of ~9973 distinct values.
            let a_val = if kind == "sparse_pk" { d } else { d / 100 };
            writer
                .add_document(doc!(a => a_val, b => (d * 7 + 13) % 9973))
                .unwrap();
        }
        writer.commit().unwrap();
    }
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    let a_distinct = if kind == "sparse_pk" {
        n
    } else {
        n.div_ceil(100)
    };
    let b_distinct = 9973u64;
    (reader.searcher(), a, b, n, a_distinct, b_distinct)
}
