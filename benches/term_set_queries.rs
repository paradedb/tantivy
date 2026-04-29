//! Microbenchmarks for `FastFieldTermSetQuery` (paradedb/paradedb#4895).
//!
//! Two tiers driven by the `TERM_SET_BENCH_TIER` environment variable:
//!
//!   - `smoke` (default, target < 60s wall-clock): N ∈ {1M},
//!     K ∈ {100, 10_000}, three corpus kinds × two sort orders × four
//!     strategies. ~48 cells. Catches regressions; not for threshold derivation.
//!   - `full` (manual, ~30min): N ∈ {1M, 10M, 50M},
//!     K ∈ {10, 100, 1_000, 10_000, 100_000}. 360-cell matrix used to derive
//!     `TermSetStrategyConfig::default()` densities (Step 7 of the execution
//!     brief).
//!
//! Strategy mapping (design.md §4):
//!
//!   - `Gallop`: planner forced via `gallop_max_density = 1.0` so any K < N qualifies.
//!   - `Linear`: planner forced to terminal `LinearScan` via
//!     `gallop_enabled = false` + zero densities.
//!   - `PostingDirect`: synthetic — `BooleanQuery` of `TermQuery::Should` over
//!     the same terms. Strategy 4 in production isn't implemented yet; this
//!     gives a representative posting-list-union measurement.
//!   - `BitsetFromPostings`: synthetic — execute the same posting-union but
//!     materialize matched DocIds into a `BitSet` and iterate. Strategy 3 in
//!     production isn't implemented yet; this captures bitset construction +
//!     iteration cost on top of posting-list iteration.

use binggan::{black_box, BenchRunner};
use common::BitSet;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::{Collector, Count, DocSetCollector, SegmentCollector};
use tantivy::query::{
    BooleanQuery, FastFieldTermSetQuery, Occur, Query, TermQuery, TermSetStrategyConfig,
};
use tantivy::schema::{IndexRecordOption, NumericOptions, SchemaBuilder};
use tantivy::query::Weight;
use tantivy::{
    doc, DocId, DocSet, Index, IndexSettings, IndexSortByField, Order, ReloadPolicy, Searcher,
    SegmentOrdinal, SegmentReader, Term,
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
}

impl Strat {
    fn label(self) -> &'static str {
        match self {
            Strat::Gallop => "gallop",
            Strat::Linear => "linear",
            Strat::PostingDirect => "posting_direct",
            Strat::BitsetFromPostings => "bitset_from_postings",
        }
    }
}

fn applicable(sort: Sort, strat: Strat) -> bool {
    !(sort == Sort::None && strat == Strat::Gallop)
}

fn bench_tier() -> &'static str {
    match std::env::var("TERM_SET_BENCH_TIER").as_deref() {
        Ok("full") => "full",
        _ => "smoke",
    }
}

fn build_corpus(
    n: u64,
    kind: CorpusKind,
    sort: Sort,
) -> (Searcher, tantivy::schema::Field) {
    let mut sb = SchemaBuilder::new();
    let field = sb.add_u64_field(
        "fk",
        NumericOptions::default().set_fast().set_indexed(),
    );
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
        // The other thresholds don't matter for sorted+small-K cases — once
        // gallop is taken, Step 2 isn't reached.
        ..TermSetStrategyConfig::default()
    }
}

fn cfg_force_linear() -> TermSetStrategyConfig {
    TermSetStrategyConfig {
        gallop_enabled: false,
        gallop_max_density: 0.0,
        // Strict less-than: nothing is < 0.0, so Step 2's posting / bitset
        // arms are rejected and we land on the LinearScan terminal.
        posting_max_density: 0.0,
        bitset_max_density: 0.0,
        hash_probe_max_density: 0.0,
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
    let _ = searcher.search(&FastFieldTermSetQuery::new(Vec::<Term>::new()), &DocSetCollector);
}
#[allow(dead_code)]
fn _weight_smoke<W: Weight>(_w: &W) {}

fn matrix_for_tier(tier: &str) -> (Vec<u64>, Vec<usize>, Vec<CorpusKind>) {
    match tier {
        "full" => (
            vec![1_000_000, 10_000_000, 50_000_000],
            vec![10, 100, 1_000, 10_000, 100_000],
            vec![CorpusKind::PrimaryKey, CorpusKind::LowFk, CorpusKind::HighFk],
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

fn main() {
    let tier = bench_tier();
    let (n_levels, k_levels, kinds) = matrix_for_tier(tier);
    let sorts = [Sort::Asc, Sort::None];
    let strategies = [
        Strat::Gallop,
        Strat::Linear,
        Strat::PostingDirect,
        Strat::BitsetFromPostings,
    ];

    let mut runner = BenchRunner::new();

    for &n in &n_levels {
        for &kind in kinds.iter() {
            for &sort in &sorts {
                let (searcher, field) = build_corpus(n, kind, sort);
                let mut group = runner.new_group();
                group.set_name(format!(
                    "n={n} kind={} sort={}",
                    kind.label(),
                    sort.label()
                ));
                group.set_input_size(n as usize);
                let distinct = kind.distinct_count(n);
                for &k in &k_levels {
                    if (k as u64) > distinct {
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
                            Strat::PostingDirect => black_box(run_posting_direct_synthetic(
                                &searcher, field, &terms_v,
                            )),
                            Strat::BitsetFromPostings => black_box(
                                run_bitset_from_postings_synthetic(&searcher, field, &terms_v),
                            ),
                        });
                    }
                }
                group.run();
            }
        }
    }
}
