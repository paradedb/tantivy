//! Microbenchmarks for `FastFieldTermSetQuery`.
//!
//! Step 0 baseline (paradedb/paradedb#4895): four cells at N=1M, K=1000 covering
//! the cross product of {primary key, foreign key} × {sorted ASC, unsorted}, all
//! against the current linear strategy. The sorted-FK cell is the closest analogue
//! to the DemandScience query whose runtime motivates the gallop optimization.
//!
//! Future steps (§6, §9 of design.md / implementation.md) expand this into the
//! full N × K × CorpusKind × Sort × Strategy matrix with smoke / full tiers.

use binggan::{black_box, BenchRunner};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::Count;
use tantivy::query::FastFieldTermSetQuery;
use tantivy::schema::{NumericOptions, SchemaBuilder};
use tantivy::{
    doc, Index, IndexSettings, IndexSortByField, Order, ReloadPolicy, Searcher, Term,
};

#[derive(Clone, Copy, Debug)]
enum CorpusKind {
    /// Foreign-key-shaped: ~100 docs per distinct value.
    ForeignKey,
    /// Primary-key-shaped: D = 1, every value unique.
    PrimaryKey,
}

#[derive(Clone, Copy, Debug)]
enum Sort {
    Asc,
    None,
}

/// Generate the column value for the given doc id under the chosen corpus shape.
fn value_for(doc_id: u64, kind: CorpusKind) -> u64 {
    match kind {
        CorpusKind::ForeignKey => doc_id / 100,
        CorpusKind::PrimaryKey => doc_id,
    }
}

fn distinct_count(n: u64, kind: CorpusKind) -> u64 {
    match kind {
        CorpusKind::ForeignKey => n.div_ceil(100),
        CorpusKind::PrimaryKey => n,
    }
}

fn build_corpus(
    n: u64,
    kind: CorpusKind,
    sort: Sort,
) -> (Index, Searcher, tantivy::schema::Field) {
    let mut schema_builder = SchemaBuilder::new();
    let field = schema_builder.add_u64_field(
        "fk",
        NumericOptions::default().set_fast().set_indexed(),
    );
    let schema = schema_builder.build();

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
        let mut writer = index.writer_with_num_threads(1, 200_000_000).unwrap();
        for doc_id in 0..n {
            writer
                .add_document(doc!(field => value_for(doc_id, kind)))
                .unwrap();
        }
        writer.commit().unwrap();
    }

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    let searcher = reader.searcher();
    (index, searcher, field)
}

/// Pick `k` distinct values from `[0, distinct)` deterministically.
fn sample_terms(distinct: u64, k: usize, seed: u64) -> Vec<u64> {
    assert!(k as u64 <= distinct, "k must fit in distinct value space");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut chosen: Vec<u64> = (0..distinct).collect();
    chosen.shuffle(&mut rng);
    chosen.truncate(k);
    chosen
}

struct Cell {
    name: String,
    searcher: Searcher,
    field: tantivy::schema::Field,
    terms: Vec<u64>,
    n: u64,
}

impl Cell {
    fn run(&self) -> usize {
        let q = FastFieldTermSetQuery::new(
            self.terms
                .iter()
                .map(|v| Term::from_field_u64(self.field, *v)),
        );
        self.searcher.search(&q, &Count).unwrap()
    }
}

fn main() {
    let n: u64 = 1_000_000;
    let k: usize = 1_000;

    // Build the four shared corpora once.
    let (_idx_fk_asc, s_fk_asc, f_fk_asc) =
        build_corpus(n, CorpusKind::ForeignKey, Sort::Asc);
    let (_idx_fk_none, s_fk_none, f_fk_none) =
        build_corpus(n, CorpusKind::ForeignKey, Sort::None);
    let (_idx_pk_asc, s_pk_asc, f_pk_asc) =
        build_corpus(n, CorpusKind::PrimaryKey, Sort::Asc);
    let (_idx_pk_none, s_pk_none, f_pk_none) =
        build_corpus(n, CorpusKind::PrimaryKey, Sort::None);

    let cells = vec![
        Cell {
            name: "fk_sorted_asc_n=1M_k=1000".to_string(),
            terms: sample_terms(distinct_count(n, CorpusKind::ForeignKey), k, 7),
            searcher: s_fk_asc,
            field: f_fk_asc,
            n,
        },
        Cell {
            name: "fk_unsorted_n=1M_k=1000".to_string(),
            terms: sample_terms(distinct_count(n, CorpusKind::ForeignKey), k, 7),
            searcher: s_fk_none,
            field: f_fk_none,
            n,
        },
        Cell {
            name: "pk_sorted_asc_n=1M_k=1000".to_string(),
            terms: sample_terms(distinct_count(n, CorpusKind::PrimaryKey), k, 7),
            searcher: s_pk_asc,
            field: f_pk_asc,
            n,
        },
        Cell {
            name: "pk_unsorted_n=1M_k=1000".to_string(),
            terms: sample_terms(distinct_count(n, CorpusKind::PrimaryKey), k, 7),
            searcher: s_pk_none,
            field: f_pk_none,
            n,
        },
    ];

    let mut runner = BenchRunner::new();
    let mut group = runner.new_group();
    group.set_name("term_set_step0_baseline_linear");
    for cell in &cells {
        group.set_input_size(cell.n as usize);
        group.register(cell.name.clone(), |_| black_box(cell.run()));
    }
    group.run();
}
