//! Router source tests: cluster admission and work accounting, with no scan.

use common::BitSet;

use crate::query::{AllQuery, Query};
use crate::schema::{Metric, VectorDType};
use crate::vector::diagnostics::VectorStorageFormat;
use crate::vector::params::{LocatedPath, ProbeParams, DEFAULT_FIXED_PROBE_COST_ROWS};
use crate::vector::search::cluster_source::ClusterSource;
use crate::vector::search::segment::plan::Source;
use crate::vector::search::source::controller::ProbeTermination;
use crate::vector::search::source::router::{
    DocFilter, GatedRows, RouterClusters, RouterReport, RowGate,
};
use crate::vector::test_support::{
    first_segment, DocsWeight, Fixture, FixtureSpec, Lcg, SegmentRun,
};
use crate::vector::tests::{exhaustive_params, routed_exhaustive_params, TestVectorIndex};
use crate::vector::RouterKind;
use crate::DocId;

// --- Driver ---

/// What a walk over a router source admitted.
struct Walk {
    clusters: Vec<usize>,
    rows: usize,
    report: RouterReport,
}

/// Walks the first segment's router source, admitting every row of every cluster.
fn walk(seg: &SegmentRun<'_>) -> crate::Result<Walk> {
    let plan = seg.plan()?;
    let Some(Source::Router(filter)) = plan.source else {
        panic!("the router source walks router plans");
    };
    let mut source = RouterClusters::for_segment(
        seg.searcher.reader(),
        seg.searcher.query(),
        plan.budget,
        filter,
        seg.segment,
        seg.searcher.params(),
    );
    let mut clusters = Vec::new();
    let mut rows = 0;
    while let Some((cluster, len)) = source.next()?.map(|rows| (rows.cluster, rows.len())) {
        clusters.push(cluster);
        rows += len;
        source.admitted(len)?;
    }
    Ok(Walk {
        clusters,
        rows,
        report: source.finish(),
    })
}

/// A walk over `fixture`'s first segment.
fn walk_fixture(
    fixture: &Fixture,
    filter: &dyn Query,
    query: &[f32],
    k: usize,
    params: ProbeParams,
) -> crate::Result<Walk> {
    fixture.with_segment(filter, query, k, &params, walk)
}

/// A 2-d fixture with `sizes[c]` docs around centroid `[10 c, 0]`.
fn sized_fixture(sizes: &[usize], spacing: f32) -> crate::Result<Fixture> {
    let centroids: Vec<Vec<f32>> = (0..sizes.len())
        .map(|c| vec![c as f32 * 10.0, 0.0])
        .collect();
    let mut vectors = Vec::new();
    for (c, &size) in sizes.iter().enumerate() {
        for i in 0..size {
            vectors.push(vec![c as f32 * 10.0 + i as f32 * spacing, 0.0]);
        }
    }
    FixtureSpec::ivf(Metric::L2, centroids, vectors).build()
}

/// Up to six separated clusters of six docs.
fn separated_fixture(clusters: usize) -> crate::Result<Fixture> {
    let centroids: Vec<Vec<f32>> = [
        [1.0, 1.0],
        [11.0, 1.0],
        [21.0, 1.0],
        [1.0, 11.0],
        [11.0, 11.0],
        [21.0, 11.0],
    ]
    .iter()
    .take(clusters)
    .map(|c| c.to_vec())
    .collect();
    let per = 36 / 6;
    let vectors = (0..clusters * per)
        .map(|i| {
            let c = &centroids[i / per];
            let offset = (i % per) as f32 * 0.01;
            vec![c[0] + offset, c[1] + offset]
        })
        .collect();
    FixtureSpec::ivf(Metric::L2, centroids, vectors).build()
}

/// A budget of `max_probe_fraction` on the routed path.
fn budget_params(max_probe_fraction: f32) -> ProbeParams {
    ProbeParams {
        max_probe_fraction,
        min_probe_clusters: 1,
        exact_plan: false,
        located: LocatedPath::Off,
        ..Default::default()
    }
}

// --- Charges ---

/// Opens charge `x` and scored rows `(1 - x)/n_avg`: an exhaustive unfiltered walk charges
/// exactly the cluster count, skew moves the stop point, and filtered rows charge nothing.
#[test]
fn controller_charges_opens_and_scored_rows() -> crate::Result<()> {
    // Sizes [5, 2, 2, 1]: an exhaustive walk charges exactly C = 4 whatever the skew.
    let fixture = sized_fixture(&[5, 2, 2, 1], 0.01)?;
    let full = walk_fixture(&fixture, &AllQuery, &[0.0, 0.0], 11, budget_params(1.0))?;
    assert_eq!(full.report.probe.termination, ProbeTermination::Exhausted);
    assert!(
        (full.report.probe.work_charged - 4.0).abs() <= 4e-6,
        "{:?}",
        full.report
    );

    // Sizes [30, 2, 2, 2, 2, 2]: n_avg = 20/3, x ~ 0.1975, a row ~ 0.1204. At f = 0.8 (budget 4.8)
    // the big cluster charges ~3.809 and each small one ~0.438, so the fourth open crosses it.
    let fixture = sized_fixture(&[30, 2, 2, 2, 2, 2], 0.001)?;
    let skewed = walk_fixture(&fixture, &AllQuery, &[0.0, 0.0], 41, budget_params(0.8))?;
    let probe = skewed.report.probe;
    assert_eq!(probe.termination, ProbeTermination::Ceiling);
    assert_eq!(skewed.clusters.len(), 4, "{:?}", skewed.report);
    assert!((probe.work_charged - 5.123).abs() < 2e-3, "{probe:?}");
    let last_charge = 0.1975 + 2.0 * 0.1204;
    assert!(
        probe.work_charged > 4.8 && probe.work_charged - 4.8 <= last_charge + 1e-4,
        "the overrun is at most the last cluster's charge: {probe:?}"
    );

    // Sizes [20, 2]: three of cluster 0's docs pass the filter; the rest are walked, not charged.
    let fixture = sized_fixture(&[20, 2], 0.001)?;
    let n_avg = 11.0;
    let x = DEFAULT_FIXED_PROBE_COST_ROWS / (DEFAULT_FIXED_PROBE_COST_ROWS + n_avg);
    let row = (1.0 - x) / n_avg;
    let docs = fixture.cluster_docs(0)?[0][..3].to_vec();
    let max_doc = first_segment(&fixture.index.reader()?.searcher())
        .1
        .max_doc();
    let filtered =
        fixture.with_segment(&AllQuery, &[0.0, 0.0], 23, &budget_params(1.0), |seg| {
            let weight = DocsWeight { max_doc, docs };
            walk(&SegmentRun {
                weight: &weight,
                ..*seg
            })
        })?;
    let counters = filtered.report.counters;
    assert_eq!(counters.postings_row + counters.postings_skipped, 2);
    assert_eq!(counters.vectors_visited, 22);
    assert_eq!(filtered.rows, 3);
    assert!(
        (filtered.report.probe.work_charged as f64 - (2.0 * x + 3.0 * row)).abs() < 1e-5,
        "two opens and three rows: {:?}",
        filtered.report
    );

    Ok(())
}

// --- Termination ---

fn grid(metric: Metric, router: RouterKind) -> crate::Result<Fixture> {
    Ok(TestVectorIndex::builder(VectorDType::F32)
        .metric(metric)
        .router(router)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .build()?
        .into())
}

/// The loop stops on the ceiling or on exhaustion.
#[test]
fn probe_termination() -> crate::Result<()> {
    // Ceiling: a budget below one cluster stops after the first.
    let fixture = separated_fixture(6)?;
    let ceiling = walk_fixture(&fixture, &AllQuery, &[11.0, 11.0], 3, budget_params(0.1))?;
    assert_eq!(ceiling.report.probe.termination, ProbeTermination::Ceiling);
    assert_eq!(ceiling.clusters.len(), 1);

    // Exhausted: an exhaustive walk opens every cluster and admits every row.
    // Exact: Stacked's 25% top prune cannot hand the probe every list on a 3x3 grid.
    let fixture = grid(Metric::L2, RouterKind::Exact)?;
    let exhausted = walk_fixture(&fixture, &AllQuery, &[0.0, 0.0], 64, exhaustive_params(9))?;
    let counters = exhausted.report.counters;
    assert_eq!(
        exhausted.report.probe.termination,
        ProbeTermination::Exhausted
    );
    assert_eq!(counters.postings_row + counters.postings_skipped, 9);
    let docs = first_segment(&fixture.index.reader()?.searcher())
        .1
        .max_doc() as usize;
    assert_eq!(exhausted.rows, docs);

    // A binding budget is recorded, stops before exhaustion, and overshoots by at most one
    // cluster's charge.
    let fixture = grid(Metric::L2, RouterKind::Exact)?;
    let params = budget_params(0.2);
    let probe = fixture.with_segment(&AllQuery, &[0.0, 0.0], 3, &params, |seg| {
        let (budget, n_avg, x) =
            params.resolved_work_budget(seg.index.num_clusters(), seg.index.num_rows())?;
        let clusters = seg.index.num_clusters();
        let rows = seg.index.num_rows() as f64;
        let walk = walk(seg)?;
        assert!(budget < clusters as f64, "the budget binds");
        assert!(walk.clusters.len() < clusters);
        let probe = walk.report.probe;
        assert!((probe.work_budget as f64 - budget).abs() <= 1e-6 * budget);
        assert!(probe.work_charged as f64 > budget);
        assert!(probe.work_charged as f64 <= budget + x + rows * (1.0 - x) / n_avg + 1e-6);
        Ok(probe)
    })?;
    assert_eq!(probe.termination, ProbeTermination::Ceiling);
    Ok(())
}

// --- Exhaustive search ---

/// A random 2-d point away from the origin, so cosine's normalization stays well-conditioned.
fn point(rng: &mut Lcg) -> Vec<f32> {
    loop {
        let point = rng.vector(2, 8.0);
        if point[0] * point[0] + point[1] * point[1] > 0.25 {
            return point;
        }
    }
}

/// One search fixture: its geometry, query and k.
struct SearchCase {
    name: String,
    metric: Metric,
    centroids: Vec<Vec<f32>>,
    docs: Vec<Vec<f32>>,
    query: Vec<f32>,
    k: usize,
}

/// Exhaustive routing visits every cluster and matches the exact oracle, including ties.
#[test]
fn exhaustive_routing_matches_oracle() -> crate::Result<()> {
    let mut rows = Vec::new();
    for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
        for seed in [11u64, 29, 47] {
            let mut rng = Lcg(seed ^ ((metric as u64) << 32));
            rows.push(SearchCase {
                name: format!("{metric:?} seed {seed}"),
                metric,
                centroids: (0..5).map(|_| point(&mut rng)).collect(),
                docs: (0..40).map(|_| point(&mut rng)).collect(),
                query: point(&mut rng),
                k: 5,
            });
        }
    }
    // The two nearest documents tie under L2.
    rows.push(SearchCase {
        name: "L2 tie".into(),
        metric: Metric::L2,
        centroids: vec![vec![0.0, 4.0], vec![4.0, 0.0], vec![8.0, 0.0]],
        docs: vec![vec![0.0, 2.0], vec![2.0, 0.0], vec![7.0, 0.0]],
        query: vec![0.0, 0.0],
        k: 1,
    });
    // Equal dot products in different clusters.
    rows.push(SearchCase {
        name: "Dot tie".into(),
        metric: Metric::Dot,
        centroids: vec![vec![2.0, 0.0], vec![4.0, 4.0]],
        docs: vec![vec![3.0, 0.0], vec![3.0, 4.0]],
        query: vec![1.0, 0.0],
        k: 1,
    });
    for SearchCase {
        name,
        metric,
        centroids,
        docs,
        query,
        k,
    } in rows
    {
        let fixture = FixtureSpec::ivf(metric, centroids, docs).build()?;
        let oracle = fixture.oracle(&AllQuery, &query, k)?;
        let walk = walk_fixture(&fixture, &AllQuery, &query, k, exhaustive_params(5))?;
        assert_eq!(
            walk.report.counters.postings_row + walk.report.counters.postings_skipped,
            fixture.cluster_docs(0)?.len(),
            "{name}"
        );
        let cluster_docs = fixture.cluster_docs(0)?;
        for (_, address) in &oracle {
            let home = cluster_docs
                .iter()
                .position(|docs| docs.contains(&address.doc_id))
                .expect("member's cluster");
            assert!(
                walk.clusters.contains(&home),
                "{name}: member's cluster {home} skipped"
            );
        }
        let fruit = fixture.search(&AllQuery, &query, k, exhaustive_params(5))?;
        assert_eq!(
            fruit.results, oracle,
            "{name}: exhaustive top-k equals the oracle"
        );
    }
    Ok(())
}

#[test]
fn row_gate_bookkeeping() -> crate::Result<()> {
    // The row gate keeps the filter's offsets; an open gate keeps every row unread.
    let mut filter = BitSet::with_max_value(32);
    filter.insert(4);
    filter.insert(12);
    let mut offsets = Vec::new();
    let gated = RowGate::new(DocFilter::Docs(filter), None).select(&[4, 9, 12, 22], &mut offsets);
    assert_eq!(
        gated,
        GatedRows {
            all: false,
            pruned_filter: 2,
            pruned_dead: 0,
        }
    );
    assert_eq!(offsets, [0, 2]);
    let open = RowGate::new(DocFilter::All, None);
    assert!(open.is_open());
    assert!(open.select(&[], &mut offsets).all);
    assert!(offsets.is_empty());

    // A gated cluster carries its rows' doc ids, aligned with its offsets; an open one reads none.
    let fixture = separated_fixture(2)?;
    let docs = fixture.cluster_docs(0)?;
    let max_doc = first_segment(&fixture.index.reader()?.searcher())
        .1
        .max_doc();
    let keep: Vec<DocId> = docs
        .iter()
        .flat_map(|docs| docs.iter().step_by(2).copied())
        .collect();
    fixture.with_segment(
        &AllQuery,
        &[1.0, 1.0],
        100,
        &routed_exhaustive_params(2),
        |seg| {
            let weight = DocsWeight {
                max_doc,
                docs: keep.clone(),
            };
            let gated = SegmentRun {
                weight: &weight,
                ..*seg
            };
            for (run, open) in [(seg, true), (&gated, false)] {
                let plan = run.plan()?;
                let Some(Source::Router(filter)) = plan.source else {
                    panic!("the router source walks router plans");
                };
                let mut source = RouterClusters::for_segment(
                    run.searcher.reader(),
                    run.searcher.query(),
                    plan.budget,
                    filter,
                    run.segment,
                    run.searcher.params(),
                );
                while let Some(rows) = source.next()? {
                    match rows.docs {
                        None => assert!(open),
                        Some(given) => {
                            assert!(!open);
                            assert_eq!(given.len(), rows.len());
                            assert!(given.iter().all(|doc| keep.contains(doc)));
                        }
                    }
                    let len = rows.len();
                    source.admitted(len)?;
                }
            }
            Ok(())
        },
    )?;
    Ok(())
}
