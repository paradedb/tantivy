//! Router source tests: the source driven by scripted feedback, with no scan.

use common::BitSet;

use crate::query::{AllQuery, Query};
use crate::schema::{Metric, VectorDType};
use crate::vector::diagnostics::VectorStorageFormat;
use crate::vector::params::{LocatedPath, ProbeParams, DEFAULT_FIXED_PROBE_COST_ROWS};
use crate::vector::router::stacked::aps::RecallEstimator;
use crate::vector::search::cluster_source::{ClusterSource, Feedback};
use crate::vector::search::segment::plan::Source;
use crate::vector::search::source::controller::{
    open_share, ProbeController, ProbeTermination, WorkBudget, WorkMeter, WorkUnits,
};
use crate::vector::search::source::router::{
    DocFilter, GatedRows, RouterClusters, RouterReport, RowGate,
};
use crate::vector::storage::bounds::{HeapPeek, QueryBound, QueryBoundTracker};
use crate::vector::test_support::{DocsWeight, Fixture, FixtureSpec, Lcg, SegmentRun};
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

/// Walks segment 0's router source as a scan would, admitting every row of every cluster;
/// `feedback(rows)` is what the scan reports after `rows` admitted rows.
fn walk(seg: &SegmentRun<'_>, feedback: impl Fn(usize) -> Feedback) -> crate::Result<Walk> {
    let plan = seg.plan()?;
    let Some(Source::Router(filter)) = plan.source else {
        panic!("the router source walks router plans");
    };
    let mut source = RouterClusters::new(
        seg.searcher.reader(),
        seg.searcher.query(),
        plan.budget,
        filter,
        seg.segment,
        seg.searcher.params(),
    );
    let mut clusters = Vec::new();
    let mut rows = 0;
    while let Some((cluster, len)) = source
        .next(feedback(rows))?
        .map(|rows| (rows.cluster, rows.len()))
    {
        clusters.push(cluster);
        rows += len;
        source.admitted(len, feedback(rows))?;
    }
    Ok(Walk {
        clusters,
        rows,
        report: source.finish(),
    })
}

/// Feedback of a scan whose heap never fills.
fn unarmed(_: usize) -> Feedback {
    Feedback::default()
}

/// Feedback of a scan that holds the final k-th score once `k` rows are in.
fn armed(k: usize, kth: f32) -> impl Fn(usize) -> Feedback {
    move |rows| {
        let kth = (rows >= k).then_some(kth);
        Feedback {
            exact_rows: None,
            kth_lower: kth,
            kth_estimate: kth,
        }
    }
}

/// A walk over `fixture`'s first segment.
fn walk_fixture(
    fixture: &Fixture,
    filter: &dyn Query,
    query: &[f32],
    k: usize,
    params: ProbeParams,
    feedback: impl Fn(usize) -> Feedback,
) -> crate::Result<Walk> {
    fixture.with_segment(filter, query, k, &params, |seg| walk(seg, feedback))
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

/// Six separated clusters of six docs: probing the home cluster arms a bound that proves the
/// other five useless.
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

/// The `k`-th oracle score for `query` over `fixture`.
fn kth_score(fixture: &Fixture, query: &[f32], k: usize) -> crate::Result<f32> {
    Ok(fixture.oracle(&AllQuery, query, k)?[k - 1].0)
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
/// exactly the cluster count, skew moves the stop point, filtered rows charge nothing, and gate
/// skips charge their open.
#[test]
fn controller_charges_opens_and_scored_rows() -> crate::Result<()> {
    // Sizes [5, 2, 2, 1]: an exhaustive walk charges exactly C = 4 whatever the skew.
    let fixture = sized_fixture(&[5, 2, 2, 1], 0.01)?;
    let full = walk_fixture(
        &fixture,
        &AllQuery,
        &[0.0, 0.0],
        11,
        budget_params(1.0),
        unarmed,
    )?;
    assert_eq!(full.report.probe.termination, ProbeTermination::Exhausted);
    assert!(
        (full.report.probe.work_charged - 4.0).abs() <= 4e-6,
        "{:?}",
        full.report
    );

    // Sizes [30, 2, 2, 2, 2, 2]: n_avg = 20/3, x ~ 0.1975, a row ~ 0.1204. At f = 0.8 (budget 4.8)
    // the big cluster charges ~3.809 and each small one ~0.438, so the fourth open crosses it.
    let fixture = sized_fixture(&[30, 2, 2, 2, 2, 2], 0.001)?;
    let skewed = walk_fixture(
        &fixture,
        &AllQuery,
        &[0.0, 0.0],
        41,
        budget_params(0.8),
        unarmed,
    )?;
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
    let max_doc = fixture.index.reader()?.searcher().segment_readers()[0].max_doc();
    let filtered =
        fixture.with_segment(&AllQuery, &[0.0, 0.0], 23, &budget_params(1.0), |seg| {
            let weight = DocsWeight { max_doc, docs };
            walk(
                &SegmentRun {
                    weight: &weight,
                    ..*seg
                },
                unarmed,
            )
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

    // The bound proves five of six clusters useless: each skip charges its open.
    let fixture = separated_fixture(6)?;
    let query = [0.2, 0.3];
    let kth = kth_score(&fixture, &query, 5)?;
    let gated = walk_fixture(
        &fixture,
        &AllQuery,
        &query,
        5,
        exhaustive_params(6),
        armed(5, kth),
    )?;
    assert_eq!(gated.clusters.len(), 1);
    assert_eq!(gated.report.counters.bounds_skips, 5);
    let x = open_share(DEFAULT_FIXED_PROBE_COST_ROWS, 6.0);
    let expected = 6.0 * x + 6.0 * (1.0 - x) / 6.0;
    assert!(
        (gated.report.probe.work_charged as f64 - expected).abs() < 1e-5,
        "skips charge the open share: {:?}",
        gated.report
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

/// The loop stops on the ceiling, on exhaustion, or at the recall target, which only the stacked
/// router estimates and only for metrics with a query ball.
#[test]
fn probe_termination() -> crate::Result<()> {
    // The controller alone: the first pull after the estimate reaches the target stops.
    let meter = WorkMeter::new(WorkBudget {
        total: WorkUnits::new(100.0),
        per_cluster: WorkUnits::new(0.5),
        per_row: WorkUnits::new(0.01),
    });
    let centroids = [[0.0f32, 0.0], [2.0, 0.0], [0.0, 2.0]];
    let estimator = RecallEstimator::from_rows(
        &[0.1, 0.0],
        centroids.iter().map(|row| &row[..]).collect(),
        Metric::L2,
    );
    let mut controller = ProbeController::new(3, Some(estimator), 0.5);
    assert!(controller.admit(&meter));
    controller.cover(None)?;
    assert!(controller.admit(&meter), "no estimate while the heap fills");
    // A ball of radius 0.2 lies inside the nearest cell.
    controller.cover(Some(-(0.2f32 * 0.2)))?;
    assert!(!controller.admit(&meter));
    let outcome = controller.finish(&meter);
    assert_eq!(outcome.termination, ProbeTermination::RecallTarget);
    assert!(outcome
        .recall_estimate
        .is_some_and(|estimate| estimate >= 0.5));

    // Ceiling: a budget below one cluster stops after the first.
    let fixture = separated_fixture(6)?;
    let ceiling = walk_fixture(
        &fixture,
        &AllQuery,
        &[11.0, 11.0],
        3,
        budget_params(0.1),
        unarmed,
    )?;
    assert_eq!(ceiling.report.probe.termination, ProbeTermination::Ceiling);
    assert_eq!(ceiling.clusters.len(), 1);

    // Exhausted: an exhaustive, unarmed walk opens every cluster and admits every row.
    let fixture = grid(Metric::L2, RouterKind::Stacked)?;
    let exhausted = walk_fixture(
        &fixture,
        &AllQuery,
        &[0.0, 0.0],
        64,
        exhaustive_params(9),
        unarmed,
    )?;
    let counters = exhausted.report.counters;
    assert_eq!(
        exhausted.report.probe.termination,
        ProbeTermination::Exhausted
    );
    assert_eq!(counters.postings_row + counters.postings_skipped, 9);
    let docs = fixture.index.reader()?.searcher().segment_readers()[0].max_doc() as usize;
    assert_eq!(exhausted.rows, docs);

    // Recall target: a loose target stops on the estimate before the budget; 1.0 leaves APS off.
    let recall = |target| ProbeParams {
        recall_target: target,
        ..budget_params(1.0)
    };
    let kth = kth_score(&fixture, &[0.0, 0.0], 3)?;
    let aps = walk_fixture(
        &fixture,
        &AllQuery,
        &[0.0, 0.0],
        3,
        recall(0.5),
        armed(3, kth),
    )?
    .report
    .probe;
    assert_eq!(aps.termination, ProbeTermination::RecallTarget, "{aps:?}");
    assert!(aps.recall_estimate.is_some_and(|estimate| estimate >= 0.5));
    let off = walk_fixture(
        &fixture,
        &AllQuery,
        &[0.0, 0.0],
        3,
        recall(1.0),
        armed(3, kth),
    )?
    .report
    .probe;
    assert_ne!(off.termination, ProbeTermination::RecallTarget, "{off:?}");
    assert_eq!(off.recall_estimate, None);
    assert!(aps.work_charged < off.work_charged && aps.work_charged < aps.work_budget);
    assert_eq!(aps.work_budget, off.work_budget);

    // Dot has no query ball; the exact router has no estimator.
    for (metric, router) in [
        (Metric::Dot, RouterKind::Stacked),
        (Metric::L2, RouterKind::Exact),
    ] {
        let fixture = grid(metric, router)?;
        let query = [1.0, 0.0];
        let kth = kth_score(&fixture, &query, 3)?;
        let probe = walk_fixture(&fixture, &AllQuery, &query, 3, recall(0.5), armed(3, kth))?
            .report
            .probe;
        assert_ne!(
            probe.termination,
            ProbeTermination::RecallTarget,
            "{metric:?} {router:?}"
        );
        assert_eq!(probe.recall_estimate, None, "{metric:?} {router:?}");
    }

    // A binding budget is recorded, stops before exhaustion, and overshoots by at most one
    // cluster's charge.
    let fixture = grid(Metric::L2, RouterKind::Exact)?;
    let params = budget_params(0.2);
    let probe = fixture.with_segment(&AllQuery, &[0.0, 0.0], 3, &params, |seg| {
        let (budget, n_avg, x) =
            params.resolved_work_budget(seg.index.num_clusters(), seg.index.num_docs())?;
        let clusters = seg.index.num_clusters();
        let docs = seg.index.num_docs() as f64;
        let walk = walk(seg, unarmed)?;
        assert!(budget < clusters as f64, "the budget binds");
        assert!(walk.clusters.len() < clusters);
        let probe = walk.report.probe;
        assert!((probe.work_budget as f64 - budget).abs() <= 1e-6 * budget);
        assert!(probe.work_charged as f64 > budget);
        assert!(probe.work_charged as f64 <= budget + x + docs * (1.0 - x) / n_avg + 1e-6);
        Ok(probe)
    })?;
    assert_eq!(probe.termination, ProbeTermination::Ceiling);
    Ok(())
}

// --- Bounds gate ---

/// A random 2-d point away from the origin, so cosine's normalization stays well-conditioned.
fn point(rng: &mut Lcg) -> Vec<f32> {
    loop {
        let point = rng.vector(2, 8.0);
        if point[0] * point[0] + point[1] * point[1] > 0.25 {
            return point;
        }
    }
}

/// One fixture of the gate property: its geometry, query and k.
struct GateCase {
    name: String,
    metric: Metric,
    centroids: Vec<Vec<f32>>,
    docs: Vec<Vec<f32>>,
    query: Vec<f32>,
    k: usize,
}

/// Under the final, tightest bound no cluster holding a true top-k member is skipped, so none is
/// skipped at any looser bound during the scan; exact ties at the bound probe.
#[test]
fn gate_never_skips_a_true_member() -> crate::Result<()> {
    let mut rows = Vec::new();
    for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
        for seed in [11u64, 29, 47] {
            let mut rng = Lcg(seed ^ ((metric as u64) << 32));
            rows.push(GateCase {
                name: format!("{metric:?} seed {seed}"),
                metric,
                centroids: (0..5).map(|_| point(&mut rng)).collect(),
                docs: (0..40).map(|_| point(&mut rng)).collect(),
                query: point(&mut rng),
                k: 5,
            });
        }
    }
    // L2: the second cluster's margin is exactly zero (powers of two), and its doc ties the k-th.
    rows.push(GateCase {
        name: "L2 tie at the bound".into(),
        metric: Metric::L2,
        centroids: vec![vec![0.0, 4.0], vec![4.0, 0.0], vec![8.0, 0.0]],
        docs: vec![vec![0.0, 2.0], vec![2.0, 0.0], vec![7.0, 0.0]],
        query: vec![0.0, 0.0],
        k: 1,
    });
    // Dot: the second cluster's best possible score q.c + |q| r equals the k-th score.
    rows.push(GateCase {
        name: "Dot tie at the bound".into(),
        metric: Metric::Dot,
        centroids: vec![vec![2.0, 0.0], vec![4.0, 4.0]],
        docs: vec![vec![3.0, 0.0], vec![3.0, 4.0]],
        query: vec![1.0, 0.0],
        k: 1,
    });
    for GateCase {
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
        let kth = oracle[k - 1].0;
        let walk = walk_fixture(&fixture, &AllQuery, &query, k, exhaustive_params(5), |_| {
            Feedback {
                exact_rows: None,
                kth_lower: Some(kth),
                kth_estimate: Some(kth),
            }
        })?;
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
            "{name}: gated top-k equals the oracle"
        );
    }
    Ok(())
}

/// The gate's bookkeeping: what the tracker arms on, what the source counts, and what the row
/// gate keeps.
#[test]
fn gate_bookkeeping() -> crate::Result<()> {
    // The bound tracks k-th improvements in bound space per metric; an unchanged k-th keeps it.
    for (metric, first, t_first, improved, t_improved) in [
        (Metric::L2, -4.0f32, 2.0f32, -1.0f32, 1.0f32),
        (Metric::Cosine, 0.5, 1.0, 0.875, 0.5),
        (Metric::Dot, 3.0, 3.0, 4.5, 4.5),
    ] {
        let mut tracker = QueryBoundTracker::new();
        tracker.observe(metric, HeapPeek::Filling, 0);
        assert_eq!(tracker.bound(), QueryBound::Filling, "{metric:?}");
        tracker.observe(metric, HeapPeek::Full { kth_key: first }, 1);
        assert_eq!(
            tracker.bound(),
            QueryBound::Armed { t: t_first },
            "{metric:?}"
        );
        tracker.observe(metric, HeapPeek::Full { kth_key: first }, 2);
        assert_eq!(
            tracker.bound(),
            QueryBound::Armed { t: t_first },
            "{metric:?}"
        );
        tracker.observe(metric, HeapPeek::Full { kth_key: improved }, 3);
        assert_eq!(
            tracker.bound(),
            QueryBound::Armed { t: t_improved },
            "{metric:?}"
        );
        assert_eq!(
            tracker.armed_at_probe(),
            Some(1),
            "the armed probe never moves"
        );
    }

    // Armed at the first cluster, the source counts the other five as skips.
    let fixture = separated_fixture(6)?;
    let query = [0.2, 0.3];
    let kth = kth_score(&fixture, &query, 5)?;
    let walk = walk_fixture(
        &fixture,
        &AllQuery,
        &query,
        5,
        exhaustive_params(6),
        armed(5, kth),
    )?;
    assert_eq!(walk.report.probe.termination, ProbeTermination::Exhausted);
    assert_eq!(walk.clusters.len(), 1);
    assert_eq!(walk.report.counters.bounds_skips, 5);
    assert_eq!(walk.report.counters.bound_armed_at, Some(0));

    // A heap that fills only in the second cluster arms there, and still skips the far tail.
    let kth = kth_score(&fixture, &query, 10)?;
    let walk = walk_fixture(
        &fixture,
        &AllQuery,
        &query,
        10,
        exhaustive_params(6),
        armed(10, kth),
    )?;
    assert_eq!(walk.report.counters.bound_armed_at, Some(1));
    assert!(walk.report.counters.bounds_skips > 0);

    // Unarmed, nothing is skipped and the armed probe stays absent.
    let fixture = separated_fixture(3)?;
    let walk = walk_fixture(
        &fixture,
        &AllQuery,
        &query,
        100,
        exhaustive_params(3),
        unarmed,
    )?;
    assert_eq!(walk.clusters.len(), 3);
    assert_eq!(walk.rows, 18);
    assert_eq!(walk.report.counters.bounds_skips, 0);
    assert_eq!(walk.report.counters.bound_armed_at, None);
    let x = open_share(DEFAULT_FIXED_PROBE_COST_ROWS, 6.0);
    let expected = 3.0 * x + 18.0 * (1.0 - x) / 6.0;
    assert!((walk.report.probe.work_charged as f64 - expected).abs() < 1e-5);

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
    let max_doc = fixture.index.reader()?.searcher().segment_readers()[0].max_doc();
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
                let mut source = RouterClusters::new(
                    run.searcher.reader(),
                    run.searcher.query(),
                    plan.budget,
                    filter,
                    run.segment,
                    run.searcher.params(),
                );
                while let Some(rows) = source.next(Feedback::default())? {
                    match rows.docs {
                        None => assert!(open),
                        Some(given) => {
                            assert!(!open);
                            assert_eq!(given.len(), rows.len());
                            assert!(given.iter().all(|doc| keep.contains(doc)));
                        }
                    }
                    let len = rows.len();
                    source.admitted(len, Feedback::default())?;
                }
            }
            Ok(())
        },
    )?;
    Ok(())
}
