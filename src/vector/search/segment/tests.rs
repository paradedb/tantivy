//! End-to-end segment search tests: results against the brute-force oracle, routing edge cases,
//! and the storage reads each plan makes.

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::collector::{AtomicSharedThreshold, SharedThreshold, TopDocs};
use crate::query::{AllQuery, BooleanQuery, EnableScoring, Query};
use crate::schema::Metric;
use crate::vector::params::{LocatedPath, ProbeParams, Selectivity};
use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
use crate::vector::search::interval::Estimate;
use crate::vector::search::scan::bound::TopK;
use crate::vector::search::scan::layer_read::LayerRead;
use crate::vector::search::scan::plan::{ExactRule, PlanOverride};
use crate::vector::stats::{AccessPath, LayerProbeStats};
use crate::vector::test_support::{
    first_segment, ColumnRanges, DocsWeight, Fixture, FixtureSpec, Lcg, ReadLog,
};
use crate::vector::tests::{exhaustive_params, routed_exhaustive_params};
use crate::vector::{ProbeStats, RouterKind, VectorSimilarityFruit, VectorStage};
use crate::{DocAddress, DocId, Score, SegmentOrdinal};

// --- Oracle matrix ---

const MATRIX_DIM: usize = 64;
const MATRIX_DOCS: usize = 240;
const MATRIX_CLUSTERS: usize = 6;

#[test]
fn adaptive_cascade_matches_oracle_and_prefix_replay() -> crate::Result<()> {
    use crate::vector::router::with_test_clusters;
    use crate::vector::{ProbeTermination, QuantizedRecallParams};

    let plans = [
        PlanOverride::read(LayerRead::Sparse),
        PlanOverride::exact(|_, _| true),
        PlanOverride::exact(|_, layer| layer == 1),
        PlanOverride::exact(|cluster, layer| cluster % 3 == layer),
    ];
    let mut stops = 0;
    let mut plain_stops = 0;
    let mut exact_layers = [0; 3];
    for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
        for schedule in [&[][..], &[1][..], &[1, 1][..], &[1, 1, 4][..]] {
            let representation = if schedule.is_empty() {
                Representation::FullPrecision
            } else {
                Representation::Quantized(schedule)
            };
            let fixture = matrix_fixture(metric, representation, true)?;
            for filter in [
                &AllQuery as &dyn Query,
                &fixture.filter("half"),
                &fixture.filter("rare"),
            ] {
                let weight = fixture.weight(filter)?;
                for query in matrix_queries() {
                    let mut ranked =
                        fixture.candidates(&(0..MATRIX_CLUSTERS).collect::<Vec<_>>(), &query)?;
                    ranked.sort_unstable_by(|a, b| {
                        b.sim
                            .score()
                            .total_cmp(&a.sim.score())
                            .then(a.node.cmp(&b.node))
                    });
                    for k in [1, 10, MATRIX_DOCS] {
                        let params = ProbeParams {
                            quantized_recall: Some(QuantizedRecallParams {
                                target: 1.0,
                                recall_exponent: 1.5,
                            }),
                            exact_plan: true,
                            ..routed_exhaustive_params(MATRIX_CLUSTERS)
                        };
                        for plan in plans {
                            for located in [
                                LocatedPath::Off,
                                LocatedPath::MaxSelectivity(Selectivity::new(1.0).unwrap()),
                            ] {
                                let (hits, reports) = forced_top_k(
                                    &fixture,
                                    filter,
                                    &query,
                                    k,
                                    &ProbeParams {
                                        located,
                                        ..params.clone()
                                    },
                                    plan,
                                )?;
                                assert_eq!(hits, fixture.oracle(filter, &query, k)?);
                                for report in reports {
                                    let planned: usize = (0..schedule.len())
                                        .filter_map(|layer| report.layers.get(layer))
                                        .map(|l| l.exact_rows())
                                        .sum();
                                    assert!(planned <= report.exact_rows);
                                    assert!(report.exact_rows <= report.candidates_scored);
                                    if k == MATRIX_DOCS {
                                        assert_eq!(report.exact_rows, report.candidates_scored);
                                    }
                                    for (layer, total) in exact_layers.iter_mut().enumerate() {
                                        *total += report
                                            .layers
                                            .get(layer)
                                            .map_or(0, |counts| counts.exact_rows());
                                    }
                                }
                            }
                            let mut previous_prefix = 0;
                            for target in [0.5, 0.95, 1.0] {
                                let (active, report) = with_test_clusters(ranked.clone(), || {
                                    fixture.run_planned(
                                        0,
                                        weight.as_ref(),
                                        &query,
                                        k,
                                        ProbeParams {
                                            quantized_recall: Some(QuantizedRecallParams {
                                                target,
                                                recall_exponent: 1.5,
                                            }),
                                            ..params.clone()
                                        },
                                        Some(plan),
                                        true,
                                    )
                                })?;
                                let prefix =
                                    report.clusters_probed() + report.bounds_skips as usize;
                                assert!(prefix >= previous_prefix);
                                previous_prefix = prefix;
                                if report.termination == ProbeTermination::QuantizedRecall {
                                    stops += 1;
                                    plain_stops += usize::from(schedule.is_empty());
                                    assert!(target < 1.0);
                                }
                                let (replay, replay_report) =
                                    with_test_clusters(ranked[..prefix].to_vec(), || {
                                        fixture.run_planned(
                                            0,
                                            weight.as_ref(),
                                            &query,
                                            k,
                                            params.clone(),
                                            Some(plan),
                                            true,
                                        )
                                    })?;
                                assert_eq!(active, replay);
                                assert_eq!(report.exact_rows, replay_report.exact_rows);
                                assert_eq!(
                                    report.candidates_scored,
                                    replay_report.candidates_scored
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(stops > 0);
    assert!(
        plain_stops > 0,
        "unquantized IVF must honor adaptive stopping"
    );
    assert!(exact_layers.iter().all(|&rows| rows > 0));
    Ok(())
}

#[test]
fn adaptive_stops_when_shared_threshold_prunes_every_local_candidate() -> crate::Result<()> {
    let clusters = 32;
    let centroids = (0..clusters)
        .map(|c| one_hot(64, 0, 10.0 * c as f32))
        .collect();
    let mut vectors = Vec::new();
    for c in 0..clusters {
        let x = 10.0 * c as f32;
        vectors.extend((0..100).map(|i| one_hot(64, 0, x - 0.45 + 0.009 * i as f32)));
        let mut outlier = one_hot(64, 0, x);
        outlier[1] = 1000.0;
        vectors.push(outlier);
    }
    let rows = vectors.len();
    let fixture = FixtureSpec::ivf(Metric::L2, centroids, vectors)
        .schedule(&[1, 4])
        .build()?;
    let query = one_hot(64, 0, 4.0);
    for target in [0.95, 1.0] {
        let slot = Arc::new(AtomicSharedThreshold::default());
        seed_slot(&slot, -1.0);
        let params = ProbeParams {
            quantized_recall: Some(crate::vector::QuantizedRecallParams {
                target: f64::from(target),
                recall_exponent: 1.5,
            }),
            exact_plan: false,
            located: LocatedPath::Off,
            ..exhaustive_params(clusters)
        };
        let fruit = fixture.search_with(&AllQuery, &query, 10, 0, params, Some(slot))?;
        let stats = &fruit.stats[0];
        assert!(fruit.results.is_empty());
        assert_eq!(stats.exact_rows, 0);
        assert_eq!(
            stats.layers.get(0).unwrap().admission_pruned(),
            stats.candidates_scored
        );
        if target < 1.0 {
            assert_eq!(
                stats.termination,
                crate::vector::ProbeTermination::QuantizedRecall
            );
            assert!(stats.clusters_probed() < clusters);
            assert!(stats.candidates_scored < rows);
        } else {
            assert_eq!(
                stats.termination,
                crate::vector::ProbeTermination::Exhausted
            );
            assert_eq!(stats.clusters_probed(), clusters);
            assert_eq!(stats.candidates_scored, rows);
        }
    }
    Ok(())
}

#[test]
fn disabled_adaptive_preserves_routing_budget_and_access_path() -> crate::Result<()> {
    for router in [RouterKind::Exact, RouterKind::Stacked] {
        for schedule in [&[][..], &[1, 4][..]] {
            let fixture = FixtureSpec::clustered(Metric::L2, 64, 64, 6400)
                .router(router)
                .schedule(schedule)
                .paged()
                .label("few", |i| i % 100 == 0)
                .label("some", |i| i % 10 == 0)
                .build()?;
            let query = Lcg(42).vector(64, 0.5);
            for filter in [
                &AllQuery as &dyn Query,
                &fixture.filter("few"),
                &fixture.filter("some"),
            ] {
                for max_probe_fraction in [0.03, 0.2] {
                    let params = ProbeParams {
                        max_probe_fraction,
                        min_probe_clusters: 1,
                        ..Default::default()
                    };
                    fixture.clear_reads();
                    let baseline = fixture.search(filter, &query, 10, params.clone())?;
                    let baseline_reads = fixture.reads();
                    let expected = &baseline.stats[0];
                    fixture.clear_reads();
                    let adaptive = fixture.search(
                        filter,
                        &query,
                        10,
                        ProbeParams {
                            quantized_recall: Some(crate::vector::QuantizedRecallParams {
                                target: 1.0,
                                recall_exponent: 1.5,
                            }),
                            ..params.clone()
                        },
                    )?;
                    let reads = fixture.reads();
                    let actual = &adaptive.stats[0];
                    assert_eq!(actual.work_budget, expected.work_budget);
                    assert_eq!(actual.located_cap, expected.located_cap);
                    assert_eq!(actual.access_path, expected.access_path);
                    assert_eq!(actual.routing_visited_count, expected.routing_visited_count);
                    assert_eq!(
                        stage_reads(&reads, |stage| stage == VectorStage::Routing),
                        stage_reads(&baseline_reads, |stage| stage == VectorStage::Routing),
                    );
                    assert!(actual.work_charged <= expected.work_charged);
                    assert!(actual.candidates_scored <= expected.candidates_scored);
                    assert_eq!(adaptive.results, baseline.results);
                    assert_eq!(actual.termination, expected.termination);
                    assert_eq!(reads, baseline_reads);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn adaptive_routing_uses_one_window_without_expansion() -> crate::Result<()> {
    let fixture = FixtureSpec::clustered(Metric::L2, 64, 64, 640)
        .router(RouterKind::Stacked)
        .paged()
        .build()?;
    let query = Lcg(42).vector(64, 0.5);
    let adaptive = fixture.search(
        &AllQuery,
        &query,
        640,
        ProbeParams {
            max_probe_fraction: 1.0,
            located: LocatedPath::Off,
            quantized_recall: Some(crate::vector::QuantizedRecallParams {
                target: 0.95,
                recall_exponent: 1.25,
            }),
            ..Default::default()
        },
    )?;
    assert_eq!(
        adaptive.stats[0].termination,
        crate::vector::ProbeTermination::Exhausted
    );
    assert_eq!(adaptive.stats[0].routing_visited_count, 2);
    assert_eq!(adaptive.stats[0].postings_row, 2);
    assert!(adaptive.stats[0].candidates_scored < 640);
    Ok(())
}

#[test]
fn adaptive_bypass_is_independent_of_ceiling_headroom() -> crate::Result<()> {
    let fixture = FixtureSpec::clustered(Metric::L2, 64, 64, 6400)
        .label("few", |i| i % 200 == 0)
        .label("some", |i| i % 10 == 0)
        .build()?;
    let query = Lcg(42).vector(64, 0.5);
    for (filter, path) in [
        (fixture.filter("few"), AccessPath::Located),
        (fixture.filter("some"), AccessPath::Routed),
    ] {
        let mut cap = None;
        for max_probe_fraction in [0.1, 0.25, 1.0] {
            let result = fixture.search(
                &filter,
                &query,
                10,
                ProbeParams {
                    max_probe_fraction,
                    quantized_recall: Some(crate::vector::QuantizedRecallParams {
                        target: 0.95,
                        recall_exponent: 3.0,
                    }),
                    ..Default::default()
                },
            )?;
            assert_eq!(result.stats[0].access_path, path);
            assert_eq!(
                *cap.get_or_insert(result.stats[0].located_cap),
                result.stats[0].located_cap
            );
        }
    }
    Ok(())
}

#[test]
fn quantized_recall_rejects_invalid_parameters() {
    use crate::vector::QuantizedRecallParams;
    let stop = QuantizedRecallParams {
        target: 0.95,
        recall_exponent: 1.5,
    };
    for target in [0.0, -1.0, 1.1, f64::NAN, f64::INFINITY] {
        assert!(ProbeParams {
            quantized_recall: Some(QuantizedRecallParams { target, ..stop }),
            ..Default::default()
        }
        .resolved_work_budget(MATRIX_CLUSTERS, MATRIX_DOCS)
        .is_err());
    }
    assert!(stop.admission_fraction().unwrap() > 0.0);
    assert_eq!(
        QuantizedRecallParams {
            target: 1.0,
            ..stop
        }
        .admission_fraction(),
        None
    );
    for recall_exponent in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(ProbeParams {
            quantized_recall: Some(QuantizedRecallParams {
                recall_exponent,
                ..stop
            }),
            ..Default::default()
        }
        .resolved_work_budget(MATRIX_CLUSTERS, MATRIX_DOCS)
        .is_err());
    }
    for target in [0.9, 0.95, 0.99] {
        let calibrated = QuantizedRecallParams { target, ..stop };
        let raw = QuantizedRecallParams {
            target: 1.0 - (1.0 - target).powf(stop.recall_exponent),
            recall_exponent: 1.0,
        };
        assert!(
            (calibrated.admission_fraction().unwrap() - raw.admission_fraction().unwrap()).abs()
                < 1e-10
        );
        let window = |recall| {
            ProbeParams {
                quantized_recall: Some(recall),
                ..Default::default()
            }
            .router_k(1000.0, 0.1, 0.1, 1000)
        };
        assert_eq!(window(calibrated), window(raw));
        assert!(
            window(calibrated)
                > window(QuantizedRecallParams {
                    target,
                    recall_exponent: 1.0,
                })
        );
    }
    assert!(QuantizedRecallParams {
        target: 1.0 - f64::EPSILON,
        recall_exponent: 100.0
    }
    .admission_fraction()
    .unwrap()
    .is_finite());
}

#[derive(Clone, Copy, Debug)]
enum Representation {
    Flat,
    FullPrecision,
    Quantized(&'static [u8]),
}

fn one_hot(dim: usize, axis: usize, value: f32) -> Vec<f32> {
    let mut vector = vec![0.0; dim];
    vector[axis] = value;
    vector
}

fn matrix_vector(metric: Metric, ordinal: usize) -> Vec<f32> {
    let mut rng = Lcg(ordinal as u64 * 7919 + 1);
    let mut vector = one_hot(MATRIX_DIM, ordinal % MATRIX_CLUSTERS, 2.0);
    for (value, noise) in vector.iter_mut().zip(rng.vector(MATRIX_DIM, 0.6)) {
        *value += noise;
    }
    if metric == Metric::Dot {
        let scale = 0.5 + (ordinal % 7) as f32 * 0.15;
        vector.iter_mut().for_each(|value| *value *= scale);
    }
    vector
}

fn matrix_queries() -> Vec<Vec<f32>> {
    let mut rng = Lcg(42);
    let mut near = one_hot(MATRIX_DIM, 0, 2.0);
    let mut between = one_hot(MATRIX_DIM, 2, 1.2);
    between[3] = 1.2;
    for query in [&mut near, &mut between] {
        for (value, noise) in query.iter_mut().zip(rng.vector(MATRIX_DIM, 0.3)) {
            *value += noise;
        }
    }
    vec![near, between]
}

fn matrix_fixture(
    metric: Metric,
    representation: Representation,
    deletes: bool,
) -> crate::Result<Fixture> {
    let vectors: Vec<Vec<f32>> = (0..MATRIX_DOCS)
        .map(|ordinal| matrix_vector(metric, ordinal))
        .collect();
    let mut spec = match representation {
        Representation::Flat => {
            FixtureSpec::flat(metric, MATRIX_DIM, vectors.into_iter().map(Some).collect())
        }
        _ => FixtureSpec::ivf(
            metric,
            (0..MATRIX_CLUSTERS)
                .map(|cluster| one_hot(MATRIX_DIM, cluster, 2.0))
                .collect(),
            vectors,
        ),
    };
    if let Representation::Quantized(schedule) = representation {
        spec = spec.schedule(schedule);
    }
    spec = spec
        .segments(2)
        .without_vectors(|ordinal| ordinal % 37 == 5)
        .label("half", |ordinal| ordinal % 2 == 0)
        .label("rare", |ordinal| ordinal % 97 == 7)
        .label("doomed", |ordinal| (60..63).contains(&ordinal))
        // 5, 5, 5 and 40 matches in the four commits of 60 docs: whichever two commits merge,
        // one segment holds at most 10 matches and the other at least 45.
        .label("lopsided", |ordinal| {
            ordinal % 60 < if ordinal >= 180 { 40 } else { 5 }
        });
    if deletes {
        spec = spec.deletes(&[3, 60, 61, 62, 130, 131]);
    }
    spec.build()
}

/// Every touched row lands in exactly one bucket: filtered, dead or scored.
fn assert_identities(stats: &ProbeStats, case: &str) {
    assert_eq!(
        stats.vectors_visited,
        stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
        "{case}: visited must equal filtered + dead + scored ({stats:?})"
    );
}

/// Hits as `(score, address)`, best first.
type Hits = Vec<(Score, DocAddress)>;

/// The global top `k` with every segment searched directly under `forced`, and each segment's
/// stats.
fn forced_top_k(
    fixture: &Fixture,
    filter: &dyn Query,
    query: &[f32],
    k: usize,
    params: &ProbeParams,
    forced: PlanOverride,
) -> crate::Result<(Hits, Vec<ProbeStats>)> {
    let weight = fixture.weight(filter)?;
    let segments = fixture.index.reader()?.searcher().segment_readers().len();
    let mut hits = Vec::new();
    let mut stats = Vec::new();
    for segment in 0..segments {
        let (segment_hits, segment_stats) = fixture.run_planned(
            segment,
            weight.as_ref(),
            query,
            k,
            params.clone(),
            Some(forced),
            true,
        )?;
        assert_identities(&segment_stats, "forced plan");
        hits.extend(segment_hits);
        stats.push(segment_stats);
    }
    hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    hits.truncate(k);
    Ok((hits, stats))
}

/// The global top `k` with every segment searched directly under the planner, with admission
/// pruning on or off, and each segment's stats.
fn planned_top_k(
    fixture: &Fixture,
    filter: &dyn Query,
    query: &[f32],
    k: usize,
    params: &ProbeParams,
    admission_pruning: bool,
) -> crate::Result<(Hits, Vec<ProbeStats>)> {
    planned_top_k_with(fixture, filter, query, k, params, admission_pruning, None)
}

/// [`planned_top_k`] with every segment seeded from `collector_topk` when given.
fn planned_top_k_with(
    fixture: &Fixture,
    filter: &dyn Query,
    query: &[f32],
    k: usize,
    params: &ProbeParams,
    admission_pruning: bool,
    collector_topk: Option<Arc<Mutex<TopK>>>,
) -> crate::Result<(Hits, Vec<ProbeStats>)> {
    let weight = fixture.weight(filter)?;
    let segments = fixture.index.reader()?.searcher().segment_readers().len();
    let mut hits = Vec::new();
    let mut stats = Vec::new();
    for segment in 0..segments {
        let (segment_hits, segment_stats) = fixture.run_planned_with(
            segment,
            weight.as_ref(),
            query,
            k,
            params.clone(),
            None,
            admission_pruning,
            collector_topk.clone(),
        )?;
        assert_identities(&segment_stats, "admission pruning");
        hits.extend(segment_hits);
        stats.push(segment_stats);
    }
    hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    hits.truncate(k);
    Ok((hits, stats))
}

/// The rows of `expected` missing from `seeded`, a run seeded with the k-th exact score `seed`:
/// each with its upper endpoint at the first layer of `stats`' traces where it fell below the
/// seed (`None` if none did), and its exact score.
fn seeded_drops(
    fixture: &Fixture,
    expected: &Hits,
    seeded: &Hits,
    stats: &[ProbeStats],
    seed: f32,
) -> crate::Result<Vec<(DocAddress, Option<f32>, Score)>> {
    let searcher = fixture.index.reader()?.searcher();
    let mut drops = Vec::new();
    for &(score, address) in expected {
        if seeded.iter().any(|&(_, kept)| kept == address) {
            continue;
        }
        let row = searcher
            .segment_reader(address.segment_ord)
            .vector_index(fixture.field)?
            .row_id(address.doc_id)?
            .expect("a result has a vector");
        let trace = &stats[address.segment_ord as usize].quantized_trace;
        let upper = trace.layer_columns.iter().find_map(|layer| {
            layer
                .iter()
                .find(|&&(traced, _)| traced == row)
                .map(|&(_, bits)| {
                    Estimate(f32::from_bits(bits[2]))
                        .upper(f32::from_bits(bits[3]), QUANTIZED_BOUNDARY_KAPPA)
                        .0
                })
                .filter(|&upper| upper < seed)
        });
        drops.push((address, upper, score));
    }
    Ok(drops)
}

/// Oracle-matrix cases compared against a seeded shared bound, and those where the seed pruned
/// more rows at admission than the scan's own bound.
static SEEDED_CASES: AtomicUsize = AtomicUsize::new(0);
static SEEDED_CASES_PRUNING_MORE: AtomicUsize = AtomicUsize::new(0);
/// Oracle-matrix windows compared against a seeded participants' slot, those where the seed
/// pruned at least one row, and the window rows the seed dropped.
static SLOT_CASES: AtomicUsize = AtomicUsize::new(0);
static SLOT_CASES_PRUNED: AtomicUsize = AtomicUsize::new(0);
static SLOT_DROPS: AtomicUsize = AtomicUsize::new(0);

/// A row of the unseeded window the seeded run lost: where it was dropped, its upper endpoint
/// there, and its exact score.
struct SlotDrop {
    address: DocAddress,
    site: String,
    upper: Option<f32>,
    score: Score,
}

/// The rows of `unseeded`'s window missing from `seeded`'s, each with its last upper endpoint in
/// the seeded run: from the last layer whose candidates held it, or layer 0's of the unseeded run
/// when admission dropped it.
fn slot_drops(
    fixture: &Fixture,
    unseeded: &VectorSimilarityFruit,
    seeded: &VectorSimilarityFruit,
) -> crate::Result<Vec<SlotDrop>> {
    let searcher = fixture.index.reader()?.searcher();
    let upper = |bits: [u32; 4]| {
        Estimate(f32::from_bits(bits[2]))
            .upper(f32::from_bits(bits[3]), QUANTIZED_BOUNDARY_KAPPA)
            .0
    };
    let mut drops = Vec::new();
    for &(score, address) in &unseeded.results {
        if seeded.results.iter().any(|&(_, kept)| kept == address) {
            continue;
        }
        let row = searcher
            .segment_reader(address.segment_ord)
            .vector_index(fixture.field)?
            .row_id(address.doc_id)?
            .expect("a result has a vector");
        let segment = address.segment_ord as usize;
        let trace = &seeded.stats[segment].quantized_trace;
        let last = trace
            .layer_columns
            .iter()
            .enumerate()
            .rev()
            .find_map(|(layer, columns)| {
                columns
                    .iter()
                    .find(|&&(traced, _)| traced == row)
                    .map(|&(_, bits)| (layer, upper(bits)))
            });
        let (site, upper) = match last {
            Some((layer, upper)) if trace.boundary_rows[layer].contains(&row) => {
                if layer + 1 < trace.layer_columns.len() {
                    (format!("recheck at layer {}", layer + 1), Some(upper))
                } else {
                    ("exact recheck".to_string(), Some(upper))
                }
            }
            Some((layer, upper)) => (format!("boundary {layer}"), Some(upper)),
            None => {
                let unseeded_trace = &unseeded.stats[segment].quantized_trace;
                let layer0 = unseeded_trace.layer_columns.first().and_then(|columns| {
                    columns
                        .iter()
                        .find(|&&(traced, _)| traced == row)
                        .map(|&(_, bits)| upper(bits))
                });
                ("admission".to_string(), layer0)
            }
        };
        drops.push(SlotDrop {
            address,
            site,
            upper,
            score,
        });
    }
    Ok(drops)
}

/// Stores `score` in an empty `slot`.
fn seed_slot(slot: &AtomicSharedThreshold, score: Score) {
    while slot
        .try_update(&None, (score, SegmentOrdinal::MAX))
        .is_err()
    {}
}

/// A participants' slot seeded with the k-th exact score of the collector's window,
/// k = `limit + offset`, prunes only rows that cannot enter the window. The seeded run returns the
/// same window, unless a dropped row's last upper endpoint fell below its exact score. A tighter
/// slot may open fewer clusters.
fn assert_slot_seeded(
    fixture: &Fixture,
    filter: &dyn Query,
    query: &[f32],
    limit: usize,
    offset: usize,
    params: &ProbeParams,
    case: &str,
) -> crate::Result<()> {
    let case = format!("{case} offset={offset}");
    let unseeded = fixture.search_with(filter, query, limit, offset, params.clone(), None)?;
    // A full window means k results exist; its last score is the k-th.
    let Some(&(seed, _)) = unseeded.results.get(limit - 1) else {
        return Ok(());
    };
    let slot = Arc::new(AtomicSharedThreshold::default());
    seed_slot(&slot, seed);
    let seeded = fixture.search_with(
        filter,
        query,
        limit,
        offset,
        params.clone(),
        Some(slot.clone()),
    )?;
    SLOT_CASES.fetch_add(1, Ordering::Relaxed);
    if seeded.stats.iter().any(|stats| {
        stats
            .layers
            .get(0)
            .is_some_and(|layer| layer.admission_pruned() > 0)
            || (1..)
                .map_while(|i| stats.layers.get(i))
                .any(|layer| layer.recheck_dropped() > 0)
            || stats.exact_recheck_dropped > 0
    }) {
        SLOT_CASES_PRUNED.fetch_add(1, Ordering::Relaxed);
    }
    assert!(
        slot.load().is_some_and(|(score, _)| score >= seed),
        "{case}"
    );
    assert_eq!(seeded.stats.len(), unseeded.stats.len(), "{case}");
    // A seeded slot may open fewer clusters. The window is what has to match.
    if seeded.results != unseeded.results {
        let drops = slot_drops(fixture, &unseeded, &seeded)?;
        assert!(
            !drops.is_empty(),
            "{case}: the seeded slot reordered the window"
        );
        for drop in &drops {
            SLOT_DROPS.fetch_add(1, Ordering::Relaxed);
            let SlotDrop {
                address,
                site,
                upper,
                score,
            } = drop;
            let row = format!("{address:?} at {site}: upper {upper:?}, exact {score}");
            eprintln!("{case}: seeded slot {seed} dropped {row}");
            assert!(
                upper.is_some_and(|upper| upper < *score),
                "{case}: seeded slot {seed} dropped {row}"
            );
        }
    }
    Ok(())
}

/// Every forced plan of the oracle matrix: each read, every cluster exact at admission, and
/// mixed exact clusters at admission and at layer 1.
fn forced_plans() -> [(&'static str, PlanOverride); 4] {
    [
        ("band", PlanOverride::read(LayerRead::Band)),
        ("sparse", PlanOverride::read(LayerRead::Sparse)),
        ("all exact", PlanOverride::exact(|_, _| true)),
        (
            "mixed exact",
            PlanOverride::exact(|cluster, layer| (cluster + layer) % 2 == 0),
        ),
    ]
}

/// The probe parameters of each path: routed, and on IVF segments located under a cap of a
/// tenth of a segment with the exact plan on, as pg_search runs it.
fn paths(representation: Representation) -> Vec<(&'static str, ProbeParams)> {
    let routed = routed_exhaustive_params(MATRIX_CLUSTERS);
    if matches!(representation, Representation::Flat) {
        return vec![("routed", routed)];
    }
    let located = ProbeParams {
        located: LocatedPath::MaxSelectivity(Selectivity::new(0.1).unwrap()),
        exact_plan: true,
        ..routed.clone()
    };
    vec![("routed", routed), ("located", located)]
}

/// One oracle-matrix case: the collector's top k, and on quantized segments every forced plan's,
/// equal the oracle's.
fn assert_case(
    fixture: &Fixture,
    representation: Representation,
    filter: &dyn Query,
    query: &[f32],
    k: usize,
    params: &ProbeParams,
    case: &str,
) -> crate::Result<Vec<ProbeStats>> {
    let fruit = fixture.search(filter, query, k, params.clone())?;
    let oracle = fixture.oracle(filter, query, k)?;
    assert_eq!(fruit.results, oracle, "{case}");
    for stats in &fruit.stats {
        assert_identities(stats, case);
    }
    for offset in [0, 3] {
        assert_slot_seeded(fixture, filter, query, k, offset, params, case)?;
    }
    if !matches!(representation, Representation::Quantized(_)) {
        return Ok(fruit.stats);
    }
    for (name, forced) in forced_plans() {
        let (hits, stats) = forced_top_k(fixture, filter, query, k, params, forced)?;
        assert_eq!(hits, oracle, "{case} forced={name}");
        if name != "all exact" {
            continue;
        }
        for stats in &stats {
            // A segment where nothing matches reports no layers.
            let Some(layer0) = stats.layers.get(0) else {
                continue;
            };
            assert_eq!(layer0.exact_clusters(), stats.postings_row, "{case}");
            assert_eq!(layer0.exact_rows(), stats.candidates_scored, "{case}");
            assert_eq!(layer0.scored(), 0, "{case}");
            assert_eq!(stats.exact_rows, stats.candidates_scored, "{case}");
        }
    }
    // Admission pruning drops only rows every boundary would drop: results, survivors and their
    // column bits must match a run without it.
    let (pruned, pruned_stats) = planned_top_k(fixture, filter, query, k, params, true)?;
    let (unpruned, unpruned_stats) = planned_top_k(fixture, filter, query, k, params, false)?;
    assert_eq!(pruned, oracle, "{case} pruned");
    assert_eq!(pruned, unpruned, "{case} admission pruning changed hits");
    for (p, u) in pruned_stats.iter().zip(&unpruned_stats) {
        let (pt, ut) = (&p.quantized_trace, &u.quantized_trace);
        assert_eq!(pt.boundary_rows, ut.boundary_rows, "{case}");
        assert_eq!(pt.exact_phase_rows, ut.exact_phase_rows, "{case}");
        // Layer 0's pre-boundary candidate list shrinks under pruning; later layers see
        // identical survivors with identical column bits.
        assert_eq!(
            pt.estimate_rows.get(1..),
            ut.estimate_rows.get(1..),
            "{case}"
        );
        assert_eq!(
            pt.layer_columns.get(1..),
            ut.layer_columns.get(1..),
            "{case}"
        );
    }
    // A collector top-k seeded with the result's k-th exact score prunes only rows that cannot
    // enter the top k: the seeded run returns the same hits, unless a dropped row's interval
    // missed its exact score.
    if let Some(&(seed, _)) = pruned.get(k - 1) {
        let collector_topk = Arc::new(Mutex::new(TopK::new(k)));
        collector_topk.lock().unwrap().merge_exact((0..k).map(|i| {
            (
                crate::vector::write::RowAddress {
                    segment_ord: SegmentOrdinal::MAX,
                    row_id: i,
                },
                seed,
            )
        }));
        let (seeded, seeded_stats) = planned_top_k_with(
            fixture,
            filter,
            query,
            k,
            params,
            true,
            Some(collector_topk),
        )?;
        SEEDED_CASES.fetch_add(1, Ordering::Relaxed);
        let admission_pruned = |stats: &[ProbeStats]| -> usize {
            stats
                .iter()
                .filter_map(|stats| stats.layers.get(0))
                .map(LayerProbeStats::admission_pruned)
                .sum()
        };
        if admission_pruned(&seeded_stats) > admission_pruned(&pruned_stats) {
            SEEDED_CASES_PRUNING_MORE.fetch_add(1, Ordering::Relaxed);
        }
        if seeded != pruned {
            let drops = seeded_drops(fixture, &pruned, &seeded, &pruned_stats, seed)?;
            assert!(!drops.is_empty(), "{case} seeded bound reordered hits");
            for (address, upper, score) in &drops {
                eprintln!("{case}: seeded drop {address:?} upper {upper:?} exact {score}");
                assert!(
                    upper.is_some_and(|upper| upper < *score),
                    "{case} seeded bound {seed} dropped {address:?}: upper {upper:?}, exact \
                     {score}"
                );
            }
        }
    }
    Ok(fruit.stats)
}

/// Under an exhaustive probe every representation, path, filter, delete pattern, k and forced
/// read plan returns exactly the oracle's top k, across segments; a filter matching few docs in
/// one segment and many in the other takes both paths.
#[test]
fn scan_matches_oracle() -> crate::Result<()> {
    let representations = [
        Representation::Flat,
        Representation::FullPrecision,
        Representation::Quantized(&[1]),
        Representation::Quantized(&[1, 4]),
    ];
    for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
        for representation in representations {
            for deletes in [false, true] {
                let fixture = matrix_fixture(metric, representation, deletes)?;
                let segments = fixture.index.reader()?.searcher().segment_readers().len();
                assert_eq!(segments, 2, "the matrix index holds two segments");
                let filters: Vec<(&str, Box<dyn Query>)> = vec![
                    ("none", Box::new(AllQuery)),
                    ("half", Box::new(fixture.filter("half"))),
                    ("rare", Box::new(fixture.filter("rare"))),
                    ("empty", Box::new(fixture.filter("absent"))),
                    ("doomed", Box::new(fixture.filter("doomed"))),
                    ("lopsided", Box::new(fixture.filter("lopsided"))),
                ];
                for (path, params) in paths(representation) {
                    for (query_ord, query) in matrix_queries().iter().enumerate() {
                        for (filter_name, filter) in &filters {
                            for k in [10, 1000] {
                                let case = format!(
                                    "{metric:?} {representation:?} {path} deletes={deletes} \
                                     query={query_ord} filter={filter_name} k={k}"
                                );
                                let stats = assert_case(
                                    &fixture,
                                    representation,
                                    filter.as_ref(),
                                    query,
                                    k,
                                    &params,
                                    &case,
                                )?;
                                if path == "located" && *filter_name == "lopsided" {
                                    let paths: Vec<_> =
                                        stats.iter().map(|stats| stats.access_path).collect();
                                    assert!(paths.contains(&AccessPath::Located), "{case}");
                                    assert!(paths.contains(&AccessPath::Routed), "{case}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let (seeded, pruning_more) = (
        SEEDED_CASES.load(Ordering::Relaxed),
        SEEDED_CASES_PRUNING_MORE.load(Ordering::Relaxed),
    );
    eprintln!("seeded collector top-k cases: {seeded}, pruning more at admission: {pruning_more}");
    assert!(
        pruning_more > 0,
        "no seeded case pruned beyond the scan's own bound"
    );
    let (slot_cases, slot_pruned, slot_drops) = (
        SLOT_CASES.load(Ordering::Relaxed),
        SLOT_CASES_PRUNED.load(Ordering::Relaxed),
        SLOT_DROPS.load(Ordering::Relaxed),
    );
    eprintln!(
        "seeded slot windows: {slot_cases}, pruning under the seed: {slot_pruned}, rows dropped: \
         {slot_drops}; two segments per index"
    );
    assert!(slot_pruned > 0, "no seeded slot pruned any rows");
    Ok(())
}

// --- Routing edge cases ---

/// A query nearest one centroid whose true nearest neighbour sits in the other cluster: a
/// one-cluster ceiling misses it and an exhaustive probe finds it.
#[test]
fn adaptive_probe_finds_nn_outside_nearest_cluster() -> crate::Result<()> {
    let fixture = FixtureSpec::ivf(
        Metric::L2,
        vec![vec![0.0, 0.0], vec![10.0, 10.0]],
        vec![
            vec![0.0, -10.0],
            vec![-10.0, 0.0],
            // Just past the bisector x + y = 10, so in the far cluster yet nearest the query.
            vec![5.0, 5.01],
            vec![10.0, 10.0],
        ],
    )
    // Exact: Stacked's hard 25% top prune keeps only one of two lists even under Exhaustive.
    .router(RouterKind::Exact)
    .build()?;
    let query = [1.0, 1.0];
    let oracle = fixture.oracle(&AllQuery, &query, 1)?;
    assert_eq!(
        fixture.ordinal(oracle[0].1)?,
        2,
        "the trap doc is the true nearest"
    );
    let one_cluster = ProbeParams {
        max_probe_fraction: 0.5,
        min_probe_clusters: 1,
        ..Default::default()
    };
    let missed = fixture.search(&AllQuery, &query, 1, one_cluster)?;
    assert_ne!(
        fixture.ordinal(missed.results[0].1)?,
        2,
        "a one-cluster ceiling probes only the nearest cluster"
    );
    let found = fixture.search(&AllQuery, &query, 1, exhaustive_params(2))?;
    assert_eq!(fixture.ordinal(found.results[0].1)?, 2);
    Ok(())
}

/// Below the ceiling the router still reaches each query's home cluster among 16 separated
/// clusters: the routed top k equals the oracle within two probes.
#[test]
fn routes_home_below_ceiling() -> crate::Result<()> {
    let centroids: Vec<Vec<f32>> = (0..16)
        .map(|i| vec![(i % 4) as f32 * 10.0, (i / 4) as f32 * 10.0])
        .collect();
    let vectors = (0..64)
        .map(|i| {
            let centroid = &centroids[i / 4];
            let offset = (i % 4) as f32 * 0.01;
            vec![centroid[0] + offset, centroid[1] + offset]
        })
        .collect();
    let fixture = FixtureSpec::ivf(Metric::L2, centroids.clone(), vectors).build()?;
    let params = ProbeParams {
        max_probe_fraction: 0.1,
        min_probe_clusters: 1,
        ..Default::default()
    };
    for centroid in centroids.iter().step_by(3) {
        let query = [centroid[0] + 0.3, centroid[1] - 0.2];
        let fruit = fixture.search(&AllQuery, &query, 3, params.clone())?;
        assert_eq!(
            fruit.results,
            fixture.oracle(&AllQuery, &query, 3)?,
            "{query:?}"
        );
        assert!(
            fruit.stats[0].clusters_probed() <= 2,
            "the ceiling bounds the probes: {:?}",
            fruit.stats[0]
        );
    }
    Ok(())
}

/// A single-centroid segment ranks its one cluster and returns the oracle's top k.
#[test]
fn single_centroid_segment_routes() -> crate::Result<()> {
    let fixture = FixtureSpec::ivf(
        Metric::L2,
        vec![vec![0.0, 0.0]],
        (0..5).map(|i| vec![i as f32 * 0.01, 0.0]).collect(),
    )
    .build()?;
    let query = [0.0, 0.0];
    let fruit = fixture.search(&AllQuery, &query, 3, ProbeParams::default())?;
    assert_eq!(fruit.results, fixture.oracle(&AllQuery, &query, 3)?);
    assert_eq!(fruit.stats[0].clusters_probed(), 1);
    Ok(())
}

// --- Read equivalence ---

/// Sparse layer reads summed over every layer.
fn sparse_clusters(stats: &ProbeStats) -> usize {
    (0..)
        .map_while(|layer| stats.layers.get(layer))
        .map(LayerProbeStats::sparse_clusters)
        .sum()
}

/// Band and sparse reads give bit-identical candidate columns, estimates, boundaries and results
/// at every layer, for every metric, in RAM and on pages, with and without exact clusters; every
/// plan returns the oracle's top k.
#[test]
fn read_plans_are_bit_identical() -> crate::Result<()> {
    let shapes: [(&[u8], usize, usize, usize, bool); 5] = [
        (&[1], 64, 8, 800, false),
        (&[1, 1], 64, 8, 800, false),
        (&[1, 4], 64, 8, 800, false),
        (&[2, 4], 64, 8, 800, false),
        (&[1, 4], 1024, 2, 1200, true),
    ];
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        for (schedule, dim, clusters, docs, paged) in shapes {
            let mut spec = FixtureSpec::clustered(metric, dim, clusters, docs)
                .schedule(schedule)
                .without_vectors(|ordinal| ordinal % 101 == 7)
                .label("few", |ordinal| ordinal % 40 == 3)
                .label("some", |ordinal| ordinal % 4 == 1);
            if paged {
                spec = spec.paged();
            }
            let fixture = spec.build()?;
            let query = Lcg(5).vector(dim, 0.5);
            let filters: [(&str, Box<dyn Query>, bool); 3] = [
                ("few", Box::new(fixture.filter("few")), true),
                ("some", Box::new(fixture.filter("some")), true),
                ("none", Box::new(AllQuery), schedule.len() > 1),
            ];
            for (name, filter, expect_sparse) in filters {
                let case = format!("{metric:?} {schedule:?} dim={dim} filter={name}");
                let weight = fixture.weight(filter.as_ref())?;
                let run = |forced| {
                    fixture.run_planned(
                        0,
                        weight.as_ref(),
                        &query,
                        10,
                        routed_exhaustive_params(clusters),
                        Some(forced),
                        true,
                    )
                };
                let oracle = fixture.oracle(filter.as_ref(), &query, 10)?;
                // Each read alone, then with mixed exact clusters at admission and at layer 1.
                let mixed: ExactRule = |cluster, layer| (cluster + layer) % 2 == 0;
                for exact in [None, Some(mixed)] {
                    let case = format!("{case} exact={}", exact.is_some());
                    let forced = |read| PlanOverride {
                        read: Some(read),
                        exact,
                    };
                    let (band_hits, band) = run(forced(LayerRead::Band))?;
                    let (sparse_hits, sparse) = run(forced(LayerRead::Sparse))?;
                    assert_eq!(sparse_clusters(&band), 0, "{case}");
                    if exact.is_none() {
                        assert_eq!(sparse_clusters(&sparse) > 0, expect_sparse, "{case}");
                    }
                    let (band, sparse) = (&band.quantized_trace, &sparse.quantized_trace);
                    assert!(!band.layer_columns.is_empty(), "{case}");
                    assert_eq!(sparse.layer_columns, band.layer_columns, "{case}");
                    assert_eq!(sparse.estimate_rows, band.estimate_rows, "{case}");
                    assert_eq!(sparse.boundary_rows, band.boundary_rows, "{case}");
                    assert_eq!(sparse_hits, band_hits, "{case}");
                    assert_eq!(band_hits, oracle, "{case}");
                }
            }
        }
    }
    Ok(())
}

// --- Reads ---

const READS_DIM: usize = 64;
const READS_DOCS: usize = 1200;

/// Clusters 1..=3 of four one-hot centroids; cluster 0 stays empty, cluster 1 holds doc 0 alone.
fn reads_cluster(doc: usize) -> usize {
    if doc == 0 {
        1
    } else {
        2 + doc % 2
    }
}

fn reads_vector(doc: usize) -> Vec<f32> {
    let mut vector: Vec<f32> = (0..READS_DIM)
        .map(|i| ((doc * READS_DIM + i) as f32 * 0.071).sin() * 0.025)
        .collect();
    vector[reads_cluster(doc)] += 1.0;
    vector
}

fn has_vector(doc: usize) -> bool {
    doc == 0 || doc % 97 != 0
}

fn reads_fixture(metric: Metric, schedule: &[u8], deletes: &[usize]) -> crate::Result<Fixture> {
    FixtureSpec::ivf(
        metric,
        (0..4).map(|axis| one_hot(READS_DIM, axis, 1.0)).collect(),
        (0..READS_DOCS).map(reads_vector).collect(),
    )
    .schedule(schedule)
    .paged()
    .router(RouterKind::Exact)
    .without_vectors(|doc| !has_vector(doc))
    .label("vector", has_vector)
    .deletes(deletes)
    .build()
}

/// Probe parameters that open every cluster the router hands over, on the routed path with no
/// cluster scored exactly before the exact phase.
fn every_cluster() -> ProbeParams {
    ProbeParams {
        max_probe_fraction: 1.0,
        min_probe_clusters: 4,
        exact_plan: false,
        located: LocatedPath::Off,
        ..Default::default()
    }
}

/// The ranges `log` requested while `stage` ran.
fn stage_reads<'a>(
    log: &'a ReadLog,
    stage: impl Fn(VectorStage) -> bool + 'a,
) -> Vec<&'a Range<usize>> {
    log.iter()
        .filter(|(running, _)| stage(*running))
        .map(|(_, range)| range)
        .collect()
}

fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

/// The admission stage of a schedule: the layer-0 scan, or the full-precision scan.
fn admission(schedule: &[u8]) -> impl Fn(VectorStage) -> bool {
    let quantized = !schedule.is_empty();
    move |stage| {
        if quantized {
            stage == VectorStage::LayerScan(0)
        } else {
            stage == VectorStage::ExactScan
        }
    }
}

/// An empty cluster counts as opened and skipped, and reads no band.
fn empty_cluster_skips_its_band() -> crate::Result<()> {
    let fixture = reads_fixture(Metric::L2, &[1], &[101, 202])?;
    let ranges = fixture.column_ranges()?;
    assert_eq!(ranges.docs[0], None, "cluster 0 is empty");
    fixture.clear_reads();
    let fruit = fixture.search_clusters(
        &[0],
        &AllQuery,
        &reads_vector(42),
        READS_DOCS + 1,
        every_cluster(),
    )?;
    let stats = &fruit.stats[0];
    assert!(fruit.results.is_empty());
    assert_eq!(stats.clusters_skipped_empty, 1);
    assert_eq!(stats.postings_skipped, 1);
    assert_eq!(stats.clusters_probed(), 1);
    assert_eq!(stats.vectors_visited, 0);
    assert_eq!(stats.layers.get(0).unwrap().io.reads, 0);
    assert!(stage_reads(&fixture.reads(), |stage| stage == VectorStage::LayerScan(0)).is_empty());
    Ok(())
}

/// A cluster whose rows all fail the filter reads its DocIds and nothing else, counts as skipped
/// and empty, and charges only its open.
fn filtered_cluster_reads_only_doc_ids() -> crate::Result<()> {
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        for schedule in [&[][..], &[1][..]] {
            let fixture = reads_fixture(metric, schedule, &[])?;
            let case = format!("{metric:?} {schedule:?}");
            let ranges = fixture.column_ranges()?;
            let searcher = fixture.index.reader()?.searcher();
            let vectors = first_segment(&searcher).1.vector_index(fixture.field)?;
            let ivf = vectors.index().unwrap();
            let ivf_rows = ivf.cluster_range(2).len();
            let (_, _, open) =
                every_cluster().resolved_work_budget(ivf.num_clusters(), ivf.num_docs())?;
            fixture.clear_reads();
            // Doc 0 is cluster 1's only row; cluster 2 holds none of the filter's matches.
            let fruit = fixture.search_clusters(
                &[2],
                &fixture.filter("d0"),
                &reads_vector(42),
                10,
                every_cluster(),
            )?;
            let stats = &fruit.stats[0];
            assert!(fruit.results.is_empty(), "{case}");
            assert_eq!(stats.postings_row, 0, "{case}");
            assert_eq!(stats.postings_skipped, 1, "{case}");
            assert_eq!(stats.clusters_skipped_empty, 1, "{case}");
            assert_eq!(stats.vectors_visited, ivf_rows, "{case}");
            assert_eq!(stats.pruned_filter, ivf_rows, "{case}");
            assert_eq!(stats.candidates_scored, 0, "{case}");
            assert_eq!(stats.layer0_eligible, 0, "{case}");
            assert!(
                (stats.work_charged - open as f32).abs() <= f32::EPSILON,
                "{case}: only the open is charged ({stats:?})"
            );
            let log = fixture.reads();
            assert_eq!(
                stage_reads(&log, admission(schedule)),
                vec![ranges.docs[2].as_ref().unwrap()],
                "{case}: admission reads the DocIds column alone"
            );
            assert!(
                stage_reads(&log, |stage| matches!(
                    stage,
                    VectorStage::ExactFetch | VectorStage::ExactScore | VectorStage::LayerScan(1..)
                ))
                .is_empty(),
                "{case}: no row is read"
            );
        }
    }
    Ok(())
}

/// Filtered clusters read their DocIds column before any payload, and clusters without a match
/// read nothing more; at full precision the survivor's row is the only payload read.
fn filtered_clusters_read_doc_ids_before_payload() -> crate::Result<()> {
    for schedule in [&[][..], &[1, 4][..]] {
        let fixture = reads_fixture(Metric::L2, schedule, &[])?;
        let ranges = fixture.column_ranges()?;
        fixture.clear_reads();
        let fruit = fixture.search_clusters(
            &[0, 1, 2, 3],
            &fixture.filter("d0"),
            &reads_vector(0),
            READS_DOCS + 1,
            every_cluster(),
        )?;
        assert_eq!(fruit.results.len(), 1, "{schedule:?}");
        let log = fixture.reads();
        let scans = stage_reads(&log, |stage| {
            admission(schedule)(stage)
                || (schedule.is_empty()
                    && matches!(stage, VectorStage::ExactFetch | VectorStage::ExactScore))
        });
        for cluster in [2, 3] {
            let docs = ranges.docs[cluster].as_ref().unwrap();
            let rows = ranges.rows[cluster].as_ref().unwrap();
            let reads: Vec<_> = scans
                .iter()
                .filter(|range| overlaps(range, docs) || overlaps(range, rows))
                .collect();
            assert_eq!(reads, vec![&docs], "{schedule:?} cluster={cluster}");
        }
        if schedule.is_empty() {
            let mut expected: Vec<_> = ranges.docs.iter().flatten().collect();
            expected.push(ranges.rows[1].as_ref().unwrap());
            expected.sort_by_key(|range| range.start);
            let mut actual = scans.clone();
            actual.sort_by_key(|range| range.start);
            assert_eq!(
                actual, expected,
                "DocIds columns and the survivor's row only"
            );
        }
    }
    Ok(())
}

/// With no filter and no deletes, DocIds are read only for competitive exact scores.
fn open_gate_reads_doc_ids_only_for_competitive_survivors() -> crate::Result<()> {
    let fixture = reads_fixture(Metric::L2, &[1, 4], &[])?;
    let ranges = fixture.column_ranges()?;
    let query = reads_vector(42);
    fixture.clear_reads();
    let open = fixture.search(&AllQuery, &query, 3, every_cluster())?;
    let log = fixture.reads();
    for range in stage_reads(&log, |stage| matches!(stage, VectorStage::LayerScan(_))) {
        assert!(
            !ranges
                .docs
                .iter()
                .flatten()
                .any(|docs| overlaps(range, docs)),
            "a layer scan touched DocIds: {range:?}"
        );
    }
    let trace = fixture.doc_trace(&open.stats[0].quantized_trace)?;
    for (cluster, docs) in fixture.cluster_docs(0)?.iter().enumerate() {
        let Some(range) = &ranges.docs[cluster] else {
            continue;
        };
        let expected = usize::from(docs.iter().any(|doc| trace.exact_phase_docs.contains(doc)));
        let fetched = log
            .iter()
            .filter(|(stage, read)| *stage == VectorStage::ExactFetch && read == range)
            .count();
        assert!(
            fetched <= expected,
            "cluster={cluster} DocIds exact-phase reads"
        );
        if open
            .results
            .iter()
            .any(|(_, address)| docs.contains(&address.doc_id))
        {
            assert_eq!(fetched, 1, "cluster={cluster} winner DocIds");
        }
        assert_eq!(
            log.iter().filter(|(_, read)| read == range).count(),
            fetched,
            "cluster={cluster} DocIds reads outside the exact phase"
        );
        if let Some(position) = log.iter().position(|(_, read)| read == range) {
            let rows = ranges.rows[cluster].as_ref().unwrap();
            assert!(
                log[..position].iter().any(|(stage, read)| {
                    *stage == VectorStage::ExactFetch && overlaps(read, rows)
                }),
                "cluster={cluster} DocIds read before its vectors"
            );
        }
    }
    for layer in 0..2 {
        let requested = stage_reads(&log, |stage| stage == VectorStage::LayerScan(layer as u8));
        let io = open.stats[0].layers.get(layer).unwrap().io;
        assert_eq!(io.reads, requested.len() as u64, "layer {layer}");
        assert_eq!(
            io.bytes_read,
            requested
                .iter()
                .map(|range| range.len() as u64)
                .sum::<u64>(),
            "layer {layer}"
        );
    }
    let filtered = fixture.search(&fixture.filter("vector"), &query, 3, every_cluster())?;
    assert_eq!(open.results, filtered.results);
    assert_eq!(
        trace,
        fixture.doc_trace(&filtered.stats[0].quantized_trace)?,
        "the open gate scans exactly what an all-vectors filter scans"
    );
    Ok(())
}

/// A cluster whose rows are all deleted reads its DocIds and nothing past them.
fn all_dead_cluster_reads_only_doc_ids() -> crate::Result<()> {
    for schedule in [&[][..], &[1][..]] {
        // Doc 0 is cluster 1's only row.
        let fixture = reads_fixture(Metric::L2, schedule, &[0])?;
        let ranges = fixture.column_ranges()?;
        fixture.clear_reads();
        let fruit = fixture.search_clusters(
            &[1, 2, 3],
            &AllQuery,
            &reads_vector(0),
            READS_DOCS + 1,
            every_cluster(),
        )?;
        let stats = &fruit.stats[0];
        assert_eq!(stats.pruned_dead, 1, "{schedule:?}: {stats:?}");
        assert_eq!(
            stats.postings_skipped, 1,
            "{schedule:?}: the dead cluster: {stats:?}"
        );
        assert_eq!(stats.postings_row, 2, "{schedule:?}: {stats:?}");
        assert_identities(stats, "all dead");
        let docs = ranges.docs[1].as_ref().unwrap();
        let rows = ranges.rows[1].as_ref().unwrap();
        let cluster_reads: Vec<_> = fixture
            .reads()
            .into_iter()
            .filter(|(_, range)| overlaps(range, docs) || overlaps(range, rows))
            .map(|(_, range)| range)
            .collect();
        assert_eq!(cluster_reads, vec![docs.clone()], "{schedule:?}");
    }
    Ok(())
}

/// The flat path reads one row per filtered survivor and touches no probe counter.
fn flat_path_reads_one_row_per_survivor() -> crate::Result<()> {
    let docs = 40;
    let fixture = FixtureSpec::flat(
        Metric::L2,
        2,
        (0..docs).map(|i| Some(vec![i as f32 * 0.1, 1.0])).collect(),
    )
    .paged()
    .build()?;
    for percent in [0usize, 1, 50, 100] {
        let weight = DocsWeight {
            max_doc: docs as DocId,
            docs: (0..docs as DocId)
                .filter(|&doc| doc as usize * 100 < percent * docs)
                .collect(),
        };
        fixture.clear_reads();
        let (hits, stats) =
            fixture.run_segment(0, &weight, &[0.0, 0.0], docs, Default::default())?;
        let admitted = weight.docs.len();
        assert_eq!(hits.len(), admitted, "{percent}%");
        assert_eq!(stats.exact_rows, admitted, "{percent}%");
        assert_eq!(stats.vectors_visited, 0, "{percent}%");
        assert_eq!(stats.candidates_scored, 0, "{percent}%");
        assert_eq!(stats.clusters_probed(), 0, "{percent}%");
        assert_eq!(
            stage_reads(&fixture.reads(), |stage| stage == VectorStage::ExactScan).len(),
            admitted,
            "{percent}%: one row request per survivor"
        );
    }
    Ok(())
}

/// Exact scoring of a filtered full-precision segment reads each DocIds column once during
/// admission and touches only the pages of the selected rows.
fn exact_reads_touch_only_survivor_pages() -> crate::Result<()> {
    let fixture = reads_fixture(Metric::L2, &[], &[])?;
    let ranges: ColumnRanges = fixture.column_ranges()?;
    let filter = BooleanQuery::union(vec![
        Box::new(fixture.filter("d42")) as Box<dyn Query>,
        Box::new(fixture.filter("d242")),
    ]);
    let searcher = fixture.index.reader()?.searcher();
    let segment = first_segment(&searcher).1;
    let vectors = segment.vector_index(fixture.field)?;
    let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
    let mut selected = Vec::new();
    weight.for_each_no_score(segment, &mut |docs| selected.extend_from_slice(docs))?;
    let mut pages = BTreeSet::new();
    for doc in selected {
        fixture.clear_reads();
        vectors.vector_bytes(doc)?.unwrap();
        let (_, row) = fixture.reads().last().unwrap().clone();
        pages.extend(row.start / 8192..=(row.end - 1) / 8192);
    }
    assert!(pages.len() >= 2, "the survivors' rows span distinct pages");
    let query = reads_vector(42);
    fixture.clear_reads();
    let fruit = fixture.search(&filter, &query, READS_DOCS + 1, every_cluster())?;
    assert_eq!(
        fruit.results,
        fixture.oracle(&filter, &query, READS_DOCS + 1)?
    );
    let log = fixture.reads();
    let admission_reads = stage_reads(&log, |stage| stage == VectorStage::ExactScan);
    for docs in ranges.docs.iter().flatten() {
        assert_eq!(
            admission_reads
                .iter()
                .filter(|range| **range == docs)
                .count(),
            1,
            "each DocIds column is read once"
        );
    }
    assert!(
        admission_reads
            .iter()
            .all(|range| ranges.docs.iter().flatten().any(|docs| docs == *range)),
        "admission reads only DocIds"
    );
    let mut payload = BTreeSet::new();
    for range in stage_reads(&log, |stage| {
        matches!(stage, VectorStage::ExactFetch | VectorStage::ExactScore)
    }) {
        payload.extend(range.start / 8192..=(range.end - 1) / 8192);
    }
    assert_eq!(
        payload, pages,
        "rows reads touch only the selected rows' pages"
    );
    Ok(())
}

/// A whole-cluster selection reads the band, even with sparse reads forced.
fn whole_cluster_selections_read_the_band() -> crate::Result<()> {
    let fixture = reads_fixture(Metric::L2, &[1, 4], &[])?;
    let (_, stats) = fixture.run_planned(
        0,
        fixture.all()?.as_ref(),
        &reads_vector(42),
        10,
        every_cluster(),
        Some(PlanOverride::read(LayerRead::Sparse)),
        true,
    )?;
    let layer0 = stats.layers.get(0).unwrap();
    assert_eq!(layer0.sparse_clusters(), 0, "{stats:?}");
    assert_eq!(layer0.full_clusters(), stats.postings_row, "{stats:?}");
    Ok(())
}

/// Left to the layout on paged d=1024 storage, a selective filter reads layer 0 sparsely, or
/// with the exact plan on scores it exactly; a dense filter reads whole bands and an unfiltered
/// query never scores exactly. Every plan returns the oracle's top k.
fn layout_plans_selective_filters() -> crate::Result<()> {
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        let fixture = FixtureSpec::clustered(metric, 1024, 2, 1200)
            .schedule(&[1, 4])
            .paged()
            .label("few", |ordinal| ordinal == 7 || ordinal == 8)
            .label("dense", |ordinal| ordinal % 10 != 0)
            .build()?;
        let query = Lcg(5).vector(1024, 0.5);
        let exact_on = ProbeParams {
            exact_plan: true,
            ..routed_exhaustive_params(2)
        };
        for (name, selective) in [("few", true), ("dense", false)] {
            let case = format!("{metric:?} {name}");
            let filter = fixture.filter(name);
            let oracle = fixture.oracle(&filter, &query, 10)?;
            let fruit = fixture.search(&filter, &query, 10, routed_exhaustive_params(2))?;
            let stats = &fruit.stats[0];
            let expected = if selective { stats.postings_row } else { 0 };
            let layer0 = stats.layers.get(0).unwrap();
            assert_eq!(layer0.sparse_clusters(), expected, "{case}: {stats:?}");
            assert_eq!(fruit.results, oracle, "{case}");
            let fruit = fixture.search(&filter, &query, 10, exact_on.clone())?;
            let stats = &fruit.stats[0];
            if selective {
                let layer0 = stats.layers.get(0).unwrap();
                assert_eq!(
                    layer0.exact_clusters(),
                    stats.postings_row,
                    "{case}: {stats:?}"
                );
            }
            assert_eq!(fruit.results, oracle, "{case}");
        }
        let unfiltered = fixture.search(&AllQuery, &query, 10, exact_on)?;
        let exact_clusters: usize = (0..2)
            .map(|layer| {
                unfiltered.stats[0]
                    .layers
                    .get(layer)
                    .unwrap()
                    .exact_clusters()
            })
            .sum();
        assert_eq!(exact_clusters, 0, "{metric:?}: {:?}", unfiltered.stats[0]);
        assert_eq!(
            unfiltered.results,
            fixture.oracle(&AllQuery, &query, 10)?,
            "{metric:?}"
        );
    }
    Ok(())
}

/// A cluster scored exactly never reads its band; its rows are read in the exact stages.
fn exact_clusters_never_read_their_band() -> crate::Result<()> {
    let fixture = reads_fixture(Metric::L2, &[1], &[])?;
    let query = reads_vector(42);
    let rules: [(ExactRule, bool); 3] = [
        (|_, _| true, true),
        (|cluster, _| cluster % 2 == 0, false),
        (|_, _| false, false),
    ];
    for (rule, all) in rules {
        fixture.clear_reads();
        let forced = PlanOverride {
            read: Some(LayerRead::Band),
            exact: Some(rule),
        };
        let (hits, stats) = fixture.run_planned(
            0,
            fixture.all()?.as_ref(),
            &query,
            10,
            every_cluster(),
            Some(forced),
            true,
        )?;
        assert_eq!(hits, fixture.oracle(&AllQuery, &query, 10)?);
        let log = fixture.reads();
        let layer0 = stats.layers.get(0).unwrap();
        // Unfiltered admission reads no DocIds: its reads are bands alone.
        assert_eq!(
            stage_reads(&log, |stage| stage == VectorStage::LayerScan(0)).len(),
            layer0.full_clusters(),
            "{stats:?}"
        );
        assert_eq!(
            layer0.full_clusters() + layer0.exact_clusters(),
            stats.postings_row
        );
        let exact_reads = stage_reads(&log, |stage| {
            matches!(stage, VectorStage::ExactFetch | VectorStage::ExactScore)
        })
        .len();
        assert_eq!(stats.exact_io.reads, exact_reads as u64, "{stats:?}");
        assert_eq!(exact_reads > 0, stats.exact_rows > 0, "{stats:?}");
        assert_eq!(
            stats.exact_rows,
            layer0.exact_rows() + layer0.survivors(),
            "{stats:?}"
        );
        if all {
            assert_eq!(layer0.full_clusters(), 0);
        }
    }
    Ok(())
}

/// A group scored exactly at a later layer leaves the candidates before that layer's boundary:
/// none of its rows reaches it, and the top k is unchanged.
fn retired_groups_never_reach_a_later_band() -> crate::Result<()> {
    let fixture = FixtureSpec::clustered(Metric::L2, 64, 8, 800)
        .schedule(&[1, 4])
        .label("some", |ordinal| ordinal % 3 == 1)
        .build()?;
    let filter = fixture.filter("some");
    let query = Lcg(5).vector(64, 0.5);
    let (hits, stats) = fixture.run_planned(
        0,
        fixture.weight(&filter)?.as_ref(),
        &query,
        10,
        routed_exhaustive_params(8),
        Some(PlanOverride::exact(|cluster, layer| {
            layer == 1 && cluster % 2 == 0
        })),
        true,
    )?;
    assert_eq!(hits, fixture.oracle(&filter, &query, 10)?);
    let layer1 = stats.layers.get(1).unwrap();
    assert!(layer1.exact_clusters() > 0, "{stats:?}");
    let searcher = fixture.index.reader()?.searcher();
    let vectors = first_segment(&searcher).1.vector_index(fixture.field)?;
    let index = vectors.index().unwrap();
    let even = |row: &usize| {
        (0..index.num_clusters())
            .find(|&cluster| index.cluster_range(cluster).contains(row))
            .is_some_and(|cluster| cluster % 2 == 0)
    };
    let trace = &stats.quantized_trace;
    assert!(trace.boundary_rows[0].iter().any(even));
    assert!(!trace.boundary_rows[1].iter().any(even));
    assert!(!trace.layer_columns[1].iter().any(|(row, _)| even(row)));
    assert_eq!(
        layer1.scored() + layer1.exact_rows() + layer1.recheck_dropped(),
        trace.boundary_rows[0].len()
    );
    Ok(())
}

/// The located path reads no DocIds column: its location-map reads run in `locate`, and each
/// quantized layer-0 cluster reads its centroid row in `centroid`.
fn located_path_reads_no_doc_ids() -> crate::Result<()> {
    for schedule in [&[][..], &[1][..], &[1, 4][..]] {
        let fixture = reads_fixture(Metric::L2, schedule, &[])?;
        let ranges = fixture.column_ranges()?;
        let filter = BooleanQuery::union(vec![
            Box::new(fixture.filter("d42")) as Box<dyn Query>,
            Box::new(fixture.filter("d242")),
            Box::new(fixture.filter("d1101")),
        ]);
        let query = reads_vector(42);
        let params = ProbeParams {
            located: LocatedPath::MaxSelectivity(Selectivity::new(1.0).unwrap()),
            ..every_cluster()
        };
        let collector = || {
            TopDocs::with_limit(10)
                .order_by_similarity(fixture.field, query.clone())
                .with_probe_params(params.clone())
                .build()
        };
        let searcher = fixture.index.reader()?.searcher();
        // The first search opens the location map; measure the second.
        searcher.search(&filter, &collector())?;
        fixture.clear_reads();
        let fruit = searcher.search(&filter, &collector())?;
        let case = format!("{schedule:?}");
        assert_eq!(
            fruit.results,
            fixture.oracle(&filter, &query, 10)?,
            "{case}"
        );
        let stats = &fruit.stats[0];
        assert_eq!(stats.access_path, AccessPath::Located, "{case}");
        assert_eq!(stats.located_matches, 3, "{case}");
        let log = fixture.reads();
        assert!(
            log.iter().all(|(_, read)| ranges
                .docs
                .iter()
                .flatten()
                .all(|docs| !overlaps(read, docs))),
            "{case}: a DocIds column was read"
        );
        let locate_reads = stage_reads(&log, |stage| stage == VectorStage::Locate).len();
        assert!(locate_reads > 0, "{case}");
        assert_eq!(stats.locate_io.reads, locate_reads as u64, "{case}");
        // Centroid rows live in the centroid file, which the read log does not record.
        let quantized_clusters = stats.layers.get(0).map_or(0, |layer0| {
            layer0.sparse_clusters() + layer0.full_clusters()
        });
        assert_eq!(stats.centroid_io.reads, quantized_clusters as u64, "{case}");
        assert_eq!(quantized_clusters > 0, !schedule.is_empty(), "{case}");
    }
    Ok(())
}

/// The located path charges every cluster's open and every scored row against the work budget,
/// and scores every match even past the budget.
#[test]
fn located_path_charges_the_work_meter() -> crate::Result<()> {
    let fixture = reads_fixture(Metric::L2, &[1], &[])?;
    let filter = BooleanQuery::union(vec![
        Box::new(fixture.filter("d42")) as Box<dyn Query>,
        Box::new(fixture.filter("d242")),
        Box::new(fixture.filter("d1101")),
    ]);
    // Far below one cluster's work: the routed path would stop after its first cluster.
    let params = ProbeParams {
        located: LocatedPath::MaxSelectivity(Selectivity::new(1.0).unwrap()),
        max_probe_fraction: 1e-6,
        min_probe_clusters: 0,
        ..Default::default()
    };
    let weight = fixture.weight(&filter)?;
    let (hits, stats) =
        fixture.run_segment(0, weight.as_ref(), &reads_vector(42), 10, params.clone())?;
    let searcher = fixture.index.reader()?.searcher();
    let vectors = first_segment(&searcher).1.vector_index(fixture.field)?;
    let ivf = vectors.index().unwrap();
    let (budget, n_avg, x) = params.resolved_work_budget(ivf.num_clusters(), ivf.num_docs())?;
    assert_eq!(stats.access_path, AccessPath::Located);
    assert_eq!(hits.len(), 3, "every match is scored");
    assert_eq!(stats.candidates_scored, 3);
    let charged = stats.located_clusters as f64 * x + 3.0 * (1.0 - x) / n_avg;
    assert!(
        (f64::from(stats.work_charged) - charged).abs() <= 1e-6,
        "{stats:?}"
    );
    assert_eq!(stats.work_budget, budget as f32);
    assert!(f64::from(stats.work_charged) > budget);
    Ok(())
}

/// One named row of the reads table.
type ReadCase = (&'static str, fn() -> crate::Result<()>);

/// Every plan reads only what it needs, in the documented order.
#[test]
fn reads_follow_the_plan() -> crate::Result<()> {
    let cases: [ReadCase; 12] = [
        (
            "an empty cluster skips its band",
            empty_cluster_skips_its_band,
        ),
        (
            "a fully filtered cluster reads only its DocIds",
            filtered_cluster_reads_only_doc_ids,
        ),
        (
            "filtered clusters read DocIds before payload",
            filtered_clusters_read_doc_ids_before_payload,
        ),
        (
            "an open gate reads DocIds only for competitive survivors",
            open_gate_reads_doc_ids_only_for_competitive_survivors,
        ),
        (
            "an all-dead cluster reads nothing past DocIds",
            all_dead_cluster_reads_only_doc_ids,
        ),
        (
            "the flat path reads one row per survivor",
            flat_path_reads_one_row_per_survivor,
        ),
        (
            "exact scoring reads only the survivors' pages",
            exact_reads_touch_only_survivor_pages,
        ),
        (
            "a whole-cluster selection reads the band",
            whole_cluster_selections_read_the_band,
        ),
        (
            "the layout reads selective filters sparsely or exactly",
            layout_plans_selective_filters,
        ),
        (
            "an exact cluster never reads its band",
            exact_clusters_never_read_their_band,
        ),
        (
            "a retired group never reaches a later band",
            retired_groups_never_reach_a_later_band,
        ),
        (
            "the located path reads no DocIds",
            located_path_reads_no_doc_ids,
        ),
    ];
    for (name, case) in cases {
        case().map_err(|error| crate::TantivyError::InternalError(format!("{name}: {error}")))?;
    }
    Ok(())
}
