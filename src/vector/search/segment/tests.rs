//! End-to-end segment search tests: results against the brute-force oracle, routing edge cases,
//! and the storage reads each plan makes.

use std::collections::BTreeSet;
use std::ops::Range;

use crate::collector::TopDocs;
use crate::query::{AllQuery, BooleanQuery, EnableScoring, Query};
use crate::schema::Metric;
use crate::vector::params::{LocatedPath, ProbeParams, Selectivity};
use crate::vector::search::scan::layer_read::LayerRead;
use crate::vector::search::scan::plan::{ExactRule, PlanOverride};
use crate::vector::stats::{AccessPath, LayerProbeStats};
use crate::vector::test_support::{ColumnRanges, DocsWeight, Fixture, FixtureSpec, Lcg, ReadLog};
use crate::vector::tests::{exhaustive_params, routed_exhaustive_params};
use crate::vector::{ProbeStats, RouterKind, VectorStage};
use crate::{DocAddress, DocId, Score};

// --- Oracle matrix ---

const MATRIX_DIM: usize = 64;
const MATRIX_DOCS: usize = 240;
const MATRIX_CLUSTERS: usize = 6;

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
        stats.pruned_filter + stats.pruned_dead + stats.pruned_duplicate + stats.candidates_scored,
        "{case}: visited must equal filtered + dead + duplicate + scored ({stats:?})"
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
            None,
            admission_pruning,
        )?;
        assert_identities(&segment_stats, "admission pruning");
        hits.extend(segment_hits);
        stats.push(segment_stats);
    }
    hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    hits.truncate(k);
    Ok((hits, stats))
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
            let vectors = searcher.segment_readers()[0].vector_index(fixture.field)?;
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
    let segment = &searcher.segment_readers()[0];
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
    let vectors = searcher.segment_readers()[0].vector_index(fixture.field)?;
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
        layer1.scored() + layer1.exact_rows(),
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
        let collector = TopDocs::with_limit(10)
            .order_by_similarity(fixture.field, query.clone())
            .with_probe_params(params);
        let searcher = fixture.index.reader()?.searcher();
        // The first search opens the location map; measure the second.
        searcher.search(&filter, &collector)?;
        fixture.clear_reads();
        let fruit = searcher.search(&filter, &collector)?;
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
    let vectors = searcher.segment_readers()[0].vector_index(fixture.field)?;
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

#[test]
fn boundary_replica_finds_the_neighbor_with_one_cluster() -> crate::Result<()> {
    let pad = |mut v: Vec<f32>| {
        v.resize(64, 0.0);
        v
    };
    for schedule in [&[][..], &[1, 1][..]] {
        let fixture = FixtureSpec::ivf(
            Metric::L2,
            vec![pad(vec![0.0, 0.0]), pad(vec![10.0, 10.0])],
            vec![
                vec![0.0, -10.0],
                vec![-10.0, 0.0],
                vec![5.0, 5.01],
                vec![10.0, 10.0],
            ]
            .into_iter()
            .map(pad)
            .collect(),
        )
        .boundary_epsilon(0.1)
        .schedule(schedule)
        .build()?;
        let query = pad(vec![1.0, 1.0]);
        let one_cluster = ProbeParams {
            max_probe_fraction: 0.5,
            min_probe_clusters: 1,
            ..routed_exhaustive_params(2)
        };
        let found = fixture.search(&AllQuery, &query, 1, one_cluster)?;
        assert_eq!(fixture.ordinal(found.results[0].1)?, 2);
        assert_eq!(found.stats[0].postings_row, 1);
    }
    Ok(())
}

#[test]
fn boundary_replicas_preserve_unique_top_k_across_search_paths() -> crate::Result<()> {
    for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
        for representation in [
            Representation::FullPrecision,
            Representation::Quantized(&[1, 1]),
        ] {
            let mut spec = FixtureSpec::clustered(metric, 64, 6, 240)
                .boundary_epsilon(100.0)
                .segments(2)
                .deletes(&[2, 70])
                .without_vectors(|ordinal| ordinal % 37 == 5)
                .label("half", |ordinal| ordinal % 2 == 0)
                .label("rare", |ordinal| ordinal % 97 == 7);
            if let Representation::Quantized(schedule) = representation {
                spec = spec.schedule(schedule);
            }
            let fixture = spec.build()?;
            let filters: Vec<Box<dyn Query>> = vec![
                Box::new(AllQuery),
                Box::new(fixture.filter("half")),
                Box::new(fixture.filter("rare")),
            ];
            for (path, params) in paths(representation) {
                for (filter_id, filter) in filters.iter().enumerate() {
                    for k in [10, 1000] {
                        let case = format!(
                            "replicas {metric:?} {representation:?} {path} filter={filter_id} \
                             k={k}"
                        );
                        let stats = assert_case(
                            &fixture,
                            representation,
                            filter.as_ref(),
                            &matrix_queries()[0],
                            k,
                            &params,
                            &case,
                        )?;
                        if filter_id == 0 && k == 1000 {
                            assert!(
                                stats.iter().map(|s| s.pruned_duplicate).sum::<usize>() > 0,
                                "{case}"
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
