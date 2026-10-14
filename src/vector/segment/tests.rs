//! End-to-end segment search tests: results against the brute-force oracle, routing edge cases,
//! and the storage reads each plan makes.

use std::collections::BTreeSet;
use std::ops::Range;

use crate::query::{AllQuery, BooleanQuery, EnableScoring, Query};
use crate::schema::Metric;
use crate::vector::ivf::AdaptiveProbeParams;
use crate::vector::test_support::{ColumnRanges, DocsWeight, Fixture, FixtureSpec, Lcg, ReadLog};
use crate::vector::tests::exhaustive_params;
use crate::vector::{ProbeStats, RouterKind, Stage};
use crate::DocId;

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
        .label("doomed", |ordinal| (60..63).contains(&ordinal));
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

/// Under an exhaustive probe every representation, filter, delete pattern and k returns exactly
/// the oracle's top k, across segments.
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
                ];
                for (query_ord, query) in matrix_queries().iter().enumerate() {
                    for (filter_name, filter) in &filters {
                        for k in [10, 1000] {
                            let case = format!(
                                "{metric:?} {representation:?} deletes={deletes} \
                                 query={query_ord} filter={filter_name} k={k}"
                            );
                            let fruit = fixture.search(
                                filter.as_ref(),
                                query,
                                k,
                                exhaustive_params(MATRIX_CLUSTERS),
                            )?;
                            let oracle = fixture.oracle(filter.as_ref(), query, k)?;
                            assert_eq!(fruit.results, oracle, "{case}");
                            for stats in &fruit.stats {
                                assert_identities(stats, &case);
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
    .build()?;
    let query = [1.0, 1.0];
    let oracle = fixture.oracle(&AllQuery, &query, 1)?;
    assert_eq!(
        fixture.ordinal(oracle[0].1)?,
        2,
        "the trap doc is the true nearest"
    );
    let one_cluster = AdaptiveProbeParams {
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
fn routes_through_graph_below_ceiling() -> crate::Result<()> {
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
    let params = AdaptiveProbeParams {
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

/// A single-centroid segment has no routing graph; it ranks its one cluster and returns the
/// oracle's top k.
#[test]
fn single_centroid_segment_routes() -> crate::Result<()> {
    let fixture = FixtureSpec::ivf(
        Metric::L2,
        vec![vec![0.0, 0.0]],
        (0..5).map(|i| vec![i as f32 * 0.01, 0.0]).collect(),
    )
    .build()?;
    let query = [0.0, 0.0];
    let fruit = fixture.search(&AllQuery, &query, 3, AdaptiveProbeParams::default())?;
    assert_eq!(fruit.results, fixture.oracle(&AllQuery, &query, 3)?);
    assert_eq!(fruit.stats[0].clusters_probed(), 1);
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

/// Probe parameters that open every cluster the router hands over.
fn every_cluster() -> AdaptiveProbeParams {
    AdaptiveProbeParams {
        max_probe_fraction: 1.0,
        min_probe_clusters: 4,
        ..Default::default()
    }
}

/// The ranges `log` requested while `stage` ran.
fn stage_reads<'a>(log: &'a ReadLog, stage: impl Fn(Stage) -> bool + 'a) -> Vec<&'a Range<usize>> {
    log.iter()
        .filter(|(running, _)| stage(*running))
        .map(|(_, range)| range)
        .collect()
}

fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

/// The admission stage of a schedule: the layer-0 scan, or the full-precision scan.
fn admission(schedule: &[u8]) -> impl Fn(Stage) -> bool {
    let quantized = !schedule.is_empty();
    move |stage| {
        if quantized {
            stage == Stage::LayerScan(0)
        } else {
            stage == Stage::ExactScan
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
    assert!(stage_reads(&fixture.reads(), |stage| stage == Stage::LayerScan(0)).is_empty());
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
                    Stage::ExactFetch | Stage::ExactScore | Stage::LayerScan(1..)
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
                || (schedule.is_empty() && matches!(stage, Stage::ExactFetch | Stage::ExactScore))
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

/// With no filter and no deletes, layer scans never touch a DocIds column; the exact phase reads
/// it once for each cluster with a survivor, and each layer's IO counts its own requests. The
/// scan is the same as under a filter that matches every vector.
fn open_gate_reads_doc_ids_only_in_the_exact_phase() -> crate::Result<()> {
    let fixture = reads_fixture(Metric::L2, &[1, 4], &[])?;
    let ranges = fixture.column_ranges()?;
    let query = reads_vector(42);
    fixture.clear_reads();
    let open = fixture.search(&AllQuery, &query, 3, every_cluster())?;
    let log = fixture.reads();
    for range in stage_reads(&log, |stage| matches!(stage, Stage::LayerScan(_))) {
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
            .filter(|(stage, read)| *stage == Stage::ExactFetch && read == range)
            .count();
        assert_eq!(
            fetched, expected,
            "cluster={cluster} DocIds exact-phase reads"
        );
        assert_eq!(
            log.iter().filter(|(_, read)| read == range).count(),
            expected,
            "cluster={cluster} DocIds reads outside the exact phase"
        );
    }
    for layer in 0..2 {
        let requested = stage_reads(&log, |stage| stage == Stage::LayerScan(layer as u8));
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
            stage_reads(&fixture.reads(), |stage| stage == Stage::ExactScan).len(),
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
    let admission_reads = stage_reads(&log, |stage| stage == Stage::ExactScan);
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
        matches!(stage, Stage::ExactFetch | Stage::ExactScore)
    }) {
        payload.extend(range.start / 8192..=(range.end - 1) / 8192);
    }
    assert_eq!(
        payload, pages,
        "rows reads touch only the selected rows' pages"
    );
    Ok(())
}

/// One named row of the reads table.
type ReadCase = (&'static str, fn() -> crate::Result<()>);

/// Every plan reads only what it needs, in the documented order.
#[test]
fn reads_follow_the_plan() -> crate::Result<()> {
    let cases: [ReadCase; 7] = [
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
            "an open gate reads DocIds only in the exact phase",
            open_gate_reads_doc_ids_only_in_the_exact_phase,
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
    ];
    for (name, case) in cases {
        case().map_err(|error| crate::TantivyError::InternalError(format!("{name}: {error}")))?;
    }
    Ok(())
}
