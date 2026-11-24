//! IVF build and merge tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use superkmeans::make_blobs;

use crate::collector::TopDocs;
use crate::directory::error::{DeleteError, OpenReadError, OpenWriteError};
use crate::index::IndexSettings;
use crate::indexer::NoMergePolicy;
use crate::query::AllQuery;
use crate::schema::{Field, Metric, Schema, Term, VectorDType, VectorOptions, STORED, STRING};
use crate::vector::clustering::{
    IvfAssignmentCandidate, IvfCentroids, IvfClusterer, SuperKMeansClusterer, VectorSource,
};
use crate::vector::diagnostics::{VectorClusterStats, VectorInfo, VectorStorageFormat};
use crate::vector::router::MIN_ROUTED_ASSIGNMENT_CENTROIDS;
use crate::vector::test_support::{Fixture, FixtureSpec, NearestClusterer};
use crate::vector::tests::{
    exhaustive_params, ground_truth, routed_exhaustive_params, TestVectorIndex,
};
use crate::vector::{RouterKind, VectorQuantizationConfig, VectorQuantizationLayer};
use crate::{Index, IndexWriter, Result, TantivyDocument};

const DOCS_PER_CLUSTER: usize = 6;

// --- Fixtures ---

/// Six centroids on a 3x2 grid, with six docs labelled `d{i}` packed around each.
fn multi_cluster_fixture() -> (Vec<Vec<f32>>, Vec<String>) {
    let centroids = vec![
        vec![0.0f32, 0.0],
        vec![10.0, 0.0],
        vec![20.0, 0.0],
        vec![0.0, 10.0],
        vec![10.0, 10.0],
        vec![20.0, 10.0],
    ];
    let labels = (0..centroids.len() * DOCS_PER_CLUSTER)
        .map(|i| format!("d{i}"))
        .collect();
    (centroids, labels)
}

fn multi_cluster_docs<'a>(
    centroids: &[Vec<f32>],
    labels: &'a [String],
) -> Vec<(&'a str, [f32; 2])> {
    (0..labels.len())
        .map(|i| {
            let c = &centroids[i / DOCS_PER_CLUSTER];
            let off = (i % DOCS_PER_CLUSTER) as f32 * 0.01;
            (labels[i].as_str(), [c[0] + off, c[1] + off])
        })
        .collect()
}

/// A fixture handle over an index built by hand.
fn handle(index: &Index, field: Field, label: Field) -> Fixture {
    Fixture {
        index: index.clone(),
        field,
        label,
        metric: Metric::L2,
        directory: None,
    }
}

// --- Merge ---

/// Merging flat segments with deletes, whose `.vec` still counts deleted rows, yields an IVF
/// segment that holds and counts only the alive docs.
#[test]
fn merge_flat_segments_with_deletes_into_ivf() -> crate::Result<()> {
    let (centroids, labels) = multi_cluster_fixture();
    let docs = multi_cluster_docs(&centroids, &labels);
    let n = docs.len();

    // Same shape as `build_inline_ivf`, but the two flat source segments
    // stay unmerged so the deletes land BEFORE the clustering merge.
    let mut sb = Schema::builder();
    let embed_field = sb.add_vector_field(
        "embedding",
        VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
    );
    let label_field = sb.add_text_field("label", STRING | STORED);
    let schema = sb.build();
    let settings = IndexSettings {
        vector_clustering_threshold: 1,
        ..IndexSettings::default()
    };
    let index = Index::builder()
        .schema(schema)
        .settings(settings)
        .ivf_clusterer(Arc::new(NearestClusterer {
            centroids: centroids.clone(),
            pairs: true,
        }))
        // Exact: exhaustive top-N below assumes every alive cluster is probed.
        .ivf_router(RouterKind::Exact)
        .create_in_ram()?;
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let mid = n / 2;
    for chunk in [&docs[..mid], &docs[mid..]] {
        for (label, v) in chunk {
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, label);
            doc.add_vector(embed_field, v.as_slice());
            writer.add_document(doc)?;
        }
        writer.commit()?;
    }

    // Tombstone docs in BOTH flat sources (d0/d7 in the first commit,
    // d35 in the second), then merge everything into one IVF segment.
    let deleted = ["d0", "d7", "d35"];
    for label in deleted {
        writer.delete_term(Term::from_field_text(label_field, label));
    }
    writer.commit()?;
    let segment_ids = index.searchable_segment_ids()?;
    writer.merge(&segment_ids).wait()?;
    writer.wait_merging_threads()?;

    let alive = n - deleted.len();
    let searcher = index.reader()?.searcher();
    assert_eq!(searcher.segment_readers().len(), 1, "one merged segment");
    let segment_reader = &searcher.segment_readers()[0];
    let vec_reader = segment_reader.vector_index(embed_field)?;
    let info = vec_reader.info()?.expect("vector info");
    assert_eq!(info.format, VectorStorageFormat::Ivf, "merge must cluster");
    assert_eq!(info.num_vectors, alive, "deleted docs must not be counted");
    assert_eq!(vec_reader.num_vectors(), alive);

    // Every alive doc comes back exactly once; no deleted label survives.
    let fixture = handle(&index, embed_field, label_field);
    let hits = fixture
        .search(
            &AllQuery,
            &[10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?
        .results;
    assert_eq!(hits.len(), alive, "exhaustive top-N must return alive docs");
    let mut seen_labels = std::collections::HashSet::new();
    for (_, addr) in &hits {
        let label = format!("d{}", fixture.ordinal(*addr)?);
        assert!(
            !deleted.contains(&label.as_str()),
            "deleted doc {label} surfaced in results"
        );
        assert!(seen_labels.insert(label), "duplicate doc in results");
    }
    Ok(())
}

/// A merge that deletes every vector of one field, while another keeps live vectors, writes the
/// emptied field's `.vec`/`.centroids` slots as empty, the same as a field with no vectors.
#[test]
fn merge_deleting_every_doc_of_one_field_writes_empty_ivf() -> crate::Result<()> {
    let (centroids, labels) = multi_cluster_fixture();
    let docs = multi_cluster_docs(&centroids, &labels);
    let n = docs.len();

    let mut sb = Schema::builder();
    let doomed_field = sb.add_vector_field(
        "embedding_doomed",
        VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
    );
    let kept_field = sb.add_vector_field(
        "embedding_kept",
        VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
    );
    let label_field = sb.add_text_field("label", STRING | STORED);
    let schema = sb.build();
    let settings = IndexSettings {
        vector_clustering_threshold: 1,
        ..IndexSettings::default()
    };
    let index = Index::builder()
        .schema(schema)
        .settings(settings)
        .ivf_clusterer(Arc::new(NearestClusterer {
            centroids: centroids.clone(),
            pairs: true,
        }))
        // Exact: exhaustive search below assumes every alive cluster is probed.
        .ivf_router(RouterKind::Exact)
        .create_in_ram()?;
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));

    // Even docs carry the doomed field, odd docs the kept one, split
    // across two flat commits so BOTH sources hold doomed vectors.
    let mid = n / 2;
    for (i, (label, v)) in docs.iter().enumerate() {
        let mut doc = TantivyDocument::new();
        doc.add_text(label_field, label);
        let field = if i % 2 == 0 { doomed_field } else { kept_field };
        doc.add_vector(field, v.as_slice());
        writer.add_document(doc)?;
        if i + 1 == mid {
            writer.commit()?;
        }
    }
    writer.commit()?;

    // Tombstone every doomed-field doc, then merge everything into one
    // IVF segment.
    for (i, (label, _)) in docs.iter().enumerate() {
        if i % 2 == 0 {
            writer.delete_term(Term::from_field_text(label_field, label));
        }
    }
    writer.commit()?;
    let segment_ids = index.searchable_segment_ids()?;
    writer.merge(&segment_ids).wait()?;
    writer.wait_merging_threads()?;

    let searcher = index.reader()?.searcher();
    assert_eq!(searcher.segment_readers().len(), 1, "one merged segment");
    let segment_reader = &searcher.segment_readers()[0];

    // The emptied field reads back as a zeroed IVF field, not an error.
    let vec_reader = segment_reader.vector_index(doomed_field)?;
    assert_eq!(vec_reader.num_vectors(), 0);
    let info = vec_reader.info()?.expect("vector info");
    assert_eq!(
        info,
        VectorInfo {
            format: VectorStorageFormat::Ivf,
            num_vectors: 0,
            num_centroids: Some(0),
            cluster_stats: Some(VectorClusterStats {
                min_cluster_size: 0,
                max_cluster_size: 0,
                avg_cluster_size: 0.0,
                empty_clusters: 0,
            }),
            router: Some(RouterKind::Exact),
            router_levels: Some(vec![]),
            quantized_layers: None,
            quantized_bytes_per_row: None,
        },
    );
    let ivf = vec_reader.index().expect("expected IVF storage");
    assert!(vec_reader.is_empty(), "no rows in the emptied field");
    assert_eq!(ivf.num_rows(), 0);
    assert_eq!(ivf.num_clusters(), 0);

    // The live field is untouched: every alive doc is counted and found.
    let kept_count = n / 2;
    assert_eq!(
        segment_reader.vector_index(kept_field)?.num_vectors(),
        kept_count
    );
    let hits = handle(&index, kept_field, label_field)
        .search(
            &AllQuery,
            &[10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?
        .results;
    assert_eq!(hits.len(), kept_count, "kept field returns alive docs");
    Ok(())
}

// --- Build log ---

/// Captures `paradedb::ivf_build` log records so a test can read back the
/// timings line the merge emits.
struct CaptureLogger;

static CAPTURED_IVF_BUILD: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

impl log::Log for CaptureLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.target() == "paradedb::ivf_build"
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            CAPTURED_IVF_BUILD
                .lock()
                .unwrap()
                .push(format!("{}", r.args()));
        }
    }
    fn flush(&self) {}
}

static CAPTURE_LOGGER: CaptureLogger = CaptureLogger;

/// The merge emits one parseable `ivf_build timings_ms ...` line per merge, printed under
/// `--nocapture`.
#[test]
fn ivf_build_emits_timings_log() -> crate::Result<()> {
    let _ = log::set_logger(&CAPTURE_LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    // 200 centroids on a 20x10 grid; ~5000 docs clustered around them.
    let mut centroids: Vec<Vec<f32>> = Vec::new();
    for x in 0..20 {
        for y in 0..10 {
            centroids.push(vec![x as f32 * 10.0, y as f32 * 10.0]);
        }
    }
    let n_per = 25usize;
    let docs: Vec<Vec<f32>> = (0..centroids.len() * n_per)
        .map(|i| {
            let c = &centroids[i / n_per];
            let off = (i % n_per) as f32 * 0.05;
            vec![c[0] + off, c[1] + off]
        })
        .collect();

    let before = CAPTURED_IVF_BUILD.lock().unwrap().len();
    FixtureSpec::ivf(Metric::L2, centroids, docs).build()?;
    let lines: Vec<String> = CAPTURED_IVF_BUILD.lock().unwrap()[before..].to_vec();
    let line = lines
        .iter()
        .find(|l| l.contains("ivf_build timings_ms") && l.contains("centroids=200"))
        .expect("expected an ivf_build timings line for the 200-centroid build");
    let field = |key: &str| -> usize {
        line.split_whitespace()
            .find_map(|part| part.strip_prefix(key)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("no {key} in {line}"))
            .parse()
            .unwrap()
    };
    let vectors = field("vectors");
    let stages = ["train", "router", "assign", "other"];
    let measured: usize = stages.iter().map(|stage| field(stage)).sum();
    let total = field("total");
    assert!(total >= measured, "overlapping stage timers: {line}");
    assert!(
        total - measured < stages.len(),
        "stages must cover total, allowing millisecond truncation: {line}"
    );
    assert!(line.contains("train_reads="));
    // The test clusterer hands batches over through the trait's default, which assigns the whole
    // source and then reads it again batch by batch.
    assert_eq!(field("assign_reads"), 2 * vectors);
    // A plain field reads its rows again for its blocks.
    assert_eq!(field("encode_reads"), vectors);
    // The Data and ExactRows parts sit inside `other`; truncating each part never exceeds the
    // truncated whole.
    let parts = ["encode_read_ms", "encode_ms", "data_write_ms", "exact_ms"];
    let within_other: usize = parts.iter().map(|part| field(part)).sum();
    assert!(within_other <= field("other"), "parts exceed other: {line}");
    assert!(
        line.ends_with(&format!(
            "encode_read_ms={} encode_ms={} data_write_ms={} exact_ms={}",
            field("encode_read_ms"),
            field("encode_ms"),
            field("data_write_ms"),
            field("exact_ms")
        )),
        "the stage parts close the line: {line}"
    );
    eprintln!("IVF_BUILD_SAMPLE {line}");
    Ok(())
}

// --- Cluster sizes ---

/// The reader's raw `cluster_sizes` are exactly the array behind `vector_info`'s cluster stats;
/// flat segments expose none.
#[test]
fn ivf_cluster_sizes_match_vector_info() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .build()?;
    let field = index.embedding_field();
    let searcher = index.index.reader()?.searcher();

    let mut segments_checked = 0;
    for segment_reader in searcher.segment_readers() {
        let vec_reader = segment_reader.vector_index(field)?;
        let sizes = vec_reader
            .cluster_sizes()?
            .expect("ivf segment exposes cluster sizes");
        let info = vec_reader.info()?.expect("vector info");
        assert_eq!(info.format, VectorStorageFormat::Ivf);
        let stats = info.cluster_stats.expect("ivf cluster stats");

        // count == num_centroids, and every aggregate index_info reports is
        // reproducible from the raw array.
        assert_eq!(sizes.len(), info.num_centroids.expect("ivf centroids"));
        let sum: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
        let min = sizes.iter().copied().min().unwrap() as usize;
        let max = sizes.iter().copied().max().unwrap() as usize;
        let empty = sizes.iter().filter(|&&s| s == 0).count();
        let avg = sum as f64 / sizes.len() as f64;

        // Per-cluster sizes count posting rows (memberships)...
        let ivf = vec_reader.index().expect("expected IVF segment");
        assert_eq!(sum as usize, ivf.num_rows(), "sizes sum to rows");
        // ...and the shared fixture has no deletes, so the row total
        // coincides with the distinct-doc `num_vectors`.
        assert_eq!(sum as usize, info.num_vectors, "rows == docs");
        assert_eq!(min, stats.min_cluster_size, "min");
        assert_eq!(max, stats.max_cluster_size, "max");
        assert_eq!(empty, stats.empty_clusters, "empty");
        assert!(
            (avg - stats.avg_cluster_size).abs() < 1e-9,
            "avg {avg} vs {}",
            stats.avg_cluster_size
        );
        segments_checked += 1;
    }
    assert!(
        segments_checked > 0,
        "fixture must produce >= 1 IVF segment"
    );

    // Flat segments yield `None`, so the SRF emits no rows for them.
    let flat = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Flat)
        .build()?;
    let flat_field = flat.embedding_field();
    let flat_searcher = flat.index.reader()?.searcher();
    for segment_reader in flat_searcher.segment_readers() {
        assert!(
            segment_reader
                .vector_index(flat_field)?
                .cluster_sizes()?
                .is_none(),
            "flat segments expose no cluster sizes"
        );
    }
    Ok(())
}

/// `vector_info` reports the router kind and its per-level centroid counts: the stacked router's
/// levels shrink from the bottom up and sit under the segment's centroids, the exact router has
/// none, and flat segments have no router.
#[test]
fn vector_info_reports_router_levels() -> crate::Result<()> {
    for router in [
        RouterKind::Stacked,
        RouterKind::Exact,
        RouterKind::QuantizedStacked,
    ] {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .router(router)
            .build()?;
        let field = index.embedding_field();
        let searcher = index.index.reader()?.searcher();
        for segment_reader in searcher.segment_readers() {
            let info = segment_reader
                .vector_index(field)?
                .info()?
                .expect("vector info");
            assert_eq!(info.router, Some(router));
            let levels = info.router_levels.expect("ivf router levels");
            match router {
                RouterKind::Stacked | RouterKind::QuantizedStacked => {
                    let num_centroids = info.num_centroids.expect("ivf centroids");
                    assert!(!levels.is_empty(), "stacked router has a level");
                    assert!(
                        levels[0] <= num_centroids,
                        "{levels:?} over {num_centroids}"
                    );
                    assert!(
                        levels.windows(2).all(|pair| pair[1] <= pair[0]),
                        "levels shrink upward: {levels:?}"
                    );
                }
                RouterKind::Exact => assert!(levels.is_empty(), "{levels:?}"),
            }
        }
    }

    let flat = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Flat)
        .build()?;
    for segment_reader in flat.index.reader()?.searcher().segment_readers() {
        let info = segment_reader
            .vector_index(flat.embedding_field())?
            .info()?
            .expect("vector info");
        assert_eq!((info.router, info.router_levels), (None, None));
    }
    Ok(())
}

// --- Parallel encode ---

/// A two-segment `[1, 1]` index of 3,000 dimension-64 vectors around 24 one-hot centroids plus one
/// centroid no vector is nearest to. Cluster 0 takes a third of the docs, so it outgrows a small
/// group cap.
fn parallel_encode_index(
    metric: Metric,
    replication: u8,
) -> crate::Result<(Index, Arc<AtomicUsize>)> {
    const DIM: usize = 64;
    const CLUSTERS: usize = 24;
    let centroid = |cluster: usize| {
        let mut centroid = vec![0.0; DIM];
        centroid[cluster] = 1.0;
        centroid
    };
    let mut centroids: Vec<Vec<f32>> = (0..CLUSTERS).map(centroid).collect();
    centroids.push(vec![-1.0; DIM]);
    let directory = TempFileCounter::default();
    let opened = Arc::clone(&directory.opened);
    let mut schema = Schema::builder();
    let options = VectorOptions::new(DIM, metric).with_dtype(VectorDType::F32);
    let field = schema.add_vector_field("embedding", options.clone());
    let settings = IndexSettings {
        vector_clustering_threshold: 1,
        vector_replication: replication,
        vector_quantization: vec![crate::vector::VectorQuantizationConfig::materialize(
            "embedding".to_string(),
            &options,
            vec![
                crate::vector::VectorQuantizationLayer { bits: 1, seed: 7 },
                crate::vector::VectorQuantizationLayer { bits: 1, seed: 11 },
            ],
        )?],
        ..IndexSettings::default()
    };
    let index = Index::builder()
        .schema(schema.build())
        .settings(settings)
        .ivf_clusterer(Arc::new(NearestClusterer {
            centroids,
            pairs: true,
        }))
        .ivf_router(RouterKind::Exact)
        .create(directory)?;
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for commit in 0..2 {
        for ordinal in commit * 1_500..(commit + 1) * 1_500 {
            let cluster = if ordinal % 3 == 0 {
                0
            } else {
                ordinal % CLUSTERS
            };
            let mut rng = crate::vector::test_support::Lcg(ordinal as u64 * 7_919 + 13);
            let vector: Vec<f32> = centroid(cluster)
                .iter()
                .zip(rng.vector(DIM, 0.3))
                .map(|(center, noise)| center + noise)
                .collect();
            let mut doc = TantivyDocument::new();
            doc.add_vector(field, &vector);
            writer.add_document(doc)?;
        }
        writer.commit()?;
    }
    Ok((index, opened))
}

/// A RAM directory that counts the temporary files opened in it.
#[derive(Clone, Debug, Default)]
struct TempFileCounter {
    inner: crate::directory::RamDirectory,
    opened: Arc<AtomicUsize>,
}

impl crate::directory::Directory for TempFileCounter {
    fn get_file_handle(
        &self,
        path: &std::path::Path,
    ) -> std::result::Result<Arc<dyn crate::directory::FileHandle>, OpenReadError> {
        self.inner.get_file_handle(path)
    }
    fn delete(&self, path: &std::path::Path) -> std::result::Result<(), DeleteError> {
        self.inner.delete(path)
    }
    fn exists(&self, path: &std::path::Path) -> std::result::Result<bool, OpenReadError> {
        self.inner.exists(path)
    }
    fn open_write_inner(
        &self,
        path: &std::path::Path,
    ) -> std::result::Result<crate::directory::InnerWritePtr, OpenWriteError> {
        self.inner.open_write_inner(path)
    }
    fn open_temp_file(&self) -> std::io::Result<crate::directory::TempFilePtr> {
        self.opened.fetch_add(1, Ordering::Relaxed);
        self.inner.open_temp_file()
    }
    fn atomic_read(&self, path: &std::path::Path) -> std::result::Result<Vec<u8>, OpenReadError> {
        self.inner.atomic_read(path)
    }
    fn atomic_write(&self, path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
        self.inner.atomic_write(path, data)
    }
    fn sync_directory(&self) -> std::io::Result<()> {
        self.inner.sync_directory()
    }
    fn watch(
        &self,
        callback: crate::directory::WatchCallback,
    ) -> crate::Result<crate::directory::WatchHandle> {
        self.inner.watch(callback)
    }
}

/// The `.vec` and `.centroids` bytes of `index`'s segments merged on the calling thread inside a
/// `threads`-thread rayon pool, with the encode budget at `group_bytes` (0: the default), and the
/// runs of encoded records the merge spilled, counted through `opened`, the temporary files of
/// `index`'s [`TempFileCounter`].
fn merged_vector_files(
    index: &Index,
    opened: &AtomicUsize,
    threads: usize,
    group_bytes: usize,
) -> crate::Result<(Vec<u8>, Vec<u8>, usize)> {
    use std::sync::atomic::Ordering;

    use crate::directory::Directory;
    use crate::index::SegmentComponent;
    use crate::indexer::merger::IndexMerger;
    use crate::vector::storage::header::{CENTROIDS_EXT, VEC_EXT};

    let segments = index.searchable_segments()?;
    assert_eq!(segments.len(), 2);
    let target = index.new_segment();
    let opened_before = opened.load(Ordering::Relaxed);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    super::encode::GROUP_DECODED_BYTES_FOR_TESTS.store(group_bytes, Ordering::Relaxed);
    let written = pool.install(|| {
        IndexMerger::open(
            index.schema(),
            index.settings().clone(),
            &segments,
            Box::new(|| false),
            false,
        )?
        .write(&target)
    });
    super::encode::GROUP_DECODED_BYTES_FOR_TESTS.store(0, Ordering::Relaxed);
    written?;
    let read = |extension: &str| {
        index
            .directory()
            .atomic_read(&target.relative_path(SegmentComponent::Custom(extension.to_string())))
    };
    let runs = opened.load(Ordering::Relaxed) - opened_before;
    Ok((read(VEC_EXT)?, read(CENTROIDS_EXT)?, runs))
}

/// How rows are split into encode chunks and record runs never changes a byte: a one-thread pool,
/// a four-thread pool, records that fit in one buffer (no run spilled), a few runs, and one run per
/// row all give byte-equal `.vec` and `.centroids` files, for cosine and L2 with and without
/// replication.
#[test]
fn encode_chunks_and_runs_write_the_same_bytes() -> crate::Result<()> {
    for metric in [Metric::Cosine, Metric::L2] {
        for replication in [1, 2] {
            let (index, opened) = parallel_encode_index(metric, replication)?;
            let (vec, centroids, runs) = merged_vector_files(&index, &opened, 4, 1)?;
            // One row per chunk and per run; the last run stays in memory.
            assert_eq!(runs, 2_999, "{metric:?} replication {replication}");
            for (threads, group_bytes, spills) in
                [(1, 0, false), (4, 0, false), (4, 64 << 10, true)]
            {
                let files = merged_vector_files(&index, &opened, threads, group_bytes)?;
                let case = format!(
                    "{metric:?} replication {replication}, {threads} threads, budget {group_bytes}"
                );
                assert_eq!(files.2 >= 3, spills, "{case}: {} runs", files.2);
                assert!(files.0 == vec, "{case}: .vec differs");
                assert!(files.1 == centroids, "{case}: .centroids differs");
            }
        }
    }
    Ok(())
}

/// A quantized build reads each source row once for assignment and never again for its Data
/// entry, with the built-in clusterer.
#[test]
fn quantized_build_reads_each_row_once_for_assign_and_encode() -> crate::Result<()> {
    const DIM: usize = 64;
    const ROWS: usize = 1_237;
    let _ = log::set_logger(&CAPTURE_LOGGER);
    log::set_max_level(log::LevelFilter::Info);
    let values = make_blobs(ROWS, DIM, 12, false, 1.0, 10.0, 7);
    let options = VectorOptions::new(DIM, Metric::L2);
    let mut schema = Schema::builder();
    let field = schema.add_vector_field("embedding", options.clone());
    let index = Index::builder()
        .schema(schema.build())
        .settings(IndexSettings {
            vector_clustering_threshold: 1,
            vector_replication: 2,
            vector_quantization: vec![VectorQuantizationConfig::materialize(
                "embedding".to_string(),
                &options,
                vec![
                    VectorQuantizationLayer { bits: 1, seed: 7 },
                    VectorQuantizationLayer { bits: 4, seed: 11 },
                ],
            )?],
            ..IndexSettings::default()
        })
        .ivf_clusterer(Arc::new(
            SuperKMeansClusterer::new()
                .with_target_leaf_size(50)
                .with_memory_limit(200 * options.bytes_per_vector()),
        ))
        .ivf_router(RouterKind::Exact)
        .create_in_ram()?;
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for chunk in values.chunks(ROWS / 2 * DIM) {
        for vector in chunk.chunks_exact(DIM) {
            let mut doc = TantivyDocument::new();
            doc.add_vector(field, vector);
            writer.add_document(doc)?;
        }
        writer.commit()?;
    }
    let before = CAPTURED_IVF_BUILD.lock().unwrap().len();
    writer.merge(&index.searchable_segment_ids()?).wait()?;
    writer.wait_merging_threads()?;
    let lines: Vec<String> = CAPTURED_IVF_BUILD.lock().unwrap()[before..].to_vec();
    let line = lines
        .iter()
        .find(|line| line.contains(&format!(" vectors={ROWS} ")))
        .expect("an ivf_build line for the merge");
    let field = |key: &str| -> usize {
        line.split_whitespace()
            .find_map(|part| part.strip_prefix(key)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("no {key} in {line}"))
            .parse()
            .unwrap()
    };
    assert_eq!(field("assign_reads"), ROWS, "{line}");
    assert_eq!(field("encode_reads"), 0, "{line}");
    assert_eq!(field("encode_read_ms"), 0, "{line}");
    assert_eq!(field("exact_reads"), ROWS, "{line}");
    Ok(())
}

struct RecordingClusterer {
    inner: SuperKMeansClusterer,
    exhaustive_calls: AtomicUsize,
    routed_calls: AtomicUsize,
}

impl IvfClusterer for RecordingClusterer {
    fn training_sample_ratio(&self, options: &VectorOptions, num_rows: usize) -> f32 {
        self.inner.training_sample_ratio(options, num_rows)
    }

    fn train(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
        training_sample_ratio: f32,
    ) -> Result<IvfCentroids> {
        self.inner.train(options, source, training_sample_ratio)
    }

    fn assign(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
        centroids: &IvfCentroids,
    ) -> Result<Vec<u32>> {
        self.exhaustive_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.assign(options, source, centroids)
    }

    fn assign_pairs(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
        centroids: &IvfCentroids,
        lambda: f32,
    ) -> Result<Vec<(u32, Option<u32>)>> {
        self.exhaustive_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.assign_pairs(options, source, centroids, lambda)
    }

    fn assign_routed(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
        centroids: &IvfCentroids,
        lambda: Option<f32>,
        candidates: &(dyn Fn(&[f32]) -> Result<Vec<IvfAssignmentCandidate>> + Sync),
    ) -> Result<Vec<(u32, Option<u32>)>> {
        self.routed_calls.fetch_add(1, Ordering::Relaxed);
        self.inner
            .assign_routed(options, source, centroids, lambda, candidates)
    }
}

#[test]
fn routed_build_preserves_documents_replicas_and_exhaustive_results() -> Result<()> {
    const DIM: usize = 64;
    const ROWS: usize = 2 * MIN_ROUTED_ASSIGNMENT_CENTROIDS;
    const ROUTED_LEAF_SIZE: usize = 2;
    const EXHAUSTIVE_LEAF_SIZE: usize = 2 * ROUTED_LEAF_SIZE;
    const MEMORY_LIMIT_BYTES: usize = 1024 * 1024;
    let values = make_blobs(ROWS, DIM, 25, false, 1.0, 10.0, 42);
    for (kind, leaf_size) in [
        (RouterKind::Exact, ROUTED_LEAF_SIZE),
        (RouterKind::Stacked, ROUTED_LEAF_SIZE),
        (RouterKind::QuantizedStacked, ROUTED_LEAF_SIZE),
        (RouterKind::QuantizedStacked, EXHAUSTIVE_LEAF_SIZE),
    ] {
        let clusterer = Arc::new(RecordingClusterer {
            inner: SuperKMeansClusterer::new()
                .with_target_leaf_size(leaf_size)
                .with_training_sample_ratio(Some(1.0))
                .with_memory_limit(MEMORY_LIMIT_BYTES),
            exhaustive_calls: AtomicUsize::new(0),
            routed_calls: AtomicUsize::new(0),
        });
        let mut schema = Schema::builder();
        let options = VectorOptions::new(DIM, Metric::L2);
        let field = schema.add_vector_field("embedding", options.clone());
        let index = Index::builder()
            .schema(schema.build())
            .settings(IndexSettings {
                vector_clustering_threshold: 1,
                vector_replication: 2,
                vector_quantization: vec![VectorQuantizationConfig::materialize(
                    "embedding".to_string(),
                    &options,
                    vec![VectorQuantizationLayer { bits: 1, seed: 7 }],
                )?],
                ..IndexSettings::default()
            })
            .ivf_clusterer(clusterer.clone())
            .ivf_router(kind)
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for chunk in values.chunks_exact(ROWS / 2 * DIM) {
            for vector in chunk.chunks_exact(DIM) {
                let mut doc = TantivyDocument::new();
                doc.add_vector(field, vector);
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }
        writer.merge(&index.searchable_segment_ids()?).wait()?;
        writer.wait_merging_threads()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let vectors = searcher.segment_reader(0).vector_index(field)?;
        let ivf = vectors.index().expect("merged IVF");
        let large_centroid_set = leaf_size == ROUTED_LEAF_SIZE;
        assert_eq!(
            ivf.num_clusters() >= MIN_ROUTED_ASSIGNMENT_CENTROIDS,
            large_centroid_set
        );
        let routed = kind == RouterKind::QuantizedStacked && large_centroid_set;
        assert_eq!(
            clusterer.routed_calls.load(Ordering::Relaxed),
            usize::from(routed),
            "{kind:?}, leaf size {leaf_size}"
        );
        assert_eq!(
            clusterer.exhaustive_calls.load(Ordering::Relaxed),
            usize::from(!routed),
            "{kind:?}, leaf size {leaf_size}"
        );
        assert_eq!(vectors.num_vectors(), ROWS);
        assert_eq!(ivf.num_rows(), 2 * ROWS);
        for query in values.chunks_exact(DIM).step_by(ROWS / 3) {
            let expected = ground_truth::top_k(&index, field, Metric::L2, query, 10)?;
            let actual = searcher.search(
                &AllQuery,
                &TopDocs::with_limit(10)
                    .order_by_similarity(field, query.to_vec())
                    .with_probe_params(routed_exhaustive_params(0))
                    .build_for_searcher(&searcher, &AllQuery)
                    .unwrap(),
            )?;
            assert_eq!(actual.results, expected, "{kind:?}");
        }
    }
    Ok(())
}
