//! IVF build and merge tests.

use std::sync::Arc;

use crate::index::IndexSettings;
use crate::indexer::NoMergePolicy;
use crate::query::AllQuery;
use crate::schema::{Field, Metric, Schema, Term, STORED, STRING};
use crate::vector::test_support::{Fixture, FixtureSpec, NearestClusterer};
use crate::vector::tests::{exhaustive_params, TestVectorIndex};
use crate::vector::{
    RouterKind, VectorClusterStats, VectorDType, VectorInfo, VectorOptions, VectorStorageFormat,
};
use crate::{Index, IndexWriter, TantivyDocument};

const DOCS_PER_CLUSTER: usize = 6;

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

/// Merging flat segments with deletes past the clustering threshold: rows
/// written for since-deleted docs still count toward the sources'
/// `count()` (tombstones don't rewrite `.vec`), so the alive-doc merge
/// iteration legitimately comes up short of `vector_count`. The merge
/// must tolerate that, and the resulting IVF segment must hold — and
/// count — the alive docs only.
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
        }))
        .ivf_router(RouterKind::Stacked)?
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
    let info = vec_reader.info().expect("vector info");
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

/// Merging past the clustering threshold when every doc carrying a
/// vector for ONE field is deleted, while another field keeps live
/// vectors. The sources still report `vector_count > 0` for the emptied
/// field (tombstones don't rewrite `.vec`), so it takes the training
/// path, collects nothing — and used to `continue` without writing the
/// field's `.vec`/`.centroids` slots. The live field still wrote, so the
/// composites existed but the emptied field's slots were missing:
/// `count()`, `open_column()` and `vector_info()` all failed with
/// InternalError. The merge must instead write the same empty slots as
/// the no-vectors-at-all fast path.
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
        }))
        .ivf_router(RouterKind::Stacked)?
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

    // The emptied field reads back as a zeroed IVF field — not an error.
    let vec_reader = segment_reader.vector_index(doomed_field)?;
    assert_eq!(vec_reader.num_vectors(), 0);
    let info = vec_reader.info().expect("vector info");
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

/// The merge emits one parseable `ivf_build timings_ms ...` line per
/// field. Builds a larger index so the phase timings are measurable,
/// captures the line, and prints it (run with `--nocapture`) so we can
/// see where build time goes.
#[test]
fn ivf_build_emits_timings_log() -> crate::Result<()> {
    let _ = log::set_logger(&CAPTURE_LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    // 200 centroids on a 20×10 grid; ~5000 docs clustered around them.
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
    assert!(line.contains("train="));
    assert!(line.contains("id_map_write="));
    eprintln!("IVF_BUILD_SAMPLE {line}");
    Ok(())
}

/// The raw per-cluster sizes from the reader's `cluster_sizes` must be
/// exactly the un-collapsed array behind `info`'s aggregate cluster stats
/// — the invariant `paradedb.ivf_cluster_sizes` relies on to reconcile
/// with `paradedb.index_info`. Flat segments expose no sizes.
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
            .cluster_sizes()
            .expect("ivf segment exposes cluster sizes");
        let info = vec_reader.info().expect("vector info");
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

    // Flat segments (no IVF data) yield None — the SRF emits no rows for them.
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
                .cluster_sizes()
                .is_none(),
            "flat segments expose no cluster sizes"
        );
    }
    Ok(())
}
