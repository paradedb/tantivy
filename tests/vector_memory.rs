//! Allocation calls and bytes of vector searches, counted by a global allocator on the searching
//! thread only.
//!
//! The scenarios run one at a time: the counters are global.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicIsize, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tantivy::collector::TopDocs;
use tantivy::index::IndexSettings;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::{AllQuery, Query, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, Metric, Schema, VectorDType, VectorOptions, STRING,
};
use tantivy::vector::clustering::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingVectors, IvfVectors,
};
use tantivy::vector::{
    LocatedPath, ProbeParams, ProbeStats, Selectivity, VectorQuantizationConfig,
    VectorQuantizationLayer,
};
use tantivy::{Index, IndexWriter, Searcher, TantivyDocument, Term};

// --- Counting allocator ---

/// Counts allocation calls and bytes while the current thread measures.
struct Counting;

static CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

thread_local! {
    /// Set while this thread's allocations are counted.
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread is measuring; `false` once its thread-locals are gone.
fn measuring() -> bool {
    MEASURING.try_with(Cell::get).unwrap_or(false)
}

fn on_alloc(size: usize) {
    if measuring() {
        CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED.fetch_add(size as u64, Ordering::Relaxed);
        let live = LIVE.fetch_add(size as isize, Ordering::Relaxed) + size as isize;
        PEAK.fetch_max(live, Ordering::Relaxed);
    }
}

fn on_dealloc(size: usize) {
    if measuring() {
        LIVE.fetch_sub(size as isize, Ordering::Relaxed);
    }
}

// SAFETY: every method passes its arguments unchanged to `System`, which upholds the
// `GlobalAlloc` contract. The bookkeeping touches only atomics and a const-initialized
// thread-local without a destructor, so it neither allocates nor reenters the allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        on_alloc(layout.size());
        // SAFETY: the caller upholds `alloc`'s contract for `layout`, which passes through.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        on_alloc(layout.size());
        // SAFETY: the caller upholds `alloc_zeroed`'s contract for `layout`, which passes through.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        on_dealloc(layout.size());
        // SAFETY: `ptr` was returned by this allocator, so by `System`, for `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        on_dealloc(layout.size());
        on_alloc(new_size);
        // SAFETY: `ptr` was returned by this allocator, so by `System`, for `layout`, and the
        // caller upholds `realloc`'s contract for `new_size`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The counters are global: one scenario at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// What one operation allocated on this thread.
#[derive(Clone, Copy, Debug)]
struct Allocations {
    calls: u64,
    /// Bytes requested; a reallocation counts its new size.
    bytes: u64,
    /// The most bytes live at once, above those live when the operation started.
    peak: isize,
}

/// Runs `op`, counting its allocations on this thread.
fn count_allocations<R>(op: impl FnOnce() -> R) -> (R, Allocations) {
    CALLS.store(0, Ordering::Relaxed);
    ALLOCATED.store(0, Ordering::Relaxed);
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    MEASURING.set(true);
    let out = op();
    MEASURING.set(false);
    let allocations = Allocations {
        calls: CALLS.load(Ordering::Relaxed),
        bytes: ALLOCATED.load(Ordering::Relaxed),
        peak: PEAK.load(Ordering::Relaxed),
    };
    (out, allocations)
}

// --- Fixture ---

/// Centroids are the first training rows, one per center; row `i` goes to cluster
/// `i % clusters`, its center's, so every cluster holds the same number of rows.
struct RoundRobinClusterer {
    clusters: usize,
}

impl IvfClusterer for RoundRobinClusterer {
    fn training_sample_ratio(&self) -> f32 {
        1.0
    }

    fn train(
        &self,
        options: &VectorOptions,
        vectors: IvfTrainingVectors,
    ) -> tantivy::Result<IvfCentroids> {
        let IvfTrainingVectors::F32(batch) = vectors;
        let dims = options.dim();
        Ok(IvfCentroids::F32(IvfMatrix {
            values: batch.matrix.values[..self.clusters * dims].to_vec(),
            rows: self.clusters,
            dims,
        }))
    }

    fn assign(
        &self,
        _options: &VectorOptions,
        vectors: IvfVectors<'_>,
        _centroids: &IvfCentroids,
    ) -> tantivy::Result<Vec<u32>> {
        let IvfVectors::F32(vectors) = vectors;
        Ok((0..vectors.matrix.rows)
            .map(|row| (row % self.clusters) as u32)
            .collect())
    }
}

/// A deterministic vector with coordinates in `[-1, 1)`.
fn vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32;
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Center `doc % centers` plus noise as wide as the centers' spread, so clusters overlap and the
/// bounds gate rarely skips one.
fn doc_vector(doc: u64, centers: u64, dim: usize) -> Vec<f32> {
    let center = vector(1_000_000 + doc % centers, dim);
    let noise = vector(doc, dim);
    center.iter().zip(&noise).map(|(c, n)| c + n).collect()
}

struct Shape {
    dim: usize,
    clusters: usize,
    docs_per_segment: u64,
    segments: usize,
    /// Docs deleted after the build.
    deleted: fn(u64) -> bool,
}

struct Fixture {
    index: Index,
    field: Field,
    label: Field,
    shape: Shape,
}

impl Fixture {
    /// A quantized (`[1, 4]`) IVF index of `shape.segments` same-shape segments. Labels: `p30` on
    /// three docs in ten, `g{n}` on docs of the first `n` clusters for `n` in 16, 32 and 64, and
    /// `d{doc}`.
    fn build(shape: Shape) -> tantivy::Result<Self> {
        let mut builder = Schema::builder();
        let options = VectorOptions::new(shape.dim, Metric::L2).with_dtype(VectorDType::F32);
        let field = builder.add_vector_field("embedding", options.clone());
        let label = builder.add_text_field("label", STRING);
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            vector_quantization: vec![VectorQuantizationConfig::materialize(
                "embedding".to_string(),
                &options,
                vec![
                    VectorQuantizationLayer { bits: 1, seed: 7 },
                    VectorQuantizationLayer { bits: 4, seed: 11 },
                ],
            )?],
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(builder.build())
            .settings(settings)
            .ivf_clusterer(Arc::new(RoundRobinClusterer {
                clusters: shape.clusters,
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 1_000_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let centers = shape.clusters as u64;
        for segment in 0..shape.segments as u64 {
            let before = index.searchable_segment_ids()?;
            for local in 0..shape.docs_per_segment {
                let doc = segment * shape.docs_per_segment + local;
                let mut document = TantivyDocument::new();
                if doc % 10 < 3 {
                    document.add_text(label, "p30");
                }
                for first in [16, 32, 64] {
                    if doc % centers < first {
                        document.add_text(label, format!("g{first}"));
                    }
                }
                document.add_text(label, format!("d{doc}"));
                document.add_vector(field, &doc_vector(doc, centers, shape.dim));
                writer.add_document(document)?;
            }
            writer.commit()?;
            // The commit's own segments, merged alone, become one IVF segment.
            let fresh: Vec<_> = index
                .searchable_segment_ids()?
                .into_iter()
                .filter(|id| !before.contains(id))
                .collect();
            writer.merge(&fresh).wait()?;
        }
        let total = shape.segments as u64 * shape.docs_per_segment;
        for doc in (0..total).filter(|&doc| (shape.deleted)(doc)) {
            writer.delete_term(Term::from_field_text(label, &format!("d{doc}")));
        }
        writer.commit()?;
        writer.wait_merging_threads()?;
        Ok(Self {
            index,
            field,
            label,
            shape,
        })
    }

    fn term(&self, text: &str) -> Box<dyn Query> {
        Box::new(TermQuery::new(
            Term::from_field_text(self.label, text),
            IndexRecordOption::Basic,
        ))
    }

    /// Searches `filter` with `params`; returns the stats and what the search allocated, after
    /// one unmeasured search with the same arguments.
    fn search(
        &self,
        searcher: &Searcher,
        filter: &dyn Query,
        params: &ProbeParams,
    ) -> (Vec<ProbeStats>, Allocations) {
        let query = vector(77, self.shape.dim);
        let run = || {
            let collector = TopDocs::with_limit(10)
                .order_by_similarity(self.field, query.clone())
                .with_probe_params(params.clone());
            searcher.search(filter, &collector).map(|fruit| fruit.stats)
        };
        run().expect("warm-up search");
        let (stats, allocations) = count_allocations(run);
        (stats.expect("measured search"), allocations)
    }
}

/// Probes about `clusters` clusters: the budget is that many average clusters of work.
fn probing(clusters: usize, located: LocatedPath) -> ProbeParams {
    ProbeParams {
        max_probe_fraction: 1e-6,
        min_probe_clusters: clusters,
        located,
        ..ProbeParams::default()
    }
}

fn located_everything() -> LocatedPath {
    LocatedPath::MaxSelectivity(Selectivity::new(1.0).expect("a selectivity"))
}

// --- Scenarios ---

/// One search's allocation calls don't grow with the clusters it probes, on the routed path
/// with and without a filter and deletes, and on the located path.
#[test]
fn allocations_do_not_grow_with_clusters_probed() -> tantivy::Result<()> {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let shape = |deleted| Shape {
        dim: 64,
        clusters: 128,
        docs_per_segment: 128 * 64,
        segments: 1,
        deleted,
    };
    let clean = Fixture::build(shape(|_| false))?;
    let deleted = Fixture::build(shape(|doc| doc % 17 == 5))?;
    let probes = [16, 32, 64];

    let mut cases: Vec<(&str, Vec<(usize, Allocations)>)> = Vec::new();
    let searcher = clean.index.reader()?.searcher();
    cases.push((
        "routed, unfiltered",
        probes
            .iter()
            .map(|&probe| {
                let (stats, allocations) =
                    clean.search(&searcher, &AllQuery, &probing(probe, LocatedPath::Off));
                (stats[0].clusters_probed(), allocations)
            })
            .collect(),
    ));
    let searcher = deleted.index.reader()?.searcher();
    let filter = deleted.term("p30");
    cases.push((
        "routed, filter and deletes",
        probes
            .iter()
            .map(|&probe| {
                let (stats, allocations) = deleted.search(
                    &searcher,
                    filter.as_ref(),
                    &probing(probe, LocatedPath::Off),
                );
                (stats[0].clusters_probed(), allocations)
            })
            .collect(),
    ));
    cases.push((
        "located, filter and deletes",
        probes
            .iter()
            .map(|&centers| {
                let filter = deleted.term(&format!("g{centers}"));
                let (stats, allocations) = deleted.search(
                    &searcher,
                    filter.as_ref(),
                    &probing(1, located_everything()),
                );
                (stats[0].located_clusters, allocations)
            })
            .collect(),
    ));

    for (case, runs) in &cases {
        for (clusters, allocations) in runs {
            println!(
                "vector_memory: {case}: {clusters} clusters, {} calls, {} bytes, {} peak",
                allocations.calls, allocations.bytes, allocations.peak
            );
        }
        let probed: Vec<usize> = runs.iter().map(|(clusters, _)| *clusters).collect();
        assert!(
            probed.windows(2).all(|pair| pair[0] < pair[1]),
            "{case}: each run probes more clusters: {probed:?}"
        );
        // An allocation per cluster would add a call per added cluster. What remains are buffers
        // growing by doubling: the admitted clusters, the filter's matches and located clusters,
        // and read scratch growing to the largest selection.
        let (first, last) = (&runs[0], &runs[runs.len() - 1]);
        let added_clusters = (last.0 - first.0) as u64;
        let added_calls = last.1.calls.saturating_sub(first.1.calls);
        assert!(
            4 * added_calls < added_clusters,
            "{case}: {added_calls} more allocation calls for {added_clusters} more clusters"
        );
    }
    Ok(())
}

/// Allocation calls and bytes of one segment's search, from an index of four same-shape
/// segments against one of them alone. Reported only.
#[test]
fn allocations_per_segment() -> tantivy::Result<()> {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let shape = |segments| Shape {
        dim: 64,
        clusters: 32,
        docs_per_segment: 32 * 64,
        segments,
        deleted: |_| false,
    };
    let params = probing(16, LocatedPath::Off);
    let mut measured = Vec::new();
    for segments in [1, 4] {
        let fixture = Fixture::build(shape(segments))?;
        let searcher = fixture.index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), segments);
        let (_, allocations) = fixture.search(&searcher, &AllQuery, &params);
        println!(
            "vector_memory: {segments} segments: {} calls, {} bytes, {} peak",
            allocations.calls, allocations.bytes, allocations.peak
        );
        measured.push(allocations);
    }
    let (one, four) = (measured[0], measured[1]);
    println!(
        "vector_memory: per added segment: {} calls, {} bytes",
        (four.calls - one.calls) / 3,
        (four.bytes - one.bytes) / 3
    );
    Ok(())
}

/// Peak and total allocated bytes of a search over about 100k 128-dimensional vectors in 256
/// clusters, probing about a quarter of them under a 30% filter with 5% deleted, on both paths.
/// Printed only; run with `--ignored --nocapture`.
#[test]
#[ignore]
fn peak_on_a_large_segment() -> tantivy::Result<()> {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let fixture = Fixture::build(Shape {
        dim: 128,
        clusters: 256,
        docs_per_segment: 100_000,
        segments: 1,
        deleted: |doc| doc % 20 == 7,
    })?;
    let searcher = fixture.index.reader()?.searcher();
    let filter = fixture.term("p30");
    for (path, located) in [
        ("routed", LocatedPath::Off),
        ("located", located_everything()),
    ] {
        let params = ProbeParams {
            max_probe_fraction: 0.25,
            min_probe_clusters: 1,
            located,
            ..ProbeParams::default()
        };
        let (stats, allocations) = fixture.search(&searcher, filter.as_ref(), &params);
        println!(
            "vector_memory: peak, {path}: {} clusters probed, {} located, {} peak bytes, {} \
             allocated bytes, {} calls",
            stats[0].clusters_probed(),
            stats[0].located_clusters,
            allocations.peak,
            allocations.bytes,
            allocations.calls
        );
    }
    Ok(())
}
