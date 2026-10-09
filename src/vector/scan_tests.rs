//! Segment-scan tests over IVF fixtures large enough to exercise read planning: thousands of
//! rows, deterministic clusters, RAM storage (byte costs) and paged storage (block costs).

use std::collections::HashSet;
use std::sync::Arc;

use super::*;
use crate::index::IndexSettings;
use crate::indexer::NoMergePolicy;
use crate::query::{
    AllQuery, BitSetDocSet, ConstScorer, EnableScoring, Explanation, Query, Scorer,
};
use crate::schema::{Schema, STORED, STRING};
use crate::vector::cluster_plan::{LayerCosts, ReadCost};
use crate::vector::storage_io::test_support::PagedDirectory;
use crate::vector::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingVectors, IvfVectors, RouterKind, VectorDType,
    VectorOptions, VectorQuantizationConfig, VectorQuantizationLayer,
};
use crate::{DocAddress, Index, IndexWriter, TantivyDocument};

const DOCS: u32 = 2000;

/// Every 101st document has no vector.
fn has_vector(doc: u32) -> bool {
    doc % 101 != 7
}

fn vector(seed: u32, dim: usize) -> Vec<f32> {
    let mut state = u64::from(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32;
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Evenly strided training rows as centroids; rows go to the nearest centroid by L2.
struct StrideClusterer {
    clusters: usize,
}

impl IvfClusterer for StrideClusterer {
    fn training_sample_ratio(&self) -> f32 {
        1.0
    }

    fn train(
        &self,
        options: &VectorOptions,
        vectors: IvfTrainingVectors,
    ) -> crate::Result<IvfCentroids> {
        let IvfTrainingVectors::F32(batch) = vectors;
        let dims = options.dim();
        let rows = batch.matrix.rows;
        let values = (0..self.clusters)
            .flat_map(|cluster| {
                let row = cluster * rows / self.clusters;
                batch.matrix.values[row * dims..(row + 1) * dims].to_vec()
            })
            .collect();
        Ok(IvfCentroids::F32(IvfMatrix {
            values,
            rows: self.clusters,
            dims,
        }))
    }

    fn assign(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<u32>> {
        let dims = options.dim();
        let IvfVectors::F32(vectors) = vectors;
        let IvfCentroids::F32(centroids) = centroids;
        Ok(vectors
            .matrix
            .values
            .chunks_exact(dims)
            .map(|row| {
                centroids
                    .values
                    .chunks_exact(dims)
                    .map(|centroid| {
                        row.iter()
                            .zip(centroid)
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f32>()
                    })
                    .enumerate()
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap()
                    .0 as u32
            })
            .collect())
    }
}

pub(super) struct Fixture {
    pub(super) index: Index,
    pub(super) field: Field,
    pub(super) metric: Metric,
    pub(super) quantized: bool,
    pub(super) dim: usize,
}

/// The shape of a fixture index.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub(super) metric: Metric,
    /// Quantized layer bit widths; empty for full precision.
    pub(super) schedule: &'static [u8],
    pub(super) dim: usize,
    pub(super) clusters: usize,
    pub(super) paged: bool,
}

impl Shape {
    pub(super) fn new(metric: Metric, schedule: &'static [u8]) -> Self {
        Self {
            metric,
            schedule,
            dim: 64,
            clusters: 16,
            paged: false,
        }
    }

    /// Two clusters of about a thousand d=1024 rows on 8 KiB pages: a full-precision row fills
    /// half a page, so selected rows and whole bands touch very different block counts.
    pub(super) fn paged_1024(metric: Metric, schedule: &'static [u8]) -> Self {
        Self {
            metric,
            schedule,
            dim: 1024,
            clusters: 2,
            paged: true,
        }
    }
}

/// One IVF segment of `DOCS` documents per entry of `picks`; a document of segment `s` carries
/// the `pick` label when `picks[s](doc)` holds.
pub(super) fn fixture_with(shape: Shape, picks: &[fn(u32) -> bool]) -> crate::Result<Fixture> {
    let mut sb = Schema::builder();
    let options = VectorOptions::new(shape.dim, shape.metric).with_dtype(VectorDType::F32);
    let field = sb.add_vector_field("embedding", options.clone());
    let label = sb.add_text_field("label", STRING | STORED);
    let mut settings = IndexSettings {
        vector_clustering_threshold: 1,
        ..IndexSettings::default()
    };
    if !shape.schedule.is_empty() {
        settings.vector_quantization = vec![VectorQuantizationConfig::materialize(
            "embedding".to_string(),
            &options,
            shape
                .schedule
                .iter()
                .enumerate()
                .map(|(layer, &bits)| VectorQuantizationLayer {
                    bits,
                    seed: 0x1111 * (layer as u64 + 1),
                })
                .collect(),
        )?];
    }
    let builder = Index::builder()
        .schema(sb.build())
        .settings(settings)
        .ivf_clusterer(Arc::new(StrideClusterer {
            clusters: shape.clusters,
        }))
        .ivf_router(RouterKind::Rng)?;
    let index = if shape.paged {
        builder.create(PagedDirectory::default())?
    } else {
        builder.create(crate::directory::RamDirectory::create())?
    };
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for (segment, pick) in picks.iter().enumerate() {
        let before: HashSet<_> = index.searchable_segment_ids()?.into_iter().collect();
        for doc in 0..DOCS {
            let mut document = TantivyDocument::new();
            document.add_text(label, format!("s{segment}d{doc}"));
            if pick(doc) {
                document.add_text(label, "pick");
            }
            if has_vector(doc) {
                document.add_vector(field, &vector(doc + 10_000 * segment as u32, shape.dim));
            }
            writer.add_document(document)?;
        }
        writer.commit()?;
        let fresh: Vec<_> = index
            .searchable_segment_ids()?
            .into_iter()
            .filter(|id| !before.contains(id))
            .collect();
        writer.merge(&fresh).wait()?;
    }
    writer.wait_merging_threads()?;
    Ok(Fixture {
        index,
        field,
        metric: shape.metric,
        quantized: !shape.schedule.is_empty(),
        dim: shape.dim,
    })
}

pub(super) fn fixture(shape: Shape) -> crate::Result<Fixture> {
    fixture_with(shape, &[|_| false])
}

pub(super) fn query(dim: usize) -> Vec<f32> {
    vector(99_999, dim)
}

/// Probes every cluster, so routed results depend only on scoring.
pub(super) fn exhaustive() -> AdaptiveProbeParams {
    AdaptiveProbeParams {
        max_probe_fraction: 1.0,
        min_probe_clusters: usize::MAX / 2,
        ..Default::default()
    }
}

/// `count` documents spread evenly over the segment, ascending.
pub(super) fn spread(max_doc: u32, count: u32) -> Vec<DocId> {
    (0..count).map(|i| i * max_doc / count + 3).collect()
}

/// An exact filter over fixed documents, without a real query.
pub(super) struct FixedDocsWeight {
    pub(super) max_doc: DocId,
    pub(super) docs: Vec<DocId>,
}

impl Weight for FixedDocsWeight {
    fn scorer(&self, _reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let mut bs = BitSet::with_max_value(self.max_doc);
        for &doc in &self.docs {
            bs.insert(doc);
        }
        Ok(Box::new(ConstScorer::new(BitSetDocSet::from(bs), boost)))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> crate::Result<Explanation> {
        unreachable!("the vector backend never explains filter docs")
    }
}

/// The segment backend with the quantized query a collector would prepare.
pub(super) fn backend(
    fx: &Fixture,
    segment_reader: &SegmentReader,
    params: AdaptiveProbeParams,
) -> crate::Result<VectorBackend<f32>> {
    let quantized = segment_reader
        .vector_index(fx.field)?
        .quantization()
        .map(|field| {
            Arc::new(QuantizedQueryCtx::new(
                Arc::clone(field.index_ctx()),
                query(fx.dim),
            ))
        });
    assert_eq!(quantized.is_some(), fx.quantized, "fixture quantization");
    VectorBackend::<f32>::for_segment(
        segment_reader,
        0,
        fx.field,
        VectorQuery::new(Arc::new(query(fx.dim)), quantized),
        params,
    )
}

pub(super) fn run_weight(
    fx: &Fixture,
    weight: &dyn Weight,
    k: usize,
    params: AdaptiveProbeParams,
) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
    let searcher = fx.index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    backend(fx, segment_reader, params)?.top_n(weight, segment_reader, k)
}

/// Runs the segment's first-segment search over a fixed filter.
pub(super) fn run(
    fx: &Fixture,
    docs: &[DocId],
    k: usize,
    params: AdaptiveProbeParams,
) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
    let max_doc = fx.index.reader()?.searcher().segment_readers()[0].max_doc();
    let weight = FixedDocsWeight {
        max_doc,
        docs: docs.to_vec(),
    };
    run_weight(fx, &weight, k, params)
}

/// Runs the first segment's search without a filter.
pub(super) fn run_all(
    fx: &Fixture,
    k: usize,
    params: AdaptiveProbeParams,
) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
    let searcher = fx.index.reader()?.searcher();
    let weight = AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))?;
    run_weight(fx, weight.as_ref(), k, params)
}

/// The exact top-k over every live document with a vector that `filter` admits (all documents
/// when `None`), across segments.
pub(super) fn oracle(
    fx: &Fixture,
    filter: Option<&HashSet<DocAddress>>,
    k: usize,
) -> crate::Result<Vec<(Score, DocAddress)>> {
    let prepared = PreparedQuery::<f32>::new(fx.metric, Arc::new(query(fx.dim)));
    let searcher = fx.index.reader()?.searcher();
    let mut scored = Vec::new();
    for (segment_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
        let vectors = segment_reader.vector_index(fx.field)?;
        let alive = segment_reader.alive_bitset();
        for doc in 0..segment_reader.max_doc() {
            let address = DocAddress::new(segment_ord as u32, doc);
            if alive.is_some_and(|alive| !alive.is_alive(doc))
                || filter.is_some_and(|filter| !filter.contains(&address))
            {
                continue;
            }
            if let Some(bytes) = vectors.vector_bytes(doc)? {
                scored.push((prepared.score_doc_bytes(&bytes), address));
            }
        }
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.truncate(k);
    Ok(scored)
}

pub(super) fn addresses(docs: &[DocId]) -> HashSet<DocAddress> {
    docs.iter().map(|&doc| DocAddress::new(0, doc)).collect()
}

// ============================================================
// Read plans: sparse and full layer reads.
// ============================================================

fn costs(full: usize, sparse: Option<usize>) -> LayerCosts {
    LayerCosts {
        sparse: sparse.map(ReadCost),
        full: ReadCost(full),
    }
}

/// Sparse only when strictly cheaper than the whole band; a whole-cluster selection has no
/// sparse read.
#[test]
fn read_plan_is_strictly_sparse_and_never_for_all_rows() {
    assert_eq!(costs(5, Some(4)).plan(), ReadPlan::Sparse);
    assert_eq!(
        costs(5, Some(5)).plan(),
        ReadPlan::Full,
        "a tie reads the band"
    );
    assert_eq!(costs(5, Some(6)).plan(), ReadPlan::Full);
    assert_eq!(costs(5, None).plan(), ReadPlan::Full, "every row selected");
    assert_eq!(costs(1000, Some(999)).plan(), ReadPlan::Sparse);
}

/// The costs follow the layout: bytes without block geometry, blocks with it, and a whole
/// cluster never has a sparse read.
#[test]
fn read_costs_follow_the_layout() -> crate::Result<()> {
    // Plain storage counts bytes: one row's code plus the sidecar and norms undercut the band.
    let fx = fixture(Shape::new(Metric::L2, &[1, 4]))?;
    let searcher = fx.index.reader()?.searcher();
    let reader = searcher.segment_readers()[0].vector_index(fx.field)?;
    let layers = reader.quantization().expect("quantized").layers();
    let one = layers[0].read_costs(0, Some(&[0]), &mut Vec::new(), &mut Vec::new())?;
    assert_eq!(one.plan(), ReadPlan::Sparse, "{one:?}");
    let all = layers[0].read_costs(0, None, &mut Vec::new(), &mut Vec::new())?;
    assert_eq!(all.sparse, None);
    assert_eq!(all.full, one.full);
    let refine = layers[1].read_costs(0, Some(&[0]), &mut Vec::new(), &mut Vec::new())?;
    assert_eq!(refine.plan(), ReadPlan::Sparse, "{refine:?}");

    // Paged storage counts blocks. One row touches far fewer blocks than the band; nine rows in
    // ten touch at least as many.
    let fx = fixture(Shape::paged_1024(Metric::L2, &[1, 4]))?;
    let searcher = fx.index.reader()?.searcher();
    let reader = searcher.segment_readers()[0].vector_index(fx.field)?;
    let rows = reader.index().expect("ivf").cluster_range(0).len();
    for layer in reader.quantization().expect("quantized").layers() {
        let one = layer.read_costs(0, Some(&[0]), &mut Vec::new(), &mut Vec::new())?;
        assert_eq!(one.plan(), ReadPlan::Sparse, "{one:?}");
        let dense: Vec<usize> = (0..rows).filter(|row| row % 10 != 0).collect();
        let dense = layer.read_costs(0, Some(&dense), &mut Vec::new(), &mut Vec::new())?;
        assert_eq!(dense.plan(), ReadPlan::Full, "{dense:?}");
        let every: Vec<usize> = (0..rows).collect();
        let every = layer.read_costs(0, Some(&every), &mut Vec::new(), &mut Vec::new())?;
        assert!(every.sparse.unwrap() >= every.full, "{every:?}");
    }
    Ok(())
}

/// Runs `search` once with every quantized read forced full and once forced sparse, and checks
/// that candidate columns, estimates, boundaries and results agree bit for bit. Only row
/// selections read sparsely: a single-layer unfiltered scan has none.
fn assert_plans_agree(
    context: &str,
    expect_sparse: bool,
    search: impl Fn() -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)>,
) -> crate::Result<ProbeStats> {
    let (full_hits, full) = {
        let _forced = force_read_plan(ReadPlan::Full);
        search()?
    };
    let (sparse_hits, sparse) = {
        let _forced = force_read_plan(ReadPlan::Sparse);
        search()?
    };
    let layers = |stats: &ProbeStats, read: fn(&LayerProbeStats) -> usize| {
        (0..stats.layers.0.len())
            .map(|layer| stats.layers.get(layer).map_or(0, read))
            .sum::<usize>()
    };
    assert_eq!(
        layers(&full, LayerProbeStats::sparse_clusters),
        0,
        "{context}"
    );
    assert_eq!(
        layers(&sparse, LayerProbeStats::sparse_clusters) > 0,
        expect_sparse,
        "{context}: {sparse:?}"
    );
    assert!(!full.quantized_trace.layer_columns.is_empty(), "{context}");
    assert_eq!(
        sparse.quantized_trace.layer_columns, full.quantized_trace.layer_columns,
        "{context}"
    );
    assert_eq!(
        sparse.quantized_trace.estimate_rows, full.quantized_trace.estimate_rows,
        "{context}"
    );
    assert_eq!(
        sparse.quantized_trace.boundary_rows, full.quantized_trace.boundary_rows,
        "{context}"
    );
    assert_eq!(sparse_hits, full_hits, "{context}");
    Ok(full)
}

/// Sparse and full reads give bit-identical candidate columns, boundaries and results at every
/// layer, for every metric, on byte-costed and block-costed storage.
#[test]
fn sparse_and_full_reads_are_bit_identical() -> crate::Result<()> {
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        for shape in [
            Shape::new(metric, &[1]),
            Shape::new(metric, &[1, 1]),
            Shape::new(metric, &[1, 4]),
            Shape::new(metric, &[2, 4]),
            Shape::paged_1024(metric, &[1, 4]),
        ] {
            let fx = fixture(shape)?;
            let max_doc = fx.index.reader()?.searcher().segment_readers()[0].max_doc();
            for count in [20, 200] {
                let docs = spread(max_doc, count);
                let context = format!(
                    "{metric:?} schedule={:?} dim={} matches={count}",
                    shape.schedule, shape.dim
                );
                assert_plans_agree(&context, true, || run(&fx, &docs, 10, exhaustive()))?;
                let (hits, _) = run(&fx, &docs, 10, exhaustive())?;
                assert_eq!(hits, oracle(&fx, Some(&addresses(&docs)), 10)?, "{context}");
            }
            let context = format!("{metric:?} schedule={:?} unfiltered", shape.schedule);
            let refines = shape.schedule.len() > 1;
            assert_plans_agree(&context, refines, || run_all(&fx, 10, exhaustive()))?;
        }
    }
    Ok(())
}

/// A whole-cluster selection reads the band even when sparse reads are forced.
#[test]
fn whole_cluster_selections_read_the_band() -> crate::Result<()> {
    let fx = fixture(Shape::new(Metric::L2, &[1, 4]))?;
    let _forced = force_read_plan(ReadPlan::Sparse);
    let (_, stats) = run_all(&fx, 10, exhaustive())?;
    let layer0 = stats.layers.get(0).expect("layer 0");
    assert_eq!(layer0.sparse_clusters(), 0, "{stats:?}");
    assert_eq!(layer0.full_clusters(), stats.postings_row, "{stats:?}");
    Ok(())
}

/// Left to the layout, a selective filter reads layer 0 sparsely on paged d=1024 storage and a
/// dense one reads whole bands; both return the exact top-k under an exhaustive probe.
#[test]
fn layout_picks_sparse_for_selective_filters() -> crate::Result<()> {
    let fx = fixture(Shape::paged_1024(Metric::L2, &[1, 4]))?;
    let max_doc = fx.index.reader()?.searcher().segment_readers()[0].max_doc();
    let sparse_docs = spread(max_doc, 4);
    let (hits, stats) = run(&fx, &sparse_docs, 10, exhaustive())?;
    let layer0 = stats.layers.get(0).expect("layer 0");
    assert_eq!(layer0.sparse_clusters(), stats.postings_row, "{stats:?}");
    assert_eq!(hits, oracle(&fx, Some(&addresses(&sparse_docs)), 10)?);
    let dense_docs: Vec<DocId> = (0..max_doc).filter(|doc| doc % 10 != 0).collect();
    let (hits, stats) = run(&fx, &dense_docs, 10, exhaustive())?;
    assert_eq!(
        stats.layers.get(0).unwrap().sparse_clusters(),
        0,
        "{stats:?}"
    );
    assert_eq!(hits, oracle(&fx, Some(&addresses(&dense_docs)), 10)?);
    Ok(())
}
