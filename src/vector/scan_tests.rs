//! Segment-scan tests over IVF fixtures large enough to exercise read planning: thousands of
//! rows, deterministic clusters, RAM storage (byte costs) and paged storage (block costs).

use std::collections::HashSet;
use std::sync::Arc;

use super::*;
use crate::collector::TopDocs;
use crate::index::IndexSettings;
use crate::indexer::NoMergePolicy;
use crate::query::{
    AllQuery, BitSetDocSet, ConstScorer, EnableScoring, Explanation, Query, Scorer, TermQuery,
};
use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
use crate::vector::cluster_plan::{BatchCosts, LayerCosts, ReadCost};
use crate::vector::ivf::{Candidate, DirectRead, WorkModel};
use crate::vector::storage_io::test_support::{doc_ids_reads, PagedDirectory};
use crate::vector::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingVectors, IvfVectors, RouterKind, Stage,
    VectorDType, VectorOptions, VectorQuantizationConfig, VectorQuantizationLayer,
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
    pub(super) label: Field,
    pub(super) metric: Metric,
    pub(super) quantized: bool,
    pub(super) dim: usize,
    /// The read log of paged storage.
    pub(super) directory: Option<PagedDirectory>,
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
    let directory = shape.paged.then(PagedDirectory::default);
    let index = match &directory {
        Some(directory) => builder.create(directory.clone())?,
        None => builder.create(crate::directory::RamDirectory::create())?,
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
        label,
        metric: shape.metric,
        quantized: !shape.schedule.is_empty(),
        dim: shape.dim,
        directory,
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

/// Deletes documents of the first segment by their unique label.
pub(super) fn delete(fx: &Fixture, docs: &[DocId]) -> crate::Result<()> {
    let mut writer: IndexWriter = fx.index.writer_with_num_threads(1, 15_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for doc in docs {
        writer.delete_term(Term::from_field_text(fx.label, &format!("s0d{doc}")));
    }
    writer.commit()?;
    Ok(())
}

/// [`exhaustive`] with the exact plan on or off.
pub(super) fn with_exact(exact_plan: bool) -> AdaptiveProbeParams {
    AdaptiveProbeParams {
        exact_plan,
        ..exhaustive()
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
        centroid: ReadCost(0),
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

// ============================================================
// The exact plan: clusters finished at full precision before `Final`.
// ============================================================

/// Exact only when strictly cheaper than the cheaper quantized read; ties stay quantized, and
/// `Final` is always exact.
#[test]
fn batch_plan_takes_exact_only_when_strictly_cheaper() {
    let batch = |exact: usize, layer: Option<LayerCosts>| BatchCosts {
        exact: ReadCost(exact),
        layer,
    };
    let layer = Some(costs(10, Some(4)));
    assert_eq!(batch(3, layer).plan(true), ReadPlan::Exact);
    assert_eq!(
        batch(4, layer).plan(true),
        ReadPlan::Sparse,
        "a tie stays quantized"
    );
    assert_eq!(
        batch(3, layer).plan(false),
        ReadPlan::Sparse,
        "exact plan off"
    );
    assert_eq!(batch(9, Some(costs(10, None))).plan(true), ReadPlan::Exact);
    assert_eq!(batch(10, Some(costs(10, None))).plan(true), ReadPlan::Full);
    assert_eq!(batch(5, Some(costs(4, Some(6)))).plan(true), ReadPlan::Full);
    assert_eq!(
        batch(usize::MAX, None).plan(false),
        ReadPlan::Exact,
        "Final"
    );
}

fn layer_sum(stats: &ProbeStats, read: fn(&LayerProbeStats) -> usize) -> usize {
    (0..stats.layers.0.len())
        .map(|layer| stats.layers.get(layer).map_or(0, read))
        .sum()
}

/// Scoring every cluster exactly at admission returns the exhaustive top-k for every metric,
/// filtered or not, with and without deletes, and never keeps a quantized candidate.
#[test]
fn forced_exact_matches_the_oracle() -> crate::Result<()> {
    let _forced = force_exact(|_, _| true);
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        let fx = fixture(Shape::new(metric, &[1, 4]))?;
        let max_doc = fx.index.reader()?.searcher().segment_readers()[0].max_doc();
        let docs = spread(max_doc, 200);
        for deletes in [false, true] {
            if deletes {
                delete(&fx, &[docs[0], docs[1], 5, 6, 7])?;
            }
            for filtered in [false, true] {
                for k in [1, 3, 8] {
                    let context = format!("{metric:?} deletes={deletes} filtered={filtered} k={k}");
                    let (hits, stats) = if filtered {
                        run(&fx, &docs, k, exhaustive())?
                    } else {
                        run_all(&fx, k, exhaustive())?
                    };
                    let filter = filtered.then(|| addresses(&docs));
                    assert_eq!(hits, oracle(&fx, filter.as_ref(), k)?, "{context}");
                    let layer0 = stats.layers.get(0).expect("layer 0");
                    assert_eq!(layer0.exact_clusters(), stats.postings_row, "{context}");
                    assert_eq!(layer0.exact_rows(), stats.candidates_scored, "{context}");
                    assert_eq!(layer0.scored(), 0, "{context}");
                    assert_eq!(stats.rerank_rows, 0, "{context}");
                    assert_eq!(
                        stats.vectors_visited,
                        stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
                        "{context}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Exact and quantized clusters in one scan, at layer 0 or a later layer, rank together: the
/// top-k equals the all-exact top-k.
#[test]
fn mixed_exact_and_quantized_match_all_exact() -> crate::Result<()> {
    let rules: [ExactRule; 3] = [
        |cluster, _| cluster % 2 == 0,
        |_, layer| layer == 1,
        |cluster, layer| layer == 1 && cluster % 3 == 0,
    ];
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        let fx = fixture(Shape::new(metric, &[1, 4]))?;
        let max_doc = fx.index.reader()?.searcher().segment_readers()[0].max_doc();
        let docs = spread(max_doc, 300);
        for k in [1, 3, 8] {
            let (all_exact, _) = {
                let _forced = force_exact(|_, _| true);
                run(&fx, &docs, k, exhaustive())?
            };
            assert_eq!(all_exact, oracle(&fx, Some(&addresses(&docs)), k)?);
            for (index, rule) in rules.iter().enumerate() {
                let _forced = force_exact(*rule);
                let (hits, stats) = run(&fx, &docs, k, exhaustive())?;
                let context = format!("{metric:?} rule={index} k={k}");
                assert_eq!(hits, all_exact, "{context}");
                assert!(
                    layer_sum(&stats, LayerProbeStats::exact_clusters) > 0,
                    "{context}"
                );
            }
        }
    }
    Ok(())
}

/// A cluster finished exactly never reads its band; its rows land in the `exact` IO slot.
#[test]
fn exact_clusters_never_read_their_band() -> crate::Result<()> {
    let rules: [(ExactRule, bool); 3] = [
        (|_, _| true, true),
        (|cluster, _| cluster % 2 == 0, false),
        (|_, _| false, false),
    ];
    let shape = Shape {
        paged: true,
        ..Shape::new(Metric::L2, &[1])
    };
    let fx = fixture(shape)?;
    for (rule, all) in rules {
        let _forced = force_exact(rule);
        let _full = force_read_plan(ReadPlan::Full);
        let directory = fx.directory.as_ref().expect("paged");
        directory.reads.lock().unwrap().clear();
        let (hits, stats) = run_all(&fx, 10, exhaustive())?;
        assert_eq!(hits, oracle(&fx, None, 10)?);
        let reads = directory.reads.lock().unwrap();
        let count = |stage: Stage| reads.iter().filter(|(read, _)| *read == stage).count();
        let layer0 = stats.layers.get(0).expect("layer 0");
        // Unfiltered admission reads no DocIds: layer 0 reads are bands alone.
        assert_eq!(
            count(Stage::LayerScan(0)),
            layer0.full_clusters(),
            "{stats:?}"
        );
        assert_eq!(
            layer0.full_clusters() + layer0.exact_clusters(),
            stats.postings_row
        );
        assert_eq!(
            stats.exact_io.reads,
            count(Stage::Exact) as u64,
            "{stats:?}"
        );
        assert_eq!(stats.exact_io.reads > 0, layer0.exact_clusters() > 0);
        if all {
            assert_eq!(layer0.full_clusters(), 0);
        }
    }
    Ok(())
}

/// A group finished exactly at a later layer leaves the candidates before that layer's band:
/// none of its rows reaches a later boundary, and the top-k is unchanged.
#[test]
fn retired_rows_never_reach_a_later_band() -> crate::Result<()> {
    let fx = fixture(Shape::new(Metric::L2, &[1, 4]))?;
    let searcher = fx.index.reader()?.searcher();
    let vectors = searcher.segment_readers()[0].vector_index(fx.field)?;
    let index = vectors.index().expect("ivf");
    let max_doc = searcher.segment_readers()[0].max_doc();
    let docs = spread(max_doc, 400);
    let _forced = force_exact(|cluster, layer| layer == 1 && cluster % 2 == 0);
    let (hits, stats) = run(&fx, &docs, 10, exhaustive())?;
    assert_eq!(hits, oracle(&fx, Some(&addresses(&docs)), 10)?);
    let layer1 = stats.layers.get(1).expect("layer 1");
    assert!(layer1.exact_clusters() > 0, "{stats:?}");
    let even_cluster = |row: &usize| {
        (0..index.num_clusters())
            .find(|&cluster| index.cluster_range(cluster).contains(row))
            .is_some_and(|cluster| cluster % 2 == 0)
    };
    let trace = &stats.quantized_trace;
    assert!(trace.boundary_rows[0].iter().any(even_cluster));
    assert!(!trace.boundary_rows[1].iter().any(even_cluster));
    assert!(!trace.layer_columns[1]
        .iter()
        .any(|(row, _)| even_cluster(row)));
    assert_eq!(
        layer1.scored() + layer1.exact_rows(),
        trace.boundary_rows[0].len()
    );
    Ok(())
}

/// Left to the layout on paged d=1024 storage, a selective filter finishes clusters exactly
/// (one row is half a page, less than any layer-0 read), an unfiltered query never does, and
/// both return the exhaustive top-k.
#[test]
fn layout_takes_exact_for_selective_filters_only() -> crate::Result<()> {
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        let fx = fixture(Shape::paged_1024(metric, &[1, 4]))?;
        let max_doc = fx.index.reader()?.searcher().segment_readers()[0].max_doc();
        let docs = spread(max_doc, 4);
        let (hits, stats) = run(&fx, &docs, 10, with_exact(true))?;
        assert_eq!(
            hits,
            oracle(&fx, Some(&addresses(&docs)), 10)?,
            "{metric:?}"
        );
        let layer0 = stats.layers.get(0).expect("layer 0");
        assert_eq!(layer0.exact_clusters(), stats.postings_row, "{stats:?}");
        let (hits, stats) = run_all(&fx, 10, with_exact(true))?;
        assert_eq!(hits, run_all(&fx, 10, with_exact(false))?.0, "{metric:?}");
        assert_eq!(
            layer_sum(&stats, LayerProbeStats::exact_clusters),
            0,
            "{stats:?}"
        );
    }
    Ok(())
}

// ============================================================
// The located source: sparse filters through the location map.
// ============================================================

/// Full precision, and two quantized layers.
const SCHEDULES: [&[u8]; 2] = [&[], &[1, 4]];

/// [`exhaustive`] with a located-path threshold and the exact plan on or off.
fn located(direct_max_selectivity: f32, exact_plan: bool) -> AdaptiveProbeParams {
    AdaptiveProbeParams {
        direct_read: DirectRead::MaxSelectivity(direct_max_selectivity),
        exact_plan,
        ..exhaustive()
    }
}

/// A threshold whose cap is exactly `cap` on a segment of `max_doc` documents.
fn selectivity_for_cap(cap: u32, max_doc: u32) -> f32 {
    (cap as f32 + 0.5) / max_doc as f32
}

/// The located path's identities: every match is admitted, nothing is filtered or routed.
fn assert_located(stats: &ProbeStats) {
    assert_eq!(stats.access_path, AccessPath::Located, "{stats:?}");
    assert_eq!(
        stats.vectors_visited,
        stats.located_matches + stats.pruned_dead,
        "{stats:?}"
    );
    assert_eq!(
        stats.vectors_visited,
        stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
        "{stats:?}"
    );
    assert_eq!(stats.pruned_filter, 0, "{stats:?}");
    assert!(stats.routing.is_none(), "{stats:?}");
    assert_eq!(stats.postings_skipped, 0, "{stats:?}");
    assert_eq!(stats.bounds_skips, 0, "{stats:?}");
    assert_eq!(stats.postings_row, stats.located_clusters, "{stats:?}");
}

fn present(fx: &Fixture, docs: &[DocId]) -> crate::Result<usize> {
    let searcher = fx.index.reader()?.searcher();
    let reader = searcher.segment_readers()[0].vector_index(fx.field)?;
    let mut count = 0;
    for &doc in docs {
        count += usize::from(reader.contains(doc)?);
    }
    Ok(count)
}

/// The located path returns the exhaustive top-k on full-precision and quantized segments, with
/// the exact plan on and off, for every metric and selectivity, and reads no DocIds column.
#[test]
fn located_path_matches_the_oracle() -> crate::Result<()> {
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        for schedule in SCHEDULES {
            let fx = fixture(Shape::new(metric, schedule))?;
            let max_doc = fx.index.reader()?.searcher().segment_readers()[0].max_doc();
            for exact_plan in [false, true] {
                for count in [1, 2, 20, 200] {
                    let docs = spread(max_doc, count);
                    let context = format!(
                        "{metric:?} schedule={schedule:?} exact={exact_plan} matches={count}"
                    );
                    let before = doc_ids_reads();
                    let (hits, stats) = run(&fx, &docs, 10, located(1.0, exact_plan))?;
                    assert_eq!(doc_ids_reads(), before, "{context}: DocIds read");
                    assert_eq!(hits, oracle(&fx, Some(&addresses(&docs)), 10)?, "{context}");
                    assert_located(&stats);
                    assert_eq!(stats.located_matches, present(&fx, &docs)?, "{context}");
                    assert_eq!(
                        stats.located_absent,
                        docs.len() - stats.located_matches,
                        "{context}"
                    );
                    assert_eq!(stats.candidates_scored, stats.located_matches, "{context}");
                    if fx.quantized {
                        let layer0 = stats.layers.get(0).expect("layer 0");
                        // Every cluster not finished exactly reads its centroid row first.
                        assert_eq!(
                            stats.centroid_reads + layer0.exact_clusters(),
                            stats.located_clusters,
                            "{context}"
                        );
                        if !exact_plan {
                            assert_eq!(layer0.scored(), stats.located_matches, "{context}");
                        }
                    } else {
                        assert_eq!(stats.centroid_reads, 0, "{context}");
                    }
                }
            }
        }
    }
    Ok(())
}

/// The drain keeps exactly `cap` matches sparse; one more spills into the bitset a zero cap
/// builds, and a zero cap never builds a sparse filter.
#[test]
fn cap_boundary_spills_to_the_routed_bitset() -> crate::Result<()> {
    let fx = fixture(Shape::new(Metric::L2, &[1, 4]))?;
    let searcher = fx.index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    let max_doc = segment_reader.max_doc();
    let docs = spread(max_doc, 20);
    let weight = FixedDocsWeight {
        max_doc,
        docs: docs.clone(),
    };
    match collect_segment_filter(&weight, segment_reader, max_doc, 20)? {
        SegmentFilter::Sparse(sparse) => assert_eq!(sparse, docs),
        _ => panic!("matches == cap stay sparse"),
    }
    let SegmentFilter::Docs(routed) = collect_segment_filter(&weight, segment_reader, max_doc, 0)?
    else {
        panic!("a zero cap builds the bitset");
    };
    let SegmentFilter::Docs(spilled) =
        collect_segment_filter(&weight, segment_reader, max_doc, 19)?
    else {
        panic!("matches == cap + 1 spill");
    };
    assert_eq!(spilled.len(), routed.len());
    assert!((0..max_doc).all(|doc| spilled.contains(doc) == routed.contains(doc)));
    assert!(docs.iter().all(|&doc| routed.contains(doc)));

    // The same boundary through the threshold: floor(selectivity * max_doc).
    let (_, at_cap) = run(
        &fx,
        &docs,
        10,
        located(selectivity_for_cap(20, max_doc), false),
    )?;
    assert_eq!(at_cap.access_path, AccessPath::Located, "{at_cap:?}");
    let (_, over) = run(
        &fx,
        &docs,
        10,
        located(selectivity_for_cap(19, max_doc), false),
    )?;
    assert_eq!(over.access_path, AccessPath::Routed, "{over:?}");
    let (_, never) = run(&fx, &docs[..1], 10, located(0.0, false))?;
    assert_eq!(never.access_path, AccessPath::Routed, "{never:?}");
    let (_, unfiltered) = run_all(&fx, 10, located(1.0, true))?;
    assert_eq!(
        unfiltered.access_path,
        AccessPath::Routed,
        "unfiltered queries route"
    );
    Ok(())
}

/// Dead matches count toward the cap but are never scored; matches without a vector are
/// absent; an empty filter, `k > m` and `top_n == 0` behave as on the routed path.
#[test]
fn located_deletes_absent_and_edge_cases() -> crate::Result<()> {
    for schedule in SCHEDULES {
        let fx = fixture(Shape::new(Metric::L2, schedule))?;
        let max_doc = fx.index.reader()?.searcher().segment_readers()[0].max_doc();
        let docs: Vec<DocId> = (0..20).map(|i| i * 97 + 5).collect();
        delete(&fx, &docs[..5])?;
        let searcher = fx.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let alive = segment_reader.alive_bitset().expect("deletes landed");
        let live: Vec<DocId> = docs
            .iter()
            .copied()
            .filter(|&d| alive.is_alive(d))
            .collect();
        assert_eq!(live.len(), 15);

        // Twenty matches, five dead: a cap of twenty goes located.
        let at_cap = located(selectivity_for_cap(20, max_doc), false);
        let (hits, stats) = run(&fx, &docs, 50, at_cap)?;
        assert_located(&stats);
        assert_eq!(stats.pruned_dead, 5, "{stats:?}");
        assert_eq!(
            stats.located_matches + stats.located_absent,
            15,
            "{stats:?}"
        );
        assert_eq!(hits, oracle(&fx, Some(&addresses(&live)), 50)?);
        assert_eq!(
            hits.len(),
            stats.located_matches,
            "k > m returns every match"
        );

        // Dead matches count toward the cap: fifteen live matches over a cap of fifteen route.
        let (_, routed) = run(
            &fx,
            &docs,
            10,
            located(selectivity_for_cap(15, max_doc), false),
        )?;
        assert_eq!(routed.access_path, AccessPath::Routed, "{routed:?}");

        // Absent matches: live documents without a vector.
        let vectors = segment_reader.vector_index(fx.field)?;
        let mut absent = Vec::new();
        for doc in 0..max_doc {
            if alive.is_alive(doc) && !vectors.contains(doc)? {
                absent.push(doc);
            }
            if absent.len() == 3 {
                break;
            }
        }
        let mut mixed = absent.clone();
        mixed.extend(&live[..2]);
        mixed.sort_unstable();
        let (hits, stats) = run(&fx, &mixed, 10, located(1.0, false))?;
        assert_located(&stats);
        assert_eq!(stats.located_absent, 3, "{stats:?}");
        assert_eq!(hits, oracle(&fx, Some(&addresses(&mixed)), 10)?);

        let (hits, _) = run(&fx, &[], 10, located(1.0, false))?;
        assert!(hits.is_empty());
        let (hits, _) = run(&fx, &live, 0, located(1.0, false))?;
        assert!(hits.is_empty());
    }
    Ok(())
}

/// One threshold, two segments: the sparse segment goes located, the dense one routes, and
/// the merged result is the exhaustive top-k.
#[test]
fn mixed_index_merges_located_and_routed_segments() -> crate::Result<()> {
    for schedule in SCHEDULES {
        let fx = fixture_with(
            Shape::new(Metric::L2, schedule),
            &[|doc| doc % 100 == 3, |doc| doc % 2 == 0],
        )?;
        let searcher = fx.index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 2);
        let filter = TermQuery::new(
            Term::from_field_text(fx.label, "pick"),
            IndexRecordOption::Basic,
        );
        let collector = TopDocs::with_limit(10)
            .order_by_similarity(fx.field, query(fx.dim))
            .with_adaptive_params(located(0.05, true));
        let fruit = searcher.search(&filter, &collector)?;
        let admitted = admitted(&fx, &filter)?;
        assert_eq!(
            fruit.results,
            oracle(&fx, Some(&admitted), 10)?,
            "schedule={schedule:?}"
        );
        let paths: HashSet<_> = fruit.stats.iter().map(|s| s.access_path).collect();
        assert_eq!(
            paths,
            HashSet::from([AccessPath::Located, AccessPath::Routed]),
            "{:?}",
            fruit.stats
        );
    }
    Ok(())
}

/// The located path reads no DocIds column; its location-map reads land in the locate slot.
#[test]
fn located_path_reads_no_doc_ids() -> crate::Result<()> {
    for schedule in SCHEDULES {
        let shape = Shape {
            paged: true,
            ..Shape::new(Metric::L2, schedule)
        };
        let fx = fixture(shape)?;
        let max_doc = fx.index.reader()?.searcher().segment_readers()[0].max_doc();
        let docs = spread(max_doc, 40);
        let before = doc_ids_reads();
        let (_, routed) = run(&fx, &docs, 10, located(0.0, false))?;
        assert!(doc_ids_reads() > before, "the routed path reads DocIds");
        assert_eq!(routed.access_path, AccessPath::Routed);

        let searcher = fx.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let weight = FixedDocsWeight {
            max_doc,
            docs: docs.clone(),
        };
        let backend = backend(&fx, segment_reader, located(1.0, false))?;
        // The first run opens this reader's location map; measure the second.
        backend.top_n(&weight, segment_reader, 10)?;
        let directory = fx.directory.as_ref().expect("paged");
        directory.reads.lock().unwrap().clear();
        let before = doc_ids_reads();
        let (_, direct) = backend.top_n(&weight, segment_reader, 10)?;
        assert_eq!(doc_ids_reads(), before, "schedule={schedule:?}");
        assert_located(&direct);
        let reads = directory.reads.lock().unwrap();
        let locate_reads = reads
            .iter()
            .filter(|(stage, _)| *stage == Stage::Locate)
            .count();
        assert!(locate_reads > 0);
        // Centroid rows live in the `.centroids` file, which the paged directory does not log;
        // the locate slot counts them too.
        assert_eq!(
            direct.locate_io.reads,
            (locate_reads + direct.centroid_reads) as u64,
            "{direct:?}"
        );
    }
    Ok(())
}

/// The located path's centroid score is the router's key, bit for bit.
#[test]
fn centroid_score_equals_the_router_key() -> crate::Result<()> {
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        let fx = fixture(Shape::new(metric, &[1, 4]))?;
        let searcher = fx.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let backend = backend(&fx, segment_reader, exhaustive())?;
        let quantized = backend.quantized_query.as_ref().expect("quantized segment");
        let index = backend.reader.index().expect("IVF segment");
        let mut workspace = RouterWorkspace::default();
        let ranked: Vec<Candidate> = index
            .rank_clusters(
                &mut workspace,
                quantized.query(),
                RoutingParams {
                    k: index.num_clusters(),
                    recall: 1.0,
                },
            )
            .collect();
        assert_eq!(ranked.len(), index.num_clusters());
        for Candidate { sim, node } in ranked {
            let row = index.centroid_row(node as usize)?;
            let located = metric.similarity_bytes::<f32>(quantized.query(), &row);
            assert_eq!(
                sim.score().to_bits(),
                located.score().to_bits(),
                "{metric:?} cluster {node}"
            );
        }
    }
    Ok(())
}

/// Documents a query admits, across segments.
fn admitted(fx: &Fixture, filter: &dyn Query) -> crate::Result<HashSet<DocAddress>> {
    let searcher = fx.index.reader()?.searcher();
    let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
    let mut admitted = HashSet::new();
    for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
        weight.for_each_no_score(segment, &mut |docs| {
            admitted.extend(
                docs.iter()
                    .map(|&doc| DocAddress::new(segment_ord as u32, doc)),
            );
        })?;
    }
    Ok(admitted)
}

// ============================================================
// Auto located-path cap: the routed budget's row equivalent.
// ============================================================

/// A run under `direct_read` with a realistic probe budget.
fn budgeted(direct_read: DirectRead, work_model: Option<WorkModel>) -> AdaptiveProbeParams {
    AdaptiveProbeParams {
        max_probe_fraction: 0.05,
        min_probe_clusters: 1,
        work_model,
        direct_read,
        ..Default::default()
    }
}

/// The Auto cap is `floor(budget * n_avg / (1 - x))` of the work budget the routed path runs
/// against, primed or not; at the cap a filter is located, one match past it routes.
#[test]
fn auto_cap_is_the_budget_in_rows() -> crate::Result<()> {
    let fx = fixture(Shape::new(Metric::L2, &[1, 4]))?;
    let searcher = fx.index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    let max_doc = segment_reader.max_doc();
    let vectors = segment_reader.vector_index(fx.field)?;
    let index = vectors.index().expect("ivf");
    for work_model in [None, Some(WorkModel { n_avg: 61.5 })] {
        let params = budgeted(DirectRead::Auto, work_model);
        let (budget, n_avg, x) =
            params.resolved_work_budget(index.num_clusters(), index.num_docs())?;
        let cap = (budget * n_avg / (1.0 - x)).floor() as usize;
        assert!(cap > 0 && cap < max_doc as usize, "cap {cap}");
        let at_cap = spread(max_doc, cap as u32);
        let (hits, stats) = run(&fx, &at_cap, 10, params.clone())?;
        assert_eq!(stats.direct_cap, cap, "{work_model:?}");
        assert_eq!(stats.access_path, AccessPath::Located, "{work_model:?}");
        assert_eq!(hits, oracle(&fx, Some(&addresses(&at_cap)), 10)?);
        let past_cap = spread(max_doc, cap as u32 + 1);
        let (_, stats) = run(&fx, &past_cap, 10, params)?;
        assert_eq!(stats.direct_cap, cap, "{work_model:?}");
        assert_eq!(stats.access_path, AccessPath::Routed, "{work_model:?}");
    }
    Ok(())
}

/// Every mode reports its cap: zero when off, the selectivity cap otherwise; unfiltered queries
/// route whatever the cap.
#[test]
fn direct_cap_is_reported_in_every_mode() -> crate::Result<()> {
    let fx = fixture(Shape::new(Metric::L2, &[1, 4]))?;
    let max_doc = fx.index.reader()?.searcher().segment_readers()[0].max_doc();
    let docs = spread(max_doc, 10);
    let (_, off) = run(&fx, &docs, 10, budgeted(DirectRead::Off, None))?;
    assert_eq!((off.direct_cap, off.access_path), (0, AccessPath::Routed));
    let (_, fixed) = run(
        &fx,
        &docs,
        10,
        budgeted(DirectRead::MaxSelectivity(0.01), None),
    )?;
    assert_eq!(
        fixed.direct_cap,
        (f64::from(0.01f32) * f64::from(max_doc)).floor() as usize
    );
    assert_eq!(fixed.access_path, AccessPath::Located);
    let (_, unfiltered) = run_all(&fx, 10, budgeted(DirectRead::Auto, None))?;
    assert!(unfiltered.direct_cap > 0);
    assert_eq!(unfiltered.access_path, AccessPath::Routed);
    Ok(())
}
