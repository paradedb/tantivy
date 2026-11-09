//! Shared fixtures for vector search tests: one fixture builder and clusterer, one filter weight,
//! the brute-force oracle, run helpers, the read log and trace translation.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use common::BitSet;

use crate::collector::{SharedThresholdArcOpt, TopDocs};
use crate::index::IndexSettings;
use crate::indexer::NoMergePolicy;
use crate::query::{
    AllQuery, BitSetDocSet, ConstScorer, EnableScoring, Explanation, Query, Scorer, TermQuery,
    Weight,
};
use crate::schema::{
    Field, IndexRecordOption, Metric, Schema, Term, Value, VectorDType, VectorOptions, STORED,
    STRING,
};
use crate::vector::clustering::{
    for_each_batch, IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingVectors, VectorSource,
};
use crate::vector::params::ProbeParams;
use crate::vector::router::candidate::Candidate;
use crate::vector::router::RouterMetrics;
use crate::vector::search::input::PreparedVectorSearch;
use crate::vector::search::prepared::{QuantizedQueryCtx, VectorQuery};
use crate::vector::search::scan::bound::TopK;
use crate::vector::search::scan::plan::PlanOverride;
use crate::vector::search::scan::ScanTrace;
use crate::vector::search::segment::plan::ScanInput;
use crate::vector::search::SegmentSearcher;
use crate::vector::stats::SegmentReport;
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::io::test_support::{with_unarmed_log, PagedDirectory};
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::tests::TestVectorIndex;
use crate::vector::{
    BlockAddress, ProbeStats, RouterKind, TopClustersBySimilarityCollector,
    VectorQuantizationConfig, VectorQuantizationLayer, VectorScanInputs, VectorSearchBudget,
    VectorSegment, VectorSimilarityFruit, VectorStage,
};
use crate::{
    DocAddress, DocId, Index, IndexWriter, Score, Searcher, SegmentOrdinal, SegmentReader,
    TantivyDocument,
};

// --- Segment searcher ---

/// A segment and its source-test search parameters.
#[derive(Clone, Copy)]
pub(crate) struct SegmentRun<'a> {
    pub(crate) field: Field,
    pub(crate) ordinal: SegmentOrdinal,
    pub(crate) reader: &'a VectorIndexReader,
    pub(crate) params: &'a ProbeParams,
    pub(crate) routing: Option<&'a TestRouting>,
    pub(crate) index: &'a ClusterIndex,
    pub(crate) weight: &'a dyn Weight,
    pub(crate) segment: &'a SegmentReader,
    pub(crate) top_n: usize,
}

impl SegmentRun<'_> {
    pub(crate) fn input(&self) -> crate::Result<Option<ScanInput>> {
        let segments = [VectorSegment::from((self.ordinal, self.segment))];
        let budget = VectorSearchBudget::for_segments(self.field, self.params, &segments)?;
        let prepared = PreparedVectorSearch::prepare(
            self.field,
            self.top_n,
            self.params,
            self.weight,
            &segments,
            budget,
            |_| {},
        )?;
        Ok(prepared.segments.into_iter().next().and_then(|s| s.scan))
    }
}

impl<T: crate::vector::VectorElement> SegmentSearcher<T> {
    pub(crate) fn with_test_candidates(self, inputs: &VectorScanInputs) -> Self {
        let ordinal = self.segment_ord();
        self.with_candidates(
            inputs
                .candidates
                .iter()
                .filter(|c| c.node.0 == ordinal)
                .map(|c| Candidate {
                    sim: c.sim,
                    node: c.node.1,
                })
                .collect(),
            inputs.routing_stats.get(&ordinal).map(|s| s.metrics),
        )
    }
}

// --- Clusterer ---

/// Trains exactly the given centroids and assigns each vector to its L2-nearest one, first wins.
pub(crate) struct NearestClusterer {
    pub(crate) centroids: Vec<Vec<f32>>,
}

impl IvfClusterer for NearestClusterer {
    fn training_sample_ratio(&self) -> f32 {
        1.0
    }

    fn train(
        &self,
        options: &VectorOptions,
        _: &mut dyn VectorSource,
    ) -> crate::Result<IvfCentroids> {
        Ok(IvfCentroids::F32(IvfMatrix {
            values: self.centroids.iter().flatten().copied().collect(),
            rows: self.centroids.len(),
            dims: options.dim(),
        }))
    }

    fn assign(
        &self,
        options: &VectorOptions,
        source: &mut dyn VectorSource,
        _: &IvfCentroids,
    ) -> crate::Result<Vec<u32>> {
        let dim = options.dim();
        let num_rows = source.num_rows();
        let mut assignments = Vec::with_capacity(num_rows);
        for_each_batch(source, num_rows, |IvfTrainingVectors::F32(batch)| {
            for vector in batch.matrix.values.chunks_exact(dim) {
                assignments.push(nearest(vector, &self.centroids) as u32);
            }
        })?;
        Ok(assignments)
    }
}

/// The L2-nearest of `centroids`, first wins.
pub(crate) fn nearest(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    let distance = |centroid: &[f32]| -> f32 {
        vector
            .iter()
            .zip(centroid)
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    };
    let mut best = 0;
    for (index, centroid) in centroids.iter().enumerate() {
        if distance(centroid) < distance(&centroids[best]) {
            best = index;
        }
    }
    best
}

// --- Filter weight ---

/// A filter over a fixed set of segment-local doc ids.
pub(crate) struct DocsWeight {
    pub(crate) max_doc: DocId,
    pub(crate) docs: Vec<DocId>,
}

impl Weight for DocsWeight {
    fn scorer(&self, _: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let mut docs = BitSet::with_max_value(self.max_doc);
        for &doc in &self.docs {
            docs.insert(doc);
        }
        Ok(Box::new(ConstScorer::new(BitSetDocSet::from(docs), boost)))
    }

    fn explain(&self, _: &SegmentReader, _: DocId) -> crate::Result<Explanation> {
        unreachable!("the vector backend never explains filter docs")
    }
}

// --- Fixture builder ---

/// Where vectors live after the build.
#[derive(Clone, Debug)]
pub(crate) enum Layout {
    /// Unclustered flat segments.
    Flat,
    /// IVF segments clustered around these centroids.
    Ivf(Vec<Vec<f32>>),
}

/// A fixture to build: docs in ordinal order, each labelled `d{ordinal}` plus its extra labels.
#[derive(Clone, Debug)]
pub(crate) struct FixtureSpec {
    pub(crate) metric: Metric,
    pub(crate) dim: usize,
    pub(crate) vectors: Vec<Option<Vec<f32>>>,
    pub(crate) labels: Vec<Vec<String>>,
    pub(crate) layout: Layout,
    /// Quantization layer bits; empty for full precision.
    pub(crate) schedule: Vec<u8>,
    pub(crate) paged: bool,
    pub(crate) router: RouterKind,
    /// Segments after the build. IVF layouts write two flat segments per IVF segment and merge
    /// them pairwise.
    pub(crate) segments: usize,
    /// Doc ordinals deleted after the build.
    pub(crate) deletes: Vec<usize>,
}

impl FixtureSpec {
    /// One IVF segment over `vectors`, clustered around `centroids`, at full precision in RAM.
    pub(crate) fn ivf(metric: Metric, centroids: Vec<Vec<f32>>, vectors: Vec<Vec<f32>>) -> Self {
        let dim = centroids[0].len();
        Self {
            metric,
            dim,
            labels: vec![Vec::new(); vectors.len()],
            vectors: vectors.into_iter().map(Some).collect(),
            layout: Layout::Ivf(centroids),
            schedule: Vec::new(),
            paged: false,
            // Exact: small fixtures have few lists; Stacked's hard 25% top prune
            // cannot open them all, so oracle/exhaustive tests would be invalid.
            router: RouterKind::Exact,
            segments: 1,
            deletes: Vec::new(),
        }
    }

    /// One IVF segment of `docs` docs around `clusters` one-hot centroids in `dim` dimensions;
    /// Dot vectors also vary in norm.
    pub(crate) fn clustered(metric: Metric, dim: usize, clusters: usize, docs: usize) -> Self {
        let centroid = |cluster: usize| {
            let mut centroid = vec![0.0; dim];
            centroid[cluster] = 1.0;
            centroid
        };
        let vectors = (0..docs)
            .map(|ordinal| {
                let mut rng = Lcg(ordinal as u64 * 7919 + 13);
                let scale = if metric == Metric::Dot {
                    0.5 + (ordinal % 7) as f32 * 0.15
                } else {
                    1.0
                };
                centroid(ordinal % clusters)
                    .iter()
                    .zip(rng.vector(dim, 0.3))
                    .map(|(center, noise)| (center + noise) * scale)
                    .collect()
            })
            .collect();
        Self::ivf(metric, (0..clusters).map(centroid).collect(), vectors)
    }

    /// One flat segment over `vectors`.
    pub(crate) fn flat(metric: Metric, dim: usize, vectors: Vec<Option<Vec<f32>>>) -> Self {
        Self {
            metric,
            dim,
            labels: vec![Vec::new(); vectors.len()],
            vectors,
            layout: Layout::Flat,
            schedule: Vec::new(),
            paged: false,
            router: RouterKind::Exact,
            segments: 1,
            deletes: Vec::new(),
        }
    }

    pub(crate) fn schedule(mut self, schedule: &[u8]) -> Self {
        self.schedule = schedule.to_vec();
        self
    }

    pub(crate) fn paged(mut self) -> Self {
        self.paged = true;
        self
    }

    pub(crate) fn router(mut self, router: RouterKind) -> Self {
        self.router = router;
        self
    }

    pub(crate) fn segments(mut self, segments: usize) -> Self {
        self.segments = segments;
        self
    }

    pub(crate) fn deletes(mut self, deletes: &[usize]) -> Self {
        self.deletes = deletes.to_vec();
        self
    }

    /// Adds `label` to every doc `select` accepts.
    pub(crate) fn label(mut self, label: &str, select: impl Fn(usize) -> bool) -> Self {
        for (ordinal, labels) in self.labels.iter_mut().enumerate() {
            if select(ordinal) {
                labels.push(label.to_string());
            }
        }
        self
    }

    /// Drops the vectors of the docs `select` accepts; the docs stay.
    pub(crate) fn without_vectors(mut self, select: impl Fn(usize) -> bool) -> Self {
        for (ordinal, vector) in self.vectors.iter_mut().enumerate() {
            if select(ordinal) {
                *vector = None;
            }
        }
        self
    }

    fn quantization(&self) -> Vec<VectorQuantizationConfig> {
        if self.schedule.is_empty() {
            return Vec::new();
        }
        let seeds = [0x1111, 0x2222, 0x3333, 0x4444];
        vec![VectorQuantizationConfig::materialize(
            "embedding".to_string(),
            &VectorOptions::new(self.dim, self.metric),
            self.schedule
                .iter()
                .enumerate()
                .map(|(layer, &bits)| VectorQuantizationLayer {
                    bits,
                    seed: seeds[layer],
                })
                .collect(),
        )
        .unwrap()]
    }

    /// Builds the index.
    pub(crate) fn build(&self) -> crate::Result<Fixture> {
        let (index, field, label, directory) = self.create()?;
        self.write(&index, field, label)?;
        Ok(Fixture {
            index,
            field,
            label,
            metric: self.metric,
            directory,
        })
    }

    /// Creates the empty index with the layout's clusterer and router.
    fn create(&self) -> crate::Result<(Index, Field, Field, Option<PagedDirectory>)> {
        let mut schema = Schema::builder();
        let field = schema.add_vector_field(
            "embedding",
            VectorOptions::new(self.dim, self.metric).with_dtype(VectorDType::F32),
        );
        let label = schema.add_text_field("label", STRING | STORED);
        let ivf = matches!(self.layout, Layout::Ivf(_));
        let settings = IndexSettings {
            vector_clustering_threshold: if ivf { 1 } else { usize::MAX },
            vector_quantization: self.quantization(),
            ..IndexSettings::default()
        };
        let directory = self.paged.then(PagedDirectory::default);
        let mut builder = Index::builder().schema(schema.build()).settings(settings);
        if let Layout::Ivf(centroids) = &self.layout {
            builder = builder
                .ivf_clusterer(Arc::new(NearestClusterer {
                    centroids: centroids.clone(),
                }))
                .ivf_router(self.router);
        }
        let index = match &directory {
            Some(directory) => builder.create(directory.clone())?,
            None => builder.create_in_ram()?,
        };
        Ok((index, field, label, directory))
    }

    /// Writes the docs over the segment commits, merges IVF pairs, then applies the deletes.
    fn write(&self, index: &Index, field: Field, label: Field) -> crate::Result<()> {
        let ivf = matches!(self.layout, Layout::Ivf(_));
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let commits = if ivf {
            2 * self.segments
        } else {
            self.segments
        };
        let per_commit = self.vectors.len().div_ceil(commits);
        for (ordinal, vector) in self.vectors.iter().enumerate() {
            let mut doc = TantivyDocument::new();
            doc.add_text(label, format!("d{ordinal}"));
            for extra in &self.labels[ordinal] {
                doc.add_text(label, extra);
            }
            if let Some(vector) = vector {
                doc.add_vector(field, vector.as_slice());
            }
            writer.add_document(doc)?;
            if (ordinal + 1) % per_commit == 0 || ordinal + 1 == self.vectors.len() {
                writer.commit()?;
            }
        }
        if ivf {
            let mut segment_ids = index.searchable_segment_ids()?;
            segment_ids.sort();
            for pair in segment_ids.chunks(2) {
                writer.merge(pair).wait()?;
            }
        }
        for &ordinal in &self.deletes {
            writer.delete_term(Term::from_field_text(label, &format!("d{ordinal}")));
        }
        writer.commit()?;
        writer.wait_merging_threads()
    }
}

/// A built fixture.
pub(crate) struct Fixture {
    pub(crate) index: Index,
    pub(crate) field: Field,
    pub(crate) label: Field,
    pub(crate) metric: Metric,
    /// The read log, for paged fixtures.
    pub(crate) directory: Option<PagedDirectory>,
}

impl From<TestVectorIndex> for Fixture {
    fn from(index: TestVectorIndex) -> Self {
        Self {
            field: index.embedding_field(),
            label: index.label_field(),
            metric: index.vector_options().metric(),
            index: index.index,
            directory: None,
        }
    }
}

// --- Running ---

pub(crate) struct TestRouting {
    clusters: Vec<Candidate>,
    metrics: RouterMetrics,
}

impl TestRouting {
    pub(crate) fn ranked_clusters(&self) -> crate::vector::router::RouterIter {
        crate::vector::router::RouterIter::new(
            self.clusters
                .iter()
                .map(|c| Candidate {
                    sim: c.sim,
                    node: c.node.into(),
                })
                .collect(),
            self.metrics,
        )
    }
}

impl Fixture {
    /// A filter on one label.
    pub(crate) fn filter(&self, label: &str) -> TermQuery {
        TermQuery::new(
            Term::from_field_text(self.label, label),
            IndexRecordOption::Basic,
        )
    }

    /// Searches every segment through the collector.
    pub(crate) fn search(
        &self,
        filter: &dyn Query,
        query: &[f32],
        k: usize,
        params: ProbeParams,
    ) -> crate::Result<VectorSimilarityFruit> {
        self.search_with(filter, query, k, 0, params, None)
    }

    /// Searches every segment through the collector for the window `offset..offset + limit`,
    /// sharing every scan's threshold through `slot` when given.
    pub(crate) fn search_with(
        &self,
        filter: &dyn Query,
        query: &[f32],
        limit: usize,
        offset: usize,
        params: ProbeParams,
        slot: SharedThresholdArcOpt<Score>,
    ) -> crate::Result<VectorSimilarityFruit> {
        let mut builder = TopDocs::with_limit(limit)
            .and_offset(offset)
            .order_by_similarity(self.field, query.to_vec())
            .with_probe_params(params);
        if let Some(slot) = slot {
            builder = builder.with_shared_threshold(slot);
        }
        let searcher = self.index.reader()?.searcher();
        let collector = builder.build_for_searcher(&searcher, filter)?;
        searcher.search(filter, &collector)
    }

    /// Searches, probing `clusters` in that order instead of the router's ranking.
    pub(crate) fn search_clusters(
        &self,
        clusters: &[usize],
        filter: &dyn Query,
        query: &[f32],
        k: usize,
        params: ProbeParams,
    ) -> crate::Result<VectorSimilarityFruit> {
        let searcher = self.index.reader()?.searcher();
        let (ordinal, segment) = first_segment(&searcher);
        let segments = [VectorSegment::from((ordinal, segment))];
        let budget = VectorSearchBudget::for_segments(self.field, &params, &segments)?;
        let builder = TopDocs::with_limit(k)
            .order_by_similarity(self.field, query.to_vec())
            .with_probe_params(params);
        let weight = self.weight(filter)?;
        let prepared = builder.prepare(weight.as_ref(), &segments, budget, |_| {})?;
        let candidates: Vec<_> = self
            .candidates(clusters, query)?
            .into_iter()
            .map(|c| Candidate {
                sim: c.sim,
                node: (ordinal, c.node),
            })
            .collect();
        let routing_budget = budget.allocate_routing(
            &candidates,
            prepared.match_fraction(),
            prepared.located_budget(),
        )?;
        let collector = builder.build(prepared.with_candidates(candidates, &routing_budget)?);
        use crate::collector::Collector;
        collector.merge_fruits(vec![collector.collect_segment(
            weight.as_ref(),
            ordinal,
            segment,
        )?])
    }

    /// Router candidates for `clusters`, keyed by the exact centroid similarity.
    pub(crate) fn candidates(
        &self,
        clusters: &[usize],
        query: &[f32],
    ) -> crate::Result<Vec<Candidate>> {
        let searcher = self.index.reader()?.searcher();
        let vectors = first_segment(&searcher).1.vector_index(self.field)?;
        let ivf = vectors.index().expect("IVF segment");
        let centroids = ivf.centroid_bytes()?;
        let stride = vectors.options().bytes_per_vector();
        Ok(clusters
            .iter()
            .map(|&cluster| Candidate {
                node: cluster as u32,
                sim: self.metric.similarity_bytes::<f32>(
                    query,
                    &centroids[cluster * stride..(cluster + 1) * stride],
                ),
            })
            .collect())
    }

    fn scan_inputs(
        &self,
        segment: VectorSegment,
        weight: &dyn Weight,
        query: &[f32],
        top_n: usize,
        params: &ProbeParams,
        candidates: Option<Vec<Candidate>>,
    ) -> crate::Result<VectorScanInputs> {
        let builder =
            crate::vector::TopDocsByVectorSimilarityBuilder::new(self.field, query.to_vec(), top_n)
                .with_probe_params(params.clone());
        let budget =
            VectorSearchBudget::for_segments(self.field, params, std::slice::from_ref(&segment))?;
        let prepared = builder.prepare(weight, std::slice::from_ref(&segment), budget, |_| {})?;
        let router = TopClustersBySimilarityCollector::new(
            self.field,
            query.to_vec(),
            params.clone(),
            prepared.match_fraction(),
        )?
        .with_budget(prepared.budget());
        let candidates: Vec<_> = match candidates {
            Some(candidates) => candidates
                .into_iter()
                .map(|c| Candidate {
                    sim: c.sim,
                    node: (segment.ordinal, c.node),
                })
                .collect(),
            None => router.collect_segments(&prepared.routing_segments(), |_| {})?,
        };
        let routing_budget = budget.allocate_routing(
            &candidates,
            prepared.match_fraction(),
            prepared.located_budget(),
        )?;
        Ok(prepared
            .with_candidates(candidates, &routing_budget)?
            .with_routing_statistics(router.take_statistics()))
    }

    /// Runs one segment's search directly, with any `top_n`, under `weight`.
    pub(crate) fn run_segment(
        &self,
        segment: usize,
        weight: &dyn Weight,
        query: &[f32],
        top_n: usize,
        params: ProbeParams,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let searcher = self.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[segment];
        let inputs = self.scan_inputs(
            VectorSegment::from((segment as u32, segment_reader)),
            weight,
            query,
            top_n,
            &params,
            None,
        )?;
        let Some(prepared) = inputs.segments().first() else {
            let (_, cap, totals) = &inputs.prepared.scope[&(segment as u32)];
            return Ok((
                Vec::new(),
                ProbeStats::new(SegmentReport::Empty { located_cap: *cap }, totals.clone()),
            ));
        };
        let backend = SegmentSearcher::<f32>::for_segment(
            prepared.clone(),
            self.field,
            Arc::new(VectorQuery::new(self.metric, Arc::new(query.to_vec()))),
            None,
            params,
        )?
        .with_test_candidates(&inputs);
        backend.top_n(top_n)
    }

    /// Searches segment `segment` directly with the quantized query a collector prepares, every
    /// cluster read as `plan` forces where it can.
    pub(crate) fn run_planned(
        &self,
        segment: usize,
        weight: &dyn Weight,
        query: &[f32],
        top_n: usize,
        params: ProbeParams,
        plan: Option<PlanOverride>,
        admission_pruning: bool,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        self.run_planned_with(
            segment,
            weight,
            query,
            top_n,
            params,
            plan,
            admission_pruning,
            None,
            None,
        )
    }

    /// [`Fixture::run_planned`], seeding the scan from `collector_topk` when given.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_planned_with(
        &self,
        segment: usize,
        weight: &dyn Weight,
        query: &[f32],
        top_n: usize,
        params: ProbeParams,
        plan: Option<PlanOverride>,
        admission_pruning: bool,
        collector_topk: Option<Arc<Mutex<TopK>>>,
        routing: Option<Vec<Candidate>>,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let searcher = self.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[segment];
        let inputs = self.scan_inputs(
            VectorSegment::from((segment as u32, segment_reader)),
            weight,
            query,
            top_n,
            &params,
            routing,
        )?;
        let Some(prepared) = inputs.segments().first() else {
            let (_, cap, totals) = &inputs.prepared.scope[&(segment as u32)];
            return Ok((
                Vec::new(),
                ProbeStats::new(SegmentReport::Empty { located_cap: *cap }, totals.clone()),
            ));
        };
        let query = Arc::new(VectorQuery::new(self.metric, Arc::new(query.to_vec())));
        let quantized = segment_reader
            .vector_index(self.field)?
            .quantization()?
            .map(|field| {
                let index = field.index_ctx();
                Arc::new(QuantizedQueryCtx::with_depth(
                    Arc::clone(index),
                    &query,
                    index.specs.len(),
                ))
            });
        let mut backend = SegmentSearcher::<f32>::for_segment(
            prepared.clone(),
            self.field,
            query,
            quantized,
            params,
        )?
        .with_test_candidates(&inputs);
        backend.plan_override = plan;
        backend.admission_pruning_disabled = !admission_pruning;
        backend.collector_topk = collector_topk;
        backend.top_n(top_n)
    }

    /// Runs `run` over the first segment's prepared source for the supplied search arguments.
    pub(crate) fn with_segment<R>(
        &self,
        filter: &dyn Query,
        query: &[f32],
        top_n: usize,
        params: &ProbeParams,
        run: impl FnOnce(&SegmentRun<'_>) -> crate::Result<R>,
    ) -> crate::Result<R> {
        let searcher = self.index.reader()?.searcher();
        let (ordinal, segment) = first_segment(&searcher);
        let reader = segment.vector_index(self.field)?;
        let index = reader.index().expect("IVF segment");
        let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let inputs = self.scan_inputs(
            VectorSegment::from((ordinal, segment)),
            weight.as_ref(),
            query,
            top_n,
            params,
            None,
        )?;
        let routing = inputs.routing_stats.get(&ordinal).map(|stats| TestRouting {
            clusters: inputs
                .candidates
                .iter()
                .map(|c| Candidate {
                    sim: c.sim,
                    node: c.node.1,
                })
                .collect(),
            metrics: stats.metrics,
        });
        run(&SegmentRun {
            routing: routing.as_ref(),
            field: self.field,
            ordinal,
            reader: &reader,
            params,
            index,
            weight: weight.as_ref(),
            segment,
            top_n,
        })
    }

    /// `query`'s weight over the fixture's searcher.
    pub(crate) fn weight(&self, query: &dyn Query) -> crate::Result<Box<dyn Weight>> {
        let searcher = self.index.reader()?.searcher();
        query.weight(EnableScoring::disabled_from_searcher(&searcher))
    }

    /// The match-all weight.
    pub(crate) fn all(&self) -> crate::Result<Box<dyn Weight>> {
        let searcher = self.index.reader()?.searcher();
        AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))
    }

    /// The brute-force top `k` among alive docs `filter` matches.
    pub(crate) fn oracle(
        &self,
        filter: &dyn Query,
        query: &[f32],
        k: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let searcher = self.index.reader()?.searcher();
        let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let mut matching = HashSet::new();
        for (ordinal, segment) in searcher.segment_readers().iter().enumerate() {
            weight.for_each_no_score(segment, &mut |docs| {
                matching.extend(docs.iter().map(|&doc| DocAddress::new(ordinal as u32, doc)));
            })?;
        }
        let mut ranked = crate::vector::tests::ground_truth::top_k(
            &self.index,
            self.field,
            self.metric,
            query,
            usize::MAX,
        )?;
        ranked.retain(|(_, address)| matching.contains(address));
        ranked.truncate(k);
        Ok(ranked)
    }

    /// The ordinal a doc was written with.
    pub(crate) fn ordinal(&self, address: DocAddress) -> crate::Result<usize> {
        let doc = self
            .index
            .reader()?
            .searcher()
            .doc::<TantivyDocument>(address)?;
        let label = doc
            .get_all(self.label)
            .filter_map(|value| value.as_str().map(str::to_string))
            .find(|label| label.starts_with('d'))
            .expect("ordinal label");
        Ok(label[1..].parse().unwrap())
    }

    /// Doc ids of segment `segment`'s clusters, in cluster order.
    pub(crate) fn cluster_docs(&self, segment: usize) -> crate::Result<Vec<Vec<DocId>>> {
        let searcher = self.index.reader()?.searcher();
        let vectors = searcher.segment_readers()[segment].vector_index(self.field)?;
        let ivf = vectors.index().expect("IVF segment");
        with_unarmed_log(|| {
            (0..ivf.num_clusters())
                .map(|cluster| {
                    let mut docs = Vec::new();
                    vectors.read_doc_ids(cluster, &mut docs)?;
                    Ok(docs)
                })
                .collect()
        })
    }
}

// --- Read log ---

/// The requests a paged fixture's searches made, by stage.
pub(crate) type ReadLog = Vec<(VectorStage, Range<usize>)>;

impl Fixture {
    fn log(&self) -> &PagedDirectory {
        self.directory.as_ref().expect("paged fixture")
    }

    /// Forgets every logged request.
    pub(crate) fn clear_reads(&self) {
        self.log().reads.lock().unwrap().clear();
        self.log().centroid_reads.lock().unwrap().clear();
    }

    /// The requests logged since the last clear.
    pub(crate) fn reads(&self) -> ReadLog {
        self.log().reads.lock().unwrap().clone()
    }

    /// Requests to `.centroids`, kept separate because offsets are relative to that file.
    pub(crate) fn centroid_reads(&self) -> ReadLog {
        self.log().centroid_reads.lock().unwrap().clone()
    }

    /// Byte ranges of the first segment's DocIds column requests and full-precision cluster spans,
    /// by cluster; `None` for an empty cluster.
    pub(crate) fn column_ranges(&self) -> crate::Result<ColumnRanges> {
        let searcher = self.index.reader()?.searcher();
        let vectors = first_segment(&searcher).1.vector_index(self.field)?;
        let clusters = vectors.index().expect("IVF segment").num_clusters();
        let mut ranges = ColumnRanges::default();
        let mut docs = Vec::new();
        for cluster in 0..clusters {
            self.clear_reads();
            vectors.read_doc_ids(cluster, &mut docs)?;
            let mut reads: Vec<_> = self.reads().into_iter().map(|(_, range)| range).collect();
            // Block boundaries precede the DocIds payload, which is absent for empty clusters.
            ranges
                .docs
                .push((!docs.is_empty()).then(|| reads.pop().expect("DocIds read")));
            ranges.block_offsets.extend(reads);
            self.clear_reads();
            if !docs.is_empty() {
                vectors.read_cluster_rows(cluster)?;
            }
            ranges.rows.push(
                self.reads()
                    .into_iter()
                    .map(|(_, range)| range)
                    .find(|range| !ranges.block_offsets.contains(range)),
            );
        }
        self.clear_reads();
        Ok(ranges)
    }
}

/// Per-cluster byte ranges of the DocIds column and the full-precision rows.
#[derive(Debug, Default)]
pub(crate) struct ColumnRanges {
    pub(crate) block_offsets: Vec<Range<usize>>,
    pub(crate) docs: Vec<Option<Range<usize>>>,
    pub(crate) rows: Vec<Option<Range<usize>>>,
}

// --- Trace ---

/// A scan trace in doc ids: scored and boundary sets sorted, estimates by row.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct DocTrace {
    pub(crate) scored_docs: Vec<DocId>,
    pub(crate) boundary_docs: Vec<Vec<DocId>>,
    /// `(row, doc, estimate bits)` after admission and after each refining layer.
    pub(crate) estimates: Vec<Vec<(BlockAddress, DocId, u32)>>,
    pub(crate) exact_phase_docs: Vec<DocId>,
}

/// The first segment and its ordinal: the one with the lowest id, which the per-segment helpers
/// read. The searcher lists segments in the segment register's hash-map order, which differs
/// between processes; test segment ids count up, so the lowest id is the same segment every run.
pub(crate) fn first_segment(searcher: &Searcher) -> (SegmentOrdinal, &SegmentReader) {
    let (ordinal, segment) = searcher
        .segment_readers()
        .iter()
        .enumerate()
        .min_by_key(|(_, segment)| segment.segment_id())
        .expect("the index has a segment");
    (ordinal as SegmentOrdinal, segment)
}

/// Translates the first segment's trace rows of `field` in `index` into doc ids.
pub(crate) fn doc_trace(index: &Index, field: Field, trace: &ScanTrace) -> crate::Result<DocTrace> {
    let searcher = index.reader()?.searcher();
    let vectors = first_segment(&searcher).1.vector_index(field)?;
    let clusters = vectors.index().expect("IVF segment").num_clusters();
    let docs: Vec<Vec<DocId>> = with_unarmed_log(|| {
        (0..clusters)
            .map(|cluster| vectors.cluster_doc_ids(cluster).map(Option::unwrap))
            .collect::<crate::Result<_>>()
    })?;
    let sorted = |rows: &[BlockAddress]| {
        let mut out: Vec<DocId> = rows
            .iter()
            .map(|address| docs[address.block as usize][address.row as usize])
            .collect();
        out.sort_unstable();
        out
    };
    Ok(DocTrace {
        scored_docs: sorted(&trace.scored_rows),
        boundary_docs: trace
            .boundary_rows
            .iter()
            .map(|rows| sorted(rows))
            .collect(),
        estimates: trace
            .estimate_rows
            .iter()
            .map(|rows| {
                rows.iter()
                    .map(|&(row, bits)| (row, docs[row.block as usize][row.row as usize], bits))
                    .collect()
            })
            .collect(),
        exact_phase_docs: sorted(&trace.exact_phase_rows),
    })
}

impl Fixture {
    /// Translates the first segment's trace rows into doc ids.
    pub(crate) fn doc_trace(&self, trace: &ScanTrace) -> crate::Result<DocTrace> {
        doc_trace(&self.index, self.field, trace)
    }
}

/// Deterministic pseudo-random values in `[-1, 1)`.
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0
    }

    pub(crate) fn vector(&mut self, dim: usize, scale: f32) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32() * scale).collect()
    }
}
