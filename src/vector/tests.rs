#![allow(dead_code)]

use std::sync::Arc;

use crate::collector::{Count, TopDocs};
use crate::index::IndexSettings;
use crate::indexer::NoMergePolicy;
use crate::query::{AllQuery, TermQuery};
use crate::schema::{Field, FieldType, IndexRecordOption, Schema, Term, STORED, STRING};
use crate::vector::ivf::AdaptiveProbeParams;
use crate::vector::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfMergeSettings, IvfTrainingVectors, IvfVectors,
    Metric, VectorDType, VectorOptions,
};
use crate::{DocAddress, Index, Score, TantivyDocument};

const EMBEDDING_FIELD_NAME: &str = "embedding";
const LABEL_FIELD_NAME: &str = "label";
const NUM_DOCS: usize = 100;
const DOCS_PER_SEGMENT: usize = 10;

// Which on-disk layout the fixture should produce, reusing the public
// descriptor enum. Selected via the index settings (clustering threshold +
// clusterer); the resulting segment is self-describing through its `.vec`
// `IdMap`, so this is purely a build knob here.
pub(crate) use crate::vector::VectorStorageFormat;

pub(crate) struct TestVectorIndex {
    pub(crate) index: Index,
}

pub(crate) struct TestVectorIndexBuilder {
    centroids: Vec<[f32; grid2d::DIM]>,
    dtype: VectorDType,
    metric: Metric,
    selectivities: Vec<f32>,
    vector_storage_format: VectorStorageFormat,
}

impl TestVectorIndexBuilder {
    pub(crate) fn vector_storage_format(
        mut self,
        vector_storage_format: VectorStorageFormat,
    ) -> Self {
        self.vector_storage_format = vector_storage_format;
        self
    }

    pub(crate) fn selectivities(mut self, selectivities: &[f32]) -> Self {
        self.selectivities = selectivities.to_vec();
        self
    }

    pub(crate) fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    pub(crate) fn centroids(mut self, centroids: &[[f32; grid2d::DIM]]) -> Self {
        assert!(!centroids.is_empty(), "need at least one centroid");
        self.centroids = centroids.to_vec();
        self
    }

    pub(crate) fn build(self) -> crate::Result<TestVectorIndex> {
        let vector_options = VectorOptions::new(grid2d::DIM, self.metric).with_dtype(self.dtype);
        let mut schema_builder = Schema::builder();
        let embedding_field =
            schema_builder.add_vector_field(EMBEDDING_FIELD_NAME, vector_options.clone());
        let label_field = schema_builder.add_text_field(LABEL_FIELD_NAME, STRING | STORED);
        let schema = schema_builder.build();
        let index = self.create_index(schema)?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let doc_labels = labels::values(NUM_DOCS, &self.selectivities);

        for (doc_ord, embedding) in grid2d::vectors(NUM_DOCS).into_iter().enumerate() {
            let mut doc = TantivyDocument::new();
            doc.add_vector(embedding_field, &embedding);
            for label in &doc_labels[doc_ord] {
                doc.add_text(label_field, label);
            }
            writer.add_document(doc)?;
            if (doc_ord + 1) % DOCS_PER_SEGMENT == 0 {
                writer.commit()?;
            }
        }

        if self.vector_storage_format == VectorStorageFormat::Ivf {
            let mut segment_ids = index.searchable_segment_ids()?;
            segment_ids.sort();
            for pair in segment_ids.chunks_exact(2) {
                writer.merge(pair).wait()?;
            }
        }
        writer.wait_merging_threads()?;

        Ok(TestVectorIndex { index })
    }

    fn create_index(&self, schema: Schema) -> crate::Result<Index> {
        let mut settings = IndexSettings::default();
        if self.vector_storage_format == VectorStorageFormat::Ivf {
            settings.vector_clustering_threshold = 1;
        }
        let mut builder = Index::builder().schema(schema).settings(settings);
        if self.vector_storage_format == VectorStorageFormat::Ivf {
            builder = builder.ivf_clusterer(Arc::new(Grid2DClusterer {
                centroids: self.centroids.clone(),
            }));
        }
        builder.create_in_ram()
    }
}

impl TestVectorIndex {
    pub(crate) fn builder(dtype: VectorDType) -> TestVectorIndexBuilder {
        TestVectorIndexBuilder {
            centroids: grid2d::centroids(),
            dtype,
            metric: Metric::L2,
            selectivities: Vec::new(),
            vector_storage_format: VectorStorageFormat::Flat,
        }
    }

    pub(crate) fn embedding_field(&self) -> Field {
        self.index.schema().get_field(EMBEDDING_FIELD_NAME).unwrap()
    }

    pub(crate) fn label_field(&self) -> Field {
        self.index.schema().get_field(LABEL_FIELD_NAME).unwrap()
    }

    pub(crate) fn vector_options(&self) -> VectorOptions {
        let schema = self.index.schema();
        let field_entry = schema.get_field_entry(self.embedding_field());
        match field_entry.field_type() {
            FieldType::Vector(options) => options.clone(),
            _ => unreachable!("embedding field must be a vector"),
        }
    }

    pub(crate) fn dtype(&self) -> VectorDType {
        self.vector_options().dtype()
    }

    pub(crate) fn embedding(&self, doc_ord: usize) -> [f32; grid2d::DIM] {
        grid2d::vectors(self.ndocs())
            .get(doc_ord)
            .copied()
            .expect("fixture doc")
    }

    pub(crate) fn ndocs(&self) -> usize {
        self.index.reader().expect("reader").searcher().num_docs() as usize
    }

    pub(crate) fn ground_truth(
        &self,
        query: [f32; grid2d::DIM],
        top_k: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        ground_truth::top_k(
            &self.index,
            self.embedding_field(),
            self.vector_options().metric(),
            &query,
            top_k,
        )
    }
}

pub(crate) struct Grid2DClusterer {
    pub(crate) centroids: Vec<[f32; grid2d::DIM]>,
}

impl IvfClusterer for Grid2DClusterer {
    fn training_sample_ratio(&self) -> f32 {
        1.0
    }

    fn merge_settings(&self, _total_target_docs: usize) -> crate::Result<IvfMergeSettings> {
        Ok(IvfMergeSettings {
            training_sample_ratio: self.training_sample_ratio(),
            assign_batch_size: self.assign_batch_size(),
        })
    }

    fn train(
        &self,
        options: &VectorOptions,
        _vectors: IvfTrainingVectors,
    ) -> crate::Result<IvfCentroids> {
        assert_eq!(options.dim(), grid2d::DIM);
        let num_centroids = self.centroids.len();
        Ok(IvfCentroids::F32(IvfMatrix {
            values: self
                .centroids
                .iter()
                .flat_map(|centroid| centroid.iter().copied())
                .collect(),
            rows: num_centroids,
            dims: grid2d::DIM,
        }))
    }

    fn assign(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<u32>> {
        assert_eq!(options.dim(), grid2d::DIM);
        let IvfVectors::F32(vectors) = vectors;
        let IvfCentroids::F32(centroids) = centroids;
        Ok(vectors
            .matrix
            .values
            .chunks_exact(vectors.matrix.dims)
            .map(|vector| grid2d::nearest_centroid(vector, centroids.values.as_slice()) as u32)
            .collect())
    }
}

#[test]
fn fixture_builds_expected_schema_docs_and_labels() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::Cosine)
        .selectivities(&[0.1, 0.5])
        .build()?;

    assert_eq!(index.ndocs(), NUM_DOCS);
    assert_eq!(index.dtype(), VectorDType::F32);
    let vector_options = index.vector_options();
    assert_eq!(vector_options.dim(), grid2d::DIM);
    assert_eq!(vector_options.dtype(), VectorDType::F32);
    assert_eq!(vector_options.metric(), Metric::Cosine);
    assert!(matches!(
        index
            .index
            .schema()
            .get_field_entry(index.label_field())
            .field_type(),
        FieldType::Str(_)
    ));
    let searcher = index.index.reader()?.searcher();
    for (selectivity, expected_count) in [(0.1, 10), (0.5, 50)] {
        let label = labels::LabelWithSelectivity::new(selectivity).label();
        let term = Term::from_field_text(index.label_field(), &label);
        assert_eq!(
            searcher.search(&TermQuery::new(term, IndexRecordOption::Basic), &Count)?,
            expected_count
        );
    }

    Ok(())
}

#[test]
fn fixture_uses_selected_storage_format() -> crate::Result<()> {
    for vector_storage_format in [VectorStorageFormat::Flat, VectorStorageFormat::Ivf] {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(vector_storage_format)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        let vec_reader = searcher.segment_readers()[0].vector_index(index.embedding_field())?;
        let is_ivf = vec_reader.index().is_some();
        assert_eq!(
            is_ivf,
            vector_storage_format == VectorStorageFormat::Ivf,
            "storage format mismatch: index present = {is_ivf}, expected {vector_storage_format:?}"
        );
    }

    Ok(())
}

/// Both vector segment files must stamp the current format-generation header
/// ahead of their composite body, so future layout changes can be gated.
#[test]
fn vector_files_stamp_format_version_header() -> crate::Result<()> {
    use crate::directory::CompositeFile;
    use crate::index::SegmentComponent;
    use crate::vector::header::{read_header, VectorFileVersion};
    use crate::vector::ivf::CENTROIDS_EXT;
    use crate::vector::VEC_EXT;

    for format in [VectorStorageFormat::Flat, VectorStorageFormat::Ivf] {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(format)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        assert!(!searcher.segment_readers().is_empty());

        for segment_reader in searcher.segment_readers() {
            let vec_file =
                segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string()))?;
            let (version, body) = read_header(&vec_file)?;
            assert_eq!(version, VectorFileVersion::V3);
            // Body must be a valid composite — proves the stamp sits in front
            // of the framing, not inside a slot.
            CompositeFile::open(&body)?;

            match format {
                VectorStorageFormat::Flat => {
                    assert!(
                        segment_reader
                            .open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string()))
                            .is_err(),
                        "flat segments must not write `.centroids`"
                    );
                }
                VectorStorageFormat::Ivf => {
                    let centroids_file = segment_reader
                        .open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string()))?;
                    let (version, body) = read_header(&centroids_file)?;
                    assert_eq!(version, VectorFileVersion::V3);
                    CompositeFile::open(&body)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn fixture_vectors_round_trip_from_readers() -> crate::Result<()> {
    let mut expected = grid2d::vectors(NUM_DOCS);
    expected.sort_by(|left, right| {
        left[0]
            .partial_cmp(&right[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left[1]
                    .partial_cmp(&right[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    for vector_storage_format in [VectorStorageFormat::Flat, VectorStorageFormat::Ivf] {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(vector_storage_format)
            .build()?;
        let searcher = index.index.reader()?.searcher();
        let mut got = Vec::new();
        for segment_reader in searcher.segment_readers() {
            let vec_reader = segment_reader.vector_index(index.embedding_field())?;
            for doc in 0..segment_reader.max_doc() {
                if let Some(bytes) = vec_reader.vector_bytes(doc)? {
                    let vector: [f32; grid2d::DIM] = bytes
                        .chunks_exact(VectorDType::F32.size_bytes())
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 bytes")))
                        .collect::<Vec<_>>()
                        .try_into()
                        .expect("2D vector");
                    got.push(vector);
                }
            }
        }
        got.sort_by(|left: &[f32; grid2d::DIM], right| {
            left[0]
                .partial_cmp(&right[0])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left[1]
                        .partial_cmp(&right[1])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        assert_eq!(got, expected);
    }

    Ok(())
}

#[test]
fn ivf_fixture_uses_custom_centroids_for_assignment() -> crate::Result<()> {
    let centroids = vec![[0.0, 0.0], [6.0, 6.0]];
    let index = TestVectorIndex::builder(VectorDType::F32)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .centroids(&centroids)
        .build()?;
    let centroid_values: Vec<f32> = centroids
        .iter()
        .flat_map(|vector| vector.iter().copied())
        .collect();
    let searcher = index.index.reader()?.searcher();
    let mut assigned_docs = 0;

    for segment_reader in searcher.segment_readers() {
        let vec_reader = segment_reader.vector_index(index.embedding_field())?;
        let ivf = vec_reader.index().expect("expected IVF storage");
        assert_eq!(
            ivf.centroid_bytes()?
                .chunks_exact(VectorDType::F32.size_bytes())
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 bytes")))
                .collect::<Vec<_>>(),
            centroid_values
        );
        for cluster_ord in 0..centroids.len() {
            let doc_ids = vec_reader
                .cluster_doc_ids(cluster_ord)
                .expect("in-bounds cluster");
            for doc in doc_ids {
                let vector: Vec<f32> = vec_reader
                    .vector_bytes(doc)?
                    .expect("vector bytes")
                    .chunks_exact(VectorDType::F32.size_bytes())
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 bytes")))
                    .collect();
                assert_eq!(
                    grid2d::nearest_centroid(&vector, &centroid_values),
                    cluster_ord
                );
                assigned_docs += 1;
            }
        }
    }

    assert_eq!(assigned_docs, NUM_DOCS);
    Ok(())
}

/// Regression for `FlatBackend::top_n` under truncation. A bare
/// `TopNComputer::new` defaults to `ReverseComparator`, which keeps
/// the K *smallest* sort_keys — for our "higher = closer" similarity
/// convention that returned the K *farthest* docs once a segment had
/// more than K matches. Latent before the fix because every previous
/// flat test had ≤ K docs per segment, so the truncate_top_n path
/// never fired. The backend now wires `NaturalComparator` explicitly;
/// this test would fail under the old code.
///
/// The shared fixture commits every `DOCS_PER_SEGMENT = 10` docs, so
/// each segment has 10 > K = 3 docs — the truncation path is on.
#[test]
fn flat_top_n_returns_nearest_when_more_than_k_docs_per_segment() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .vector_storage_format(VectorStorageFormat::Flat)
        .build()?;
    let query = grid2d::centroids()[0];
    let top_k = 3;
    let expected = index.ground_truth(query, top_k)?;
    let hits = index
        .index
        .reader()?
        .searcher()
        .search(
            &AllQuery,
            &TopDocs::with_limit(top_k)
                .order_by_similarity(index.embedding_field(), query.to_vec()),
        )?
        .results;
    assert_eq!(hits, expected);
    Ok(())
}

#[test]
fn ivf_merge_writes_centroid_graph_slot() -> crate::Result<()> {
    use crate::directory::CompositeFile;
    use crate::index::SegmentComponent;
    use crate::vector::ivf::graph::EMPTY;
    use crate::vector::ivf::CENTROIDS_EXT;
    use crate::vector::NeighborhoodGraphConfig;

    let centroids = vec![[0.0, 0.0], [6.0, 6.0]];
    let index = TestVectorIndex::builder(VectorDType::F32)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .centroids(&centroids)
        .build()?;
    let searcher = index.index.reader()?.searcher();
    assert!(!searcher.segment_readers().is_empty());

    for segment_reader in searcher.segment_readers() {
        let centroids_file =
            segment_reader.open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string()))?;
        let (_version, body) = super::header::read_header(&centroids_file)?;
        let composite = CompositeFile::open(&body)?;
        let graph_bytes = composite
            .open_read_with_idx(index.embedding_field(), 2)
            .expect("IVF merge should write the centroid graph slot")
            .read_bytes()?;
        assert_eq!(graph_bytes[0], 0, "V3 router kind byte (Graph)");
        let graph_payload = &graph_bytes[1..];

        let words: Vec<u32> = graph_payload
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("u32 word")))
            .collect();
        assert_eq!(words.len() * 4, graph_payload.len(), "whole number of u32s");
        let max_edges = words[0] as usize;
        assert_eq!(max_edges, NeighborhoodGraphConfig::default().max_edges);
        let adjacency = &words[1..];
        assert_eq!(adjacency.len(), centroids.len() * max_edges);
        // Two distinct centroids prune to each other's single neighbor; the
        // rest of each run is EMPTY padding.
        assert_eq!(adjacency[0], 1);
        assert!(adjacency[1..max_edges].iter().all(|&id| id == EMPTY));
        assert_eq!(adjacency[max_edges], 0);
        assert!(adjacency[max_edges + 1..].iter().all(|&id| id == EMPTY));
    }
    Ok(())
}

#[test]
fn ground_truth_orders_by_metric() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .build()?;
    let query = grid2d::centroids()[0];
    let hits = index.ground_truth(query, 5)?;
    let mut expected_scores: Vec<Score> = grid2d::vectors(NUM_DOCS)
        .iter()
        .map(|vector| Metric::L2.similarity(&query, vector).score())
        .collect();
    expected_scores
        .sort_by(|left, right| right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal));

    assert_eq!(hits.len(), 5);
    for (got, expected) in hits.iter().map(|(score, _)| *score).zip(expected_scores) {
        assert!((got - expected).abs() < 1e-6);
    }

    Ok(())
}

/// Non-finite elements are rejected at ingest on normalizing fields
/// (Cosine+F32) and accepted on non-normalizing ones (L2) — validation
/// rides the normalize path only. `IndexWriter::add_document` enqueues
/// to a worker, so the rejection may surface either from the enqueue or
/// from the following `commit`.
#[test]
fn ingest_rejects_non_finite_cosine_vector() -> crate::Result<()> {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // L2: same vector is accepted; nothing normalizes, data is stored raw.
        let mut schema_builder = Schema::builder();
        let l2_field = schema_builder.add_vector_field("l2", VectorOptions::new(2, Metric::L2));
        let schema = schema_builder.build();
        let index = Index::builder().schema(schema).create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        let mut doc = TantivyDocument::new();
        doc.add_vector(l2_field, &[bad, 1.0]);
        writer.add_document(doc)?;
        writer.commit()?;

        // Cosine: rejected.
        let mut schema_builder = Schema::builder();
        let cos_field =
            schema_builder.add_vector_field("cos", VectorOptions::new(2, Metric::Cosine));
        let schema = schema_builder.build();
        let index = Index::builder().schema(schema).create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        let mut doc = TantivyDocument::new();
        doc.add_vector(cos_field, &[bad, 1.0]);
        let err = match writer.add_document(doc) {
            Err(err) => err.to_string(),
            Ok(_) => writer
                .commit()
                .expect_err("non-finite vector must fail ingest")
                .to_string(),
        };
        assert!(err.contains("non-finite"), "bad={bad}, err={err}");
    }
    Ok(())
}

/// A zero vector is honest data: ingest accepts it (`ZeroSkipped`), and
/// at query time it scores exactly 0.0 — behind any non-zero doc.
#[test]
fn ingest_accepts_zero_vector() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let embedding_field =
        schema_builder.add_vector_field("embedding", VectorOptions::new(2, Metric::Cosine));
    let schema = schema_builder.build();
    let index = Index::builder().schema(schema).create_in_ram()?;
    let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
    let mut zero_doc = TantivyDocument::new();
    zero_doc.add_vector(embedding_field, &[0.0_f32, 0.0]);
    writer.add_document(zero_doc)?;
    let mut unit_doc = TantivyDocument::new();
    unit_doc.add_vector(embedding_field, &[0.6_f32, 0.8]);
    writer.add_document(unit_doc)?;
    writer.commit()?;

    let searcher = index.reader()?.searcher();
    let collector = TopDocs::with_limit(2).order_by_similarity(embedding_field, vec![1.0_f32, 0.0]);
    let hits = searcher.search(&AllQuery, &collector)?.results;
    assert_eq!(hits.len(), 2, "zero vector must be ingested and returned");
    assert!(hits[0].0 > 0.0, "non-zero doc must rank first: {hits:?}");
    assert_eq!(hits[1].0, 0.0, "zero vector scores 0.0: {hits:?}");
    Ok(())
}

/// "Scan everything" probe params: the full-capacity ceiling, so the
/// budget never binds before the stream is exhausted. Used by
/// oracle-equality tests, where every cluster the bounds gate cannot
/// PROVE useless must be probed - provable skips never change the
/// top-K, so oracle equality still holds under them.
pub(crate) fn exhaustive_params(_num_centroids: usize) -> AdaptiveProbeParams {
    AdaptiveProbeParams {
        max_probe_fraction: 1.0,
        min_probe_clusters: 1,
        ..Default::default()
    }
}

pub(crate) mod ground_truth {
    use std::cmp::Ordering;
    use std::sync::Arc;

    use crate::schema::Field;
    use crate::vector::{Metric, PreparedQuery};
    use crate::{DocAddress, Index, Score};

    pub(crate) fn top_k(
        index: &Index,
        vec_field: Field,
        metric: Metric,
        query: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let query = PreparedQuery::<f32>::new(metric, Arc::new(query.to_vec()));
        let searcher = index.reader()?.searcher();
        let mut scored = Vec::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            let vec_reader = segment_reader.vector_index(vec_field)?;
            let alive = segment_reader.alive_bitset();
            for doc in 0..segment_reader.max_doc() {
                if let Some(alive) = alive {
                    if !alive.is_alive(doc) {
                        continue;
                    }
                }
                if let Some(bytes) = vec_reader.vector_bytes(doc)? {
                    scored.push((
                        query.score_doc_bytes(&bytes),
                        DocAddress::new(seg_ord as u32, doc),
                    ));
                }
            }
        }
        scored.sort_by(|a: &(Score, DocAddress), b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.segment_ord.cmp(&b.1.segment_ord))
                .then(a.1.doc_id.cmp(&b.1.doc_id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

// Generates mock string labels with controlled selectivity for filter tests.
mod labels {
    use std::fmt;

    pub(crate) fn values(ndocs: usize, selectivities: &[f32]) -> Vec<Vec<String>> {
        let mut labels = vec![Vec::new(); ndocs];
        for selectivity in selectivities.iter().copied().map(LabelWithSelectivity::new) {
            let label = selectivity.label();
            let doc_count = selectivity.doc_count(ndocs);
            for doc_labels in labels.iter_mut().take(doc_count) {
                doc_labels.push(label.clone());
            }
        }
        labels
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct LabelWithSelectivity(f32);

    impl LabelWithSelectivity {
        pub(crate) fn new(value: f32) -> Self {
            assert!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "selectivity must be in [0, 1]"
            );
            Self(value)
        }

        pub(crate) fn label(self) -> String {
            format!("selectivity_{self}")
        }

        pub(crate) fn doc_count(self, total_docs: usize) -> usize {
            ((total_docs as f64) * f64::from(self.0)).round() as usize
        }
    }

    impl fmt::Display for LabelWithSelectivity {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut formatted = format!("{:.6}", self.0);
            while formatted.contains('.') && formatted.ends_with('0') {
                formatted.pop();
            }
            if formatted.ends_with('.') {
                formatted.pop();
            }
            f.write_str(&formatted)
        }
    }
}

// Generates deterministic mock 2D embeddings scattered around a centroid grid.
mod grid2d {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    pub(crate) const DIM: usize = 2;

    const CLUSTER_RADIUS: f32 = 0.5;
    const GRID_GAP: f32 = 3.0;
    const GRID_ROWS: usize = 3;
    const GRID_COLS: usize = 3;
    const SEED: u64 = 21;

    pub(crate) fn vectors(ndocs: usize) -> Vec<[f32; DIM]> {
        if ndocs == 0 {
            return Vec::new();
        }
        let centroids = centroids();
        let fixture = Fixture2D {
            points_per_cluster: points_per_cluster(ndocs, centroids.len()),
            cluster_radius: CLUSTER_RADIUS,
            seed: SEED,
        };
        fixture
            .points(&centroids)
            .into_iter()
            .take(ndocs)
            .map(|point| point.vector)
            .collect()
    }

    pub(crate) fn centroids() -> Vec<[f32; DIM]> {
        grid([0.0, 0.0], GRID_ROWS, GRID_COLS, GRID_GAP)
    }

    pub(crate) fn nearest_centroid(vector: &[f32], centroids: &[f32]) -> usize {
        assert_eq!(vector.len(), DIM);
        let mut best = 0;
        let mut best_d2 = f32::INFINITY;
        for (ord, centroid) in centroids.chunks_exact(DIM).enumerate() {
            let dx = vector[0] - centroid[0];
            let dy = vector[1] - centroid[1];
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best = ord;
                best_d2 = d2;
            }
        }
        best
    }

    #[derive(Clone, Copy, Debug)]
    struct Fixture2D {
        points_per_cluster: usize,
        cluster_radius: f32,
        seed: u64,
    }

    #[derive(Clone, Copy, Debug)]
    struct Point {
        vector: [f32; DIM],
        cluster_ord: usize,
    }

    impl Fixture2D {
        fn points(&self, centroids: &[[f32; DIM]]) -> Vec<Point> {
            assert!(!centroids.is_empty(), "need at least one centroid");
            assert!(self.points_per_cluster >= 1);
            assert_non_overlapping(centroids, self.cluster_radius);

            let mut rng = StdRng::seed_from_u64(self.seed);
            let mut points = Vec::with_capacity(centroids.len() * self.points_per_cluster);
            for (cluster_ord, centroid) in centroids.iter().enumerate() {
                for _ in 0..self.points_per_cluster {
                    points.push(Point {
                        vector: sample_disk(centroid, self.cluster_radius, &mut rng),
                        cluster_ord,
                    });
                }
            }
            points
        }
    }

    fn points_per_cluster(ndocs: usize, num_clusters: usize) -> usize {
        (ndocs + num_clusters - 1) / num_clusters
    }

    fn grid(origin: [f32; DIM], rows: usize, cols: usize, gap: f32) -> Vec<[f32; DIM]> {
        assert!(rows >= 1 && cols >= 1);
        let mut out = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                out.push([
                    origin[0] + (col as f32) * gap,
                    origin[1] + (row as f32) * gap,
                ]);
            }
        }
        out
    }

    fn sample_disk(center: &[f32; DIM], radius: f32, rng: &mut StdRng) -> [f32; DIM] {
        let u: f32 = rng.random_range(0.0..1.0);
        let v: f32 = rng.random_range(0.0..1.0);
        let r = radius * u.sqrt();
        let theta = 2.0 * std::f32::consts::PI * v;
        [center[0] + r * theta.cos(), center[1] + r * theta.sin()]
    }

    fn assert_non_overlapping(centroids: &[[f32; DIM]], radius: f32) {
        let min_dist = 2.0 * radius;
        for left in 0..centroids.len() {
            for right in (left + 1)..centroids.len() {
                let d = dist(&centroids[left], &centroids[right]);
                assert!(d >= min_dist);
            }
        }
    }

    fn dist(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        (dx * dx + dy * dy).sqrt()
    }
}

// ======================================================================
// P1: bounds storage
// ======================================================================

/// C1 fixture tests: the `.centroids` bounds slot end to end — write
/// fold, roundtrip, merge recomputation, and the pre-V2 refusal. The
/// pure builder/kind cases live in `vector::bounds::bounds_storage_tests`.
mod bounds_storage_tests {
    use std::io::Write;
    use std::sync::Arc;

    use common::HasLen;

    use super::{TestVectorIndex, EMBEDDING_FIELD_NAME};
    use crate::collector::TopDocs;
    use crate::directory::{CompositeFile, CompositeWrite, Directory, RamDirectory};
    use crate::index::{IndexSettings, SegmentComponent};
    use crate::indexer::NoMergePolicy;
    use crate::query::AllQuery;
    use crate::schema::{Schema, STORED, STRING};
    use crate::vector::ivf::{IvfIndex, CENTROIDS_EXT};
    use crate::vector::{
        residual_norm, BoundKind, BuiltRouter, IvfCentroids, IvfClusterer, IvfConfig,
        IvfIndexBuilder, IvfMatrix, IvfMergeSettings, IvfTrainingVectors, IvfVectors, Metric,
        StackedIvfIndex, SuperKMeansLevelClusterer, VectorDType, VectorOptions,
        VectorStorageFormat,
    };
    use crate::{Index, IndexWriter, TantivyDocument};

    /// Recompute one segment's expected fold from its stored artifacts:
    /// per cluster, max [`residual_norm`] over the cluster's rows against
    /// the stored centroid.
    fn fresh_fold(
        vec_reader: &crate::vector::VectorIndexReader,
        ivf: &IvfIndex,
        dim: usize,
        metric: Metric,
    ) -> crate::Result<Vec<f32>> {
        let centroid_bytes = ivf.centroid_bytes()?;
        let stride = dim * std::mem::size_of::<f32>();
        let mut expected = vec![0.0f32; ivf.num_clusters()];
        for cluster in 0..ivf.num_clusters() {
            let centroid: Vec<f32> = centroid_bytes[cluster * stride..(cluster + 1) * stride]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            // The writer's degenerate-centroid rule: non-finite, or
            // zero-norm under cosine renormalization, saturates.
            let zero_norm = centroid.iter().all(|&value| value == 0.0);
            if centroid.iter().any(|value| !value.is_finite())
                || (metric == Metric::Cosine && zero_norm)
            {
                expected[cluster] = f32::INFINITY;
                continue;
            }
            for row in ivf.cluster_range(cluster) {
                let row_bytes = vec_reader.vector_bytes_for_row(row)?;
                let residual = residual_norm::<f32>(&row_bytes, &centroid);
                if !residual.is_finite() {
                    expected[cluster] = f32::INFINITY;
                } else if residual > expected[cluster] {
                    expected[cluster] = residual;
                }
            }
        }
        Ok(expected)
    }

    /// Build, write, reopen: the stored bounds are bit-equal to a fresh
    /// fold over the stored rows and centroids — for every metric.
    #[test]
    fn roundtrip_per_metric() -> crate::Result<()> {
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let fixture = TestVectorIndex::builder(VectorDType::F32)
                .metric(metric)
                .vector_storage_format(VectorStorageFormat::Ivf)
                .build()?;
            let field = fixture.embedding_field();
            let searcher = fixture.index.reader()?.searcher();
            let mut ivf_segments = 0usize;
            for segment_reader in searcher.segment_readers() {
                let vec_reader = segment_reader.vector_index(field)?;
                let Some(ivf) = vec_reader.index() else {
                    continue;
                };
                ivf_segments += 1;
                let bounds = ivf.bounds();
                assert_eq!(bounds.kind(), BoundKind::Ball);
                let expected =
                    fresh_fold(&vec_reader, ivf, fixture.vector_options().dim(), metric)?;
                assert_eq!(bounds.values().len(), expected.len());
                for (cluster, (&stored, &fold)) in
                    bounds.values().iter().zip(expected.iter()).enumerate()
                {
                    assert_eq!(
                        stored.to_bits(),
                        fold.to_bits(),
                        "{metric:?} cluster {cluster}: stored {stored} != fold {fold}"
                    );
                }
            }
            assert!(ivf_segments > 0, "{metric:?}: fixture built no IVF segment");
        }
        Ok(())
    }

    /// A clusterer with deterministic centroids for crafted-geometry
    /// builds: fixed rows when supplied, else the first
    /// `num_centroids` training samples — data-dependent, so a merge of
    /// merged segments re-trains onto different centroids.
    struct TestClusterer {
        fixed_centroids: Option<Vec<[f32; 2]>>,
        num_centroids: usize,
        /// Build a stacked router over the trained centroids (slot `[2]`).
        stacked: bool,
    }

    impl IvfClusterer for TestClusterer {
        fn training_sample_ratio(&self) -> f32 {
            1.0
        }
        fn merge_settings(&self, _total_target_docs: usize) -> crate::Result<IvfMergeSettings> {
            Ok(IvfMergeSettings {
                training_sample_ratio: self.training_sample_ratio(),
                assign_batch_size: self.assign_batch_size(),
            })
        }
        fn train(
            &self,
            options: &VectorOptions,
            vectors: IvfTrainingVectors,
        ) -> crate::Result<IvfCentroids> {
            assert_eq!(options.dim(), 2);
            let num_centroids = self.num_centroids;
            let values = match &self.fixed_centroids {
                Some(centroids) => centroids
                    .iter()
                    .take(num_centroids)
                    .flat_map(|centroid| centroid.iter().copied())
                    .collect(),
                None => {
                    let IvfTrainingVectors::F32(batch) = vectors;
                    batch.matrix.values[..num_centroids * 2].to_vec()
                }
            };
            Ok(IvfCentroids::F32(IvfMatrix {
                values,
                rows: num_centroids,
                dims: 2,
            }))
        }
        fn assign(
            &self,
            options: &VectorOptions,
            vectors: IvfVectors<'_>,
            centroids: &IvfCentroids,
        ) -> crate::Result<Vec<u32>> {
            assert_eq!(options.dim(), 2);
            let IvfVectors::F32(vectors) = vectors;
            let IvfCentroids::F32(centroids) = centroids;
            Ok(vectors
                .matrix
                .values
                .chunks_exact(2)
                .map(|vector| {
                    let mut best = 0u32;
                    let mut best_d2 = f32::INFINITY;
                    for (i, centroid) in centroids.values.chunks_exact(2).enumerate() {
                        let dx = vector[0] - centroid[0];
                        let dy = vector[1] - centroid[1];
                        let d2 = dx * dx + dy * dy;
                        if d2 < best_d2 {
                            best = i as u32;
                            best_d2 = d2;
                        }
                    }
                    best
                })
                .collect())
        }
        fn build_router(
            &self,
            options: &VectorOptions,
            centroids: &IvfCentroids,
        ) -> crate::Result<Option<BuiltRouter>> {
            if !self.stacked {
                return Ok(None);
            }
            let IvfCentroids::F32(matrix) = centroids;
            let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
            let (index, perm) = IvfIndexBuilder::new(
                matrix.values.clone(),
                matrix.rows,
                options.dim(),
                &clusterer,
                IvfConfig::new(2),
            )
            .build();
            Ok(Some(BuiltRouter::Stacked { index, perm }))
        }
    }

    /// A 2-dim IVF index over `commits` (one flat segment per inner
    /// slice), merged per `merge_plan` (segment ordinals into the
    /// searchable set at each step). Returns the index and the field.
    fn build_ivf_with_plan(
        metric: Metric,
        clusterer: TestClusterer,
        commits: &[&[[f32; 2]]],
        directory: Option<RamDirectory>,
    ) -> crate::Result<(Index, crate::schema::Field)> {
        let mut schema_builder = Schema::builder();
        let embed_field = schema_builder.add_vector_field(
            EMBEDDING_FIELD_NAME,
            VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
        );
        schema_builder.add_text_field("label", STRING | STORED);
        let schema = schema_builder.build();
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let builder = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(clusterer));
        let index = match directory {
            Some(directory) => builder.open_or_create(directory)?,
            None => builder.create_in_ram()?,
        };
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for chunk in commits {
            for vector in *chunk {
                let mut doc = TantivyDocument::new();
                doc.add_vector(embed_field, vector.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }
        Ok((index, embed_field))
    }

    /// Merge every searchable segment pair-wise per `pairs`, then all
    /// remaining segments into one.
    fn merge_all(index: &Index) -> crate::Result<()> {
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;
        Ok(())
    }

    /// Merge-level saturation: a huge-but-finite L2 member whose residual
    /// overflows `f32` saturates its cluster through `add_native`; a
    /// zero-norm cosine centroid saturates through the explicit
    /// degenerate-centroid mark. Finite clusters stay finite.
    #[test]
    fn saturated_sentinel() -> crate::Result<()> {
        // L2: the doc at (3e38, 3e38) is finite (passes ingest) but its
        // residual against centroid (0, 0) is sqrt(2)*3e38 > f32::MAX.
        let (index, field) = build_ivf_with_plan(
            Metric::L2,
            TestClusterer {
                fixed_centroids: Some(vec![[0.0, 0.0], [50.0, 50.0]]),
                num_centroids: 2,
                stacked: false,
            },
            &[&[[3.0e38, 3.0e38]], &[[50.0, 50.0], [50.5, 50.0]]],
            None,
        )?;
        merge_all(&index)?;
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(field)?;
        let ivf = vec_reader.index().expect("IVF segment");
        let bounds = ivf.bounds();
        // Cluster ids are trained-centroid indices: the big doc's d2
        // overflows to +inf against both centroids, and the assign rule's
        // strict `<` keeps the first — cluster 0, whatever the merge's
        // doc order. The (50, *) docs sit in cluster 1.
        assert_eq!(
            bounds.ball_r(0),
            f32::INFINITY,
            "overflowing residual must saturate"
        );
        assert!(
            bounds.ball_r(1).is_finite(),
            "finite cluster must stay finite: {}",
            bounds.ball_r(1)
        );

        // Cosine: an all-zero cluster renormalizes its centroid to
        // zero-norm → the degenerate-centroid saturation path.
        let (index, field) = build_ivf_with_plan(
            Metric::Cosine,
            TestClusterer {
                fixed_centroids: Some(vec![[0.0, 0.0], [10.0, 10.0]]),
                num_centroids: 2,
                stacked: false,
            },
            &[&[[0.0, 0.0], [0.0, 0.0]], &[[10.0, 10.0], [10.0, 10.5]]],
            None,
        )?;
        merge_all(&index)?;
        let searcher = index.reader()?.searcher();
        let vec_reader = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vec_reader.index().expect("IVF segment");
        assert_eq!(
            ivf.bounds().ball_r(0),
            f32::INFINITY,
            "zero-norm cosine centroid must saturate its cluster"
        );
        assert!(ivf.bounds().ball_r(1).is_finite());
        Ok(())
    }

    /// A merge of merged segments re-runs the fold against the NEW
    /// centroids over the re-assignment output: the stored bounds equal a
    /// fresh fold, and exceed every input segment's bounds — no
    /// combination of input radii could produce them.
    #[test]
    fn merge_recomputes_bounds() -> crate::Result<()> {
        let clusterer = || TestClusterer {
            fixed_centroids: None, // train on the first sample → data-dependent
            num_centroids: 1,
            stacked: false,
        };
        let (index, field) = build_ivf_with_plan(
            Metric::L2,
            clusterer(),
            &[
                &[[0.0, 0.0], [0.1, 0.1]],
                &[[0.05, 0.0], [0.0, 0.05]],
                &[[10.0, 10.0], [10.1, 10.1]],
                &[[10.0, 10.05], [10.05, 10.0]],
            ],
            None,
        )?;
        // Stage 1: two IVF segments, each trained on its own half.
        {
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            let mut segment_ids = index.searchable_segment_ids()?;
            segment_ids.sort();
            for pair in segment_ids.chunks_exact(2) {
                writer.merge(pair).wait()?;
            }
            writer.wait_merging_threads()?;
        }
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 2);
        let mut max_input_bound = 0.0f32;
        for segment_reader in searcher.segment_readers() {
            let vec_reader = segment_reader.vector_index(field)?;
            let ivf = vec_reader.index().expect("stage-1 segments are IVF");
            for &value in ivf.bounds().values() {
                assert!(value.is_finite());
                max_input_bound = max_input_bound.max(value);
            }
        }
        // Tight per-half clusters: every input bound is small.
        assert!(
            max_input_bound < 1.0,
            "stage-1 bounds should be tight: {max_input_bound}"
        );

        // Stage 2: merge the merged segments. Training now sees the
        // union and re-anchors the single centroid near (0, 0), so the
        // far half's residuals stretch the fold far past any input value.
        merge_all(&index)?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let vec_reader = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vec_reader.index().expect("merged segment is IVF");
        let stored: Vec<f32> = ivf.bounds().values().to_vec();
        let expected = fresh_fold(&vec_reader, ivf, 2, Metric::L2)?;
        for (cluster, (&got, &fold)) in stored.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                fold.to_bits(),
                "cluster {cluster}: stored {got} != fresh fold {fold}"
            );
        }
        let max_merged = stored.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_merged > max_input_bound * 10.0,
            "merged fold ({max_merged}) must exceed any fold of input bounds ({max_input_bound})"
        );
        Ok(())
    }

    /// The two-cluster fixture the version-policy tests below share.
    fn version_policy_fixture() -> crate::Result<(Index, crate::schema::Field)> {
        let (index, field) = build_ivf_with_plan(
            Metric::L2,
            TestClusterer {
                fixed_centroids: Some(vec![[0.0, 0.0], [10.0, 10.0]]),
                num_centroids: 2,
                stacked: false,
            },
            &[&[[0.0, 0.0], [0.1, 0.0]], &[[10.0, 10.0], [10.1, 10.0]]],
            None,
        )?;
        merge_all(&index)?;
        Ok((index, field))
    }

    /// Rewrite the merged segment's `.centroids` as a file stamped
    /// `version` carrying only `slots`, each copied verbatim from the
    /// original composite — a facsimile of what an older writer produced.
    /// Reads and writes go through the index's managed directory so the
    /// file-level footer is stripped and re-appended correctly.
    fn rewrite_centroids_with_slots(
        index: &Index,
        field: crate::schema::Field,
        version: u32,
        slots: &[usize],
    ) -> crate::Result<()> {
        let segment = index
            .searchable_segments()?
            .into_iter()
            .next()
            .expect("one merged segment");
        let path = segment.relative_path(SegmentComponent::Custom(CENTROIDS_EXT.to_string()));
        let directory = index.directory();
        let file = directory.open_read(&path)?;
        assert!(file.len() > 4);
        let composite = CompositeFile::open(&file.slice_from(4))?;
        let slot_payloads: Vec<(usize, Vec<u8>)> = slots
            .iter()
            .map(|&slot| {
                let slice = composite
                    .open_read_with_idx(field, slot)
                    .unwrap_or_else(|| panic!("slot {slot} missing from the original file"));
                Ok((slot, slice.read_bytes()?.to_vec()))
            })
            .collect::<crate::Result<_>>()?;

        directory.delete(&path).expect("delete .centroids");
        let mut writer = directory.open_write(&path).expect("rewrite .centroids");
        writer.write_all(&version.to_le_bytes())?;
        let mut composite_write = CompositeWrite::wrap(writer);
        for (slot, payload) in slot_payloads {
            let slot_w = composite_write.for_field_with_idx(field, slot);
            slot_w.write_all(&payload)?;
            slot_w.flush()?;
        }
        composite_write.close()?;
        Ok(())
    }

    /// A V1 `.centroids` (no bounds slot — the shape 0.25 wrote) opens
    /// with SATURATED bounds: every cluster probes, search stays correct.
    #[test]
    fn v1_centroids_without_bounds_opens_saturated() -> crate::Result<()> {
        let (index, field) = version_policy_fixture()?;
        rewrite_centroids_with_slots(&index, field, 1, &[0, 1])?;

        let searcher = index.reader()?.searcher();
        let vec_reader = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vec_reader.index().expect("V1 file must open as IVF");
        assert_eq!(ivf.num_clusters(), 2);
        assert_eq!(ivf.bounds().kind(), BoundKind::Ball);
        assert!(
            ivf.bounds()
                .values()
                .iter()
                .all(|bound| bound.is_infinite()),
            "V1 bounds must synthesize as SATURATED: {:?}",
            ivf.bounds().values()
        );

        // Saturated bounds never certify a skip, so the probe loop scores
        // everything it routes to — the true nearest doc must win.
        let query = vec![10.0f32, 10.0];
        let hits = searcher
            .search(
                &AllQuery,
                &TopDocs::with_limit(1).order_by_similarity(field, query.clone()),
            )?
            .results;
        assert_eq!(hits.len(), 1);
        let top_bytes = vec_reader
            .vector_bytes(hits[0].1.doc_id)?
            .expect("top hit has a vector");
        let top_vector: Vec<f32> = top_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(top_vector, query, "top hit must be the exact-match doc");
        Ok(())
    }

    /// Bounds are read whenever the slot exists, whatever the stamp: a
    /// V1-restamped file that still carries slot [3] keeps its real bounds.
    #[test]
    fn v1_centroids_with_bounds_reads_them() -> crate::Result<()> {
        let (index, field) = version_policy_fixture()?;
        rewrite_centroids_with_slots(&index, field, 1, &[0, 1, 3])?;

        let searcher = index.reader()?.searcher();
        let vec_reader = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vec_reader.index().expect("V1 file must open as IVF");
        assert!(
            ivf.bounds().values().iter().all(|bound| bound.is_finite()),
            "present bounds must be read, not synthesized: {:?}",
            ivf.bounds().values()
        );
        Ok(())
    }

    /// A V2 `.centroids` without slot [3] is corrupt, not old: V2 writers
    /// always fold bounds, so absence means the file is damaged.
    #[test]
    fn v2_centroids_missing_bounds_is_corrupt() -> crate::Result<()> {
        let (index, field) = version_policy_fixture()?;
        rewrite_centroids_with_slots(&index, field, 2, &[0, 1])?;

        let searcher = index.reader()?.searcher();
        let message = match searcher.segment_readers()[0].vector_index(field) {
            Ok(_) => panic!("a V2 .centroids without bounds must be refused"),
            Err(err) => err.to_string(),
        };
        assert!(
            message.contains("no bounds slot") && message.contains("\"embedding\""),
            "unexpected error text: {message}"
        );
        Ok(())
    }

    /// A clusterer that builds a stacked router: the merge stamps V3,
    /// writes the router into slot [2], and cascades the canonical
    /// centroid permutation into slot [0] / the assignments / the bounds —
    /// proven by parsing the router against the stored rows and by search
    /// staying exact.
    #[test]
    fn ivf_merge_writes_stacked_slot_at_v3() -> crate::Result<()> {
        use crate::vector::header::{read_header, VectorFileVersion};

        // 8 fixed centroids on a line; two tight docs per centroid split
        // across two commits.
        let centroids: Vec<[f32; 2]> = (0..8).map(|i| [i as f32 * 10.0, 0.0]).collect();
        let docs: Vec<[f32; 2]> = centroids
            .iter()
            .flat_map(|c| [[c[0] + 0.1, 0.0], [c[0] + 0.2, 0.0]])
            .collect();
        let (first, second) = docs.split_at(docs.len() / 2);
        let (index, field) = build_ivf_with_plan(
            Metric::L2,
            TestClusterer {
                fixed_centroids: Some(centroids.clone()),
                num_centroids: centroids.len(),
                stacked: true,
            },
            &[first, second],
            None,
        )?;
        merge_all(&index)?;

        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let centroids_file =
            segment_reader.open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string()))?;
        let (version, body) = read_header(&centroids_file)?;
        assert_eq!(version, VectorFileVersion::V3, "stacked files stamp V3");
        let composite = CompositeFile::open(&body)?;
        let stacked_bytes = composite
            .open_read_with_idx(field, 2)
            .expect("stacked router slot must be written")
            .read_bytes()?;

        // The reader parses the router at open and exposes it.
        let vec_reader = segment_reader.vector_index(field)?;
        let ivf = vec_reader.index().expect("IVF segment");
        let persisted = ivf
            .stacked()
            .expect("reader must parse the V3 stacked router");
        assert!(
            persisted.depth() >= 2,
            "builder must hang a parent over 8 centroids at branching_factor 2"
        );
        assert_eq!(persisted.vectors.len(), centroids.len());
        assert_eq!(
            persisted
                .offsets
                .iter()
                .map(|&(_, end)| end)
                .max()
                .unwrap_or(0) as usize,
            centroids.len(),
            "persisted level-0 offsets must cover the stored centroid rows"
        );
        let stored_rows: Vec<f32> = ivf
            .centroid_bytes()?
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(stacked_bytes[0], 1, "V3 stacked router kind byte");
        let stacked = StackedIvfIndex::deserialize_owned(
            &stacked_bytes[1..],
            stored_rows.clone(),
            2,
            IvfConfig::default(),
        )
        .expect("slot [2] must parse as a stacked router");
        assert_eq!(stacked.vectors.len(), centroids.len());
        assert_eq!(
            stacked
                .offsets
                .iter()
                .map(|&(_, end)| end)
                .max()
                .unwrap_or(0) as usize,
            centroids.len(),
            "level-0 offsets must cover the stored centroid rows"
        );

        // The reader's lazy router and the owned parse agree, query by
        // query — the FileSlice path scores the same rows.
        for c in &centroids {
            let query = [c[0] + 0.3, 0.2];
            let owned = stacked.search(&query, 2, 1.0, Metric::L2);
            let lazy = persisted.search(&query, 2, 1.0, Metric::L2);
            assert_eq!(owned.len(), lazy.len());
            for (o, l) in owned.iter().zip(&lazy) {
                assert_eq!(u32::from(o.node), u32::from(l.node));
                assert_eq!(o.sim, l.sim);
            }
        }

        // The stored rows are a permutation of the trained centroids.
        let mut stored_sorted: Vec<[f32; 2]> = stored_rows
            .chunks_exact(2)
            .map(|row| [row[0], row[1]])
            .collect();
        stored_sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));
        assert_eq!(stored_sorted, centroids, "rows permuted, none lost");

        // End-to-end search stays exact — the permutation cascaded
        // coherently into the assignments and posting rows.
        for c in &centroids {
            let query = vec![c[0] + 0.12, 0.0];
            let hits = searcher
                .search(
                    &AllQuery,
                    &TopDocs::with_limit(1).order_by_similarity(field, query),
                )?
                .results;
            assert_eq!(hits.len(), 1);
            let bytes = vec_reader
                .vector_bytes(hits[0].1.doc_id)?
                .expect("top hit has a vector");
            let top: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            assert_eq!(top, vec![c[0] + 0.1, 0.0], "nearest doc must win");
        }
        Ok(())
    }
}
