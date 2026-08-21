#![allow(dead_code)]

use std::sync::Arc;

use crate::collector::Count;
use crate::index::CentroidSetMeta;
use crate::indexer::NoMergePolicy;
use crate::query::TermQuery;
use crate::schema::{Field, FieldType, IndexRecordOption, Schema, Term, STORED, STRING};
use crate::vector::ivf::AdaptiveProbeParams;
use crate::vector::{
    CentroidProducer, IvfCentroids, IvfMatrix, Metric, VectorDType, VectorOptions,
};
use crate::{DocAddress, Index, Score, TantivyDocument};

const EMBEDDING_FIELD_NAME: &str = "embedding";
const LABEL_FIELD_NAME: &str = "label";
const NUM_DOCS: usize = 100;
const DOCS_PER_SEGMENT: usize = 10;

pub(crate) struct TestVectorIndex {
    pub(crate) index: Index,
}

pub(crate) struct TestVectorIndexBuilder {
    centroids: Vec<[f32; grid2d::DIM]>,
    dtype: VectorDType,
    metric: Metric,
    selectivities: Vec<f32>,
}

impl TestVectorIndexBuilder {
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
        let index = Index::builder()
            .schema(schema)
            .centroid_producer(Arc::new(Grid2DCentroidProducer {
                centroids: self.centroids.clone(),
            }))
            .create_in_ram()?;
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

        // Merge pairwise so the index holds both merged and per-commit
        // segments — every one clustered against the same set.
        let mut segment_ids = index.searchable_segment_ids()?;
        segment_ids.sort();
        for pair in segment_ids.chunks_exact(2) {
            writer.merge(pair).wait()?;
        }
        writer.wait_merging_threads()?;

        Ok(TestVectorIndex { index })
    }
}

impl TestVectorIndex {
    pub(crate) fn builder(dtype: VectorDType) -> TestVectorIndexBuilder {
        TestVectorIndexBuilder {
            centroids: grid2d::centroids(),
            dtype,
            metric: Metric::L2,
            selectivities: Vec::new(),
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

/// Fixed-centroid [`CentroidProducer`]: the consumer "trained" these
/// centroids elsewhere; tantivy only assigns against them.
pub(crate) struct Grid2DCentroidProducer {
    pub(crate) centroids: Vec<[f32; grid2d::DIM]>,
}

impl CentroidProducer for Grid2DCentroidProducer {
    fn centroids(&self, _field: Field, options: &VectorOptions) -> crate::Result<IvfCentroids> {
        assert_eq!(options.dim(), grid2d::DIM);
        Ok(IvfCentroids::F32(IvfMatrix {
            values: self
                .centroids
                .iter()
                .flat_map(|centroid| centroid.iter().copied())
                .collect(),
            rows: self.centroids.len(),
            dims: grid2d::DIM,
        }))
    }
}

/// Resolve and open the index's newest centroid set file.
pub(crate) fn open_centroid_set(
    index: &Index,
) -> crate::Result<crate::vector::centroid_set::CentroidSetReader> {
    let meta = index.load_metas()?;
    let set: &CentroidSetMeta = meta
        .centroid_set
        .as_ref()
        .expect("index has a centroid set");
    crate::vector::centroid_set::CentroidSetReader::open(
        index.directory(),
        std::path::Path::new(&set.filename),
    )
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

/// Every segment — merged or straight from a commit — is clustered
/// against the index-level set.
#[test]
fn every_segment_is_clustered_against_the_set() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32).build()?;
    let searcher = index.index.reader()?.searcher();
    assert!(!searcher.segment_readers().is_empty());
    for segment_reader in searcher.segment_readers() {
        let vec_reader = segment_reader.vector_index(index.embedding_field())?;
        let ivf = vec_reader.index().expect("every segment is IVF");
        assert_eq!(ivf.num_clusters(), grid2d::centroids().len());
    }
    Ok(())
}

/// The `.vec` file stamps the current format-generation header ahead of
/// its composite body, and no per-segment `.centroids` sidecar exists.
#[test]
fn vector_files_stamp_format_version_header() -> crate::Result<()> {
    use crate::directory::CompositeFile;
    use crate::index::SegmentComponent;
    use crate::vector::header::{read_header, VectorFileVersion};
    use crate::vector::VEC_EXT;

    let index = TestVectorIndex::builder(VectorDType::F32).build()?;
    let searcher = index.index.reader()?.searcher();
    assert!(!searcher.segment_readers().is_empty());

    for segment_reader in searcher.segment_readers() {
        let vec_file = segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string()))?;
        let (version, body) = read_header(&vec_file)?;
        assert_eq!(version, VectorFileVersion::V3);
        // Body must be a valid composite — proves the stamp sits in front
        // of the framing, not inside a slot.
        CompositeFile::open(&body)?;

        assert!(
            segment_reader
                .open_read(SegmentComponent::Custom("centroids".to_string()))
                .is_err(),
            "the per-segment `.centroids` sidecar must not exist"
        );
    }
    Ok(())
}

#[test]
fn fixture_vectors_round_trip_from_readers() -> crate::Result<()> {
    let sort_2d = |values: &mut Vec<[f32; grid2d::DIM]>| {
        values.sort_by(|left, right| {
            left[0]
                .partial_cmp(&right[0])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left[1]
                        .partial_cmp(&right[1])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
    };
    let mut expected = grid2d::vectors(NUM_DOCS);
    sort_2d(&mut expected);

    let index = TestVectorIndex::builder(VectorDType::F32).build()?;
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
    sort_2d(&mut got);
    assert_eq!(got, expected);
    Ok(())
}

/// The set file stores the consumer's centroids verbatim (L2: no
/// normalization), and every doc lands in its nearest cell.
#[test]
fn set_centroids_round_trip_and_drive_assignment() -> crate::Result<()> {
    let centroids = vec![[0.0, 0.0], [6.0, 6.0]];
    let index = TestVectorIndex::builder(VectorDType::F32)
        .centroids(&centroids)
        .build()?;
    let centroid_values: Vec<f32> = centroids
        .iter()
        .flat_map(|vector| vector.iter().copied())
        .collect();

    let set = open_centroid_set(&index.index)?;
    let field_centroids = set.field_centroids(index.embedding_field(), &index.vector_options())?;
    assert_eq!(field_centroids.num_centroids(), centroids.len());
    assert_eq!(
        field_centroids.values_f32(&index.vector_options())?,
        centroid_values
    );

    let searcher = index.index.reader()?.searcher();
    let mut assigned_docs = 0;
    for segment_reader in searcher.segment_readers() {
        let vec_reader = segment_reader.vector_index(index.embedding_field())?;
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

/// The set file's router slot: tag byte 0 (tantivy RNG) followed by the
/// serialized centroid graph.
#[test]
fn centroid_set_writes_tagged_router_slot() -> crate::Result<()> {
    use crate::vector::ivf::graph::EMPTY;
    use crate::vector::{NeighborhoodGraphConfig, ROUTER_KIND_RNG};

    let centroids = vec![[0.0, 0.0], [6.0, 6.0]];
    let index = TestVectorIndex::builder(VectorDType::F32)
        .centroids(&centroids)
        .build()?;
    let set = open_centroid_set(&index.index)?;
    let router_bytes = set
        .router_slice(index.embedding_field())
        .expect("the set must write a router slot for C > 1")
        .read_bytes()?;
    assert_eq!(router_bytes[0], ROUTER_KIND_RNG);

    let words: Vec<u32> = router_bytes[1..]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("u32 word")))
        .collect();
    assert_eq!(
        words.len() * 4 + 1,
        router_bytes.len(),
        "tag byte + whole number of u32s"
    );
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

/// A single-centroid provider, for tests that only need a valid set.
pub(crate) struct SingleCellCentroidProducer {
    pub(crate) dim: usize,
}

impl CentroidProducer for SingleCellCentroidProducer {
    fn centroids(&self, _field: Field, options: &VectorOptions) -> crate::Result<IvfCentroids> {
        Ok(IvfCentroids::F32(IvfMatrix {
            values: vec![0.0; options.dim()],
            rows: 1,
            dims: options.dim(),
        }))
    }
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
        let index = Index::builder()
            .schema(schema)
            .centroid_producer(Arc::new(SingleCellCentroidProducer { dim: 2 }))
            .create_in_ram()?;
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
        let index = Index::builder()
            .schema(schema)
            .centroid_producer(Arc::new(SingleCellCentroidProducer { dim: 2 }))
            .create_in_ram()?;
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
/// it is stored — as zeros — alongside the normalized rows.
#[test]
fn ingest_accepts_zero_vector() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let embedding_field =
        schema_builder.add_vector_field("embedding", VectorOptions::new(2, Metric::Cosine));
    let schema = schema_builder.build();
    let index = Index::builder()
        .schema(schema)
        .centroid_producer(Arc::new(SingleCellCentroidProducer { dim: 2 }))
        .create_in_ram()?;
    let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
    let mut zero_doc = TantivyDocument::new();
    zero_doc.add_vector(embedding_field, &[0.0_f32, 0.0]);
    writer.add_document(zero_doc)?;
    let mut unit_doc = TantivyDocument::new();
    unit_doc.add_vector(embedding_field, &[0.6_f32, 0.8]);
    writer.add_document(unit_doc)?;
    writer.commit()?;

    let searcher = index.reader()?.searcher();
    let vec_reader = searcher.segment_readers()[0].vector_index(embedding_field)?;
    assert_eq!(vec_reader.num_vectors(), 2);
    let zero = vec_reader.vector_bytes(0)?.expect("zero doc stored");
    assert!(zero.iter().all(|&b| b == 0), "zero vector stays zero");
    let unit = vec_reader.vector_bytes(1)?.expect("unit doc stored");
    assert!(unit.iter().any(|&b| b != 0));
    Ok(())
}

/// "Scan everything" probe params: the full-capacity ceiling, so the
/// budget never binds before the stream is exhausted. Kept for the
/// cross-segment search path (TODO).
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
// Index creation, meta, and GC around the centroid set file
// ======================================================================

mod centroid_set_lifecycle_tests {
    use std::sync::Arc;

    use super::{open_centroid_set, Grid2DCentroidProducer, SingleCellCentroidProducer};
    use crate::directory::{Directory, RamDirectory};
    use crate::indexer::NoMergePolicy;
    use crate::schema::{Schema, STORED, STRING};
    use crate::vector::centroid_set::centroid_set_filename;
    use crate::vector::{Metric, VectorOptions};
    use crate::{Index, IndexWriter, TantivyDocument};

    fn vector_schema() -> Schema {
        let mut sb = Schema::builder();
        sb.add_vector_field("embedding", VectorOptions::new(2, Metric::L2));
        sb.add_text_field("label", STRING | STORED);
        sb.build()
    }

    /// A centroid producer without vector fields is refused; the converse —
    /// vector fields without a set — is the flat (mutable/staging) tier
    /// and creates fine.
    #[test]
    fn centroid_index_requires_vector_fields() {
        let mut sb = Schema::builder();
        sb.add_text_field("label", STRING);
        let err = Index::builder()
            .schema(sb.build())
            .centroid_producer(Arc::new(SingleCellCentroidProducer { dim: 2 }))
            .create_in_ram()
            .unwrap_err();
        assert!(
            err.to_string().contains("no vector fields"),
            "unexpected: {err}"
        );

        let index = Index::builder()
            .schema(vector_schema())
            .create_in_ram()
            .expect("a no-set index stores vectors flat");
        assert!(index.load_metas().unwrap().centroid_set.is_none());
    }

    /// The set file is written at creation, listed in the meta, carried
    /// forward through commits, and survives commit-triggered GC.
    #[test]
    fn centroid_set_file_survives_commits_and_gc() -> crate::Result<()> {
        let directory = RamDirectory::create();
        let index = Index::builder()
            .schema(vector_schema())
            .centroid_producer(Arc::new(Grid2DCentroidProducer {
                centroids: vec![[0.0, 0.0], [10.0, 10.0]],
            }))
            .create(directory.clone())?;
        let embed_field = index.schema().get_field("embedding").unwrap();
        let set_path = centroid_set_filename();
        assert!(directory.exists(&set_path)?, "set file written at creation");
        assert_eq!(
            index.load_metas()?.centroid_set,
            Some(crate::index::CentroidSetMeta {
                filename: set_path.to_string_lossy().into_owned(),
            })
        );

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for v in [[0.1_f32, 0.0], [9.9, 10.1], [0.2, 0.1], [10.2, 9.8]] {
            let mut doc = TantivyDocument::new();
            doc.add_vector(embed_field, &v);
            writer.add_document(doc)?;
            writer.commit()?;
        }
        // Commits run GC; an explicit pass on top for good measure.
        writer.garbage_collect_files().wait()?;
        assert!(
            directory.exists(&set_path)?,
            "GC must keep the centroid set file alive"
        );
        // Meta still lists the set after save_metas rebuilds the meta.
        assert!(index.load_metas()?.centroid_set.is_some());
        // And the file still opens.
        open_centroid_set(&index)?;

        // Merges keep working against the set.
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        searcher.segment_readers()[0]
            .vector_index(embed_field)?
            .index()
            .expect("merged segment is IVF");
        Ok(())
    }

    /// An index created without a centroid set writes the flat layout:
    /// no set file, no IVF remainder, doc-ordered rows.
    #[test]
    fn index_without_set_writes_flat() -> crate::Result<()> {
        let index = Index::builder().schema(vector_schema()).create_in_ram()?;
        assert!(index.load_metas()?.centroid_set.is_none());

        let embed_field = index.schema().get_field("embedding").unwrap();
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for v in [[0.1_f32, 0.0], [9.9, 10.1]] {
            let mut doc = TantivyDocument::new();
            doc.add_vector(embed_field, &v);
            writer.add_document(doc)?;
        }
        writer.commit()?;

        let searcher = index.reader()?.searcher();
        let vec = searcher.segment_readers()[0].vector_index(embed_field)?;
        assert!(vec.index().is_none(), "flat segments carry no IvfIndex");
        assert_eq!(vec.num_vectors(), 2);
        for (row, doc) in [(0usize, 0u32), (1, 1)] {
            assert_eq!(vec.doc_id_at(row), doc, "flat rows are doc-ordered");
        }
        let info = vec.info().expect("flat segments still report info");
        assert_eq!(info.num_centroids, 0);
        Ok(())
    }

    /// `open_or_create` on an existing index accepts a producer when a
    /// set is already installed (assumed identical) and refuses to
    /// install one into an existing set-less index.
    #[test]
    fn open_or_create_checks_set_presence() -> crate::Result<()> {
        let provider = || {
            Arc::new(Grid2DCentroidProducer {
                centroids: vec![[0.0, 0.0], [10.0, 10.0]],
            })
        };

        let directory = RamDirectory::create();
        let _ = Index::builder()
            .schema(vector_schema())
            .centroid_producer(provider())
            .create(directory.clone())?;
        Index::builder()
            .schema(vector_schema())
            .centroid_producer(provider())
            .open_or_create(directory)?;

        // A flat index cannot gain a set after the fact.
        let directory = RamDirectory::create();
        let _ = Index::builder()
            .schema(vector_schema())
            .create(directory.clone())?;
        let err = Index::builder()
            .schema(vector_schema())
            .centroid_producer(provider())
            .open_or_create(directory)
            .unwrap_err();
        assert!(
            err.to_string().contains("installing one after creation"),
            "unexpected: {err}"
        );
        Ok(())
    }
}

// ======================================================================
// P1: bounds storage
// ======================================================================

/// Fixture tests: the `.vec` bounds slot end to end — write fold,
/// roundtrip, merge recomputation, and the pre-V3 refusal. The pure
/// builder/kind cases live in `vector::bounds::bounds_storage_tests`.
mod bounds_storage_tests {
    use std::io::Write;
    use std::sync::Arc;

    use super::{open_centroid_set, Grid2DCentroidProducer, TestVectorIndex, EMBEDDING_FIELD_NAME};
    use crate::directory::{Directory, RamDirectory, TerminatingWrite};
    use crate::index::SegmentComponent;
    use crate::indexer::NoMergePolicy;
    use crate::schema::{Schema, STORED, STRING};
    use crate::vector::centroid_set::FieldCentroids;
    use crate::vector::ivf::IvfIndex;
    use crate::vector::{residual_norm, BoundKind, Metric, VectorDType, VectorOptions, VEC_EXT};
    use crate::{Index, IndexWriter, TantivyDocument};

    /// Recompute one segment's expected fold from its stored artifacts:
    /// per cluster, max [`residual_norm`] over the cluster's rows against
    /// the SET's stored centroid. Valid for `replicas == 1` builds, where
    /// every posting row is native.
    fn fresh_fold(
        vec_reader: &crate::vector::VectorIndexReader,
        ivf: &IvfIndex,
        set: &FieldCentroids,
        metric: Metric,
    ) -> crate::Result<Vec<f32>> {
        let mut expected = vec![0.0f32; ivf.num_clusters()];
        for cluster in 0..ivf.num_clusters() {
            let centroid: Vec<f32> = set
                .centroid_bytes(cluster)
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            // The writer's degenerate-centroid rule: non-finite, or
            // non-unit under cosine (a zero-norm row normalization left
            // as-is), saturates.
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
    /// fold over the stored rows and the set's centroids — for every
    /// metric.
    #[test]
    fn roundtrip_per_metric() -> crate::Result<()> {
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let fixture = TestVectorIndex::builder(VectorDType::F32)
                .metric(metric)
                .build()?;
            let field = fixture.embedding_field();
            let set = open_centroid_set(&fixture.index)?;
            let set_field = set.field_centroids(field, &fixture.vector_options())?;
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
                let expected = fresh_fold(&vec_reader, ivf, &set_field, metric)?;
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

    /// A 2-dim IVF index over `commits` (one segment per inner slice),
    /// assigned against the given fixed centroids. Returns the index and
    /// the field.
    fn build_ivf_with_plan(
        metric: Metric,
        centroids: Vec<[f32; 2]>,
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
        let builder = Index::builder()
            .schema(schema)
            .centroid_producer(Arc::new(Grid2DCentroidProducer { centroids }));
        let index = match directory {
            Some(directory) => builder.create(directory)?,
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

    /// Merge every searchable segment into one.
    fn merge_all(index: &Index) -> crate::Result<()> {
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;
        Ok(())
    }

    /// Saturation: a huge-but-finite L2 member whose residual overflows
    /// `f32` saturates its cluster through `add_native`; a zero-norm
    /// cosine centroid saturates through the degenerate-centroid mark
    /// (recomputed from the SET's stored bytes). Finite clusters stay
    /// finite.
    #[test]
    fn saturated_sentinel() -> crate::Result<()> {
        // L2: the doc at (3e38, 3e38) is finite (passes ingest) but its
        // residual against centroid (0, 0) is sqrt(2)*3e38 > f32::MAX.
        // Its d2 against BOTH centroids overflows to +inf, and the
        // selector's id tie-break assigns it to cluster 0.
        let (index, field) = build_ivf_with_plan(
            Metric::L2,
            vec![[0.0, 0.0], [50.0, 50.0]],
            &[&[[3.0e38, 3.0e38]], &[[50.0, 50.0], [50.5, 50.0]]],
            None,
        )?;
        merge_all(&index)?;
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(field)?;
        let ivf = vec_reader.index().expect("IVF segment");
        let bounds = ivf.bounds();
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

        // Cosine: a zero-norm centroid survives set-file normalization
        // as-is and must saturate its (empty) cluster via the
        // degenerate-centroid mark, whatever its membership.
        let (index, field) = build_ivf_with_plan(
            Metric::Cosine,
            vec![[0.0, 0.0], [10.0, 10.0]],
            &[&[[10.0, 10.0]], &[[10.0, 10.5]]],
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

    /// A merge of already-merged segments re-runs the fold over the
    /// re-assignment output against the same set: the stored bounds equal
    /// a fresh fold at every stage.
    #[test]
    fn merge_recomputes_bounds() -> crate::Result<()> {
        let (index, field) = build_ivf_with_plan(
            Metric::L2,
            vec![[0.0, 0.0], [10.0, 10.0]],
            &[
                &[[0.0, 0.0], [0.1, 0.1]],
                &[[0.05, 0.0], [0.0, 0.05]],
                &[[10.0, 10.0], [10.1, 10.1]],
                &[[10.0, 10.05], [10.05, 10.0]],
            ],
            None,
        )?;
        // Stage 1: merge pairwise.
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
        // Stage 2: merge the merged segments.
        merge_all(&index)?;

        let set = open_centroid_set(&index)?;
        let opts = VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32);
        let set_field = set.field_centroids(field, &opts)?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let vec_reader = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vec_reader.index().expect("merged segment is IVF");
        let stored: Vec<f32> = ivf.bounds().values().to_vec();
        let expected = fresh_fold(&vec_reader, ivf, &set_field, Metric::L2)?;
        for (cluster, (&got, &fold)) in stored.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                fold.to_bits(),
                "cluster {cluster}: stored {got} != fresh fold {fold}"
            );
            assert!(
                fold.is_finite() && fold > 0.0,
                "cluster {cluster} non-trivial"
            );
        }
        Ok(())
    }

    /// A pre-V3 `.vec` file is refused at open with the REINDEX remedy,
    /// verbatim.
    #[test]
    fn old_index_read_errors() -> crate::Result<()> {
        let directory = RamDirectory::create();
        let (index, field) = build_ivf_with_plan(
            Metric::L2,
            vec![[0.0, 0.0], [10.0, 10.0]],
            &[&[[0.0, 0.0], [0.1, 0.0]], &[[10.0, 10.0], [10.1, 10.0]]],
            Some(directory.clone()),
        )?;
        merge_all(&index)?;

        // Restamp the merged segment's `.vec` header to V2, body
        // unchanged — the version gate must refuse before parsing slots.
        let segment = index
            .searchable_segments()?
            .into_iter()
            .next()
            .expect("one merged segment");
        let path = segment.relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
        let bytes = directory.open_read(&path)?.read_bytes()?;
        assert!(bytes.len() > 4);
        let mut restamped = 2u32.to_le_bytes().to_vec();
        restamped.extend_from_slice(&bytes[4..]);
        directory.delete(&path).expect("delete .vec");
        let mut writer = directory.open_write(&path).expect("rewrite .vec");
        writer.write_all(&restamped)?;
        writer.terminate()?;

        let searcher = index.reader()?.searcher();
        let message = match searcher.segment_readers()[0].vector_index(field) {
            Ok(_) => panic!("pre-V3 .vec must be refused"),
            Err(err) => err.to_string(),
        };
        assert!(
            message.contains("predates the V3 index-level centroid format")
                && message.contains("\"embedding\""),
            "unexpected error text: {message}"
        );
        Ok(())
    }
}
