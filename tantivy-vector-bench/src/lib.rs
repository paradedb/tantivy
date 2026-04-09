use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use tantivy::collector::TopDocs;
use tantivy::index::Index;
use tantivy::plugin::SegmentPlugin;
use tantivy::schema::{Schema, STORED, TEXT};
use tantivy::vector::bqvec::BqVecPlugin;
use tantivy::vector::cluster::kmeans::KMeansConfig;
use tantivy::vector::cluster::plugin::{ClusterConfig, ClusterFieldConfig, ClusterPlugin, ProbeConfig};
use tantivy::vector::cluster::sampler::{VectorSampler, VectorSamplerFactory};
use tantivy::vector::rabitq::rotation::{DynamicRotator, RotatorType};
use tantivy::vector::rabitq::{self, Metric, RabitqConfig};
use tantivy::vector::search::{VectorQuery, VectorQueryConfig};
use tantivy::{DocId, IndexWriter, TantivyDocument};

use tantivy::index::SegmentReader;
use tantivy::indexer::doc_id_mapping::SegmentDocIdMapping;
use tantivy::schema::Field;

struct InMemorySampler {
    vectors: Vec<Vec<f32>>,
}

impl VectorSampler for InMemorySampler {
    fn sample_vectors(
        &self,
        _field: Field,
        doc_ids: &[DocId],
    ) -> tantivy::Result<Vec<Option<Vec<f32>>>> {
        Ok(doc_ids
            .iter()
            .map(|&id| self.vectors.get(id as usize).cloned())
            .collect())
    }

    fn dims(&self, _field: Field) -> usize {
        self.vectors.first().map_or(0, |v| v.len())
    }
}

struct InMemorySamplerFactory {
    vectors: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl VectorSamplerFactory for InMemorySamplerFactory {
    fn create_sampler(
        &self,
        _readers: &[SegmentReader],
        _doc_id_mapping: &SegmentDocIdMapping,
    ) -> tantivy::Result<Box<dyn VectorSampler>> {
        let vecs = self.vectors.lock().unwrap().clone();
        Ok(Box::new(InMemorySampler { vectors: vecs }))
    }
}

fn parse_metric(s: &str) -> PyResult<Metric> {
    match s.to_lowercase().as_str() {
        "l2" => Ok(Metric::L2),
        "ip" | "innerproduct" | "inner_product" => Ok(Metric::InnerProduct),
        "cosine" => Ok(Metric::L2), // cosine on normalized vectors = L2
        other => Err(PyRuntimeError::new_err(format!("unknown metric: {other}"))),
    }
}

#[pyclass]
struct TantivyVectorIndex {
    index: Index,
    writer: Option<IndexWriter<TantivyDocument>>,
    vec_field: Field,
    text_field: Field,
    metric: Metric,
    rotator: Arc<DynamicRotator>,
    shared_vecs: Arc<Mutex<Vec<Vec<f32>>>>,
}

#[pymethods]
impl TantivyVectorIndex {
    #[new]
    #[pyo3(signature = (dim, metric, data_dir, num_clusters_per_1k=1))]
    fn new(dim: usize, metric: &str, data_dir: &str, num_clusters_per_1k: usize) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let rotator = Arc::new(DynamicRotator::new(dim, RotatorType::MatrixRotator, 42));
        let padded_dims = rotator.padded_dim();

        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("id", TEXT | STORED);
        let vec_field = schema_builder.add_vector_field("embedding", dim);
        let schema = schema_builder.build();

        let config = RabitqConfig::new(1);
        let rotator_enc = rotator.clone();
        let bqvec = Arc::new(
            BqVecPlugin::builder()
                .vector_field(
                    vec_field,
                    rabitq::bytes_per_record(padded_dims, 0),
                    Arc::new(move |v: &[f32]| rabitq::encode(&rotator_enc, &config, metric, v)),
                )
                .build(),
        );

        let shared_vecs: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));

        let cluster_plugin = Arc::new(ClusterPlugin::new(ClusterConfig {
            clustering_threshold: 100,
            sample_ratio: 0.1,
            sample_cap: 100_000,
            kmeans: KMeansConfig {
                niter: 20,
                nredo: 1,
                seed: 42,
                ..Default::default()
            },
            num_clusters_fn: Arc::new(move |n| (n * num_clusters_per_1k / 1000).max(2)),
            fields: vec![ClusterFieldConfig {
                field: vec_field,
                dims: dim,
                padded_dims,
                ex_bits: 0,
                metric,
                rotator: rotator.clone(),
            }],
            sampler_factory: Arc::new(InMemorySamplerFactory {
                vectors: shared_vecs.clone(),
            }),
        }));

        let path = PathBuf::from(data_dir);
        std::fs::create_dir_all(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create dir: {e}")))?;

        let index = Index::builder()
            .schema(schema)
            .plugin(bqvec as Arc<dyn SegmentPlugin>)
            .plugin(cluster_plugin as Arc<dyn SegmentPlugin>)
            .create_in_dir(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("index create failed: {e}")))?;

        let writer = index
            .writer_with_num_threads(1, 500_000_000)
            .map_err(|e| PyRuntimeError::new_err(format!("writer create failed: {e}")))?;

        Ok(Self {
            index,
            writer: Some(writer),
            vec_field,
            text_field,
            metric,
            rotator,
            shared_vecs,
        })
    }

    fn insert(&mut self, ids: Vec<i64>, vectors: Vec<Vec<f32>>) -> PyResult<usize> {
        let writer = self.writer.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("writer closed after optimize"))?;

        {
            let mut vecs = self.shared_vecs.lock().unwrap();
            for v in &vectors {
                vecs.push(v.clone());
            }
        }

        let count = ids.len();
        for (id, vec) in ids.into_iter().zip(vectors.into_iter()) {
            writer
                .add_document(tantivy::doc!(
                    self.text_field => id.to_string(),
                    self.vec_field => vec
                ))
                .map_err(|e| PyRuntimeError::new_err(format!("add_document failed: {e}")))?;
        }
        Ok(count)
    }

    fn optimize(&mut self) -> PyResult<()> {
        let mut writer = self.writer.take()
            .ok_or_else(|| PyRuntimeError::new_err("already optimized"))?;

        writer.commit()
            .map_err(|e| PyRuntimeError::new_err(format!("commit failed: {e}")))?;

        let segment_ids = self.index.searchable_segment_ids()
            .map_err(|e| PyRuntimeError::new_err(format!("segment ids failed: {e}")))?;

        if segment_ids.len() > 1 {
            let mut writer: IndexWriter<TantivyDocument> = self.index
                .writer_with_num_threads(1, 500_000_000)
                .map_err(|e| PyRuntimeError::new_err(format!("writer failed: {e}")))?;
            writer.merge(&segment_ids).wait()
                .map_err(|e| PyRuntimeError::new_err(format!("merge failed: {e}")))?;
            writer.wait_merging_threads()
                .map_err(|e| PyRuntimeError::new_err(format!("wait failed: {e}")))?;
        }

        Ok(())
    }

    #[pyo3(signature = (query, k=100, max_probe=10, distance_ratio=2.0))]
    fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        max_probe: usize,
        distance_ratio: f32,
    ) -> PyResult<Vec<i64>> {
        let reader = self.index.reader()
            .map_err(|e| PyRuntimeError::new_err(format!("reader failed: {e}")))?;
        let searcher = reader.searcher();

        let config = VectorQueryConfig {
            field: self.vec_field,
            padded_dims: self.rotator.padded_dim(),
            ex_bits: 0,
            metric: self.metric,
            rotator: self.rotator.clone(),
            probe: ProbeConfig::new(max_probe, distance_ratio),
        };

        let vector_query = VectorQuery::new(query, config);
        let top_docs = searcher
            .search(&vector_query, &TopDocs::with_limit(k).order_by_score())
            .map_err(|e| PyRuntimeError::new_err(format!("search failed: {e}")))?;

        Ok(top_docs.into_iter().map(|(_, addr)| addr.doc_id as i64).collect())
    }
}

#[pymodule]
fn tantivy_vector_bench(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TantivyVectorIndex>()?;
    Ok(())
}
