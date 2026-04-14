use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use tantivy::collector::TopDocs;
use tantivy::index::Index;
use tantivy::plugin::SegmentPlugin;
use tantivy::schema::{Schema, FAST, STORED, TEXT};
use tantivy::vector::bqvec::BqVecPlugin;
use tantivy::vector::cluster::kmeans::KMeansConfig;
use tantivy::vector::cluster::plugin::{ClusterConfig, ClusterFieldConfig, ClusterPlugin, ProbeConfig};
use tantivy::vector::cluster::sampler::{VectorSampler, VectorSamplerFactory};
use tantivy::vector::rabitq::rotation::{DynamicRotator, RotatorType};
use tantivy::vector::rabitq::{self, Metric, RabitqConfig};
use tantivy::vector::search::{VectorQuery, VectorQueryConfig};
use tantivy::indexer::NoMergePolicy;
use tantivy::{DocId, IndexWriter, TantivyDocument};

use tantivy::index::SegmentReader;
use tantivy::indexer::doc_id_mapping::SegmentDocIdMapping;
use tantivy::schema::Field;

struct ParquetBackedSampler {
    parquet_dir: PathBuf,
    num_shards: usize,
    vectors_per_shard: usize,
    dim: usize,
}

impl VectorSampler for ParquetBackedSampler {
    fn sample_vectors(
        &self,
        _field: Field,
        doc_ids: &[DocId],
    ) -> tantivy::Result<Vec<Option<Vec<f32>>>> {
        use std::collections::HashMap;

        let mut by_shard: HashMap<usize, Vec<(usize, DocId)>> = HashMap::new();
        for (result_idx, &doc_id) in doc_ids.iter().enumerate() {
            let shard = (doc_id as usize) / self.vectors_per_shard;
            let shard = shard.min(self.num_shards - 1);
            by_shard.entry(shard).or_default().push((result_idx, doc_id));
        }

        let mut results: Vec<Option<Vec<f32>>> = vec![None; doc_ids.len()];

        for (shard_idx, requests) in &by_shard {
            let path = self.parquet_dir.join(format!("train-{:02}-of-{}.parquet", shard_idx, self.num_shards));
            if !path.exists() {
                continue;
            }

            let needed: std::collections::HashSet<u32> =
                requests.iter().map(|&(_, doc_id)| doc_id).collect();

            let file = std::fs::File::open(&path).map_err(|e| {
                tantivy::TantivyError::InternalError(format!("parquet open: {e}"))
            })?;
            let reader = parquet::file::reader::SerializedFileReader::new(file).map_err(|e| {
                tantivy::TantivyError::InternalError(format!("parquet reader: {e}"))
            })?;

            use parquet::file::reader::FileReader;
            let mut row_idx = *shard_idx * self.vectors_per_shard;
            let mut iter = reader.get_row_iter(None).map_err(|e| {
                tantivy::TantivyError::InternalError(format!("parquet iter: {e}"))
            })?;

            while let Some(Ok(row)) = iter.next() {
                let global_doc_id = row_idx as u32;
                if needed.contains(&global_doc_id) {
                    if let Some(parquet::record::Field::ListInternal(list)) = row.get_column_iter().find(|(name, _)| *name == "emb").map(|(_, v)| v) {
                        let vec: Vec<f32> = list.elements().iter().filter_map(|e| {
                            if let parquet::record::Field::Float(f) = e { Some(*f) } else { None }
                        }).collect();
                        if vec.len() == self.dim {
                            for &(result_idx, doc_id) in requests {
                                if doc_id == global_doc_id {
                                    results[result_idx] = Some(vec.clone());
                                }
                            }
                        }
                    }
                }
                row_idx += 1;
            }
        }

        Ok(results)
    }

    fn dims(&self, _field: Field) -> usize {
        self.dim
    }
}

struct ParquetSamplerFactory {
    parquet_dir: PathBuf,
    num_shards: usize,
    vectors_per_shard: usize,
    dim: usize,
}

impl VectorSamplerFactory for ParquetSamplerFactory {
    fn create_sampler(
        &self,
        _readers: &[SegmentReader],
        _doc_id_mapping: &SegmentDocIdMapping,
    ) -> tantivy::Result<Box<dyn VectorSampler>> {
        Ok(Box::new(ParquetBackedSampler {
            parquet_dir: self.parquet_dir.clone(),
            num_shards: self.num_shards,
            vectors_per_shard: self.vectors_per_shard,
            dim: self.dim,
        }))
    }
}

fn parse_metric(s: &str) -> PyResult<Metric> {
    match s.to_lowercase().as_str() {
        "l2" => Ok(Metric::L2),
        "ip" | "innerproduct" | "inner_product" => Ok(Metric::InnerProduct),
        "cosine" => Ok(Metric::L2),
        other => Err(PyRuntimeError::new_err(format!("unknown metric: {other}"))),
    }
}

#[pyclass]
struct TantivyVectorIndex {
    index: Index,
    writer: Option<IndexWriter<TantivyDocument>>,
    vec_field: Field,
    text_field: Field,
    id_field: Field,
    metric: Metric,
    rotator: Arc<DynamicRotator>,
    total_bits: usize,
}

#[pymethods]
impl TantivyVectorIndex {
    #[new]
    #[pyo3(signature = (dim, metric, data_dir, parquet_dir, num_shards=10, vectors_per_shard=1_000_000, num_clusters_per_1k=1, total_bits=7))]
    fn new(
        dim: usize,
        metric: &str,
        data_dir: &str,
        parquet_dir: &str,
        num_shards: usize,
        vectors_per_shard: usize,
        num_clusters_per_1k: usize,
        total_bits: usize,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let rotator = Arc::new(DynamicRotator::new(dim, RotatorType::FhtKacRotator, 42));
        let padded_dims = rotator.padded_dim();

        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let id_field = schema_builder.add_i64_field("id", FAST);
        let vec_field = schema_builder.add_vector_field("embedding", dim);
        let schema = schema_builder.build();

        let config = RabitqConfig::new(1);
        let rotator_enc = rotator.clone();
        let bqvec = Arc::new(
            BqVecPlugin::builder()
                .vector_field(
                    vec_field,
                    rabitq::bytes_per_record(padded_dims, 6),
                    Arc::new(move |v: &[f32]| {
                        let zero = vec![0.0f32; v.len()];
                        rabitq::encode(&rotator_enc, &config, metric, v, &zero)
                    }),
                )
                .build(),
        );

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
                ex_bits: total_bits.saturating_sub(1),
                metric,
                rotator: rotator.clone(),
            }],
            sampler_factory: Arc::new(ParquetSamplerFactory {
                parquet_dir: PathBuf::from(parquet_dir),
                num_shards,
                vectors_per_shard,
                dim,
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

        let writer: IndexWriter<TantivyDocument> = index
            .writer_with_num_threads(1, 4_000_000_000)
            .map_err(|e| PyRuntimeError::new_err(format!("writer create failed: {e}")))?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        Ok(Self {
            index,
            writer: Some(writer),
            vec_field,
            text_field,
            id_field,
            metric,
            rotator,
            total_bits,
        })
    }

    fn insert(&mut self, ids: Vec<i64>, vectors: Vec<Vec<f32>>) -> PyResult<usize> {
        let writer = self.writer.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("writer closed after optimize"))?;

        let count = ids.len();
        for (id, vec) in ids.into_iter().zip(vectors.into_iter()) {
            writer
                .add_document(tantivy::doc!(
                    self.id_field => id,
                    self.vec_field => vec
                ))
                .map_err(|e| PyRuntimeError::new_err(format!("add_document failed: {e}")))?;
        }
        Ok(count)
    }

    fn commit(&mut self) -> PyResult<()> {
        let writer = self.writer.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("writer closed"))?;
        writer.commit()
            .map_err(|e| PyRuntimeError::new_err(format!("commit failed: {e}")))?;
        Ok(())
    }

    fn finalize(&mut self) -> PyResult<()> {
        let writer = self.writer.take()
            .ok_or_else(|| PyRuntimeError::new_err("already finalized"))?;
        writer.wait_merging_threads()
            .map_err(|e| PyRuntimeError::new_err(format!("wait failed: {e}")))?;
        Ok(())
    }

    #[staticmethod]
    #[pyo3(signature = (dim, metric, data_dir, total_bits=7))]
    fn open(dim: usize, metric: &str, data_dir: &str, total_bits: usize) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let rotator = Arc::new(DynamicRotator::new(dim, RotatorType::FhtKacRotator, 42));
        let padded_dims = rotator.padded_dim();
        let ex_bits = total_bits.saturating_sub(1);

        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let id_field = schema_builder.add_i64_field("id", FAST);
        let vec_field = schema_builder.add_vector_field("embedding", dim);
        let schema = schema_builder.build();

        let config = RabitqConfig::new(1);
        let rotator_enc = rotator.clone();
        let bqvec = Arc::new(
            BqVecPlugin::builder()
                .vector_field(
                    vec_field,
                    rabitq::bytes_per_record(padded_dims, ex_bits),
                    Arc::new(move |v: &[f32]| {
                        let zero = vec![0.0f32; v.len()];
                        rabitq::encode(&rotator_enc, &config, metric, v, &zero)
                    }),
                )
                .build(),
        );

        let cluster_plugin = Arc::new(ClusterPlugin::new(ClusterConfig {
            clustering_threshold: u32::MAX,
            sample_ratio: 0.0,
            sample_cap: 0,
            kmeans: KMeansConfig::default(),
            num_clusters_fn: Arc::new(|_| 0),
            fields: vec![ClusterFieldConfig {
                field: vec_field,
                dims: dim,
                padded_dims,
                ex_bits,
                metric,
                rotator: rotator.clone(),
            }],
            sampler_factory: Arc::new(ParquetSamplerFactory {
                parquet_dir: PathBuf::new(),
                num_shards: 0,
                vectors_per_shard: 0,
                dim,
            }),
        }));

        let dir = tantivy::directory::MmapDirectory::open(data_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("dir open failed: {e}")))?;
        let index = Index::builder()
            .schema(schema)
            .plugin(bqvec as Arc<dyn SegmentPlugin>)
            .plugin(cluster_plugin as Arc<dyn SegmentPlugin>)
            .open_or_create(dir)
            .map_err(|e| PyRuntimeError::new_err(format!("index open failed: {e}")))?;

        Ok(Self {
            index,
            writer: None,
            vec_field,
            text_field,
            id_field,
            metric,
            rotator,
            total_bits,
        })
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
            ex_bits: self.total_bits.saturating_sub(1),
            metric: self.metric,
            rotator: self.rotator.clone(),
            probe: ProbeConfig::new(max_probe, distance_ratio),
        };

        let vector_query = VectorQuery::new(query, config);
        let top_docs = searcher
            .search(&vector_query, &TopDocs::with_limit(k).order_by_score())
            .map_err(|e| PyRuntimeError::new_err(format!("search failed: {e}")))?;

        let mut result_ids = Vec::with_capacity(top_docs.len());
        for (_, addr) in &top_docs {
            let seg_reader = searcher.segment_reader(addr.segment_ord);
            let id_reader = seg_reader
                .fast_fields()
                .i64("id")
                .map_err(|e| PyRuntimeError::new_err(format!("fast field read: {e}")))?;
            let global_id = id_reader.first(addr.doc_id).unwrap_or(0);
            result_ids.push(global_id);
        }
        Ok(result_ids)
    }
}

#[pymodule]
fn tantivy_vector_bench(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TantivyVectorIndex>()?;
    Ok(())
}
