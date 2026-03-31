use std::any::Any;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use common::file_slice::FileSlice;

use crate::bqvec::{BqVecFieldReader, BqVecPluginReader};
use crate::cluster::centroid_index::CentroidIndex;
use crate::cluster::kmeans::{run_kmeans_with_config, KMeansConfig};
use crate::cluster::sampler::VectorSamplerFactory;
use crate::directory::{CompositeFile, CompositeWrite};
use crate::index::SegmentComponent;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use crate::rabitq::math::l2_distance_sqr;
use crate::rabitq::rotation::DynamicRotator;
use crate::rabitq::{self, Metric, RaBitQQuery};
use crate::schema::document::{Document, Value};
use crate::schema::{Field, FieldType, Schema};
use crate::{DocId, Segment};

fn component() -> SegmentComponent {
    SegmentComponent::Custom("cluster".to_string())
}

fn bqvec_component() -> SegmentComponent {
    SegmentComponent::Custom("bqvec".to_string())
}

#[derive(Clone)]
pub struct ClusterFieldConfig {
    pub field: Field,
    pub dims: usize,
    pub padded_dims: usize,
    pub ex_bits: usize,
    pub metric: Metric,
    pub rotator: Arc<DynamicRotator>,
}

pub struct ClusterConfig {
    pub clustering_threshold: u32,
    pub sample_ratio: f32,
    pub sample_cap: usize,
    pub kmeans: KMeansConfig,
    pub num_clusters_fn: Arc<dyn Fn(usize) -> usize + Send + Sync>,
    pub fields: Vec<ClusterFieldConfig>,
    pub sampler_factory: Arc<dyn VectorSamplerFactory>,
}

pub struct ClusterPlugin {
    config: Arc<ClusterConfig>,
}

impl ClusterPlugin {
    pub fn new(config: ClusterConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

struct ClusterData {
    centroid_index: CentroidIndex,
    per_doc_cluster: Vec<u16>,
    cluster_offsets: Vec<u32>,
    cluster_doc_ids: Vec<DocId>,
    num_clusters: usize,
    dims: usize,
}

fn sample_indices(total: usize, ratio: f32, cap: usize) -> Vec<usize> {
    let target = ((total as f32 * ratio) as usize).min(cap).max(1).min(total);
    let step = total / target;
    (0..total).step_by(step.max(1)).take(target).collect()
}

fn build_cluster_data(
    centroids: Vec<Vec<f32>>,
    assignments: &[usize],
    num_docs: usize,
    metric: Metric,
) -> ClusterData {
    let num_clusters = centroids.len();
    let dims = centroids.first().map_or(0, |v| v.len());
    let centroid_ids: Vec<u32> = (0..num_clusters as u32).collect();
    let centroid_index = CentroidIndex::build(centroids, centroid_ids, metric);

    let per_doc_cluster: Vec<u16> = assignments.iter().map(|&c| c as u16).collect();

    let mut cluster_counts = vec![0u32; num_clusters];
    for &c in assignments {
        cluster_counts[c] += 1;
    }
    let mut cluster_offsets = Vec::with_capacity(num_clusters + 1);
    cluster_offsets.push(0u32);
    for &count in &cluster_counts {
        let prev = *cluster_offsets.last().unwrap();
        cluster_offsets.push(prev + count);
    }

    let mut cluster_doc_ids = vec![0 as DocId; num_docs];
    let mut write_pos = cluster_offsets.clone();
    for doc_id in 0..num_docs {
        let c = assignments[doc_id];
        let pos = write_pos[c] as usize;
        cluster_doc_ids[pos] = doc_id as DocId;
        write_pos[c] += 1;
    }

    ClusterData {
        centroid_index,
        per_doc_cluster,
        cluster_offsets,
        cluster_doc_ids,
        num_clusters,
        dims,
    }
}

fn cluster_from_vectors(
    vectors: &[Vec<f32>],
    config: &ClusterConfig,
    field_config: &ClusterFieldConfig,
) -> crate::Result<ClusterData> {
    let num_docs = vectors.len();
    let sample_ids = sample_indices(num_docs, config.sample_ratio, config.sample_cap);
    let sampled: Vec<Vec<f32>> = sample_ids.iter().map(|&i| vectors[i].clone()).collect();

    let k = (config.num_clusters_fn)(num_docs);
    let k = k.min(sampled.len()).max(1);
    let result = run_kmeans_with_config(&sampled, k, config.kmeans.clone());

    let mut assignments = vec![0usize; num_docs];
    for (doc_id, vec) in vectors.iter().enumerate() {
        let mut best_cluster = 0;
        let mut best_dist = f32::INFINITY;
        for (ci, centroid) in result.centroids.iter().enumerate() {
            let dist = l2_distance_sqr(vec, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_cluster = ci;
            }
        }
        assignments[doc_id] = best_cluster;
    }

    Ok(build_cluster_data(
        result.centroids,
        &assignments,
        num_docs,
        field_config.metric,
    ))
}

fn cluster_from_merge(
    sampler: &dyn crate::cluster::sampler::VectorSampler,
    bqvec_reader: &BqVecFieldReader,
    field_config: &ClusterFieldConfig,
    config: &ClusterConfig,
    num_docs: usize,
) -> crate::Result<ClusterData> {
    let sample_ids = sample_indices(num_docs, config.sample_ratio, config.sample_cap);
    let sample_doc_ids: Vec<DocId> = sample_ids.iter().map(|&i| i as DocId).collect();
    let sampled_vecs = sampler.sample_vectors(field_config.field, &sample_doc_ids)?;

    let valid_vecs: Vec<Vec<f32>> = sampled_vecs.into_iter().flatten().collect();
    if valid_vecs.is_empty() {
        return Err(crate::TantivyError::InternalError(
            "no valid vectors returned by sampler".into(),
        ));
    }

    let k = (config.num_clusters_fn)(num_docs);
    let k = k.min(valid_vecs.len()).max(1);
    let result = run_kmeans_with_config(&valid_vecs, k, config.kmeans.clone());

    let centroid_queries: Vec<RaBitQQuery> = result
        .centroids
        .iter()
        .map(|c| rabitq::prepare_query(&field_config.rotator, c, field_config.ex_bits, field_config.metric))
        .collect();

    let mut assignments = vec![0usize; num_docs];
    for doc_id in 0..num_docs {
        let record = bqvec_reader.record(doc_id as DocId)?;
        let mut best_cluster = 0;
        let mut best_dist = f32::INFINITY;
        for (ci, query) in centroid_queries.iter().enumerate() {
            let dist = query.estimate_distance_from_record(&record, field_config.padded_dims);
            if dist < best_dist {
                best_dist = dist;
                best_cluster = ci;
            }
        }
        assignments[doc_id] = best_cluster;
    }

    Ok(build_cluster_data(
        result.centroids,
        &assignments,
        num_docs,
        field_config.metric,
    ))
}

fn serialize_cluster_data(data: &ClusterData, w: &mut dyn Write) -> crate::Result<()> {
    w.write_all(&(data.num_clusters as u32).to_le_bytes())?;
    w.write_all(&(data.per_doc_cluster.len() as u32).to_le_bytes())?;
    w.write_all(&(data.dims as u32).to_le_bytes())?;

    let ci_bytes = data.centroid_index.save_to_bytes()?;
    w.write_all(&(ci_bytes.len() as u32).to_le_bytes())?;
    w.write_all(&ci_bytes)?;

    for &cluster_id in &data.per_doc_cluster {
        w.write_all(&cluster_id.to_le_bytes())?;
    }
    for &offset in &data.cluster_offsets {
        w.write_all(&offset.to_le_bytes())?;
    }
    for &doc_id in &data.cluster_doc_ids {
        w.write_all(&doc_id.to_le_bytes())?;
    }

    Ok(())
}

fn serialize_empty(w: &mut dyn Write) -> crate::Result<()> {
    w.write_all(&0u32.to_le_bytes())?; // num_clusters = 0
    w.write_all(&0u32.to_le_bytes())?; // num_docs = 0
    w.write_all(&0u32.to_le_bytes())?; // dims = 0
    Ok(())
}

impl SegmentPlugin for ClusterPlugin {
    fn name(&self) -> &str {
        "cluster"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["cluster"]
    }

    fn write_phase(&self) -> u32 {
        3
    }

    fn create_writer(&self, _ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(ClusterPluginWriter {
            config: self.config.clone(),
            per_field_vectors: HashMap::new(),
        }))
    }

    fn open_reader(&self, ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        let file_slice = ctx.segment_reader.open_read(component())?;
        let composite = CompositeFile::open(&file_slice)?;

        let mut field_readers = HashMap::new();
        for field_cfg in &self.config.fields {
            if let Some(field_slice) = composite.open_read(field_cfg.field) {
                let reader = ClusterFieldReader::open(field_slice, field_cfg.metric)?;
                field_readers.insert(field_cfg.field, reader);
            }
        }

        Ok(Arc::new(ClusterPluginReader { field_readers }))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        let num_docs = ctx.doc_id_mapping.iter_old_doc_addrs().count();

        let write = ctx.target_segment.open_write(component())?;
        let mut composite = CompositeWrite::wrap(write);

        if (num_docs as u32) < self.config.clustering_threshold {
            for field_cfg in &self.config.fields {
                let w = composite.for_field(field_cfg.field);
                serialize_empty(w)?;
                w.flush()?;
            }
            composite.close()?;
            return Ok(());
        }

        let sampler = self
            .config
            .sampler_factory
            .create_sampler(ctx.readers, ctx.doc_id_mapping)?;

        let bqvec_file = ctx.target_segment.open_read(bqvec_component())?;
        let bqvec_fields: Vec<Field> = self.config.fields.iter().map(|f| f.field).collect();
        let bqvec_reader = BqVecPluginReader::open(bqvec_file, &bqvec_fields)?;

        for field_cfg in &self.config.fields {
            let bqvec_field_reader = bqvec_reader.field_reader(field_cfg.field).ok_or_else(|| {
                crate::TantivyError::InternalError(format!(
                    "bqvec field reader missing for cluster merge"
                ))
            })?;

            let data = cluster_from_merge(
                sampler.as_ref(),
                bqvec_field_reader,
                field_cfg,
                &self.config,
                num_docs,
            )?;

            let w = composite.for_field(field_cfg.field);
            serialize_cluster_data(&data, w)?;
            w.flush()?;
        }

        composite.close()?;
        Ok(())
    }
}

pub struct ClusterPluginWriter {
    config: Arc<ClusterConfig>,
    per_field_vectors: HashMap<Field, Vec<Vec<f32>>>,
}

impl ClusterPluginWriter {
    pub fn ingest_vectors<D: Document>(&mut self, doc: &D, schema: &Schema) {
        for field_cfg in &self.config.fields {
            let vectors = self
                .per_field_vectors
                .entry(field_cfg.field)
                .or_insert_with(Vec::new);

            let mut found = false;
            for (field, value) in doc.iter_fields_and_values() {
                if field != field_cfg.field {
                    continue;
                }
                let field_entry = schema.get_field_entry(field);
                if !matches!(field_entry.field_type(), FieldType::Vector(_)) {
                    continue;
                }
                let value = value.as_value();
                if let Some(vec_data) = value.as_leaf().and_then(|leaf| leaf.as_vector()) {
                    vectors.push(vec_data.to_vec());
                    found = true;
                    break;
                }
            }
            if !found {
                vectors.push(vec![0.0f32; field_cfg.dims]);
            }
        }
    }
}

impl PluginWriter for ClusterPluginWriter {
    fn serialize(
        &mut self,
        segment: &mut Segment,
        _doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        let write = segment.open_write(component())?;
        let mut composite = CompositeWrite::wrap(write);

        for field_cfg in &self.config.fields {
            let w = composite.for_field(field_cfg.field);

            let vectors = self.per_field_vectors.get(&field_cfg.field);
            let num_docs = vectors.map_or(0, |v| v.len());

            if (num_docs as u32) < self.config.clustering_threshold || num_docs == 0 {
                serialize_empty(w)?;
                w.flush()?;
                continue;
            }

            let vectors = vectors.unwrap();
            let data = cluster_from_vectors(vectors, &self.config, field_cfg)?;
            serialize_cluster_data(&data, w)?;
            w.flush()?;
        }

        composite.close()?;
        Ok(())
    }

    fn close(self: Box<Self>) -> crate::Result<()> {
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.per_field_vectors
            .values()
            .map(|vecs| vecs.iter().map(|v| v.len() * 4).sum::<usize>())
            .sum::<usize>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct ClusterPluginReader {
    field_readers: HashMap<Field, ClusterFieldReader>,
}

impl ClusterPluginReader {
    pub fn field_reader(&self, field: Field) -> Option<&ClusterFieldReader> {
        self.field_readers.get(&field)
    }
}

impl PluginReader for ClusterPluginReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct ClusterFieldReader {
    centroid_index: Option<CentroidIndex>,
    per_doc_cluster: Vec<u16>,
    cluster_offsets: Vec<u32>,
    cluster_doc_ids: Vec<DocId>,
}

impl ClusterFieldReader {
    fn open(file_slice: FileSlice, metric: Metric) -> crate::Result<Self> {
        let data = file_slice.read_bytes()?;
        if data.len() < 12 {
            return Err(crate::TantivyError::InternalError(
                "cluster field section too short".into(),
            ));
        }

        let num_clusters =
            u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let num_docs =
            u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let dims =
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        if num_clusters == 0 {
            return Ok(Self {
                centroid_index: None,
                per_doc_cluster: Vec::new(),
                cluster_offsets: Vec::new(),
                cluster_doc_ids: Vec::new(),
            });
        }

        let mut offset = 12;

        let ci_len =
            u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
                as usize;
        offset += 4;

        let ci_bytes = &data[offset..offset + ci_len];
        let centroid_ids: Vec<u32> = (0..num_clusters as u32).collect();
        let centroid_index =
            CentroidIndex::load_from_bytes(ci_bytes, centroid_ids, dims, metric)?;
        offset += ci_len;

        let mut per_doc_cluster = Vec::with_capacity(num_docs);
        for _ in 0..num_docs {
            let v = u16::from_le_bytes([data[offset], data[offset + 1]]);
            per_doc_cluster.push(v);
            offset += 2;
        }

        let mut cluster_offsets = Vec::with_capacity(num_clusters + 1);
        for _ in 0..=num_clusters {
            let v = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            cluster_offsets.push(v);
            offset += 4;
        }

        let mut cluster_doc_ids = Vec::with_capacity(num_docs);
        for _ in 0..num_docs {
            let v = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            cluster_doc_ids.push(v as DocId);
            offset += 4;
        }

        Ok(Self {
            centroid_index: Some(centroid_index),
            per_doc_cluster,
            cluster_offsets,
            cluster_doc_ids,
        })
    }

    pub fn is_clustered(&self) -> bool {
        self.centroid_index.is_some()
    }

    pub fn search_centroids(&self, query: &[f32], ef_search: usize) -> Vec<(u32, f32)> {
        match &self.centroid_index {
            Some(ci) => ci.search(query, ef_search),
            None => vec![],
        }
    }

    pub fn cluster_docs(&self, cluster_id: usize) -> &[DocId] {
        let start = self.cluster_offsets[cluster_id] as usize;
        let end = self.cluster_offsets[cluster_id + 1] as usize;
        &self.cluster_doc_ids[start..end]
    }

    pub fn doc_cluster(&self, doc_id: DocId) -> u16 {
        self.per_doc_cluster[doc_id as usize]
    }

    pub fn num_clusters(&self) -> usize {
        self.centroid_index.as_ref().map_or(0, |ci| ci.len())
    }
}
