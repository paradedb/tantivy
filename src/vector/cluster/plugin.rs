use std::any::Any;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use common::file_slice::FileSlice;

use crate::vector::bqvec::bqvec_component;
use crate::vector::cluster::centroid_index::CentroidIndex;
use crate::vector::cluster::kmeans::{run_kmeans_with_config, KMeansConfig};
use crate::vector::cluster::sampler::VectorSamplerFactory;
use crate::directory::{CompositeFile, CompositeWrite};
use crate::index::SegmentComponent;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use crate::vector::rabitq::math::l2_distance_sqr;
use crate::vector::rabitq::rotation::{DynamicRotator, RotatorType};
use crate::vector::rabitq::{self, Metric, RabitqConfig};
use crate::schema::document::{Document, Value};
use crate::schema::{Field, FieldType, Schema};
use crate::{DocId, Segment};

pub const WINDOW_SIZE: usize = 122_880;

fn component() -> SegmentComponent {
    SegmentComponent::Custom("cluster".to_string())
}

#[derive(Clone)]
pub struct ClusterFieldConfig {
    pub field: Field,
    pub dims: usize,
    pub padded_dims: usize,
    pub ex_bits: usize,
    pub metric: Metric,
    pub rotator: Arc<DynamicRotator>,
    pub rotator_seed: u64,
}

#[derive(Clone, Debug)]
pub struct VectorFieldMeta {
    pub dims: usize,
    pub padded_dims: usize,
    pub ex_bits: usize,
    pub metric: Metric,
    pub rotator_type: RotatorType,
    pub rotator_seed: u64,
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

struct ClusterBatchData {
    doc_ids: Vec<DocId>,
    transposed_codes: Vec<u8>,
    f_add: Vec<f32>,
    f_rescale: Vec<f32>,
    f_error: Vec<f32>,
    num_batches: u32,
}

struct ClusterData {
    centroid_index: CentroidIndex,
    cluster_batch_data: Vec<ClusterBatchData>,
    num_clusters: usize,
    num_docs: usize,
    dims: usize,
}

fn sample_indices(total: usize, ratio: f32, cap: usize) -> Vec<usize> {
    let target = ((total as f32 * ratio) as usize).min(cap).max(1).min(total);
    let step = total / target;
    (0..total).step_by(step.max(1)).take(target).collect()
}

fn build_cluster_data(
    centroids: Vec<Vec<f32>>,
    _assignments: &[usize],
    num_docs: usize,
    metric: Metric,
) -> ClusterData {
    let num_clusters = centroids.len();
    let dims = centroids.first().map_or(0, |v| v.len());
    let centroid_ids: Vec<u32> = (0..num_clusters as u32).collect();
    let centroid_index = CentroidIndex::build(centroids, centroid_ids, metric);

    ClusterData {
        centroid_index,
        cluster_batch_data: Vec::new(),
        num_clusters,
        num_docs,
        dims,
    }
}

struct ClusterResult {
    data: ClusterData,
    centroids: Vec<Vec<f32>>,
    assignments: Vec<usize>,
}

fn cluster_from_vectors(
    vectors: &[Vec<f32>],
    config: &ClusterConfig,
    field_config: &ClusterFieldConfig,
) -> crate::Result<ClusterResult> {
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

    let centroids = result.centroids.clone();
    let data = build_cluster_data(result.centroids, &assignments, num_docs, field_config.metric);

    Ok(ClusterResult {
        data,
        centroids,
        assignments,
    })
}

fn train_centroids(
    sampler: &dyn crate::vector::cluster::sampler::VectorSampler,
    field_config: &ClusterFieldConfig,
    config: &ClusterConfig,
    num_docs: usize,
) -> crate::Result<Vec<Vec<f32>>> {
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
    Ok(result.centroids)
}

fn assign_nearest_centroid(vec: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_dist = f32::INFINITY;
    for (ci, centroid) in centroids.iter().enumerate() {
        let dist = l2_distance_sqr(vec, centroid);
        if dist < best_dist {
            best_dist = dist;
            best = ci;
        }
    }
    best
}

fn build_cluster_batch_data(
    docs: &[(DocId, Vec<u8>)],
    dim_bytes: usize,
    scalar_off: usize,
) -> ClusterBatchData {
    use crate::vector::rabitq::fastscan::{pack_batch_simple, BATCH_SIZE};

    let num_docs = docs.len();
    let num_batches = num_docs.div_ceil(BATCH_SIZE);

    let mut doc_ids = Vec::with_capacity(num_docs);
    let mut transposed_codes = Vec::new();
    let mut f_add = Vec::new();
    let mut f_rescale = Vec::new();
    let mut f_error = Vec::new();

    for chunk in docs.chunks(BATCH_SIZE) {
        let mut codes: Vec<Vec<u8>> = chunk.iter().map(|(_, rec)| rec[..dim_bytes].to_vec()).collect();
        while codes.len() < BATCH_SIZE {
            codes.push(vec![0u8; dim_bytes]);
        }

        let code_refs: Vec<&[u8]> = codes.iter().map(|c| c.as_slice()).collect();
        let mut batch_transposed = vec![0u8; dim_bytes * BATCH_SIZE];
        pack_batch_simple(&code_refs, dim_bytes, &mut batch_transposed);
        transposed_codes.extend_from_slice(&batch_transposed);

        let mut batch_f_add = vec![0.0f32; BATCH_SIZE];
        let mut batch_f_rescale = vec![0.0f32; BATCH_SIZE];
        let mut batch_f_error = vec![0.0f32; BATCH_SIZE];
        for (i, (_, rec)) in chunk.iter().enumerate() {
            let read_f32 = |off: usize| -> f32 {
                f32::from_le_bytes([
                    rec[scalar_off + off],
                    rec[scalar_off + off + 1],
                    rec[scalar_off + off + 2],
                    rec[scalar_off + off + 3],
                ])
            };
            batch_f_add[i] = read_f32(8);
            batch_f_rescale[i] = read_f32(12);
            batch_f_error[i] = read_f32(16);
        }
        f_add.extend_from_slice(&batch_f_add);
        f_rescale.extend_from_slice(&batch_f_rescale);
        f_error.extend_from_slice(&batch_f_error);

        for (did, _) in chunk {
            doc_ids.push(*did);
        }
    }

    ClusterBatchData {
        doc_ids,
        transposed_codes,
        f_add,
        f_rescale,
        f_error,
        num_batches: num_batches as u32,
    }
}

/// Hot section: header + centroids + batch_meta. Read on every query for
/// centroid search and to know batch offsets.
fn serialize_cluster_hot(data: &ClusterData, w: &mut dyn Write) -> crate::Result<()> {
    w.write_all(&(data.num_clusters as u32).to_le_bytes())?;
    w.write_all(&(data.num_docs as u32).to_le_bytes())?;
    w.write_all(&(data.dims as u32).to_le_bytes())?;

    let ci_bytes = data.centroid_index.save_to_bytes()?;
    w.write_all(&(ci_bytes.len() as u32).to_le_bytes())?;
    w.write_all(&ci_bytes)?;

    for batch in &data.cluster_batch_data {
        w.write_all(&batch.num_batches.to_le_bytes())?;
        w.write_all(&(batch.doc_ids.len() as u32).to_le_bytes())?;
        let data_len = batch.transposed_codes.len()
            + batch.f_add.len() * 4
            + batch.f_rescale.len() * 4
            + batch.f_error.len() * 4
            + batch.doc_ids.len() * 4;
        w.write_all(&(data_len as u32).to_le_bytes())?;
    }

    Ok(())
}

/// Cold section: per-cluster batch_data. Read lazily, only for clusters
/// selected by centroid probing during query execution.
fn serialize_cluster_cold(data: &ClusterData, w: &mut dyn Write) -> crate::Result<()> {
    for batch in &data.cluster_batch_data {
        for &did in &batch.doc_ids {
            w.write_all(&did.to_le_bytes())?;
        }
        w.write_all(&batch.transposed_codes)?;
        for &v in &batch.f_add {
            w.write_all(&v.to_le_bytes())?;
        }
        for &v in &batch.f_rescale {
            w.write_all(&v.to_le_bytes())?;
        }
        for &v in &batch.f_error {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn serialize_empty_hot(w: &mut dyn Write) -> crate::Result<()> {
    w.write_all(&0u32.to_le_bytes())?; // num_clusters = 0
    w.write_all(&0u32.to_le_bytes())?; // num_docs = 0
    w.write_all(&0u32.to_le_bytes())?; // dims = 0
    Ok(())
}

fn serialize_field_meta(meta: &VectorFieldMeta, w: &mut dyn Write) -> crate::Result<()> {
    w.write_all(&(meta.dims as u32).to_le_bytes())?;
    w.write_all(&(meta.padded_dims as u32).to_le_bytes())?;
    w.write_all(&(meta.ex_bits as u32).to_le_bytes())?;
    w.write_all(&[match meta.metric {
        Metric::L2 => 0u8,
        Metric::InnerProduct => 1u8,
    }])?;
    w.write_all(&[meta.rotator_type as u8])?;
    w.write_all(&meta.rotator_seed.to_le_bytes())?;
    Ok(())
}

fn deserialize_field_meta(data: &[u8], pos: &mut usize) -> VectorFieldMeta {
    let read_u32 = |p: &mut usize| -> u32 {
        let v = u32::from_le_bytes([data[*p], data[*p+1], data[*p+2], data[*p+3]]);
        *p += 4;
        v
    };
    let dims = read_u32(pos) as usize;
    let padded_dims = read_u32(pos) as usize;
    let ex_bits = read_u32(pos) as usize;
    let metric = if data[*pos] == 0 { Metric::L2 } else { Metric::InnerProduct };
    *pos += 1;
    let rotator_type = RotatorType::from_u8(data[*pos]).unwrap_or(RotatorType::FhtKacRotator);
    *pos += 1;
    let rotator_seed = u64::from_le_bytes([
        data[*pos], data[*pos+1], data[*pos+2], data[*pos+3],
        data[*pos+4], data[*pos+5], data[*pos+6], data[*pos+7],
    ]);
    *pos += 8;
    VectorFieldMeta { dims, padded_dims, ex_bits, metric, rotator_type, rotator_seed }
}

const FIELD_META_SIZE: usize = 4 + 4 + 4 + 1 + 1 + 8; // 22 bytes

impl VectorFieldMeta {
    fn from_config(cfg: &ClusterFieldConfig) -> Self {
        Self {
            dims: cfg.dims,
            padded_dims: cfg.padded_dims,
            ex_bits: cfg.ex_bits,
            metric: cfg.metric,
            rotator_type: cfg.rotator.rotator_type(),
            rotator_seed: cfg.rotator_seed,
        }
    }
}

/// Per-window directory entry size: doc_offset + num_docs + hot_size + cold_size.
const WINDOW_DIR_ENTRY_SIZE: usize = 16;

/// File layout:
///   [u32 num_windows][u32 window_size][22 B field_meta]
///   directory: [u32 doc_offset][u32 num_docs][u32 hot_size][u32 cold_size] × num_windows
///   per window in order: hot bytes, then cold bytes
///
/// Reading this lets the reader pull only each window's small hot section
/// (centroids + batch_meta) up front, leaving the much larger cold section
/// (batch_data) to be sliced lazily per probed cluster.
fn serialize_windowed_field(
    windows: &[(u32, u32, ClusterData)],
    meta: &VectorFieldMeta,
    w: &mut dyn Write,
) -> crate::Result<()> {
    let mut serialized: Vec<(u32, u32, Vec<u8>, Vec<u8>)> = Vec::with_capacity(windows.len());
    for (doc_offset, num_docs, data) in windows {
        let mut hot = Vec::new();
        let mut cold = Vec::new();
        serialize_cluster_hot(data, &mut hot)?;
        serialize_cluster_cold(data, &mut cold)?;
        serialized.push((*doc_offset, *num_docs, hot, cold));
    }

    let num_windows = serialized.len() as u32;
    w.write_all(&num_windows.to_le_bytes())?;
    w.write_all(&(WINDOW_SIZE as u32).to_le_bytes())?;
    serialize_field_meta(meta, w)?;

    for (doc_offset, num_docs, hot, cold) in &serialized {
        w.write_all(&doc_offset.to_le_bytes())?;
        w.write_all(&num_docs.to_le_bytes())?;
        w.write_all(&(hot.len() as u32).to_le_bytes())?;
        w.write_all(&(cold.len() as u32).to_le_bytes())?;
    }

    for (_, _, hot, cold) in &serialized {
        w.write_all(hot)?;
        w.write_all(cold)?;
    }

    Ok(())
}

fn serialize_windowed_empty(
    w: &mut dyn Write,
    num_docs: usize,
    meta: &VectorFieldMeta,
) -> crate::Result<()> {
    let num_windows = num_docs.div_ceil(WINDOW_SIZE).max(1) as u32;
    w.write_all(&num_windows.to_le_bytes())?;
    w.write_all(&(WINDOW_SIZE as u32).to_le_bytes())?;
    serialize_field_meta(meta, w)?;

    // Empty windows have hot_size = 12 (just the cluster header) and cold_size = 0.
    let mut offset = 0u32;
    let mut window_doc_counts = Vec::with_capacity(num_windows as usize);
    for _ in 0..num_windows {
        let window_docs = (num_docs as u32).saturating_sub(offset).min(WINDOW_SIZE as u32);
        window_doc_counts.push(window_docs);
        w.write_all(&offset.to_le_bytes())?;
        w.write_all(&window_docs.to_le_bytes())?;
        w.write_all(&12u32.to_le_bytes())?; // hot_size
        w.write_all(&0u32.to_le_bytes())?; // cold_size
        offset += window_docs;
    }

    for _ in 0..num_windows {
        serialize_empty_hot(w)?;
    }

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

        let cluster_write = ctx.target_segment.open_write(component())?;
        let mut cluster_composite = CompositeWrite::wrap(cluster_write);

        let bqvec_write = ctx.target_segment.open_write(bqvec_component())?;
        let mut bqvec_composite = CompositeWrite::wrap(bqvec_write);

        if (num_docs as u32) < self.config.clustering_threshold {
            for field_cfg in &self.config.fields {
                let rabitq_config = RabitqConfig::new(field_cfg.ex_bits + 1);
                let cw = cluster_composite.for_field(field_cfg.field);
                let meta = VectorFieldMeta::from_config(field_cfg);
                serialize_windowed_empty(cw, num_docs, &meta)?;
                cw.flush()?;

                let bytes_per_record =
                    rabitq::bytes_per_record(field_cfg.padded_dims, field_cfg.ex_bits);
                let bw = bqvec_composite.for_field(field_cfg.field);
                bw.write_all(&(bytes_per_record as u32).to_le_bytes())?;
                bw.write_all(&(num_docs as u32).to_le_bytes())?;

                let sampler = self.config.sampler_factory.create_sampler(
                    ctx.readers,
                    ctx.doc_id_mapping,
                )?;
                let zero = vec![0.0f32; field_cfg.dims];
                for doc_id in 0..num_docs {
                    let vecs =
                        sampler.sample_vectors(field_cfg.field, &[doc_id as DocId])?;
                    let vec = vecs.into_iter().next().flatten().unwrap_or_else(|| {
                        vec![0.0f32; field_cfg.dims]
                    });
                    let record = rabitq::encode(
                        &field_cfg.rotator, &rabitq_config, field_cfg.metric, &vec, &zero,
                    );
                    bw.write_all(&record)?;
                }
                bw.flush()?;
            }
            cluster_composite.close()?;
            bqvec_composite.close()?;
            return Ok(());
        }

        let sampler = self
            .config
            .sampler_factory
            .create_sampler(ctx.readers, ctx.doc_id_mapping)?;

        for field_cfg in &self.config.fields {
            let rabitq_config = RabitqConfig::new(field_cfg.ex_bits + 1);
            let dim_bytes = field_cfg.padded_dims / 8;
            let ex_b = rabitq::record::ex_bytes(field_cfg.padded_dims, field_cfg.ex_bits);
            let scalar_off = dim_bytes + ex_b;

            let bytes_per_record =
                rabitq::bytes_per_record(field_cfg.padded_dims, field_cfg.ex_bits);
            let bw = bqvec_composite.for_field(field_cfg.field);
            bw.write_all(&(bytes_per_record as u32).to_le_bytes())?;
            bw.write_all(&(num_docs as u32).to_le_bytes())?;

            let num_windows = num_docs.div_ceil(WINDOW_SIZE);
            let mut windows: Vec<(u32, u32, ClusterData)> = Vec::with_capacity(num_windows);

            for win_idx in 0..num_windows {
                let win_start = win_idx * WINDOW_SIZE;
                let win_end = (win_start + WINDOW_SIZE).min(num_docs);
                let win_num_docs = win_end - win_start;

                let win_sampler = WindowSampler {
                    inner: sampler.as_ref(),
                    doc_offset: win_start,
                };

                let centroids = train_centroids(
                    &win_sampler,
                    field_cfg,
                    &self.config,
                    win_num_docs,
                )?;
                let num_clusters = centroids.len();

                let mut assignments = vec![0usize; win_num_docs];
                let mut per_cluster_docs: Vec<Vec<(DocId, Vec<u8>)>> =
                    vec![Vec::new(); num_clusters];
                for local_doc_id in 0..win_num_docs {
                    let global_doc_id = win_start + local_doc_id;
                    let vecs =
                        sampler.sample_vectors(field_cfg.field, &[global_doc_id as DocId])?;
                    let vec = vecs.into_iter().next().flatten().unwrap_or_else(|| {
                        vec![0.0f32; field_cfg.dims]
                    });
                    let cluster_id = assign_nearest_centroid(&vec, &centroids);
                    assignments[local_doc_id] = cluster_id;
                    let centroid = &centroids[cluster_id];
                    let record = rabitq::encode(
                        &field_cfg.rotator, &rabitq_config, field_cfg.metric, &vec, centroid,
                    );
                    bw.write_all(&record)?;
                    per_cluster_docs[cluster_id].push((local_doc_id as DocId, record));
                }

                let mut data = build_cluster_data(
                    centroids, &assignments, win_num_docs, field_cfg.metric,
                );
                data.cluster_batch_data = per_cluster_docs
                    .iter()
                    .map(|docs| build_cluster_batch_data(docs, dim_bytes, scalar_off))
                    .collect();

                windows.push((win_start as u32, win_num_docs as u32, data));
            }

            bw.flush()?;

            let cw = cluster_composite.for_field(field_cfg.field);
            let meta = VectorFieldMeta::from_config(field_cfg);
            serialize_windowed_field(&windows, &meta, cw)?;
            cw.flush()?;
        }

        cluster_composite.close()?;
        bqvec_composite.close()?;
        Ok(())
    }
}

struct WindowSampler<'a> {
    inner: &'a (dyn crate::vector::cluster::sampler::VectorSampler + 'a),
    doc_offset: usize,
}

unsafe impl<'a> Send for WindowSampler<'a> {}
unsafe impl<'a> Sync for WindowSampler<'a> {}

impl<'a> crate::vector::cluster::sampler::VectorSampler for WindowSampler<'a> {
    fn sample_vectors(
        &self,
        field: Field,
        doc_ids: &[DocId],
    ) -> crate::Result<Vec<Option<Vec<f32>>>> {
        let mapped: Vec<DocId> = doc_ids
            .iter()
            .map(|&id| (id as usize + self.doc_offset) as DocId)
            .collect();
        self.inner.sample_vectors(field, &mapped)
    }

    fn dims(&self, field: Field) -> usize {
        self.inner.dims(field)
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
        let cluster_write = segment.open_write(component())?;
        let mut cluster_composite = CompositeWrite::wrap(cluster_write);

        let bqvec_write = segment.open_write(bqvec_component())?;
        let mut bqvec_composite = CompositeWrite::wrap(bqvec_write);

        for field_cfg in &self.config.fields {
            let rabitq_config = RabitqConfig::new(field_cfg.ex_bits + 1);
            let vectors = self.per_field_vectors.get(&field_cfg.field);
            let num_docs = vectors.map_or(0, |v| v.len());

            let cw = cluster_composite.for_field(field_cfg.field);

            let bytes_per_record = rabitq::bytes_per_record(field_cfg.padded_dims, field_cfg.ex_bits);
            let bw = bqvec_composite.for_field(field_cfg.field);
            bw.write_all(&(bytes_per_record as u32).to_le_bytes())?;
            bw.write_all(&(num_docs as u32).to_le_bytes())?;

            let meta = VectorFieldMeta::from_config(field_cfg);
            if (num_docs as u32) < self.config.clustering_threshold || num_docs == 0 {
                serialize_windowed_empty(cw, num_docs, &meta)?;
                cw.flush()?;
                let zero = vec![0.0f32; field_cfg.dims];
                if let Some(vectors) = vectors {
                    for vec in vectors {
                        let record = rabitq::encode(
                            &field_cfg.rotator, &rabitq_config, field_cfg.metric, vec, &zero,
                        );
                        bw.write_all(&record)?;
                    }
                }
                bw.flush()?;
                continue;
            }

            let vectors = vectors.unwrap();
            let dim_bytes = field_cfg.padded_dims / 8;
            let ex_b = rabitq::record::ex_bytes(field_cfg.padded_dims, field_cfg.ex_bits);
            let scalar_off = dim_bytes + ex_b;

            let num_windows = num_docs.div_ceil(WINDOW_SIZE);
            let mut windows: Vec<(u32, u32, ClusterData)> = Vec::with_capacity(num_windows);

            for win_idx in 0..num_windows {
                let win_start = win_idx * WINDOW_SIZE;
                let win_end = (win_start + WINDOW_SIZE).min(num_docs);
                let win_vectors = &vectors[win_start..win_end];
                let win_num_docs = win_vectors.len();

                let mut result = cluster_from_vectors(win_vectors, &self.config, field_cfg)?;

                let mut per_cluster_docs: Vec<Vec<(DocId, Vec<u8>)>> =
                    vec![Vec::new(); result.data.num_clusters];

                for (local_doc_id, vec) in win_vectors.iter().enumerate() {
                    let centroid = &result.centroids[result.assignments[local_doc_id]];
                    let record = rabitq::encode(
                        &field_cfg.rotator, &rabitq_config, field_cfg.metric, vec, centroid,
                    );
                    bw.write_all(&record)?;

                    let cluster_id = result.assignments[local_doc_id];
                    per_cluster_docs[cluster_id].push((local_doc_id as DocId, record));
                }

                result.data.cluster_batch_data = per_cluster_docs
                    .iter()
                    .map(|docs| build_cluster_batch_data(docs, dim_bytes, scalar_off))
                    .collect();

                windows.push((win_start as u32, win_num_docs as u32, result.data));
            }

            bw.flush()?;

            serialize_windowed_field(&windows, &meta, cw)?;
            cw.flush()?;
        }

        cluster_composite.close()?;
        bqvec_composite.close()?;
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

pub struct ClusterBatchMeta {
    pub num_batches: u32,
    pub num_docs: u32,
    pub byte_offset: usize,
    pub byte_len: usize,
}

pub struct WindowReader {
    centroid_index: Option<CentroidIndex>,
    batch_meta: Vec<ClusterBatchMeta>,
    batch_data: FileSlice,
    pub doc_offset: u32,
    pub num_docs: u32,
    dims: usize,
}

impl WindowReader {
    /// Parse a window from its hot bytes (centroids + batch_meta) plus a
    /// `FileSlice` for its cold section (batch_data, read lazily).
    fn open(
        hot: &[u8],
        cold_slice: FileSlice,
        doc_offset: u32,
        num_docs_in_window: u32,
        metric: Metric,
    ) -> crate::Result<Self> {
        if hot.len() < 12 {
            return Err(crate::TantivyError::InternalError(
                "window hot section too short".into(),
            ));
        }

        let num_clusters =
            u32::from_le_bytes([hot[0], hot[1], hot[2], hot[3]]) as usize;
        let _num_docs_field =
            u32::from_le_bytes([hot[4], hot[5], hot[6], hot[7]]) as usize;
        let dims =
            u32::from_le_bytes([hot[8], hot[9], hot[10], hot[11]]) as usize;

        if num_clusters == 0 {
            return Ok(Self {
                centroid_index: None,
                batch_meta: Vec::new(),
                batch_data: FileSlice::empty(),
                doc_offset,
                num_docs: num_docs_in_window,
                dims,
            });
        }

        let mut offset = 12;
        let ci_len =
            u32::from_le_bytes([hot[offset], hot[offset + 1], hot[offset + 2], hot[offset + 3]])
                as usize;
        offset += 4;

        let ci_bytes = &hot[offset..offset + ci_len];
        let centroid_ids: Vec<u32> = (0..num_clusters as u32).collect();
        let centroid_index =
            CentroidIndex::load_from_bytes(ci_bytes, centroid_ids, dims, metric)?;
        offset += ci_len;

        let mut batch_meta = Vec::with_capacity(num_clusters);
        let mut batch_cumulative = 0usize;
        for _ in 0..num_clusters {
            let nb = u32::from_le_bytes([
                hot[offset], hot[offset + 1], hot[offset + 2], hot[offset + 3],
            ]);
            offset += 4;
            let nd = u32::from_le_bytes([
                hot[offset], hot[offset + 1], hot[offset + 2], hot[offset + 3],
            ]);
            offset += 4;
            let bl = u32::from_le_bytes([
                hot[offset], hot[offset + 1], hot[offset + 2], hot[offset + 3],
            ]) as usize;
            offset += 4;
            batch_meta.push(ClusterBatchMeta {
                num_batches: nb,
                num_docs: nd,
                byte_offset: batch_cumulative,
                byte_len: bl,
            });
            batch_cumulative += bl;
        }

        Ok(Self {
            centroid_index: Some(centroid_index),
            batch_meta,
            batch_data: cold_slice,
            doc_offset,
            num_docs: num_docs_in_window,
            dims,
        })
    }

    pub fn is_clustered(&self) -> bool {
        self.centroid_index.is_some()
    }

    pub fn num_clusters(&self) -> usize {
        self.centroid_index.as_ref().map_or(0, |ci| ci.len())
    }

    pub fn search_centroids(&self, query: &[f32], ef_search: usize) -> Vec<(u32, f32)> {
        match &self.centroid_index {
            Some(ci) => ci.search(query, ef_search),
            None => vec![],
        }
    }

    pub fn has_batch_data(&self, cluster_id: usize) -> bool {
        cluster_id < self.batch_meta.len() && self.batch_meta[cluster_id].num_batches > 0
    }

    /// Read just the doc_id prefix of a cluster's batch_data (small, ~num_docs * 4 bytes).
    /// Cheap pre-check for filter-bitset overlap without paying the full ~63KB read.
    pub fn cluster_doc_ids(&self, cluster_id: usize) -> crate::Result<Option<Vec<DocId>>> {
        if cluster_id >= self.batch_meta.len() {
            return Ok(None);
        }
        let meta = &self.batch_meta[cluster_id];
        if meta.num_batches == 0 {
            return Ok(None);
        }
        let num_docs = meta.num_docs as usize;
        let prefix = self
            .batch_data
            .read_bytes_slice(meta.byte_offset..meta.byte_offset + num_docs * 4)?;
        let mut doc_ids = Vec::with_capacity(num_docs);
        for i in 0..num_docs {
            let off = i * 4;
            doc_ids.push(u32::from_le_bytes([
                prefix[off], prefix[off + 1], prefix[off + 2], prefix[off + 3],
            ]));
        }
        Ok(Some(doc_ids))
    }

    pub fn cluster_batch_raw(
        &self,
        cluster_id: usize,
    ) -> crate::Result<Option<(Vec<DocId>, &ClusterBatchMeta, common::OwnedBytes)>> {
        if cluster_id >= self.batch_meta.len() {
            return Ok(None);
        }
        let meta = &self.batch_meta[cluster_id];
        if meta.num_batches == 0 {
            return Ok(None);
        }
        let raw = self
            .batch_data
            .read_bytes_slice(meta.byte_offset..meta.byte_offset + meta.byte_len)?;
        let num_docs = meta.num_docs as usize;
        let mut doc_ids = Vec::with_capacity(num_docs);
        for i in 0..num_docs {
            let off = i * 4;
            doc_ids.push(u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]));
        }
        Ok(Some((doc_ids, meta, raw)))
    }

}

pub struct ClusterFieldReader {
    windows: Vec<WindowReader>,
    dim_bytes: usize,
    field_meta: Option<VectorFieldMeta>,
}

impl ClusterFieldReader {
    fn open(file_slice: FileSlice, metric: Metric) -> crate::Result<Self> {
        // Header: u32 num_windows + u32 window_size + 22 B field_meta = 30 B.
        const HEADER_SIZE: usize = 8 + FIELD_META_SIZE;
        let header = file_slice.read_bytes_slice(0..HEADER_SIZE)?;

        let num_windows =
            u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let _window_size =
            u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let mut field_meta_pos = 8;
        let field_meta = Some(deserialize_field_meta(&header, &mut field_meta_pos));

        // Directory: 16 B per window.
        let dir_start = HEADER_SIZE;
        let dir_end = dir_start + num_windows * WINDOW_DIR_ENTRY_SIZE;
        let dir = file_slice.read_bytes_slice(dir_start..dir_end)?;

        let mut entries: Vec<(u32, u32, usize, usize, usize)> = Vec::with_capacity(num_windows);
        let mut cumulative = dir_end;
        for i in 0..num_windows {
            let off = i * WINDOW_DIR_ENTRY_SIZE;
            let win_doc_offset =
                u32::from_le_bytes([dir[off], dir[off + 1], dir[off + 2], dir[off + 3]]);
            let win_num_docs = u32::from_le_bytes([
                dir[off + 4], dir[off + 5], dir[off + 6], dir[off + 7],
            ]);
            let hot_size = u32::from_le_bytes([
                dir[off + 8], dir[off + 9], dir[off + 10], dir[off + 11],
            ]) as usize;
            let cold_size = u32::from_le_bytes([
                dir[off + 12], dir[off + 13], dir[off + 14], dir[off + 15],
            ]) as usize;
            entries.push((win_doc_offset, win_num_docs, cumulative, hot_size, cold_size));
            cumulative += hot_size + cold_size;
        }

        let mut windows = Vec::with_capacity(num_windows);
        let mut dim_bytes = 0usize;
        for (win_doc_offset, win_num_docs, win_off, hot_size, cold_size) in entries {
            let hot = file_slice.read_bytes_slice(win_off..win_off + hot_size)?;
            let cold_slice = if cold_size > 0 {
                file_slice.slice((win_off + hot_size)..(win_off + hot_size + cold_size))
            } else {
                FileSlice::empty()
            };
            let win_reader =
                WindowReader::open(&hot, cold_slice, win_doc_offset, win_num_docs, metric)?;
            if win_reader.dims > 0 {
                dim_bytes = win_reader.dims / 8;
            }
            windows.push(win_reader);
        }

        Ok(Self { windows, dim_bytes, field_meta })
    }

    pub fn is_clustered(&self) -> bool {
        self.windows.iter().any(|w| w.is_clustered())
    }

    pub fn num_clusters(&self) -> usize {
        self.windows.iter().map(|w| w.num_clusters()).sum()
    }

    pub fn num_windows(&self) -> usize {
        self.windows.len()
    }

    pub fn window_reader(&self, idx: usize) -> &WindowReader {
        &self.windows[idx]
    }

    pub fn search_centroids(&self, query: &[f32], ef_search: usize) -> Vec<(u32, f32)> {
        if self.windows.len() == 1 {
            return self.windows[0].search_centroids(query, ef_search);
        }
        let mut all = Vec::new();
        for win in &self.windows {
            all.extend(win.search_centroids(query, ef_search));
        }
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(ef_search);
        all
    }

    pub fn has_batch_data(&self, cluster_id: usize) -> bool {
        if self.windows.len() == 1 {
            return self.windows[0].has_batch_data(cluster_id);
        }
        let mut remaining = cluster_id;
        for win in &self.windows {
            let nc = win.num_clusters();
            if remaining < nc {
                return win.has_batch_data(remaining);
            }
            remaining -= nc;
        }
        false
    }

    pub fn cluster_batch_raw(
        &self,
        cluster_id: usize,
    ) -> crate::Result<Option<(Vec<DocId>, &ClusterBatchMeta, common::OwnedBytes)>> {
        if self.windows.len() == 1 {
            return self.windows[0].cluster_batch_raw(cluster_id);
        }
        let mut remaining = cluster_id;
        for win in &self.windows {
            let nc = win.num_clusters();
            if remaining < nc {
                return win.cluster_batch_raw(remaining);
            }
            remaining -= nc;
        }
        Ok(None)
    }

    pub fn dim_bytes(&self) -> usize {
        self.dim_bytes
    }

    pub fn field_meta(&self) -> Option<&VectorFieldMeta> {
        self.field_meta.as_ref()
    }

}

#[derive(Clone)]
pub struct ProbeConfig {
    pub max_probe: usize,
    pub distance_ratio: f32,
    pub min_probe: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            max_probe: 50,
            distance_ratio: 1000.0,
            min_probe: 1,
        }
    }
}

impl ProbeConfig {
    pub fn new(max_probe: usize, distance_ratio: f32) -> Self {
        Self {
            max_probe,
            distance_ratio,
            min_probe: 1,
        }
    }

    pub fn with_min_probe(mut self, min_probe: usize) -> Self {
        self.min_probe = min_probe;
        self
    }
}
