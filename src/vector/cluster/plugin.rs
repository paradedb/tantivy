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
use crate::postings::{BlockSegmentPostings, PostingsSerializer, SegmentPostings};
use crate::vector::rabitq::math::l2_distance_sqr;
use crate::vector::rabitq::rotation::DynamicRotator;
use crate::vector::rabitq::{self, Metric, RabitqConfig};
use crate::schema::document::{Document, Value};
use crate::schema::{Field, FieldType, IndexRecordOption, Schema};
use crate::{DocId, Segment};

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

struct ClusterPostingsList {
    doc_freq: u32,
    data: Vec<u8>,
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
    per_doc_cluster: Vec<u16>,
    cluster_postings: Vec<ClusterPostingsList>,
    cluster_batch_data: Vec<ClusterBatchData>,
    num_clusters: usize,
    dims: usize,
}

fn sample_indices(total: usize, ratio: f32, cap: usize) -> Vec<usize> {
    let target = ((total as f32 * ratio) as usize).min(cap).max(1).min(total);
    let step = total / target;
    (0..total).step_by(step.max(1)).take(target).collect()
}

fn encode_posting_list(sorted_doc_ids: &[DocId]) -> ClusterPostingsList {
    let doc_freq = sorted_doc_ids.len() as u32;
    let mut buffer = Vec::new();
    if doc_freq > 0 {
        let mut serializer = PostingsSerializer::new(0.0, IndexRecordOption::Basic, None);
        serializer.new_term(doc_freq, false);
        for &doc_id in sorted_doc_ids {
            serializer.write_doc(doc_id, 1);
        }
        serializer
            .close_term(doc_freq, &mut buffer)
            .expect("in-memory serialization should not fail");
    }
    ClusterPostingsList {
        doc_freq,
        data: buffer,
    }
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

    let mut cluster_doc_lists: Vec<Vec<DocId>> = vec![Vec::new(); num_clusters];
    for doc_id in 0..num_docs {
        let c = assignments[doc_id];
        cluster_doc_lists[c].push(doc_id as DocId);
    }

    let cluster_postings: Vec<ClusterPostingsList> = cluster_doc_lists
        .iter()
        .map(|doc_ids| encode_posting_list(doc_ids))
        .collect();

    ClusterData {
        centroid_index,
        per_doc_cluster,
        cluster_postings,
        cluster_batch_data: Vec::new(),
        num_clusters,
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

fn cluster_from_merge(
    sampler: &dyn crate::vector::cluster::sampler::VectorSampler,
    field_config: &ClusterFieldConfig,
    config: &ClusterConfig,
    num_docs: usize,
) -> crate::Result<ClusterResult> {
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

    // Assign docs to clusters using exact distances via sampler
    let mut assignments = vec![0usize; num_docs];
    for doc_id in 0..num_docs {
        let vecs = sampler.sample_vectors(field_config.field, &[doc_id as DocId])?;
        if let Some(vec) = vecs.into_iter().next().flatten() {
            let mut best_cluster = 0;
            let mut best_dist = f32::INFINITY;
            for (ci, centroid) in result.centroids.iter().enumerate() {
                let dist = l2_distance_sqr(&vec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = ci;
                }
            }
            assignments[doc_id] = best_cluster;
        }
    }

    let centroids = result.centroids.clone();
    let data = build_cluster_data(result.centroids, &assignments, num_docs, field_config.metric);

    Ok(ClusterResult {
        data,
        centroids,
        assignments,
    })
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

    // Cluster postings header: (doc_freq: u32, byte_len: u32) per cluster
    for posting in &data.cluster_postings {
        w.write_all(&posting.doc_freq.to_le_bytes())?;
        w.write_all(&(posting.data.len() as u32).to_le_bytes())?;
    }

    // Cluster postings data: concatenated posting list bytes
    for posting in &data.cluster_postings {
        w.write_all(&posting.data)?;
    }

    // Batch data header: (num_batches: u32, num_docs: u32, batch_data_len: u32) per cluster
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

    // Batch data: per cluster
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

        let cluster_write = ctx.target_segment.open_write(component())?;
        let mut cluster_composite = CompositeWrite::wrap(cluster_write);

        let bqvec_write = ctx.target_segment.open_write(bqvec_component())?;
        let mut bqvec_composite = CompositeWrite::wrap(bqvec_write);

        if (num_docs as u32) < self.config.clustering_threshold {
            for field_cfg in &self.config.fields {
                let rabitq_config = RabitqConfig::new(field_cfg.ex_bits + 1);
                let cw = cluster_composite.for_field(field_cfg.field);
                serialize_empty(cw)?;
                cw.flush()?;

                // Re-encode BQ records against zero centroid via sampler
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

            let result = cluster_from_merge(
                sampler.as_ref(),
                field_cfg,
                &self.config,
                num_docs,
            )?;

            let cw = cluster_composite.for_field(field_cfg.field);
            serialize_cluster_data(&result.data, cw)?;
            cw.flush()?;

            // Re-encode BQ records against new cluster centroids
            let bytes_per_record =
                rabitq::bytes_per_record(field_cfg.padded_dims, field_cfg.ex_bits);
            let bw = bqvec_composite.for_field(field_cfg.field);
            bw.write_all(&(bytes_per_record as u32).to_le_bytes())?;
            bw.write_all(&(num_docs as u32).to_le_bytes())?;

            for doc_id in 0..num_docs {
                let vecs =
                    sampler.sample_vectors(field_cfg.field, &[doc_id as DocId])?;
                let vec = vecs.into_iter().next().flatten().unwrap_or_else(|| {
                    vec![0.0f32; field_cfg.dims]
                });
                let centroid = &result.centroids[result.assignments[doc_id]];
                let record = rabitq::encode(
                    &field_cfg.rotator, &rabitq_config, field_cfg.metric, &vec, centroid,
                );
                bw.write_all(&record)?;
            }
            bw.flush()?;
        }

        cluster_composite.close()?;
        bqvec_composite.close()?;
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

            if (num_docs as u32) < self.config.clustering_threshold || num_docs == 0 {
                serialize_empty(cw)?;
                cw.flush()?;
                // Write BQ records against zero centroid
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
            let mut result = cluster_from_vectors(vectors, &self.config, field_cfg)?;

            // Encode BQ records and collect per-cluster batch data
            let dim_bytes = field_cfg.padded_dims / 8;
            let ex_b = rabitq::record::ex_bytes(field_cfg.padded_dims, field_cfg.ex_bits);
            let scalar_off = dim_bytes + ex_b;

            let mut per_cluster_docs: Vec<Vec<(DocId, Vec<u8>)>> =
                vec![Vec::new(); result.data.num_clusters];

            for (doc_id, vec) in vectors.iter().enumerate() {
                let centroid = &result.centroids[result.assignments[doc_id]];
                let record = rabitq::encode(
                    &field_cfg.rotator, &rabitq_config, field_cfg.metric, vec, centroid,
                );
                bw.write_all(&record)?;

                let cluster_id = result.assignments[doc_id];
                per_cluster_docs[cluster_id].push((doc_id as DocId, record));
            }
            bw.flush()?;

            // Build batch data per cluster
            result.data.cluster_batch_data = per_cluster_docs
                .iter()
                .map(|docs| build_cluster_batch_data(docs, dim_bytes, scalar_off))
                .collect();

            serialize_cluster_data(&result.data, cw)?;
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

struct ClusterPostingsMeta {
    doc_freq: u32,
    byte_offset: usize,
    byte_len: usize,
}

pub struct ClusterBatchMeta {
    pub num_batches: u32,
    pub num_docs: u32,
    pub byte_offset: usize,
    pub byte_len: usize,
}

pub struct ClusterFieldReader {
    centroid_index: Option<CentroidIndex>,
    per_doc_cluster: Vec<u16>,
    postings_meta: Vec<ClusterPostingsMeta>,
    postings_data: FileSlice,
    batch_meta: Vec<ClusterBatchMeta>,
    batch_data: FileSlice,
    dim_bytes: usize,
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
                postings_meta: Vec::new(),
                postings_data: FileSlice::empty(),
                batch_meta: Vec::new(),
                batch_data: FileSlice::empty(),
                dim_bytes: 0,
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

        // Read postings header: (doc_freq: u32, byte_len: u32) per cluster
        let mut postings_meta = Vec::with_capacity(num_clusters);
        let mut cumulative_offset = 0usize;
        for _ in 0..num_clusters {
            let doc_freq = u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            ]);
            offset += 4;
            let byte_len = u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            ]) as usize;
            offset += 4;
            postings_meta.push(ClusterPostingsMeta {
                doc_freq,
                byte_offset: cumulative_offset,
                byte_len,
            });
            cumulative_offset += byte_len;
        }

        let postings_end = offset + cumulative_offset;
        let postings_data = file_slice.slice(offset..postings_end);
        offset = postings_end;

        // Read batch data header: (num_batches: u32, num_docs: u32, data_len: u32) per cluster
        let mut batch_meta = Vec::with_capacity(num_clusters);
        let mut batch_cumulative = 0usize;
        if offset + num_clusters * 12 <= data.len() {
            for _ in 0..num_clusters {
                let nb = u32::from_le_bytes([
                    data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                ]);
                offset += 4;
                let nd = u32::from_le_bytes([
                    data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                ]);
                offset += 4;
                let bl = u32::from_le_bytes([
                    data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
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
        }
        let batch_data = if batch_cumulative > 0 {
            file_slice.slice(offset..offset + batch_cumulative)
        } else {
            FileSlice::empty()
        };

        let dim_bytes = dims / 8;

        Ok(Self {
            centroid_index: Some(centroid_index),
            per_doc_cluster,
            postings_meta,
            postings_data,
            batch_meta,
            batch_data,
            dim_bytes,
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

    pub fn cluster_postings(&self, cluster_id: usize) -> crate::Result<SegmentPostings> {
        let meta = &self.postings_meta[cluster_id];
        if meta.doc_freq == 0 {
            return Ok(SegmentPostings::empty());
        }
        let bytes = self
            .postings_data
            .read_bytes_slice(meta.byte_offset..meta.byte_offset + meta.byte_len)?;
        let block_postings = BlockSegmentPostings::open(
            meta.doc_freq,
            bytes,
            IndexRecordOption::Basic,
            IndexRecordOption::Basic,
        )?;
        Ok(SegmentPostings::from_block_postings(block_postings, None))
    }

    pub fn doc_cluster(&self, doc_id: DocId) -> u16 {
        self.per_doc_cluster[doc_id as usize]
    }

    pub fn num_clusters(&self) -> usize {
        self.centroid_index.as_ref().map_or(0, |ci| ci.len())
    }

    pub fn has_batch_data(&self, cluster_id: usize) -> bool {
        cluster_id < self.batch_meta.len() && self.batch_meta[cluster_id].num_batches > 0
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
        // First num_docs × 4 bytes are doc_ids
        let num_docs = meta.num_docs as usize;
        let mut doc_ids = Vec::with_capacity(num_docs);
        for i in 0..num_docs {
            let off = i * 4;
            doc_ids.push(u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]));
        }
        Ok(Some((doc_ids, meta, raw)))
    }

    pub fn dim_bytes(&self) -> usize {
        self.dim_bytes
    }

    /// Adaptive cluster probing: fetch candidate centroids from HNSW, then
    /// lazily select which clusters to actually scan based on a distance-ratio
    /// cutoff. This avoids the two-step pattern of calling `search_centroids`
    /// then `cluster_postings` separately with a hardcoded n_probe.
    ///
    /// The key insight is that the HNSW centroid search is cheap (microseconds
    /// even for thousands of centroids), while scanning a cluster's posting
    /// list is expensive. So we over-fetch centroids up to `max_probe`, but
    /// only materialize posting lists for clusters whose centroid distance
    /// is within `distance_ratio` of the nearest centroid's distance.
    ///
    /// For example, with distance_ratio=1.5: if the nearest centroid is at
    /// distance 10, we stop once we hit a centroid farther than 15 — those
    /// clusters are unlikely to contain true nearest neighbors.
    pub fn probe_clusters(
        &self,
        query: &[f32],
        config: &ProbeConfig,
    ) -> crate::Result<Vec<(u32, SegmentPostings)>> {
        // Over-fetch centroid candidates from HNSW — this is cheap
        let candidates = self.search_centroids(query, config.max_probe);
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        let nearest_dist = candidates[0].1;
        let mut result = Vec::new();
        for (i, &(cluster_id, dist)) in candidates.iter().enumerate() {
            // Always probe at least min_probe clusters, then apply the
            // distance-ratio cutoff: if this centroid is much farther than
            // the nearest, remaining clusters won't help.
            if i >= config.min_probe
                && nearest_dist > 0.0
                && dist / nearest_dist > config.distance_ratio
            {
                break;
            }
            result.push((cluster_id, self.cluster_postings(cluster_id as usize)?));
        }
        Ok(result)
    }
}

pub struct ProbeConfig {
    pub max_probe: usize,
    pub distance_ratio: f32,
    pub min_probe: usize,
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
