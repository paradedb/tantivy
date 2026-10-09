use std::any::Any;
use std::io::Write;
use std::sync::Arc;

use cascade::{
    encode_batch_in_place_with_workspace, BatchEncodeWorkspace, PreparedCentroidWorkspace,
    QueryRotationPlan,
};
use common::BinarySerializable;

use super::assignments::{BatchAssigner, ASSIGN_BATCH_SIZE};
use super::centroid_index::CentroidIndex;
use super::{decode_row, decode_row_append, IvfIndex, CENTROIDS_EXT};
use crate::directory::CompositeWrite;
use crate::index::{CentroidIndexMeta, SegmentComponent};
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::indexer::segment_updater::CancelSentinel;
use crate::plugin::{PluginMergeContext, PluginWriter};
use crate::schema::document::ErasedDocument;
use crate::schema::{Field, FieldType, Schema, VectorDType, VectorOptions};
use crate::vector::blocks::{block_len, column_range, pad, write_metadata, BlockDirectory};
use crate::vector::buffer::VectorBuffer;
use crate::vector::flat::IdMap;
use crate::vector::header::{
    write_vector_header, CentroidSlot, VectorEntry, VectorFileVersion, HEADER_LEN,
};
use crate::vector::metadata::{SlotType, VectorColMetadata};
use crate::vector::{
    residual_norm, BoundKind, BoundsBuilder, VectorQuantizationConfig, ENTRY_ALIGN, VEC_EXT,
};
use crate::{DocId, Segment, TantivyError};

pub(crate) struct IvfVecWriter {
    buffer: VectorBuffer,
}

impl IvfVecWriter {
    pub(crate) fn for_schema(schema: &Schema) -> Self {
        Self {
            buffer: VectorBuffer::for_schema(schema),
        }
    }
}

impl PluginWriter for IvfVecWriter {
    fn add_document(
        &mut self,
        doc_id: DocId,
        doc: &dyn ErasedDocument,
        schema: &Schema,
    ) -> crate::Result<()> {
        self.buffer.add_document(doc_id, doc, schema)
    }

    fn serialize(
        self: Box<Self>,
        segment: &Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        let VectorBuffer { fields, num_docs } = self.buffer;
        if fields.is_empty() {
            return Ok(());
        }
        let mut writer = SharedSegmentWriter::new(segment)?;
        for (field, buf) in fields {
            let stride = buf.opts.bytes_per_vector();
            let rows: Box<dyn Iterator<Item = (DocId, usize, usize, Option<usize>)> + '_> =
                if let Some(map) = doc_id_map {
                    Box::new(map.iter_source_doc_ids().enumerate().filter_map(
                        |(target_doc_id, source_doc_id)| {
                            buf.present_doc_ids
                                .binary_search(&source_doc_id)
                                .ok()
                                .map(|row| (target_doc_id as DocId, 0, row, None))
                        },
                    ))
                } else {
                    Box::new(
                        buf.present_doc_ids
                            .iter()
                            .enumerate()
                            .map(|(row, &doc)| (doc, 0, row, None)),
                    )
                };
            writer.write_field(
                field,
                num_docs,
                rows,
                BoundsBuilder::new(writer.routers[&field].num_clusters()),
                |_, row| Ok(&buf.row_bytes[row * stride..(row + 1) * stride]),
                &|| false,
            )?;
        }
        writer.finish()
    }

    fn mem_usage(&self) -> usize {
        self.buffer.mem_usage()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

struct AssignedVector {
    cluster: usize,
    new_assignment: bool,
    target_doc_id: DocId,
    source_segment_ord: usize,
    source_row: usize,
}

fn write_u16_run(writer: &mut impl Write, values: &[u16]) -> std::io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_f32_run(writer: &mut impl Write, values: &[f32]) -> std::io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

struct WrittenField {
    offsets: Vec<u64>,
    doc_ids: Vec<DocId>,
    bounds: Vec<f32>,
}

struct ClusteredField<'a> {
    field: Field,
    opts: &'a VectorOptions,
    quantization: Option<&'a VectorQuantizationConfig>,
    num_docs: DocId,
    centroid_bytes: &'a [u8],
}

impl ClusteredField<'_> {
    fn write<R: AsRef<[u8]>>(
        &self,
        vec_write: &mut CompositeWrite,
        mut assigned_vectors: Vec<AssignedVector>,
        mut read_row: impl FnMut(&AssignedVector) -> crate::Result<R>,
        cancel: &dyn CancelSentinel,
        mut bounds_builder: BoundsBuilder,
    ) -> crate::Result<WrittenField> {
        let Self {
            field,
            opts,
            quantization,
            num_docs: num_target_docs,
            centroid_bytes,
        } = *self;
        let centroid_stride = opts.bytes_per_vector();
        let num_centroids = centroid_bytes.len() / centroid_stride;
        let residual: fn(&[u8], &[f32]) -> f32 = match opts.dtype() {
            VectorDType::F32 => residual_norm::<f32>,
        };
        let mut cluster_counts = vec![0usize; num_centroids];
        for assigned_vector in &assigned_vectors {
            if assigned_vector.target_doc_id >= num_target_docs {
                return Err(TantivyError::InvalidArgument(
                    "vector document id exceeds max_doc".into(),
                ));
            }
            cluster_counts[assigned_vector.cluster] += 1;
        }

        assigned_vectors.sort_unstable_by_key(|vector| (vector.cluster, vector.target_doc_id));

        let mut cluster_offsets: Vec<u64> = Vec::with_capacity(num_centroids + 1);
        let mut next_offset = 0u64;
        cluster_offsets.push(next_offset);
        for cluster_count in cluster_counts {
            next_offset += cluster_count as u64;
            cluster_offsets.push(next_offset);
        }

        let doc_ids = assigned_vectors
            .iter()
            .map(|vector| vector.target_doc_id)
            .collect();

        // Data entry: metadata followed by aligned cluster blocks.
        let meta = VectorColMetadata::build_ivf(opts, quantization)?;
        let slots = meta.slots();
        vec_write.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
        let data = vec_write.for_field_with_idx(field, VectorEntry::Data.index());
        let entry_start = data.written_bytes();
        let mut pos = write_metadata(data, &meta)?;
        let mut directory = BlockDirectory::new(data.written_bytes() - entry_start);
        let (specs, grids) = meta.runtime();
        let row_bytes = opts.bytes_per_vector();
        let fixed_scratch = row_bytes + opts.dim() + opts.dim().div_ceil(64) * 8;
        let per_row_scratch = 2 * row_bytes + quantization.map_or(0, |c| c.bytes_per_row()) + 16;
        let tile_rows = (1usize << 20)
            .saturating_sub(fixed_scratch)
            .checked_div(per_row_scratch)
            .unwrap_or(0)
            .max(1);
        let mut centroid_workspace = quantization.map(|_| {
            PreparedCentroidWorkspace::new(Arc::new(QueryRotationPlan::new(opts.dim(), &specs)))
        });
        let mut encode_workspace = quantization
            .map(|_| BatchEncodeWorkspace::with_capacity(opts.dim(), tile_rows, &specs));
        let mut bufs: Vec<Vec<u8>> = slots.iter().map(|_| Vec::new()).collect();
        let mut batch_values = Vec::with_capacity(tile_rows * opts.dim());
        let mut centroid = Vec::with_capacity(opts.dim());
        for (cluster, offsets) in cluster_offsets.windows(2).enumerate() {
            let start = offsets[0] as usize;
            let end = offsets[1] as usize;
            let n = end - start;
            if n == 0 {
                directory.push(data.written_bytes() - entry_start, offsets[1] as u32);
                continue;
            }
            let block_start = pos;
            assert_eq!(data.written_bytes() - entry_start, block_start as u64);
            centroid.clear();
            decode_row_append::<f32>(
                &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
                opts.dim(),
                &mut centroid,
            )?;
            let prepared = centroid_workspace
                .as_mut()
                .map(|workspace| workspace.prepare(&centroid));
            // Rows stream directly to disk; only encoded columns need cluster buffers.
            for tile in assigned_vectors[start..end].chunks(tile_rows) {
                if cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }
                batch_values.clear();
                for assigned in tile {
                    let bytes = read_row(assigned)?;
                    let row = bytes.as_ref();
                    data.write_all(row)?;
                    pos += row.len();
                    if assigned.new_assignment {
                        bounds_builder.add_native(cluster, residual(row, &centroid));
                    }
                    if quantization.is_some() {
                        decode_row_append::<f32>(row, opts.dim(), &mut batch_values)?;
                    }
                }
                if let Some(prepared) = prepared.as_ref() {
                    let batch = encode_batch_in_place_with_workspace(
                        &mut batch_values,
                        tile.len(),
                        prepared,
                        &specs,
                        &grids,
                        encode_workspace.as_mut().unwrap(),
                        opts.metric() == crate::schema::Metric::L2,
                    );
                    for (idx, slot) in slots.iter().enumerate().skip(1) {
                        match &slot.slot_type {
                            SlotType::ResidualNorms => {
                                write_f32_run(&mut bufs[idx], &batch.residual_norms_squared)?
                            }
                            SlotType::QuantLayerCodes { layer, .. } => {
                                bufs[idx].extend_from_slice(&batch.layers[*layer as usize].codes)
                            }
                            SlotType::QuantLayerScales { layer } => write_f32_run(
                                &mut bufs[idx],
                                &batch.layers[*layer as usize].scales,
                            )?,
                            SlotType::QuantLayerGammas { layer } => write_u16_run(
                                &mut bufs[idx],
                                &batch.layers[*layer as usize].gammas,
                            )?,
                            SlotType::QuantLayerErrors { layer } => write_u16_run(
                                &mut bufs[idx],
                                &batch.layers[*layer as usize].corrected_error_ratios,
                            )?,
                            SlotType::QuantLayerConstants { layer } => write_f32_run(
                                &mut bufs[idx],
                                &batch.layers[*layer as usize].constants,
                            )?,
                            SlotType::Rows { .. } => unreachable!(),
                        }
                    }
                }
            }
            // Column flushes poll cancellation and retain capacity for the next cluster.
            for idx in 1..slots.len() {
                let col = column_range(&slots, n, idx);
                let padding = block_start + col.start - pos;
                pad(data, padding)?;
                assert_eq!(bufs[idx].len(), col.len());
                for chunk in bufs[idx].chunks(1 << 20) {
                    if cancel.wants_cancel() {
                        return Err(TantivyError::Cancelled);
                    }
                    data.write_all(chunk)?;
                }
                bufs[idx].clear();
                pos = block_start + col.end;
            }
            let padding = block_start + block_len(&slots, n) - pos;
            pad(data, padding)?;
            pos += padding;
            assert_eq!(data.written_bytes() - entry_start, pos as u64);
            directory.push(data.written_bytes() - entry_start, offsets[1] as u32);
        }
        directory.finish(data)?;
        assert_eq!(
            (data.written_bytes() - entry_start) as usize % ENTRY_ALIGN,
            0
        );
        data.flush()?;
        Ok(WrittenField {
            offsets: cluster_offsets,
            doc_ids,
            bounds: bounds_builder.finish(),
        })
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SharedSegmentMeta {
    pub centroid_index: CentroidIndexMeta,
    pub num_docs: u32,
}

struct SharedSegmentWriter<'a> {
    segment: &'a Segment,
    routers: Arc<CentroidIndex>,
    vectors: CompositeWrite,
    centroids: CompositeWrite,
    id_maps: Vec<(Field, Vec<DocId>)>,
}

impl<'a> SharedSegmentWriter<'a> {
    fn new(segment: &'a Segment) -> crate::Result<Self> {
        let routers = segment.index().cached_centroid_index()?.ok_or_else(|| {
            TantivyError::InvalidArgument("shared segment requires a centroid index".into())
        })?;
        let mut vectors = segment.open_write(SegmentComponent::Custom(VEC_EXT.into()))?;
        write_vector_header(&mut vectors)?;
        let mut centroids = segment.open_write(SegmentComponent::Custom(CENTROIDS_EXT.into()))?;
        VectorFileVersion::V5.serialize(&mut centroids)?;
        Ok(Self {
            segment,
            routers,
            vectors: CompositeWrite::wrap(vectors),
            centroids: CompositeWrite::wrap(centroids),
            id_maps: Vec::new(),
        })
    }

    fn write_field<R: AsRef<[u8]>>(
        &mut self,
        field: Field,
        num_docs: DocId,
        rows: impl IntoIterator<Item = (DocId, usize, usize, Option<usize>)>,
        mut bounds: BoundsBuilder,
        mut read_row: impl FnMut(usize, usize) -> crate::Result<R>,
        cancel: &dyn CancelSentinel,
    ) -> crate::Result<()> {
        let schema = self.segment.schema();
        let entry = schema.get_field_entry(field);
        let FieldType::Vector(opts) = entry.field_type() else {
            return Err(TantivyError::InvalidArgument(
                "expected a vector field".into(),
            ));
        };
        let router = &self.routers[&field];
        let centroid_bytes = router.centroid_bytes()?;
        let stride = opts.bytes_per_vector();
        let mut assigned = Vec::new();
        for (target_doc_id, source_segment_ord, source_row, existing_cluster) in rows {
            if cancel.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            assigned.push(AssignedVector {
                cluster: existing_cluster.unwrap_or_default(),
                new_assignment: existing_cluster.is_none(),
                target_doc_id,
                source_segment_ord,
                source_row,
            });
        }
        if assigned.iter().any(|row| row.new_assignment) {
            let mut assigner = BatchAssigner::new(
                decode_row::<f32>(&centroid_bytes, router.num_clusters() * opts.dim())?,
                opts,
            );
            let mut values = Vec::with_capacity(ASSIGN_BATCH_SIZE * opts.dim());
            for batch in assigned.chunks_mut(ASSIGN_BATCH_SIZE) {
                if cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }
                values.clear();
                for row in batch.iter().filter(|row| row.new_assignment) {
                    let bytes = read_row(row.source_segment_ord, row.source_row)?;
                    decode_row_append::<f32>(bytes.as_ref(), opts.dim(), &mut values)?;
                }
                for (row, cluster) in batch
                    .iter_mut()
                    .filter(|row| row.new_assignment)
                    .zip(assigner.assign(&values))
                {
                    row.cluster = cluster;
                }
            }
        }
        let num_present = assigned.len() as u32;
        if opts.needs_normalization() {
            for (cluster, centroid) in centroid_bytes.chunks_exact(stride).enumerate() {
                if centroid
                    .chunks_exact(4)
                    .all(|v| f32::from_le_bytes(v.try_into().unwrap()) == 0.0)
                {
                    bounds.saturate(cluster);
                }
            }
        }
        let quantization = self
            .segment
            .index()
            .settings()
            .vector_quantization
            .iter()
            .find(|config| config.field == entry.name());
        let written = ClusteredField {
            field,
            opts,
            quantization,
            num_docs,
            centroid_bytes: &centroid_bytes,
        }
        .write(
            &mut self.vectors,
            assigned,
            |row| read_row(row.source_segment_ord, row.source_row),
            cancel,
            bounds,
        )?;
        serde_json::to_writer(
            self.centroids
                .for_field_with_idx(field, CentroidSlot::Centroids.index()),
            &SharedSegmentMeta {
                centroid_index: self.segment.index().centroid_index_meta().unwrap().clone(),
                num_docs: num_present,
            },
        )?;
        IvfIndex::serialize_offsets(
            &written.offsets,
            self.centroids
                .for_field_with_idx(field, CentroidSlot::Offsets.index()),
        )?;
        IvfIndex::serialize_bounds(
            BoundKind::Ball,
            &written.bounds,
            self.centroids
                .for_field_with_idx(field, CentroidSlot::Bounds.index()),
        )?;
        self.id_maps.push((field, written.doc_ids));
        Ok(())
    }

    fn finish(mut self) -> crate::Result<()> {
        for (field, doc_ids) in self.id_maps {
            IdMap::serialize_explicit(
                &doc_ids,
                self.vectors
                    .for_field_with_idx(field, VectorEntry::IdMap.index()),
            )?;
        }
        self.vectors.close()?;
        self.centroids.close()?;
        Ok(())
    }
}

pub(crate) fn merge_shared(ctx: &PluginMergeContext) -> crate::Result<()> {
    if ctx.cancel.wants_cancel() {
        return Err(TantivyError::Cancelled);
    }
    let num_docs = ctx.readers.iter().map(|reader| reader.num_docs()).sum();
    let mut writer = SharedSegmentWriter::new(ctx.target_segment)?;
    for (field, entry) in ctx.schema.fields() {
        if !matches!(entry.field_type(), FieldType::Vector(_)) {
            continue;
        }
        let readers = ctx
            .readers
            .iter()
            .map(|reader| reader.vector_index(field))
            .collect::<crate::Result<Vec<_>>>()?;
        let mut bounds = BoundsBuilder::new(writer.routers[&field].num_clusters());
        for reader in &readers {
            if let Some(source) = reader.index() {
                if Some(source.centroid_index_meta())
                    != ctx.target_segment.index().centroid_index_meta()
                {
                    return Err(TantivyError::InvalidArgument(
                        "cannot preserve memberships from a different centroid index".into(),
                    ));
                }
                for cluster in 0..source.num_clusters() {
                    if ctx.cancel.wants_cancel() {
                        return Err(TantivyError::Cancelled);
                    }
                    bounds.add_native(cluster, source.bounds().ball_r(cluster));
                }
            }
        }
        let source_rows = crate::vector::plugin::merge_source_rows(ctx, &readers)?;
        writer.write_field(
            field,
            num_docs,
            source_rows
                .into_iter()
                .enumerate()
                .filter_map(|(doc, source)| {
                    source.map(|(segment, row)| {
                        (
                            doc as DocId,
                            segment,
                            row,
                            readers[segment].row_cluster(row),
                        )
                    })
                }),
            bounds,
            |segment, row| readers[segment].vector_bytes_for_row(row),
            ctx.cancel,
        )?;
    }
    writer.finish()
}
