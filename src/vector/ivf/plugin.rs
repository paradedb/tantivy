//! IVF merge-time clustering and vector encoding.

use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(test)]
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(test)]
use cascade::prepare_centroid;
use cascade::{
    encode_batch_in_place_with_workspace, BatchEncodeWorkspace, LayerSpec,
    PreparedCentroidWorkspace, QueryRotationPlan,
};
use quant_model::Grid;

#[cfg(test)]
use super::decode_row;
use super::{
    decode_row_append, encode_vector, BuiltRouter, IvfCentroids, IvfClusterer, IvfIndex, IvfMatrix,
    IvfMatrixView, IvfTrainingBatch, IvfTrainingVectors, IvfVectorBatch, IvfVectors, RoutingIndex,
    CENTROIDS_EXT,
};
use crate::directory::{CompositeWrite, Directory, TempFilePtr};
use crate::index::SegmentComponent;
use crate::indexer::segment_updater::CancelSentinel;
use crate::plugin::PluginMergeContext;
#[cfg(test)]
use crate::schema::Metric;
use crate::schema::{Field, FieldType, VectorDType, VectorOptions};
#[cfg(test)]
use crate::vector::distance::l2_squared;
use crate::vector::distance::{maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::flat::IdMap;
use crate::vector::header::{
    centroid_slot, write_centroid_header, write_vector_header, CentroidSlot, VectorSlot, CURRENT,
    HEADER_LEN,
};
use crate::vector::{
    quantized_code_stride, residual_norm, BoundKind, BoundsBuilder, NeighborhoodGraphConfig,
    RelativeNeighborhoodGraph, VectorQuantizationConfig, QUANTIZED_CODE_ALIGNMENT,
    QUANTIZED_CONSTANT_STRIDE, QUANTIZED_RESIDUAL_NORM_STRIDE, QUANTIZED_SIDECAR_STRIDE, VEC_EXT,
};
use crate::{DocId, Executor, TantivyError};

struct AssignedVector {
    cluster: usize,
    target_doc_id: DocId,
    source_segment_ord: usize,
    source_doc_id: DocId,
}

/// Returns an executor sized to host parallelism.
fn build_executor(name: &'static str) -> crate::Result<Executor> {
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if num_threads > 1 {
        Executor::multi_thread(num_threads, name)
    } else {
        Ok(Executor::single_thread())
    }
}

/// Per-field IVF build timings (one phase per field), emitted at end of build
/// as a parseable `log::info!` line on target `paradedb::ivf_build`.
#[derive(Default)]
struct IvfBuildTimings {
    train: Duration,
    assign: Duration,
    posting_write: Duration,
    quantize: Duration,
}

/// Logical byte layout for one quantized slot.
#[derive(Clone, Debug, Eq, PartialEq)]
struct QuantizedSlotLayout {
    row_stride: usize,
    cluster_offsets: Vec<usize>,
    total_bytes: usize,
}

impl QuantizedSlotLayout {
    fn from_posting_offsets(row_stride: usize, posting_offsets: &[u64]) -> crate::Result<Self> {
        if posting_offsets.first().copied() != Some(0) {
            return Err(TantivyError::InternalError(
                "quantized layout requires posting offsets to start at row 0".to_string(),
            ));
        }
        let mut cluster_offsets = Vec::with_capacity(posting_offsets.len());
        cluster_offsets.push(0);
        let mut total_bytes = 0usize;
        for (cluster, rows) in posting_offsets.windows(2).enumerate() {
            let posting_rows = rows[1].checked_sub(rows[0]).ok_or_else(|| {
                TantivyError::InternalError(format!(
                    "quantized layout posting offsets decrease at cluster {cluster}: {} > {}",
                    rows[0], rows[1]
                ))
            })?;
            let posting_rows = usize::try_from(posting_rows).map_err(|_| {
                TantivyError::InternalError(format!(
                    "quantized layout row count does not fit usize at cluster {cluster}"
                ))
            })?;
            let posting_bytes = posting_rows.checked_mul(row_stride).ok_or_else(|| {
                TantivyError::InternalError(format!(
                    "quantized layout byte size overflows at cluster {cluster}"
                ))
            })?;
            total_bytes = total_bytes.checked_add(posting_bytes).ok_or_else(|| {
                TantivyError::InternalError(
                    "quantized layout total byte size overflows usize".to_string(),
                )
            })?;
            cluster_offsets.push(total_bytes);
        }
        Ok(Self {
            row_stride,
            cluster_offsets,
            total_bytes,
        })
    }

    #[cfg(test)]
    fn cluster_span(&self, cluster: usize) -> Range<usize> {
        self.cluster_offsets[cluster]..self.cluster_offsets[cluster + 1]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QuantizedLayerLayout {
    codes: QuantizedSlotLayout,
    sidecar: QuantizedSlotLayout,
    constants: Option<QuantizedSlotLayout>,
}

/// Per-field quantized slot layout.
#[derive(Clone, Debug, Eq, PartialEq)]
struct QuantizedWriteLayout {
    layers: Vec<QuantizedLayerLayout>,
    residual_norms: QuantizedSlotLayout,
}

impl QuantizedWriteLayout {
    fn build(config: &VectorQuantizationConfig, posting_offsets: &[u64]) -> crate::Result<Self> {
        let layers = config
            .layers
            .iter()
            .map(|layer| {
                Ok(QuantizedLayerLayout {
                    codes: QuantizedSlotLayout::from_posting_offsets(
                        quantized_code_stride(config.dim, layer.bits),
                        posting_offsets,
                    )?,
                    sidecar: QuantizedSlotLayout::from_posting_offsets(
                        QUANTIZED_SIDECAR_STRIDE,
                        posting_offsets,
                    )?,
                    constants: config
                        .needs_constants()
                        .then(|| {
                            QuantizedSlotLayout::from_posting_offsets(
                                QUANTIZED_CONSTANT_STRIDE,
                                posting_offsets,
                            )
                        })
                        .transpose()?,
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let residual_norms = QuantizedSlotLayout::from_posting_offsets(
            QUANTIZED_RESIDUAL_NORM_STRIDE,
            posting_offsets,
        )?;
        Ok(Self {
            layers,
            residual_norms,
        })
    }

    fn cluster_count(&self) -> usize {
        self.layers
            .first()
            .map_or(0, |layer| layer.codes.cluster_offsets.len() - 1)
    }
}

/// Merge-local spill file for one quantized slot.
struct QuantizedTempSlot {
    file: TempFilePtr,
    expected_len: usize,
    written_len: usize,
}

impl QuantizedTempSlot {
    fn create(directory: &dyn Directory, expected_len: usize) -> crate::Result<Self> {
        let file = directory.open_temp_file()?;
        Ok(Self {
            file,
            expected_len,
            written_len: 0,
        })
    }

    fn validate_offset(&self, expected: usize, context: &str) -> crate::Result<()> {
        if self.written_len != expected {
            return Err(TantivyError::InternalError(format!(
                "quantized layout mismatch for {context}: wrote {} bytes, expected {expected}",
                self.written_len
            )));
        }
        Ok(())
    }

    fn splice_into(
        &mut self,
        destination: &mut impl Write,
        cancel: &dyn CancelSentinel,
    ) -> crate::Result<()> {
        self.validate_offset(self.expected_len, "temporary quantized slot")?;
        self.file.flush()?;
        self.file.seek(SeekFrom::Start(0))?;
        let mut chunk = vec![0_u8; 1 << 20];
        loop {
            if cancel.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            let read = self.file.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            destination.write_all(&chunk[..read])?;
        }
        Ok(())
    }
}

impl Write for QuantizedTempSlot {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let remaining = self.expected_len.saturating_sub(self.written_len);
        if buf.len() > remaining {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "quantized temp slot would exceed layout length {} (written {}, write {})",
                    self.expected_len,
                    self.written_len,
                    buf.len()
                ),
            ));
        }
        let written = self.file.write(buf)?;
        self.written_len += written;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

struct QuantizedLayerTemps {
    codes: QuantizedTempSlot,
    sidecar: QuantizedTempSlot,
    constants: Option<QuantizedTempSlot>,
}

struct QuantizedTempSlots {
    layers: Vec<QuantizedLayerTemps>,
    residual_norms: QuantizedTempSlot,
}

impl QuantizedTempSlots {
    fn create(directory: &dyn Directory, layout: &QuantizedWriteLayout) -> crate::Result<Self> {
        let mut layers = Vec::with_capacity(layout.layers.len());
        for layer_layout in &layout.layers {
            layers.push(QuantizedLayerTemps {
                codes: QuantizedTempSlot::create(directory, layer_layout.codes.total_bytes)?,
                sidecar: QuantizedTempSlot::create(directory, layer_layout.sidecar.total_bytes)?,
                constants: layer_layout
                    .constants
                    .as_ref()
                    .map(|slot| QuantizedTempSlot::create(directory, slot.total_bytes))
                    .transpose()?,
            });
        }
        let residual_norms =
            QuantizedTempSlot::create(directory, layout.residual_norms.total_bytes)?;
        Ok(Self {
            layers,
            residual_norms,
        })
    }

    fn validate_cluster_boundary(
        &self,
        layout: &QuantizedWriteLayout,
        boundary: usize,
    ) -> crate::Result<()> {
        if boundary > layout.cluster_count() {
            return Err(TantivyError::InternalError(format!(
                "quantized layout boundary {boundary} exceeds cluster count {}",
                layout.cluster_count()
            )));
        }
        if self.layers.len() != layout.layers.len() {
            return Err(TantivyError::InternalError(format!(
                "quantized temp slot layer count {} disagrees with layout layer count {}",
                self.layers.len(),
                layout.layers.len()
            )));
        }
        for (layer, (temp, layer_layout)) in self.layers.iter().zip(&layout.layers).enumerate() {
            temp.codes.validate_offset(
                layer_layout.codes.cluster_offsets[boundary],
                &format!("layer {layer} codes at cluster boundary {boundary}"),
            )?;
            temp.sidecar.validate_offset(
                layer_layout.sidecar.cluster_offsets[boundary],
                &format!(
                    "layer {layer} scale/gamma/corrected-error sidecar at cluster boundary \
                     {boundary}"
                ),
            )?;
            match (&temp.constants, &layer_layout.constants) {
                (Some(temp), Some(slot)) => temp.validate_offset(
                    slot.cluster_offsets[boundary],
                    &format!("layer {layer} constants at cluster boundary {boundary}"),
                )?,
                (None, None) => {}
                _ => {
                    return Err(TantivyError::InternalError(format!(
                        "layer {layer} temp constants disagree with the metric layout"
                    )));
                }
            }
        }
        self.residual_norms.validate_offset(
            layout.residual_norms.cluster_offsets[boundary],
            &format!("residual norms at cluster boundary {boundary}"),
        )?;
        Ok(())
    }

    fn splice_into(
        &mut self,
        vec_write: &mut CompositeWrite,
        field: Field,
        cancel: &dyn CancelSentinel,
    ) -> crate::Result<()> {
        self.residual_norms.splice_into(
            vec_write.for_field_with_idx(field, VectorSlot::ResidualNorms.index()),
            cancel,
        )?;
        for (layer, temp) in self.layers.iter_mut().enumerate() {
            vec_write.align_next_field(QUANTIZED_CODE_ALIGNMENT, HEADER_LEN)?;
            temp.codes.splice_into(
                vec_write.for_field_with_idx(field, VectorSlot::codes(layer).index()),
                cancel,
            )?;
            temp.sidecar.splice_into(
                vec_write.for_field_with_idx(field, VectorSlot::sidecar(layer).index()),
                cancel,
            )?;
            if let Some(constants) = temp.constants.as_mut() {
                constants.splice_into(
                    vec_write.for_field_with_idx(field, VectorSlot::constants(layer).index()),
                    cancel,
                )?;
            }
        }
        Ok(())
    }
}

fn write_u16_run(writer: &mut impl Write, values: &[u16]) -> std::io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_u16_run_cancellable(
    writer: &mut impl Write,
    values: &[u16],
    cancel: &dyn CancelSentinel,
) -> crate::Result<()> {
    const VALUES_PER_CANCEL_POLL: usize = (1024 * 1024) / std::mem::size_of::<u16>();
    for chunk in values.chunks(VALUES_PER_CANCEL_POLL) {
        if cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
        write_u16_run(writer, chunk)?;
    }
    Ok(())
}

fn write_f32_run(writer: &mut impl Write, values: &[f32]) -> std::io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
fn write_sidecar_block(
    writer: &mut impl Write,
    scales: &[f32],
    gammas: &[u16],
    error_ratios: &[u16],
) -> std::io::Result<()> {
    assert_eq!(scales.len(), gammas.len());
    assert_eq!(scales.len(), error_ratios.len());
    write_f32_run(writer, scales)?;
    write_u16_run(writer, gammas)?;
    write_u16_run(writer, error_ratios)
}

fn quantization_runtime(
    config: &VectorQuantizationConfig,
    opts: &VectorOptions,
) -> crate::Result<(Vec<LayerSpec>, Vec<Grid>)> {
    config.validate(opts)?;
    let specs = config
        .layers
        .iter()
        .map(|layer| LayerSpec {
            bits: layer.bits,
            seed: layer.seed,
            rotate: true,
        })
        .collect();
    let grids = config
        .layers
        .iter()
        .map(|layer| {
            let grid = config
                .grids
                .iter()
                .find(|grid| grid.bits == layer.bits)
                .ok_or_else(|| {
                    TantivyError::InvalidArgument(format!(
                        "quantization field {:?} layer width {} has no persisted grid/model \
                         entry; rebuild required",
                        config.field, layer.bits
                    ))
                })?;
            Ok(Grid {
                bits: grid.bits,
                points: grid.points.clone(),
                rho_model: grid.rho_model,
            })
        })
        .collect::<crate::Result<Vec<_>>>()?;
    Ok((specs, grids))
}

fn write_empty_quantized_slots(
    vec_write: &mut CompositeWrite,
    field: Field,
    layer_count: usize,
    constants: bool,
) -> crate::Result<()> {
    let writer = vec_write.for_field_with_idx(field, VectorSlot::ResidualNorms.index());
    writer.flush()?;
    for layer in 0..layer_count {
        vec_write.align_next_field(QUANTIZED_CODE_ALIGNMENT, HEADER_LEN)?;
        {
            let writer = vec_write.for_field_with_idx(field, VectorSlot::codes(layer).index());
            writer.flush()?;
        }
        {
            let writer = vec_write.for_field_with_idx(field, VectorSlot::sidecar(layer).index());
            writer.flush()?;
        }
        if constants {
            let writer = vec_write.for_field_with_idx(field, VectorSlot::constants(layer).index());
            writer.flush()?;
        }
    }
    Ok(())
}

/// Writes an empty IVF field to both vector composites.
fn write_empty_field_slots(
    vec_write: &mut CompositeWrite,
    centroids_write: &mut CompositeWrite,
    field: Field,
    opts: &VectorOptions,
    quantization: Option<&VectorQuantizationConfig>,
) -> crate::Result<()> {
    {
        let id_map_w = vec_write.for_field_with_idx(field, VectorSlot::IdMap.index());
        IdMap::serialize_explicit(&[], id_map_w)?;
        id_map_w.flush()?;
    }
    {
        let rows_w = vec_write.for_field_with_idx(field, VectorSlot::Rows.index());
        rows_w.flush()?;
    }
    if let Some(config) = quantization {
        write_empty_quantized_slots(
            vec_write,
            field,
            config.layers.len(),
            config.needs_constants(),
        )?;
    }
    {
        let centroids_w =
            centroids_write.for_field_with_idx(field, CentroidSlot::Centroids.index());
        IvfIndex::serialize_centroids(0, 0, &[], opts, centroids_w)?;
        centroids_w.flush()?;
    }
    {
        let offsets_w = centroids_write.for_field_with_idx(field, CentroidSlot::Offsets.index());
        IvfIndex::serialize_offsets(&[0u64], offsets_w)?;
        offsets_w.flush()?;
    }
    {
        let bounds_w = centroids_write.for_field_with_idx(field, CentroidSlot::Bounds.index());
        IvfIndex::serialize_bounds(BoundKind::Ball, &[], bounds_w)?;
        bounds_w.flush()?;
    }
    Ok(())
}

pub(crate) fn merge_ivf(
    ctx: &PluginMergeContext,
    clusterer: Option<&dyn IvfClusterer>,
) -> crate::Result<()> {
    if ctx.cancel.wants_cancel() {
        return Err(TantivyError::Cancelled);
    }

    let has_vector_field = ctx
        .schema
        .fields()
        .any(|(_, entry)| matches!(entry.field_type(), FieldType::Vector(_)));
    if !has_vector_field {
        return Ok(());
    }

    let clusterer = clusterer.ok_or_else(|| {
        TantivyError::InvalidArgument(
            "vector_clustering_threshold selected IVF merge, but no IvfClusterer is configured"
                .to_string(),
        )
    })?;

    let num_target_docs: u32 = ctx.readers.iter().map(|r| r.num_docs()).sum();
    if num_target_docs == 0 {
        return Ok(());
    }

    let settings = clusterer.merge_settings(num_target_docs as usize)?;
    let directory = ctx.target_segment.index().directory();
    let vec_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(VEC_EXT.to_string()));
    let centroids_path = ctx
        .target_segment
        .relative_path(SegmentComponent::Custom(CENTROIDS_EXT.to_string()));
    let mut vec_file = directory.open_write(&vec_path)?;
    write_vector_header(&mut vec_file)?;
    let mut vec_write = CompositeWrite::wrap(vec_file);
    let mut centroids_file = directory.open_write(&centroids_path)?;
    write_centroid_header(&mut centroids_file)?;
    let mut centroids_write = CompositeWrite::wrap(centroids_file);

    for (field, entry) in ctx.schema.fields() {
        let opts = match entry.field_type() {
            FieldType::Vector(opts) => opts,
            _ => continue,
        };
        let quantization = ctx
            .settings
            .vector_quantization
            .iter()
            .find(|config| config.field == entry.name());
        if let Some(config) = quantization {
            config.validate(opts)?;
        }
        let field_readers: Vec<_> = ctx
            .readers
            .iter()
            .map(|reader| reader.vector_index(field))
            .collect::<crate::Result<Vec<_>>>()?;
        let vector_count = field_readers
            .iter()
            .map(|reader| reader.num_vectors())
            .sum::<usize>();
        if vector_count == 0 {
            write_empty_field_slots(
                &mut vec_write,
                &mut centroids_write,
                field,
                opts,
                quantization,
            )?;
            continue;
        }
        let training_sample_size = {
            let ratio = f64::from(settings.training_sample_ratio).clamp(f64::MIN_POSITIVE, 1.0);
            let target = ((vector_count as f64) * ratio).ceil() as usize;
            target.clamp(1, vector_count)
        };
        let training_sample_interval = (vector_count / training_sample_size).max(1);

        let residual: fn(&[u8], &[f32]) -> f32 = match opts.dtype() {
            VectorDType::F32 => residual_norm::<f32>,
        };
        let centroid_stride = opts.bytes_per_vector();
        let mut current_cluster = usize::MAX;
        let mut current_centroid = Vec::with_capacity(opts.dim());

        match opts.dtype() {
            VectorDType::F32 => {
                let field_build_start = Instant::now();
                let mut timings = IvfBuildTimings::default();
                let mut training_values = Vec::with_capacity(training_sample_size * opts.dim());
                let mut training_doc_ids = Vec::with_capacity(training_sample_size);
                let mut target_doc_id: DocId = 0;
                let mut present_vector_ord = 0usize;
                let mut sampled_count = 0usize;
                for source_doc_addr in ctx.doc_id_mapping.iter_source_doc_addrs() {
                    let reader = &field_readers[source_doc_addr.segment_ord as usize];
                    if let Some(bytes) = reader.vector_bytes(source_doc_addr.doc_id)? {
                        let should_sample = sampled_count < training_sample_size
                            && present_vector_ord % training_sample_interval == 0;
                        if should_sample {
                            training_doc_ids.push(target_doc_id);
                            decode_row_append::<f32>(&bytes, opts.dim(), &mut training_values)?;
                            sampled_count += 1;
                        }
                        present_vector_ord += 1;
                    }
                    target_doc_id += 1;
                }
                debug_assert_eq!(target_doc_id, num_target_docs);
                debug_assert!(
                    if ctx.readers.iter().any(|reader| reader.has_deletes()) {
                        present_vector_ord <= vector_count
                    } else {
                        present_vector_ord == vector_count
                    },
                    "{present_vector_ord} alive docs with vectors vs {vector_count} reported by \
                     source count()"
                );
                if training_doc_ids.is_empty() {
                    write_empty_field_slots(
                        &mut vec_write,
                        &mut centroids_write,
                        field,
                        opts,
                        quantization,
                    )?;
                    continue;
                }

                let training_rows = training_doc_ids.len();
                let training_vectors = IvfTrainingVectors::F32(IvfTrainingBatch {
                    doc_ids: training_doc_ids,
                    matrix: IvfMatrix {
                        values: training_values,
                        rows: training_rows,
                        dims: opts.dim(),
                    },
                });
                let train_start = Instant::now();
                let centroids = clusterer.train(opts, training_vectors)?;

                timings.train = train_start.elapsed();

                if ctx.cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }

                let IvfCentroids::F32(centroid_matrix) = &centroids;
                if centroid_matrix.dims != opts.dim() {
                    return Err(TantivyError::InvalidArgument(format!(
                        "IvfClusterer produced centroids with {} dimensions, expected {}",
                        centroid_matrix.dims,
                        opts.dim()
                    )));
                }
                if centroid_matrix.values.len() != centroid_matrix.rows * centroid_matrix.dims {
                    return Err(TantivyError::InvalidArgument(format!(
                        "IvfClusterer produced {} centroid values for {} rows x {} dimensions",
                        centroid_matrix.values.len(),
                        centroid_matrix.rows,
                        centroid_matrix.dims
                    )));
                }
                if centroid_matrix.rows == 0 {
                    return Err(TantivyError::InvalidArgument(
                        "IvfClusterer produced zero centroids".to_string(),
                    ));
                }
                let num_centroids = centroid_matrix.rows;

                // Optional router (slot [2]). Default clusterers return `None`
                // and the merge builds a routing RNG. When present, a stacked
                // build's canonical cluster-sort permutation is applied to the
                // trained centroid matrix before assign.
                let built_router = clusterer.build_router(opts, &centroids)?;
                let centroids = match &built_router {
                    Some(BuiltRouter::Stacked { perm, .. }) => {
                        let IvfCentroids::F32(matrix) = &centroids;
                        if perm.len() != num_centroids {
                            return Err(TantivyError::InvalidArgument(format!(
                                "build_router returned a permutation over {} centroids, expected \
                                 {num_centroids}",
                                perm.len()
                            )));
                        }
                        let dims = matrix.dims;
                        let mut values = vec![0.0f32; matrix.values.len()];
                        let mut seen = vec![false; num_centroids];
                        for (old, &new) in perm.iter().enumerate() {
                            let new = new as usize;
                            if new >= num_centroids || seen[new] {
                                return Err(TantivyError::InvalidArgument(
                                    "build_router permutation is not a bijection".to_string(),
                                ));
                            }
                            seen[new] = true;
                            values[new * dims..(new + 1) * dims]
                                .copy_from_slice(&matrix.values[old * dims..(old + 1) * dims]);
                        }
                        IvfCentroids::F32(IvfMatrix {
                            values,
                            rows: matrix.rows,
                            dims,
                        })
                    }
                    _ => centroids,
                };
                let IvfCentroids::F32(centroid_matrix) = &centroids;

                // Float working copy of the trained centroids — the
                // `.centroids` encode below reads per-row slices. Encoding +
                // Cosine normalization happen at the `.centroids` write
                // below.
                let centroid_rows: Vec<Vec<f32>> = centroid_matrix
                    .values
                    .chunks_exact(opts.dim())
                    .map(|centroid| centroid.to_vec())
                    .collect();

                let mut assigned_vectors = Vec::with_capacity(vector_count);
                let mut target_doc_id: DocId = 0;
                {
                    let mut batch_values = Vec::with_capacity(
                        settings.assign_batch_size.min(vector_count) * opts.dim(),
                    );
                    let mut batch_doc_ids =
                        Vec::with_capacity(settings.assign_batch_size.min(vector_count));
                    let mut batch_sources =
                        Vec::with_capacity(settings.assign_batch_size.min(vector_count));
                    let mut flush_assign_batch =
                        |batch_values: &mut Vec<f32>,
                         batch_doc_ids: &mut Vec<DocId>,
                         batch_sources: &mut Vec<(DocId, usize, DocId)>|
                         -> crate::Result<()> {
                            if batch_doc_ids.is_empty() {
                                return Ok(());
                            }
                            if ctx.cancel.wants_cancel() {
                                return Err(TantivyError::Cancelled);
                            }
                            let batch_len = batch_doc_ids.len();
                            let assign_start = Instant::now();
                            let clusters = clusterer.assign(
                                opts,
                                IvfVectors::F32(IvfVectorBatch {
                                    doc_ids: batch_doc_ids.as_slice(),
                                    matrix: IvfMatrixView {
                                        values: batch_values.as_slice(),
                                        rows: batch_len,
                                        dims: opts.dim(),
                                    },
                                }),
                                &centroids,
                            )?;
                            timings.assign += assign_start.elapsed();
                            if clusters.len() != batch_len {
                                return Err(TantivyError::InvalidArgument(format!(
                                    "IvfClusterer assigned {} clusters for {} vectors",
                                    clusters.len(),
                                    batch_len
                                )));
                            }
                            for (cluster, (target_doc_id, source_segment_ord, source_doc_id)) in
                                clusters.into_iter().zip(batch_sources.drain(..))
                            {
                                let cluster = cluster as usize;
                                if cluster >= num_centroids {
                                    return Err(TantivyError::InvalidArgument(format!(
                                        "IvfClusterer assigned vector to cluster {cluster}, but \
                                         only {num_centroids} centroids were trained"
                                    )));
                                }
                                assigned_vectors.push(AssignedVector {
                                    cluster,
                                    target_doc_id,
                                    source_segment_ord,
                                    source_doc_id,
                                });
                            }
                            batch_values.clear();
                            batch_doc_ids.clear();
                            Ok(())
                        };
                    for source_doc_addr in ctx.doc_id_mapping.iter_source_doc_addrs() {
                        let reader = &field_readers[source_doc_addr.segment_ord as usize];
                        if let Some(bytes) = reader.vector_bytes(source_doc_addr.doc_id)? {
                            batch_doc_ids.push(target_doc_id);
                            decode_row_append::<f32>(&bytes, opts.dim(), &mut batch_values)?;
                            batch_sources.push((
                                target_doc_id,
                                source_doc_addr.segment_ord as usize,
                                source_doc_addr.doc_id,
                            ));
                            if batch_doc_ids.len() == settings.assign_batch_size {
                                flush_assign_batch(
                                    &mut batch_values,
                                    &mut batch_doc_ids,
                                    &mut batch_sources,
                                )?;
                            }
                        }
                        target_doc_id += 1;
                    }
                    flush_assign_batch(&mut batch_values, &mut batch_doc_ids, &mut batch_sources)?;
                }
                debug_assert_eq!(target_doc_id, num_target_docs);
                debug_assert_eq!(assigned_vectors.len(), present_vector_ord);
                // The `.centroids` doc count: one posting row per document.
                let num_present_docs = assigned_vectors.len();

                let mut cluster_counts = vec![0usize; num_centroids];
                for assigned_vector in &assigned_vectors {
                    cluster_counts[assigned_vector.cluster] += 1;
                }

                assigned_vectors
                    .sort_unstable_by_key(|vector| (vector.cluster, vector.target_doc_id));

                let mut cluster_offsets: Vec<u64> = Vec::with_capacity(num_centroids + 1);
                let mut next_offset = 0u64;
                cluster_offsets.push(next_offset);
                for cluster_count in cluster_counts {
                    next_offset += cluster_count as u64;
                    cluster_offsets.push(next_offset);
                }

                let mut centroid_bytes =
                    Vec::with_capacity(num_centroids * opts.bytes_per_vector());
                let mut bounds_builder = BoundsBuilder::new(num_centroids);
                let mut stored_centroid = Vec::with_capacity(opts.dim());
                for (centroid_ord, centroid) in centroid_rows.iter().enumerate() {
                    let mut bytes = encode_vector(centroid, opts.dim())?;
                    let outcome = maybe_normalize_bytes(opts, &mut bytes);
                    if outcome == NormalizeOutcome::NonFinite {
                        log::warn!(
                            "non-finite centroid {centroid_ord} in field '{}' written \
                             un-normalized during merge",
                            entry.name(),
                        );
                    }
                    stored_centroid.clear();
                    decode_row_append::<f32>(&bytes, opts.dim(), &mut stored_centroid)?;
                    if outcome != NormalizeOutcome::Normalized
                        || stored_centroid.iter().any(|value| !value.is_finite())
                    {
                        bounds_builder.saturate(centroid_ord);
                    }
                    centroid_bytes.extend_from_slice(&bytes);
                }

                let posting_start = Instant::now();
                {
                    let id_map_w = vec_write.for_field_with_idx(field, VectorSlot::IdMap.index());
                    let row_doc_ids: Vec<DocId> = assigned_vectors
                        .iter()
                        .map(|assigned_vector| assigned_vector.target_doc_id)
                        .collect();
                    IdMap::serialize_explicit(&row_doc_ids, id_map_w)?;
                    id_map_w.flush()?;
                }

                {
                    const CANCEL_POLL_ROWS: usize = 4096;
                    let rows_w = vec_write.for_field_with_idx(field, VectorSlot::Rows.index());
                    let needs_norm = opts.needs_normalization();
                    let mut row_buf: Vec<u8> = Vec::with_capacity(opts.bytes_per_vector());
                    for (row_idx, assigned_vector) in assigned_vectors.iter().enumerate() {
                        if row_idx % CANCEL_POLL_ROWS == 0 && ctx.cancel.wants_cancel() {
                            return Err(TantivyError::Cancelled);
                        }
                        let reader = &field_readers[assigned_vector.source_segment_ord];
                        let bytes = reader
                            .vector_bytes(assigned_vector.source_doc_id)?
                            .ok_or_else(|| {
                                TantivyError::InternalError(format!(
                                    "missing source vector for doc {:?}",
                                    assigned_vector.source_doc_id
                                ))
                            })?;
                        let written_bytes: &[u8] = if needs_norm {
                            row_buf.clear();
                            row_buf.extend_from_slice(&bytes);
                            if maybe_normalize_bytes(opts, &mut row_buf)
                                == NormalizeOutcome::NonFinite
                            {
                                log::warn!(
                                    "non-finite vector in field '{}' (doc {}) written \
                                     un-normalized during merge",
                                    entry.name(),
                                    assigned_vector.target_doc_id,
                                );
                            }
                            &row_buf
                        } else {
                            &bytes
                        };
                        rows_w.write_all(written_bytes)?;
                        // The bounds fold uses the exact bytes written above
                        // against the stored centroid.
                        // A non-finite row residual saturates its cluster
                        // inside `add_native`.
                        if assigned_vector.cluster != current_cluster {
                            current_cluster = assigned_vector.cluster;
                            current_centroid.clear();
                            decode_row_append::<f32>(
                                &centroid_bytes[current_cluster * centroid_stride..]
                                    [..centroid_stride],
                                opts.dim(),
                                &mut current_centroid,
                            )?;
                        }
                        bounds_builder.add_native(
                            assigned_vector.cluster,
                            residual(written_bytes, &current_centroid),
                        );
                    }
                    rows_w.flush()?;
                }
                timings.posting_write = posting_start.elapsed();

                if let Some(config) = quantization {
                    let quantize_start = Instant::now();
                    let (specs, grids) = quantization_runtime(config, opts)?;
                    let quantized_layout = QuantizedWriteLayout::build(config, &cluster_offsets)?;
                    let rotation_plan = Arc::new(QueryRotationPlan::new(opts.dim(), &specs));
                    let mut centroid_workspace = PreparedCentroidWorkspace::new(rotation_plan);
                    let mut temp_slots = QuantizedTempSlots::create(directory, &quantized_layout)?;

                    const MAX_QUANTIZATION_SCRATCH_BYTES: usize = 1 << 20;
                    let row_bytes = opts.dim() * std::mem::size_of::<f32>();
                    let fixed_scratch = row_bytes
                        + opts.dim() * std::mem::size_of::<u8>()
                        + opts.dim().div_ceil(64) * std::mem::size_of::<u64>();
                    let per_row_scratch =
                        2 * row_bytes + config.bytes_per_row() + 2 * std::mem::size_of::<f64>();
                    let tile_rows = MAX_QUANTIZATION_SCRATCH_BYTES
                        .saturating_sub(fixed_scratch)
                        .checked_div(per_row_scratch)
                        .unwrap_or(0)
                        .max(1);
                    let needs_norm = opts.needs_normalization();
                    let mut normalized = Vec::with_capacity(opts.bytes_per_vector());
                    let mut batch_values = Vec::with_capacity(tile_rows * opts.dim());
                    let mut encode_workspace =
                        BatchEncodeWorkspace::with_capacity(opts.dim(), tile_rows, &specs);
                    let mut cluster_gammas: Vec<Vec<u16>> =
                        (0..config.layers.len()).map(|_| Vec::new()).collect();
                    let mut cluster_error_ratios: Vec<Vec<u16>> =
                        (0..config.layers.len()).map(|_| Vec::new()).collect();
                    let mut quantized_centroid = Vec::with_capacity(opts.dim());
                    for (cluster, offsets) in cluster_offsets.windows(2).enumerate() {
                        temp_slots.validate_cluster_boundary(&quantized_layout, cluster)?;
                        let start = offsets[0] as usize;
                        let end = offsets[1] as usize;
                        if start == end {
                            temp_slots.validate_cluster_boundary(&quantized_layout, cluster + 1)?;
                            continue;
                        }
                        quantized_centroid.clear();
                        decode_row_append::<f32>(
                            &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
                            opts.dim(),
                            &mut quantized_centroid,
                        )?;
                        let prepared = centroid_workspace.prepare(&quantized_centroid);
                        for gammas in &mut cluster_gammas {
                            gammas.clear();
                            gammas.reserve(end - start);
                        }
                        for error_ratios in &mut cluster_error_ratios {
                            error_ratios.clear();
                            error_ratios.reserve(end - start);
                        }
                        for tile in assigned_vectors[start..end].chunks(tile_rows) {
                            if ctx.cancel.wants_cancel() {
                                return Err(TantivyError::Cancelled);
                            }
                            batch_values.clear();
                            for assigned_vector in tile {
                                let reader = &field_readers[assigned_vector.source_segment_ord];
                                let bytes = reader
                                    .vector_bytes(assigned_vector.source_doc_id)?
                                    .ok_or_else(|| {
                                        TantivyError::InternalError(format!(
                                            "missing source vector for doc {:?}",
                                            assigned_vector.source_doc_id
                                        ))
                                    })?;
                                let encoded_bytes: &[u8] = if needs_norm {
                                    normalized.clear();
                                    normalized.extend_from_slice(&bytes);
                                    if maybe_normalize_bytes(opts, &mut normalized)
                                        == NormalizeOutcome::NonFinite
                                    {
                                        log::warn!(
                                            "non-finite vector in field '{}' (doc {}) encoded \
                                             un-normalized during merge",
                                            entry.name(),
                                            assigned_vector.target_doc_id,
                                        );
                                    }
                                    &normalized
                                } else {
                                    &bytes
                                };
                                decode_row_append::<f32>(
                                    encoded_bytes,
                                    opts.dim(),
                                    &mut batch_values,
                                )?;
                            }
                            let batch = encode_batch_in_place_with_workspace(
                                &mut batch_values,
                                tile.len(),
                                prepared,
                                &specs,
                                &grids,
                                &mut encode_workspace,
                                config.needs_constants(),
                            );
                            write_f32_run(
                                &mut temp_slots.residual_norms,
                                &batch.residual_norms_squared,
                            )?;
                            for (layer_index, (target, layer)) in
                                temp_slots.layers.iter_mut().zip(&batch.layers).enumerate()
                            {
                                target.codes.write_all(&layer.codes)?;
                                match &mut target.constants {
                                    Some(constants) => {
                                        write_f32_run(constants, &layer.constants)?;
                                    }
                                    None => debug_assert!(layer.constants.is_empty()),
                                }
                                write_f32_run(&mut target.sidecar, &layer.scales)?;
                                cluster_gammas[layer_index].extend_from_slice(&layer.gammas);
                                cluster_error_ratios[layer_index]
                                    .extend_from_slice(&layer.corrected_error_ratios);
                            }
                        }
                        for ((target, gammas), error_ratios) in temp_slots
                            .layers
                            .iter_mut()
                            .zip(&cluster_gammas)
                            .zip(&cluster_error_ratios)
                        {
                            write_u16_run_cancellable(&mut target.sidecar, gammas, ctx.cancel)?;
                            write_u16_run_cancellable(
                                &mut target.sidecar,
                                error_ratios,
                                ctx.cancel,
                            )?;
                        }
                        temp_slots.validate_cluster_boundary(&quantized_layout, cluster + 1)?;
                    }
                    temp_slots.splice_into(&mut vec_write, field, ctx.cancel)?;
                    timings.quantize = quantize_start.elapsed();
                }

                {
                    let centroids_w =
                        centroids_write.for_field_with_idx(field, CentroidSlot::Centroids.index());
                    IvfIndex::serialize_centroids(
                        num_centroids,
                        num_present_docs,
                        &centroid_bytes,
                        opts,
                        centroids_w,
                    )?;
                    centroids_w.flush()?;
                }
                {
                    let offsets_w =
                        centroids_write.for_field_with_idx(field, CentroidSlot::Offsets.index());
                    IvfIndex::serialize_offsets(&cluster_offsets, offsets_w)?;
                    offsets_w.flush()?;
                }
                {
                    let bounds_w =
                        centroids_write.for_field_with_idx(field, CentroidSlot::Bounds.index());
                    IvfIndex::serialize_bounds(
                        BoundKind::Ball,
                        &bounds_builder.finish(),
                        bounds_w,
                    )?;
                    bounds_w.flush()?;
                }

                // `.centroids` slot [2]: the tagged router. Skipped when no
                // clusterer router was built and `num_centroids <= 1`.
                if num_centroids > 1 {
                    if ctx.cancel.wants_cancel() {
                        return Err(TantivyError::Cancelled);
                    }
                    let router_w = centroids_write.for_field_with_idx(field, centroid_slot::ROUTER);
                    match built_router.as_ref() {
                        Some(BuiltRouter::Stacked { index, .. }) => {
                            RoutingIndex::serialize_stacked(CURRENT, index, router_w)?;
                        }
                        Some(BuiltRouter::Graph(graph)) => {
                            RoutingIndex::serialize_graph(CURRENT, graph, router_w)?;
                        }
                        None => {
                            let mut rng = RelativeNeighborhoodGraph::new(
                                centroid_matrix.values.as_slice(),
                                opts.dim(),
                                opts.metric(),
                                NeighborhoodGraphConfig::default(),
                            );
                            rng.build(&build_executor("rng-build-")?);
                            RoutingIndex::serialize_graph(CURRENT, &rng, router_w)?;
                        }
                    }
                    router_w.flush()?;
                }

                log::info!(
                    target: "paradedb::ivf_build",
                    "ivf_build timings_ms train={} assign={} posting_write={} quantize={} total={} \
                     centroids={} vectors={}",
                    timings.train.as_millis(),
                    timings.assign.as_millis(),
                    timings.posting_write.as_millis(),
                    timings.quantize.as_millis(),
                    field_build_start.elapsed().as_millis(),
                    num_centroids,
                    vector_count,
                );
            }
        }
    }

    vec_write.close()?;
    centroids_write.close()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::directory::{ManagedDirectory, RamDirectory};
    use crate::vector::VectorQuantizationLayer;

    fn quant_fixture_config_for(
        dim: usize,
        metric: Metric,
        schedule: &[u8],
    ) -> VectorQuantizationConfig {
        let seeds = [0x1111, 0x2222, 0x3333, 0x4444];
        VectorQuantizationConfig::materialize(
            "embedding".to_string(),
            &VectorOptions::new(dim, metric),
            schedule
                .iter()
                .enumerate()
                .map(|(layer, &bits)| VectorQuantizationLayer {
                    bits,
                    seed: seeds[layer],
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn blocked_sidecar_is_cluster_local_and_deterministic() {
        let mut bytes = Vec::new();
        write_sidecar_block(
            &mut bytes,
            &[1.0, 2.0],
            &[0x1112, 0x1314],
            &[0x2122, 0x2324],
        )
        .unwrap();
        write_sidecar_block(&mut bytes, &[], &[], &[]).unwrap();
        write_sidecar_block(&mut bytes, &[3.0], &[0x1516], &[0x2526]).unwrap();
        assert_eq!(
            bytes,
            [
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, // cluster 0 scale run
                0x12, 0x11, 0x14, 0x13, // cluster 0 gamma run
                0x22, 0x21, 0x24, 0x23, // cluster 0 corrected-error run
                0x00, 0x00, 0x40, 0x40, // cluster 1 scale run
                0x16, 0x15, // cluster 1 gamma run
                0x26, 0x25, // cluster 1 corrected-error run
            ]
        );
    }

    #[test]
    fn quantized_layout_is_exact_for_odd_d_empty_cluster_l2_1_plus_4() -> crate::Result<()> {
        let config = quant_fixture_config_for(100, Metric::L2, &[1, 4]);
        let posting_offsets = [0_u64, 2, 2, 5];
        let layout = QuantizedWriteLayout::build(&config, &posting_offsets)?;
        assert_eq!(
            layout,
            QuantizedWriteLayout::build(&config, &posting_offsets)?,
            "layout construction must be deterministic"
        );
        assert_eq!(layout.cluster_count(), 3);
        assert_eq!(layout.layers.len(), 2);

        let layer0 = &layout.layers[0];
        assert_eq!(layer0.codes.row_stride, 16);
        assert_eq!(layer0.codes.cluster_offsets, [0, 32, 32, 80]);
        assert_eq!(layer0.codes.cluster_span(0), 0..32);
        assert_eq!(layer0.codes.cluster_span(1), 32..32);
        assert_eq!(layer0.codes.cluster_span(2), 32..80);
        assert_eq!(layer0.codes.total_bytes, 80);
        assert_eq!(layer0.sidecar.row_stride, 8);
        assert_eq!(layer0.sidecar.cluster_offsets, [0, 16, 16, 40]);
        assert_eq!(layer0.sidecar.total_bytes, 40);
        let layer0_constants = layer0.constants.as_ref().unwrap();
        assert_eq!(layer0_constants.cluster_offsets, [0, 8, 8, 20]);
        assert_eq!(layer0_constants.total_bytes, 20);

        let layer1 = &layout.layers[1];
        assert_eq!(layer1.codes.row_stride, 56);
        assert_eq!(layer1.codes.cluster_offsets, [0, 112, 112, 280]);
        assert_eq!(layer1.codes.total_bytes, 280);
        assert_eq!(layer1.sidecar.cluster_offsets, [0, 16, 16, 40]);
        assert_eq!(
            layer1.constants.as_ref().unwrap().cluster_offsets,
            [0, 8, 8, 20]
        );

        let residual_norms = &layout.residual_norms;
        assert_eq!(residual_norms.row_stride, 4);
        assert_eq!(residual_norms.cluster_offsets, [0, 8, 8, 20]);
        assert_eq!(residual_norms.cluster_span(1), 8..8);
        assert_eq!(residual_norms.total_bytes, 20);

        let logical_total: usize = layout
            .layers
            .iter()
            .map(|layer| {
                layer.codes.total_bytes
                    + layer.sidecar.total_bytes
                    + layer.constants.as_ref().unwrap().total_bytes
            })
            .sum::<usize>()
            + residual_norms.total_bytes;
        assert_eq!(logical_total, 5 * config.bytes_per_row());
        assert_eq!(logical_total, 500);

        let backing = RamDirectory::create();
        let directory = ManagedDirectory::wrap(Box::new(backing))?;
        let mut temps = QuantizedTempSlots::create(&directory, &layout)?;
        for cluster in 0..layout.cluster_count() {
            temps.validate_cluster_boundary(&layout, cluster)?;
            for (temp, layer) in temps.layers.iter_mut().zip(&layout.layers) {
                temp.codes
                    .write_all(&vec![0; layer.codes.cluster_span(cluster).len()])?;
                temp.sidecar
                    .write_all(&vec![0; layer.sidecar.cluster_span(cluster).len()])?;
                temp.constants.as_mut().unwrap().write_all(&vec![
                    0;
                    layer
                        .constants
                        .as_ref()
                        .unwrap()
                        .cluster_span(cluster)
                        .len()
                ])?;
            }
            temps
                .residual_norms
                .write_all(&vec![0; residual_norms.cluster_span(cluster).len()])?;
            temps.validate_cluster_boundary(&layout, cluster + 1)?;
        }
        Ok(())
    }

    #[test]
    fn quantized_temp_slot_splices_exact_payload() -> crate::Result<()> {
        let backing = RamDirectory::create();
        let directory = ManagedDirectory::wrap(Box::new(backing))?;
        let mut temp = QuantizedTempSlot::create(&directory, b"first-second".len())?;
        temp.write_all(b"first")?;
        temp.write_all(b"-second")?;
        let mut destination = Vec::new();
        temp.splice_into(&mut destination, &|| false)?;
        assert_eq!(destination, b"first-second");
        Ok(())
    }

    #[test]
    fn quantized_temp_slot_rejects_layout_underwrite() -> crate::Result<()> {
        let backing = RamDirectory::create();
        let directory = ManagedDirectory::wrap(Box::new(backing))?;
        let mut temp = QuantizedTempSlot::create(&directory, 4)?;
        temp.write_all(&[1, 2, 3])?;
        let mut destination = Vec::new();
        let error = temp
            .splice_into(&mut destination, &|| false)
            .expect_err("layout underwrite must fail before splice");
        assert!(error.to_string().contains("wrote 3 bytes, expected 4"));
        assert!(destination.is_empty());
        Ok(())
    }

    #[test]
    fn quantized_temp_slot_cancellation_stops_between_chunks() -> crate::Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
        use std::sync::Arc;

        let backing = RamDirectory::create();
        let directory = ManagedDirectory::wrap(Box::new(backing))?;
        let expected_len = 2 * 1024 * 1024 + 17;
        let mut temp = QuantizedTempSlot::create(&directory, expected_len)?;
        temp.write_all(&vec![0x5a; expected_len])?;

        let polls = Arc::new(AtomicUsize::new(0));
        let cancel = {
            let polls = Arc::clone(&polls);
            move || polls.fetch_add(1, AtomicOrdering::SeqCst) >= 1
        };
        let mut destination = Vec::new();
        assert!(matches!(
            temp.splice_into(&mut destination, &cancel),
            Err(TantivyError::Cancelled)
        ));
        assert_eq!(destination.len(), 1024 * 1024);
        Ok(())
    }

    #[test]
    fn quantization_merge_source_has_no_estimator_analysis_entrypoint() {
        let source = include_str!("plugin.rs");
        let test_module_start = source
            .rfind("\n#[cfg(test)]\nmod tests {")
            .expect("plugin source must retain one terminal test module");
        let production_source = &source[..test_module_start];
        for forbidden in [
            "build_grid(",
            "audit_prefix_error_model(",
            "prepare_fp_query(",
            "audit_error",
            "diagnostic_error",
            "VectorEstimatorMeasurements",
            "VectorEstimatorQuery",
        ] {
            assert!(
                !production_source.contains(forbidden),
                "quantization merge production source contains forbidden analysis hook {forbidden}"
            );
        }
    }

    #[test]
    fn merge_runtime_uses_persisted_grid_and_rho() {
        let config = quant_fixture_config_for(100, Metric::Dot, &[1, 4]);
        let (_, resolved) =
            quantization_runtime(&config, &VectorOptions::new(100, Metric::Dot)).unwrap();
        assert_eq!(resolved.len(), 2);
        for (layer, grid) in resolved.iter().enumerate() {
            let persisted = config
                .grids
                .iter()
                .find(|grid| grid.bits == config.layers[layer].bits)
                .unwrap();
            assert_eq!(grid.bits, persisted.bits);
            assert_eq!(grid.points, persisted.points);
            assert_eq!(grid.rho_model, persisted.rho_model);
        }
    }
}
