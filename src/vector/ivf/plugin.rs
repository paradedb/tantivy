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
    use std::sync::Arc;

    use super::*;
    use crate::directory::{ManagedDirectory, RamDirectory};
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
    use crate::vector::ivf::AdaptiveProbeParams;
    use crate::vector::prepared::{QuantizedIndexCtx, QuantizedQueryCtx};
    use crate::vector::tests::ground_truth;
    use crate::vector::{
        TopDocsByVectorSimilarity, VectorEstimatorMeasurements, VectorEstimatorQuery,
        VectorEstimatorSource, VectorQuantizationLayer,
    };
    use crate::{Index, TantivyDocument};

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

    const QUANT_FIXTURE_DIM: usize = 64;

    struct QuantFixtureClusterer {
        dim: usize,
        metric: Metric,
    }

    impl IvfClusterer for QuantFixtureClusterer {
        fn training_sample_ratio(&self) -> f32 {
            0.5
        }

        fn train(
            &self,
            options: &VectorOptions,
            _vectors: IvfTrainingVectors,
        ) -> crate::Result<IvfCentroids> {
            assert_eq!(options.dim(), self.dim);
            let values = match self.metric {
                Metric::L2 => [0.0_f32, 1.0]
                    .into_iter()
                    .flat_map(|center| std::iter::repeat_n(center, self.dim))
                    .collect(),
                Metric::Cosine => {
                    let mut values = vec![0.0; 2 * self.dim];
                    values[0] = 1.0;
                    values[self.dim + 1] = 1.0;
                    values
                }
                Metric::Dot => unreachable!("quantized matrix fixture covers L2 and cosine"),
            };
            Ok(IvfCentroids::F32(IvfMatrix {
                values,
                rows: 2,
                dims: self.dim,
            }))
        }

        fn assign(
            &self,
            _options: &VectorOptions,
            vectors: IvfVectors<'_>,
            _centroids: &IvfCentroids,
        ) -> crate::Result<Vec<u32>> {
            let IvfVectors::F32(vectors) = vectors;
            Ok(vectors
                .matrix
                .values
                .chunks_exact(self.dim)
                .map(|row| match self.metric {
                    Metric::L2 => u32::from(row[0] >= 0.5),
                    Metric::Cosine => u32::from(row[1] > row[0]),
                    Metric::Dot => unreachable!("quantized matrix fixture covers L2 and cosine"),
                })
                .collect())
        }
    }

    fn quant_fixture_config(dim: usize) -> VectorQuantizationConfig {
        quant_fixture_config_for(dim, Metric::L2, &[1, 4])
    }

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

    fn fixture_vector(metric: Metric, dim: usize, doc: usize) -> Vec<f32> {
        match metric {
            Metric::L2 => {
                let center = if doc < 4 { 0.0 } else { 1.0 };
                (0..dim)
                    .map(|coordinate| {
                        center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1
                    })
                    .collect()
            }
            Metric::Cosine => {
                let cluster = usize::from(doc >= 4);
                let mut vector: Vec<f32> = (0..dim)
                    .map(|coordinate| ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.025)
                    .collect();
                vector[cluster] += 1.0;
                vector
            }
            Metric::Dot => unreachable!("quantized matrix fixture covers L2 and cosine"),
        }
    }

    fn fixture_estimator_queries_for(metric: Metric, dim: usize) -> Vec<Vec<f32>> {
        (0..4)
            .map(|query| match metric {
                Metric::L2 => {
                    let center = if query < 2 { 0.0 } else { 1.0 };
                    (0..dim)
                        .map(|coordinate| {
                            center + ((query * dim + coordinate) as f32 * 0.023).cos() * 0.1
                        })
                        .collect()
                }
                Metric::Cosine => {
                    let cluster = usize::from(query >= 2);
                    let mut vector: Vec<f32> = (0..dim)
                        .map(|coordinate| ((query * dim + coordinate) as f32 * 0.023).cos() * 0.025)
                        .collect();
                    vector[cluster] += 1.0;
                    vector
                }
                Metric::Dot => {
                    unreachable!("quantized matrix fixture covers L2 and cosine")
                }
            })
            .collect()
    }

    fn fixture_estimator_queries(dim: usize) -> Vec<Vec<f32>> {
        fixture_estimator_queries_for(Metric::L2, dim)
    }

    fn estimator_measurements(
        vector_reader: &crate::vector::VectorIndexReader,
        queries: &[Vec<f32>],
        sample_rows: usize,
        alive: Option<&crate::fastfield::AliveBitSet>,
    ) -> crate::Result<Option<VectorEstimatorMeasurements>> {
        let queries = queries
            .iter()
            .cloned()
            .map(|values| VectorEstimatorQuery {
                values,
                excluded_doc_id: None,
            })
            .collect::<Vec<_>>();
        Ok(vector_reader.measure_estimator_queries(
            VectorEstimatorSource::Provided,
            &queries,
            sample_rows,
            alive,
        )?)
    }

    fn build_quantized_fixture_with_schedule(dim: usize, quantized: bool) -> crate::Result<Index> {
        build_quantized_fixture_case(dim, Metric::L2, &[1, 4], quantized)
    }

    fn build_quantized_fixture_case(
        dim: usize,
        metric: Metric,
        schedule: &[u8],
        quantized: bool,
    ) -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_vector_field("embedding", VectorOptions::new(dim, metric));
        let label_field = schema_builder.add_text_field("label", STRING | STORED);
        let schema = schema_builder.build();
        let mut settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..Default::default()
        };
        if quantized {
            settings.vector_quantization = vec![quant_fixture_config_for(dim, metric, schedule)];
        }
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(QuantFixtureClusterer { dim, metric }))
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for doc in 0..8 {
            let vector = fixture_vector(metric, dim, doc);
            let mut document = TantivyDocument::new();
            document.add_vector(field, &vector);
            document.add_text(label_field, format!("d{doc}"));
            if doc % 2 == 0 {
                document.add_text(label_field, "keep");
            }
            writer.add_document(document)?;
            if doc == 3 || doc == 7 {
                writer.commit()?;
            }
        }
        let mut segments = index.searchable_segment_ids()?;
        segments.sort();
        writer.merge(&segments).wait()?;
        writer.wait_merging_threads()?;
        Ok(index)
    }

    fn build_quantized_fixture(dim: usize, quantized: bool) -> crate::Result<Index> {
        build_quantized_fixture_with_schedule(dim, quantized)
    }

    fn build_flat_quantized_fixture(dim: usize) -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let field =
            schema_builder.add_vector_field("embedding", VectorOptions::new(dim, Metric::L2));
        let schema = schema_builder.build();
        let settings = IndexSettings {
            vector_clustering_threshold: usize::MAX,
            vector_quantization: vec![quant_fixture_config(dim)],
            ..Default::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .create_in_ram()?;
        let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for doc in 0..8 {
            let center = if doc < 4 { 0.0 } else { 1.0 };
            let vector: Vec<f32> = (0..dim)
                .map(|coordinate| center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1)
                .collect();
            let mut document = TantivyDocument::new();
            document.add_vector(field, &vector);
            writer.add_document(document)?;
        }
        writer.commit()?;
        Ok(index)
    }

    fn fixture_expected(query: &[f32], dim: usize, top_n: usize) -> Vec<(u32, u32)> {
        let mut expected: Vec<(f32, u32)> = (0..8)
            .map(|doc| {
                let center = if doc < 4 { 0.0 } else { 1.0 };
                let vector: Vec<f32> = (0..dim)
                    .map(|coordinate| {
                        center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1
                    })
                    .collect();
                (-l2_squared(query, &vector), doc as u32)
            })
            .collect();
        expected.sort_unstable_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        expected[..top_n]
            .iter()
            .map(|&(score, doc)| (score.to_bits(), doc))
            .collect()
    }

    #[derive(Clone, Copy, Debug)]
    enum QuantizedMatrixScenario {
        None,
        Filter,
        Deletes,
    }

    fn fixture_search_query(metric: Metric, dim: usize) -> Vec<f32> {
        match metric {
            Metric::L2 => vec![0.05; dim],
            Metric::Cosine => {
                let mut query: Vec<f32> = (0..dim)
                    .map(|coordinate| ((coordinate as f32 + 0.5) * 0.031).cos() * 0.01)
                    .collect();
                query[0] += 0.8;
                query[1] += 0.6;
                query
            }
            Metric::Dot => unreachable!("quantized matrix fixture covers L2 and cosine"),
        }
    }

    fn fixture_filter_docs(
        index: &Index,
        filter: &dyn Query,
    ) -> crate::Result<std::collections::HashSet<crate::DocAddress>> {
        let searcher = index.reader()?.searcher();
        let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let mut admitted = std::collections::HashSet::new();
        for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
            weight.for_each_no_score(segment, &mut |docs| {
                admitted.extend(
                    docs.iter()
                        .copied()
                        .map(|doc| crate::DocAddress::new(segment_ord as u32, doc)),
                );
            })?;
        }
        Ok(admitted)
    }

    fn fixture_exact_hits(
        index: &Index,
        metric: Metric,
        query: &[f32],
        filter: Option<&dyn Query>,
        top_n: usize,
    ) -> crate::Result<Vec<(crate::Score, crate::DocAddress)>> {
        let field = index.schema().get_field("embedding")?;
        let mut hits = ground_truth::top_k(index, field, metric, query, 8)?;
        if let Some(filter) = filter {
            let admitted = fixture_filter_docs(index, filter)?;
            hits.retain(|(_, address)| admitted.contains(address));
        }
        hits.truncate(top_n);
        Ok(hits)
    }

    fn assert_matrix_results(
        context: &str,
        actual: &[(crate::Score, crate::DocAddress)],
        expected: &[(crate::Score, crate::DocAddress)],
        stats: &crate::vector::backend::ProbeStats,
    ) {
        if actual == expected {
            return;
        }

        let actual_docs: std::collections::HashSet<_> =
            actual.iter().map(|(_, address)| address.doc_id).collect();
        let missing = expected
            .iter()
            .map(|(_, address)| address.doc_id)
            .find(|doc| !actual_docs.contains(doc));
        let attribution = if let Some(doc) = missing {
            if !stats.quantized_trace.scored_docs.contains(&doc) {
                "routing/admission miss"
            } else if stats
                .quantized_trace
                .boundary_docs
                .iter()
                .any(|survivors| !survivors.contains(&doc))
            {
                "band drop"
            } else if !stats.quantized_trace.rerank_docs.contains(&doc) {
                "rerank fetch bug"
            } else {
                "rerank scoring/order bug"
            }
        } else {
            "rerank scoring/order bug"
        };
        panic!(
            "{context}: {attribution}; actual={actual:?} expected={expected:?} trace={:?}",
            stats.quantized_trace
        );
    }

    fn assert_quantized_matrix_storage(
        index: &Index,
        metric: Metric,
        schedule: &[u8],
    ) -> crate::Result<()> {
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let field = index.schema().get_field("embedding")?;
        let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vector_reader.index().expect("matrix fixture must be IVF");
        assert_eq!(ivf.num_rows(), 8);
        let quantized = vector_reader
            .quantization()
            .expect("matrix fixture must carry quantized slots");
        assert_eq!(
            quantized
                .config()
                .layers
                .iter()
                .map(|layer| layer.bits)
                .collect::<Vec<_>>(),
            schedule
        );
        for row in 0..ivf.num_rows() {
            assert!(quantized.residual_norm(row)?.is_finite());
            for (layer, stored) in quantized.layers().iter().enumerate() {
                assert!((1.0..=4.0).contains(&stored.gamma(row)?));
                let corrected_error = stored.error_ratio(row)?;
                assert!(corrected_error.is_finite() && corrected_error >= 0.0);
                assert_eq!(
                    stored.constant(row)?.is_some(),
                    metric == Metric::L2,
                    "layer {layer} split-constant presence must follow the metric at row {row}"
                );
            }
        }
        Ok(())
    }

    fn run_quantized_matrix_query(
        index: &Index,
        metric: Metric,
        schedule: &[u8],
        scenario: QuantizedMatrixScenario,
        depth: usize,
    ) -> crate::Result<()> {
        const TOP_N: usize = 3;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let field = index.schema().get_field("embedding")?;
        let label = index.schema().get_field("label")?;
        let query = fixture_search_query(metric, index.settings().vector_quantization[0].dim);
        let keep = TermQuery::new(
            Term::from_field_text(label, "keep"),
            IndexRecordOption::Basic,
        );
        let filter: &dyn Query = if matches!(scenario, QuantizedMatrixScenario::Filter) {
            &keep
        } else {
            &AllQuery
        };
        let expected = fixture_exact_hits(index, metric, &query, Some(filter), TOP_N)?;
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), TOP_N)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 2,
                ..Default::default()
            })
            .with_max_scan_levels(depth);
        let fruit = searcher.search(filter, &collector)?;
        assert_eq!(fruit.stats.len(), 1);
        let stats = &fruit.stats[0];
        let context =
            format!("metric={metric:?} schedule={schedule:?} scenario={scenario:?} depth={depth}");
        assert_matrix_results(&context, &fruit.results, &expected, stats);

        let layer0 = stats.layers.get(0).expect("layer 0 must execute");
        assert_eq!(
            stats.layer0_eligible,
            layer0.scored(),
            "{context}: layer 0 must score exactly the admitted eligible rows"
        );
        assert_eq!(
            stats.eligible_charged,
            layer0.scored(),
            "{context}: the probe budget must charge exactly the selected rows"
        );
        assert!(layer0.scored() > 0, "{context}: {stats:?}");
        assert!(
            layer0.survivors() <= layer0.scored(),
            "{context}: {stats:?}"
        );
        if depth == 1 {
            assert!(stats.layers.get(1).is_none(), "{context}: {stats:?}");
        } else {
            let layer1 = stats.layers.get(1).expect("layer 1 must execute");
            assert_eq!(
                layer1.scored(),
                layer0.survivors(),
                "{context}: every boundary-0 survivor must be refined"
            );
            assert!(
                layer1.survivors() <= layer1.scored(),
                "{context}: {stats:?}"
            );
        }
        if metric == Metric::L2 && matches!(scenario, QuantizedMatrixScenario::None) {
            assert!(
                layer0.survivors() < layer0.scored(),
                "{context}: the unfiltered L2 matrix cell must prove that boundary 0 measurably \
                 drops at least one scored candidate: {stats:?}"
            );
        }
        assert_eq!(
            stats.quantized_trace.boundary_docs.len(),
            depth,
            "{context}: one identity snapshot per executed boundary"
        );
        for boundary in &stats.quantized_trace.boundary_docs {
            assert!(
                boundary
                    .iter()
                    .all(|doc| stats.quantized_trace.scored_docs.contains(doc)),
                "{context}: a later layer retained a row absent from layer-0 selection"
            );
        }
        assert!(
            stats.quantized_trace.rerank_docs.iter().all(|doc| stats
                .quantized_trace
                .boundary_docs
                .last()
                .is_some_and(|boundary| boundary.contains(doc))),
            "{context}: rerank read a row absent from the final selection"
        );
        for max_probe_fraction in [0.25, 1.0] {
            const ELIGIBILITY_TOP_N: usize = 8;
            let params = AdaptiveProbeParams {
                max_probe_fraction,
                min_probe_clusters: 2,
                ..Default::default()
            };
            let quantized = searcher.search(
                filter,
                &TopDocsByVectorSimilarity::new(field, query.clone(), ELIGIBILITY_TOP_N)
                    .with_adaptive_params(params.clone())
                    .with_max_scan_levels(depth),
            )?;
            let level0 = searcher.search(
                filter,
                &TopDocsByVectorSimilarity::new(field, query.clone(), ELIGIBILITY_TOP_N)
                    .with_adaptive_params(params)
                    .with_max_scan_levels(0),
            )?;
            assert_eq!(
                quantized.stats[0].quantized_trace.scored_docs,
                level0.stats[0].quantized_trace.scored_docs,
                "{context}: selected candidate set at probe fraction {max_probe_fraction}"
            );
        }
        match scenario {
            QuantizedMatrixScenario::Filter => {
                assert!(stats.pruned_filter > 0, "{context}: {stats:?}")
            }
            QuantizedMatrixScenario::Deletes => {
                assert!(stats.pruned_dead > 0, "{context}: {stats:?}")
            }
            QuantizedMatrixScenario::None => {}
        }
        Ok(())
    }

    fn quantized_vec_file(index: &Index) -> crate::Result<Vec<u8>> {
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        Ok(searcher.segment_readers()[0]
            .open_read(SegmentComponent::Custom(VEC_EXT.to_string()))?
            .read_bytes()?
            .to_vec())
    }

    fn assert_relative_1e5(actual: f32, expected: f32, context: &str) {
        let tolerance = 1e-5 * expected.abs().max(f32::MIN_POSITIVE);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }

    fn assert_quantized_bridge_exactness(dim: usize) -> crate::Result<()> {
        let index = build_quantized_fixture(dim, true)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let field = index.schema().get_field("embedding")?;
        let vector_reader = segment.vector_index(field)?;
        let ivf = vector_reader.index().expect("merged fixture must be IVF");
        let quantized = vector_reader
            .quantization()
            .expect("configured IVF fixture must carry quantized slots");
        let (specs, grids) = quantization_runtime(quantized.config(), vector_reader.options())?;
        let query: Vec<f32> = (0..dim)
            .map(|coordinate| ((coordinate as f32 + 0.5) * 0.031).cos())
            .collect();
        let harness = cascade::prepare_split_query(&query, &specs, &grids, 4);
        let scan = QuantizedQueryCtx::new(
            QuantizedIndexCtx::resolve_from_config(quantized.config().clone()).unwrap(),
            query,
        );

        for row in 0..ivf.num_rows() {
            let mut scan_sum = 0.0;
            let mut harness_sum = 0.0;
            for (layer, stored) in quantized.layers().iter().enumerate() {
                let codes = stored.code_bytes(row)?;
                let scale = stored.scale(row)?;
                let constant = stored.constant(row)?;
                let scan_estimate = scan.score_layer(layer, &codes, scale, constant)?;
                let harness_estimate = match constant {
                    Some(constant) => {
                        harness.score_layer(layer, &codes, scale, constant, specs[layer])
                    }
                    None => {
                        harness.score_layer_without_constant(layer, &codes, scale, specs[layer])
                    }
                };
                assert_relative_1e5(
                    scan_estimate,
                    harness_estimate,
                    &format!("d={dim} row={row} layer={layer}"),
                );
                scan_sum += scan_estimate;
                harness_sum += harness_estimate;
            }
            assert_relative_1e5(scan_sum, harness_sum, &format!("d={dim} row={row} summed"));
        }
        Ok(())
    }

    #[test]
    fn bridge_exactness_d768_and_d100() -> crate::Result<()> {
        assert_quantized_bridge_exactness(768)?;
        assert_quantized_bridge_exactness(100)
    }

    #[test]
    fn vector_open_rejects_settings_file_format_mismatch() -> crate::Result<()> {
        let mut index = build_quantized_fixture(QUANT_FIXTURE_DIM, true)?;
        let field = index.schema().get_field("embedding")?;
        index.settings_mut().vector_quantization[0].format_version = 2;

        let searcher = index.reader()?.searcher();
        let message = match searcher.segment_readers()[0].vector_index(field) {
            Ok(_) => panic!("mismatched settings and `.vec` formats must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("settings format version 2 does not match `.vec` format version 3")
                && message.contains("rebuild required"),
            "unexpected error text: {message}"
        );
        Ok(())
    }

    #[test]
    fn level_zero_matches_unquantized_ivf() -> crate::Result<()> {
        const DIM: usize = 64;
        let query = vec![0.05_f32; DIM];
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.5,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let unquantized = build_quantized_fixture(DIM, false)?;
        let quantized = build_quantized_fixture(DIM, true)?;
        let field = unquantized.schema().get_field("embedding")?;
        let unquantized_fruit = unquantized.reader()?.searcher().search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(field, query.clone(), 3)
                .with_adaptive_params(params.clone()),
        )?;
        let level_zero_collector = TopDocsByVectorSimilarity::new(field, query, 3)
            .with_adaptive_params(params)
            .with_max_scan_levels(0);
        let quantized_reader = quantized.reader()?;
        let quantized_searcher = quantized_reader.searcher();
        let quantized_storage = quantized_searcher.segment_readers()[0].vector_index(field)?;
        let quantized_field = quantized_storage
            .quantization()
            .expect("fixture must carry quantized slots");
        assert!(!quantized_field.index_ctx_is_initialized());
        let level_zero_fruit = quantized_searcher.search(&AllQuery, &level_zero_collector)?;
        assert_eq!(level_zero_collector.cached_quantized_query_count(), 0);
        assert!(!quantized_field.index_ctx_is_initialized());

        assert_eq!(level_zero_fruit.results, unquantized_fruit.results);
        assert_eq!(level_zero_fruit.stats.len(), 1);
        assert_eq!(unquantized_fruit.stats.len(), 1);
        let level_zero = &level_zero_fruit.stats[0];
        let baseline = &unquantized_fruit.stats[0];
        assert!(level_zero.layers.get(0).is_none(), "{level_zero:?}");
        assert!(level_zero.routing_visited_count > 0, "{level_zero:?}");
        assert!(level_zero.clusters_probed() > 0, "{level_zero:?}");
        assert!(level_zero.candidates_scored > 0, "{level_zero:?}");
        assert!(level_zero.exact_scan_ns.is_some(), "{level_zero:?}");
        assert_eq!(level_zero.candidates_scored, baseline.candidates_scored);
        assert_eq!(level_zero.exact_rows_read, baseline.exact_rows_read);
        assert_eq!(level_zero.postings_row, baseline.postings_row);
        assert_eq!(level_zero.postings_skipped, baseline.postings_skipped);
        assert_eq!(
            level_zero.routing_visited_count,
            baseline.routing_visited_count
        );
        assert_eq!(
            level_zero.work_charged.to_bits(),
            baseline.work_charged.to_bits()
        );
        Ok(())
    }

    #[test]
    fn level_zero_flat_segment_remains_exact() -> crate::Result<()> {
        const DIM: usize = 64;
        let query = vec![0.05_f32; DIM];
        let expected = fixture_expected(&query, DIM, 3);
        let flat = build_flat_quantized_fixture(DIM)?;
        let reader = flat.reader()?;
        let searcher = reader.searcher();
        let field = flat.schema().get_field("embedding")?;
        let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
        assert!(vector_reader.index().is_none());
        assert!(vector_reader.quantization().is_none());

        let fruit = searcher.search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(field, query, 3).with_max_scan_levels(0),
        )?;
        assert_eq!(
            fruit
                .results
                .iter()
                .map(|&(score, address)| (score.to_bits(), address.doc_id))
                .collect::<Vec<_>>(),
            expected
        );
        let stats = &fruit.stats[0];
        assert_eq!(stats.exact_rows_read, 8, "{stats:?}");
        assert_eq!(stats.routing_visited_count, 0, "{stats:?}");
        assert_eq!(stats.clusters_probed(), 0, "{stats:?}");
        assert!(stats.layers.get(0).is_none(), "{stats:?}");
        Ok(())
    }

    #[test]
    fn merge_quantization_matches_kernel_harness_and_is_reproducible() -> crate::Result<()> {
        let index = build_quantized_fixture(QUANT_FIXTURE_DIM, true)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let segment = &searcher.segment_readers()[0];
        let field = index.schema().get_field("embedding")?;
        let vector_reader = segment.vector_index(field)?;
        let ivf = vector_reader.index().expect("merged fixture must be IVF");
        assert_eq!(ivf.num_rows(), 8);
        let quantized = vector_reader
            .quantization()
            .expect("configured IVF fixture must carry quantized slots");
        let (specs, grids) = quantization_runtime(quantized.config(), vector_reader.options())?;
        let centroid_bytes = ivf.centroid_bytes()?;
        let centroid_stride = QUANT_FIXTURE_DIM * std::mem::size_of::<f32>();

        for cluster in 0..ivf.num_clusters() {
            let centroid = decode_row::<f32>(
                &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
                QUANT_FIXTURE_DIM,
            )?;
            let prepared = prepare_centroid(&centroid, &specs);
            for row in ivf.cluster_range(cluster) {
                let vector = decode_row::<f32>(
                    &vector_reader.vector_bytes_for_row(row)?,
                    QUANT_FIXTURE_DIM,
                )?;
                let mut expected_input = vector.clone();
                let mut workspace = cascade::BatchEncodeWorkspace::new();
                let expected = cascade::encode_batch_in_place_with_workspace(
                    &mut expected_input,
                    1,
                    &prepared,
                    &specs,
                    &grids,
                    &mut workspace,
                    true,
                );
                assert_eq!(
                    quantized.residual_norm(row)?.to_bits(),
                    l2_squared(&vector, &centroid).to_bits()
                );
                for (layer, stored) in quantized.layers().iter().enumerate() {
                    let stored_codes = stored.code_bytes(row)?;
                    assert_eq!(
                        stored_codes.len(),
                        QUANT_FIXTURE_DIM * usize::from(specs[layer].bits) / 8,
                        "divisible dimensions use exact byte strides"
                    );
                    assert_eq!(
                        stored_codes.as_slice(),
                        expected.layers[layer].codes.as_slice()
                    );
                    assert_eq!(stored.scale(row)?, expected.layers[layer].scales[0]);
                    assert_eq!(
                        stored.gamma(row)?.to_bits(),
                        quant_model::f16::f16_to_f32(expected.layers[layer].gammas[0]).to_bits()
                    );
                    assert_eq!(
                        stored.error_ratio(row)?.to_bits(),
                        quant_model::f16::f16_to_f32(
                            expected.layers[layer].corrected_error_ratios[0]
                        )
                        .to_bits()
                    );
                    assert_eq!(
                        stored
                            .constant(row)?
                            .expect("L2 fixture requires split constants")
                            .to_bits(),
                        expected.layers[layer].constants[0].to_bits()
                    );
                }
            }
        }

        let query = vec![0.05_f32; QUANT_FIXTURE_DIM];
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 3)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 2,
                ..Default::default()
            });
        let quantized_fruit = searcher.search(&AllQuery, &collector)?;
        assert_eq!(collector.cached_quantized_query_count(), 1);
        assert!(quantized.index_ctx_is_initialized());
        assert_eq!(quantized_fruit.stats.len(), 1);
        let stats = &quantized_fruit.stats[0];
        let layer0 = stats.layers.get(0).expect("layer 0 must execute");
        let layer1 = stats.layers.get(1).expect("layer 1 must execute");
        assert!(layer0.scored() > 0, "{stats:?}");
        assert!(layer0.survivors() <= layer0.scored(), "{stats:?}");
        assert_eq!(
            layer1.scored(),
            layer0.survivors(),
            "the two-layer fixture refines every first-boundary survivor: {stats:?}"
        );
        assert!(layer1.survivors() <= layer0.survivors(), "{stats:?}");
        assert!(stats.rerank_rows <= layer1.survivors(), "{stats:?}");
        assert_eq!(stats.exact_rows_read, stats.rerank_rows, "{stats:?}");
        let hits = quantized_fruit.results;
        let mut expected: Vec<(f32, u32)> = (0..8)
            .map(|doc| {
                let center = if doc < 4 { 0.0 } else { 1.0 };
                let vector: Vec<f32> = (0..QUANT_FIXTURE_DIM)
                    .map(|coordinate| {
                        center + ((doc * QUANT_FIXTURE_DIM + coordinate) as f32 * 0.017).sin() * 0.1
                    })
                    .collect();
                (-l2_squared(&query, &vector), doc as u32)
            })
            .collect();
        expected.sort_unstable_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        assert_eq!(
            hits.iter()
                .map(|&(score, address)| (score.to_bits(), address.doc_id))
                .collect::<Vec<_>>(),
            expected[..3]
                .iter()
                .map(|&(score, doc)| (score.to_bits(), doc))
                .collect::<Vec<_>>()
        );

        let first = quantized_vec_file(&index)?;
        let second = quantized_vec_file(&build_quantized_fixture(QUANT_FIXTURE_DIM, true)?)?;
        assert_eq!(
            first, second,
            "fixed assignment and seeds must be byte-identical"
        );
        Ok(())
    }

    #[test]
    fn quantized_top_n_fixture_matrix_matches_direct_exact_oracle() -> crate::Result<()> {
        const SCHEDULES: &[&[u8]] = &[&[1], &[1, 4], &[1, 1, 4], &[2, 4]];

        for metric in [Metric::Cosine, Metric::L2] {
            for &schedule in SCHEDULES {
                let dim = 100;

                let primary = build_quantized_fixture_case(dim, metric, schedule, true)?;
                assert_quantized_matrix_storage(&primary, metric, schedule)?;
                for scenario in [
                    QuantizedMatrixScenario::None,
                    QuantizedMatrixScenario::Filter,
                ] {
                    for depth in 1..=schedule.len() {
                        run_quantized_matrix_query(&primary, metric, schedule, scenario, depth)?;
                    }
                }

                let label = primary.schema().get_field("label")?;
                let mut writer: crate::IndexWriter<TantivyDocument> =
                    primary.writer_with_num_threads(1, 30_000_000)?;
                writer.set_merge_policy(Box::new(NoMergePolicy));
                for doc in [0, 4] {
                    writer.delete_term(Term::from_field_text(label, &format!("d{doc}")));
                }
                writer.commit()?;
                drop(writer);
                for depth in 1..=schedule.len() {
                    run_quantized_matrix_query(
                        &primary,
                        metric,
                        schedule,
                        QuantizedMatrixScenario::Deletes,
                        depth,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn general_dimension_quantized_bridge_at_d100() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture(DIM, true)?;
        let reader = index.reader().map_err(|error| {
            TantivyError::InternalError(format!("general-d reader open failed: {error}"))
        })?;
        reader.reload().map_err(|error| {
            TantivyError::InternalError(format!("general-d reader reload failed: {error}"))
        })?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let field = index.schema().get_field("embedding")?;
        let vector_reader = segment.vector_index(field).map_err(|error| {
            TantivyError::InternalError(format!("general-d vector reader failed: {error}"))
        })?;
        let ivf = vector_reader.index().expect("merged fixture must be IVF");
        let quantized = vector_reader
            .quantization()
            .expect("configured IVF fixture must carry quantized slots");
        let (specs, grids) = quantization_runtime(quantized.config(), vector_reader.options())?;
        let centroid_bytes = ivf.centroid_bytes()?;
        let centroid_stride = DIM * std::mem::size_of::<f32>();

        for cluster in 0..ivf.num_clusters() {
            let centroid = decode_row::<f32>(
                &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
                DIM,
            )?;
            let prepared = prepare_centroid(&centroid, &specs);
            for row in ivf.cluster_range(cluster) {
                let vector = decode_row::<f32>(&vector_reader.vector_bytes_for_row(row)?, DIM)?;
                let residual: Vec<f32> = vector
                    .iter()
                    .zip(&centroid)
                    .map(|(&value, &center)| value - center)
                    .collect();
                let expected = cascade::encode_layers(&residual, Some(&prepared), &specs, &grids);
                for (layer, stored) in quantized.layers().iter().enumerate() {
                    assert_eq!(stored.code_bytes(row)?.as_slice(), expected.codes[layer]);
                    assert_eq!(stored.scale(row)?, expected.scales[layer]);
                    assert_eq!(
                        stored
                            .constant(row)?
                            .expect("L2 fixture requires split constants")
                            .to_bits(),
                        expected.constants[layer].to_bits()
                    );
                }
            }
        }

        let query = vec![0.05_f32; DIM];
        let quantized_hits = searcher
            .search(
                &AllQuery,
                &TopDocsByVectorSimilarity::new(field, query.clone(), 3).with_adaptive_params(
                    AdaptiveProbeParams {
                        max_probe_fraction: 1.0,
                        min_probe_clusters: 2,
                        ..Default::default()
                    },
                ),
            )?
            .results;
        assert_eq!(
            quantized_hits
                .iter()
                .map(|&(score, address)| (score.to_bits(), address.doc_id))
                .collect::<Vec<_>>(),
            fixture_expected(&query, DIM, 3)
        );
        Ok(())
    }

    #[test]
    fn l2_quantized_fixture_growth_matches_768_1_plus_4_ledger() -> crate::Result<()> {
        const DIM: usize = 768;
        const ROWS: usize = 8;
        let config = quant_fixture_config(DIM);
        assert_eq!(config.bytes_per_row(), 508);

        let plain = quantized_vec_file(&build_quantized_fixture(DIM, false)?)?;
        let quantized = quantized_vec_file(&build_quantized_fixture(DIM, true)?)?;
        let physical_growth = quantized.len() - plain.len();
        let logical_growth = ROWS * config.bytes_per_row();
        println!(
            "VECTOR_QUANTIZATION_SIZE dim={DIM} rows={ROWS} plain_bytes={} quantized_bytes={} \
             physical_growth={physical_growth} logical_growth={logical_growth}",
            plain.len(),
            quantized.len(),
        );
        assert!(physical_growth >= logical_growth);
        assert!(physical_growth <= logical_growth + 512);
        assert_eq!(logical_growth, ROWS * 508);
        Ok(())
    }

    #[test]
    fn estimator_measurement_uses_production_path_and_is_centered() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture_with_schedule(DIM, true)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let field = index.schema().get_field("embedding")?;
        let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
        assert!(vector_reader.quantization().is_some());
        let queries = fixture_estimator_queries(DIM);
        let estimator_queries = queries
            .iter()
            .cloned()
            .map(|values| VectorEstimatorQuery {
                values,
                excluded_doc_id: None,
            })
            .collect::<Vec<_>>();
        let measurements = vector_reader
            .measure_estimator_queries(
                VectorEstimatorSource::Provided,
                &estimator_queries,
                1_000,
                None,
            )?
            .expect("quantized slots remain available to explicit diagnostics");
        assert_eq!(measurements.source(), VectorEstimatorSource::Provided);
        assert_eq!(measurements.sample_rows(), 8);
        assert_eq!(measurements.query_count(), queries.len() as u32);
        assert!(measurements
            .aggregate()
            .iter()
            .all(|depth| depth.sample_count == 8 * queries.len() as u64));
        for (depth, moments) in measurements.aggregate().iter().enumerate() {
            let bias = moments
                .bias()
                .expect("fixture must produce estimator errors");
            assert!(
                bias.abs() <= 0.3,
                "depth {} normalized estimator bias {bias} exceeds 0.3",
                depth + 1
            );
        }

        let fruit = searcher.search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(field, queries[0].clone(), 3).with_adaptive_params(
                AdaptiveProbeParams {
                    max_probe_fraction: 1.0,
                    min_probe_clusters: 2,
                    ..Default::default()
                },
            ),
        )?;
        assert!(fruit.stats[0].layers.get(0).is_some());
        assert!(fruit.stats[0].postings_row > 0);
        Ok(())
    }

    #[test]
    fn estimator_samples_only_live_posting_rows() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture_with_schedule(DIM, true)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let field = index.schema().get_field("embedding")?;
        let vector_reader = segment.vector_index(field)?;
        let alive = crate::fastfield::AliveBitSet::for_test_from_deleted_docs(&[1, 3], 8);
        let live_posting_rows = (0..vector_reader.index().unwrap().num_rows())
            .filter(|&row| alive.is_alive(vector_reader.doc_id_at(row)))
            .count();
        assert_eq!(live_posting_rows, 6);

        let queries = fixture_estimator_queries(DIM);
        let measurements =
            estimator_measurements(vector_reader.as_ref(), &queries, usize::MAX, Some(&alive))?
                .unwrap();
        assert!(measurements
            .aggregate()
            .iter()
            .all(|depth| { depth.sample_count == (live_posting_rows * queries.len()) as u64 }));

        let bounded =
            estimator_measurements(vector_reader.as_ref(), &queries, 5, Some(&alive))?.unwrap();
        assert!(bounded
            .aggregate()
            .iter()
            .all(|depth| depth.sample_count == (5 * queries.len()) as u64));
        Ok(())
    }

    #[test]
    fn held_out_estimator_excludes_the_source_row() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture_with_schedule(DIM, true)?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let field = index.schema().get_field("embedding")?;
        let vector_reader = segment.vector_index(field)?;
        let queries = vector_reader
            .sample_estimator_pseudo_queries(1, segment.alive_bitset())?
            .expect("quantized fixture must support pseudo-query sampling");
        assert_eq!(queries.len(), 1);
        assert!(queries.iter().all(|query| query.excluded_doc_id.is_some()));

        let measurements = vector_reader
            .measure_estimator_queries(
                VectorEstimatorSource::HeldOut,
                &queries,
                usize::MAX,
                segment.alive_bitset(),
            )?
            .expect("quantized fixture must support estimator measurement");
        assert_eq!(measurements.source(), VectorEstimatorSource::HeldOut);
        assert_eq!(measurements.sample_rows(), 8);
        assert_eq!(measurements.query_count(), 1);
        assert!(measurements
            .aggregate()
            .iter()
            .all(|moments| moments.sample_count == 7));
        Ok(())
    }
}
