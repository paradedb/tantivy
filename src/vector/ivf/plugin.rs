//! IVF-format merge routine.
//!
//! The IVF format is one of two storage modes the unified
//! [`VectorPlugin`](crate::vector::VectorPlugin) can produce per merge.
//! This module exposes the merge body so the parent plugin can call it
//! after the threshold check.

use std::io::Write;
use std::time::{Duration, Instant};

use cascade::{
    encode_batch_in_place_with_residual_observer, prepare_centroid, prepare_fp_query, LayerSpec,
};
use quant_model::f16::f16_to_f32;
use quant_model::{build_grid, Grid, DEFAULT_CAL};

use super::{
    decode_row, encode_vector, BuiltRouter, IvfCentroids, IvfClusterer, IvfIndex, IvfMatrix,
    IvfMatrixView, IvfTrainingBatch, IvfTrainingVectors, IvfVectorBatch, IvfVectors, RoutingIndex,
    CENTROIDS_EXT,
};
use crate::directory::{CompositeWrite, Directory};
use crate::index::SegmentComponent;
use crate::plugin::PluginMergeContext;
use crate::schema::{Field, FieldType, Metric, VectorDType, VectorOptions};
use crate::vector::distance::{l2_squared, maybe_normalize_bytes, NormalizeOutcome};
use crate::vector::flat::IdMap;
use crate::vector::header::{centroid_slot, vec_slot, write_header, CURRENT, HEADER_LEN};
use crate::vector::quantization::{
    VectorQuantizationCalibration, VectorQuantizationDepthCalibration,
};
use crate::vector::{
    quantized_code_stride, residual_norm, BoundKind, BoundsBuilder, NeighborhoodGraphConfig,
    RelativeNeighborhoodGraph, VectorQuantizationConfig, VectorQuantizer,
    QUANTIZED_CODE_ALIGNMENT, VEC_EXT,
};
use crate::{DocId, Executor, TantivyError};

struct AssignedVector {
    cluster: usize,
    target_doc_id: DocId,
    source_segment_ord: usize,
    source_doc_id: DocId,
}

/// A multi-threaded [`Executor`] when the host has the parallelism, the
/// single-threaded one otherwise.
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

struct QuantizedLayerSlots {
    codes: Vec<u8>,
    scales: Vec<u16>,
    constants: Vec<f32>,
}

const QUANTIZATION_CALIBRATION_SAMPLE_ROWS: usize = 1_024;
// Production-query calibration measured from 1,000 Cohere scan rows. The
// held-out stored-vector protocol measures encoder error but underestimates
// the query-distribution tail seen by the real scan workload. Persist this
// observed floor until an index build has a larger field-local measurement.
const REAL_QUERY_CALIBRATION: VectorQuantizationDepthCalibration =
    VectorQuantizationDepthCalibration {
        bias: 0.0,
        cal: 2.265_918_2,
        sample_count: 1_000,
    };

struct QuantizationCalibrator {
    interval: usize,
    rows_seen: usize,
    stored_query: Vec<CalibrationMeasurement>,
    gaussian_query: Vec<CalibrationMeasurement>,
    heldout_query: Vec<CalibrationMeasurement>,
    heldout_query_by_source: Vec<Vec<CalibrationMeasurement>>,
}

#[derive(Clone, Default)]
struct CalibrationMeasurement {
    sample_count: usize,
    empirical_squared_sum: f64,
    empirical_model_cross_sum: f64,
    model_variance_sum: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct BiasStability {
    query_count: usize,
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
}

struct QuantizationCalibrationReport {
    stored_query: Vec<VectorQuantizationDepthCalibration>,
    gaussian_query: Vec<VectorQuantizationDepthCalibration>,
    heldout_query: Vec<VectorQuantizationDepthCalibration>,
    heldout_bias_stability: Vec<BiasStability>,
}

fn persisted_depth_calibration(
    heldout: VectorQuantizationDepthCalibration,
) -> VectorQuantizationDepthCalibration {
    let heldout_envelope = f64::from(heldout.bias).hypot(f64::from(heldout.cal));
    let real_query_envelope =
        f64::from(REAL_QUERY_CALIBRATION.bias).hypot(f64::from(REAL_QUERY_CALIBRATION.cal));
    if heldout.sample_count == 0 || heldout_envelope >= real_query_envelope {
        heldout
    } else {
        REAL_QUERY_CALIBRATION
    }
}

impl CalibrationMeasurement {
    fn observe(
        &mut self,
        error: &[f32],
        scale: u16,
        final_rho: f64,
        query: &[f32],
        query_norm: f32,
    ) {
        let dot_error = error
            .iter()
            .zip(query)
            .map(|(&value, &query)| f64::from(value) * f64::from(query))
            .sum::<f64>();
        let model_sigma = f64::from(f16_to_f32(scale)) * final_rho * f64::from(query_norm);
        self.empirical_squared_sum += dot_error.powi(2);
        self.empirical_model_cross_sum += dot_error * model_sigma;
        self.model_variance_sum += model_sigma.powi(2);
        self.sample_count += 1;
    }

    fn finish(self) -> VectorQuantizationDepthCalibration {
        if self.sample_count == 0 || self.model_variance_sum == 0.0 {
            return VectorQuantizationDepthCalibration {
                bias: 0.0,
                cal: DEFAULT_CAL as f32,
                sample_count: self.sample_count as u32,
            };
        }
        let bias = self.empirical_model_cross_sum / self.model_variance_sum;
        let centered_squared_sum = (self.empirical_squared_sum
            - 2.0 * bias * self.empirical_model_cross_sum
            + bias.powi(2) * self.model_variance_sum)
            .max(0.0);
        let cal = (centered_squared_sum / self.model_variance_sum).sqrt();
        VectorQuantizationDepthCalibration {
            bias: bias as f32,
            cal: cal as f32,
            sample_count: self.sample_count as u32,
        }
    }
}

impl QuantizationCalibrator {
    fn new(num_rows: usize, layer_count: usize, heldout_query_count: usize) -> Self {
        Self {
            interval: num_rows
                .div_ceil(QUANTIZATION_CALIBRATION_SAMPLE_ROWS)
                .max(1),
            rows_seen: 0,
            stored_query: vec![CalibrationMeasurement::default(); layer_count],
            gaussian_query: vec![CalibrationMeasurement::default(); layer_count],
            heldout_query: vec![CalibrationMeasurement::default(); layer_count],
            heldout_query_by_source: vec![
                vec![
                    CalibrationMeasurement::default();
                    heldout_query_count
                ];
                layer_count
            ],
        }
    }

    /// `errors` is one prefix's rotated reconstruction residual. `query` is a
    /// build-sampled real vector in that exact coordinate space, so every
    /// accumulator matches the scan's prefix-local sigma chain.
    fn observe_layer(
        &mut self,
        layer: usize,
        row_base: usize,
        errors: &[f32],
        scales: &[u16],
        rho: f64,
        stored_query: (&[f32], f32),
        gaussian_query: (&[f32], f32),
        heldout_query: Option<(usize, &[f32], f32)>,
    ) {
        debug_assert_eq!(errors.len(), scales.len() * stored_query.0.len());
        for (local_row, (error, &scale)) in errors
            .chunks_exact(stored_query.0.len())
            .zip(scales)
            .enumerate()
        {
            let sample = (row_base + local_row).is_multiple_of(self.interval)
                && self.stored_query[layer].sample_count < QUANTIZATION_CALIBRATION_SAMPLE_ROWS;
            if !sample {
                continue;
            }
            self.stored_query[layer].observe(error, scale, rho, stored_query.0, stored_query.1);
            self.gaussian_query[layer].observe(
                error,
                scale,
                rho,
                gaussian_query.0,
                gaussian_query.1,
            );
            if let Some((query_index, query, query_norm)) = heldout_query {
                self.heldout_query[layer].observe(error, scale, rho, query, query_norm);
                self.heldout_query_by_source[layer][query_index]
                    .observe(error, scale, rho, query, query_norm);
            }
        }
    }

    fn rows_seen(&self) -> usize {
        self.rows_seen
    }

    fn advance(&mut self, rows: usize) {
        self.rows_seen += rows;
    }

    fn finish(self) -> QuantizationCalibrationReport {
        let heldout_bias_stability = self
            .heldout_query_by_source
            .into_iter()
            .map(|queries| {
                let biases = queries
                    .into_iter()
                    .filter(|query| query.sample_count > 0 && query.model_variance_sum > 0.0)
                    .map(|query| query.finish().bias as f64)
                    .collect::<Vec<_>>();
                if biases.is_empty() {
                    return BiasStability::default();
                }
                let mean = biases.iter().sum::<f64>() / biases.len() as f64;
                let stddev = (biases.iter().map(|bias| (bias - mean).powi(2)).sum::<f64>()
                    / biases.len() as f64)
                    .sqrt();
                BiasStability {
                    query_count: biases.len(),
                    mean,
                    stddev,
                    min: biases.iter().copied().fold(f64::INFINITY, f64::min),
                    max: biases.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                }
            })
            .collect();
        QuantizationCalibrationReport {
            stored_query: self
                .stored_query
                .into_iter()
                .map(CalibrationMeasurement::finish)
                .collect(),
            gaussian_query: self
                .gaussian_query
                .into_iter()
                .map(CalibrationMeasurement::finish)
                .collect(),
            heldout_query: self
                .heldout_query
                .into_iter()
                .map(CalibrationMeasurement::finish)
                .collect(),
            heldout_bias_stability,
        }
    }
}

fn deterministic_gaussian_query(dim: usize) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(0x4341_4c2d_4741_5553);
    let mut query = Vec::with_capacity(dim);
    while query.len() < dim {
        let u1 = rng.f64().max(f64::MIN_POSITIVE);
        let u2 = rng.f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        query.push((radius * angle.cos()) as f32);
        if query.len() < dim {
            query.push((radius * angle.sin()) as f32);
        }
    }
    query
}

fn prepare_calibration_query(
    query: &[f32],
    metric: Metric,
    centroid: &[f32],
    specs: &[LayerSpec],
) -> (cascade::PreparedFpQuery, f32) {
    let mut score_query = query.to_vec();
    if metric == Metric::L2 {
        for (value, &center) in score_query.iter_mut().zip(centroid) {
            *value -= center;
        }
    }
    let query_norm = score_query
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    (prepare_fp_query(&score_query, specs), query_norm)
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
            let modeled = build_grid(config.dim, layer.bits);
            match layer.quantizer {
                // The sign encoder does not consume grid points. This format-shaped
                // placeholder keeps the width-generic cascade API allocation-free.
                VectorQuantizer::RaBitQ => Grid {
                    bits: 1,
                    points: vec![-1.0, 1.0],
                    rho_model: modeled.rho_model,
                },
                VectorQuantizer::TurboQuant => {
                    let grid = config
                        .grids
                        .iter()
                        .find(|grid| grid.bits == layer.bits)
                        .expect("validated TurboQuant grid must be present");
                    Grid {
                        bits: grid.bits,
                        points: grid.points.clone(),
                        rho_model: modeled.rho_model,
                    }
                }
            }
        })
        .collect();
    Ok((specs, grids))
}

fn write_quantized_slots(
    vec_write: &mut CompositeWrite,
    field: Field,
    layers: &[QuantizedLayerSlots],
    residual_norms: Option<&[f32]>,
    calibration: &VectorQuantizationCalibration,
) -> crate::Result<()> {
    for (layer, encoded) in layers.iter().enumerate() {
        vec_write.align_next_field(QUANTIZED_CODE_ALIGNMENT, HEADER_LEN)?;
        {
            let writer = vec_write.for_field_with_idx(field, vec_slot::quantized_codes(layer));
            writer.write_all(&encoded.codes)?;
            writer.flush()?;
        }
        {
            let writer = vec_write.for_field_with_idx(field, vec_slot::quantized_scales(layer));
            for &scale in &encoded.scales {
                writer.write_all(&scale.to_le_bytes())?;
            }
            writer.flush()?;
        }
        {
            let writer = vec_write.for_field_with_idx(field, vec_slot::quantized_constants(layer));
            for &constant in &encoded.constants {
                writer.write_all(&constant.to_le_bytes())?;
            }
            writer.flush()?;
        }
    }
    if let Some(residual_norms) = residual_norms {
        let writer = vec_write.for_field_with_idx(field, vec_slot::QUANTIZED_RESIDUAL_NORMS);
        for &residual_norm in residual_norms {
            writer.write_all(&residual_norm.to_le_bytes())?;
        }
        writer.flush()?;
    }
    {
        let writer = vec_write.for_field_with_idx(field, vec_slot::QUANTIZED_CALIBRATION);
        writer.write_all(&calibration.encode())?;
        writer.flush()?;
    }
    Ok(())
}

/// Write `field`'s slots in both composites as an empty IVF field: empty
/// Explicit id-map, empty rows, zero centroids, zero docs, and a single
/// zero cluster offset. Every vector field must own its slots in every
/// IVF segment — the reader treats a missing slot as corruption — so both
/// the "sources report no vectors" fast path and the "every vector-bearing
/// doc is deleted" path write this same shape.
fn write_empty_field_slots(
    vec_write: &mut CompositeWrite,
    centroids_write: &mut CompositeWrite,
    field: Field,
    opts: &VectorOptions,
    quantization: Option<&VectorQuantizationConfig>,
) -> crate::Result<()> {
    // `.vec`: empty Explicit id-map + empty rows.
    {
        let id_map_w = vec_write.for_field_with_idx(field, vec_slot::ID_MAP);
        IdMap::serialize_explicit(&[], id_map_w)?;
        id_map_w.flush()?;
    }
    {
        let rows_w = vec_write.for_field_with_idx(field, vec_slot::ROWS);
        rows_w.flush()?;
    }
    if let Some(config) = quantization {
        let layers: Vec<QuantizedLayerSlots> = config
            .layers
            .iter()
            .map(|_| QuantizedLayerSlots {
                codes: Vec::new(),
                scales: Vec::new(),
                constants: Vec::new(),
            })
            .collect();
        let residual_norms = config.needs_residual_norm().then_some([].as_slice());
        write_quantized_slots(
            vec_write,
            field,
            &layers,
            residual_norms,
            &VectorQuantizationCalibration {
                depths: vec![
                    VectorQuantizationDepthCalibration {
                        bias: 0.0,
                        cal: DEFAULT_CAL as f32,
                        sample_count: 0,
                    };
                    config.layers.len()
                ],
            },
        )?;
    }
    // `.centroids`: zero centroids, zero docs, single zero offset, and an
    // empty (but present — the slot is mandatory in V2) bounds slot.
    {
        let centroids_w = centroids_write.for_field_with_idx(field, centroid_slot::CENTROIDS);
        IvfIndex::serialize_centroids(0, 0, &[], opts, centroids_w)?;
        centroids_w.flush()?;
    }
    {
        let offsets_w = centroids_write.for_field_with_idx(field, centroid_slot::OFFSETS);
        IvfIndex::serialize_offsets(&[0u64], offsets_w)?;
        offsets_w.flush()?;
    }
    {
        let bounds_w = centroids_write.for_field_with_idx(field, centroid_slot::BOUNDS);
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
    write_header(&mut vec_file)?;
    let mut vec_write = CompositeWrite::wrap(vec_file);
    let mut centroids_file = directory.open_write(&centroids_path)?;
    write_header(&mut centroids_file)?;
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
        // Per-segment readers for this field (cached on the SegmentReaders).
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
        let mut current_centroid: Vec<f32> = Vec::new();

        match opts.dtype() {
            VectorDType::F32 => {
                let field_build_start = Instant::now();
                let mut timings = IvfBuildTimings::default();
                let mut training_values = Vec::with_capacity(training_sample_size * opts.dim());
                let mut training_doc_ids = Vec::with_capacity(training_sample_size);
                let mut target_doc_id: DocId = 0;
                let mut present_vector_ord = 0usize;
                let mut sampled_count = 0usize;
                for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                    let reader = &field_readers[old_doc_addr.segment_ord as usize];
                    if let Some(bytes) = reader.vector_bytes(old_doc_addr.doc_id)? {
                        let should_sample = sampled_count < training_sample_size
                            && present_vector_ord % training_sample_interval == 0;
                        if should_sample {
                            training_doc_ids.push(target_doc_id);
                            training_values
                                .extend_from_slice(&decode_row::<f32>(&bytes, opts.dim())?);
                            sampled_count += 1;
                        }
                        present_vector_ord += 1;
                    }
                    target_doc_id += 1;
                }
                debug_assert_eq!(target_doc_id, num_target_docs);
                // Rows written for docs deleted afterwards still count toward
                // the sources' `count()` (neither layout rewrites `.vec` on
                // delete), so the alive-doc walk can come up short of
                // `vector_count` — never over. Equality holds exactly when no
                // source carries deletes.
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
                    // `vector_count > 0`, yet the alive-doc walk found
                    // nothing to sample: every vector-bearing doc was
                    // deleted. Write the same empty slots as the
                    // no-vectors fast path — skipping the field would
                    // leave its slots missing from composites the other
                    // fields still write, and the reader errors on
                    // missing slots.
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
                            // Poll for cancellation once per batch so a large
                            // assign phase (minutes on a big segment) stays
                            // interruptible instead of only checking at phase
                            // boundaries.
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
                    for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                        let reader = &field_readers[old_doc_addr.segment_ord as usize];
                        if let Some(bytes) = reader.vector_bytes(old_doc_addr.doc_id)? {
                            batch_doc_ids.push(target_doc_id);
                            batch_values.extend_from_slice(&decode_row::<f32>(&bytes, opts.dim())?);
                            batch_sources.push((
                                target_doc_id,
                                old_doc_addr.segment_ord as usize,
                                old_doc_addr.doc_id,
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
                // Same alive-doc walk as the training pass above — it must
                // have found the same vectors (deletes make both fall short
                // of `vector_count` together, so compare them to each other).
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

                // `.centroids` slot [0] payload, built BEFORE the posting
                // rows so the bounds fold below measures residuals against
                // the STORED centroid. K-means cluster means are not
                // unit-norm; for Cosine+F32 normalize each centroid here so
                // the search path can score both docs and centroids with
                // the same `dot * inv_norm_q` fast kernel.
                let mut centroid_bytes =
                    Vec::with_capacity(num_centroids * opts.bytes_per_vector());
                // P1: `BoundsBuilder` is the ONLY producer of bounds. The
                // fold runs over THIS merge's re-assignment output against
                // the NEW centroids — combining the sources' stored bounds
                // would be unsound (their centroids no longer exist), which
                // is why no bound-combining API exists.
                let mut bounds_builder = BoundsBuilder::new(num_centroids);
                for (centroid_ord, centroid) in centroid_rows.iter().enumerate() {
                    let mut bytes = encode_vector(centroid, opts.dim())?;
                    // Centroids are means of ingest-validated rows, so
                    // NonFinite is should-never-happen; same warn-and-write
                    // policy as the posting rows below.
                    let outcome = maybe_normalize_bytes(opts, &mut bytes);
                    if outcome == NormalizeOutcome::NonFinite {
                        log::warn!(
                            "non-finite centroid {centroid_ord} in field '{}' written \
                             un-normalized during merge",
                            entry.name(),
                        );
                    }
                    let stored = decode_row::<f32>(&bytes, opts.dim())?;
                    // A degenerate centroid — non-finite, or zero-norm under
                    // cosine renormalization — anchors no residual geometry:
                    // SATURATE, so the cluster always probes. (A non-finite
                    // centroid would also self-saturate through non-finite
                    // residuals, but only if the cluster has members.)
                    if outcome != NormalizeOutcome::Normalized
                        || stored.iter().any(|value| !value.is_finite())
                    {
                        bounds_builder.saturate(centroid_ord);
                    }
                    centroid_bytes.extend_from_slice(&bytes);
                }

                let posting_start = Instant::now();
                // `.vec` slot [0]: the row→doc_id permutation (Explicit), in
                // cluster-sorted row order — parallel to the rows in slot [1].
                {
                    let id_map_w = vec_write.for_field_with_idx(field, vec_slot::ID_MAP);
                    let row_doc_ids: Vec<DocId> = assigned_vectors
                        .iter()
                        .map(|assigned_vector| assigned_vector.target_doc_id)
                        .collect();
                    IdMap::serialize_explicit(&row_doc_ids, id_map_w)?;
                    id_map_w.flush()?;
                }

                // `.vec` slot [1]: the cluster-sorted vector rows.
                {
                    // Poll for cancellation every this-many rows during the
                    // posting-write phase — often enough to stay responsive,
                    // rare enough to keep the FFI cancel check off the per-row
                    // path.
                    const CANCEL_POLL_ROWS: usize = 4096;
                    let rows_w = vec_write.for_field_with_idx(field, vec_slot::ROWS);
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
                        // Sources are already unit-normalized at ingest for
                        // Cosine+F32 (see `FlatVecWriter`), but re-normalize on
                        // the way into the cluster rows so the IVF invariant —
                        // the query path scores pre-normalized rows — holds
                        // locally, even for a source segment written before
                        // ingest-time normalization existed. Idempotent. L2/Dot
                        // don't normalize and write the source bytes directly;
                        // Cosine+F32 copies into one buffer reused across rows.
                        //
                        // Ingest rejects non-finite vectors, so NonFinite here
                        // is a should-never-happen path: erroring would wedge
                        // merge retries forever on one poison doc, and dropping
                        // the row would desync the already-computed assignments
                        // and IdMap. Warn-and-write-as-is is visible,
                        // self-limiting, and non-desyncing.
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
                        // P1: the bounds fold — every written row, using the exact
                        // bytes written above against the stored centroid. A
                        // non-finite row residual saturates its cluster
                        // inside `add_native`.
                        if assigned_vector.cluster != current_cluster {
                            current_cluster = assigned_vector.cluster;
                            current_centroid = decode_row::<f32>(
                                &centroid_bytes[current_cluster * centroid_stride..]
                                    [..centroid_stride],
                                opts.dim(),
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
                    let num_rows = assigned_vectors.len();
                    let mut encoded_layers: Vec<QuantizedLayerSlots> = config
                        .layers
                        .iter()
                        .map(|layer| QuantizedLayerSlots {
                            codes: Vec::with_capacity(
                                num_rows * quantized_code_stride(opts.dim(), layer.bits),
                            ),
                            scales: Vec::with_capacity(num_rows),
                            constants: Vec::with_capacity(num_rows),
                        })
                        .collect();
                    let mut residual_norms = config
                        .needs_residual_norm()
                        .then(|| Vec::with_capacity(num_rows));

                    // Two row-major f32 tile buffers live in the cascade
                    // encoder. Rotation adds one d-sized transient buffer.
                    const MAX_QUANTIZATION_SCRATCH_BYTES: usize = 1 << 20;
                    let row_bytes = opts.dim() * std::mem::size_of::<f32>();
                    let tile_rows = MAX_QUANTIZATION_SCRATCH_BYTES
                        .saturating_sub(row_bytes)
                        .checked_div(2 * row_bytes)
                        .unwrap_or(0)
                        .max(1);
                    let needs_norm = opts.needs_normalization();
                    let mut normalized = Vec::with_capacity(opts.bytes_per_vector());
                    let mut batch_values = Vec::with_capacity(tile_rows * opts.dim());
                    let load_calibration_query =
                        |assigned_vector: &AssignedVector| -> crate::Result<Vec<f32>> {
                            let reader = &field_readers[assigned_vector.source_segment_ord];
                            let bytes = reader
                                .vector_bytes(assigned_vector.source_doc_id)?
                                .ok_or_else(|| {
                                    TantivyError::InternalError(format!(
                                        "missing source vector for doc {:?}",
                                        assigned_vector.source_doc_id
                                    ))
                                })?;
                            let mut bytes = bytes.to_vec();
                            if needs_norm {
                                let _ = maybe_normalize_bytes(opts, &mut bytes);
                            }
                            decode_row::<f32>(&bytes, opts.dim())
                        };
                    let stored_calibration_query = load_calibration_query(&assigned_vectors[0])?;
                    let gaussian_calibration_query = deterministic_gaussian_query(opts.dim());
                    const CALIBRATION_PSEUDO_QUERIES: usize = 64;
                    let native_rows: Vec<_> = assigned_vectors
                        .iter()
                        .filter(|assigned| assigned.native)
                        .collect();
                    let pseudo_query_count = native_rows.len().min(CALIBRATION_PSEUDO_QUERIES);
                    let mut heldout_calibration_queries = Vec::with_capacity(pseudo_query_count);
                    for sample in 0..pseudo_query_count {
                        let assigned = native_rows[sample * native_rows.len() / pseudo_query_count];
                        heldout_calibration_queries
                            .push((assigned.cluster, load_calibration_query(assigned)?));
                    }
                    let mut calibrator = QuantizationCalibrator::new(
                        num_rows,
                        specs.len(),
                        heldout_calibration_queries.len(),
                    );

                    for (cluster, offsets) in cluster_offsets.windows(2).enumerate() {
                        let start = offsets[0] as usize;
                        let end = offsets[1] as usize;
                        if start == end {
                            continue;
                        }
                        let centroid = decode_row::<f32>(
                            &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
                            opts.dim(),
                        )?;
                        let prepared = prepare_centroid(&centroid, &specs);
                        let stored_calibration_query = prepare_calibration_query(
                            &stored_calibration_query,
                            opts.metric(),
                            &centroid,
                            &specs,
                        );
                        let gaussian_calibration_query = prepare_calibration_query(
                            &gaussian_calibration_query,
                            opts.metric(),
                            &centroid,
                            &specs,
                        );
                        let heldout_calibration_query = (!heldout_calibration_queries.is_empty())
                            .then(|| {
                                (0..heldout_calibration_queries.len())
                                    .map(|offset| {
                                        (cluster + offset) % heldout_calibration_queries.len()
                                    })
                                    .find_map(|query_idx| {
                                        let (query_cluster, query) =
                                            &heldout_calibration_queries[query_idx];
                                        (*query_cluster != cluster).then(|| {
                                            (
                                                query_idx,
                                                prepare_calibration_query(
                                                    query,
                                                    opts.metric(),
                                                    &centroid,
                                                    &specs,
                                                ),
                                            )
                                        })
                                    })
                            })
                            .flatten();
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
                                batch_values.extend_from_slice(&decode_row::<f32>(
                                    encoded_bytes,
                                    opts.dim(),
                                )?);
                            }
                            if let Some(residual_norms) = residual_norms.as_mut() {
                                residual_norms.extend(
                                    batch_values
                                        .chunks_exact(opts.dim())
                                        .map(|row| l2_squared(row, &centroid)),
                                );
                            }
                            let row_base = calibrator.rows_seen();
                            let batch = encode_batch_in_place_with_residual_observer(
                                &mut batch_values,
                                tile.len(),
                                &prepared,
                                &specs,
                                &grids,
                                |layer, errors, scales| {
                                    calibrator.observe_layer(
                                        layer,
                                        row_base,
                                        errors,
                                        scales,
                                        grids[layer].rho_model,
                                        (
                                            stored_calibration_query.0.layer(layer),
                                            stored_calibration_query.1,
                                        ),
                                        (
                                            gaussian_calibration_query.0.layer(layer),
                                            gaussian_calibration_query.1,
                                        ),
                                        heldout_calibration_query.as_ref().map(
                                            |(query_index, query)| {
                                                (*query_index, query.0.layer(layer), query.1)
                                            },
                                        ),
                                    );
                                },
                            );
                            calibrator.advance(tile.len());
                            for (target, layer) in encoded_layers.iter_mut().zip(batch.layers) {
                                target.codes.extend_from_slice(&layer.codes);
                                target.scales.extend_from_slice(&layer.scales);
                                target.constants.extend_from_slice(&layer.constants);
                            }
                        }
                    }
                    let calibration_report = calibrator.finish();
                    // Persist a centered field-local model when its total
                    // bias+spread envelope is larger than the production-query
                    // floor. The floor remains conservative when the build
                    // sample cannot characterize the production query center.
                    let calibration = VectorQuantizationCalibration {
                        depths: calibration_report
                            .heldout_query
                            .iter()
                            .copied()
                            .map(persisted_depth_calibration)
                            .collect(),
                    };
                    write_quantized_slots(
                        &mut vec_write,
                        field,
                        &encoded_layers,
                        residual_norms.as_deref(),
                        &calibration,
                    )?;
                    for (layer, ((((stored, gaussian), heldout), stability), persisted)) in
                        calibration_report
                            .stored_query
                            .iter()
                            .zip(&calibration_report.gaussian_query)
                            .zip(&calibration_report.heldout_query)
                            .zip(&calibration_report.heldout_bias_stability)
                            .zip(&calibration.depths)
                            .enumerate()
                    {
                        log::info!(
                            target: "paradedb::ivf_build",
                            "quantization_calibration field={} depth={} bias={} cal={} samples={} \
                             stored_query_bias={} stored_query_cal={} stored_query_samples={} \
                             gaussian_query_bias={} gaussian_query_cal={} \
                             gaussian_query_samples={} heldout_query_bias={} \
                             heldout_query_cal={} heldout_query_samples={} \
                             heldout_bias_queries={} heldout_bias_mean={} \
                             heldout_bias_stddev={} heldout_bias_min={} heldout_bias_max={}",
                            entry.name(),
                            layer + 1,
                            persisted.bias,
                            persisted.cal,
                            persisted.sample_count,
                            stored.bias,
                            stored.cal,
                            stored.sample_count,
                            gaussian.bias,
                            gaussian.cal,
                            gaussian.sample_count,
                            heldout.bias,
                            heldout.cal,
                            heldout.sample_count,
                            stability.query_count,
                            stability.mean,
                            stability.stddev,
                            stability.min,
                            stability.max,
                        );
                    }
                    timings.quantize = quantize_start.elapsed();
                }

                {
                    let centroids_w =
                        centroids_write.for_field_with_idx(field, centroid_slot::CENTROIDS);
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
                        centroids_write.for_field_with_idx(field, centroid_slot::OFFSETS);
                    IvfIndex::serialize_offsets(&cluster_offsets, offsets_w)?;
                    offsets_w.flush()?;
                }
                // `.centroids` slot [3]: the per-cluster centroid bounds
                // this merge's fold produced.
                {
                    let bounds_w = centroids_write.for_field_with_idx(field, centroid_slot::BOUNDS);
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
    use std::sync::Arc;

    use super::*;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::AllQuery;
    use crate::schema::Schema;
    use crate::vector::ivf::AdaptiveProbeParams;
    use crate::vector::prepared::{QuantizedIndexCtx, QuantizedQueryCtx};
    use crate::vector::{
        TopDocsByVectorSimilarity, VectorNormPolicy, VectorQuantizationGrid,
        VectorQuantizationLayer, GRID_FORMAT_VERSION, VECTOR_QUANTIZATION_FORMAT_VERSION,
    };
    use crate::{Index, TantivyDocument};

    #[test]
    fn build_calibration_matches_the_scan_sigma_ratio() {
        let errors = [1.0_f32, 2.0, 3.0, 4.0];
        let query = [0.6_f32, 0.8];
        let scales = [
            quant_model::f16::f32_to_f16(0.5),
            quant_model::f16::f32_to_f16(0.25),
        ];
        let rho = 0.1;
        let mut calibrator = QuantizationCalibrator::new(2, 1, 1);
        calibrator.observe_layer(
            0,
            0,
            &errors,
            &scales,
            rho,
            (&query, 1.0),
            (&query, 1.0),
            Some((0, &query, 1.0)),
        );
        let measured = calibrator.finish();

        let model_variance = (0.5_f64 * rho).powi(2) + (0.25_f64 * rho).powi(2);
        let cross = 2.2 * 0.5 * rho + 5.0 * 0.25 * rho;
        let bias = cross / model_variance;
        let centered = (2.2_f64.powi(2) + 5.0_f64.powi(2) - 2.0 * bias * cross
            + bias.powi(2) * model_variance)
            / model_variance;
        assert!((f64::from(measured.stored_query[0].bias) - bias).abs() < 1e-4);
        assert!((f64::from(measured.stored_query[0].cal) - centered.sqrt()).abs() < 1e-4);
        assert_eq!(measured.stored_query[0].sample_count, 2);
        assert_eq!(measured.gaussian_query, measured.stored_query);
        assert_eq!(measured.heldout_query, measured.stored_query);
    }

    #[test]
    fn per_query_bias_stability_tracks_independent_query_centers() {
        let query_x = [1.0_f32, 0.0];
        let query_y = [0.0_f32, 1.0];
        let scale = quant_model::f16::f32_to_f16(1.0);
        let mut calibrator = QuantizationCalibrator::new(4, 1, 2);
        calibrator.observe_layer(
            0,
            0,
            &[1.0, 0.0, 2.0, 0.0],
            &[scale, scale],
            0.1,
            (&query_x, 1.0),
            (&query_x, 1.0),
            Some((0, &query_x, 1.0)),
        );
        calibrator.advance(2);
        calibrator.observe_layer(
            0,
            2,
            &[0.0, 1.0, 0.0, 2.0],
            &[scale, scale],
            0.1,
            (&query_y, 1.0),
            (&query_y, 1.0),
            Some((1, &query_y, 1.0)),
        );
        let stability = calibrator.finish().heldout_bias_stability[0];
        assert_eq!(stability.query_count, 2);
        assert!((stability.mean - 15.0).abs() < 1e-5);
        assert!(stability.stddev < 1e-6);
        assert!((stability.min - stability.max).abs() < 1e-6);
    }

    #[test]
    fn persisted_calibration_uses_the_real_query_envelope() {
        let below = VectorQuantizationDepthCalibration {
            bias: 0.0,
            cal: 1.08,
            sample_count: 1_024,
        };
        assert_eq!(persisted_depth_calibration(below), REAL_QUERY_CALIBRATION);

        let above = VectorQuantizationDepthCalibration {
            bias: -2.0,
            cal: 1.5,
            sample_count: 777,
        };
        assert_eq!(persisted_depth_calibration(above), above);

        let empty = VectorQuantizationDepthCalibration {
            bias: 0.0,
            cal: 1.0,
            sample_count: 0,
        };
        assert_eq!(persisted_depth_calibration(empty), empty);
    }

    const QUANT_FIXTURE_DIM: usize = 64;

    struct QuantFixtureClusterer {
        dim: usize,
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
            let values = [0.0_f32, 1.0]
                .into_iter()
                .flat_map(|center| std::iter::repeat_n(center, self.dim))
                .collect();
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
                .map(|row| u32::from(row[0] >= 0.5))
                .collect())
        }
    }

    fn quant_fixture_config(dim: usize) -> VectorQuantizationConfig {
        let grid = quant_model::build_grid(dim, 4);
        VectorQuantizationConfig {
            field: "embedding".to_string(),
            format_version: VECTOR_QUANTIZATION_FORMAT_VERSION,
            dim,
            metric: Metric::L2,
            norm_policy: VectorNormPolicy::None,
            layers: vec![
                VectorQuantizationLayer {
                    bits: 1,
                    quantizer: VectorQuantizer::RaBitQ,
                    seed: 0x1111,
                },
                VectorQuantizationLayer {
                    bits: 4,
                    quantizer: VectorQuantizer::TurboQuant,
                    seed: 0x2222,
                },
            ],
            grids: vec![VectorQuantizationGrid {
                bits: 4,
                version: GRID_FORMAT_VERSION,
                points: grid.points,
                rho_model: Some(grid.rho_model),
            }],
        }
    }

    fn build_quantized_fixture(dim: usize, quantized: bool) -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let field =
            schema_builder.add_vector_field("embedding", VectorOptions::new(dim, Metric::L2));
        let schema = schema_builder.build();
        let mut settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..Default::default()
        };
        if quantized {
            settings.vector_quantization = vec![quant_fixture_config(dim)];
        }
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(QuantFixtureClusterer { dim }))
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

    fn fixture_hits(
        index: &Index,
        query: Vec<f32>,
        level_zero: bool,
    ) -> crate::Result<Vec<(u32, u32)>> {
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let field = index.schema().get_field("embedding")?;
        let mut collector = TopDocsByVectorSimilarity::new(field, query, 3).with_adaptive_params(
            AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 2,
                ..Default::default()
            },
        );
        if level_zero {
            collector = collector.with_max_scan_levels(0);
        }
        Ok(searcher
            .search(&AllQuery, &collector)?
            .results
            .iter()
            .map(|&(score, address)| (score.to_bits(), address.doc_id))
            .collect())
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
            Arc::new(QuantizedIndexCtx::new(quantized.config().clone())),
            query,
        );

        for row in 0..ivf.num_rows() {
            let mut scan_sum = 0.0;
            let mut harness_sum = 0.0;
            for (layer, stored) in quantized.layers().iter().enumerate() {
                let codes = stored.code_bytes(row)?;
                let scale = stored.scale(row)?;
                let constant = stored.constant(row)?;
                let scan_estimate = scan.score_layer(layer, &codes, scale, constant);
                let harness_estimate =
                    harness.score_layer(layer, &codes, scale, constant, specs[layer]);
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
    fn gate_a_bridge_exactness_d768_and_d100() -> crate::Result<()> {
        assert_quantized_bridge_exactness(768)?;
        assert_quantized_bridge_exactness(100)
    }

    #[test]
    fn gate_c_exact_path_equivalence() -> crate::Result<()> {
        const DIM: usize = 64;
        let query = vec![0.05_f32; DIM];
        let expected = fixture_expected(&query, DIM, 3);

        let opted_out = build_quantized_fixture(DIM, false)?;
        assert_eq!(fixture_hits(&opted_out, query.clone(), false)?, expected);

        let no_slot = build_flat_quantized_fixture(DIM)?;
        let reader = no_slot.reader()?;
        let searcher = reader.searcher();
        let field = no_slot.schema().get_field("embedding")?;
        let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
        assert!(vector_reader.index().is_none());
        assert!(vector_reader.quantization().is_none());
        assert_eq!(fixture_hits(&no_slot, query.clone(), false)?, expected);

        let quantized = build_quantized_fixture(DIM, true)?;
        assert_eq!(fixture_hits(&quantized, query, true)?, expected);
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
                let residual: Vec<f32> = vector
                    .iter()
                    .zip(&centroid)
                    .map(|(&value, &center)| value - center)
                    .collect();
                let expected = cascade::encode_layers(&residual, Some(&prepared), &specs, &grids);
                assert_eq!(
                    quantized
                        .residual_norm(row)?
                        .expect("L2 fixture requires residual norm")
                        .to_bits(),
                    l2_squared(&vector, &centroid).to_bits()
                );
                for (layer, stored) in quantized.layers().iter().enumerate() {
                    let stored_codes = stored.code_bytes(row)?;
                    assert_eq!(
                        stored_codes.len(),
                        QUANT_FIXTURE_DIM * usize::from(specs[layer].bits) / 8,
                        "divisible-d V3 fixture must retain its original row stride"
                    );
                    assert_eq!(stored_codes.as_slice(), expected.codes[layer]);
                    assert_eq!(stored.scale(row)?, expected.scales[layer]);
                    assert_eq!(
                        stored.constant(row)?.to_bits(),
                        expected.constants[layer].to_bits()
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
        let exact_hits = searcher
            .search(
                &AllQuery,
                &TopDocsByVectorSimilarity::new(field, query.clone(), 3).with_max_scan_levels(0),
            )?
            .results;
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
        assert_eq!(hits, exact_hits, "level zero must preserve the exact scan");

        let first = quantized_vec_file(&index)?;
        let second = quantized_vec_file(&build_quantized_fixture(QUANT_FIXTURE_DIM, true)?)?;
        assert_eq!(
            first, second,
            "fixed assignment and seeds must be byte-identical"
        );
        Ok(())
    }

    #[test]
    fn general_dimension_quantized_bridge_at_d100() -> crate::Result<()> {
        const DIM: usize = 100;
        let index = build_quantized_fixture(DIM, true)?;
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
                        stored.constant(row)?.to_bits(),
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
        let exact_hits = searcher
            .search(
                &AllQuery,
                &TopDocsByVectorSimilarity::new(field, query, 3).with_max_scan_levels(0),
            )?
            .results;
        assert_eq!(quantized_hits, exact_hits);
        Ok(())
    }

    #[test]
    fn quantized_fixture_growth_matches_768_1_plus_4_ledger() -> crate::Result<()> {
        const DIM: usize = 768;
        const ROWS: usize = 8;
        let config = quant_fixture_config(DIM);
        assert_eq!(config.bytes_per_row(), 496);

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
        assert!(
            (logical_growth as f64 / (ROWS * DIM * 4) as f64 - 0.161_458_333_333).abs() < 1e-12
        );
        Ok(())
    }
}
