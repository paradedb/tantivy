//! Reading and decoding one quantized layer for a cluster's selected rows.

use quant_model::f16::f16_to_f32;

use crate::error::DataCorruption;
use crate::schema::Metric;
use crate::vector::cluster_source::ClusterRows;
use crate::vector::index_reader::{
    validate_decoded_sidecar, QuantizedLayerBatch, QuantizedLayerReader, ReadScratch,
};
use crate::vector::prepared::QuantizedQueryCtx;
use crate::TantivyError;

/// One quantized layer as the scan reads and combines it.
#[derive(Clone, Copy)]
pub(crate) struct LayerSpec<'a> {
    pub(crate) query: &'a QuantizedQueryCtx,
    pub(crate) reader: &'a QuantizedLayerReader,
    pub(crate) index: usize,
    /// Whether the layer's band carries the rows' residual norms: layer 0 only.
    pub(crate) has_norms: bool,
    /// The layer's own squared query-quantization error: zero for a grid layer.
    pub(crate) query_error_squared: f32,
}

impl LayerSpec<'_> {
    fn metric(&self) -> Metric {
        self.query.index.meta.field().metric()
    }
}

/// How a cluster's selected rows are read at one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerRead {
    /// The cluster's whole band, in one request.
    Band,
    /// The planned code runs in scratch and the sidecar span.
    Sparse,
}

/// One layer decoded for a cluster's selected rows, one entry per row.
#[derive(Default)]
pub struct Decoded {
    /// Unscaled kernel scores.
    pub(crate) kernel: Vec<f32>,
    pub(crate) scales: Vec<f32>,
    pub(crate) gammas: Vec<f32>,
    /// Corrected-error ratios.
    pub(crate) errors: Vec<f32>,
    /// L2 constants; empty for other metrics.
    pub(crate) constants: Vec<f32>,
    /// Residual norms, from a band that carries them; empty otherwise.
    pub(crate) norms: Vec<f32>,
}

impl Decoded {
    /// Sizes every column for `rows` selected rows.
    fn resize(&mut self, rows: usize, metric: Metric) {
        self.kernel.resize(rows, 0.0);
        self.scales.resize(rows, 0.0);
        self.gammas.resize(rows, 0.0);
        self.errors.resize(rows, 0.0);
        if metric == Metric::L2 {
            self.constants.resize(rows, 0.0);
        }
    }
}

// --- Planning ---

/// Plans the code runs a sparse read of `selected` requests, into `scratch`.
pub(crate) fn plan_runs(
    spec: &LayerSpec<'_>,
    selected: &ClusterRows<'_>,
    scratch: &mut ReadScratch,
) -> crate::Result<()> {
    scratch.rows.clear();
    match selected.offsets {
        None => scratch.rows.extend(selected.rows.clone()),
        Some(offsets) => scratch
            .rows
            .extend(offsets.iter().map(|&offset| selected.rows.start + offset)),
    }
    spec.reader
        .cluster_in_block(selected.cluster)?
        .plan_codes(&scratch.rows, &mut scratch.runs);
    Ok(())
}

// --- Decoding ---

/// Decodes `spec` for the rows of `selected` into `out`, reading as `read` says.
///
/// A sparse read uses the runs [`plan_runs`] left in `scratch`; a whole-cluster selection always
/// reads the band.
pub(crate) fn decode_layer(
    spec: &LayerSpec<'_>,
    selected: &ClusterRows<'_>,
    read: LayerRead,
    out: &mut Decoded,
    scratch: &mut ReadScratch,
) -> crate::Result<()> {
    debug_assert!(selected.len() > 0, "empty clusters never reach the scan");
    out.resize(selected.len(), spec.metric());
    match (read, selected.offsets) {
        (LayerRead::Sparse, Some(_)) => decode_runs(spec, selected, out, scratch),
        _ => decode_band(spec, selected, out),
    }
}

/// Reads the cluster's band and decodes the selected rows from it.
fn decode_band(
    spec: &LayerSpec<'_>,
    selected: &ClusterRows<'_>,
    out: &mut Decoded,
) -> crate::Result<()> {
    let band = spec
        .reader
        .read_batch_in_block(selected.cluster, selected.rows.clone())?;
    let metric = spec.metric();
    let count = selected.len();
    match selected.offsets {
        None => {
            if spec.has_norms {
                out.norms.resize(count, 0.0);
                decode_f32s(band_norms(&band)?, &mut out.norms);
            }
            spec.query.score_layer_batch_unscaled(
                spec.index,
                band.codes(),
                band.code_stride(),
                &mut out.kernel[..count],
            );
            decode_f32s(band.scales(), &mut out.scales);
            decode_f16s(band.gammas(), &mut out.gammas);
            decode_f16s(band.error_ratios(), &mut out.errors);
            validate_decoded_sidecar(
                &out.gammas[..count],
                &out.errors[..count],
                selected.rows.start,
            )?;
            if metric == Metric::L2 {
                decode_f32s(band_constants(&band)?, &mut out.constants);
            }
        }
        Some(offsets) => decode_band_rows(spec, &band, offsets, out, selected.rows.start)?,
    }
    Ok(())
}

/// Decodes the rows at `offsets` from a read band; `start` is the band's first row.
fn decode_band_rows(
    spec: &LayerSpec<'_>,
    band: &QuantizedLayerBatch,
    offsets: &[usize],
    out: &mut Decoded,
    start: usize,
) -> crate::Result<()> {
    spec.query.score_layer_batch_unscaled_indexed(
        spec.index,
        band.codes(),
        band.code_stride(),
        offsets,
        &mut out.kernel,
    );
    let l2 = spec.metric() == Metric::L2;
    for (i, &row) in offsets.iter().enumerate() {
        out.scales[i] = f32_at(band.scales(), row);
        out.gammas[i] = f16_at(band.gammas(), row);
        out.errors[i] = f16_at(band.error_ratios(), row);
        if l2 {
            out.constants[i] = f32_at(band_constants(band)?, row);
        }
    }
    if spec.has_norms {
        let norms = band_norms(band)?;
        out.norms.clear();
        out.norms
            .extend(offsets.iter().map(|&row| f32_at(norms, row)));
    }
    validate_decoded_sidecar(&out.gammas, &out.errors, start)
}

/// Reads the planned code runs and the sidecar span, and decodes the selected rows.
fn decode_runs(
    spec: &LayerSpec<'_>,
    selected: &ClusterRows<'_>,
    out: &mut Decoded,
    scratch: &mut ReadScratch,
) -> crate::Result<()> {
    let ReadScratch {
        rows,
        runs,
        offsets,
        ..
    } = scratch;
    let cluster = spec.reader.cluster_in_block(selected.cluster)?;
    debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
    let mut start = 0;
    for run in runs.iter().cloned() {
        let end = start + rows[start..].partition_point(|&row| row < run.end);
        debug_assert!(rows[start..end].iter().all(|&row| row >= run.start));
        offsets.clear();
        offsets.extend(rows[start..end].iter().map(|&row| row - run.start));
        let codes = cluster.read_codes(run)?;
        spec.query.score_layer_batch_unscaled_indexed(
            spec.index,
            codes.as_slice(),
            spec.reader.code_stride(),
            offsets,
            &mut out.kernel[start..end],
        );
        start = end;
    }
    debug_assert_eq!(start, rows.len());
    let l2 = spec.metric() == Metric::L2;
    cluster.read_sidecar()?.decode_selected(
        rows,
        &mut out.scales,
        &mut out.gammas,
        &mut out.errors,
        if l2 { &mut out.constants } else { &mut [] },
    )?;
    // The reported error row is the first selected row.
    validate_decoded_sidecar(&out.gammas, &out.errors, rows[0])
}

fn band_norms(band: &QuantizedLayerBatch) -> crate::Result<&[u8]> {
    band.residual_norms.as_deref().ok_or_else(|| {
        TantivyError::DataCorruption(DataCorruption::comment_only(
            "quantized layer-0 band is missing residual norms",
        ))
    })
}

fn band_constants(band: &QuantizedLayerBatch) -> crate::Result<&[u8]> {
    band.constants().ok_or_else(|| {
        TantivyError::DataCorruption(DataCorruption::comment_only(
            "quantized L2 field is missing a constants slot",
        ))
    })
}

fn f32_at(bytes: &[u8], row: usize) -> f32 {
    f32::from_le_bytes(bytes[row * 4..row * 4 + 4].try_into().unwrap())
}

fn f16_at(bytes: &[u8], row: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes(
        bytes[row * 2..row * 2 + 2].try_into().unwrap(),
    ))
}

/// Decodes a binary16 run into binary32.
#[inline(always)]
pub(super) fn decode_f16s(values: &[u8], decoded: &mut Vec<f32>) {
    assert_eq!(values.len() % std::mem::size_of::<u16>(), 0);
    decoded.resize(values.len() / std::mem::size_of::<u16>(), 0.0);
    for (out, bytes) in decoded.iter_mut().zip(values.chunks_exact(2)) {
        let bits = bytes[0] as u16 | (bytes[1] as u16) << 8;
        *out = f16_to_f32(bits);
    }
}

/// Decodes a little-endian binary32 run.
#[inline(always)]
pub(super) fn decode_f32s(bytes: &[u8], decoded: &mut Vec<f32>) {
    assert_eq!(bytes.len() % std::mem::size_of::<f32>(), 0);
    decoded.resize(bytes.len() / std::mem::size_of::<f32>(), 0.0);
    for (out, bytes) in decoded.iter_mut().zip(bytes.chunks_exact(4)) {
        let bits = bytes[0] as u32
            | (bytes[1] as u32) << 8
            | (bytes[2] as u32) << 16
            | (bytes[3] as u32) << 24;
        *out = f32::from_bits(bits);
    }
}
