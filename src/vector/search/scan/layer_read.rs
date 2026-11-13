//! Reading and decoding one quantized layer for a cluster's selected rows.

use quant::model::f16::{decode_f16s_to_f32s, f16_to_f32};

use crate::error::DataCorruption;
use crate::schema::Metric;
use crate::vector::search::cluster_source::ClusterRows;
use crate::vector::search::prepared::QuantizedQueryCtx;
use crate::vector::storage::quantized::{
    validate_decoded_sidecar, QuantizedClusterReader, QuantizedLayerBatch, QuantizedLayerReader,
    ReadScratch,
};
use crate::TantivyError;

/// One quantized layer as the scan reads and combines it.
#[derive(Clone, Copy)]
pub(crate) struct LayerSpec<'a> {
    pub(crate) query: &'a QuantizedQueryCtx,
    pub(crate) reader: &'a QuantizedLayerReader,
    pub(crate) index: usize,
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
    /// The code runs planned in scratch, the sidecar span and, at layer 0, the norms column.
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
    /// One whole band column, decoded before its selected rows are gathered.
    pub(crate) band_column: Vec<f32>,
}

impl Decoded {
    /// Sizes every column for `rows` selected rows.
    fn resize(&mut self, rows: usize, metric: Metric) {
        self.kernel.resize(rows, 0.0);
        self.scales.resize(rows, 0.0);
        self.gammas.resize(rows, 0.0);
        self.errors.resize(rows, 0.0);
        if metric.scores_by_distance() {
            self.constants.resize(rows, 0.0);
        }
    }
}

// --- Decoding ---

/// Decodes `spec` for the rows of `selected` into `out`, reading as `read` says from `cluster`, the
/// cluster `selected` names, as pricing resolved it.
///
/// A sparse read uses the rows and runs pricing left in `scratch`; a whole-cluster selection always
/// reads the band.
pub(crate) fn decode_layer(
    spec: &LayerSpec<'_>,
    selected: &ClusterRows<'_>,
    read: LayerRead,
    cluster: &QuantizedClusterReader<'_>,
    out: &mut Decoded,
    scratch: &mut ReadScratch,
) -> crate::Result<()> {
    debug_assert!(selected.len() > 0, "empty clusters never reach the scan");
    out.resize(selected.len(), spec.metric());
    match (read, selected.offsets) {
        (LayerRead::Sparse, Some(_)) => decode_runs(spec, cluster, out, scratch),
        _ => decode_band(spec, selected, cluster, out, &mut scratch.offsets),
    }
}

/// Reads the cluster's band and decodes the selected rows from it; `kernel_offsets` holds the
/// selected rows' offsets as the scoring kernel indexes them.
fn decode_band(
    spec: &LayerSpec<'_>,
    selected: &ClusterRows<'_>,
    cluster: &QuantizedClusterReader<'_>,
    out: &mut Decoded,
    kernel_offsets: &mut Vec<usize>,
) -> crate::Result<()> {
    let band = cluster.read_batch(selected.rows.clone())?;
    let metric = spec.metric();
    let count = selected.len();
    match selected.offsets {
        None => {
            if spec.reader.has_norms() {
                out.norms.resize(count, 0.0);
                decode_f32s(band_norms(&band)?, &mut out.norms);
            }
            spec.query.score_layer_batch_unscaled_fastscan(
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
            if metric.scores_by_distance() {
                decode_f32s(band_constants(&band)?, &mut out.constants);
            }
        }
        Some(offsets) => {
            kernel_offsets.clear();
            kernel_offsets.extend(offsets.iter().map(|&offset| offset as usize));
            decode_band_rows(spec, &band, kernel_offsets, out, selected.rows.start)?
        }
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
    let band_rows = band.scales().len() / std::mem::size_of::<f32>();
    decode_sidecar_rows(
        band,
        offsets,
        spec.metric().scores_by_distance(),
        spec.reader.has_norms(),
        SidecarDecode::for_selection(offsets.len(), band_rows),
        out,
    )?;
    validate_decoded_sidecar(&out.gammas, &out.errors, start)
}

/// How the sidecar of a band's selected rows is decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SidecarDecode {
    /// Each column decoded whole by the batch decoders, then the selected rows gathered.
    Gather,
    /// Each selected row decoded on its own.
    PerRow,
}

impl SidecarDecode {
    /// Whole columns once the selection covers a quarter of the band: the batch decoders then
    /// cost less than the per-row conversions they replace.
    fn for_selection(selected: usize, band_rows: usize) -> Self {
        if 4 * selected >= band_rows {
            SidecarDecode::Gather
        } else {
            SidecarDecode::PerRow
        }
    }
}

/// Decodes the scales, gammas, error ratios and, for L2, constants of the rows at `offsets` from a
/// read band into `out`, with the residual norms when the band carries them. Both ways give the
/// same values: binary16 widens to binary32 exactly.
fn decode_sidecar_rows(
    band: &QuantizedLayerBatch,
    offsets: &[usize],
    l2: bool,
    norms: bool,
    how: SidecarDecode,
    out: &mut Decoded,
) -> crate::Result<()> {
    let constants = l2.then(|| band_constants(band)).transpose()?;
    let norms = norms.then(|| band_norms(band)).transpose()?;
    if norms.is_some() {
        out.norms.resize(offsets.len(), 0.0);
    }
    match how {
        SidecarDecode::Gather => {
            let Decoded {
                scales,
                gammas,
                errors,
                constants: constants_out,
                norms: norms_out,
                band_column,
                ..
            } = out;
            gather(band_column, offsets, scales, |column| {
                decode_f32s(band.scales(), column)
            });
            gather(band_column, offsets, gammas, |column| {
                decode_f16s(band.gammas(), column)
            });
            gather(band_column, offsets, errors, |column| {
                decode_f16s(band.error_ratios(), column)
            });
            if let Some(constants) = constants {
                gather(band_column, offsets, constants_out, |column| {
                    decode_f32s(constants, column)
                });
            }
            if let Some(norms) = norms {
                gather(band_column, offsets, norms_out, |column| {
                    decode_f32s(norms, column)
                });
            }
        }
        SidecarDecode::PerRow => {
            for (i, &row) in offsets.iter().enumerate() {
                out.scales[i] = f32_at(band.scales(), row);
                out.gammas[i] = f16_at(band.gammas(), row);
                out.errors[i] = f16_at(band.error_ratios(), row);
                if let Some(constants) = constants {
                    out.constants[i] = f32_at(constants, row);
                }
                if let Some(norms) = norms {
                    out.norms[i] = f32_at(norms, row);
                }
            }
        }
    }
    Ok(())
}

/// Decodes one whole column into `column` with `decode`, then copies its rows at `offsets` into
/// `out`, in order.
#[inline(always)]
fn gather(
    column: &mut Vec<f32>,
    offsets: &[usize],
    out: &mut [f32],
    decode: impl FnOnce(&mut Vec<f32>),
) {
    decode(column);
    for (out, &row) in out.iter_mut().zip(offsets) {
        *out = column[row];
    }
}

/// Reads the planned code runs, the sidecar span and, at layer 0, the norms column, and decodes
/// the selected rows.
fn decode_runs(
    spec: &LayerSpec<'_>,
    cluster: &QuantizedClusterReader<'_>,
    out: &mut Decoded,
    scratch: &mut ReadScratch,
) -> crate::Result<()> {
    let ReadScratch {
        rows,
        runs,
        offsets,
        ..
    } = scratch;
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
    let l2 = spec.metric().scores_by_distance();
    cluster.read_sidecar()?.decode_selected(
        rows,
        &mut out.scales,
        &mut out.gammas,
        &mut out.errors,
        if l2 { &mut out.constants } else { &mut [] },
    )?;
    if spec.reader.has_norms() {
        let norms = cluster.read_norms()?;
        out.norms.clear();
        out.norms.extend(
            rows.iter()
                .map(|&row| f32_at(&norms, row - cluster.rows.start)),
        );
    }
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
    decode_f16s_to_f32s(values, decoded);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::test_support::{first_segment, FixtureSpec};

    /// The gathered and per-row sidecar decodes give bit-identical rows, with and without L2
    /// constants and residual norms, for selections on both sides of the density switch.
    #[test]
    fn gathered_sidecar_matches_per_row_decode() -> crate::Result<()> {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let mut both = [false; 2];
        for metric in [Metric::L2, Metric::Dot] {
            let fixture = FixtureSpec::clustered(metric, 64, 2, 400)
                .schedule(&[1, 4])
                .build()?;
            let searcher = fixture.index.reader()?.searcher();
            let reader = first_segment(&searcher).1.vector_index(fixture.field)?;
            let rows = reader.index().expect("IVF segment").cluster_num_rows(0)?;
            for layer in reader.quantization()?.unwrap().layers() {
                let band = layer.cluster_in_block(0)?.read_batch(0..rows)?;
                for step in [1, 2, 3, 4, 5, 9, 40] {
                    let offsets: Vec<usize> = (step / 2..rows).step_by(step).collect();
                    let decode = |how| -> crate::Result<Decoded> {
                        let mut out = Decoded::default();
                        out.resize(offsets.len(), metric);
                        decode_sidecar_rows(
                            &band,
                            &offsets,
                            metric.scores_by_distance(),
                            layer.has_norms(),
                            how,
                            &mut out,
                        )?;
                        Ok(out)
                    };
                    let (gathered, per_row) = (
                        decode(SidecarDecode::Gather)?,
                        decode(SidecarDecode::PerRow)?,
                    );
                    let case = format!("{metric:?} layer={} step={step}", layer.layer);
                    assert_eq!(bits(&gathered.scales), bits(&per_row.scales), "{case}");
                    assert_eq!(bits(&gathered.gammas), bits(&per_row.gammas), "{case}");
                    assert_eq!(bits(&gathered.errors), bits(&per_row.errors), "{case}");
                    assert_eq!(
                        bits(&gathered.constants),
                        bits(&per_row.constants),
                        "{case}"
                    );
                    assert_eq!(bits(&gathered.norms), bits(&per_row.norms), "{case}");
                    assert_eq!(
                        gathered.norms.len(),
                        offsets.len() * usize::from(layer.has_norms())
                    );
                    let how = SidecarDecode::for_selection(offsets.len(), rows);
                    both[usize::from(how == SidecarDecode::Gather)] = true;
                }
            }
        }
        assert_eq!(both, [true, true], "both decodes are chosen");
        Ok(())
    }
}
