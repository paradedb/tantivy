//! Reading and decoding one quantized layer for a batch.

use std::ops::Range;

use quant_model::f16::f16_to_f32;

use crate::error::DataCorruption;
use crate::schema::Metric;
use crate::vector::index_reader::{
    validate_decoded_sidecar, QuantizedLayerBatch, QuantizedLayerReader,
};
use crate::vector::prepared::QuantizedQueryCtx;
use crate::TantivyError;

/// Storage-row selection resolved before a quantized layer is read.
pub(in crate::vector) enum Selection<'a> {
    All,
    Rows(&'a [usize]),
    None,
}

impl Selection<'_> {
    #[inline]
    pub(in crate::vector) fn len(&self, rows: &Range<usize>) -> usize {
        match self {
            Self::All => rows.len(),
            Self::Rows(offsets) => offsets.len(),
            Self::None => 0,
        }
    }
}

/// Decodes a binary16 run into binary32 scratch storage.
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

/// Reads and scores one selected layer range.
#[inline(always)]
pub(in crate::vector) fn score_layer(
    query: &QuantizedQueryCtx,
    layer_idx: usize,
    layer: &QuantizedLayerReader,
    first_layer: Option<(&QuantizedLayerBatch, &mut Vec<f32>)>,
    known_block: Option<usize>,
    rows: Range<usize>,
    selection: &Selection<'_>,
    kernel_scores: &mut Vec<f32>,
    decoded_scales: &mut Vec<f32>,
    decoded_gammas: &mut Vec<f32>,
    decoded_error_ratios: &mut Vec<f32>,
    decoded_constants: &mut Vec<f32>,
    read_ranges: &mut Vec<Range<usize>>,
    selected_rows: &mut Vec<usize>,
    row_offsets: &mut Vec<usize>,
) -> crate::Result<usize> {
    let (pinned, residual_norms) =
        first_layer.map_or((None, None), |(batch, norms)| (Some(batch), Some(norms)));
    let metric = query.index.meta.field().metric();
    let selected_count = selection.len(&rows);
    if selected_count == 0 {
        unreachable!("empty selections are skipped before scoring");
    }
    kernel_scores.resize(selected_count, 0.0);
    decoded_scales.resize(selected_count, 0.0);
    decoded_gammas.resize(selected_count, 0.0);
    decoded_error_ratios.resize(selected_count, 0.0);
    if metric == Metric::L2 {
        decoded_constants.resize(selected_count, 0.0);
    }

    if matches!(selection, Selection::All) {
        let first_row = rows.start;
        let owned_batch;
        let batch = match pinned {
            Some(batch) => batch,
            None => {
                owned_batch = match known_block {
                    Some(block) => layer.read_batch_in_block(block, rows)?,
                    None => layer.read_batch(rows)?,
                };
                &owned_batch
            }
        };
        if let Some(out) = residual_norms {
            out.resize(selected_count, 0.0);
            decode_f32s(
                batch
                    .residual_norms
                    .as_ref()
                    .expect("layer zero band contains norms"),
                out,
            );
        }
        query.score_layer_batch_unscaled(
            layer_idx,
            batch.codes(),
            batch.code_stride(),
            &mut kernel_scores[..selected_count],
        );
        decode_f32s(batch.scales(), decoded_scales);
        decode_f16s(batch.gammas(), decoded_gammas);
        decode_f16s(batch.error_ratios(), decoded_error_ratios);
        validate_decoded_sidecar(
            &decoded_gammas[..selected_count],
            &decoded_error_ratios[..selected_count],
            first_row,
        )?;
        if metric == Metric::L2 {
            let constants = batch.constants().ok_or_else(|| {
                TantivyError::DataCorruption(DataCorruption::comment_only(
                    "quantized L2 field is missing a constants slot",
                ))
            })?;
            decode_f32s(constants, decoded_constants);
        }
        return Ok(selected_count);
    }

    let Selection::Rows(offsets) = selection else {
        unreachable!("empty selections are skipped before scoring");
    };
    if let Some(batch) = pinned {
        query.score_layer_batch_unscaled_indexed(
            layer_idx,
            batch.codes(),
            batch.code_stride(),
            offsets,
            kernel_scores,
        );
        let f32_at = |bytes: &[u8], row: usize| {
            f32::from_le_bytes(bytes[row * 4..row * 4 + 4].try_into().unwrap())
        };
        let f16_at = |bytes: &[u8], row: usize| {
            f16_to_f32(u16::from_le_bytes(
                bytes[row * 2..row * 2 + 2].try_into().unwrap(),
            ))
        };
        for (i, &row) in offsets.iter().enumerate() {
            decoded_scales[i] = f32_at(batch.scales(), row);
            decoded_gammas[i] = f16_at(batch.gammas(), row);
            decoded_error_ratios[i] = f16_at(batch.error_ratios(), row);
            if metric == Metric::L2 {
                decoded_constants[i] = f32_at(batch.constants().expect("L2 constants"), row);
            }
        }
        if let Some(out) = residual_norms {
            out.clear();
            out.extend(offsets.iter().map(|&row| {
                f32_at(
                    batch.residual_norms.as_ref().expect("layer zero norms"),
                    row,
                )
            }));
        }
        validate_decoded_sidecar(decoded_gammas, decoded_error_ratios, rows.start)?;
        return Ok(selected_count);
    }
    debug_assert!(offsets.windows(2).all(|pair| pair[0] < pair[1]));
    debug_assert!(offsets.iter().all(|&offset| offset < rows.len()));
    selected_rows.clear();
    selected_rows.extend(offsets.iter().map(|&offset| rows.start + offset));

    // Selected rows and their code ranges are ordered, so each cluster can be consumed
    // directly without allocating request/view lists or sorting them.
    let mut cluster_start = 0;
    while cluster_start < selected_count {
        let cluster = match known_block {
            Some(block) => layer.cluster_in_block(block),
            None => layer.cluster(selected_rows[cluster_start]),
        }?;
        let cluster_end = cluster_start
            + selected_rows[cluster_start..].partition_point(|&row| row < cluster.rows.end);
        cluster.plan_codes(&selected_rows[cluster_start..cluster_end], read_ranges);
        let mut selected_start = cluster_start;
        for read_range in read_ranges.iter().cloned() {
            let mut selected_end = selected_start;
            while selected_end < cluster_end && selected_rows[selected_end] < read_range.end {
                debug_assert!(selected_rows[selected_end] >= read_range.start);
                selected_end += 1;
            }
            row_offsets.clear();
            row_offsets.extend(
                selected_rows[selected_start..selected_end]
                    .iter()
                    .map(|&row| row - read_range.start),
            );
            let codes = cluster.read_codes(read_range)?;
            query.score_layer_batch_unscaled_indexed(
                layer_idx,
                codes.as_slice(),
                layer.code_stride(),
                row_offsets,
                &mut kernel_scores[selected_start..selected_end],
            );
            selected_start = selected_end;
        }
        debug_assert_eq!(selected_start, cluster_end);
        cluster.read_sidecar()?.decode_selected(
            &selected_rows[cluster_start..cluster_end],
            &mut decoded_scales[cluster_start..cluster_end],
            &mut decoded_gammas[cluster_start..cluster_end],
            &mut decoded_error_ratios[cluster_start..cluster_end],
            if metric == Metric::L2 {
                &mut decoded_constants[cluster_start..cluster_end]
            } else {
                &mut []
            },
        )?;
        cluster_start = cluster_end;
    }
    // For sparse selections, the reported error row is approximate after the first selected row.
    validate_decoded_sidecar(
        &decoded_gammas[..selected_count],
        &decoded_error_ratios[..selected_count],
        selected_rows[0],
    )?;

    Ok(selected_count)
}

pub(in crate::vector) fn candidate_selection<'a>(
    candidate_rows: &[usize],
    available_rows: &Range<usize>,
    offsets: &'a mut Vec<usize>,
) -> Selection<'a> {
    debug_assert!(!candidate_rows.is_empty());
    debug_assert!(candidate_rows.windows(2).all(|pair| pair[0] < pair[1]));
    debug_assert!(candidate_rows
        .iter()
        .all(|row| available_rows.contains(row)));
    offsets.clear();
    offsets.extend(candidate_rows.iter().map(|&row| row - available_rows.start));
    Selection::Rows(offsets)
}
