//! 4-bit scalar quantization (SQ4) row sketches — the certified per-row
//! gate's storage.
//!
//! `.vec` slot `[2]` (see `vec_slot::SQ`), written by the IVF merge, holds
//! for every posting row a nibble-packed 4-bit code per dimension plus one
//! f32 reconstruction-error bound, preceded by the segment-wide per-dim
//! grid:
//!
//! ```text
//! [lo:   f32 * dim]                     // grid origin per dimension
//! [step: f32 * dim]                     // grid step per dimension
//! [records: num_rows * record_bytes]    // one PER-ROW record:
//!     [codes: ceil(dim/2) B]            //   nibble-packed 4-bit codes
//!     [e: f32]                          //   e >= ||x - x̂||, rounded UP
//! ```
//!
//! Records are row-major in cluster-sorted row order with the error
//! INTERLEAVED after its codes, so gating a probed cluster is ONE
//! contiguous ranged read — codes and errors arrive together instead of
//! a second (page-touch-sized) read for 4 bytes per row.
//!
//! The gate's proof is the triangle inequality: `x` lies within `e` of the
//! reconstruction `x̂`, so any similarity upper bound computed from `x̂`
//! plus `e` certifies "this row cannot beat the current kth" — and only
//! then is the full-precision row read skipped. Soundness rests on two
//! write-time invariants:
//!
//! 1. The stored `e` is computed against the EXACT reconstruction the query side will produce —
//!    same f32 grid values, same `lo + q * step` arithmetic in f64 — and rounded up
//!    ([`round_error_up`]).
//! 2. Rows outside the grid (the grid is sampled, not exact) clamp to the nearest level; their `e`
//!    simply measures the larger miss. Loose, never wrong.
//!
//! Absence of the slot means no gate — flat segments and pre-SQ segments
//! read every survivor row exactly as before.

use std::io::{self, Write};

use common::{BinarySerializable, HasLen, OwnedBytes};

use crate::directory::FileSlice;
use crate::schema::Metric;

/// Quantization levels per dimension (4 bits).
const LEVELS: u32 = 15;

/// Bytes per row of nibble-packed codes.
pub(crate) fn code_stride(dim: usize) -> usize {
    dim.div_ceil(2)
}

/// Bytes per full row record: codes plus the interleaved f32 error.
pub(crate) fn record_bytes(dim: usize) -> usize {
    code_stride(dim) + 4
}

/// The segment-wide per-dimension grid. Built from a SAMPLE of the
/// segment's rows; rows outside a dimension's sampled range clamp, which
/// the per-row error absorbs.
pub(crate) struct SqGrid {
    lo: Vec<f32>,
    step: Vec<f32>,
}

impl SqGrid {
    /// Fold `sample` rows into per-dim `[min, max]` and derive the grid.
    /// Non-finite sampled coordinates degrade that dimension to a
    /// constant-zero grid line (still sound: the row error measures the
    /// full miss).
    pub(crate) fn from_sample<'a>(dim: usize, sample: impl Iterator<Item = &'a [f32]>) -> SqGrid {
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for row in sample {
            for d in 0..dim {
                let v = row[d];
                if v.is_finite() {
                    lo[d] = lo[d].min(v);
                    hi[d] = hi[d].max(v);
                }
            }
        }
        let step: Vec<f32> = lo
            .iter_mut()
            .zip(hi)
            .map(|(lo, hi)| {
                if !lo.is_finite() || !hi.is_finite() {
                    *lo = 0.0;
                    return 0.0;
                }
                let step = (hi - *lo) / LEVELS as f32;
                if step.is_finite() {
                    step
                } else {
                    0.0
                }
            })
            .collect();
        SqGrid { lo, step }
    }

    /// Reconstructed value of `code` on dimension `d`, in f64 — the ONE
    /// definition both the encoder's error measurement and the query
    /// kernel use, so the stored error certifies the query's exact
    /// arithmetic.
    #[inline]
    fn decode(&self, d: usize, code: u32) -> f64 {
        self.lo[d] as f64 + code as f64 * self.step[d] as f64
    }

    /// Encode one row: append `ceil(dim/2)` nibble-packed bytes to
    /// `codes` (low nibble first) and return the conservatively
    /// rounded-up reconstruction error `e >= ||x - x̂||`.
    pub(crate) fn encode_row(&self, row: &[f32], codes: &mut Vec<u8>) -> f32 {
        let dim = self.lo.len();
        debug_assert_eq!(row.len(), dim);
        let mut err_sq = 0.0f64;
        let mut pending: Option<u8> = None;
        for d in 0..dim {
            let code = if self.step[d] > 0.0 && row[d].is_finite() {
                (((row[d] - self.lo[d]) / self.step[d]).round() as i64).clamp(0, LEVELS as i64)
                    as u32
            } else {
                0
            };
            let miss = row[d] as f64 - self.decode(d, code);
            err_sq += miss * miss;
            match pending.take() {
                None => pending = Some(code as u8),
                Some(low) => codes.push(low | ((code as u8) << 4)),
            }
        }
        if let Some(low) = pending {
            codes.push(low);
        }
        round_error_up(err_sq.sqrt())
    }
}

/// Round a computed error bound UP into its stored f32: a relative
/// inflation covering the encoder's own f64 arithmetic error plus one
/// ULP, so `e_stored >= ||x - x̂||` holds against the query kernel's
/// arithmetic. A non-finite error stores `+inf` — the gate then never
/// fires for that row (fail-open).
fn round_error_up(e: f64) -> f32 {
    if !e.is_finite() {
        return f32::INFINITY;
    }
    ((e * (1.0 + 1e-6)) as f32).next_up()
}

/// Serialize the slot: grid, then one interleaved `[codes][e]` record
/// per row.
pub(crate) fn serialize<W: Write + ?Sized>(
    grid: &SqGrid,
    codes: &[u8],
    errors: &[f32],
    out: &mut W,
) -> io::Result<()> {
    for v in grid.lo.iter().chain(grid.step.iter()) {
        v.serialize(out)?;
    }
    let stride = code_stride(grid.lo.len());
    debug_assert_eq!(codes.len(), errors.len() * stride);
    for (row_codes, e) in codes.chunks_exact(stride).zip(errors) {
        out.write_all(row_codes)?;
        e.serialize(out)?;
    }
    Ok(())
}

/// The reader half: pinned grid + deferred row records, opened from
/// `.vec` slot `[2]`. Records are row-major in the same cluster-sorted
/// row order as the vector rows, so a probed cluster's codes AND errors
/// are together one contiguous ranged read.
pub struct SqCodes {
    grid: SqGrid,
    records: FileSlice,
    stride: usize,
    record: usize,
    dim: usize,
}

impl SqCodes {
    pub(crate) fn open(slice: FileSlice, dim: usize, num_rows: usize) -> crate::Result<SqCodes> {
        let stride = code_stride(dim);
        let record = record_bytes(dim);
        let grid_bytes = 2 * dim * 4;
        let expected = grid_bytes + num_rows * record;
        if slice.len() != expected {
            return Err(crate::TantivyError::InternalError(format!(
                "SQ slot length {} does not match {} rows of dim {} (expected {})",
                slice.len(),
                num_rows,
                dim,
                expected
            )));
        }
        let grid_raw = slice.slice_to(grid_bytes).read_bytes()?;
        let mut reader = grid_raw.as_slice();
        let mut read_dims = |n: usize| -> io::Result<Vec<f32>> {
            (0..n).map(|_| f32::deserialize(&mut reader)).collect()
        };
        let lo = read_dims(dim)?;
        let step = read_dims(dim)?;
        Ok(SqCodes {
            grid: SqGrid { lo, step },
            records: slice.slice_from(grid_bytes),
            stride,
            record,
            dim,
        })
    }

    /// Bytes per row record — the gate's real per-row read cost, for
    /// work-unit pricing against the full vector stride.
    pub(crate) fn row_record_bytes(&self) -> usize {
        self.record
    }

    /// ONE contiguous read for a cluster's records (codes + errors
    /// interleaved), `rows` being the cluster's posting-row range.
    pub(crate) fn cluster_records(
        &self,
        rows: std::ops::Range<usize>,
    ) -> crate::Result<OwnedBytes> {
        Ok(self
            .records
            .slice(rows.start * self.record..rows.end * self.record)
            .read_bytes()?)
    }

    /// Certified upper bound, in NATIVE heap-key space, on what row
    /// `row_in_cluster` could score — computed from its code bytes alone.
    /// If this is below the current kth key, the row provably cannot
    /// enter the heap and its full-precision read can be skipped.
    ///
    /// * Cosine: rows are stored unit-normalized, so the true key is `q·x/||q||`; `q·x <= q·x̂ +
    ///   ||q||·e` (Cauchy-Schwarz) gives `key <= q·x̂/||q|| + e`.
    /// * Dot: `key = q·x <= q·x̂ + ||q||·e`.
    /// * L2: `key = -||q-x||²` and `||q-x|| >= max(0, ||q-x̂|| - e)`.
    ///
    /// All accumulation in f64; a non-finite result compares `false`
    /// against any threshold, so degenerate inputs fail open.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upper_bound_key(
        &self,
        metric: Metric,
        query: &[f32],
        inv_norm_q: f64,
        q_norm: f64,
        cluster_records: &[u8],
        row_in_cluster: usize,
    ) -> f64 {
        let record =
            &cluster_records[row_in_cluster * self.record..(row_in_cluster + 1) * self.record];
        let (codes, e_bytes) = record.split_at(self.stride);
        let e = f32::from_le_bytes(e_bytes.try_into().unwrap()) as f64;
        let mut dot = 0.0f64;
        let mut l2_sq = 0.0f64;
        let want_l2 = metric == Metric::L2;
        for d in 0..self.dim {
            let nibble = (codes[d / 2] >> ((d % 2) * 4)) & 0x0f;
            let xhat = self.grid.decode(d, nibble as u32);
            let q = query[d] as f64;
            if want_l2 {
                let diff = q - xhat;
                l2_sq += diff * diff;
            } else {
                dot += q * xhat;
            }
        }
        match metric {
            Metric::Cosine => dot * inv_norm_q + e,
            Metric::Dot => dot + q_norm * e,
            Metric::L2 => {
                let lb = (l2_sq.sqrt() - e).max(0.0);
                -(lb * lb)
            }
        }
    }
}

impl std::fmt::Debug for SqCodes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqCodes")
            .field("dim", &self.dim)
            .field("rows", &(self.records.len() / self.record))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(rows: &[Vec<f32>], dim: usize) -> (SqCodes, Vec<f32>) {
        let grid = SqGrid::from_sample(dim, rows.iter().map(|r| r.as_slice()));
        let mut codes = Vec::new();
        let mut errors = Vec::new();
        for row in rows {
            errors.push(grid.encode_row(row, &mut codes));
        }
        let mut buf = Vec::new();
        serialize(&grid, &codes, &errors, &mut buf).unwrap();
        let sq = SqCodes::open(FileSlice::from(buf), dim, rows.len()).unwrap();
        (sq, errors)
    }

    /// The load-bearing property: for random rows and queries, under every
    /// metric, the certified upper bound NEVER undershoots the true heap
    /// key. One violation is an incorrect prune, so this is exhaustive
    /// over many pairs rather than spot-checked.
    #[test]
    fn upper_bound_never_undershoots_true_key() {
        let dim = 96;
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut rows: Vec<Vec<f32>> = (0..64)
            .map(|_| (0..dim).map(|_| rand() as f32).collect())
            .collect();
        // Unit-normalize half the rows (the cosine layout invariant).
        for row in rows.iter_mut().take(32) {
            let n = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            row.iter_mut().for_each(|v| *v /= n);
        }
        let (sq, _) = roundtrip(&rows, dim);
        let recs = sq.cluster_records(0..rows.len()).unwrap();

        for _ in 0..200 {
            let q: Vec<f32> = (0..dim).map(|_| rand() as f32 * 2.0).collect();
            let q_norm = q.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            for (i, row) in rows.iter().enumerate() {
                let dot: f64 = q.iter().zip(row).map(|(a, b)| *a as f64 * *b as f64).sum();
                let l2_sq: f64 = q
                    .iter()
                    .zip(row)
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum();
                for (metric, true_key) in [
                    (Metric::Dot, dot),
                    (Metric::L2, -l2_sq),
                    // Only the normalized rows honor the cosine layout.
                    (Metric::Cosine, dot / q_norm),
                ] {
                    if metric == Metric::Cosine && i >= 32 {
                        continue;
                    }
                    let ub = sq.upper_bound_key(metric, &q, 1.0 / q_norm, q_norm, &recs, i);
                    assert!(
                        ub >= true_key - 1e-9,
                        "{metric:?} bound {ub} undershoots true key {true_key} for row {i}"
                    );
                }
            }
        }
    }

    /// The bound must also be TIGHT enough to fire: for a well-spread
    /// sample the error stays a small fraction of the row norm.
    #[test]
    fn error_is_small_for_in_grid_rows() {
        let dim = 128;
        let rows: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i * 31 + d * 17) % 97) as f32 / 97.0 - 0.5)
                    .collect()
            })
            .collect();
        let (_, errors) = roundtrip(&rows, dim);
        let norm = (rows[0].iter().map(|v| v * v).sum::<f32>()).sqrt();
        for e in errors {
            assert!(e < 0.1 * norm, "error {e} too large vs norm {norm}");
        }
    }

    /// Odd dimension exercises the trailing lone nibble.
    #[test]
    fn odd_dim_roundtrips() {
        let rows: Vec<Vec<f32>> = (0..8)
            .map(|i| (0..7).map(|d| (i * 7 + d) as f32 / 10.0).collect())
            .collect();
        let (sq, errors) = roundtrip(&rows, 7);
        assert_eq!(sq.stride, 4);
        assert_eq!(sq.row_record_bytes(), 8);
        let recs = sq.cluster_records(2..5).unwrap();
        assert_eq!(recs.len(), 3 * 8);
        assert!(errors.iter().all(|e| e.is_finite()));
        // Row 3's bound against itself as the query must not undershoot.
        let q = &rows[3];
        let q_norm = q.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let true_dot: f64 = q.iter().map(|v| (*v as f64).powi(2)).sum();
        let ub = sq.upper_bound_key(Metric::Dot, q, 1.0 / q_norm, q_norm, &recs, 1);
        assert!(ub >= true_dot - 1e-9);
    }

    /// Non-finite coordinates: sampled ones degrade their grid line,
    /// encoded ones store an infinite error — both fail open.
    #[test]
    fn non_finite_rows_fail_open() {
        let mut rows: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32; 8]).collect();
        rows[1][3] = f32::NAN;
        let (sq, errors) = roundtrip(&rows, 8);
        assert!(errors[1].is_infinite());
        let recs = sq.cluster_records(0..4).unwrap();
        let q = vec![1.0f32; 8];
        let ub = sq.upper_bound_key(Metric::Dot, &q, 1.0, 1.0, &recs, 1);
        assert!(
            ub.is_infinite() && ub > 0.0,
            "infinite error must never gate"
        );
    }
}
