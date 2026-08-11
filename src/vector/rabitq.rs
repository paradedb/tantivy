//! RaBitQ-style residual-direction sketches: the directional term the
//! radius-only row gate lacks.
//!
//! Geometry (Gao & Long, "RaBitQ"): with `s = (x - c) / ||x - c||` and
//! `u = (q - c) / ||q - c||`,
//!
//! ```text
//! ||q - x||^2 = d_c^2 + r_x^2 - 2 d_c r_x <u, s>
//! ```
//!
//! so a per-row estimate of `cos = <u, s>` with a conservative error
//! bound yields a certified-probabilistic lower bound on the distance.
//! Stored per row: the SIGN BITS of the rotated direction (`dim/8`
//! bytes), `a = <s, s_hat>` (u16 fixed-point, rounded DOWN so the
//! decoded interval brackets the true value), and `b = <s_hat, Rc>`
//! (u16 fixed-point) — the per-row constant that makes the query side
//! ONE global rotation + quantization:
//!
//! ```text
//! <s_hat, u> = <s_hat, R(q - c)> / d_c = (<s_hat, Rq> - b) / d_c
//! ```
//!
//! The bound is the RaBitQ `/a` estimator. Decomposing the QUERY side
//! `u = z * s + w` (with `w` orthogonal to `s`) gives
//! `<u, s_hat> = z * a + <w, s_hat>`, where `<w, s_hat>` is genuinely
//! zero-mean over the rotation (`w` is orthogonal to `s`, so it only
//! meets the `sqrt(1 - a^2)` off-axis part of `s_hat`):
//!
//! ```text
//! z_up = (v_up + t * sqrt((1 - a^2)/(d - 1))) / a
//! ```
//!
//! with `v_up` folding the kernel numerator, the stored-b decode step,
//! and the query-quantization dither (whose rounding-error vector is
//! uncorrelated with any row's sign pattern, so it concentrates like
//! the code-side term), and the division taking the conservative `a`
//! interval endpoint by the numerator's sign. The tempting identity
//! form `cos = a * <u, s_hat> + <u, e>` is NOT usable with a zero-mean
//! model for `<u, e>`: `e` correlates with `u` through `s`
//! (`E[<u, e>] = z * (1 - a^2)`), which biases high-z rows — exactly
//! the rows a wrong prune costs recall on. The failure probability per
//! gate decision is the sub-Gaussian tail at `t`; [`SKETCH_T`] fixes
//! the budget.
//!
//! Everything is fail-open: a degenerate `a` (zero residual, corrupt
//! metadata) blows up `eps`, the upper bound clamps to 1, and the gate's
//! bound degenerates to the radius-only shell bound `|d_c - r_x|`.
//!
//! The rotation is a seeded 3-round `H * D_i` product (random sign
//! diagonal, then a fast Walsh-Hadamard transform), an orthogonal
//! transform applied identically to stored residuals at build and query
//! residuals at search. Power-of-two dims only — the reason the sketch
//! slot is optional per segment.

use std::io::{self, Write};

use common::{BinarySerializable, HasLen, OwnedBytes};

use crate::directory::FileSlice;

/// The gate's tail multiplier `t`. Per gate decision the wrong-prune
/// probability is bounded by the sub-Gaussian tail `~exp(-t^2/2)`
/// (~2e-5 at 4.5); a query makes `k` decisions that can cost recall
/// (its true top-k rows), so the per-query recall loss budget is
/// `~k * 2e-5`. Raise to trade pruning for a tighter budget.
pub const SKETCH_T: f32 = 4.5;

/// Rounds of sign-diagonal + Walsh-Hadamard in the rotation. Three
/// rounds is the standard choice for smoothing worst-case coordinate
/// concentration.
const ROTATION_ROUNDS: usize = 3;

/// Query-side scalar quantization width, in bits. The query is
/// quantized ONCE globally (not per cluster-residual), and the dither
/// is amplified by `1/d_c` per cluster, so the grid is finer than the
/// per-cluster form needs: 8 bits restores full parity with exact-query
/// pruning on the cohere replica (4 bits costs ~9pp there).
const QUERY_BITS: u32 = 8;
const QUERY_LEVELS: u32 = (1 << QUERY_BITS) - 1;

/// Deterministic splitmix64 — the sign diagonals must reproduce across
/// platforms and dependency bumps forever, so no external RNG.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Whether a field's geometry supports the sketch: the FWHT needs a
/// power-of-two dim and the codes need whole bytes.
pub(crate) fn dim_supported(dim: usize) -> bool {
    dim >= 64 && dim.is_power_of_two()
}

/// The seeded orthogonal rotation shared by one segment-field's build
/// and queries: `ROTATION_ROUNDS` of (sign diagonal, FWHT), normalized
/// so the transform is orthonormal.
pub struct Rotation {
    /// Packed sign diagonals, one bit per (round, dim): 1 = negate.
    signs: Vec<u64>,
    dim: usize,
}

impl Rotation {
    /// Derives the rotation for `dim` from `seed`.
    ///
    /// * `seed` (`u64`) — the per-segment-field seed stored in the sketch slot.
    /// * `dim` (`usize`) — vector dimensionality; must satisfy [`dim_supported`].
    ///
    /// Returns (`Rotation`): the reusable transform (allocates only the
    /// packed diagonals: `ROTATION_ROUNDS * dim / 64` words).
    pub fn new(seed: u64, dim: usize) -> Rotation {
        debug_assert!(dim_supported(dim));
        let mut state = seed;
        let words_per_round = dim / 64;
        let signs = (0..ROTATION_ROUNDS * words_per_round)
            .map(|_| splitmix64(&mut state))
            .collect();
        Rotation { signs, dim }
    }

    /// Applies the rotation to `v` in place.
    ///
    /// * `v` (`&mut [f32]`) — a `dim`-length vector, overwritten with `R v`. Norms and inner
    ///   products are preserved (orthonormal).
    pub fn apply(&self, v: &mut [f32]) {
        debug_assert_eq!(v.len(), self.dim);
        let words_per_round = self.dim / 64;
        let scale = 1.0 / (self.dim as f32).sqrt();
        for round in 0..ROTATION_ROUNDS {
            let signs = &self.signs[round * words_per_round..(round + 1) * words_per_round];
            for (i, value) in v.iter_mut().enumerate() {
                if signs[i / 64] >> (i % 64) & 1 == 1 {
                    *value = -*value;
                }
            }
            fwht(v);
            // Fold the 1/sqrt(dim) normalization into the pass so each
            // round is orthonormal and values stay well-scaled.
            for value in v.iter_mut() {
                *value *= scale;
            }
        }
    }
}

/// In-place fast Walsh-Hadamard transform (unnormalized butterflies).
fn fwht(v: &mut [f32]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        for block in (0..n).step_by(h * 2) {
            for i in block..block + h {
                let (a, b) = (v[i], v[i + h]);
                v[i] = a + b;
                v[i + h] = a - b;
            }
        }
        h *= 2;
    }
}

// ======================================================================
// Build side
// ======================================================================

/// One row's sketch: the rotated direction's sign bits, the fixed-point
/// `a`, and the fixed-point centroid projection `b`.
pub struct EncodedRow {
    /// `dim / 8` bytes, bit `i` = 1 iff rotated coordinate `i` is
    /// non-negative.
    pub code: Vec<u8>,
    /// `a = <s_rot, s_hat>` in u16 fixed-point over [0, 1], rounded
    /// DOWN. `0` = degenerate (zero/non-finite residual): fail-open.
    pub a: u16,
    /// `b = <s_hat, R c>` in u16 fixed-point over [-B_RANGE, B_RANGE].
    /// The per-row constant that turns the query side into ONE global
    /// rotation + quantization: `<s_hat, R(q - c)> = <s_hat, Rq> - b`.
    pub b: u16,
}

/// Fixed-point range of the stored `b`: `|<s_hat, Rc>| <= ||c||`, and
/// stored centroids are unit for cosine and O(1) for realistic L2
/// embeddings. Values outside the range saturate, which only loosens
/// the bound (the decode interval still brackets the clamped truth
/// conservatively via [`B_ULP`] + the clamp check below).
const B_RANGE: f32 = 4.0;
/// One decode step of the stored `b`.
pub(crate) const B_ULP: f32 = 2.0 * B_RANGE / u16::MAX as f32;

fn encode_b(b: f64) -> u16 {
    (((b as f32).clamp(-B_RANGE, B_RANGE) + B_RANGE) / (2.0 * B_RANGE) * u16::MAX as f32).round()
        as u16
}

#[inline]
pub(crate) fn decode_b(code: u16) -> f32 {
    code as f32 / u16::MAX as f32 * (2.0 * B_RANGE) - B_RANGE
}

/// Encodes one row's residual against its cluster centroid.
///
/// * `rotation` (`&Rotation`) — the segment's shared rotation.
/// * `residual` (`&mut [f32]`) — `x - c` in the ORIGINAL basis; consumed as scratch (rotated in
///   place). Need not be pre-normalized: the norm is divided out here.
/// * `rotated_centroid` (`&[f32]`) — `R c`, the row's cluster centroid in the ROTATED basis (rotate
///   once per cluster at build).
///
/// Returns (`EncodedRow`): the packed sign code and conservative `a`,
/// `b`. A zero or non-finite residual encodes as `a = 0`, which the
/// query side treats as "no directional information" (fail-open).
pub fn encode_row(
    rotation: &Rotation,
    residual: &mut [f32],
    rotated_centroid: &[f32],
) -> EncodedRow {
    let dim = residual.len();
    let norm = residual
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let mut code = vec![0u8; dim / 8];
    if !(norm.is_finite() && norm > 0.0) {
        return EncodedRow {
            code,
            a: 0,
            b: encode_b(0.0),
        };
    }
    rotation.apply(residual);
    // a = <s, sign(s)>/sqrt(dim) = ||s||_1 / (||s||_2 * sqrt(dim)).
    // b = <sign(s), Rc>/sqrt(dim).
    let mut l1 = 0.0f64;
    let mut b_acc = 0.0f64;
    for (i, &value) in residual.iter().enumerate() {
        let ci = rotated_centroid[i] as f64;
        if value >= 0.0 {
            code[i / 8] |= 1 << (i % 8);
            b_acc += ci;
        } else {
            b_acc -= ci;
        }
        l1 += value.abs() as f64;
    }
    let a = l1 / (norm * (dim as f64).sqrt());
    let b = b_acc / (dim as f64).sqrt();
    if !a.is_finite() || !b.is_finite() || b.abs() > B_RANGE as f64 {
        // An out-of-range b cannot be bracketed by the fixed-point
        // interval; drop the row's sketch instead of mis-bounding it.
        return EncodedRow {
            code,
            a: 0,
            b: encode_b(0.0),
        };
    }
    // Round DOWN so the decoded interval [a_lo, a_lo + ULP] brackets the
    // true value; clamp to the representable range.
    let a_fixed = (a.clamp(0.0, 1.0) * u16::MAX as f64).floor() as u16;
    EncodedRow {
        code,
        a: a_fixed,
        b: encode_b(b),
    }
}

// ======================================================================
// Slot [5] payload
// ======================================================================

/// Sketch kind byte. One kind today; the byte is the growth point.
const KIND_SIGN_RABITQ: u8 = 0;

/// Serializes the sketch slot payload: kind, seed, then the SoA arrays
/// (all codes contiguous — 128 B/row at 1024 dims, two cache lines —
/// then all `a`s).
///
/// * `seed` (`u64`) — the rotation seed the rows were encoded with.
/// * `codes` (`&[u8]`) — `num_rows * dim / 8` bytes, row-major.
/// * `a_values` (`&[u16]`) — `num_rows` fixed-point `a`s.
/// * `out` (`&mut W`) — the slot writer.
///
/// Returns (`io::Result<()>`): write errors only; lengths are validated
/// at open against the row total and dim.
pub(crate) fn serialize_sketches<W: Write + ?Sized>(
    seed: u64,
    codes: &[u8],
    a_values: &[u16],
    b_values: &[u16],
    out: &mut W,
) -> io::Result<()> {
    debug_assert_eq!(a_values.len(), b_values.len());
    KIND_SIGN_RABITQ.serialize(out)?;
    seed.serialize(out)?;
    // Interleaved rows: [code][a][b] per row, so one probed cluster is
    // ONE contiguous ranged read. The gate always consumes all three
    // together; parallel arrays would cost a read per array per cluster
    // for zero benefit.
    let code_stride = if a_values.is_empty() {
        0
    } else {
        codes.len() / a_values.len()
    };
    for (i, (a, b)) in a_values.iter().zip(b_values).enumerate() {
        out.write_all(&codes[i * code_stride..(i + 1) * code_stride])?;
        a.serialize(out)?;
        b.serialize(out)?;
    }
    Ok(())
}

/// The sketch slot of one segment-field. Only the 9-byte header is read
/// at open — the SoA arrays stay behind [`FileSlice`]s and are fetched
/// per probed cluster as one contiguous ranged read each
/// ([`Self::cluster_view`]), because the caller may rebuild the reader
/// per query (pg_search's per-snapshot directory does) and an eager
/// slot-wide materialization would be paid every time.
pub struct SketchStore {
    rotation: Rotation,
    /// Interleaved `[code][a][b]` rows, `row_stride` bytes each.
    rows: FileSlice,
    row_stride: usize,
    code_stride: usize,
}

/// One probed cluster's sketch rows — ONE contiguous ranged read; row
/// indices are relative to the cluster's first row.
pub struct SketchClusterView {
    rows: OwnedBytes,
    row_stride: usize,
    code_stride: usize,
}

impl SketchClusterView {
    /// The fetched payload size, for the caller's I/O accounting.
    pub fn len_bytes(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn code(&self, offset: usize) -> &[u8] {
        let start = offset * self.row_stride;
        &self.rows[start..start + self.code_stride]
    }

    #[inline]
    fn a(&self, offset: usize) -> u16 {
        let at = offset * self.row_stride + self.code_stride;
        u16::from_le_bytes(self.rows[at..at + 2].try_into().unwrap())
    }

    #[inline]
    fn b(&self, offset: usize) -> u16 {
        let at = offset * self.row_stride + self.code_stride + 2;
        u16::from_le_bytes(self.rows[at..at + 2].try_into().unwrap())
    }
}

impl SketchStore {
    /// Parses a slot [5] payload.
    ///
    /// * `bytes` (`OwnedBytes`) — the whole slot.
    /// * `dim` (`usize`) — the field's dimensionality.
    /// * `num_rows` (`usize`) — the posting-row total from slot [1].
    ///
    /// Returns (`io::Result<SketchStore>`): the store, or `InvalidData`
    /// on an unknown kind or a length mismatch.
    pub(crate) fn open(slot: FileSlice, dim: usize, num_rows: usize) -> io::Result<SketchStore> {
        const HEADER: usize = 9; // kind u8 + seed u64
        if slot.len() < HEADER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sketch slot is smaller than its header",
            ));
        }
        let header_bytes = slot
            .slice_to(HEADER)
            .read_bytes()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut reader = header_bytes.as_slice();
        let kind = u8::deserialize(&mut reader)?;
        if kind != KIND_SIGN_RABITQ {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown vector sketch kind: {kind}"),
            ));
        }
        let seed = u64::deserialize(&mut reader)?;
        if !dim_supported(dim) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sketch slot present for an unsupported dim",
            ));
        }
        let code_stride = dim / 8;
        let row_stride = code_stride + 4;
        let rows_len = num_rows.checked_mul(row_stride).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "sketch rows length overflow")
        })?;
        if slot.len() - HEADER != rows_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sketch slot byte length mismatch",
            ));
        }
        Ok(SketchStore {
            rotation: Rotation::new(seed, dim),
            rows: slot.slice_from(HEADER),
            row_stride,
            code_stride,
        })
    }

    /// The shared rotation, for the query side.
    pub fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    /// Fetches one cluster's sketch rows with ONE contiguous ranged
    /// read over `rows` (a cluster's row range in the segment-wide
    /// dense numbering).
    pub fn cluster_view(&self, rows: std::ops::Range<usize>) -> crate::Result<SketchClusterView> {
        Ok(SketchClusterView {
            rows: self
                .rows
                .slice(rows.start * self.row_stride..rows.end * self.row_stride)
                .read_bytes()?,
            row_stride: self.row_stride,
            code_stride: self.code_stride,
        })
    }
}

// ======================================================================
// Query side
// ======================================================================

/// One QUERY's prepared sketch side: the rotated, `B_q`-bit-quantized
/// query vector in the bitplane layout the popcount kernel consumes,
/// plus the probabilistic dither bound. Built ONCE per segment-query;
/// per probed cluster the gate only needs the routing separation `d_c`,
/// because each row's `b = <s_hat, Rc>` was folded in at build:
/// `<s_hat, u> = (<s_hat, Rq> - b) / d_c`.
pub struct PreparedSketchQuery {
    /// `QUERY_BITS` bitplanes, each `dim / 64` u64 words: plane `j`
    /// holds bit `j` of every coordinate's quantized level.
    planes: Vec<u64>,
    words: usize,
    /// Quantization grid: `q_hat_i = lo + delta * level_i`.
    lo: f32,
    delta: f32,
    /// `sum(level_i)` — the kernel's constant term.
    level_sum: u32,
    /// `t * ||Rq - Rq_hat|| / sqrt(dim) + B_ULP/2`: the per-row bound on
    /// the dither + stored-b decode error of the numerator
    /// `<s_hat, Rq_hat> - b_hat`. The dither part is probabilistic (the
    /// rounding-error vector is uncorrelated with any row's sign
    /// pattern); the b part is deterministic.
    num_err: f32,
    dim: usize,
}

impl PreparedSketchQuery {
    /// Prepares the query side, once per segment-query.
    ///
    /// * `rotation` (`&Rotation`) — the store's rotation.
    /// * `query` (`&mut [f32]`) — the query in the ORIGINAL basis (unit-normalized upstream for
    ///   cosine); consumed as scratch (rotated in place).
    ///
    /// Returns (`Option<PreparedSketchQuery>`): `None` when the query is
    /// degenerate (non-finite, or a flat rotated vector) — the gate then
    /// falls back to the radius-only bound.
    pub fn prepare(rotation: &Rotation, query: &mut [f32]) -> Option<PreparedSketchQuery> {
        let dim = query.len();
        rotation.apply(query);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &value in query.iter() {
            lo = lo.min(value);
            hi = hi.max(value);
        }
        if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
            return None;
        }
        let delta = (hi - lo) / QUERY_LEVELS as f32;
        let words = dim / 64;
        let mut planes = vec![0u64; QUERY_BITS as usize * words];
        let mut level_sum = 0u32;
        let mut dither2 = 0.0f64;
        for (i, &value) in query.iter().enumerate() {
            let level = (((value - lo) / delta).round() as u32).min(QUERY_LEVELS);
            level_sum += level;
            let decoded = lo + delta * level as f32;
            dither2 += ((value - decoded) as f64).powi(2);
            for j in 0..QUERY_BITS as usize {
                if level >> j & 1 == 1 {
                    planes[j * words + i / 64] |= 1 << (i % 64);
                }
            }
        }
        let num_err = SKETCH_T * (dither2.sqrt() as f32) / (dim as f32).sqrt() + B_ULP / 2.0;
        Some(PreparedSketchQuery {
            planes,
            words,
            lo,
            delta,
            level_sum,
            num_err,
            dim,
        })
    }

    /// The raw estimator numerator `<s_hat, u_hat>` for one row's code,
    /// via bitplane popcounts — the production kernel: `QUERY_BITS * words`
    /// AND+POPCNT per row on the two-cache-line code.
    #[inline]
    fn code_dot(&self, code: &[u8]) -> f32 {
        debug_assert_eq!(code.len(), self.words * 8);
        let mut ones = 0u32; // popcount of the sign code
        let mut weighted = 0u32; // sum of levels where the sign bit is 1
        for w in 0..self.words {
            let cw = u64::from_le_bytes(code[w * 8..(w + 1) * 8].try_into().unwrap());
            ones += cw.count_ones();
            for j in 0..QUERY_BITS as usize {
                weighted += (cw & self.planes[j * self.words + w]).count_ones() << j;
            }
        }
        // sum over dims of sign_i * u_hat_i, with sign in {-1, +1}:
        // lo * (2 * ones - dim) + delta * (2 * weighted - level_sum),
        // then the 1/sqrt(dim) of the unit code s_hat.
        let signed_lo = self.lo * (2.0 * ones as f32 - self.dim as f32);
        let signed_lv = self.delta * (2.0 * weighted as f32 - self.level_sum as f32);
        (signed_lo + signed_lv) / (self.dim as f32).sqrt()
    }

    /// Conservative upper bound on `cos = <u, s>` for one row, with
    /// `u = (q - c)/d_c` and `s` the row's unit residual direction —
    /// the RaBitQ `/a` estimator plus every stored/quantized term's
    /// conservative endpoint (see the module docs for why the
    /// division-free identity form is NOT sound here).
    ///
    /// * `view` (`&SketchClusterView`) — the probed cluster's fetched sketch rows.
    /// * `offset` (`usize`) — row index RELATIVE to the cluster's first row.
    /// * `d_c` (`f32`) — `||q - c||` of the row's cluster, from the routing key.
    ///
    /// Returns (`f32`): the upper bound, clamped to 1; `1.0` for rows
    /// with no usable sketch (`a = 0`) or a degenerate `d_c` — the
    /// radius-only bound then applies unchanged. NaN inputs propagate to
    /// a non-pruning bound.
    #[inline]
    pub fn cos_upper_bound(&self, view: &SketchClusterView, offset: usize, d_c: f32) -> f32 {
        let a_lo = view.a(offset) as f32 / u16::MAX as f32;
        if a_lo <= 0.0 || !(d_c > 0.0) {
            return 1.0;
        }
        let a_hi = a_lo + 1.0 / u16::MAX as f32;
        let b_lo = decode_b(view.b(offset)) - B_ULP / 2.0;
        let num_up = self.code_dot(view.code(offset)) - b_lo + self.num_err;
        // <u, s_hat> upper endpoint; |<u, s_hat>| <= 1 always.
        let v_up = (num_up / d_c).clamp(-1.0, 1.0);
        // z_up = (v_up + concentration) / a, the division taking the
        // conservative interval endpoint by the numerator's sign.
        let zn = v_up + SKETCH_T * ((1.0 - a_lo * a_lo).max(0.0) / (self.dim as f32 - 1.0)).sqrt();
        let z_up = if zn >= 0.0 { zn / a_lo } else { zn / a_hi };
        z_up.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_vec(state: &mut u64, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|_| {
                let bits = splitmix64(state);
                (bits >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    /// The rotation is orthonormal: norms and inner products survive.
    #[test]
    fn rotation_preserves_geometry() {
        let dim = 256;
        let rot = Rotation::new(42, dim);
        let mut state = 7u64;
        let a0 = lcg_vec(&mut state, dim);
        let b0 = lcg_vec(&mut state, dim);
        let (mut a, mut b) = (a0.clone(), b0.clone());
        rot.apply(&mut a);
        rot.apply(&mut b);
        assert!((norm(&a) - norm(&a0)).abs() < 1e-3);
        assert!((dot(&a, &b) - dot(&a0, &b0)).abs() < 1e-3);
    }

    /// Same seed, same transform; different seed, different transform.
    #[test]
    fn rotation_is_seeded() {
        let dim = 128;
        let mut state = 3u64;
        let v0 = lcg_vec(&mut state, dim);
        let (mut a, mut b, mut c) = (v0.clone(), v0.clone(), v0);
        Rotation::new(1, dim).apply(&mut a);
        Rotation::new(1, dim).apply(&mut b);
        Rotation::new(2, dim).apply(&mut c);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    /// The popcount kernel computes exactly the float dot of the sign
    /// code against the decoded quantized query.
    #[test]
    fn kernel_matches_float_reference() {
        let dim = 128;
        let rot = Rotation::new(9, dim);
        let mut state = 11u64;
        for _ in 0..10 {
            let mut q_res = lcg_vec(&mut state, dim);
            let x_res = lcg_vec(&mut state, dim);
            let encoded = encode_row(&rot, &mut x_res.clone(), &vec![0.0; dim]);
            let prepared = PreparedSketchQuery::prepare(&rot, &mut q_res).unwrap();
            // Float reference: rebuild s_hat and u_hat by hand.
            let s_hat: Vec<f32> = (0..dim)
                .map(|i| {
                    let bit = encoded.code[i / 8] >> (i % 8) & 1;
                    if bit == 1 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .map(|sign| sign / (dim as f32).sqrt())
                .collect();
            // Decode the quantized query back out of the bitplanes.
            let u_hat: Vec<f32> = (0..dim)
                .map(|i| {
                    let mut level = 0u32;
                    for j in 0..QUERY_BITS as usize {
                        level |= ((prepared.planes[j * prepared.words + i / 64] >> (i % 64)) & 1)
                            as u32
                            * (1 << j);
                    }
                    prepared.lo + prepared.delta * level as f32
                })
                .collect();
            let reference = dot(&s_hat, &u_hat);
            let kernel = prepared.code_dot(&encoded.code);
            assert!(
                (reference - kernel).abs() < 1e-4,
                "kernel {kernel} != reference {reference}"
            );
        }
    }

    /// The estimator concentrates: over random cluster geometry, the
    /// true cos of every (query residual, row residual) pair stays
    /// within the conservative bound and the bound stays useful (well
    /// below 1 for uncorrelated pairs).
    #[test]
    fn estimator_bounds_hold() {
        let dim = 1024;
        let rot = Rotation::new(21, dim);
        let mut state = 5u64;
        // One cluster centroid; rows are centroid + residual.
        let centroid = lcg_vec(&mut state, dim);
        let mut rotated_centroid = centroid.clone();
        rot.apply(&mut rotated_centroid);
        let mut codes = Vec::new();
        let mut a_values = Vec::new();
        let mut b_values = Vec::new();
        let mut residuals = Vec::new();
        for _ in 0..200 {
            let x = lcg_vec(&mut state, dim);
            let encoded = encode_row(&rot, &mut x.clone(), &rotated_centroid);
            codes.extend_from_slice(&encoded.code);
            a_values.push(encoded.a);
            b_values.push(encoded.b);
            residuals.push(x);
        }
        let mut buf = Vec::new();
        serialize_sketches(21, &codes, &a_values, &b_values, &mut buf).unwrap();
        let store = SketchStore::open(FileSlice::from(buf), dim, 200).unwrap();
        let view = store.cluster_view(0..200).unwrap();

        let mut state = 77u64;
        let mut below_one = 0usize;
        let mut high_z_seen = 0usize;
        for probe in 0..10 {
            // The query lives in the original basis; its residual
            // against the centroid defines u and d_c. Half the queries
            // are CORRELATED with a stored row (query residual = that
            // row's residual + small noise), driving z toward 1 — the
            // regime where a biased estimator (e.g. the identity form's
            // zero-mean <u, e> assumption, off by z * (1 - a^2)) breaks
            // the bound on exactly the rows that cost recall.
            let q: Vec<f32> = if probe % 2 == 0 {
                lcg_vec(&mut state, dim)
                    .iter()
                    .zip(&centroid)
                    .map(|(&r, &c)| c + r)
                    .collect()
            } else {
                let anchor = &residuals[probe * 17 % residuals.len()];
                let noise = lcg_vec(&mut state, dim);
                let anchor_norm = norm(anchor);
                anchor
                    .iter()
                    .zip(&noise)
                    .zip(&centroid)
                    .map(|((&r, &n), &c)| c + r + 0.05 * anchor_norm * n)
                    .collect()
            };
            let q_res: Vec<f32> = q.iter().zip(&centroid).map(|(&qi, &ci)| qi - ci).collect();
            let d_c = norm(&q_res);
            let prepared = PreparedSketchQuery::prepare(&rot, &mut q.clone()).unwrap();
            for (row, x) in residuals.iter().enumerate() {
                let true_cos = dot(&q_res, x) / (d_c * norm(x));
                let bound = prepared.cos_upper_bound(&view, row, d_c);
                assert!(
                    bound >= true_cos - 1e-4,
                    "row {row}: bound {bound} below true cos {true_cos}"
                );
                if bound < 0.9 {
                    below_one += 1;
                }
                if true_cos > 0.7 {
                    high_z_seen += 1;
                }
            }
        }
        // The correlated queries must actually produce high-z pairs, or
        // the bias regime went untested.
        assert!(high_z_seen >= 5, "want high-z coverage, got {high_z_seen}");
        // Uncorrelated 1024-dim pairs: the bound must actually bite.
        assert!(
            below_one > 1500,
            "bound should be informative for most random pairs, got {below_one}/2000"
        );
    }

    /// Degenerate inputs fail open: zero residuals encode a = 0 which
    /// bounds cos at 1.0; zero query residuals refuse to prepare.
    #[test]
    fn degenerate_inputs_fail_open() {
        let dim = 128;
        let rot = Rotation::new(4, dim);
        let zero_c = vec![0.0; dim];
        let encoded = encode_row(&rot, &mut vec![0.0; dim], &zero_c);
        assert_eq!(encoded.a, 0);
        let nan = encode_row(&rot, &mut vec![f32::NAN; dim], &zero_c);
        assert_eq!(nan.a, 0);
        // A flat (all-equal after rotation: all-zero) query refuses to
        // prepare.
        assert!(PreparedSketchQuery::prepare(&rot, &mut vec![0.0; dim]).is_none());

        let mut buf = Vec::new();
        serialize_sketches(4, &encoded.code, &[encoded.a], &[encoded.b], &mut buf).unwrap();
        let store = SketchStore::open(FileSlice::from(buf), dim, 1).unwrap();
        let view = store.cluster_view(0..1).unwrap();
        let mut q = vec![1.0; dim];
        let prepared = PreparedSketchQuery::prepare(&rot, &mut q).unwrap();
        assert_eq!(prepared.cos_upper_bound(&view, 0, 1.0), 1.0);
        // Degenerate separation also fails open.
        assert_eq!(prepared.cos_upper_bound(&view, 0, 0.0), 1.0);
    }

    /// Slot round-trip and the corrupt-length rejections.
    #[test]
    fn slot_validation() {
        let dim = 64;
        let rot = Rotation::new(1, dim);
        let mut residual: Vec<f32> = (0..dim).map(|i| i as f32 - 31.5).collect();
        let encoded = encode_row(&rot, &mut residual, &vec![0.25; dim]);
        let mut buf = Vec::new();
        serialize_sketches(1, &encoded.code, &[encoded.a], &[encoded.b], &mut buf).unwrap();
        assert!(SketchStore::open(FileSlice::from(buf.clone()), dim, 1).is_ok());
        assert!(SketchStore::open(FileSlice::from(buf.clone()), dim, 2).is_err());
        let mut bad_kind = buf.clone();
        bad_kind[0] = 9;
        assert!(SketchStore::open(FileSlice::from(bad_kind), dim, 1).is_err());
        buf.pop();
        assert!(SketchStore::open(FileSlice::from(buf), dim, 1).is_err());
    }
}
