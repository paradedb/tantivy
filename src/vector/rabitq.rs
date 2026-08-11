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
//! bytes) plus `a = <s, s_hat>` (u16 fixed-point, rounded DOWN so the
//! decoded interval `[a, a + ULP]` brackets the true value). The
//! estimator is `z_hat = <s_hat, u> / a` — unbiased over the rotation —
//! with error concentrating at
//!
//! ```text
//! eps = t * sqrt((1 - a^2)/(d - 1) + ||u - u_hat||^2 / d) / a
//! ```
//!
//! where the second variance term is the query-side `B_q`-bit
//! quantization dither (`||u - u_hat||` is computed EXACTLY once per
//! probed cluster, so nothing about the query side is estimated). The
//! failure probability per gate decision is the sub-Gaussian tail at
//! `t`; [`SKETCH_T`] fixes the budget.
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

use common::BinarySerializable;

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

/// Query-side scalar quantization width, in bits. 4 keeps the dither
/// variance term negligible against the code-side concentration term.
const QUERY_BITS: u32 = 4;
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

/// One row's sketch: the rotated direction's sign bits and the
/// fixed-point `a`.
pub struct EncodedRow {
    /// `dim / 8` bytes, bit `i` = 1 iff rotated coordinate `i` is
    /// non-negative.
    pub code: Vec<u8>,
    /// `a = <s_rot, s_hat>` in u16 fixed-point over [0, 1], rounded
    /// DOWN. `0` = degenerate (zero/non-finite residual): fail-open.
    pub a: u16,
}

/// Encodes one row's residual against its cluster centroid.
///
/// * `rotation` (`&Rotation`) — the segment's shared rotation.
/// * `residual` (`&mut [f32]`) — `x - c` in the ORIGINAL basis; consumed as scratch (rotated in
///   place). Need not be pre-normalized: the norm is divided out here.
///
/// Returns (`EncodedRow`): the packed sign code and conservative `a`.
/// A zero or non-finite residual encodes as `a = 0`, which the query
/// side treats as "no directional information" (fail-open).
pub fn encode_row(rotation: &Rotation, residual: &mut [f32]) -> EncodedRow {
    let dim = residual.len();
    let norm = residual
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let mut code = vec![0u8; dim / 8];
    if !(norm.is_finite() && norm > 0.0) {
        return EncodedRow { code, a: 0 };
    }
    rotation.apply(residual);
    // a = <s, sign(s)>/sqrt(dim) = ||s||_1 / (||s||_2 * sqrt(dim)).
    let mut l1 = 0.0f64;
    for (i, &value) in residual.iter().enumerate() {
        if value >= 0.0 {
            code[i / 8] |= 1 << (i % 8);
        }
        l1 += value.abs() as f64;
    }
    let a = l1 / (norm * (dim as f64).sqrt());
    if !a.is_finite() {
        return EncodedRow { code, a: 0 };
    }
    // Round DOWN so the decoded interval [a_lo, a_lo + ULP] brackets the
    // true value; clamp to the representable range.
    let a_fixed = (a.clamp(0.0, 1.0) * u16::MAX as f64).floor() as u16;
    EncodedRow { code, a: a_fixed }
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
    out: &mut W,
) -> io::Result<()> {
    KIND_SIGN_RABITQ.serialize(out)?;
    seed.serialize(out)?;
    out.write_all(codes)?;
    for a in a_values {
        a.serialize(out)?;
    }
    Ok(())
}

/// The pinned, parsed sketch slot of one segment-field.
pub struct SketchStore {
    rotation: Rotation,
    /// SoA: all sign codes, `code_stride` bytes each.
    codes: common::OwnedBytes,
    /// SoA: per-row fixed-point `a`.
    a_values: Vec<u16>,
    code_stride: usize,
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
    pub(crate) fn open(
        bytes: common::OwnedBytes,
        dim: usize,
        num_rows: usize,
    ) -> io::Result<SketchStore> {
        let mut reader = bytes.as_slice();
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
        let header = bytes.len() - reader.len();
        let codes_len = num_rows.checked_mul(code_stride).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "sketch codes length overflow")
        })?;
        let expected = codes_len + num_rows * 2;
        if reader.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sketch slot byte length mismatch",
            ));
        }
        let codes = bytes.slice(header..header + codes_len);
        let mut a_reader = &bytes.as_slice()[header + codes_len..];
        let a_values: Vec<u16> = (0..num_rows)
            .map(|_| u16::deserialize(&mut a_reader))
            .collect::<io::Result<_>>()?;
        Ok(SketchStore {
            rotation: Rotation::new(seed, dim),
            codes,
            a_values,
            code_stride,
        })
    }

    /// The shared rotation, for the query side.
    pub fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    /// One row's packed sign code.
    #[inline]
    fn code(&self, row: usize) -> &[u8] {
        &self.codes[row * self.code_stride..(row + 1) * self.code_stride]
    }
}

// ======================================================================
// Query side
// ======================================================================

/// One probed cluster's prepared query: the rotated, normalized,
/// `B_q`-bit-quantized residual direction in the bitplane layout the
/// popcount kernel consumes, plus the EXACT dither norm.
pub struct PreparedSketchQuery {
    /// `QUERY_BITS` bitplanes, each `dim / 64` u64 words: plane `j`
    /// holds bit `j` of every coordinate's quantized level.
    planes: Vec<u64>,
    words: usize,
    /// Quantization grid: `u_hat_i = lo + delta * level_i`.
    lo: f32,
    delta: f32,
    /// `sum(level_i)` — the kernel's constant term.
    level_sum: u32,
    /// `||u - u_hat||^2 / dim`: the dither variance term, exact.
    dither_var: f32,
    dim: usize,
}

impl PreparedSketchQuery {
    /// Prepares one probed cluster's query residual.
    ///
    /// * `rotation` (`&Rotation`) — the store's rotation.
    /// * `residual` (`&mut [f32]`) — `q - c` in the ORIGINAL basis; consumed as scratch.
    ///
    /// Returns (`Option<PreparedSketchQuery>`): `None` when the residual
    /// is degenerate (zero/non-finite norm, or a flat rotated vector) —
    /// the gate then falls back to the radius-only bound.
    pub fn prepare(rotation: &Rotation, residual: &mut [f32]) -> Option<PreparedSketchQuery> {
        let dim = residual.len();
        let norm = residual
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return None;
        }
        rotation.apply(residual);
        let inv = (1.0 / norm) as f32;
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for value in residual.iter_mut() {
            *value *= inv;
            lo = lo.min(*value);
            hi = hi.max(*value);
        }
        if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
            return None;
        }
        let delta = (hi - lo) / QUERY_LEVELS as f32;
        let words = dim / 64;
        let mut planes = vec![0u64; QUERY_BITS as usize * words];
        let mut level_sum = 0u32;
        let mut dither = 0.0f64;
        for (i, &value) in residual.iter().enumerate() {
            let level = (((value - lo) / delta).round() as u32).min(QUERY_LEVELS);
            level_sum += level;
            let decoded = lo + delta * level as f32;
            dither += ((value - decoded) as f64).powi(2);
            for j in 0..QUERY_BITS as usize {
                if level >> j & 1 == 1 {
                    planes[j * words + i / 64] |= 1 << (i % 64);
                }
            }
        }
        Some(PreparedSketchQuery {
            planes,
            words,
            lo,
            delta,
            level_sum,
            dither_var: (dither / dim as f64) as f32,
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

    /// Conservative upper bound on `cos = <u, s>` for one row.
    ///
    /// * `store` (`&SketchStore`) — the segment's sketches.
    /// * `row` (`usize`) — dense row id.
    ///
    /// Returns (`f32`): `min(1, z_hat + eps)` with the `a` interval and
    /// both variance terms folded in conservatively; `1.0` for rows with
    /// no usable sketch (`a = 0`) — the radius-only bound then applies
    /// unchanged. NaN inputs propagate to a non-pruning bound.
    #[inline]
    pub fn cos_upper_bound(&self, store: &SketchStore, row: usize) -> f32 {
        let a_lo = store.a_values[row] as f32 / u16::MAX as f32;
        if a_lo <= 0.0 {
            return 1.0;
        }
        let a_hi = a_lo + 1.0 / u16::MAX as f32;
        let raw = self.code_dot(store.code(row));
        // z_hat = raw / a: the interval endpoint that maximizes it.
        let z_hat = if raw >= 0.0 { raw / a_lo } else { raw / a_hi };
        let eps = SKETCH_T
            * ((1.0 - a_lo * a_lo).max(0.0) / (self.dim as f32 - 1.0) + self.dither_var).sqrt()
            / a_lo;
        (z_hat + eps).min(1.0)
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
            let mut x_res = lcg_vec(&mut state, dim);
            let encoded = encode_row(&rot, &mut x_res.clone());
            let prepared = PreparedSketchQuery::prepare(&rot, &mut q_res).unwrap();
            // Float reference: rebuild s_hat and u_hat by hand.
            rot.apply(&mut x_res);
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

    /// The estimator concentrates: over random directions, the true cos
    /// stays within the conservative bound and the bound stays useful
    /// (well below 1 for uncorrelated pairs).
    #[test]
    fn estimator_bounds_hold() {
        let dim = 1024;
        let rot = Rotation::new(21, dim);
        let store = {
            let mut state = 5u64;
            let mut codes = Vec::new();
            let mut a_values = Vec::new();
            let mut residuals = Vec::new();
            for _ in 0..200 {
                let x = lcg_vec(&mut state, dim);
                let encoded = encode_row(&rot, &mut x.clone());
                codes.extend_from_slice(&encoded.code);
                a_values.push(encoded.a);
                residuals.push(x);
            }
            let mut buf = Vec::new();
            serialize_sketches(21, &codes, &a_values, &mut buf).unwrap();
            (
                SketchStore::open(common::OwnedBytes::new(buf), dim, 200).unwrap(),
                residuals,
            )
        };
        let (store, residuals) = store;
        let mut state = 77u64;
        let mut below_one = 0usize;
        for _ in 0..5 {
            let mut q = lcg_vec(&mut state, dim);
            let q_unrot = q.clone();
            let prepared = PreparedSketchQuery::prepare(&rot, &mut q).unwrap();
            let qn = norm(&q_unrot);
            for (row, x) in residuals.iter().enumerate() {
                let true_cos = dot(&q_unrot, x) / (qn * norm(x));
                let bound = prepared.cos_upper_bound(&store, row);
                assert!(
                    bound >= true_cos - 1e-4,
                    "row {row}: bound {bound} below true cos {true_cos}"
                );
                if bound < 0.9 {
                    below_one += 1;
                }
            }
        }
        // Uncorrelated 1024-dim pairs: the bound must actually bite.
        assert!(
            below_one > 800,
            "bound should be informative for most random pairs, got {below_one}/1000"
        );
    }

    /// Degenerate inputs fail open: zero residuals encode a = 0 which
    /// bounds cos at 1.0; zero query residuals refuse to prepare.
    #[test]
    fn degenerate_inputs_fail_open() {
        let dim = 128;
        let rot = Rotation::new(4, dim);
        let encoded = encode_row(&rot, &mut vec![0.0; dim]);
        assert_eq!(encoded.a, 0);
        let nan = encode_row(&rot, &mut vec![f32::NAN; dim]);
        assert_eq!(nan.a, 0);
        assert!(PreparedSketchQuery::prepare(&rot, &mut vec![0.0; dim]).is_none());

        let mut buf = Vec::new();
        serialize_sketches(4, &encoded.code, &[encoded.a], &mut buf).unwrap();
        let store = SketchStore::open(common::OwnedBytes::new(buf), dim, 1).unwrap();
        let mut q = vec![1.0; dim];
        let prepared = PreparedSketchQuery::prepare(&rot, &mut q).unwrap();
        assert_eq!(prepared.cos_upper_bound(&store, 0), 1.0);
    }

    /// Slot round-trip and the corrupt-length rejections.
    #[test]
    fn slot_validation() {
        let dim = 64;
        let rot = Rotation::new(1, dim);
        let mut residual: Vec<f32> = (0..dim).map(|i| i as f32 - 31.5).collect();
        let encoded = encode_row(&rot, &mut residual);
        let mut buf = Vec::new();
        serialize_sketches(1, &encoded.code, &[encoded.a], &mut buf).unwrap();
        assert!(SketchStore::open(common::OwnedBytes::new(buf.clone()), dim, 1).is_ok());
        assert!(SketchStore::open(common::OwnedBytes::new(buf.clone()), dim, 2).is_err());
        let mut bad_kind = buf.clone();
        bad_kind[0] = 9;
        assert!(SketchStore::open(common::OwnedBytes::new(bad_kind), dim, 1).is_err());
        buf.pop();
        assert!(SketchStore::open(common::OwnedBytes::new(buf), dim, 1).is_err());
    }
}
