//! 4-bit scalar quantization (SQ4) for centroid storage.
//!
//! Each dimension is quantized independently to 16 levels over its
//! `[min, max]` range across the centroid set: `code = round((v - min) /
//! scale)` with `scale = (max - min) / 15`, reconstructed as `min + code *
//! scale`. Two codes pack per byte in a PLANAR block layout (see
//! [`code_at`]), so a row costs `ceil(dim / 2)` bytes — 8× smaller than
//! f32.
//!
//! The reconstruction IS the value everything downstream represents, at
//! two tiers of exactness:
//!
//! - The reference kernels ([`l2_squared_sq4`], [`dot_sq4`], [`cosine_sq4`]) decode codes to
//!   exactly the values [`Sq4Params::decode_row_into`] produces, in the same chunked accumulation
//!   order as the f32 kernels in `distance.rs` — bit-identical to scoring a materialized
//!   reconstruction.
//! - The hot paths score through [`Sq4Scorer`], the ADC form: per-query preparation collapses the
//!   per-row loop to one multiply-accumulate over the codes. The reassociated sums differ from the
//!   reference kernels at float-rounding level, but every scoring path — the RNG build, routing,
//!   margin checks — goes through the one scorer, so the shift is uniform and rankings stay
//!   self-consistent. The bounds fold is unaffected: residuals are measured in f64 against the
//!   decoded reconstruction.
//!
//! Non-finite inputs cannot be encoded: they quantize to code 0 and are
//! excluded from the fit. Callers gate on that (the merge saturates the
//! cluster's bound) rather than this module guessing a policy.

use std::io::{self, Write};

use common::BinarySerializable;

use crate::directory::FileSlice;
use crate::schema::Metric;
use crate::vector::distance::{dot, norm_squared, LANES};
use crate::vector::ivf::NodeId;
use crate::vector::{ArenaScorer, BuildArena, Similarity, VectorArena};

/// Dimensions covered by one packed block: two LANES-wide nibble planes.
const BLOCK_DIMS: usize = 2 * LANES;
/// Bytes of one full packed block.
const BLOCK_BYTES: usize = LANES;

/// Bytes per SQ4 row: two codes per byte; a lone trailing dimension pads
/// into a zero high nibble.
#[inline]
pub const fn sq4_stride(dim: usize) -> usize {
    dim.div_ceil(2)
}

/// The code of dimension `d` in a packed row of `dim` dimensions.
///
/// Rows pack PLANAR, block by block: a block covers `r` consecutive
/// dimensions (`BLOCK_DIMS`, except a trailing partial block) in
/// `half = ceil(r / 2)` bytes, where byte `j` holds dimension `j` of the
/// block in its low nibble and dimension `half + j` in its high nibble.
/// For full blocks `half == LANES`, so `byte & 0x0F` across a block's
/// bytes yields LANES *consecutive* dimensions and `byte >> 4` the next
/// LANES — the kernels' unpack is two straight vector ops with no
/// interleave shuffle.
#[inline(always)]
fn code_at(codes: &[u8], dim: usize, d: usize) -> u8 {
    debug_assert!(d < dim);
    let block = d / BLOCK_DIMS;
    let within = d % BLOCK_DIMS;
    let r = (dim - block * BLOCK_DIMS).min(BLOCK_DIMS);
    let half = r.div_ceil(2);
    let base = block * BLOCK_BYTES;
    if within < half {
        codes[base + within] & 0x0F
    } else {
        codes[base + within - half] >> 4
    }
}

/// Per-dimension SQ4 quantization parameters: `dim` mins and `dim` scales.
///
/// Fit over the set being quantized ([`fit`](Self::fit)); serialized beside
/// the codes so the reader reconstructs the exact same values.
#[derive(Clone, Debug, PartialEq)]
pub struct Sq4Params {
    mins: Vec<f32>,
    scales: Vec<f32>,
}

impl Sq4Params {
    /// Fits per-dimension `[min, max]` ranges over `values`, a flat
    /// `dim`-strided matrix. Non-finite elements are excluded; a dimension
    /// with no finite values (or none at all) gets `min = 0, scale = 0`, so
    /// every row reconstructs to `0.0` there.
    pub fn fit(values: &[f32], dim: usize) -> Sq4Params {
        assert!(dim > 0, "dim must be non-zero");
        assert_eq!(values.len() % dim, 0, "matrix not a multiple of dim");
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for row in values.chunks_exact(dim) {
            for (d, &v) in row.iter().enumerate() {
                if v.is_finite() {
                    mins[d] = mins[d].min(v);
                    maxs[d] = maxs[d].max(v);
                }
            }
        }
        let scales = mins
            .iter_mut()
            .zip(&maxs)
            .map(|(min, &max)| {
                if max < *min {
                    // No finite value seen in this dimension.
                    *min = 0.0;
                    return 0.0;
                }
                let range = max as f64 - *min as f64;
                // A range this wide would overflow `min + code * scale`
                // in f32 (or come within rounding error of it). No real
                // embedding dimension is near it; degrade to a constant
                // midpoint so reconstructions stay finite.
                if range > f32::MAX as f64 / 16.0 {
                    *min = ((*min as f64 + max as f64) / 2.0) as f32;
                    return 0.0;
                }
                (range / 15.0) as f32
            })
            .collect();
        Sq4Params { mins, scales }
    }

    pub fn dim(&self) -> usize {
        self.mins.len()
    }

    /// Reconstruction of `code` in dimension `d` — the single expression
    /// every decode path and kernel in this module shares.
    #[inline(always)]
    fn reconstruct(&self, d: usize, code: u8) -> f32 {
        self.mins[d] + code as f32 * self.scales[d]
    }

    /// The quantized code of value `v` in dimension `d`: clamped into the
    /// fitted range; non-finite values encode as code 0.
    #[inline]
    fn code_of(&self, d: usize, v: f32) -> u8 {
        let scale = self.scales[d];
        if scale > 0.0 && v.is_finite() {
            (((v - self.mins[d]) / scale).round()).clamp(0.0, 15.0) as u8
        } else {
            0
        }
    }

    /// Packs `row` into codes appended to `out` (`sq4_stride(dim)` bytes),
    /// in the planar block layout [`code_at`] documents.
    pub fn quantize_row_into(&self, row: &[f32], out: &mut Vec<u8>) {
        let dim = self.dim();
        assert_eq!(row.len(), dim, "row dimension mismatch");
        let mut base = 0;
        while base < dim {
            let r = (dim - base).min(BLOCK_DIMS);
            let half = r.div_ceil(2);
            for j in 0..half {
                let lo = self.code_of(base + j, row[base + j]);
                let hi = if base + half + j < dim {
                    self.code_of(base + half + j, row[base + half + j])
                } else {
                    0
                };
                out.push(lo | (hi << 4));
            }
            base += r;
        }
    }

    /// Decodes a packed row into `out` (`dim` f32s) — the reconstruction
    /// the kernels score against.
    pub fn decode_row_into(&self, codes: &[u8], out: &mut [f32]) {
        let dim = self.dim();
        debug_assert_eq!(codes.len(), sq4_stride(dim));
        assert_eq!(out.len(), dim, "output dimension mismatch");
        for (d, slot) in out.iter_mut().enumerate() {
            *slot = self.reconstruct(d, code_at(codes, dim, d));
        }
    }

    /// Writes `dim` mins then `dim` scales as little-endian f32s.
    pub fn serialize<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        for value in self.mins.iter().chain(&self.scales) {
            value.serialize(out)?;
        }
        Ok(())
    }

    /// Bytes [`serialize`](Self::serialize) writes: `2 * dim` f32s.
    pub const fn serialized_len(dim: usize) -> usize {
        2 * dim * std::mem::size_of::<f32>()
    }

    /// Reads back what [`serialize`](Self::serialize) wrote.
    pub fn deserialize(mut bytes: &[u8], dim: usize) -> io::Result<Sq4Params> {
        let read = |bytes: &mut &[u8]| -> io::Result<Vec<f32>> {
            (0..dim).map(|_| f32::deserialize(bytes)).collect()
        };
        let mins = read(&mut bytes)?;
        let scales = read(&mut bytes)?;
        Ok(Sq4Params { mins, scales })
    }
}

// =====================================================================
// Kernels: f32 query × packed SQ4 row, accumulating into the same
// LANES-wide sums in the same 16-dim chunk order as the f32 kernels in
// `distance.rs`, so each kernel is bit-identical to its f32 counterpart
// applied to the decoded reconstruction. The planar block layout makes
// a full block's unpack two straight vector ops (`b & 0x0F`, `b >> 4`),
// each yielding LANES consecutive dimensions; the boundary chunk and
// scalar tail go through `code_at`.
//
// `sq4_kernel!` expands the shared walk; per-kernel arithmetic plugs in
// as `lane(q, recon) -> term` over the caller's accumulator element.
// =====================================================================

macro_rules! sq4_kernel {
    ($query:expr, $codes:expr, $params:expr, $zero:expr, $lane:expr, $fold:expr) => {{
        let params: &Sq4Params = $params;
        let query: &[f32] = $query;
        let codes: &[u8] = $codes;
        let dim = params.dim();
        debug_assert_eq!(query.len(), dim, "query dimension mismatch");
        debug_assert_eq!(codes.len(), sq4_stride(dim), "code length mismatch");

        let mut sums = [$zero; LANES];
        let full_blocks = dim / BLOCK_DIMS;
        for b in 0..full_blocks {
            let bc = &codes[b * BLOCK_BYTES..][..BLOCK_BYTES];
            let qc = &query[b * BLOCK_DIMS..][..BLOCK_DIMS];
            let mc = &params.mins[b * BLOCK_DIMS..][..BLOCK_DIMS];
            let sc = &params.scales[b * BLOCK_DIMS..][..BLOCK_DIMS];
            for i in 0..LANES {
                let code = (bc[i] & 0x0F) as f32;
                sums[i] = $fold(sums[i], $lane(qc[i], mc[i] + code * sc[i]));
            }
            for i in 0..LANES {
                let code = (bc[i] >> 4) as f32;
                sums[i] = $fold(
                    sums[i],
                    $lane(qc[LANES + i], mc[LANES + i] + code * sc[LANES + i]),
                );
            }
        }
        // The half-block boundary chunk, if the remainder holds one: the
        // f32 kernels chunk by LANES, so parity needs it accumulated into
        // `sums`, not the scalar tail.
        let mut chunked = full_blocks * BLOCK_DIMS;
        if dim - chunked >= LANES {
            for i in 0..LANES {
                let d = chunked + i;
                let recon = params.reconstruct(d, code_at(codes, dim, d));
                sums[i] = $fold(sums[i], $lane(query[d], recon));
            }
            chunked += LANES;
        }
        let mut acc = $zero;
        for s in sums {
            acc = $fold(acc, s);
        }
        for d in chunked..dim {
            let recon = params.reconstruct(d, code_at(codes, dim, d));
            acc = $fold(acc, $lane(query[d], recon));
        }
        acc
    }};
}

/// `l2_squared` of `query` against a packed row's reconstruction.
#[inline]
pub fn l2_squared_sq4(query: &[f32], codes: &[u8], params: &Sq4Params) -> f32 {
    sq4_kernel!(
        query,
        codes,
        params,
        0f32,
        |q: f32, v: f32| {
            let d = q - v;
            d * d
        },
        |acc: f32, term: f32| acc + term
    )
}

/// `dot` of `query` with a packed row's reconstruction.
#[inline]
pub fn dot_sq4(query: &[f32], codes: &[u8], params: &Sq4Params) -> f32 {
    sq4_kernel!(
        query,
        codes,
        params,
        0f32,
        |q: f32, v: f32| q * v,
        |acc: f32, term: f32| acc + term
    )
}

/// `norm_squared` of a packed row's reconstruction, with the same wide
/// (f64) per-lane accumulation as `norm_squared_wide` — the same block
/// walk as `sq4_kernel!`, minus the query side.
#[inline]
pub fn norm_squared_sq4(codes: &[u8], params: &Sq4Params) -> f32 {
    let dim = params.dim();
    debug_assert_eq!(codes.len(), sq4_stride(dim), "code length mismatch");

    let square = |v: f32| {
        let v = v as f64;
        v * v
    };
    let mut sums = [0f64; LANES];
    let full_blocks = dim / BLOCK_DIMS;
    for b in 0..full_blocks {
        let bc = &codes[b * BLOCK_BYTES..][..BLOCK_BYTES];
        let mc = &params.mins[b * BLOCK_DIMS..][..BLOCK_DIMS];
        let sc = &params.scales[b * BLOCK_DIMS..][..BLOCK_DIMS];
        for i in 0..LANES {
            let code = (bc[i] & 0x0F) as f32;
            sums[i] += square(mc[i] + code * sc[i]);
        }
        for i in 0..LANES {
            let code = (bc[i] >> 4) as f32;
            sums[i] += square(mc[LANES + i] + code * sc[LANES + i]);
        }
    }
    let mut chunked = full_blocks * BLOCK_DIMS;
    if dim - chunked >= LANES {
        for i in 0..LANES {
            let d = chunked + i;
            sums[i] += square(params.reconstruct(d, code_at(codes, dim, d)));
        }
        chunked += LANES;
    }
    let mut acc = 0f64;
    for s in sums {
        acc += s;
    }
    for d in chunked..dim {
        acc += square(params.reconstruct(d, code_at(codes, dim, d)));
    }
    acc as f32
}

/// `cosine` of `query` with a packed row's reconstruction; 0.0 if either
/// side has zero norm, mirroring [`cosine`](crate::vector::cosine).
#[inline]
pub fn cosine_sq4(query: &[f32], codes: &[u8], params: &Sq4Params) -> f32 {
    let nq = norm_squared(query).sqrt();
    let nd = norm_squared_sq4(codes, params).sqrt();
    if nq == 0.0 || nd == 0.0 {
        return 0.0;
    }
    dot_sq4(query, codes, params) / (nq * nd)
}

/// [`Metric::similarity`] against a packed row — the one place the three
/// kernels fold into similarity space (L2 negated), mirroring
/// `Metric::similarity_bytes`.
#[inline]
pub fn similarity_sq4(
    metric: Metric,
    query: &[f32],
    codes: &[u8],
    params: &Sq4Params,
) -> Similarity {
    Similarity::new(match metric {
        Metric::L2 => -l2_squared_sq4(query, codes, params),
        Metric::Cosine => cosine_sq4(query, codes, params),
        Metric::Dot => dot_sq4(query, codes, params),
    })
}

/// Per-centroid `||recon||²` values, computed once from the codes: the
/// row-side constant of the ADC identities (see [`Sq4Scorer`]), pinned
/// beside the codes so no scoring path recomputes a row norm.
///
/// Deterministic: the same fold as `norm_squared` over the decoded row
/// ([`norm_squared_sq4`]), so every producer — the merge, an open, a test
/// — derives bit-identical values.
pub fn sq4_row_norms(codes: &[u8], params: &Sq4Params) -> Vec<f32> {
    let stride = sq4_stride(params.dim());
    debug_assert_eq!(codes.len() % stride, 0, "codes not a multiple of stride");
    codes
        .chunks_exact(stride)
        .map(|row| norm_squared_sq4(row, params))
        .collect()
}

/// A [`VectorArena`] over in-memory packed SQ4 rows — the build-time
/// counterpart of a `&[f32]` arena. `Copy`, so the RNG build can hand it
/// across executor threads. `norms_sq` holds each row's `||recon||²`
/// ([`sq4_row_norms`]).
#[derive(Clone, Copy)]
pub struct Sq4Arena<'a> {
    codes: &'a [u8],
    params: &'a Sq4Params,
    norms_sq: &'a [f32],
}

impl<'a> Sq4Arena<'a> {
    /// Wraps `codes`, a contiguous run of `sq4_stride(params.dim())`-byte
    /// rows, with their precomputed row norms.
    pub fn new(codes: &'a [u8], params: &'a Sq4Params, norms_sq: &'a [f32]) -> Self {
        let stride = sq4_stride(params.dim());
        debug_assert_eq!(
            codes.len() % stride,
            0,
            "arena not a multiple of the row stride"
        );
        assert_eq!(
            norms_sq.len(),
            codes.len() / stride,
            "row norm count mismatch"
        );
        Sq4Arena {
            codes,
            params,
            norms_sq,
        }
    }

    #[inline]
    fn row(&self, node: NodeId) -> &'a [u8] {
        let stride = sq4_stride(self.params.dim());
        &self.codes[node as usize * stride..][..stride]
    }
}

impl VectorArena for Sq4Arena<'_> {
    type Elem = f32;
    type Scorer<'s>
        = Sq4Scorer<'s>
    where Self: 's;

    #[inline]
    fn num_vectors(&self, dim: usize) -> usize {
        debug_assert_eq!(dim, self.params.dim(), "arena dimension mismatch");
        self.codes.len() / sq4_stride(dim)
    }

    #[inline]
    fn scorer<'s>(&'s self, metric: Metric, dim: usize, query: &'s [f32]) -> Sq4Scorer<'s> {
        debug_assert_eq!(dim, self.params.dim(), "arena dimension mismatch");
        Sq4Scorer::new(
            metric,
            Sq4Rows::Slice(self.codes),
            self.params,
            self.norms_sq,
            query,
        )
    }
}

impl BuildArena for Sq4Arena<'_> {
    #[inline(always)]
    fn coord(&self, dim: usize, node: NodeId, d: usize) -> f32 {
        debug_assert_eq!(dim, self.params.dim(), "arena dimension mismatch");
        self.params.reconstruct(d, code_at(self.row(node), dim, d))
    }

    #[inline]
    fn decode_into(&self, dim: usize, node: NodeId, out: &mut [f32]) {
        debug_assert_eq!(dim, self.params.dim(), "arena dimension mismatch");
        self.params.decode_row_into(self.row(node), out);
    }
}

/// A [`VectorArena`] over packed SQ4 rows behind a [`FileSlice`] — the
/// query-time counterpart of `FileSliceArena<f32>`: each score fetches
/// only that node's `sq4_stride(dim)` bytes. The params (2 × dim f32s)
/// and per-row norms (one f32 per row, from the `.centroids` slot) are
/// pinned.
#[derive(Clone)]
pub struct Sq4SliceArena {
    slice: FileSlice,
    params: Sq4Params,
    norms_sq: Vec<f32>,
}

impl Sq4SliceArena {
    pub fn new(slice: FileSlice, params: Sq4Params, norms_sq: Vec<f32>) -> Self {
        use common::HasLen;
        assert_eq!(
            norms_sq.len(),
            slice.len() / sq4_stride(params.dim()),
            "row norm count mismatch"
        );
        Sq4SliceArena {
            slice,
            params,
            norms_sq,
        }
    }

    pub fn params(&self) -> &Sq4Params {
        &self.params
    }
}

impl VectorArena for Sq4SliceArena {
    type Elem = f32;
    type Scorer<'s>
        = Sq4Scorer<'s>
    where Self: 's;

    #[inline]
    fn num_vectors(&self, dim: usize) -> usize {
        use common::HasLen;
        debug_assert_eq!(dim, self.params.dim(), "arena dimension mismatch");
        let stride = sq4_stride(dim);
        assert_eq!(
            self.slice.len() % stride,
            0,
            "arena not a multiple of stride"
        );
        self.slice.len() / stride
    }

    #[inline]
    fn scorer<'s>(&'s self, metric: Metric, dim: usize, query: &'s [f32]) -> Sq4Scorer<'s> {
        debug_assert_eq!(dim, self.params.dim(), "arena dimension mismatch");
        Sq4Scorer::new(
            metric,
            Sq4Rows::File(&self.slice),
            &self.params,
            &self.norms_sq,
            query,
        )
    }
}

/// The two row-storage shapes a [`Sq4Scorer`] reads codes from.
enum Sq4Rows<'a> {
    Slice(&'a [u8]),
    File(&'a FileSlice),
}

/// One query's ADC scorer over packed SQ4 rows.
///
/// Uses the identity `dot(q, recon) = Σ q·min + Σ (q·scale)·code`: the
/// query folds with the params ONCE ([`Sq4Scorer::new`]) into `t = q·scale`
/// and the scalar `qm = Σ q·min`, so the per-row loop is one
/// multiply-accumulate over the codes — no min/scale loads, no dequant
/// FMA. L2 and cosine reduce to the same loop through the precomputed row
/// norms: `||q - recon||² = ||q||² - 2·dot + ||recon||²` (clamped at 0
/// against cancellation) and `cos = dot / (||q||·||recon||)`.
///
/// Scores agree bit-exactly across scorer instances built from the same
/// `(metric, query)` — the [`VectorArena::scorer`] determinism contract —
/// but differ from scoring the decoded reconstruction with the f32
/// kernels at float-rounding level: the expansion reassociates the sums.
/// Build, routing, and margin checks all score through this one path, so
/// the shift is uniform and self-consistent.
pub struct Sq4Scorer<'a> {
    metric: Metric,
    dim: usize,
    rows: Sq4Rows<'a>,
    norms_sq: &'a [f32],
    /// `q[d] * scale[d]`.
    t: Vec<f32>,
    /// `Σ q[d] * min[d]`.
    qm: f32,
    /// `||q||²`.
    qq: f32,
    /// `||q||`, for cosine.
    q_norm: f32,
}

impl<'a> Sq4Scorer<'a> {
    fn new(
        metric: Metric,
        rows: Sq4Rows<'a>,
        params: &'a Sq4Params,
        norms_sq: &'a [f32],
        query: &'a [f32],
    ) -> Self {
        let dim = params.dim();
        debug_assert_eq!(query.len(), dim, "query dimension mismatch");
        let t: Vec<f32> = query
            .iter()
            .zip(&params.scales)
            .map(|(&q, &s)| q * s)
            .collect();
        let qm = dot(query, &params.mins);
        let qq = norm_squared(query);
        Sq4Scorer {
            metric,
            dim,
            rows,
            norms_sq,
            t,
            qm,
            qq,
            q_norm: qq.sqrt(),
        }
    }

    /// `dot(q, recon)` of one packed row: `qm + Σ t·code`, walked in the
    /// planar block layout — the full blocks through the vectorized core,
    /// then the half-block boundary chunk (accumulated into the lane sums
    /// before the fold, matching the LANES-chunk order) and the scalar
    /// tail.
    #[inline]
    fn dot_row(&self, codes: &[u8]) -> f32 {
        let dim = self.dim;
        debug_assert_eq!(codes.len(), sq4_stride(dim), "code length mismatch");
        let t = self.t.as_slice();
        let full_blocks = dim / BLOCK_DIMS;
        let mut sums = adc_full_blocks(codes, t, full_blocks);
        let mut chunked = full_blocks * BLOCK_DIMS;
        if dim - chunked >= LANES {
            for i in 0..LANES {
                let d = chunked + i;
                sums[i] += code_at(codes, dim, d) as f32 * t[d];
            }
            chunked += LANES;
        }
        let mut acc = self.qm;
        for s in sums {
            acc += s;
        }
        for d in chunked..dim {
            acc += code_at(codes, dim, d) as f32 * t[d];
        }
        acc
    }

    #[inline]
    fn score_row(&self, node: NodeId, codes: &[u8]) -> Similarity {
        let dot = self.dot_row(codes);
        Similarity::new(match self.metric {
            Metric::Dot => dot,
            Metric::L2 => -(self.qq - 2.0 * dot + self.norms_sq[node as usize]).max(0.0),
            Metric::Cosine => {
                let n_sq = self.norms_sq[node as usize];
                // Either side zero-norm scores 0.0 (the `cosine`
                // convention); a NaN row norm lands there too and fails
                // to a harmless constant.
                if self.q_norm == 0.0 || n_sq.is_nan() || n_sq <= 0.0 {
                    0.0
                } else {
                    dot / (self.q_norm * n_sq.sqrt())
                }
            }
        })
    }
}

impl ArenaScorer for Sq4Scorer<'_> {
    /// # Panics
    ///
    /// File-backed rows panic on a failed read, same contract as
    /// [`FileSliceScorer`](crate::vector::FileSliceScorer).
    #[inline]
    fn score(&self, node: NodeId) -> Similarity {
        let stride = sq4_stride(self.dim);
        match &self.rows {
            Sq4Rows::Slice(codes) => {
                self.score_row(node, &codes[node as usize * stride..][..stride])
            }
            Sq4Rows::File(slice) => {
                let bytes = slice
                    .slice(node as usize * stride..(node as usize + 1) * stride)
                    .read_bytes()
                    .expect("failed to read sq4 arena row");
                self.score_row(node, &bytes)
            }
        }
    }
}

/// The vectorizable core of [`Sq4Scorer::dot_row`]: `full_blocks` planar
/// blocks of `Σ t·code`, one lane-sum array out.
///
/// Deliberately a separate, never-inlined function: fused into `dot_row`
/// beside the scalar boundary/tail handling, LLVM abandons vectorizing
/// the whole body and emits scalar unpack/convert/FMA chains (measured
/// 3× slower); isolated, it compiles to the intended `and.16b`/`ushr` +
/// widen + `ucvtf.4s` + `fmul.4s`/`fadd.4s` loops. The unpack+convert
/// loop is split from the FMA loops for the same reason.
#[inline(never)]
fn adc_full_blocks(codes: &[u8], t: &[f32], full_blocks: usize) -> [f32; LANES] {
    let mut sums = [0f32; LANES];
    for b in 0..full_blocks {
        // Fixed-size views: known trip counts and no bounds checks in
        // the lane loops.
        let bc: &[u8; BLOCK_BYTES] = codes[b * BLOCK_BYTES..][..BLOCK_BYTES]
            .try_into()
            .expect("exact block");
        let tc: &[f32; BLOCK_DIMS] = t[b * BLOCK_DIMS..][..BLOCK_DIMS]
            .try_into()
            .expect("exact block");
        let mut lo = [0f32; LANES];
        let mut hi = [0f32; LANES];
        for i in 0..LANES {
            lo[i] = (bc[i] & 0x0F) as f32;
            hi[i] = (bc[i] >> 4) as f32;
        }
        for i in 0..LANES {
            sums[i] += lo[i] * tc[i];
        }
        for i in 0..LANES {
            sums[i] += hi[i] * tc[LANES + i];
        }
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::{cosine, dot, l2_squared};

    fn sample_matrix(rows: usize, dim: usize) -> Vec<f32> {
        (0..rows * dim)
            .map(|i| ((i as f32) * 0.137).sin() * (1.0 + (i % 7) as f32))
            .collect()
    }

    fn quantize_all(params: &Sq4Params, values: &[f32], dim: usize) -> Vec<u8> {
        let mut codes = Vec::new();
        for row in values.chunks_exact(dim) {
            params.quantize_row_into(row, &mut codes);
        }
        codes
    }

    fn decode_all(params: &Sq4Params, codes: &[u8], rows: usize) -> Vec<f32> {
        let dim = params.dim();
        let stride = sq4_stride(dim);
        let mut out = vec![0f32; rows * dim];
        for r in 0..rows {
            params.decode_row_into(&codes[r * stride..][..stride], &mut out[r * dim..][..dim]);
        }
        out
    }

    #[test]
    fn roundtrip_error_within_half_step() {
        for dim in [1, 2, 7, 16, 33] {
            let values = sample_matrix(20, dim);
            let params = Sq4Params::fit(&values, dim);
            let codes = quantize_all(&params, &values, dim);
            assert_eq!(codes.len(), 20 * sq4_stride(dim));
            let recon = decode_all(&params, &codes, 20);
            for (d, (&v, &r)) in values.iter().zip(&recon).enumerate() {
                let scale = params.scales[d % dim];
                assert!(
                    (v - r).abs() <= scale * 0.5 + 1e-6,
                    "dim {dim}, elem {d}: {v} -> {r}, scale {scale}"
                );
            }
        }
    }

    #[test]
    fn kernels_bit_identical_to_f32_over_reconstruction() {
        for dim in [1, 5, 16, 21, 32, 37] {
            let values = sample_matrix(8, dim);
            let params = Sq4Params::fit(&values, dim);
            let codes = quantize_all(&params, &values, dim);
            let recon = decode_all(&params, &codes, 8);
            let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.31 - 2.0).collect();
            let stride = sq4_stride(dim);
            for r in 0..8 {
                let row_codes = &codes[r * stride..][..stride];
                let row_recon = &recon[r * dim..][..dim];
                assert_eq!(
                    l2_squared_sq4(&query, row_codes, &params).to_bits(),
                    l2_squared(&query, row_recon).to_bits(),
                    "l2 mismatch at dim {dim}, row {r}"
                );
                assert_eq!(
                    dot_sq4(&query, row_codes, &params).to_bits(),
                    dot(&query, row_recon).to_bits(),
                    "dot mismatch at dim {dim}, row {r}"
                );
                assert_eq!(
                    cosine_sq4(&query, row_codes, &params).to_bits(),
                    cosine(&query, row_recon).to_bits(),
                    "cosine mismatch at dim {dim}, row {r}"
                );
            }
        }
    }

    /// |a - b| within float-rounding slack of the ADC reassociation:
    /// absolute tolerance scaled by the magnitudes the expansion
    /// subtracts through.
    fn assert_close(a: f32, b: f32, scale: f32, ctx: &str) {
        let tol = 1e-5 * scale.abs().max(1.0);
        assert!((a - b).abs() <= tol, "{ctx}: {a} vs {b} (tol {tol})");
    }

    /// The ADC scorer reassociates the sums, so it matches scoring the
    /// reconstruction within float rounding — and is bit-deterministic
    /// across scorer instances (the routing/margin consistency contract).
    #[test]
    fn scorer_close_to_reconstruction_and_deterministic() {
        for dim in [1, 5, 16, 21, 32, 37, 48] {
            let rows = 6;
            let values = sample_matrix(rows, dim);
            let params = Sq4Params::fit(&values, dim);
            let codes = quantize_all(&params, &values, dim);
            let recon = decode_all(&params, &codes, rows);
            let norms = sq4_row_norms(&codes, &params);
            let arena = Sq4Arena::new(&codes, &params, &norms);
            let query: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
            assert_eq!(arena.num_vectors(dim), rows);
            let qq = crate::vector::distance::norm_squared(&query);
            for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
                let scorer_a = arena.scorer(metric, dim, &query);
                let scorer_b = arena.scorer(metric, dim, &query);
                for node in 0..rows as u32 {
                    let expected = metric
                        .similarity(query.as_slice(), &recon[node as usize * dim..][..dim])
                        .score();
                    let got = scorer_a.score(node).score();
                    let scale = match metric {
                        Metric::Cosine => 1.0,
                        _ => qq + norms[node as usize],
                    };
                    assert_close(got, expected, scale, &format!("{metric:?} node {node}"));
                    assert_eq!(
                        got.to_bits(),
                        scorer_b.score(node).score().to_bits(),
                        "{metric:?} node {node}: scorers must agree bitwise"
                    );
                }
            }
        }
    }

    #[test]
    fn constant_dimension_reconstructs_exactly() {
        let dim = 3;
        // Dim 1 constant at 4.25 across rows.
        let values = vec![1.0, 4.25, -2.0, 3.0, 4.25, 5.0];
        let params = Sq4Params::fit(&values, dim);
        assert_eq!(params.scales[1], 0.0);
        let codes = quantize_all(&params, &values, dim);
        let recon = decode_all(&params, &codes, 2);
        assert_eq!(recon[1], 4.25);
        assert_eq!(recon[4], 4.25);
    }

    #[test]
    fn non_finite_values_excluded_from_fit_and_encode_as_zero() {
        let dim = 2;
        let values = vec![f32::NAN, 1.0, 3.0, f32::INFINITY, 5.0, 2.0];
        let params = Sq4Params::fit(&values, dim);
        // Dim 0 fit over {3.0, 5.0}; dim 1 over {1.0, 2.0}.
        assert_eq!(params.mins, vec![3.0, 1.0]);
        let codes = quantize_all(&params, &values, dim);
        let recon = decode_all(&params, &codes, 3);
        // NaN/inf encode as code 0 => reconstruct to the dimension min.
        assert_eq!(recon[0], 3.0);
        assert_eq!(recon[3], 1.0);
        assert!(recon.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn all_non_finite_dimension_reconstructs_to_zero() {
        let values = vec![f32::NAN, 1.0, f32::NEG_INFINITY, 2.0];
        let params = Sq4Params::fit(&values, 2);
        assert_eq!(params.mins[0], 0.0);
        assert_eq!(params.scales[0], 0.0);
        let codes = quantize_all(&params, &values, 2);
        let recon = decode_all(&params, &codes, 2);
        assert_eq!(recon[0], 0.0);
        assert_eq!(recon[2], 0.0);
    }

    #[test]
    fn extreme_range_degrades_to_finite_midpoint() {
        let values = vec![f32::MAX, -f32::MAX];
        let params = Sq4Params::fit(&values, 1);
        assert_eq!(params.scales[0], 0.0);
        let codes = quantize_all(&params, &values, 1);
        let recon = decode_all(&params, &codes, 2);
        assert!(recon.iter().all(|v| v.is_finite()));
        assert_eq!(recon, vec![0.0, 0.0], "midpoint of the symmetric range");
    }

    #[test]
    fn large_but_sane_range_quantizes_normally() {
        let values = vec![1e36_f32, -1e36];
        let params = Sq4Params::fit(&values, 1);
        assert!(params.scales[0] > 0.0);
        let codes = quantize_all(&params, &values, 1);
        let recon = decode_all(&params, &codes, 2);
        assert!(recon.iter().all(|v| v.is_finite()));
        assert!(recon[0] > 0.0 && recon[1] < 0.0);
    }

    #[test]
    fn params_serialization_roundtrip() {
        let values = sample_matrix(5, 9);
        let params = Sq4Params::fit(&values, 9);
        let mut buf = Vec::new();
        params.serialize(&mut buf).unwrap();
        assert_eq!(buf.len(), Sq4Params::serialized_len(9));
        let back = Sq4Params::deserialize(&buf, 9).unwrap();
        assert_eq!(params, back);
    }

    #[test]
    fn odd_dim_pads_high_nibble_with_zero() {
        let values = vec![1.0, 2.0, 3.0];
        let params = Sq4Params::fit(&values, 3);
        let mut codes = Vec::new();
        params.quantize_row_into(&values, &mut codes);
        assert_eq!(codes.len(), 2);
        assert_eq!(codes[1] >> 4, 0, "padding nibble must be zero");
    }

    /// The RNG build over codes is deterministic, and searching it with
    /// every node seeded returns exactly the scorer's global ranking —
    /// with all nodes visited up front, the beam reduces to a global
    /// top-k, so this pins build/search/scorer self-consistency without
    /// depending on edge quality.
    #[test]
    fn rng_build_over_codes_is_deterministic_and_search_consistent() {
        use crate::vector::{NeighborhoodGraphConfig, RelativeNeighborhoodGraph, Workspace};
        use crate::Executor;

        let dim = 12;
        let rows = 300;
        let values = sample_matrix(rows, dim);
        let params = Sq4Params::fit(&values, dim);
        let codes = quantize_all(&params, &values, dim);
        let norms = sq4_row_norms(&codes, &params);
        let arena = Sq4Arena::new(&codes, &params, &norms);
        let executor = Executor::single_thread();
        let config = NeighborhoodGraphConfig::default();
        let all_seeds: Vec<NodeId> = (0..rows as NodeId).collect();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.61).sin()).collect();

        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let mut first = RelativeNeighborhoodGraph::new(arena, dim, metric, config);
            first.build(&executor);
            let mut second = RelativeNeighborhoodGraph::new(arena, dim, metric, config);
            second.build(&executor);
            let mut first_bytes = Vec::new();
            first.serialize(&mut first_bytes).unwrap();
            let mut second_bytes = Vec::new();
            second.serialize(&mut second_bytes).unwrap();
            assert_eq!(
                first_bytes, second_bytes,
                "build not deterministic: {metric:?}"
            );

            let scorer = arena.scorer(metric, dim, &query);
            let mut expected: Vec<(Similarity, NodeId)> = (0..rows as NodeId)
                .map(|node| (scorer.score(node), node))
                .collect();
            expected.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut ws = Workspace::new();
            let (got, _) = first.search(&mut ws, &query, &all_seeds, 10);
            let got: Vec<(Similarity, NodeId)> = got.into_iter().map(|c| (c.sim, c.node)).collect();
            assert_eq!(got, expected[..10].to_vec(), "search vs scorer: {metric:?}");
        }
    }

    #[test]
    fn slice_arena_matches_borrowed_arena() {
        let dim = 8;
        let values = sample_matrix(4, dim);
        let params = Sq4Params::fit(&values, dim);
        let codes = quantize_all(&params, &values, dim);
        let norms = sq4_row_norms(&codes, &params);
        let borrowed = Sq4Arena::new(&codes, &params, &norms);
        let sliced = Sq4SliceArena::new(
            FileSlice::from(codes.clone()),
            params.clone(),
            norms.clone(),
        );
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5 - 1.0).collect();
        assert_eq!(sliced.num_vectors(dim), 4);
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            for node in 0..4u32 {
                assert_eq!(
                    sliced.similarity(metric, dim, node, &query),
                    borrowed.similarity(metric, dim, node, &query),
                );
            }
        }
    }
}
