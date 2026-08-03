//! Centroid bounds: the stored cluster-side extents the probe gate
//! certifies skips against.
//!
//! Model: cluster = centroid + stored shape (grows: Ball today, Aabb/...
//! later). Query = position in ctx + the metric's sublevel set at the kth
//! threshold — fully determined by (metric, t), never stored, never
//! shaped. Collision matrix is therefore 2 x K:
//!
//! ```text
//!                 | Ball(r)              | Aabb(h[dim])   (future)
//!   BallRegion    | margin_ball_ball     | margin_ball_box
//!   Halfspace     | margin_ball_hspace   | margin_box_hspace
//! ```
//!
//! Ball-column margins are scalar-only (separation = the routing key,
//! already computed). Box-column margins need the residual / query vector
//! — one pass over dims per check. That is the price of AABB; it is not
//! paid today.
//!
//! This module owns the semantic types; the `.centroids` slot byte format
//! lives with the other slot serializers in [`ivf::index`](super::ivf).

// ======================================================================
// P1: storage
// ======================================================================

use std::io;

/// Segment-level bound shape, captured in the `.centroids` V2 bounds slot
/// at build time — one kind for every cluster of a field's segment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BoundKind {
    /// One `f32` per cluster: max `||x - c||` over NATIVE members, in the
    /// stored representation. Metric-uniform by the cosine
    /// write-normalization invariant (unit members + renormalized
    /// centroid make the residual norm the chord).
    /// `f32::INFINITY` = SATURATED: every margin comes out +inf, the
    /// cluster probes. Fail-open is arithmetic.
    ///
    /// Quantized elements (i8) are a rule, not code, until the dtype
    /// exists: a bound folded over quantized rows must inflate by the
    /// quantization error at write, or a true member can sit outside it.
    Ball = 0,
    // Aabb = 1,   // dim f32 half-extents about the centroid
}

impl BoundKind {
    /// Decodes the on-disk kind byte.
    ///
    /// * `code` (`u8`) — the kind byte leading a bounds slot.
    ///
    /// Returns (`io::Result<BoundKind>`): the kind, or `InvalidData` for a
    /// byte no known kind uses — a corrupt or future-format slot.
    pub fn from_code(code: u8) -> io::Result<BoundKind> {
        match code {
            0 => Ok(BoundKind::Ball),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown centroid bound kind: {other}"),
            )),
        }
    }

    /// Per-cluster payload stride of this kind, in `f32`s.
    ///
    /// * `_dim` (`usize`) — the field's vector dimensionality; unused by `Ball`, consumed by the
    ///   future per-dimension kinds.
    ///
    /// Returns (`usize`): how many `f32`s one cluster's bound occupies.
    pub fn stride(self, _dim: usize) -> usize {
        match self {
            BoundKind::Ball => 1,
            // BoundKind::Aabb => dim,
        }
    }
}

/// Which rows a cluster's stored bound covers. Captured into the stored
/// [`IndexSettings`](crate::index::IndexSettings) at build.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum BoundsScope {
    /// The fold runs over a cluster's NATIVE (primary-assignment) members
    /// only; replica spill is excluded. Sound because a qualifying row's
    /// HOME cluster always fails the skip test — the replica-closure
    /// argument at the gate's property test.
    #[default]
    #[serde(rename = "native")]
    Native,
    // Posting — fold over every posting row incl. replicas — is a named
    // non-goal; the variant lands with its write path.
}

/// Read view over the bounds payload of one segment's `.centroids` field
/// slot.
pub struct BoundStore<'a> {
    kind: BoundKind,
    data: &'a [f32],
}

impl<'a> BoundStore<'a> {
    /// Wraps a decoded bounds payload.
    ///
    /// * `kind` (`BoundKind`) — the slot's segment-level kind byte.
    /// * `data` (`&[f32]`) — the payload, `num_clusters * kind.stride(dim)` values in cluster
    ///   order.
    ///
    /// Returns (`BoundStore`): the view; per-kind accessors validate the
    /// kind on read.
    pub fn new(kind: BoundKind, data: &'a [f32]) -> Self {
        Self { kind, data }
    }

    /// The segment-level bound kind of this store.
    ///
    /// Returns (`BoundKind`): the kind every cluster in the store uses.
    pub fn kind(&self) -> BoundKind {
        self.kind
    }

    /// The ball bound of `cluster`.
    ///
    /// * `cluster` (`usize`) — dense cluster id, `0..num_clusters`.
    ///
    /// Returns (`f32`): max native-member residual norm against the stored
    /// centroid; `f32::INFINITY` = SATURATED (always probes).
    #[inline]
    pub fn ball_r(&self, cluster: usize) -> f32 {
        debug_assert_eq!(self.kind, BoundKind::Ball);
        self.data[cluster]
    }

    /// The raw payload, for serialization and tests.
    ///
    /// Returns (`&[f32]`): the values [`Self::new`] was built over.
    pub fn values(&self) -> &'a [f32] {
        self.data
    }
}

// ---- build ------------------------------------------------------------
//
// The ONLY producer of bounds. The merge path runs this same fold over its
// re-assignment output against the NEW centroids; no bound-combining API
// exists, by design -- folded input radii under merge are unsound.

/// Accumulates one segment's per-cluster ball bounds during a merge.
pub struct BoundsBuilder {
    r: Vec<f32>,
}

impl BoundsBuilder {
    /// Starts a fold with every cluster's bound at `0.0` — exact for
    /// clusters that end up empty or whose members all sit on the
    /// centroid.
    ///
    /// * `num_clusters` (`usize`) — cluster count of the segment being written.
    ///
    /// Returns (`BoundsBuilder`): the zeroed fold state.
    pub fn new(num_clusters: usize) -> Self {
        Self {
            r: vec![0.0; num_clusters],
        }
    }

    /// Folds one native member assigned to `cluster`.
    ///
    /// * `cluster` (`usize`) — the member's HOME (primary) cluster.
    /// * `residual_norm` (`f32`) — `||x - c||` of the member's STORED row against the STORED
    ///   centroid (post-renormalization for cosine). A non-finite residual saturates the cluster.
    pub fn add_native(&mut self, cluster: usize, residual_norm: f32) {
        let slot = &mut self.r[cluster];
        if !residual_norm.is_finite() {
            *slot = f32::INFINITY; // SATURATED
        } else if residual_norm > *slot {
            *slot = residual_norm;
        }
    }

    /// Marks `cluster` SATURATED — its centroid is degenerate (non-finite,
    /// or zero-norm under cosine renormalization), so no residual against
    /// it can certify a skip.
    ///
    /// * `cluster` (`usize`) — the cluster to saturate.
    pub fn saturate(&mut self, cluster: usize) {
        self.r[cluster] = f32::INFINITY;
    }

    /// Finishes the fold.
    ///
    /// Returns (`Vec<f32>`): the per-cluster ball bounds in cluster order —
    /// the [`BoundKind::Ball`] slot payload.
    pub fn finish(self) -> Vec<f32> {
        self.r
    }
}

/// `||x - c||` of one stored row against its stored centroid — the
/// residual the write fold hands to [`BoundsBuilder::add_native`]. One
/// kernel for producer and verifier, so a recomputed fold is bit-equal.
///
/// * `row_bytes` (`&[u8]`) — the member's STORED row, little-endian `f32`s exactly as written
///   (post-renormalization for cosine).
/// * `c` (`&[f32]`) — the STORED centroid values; `row_bytes.len() == 4 * c.len()`.
///
/// Returns (`f32`): the L2 residual norm, accumulated in `f64` so no
/// finite input overflows; non-finite inputs propagate to a non-finite
/// result (which saturates at the builder).
pub fn residual_norm(row_bytes: &[u8], c: &[f32]) -> f32 {
    debug_assert_eq!(row_bytes.len(), std::mem::size_of_val(c));
    let mut acc = 0.0f64;
    for (chunk, &ci) in row_bytes.chunks_exact(4).zip(c.iter()) {
        let xi = f32::from_le_bytes(chunk.try_into().unwrap());
        let d = xi as f64 - ci as f64;
        acc += d * d;
    }
    acc.sqrt() as f32
}

#[cfg(test)]
mod bounds_storage_tests {
    use super::*;

    /// Ball stride is 1 f32/cluster whatever the dimensionality — the
    /// stride table consults the kind, not the field.
    #[test]
    fn stride_table() {
        for dim in [1usize, 2, 128, 1536] {
            assert_eq!(BoundKind::Ball.stride(dim), 1);
        }
    }

    #[test]
    fn kind_byte_round_trips() {
        assert_eq!(
            BoundKind::from_code(BoundKind::Ball as u8).unwrap(),
            BoundKind::Ball
        );
        assert!(BoundKind::from_code(7).is_err());
    }

    /// The builder folds a max, starts exact at 0.0, and saturates on
    /// non-finite residuals and degenerate centroids.
    #[test]
    fn builder_folds_max_and_saturates() {
        let mut builder = BoundsBuilder::new(3);
        builder.add_native(0, 1.0);
        builder.add_native(0, 0.5);
        builder.add_native(0, 2.0);
        builder.add_native(1, f32::NAN); // non-finite member residual
        builder.saturate(2); // degenerate centroid
        let r = builder.finish();
        assert_eq!(r[0], 2.0);
        assert_eq!(r[1], f32::INFINITY);
        assert_eq!(r[2], f32::INFINITY);
    }

    /// A saturated cluster stays saturated: later finite members must not
    /// un-saturate it (max against +inf).
    #[test]
    fn saturation_is_sticky() {
        let mut builder = BoundsBuilder::new(1);
        builder.add_native(0, f32::INFINITY);
        builder.add_native(0, 0.25);
        assert_eq!(builder.finish()[0], f32::INFINITY);
    }

    /// Little-endian `f32` row bytes, as the row store holds them.
    fn row(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn residual_norm_matches_hand_values() {
        assert_eq!(residual_norm(&row(&[3.0, 4.0]), &[0.0, 0.0]), 5.0);
        assert_eq!(residual_norm(&row(&[1.0, 1.0]), &[1.0, 1.0]), 0.0);
        assert!(residual_norm(&row(&[f32::NAN, 0.0]), &[0.0, 0.0]).is_nan());
        assert_eq!(
            residual_norm(&row(&[f32::INFINITY, 0.0]), &[0.0, 0.0]),
            f32::INFINITY
        );
    }

    #[test]
    fn bounds_scope_serde_is_the_reloption_token() {
        // The reloption's only legal value round-trips as "native".
        let json = serde_json::to_string(&BoundsScope::Native).unwrap();
        assert_eq!(json, "\"native\"");
        let back: BoundsScope = serde_json::from_str(&json).unwrap();
        assert_eq!(back, BoundsScope::Native);
    }
}
