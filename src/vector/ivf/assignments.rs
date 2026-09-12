//! Assignments: computing a vector's cells against the index-level
//! centroid index, and the persisted per-segment result,
//! [`SegmentClusters`].
//!
//! WRITE SIDE — both write paths (per-commit serialize and merge) assign
//! every vector against the same frozen centroid index, taking the
//! primary cell and the `replicas - 1` next-nearest cells using batched
//! matrix multiplication. Centroids are materialized on the calling thread
//! before assignment, so the numerical kernel never accesses the Directory.
//!
//! READ SIDE — a segment keeps only what is genuinely per-segment: which
//! rows landed in which cluster, and the residual geometry of those
//! rows. This module owns the wire format of those `.vec` slots end to
//! end: the serializers the write paths call and the
//! [`SegmentClusters::open`] that parses them back sit side by side.
//!
//! The `.vec` composite slots (see `vector::header::vec_slot`):
//!
//! ```text
//! [2] cluster_offsets (u64[C+1], prefix sum over the cluster-sorted rows)
//! [3] centroid bounds, REQUIRED: a segment-level BoundKind byte, then
//!     C · stride(kind) f32s in cluster order — for Ball, one f32 per
//!     cluster: max ||x - c|| over the cluster's NATIVE members' stored
//!     rows against the centroid index's stored centroid (replica spill is excluded
//!     per the stored `bounds_scope = native`)
//! [4] IVF meta: num_docs (u32) + num_centroids (u32)
//! ```
//!
//! One dense `centroid_id = 0..C` indexes the set file's rows and these
//! slots alike: `cluster_offsets[c]` is the first row of cluster `c` in the
//! parallel `.vec` rows/`IdMap`.

use std::io::{self, Write};
use std::mem;
use std::ops::Range;

use common::{BinarySerializable, HasLen, OwnedBytes};
use superkmeans::gemm::sgemm_row_major_b_transposed;

use crate::collector::sort_key::NaturalComparator;
use crate::collector::TopNComputer;
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::{BoundKind, BoundStore, Similarity};

/// The per-segment IVF remainder for one field: which contiguous row
/// ranges of the `.vec` rows form each cluster, plus the per-cluster
/// bounds.
///
/// Everything here is small and pinned; the row-scale payload (rows,
/// id-map) lives on [`VectorIndexReader`](crate::vector::VectorIndexReader),
/// and the centroid rows live in the index-level set file.
pub struct SegmentClusters {
    num_centroids: usize,
    /// Distinct documents with a vector in this field. Rows including
    /// replicas are [`Self::num_rows`].
    num_docs: usize,
    /// Slot `[2]`: the `u64[C+1]` prefix sum, pinned.
    cluster_offsets: OwnedBytes,
    /// Slot `[3]`, pinned: the segment-level bound kind.
    bound_kind: BoundKind,
    /// Slot `[3]`, pinned: the per-cluster bound payload,
    /// `num_centroids * bound_kind.stride(dim)` f32s in cluster order.
    bounds: Vec<f32>,
    /// Derived at open (one pass over the offsets): bit `c` set ⟺ cluster
    /// `c` has rows in THIS segment. The probe loop's presence check
    /// touches one bit instead of two random u64s of the offsets array.
    non_empty: Vec<u64>,
    /// Count of set bits in [`Self::non_empty`] — this segment's share of
    /// the index's open-charge capacity.
    num_non_empty: usize,
}

impl SegmentClusters {
    /// Write slot `[2]` of the `.vec` composite for a field.
    pub(crate) fn serialize_offsets<W: Write + ?Sized>(
        cluster_offsets: &[u64],
        out: &mut W,
    ) -> io::Result<()> {
        for offset in cluster_offsets {
            offset.serialize(out)?;
        }
        Ok(())
    }

    /// Write slot `[3]` of the `.vec` composite for a field: the
    /// segment-level kind byte, then the per-cluster payload.
    ///
    /// * `kind` (`BoundKind`) — the segment-level bound kind.
    /// * `values` (`&[f32]`) — `num_centroids * kind.stride(dim)` values in cluster order; the
    ///   caller's [`BoundsBuilder`] output.
    /// * `out` (`&mut W`) — the slot writer.
    ///
    /// Returns (`io::Result<()>`): write errors only — the payload length
    /// is validated at open, against the count words of slot `[4]`.
    ///
    /// [`BoundsBuilder`]: crate::vector::BoundsBuilder
    pub(crate) fn serialize_bounds<W: Write + ?Sized>(
        kind: BoundKind,
        values: &[f32],
        out: &mut W,
    ) -> io::Result<()> {
        (kind as u8).serialize(out)?;
        for value in values {
            value.serialize(out)?;
        }
        Ok(())
    }

    /// Write slot `[4]` of the `.vec` composite for a field. `num_docs` is
    /// the number of distinct docs assigned — NOT the posting-row total,
    /// which replication can multiply.
    pub(crate) fn serialize_ivf_meta<W: Write + ?Sized>(
        num_docs: usize,
        num_centroids: usize,
        out: &mut W,
    ) -> io::Result<()> {
        u32::try_from(num_docs)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "doc count exceeds u32"))?
            .serialize(out)?;
        u32::try_from(num_centroids)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "centroid count exceeds u32"))?
            .serialize(out)
    }

    /// Parse a field's per-segment IVF slots. Everything is materialized and
    /// pinned — offsets, bounds, and the meta words are all `O(C)`.
    pub(crate) fn open(
        options: &VectorOptions,
        offsets_slice: FileSlice,
        bounds_slice: FileSlice,
        meta_slice: FileSlice,
    ) -> crate::Result<Self> {
        let meta_len = 2 * mem::size_of::<u32>();
        if meta_slice.len() != meta_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF meta slot has the wrong length",
            )
            .into());
        }
        let meta = meta_slice.read_bytes()?;
        let mut reader = meta.as_slice();
        let num_docs = u32::deserialize(&mut reader)? as usize;
        let num_centroids = u32::deserialize(&mut reader)? as usize;

        let cluster_offsets = offsets_slice.read_bytes()?;
        let expected_offsets = (num_centroids + 1)
            .checked_mul(mem::size_of::<u64>())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "cluster offset length overflow")
            })?;
        if cluster_offsets.len() != expected_offsets {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF cluster offset byte length mismatch",
            )
            .into());
        }

        // Bounds slot — one kind byte, then the stride-derived payload.
        let (bound_kind, bounds) = {
            let bytes = bounds_slice.read_bytes()?;
            let Some((&kind_code, payload)) = bytes.as_slice().split_first() else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF bounds slot is missing its kind byte",
                )
                .into());
            };
            let kind = BoundKind::from_code(kind_code)?;
            let expected = num_centroids
                .checked_mul(kind.stride(options.dim()))
                .and_then(|values| values.checked_mul(mem::size_of::<f32>()))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "bounds byte length overflow")
                })?;
            if payload.len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF bounds byte length mismatch",
                )
                .into());
            }
            let mut reader = payload;
            let values: Vec<f32> = (0..num_centroids * kind.stride(options.dim()))
                .map(|_| f32::deserialize(&mut reader))
                .collect::<io::Result<_>>()?;
            // A negative bound is corrupt, never produced: the fold is a
            // max of norms seeded at 0.0. NaN / +inf are NOT rejected —
            // they fail open arithmetically at the margin comparisons.
            if values.iter().any(|&value| value < 0.0) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF bounds slot holds a negative bound",
                )
                .into());
            }
            (kind, values)
        };
        let mut index = SegmentClusters {
            num_centroids,
            num_docs,
            cluster_offsets,
            bound_kind,
            bounds,
            non_empty: vec![0u64; num_centroids.div_ceil(64)],
            num_non_empty: 0,
        };
        for cluster in 0..num_centroids {
            if index.cluster_offset(cluster + 1) > index.cluster_offset(cluster) {
                index.non_empty[cluster / 64] |= 1u64 << (cluster % 64);
                index.num_non_empty += 1;
            }
        }
        // Every distinct doc owns at least its primary row, so a doc count
        // above the row total means a corrupt file.
        if index.num_docs > index.num_rows() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF doc count exceeds the posting-row total",
            )
            .into());
        }
        Ok(index)
    }

    pub fn num_clusters(&self) -> usize {
        self.num_centroids
    }

    /// Distinct docs with a vector; replication inflates the row total,
    /// [`Self::num_rows`].
    pub(crate) fn num_docs(&self) -> usize {
        self.num_docs
    }

    /// Total posting rows across all clusters — memberships, counting a
    /// replicated doc once per cell it lives in.
    pub fn num_rows(&self) -> usize {
        self.cluster_offset(self.num_centroids) as usize
    }

    fn cluster_offset(&self, cluster: usize) -> u64 {
        let start = cluster * mem::size_of::<u64>();
        let end = start + mem::size_of::<u64>();
        u64::from_le_bytes(self.cluster_offsets[start..end].try_into().unwrap())
    }

    /// The contiguous row range of `cluster` within the `.vec` rows.
    #[inline]
    pub fn cluster_range(&self, cluster: usize) -> Range<usize> {
        debug_assert!(cluster < self.num_centroids, "cluster out of bounds");
        self.cluster_offset(cluster) as usize..self.cluster_offset(cluster + 1) as usize
    }

    /// Presence: `true` iff `cluster` has rows in this segment. One bit of
    /// pinned state; the probe loop's cheapest gate.
    #[inline]
    pub fn has_cluster(&self, cluster: usize) -> bool {
        debug_assert!(cluster < self.num_centroids, "cluster out of bounds");
        (self.non_empty[cluster / 64] >> (cluster % 64)) & 1 == 1
    }

    /// The non-empty row range of `cluster`, or `None` when the cluster has
    /// no rows here — the driver-facing form, so the offsets encoding can
    /// change underneath (a sparse layout is the flagged follow-up).
    #[inline]
    pub fn non_empty_cluster_range(&self, cluster: usize) -> Option<Range<usize>> {
        self.has_cluster(cluster)
            .then(|| self.cluster_range(cluster))
    }

    /// Number of clusters with at least one row in this segment.
    pub fn num_non_empty_clusters(&self) -> usize {
        self.num_non_empty
    }

    /// The stored centroid bounds of this segment's clusters.
    ///
    /// Returns (`BoundStore`): a view over the pinned slot `[3]` payload —
    /// segment-level kind plus per-cluster values; `f32::INFINITY` =
    /// SATURATED (always probes).
    #[inline]
    pub fn bounds(&self) -> BoundStore<'_> {
        BoundStore::new(self.bound_kind, &self.bounds)
    }

    /// Per-cluster posting-list sizes, in cluster order — memberships, like
    /// [`Self::num_rows`].
    pub(crate) fn cluster_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.num_centroids).map(|cluster| {
            (self.cluster_offset(cluster + 1) - self.cluster_offset(cluster)) as usize
        })
    }
}

// ---- computing assignments ------------------------------------------

const ASSIGN_ROW_TILE: usize = 256;
const ASSIGN_CENTROID_TILE: usize = 1024;

pub(crate) struct BatchAssigner {
    centroids: Vec<f32>,
    centroid_norms: Vec<f32>,
    dim: usize,
    metric: Metric,
    scores: Vec<f32>,
}

impl BatchAssigner {
    pub(crate) fn new(centroids: Vec<f32>, options: &VectorOptions) -> Self {
        let dim = options.dim();
        assert!(dim > 0 && !centroids.is_empty() && centroids.len() % dim == 0);
        let centroid_norms = if options.metric() == Metric::L2 {
            superkmeans::squared_norms(&centroids, centroids.len() / dim, dim)
        } else {
            Vec::new()
        };
        Self {
            centroids,
            centroid_norms,
            dim,
            metric: options.metric(),
            scores: Vec::new(),
        }
    }

    pub(crate) fn assign_cells(
        &mut self,
        values: &[f32],
        cells_per_vector: usize,
    ) -> Vec<Vec<usize>> {
        assert_eq!(values.len() % self.dim, 0);
        let num_centroids = self.centroids.len() / self.dim;
        assert!((1..=num_centroids).contains(&cells_per_vector));
        let num_rows = values.len() / self.dim;
        let row_norms = if self.metric == Metric::L2 {
            superkmeans::squared_norms(values, num_rows, self.dim)
        } else {
            Vec::new()
        };
        self.scores.resize(
            num_rows.min(ASSIGN_ROW_TILE) * num_centroids.min(ASSIGN_CENTROID_TILE),
            0.0,
        );
        let mut assignments = Vec::with_capacity(num_rows);
        for (row_tile, rows) in values.chunks(ASSIGN_ROW_TILE * self.dim).enumerate() {
            let row_count = rows.len() / self.dim;
            let mut nearest: Vec<_> = (0..row_count)
                .map(|_| TopNComputer::new_with_comparator(cells_per_vector, NaturalComparator))
                .collect();
            for (tile, centroids) in self
                .centroids
                .chunks(ASSIGN_CENTROID_TILE * self.dim)
                .enumerate()
            {
                let centroid_count = centroids.len() / self.dim;
                sgemm_row_major_b_transposed(
                    row_count,
                    self.dim,
                    centroid_count,
                    rows,
                    centroids,
                    &mut self.scores[..row_count * centroid_count],
                );
                for (row, top) in nearest.iter_mut().enumerate() {
                    for col in 0..centroid_count {
                        let centroid = tile * ASSIGN_CENTROID_TILE + col;
                        let dot = self.scores[row * centroid_count + col];
                        let score = match self.metric {
                            Metric::L2 => {
                                let norms = row_norms[row_tile * ASSIGN_ROW_TILE + row]
                                    + self.centroid_norms[centroid];
                                let distance = norms - 2.0 * dot;
                                let error = (norms + 2.0 * dot.abs())
                                    * (2.0 * self.dim as f32 * f32::EPSILON);
                                if !distance.is_finite() || distance <= error {
                                    Metric::L2
                                        .similarity(
                                            &rows[row * self.dim..(row + 1) * self.dim],
                                            &centroids[col * self.dim..(col + 1) * self.dim],
                                        )
                                        .score()
                                } else {
                                    -distance
                                }
                            }
                            // Cosine rows and centroids are normalized before assignment.
                            Metric::Cosine | Metric::Dot => dot,
                        };
                        top.push(Similarity::new(score), centroid);
                    }
                }
            }
            assignments.extend(nearest.into_iter().map(|top| {
                top.into_sorted_vec()
                    .into_iter()
                    .map(|hit| hit.doc)
                    .collect()
            }));
        }
        assignments
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::*;

    /// Pins the Dot selection semantics: cells follow RAW dot — the
    /// query-time router's ranking — not angular order. Centroid norms are
    /// deliberately unequal so the two orderings disagree.
    #[test]
    fn dot_selector_uses_raw_dot_not_angular() {
        let centroids: Vec<f32> = vec![
            10.0, 0.0, // long, off-direction: dot 10, cosine 0.45
            0.0, 1.0, // short, near-direction: dot 2, cosine 0.89
            7.0, 7.0, // long, near-direction: dot 21, cosine 0.95
        ];
        let mut assigner = BatchAssigner::new(centroids, &VectorOptions::new(2, Metric::Dot));
        let picked = assigner.assign_cells(&[1.0_f32, 2.0], 3).remove(0);
        // Raw-dot order: [7,7] (21), then [10,0] (10), then [0,1] (2).
        // Angular order would put [0,1] ahead of [10,0].
        assert_eq!(picked, vec![2, 0, 1], "must rank by raw dot");
    }

    #[test]
    fn assign_cells_is_nearest_first_and_order_preserving() {
        let mut assigner = BatchAssigner::new(
            vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0],
            &VectorOptions::new(2, Metric::L2),
        );
        let values: Vec<f32> = vec![
            1.0, 0.0, // nearest 0, then 1
            9.0, 1.0, // nearest 1, then 0
            0.5, 9.0, // nearest 2, then 0
        ];
        let cells = assigner.assign_cells(&values, 2);
        assert_eq!(cells, vec![vec![0, 1], vec![1, 0], vec![2, 0]]);
    }

    #[test]
    fn batches_match_scalar_assignment_across_tiles_and_metrics() {
        let dim = 9;
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let mut centroids: Vec<f32> = (0..(ASSIGN_CENTROID_TILE + 7) * dim)
                .map(|_| rng.random_range(-20..=20) as f32)
                .collect();
            let mut values: Vec<f32> = (0..(ASSIGN_ROW_TILE + 3) * dim)
                .map(|_| rng.random_range(-20..=20) as f32)
                .collect();
            if metric == Metric::Cosine {
                for row in centroids
                    .chunks_exact_mut(dim)
                    .chain(values.chunks_exact_mut(dim))
                {
                    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                    for x in row {
                        *x /= norm;
                    }
                }
            }
            let mut assigner =
                BatchAssigner::new(centroids.clone(), &VectorOptions::new(dim, metric));
            for replicas in [1, 3] {
                let assigned = assigner.assign_cells(&values, replicas);
                for (row, actual) in values.chunks_exact(dim).zip(assigned) {
                    let mut expected: Vec<_> = centroids
                        .chunks_exact(dim)
                        .enumerate()
                        .map(|(id, centroid)| (metric.similarity(row, centroid), id))
                        .collect();
                    expected.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                    assert_eq!(
                        actual,
                        expected[..replicas].iter().map(|c| c.1).collect::<Vec<_>>(),
                        "{metric:?}"
                    );
                }
            }
            assert!(assigner.assign_cells(&[], 1).is_empty());
            assert_eq!(assigner.assign_cells(&values[..dim], 1).len(), 1);
            assert!(assigner.scores.capacity() <= ASSIGN_ROW_TILE * ASSIGN_CENTROID_TILE);
        }
    }

    #[test]
    fn ties_and_positive_dot_scores_across_centroid_tiles() {
        let mut centroids = vec![0.0; (ASSIGN_CENTROID_TILE + 2) * 2];
        centroids[..2].copy_from_slice(&[10.0, 0.0]);
        centroids[ASSIGN_CENTROID_TILE * 2..].copy_from_slice(&[1.0, 0.0, 10.0, 0.0]);
        let mut assigner = BatchAssigner::new(centroids, &VectorOptions::new(2, Metric::Dot));
        assert_eq!(
            assigner.assign_cells(&[1.0, 0.0], 2),
            vec![vec![0, ASSIGN_CENTROID_TILE + 1]]
        );
        assert_eq!(assigner.assign_cells(&[0.0, 0.0], 3), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn l2_assignment_handles_cancellation_and_overflow() {
        let mut assigner = BatchAssigner::new(
            vec![1e6, 1e6, 1e6 + 1.0, 1e6],
            &VectorOptions::new(2, Metric::L2),
        );
        assert_eq!(assigner.assign_cells(&[1e6 + 1.0, 1e6], 1), vec![vec![1]]);
        assert_eq!(assigner.assign_cells(&[3e38, 3e38], 2), vec![vec![0, 1]]);
    }
}
