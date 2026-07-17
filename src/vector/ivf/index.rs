//! The `.centroids` file and its reader, [`IvfIndex`] — the per-field IVF
//! routing index. This module owns the wire format end to end: the
//! serializers the merge calls and the [`IvfIndex::open`] that parses them
//! back sit side by side.
//!
//! Written per field, only for IVF segments (⟺ the field's `.vec` `IdMap` is
//! `Explicit`). A [`CompositeFile`](crate::directory::CompositeFile) with
//! three slots per field:
//!
//! ```text
//! [0] num_centroids (u32) + num_docs (u32) + centroid_bytes (N · stride)
//! [1] cluster_offsets (u64[N+1], prefix sum)
//! [2] RNG over the centroids (see `Graph::serialize` for the layout;
//!     absent for degenerate centroid counts — routing then falls back to a
//!     linear scan of the centroids)
//! ```
//!
//! One dense `centroid_id = 0..N` indexes all three: `cluster_offsets[c]` is
//! the first row of cluster `c` in the parallel `.vec` rows/`IdMap`, and graph
//! node `c` is centroid `c` (its vector is row `c` of slot `[0]`, which is why
//! the graph slot stores no vectors of its own).

use std::io::{self, Write};
use std::mem;
use std::ops::Range;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;

use common::{BinarySerializable, HasLen, OwnedBytes};

use super::graph::{
    Candidate, NeighborhoodGraphConfig, NodeId, RelativeNeighborhoodGraph, Workspace,
};
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorDType, VectorOptions};
use crate::vector::index_reader::MAX_FETCH_SPAN_BYTES;
use crate::vector::FileSliceArena;

/// The IVF routing index over one field's clusters: says which clusters —
/// contiguous row ranges of the `.vec` rows — a query should probe.
///
/// Pinned state is small and touched by every query: the cluster offsets and
/// the RNG adjacency (edges only, `num_centroids × max_edges × 4` bytes). The
/// centroid vectors stay behind a [`FileSliceArena`] and are fetched one node
/// at a time as routing visits them. Everything row-scale (the rows and
/// id-map) lives on [`VectorIndexReader`](crate::vector::VectorIndexReader).
pub struct IvfIndex {
    num_centroids: usize,
    /// Distinct documents with a vector in this field. Rows including
    /// replicas are [`Self::num_rows`].
    num_docs: usize,
    /// The centroid rows (slot `[0]` past the two count words).
    centroids_slice: FileSlice,
    /// Slot `[1]`: the `u64[N+1]` prefix sum, pinned.
    cluster_offsets: OwnedBytes,
    dim: usize,
    metric: Metric,
    /// The persisted RNG over the centroids (slot `[2]`). `None` for
    /// degenerate centroid counts, where routing falls back to a linear scan.
    graph: Option<RelativeNeighborhoodGraph<FileSliceArena<f32>>>,
    /// Test-only: route `limit >= num_centroids` rankings through the beam
    /// anyway — the pre-guard behavior, used as the equivalence baseline
    /// for the sequential slab scan.
    #[cfg(test)]
    force_beam: bool,
    /// Test-only shrink of [`MAX_FETCH_SPAN_BYTES`] for the sequential
    /// slab scan, so window splitting is exercisable on small fixtures.
    #[cfg(test)]
    forced_span_cap: Option<usize>,
    /// Test-only: the largest single slab read served by
    /// [`Self::rank_all_clusters`], in bytes — asserts span-cap compliance,
    /// and (when zero) that ranking never took the sequential-scan path.
    #[cfg(test)]
    max_read_bytes: AtomicUsize,
}

impl IvfIndex {
    /// Write slot `[0]` of the `.centroids` composite for a field. `num_docs`
    /// is the number of distinct docs assigned — NOT the posting-row total,
    /// which replication can multiply.
    pub(crate) fn serialize_centroids<W: Write + ?Sized>(
        num_centroids: usize,
        num_docs: usize,
        centroid_bytes: &[u8],
        options: &VectorOptions,
        out: &mut W,
    ) -> io::Result<()> {
        let expected = num_centroids
            .checked_mul(options.bytes_per_vector())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
            })?;
        if centroid_bytes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid IVF centroid byte length",
            ));
        }
        u32::try_from(num_centroids)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "centroid count exceeds u32"))?
            .serialize(out)?;
        u32::try_from(num_docs)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "doc count exceeds u32"))?
            .serialize(out)?;
        out.write_all(centroid_bytes)
    }

    /// Write slot `[1]` of the `.centroids` composite for a field.
    pub(crate) fn serialize_offsets<W: Write + ?Sized>(
        cluster_offsets: &[u64],
        out: &mut W,
    ) -> io::Result<()> {
        for offset in cluster_offsets {
            offset.serialize(out)?;
        }
        Ok(())
    }

    /// Parse a field's `.centroids` slots. Only the count words, the offsets,
    /// and the graph adjacency are materialized; the centroid rows stay
    /// behind a [`FileSlice`] for lazy per-node reads.
    pub(crate) fn open(
        options: &VectorOptions,
        centroids_slice: FileSlice,
        offsets_slice: FileSlice,
        graph_slice: Option<FileSlice>,
    ) -> crate::Result<Self> {
        let count_words = 2 * mem::size_of::<u32>();
        if centroids_slice.len() < count_words {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF centroids slot is smaller than its count words",
            )
            .into());
        }
        let header = centroids_slice.slice_to(count_words).read_bytes()?;
        let mut reader = header.as_slice();
        let num_centroids = u32::deserialize(&mut reader)? as usize;
        let num_docs = u32::deserialize(&mut reader)? as usize;
        let centroid_len = num_centroids
            .checked_mul(options.bytes_per_vector())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
            })?;
        let centroids_slice = centroids_slice.slice_from(count_words);
        if centroids_slice.len() != centroid_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF centroid byte length mismatch",
            )
            .into());
        }

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

        let graph = match graph_slice {
            Some(slice) => {
                let vectors = match options.dtype() {
                    VectorDType::F32 => FileSliceArena::<f32>::new(centroids_slice.clone()),
                };
                // Adjacency length is validated against the arena's node
                // count inside `Graph::open`.
                let adjacency = slice.read_bytes()?;
                Some(RelativeNeighborhoodGraph::open(
                    &adjacency,
                    vectors,
                    options.dim(),
                    options.metric(),
                    NeighborhoodGraphConfig::default(),
                )?)
            }
            None => None,
        };

        let index = IvfIndex {
            num_centroids,
            num_docs,
            centroids_slice,
            cluster_offsets,
            dim: options.dim(),
            metric: options.metric(),
            graph,
            #[cfg(test)]
            force_beam: false,
            #[cfg(test)]
            forced_span_cap: None,
            #[cfg(test)]
            max_read_bytes: AtomicUsize::new(0),
        };
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

    /// Per-cluster posting-list sizes, in cluster order — memberships, like
    /// [`Self::num_rows`].
    pub(crate) fn cluster_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.num_centroids).map(|cluster| {
            (self.cluster_offset(cluster + 1) - self.cluster_offset(cluster)) as usize
        })
    }

    /// The centroid rows, materialized in one read — for introspection and
    /// tests only. Routing fetches per-node ranges through the lazy arena.
    pub fn centroid_bytes(&self) -> crate::Result<OwnedBytes> {
        Ok(self.centroids_slice.read_bytes()?)
    }

    /// Clusters to probe for `query`, best routing score first, as
    /// `(score, cluster)` pairs, plus the number of centroids scored
    /// (surfaced as `ProbeStats::centroids_ranked`).
    ///
    /// With a persisted RNG and `limit` below the centroid count this is a
    /// beam search ([`RelativeNeighborhoodGraph::search`]); otherwise —
    /// no graph, or a `limit` that asks for every cluster — each centroid
    /// is scored exactly by [`Self::rank_all_clusters`]. Both paths score
    /// with the same byte kernels and order ties identically, so their
    /// rankings agree.
    pub(crate) fn rank_clusters(&self, query: &[f32], limit: usize) -> (Vec<(f32, u32)>, usize) {
        match &self.graph {
            Some(graph) if self.route_via_beam(limit) => {
                let mut ws = Workspace::new();
                // TODO: Replace with proper seed generation
                let seeds: Vec<NodeId> = {
                    (0..graph.len())
                        .step_by((graph.len() / 8).max(1))
                        .take(8)
                        .map(|node| node as NodeId)
                        .collect()
                };
                let candidates = graph.search(&mut ws, query, &seeds, limit);
                let ranked = candidates
                    .into_iter()
                    .map(|candidate| (candidate.sim.score(), candidate.node))
                    .collect();
                (ranked, ws.num_visited())
            }
            _ => self.rank_all_clusters(query, limit),
        }
    }

    /// `true` when the RNG beam should route this ranking: only while
    /// `limit` leaves the beam clusters to skip.
    ///
    /// At `limit >= num_centroids` the beam's early-stop can never arm (the
    /// beam is `max(ef, limit)` wide, so its result set never fills), and
    /// it degenerates into a full traversal of the graph that fetches every
    /// centroid with a scattered per-node read. Ranking every cluster needs
    /// no navigation at all: the sequential slab scan is exhaustive by
    /// construction — no seed-connectivity caveat — and strictly cheaper,
    /// making the exhaustive-semantics contract for
    /// `max_probes >= num_centroids` unconditional.
    fn route_via_beam(&self, limit: usize) -> bool {
        #[cfg(test)]
        if self.force_beam {
            return true;
        }
        limit < self.num_centroids
    }

    /// Rank every centroid: a sequential scan of the centroid slab in
    /// span-capped windows — one ranged read per [`Self::fetch_span_cap`]
    /// bytes, in slab order — scored with the same byte kernels the beam's
    /// arena uses and sorted with the beam's exact ordering (descending
    /// similarity, ties by ascending centroid id). Every centroid is
    /// scored, empty clusters included, and the scored count returned is
    /// `num_centroids`.
    ///
    /// # Panics
    ///
    /// Like [`FileSliceArena`], ranking has no error channel; a failed
    /// slab read panics.
    fn rank_all_clusters(&self, query: &[f32], limit: usize) -> (Vec<(f32, u32)>, usize) {
        let stride = self.dim * mem::size_of::<f32>();
        // A row is scored from contiguous bytes, so a single row wider
        // than the cap is still read whole.
        let rows_per_read = (self.fetch_span_cap() / stride).max(1);
        let mut candidates: Vec<Candidate> = Vec::with_capacity(self.num_centroids);
        let mut window_start = 0;
        while window_start < self.num_centroids {
            let window_end = self.num_centroids.min(window_start + rows_per_read);
            let bytes = self
                .centroids_slice
                .slice(window_start * stride..window_end * stride)
                .read_bytes()
                .expect("failed to read centroid slab window");
            #[cfg(test)]
            self.max_read_bytes
                .fetch_max(bytes.len(), std::sync::atomic::Ordering::Relaxed);
            for node in window_start..window_end {
                let offset = (node - window_start) * stride;
                let sim = self
                    .metric
                    .similarity_bytes(query, &bytes[offset..offset + stride]);
                candidates.push(Candidate {
                    sim,
                    node: node as NodeId,
                });
            }
            window_start = window_end;
        }
        candidates.sort_unstable_by(|a, b| b.sim.cmp(&a.sim).then_with(|| a.node.cmp(&b.node)));
        candidates.truncate(limit);
        let ranked = candidates
            .into_iter()
            .map(|candidate| (candidate.sim.score(), candidate.node))
            .collect();
        (ranked, self.num_centroids)
    }

    /// The span ceiling one slab ranged read may cover —
    /// [`MAX_FETCH_SPAN_BYTES`], shrinkable in tests so window splitting
    /// is exercisable on small fixtures.
    fn fetch_span_cap(&self) -> usize {
        #[cfg(test)]
        if let Some(cap) = self.forced_span_cap {
            return cap;
        }
        MAX_FETCH_SPAN_BYTES
    }

    /// Test-only: the largest single slab read served by
    /// [`Self::rank_all_clusters`], in bytes — zero when ranking never took
    /// the sequential-scan path.
    #[cfg(test)]
    fn max_read_bytes(&self) -> usize {
        self.max_read_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Executor;

    const DIM: usize = 2;
    const NUM_CENTROIDS: usize = 24;

    /// A persisted-RNG `IvfIndex` over `num_centroids` centroids laid out
    /// on a line (centroid `c` at `[c, 0]`), one posting row per cluster.
    /// Collinear points make consecutive-neighbor RNG edges unoccludable
    /// (no third point is strictly closer to both endpoints), so the built
    /// graph is connected and a full beam traversal provably visits every
    /// node — which the beam-vs-scan equivalence tests rely on.
    fn ivf_fixture(num_centroids: usize) -> IvfIndex {
        let options = VectorOptions::new(DIM, Metric::L2).with_dtype(VectorDType::F32);
        let mut matrix: Vec<f32> = Vec::with_capacity(num_centroids * DIM);
        for c in 0..num_centroids {
            matrix.push(c as f32);
            matrix.push(0.0);
        }
        let centroid_bytes: Vec<u8> = matrix.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut centroids_slot = Vec::new();
        IvfIndex::serialize_centroids(
            num_centroids,
            num_centroids,
            &centroid_bytes,
            &options,
            &mut centroids_slot,
        )
        .unwrap();

        let offsets: Vec<u64> = (0..=num_centroids as u64).collect();
        let mut offsets_slot = Vec::new();
        IvfIndex::serialize_offsets(&offsets, &mut offsets_slot).unwrap();

        let mut rng = RelativeNeighborhoodGraph::new(
            matrix.as_slice(),
            DIM,
            options.metric(),
            NeighborhoodGraphConfig::default(),
        );
        rng.build(&Executor::single_thread());
        let mut graph_slot = Vec::new();
        rng.serialize(&mut graph_slot).unwrap();

        IvfIndex::open(
            &options,
            FileSlice::from(centroids_slot),
            FileSlice::from(offsets_slot),
            Some(FileSlice::from(graph_slot)),
        )
        .unwrap()
    }

    /// Off-lattice x so every centroid's L2 similarity is distinct.
    const QUERY: [f32; 2] = [0.3, 0.7];

    /// `limit = C+1` — the exhaustive-cap shape (probe ceiling clamped to
    /// the cluster count, plus one) — must rank via the sequential slab
    /// scan despite the persisted RNG, and produce output identical (ids
    /// and scores) to the beam's on the same fixture.
    #[test]
    fn exhaustive_guard_takes_linear_path() {
        let index = ivf_fixture(NUM_CENTROIDS);
        let (ranked, scored) = index.rank_clusters(&QUERY, NUM_CENTROIDS + 1);
        // The slab was scanned — the beam's per-node arena reads never
        // touch the instrumented window path...
        assert!(index.max_read_bytes() > 0, "guard must take the slab scan");
        // ...and exhaustively: every cluster ranked, every centroid scored.
        assert_eq!(scored, NUM_CENTROIDS);
        assert_eq!(ranked.len(), NUM_CENTROIDS);
        let mut ids: Vec<u32> = ranked.iter().map(|&(_, cluster)| cluster).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..NUM_CENTROIDS as u32).collect::<Vec<_>>());
        for pair in ranked.windows(2) {
            assert!(pair[0].0 >= pair[1].0, "ranking must be descending");
        }

        // Identical to the beam with the guard disabled.
        let mut forced = ivf_fixture(NUM_CENTROIDS);
        forced.force_beam = true;
        let (beam_ranked, beam_scored) = forced.rank_clusters(&QUERY, NUM_CENTROIDS + 1);
        assert_eq!(
            forced.max_read_bytes(),
            0,
            "force_beam must never touch the slab scan"
        );
        assert_eq!(
            beam_scored, NUM_CENTROIDS,
            "connected fixture: the beam visits every node"
        );
        assert_eq!(ranked, beam_ranked, "scan and beam must rank identically");
    }

    /// The guard is `limit >= num_centroids`, pinned here: `limit == C-1`
    /// still routes via the beam, while `limit == C` — where the beam's
    /// early-stop needs `results.len() >= ef >= C` and so already implies
    /// a full traversal — takes the scan.
    #[test]
    fn guard_boundary_is_num_centroids() {
        let index = ivf_fixture(NUM_CENTROIDS);
        let (ranked, _) = index.rank_clusters(&QUERY, NUM_CENTROIDS - 1);
        assert_eq!(
            index.max_read_bytes(),
            0,
            "limit < num_centroids must stay on the beam"
        );
        assert_eq!(ranked.len(), NUM_CENTROIDS - 1);

        let (ranked, scored) = index.rank_clusters(&QUERY, NUM_CENTROIDS);
        assert!(
            index.max_read_bytes() > 0,
            "limit == num_centroids must take the slab scan"
        );
        assert_eq!(ranked.len(), NUM_CENTROIDS);
        assert_eq!(scored, NUM_CENTROIDS);
    }

    /// A slab wider than the (test-shrunk) span cap splits into capped
    /// windows; capping changes read shapes only, never the ranking.
    #[test]
    fn linear_scan_respects_span_cap() {
        let stride = DIM * mem::size_of::<f32>();
        let uncapped = ivf_fixture(NUM_CENTROIDS);
        let (baseline, _) = uncapped.rank_clusters(&QUERY, NUM_CENTROIDS + 1);
        assert_eq!(
            uncapped.max_read_bytes(),
            NUM_CENTROIDS * stride,
            "an uncapped scan reads the slab in one span"
        );

        let mut capped = ivf_fixture(NUM_CENTROIDS);
        capped.forced_span_cap = Some(5 * stride);
        let (ranked, scored) = capped.rank_clusters(&QUERY, NUM_CENTROIDS + 1);
        assert!(
            capped.max_read_bytes() <= 5 * stride,
            "no read may exceed the cap"
        );
        assert!(capped.max_read_bytes() > 0);
        assert_eq!(scored, NUM_CENTROIDS);
        assert_eq!(ranked, baseline);

        // A cap below one row still reads whole rows — a row is scored
        // from contiguous bytes, so the cap bounds multi-row spans only.
        let mut tiny = ivf_fixture(NUM_CENTROIDS);
        tiny.forced_span_cap = Some(1);
        let (ranked, _) = tiny.rank_clusters(&QUERY, NUM_CENTROIDS + 1);
        assert_eq!(tiny.max_read_bytes(), stride);
        assert_eq!(ranked, baseline);
    }
}
