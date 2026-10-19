//! Multi-level (stacked) IVF.
//!
//! L0 members are the items being routed. L1 is an IVF over L0's
//! centroids, L2 over L1's, and so on. Each list is a contiguous row
//! range in [`StackedIvf::offsets`].
//!
//! [`StackedIvf::build`] creates one level. [`StackedIvf::add_level`]
//! hangs a parent on this level and reorders this level's centroids and
//! offset ranges so parent member `i` is list `i`. Member rows stay put;
//! a later parent only shuffles this level's metadata. Each level's `nlist`
//! is `ceil(√n)`; [`StackedIvfBuilder`] hangs a parent while the current
//! top has at least [`MIN_NLIST_FOR_PARENT`] lists.
//!
//! Search works on any [`VectorArena`]. Build and payload serialization require
//! owned [`InMemoryStore`]s. [`StackedIvf::open`] is search-only.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{self, Write};
use std::ops::Deref;
use std::{fmt, mem};

use common::{BinarySerializable, HasLen};
use itertools::Itertools;
use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, SuperKMeansConfig};

use crate::directory::FileSlice;
use crate::schema::Metric;
use crate::vector::distance::Similarity;
use crate::vector::router::arena::{FileSliceArena, VectorArena};
use crate::vector::router::candidate::Candidate;
use crate::vector::router::stacked::aps;
use crate::vector::router::stacked::aps::RecallEstimator;
use crate::vector::VectorElement;

/// Row index into a level's centroid or member arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClusterId(pub u32);

impl From<ClusterId> for u32 {
    #[inline]
    fn from(id: ClusterId) -> u32 {
        id.0
    }
}

impl From<ClusterId> for usize {
    #[inline]
    fn from(id: ClusterId) -> usize {
        id.0 as usize
    }
}

impl From<u32> for ClusterId {
    #[inline]
    fn from(i: u32) -> Self {
        ClusterId(i)
    }
}

impl From<usize> for ClusterId {
    #[inline]
    fn from(i: usize) -> Self {
        ClusterId(i as u32)
    }
}

/// Default `nprobe_fraction` for parent levels (L1, L2, …).
pub(crate) const PARENT_NPROBE_FRACTION: f32 = 0.25;

/// Floor of [`member_budget`], as a fraction of the bottom router level's members.
pub(crate) const MEMBER_BUDGET_FLOOR_FRACTION: f64 = 0.10;

/// Members the bottom router level scores before it stops opening lists: `multiplier × k`,
/// clamped to [`MEMBER_BUDGET_FLOOR_FRACTION`] and [`PARENT_NPROBE_FRACTION`] of its `members`
/// and never below `k`. Lists are scanned whole, so the scan ends at the first list boundary at
/// or past the budget.
pub(crate) fn member_budget(members: usize, k: usize, multiplier: f32) -> usize {
    let fraction = |f: f64| (members as f64 * f).ceil() as usize;
    let wanted = (k as f64 * f64::from(multiplier))
        .ceil()
        .min(usize::MAX as f64) as usize;
    wanted
        .clamp(
            fraction(MEMBER_BUDGET_FLOOR_FRACTION),
            fraction(f64::from(PARENT_NPROBE_FRACTION)),
        )
        .max(k)
}

/// Recall target used when this level searches its parent.
pub(crate) const PARENT_RECALL_TARGET: f32 = 0.99;

/// [`StackedIvfBuilder`] hangs a parent only while the current top has at
/// least this many lists. Below that, the top is scored in full and pruned
/// to [`PARENT_NPROBE_FRACTION`].
pub(crate) const MIN_NLIST_FOR_PARENT: usize = 256;

/// Search and clustering knobs for this level. Not persisted.
#[derive(Clone, Debug)]
pub struct StackedIvfConfig {
    /// Initial candidate fraction `f_M`: `|S| = nprobe_fraction * nlist`.
    /// At the top level (no parent), lists are pruned to
    /// [`PARENT_NPROBE_FRACTION`] after scoring every centroid.
    pub nprobe_fraction: f32,
    /// Recall target this level passes to its parent when APS is on.
    /// `1.0` makes the parent scan its full candidate set (exact routing
    /// when the parent is the top level).
    pub parent_recall_target: f32,
}

impl Default for StackedIvfConfig {
    fn default() -> Self {
        Self {
            nprobe_fraction: 0.02,
            parent_recall_target: PARENT_RECALL_TARGET,
        }
    }
}

impl StackedIvfConfig {
    fn for_parent(&self) -> Self {
        Self {
            nprobe_fraction: PARENT_NPROBE_FRACTION,
            parent_recall_target: self.parent_recall_target,
        }
    }

    /// k-means target for a level over `n` members: `ceil(√n)`.
    fn nlist_for(n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((n as f64).sqrt().ceil() as usize).max(1).min(n)
    }
}

/// No centroids at the level [`StackedIvf::add_level`] would extend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddLevelError {
    Empty,
}

impl fmt::Display for AddLevelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddLevelError::Empty => {
                f.write_str("no centroids — build this level before calling add_level")
            }
        }
    }
}

impl std::error::Error for AddLevelError {}

/// Owned row-major `f32` matrix.
#[derive(Clone, Debug)]
pub struct InMemoryStore {
    data: Vec<f32>,
    dim: usize,
}

impl InMemoryStore {
    /// Row-major `n × dim` matrix. Panics if `data.len()` is not a multiple of `dim`.
    pub fn new(data: Vec<f32>, dim: usize) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(data.len() % dim, 0, "data length must be a multiple of dim");
        InMemoryStore { data, dim }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }
}

impl Deref for InMemoryStore {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.data
    }
}

/// Cluster `n` vectors into about `nlist` balanced lists.
pub trait LevelClusterer {
    type Elem: VectorElement;

    /// Returns `(centroids row-major nlist'×dim, assignment[i] → list id)`.
    fn cluster(
        &self,
        data: &[Self::Elem],
        n: usize,
        dim: usize,
        nlist: usize,
    ) -> (Vec<Self::Elem>, Vec<u32>);
}

/// [`HierarchicalSuperKMeans`] with `max_leaf_size ≈ n / nlist`. Meso and fine
/// iterations keep the library defaults. [`Self::for_metric`] clusters Cosine
/// and Dot by angle, the same rule as the segment's IVF.
#[derive(Clone, Copy, Debug, Default)]
pub struct SuperKMeansLevelClusterer {
    angular: bool,
}

impl SuperKMeansLevelClusterer {
    /// Angular clustering for Cosine and Dot; Euclidean for L2.
    pub fn for_metric(metric: Metric) -> Self {
        Self {
            angular: matches!(metric, Metric::Cosine | Metric::Dot),
        }
    }
}

impl LevelClusterer for SuperKMeansLevelClusterer {
    type Elem = f32;

    fn cluster(&self, data: &[f32], n: usize, dim: usize, nlist: usize) -> (Vec<f32>, Vec<u32>) {
        assert!(n > 0 && dim > 0 && nlist > 0);
        assert_eq!(data.len(), n * dim);
        let nlist = nlist.min(n).max(1);
        let max_leaf_size = n.div_ceil(nlist).max(1);

        let base = SuperKMeansConfig {
            data_already_rotated: true,
            angular: self.angular,
            ..Default::default()
        };
        let cfg = HierarchicalSuperKMeansConfig {
            base,
            max_leaf_size,
            ..Default::default()
        };
        let mut kmeans = HierarchicalSuperKMeans::with_config(dim, cfg);
        let centroids = kmeans.train(data, n);
        let assignments = kmeans.assign(data, &centroids, n);
        (centroids, assignments)
    }
}

/// One IVF level. `parent` is the next level up (L0's parent is L1).
///
/// List `j` is `vectors` rows `offsets[j].0..offsets[j].1`. Search is
/// available for any [`VectorArena`]; [`build`](StackedIvf::build) and
/// [`add_level`](StackedIvf::add_level) require [`InMemoryStore`].
pub struct StackedIvf<S> {
    pub config: StackedIvfConfig,
    pub parent: Option<Box<StackedIvf<S>>>,
    /// Per-list `(start, end)` into [`Self::vectors`]. Not necessarily
    /// packed in list-id order after a parent is added above this level.
    pub offsets: Vec<(u64, u64)>,
    pub centroids: S,
    pub vectors: S,
}

/// File-backed row-major `f32` matrix.
///
/// Rows are fetched per call through a [`FileSliceArena`] unless the store
/// was opened [`pinned`](Self::pinned), in which case the decoded rows are
/// held in memory. Router-level centroids are pinned at open: they are
/// `nlist × dim` (tiny next to the member rows) and APS needs the full
/// matrix to compute query-to-bisector distances.
pub(crate) struct LazyStore {
    arena: FileSliceArena<f32>,
    dim: usize,
    pinned: Option<Vec<f32>>,
}

impl LazyStore {
    /// Wraps `rows` (row-major `n × dim` little-endian `f32`s).
    pub(crate) fn new(rows: FileSlice, dim: usize) -> Self {
        LazyStore {
            arena: FileSliceArena::new(rows),
            dim,
            pinned: None,
        }
    }

    /// Wraps `rows` and decodes them into memory once, so
    /// [`CentroidMatrix`] is available and scoring skips the file.
    pub(crate) fn pinned(rows: FileSlice, dim: usize) -> io::Result<Self> {
        let bytes = rows.read_bytes()?;
        if bytes.len() % (dim * mem::size_of::<f32>()) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "pinned store bytes are not a multiple of the row stride",
            ));
        }
        let values = bytes
            .as_slice()
            .chunks_exact(mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
            .collect();
        Ok(LazyStore {
            arena: FileSliceArena::new(rows),
            dim,
            pinned: Some(values),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.arena.num_vectors(self.dim)
    }

    /// Appends row `index` to `out`.
    pub(crate) fn extend_with_row(&self, index: u32, out: &mut Vec<f32>) -> io::Result<()> {
        match &self.pinned {
            Some(rows) => out.extend_from_slice(&rows[index as usize * self.dim..][..self.dim]),
            None => {
                let bytes = self.arena.row_bytes(self.dim, index)?;
                out.extend(
                    bytes
                        .as_slice()
                        .chunks_exact(mem::size_of::<f32>())
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk"))),
                );
            }
        }
        Ok(())
    }
}

impl VectorArena for LazyStore {
    type Elem = f32;

    #[inline]
    fn num_vectors(&self, dim: usize) -> usize {
        self.arena.num_vectors(dim)
    }

    #[inline]
    fn similarity(
        &self,
        metric: Metric,
        dim: usize,
        index: u32,
        query: &[Self::Elem],
    ) -> Similarity {
        match &self.pinned {
            Some(rows) => metric.similarity(query, &rows[index as usize * dim..][..dim]),
            None => self.arena.similarity(metric, dim, index, query),
        }
    }
}

trait SerializableStore {
    fn len(&self) -> usize;
    fn serialize_rows<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()>;
}

impl SerializableStore for InMemoryStore {
    fn len(&self) -> usize {
        self.len()
    }

    fn serialize_rows<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        for &value in self.as_slice() {
            value.serialize(out)?;
        }
        Ok(())
    }
}

impl SerializableStore for LazyStore {
    fn len(&self) -> usize {
        self.len()
    }

    fn serialize_rows<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        for chunk in self.arena.slice.stream_file_chunks() {
            out.write_all(&chunk?)?;
        }
        Ok(())
    }
}

fn serialize_stacked_index<S, W>(index: &StackedIvf<S>, out: &mut W) -> io::Result<()>
where
    S: SerializableStore,
    W: Write + ?Sized,
{
    let mut num_levels = 1usize;
    let mut parent = index.parent.as_deref();
    while let Some(level) = parent {
        num_levels += 1;
        parent = level.parent.as_deref();
    }

    let mut topology = Vec::new();
    u32::try_from(num_levels)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "level count exceeds u32"))?
        .serialize(&mut topology)?;

    fn serialize_level<S>(index: &StackedIvf<S>, out: &mut Vec<u8>) -> io::Result<()>
    where S: SerializableStore {
        let nlist = index.centroids.len();
        if index.offsets.len() != nlist {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "level has {} offset ranges for {nlist} lists",
                    index.offsets.len()
                ),
            ));
        }
        u32::try_from(nlist)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "nlist exceeds u32"))?
            .serialize(out)?;
        for &(start, end) in &index.offsets {
            start.serialize(out)?;
            end.serialize(out)?;
        }
        Ok(())
    }

    serialize_level(index, &mut topology)?;
    let mut parent = index.parent.as_deref();
    while let Some(level) = parent {
        serialize_level(level, &mut topology)?;
        parent = level.parent.as_deref();
    }

    let centroids_byte_offset = (mem::size_of::<u64>() + topology.len()) as u64;
    centroids_byte_offset.serialize(out)?;
    out.write_all(&topology)?;
    index.centroids.serialize_rows(out)?;
    let mut parent = index.parent.as_deref();
    while let Some(level) = parent {
        level.centroids.serialize_rows(out)?;
        parent = level.parent.as_deref();
    }
    Ok(())
}

/// Row-major centroid access for APS. Stores without an in-memory matrix
/// return `None` and search falls back to scanning every candidate list.
pub(crate) trait CentroidMatrix {
    fn centroid_matrix(&self) -> Option<(&[f32], usize)>;
}

impl CentroidMatrix for InMemoryStore {
    fn centroid_matrix(&self) -> Option<(&[f32], usize)> {
        Some((self.as_slice(), self.dim()))
    }
}

impl CentroidMatrix for LazyStore {
    fn centroid_matrix(&self) -> Option<(&[f32], usize)> {
        self.pinned.as_deref().map(|rows| (rows, self.dim))
    }
}

/// Work done by one [`StackedIvf::search`], summed over this level and
/// every parent above it.
#[derive(Clone, Copy, Debug, Default)]
pub struct StackedSearchStats {
    /// Lists opened at any level.
    pub lists_scanned: usize,
    /// Similarity computations at any level: rows in scanned lists plus
    /// the top level's brute-force centroid ranking. A parent's members
    /// are this level's centroids, so this is the total query cost.
    pub members_scored: usize,
}

/// Counting sort into list order. `perm[old] = new_row`.
fn cluster_sort(assignments: &[u32], nlist: usize) -> (Vec<u32>, Vec<(u64, u64)>) {
    let mut prefix = vec![0u64; nlist + 1];
    for &list in assignments {
        assert!(
            (list as usize) < nlist,
            "assignment {list} out of range for {nlist} lists"
        );
        prefix[list as usize + 1] += 1;
    }
    for j in 0..nlist {
        prefix[j + 1] += prefix[j];
    }
    let mut next = prefix.clone();
    let mut perm = vec![0u32; assignments.len()];
    for (old, &list) in assignments.iter().enumerate() {
        perm[old] = next[list as usize] as u32;
        next[list as usize] += 1;
    }
    let ranges = (0..nlist).map(|j| (prefix[j], prefix[j + 1])).collect();
    (perm, ranges)
}

fn permute_rows(data: &[f32], dim: usize, perm: &[u32]) -> Vec<f32> {
    debug_assert_eq!(data.len(), perm.len() * dim);
    let mut out = vec![0.0f32; data.len()];
    for (old, &new) in perm.iter().enumerate() {
        let new = new as usize;
        out[new * dim..(new + 1) * dim].copy_from_slice(&data[old * dim..(old + 1) * dim]);
    }
    out
}

/// Builds a stacked IVF: L0 via [`StackedIvf::build`], then parents while
/// the current top has at least [`MIN_NLIST_FOR_PARENT`] lists.
///
/// Only L0's member permutation is returned — later levels shuffle
/// metadata, not L0 rows.
pub struct StackedIvfBuilder<'a, Cl> {
    data: Vec<f32>,
    n: usize,
    dim: usize,
    clusterer: &'a Cl,
    config: StackedIvfConfig,
}

impl<'a, Cl: LevelClusterer<Elem = f32>> StackedIvfBuilder<'a, Cl> {
    pub fn new(
        data: Vec<f32>,
        n: usize,
        dim: usize,
        clusterer: &'a Cl,
        config: StackedIvfConfig,
    ) -> Self {
        Self {
            data,
            n,
            dim,
            clusterer,
            config,
        }
    }

    /// Clusters L0 (`nlist = ceil(√n)`), then hangs parents while the
    /// current top's `nlist` is at least [`MIN_NLIST_FOR_PARENT`].
    ///
    /// Returns the stacked index and L0's member permutation
    /// (`perm[old] = new`).
    pub fn build(self) -> (StackedIvf<InMemoryStore>, Vec<u32>) {
        let (mut index, perm) = StackedIvf::<InMemoryStore>::build(
            self.data,
            self.n,
            self.dim,
            self.clusterer,
            self.config,
        );
        let parent_cfg = index.config.for_parent();
        let mut cur = &mut index;
        while cur.nlist() >= MIN_NLIST_FOR_PARENT {
            cur.add_level(self.clusterer, parent_cfg.clone())
                .expect("L0 build produced centroids");
            cur = cur.parent.as_mut().expect("add_level hung a parent");
        }
        (index, perm)
    }
}

impl StackedIvf<InMemoryStore> {
    fn empty(config: StackedIvfConfig, dim: usize) -> Self {
        Self {
            config,
            parent: None,
            offsets: Vec::new(),
            centroids: InMemoryStore::new(Vec::new(), dim),
            vectors: InMemoryStore::new(Vec::new(), dim),
        }
    }

    fn populate(
        &mut self,
        members: Vec<f32>,
        centroids: Vec<f32>,
        assignments: Vec<u32>,
    ) -> Vec<u32> {
        let dim = self.centroids.dim();
        assert_eq!(centroids.len() % dim, 0);
        let nlist = centroids.len() / dim;
        assert_eq!(assignments.len(), members.len() / dim);
        let (member_perm, offsets) = cluster_sort(&assignments, nlist);
        let members = permute_rows(&members, dim, &member_perm);
        self.offsets = offsets;
        self.centroids = InMemoryStore::new(centroids, dim);
        self.vectors = InMemoryStore::new(members, dim);
        member_perm
    }

    /// Reorders this level's centroids and offset ranges (`perm[old] = new`).
    /// Member rows stay put.
    fn permute_lists(&mut self, perm: &[u32]) {
        let dim = self.centroids.dim();
        let nlist = self.nlist();
        debug_assert_eq!(perm.len(), nlist);
        let src = self.centroids.as_slice();
        let mut centroids = vec![0.0f32; nlist * dim];
        let mut offsets = vec![(0, 0); nlist];
        for (old, &new) in perm.iter().enumerate() {
            let new = new as usize;
            centroids[new * dim..(new + 1) * dim].copy_from_slice(&src[old * dim..(old + 1) * dim]);
            offsets[new] = self.offsets[old];
        }
        self.centroids = InMemoryStore::new(centroids, dim);
        self.offsets = offsets;
    }

    /// Hangs a parent on this level (or on the current top).
    ///
    /// Reorders this level's centroids and offset ranges so the new
    /// parent's member `i` is list `i`. Member rows are not moved; a
    /// parent already present is extended upward without touching this
    /// level. Returns the list permutation applied at the level that
    /// gained the parent. Fails if the top has no centroids.
    pub fn add_level<Cl: LevelClusterer<Elem = f32>>(
        &mut self,
        clusterer: &Cl,
        config: StackedIvfConfig,
    ) -> Result<Vec<u32>, AddLevelError> {
        if let Some(parent) = &mut self.parent {
            return parent.add_level(clusterer, config);
        }
        let dim = self.centroids.dim();
        let n = self.centroids.len();
        if n == 0 {
            return Err(AddLevelError::Empty);
        }
        let centroids = self.centroids.as_slice().to_vec();
        let (parent, perm) = Self::build(centroids, n, dim, clusterer, config);
        self.permute_lists(&perm);
        self.parent = Some(Box::new(parent));
        Ok(perm)
    }

    /// Clusters `data` into a single level (no parent) with
    /// `nlist = ceil(√n)`.
    ///
    /// Returns the index and the permutation of input rows into cluster
    /// order (`perm[old] = new`). [`add_level`] does not permute members.
    pub fn build<Cl: LevelClusterer<Elem = f32>>(
        data: Vec<f32>,
        n: usize,
        dim: usize,
        clusterer: &Cl,
        config: StackedIvfConfig,
    ) -> (Self, Vec<u32>) {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(data.len(), n * dim);
        if n == 0 {
            return (Self::empty(config, dim), Vec::new());
        }
        let nlist = StackedIvfConfig::nlist_for(n);
        let (centroids, assignments) = clusterer.cluster(&data, n, dim, nlist);
        let mut index = Self::empty(config, dim);
        let perm = index.populate(data, centroids, assignments);
        (index, perm)
    }

    /// Writes offsets and centroid rows for every level (L0, L1, …).
    /// L0 member vectors and [`StackedIvfConfig`] are not written.
    pub(crate) fn serialize_router_payload<W: Write + ?Sized>(
        &self,
        out: &mut W,
    ) -> io::Result<()> {
        serialize_stacked_index(self, out)
    }

    /// Rebuilds an owned index from [`serialize_router_payload`](Self::serialize_router_payload).
    ///
    /// `members` are the L0 rows in serialized order. `config` applies to
    /// L0; parents use [`PARENT_NPROBE_FRACTION`].
    #[cfg(test)]
    pub fn deserialize_owned(
        bytes: &[u8],
        members: Vec<f32>,
        dim: usize,
        config: StackedIvfConfig,
    ) -> io::Result<Self> {
        if dim == 0 || members.len() % dim != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "member matrix is not a multiple of dim",
            ));
        }
        if bytes.len() < mem::size_of::<u64>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stacked IVF payload shorter than centroids_byte_offset",
            ));
        }
        let mut cursor = bytes;
        let centroids_byte_offset = u64::deserialize(&mut cursor)? as usize;
        if centroids_byte_offset > bytes.len() || centroids_byte_offset < mem::size_of::<u64>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stacked IVF centroids_byte_offset out of range",
            ));
        }
        let level_topology =
            parse_level_topology(&bytes[mem::size_of::<u64>()..centroids_byte_offset])?;

        let mut centroid_cursor = &bytes[centroids_byte_offset..];
        let expected_values: usize = level_topology.iter().map(|(nlist, _)| nlist * dim).sum();
        if centroid_cursor.len() != expected_values * mem::size_of::<f32>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stacked IVF centroid blob length mismatch",
            ));
        }
        let mut level_centroids = Vec::with_capacity(level_topology.len());
        for (nlist, _) in &level_topology {
            let mut values = Vec::with_capacity(nlist * dim);
            for _ in 0..nlist * dim {
                values.push(f32::deserialize(&mut centroid_cursor)?);
            }
            level_centroids.push(values);
        }

        let mut index: Option<Box<StackedIvf<InMemoryStore>>> = None;
        for level in (0..level_topology.len()).rev() {
            let (nlist, offsets) = &level_topology[level];
            let level_members = if level == 0 {
                members.clone()
            } else {
                level_centroids[level - 1].clone()
            };
            if !ranges_cover_members(offsets, level_members.len() / dim) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("level {level} offsets do not cover its members"),
                ));
            }
            let _ = nlist;
            index = Some(Box::new(StackedIvf {
                config: if level == 0 {
                    config.clone()
                } else {
                    config.for_parent()
                },
                parent: index,
                offsets: offsets.clone(),
                centroids: InMemoryStore::new(level_centroids[level].clone(), dim),
                vectors: InMemoryStore::new(level_members, dim),
            }));
        }
        Ok(*index.expect("at least one level"))
    }
}

fn parse_level_topology(mut topology: &[u8]) -> io::Result<Vec<(usize, Vec<(u64, u64)>)>> {
    let num_levels = u32::deserialize(&mut topology)? as usize;
    if num_levels == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "stacked IVF has zero levels",
        ));
    }
    let mut level_topology = Vec::with_capacity(num_levels);
    for _ in 0..num_levels {
        let nlist = u32::deserialize(&mut topology)? as usize;
        let mut offsets = Vec::with_capacity(nlist);
        for _ in 0..nlist {
            let start = u64::deserialize(&mut topology)?;
            let end = u64::deserialize(&mut topology)?;
            if start > end {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "stacked IVF offset range is inverted",
                ));
            }
            offsets.push((start, end));
        }
        level_topology.push((nlist, offsets));
    }
    if !topology.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "stacked IVF topology has trailing bytes before centroids",
        ));
    }
    Ok(level_topology)
}

/// Ranges must be a partition of `0..n` (possibly not in list-id order).
fn ranges_cover_members(ranges: &[(u64, u64)], n: usize) -> bool {
    if ranges.is_empty() {
        return n == 0;
    }
    let mut packed: Vec<(u64, u64)> = ranges.to_vec();
    // Sort by (start, end) so empty ranges `(s, s)` precede a non-empty
    // `(s, e)` that shares their start. List-id order after `permute_lists`
    // is arbitrary, and sorting by start alone leaves that tie unstable —
    // an empty range after its sibling fails the adjacency check even though
    // the ranges still partition `[0, n)`.
    packed.sort_unstable();
    if packed[0].0 != 0 || packed.last().map(|&(_, end)| end as usize) != Some(n) {
        return false;
    }
    packed.windows(2).all(|pair| pair[0].1 == pair[1].0)
}

impl StackedIvf<LazyStore> {
    /// Opens a serialized index for search.
    ///
    /// `member_rows` are the L0 vectors in serialized order. `config` is
    /// not read from the file.
    pub fn open(
        slot: FileSlice,
        member_rows: FileSlice,
        dim: usize,
        config: StackedIvfConfig,
    ) -> io::Result<Self> {
        if dim == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "dim must be positive",
            ));
        }
        let header_len = mem::size_of::<u64>();
        if slot.len() < header_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stacked IVF payload shorter than centroids_byte_offset",
            ));
        }
        let header = slot.slice_to(header_len).read_bytes()?;
        let centroids_byte_offset = u64::deserialize(&mut header.as_slice())? as usize;
        if centroids_byte_offset > slot.len() || centroids_byte_offset < header_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stacked IVF centroids_byte_offset out of range",
            ));
        }
        let topology_bytes = slot.slice(header_len..centroids_byte_offset).read_bytes()?;
        let level_topology = parse_level_topology(&topology_bytes)?;

        let row_bytes = dim * mem::size_of::<f32>();
        if member_rows.len() % row_bytes != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stacked IVF member rows are not a multiple of the row stride",
            ));
        }
        let num_members = member_rows.len() / row_bytes;
        let expected: usize = level_topology
            .iter()
            .map(|(nlist, _)| nlist * row_bytes)
            .sum();
        if slot.len() - centroids_byte_offset != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stacked IVF centroid blob length mismatch",
            ));
        }

        let mut centroid_slices = Vec::with_capacity(level_topology.len());
        let mut start = centroids_byte_offset;
        for (nlist, _) in &level_topology {
            let end = start + nlist * row_bytes;
            centroid_slices.push(slot.slice(start..end));
            start = end;
        }

        // Every level's centroids are pinned: they are what APS reads, and
        // they are small. L0 members (the caller's rows) stay file-backed;
        // a parent's members are the level below's centroids, pinned too.
        let mut index: Option<Box<StackedIvf<LazyStore>>> = None;
        for level in (0..level_topology.len()).rev() {
            let (_, offsets) = &level_topology[level];
            let (level_members, level_member_count) = if level == 0 {
                (LazyStore::new(member_rows.clone(), dim), num_members)
            } else {
                (
                    LazyStore::pinned(centroid_slices[level - 1].clone(), dim)?,
                    level_topology[level - 1].0,
                )
            };
            if !ranges_cover_members(offsets, level_member_count) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("level {level} offsets do not cover its members"),
                ));
            }
            index = Some(Box::new(StackedIvf {
                config: if level == 0 {
                    config.clone()
                } else {
                    config.for_parent()
                },
                parent: index,
                offsets: offsets.clone(),
                centroids: LazyStore::pinned(centroid_slices[level].clone(), dim)?,
                vectors: level_members,
            }));
        }
        Ok(*index.expect("at least one level"))
    }
}

impl<S> StackedIvf<S>
where S: VectorArena<Elem = f32> + CentroidMatrix
{
    pub fn nlist(&self) -> usize {
        self.offsets.len()
    }

    /// Levels from this one up (L0 alone is 1; L0+L1 is 2).
    #[cfg(any(test, feature = "unstable"))]
    pub fn depth(&self) -> usize {
        1 + self.parent.as_ref().map(|p| p.depth()).unwrap_or(0)
    }

    pub fn n_probe(&self) -> usize {
        let nlist = self.nlist();
        let n = ((nlist as f32 * self.config.nprobe_fraction).ceil().max(1.0)) as usize;
        n.min(nlist).max(1)
    }

    /// Nearest members from the lists selected at this level, plus the
    /// work done across the whole stack to find them.
    ///
    /// If a parent is present it ranks which lists to probe (`n_probe`,
    /// typically 25% of this level's lists). Otherwise every centroid is
    /// scored and only the best [`PARENT_NPROBE_FRACTION`] are expanded.
    /// Lists are scanned nearest centroid first. When `recall < 1.0` and
    /// centroids are in-memory, scanning stops once the estimated recall
    /// of the lists scanned so far reaches `recall` (Quake APS).
    pub fn search(
        &self,
        query: &[S::Elem],
        k: usize,
        recall: f32,
        metric: Metric,
    ) -> (Vec<Candidate<ClusterId>>, StackedSearchStats) {
        self.search_within(query, k, recall, metric, None)
    }

    /// [`search`](Self::search) that stops opening this level's lists once it has scored
    /// `member_budget` members here (see [`member_budget`]). Parents always search without one.
    pub(crate) fn search_within(
        &self,
        query: &[S::Elem],
        k: usize,
        recall: f32,
        metric: Metric,
        member_budget: Option<usize>,
    ) -> (Vec<Candidate<ClusterId>>, StackedSearchStats) {
        if k == 0 {
            return (Vec::new(), StackedSearchStats::default());
        }

        let dim = query.len();
        let can_aps = recall < 1.0
            && aps::supports_metric(metric)
            && self.centroids.centroid_matrix().is_some();

        // Candidate lists, nearest centroid first; `candidates[0]` is P0.
        let (candidates, mut stats) = if let Some(parent) = &self.parent {
            let parent_recall = if can_aps {
                self.config.parent_recall_target
            } else {
                recall.max(1.0)
            };
            parent.search(query, self.n_probe(), parent_recall, metric)
        } else {
            let nlist = self.nlist();
            let keep = ((nlist as f32 * PARENT_NPROBE_FRACTION).ceil() as usize)
                .max(1)
                .min(nlist);
            let all: Vec<Candidate<ClusterId>> = (0..nlist)
                .map(|i| Candidate {
                    sim: self.centroids.similarity(metric, dim, i as u32, query),
                    node: ClusterId(i as u32),
                })
                .sorted()
                .rev()
                .take(keep)
                .collect();
            let stats = StackedSearchStats {
                lists_scanned: 0,
                members_scored: nlist,
            };
            (all, stats)
        };

        if candidates.is_empty() {
            return (Vec::new(), stats);
        }

        let mut estimator = can_aps.then(|| {
            let (matrix, _) = self.centroids.centroid_matrix().expect("can_aps");
            let rows: Vec<&[f32]> = candidates
                .iter()
                .map(|c| {
                    let j = usize::from(c.node);
                    &matrix[j * dim..(j + 1) * dim]
                })
                .collect();
            let sims: Vec<Similarity> = candidates.iter().map(|c| c.sim).collect();
            RecallEstimator::new(query, &sims, Box::new(rows), metric)
        });

        let mut result = BinaryHeap::with_capacity(k);
        let mut scored_here = 0;
        for c in &candidates {
            let scanned = self.scan_cluster(query, metric, c.node, &mut result, k);
            stats.members_scored += scanned;
            stats.lists_scanned += 1;
            scored_here += scanned;
            if member_budget.is_some_and(|budget| scored_here >= budget) {
                break;
            }
            let Some(estimator) = estimator.as_mut() else {
                continue;
            };
            let kth = result
                .peek()
                .filter(|_| result.len() >= k)
                .map(|Reverse(kth)| kth.sim);
            let estimate = estimator
                .cover_next(kth)
                .expect("in-memory centroid rows cannot fail to load");
            if estimate.is_some_and(|est| est >= recall) {
                break;
            }
        }

        let hits = result
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(c)| c)
            .collect();
        (hits, stats)
    }

    fn scan_cluster(
        &self,
        query: &[S::Elem],
        metric: Metric,
        list: ClusterId,
        result: &mut BinaryHeap<Reverse<Candidate<ClusterId>>>,
        k: usize,
    ) -> usize {
        let dim = query.len();
        let (start, end) = self.offsets[usize::from(list)];
        let start = start as usize;
        let end = end as usize;
        for row in start..end {
            let id = ClusterId::from(row);
            let cand = Candidate {
                sim: self.vectors.similarity(metric, dim, id.0, query),
                node: id,
            };
            if result.len() < k {
                result.push(Reverse(cand));
            } else if let Some(mut worst) = result.peek_mut() {
                if cand > worst.0 {
                    *worst = Reverse(cand);
                }
            }
        }
        end.saturating_sub(start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_data(n: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.push(i as f32);
            data.push(0.0);
        }
        data
    }

    fn test_config() -> StackedIvfConfig {
        StackedIvfConfig::default()
    }

    fn build_with_parent(
        data: &[f32],
        n: usize,
        dim: usize,
        clusterer: &SuperKMeansLevelClusterer,
    ) -> (StackedIvf<InMemoryStore>, Vec<u32>) {
        let (mut index, perm) =
            StackedIvf::<InMemoryStore>::build(data.to_vec(), n, dim, clusterer, test_config());
        let parent_cfg = index.config.for_parent();
        index
            .add_level(clusterer, parent_cfg)
            .expect("centroids exist");
        (index, perm)
    }

    #[test]
    fn test_build() {
        let dim = 2;
        let n = 16;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (index, member_perm) =
            StackedIvf::<InMemoryStore>::build(data, n, dim, &clusterer, test_config());
        assert!(index.nlist() > 1);
        assert_eq!(index.vectors.len(), n);
        assert_eq!(member_perm.len(), n);
        assert!(index.parent.is_none());
        assert_eq!(index.depth(), 1);

        let query = [1.0f32, 0.0];
        let (hits, _) = index.search(&query, 4, 1.0, Metric::L2);
        assert!(!hits.is_empty());
        assert!(
            hits.len() < n,
            "nprobe search should not score every L0 member"
        );
    }

    #[test]
    fn test_build_canonicalizes_cluster_order() {
        let dim = 2;
        let n = 32;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (index, member_perm) = build_with_parent(&data, n, dim, &clusterer);

        let mut seen = vec![false; n];
        for &new in &member_perm {
            assert!(!seen[new as usize], "duplicate row in member permutation");
            seen[new as usize] = true;
        }

        assert!(
            ranges_cover_members(&index.offsets, n),
            "L0 ranges must partition the members"
        );
        let parent = index.parent.as_ref().expect("L1 parent");
        assert!(
            ranges_cover_members(&parent.offsets, index.nlist()),
            "L1 ranges must partition the L0 centroids"
        );

        for old in 0..n {
            let new = member_perm[old] as usize;
            let stored = &index.vectors.as_slice()[new * dim..(new + 1) * dim];
            let original = &data[old * dim..(old + 1) * dim];
            assert_eq!(stored, original, "row content must survive the reorder");
        }
    }

    #[test]
    fn test_serialize_round_trips() {
        let dim = 2;
        let n = 32;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (index, _perm) = build_with_parent(&data, n, dim, &clusterer);

        let mut bytes = Vec::new();
        index.serialize_router_payload(&mut bytes).unwrap();
        let decoded = StackedIvf::<InMemoryStore>::deserialize_owned(
            &bytes,
            index.vectors.as_slice().to_vec(),
            dim,
            index.config.clone(),
        )
        .unwrap();

        assert_eq!(decoded.depth(), index.depth());
        assert_eq!(decoded.offsets, index.offsets);
        assert_eq!(decoded.centroids.as_slice(), index.centroids.as_slice());
        let (decoded_parent, parent) = (
            decoded.parent.as_ref().expect("L1"),
            index.parent.as_ref().expect("L1"),
        );
        assert_eq!(decoded_parent.offsets, parent.offsets);
        assert_eq!(
            decoded_parent.centroids.as_slice(),
            parent.centroids.as_slice()
        );

        for query in [[0.0f32, 0.0], [7.5, 0.0], [31.0, 0.0]] {
            let (expected, _) = index.search(&query, 4, 1.0, Metric::L2);
            let (got, _) = decoded.search(&query, 4, 1.0, Metric::L2);
            assert_eq!(
                got.len(),
                expected.len(),
                "reopened search must match in-memory search"
            );
            for (g, e) in got.iter().zip(&expected) {
                assert_eq!(g.node, e.node);
                assert_eq!(g.sim, e.sim);
            }
        }
    }

    #[test]
    fn test_add_level() {
        let dim = 2;
        let n = 64;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (mut index, _) =
            StackedIvf::<InMemoryStore>::build(data, n, dim, &clusterer, test_config());
        assert_eq!(index.depth(), 1);

        let parent_cfg = index.config.for_parent();
        index
            .add_level(&clusterer, parent_cfg.clone())
            .expect("L0 centroids");
        assert_eq!(index.depth(), 2);
        index
            .add_level(&clusterer, parent_cfg)
            .expect("L1 centroids");
        assert_eq!(index.depth(), 3);
        let l1 = index.parent.as_ref().expect("L1");
        let l2 = l1.parent.as_ref().expect("L2");
        assert!(l2.parent.is_none());
        assert_eq!(l2.vectors.len(), l1.nlist());
    }

    #[test]
    fn test_add_l2_does_not_reorder_l0() {
        let dim = 2;
        let n = 64;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (mut index, _) =
            StackedIvf::<InMemoryStore>::build(data, n, dim, &clusterer, test_config());
        let parent_cfg = index.config.for_parent();
        index.add_level(&clusterer, parent_cfg.clone()).expect("L1");
        let members = index.vectors.as_slice().to_vec();
        let offsets = index.offsets.clone();
        let centroids = index.centroids.as_slice().to_vec();
        index.add_level(&clusterer, parent_cfg).expect("L2");
        assert_eq!(index.vectors.as_slice(), members.as_slice());
        assert_eq!(index.offsets, offsets);
        assert_eq!(index.centroids.as_slice(), centroids.as_slice());
    }

    /// Cosine and Dot cluster by angle; L2 does not. Same rule as the segment IVF.
    #[test]
    fn level_clusterer_follows_metric() {
        assert!(!SuperKMeansLevelClusterer::for_metric(Metric::L2).angular);
        assert!(SuperKMeansLevelClusterer::for_metric(Metric::Cosine).angular);
        assert!(SuperKMeansLevelClusterer::for_metric(Metric::Dot).angular);
        assert!(!SuperKMeansLevelClusterer::default().angular);
    }

    #[test]
    fn test_nlist_for_sqrt() {
        assert_eq!(StackedIvfConfig::nlist_for(0), 0);
        assert_eq!(StackedIvfConfig::nlist_for(1), 1);
        assert_eq!(StackedIvfConfig::nlist_for(2_000), 45);
        assert_eq!(StackedIvfConfig::nlist_for(65_536), 256);
        assert_eq!(StackedIvfConfig::nlist_for(1_000_000), 1_000);
        assert_eq!(StackedIvfConfig::nlist_for(400_000), 633);
    }

    /// `c × k` is clamped to 10%..25% of the members and never drops below `k`.
    #[test]
    fn test_member_budget_clamps() {
        assert_eq!(member_budget(24_000, 1000, 3.0), 3000);
        assert_eq!(member_budget(24_000, 1000, 2.0), 2400);
        assert_eq!(member_budget(24_000, 20, 3.0), 2400);
        assert_eq!(member_budget(24_000, 1000, 8.0), 6000);
        assert_eq!(member_budget(24_000, 1000, f32::MAX), 6000);
        assert_eq!(member_budget(24_000, 10_000, 3.0), 10_000);
        assert_eq!(member_budget(10, 1, 1.0), 1);
    }

    /// The budget stops the bottom level at the first list boundary past it, while its parent
    /// still runs without a member budget; the nearest members are the same as the full scan's.
    #[test]
    fn test_member_budget_stops_bottom_level_only() {
        let dim = 2;
        let n = 2048;
        let clusterer = SuperKMeansLevelClusterer::default();
        let config = StackedIvfConfig {
            nprobe_fraction: PARENT_NPROBE_FRACTION,
            ..test_config()
        };
        let (mut index, _) =
            StackedIvf::<InMemoryStore>::build(line_data(n), n, dim, &clusterer, config);
        let parent_cfg = index.config.for_parent();
        index.add_level(&clusterer, parent_cfg).expect("L1");
        let parent = index.parent.as_deref().expect("L1");
        assert!(parent.parent.is_none());

        let query = [0.0f32, 0.0];
        let k = 16;
        let (_, parent_stats) = parent.search(&query, index.n_probe(), 1.0, Metric::L2);
        let parent_cost = parent_stats.members_scored;
        let budget = member_budget(n, k, 2.0);
        let (full_hits, full) = index.search(&query, k, 1.0, Metric::L2);
        let (hits, stats) = index.search_within(&query, k, 1.0, Metric::L2, Some(budget));

        let scored_here = stats.members_scored - parent_cost;
        let largest_list = index
            .offsets
            .iter()
            .map(|&(start, end)| (end - start) as usize)
            .max()
            .unwrap();
        assert!(
            full.members_scored - parent_cost > budget,
            "fixture must exceed the budget"
        );
        assert!(scored_here >= budget, "{scored_here} < {budget}");
        assert!(
            scored_here < budget + largest_list,
            "{scored_here} vs {budget}"
        );
        assert!(stats.lists_scanned < full.lists_scanned);
        let nodes = |hits: &[Candidate<ClusterId>]| hits.iter().map(|h| h.node).collect::<Vec<_>>();
        assert_eq!(nodes(&hits), nodes(&full_hits));
    }

    #[test]
    fn test_top_level_prunes_to_parent_nprobe_fraction() {
        let dim = 2;
        let n = 64;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (index, _) =
            StackedIvf::<InMemoryStore>::build(data, n, dim, &clusterer, test_config());
        assert!(index.parent.is_none());
        let nlist = index.nlist();
        assert!(nlist > 1);
        let keep = ((nlist as f32 * PARENT_NPROBE_FRACTION).ceil() as usize).max(1);

        let query = [0.0f32, 0.0];
        let (hits, stats) = index.search(&query, n, 1.0, Metric::L2);
        assert!(!hits.is_empty());
        assert_eq!(
            stats.lists_scanned, keep,
            "top level must expand only ceil(0.25 × nlist) lists"
        );
        assert!(
            stats.members_scored >= nlist,
            "every top centroid is scored before the prune"
        );
    }

    #[test]
    fn test_builder_skips_parent_when_below_threshold() {
        let dim = 2;
        let n = 64;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer::default();
        assert!(StackedIvfConfig::nlist_for(n) < MIN_NLIST_FOR_PARENT);
        let (index, perm) = StackedIvfBuilder::new(data, n, dim, &clusterer, test_config()).build();
        assert!(index.parent.is_none());
        assert!(index.nlist() < MIN_NLIST_FOR_PARENT);
        assert_eq!(perm.len(), n);
        let mut seen = vec![false; n];
        for &new in &perm {
            assert!(!seen[new as usize], "duplicate row in L0 permutation");
            seen[new as usize] = true;
        }
    }

    #[test]
    fn test_add_level_errors_when_empty() {
        let dim = 2;
        let mut index = StackedIvf::<InMemoryStore>::empty(StackedIvfConfig::default(), dim);
        let clusterer = SuperKMeansLevelClusterer::default();
        assert_eq!(
            index.add_level(&clusterer, StackedIvfConfig::default()),
            Err(AddLevelError::Empty)
        );
    }

    #[test]
    fn test_slice_backed_open_matches_owned_search() {
        let dim = 2;
        let n = 32;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (index, _perm) = build_with_parent(&data, n, dim, &clusterer);

        let mut slot = Vec::new();
        index.serialize_router_payload(&mut slot).unwrap();
        let member_bytes: Vec<u8> = index
            .vectors
            .as_slice()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let opened = StackedIvf::<LazyStore>::open(
            FileSlice::from(slot),
            FileSlice::from(member_bytes),
            dim,
            index.config.clone(),
        )
        .unwrap();

        assert_eq!(opened.depth(), index.depth());
        assert_eq!(opened.offsets, index.offsets);
        assert_eq!(opened.nlist(), index.nlist());
        assert_eq!(
            opened.parent.as_ref().unwrap().offsets,
            index.parent.as_ref().unwrap().offsets
        );

        for query in [[0.0f32, 0.0], [7.5, 0.0], [15.2, 0.0], [31.0, 0.0]] {
            let (expected, _) = index.search(&query, 4, 1.0, Metric::L2);
            let (got, _) = opened.search(&query, 4, 1.0, Metric::L2);
            assert_eq!(got.len(), expected.len());
            for (g, e) in got.iter().zip(&expected) {
                assert_eq!(g.node, e.node);
                assert_eq!(g.sim, e.sim);
            }
        }
    }

    /// Empty lists after `add_level` share start offsets with the next
    /// non-empty list. Cover validation must still accept the partition
    /// (reproduces the centroid_ratio-sensitive open failure).
    #[test]
    fn stacked_open_accepts_empty_lists_after_add_level() {
        let dim = 32usize;
        let n = 20_000usize;
        let mut data = vec![0.0f32; n * dim];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i * 2654435761usize) % 1000) as f32 / 1000.0;
        }
        let clusterer = SuperKMeansLevelClusterer::default();
        let config = StackedIvfConfig {
            nprobe_fraction: PARENT_NPROBE_FRACTION,
            ..StackedIvfConfig::default()
        };
        // √20k ≈ 142 < 256, so the builder would stay flat; hang a parent
        // so permute_lists scrambles L0 list order (and can leave empties).
        let (mut index, _perm) =
            StackedIvf::<InMemoryStore>::build(data, n, dim, &clusterer, config.clone());
        let parent_cfg = index.config.for_parent();
        index.add_level(&clusterer, parent_cfg).expect("L1");
        assert_eq!(index.depth(), 2);
        assert!(
            index.offsets.iter().any(|&(start, end)| start == end),
            "need at least one empty list to exercise the tie-break"
        );
        let mut slot = Vec::new();
        index.serialize_router_payload(&mut slot).unwrap();
        let member_bytes: Vec<u8> = index
            .vectors
            .as_slice()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        StackedIvf::<LazyStore>::open(
            FileSlice::from(slot),
            FileSlice::from(member_bytes),
            dim,
            config,
        )
        .expect("empty lists after add_level must still open");
    }

    #[test]
    fn test_slice_backed_open_rejects_truncation() {
        let dim = 2;
        let n = 32;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (index, _perm) =
            StackedIvf::<InMemoryStore>::build(data, n, dim, &clusterer, test_config());
        let mut slot = Vec::new();
        index.serialize_router_payload(&mut slot).unwrap();
        let member_bytes: Vec<u8> = index
            .vectors
            .as_slice()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        let truncated = slot[..slot.len() - 4].to_vec();
        assert!(StackedIvf::<LazyStore>::open(
            FileSlice::from(truncated),
            FileSlice::from(member_bytes.clone()),
            dim,
            StackedIvfConfig::default(),
        )
        .is_err());

        // Wrong member count: offsets no longer cover the members.
        let short_members = member_bytes[..member_bytes.len() - dim * 4].to_vec();
        assert!(StackedIvf::<LazyStore>::open(
            FileSlice::from(slot),
            FileSlice::from(short_members),
            dim,
            StackedIvfConfig::default(),
        )
        .is_err());
    }

    fn two_blobs(n_per: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n_per * 2 * 2);
        for i in 0..n_per {
            data.push(i as f32 * 0.05);
            data.push(0.0);
        }
        for i in 0..n_per {
            data.push(100.0 + i as f32 * 0.05);
            data.push(0.0);
        }
        data
    }

    #[test]
    fn test_aps_recall_zero_still_scans_p0() {
        let n_per = 16;
        let data = two_blobs(n_per);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (index, _) =
            StackedIvf::<InMemoryStore>::build(data, n_per * 2, 2, &clusterer, test_config());
        let (hits, stats) = index.search(&[0.0f32, 0.0], 4, 0.0, Metric::L2);
        assert!(
            stats.lists_scanned >= 1,
            "recall=0 must still scan P0, got {}",
            stats.lists_scanned
        );
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_aps_low_recall_scans_fewer_lists_than_full() {
        let n_per = 32;
        let data = two_blobs(n_per);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (mut index, _) =
            StackedIvf::<InMemoryStore>::build(data, n_per * 2, 2, &clusterer, test_config());
        index.config.nprobe_fraction = 1.0;
        let query = [0.0f32, 0.0];
        let (full_hits, full) = index.search(&query, 4, 1.0, Metric::L2);
        let (low_hits, low) = index.search(&query, 4, 0.8, Metric::L2);
        assert!(
            low.lists_scanned <= full.lists_scanned,
            "APS 0.8 scanned {} lists, full scanned {}",
            low.lists_scanned,
            full.lists_scanned
        );
        assert!(!low_hits.is_empty());
        assert!(
            low_hits[0].sim >= full_hits[0].sim
                || low_hits.iter().any(|h| h.node == full_hits[0].node),
            "low-recall top hit should be in or as good as the full-scan nearest"
        );
    }

    /// An opened (file-backed) index pins its centroids, so APS runs there
    /// too: a low recall target scans fewer lists than the full nprobe
    /// pass, and the two agree on the nearest member.
    #[test]
    fn test_opened_index_runs_aps() {
        let n_per = 32;
        let data = two_blobs(n_per);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (mut index, _) =
            StackedIvf::<InMemoryStore>::build(data, n_per * 2, 2, &clusterer, test_config());
        index.config.nprobe_fraction = 1.0;

        let mut slot = Vec::new();
        index.serialize_router_payload(&mut slot).unwrap();
        let member_bytes: Vec<u8> = index
            .vectors
            .as_slice()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let opened = StackedIvf::<LazyStore>::open(
            FileSlice::from(slot),
            FileSlice::from(member_bytes),
            2,
            index.config.clone(),
        )
        .unwrap();
        assert!(
            opened.centroids.centroid_matrix().is_some(),
            "opened router centroids must be pinned for APS"
        );

        let query = [0.0f32, 0.0];
        let (full_hits, full) = opened.search(&query, 4, 1.0, Metric::L2);
        let (low_hits, low) = opened.search(&query, 4, 0.8, Metric::L2);
        assert!(
            low.lists_scanned < full.lists_scanned,
            "APS 0.8 scanned {} lists, full scanned {}",
            low.lists_scanned,
            full.lists_scanned
        );
        assert_eq!(low_hits[0].node, full_hits[0].node);
    }

    #[test]
    fn test_aps_parent_keeps_high_recall() {
        let n_per = 32;
        let data = two_blobs(n_per);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (index, _) = build_with_parent(&data, n_per * 2, 2, &clusterer);
        let query = [0.0f32, 0.0];
        let (leaf, _) = index.search(&query, 4, 0.9, Metric::L2);
        let (full, _) = index.search(&query, 4, 1.0, Metric::L2);
        assert!(!leaf.is_empty());
        assert_eq!(leaf[0].node, full[0].node);
    }

    /// Cosine APS depends only on direction: scaling the query or the
    /// data leaves the scan unchanged, and a low target scans no more than
    /// the full nprobe.
    #[test]
    fn test_aps_cosine_ignores_vector_norms() {
        let n = 256;
        let angle = |i: usize| i as f32 * std::f32::consts::TAU / n as f32;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let norm = 1.0 + (i % 7) as f32;
                [norm * angle(i).cos(), norm * angle(i).sin()]
            })
            .collect();
        let clusterer = SuperKMeansLevelClusterer::default();
        let (mut index, _) =
            StackedIvf::<InMemoryStore>::build(data, n, 2, &clusterer, test_config());
        index.config.nprobe_fraction = 1.0;
        let query = [0.3f32.cos(), 0.3f32.sin()];
        let scaled = [query[0] * 10.0, query[1] * 10.0];
        let (_, full) = index.search(&query, 8, 1.0, Metric::Cosine);
        let (unit_hits, unit) = index.search(&query, 8, 0.8, Metric::Cosine);
        let (scaled_hits, scaled) = index.search(&scaled, 8, 0.8, Metric::Cosine);
        assert!(unit.lists_scanned <= full.lists_scanned);
        assert_eq!(unit.lists_scanned, scaled.lists_scanned);
        let nodes = |hits: &[Candidate<ClusterId>]| hits.iter().map(|h| h.node).collect::<Vec<_>>();
        assert_eq!(nodes(&unit_hits), nodes(&scaled_hits));
    }

    /// Dot scores past 1 would clamp the query radius to zero and stop
    /// APS after the first full heap; Dot scans its full nprobe instead.
    #[test]
    fn test_aps_is_off_for_dot() {
        let n_per = 32;
        let data = two_blobs(n_per);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (mut index, _) =
            StackedIvf::<InMemoryStore>::build(data, n_per * 2, 2, &clusterer, test_config());
        index.config.nprobe_fraction = 1.0;
        let query = [1.0f32, 0.0];
        let (full_hits, full) = index.search(&query, 4, 1.0, Metric::Dot);
        let (low_hits, low) = index.search(&query, 4, 0.5, Metric::Dot);
        assert!(full.lists_scanned > 1);
        assert_eq!(low.lists_scanned, full.lists_scanned);
        assert_eq!(low_hits, full_hits);
    }
}
