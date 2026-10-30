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
use crate::vector::search::prepared::VectorQuery;
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

/// Default `nprobe_fraction` for parent levels (L1, L2, ...).
pub(crate) const PARENT_NPROBE_FRACTION: f32 = 0.25;

/// Member-heap overscan when converting a size-`k` request into how many
/// lists to open: beam ~ `LIST_OVERSCAN * k` members, then to lists.
pub(crate) const LIST_OVERSCAN: usize = 2;

/// [`StackedIvfBuilder`] hangs a parent only while the current top has at
/// least this many lists. Below that, the top is scored in full and pruned
/// with [`StackedIvf::lists_to_open`] (floor [`PARENT_NPROBE_FRACTION`]).
pub(crate) const MIN_NLIST_FOR_PARENT: usize = 256;

/// Search and clustering knobs for this level. Not persisted.
#[derive(Clone, Debug)]
pub struct StackedIvfConfig {
    /// Initial candidate fraction `f_M`: `|S| = nprobe_fraction * nlist`.
    /// List selection at search time is [`StackedIvf::lists_to_open`]: floor
    /// [`PARENT_NPROBE_FRACTION`], beam ~ [`LIST_OVERSCAN`] * `k` members, cap
    /// `nlist`.
    pub nprobe_fraction: f32,
}

impl Default for StackedIvfConfig {
    fn default() -> Self {
        Self {
            nprobe_fraction: 0.02,
        }
    }
}

impl StackedIvfConfig {
    fn for_parent(&self) -> Self {
        Self {
            nprobe_fraction: PARENT_NPROBE_FRACTION,
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
    /// held in memory so scoring skips the file.
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

    fn batch_similarity(
        &self,
        query: &[Self::Elem],
        metric: Metric,
        query_norm: f32,
        rows: std::ops::Range<usize>,
        out: &mut Vec<Similarity>,
    ) {
        match &self.pinned {
            Some(values) => values.batch_similarity(query, metric, query_norm, rows, out),
            None => self
                .arena
                .batch_similarity(query, metric, query_norm, rows, out),
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

/// Work done by one [`StackedIvf::search_prepared`], summed over this level and
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
where S: VectorArena<Elem = f32>
{
    pub fn nlist(&self) -> usize {
        self.offsets.len()
    }

    /// Levels from this one up (L0 alone is 1; L0+L1 is 2).
    #[cfg(any(test, feature = "unstable"))]
    pub fn depth(&self) -> usize {
        1 + self.parent.as_ref().map(|p| p.depth()).unwrap_or(0)
    }

    /// Fraction-only probe size (`nprobe_fraction * nlist`). Search uses
    /// [`Self::lists_to_open`] instead; this remains for benches that sweep
    /// `nprobe_fraction` directly.
    #[cfg(any(test, feature = "unstable"))]
    pub fn n_probe(&self) -> usize {
        let nlist = self.nlist();
        let n = ((nlist as f32 * self.config.nprobe_fraction).ceil().max(1.0)) as usize;
        n.min(nlist).max(1)
    }

    /// How many lists to open so a size-`k` member heap stays fed.
    ///
    /// Converts the member request into a list count with beam
    /// [`LIST_OVERSCAN`] * `k`, then clamps to
    /// `[ceil(PARENT_NPROBE_FRACTION * nlist), nlist]`. Used as the parent
    /// ask when a parent exists, and as the flat-top keep otherwise.
    pub fn lists_to_open(&self, k: usize) -> usize {
        let nlist = self.nlist();
        if nlist == 0 {
            return 0;
        }
        let floor = ((nlist as f32 * PARENT_NPROBE_FRACTION).ceil() as usize).max(1);
        let n_members = self
            .offsets
            .iter()
            .map(|&(start, end)| (end - start) as usize)
            .sum::<usize>()
            .max(1);
        let beam_members = k.saturating_mul(LIST_OVERSCAN).max(1);
        let for_k = ((beam_members as f64 * nlist as f64) / n_members as f64).ceil() as usize;
        for_k.max(floor).min(nlist).max(1)
    }

    /// Nearest members from the lists selected at this level, plus the
    /// work done across the whole stack to find them.
    ///
    /// List count is [`Self::lists_to_open`]: enough clusters to feed the
    /// size-`k` heap (~2x member overscan), floored at 25% of `nlist` and
    /// capped at `nlist`. A parent is asked for that many of this level's
    /// lists; without a parent every centroid is scored and only that many
    /// are expanded. Lists are scanned nearest centroid first.
    /// [`super::aps`] decides when to stop short of the shortlist.
    #[cfg(any(test, feature = "unstable"))]
    pub fn search(
        &self,
        query: &[S::Elem],
        k: usize,
        metric: Metric,
    ) -> (Vec<Candidate<ClusterId>>, StackedSearchStats) {
        self.search_prepared(
            &VectorQuery::new(metric, std::sync::Arc::new(query.to_vec())),
            k,
        )
    }

    pub(crate) fn search_prepared<T: VectorElement>(
        &self,
        query: &VectorQuery<T>,
        k: usize,
    ) -> (Vec<Candidate<ClusterId>>, StackedSearchStats) {
        if k == 0 {
            return (Vec::new(), StackedSearchStats::default());
        }

        let open = self.lists_to_open(k);

        // Candidate lists, nearest centroid first; `candidates[0]` is P0.
        let (candidates, mut stats) = if let Some(parent) = &self.parent {
            parent.search_prepared(query, open)
        } else {
            let nlist = self.nlist();
            let mut scores = Vec::with_capacity(nlist);
            self.centroids.batch_similarity(
                query.widened(),
                query.metric(),
                query.routing_norm(),
                0..nlist,
                &mut scores,
            );
            let all: Vec<Candidate<ClusterId>> = scores
                .into_iter()
                .enumerate()
                .map(|(i, sim)| Candidate {
                    sim,
                    node: ClusterId(i as u32),
                })
                .sorted()
                .rev()
                .take(open)
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

        let mut probe = super::aps::open();
        let mut result = BinaryHeap::with_capacity(k);
        let mut scores = Vec::new();
        for c in &candidates {
            let scanned = self.scan_cluster(query, c.node, &mut result, k, &mut scores);
            stats.members_scored += scanned;
            stats.lists_scanned += 1;
            let kth = if result.len() >= k {
                result
                    .peek()
                    .map(|worst: &Reverse<Candidate<ClusterId>>| worst.0.sim)
            } else {
                None
            };
            if probe.observe(kth) {
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

    fn scan_cluster<T: VectorElement>(
        &self,
        query: &VectorQuery<T>,
        list: ClusterId,
        result: &mut BinaryHeap<Reverse<Candidate<ClusterId>>>,
        k: usize,
        scores: &mut Vec<Similarity>,
    ) -> usize {
        let (start, end) = self.offsets[usize::from(list)];
        let start = start as usize;
        let end = end as usize;
        self.vectors.batch_similarity(
            query.widened(),
            query.metric(),
            query.routing_norm(),
            start..end,
            scores,
        );
        for (row, &sim) in (start..end).zip(scores.iter()) {
            let cand = Candidate {
                sim,
                node: ClusterId::from(row),
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
        let (hits, _) = index.search(&query, 4, Metric::L2);
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
            let (expected, _) = index.search(&query, 4, Metric::L2);
            let (got, _) = decoded.search(&query, 4, Metric::L2);
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

    /// A run of lists that miss the full heap stops the walk short of the shortlist.
    #[test]
    fn test_patience_stops_before_the_shortlist_is_exhausted() {
        let dim = 2;
        let n = 2048;
        let clusterer = SuperKMeansLevelClusterer::default();
        let (index, _) =
            StackedIvf::<InMemoryStore>::build(line_data(n), n, dim, &clusterer, test_config());
        assert!(index.parent.is_none());

        let query = [0.0f32, 0.0];
        let k = 16;
        let shortlist = index.lists_to_open(k);
        let (hits, stats) = index.search(&query, k, Metric::L2);

        assert_eq!(hits.len(), k);
        assert!(
            stats.lists_scanned < shortlist,
            "stop scanned {} lists, shortlist {}",
            stats.lists_scanned,
            shortlist
        );
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
        let floor = ((nlist as f32 * PARENT_NPROBE_FRACTION).ceil() as usize).max(1);
        // Small k: beam stays under the floor, so keep == 25% of nlist.
        assert_eq!(index.lists_to_open(1), floor);

        let query = [0.0f32, 0.0];
        let (hits, stats) = index.search(&query, 1, Metric::L2);
        assert!(!hits.is_empty());
        // Cap is the shortlist; patience may stop earlier after five quiet lists.
        assert!(
            stats.lists_scanned <= floor,
            "small-k top level must not expand past ceil(0.25 × nlist), scanned {} floor {}",
            stats.lists_scanned,
            floor
        );
        assert!(
            stats.members_scored >= nlist,
            "every top centroid is scored before the prune"
        );
    }

    #[test]
    fn test_lists_to_open_grows_with_k_then_caps_at_nlist() {
        let dim = 2;
        let n = 64;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer::default();
        let (index, _) =
            StackedIvf::<InMemoryStore>::build(data, n, dim, &clusterer, test_config());
        let nlist = index.nlist();
        let floor = ((nlist as f32 * PARENT_NPROBE_FRACTION).ceil() as usize).max(1);

        assert_eq!(index.lists_to_open(1), floor);
        // 2k members ≈ half the corpus → about half the lists (above the floor).
        let mid = index.lists_to_open(n / 4);
        assert!(
            mid > floor && mid < nlist,
            "mid-size k should sit between the 25% floor and nlist, got {mid}"
        );
        assert_eq!(index.lists_to_open(n), nlist);

        let query = [0.0f32, 0.0];
        let (_, stats) = index.search(&query, n, Metric::L2);
        assert_eq!(
            stats.lists_scanned, nlist,
            "k covering all members must open every list"
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

        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            for query in [[0.0f32, 0.0], [7.5, 0.0], [15.2, 0.0], [31.0, 0.0]] {
                let (expected, _) = index.search(&query, 4, metric);
                let (got, _) = opened.search(&query, 4, metric);
                assert_eq!(got, expected);
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
}
