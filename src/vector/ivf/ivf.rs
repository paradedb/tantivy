//! Multi-level (stacked) IVF.
//!
//! L0 members are the items being routed. L1 is an IVF over L0's
//! centroids, L2 over L1's, and so on. Each list is a contiguous row
//! range in [`IvfIndex::offsets`].
//!
//! [`InMemoryStackedIvf::build`] creates one level. [`InMemoryStackedIvf::add_level`]
//! hangs a parent on this level and reorders this level's centroids and
//! offset ranges so parent member `i` is list `i`. Member rows stay put;
//! a later parent only shuffles this level's metadata. [`IvfIndexBuilder`]
//! walks up, calling `add_level` until the top has at most
//! `branching_factor` lists.
//!
//! Search works on any [`VectorArena`]. Build and payload serialization require
//! owned [`InMemoryStore`]s. [`LazyStackedIvf::open`] is search-only.

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
use crate::vector::{Candidate, FileSliceArena, Similarity, VectorArena, VectorElement};

/// Row index into a level's centroid or member arena. Not a graph [`super::NodeId`].
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
pub const PARENT_NPROBE_FRACTION: f32 = 0.1;

/// Search and clustering knobs for this level. Not persisted.
#[derive(Clone, Debug)]
pub struct IvfConfig {
    pub nprobe_fraction: f32,
    /// Target list size: `nlist ≈ n / branching_factor`.
    pub branching_factor: usize,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            nprobe_fraction: 0.1,
            branching_factor: 16,
        }
    }
}

impl IvfConfig {
    pub fn new(branching_factor: usize) -> Self {
        Self {
            branching_factor,
            ..Default::default()
        }
    }

    fn for_parent(&self) -> Self {
        Self {
            nprobe_fraction: PARENT_NPROBE_FRACTION,
            branching_factor: self.branching_factor,
        }
    }

    fn nlist_for(&self, n: usize) -> usize {
        assert!(
            self.branching_factor >= 2,
            "branching_factor must be >= 2, got {}",
            self.branching_factor
        );
        (n / self.branching_factor).max(1).min(n)
    }
}

/// No centroids at the level [`InMemoryStackedIvf::add_level`] would extend.
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

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Deref for InMemoryStore {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.data
    }
}

/// Cluster `n` vectors into about `nlist` balanced lists.
pub trait IvfLevelClusterer {
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

/// [`HierarchicalSuperKMeans`] with `max_leaf_size ≈ n / nlist`.
#[derive(Clone, Debug, Default)]
pub struct SuperKMeansLevelClusterer {
    pub iters_per_split: u32,
}

impl IvfLevelClusterer for SuperKMeansLevelClusterer {
    type Elem = f32;

    fn cluster(&self, data: &[f32], n: usize, dim: usize, nlist: usize) -> (Vec<f32>, Vec<u32>) {
        assert!(n > 0 && dim > 0 && nlist > 0);
        assert_eq!(data.len(), n * dim);
        let nlist = nlist.min(n).max(1);
        let max_leaf_size = n.div_ceil(nlist).max(1);

        let base = SuperKMeansConfig {
            data_already_rotated: true,
            ..Default::default()
        };
        let cfg = HierarchicalSuperKMeansConfig {
            base,
            max_leaf_size,
            iters_per_split: self.iters_per_split.max(1),
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
/// available for any [`VectorArena`]; [`build`](IvfIndex::build) and
/// [`add_level`](IvfIndex::add_level) require [`InMemoryStore`].
pub struct IvfIndex<C, M> {
    pub config: IvfConfig,
    pub parent: Option<Box<IvfIndex<C, C>>>,
    /// Per-list `(start, end)` into [`Self::vectors`]. Not necessarily
    /// packed in list-id order after a parent is added above this level.
    pub offsets: Vec<(u64, u64)>,
    pub centroids: C,
    pub vectors: M,
}

/// Owned stacked IVF (`build` / `add_level` / `serialize`).
pub type InMemoryStackedIvf = IvfIndex<InMemoryStore, InMemoryStore>;

/// File-backed stacked IVF (`open` / search only).
pub type LazyStackedIvf = IvfIndex<LazyStore, LazyStore>;

/// File-backed row-major `f32` matrix.
pub struct LazyStore {
    arena: FileSliceArena<f32>,
    dim: usize,
}

impl LazyStore {
    /// Wraps `rows` (row-major `n × dim` little-endian `f32`s).
    pub fn new(rows: FileSlice, dim: usize) -> Self {
        LazyStore {
            arena: FileSliceArena::new(rows),
            dim,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.arena.num_vectors(self.dim)
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
        self.arena.similarity(metric, dim, index, query)
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

fn serialize_stacked_index<C, M, W>(index: &IvfIndex<C, M>, out: &mut W) -> io::Result<()>
where
    C: SerializableStore,
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

    fn serialize_level<C, M>(index: &IvfIndex<C, M>, out: &mut Vec<u8>) -> io::Result<()>
    where C: SerializableStore {
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

/// Builds a stacked IVF, hanging parents until the top has at most
/// [`IvfConfig::branching_factor`] lists.
///
/// [`InMemoryStackedIvf::build`] is one level. The builder walks up, calling
/// [`InMemoryStackedIvf::add_level`] on each current top. Only L0's member
/// permutation is returned — later levels shuffle metadata, not L0 rows.
pub struct IvfIndexBuilder<'a, Cl> {
    data: Vec<f32>,
    n: usize,
    dim: usize,
    clusterer: &'a Cl,
    config: IvfConfig,
}

impl<'a, Cl: IvfLevelClusterer<Elem = f32>> IvfIndexBuilder<'a, Cl> {
    pub fn new(data: Vec<f32>, n: usize, dim: usize, clusterer: &'a Cl, config: IvfConfig) -> Self {
        Self {
            data,
            n,
            dim,
            clusterer,
            config,
        }
    }

    /// Clusters L0, then walks up calling [`InMemoryStackedIvf::add_level`]
    /// while the current top's `nlist` is greater than `branching_factor`.
    ///
    /// Returns the stacked index and L0's member permutation
    /// (`perm[old] = new`).
    pub fn build(self) -> (InMemoryStackedIvf, Vec<u32>) {
        let (mut index, perm) =
            InMemoryStackedIvf::build(self.data, self.n, self.dim, self.clusterer, self.config);
        let parent_cfg = index.config.for_parent();
        let branching_factor = index.config.branching_factor;
        let mut cur = &mut index;
        while cur.nlist() > branching_factor {
            cur.add_level(self.clusterer, parent_cfg.clone())
                .expect("L0 build produced centroids");
            cur = cur.parent.as_mut().expect("add_level hung a parent");
        }
        (index, perm)
    }
}

impl IvfIndex<InMemoryStore, InMemoryStore> {
    fn empty(config: IvfConfig, dim: usize) -> Self {
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
    pub fn add_level<Cl: IvfLevelClusterer<Elem = f32>>(
        &mut self,
        clusterer: &Cl,
        config: IvfConfig,
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

    /// Clusters `data` into a single level (no parent).
    ///
    /// Returns the index and the permutation of input rows into cluster
    /// order (`perm[old] = new`). [`add_level`] does not permute members.
    pub fn build<Cl: IvfLevelClusterer<Elem = f32>>(
        data: Vec<f32>,
        n: usize,
        dim: usize,
        clusterer: &Cl,
        config: IvfConfig,
    ) -> (Self, Vec<u32>) {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(data.len(), n * dim);
        if n == 0 {
            return (Self::empty(config, dim), Vec::new());
        }
        let nlist = config.nlist_for(n);
        let (centroids, assignments) = clusterer.cluster(&data, n, dim, nlist);
        let mut index = Self::empty(config, dim);
        let perm = index.populate(data, centroids, assignments);
        (index, perm)
    }

    /// Writes offsets and centroid rows for every level (L0, L1, …).
    /// L0 member vectors and [`IvfConfig`] are not written.
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
    pub fn deserialize_owned(
        bytes: &[u8],
        members: Vec<f32>,
        dim: usize,
        config: IvfConfig,
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

        let mut index: Option<Box<InMemoryStackedIvf>> = None;
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
            index = Some(Box::new(IvfIndex {
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
    packed.sort_unstable_by_key(|&(start, _)| start);
    if packed[0].0 != 0 || packed.last().map(|&(_, end)| end as usize) != Some(n) {
        return false;
    }
    packed.windows(2).all(|pair| pair[0].1 == pair[1].0)
}

impl LazyStackedIvf {
    /// Opens a serialized index for search.
    ///
    /// `member_rows` are the L0 vectors in serialized order. `config` is
    /// not read from the file.
    pub fn open(
        slot: FileSlice,
        member_rows: FileSlice,
        dim: usize,
        config: IvfConfig,
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

        let mut index: Option<Box<LazyStackedIvf>> = None;
        for level in (0..level_topology.len()).rev() {
            let (_, offsets) = &level_topology[level];
            let (level_members, level_member_count) = if level == 0 {
                (member_rows.clone(), num_members)
            } else {
                (
                    centroid_slices[level - 1].clone(),
                    level_topology[level - 1].0,
                )
            };
            if !ranges_cover_members(offsets, level_member_count) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("level {level} offsets do not cover its members"),
                ));
            }
            index = Some(Box::new(IvfIndex {
                config: if level == 0 {
                    config.clone()
                } else {
                    config.for_parent()
                },
                parent: index,
                offsets: offsets.clone(),
                centroids: LazyStore::new(centroid_slices[level].clone(), dim),
                vectors: LazyStore::new(level_members, dim),
            }));
        }
        Ok(*index.expect("at least one level"))
    }
}

impl<C, M> IvfIndex<C, M>
where
    C: VectorArena,
    M: VectorArena<Elem = C::Elem>,
{
    pub fn nlist(&self) -> usize {
        self.offsets.len()
    }

    /// `nlist` of the highest parent (this level when there is none).
    fn top_nlist(&self) -> usize {
        match &self.parent {
            Some(parent) => parent.top_nlist(),
            None => self.nlist(),
        }
    }

    /// Levels from this one up (L0 alone is 1; L0+L1 is 2).
    pub fn depth(&self) -> usize {
        1 + self.parent.as_ref().map(|p| p.depth()).unwrap_or(0)
    }

    pub fn n_probe(&self) -> usize {
        let nlist = self.nlist();
        let n = ((nlist as f32 * self.config.nprobe_fraction).ceil().max(1.0)) as usize;
        n.min(nlist).max(1)
    }

    /// Nearest members from the lists selected at this level.
    ///
    /// If a parent is present it ranks which lists to probe; otherwise
    /// all centroids at this level are scored.
    pub fn search(
        &self,
        query: &[C::Elem],
        k: usize,
        recall: f32,
        metric: Metric,
    ) -> Vec<Candidate<ClusterId>> {
        let dim = query.len();
        let mut frontier: BinaryHeap<Candidate<ClusterId>> = BinaryHeap::new();
        let mut result: BinaryHeap<Reverse<Candidate<ClusterId>>> = BinaryHeap::with_capacity(k);

        let n_probe = self.n_probe();
        if let Some(parent) = &self.parent {
            frontier.extend(parent.search(query, n_probe, recall, metric));
        } else {
            for i in 0..self.nlist() {
                let id = ClusterId::from(i);
                frontier.push(Candidate {
                    sim: self.centroids.similarity(metric, dim, id.0, query),
                    node: id,
                });
            }
        }

        let cumulative_recall = 0.0;
        while let Some(candidate) = frontier.pop() {
            let cluster = usize::from(candidate.node);
            let (start, end) = self.offsets[cluster];
            let start = start as usize;
            let end = end as usize;
            for row in start..end {
                let id = ClusterId::from(row);
                result.push(Reverse(Candidate {
                    sim: self.vectors.similarity(metric, dim, id.0, query),
                    node: id,
                }));
            }

            if cumulative_recall >= recall {
                break;
            }
        }

        result
            .into_iter()
            .map(|Reverse(c)| c)
            .sorted()
            .rev()
            .take(k)
            .collect()
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

    fn build_with_parent(
        data: &[f32],
        n: usize,
        dim: usize,
        clusterer: &SuperKMeansLevelClusterer,
        branching_factor: usize,
    ) -> (InMemoryStackedIvf, Vec<u32>) {
        let (mut index, perm) = InMemoryStackedIvf::build(
            data.to_vec(),
            n,
            dim,
            clusterer,
            IvfConfig::new(branching_factor),
        );
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
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (index, member_perm) =
            InMemoryStackedIvf::build(data, n, dim, &clusterer, IvfConfig::new(4));
        assert!(index.nlist() > 1);
        assert_eq!(index.vectors.len(), n);
        assert_eq!(member_perm.len(), n);
        assert!(index.parent.is_none());
        assert_eq!(index.depth(), 1);

        let query = [1.0f32, 0.0];
        let hits = index.search(&query, 4, 1.0, Metric::L2);
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
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (index, member_perm) = build_with_parent(&data, n, dim, &clusterer, 2);

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
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (index, _perm) = build_with_parent(&data, n, dim, &clusterer, 2);

        let mut bytes = Vec::new();
        index.serialize_router_payload(&mut bytes).unwrap();
        let decoded = InMemoryStackedIvf::deserialize_owned(
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
            let expected = index.search(&query, 4, 1.0, Metric::L2);
            let got = decoded.search(&query, 4, 1.0, Metric::L2);
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
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (mut index, _) = InMemoryStackedIvf::build(data, n, dim, &clusterer, IvfConfig::new(2));
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
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (mut index, _) = InMemoryStackedIvf::build(data, n, dim, &clusterer, IvfConfig::new(2));
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

    #[test]
    fn test_builder_stacks_until_top_nlist_le_branching_factor() {
        let dim = 2;
        let n = 64;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let config = IvfConfig::new(2);
        let (index, perm) = IvfIndexBuilder::new(data, n, dim, &clusterer, config.clone()).build();
        assert!(index.depth() > 1, "builder must hang at least one parent");
        assert!(
            index.top_nlist() <= config.branching_factor,
            "top nlist {} should be <= branching_factor {}",
            index.top_nlist(),
            config.branching_factor
        );
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
        let mut index = InMemoryStackedIvf::empty(IvfConfig::default(), dim);
        let clusterer = SuperKMeansLevelClusterer::default();
        assert_eq!(
            index.add_level(&clusterer, IvfConfig::default()),
            Err(AddLevelError::Empty)
        );
    }

    #[test]
    fn test_slice_backed_open_matches_owned_search() {
        let dim = 2;
        let n = 32;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (index, _perm) = build_with_parent(&data, n, dim, &clusterer, 2);

        let mut slot = Vec::new();
        index.serialize_router_payload(&mut slot).unwrap();
        let member_bytes: Vec<u8> = index
            .vectors
            .as_slice()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let opened = LazyStackedIvf::open(
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
            let expected = index.search(&query, 4, 1.0, Metric::L2);
            let got = opened.search(&query, 4, 1.0, Metric::L2);
            assert_eq!(got.len(), expected.len());
            for (g, e) in got.iter().zip(&expected) {
                assert_eq!(g.node, e.node);
                assert_eq!(g.sim, e.sim);
            }
        }
    }

    #[test]
    fn test_slice_backed_open_rejects_truncation() {
        let dim = 2;
        let n = 32;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (index, _perm) = InMemoryStackedIvf::build(data, n, dim, &clusterer, IvfConfig::new(2));
        let mut slot = Vec::new();
        index.serialize_router_payload(&mut slot).unwrap();
        let member_bytes: Vec<u8> = index
            .vectors
            .as_slice()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        let truncated = slot[..slot.len() - 4].to_vec();
        assert!(LazyStackedIvf::open(
            FileSlice::from(truncated),
            FileSlice::from(member_bytes.clone()),
            dim,
            IvfConfig::default(),
        )
        .is_err());

        // Wrong member count: offsets no longer cover the members.
        let short_members = member_bytes[..member_bytes.len() - dim * 4].to_vec();
        assert!(LazyStackedIvf::open(
            FileSlice::from(slot),
            FileSlice::from(short_members),
            dim,
            IvfConfig::default(),
        )
        .is_err());
    }
}
