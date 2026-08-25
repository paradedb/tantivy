//! Quake-shaped multi-level (stacked) IVF.
//!
//! Levels stack bottom-up: level 0's members are the items being routed
//! (in production, the segment's IVF centroids); each parent level is an
//! IVF over the level below's list centroids.
//!
//! **Canonical ordering invariant:** each level's clustering dictates the
//! storage order of the level below — every level's members are stored
//! grouped by their list, so a posting list is a contiguous row range and
//! the only per-level topology is a prefix-sum offsets array (the same
//! shape as the base `.centroids` slot `[1]`). [`IvfIndex::build`] /
//! [`IvfIndex::from_l0`] apply the reorder and return the member
//! permutation so callers can cascade it into external structures that
//! address the members (the merge's slot `[0]` rows and assignment ids).
//!
//! Build follows Quake's split:
//! - [`StackedIvfIndex::build`] / [`StackedIvfIndex::from_l0`] construct L0
//!   (cluster members, attach one parent when `nlist > branching_factor`).
//! - [`StackedIvfIndex::add_level`] walks to the topmost parent and deepens
//!   the stack by one level (Quake's deferred [`add_level`]).
//!
//! Search starts at the base level: the parent returns ranked seeds; the
//! level probes those lists and keeps a globally ordered member heap.
//!
//! Serialization ([`IvfIndex::serialize`]) writes the whole N-level
//! structure into one payload — depth is data, not schema: a level count,
//! then per level its offsets, then the per-level centroid rows as a
//! trailing blob (so read-back can pin topology and leave rows lazy).
//! Level 0's member *vectors* are never written; they live outside the
//! payload (slot `[0]`). Runtime knobs ([`IvfConfig`]) are never
//! persisted.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::io::{self, Write};
use std::marker::PhantomData;
use std::mem;

use common::{BinarySerializable, HasLen};
use itertools::Itertools;
use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, SuperKMeansConfig};

use crate::directory::FileSlice;
use crate::schema::Metric;
use crate::vector::{
    Candidate, FileSliceArena, Similarity, VectorArena, VectorElement, VectorStore,
};

/// Dense row into a level's centroid / member [`FlatStore`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClusterId(pub u32);

impl From<ClusterId> for usize {
    #[inline]
    fn from(id: ClusterId) -> usize {
        id.0 as usize
    }
}

impl From<usize> for ClusterId {
    #[inline]
    fn from(i: usize) -> Self {
        ClusterId(i as u32)
    }
}

impl From<ClusterId> for u32 {
    #[inline]
    fn from(id: ClusterId) -> u32 {
        id.0
    }
}

impl From<u32> for ClusterId {
    #[inline]
    fn from(i: u32) -> Self {
        ClusterId(i)
    }
}

/// Scan fraction for parent (centroid) levels. Upper levels rank the next
/// level's centroids, so they need a much larger budget than the base level
/// or end-to-end recall degrades — Quake scans an initial 25% of upper-level
/// partitions (with a 99% recall target) vs 1-10% at L0.
pub const PARENT_NPROBE_FRACTION: f32 = 0.1;

/// Runtime search configuration. Never persisted — supplied when the index
/// is built or opened.
#[derive(Clone, Debug)]
pub struct IvfConfig {
    pub nprobe_fraction: f32,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            nprobe_fraction: 0.02,
        }
    }
}

/// Build parameters for stacked IVF construction (internal).
#[derive(Clone, Debug)]
pub(crate) struct IvfBuildParams {
    /// Target fan-out: `nlist ≈ n / branching_factor` at each level.
    pub branching_factor: usize,
    /// Probe budget at the level being built (typically L0).
    pub config: IvfConfig,
    /// Probe budget for parent levels created by `build` / `add_level`.
    pub parent_config: IvfConfig,
}

impl Default for IvfBuildParams {
    fn default() -> Self {
        Self {
            branching_factor: 16,
            config: IvfConfig::default(),
            parent_config: IvfConfig {
                nprobe_fraction: PARENT_NPROBE_FRACTION,
            },
        }
    }
}

impl IvfBuildParams {
    fn new(branching_factor: usize, config: IvfConfig) -> Self {
        Self {
            branching_factor,
            config,
            ..Default::default()
        }
    }

    fn nlist_for(&self, n: usize) -> usize {
        assert!(self.branching_factor >= 2);
        (n / self.branching_factor).max(1).min(n)
    }

    fn should_attach_parent(&self, nlist: usize) -> bool {
        let parent_nlist = self.nlist_for(nlist);
        nlist > self.branching_factor && parent_nlist < nlist
    }
}

/// [`StackedIvfIndex::add_level`] requires an existing parent (call after [`StackedIvfIndex::build`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddLevelError {
    NoParent,
}

impl fmt::Display for AddLevelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddLevelError::NoParent => {
                f.write_str("no parent index — build this level before calling add_level")
            }
        }
    }
}

impl std::error::Error for AddLevelError {}

/// Contiguous row-major `f32` vectors addressed by a newtyped id.
#[derive(Clone, Debug)]
pub struct FlatStore<I> {
    data: Vec<f32>,
    dim: usize,
    _id: PhantomData<I>,
}

impl FlatStore<ClusterId> {
    /// Row-major `n × dim` matrix. Panics if `data.len()` is not a multiple of `dim`.
    pub fn new(data: Vec<f32>, dim: usize) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(data.len() % dim, 0, "data length must be a multiple of dim");
        FlatStore {
            data,
            dim,
            _id: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl VectorStore for FlatStore<ClusterId> {
    type Id = ClusterId;
    type Elem = f32;

    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn num_vectors(&self, dim: usize) -> usize {
        debug_assert_eq!(dim, self.dim);
        self.len()
    }

    #[inline]
    fn similarity(&self, id: Self::Id, query: &[Self::Elem], metric: Metric) -> Similarity {
        let i: usize = id.into();
        let d = self.dim;
        metric.similarity(query, &self.data[i * d..(i + 1) * d])
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

/// One IVF level; optional parent is IVF over this level's centroids.
///
/// Members are stored in cluster-sorted order (the canonical ordering
/// invariant), so list `j` owns exactly the contiguous member rows
/// `offsets[j]..offsets[j + 1]`.
pub struct IvfIndex<C: VectorStore, M: VectorStore> {
    pub config: IvfConfig,
    pub parent: Option<Box<IvfIndex<C, C>>>,
    /// Posting-list boundaries: prefix sums, `nlist + 1` entries, ending at
    /// the member count. The only per-level topology.
    pub offsets: Vec<u64>,
    pub centroids: C,
    pub members: M,
}

/// In-memory Quake-shaped stacked IVF (owned stores).
pub type StackedIvfIndex = IvfIndex<FlatStore<ClusterId>, FlatStore<ClusterId>>;

/// The persisted form: every vector row stays behind a [`FileSliceArena`]
/// and is fetched lazily, per row; only the per-level offsets are pinned.
pub type PersistedStackedIvf = IvfIndex<SliceStore, SliceStore>;

/// [`FileSlice`]-backed vector store: `dim`-strided `f32` rows fetched with
/// one ranged read each through a [`FileSliceArena`].
pub struct SliceStore {
    arena: FileSliceArena<f32>,
    dim: usize,
}

impl SliceStore {
    /// Wraps `rows` (row-major `n × dim` little-endian `f32`s).
    pub fn new(rows: FileSlice, dim: usize) -> Self {
        SliceStore {
            arena: FileSliceArena::new(rows),
            dim,
        }
    }
}

impl VectorStore for SliceStore {
    type Id = ClusterId;
    type Elem = f32;

    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn num_vectors(&self, dim: usize) -> usize {
        self.arena.num_vectors(dim)
    }

    #[inline]
    fn similarity(&self, id: Self::Id, query: &[Self::Elem], metric: Metric) -> Similarity {
        self.arena.similarity(metric, self.dim, id.0, query)
    }
}

/// Stable counting sort of `assignments` into list order. Returns
/// `(perm, offsets)` where `perm[old_index] = new_row` and `offsets` is the
/// `u64[nlist + 1]` prefix sum over list sizes.
fn cluster_sort(assignments: &[u32], nlist: usize) -> (Vec<u32>, Vec<u64>) {
    let mut offsets = vec![0u64; nlist + 1];
    for &list in assignments {
        assert!(
            (list as usize) < nlist,
            "assignment {list} out of range for {nlist} lists"
        );
        offsets[list as usize + 1] += 1;
    }
    for j in 0..nlist {
        offsets[j + 1] += offsets[j];
    }
    let mut next = offsets.clone();
    let mut perm = vec![0u32; assignments.len()];
    for (old, &list) in assignments.iter().enumerate() {
        perm[old] = next[list as usize] as u32;
        next[list as usize] += 1;
    }
    (perm, offsets)
}

/// Applies `perm` (old row → new row) to row-major `dim`-strided `data`.
fn permute_rows(data: &[f32], dim: usize, perm: &[u32]) -> Vec<f32> {
    debug_assert_eq!(data.len(), perm.len() * dim);
    let mut out = vec![0.0f32; data.len()];
    for (old, &new) in perm.iter().enumerate() {
        let new = new as usize;
        out[new * dim..(new + 1) * dim].copy_from_slice(&data[old * dim..(old + 1) * dim]);
    }
    out
}

impl IvfIndex<FlatStore<ClusterId>, FlatStore<ClusterId>> {
    fn empty(config: IvfConfig, dim: usize) -> Self {
        Self {
            config,
            parent: None,
            offsets: vec![0],
            centroids: FlatStore::new(Vec::new(), dim),
            members: FlatStore::new(Vec::new(), dim),
        }
    }

    /// Cluster `members`, apply canonical ordering, and populate this level.
    /// When `attach_parent` is true and `nlist > branching_factor`, attaches
    /// one parent built over this level's centroids.
    fn build_level_canonical<Cl: IvfLevelClusterer<Elem = f32>>(
        &mut self,
        members: Vec<f32>,
        mut centroids: Vec<f32>,
        mut assignments: Vec<u32>,
        dim: usize,
        clusterer: &Cl,
        params: &IvfBuildParams,
        attach_parent: bool,
    ) -> Vec<u32> {
        let nlist = centroids.len() / dim;
        let parent = if attach_parent && params.should_attach_parent(nlist) {
            Some(Self::build_parent_box(
                &mut centroids,
                &mut assignments,
                dim,
                clusterer,
                params,
            ))
        } else {
            None
        };
        let (member_perm, offsets) = cluster_sort(&assignments, nlist);
        let members = permute_rows(&members, dim, &member_perm);
        self.config = params.config.clone();
        self.parent = parent;
        self.offsets = offsets;
        self.centroids = FlatStore::new(centroids, dim);
        self.members = FlatStore::new(members, dim);
        member_perm
    }

    fn build_parent_box<Cl: IvfLevelClusterer<Elem = f32>>(
        centroids: &mut Vec<f32>,
        assignments: &mut Vec<u32>,
        dim: usize,
        clusterer: &Cl,
        params: &IvfBuildParams,
    ) -> Box<Self> {
        let nlist = centroids.len() / dim;
        let parent_nlist = params.nlist_for(nlist);
        let (p_centroids, p_assign) = clusterer.cluster(centroids, nlist, dim, parent_nlist);
        let p_nlist = p_centroids.len() / dim;
        let (l0_perm, p_offsets) = cluster_sort(&p_assign, p_nlist);
        *centroids = permute_rows(centroids, dim, &l0_perm);
        for list in assignments.iter_mut() {
            *list = l0_perm[*list as usize];
        }
        Box::new(IvfIndex {
            config: params.parent_config.clone(),
            parent: None,
            offsets: p_offsets,
            centroids: FlatStore::new(p_centroids, dim),
            members: FlatStore::new(centroids.clone(), dim),
        })
    }

    /// Extend the stack by one level at the top (Quake [`add_level`]).
    pub fn add_level<Cl: IvfLevelClusterer<Elem = f32>>(
        &mut self,
        clusterer: &Cl,
        branching_factor: usize,
        config: IvfConfig,
    ) -> Result<(), AddLevelError> {
        let params = IvfBuildParams::new(branching_factor, config);
        let parent = self.parent.as_mut().ok_or(AddLevelError::NoParent)?;
        if parent.parent.is_some() {
            return parent.add_level(clusterer, branching_factor, params.config.clone());
        }
        let dim = parent.centroids.dim();
        let centroids = parent.centroids.as_slice().to_vec();
        let n = centroids.len() / dim;
        assert!(n > 0, "parent centroids must be non-empty");
        let nlist = params.nlist_for(n);
        let (p_centroids, p_assign) = clusterer.cluster(&centroids, n, dim, nlist);
        parent.build_level_canonical(
            centroids,
            p_centroids,
            p_assign,
            dim,
            clusterer,
            &params,
            true,
        );
        Ok(())
    }

    /// Build L0 from raw member vectors, then attach one parent when
    /// `nlist > branching_factor`.
    ///
    /// Returns the index plus the member permutation
    /// (`perm[old_member_index] = new_member_row`) produced by the canonical
    /// cluster-sort — callers owning an external copy of the members (the
    /// merge's slot `[0]` rows) must apply the same reorder.
    pub fn build<Cl: IvfLevelClusterer<Elem = f32>>(
        data: &[f32],
        n: usize,
        dim: usize,
        clusterer: &Cl,
        branching_factor: usize,
        config: IvfConfig,
    ) -> (Self, Vec<u32>) {
        assert!(n > 0 && dim > 0, "n and dim must be positive");
        assert!(
            branching_factor >= 2,
            "branching_factor must be >= 2, got {branching_factor}"
        );
        assert_eq!(data.len(), n * dim);
        let params = IvfBuildParams::new(branching_factor, config);
        let mut index = Self::empty(params.config.clone(), dim);
        let nlist = params.nlist_for(n);
        let (centroids, assignments) = clusterer.cluster(data, n, dim, nlist);
        let perm = index.build_level_canonical(
            data.to_vec(),
            centroids,
            assignments,
            dim,
            clusterer,
            &params,
            true,
        );
        (index, perm)
    }

    /// L0 from existing centroids + assignments (e.g. merge-time
    /// [`IvfClusterer::train`](super::training::IvfClusterer::train)). Attaches
    /// one parent over the centroids when `nlist > branching_factor`, then
    /// applies the canonical ordering top-down. Returns the index plus the
    /// member permutation.
    pub fn from_l0<Cl: IvfLevelClusterer<Elem = f32>>(
        members: Vec<f32>,
        centroids: Vec<f32>,
        assignments: Vec<u32>,
        clusterer: &Cl,
        branching_factor: usize,
        config: IvfConfig,
    ) -> (Self, Vec<u32>) {
        let params = IvfBuildParams::new(branching_factor, config);
        assert!(
            params.branching_factor >= 2,
            "branching_factor must be >= 2, got {}",
            params.branching_factor
        );
        assert!(!centroids.is_empty(), "L0 must have centroids");
        let dim = {
            let n = assignments.len();
            assert!(n > 0, "assignments must be non-empty");
            assert_eq!(members.len() % n, 0, "members length must be n × dim");
            members.len() / n
        };
        assert_eq!(centroids.len() % dim, 0);
        assert_eq!(assignments.len(), members.len() / dim);

        let mut index = Self::empty(params.config.clone(), dim);
        let perm = index.build_level_canonical(
            members,
            centroids,
            assignments,
            dim,
            clusterer,
            &params,
            true,
        );
        (index, perm)
    }

    /// Serialize the whole N-level structure (for `.centroids` slot `[2]`):
    ///
    /// ```text
    /// centroids_byte_offset: u64
    /// num_levels: u32
    /// per level, bottom-up: nlist u32, offsets u64[nlist + 1]
    /// per level, bottom-up: centroid rows f32[nlist · dim]
    /// ```
    ///
    /// Level 0's offsets index the external member store (slot `[0]` rows);
    /// level ℓ's offsets index level ℓ−1's centroid rows. Member vectors,
    /// `dim`, `metric`, and [`IvfConfig`] are never written.
    pub fn serialize<W: Write + ?Sized>(&self, out: &mut W) -> io::Result<()> {
        let mut levels: Vec<&StackedIvfIndex> = vec![self];
        while let Some(parent) = levels.last().and_then(|level| level.parent.as_deref()) {
            levels.push(parent);
        }

        let mut topology = Vec::new();
        u32::try_from(levels.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "level count exceeds u32"))?
            .serialize(&mut topology)?;
        for level in &levels {
            let nlist = level.centroids.len();
            if level.offsets.len() != nlist + 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "level has {} offsets for {nlist} lists",
                        level.offsets.len()
                    ),
                ));
            }
            u32::try_from(nlist)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "nlist exceeds u32"))?
                .serialize(&mut topology)?;
            for &offset in &level.offsets {
                offset.serialize(&mut topology)?;
            }
        }

        let centroids_byte_offset = (mem::size_of::<u64>() + topology.len()) as u64;
        centroids_byte_offset.serialize(out)?;
        out.write_all(&topology)?;
        for level in &levels {
            for &value in level.centroids.as_slice() {
                value.serialize(out)?;
            }
        }
        Ok(())
    }

    /// Deserialize a payload from [`Self::serialize`] into owned stores.
    ///
    /// `members` is level 0's member matrix (in production the slot `[0]`
    /// centroid rows, already in the canonical order the writer applied);
    /// higher levels' members are the level below's centroid rows from the
    /// payload itself. `config` applies to level 0; parent levels get
    /// [`PARENT_NPROBE_FRACTION`].
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

        // Assemble top-down so each level can own its parent; validate each
        // level's offsets end at the member count of the level below.
        let mut index: Option<Box<StackedIvfIndex>> = None;
        for level in (0..level_topology.len()).rev() {
            let (nlist, offsets) = &level_topology[level];
            let level_members = if level == 0 {
                members.clone()
            } else {
                level_centroids[level - 1].clone()
            };
            if *offsets.last().expect("nlist + 1 offsets") as usize != level_members.len() / dim {
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
                    IvfConfig {
                        nprobe_fraction: PARENT_NPROBE_FRACTION,
                    }
                },
                parent: index,
                offsets: offsets.clone(),
                centroids: FlatStore::new(level_centroids[level].clone(), dim),
                members: FlatStore::new(level_members, dim),
            }));
        }
        Ok(*index.expect("at least one level"))
    }
}

/// Parses `num_levels` then per level `(nlist, offsets u64[nlist + 1])`
/// from the topology bytes between `centroids_byte_offset` and the blobs,
/// validating prefix-sum shape and exact consumption.
fn parse_level_topology(mut topology: &[u8]) -> io::Result<Vec<(usize, Vec<u64>)>> {
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
        let mut offsets = Vec::with_capacity(nlist + 1);
        for _ in 0..nlist + 1 {
            offsets.push(u64::deserialize(&mut topology)?);
        }
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) || offsets[0] != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stacked IVF offsets are not a prefix sum",
            ));
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

impl PersistedStackedIvf {
    /// Opens a payload written by [`StackedIvfIndex::serialize`]
    /// (`.centroids` slot `[2]`). Pins only the per-level offsets; centroid
    /// rows and the level-0 members (`member_rows` — the slot `[0]` rows
    /// past the count words, already in the writer's canonical order) stay
    /// behind [`FileSliceArena`]s and are fetched per row at search time.
    /// `config` is caller-supplied runtime configuration, never read from
    /// the file; parent levels get [`PARENT_NPROBE_FRACTION`].
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

        // Per-level centroid sub-slices, bottom-up in payload order.
        let mut centroid_slices = Vec::with_capacity(level_topology.len());
        let mut start = centroids_byte_offset;
        for (nlist, _) in &level_topology {
            let end = start + nlist * row_bytes;
            centroid_slices.push(slot.slice(start..end));
            start = end;
        }

        // Assemble top-down so each level can own its parent; validate each
        // level's offsets end at the member count of the level below.
        let mut index: Option<Box<PersistedStackedIvf>> = None;
        for level in (0..level_topology.len()).rev() {
            let (_, offsets) = &level_topology[level];
            let (level_members, level_member_count) = if level == 0 {
                (member_rows.clone(), num_members)
            } else {
                (centroid_slices[level - 1].clone(), level_topology[level - 1].0)
            };
            if *offsets.last().expect("nlist + 1 offsets") as usize != level_member_count {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("level {level} offsets do not cover its members"),
                ));
            }
            index = Some(Box::new(IvfIndex {
                config: if level == 0 {
                    config.clone()
                } else {
                    IvfConfig {
                        nprobe_fraction: PARENT_NPROBE_FRACTION,
                    }
                },
                parent: index,
                offsets: offsets.clone(),
                centroids: SliceStore::new(centroid_slices[level].clone(), dim),
                members: SliceStore::new(level_members, dim),
            }));
        }
        Ok(*index.expect("at least one level"))
    }
}

impl<C, M> IvfIndex<C, M>
where
    C: VectorStore,
    M: VectorStore<Elem = C::Elem>,
    C::Id: Copy + Ord + Into<usize> + From<usize>,
    M::Id: Copy + Ord + From<usize>,
{
    /// Number of inverted lists at this level.
    pub fn nlist(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// How many parent links above this level (1 = this level only).
    pub fn depth(&self) -> usize {
        1 + self.parent.as_ref().map(|p| p.depth()).unwrap_or(0)
    }

    /// Lists to probe at this level: `nprobe_fraction` of `nlist`. The root
    /// still scores all of its own centroids to rank them, but only probes
    /// this many lists.
    pub fn n_probe(&self) -> usize {
        let nlist = self.nlist();
        let n = ((nlist as f32 * self.config.nprobe_fraction).ceil().max(1.0)) as usize;
        n.min(nlist).max(1)
    }

    /// Ranked members from probed partitions (full bag, globally ordered).
    pub fn search(
        &self,
        query: &[C::Elem],
        k: usize,
        recall: f32,
        metric: Metric,
    ) -> Vec<Candidate<M::Id>> {
        let mut frontier: BinaryHeap<Candidate<C::Id>> = BinaryHeap::new();
        let mut result: BinaryHeap<Reverse<Candidate<M::Id>>> = BinaryHeap::with_capacity(k);

        let n_probe = self.n_probe();

        if let Some(parent) = &self.parent {
            frontier.extend(parent.search(query, n_probe, recall, metric));
        } else {
            for i in 0..self.nlist() {
                let cluster_id = C::Id::from(i);
                frontier.push(Candidate {
                    sim: self.centroids.similarity(cluster_id, query, metric),
                    node: cluster_id,
                });
            }
        }

        let cumulative_recall = 0.0;
        let mut probe_count = 0;
        while let Some(candidate) = frontier.pop() {
            let cluster: usize = candidate.node.into();
            let start = self.offsets[cluster] as usize;
            let end = self.offsets[cluster + 1] as usize;
            for row in start..end {
                let member = M::Id::from(row);
                let result_candidate = Candidate {
                    sim: self.members.similarity(member, query, metric),
                    node: member,
                };
                result.push(Reverse(result_candidate));
            }

            probe_count += 1;
            if cumulative_recall >= recall || probe_count >= n_probe {
                break;
            }
        }

        result
            .into_iter()
            .map(|Reverse(c)| c)
            .sorted()
            .rev()
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

    #[test]
    fn build_is_exactly_two_levels() {
        let dim = 2;
        let n = 16;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (index, member_perm) =
            StackedIvfIndex::build(&data, n, dim, &clusterer, 4, IvfConfig::default());
        assert!(index.nlist() > 1);
        assert_eq!(index.members.len(), n);
        assert_eq!(member_perm.len(), n);
        let parent = index.parent.as_ref().expect("L1 parent");
        assert!(parent.parent.is_none());
        assert_eq!(parent.members.len(), index.nlist());
        assert_eq!(index.depth(), 2);

        let query = [1.0f32, 0.0];
        let hits = index.search(&query, 4, 1.0, Metric::L2);
        assert!(!hits.is_empty());
        assert!(
            hits.len() < n,
            "nprobe search should not score every L0 member"
        );
    }

    /// The canonical ordering invariant: every level's offsets are a
    /// prefix sum covering the level below, the member permutation is a
    /// bijection, and members landed in the list their (reordered) row
    /// range says they did.
    #[test]
    fn build_canonicalizes_cluster_order() {
        let dim = 2;
        let n = 32;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (index, member_perm) =
            StackedIvfIndex::build(&data, n, dim, &clusterer, 2, IvfConfig::default());

        // The permutation is a bijection over the members.
        let mut seen = vec![false; n];
        for &new in &member_perm {
            assert!(!seen[new as usize], "duplicate row in member permutation");
            seen[new as usize] = true;
        }

        // Offsets are prefix sums ending at the member count, per level.
        assert_eq!(*index.offsets.first().unwrap(), 0);
        assert_eq!(*index.offsets.last().unwrap() as usize, n);
        assert!(index.offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        let parent = index.parent.as_ref().expect("L1 parent");
        assert_eq!(*parent.offsets.last().unwrap() as usize, index.nlist());

        // Each stored member is nearest (among list centroids) to a
        // centroid consistent with its list: reordering must not scramble
        // rows across lists — verify via reconstructed original rows.
        for old in 0..n {
            let new = member_perm[old] as usize;
            let stored = &index.members.as_slice()[new * dim..(new + 1) * dim];
            let original = &data[old * dim..(old + 1) * dim];
            assert_eq!(stored, original, "row content must survive the reorder");
        }
    }

    /// Serialize → deserialize round trip: identical topology, centroids,
    /// and search results.
    #[test]
    fn serialize_round_trips() {
        let dim = 2;
        let n = 32;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (index, _perm) =
            StackedIvfIndex::build(&data, n, dim, &clusterer, 2, IvfConfig::default());

        let mut bytes = Vec::new();
        index.serialize(&mut bytes).unwrap();
        let decoded = StackedIvfIndex::deserialize_owned(
            &bytes,
            index.members.as_slice().to_vec(),
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
    fn add_level_extends_stack_at_the_top() {
        let dim = 2;
        let n = 64;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (mut index, _) =
            StackedIvfIndex::build(&data, n, dim, &clusterer, 2, IvfConfig::default());
        assert_eq!(index.depth(), 2);

        index.add_level(&clusterer, 2, IvfConfig::default())
            .expect("L1 exists");
        assert_eq!(index.depth(), 3);
        let l1 = index.parent.as_ref().expect("L1");
        let l2 = l1.parent.as_ref().expect("L2");
        assert!(l2.parent.is_none());
        assert_eq!(l2.members.len(), l1.nlist());
    }

    #[test]
    fn add_level_errors_without_parent() {
        let dim = 2;
        let mut index = StackedIvfIndex::empty(IvfConfig::default(), dim);
        let clusterer = SuperKMeansLevelClusterer::default();
        assert_eq!(
            index.add_level(&clusterer, 16, IvfConfig::default()),
            Err(AddLevelError::NoParent)
        );
    }

    /// The FileSlice-lazy open scores identically to the owned index:
    /// same topology, same search output, every row fetched through the
    /// arena instead of a materialized copy.
    #[test]
    fn slice_backed_open_matches_owned_search() {
        let dim = 2;
        let n = 32;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (index, _perm) =
            StackedIvfIndex::build(&data, n, dim, &clusterer, 2, IvfConfig::default());

        let mut slot = Vec::new();
        index.serialize(&mut slot).unwrap();
        let member_bytes: Vec<u8> = index
            .members
            .as_slice()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let opened = PersistedStackedIvf::open(
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
                assert_eq!(u32::from(g.node), u32::from(e.node));
                assert_eq!(g.sim, e.sim);
            }
        }
    }

    /// A truncated payload is refused, not misparsed.
    #[test]
    fn slice_backed_open_rejects_truncation() {
        let dim = 2;
        let n = 32;
        let data = line_data(n);
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let (index, _perm) =
            StackedIvfIndex::build(&data, n, dim, &clusterer, 2, IvfConfig::default());
        let mut slot = Vec::new();
        index.serialize(&mut slot).unwrap();
        let member_bytes: Vec<u8> = index
            .members
            .as_slice()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        let truncated = slot[..slot.len() - 4].to_vec();
        assert!(PersistedStackedIvf::open(
            FileSlice::from(truncated),
            FileSlice::from(member_bytes.clone()),
            dim,
            IvfConfig::default(),
        )
        .is_err());

        // Wrong member count: offsets no longer cover the members.
        let short_members = member_bytes[..member_bytes.len() - dim * 4].to_vec();
        assert!(PersistedStackedIvf::open(
            FileSlice::from(slot),
            FileSlice::from(short_members),
            dim,
            IvfConfig::default(),
        )
        .is_err());
    }
}
