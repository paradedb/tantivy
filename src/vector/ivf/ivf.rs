//! Quake-shaped multi-level IVF prototype (in-memory).
//!
//! L0 members are the base vectors (~200k). L0 lists are the ~11k IVF
//! clusters. L1 is IVF over those L0 centroids (~100 lists).
//!
//! Build follows Quake's split:
//! - [`StackedIvfIndex::build`] / [`StackedIvfIndex::from_l0`] construct L0
//!   (cluster members, attach one parent when `nlist > branching_factor`).
//! - [`StackedIvfIndex::add_level`] walks to the topmost parent and deepens
//!   the stack by one level (Quake's deferred [`add_level`]).
//!
//! Search starts at L0: parent returns ranked seeds; L0 probes those lists
//! and keeps a globally ordered member heap.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::marker::PhantomData;

use itertools::Itertools;

use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, SuperKMeansConfig};

use crate::schema::Metric;
use crate::vector::{Candidate, Similarity, VectorElement, VectorStore};

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

/// One inverted list: member ids belonging to this cluster.
#[derive(Clone, Debug, Default)]
pub struct Cluster<I> {
    pub assignments: Vec<I>,
}

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
pub struct IvfIndex<C: VectorStore, M: VectorStore> {
    pub config: IvfConfig,
    pub parent: Option<Box<IvfIndex<C, C>>>,
    pub clusters: Vec<Cluster<M::Id>>,
    pub centroids: C,
    pub members: M,
}

/// In-memory Quake-shaped stacked IVF used by the recall prototype.
pub type StackedIvfIndex = IvfIndex<FlatStore<ClusterId>, FlatStore<ClusterId>>;

impl IvfIndex<FlatStore<ClusterId>, FlatStore<ClusterId>> {
    fn empty(config: IvfConfig, dim: usize) -> Self {
        Self {
            config,
            parent: None,
            clusters: Vec::new(),
            centroids: FlatStore::new(Vec::new(), dim),
            members: FlatStore::new(Vec::new(), dim),
        }
    }

    fn cluster_members<Cl: IvfLevelClusterer<Elem = f32>>(
        &mut self,
        members: Vec<f32>,
        dim: usize,
        clusterer: &Cl,
        params: &IvfBuildParams,
    ) {
        assert!(dim > 0, "dim must be positive");
        assert!(
            params.branching_factor >= 2,
            "branching_factor must be >= 2, got {}",
            params.branching_factor
        );
        assert_eq!(members.len() % dim, 0, "members length must be n × dim");
        let n = members.len() / dim;
        assert!(n > 0, "members must be non-empty");

        let nlist = params.nlist_for(n);
        let (centroids, assignments) = clusterer.cluster(&members, n, dim, nlist);
        self.populate_from_clustering(members, centroids, assignments, params.config.clone());
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
        parent.cluster_members(centroids, dim, clusterer, &params);
        parent.attach_parent_if_needed(clusterer, &params);
        Ok(())
    }

    /// Build L0 from raw member vectors, then attach one parent when
    /// `nlist > branching_factor`.
    pub fn build<Cl: IvfLevelClusterer<Elem = f32>>(
        data: &[f32],
        n: usize,
        dim: usize,
        clusterer: &Cl,
        branching_factor: usize,
        config: IvfConfig,
    ) -> Self {
        assert!(n > 0 && dim > 0, "n and dim must be positive");
        assert!(
            branching_factor >= 2,
            "branching_factor must be >= 2, got {branching_factor}"
        );
        assert_eq!(data.len(), n * dim);
        let params = IvfBuildParams::new(branching_factor, config);
        let mut index = Self::empty(params.config.clone(), dim);
        index.cluster_members(data.to_vec(), dim, clusterer, &params);
        index.attach_parent_if_needed(clusterer, &params);
        index
    }

    /// L0 from existing centroids + assignments (e.g. merge-time
    /// [`IvfClusterer::train`](super::training::IvfClusterer::train)). Attaches
    /// one parent over the centroids when `nlist > branching_factor`.
    pub fn from_l0<Cl: IvfLevelClusterer<Elem = f32>>(
        members: Vec<f32>,
        centroids: Vec<f32>,
        assignments: Vec<u32>,
        clusterer: &Cl,
        branching_factor: usize,
        config: IvfConfig,
    ) -> Self {
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
        index.populate_from_clustering(members, centroids, assignments, params.config.clone());
        index.attach_parent_if_needed(clusterer, &params);
        index
    }

    fn populate_from_clustering(
        &mut self,
        members: Vec<f32>,
        centroids: Vec<f32>,
        assignments: Vec<u32>,
        config: IvfConfig,
    ) {
        let dim = self.centroids.dim();
        assert_eq!(centroids.len() % dim, 0);
        let nlist = centroids.len() / dim;
        assert_eq!(assignments.len(), members.len() / dim);
        self.config = config;
        self.clusters = clusters_from_assignments(&assignments, nlist);
        self.centroids = FlatStore::new(centroids, dim);
        self.members = FlatStore::new(members, dim);
    }

    fn attach_parent_if_needed<Cl: IvfLevelClusterer<Elem = f32>>(
        &mut self,
        clusterer: &Cl,
        params: &IvfBuildParams,
    ) {
        let nlist = self.nlist();
        if !params.should_attach_parent(nlist) {
            return;
        }
        let dim = self.centroids.dim();
        let centroids = self.centroids.as_slice().to_vec();
        let mut parent = Self::empty(params.parent_config.clone(), dim);
        parent.cluster_members(centroids, dim, clusterer, params);
        self.parent = Some(Box::new(parent));
    }
}

fn clusters_from_assignments(assignments: &[u32], nlist: usize) -> Vec<Cluster<ClusterId>> {
    let mut clusters = (0..nlist)
        .map(|_| Cluster {
            assignments: Vec::new(),
        })
        .collect::<Vec<_>>();
    for (point_id, &list_id) in assignments.iter().enumerate() {
        let list_id = list_id as usize;
        assert!(
            list_id < nlist,
            "assignment {list_id} out of range for {nlist} lists"
        );
        clusters[list_id]
            .assignments
            .push(ClusterId::from(point_id));
    }
    clusters
}

impl<C, M> IvfIndex<C, M>
where
    C: VectorStore,
    M: VectorStore<Elem = C::Elem>,
    C::Id: Copy + Ord + Into<usize> + From<usize>,
    M::Id: Copy + Ord,
{
    /// Number of inverted lists at this level.
    pub fn nlist(&self) -> usize {
        self.clusters.len()
    }

    /// How many parent links above this level (1 = this level only).
    pub fn depth(&self) -> usize {
        1 + self.parent.as_ref().map(|p| p.depth()).unwrap_or(0)
    }

    /// Lists to probe at this level: `nprobe_fraction` of `nlist`. The root
    /// still scores all of its own centroids to rank them, but only probes
    /// this many lists.
    pub fn n_probe(&self) -> usize {
        let nlist = self.clusters.len();
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
            for i in 0..self.clusters.len() {
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
            let cluster = &self.clusters[candidate.node.into()];
            for member in &cluster.assignments {
                let result_candidate = Candidate {
                    sim: self.members.similarity(*member, query, metric),
                    node: *member,
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

    #[test]
    fn build_is_exactly_two_levels() {
        let dim = 2;
        let n = 16;
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            data.push(i as f32);
            data.push(0.0);
        }
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let index = StackedIvfIndex::build(&data, n, dim, &clusterer, 4, IvfConfig::default());
        assert!(index.nlist() > 1);
        assert_eq!(index.members.len(), n);
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

    #[test]
    fn add_level_extends_stack_at_the_top() {
        let dim = 2;
        let n = 64;
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            data.push(i as f32);
            data.push(0.0);
        }
        let clusterer = SuperKMeansLevelClusterer { iters_per_split: 3 };
        let mut index =
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
        let mut index = IvfIndex {
            config: IvfConfig::default(),
            parent: None,
            clusters: Vec::new(),
            centroids: FlatStore::new(Vec::new(), dim),
            members: FlatStore::new(Vec::new(), dim),
        };
        let clusterer = SuperKMeansLevelClusterer::default();
        assert_eq!(
            index.add_level(&clusterer, 16, IvfConfig::default()),
            Err(AddLevelError::NoParent)
        );
    }
}
