//! Trinary-projection-tree partitioning for graph construction.

use std::ops::Range;

use super::graph::NodeId;

/// Tuning knobs for [`TPTree`].
#[derive(Clone, Copy, Debug)]
pub struct TPTreeConfig {
    /// Maximum points per leaf.
    pub leaf_size: usize,
    /// Samples used to fit each split.
    pub samples: usize,
    /// Dimensions used by each sparse projection.
    pub top_dims: usize,
    /// Candidate projections evaluated per split.
    pub iterations: usize,
}

impl Default for TPTreeConfig {
    fn default() -> Self {
        TPTreeConfig {
            leaf_size: 2000,
            samples: 1000,
            top_dims: 5,
            iterations: 100,
        }
    }
}

/// Trinary projection tree over strided vectors.
pub struct TPTree<'a> {
    vectors: &'a [f32],
    dim: usize,
    config: TPTreeConfig,
    rng: fastrand::Rng,
}

impl<'a> TPTree<'a> {
    /// Creates a partition tree over strided vectors.
    pub fn new(config: TPTreeConfig, dim: usize, vectors: &'a [f32], seed: u64) -> Self {
        debug_assert!(dim > 0, "dim must be non-zero");
        debug_assert_eq!(vectors.len() % dim, 0, "arena not a multiple of dim");
        TPTree {
            vectors,
            dim,
            config,
            rng: fastrand::Rng::with_seed(seed),
        }
    }

    /// Partitions node identifiers in place and returns leaf ranges.
    pub fn partition(&mut self, indices: &mut [NodeId]) -> Vec<Range<usize>> {
        let mut leaves = Vec::new();
        if !indices.is_empty() {
            self.subdivide(indices, 0, &mut leaves);
        }
        leaves
    }

    #[inline]
    fn coord(&self, node: NodeId, d: usize) -> f32 {
        self.vectors[node as usize * self.dim + d]
    }

    /// Recursively partitions one node range.
    fn subdivide(&mut self, indices: &mut [NodeId], offset: usize, leaves: &mut Vec<Range<usize>>) {
        if indices.len() <= self.config.leaf_size {
            leaves.push(offset..offset + indices.len());
            return;
        }
        let split = self.choose_split(indices);
        let (left, right) = indices.split_at_mut(split);
        self.subdivide(left, offset, leaves);
        self.subdivide(right, offset + split, leaves);
    }

    /// Partitions nodes around a fitted projection.
    fn choose_split(&mut self, indices: &mut [NodeId]) -> usize {
        let n = indices.len();
        let dim = self.dim;
        let sample = n.min(self.config.samples);
        let top_dims = self.config.top_dims.min(dim).max(1);

        let mut mean = vec![0.0f32; dim];
        for &node in &indices[..sample] {
            for (d, m) in mean.iter_mut().enumerate() {
                *m += self.coord(node, d);
            }
        }
        for m in &mut mean {
            *m /= sample as f32;
        }

        let mut variance = vec![0.0f32; dim];
        for &node in &indices[..sample] {
            for (d, var) in variance.iter_mut().enumerate() {
                let diff = self.coord(node, d) - mean[d];
                *var += diff * diff;
            }
        }

        let mut dims: Vec<usize> = (0..dim).collect();
        dims.sort_unstable_by(|&a, &b| variance[b].total_cmp(&variance[a]));
        dims.truncate(top_dims);

        let mut best_weight = vec![0.0f32; top_dims];
        best_weight[0] = 1.0;
        let mut best_mean = mean[dims[0]];
        let mut best_var = variance[dims[0]];

        let mut proj = vec![0.0f32; sample];
        let mut weight = vec![0.0f32; top_dims];
        for _ in 0..self.config.iterations {
            let mut norm = 0.0f32;
            for w in &mut weight {
                *w = self.rng.f32() * 2.0 - 1.0; // [-1, 1)
                norm += *w * *w;
            }
            let norm = norm.sqrt();
            if norm == 0.0 {
                continue;
            }
            for w in &mut weight {
                *w /= norm;
            }

            let mut m = 0.0f32;
            for (slot, &node) in proj.iter_mut().zip(&indices[..sample]) {
                let mut v = 0.0f32;
                for (k, &d) in dims.iter().enumerate() {
                    v += weight[k] * self.coord(node, d);
                }
                *slot = v;
                m += v;
            }
            m /= sample as f32;

            let mut var = 0.0f32;
            for &p in &proj {
                let diff = p - m;
                var += diff * diff;
            }
            if var > best_var {
                best_var = var;
                best_mean = m;
                best_weight.copy_from_slice(&weight);
            }
        }

        let mut i: isize = 0;
        let mut j: isize = n as isize - 1;
        while i <= j {
            let node = indices[i as usize];
            let mut val = 0.0f32;
            for (k, &d) in dims.iter().enumerate() {
                val += best_weight[k] * self.coord(node, d);
            }
            if val < best_mean {
                i += 1;
            } else {
                indices.swap(i as usize, j as usize);
                j -= 1;
            }
        }

        let split = i as usize;
        if split == 0 || split == n {
            n / 2
        } else {
            split
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena(pts: &[[f32; 3]]) -> Vec<f32> {
        pts.iter().flatten().copied().collect()
    }

    #[test]
    fn partition_separates_two_far_clusters() {
        let pts = [
            [1., 5., 1.],
            [2., 5., 0.],
            [0., 4., 2.],
            [1., 6., 1.],
            [9., 5., 10.],
            [10., 5., 9.],
            [8., 4., 11.],
            [9., 6., 10.],
        ];
        let v = arena(&pts);
        let config = TPTreeConfig {
            leaf_size: 4,
            samples: 8,
            top_dims: 2,
            iterations: 100,
        };
        let mut tpt = TPTree::new(config, 3, &v, 42);
        let mut indices: Vec<NodeId> = (0..8).collect();

        let leaves = tpt.partition(&mut indices);

        assert_eq!(leaves.len(), 2, "8 points / leaf_size 4 → one split");
        for leaf in leaves {
            let ids = &indices[leaf];
            let all_a = ids.iter().all(|&id| id < 4);
            let all_b = ids.iter().all(|&id| id >= 4);
            assert!(all_a || all_b, "leaf mixes clusters: {ids:?}");
        }
    }

    #[test]
    fn partition_terminates_on_identical_vectors() {
        let v = vec![0.0f32; 3 * 8];
        let config = TPTreeConfig {
            leaf_size: 2,
            samples: 8,
            top_dims: 2,
            iterations: 8,
        };
        let mut tpt = TPTree::new(config, 3, &v, 42);
        let mut indices: Vec<NodeId> = (0..8).collect();

        let leaves = tpt.partition(&mut indices);

        assert!(leaves.iter().all(|l| l.len() <= 2));
        assert_eq!(leaves.iter().map(|l| l.len()).sum::<usize>(), 8);
    }
}
