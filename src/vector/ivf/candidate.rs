//! A ranked routing candidate.

use std::cmp::Ordering;

use crate::vector::Similarity;

/// A `(similarity, node)` pair ordered by similarity, ties broken by node id, so a max-heap yields
/// the most similar first.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Candidate<N = u32> {
    /// Similarity to the query; higher is more similar.
    pub sim: Similarity,
    pub node: N,
}

impl<N: Eq> Eq for Candidate<N> {}

impl<N: Ord> Ord for Candidate<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim
            .cmp(&other.sim)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl<N: Ord> PartialOrd for Candidate<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
