//! The second cell of a replicated vector, which the SOAR paper calls a spill: among the
//! [`SOAR_CANDIDATES`] centroids nearest to `x` after its nearest, the one minimizing
//! `||x - c||^2 + lambda * <x - c, r1_hat>^2`, `r1_hat` being the unit residual of `x` to its
//! nearest centroid, so the two cells miss different queries. Clusterers compute the pair through
//! [`IvfClusterer::assign_pairs`](crate::vector::clustering::IvfClusterer::assign_pairs).

/// Policy constant, chosen empirically: the weight of the alignment term in the secondary's
/// score. Zero picks the second-nearest centroid; larger values push the secondary off the
/// primary residual's direction at the cost of distance.
pub(crate) const SOAR_LAMBDA: f32 = 2.0;

/// Policy constant: each secondary is chosen among this many nearest centroids.
pub(crate) const SOAR_CANDIDATES: usize = 64;
