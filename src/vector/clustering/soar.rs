//! Exhaustive assignment chooses the second cell (the SOAR spill) among the
//! [`SOAR_CANDIDATES`] centroids nearest to `x` after its nearest, the one minimizing
//! `||x - c||^2 + lambda * <x - c, r1_hat>^2`, `r1_hat` being the unit residual of `x` to its
//! nearest centroid, so the two cells miss different queries. Clusterers compute the pair through
//! [`IvfClusterer::assign_pairs`](crate::vector::clustering::IvfClusterer::assign_pairs).

/// Policy constant, chosen empirically: the weight of the alignment term in the secondary's
/// score. Zero picks the second-nearest centroid; larger values push the secondary off the
/// primary residual's direction at the cost of distance.
pub(crate) const SOAR_LAMBDA: f32 = 2.0;

/// Number of non-primary candidates scored for each secondary.
///
/// Exhaustive assignment includes `c2`, the nearest centroid after the primary. Its
/// score is at most `(1 + lambda) * ||x - c2||^2` since the alignment term never exceeds the
/// distance, while every centroid scores at least `||x - c2||^2`. The secondary chosen here
/// therefore scores within a factor `1 + lambda` of the best over every centroid.
pub(crate) const SOAR_CANDIDATES: usize = 64;
