//! SOAR secondary cells minimize distance plus the cumulative squared projection onto
//! the unit residuals of preceding assignments, among the vector's [`SOAR_CANDIDATES`] nearest
//! centroids. The nearest centroid remains primary.

pub(crate) const SOAR_REPLICAS: usize = 4;

/// Policy constant, chosen empirically: each secondary is chosen among this many nearest
/// centroids, which bounds the search for every secondary to `SOAR_CANDIDATES · d`
/// multiply-adds per vector instead of a pass over every centroid.
pub(crate) const SOAR_CANDIDATES: usize = 256;

// The shortlist holds the primary and every secondary.
const _: () = assert!(SOAR_CANDIDATES >= SOAR_REPLICAS);

/// Policy constant, chosen empirically: the weight of the alignment term in the secondary's
/// score. Zero picks the second-nearest centroid; larger values push the secondary off the
/// primary residual's direction at the cost of distance.
pub(crate) const SOAR_LAMBDA: f32 = 2.0;
