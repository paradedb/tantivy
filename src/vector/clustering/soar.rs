//! SOAR secondary cells minimize distance plus the cumulative squared projection onto
//! the unit residuals of preceding assignments. The nearest centroid remains primary.

pub(crate) const SOAR_REPLICAS: usize = 4;

/// Policy constant, chosen empirically: the weight of the alignment term in the secondary's
/// score. Zero picks the second-nearest centroid; larger values push the secondary off the
/// primary residual's direction at the cost of distance.
pub(crate) const SOAR_LAMBDA: f32 = 2.0;
