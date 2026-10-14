//! Ordering of candidates by lower endpoint.

use crate::vector::scan::columns::QuantizedCandidates;

#[inline]
pub(super) fn lower_endpoint_order(
    candidates: &QuantizedCandidates,
    a: usize,
    b: usize,
    kappa: f32,
) -> std::cmp::Ordering {
    candidates
        .estimate(b)
        .lower(candidates.sigmas[b], kappa)
        .0
        .total_cmp(&candidates.estimate(a).lower(candidates.sigmas[a], kappa).0)
        .then(candidates.rows[a].cmp(&candidates.rows[b]))
}
