/// Query-time probe budget for IVF vector search.
///
/// Stop condition: the probe-budget ceiling, or stream exhaustion. There
/// is no distance-ratio knob and no second stop — the bounds gate skips
/// clusters it can prove are useless, which spends LESS than the ceiling
/// but never ends the scan.
///
/// The ceiling is measured in WORK UNITS, not raw clusters: 1 unit is
/// one average cluster of work, charged event-wise as the global loop
/// proceeds (an opening share per non-empty (cluster, segment) pair, a
/// per-row share per row actually read and scored - see the work-unit
/// model in `backend`). A selective filter therefore probes deeper into
/// the ranked list before the ceiling binds, since the rows it rejects
/// are never scored and never charged.
///
/// The budget is GLOBAL: one ceiling and one floor per query, resolved
/// against the whole index's capacity in `search::resolve_budget` — not
/// per segment, which is what used to inflate work linearly with the
/// segment count.
///
/// All defaults are provisional pending real-data benchmarking.
#[derive(Clone, Debug)]
pub struct AdaptiveProbeParams {
    /// Filter-effective work ceiling, as a FRACTION of the index's
    /// capacity. Default 0.01, PROVISIONAL.
    pub max_probe_fraction: f32,
    /// Lower bound on the resolved budget, in work units, applied before
    /// the capacity clamp. Keeps small indexes, where
    /// `max_probe_fraction` rounds down to a single cluster, probing
    /// more than that one cluster. Defaults to [`MIN_PROBE_CLUSTERS`].
    /// Denominated in work units (~ that many average clusters), NOT a
    /// probed-cluster count.
    pub min_probe_clusters: usize,
    /// Estimated recall target for stacked centroid routing; `1.0` disables APS.
    pub router_recall_target: f32,
}

impl Default for AdaptiveProbeParams {
    fn default() -> Self {
        Self {
            max_probe_fraction: 0.01,
            min_probe_clusters: MIN_PROBE_CLUSTERS,
            router_recall_target: DEFAULT_ROUTER_RECALL,
        }
    }
}

pub(crate) const MIN_PROBE_CLUSTERS: usize = 16;

pub const DEFAULT_ROUTER_RECALL: f32 = 0.9;
const ROUTER_K_SLACK: usize = 2;

impl AdaptiveProbeParams {
    pub(crate) fn router_k(&self, budget: f64, num_centroids: usize) -> usize {
        let base = budget.ceil().max(0.0) as usize;
        base.saturating_mul(ROUTER_K_SLACK)
            .max(self.min_probe_clusters)
            .min(num_centroids)
            .max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::AdaptiveProbeParams;

    #[test]
    fn router_k_tracks_budget_with_slack_and_caps() {
        let params = AdaptiveProbeParams::default();
        assert_eq!(params.router_k(20.0, 1000), 40);
        assert_eq!(params.router_k(20.4, 1000), 42);
        assert_eq!(params.router_k(2.0, 1000), super::MIN_PROBE_CLUSTERS);
        assert_eq!(params.router_k(20.0, 30), 30);
        assert_eq!(params.router_k(0.0, 0), 1);
    }
}
