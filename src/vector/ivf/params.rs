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
}

impl Default for AdaptiveProbeParams {
    fn default() -> Self {
        Self {
            max_probe_fraction: 0.01,
            min_probe_clusters: MIN_PROBE_CLUSTERS,
        }
    }
}

pub(crate) const MIN_PROBE_CLUSTERS: usize = 16;
