/// Query-time configuration for IVF adaptive probing — the SPANN shape
/// (NeurIPS 2021), defaults aligned with the paper and SPTAG's shipped
/// config.
///
/// Stop condition, evaluated for the NEXT ranked centroid between
/// clusters — so the first cluster is always scanned: stop at the
/// probe-budget ceiling, OR once the `min_candidates` floor is
/// met AND the next centroid breaches the per-metric distance-ratio
/// gate (SPANN eq. 3). The ceiling is checked first — the
/// [`ProbeTermination`](crate::vector::ProbeTermination) attribution
/// contract.
///
/// The ceiling is measured in filter-effective clusters, not raw
/// clusters: a probed cluster consumes budget on an affine map of its
/// filter pass rate — `SKIPPED_CLUSTER_COST` (a small floor, since the
/// gate pre-pass still scans the cluster) when all rows are filtered,
/// `1.0` when none are, in between otherwise. A selective filter
/// therefore probes deeper into the ranked list before the ceiling
/// binds, since the clusters it skips over cost little (but not nothing).
///
/// All defaults are provisional pending real-data benchmarking.
#[derive(Clone, Debug)]
pub struct AdaptiveProbeParams {
    /// SPANN's query-aware dynamic pruning coefficient — a posting list
    /// is searched iff `Dist(q, c) <= (1 + epsilon) * Dist(q, c_closest)`,
    /// on the per-metric distance defined in the backend gate. The
    /// paper uses 0.6 (recall@1-tuned) to 7.0 (recall@10-tuned); SPTAG
    /// ships `MaxDistRatio = 8.0` = `(1 + 7.0)`. Default 7.0,
    /// PROVISIONAL pending our own benchmarks.
    pub epsilon: f32,
    /// Absolute survivor floor. The call site widens this to
    /// `min_candidates.max(top_n + overfetch_margin)`, so a 0 default
    /// still gives a sane `top_n + overfetch_margin` floor.
    pub min_candidates: usize,
    /// Additive over-fetch margin: the resolved survivor floor is
    /// `top_n + overfetch_margin`. Unlike a multiplicative `m × top_n`
    /// floor, an additive margin keeps the over-probe cushion a *fixed*
    /// number of clusters as `top_n` grows, so the `epsilon` needed for a
    /// target recall stays roughly constant across K instead of shrinking
    /// with it. Default 32, PROVISIONAL.
    pub overfetch_margin: usize,
    /// Filter-effective cluster ceiling, clamped to the segment's cluster
    /// count. Each probed cluster consumes budget equal to its filter
    /// pass rate (`(rows - filtered) / rows`), so an unfiltered query
    /// probes at most this many clusters while a selective filter probes
    /// proportionally more. SPANN Fig. 2: 99% of SIFT1M queries reach
    /// perfect recall@1 within 114 postings; 128 clears that. Default 128,
    /// PROVISIONAL.
    pub max_probe_count: usize,
}

impl Default for AdaptiveProbeParams {
    fn default() -> Self {
        Self {
            epsilon: 7.0,
            min_candidates: 0,
            overfetch_margin: 32,
            max_probe_count: 128,
        }
    }
}

impl AdaptiveProbeParams {
    /// The probe ceiling for a segment with `num_clusters` IVF
    /// clusters: `max_probe_count` clamped to the cluster count. Zero
    /// is a configuration error, not "no probing".
    pub(crate) fn resolved_probe_ceiling(&self, num_clusters: usize) -> crate::Result<usize> {
        if self.max_probe_count == 0 {
            return Err(crate::TantivyError::InvalidArgument(
                "max_probe_count must be greater than 0".to_string(),
            ));
        }
        Ok(self.max_probe_count.min(num_clusters))
    }
}

#[cfg(test)]
mod tests {
    use super::AdaptiveProbeParams;

    #[test]
    fn probe_ceiling_resolves_against_cluster_count() -> crate::Result<()> {
        let params = AdaptiveProbeParams {
            max_probe_count: 128,
            ..Default::default()
        };
        // Cap above the cluster count clamps — small segments scan
        // exhaustively.
        assert_eq!(params.resolved_probe_ceiling(10)?, 10);
        // Cap below the cluster count binds.
        assert_eq!(params.resolved_probe_ceiling(1000)?, 128);
        Ok(())
    }

    #[test]
    fn zero_probe_ceiling_errors() {
        let params = AdaptiveProbeParams {
            max_probe_count: 0,
            ..Default::default()
        };
        assert!(params.resolved_probe_ceiling(9).is_err());
    }
}
