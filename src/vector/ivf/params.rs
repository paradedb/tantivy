/// Query-time configuration for IVF adaptive probing — the SPANN shape
/// (NeurIPS 2021), defaults aligned with the paper and SPTAG's shipped
/// config.
///
/// Stop condition, evaluated for the NEXT ranked centroid between
/// clusters — so the first cluster is always scanned: stop at the
/// absolute probe-count ceiling, OR once the `min_candidates` floor is
/// met AND the next centroid breaches the per-metric distance-ratio
/// gate (SPANN eq. 3). The ceiling is checked first — the
/// [`ProbeTermination`](crate::vector::ProbeTermination) attribution
/// contract.
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
    /// `min_candidates.max(4 * top_n)`, so a 0 default still gives a
    /// sane `4 × top_n` floor.
    pub min_candidates: usize,
    /// Absolute cluster ceiling, clamped to the segment's cluster count
    /// — segments with `C <= cap` scan exhaustively unless the gate
    /// stops earlier. SPANN Fig. 2: 99% of SIFT1M queries reach perfect
    /// recall@1 within 114 postings; 128 clears that. Default 128,
    /// PROVISIONAL.
    pub max_probe_count: usize,
}

impl Default for AdaptiveProbeParams {
    fn default() -> Self {
        Self {
            epsilon: 7.0,
            min_candidates: 0,
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
