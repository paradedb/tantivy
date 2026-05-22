/// Query-time configuration for IVF adaptive probing.
///
/// SPANN-style probe loop: visit centroids in descending similarity order
/// and stop when (a) the current centroid score has fallen below a
/// geometric widening of the best (`epsilon`), AND (b) enough survivors
/// have been scored (`min_candidates`), AND (c) enough clusters have
/// been visited (`min_probe_fanout`). `max_probe_fanout` is the hard
/// latency ceiling — always wins.
///
/// All four knobs are provisional defaults pending real-data benchmarking.
#[derive(Clone, Debug)]
pub struct AdaptiveProbeParams {
    /// Geometric widening factor used in the per-metric stopping
    /// threshold. Larger ⇒ probe further past the best centroid before
    /// the threshold gate trips. Combined with the other knobs via the
    /// AND-of-three stop condition.
    pub epsilon: f32,
    /// Absolute survivor floor. The call site widens this to
    /// `min_candidates.max(4 * top_n)`, so a 0 default still gives a
    /// sane `4 × top_n` floor.
    pub min_candidates: usize,
    /// Probe fanout floor as a fraction of this segment's IVF clusters.
    /// For example, `0.10` keeps probing until at least 10% of the
    /// segment's clusters have been visited, rounded up. The backend
    /// still probes at least one cluster.
    pub min_probe_fanout: f32,
    /// Hard latency ceiling as a fraction of this segment's IVF clusters.
    /// For example, `0.10` caps probing at 10% of the segment's clusters,
    /// rounded up. `1.0` means no cap beyond the segment's cluster count.
    pub max_probe_fanout: f32,
}

impl Default for AdaptiveProbeParams {
    fn default() -> Self {
        // PROVISIONAL — placeholders pending real-data benchmarking.
        // Conservative direction: epsilon wide enough to admit a
        // reasonable probe radius, candidate floor resolved at the
        // call site, no fanout ceiling.
        Self {
            epsilon: 0.3,
            min_candidates: 0,
            min_probe_fanout: 0.0,
            max_probe_fanout: 1.0,
        }
    }
}

impl AdaptiveProbeParams {
    pub(crate) fn resolved_probe_counts(
        &self,
        num_clusters: usize,
    ) -> crate::Result<(usize, usize)> {
        if num_clusters == 0 {
            return Ok((0, 0));
        }
        let min_probe_count =
            fanout_to_probe_count("min_probe_fanout", self.min_probe_fanout, num_clusters)?.max(1);
        let max_probe_count =
            fanout_to_probe_count("max_probe_fanout", self.max_probe_fanout, num_clusters)?;
        if max_probe_count == 0 {
            return Err(crate::TantivyError::InvalidArgument(
                "max_probe_fanout must be greater than 0".to_string(),
            ));
        }
        if self.min_probe_fanout > self.max_probe_fanout {
            return Err(crate::TantivyError::InvalidArgument(format!(
                "min_probe_fanout ({}) must be less than or equal to max_probe_fanout ({})",
                self.min_probe_fanout, self.max_probe_fanout
            )));
        }
        Ok((
            min_probe_count.min(num_clusters),
            max_probe_count.min(num_clusters),
        ))
    }
}

fn fanout_to_probe_count(name: &str, fanout: f32, num_clusters: usize) -> crate::Result<usize> {
    if !fanout.is_finite() {
        return Err(crate::TantivyError::InvalidArgument(format!(
            "{name} must be finite"
        )));
    }
    if !(0.0..=1.0).contains(&fanout) {
        return Err(crate::TantivyError::InvalidArgument(format!(
            "{name} must be between 0 and 1 inclusive, got {fanout}"
        )));
    }
    if fanout == 0.0 {
        return Ok(0);
    }
    if fanout == 1.0 {
        return Ok(num_clusters);
    }
    Ok(((fanout as f64) * (num_clusters as f64)).ceil() as usize)
}

#[cfg(test)]
mod tests {
    use super::AdaptiveProbeParams;

    fn params(min_probe_fanout: f32, max_probe_fanout: f32) -> AdaptiveProbeParams {
        AdaptiveProbeParams {
            min_probe_fanout,
            max_probe_fanout,
            ..Default::default()
        }
    }

    #[test]
    fn probe_fanout_resolves_against_cluster_count() -> crate::Result<()> {
        assert_eq!(params(0.0, 1.0).resolved_probe_counts(9)?, (1, 9));
        assert_eq!(params(0.2, 0.5).resolved_probe_counts(9)?, (2, 5));
        assert_eq!(params(0.01, 0.01).resolved_probe_counts(9)?, (1, 1));
        Ok(())
    }

    #[test]
    fn invalid_probe_fanout_errors() {
        let invalid = [
            params(f32::NAN, 1.0),
            params(-0.1, 1.0),
            params(0.1, f32::INFINITY),
            params(0.1, 1.1),
            params(0.5, 0.0),
            params(0.6, 0.5),
        ];

        for params in invalid {
            assert!(params.resolved_probe_counts(9).is_err());
        }
    }
}
