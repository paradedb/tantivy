/// Query-time probe budget for IVF vector search.
///
/// The probe loop visits clusters best-routed-first and stops at the
/// budget ceiling this struct resolves, or when the ranked stream is
/// exhausted. There is no distance-ratio knob and no second stop
/// condition: what a query may spend is the whole of this configuration.
/// A gate policy that proves a query can stop early spends LESS than the
/// ceiling; nothing raises it.
///
/// The ceiling is measured in filter-effective clusters, not raw
/// clusters: a probed cluster consumes budget on an affine map of its
/// filter pass rate - a small floor when all its rows are filtered (the
/// pre-pass scanned them), `1.0` when none are. A selective filter
/// therefore probes deeper into the ranked list before the ceiling binds.
/// The pricing lives in `ChargeModel`, not here: this struct resolves
/// what a segment may spend, never what an event costs.
#[derive(Clone, Debug)]
pub struct ProbeBudget {
    /// Filter-effective cluster ceiling, expressed as a FRACTION of the
    /// segment's cluster count and resolved per-segment: a segment with
    /// `num_clusters` clusters probes at most `ceil(max_probe_fraction *
    /// num_clusters)` of them (clamped to `[1, num_clusters]`). A fraction
    /// rather than an absolute count because cluster counts vary segment to
    /// segment — an absolute cap over-probes small segments (clamping to a
    /// full scan) and under-probes large ones. Each probed cluster still
    /// consumes budget equal to its filter pass rate (`(rows - filtered) /
    /// rows`), so an unfiltered query probes at most this fraction while a
    /// selective filter probes proportionally more. This is the single
    /// user-facing tuning knob for IVF probing. SPANN Fig. 2: 99% of
    /// SIFT1M queries reach perfect recall@1 within 114 postings, ~1% of a
    /// 1%-centroid-ratio index's clusters. Default 0.01, PROVISIONAL.
    pub max_probe_fraction: f32,
    /// Lower bound on the resolved probe ceiling, applied before the
    /// `num_clusters` clamp — see [`Self::resolved_probe_ceiling`]. Keeps
    /// small segments, where `max_probe_fraction` rounds down to a handful
    /// of clusters, probing enough to fill a top-N heap. Defaults to
    /// [`MIN_PROBE_CLUSTERS`].
    pub min_probe_clusters: usize,
}

impl Default for ProbeBudget {
    fn default() -> Self {
        Self {
            max_probe_fraction: 0.01,
            min_probe_clusters: MIN_PROBE_CLUSTERS,
        }
    }
}

pub(crate) const MIN_PROBE_CLUSTERS: usize = 16;

impl ProbeBudget {
    /// The probe ceiling for a segment with `num_clusters` IVF clusters:
    /// `ceil(max_probe_fraction * num_clusters)`, lifted to at least
    /// `min_probe_clusters` then clamped to `num_clusters` (so a segment
    /// with fewer clusters than the floor just scans them all). A
    /// non-positive fraction is a configuration error, not "no probing".
    pub(crate) fn resolved_probe_ceiling(&self, num_clusters: usize) -> crate::Result<usize> {
        if !(self.max_probe_fraction > 0.0) {
            return Err(crate::TantivyError::InvalidArgument(
                "max_probe_fraction must be greater than 0".to_string(),
            ));
        }
        let ceiling = (self.max_probe_fraction * num_clusters as f32).ceil() as usize;
        Ok(ceiling.max(self.min_probe_clusters).min(num_clusters))
    }
}

#[cfg(test)]
mod tests {
    use super::ProbeBudget;

    #[test]
    fn probe_ceiling_resolves_against_cluster_count() -> crate::Result<()> {
        let params = ProbeBudget {
            max_probe_fraction: 0.25,
            ..Default::default()
        };
        // A quarter of the clusters, rounded up — well above the floor.
        assert_eq!(params.resolved_probe_ceiling(1000)?, 250);
        // A fraction resolving below MIN_PROBE_CLUSTERS is lifted to it...
        assert_eq!(
            params.resolved_probe_ceiling(40)?,
            super::MIN_PROBE_CLUSTERS
        );
        // ...but the floor never exceeds the cluster count.
        assert_eq!(params.resolved_probe_ceiling(2)?, 2);
        // A fraction above 1.0 clamps to the cluster count — small segments
        // scan exhaustively.
        let all = ProbeBudget {
            max_probe_fraction: 2.0,
            ..Default::default()
        };
        assert_eq!(all.resolved_probe_ceiling(10)?, 10);
        Ok(())
    }

    #[test]
    fn non_positive_probe_fraction_errors() {
        for fraction in [0.0, -1.0] {
            let params = ProbeBudget {
                max_probe_fraction: fraction,
                ..Default::default()
            };
            assert!(params.resolved_probe_ceiling(9).is_err());
        }
    }
}
