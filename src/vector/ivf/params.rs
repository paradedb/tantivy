/// Global work-unit statistics, computed once at query init across the
/// index's IVF segments: `n_avg = N / C`, native docs over clusters.
/// When absent, budgeting falls back to the segment's own ratio.
#[derive(Clone, Copy, Debug)]
pub struct WorkModel {
    /// Native docs per cluster (as written; see `for_searcher` on
    /// deletions), index-wide over IVF segments.
    pub n_avg: f64,
}

impl WorkModel {
    /// Compute `n_avg` for `field` across `searcher`'s IVF segments.
    ///
    /// * `searcher` (`&Searcher`) — the searcher whose segments are scanned.
    /// * `field` (`Field`) — the vector field to meter.
    ///
    /// Returns (`crate::Result<Option<WorkModel>>`): the primed model, or
    /// `None` when the index holds no IVF segment (an all-flat index has no
    /// clusters to meter).
    pub fn for_searcher(
        searcher: &crate::Searcher,
        field: crate::schema::Field,
    ) -> crate::Result<Option<WorkModel>> {
        let (mut n_native, mut clusters) = (0u64, 0u64);
        for segment_reader in searcher.segment_readers() {
            let vec_reader = segment_reader.vector_index(field)?;
            if let Some(ivf) = vec_reader.index() {
                // Native docs as WRITTEN: dead rows charge nothing (alive pre-pass), so
                // deletes only ever cheapen a scan. As-written counts are stable and free;
                // merges purge deletions and shrink N and C together.
                n_native += ivf.num_docs() as u64;
                clusters += ivf.num_clusters() as u64;
            }
        }
        Ok((clusters > 0).then(|| WorkModel {
            n_avg: n_native as f64 / clusters as f64,
        }))
    }
}

/// Query-time probe budget for IVF vector search.
///
/// Stop condition: the probe-budget ceiling, or stream exhaustion. There
/// is no distance-ratio knob and no second stop — the bounds gate skips
/// clusters it can prove are useless, which spends LESS than the ceiling
/// but never ends the scan.
///
/// The ceiling is measured in WORK UNITS, not raw clusters: 1 unit is
/// one average cluster of work, charged event-wise as probing proceeds
/// (an opening share per probed cluster, a per-row share per row actually
/// read and scored - see the work-unit model in `backend`). A selective
/// filter therefore probes deeper into the ranked list before the ceiling
/// binds, since the rows it rejects are never scored and never charged.
///
/// All defaults are provisional pending real-data benchmarking.
#[derive(Clone, Debug)]
pub struct AdaptiveProbeParams {
    /// Filter-effective work ceiling, as a FRACTION of the segment's
    /// capacity and resolved per segment - a fraction tracks each
    /// segment's own cluster count where an absolute cap cannot; a
    /// selective filter probes proportionally deeper within it.
    /// Default 0.01, PROVISIONAL.
    pub max_probe_fraction: f32,
    /// Lower bound on the resolved budget, in work units, applied before
    /// the capacity clamp - see [`Self::resolved_work_budget`]. Keeps
    /// small segments, where `max_probe_fraction` rounds down to a single
    /// cluster, probing more than that one cluster. Defaults to
    /// [`MIN_PROBE_CLUSTERS`]. Denominated in work units (~ that many
    /// average clusters), NOT a probed-cluster count.
    pub min_probe_clusters: usize,
    /// The global work-unit statistics, primed at query init by callers
    /// holding a [`Searcher`](crate::Searcher) - see [`WorkModel`]. `None`
    /// falls back to per-segment normalization.
    pub work_model: Option<WorkModel>,
    /// Consecutive gate skips tolerated before the scan stops pulling
    /// the ranked stream (`ProbeTermination::SkipRun`). HEURISTIC: a
    /// farther cluster with a larger radius than any seen can still
    /// qualify, so small values trade recall for routing; `u32::MAX`
    /// disables the cap and restores exact-by-construction probing.
    pub max_consecutive_bounds_skips: u32,
}

impl Default for AdaptiveProbeParams {
    fn default() -> Self {
        Self {
            max_probe_fraction: 0.01,
            min_probe_clusters: MIN_PROBE_CLUSTERS,
            work_model: None,
            max_consecutive_bounds_skips: crate::vector::backend::MAX_CONSECUTIVE_BOUNDS_SKIPS,
        }
    }
}

pub(crate) const MIN_PROBE_CLUSTERS: usize = 16;

impl AdaptiveProbeParams {
    /// The segment's work-unit budget: `max_probe_fraction *
    /// segment_units`, floored at `min_probe_clusters` units and capped
    /// at `segment_units` (`n_avg` global when primed - see
    /// [`WorkModel`]).
    ///
    /// * `clusters_in_segment` (`usize`) — IVF cluster count in this segment.
    /// * `native_docs_in_segment` (`usize`) — distinct docs with a vector in this segment, as
    ///   written (replication does not inflate this).
    ///
    /// Returns (`crate::Result<(f64, f64, f64)>`): `(budget, n_avg, x)` —
    /// the segment's work-unit budget, the docs-per-cluster average, and
    /// the open share; the loop's pricing is built from all three. A
    /// non-positive `max_probe_fraction` is a configuration error, not
    /// "no probing".
    pub(crate) fn resolved_work_budget(
        &self,
        clusters_in_segment: usize,
        native_docs_in_segment: usize,
    ) -> crate::Result<(f64, f64, f64)> {
        if !(self.max_probe_fraction > 0.0) {
            return Err(crate::TantivyError::InvalidArgument(
                "max_probe_fraction must be greater than 0".to_string(),
            ));
        }
        let n_avg = match self.work_model {
            Some(model) => model.n_avg,
            None => native_docs_in_segment as f64 / clusters_in_segment.max(1) as f64,
        };
        // The open share is PER INDEX, derived from the measured
        // rows-per-open hardware ratio and this index's own granularity -
        // see `backend::open_share`. The unit identity (1 unit = x +
        // (1 - x); exhaustive scan = C units) holds for any x.
        let x = crate::vector::backend::open_share(n_avg);
        let segment_units = clusters_in_segment as f64 * x
            + (1.0 - x) * native_docs_in_segment as f64 / n_avg.max(f64::MIN_POSITIVE);
        let budget = (self.max_probe_fraction as f64 * segment_units)
            .max(self.min_probe_clusters as f64)
            .min(segment_units);
        Ok((budget, n_avg, x))
    }
}

#[cfg(test)]
mod tests {
    use super::{AdaptiveProbeParams, WorkModel};

    /// Capacity is the cluster count, exactly, whatever the granularity:
    /// `units_seg = C*x + (1 - x)*N/n_avg == C` for every x the open
    /// share can resolve to. That identity is what lets `f` keep meaning
    /// "this fraction of the index's work" at any granularity.
    #[test]
    fn unprimed_capacity_is_the_cluster_count() -> crate::Result<()> {
        let all = AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 1,
            ..Default::default()
        };
        for (clusters, docs) in [(1000usize, 20_000usize), (9, 20), (2, 2), (64, 6400)] {
            let (budget, _n_avg, _x) = all.resolved_work_budget(clusters, docs)?;
            assert!(
                (budget - clusters as f64).abs() <= 1e-9 * clusters as f64,
                "f=1 must buy exactly C units at C={clusters}, N={docs}: {budget}"
            );
        }
        Ok(())
    }

    /// The fraction scales the capacity, the floor lifts small segments,
    /// and the capacity caps both.
    #[test]
    fn budget_resolves_against_capacity() -> crate::Result<()> {
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.25,
            ..Default::default()
        };
        // A quarter of the capacity - well above the floor.
        let (budget, _, _) = params.resolved_work_budget(1000, 20_000)?;
        assert!((budget - 250.0).abs() < 1e-6, "{budget}");
        // A fraction resolving below MIN_PROBE_CLUSTERS is lifted to it...
        let (budget, _, _) = params.resolved_work_budget(40, 800)?;
        assert!(
            (budget - super::MIN_PROBE_CLUSTERS as f64).abs() < 1e-6,
            "{budget}"
        );
        // ...but the floor never exceeds the capacity.
        let (budget, _, _) = params.resolved_work_budget(2, 40)?;
        assert!((budget - 2.0).abs() < 1e-6, "{budget}");
        // A fraction above 1.0 clamps to the capacity - small segments
        // scan exhaustively.
        let all = AdaptiveProbeParams {
            max_probe_fraction: 2.0,
            ..Default::default()
        };
        let (budget, _, _) = all.resolved_work_budget(10, 200)?;
        assert!((budget - 10.0).abs() < 1e-6, "{budget}");
        Ok(())
    }

    /// A primed model allocates across segments by where the WORK is: two
    /// segments of equal cluster count but unequal size get budgets in
    /// proportion to their rows, and the two together sum to `f*C`.
    #[test]
    fn primed_model_allocates_by_work() -> crate::Result<()> {
        // 8 clusters each; one holds 800 docs, the other 200. Index-wide
        // n_avg = 1000/16 = 62.5.
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.5,
            min_probe_clusters: 1,
            work_model: Some(WorkModel { n_avg: 62.5 }),
            ..Default::default()
        };
        let (big, _, _) = params.resolved_work_budget(8, 800)?;
        let (small, _, _) = params.resolved_work_budget(8, 200)?;
        assert!(
            big > small,
            "the fuller segment gets more: {big} vs {small}"
        );
        assert!(
            (big + small - 0.5 * 16.0).abs() < 1e-6,
            "allocation sums to f*C index-wide: {big} + {small}"
        );
        Ok(())
    }

    #[test]
    fn non_positive_probe_fraction_errors() {
        for fraction in [0.0, -1.0] {
            let params = AdaptiveProbeParams {
                max_probe_fraction: fraction,
                ..Default::default()
            };
            assert!(params.resolved_work_budget(9, 180).is_err());
        }
    }
}
