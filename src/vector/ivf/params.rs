/// Global statistics of the work-unit model, computed ONCE at query init
/// across the index's IVF segments (flat segments excluded):
/// `n_avg = N / C`, native docs over clusters. One constant; no
/// per-segment term in the unit definition, no per-cluster metadata. When
/// absent (single-segment test drives, unprimed callers), budgeting falls
/// back to the segment's own `N_seg / C_seg` - under which a segment's
/// capacity reduces to exactly its cluster count (the normalization
/// identity, per segment), so charging stays event-wise and only
/// cross-segment allocation loses its index-wide view.
#[derive(Clone, Copy, Debug)]
pub struct WorkModel {
    /// Native docs per cluster (as written; see `for_searcher` on
    /// deletions), index-wide over IVF segments.
    pub n_avg: f64,
}

impl WorkModel {
    /// Compute `n_avg` for `field` across `searcher`'s IVF segments.
    /// `None` when the index holds no IVF segment (an all-flat index has
    /// no clusters to meter).
    pub fn for_searcher(
        searcher: &crate::Searcher,
        field: crate::schema::Field,
    ) -> crate::Result<Option<WorkModel>> {
        let (mut n_native, mut clusters) = (0u64, 0u64);
        for segment_reader in searcher.segment_readers() {
            let vec_reader = segment_reader.vector_index(field)?;
            if let Some(ivf) = vec_reader.index() {
                // Native docs as WRITTEN, not live-subtracted: deleted
                // docs still occupy posting rows and charge on first
                // touch, so counting them in capacity is what keeps the
                // normalization identity exact (an exhaustive scan
                // charges exactly C units). Merges purge deletions and
                // re-shrink both sides together.
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
/// The probe loop visits clusters best-routed-first and stops at the
/// budget ceiling this struct resolves, or when the ranked stream is
/// exhausted. There is no distance-ratio knob and no second stop
/// condition: what a query may spend is the whole of this configuration.
/// A gate policy that proves a query can stop early spends LESS than the
/// ceiling; nothing raises it.
///
/// The ceiling is measured in WORK UNITS, not raw clusters: 1 unit is
/// one average cluster of work, charged event-wise as probing proceeds
/// (an opening share per probed cluster, a per-row share per first-seen
/// row - see the work-unit model in `backend`). A selective filter
/// therefore probes deeper into the ranked list before the ceiling
/// binds, since the clusters it passes over stream few unseen rows.
#[derive(Clone, Debug)]
pub struct AdaptiveProbeParams {
    /// Filter-effective cluster ceiling, expressed as a FRACTION of the
    /// segment's cluster count and resolved per-segment: a segment with
    /// `num_clusters` clusters probes at most `ceil(max_probe_fraction *
    /// num_clusters)` of them (clamped to `[1, num_clusters]`). A fraction
    /// rather than an absolute count because cluster counts vary segment to
    /// segment — an absolute cap over-probes small segments (clamping to a
    /// full scan) and under-probes large ones. Each probed cluster still
    /// consumes budget equal to its filter pass rate (`(rows - filtered) /
    /// rows`), so an unfiltered query probes at most this fraction while a
    /// selective filter probes proportionally more. SPANN Fig. 2: 99% of
    /// SIFT1M queries reach perfect recall@1 within 114 postings, ~1% of a
    /// 1%-centroid-ratio index's clusters. Default 0.01, PROVISIONAL.
    pub max_probe_fraction: f32,
    /// Lower bound on the resolved budget, in work units, applied before
    /// the capacity clamp - see [`Self::resolved_work_budget`]. Keeps
    /// small segments, where `max_probe_fraction` rounds down to a single
    /// cluster, probing enough to fill the survivor floor. Defaults to
    /// [`MIN_PROBE_CLUSTERS`].
    pub min_probe_clusters: usize,
    /// The global work-unit statistics, primed at query init by callers
    /// holding a [`Searcher`](crate::Searcher) - see [`WorkModel`]. `None`
    /// falls back to per-segment normalization.
    pub work_model: Option<WorkModel>,
    /// Run the probe loop with NO gate policy - the gateless control arm
    /// for gate-vs-gateless comparisons on one binary. TEST AND BENCH
    /// ONLY: this field does not exist in a shipped build, and neither
    /// does the branch that reads it.
    #[cfg(any(test, feature = "bench-control"))]
    pub disable_gate: bool,
}

impl Default for AdaptiveProbeParams {
    fn default() -> Self {
        Self {
            max_probe_fraction: 0.01,
            min_probe_clusters: MIN_PROBE_CLUSTERS,
            work_model: None,
            #[cfg(any(test, feature = "bench-control"))]
            disable_gate: false,
        }
    }
}

pub(crate) const MIN_PROBE_CLUSTERS: usize = 16;

impl AdaptiveProbeParams {
    /// The work-unit budget for one segment: `f * units_seg` with
    /// `units_seg = C_seg*x + (1 - x)*N_seg/n_avg` (n_avg global when
    /// primed, the segment's own otherwise - see [`WorkModel`]). This is
    /// ALLOCATION, not unit definition: f of the index's work, spent
    /// where the work is; it sums to `f*C` index-wide and reduces to
    /// `f * C_seg` on homogeneous segments. Floored at
    /// `min_probe_clusters` units and capped at `units_seg`.
    ///
    /// Returns `(budget, n_avg, x)` - the loop's pricing is built from
    /// all three. A non-positive fraction is a configuration error, not
    /// "no probing".
    pub(crate) fn resolved_work_budget(
        &self,
        c_seg: usize,
        n_seg: usize,
    ) -> crate::Result<(f64, f64, f64)> {
        if !(self.max_probe_fraction > 0.0) {
            return Err(crate::TantivyError::InvalidArgument(
                "max_probe_fraction must be greater than 0".to_string(),
            ));
        }
        let n_avg = match self.work_model {
            Some(model) => model.n_avg,
            None => n_seg as f64 / c_seg.max(1) as f64,
        };
        // The open share is PER INDEX, derived from the measured
        // rows-per-open hardware ratio and this index's own granularity -
        // see `backend::open_share`. The unit identity (1 unit = x +
        // (1 - x); exhaustive scan = C units) holds for any x.
        let x = crate::vector::backend::open_share(n_avg);
        let units_seg = c_seg as f64 * x + (1.0 - x) * n_seg as f64 / n_avg.max(f64::MIN_POSITIVE);
        let budget = (self.max_probe_fraction as f64 * units_seg)
            .max(self.min_probe_clusters as f64)
            .min(units_seg);
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
        for (c_seg, n_seg) in [(1000usize, 20_000usize), (9, 20), (2, 2), (64, 6400)] {
            let (budget, _n_avg, _x) = all.resolved_work_budget(c_seg, n_seg)?;
            assert!(
                (budget - c_seg as f64).abs() <= 1e-9 * c_seg as f64,
                "f=1 must buy exactly C units at C={c_seg}, N={n_seg}: {budget}"
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
