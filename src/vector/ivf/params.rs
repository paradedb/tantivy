/// Index-wide IVF work model.
#[derive(Clone, Copy, Debug)]
pub struct WorkModel {
    /// Native documents per cluster.
    pub n_avg: f64,
}

impl WorkModel {
    /// Computes the work model across a searcher's IVF segments.
    ///
    /// # Errors
    ///
    /// Returns an error when a segment's vector index cannot be opened.
    pub fn for_searcher(
        searcher: &crate::Searcher,
        field: crate::schema::Field,
    ) -> crate::Result<Option<WorkModel>> {
        let (mut n_native, mut clusters) = (0u64, 0u64);
        for segment_reader in searcher.segment_readers() {
            let vec_reader = segment_reader.vector_index(field)?;
            if let Some(ivf) = vec_reader.index() {
                n_native += ivf.num_docs() as u64;
                clusters += ivf.num_clusters() as u64;
            }
        }
        Ok((clusters > 0).then(|| WorkModel {
            n_avg: n_native as f64 / clusters as f64,
        }))
    }
}

/// Query-time IVF probe budget.
#[derive(Clone, Debug)]
pub struct AdaptiveProbeParams {
    /// Maximum fraction of segment work.
    pub max_probe_fraction: f32,
    /// Minimum work-unit budget.
    pub min_probe_clusters: usize,
    /// Optional index-wide work model.
    pub work_model: Option<WorkModel>,
}

impl Default for AdaptiveProbeParams {
    fn default() -> Self {
        Self {
            max_probe_fraction: 0.01,
            min_probe_clusters: MIN_PROBE_CLUSTERS,
            work_model: None,
        }
    }
}

/// Default minimum IVF work budget.
pub(crate) const MIN_PROBE_CLUSTERS: usize = 16;

impl AdaptiveProbeParams {
    /// Resolves the segment work budget, mean posting size, and open share.
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

    #[test]
    fn budget_resolves_against_capacity() -> crate::Result<()> {
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.25,
            ..Default::default()
        };
        let (budget, _, _) = params.resolved_work_budget(1000, 20_000)?;
        assert!((budget - 250.0).abs() < 1e-6, "{budget}");
        let (budget, _, _) = params.resolved_work_budget(40, 800)?;
        assert!(
            (budget - super::MIN_PROBE_CLUSTERS as f64).abs() < 1e-6,
            "{budget}"
        );
        let (budget, _, _) = params.resolved_work_budget(2, 40)?;
        assert!((budget - 2.0).abs() < 1e-6, "{budget}");
        let all = AdaptiveProbeParams {
            max_probe_fraction: 2.0,
            ..Default::default()
        };
        let (budget, _, _) = all.resolved_work_budget(10, 200)?;
        assert!((budget - 10.0).abs() < 1e-6, "{budget}");
        Ok(())
    }

    #[test]
    fn primed_model_allocates_by_work() -> crate::Result<()> {
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
