//! The probe budget: work units, their pricing and the probe controller.
//!
//! One unit is one average cluster of work, `n_avg = N / C` across the index's IVF segments. An
//! open charges `x` and a scored row `(1 - x) / n_avg`, so an exhaustive, unfiltered, delete-free
//! scan charges exactly `C`. The budget is checked only between clusters.

use crate::vector::distance::Similarity;
use crate::vector::router::stacked::aps::RecallEstimator;
use crate::Score;

/// How the probe loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
pub enum ProbeTermination {
    /// The quantized-gap estimate reached the requested recall.
    QuantizedRecall,
    /// The work budget was spent.
    Ceiling,
    /// The router ran out of clusters before the budget was spent.
    #[default]
    Exhausted,
    /// The APS recall estimate of the covered clusters reached
    /// [`ProbeParams::recall_target`](crate::vector::ProbeParams::recall_target).
    RecallTarget,
}

// --- Work units ---

/// The open share `x = cost / (cost + n_avg)` for a cluster-open cost of `fixed_cost_rows` rows
/// and `n_avg` native docs per cluster.
///
/// Clamped to at most one half: a larger share means opens outweigh rows, which only clusters of
/// under two rows produce.
pub(crate) fn open_share(fixed_cost_rows: f64, n_avg: f64) -> f64 {
    (fixed_cost_rows / (fixed_cost_rows + n_avg.max(0.0))).min(0.5)
}

/// Probe work in average-cluster units; budgets, prices and spends compose only with each other.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
pub(crate) struct WorkUnits(f64);

impl WorkUnits {
    /// No work.
    pub(crate) const ZERO: WorkUnits = WorkUnits(0.0);

    /// Wraps an amount in work units.
    #[inline]
    pub(crate) fn new(units: f64) -> WorkUnits {
        WorkUnits(units)
    }

    /// Returns the binary64 amount.
    #[inline]
    pub(crate) fn get(self) -> f64 {
        self.0
    }

    /// Returns the amount narrowed to binary32, for the probe statistics.
    #[inline]
    pub(crate) fn to_f32(self) -> f32 {
        self.0 as f32
    }
}

impl std::ops::Add for WorkUnits {
    type Output = WorkUnits;
    #[inline]
    fn add(self, rhs: WorkUnits) -> WorkUnits {
        WorkUnits(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for WorkUnits {
    #[inline]
    fn add_assign(&mut self, rhs: WorkUnits) {
        self.0 += rhs.0;
    }
}

impl std::ops::Mul<f64> for WorkUnits {
    type Output = WorkUnits;
    /// Scales work by a count.
    #[inline]
    fn mul(self, rhs: f64) -> WorkUnits {
        WorkUnits(self.0 * rhs)
    }
}

/// A segment's work budget and what a cluster open and a scored row cost against it, from
/// [`ProbeParams::resolved_work_budget`](crate::vector::params::ProbeParams::resolved_work_budget).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WorkBudget {
    /// Work the segment may spend before the ceiling binds.
    pub(crate) total: WorkUnits,
    /// The per-index open share `x`.
    pub(crate) per_cluster: WorkUnits,
    /// `(1 - x) / n_avg`.
    pub(crate) per_row: WorkUnits,
}

/// Charges cluster opens and scored rows against a budget.
///
/// Only scored rows charge row work, and a cluster is never cut short, so the charge overshoots
/// the budget by at most the last cluster's.
pub(crate) struct WorkMeter {
    budget: WorkBudget,
    charged: WorkUnits,
}

impl WorkMeter {
    pub(crate) fn new(budget: WorkBudget) -> Self {
        Self {
            budget,
            charged: WorkUnits::ZERO,
        }
    }

    /// Charges one cluster open.
    pub(crate) fn open(&mut self) {
        self.charged += self.budget.per_cluster;
    }

    /// Charges `rows` scored rows.
    pub(crate) fn rows(&mut self, rows: usize) {
        self.charged += self.budget.per_row * rows as f64;
    }

    /// Whether the charges have reached the budget.
    pub(crate) fn exhausted(&self) -> bool {
        self.charged >= self.budget.total
    }

    /// The work charged so far.
    pub(crate) fn charged(&self) -> WorkUnits {
        self.charged
    }

    pub(crate) fn budget(&self) -> WorkBudget {
        self.budget
    }
}

// --- Controller ---

/// Tracks the APS recall estimate and decides, from the work meter, whether the next ranked
/// cluster opens.
pub(in crate::vector) struct ProbeController<'a> {
    enforce_budget: bool,
    termination: ProbeTermination,
    /// Clusters in the segment and clusters pulled from the ranking so far.
    clusters: usize,
    pulled: usize,
    /// APS over the ranked clusters; `None` when APS is off.
    estimator: Option<RecallEstimator<'a>>,
    recall_target: f32,
    recall_estimate: Option<f32>,
}

impl<'a> ProbeController<'a> {
    pub(in crate::vector) fn new(
        clusters: usize,
        estimator: Option<RecallEstimator<'a>>,
        recall_target: f32,
        enforce_budget: bool,
    ) -> Self {
        Self {
            enforce_budget,
            termination: ProbeTermination::Exhausted,
            clusters,
            pulled: 0,
            estimator,
            recall_target,
            recall_estimate: None,
        }
    }

    /// Whether the cluster just pulled from the ranking may open.
    ///
    /// Checked after the pull, so a stop proves another cluster existed.
    pub(in crate::vector) fn admit(&mut self, meter: &WorkMeter) -> bool {
        self.pulled += 1;
        if self
            .recall_estimate
            .is_some_and(|estimate| estimate >= self.recall_target)
        {
            self.termination = ProbeTermination::RecallTarget;
            return false;
        }
        if self.enforce_budget && meter.exhausted() {
            self.termination = ProbeTermination::Ceiling;
            return false;
        }
        true
    }

    /// Marks the admitted cluster covered: probed, skipped or without survivors.
    ///
    /// `kth` is the k-th result score after it, `None` while the heap fills.
    pub(in crate::vector) fn cover(&mut self, kth: Option<Score>) -> crate::Result<()> {
        if let Some(estimator) = self.estimator.as_mut() {
            if let Some(estimate) = estimator.cover_next(kth.map(Similarity::new))? {
                self.recall_estimate = Some(estimate);
            }
        }
        Ok(())
    }

    /// How the probe loop ended and what `meter` charged.
    pub(crate) fn finish(&self, meter: &WorkMeter) -> ProbeOutcome {
        // A stacked ranking sized to the budget can run out as the budget is spent, short of the
        // segment's clusters; that stop is the ceiling's.
        let termination = match self.termination {
            ProbeTermination::Exhausted
                if self.enforce_budget && self.pulled < self.clusters && meter.exhausted() =>
            {
                ProbeTermination::Ceiling
            }
            termination => termination,
        };
        ProbeOutcome {
            termination,
            work_charged: meter.charged().to_f32(),
            work_budget: meter.budget().total.to_f32(),
            recall_estimate: self.recall_estimate,
        }
    }
}

/// How a routed probe loop ended.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProbeOutcome {
    pub(crate) termination: ProbeTermination,
    /// Work units charged: opens at `x`, scored rows at `(1 - x)/n_avg`.
    pub(crate) work_charged: f32,
    /// The resolved budget the loop ran against, in work units.
    pub(crate) work_budget: f32,
    /// The APS recall estimate at the stop; `None` when APS was off or never armed.
    pub(crate) recall_estimate: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_probing_ignores_exhausted_work_meter() {
        let meter = WorkMeter::new(WorkBudget::default());
        assert!(meter.exhausted());
        let mut fixed = ProbeController::new(10, None, 1.0, true);
        assert!(!fixed.admit(&meter));
        assert_eq!(fixed.finish(&meter).termination, ProbeTermination::Ceiling);
        let mut adaptive = ProbeController::new(10, None, 1.0, false);
        assert!(adaptive.admit(&meter));
        assert_eq!(
            adaptive.finish(&meter).termination,
            ProbeTermination::Exhausted
        );
    }
}
