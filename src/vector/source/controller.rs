//! The probe budget: work units, their pricing and the probe controller.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use crate::vector::ivf::RecallEstimator;
use crate::vector::stats::ProbeStats;
use crate::vector::Similarity;
use crate::Score;

/// How the probe loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
pub enum ProbeTermination {
    /// The work-unit probe budget was spent - the probe ceiling.
    Ceiling,
    /// The ranked centroids were exhausted before the ceiling bound. The
    /// bounds gate never terminates the scan - a skip is per-cluster and
    /// charges the open share; only the ceiling and the stream end it.
    /// The centroid stream was exhausted.
    #[default]
    Exhausted,
    /// The APS estimate of the covered clusters reached
    /// [`AdaptiveProbeParams::recall_target`](crate::vector::ivf::AdaptiveProbeParams::recall_target).
    RecallTarget,
}

/// THE WORK-UNIT MODEL
///
/// The probe budget meters WORK: 1 unit = one average cluster of work,
/// with `n_avg = N / C` global across the index's IVF segments. Charging
/// is event-wise:
///
/// | event                    | charge          |
/// |--------------------------|-----------------|
/// | open a cluster           | `x`             |
/// | scored row               | `(1 - x)/n_avg` |
///
/// Only pre-pass survivors charge row work: filter/alive-rejected rows
/// and deduped replica re-encounters charge nothing (their buffer I/O
/// may still be paid), so a doc charges one row-deduction index-wide.
///
/// NORMALIZATION IDENTITY: an exhaustive, unfiltered, delete-free scan
/// charges `C*x + (1 - x)*N/n_avg = exactly C` units, so the probe
/// fraction keeps its scale across cluster granularities.
///
/// BOUNDARY RULE: the budget is inspected only at cluster boundaries -
/// open iff `remaining > 0`, deduct as-you-go, never truncate mid-cluster
/// (posting order is not distance order, so a partial scan is random loss
/// on a paid open). Overshoot is bounded by the last cluster's charge. No
/// pre-open cost knowledge is needed or used.
///
/// The bounds gate rides on this accounting: a skipped cluster charges
/// the open share (invariant: free skips break the normalization
/// identity), spends no row work, and never terminates the scan - the
/// budget and stream exhaustion are the only stops.
///
/// FIXED_PROBE_COST_ROWS is the fixed component of a probe — the cluster
/// OPEN — denominated in rows of full work, fitted on the reference
/// fixture; `x = fixed_probe_cost_rows() / (fixed_probe_cost_rows() +
/// n_avg)` self-calibrates to the index's granularity. Defaults to this
/// fitted value; runtime-settable via [`set_fixed_probe_cost_rows`] for
/// testing/calibration only. Despite "probe" in the name it covers ONLY
/// the open - routing/search cost is NOT modeled; removed once search is
/// costed.
/// Fixed cluster-open cost in row-work units.
pub const DEFAULT_FIXED_PROBE_COST_ROWS: f64 = 1.64;

/// Current FIXED_PROBE_COST_ROWS value, stored as f64 bits. See
/// [`DEFAULT_FIXED_PROBE_COST_ROWS`].
/// Current fixed cluster-open cost stored as binary64 bits.
static FIXED_PROBE_COST_ROWS_BITS: AtomicU64 =
    AtomicU64::new(DEFAULT_FIXED_PROBE_COST_ROWS.to_bits());

/// Overrides the fixed per-probe cost (the cluster OPEN), in rows of full
/// work. Testing/calibration knob; non-finite or non-positive values reset
/// to [`DEFAULT_FIXED_PROBE_COST_ROWS`].
/// Sets the fixed cluster-open cost in row-work units.
pub fn set_fixed_probe_cost_rows(v: f64) {
    let v = if v.is_finite() && v > 0.0 {
        v
    } else {
        DEFAULT_FIXED_PROBE_COST_ROWS
    };
    FIXED_PROBE_COST_ROWS_BITS.store(v.to_bits(), Relaxed);
}

/// The current fixed per-probe cost (the cluster OPEN), in rows of full
/// work. See [`DEFAULT_FIXED_PROBE_COST_ROWS`].
/// Returns the fixed cluster-open cost in row-work units.
pub(crate) fn fixed_probe_cost_rows() -> f64 {
    f64::from_bits(FIXED_PROBE_COST_ROWS_BITS.load(Relaxed))
}

/// The per-index open share: what fraction of one average cluster's work
/// opening it costs. Covers the open only - routing/search cost is NOT
/// modeled (see [`DEFAULT_FIXED_PROBE_COST_ROWS`]).
///
/// * `n_avg` (`f64`) — native docs per cluster (see [`WorkModel`](crate::vector::ivf::WorkModel)).
///
/// Returns (`f64`): `fixed_probe_cost_rows() / (fixed_probe_cost_rows() +
/// n_avg)`, clamped to (0, 0.5] — a share above one half would mean opens
/// dominate rows, which only degenerate sub-2-row clusters produce.
/// Returns the cluster-open share of average posting work.
pub(crate) fn open_share(n_avg: f64) -> f64 {
    let fixed = fixed_probe_cost_rows();
    (fixed / (fixed + n_avg.max(0.0))).min(0.5)
}

/// An amount of probe WORK, in the model's own unit: 1 unit is one
/// average cluster of work. Budgets, prices, and running spends share
/// this type so they compose only with each other; accumulation is f64.
///
/// NORMALIZATION IDENTITY: an exhaustive, unfiltered, delete-free scan of
/// a segment with `C` clusters charges exactly `C` units - the property
/// that lets the probe fraction keep its meaning across indexes with
/// different cluster granularity.
/// Probe work measured in average-cluster units.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
pub struct WorkUnits(f64);

impl WorkUnits {
    /// No work.
    pub const ZERO: WorkUnits = WorkUnits(0.0);

    /// Wraps an amount already denominated in work units.
    ///
    /// * `units` (`f64`) — the amount, in work units.
    ///
    /// Returns (`WorkUnits`): the typed amount.
    /// Wraps an amount in work units.
    #[inline]
    pub fn new(units: f64) -> WorkUnits {
        WorkUnits(units)
    }

    /// The raw amount, for arithmetic that genuinely leaves the unit.
    ///
    /// Returns (`f64`): the amount, in work units.
    /// Returns the binary64 amount.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }

    /// The single narrowing point, for the telemetry fold.
    ///
    /// Returns (`f32`): the amount, narrowed once for `ProbeStats`.
    /// Returns the amount narrowed to binary32.
    #[inline]
    pub fn to_f32(self) -> f32 {
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
    /// Scaling by a COUNT (rows charged at one price) stays in the unit.
    /// Scales work by a count.
    #[inline]
    fn mul(self, rhs: f64) -> WorkUnits {
        WorkUnits(self.0 * rhs)
    }
}

/// The resolved per-segment prices the probe loop charges against its
/// budget: an open costs `open`, a scored row costs `row`. Built once
/// per segment from
/// [`AdaptiveProbeParams::resolved_work_budget`](crate::vector::ivf::AdaptiveProbeParams::resolved_work_budget)'
/// s `(budget, n_avg, x)`.
/// Per-segment probe budget and event prices.
#[derive(Clone, Copy, Debug)]
pub(in crate::vector) struct UnitPricing {
    /// Work this segment may spend before the ceiling binds.
    pub(in crate::vector) budget: WorkUnits,
    /// The per-index open share `x`: what opening one cluster costs.
    pub(in crate::vector) open: WorkUnits,
    /// `(1 - x)/n_avg`: what one scored row costs.
    pub(in crate::vector) row: WorkUnits,
}

/// Cost accounting and termination for the probe loop: charges each
/// cluster's work against the budget, tracks the APS recall estimate, and
/// decides whether the next ranked cluster is opened.
pub(in crate::vector) struct ProbeController<'a> {
    pricing: UnitPricing,
    work_spent: WorkUnits,
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
        pricing: UnitPricing,
        clusters: usize,
        estimator: Option<RecallEstimator<'a>>,
        recall_target: f32,
    ) -> Self {
        Self {
            pricing,
            work_spent: WorkUnits::ZERO,
            termination: ProbeTermination::Exhausted,
            clusters,
            pulled: 0,
            estimator,
            recall_target,
            recall_estimate: None,
        }
    }

    /// Whether the cluster just pulled from the ranking may be opened.
    /// Checked after the pull: a stop proves another ranked cluster
    /// existed, keeping `Ceiling` and `RecallTarget` distinct from
    /// `Exhausted`.
    pub(in crate::vector) fn admit(&mut self) -> bool {
        self.pulled += 1;
        if self
            .recall_estimate
            .is_some_and(|estimate| estimate >= self.recall_target)
        {
            self.termination = ProbeTermination::RecallTarget;
            return false;
        }
        if self.work_spent >= self.pricing.budget {
            self.termination = ProbeTermination::Ceiling;
            return false;
        }
        true
    }

    /// Charges one cluster open, probed or skipped by the bounds gate.
    pub(in crate::vector) fn charge_open(&mut self) {
        self.work_spent += self.pricing.open;
    }

    /// Charges `rows` scored rows.
    pub(in crate::vector) fn charge_rows(&mut self, rows: usize) {
        self.work_spent += self.pricing.row * rows as f64;
    }

    /// Marks the admitted cluster covered: probed, skipped by the bounds
    /// gate, or without survivors. Either way no result inside the query
    /// ball remains there. `kth` is the k-th result score after it,
    /// `None` while the heap is filling.
    pub(in crate::vector) fn cover(&mut self, kth: Option<Score>) -> crate::Result<()> {
        if let Some(estimator) = self.estimator.as_mut() {
            if let Some(estimate) = estimator.cover_next(kth.map(Similarity::new))? {
                self.recall_estimate = Some(estimate);
            }
        }
        Ok(())
    }

    pub(in crate::vector) fn finish(&self, stats: &mut ProbeStats) {
        // A stacked ranking is sized to the budget, so it can run out just
        // as the budget is spent, short of the segment's clusters; that stop
        // is the ceiling's, not exhaustion.
        stats.termination = match self.termination {
            ProbeTermination::Exhausted
                if self.pulled < self.clusters && self.work_spent >= self.pricing.budget =>
            {
                ProbeTermination::Ceiling
            }
            termination => termination,
        };
        stats.work_charged += self.work_spent.to_f32();
        stats.work_budget += self.pricing.budget.to_f32();
        stats.recall_estimate = self.recall_estimate;
    }
}
