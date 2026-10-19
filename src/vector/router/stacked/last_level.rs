//! Experiment: early stop while expanding the bottom router's parent beam.
//!
//! The level still asks its parent for one beam (stock is 25% of this level's
//! lists). Parents are expanded in the order that beam already has. After each
//! parent its children — the leaf centroids — are scored into the `k_req` heap.
//! A rule file overrides [`StackedIvfConfig::last_level`](super::levels::StackedIvfConfig)
//! for that query; a missing file leaves the typed config, whose default is
//! stock (expand the whole beam).
//!
//! Grammar, one line in `/tmp/tv_parent_rule` or `$TV_PARENT_RULE_FILE`:
//!
//! * `none` / empty — stock
//! * `frac:F` — expand `ceil(F · nlist)` parents. `F >= 1` (`frac:1.0`) expands every parent.
//!   Omitted `F` defaults to `0.25`.
//! * `kscale:A:MIN` — expand `max(ceil(A · k_req / children_per_parent), MIN)` parents, capped at
//!   `nlist`. Omitted fields default to `A = 4`, `MIN = 4`.
//! * `aps:GAMMA:TARGET|auto:MINPARENTS:SIGMA[:OFFSET]`
//! * `gap:C:MINPARENTS:SIGMA[:OFFSET]`
//!
//! `TARGET=auto` (the typed-config default) uses the recall passed into this
//! level's search: the harness target (`r90` / `r95` / `r99`), not the value
//! geometric APS may have forced to `1` above 128 dimensions. A numeric
//! `TARGET` is fixed. `SIGMA` is `child` or `parent`. `OFFSET` is `mean`
//! (default) or `max`.
//!
//! # Sigma
//!
//! Population standard deviation, `sqrt(Σ(x − mean)² / n)`, then lifted to
//! [`SIGMA_EPS`] when the result is non-finite or smaller than that floor.
//!
//! * `child` — std dev of every leaf-centroid score seen when the floor is first reached (after the
//!   first `max(min_parents, parents needed to fill k_req)` parents). Frozen from then on.
//! * `parent` — std dev of the first-pass parent scores across the full requested beam, each
//!   shifted onto the child scale by the offset known at that same floor. A single global offset
//!   does not change a standard deviation; the shift is applied anyway so the definition stays
//!   literal. Frozen from then on.
//!
//! # Offset
//!
//! Both modes are one global shift estimated from parents expanded so far,
//! the same accumulator the mean offset uses. Per-parent max offsets are not
//! stored on the beam.
//!
//! * `mean` — `(Σ child scores − Σ parent_score · rows) / Σ rows`, i.e. the mean child score minus
//!   the child-weighted mean parent score (PR #332's child-minus-parent offset).
//! * `max` — the same quotient with every child score replaced by that parent's best child score:
//!   `(Σ best_child · rows − Σ parent_score · rows) / Σ rows`. That is the child-weighted average
//!   of `(best_child − parent_score)`.
//!
//! # Mass
//!
//! Opened children (leaf centroids already scored) weigh
//! `exp((s − s_best) / σ)` each — one row per centroid; the router does not
//! know how many vectors sit under a leaf cluster. An unexpanded parent's
//! estimated mass is `rows · exp((shifted parent score − s_best) / σ)`, where
//! `rows` is the parent's child count from this level's offsets. Centering by
//! `s_best` is PR #332's `rows_i` weight; it cancels in the mass ratio and
//! keeps the exponentials finite. `s_best` is the best leaf score seen so far.

use std::cell::RefCell;
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;

/// Floor under a computed sigma. A zero spread would divide the gap.
pub const SIGMA_EPS: f64 = 1e-6;

/// Default `MINPARENTS` for the typed `aps` / `gap` constructors.
#[cfg_attr(not(test), allow(dead_code))]
pub const DEFAULT_MIN_PARENTS: usize = 8;

const DEFAULT_FRAC: f64 = 0.25;
const DEFAULT_KSCALE_ALPHA: f64 = 4.0;
const DEFAULT_KSCALE_MIN: usize = 4;

/// Where σ comes from. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigmaSource {
    /// Leaf-centroid scores at the floor, then frozen.
    Child,
    /// Shifted parent scores of the whole beam, frozen at the floor.
    Parent,
}

/// How a parent score moves onto the child scale. See the module docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScoreOffset {
    /// Mean child score minus the child-weighted parent score.
    #[default]
    Mean,
    /// Child-weighted average of `(best child − parent score)`.
    Max,
}

/// Recall the estimator spends. `Auto` reads the level's requested recall.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RecallTarget {
    /// The recall passed into the bottom-level search.
    Auto,
    /// A fixed target in `[0, 1]`.
    Fixed(f64),
}

/// Why the bottom level stopped expanding parents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Normalized mass estimate fell inside the recall budget after the floor.
    Estimator,
    /// Variant B: the next shifted parent score cleared `c · σ` below the cutoff.
    Gap,
    /// The estimate or the gap rule already held when the floor was first met.
    Floor,
    /// The requested beam was expanded.
    Cap,
    /// The parent returned fewer lists than were requested.
    Exhausted,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Estimator => "estimator",
            Self::Gap => "gap",
            Self::Floor => "floor",
            Self::Cap => "cap",
            Self::Exhausted => "exhausted",
        })
    }
}

/// Typed bottom-level rule. [`Default`] is stock; the rule file overrides it.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum LastLevelRule {
    /// Expand the stock parent beam. Geometric APS at this level is unchanged.
    #[default]
    Stock,
    /// `frac:F`.
    Frac(f64),
    /// `kscale:A:MIN`.
    KScale { alpha: f64, min_parents: usize },
    /// Normalized estimator. `target: Auto`, `min_parents: 8`, `sigma: Child`,
    /// `offset: Mean` are the constructor defaults.
    Aps {
        gamma: f64,
        target: RecallTarget,
        min_parents: usize,
        sigma: SigmaSource,
        offset: ScoreOffset,
    },
    /// Variant B. Same constructor defaults as [`Self::Aps`] aside from `c`.
    Gap {
        c: f64,
        min_parents: usize,
        sigma: SigmaSource,
        offset: ScoreOffset,
    },
}

impl LastLevelRule {
    /// `aps` with `TARGET=auto`, [`DEFAULT_MIN_PARENTS`], `SIGMA=child`, `OFFSET=mean`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn aps(gamma: f64) -> Self {
        Self::Aps {
            gamma,
            target: RecallTarget::Auto,
            min_parents: DEFAULT_MIN_PARENTS,
            sigma: SigmaSource::Child,
            offset: ScoreOffset::Mean,
        }
    }

    /// `gap` with [`DEFAULT_MIN_PARENTS`], `SIGMA=child`, `OFFSET=mean`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn gap(c: f64) -> Self {
        Self::Gap {
            c,
            min_parents: DEFAULT_MIN_PARENTS,
            sigma: SigmaSource::Child,
            offset: ScoreOffset::Mean,
        }
    }

    pub fn is_stock(&self) -> bool {
        matches!(self, Self::Stock)
    }

    /// Canonical spelling. The log prefers the raw rule-file line when there is one.
    pub fn label(&self) -> String {
        match self {
            Self::Stock => "stock".to_string(),
            Self::Frac(f) => format!("frac:{f}"),
            Self::KScale { alpha, min_parents } => format!("kscale:{alpha}:{min_parents}"),
            Self::Aps {
                gamma,
                target,
                min_parents,
                sigma,
                offset,
            } => format!(
                "aps:{gamma}:{}:{min_parents}:{}:{}",
                target.label(),
                sigma.label(),
                offset.label()
            ),
            Self::Gap {
                c,
                min_parents,
                sigma,
                offset,
            } => format!("gap:{c}:{min_parents}:{}:{}", sigma.label(), offset.label()),
        }
    }

    /// How many parents to ask for. `Stock` keeps the 25% beam (and `aps` / `gap`
    /// early-stop inside it). `Count` is an exact list count with geometric APS off,
    /// so `frac:1.0` returns every child when the parent is the top router level.
    pub fn parent_probe(
        &self,
        k_req: usize,
        nlist: usize,
        children_per_parent: f64,
    ) -> ParentProbe {
        let nlist = nlist.max(1);
        match self {
            Self::Stock | Self::Aps { .. } | Self::Gap { .. } => ParentProbe::Stock,
            Self::Frac(f) => {
                let n = if *f >= 1.0 {
                    nlist
                } else {
                    ((nlist as f64) * (*f).max(0.0)).ceil() as usize
                };
                ParentProbe::Count(n.clamp(1, nlist))
            }
            Self::KScale { alpha, min_parents } => {
                let cpp = if children_per_parent > 0.0 {
                    children_per_parent
                } else {
                    1.0
                };
                let n = ((*alpha * k_req as f64 / cpp).ceil() as usize).max(*min_parents);
                ParentProbe::Count(n.clamp(1, nlist))
            }
        }
    }
}

impl RecallTarget {
    fn label(self) -> String {
        match self {
            Self::Auto => "auto".to_string(),
            Self::Fixed(t) => format!("{t}"),
        }
    }

    fn resolve(self, requested_recall: f32) -> f64 {
        let raw = match self {
            Self::Auto => f64::from(requested_recall),
            Self::Fixed(target) => target,
        };
        if raw.is_finite() {
            raw.clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

impl SigmaSource {
    fn label(self) -> &'static str {
        match self {
            Self::Child => "child",
            Self::Parent => "parent",
        }
    }
}

impl ScoreOffset {
    fn label(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Max => "max",
        }
    }
}

/// What the bottom level asks of its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParentProbe {
    /// This level's `n_probe` lists, with the parent's usual recall.
    /// This tip has no routing-expansion factor, so the stock beam is
    /// `n_probe` itself. `aps` / `gap` early-stop inside that beam.
    Stock,
    /// Exactly this many lists, geometric APS disabled on that call.
    Count(usize),
}

/// Typed config carried on [`StackedIvfConfig`]. The rule file replaces [`Self::rule`].
#[derive(Clone, Debug, PartialEq)]
pub struct LastLevelStop {
    pub rule: LastLevelRule,
}

impl Default for LastLevelStop {
    fn default() -> Self {
        Self {
            rule: LastLevelRule::Stock,
        }
    }
}

impl LastLevelStop {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn rule(rule: LastLevelRule) -> Self {
        Self { rule }
    }
}

/// One parent already ordered by the beam: its score, and how many leaf
/// centroids its offset range holds.
#[derive(Clone, Copy, Debug)]
pub struct ParentBeam {
    pub score: f64,
    pub rows: usize,
}

/// Online early-stop over a fixed parent beam. Production and the synthetic
/// tests share this.
pub struct Stopper {
    kind: StopKind,
    parents: Vec<ParentBeam>,
    min_parents: usize,
    gamma: f64,
    c: f64,
    target: f64,
    sigma_source: SigmaSource,
    offset_mode: ScoreOffset,
    expanded: usize,
    child_scores: Vec<f64>,
    child_score_sum: f64,
    parent_weighted: f64,
    best_weighted: f64,
    members: usize,
    /// Frozen at the floor. `None` until then.
    sigma: Option<f64>,
    /// `expanded` at the moment the floor was first met.
    floor_at: Option<usize>,
}

#[derive(Clone, Copy)]
enum StopKind {
    Aps,
    Gap,
}

impl StopKind {
    fn reason(self) -> StopReason {
        match self {
            Self::Aps => StopReason::Estimator,
            Self::Gap => StopReason::Gap,
        }
    }
}

impl Stopper {
    /// `parents` is the beam, best parent first. `requested_recall` is what
    /// `TARGET=auto` resolves to.
    pub fn new(
        rule: &LastLevelRule,
        parents: Vec<ParentBeam>,
        requested_recall: f32,
    ) -> Option<Self> {
        let (kind, gamma, c, target, min_parents, sigma_source, offset_mode) = match *rule {
            LastLevelRule::Aps {
                gamma,
                target,
                min_parents,
                sigma,
                offset,
            } => (
                StopKind::Aps,
                gamma,
                0.0,
                target.resolve(requested_recall),
                min_parents,
                sigma,
                offset,
            ),
            LastLevelRule::Gap {
                c,
                min_parents,
                sigma,
                offset,
            } => (StopKind::Gap, 0.0, c, 1.0, min_parents, sigma, offset),
            _ => return None,
        };
        Some(Self {
            kind,
            parents,
            min_parents,
            gamma,
            c,
            target,
            sigma_source,
            offset_mode,
            expanded: 0,
            child_scores: Vec::new(),
            child_score_sum: 0.0,
            parent_weighted: 0.0,
            best_weighted: 0.0,
            members: 0,
            sigma: None,
            floor_at: None,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn expanded(&self) -> usize {
        self.expanded
    }

    /// Sigma after it has been frozen, if the floor has been reached.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn sigma(&self) -> Option<f64> {
        self.sigma
    }

    /// Running child-scale shift. `0` before any child has been scored.
    pub fn shift(&self) -> f64 {
        if self.members == 0 {
            return 0.0;
        }
        let numer = match self.offset_mode {
            ScoreOffset::Mean => self.child_score_sum - self.parent_weighted,
            ScoreOffset::Max => self.best_weighted - self.parent_weighted,
        };
        let shift = numer / self.members as f64;
        if shift.is_finite() {
            shift
        } else {
            0.0
        }
    }

    /// Centered masses `(opened, unopened estimate)` over the beam.
    pub fn masses(&self) -> (f64, f64) {
        let sigma = self.sigma.unwrap_or_else(|| self.compute_sigma());
        let shift = self.shift();
        let s_best = self
            .child_scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !s_best.is_finite() {
            return (0.0, 0.0);
        }
        let mut opened = 0.0;
        for score in &self.child_scores {
            opened += ((*score - s_best) / sigma).exp();
        }
        let mut unopened = 0.0;
        for parent in self.parents.iter().skip(self.expanded) {
            unopened += parent.rows as f64 * ((parent.score + shift - s_best) / sigma).exp();
        }
        (opened, unopened)
    }

    /// Record `child_scores` of the next parent. `cutoff` is the `k_req`-th
    /// leaf score once the heap holds `k_req` candidates, otherwise `None`.
    ///
    /// `Some` means do not expand another parent.
    pub fn observe(&mut self, child_scores: &[f64], cutoff: Option<f64>) -> Option<StopReason> {
        let parent = self.parents.get(self.expanded).copied()?;
        let n = child_scores.len();
        let mut best = f64::NEG_INFINITY;
        for score in child_scores {
            self.child_score_sum += *score;
            self.child_scores.push(*score);
            best = best.max(*score);
        }
        self.parent_weighted += parent.score * n as f64;
        if best.is_finite() {
            self.best_weighted += best * n as f64;
        }
        self.members += n;
        self.expanded += 1;

        let floor = self.expanded >= self.min_parents && cutoff.is_some();
        if floor && self.floor_at.is_none() {
            self.floor_at = Some(self.expanded);
            self.sigma = Some(self.compute_sigma());
        }
        if !floor {
            return None;
        }
        let sigma = self.sigma.unwrap_or(SIGMA_EPS);
        let next = self.parents.get(self.expanded).copied()?;
        let cutoff = cutoff?;
        let shifted_next = next.score + self.shift();
        if !shifted_next.is_finite() || !cutoff.is_finite() {
            return None;
        }
        let fires = match self.kind {
            StopKind::Gap => shifted_next < cutoff - self.c * sigma,
            StopKind::Aps => {
                let gap = cutoff - shifted_next;
                self.estimator_fires(gap, sigma)
            }
        };
        if !fires {
            return None;
        }
        Some(if self.floor_at == Some(self.expanded) {
            StopReason::Floor
        } else {
            self.kind.reason()
        })
    }

    /// The beam is used up. `requested` is how many parents the level asked for.
    pub fn finish(&self, requested: usize) -> StopReason {
        if self.parents.len() < requested {
            StopReason::Exhausted
        } else {
            StopReason::Cap
        }
    }

    fn estimator_fires(&self, gap: f64, sigma: f64) -> bool {
        let (opened, unopened) = self.masses();
        let total = opened + unopened;
        if !(total.is_finite() && total > 0.0) || self.expanded == 0 || !gap.is_finite() {
            return false;
        }
        let lhs = (unopened / total) / self.expanded as f64 * (-gap / sigma).exp();
        let rhs = (1.0 - self.target).powf(self.gamma);
        lhs.is_finite() && rhs.is_finite() && lhs <= rhs
    }

    /// Population std dev, guarded by [`SIGMA_EPS`].
    fn compute_sigma(&self) -> f64 {
        let std = match self.sigma_source {
            SigmaSource::Child => population_std(&self.child_scores),
            SigmaSource::Parent => {
                // Shift the whole beam by the offset known right now. A uniform
                // shift leaves the std dev unchanged; applying it keeps the
                // `parent` definition (child scale) literal.
                let shift = self.shift();
                let shifted: Vec<f64> = self.parents.iter().map(|p| p.score + shift).collect();
                population_std(&shifted)
            }
        };
        if std.is_finite() && std >= SIGMA_EPS {
            std
        } else {
            SIGMA_EPS
        }
    }
}

/// `sqrt(mean square deviation)`. Empty input is `0` (the caller applies [`SIGMA_EPS`]).
pub fn population_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / n;
    var.sqrt()
}

/// Drive `rule` over synthetic parents. `children[i]` are parent `i`'s leaf scores.
/// Returns `(parents_expanded, reason, sigma)`.
#[cfg_attr(not(test), allow(dead_code))]
pub fn run_stop(
    rule: &LastLevelRule,
    parent_scores: &[f64],
    children: &[Vec<f64>],
    k_req: usize,
    requested_recall: f32,
) -> (usize, StopReason, Option<f64>) {
    let parents = parent_scores
        .iter()
        .zip(children)
        .map(|(&score, kids)| ParentBeam {
            score,
            rows: kids.len(),
        })
        .collect();
    let mut stopper = Stopper::new(rule, parents, requested_recall).expect("aps or gap rule");
    let mut heap: Vec<f64> = Vec::new();
    for kids in children {
        for score in kids {
            heap.push(*score);
        }
        heap.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        heap.truncate(k_req);
        let cutoff = (heap.len() >= k_req)
            .then(|| heap.last().copied())
            .flatten();
        if let Some(reason) = stopper.observe(kids, cutoff) {
            return (stopper.expanded(), reason, stopper.sigma());
        }
    }
    let requested = parent_scores.len();
    (
        stopper.expanded(),
        stopper.finish(requested),
        stopper.sigma(),
    )
}

/// Rule in force for this query: the file if it exists and parses, otherwise `config`.
pub fn effective_rule(config: &LastLevelStop) -> (LastLevelRule, String) {
    match rule_slot() {
        RuleSlot::Absent => (config.rule.clone(), config.rule.label()),
        RuleSlot::Text(text) => parse_or_config(config, &text),
        RuleSlot::ReadFile => {
            let path = std::env::var("TV_PARENT_RULE_FILE")
                .unwrap_or_else(|_| "/tmp/tv_parent_rule".to_string());
            rule_from_path(std::path::Path::new(&path), config)
        }
    }
}

/// Missing path, empty file, and unparseable text keep `config` (stock, by default).
pub fn rule_from_path(path: &std::path::Path, config: &LastLevelStop) -> (LastLevelRule, String) {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_or_config(config, text.trim()),
        Err(_) => (config.rule.clone(), config.rule.label()),
    }
}

fn parse_or_config(config: &LastLevelStop, text: &str) -> (LastLevelRule, String) {
    if text.is_empty() {
        return (config.rule.clone(), config.rule.label());
    }
    match parse_rule(text) {
        Some(rule) => (rule, text.to_string()),
        None => (config.rule.clone(), config.rule.label()),
    }
}

/// `None` when `text` is not a known rule. `none`, `stock`, and empty are stock.
pub fn parse_rule(text: &str) -> Option<LastLevelRule> {
    let text = text.trim();
    if text.is_empty() || text == "none" || text == "stock" {
        return Some(LastLevelRule::Stock);
    }
    let mut parts = text.split(':');
    let name = parts.next()?;
    let rest: Vec<&str> = parts.collect();
    match name {
        "frac" => {
            let fraction = match rest.first() {
                Some(value) => value.parse().ok()?,
                None => DEFAULT_FRAC,
            };
            if rest.len() > 1 {
                return None;
            }
            Some(LastLevelRule::Frac(fraction))
        }
        "kscale" => {
            let alpha = match rest.first() {
                Some(value) => value.parse().ok()?,
                None => DEFAULT_KSCALE_ALPHA,
            };
            let min_parents = match rest.get(1) {
                Some(value) => value.parse().ok()?,
                None => DEFAULT_KSCALE_MIN,
            };
            if rest.len() > 2 {
                return None;
            }
            Some(LastLevelRule::KScale { alpha, min_parents })
        }
        "aps" => parse_aps(&rest),
        "gap" => parse_gap(&rest),
        _ => None,
    }
}

fn parse_aps(rest: &[&str]) -> Option<LastLevelRule> {
    if !(4..=5).contains(&rest.len()) {
        return None;
    }
    Some(LastLevelRule::Aps {
        gamma: rest[0].parse().ok()?,
        target: parse_target(rest[1])?,
        min_parents: rest[2].parse().ok()?,
        sigma: parse_sigma(rest[3])?,
        offset: parse_optional_offset(rest.get(4).copied())?,
    })
}

fn parse_gap(rest: &[&str]) -> Option<LastLevelRule> {
    if !(3..=4).contains(&rest.len()) {
        return None;
    }
    Some(LastLevelRule::Gap {
        c: rest[0].parse().ok()?,
        min_parents: rest[1].parse().ok()?,
        sigma: parse_sigma(rest[2])?,
        offset: parse_optional_offset(rest.get(3).copied())?,
    })
}

fn parse_target(text: &str) -> Option<RecallTarget> {
    if text == "auto" {
        Some(RecallTarget::Auto)
    } else {
        text.parse().ok().map(RecallTarget::Fixed)
    }
}

fn parse_sigma(text: &str) -> Option<SigmaSource> {
    match text {
        "child" => Some(SigmaSource::Child),
        "parent" => Some(SigmaSource::Parent),
        _ => None,
    }
}

fn parse_optional_offset(text: Option<&str>) -> Option<ScoreOffset> {
    match text {
        None => Some(ScoreOffset::Mean),
        Some("mean") => Some(ScoreOffset::Mean),
        Some("max") => Some(ScoreOffset::Max),
        Some(_) => None,
    }
}

/// One bottom-level query, written when `/tmp/tv_log_parents` exists.
///
/// CSV columns:
/// `rule,k_req,parents_requested,parents_expanded,leaf_centroids_scored,stop_reason,opened_mass,
/// unopened_mass_est`
pub struct ParentLog<'a> {
    pub rule: &'a str,
    pub k_req: usize,
    pub parents_requested: usize,
    pub parents_expanded: usize,
    pub leaf_centroids_scored: usize,
    /// `estimator`, `gap`, `floor`, `cap`, `exhausted`, or `stock`.
    pub stop_reason: &'a str,
    pub opened_mass: f64,
    pub unopened_mass_est: f64,
}

/// Append [`ParentLog`] as one CSV row. A missing flag file writes nothing.
pub fn log_bottom(row: &ParentLog<'_>) {
    if !std::path::Path::new("/tmp/tv_log_parents")
        .try_exists()
        .unwrap_or(false)
    {
        return;
    }
    let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/tv_parents.log")
    else {
        return;
    };
    let _ = writeln!(
        file,
        "{},{},{},{},{},{},{:.16e},{:.16e}",
        row.rule,
        row.k_req,
        row.parents_requested,
        row.parents_expanded,
        row.leaf_centroids_scored,
        row.stop_reason,
        row.opened_mass,
        row.unopened_mass_est,
    );
}

#[derive(Clone)]
#[cfg_attr(not(test), allow(dead_code))]
enum RuleSlot {
    ReadFile,
    Absent,
    Text(String),
}

thread_local! {
    static RULE_SLOT: RefCell<RuleSlot> = const { RefCell::new(RuleSlot::ReadFile) };
}

fn rule_slot() -> RuleSlot {
    RULE_SLOT.with(|slot| slot.borrow().clone())
}

/// Test guard: this thread treats the rule file as missing.
#[cfg_attr(not(test), allow(dead_code))]
pub struct RuleGuard;

impl Drop for RuleGuard {
    fn drop(&mut self) {
        RULE_SLOT.with(|slot| *slot.borrow_mut() = RuleSlot::ReadFile);
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn force_absent() -> RuleGuard {
    RULE_SLOT.with(|slot| *slot.borrow_mut() = RuleSlot::Absent);
    RuleGuard
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn force_text(text: impl Into<String>) -> RuleGuard {
    RULE_SLOT.with(|slot| *slot.borrow_mut() = RuleSlot::Text(text.into()));
    RuleGuard
}

/// Children per parent from offset ranges. Used by `kscale`.
pub fn children_per_parent(nlist: usize, offsets: &[(u64, u64)]) -> f64 {
    let rows: u64 = offsets
        .iter()
        .map(|(start, end)| end.saturating_sub(*start))
        .sum();
    rows as f64 / nlist.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kids(parent: f64) -> Vec<f64> {
        vec![parent + 0.2, parent, parent - 0.2, parent - 0.4]
    }

    fn beam(n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let scores: Vec<f64> = (0..n).map(|i| -0.5 * i as f64).collect();
        let children = scores.iter().copied().map(kids).collect();
        (scores, children)
    }

    fn gap(c: f64, min_parents: usize) -> LastLevelRule {
        LastLevelRule::Gap {
            c,
            min_parents,
            sigma: SigmaSource::Parent,
            offset: ScoreOffset::Mean,
        }
    }

    fn aps(gamma: f64, min_parents: usize) -> LastLevelRule {
        LastLevelRule::Aps {
            gamma,
            target: RecallTarget::Fixed(0.9),
            min_parents,
            sigma: SigmaSource::Parent,
            offset: ScoreOffset::Mean,
        }
    }

    #[test]
    fn floor_is_respected() {
        let (scores, children) = beam(12);
        // The heap fills on the first parent (4 children, k=1) and c=0 fires
        // immediately, but min_parents holds the expansion at 8.
        let (expanded, reason, _) = run_stop(&gap(0.0, 8), &scores, &children, 1, 1.0);
        assert_eq!(expanded, 8, "{reason}");
        assert_eq!(reason, StopReason::Floor);

        // Filling k_req takes 4 parents (3 children each); that beats min_parents=2.
        let wide: Vec<Vec<f64>> = (0..12)
            .map(|i| vec![-i as f64, -i as f64 - 1.0, -i as f64 - 2.0])
            .collect();
        let parent_scores: Vec<f64> = (0..12).map(|i| -i as f64).collect();
        let (expanded, reason, _) = run_stop(&gap(0.0, 2), &parent_scores, &wide, 10, 1.0);
        assert_eq!(expanded, 4, "{reason}");
        assert_eq!(reason, StopReason::Floor);
    }

    #[test]
    fn cap_is_respected() {
        let (scores, children) = beam(6);
        // Enormous c never fires. min_parents past the beam still stops at the beam.
        let (expanded, reason, _) = run_stop(&gap(1.0e6, 100), &scores, &children, 1, 1.0);
        assert_eq!(expanded, 6);
        assert_eq!(reason, StopReason::Cap);

        let (expanded, reason, _) = run_stop(&gap(1.0e6, 1), &scores, &children, 1, 1.0);
        assert_eq!(expanded, scores.len());
        assert_eq!(reason, StopReason::Cap);
    }

    #[test]
    fn gap_rule_stops_when_the_next_parent_falls_c_sigma_below_cutoff() {
        let (scores, children) = beam(12);
        let (tight, tight_reason, _) = run_stop(&gap(0.0, 2), &scores, &children, 4, 1.0);
        let (loose, loose_reason, _) = run_stop(&gap(50.0, 2), &scores, &children, 4, 1.0);
        assert_eq!(tight_reason, StopReason::Floor);
        assert!(tight < loose, "c=0 expanded {tight}, c=50 expanded {loose}");
        assert_eq!(loose, scores.len());
        assert_eq!(loose_reason, StopReason::Cap);

        // A moderate c stops strictly inside the beam, on the gap rule.
        let (mid, mid_reason, _) = run_stop(&gap(1.0, 2), &scores, &children, 4, 1.0);
        assert!(mid > tight && mid < loose, "c=1 expanded {mid}");
        assert_eq!(mid_reason, StopReason::Gap);
    }

    #[test]
    fn estimator_is_monotone_in_gamma() {
        let (scores, children) = beam(16);
        let mut previous = 0usize;
        for gamma in [1.0, 1.5, 2.0] {
            let (expanded, reason, _) = run_stop(&aps(gamma, 2), &scores, &children, 4, 0.9);
            assert!(
                expanded >= previous,
                "gamma {gamma} expanded {expanded} after {previous} ({reason})"
            );
            previous = expanded;
        }
        let (low, _, _) = run_stop(&aps(1.0, 2), &scores, &children, 4, 0.9);
        let (high, _, _) = run_stop(&aps(2.0, 2), &scores, &children, 4, 0.9);
        assert!(
            low < high,
            "gamma should move the stop on this beam: {low} vs {high}"
        );
    }

    #[test]
    fn child_sigma_freezes_at_the_floor_and_parent_sigma_ignores_a_uniform_shift() {
        let (scores, mut children) = beam(10);
        // Floor is 4 parents (min_parents). Later parents are far away; a sigma
        // that kept updating would leave the eps floor.
        for kids in children.iter_mut().skip(4) {
            for score in kids {
                *score -= 100.0;
            }
        }
        let rule = LastLevelRule::Gap {
            c: 1.0e6,
            min_parents: 4,
            sigma: SigmaSource::Child,
            offset: ScoreOffset::Mean,
        };
        let (_expanded, _reason, sigma) = run_stop(&rule, &scores, &children, 1, 1.0);
        let frozen = sigma.expect("floor freezes sigma");
        let floor_scores: Vec<f64> = children.iter().take(4).flatten().copied().collect();
        // The first four parents' children are an arithmetic progression with
        // non-zero spread, and later parents must not change it.
        assert!((frozen - population_std(&floor_scores)).abs() < 1e-12);

        let parent_rule = LastLevelRule::Gap {
            c: 1.0e6,
            min_parents: 4,
            sigma: SigmaSource::Parent,
            offset: ScoreOffset::Mean,
        };
        let (_e, _r, parent_sigma) = run_stop(&parent_rule, &scores, &children, 1, 1.0);
        let parent_sigma = parent_sigma.unwrap();
        assert!(
            (parent_sigma - population_std(&scores)).abs() < 1e-9,
            "a global child-scale shift must not change parent sigma: {parent_sigma}"
        );
    }

    #[test]
    fn zero_spread_uses_sigma_eps() {
        let scores = vec![1.0; 6];
        let children = vec![vec![1.0; 2]; 6];
        let (_e, _r, sigma) = run_stop(&gap(1.0, 2), &scores, &children, 1, 1.0);
        assert_eq!(sigma, Some(SIGMA_EPS));
    }

    #[test]
    fn max_offset_is_measurable_against_mean() {
        // One child sits far above the others, so best-child and mean diverge.
        let scores = vec![0.0, -1.0, -2.0, -3.0, -8.0, -9.0];
        let children: Vec<Vec<f64>> = scores
            .iter()
            .map(|parent| vec![parent + 5.0, parent - 1.0, parent - 1.0, parent - 1.0])
            .collect();
        let mean = LastLevelRule::Gap {
            c: 0.5,
            min_parents: 1,
            sigma: SigmaSource::Parent,
            offset: ScoreOffset::Mean,
        };
        let max = LastLevelRule::Gap {
            c: 0.5,
            min_parents: 1,
            sigma: SigmaSource::Parent,
            offset: ScoreOffset::Max,
        };
        let (mean_n, _, _) = run_stop(&mean, &scores, &children, 2, 1.0);
        let (max_n, _, _) = run_stop(&max, &scores, &children, 2, 1.0);
        assert_ne!(
            mean_n, max_n,
            "mean expanded {mean_n}, max expanded {max_n}"
        );
    }

    #[test]
    fn rule_grammar() {
        assert!(matches!(parse_rule("none"), Some(LastLevelRule::Stock)));
        assert!(matches!(parse_rule("frac:1.0"), Some(LastLevelRule::Frac(f)) if f == 1.0));
        assert!(matches!(parse_rule("frac"), Some(LastLevelRule::Frac(f)) if f == 0.25));
        match parse_rule("kscale:8:96").unwrap() {
            LastLevelRule::KScale { alpha, min_parents } => {
                assert_eq!(alpha, 8.0);
                assert_eq!(min_parents, 96);
            }
            other => panic!("{other:?}"),
        }
        match parse_rule("aps:1.5:auto:8:child").unwrap() {
            LastLevelRule::Aps {
                gamma,
                target,
                min_parents,
                sigma,
                offset,
            } => {
                assert_eq!(gamma, 1.5);
                assert_eq!(target, RecallTarget::Auto);
                assert_eq!(min_parents, 8);
                assert_eq!(sigma, SigmaSource::Child);
                assert_eq!(offset, ScoreOffset::Mean);
            }
            other => panic!("{other:?}"),
        }
        match parse_rule("aps:2:0.95:4:parent:max").unwrap() {
            LastLevelRule::Aps {
                target: RecallTarget::Fixed(t),
                sigma: SigmaSource::Parent,
                offset: ScoreOffset::Max,
                min_parents: 4,
                ..
            } => assert!((t - 0.95).abs() < 1e-12),
            other => panic!("{other:?}"),
        }
        match parse_rule("gap:2:8:parent").unwrap() {
            LastLevelRule::Gap {
                c,
                min_parents,
                sigma,
                offset,
            } => {
                assert_eq!(c, 2.0);
                assert_eq!(min_parents, 8);
                assert_eq!(sigma, SigmaSource::Parent);
                assert_eq!(offset, ScoreOffset::Mean);
            }
            other => panic!("{other:?}"),
        }
        match LastLevelRule::gap(2.0) {
            LastLevelRule::Gap {
                c,
                min_parents,
                sigma,
                offset,
            } => {
                assert_eq!(c, 2.0);
                assert_eq!(min_parents, DEFAULT_MIN_PARENTS);
                assert_eq!(sigma, SigmaSource::Child);
                assert_eq!(offset, ScoreOffset::Mean);
            }
            other => panic!("{other:?}"),
        }
        match LastLevelRule::aps(1.5) {
            LastLevelRule::Aps {
                target: RecallTarget::Auto,
                min_parents: DEFAULT_MIN_PARENTS,
                sigma: SigmaSource::Child,
                offset: ScoreOffset::Mean,
                ..
            } => {}
            other => panic!("{other:?}"),
        }
        assert!(parse_rule("gap:1:8:nope").is_none());
        assert!(parse_rule("adapt:pat:1").is_none());
        assert_eq!(
            LastLevelRule::Frac(1.0).parent_probe(4, 10, 3.0),
            ParentProbe::Count(10)
        );
        assert_eq!(
            LastLevelRule::aps(1.5).parent_probe(4, 40, 5.0),
            ParentProbe::Stock
        );
    }

    #[test]
    fn missing_file_keeps_the_typed_rule() {
        let _guard = force_absent();
        let config = LastLevelStop::rule(LastLevelRule::aps(1.5));
        let (rule, label) = effective_rule(&config);
        assert_eq!(rule, LastLevelRule::aps(1.5));
        assert_eq!(label, rule.label());
        let (stock, _) = effective_rule(&LastLevelStop::default());
        assert!(stock.is_stock());
    }
}
