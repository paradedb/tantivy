//! Per-segment vector search execution.
//!
//! Built once per segment by
//! [`TopDocsByVectorSimilarity`](super::collector::TopDocsByVectorSimilarity)
//! around the segment's cached [`VectorIndexReader`]. The search strategy
//! branches once, on whether the reader carries an [`IvfIndex`]: with it, the
//! filter is drained into a bitmap and the routed clusters are probed
//! adaptively; without it, the filter `Scorer` is iterated doc-by-doc and
//! every vector is scored exactly. Either way, every survivor's bytes are
//! fetched with one stride-sized read ([`VectorIndexReader::vector_bytes_for_row`])
//! — the unit the pg-backed `Directory` can serve zero-copy.

use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use common::BitSet;

use super::bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, to_bound_space, HeapPeek, QueryBound,
    QueryBoundTracker, Verdict,
};
use super::distance::norm_squared_wide;
use super::index_reader::VectorIndexReader;
use super::ivf::{AdaptiveProbeParams, Candidate, IvfIndex, IvfSearchMetrics, Workspace};
use super::prepared::PreparedQuery;
use super::tie_break::NoTieBreak;
use super::VectorElement;
use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, TopNComputer};
use crate::fastfield::AliveBitSet;
use crate::query::Weight;
use crate::schema::{Field, Metric};
use crate::{DocAddress, DocId, Score, SegmentOrdinal, SegmentReader, TantivyError};

/// The settled result.
type TieBreakHits<K> = Vec<(
    (Score, <K as SegmentSortKeyComputer>::SegmentSortKey),
    DocAddress,
)>;

/// The in-flight accumulator.
type TieBreakHeap<K, CTail> = TopNComputer<
    (Score, <K as SegmentSortKeyComputer>::SegmentSortKey),
    DocId,
    (NaturalComparator, CTail),
>;

/// Per-segment vector search: the segment's [`VectorIndexReader`] plus the
/// per-query state. Build via [`VectorBackend::for_segment`].
pub struct VectorBackend<T: VectorElement> {
    reader: Arc<VectorIndexReader>,
    query: Arc<PreparedQuery<T>>,
    adaptive: AdaptiveProbeParams,
    segment_ord: SegmentOrdinal,
}

impl<T: VectorElement> VectorBackend<T> {
    /// Opens the segment's cached vector reader for `field` and prepares the
    /// query against the field's metric. A segment with no vector data gets
    /// the empty reader and yields no hits.
    pub fn for_segment(
        segment_reader: &SegmentReader,
        segment_ord: SegmentOrdinal,
        field: Field,
        query: Arc<Vec<T>>,
        adaptive: AdaptiveProbeParams,
    ) -> crate::Result<Self> {
        let reader = segment_reader.vector_index(field)?;
        let query = Arc::new(PreparedQuery::<T>::new(reader.options().metric(), query));
        Ok(Self {
            reader,
            query,
            adaptive,
            segment_ord,
        })
    }

    /// Top-N within this segment: probe routed clusters when the reader has
    /// an index, exact-scan otherwise. Hits come back already tagged with
    /// `DocAddress`, so the collector doesn't need a second pass to attach
    /// the segment. The segment's [`ProbeStats`] ride along: the IVF path
    /// fills the probe-loop counters, the flat/exact path only
    /// `exact_rows_read`.
    pub fn top_n(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let (hits, stats) = self.top_n_by(
            weight,
            segment_reader,
            top_n,
            &mut NoTieBreak,
            NaturalComparator,
        )?;
        Ok((
            hits.into_iter()
                .map(|((score, ()), address)| (score, address))
                .collect(),
            stats,
        ))
    }

    /// [`Self::top_n`] ordered by `(similarity, tie_break)` rather than
    /// similarity alone.
    ///
    /// The tie-break participates in the heap's eviction decision, not just in
    /// the ordering of what survives: candidates that tie on similarity at the
    /// k/k+1 boundary are separated by `tie_break` before `DocId` is consulted.
    /// Applying a secondary key to the returned rows instead would be too late,
    /// since the losing ties are already gone.
    ///
    /// The similarity component is always compared with [`NaturalComparator`]
    /// ("higher is better"); `tie_comparator` orders the tail alone.
    pub fn top_n_by<K, CTail>(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
    ) -> crate::Result<(TieBreakHits<K>, ProbeStats)>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let mut stats = ProbeStats::default();
        let hits = match self.reader.index() {
            Some(index) => self.approximate_top_n(
                index,
                weight,
                segment_reader,
                top_n,
                tie_break,
                tie_comparator,
                &mut stats,
            )?,
            None => self.exact_top_n(
                weight,
                segment_reader,
                top_n,
                tie_break,
                tie_comparator,
                &mut stats,
            )?,
        };
        Ok((hits, stats))
    }

    /// Flat/exact scan: drain the filter DocSet doc-by-doc, scoring each
    /// survivor from one stride-sized row read. Fills only the
    /// `exact_rows_read` stat.
    fn exact_top_n<K, CTail>(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        // `for_each_no_score` walks the filter DocSet in ascending doc order,
        // which permits the fast `TopNComputer::push` path (valid only under
        // ascending-doc pushes).
        // `NaturalComparator` because similarity is "higher = better" — see
        // the note on `scan_clusters`.
        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        let alive = segment_reader.alive_bitset();
        let mut rows_read = 0usize;
        // Row reads are ranged and can fail; the `for_each` closure can't
        // return an error, so the first one is parked here and re-raised
        // after the walk.
        let mut read_err: Option<TantivyError> = None;
        weight.for_each_no_score(segment_reader, &mut |docs| {
            if read_err.is_some() {
                return;
            }
            for &doc in docs {
                if let Some(bs) = alive {
                    if !bs.is_alive(doc) {
                        continue;
                    }
                }
                let Some(row) = self.reader.row_id(doc) else {
                    continue;
                };
                match self.reader.vector_bytes_for_row(row) {
                    Ok(vbytes) => {
                        rows_read += 1;
                        let score = self.query.score_doc_bytes(&vbytes);
                        if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                            topn.push(key, doc);
                        }
                    }
                    Err(err) => {
                        read_err = Some(err);
                        return;
                    }
                }
            }
        })?;
        if let Some(err) = read_err {
            return Err(err);
        }
        stats.exact_rows_read += rows_read;
        let segment_ord = self.segment_ord;
        Ok(topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect())
    }
}

/// A candidate's composite heap key, or `None` when its similarity alone
/// cannot beat the heap threshold, so the tie-break column is read only for
/// competitive candidates. The caller pushes the key itself, with whichever of
/// [`TopNComputer::push`]/`push_unordered` its doc arrival order permits.
///
/// The skip is exact rather than approximate: `(s, t) < (ts, tt)` requires
/// either `s < ts`, or `s == ts` with `t < tt`. So a candidate rejected here on
/// similarity alone could never have survived the full composite comparison,
/// and the tie-break lookup it would have cost is pure waste. Once the heap has
/// filled this is the common case.
#[inline(always)]
fn tie_break_key<K, CTail>(
    topn: &TieBreakHeap<K, CTail>,
    tie_break: &mut K,
    score: Score,
    doc: DocId,
) -> Option<(Score, K::SegmentSortKey)>
where
    K: SegmentSortKeyComputer,
    CTail: Comparator<K::SegmentSortKey>,
{
    if let Some(((threshold_score, _), _)) = &topn.threshold {
        if score < *threshold_score {
            return None;
        }
    }
    Some((score, tie_break.segment_sort_key(doc, score)))
}

/// How the probe loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
pub enum ProbeTermination {
    /// The work-unit probe budget was spent - the probe ceiling.
    Ceiling,
    /// The ranked centroids were exhausted before the ceiling bound. The
    /// bounds gate never terminates the scan - a skip is per-cluster and
    /// charges the open share; only the ceiling and the stream end it.
    #[default]
    Exhausted,
}

/// Per-segment probe-loop instrumentation: a prune breakdown of every
/// doc the inner loop touched, plus posting-fetch counters. Returned by
/// [`VectorBackend::top_n`] alongside the hits. The flat/exact path fills
/// only `exact_rows_read`; every other field is IVF-probe-only.
#[derive(Debug, Default, serde::Serialize)]
pub struct ProbeStats {
    /// Docs that passed filter + alive + seen and were scored against the
    /// query. This stays the "scored" bucket and equals the final survivor
    /// `candidates`.
    pub candidates_scored: usize,
    /// Every doc-id the inner loop touched, before any gate — the denominator
    /// for the prune breakdown.
    pub vectors_visited: usize,
    /// Touched docs rejected by `filter.contains`.
    pub pruned_filter: usize,
    /// Touched docs rejected by `is_alive`.
    pub pruned_dead: usize,
    /// Touched docs rejected by the replica `seen` dedup.
    pub pruned_seen: usize,
    /// Probed clusters whose surviving rows' posting bytes were fetched —
    /// one stride-sized ranged read per surviving row. Counts clusters,
    /// not rows.
    pub postings_row: usize,
    /// Probed clusters that fetched no posting bytes at all: the
    /// `filter → alive → seen` pre-pass left zero survivors (fully
    /// filtered / dead / already-seen, or the cluster is empty). The two
    /// `postings_*` counters partition the probed clusters:
    /// [`clusters_probed`](Self::clusters_probed) `== postings_row + postings_skipped`.
    pub postings_skipped: usize,
    /// Flat/exact-path stride-sized row reads — one per survivor scored.
    /// Filled only by the exact (non-IVF) path.
    pub exact_rows_read: usize,
    /// Routing cost of ranking the clusters to probe: centroids scored
    /// (`routing.visited_count`), plus the centroid-graph beam counters when
    /// routing went through the RNG. Ranking is lazy, so this covers only as
    /// much routing as the probe loop actually pulled. See
    /// [`IvfSearchMetrics`].
    pub routing: IvfSearchMetrics,
    /// Clusters the bounds gate passed over with a Skip verdict, without
    /// opening them: their margins proved they could not improve the
    /// armed result. Each charged the open share. Disjoint from the
    /// `postings_*` partition, which only counts opened clusters.
    pub bounds_skips: u32,
    /// Probe index (0-based, counting opened clusters) at which the
    /// query bound first armed - the boundary where the heap filled and
    /// margins existed to certify against. `None` = never armed (the
    /// heap never held k results), serialized as JSON null - the
    /// harness's armed-share column depends on the null contract.
    /// Per-segment; does not sum.
    pub bound_armed_at_probe: Option<u32>,
    /// How the probe loop terminated. Per-segment; does not sum.
    pub termination: ProbeTermination,
    /// Work units this segment's probe loop charged against its resolved
    /// budget: opens at `x`, scored rows at `(1 - x)/n_avg`. The
    /// budget identity is per segment:
    /// `budget <= work_charged <= budget + last cluster's charge` on
    /// Ceiling terminations.
    pub work_charged: f32,
}

impl ProbeStats {
    /// Clusters the probe loop visited.
    ///
    /// Returns (`usize`): `postings_row + postings_skipped` — every probed
    /// cluster either fetched survivors or fetched nothing.
    #[inline]
    pub fn clusters_probed(&self) -> usize {
        self.postings_row + self.postings_skipped
    }
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
pub const DEFAULT_FIXED_PROBE_COST_ROWS: f64 = 1.64;

/// Current FIXED_PROBE_COST_ROWS value, stored as f64 bits. See
/// [`DEFAULT_FIXED_PROBE_COST_ROWS`].
static FIXED_PROBE_COST_ROWS_BITS: AtomicU64 =
    AtomicU64::new(DEFAULT_FIXED_PROBE_COST_ROWS.to_bits());

/// Overrides the fixed per-probe cost (the cluster OPEN), in rows of full
/// work. Testing/calibration knob; non-finite or non-positive values reset
/// to [`DEFAULT_FIXED_PROBE_COST_ROWS`].
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
pub(crate) fn fixed_probe_cost_rows() -> f64 {
    f64::from_bits(FIXED_PROBE_COST_ROWS_BITS.load(Relaxed))
}

/// The per-index open share: what fraction of one average cluster's work
/// opening it costs. Covers the open only - routing/search cost is NOT
/// modeled (see [`DEFAULT_FIXED_PROBE_COST_ROWS`]).
///
/// * `n_avg` (`f64`) — native docs per cluster (see [`WorkModel`]).
///
/// Returns (`f64`): `fixed_probe_cost_rows() / (fixed_probe_cost_rows() +
/// n_avg)`, clamped to (0, 0.5] — a share above one half would mean opens
/// dominate rows, which only degenerate sub-2-row clusters produce.
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
    #[inline]
    pub fn new(units: f64) -> WorkUnits {
        WorkUnits(units)
    }

    /// The raw amount, for arithmetic that genuinely leaves the unit.
    ///
    /// Returns (`f64`): the amount, in work units.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }

    /// The single narrowing point, for the telemetry fold.
    ///
    /// Returns (`f32`): the amount, narrowed once for `ProbeStats`.
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
    #[inline]
    fn mul(self, rhs: f64) -> WorkUnits {
        WorkUnits(self.0 * rhs)
    }
}

/// The resolved per-segment prices the probe loop charges against its
/// budget: an open costs `open`, a scored row costs `row`. Built once
/// per segment from [`AdaptiveProbeParams::resolved_work_budget`]'s
/// `(budget, n_avg, x)`.
#[derive(Clone, Copy, Debug)]
struct UnitPricing {
    /// Work this segment may spend before the ceiling binds.
    budget: WorkUnits,
    /// The per-index open share `x`: what opening one cluster costs.
    open: WorkUnits,
    /// `(1 - x)/n_avg`: what one scored row costs.
    row: WorkUnits,
}

/// One gate survivor from the pre-pass over a cluster's rows: `row`
/// indexes into the segment-wide dense rows slot.
#[derive(Clone, Copy)]
struct Survivor {
    row: usize,
    doc: DocId,
}

impl<T: VectorElement> VectorBackend<T> {
    /// Top-N by IVF probe. Fills `stats` with this segment's probe-loop
    /// counters.
    #[allow(clippy::too_many_arguments)]
    fn approximate_top_n<K, CTail>(
        &self,
        index: &IvfIndex,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        if top_n == 0 {
            return Ok(Vec::new());
        }
        let max_doc = segment_reader.max_doc();
        if max_doc == 0 {
            return Ok(Vec::new());
        }

        let filter = build_filter_bitset(weight, segment_reader, max_doc)?;
        if filter.len() == 0 {
            return Ok(Vec::new());
        }
        let alive = segment_reader.alive_bitset();

        let num_centroids = index.num_clusters();
        if num_centroids == 0 {
            return Ok(Vec::new());
        }
        // Capacity counts native docs as WRITTEN (deleted rows still
        // charge on first touch - see `WorkModel::for_searcher`), and the
        // open share x is derived from the index's own n_avg at query
        // init - see `open_share`.
        let (work_budget, n_avg, x) = self
            .adaptive
            .resolved_work_budget(num_centroids, index.num_docs())?;
        debug_assert!(n_avg > 0.0);
        let pricing = UnitPricing {
            budget: WorkUnits::new(work_budget),
            open: WorkUnits::new(x),
            row: WorkUnits::new((1.0 - x) / n_avg),
        };

        // Phase 1: rank the clusters to probe, lazily — the scan below pulls
        // ranked clusters on demand, so routing cost is paid only as far as
        // probing actually reaches. The filter-effective budget can pull far
        // past its nominal cluster count on a selective filter (each passed-
        // over cluster streams few unseen rows), and lazy routing keeps that
        // cheap.
        // Routing operates in `f32` (centroid rows are `f32` today), so the
        // query is widened losslessly per element.
        let query_f32: Vec<f32> = self.query.query().iter().map(|e| e.to_f32()).collect();
        let mut routing_ws = Workspace::new();
        let mut ranked = index.rank_clusters(&mut routing_ws, &query_f32);

        let topn = self.scan_clusters(
            index,
            &mut ranked,
            pricing,
            &filter,
            max_doc,
            alive,
            top_n,
            tie_break,
            tie_comparator,
            &query_f32,
            stats,
        )?;

        // The routing cost is only known once the scan stops pulling.
        stats.routing = ranked.metrics();

        let segment_ord = self.segment_ord;
        Ok(topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect())
    }

    /// Phase 2: the probe loop. Each ranked cluster first passes the
    /// bounds verdict — armed, the cluster's stored bound is collided
    /// with the query bound and a strict-negative margin skips it for
    /// the open share, without touching its rows. A probed cluster is
    /// then gated per row — [`Self::collect_cluster_survivors`] runs
    /// `filter → alive → seen` off the pinned id-map with no posting
    /// bytes in hand — and only the survivors' bytes are fetched, one
    /// stride-sized read per surviving row. Cluster-order arrival of
    /// survivors forbids the ascending-doc shortcut in `push`; use
    /// `push_unordered`.
    ///
    /// Note on `NaturalComparator` (vs the `TopNComputer::new` default):
    /// vector similarity is "higher = better", so we want top-N *largest*
    /// scores. The default `new()` wires `ReverseComparator`, which keeps
    /// top-N *smallest* — correct for ascending-distance metrics but inverted
    /// for our convention.
    ///
    /// `ranked` is pulled lazily, one cluster per probe: with graph routing,
    /// pulling past a converged batch resumes the beam search, so routing
    /// work interleaves with (and is bounded by) probing.
    ///
    /// `#[inline(never)]` so it forms its own flamegraph frame carrying its
    /// `score_doc_bytes` cost.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn scan_clusters<K, CTail>(
        &self,
        index: &IvfIndex,
        ranked: impl Iterator<Item = Candidate>,
        pricing: UnitPricing,
        filter: &BitSet,
        max_doc: DocId,
        alive: Option<&AliveBitSet>,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        routing_query: &[f32],
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHeap<K, CTail>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        // `candidates` is the cumulative scored count that drives the gate; the
        // prune counters accumulate into locals and fold into `ProbeStats` once
        // after the loop, keeping the hot per-doc path free of indirection.
        let mut candidates = 0usize;
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut pruned_seen = 0usize;
        let mut postings_row = 0usize;
        let mut postings_skipped = 0usize;
        let mut bounds_skips = 0u32;
        let mut termination = ProbeTermination::Exhausted;
        // P2: the query bound, maintained at cluster boundaries. The
        // bound-space conversion runs on kth improvement only, inside the
        // tracker.
        let metric = self.query.metric();
        let mut bound_tracker = QueryBoundTracker::new();
        // P4: `||q||` for the dot margin's Cauchy-Schwarz term; once per
        // segment-query.
        let q_norm = norm_squared_wide(self.query.query()).sqrt() as f32;
        let bounds = index.bounds();
        // Replication can place the same doc in several probed clusters; dedup
        // by doc id so a vector is scored at most once.
        let mut seen = BitSet::with_max_value(max_doc);
        // The probed cluster's gate survivors; allocated once, reused
        // across clusters.
        let mut survivors: Vec<Survivor> = Vec::new();
        // f64 accumulation in the loop; f32 only at the telemetry fold.
        let mut work_spent = WorkUnits::ZERO;
        let work_budget = pricing.budget;

        for Candidate { sim, node: cluster } in ranked {
            // Boundary rule: open iff remaining > 0. The tripping pull
            // proves another ranked cluster existed, keeping `Ceiling`
            // distinct from `Exhausted`.
            if work_spent >= work_budget {
                termination = ProbeTermination::Ceiling;
                break;
            }
            let cluster = cluster as usize;

            // P5: the bounds verdict. The bound is consumed only through
            // `Armed` (the heap holds k results) — enforced by the enum;
            // Filling probes, and SATURATED probes arithmetically (+inf
            // margin). The margin closure runs on armed clusters only.
            let qb = bound_tracker.bound();
            let verdict = bounds_verdict(qb, || {
                let QueryBound::Armed { t } = qb else {
                    // `bounds_verdict` never calls the margin while
                    // Filling; +inf keeps even that impossibility
                    // fail-open.
                    return f32::INFINITY;
                };
                // The separation IS the routing key the ranked stream
                // already computed: `to_bound_space` maps the similarity
                // key into the metric's distance space for L2/cosine
                // (the heap-key and routing-key spaces coincide), and
                // dot consumes the raw `q . c` key directly.
                #[cfg(debug_assertions)]
                {
                    // Precondition of every margin: the stream key is the
                    // EXACT centroid similarity — an approximate key
                    // makes a skip unsound.
                    let exact = index.centroid_similarity(cluster, routing_query);
                    debug_assert_eq!(
                        sim, exact,
                        "routing stream key must be the exact centroid similarity"
                    );
                }
                let r = bounds.ball_r(cluster);
                match metric {
                    Metric::L2 | Metric::Cosine => {
                        margin_ball_ball(t, r, to_bound_space(metric, sim.score()))
                    }
                    Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t),
                }
            });
            if let Verdict::Skip = verdict {
                // A skip charges the open share: skips are search work,
                // and free skips break the work identity (validated to
                // +-0.03% in benchmarks). No row work is spent.
                work_spent += pricing.open;
                bounds_skips += 1;
                continue;
            }

            // Event-wise charging, part 1: the open.
            work_spent += pricing.open;

            let rows = index.cluster_range(cluster);

            // Pre-pass: gate off the pinned id-map BEFORE fetching any
            // posting bytes, so only rows that will be scored are read.
            let (v, pf, pd, ps, scored_rows) =
                self.collect_cluster_survivors(rows, filter, alive, &mut seen, &mut survivors);
            visited += v;
            pruned_filter += pf;
            pruned_dead += pd;
            pruned_seen += ps;

            // Event-wise charging, part 2: the rows that survive the
            // pre-pass — exactly the rows fetched and scored below.
            // Rejected and deduped rows charge nothing.
            work_spent += pricing.row * scored_rows as f64;

            if survivors.is_empty() {
                postings_skipped += 1;
            } else {
                postings_row += 1;
                // One stride-sized read per survivor — the unit the
                // pg-backed `Directory` serves zero-copy (see
                // `vector_bytes_for_row`).
                for &Survivor { row, doc } in &survivors {
                    let vbytes = self.reader.vector_bytes_for_row(row)?;
                    let score = self.query.score_doc_bytes(&vbytes);
                    if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                        topn.push_unordered(key, doc);
                    }
                }
            }
            candidates += survivors.len();

            // P2: fold the exact kth into the bound at the cluster
            // boundary. `kth_best` is O(buffer) and force-truncates —
            // results and every counter above are unaffected (truncation
            // only drops already-lost entries and tightens the push
            // threshold, which prunes pushes, not scoring).
            let probe_idx = (postings_row + postings_skipped - 1) as u32;
            let peek = HeapPeek::from_kth(topn.kth_best().map(|(score, _tie)| score));
            bound_tracker.observe(metric, peek, probe_idx);
        }
        // The armed index exists exactly when the bound armed.
        debug_assert!(
            bound_tracker.armed_at_probe().is_some()
                == matches!(bound_tracker.bound(), QueryBound::Armed { .. })
        );

        stats.vectors_visited += visited;
        stats.pruned_filter += pruned_filter;
        stats.pruned_dead += pruned_dead;
        stats.pruned_seen += pruned_seen;
        stats.postings_row += postings_row;
        stats.postings_skipped += postings_skipped;
        stats.candidates_scored += candidates;
        stats.bounds_skips += bounds_skips;
        stats.bound_armed_at_probe = bound_tracker.armed_at_probe();
        stats.termination = termination;
        stats.work_charged += work_spent.to_f32();

        Ok(topn)
    }

    /// Phase 2 pre-pass: run one cluster's rows through the
    /// `filter → alive → seen` gate — off the pinned id-map alone, with no
    /// posting bytes fetched — collecting into `survivors` (cleared first)
    /// the rows to score.
    /// Returns `(visited, pruned_filter, pruned_dead, pruned_seen,
    /// scored_rows)` - the last being the survivor count, which is the
    /// work-unit row-charge basis and equals the partition identity's
    /// `scored` term.
    /// `#[inline(never)]` so per-cluster gate cost forms its own frame,
    /// while the per-row loop stays inlined inside it.
    #[inline(never)]
    fn collect_cluster_survivors(
        &self,
        rows: Range<usize>,
        filter: &BitSet,
        alive: Option<&AliveBitSet>,
        seen: &mut BitSet,
        survivors: &mut Vec<Survivor>,
    ) -> (usize, usize, usize, usize, usize) {
        survivors.clear();
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut pruned_seen = 0usize;
        let mut scored_rows = 0usize;
        for row in rows {
            let doc = self.reader.doc_id_at(row);
            visited += 1;
            // Dedup FIRST, marking on first encounter whatever the later
            // verdicts say, so a replica's second copy is never
            // re-checked (it counts as `pruned_seen`, not the original
            // verdict's bucket). The charge basis is `scored_rows` alone:
            // the partition identity
            // `visited == filter + dead + seen + scored` holds, and only
            // its `scored` term ever charges budget.
            if seen.contains(doc) {
                pruned_seen += 1;
                continue;
            }
            seen.insert(doc);
            if !filter.contains(doc) {
                pruned_filter += 1;
                continue;
            }
            if let Some(bs) = alive {
                if !bs.is_alive(doc) {
                    pruned_dead += 1;
                    continue;
                }
            }
            survivors.push(Survivor { row, doc });
            scored_rows += 1;
        }
        (
            visited,
            pruned_filter,
            pruned_dead,
            pruned_seen,
            scored_rows,
        )
    }
}

/// Drain the filter `DocSet` into a dense BitSet for O(1) random membership
/// testing per cluster doc. The BitSet allocates `max_doc / 8` bytes regardless
/// of filter selectivity — inherent to IVF needing membership tests on
/// out-of-order doc ids. `#[inline(never)]` so it forms its own flamegraph
/// frame; at low selectivity over a large segment this drain is real cost
/// otherwise hidden in the search entry.
#[inline(never)]
fn build_filter_bitset(
    weight: &dyn Weight,
    segment_reader: &SegmentReader,
    max_doc: DocId,
) -> crate::Result<BitSet> {
    let mut filter = BitSet::with_max_value(max_doc);
    weight.for_each_no_score(segment_reader, &mut |docs| {
        for &doc in docs {
            filter.insert(doc);
        }
    })?;
    Ok(filter)
}

#[cfg(test)]
mod tests {
    // ============================================================
    // IVF `top_n` test gate.
    //
    // Built on top of `crate::vector::tests::TestVectorIndex` (the
    // shared fixture) where the geometry fits — the 100-doc grid +
    // selectivity-based labels covers oracle / filter / delete /
    // overflow / zero-K. The handful of tests that need crafted point
    // geometry (the trap case + the result-level candidate-floor
    // demonstration) build a tiny IVF index inline via `build_inline_ivf`
    // and an `InlineClusterer` that's compatible with the batched
    // IvfClusterer trait.
    // ============================================================
    use std::cmp::Ordering;

    use super::*;
    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{
        AllQuery, BitSetDocSet, ConstScorer, EnableScoring, Explanation, Query, Scorer, TermQuery,
    };
    use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
    use crate::vector::tests::{exhaustive_params, TestVectorIndex};
    use crate::vector::{
        IvfCentroids, IvfClusterer, IvfMatrix, IvfMergeSettings, IvfTrainingVectors, IvfVectors,
        NeighborhoodGraphSearchMetrics, SearchTerminationReason, VectorClusterStats, VectorDType,
        VectorInfo, VectorOptions, VectorStorageFormat,
    };
    use crate::{Index, IndexWriter, TantivyDocument};

    const FIXTURE_NUM_DOCS: usize = 100;
    /// Number of centroids the shared fixture uses by default (the
    /// 3×3 `grid2d::centroids()` grid). Used by tests that need an
    /// "exhaustive" probe ceiling.
    const DEFAULT_NUM_CENTROIDS: usize = 9;

    /// Run the full collector path with the given filter and adaptive
    /// params. Returns the global top-K (already merged across
    /// segments) in descending-score / (seg_ord, doc_id) order — the
    /// same order `ground_truth::top_k` uses, so equality checks are
    /// well-defined.
    fn search(
        index: &Index,
        field: Field,
        filter: &dyn Query,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let collector = TopDocs::with_limit(k)
            .order_by_similarity(field, query)
            .with_adaptive_params(params);
        Ok(index
            .reader()?
            .searcher()
            .search(filter, &collector)?
            .results)
    }

    /// Probe-stat helper: run `VectorBackend::top_n` against
    /// the first segment of `index` and return (hits, stats).
    /// The contracts are per-segment, so collecting from segment 0 is
    /// what each assertion is talking about.
    fn run_top_n(
        index: &Index,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let weight = AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            params,
        )?;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_some(),
            "expected IVF storage"
        );
        backend.top_n(weight.as_ref(), segment_reader, k)
    }

    // ---- Inline IVF builder for crafted-geometry tests ----
    //
    // The shared fixture's `grid2d::vectors` lays 100 deterministic
    // points around a 3×3 grid; it doesn't expose a per-doc-vector
    // override. The trap-case and result-level candidate-floor tests
    // need points at specific coordinates, so they build a small IVF
    // index inline via the helper below.

    struct InlineClusterer {
        centroids: Vec<[f32; 2]>,
        replicas: usize,
    }

    impl IvfClusterer for InlineClusterer {
        fn centroid_ratio(&self) -> f32 {
            1.0
        }
        fn training_samples_per_centroid(&self) -> usize {
            2
        }
        fn merge_settings(&self, _total_target_docs: usize) -> crate::Result<IvfMergeSettings> {
            Ok(IvfMergeSettings {
                num_centroids: self.centroids.len(),
                training_samples_per_centroid: self.training_samples_per_centroid(),
                assign_batch_size: self.assign_batch_size(),
                replicas: self.replicas,
            })
        }
        fn train(
            &self,
            options: &VectorOptions,
            _vectors: IvfTrainingVectors,
            num_centroids: usize,
        ) -> crate::Result<IvfCentroids> {
            assert_eq!(options.dim(), 2);
            Ok(IvfCentroids::F32(IvfMatrix {
                values: self
                    .centroids
                    .iter()
                    .take(num_centroids)
                    .flat_map(|c| c.iter().copied())
                    .collect(),
                rows: num_centroids,
                dims: 2,
            }))
        }
        fn assign(
            &self,
            options: &VectorOptions,
            vectors: IvfVectors<'_>,
            centroids: &IvfCentroids,
        ) -> crate::Result<Vec<u32>> {
            assert_eq!(options.dim(), 2);
            let IvfVectors::F32(vectors) = vectors;
            let mut c = [0f32; 2];
            Ok(vectors
                .matrix
                .values
                .chunks_exact(2)
                .map(|v| {
                    let mut best = 0u32;
                    let mut best_d2 = f32::INFINITY;
                    for i in 0..centroids.rows() {
                        centroids.decode_row_into(i, &mut c);
                        let dx = v[0] - c[0];
                        let dy = v[1] - c[1];
                        let d2 = dx * dx + dy * dy;
                        if d2 < best_d2 {
                            best = i as u32;
                            best_d2 = d2;
                        }
                    }
                    best
                })
                .collect())
        }
    }

    /// Build a single-IVF-segment index with the supplied centroids and
    /// labelled docs. Splits docs across two commits so `merge_ivf`
    /// has ≥ 2 source segments to consume. Returns the index plus the
    /// `(embedding, label)` field handles.
    fn build_inline_ivf(
        metric: Metric,
        centroids: &[[f32; 2]],
        docs: &[(&str, [f32; 2])],
        replicas: usize,
    ) -> crate::Result<(Index, Field, Field)> {
        assert!(docs.len() >= 2, "need ≥ 2 docs for ≥ 2 source segments");
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();

        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(InlineClusterer {
                centroids: centroids.to_vec(),
                replicas,
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        let mid = docs.len() / 2;
        for chunk in [&docs[..mid.max(1)], &docs[mid.max(1)..]] {
            for (label, v) in chunk {
                let mut doc = TantivyDocument::new();
                doc.add_text(label_field, label);
                doc.add_vector(embed_field, v.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }
        let segment_ids: Vec<_> = index.searchable_segment_ids()?.into_iter().collect();
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;
        Ok((index, embed_field, label_field))
    }

    /// Decode a stored little-endian `[f32; 2]` row.
    fn decode_2d(bytes: &[u8]) -> [f32; 2] {
        [
            f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        ]
    }

    /// L2-nearest centroid with first-wins tie-break on strict `<` — the
    /// same rule `InlineClusterer::assign` uses for the primary.
    fn nearest_centroid(p: [f32; 2], centroids: &[[f32; 2]]) -> usize {
        let mut best = 0;
        let mut best_d2 = f32::INFINITY;
        for (i, c) in centroids.iter().enumerate() {
            let dx = p[0] - c[0];
            let dy = p[1] - c[1];
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = i;
            }
        }
        best
    }

    /// Docs per centroid in the replication fixture.
    const REPLICATION_N_PER: usize = 6;

    /// Six well-separated centroids (3×2 grid, gap 10) and one label per
    /// doc. Docs sit tightly around their centroid (offsets ≤ 0.05
    /// against the grid gap of 10 — see [`replication_docs`]) so the
    /// primary and the next-nearest replica ranking are unambiguous.
    fn replication_fixture() -> (Vec<[f32; 2]>, Vec<String>) {
        let centroids = vec![
            [0.0f32, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 10.0],
        ];
        let labels = (0..centroids.len() * REPLICATION_N_PER)
            .map(|i| format!("d{i}"))
            .collect();
        (centroids, labels)
    }

    /// The replication fixture's docs: `REPLICATION_N_PER` per centroid,
    /// at offset `(i % REPLICATION_N_PER) * 0.01` along both axes.
    fn replication_docs<'a>(
        centroids: &[[f32; 2]],
        labels: &'a [String],
    ) -> Vec<(&'a str, [f32; 2])> {
        (0..labels.len())
            .map(|i| {
                let c = centroids[i / REPLICATION_N_PER];
                let off = (i % REPLICATION_N_PER) as f32 * 0.01;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect()
    }

    /// Fixed-k replication is additive and, at small centroid counts, EXACT:
    /// the fixture's 6 centroids sit far below the exact-selection threshold
    /// (the search's `ef` budget), so replica cells come from a brute k-NN
    /// scan, not the approximate graph selector — every vector is written into exactly
    /// `min(replicas, num_centroids)` distinct cells: its primary (once) plus
    /// the `replicas - 1` next-nearest centroids. Total posting entries are
    /// exactly `replicas × N`. `replicas == 1` is the identity: every doc in
    /// exactly its primary cluster, no replica selector constructed at all
    /// (`replica_selector` stays `None`) — byte-identical to no replication.
    /// Query results never repeat a doc id (the `seen` dedup).
    ///
    /// Every assertion here is deterministic — no envelopes, no retries.
    #[test]
    fn ivf_fixed_k_replication_is_additive() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let replicas = 3usize;
        assert!(
            centroids.len() >= replicas,
            "fixture needs >= replicas centroids for full fill"
        );

        // A replicated build read back through Ming's cluster iteration:
        // each doc's cluster memberships plus its primary (recomputed from
        // the stored vector).
        struct ReplicatedBuild {
            index: Index,
            embed_field: Field,
            memberships: Vec<Vec<usize>>,
            primaries: Vec<usize>,
        }
        let build_and_read = |replicas: usize| -> crate::Result<ReplicatedBuild> {
            let (index, embed_field, _label) =
                build_inline_ivf(Metric::L2, &centroids, &docs, replicas)?;
            let searcher = index.reader()?.searcher();
            assert_eq!(searcher.segment_readers().len(), 1, "one merged segment");
            let segment_reader = &searcher.segment_readers()[0];
            let vec_reader = segment_reader.vector_index(embed_field)?;
            let ivf = vec_reader.index().expect("expected IVF segment");
            assert_eq!(ivf.num_clusters(), centroids.len());
            let max_doc = segment_reader.max_doc() as usize;
            assert_eq!(max_doc, n, "every fixture doc must survive the merge");
            let mut memberships: Vec<Vec<usize>> = vec![Vec::new(); max_doc];
            for cluster in 0..ivf.num_clusters() {
                for doc in vec_reader
                    .cluster_doc_ids(cluster)
                    .expect("in-bounds cluster")
                {
                    memberships[doc as usize].push(cluster);
                }
            }
            let primaries: Vec<usize> = (0..max_doc)
                .map(|doc| {
                    let bytes = vec_reader
                        .vector_bytes(doc as u32)
                        .expect("readable vector bytes")
                        .expect("stored vector bytes");
                    nearest_centroid(decode_2d(&bytes), &centroids)
                })
                .collect();
            Ok(ReplicatedBuild {
                index,
                embed_field,
                memberships,
                primaries,
            })
        };

        // replicas = 3: exact fill. Per doc — ceiling and fill
        // (exactly min(replicas, num_centroids) = 3 cells), dedup (cells
        // distinct, primary present exactly once). Corpus-wide — total
        // memberships exactly replicas × N.
        let built3 = build_and_read(replicas)?;
        let mut total = 0usize;
        for (doc, cells) in built3.memberships.iter().enumerate() {
            assert_eq!(
                cells.len(),
                replicas,
                "doc {doc}: expected exactly {replicas} cells, got {cells:?}"
            );
            let mut distinct = cells.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(
                distinct.len(),
                replicas,
                "doc {doc}: duplicate cells in {cells:?}"
            );
            assert_eq!(
                cells
                    .iter()
                    .filter(|&&c| c == built3.primaries[doc])
                    .count(),
                1,
                "doc {doc}: primary {} must appear exactly once in {cells:?}",
                built3.primaries[doc]
            );
            total += cells.len();
        }
        assert_eq!(
            total,
            replicas * n,
            "total memberships must be replicas × N"
        );

        // Query-time dedup: a doc sits in several probed clusters, but a
        // search must return each doc id exactly once — and with exhaustive
        // params over an all-alive corpus, all N of them.
        let hits = search(
            &built3.index,
            built3.embed_field,
            &AllQuery,
            vec![10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?;
        assert_eq!(hits.len(), n, "exhaustive top-N must return every doc");
        let mut ids: Vec<_> = hits.iter().map(|(_, addr)| addr.doc_id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "search returned duplicate doc ids");

        // replicas = 1: identity. Every doc lives in exactly one cluster —
        // its primary — which is the byte-level content of the primary-only
        // layout (the id-map and rows are fully determined by these
        // memberships plus the merge's (cluster, doc) sort).
        let built1 = build_and_read(1)?;
        for (doc, cells) in built1.memberships.iter().enumerate() {
            assert_eq!(
                cells.as_slice(),
                &[built1.primaries[doc]],
                "replicas=1: doc {doc} must live only in its primary cluster"
            );
        }
        Ok(())
    }

    /// Replica dedup is counted, exactly: exact small-set selection puts
    /// every doc in exactly `replicas` cells, so exhaustive probing visits
    /// `replicas × N` entries, re-encounters each doc exactly `replicas - 1`
    /// times (`pruned_seen`), scores each exactly once, and the counter
    /// invariant holds with zero filter/dead prunes.
    #[test]
    fn ivf_probe_stats_counts_replica_dedup() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let replicas = 4usize;
        let (index, embed_field, _label) =
            build_inline_ivf(Metric::L2, &centroids, &docs, replicas)?;

        let (_, stats) = run_top_n(
            &index,
            embed_field,
            vec![10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?;
        // Every doc sits in exactly `replicas` probed cells; the
        // `replicas - 1` re-encounters are deduped.
        assert_eq!(
            stats.vectors_visited,
            replicas * n,
            "exhaustive probe must touch every posting entry: {stats:?}"
        );
        assert_eq!(
            stats.pruned_seen,
            (replicas - 1) * n,
            "replica dedup must fire exactly replicas-1 times per doc: {stats:?}"
        );
        // Still scored exactly once each; nothing filtered or dead here.
        assert_eq!(stats.candidates_scored, n);
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
            "visited must equal filter+dead+seen+scored ({stats:?})"
        );
        Ok(())
    }

    /// Re-merging a replicated IVF segment must account in DISTINCT docs, not
    /// posting entries. Regression test for the reader's doc count returning
    /// memberships (rows incl. replicas): with `replicas = 3` the IVF source
    /// used to report 3 × 36 = 108 "vectors" into the next merge's
    /// `vector_count`, tripping the `present_vector_ord == vector_count`
    /// debug_asserts (and, in release, inflating the centroid count and the
    /// training sample interval).
    #[test]
    fn remerge_replicated_segment() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let replicas = 3usize;
        let (index, embed_field, label_field) =
            build_inline_ivf(Metric::L2, &centroids, &docs, replicas)?;

        // Two more docs in a fresh (flat) segment, then merge everything —
        // the replicated IVF segment is now a merge SOURCE.
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for (label, v) in [("extra0", [5.0_f32, 5.0]), ("extra1", [15.0, 5.0])] {
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, label);
            doc.add_vector(embed_field, v.as_slice());
            writer.add_document(doc)?;
        }
        writer.commit()?;
        let segment_ids = index.searchable_segment_ids()?;
        assert_eq!(segment_ids.len(), 2, "IVF segment + fresh flat segment");
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        let total = n + 2;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1, "one merged segment");
        let segment_reader = &searcher.segment_readers()[0];

        // num_vectors reports distinct docs; per-cluster sizes keep
        // membership semantics (each doc exact-fills `replicas` cells here).
        let vec_reader = segment_reader.vector_index(embed_field)?;
        assert_eq!(vec_reader.num_vectors(), total);
        let info = vec_reader.info().expect("vector info");
        assert_eq!(info.format, VectorStorageFormat::Ivf);
        assert_eq!(info.num_vectors, total, "num_vectors counts distinct docs");
        let sizes = vec_reader.cluster_sizes().expect("ivf cluster sizes");
        let memberships: usize = sizes.iter().map(|&s| s as usize).sum();
        assert_eq!(
            memberships,
            replicas * total,
            "per-cluster sizes keep membership semantics"
        );

        // Exhaustive search returns every distinct doc exactly once.
        let hits = search(
            &index,
            embed_field,
            &AllQuery,
            vec![10.0, 10.0],
            total,
            exhaustive_params(centroids.len()),
        )?;
        assert_eq!(hits.len(), total, "exhaustive top-N must return every doc");
        let mut ids: Vec<_> = hits.iter().map(|(_, addr)| addr.doc_id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "search returned duplicate doc ids");
        Ok(())
    }

    /// Merging flat segments with deletes past the clustering threshold: rows
    /// written for since-deleted docs still count toward the sources'
    /// `count()` (tombstones don't rewrite `.vec`), so the alive-doc merge
    /// iteration legitimately comes up short of `vector_count`. The merge
    /// must tolerate that, and the resulting IVF segment must hold — and
    /// count — the alive docs only.
    #[test]
    fn merge_flat_segments_with_deletes_into_ivf() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();

        // Same shape as `build_inline_ivf`, but the two flat source segments
        // stay unmerged so the deletes land BEFORE the clustering merge.
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(InlineClusterer {
                centroids: centroids.clone(),
                replicas: 1,
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let mid = n / 2;
        for chunk in [&docs[..mid], &docs[mid..]] {
            for (label, v) in chunk {
                let mut doc = TantivyDocument::new();
                doc.add_text(label_field, label);
                doc.add_vector(embed_field, v.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }

        // Tombstone docs in BOTH flat sources (d0/d7 in the first commit,
        // d35 in the second), then merge everything into one IVF segment.
        let deleted = ["d0", "d7", "d35"];
        for label in deleted {
            writer.delete_term(Term::from_field_text(label_field, label));
        }
        writer.commit()?;
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        let alive = n - deleted.len();
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1, "one merged segment");
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        let info = vec_reader.info().expect("vector info");
        assert_eq!(info.format, VectorStorageFormat::Ivf, "merge must cluster");
        assert_eq!(info.num_vectors, alive, "deleted docs must not be counted");
        assert_eq!(vec_reader.num_vectors(), alive);

        // Every alive doc comes back exactly once; no deleted label survives.
        let hits = search(
            &index,
            embed_field,
            &AllQuery,
            vec![10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?;
        assert_eq!(hits.len(), alive, "exhaustive top-N must return alive docs");
        let mut seen_labels = std::collections::HashSet::new();
        for (_, addr) in &hits {
            let label = stored_label_at(&index, label_field, *addr)?;
            assert!(
                !deleted.contains(&label.as_str()),
                "deleted doc {label} surfaced in results"
            );
            assert!(seen_labels.insert(label), "duplicate doc in results");
        }
        Ok(())
    }

    /// Merging past the clustering threshold when every doc carrying a
    /// vector for ONE field is deleted, while another field keeps live
    /// vectors. The sources still report `vector_count > 0` for the emptied
    /// field (tombstones don't rewrite `.vec`), so it takes the training
    /// path, collects nothing — and used to `continue` without writing the
    /// field's `.vec`/`.centroids` slots. The live field still wrote, so the
    /// composites existed but the emptied field's slots were missing:
    /// `count()`, `open_column()` and `vector_info()` all failed with
    /// InternalError. The merge must instead write the same empty slots as
    /// the no-vectors-at-all fast path.
    #[test]
    fn merge_deleting_every_doc_of_one_field_writes_empty_ivf() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();

        let mut sb = Schema::builder();
        let doomed_field = sb.add_vector_field(
            "embedding_doomed",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let kept_field = sb.add_vector_field(
            "embedding_kept",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(InlineClusterer {
                centroids: centroids.clone(),
                replicas: 1,
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Even docs carry the doomed field, odd docs the kept one, split
        // across two flat commits so BOTH sources hold doomed vectors.
        let mid = n / 2;
        for (i, (label, v)) in docs.iter().enumerate() {
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, label);
            let field = if i % 2 == 0 { doomed_field } else { kept_field };
            doc.add_vector(field, v.as_slice());
            writer.add_document(doc)?;
            if i + 1 == mid {
                writer.commit()?;
            }
        }
        writer.commit()?;

        // Tombstone every doomed-field doc, then merge everything into one
        // IVF segment.
        for (i, (label, _)) in docs.iter().enumerate() {
            if i % 2 == 0 {
                writer.delete_term(Term::from_field_text(label_field, label));
            }
        }
        writer.commit()?;
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1, "one merged segment");
        let segment_reader = &searcher.segment_readers()[0];

        // The emptied field reads back as a zeroed IVF field — not an error.
        let vec_reader = segment_reader.vector_index(doomed_field)?;
        assert_eq!(vec_reader.num_vectors(), 0);
        let info = vec_reader.info().expect("vector info");
        assert_eq!(
            info,
            VectorInfo {
                format: VectorStorageFormat::Ivf,
                num_vectors: 0,
                num_centroids: Some(0),
                cluster_stats: Some(VectorClusterStats {
                    min_cluster_size: 0,
                    max_cluster_size: 0,
                    avg_cluster_size: 0.0,
                    empty_clusters: 0,
                }),
            },
        );
        let ivf = vec_reader.index().expect("expected IVF storage");
        assert!(vec_reader.is_empty(), "no rows in the emptied field");
        assert_eq!(ivf.num_rows(), 0);
        assert_eq!(ivf.num_clusters(), 0);

        // The live field is untouched: every alive doc is counted and found.
        let kept_count = n / 2;
        assert_eq!(
            segment_reader.vector_index(kept_field)?.num_vectors(),
            kept_count
        );
        let hits = search(
            &index,
            kept_field,
            &AllQuery,
            vec![10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?;
        assert_eq!(hits.len(), kept_count, "kept field returns alive docs");
        Ok(())
    }

    /// Captures `paradedb::ivf_build` log records so a test can read back the
    /// timings line the merge emits.
    struct CaptureLogger;
    static CAPTURED_IVF_BUILD: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    impl log::Log for CaptureLogger {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.target() == "paradedb::ivf_build"
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                CAPTURED_IVF_BUILD
                    .lock()
                    .unwrap()
                    .push(format!("{}", r.args()));
            }
        }
        fn flush(&self) {}
    }
    static CAPTURE_LOGGER: CaptureLogger = CaptureLogger;

    /// The merge emits one parseable `ivf_build timings_ms ...` line per field,
    /// with `replica_knn` non-trivial at `replicas > 1`. Builds a larger index
    /// so the phase timings are measurable, captures the line, and prints it
    /// (run with `--nocapture`) so we can see where build time goes.
    #[test]
    fn ivf_build_emits_timings_log() -> crate::Result<()> {
        let _ = log::set_logger(&CAPTURE_LOGGER);
        log::set_max_level(log::LevelFilter::Info);

        // 200 centroids on a 20×10 grid; ~5000 docs clustered around them.
        let mut centroids: Vec<[f32; 2]> = Vec::new();
        for x in 0..20 {
            for y in 0..10 {
                centroids.push([x as f32 * 10.0, y as f32 * 10.0]);
            }
        }
        let n_per = 25usize;
        let labels: Vec<String> = (0..centroids.len() * n_per)
            .map(|i| format!("d{i}"))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
            .map(|i| {
                let c = centroids[i / n_per];
                let off = (i % n_per) as f32 * 0.05;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect();

        let before = CAPTURED_IVF_BUILD.lock().unwrap().len();
        let _ = build_inline_ivf(Metric::L2, &centroids, &docs, 8)?;
        let lines: Vec<String> = CAPTURED_IVF_BUILD.lock().unwrap()[before..].to_vec();
        let line = lines
            .iter()
            .find(|l| l.contains("ivf_build timings_ms") && l.contains("centroids=200"))
            .expect("expected an ivf_build timings line for the 200-centroid build");
        assert!(line.contains("replicas=8"));
        assert!(line.contains("replica_knn="));
        eprintln!("IVF_BUILD_SAMPLE {line}");
        Ok(())
    }

    // ---- IVF top_n correctness tests ----

    /// Exhaustive probing on a multi-segment IVF index built by the
    /// shared fixture must match the brute-force oracle. Sweep over
    /// several queries and K values to cover ranking + drain edges.
    #[test]
    fn ivf_top_n_brute_force_oracle_l2() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
        for query in [[0.5_f32, 0.5], [9.5, 9.5], [5.0, 0.0], [3.7, 11.2]] {
            for k in [1usize, 3, 6, 10] {
                let expected = index.ground_truth(query, k)?;
                let actual = search(
                    &index.index,
                    index.embedding_field(),
                    &AllQuery,
                    query.to_vec(),
                    k,
                    params.clone(),
                )?;
                assert_eq!(actual, expected, "L2 exhaustive query={query:?} k={k}");
            }
        }
        Ok(())
    }

    /// Same exhaustive correctness, confirming the metric threads
    /// through generically.
    #[test]
    fn ivf_top_n_brute_force_oracle_cosine() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::Cosine)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
        for query in [[1.0_f32, 0.0], [0.0, 1.0], [0.7, 0.3]] {
            for k in [1usize, 3, 6] {
                let expected = index.ground_truth(query, k)?;
                let actual = search(
                    &index.index,
                    index.embedding_field(),
                    &AllQuery,
                    query.to_vec(),
                    k,
                    params.clone(),
                )?;
                assert_eq!(actual, expected, "Cosine exhaustive query={query:?} k={k}");
            }
        }
        Ok(())
    }

    /// Exhaustive-probe correctness for Dot. EXHAUSTIVE-PROBE ONLY by
    /// design: Dot isn't a metric (no triangle inequality), so the IVF
    /// cluster-locality assumption is heuristic for unnormalized dot
    /// and can break on high-magnitude vectors in a far cluster.
    /// Adaptive Dot recall is a benchmark question, deferred. This
    /// test confirms only that `Metric::Dot` threads through the
    /// backend's full top_n loop and matches brute force when every
    /// cluster is visited.
    #[test]
    fn ivf_top_n_brute_force_oracle_dot() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::Dot)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
        for query in [[1.0_f32, 0.0], [2.0, 0.0], [0.5, -0.5]] {
            for k in [1usize, 3, 6] {
                let expected = index.ground_truth(query, k)?;
                let actual = search(
                    &index.index,
                    index.embedding_field(),
                    &AllQuery,
                    query.to_vec(),
                    k,
                    params.clone(),
                )?;
                assert_eq!(actual, expected, "Dot exhaustive query={query:?} k={k}");
            }
        }
        Ok(())
    }

    /// The trap: query closest to centroid A, true NN in cluster B.
    /// Adaptive probing finds it; a 1-cluster probe ceiling must miss. Setup
    /// assertions confirm the geometry is genuinely a trap before
    /// the behavioral check — a slightly-off geometry could trivialize
    /// the test. INLINE because the shared fixture's 100-doc grid
    /// doesn't permit a single misplaced trap doc.
    #[test]
    fn ivf_top_n_trap_case() -> crate::Result<()> {
        let centroids = vec![[0.0_f32, 0.0], [10.0, 10.0]];
        // Two A-side docs far from the [1,1] query; a B-side trap
        // doc at [5, 5.01] just over the perpendicular bisector
        // (x+y=10) so it lands in cluster 1 yet is much closer to
        // the query than any A-side doc.
        let docs = [
            ("far_a", [0.0_f32, -10.0]),
            ("far_a", [-10.0, 0.0]),
            ("trap_b", [5.0, 5.01]),
            ("anchor_b", [10.0, 10.0]),
        ];
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        let query = [1.0_f32, 1.0];

        // Setup assertions.
        //
        // (i) The trap doc is genuinely the true top-1 — without
        // this, "miss" and "find" would be indistinguishable.
        let oracle = ground_truth_top_k(&index, embed_field, Metric::L2, &query, 1)?;
        let trap_doc = stored_label_at(&index, label_field, oracle[0].1)?;
        assert_eq!(trap_doc, "trap_b", "true NN must be the trap doc");

        // (ii) Query's nearest centroid is A (the one at the origin).
        //
        // With the inline IVF building exactly one segment, segment 0
        // holds both centroids. We don't need to open the column
        // directly — the geometry says distance to A = √2 ≈ 1.41,
        // distance to B = √162 ≈ 12.73, so A wins decisively.

        // Behavioral check 1: a probe ceiling of 1 misses the trap.
        let one_probe = AdaptiveProbeParams {
            max_probe_fraction: 0.5,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let hits1 = search(&index, embed_field, &AllQuery, query.to_vec(), 1, one_probe)?;
        assert_eq!(hits1.len(), 1);
        assert_ne!(
            stored_label_at(&index, label_field, hits1[0].1)?,
            "trap_b",
            "a 1-cluster probe ceiling should miss the trap (probes only cluster A)",
        );

        // Behavioral check 2: exhaustive probing finds it.
        let hits2 = search(
            &index,
            embed_field,
            &AllQuery,
            query.to_vec(),
            1,
            exhaustive_params(2),
        )?;
        assert_eq!(hits2.len(), 1);
        assert_eq!(
            stored_label_at(&index, label_field, hits2[0].1)?,
            "trap_b",
            "exhaustive probing should find the trap doc",
        );
        Ok(())
    }

    /// Filter selectivity: only docs in the filter set surface, and
    /// the result equals the oracle restricted to that set. Uses the
    /// shared fixture's `.selectivities(..)` to drop a "selectivity_0.1"
    /// label on the first 10 of 100 docs.
    #[test]
    fn ivf_top_n_filter_selectivity() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .selectivities(&[0.1])
            .build()?;
        let filter = TermQuery::new(
            Term::from_field_text(index.label_field(), "selectivity_0.1"),
            IndexRecordOption::Basic,
        );
        let query = [0.5_f32, 0.5];
        let k = 5;
        // Oracle restricted to the filter set: brute force across the
        // whole index, then keep only the docs that carry the label.
        let filter_set = collect_filter_doc_set(&index.index, &filter)?;
        let mut restricted = ground_truth_top_k(
            &index.index,
            index.embedding_field(),
            Metric::L2,
            &query,
            FIXTURE_NUM_DOCS,
        )?;
        restricted.retain(|(_, addr)| filter_set.contains(addr));
        restricted.truncate(k);

        let actual = search(
            &index.index,
            index.embedding_field(),
            &filter,
            query.to_vec(),
            k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(actual, restricted);
        for (_, addr) in &actual {
            assert!(filter_set.contains(addr), "hit outside filter: {addr:?}");
        }
        Ok(())
    }

    /// Empty filter returns empty results, no panic.
    #[test]
    fn ivf_top_n_empty_filter() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        // No doc carries "absent" — the term query yields an empty
        // DocSet.
        let empty = TermQuery::new(
            Term::from_field_text(index.label_field(), "absent"),
            IndexRecordOption::Basic,
        );
        let hits = search(
            &index.index,
            index.embedding_field(),
            &empty,
            vec![0.0_f32, 0.0],
            5,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert!(hits.is_empty());
        Ok(())
    }

    /// K > total candidates: returns all docs in descending order,
    /// no panic.
    #[test]
    fn ivf_top_n_k_exceeds_candidates() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let query = [0.0_f32, 0.0];
        let big_k = FIXTURE_NUM_DOCS + 50;
        let expected = index.ground_truth(query, big_k)?;
        let actual = search(
            &index.index,
            index.embedding_field(),
            &AllQuery,
            query.to_vec(),
            big_k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(actual.len(), FIXTURE_NUM_DOCS);
        assert_eq!(actual, expected);
        Ok(())
    }

    /// Deletes: a doc marked deleted must never appear, even if it
    /// would otherwise rank top-K. Confirms the IVF backend's separate
    /// alive-check (the filter bitmap doesn't carry delete info).
    #[test]
    fn ivf_top_n_respects_deletes() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .selectivities(&[0.1])
            .build()?;
        // Delete every doc carrying the 0.1-selectivity label — the
        // 10 docs nearest to the grid's origin centroid by
        // construction (they're inserted first).
        {
            let mut writer: IndexWriter = index.index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            writer.delete_term(Term::from_field_text(
                index.label_field(),
                "selectivity_0.1",
            ));
            writer.commit()?;
        }

        // Oracle restricted to the surviving docs.
        let query = [0.0_f32, 0.0];
        let searcher = index.index.reader()?.searcher();
        let mut alive_addrs = std::collections::HashSet::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            let alive = segment_reader.alive_bitset();
            for doc in 0..segment_reader.max_doc() {
                let is_alive = alive.is_none_or(|bs| bs.is_alive(doc));
                if is_alive {
                    alive_addrs.insert(DocAddress::new(seg_ord as u32, doc));
                }
            }
        }
        assert!(
            alive_addrs.len() < FIXTURE_NUM_DOCS,
            "delete didn't remove anything (alive={})",
            alive_addrs.len(),
        );
        let k = 10;
        let mut expected = ground_truth_top_k(
            &index.index,
            index.embedding_field(),
            Metric::L2,
            &query,
            FIXTURE_NUM_DOCS,
        )?;
        expected.retain(|(_, addr)| alive_addrs.contains(addr));
        expected.truncate(k);

        let actual = search(
            &index.index,
            index.embedding_field(),
            &AllQuery,
            query.to_vec(),
            k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(actual, expected);
        for (_, addr) in &actual {
            assert!(
                alive_addrs.contains(addr),
                "deleted doc {addr:?} surfaced in results",
            );
        }
        Ok(())
    }
    /// `top_n == 0` returns empty without touching the column. The
    /// collector layer rejects `TopDocs::with_limit(0)` before it
    /// reaches the backend, so this test calls the backend directly
    /// via the instrumented seam — the short-circuit lives in
    /// `approximate_top_n`.
    #[test]
    fn ivf_top_n_zero_returns_empty() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let (hits, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            0,
            AdaptiveProbeParams::default(),
        )?;
        assert!(hits.is_empty());
        // Short-circuit fires before the probe loop, so no clusters
        // visited and no candidates scored.
        assert_eq!(stats.clusters_probed(), 0);
        assert_eq!(stats.candidates_scored, 0);
        Ok(())
    }

    /// Smoke for the instrumented seam: every centroid is probed under
    /// exhaustive params, and candidates_scored ≤ total docs in the
    /// inspected segment. Exhaustive params on a 9-centroid segment
    /// visit all 9.
    #[test]
    fn ivf_top_n_collects_probe_stats() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        // k = 64 exceeds any segment's doc count, so the query bound
        // never arms and the bounds gate skips nothing - every cluster
        // is probed and every counter equality below is exact.
        let (_, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            64,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(stats.clusters_probed(), DEFAULT_NUM_CENTROIDS);
        // The first segment has docs distributed across all 9 clusters;
        // candidates_scored equals the segment's doc count under
        // exhaustive probe + AllQuery.
        let segment_doc_count =
            index.index.reader()?.searcher().segment_readers()[0].max_doc() as usize;
        assert_eq!(stats.candidates_scored, segment_doc_count);

        // Counter invariant: every touched doc lands in exactly one bucket.
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
            "visited must equal filter+dead+seen+scored ({stats:?})"
        );
        // Navigation cost == the centroids ranked for this query.
        assert_eq!(stats.routing.visited_count, DEFAULT_NUM_CENTROIDS);
        // Exhaustive params (unclamped ceiling, unsatisfiable floor)
        // drain the ranked list.
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        Ok(())
    }

    /// A `max_probe_fraction` resolving below the cluster count forces the
    /// hard ceiling: the loop stops with `termination == Ceiling`, having
    /// probed exactly the cap, and the counter invariant still holds. Uses the
    /// deterministic `build_inline_ivf` fixture (fixed 6 centroids) so the
    /// cutoff is stable.
    #[test]
    fn ivf_probe_stats_termination_ceiling() -> crate::Result<()> {
        let centroids = [
            [0.0f32, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 10.0],
        ];
        let n_per = 6usize;
        let labels: Vec<String> = (0..centroids.len() * n_per)
            .map(|i| format!("d{i}"))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
            .map(|i| {
                let c = centroids[i / n_per];
                let off = (i % n_per) as f32 * 0.01;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;

        // Cap 1 → ceiling at the first probe; an unsatisfiable survivor
        // floor keeps the gate from firing first.
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let (_, stats) = run_top_n(&index, embed_field, vec![10.0, 10.0], 3, params)?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        // Stopped at exactly the cap, short of the ranked list.
        assert_eq!(stats.clusters_probed(), 1);
        assert_eq!(stats.routing.visited_count, centroids.len());
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
            "visited must equal filter+dead+seen+scored ({stats:?})"
        );
        Ok(())
    }

    /// A single-centroid IVF merge skips the `.centroids` graph slot
    /// (nothing to route between), so the reader must take the linear
    /// fallback: rank the lone cluster without a graph and still return the
    /// exact top-K.
    #[test]
    fn ivf_single_centroid_routes_without_graph() -> crate::Result<()> {
        let centroids = [[0.0f32, 0.0]];
        let labels: Vec<String> = (0..5).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, [f32; 2])> = (0..5)
            .map(|i| (labels[i].as_str(), [i as f32 * 0.01, 0.0]))
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;

        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_some(),
            "expected IVF storage"
        );

        let query = [0.0f32, 0.0];
        let k = 3;
        let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, k)?;
        let (hits, stats) = run_top_n(
            &index,
            embed_field,
            query.to_vec(),
            k,
            AdaptiveProbeParams::default(),
        )?;
        assert_eq!(hits, expected, "linear fallback must match the oracle");
        assert_eq!(stats.clusters_probed(), 1, "one cluster, one probe");
        assert_eq!(stats.routing.visited_count, 1);
        Ok(())
    }

    /// When the probe ceiling is below the cluster count, cluster ranking
    /// routes via the persisted RNG instead of scanning every centroid. With
    /// 16 well-separated clusters and only 8 beam seeds, the router must
    /// still navigate to the true nearest cluster: the routed top-K equals
    /// the brute-force oracle, and the recorded navigation cost is the
    /// beam-visited count, not a full scan of the ranked list.
    #[test]
    fn ivf_routed_ranking_matches_oracle_on_separated_clusters() -> crate::Result<()> {
        // 4×4 grid of well-separated centroids (spacing 10), 4 docs each,
        // tightly packed around their centroid so each query's true top-K
        // lives entirely in one cluster.
        let side = 4usize;
        let centroids: Vec<[f32; 2]> = (0..side * side)
            .map(|i| [(i % side) as f32 * 10.0, (i / side) as f32 * 10.0])
            .collect();
        let n_per = 4usize;
        let labels: Vec<String> = (0..centroids.len() * n_per)
            .map(|i| format!("d{i}"))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
            .map(|i| {
                let c = centroids[i / n_per];
                let off = (i % n_per) as f32 * 0.01;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;

        // The merged segment must carry the routing graph, and cap 2 (< 16
        // clusters) must engage it.
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        let ivf = vec_reader.index().expect("expected IVF segment");
        assert_eq!(ivf.num_clusters(), centroids.len());

        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let k = 3usize;
        for (ord, centroid) in centroids.iter().enumerate().step_by(3) {
            let query = [centroid[0] + 0.3, centroid[1] - 0.2];
            let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, k)?;
            let (hits, stats) = run_top_n(&index, embed_field, query.to_vec(), k, params.clone())?;
            assert_eq!(hits, expected, "routed top-{k} near centroid {ord}");
            assert!(
                stats.clusters_probed() <= 2,
                "cap 2 must bound the probes, got {}",
                stats.clusters_probed()
            );
            assert!(
                stats.routing.visited_count <= centroids.len(),
                "navigation cost is the beam-visited count"
            );
        }
        Ok(())
    }

    /// The raw per-cluster sizes from the reader's `cluster_sizes` must be
    /// exactly the un-collapsed array behind `info`'s aggregate cluster stats
    /// — the invariant `paradedb.ivf_cluster_sizes` relies on to reconcile
    /// with `paradedb.index_info`. Flat segments expose no sizes.
    #[test]
    fn ivf_cluster_sizes_match_vector_info() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let field = index.embedding_field();
        let searcher = index.index.reader()?.searcher();

        let mut segments_checked = 0;
        for segment_reader in searcher.segment_readers() {
            let vec_reader = segment_reader.vector_index(field)?;
            let sizes = vec_reader
                .cluster_sizes()
                .expect("ivf segment exposes cluster sizes");
            let info = vec_reader.info().expect("vector info");
            assert_eq!(info.format, VectorStorageFormat::Ivf);
            let stats = info.cluster_stats.expect("ivf cluster stats");

            // count == num_centroids, and every aggregate index_info reports is
            // reproducible from the raw array.
            assert_eq!(sizes.len(), info.num_centroids.expect("ivf centroids"));
            let sum: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
            let min = sizes.iter().copied().min().unwrap() as usize;
            let max = sizes.iter().copied().max().unwrap() as usize;
            let empty = sizes.iter().filter(|&&s| s == 0).count();
            let avg = sum as f64 / sizes.len() as f64;

            // Per-cluster sizes count posting rows (memberships)...
            let ivf = vec_reader.index().expect("expected IVF segment");
            assert_eq!(sum as usize, ivf.num_rows(), "sizes sum to rows");
            // ...and the shared fixture runs replicas=1 with no deletes, so
            // the row total coincides with the distinct-doc `num_vectors`.
            assert_eq!(sum as usize, info.num_vectors, "replicas=1: rows == docs");
            assert_eq!(min, stats.min_cluster_size, "min");
            assert_eq!(max, stats.max_cluster_size, "max");
            assert_eq!(empty, stats.empty_clusters, "empty");
            assert!(
                (avg - stats.avg_cluster_size).abs() < 1e-9,
                "avg {avg} vs {}",
                stats.avg_cluster_size
            );
            segments_checked += 1;
        }
        assert!(
            segments_checked > 0,
            "fixture must produce >= 1 IVF segment"
        );

        // Flat segments (no IVF data) yield None — the SRF emits no rows for them.
        let flat = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Flat)
            .build()?;
        let flat_field = flat.embedding_field();
        let flat_searcher = flat.index.reader()?.searcher();
        for segment_reader in flat_searcher.segment_readers() {
            assert!(
                segment_reader
                    .vector_index(flat_field)?
                    .cluster_sizes()
                    .is_none(),
                "flat segments expose no cluster sizes"
            );
        }
        Ok(())
    }

    // ============================================================
    // Adaptive-probing parameter contracts.
    //
    // The stop condition couples the three adaptive knobs. Each test
    // below holds the others permissive so one becomes the binding
    // constraint, then asserts the contract implied by the knob's
    // definition — exact only where the knob pins it (the absolute
    // ceiling), an inequality otherwise.
    // ============================================================

    // ============================================================
    // Work-unit budget properties. Unprimed (single-segment) runs use the
    // segment-local n_avg, under which units_seg is exactly C_seg - the
    // normalization identity, per segment. Every test here parks the
    // distance-ratio gate: K > N keeps the resolved floor
    // (top_n + overfetch_margin) unreachable, and a huge epsilon makes the
    // threshold vacuous - the stop point under test is the budget's alone.
    // ============================================================

    /// Full-budget params with the gate parked - the shared configuration
    /// for the budget properties.
    fn budget_only_params() -> AdaptiveProbeParams {
        AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 1,
            ..Default::default()
        }
    }

    /// An exhaustive scan charges `C*x + (1 - x)*N/n_avg = exactly C`
    /// units - capacity is the cluster count, whatever the size skew.
    #[test]
    fn unit_normalization_exact() -> crate::Result<()> {
        // Uneven sizes on purpose: [5, 2, 2, 1] docs across 4 clusters.
        let centroids = vec![[0.0_f32, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]];
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for (c, count) in [(0usize, 5usize), (1, 2), (2, 2), (3, 1)] {
            for i in 0..count {
                docs.push((
                    format!("d{c}_{i}"),
                    [centroids[c][0] + i as f32 * 0.01, 0.0],
                ));
            }
        }
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        let (_, stats) = run_top_n(
            &index,
            embed_field,
            vec![0.0, 0.0],
            11,
            budget_only_params(),
        )?;
        let c = centroids.len() as f32;
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        assert!(
            (stats.work_charged - c).abs() <= 1e-6 * c,
            "an exhaustive scan must charge exactly C units: {stats:?}"
        );
        Ok(())
    }

    /// Replicas charge nothing on re-encounter: at replicas = 2 a full
    /// scan still charges exactly C units - every doc costs one
    /// row-deduction, whichever copy arrives first - and every second copy
    /// lands in `pruned_seen`.
    #[test]
    fn replicas_charge_once() -> crate::Result<()> {
        let centroids = vec![[0.0_f32, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]];
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for (c, count) in [(0usize, 4usize), (1, 2), (2, 2), (3, 2)] {
            for i in 0..count {
                docs.push((
                    format!("d{c}_{i}"),
                    [centroids[c][0] + i as f32 * 0.01, 0.0],
                ));
            }
        }
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let n_docs = docs.len();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 2)?;
        let (_, stats) = run_top_n(
            &index,
            embed_field,
            vec![0.0, 0.0],
            11,
            budget_only_params(),
        )?;
        let c = centroids.len() as f32;
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        assert!(
            (stats.work_charged - c).abs() <= 1e-6 * c,
            "replica copies must not re-charge: {stats:?}"
        );
        assert_eq!(
            stats.pruned_seen, n_docs,
            "each doc's second copy prunes as seen: {stats:?}"
        );
        Ok(())
    }

    /// Skew charges proportionally: a 30-doc cluster consumes most of the
    /// budget a cluster-count budget would spread over five clusters, and
    /// the stop point shifts - hand-verified numbers. Also pins the
    /// overshoot bound: the final overrun is at most the last cluster's
    /// charge.
    #[test]
    fn imbalance_charges_proportionally() -> crate::Result<()> {
        // Sizes [30, 2, 2, 2, 2, 2]: C = 6, N = 40, n_avg = 20/3.
        // Per-index x = 1.64/(1.64 + 20/3) ~ 0.1975; row_charge =
        // (1 - x)*3/20 ~ 0.1204. units_seg = 6x + (1 - x)*40/(20/3) =
        // 1.185 + 4.815 = 6.0 (identity). The query sits on the big
        // centroid, so it opens first: charge 0.1975 + 30*0.1204 ~ 3.809.
        // Each small cluster charges 0.1975 + 2*0.1204 ~ 0.438. At
        // f = 0.8 (budget 4.8): big -> 3.809, small -> 4.247, small ->
        // 4.685, small -> 5.123 >= 4.8 at the next boundary, so 4 clusters
        // are probed. A cluster-count budget at f = 0.8 would probe
        // ceil(4.8) = 5.
        let centroids = vec![
            [0.0_f32, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
            [50.0, 0.0],
        ];
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for (c, count) in [(0usize, 30usize), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)] {
            for i in 0..count {
                docs.push((
                    format!("d{c}_{i}"),
                    [centroids[c][0] + i as f32 * 0.001, 0.0],
                ));
            }
        }
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.8,
            ..budget_only_params()
        };
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        let (_, stats) = run_top_n(&index, embed_field, vec![0.0, 0.0], 41, params)?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling, "{stats:?}");
        assert_eq!(
            stats.clusters_probed(),
            4,
            "the big cluster eats the budget a count regime spreads over 5: {stats:?}"
        );
        assert!(
            (stats.work_charged - 5.123).abs() < 2e-3,
            "hand-computed spend: {stats:?}"
        );
        // Overshoot bound: the overrun is at most the last (small)
        // cluster's charge.
        let budget = 0.8f32 * 6.0;
        let last_charge = 0.1975 + 2.0 * 0.1204;
        assert!(
            stats.work_charged > budget && stats.work_charged - budget <= last_charge + 1e-4,
            "overshoot bounded by the last cluster's charge: {stats:?}"
        );
        Ok(())
    }

    /// A cluster's row charge tracks the rows it actually READ AND SCORED,
    /// not the rows it walked past. Same fixture, same opens, two runs:
    /// unfiltered (every row scored) and filtered to 3 of 22 docs. The
    /// filtered run must charge 19 row-shares less - the rows the filter
    /// rejected are never fetched, so they cost nothing. Under a
    /// first-seen basis the two runs would charge identically, since both
    /// walk all 22 rows and mark all 22 seen.
    #[test]
    fn filtered_rows_are_not_charged() -> crate::Result<()> {
        // Sizes [20, 2]: C = 2, N = 22, n_avg = 11.
        let centroids = vec![[0.0_f32, 0.0], [50.0, 0.0]];
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for (c, count) in [(0usize, 20usize), (1, 2)] {
            for i in 0..count {
                docs.push((
                    format!("d{c}_{i}"),
                    [centroids[c][0] + i as f32 * 0.001, 0.0],
                ));
            }
        }
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;

        let n_avg = 22.0 / 2.0;
        let fixed = fixed_probe_cost_rows();
        let x = fixed / (fixed + n_avg);
        let row = (1.0 - x) / n_avg;

        // Unfiltered: every cluster opened, every row scored - the
        // identity's reference case, so exactly C = 2 units.
        let (_, full) = run_top_n(
            &index,
            embed_field,
            vec![0.0, 0.0],
            23,
            budget_only_params(),
        )?;
        assert_eq!(full.clusters_probed(), 2, "{full:?}");
        assert_eq!(full.candidates_scored, 22, "{full:?}");
        assert!(
            (full.work_charged as f64 - 2.0).abs() < 1e-5,
            "unfiltered exhaustive scan charges exactly C: {full:?}"
        );

        // Admit 3 docs of the big cluster; the other 19 rows are walked
        // and rejected, and cluster 1's 2 rows are walked and rejected.
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let admitted: Vec<DocId> = segment_reader
            .vector_index(embed_field)?
            .cluster_doc_ids(0)
            .expect("cluster 0 doc ids")
            .into_iter()
            .take(3)
            .collect();
        assert_eq!(admitted.len(), 3, "setup: need 3 admitted docs");
        let weight = FixedDocsWeight {
            max_doc: segment_reader.max_doc(),
            docs: admitted,
        };
        drop(searcher);

        let (_, filtered) = run_top_n_with_weight(
            &index,
            embed_field,
            vec![0.0, 0.0],
            23,
            budget_only_params(),
            &weight,
        )?;
        // Same opens, same rows walked - only the scored count differs.
        assert_eq!(filtered.clusters_probed(), 2, "{filtered:?}");
        assert_eq!(filtered.vectors_visited, 22, "{filtered:?}");
        assert_eq!(filtered.candidates_scored, 3, "{filtered:?}");

        let expected = 2.0 * x + 3.0 * row;
        assert!(
            (filtered.work_charged as f64 - expected).abs() < 1e-5,
            "charge must be 2 opens + 3 scored rows ({expected}): {filtered:?}"
        );
        // The 19 rejected rows cost exactly nothing.
        assert!(
            ((full.work_charged - filtered.work_charged) as f64 - 19.0 * row).abs() < 1e-5,
            "the filtered rows must account for the entire difference: full={full:?} \
             filtered={filtered:?}"
        );
        Ok(())
    }

    /// A budget below capacity binds, is attributed to the ceiling, and
    /// overshoots by at most one cluster's charge - the boundary rule on
    /// a real fixture rather than a hand-built one. The distance-ratio
    /// gate is parked (floor unreachable), so the stop point under test
    /// is the budget's alone.
    #[test]
    fn probe_stats_max_probe_fraction_ceiling() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.2,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let searcher = index.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(index.embedding_field())?;
        let ivf = vec_reader.index().expect("expected IVF storage");
        let (clusters, docs) = (ivf.num_clusters(), ivf.num_docs());
        let (budget, n_avg, x) = params.resolved_work_budget(clusters, docs)?;
        assert!(budget < clusters as f64, "setup: the budget must bind");
        drop(searcher);

        let (_, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            3,
            params,
        )?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert!(
            stats.clusters_probed() < clusters,
            "the budget must bind before exhaustion: {stats:?}"
        );
        assert!(
            stats.work_charged as f64 > budget,
            "the ceiling fires only once the budget is spent: {stats:?}"
        );
        // Overshoot bound: no single cluster can charge more than one
        // open plus every row in the segment.
        let max_cluster_charge = x + docs as f64 * (1.0 - x) / n_avg;
        assert!(
            stats.work_charged as f64 <= budget + max_cluster_charge + 1e-6,
            "overshoot is bounded by the last cluster's charge: {stats:?}"
        );
        Ok(())
    }

    /// `ProbeStats` (and nested routing / optional graph metrics) round-trip
    /// through `serde_json` with the field names callers rely on.
    #[test]
    fn probe_stats_serializes_to_json() {
        let stats = ProbeStats {
            candidates_scored: 10,
            vectors_visited: 20,
            pruned_filter: 4,
            pruned_dead: 3,
            pruned_seen: 3,
            postings_row: 1,
            postings_skipped: 1,
            exact_rows_read: 0,
            routing: IvfSearchMetrics {
                visited_count: 7,
                graph: Some(NeighborhoodGraphSearchMetrics {
                    visited_count: 7,
                    expanded_count: 4,
                    edges_scanned: 12,
                    evictions: 1,
                    result_count: 3,
                    termination_reason: SearchTerminationReason::SearchConverged,
                }),
            },
            bounds_skips: 2,
            bound_armed_at_probe: Some(1),
            termination: ProbeTermination::Ceiling,
            work_charged: 1.75,
        };

        let value = serde_json::to_value(&stats).expect("ProbeStats should serialize to JSON");
        assert_eq!(
            value,
            serde_json::json!({
                "candidates_scored": 10,
                "vectors_visited": 20,
                "pruned_filter": 4,
                "pruned_dead": 3,
                "pruned_seen": 3,
                "postings_row": 1,
                "postings_skipped": 1,
                "exact_rows_read": 0,
                "routing": {
                    "visited_count": 7,
                    "graph": {
                        "visited_count": 7,
                        "expanded_count": 4,
                        "edges_scanned": 12,
                        "evictions": 1,
                        "result_count": 3,
                        "termination_reason": "SearchConverged"
                    }
                },
                "bounds_skips": 2,
                "bound_armed_at_probe": 1,
                "termination": "Ceiling",
                "work_charged": 1.75
            })
        );
        assert_eq!(stats.clusters_probed(), 2);

        // Exact routing leaves `graph` unset — still must serialize as null.
        let mut exact_routing = stats;
        exact_routing.routing.graph = None;
        let exact_value =
            serde_json::to_value(&exact_routing).expect("ProbeStats should serialize to JSON");
        assert_eq!(exact_value["routing"]["graph"], serde_json::Value::Null);
    }

    // ============================================================
    // Filter-aware posting fetches.
    //
    // The probe loop decides each cluster's survivors from the pinned
    // id-map BEFORE touching posting bytes, then fetches survivors with
    // one stride-sized read per row — or nothing at all when the gate
    // leaves no survivors. These tests pin the skip/fetch behavior and
    // that the two `postings_*` counters partition the probed clusters.
    // ============================================================

    /// A `Weight` over a fixed, hand-built doc-id set, so tests can hand
    /// the backend an exact filter BitSet without routing it through a
    /// real query.
    struct FixedDocsWeight {
        max_doc: DocId,
        docs: Vec<DocId>,
    }

    impl Weight for FixedDocsWeight {
        fn scorer(&self, _reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
            let mut bs = BitSet::with_max_value(self.max_doc);
            for &doc in &self.docs {
                bs.insert(doc);
            }
            Ok(Box::new(ConstScorer::new(BitSetDocSet::from(bs), boost)))
        }

        fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> crate::Result<Explanation> {
            unreachable!("the vector backend never explains filter docs")
        }
    }

    /// Like [`run_top_n`] but with a caller-supplied filter
    /// weight.
    fn run_top_n_with_weight(
        index: &Index,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
        weight: &dyn Weight,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            params,
        )?;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_some(),
            "expected IVF storage"
        );
        backend.top_n(weight, segment_reader, k)
    }

    /// Every touched row lands in exactly one prune bucket.
    fn assert_stats_identities(stats: &ProbeStats) {
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
            "visited must equal filter+dead+seen+scored ({stats:?})"
        );
    }

    /// Build a single-segment FLAT index (one commit, never merged past
    /// the clustering threshold): `docs` are `(label, Some(vector))`, or
    /// `(label, None)` for vectorless docs — mixing the two forces the
    /// `Bitmap` id-map.
    fn build_flat(
        dim: usize,
        docs: &[(&str, Option<Vec<f32>>)],
    ) -> crate::Result<(Index, Field, Field)> {
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(dim, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let index = Index::create_in_ram(sb.build());
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for (label, v) in docs {
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, label);
            if let Some(v) = v {
                doc.add_vector(embed_field, v.as_slice());
            }
            writer.add_document(doc)?;
        }
        writer.commit()?;
        Ok((index, embed_field, label_field))
    }

    /// Run the flat/exact path against `segment_reader` with a
    /// caller-supplied filter weight.
    fn run_exact_on_segment(
        segment_reader: &SegmentReader,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        weight: &dyn Weight,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            AdaptiveProbeParams::default(),
        )?;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_none(),
            "expected flat storage"
        );
        backend.top_n(weight, segment_reader, k)
    }

    /// Filter-aware fetches across selectivities: on the replicated
    /// multi-cluster fixture, hand-built filters admitting {0, 1, 50,
    /// 100}% of docs return every admitted doc exactly once, with both
    /// partition identities intact. 0% admits nothing — the empty-filter
    /// short-circuit returns before the probe loop, so zero clusters
    /// probe and zero fetches happen (postings_skipped == clusters
    /// probed == 0).
    #[test]
    fn filter_aware_fetch_across_selectivities() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 3)?;
        let params = exhaustive_params(centroids.len());

        for pct in [0usize, 1, 50, 100] {
            // Admit the first ceil(pct% · n) doc ids: doc · 100 < pct · n.
            let weight = FixedDocsWeight {
                max_doc: n as DocId,
                docs: (0..n as DocId)
                    .filter(|&doc| (doc as usize) * 100 < pct * n)
                    .collect(),
            };
            let admitted = weight.docs.len();
            let (hits, stats) = run_top_n_with_weight(
                &index,
                embed_field,
                vec![10.0, 10.0],
                n,
                params.clone(),
                &weight,
            )?;

            assert_eq!(
                hits.len(),
                admitted,
                "{pct}%: every admitted doc returns exactly once"
            );
            assert_stats_identities(&stats);

            match pct {
                0 => {
                    assert_eq!(stats.postings_row, 0, "0%: no fetches");
                    assert_eq!(
                        stats.postings_skipped,
                        stats.clusters_probed(),
                        "0%: every probed cluster skips its fetch"
                    );
                }
                1 => {
                    // One admitted doc → exactly one cluster fetches (the
                    // first-probed of its replica cells); every other
                    // probed cluster skips.
                    assert_eq!(stats.postings_row, 1, "{stats:?}");
                    assert_eq!(
                        stats.postings_skipped,
                        stats.clusters_probed() - 1,
                        "{stats:?}"
                    );
                }
                _ => {
                    assert!(stats.postings_row > 0, "{stats:?}");
                }
            }
        }
        Ok(())
    }

    /// Replica dedup is decided in the pre-pass: a doc whose copies land
    /// in several probed clusters is fetched and scored exactly once, and
    /// its later cells (holding no other survivor) skip their fetches
    /// entirely.
    #[test]
    fn filter_aware_fetch_preserves_replica_dedup() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let replicas = 3usize;
        let (index, embed_field, _label) =
            build_inline_ivf(Metric::L2, &centroids, &docs, replicas)?;
        let params = exhaustive_params(centroids.len());

        // Admit exactly one doc; exact small-N replica selection puts its
        // copies in exactly `replicas` distinct cells, all probed under
        // exhaustive params.
        let weight = FixedDocsWeight {
            max_doc: n as DocId,
            docs: vec![0],
        };
        let (hits, stats) =
            run_top_n_with_weight(&index, embed_field, vec![10.0, 10.0], n, params, &weight)?;

        assert_eq!(hits.len(), 1, "the admitted doc must return exactly once");
        assert_eq!(stats.candidates_scored, 1, "scored on first encounter only");
        // First-touch marking (the work-unit charge basis): EVERY doc
        // marks `seen` on its first structural encounter, filter verdict
        // irrelevant - so all replica re-encounters land in `pruned_seen`
        // (visited * (replicas - 1)/replicas), not just the admitted
        // doc's.
        assert_eq!(
            stats.pruned_seen,
            stats.vectors_visited * (replicas - 1) / replicas,
            "every later cell prunes as seen: {stats:?}"
        );
        // Only the first-probed cell fetches anything.
        assert_eq!(stats.postings_row, 1, "{stats:?}");
        assert_eq!(stats.postings_skipped, stats.clusters_probed() - 1);
        assert_stats_identities(&stats);
        Ok(())
    }

    /// Deletes are decided in the pre-pass: a cluster whose rows are all
    /// dead yields zero survivors and fetches nothing.
    #[test]
    fn filter_aware_fetch_skips_all_dead_clusters() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        // replicas = 1, so cluster 0's rows are exactly its 6 primary
        // docs — deleting those leaves a fully-dead cluster with no
        // replica rows from elsewhere.
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        {
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for i in 0..REPLICATION_N_PER {
                writer.delete_term(Term::from_field_text(label_field, &format!("d{i}")));
            }
            writer.commit()?;
        }

        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let segment_reader = &searcher.segment_readers()[0];
        // Setup: the tombstones landed, and cluster 0 is exactly the
        // dead docs.
        let alive = segment_reader.alive_bitset().expect("deletes must land");
        let cluster0 = segment_reader
            .vector_index(embed_field)?
            .cluster_doc_ids(0)
            .expect("ivf cluster 0");
        assert_eq!(cluster0.len(), REPLICATION_N_PER);
        assert!(
            cluster0.iter().all(|&doc| !alive.is_alive(doc)),
            "cluster 0 must be fully dead"
        );

        let max_doc = segment_reader.max_doc();
        let weight = FixedDocsWeight {
            max_doc,
            docs: (0..max_doc).collect(),
        };
        let params = exhaustive_params(centroids.len());
        let (hits, stats) =
            run_top_n_with_weight(&index, embed_field, vec![10.0, 10.0], n, params, &weight)?;

        assert_eq!(hits.len(), n - REPLICATION_N_PER, "only alive docs surface");
        assert_eq!(
            stats.pruned_dead, REPLICATION_N_PER,
            "every dead row prunes as dead: {stats:?}"
        );
        // The fully-dead cluster fetches nothing; the other five fetch.
        assert_eq!(stats.postings_skipped, 1, "{stats:?}");
        assert_eq!(stats.postings_row, centroids.len() - 1);
        assert_stats_identities(&stats);
        Ok(())
    }

    /// An empty cluster still counts as probed — the loop visits it and
    /// takes the skip path: `postings_skipped` increments, nothing is
    /// fetched, and the visited/prune counters don't move.
    #[test]
    fn empty_cluster_probed_but_fetch_skipped() -> crate::Result<()> {
        // No doc is nearest to the third centroid, so its cluster is
        // empty (replicas = 1 keeps replica fill away from it too). The
        // empty centroid sits ON the query so its zero-radius bound can
        // never prove it useless - the bounds gate must not be what
        // skips it; the empty-fetch path is what's under test.
        let centroids = vec![[0.0f32, 0.0], [10.0, 0.0], [5.0, 0.1]];
        let labels: Vec<String> = (0..8).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, [f32; 2])> = (0..8)
            .map(|i| {
                let c = centroids[i % 2];
                (labels[i].as_str(), [c[0] + (i / 2) as f32 * 0.01, c[1]])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;

        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        assert_eq!(
            vec_reader.cluster_sizes(),
            Some(vec![4, 4, 0]),
            "setup: cluster 2 must be empty"
        );

        let max_doc = segment_reader.max_doc();
        let weight = FixedDocsWeight {
            max_doc,
            docs: (0..max_doc).collect(),
        };
        let (hits, stats) = run_top_n_with_weight(
            &index,
            embed_field,
            vec![5.0, 0.0],
            8,
            exhaustive_params(centroids.len()),
            &weight,
        )?;
        assert_eq!(hits.len(), 8);
        assert_eq!(
            stats.clusters_probed(),
            centroids.len(),
            "the empty cluster still counts as probed: {stats:?}"
        );
        assert_eq!(
            stats.postings_skipped, 1,
            "the empty cluster fetched nothing: {stats:?}"
        );
        assert_eq!(stats.postings_row, 2);
        assert_eq!(
            stats.vectors_visited, 8,
            "an empty cluster contributes no visited rows"
        );
        assert_stats_identities(&stats);
        Ok(())
    }

    /// Flat-path behavior across filter selectivities: hand-built filters
    /// admitting {0, 1, 50, 100}% of docs return every admitted doc
    /// exactly once, the probe-loop counters stay zeroed, and the path
    /// serves exactly one stride-sized row read per survivor.
    #[test]
    fn flat_exact_reads_one_row_per_survivor() -> crate::Result<()> {
        let n = 40usize;
        let labels: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, Option<Vec<f32>>)> = (0..n)
            .map(|i| (labels[i].as_str(), Some(vec![i as f32 * 0.1, 1.0])))
            .collect();
        let (index, embed_field, _label) = build_flat(2, &docs)?;
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let max_doc = segment_reader.max_doc();
        assert_eq!(max_doc as usize, n, "one segment holding every doc");

        for pct in [0usize, 1, 50, 100] {
            let weight = FixedDocsWeight {
                max_doc,
                docs: (0..max_doc)
                    .filter(|&doc| (doc as usize) * 100 < pct * n)
                    .collect(),
            };
            let admitted = weight.docs.len();

            let (hits, stats) =
                run_exact_on_segment(segment_reader, embed_field, vec![0.0, 0.0], n, &weight)?;

            assert_eq!(hits.len(), admitted, "{pct}%");
            // The exact path fills only `exact_rows_read`; the probe-loop
            // fields must stay zeroed.
            assert_eq!(stats.vectors_visited, 0, "{pct}%: {stats:?}");
            assert_eq!(stats.candidates_scored, 0, "{pct}%: {stats:?}");
            assert_eq!(stats.clusters_probed(), 0, "{pct}%: {stats:?}");
            assert_eq!(
                stats.exact_rows_read, admitted,
                "{pct}%: one row read per survivor"
            );
        }
        Ok(())
    }

    /// Flat scan over a `Bitmap` id-map with tombstoned docs: vectorless
    /// and deleted docs leave holes between surviving rows; every alive
    /// doc with a vector is read and scored exactly once, matching the
    /// brute-force oracle.
    #[test]
    fn flat_exact_handles_bitmap_holes_and_deletes() -> crate::Result<()> {
        let n = 30usize;
        let labels: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
        // Every third doc carries no vector at all → the id-map is Bitmap
        // and those docs own no row.
        let docs: Vec<(&str, Option<Vec<f32>>)> = (0..n)
            .map(|i| {
                let v = (i % 3 != 2).then(|| vec![i as f32 * 0.1, 1.0]);
                (labels[i].as_str(), v)
            })
            .collect();
        let (index, embed_field, label_field) = build_flat(2, &docs)?;
        // Tombstone a few vectored docs → alive holes between rows that
        // do exist.
        {
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for i in [0usize, 6, 7, 12] {
                writer.delete_term(Term::from_field_text(label_field, &format!("d{i}")));
            }
            writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        // Setup: mixed vector coverage (Bitmap id-map: fewer rows than
        // docs) and the tombstones landed.
        assert!(vec_reader.num_vectors() < segment_reader.max_doc() as usize);
        assert!(segment_reader.alive_bitset().is_some());

        let max_doc = segment_reader.max_doc();
        let weight = FixedDocsWeight {
            max_doc,
            docs: (0..max_doc).collect(),
        };
        let query = vec![0.0f32, 0.0];
        let (hits, stats) =
            run_exact_on_segment(segment_reader, embed_field, query.clone(), n, &weight)?;

        // Oracle: every alive doc with a vector, exactly once.
        let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, n)?;
        assert_eq!(hits, expected);

        let survivors = hits.len();
        assert!(survivors > 0, "fixture must leave survivors");
        assert_eq!(
            stats.exact_rows_read, survivors,
            "one row read per survivor: {stats:?}"
        );
        Ok(())
    }

    // ---- Test-only helpers ----

    /// Compute a brute-force top-K with the same convention as the
    /// shared fixture's `ground_truth::top_k`, but accepting any
    /// `&Index` (the inline-built IVF index for crafted tests doesn't
    /// have a `TestVectorIndex` wrapper).
    fn ground_truth_top_k(
        index: &Index,
        vec_field: Field,
        metric: Metric,
        query: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let query = PreparedQuery::<f32>::new(metric, Arc::new(query.to_vec()));
        let searcher = index.reader()?.searcher();
        let mut scored = Vec::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            let vec_reader = segment_reader.vector_index(vec_field)?;
            let alive = segment_reader.alive_bitset();
            for doc in 0..segment_reader.max_doc() {
                if let Some(alive) = alive {
                    if !alive.is_alive(doc) {
                        continue;
                    }
                }
                if let Some(bytes) = vec_reader.vector_bytes(doc)? {
                    scored.push((
                        query.score_doc_bytes(&bytes),
                        DocAddress::new(seg_ord as u32, doc),
                    ));
                }
            }
        }
        scored.sort_by(|a: &(Score, DocAddress), b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.segment_ord.cmp(&b.1.segment_ord))
                .then(a.1.doc_id.cmp(&b.1.doc_id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Collect the set of `DocAddress`es that a `Query` admits, by
    /// walking the per-segment weight. Used by the filter selectivity
    /// test to build an oracle restricted to the filter set.
    fn collect_filter_doc_set(
        index: &Index,
        filter: &dyn Query,
    ) -> crate::Result<std::collections::HashSet<DocAddress>> {
        let searcher = index.reader()?.searcher();
        let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let mut admitted = std::collections::HashSet::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            weight.for_each_no_score(segment_reader, &mut |docs| {
                for &d in docs {
                    admitted.insert(DocAddress::new(seg_ord as u32, d));
                }
            })?;
        }
        Ok(admitted)
    }

    /// Read the stored label text at the given `DocAddress`.
    /// Used by the trap-case + floor tests to identify docs by name
    /// rather than relying on DocId (which the merger reassigns).
    fn stored_label_at(
        index: &Index,
        label_field: Field,
        addr: DocAddress,
    ) -> crate::Result<String> {
        use crate::schema::Value;
        let searcher = index.reader()?.searcher();
        let doc = searcher.doc::<TantivyDocument>(addr)?;
        Ok(doc
            .get_first(label_field)
            .and_then(|v| Value::as_str(&v))
            .expect("stored label")
            .to_string())
    }

    /// The shared fixture's first centroid (top-left of the 3×3 grid).
    fn grid2d_first_centroid() -> [f32; 2] {
        [0.0, 0.0]
    }

    // ==================================================================
    // P5: the bounds gate
    // ==================================================================

    mod bounds_gate_tests {
        use super::*;
        use crate::vector::bounds::{HeapPeek, QueryBound, QueryBoundTracker};
        use crate::vector::{margin_ball_ball, margin_ball_halfspace, to_bound_space};

        /// A single-segment IVF index over `docs` with fixed `centroids`:
        /// ONE commit, then a single-segment merge — a multi-segment
        /// merge's source order varies across processes and permutes
        /// target doc ids, and these tests assert doc-id-level results.
        pub(super) fn single_segment_fixture(
            metric: Metric,
            centroids: &[[f32; 2]],
            docs: &[[f32; 2]],
            replicas: usize,
        ) -> crate::Result<(Index, Field)> {
            let mut sb = Schema::builder();
            let embed_field = sb.add_vector_field(
                "embedding",
                VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
            );
            sb.add_text_field("label", STRING | STORED);
            let schema = sb.build();
            let settings = IndexSettings {
                vector_clustering_threshold: 1,
                ..IndexSettings::default()
            };
            let index = Index::builder()
                .schema(schema)
                .settings(settings)
                .ivf_clusterer(Arc::new(InlineClusterer {
                    centroids: centroids.to_vec(),
                    replicas,
                }))
                .create_in_ram()?;
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for (i, vector) in docs.iter().enumerate() {
                let mut doc = TantivyDocument::new();
                doc.add_text(index.schema().get_field("label").unwrap(), format!("d{i}"));
                doc.add_vector(embed_field, vector.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
            let segment_ids = index.searchable_segment_ids()?;
            assert_eq!(segment_ids.len(), 1, "single flat segment");
            writer.merge(&segment_ids).wait()?;
            writer.wait_merging_threads()?;
            Ok((index, embed_field))
        }

        /// Deterministic pseudo-random `f32` in `[-8, 8)` — a tiny LCG so
        /// the sweep needs no RNG dependency and every run replays.
        struct Lcg(u64);
        impl Lcg {
            fn next_f32(&mut self) -> f32 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 16.0
            }
            fn point(&mut self) -> [f32; 2] {
                loop {
                    let p = [self.next_f32(), self.next_f32()];
                    // Keep cosine's write normalization well-conditioned.
                    if p[0] * p[0] + p[1] * p[1] > 0.25 {
                        return p;
                    }
                }
            }
        }

        /// PROPERTY TEST — the replica-closure theorem. Random data x
        /// {L2, cosine, dot} x replication ON; brute-force top-k vs gated
        /// top-k under a full budget. Asserts:
        ///
        /// (a) the result sets are identical — a wrong skip would lose a
        ///     true member;
        /// (b) no true member's HOME cluster is skippable even at the
        ///     FINAL (tightest) bound: `margin(t_final) >= 0`. The
        ///     running bound is never tighter than the final one (the kth
        ///     only improves), so this proves the home cluster probed at
        ///     every point of the scan. Native-scope bounds are safe iff
        ///     a qualifying row's HOME cluster always fails the skip test
        ///     — this assertion IS the theorem; replica copies may sit in
        ///     skipped clusters, the home closure is what scores them.
        #[test]
        fn closure_no_true_member_skipped() -> crate::Result<()> {
            let k = 5;
            for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
                for seed in [11u64, 29, 47] {
                    let mut rng = Lcg(seed ^ (metric as u64) << 32);
                    let centroids: Vec<[f32; 2]> = (0..5).map(|_| rng.point()).collect();
                    let docs: Vec<[f32; 2]> = (0..40).map(|_| rng.point()).collect();
                    let (index, field) = single_segment_fixture(metric, &centroids, &docs, 2)?;
                    let query: Vec<f32> = rng.point().to_vec();

                    let brute = ground_truth_top_k(&index, field, metric, &query, k)?;
                    let gated = search(
                        &index,
                        field,
                        &AllQuery,
                        query.clone(),
                        k,
                        exhaustive_params(centroids.len()),
                    )?;
                    assert_eq!(
                        gated, brute,
                        "{metric:?} seed {seed}: gated top-k must equal brute force"
                    );

                    // (b): the theorem, cluster by cluster. Homes are
                    // recomputed with the clusterer's own rule: stored
                    // (post-normalization) doc values against the
                    // QUANTIZED centroids - the reconstructions `assign`
                    // saw. The margin then runs against the STORED
                    // centroid, exactly as the gate does; the fold covers
                    // members whatever rule assigned them, so the
                    // triangle argument is assignment-rule-agnostic.
                    let searcher = index.reader()?.searcher();
                    let segment_reader = &searcher.segment_readers()[0];
                    let vec_reader = segment_reader.vector_index(field)?;
                    let ivf = vec_reader.index().expect("IVF segment");
                    let bounds = ivf.bounds();
                    let assign_centroids: Vec<[f32; 2]> = (0..ivf.num_clusters())
                        .map(|cluster| {
                            let mut decoded = [0f32; 2];
                            ivf.decode_centroid(cluster, &mut decoded)
                                .expect("readable centroid");
                            decoded
                        })
                        .collect();
                    let kth_key = brute[k - 1].0;
                    let t_final = to_bound_space(metric, kth_key);
                    let q_norm = (query[0] * query[0] + query[1] * query[1]).sqrt();
                    for &(_, addr) in &brute {
                        let stored = decode_2d(
                            &vec_reader
                                .vector_bytes(addr.doc_id)?
                                .expect("stored vector"),
                        );
                        let home = nearest_centroid(stored, &assign_centroids);
                        let sim = ivf.centroid_similarity(home, &query);
                        let r = bounds.ball_r(home);
                        let margin = match metric {
                            Metric::L2 | Metric::Cosine => {
                                margin_ball_ball(t_final, r, to_bound_space(metric, sim.score()))
                            }
                            Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t_final),
                        };
                        assert!(
                            margin >= 0.0,
                            "{metric:?} seed {seed}: true member {} home cluster {home} must \
                             never be skippable (margin {margin})",
                            addr.doc_id
                        );
                    }
                }
            }
            Ok(())
        }

        /// Boundary ties, L2: a cluster whose margin is EXACTLY zero (all
        /// values powers of two — the arithmetic is exact) holds a doc
        /// tying the kth at d == t. Exact touch must PROBE: the tie doc
        /// is scored and the doc-id tie-break decides, identically to
        /// brute force. The far cluster is provably useless and skipped.
        #[test]
        fn boundary_tie_probes_l2() -> crate::Result<()> {
            let centroids = [[0.0f32, 4.0], [4.0, 0.0], [8.0, 0.0]];
            // d0 home A (r 2), d1 home C at margin-zero touch (r 2),
            // d2 home B (r 1, disjoint by 5 - strictly nearest B, no
            // assignment tie with C).
            let docs = [[0.0f32, 2.0], [2.0, 0.0], [7.0, 0.0]];
            let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs, 1)?;
            let query = vec![0.0f32, 0.0];

            let brute = ground_truth_top_k(&index, field, Metric::L2, &query, 1)?;
            assert_eq!(
                brute[0].1.doc_id, 0,
                "tie at d = 2 breaks to the lower doc id"
            );

            let (hits, stats) = run_top_n(&index, field, query, 1, exhaustive_params(3))?;
            assert_eq!(
                hits, brute,
                "exact-touch cluster must probe, preserving the tie"
            );
            assert_eq!(
                stats.candidates_scored, 2,
                "both d = 2 docs are scored - the margin == 0 cluster probed"
            );
            assert_eq!(
                stats.clusters_probed(),
                2,
                "the disjoint cluster is skipped, the touching one is not"
            );
            assert_eq!(stats.termination, ProbeTermination::Exhausted);
            Ok(())
        }

        /// Boundary ties, dot: the second cluster's best possible score
        /// (q.c + ||q||*r) EQUALS the kth score — margin exactly zero,
        /// integer arithmetic. Exact touch probes; the tied doc wins on
        /// doc id exactly as brute force says.
        #[test]
        fn boundary_tie_probes_dot() -> crate::Result<()> {
            let centroids = [[2.0f32, 0.0], [4.0, 4.0]];
            // d0 = (3, 0) home c0, score 3; d1 = (3, 4) home c1, score 3.
            let docs = [[3.0f32, 0.0], [3.0, 4.0]];
            let (index, field) = single_segment_fixture(Metric::Dot, &centroids, &docs, 1)?;
            let query = vec![1.0f32, 0.0];

            let brute = ground_truth_top_k(&index, field, Metric::Dot, &query, 1)?;
            assert_eq!(
                brute[0].1.doc_id, 0,
                "score tie at 3 breaks to the lower doc id"
            );

            let (hits, stats) = run_top_n(&index, field, query, 1, exhaustive_params(2))?;
            assert_eq!(hits, brute);
            assert_eq!(
                stats.candidates_scored, 2,
                "the margin == 0 cluster probed and scored its tied doc"
            );
            Ok(())
        }

        /// Forced skips charge exactly the open share: on well-separated
        /// clusters the scan probes one cluster and proves the other five
        /// useless, and the work charge equals
        /// `probed*x + skipped*x + scored*(1 - x)/n_avg` — free skips
        /// would break this identity.
        #[test]
        fn skips_charge_open_share() -> crate::Result<()> {
            let centroids: Vec<[f32; 2]> = vec![
                [1.0, 1.0],
                [11.0, 1.0],
                [21.0, 1.0],
                [1.0, 11.0],
                [11.0, 11.0],
                [21.0, 11.0],
            ];
            let docs: Vec<[f32; 2]> = (0..36)
                .map(|i| {
                    let c = centroids[i / 6];
                    let off = (i % 6) as f32 * 0.01;
                    [c[0] + off, c[1] + off]
                })
                .collect();
            let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs, 1)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;

            assert_eq!(stats.termination, ProbeTermination::Exhausted);
            assert_eq!(
                stats.clusters_probed(),
                1,
                "only the home cluster survives the margins: {stats:?}"
            );
            assert_eq!(stats.candidates_scored, 6);
            // All six ranked clusters were pulled (Exhausted); five were
            // passed over by the gate.
            let skipped = 6 - stats.clusters_probed();
            let n_avg = 36.0f64 / 6.0;
            let x = open_share(n_avg);
            let row = (1.0 - x) / n_avg;
            let expected = (stats.clusters_probed() + skipped) as f64 * x
                + stats.candidates_scored as f64 * row;
            assert!(
                (stats.work_charged as f64 - expected).abs() < 1e-5,
                "skips must charge the open share: expected {expected}, got {}",
                stats.work_charged
            );
            Ok(())
        }

        /// With k unreachable the heap never fills, the bound never arms,
        /// and NOTHING is skipped — every cluster probes and the work
        /// identity has no skip term.
        #[test]
        fn unarmed_never_skips() -> crate::Result<()> {
            let centroids: Vec<[f32; 2]> = vec![[1.0, 1.0], [11.0, 1.0], [21.0, 1.0]];
            let docs: Vec<[f32; 2]> = (0..9)
                .map(|i| {
                    let c = centroids[i / 3];
                    let off = (i % 3) as f32 * 0.01;
                    [c[0] + off, c[1] + off]
                })
                .collect();
            let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs, 1)?;
            // k = 100 > 9 docs: the heap can never hold k results.
            let (hits, stats) =
                run_top_n(&index, field, vec![0.2, 0.3], 100, exhaustive_params(3))?;
            assert_eq!(hits.len(), 9, "every doc is a hit at k > N");
            assert_eq!(
                stats.clusters_probed(),
                3,
                "unarmed, every cluster probes: {stats:?}"
            );
            assert_eq!(stats.candidates_scored, 9);
            let n_avg = 3.0f64;
            let x = open_share(n_avg);
            let row = (1.0 - x) / n_avg;
            let expected = 3.0 * x + 9.0 * row;
            assert!(
                (stats.work_charged as f64 - expected).abs() < 1e-5,
                "no skip term in the identity: expected {expected}, got {}",
                stats.work_charged
            );
            Ok(())
        }

        /// `t` tracks kth improvements in bound space per metric, and an
        /// unchanged kth leaves the bound untouched. The bound-space
        /// conversion runs inside the improvement branch only, so the
        /// cosine sqrt is paid per improvement — asserted here by value
        /// (the cached and recomputed paths are indistinguishable by
        /// construction when the key is unchanged).
        #[test]
        fn t_maintenance_per_metric() {
            // (metric, first kth key, expected t, improved key, expected t)
            let cases = [
                // L2 keys -d^2: d = 2, then d = 1.
                (Metric::L2, -4.0f32, 2.0f32, -1.0f32, 1.0f32),
                // Cosine keys cos: chord sqrt(2*(1-cos)).
                (Metric::Cosine, 0.5, 1.0, 0.875, 0.5),
                // Dot keys the score; identity.
                (Metric::Dot, 3.0, 3.0, 4.5, 4.5),
            ];
            for (metric, first, t_first, improved, t_improved) in cases {
                let mut tracker = QueryBoundTracker::new();
                assert_eq!(tracker.bound(), QueryBound::Filling);
                tracker.observe(metric, HeapPeek::Filling, 0);
                assert_eq!(
                    tracker.bound(),
                    QueryBound::Filling,
                    "{metric:?}: no arm on Filling"
                );

                tracker.observe(metric, HeapPeek::Full { kth_key: first }, 1);
                assert_eq!(
                    tracker.bound(),
                    QueryBound::Armed { t: t_first },
                    "{metric:?}: t from first kth"
                );
                // Unchanged kth: bound bit-identical.
                tracker.observe(metric, HeapPeek::Full { kth_key: first }, 2);
                assert_eq!(tracker.bound(), QueryBound::Armed { t: t_first });

                tracker.observe(metric, HeapPeek::Full { kth_key: improved }, 3);
                assert_eq!(
                    tracker.bound(),
                    QueryBound::Armed { t: t_improved },
                    "{metric:?}: t tracks the improvement"
                );
            }
        }

        /// The armed index is the first probe at which the heap held k
        /// results, and never moves after.
        #[test]
        fn armed_index_recorded() {
            let mut tracker = QueryBoundTracker::new();
            tracker.observe(Metric::L2, HeapPeek::Filling, 0);
            tracker.observe(Metric::L2, HeapPeek::Filling, 1);
            assert_eq!(tracker.armed_at_probe(), None, "unarmed while filling");
            tracker.observe(Metric::L2, HeapPeek::Full { kth_key: -1.0 }, 2);
            assert_eq!(tracker.armed_at_probe(), Some(2), "arms at the first Full");
            tracker.observe(Metric::L2, HeapPeek::Full { kth_key: -0.5 }, 3);
            assert_eq!(
                tracker.armed_at_probe(),
                Some(2),
                "later improvements never move the armed index"
            );
        }
    }

    // ==================================================================
    // P6: probe-stats telemetry
    // ==================================================================

    mod bounds_stats_tests {
        use super::bounds_gate_tests::single_segment_fixture;
        use super::*;

        /// The six-centroid separated fixture: probing the home cluster
        /// arms the bound and the other five clusters are provably
        /// useless.
        fn separated_fixture(metric: Metric) -> crate::Result<(Index, Field)> {
            let centroids: Vec<[f32; 2]> = vec![
                [1.0, 1.0],
                [11.0, 1.0],
                [21.0, 1.0],
                [1.0, 11.0],
                [11.0, 11.0],
                [21.0, 11.0],
            ];
            let docs: Vec<[f32; 2]> = (0..36)
                .map(|i| {
                    let c = centroids[i / 6];
                    let off = (i % 6) as f32 * 0.01;
                    [c[0] + off, c[1] + off]
                })
                .collect();
            single_segment_fixture(metric, &centroids, &docs, 1)
        }

        /// `bounds_skips` counts exactly the clusters the gate passed
        /// over: all ranked clusters minus the probed ones on an
        /// exhausted stream.
        #[test]
        fn skip_count_matches() -> crate::Result<()> {
            let (index, field) = separated_fixture(Metric::L2)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;
            assert_eq!(stats.termination, ProbeTermination::Exhausted);
            assert_eq!(stats.clusters_probed(), 1, "{stats:?}");
            assert_eq!(
                stats.bounds_skips, 5,
                "every non-home cluster is a counted skip: {stats:?}"
            );
            Ok(())
        }

        /// Unarmed (k > N): zero skips, and the armed index serializes
        /// as JSON null, not 0 - the harness's armed-share column
        /// depends on the null contract.
        #[test]
        fn armed_null_when_unarmed() -> crate::Result<()> {
            let (index, field) = separated_fixture(Metric::L2)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 100, exhaustive_params(6))?;
            assert_eq!(stats.bound_armed_at_probe, None);
            assert_eq!(stats.bounds_skips, 0);
            let value = serde_json::to_value(&stats).expect("ProbeStats serializes");
            assert_eq!(
                value["bound_armed_at_probe"],
                serde_json::Value::Null,
                "unarmed must serialize as null, not 0"
            );
            Ok(())
        }

        /// The armed index is the tracker's recorded value: the heap
        /// fills inside the first probed cluster at k <= its size
        /// (index 0), and spans into the second at larger k (index 1).
        #[test]
        fn armed_index_value() -> crate::Result<()> {
            let (index, field) = separated_fixture(Metric::L2)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;
            assert_eq!(
                stats.bound_armed_at_probe,
                Some(0),
                "k = 5 fills inside the 6-doc home cluster: {stats:?}"
            );
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 10, exhaustive_params(6))?;
            assert_eq!(
                stats.bound_armed_at_probe,
                Some(1),
                "k = 10 needs the second probed cluster: {stats:?}"
            );
            assert!(
                stats.bounds_skips > 0,
                "armed late still skips the far tail"
            );
            Ok(())
        }
    }
}
