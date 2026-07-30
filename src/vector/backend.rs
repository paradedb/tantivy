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
use std::sync::Arc;

use common::BitSet;

use super::index_reader::VectorIndexReader;
use super::ivf::{Candidate, IvfIndex, IvfSearchMetrics, ProbeBudget, Workspace};
use super::prepared::PreparedQuery;
use super::tie_break::NoTieBreak;
use super::VectorElement;
use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, TopNComputer};
use crate::fastfield::AliveBitSet;
use crate::query::Weight;
use crate::schema::Field;
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
    adaptive: ProbeBudget,
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
        adaptive: ProbeBudget,
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
    /// The probe budget was spent: the ceiling bound the scan.
    Ceiling,
    /// The gate policy terminated the scan: it proved no remaining ranked
    /// cluster can improve the current top-N.
    Gate,
    /// The ranked centroids were exhausted without hitting either stop.
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
    /// count; starvation reads as `candidates_scored < top_n`, and
    /// [`heap_saturated`](Self::heap_saturated) is its per-segment flag.
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
    /// How the probe loop terminated. Per-segment; does not sum.
    pub termination: ProbeTermination,
    /// Whether the segment's top-N heap ended the scan holding `top_n`
    /// scored candidates. `false` flags starvation - fewer filter-passing
    /// docs than requested reached the heap.
    pub heap_saturated: bool,
    /// How many clusters had been probed when the gate armed: the heap
    /// first held `top_n` candidates, observed at a between-cluster
    /// boundary. `None` when the heap never saturated before the loop
    /// stopped.
    pub gate_armed_at_probe: Option<usize>,
    /// The gate's Terminate verdict held on the very yield where the
    /// Ceiling fired: the two stops tied, and Ceiling won by the
    /// checked-first attribution contract.
    pub gate_armed_at_ceiling: bool,
    /// Clusters the gate passed over with a Skip verdict, without opening
    /// them. Skips only: a Defer (the policy's unpaid hedge) is neither
    /// counted here nor charged. The control policy never skips, so this
    /// reads zero until the certificate lands.
    pub radius_skips: usize,
    /// Work units this segment's probe loop charged against its resolved
    /// budget: opens at `x`, first-seen rows at `(1 - x)/n_bar`, Skips at
    /// `x`. The budget identity is per segment:
    /// `budget < work_charged <= budget + last cluster's charge` on
    /// Ceiling terminations.
    pub work_charged: f32,
}

impl ProbeStats {
    /// Clusters the probe loop visited — each either fetched survivors
    /// (`postings_row`) or skipped (`postings_skipped`).
    #[inline]
    pub fn clusters_probed(&self) -> usize {
        self.postings_row + self.postings_skipped
    }
}

/// What the probe loop does with the ranked cluster it is holding. The
/// loop owns no policy of its own: it asks the gate for one of these,
/// prices it through the [`ChargeModel`], and acts.
// The policy this tree ships produces only `Probe`. The loop and the
// pricing table handle the whole verdict set from the start, so the seam
// is complete before a policy that exercises it exists.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Open the cluster: stream its rows and score the survivors.
    Probe,
    /// Pass over this cluster without opening it. A verdict about THIS
    /// cluster alone -- the scan continues with the next yield.
    Skip,
    /// Pass over without opening AND without charging: the policy wants
    /// one more yield before it commits to stopping. Priced (at zero) by
    /// [`ChargeModel::verdict_charge`] like every verdict - the accounting
    /// contract that keeps unpaid pulls bounded at one per paid pull,
    /// since a non-deferring follow-up is itself a paid Probe or Skip.
    /// A Defer never reorders the stream and never stashes the deferred
    /// cluster for later: hardening against adversarial yield order (a
    /// deferred-frontier or lower-bound-reordered stream) is out of scope
    /// here, pending the routing-interface discussion. The control policy
    /// (`NoGate`) never emits this verdict.
    Defer,
    /// Stop the scan.
    Terminate,
}

/// One ranked cluster as a gate policy sees it: what the policy may read,
/// and nothing of the loop's own accounting.
#[derive(Clone, Copy, Debug)]
struct GateContext {
    /// The current k-th best score; `None` while the heap is unsaturated.
    kth: Option<Score>,
    /// Clusters opened or skipped so far -- the arming clock's tick.
    clusters_probed: usize,
}

/// Gate policy: the probe loop's skip/stop authority, and the only place
/// early termination can come from.
///
/// This is one half of the probe loop's seam. The other half is the
/// [`ChargeModel`], which prices verdicts. The split is deliberate and
/// load-bearing for review: a policy decides WHETHER a cluster is worth
/// opening and never touches the budget; the charge model decides what
/// that decision COSTS and never inspects geometry. Neither can silently
/// become the other, and the loop between them branches on nothing but
/// the verdict it is handed.
///
/// Arming is policy-side by the same rule: a heap holding fewer than
/// `top_n` candidates has no band to certify against, so saturation is a
/// policy's precondition to reason about (see [`ArmingClock`]), not a
/// condition the loop imposes on it.
trait ProbeGate {
    /// The verdict for the cluster the loop is holding. `&mut` because a
    /// policy may carry state across yields (arming, patience).
    fn verdict(&mut self, ctx: GateContext) -> Verdict;

    /// Would this same yield have terminated? Consulted only when the
    /// budget ceiling fired first, to record that the two stops tied --
    /// see the attribution contract on [`VectorBackend::scan_clusters`].
    /// Pure: it must not disturb the policy's state.
    fn would_terminate(&self, ctx: GateContext) -> bool;

    /// Fold policy-owned telemetry into the segment's stats once the scan
    /// has stopped.
    fn fold_stats(&self, stats: &mut ProbeStats);
}

/// Heap-saturation clock: the boundary at which the top-N heap was first
/// seen holding `top_n` candidates. Shared by every policy, since every
/// policy needs the same precondition, and reported as
/// [`ProbeStats::gate_armed_at_probe`].
#[derive(Clone, Copy, Debug, Default)]
struct ArmingClock {
    at: Option<usize>,
}

impl ArmingClock {
    /// Observe this boundary; returns whether the heap is saturated.
    #[inline]
    fn observe(&mut self, ctx: GateContext) -> bool {
        let saturated = ctx.kth.is_some();
        if saturated && self.at.is_none() {
            self.at = Some(ctx.clusters_probed);
        }
        saturated
    }
}

/// The policy that never gates: every yield is probed and the scan ends
/// at the budget ceiling or at stream exhaustion. The loop is then a pure
/// budget-taker -- which is what this tree ships, and what the envelope
/// study measured as the old gate's only zero-regret configuration: no
/// setting of the deleted distance-ratio gate beat gateless probing at
/// matched cost. A certificate that can prove its skips is the successor;
/// a heuristic band is not.
#[derive(Debug, Default)]
struct NoGate {
    arming: ArmingClock,
}

impl ProbeGate for NoGate {
    #[inline]
    fn verdict(&mut self, ctx: GateContext) -> Verdict {
        self.arming.observe(ctx);
        Verdict::Probe
    }

    #[inline]
    fn would_terminate(&self, _ctx: GateContext) -> bool {
        false
    }

    fn fold_stats(&self, stats: &mut ProbeStats) {
        stats.gate_armed_at_probe = self.arming.at;
    }
}

/// THE WORK-UNIT MODEL
///
/// The probe budget meters WORK, denominated so that 1 unit = one average
/// cluster of work. Charging is event-wise as probing proceeds:
///
/// | event                    | charge          |
/// |--------------------------|-----------------|
/// | open a cluster           | `x`             |
/// | first-seen (dedup'd) row | `(1 - x)/n_bar` |
/// | Skip verdict             | `x`             |
///
/// with `n_bar = N / C` - native docs over clusters, GLOBAL across the
/// index's IVF segments and computed once at query init: one constant, no
/// per-segment term in the unit definition, no per-cluster metadata. Rows
/// charge on FIRST structural encounter regardless of filter verdict (the
/// row was read and checked; work is work); replica re-encounters charge
/// nothing, so a doc costs exactly one row-deduction index-wide,
/// whichever copy arrives first.
///
/// NORMALIZATION IDENTITY: an exhaustive scan of the whole index charges
/// `C*x + (1 - x)*N/n_bar = exactly C` units - capacity is unchanged and
/// the probe fraction keeps its scale. Big clusters penalize themselves
/// by streaming more unique rows, so balance is accounted for with zero
/// stored state.
///
/// BOUNDARY RULE: the budget is inspected only at cluster boundaries -
/// open iff `remaining > 0`, deduct as-you-go, never truncate mid-cluster
/// (posting order is not distance order, so a partial scan is random loss
/// on a paid open). Overshoot is bounded by the last cluster's charge. No
/// pre-open cost knowledge is needed or used.
///
/// The measured hardware/layout ratio behind the open share: how many
/// rows of full work one cluster OPEN costs - INTERNAL CONSTANT,
/// deliberately not a knob. The open share is derived from it PER INDEX
/// at query init: `x = ROWS_PER_OPEN / (ROWS_PER_OPEN + n_bar)`, so x
/// self-calibrates to the index's cluster granularity (the same
/// philosophy as a result-anchored gate: measure, do not configure). At
/// the reference fixture's n_bar = 20 this gives x ~ 0.076; at production
/// granularity (n_bar ~ 100-200) x ~ 0.008-0.016. Mis-setting the ratio
/// distorts cost attribution only; no gate policy reads it.
///
/// Derivation (reference fixture, cohere-100k repl-2, release build):
/// per-query latency regressed on (clusters opened, rows read, centroids
/// routed) across two cluster-granularity regimes (n_bar = 20 and
/// n_bar = 6.25 - the cross-fixture contrast separates the open cost from
/// the collinear row cost; the routing term is required or beam-ranking
/// cost pollutes the open coefficient; "routed" = centroids the router
/// actually scored):
///   t ~ 0.19us*open + 0.116us*row + 1.45us*routed + c   (R^2 = 0.993)
///   ROWS_PER_OPEN = a/b = 0.19/0.116 ~ 1.64.
/// The ratio was measured at this dimensionality: b (per-row work) scales
/// with vector dimension while a (open bookkeeping) mostly does not, so
/// the ratio is dimension/layout-specific - a per-dimension derivation is
/// a named follow-up, not built here.
pub(crate) const ROWS_PER_OPEN: f64 = 1.64;

/// The per-index open share: the fraction of one average cluster's work
/// that opening it costs, from the measured [`ROWS_PER_OPEN`] ratio and
/// the index's own n_bar. Clamped to (0, 0.5] - a share above one half
/// would mean opens dominate rows, which only degenerate sub-2-row
/// clusters produce, and the clamp keeps the pull-bounding arithmetic
/// sane there.
pub(crate) fn open_share(n_bar: f64) -> f64 {
    (ROWS_PER_OPEN / (ROWS_PER_OPEN + n_bar.max(0.0))).min(0.5)
}

/// Work accounting: the segment's budget, and the price of everything the
/// probe loop does against it. The other half of the seam (see
/// [`ProbeGate`]) - every verdict is priced here, never in a policy.
#[derive(Clone, Copy, Debug)]
struct ChargeModel {
    /// Work units this segment may spend before the ceiling binds:
    /// `f * units_seg`, resolved by [`ProbeBudget::resolved_work_budget`].
    budget: f64,
    /// The per-index open share `x`: what opening a cluster costs, and
    /// what a Skip costs (the open decision without the rows).
    open: f64,
    /// `(1 - x)/n_bar`: what one first-seen row costs.
    row: f64,
}

impl ChargeModel {
    /// The price of a verdict, settled before the cluster's rows are
    /// known. A Skip pays the open share: deciding not to open a cluster
    /// costs what deciding to open one costs, which is what bounds pulls
    /// per ceiling at ~1/x whether the policy skips or probes.
    #[inline]
    fn verdict_charge(&self, verdict: Verdict) -> f64 {
        match verdict {
            Verdict::Probe | Verdict::Skip => self.open,
            // A Defer is the policy's unpaid hedge, and a Terminate ends
            // the scan: neither reads a row or opens anything.
            Verdict::Defer | Verdict::Terminate => 0.0,
        }
    }

    /// The price of the rows an opened cluster streamed: dedup-unique
    /// rows only, filter verdict irrelevant - the row was read and
    /// checked, and work is work.
    #[inline]
    fn row_charge(&self, first_seen: usize) -> f64 {
        first_seen as f64 * self.row
    }
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
        // open share x is derived from the index's own n_bar at query init
        // - see `open_share`.
        let (budget, n_bar, x) = self
            .adaptive
            .resolved_work_budget(num_centroids, index.num_docs())?;
        let charge = ChargeModel {
            budget,
            open: x,
            row: (1.0 - x) / n_bar,
        };

        // Phase 1: rank the clusters to probe, lazily — the scan below pulls
        // ranked clusters on demand, so routing cost is paid only as far as
        // probing actually reaches. The filter-effective budget can pull far
        // past its nominal cluster count on a selective filter (each skipped
        // cluster costs ~0), and lazy routing keeps that cheap.
        // Routing operates in `f32` (centroid rows are `f32` today), so the
        // query is widened losslessly per element.
        let query_f32: Vec<f32> = self.query.query().iter().map(|e| e.to_f32()).collect();
        let mut routing_ws = Workspace::new();
        let mut ranked = index.rank_clusters(&mut routing_ws, &query_f32);

        // The production policy. One implementor, chosen here, at query
        // init - the loop below never asks which policy it is holding.
        let mut gate = NoGate::default();

        let topn = self.scan_clusters(
            index,
            &mut ranked,
            &mut gate,
            charge,
            &filter,
            max_doc,
            alive,
            top_n,
            tie_break,
            tie_comparator,
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

    /// Phase 2: adaptive probe loop. Each probed cluster is gated first —
    /// [`Self::collect_cluster_survivors`] runs `filter → alive → seen`
    /// off the pinned id-map with no posting bytes in hand — and only the
    /// survivors' bytes are then fetched, one stride-sized read per
    /// surviving row. Cluster-order arrival of survivors forbids the
    /// ascending-doc shortcut in `push`; use `push_unordered`.
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
    /// ATTRIBUTION CONTRACT (interface, not code order): the budget
    /// ceiling is the checked-first stop. When the ceiling and the gate's
    /// Terminate hold on the same yield the termination is `Ceiling`, and
    /// [`ProbeStats::gate_armed_at_ceiling`] records the tie - so a
    /// budget-starved query is never mistaken for a certified one in the
    /// telemetry, whichever order a future rewrite evaluates them in.
    ///
    /// `#[inline(never)]` so it forms its own flamegraph frame carrying its
    /// `score_doc_bytes` cost.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn scan_clusters<K, CTail>(
        &self,
        index: &IvfIndex,
        ranked: impl Iterator<Item = Candidate>,
        gate: &mut impl ProbeGate,
        charge: ChargeModel,
        filter: &BitSet,
        max_doc: DocId,
        alive: Option<&AliveBitSet>,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHeap<K, CTail>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let mut topn =
            TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_comparator));
        // The prune counters accumulate into locals and fold into `ProbeStats`
        // once after the loop, keeping the hot per-doc path free of
        // indirection.
        let mut candidates = 0usize;
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut pruned_seen = 0usize;
        let mut postings_row = 0usize;
        let mut postings_skipped = 0usize;
        let mut termination = ProbeTermination::Exhausted;
        let mut gate_armed_at_ceiling = false;
        // Replication can place the same doc in several probed clusters; dedup
        // by doc id so a vector is scored at most once.
        let mut seen = BitSet::with_max_value(max_doc);
        // The probed cluster's gate survivors; allocated once, reused
        // across clusters.
        let mut survivors: Vec<Survivor> = Vec::new();
        // Accounting: `work_spent` accrues against the resolved budget, all
        // of it priced by `charge` - the loop adds up what it is told.
        let mut work_spent = 0.0f64;
        let work_budget = charge.budget;

        // The ranked stream is consumed exactly as the router yields it.
        // FOLLOW-UP (owned by Ruchir): lower-bound `(d - r)` reordering of
        // the yield stream is a deliberate non-goal of this change - do not
        // "improve" the ordering here.
        for Candidate {
            sim: _,
            node: cluster,
        } in ranked
        {
            // The heap key is the composite `(similarity, tie_break)`;
            // lexicographic order means the composite minimum carries the
            // minimum SIMILARITY in the top-N, so its score component is
            // exactly the k-th best the gate needs. Ties stay sound: the
            // strict `>` bounds only pass over clusters that cannot reach
            // the k-th score, and equal-score candidates - the ones a tie
            // key could still admit - are never behind a Skip.
            let ctx = GateContext {
                kth: topn.kth_best().map(|(score, _)| score),
                clusters_probed: postings_row + postings_skipped,
            };
            // The pull that trips the ceiling proves another ranked cluster
            // existed, keeping `Ceiling` distinct from `Exhausted`. The
            // budget is filter-effective (see `ChargeModel::cluster_charge`),
            // so a selective filter walks far past its nominal cluster count.
            // Checked first - the attribution contract above.
            // Boundary rule: open iff remaining > 0 - never truncate
            // mid-cluster; overshoot is bounded by the last cluster's charge.
            if work_spent >= work_budget {
                termination = ProbeTermination::Ceiling;
                gate_armed_at_ceiling = gate.would_terminate(ctx);
                break;
            }

            let verdict = gate.verdict(ctx);
            work_spent += charge.verdict_charge(verdict);
            match verdict {
                Verdict::Terminate => {
                    termination = ProbeTermination::Gate;
                    break;
                }
                // Passed over: `Skip` paid its price above, `Defer` is the
                // policy's unpaid hedge. Neither opens the cluster, so
                // neither touches the prune counters.
                Verdict::Skip | Verdict::Defer => continue,
                Verdict::Probe => {}
            }
            let cluster = cluster as usize;

            let rows = index.cluster_range(cluster);

            // Pre-pass: run the gate off the pinned id-map alone, BEFORE
            // any posting bytes are fetched, so the fetch below can be
            // skipped for rows that won't be scored. Gate order, the
            // `seen` marking point, and every prune counter are exactly
            // the fetch-then-gate scan's; only the byte fetch moved.
            let (v, pf, pd, ps, first_seen) =
                self.collect_cluster_survivors(rows, filter, alive, &mut seen, &mut survivors);
            visited += v;
            pruned_filter += pf;
            pruned_dead += pd;
            pruned_seen += ps;

            work_spent += charge.row_charge(first_seen);

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
        }

        stats.vectors_visited += visited;
        stats.pruned_filter += pruned_filter;
        stats.pruned_dead += pruned_dead;
        stats.pruned_seen += pruned_seen;
        stats.postings_row += postings_row;
        stats.postings_skipped += postings_skipped;
        stats.candidates_scored += candidates;
        stats.termination = termination;
        stats.gate_armed_at_ceiling = gate_armed_at_ceiling;
        // f64 accumulation in the loop (the normalization identity is
        // asserted to 1e-6*C); f32 is plenty for telemetry.
        stats.work_charged += work_spent as f32;
        gate.fold_stats(stats);
        // Final-state saturation, exact even when the heap filled inside the
        // last probed cluster (arming is only *observed* at boundaries). The
        // forced truncation this implies is invisible past this point - the
        // scan is over.
        stats.heap_saturated = topn.kth_best().is_some();

        Ok(topn)
    }

    /// Phase 2 pre-pass: run one cluster's rows through the
    /// `filter → alive → seen` gate — off the pinned id-map alone, with no
    /// posting bytes fetched — collecting into `survivors` (cleared first)
    /// the rows to score.
    /// Returns `(visited, pruned_filter, pruned_dead, pruned_seen,
    /// first_seen)`.
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
        let mut first_seen = 0usize;
        for row in rows {
            let doc = self.reader.doc_id_at(row);
            visited += 1;
            // Dedup FIRST, and mark on first structural encounter whatever
            // the later verdicts say: `first_seen` is the work-unit charge
            // basis (a doc costs one row-deduction index-wide, whichever
            // copy arrives first), and marking rejected docs too means a
            // replica's second copy is never re-checked. Consequence for
            // the prune buckets: a re-encountered copy of a
            // filter-rejected doc counts as `pruned_seen`, not
            // `pruned_filter`; the partition identity
            // `visited == filter + dead + seen + scored` is unchanged.
            if seen.contains(doc) {
                pruned_seen += 1;
                continue;
            }
            seen.insert(doc);
            first_seen += 1;
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
        }
        (visited, pruned_filter, pruned_dead, pruned_seen, first_seen)
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
    // geometry (the trap case and the certificate-tier fixtures) build a
    // tiny IVF index inline via `build_inline_ivf` and an
    // `InlineClusterer` that's compatible with the batched IvfClusterer
    // trait.
    // ============================================================
    use std::cmp::Ordering;

    use super::*;
    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{
        AllQuery, BitSetDocSet, ConstScorer, EnableScoring, Explanation, Query, Scorer, TermQuery,
    };
    use crate::schema::{IndexRecordOption, Metric, Schema, Term, STORED, STRING};
    use crate::vector::tests::{exhaustive_params, TestVectorIndex};
    use crate::vector::{
        IvfCentroids, IvfClusterer, IvfMatrix, IvfMergeSettings, IvfVectors,
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
        params: ProbeBudget,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let collector = TopDocs::with_limit(k)
            .order_by_similarity(field, query)
            .with_probe_budget(params);
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
        params: ProbeBudget,
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
    // override. Tests that need points at specific coordinates (the trap
    // case, the certificate tiers) build a small IVF index inline via
    // the helper below.

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
            _vectors: IvfVectors<'_>,
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
            let IvfCentroids::F32(centroids) = centroids;
            Ok(vectors
                .matrix
                .values
                .chunks_exact(2)
                .map(|v| {
                    let mut best = 0u32;
                    let mut best_d2 = f32::INFINITY;
                    for (i, c) in centroids.values.chunks_exact(2).enumerate() {
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
        let one_probe = ProbeBudget {
            max_probe_fraction: 0.5,
            min_probe_clusters: 1,
            work_model: None,
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
            ProbeBudget::default(),
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
        let (_, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            4,
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
        let params = ProbeBudget {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            work_model: None,
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
            ProbeBudget::default(),
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

        let params = ProbeBudget {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            work_model: None,
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

    /// A budget below capacity binds, is attributed to the ceiling, and
    /// overshoots by at most one cluster's charge - the boundary rule on
    /// a real fixture rather than a hand-built one. K > the segment's doc
    /// count keeps the heap unsaturated, so no gate policy can end the
    /// scan: the stop point under test is the budget's alone.
    #[test]
    fn probe_stats_max_probe_fraction_ceiling() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = ProbeBudget {
            max_probe_fraction: 0.2,
            min_probe_clusters: 1,
            work_model: None,
        };
        let searcher = index.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(index.embedding_field())?;
        let ivf = vec_reader.index().expect("expected IVF storage");
        let (c_seg, n_seg) = (ivf.num_clusters(), ivf.num_docs());
        let (budget, n_bar, x) = params.resolved_work_budget(c_seg, n_seg)?;
        assert!(budget < c_seg as f64, "setup: the budget must bind");
        drop(searcher);

        let (_, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            n_seg + 1,
            params,
        )?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert!(
            stats.clusters_probed() < c_seg,
            "the budget must bind before exhaustion: {stats:?}"
        );
        assert!(
            stats.work_charged as f64 > budget,
            "the ceiling fires only once the budget is spent: {stats:?}"
        );
        // Overshoot bound: no single cluster can charge more than one
        // open plus every row in the segment.
        let max_cluster_charge = x + n_seg as f64 * (1.0 - x) / n_bar;
        assert!(
            stats.work_charged as f64 <= budget + max_cluster_charge + 1e-6,
            "overshoot is bounded by the last cluster's charge: {stats:?}"
        );
        Ok(())
    }

    /// Empty clusters charge only the open share, never a full unit: at
    /// f = 1 the grid fixture's segment - 20 docs concentrated in a few of
    /// its 9 clusters - runs to exhaustion and charges exactly C units.
    /// That is the normalization identity on a real, unevenly-filled
    /// fixture; `unit_normalization_exact` pins it on a hand-built one.
    #[test]
    fn empty_clusters_charge_only_the_open_share() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let query = grid2d_first_centroid();
        let params = ProbeBudget {
            max_probe_fraction: 1.0,
            min_probe_clusters: 1,
            work_model: None,
        };
        let searcher = index.index.reader()?.searcher();
        let vec_reader = searcher.segment_readers()[0].vector_index(index.embedding_field())?;
        let n_seg = vec_reader.index().expect("expected IVF storage").num_docs();
        drop(searcher);

        // K > the segment's docs: the heap never saturates, so the stop
        // point is the budget's alone.
        let (_, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            query.to_vec(),
            n_seg + 1,
            params,
        )?;
        // The probed segment holds its docs concentrated in a few of its
        // 9 clusters (which few is a per-build draw - the pairwise merge
        // sorts random segment UUIDs - so the exact split is not pinned).
        assert_eq!(stats.termination, ProbeTermination::Exhausted, "{stats:?}");
        assert_eq!(stats.clusters_probed(), DEFAULT_NUM_CENTROIDS, "{stats:?}");
        assert!(
            stats.postings_row <= 3 && stats.postings_skipped >= 6,
            "grid docs concentrate in a few clusters; empties dominate: {stats:?}"
        );
        let c = DEFAULT_NUM_CENTROIDS as f32;
        assert!(
            (stats.work_charged - c).abs() <= 1e-5 * c,
            "an exhaustive scan charges exactly C units: {stats:?}"
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
            termination: ProbeTermination::Gate,
            heap_saturated: true,
            gate_armed_at_probe: Some(3),
            gate_armed_at_ceiling: false,
            radius_skips: 2,
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
                "termination": "Gate",
                "heap_saturated": true,
                "gate_armed_at_probe": 3,
                "gate_armed_at_ceiling": false,
                "radius_skips": 2,
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
    // Result-anchored termination scaffold.
    //
    // This tree has no gate: termination is Ceiling or Exhausted only,
    // and the scaffold's job is the shared plumbing the radius
    // certificate slots into - heap-saturation arming telemetry,
    // ceiling-first attribution, and the budget identity. The
    // certificate itself (and `ProbeTermination::Gate`) arrives with the
    // stored radii.
    // ============================================================

    /// Full-visibility budget params: fraction 1.0 (ceiling = every
    /// cluster), floor 1.
    fn budget_params() -> ProbeBudget {
        ProbeBudget {
            max_probe_fraction: 1.0,
            min_probe_clusters: 1,
            work_model: None,
        }
    }

    /// Line fixture for the arming/starvation tests: four well-separated
    /// clusters along the x-axis, `n_per` docs tightly around each.
    fn line_fixture(n_per: usize) -> crate::Result<(Index, Field, Field, Vec<[f32; 2]>)> {
        let centroids = vec![[0.0_f32, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]];
        let labels: Vec<String> = (0..centroids.len() * n_per)
            .map(|i| format!("d{i}"))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = (0..labels.len())
            .map(|i| {
                let c = centroids[i / n_per];
                (labels[i].as_str(), [c[0] + (i % n_per) as f32 * 0.01, c[1]])
            })
            .collect();
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        Ok((index, embed_field, label_field, centroids))
    }

    /// With the control policy in charge, `ProbeTermination::Gate` is
    /// unreachable: every scan ends at the budget ceiling or at stream
    /// exhaustion. This fixture is balanced (4 clusters, 3 docs each), so
    /// each cluster charges exactly one unit and the probed count reads
    /// straight off the budget.
    #[test]
    fn probe_loop_is_a_pure_budget_taker() -> crate::Result<()> {
        let (index, embed_field, _label, centroids) = line_fixture(3)?;
        for (fraction, expect_probed, expect_term) in [
            // ceil(0.3 * 4) = 2 of 4 clusters -> the ceiling binds.
            (0.3, 2, ProbeTermination::Ceiling),
            // full fraction -> the stream runs dry first.
            (1.0, centroids.len(), ProbeTermination::Exhausted),
        ] {
            let params = ProbeBudget {
                max_probe_fraction: fraction,
                min_probe_clusters: 1,
                work_model: None,
            };
            let (_, stats) = run_top_n(&index, embed_field, vec![0.0, 0.0], 2, params.clone())?;
            assert_eq!(stats.termination, expect_term, "f={fraction}: {stats:?}");
            assert_eq!(
                stats.clusters_probed(),
                expect_probed,
                "probed must equal the resolved budget: f={fraction}, {stats:?}"
            );
            assert!(
                (stats.work_charged - stats.clusters_probed() as f32).abs() < 1e-5,
                "balanced geometry charges exactly one unit per cluster: {stats:?}"
            );
            assert_eq!(stats.radius_skips, 0, "no certificate, no skips: {stats:?}");
            assert!(
                !stats.gate_armed_at_ceiling,
                "no certificate, no tie: {stats:?}"
            );
        }
        Ok(())
    }

    /// Arming telemetry is boundary-observed: with a filter passing only
    /// docs in the third-ranked cluster, the heap first reads full at the
    /// boundary after the third probe - and arming is telemetry only; the
    /// scan keeps probing to exhaustion.
    #[test]
    fn arming_waits_for_heap_saturation() -> crate::Result<()> {
        let (index, embed_field, _label, centroids) = line_fixture(2)?;
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let admitted = segment_reader
            .vector_index(embed_field)?
            .cluster_doc_ids(2)
            .expect("cluster 2 doc ids");
        assert_eq!(admitted.len(), 2, "setup: cluster 2 holds exactly 2 docs");
        let weight = FixedDocsWeight {
            max_doc: segment_reader.max_doc(),
            docs: admitted.clone(),
        };

        let k = 2;
        let (hits, stats) = run_top_n_with_weight(
            &index,
            embed_field,
            vec![0.0, 0.0],
            k,
            budget_params(),
            &weight,
        )?;
        // Every admitted doc surfaces; the ranked-earlier clusters were
        // probed on the way (their fetches skip under the filter).
        assert_eq!(hits.len(), k);
        let hit_docs: std::collections::HashSet<DocId> =
            hits.iter().map(|(_, addr)| addr.doc_id).collect();
        assert_eq!(hit_docs, admitted.iter().copied().collect());
        assert_eq!(
            (stats.postings_skipped, stats.postings_row),
            (3, 1),
            "{stats:?}"
        );
        assert_eq!(
            stats.gate_armed_at_probe,
            Some(3),
            "armed at the boundary after the third probe: {stats:?}"
        );
        assert_eq!(
            stats.clusters_probed(),
            centroids.len(),
            "arming is telemetry, not termination: the scan ran on to exhaustion"
        );
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        assert!(stats.heap_saturated);
        Ok(())
    }

    /// Fewer passing docs than `top_n`: the heap never fills, arming never
    /// happens, the loop runs to exhaustion, and the partial results all
    /// surface - the invariant the saturation precondition exists for.
    #[test]
    fn starvation_runs_to_exhaustion() -> crate::Result<()> {
        let (index, embed_field, _label, centroids) = line_fixture(2)?;
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let admitted = segment_reader
            .vector_index(embed_field)?
            .cluster_doc_ids(2)
            .expect("cluster 2 doc ids");
        let weight = FixedDocsWeight {
            max_doc: segment_reader.max_doc(),
            docs: admitted.clone(),
        };

        let k = 5; // > the 2 passing docs
        let (hits, stats) = run_top_n_with_weight(
            &index,
            embed_field,
            vec![0.0, 0.0],
            k,
            budget_params(),
            &weight,
        )?;
        assert_eq!(hits.len(), admitted.len(), "partial results surface");
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        assert!(!stats.heap_saturated, "{stats:?}");
        assert_eq!(stats.gate_armed_at_probe, None, "starved heap never arms");
        assert_eq!(stats.radius_skips, 0);
        assert_eq!(
            stats.clusters_probed(),
            centroids.len(),
            "every cluster probed: {stats:?}"
        );
        Ok(())
    }

    /// A ceiling that binds before the heap fills is attributed to
    /// `Ceiling` with `heap_saturated == false` - starvation and the
    /// ceiling are separately visible.
    #[test]
    fn ceiling_starved_attribution() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        // fraction 0.1 of 6 clusters -> ceiling 1: one full-price probe
        // exhausts the budget.
        let params = ProbeBudget {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            work_model: None,
        };

        // K larger than the probed cluster's 6 docs: the heap never fills.
        let (_, stats) = run_top_n(&index, embed_field, vec![10.0, 10.0], 10, params)?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert_eq!(stats.clusters_probed(), 1);
        assert!(!stats.heap_saturated, "{stats:?}");
        assert!(!stats.gate_armed_at_ceiling, "no certificate exists to tie");
        Ok(())
    }

    // ============================================================
    // Work-unit budget properties. Unprimed (single-segment) runs use the
    // segment-local n_bar, under which units_seg is exactly C_seg - the
    // normalization identity, per segment. Every test here takes K > N so
    // the heap cannot saturate: no gate policy can end these scans, and
    // the stop point under test is the budget's alone.
    // ============================================================

    /// An exhaustive scan charges `C*x + (1 - x)*N/n_bar = exactly C`
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
        let (_, stats) = run_top_n(&index, embed_field, vec![0.0, 0.0], 11, budget_params())?;
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
        let (_, stats) = run_top_n(&index, embed_field, vec![0.0, 0.0], 11, budget_params())?;
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
        // Sizes [30, 2, 2, 2, 2, 2]: C = 6, N = 40, n_bar = 20/3.
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
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        let params = ProbeBudget {
            max_probe_fraction: 0.8,
            min_probe_clusters: 1,
            work_model: None,
        };
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

    // ============================================================
    // The probe-loop seam.
    //
    // These drive `scan_clusters` with a SCRIPTED policy, so they pin the
    // loop's half of the contract - what each verdict makes it do, and
    // what the pricing table charges - with no policy geometry in the
    // way. They stay valid whatever policy ships.
    // ============================================================

    /// A policy that replays a fixed verdict script, one verdict per
    /// yield, and probes once the script runs out.
    struct ScriptedGate {
        script: Vec<Verdict>,
        calls: usize,
        skips: usize,
        arming: ArmingClock,
        /// What [`ProbeGate::would_terminate`] answers at the ceiling.
        terminates_at_ceiling: bool,
    }

    impl ScriptedGate {
        fn new(script: &[Verdict]) -> Self {
            Self {
                script: script.to_vec(),
                calls: 0,
                skips: 0,
                arming: ArmingClock::default(),
                terminates_at_ceiling: false,
            }
        }

        fn terminating_at_ceiling(mut self) -> Self {
            self.terminates_at_ceiling = true;
            self
        }
    }

    impl ProbeGate for ScriptedGate {
        fn verdict(&mut self, ctx: GateContext) -> Verdict {
            self.arming.observe(ctx);
            let verdict = self
                .script
                .get(self.calls)
                .copied()
                .unwrap_or(Verdict::Probe);
            self.calls += 1;
            if verdict == Verdict::Skip {
                self.skips += 1;
            }
            verdict
        }

        fn would_terminate(&self, _ctx: GateContext) -> bool {
            self.terminates_at_ceiling
        }

        fn fold_stats(&self, stats: &mut ProbeStats) {
            stats.gate_armed_at_probe = self.arming.at;
            stats.radius_skips = self.skips;
        }
    }

    /// Five clusters on a line, two tight docs each.
    fn seam_fixture() -> crate::Result<(Index, Field, Vec<[f32; 2]>)> {
        let centroids = vec![
            [0.0_f32, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
        ];
        let labels: Vec<String> = (0..centroids.len() * 2).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, [f32; 2])> = (0..labels.len())
            .map(|i| {
                let c = centroids[i / 2];
                (labels[i].as_str(), [c[0] + (i % 2) as f32 * 0.01, c[1]])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        Ok((index, embed_field, centroids))
    }

    /// Drive the loop over `order`'s clusters with `gate` deciding and
    /// `charge` pricing, filter passing everything.
    fn drive_scan(
        index: &Index,
        embed_field: Field,
        centroids: &[[f32; 2]],
        order: &[usize],
        gate: &mut impl ProbeGate,
        charge: ChargeModel,
        top_n: usize,
    ) -> crate::Result<ProbeStats> {
        let query = vec![0.0_f32, 0.0];
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query.clone()),
            budget_params(),
        )?;
        let vec_reader = segment_reader.vector_index(embed_field)?;
        let ivf = vec_reader.index().expect("expected IVF storage");
        let stream: Vec<Candidate> = order
            .iter()
            .map(|&c| Candidate {
                sim: Metric::L2.similarity(query.as_slice(), centroids[c].as_slice()),
                node: c as u32,
            })
            .collect();
        let max_doc = segment_reader.max_doc();
        let mut filter = BitSet::with_max_value(max_doc);
        for doc in 0..max_doc {
            filter.insert(doc);
        }
        let mut stats = ProbeStats::default();
        backend.scan_clusters(
            ivf,
            stream.into_iter(),
            gate,
            charge,
            &filter,
            max_doc,
            None,
            top_n,
            &mut NoTieBreak,
            NaturalComparator,
            &mut stats,
        )?;
        Ok(stats)
    }

    /// Every verdict does exactly one thing to the loop: `Probe` opens
    /// the cluster, `Skip` and `Defer` pass over it unopened, `Terminate`
    /// stops the scan. Nothing else in the loop reads the policy.
    #[test]
    fn seam_acts_on_every_verdict() -> crate::Result<()> {
        let (index, embed_field, centroids) = seam_fixture()?;
        let mut gate = ScriptedGate::new(&[
            Verdict::Probe,
            Verdict::Skip,
            Verdict::Defer,
            Verdict::Probe,
            Verdict::Terminate,
        ]);
        let stats = drive_scan(
            &index,
            embed_field,
            &centroids,
            &[0, 1, 2, 3, 4],
            &mut gate,
            // Opens and rows both free: this test is about what each
            // verdict makes the loop DO, not what it costs.
            ChargeModel {
                budget: 100.0,
                open: 0.0,
                row: 0.0,
            },
            1,
        )?;
        assert_eq!(stats.termination, ProbeTermination::Gate);
        assert_eq!(
            stats.clusters_probed(),
            2,
            "only Probe opens a cluster: {stats:?}"
        );
        assert_eq!(
            stats.vectors_visited, 4,
            "two opened clusters, two rows each: {stats:?}"
        );
        assert_eq!(
            stats.radius_skips, 1,
            "the policy folds its own Skip tally: {stats:?}"
        );
        Ok(())
    }

    /// The ceiling is checked before the policy is consulted, and when
    /// both stops hold on one yield the termination is `Ceiling` with the
    /// tie recorded. One full-price open spends the whole budget, so the
    /// second yield trips it.
    #[test]
    fn ceiling_wins_the_tie_and_records_it() -> crate::Result<()> {
        let (index, embed_field, centroids) = seam_fixture()?;
        let mut gate = ScriptedGate::new(&[Verdict::Probe]).terminating_at_ceiling();
        let stats = drive_scan(
            &index,
            embed_field,
            &centroids,
            &[0, 1, 2],
            &mut gate,
            // One open spends the whole budget, so the second yield
            // trips the ceiling.
            ChargeModel {
                budget: 1.0,
                open: 1.0,
                row: 0.0,
            },
            1,
        )?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert_eq!(stats.clusters_probed(), 1, "{stats:?}");
        assert!(
            stats.gate_armed_at_ceiling,
            "the tie must be recorded: {stats:?}"
        );
        Ok(())
    }

    /// Cost lives in the pricing table, never in a policy: this budget
    /// charges an opened cluster by its filter pass rate, and prices no
    /// verdict before that cluster's rows are known.
    #[test]
    fn charge_model_prices_every_verdict() {
        // n_bar = 4, so x = 1.64/5.64 and one row costs (1 - x)/4.
        let n_bar = 4.0;
        let x = open_share(n_bar);
        let charge = ChargeModel {
            budget: 1.0,
            open: x,
            row: (1.0 - x) / n_bar,
        };
        assert_eq!(
            charge.verdict_charge(Verdict::Probe),
            x,
            "an open costs the open share"
        );
        assert_eq!(
            charge.verdict_charge(Verdict::Skip),
            x,
            "a Skip costs an open: deciding not to open is the same decision"
        );
        assert_eq!(
            charge.verdict_charge(Verdict::Defer),
            0.0,
            "the patience hedge is unpaid"
        );
        assert_eq!(charge.verdict_charge(Verdict::Terminate), 0.0);
        assert_eq!(charge.row_charge(0), 0.0);
        // One average cluster of work: an open plus n_bar rows == 1 unit.
        assert!(
            (charge.verdict_charge(Verdict::Probe) + charge.row_charge(n_bar as usize) - 1.0).abs()
                < 1e-12,
            "the unit is one average cluster, by construction"
        );
    }

    /// The open share self-calibrates to cluster granularity and stays in
    /// (0, 0.5]: fine clusters make opens matter, coarse ones make them
    /// vanish, and a degenerate sub-2-row cluster hits the clamp.
    #[test]
    fn open_share_tracks_granularity() {
        assert!((open_share(20.0) - 0.0758).abs() < 1e-3, "reference n_bar");
        assert!(open_share(200.0) < open_share(20.0));
        assert!(open_share(2.0) > open_share(20.0));
        assert_eq!(open_share(0.0), 0.5, "clamped, not unbounded");
        assert_eq!(open_share(1.0), 0.5, "clamped at one row per cluster");
        for n_bar in [0.5, 1.0, 6.25, 20.0, 100.0, 1000.0] {
            let x = open_share(n_bar);
            assert!(x > 0.0 && x <= 0.5, "x out of range at n_bar={n_bar}: {x}");
        }
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
        params: ProbeBudget,
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
            ProbeBudget::default(),
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
        // empty (replicas = 1 keeps replica fill away from it too).
        let centroids = vec![[0.0f32, 0.0], [10.0, 0.0], [100.0, 100.0]];
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
}
