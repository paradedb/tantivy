//! Per-query search instrumentation: what the cross-segment driver
//! reports ([`ProbeStats`]) and how it stopped ([`ProbeTermination`]).

use crate::vector::ivf::IvfSearchMetrics;

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

/// Probe-loop instrumentation for one query, filled by the cross-segment
/// loop in [`search`](super::search): a prune breakdown of every doc the
/// inner loop touched, plus posting-fetch counters. One instance per query
/// — the loop is global, so the counters are too.
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
    /// Probed CLUSTERS that yielded at least one survivor to score,
    /// counted once however many segments the cluster's rows span.
    pub postings_row: usize,
    /// Probed CLUSTERS that yielded no survivors in ANY segment: the
    /// `filter → alive → seen` pre-pass rejected every row (fully
    /// filtered / dead / already-seen). The two `postings_*` counters
    /// partition the probed clusters:
    /// [`clusters_probed`](Self::clusters_probed) `== postings_row + postings_skipped`.
    pub postings_skipped: usize,
    /// Per-(cluster, segment) opens — the same clusters counted once per
    /// segment their rows live in. Scales with segment count where
    /// [`clusters_probed`](Self::clusters_probed) does not, so the ratio
    /// is the fragmentation the probe loop is paying for.
    pub segment_opens: usize,
    /// Routing cost of ranking the clusters to probe — ONE routing pass
    /// per query over the index-level set. See [`IvfSearchMetrics`].
    pub routing: IvfSearchMetrics,
    /// Clusters the bounds gate passed over with a Skip verdict, without
    /// opening them: their margins proved they could not improve the
    /// armed result. Each charged the open share. Disjoint from the
    /// `postings_*` partition, which only counts opened clusters.
    pub bounds_skips: u32,
    /// Probe index (0-based, counting ranked clusters that did any work
    /// in any segment) at which the query bound first armed - the
    /// boundary where the heap filled and margins existed to certify
    /// against. `None` = never armed (the heap never held k results),
    /// serialized as JSON null - the harness's armed-share column
    /// depends on the null contract.
    pub bound_armed_at_probe: Option<u32>,
    /// How the probe loop terminated.
    pub termination: ProbeTermination,
    /// Work units the probe loop charged against its resolved budget:
    /// opens at `x` per non-empty (cluster, segment) pair, scored rows
    /// at `(1 - x)/n_avg`.
    pub work_charged: f32,
    /// Rows fetched and scored by the exact tier — flat (unclustered)
    /// segments scanned exhaustively. Mandatory work, outside the probe
    /// budget; disjoint from every clustered counter above.
    pub exact_rows_read: usize,
    /// Vector-bearing segments this query considered — clustered and flat.
    pub segments_searched: u32,
    /// Segments whose filter bitset was actually materialized — lazy
    /// filters mean a segment whose every touched (cluster, segment)
    /// pair was absent or bounds-skipped never evaluates its filter.
    pub filters_built: u32,
}

impl ProbeStats {
    /// Fold another pass's stats into this one: counters add,
    /// `bound_armed_at_probe` keeps the earliest arming, and
    /// `termination` reports `Ceiling` if either side hit its ceiling.
    /// Used when merging per-segment / per-tier / per-worker fruits of
    /// one query.
    pub fn absorb(&mut self, other: ProbeStats) {
        self.candidates_scored += other.candidates_scored;
        self.vectors_visited += other.vectors_visited;
        self.pruned_filter += other.pruned_filter;
        self.pruned_dead += other.pruned_dead;
        self.pruned_seen += other.pruned_seen;
        self.postings_row += other.postings_row;
        self.postings_skipped += other.postings_skipped;
        self.segment_opens += other.segment_opens;
        self.routing.visited_count += other.routing.visited_count;
        self.bounds_skips += other.bounds_skips;
        self.bound_armed_at_probe = match (self.bound_armed_at_probe, other.bound_armed_at_probe) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        if other.termination == ProbeTermination::Ceiling {
            self.termination = ProbeTermination::Ceiling;
        }
        self.work_charged += other.work_charged;
        self.exact_rows_read += other.exact_rows_read;
        self.segments_searched += other.segments_searched;
        self.filters_built += other.filters_built;
    }

    /// Distinct clusters the probe loop opened — segment-count
    /// invariant. [`segment_opens`](Self::segment_opens) is the
    /// per-(cluster, segment) count.
    ///
    /// Returns (`usize`): `postings_row + postings_skipped` — every probed
    /// cluster either fetched survivors or fetched nothing.
    #[inline]
    pub fn clusters_probed(&self) -> usize {
        self.postings_row + self.postings_skipped
    }
}
