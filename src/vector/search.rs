//! The cross-segment vector search driver.
//!
//! Centroids live in the index-level set file and every segment shares its
//! cluster ids, so search ranks the set's centroids ONCE and, per ranked
//! cluster, gathers that cluster's rows across ALL segments into one
//! global heap. The kth threshold is shared by construction: where the old
//! per-segment loop probed ~4 clusters in every segment to find a
//! neighbor in its 4th-closest cluster, this loop probes ~4 clusters
//! total.
//!
//! Per `(cluster, segment)` pair the gates run cheapest first:
//!
//! 1. presence — one pinned bit; an absent cluster costs nothing
//! 2. bounds — the margin against the GLOBAL query bound; a provable skip
//!    charges the open share without touching rows or filter
//! 3. filter — the segment's filter bitset, materialized lazily HERE, at
//!    most once per segment per query; an empty filter kills the segment
//! 4. row gate — `seen → filter → alive` off the pinned id-map
//! 5. fetch + score survivors into the one heap
//!
//! Lazy filters matter for tiny NRT segments (mostly-empty ranges),
//! bounds-skipped segments once the bound arms, and future spatially
//! partitioned segments where most pairs are absent; on randomly sharded
//! big segments every filter materializes at the first probed cluster.

use std::sync::Arc;

use common::BitSet;

use super::backend::{open_share, ProbeStats, ProbeTermination, WorkUnits};
use super::bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, to_bound_space, HeapPeek, QueryBound,
    QueryBoundTracker, Verdict,
};
use super::distance::norm_squared_wide;
use super::index_reader::VectorIndexReader;
use super::ivf::{AdaptiveProbeParams, Candidate, IvfIndex, Workspace};
use super::prepared::PreparedQuery;
use super::VectorElement;
use crate::collector::sort_key::NaturalComparator;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer, TopNComputer};
use crate::fastfield::AliveBitSet;
use crate::query::Weight;
use crate::schema::{Field, Metric};
use crate::{DocAddress, DocId, Score, Searcher, SegmentOrdinal, SegmentReader, TantivyError};

/// The global heap: keyed by `(similarity, global tie-break key)`, holding
/// `DocAddress`es from every segment; ascending `DocAddress` breaks full
/// ties, matching the old cross-segment merge order.
type GlobalHeap<S, C> =
    TopNComputer<(Score, <S as SortKeyComputer>::SortKey), DocAddress, (NaturalComparator, C)>;

/// One segment's per-query state. Everything lazy stays lazy: the filter
/// bitset and the `seen` dedup bitset are allocated only when the loop
/// first gates this segment's rows.
struct SegmentSearch<'a, TChild> {
    ord: SegmentOrdinal,
    reader: &'a SegmentReader,
    vec: Arc<VectorIndexReader>,
    tie: TChild,
    alive: Option<&'a AliveBitSet>,
    filter: FilterState,
    /// Replica dedup; doc ids are segment-local, so dedup is too.
    seen: Option<BitSet>,
    /// Set when the filter materializes empty: no row of this segment can
    /// ever qualify, all its future ranges skip for free.
    dead: bool,
}

/// The lazily materialized filter bitset of one segment.
enum FilterState {
    NotBuilt,
    Built(BitSet),
}

impl<TChild> SegmentSearch<'_, TChild> {
    fn ivf(&self) -> &IvfIndex {
        self.vec
            .index()
            .expect("vector-bearing segments carry an IvfIndex")
    }
}

/// One gate survivor from the pre-pass over a cluster's rows: `row`
/// indexes into the segment-wide dense rows slot.
#[derive(Clone, Copy)]
struct Survivor {
    row: usize,
    doc: DocId,
}

/// Global top-N by vector similarity across the searcher's segments,
/// ordered by `(similarity, tie_break)` descending. Returns the hits (best
/// first) and the query's [`ProbeStats`].
pub(crate) fn global_top_n_by<T, S>(
    searcher: &Searcher,
    weight: &dyn Weight,
    field: Field,
    query: &Arc<Vec<T>>,
    top_n: usize,
    adaptive: &AdaptiveProbeParams,
    tie_break: &S,
) -> crate::Result<(Vec<((Score, S::SortKey), DocAddress)>, ProbeStats)>
where
    T: VectorElement,
    S: SortKeyComputer,
{
    let mut stats = ProbeStats::default();

    // The participating segments: the searcher's snapshot, minus segments
    // with no vector data for the field.
    let mut segments: Vec<SegmentSearch<'_, S::Child>> = Vec::new();
    let mut set_version: Option<u64> = None;
    for (ord, reader) in searcher.segment_readers().iter().enumerate() {
        let vec = reader.vector_index(field)?;
        let Some(ivf) = vec.index() else {
            continue;
        };
        // Multi-version snapshots are unsupported, like multi-version
        // merges: a shared routing order requires shared cluster ids.
        match set_version {
            None => set_version = Some(ivf.centroid_set_version()),
            Some(version) if version != ivf.centroid_set_version() => {
                return Err(TantivyError::InvalidArgument(format!(
                    "segments assigned against different centroid set versions ({} vs {version}); \
                     multi-version search is not supported",
                    ivf.centroid_set_version(),
                )));
            }
            Some(_) => {}
        }
        segments.push(SegmentSearch {
            ord: ord as SegmentOrdinal,
            reader,
            tie: tie_break.segment_sort_key_computer(reader)?,
            alive: reader.alive_bitset(),
            filter: FilterState::NotBuilt,
            seen: None,
            dead: false,
            vec,
        });
    }
    let Some(set_version) = set_version else {
        return Ok((Vec::new(), stats));
    };
    if top_n == 0 {
        return Ok((Vec::new(), stats));
    }
    stats.segments_searched = segments.len() as u32;

    let set = searcher.index().centroid_set_search_index(set_version)?;
    let router = set.field_router(field).ok_or_else(|| {
        TantivyError::InternalError(format!(
            "centroid set v{set_version} has no router for field {field:?}"
        ))
    })?;
    let num_centroids = router.num_centroids();
    for segment in &segments {
        if segment.ivf().num_clusters() != num_centroids {
            return Err(TantivyError::InternalError(format!(
                "segment {} holds {} clusters but centroid set v{set_version} holds \
                 {num_centroids}",
                segment.ord,
                segment.ivf().num_clusters(),
            )));
        }
    }

    let options = segments[0].vec.options().clone();
    let metric = options.metric();
    let prepared = PreparedQuery::<T>::new(metric, Arc::clone(query));

    let (work_budget, pricing_open, pricing_row) =
        resolve_budget(adaptive, num_centroids, &segments)?;

    // ONE routing pass. Routing operates in `f32` (centroid rows are `f32`
    // today), so the query is widened losslessly per element.
    let query_f32: Vec<f32> = prepared.query().iter().map(|e| e.to_f32()).collect();
    let mut routing_ws = Workspace::new();
    let mut ranked = router.rank_clusters(&mut routing_ws, &query_f32);

    let mut topn: GlobalHeap<S, S::Comparator> =
        TopNComputer::new_with_comparator(top_n, (NaturalComparator, tie_break.comparator()));

    // The global query bound, maintained at cluster boundaries; `||q||`
    // once, for the dot margin's Cauchy-Schwarz term.
    let mut bound_tracker = QueryBoundTracker::new();
    let q_norm = norm_squared_wide(prepared.query()).sqrt() as f32;

    let mut work_spent = WorkUnits::ZERO;
    let mut termination = ProbeTermination::Exhausted;
    // Ranked clusters that did any work in any segment — the arming index's
    // denominator and the boundary at which the kth folds into the bound.
    let mut touched_clusters = 0u32;
    // The probed cluster's gate survivors; allocated once, reused.
    let mut survivors: Vec<Survivor> = Vec::new();

    for Candidate { sim, node: cluster } in &mut ranked {
        // Boundary rule: open iff remaining > 0. The tripping pull proves
        // another ranked cluster existed, keeping `Ceiling` distinct from
        // `Exhausted`.
        if work_spent >= work_budget {
            termination = ProbeTermination::Ceiling;
            break;
        }
        let cluster = cluster as usize;
        let mut touched = false;

        for segment in &mut segments {
            if segment.dead {
                continue;
            }
            // Gate 1: presence — absent clusters cost nothing, not even
            // the margin computation (an empty cluster's 0.0 bound would
            // masquerade as a bounds skip).
            let Some(rows) = segment.ivf().non_empty_cluster_range(cluster) else {
                continue;
            };
            touched = true;

            // Gate 2: the bounds verdict, against the GLOBAL bound — the
            // shared-kth win. Consumed only through `Armed` (the heap
            // holds k results); Filling probes, and SATURATED probes
            // arithmetically (+inf margin).
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
                // key into the metric's distance space for L2/cosine,
                // and dot consumes the raw `q . c` key directly.
                let r = segment.ivf().bounds().ball_r(cluster);
                match metric {
                    Metric::L2 | Metric::Cosine => {
                        margin_ball_ball(t, r, to_bound_space(metric, sim.score()))
                    }
                    Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t),
                }
            });
            if let Verdict::Skip = verdict {
                // A skip charges the open share: skips are search work,
                // and free skips break the work identity. No row work is
                // spent, no filter is materialized.
                work_spent += pricing_open;
                stats.bounds_skips += 1;
                continue;
            }

            // Gate 3: the segment's filter, materialized at most once per
            // query — only for segments that reach a real probe.
            if matches!(segment.filter, FilterState::NotBuilt) {
                let filter = build_filter_bitset(weight, segment.reader)?;
                stats.filters_built += 1;
                if filter.len() == 0 {
                    segment.dead = true;
                    continue;
                }
                segment.filter = FilterState::Built(filter);
            }
            let FilterState::Built(filter) = &segment.filter else {
                unreachable!("filter materialized above")
            };
            let seen = segment
                .seen
                .get_or_insert_with(|| BitSet::with_max_value(segment.reader.max_doc()));

            // Event-wise charging, part 1: the open.
            work_spent += pricing_open;

            // Gate 4 pre-pass: `seen → filter → alive` off the pinned
            // id-map, no posting bytes fetched.
            let (visited, pruned_filter, pruned_dead, pruned_seen, scored_rows) =
                collect_cluster_survivors(
                    &segment.vec,
                    rows,
                    filter,
                    segment.alive,
                    seen,
                    &mut survivors,
                );
            stats.vectors_visited += visited;
            stats.pruned_filter += pruned_filter;
            stats.pruned_dead += pruned_dead;
            stats.pruned_seen += pruned_seen;

            // Event-wise charging, part 2: the rows that survive the
            // pre-pass — exactly the rows fetched and scored below.
            work_spent += pricing_row * scored_rows as f64;

            if survivors.is_empty() {
                stats.postings_skipped += 1;
                continue;
            }
            stats.postings_row += 1;
            stats.candidates_scored += survivors.len();

            // Gate 5: fetch + score — one stride-sized read per survivor
            // (the unit the pg-backed `Directory` serves zero-copy).
            for &Survivor { row, doc } in &survivors {
                let vbytes = segment.vec.vector_bytes_for_row(row)?;
                let score = prepared.score_doc_bytes(&vbytes);
                // The skip is exact: `(s, t) < (ts, tt)` requires either
                // `s < ts`, or a tie the composite comparator resolves —
                // so a candidate rejected on similarity alone could never
                // have survived, and its tie-break conversion (a possible
                // dictionary lookup) is pure waste.
                if let Some(((threshold_score, _), _)) = &topn.threshold {
                    if score < *threshold_score {
                        continue;
                    }
                }
                let segment_key = segment.tie.segment_sort_key(doc, score);
                let global_key = segment.tie.convert_segment_sort_key(segment_key);
                topn.push_unordered((score, global_key), DocAddress::new(segment.ord, doc));
            }
        }

        if touched {
            touched_clusters += 1;
            // Fold the exact kth into the bound at the cluster boundary.
            // `kth_best` is O(buffer) and force-truncates — results and
            // every counter above are unaffected (truncation only drops
            // already-lost entries and tightens the push threshold).
            let peek = HeapPeek::from_kth(topn.kth_best().map(|(score, _tie)| score));
            bound_tracker.observe(metric, peek, touched_clusters - 1);
        }
    }
    // The armed index exists exactly when the bound armed.
    debug_assert!(
        bound_tracker.armed_at_probe().is_some()
            == matches!(bound_tracker.bound(), QueryBound::Armed { .. })
    );

    stats.routing = ranked.metrics();
    stats.bound_armed_at_probe = bound_tracker.armed_at_probe();
    stats.termination = termination;
    stats.work_charged = work_spent.to_f32();

    let hits = topn
        .into_sorted_vec()
        .into_iter()
        .map(|cd| (cd.sort_key, cd.doc))
        .collect();
    Ok((hits, stats))
}

/// The query's global work budget and unit prices.
///
/// Capacity counts an open share per non-empty `(cluster, segment)` pair
/// plus a row share per native doc — exactly what an exhaustive,
/// unfiltered, delete-free scan would charge — so `max_probe_fraction`
/// keeps meaning "this fraction of the index's work" whatever the segment
/// count. One `min_probe_clusters` floor per QUERY: the old per-segment
/// floor inflated work linearly with the segment count.
fn resolve_budget<TChild>(
    adaptive: &AdaptiveProbeParams,
    num_centroids: usize,
    segments: &[SegmentSearch<'_, TChild>],
) -> crate::Result<(WorkUnits, WorkUnits, WorkUnits)> {
    if !(adaptive.max_probe_fraction > 0.0) {
        return Err(TantivyError::InvalidArgument(
            "max_probe_fraction must be greater than 0".to_string(),
        ));
    }
    let total_docs: usize = segments.iter().map(|s| s.ivf().num_docs()).sum();
    let total_nonempty: usize = segments
        .iter()
        .map(|s| s.ivf().num_non_empty_clusters())
        .sum();
    // Native docs as WRITTEN: dead rows charge nothing (alive pre-pass),
    // so deletes only ever cheapen a scan.
    let n_avg = total_docs as f64 / num_centroids.max(1) as f64;
    let x = open_share(n_avg);
    let capacity =
        total_nonempty as f64 * x + (1.0 - x) * total_docs as f64 / n_avg.max(f64::MIN_POSITIVE);
    let budget = (adaptive.max_probe_fraction as f64 * capacity)
        .max(adaptive.min_probe_clusters as f64)
        .min(capacity);
    Ok((
        WorkUnits::new(budget),
        WorkUnits::new(x),
        WorkUnits::new((1.0 - x) / n_avg.max(f64::MIN_POSITIVE)),
    ))
}

/// Run one cluster's rows in one segment through the `seen → filter →
/// alive` gate — off the pinned id-map alone, with no posting bytes
/// fetched — collecting into `survivors` (cleared first) the rows to
/// score. Returns `(visited, pruned_filter, pruned_dead, pruned_seen,
/// scored_rows)`; the partition identity
/// `visited == filter + dead + seen + scored` holds, and only the `scored`
/// term ever charges budget.
#[inline(never)]
fn collect_cluster_survivors(
    vec: &VectorIndexReader,
    rows: std::ops::Range<usize>,
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
        let doc = vec.doc_id_at(row);
        visited += 1;
        // Dedup FIRST, marking on first encounter whatever the later
        // verdicts say, so a replica's second copy is never re-checked
        // (it counts as `pruned_seen`, not the original verdict's
        // bucket).
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

/// Drain the filter `DocSet` into a dense BitSet for O(1) random
/// membership testing per cluster doc. Successive probed clusters each
/// re-span the segment's whole doc range, so a forward-only `DocSet`
/// cannot serve the probe order. The BitSet allocates `max_doc / 8` bytes
/// regardless of filter selectivity.
#[inline(never)]
fn build_filter_bitset(
    weight: &dyn Weight,
    segment_reader: &SegmentReader,
) -> crate::Result<BitSet> {
    let mut filter = BitSet::with_max_value(segment_reader.max_doc());
    weight.for_each_no_score(segment_reader, &mut |docs| {
        for &doc in docs {
            filter.insert(doc);
        }
    })?;
    Ok(filter)
}
