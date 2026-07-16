//! Per-segment vector search execution.
//!
//! Built once per segment by
//! [`TopDocsByVectorSimilarity`](super::collector::TopDocsByVectorSimilarity)
//! around the segment's cached [`VectorIndexReader`]. The search strategy
//! branches once, on whether the reader carries an [`IvfIndex`]: with it, the
//! filter is drained into a bitmap and the routed clusters are probed
//! adaptively; without it, the filter `Scorer` is iterated doc-by-doc and
//! every vector is scored exactly, with survivor reads coalesced into
//! contiguous row runs where the filter is dense.

use std::ops::Range;
use std::sync::Arc;

use common::BitSet;

use super::index_reader::{read_blocks, VectorIndexReader};
use super::ivf::{AdaptiveProbeParams, IvfIndex};
use super::prepared::PreparedQuery;
use super::VectorElement;
use crate::collector::sort_key::NaturalComparator;
use crate::collector::TopNComputer;
use crate::fastfield::AliveBitSet;
use crate::query::Weight;
use crate::schema::{Field, Metric};
use crate::{DocAddress, DocId, Score, SegmentOrdinal, SegmentReader, TantivyError};

/// Per-segment vector search: the segment's [`VectorIndexReader`] plus the
/// per-query state. Build via [`VectorBackend::for_segment`].
pub struct VectorBackend<T: VectorElement> {
    reader: Arc<VectorIndexReader>,
    query: Arc<PreparedQuery<T>>,
    adaptive: AdaptiveProbeParams,
    segment_ord: SegmentOrdinal,
    /// Test-only override of the per-cluster posting-fetch dispatch, so
    /// each [`FetchMode`] path can be exercised regardless of fixture
    /// geometry. `None` — always, in production — dispatches on
    /// [`row_fetch_is_cheaper`].
    #[cfg(test)]
    forced_fetch: Option<ForcedFetch>,
    /// Test-only override of the flat/exact path's run batching. `None` —
    /// always, in production — applies [`absorb_into_run`].
    #[cfg(test)]
    forced_exact: Option<ForcedExactRead>,
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
            #[cfg(test)]
            forced_fetch: None,
            #[cfg(test)]
            forced_exact: None,
        })
    }

    /// Top-N within this segment: probe routed clusters when the reader has
    /// an index, exact-scan otherwise. Hits come back already tagged with
    /// `DocAddress`, so the collector doesn't need a second pass to attach
    /// the segment.
    pub fn top_n(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        self.top_n_with_stats(weight, segment_reader, top_n, None)
    }

    /// Like [`Self::top_n`] but threads an optional [`ProbeStats`] sink into
    /// the IVF probe loop; `None` is identical in behavior and cost to
    /// `top_n`. The exact path fills only the `exact_reads_*` counters and
    /// leaves every probe-loop field untouched.
    pub fn top_n_with_stats(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        stats: Option<&mut ProbeStats>,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        match self.reader.index() {
            Some(index) => self.probe_top_n(index, weight, segment_reader, top_n, stats),
            None => self.exact_top_n(weight, segment_reader, top_n, stats),
        }
    }

    /// Flat/exact scan: drain the filter DocSet doc-by-doc, but batch the
    /// survivors' row reads into contiguous runs grown by
    /// [`absorb_into_run`] — one ranged read per run, sized to survivor
    /// density. A selective filter degrades to length-1 runs (exactly the
    /// per-doc reads this replaced); a dense one coalesces consecutive
    /// rows into few large reads. Fills only the `exact_reads_*` stats.
    fn exact_top_n(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        stats: Option<&mut ProbeStats>,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        // `for_each_no_score` walks the filter DocSet in ascending doc order,
        // which permits the fast `TopNComputer::push` path (valid only under
        // ascending-doc pushes) — the flat id-maps (`Identity`/`Bitmap`) keep
        // rows ascending with doc ids, so run flushes preserve that order.
        // `NaturalComparator` because similarity is "higher = better" — see
        // the note on `scan_clusters`.
        let mut topn = TopNComputer::<Score, DocId, NaturalComparator>::new_with_comparator(
            top_n,
            NaturalComparator,
        );
        let alive = segment_reader.alive_bitset();
        let stride = self.reader.options().bytes_per_vector();
        let mut reads_chunked = 0usize;
        let mut reads_single = 0usize;
        // The pending run of survivors, `(doc, row)` with rows ascending;
        // flushed (one ranged read, one score per member) when the next
        // survivor doesn't absorb, and once more after the drain.
        let mut run: Vec<(DocId, usize)> = Vec::new();
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
                if let Some(&(_, last_row)) = run.last() {
                    debug_assert!(row > last_row, "flat rows must ascend with doc ids");
                    if self.absorb_next_survivor(run[0].1, last_row, row, stride) {
                        run.push((doc, row));
                        continue;
                    }
                    match self.score_run(&run, stride, &mut topn) {
                        Ok(()) => tally_run(run.len(), &mut reads_chunked, &mut reads_single),
                        Err(err) => {
                            read_err = Some(err);
                            return;
                        }
                    }
                    run.clear();
                }
                run.push((doc, row));
            }
        })?;
        if let Some(err) = read_err {
            return Err(err);
        }
        if !run.is_empty() {
            self.score_run(&run, stride, &mut topn)?;
            tally_run(run.len(), &mut reads_chunked, &mut reads_single);
        }
        if let Some(s) = stats {
            s.exact_reads_chunked += reads_chunked;
            s.exact_reads_single += reads_single;
        }
        let segment_ord = self.segment_ord;
        Ok(topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect())
    }

    /// Flush one run: a single ranged read spanning `run[0].1..=last.1`,
    /// then score each member from its slice offset. Rows inside the span
    /// that belong to no member — holes [`absorb_into_run`] chose to
    /// absorb — are fetched but never scored; that cost is exactly what
    /// the run rule priced against a separate read.
    fn score_run(
        &self,
        run: &[(DocId, usize)],
        stride: usize,
        topn: &mut TopNComputer<Score, DocId, NaturalComparator>,
    ) -> crate::Result<()> {
        let first = run[0].1;
        let last = run[run.len() - 1].1;
        let bytes = self.reader.vector_bytes_for_rows(first..last + 1)?;
        for &(doc, row) in run {
            let vbytes = &bytes[(row - first) * stride..(row - first + 1) * stride];
            topn.push(self.query.score_doc_bytes(vbytes), doc);
        }
        Ok(())
    }

    /// Run-absorption decision for the flat path (see [`absorb_into_run`]).
    fn absorb_next_survivor(
        &self,
        first_row: usize,
        last_row: usize,
        next_row: usize,
        stride: usize,
    ) -> bool {
        #[cfg(test)]
        if let Some(forced) = self.forced_exact {
            return match forced {
                ForcedExactRead::PerDoc => false,
                ForcedExactRead::Chunked => true,
            };
        }
        absorb_into_run(first_row, last_row, next_row, stride)
    }
}

/// Bump the flushed-run counters: a run of `len > 1` survivors was one
/// chunked read; `len == 1` is the per-doc-equivalent single read.
fn tally_run(len: usize, chunked: &mut usize, single: &mut usize) {
    if len > 1 {
        *chunked += 1;
    } else {
        *single += 1;
    }
}

/// How the probe loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ProbeTermination {
    /// `probe_count >= max_probe_count` — the absolute probe ceiling.
    Ceiling,
    /// The distance-ratio gate fired with the survivor floor met.
    Gate,
    /// The ranked centroids were exhausted without hitting either stop.
    #[default]
    Exhausted,
}

/// Per-segment probe-loop instrumentation: which clusters were probed
/// (in probe order) and a prune breakdown of every doc the inner loop
/// touched. Filled by [`VectorBackend::top_n_with_stats`] when a sink is
/// supplied. The flat/exact path fills only `exact_reads_chunked` /
/// `exact_reads_single`; every other field is IVF-probe-only.
#[derive(Debug, Default)]
pub struct ProbeStats {
    /// Clusters visited by the probe loop, in probe order. A cluster
    /// appears here once we've passed the stop-condition gate for it,
    /// regardless of whether its doc-ids slice ends up empty.
    pub probed_clusters: Vec<usize>,
    /// Docs that passed filter + alive + seen and were scored against the
    /// query. This stays the "scored" bucket and equals the final survivor
    /// `candidates`, so starvation is just `candidates_scored < min_candidates`.
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
    /// Probed clusters whose posting bytes were fetched with one bulk
    /// whole-cluster ranged read (the gate pre-pass left a survivor set
    /// dense enough that one contiguous read beats per-row reads). Counts
    /// clusters, not rows.
    pub postings_bulk: usize,
    /// Probed clusters whose posting bytes were fetched with one
    /// stride-sized ranged read per surviving row (sparse survivor set —
    /// see [`row_fetch_is_cheaper`]). Counts clusters, not rows.
    pub postings_row: usize,
    /// Probed clusters that fetched no posting bytes at all: the
    /// `filter → alive → seen` pre-pass left zero survivors (fully
    /// filtered / dead / already-seen, or the cluster is empty). Together
    /// the three `postings_*` counters partition the probed clusters:
    /// `postings_bulk + postings_row + postings_skipped ==
    /// probed_clusters.len()`.
    pub postings_skipped: usize,
    /// Flat/exact-path ranged reads that covered a run of two or more
    /// surviving rows — one merged read scored several survivors (see
    /// [`absorb_into_run`]). Filled only by the exact (non-IVF) path;
    /// counts reads, not rows.
    pub exact_reads_chunked: usize,
    /// Flat/exact-path ranged reads that covered exactly one surviving
    /// row — byte-identical to the pre-chunking per-doc read. Filled only
    /// by the exact (non-IVF) path; counts reads, not rows.
    pub exact_reads_single: usize,
    /// Centroids scored to route this query (the navigation cost):
    /// `num_centroids` on the exact path, the beam-visited count when routed
    /// via the RNG.
    pub centroids_ranked: usize,
    /// The resolved survivor floor the gate used for this query.
    pub min_candidates: usize,
    /// How the probe loop terminated. Per-segment; does not sum.
    pub termination: ProbeTermination,
}

/// How many candidate docs the IVF probe loop is willing to score per
/// requested top-K result before the threshold gate is allowed to
/// terminate it. Combined with the user-supplied `min_candidates` at
/// the call site as `min_candidates.max(CANDIDATE_OVERFETCH_MULTIPLIER * top_n)`,
/// so a default `min_candidates = 0` still gives a sane floor.
///
/// The "4×" rule of thumb is intentionally conservative — enough
/// overfetch that one near-cluster with a tail of duplicates can't
/// short-circuit recall. Provisional; revisit alongside the other
/// adaptive defaults once real benchmarks land.
pub(crate) const CANDIDATE_OVERFETCH_MULTIPLIER: usize = 4;

/// How a probed cluster's posting bytes get fetched, decided per cluster
/// from the gate pre-pass's survivor count (see
/// [`VectorBackend::fetch_mode`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FetchMode {
    /// Zero survivors — no posting bytes fetched at all.
    Skip,
    /// One stride-sized ranged read per surviving row.
    Row,
    /// One contiguous ranged read of the whole cluster.
    Bulk,
}

/// Test-only [`FetchMode`] forcing, set through
/// [`VectorBackend::forced_fetch`].
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ForcedFetch {
    /// Bulk-fetch every probed cluster, survivors or not — the
    /// pre-change behavior, used as the equivalence baseline.
    Bulk,
    /// Per-row-fetch every cluster's survivors. Still [`FetchMode::Skip`]s
    /// at zero survivors: there is nothing to read.
    Row,
}

/// Test-only forcing of the flat/exact path's run batching, set through
/// [`VectorBackend::forced_exact`].
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ForcedExactRead {
    /// Never absorb — every survivor is its own length-1 run: the
    /// pre-chunking per-doc baseline.
    PerDoc,
    /// Always absorb — one run from the first survivor to the last,
    /// regardless of hole cost.
    Chunked,
}

/// One gate survivor from the pre-pass over a cluster's rows: `local_i`
/// indexes into the cluster's bulk byte buffer, `row` into the
/// segment-wide dense rows slot (`row == cluster_range.start + local_i`).
#[derive(Clone, Copy)]
struct Survivor {
    local_i: usize,
    row: usize,
    doc: DocId,
}

/// Blocks one row read touches under the cost model:
/// `read_blocks(stride) = ceil(stride / 8192) + 1` — 2 for any stride up
/// to one `FETCH_BLOCK_BYTES` block (e.g. a 768-dim f32 row, 3072 B), 3
/// for a 3072-dim f32 row (12288 B), and so on. Named separately from
/// [`read_blocks`] because it prices the *per-survivor* alternative in
/// both fetch policies below.
const fn per_row_blocks(stride: usize) -> usize {
    read_blocks(stride)
}

/// The IVF sparse/dense fetch cutoff: per-row reads win while the
/// survivors touch fewer ranged-read blocks than one bulk read of the
/// cluster.
///
/// One bulk read of a cluster of `n` rows × `stride` bytes touches
/// `read_blocks(n * stride) = ceil(n * stride / 8192) + 1` blocks at
/// arbitrary alignment, while each per-row read touches
/// [`per_row_blocks`]`(stride)`. So `s` per-row reads are the cheaper
/// fetch while
///
/// `s * per_row_blocks(stride) < ceil(n * stride / 8192) + 1`
///
/// Worked examples on a 100-row cluster:
/// - 768-dim f32 rows (stride 3072, 2 blocks/row): bulk = 300 KiB → 38 + 1 = 39 blocks; per-row
///   wins through s = 19.
/// - 3072-dim f32 rows (stride 12288, 3 blocks/row): bulk = 1.2 MiB → 150 + 1 = 151 blocks; per-row
///   wins only through s = 50. A flat 2-blocks-per-row pricing would have oversold per-row fetching
///   through s = 75 — fat rows justify bulk at fewer survivors.
const fn row_fetch_is_cheaper(survivors: usize, cluster_rows: usize, stride: usize) -> bool {
    survivors * per_row_blocks(stride) < read_blocks(cluster_rows * stride)
}

/// The flat/exact path's greedy run rule: absorb `next_row` into the
/// current run `[first_row, last_row]` iff one merged ranged read over the
/// grown span costs no more block touches than reading the run as it
/// stands plus the newcomer separately:
///
/// `read_blocks((next - first + 1) * stride) <= read_blocks((last - first + 1) * stride) +
/// per_row_blocks(stride)`
///
/// Rows inside the grown span that belong to no survivor (filtered,
/// deleted, or vectorless docs between them) are fetched but never scored
/// — that is exactly the trade this inequality prices: the hole's bytes
/// must cost no more block touches than a separate read for the newcomer
/// would. Adjacent survivors always absorb (the span grows by strictly
/// fewer blocks than a separate read); a run of length 1 reads exactly
/// the bytes a per-doc read would, so sparse filters degrade to the
/// pre-chunking behavior by construction.
const fn absorb_into_run(
    first_row: usize,
    last_row: usize,
    next_row: usize,
    stride: usize,
) -> bool {
    read_blocks((next_row - first_row + 1) * stride)
        <= read_blocks((last_row - first_row + 1) * stride) + per_row_blocks(stride)
}

impl<T: VectorElement> VectorBackend<T> {
    /// Test helper: run `top_n` with a fresh `ProbeStats` and return both.
    #[cfg(test)]
    pub(crate) fn top_n_instrumented(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let mut stats = ProbeStats::default();
        let hits = self.top_n_with_stats(weight, segment_reader, top_n, Some(&mut stats))?;
        Ok((hits, stats))
    }

    /// Top-N by IVF probe. When `stats` is `Some`, it is filled with this
    /// segment's probe-loop counters; `None` is the zero-cost production
    /// path.
    fn probe_top_n(
        &self,
        index: &IvfIndex,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        mut stats: Option<&mut ProbeStats>,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
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

        let stride = self.reader.options().bytes_per_vector();
        let num_centroids = index.num_clusters();
        if num_centroids == 0 {
            return Ok(Vec::new());
        }
        let max_probe_count = self.adaptive.resolved_probe_ceiling(num_centroids)?;

        // Asking for one candidate past the ceiling keeps the `Ceiling`
        // termination attribution meaningful.
        let (ranked, centroids_ranked) =
            self.rank_clusters(index, max_probe_count.saturating_add(1));
        if ranked.is_empty() {
            return Ok(Vec::new());
        }

        let best = ranked[0].0;
        let threshold = adaptive_threshold(self.query.metric(), best, self.adaptive.epsilon);
        // Without this floor, a selective filter can trip the threshold gate
        // immediately and return < K results.
        let min_candidates = self
            .adaptive
            .min_candidates
            .max(CANDIDATE_OVERFETCH_MULTIPLIER * top_n);

        if let Some(s) = stats.as_deref_mut() {
            s.centroids_ranked = centroids_ranked;
            s.min_candidates = min_candidates;
        }

        let topn = self.scan_clusters(
            index,
            ranked,
            threshold,
            min_candidates,
            max_probe_count,
            stride,
            &filter,
            max_doc,
            alive,
            top_n,
            stats,
        )?;

        let segment_ord = self.segment_ord;
        Ok(topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect())
    }

    /// Phase 1: rank the clusters to probe, best routing score first, plus
    /// the number of centroids scored to do it. Delegates to
    /// [`IvfIndex::rank_clusters`]; routing operates in `f32` (centroid rows
    /// are `f32` today), so the query is widened losslessly per element.
    /// `#[inline(never)]` so it forms its own flamegraph frame.
    #[inline(never)]
    fn rank_clusters(&self, index: &IvfIndex, limit: usize) -> (Vec<(f32, u32)>, usize) {
        let query_f32: Vec<f32> = self.query.query().iter().map(|e| e.to_f32()).collect();
        index.rank_clusters(&query_f32, limit)
    }

    /// Phase 2: adaptive probe loop. Each probed cluster is gated first —
    /// [`Self::collect_cluster_survivors`] runs `filter → alive → seen`
    /// off the pinned id-map with no posting bytes in hand — and only the
    /// survivors' bytes are then fetched, per [`Self::fetch_mode`]:
    /// nothing, per-row reads, or one bulk cluster read. Cluster-order
    /// arrival of survivors forbids the ascending-doc shortcut in `push`;
    /// use `push_unordered`.
    ///
    /// Note on `NaturalComparator` (vs the `TopNComputer::new` default):
    /// vector similarity is "higher = better", so we want top-N *largest*
    /// scores. The default `new()` wires `ReverseComparator`, which keeps
    /// top-N *smallest* — correct for ascending-distance metrics but inverted
    /// for our convention.
    ///
    /// `#[inline(never)]` so it forms its own flamegraph frame carrying its
    /// `score_doc_bytes` cost, distinct from `rank_clusters`.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn scan_clusters(
        &self,
        index: &IvfIndex,
        ranked: Vec<(f32, u32)>,
        threshold: f32,
        min_candidates: usize,
        max_probe_count: usize,
        stride: usize,
        filter: &BitSet,
        max_doc: DocId,
        alive: Option<&AliveBitSet>,
        top_n: usize,
        mut stats: Option<&mut ProbeStats>,
    ) -> crate::Result<TopNComputer<Score, DocId, NaturalComparator>> {
        let mut topn = TopNComputer::<Score, DocId, NaturalComparator>::new_with_comparator(
            top_n,
            NaturalComparator,
        );
        // `candidates` is the cumulative scored count that drives the gate; the
        // prune counters accumulate into locals and fold into `ProbeStats` once
        // after the loop, so the hot per-doc path carries no `Option` check.
        let mut candidates = 0usize;
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut pruned_seen = 0usize;
        let mut postings_bulk = 0usize;
        let mut postings_row = 0usize;
        let mut postings_skipped = 0usize;
        let mut termination = ProbeTermination::Exhausted;
        // Replication can place the same doc in several probed clusters; dedup
        // by doc id so a vector is scored at most once.
        let mut seen = BitSet::with_max_value(max_doc);
        // The probed cluster's gate survivors; allocated once, reused
        // across clusters.
        let mut survivors: Vec<Survivor> = Vec::new();

        for (probe_count, (centroid_score, cluster)) in ranked.into_iter().enumerate() {
            if probe_count >= max_probe_count {
                termination = ProbeTermination::Ceiling;
                break;
            }
            if centroid_score < threshold && candidates >= min_candidates {
                termination = ProbeTermination::Gate;
                break;
            }
            let cluster = cluster as usize;

            // Record the probe before doing any work, so even an empty
            // cluster counts as "probed".
            if let Some(s) = stats.as_deref_mut() {
                s.probed_clusters.push(cluster);
            }

            let rows = index.cluster_range(cluster);
            let num_rows = rows.len();

            // Pre-pass: run the gate off the pinned id-map alone, BEFORE
            // any posting bytes are fetched, so the fetch below can be
            // sized to — or skipped for — the rows that will actually be
            // scored. Gate order, the `seen` marking point, and every
            // prune counter are exactly the fetch-then-gate scan's; only
            // the byte fetch moved.
            let (v, pf, pd, ps) =
                self.collect_cluster_survivors(rows, filter, alive, &mut seen, &mut survivors);
            visited += v;
            pruned_filter += pf;
            pruned_dead += pd;
            pruned_seen += ps;

            match self.fetch_mode(survivors.len(), num_rows, stride) {
                FetchMode::Skip => postings_skipped += 1,
                FetchMode::Row => {
                    postings_row += 1;
                    for &Survivor { row, doc, .. } in &survivors {
                        let vbytes = self.reader.vector_bytes_for_row(row)?;
                        topn.push_unordered(self.query.score_doc_bytes(&vbytes), doc);
                    }
                }
                FetchMode::Bulk => {
                    postings_bulk += 1;
                    // One contiguous ranged read for the whole cluster —
                    // under a copying `Directory`, only this cluster's
                    // bytes are materialized.
                    let cluster_vec_bytes = self.reader.cluster_vector_bytes(cluster)?;
                    for &Survivor { local_i, doc, .. } in &survivors {
                        let vbytes = &cluster_vec_bytes[local_i * stride..(local_i + 1) * stride];
                        topn.push_unordered(self.query.score_doc_bytes(vbytes), doc);
                    }
                }
            }
            candidates += survivors.len();
        }

        if let Some(s) = stats {
            s.vectors_visited += visited;
            s.pruned_filter += pruned_filter;
            s.pruned_dead += pruned_dead;
            s.pruned_seen += pruned_seen;
            s.postings_bulk += postings_bulk;
            s.postings_row += postings_row;
            s.postings_skipped += postings_skipped;
            s.candidates_scored += candidates;
            s.termination = termination;
        }

        Ok(topn)
    }

    /// Phase 2 pre-pass: run one cluster's rows through the
    /// `filter → alive → seen` gate — off the pinned id-map alone, with no
    /// posting bytes fetched — collecting into `survivors` (cleared first)
    /// the rows to score. `seen` is marked here, at gate-pass time, NOT at
    /// scoring time, so replica dedup counts across clusters are identical
    /// to the fetch-then-gate scan this pre-pass replaced.
    /// Returns `(visited, pruned_filter, pruned_dead, pruned_seen)`.
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
    ) -> (usize, usize, usize, usize) {
        survivors.clear();
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut pruned_seen = 0usize;
        for (local_i, row) in rows.enumerate() {
            let doc = self.reader.doc_id_at(row);
            visited += 1;
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
            if seen.contains(doc) {
                pruned_seen += 1;
                continue;
            }
            seen.insert(doc);
            survivors.push(Survivor { local_i, row, doc });
        }
        (visited, pruned_filter, pruned_dead, pruned_seen)
    }

    /// Per-cluster posting-fetch dispatch from the pre-pass survivor
    /// count: nothing survived → fetch nothing; a sparse survivor set →
    /// per-row reads; dense → one bulk cluster read.
    fn fetch_mode(&self, survivors: usize, cluster_rows: usize, stride: usize) -> FetchMode {
        #[cfg(test)]
        if let Some(forced) = self.forced_fetch {
            return match forced {
                ForcedFetch::Bulk => FetchMode::Bulk,
                ForcedFetch::Row if survivors == 0 => FetchMode::Skip,
                ForcedFetch::Row => FetchMode::Row,
            };
        }
        if survivors == 0 {
            FetchMode::Skip
        } else if row_fetch_is_cheaper(survivors, cluster_rows, stride) {
            FetchMode::Row
        } else {
            FetchMode::Bulk
        }
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

/// Per-metric distance-ratio pruning threshold (SPANN eq. 3): a posting
/// list is searched iff `Dist(q, c) <= (1 + epsilon) * Dist(q, c_closest)`,
/// re-expressed on the similarity scale (higher = better) so the probe
/// loop compares scores directly. `best` is the top-ranked centroid's
/// score.
///
/// - **L2:** `score = -d²`, so `threshold = best - epsilon * best.abs()` is `d² > (1 + eps) *
///   d²_min` — SPANN's inequality verbatim (their `Dist` is squared L2).
/// - **Cosine:** `threshold = best - epsilon * (1 - best)` gates on `(1 - score) > (1 + eps) * (1 -
///   best)`. For unit vectors `d² = 2(1 - cos)` and the 2 cancels in the ratio, so this IS SPANN's
///   rule applied to our (write-time-normalized) data.
/// - **Dot:** no natural distance for raw MIPS; a pragmatic linear widening `best - epsilon *
///   best.abs()`. With paper-scale epsilon the gate rarely fires and the ceiling governs. NOTE:
///   with unnormalized dot, the IVF locality assumption itself is heuristic — that's the
///   clusterer's problem, not the threshold's.
///
/// Degenerate scales: L2 with `d_min = 0` and Cosine with `best = 1.0`
/// both give `threshold = best` — the gate arms immediately and only
/// the candidate floor keeps probing. Known property of ratio pruning;
/// do not "fix".
fn adaptive_threshold(metric: Metric, best: f32, epsilon: f32) -> f32 {
    match metric {
        Metric::L2 | Metric::Dot => best - epsilon * best.abs(),
        Metric::Cosine => best - epsilon * (1.0 - best),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_threshold_identity_at_zero_epsilon() {
        // With epsilon = 0 the threshold is exactly `best` for every
        // metric — no ratio slack, no permissiveness.
        for &best in &[-10.0_f32, -1.0, 0.0, 0.5, 1.0] {
            assert_eq!(adaptive_threshold(Metric::L2, best, 0.0), best);
            assert_eq!(adaptive_threshold(Metric::Cosine, best, 0.0), best);
            assert_eq!(adaptive_threshold(Metric::Dot, best, 0.0), best);
        }
    }

    #[test]
    fn adaptive_threshold_lowers_with_positive_epsilon() {
        // "Higher score = closer" convention; ratio slack means the
        // threshold is *lower* (more permissive) than `best`.
        let eps = 0.1;
        // L2 similarity is `-d²`, so `best` is always ≤ 0 and
        // `best - eps * |best|` is more negative (= more permissive).
        // For best = 0 the threshold is also 0 (d_min = 0 — the
        // degenerate ratio scale; the gate arms immediately).
        for &best in &[-10.0_f32, -1.0, -0.001] {
            let l2 = adaptive_threshold(Metric::L2, best, eps);
            assert!(l2 < best, "L2 threshold {l2} should be < best {best}");
        }
        let cos_best = 0.8;
        let cos = adaptive_threshold(Metric::Cosine, cos_best, eps);
        assert!(
            cos < cos_best,
            "Cosine threshold {cos} should be < {cos_best}"
        );

        // Dot: pinned linear widening. Lower than `best` for positive
        // `best`; *also* lower (more negative) for negative `best`,
        // because we subtract `eps * |best|`, never add. This is the
        // intentional behavior — `best - eps * |best|` is monotonic
        // in the "more permissive" direction regardless of sign.
        let pos = adaptive_threshold(Metric::Dot, 10.0, eps);
        assert!(pos < 10.0, "Dot threshold {pos} should be < 10.0");
        let neg = adaptive_threshold(Metric::Dot, -10.0, eps);
        assert!(neg < -10.0, "Dot threshold {neg} should be < -10.0");
    }

    #[test]
    fn adaptive_threshold_hand_checked_values() {
        // L2: best = -10 (d² = 10), eps = 0.1 ⇒ -10 - 0.1·10 = -11,
        // i.e. gate at d² > 1.1 · d²_min.
        let l2 = adaptive_threshold(Metric::L2, -10.0, 0.1);
        assert!((l2 - -11.0).abs() < 1e-5, "got {l2}");

        // Cosine: best = 0.8, eps = 0.1 ⇒ 0.8 - 0.1 · 0.2 = 0.78.
        let cos = adaptive_threshold(Metric::Cosine, 0.8, 0.1);
        assert!((cos - 0.78).abs() < 1e-5, "got {cos}");

        // Cosine at paper-scale epsilon: best = 0.9, eps = 7.0 ⇒
        // 0.9 - 7 · 0.1 = 0.2 — the gate CAN fire on realistic angular
        // gaps (a |best|-scaled threshold would sit at -5.4 and never
        // trip on the cosine range).
        let cos_wide = adaptive_threshold(Metric::Cosine, 0.9, 7.0);
        assert!((cos_wide - 0.2).abs() < 1e-5, "got {cos_wide}");

        // Dot: pinned `best - eps * |best|`.
        // best =  10, eps = 0.1 ⇒  9.0
        // best = -10, eps = 0.1 ⇒ -11.0
        let dot_pos = adaptive_threshold(Metric::Dot, 10.0, 0.1);
        assert!((dot_pos - 9.0).abs() < 1e-5, "got {dot_pos}");
        let dot_neg = adaptive_threshold(Metric::Dot, -10.0, 0.1);
        assert!((dot_neg - -11.0).abs() < 1e-5, "got {dot_neg}");
        // Origin: degenerate (query orthogonal to nearest centroid);
        // threshold collapses to 0 because |0| = 0.
        let dot_zero = adaptive_threshold(Metric::Dot, 0.0, 0.5);
        assert_eq!(dot_zero, 0.0);
    }

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

    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{
        AllQuery, BitSetDocSet, ConstScorer, EnableScoring, Explanation, Query, Scorer, TermQuery,
    };
    use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
    use crate::vector::tests::{exhaustive_params, TestVectorIndex};
    use crate::vector::{
        IvfCentroids, IvfClusterer, IvfMatrix, IvfMergeSettings, IvfVectors, VectorClusterStats,
        VectorDType, VectorInfo, VectorOptions, VectorStorageFormat,
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
        index.reader()?.searcher().search(filter, &collector)
    }

    /// Probe-stat helper: run `VectorBackend::top_n_instrumented` against
    /// the first segment of `index` and return (hits, stats).
    /// The contracts are per-segment, so collecting from segment 0 is
    /// what each assertion is talking about.
    fn run_top_n_instrumented(
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
        backend.top_n_instrumented(weight.as_ref(), segment_reader, k)
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

        let (_, stats) = run_top_n_instrumented(
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
            epsilon: 0.0,
            min_candidates: usize::MAX,
            max_probe_count: 1,
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

    /// `min_candidates` floor: cluster A has one doc near the query;
    /// cluster B holds the true NN. Without the floor, the threshold
    /// trips immediately after A (epsilon=0) and the loop stops; the
    /// floor (`CANDIDATE_OVERFETCH_MULTIPLIER * top_n`) forces it to
    /// keep probing into B. INLINE because the shared fixture's
    /// uniform-grid points don't naturally produce a "near cluster
    /// with one survivor" geometry.
    ///
    /// Setup assertions below pin the geometry so the test can't quietly
    /// rot vacuous if a doc drifts across the bisector x+y=10 — it has
    /// happened before (a_only was originally close enough to the query
    /// to BE the top-1, which let A alone satisfy top-k and made the
    /// floor irrelevant). The assertions enforce: top-1 lives in B,
    /// `a_only` lives in A, and A has fewer survivors than the floor —
    /// jointly, reaching the answer REQUIRES probing B.
    #[test]
    fn ivf_top_n_min_candidates_floor() -> crate::Result<()> {
        let centroids = vec![[0.0_f32, 0.0], [10.0, 10.0]];
        // a_only is on the A side (closer to (0,0) than (10,10)) but
        // *deliberately far* from the query so b_close is the true
        // NN. Without the floor, the loop stops after A — recall = 0.
        // With the floor, it probes B and finds b_close.
        let docs = [
            ("a_only", [0.0_f32, -10.0]), // A-side, far from query
            ("b_close", [5.0_f32, 5.01]), // B-side, true NN
            ("b_far", [10.0_f32, 10.0]),
            ("b_far2", [11.0_f32, 9.5]),
        ];
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        let query = [1.0_f32, 1.0];
        let top_k = 1;

        // Open segment 0's IVF reader for the geometry assertions.
        // After `build_inline_ivf`'s merge, all docs sit in segment 0.
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        assert!(
            vec_reader.index().is_some(),
            "expected IVF segment for this test"
        );
        // Setup assertion (i): b_close is the brute-force top-1, and
        // its vector maps to cluster B (index 1). Mirrors the trap
        // test's `assert_eq!(oracle[0].1, trap_doc)` — this is the
        // assertion whose absence let the test rot vacuous.
        let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, 1)?;
        let oracle_addr = expected[0].1;
        assert_eq!(
            stored_label_at(&index, label_field, oracle_addr)?,
            "b_close",
            "test geometry: b_close must be the true NN",
        );
        let oracle_bytes = vec_reader
            .vector_bytes(oracle_addr.doc_id)?
            .expect("oracle vector bytes");
        assert_eq!(
            nearest_centroid(decode_2d(&oracle_bytes), &centroids),
            1,
            "oracle top-1 must live in cluster B — the far cluster the floor has to reach",
        );

        // Setup assertion (ii): a_only still lands in cluster A. If
        // [0,-10] ever drifts across the bisector x+y=10 (it won't with
        // these coords, but coordinates evolve), the premise "the near
        // cluster has too few survivors" stops holding — the test
        // would no longer exercise the floor.
        let cluster_a_docs = vec_reader.cluster_doc_ids(0).unwrap_or_default();
        let mut a_only_doc = None;
        for doc in 0..segment_reader.max_doc() {
            if stored_label_at(&index, label_field, DocAddress::new(0, doc))? == "a_only" {
                a_only_doc = Some(doc);
                break;
            }
        }
        let a_only_doc = a_only_doc.expect("a_only must exist in segment 0");
        assert!(
            cluster_a_docs.contains(&a_only_doc),
            "a_only must land in cluster A (index 0) — got cluster_a = {cluster_a_docs:?}, a_only \
             doc = {a_only_doc}",
        );

        // Setup assertion (iii): the near cluster has fewer survivors
        // than the candidate floor (4 × top_k = 4). Combined with (i),
        // reaching the oracle's top-1 REQUIRES probing B — which only
        // the floor causes, since epsilon=0 trips the threshold gate
        // immediately after A.
        assert!(
            cluster_a_docs.len() < CANDIDATE_OVERFETCH_MULTIPLIER * top_k,
            "cluster A must have fewer than the candidate floor ({}) for the floor to actually \
             have to probe out — got {} docs",
            CANDIDATE_OVERFETCH_MULTIPLIER * top_k,
            cluster_a_docs.len(),
        );

        // Behavioral check: epsilon=0 trips the threshold after A;
        // only the candidate floor keeps the loop probing into B.
        let params = AdaptiveProbeParams {
            epsilon: 0.0,
            min_candidates: 0,
            max_probe_count: usize::MAX,
        };
        let hits = search(
            &index,
            embed_field,
            &AllQuery,
            query.to_vec(),
            top_k,
            params,
        )?;
        assert_eq!(hits, expected);
        assert_eq!(
            stored_label_at(&index, label_field, hits[0].1)?,
            "b_close",
            "floor must keep probing past A to find the B-side true NN",
        );
        Ok(())
    }

    /// `top_n == 0` returns empty without touching the column. The
    /// collector layer rejects `TopDocs::with_limit(0)` before it
    /// reaches the backend, so this test calls the backend directly
    /// via the instrumented seam — the short-circuit lives in
    /// `probe_top_n`.
    #[test]
    fn ivf_top_n_zero_returns_empty() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let (hits, stats) = run_top_n_instrumented(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            0,
            AdaptiveProbeParams::default(),
        )?;
        assert!(hits.is_empty());
        // Short-circuit fires before the probe loop, so no clusters
        // visited and no candidates scored.
        assert!(stats.probed_clusters.is_empty());
        assert_eq!(stats.candidates_scored, 0);
        Ok(())
    }

    /// Smoke for the instrumented seam: probed_clusters is non-empty,
    /// every entry is < num_centroids, and candidates_scored ≤ total
    /// docs in the inspected segment. Exhaustive params on a 9-centroid
    /// segment visit all 9.
    #[test]
    fn ivf_top_n_instrumented_collects_probe_stats() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let (_, stats) = run_top_n_instrumented(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            4,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(stats.probed_clusters.len(), DEFAULT_NUM_CENTROIDS);
        for &c in &stats.probed_clusters {
            assert!(c < DEFAULT_NUM_CENTROIDS, "probed cluster {c} out of range");
        }
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
        assert_eq!(stats.centroids_ranked, DEFAULT_NUM_CENTROIDS);
        // Exhaustive params (unclamped ceiling, unsatisfiable floor)
        // drain the ranked list.
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        Ok(())
    }

    /// A `max_probe_count` below the cluster count forces the hard ceiling:
    /// the loop stops with `termination == Ceiling`, having probed exactly
    /// the cap, and the counter invariant still holds. Uses the deterministic
    /// `build_inline_ivf` fixture (fixed 6 centroids) so the cutoff is stable.
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
            epsilon: 0.0,
            min_candidates: usize::MAX,
            max_probe_count: 1,
        };
        let (_, stats) = run_top_n_instrumented(&index, embed_field, vec![10.0, 10.0], 3, params)?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        // Stopped at exactly the cap, short of the ranked list.
        assert_eq!(stats.probed_clusters.len(), 1);
        assert_eq!(stats.centroids_ranked, centroids.len());
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
        let (hits, stats) = run_top_n_instrumented(
            &index,
            embed_field,
            query.to_vec(),
            k,
            AdaptiveProbeParams::default(),
        )?;
        assert_eq!(hits, expected, "linear fallback must match the oracle");
        assert_eq!(stats.probed_clusters, vec![0], "one cluster, one probe");
        assert_eq!(stats.centroids_ranked, 1);
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
            epsilon: 7.0,
            min_candidates: 0,
            max_probe_count: 2,
        };
        let k = 3usize;
        for (ord, centroid) in centroids.iter().enumerate().step_by(3) {
            let query = [centroid[0] + 0.3, centroid[1] - 0.2];
            let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, k)?;
            let (hits, stats) =
                run_top_n_instrumented(&index, embed_field, query.to_vec(), k, params.clone())?;
            assert_eq!(hits, expected, "routed top-{k} near centroid {ord}");
            assert!(
                stats.probed_clusters.len() <= 2,
                "cap 2 must bound the probes, got {:?}",
                stats.probed_clusters
            );
            assert!(
                stats.centroids_ranked <= centroids.len(),
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

    /// A cap of 2 on the 9-centroid fixture ⇒ the loop probes exactly
    /// the cap and attributes the stop to the ceiling, regardless of
    /// how generous the other knobs are.
    #[test]
    fn probe_stats_max_probe_count_ceiling() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = AdaptiveProbeParams {
            epsilon: 0.0,
            min_candidates: usize::MAX,
            max_probe_count: 2,
        };
        // The cap must actually bind for this test to mean anything.
        assert!(
            params.resolved_probe_ceiling(DEFAULT_NUM_CENTROIDS)? < DEFAULT_NUM_CENTROIDS,
            "resolved ceiling does not bind",
        );
        let (_, stats) = run_top_n_instrumented(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            3,
            params,
        )?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert_eq!(
            stats.probed_clusters.len(),
            2,
            "absolute cap ⇒ exactly 2 probed, got {:?}",
            stats.probed_clusters,
        );
        Ok(())
    }

    /// Candidate floor: regardless of how stingy the threshold gate
    /// is, the loop scores at least `min(total_docs, resolved_floor)`
    /// docs. Threshold maximally stingy (`epsilon = 0`) and the
    /// ceiling unbounded, so the floor is the binding constraint.
    #[test]
    fn probe_stats_min_candidates_floor_scores_floor_or_total() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let top_k = 4;
        let resolved_floor = CANDIDATE_OVERFETCH_MULTIPLIER * top_k;
        let segment_doc_count =
            index.index.reader()?.searcher().segment_readers()[0].max_doc() as usize;
        let expected_min = segment_doc_count.min(resolved_floor);

        let params = AdaptiveProbeParams {
            epsilon: 0.0,
            min_candidates: 0,
            max_probe_count: usize::MAX,
        };
        let (_, stats) = run_top_n_instrumented(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            top_k,
            params,
        )?;
        assert!(
            stats.candidates_scored >= expected_min,
            "candidate floor (resolved {resolved_floor}, segment {segment_doc_count}) ⇒ ≥ \
             {expected_min} candidates scored; got {}",
            stats.candidates_scored,
        );
        Ok(())
    }

    /// The distance-ratio gate fires on Cosine at paper-scale epsilon:
    /// with write-time-normalized centroids, a wide angular gap puts
    /// the far centroid below `best - eps * (1 - best)` once the
    /// survivor floor is met. (Under a `|best|`-scaled threshold the
    /// gate could never fire on the [0, 1] cosine range at eps = 7.)
    #[test]
    fn probe_stats_cosine_gate_fires() -> crate::Result<()> {
        // Cluster A hugs the x-axis (4 docs — exactly the 4·top_k
        // floor); cluster B hugs the y-axis, far outside the ratio.
        let centroids = vec![[10.0_f32, 0.0], [0.0, 10.0]];
        let docs = [
            ("a0", [10.0_f32, 0.0]),
            ("a1", [10.0_f32, 0.2]),
            ("a2", [9.8_f32, 0.1]),
            ("a3", [10.1_f32, 0.3]),
            ("b0", [0.0_f32, 10.0]),
            ("b1", [0.2_f32, 9.9]),
        ];
        let (index, embed_field, _label) = build_inline_ivf(Metric::Cosine, &centroids, &docs, 1)?;

        // Query ~17° off the x-axis: best ≈ cos 17° ≈ 0.958, threshold
        // ≈ 0.958 - 7 · 0.042 ≈ 0.66; centroid B scores ≈ 0.29 < 0.66.
        let params = AdaptiveProbeParams {
            epsilon: 7.0,
            min_candidates: 0,
            max_probe_count: usize::MAX,
        };
        let (_, stats) = run_top_n_instrumented(&index, embed_field, vec![1.0, 0.3], 1, params)?;
        assert_eq!(stats.termination, ProbeTermination::Gate);
        assert_eq!(
            stats.probed_clusters.len(),
            1,
            "gate must stop before the far angular cluster ({:?})",
            stats.probed_clusters,
        );
        Ok(())
    }

    /// With default adaptive params and a query right on one cluster's
    /// centroid, the probe loop should prune — visit strictly fewer
    /// clusters than the segment's total. Loose contract: no exact
    /// number, stays stable when defaults are tuned.
    #[test]
    fn probe_stats_pruning_happens() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        // Query at the first centroid — maximally biased toward cluster 0.
        let query = grid2d_first_centroid();
        let (_, stats) = run_top_n_instrumented(
            &index.index,
            index.embedding_field(),
            query.to_vec(),
            4,
            AdaptiveProbeParams::default(),
        )?;
        assert!(
            stats.probed_clusters.len() < DEFAULT_NUM_CENTROIDS,
            "default-params pruning should visit strictly fewer than {DEFAULT_NUM_CENTROIDS} \
             clusters; got {} ({:?})",
            stats.probed_clusters.len(),
            stats.probed_clusters,
        );
        Ok(())
    }

    /// Structural invariants on the probe stats themselves —
    /// independent of any specific stop-condition behavior.
    ///   - all probed indices live in [0, num_centroids)
    ///   - no duplicates (a cluster is probed at most once)
    ///   - the first probed cluster is the centroid nearest the query
    #[test]
    fn probe_stats_probed_clusters_validity() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let query = [9.0_f32, 0.5];
        let (_, stats) = run_top_n_instrumented(
            &index.index,
            index.embedding_field(),
            query.to_vec(),
            2,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;

        for &c in &stats.probed_clusters {
            assert!(
                c < DEFAULT_NUM_CENTROIDS,
                "probed cluster {c} out of range (num_centroids={DEFAULT_NUM_CENTROIDS})",
            );
        }
        let unique: std::collections::HashSet<usize> =
            stats.probed_clusters.iter().copied().collect();
        assert_eq!(
            unique.len(),
            stats.probed_clusters.len(),
            "duplicate probed cluster: {:?}",
            stats.probed_clusters,
        );

        let nearest = nearest_centroid_to(&query);
        assert_eq!(
            stats.probed_clusters.first().copied(),
            Some(nearest),
            "first probed should be the centroid nearest the query; nearest = {nearest}, \
             probed_clusters = {:?}",
            stats.probed_clusters,
        );
        Ok(())
    }

    // ============================================================
    // Filter-aware posting fetches.
    //
    // The probe loop decides each cluster's survivors from the pinned
    // id-map BEFORE touching posting bytes, then fetches nothing /
    // per-row / bulk by survivor density (`row_fetch_is_cheaper`).
    // These tests pin (a) result- and counter-equivalence against the
    // forced-bulk (pre-change) baseline, (b) the dispatch itself, and
    // (c) that the three `postings_*` counters partition the probed
    // clusters.
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

    /// Like [`run_top_n_instrumented`] but with a caller-supplied filter
    /// weight and an optional [`ForcedFetch`] override (`None` = the
    /// production dispatch).
    fn run_top_n_forced(
        index: &Index,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
        weight: &dyn Weight,
        forced_fetch: Option<ForcedFetch>,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let mut backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            params,
        )?;
        backend.forced_fetch = forced_fetch;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_some(),
            "expected IVF storage"
        );
        backend.top_n_instrumented(weight, segment_reader, k)
    }

    /// The two partition identities every scan must uphold: each touched
    /// row lands in exactly one prune bucket, and each probed cluster
    /// takes exactly one fetch path.
    fn assert_stats_identities(stats: &ProbeStats) {
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
            "visited must equal filter+dead+seen+scored ({stats:?})"
        );
        assert_eq!(
            stats.probed_clusters.len(),
            stats.postings_bulk + stats.postings_row + stats.postings_skipped,
            "probed clusters must partition into bulk+row+skipped ({stats:?})"
        );
    }

    /// Probe-loop counter equality between a natural-dispatch run and its
    /// forced-bulk baseline — the "semantics preserved bit-for-bit" claim
    /// for everything except the `postings_*` fetch counters themselves.
    fn assert_prune_counters_match(stats: &ProbeStats, baseline: &ProbeStats) {
        assert_eq!(
            stats.probed_clusters, baseline.probed_clusters,
            "probe order"
        );
        assert_eq!(stats.vectors_visited, baseline.vectors_visited, "visited");
        assert_eq!(stats.pruned_filter, baseline.pruned_filter, "pruned_filter");
        assert_eq!(stats.pruned_dead, baseline.pruned_dead, "pruned_dead");
        assert_eq!(stats.pruned_seen, baseline.pruned_seen, "pruned_seen");
        assert_eq!(
            stats.candidates_scored, baseline.candidates_scored,
            "candidates_scored"
        );
        assert_eq!(stats.termination, baseline.termination, "termination");
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
    /// caller-supplied filter weight and an optional [`ForcedExactRead`]
    /// override (`None` = the production run rule). Takes the segment
    /// reader — not the index — so callers can watch one
    /// `VectorIndexReader`'s `read_block_touches` across several runs.
    fn run_exact_on_segment(
        segment_reader: &SegmentReader,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        weight: &dyn Weight,
        forced_exact: Option<ForcedExactRead>,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let mut backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            AdaptiveProbeParams::default(),
        )?;
        backend.forced_exact = forced_exact;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_none(),
            "expected flat storage"
        );
        backend.top_n_instrumented(weight, segment_reader, k)
    }

    /// Dim-generic inline clusterer for fat-stride fixtures: centroids
    /// spaced 100 apart along the first axis, docs assigned by full-L2
    /// nearest centroid.
    struct WideClusterer {
        dim: usize,
        num_centroids: usize,
    }

    impl IvfClusterer for WideClusterer {
        fn centroid_ratio(&self) -> f32 {
            1.0
        }
        fn training_samples_per_centroid(&self) -> usize {
            2
        }
        fn merge_settings(&self, _total_target_docs: usize) -> crate::Result<IvfMergeSettings> {
            Ok(IvfMergeSettings {
                num_centroids: self.num_centroids,
                training_samples_per_centroid: self.training_samples_per_centroid(),
                assign_batch_size: self.assign_batch_size(),
                replicas: 1,
            })
        }
        fn train(
            &self,
            options: &VectorOptions,
            _vectors: IvfVectors<'_>,
            num_centroids: usize,
        ) -> crate::Result<IvfCentroids> {
            assert_eq!(options.dim(), self.dim);
            let mut values = vec![0.0f32; num_centroids * self.dim];
            for k in 0..num_centroids {
                values[k * self.dim] = k as f32 * 100.0;
            }
            Ok(IvfCentroids::F32(IvfMatrix {
                values,
                rows: num_centroids,
                dims: self.dim,
            }))
        }
        fn assign(
            &self,
            options: &VectorOptions,
            vectors: IvfVectors<'_>,
            centroids: &IvfCentroids,
        ) -> crate::Result<Vec<u32>> {
            assert_eq!(options.dim(), self.dim);
            let IvfVectors::F32(vectors) = vectors;
            let IvfCentroids::F32(centroids) = centroids;
            Ok(vectors
                .matrix
                .values
                .chunks_exact(self.dim)
                .map(|v| {
                    let mut best = 0u32;
                    let mut best_d2 = f32::INFINITY;
                    for (i, c) in centroids.values.chunks_exact(self.dim).enumerate() {
                        let d2: f32 = v.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
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

    /// Single-IVF-segment index of `num_clusters × n_per` docs carrying
    /// `dim`-wide f32 vectors (stride = 4 · dim), clustered by
    /// [`WideClusterer`]. Doc `i` sits near centroid `i / n_per`.
    fn build_wide_ivf(
        dim: usize,
        num_clusters: usize,
        n_per: usize,
    ) -> crate::Result<(Index, Field)> {
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(dim, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(sb.build())
            .settings(settings)
            .ivf_clusterer(Arc::new(WideClusterer {
                dim,
                num_centroids: num_clusters,
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let total = num_clusters * n_per;
        for i in 0..total {
            let mut v = vec![0.0f32; dim];
            v[0] = (i / n_per) as f32 * 100.0 + (i % n_per) as f32 * 0.001;
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, format!("d{i}"));
            doc.add_vector(embed_field, v.as_slice());
            writer.add_document(doc)?;
            // Two source segments so `merge_ivf` has ≥ 2 inputs.
            if i + 1 == total / 2 {
                writer.commit()?;
            }
        }
        writer.commit()?;
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;
        Ok((index, embed_field))
    }

    /// Hand-checked values for the fetch pricing: the IVF sparse/dense
    /// cutoff `s * per_row_blocks(stride) < ceil(n * stride / 8192) + 1`
    /// and the flat path's run-absorption inequality.
    #[test]
    fn row_fetch_threshold_math() {
        // Stride-aware per-row pricing: one row read costs
        // ceil(stride / 8192) + 1 blocks.
        assert_eq!(per_row_blocks(8), 2);
        assert_eq!(per_row_blocks(3072), 2);
        assert_eq!(per_row_blocks(8192), 2);
        assert_eq!(per_row_blocks(8193), 3);
        assert_eq!(per_row_blocks(12288), 3);

        // Tiny cluster: 6 rows × 8 B = 48 B → 1 + 1 = 2 bulk blocks. One
        // survivor already touches up to 2 blocks — bulk.
        assert!(!row_fetch_is_cheaper(1, 6, 8));
        // Exactly one 8 KiB block (1024 rows × 8 B): still 2 bulk blocks.
        assert!(!row_fetch_is_cheaper(1, 1024, 8));
        // One byte past a block boundary tips it: 2 + 1 = 3 bulk blocks,
        // so a lone survivor row-fetches, two survivors don't.
        assert!(row_fetch_is_cheaper(1, 1025, 8));
        assert!(!row_fetch_is_cheaper(2, 1025, 8));
        // 3000 rows × 8 B = 24000 B → 3 + 1 = 4 bulk blocks.
        assert!(row_fetch_is_cheaper(1, 3000, 8));
        assert!(!row_fetch_is_cheaper(2, 3000, 8));
        // Realistic stride (768-dim f32 = 3072 B): a 100-row cluster is
        // 300 KiB → 38 + 1 = 39 bulk blocks — per-row wins through s = 19.
        assert!(row_fetch_is_cheaper(19, 100, 3072));
        assert!(!row_fetch_is_cheaper(20, 100, 3072));
        // Fat rows (3072-dim f32 = 12288 B, 3 blocks/row): 100-row bulk =
        // 150 + 1 = 151 blocks; per-row wins only through s = 50
        // (3 · 50 = 150 < 151). The old flat 2-blocks-per-row pricing
        // would have oversold per-row through s = 75 — fewer survivors
        // justify bulk when rows are fat.
        assert!(row_fetch_is_cheaper(50, 100, 12288));
        assert!(!row_fetch_is_cheaper(51, 100, 12288));

        // Run absorption, fat stride: a 1-row hole merges (span of 3 rows
        // = 36864 B → 5 + 1 = 6 blocks, vs 3 + 3 for run-plus-separate),
        // a 2-row hole splits (49152 B → 6 + 1 = 7 > 6). Flat 2-block row
        // pricing would have split even the 1-row hole (6 > 3 + 2).
        assert!(absorb_into_run(0, 0, 2, 12288));
        assert!(!absorb_into_run(0, 0, 3, 12288));
        // Adjacent survivors always absorb, at any stride.
        assert!(absorb_into_run(0, 0, 1, 8));
        assert!(absorb_into_run(0, 0, 1, 12288));
        // Thin rows absorb holes generously: 8 B rows pack 1024 per
        // block, so the gap only splits past ~2 blocks' worth of rows.
        assert!(absorb_into_run(0, 0, 3071, 8));
        assert!(!absorb_into_run(0, 0, 3072, 8));
    }

    /// Result equivalence across filter selectivities: on the replicated
    /// multi-cluster fixture, natural fetch dispatch returns doc-id- and
    /// score-identical hits to the forced-bulk (pre-change) baseline for
    /// hand-built filters admitting {0, 1, 50, 100}% of docs, with
    /// identical probe-loop counters and both partition identities
    /// intact. 0% admits nothing — the empty-filter short-circuit returns
    /// before the probe loop, so zero clusters probe and zero fetches of
    /// either kind happen (postings_skipped == clusters probed == 0).
    #[test]
    fn filter_aware_fetch_matches_forced_bulk_across_selectivities() -> crate::Result<()> {
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
            let (hits, stats) = run_top_n_forced(
                &index,
                embed_field,
                vec![10.0, 10.0],
                n,
                params.clone(),
                &weight,
                None,
            )?;
            let (bulk_hits, bulk_stats) = run_top_n_forced(
                &index,
                embed_field,
                vec![10.0, 10.0],
                n,
                params.clone(),
                &weight,
                Some(ForcedFetch::Bulk),
            )?;

            assert_eq!(
                hits, bulk_hits,
                "{pct}%: doc ids and scores must match forced-bulk"
            );
            assert_eq!(
                hits.len(),
                admitted,
                "{pct}%: every admitted doc returns exactly once"
            );
            assert_stats_identities(&stats);
            assert_stats_identities(&bulk_stats);
            assert_prune_counters_match(&stats, &bulk_stats);

            match pct {
                0 => {
                    assert_eq!(stats.postings_bulk, 0, "0%: no bulk fetches");
                    assert_eq!(stats.postings_row, 0, "0%: no row fetches");
                    assert_eq!(
                        stats.postings_skipped,
                        stats.probed_clusters.len(),
                        "0%: every probed cluster skips its fetch"
                    );
                }
                1 => {
                    // One admitted doc → exactly one cluster fetches (the
                    // first-probed of its replica cells); every other
                    // probed cluster skips.
                    assert_eq!(stats.postings_bulk + stats.postings_row, 1, "{stats:?}");
                    assert_eq!(
                        stats.postings_skipped,
                        stats.probed_clusters.len() - 1,
                        "{stats:?}"
                    );
                }
                _ => {
                    assert!(stats.postings_bulk + stats.postings_row > 0, "{stats:?}");
                }
            }
        }
        Ok(())
    }

    /// Replica dedup is decided in the pre-pass: a doc whose copies land
    /// in several probed clusters is fetched and scored exactly once, its
    /// later cells (holding no other survivor) skip their fetches
    /// entirely, and `pruned_seen` is identical to the forced-bulk
    /// baseline.
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
        let (hits, stats) = run_top_n_forced(
            &index,
            embed_field,
            vec![10.0, 10.0],
            n,
            params.clone(),
            &weight,
            None,
        )?;
        let (bulk_hits, bulk_stats) = run_top_n_forced(
            &index,
            embed_field,
            vec![10.0, 10.0],
            n,
            params,
            &weight,
            Some(ForcedFetch::Bulk),
        )?;

        assert_eq!(hits, bulk_hits);
        assert_eq!(hits.len(), 1, "the admitted doc must return exactly once");
        assert_eq!(stats.candidates_scored, 1, "scored on first encounter only");
        assert_eq!(
            stats.pruned_seen,
            replicas - 1,
            "each later cell prunes it as seen: {stats:?}"
        );
        assert_eq!(bulk_stats.pruned_seen, replicas - 1);
        assert_prune_counters_match(&stats, &bulk_stats);
        // Only the first-probed cell fetches anything.
        assert_eq!(stats.postings_bulk + stats.postings_row, 1, "{stats:?}");
        assert_eq!(stats.postings_skipped, stats.probed_clusters.len() - 1);
        assert_stats_identities(&stats);
        Ok(())
    }

    /// One query drives all three fetch modes at once. Geometry: a
    /// 3000-row cluster whose filter admits one doc (its survivor buffer
    /// touches fewer ~8 KiB blocks than the 24 KB bulk read → per-row), a
    /// 4-row cluster with every doc admitted (bulk), and a 4-row cluster
    /// with none (skip). Forced-row and forced-bulk runs return identical
    /// hits, so the sparse path is score-equivalent to bulk on the same
    /// clusters.
    #[test]
    fn fetch_modes_split_skip_row_bulk_on_one_query() -> crate::Result<()> {
        const BIG: usize = 3000;
        let centroids = vec![[0.0f32, 0.0], [50.0, 0.0], [0.0, 50.0]];
        let labels: Vec<String> = (0..BIG + 8).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, [f32; 2])> = (0..labels.len())
            .map(|i| {
                let (c, j) = match i {
                    _ if i < BIG => (centroids[0], i),
                    _ if i < BIG + 4 => (centroids[1], i - BIG),
                    _ => (centroids[2], i - BIG - 4),
                };
                (labels[i].as_str(), [c[0] + j as f32 * 0.001, c[1]])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;

        // Setup: cluster populations landed as designed, and the cutoff
        // math this geometry is built around holds.
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        assert_eq!(vec_reader.cluster_sizes(), Some(vec![BIG as u32, 4, 4]));
        let stride = vec_reader.options().bytes_per_vector();
        assert!(row_fetch_is_cheaper(1, BIG, stride), "stride {stride}");
        assert!(!row_fetch_is_cheaper(4, 4, stride), "stride {stride}");

        // Admit one doc of the big cluster and all of cluster 1 — taken
        // from the reader's own membership, so the filter can't drift
        // from the clustering.
        let big_member = vec_reader.cluster_doc_ids(0).expect("cluster 0")[BIG / 2];
        let mut admitted = vec![big_member];
        admitted.extend(vec_reader.cluster_doc_ids(1).expect("cluster 1"));
        let weight = FixedDocsWeight {
            max_doc: segment_reader.max_doc(),
            docs: admitted.clone(),
        };

        let k = admitted.len();
        let params = exhaustive_params(centroids.len());
        let query = vec![25.0f32, 0.0];
        let (hits, stats) = run_top_n_forced(
            &index,
            embed_field,
            query.clone(),
            k,
            params.clone(),
            &weight,
            None,
        )?;
        assert_eq!(
            stats.probed_clusters.len(),
            centroids.len(),
            "exhaustive probe covers all clusters"
        );
        assert_eq!(
            (
                stats.postings_row,
                stats.postings_bulk,
                stats.postings_skipped
            ),
            (1, 1, 1),
            "one cluster per fetch mode: {stats:?}"
        );
        assert_eq!(hits.len(), k, "every admitted doc surfaces");
        assert_stats_identities(&stats);

        // Forced row / forced bulk / natural dispatch all agree on hits.
        let (row_hits, row_stats) = run_top_n_forced(
            &index,
            embed_field,
            query.clone(),
            k,
            params.clone(),
            &weight,
            Some(ForcedFetch::Row),
        )?;
        let (bulk_hits, bulk_stats) = run_top_n_forced(
            &index,
            embed_field,
            query,
            k,
            params,
            &weight,
            Some(ForcedFetch::Bulk),
        )?;
        assert_eq!(hits, row_hits, "forced-row must be hit-identical");
        assert_eq!(hits, bulk_hits, "forced-bulk must be hit-identical");
        // Forcing rewires only the fetch: row-forcing still skips the
        // survivor-less cluster, while bulk-forcing (the pre-change
        // baseline) fetches every probed cluster.
        assert_eq!(
            (
                row_stats.postings_row,
                row_stats.postings_bulk,
                row_stats.postings_skipped
            ),
            (2, 0, 1),
            "{row_stats:?}"
        );
        assert_eq!(
            (
                bulk_stats.postings_row,
                bulk_stats.postings_bulk,
                bulk_stats.postings_skipped
            ),
            (0, 3, 0),
            "{bulk_stats:?}"
        );
        assert_prune_counters_match(&stats, &bulk_stats);
        Ok(())
    }

    /// Deletes are decided in the pre-pass: a cluster whose rows are all
    /// dead yields zero survivors and fetches nothing, while
    /// `pruned_dead` stays identical to the forced-bulk baseline.
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
        let (hits, stats) = run_top_n_forced(
            &index,
            embed_field,
            vec![10.0, 10.0],
            n,
            params.clone(),
            &weight,
            None,
        )?;
        let (bulk_hits, bulk_stats) = run_top_n_forced(
            &index,
            embed_field,
            vec![10.0, 10.0],
            n,
            params,
            &weight,
            Some(ForcedFetch::Bulk),
        )?;

        assert_eq!(hits, bulk_hits);
        assert_eq!(hits.len(), n - REPLICATION_N_PER, "only alive docs surface");
        assert_eq!(
            stats.pruned_dead, REPLICATION_N_PER,
            "every dead row prunes as dead: {stats:?}"
        );
        assert_eq!(bulk_stats.pruned_dead, REPLICATION_N_PER);
        assert_prune_counters_match(&stats, &bulk_stats);
        // The fully-dead cluster fetches nothing; the other five fetch.
        assert_eq!(stats.postings_skipped, 1, "{stats:?}");
        assert_eq!(
            stats.postings_bulk + stats.postings_row,
            centroids.len() - 1
        );
        assert_stats_identities(&stats);
        Ok(())
    }

    /// An empty cluster still counts as probed — the probe is recorded
    /// before any work — and takes the skip path: `postings_skipped`
    /// increments, nothing is fetched, and the visited/prune counters
    /// don't move.
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
        let (hits, stats) = run_top_n_forced(
            &index,
            embed_field,
            vec![5.0, 0.0],
            8,
            exhaustive_params(centroids.len()),
            &weight,
            None,
        )?;
        assert_eq!(hits.len(), 8);
        assert_eq!(
            stats.probed_clusters.len(),
            centroids.len(),
            "the empty cluster still counts as probed: {stats:?}"
        );
        assert!(
            stats.probed_clusters.contains(&2),
            "the empty cluster is in the probe list: {stats:?}"
        );
        assert_eq!(
            stats.postings_skipped, 1,
            "the empty cluster fetched nothing: {stats:?}"
        );
        assert_eq!(stats.postings_bulk + stats.postings_row, 2);
        assert_eq!(
            stats.vectors_visited, 8,
            "an empty cluster contributes no visited rows"
        );
        assert_stats_identities(&stats);
        Ok(())
    }

    /// Flat-path result equivalence across filter selectivities: natural
    /// run chunking returns doc-id- and score-identical hits to the
    /// forced-per-doc (pre-change) baseline for hand-built filters
    /// admitting {0, 1, 50, 100}% of docs, the probe-loop counters stay
    /// zeroed on both runs, and at dense selectivities the merged reads
    /// touch strictly fewer cost-model blocks than per-doc reads.
    #[test]
    fn flat_chunked_reads_match_per_doc_across_selectivities() -> crate::Result<()> {
        let n = 40usize;
        let labels: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, Option<Vec<f32>>)> = (0..n)
            .map(|i| (labels[i].as_str(), Some(vec![i as f32 * 0.1, 1.0])))
            .collect();
        let (index, embed_field, _label) = build_flat(2, &docs)?;
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
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

            let before = vec_reader.read_block_touches();
            let (hits, stats) = run_exact_on_segment(
                segment_reader,
                embed_field,
                vec![0.0, 0.0],
                n,
                &weight,
                None,
            )?;
            let chunked_touches = vec_reader.read_block_touches() - before;

            let before = vec_reader.read_block_touches();
            let (pd_hits, pd_stats) = run_exact_on_segment(
                segment_reader,
                embed_field,
                vec![0.0, 0.0],
                n,
                &weight,
                Some(ForcedExactRead::PerDoc),
            )?;
            let perdoc_touches = vec_reader.read_block_touches() - before;

            assert_eq!(hits, pd_hits, "{pct}%: hits must match per-doc reads");
            assert_eq!(hits.len(), admitted, "{pct}%");
            // The exact path fills only the read counters; the probe-loop
            // fields must stay zeroed on both runs.
            assert_eq!(stats.vectors_visited, 0, "{pct}%: {stats:?}");
            assert_eq!(stats.candidates_scored, 0, "{pct}%: {stats:?}");
            assert!(stats.probed_clusters.is_empty(), "{pct}%: {stats:?}");
            assert_eq!(
                (pd_stats.exact_reads_chunked, pd_stats.exact_reads_single),
                (0, admitted),
                "{pct}%: the baseline reads once per survivor"
            );
            match pct {
                0 => {
                    assert_eq!(
                        (stats.exact_reads_chunked, stats.exact_reads_single),
                        (0, 0)
                    );
                    assert_eq!(chunked_touches, 0, "no survivors, no reads");
                }
                1 => {
                    assert_eq!(
                        (stats.exact_reads_chunked, stats.exact_reads_single),
                        (0, 1)
                    );
                    assert_eq!(
                        chunked_touches, perdoc_touches,
                        "a lone survivor reads exactly the per-doc bytes"
                    );
                }
                _ => {
                    // Consecutive survivors coalesce into one merged read
                    // touching strictly fewer blocks.
                    assert_eq!(
                        (stats.exact_reads_chunked, stats.exact_reads_single),
                        (1, 0),
                        "{pct}%: {stats:?}"
                    );
                    assert!(
                        chunked_touches < perdoc_touches,
                        "{pct}%: chunked {chunked_touches} must beat per-doc {perdoc_touches}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Flat runs over a `Bitmap` id-map with tombstoned docs: vectorless
    /// and deleted docs leave holes between surviving rows, which 8-byte
    /// rows absorb into a single merged read whose hole bytes are fetched
    /// but never scored. Results are identical to per-doc reads and to
    /// force-merged reads, and match the brute-force oracle.
    #[test]
    fn flat_runs_handle_bitmap_holes_and_deletes() -> crate::Result<()> {
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
            run_exact_on_segment(segment_reader, embed_field, query.clone(), n, &weight, None)?;
        let (pd_hits, pd_stats) = run_exact_on_segment(
            segment_reader,
            embed_field,
            query.clone(),
            n,
            &weight,
            Some(ForcedExactRead::PerDoc),
        )?;
        let (fc_hits, fc_stats) = run_exact_on_segment(
            segment_reader,
            embed_field,
            query.clone(),
            n,
            &weight,
            Some(ForcedExactRead::Chunked),
        )?;

        assert_eq!(hits, pd_hits, "natural chunking must match per-doc");
        assert_eq!(hits, fc_hits, "forced chunking must match per-doc");
        // Oracle: every alive doc with a vector, exactly once.
        let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, n)?;
        assert_eq!(hits, expected);

        // 8-byte rows absorb every hole here (gap tolerance ≫ segment):
        // one merged read; the baseline reads once per survivor.
        let survivors = hits.len();
        assert!(survivors > 0, "fixture must leave survivors");
        assert_eq!(
            (stats.exact_reads_chunked, stats.exact_reads_single),
            (1, 0),
            "{stats:?}"
        );
        assert_eq!(
            (pd_stats.exact_reads_chunked, pd_stats.exact_reads_single),
            (0, survivors)
        );
        assert_eq!(
            (fc_stats.exact_reads_chunked, fc_stats.exact_reads_single),
            (1, 0)
        );
        Ok(())
    }

    /// Fat rows (3072-dim f32, stride 12288 = 3 blocks/row) shift the IVF
    /// sparse/dense crossover: a 100-row cluster bulk-reads at 151
    /// blocks, so 50 survivors still fetch per-row (150 < 151) while 51
    /// flip to bulk — the flat 2-block pricing would have kept per-row
    /// through 75.
    #[test]
    fn ivf_dispatch_prices_fat_rows() -> crate::Result<()> {
        let dim = 3072usize;
        let n_per = 100usize;
        let (index, embed_field) = build_wide_ivf(dim, 2, n_per)?;
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        let stride = vec_reader.options().bytes_per_vector();
        assert_eq!(stride, 12288, "3072-dim f32 stride");
        assert_eq!(per_row_blocks(stride), 3);
        let cluster0 = vec_reader.cluster_doc_ids(0).expect("ivf cluster 0");
        assert_eq!(cluster0.len(), n_per, "cluster 0 holds its 100 docs");

        let max_doc = segment_reader.max_doc();
        let params = exhaustive_params(2);
        let mut query = vec![0.0f32; dim];
        query[0] = 0.05;
        for (admit, expect_row, expect_bulk) in [(50usize, 1, 0), (51, 0, 1)] {
            let weight = FixedDocsWeight {
                max_doc,
                docs: cluster0[..admit].to_vec(),
            };
            let (hits, stats) = run_top_n_forced(
                &index,
                embed_field,
                query.clone(),
                admit,
                params.clone(),
                &weight,
                None,
            )?;
            let (bulk_hits, bulk_stats) = run_top_n_forced(
                &index,
                embed_field,
                query.clone(),
                admit,
                params.clone(),
                &weight,
                Some(ForcedFetch::Bulk),
            )?;
            assert_eq!(hits, bulk_hits, "s = {admit}");
            assert_eq!(hits.len(), admit);
            assert_eq!(
                (stats.postings_row, stats.postings_bulk),
                (expect_row, expect_bulk),
                "s = {admit}: {stats:?}"
            );
            assert_eq!(
                stats.postings_skipped, 1,
                "cluster 1 admits nothing: {stats:?}"
            );
            assert_prune_counters_match(&stats, &bulk_stats);
            assert_stats_identities(&stats);
        }
        Ok(())
    }

    /// Fat rows in the flat run builder: with 3-block rows a 1-row hole
    /// absorbs into the run (a 6-block merged read vs 3 + 3 separate)
    /// while a 2-row hole splits (7 > 6) — under a flat 2-block row
    /// pricing even the 1-row hole would have split. Forced modes stay
    /// hit-identical throughout.
    #[test]
    fn flat_run_builder_prices_fat_rows() -> crate::Result<()> {
        let dim = 3072usize;
        let n = 6usize;
        let labels: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, Option<Vec<f32>>)> = (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[0] = i as f32;
                (labels[i].as_str(), Some(v))
            })
            .collect();
        let (index, embed_field, _label) = build_flat(dim, &docs)?;
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        let stride = vec_reader.options().bytes_per_vector();
        assert_eq!(stride, 12288, "3072-dim f32 stride");
        // Identity id-map (every doc has a vector): row == doc id, so the
        // filters below place survivors at exactly these rows.
        assert_eq!(vec_reader.num_vectors(), n);
        let max_doc = segment_reader.max_doc();
        let query = vec![0.0f32; dim];

        // Survivors at rows {0, 2}: the 1-row hole absorbs → one chunked
        // read (row 1 fetched but never scored).
        let w_gap1 = FixedDocsWeight {
            max_doc,
            docs: vec![0, 2],
        };
        let (h1, s1) =
            run_exact_on_segment(segment_reader, embed_field, query.clone(), n, &w_gap1, None)?;
        let (h1_pd, s1_pd) = run_exact_on_segment(
            segment_reader,
            embed_field,
            query.clone(),
            n,
            &w_gap1,
            Some(ForcedExactRead::PerDoc),
        )?;
        assert_eq!(h1, h1_pd, "gap-1 chunking must match per-doc");
        assert_eq!(h1.len(), 2);
        assert_eq!(
            (s1.exact_reads_chunked, s1.exact_reads_single),
            (1, 0),
            "{s1:?}"
        );
        assert_eq!(
            (s1_pd.exact_reads_chunked, s1_pd.exact_reads_single),
            (0, 2)
        );

        // Survivors at rows {0, 3}: the 2-row hole splits → two single
        // reads, per-doc-equivalent.
        let w_gap2 = FixedDocsWeight {
            max_doc,
            docs: vec![0, 3],
        };
        let (h2, s2) =
            run_exact_on_segment(segment_reader, embed_field, query.clone(), n, &w_gap2, None)?;
        let (h2_pd, _) = run_exact_on_segment(
            segment_reader,
            embed_field,
            query.clone(),
            n,
            &w_gap2,
            Some(ForcedExactRead::PerDoc),
        )?;
        assert_eq!(h2, h2_pd, "gap-2 split must match per-doc");
        assert_eq!(
            (s2.exact_reads_chunked, s2.exact_reads_single),
            (0, 2),
            "{s2:?}"
        );

        // Force-merging across the 2-row hole reads more but must stay
        // hit-identical.
        let (h2_fc, s2_fc) = run_exact_on_segment(
            segment_reader,
            embed_field,
            query,
            n,
            &w_gap2,
            Some(ForcedExactRead::Chunked),
        )?;
        assert_eq!(h2, h2_fc, "forced chunking must match per-doc");
        assert_eq!(
            (s2_fc.exact_reads_chunked, s2_fc.exact_reads_single),
            (1, 0)
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

    /// L2-nearest centroid index for a query against the shared
    /// fixture's default 3×3 grid centroids.
    fn nearest_centroid_to(query: &[f32; 2]) -> usize {
        // Match the grid in `crate::vector::tests::grid2d::centroids()`:
        // origin=(0,0), 3×3, gap=3.0, row-major.
        let centroids: Vec<[f32; 2]> = (0..3)
            .flat_map(|row| (0..3).map(move |col| [col as f32 * 3.0, row as f32 * 3.0]))
            .collect();
        nearest_centroid(*query, &centroids)
    }
}
