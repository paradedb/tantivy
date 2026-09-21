//! The probe work-unit model and search instrumentation.
//!
//! The cross-segment probe loop itself lives in [`search`](super::search);
//! this module owns what the loop charges (the work-unit model, calibrated
//! on the reference fixture) and what it reports ([`ProbeStats`]).

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use crate::vector::RouterMetrics;

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[repr(C)]
pub struct RoutingPhases {
    pub segment_metadata_time_ns: u64,
    pub router_open_time_ns: u64,
    pub centroid_precompute_time_ns: u64,
    /// Ranking and prefix authorization, including incremental publication.
    pub router_prefix_time_ns: u64,
}

impl std::ops::AddAssign for RoutingPhases {
    fn add_assign(&mut self, rhs: Self) {
        self.segment_metadata_time_ns += rhs.segment_metadata_time_ns;
        self.router_open_time_ns += rhs.router_open_time_ns;
        self.centroid_precompute_time_ns += rhs.centroid_precompute_time_ns;
        self.router_prefix_time_ns += rhs.router_prefix_time_ns;
    }
}

/// Probe-loop instrumentation for one query, filled by the cross-segment
/// loop in [`search`](super::search): a prune breakdown of every doc the
/// inner loop touched, plus posting-fetch counters. One instance per query
/// — the loop is global, so the counters are too.
#[derive(Debug, Default, serde::Serialize)]
pub struct ProbeStats {
    pub routing_time_ns: u64,
    #[serde(flatten)]
    pub routing_phases: RoutingPhases,
    pub precomputed_centroids: usize,
    pub filter_time_ns: u64,
    pub probe_time_ns: u64,
    pub segment_setup_time_ns: u64,
    pub pruned_invisible: usize,
    #[serde(skip)]
    pub cluster_flags: Vec<u8>,
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
    /// Rows fetched and scored by exhaustive physical scans. Flat segments
    /// are mandatory work outside the probe budget. Exact filtered scans
    /// of clustered segments also count as candidates and charge row work.
    pub exact_rows_read: usize,
    /// Statistics from the configured router's ranking implementation.
    pub routing: Option<RouterMetrics>,
    /// Vector-bearing segments this query considered — clustered and flat.
    pub segments_searched: u32,
    /// Segments whose filter bitset was actually materialized — lazy
    /// filters mean a segment whose every touched (cluster, segment)
    /// pair was absent or bounds-skipped never evaluates its filter.
    pub filters_built: u32,
}

impl ProbeStats {
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
/// * `n_avg` (`f64`) — native docs per cluster (see `WorkModel`).
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

    pub(super) fn get(self) -> f64 {
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

#[cfg(test)]
mod tests {
    // ============================================================
    // Cross-segment search gate + write-path assertions.
    //
    // Search tests drive the full global loop (one routing pass, one
    // heap) through the collector or the `global_top_n_by` seam and
    // compare against `ground_truth::top_k`. Write-path tests assert
    // the stored state through the reader's introspection surface.
    // ============================================================
    use std::sync::{Arc, Barrier, Condvar, Mutex};

    use super::*;
    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
    use crate::vector::ivf::AdaptiveProbeParams;
    use crate::vector::tests::{exhaustive_params, ground_truth, TestVectorIndex};
    use crate::vector::{
        CentroidProducer, ClusterWork, IvfCentroids, IvfMatrix, Metric, NoTieBreak,
        PreparedVectorSearch, ProbeBudget, ProbeWave, RankedCluster, RouterKind, VectorDType,
        VectorInfo, VectorOptions, VectorSearchControl, PROBE_WAVE_SIZE,
    };
    use crate::{DocAddress, Index, IndexWriter, Score, TantivyDocument};

    const FIXTURE_NUM_DOCS: usize = 100;
    /// Number of centroids the shared fixture uses by default (the
    /// 3×3 `grid2d::centroids()` grid).
    const DEFAULT_NUM_CENTROIDS: usize = 9;
    /// Segments the shared fixture produces: ten 10-doc commits, merged
    /// pairwise into five 20-doc segments.
    const FIXTURE_NUM_SEGMENTS: usize = 5;

    /// Run the full collector path with the given filter and adaptive
    /// params. Returns the global top-K in descending-score /
    /// (seg_ord, doc_id) order — the same order `ground_truth::top_k`
    /// uses, so equality checks are well-defined.
    fn search(
        index: &Index,
        field: crate::schema::Field,
        filter: &dyn Query,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let collector = TopDocs::with_limit(k)
            .order_by_similarity(field, query)
            .with_adaptive_params(params);
        let searcher = index.reader()?.searcher();
        Ok(collector.search(&searcher, filter)?.results)
    }

    /// Probe-stat seam: run the global driver directly and return
    /// (hits, stats).
    fn run_global(
        index: &Index,
        field: crate::schema::Field,
        filter: &dyn Query,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let searcher = index.reader()?.searcher();
        let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let (hits, stats) = crate::vector::search::global_top_n_by(
            &searcher,
            weight.as_ref(),
            field,
            &Arc::new(query),
            k,
            &params,
            &NoTieBreak,
        )?;
        Ok((
            hits.into_iter()
                .map(|((score, ()), addr)| (score, addr))
                .collect(),
            stats,
        ))
    }

    /// Every doc address matching `filter`, across all segments.
    fn collect_filter_doc_set(
        index: &Index,
        filter: &dyn Query,
    ) -> crate::Result<std::collections::HashSet<DocAddress>> {
        let searcher = index.reader()?.searcher();
        let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let mut set = std::collections::HashSet::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            weight.for_each_no_score(segment_reader, &mut |docs| {
                for &doc in docs {
                    set.insert(DocAddress::new(seg_ord as u32, doc));
                }
            })?;
        }
        Ok(set)
    }

    /// The stored label of `addr` — the segment-independent doc identity.
    fn stored_label_at(
        index: &Index,
        label_field: crate::schema::Field,
        addr: DocAddress,
    ) -> crate::Result<String> {
        use crate::schema::document::Value;
        use crate::schema::TantivyDocument;
        let searcher = index.reader()?.searcher();
        let doc: TantivyDocument = searcher.doc(addr)?;
        Ok(doc
            .get_first(label_field)
            .and_then(|v| v.as_str())
            .expect("stored label")
            .to_string())
    }

    // ---- Inline fixtures ----

    #[test]
    fn global_rng_routing_uses_one_budget() -> crate::Result<()> {
        let fixture = TestVectorIndex::builder(VectorDType::F32)
            .router(RouterKind::Rng)
            .build()?;
        let field = fixture.embedding_field();
        let query = vec![1.0, 1.0];
        let params = AdaptiveProbeParams {
            max_probe_fraction: 1e-6,
            min_probe_clusters: 1,
        };
        let (_, stats) = run_global(&fixture.index, field, &AllQuery, query.clone(), 5, params)?;
        assert_eq!(stats.segments_searched, FIXTURE_NUM_SEGMENTS as u32);
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert!(matches!(stats.routing, Some(RouterMetrics::Rng(_))));
        let (hits, stats) = run_global(
            &fixture.index,
            field,
            &AllQuery,
            query.clone(),
            5,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(
            hits,
            ground_truth::top_k(&fixture.index, field, Metric::L2, &query, 5)?
        );
        assert!(matches!(stats.routing, Some(RouterMetrics::Rng(_))));
        Ok(())
    }

    /// Fixed-centroid [`CentroidProducer`]: the consumer "trained" these
    /// centroids elsewhere; tantivy only assigns against them.
    pub(crate) struct InlineCentroidProducer {
        pub(crate) centroids: Vec<[f32; 2]>,
    }

    impl CentroidProducer for InlineCentroidProducer {
        fn centroids(
            &self,
            _field: crate::schema::Field,
            options: &VectorOptions,
        ) -> crate::Result<IvfCentroids> {
            assert_eq!(options.dim(), 2);
            Ok(IvfCentroids::F32(IvfMatrix {
                values: self.centroids.iter().flatten().copied().collect(),
                rows: self.centroids.len(),
                dims: 2,
            }))
        }
    }

    /// An index over `commits` (one segment per inner slice), assigned
    /// against the given centroids; merged into one segment iff `merge`.
    fn build_ivf(
        metric: Metric,
        centroids: &[[f32; 2]],
        commits: &[&[(&str, [f32; 2])]],
        replicas: usize,
        merge: bool,
    ) -> crate::Result<(Index, crate::schema::Field, crate::schema::Field)> {
        build_ivf_with_router(metric, centroids, commits, replicas, merge, RouterKind::Rng)
    }

    fn build_ivf_with_router(
        metric: Metric,
        centroids: &[[f32; 2]],
        commits: &[&[(&str, [f32; 2])]],
        replicas: usize,
        merge: bool,
        router: RouterKind,
    ) -> crate::Result<(Index, crate::schema::Field, crate::schema::Field)> {
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let settings = IndexSettings {
            vector_replicas: replicas,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(sb.build())
            .settings(settings)
            .centroid_producer(Arc::new(InlineCentroidProducer {
                centroids: centroids.to_vec(),
            }))
            .ivf_router(router)?
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for chunk in commits {
            for (label, v) in *chunk {
                let mut doc = TantivyDocument::new();
                doc.add_text(label_field, label);
                doc.add_vector(embed_field, v.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }
        if merge {
            let segment_ids = index.searchable_segment_ids()?;
            writer.merge(&segment_ids).wait()?;
        }
        writer.wait_merging_threads()?;
        Ok((index, embed_field, label_field))
    }

    /// [`build_ivf`] with docs split across two commits and merged — the
    /// single-segment shape most write-path tests want.
    fn build_inline_ivf(
        metric: Metric,
        centroids: &[[f32; 2]],
        docs: &[(&str, [f32; 2])],
        replicas: usize,
    ) -> crate::Result<(Index, crate::schema::Field, crate::schema::Field)> {
        assert!(docs.len() >= 2, "need ≥ 2 docs for ≥ 2 source segments");
        let mid = (docs.len() / 2).max(1);
        build_ivf(
            metric,
            centroids,
            &[&docs[..mid], &docs[mid..]],
            replicas,
            true,
        )
    }

    /// Decode a stored little-endian `[f32; 2]` row.
    fn decode_2d(bytes: &[u8]) -> [f32; 2] {
        [
            f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        ]
    }

    /// L2-nearest centroid with ascending-id tie-break — the assignment
    /// selector's primary rule for L2.
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

    /// A doc's cluster memberships plus its recomputed primary, read back
    /// through the cluster iteration.
    struct ReadBack {
        memberships: Vec<Vec<usize>>,
        primaries: Vec<usize>,
    }

    fn read_back(
        index: &Index,
        embed_field: crate::schema::Field,
        centroids: &[[f32; 2]],
        expected_docs: usize,
    ) -> crate::Result<ReadBack> {
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1, "one segment expected");
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        let ivf = vec_reader.clusters().expect("expected IVF storage");
        assert_eq!(ivf.num_clusters(), centroids.len());
        let max_doc = segment_reader.max_doc() as usize;
        assert_eq!(max_doc, expected_docs, "unexpected doc count");
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
                nearest_centroid(decode_2d(&bytes), centroids)
            })
            .collect();
        Ok(ReadBack {
            memberships,
            primaries,
        })
    }

    // ==========================================================
    // Search: brute-force oracle equality
    // ==========================================================

    /// Exhaustive probing on the multi-segment fixture must match the
    /// brute-force oracle — per metric. Dot is EXHAUSTIVE-PROBE ONLY by
    /// design: it isn't a metric (no triangle inequality), so adaptive
    /// Dot recall is a benchmark question, deferred.
    #[test]
    fn global_search_matches_brute_force_oracle_per_metric() -> crate::Result<()> {
        for (metric, queries) in [
            (
                Metric::L2,
                vec![[0.5_f32, 0.5], [9.5, 9.5], [5.0, 0.0], [3.7, 11.2]],
            ),
            (Metric::Cosine, vec![[1.0_f32, 0.0], [0.0, 1.0], [0.7, 0.3]]),
            (Metric::Dot, vec![[1.0_f32, 0.0], [2.0, 0.0], [0.5, -0.5]]),
        ] {
            let index = TestVectorIndex::builder(VectorDType::F32)
                .metric(metric)
                .build()?;
            let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
            for query in queries {
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
                    assert_eq!(
                        actual, expected,
                        "{metric:?} exhaustive query={query:?} k={k}"
                    );
                }
            }
        }
        Ok(())
    }

    /// The trap: query closest to centroid A, true NN in cluster B.
    /// Exhaustive probing finds it; a 1-cluster probe ceiling must miss.
    /// Setup assertions confirm the geometry is genuinely a trap.
    #[test]
    fn global_search_trap_case() -> crate::Result<()> {
        let centroids = vec![[0.0_f32, 0.0], [10.0, 10.0]];
        // Two A-side docs far from the [1,1] query; a B-side trap doc at
        // [5, 5.01] just over the perpendicular bisector (x+y=10) so it
        // lands in cluster 1 yet is much closer to the query than any
        // A-side doc.
        let docs = [
            ("far_a0", [0.0_f32, -10.0]),
            ("far_a1", [-10.0, 0.0]),
            ("trap_b", [5.0, 5.01]),
            ("anchor_b", [10.0, 10.0]),
        ];
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        let query = [1.0_f32, 1.0];

        // (i) The trap doc is genuinely the true top-1.
        let oracle = ground_truth::top_k(&index, embed_field, Metric::L2, &query, 1)?;
        assert_eq!(
            stored_label_at(&index, label_field, oracle[0].1)?,
            "trap_b",
            "true NN must be the trap doc"
        );

        // A tight ceiling misses the trap (probes only cluster A)...
        let one_probe = AdaptiveProbeParams {
            max_probe_fraction: 0.5,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let hits1 = search(&index, embed_field, &AllQuery, query.to_vec(), 1, one_probe)?;
        assert_eq!(hits1.len(), 1);
        assert_ne!(stored_label_at(&index, label_field, hits1[0].1)?, "trap_b");

        // ...and exhaustive probing finds it.
        let hits2 = search(
            &index,
            embed_field,
            &AllQuery,
            query.to_vec(),
            1,
            exhaustive_params(2),
        )?;
        assert_eq!(stored_label_at(&index, label_field, hits2[0].1)?, "trap_b");
        Ok(())
    }

    /// Filter selectivity: only docs in the filter set surface, and the
    /// result equals the oracle restricted to that set.
    #[test]
    fn global_search_filter_selectivity() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .selectivities(&[0.1])
            .build()?;
        let filter = TermQuery::new(
            Term::from_field_text(index.label_field(), "selectivity_0.1"),
            IndexRecordOption::Basic,
        );
        let query = [0.5_f32, 0.5];
        let k = 5;
        let filter_set = collect_filter_doc_set(&index.index, &filter)?;
        let mut restricted = ground_truth::top_k(
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

    /// Empty filter returns empty results, no panic — and kills every
    /// segment after one materialization each.
    #[test]
    fn global_search_empty_filter() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .build()?;
        let empty = TermQuery::new(
            Term::from_field_text(index.label_field(), "absent"),
            IndexRecordOption::Basic,
        );
        let (hits, stats) = run_global(
            &index.index,
            index.embedding_field(),
            &empty,
            vec![0.0_f32, 0.0],
            5,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert!(hits.is_empty());
        assert_eq!(stats.candidates_scored, 0);
        assert_eq!(
            stats.filters_built as usize, FIXTURE_NUM_SEGMENTS,
            "every segment materializes its (empty) filter exactly once"
        );
        Ok(())
    }

    /// K > total candidates returns all docs in oracle order; k == 0
    /// returns empty without touching anything.
    #[test]
    fn global_search_k_edges() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
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

        let (hits, stats) = run_global(
            &index.index,
            index.embedding_field(),
            &AllQuery,
            query.to_vec(),
            0,
            AdaptiveProbeParams::default(),
        )?;
        assert!(hits.is_empty());
        assert_eq!(stats.clusters_probed(), 0);
        assert_eq!(stats.candidates_scored, 0);
        Ok(())
    }

    /// Deletes: a doc marked deleted must never appear, even if it would
    /// otherwise rank top-K — the alive check is separate from the filter.
    #[test]
    fn global_search_respects_deletes() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .selectivities(&[0.1])
            .build()?;
        {
            let mut writer: IndexWriter = index.index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            writer.delete_term(Term::from_field_text(
                index.label_field(),
                "selectivity_0.1",
            ));
            writer.commit()?;
        }
        let query = [0.0_f32, 0.0];
        let searcher = index.index.reader()?.searcher();
        let mut alive_addrs = std::collections::HashSet::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            let alive = segment_reader.alive_bitset();
            for doc in 0..segment_reader.max_doc() {
                if alive.is_none_or(|bs| bs.is_alive(doc)) {
                    alive_addrs.insert(DocAddress::new(seg_ord as u32, doc));
                }
            }
        }
        assert!(
            alive_addrs.len() < FIXTURE_NUM_DOCS,
            "delete removed nothing"
        );
        let k = 10;
        let mut expected = ground_truth::top_k(
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
            assert!(alive_addrs.contains(addr), "deleted doc {addr:?} surfaced");
        }
        Ok(())
    }

    /// Segment-count invariance: the same corpus in ONE merged segment
    /// and in FOUR unmerged commit segments returns identical exhaustive
    /// results by (score, label) — the global loop makes the physical
    /// layout invisible.
    #[test]
    fn global_search_is_segment_count_invariant() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let chunk = n / 4;
        let commits: Vec<&[(&str, [f32; 2])]> = docs.chunks(chunk).collect();

        let (merged, merged_field, merged_label) =
            build_ivf(Metric::L2, &centroids, &commits, 1, true)?;
        let (sharded, sharded_field, sharded_label) =
            build_ivf(Metric::L2, &centroids, &commits, 1, false)?;
        assert_eq!(
            sharded.reader()?.searcher().segment_readers().len(),
            commits.len(),
            "unmerged build must keep one segment per commit"
        );

        for query in [[0.0_f32, 0.0], [10.0, 10.0], [15.0, 5.0]] {
            for k in [1usize, 5, n] {
                let labeled = |index: &Index,
                               field,
                               label_field|
                 -> crate::Result<Vec<(Score, String)>> {
                    let mut hits: Vec<(Score, String)> = search(
                        index,
                        field,
                        &AllQuery,
                        query.to_vec(),
                        k,
                        exhaustive_params(centroids.len()),
                    )?
                    .into_iter()
                    .map(|(score, addr)| Ok((score, stored_label_at(index, label_field, addr)?)))
                    .collect::<crate::Result<_>>()?;
                    // Exact score ties break by DocAddress, which is
                    // layout-dependent by design (merges permute doc ids);
                    // normalize tie order by label so only the layout-
                    // INDEPENDENT ranking is compared.
                    hits.sort_by(|a, b| {
                        b.0.partial_cmp(&a.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.1.cmp(&b.1))
                    });
                    Ok(hits)
                };
                let merged_hits = labeled(&merged, merged_field, merged_label)?;
                let sharded_hits = labeled(&sharded, sharded_field, sharded_label)?;
                assert_eq!(
                    merged_hits, sharded_hits,
                    "layout leaked into results: query={query:?} k={k}"
                );
            }
        }
        Ok(())
    }

    // ==========================================================
    // Search: probe stats, budget, bounds gate, lazy filters
    // ==========================================================

    /// Exhaustive probe over the multi-segment fixture: the counter
    /// partition identity holds globally, every doc is scored exactly
    /// once, one routing pass ranks all centroids, and every segment
    /// materializes its filter exactly once.
    #[test]
    fn probe_stats_exhaustive_counters() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .build()?;
        let (_, stats) = run_global(
            &index.index,
            index.embedding_field(),
            &AllQuery,
            vec![0.0_f32, 0.0],
            FIXTURE_NUM_DOCS + 28,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(stats.candidates_scored, FIXTURE_NUM_DOCS);
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
            "visited must equal filter+dead+seen+scored ({stats:?})"
        );
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        assert_eq!(stats.segments_searched as usize, FIXTURE_NUM_SEGMENTS);
        // AllQuery matches every doc: the fast path builds NO filter
        // bitsets and prunes nothing on filters.
        assert_eq!(stats.filters_built, 0);
        assert_eq!(stats.pruned_filter, 0);
        // k > total docs: the bound never arms, nothing is skipped.
        assert_eq!(stats.bounds_skips, 0);
        assert_eq!(stats.bound_armed_at_probe, None);
        Ok(())
    }

    /// The normalization identity, cross-segment form: an exhaustive,
    /// unfiltered, delete-free scan charges exactly the index's capacity —
    /// an open share per non-empty (cluster, segment) pair plus a row
    /// share per doc.
    #[test]
    fn probe_stats_exhaustive_scan_charges_capacity() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let chunk = n / 3;
        let commits: Vec<&[(&str, [f32; 2])]> = docs.chunks(chunk).collect();
        let (index, field, _) = build_ivf(Metric::L2, &centroids, &commits, 1, false)?;

        let searcher = index.reader()?.searcher();
        let mut total_nonempty = 0usize;
        let mut total_docs = 0usize;
        for segment_reader in searcher.segment_readers() {
            let ivf_reader = segment_reader.vector_index(field)?;
            let ivf = ivf_reader.clusters().expect("IVF segment");
            total_nonempty += ivf.num_non_empty_clusters();
            total_docs += ivf.num_docs();
        }
        let n_avg = total_docs as f64 / centroids.len() as f64;
        let x = crate::vector::search::backend::open_share(n_avg);
        let capacity = total_nonempty as f64 * x + (1.0 - x) * total_docs as f64 / n_avg;

        let (_, stats) = run_global(
            &index,
            field,
            &AllQuery,
            vec![50.0, 50.0],
            n + 1, // never arms: no bounds skips distort the identity
            exhaustive_params(centroids.len()),
        )?;
        assert!(
            (f64::from(stats.work_charged) - capacity).abs() < 1e-4 * capacity,
            "exhaustive scan must charge exactly the capacity: charged={} capacity={capacity}",
            stats.work_charged
        );
        Ok(())
    }

    /// A tiny budget forces the hard ceiling: the loop stops with
    /// `termination == Ceiling` short of the ranked list, and the counter
    /// identity still holds.
    #[test]
    fn probe_stats_termination_ceiling() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let (_, stats) = run_global(&index, embed_field, &AllQuery, vec![10.0, 10.0], 3, params)?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert_eq!(stats.clusters_probed(), 1);
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
        );
        Ok(())
    }

    /// Replica dedup is counted, exactly: exhaustive probing over a
    /// replicated single segment visits `replicas × N` entries,
    /// re-encounters each doc exactly `replicas - 1` times, scores each
    /// exactly once.
    #[test]
    fn probe_stats_counts_replica_dedup() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let replicas = 4usize;
        let (index, embed_field, _label) =
            build_inline_ivf(Metric::L2, &centroids, &docs, replicas)?;

        let (_, stats) = run_global(
            &index,
            embed_field,
            &AllQuery,
            vec![10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?;
        assert_eq!(stats.vectors_visited, replicas * n);
        assert_eq!(stats.pruned_seen, (replicas - 1) * n);
        assert_eq!(stats.candidates_scored, n);
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.pruned_seen + stats.candidates_scored,
        );
        Ok(())
    }

    /// The shared-kth bound at work across segments, and the lazy filter
    /// riding on it: two tight, far-apart clusters live in two SEPARATE
    /// segments. Probing the query's cluster arms the global bound; the
    /// far segment's only cluster is then provably useless — skipped for
    /// the open share, WITHOUT ever materializing that segment's filter.
    /// The filter is a TermQuery every doc matches (an `AllQuery` would
    /// take the no-bitset fast path and build nothing anywhere).
    #[test]
    fn bounds_skip_spares_far_segment_and_its_filter() -> crate::Result<()> {
        let near: Vec<(String, [f32; 2])> = (0..8)
            .map(|i| (format!("near{i}"), [i as f32 * 0.001, 0.0]))
            .collect();
        let far: Vec<(String, [f32; 2])> = (0..8)
            .map(|i| (format!("far{i}"), [100.0 + i as f32 * 0.001, 100.0]))
            .collect();
        let near_ref: Vec<(&str, [f32; 2])> = near.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let far_ref: Vec<(&str, [f32; 2])> = far.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, label_field) = build_ivf(
            Metric::L2,
            &[[0.0, 0.0], [100.0, 100.0]],
            &[&near_ref, &far_ref],
            1,
            false,
        )?;
        // Filter on the near half only; the far segment's filter must
        // STILL never build (bounds-skipped before the filter gate).
        let near_filter = crate::query::BooleanQuery::union(
            (0..8)
                .map(|i| {
                    Box::new(TermQuery::new(
                        Term::from_field_text(label_field, &format!("near{i}")),
                        IndexRecordOption::Basic,
                    )) as Box<dyn Query>
                })
                .collect::<Vec<_>>(),
        );
        let (hits, stats) = run_global(
            &index,
            embed_field,
            &near_filter,
            vec![0.0, 0.0],
            1,
            exhaustive_params(2),
        )?;
        assert_eq!(hits.len(), 1);
        assert_eq!(stats.segments_searched, 2);
        // The near cluster arms the bound at the first touched cluster...
        assert_eq!(stats.bound_armed_at_probe, Some(0));
        // ...so the far segment's cluster is skipped without opening it,
        // and its filter is never evaluated.
        assert_eq!(stats.bounds_skips, 1);
        assert_eq!(stats.filters_built, 1, "far segment must stay filter-less");
        assert_eq!(stats.candidates_scored, near.len());
        Ok(())
    }

    /// The `AllQuery` fast path: an unfiltered search never materializes a
    /// filter bitset, and returns exactly what the (bitset-building)
    /// equivalent filter returns.
    #[test]
    fn all_query_never_builds_filters() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .selectivities(&[1.0])
            .build()?;
        let query = [0.5_f32, 0.5];
        // k >= every doc: the bound never arms, so no segment is
        // bounds-skipped and the term-filter run below must build ALL
        // bitsets — keeping the counts on both sides exact.
        let k = FIXTURE_NUM_DOCS;
        let (all_hits, all_stats) = run_global(
            &index.index,
            index.embedding_field(),
            &AllQuery,
            query.to_vec(),
            k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(all_stats.filters_built, 0, "AllQuery must build no bitsets");
        assert_eq!(all_stats.pruned_filter, 0);

        // "selectivity_1" labels every doc: same match set, but through a
        // real TermQuery, so every touched segment builds its bitset.
        let term_filter = TermQuery::new(
            Term::from_field_text(index.label_field(), "selectivity_1"),
            IndexRecordOption::Basic,
        );
        let (term_hits, term_stats) = run_global(
            &index.index,
            index.embedding_field(),
            &term_filter,
            query.to_vec(),
            k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(
            term_stats.filters_built as usize, FIXTURE_NUM_SEGMENTS,
            "the term filter takes the bitset path"
        );
        assert_eq!(all_hits, term_hits, "fast path must not change results");
        Ok(())
    }

    /// While the heap is still FILLING (k larger than everything seen),
    /// the bound never arms and nothing is ever skipped.
    #[test]
    fn unarmed_bound_never_skips() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        let (_, stats) = run_global(
            &index,
            embed_field,
            &AllQuery,
            vec![0.0, 0.0],
            n + 1,
            exhaustive_params(centroids.len()),
        )?;
        assert_eq!(stats.bounds_skips, 0);
        assert_eq!(stats.bound_armed_at_probe, None);
        assert_eq!(stats.candidates_scored, n);
        Ok(())
    }

    /// A configured router handles a single-centroid index and still returns
    /// the exact top-K.
    #[test]
    fn single_centroid_routes_with_configured_router() -> crate::Result<()> {
        let labels: Vec<String> = (0..5).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, [f32; 2])> = (0..5)
            .map(|i| (labels[i].as_str(), [i as f32 * 0.01, 0.0]))
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &[[0.0, 0.0]], &docs, 1)?;
        let expected = ground_truth::top_k(&index, embed_field, Metric::L2, &[0.0, 0.0], 3)?;
        let actual = search(
            &index,
            embed_field,
            &AllQuery,
            vec![0.0, 0.0],
            3,
            exhaustive_params(1),
        )?;
        assert_eq!(actual, expected);
        Ok(())
    }

    // ==========================================================
    // Write path
    // ==========================================================

    /// A single commit — no merge — already stores the clustered V3
    /// layout against the index-level centroid index: correct centroid
    /// count, every doc in its primary cluster.
    #[test]
    fn commit_segment_is_clustered_against_the_set() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();

        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let index = Index::builder()
            .schema(sb.build())
            .centroid_producer(Arc::new(InlineCentroidProducer {
                centroids: centroids.clone(),
            }))
            .ivf_router(RouterKind::Rng)?
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for (label, v) in &docs {
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, label);
            doc.add_vector(embed_field, v.as_slice());
            writer.add_document(doc)?;
        }
        writer.commit()?;

        let built = read_back(&index, embed_field, &centroids, n)?;
        for (doc, cells) in built.memberships.iter().enumerate() {
            assert_eq!(
                cells.as_slice(),
                &[built.primaries[doc]],
                "replicas=1: doc {doc} must live only in its primary cluster"
            );
        }

        let searcher = index.reader()?.searcher();
        let vec_reader = searcher.segment_readers()[0].vector_index(embed_field)?;
        let info = vec_reader.info().expect("vector info");
        assert_eq!(
            info,
            VectorInfo {
                num_vectors: n,
                num_centroids: centroids.len(),
                cluster_stats: crate::vector::VectorClusterStats {
                    min_cluster_size: REPLICATION_N_PER,
                    max_cluster_size: REPLICATION_N_PER,
                    avg_cluster_size: REPLICATION_N_PER as f64,
                    empty_clusters: 0,
                },
            },
        );
        Ok(())
    }

    /// Fixed-k replication is additive and, at small centroid counts, EXACT:
    /// the fixture's 6 centroids sit far below the exact-selection threshold
    /// (the search's `ef` budget), so cells come from a brute k-NN scan, not
    /// the approximate graph selector — every vector is written into exactly
    /// `min(replicas, num_centroids)` distinct cells: its primary (once) plus
    /// the `replicas - 1` next-nearest centroids. Total posting entries are
    /// exactly `replicas × N`. `replicas == 1` is the identity: every doc in
    /// exactly its primary cluster.
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

        // replicas = 3: exact fill. Per doc — ceiling and fill
        // (exactly min(replicas, num_centroids) = 3 cells), dedup (cells
        // distinct, primary present exactly once). Corpus-wide — total
        // memberships exactly replicas × N.
        let (index3, embed3, _) = build_inline_ivf(Metric::L2, &centroids, &docs, replicas)?;
        let built3 = read_back(&index3, embed3, &centroids, n)?;
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

        // replicas = 1: identity. Every doc lives in exactly one cluster —
        // its primary.
        let (index1, embed1, _) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        let built1 = read_back(&index1, embed1, &centroids, n)?;
        for (doc, cells) in built1.memberships.iter().enumerate() {
            assert_eq!(
                cells.as_slice(),
                &[built1.primaries[doc]],
                "replicas=1: doc {doc} must live only in its primary cluster"
            );
        }
        Ok(())
    }

    /// Merging carries the source postings over instead of re-assigning:
    /// every doc lands in exactly the cells it already occupied. Compared
    /// by LABEL, since the merge permutes doc ids.
    #[test]
    fn merge_preserves_source_memberships() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let chunk = n / 4;
        let commits: Vec<&[(&str, [f32; 2])]> = docs.chunks(chunk).collect();

        // Cells per label, across however many segments the index holds.
        let cells_by_label =
            |index: &Index,
             field: crate::schema::Field,
             label_field: crate::schema::Field|
             -> crate::Result<std::collections::BTreeMap<String, Vec<usize>>> {
                let searcher = index.reader()?.searcher();
                let mut out: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
                for (segment_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
                    let vec_reader = segment_reader.vector_index(field)?;
                    let ivf = vec_reader.clusters().expect("IVF segment");
                    for cluster in 0..ivf.num_clusters() {
                        for doc in vec_reader.cluster_doc_ids(cluster).expect("in-bounds") {
                            let label = stored_label_at(
                                index,
                                label_field,
                                DocAddress::new(segment_ord as u32, doc),
                            )?;
                            out.entry(label).or_default().push(cluster);
                        }
                    }
                }
                for cells in out.values_mut() {
                    cells.sort_unstable();
                }
                Ok(out)
            };

        let (sharded, sharded_field, sharded_label) =
            build_ivf(Metric::L2, &centroids, &commits, 3, false)?;
        let before = cells_by_label(&sharded, sharded_field, sharded_label)?;
        assert_eq!(before.len(), n, "every doc must have cells before merging");

        let (merged, merged_field, merged_label) =
            build_ivf(Metric::L2, &centroids, &commits, 3, true)?;
        let after = cells_by_label(&merged, merged_field, merged_label)?;
        assert_eq!(
            before, after,
            "merging must carry postings over, not re-assign"
        );

        // The merged bounds are the element-wise max of the sources' —
        // exactly a fresh fold when nothing was deleted.
        let searcher = sharded.reader()?.searcher();
        let mut source_max = vec![0.0f32; centroids.len()];
        for segment_reader in searcher.segment_readers() {
            let vec_reader = segment_reader.vector_index(sharded_field)?;
            let bounds = vec_reader.clusters().expect("IVF segment").bounds();
            for (cluster, slot) in source_max.iter_mut().enumerate() {
                *slot = slot.max(bounds.ball_r(cluster));
            }
        }
        let merged_searcher = merged.reader()?.searcher();
        let merged_reader = merged_searcher.segment_readers()[0].vector_index(merged_field)?;
        let merged_bounds = merged_reader.clusters().expect("IVF segment").bounds();
        for (cluster, &expected) in source_max.iter().enumerate() {
            assert_eq!(
                merged_bounds.ball_r(cluster).to_bits(),
                expected.to_bits(),
                "cluster {cluster}: merged bound must be the max of the sources'"
            );
        }
        Ok(())
    }

    /// A replicated IVF segment can be a merge SOURCE: merge it with a
    /// fresh commit segment and every doc — old and new — fills its cells
    /// against the same centroids.
    #[test]
    fn remerge_replicated_segment() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();
        let replicas = 3usize;
        let (index, embed_field, label_field) =
            build_inline_ivf(Metric::L2, &centroids, &docs, replicas)?;

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
        assert_eq!(segment_ids.len(), 2, "IVF segment + fresh segment");
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
        assert_eq!(info.num_vectors, total, "num_vectors counts distinct docs");
        let sizes = vec_reader.cluster_sizes().expect("ivf cluster sizes");
        let memberships: usize = sizes.iter().map(|&s| s as usize).sum();
        assert_eq!(
            memberships,
            replicas * total,
            "per-cluster sizes keep membership semantics"
        );
        Ok(())
    }

    /// Merging segments with deletes: rows written for since-deleted docs
    /// still count toward the sources' `count()` (tombstones don't rewrite
    /// `.vec`), so the alive-doc merge iteration legitimately comes up
    /// short of `vector_count`. The merge must tolerate that, and the
    /// resulting segment must hold — and count — the alive docs only.
    #[test]
    fn merge_segments_with_deletes() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let n = docs.len();

        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let index = Index::builder()
            .schema(sb.build())
            .centroid_producer(Arc::new(InlineCentroidProducer {
                centroids: centroids.clone(),
            }))
            .ivf_router(RouterKind::Rng)?
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

        // Tombstone docs in BOTH sources (d0/d7 in the first commit,
        // d35 in the second), then merge everything into one segment.
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
        assert_eq!(
            vec_reader.num_vectors(),
            alive,
            "deleted docs must not be counted"
        );
        // Every alive doc holds exactly one (replicas=1) membership, and
        // the memberships cover the merged doc space exactly.
        let ivf = vec_reader.clusters().expect("expected IVF storage");
        assert_eq!(ivf.num_rows(), alive);
        let mut all_docs: Vec<u32> = (0..ivf.num_clusters())
            .flat_map(|c| vec_reader.cluster_doc_ids(c).expect("in-bounds"))
            .collect();
        all_docs.sort_unstable();
        let expected: Vec<u32> = (0..alive as u32).collect();
        assert_eq!(all_docs, expected, "memberships must cover the alive docs");
        Ok(())
    }

    /// Merging when every doc carrying a vector for ONE field is deleted,
    /// while another field keeps live vectors: the emptied field owns no
    /// `.vec` slots at all and reads back as the empty placeholder — not
    /// an error — while the live field is untouched.
    #[test]
    fn merge_deleting_every_doc_of_one_field_writes_no_slots() -> crate::Result<()> {
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
        let index = Index::builder()
            .schema(sb.build())
            .centroid_producer(Arc::new(InlineCentroidProducer {
                centroids: centroids.clone(),
            }))
            .ivf_router(RouterKind::Rng)?
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Even docs carry the doomed field, odd docs the kept one, split
        // across two commits so BOTH sources hold doomed vectors.
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

        // Tombstone every doomed-field doc, then merge everything.
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

        // The emptied field reads back as the empty placeholder.
        let vec_reader = segment_reader.vector_index(doomed_field)?;
        assert_eq!(vec_reader.num_vectors(), 0);
        assert!(vec_reader.is_empty());
        assert!(vec_reader.info().is_none(), "no slots ⇒ no info");
        assert!(vec_reader.clusters().is_none());

        // The live field is untouched: every alive doc is counted.
        let kept_count = n / 2;
        assert_eq!(
            segment_reader.vector_index(kept_field)?.num_vectors(),
            kept_count
        );
        Ok(())
    }

    /// Captures `paradedb::ivf_build` log records so a test can read back the
    /// timings line the build emits.
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

    /// Every field build emits one parseable `ivf_build timings_ms ...`
    /// line. Builds a larger index so the phase timings are measurable,
    /// captures the line, and prints it (run with `--nocapture`) so we can
    /// see where build time goes.
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
        assert!(line.contains("assign="));
        eprintln!("IVF_BUILD_SAMPLE {line}");
        Ok(())
    }
    // ---- The flat (mutable/staging) tier ----

    /// [`build_ivf`] without a centroid index: every segment stores flat.
    fn build_flat(
        metric: Metric,
        commits: &[&[(&str, [f32; 2])]],
        merge: bool,
    ) -> crate::Result<(Index, crate::schema::Field, crate::schema::Field)> {
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let index = Index::builder().schema(sb.build()).create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for chunk in commits {
            for (label, v) in *chunk {
                let mut doc = TantivyDocument::new();
                doc.add_text(label_field, label);
                doc.add_vector(embed_field, v.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }
        if merge {
            let segment_ids = index.searchable_segment_ids()?;
            writer.merge(&segment_ids).wait()?;
        }
        writer.wait_merging_threads()?;
        Ok((index, embed_field, label_field))
    }

    /// Copy `source`'s segments (files + meta entries) into `dest` — same
    /// schema required. The tantivy-level stand-in for how a consumer
    /// moves a staged mutable segment into its real index.
    fn graft_segments(source: &Index, dest: &Index) -> crate::Result<()> {
        use std::io::Write as _;

        use crate::directory::{Directory, TerminatingWrite};
        use crate::index::SegmentComponent;
        let components = [
            SegmentComponent::Postings,
            SegmentComponent::Positions,
            SegmentComponent::Terms,
            SegmentComponent::Store,
            SegmentComponent::FastFields,
            SegmentComponent::FieldNorms,
            SegmentComponent::Custom(crate::vector::VEC_EXT.to_string()),
        ];
        let mut writer: IndexWriter = dest.writer_with_num_threads(1, 15_000_000)?;
        for meta in source.searchable_segment_metas()? {
            for component in &components {
                let path = meta.relative_path(component.clone());
                if !source.directory().exists(&path)? {
                    continue;
                }
                let bytes = source.directory().open_read(&path)?.read_bytes()?;
                let mut write = dest.directory().open_write(&path)?;
                write.write_all(&bytes)?;
                write.terminate()?;
            }
            writer.add_segment(dest.new_segment_meta(meta.id(), meta.max_doc()))?;
        }
        writer.commit()?;
        writer.wait_merging_threads()?;
        Ok(())
    }

    /// A no-set index scans exhaustively: results equal ground truth for
    /// any budget, all work lands in `exact_rows_read`, and the routed
    /// tier never runs.
    #[test]
    fn flat_index_searches_exactly() -> crate::Result<()> {
        let docs: Vec<(String, [f32; 2])> = (0..30)
            .map(|i| (format!("d{i}"), [i as f32, (i * 7 % 13) as f32]))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, label_field) =
            build_flat(Metric::L2, &[&docs[..10], &docs[10..]], false)?;

        let query = vec![4.2f32, 3.1];
        // A tiny budget must change nothing — the exact tier ignores it.
        let params = AdaptiveProbeParams {
            max_probe_fraction: 1e-6,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let plan = PreparedVectorSearch::new(
            &index.reader()?.searcher(),
            embed_field,
            &query,
            &params,
            true,
        )?;
        assert!(plan.routing_phases.segment_metadata_time_ns <= plan.routing_time_ns);
        assert_eq!(plan.routing_phases.router_open_time_ns, 0);
        assert_eq!(plan.routing_phases.centroid_precompute_time_ns, 0);
        assert_eq!(plan.routing_phases.router_prefix_time_ns, 0);
        let (hits, stats) = run_global(&index, embed_field, &AllQuery, query.clone(), 5, params)?;
        let truth = ground_truth::top_k(&index, embed_field, Metric::L2, &query, 5)?;
        assert_eq!(hits, truth);
        assert_eq!(stats.exact_rows_read, 30);
        assert_eq!(stats.segments_searched, 2);
        assert_eq!(stats.clusters_probed(), 0, "no routed tier without a set");
        assert_eq!(stats.filters_built, 0, "AllQuery builds no bitset");

        // Filtered: only matching docs qualify, filter bitsets built per
        // flat segment.
        let filter = TermQuery::new(
            Term::from_field_text(label_field, "d7"),
            IndexRecordOption::Basic,
        );
        let (hits, stats) =
            run_global(&index, embed_field, &filter, query, 5, exhaustive_params(1))?;
        assert_eq!(hits.len(), 1);
        assert_eq!(stored_label_at(&index, label_field, hits[0].1)?, "d7");
        assert_eq!(stats.exact_rows_read, 1, "filtered rows are never fetched");
        assert_eq!(
            stats.filters_built, 2,
            "the exact tier evaluates every flat segment's filter"
        );
        Ok(())
    }

    /// Merging inside a no-centroid-index index is refused: flat segments
    /// only ever merge INTO a clustered index (where their rows are
    /// assigned); the staging tier itself never merges.
    #[test]
    fn flat_only_merge_errors() -> crate::Result<()> {
        let docs: Vec<(String, [f32; 2])> = (0..20)
            .map(|i| (format!("d{i}"), [i as f32, (i * 3 % 7) as f32]))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, _) = build_flat(
            Metric::Cosine,
            &[&docs[..7], &docs[7..14], &docs[14..]],
            false,
        )?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        let segment_ids = index.searchable_segment_ids()?;
        // Foreground merge: the background path panics its merge thread on
        // any error under cfg(test).
        let err = writer.merge_foreground(&segment_ids, false).unwrap_err();
        assert!(
            err.to_string().contains("without a centroid index"),
            "unexpected: {err}"
        );
        drop(writer);

        // The failed merge changes nothing: still three flat segments,
        // still exact results.
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 3);
        let query = vec![0.6f32, 0.8];
        let (hits, stats) = run_global(
            &index,
            embed_field,
            &AllQuery,
            query.clone(),
            4,
            exhaustive_params(1),
        )?;
        let truth = ground_truth::top_k(&index, embed_field, Metric::Cosine, &query, 4)?;
        assert_eq!(hits, truth);
        assert_eq!(stats.exact_rows_read, 20);
        Ok(())
    }

    /// The grid the mixed-tier tests share: 4 well-separated centroids,
    /// clustered docs on the first three, flat (staged) docs near the
    /// fourth AND near the first — fresh data both inside and outside the
    /// clustered vocabulary's reach.
    const MIXED_CENTROIDS: [[f32; 2]; 4] = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]];

    fn mixed_fixture() -> crate::Result<(Index, crate::schema::Field, crate::schema::Field)> {
        let clustered: Vec<(String, [f32; 2])> = (0..30)
            .map(|i| {
                let c = MIXED_CENTROIDS[i % 3];
                (
                    format!("c{i}"),
                    [c[0] + (i / 3) as f32 * 0.5, c[1] + (i / 3) as f32 * 0.25],
                )
            })
            .collect();
        let clustered: Vec<(&str, [f32; 2])> =
            clustered.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, label_field) = build_ivf(
            Metric::L2,
            &MIXED_CENTROIDS,
            &[&clustered[..15], &clustered[15..]],
            1,
            false,
        )?;

        let staged: Vec<(String, [f32; 2])> = (0..8)
            .map(|i| {
                let c = MIXED_CENTROIDS[if i % 2 == 0 { 3 } else { 0 }];
                (format!("f{i}"), [c[0] + i as f32 * 0.3, c[1] + 1.0])
            })
            .collect();
        let staged: Vec<(&str, [f32; 2])> = staged.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (flat_index, _, _) = build_flat(Metric::L2, &[&staged], false)?;
        graft_segments(&flat_index, &index)?;
        Ok((index, embed_field, label_field))
    }

    /// Clustered and flat segments search into ONE heap: results equal
    /// ground truth over the union, the flat rows all pass through the
    /// exact tier, and the routed tier still probes.
    #[test]
    fn mixed_flat_and_clustered_search() -> crate::Result<()> {
        let (index, embed_field, label_field) = mixed_fixture()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 3);

        // Query near centroid 3: the best hits are staged docs, which only
        // the exact tier can find (no clustered segment has rows there).
        let query = vec![100.0f32, 101.0];
        let (hits, stats) = run_global(
            &index,
            embed_field,
            &AllQuery,
            query.clone(),
            6,
            exhaustive_params(4),
        )?;
        let truth = ground_truth::top_k(&index, embed_field, Metric::L2, &query, 6)?;
        assert_eq!(hits, truth);
        assert_eq!(stats.exact_rows_read, 8, "every staged row is read");
        assert_eq!(stats.segments_searched, 3);
        assert!(stats.clusters_probed() > 0, "the routed tier still runs");
        let top_label = stored_label_at(&index, label_field, hits[0].1)?;
        assert!(
            top_label.starts_with('f'),
            "freshest data wins: {top_label}"
        );

        // Query near centroid 0, where clustered and staged docs compete
        // in one heap.
        let query = vec![0.5f32, 0.5];
        let (hits, _) = run_global(
            &index,
            embed_field,
            &AllQuery,
            query.clone(),
            10,
            exhaustive_params(4),
        )?;
        let truth = ground_truth::top_k(&index, embed_field, Metric::L2, &query, 10)?;
        assert_eq!(hits, truth);
        Ok(())
    }

    /// Merging a mix of clustered and flat sources produces one clustered
    /// segment: carried-over postings for the clustered rows, fresh
    /// assignment for the flat rows — each doc in its nearest cluster.
    #[test]
    fn mixed_merge_assigns_only_flat_rows() -> crate::Result<()> {
        let (index, embed_field, label_field) = mixed_fixture()?;
        let labeled = |hits: &[(Score, DocAddress)]| -> crate::Result<Vec<(u32, String)>> {
            hits.iter()
                .map(|(score, addr)| {
                    Ok((
                        score.to_bits(),
                        stored_label_at(&index, label_field, *addr)?,
                    ))
                })
                .collect()
        };
        let query = vec![50.0f32, 50.0];
        let (before, _) = run_global(
            &index,
            embed_field,
            &AllQuery,
            query.clone(),
            12,
            exhaustive_params(4),
        )?;
        // Labels resolve against the CURRENT snapshot — before the merge
        // rewrites every address.
        let before_labeled = labeled(&before)?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let segment_reader = &searcher.segment_readers()[0];
        let vec = segment_reader.vector_index(embed_field)?;
        let ivf = vec.clusters().expect("the merged segment is clustered");
        assert_eq!(vec.num_vectors(), 38);

        // Every doc — carried or assigned — sits in its nearest cluster.
        for cluster in 0..ivf.num_clusters() {
            for doc in vec.cluster_doc_ids(cluster).unwrap() {
                let row = vec.vector_bytes(doc)?.unwrap();
                let point = decode_2d(&row);
                assert_eq!(
                    nearest_centroid(point, &MIXED_CENTROIDS),
                    cluster,
                    "doc {} landed in cluster {cluster}",
                    stored_label_at(&index, label_field, DocAddress::new(0, doc))?,
                );
            }
        }

        // Search results survive the merge bit-for-bit (modulo addresses):
        // compare (score, label) sequences.
        let (after, stats) = run_global(
            &index,
            embed_field,
            &AllQuery,
            query,
            12,
            exhaustive_params(4),
        )?;
        assert_eq!(before_labeled, labeled(&after)?);
        assert_eq!(stats.exact_rows_read, 0, "no flat segments remain");
        Ok(())
    }
    #[derive(Clone, Debug, PartialEq)]
    enum ProbeEvent {
        Extend {
            start: usize,
            len: usize,
            calls: usize,
        },
        Select {
            start: usize,
            end: usize,
            spent: u64,
        },
        Synchronize,
    }
    struct ParallelProbeState {
        barrier: Barrier,
        ranked: Mutex<Vec<RankedCluster>>,
        rank_ready: Condvar,
        rank_exhausted: std::sync::atomic::AtomicBool,
        independent_routing: bool,
        overlap_routing: bool,
        events: Mutex<Vec<ProbeEvent>>,
        next: Vec<std::sync::atomic::AtomicUsize>,
        costs: Mutex<Vec<ClusterWork>>,
        wave: Mutex<ProbeWave>,
        hits: Mutex<Vec<(Score, DocAddress)>>,
        limit: usize,
        reverse_delays: bool,
        selections: Mutex<Vec<(usize, usize, ProbeWave)>>,
    }
    struct ParallelProbeControl<'a> {
        shared: &'a ParallelProbeState,
        worker: usize,
    }
    impl VectorSearchControl for ParallelProbeControl<'_> {
        fn routes_clusters(&mut self) -> bool {
            self.worker == 0
        }
        fn can_overlap_routing(&self) -> bool {
            self.shared.overlap_routing
        }
        fn extend_clusters(
            &mut self,
            start: usize,
            clusters: &mut Vec<RankedCluster>,
            next: &mut dyn FnMut() -> Option<RankedCluster>,
        ) {
            if self.shared.independent_routing {
                let target = start + PROBE_WAVE_SIZE + 1;
                let mut ranked = self.shared.ranked.lock().unwrap();
                if self.worker == 0 {
                    let mut calls = 0;
                    let remaining = target.saturating_sub(ranked.len());
                    ranked.extend(
                        std::iter::from_fn(|| {
                            calls += 1;
                            next()
                        })
                        .take(remaining),
                    );
                    self.shared
                        .rank_exhausted
                        .store(ranked.len() < target, std::sync::atomic::Ordering::Relaxed);
                    self.shared.events.lock().unwrap().push(ProbeEvent::Extend {
                        start,
                        len: ranked.len(),
                        calls,
                    });
                    self.shared.rank_ready.notify_all();
                } else {
                    while ranked.len() < target
                        && !self
                            .shared
                            .rank_exhausted
                            .load(std::sync::atomic::Ordering::Relaxed)
                    {
                        ranked = self.shared.rank_ready.wait(ranked).unwrap();
                    }
                }
                clusters.extend_from_slice(&ranked[clusters.len()..]);
                return;
            }
            if self.worker == 0 {
                let mut ranked = self.shared.ranked.lock().unwrap();
                let remaining = (start + PROBE_WAVE_SIZE + 1).saturating_sub(ranked.len());
                ranked.extend(std::iter::from_fn(next).take(remaining));
            }
            self.shared.barrier.wait();
            {
                let ranked = self.shared.ranked.lock().unwrap();
                clusters.extend_from_slice(&ranked[clusters.len()..]);
            }
            self.shared.barrier.wait();
        }
        fn work_sharing(&self) -> bool {
            true
        }
        fn claim_work(&mut self, segment: u32) -> usize {
            self.shared.next[segment as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        }
        fn publish_initial_wave(&mut self, wave: ProbeWave) {
            let mut selections = self.shared.selections.lock().unwrap();
            if selections.is_empty() {
                selections.push((0, wave.end, wave));
            }
        }
        fn select_wave(
            &mut self,
            _: usize,
            start: usize,
            costs: &[ClusterWork],
            budget: ProbeBudget,
            spent: f64,
        ) -> ProbeWave {
            let delay = if self.shared.reverse_delays {
                7 - self.worker
            } else {
                self.worker
            };
            std::thread::sleep(std::time::Duration::from_micros((delay * 13) as u64));
            {
                let mut total = self.shared.costs.lock().unwrap();
                total.resize(costs.len(), ClusterWork::default());
                for (total, cost) in total.iter_mut().zip(costs) {
                    total.opens += cost.opens;
                    total.rows += cost.rows;
                }
            }
            if self.shared.barrier.wait().is_leader() {
                let mut costs = self.shared.costs.lock().unwrap();
                let wave = budget.select(start, &costs, spent);
                *self.shared.wave.lock().unwrap() = wave;
                self.shared
                    .selections
                    .lock()
                    .unwrap()
                    .push((start, costs.len(), wave));
                costs.clear();
                for next in &self.shared.next {
                    next.store(0, std::sync::atomic::Ordering::Relaxed);
                }
            }
            self.shared.barrier.wait();
            let wave = *self.shared.wave.lock().unwrap();
            self.shared.barrier.wait();
            if self.worker == 0 && self.shared.independent_routing {
                self.shared.events.lock().unwrap().push(ProbeEvent::Select {
                    start,
                    end: wave.end,
                    spent: wave.spent.to_bits(),
                });
            }
            wave
        }
        fn publish(&mut self, candidates: &[(Score, DocAddress)]) {
            let mut hits = self.shared.hits.lock().unwrap();
            hits.extend_from_slice(candidates);
            hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            hits.dedup_by_key(|hit| hit.1);
            hits.truncate(self.shared.limit);
        }
        fn synchronize(&mut self, _: usize, _: Option<Score>) -> Option<Score> {
            if self.worker == 0 && self.shared.independent_routing {
                self.shared
                    .events
                    .lock()
                    .unwrap()
                    .push(ProbeEvent::Synchronize);
            }
            self.shared.barrier.wait();
            let threshold = (self.shared.limit <= 1024)
                .then(|| {
                    self.shared
                        .hits
                        .lock()
                        .unwrap()
                        .get(self.shared.limit - 1)
                        .map(|hit| hit.0)
                })
                .flatten();
            self.shared.barrier.wait();
            threshold
        }
    }
    impl ParallelProbeState {
        fn new(workers: usize, segments: usize, limit: usize) -> Self {
            Self {
                barrier: Barrier::new(workers),
                ranked: Mutex::new(Vec::new()),
                rank_ready: Condvar::new(),
                rank_exhausted: std::sync::atomic::AtomicBool::new(false),
                independent_routing: false,
                overlap_routing: false,
                events: Mutex::new(Vec::new()),
                next: (0..segments)
                    .map(|_| std::sync::atomic::AtomicUsize::new(0))
                    .collect(),
                costs: Mutex::new(Vec::new()),
                wave: Mutex::new(ProbeWave::default()),
                hits: Mutex::new(Vec::new()),
                limit,
                reverse_delays: false,
                selections: Mutex::new(Vec::new()),
            }
        }
    }

    #[test]
    fn overlapped_filtered_prefix_preserves_budget_eof_and_work() -> crate::Result<()> {
        assert!(!().can_overlap_routing());
        for count in [7, 255, 256, 257, 529] {
            let centroids: Vec<_> = (0..count).map(|i| [i as f32, 0.0]).collect();
            let segments: Vec<Vec<_>> = (0..4)
                .map(|segment| {
                    centroids
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| i % 11 != 7)
                        .map(|(i, &point)| {
                            (
                                if segment < 3 && i % 3 != 1 {
                                    "keep"
                                } else {
                                    "other"
                                },
                                point,
                            )
                        })
                        .collect()
                })
                .collect();
            let commits: Vec<_> = segments.iter().map(Vec::as_slice).collect();
            let (index, field, label) = build_ivf_with_router(
                Metric::L2,
                &centroids,
                &commits,
                1,
                false,
                RouterKind::Exact,
            )?;
            let searcher = index.reader()?.searcher();
            assert_eq!(searcher.segment_readers().len(), 4);
            let query = vec![0.0f32, 0.0];
            let all_limit = searcher.num_docs() as usize + 1;
            let mut plan = PreparedVectorSearch::new(
                &searcher,
                field,
                &query,
                &exhaustive_params(count),
                false,
            )?;
            assert!(plan.shareable && plan.incremental);
            plan.budget.open = 1.0;
            plan.budget.row = 0.0;
            let first_wave_charge = 3.0
                * (0..count.min(PROBE_WAVE_SIZE))
                    .filter(|i| i % 11 != 7)
                    .count() as f64;
            let mut budgets = vec![
                0.0,
                0.5,
                3.0,
                first_wave_charge - 0.5,
                first_wave_charge,
                first_wave_charge + 0.5,
                count as f64 * 3.0 + 1.0,
            ];
            budgets.sort_by(f64::total_cmp);
            budgets.dedup();
            for budget in budgets {
                plan.budget.limit = budget;
                let limits = if count == 529 {
                    vec![10, all_limit]
                } else {
                    vec![all_limit]
                };
                for limit in limits {
                    let collector =
                        TopDocs::with_limit(limit).order_by_similarity(field, query.clone());
                    for workers in [1, 2, 4] {
                        let filter = TermQuery::new(
                            Term::from_field_text(label, "keep"),
                            IndexRecordOption::Basic,
                        );
                        let run = |overlap| {
                            let mut shared = ParallelProbeState::new(workers, 4, limit);
                            shared.independent_routing = true;
                            shared.overlap_routing = overlap;
                            shared.reverse_delays = workers != 2;
                            let results = std::thread::scope(|scope| {
                                let handles: Vec<_> = (0..workers)
                                    .map(|worker| {
                                        let (searcher, collector, plan, filter, shared) =
                                            (&searcher, &collector, &plan, &filter, &shared);
                                        scope.spawn(move || {
                                            collector
                                                .search_prepared(
                                                    searcher,
                                                    filter,
                                                    plan,
                                                    (0..4).filter(|ord| {
                                                        *ord as usize % workers == worker
                                                    }),
                                                    &mut ParallelProbeControl { shared, worker },
                                                )
                                                .unwrap()
                                        })
                                    })
                                    .collect();
                                handles
                                    .into_iter()
                                    .map(|handle| handle.join().unwrap())
                                    .collect::<Vec<_>>()
                            });
                            let prefix: Vec<_> = shared
                                .ranked
                                .lock()
                                .unwrap()
                                .iter()
                                .map(|cluster| (cluster.id, cluster.similarity.to_bits()))
                                .collect();
                            let events = shared.events.lock().unwrap().clone();
                            (results, prefix, events)
                        };
                        let (baseline, prefix, events) = run(false);
                        let (overlapped, actual_prefix, actual_events) = run(true);
                        assert_eq!(
                            actual_prefix, prefix,
                            "count={count}, budget={budget}, workers={workers}"
                        );
                        for (actual, expected) in overlapped.iter().zip(&baseline) {
                            let hits = |results: &[(Score, DocAddress)]| {
                                results
                                    .iter()
                                    .map(|(score, doc)| (score.to_bits(), *doc))
                                    .collect::<Vec<_>>()
                            };
                            assert_eq!(hits(&actual.results), hits(&expected.results));
                            let semantic = |stats: &ProbeStats| {
                                let mut value = serde_json::to_value(stats).unwrap();
                                value
                                    .as_object_mut()
                                    .unwrap()
                                    .retain(|key, _| !key.ends_with("_time_ns"));
                                value
                            };
                            assert_eq!(semantic(&actual.stats), semantic(&expected.stats));
                            assert_eq!(actual.stats.cluster_flags, expected.stats.cluster_flags);
                        }
                        let extensions = |events: &[ProbeEvent]| {
                            events
                                .iter()
                                .filter(|event| matches!(event, ProbeEvent::Extend { .. }))
                                .cloned()
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(extensions(&actual_events), extensions(&events));
                        let selections = |events: &[ProbeEvent]| {
                            events
                                .iter()
                                .filter(|event| matches!(event, ProbeEvent::Select { .. }))
                                .cloned()
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(selections(&actual_events), selections(&events));
                        let syncs = |events: &[ProbeEvent]| {
                            events
                                .iter()
                                .filter(|event| matches!(event, ProbeEvent::Synchronize))
                                .count()
                        };
                        assert_eq!(syncs(&actual_events), syncs(&events));
                        let mut previous_len = 0;
                        for event in extensions(&actual_events) {
                            let ProbeEvent::Extend { start, len, calls } = event else {
                                unreachable!()
                            };
                            let target = start + PROBE_WAVE_SIZE + 1;
                            assert_eq!(len, count.min(target));
                            assert_eq!(calls, len - previous_len + usize::from(count < target));
                            previous_len = len;
                        }
                        for (i, event) in events.iter().enumerate() {
                            if matches!(event, ProbeEvent::Select { .. }) {
                                assert_eq!(events[i + 1], ProbeEvent::Synchronize);
                            }
                        }
                        for (i, event) in actual_events.iter().enumerate() {
                            if let ProbeEvent::Select { start, end, spent } = event {
                                if end > start && f64::from_bits(*spent) < budget {
                                    assert!(
                                        matches!(actual_events[i + 1], ProbeEvent::Extend { start, .. } if start == *end)
                                    );
                                    assert_eq!(actual_events[i + 2], ProbeEvent::Synchronize);
                                } else {
                                    assert_eq!(actual_events[i + 1], ProbeEvent::Synchronize);
                                }
                            }
                        }
                        if budget > count as f64 * 3.0 {
                            assert_eq!(prefix.len(), count);
                            assert!(overlapped
                                .iter()
                                .all(|result| result.stats.termination
                                    == ProbeTermination::Exhausted));
                            let matching = collect_filter_doc_set(&index, &filter)?;
                            let expected: Vec<_> =
                                ground_truth::top_k(&index, field, Metric::L2, &query, all_limit)?
                                    .into_iter()
                                    .filter(|(_, doc)| matching.contains(doc))
                                    .take(limit)
                                    .collect();
                            let mut hits: Vec<_> = overlapped
                                .iter()
                                .flat_map(|result| result.results.iter().copied())
                                .collect();
                            hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                            hits.truncate(limit);
                            assert_eq!(hits, expected);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn adaptive_budget_is_independent_of_workers_and_completion_order() -> crate::Result<()> {
        let centroids: Vec<_> = (0..400)
            .map(|i| [(i % 20) as f32, (i / 20) as f32])
            .collect();
        let labels: Vec<_> = (0..2400)
            .map(|i| {
                if i % 7 == 0 {
                    "keep".to_owned()
                } else {
                    format!("d{i}")
                }
            })
            .collect();
        let docs: Vec<_> = labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let c = centroids[i % centroids.len()];
                (
                    label.as_str(),
                    [c[0] + (i / 400) as f32 * 0.01, c[1] + 0.03],
                )
            })
            .collect();
        let commits: Vec<_> = docs.chunks(300).collect();
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            for replicas in [1, 2] {
                let (index, field, label) =
                    build_ivf(metric, &centroids, &commits, replicas, false)?;
                let searcher = index.reader()?.searcher();
                let query = vec![6.8f32, 8.1];
                for fraction in [0.02, 0.15, 0.7, 1.0] {
                    let params = AdaptiveProbeParams {
                        max_probe_fraction: fraction,
                        min_probe_clusters: 1,
                    };
                    let filters: Vec<Box<dyn Query>> = vec![
                        Box::new(AllQuery),
                        Box::new(TermQuery::new(
                            Term::from_field_text(label, "keep"),
                            IndexRecordOption::Basic,
                        )),
                        Box::new(TermQuery::new(
                            Term::from_field_text(label, "missing"),
                            IndexRecordOption::Basic,
                        )),
                    ];
                    for filter in filters {
                        let all = filter
                            .weight(EnableScoring::disabled_from_searcher(&searcher))?
                            .matches_all_docs();
                        let plan =
                            PreparedVectorSearch::new(&searcher, field, &query, &params, all)?;
                        assert_eq!(plan.initial_wave.is_some(), all && plan.shareable);
                        let compare_legacy =
                            metric == Metric::L2 && replicas == 1 && fraction == 0.7 && all;
                        let limit = if compare_legacy { docs.len() + 1 } else { 10 };
                        let collector =
                            TopDocs::with_limit(limit).order_by_similarity(field, query.clone());
                        let serial_state =
                            ParallelProbeState::new(1, searcher.segment_readers().len(), limit);
                        let serial = collector.search_prepared(
                            &searcher,
                            filter.as_ref(),
                            &plan,
                            0..searcher.segment_readers().len() as u32,
                            &mut ParallelProbeControl {
                                shared: &serial_state,
                                worker: 0,
                            },
                        )?;
                        if plan.incremental {
                            let set = searcher.index().cached_centroid_index()?;
                            let router = set.field_router(field).unwrap();
                            let mut values = query.clone();
                            if metric == Metric::Cosine {
                                let norm =
                                    crate::vector::distance::norm_squared_wide(&values).sqrt();
                                for value in &mut values {
                                    *value = (f64::from(*value) / norm) as f32;
                                }
                            }
                            let mut workspace = crate::vector::router::RouterWorkspace::default();
                            let mut eager = plan.clone();
                            eager.incremental = false;
                            eager.clusters = router
                                .rank_clusters(&mut workspace, &values)
                                .map(|candidate| RankedCluster {
                                    id: candidate.node,
                                    similarity: candidate.sim.score(),
                                })
                                .collect();
                            let expected = collector.search_prepared(
                                &searcher,
                                filter.as_ref(),
                                &eager,
                                0..searcher.segment_readers().len() as u32,
                                &mut (),
                            )?;
                            assert_eq!(serial.results, expected.results);
                            let semantic = |stats: &ProbeStats| -> crate::Result<_> {
                                let mut value = serde_json::to_value(stats)?;
                                let fields = value.as_object_mut().unwrap();
                                for key in [
                                    "routing",
                                    "routing_time_ns",
                                    "segment_metadata_time_ns",
                                    "router_open_time_ns",
                                    "centroid_precompute_time_ns",
                                    "router_prefix_time_ns",
                                    "filter_time_ns",
                                    "probe_time_ns",
                                    "segment_setup_time_ns",
                                ] {
                                    fields.remove(key);
                                }
                                Ok(value)
                            };
                            assert_eq!(semantic(&serial.stats)?, semantic(&expected.stats)?);
                        }
                        if compare_legacy {
                            let legacy = TopDocs::with_limit(limit)
                                .order_by_similarity(field, query.clone())
                                .with_adaptive_params(params.clone())
                                .search(&searcher, filter.as_ref())?;
                            assert!(legacy.stats.candidates_scored > 1000);
                            assert_eq!(legacy.stats.bounds_skips, 0);
                            assert_eq!(legacy.results, serial.results);
                            assert_eq!(
                                legacy.stats.candidates_scored,
                                serial.stats.candidates_scored
                            );
                            assert_eq!(legacy.stats.segment_opens, serial.stats.segment_opens);
                        }
                        for workers in [1, 2, 4, 8] {
                            let shared = ParallelProbeState::new(
                                workers,
                                searcher.segment_readers().len(),
                                limit,
                            );
                            let results = std::thread::scope(|scope| {
                                let handles: Vec<_> = (0..workers)
                                    .map(|worker| {
                                        let (searcher, collector, filter, plan, shared) = (
                                            &searcher,
                                            &collector,
                                            filter.as_ref(),
                                            &plan,
                                            &shared,
                                        );
                                        scope.spawn(move || {
                                            std::thread::sleep(std::time::Duration::from_micros(
                                                (worker * 47) as u64,
                                            ));
                                            collector
                                                .search_prepared(
                                                    searcher,
                                                    filter,
                                                    plan,
                                                    (0..searcher.segment_readers().len() as u32)
                                                        .filter(|ord| {
                                                            *ord as usize % workers == worker
                                                        }),
                                                    &mut ParallelProbeControl { shared, worker },
                                                )
                                                .unwrap()
                                        })
                                    })
                                    .collect();
                                handles
                                    .into_iter()
                                    .map(|handle| handle.join().unwrap())
                                    .collect::<Vec<_>>()
                            });
                            if replicas == 1
                                && filter
                                    .weight(EnableScoring::disabled_from_searcher(&searcher))?
                                    .matches_all_docs()
                            {
                                assert!(shared.next.iter().any(|next| {
                                    next.load(std::sync::atomic::Ordering::Relaxed) > 0
                                }));
                            }
                            let mut hits: Vec<_> = results
                                .iter()
                                .flat_map(|fruit| fruit.results.iter().copied())
                                .collect();
                            if plan.incremental {
                                let routing: Vec<_> = results
                                    .iter()
                                    .filter_map(|result| result.stats.routing)
                                    .collect();
                                assert_eq!(routing.len(), 1);
                                assert_eq!(
                                    serde_json::to_value(routing[0])?,
                                    serde_json::to_value(serial.stats.routing)?
                                );
                            }
                            hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                            hits.truncate(limit);
                            assert_eq!(
                                hits, serial.results,
                                "metric={metric:?}, replicas={replicas}, budget={fraction}, \
                                 workers={workers}"
                            );
                            assert_eq!(
                                results
                                    .iter()
                                    .map(|r| r.stats.candidates_scored)
                                    .sum::<usize>(),
                                serial.stats.candidates_scored
                            );
                            assert_eq!(
                                results.iter().map(|r| r.stats.bounds_skips).sum::<u32>(),
                                serial.stats.bounds_skips
                            );
                            if !all || !plan.shareable {
                                assert!(shared
                                    .selections
                                    .lock()
                                    .unwrap()
                                    .iter()
                                    .all(|(_, len, _)| *len <= 256));
                            }
                            let work: f32 = results.iter().map(|r| r.stats.work_charged).sum();
                            assert!(
                                (work - serial.stats.work_charged).abs() < 0.01,
                                "{work} vs {}",
                                serial.stats.work_charged
                            );
                        }
                        if fraction == 1.0
                            && filter
                                .weight(EnableScoring::disabled_from_searcher(&searcher))?
                                .matches_all_docs()
                        {
                            assert_eq!(
                                serial.results,
                                ground_truth::top_k(&index, field, metric, &query, limit)?
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn filtered_capacity_excludes_empty_segments_and_precedes_routing() -> crate::Result<()> {
        #[derive(Default)]
        struct CapacityControl {
            capacity: Option<ClusterWork>,
            synchronizations: usize,
            routes: usize,
            finished: Option<(u64, RoutingPhases)>,
        }
        impl VectorSearchControl for CapacityControl {
            fn add_capacity(&mut self, capacity: ClusterWork) {
                assert_eq!(self.synchronizations, 0);
                assert!(self.capacity.is_none());
                self.capacity = Some(capacity);
            }
            fn capacity(&self) -> Option<ClusterWork> {
                assert!(self.synchronizations > 0);
                self.capacity
            }
            fn synchronize(&mut self, _: usize, local: Option<Score>) -> Option<Score> {
                self.synchronizations += 1;
                local
            }
            fn routes_clusters(&mut self) -> bool {
                assert!(self.synchronizations > 0);
                self.routes += 1;
                true
            }
            fn finish(&mut self, stats: &ProbeStats) {
                self.finished = Some((stats.routing_time_ns, stats.routing_phases));
            }
        }
        let centroids = [[0.0f32, 0.0], [10.0, 0.0], [20.0, 0.0]];
        let first = [
            ("keep", [0.0, 0.0]),
            ("other", [10.0, 0.0]),
            ("other", [20.0, 0.0]),
        ];
        let empty = [
            ("other", [0.0, 0.0]),
            ("other", [10.0, 0.0]),
            ("other", [20.0, 0.0]),
        ];
        let last = [
            ("keep", [0.0, 0.0]),
            ("keep", [10.0, 0.0]),
            ("other", [20.0, 0.0]),
        ];
        let (index, field, label) =
            build_ivf(Metric::L2, &centroids, &[&first, &empty, &last], 1, false)?;
        let searcher = index.reader()?.searcher();
        let query = vec![0.0f32, 0.0];
        let plan = PreparedVectorSearch::new(
            &searcher,
            field,
            &query,
            &exhaustive_params(centroids.len()),
            false,
        )?;
        assert!(plan.shareable);
        let collector = TopDocs::with_limit(10).order_by_similarity(field, query.clone());
        for (term, opens, rows) in [("keep", 6, 3), ("missing", 0, 0)] {
            let filter =
                TermQuery::new(Term::from_field_text(label, term), IndexRecordOption::Basic);
            let mut boundary = plan.clone();
            boundary.budget.limit = 2.0 * (rows as f64 * plan.budget.row);
            let mut control = CapacityControl::default();
            let actual =
                collector.search_prepared(&searcher, &filter, &boundary, 0..3, &mut control)?;
            let capacity = control.capacity.unwrap();
            assert_eq!(capacity.opens, opens);
            assert_eq!(capacity.rows, rows);
            let matching = collect_filter_doc_set(&index, &filter)?;
            let expected: Vec<_> = ground_truth::top_k(
                &index,
                field,
                Metric::L2,
                &query,
                searcher.num_docs() as usize,
            )?
            .into_iter()
            .filter(|(_, addr)| matching.contains(addr))
            .take(10)
            .collect();
            assert_eq!(actual.results, expected);
            assert_eq!(actual.stats.candidates_scored as u64, rows);
            assert_eq!(actual.stats.exact_rows_read as u64, rows);
            assert_eq!(actual.stats.vectors_visited as u64, opens);
            assert_eq!(actual.stats.pruned_filter as u64, opens - rows);
            assert_eq!(actual.stats.segment_opens, 0);
            assert_eq!(actual.stats.clusters_probed(), 0);
            assert_eq!(actual.stats.precomputed_centroids, 0);
            assert!(actual.stats.routing.is_none());
            assert_eq!(
                actual.stats.work_charged,
                (rows as f64 * plan.budget.row) as f32
            );
            assert_eq!(control.routes, 0);
            assert_eq!(control.synchronizations, 2);
            assert_eq!(control.finished, Some((0, RoutingPhases::default())));
            assert_eq!(actual.stats.routing_phases, boundary.routing_phases);
            assert_eq!(actual.stats.routing_time_ns, boundary.routing_time_ns);
            let lazy = collector.search_prepared(&searcher, &filter, &plan, 0..3, &mut ())?;
            assert_eq!(lazy.stats.precomputed_centroids, 0);
            assert_eq!(lazy.stats.exact_rows_read, 0);
            assert!(lazy.stats.routing.is_some());
            assert_eq!(actual.results, lazy.results);
            if rows == 0 {
                continue;
            }
            let mut limited = boundary.clone();
            limited.budget.limit -= 0.01 * plan.budget.row;
            assert!(limited.budget.limit >= limited.budget.charge(capacity) * 0.125);
            let mut control = CapacityControl::default();
            let actual =
                collector.search_prepared(&searcher, &filter, &limited, 0..3, &mut control)?;
            let lazy = collector.search_prepared(&searcher, &filter, &limited, 0..3, &mut ())?;
            assert_eq!(actual.stats.exact_rows_read, 0);
            assert_eq!(actual.stats.precomputed_centroids, centroids.len());
            assert_eq!(control.routes, 1);
            let (producer_time, producer_phases) = control.finished.unwrap();
            assert_eq!(producer_phases.segment_metadata_time_ns, 0);
            let producer_sum = producer_phases.router_open_time_ns
                + producer_phases.centroid_precompute_time_ns
                + producer_phases.router_prefix_time_ns;
            assert!(producer_sum <= producer_time);
            let mut expected_phases = producer_phases;
            expected_phases += limited.routing_phases;
            assert_eq!(actual.stats.routing_phases, expected_phases);
            assert_eq!(
                actual.stats.routing_time_ns,
                producer_time + limited.routing_time_ns
            );
            assert_eq!(actual.results, lazy.results);
            assert_eq!(actual.stats.candidates_scored, lazy.stats.candidates_scored);
            assert_eq!(
                actual.stats.work_charged.to_bits(),
                lazy.stats.work_charged.to_bits()
            );
            assert_eq!(
                serde_json::to_value(actual.stats.routing)?,
                serde_json::to_value(lazy.stats.routing)?
            );
        }
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        writer.delete_term(Term::from_field_text(label, "keep"));
        writer.commit()?;
        writer.wait_merging_threads()?;
        let searcher = index.reader()?.searcher();
        let deleted = PreparedVectorSearch::new(
            &searcher,
            field,
            &query,
            &exhaustive_params(centroids.len()),
            false,
        )?;
        assert!(!deleted.shareable);
        let filter = TermQuery::new(
            Term::from_field_text(label, "other"),
            IndexRecordOption::Basic,
        );
        let mut control = CapacityControl::default();
        let actual = collector.search_prepared(&searcher, &filter, &deleted, 0..3, &mut control)?;
        assert!(control.capacity.is_none());
        assert_eq!(actual.stats.exact_rows_read, 0);
        assert!(actual.stats.routing.is_some());
        let (index, field, label) = build_ivf(Metric::L2, &centroids, &[&first, &last], 2, false)?;
        let searcher = index.reader()?.searcher();
        let replicated = PreparedVectorSearch::new(
            &searcher,
            field,
            &query,
            &exhaustive_params(centroids.len()),
            false,
        )?;
        assert!(!replicated.shareable);
        let filter = TermQuery::new(
            Term::from_field_text(label, "keep"),
            IndexRecordOption::Basic,
        );
        let mut control = CapacityControl::default();
        let actual = TopDocs::with_limit(10)
            .order_by_similarity(field, query)
            .search_prepared(&searcher, &filter, &replicated, 0..2, &mut control)?;
        assert!(control.capacity.is_none());
        assert_eq!(actual.stats.exact_rows_read, 0);
        assert!(actual.stats.routing.is_some());
        Ok(())
    }

    #[test]
    fn routing_metadata_sidecar_preserves_legacy_prefix_work_and_results() -> crate::Result<()> {
        use crate::directory::{Directory, TerminatingWrite};
        use crate::index::SegmentComponent;
        use crate::vector::routing_metadata::VectorRoutingMetadata;
        use crate::vector::VMETA_EXT;

        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        for (replicas, merge, delete) in [
            (1, false, false),
            (1, false, true),
            (2, false, false),
            (2, true, true),
        ] {
            let (index, field, label) = build_ivf(
                Metric::L2,
                &centroids,
                &[&docs[..18], &docs[18..]],
                replicas,
                merge,
            )?;
            if delete {
                let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
                writer.set_merge_policy(Box::new(NoMergePolicy));
                writer.delete_term(Term::from_field_text(label, docs[0].0));
                writer.commit()?;
                writer.wait_merging_threads()?;
            }
            let segments = index.searchable_segments()?;
            let paths: Vec<_> = segments
                .iter()
                .map(|segment| {
                    segment.relative_path(SegmentComponent::Custom(VMETA_EXT.to_string()))
                })
                .collect();
            assert!(paths
                .iter()
                .all(|path| index.directory().exists(path).unwrap()));
            let query = vec![1.0f32, 1.5];
            let collector = TopDocs::with_limit(10).order_by_similarity(field, query.clone());
            let filters: Vec<Box<dyn Query>> = vec![
                Box::new(AllQuery),
                Box::new(TermQuery::new(
                    Term::from_field_text(label, docs[3].0),
                    IndexRecordOption::Basic,
                )),
                Box::new(TermQuery::new(
                    Term::from_field_text(label, "missing"),
                    IndexRecordOption::Basic,
                )),
            ];
            let run = || -> crate::Result<Vec<_>> {
                let searcher = index.reader()?.searcher();
                let mut observations = Vec::new();
                for fraction in [0.001, 0.5, 1.0] {
                    let params = AdaptiveProbeParams {
                        max_probe_fraction: fraction,
                        min_probe_clusters: 1,
                    };
                    for (i, filter) in filters.iter().enumerate() {
                        let plan =
                            PreparedVectorSearch::new(&searcher, field, &query, &params, i == 0)?;
                        let fruit = collector.search_prepared(
                            &searcher,
                            filter.as_ref(),
                            &plan,
                            0..searcher.segment_readers().len() as u32,
                            &mut (),
                        )?;
                        let mut stats = serde_json::to_value(&fruit.stats)?;
                        stats
                            .as_object_mut()
                            .unwrap()
                            .retain(|key, _| !key.ends_with("_time_ns"));
                        let plan = serde_json::json!({
                            "incremental": plan.incremental,
                            "shareable": plan.shareable,
                            "centroids": plan.num_centroids,
                            "budget": [plan.budget.limit.to_bits(), plan.budget.open.to_bits(), plan.budget.row.to_bits()],
                            "wave": plan.initial_wave.map(|wave| (wave.end, wave.spent.to_bits())),
                            "clusters": plan.clusters.iter().map(|cluster| (cluster.id, cluster.similarity.to_bits())).collect::<Vec<_>>(),
                            "routing": plan.routing,
                        });
                        let results: Vec<_> = fruit
                            .results
                            .iter()
                            .map(|(score, addr)| (score.to_bits(), *addr))
                            .collect();
                        observations.push((plan, stats, results, fruit.stats.cluster_flags));
                    }
                }
                Ok(observations)
            };
            let expected = run()?;
            for path in &paths {
                index.directory().delete(path).unwrap();
                assert_eq!(run()?, expected);
            }
            let searcher = index.reader()?.searcher();
            for segment in searcher.segment_readers() {
                let routing = VectorRoutingMetadata::open(segment, field)?;
                let full = segment.vector_index(field)?;
                assert_eq!(routing.num_docs(), full.num_vectors());
                assert_eq!(routing.num_rows(), full.clusters().unwrap().num_rows());
            }
            let mut write = index.directory().open_write(&paths[0])?;
            use std::io::Write;
            write.write_all(b"corrupt routing metadata")?;
            write.terminate()?;
            assert!(PreparedVectorSearch::new(
                &index.reader()?.searcher(),
                field,
                &query,
                &exhaustive_params(centroids.len()),
                true,
            )
            .is_err());
        }
        Ok(())
    }

    #[test]
    fn precomputed_centroid_plan_preserves_prefix_and_eligibility() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let (index, field, label) =
                build_ivf(metric, &centroids, &[&docs[..18], &docs[18..]], 1, false)?;
            let searcher = index.reader()?.searcher();
            let query = vec![1.0f32, 1.5];
            for fraction in [0.1, 0.299, 0.3, 0.5, 0.8, 1.0] {
                let params = AdaptiveProbeParams {
                    max_probe_fraction: fraction,
                    min_probe_clusters: 1,
                };
                for all in [false, true] {
                    let lazy = PreparedVectorSearch::new(&searcher, field, &query, &params, all)?;
                    let scored = PreparedVectorSearch::new_with_precomputed_centroid_scores(
                        &searcher, field, &query, &params, all,
                    )?;
                    assert_eq!(lazy.precomputed_centroids, 0);
                    assert_eq!(lazy.routing_phases.centroid_precompute_time_ns, 0);
                    for plan in [&lazy, &scored] {
                        let phases = plan.routing_phases;
                        let sum = phases.segment_metadata_time_ns
                            + phases.router_open_time_ns
                            + phases.centroid_precompute_time_ns
                            + phases.router_prefix_time_ns;
                        assert!(sum <= plan.routing_time_ns);
                        if !all {
                            assert_eq!(phases.router_prefix_time_ns, 0);
                        }
                        if plan.precomputed_centroids == 0 {
                            assert_eq!(phases.centroid_precompute_time_ns, 0);
                        }
                    }
                    assert_eq!(
                        scored.precomputed_centroids,
                        if all && fraction >= 0.3 {
                            centroids.len()
                        } else {
                            0
                        },
                    );
                    assert_eq!(
                        lazy.clusters
                            .iter()
                            .map(|c| (c.id, c.similarity.to_bits()))
                            .collect::<Vec<_>>(),
                        scored
                            .clusters
                            .iter()
                            .map(|c| (c.id, c.similarity.to_bits()))
                            .collect::<Vec<_>>(),
                    );
                    assert_eq!(
                        lazy.initial_wave.map(|w| (w.end, w.spent.to_bits())),
                        scored.initial_wave.map(|w| (w.end, w.spent.to_bits())),
                    );
                    assert_eq!(
                        serde_json::to_value(lazy.routing)?,
                        serde_json::to_value(scored.routing)?
                    );
                    if all {
                        let collector =
                            TopDocs::with_limit(10).order_by_similarity(field, query.clone());
                        let expected = collector.search_prepared(
                            &searcher,
                            &AllQuery,
                            &lazy,
                            0..2,
                            &mut (),
                        )?;
                        let actual = collector.search_prepared(
                            &searcher,
                            &AllQuery,
                            &scored,
                            0..2,
                            &mut (),
                        )?;
                        assert_eq!(actual.results, expected.results);
                        assert_eq!(actual.stats.routing_phases, scored.routing_phases);
                        assert_eq!(actual.stats.routing_time_ns, scored.routing_time_ns);
                        assert_eq!(
                            actual.stats.candidates_scored,
                            expected.stats.candidates_scored
                        );
                        assert_eq!(
                            actual.stats.work_charged.to_bits(),
                            expected.stats.work_charged.to_bits()
                        );
                    }
                }
            }
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.delete_term(Term::from_field_text(label, docs[0].0));
            writer.commit()?;
            writer.wait_merging_threads()?;
            let searcher = index.reader()?.searcher();
            let deleted = PreparedVectorSearch::new_with_precomputed_centroid_scores(
                &searcher,
                field,
                &query,
                &exhaustive_params(centroids.len()),
                true,
            )?;
            assert!(!deleted.shareable);
            assert_eq!(deleted.precomputed_centroids, 0);
        }
        let (index, field, _) = build_ivf(Metric::L2, &centroids, &[&docs], 2, false)?;
        let searcher = index.reader()?.searcher();
        let replicated = PreparedVectorSearch::new_with_precomputed_centroid_scores(
            &searcher,
            field,
            &[1.0f32, 1.5],
            &exhaustive_params(centroids.len()),
            true,
        )?;
        assert!(!replicated.shareable);
        assert_eq!(replicated.precomputed_centroids, 0);
        Ok(())
    }

    #[test]
    fn budgeted_initial_prefix_preserves_work_and_order_across_workers() -> crate::Result<()> {
        let count = 2 * PROBE_WAVE_SIZE + 17;
        let centroids: Vec<_> = (0..count)
            .map(|i| [(i % 47) as f32 * 4.0, (i / 47) as f32 * 4.0])
            .collect();
        let owned_docs: Vec<_> = centroids
            .iter()
            .enumerate()
            .flat_map(|(cluster, &vector)| {
                (0..1 + cluster % 5).map(move |copy| {
                    (
                        format!("{cluster}:{copy}"),
                        vector,
                        (cluster + copy * 3) % 8,
                    )
                })
            })
            .collect();
        let mut segments = vec![Vec::new(); 8];
        for (label, vector, segment) in &owned_docs {
            segments[*segment].push((label.as_str(), *vector));
        }
        let commits: Vec<_> = segments.iter().map(Vec::as_slice).collect();
        let (index, field, _) = build_ivf(Metric::L2, &centroids, &commits, 1, false)?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 8);
        let query = vec![53.0f32, 69.0];
        let full_plan =
            PreparedVectorSearch::new(&searcher, field, &query, &exhaustive_params(count), true)?;
        assert_eq!(full_plan.clusters.len(), count);
        assert!(full_plan.shareable);
        assert!(full_plan.initial_wave.is_some());
        let limit = owned_docs.len() + 1;
        let collector = TopDocs::with_limit(limit).order_by_similarity(field, query.clone());
        let all_hits = ground_truth::top_k(&index, field, Metric::L2, &query, limit)?;
        let mut costs = vec![ClusterWork::default(); count];
        let mut docs_by_rank = vec![Vec::new(); count];
        for (ord, reader) in searcher.segment_readers().iter().enumerate() {
            let vector = reader.vector_index(field)?;
            for (rank, cluster) in full_plan.clusters.iter().enumerate() {
                if let Some(rows) = vector
                    .clusters()
                    .unwrap()
                    .non_empty_cluster_range(cluster.id as usize)
                {
                    costs[rank].opens += 1;
                    costs[rank].rows += rows.len() as u64;
                    docs_by_rank[rank]
                        .extend(rows.map(|row| DocAddress::new(ord as u32, vector.doc_id_at(row))));
                }
            }
        }
        assert!(costs.windows(2).any(|pair| pair[0].rows != pair[1].rows));
        let mut prefix_work = vec![0.0];
        for &cost in &costs {
            prefix_work.push(prefix_work.last().unwrap() + full_plan.budget.charge(cost));
        }
        for cutoff in [
            PROBE_WAVE_SIZE - 1,
            PROBE_WAVE_SIZE,
            PROBE_WAVE_SIZE + 1,
            2 * PROBE_WAVE_SIZE + 5,
            count,
        ] {
            let params = AdaptiveProbeParams {
                max_probe_fraction: if cutoff == count {
                    1.0
                } else {
                    ((prefix_work[cutoff - 1] + prefix_work[cutoff])
                        / (2.0 * full_plan.budget.limit)) as f32
                },
                min_probe_clusters: 1,
            };
            let visited: std::collections::HashSet<_> =
                docs_by_rank[..cutoff].iter().flatten().copied().collect();
            let expected_hits: Vec<_> = all_hits
                .iter()
                .filter(|(_, doc)| visited.contains(doc))
                .copied()
                .collect();
            let expected_opens: u64 = costs[..cutoff].iter().map(|cost| cost.opens).sum();
            for initial in [false, true] {
                let plan = PreparedVectorSearch::new(&searcher, field, &query, &params, initial)?;
                if initial {
                    let wave = plan.initial_wave.unwrap();
                    assert_eq!(wave.end, cutoff);
                    assert_eq!(wave.spent, prefix_work[cutoff]);
                    assert_eq!(plan.clusters.len(), (cutoff + 1).min(count));
                    assert!(plan.clusters.iter().zip(&full_plan.clusters).all(|(a, b)| {
                        a.id == b.id && a.similarity.to_bits() == b.similarity.to_bits()
                    }));
                } else {
                    assert!(plan.clusters.is_empty());
                    assert!(plan.incremental);
                    assert!(plan.initial_wave.is_none());
                }
                let serial =
                    collector.search_prepared(&searcher, &AllQuery, &plan, 0..8, &mut ())?;
                assert_eq!(serial.results, expected_hits);
                assert_eq!(serial.stats.bounds_skips, 0);
                assert_eq!(
                    serial.stats.termination,
                    if cutoff == count {
                        ProbeTermination::Exhausted
                    } else {
                        ProbeTermination::Ceiling
                    }
                );
                for workers in [1, 2, 4, 8] {
                    for reverse_delays in [false, true] {
                        let mut shared = ParallelProbeState::new(workers, 8, limit);
                        shared.reverse_delays = reverse_delays;
                        let results = std::thread::scope(|scope| {
                            let handles: Vec<_> = (0..workers)
                                .map(|worker| {
                                    let (searcher, collector, plan, shared) =
                                        (&searcher, &collector, &plan, &shared);
                                    scope.spawn(move || {
                                        let delay = if reverse_delays {
                                            workers - worker - 1
                                        } else {
                                            worker
                                        };
                                        std::thread::sleep(std::time::Duration::from_micros(
                                            (delay * 47) as u64,
                                        ));
                                        collector
                                            .search_prepared(
                                                searcher,
                                                &AllQuery,
                                                plan,
                                                (0..8).filter(|ord| {
                                                    *ord as usize % workers == worker
                                                }),
                                                &mut ParallelProbeControl { shared, worker },
                                            )
                                            .unwrap()
                                    })
                                })
                                .collect();
                            handles
                                .into_iter()
                                .map(|handle| handle.join().unwrap())
                                .collect::<Vec<_>>()
                        });
                        let mut hits: Vec<_> = results
                            .iter()
                            .flat_map(|result| result.results.iter().copied())
                            .collect();
                        hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                        assert_eq!(
                            hits, expected_hits,
                            "cutoff={cutoff}, workers={workers}, initial={initial}, \
                             reverse={reverse_delays}"
                        );
                        assert_eq!(
                            results
                                .iter()
                                .map(|r| r.stats.candidates_scored)
                                .sum::<usize>(),
                            visited.len()
                        );
                        assert_eq!(
                            results
                                .iter()
                                .map(|r| r.stats.vectors_visited)
                                .sum::<usize>(),
                            visited.len()
                        );
                        assert_eq!(
                            results.iter().map(|r| r.stats.segment_opens).sum::<usize>(),
                            expected_opens as usize
                        );
                        assert_eq!(results.iter().map(|r| r.stats.bounds_skips).sum::<u32>(), 0);
                        assert!(results
                            .iter()
                            .all(|r| r.stats.termination == serial.stats.termination));
                        let charged: f32 = results.iter().map(|r| r.stats.work_charged).sum();
                        assert!((f64::from(charged) - prefix_work[cutoff]).abs() < 0.01);
                        for rank in 0..plan.clusters.len() {
                            let flags = results
                                .iter()
                                .fold(0, |flags, result| flags | result.stats.cluster_flags[rank]);
                            assert_eq!(flags, if rank < cutoff { 3 } else { 0 });
                        }
                        let selections = shared.selections.lock().unwrap();
                        assert_eq!(
                            selections.len(),
                            if initial {
                                1
                            } else {
                                cutoff.div_ceil(PROBE_WAVE_SIZE)
                            }
                        );
                        for (wave, &(start, _, selected)) in selections.iter().enumerate() {
                            assert_eq!(start, wave * PROBE_WAVE_SIZE);
                            assert_eq!(
                                selected.end,
                                if initial {
                                    cutoff
                                } else {
                                    ((wave + 1) * PROBE_WAVE_SIZE).min(cutoff)
                                }
                            );
                            assert!((selected.spent - prefix_work[selected.end]).abs() < 1e-8);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn budgeted_prefix_rejects_filtered_reuse_and_preserves_fallbacks() -> crate::Result<()> {
        struct PrivateControl;
        impl VectorSearchControl for PrivateControl {
            fn publish_initial_wave(&mut self, _: ProbeWave) {
                panic!("private execution must not publish a shared wave");
            }
        }
        let centroids = [[0.0f32, 0.0], [10.0, 0.0], [20.0, 0.0]];
        let docs = [
            ("keep", [0.0, 0.0]),
            ("a", [1.0, 0.0]),
            ("b", [10.0, 0.0]),
            ("c", [20.0, 0.0]),
        ];
        let (index, field, label) = build_ivf(Metric::L2, &centroids, &[&docs], 1, false)?;
        let searcher = index.reader()?.searcher();
        let query = vec![0.0f32, 0.0];
        let params = exhaustive_params(centroids.len());
        let plan = PreparedVectorSearch::new(&searcher, field, &query, &params, true)?;
        assert!(plan.initial_wave.is_some());
        let collector = TopDocs::with_limit(10).order_by_similarity(field, query.clone());
        let private =
            collector.search_prepared(&searcher, &AllQuery, &plan, 0..1, &mut PrivateControl)?;
        assert_eq!(private.results.len(), docs.len());
        let filter = TermQuery::new(
            Term::from_field_text(label, "keep"),
            IndexRecordOption::Basic,
        );
        let error = collector
            .search_prepared(&searcher, &filter, &plan, 0..1, &mut ())
            .err()
            .expect("truncated all-docs plan must reject a filtered query");
        assert!(error.to_string().contains("cannot search a filtered query"));
        let filtered = PreparedVectorSearch::new(&searcher, field, &query, &params, false)?;
        assert!(filtered.initial_wave.is_none());
        assert!(filtered.clusters.is_empty());
        assert!(filtered.incremental);
        let result = collector.search_prepared(&searcher, &filter, &filtered, 0..1, &mut ())?;
        assert_eq!(result.results.len(), 1);
        assert_eq!(stored_label_at(&index, label, result.results[0].1)?, "keep");
        let weight = AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let (zero, stats) = crate::vector::search::parallel::search(
            &searcher,
            weight.as_ref(),
            field,
            &Arc::new(query.clone()),
            0,
            &NoTieBreak,
            &plan,
            0..1,
            &mut PrivateControl,
        )?;
        assert!(zero.is_empty());
        assert_eq!(stats.candidates_scored, 0);

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        writer.delete_term(Term::from_field_text(label, "keep"));
        writer.commit()?;
        writer.wait_merging_threads()?;
        let searcher = index.reader()?.searcher();
        let deleted = PreparedVectorSearch::new(&searcher, field, &query, &params, true)?;
        assert!(!deleted.shareable);
        assert!(deleted.initial_wave.is_none());
        assert!(deleted.clusters.is_empty());
        assert!(deleted.incremental);
        let result = collector.search_prepared(&searcher, &AllQuery, &deleted, 0..1, &mut ())?;
        assert_eq!(result.results.len(), 3);
        assert_eq!(result.stats.pruned_dead, 1);

        let empty_index = Index::create_in_ram(index.schema());
        let empty_searcher = empty_index.reader()?.searcher();
        let empty = PreparedVectorSearch::new(&empty_searcher, field, &query, &params, true)?;
        assert!(empty.clusters.is_empty());
        assert!(empty.initial_wave.is_none());
        let result =
            collector.search_prepared(&empty_searcher, &AllQuery, &empty, 0..0, &mut ())?;
        assert!(result.results.is_empty());
        assert_eq!(result.stats.candidates_scored, 0);
        Ok(())
    }

    #[test]
    fn coalesced_direct_ranges_preserve_order_work_and_fallbacks() -> crate::Result<()> {
        struct RecordingControl {
            coalesce: bool,
            stream: bool,
            next: Vec<usize>,
            claims: Vec<(u32, usize, usize)>,
            acceptance: Vec<DocAddress>,
            interrupts: Vec<usize>,
        }
        impl VectorSearchControl for RecordingControl {
            fn coalesce_direct_ranges(&self) -> bool {
                self.coalesce
            }
            fn stream_row_scores(&self) -> bool {
                self.stream
            }
            fn work_sharing(&self) -> bool {
                true
            }
            fn claim_work(&mut self, segment: u32) -> usize {
                let batch = self.next[segment as usize];
                self.next[segment as usize] += 1;
                self.claims.push((segment, batch, self.acceptance.len()));
                batch
            }
            fn accept(&mut self, doc: DocAddress) -> bool {
                self.acceptance.push(doc);
                doc.doc_id % 5 != 0
            }
            fn check_interrupt(&mut self) {
                self.interrupts.push(self.acceptance.len());
            }
        }
        let centroids: Vec<_> = (0..70)
            .map(|cluster| {
                let angle = cluster as f32 * std::f32::consts::TAU / 70.0;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let mut owned = Vec::new();
        for copy in 0..96 {
            for cluster in (0..70).map(|cluster| cluster * 29 % 70) {
                let count = if cluster % 11 == 0 { 96 } else { 5 };
                if cluster % 9 != 1 && copy < count {
                    owned.push((format!("{cluster}:{copy}"), centroids[cluster]));
                }
            }
        }
        let docs: Vec<_> = owned
            .iter()
            .map(|(label, vector)| (label.as_str(), *vector))
            .collect();
        let (index, field, label) = build_ivf_with_router(
            Metric::Cosine,
            &centroids,
            &[&docs],
            1,
            false,
            RouterKind::Exact,
        )?;
        let searcher = index.reader()?.searcher();
        let vector = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vector.clusters().unwrap();
        assert_eq!(ivf.num_clusters(), 70);
        assert!((0..70).any(|cluster| ivf.cluster_range(cluster).is_empty()));
        assert!((0..70).any(|cluster| ivf.cluster_range(cluster).len() > 64));
        assert!((0..docs.len()).any(|row| vector.doc_id_at(row) as usize != row));
        let compare = |index: &Index,
                       filter: &dyn Query,
                       query: Vec<f32>,
                       stream: bool,
                       clean: bool|
         -> crate::Result<()> {
            let searcher = index.reader()?.searcher();
            let all_docs = filter
                .weight(EnableScoring::disabled_from_searcher(&searcher))?
                .matches_all_docs();
            let plan = PreparedVectorSearch::new_with_precomputed_centroid_scores(
                &searcher,
                field,
                &query,
                &exhaustive_params(centroids.len()),
                all_docs,
            )?;
            let collector = TopDocs::with_limit(10).order_by_similarity(field, query.clone());
            let run = |coalesce| {
                let mut control = RecordingControl {
                    coalesce,
                    stream,
                    next: vec![0; searcher.segment_readers().len()],
                    claims: Vec::new(),
                    acceptance: Vec::new(),
                    interrupts: Vec::new(),
                };
                let result = collector.search_prepared(
                    &searcher,
                    filter,
                    &plan,
                    0..searcher.segment_readers().len() as u32,
                    &mut control,
                )?;
                crate::Result::Ok((result, control))
            };
            let (expected, before) = run(false)?;
            let (actual, after) = run(true)?;
            assert_eq!(
                actual
                    .results
                    .iter()
                    .map(|(score, doc)| (score.to_bits(), *doc))
                    .collect::<Vec<_>>(),
                expected
                    .results
                    .iter()
                    .map(|(score, doc)| (score.to_bits(), *doc))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(after.acceptance, before.acceptance);
            assert_eq!(after.claims, before.claims);
            assert_eq!(actual.stats.cluster_flags, expected.stats.cluster_flags);
            assert_eq!(
                actual.stats.work_charged.to_bits(),
                expected.stats.work_charged.to_bits()
            );
            let semantic = |stats: &ProbeStats| -> crate::Result<_> {
                let mut value = serde_json::to_value(stats)?;
                value
                    .as_object_mut()
                    .unwrap()
                    .retain(|key, _| !key.ends_with("_time_ns"));
                Ok(value)
            };
            assert_eq!(semantic(&actual.stats)?, semantic(&expected.stats)?);
            if clean {
                assert!(plan.shareable && all_docs);
                assert_eq!(plan.initial_wave.unwrap().end, centroids.len());
                assert!(after.claims.iter().any(|&(_, batch, _)| batch == 2));
                let truth: Vec<_> =
                    ground_truth::top_k(index, field, Metric::Cosine, &query, docs.len())?
                        .into_iter()
                        .filter(|(_, doc)| doc.doc_id % 5 != 0)
                        .take(10)
                        .collect();
                assert_eq!(actual.results, truth);
                if query == [0.0, 0.0] {
                    let vector = searcher.segment_readers()[0].vector_index(field)?;
                    let mut ranks: Vec<_> = (0..plan.initial_wave.unwrap().end).collect();
                    ranks[16..].sort_unstable_by_key(|&rank| plan.clusters[rank].id);
                    let order: Vec<_> = ranks
                        .into_iter()
                        .flat_map(|rank| {
                            vector
                                .clusters()
                                .unwrap()
                                .cluster_range(plan.clusters[rank].id as usize)
                                .map(|row| DocAddress::new(0, vector.doc_id_at(row)))
                        })
                        .collect();
                    assert_eq!(after.acceptance, order);
                    assert_eq!(after.acceptance.len(), actual.stats.candidates_scored);
                    assert!(actual.stats.pruned_invisible > 0);
                    for control in [&before, &after] {
                        let mut checkpoints = control.interrupts.clone();
                        checkpoints.push(control.acceptance.len());
                        assert!(checkpoints.windows(2).all(|pair| pair[1] - pair[0] <= 64));
                    }
                    assert!(
                        after.interrupts.len() < before.interrupts.len(),
                        "fixture must exercise coalescing"
                    );
                }
            } else {
                assert!(!all_docs || !plan.shareable);
                assert!(after.claims.is_empty());
                assert_eq!(after.interrupts, before.interrupts);
            }
            Ok(())
        };
        for stream in [false, true] {
            compare(&index, &AllQuery, vec![0.0, 0.0], stream, true)?;
            compare(&index, &AllQuery, vec![-1.0, 0.0], stream, true)?;
        }
        let filter = TermQuery::new(
            Term::from_field_text(label, docs[0].0),
            IndexRecordOption::Basic,
        );
        compare(&index, &filter, vec![0.0, 0.0], true, false)?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        writer.delete_term(Term::from_field_text(label, docs[0].0));
        writer.commit()?;
        writer.wait_merging_threads()?;
        compare(&index, &AllQuery, vec![0.0, 0.0], true, false)?;
        let (replicated, replicated_field, _) = build_ivf_with_router(
            Metric::Cosine,
            &centroids,
            &[&docs],
            2,
            false,
            RouterKind::Exact,
        )?;
        assert_eq!(replicated_field, field);
        compare(&replicated, &AllQuery, vec![0.0, 0.0], true, false)?;
        Ok(())
    }

    #[test]
    fn adaptive_checks_eligibility_before_heap_and_deduplicates_replicas() -> crate::Result<()> {
        use crate::vector::{PreparedVectorSearch, VectorSearchControl};
        struct Eligible;
        impl VectorSearchControl for Eligible {
            fn accept(&mut self, doc: DocAddress) -> bool {
                doc.doc_id % 2 == 0
            }
        }
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        for replicas in [1, 2, 3] {
            let (index, field, _) = build_ivf(
                Metric::L2,
                &centroids,
                &[&docs[..18], &docs[18..]],
                replicas,
                false,
            )?;
            let searcher = index.reader()?.searcher();
            let query = vec![1.0f32, 1.0];
            let plan = PreparedVectorSearch::new(
                &searcher,
                field,
                &query,
                &exhaustive_params(centroids.len()),
                true,
            )?;
            let truth: Vec<_> = ground_truth::top_k(&index, field, Metric::L2, &query, 100)?
                .into_iter()
                .filter(|(_, doc)| doc.doc_id % 2 == 0)
                .take(10)
                .collect();
            let fruit = TopDocs::with_limit(10)
                .order_by_similarity(field, query)
                .search_prepared(
                    &searcher,
                    &AllQuery,
                    &plan,
                    0..searcher.segment_readers().len() as u32,
                    &mut Eligible,
                )?;
            assert_eq!(fruit.results, truth);
            assert!(fruit.stats.pruned_invisible > 0);
            if replicas > 1 {
                assert!(fruit.stats.pruned_seen > 0);
            }
        }
        Ok(())
    }

    #[test]
    fn adaptive_searches_flat_segments_and_filters() -> crate::Result<()> {
        use crate::vector::PreparedVectorSearch;
        let (index, field, label) = mixed_fixture()?;
        let searcher = index.reader()?.searcher();
        let query = vec![100.0f32, 101.0];
        let plan =
            PreparedVectorSearch::new(&searcher, field, &query, &exhaustive_params(4), true)?;
        let collector = TopDocs::with_limit(10).order_by_similarity(field, query.clone());
        let result = collector.search_prepared(
            &searcher,
            &AllQuery,
            &plan,
            0..searcher.segment_readers().len() as u32,
            &mut (),
        )?;
        assert_eq!(
            result.results,
            ground_truth::top_k(&index, field, Metric::L2, &query, 10)?
        );
        assert_eq!(result.stats.exact_rows_read, 8);
        for value in ["f0", "c0", "missing"] {
            let filter = TermQuery::new(
                Term::from_field_text(label, value),
                IndexRecordOption::Basic,
            );
            let result = collector.search_prepared(
                &searcher,
                &filter,
                &plan,
                0..searcher.segment_readers().len() as u32,
                &mut (),
            )?;
            assert_eq!(result.results.len(), usize::from(value != "missing"));
            if let Some((_, doc)) = result.results.first() {
                assert_eq!(stored_label_at(&index, label, *doc)?, value);
            }
        }
        Ok(())
    }
}
