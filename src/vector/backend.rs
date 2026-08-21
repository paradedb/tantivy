//! The probe work-unit model and search instrumentation.
//!
//! The cross-segment probe loop itself lives in [`search`](super::search);
//! this module owns what the loop charges (the work-unit model, calibrated
//! on the reference fixture) and what it reports ([`ProbeStats`]).

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use super::ivf::IvfSearchMetrics;

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
    use std::sync::Arc;

    use super::*;
    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
    use crate::vector::ivf::AdaptiveProbeParams;
    use crate::vector::tests::{exhaustive_params, ground_truth, TestVectorIndex};
    use crate::vector::{
        CentroidProducer, IvfCentroids, IvfMatrix, Metric, NoTieBreak, VectorDType, VectorInfo,
        VectorOptions,
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
        Ok(index
            .reader()?
            .searcher()
            .search(filter, &collector)?
            .results)
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
        let ivf = vec_reader.index().expect("expected IVF storage");
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
        assert_eq!(stats.routing.visited_count, DEFAULT_NUM_CENTROIDS);
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
            let ivf = ivf_reader.index().expect("IVF segment");
            total_nonempty += ivf.num_non_empty_clusters();
            total_docs += ivf.num_docs();
        }
        let n_avg = total_docs as f64 / centroids.len() as f64;
        let x = crate::vector::backend::open_share(n_avg);
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
        };
        let (_, stats) = run_global(&index, embed_field, &AllQuery, vec![10.0, 10.0], 3, params)?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert_eq!(stats.clusters_probed(), 1);
        assert_eq!(stats.routing.visited_count, centroids.len());
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

    /// A single-centroid set writes no router slot; routing takes the
    /// exact linear fallback and still returns the exact top-K.
    #[test]
    fn single_centroid_routes_without_router_slot() -> crate::Result<()> {
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
    /// layout against the index-level set: correct centroid count, the
    /// set's version stamp, and every doc in its primary cluster.
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
                    let ivf = vec_reader.index().expect("IVF segment");
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
            let bounds = vec_reader.index().expect("IVF segment").bounds();
            for (cluster, slot) in source_max.iter_mut().enumerate() {
                *slot = slot.max(bounds.ball_r(cluster));
            }
        }
        let merged_searcher = merged.reader()?.searcher();
        let merged_reader = merged_searcher.segment_readers()[0].vector_index(merged_field)?;
        let merged_bounds = merged_reader.index().expect("IVF segment").bounds();
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
    /// against the same set, stamped with the same version.
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
        let ivf = vec_reader.index().expect("expected IVF storage");
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
        assert!(vec_reader.index().is_none());

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

    /// [`build_ivf`] without a centroid set: every segment stores flat.
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
        };
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

    /// Merging inside a no-set index keeps the flat layout and the exact
    /// results.
    #[test]
    fn flat_merge_stays_flat() -> crate::Result<()> {
        let docs: Vec<(String, [f32; 2])> = (0..20)
            .map(|i| (format!("d{i}"), [i as f32, (i * 3 % 7) as f32]))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, _) = build_flat(
            Metric::Cosine,
            &[&docs[..7], &docs[7..14], &docs[14..]],
            true,
        )?;

        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let vec = searcher.segment_readers()[0].vector_index(embed_field)?;
        assert!(vec.index().is_none(), "the merged segment stays flat");
        assert_eq!(vec.num_vectors(), 20);

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
        let ivf = vec.index().expect("the merged segment is clustered");
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
}
