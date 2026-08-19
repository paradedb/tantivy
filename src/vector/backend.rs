//! Per-segment vector search execution.
//!
//! Built once per segment by
//! [`TopDocsByVectorSimilarity`](super::collector::TopDocsByVectorSimilarity)
//! around the segment's cached [`VectorIndexReader`].
//!
//! TODO(cross-segment search): centroids moved to the index-level set file
//! and every segment shares its cluster ids, so search should rank the
//! set's centroids ONCE and, per ranked cluster, gather that cluster's
//! rows across ALL segments into one global heap (shared kth threshold),
//! instead of exhausting a per-segment probe budget segment by segment.
//! Until that lands, `top_n`/`top_n_by` return an error. The work-unit
//! model below is kept: it meters probe work independently of which loop
//! spends it, and its constants are calibrated.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use super::index_reader::VectorIndexReader;
use super::ivf::{AdaptiveProbeParams, IvfSearchMetrics};
use super::prepared::PreparedQuery;
use super::tie_break::NoTieBreak;
use super::VectorElement;
use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::SegmentSortKeyComputer;
use crate::query::Weight;
use crate::schema::Field;
use crate::{DocAddress, Score, SegmentOrdinal, SegmentReader, TantivyError};

/// The settled result.
type TieBreakHits<K> = Vec<(
    (Score, <K as SegmentSortKeyComputer>::SegmentSortKey),
    DocAddress,
)>;

/// Per-segment vector search: the segment's [`VectorIndexReader`] plus the
/// per-query state. Build via [`VectorBackend::for_segment`].
pub struct VectorBackend<T: VectorElement> {
    #[allow(dead_code)] // Consumed by the cross-segment search path (TODO).
    reader: Arc<VectorIndexReader>,
    #[allow(dead_code)] // Consumed by the cross-segment search path (TODO).
    query: Arc<PreparedQuery<T>>,
    #[allow(dead_code)] // Consumed by the cross-segment search path (TODO).
    adaptive: AdaptiveProbeParams,
    #[allow(dead_code)] // Consumed by the cross-segment search path (TODO).
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

    /// Top-N within this segment.
    ///
    /// TODO(cross-segment search): not yet implemented for the index-level
    /// centroid set format — see the module docs for the intended shape.
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
    /// TODO(cross-segment search): not yet implemented for the index-level
    /// centroid set format — see the module docs for the intended shape.
    pub fn top_n_by<K, CTail>(
        &self,
        _weight: &dyn Weight,
        _segment_reader: &SegmentReader,
        _top_n: usize,
        _tie_break: &mut K,
        _tie_comparator: CTail,
    ) -> crate::Result<(TieBreakHits<K>, ProbeStats)>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        Err(TantivyError::InvalidArgument(
            "vector search over the index-level centroid set format is not yet implemented: the \
             cross-segment probe loop is TODO"
                .to_string(),
        ))
    }
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

/// Probe-loop instrumentation: a prune breakdown of every doc the inner
/// loop touched, plus posting-fetch counters. Kept through the search
/// rewrite so the harness's schema stays stable; the cross-segment loop
/// (TODO) fills it again.
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
    /// Exact-path stride-sized row reads — one per survivor scored.
    pub exact_rows_read: usize,
    /// Routing cost of ranking the clusters to probe. See
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
    pub bound_armed_at_probe: Option<u32>,
    /// How the probe loop terminated.
    pub termination: ProbeTermination,
    /// Work units the probe loop charged against its resolved budget:
    /// opens at `x`, scored rows at `(1 - x)/n_avg`.
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
#[allow(dead_code)] // Consumed by the cross-segment search path (TODO).
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
#[allow(dead_code)] // Consumed by the cross-segment search path (TODO).
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

#[allow(dead_code)] // Consumed by the cross-segment search path (TODO).
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
    // Write-path test gate: every segment — per-commit and merged —
    // assigns against the index-level centroid set and stores the V3
    // clustered layout. Search is TODO (cross-segment probe loop), so
    // these tests assert the WRITTEN state through the reader's
    // introspection surface (`cluster_doc_ids`, `info`, `vector_bytes`)
    // rather than through queries.
    // ============================================================
    use std::sync::Arc;

    use super::*;
    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::AllQuery;
    use crate::schema::{Schema, Term, STORED, STRING};
    use crate::vector::{
        CentroidIndex, IvfCentroids, IvfMatrix, Metric, VectorDType, VectorInfo, VectorOptions,
    };
    use crate::{Index, IndexWriter, TantivyDocument};

    /// Fixed-centroid [`CentroidIndex`]: the consumer "trained" these
    /// centroids elsewhere; tantivy only assigns against them.
    pub(crate) struct InlineCentroidIndex {
        pub(crate) centroids: Vec<[f32; 2]>,
        pub(crate) version: u64,
    }

    impl CentroidIndex for InlineCentroidIndex {
        fn version(&self) -> u64 {
            self.version
        }
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

    /// Build a single-IVF-segment index with the supplied centroids and
    /// labelled docs. Splits docs across two commits so the merge has
    /// ≥ 2 source segments to consume. Returns the index plus the
    /// `(embedding, label)` field handles.
    fn build_inline_ivf(
        metric: Metric,
        centroids: &[[f32; 2]],
        docs: &[(&str, [f32; 2])],
        replicas: usize,
    ) -> crate::Result<(Index, crate::schema::Field, crate::schema::Field)> {
        assert!(docs.len() >= 2, "need ≥ 2 docs for ≥ 2 source segments");
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();

        let settings = IndexSettings {
            vector_replicas: replicas,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .centroid_index(Arc::new(InlineCentroidIndex {
                centroids: centroids.to_vec(),
                version: 1,
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
            .centroid_index(Arc::new(InlineCentroidIndex {
                centroids: centroids.clone(),
                version: 42,
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
                centroid_set_version: 42,
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
        assert_eq!(info.centroid_set_version, 1);
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
            .centroid_index(Arc::new(InlineCentroidIndex {
                centroids: centroids.clone(),
                version: 1,
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
            .centroid_index(Arc::new(InlineCentroidIndex {
                centroids: centroids.clone(),
                version: 1,
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

    /// TODO(cross-segment search): until the global probe loop lands,
    /// vector search fails loudly instead of returning wrong results.
    #[test]
    fn vector_search_is_a_hard_todo() -> crate::Result<()> {
        let (centroids, labels) = replication_fixture();
        let docs = replication_docs(&centroids, &labels);
        let (index, embed_field, _) = build_inline_ivf(Metric::L2, &centroids, &docs, 1)?;
        let searcher = index.reader()?.searcher();
        let collector = TopDocs::with_limit(3).order_by_similarity(embed_field, vec![0.0_f32, 0.0]);
        let err = searcher.search(&AllQuery, &collector).unwrap_err();
        assert!(
            matches!(err, TantivyError::InvalidArgument(ref msg) if msg.contains("not yet implemented")),
            "unexpected error: {err:?}"
        );
        Ok(())
    }
}
