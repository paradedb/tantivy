//! Per-segment dispatch over vector storage formats.
//!
//! Picked once per segment by
//! [`TopDocsByVectorSimilarity`](super::collector::TopDocsByVectorSimilarity). Each variant owns
//! its top-N loop: [`FlatBackend`] iterates the filter `Scorer` doc-by-doc, [`IvfBackend`] drains
//! the filter into a bitmap and probes clusters adaptively.
//!
//! Adding a new format (HNSW, etc.) is a new enum variant — the
//! collector layer doesn't change.

use std::cmp::Ordering;
use std::sync::Arc;

use common::BitSet;

use super::flat::FlatVectorColumn;
use super::ivf::{AdaptiveProbeParams, IvfVectorColumn};
use super::options::{Metric, VectorElement};
use super::prepared::PreparedQuery;
use super::reader::{VectorColumn, VectorColumnReader, VectorReader};
use crate::collector::sort_key::NaturalComparator;
use crate::collector::TopNComputer;
use crate::fastfield::AliveBitSet;
use crate::query::Weight;
use crate::schema::{Field, FieldType, Schema};
use crate::{DocAddress, DocId, Score, SegmentOrdinal, SegmentReader, TantivyError};

/// Per-segment vector backend. Pick via [`VectorBackend::for_segment`].
pub enum VectorBackend<T: VectorElement> {
    Flat(FlatBackend<T>),
    Ivf(IvfBackend<T>),
}

pub struct FlatBackend<T: VectorElement> {
    column: FlatVectorColumn,
    query: Arc<PreparedQuery<T>>,
    segment_ord: SegmentOrdinal,
}

pub struct IvfBackend<T: VectorElement> {
    column: IvfVectorColumn,
    query: Arc<PreparedQuery<T>>,
    adaptive: AdaptiveProbeParams,
    segment_ord: SegmentOrdinal,
}

impl<T: VectorElement> VectorBackend<T> {
    /// Open the segment's vector column using the storage format recorded in
    /// vector metadata.
    /// Returns an error if the segment has no vector data at all.
    pub fn for_segment(
        segment_reader: &SegmentReader,
        segment_ord: SegmentOrdinal,
        field: Field,
        query: Arc<Vec<T>>,
        adaptive: AdaptiveProbeParams,
    ) -> crate::Result<Self> {
        let schema = segment_reader.schema();
        let metric = lookup_metric(schema, field)?;

        let vec_reader = VectorReader::open(segment_reader)?;

        let query = Arc::new(PreparedQuery::<T>::new(metric, query));
        match vec_reader.open_column(field)? {
            VectorColumn::Ivf(column) => Ok(Self::Ivf(IvfBackend {
                column,
                query,
                adaptive,
                segment_ord,
            })),
            VectorColumn::Flat(column) => Ok(Self::Flat(FlatBackend {
                column,
                query,
                segment_ord,
            })),
        }
    }

    /// Top-N within this segment. Each variant decides whether to
    /// iterate the filter (flat) or drain it into a bitmap (IVF).
    /// Hits come back already tagged with `DocAddress` (the backend
    /// holds its own `SegmentOrdinal`), so the collector doesn't need
    /// a second pass to attach the segment.
    pub fn top_n(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        match self {
            Self::Flat(b) => b.top_n(weight, segment_reader, top_n),
            Self::Ivf(b) => b.top_n(weight, segment_reader, top_n),
        }
    }
}

impl<T: VectorElement> FlatBackend<T> {
    fn top_n(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        // `for_each_no_score` walks the filter DocSet in ascending doc
        // order, which lets us use the fast `TopNComputer::push` path
        // (strict-greater threshold short-circuit, valid only under
        // ascending-D pushes). IVF's cluster-order iteration would use
        // `push_unordered` instead.
        //
        // The heap keys on segment-local `DocId` (cheaper compares than
        // `DocAddress`); we tag with `self.segment_ord` at drain time
        // so the collector returns ready-to-use `DocAddress`es without
        // a second pass.
        //
        // `NaturalComparator` is required: vector similarity is
        // "higher = better", so we want top-N *largest*. The default
        // `TopNComputer::new()` wires `ReverseComparator`, which keeps
        // top-N *smallest* — for our convention that returns the K
        // *farthest* docs under truncation. See the matching note in
        // `IvfBackend::top_n`.
        let mut topn = TopNComputer::<Score, DocId, NaturalComparator>::new_with_comparator(
            top_n,
            NaturalComparator,
        );
        let alive = segment_reader.alive_bitset();
        weight.for_each_no_score(segment_reader, &mut |docs| {
            for &doc in docs {
                if let Some(bs) = alive {
                    if !bs.is_alive(doc) {
                        continue;
                    }
                }
                if let Some(bytes) = self.column.vector_bytes_at(doc) {
                    let score = self.query.score_doc_bytes(bytes);
                    topn.push(score, doc);
                }
            }
        })?;
        let segment_ord = self.segment_ord;
        Ok(topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect())
    }
}

/// Test-only instrumentation collected by `IvfBackend::top_n_instrumented`:
/// which clusters were probed (in probe order) and how many candidate
/// docs were scored. Useful for asserting efficiency properties of the
/// adaptive probe loop in scenarios that pair the IVF backend with a
/// known geometry.
///
/// The production `top_n` entry point passes `None` to the shared inner
/// helper and pays no allocation for stats accumulation.
#[derive(Debug, Default)]
pub struct ProbeStats {
    /// Clusters visited by the probe loop, in probe order. A cluster
    /// appears here once we've passed the stop-condition gate for it,
    /// regardless of whether its doc-ids slice ends up empty.
    pub probed_clusters: Vec<usize>,
    /// Number of docs that survived the filter + alive checks and got
    /// scored against the query.
    pub candidates_scored: usize,
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

impl<T: VectorElement> IvfBackend<T> {
    fn top_n(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        self.top_n_inner(weight, segment_reader, top_n, None)
    }

    /// Same logic as `top_n` but also returns a `ProbeStats` describing
    /// which clusters the adaptive loop visited. Test-only seam used by
    /// the 2D fixture scenarios.
    #[cfg(test)]
    pub(crate) fn top_n_instrumented(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let mut stats = ProbeStats::default();
        let hits = self.top_n_inner(weight, segment_reader, top_n, Some(&mut stats))?;
        Ok((hits, stats))
    }

    fn top_n_inner(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        stats: Option<&mut ProbeStats>,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        if top_n == 0 {
            return Ok(Vec::new());
        }
        let max_doc = segment_reader.max_doc();
        if max_doc == 0 {
            return Ok(Vec::new());
        }

        // Drain the filter `DocSet` into a dense BitSet for random
        // membership testing per cluster doc. The BitSet allocates
        // `max_doc / 8` bytes regardless of filter selectivity —
        // inherent to IVF needing O(1) membership tests on
        // out-of-order doc ids. Revisit only if memory profiling
        // flags it.
        let mut filter = BitSet::with_max_value(max_doc);
        weight.for_each_no_score(segment_reader, &mut |docs| {
            for &doc in docs {
                filter.insert(doc);
            }
        })?;
        if filter.len() == 0 {
            return Ok(Vec::new());
        }
        let alive = segment_reader.alive_bitset();

        let stride = self.column.dim() * T::SIZE_BYTES;
        let centroid_bytes = self.column.centroid_bytes();
        let num_centroids = centroid_bytes.len() / stride;
        if num_centroids == 0 {
            return Ok(Vec::new());
        }

        // Rank centroids descending by similarity. Extracted into a
        // `#[inline(never)]` method so this phase shows as its own
        // flamegraph frame (carrying its own `score_doc_bytes` cost).
        let ranked = self.rank_centroids(centroid_bytes, stride, num_centroids);

        let best = ranked[0].0;
        let threshold = adaptive_threshold(self.query.metric(), best, self.adaptive.epsilon);
        // Resolve the candidate floor at the call site so a default
        // `min_candidates = 0` still gives a sane
        // `CANDIDATE_OVERFETCH_MULTIPLIER * top_n` floor. Critical for
        // selective filters where a single near cluster yields few
        // survivors — without the floor the loop trips the threshold
        // gate immediately and returns < K results.
        let min_candidates = self
            .adaptive
            .min_candidates
            .max(CANDIDATE_OVERFETCH_MULTIPLIER * top_n);
        let (min_probe_count, max_probe_count) =
            self.adaptive.resolved_probe_counts(num_centroids)?;

        // Adaptive probe loop, extracted into a `#[inline(never)]`
        // method so this phase shows as its own flamegraph frame
        // (carrying its own `score_doc_bytes` cost), distinct from the
        // centroid-ranking frame above.
        let topn = self.scan_clusters(
            ranked,
            threshold,
            min_candidates,
            min_probe_count,
            max_probe_count,
            stride,
            &filter,
            alive,
            top_n,
            max_doc,
            stats,
        )?;

        // Drain best-first, tag with our segment_ord. The collector's
        // `merge_fruits` flattens across segments, sorts descending,
        // and applies offset/limit.
        let segment_ord = self.segment_ord;
        Ok(topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect())
    }

    /// Phase 1: rank centroids descending by similarity. Full scan over
    /// `num_centroids` is the dominant fixed cost; inherent to flat-centroid
    /// IVF, unrelated to the storage layout. `#[inline(never)]` so it forms
    /// its own flamegraph frame carrying its `score_doc_bytes` cost.
    #[inline(never)]
    fn rank_centroids(
        &self,
        centroid_bytes: &[u8],
        stride: usize,
        num_centroids: usize,
    ) -> Vec<(f32, usize)> {
        let mut ranked: Vec<(f32, usize)> = (0..num_centroids)
            .map(|c| {
                let cb = &centroid_bytes[c * stride..(c + 1) * stride];
                (self.query.score_doc_bytes(cb), c)
            })
            .collect();
        ranked.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        ranked
    }

    /// Phase 2: adaptive probe loop. Cluster-order arrival of survivors
    /// forbids the ascending-D shortcut in `push`; use `push_unordered`. The
    /// filter check is cheap (constant-time bitset lookup) so we do it before
    /// the more expensive alive check + similarity score.
    ///
    /// Note on `NaturalComparator` (vs the `TopNComputer::new` default):
    /// vector similarity is "higher = better", so we want top-N *largest*
    /// scores in descending order. The default `new()` wires
    /// `ReverseComparator`, which keeps top-N *smallest* in ascending order —
    /// correct for ascending-distance metrics but inverted for our convention.
    ///
    /// `#[inline(never)]` so it forms its own flamegraph frame carrying its
    /// `score_doc_bytes` cost, distinct from `rank_centroids`.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn scan_clusters(
        &self,
        ranked: Vec<(f32, usize)>,
        threshold: f32,
        min_candidates: usize,
        min_probe_count: usize,
        max_probe_count: usize,
        stride: usize,
        filter: &BitSet,
        alive: Option<&AliveBitSet>,
        top_n: usize,
        max_doc: DocId,
        mut stats: Option<&mut ProbeStats>,
    ) -> crate::Result<TopNComputer<Score, DocId, NaturalComparator>> {
        let mut topn = TopNComputer::<Score, DocId, NaturalComparator>::new_with_comparator(
            top_n,
            NaturalComparator,
        );
        let mut candidates = 0usize;
        // Phase 2 (boundary replication): a single doc can be physically
        // present in several clusters, so probing more than one of them would
        // score the same doc repeatedly. Track scored docs and skip repeats —
        // not just for correct counts, but because a duplicate pushed into the
        // top-K heap can evict a legitimately-distinct result (a post-pass over
        // the final heap can't recover that). Costs one doc-id bitset, like the
        // filter; a no-replica index simply never hits a repeat.
        let mut seen = BitSet::with_max_value(max_doc);

        for (probe_count, (centroid_score, cluster)) in ranked.into_iter().enumerate() {
            if probe_count >= max_probe_count {
                break;
            }
            if centroid_score < threshold
                && candidates >= min_candidates
                && probe_count >= min_probe_count
            {
                break;
            }

            // Record the probe before doing any work, so even an empty
            // cluster (no doc-ids slice) counts as "probed" — that's
            // the right unit for the efficiency assertions.
            if let Some(s) = stats.as_deref_mut() {
                s.probed_clusters.push(cluster);
            }

            let Some(doc_ids) = self.column.cluster_doc_ids(cluster)? else {
                continue;
            };
            let cluster_vecs = self.column.cluster_vector_bytes(cluster)?;
            let cluster_vec_slice = cluster_vecs.as_slice();

            for (local_i, &doc) in doc_ids.iter().enumerate() {
                if !filter.contains(doc) {
                    continue;
                }
                if let Some(bs) = alive {
                    if !bs.is_alive(doc) {
                        continue;
                    }
                }
                if seen.contains(doc) {
                    continue; // already scored from an earlier probed cluster (a replica)
                }
                seen.insert(doc);
                let vbytes = &cluster_vec_slice[local_i * stride..(local_i + 1) * stride];
                let score = self.query.score_doc_bytes(vbytes);
                topn.push_unordered(score, doc);
                candidates += 1;
                if let Some(s) = stats.as_deref_mut() {
                    s.candidates_scored += 1;
                }
            }
        }

        Ok(topn)
    }
}

fn lookup_metric(schema: &Schema, field: Field) -> crate::Result<Metric> {
    let entry = schema.get_field_entry(field);
    match entry.field_type() {
        FieldType::Vector(opts) => Ok(opts.metric()),
        other => Err(TantivyError::SchemaError(format!(
            "field {:?} is not a vector field (got {:?})",
            entry.name(),
            other.value_type(),
        ))),
    }
}

/// SPANN-style stopping threshold: how far below the best centroid
/// score the per-cluster similarity may drop before the adaptive
/// probe loop is allowed to terminate (given the other floors).
///
/// Per-metric because the relationship between "similarity score" and
/// "distance one wants to widen by `(1 + epsilon)`" differs:
///
/// - **L2:** `similarity = -d²`. Widening distance by `(1 + eps)` squares to `(1 + eps)²` on the d²
///   side, so the threshold is `best * (1 + eps)²` (more negative ⇒ more permissive).
/// - **Cosine:** `distance = 1 - similarity`. Widening that distance by `(1 + eps)` gives
///   `threshold = 1 - (1 - best) * (1 + eps)`.
/// - **Dot:** has no bounded distance interpretation (raw dot isn't a metric — no triangle
///   inequality). The threshold here is a pragmatic linear widening of the score floor: `best -
///   epsilon * best.abs()`. NOTE: with unnormalized dot, the IVF locality assumption itself is
///   heuristic — "query near a centroid ⇒ true nearest neighbors live in that cluster" can fail
///   when a high-magnitude vector in a far cluster outscores nearby ones. That's the clusterer's
///   problem, not the threshold's; this function just controls when probing stops.
fn adaptive_threshold(metric: Metric, best: f32, epsilon: f32) -> f32 {
    match metric {
        Metric::L2 => best * (1.0 + epsilon) * (1.0 + epsilon),
        Metric::Cosine => 1.0 - (1.0 - best) * (1.0 + epsilon),
        Metric::Dot => best - epsilon * best.abs(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_threshold_identity_at_zero_epsilon() {
        // With epsilon = 0 the threshold is exactly `best` for the
        // bounded-distance metrics — no widening, no permissiveness.
        for &best in &[-10.0_f32, -1.0, 0.0, 0.5, 1.0] {
            assert_eq!(adaptive_threshold(Metric::L2, best, 0.0), best);
            assert_eq!(adaptive_threshold(Metric::Cosine, best, 0.0), best);
            assert_eq!(adaptive_threshold(Metric::Dot, best, 0.0), best);
        }
    }

    #[test]
    fn adaptive_threshold_lowers_with_positive_epsilon() {
        // "Higher score = closer" convention; widening means the
        // threshold is *lower* (more permissive) than `best`.
        let eps = 0.1;
        // L2 similarity is `-d²`, so `best` is always ≤ 0. Multiplying
        // a non-positive value by `(1+eps)² > 1` makes it more negative
        // (= lower / more permissive). For best = 0 the threshold is
        // also 0 (zero distance gives zero similarity, nothing to
        // widen).
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
        // L2: best = -10, eps = 0.1 ⇒ -10 * 1.21 = -12.1.
        let l2 = adaptive_threshold(Metric::L2, -10.0, 0.1);
        assert!((l2 - -12.1).abs() < 1e-5, "got {l2}");

        // Cosine: best = 0.8, eps = 0.1 ⇒ 1 - 0.2 * 1.1 = 0.78.
        let cos = adaptive_threshold(Metric::Cosine, 0.8, 0.1);
        assert!((cos - 0.78).abs() < 1e-5, "got {cos}");

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
    // Built on top of `crate::vector::tests::TestVectorIndex` (Ming's
    // shared fixture) where the geometry fits — the 100-doc grid +
    // selectivity-based labels covers oracle / filter / delete /
    // overflow / zero-K. The handful of tests that need crafted point
    // geometry (the trap case + the result-level candidate-floor
    // demonstration) build a tiny IVF index inline via `build_inline_ivf`
    // and an `InlineClusterer` that's compatible with the batched
    // IvfClusterer trait.
    // ============================================================

    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
    use crate::vector::meta::VectorStorageFormat;
    use crate::vector::tests::TestVectorIndex;
    use crate::vector::{
        Assignment, IvfCentroids, IvfClusterer, IvfMatrix, IvfMergeSettings, IvfVectors,
        VectorColumn,
        VectorColumnReader, VectorDType, VectorOptions, VectorReader,
    };
    use crate::{Index, IndexWriter, TantivyDocument};

    const FIXTURE_NUM_DOCS: usize = 100;
    /// Number of centroids the shared fixture uses by default (the
    /// 3×3 `grid2d::centroids()` grid). Used by tests that need an
    /// "exhaustive" probe ceiling.
    const DEFAULT_NUM_CENTROIDS: usize = 9;

    /// Wide-epsilon + 100% fanout params: every probe gate
    /// stays open, so the IVF backend visits every cluster. Used by
    /// oracle-equality tests where any kind of pruning would make the
    /// equality check fail.
    fn exhaustive_params(_num_centroids: usize) -> AdaptiveProbeParams {
        AdaptiveProbeParams {
            epsilon: 1e6,
            min_candidates: 0,
            min_probe_fanout: 1.0,
            max_probe_fanout: 1.0,
        }
    }

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

    /// Probe-stat helper: run `IvfBackend::top_n_instrumented` against
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
        match backend {
            VectorBackend::Ivf(b) => b.top_n_instrumented(weight.as_ref(), segment_reader, k),
            VectorBackend::Flat(_) => panic!("expected IVF backend"),
        }
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
                // Crafted-geometry tests assert exact cluster membership;
                // keep balancing disabled so the inline centroids are used
                // verbatim.
                max_posting_len: usize::MAX,
                min_posting_len: 0,
                max_replicas_per_vector: 0,
                max_replicas_per_cluster: 0,
                replica_epsilon: 10.0,
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
        ) -> crate::Result<Vec<Assignment>> {
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
                    Assignment::primary_only(best)
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

    // ---- Merge-time cluster balancing (Phase 1) ----

    /// Clusterer that drives the merge-time rebalance path. The first
    /// `train()` is the top-level clustering and returns the configured
    /// (deliberately imbalanced) centroids; every later `train()` is a split
    /// sub-clustering and seeds sub-centroids by striding through the
    /// oversized cluster's members, so splits genuinely partition it.
    /// Assignment is real nearest-centroid (L2). The size bounds are
    /// configurable so a single test can exercise both the split (oversized)
    /// and merge (undersized) branches.
    struct BalancedTestClusterer {
        centroids: Vec<[f32; 2]>,
        max_posting_len: usize,
        min_posting_len: usize,
        top_level_done: std::sync::atomic::AtomicBool,
    }

    impl IvfClusterer for BalancedTestClusterer {
        fn centroid_ratio(&self) -> f32 {
            1.0
        }
        fn training_samples_per_centroid(&self) -> usize {
            2
        }
        fn merge_settings(&self, total_target_docs: usize) -> crate::Result<IvfMergeSettings> {
            Ok(IvfMergeSettings {
                num_centroids: self.centroids.len().min(total_target_docs),
                training_samples_per_centroid: self.training_samples_per_centroid(),
                assign_batch_size: self.assign_batch_size(),
                max_posting_len: self.max_posting_len,
                min_posting_len: self.min_posting_len,
                max_replicas_per_vector: 0,
                max_replicas_per_cluster: 0,
                replica_epsilon: 10.0,
            })
        }
        fn train(
            &self,
            options: &VectorOptions,
            vectors: IvfVectors<'_>,
            num_centroids: usize,
        ) -> crate::Result<IvfCentroids> {
            use std::sync::atomic::Ordering;
            assert_eq!(options.dim(), 2);
            let IvfVectors::F32(batch) = vectors;
            if !self.top_level_done.swap(true, Ordering::Relaxed) {
                return Ok(IvfCentroids::F32(IvfMatrix {
                    values: self
                        .centroids
                        .iter()
                        .take(num_centroids)
                        .flat_map(|c| c.iter().copied())
                        .collect(),
                    rows: num_centroids,
                    dims: 2,
                }));
            }
            let rows = batch.matrix.rows;
            assert!(
                rows >= num_centroids,
                "split needs >= num_centroids members"
            );
            let mut values = Vec::with_capacity(num_centroids * 2);
            for j in 0..num_centroids {
                let row = (j * rows) / num_centroids;
                values.extend_from_slice(&batch.matrix.values[row * 2..row * 2 + 2]);
            }
            Ok(IvfCentroids::F32(IvfMatrix {
                values,
                rows: num_centroids,
                dims: 2,
            }))
        }
        fn assign(
            &self,
            options: &VectorOptions,
            vectors: IvfVectors<'_>,
            centroids: &IvfCentroids,
        ) -> crate::Result<Vec<Assignment>> {
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
                            best_d2 = d2;
                            best = i as u32;
                        }
                    }
                    Assignment::primary_only(best)
                })
                .collect())
        }
    }

    /// End-to-end Step-6 check: a deliberately imbalanced top-level
    /// clustering (one ~60-member fat cluster + one 3-member outlier
    /// cluster) is balanced at merge time. The fat cluster is split below
    /// the cap and the outlier cluster is dissolved above the floor, with
    /// every vector preserved and still retrievable.
    #[test]
    fn ivf_merge_rebalances_cluster_sizes() -> crate::Result<()> {
        use std::sync::atomic::AtomicBool;

        const MAX_POSTING: usize = 20;
        const MIN_POSTING: usize = 5;
        const NUM_DENSE: usize = 60;
        const NUM_FAR: usize = 3;
        let total = NUM_DENSE + NUM_FAR;

        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for i in 0..NUM_DENSE {
            docs.push((format!("dense-{i}"), [i as f32, 0.0]));
        }
        for i in 0..NUM_FAR {
            docs.push((format!("far-{i}"), [1000.0, 1000.0 + i as f32]));
        }

        let clusterer = Arc::new(BalancedTestClusterer {
            centroids: vec![[0.0, 0.0], [1000.0, 1000.0]],
            max_posting_len: MAX_POSTING,
            min_posting_len: MIN_POSTING,
            top_level_done: AtomicBool::new(false),
        });

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
            .ivf_clusterer(clusterer)
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Two commits so the merge has >= 2 source segments.
        let mid = docs.len() / 2;
        for chunk in [&docs[..mid], &docs[mid..]] {
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

        let searcher = index.reader()?.searcher();
        assert_eq!(
            searcher.segment_readers().len(),
            1,
            "expected a single merged segment"
        );
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = VectorReader::open(segment_reader)?;
        let info = vec_reader.info(embed_field)?.expect("vector info");
        assert_eq!(info.format, VectorStorageFormat::Ivf);
        let stats = info.cluster_stats.expect("ivf cluster stats");

        // No vectors lost across the rebalance.
        assert_eq!(info.num_vectors, total);
        // Max cap enforced (the dense cloud was a single ~60-member cluster).
        assert!(
            stats.max_cluster_size <= MAX_POSTING,
            "max cluster size {} exceeds cap {MAX_POSTING}",
            stats.max_cluster_size
        );
        // Min floor enforced (the 3-member outlier cluster was dissolved, not
        // left as an undersized posting).
        assert!(
            stats.min_cluster_size >= MIN_POSTING,
            "min cluster size {} below floor {MIN_POSTING}",
            stats.min_cluster_size
        );
        // Dissolving undersized clusters also clears empties.
        assert_eq!(stats.empty_clusters, 0);
        // Splitting the fat cluster added centroids beyond the original 2.
        assert!(
            info.num_centroids.unwrap() >= 2,
            "expected splits to add centroids, got {:?}",
            info.num_centroids
        );

        // Every document is still retrievable from its posting.
        let column = vec_reader.open_column(embed_field)?;
        for doc_id in 0..total as u32 {
            assert!(
                column.contains(doc_id),
                "doc {doc_id} missing from IVF index after rebalance"
            );
        }

        Ok(())
    }

    // ---- Phase 2: boundary replication ----

    /// Test clusterer that replicates near-equidistant boundary vectors. The
    /// primary is the nearest centroid; any other centroid within
    /// `epsilon × dist(nearest)` becomes a replica candidate (capped at
    /// `max_replicas_per_vector`). Balancing is disabled so the clusters stay
    /// exactly as trained, isolating the replication behavior.
    struct ReplicatingClusterer {
        centroids: Vec<[f32; 2]>,
        epsilon: f32,
        max_replicas_per_vector: usize,
    }

    impl IvfClusterer for ReplicatingClusterer {
        fn centroid_ratio(&self) -> f32 {
            1.0
        }
        fn training_samples_per_centroid(&self) -> usize {
            2
        }
        fn merge_settings(&self, total_target_docs: usize) -> crate::Result<IvfMergeSettings> {
            Ok(IvfMergeSettings {
                num_centroids: self.centroids.len().min(total_target_docs),
                training_samples_per_centroid: self.training_samples_per_centroid(),
                assign_batch_size: self.assign_batch_size(),
                max_posting_len: usize::MAX, // no split — keep the trained clusters
                min_posting_len: 0,          // no merge
                max_replicas_per_vector: self.max_replicas_per_vector,
                max_replicas_per_cluster: 10,
                replica_epsilon: self.epsilon,
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
        ) -> crate::Result<Vec<Assignment>> {
            assert_eq!(options.dim(), 2);
            let IvfVectors::F32(vectors) = vectors;
            let IvfCentroids::F32(centroids) = centroids;
            Ok(vectors
                .matrix
                .values
                .chunks_exact(2)
                .map(|v| {
                    let mut dists: Vec<(f32, u32)> = centroids
                        .values
                        .chunks_exact(2)
                        .enumerate()
                        .map(|(i, c)| {
                            let dx = v[0] - c[0];
                            let dy = v[1] - c[1];
                            (dx * dx + dy * dy, i as u32)
                        })
                        .collect();
                    dists.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let nearest = dists[0].0.sqrt();
                    let mut assignment = Assignment::primary_only(dists[0].1);
                    for &(d2, c) in dists[1..].iter() {
                        if assignment.replicas.len() >= self.max_replicas_per_vector {
                            break;
                        }
                        if self.max_replicas_per_vector > 0 && d2.sqrt() <= self.epsilon * nearest {
                            assignment.replicas.push(c);
                        }
                    }
                    assignment
                })
                .collect())
        }
    }

    /// A boundary vector equidistant from two centroids must be physically
    /// stored in BOTH clusters (one replica), yet a probe of both clusters
    /// must return it exactly once (Step 7 dedup).
    #[test]
    fn ivf_replication_places_boundary_vector_in_two_clusters() -> crate::Result<()> {
        // Interior docs near A=[0,0] and B=[10,0], plus one boundary doc at the
        // midpoint [5,0] equidistant from both.
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for i in 0..5 {
            docs.push((format!("a-{i}"), [i as f32, 0.0]));
        }
        for i in 0..5 {
            docs.push((format!("b-{i}"), [(10 - i) as f32, 0.0]));
        }
        docs.push(("boundary".to_string(), [5.0, 0.0]));
        let total = docs.len();

        // ε = 1.2: the midpoint [5,0] (ratio 1.0) replicates; the next-closest
        // interior docs [4,0]/[6,0] (ratio 1.5) do not.
        let clusterer = Arc::new(ReplicatingClusterer {
            centroids: vec![[0.0, 0.0], [10.0, 0.0]],
            epsilon: 1.2,
            max_replicas_per_vector: 1,
        });

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
            .ivf_clusterer(clusterer)
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let mid = docs.len() / 2;
        for chunk in [&docs[..mid], &docs[mid..]] {
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

        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = VectorReader::open(segment_reader)?;
        let info = vec_reader.info(embed_field)?.expect("vector info");
        assert_eq!(info.format, VectorStorageFormat::Ivf);

        // Exactly one replica appended: stored entries = docs + 1.
        assert_eq!(
            info.num_vectors,
            total + 1,
            "expected one boundary replica beyond the {total} primaries"
        );
        assert_eq!(info.num_centroids, Some(2), "no rebalance: two clusters");

        // The boundary doc is the only doc-id present in BOTH clusters.
        let column = match vec_reader.open_column(embed_field)? {
            VectorColumn::Ivf(column) => column,
            VectorColumn::Flat(_) => panic!("expected IVF column"),
        };
        let cluster0: std::collections::HashSet<DocId> = column
            .cluster_doc_ids(0)?
            .expect("cluster 0")
            .iter()
            .copied()
            .collect();
        let in_both: Vec<DocId> = column
            .cluster_doc_ids(1)?
            .expect("cluster 1")
            .iter()
            .copied()
            .filter(|d| cluster0.contains(d))
            .collect();
        assert_eq!(
            in_both.len(),
            1,
            "exactly one (boundary) doc should be replicated into both clusters, got {in_both:?}"
        );

        // Probing BOTH clusters must return each doc exactly once — the
        // replicated boundary doc must not appear twice (Step 7 dedup).
        let params = exhaustive_params(2);
        let hits = search(
            &index,
            embed_field,
            &AllQuery,
            vec![5.0_f32, 0.0],
            total,
            params,
        )?;
        let mut doc_ids: Vec<u32> = hits.iter().map(|(_, addr)| addr.doc_id).collect();
        let unique: std::collections::HashSet<u32> = doc_ids.iter().copied().collect();
        assert_eq!(
            doc_ids.len(),
            unique.len(),
            "duplicate doc-ids in top-K from a replicated doc: {doc_ids:?}"
        );
        doc_ids.sort_unstable();
        assert_eq!(doc_ids.len(), total, "all primaries retrievable, deduped");

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
    /// Adaptive probing finds it; 50% max fanout must miss. Setup
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
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
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

        // Behavioral check 1: 50% max fanout misses the trap.
        let one_probe = AdaptiveProbeParams {
            epsilon: 1e6,
            min_candidates: 0,
            min_probe_fanout: 0.5,
            max_probe_fanout: 0.5,
        };
        let hits1 = search(&index, embed_field, &AllQuery, query.to_vec(), 1, one_probe)?;
        assert_eq!(hits1.len(), 1);
        assert_ne!(
            stored_label_at(&index, label_field, hits1[0].1)?,
            "trap_b",
            "50% max fanout should miss the trap (probes only cluster A)",
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
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let query = [1.0_f32, 1.0];
        let top_k = 1;

        // Open segment 0's IVF column for the geometry assertions.
        // After `build_inline_ivf`'s merge, all docs sit in segment 0.
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = VectorReader::open(segment_reader)?;
        let column = match vec_reader.open_column(embed_field)? {
            VectorColumn::Ivf(c) => c,
            VectorColumn::Flat(_) => panic!("expected IVF segment for this test"),
        };
        let nearest = |bytes: &[u8]| {
            let p = [
                f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            ];
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
        };

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
        let oracle_bytes = column
            .vector_bytes_at(oracle_addr.doc_id)
            .expect("oracle vector bytes");
        assert_eq!(
            nearest(oracle_bytes),
            1,
            "oracle top-1 must live in cluster B — the far cluster the floor has to reach",
        );

        // Setup assertion (ii): a_only still lands in cluster A. If
        // [0,-10] ever drifts across the bisector x+y=10 (it won't with
        // these coords, but coordinates evolve), the premise "the near
        // cluster has too few survivors" stops holding — the test
        // would no longer exercise the floor.
        let cluster_a_docs = column
            .cluster_doc_ids(0)?
            .map(<[_]>::to_vec)
            .unwrap_or_default();
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
            min_probe_fanout: 0.5,
            max_probe_fanout: 1.0,
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
    /// `IvfBackend::top_n_inner`.
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
        Ok(())
    }

    // ============================================================
    // Adaptive-probing parameter contracts.
    //
    // The stop condition couples all four adaptive knobs. Each test
    // below holds three of them permissive so the fourth becomes the
    // binding constraint, then asserts the contract (an inequality
    // implied by the knob's definition) — never an exact emergent
    // count.
    // ============================================================

    /// 2 / 9 max fanout ⇒ at most 2 probes per segment, regardless of
    /// how generous the other gates are.
    #[test]
    fn probe_stats_max_fanout_ceiling() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = AdaptiveProbeParams {
            epsilon: 1e6,
            min_candidates: 0,
            min_probe_fanout: 0.0,
            max_probe_fanout: 2.0 / DEFAULT_NUM_CENTROIDS as f32,
        };
        let (_, stats) = run_top_n_instrumented(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            3,
            params,
        )?;
        assert!(
            stats.probed_clusters.len() <= 2,
            "2 / 9 max fanout ⇒ ≤ 2 probed, got {} ({:?})",
            stats.probed_clusters.len(),
            stats.probed_clusters,
        );
        Ok(())
    }

    /// 5 / 9 min fanout keeps the loop going even after the candidate
    /// floor and threshold gates both want to terminate. The shared
    /// fixture's 9-centroid grid yields ~2 docs per cluster per IVF
    /// segment; with `top_k = 1` the floor (4) is met after 2 probes,
    /// but min fanout forces 5.
    #[test]
    fn probe_stats_min_fanout_floor() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = AdaptiveProbeParams {
            epsilon: 0.0,
            min_candidates: 0,
            min_probe_fanout: 5.0 / DEFAULT_NUM_CENTROIDS as f32,
            max_probe_fanout: 1.0,
        };
        let (_, stats) = run_top_n_instrumented(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            1,
            params,
        )?;
        assert!(
            stats.probed_clusters.len() >= 5,
            "5 / 9 min fanout ⇒ ≥ 5 probed, got {} ({:?})",
            stats.probed_clusters.len(),
            stats.probed_clusters,
        );
        Ok(())
    }

    /// Candidate floor: regardless of how stingy the threshold gate
    /// is, the loop scores at least `min(total_docs, resolved_floor)`
    /// docs. With default params and the shared fixture's ~20 docs per
    /// segment + top_k = 4 (floor 16), `candidates_scored >= 16` or
    /// `>= segment_doc_count` if that's smaller.
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

        let (_, stats) = run_top_n_instrumented(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            top_k,
            AdaptiveProbeParams::default(),
        )?;
        assert!(
            stats.candidates_scored >= expected_min,
            "candidate floor (resolved {resolved_floor}, segment {segment_doc_count}) ⇒ ≥ \
             {expected_min} candidates scored; got {}",
            stats.candidates_scored,
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
            let vec_reader = VectorReader::open(segment_reader)?;
            let column = vec_reader.open_column(vec_field)?;
            let alive = segment_reader.alive_bitset();
            for doc in 0..segment_reader.max_doc() {
                if let Some(alive) = alive {
                    if !alive.is_alive(doc) {
                        continue;
                    }
                }
                if let Some(bytes) = column.vector_bytes_at(doc) {
                    scored.push((
                        query.score_doc_bytes(bytes),
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
        let mut best = 0;
        let mut best_d2 = f32::INFINITY;
        for row in 0..3 {
            for col in 0..3 {
                let cx = (col as f32) * 3.0;
                let cy = (row as f32) * 3.0;
                let dx = query[0] - cx;
                let dy = query[1] - cy;
                let d2 = dx * dx + dy * dy;
                if d2 < best_d2 {
                    best = row * 3 + col;
                    best_d2 = d2;
                }
            }
        }
        best
    }
}
