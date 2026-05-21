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
use super::reader::{VectorColumn, VectorColumnReader, VectorReader};
use crate::collector::sort_key::NaturalComparator;
use crate::collector::TopNComputer;
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
    metric: Metric,
    query: Arc<Vec<T>>,
    segment_ord: SegmentOrdinal,
}

pub struct IvfBackend<T: VectorElement> {
    column: IvfVectorColumn,
    metric: Metric,
    query: Arc<Vec<T>>,
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

        match vec_reader.open_column(field)? {
            VectorColumn::Ivf(column) => Ok(Self::Ivf(IvfBackend {
                column,
                metric,
                query,
                adaptive,
                segment_ord,
            })),
            VectorColumn::Flat(column) => Ok(Self::Flat(FlatBackend {
                column,
                metric,
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
                    let score = self.metric.similarity_bytes(&self.query[..], bytes);
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
        mut stats: Option<&mut ProbeStats>,
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

        // Rank centroids descending by similarity. Full scan over
        // `num_centroids` is the dominant fixed cost; inherent to
        // flat-centroid IVF, unrelated to the storage layout.
        let mut ranked: Vec<(f32, usize)> = (0..num_centroids)
            .map(|c| {
                let cb = &centroid_bytes[c * stride..(c + 1) * stride];
                (self.metric.similarity_bytes(&self.query[..], cb), c)
            })
            .collect();
        ranked.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let best = ranked[0].0;
        let threshold = adaptive_threshold(self.metric, best, self.adaptive.epsilon);
        // Resolve the candidate floor at the call site so a default
        // `min_candidates = 0` still gives a sane `4 × top_n` floor.
        // Critical for selective filters where a single near cluster
        // yields few survivors — without the floor the loop trips the
        // threshold gate immediately and returns < K results.
        let min_candidates = self.adaptive.min_candidates.max(4 * top_n);

        // Adaptive probe loop. Cluster-order arrival of survivors
        // forbids the ascending-D shortcut in `push`; use
        // `push_unordered`. The filter check is cheap (constant-time
        // bitset lookup) so we do it before the more expensive alive
        // check + similarity score.
        //
        // Note on `NaturalComparator` (vs the `TopNComputer::new`
        // default): vector similarity is "higher = better", so we
        // want top-N *largest* scores in descending order. The
        // default `new()` wires `ReverseComparator`, which keeps
        // top-N *smallest* in ascending order — correct for
        // ascending-distance metrics but inverted for our convention.
        let mut topn = TopNComputer::<Score, DocId, NaturalComparator>::new_with_comparator(
            top_n,
            NaturalComparator,
        );
        let mut candidates = 0usize;

        for (probe_count, (centroid_score, cluster)) in ranked.into_iter().enumerate() {
            if probe_count >= self.adaptive.max_nprobe {
                break;
            }
            if centroid_score < threshold
                && candidates >= min_candidates
                && probe_count >= self.adaptive.min_nprobe
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
                let vbytes = &cluster_vec_slice[local_i * stride..(local_i + 1) * stride];
                let score = self.metric.similarity_bytes(&self.query[..], vbytes);
                topn.push_unordered(score, doc);
                candidates += 1;
                if let Some(s) = stats.as_deref_mut() {
                    s.candidates_scored += 1;
                }
            }
        }

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
    // ============================================================

    use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
    use crate::schema::{IndexRecordOption, Term};
    use crate::vector::ivf::test_harness::{
        brute_force_oracle, build_ivf_segment, decode_vec, open_ivf_column, IvfFixture,
    };

    /// Run `IvfBackend::top_n` against the fixture's single segment with
    /// the given filter query (`None` ⇒ `AllQuery`) and adaptive params.
    /// Returns the per-segment hits in the backend's native order
    /// (best-first / ascending-doc tiebreak).
    fn run_top_n(
        fixture: &IvfFixture,
        filter: Option<&dyn Query>,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let searcher = fixture.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let scoring = EnableScoring::disabled_from_searcher(&searcher);
        let weight: Box<dyn Weight> = match filter {
            Some(q) => q.weight(scoring)?,
            None => AllQuery.weight(scoring)?,
        };
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            fixture.vec_field,
            Arc::new(query),
            params,
        )?;
        backend.top_n(weight.as_ref(), segment_reader, k)
    }

    fn exhaustive_params(num_centroids: usize) -> AdaptiveProbeParams {
        // Wide epsilon + nprobe ceiling = num_centroids forces a full
        // exhaustive scan. Used to assert correctness against the
        // brute-force oracle.
        AdaptiveProbeParams {
            epsilon: 1e6,
            min_candidates: 0,
            min_nprobe: num_centroids,
            max_nprobe: num_centroids,
        }
    }

    fn assert_hits_match_oracle(
        hits: &[(Score, DocAddress)],
        oracle: &[(Score, DocId)],
        label: &str,
    ) {
        assert_eq!(hits.len(), oracle.len(), "{label}: length mismatch");
        for (i, ((score, addr), (oscore, odoc))) in hits.iter().zip(oracle.iter()).enumerate() {
            assert_eq!(addr.doc_id, *odoc, "{label}: doc mismatch at rank {i}");
            assert!(
                (*score - *oscore).abs() < 1e-5,
                "{label}: score mismatch at rank {i}: {score} vs {oscore}",
            );
        }
    }

    // 1+2. Exhaustive probing matches the brute-force oracle exactly
    // for L2 across several queries and K. The oracle is ground truth;
    // if this fails, the bug is in ranking/scoring/drain.
    #[test]
    fn ivf_top_n_brute_force_oracle_l2() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let docs = vec![
            ("a", vec![1.0_f32, 0.2]),
            ("a", vec![0.3, 1.1]),
            ("a", vec![2.2, 2.7]),
            ("b", vec![9.4, 9.1]),
            ("b", vec![11.3, 10.6]),
            ("b", vec![8.2, 12.1]),
        ];
        let fixture = build_ivf_segment(2, Metric::L2, centroids, &docs)?;
        let column = open_ivf_column(&fixture)?;
        let params = exhaustive_params(2);

        for query in [
            vec![0.5_f32, 0.5],
            vec![9.5, 9.5],
            vec![5.0, 0.0],
            vec![3.7, 11.2],
        ] {
            for k in [1usize, 3, 6, 10] {
                let oracle = brute_force_oracle(&column, Metric::L2, &query, k, None, None);
                let hits = run_top_n(&fixture, None, query.clone(), k, params.clone())?;
                assert_hits_match_oracle(
                    &hits,
                    &oracle,
                    &format!("L2 exhaustive query={query:?} k={k}"),
                );
            }
        }
        Ok(())
    }

    // 8. Same exhaustive correctness for Cosine, confirming the metric
    // threads through generically.
    #[test]
    fn ivf_top_n_brute_force_oracle_cosine() -> crate::Result<()> {
        let centroids = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
        let docs = vec![
            ("a", vec![0.95_f32, 0.1]),
            ("a", vec![0.8, 0.15]),
            ("a", vec![1.0, -0.05]),
            ("b", vec![0.05, 0.97]),
            ("b", vec![0.12, 0.85]),
            ("b", vec![-0.04, 1.0]),
        ];
        let fixture = build_ivf_segment(2, Metric::Cosine, centroids, &docs)?;
        let column = open_ivf_column(&fixture)?;
        let params = exhaustive_params(2);
        for query in [vec![1.0_f32, 0.0], vec![0.0, 1.0], vec![0.7, 0.3]] {
            for k in [1usize, 3, 6] {
                let oracle = brute_force_oracle(&column, Metric::Cosine, &query, k, None, None);
                let hits = run_top_n(&fixture, None, query.clone(), k, params.clone())?;
                assert_hits_match_oracle(
                    &hits,
                    &oracle,
                    &format!("Cosine exhaustive query={query:?} k={k}"),
                );
            }
        }
        Ok(())
    }

    // Exhaustive-probe correctness for the Dot metric. Mirrors the
    // L2 and Cosine cases but uses vectors with varied magnitudes so
    // the ranking is genuinely magnitude-sensitive (not accidentally
    // cosine-like behavior on near-unit vectors): e.g. query
    // [1, 0] ranks doc [3, 0.1] (dot = 3) above doc [1, 0] (dot = 1),
    // which would tie under direction-only similarity.
    //
    // EXHAUSTIVE-PROBE ONLY. We deliberately don't add an adaptive
    // (default-params) Dot test. Dot isn't a metric — no triangle
    // inequality — so the IVF cluster-locality assumption ("query
    // near a centroid ⇒ true nearest neighbors live in that cluster")
    // is heuristic for Dot and can break when a high-magnitude vector
    // in a far cluster outscores nearby ones. Adaptive Dot recall is
    // a benchmark question, deferred. This test confirms only that
    // `Metric::Dot` threads through the backend's full top_n loop and
    // produces scores that match brute force when every cluster is
    // visited.
    #[test]
    fn ivf_top_n_brute_force_oracle_dot() -> crate::Result<()> {
        // Centroids on opposite sides of the x-axis; ParametricClusterer
        // assigns by L2-nearest, so the partition is sign(x).
        let centroids = vec![vec![1.0_f32, 0.0], vec![-1.0, 0.0]];
        let docs = vec![
            ("a", vec![1.0_f32, 0.0]),
            ("a", vec![3.0, 0.1]),
            ("a", vec![0.5, 0.5]),
            ("b", vec![-1.0, 0.0]),
            ("b", vec![-3.0, 0.0]),
            ("b", vec![-0.5, 0.5]),
        ];
        let fixture = build_ivf_segment(2, Metric::Dot, centroids, &docs)?;
        let column = open_ivf_column(&fixture)?;
        let params = exhaustive_params(2);
        for query in [vec![1.0_f32, 0.0], vec![2.0, 0.0], vec![0.5, -0.5]] {
            for k in [1usize, 3, 6] {
                let oracle = brute_force_oracle(&column, Metric::Dot, &query, k, None, None);
                let hits = run_top_n(&fixture, None, query.clone(), k, params.clone())?;
                assert_hits_match_oracle(
                    &hits,
                    &oracle,
                    &format!("Dot exhaustive query={query:?} k={k}"),
                );
            }
        }
        Ok(())
    }

    // 3. The trap: query closest to centroid A, true NN in cluster B.
    // Adaptive probing finds it; max_nprobe=1 must miss. The setup
    // assertions confirm we actually constructed the trap geometry
    // before exercising the adaptive boundary — without them a
    // slightly-off geometry could trivialize the test.
    #[test]
    fn ivf_top_n_trap_case() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        // A-side docs deliberately far from the [1,1] query; B-side
        // trap doc [5, 5.01] sits just over the perpendicular bisector
        // (x+y=10 boundary) so it lands in cluster 1 yet is much
        // closer to the query than any A-side doc.
        let docs = vec![
            ("far_a", vec![0.0_f32, -10.0]),
            ("far_a", vec![-10.0, 0.0]),
            ("trap_b", vec![5.0, 5.01]),
            ("anchor_b", vec![10.0, 10.0]),
        ];
        let fixture = build_ivf_segment(2, Metric::L2, centroids.clone(), &docs)?;
        let column = open_ivf_column(&fixture)?;
        let query = vec![1.0_f32, 1.0];

        // ---- Setup assertions (per Ruchir's instruction #4) ----
        // (i) Trap doc must be in cluster 1 (B), not cluster 0 (A).
        // Identify it by its known vector signature.
        let cluster_a = column.cluster_doc_ids(0)?.expect("cluster A");
        let cluster_b = column.cluster_doc_ids(1)?.expect("cluster B");
        let is_trap = |doc: DocId| -> bool {
            let v = decode_vec(column.vector_bytes_at(doc).expect("vec"));
            (v[0] - 5.0).abs() < 1e-3 && (v[1] - 5.01).abs() < 1e-3
        };
        let trap_doc = *cluster_b
            .iter()
            .find(|&&d| is_trap(d))
            .expect("trap doc [5, 5.01] must be in cluster B");
        assert!(
            !cluster_a.iter().any(|&d| is_trap(d)),
            "trap doc must not be in cluster A"
        );

        // (ii) Query's nearest centroid must be A (cluster 0).
        let stride = column.dim() * <f32 as VectorElement>::SIZE_BYTES;
        let cb = column.centroid_bytes();
        let score_a = Metric::L2.similarity_bytes(&query, &cb[0..stride]);
        let score_b = Metric::L2.similarity_bytes(&query, &cb[stride..2 * stride]);
        assert!(
            score_a > score_b,
            "query's nearest centroid must be A: score_a={score_a} vs score_b={score_b}",
        );

        // (iii) Trap doc must genuinely be the true top-1 — without
        // this, "miss" and "find" would be indistinguishable.
        let oracle = brute_force_oracle(&column, Metric::L2, &query, 1, None, None);
        assert_eq!(oracle[0].1, trap_doc, "true NN must be the trap doc");

        // ---- Behavioral check 1: max_nprobe=1 misses the trap. ----
        let one_probe = AdaptiveProbeParams {
            epsilon: 1e6,
            min_candidates: 0,
            min_nprobe: 1,
            max_nprobe: 1,
        };
        let hits1 = run_top_n(&fixture, None, query.clone(), 1, one_probe)?;
        assert_eq!(hits1.len(), 1);
        assert_ne!(
            hits1[0].1.doc_id, trap_doc,
            "max_nprobe=1 should miss the trap doc (probes only cluster A)",
        );

        // ---- Behavioral check 2: adaptive probing finds it. ----
        let hits2 = run_top_n(&fixture, None, query, 1, exhaustive_params(2))?;
        assert_eq!(hits2.len(), 1);
        assert_eq!(
            hits2[0].1.doc_id, trap_doc,
            "exhaustive probing should find the trap doc",
        );

        Ok(())
    }

    // 4. Filter selectivity: only docs in the filter set appear, and
    // the result equals the oracle restricted to the filter.
    #[test]
    fn ivf_top_n_filter_selectivity() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let docs = vec![
            ("a", vec![1.0_f32, 0.2]),
            ("a", vec![0.3, 1.1]),
            ("a", vec![2.2, 2.7]),
            ("b", vec![9.4, 9.1]),
            ("b", vec![11.3, 10.6]),
            ("b", vec![8.2, 12.1]),
        ];
        let fixture = build_ivf_segment(2, Metric::L2, centroids, &docs)?;
        let column = open_ivf_column(&fixture)?;
        // Build the filter bitset that matches the "b" category.
        let searcher = fixture.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let max_doc = segment_reader.max_doc();
        let mut filter_set = BitSet::with_max_value(max_doc);
        let b_term = Term::from_field_text(fixture.category_field, "b");
        let filter_query = TermQuery::new(b_term, IndexRecordOption::Basic);
        let weight = filter_query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        weight.for_each_no_score(segment_reader, &mut |docs| {
            for &d in docs {
                filter_set.insert(d);
            }
        })?;

        let query = vec![0.5_f32, 0.5];
        let oracle = brute_force_oracle(&column, Metric::L2, &query, 10, Some(&filter_set), None);
        let hits = run_top_n(
            &fixture,
            Some(&filter_query),
            query,
            10,
            exhaustive_params(2),
        )?;
        assert_hits_match_oracle(&hits, &oracle, "filtered L2");
        // Every hit is in the filter set (sanity beyond oracle equality).
        for (_, addr) in &hits {
            assert!(
                filter_set.contains(addr.doc_id),
                "hit outside filter: {addr:?}"
            );
        }
        Ok(())
    }

    // 5. Empty filter returns empty results, no panic.
    #[test]
    fn ivf_top_n_empty_filter() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let docs = vec![
            ("a", vec![1.0_f32, 0.2]),
            ("a", vec![0.3, 1.1]),
            ("b", vec![9.4, 9.1]),
            ("b", vec![11.3, 10.6]),
        ];
        let fixture = build_ivf_segment(2, Metric::L2, centroids, &docs)?;
        // No "z" docs exist ⇒ TermQuery yields an empty DocSet.
        let z_term = Term::from_field_text(fixture.category_field, "z");
        let empty_filter = TermQuery::new(z_term, IndexRecordOption::Basic);
        let hits = run_top_n(
            &fixture,
            Some(&empty_filter),
            vec![0.0_f32, 0.0],
            5,
            exhaustive_params(2),
        )?;
        assert!(hits.is_empty(), "empty filter must produce no hits");
        Ok(())
    }

    // 6. K > surviving candidates: returns all survivors, sorted, no
    // panic.
    #[test]
    fn ivf_top_n_k_exceeds_candidates() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let docs = vec![
            ("a", vec![1.0_f32, 0.2]),
            ("a", vec![0.3, 1.1]),
            ("b", vec![9.4, 9.1]),
            ("b", vec![11.3, 10.6]),
        ];
        let fixture = build_ivf_segment(2, Metric::L2, centroids, &docs)?;
        let column = open_ivf_column(&fixture)?;
        let query = vec![0.0_f32, 0.0];
        let hits = run_top_n(&fixture, None, query.clone(), 100, exhaustive_params(2))?;
        let oracle = brute_force_oracle(&column, Metric::L2, &query, 100, None, None);
        assert_eq!(hits.len(), 4, "should return all four docs");
        assert_hits_match_oracle(&hits, &oracle, "K > candidates");
        Ok(())
    }

    // 7. Deletes: a doc marked deleted must never appear, even if it
    // would otherwise rank top-K. Confirms the separate alive-check
    // (filter bitmap doesn't carry delete info).
    #[test]
    fn ivf_top_n_respects_deletes() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let docs = vec![
            ("doomed", vec![0.01_f32, 0.01]), // would-be top-1 — gets deleted
            ("survivor_a", vec![1.0, 0.5]),
            ("survivor_b", vec![9.4, 9.1]),
            ("survivor_b", vec![11.3, 10.6]),
        ];
        let fixture = build_ivf_segment(2, Metric::L2, centroids, &docs)?;

        // Apply delete to the "doomed" docs and commit. NoMergePolicy
        // so we don't lose the IVF segment.
        {
            let mut writer: crate::IndexWriter =
                fixture.index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(crate::indexer::NoMergePolicy));
            writer.delete_term(Term::from_field_text(fixture.category_field, "doomed"));
            writer.commit()?;
        }

        // Identify the "doomed" doc id so we can assert it's absent
        // from the results.
        let column = open_ivf_column(&fixture)?;
        let searcher = fixture.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let alive = segment_reader
            .alive_bitset()
            .expect("delete should produce an alive bitset");
        let doomed: Vec<DocId> = (0..segment_reader.max_doc())
            .filter(|&d| !alive.is_alive(d))
            .collect();
        assert!(!doomed.is_empty(), "expected at least one deleted doc");
        // Sanity: the doomed doc is the one near the origin.
        for &d in &doomed {
            let v = decode_vec(column.vector_bytes_at(d).expect("doomed has vec"));
            assert!(
                v[0].abs() < 0.1 && v[1].abs() < 0.1,
                "doomed should be near origin: {v:?}"
            );
        }

        let query = vec![0.0_f32, 0.0];
        let hits = run_top_n(&fixture, None, query, 10, exhaustive_params(2))?;
        for (_, addr) in &hits {
            assert!(
                !doomed.contains(&addr.doc_id),
                "deleted doc {addr:?} surfaced in results",
            );
        }
        assert!(!hits.is_empty(), "should return surviving docs");
        Ok(())
    }

    // 9. min_candidates floor: with a setup where cluster A's
    // closest-to-query has fewer survivors than 4*top_n, the loop must
    // keep probing into cluster B to satisfy the floor even though
    // epsilon=0 makes the threshold trip immediately.
    #[test]
    fn ivf_top_n_min_candidates_floor() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        // Cluster A has one doc near the query; cluster B has the true
        // top-1 by query distance. Query nearest centroid is A.
        let docs = vec![
            ("a_only", vec![3.0_f32, 3.0]),   // A-side
            ("b_close", vec![5.0_f32, 5.01]), // B-side, true NN
            ("b_far", vec![10.0_f32, 10.0]),  // B-side anchor
            ("b_far2", vec![11.0_f32, 9.5]),  // B-side anchor
        ];
        let fixture = build_ivf_segment(2, Metric::L2, centroids, &docs)?;
        let column = open_ivf_column(&fixture)?;
        let query = vec![1.0_f32, 1.0];

        // Setup: verify nearest centroid is A, and the true top-1 is
        // the [5, 5.01] B-side doc (so the floor MUST kick in to find
        // it past the immediate threshold trip).
        let stride = column.dim() * <f32 as VectorElement>::SIZE_BYTES;
        let cb = column.centroid_bytes();
        let score_a = Metric::L2.similarity_bytes(&query, &cb[0..stride]);
        let score_b = Metric::L2.similarity_bytes(&query, &cb[stride..2 * stride]);
        assert!(score_a > score_b, "nearest centroid must be A");

        // Params: epsilon=0 makes the threshold trip the moment we
        // move off centroid A; max_nprobe=2 means we *could* probe B
        // but the threshold + min_nprobe=1 gate alone would stop us
        // after A. Only the candidate floor (4 * top_n = 4) — with
        // only 1 candidate after A — keeps us probing.
        let params = AdaptiveProbeParams {
            epsilon: 0.0,
            min_candidates: 0,
            min_nprobe: 1,
            max_nprobe: 2,
        };
        let top_k = 1;
        let hits = run_top_n(&fixture, None, query.clone(), top_k, params)?;
        let oracle = brute_force_oracle(&column, Metric::L2, &query, top_k, None, None);
        assert_hits_match_oracle(&hits, &oracle, "floor activated");

        // Confirm the floor was load-bearing: the floor at the call
        // site is 4 * top_n = 4, A yields only 1 survivor, so probing
        // had to continue to B to reach the floor.
        let cluster_a_docs = column.cluster_doc_ids(0)?.expect("A");
        assert!(
            cluster_a_docs.len() < 4 * top_k,
            "test geometry requires |A| < 4*top_n: |A|={}",
            cluster_a_docs.len(),
        );
        Ok(())
    }

    // Extra sanity: top_n=0 returns empty without touching the column.
    #[test]
    fn ivf_top_n_zero_returns_empty() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let docs = vec![("a", vec![1.0_f32, 0.2]), ("b", vec![9.4, 9.1])];
        let fixture = build_ivf_segment(2, Metric::L2, centroids, &docs)?;
        let hits = run_top_n(
            &fixture,
            None,
            vec![0.0_f32, 0.0],
            0,
            AdaptiveProbeParams::default(),
        )?;
        assert!(hits.is_empty());
        Ok(())
    }

    // Smoke test for the instrumented entry point: probed_clusters is
    // non-empty, every entry is within [0, num_centroids), and
    // candidates_scored is at most total docs. Exhaustive params on a
    // 2-cluster segment must visit both clusters.
    #[test]
    fn ivf_top_n_instrumented_collects_probe_stats() -> crate::Result<()> {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0]];
        let docs = vec![
            ("a", vec![1.0_f32, 0.0]),
            ("a", vec![0.0, 1.0]),
            ("b", vec![9.0, 11.0]),
            ("b", vec![11.0, 9.0]),
        ];
        let fixture = build_ivf_segment(2, Metric::L2, centroids, &docs)?;
        let searcher = fixture.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let weight = AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            fixture.vec_field,
            Arc::new(vec![0.0_f32, 0.0]),
            exhaustive_params(2),
        )?;
        let ivf = match &backend {
            VectorBackend::Ivf(b) => b,
            _ => panic!("expected IVF backend"),
        };
        let (hits, stats) = ivf.top_n_instrumented(weight.as_ref(), segment_reader, 4)?;
        assert_eq!(hits.len(), 4);
        // Exhaustive params on a 2-cluster segment ⇒ both clusters probed.
        assert_eq!(
            stats.probed_clusters.len(),
            2,
            "probed: {:?}",
            stats.probed_clusters
        );
        for &c in &stats.probed_clusters {
            assert!(c < 2, "probed cluster {c} out of range");
        }
        // 4 docs all pass AllQuery + are alive ⇒ all 4 scored.
        assert_eq!(stats.candidates_scored, 4);
        Ok(())
    }
}
