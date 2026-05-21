//! Per-segment dispatch over vector storage formats.
//!
//! Picked once per segment by
//! [`TopDocsByVectorSimilarity`](super::collector::TopDocsByVectorSimilarity). Each variant owns
//! its top-N loop: [`FlatBackend`] iterates the filter `Scorer` doc-by-doc, [`IvfBackend`] drains
//! the filter into a bitmap and probes clusters adaptively.
//!
//! Adding a new format (HNSW, etc.) is a new enum variant — the
//! collector layer doesn't change.

use std::sync::Arc;

use super::flat::FlatVectorColumn;
use super::ivf::{AdaptiveProbeParams, IvfVectorColumn};
use super::options::{Metric, VectorElement};
use super::reader::{VectorColumn, VectorColumnReader, VectorReader};
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
    #[allow(dead_code)] // wired up when the IVF backend lands
    column: IvfVectorColumn,
    #[allow(dead_code)]
    metric: Metric,
    #[allow(dead_code)]
    query: Arc<Vec<T>>,
    #[allow(dead_code)]
    adaptive: AdaptiveProbeParams,
    #[allow(dead_code)]
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
        let mut topn = TopNComputer::<Score, DocId, _>::new(top_n);
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

impl<T: VectorElement> IvfBackend<T> {
    fn top_n(
        &self,
        _weight: &dyn Weight,
        _segment_reader: &SegmentReader,
        _top_n: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        // Sketch of the implementation when the IVF reader lands:
        //   1. Drain the filter `DocSet` into a RoaringBitmap via weight.for_each_no_score(reader,
        //      |docs| bitmap.insert_many(docs)).
        //   2. probe_order = self.column.rank_centroids(&self.query, self.metric).
        //   3. for each cluster in probe_order, intersect its doc ids with the bitmap, score
        //      survivors via metric.similarity_bytes against
        //      self.column.vector_bytes_in_cluster(cluster, doc), and push each into a TopNComputer
        //      via `push_unordered` (cluster-order iteration breaks the ascending-D invariant of
        //      plain `push`).
        //   4. Stop when AdaptiveProbeParams convergence criterion fires.
        //   5. Tag results with `self.segment_ord` as a `DocAddress` on the way out.
        todo!("IVF segment-level top-N")
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
///   inequality). The threshold here is a pragmatic linear widening of the score floor: `best - eps
///   * |best|`. NOTE: with unnormalized dot, the IVF locality assumption itself is heuristic —
///   "query near a centroid ⇒ true nearest neighbors live in that cluster" can fail when a
///   high-magnitude vector in a far cluster outscores nearby ones. That's the clusterer's problem,
///   not the threshold's; this function just controls when probing stops.
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
}
