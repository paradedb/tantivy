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
use super::VECTOR_PLUGIN_NAME;
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
        let metric = lookup_metric(&schema, field)?;

        let vec_reader = segment_reader
            .plugin_reader::<VectorReader>(VECTOR_PLUGIN_NAME)?
            .ok_or_else(|| TantivyError::InternalError("vectors plugin reader missing".into()))?;

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
