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

use super::flat::{FlatVecReader, VectorColumn as FlatVectorColumn};
use super::ivf::{AdaptiveProbeParams, IvfVecReader, IvfVectorColumn};
use super::options::{Metric, VectorElement};
use crate::query::Weight;
use crate::schema::{Field, FieldType, Schema};
use crate::{DocId, Score, SegmentReader, TantivyError};

/// Per-segment vector backend. Pick via [`VectorBackend::for_segment`].
pub enum VectorBackend<T: VectorElement> {
    Flat(FlatBackend<T>),
    Ivf(IvfBackend<T>),
}

pub struct FlatBackend<T: VectorElement> {
    #[allow(dead_code)] // wired up when the flat backend lands
    column: FlatVectorColumn,
    #[allow(dead_code)]
    metric: Metric,
    #[allow(dead_code)]
    query: Arc<Vec<T>>,
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
}

impl<T: VectorElement> VectorBackend<T> {
    /// Probe plugins in priority order: IVF if the segment has it, else
    /// flat. Returns an error if the segment has no vector data at all.
    pub fn for_segment(
        segment_reader: &SegmentReader,
        field: Field,
        query: Arc<Vec<T>>,
        adaptive: AdaptiveProbeParams,
    ) -> crate::Result<Self> {
        let schema = segment_reader.schema();
        let metric = lookup_metric(&schema, field)?;

        if let Some(ivf) = segment_reader.plugin_reader::<IvfVecReader>("ivf_vec")? {
            if let Some(column) = ivf.open_column(field) {
                return Ok(Self::Ivf(IvfBackend {
                    column,
                    metric,
                    query,
                    adaptive,
                }));
            }
        }

        let flat = segment_reader
            .plugin_reader::<FlatVecReader>("flat_vec")?
            .ok_or_else(|| TantivyError::InternalError("flat_vec plugin reader missing".into()))?;
        let column = flat.open_column(field).ok_or_else(|| {
            TantivyError::InternalError(format!(
                "no vector data for field {:?} in segment",
                schema.get_field_name(field),
            ))
        })?;
        Ok(Self::Flat(FlatBackend {
            column,
            metric,
            query,
        }))
    }

    /// Top-N within this segment. Each variant decides whether to
    /// iterate the filter (flat) or drain it into a bitmap (IVF).
    pub fn top_n(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<Vec<(Score, DocId)>> {
        match self {
            Self::Flat(b) => b.top_n(weight, segment_reader, top_n),
            Self::Ivf(b) => b.top_n(weight, segment_reader, top_n),
        }
    }
}

impl<T: VectorElement> FlatBackend<T> {
    fn top_n(
        &self,
        _weight: &dyn Weight,
        _segment_reader: &SegmentReader,
        _top_n: usize,
    ) -> crate::Result<Vec<(Score, DocId)>> {
        // Sketch of the implementation when the flat reader lands:
        //   1. Walk the filter `DocSet` in ascending doc order via
        //      weight.for_each_no_score, gated by segment_reader.alive_bitset().
        //   2. For each surviving doc, fetch self.column.vector_bytes_at(doc) and score it via
        //      self.metric.similarity_bytes(&self.query[..], bytes).
        //   3. Push each (score, doc) into a TopNComputer<Score, DocId>. The ascending-doc-id walk
        //      lets us use the fast `TopNComputer::push` path; cluster-order backends like IVF use
        //      `push_unordered` instead.
        //   4. Return the descending-similarity sorted vec.
        todo!("flat segment-level top-N")
    }
}

impl<T: VectorElement> IvfBackend<T> {
    fn top_n(
        &self,
        _weight: &dyn Weight,
        _segment_reader: &SegmentReader,
        _top_n: usize,
    ) -> crate::Result<Vec<(Score, DocId)>> {
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
