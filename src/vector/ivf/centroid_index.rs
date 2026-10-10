//! The index-level centroid index: the consumer-provided [`CentroidProducer`]
//! trait and the immutable `centroids` file it is serialized into.
//!
//! Centroids are an index-level artifact, installed at index creation like
//! the schema and settings: the consumer trains them externally (over the
//! whole corpus, not a per-segment shard) and hands them over through
//! [`CentroidProducer`]. The file is written once, before the first
//! `meta.json` references it, and never mutates — one set per index, for
//! the index's whole life. A future re-publish (background reclustering,
//! SPFresh-style maintenance) will need its own versioning design; none
//! exists today.
//!
//! On-disk layout: the 4-byte vector format header, then a
//! [`CompositeFile`] with per-field slots (see
//! `header::centroid_index_slot`):
//!
//! ```text
//! [0] num_centroids (u32) + centroid rows (C · stride, normalized here
//!     for Cosine so every downstream consumer scores the same bytes)
//! [1] required router-kind discriminant followed by the router payload
//! ```

use std::io::Write;

use common::{BinarySerializable, HasLen, OwnedBytes};

use super::{decode_row, encode_vector, IvfCentroids};
use crate::core::CENTROIDS_FILEPATH;
use crate::directory::{CompositeFile, CompositeWrite, Directory, FileSlice};
use crate::schema::{Field, FieldType, Metric, Schema, VectorDType, VectorOptions};
use crate::vector::distance::{
    maybe_normalize_bytes, DotAccumulator, NormalizeOutcome, Similarity,
};
use crate::vector::header::{centroid_index_slot, read_header, write_header, VectorFileVersion};
use crate::vector::index_reader::{visit_row_fragments, visit_rows};
use crate::vector::router::{InMemoryRouter, LazyRouter, RouterIter, RouterKind, RouterWorkspace};
use crate::TantivyError;

/// The consumer-provided centroid producer, pulled once at index creation.
///
/// Implementors train the base centroids; Tantivy builds their router.
/// Segments assign batches against the serialized centroid matrix; the
/// selected router is used for queries.
pub trait CentroidProducer: Send + Sync + 'static {
    /// The centroids for `field`. Required for every vector field in the
    /// schema; erroring here fails index creation.
    fn centroids(&self, field: Field, options: &VectorOptions) -> crate::Result<IvfCentroids>;
}

impl dyn CentroidProducer {
    /// Pull every vector field's centroids, validate and normalize them,
    /// build the selected router, and write the [`CENTROIDS_FILEPATH`] file.
    pub(crate) fn serialize(
        &self,
        directory: &dyn Directory,
        schema: &Schema,
        router: RouterKind,
    ) -> crate::Result<()> {
        let mut write = directory.open_write(&CENTROIDS_FILEPATH)?;
        write_header(&mut write)?;
        let mut composite = CompositeWrite::wrap(write);

        for (field, entry) in schema.fields() {
            let opts = match entry.field_type() {
                FieldType::Vector(opts) => opts,
                _ => continue,
            };
            let mut centroids = self.centroids(field, opts)?;
            {
                let IvfCentroids::F32(matrix) = &centroids;
                if matrix.dims != opts.dim() {
                    return Err(TantivyError::InvalidArgument(format!(
                        "CentroidProducer produced centroids with {} dimensions for field '{}', \
                         expected {}",
                        matrix.dims,
                        entry.name(),
                        opts.dim()
                    )));
                }
                if matrix.values.len() != matrix.rows * matrix.dims {
                    return Err(TantivyError::InvalidArgument(format!(
                        "CentroidProducer produced {} centroid values for {} rows x {} dimensions \
                         in field '{}'",
                        matrix.values.len(),
                        matrix.rows,
                        matrix.dims,
                        entry.name()
                    )));
                }
                if matrix.rows == 0 {
                    return Err(TantivyError::InvalidArgument(format!(
                        "CentroidProducer produced no centroids for field '{}'",
                        entry.name()
                    )));
                }
                u32::try_from(matrix.rows).map_err(|_| {
                    TantivyError::InvalidArgument(format!(
                        "CentroidProducer produced more than u32::MAX centroids for field '{}'",
                        entry.name()
                    ))
                })?;
            }

            // Normalize INTO the stored bytes for Cosine, so the segment
            // bounds folds and the future search path all score the exact
            // bytes written here. Non-finite centroids are a hard creation
            // error — this is consumer input at its validation boundary. A
            // zero-norm row under Cosine stays as-is; assignment tolerates it
            // and the segment bounds fold saturates its cluster.
            {
                let IvfCentroids::F32(matrix) = &mut centroids;
                for (centroid_ord, centroid) in
                    matrix.values.chunks_exact_mut(opts.dim()).enumerate()
                {
                    let mut bytes = encode_vector(centroid, opts.dim())?;
                    if maybe_normalize_bytes(opts, &mut bytes) == NormalizeOutcome::NonFinite {
                        return Err(TantivyError::InvalidArgument(format!(
                            "CentroidProducer produced a non-finite centroid (ord {centroid_ord}) \
                             for field '{}'",
                            entry.name()
                        )));
                    }
                    centroid.copy_from_slice(&decode_row::<f32>(&bytes, opts.dim())?);
                }
            }

            let router = InMemoryRouter::from(router, &router_options(opts), &mut centroids)?;
            let IvfCentroids::F32(matrix) = &centroids;

            let mut centroid_bytes = Vec::with_capacity(matrix.rows * opts.bytes_per_vector());
            for centroid in matrix.values.chunks_exact(opts.dim()) {
                centroid_bytes.extend_from_slice(&encode_vector(centroid, opts.dim())?);
            }
            {
                let centroids_w =
                    composite.for_field_with_idx(field, centroid_index_slot::CENTROIDS);
                (matrix.rows as u32).serialize(centroids_w)?;
                centroids_w.write_all(&centroid_bytes)?;
                centroids_w.flush()?;
            }
            let router_w = composite.for_field_with_idx(field, centroid_index_slot::ROUTER);
            router.serialize(router_w)?;
            router_w.flush()?;
        }
        composite.close()?;
        Ok(())
    }
}

fn router_options(options: &VectorOptions) -> VectorOptions {
    VectorOptions::new(options.dim(), routing_metric(options)).with_dtype(options.dtype())
}

pub(crate) fn routing_metric(options: &VectorOptions) -> Metric {
    match options.metric() {
        Metric::Cosine => Metric::Dot,
        metric => metric,
    }
}

/// Reader over one `centroids` file.
pub(crate) struct CentroidIndexReader {
    version: VectorFileVersion,
    composite: CompositeFile,
}

impl CentroidIndexReader {
    /// Open the set file named `filename` (from the meta's
    /// `centroid_index` record) in `directory`.
    pub(crate) fn open(
        directory: &dyn Directory,
        filename: &std::path::Path,
    ) -> crate::Result<Self> {
        let file = directory.open_read(filename)?;
        let (version, body) = read_header(&file)?;
        if version < VectorFileVersion::V3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("file {} is not a centroid index file", filename.display()),
            )
            .into());
        }
        let composite = CompositeFile::open(&body)?;
        Ok(CentroidIndexReader { version, composite })
    }

    /// The stored centroids of `field`. Every vector field is validated to
    /// have a slot at creation, so absence here is corruption or a
    /// schema/set mismatch, not an old file.
    pub(crate) fn field_centroids(
        &self,
        field: Field,
        options: &VectorOptions,
    ) -> crate::Result<FieldCentroids> {
        let Some(slice) = self
            .composite
            .open_read_with_idx(field, centroid_index_slot::CENTROIDS)
        else {
            return Err(TantivyError::InternalError(format!(
                "centroid index has no centroids for field {field:?}; the set does not match the \
                 schema"
            )));
        };
        let count_len = std::mem::size_of::<u32>();
        if slice.len() < count_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid index slot is smaller than its count word",
            )
            .into());
        }
        let count_bytes = slice.slice_to(count_len).read_bytes()?;
        let num_centroids = u32::deserialize(&mut count_bytes.as_slice())? as usize;
        let stride = options.bytes_per_vector();
        let rows = slice.slice_from(count_len).read_bytes()?;
        if rows.len() != num_centroids * stride {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid index byte length mismatch",
            )
            .into());
        }
        Ok(FieldCentroids {
            num_centroids,
            stride,
            rows,
        })
    }

    pub(crate) fn router_slice(&self, field: Field) -> crate::Result<FileSlice> {
        self.composite
            .open_read_with_idx(field, centroid_index_slot::ROUTER)
            .ok_or_else(|| {
                TantivyError::InternalError(format!(
                    "centroid index has no router for field {field:?}"
                ))
            })
    }

    /// The centroid count and the rows as a lazy [`FileSlice`] (past the
    /// count word) — the search path's view, which never materializes the
    /// rows whole.
    pub(crate) fn field_rows(
        &self,
        field: Field,
        options: &VectorOptions,
    ) -> crate::Result<(usize, FileSlice)> {
        let Some(slice) = self
            .composite
            .open_read_with_idx(field, centroid_index_slot::CENTROIDS)
        else {
            return Err(TantivyError::InternalError(format!(
                "centroid index has no centroids for field {field:?}; the set does not match the \
                 schema"
            )));
        };
        let count_len = std::mem::size_of::<u32>();
        if slice.len() < count_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid index slot is smaller than its count word",
            )
            .into());
        }
        let count_bytes = slice.slice_to(count_len).read_bytes()?;
        let num_centroids = u32::deserialize(&mut count_bytes.as_slice())? as usize;
        let rows = slice.slice_from(count_len);
        if rows.len() != num_centroids * options.bytes_per_vector() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "centroid index byte length mismatch",
            )
            .into());
        }
        Ok((num_centroids, rows))
    }
}

/// One field's centroids, materialized from the set file.
pub(crate) struct FieldCentroids {
    num_centroids: usize,
    stride: usize,
    rows: OwnedBytes,
}

impl FieldCentroids {
    pub(crate) fn num_centroids(&self) -> usize {
        self.num_centroids
    }

    pub(crate) fn decode_rows(&self) -> Vec<f32> {
        self.rows
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 centroid")))
            .collect()
    }

    /// The stored bytes of centroid `c`.
    pub(crate) fn centroid_bytes(&self, c: usize) -> &[u8] {
        &self.rows[c * self.stride..(c + 1) * self.stride]
    }
}

/// The search-time view of the centroid index: per vector field, the lazy
/// centroid rows plus the parsed router. Opened once and cached on
/// [`Index`](crate::Index) — the router adjacency alone is
/// `C × max_edges × 4` bytes, far too heavy to parse per query.
pub(crate) struct CachedCentroidIndex {
    fields: std::collections::HashMap<Field, FieldRouter>,
}

impl CachedCentroidIndex {
    /// Open the set file and parse every vector field's router.
    pub(crate) fn open(
        directory: &dyn Directory,
        filename: &std::path::Path,
        schema: &Schema,
        router: RouterKind,
    ) -> crate::Result<Self> {
        let reader = CentroidIndexReader::open(directory, filename)?;
        let mut fields = std::collections::HashMap::new();
        for (field, entry) in schema.fields() {
            let opts = match entry.field_type() {
                FieldType::Vector(opts) => opts,
                _ => continue,
            };
            let (num_centroids, rows_slice) = reader.field_rows(field, opts)?;
            let routing_options = router_options(opts);
            let router = router.open(
                reader.version,
                reader.router_slice(field)?,
                rows_slice.clone(),
                &routing_options,
            )?;
            fields.insert(
                field,
                FieldRouter {
                    num_centroids,
                    routing_options,
                    router,
                    rows: rows_slice,
                },
            );
        }
        Ok(CachedCentroidIndex { fields })
    }

    pub(crate) fn field_router(&self, field: Field) -> Option<&FieldRouter> {
        self.fields.get(&field)
    }
}

/// One field's routing state within a [`CachedCentroidIndex`]: says which
/// clusters a query should probe, index-wide — every segment shares these
/// cluster ids.
pub(crate) struct FieldRouter {
    num_centroids: usize,
    routing_options: VectorOptions,
    router: LazyRouter,
    rows: FileSlice,
}

impl FieldRouter {
    pub(crate) fn num_centroids(&self) -> usize {
        self.num_centroids
    }

    #[cfg(test)]
    pub(crate) fn router(&self) -> &LazyRouter {
        &self.router
    }

    pub(crate) fn rank_clusters<'router, 'workspace>(
        &'router self,
        workspace: &'workspace mut RouterWorkspace,
        query: &'router [f32],
    ) -> RouterIter<'router, 'workspace> {
        self.rank_clusters_with_scores(workspace, query, None)
    }

    pub(crate) fn rank_clusters_with_scores<'router, 'workspace>(
        &'router self,
        workspace: &'workspace mut RouterWorkspace,
        query: &'router [f32],
        scores: Option<&'router [Similarity]>,
    ) -> RouterIter<'router, 'workspace> {
        self.router
            .rank(workspace, query, self.routing_options.metric(), scores)
    }

    pub(crate) fn precompute_scores(
        &self,
        query: &[f32],
    ) -> crate::Result<Option<Vec<Similarity>>> {
        const MAX_SCORE_BYTES: usize = 1024 * 1024;
        if !matches!(self.router, LazyRouter::Rng(_))
            || self.num_centroids > MAX_SCORE_BYTES / std::mem::size_of::<Similarity>()
        {
            return Ok(None);
        }
        let mut scores = Vec::with_capacity(self.num_centroids);
        let mut scratch = Vec::new();
        let stride = self.routing_options.bytes_per_vector();
        let metric = self.routing_options.metric();
        let mut accumulator = (metric == Metric::Dot
            && self.routing_options.dtype() == VectorDType::F32)
            .then(DotAccumulator::new);
        for begin in (0..self.num_centroids).step_by(64) {
            let rows = begin..(begin + 64).min(self.num_centroids);
            if let Some(accumulator) = &mut accumulator {
                visit_row_fragments(&self.rows, stride, rows, |_, offset, bytes| {
                    if offset == 0 && bytes.len() == stride {
                        scores.push(metric.similarity_bytes(query, bytes));
                    } else {
                        accumulator.push::<f32>(query, bytes);
                        if offset + bytes.len() == stride {
                            scores.push(Similarity::new(accumulator.finish::<f32>(query)));
                        }
                    }
                })?;
            } else {
                visit_rows(&self.rows, stride, rows, &mut scratch, |_, bytes| {
                    scores.push(metric.similarity_bytes(query, bytes))
                })?;
            }
        }
        Ok(Some(scores))
    }
}

#[cfg(test)]
mod score_tests {
    use std::io;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::directory::FileHandle;
    use crate::vector::distance::norm_squared_wide;
    use crate::vector::ivf::graph::Graph;

    const ROWS_OFFSET: usize = 5;

    #[derive(Clone, Copy, Debug)]
    enum ChunkFault {
        Short,
        Long,
        Io,
    }

    #[derive(Debug)]
    struct CountedRows {
        bytes: OwnedBytes,
        chunk_size: usize,
        requests: Mutex<Vec<Range<usize>>>,
        fault: Mutex<Option<ChunkFault>>,
    }

    impl HasLen for CountedRows {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl FileHandle for CountedRows {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            self.requests.lock().unwrap().push(range.clone());
            Ok(self.bytes.slice(range))
        }

        fn read_bytes_chunks(
            &self,
            range: Range<usize>,
            visitor: &mut dyn FnMut(&[u8]),
        ) -> io::Result<()> {
            self.requests.lock().unwrap().push(range.clone());
            let fault = *self.fault.lock().unwrap();
            let end = match fault {
                Some(ChunkFault::Short) => range.end - 1,
                Some(ChunkFault::Long) => range.end + 1,
                _ => range.end,
            };
            visitor(&[]);
            let mut start = range.start;
            while start < end {
                let next = if self.chunk_size == 0 {
                    end
                } else {
                    ((start / self.chunk_size + 1) * self.chunk_size).min(end)
                };
                visitor(&self.bytes[start..next]);
                start = next;
                if matches!(fault, Some(ChunkFault::Io)) {
                    return Err(io::Error::other("centroid read failed after a chunk"));
                }
            }
            visitor(&[]);
            Ok(())
        }
    }

    fn field_router(
        metric: Metric,
        count: usize,
        dim: usize,
        chunk_size: usize,
    ) -> crate::Result<(FieldRouter, Arc<CountedRows>)> {
        let options = VectorOptions::new(dim, metric);
        let mut bytes = vec![0; ROWS_OFFSET];
        for row in 0..count {
            let mut values: Vec<u8> = (0..dim)
                .flat_map(|col| (((row / 2 + col * 3) % 23) as f32 * 0.125).to_le_bytes())
                .collect();
            maybe_normalize_bytes(&options, &mut values);
            bytes.extend_from_slice(&values);
        }
        let end = bytes.len();
        bytes.extend_from_slice(&[0; 8]);
        let file = Arc::new(CountedRows {
            bytes: OwnedBytes::new(bytes),
            chunk_size,
            requests: Mutex::new(Vec::new()),
            fault: Mutex::new(None),
        });
        let rows = FileSlice::new(file.clone()).slice(ROWS_OFFSET..end);
        let mut graph = Graph::new(vec![0.0f32; count * dim], dim, 2);
        for row in 0..count {
            if row > 0 {
                graph.add_edge(row as u32, row as u32 - 1, Similarity::new(1.0));
            }
            if row + 1 < count {
                graph.add_edge(row as u32, row as u32 + 1, Similarity::new(1.0));
            }
        }
        let mut payload = vec![RouterKind::Rng as u8];
        graph.serialize(&mut payload)?;
        let routing_options = router_options(&options);
        let router = RouterKind::Rng.open(
            VectorFileVersion::V3,
            FileSlice::from(payload),
            rows.clone(),
            &routing_options,
        )?;
        Ok((
            FieldRouter {
                num_centroids: count,
                routing_options,
                router,
                rows,
            },
            file,
        ))
    }

    #[test]
    fn streamed_scores_preserve_rng_order_bits_and_metrics() -> crate::Result<()> {
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            for dim in [17, 1024] {
                for chunk_size in [0, 29, 8160] {
                    let (router, file) = field_router(metric, 139, dim, chunk_size)?;
                    let mut query: Vec<_> = (0..dim).map(|i| i as f32 * 0.03 - 0.2).collect();
                    if metric == Metric::Cosine {
                        let norm = norm_squared_wide(&query).sqrt();
                        for value in &mut query {
                            *value = (f64::from(*value) / norm) as f32;
                        }
                    }
                    let scores = router.precompute_scores(&query)?.unwrap();
                    assert_eq!(scores.len(), 139);
                    let requests = file.requests.lock().unwrap();
                    assert_eq!(
                        *requests,
                        (0..139)
                            .step_by(64)
                            .map(|begin| ROWS_OFFSET + begin * dim * 4
                                ..ROWS_OFFSET + (begin + 64).min(139) * dim * 4)
                            .collect::<Vec<_>>()
                    );
                    drop(requests);
                    let mut lazy_workspace = RouterWorkspace::default();
                    let mut scored_workspace = RouterWorkspace::default();
                    let mut lazy = router.rank_clusters(&mut lazy_workspace, &query);
                    let mut scored = router.rank_clusters_with_scores(
                        &mut scored_workspace,
                        &query,
                        Some(&scores),
                    );
                    for _ in 0..140 {
                        let expected = lazy.next();
                        let reads = file.requests.lock().unwrap().len();
                        let actual = scored.next();
                        assert_eq!(file.requests.lock().unwrap().len(), reads);
                        assert_eq!(
                            actual.map(|c| (c.node, c.sim.score().to_bits())),
                            expected.map(|c| (c.node, c.sim.score().to_bits())),
                        );
                        assert_eq!(
                            serde_json::to_value(scored.metrics())?,
                            serde_json::to_value(lazy.metrics())?,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn precomputed_scores_match_contiguous_bits_across_fragments() -> crate::Result<()> {
        for metric in [Metric::Dot, Metric::L2, Metric::Cosine] {
            for dim in [1, 15, 16, 17, 31, 33, 1024] {
                for chunk_size in [1, 3, 63, 64, 65, 8160] {
                    let (router, file) = field_router(metric, 65, dim, chunk_size)?;
                    for profile in 0..3 {
                        let mut query: Vec<_> = (0..dim)
                            .map(|i| match profile {
                                0 => (i as f32 * 0.0137) - 0.2,
                                1 => [1.0e20, -1.0e20, 1.0e-20, -1.0e-20][i % 4],
                                _ => [0.0, -0.0][i % 2],
                            })
                            .collect();
                        if metric == Metric::Cosine {
                            let norm = norm_squared_wide(&query).sqrt();
                            if norm != 0.0 {
                                for value in &mut query {
                                    *value = (f64::from(*value) / norm) as f32;
                                }
                            }
                        }
                        let actual = router.precompute_scores(&query)?.unwrap();
                        let stride = dim * 4;
                        let expected = file.bytes[ROWS_OFFSET..ROWS_OFFSET + 65 * stride]
                            .chunks_exact(stride)
                            .map(|bytes| {
                                router
                                    .routing_options
                                    .metric()
                                    .similarity_bytes(&query, bytes)
                            });
                        assert_eq!(actual.len(), 65);
                        for (actual, expected) in actual.iter().zip(expected) {
                            assert_eq!(actual.score().to_bits(), expected.score().to_bits());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn precompute_preserves_empty_and_partial_batch_reads() -> crate::Result<()> {
        for count in [0, 1, 63, 64, 65, 128, 129] {
            let (router, file) = field_router(Metric::Dot, count, 17, 29)?;
            assert_eq!(router.precompute_scores(&[1.0; 17])?.unwrap().len(), count);
            assert_eq!(
                *file.requests.lock().unwrap(),
                (0..count)
                    .step_by(64)
                    .map(|begin| ROWS_OFFSET + begin * 17 * 4
                        ..ROWS_OFFSET + (begin + 64).min(count) * 17 * 4)
                    .collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    #[test]
    fn precompute_rejects_incomplete_and_failed_chunk_reads() -> crate::Result<()> {
        for metric in [Metric::Dot, Metric::L2] {
            let (router, file) = field_router(metric, 65, 3, 17)?;
            for fault in [ChunkFault::Short, ChunkFault::Long, ChunkFault::Io] {
                *file.fault.lock().unwrap() = Some(fault);
                let error = router.precompute_scores(&[1.0; 3]).unwrap_err();
                let expected = if matches!(fault, ChunkFault::Io) {
                    "centroid read failed after a chunk"
                } else {
                    "do not cover the requested range"
                };
                assert!(error.to_string().contains(expected), "{error}");
            }
            *file.fault.lock().unwrap() = None;
            assert_eq!(router.precompute_scores(&[1.0; 3])?.unwrap().len(), 65);
        }
        Ok(())
    }

    #[test]
    fn score_cap_and_non_rng_leave_rows_lazy() -> crate::Result<()> {
        let (router, file) = field_router(Metric::Dot, 1024 * 1024 / 4 + 1, 1, 0)?;
        assert!(router.precompute_scores(&[1.0])?.is_none());
        assert!(file.requests.lock().unwrap().is_empty());

        let options = VectorOptions::new(1, Metric::Dot);
        let mut centroids = IvfCentroids::F32(crate::vector::IvfMatrix {
            values: vec![0.0, 1.0, 1.0],
            rows: 3,
            dims: 1,
        });
        let built = RouterKind::Exact.build(&options, &mut centroids)?;
        let mut payload = Vec::new();
        built.serialize(&mut payload)?;
        let rows = FileSlice::from(vec![0; 12]);
        let router = FieldRouter {
            num_centroids: 3,
            routing_options: options.clone(),
            router: RouterKind::Exact.open(
                VectorFileVersion::V3,
                FileSlice::from(payload),
                rows.clone(),
                &options,
            )?,
            rows,
        };
        assert!(router.precompute_scores(&[1.0])?.is_none());
        Ok(())
    }
}
