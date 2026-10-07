//! Immutable index-level centroids and routers. See `vector/FORMAT.md`.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use common::{BinarySerializable, HasLen};

use super::index::RouterIndex;
use super::{decode_row, encode_vector, IvfCentroids};
use crate::directory::{CompositeFile, CompositeWrite, Directory};
use crate::error::DataCorruption;
use crate::index::CentroidIndexMeta;
use crate::schema::{Field, FieldType, Schema, VectorOptions};
use crate::vector::distance::maybe_normalize_bytes;
use crate::vector::RouterKind;
use crate::TantivyError;

const MAGIC: &[u8; 4] = b"TVRI";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 8;
const META: usize = 0;
const CENTROIDS: usize = 1;
const ROUTER: usize = 2;

/// Supplies trained centroids once per vector field at index creation.
/// Tantivy validates and normalizes the rows, builds their router, and persists
/// its final centroid order. Reopening an index does not need this producer.
pub trait CentroidProducer: Send + Sync + 'static {
    /// Returns a nonempty, finite centroid matrix matching the field's dimensions.
    fn centroids(&self, field: Field, options: &VectorOptions) -> crate::Result<IvfCentroids>;
}

pub(crate) type CentroidIndex = HashMap<Field, Arc<RouterIndex>>;

#[derive(serde::Serialize, serde::Deserialize)]
struct FieldMeta {
    num_centroids: u32,
}

pub(crate) fn write_centroid_index(
    directory: &dyn Directory,
    schema: &Schema,
    producer: &dyn CentroidProducer,
    router: RouterKind,
) -> crate::Result<CentroidIndexMeta> {
    let meta = CentroidIndexMeta {
        file_name: format!("centroids-{}", uuid::Uuid::new_v4().simple()).into(),
    };
    let mut write = directory.open_write(&meta.file_name)?;
    write.write_all(MAGIC)?;
    VERSION.serialize(&mut write)?;
    let mut composite = CompositeWrite::wrap(write);
    for (field, entry) in schema.fields() {
        let FieldType::Vector(options) = entry.field_type() else {
            continue;
        };
        let mut centroids = producer.centroids(field, options)?;
        let IvfCentroids::F32(matrix) = &mut centroids;
        if options.dim() == 0
            || matrix.dims != options.dim()
            || matrix.rows == 0
            || matrix.rows >= u32::MAX as usize
            || matrix.rows.checked_mul(matrix.dims) != Some(matrix.values.len())
            || matrix.values.len().checked_mul(size_of::<f32>()).is_none()
        {
            return Err(TantivyError::InvalidArgument(format!(
                "invalid centroid matrix for field '{}': {} values, {} rows, {} dimensions; \
                 expected nonempty rows of {} dimensions",
                entry.name(),
                matrix.values.len(),
                matrix.rows,
                matrix.dims,
                options.dim()
            )));
        }
        if matrix.values.iter().any(|value| !value.is_finite()) {
            return Err(TantivyError::InvalidArgument(format!(
                "non-finite centroid in field '{}'",
                entry.name()
            )));
        }
        if options.needs_normalization() {
            for row in matrix.values.chunks_exact_mut(matrix.dims) {
                let mut bytes = encode_vector(row, options.dim())?;
                maybe_normalize_bytes(options, &mut bytes);
                row.copy_from_slice(&decode_row::<f32>(&bytes, options.dim())?);
            }
        }
        let router = router.build(options, &mut centroids)?;
        let IvfCentroids::F32(matrix) = centroids;
        let metadata = FieldMeta {
            num_centroids: matrix.rows as u32,
        };
        serde_json::to_writer(composite.for_field_with_idx(field, META), &metadata)?;
        let rows = composite.for_field_with_idx(field, CENTROIDS);
        for row in matrix.values.chunks_exact(matrix.dims) {
            rows.write_all(&encode_vector(row, matrix.dims)?)?;
        }
        router.serialize(composite.for_field_with_idx(field, ROUTER))?;
    }
    composite.close()?;
    directory.sync_directory()?;
    Ok(meta)
}

pub(crate) fn open_centroid_index(
    directory: &dyn Directory,
    meta: &CentroidIndexMeta,
    schema: &Schema,
) -> crate::Result<CentroidIndex> {
    let corrupt = |message: &str| DataCorruption::new(meta.file_name.clone(), message.to_string());
    let file = directory.open_read(&meta.file_name)?;
    if file.len() < HEADER_LEN + 5 {
        return Err(corrupt("centroid index is truncated").into());
    }
    let header = file.slice_to(HEADER_LEN).read_bytes()?;
    if &header[..4] != MAGIC {
        return Err(corrupt("invalid centroid index magic").into());
    }
    let version = u32::deserialize(&mut &header[4..])?;
    if version != VERSION {
        return Err(corrupt("unsupported centroid index version; rebuild required").into());
    }
    let body = file.slice_from(HEADER_LEN);
    let composite = CompositeFile::open(&body)?;
    let fields: HashMap<_, _> = schema
        .fields()
        .filter_map(|(field, entry)| match entry.field_type() {
            FieldType::Vector(options) => Some((field, options)),
            _ => None,
        })
        .collect();
    if fields.is_empty()
        || composite
            .field_indices()
            .any(|(field, slot)| !fields.contains_key(&field) || slot > ROUTER)
    {
        return Err(corrupt("centroid index fields do not match the schema").into());
    }
    let mut routers = HashMap::new();
    for (field, options) in fields {
        let slot = |slot| {
            composite
                .open_read_with_idx(field, slot)
                .ok_or_else(|| corrupt("centroid index is missing a field slot"))
        };
        let metadata: FieldMeta = serde_json::from_slice(&slot(META)?.read_bytes()?)
            .map_err(|error| corrupt(&format!("invalid centroid metadata: {error}")))?;
        if options.dim() == 0 {
            return Err(corrupt("centroid field has zero dimensions").into());
        }
        let count = metadata.num_centroids as usize;
        let rows = slot(CENTROIDS)?;
        let expected = count
            .checked_mul(options.dim())
            .and_then(|values| values.checked_mul(options.dtype().size_bytes()));
        if count == 0 || count >= u32::MAX as usize || expected != Some(rows.len()) {
            return Err(corrupt("invalid centroid row count or byte length").into());
        }
        routers.insert(
            field,
            Arc::new(RouterIndex::open(options, count, rows, slot(ROUTER)?)?),
        );
    }
    Ok(routers)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::core::META_FILEPATH;
    use crate::directory::{RamDirectory, TerminatingWrite};
    use crate::indexer::{DocIdMapping, NoMergePolicy};
    use crate::schema::Metric;
    use crate::vector::router::{RouterWorkspace, RoutingParams};
    use crate::vector::{IvfMatrix, VectorStorageFormat};
    use crate::{Index, IndexBuilder, IndexWriter, TantivyDocument};

    #[derive(Default)]
    struct TestProducer {
        fields: HashMap<Field, IvfCentroids>,
        calls: AtomicUsize,
    }

    impl CentroidProducer for TestProducer {
        fn centroids(&self, field: Field, _: &VectorOptions) -> crate::Result<IvfCentroids> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.fields.get(&field).cloned().ok_or_else(|| {
                TantivyError::InvalidArgument(format!("no centroids supplied for {field:?}"))
            })
        }
    }

    struct Fixture {
        schema: Schema,
        field: Field,
        producer: Arc<TestProducer>,
        directory: RamDirectory,
    }

    impl Fixture {
        fn new(metric: Metric) -> Self {
            let mut schema = Schema::builder();
            let field = schema.add_vector_field("embedding", VectorOptions::new(2, metric));
            Self {
                schema: schema.build(),
                field,
                producer: Arc::new(TestProducer {
                    fields: HashMap::from([(
                        field,
                        centroids(&[[0.0, 0.0], [3.0, 4.0], [-4.0, 3.0]]),
                    )]),
                    ..Default::default()
                }),
                directory: RamDirectory::create(),
            }
        }

        fn builder(&self) -> IndexBuilder {
            Index::builder()
                .schema(self.schema.clone())
                .centroid_producer(self.producer.clone())
        }

        fn create(&self, router: RouterKind) -> crate::Result<Index> {
            self.builder()
                .ivf_router(router)?
                .create(self.directory.clone())
        }

        fn replace_centroids(&mut self, centroids: IvfCentroids) -> IvfCentroids {
            Arc::get_mut(&mut self.producer)
                .unwrap()
                .fields
                .insert(self.field, centroids)
                .unwrap()
        }
    }

    fn centroids<const D: usize>(rows: &[[f32; D]]) -> IvfCentroids {
        IvfCentroids::F32(IvfMatrix {
            values: rows.iter().flatten().copied().collect(),
            rows: rows.len(),
            dims: D,
        })
    }

    fn artifact(index: &Index) -> crate::Result<(CentroidIndexMeta, common::OwnedBytes)> {
        let meta = index.load_metas()?.centroid_index.unwrap();
        let bytes = index.directory().open_read(&meta.file_name)?.read_bytes()?;
        Ok((meta, bytes))
    }

    #[test]
    fn round_trip_all_metrics_and_routers() -> crate::Result<()> {
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            for kind in [RouterKind::Exact, RouterKind::Rng, RouterKind::Stacked] {
                let fixture = Fixture::new(metric);
                let field = fixture.field;
                let index = fixture.create(kind)?;
                assert_eq!(fixture.producer.calls.load(Ordering::SeqCst), 1);
                let cache = index.cached_centroid_index()?.unwrap();
                assert!(Arc::ptr_eq(
                    &cache,
                    &index.clone().cached_centroid_index()?.unwrap()
                ));
                let reopened = Index::open(fixture.directory.clone())?;
                let reopened_cache = reopened.cached_centroid_index()?.unwrap();
                assert_eq!(fixture.producer.calls.load(Ordering::SeqCst), 1);
                for router in [&cache[&field], &reopened_cache[&field]] {
                    assert_eq!(router.router(), kind);
                    assert_eq!(router.num_clusters(), 3);
                    let bytes = router.centroid_bytes()?;
                    let rows = decode_row::<f32>(&bytes, 6)?;
                    let mut stored: Vec<_> = rows.chunks_exact(2).collect();
                    stored.sort_by(|a, b| a[0].total_cmp(&b[0]));
                    let expected = if metric == Metric::Cosine {
                        [[-0.8, 0.6], [0.0, 0.0], [0.6, 0.8]]
                    } else {
                        [[-4.0, 3.0], [0.0, 0.0], [3.0, 4.0]]
                    };
                    assert_eq!(stored, expected);
                    let query = [0.6, 0.8];
                    let mut workspace = RouterWorkspace::default();
                    let ranked: Vec<_> = router
                        .rank_clusters(&mut workspace, &query, RoutingParams::default())
                        .collect();
                    assert!(!ranked.is_empty());
                    for candidate in ranked {
                        let start = candidate.node as usize * 8;
                        assert_eq!(
                            candidate.sim,
                            metric.similarity_bytes(&query, &bytes[start..start + 8])
                        );
                    }
                }
                assert_eq!(
                    cache[&field].centroid_bytes()?,
                    reopened_cache[&field].centroid_bytes()?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn stacked_router_persists_its_centroid_permutation() -> crate::Result<()> {
        let mut fixture = Fixture::new(Metric::L2);
        let values: Vec<[f32; 2]> = (0..256)
            .map(|i| [(i % 2) as f32 * 1000.0, (i / 2) as f32])
            .collect();
        fixture.replace_centroids(centroids(&values));
        fixture.create(RouterKind::Stacked)?;
        let reopened = Index::open(fixture.directory)?;
        let cache = reopened.cached_centroid_index()?.unwrap();
        let router = &cache[&fixture.field];
        let bytes = router.centroid_bytes()?;
        let stored = decode_row::<f32>(&bytes, values.len() * 2)?;
        assert_ne!(stored, values.iter().flatten().copied().collect::<Vec<_>>());
        let mut workspace = RouterWorkspace::default();
        for query in &values {
            let nearest = router
                .rank_clusters(&mut workspace, query, RoutingParams { k: 1, recall: 1.0 })
                .next()
                .unwrap();
            let start = nearest.node as usize * 2;
            assert_eq!(&stored[start..start + 2], query);
            assert_eq!(nearest.sim.score(), 0.0);
        }
        Ok(())
    }

    #[test]
    fn every_vector_field_has_its_own_centroids() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let text = schema.add_text_field("text", crate::schema::TEXT);
        let first = schema.add_vector_field("first", VectorOptions::new(2, Metric::L2));
        let second = schema.add_vector_field("second", VectorOptions::new(3, Metric::Dot));
        let producer = Arc::new(TestProducer {
            fields: HashMap::from([
                (first, centroids(&[[1.0, 2.0]])),
                (second, centroids(&[[3.0, 4.0, 5.0]])),
            ]),
            ..Default::default()
        });
        let index = Index::builder()
            .schema(schema.build())
            .centroid_producer(producer.clone())
            .ivf_router(RouterKind::Exact)?
            .create_in_ram()?;
        let cache = index.cached_centroid_index()?.unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache[&first].centroid_bytes()?.len(), 8);
        assert_eq!(cache[&second].centroid_bytes()?.len(), 12);
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        for field in [Some(first), Some(second), None] {
            let mut doc = TantivyDocument::new();
            if field == Some(first) {
                doc.add_text(text, "drop");
            }
            if let Some(field) = field {
                doc.add_vector(
                    field,
                    if field == first {
                        &[1.0, 2.0][..]
                    } else {
                        &[3.0, 4.0, 5.0][..]
                    },
                );
            }
            writer.add_document(doc)?;
        }
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        for (present_doc, field) in [first, second].into_iter().enumerate() {
            let vectors = searcher.segment_readers()[0].vector_index(field)?;
            assert_eq!(vectors.num_vectors(), 1);
            assert_eq!(vectors.index().unwrap().cluster_range(0), 0..1);
            for doc in 0..3 {
                assert_eq!(
                    vectors.vector_bytes(doc)?.is_some(),
                    doc == present_doc as u32
                );
            }
        }
        writer.delete_term(crate::Term::from_field_text(text, "drop"));
        writer.commit()?;
        writer.merge(&index.searchable_segment_ids()?).wait()?;
        let merged = index.reader()?.searcher();
        assert_eq!(merged.num_docs(), 2);
        let empty = merged.segment_readers()[0].vector_index(first)?;
        assert_eq!(empty.num_vectors(), 0);
        assert_eq!(empty.index().unwrap().num_clusters(), 1);
        assert_eq!(empty.index().unwrap().cluster_range(0), 0..0);
        assert_eq!(
            merged.segment_readers()[0]
                .vector_index(second)?
                .num_vectors(),
            1
        );
        assert_eq!(producer.calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn shared_segments_use_stored_centroids_for_assignment_and_encoding() -> crate::Result<()> {
        use crate::index::SegmentComponent;
        use crate::query::AllQuery;
        use crate::vector::header::{read_centroid_header, VectorFileVersion};
        use crate::vector::ivf::AdaptiveProbeParams;
        use crate::vector::{
            residual_norm, TopDocsByVectorSimilarity, VectorQuantizationConfig,
            VectorQuantizationLayer,
        };

        const DIM: usize = 64;
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            for kind in [RouterKind::Exact, RouterKind::Rng, RouterKind::Stacked] {
                for quantized in [false, true] {
                    let mut fixture = Fixture::new(metric);
                    let mut schema = Schema::builder();
                    schema.add_vector_field("embedding", VectorOptions::new(DIM, metric));
                    fixture.schema = schema.build();
                    let mut supplied: Vec<_> = (0..256)
                        .map(|i| {
                            let mut row = [0.0; DIM];
                            row[0] = (i % 2) as f32 * 8.0 - 4.0;
                            row[1] = (i / 2) as f32 * 0.125 - 8.0;
                            row
                        })
                        .collect();
                    supplied.push([0.0; DIM]);
                    fixture.replace_centroids(centroids(&supplied));
                    let config = VectorQuantizationConfig::materialize(
                        "embedding".into(),
                        &VectorOptions::new(DIM, metric),
                        vec![
                            VectorQuantizationLayer { bits: 1, seed: 7 },
                            VectorQuantizationLayer { bits: 4, seed: 11 },
                        ],
                    )?;
                    let index = fixture
                        .builder()
                        .settings(crate::IndexSettings {
                            vector_quantization: if quantized { vec![config] } else { Vec::new() },
                            ..Default::default()
                        })
                        .ivf_router(kind)?
                        .create(fixture.directory.clone())?;
                    let original = artifact(&index)?;
                    for batch in 0..3 {
                        let reopened = Index::open(fixture.directory.clone())?;
                        let mut writer: IndexWriter =
                            reopened.writer_with_num_threads(1, 15_000_000)?;
                        writer.set_merge_policy(Box::new(NoMergePolicy));
                        for (doc, value) in [[0.0, 0.0], [3.0, 4.0], [-4.0, 3.0], [8.0, -2.0]]
                            .iter()
                            .enumerate()
                        {
                            let mut document = TantivyDocument::new();
                            if batch != 2 && doc != 3 {
                                let mut row = [0.0; DIM];
                                row[..2].copy_from_slice(value);
                                document.add_vector(fixture.field, &row);
                                document.add_vector(fixture.field, &[999.0; DIM]);
                            }
                            writer.add_document(document)?;
                        }
                        writer.commit()?;
                    }
                    let cache = index.cached_centroid_index()?.unwrap();
                    let shared = &cache[&fixture.field];
                    let bytes = shared.centroid_bytes()?;
                    let stored = decode_row::<f32>(&bytes, supplied.len() * DIM)?;
                    if kind == RouterKind::Stacked {
                        let mut original_rows = supplied
                            .iter()
                            .map(|row| encode_vector(row, DIM).unwrap())
                            .collect::<Vec<_>>();
                        for row in &mut original_rows {
                            maybe_normalize_bytes(&VectorOptions::new(DIM, metric), row);
                        }
                        assert_ne!(bytes.as_slice(), original_rows.concat());
                    }
                    for merged in [false, true] {
                        if merged {
                            let mut writer: IndexWriter =
                                index.writer_with_num_threads(1, 15_000_000)?;
                            writer.merge(&index.searchable_segment_ids()?).wait()?;
                        }
                        let searcher = index.reader()?.searcher();
                        assert_eq!(searcher.segment_readers().len(), if merged { 1 } else { 3 });
                        let mut total = 0;
                        for segment in searcher.segment_readers() {
                            let vector = segment.vector_index(fixture.field)?;
                            let ivf = vector.index().unwrap();
                            assert_eq!(ivf.router(), kind);
                            assert_eq!(ivf.num_clusters(), supplied.len());
                            assert_eq!(ivf.centroid_bytes()?, bytes);
                            assert_eq!(vector.quantization().is_some(), quantized);
                            let sidecar =
                                segment.open_read(SegmentComponent::Custom("centroids".into()))?;
                            let (version, body) = read_centroid_header(&sidecar)?;
                            assert_eq!(version, VectorFileVersion::V5);
                            let composite = CompositeFile::open(&body)?;
                            let mut slots = composite.field_indices().collect::<Vec<_>>();
                            slots.sort();
                            assert_eq!(slots, [0, 1, 3].map(|slot| (fixture.field, slot)));
                            for cluster in 0..ivf.num_clusters() {
                                let centroid = &stored[cluster * DIM..(cluster + 1) * DIM];
                                let mut bound = if metric == Metric::Cosine
                                    && centroid.iter().all(|&v| v == 0.0)
                                {
                                    f32::INFINITY
                                } else {
                                    0.0
                                };
                                for row in ivf.cluster_range(cluster) {
                                    total += 1;
                                    let doc = vector.doc_id_at(row)?;
                                    let row_bytes = vector.vector_bytes_for_row(row)?;
                                    assert_eq!(vector.vector_bytes(doc)?.unwrap(), row_bytes);
                                    let values = decode_row::<f32>(&row_bytes, DIM)?;
                                    assert!(values.iter().all(|v| v.abs() <= 4.0));
                                    let score = metric.similarity(&values, centroid);
                                    assert!(stored
                                        .chunks_exact(DIM)
                                        .all(|c| metric.similarity(&values, c) <= score));
                                    if metric != Metric::L2 && values.iter().all(|&v| v == 0.0) {
                                        assert_eq!(cluster, 0);
                                    }
                                    bound = bound.max(residual_norm::<f32>(&row_bytes, centroid));
                                    if let Some(quant) = vector.quantization() {
                                        let ctx = quant.index_ctx();
                                        let prepared =
                                            cascade::prepare_centroid(centroid, &ctx.specs);
                                        let mut workspace = cascade::BatchEncodeWorkspace::new();
                                        let mut input = values.clone();
                                        let expected =
                                            cascade::encode_batch_in_place_with_workspace(
                                                &mut input,
                                                1,
                                                &prepared,
                                                &ctx.specs,
                                                &ctx.grids,
                                                &mut workspace,
                                                metric == Metric::L2,
                                            );
                                        assert_eq!(
                                            quant.residual_norm(row)?,
                                            expected.residual_norms_squared[0]
                                        );
                                        for (layer, encoded) in
                                            quant.layers().iter().zip(&expected.layers)
                                        {
                                            assert_eq!(
                                                layer.code_bytes(row)?.as_slice(),
                                                encoded.codes
                                            );
                                            assert_eq!(layer.scale(row)?, encoded.scales[0]);
                                        }
                                    }
                                }
                                assert_eq!(ivf.bounds().ball_r(cluster), bound);
                            }
                            for doc in 0..segment.max_doc() {
                                if doc % 4 == 3 || vector.num_vectors() == 0 {
                                    assert!(vector.vector_bytes(doc)?.is_none());
                                }
                            }
                        }
                        assert_eq!(total, 6);
                        assert!(Arc::strong_count(shared) > 1);
                        if kind == RouterKind::Exact {
                            let mut query = vec![0.0; DIM];
                            query[..2].copy_from_slice(&[3.1, 4.2]);
                            let expected = crate::vector::tests::ground_truth::top_k(
                                &index,
                                fixture.field,
                                metric,
                                &query,
                                6,
                            )?;
                            let result = searcher.search(
                                &AllQuery,
                                &TopDocsByVectorSimilarity::new(fixture.field, query, 6)
                                    .with_adaptive_params(AdaptiveProbeParams {
                                        max_probe_fraction: 1.0,
                                        min_probe_clusters: supplied.len(),
                                        ..Default::default()
                                    }),
                            )?;
                            assert_eq!(result.results, expected);
                        }
                    }
                    assert_eq!(artifact(&index)?, original);
                    assert_eq!(fixture.producer.calls.load(Ordering::SeqCst), 1);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn shared_search_routes_once_across_encodings_and_executors() -> crate::Result<()> {
        use crate::collector::sort_key::SortByStaticFastValue;
        use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
        use crate::schema::{IndexRecordOption, FAST, INDEXED};
        use crate::vector::ivf::AdaptiveProbeParams;
        use crate::vector::tests::ground_truth;
        use crate::vector::{
            TopDocsByVectorSimilarity, VectorQuantizationConfig, VectorQuantizationLayer,
        };
        use crate::{Executor, Order, Term};

        const DIM: usize = 64;
        let parallel = Executor::multi_thread(3, "shared-vector-test")?;
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            for kind in [RouterKind::Exact, RouterKind::Rng, RouterKind::Stacked] {
                let mut fixture = Fixture::new(metric);
                let mut schema = Schema::builder();
                schema.add_vector_field("embedding", VectorOptions::new(DIM, metric));
                let ordinal = schema.add_u64_field("ordinal", FAST | INDEXED);
                let keep = schema.add_u64_field("keep", FAST | INDEXED);
                fixture.schema = schema.build();
                let rows = [[0.0, 0.0], [3.0, 4.0], [-4.0, 3.0]].map(|value| {
                    let mut row = [0.0; DIM];
                    row[..2].copy_from_slice(&value);
                    row
                });
                fixture.replace_centroids(centroids(&rows));
                let mut index = fixture.create(kind)?;
                for batch in 0..4 {
                    index.settings_mut().vector_quantization = if batch == 1 || batch == 2 {
                        vec![VectorQuantizationConfig::materialize(
                            "embedding".into(),
                            &VectorOptions::new(DIM, metric),
                            (0..batch)
                                .map(|layer| VectorQuantizationLayer {
                                    bits: if layer == 0 { 1 } else { 4 },
                                    seed: batch + layer,
                                })
                                .collect(),
                        )?]
                    } else {
                        Vec::new()
                    };
                    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
                    writer.set_merge_policy(Box::new(NoMergePolicy));
                    for local in 0..9 {
                        let id = batch * 9 + local;
                        let mut doc = TantivyDocument::new();
                        doc.add_u64(ordinal, id);
                        doc.add_u64(keep, id % 2);
                        if batch != 3 && local != 8 {
                            let mut row = rows[local as usize % rows.len()];
                            row[0] += 0.01 * (local + 1) as f32;
                            row[2 + batch as usize] = 0.02 * (local + 1) as f32;
                            doc.add_vector(fixture.field, &row);
                        }
                        writer.add_document(doc)?;
                    }
                    writer.commit()?;
                    for id in [4, 13] {
                        writer.delete_term(Term::from_field_u64(ordinal, id));
                    }
                    writer.commit()?;
                    writer.wait_merging_threads()?;
                }
                let mut query = vec![0.0; DIM];
                query[..2].copy_from_slice(&[3.1, 4.2]);
                let collectors: Vec<_> = [0, usize::MAX]
                    .into_iter()
                    .map(|levels| {
                        TopDocsByVectorSimilarity::new(fixture.field, query.clone(), 5)
                            .and_offset(2)
                            .with_max_scan_levels(levels)
                            .with_adaptive_params(AdaptiveProbeParams {
                                max_probe_fraction: 1.0,
                                min_probe_clusters: 3,
                                ..Default::default()
                            })
                            .with_tie_break((
                                SortByStaticFastValue::<u64>::for_field("ordinal"),
                                Order::Asc,
                            ))
                    })
                    .collect();
                for merged in [false, true] {
                    if merged {
                        let mut writer: IndexWriter =
                            index.writer_with_num_threads(1, 15_000_000)?;
                        writer.merge(&index.searchable_segment_ids()?).wait()?;
                    }
                    let searcher = index.reader()?.searcher();
                    let mut oracle =
                        ground_truth::top_k(&index, fixture.field, metric, &query, usize::MAX)?;
                    let id = |address: crate::DocAddress| {
                        searcher.segment_readers()[address.segment_ord as usize]
                            .fast_fields()
                            .u64("ordinal")
                            .unwrap()
                            .first(address.doc_id)
                            .unwrap()
                    };
                    oracle.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| id(a.1).cmp(&id(b.1))));
                    for selected in [None, Some(1), Some(9)] {
                        let filter: Box<dyn Query> = match selected {
                            None => Box::new(AllQuery),
                            Some(value) => Box::new(TermQuery::new(
                                Term::from_field_u64(keep, value),
                                IndexRecordOption::Basic,
                            )),
                        };
                        let expected: Vec<_> = oracle
                            .iter()
                            .copied()
                            .filter(|(_, address)| {
                                selected.is_none_or(|value| id(*address) % 2 == value)
                            })
                            .skip(2)
                            .take(5)
                            .collect();
                        for collector in &collectors {
                            for executor in [&Executor::SingleThread, &parallel] {
                                let result = searcher.search_with_executor(
                                    filter.as_ref(),
                                    collector,
                                    executor,
                                    EnableScoring::disabled_from_searcher(&searcher),
                                )?;
                                assert_eq!(
                                    result.results, expected,
                                    "{metric:?} {kind:?} merged={merged}"
                                );
                                assert_eq!(
                                    result
                                        .stats
                                        .iter()
                                        .filter(|stats| stats.routing.is_some())
                                        .count(),
                                    usize::from(selected != Some(9))
                                );
                                if kind == RouterKind::Exact && selected != Some(9) {
                                    assert_eq!(
                                        result
                                            .stats
                                            .iter()
                                            .map(|stats| stats.routing_visited_count)
                                            .sum::<usize>(),
                                        3
                                    );
                                }
                            }
                        }
                    }
                    let concurrent =
                        parallel.map(|_| searcher.search(&AllQuery, &collectors[1]), 0..3)?;
                    for result in concurrent {
                        assert_eq!(result.results, oracle[2..7]);
                        assert_eq!(
                            result
                                .stats
                                .iter()
                                .filter(|stats| stats.routing.is_some())
                                .count(),
                            1
                        );
                    }
                    let zero = TopDocsByVectorSimilarity::new(fixture.field, query.clone(), 0);
                    let result = searcher.search(&AllQuery, &zero)?;
                    assert!(result.results.is_empty());
                    assert!(result.stats.iter().all(|stats| stats.routing.is_none()));
                }
                assert_eq!(collectors[1].quantized_query_count(), 2);
            }
        }
        Ok(())
    }

    #[test]
    fn global_probe_budget_is_independent_of_segment_partitioning() -> crate::Result<()> {
        use crate::collector::sort_key::SortByStaticFastValue;
        use crate::query::{AllQuery, Query, TermQuery};
        use crate::schema::{IndexRecordOption, FAST, INDEXED};
        use crate::vector::ivf::{AdaptiveProbeParams, WorkModel};
        use crate::vector::{
            TopDocsByVectorSimilarity, VectorQuantizationConfig, VectorQuantizationLayer,
        };
        use crate::{Order, Term};

        for kind in [RouterKind::Exact, RouterKind::Rng, RouterKind::Stacked] {
            let mut fixture = Fixture::new(Metric::L2);
            let mut schema = Schema::builder();
            schema.add_vector_field("embedding", VectorOptions::new(64, Metric::L2));
            let ordinal = schema.add_u64_field("ordinal", FAST);
            let keep = schema.add_u64_field("keep", INDEXED);
            fixture.schema = schema.build();
            let rows: Vec<_> = (0..256)
                .map(|i| {
                    let mut row = [0.0; 64];
                    row[0] = i as f32 * 20.0;
                    row
                })
                .collect();
            fixture.replace_centroids(centroids(&rows));
            let mut index = fixture.create(kind)?;
            for batch in 0..4 {
                index.settings_mut().vector_quantization = if batch == 0 {
                    Vec::new()
                } else {
                    vec![VectorQuantizationConfig::materialize(
                        "embedding".into(),
                        &VectorOptions::new(64, Metric::L2),
                        vec![VectorQuantizationLayer {
                            bits: if batch == 1 { 1 } else { 4 },
                            seed: batch,
                        }],
                    )?]
                };
                let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
                writer.set_merge_policy(Box::new(NoMergePolicy));
                for (cluster, row) in rows.iter().enumerate() {
                    let mut doc = TantivyDocument::new();
                    doc.add_vector(fixture.field, row);
                    doc.add_u64(ordinal, cluster as u64 * 4 + batch);
                    doc.add_u64(keep, batch % 2);
                    writer.add_document(doc)?;
                }
                writer.commit()?;
                writer.wait_merging_threads()?;
            }
            let mut before = Vec::new();
            let mut recall_before = None;
            for merged in [false, true] {
                if merged {
                    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
                    writer.merge(&index.searchable_segment_ids()?).wait()?;
                }
                let searcher = index.reader()?.searcher();
                let model = WorkModel::for_searcher(&searcher, fixture.field)?.unwrap();
                assert_eq!(model.n_avg, 4.0);
                for selected in [false, true] {
                    let filter: Box<dyn Query> = if selected {
                        Box::new(TermQuery::new(
                            Term::from_field_u64(keep, 1),
                            IndexRecordOption::Basic,
                        ))
                    } else {
                        Box::new(AllQuery)
                    };
                    for levels in [0, usize::MAX] {
                        let collector =
                            TopDocsByVectorSimilarity::new(fixture.field, rows[0].to_vec(), 2048)
                                .with_tie_break((
                                    SortByStaticFastValue::<u64>::for_field("ordinal"),
                                    Order::Asc,
                                ))
                                .with_max_scan_levels(levels)
                                .with_adaptive_params(AdaptiveProbeParams {
                                    max_probe_fraction: 0.005,
                                    min_probe_clusters: 1,
                                    work_model: Some(model),
                                    router_recall_target: 1.0,
                                    ..Default::default()
                                });
                        let result = searcher.search(filter.as_ref(), &collector)?;
                        let ids: Vec<_> = result
                            .results
                            .iter()
                            .map(|(_, address)| {
                                searcher
                                    .segment_reader(address.segment_ord)
                                    .fast_fields()
                                    .u64("ordinal")
                                    .unwrap()
                                    .first(address.doc_id)
                                    .unwrap()
                            })
                            .collect();
                        let budget: f32 = result.stats.iter().map(|s| s.work_budget).sum();
                        let work: f32 = result.stats.iter().map(|s| s.work_charged).sum();
                        let scored: usize = result.stats.iter().map(|s| s.candidates_scored).sum();
                        assert!((budget - 1.28).abs() < 1e-6, "{budget}");
                        assert!(scored > 0 && scored < 32, "{scored}");
                        assert_eq!(
                            result.stats.iter().filter(|s| s.work_budget > 0.0).count(),
                            1
                        );
                        if let Some(crate::vector::router::RouterMetrics::Rng(metrics)) =
                            result.stats[0].routing
                        {
                            assert!(metrics.visited_count < rows.len());
                            assert!(metrics.result_count < 8);
                        }
                        let snapshot = (
                            ids,
                            budget.to_bits(),
                            work.to_bits(),
                            scored,
                            result.stats[0].termination,
                        );
                        if merged {
                            assert_eq!(
                                snapshot,
                                before.remove(0),
                                "{kind:?} filtered={selected} levels={levels}"
                            );
                        } else {
                            before.push(snapshot);
                        }
                    }
                }
                if kind == RouterKind::Stacked {
                    let mut query = rows[0].to_vec();
                    query[0] = 0.1;
                    let result = searcher.search(
                        &AllQuery,
                        &TopDocsByVectorSimilarity::new(fixture.field, query, 1)
                            .with_max_scan_levels(0)
                            .with_adaptive_params(AdaptiveProbeParams {
                                max_probe_fraction: 1.0,
                                min_probe_clusters: 1,
                                recall_target: 0.5,
                                work_model: Some(model),
                                ..Default::default()
                            }),
                    )?;
                    let stats = &result.stats[0];
                    assert_eq!(
                        stats.termination,
                        crate::vector::backend::ProbeTermination::RecallTarget
                    );
                    assert!(stats.recall_estimate.is_some_and(|recall| recall >= 0.5));
                    assert_eq!(
                        result
                            .stats
                            .iter()
                            .filter(|s| s.recall_estimate.is_some())
                            .count(),
                        1
                    );
                    let snapshot = (
                        stats.work_charged.to_bits(),
                        stats.recall_estimate.map(f32::to_bits),
                        result
                            .stats
                            .iter()
                            .map(|s| s.candidates_scored)
                            .sum::<usize>(),
                    );
                    if let Some(before) = recall_before {
                        assert_eq!(snapshot, before);
                    } else {
                        recall_before = Some(snapshot);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn global_boundaries_prune_across_encodings_and_preserve_string_ties() -> crate::Result<()> {
        use crate::collector::sort_key::SortByString;
        use crate::query::AllQuery;
        use crate::schema::{FAST, STRING};
        use crate::vector::{
            TopDocsByVectorSimilarity, VectorQuantizationConfig, VectorQuantizationLayer,
        };
        use crate::Order;

        for exact_winner in [false, true] {
            let mut fixture = Fixture::new(Metric::L2);
            let mut schema = Schema::builder();
            schema.add_vector_field("embedding", VectorOptions::new(64, Metric::L2));
            let name = schema.add_text_field("name", FAST | STRING);
            fixture.schema = schema.build();
            fixture.replace_centroids(centroids(&[[0.0; 64]]));
            let mut index = fixture.create(RouterKind::Exact)?;
            for (batch, names) in [["z", "y"], ["b", "c"], ["a", "b"]].into_iter().enumerate() {
                index.settings_mut().vector_quantization = if exact_winner && batch == 2 {
                    Vec::new()
                } else {
                    vec![VectorQuantizationConfig::materialize(
                        "embedding".into(),
                        &VectorOptions::new(64, Metric::L2),
                        (0..batch + 1)
                            .map(|layer| VectorQuantizationLayer {
                                bits: 1,
                                seed: layer as u64 + 7,
                            })
                            .collect(),
                    )?]
                };
                let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
                writer.set_merge_policy(Box::new(NoMergePolicy));
                for name_value in names {
                    let mut row = [0.0; 64];
                    row[0] = if batch == 0 { 10.0 } else { 0.0 };
                    let mut doc = TantivyDocument::new();
                    doc.add_vector(fixture.field, &row);
                    doc.add_text(name, name_value);
                    writer.add_document(doc)?;
                }
                writer.commit()?;
                writer.wait_merging_threads()?;
            }
            let searcher = index.reader()?.searcher();
            let result = searcher.search(
                &AllQuery,
                &TopDocsByVectorSimilarity::new(fixture.field, vec![0.0; 64], 2)
                    .with_tie_break((SortByString::for_field("name"), Order::Asc)),
            )?;
            let names: Vec<_> = result
                .results
                .iter()
                .map(|(_, address)| {
                    let column = searcher
                        .segment_reader(address.segment_ord)
                        .fast_fields()
                        .str("name")
                        .unwrap()
                        .unwrap();
                    let mut name = String::new();
                    column
                        .ord_to_str(column.term_ords(address.doc_id).next().unwrap(), &mut name)
                        .unwrap();
                    name
                })
                .collect();
            assert_eq!(names, ["a", "b"]);
            assert_eq!(
                result
                    .stats
                    .iter()
                    .map(|s| s.candidates_scored)
                    .sum::<usize>(),
                6
            );
            let loser = result
                .stats
                .iter()
                .find(|s| {
                    s.layers.get(0).is_some_and(|layer| layer.scored() == 2) && s.rerank_rows == 0
                })
                .unwrap();
            assert_eq!(loser.quantized_trace.boundary_docs, [Vec::<u32>::new()]);
            assert_eq!(loser.rerank_io.reads, 0);
            assert_eq!(
                result.stats.iter().map(|s| s.rerank_rows).sum::<usize>(),
                if exact_winner { 2 } else { 4 }
            );
        }
        Ok(())
    }

    #[test]
    fn global_bounds_use_hits_from_other_segments() -> crate::Result<()> {
        use crate::query::AllQuery;
        use crate::vector::TopDocsByVectorSimilarity;

        let fixture = Fixture::new(Metric::L2);
        let index = fixture.create(RouterKind::Exact)?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for row in [[0.0_f32, 0.0], [3.0, 4.0]] {
            let mut doc = TantivyDocument::new();
            doc.add_vector(fixture.field, &row);
            writer.add_document(doc)?;
            writer.commit()?;
        }
        let result = index.reader()?.searcher().search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(fixture.field, vec![0.0_f32, 0.0], 1),
        )?;
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].0, 0.0);
        assert_eq!(
            result
                .stats
                .iter()
                .map(|s| s.candidates_scored)
                .sum::<usize>(),
            1
        );
        assert_eq!(
            result
                .stats
                .iter()
                .filter(|s| s.candidates_scored == 0 && s.bounds_skips > 0)
                .count(),
            1
        );
        Ok(())
    }

    #[test]
    fn shared_merges_preserve_memberships_rows_and_bounds() -> crate::Result<()> {
        use std::collections::BTreeMap;

        use crate::indexer::merger::IndexMerger;
        use crate::schema::{FAST, INDEXED};
        use crate::vector::storage_io::test_support::PagedDirectory;
        use crate::vector::{residual_norm, VectorQuantizationConfig, VectorQuantizationLayer};
        use crate::{IndexSettings, IndexSortByField, Order, SegmentReader, Term};

        const DIM: usize = 64;
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            for quantized in [false, true] {
                for mixed in [false, true] {
                    let mut fixture = Fixture::new(metric);
                    let mut schema = Schema::builder();
                    fixture.field =
                        schema.add_vector_field("embedding", VectorOptions::new(DIM, metric));
                    let ordinal = schema.add_u64_field("ordinal", FAST | INDEXED);
                    fixture.schema = schema.build();
                    fixture.replace_centroids(centroids(
                        &[[0.0, 0.0], [3.0, 4.0], [-4.0, 3.0]].map(|value| {
                            let mut row = [0.0; DIM];
                            row[..2].copy_from_slice(&value);
                            row
                        }),
                    ));
                    let settings = IndexSettings {
                        sort_by_field: Some(IndexSortByField {
                            field: "ordinal".into(),
                            order: Order::Asc,
                        }),
                        vector_quantization: if quantized {
                            vec![VectorQuantizationConfig::materialize(
                                "embedding".into(),
                                &VectorOptions::new(DIM, metric),
                                vec![VectorQuantizationLayer { bits: 1, seed: 7 }],
                            )?]
                        } else {
                            Vec::new()
                        },
                        ..Default::default()
                    };
                    let directory = PagedDirectory::default();
                    let shared = fixture
                        .builder()
                        .settings(settings.clone())
                        .ivf_router(RouterKind::Exact)?
                        .create(directory.clone())?;
                    let flat = Index::builder()
                        .schema(fixture.schema.clone())
                        .settings(settings.clone())
                        .create_in_ram()?;
                    let mut segments = Vec::new();
                    for (index, batches) in [
                        (
                            &shared,
                            vec![
                                vec![
                                    (4, Some([3.0, 4.0])),
                                    (0, Some([0.0, 0.0])),
                                    (7, Some([30.0, 40.0])),
                                    (1, None),
                                ],
                                vec![
                                    (6, Some([-4.0, 3.0])),
                                    (3, Some([0.1, 0.2])),
                                    (8, Some([-20.0, 15.0])),
                                ],
                            ],
                        ),
                        (
                            &flat,
                            vec![vec![
                                (5, Some([2.0, 2.0])),
                                (2, Some([0.3, -0.7])),
                                (9, None),
                                (10, Some([4.0, 4.0])),
                            ]],
                        ),
                    ] {
                        if !mixed && std::ptr::eq(index, &flat) {
                            continue;
                        }
                        let mut writer: IndexWriter =
                            index.writer_with_num_threads(1, 15_000_000)?;
                        writer.set_merge_policy(Box::new(NoMergePolicy));
                        for batch in batches {
                            for (id, row) in batch {
                                let mut doc = TantivyDocument::new();
                                doc.add_u64(ordinal, id);
                                if let Some(value) = row {
                                    let mut row = [0.0; DIM];
                                    row[..2].copy_from_slice(&value);
                                    doc.add_vector(fixture.field, &row);
                                }
                                writer.add_document(doc)?;
                            }
                            writer.commit()?;
                        }
                        for id in [7, 8, 10] {
                            writer.delete_term(Term::from_field_u64(ordinal, id));
                        }
                        writer.commit()?;
                        segments.extend(index.searchable_segments()?);
                    }
                    let target_index = Index::open(directory.clone())?;
                    let merger = IndexMerger::open(
                        fixture.schema.clone(),
                        settings,
                        &segments,
                        Box::new(|| false),
                        false,
                    )?;
                    let centroid_bytes = target_index.cached_centroid_index()?.unwrap()
                        [&fixture.field]
                        .centroid_bytes()?;
                    let centroids = decode_row::<f32>(&centroid_bytes, 3 * DIM)?;
                    let mut expected = BTreeMap::new();
                    let mut bounds = crate::vector::BoundsBuilder::new(3);
                    let mut expected_reads = Vec::new();
                    for segment in &merger.readers {
                        let vectors = segment.vector_index(fixture.field)?;
                        let ordinals = segment.fast_fields().u64("ordinal")?;
                        directory.reads.lock().unwrap().clear();
                        if let Some(ivf) = vectors.index() {
                            for cluster in 0..ivf.num_clusters() {
                                bounds.add_native(cluster, ivf.bounds().ball_r(cluster));
                                vectors.read_doc_ids(cluster, &mut Vec::new())?;
                            }
                        }
                        expected_reads.extend(
                            directory
                                .reads
                                .lock()
                                .unwrap()
                                .drain(..)
                                .map(|(_, range)| range),
                        );
                        for doc in segment.doc_ids_alive() {
                            let id = ordinals.first(doc).unwrap();
                            let row = vectors.row_id(doc)?;
                            let value = if let Some(row) = row {
                                directory.reads.lock().unwrap().clear();
                                let bytes = vectors.vector_bytes_for_row(row)?;
                                expected_reads.extend(
                                    directory
                                        .reads
                                        .lock()
                                        .unwrap()
                                        .drain(..)
                                        .map(|(_, range)| range),
                                );
                                let values = decode_row::<f32>(&bytes, DIM)?;
                                let cluster = vectors.row_cluster(row).unwrap_or_else(|| {
                                    (0..3)
                                        .max_by(|&a, &b| {
                                            metric
                                                .similarity(
                                                    &values,
                                                    &centroids[a * DIM..(a + 1) * DIM],
                                                )
                                                .cmp(&metric.similarity(
                                                    &values,
                                                    &centroids[b * DIM..(b + 1) * DIM],
                                                ))
                                                .then_with(|| b.cmp(&a))
                                        })
                                        .unwrap()
                                });
                                if vectors.index().is_none() {
                                    bounds.add_native(
                                        cluster,
                                        residual_norm::<f32>(
                                            &bytes,
                                            &centroids[cluster * DIM..(cluster + 1) * DIM],
                                        ),
                                    );
                                }
                                Some((bytes, cluster))
                            } else {
                                None
                            };
                            expected.insert(id, value);
                        }
                    }
                    directory.reads.lock().unwrap().clear();
                    let target = target_index.new_segment();
                    let num_docs = merger.write(&target)?;
                    let mut actual_reads: Vec<_> = directory
                        .reads
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|(_, range)| range.clone())
                        .collect();
                    actual_reads.sort_by_key(|range| (range.start, range.end));
                    expected_reads.sort_by_key(|range| (range.start, range.end));
                    assert_eq!(
                        actual_reads, expected_reads,
                        "clustered rows must be read only for encoding"
                    );
                    let merged = SegmentReader::open(&target.with_max_doc(num_docs))?;
                    let vectors = merged.vector_index(fixture.field)?;
                    assert_eq!(vectors.index().unwrap().bounds().values(), bounds.finish());
                    assert_eq!(num_docs as usize, expected.len());
                    for (doc, (id, expected)) in expected.into_iter().enumerate() {
                        let doc = doc as u32;
                        assert_eq!(merged.fast_fields().u64("ordinal")?.first(doc), Some(id));
                        match expected {
                            None => assert!(vectors.vector_bytes(doc)?.is_none()),
                            Some((bytes, cluster)) => {
                                assert_eq!(vectors.vector_bytes(doc)?.unwrap(), bytes);
                                let row = vectors.row_id(doc)?.unwrap();
                                assert_eq!(vectors.row_cluster(row), Some(cluster));
                                if let Some(quant) = vectors.quantization() {
                                    let context = quant.index_ctx();
                                    let prepared = cascade::prepare_centroid(
                                        &centroids[cluster * DIM..(cluster + 1) * DIM],
                                        &context.specs,
                                    );
                                    let mut values = decode_row::<f32>(&bytes, DIM)?;
                                    let mut workspace = cascade::BatchEncodeWorkspace::new();
                                    let encoded = cascade::encode_batch_in_place_with_workspace(
                                        &mut values,
                                        1,
                                        &prepared,
                                        &context.specs,
                                        &context.grids,
                                        &mut workspace,
                                        metric == Metric::L2,
                                    );
                                    assert_eq!(
                                        quant.residual_norm(row)?,
                                        encoded.residual_norms_squared[0]
                                    );
                                    for (layer, encoded) in
                                        quant.layers().iter().zip(&encoded.layers)
                                    {
                                        assert_eq!(
                                            layer.code_bytes(row)?.as_slice(),
                                            encoded.codes
                                        );
                                        assert_eq!(layer.scale(row)?, encoded.scales[0]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn shared_merge_rejects_foreign_and_segment_local_centroids() -> crate::Result<()> {
        use crate::indexer::merger::IndexMerger;
        use crate::vector::tests::Grid2DClusterer;

        let fixture = Fixture::new(Metric::L2);
        let target_index = fixture.create(RouterKind::Exact)?;
        for local in [false, true] {
            let source = if local {
                Index::builder()
                    .schema(fixture.schema.clone())
                    .settings(crate::IndexSettings {
                        vector_clustering_threshold: 1,
                        ..Default::default()
                    })
                    .ivf_clusterer(Arc::new(Grid2DClusterer {
                        centroids: vec![[0.0, 0.0], [3.0, 4.0], [-4.0, 3.0]],
                    }))
                    .ivf_router(RouterKind::Exact)?
                    .create_in_ram()?
            } else {
                fixture
                    .builder()
                    .ivf_router(RouterKind::Exact)?
                    .create_in_ram()?
            };
            let mut writer: IndexWriter = source.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            let mut doc = TantivyDocument::new();
            doc.add_vector(fixture.field, &[3.0, 4.0]);
            writer.add_document(doc)?;
            writer.commit()?;
            if local {
                writer.merge(&source.searchable_segment_ids()?).wait()?;
            }
            let segments = source.searchable_segments()?;
            assert_eq!(
                crate::SegmentReader::open(&segments[0])?
                    .vector_index(fixture.field)?
                    .index()
                    .unwrap()
                    .num_clusters(),
                3
            );
            let merger = IndexMerger::open(
                fixture.schema.clone(),
                target_index.settings().clone(),
                &segments,
                Box::new(|| false),
                false,
            )?;
            let error = merger.write(&target_index.new_segment()).unwrap_err();
            assert!(
                error.to_string().contains("different centroid index"),
                "{error}"
            );
        }
        Ok(())
    }

    #[test]
    fn invalid_producers_do_not_publish_metadata() -> crate::Result<()> {
        for (name, values, rows, dims) in [
            ("empty", vec![], 0, 2),
            ("shape", vec![1.0], 1, 2),
            ("dimensions", vec![1.0], 1, 1),
            ("overflow", vec![], usize::MAX, 2),
            ("row count", vec![], u32::MAX as usize, 2),
            ("NaN", vec![f32::NAN, 1.0], 1, 2),
            ("infinity", vec![1.0, f32::INFINITY], 1, 2),
            ("negative infinity", vec![f32::NEG_INFINITY, 1.0], 1, 2),
        ] {
            for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
                let mut fixture = Fixture::new(metric);
                let valid = fixture.replace_centroids(IvfCentroids::F32(IvfMatrix {
                    values: values.clone(),
                    rows,
                    dims,
                }));
                assert!(
                    matches!(
                        fixture.create(RouterKind::Exact),
                        Err(TantivyError::InvalidArgument(_))
                    ),
                    "{name}: {metric:?}"
                );
                assert!(!Index::exists(&fixture.directory)?);
                fixture.replace_centroids(valid);
                assert!(fixture
                    .create(RouterKind::Exact)?
                    .cached_centroid_index()?
                    .is_some());
            }
        }
        let mut fixture = Fixture::new(Metric::L2);
        let mut schema = Schema::builder();
        schema.add_vector_field("zero", VectorOptions::new(0, Metric::L2));
        fixture.schema = schema.build();
        fixture.replace_centroids(centroids(&[[]]));
        assert!(fixture.create(RouterKind::Exact).is_err());
        Ok(())
    }

    #[test]
    fn producer_requires_vector_fields_and_explicit_router() -> crate::Result<()> {
        let fixture = Fixture::new(Metric::L2);
        assert!(fixture
            .builder()
            .create_in_ram()
            .unwrap_err()
            .to_string()
            .contains("explicitly configured Router"));
        assert!(fixture
            .builder()
            .schema(Schema::builder().build())
            .ivf_router(RouterKind::Exact)?
            .create_in_ram()
            .unwrap_err()
            .to_string()
            .contains("vector field"));
        assert_eq!(fixture.producer.calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[test]
    fn provider_failure_keeps_existing_index_and_allows_retry() -> crate::Result<()> {
        let fixture = Fixture::new(Metric::L2);
        let index = fixture.create(RouterKind::Exact)?;
        let original_meta = fixture.directory.atomic_read(&META_FILEPATH)?;
        let original = artifact(&index)?;
        assert!(fixture
            .builder()
            .centroid_producer(Arc::new(TestProducer::default()))
            .ivf_router(RouterKind::Exact)?
            .create(fixture.directory.clone())
            .is_err());
        assert_eq!(
            fixture.directory.atomic_read(&META_FILEPATH)?,
            original_meta
        );
        assert_eq!(
            artifact(&Index::open(fixture.directory.clone())?)?,
            original
        );
        let replacement = fixture.create(RouterKind::Exact)?;
        assert_ne!(
            replacement.load_metas()?.centroid_index.as_ref(),
            Some(&original.0)
        );
        assert!(replacement.cached_centroid_index()?.is_some());
        Ok(())
    }

    #[test]
    fn open_or_create_preserves_centroids_and_legacy_metadata() -> crate::Result<()> {
        let fixture = Fixture::new(Metric::L2);
        let index = fixture
            .builder()
            .ivf_router(RouterKind::Exact)?
            .open_or_create(fixture.directory.clone())?;
        let centroid_meta: Option<crate::CentroidIndexMeta> = index.load_metas()?.centroid_index;
        let stored_meta: serde_json::Value =
            serde_json::from_slice(&fixture.directory.atomic_read(&META_FILEPATH)?)?;
        assert_eq!(
            stored_meta["centroid_index"],
            serde_json::json!({"file_name": centroid_meta.as_ref().unwrap().file_name})
        );
        let reopened = fixture
            .builder()
            .ivf_router(RouterKind::Rng)?
            .open_or_create(fixture.directory.clone())?;
        assert_eq!(fixture.producer.calls.load(Ordering::SeqCst), 1);
        assert_eq!(reopened.load_metas()?.centroid_index, centroid_meta);
        assert_eq!(
            reopened.cached_centroid_index()?.unwrap()[&fixture.field].router(),
            RouterKind::Exact
        );
        let reopened = Index::builder()
            .schema(fixture.schema.clone())
            .open_or_create(fixture.directory.clone())?;
        assert_eq!(reopened.load_metas()?.centroid_index, centroid_meta);

        let legacy = RamDirectory::create();
        let index = Index::builder()
            .schema(fixture.schema.clone())
            .create(legacy.clone())?;
        assert!(index.cached_centroid_index()?.is_none());
        assert!(!String::from_utf8(legacy.atomic_read(&META_FILEPATH)?)
            .unwrap()
            .contains("centroid_index"));
        assert!(Index::open(legacy.clone())?
            .cached_centroid_index()?
            .is_none());
        assert!(fixture
            .builder()
            .ivf_router(RouterKind::Exact)?
            .open_or_create(legacy)
            .is_err());
        assert_eq!(fixture.producer.calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn centroid_artifact_survives_commits_merges_rollback_deletes_and_gc() -> crate::Result<()> {
        let fixture = Fixture::new(Metric::L2);
        let index = fixture.create(RouterKind::Exact)?;
        let original = artifact(&index)?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for value in [1.0, 2.0] {
            let mut doc = TantivyDocument::new();
            doc.add_vector(fixture.field, &[value, 0.0]);
            writer.add_document(doc)?;
            writer.commit()?;
        }
        writer.merge(&index.searchable_segment_ids()?).wait()?;
        writer.garbage_collect_files().wait()?;
        let reader = index.reader()?;
        assert_eq!(reader.searcher().num_docs(), 2);
        assert_eq!(
            reader.searcher().segment_readers()[0]
                .vector_index(fixture.field)?
                .info()
                .unwrap()
                .format,
            VectorStorageFormat::Ivf
        );
        writer.delete_all_documents()?;
        writer.rollback()?;
        assert_eq!(
            Index::open(fixture.directory.clone())?
                .reader()?
                .searcher()
                .num_docs(),
            2
        );
        writer.delete_all_documents()?;
        writer.commit()?;
        writer.garbage_collect_files().wait()?;
        drop(writer);
        let reopened = Index::open(fixture.directory.clone())?;
        assert_eq!(reopened.reader()?.searcher().num_docs(), 0);
        assert_eq!(artifact(&reopened)?, original);
        assert!(reopened.cached_centroid_index()?.is_some());
        assert!(reopened.validate_checksum()?.is_empty());
        assert_eq!(fixture.producer.calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn single_segment_finalization_retains_centroid_artifact() -> crate::Result<()> {
        for remap in [false, true] {
            let fixture = Fixture::new(Metric::L2);
            let mut writer = fixture
                .builder()
                .settings(crate::IndexSettings {
                    manual_doc_id_mapping: remap,
                    ..Default::default()
                })
                .ivf_router(RouterKind::Exact)?
                .single_segment_index_writer(fixture.directory.clone(), 15_000_000)?;
            for value in [1.0, 2.0] {
                let mut doc = TantivyDocument::new();
                doc.add_vector(fixture.field, &[value, 0.0]);
                writer.add_document(doc)?;
            }
            let index = if remap {
                writer.finalize_with_doc_id_mapping(&DocIdMapping::new_permutation(vec![1, 0])?)?
            } else {
                writer.finalize()?
            };
            let centroid_meta = index.load_metas()?.centroid_index.unwrap();
            assert!(index.directory().exists(&centroid_meta.file_name)?);
            let reopened = Index::open(fixture.directory)?;
            assert_eq!(reopened.reader()?.searcher().num_docs(), 2);
            assert!(reopened.cached_centroid_index()?.is_some());
            let vectors =
                reopened.reader()?.searcher().segment_readers()[0].vector_index(fixture.field)?;
            for doc in 0..2 {
                let expected = if remap { 2 - doc } else { doc + 1 } as f32;
                assert_eq!(
                    decode_row::<f32>(&vectors.vector_bytes(doc)?.unwrap(), 2)?,
                    [expected, 0.0]
                );
            }
            assert_eq!(fixture.producer.calls.load(Ordering::SeqCst), 1);
        }
        Ok(())
    }

    #[test]
    fn missing_artifact_fails_reopen_but_clones_keep_cached_readers() -> crate::Result<()> {
        let fixture = Fixture::new(Metric::L2);
        let index = fixture.create(RouterKind::Exact)?;
        let cache = index.cached_centroid_index()?.unwrap();
        let centroid_meta = index.load_metas()?.centroid_index.unwrap();
        fixture.directory.delete(&centroid_meta.file_name).unwrap();
        assert!(Index::open(fixture.directory).is_err());
        let clone = index.clone();
        let cached = std::thread::spawn(move || clone.cached_centroid_index().unwrap().unwrap())
            .join()
            .unwrap();
        assert!(Arc::ptr_eq(&cache, &cached));
        assert_eq!(cached[&fixture.field].centroid_bytes()?.len(), 24);
        Ok(())
    }

    #[test]
    fn checksum_validation_includes_centroid_artifact() -> crate::Result<()> {
        let fixture = Fixture::new(Metric::L2);
        let index = fixture.create(RouterKind::Exact)?;
        let centroid_meta = index.load_metas()?.centroid_index.unwrap();
        let path = &centroid_meta.file_name;
        assert!(index.validate_checksum()?.is_empty());
        let mut raw = fixture.directory.atomic_read(path)?;
        raw[0] ^= 1;
        fixture.directory.atomic_write(path, &raw)?;
        assert_eq!(
            index.validate_checksum()?,
            std::collections::HashSet::from([path.to_path_buf()])
        );
        Ok(())
    }

    #[test]
    fn shared_segments_validate_artifact_identity_and_posting_slots() -> crate::Result<()> {
        use crate::directory::FileSlice;
        use crate::index::SegmentComponent;
        use crate::vector::header::read_centroid_header;
        use crate::vector::ivf::SharedSegmentMeta;

        let fixture = Fixture::new(Metric::L2);
        let index = fixture.create(RouterKind::Exact)?;
        {
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            let mut doc = TantivyDocument::new();
            doc.add_vector(fixture.field, &[1.0, 2.0]);
            writer.add_document(doc)?;
            writer.commit()?;
        }
        let path = index.searchable_segments()?[0]
            .relative_path(SegmentComponent::Custom("centroids".into()));
        let original = index.directory().open_read(&path)?.read_bytes()?;
        let (_, body) = read_centroid_header(&FileSlice::from(original.to_vec()))?;
        let composite = CompositeFile::open(&body)?;
        let slots = [0, 1, 3].map(|slot| {
            (
                slot,
                composite
                    .open_read_with_idx(fixture.field, slot)
                    .unwrap()
                    .read_bytes()
                    .unwrap(),
            )
        });
        let foreign = Fixture::new(Metric::L2).create(RouterKind::Exact)?;
        let foreign_meta = serde_json::to_vec(&SharedSegmentMeta {
            centroid_index: foreign.centroid_index_meta().unwrap().clone(),
            num_docs: 1,
        })?;
        for (name, changed, data) in [
            (
                "different artifact with the same centroid count",
                0,
                Some(foreign_meta),
            ),
            ("missing metadata", 0, None),
            ("missing offsets", 1, None),
            ("missing bounds", 3, None),
            ("router must be shared", 2, Some(vec![2])),
            ("short offsets", 1, Some(vec![0; 8])),
            ("short bounds", 3, Some(vec![0])),
        ] {
            let mut bytes = original[..4].to_vec();
            let mut writer = CompositeWrite::wrap(&mut bytes);
            for slot in 0..4 {
                let payload = if slot == changed {
                    data.as_deref()
                } else {
                    slots
                        .iter()
                        .find(|(s, _)| *s == slot)
                        .map(|(_, bytes)| bytes.as_slice())
                };
                if let Some(payload) = payload {
                    writer
                        .for_field_with_idx(fixture.field, slot)
                        .write_all(payload)?;
                }
            }
            writer.close()?;
            fixture.directory.delete(&path).unwrap();
            let mut writer = index.directory().open_write(&path)?;
            writer.write_all(&bytes)?;
            writer.terminate()?;
            let reopened = Index::open(fixture.directory.clone())?;
            assert!(
                reopened.reader()?.searcher().segment_readers()[0]
                    .vector_index(fixture.field)
                    .is_err(),
                "{name}"
            );
        }
        Ok(())
    }

    #[test]
    fn malformed_artifacts_fail_reopen() -> crate::Result<()> {
        let fixture = Fixture::new(Metric::L2);
        let index = fixture.create(RouterKind::Exact)?;
        let (meta, original) = artifact(&index)?;
        let composite = CompositeFile::open(&crate::directory::FileSlice::from(
            original[HEADER_LEN..].to_vec(),
        ))?;
        let slots: Vec<_> = (META..=ROUTER)
            .map(|slot| {
                composite
                    .open_read_with_idx(fixture.field, slot)
                    .unwrap()
                    .read_bytes()
                    .unwrap()
            })
            .collect();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&slots[META])?,
            serde_json::json!({"num_centroids": 3})
        );
        let mut cases: Vec<_> = (0..HEADER_LEN + 5)
            .map(|len| ("truncated", original[..len].to_vec()))
            .collect();
        for (name, offset, value) in [("magic", 0, *b"BAD!"), ("version", 4, 2u32.to_le_bytes())] {
            let mut bytes = original.to_vec();
            bytes[offset..offset + 4].copy_from_slice(&value);
            cases.push((name, bytes));
        }
        let mutations = [
            ("metadata", META, Some(b"{}".to_vec())),
            (
                "zero centroids",
                META,
                Some(br#"{"num_centroids":0}"#.to_vec()),
            ),
            (
                "row length",
                CENTROIDS,
                Some(slots[CENTROIDS][1..].to_vec()),
            ),
            ("router tag", ROUTER, Some(vec![255])),
            ("empty router", ROUTER, Some(vec![])),
            ("missing router", ROUTER, None),
        ];
        for (name, changed_slot, replacement) in mutations {
            let mut bytes = original[..HEADER_LEN].to_vec();
            let mut write = CompositeWrite::wrap(&mut bytes);
            for (slot, data) in slots.iter().enumerate() {
                let data = if slot == changed_slot {
                    replacement.as_deref()
                } else {
                    Some(data.as_slice())
                };
                if let Some(data) = data {
                    write
                        .for_field_with_idx(fixture.field, slot)
                        .write_all(data)?;
                }
            }
            write.close()?;
            cases.push((name, bytes));
        }
        for (name, bytes) in cases {
            fixture.directory.delete(&meta.file_name).unwrap();
            let mut write = index.directory().open_write(&meta.file_name)?;
            write.write_all(&bytes)?;
            write.terminate()?;
            assert!(Index::open(fixture.directory.clone()).is_err(), "{name}");
        }
        Ok(())
    }
}
