//! IVF merge tests: the written layout, quantized encoding, and searches over merged segments.

use std::io::Write;
use std::sync::Arc;

use common::TerminatingWrite;

use super::*;
use crate::index::{IndexSettings, SegmentComponent};
use crate::indexer::NoMergePolicy;
use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
use crate::vector::clustering::{for_each_batch, IvfTrainingVectors, VectorSource};
use crate::vector::diagnostics::{
    VectorEstimatorMeasurements, VectorEstimatorQuery, VectorEstimatorSource,
};
use crate::vector::distance::maybe_normalize_bytes;
use crate::vector::params::{LocatedPath, ProbeParams};
use crate::vector::search::prepared::{QuantizedQueryCtx, VectorQuery};
use crate::vector::storage::blocks::block_len;
use crate::vector::storage::header::{write_vector_header, VectorEntry, VEC_EXT};
use crate::vector::storage::metadata::SlotType;
use crate::vector::test_support::{Fixture, FixtureSpec, Lcg};
use crate::vector::tests::{ground_truth, routed_exhaustive_params};
use crate::vector::{
    BlockAddress, TopDocsByVectorSimilarityBuilder, VectorQuantizationLayer, ENTRY_ALIGN,
};
use crate::{DocAddress, Index, IndexWriter, TantivyDocument};

/// The merge's production sources carry no estimator or audit entry point.
#[test]
fn quantization_merge_source_has_no_estimator_analysis_entrypoint() {
    let production_source = [
        include_str!("mod.rs"),
        include_str!("source.rs"),
        include_str!("train.rs"),
        include_str!("assign.rs"),
        include_str!("encode.rs"),
        include_str!("write.rs"),
    ]
    .concat();
    for forbidden in [
        "build_grid(",
        "audit_prefix_error_model(",
        "prepare_fp_query(",
        "audit_error",
        "diagnostic_error",
        "VectorEstimatorMeasurements",
        "VectorEstimatorQuery",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "quantization merge production source contains forbidden analysis hook {forbidden}"
        );
    }
}

#[test]
fn merge_runtime_uses_persisted_grid_and_rho() {
    let config = quant_fixture_config_for(100, Metric::Dot, &[1, 4]);
    let (_, resolved) =
        quantization_runtime(&config, &VectorOptions::new(100, Metric::Dot)).unwrap();
    assert_eq!(resolved.len(), 2);
    for (layer, grid) in resolved.iter().enumerate() {
        let persisted = config
            .grids
            .iter()
            .find(|grid| grid.bits == config.layers[layer].bits)
            .unwrap();
        assert_eq!(grid.bits, persisted.bits);
        if grid.bits != 1 {
            assert_eq!(grid.points, persisted.points);
        } else {
            assert!(grid.points.is_empty());
        }
        assert_eq!(grid.rho_model, persisted.rho_model);
    }
}

fn block_rows(
    index: &crate::vector::storage::centroids::ClusterIndex,
) -> crate::Result<Vec<(usize, usize)>> {
    let mut rows = Vec::new();
    for block in 0..index.num_clusters() {
        rows.extend((0..index.cluster_num_rows(block)?).map(|row| (block, row)));
    }
    Ok(rows)
}

// Mixed field contracts retain exact entry lengths and aligned Data boundaries.
fn check_two_field_entries(empty_second: bool) -> crate::Result<()> {
    use common::{HasLen, OwnedBytes};

    use crate::directory::{CompositeFile, FileHandle, FileSlice};
    use crate::vector::storage::blocks::{block_align, data_entry_len, Blocks};
    use crate::vector::storage::header::{read_centroid_header, read_vector_header};
    #[derive(Debug)]
    struct ByteAddressed(Vec<u8>);
    impl HasLen for ByteAddressed {
        fn len(&self) -> usize {
            self.0.len()
        }
    }
    impl FileHandle for ByteAddressed {
        fn read_bytes(&self, range: std::ops::Range<usize>) -> std::io::Result<OwnedBytes> {
            Ok(OwnedBytes::new(self.0[range].to_vec()))
        }
        fn storage_block_len(&self) -> Option<usize> {
            Some(1)
        }
    }
    let opts = VectorOptions::new(65, Metric::L2);
    let mut schema = Schema::builder();
    let ordinal = schema.add_u64_field("ordinal", crate::schema::FAST);
    let plain = schema.add_vector_field("plain", opts.clone());
    let quant = schema.add_vector_field("quant", opts.clone());
    let config = VectorQuantizationConfig::materialize(
        "quant".into(),
        &opts,
        vec![VectorQuantizationLayer { bits: 1, seed: 17 }],
    )?;
    let index = Index::builder()
        .schema(schema.build())
        .settings(IndexSettings {
            vector_clustering_threshold: 1,
            vector_quantization: vec![config],
            ..Default::default()
        })
        .ivf_clusterer(Arc::new(QuantFixtureClusterer {
            dim: 65,
            metric: Metric::L2,
        }))
        .create_in_ram()?;
    let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for doc in 0..7 {
        let values = vec![doc as f32 / 7.0; 65];
        let mut document = TantivyDocument::new();
        document.add_u64(ordinal, doc);
        document.add_vector(plain, &values);
        if !empty_second {
            document.add_vector(quant, &values);
        }
        writer.add_document(document)?;
        if doc == 2 {
            writer.commit()?;
        }
    }
    writer.commit()?;
    writer.merge(&index.searchable_segment_ids()?).wait()?;
    writer.wait_merging_threads()?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let segment = &searcher.segment_readers()[0];
    let file = segment.open_read(SegmentComponent::Custom(VEC_EXT.into()))?;
    let file = FileSlice::new(Arc::new(ByteAddressed(file.read_bytes()?.to_vec())));
    let (_, body) = read_vector_header(&file)?;
    let composite = CompositeFile::open(&body)?;
    let mut data_ends = Vec::new();
    let mut id_starts = Vec::new();
    let mut aligns = Vec::new();
    for (field, count) in [(plain, 7), (quant, if empty_second { 0 } else { 7 })] {
        let vector = segment.vector_index(field)?;
        assert!(vector.metadata().is_some());
        assert!(!vector.id_map_initialized());
        let hits = searcher.search(
            &AllQuery,
            &TopDocsByVectorSimilarityBuilder::new(field, vec![0.25; 65], 3)
                .with_probe_params(ProbeParams {
                    max_probe: usize::MAX,
                    min_probe_clusters: 2,
                    ..Default::default()
                })
                .build_for_searcher(&searcher, &AllQuery)?,
        )?;
        assert_eq!(hits.results.len(), count.min(3));
        assert!(!vector.id_map_initialized(), "scan must not open IdMap");
        let ivf = vector.index().unwrap();
        for b in 0..ivf.num_clusters() {
            let range = 0..ivf.cluster_num_rows(b)?;
            let docs = vector.cluster_doc_ids(b).unwrap().unwrap();
            assert!(docs.windows(2).all(|pair| pair[0] < pair[1]));
            for (row, doc) in range.zip(docs) {
                assert_eq!(
                    vector.row_location(doc)?,
                    Some(BlockAddress::new(b as u32, row as u32))
                );
                let bytes = vector.cluster_row_bytes(b, row, doc)?;
                assert_eq!(Some(bytes.clone()), vector.vector_bytes(doc)?);
                assert_eq!(
                    decode_row::<f32>(&bytes, 65)?,
                    vec![
                        segment.fast_fields().u64("ordinal")?.first(doc).unwrap() as f32 / 7.0;
                        65
                    ]
                );
            }
        }
        if count == 0 {
            for doc in 0..segment.max_doc() {
                assert!(vector.vector_bytes(doc)?.is_none());
            }
        }
        let data = composite
            .open_read_with_idx(field, VectorEntry::Data.index())
            .unwrap();
        let id_map = composite
            .open_read_with_idx(field, VectorEntry::IdMap.index())
            .unwrap();
        assert_eq!(id_map.len(), 1 + segment.max_doc() as usize * 8);
        // A quantized field declares the exact-row entries, a plain field exactly the other two.
        let mut entries: Vec<_> = composite
            .field_indices()
            .filter_map(|(f, idx)| (f == field).then_some(idx))
            .collect();
        entries.sort_unstable();
        let quantized = vector.has_exact_rows();
        assert_eq!(
            entries,
            if quantized {
                vec![0, 1, 2, 3]
            } else {
                vec![0, 1]
            }
        );
        // Row entries start and end 8-aligned and precede every addressing entry.
        for entry in [VectorEntry::Data, VectorEntry::ExactRows] {
            let Some(rows) = composite.open_read_with_idx(field, entry.index()) else {
                continue;
            };
            let start = rows.storage_block_ord(0).unwrap();
            assert_eq!(start % ENTRY_ALIGN, 0);
            assert_eq!(rows.len() % ENTRY_ALIGN, 0);
            data_ends.push(start + rows.len());
        }
        for entry in [VectorEntry::IdMap, VectorEntry::ExactIdMap] {
            if let Some(map) = composite.open_read_with_idx(field, entry.index()) {
                id_starts.push(map.storage_block_ord(0).unwrap());
            }
        }
        let centroid_file = segment.open_read(SegmentComponent::Custom(CENTROIDS_EXT.into()))?;
        let (_, centroid_body) = read_centroid_header(&centroid_file)?;
        let centroid_composite = CompositeFile::open(&centroid_body)?;
        assert!(centroid_composite.open_read_with_idx(field, 1).is_none());
        let blocks = Blocks::open(data.clone(), &opts, true)?;
        assert_eq!(
            data.len(),
            data_entry_len(
                data.len() - 16 - (blocks.num_blocks() + 1) * 16,
                blocks.num_blocks()
            )
        );
        if quantized {
            let exact = composite
                .open_read_with_idx(field, VectorEntry::ExactRows.index())
                .unwrap();
            let exact_blocks = Blocks::open(exact.clone(), &opts, false)?;
            assert_eq!(exact_blocks.num_rows(), count);
            assert_eq!(exact_blocks.num_blocks(), usize::from(count > 0));
        }
        aligns.push(block_align(&blocks.schema));
    }
    assert_ne!(aligns[0], aligns[1]);
    assert_eq!(data_ends.iter().max(), id_starts.iter().min());
    assert!(data_ends
        .iter()
        .all(|end| id_starts.iter().all(|start| end <= start)));
    Ok(())
}

// Empty quantized IVF fields still carry exact metadata-only Data and DocLocations IdMap
// entries.
#[test]
fn two_field_ivf_includes_empty_field() -> crate::Result<()> {
    check_two_field_entries(true)
}

// Plain and SignPlane blocks use different element sizes in one composite file.
#[test]
fn mixed_plain_quantized_entries_have_exact_lengths() -> crate::Result<()> {
    check_two_field_entries(false)
}

// Block traversal preserves vectors and assignments while dropping deleted documents.
#[test]
fn clustered_merge_preserves_vectors_assignments_and_missing_docs() -> crate::Result<()> {
    use std::collections::BTreeMap;

    use crate::schema::Value;
    for quantized in [false, true] {
        let index = build_quantized_fixture(64, quantized)?;
        let field = index.schema().get_field("embedding")?;
        let label = index.schema().get_field("label")?;
        let original = index.searchable_segment_ids()?;
        let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for i in 0..8 {
            let mut doc = TantivyDocument::new();
            doc.add_text(label, format!("extra{i}"));
            if i != 2 {
                doc.add_vector(field, &fixture_vector(Metric::L2, 64, i));
            }
            writer.add_document(doc)?;
            if i == 3 {
                writer.commit()?;
            }
        }
        writer.commit()?;
        let additions: Vec<_> = index
            .searchable_segment_ids()?
            .into_iter()
            .filter(|id| !original.contains(id))
            .collect();
        writer.merge(&additions).wait()?;
        writer.delete_term(Term::from_field_text(label, "d1"));
        writer.delete_term(Term::from_field_text(label, "extra5"));
        writer.commit()?;
        let capture = || -> crate::Result<_> {
            let searcher = index.reader()?.searcher();
            let mut out = BTreeMap::new();
            for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
                let vectors = segment.vector_index(field)?;
                let ivf = vectors.index().expect("IVF source");
                let mut rows_by_doc = BTreeMap::new();
                for cluster in 0..ivf.num_clusters() {
                    let docs = vectors.cluster_doc_ids(cluster).unwrap().unwrap();
                    for (row, doc) in (0..ivf.cluster_num_rows(cluster)?).zip(docs) {
                        rows_by_doc.insert(
                            doc,
                            (
                                cluster,
                                vectors.cluster_row_bytes(cluster, row, doc)?.to_vec(),
                            ),
                        );
                    }
                }
                for doc in 0..segment.max_doc() {
                    if segment
                        .alive_bitset()
                        .is_some_and(|alive| !alive.is_alive(doc))
                    {
                        continue;
                    }
                    let stored: TantivyDocument =
                        searcher.doc(crate::DocAddress::new(segment_ord as u32, doc))?;
                    let name = stored
                        .get_first(label)
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .to_owned();
                    out.insert(name, rows_by_doc.remove(&doc));
                }
                assert!(!vectors.id_map_initialized());
            }
            Ok(out)
        };
        let before = capture()?;
        assert_eq!(before.get("extra2"), Some(&None));
        writer.merge(&index.searchable_segment_ids()?).wait()?;
        writer.wait_merging_threads()?;
        assert_eq!(capture()?, before);
    }
    Ok(())
}

const QUANT_FIXTURE_DIM: usize = 64;

struct QuantFixtureClusterer {
    dim: usize,
    metric: Metric,
}

impl IvfClusterer for QuantFixtureClusterer {
    fn training_sample_ratio(&self) -> f32 {
        0.5
    }

    fn train(
        &self,
        options: &VectorOptions,
        _source: &mut dyn VectorSource,
    ) -> crate::Result<IvfCentroids> {
        assert_eq!(options.dim(), self.dim);
        let values = match self.metric {
            Metric::L2 => [0.0_f32, 1.0]
                .into_iter()
                .flat_map(|center| std::iter::repeat_n(center, self.dim))
                .collect(),
            Metric::Cosine | Metric::Dot => {
                let mut values = vec![0.0; 2 * self.dim];
                values[0] = 1.0;
                values[self.dim + 1] = 1.0;
                values
            }
        };
        Ok(IvfCentroids::F32(IvfMatrix {
            values,
            rows: 2,
            dims: self.dim,
        }))
    }

    fn assign(
        &self,
        _options: &VectorOptions,
        source: &mut dyn VectorSource,
        _centroids: &IvfCentroids,
    ) -> crate::Result<Vec<u32>> {
        let num_rows = source.num_rows();
        let mut assignments = Vec::with_capacity(num_rows);
        for_each_batch(source, num_rows, |IvfTrainingVectors::F32(batch)| {
            for row in batch.matrix.values.chunks_exact(self.dim) {
                assignments.push(match self.metric {
                    Metric::L2 => u32::from(row[0] >= 0.5),
                    Metric::Cosine | Metric::Dot => u32::from(row[1] > row[0]),
                });
            }
        })?;
        Ok(assignments)
    }
}

fn quant_fixture_config(dim: usize) -> VectorQuantizationConfig {
    quant_fixture_config_for(dim, Metric::L2, &[1, 4])
}

fn quant_fixture_config_for(
    dim: usize,
    metric: Metric,
    schedule: &[u8],
) -> VectorQuantizationConfig {
    let seeds = [0x1111, 0x2222, 0x3333, 0x4444];
    VectorQuantizationConfig::materialize(
        "embedding".to_string(),
        &VectorOptions::new(dim, metric),
        schedule
            .iter()
            .enumerate()
            .map(|(layer, &bits)| VectorQuantizationLayer {
                bits,
                seed: seeds[layer],
            })
            .collect(),
    )
    .unwrap()
}

fn fixture_vector(metric: Metric, dim: usize, doc: usize) -> Vec<f32> {
    match metric {
        Metric::L2 => {
            let center = if doc < 4 { 0.0 } else { 1.0 };
            (0..dim)
                .map(|coordinate| center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1)
                .collect()
        }
        Metric::Cosine | Metric::Dot => {
            let cluster = usize::from(doc >= 4);
            let mut vector: Vec<f32> = (0..dim)
                .map(|coordinate| ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.025)
                .collect();
            vector[cluster] += 1.0;
            vector
        }
    }
}

fn fixture_estimator_queries_for(metric: Metric, dim: usize) -> Vec<Vec<f32>> {
    (0..4)
        .map(|query| match metric {
            Metric::L2 => {
                let center = if query < 2 { 0.0 } else { 1.0 };
                (0..dim)
                    .map(|coordinate| {
                        center + ((query * dim + coordinate) as f32 * 0.023).cos() * 0.1
                    })
                    .collect()
            }
            Metric::Cosine => {
                let cluster = usize::from(query >= 2);
                let mut vector: Vec<f32> = (0..dim)
                    .map(|coordinate| ((query * dim + coordinate) as f32 * 0.023).cos() * 0.025)
                    .collect();
                vector[cluster] += 1.0;
                vector
            }
            Metric::Dot => {
                unreachable!("quantized matrix fixture covers L2 and cosine")
            }
        })
        .collect()
}

fn fixture_estimator_queries(dim: usize) -> Vec<Vec<f32>> {
    fixture_estimator_queries_for(Metric::L2, dim)
}

fn estimator_measurements(
    vector_reader: &crate::vector::VectorIndexReader,
    queries: &[Vec<f32>],
    sample_rows: usize,
    alive: Option<&crate::fastfield::AliveBitSet>,
) -> crate::Result<Option<VectorEstimatorMeasurements>> {
    let queries = queries
        .iter()
        .cloned()
        .map(|values| VectorEstimatorQuery {
            values,
            excluded_doc_id: None,
        })
        .collect::<Vec<_>>();
    Ok(vector_reader.measure_estimator_queries(
        VectorEstimatorSource::Provided,
        &queries,
        sample_rows,
        alive,
    )?)
}

fn build_quantized_fixture_with_schedule(dim: usize, quantized: bool) -> crate::Result<Index> {
    build_quantized_fixture_case(dim, Metric::L2, &[1, 4], quantized)
}

fn build_quantized_fixture_case(
    dim: usize,
    metric: Metric,
    schedule: &[u8],
    quantized: bool,
) -> crate::Result<Index> {
    // Exact: the fixture has two lists; Stacked's hard 25% top prune keeps one,
    // which breaks oracle and probe-budget assertions that need both.
    build_quantized_fixture_case_with_router(dim, metric, schedule, quantized, RouterKind::Exact)
}

fn build_quantized_fixture_case_with_router(
    dim: usize,
    metric: Metric,
    schedule: &[u8],
    quantized: bool,
    router: RouterKind,
) -> crate::Result<Index> {
    build_quantized_fixture_in_directory_with_router(
        dim,
        metric,
        schedule,
        quantized,
        crate::directory::RamDirectory::create(),
        router,
    )
}

fn build_quantized_fixture_in_directory(
    dim: usize,
    metric: Metric,
    schedule: &[u8],
    quantized: bool,
    directory: impl crate::directory::Directory,
) -> crate::Result<Index> {
    build_quantized_fixture_in_directory_with_router(
        dim,
        metric,
        schedule,
        quantized,
        directory,
        RouterKind::Exact,
    )
}

fn build_quantized_fixture_in_directory_with_router(
    dim: usize,
    metric: Metric,
    schedule: &[u8],
    quantized: bool,
    directory: impl crate::directory::Directory,
    router: RouterKind,
) -> crate::Result<Index> {
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_vector_field("embedding", VectorOptions::new(dim, metric));
    let label_field = schema_builder.add_text_field("label", STRING | STORED);
    let schema = schema_builder.build();
    let mut settings = IndexSettings {
        vector_clustering_threshold: 1,
        ..Default::default()
    };
    if quantized {
        settings.vector_quantization = vec![quant_fixture_config_for(dim, metric, schedule)];
    }
    let index = Index::builder()
        .schema(schema)
        .settings(settings)
        .ivf_clusterer(Arc::new(QuantFixtureClusterer { dim, metric }))
        .ivf_router(router)
        .create(directory)?;
    let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let mut segments = Vec::new();
    for doc in 0..8 {
        let vector = fixture_vector(metric, dim, doc);
        let mut document = TantivyDocument::new();
        document.add_vector(field, &vector);
        document.add_text(label_field, format!("d{doc}"));
        if doc % 2 == 0 {
            document.add_text(label_field, "keep");
        }
        writer.add_document(document)?;
        if doc == 3 || doc == 7 {
            writer.commit()?;
            for id in index.searchable_segment_ids()? {
                if !segments.contains(&id) {
                    segments.push(id);
                }
            }
        }
    }
    writer.merge(&segments).wait()?;
    writer.wait_merging_threads()?;
    Ok(index)
}

fn build_quantized_fixture(dim: usize, quantized: bool) -> crate::Result<Index> {
    build_quantized_fixture_with_schedule(dim, quantized)
}

fn build_flat_quantized_fixture(dim: usize) -> crate::Result<Index> {
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_vector_field("embedding", VectorOptions::new(dim, Metric::L2));
    let schema = schema_builder.build();
    let settings = IndexSettings {
        vector_clustering_threshold: usize::MAX,
        vector_quantization: vec![quant_fixture_config(dim)],
        ..Default::default()
    };
    let index = Index::builder()
        .schema(schema)
        .settings(settings)
        .create_in_ram()?;
    let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for doc in 0..8 {
        let center = if doc < 4 { 0.0 } else { 1.0 };
        let vector: Vec<f32> = (0..dim)
            .map(|coordinate| center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1)
            .collect();
        let mut document = TantivyDocument::new();
        document.add_vector(field, &vector);
        writer.add_document(document)?;
    }
    writer.commit()?;
    Ok(index)
}

fn fixture_expected(query: &[f32], dim: usize, top_n: usize) -> Vec<(u32, u32)> {
    let mut expected: Vec<(f32, u32)> = (0..8)
        .map(|doc| {
            let center = if doc < 4 { 0.0 } else { 1.0 };
            let vector: Vec<f32> = (0..dim)
                .map(|coordinate| center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1)
                .collect();
            (-l2_squared(query, &vector), doc as u32)
        })
        .collect();
    expected.sort_unstable_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    expected[..top_n]
        .iter()
        .map(|&(score, doc)| (score.to_bits(), doc))
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum QuantizedMatrixScenario {
    None,
    Filter,
    Deletes,
}

fn fixture_search_query(metric: Metric, dim: usize) -> Vec<f32> {
    match metric {
        Metric::L2 => vec![0.05; dim],
        Metric::Cosine | Metric::Dot => {
            let mut query: Vec<f32> = (0..dim)
                .map(|coordinate| ((coordinate as f32 + 0.5) * 0.031).cos() * 0.01)
                .collect();
            query[0] += 0.8;
            query[1] += 0.6;
            query
        }
    }
}

fn fixture_filter_docs(
    index: &Index,
    filter: &dyn Query,
) -> crate::Result<std::collections::HashSet<crate::DocAddress>> {
    let searcher = index.reader()?.searcher();
    let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
    let mut admitted = std::collections::HashSet::new();
    for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
        weight.for_each_no_score(segment, &mut |docs| {
            admitted.extend(
                docs.iter()
                    .copied()
                    .map(|doc| crate::DocAddress::new(segment_ord as u32, doc)),
            );
        })?;
    }
    Ok(admitted)
}

fn fixture_exact_hits(
    index: &Index,
    metric: Metric,
    query: &[f32],
    filter: Option<&dyn Query>,
    top_n: usize,
) -> crate::Result<Vec<(crate::Score, crate::DocAddress)>> {
    let field = index.schema().get_field("embedding")?;
    let mut hits = ground_truth::top_k(index, field, metric, query, 8)?;
    if let Some(filter) = filter {
        let admitted = fixture_filter_docs(index, filter)?;
        hits.retain(|(_, address)| admitted.contains(address));
    }
    hits.truncate(top_n);
    Ok(hits)
}

fn assert_matrix_results(
    context: &str,
    actual: &[(crate::Score, crate::DocAddress)],
    expected: &[(crate::Score, crate::DocAddress)],
    trace: &crate::vector::test_support::DocTrace,
) {
    if actual == expected {
        return;
    }

    let actual_docs: std::collections::HashSet<_> =
        actual.iter().map(|(_, address)| address.doc_id).collect();
    let missing = expected
        .iter()
        .map(|(_, address)| address.doc_id)
        .find(|doc| !actual_docs.contains(doc));
    let attribution = if let Some(doc) = missing {
        if !trace.scored_docs.contains(&doc) {
            "routing/admission miss"
        } else if trace
            .boundary_docs
            .iter()
            .any(|survivors| !survivors.contains(&doc))
        {
            "band drop"
        } else if !trace.exact_phase_docs.contains(&doc) {
            "exact-phase fetch bug"
        } else {
            "exact scoring/order bug"
        }
    } else {
        "exact scoring/order bug"
    };
    panic!("{context}: {attribution}; actual={actual:?} expected={expected:?} trace={trace:?}");
}

fn assert_quantized_matrix_storage(
    index: &Index,
    metric: Metric,
    schedule: &[u8],
) -> crate::Result<()> {
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    let field = index.schema().get_field("embedding")?;
    let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
    let ivf = vector_reader.index().expect("matrix fixture must be IVF");
    assert_eq!(ivf.num_rows(), 8);
    let quantized = vector_reader
        .quantization()?
        .expect("matrix fixture must carry quantized slots");
    assert_eq!(
        quantized
            .index_ctx()
            .meta
            .layers()
            .iter()
            .map(|layer| layer.bits())
            .collect::<Vec<_>>(),
        schedule
    );
    for (cluster, row) in block_rows(ivf)? {
        assert!(quantized
            .residual_norm(BlockAddress::new(cluster as u32, row as u32))?
            .is_finite());
        for (layer, stored) in quantized.layers().iter().enumerate() {
            assert!(
                (1.0..=4.0).contains(&stored.gamma(BlockAddress::new(cluster as u32, row as u32))?)
            );
            let corrected_error =
                stored.error_ratio(BlockAddress::new(cluster as u32, row as u32))?;
            assert!(corrected_error.is_finite() && corrected_error >= 0.0);
            assert_eq!(
                stored
                    .constant(BlockAddress::new(cluster as u32, row as u32))?
                    .is_some(),
                metric == Metric::L2,
                "layer {layer} split-constant presence must follow the metric at row {row}"
            );
        }
    }
    Ok(())
}

fn run_quantized_matrix_query(
    index: &Index,
    metric: Metric,
    schedule: &[u8],
    scenario: QuantizedMatrixScenario,
    depth: usize,
) -> crate::Result<()> {
    const TOP_N: usize = 3;
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    let field = index.schema().get_field("embedding")?;
    let label = index.schema().get_field("label")?;
    let query = fixture_search_query(metric, index.settings().vector_quantization[0].dim);
    let keep = TermQuery::new(
        Term::from_field_text(label, "keep"),
        IndexRecordOption::Basic,
    );
    let filter: &dyn Query = if matches!(scenario, QuantizedMatrixScenario::Filter) {
        &keep
    } else {
        &AllQuery
    };
    let expected = fixture_exact_hits(index, metric, &query, Some(filter), TOP_N)?;
    let collector = TopDocsByVectorSimilarityBuilder::new(field, query.clone(), TOP_N)
        .with_probe_params(ProbeParams {
            max_probe: usize::MAX,
            min_probe_clusters: 2,
            exact_plan: false,
            located: LocatedPath::Off,
            ..Default::default()
        })
        .with_max_scan_levels(depth)
        .build_for_searcher(&searcher, filter)?;
    let fruit = searcher.search(filter, &collector)?;
    assert_eq!(fruit.stats.len(), 1);
    let stats = &fruit.stats[0];
    let context =
        format!("metric={metric:?} schedule={schedule:?} scenario={scenario:?} depth={depth}");
    let trace = crate::vector::test_support::doc_trace(index, field, &stats.quantized_trace)?;
    assert_matrix_results(&context, &fruit.results, &expected, &trace);

    let layer0 = stats.layers.get(0).expect("layer 0 must execute");
    assert_eq!(
        stats.layer0_eligible,
        layer0.scored(),
        "{context}: layer 0 must score exactly the admitted eligible rows"
    );
    assert!(layer0.scored() > 0, "{context}: {stats:?}");
    assert!(
        layer0.survivors() <= layer0.scored(),
        "{context}: {stats:?}"
    );
    if depth == 1 {
        assert!(stats.layers.get(1).is_none(), "{context}: {stats:?}");
    } else {
        let layer1 = stats.layers.get(1).expect("layer 1 must execute");
        assert_eq!(
            layer1.scored(),
            layer0.survivors(),
            "{context}: every boundary-0 survivor must be refined"
        );
        assert!(
            layer1.survivors() <= layer1.scored(),
            "{context}: {stats:?}"
        );
    }
    if metric == Metric::L2 && matches!(scenario, QuantizedMatrixScenario::None) {
        assert!(
            layer0.survivors() < layer0.scored(),
            "{context}: the unfiltered L2 matrix cell must prove that boundary 0 measurably drops \
             at least one scored candidate: {stats:?}"
        );
    }
    assert_eq!(
        trace.boundary_docs.len(),
        depth,
        "{context}: one identity snapshot per executed boundary"
    );
    for boundary in &trace.boundary_docs {
        assert!(
            boundary.iter().all(|doc| trace.scored_docs.contains(doc)),
            "{context}: a later layer retained a row absent from layer-0 selection"
        );
    }
    assert!(
        trace.exact_phase_docs.iter().all(|doc| trace
            .boundary_docs
            .last()
            .is_some_and(|boundary| boundary.contains(doc))),
        "{context}: the exact phase read a row absent from the final selection"
    );
    for max_probe in [1, usize::MAX] {
        const ELIGIBILITY_TOP_N: usize = 8;
        let params = ProbeParams {
            max_probe,
            min_probe_clusters: 2,
            exact_plan: false,
            located: LocatedPath::Off,
            ..Default::default()
        };
        let quantized = searcher.search(
            filter,
            &TopDocsByVectorSimilarityBuilder::new(field, query.clone(), ELIGIBILITY_TOP_N)
                .with_probe_params(params.clone())
                .with_max_scan_levels(depth)
                .build_for_searcher(&searcher, filter)?,
        )?;
        let level0 = searcher.search(
            filter,
            &TopDocsByVectorSimilarityBuilder::new(field, query.clone(), ELIGIBILITY_TOP_N)
                .with_probe_params(params)
                .with_max_scan_levels(0)
                .build_for_searcher(&searcher, filter)?,
        )?;
        assert_eq!(
            crate::vector::test_support::doc_trace(
                index,
                field,
                &quantized.stats[0].quantized_trace
            )?
            .scored_docs,
            crate::vector::test_support::doc_trace(index, field, &level0.stats[0].quantized_trace)?
                .scored_docs,
            "{context}: selected candidate set at probe budget {max_probe}"
        );
    }
    match scenario {
        QuantizedMatrixScenario::Filter => {
            assert!(stats.pruned_filter > 0, "{context}: {stats:?}")
        }
        QuantizedMatrixScenario::Deletes => {
            assert!(stats.pruned_dead > 0, "{context}: {stats:?}")
        }
        QuantizedMatrixScenario::None => {}
    }
    Ok(())
}

fn quantized_vec_file(index: &Index) -> crate::Result<Vec<u8>> {
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    Ok(searcher.segment_readers()[0]
        .open_read(SegmentComponent::Custom(VEC_EXT.to_string()))?
        .read_bytes()?
        .to_vec())
}

fn assert_relative_1e5(actual: f32, expected: f32, context: &str) {
    let tolerance = 1e-5 * expected.abs().max(f32::MIN_POSITIVE);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{context}: actual={actual} expected={expected} tolerance={tolerance}"
    );
}

fn assert_quantized_bridge_exactness(dim: usize) -> crate::Result<()> {
    let index = build_quantized_fixture(dim, true)?;
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    let segment = &searcher.segment_readers()[0];
    let field = index.schema().get_field("embedding")?;
    let vector_reader = segment.vector_index(field)?;
    let ivf = vector_reader.index().expect("merged fixture must be IVF");
    let quantized = vector_reader
        .quantization()?
        .expect("configured IVF fixture must carry quantized slots");
    let (specs, grids) = (
        quantized.index_ctx().specs.clone(),
        quantized.index_ctx().grids.clone(),
    );
    let query: Vec<f32> = (0..dim)
        .map(|coordinate| ((coordinate as f32 + 0.5) * 0.031).cos())
        .collect();
    let harness = quant::cascade::prepare_split_query(&query, &specs, &grids, 4);
    let scan = QuantizedQueryCtx::new(
        Arc::clone(quantized.index_ctx()),
        &VectorQuery::new(quantized.index_ctx().meta.field().metric, Arc::new(query)),
    );

    for (cluster, row) in block_rows(ivf)? {
        let mut scan_sum = 0.0;
        let mut harness_sum = 0.0;
        for (layer, stored) in quantized.layers().iter().enumerate() {
            let codes = stored.code_bytes(BlockAddress::new(cluster as u32, row as u32))?;
            let scale = stored.scale(BlockAddress::new(cluster as u32, row as u32))?;
            let constant = stored.constant(BlockAddress::new(cluster as u32, row as u32))?;
            let scan_estimate = scan.score_layer(layer, &codes, scale, constant)?;
            let harness_estimate = match constant {
                Some(constant) => harness.score_layer(layer, &codes, scale, constant, specs[layer]),
                None => harness.score_layer_without_constant(layer, &codes, scale, specs[layer]),
            };
            assert_relative_1e5(
                scan_estimate,
                harness_estimate,
                &format!("d={dim} row={row} layer={layer}"),
            );
            scan_sum += scan_estimate;
            harness_sum += harness_estimate;
        }
        assert_relative_1e5(scan_sum, harness_sum, &format!("d={dim} row={row} summed"));
    }
    Ok(())
}

#[test]
fn bridge_exactness_d768_and_d100() -> crate::Result<()> {
    assert_quantized_bridge_exactness(768)?;
    assert_quantized_bridge_exactness(100)
}

// Settings version three remains a valid write target for vector format four.
#[test]
fn settings_v3_builds_v6_and_readers_ignore_global_changes() -> crate::Result<()> {
    let mut index = build_quantized_fixture(QUANT_FIXTURE_DIM, true)?;
    let field = index.schema().get_field("embedding")?;
    let config = &index.settings().vector_quantization[0];
    assert_eq!(config.format_version, 3);
    config.validate(&VectorOptions::new(QUANT_FIXTURE_DIM, Metric::L2))?;
    assert_eq!(&quantized_vec_file(&index)?[..4], &[6, 0, 0, 0]);
    index.settings_mut().vector_quantization.clear();
    let searcher = index.reader()?.searcher();
    let vector = searcher.segment_readers()[0].vector_index(field)?;
    assert_eq!(vector.quantization()?.unwrap().layers().len(), 2);
    assert!(vector
        .quantization()?
        .unwrap()
        .residual_norm(BlockAddress::new(0, 0))?
        .is_finite());
    Ok(())
}

#[test]
fn level_zero_matches_unquantized_ivf() -> crate::Result<()> {
    const DIM: usize = 64;
    let query = vec![0.05_f32; DIM];
    let params = ProbeParams {
        max_probe: 1,
        min_probe_clusters: 1,
        exact_plan: false,
        located: LocatedPath::Off,
        ..Default::default()
    };
    let unquantized = build_quantized_fixture(DIM, false)?;
    let quantized = build_quantized_fixture(DIM, true)?;
    let field = unquantized.schema().get_field("embedding")?;
    let unquantized_searcher = unquantized.reader()?.searcher();
    let quantized_reader = quantized.reader()?;
    let quantized_searcher = quantized_reader.searcher();
    let unquantized_fruit = unquantized_searcher.search(
        &AllQuery,
        &TopDocsByVectorSimilarityBuilder::new(field, query.clone(), 3)
            .with_probe_params(params.clone())
            .build_for_searcher(&unquantized_searcher, &AllQuery)?,
    )?;
    let level_zero_collector = TopDocsByVectorSimilarityBuilder::new(field, query, 3)
        .with_probe_params(params)
        .with_max_scan_levels(0)
        .build_for_searcher(&quantized_searcher, &AllQuery)?;
    assert!(quantized_searcher.segment_readers()[0]
        .vector_index(field)?
        .quantization()?
        .is_some());
    let level_zero_fruit = quantized_searcher.search(&AllQuery, &level_zero_collector)?;
    assert!(level_zero_collector.quantized_query_count() == 0);

    assert_eq!(level_zero_fruit.results, unquantized_fruit.results);
    assert_eq!(level_zero_fruit.stats.len(), 1);
    assert_eq!(unquantized_fruit.stats.len(), 1);
    let level_zero = &level_zero_fruit.stats[0];
    let baseline = &unquantized_fruit.stats[0];
    assert!(level_zero.layers.get(0).is_none(), "{level_zero:?}");
    assert!(level_zero.routing_visited_count > 0, "{level_zero:?}");
    assert!(level_zero.clusters_probed() > 0, "{level_zero:?}");
    assert!(level_zero.candidates_scored > 0, "{level_zero:?}");
    assert!(level_zero.exact_scan_ns.is_some(), "{level_zero:?}");
    assert_eq!(level_zero.candidates_scored, baseline.candidates_scored);
    assert_eq!(level_zero.exact_rows, baseline.exact_rows);
    assert_eq!(level_zero.postings_row, baseline.postings_row);
    assert_eq!(level_zero.postings_skipped, baseline.postings_skipped);
    assert_eq!(
        level_zero.routing_visited_count,
        baseline.routing_visited_count
    );
    assert_eq!(
        level_zero.work_charged.to_bits(),
        baseline.work_charged.to_bits()
    );
    Ok(())
}

// Metadata reports the stored contract for each backend, independently of the build target.
#[test]
fn public_metadata_reports_stored_contract_after_settings_change() -> crate::Result<()> {
    use crate::vector::storage::metadata::{Partition, Quantizer, VectorColMetadata};
    for mode in 0..3 {
        let mut index = match mode {
            0 => build_quantized_fixture(64, false)?,
            1 => build_quantized_fixture(64, true)?,
            _ => build_flat_quantized_fixture(64)?,
        };
        let field = index.schema().get_field("embedding")?;
        let before = {
            let reader = index.reader()?;
            let searcher = reader.searcher();
            let vector = searcher.segment_readers()[0].vector_index(field)?;
            let meta = vector.metadata().expect("stored field");
            assert_eq!(meta.field().dim(), 64);
            assert_eq!(meta.field().dtype(), crate::schema::VectorDType::F32);
            assert_eq!(meta.field().metric(), Metric::L2);
            assert_eq!(
                meta.field().norm_policy(),
                crate::vector::quantization::VectorNormPolicy::None
            );
            if mode == 2 {
                assert!(matches!(
                    meta.field().partition(),
                    Partition::Uniform { .. }
                ));
            } else {
                assert!(matches!(meta.field().partition(), Partition::Clusters));
            }
            if mode == 1 {
                assert!(matches!(meta, VectorColMetadata::Quantized { .. }));
                assert_eq!(
                    meta.layers()
                        .iter()
                        .map(Quantizer::bits)
                        .collect::<Vec<_>>(),
                    [1, 4]
                );
                assert_eq!(
                    meta.quantized_bytes_per_row(),
                    Some(quant_fixture_config(64).bytes_per_row())
                );
                assert!(matches!(meta.layers()[0], Quantizer::SignPlane { .. }));
            } else {
                assert!(matches!(meta, VectorColMetadata::Plain(_)));
                assert!(meta.layers().is_empty());
                assert_eq!(meta.quantized_bytes_per_row(), None);
            }
            meta.to_bytes()
        };
        index.settings_mut().vector_quantization =
            vec![quant_fixture_config_for(64, Metric::L2, &[4])];
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let vector = searcher.segment_readers()[0].vector_index(field)?;
        assert_eq!(vector.metadata().unwrap().to_bytes(), before);
    }
    let absent = crate::vector::VectorIndexReader::empty(VectorOptions::new(64, Metric::L2));
    assert!(absent.metadata().is_none());
    Ok(())
}

// Each stored schedule has one cache cell and merged search equals the segment union.
#[test]
fn mixed_segments_share_preparation_by_stored_metadata() -> crate::Result<()> {
    use crate::collector::Collector;
    for mixed in [false, true] {
        let mut index = build_quantized_fixture_case(64, Metric::L2, &[1, 4], true)?;
        let field = index.schema().get_field("embedding")?;
        let mut existing = index.searchable_segment_ids()?;
        for _ in 0..2 {
            index.settings_mut().vector_quantization = vec![quant_fixture_config_for(
                64,
                Metric::L2,
                if mixed { &[1] } else { &[1, 4] },
            )];
            let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for doc in 0..8 {
                let mut document = TantivyDocument::new();
                document.add_vector(field, &fixture_vector(Metric::L2, 64, doc));
                writer.add_document(document)?;
                if doc == 3 || doc == 7 {
                    writer.commit()?;
                }
            }
            let fresh: Vec<_> = index
                .searchable_segment_ids()?
                .into_iter()
                .filter(|id| !existing.contains(id))
                .collect();
            writer.merge(&fresh).wait()?;
            writer.wait_merging_threads()?;
            existing = index.searchable_segment_ids()?;
        }
        index.settings_mut().vector_quantization.clear();
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 3);
        let make_collector = || {
            TopDocsByVectorSimilarityBuilder::new(field, fixture_search_query(Metric::L2, 64), 5)
                .with_probe_params(ProbeParams {
                    max_probe: usize::MAX,
                    min_probe_clusters: 2,
                    ..Default::default()
                })
                .build_for_searcher(&searcher, &crate::query::AllQuery)
                .unwrap()
        };
        let collector = make_collector();
        let combined = searcher.search(&AllQuery, &collector)?;
        assert_eq!(collector.quantized_query_count(), if mixed { 2 } else { 1 });
        let weight = AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let segments = searcher
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(ord, reader)| crate::vector::VectorSegment::from((ord as u32, reader)))
            .collect::<Vec<_>>();
        let params = ProbeParams {
            max_probe: usize::MAX,
            ..Default::default()
        };
        let budget = crate::vector::VectorSearchBudget::for_segments(field, &params, &segments)?;
        let builder =
            TopDocsByVectorSimilarityBuilder::new(field, fixture_search_query(Metric::L2, 64), 5)
                .with_probe_params(params.clone());
        let router = crate::vector::TopClustersBySimilarityCollector::new(
            field,
            fixture_search_query(Metric::L2, 64),
            params,
            1.0,
        )?
        .with_budget(budget);
        let candidates = router.collect_segments(&segments, |_| {})?;
        let work: std::sync::Arc<dyn crate::vector::SharedVectorWork> =
            std::sync::Arc::new(crate::vector::AtomicVectorWork::default());
        let mut incremental: Option<crate::vector::TopDocsByVectorSimilarity<f32>> = None;
        let mut fruits = Vec::new();
        for segment in &segments {
            let prepared = builder.prepare(
                weight.as_ref(),
                std::slice::from_ref(segment),
                budget,
                |_| {},
            )?;
            let inputs = prepared.with_candidates(
                candidates
                    .iter()
                    .filter(|c| c.node.0 == segment.ordinal)
                    .copied()
                    .collect(),
                Some(work.clone()),
            )?;
            let collector = match &mut incremental {
                Some(collector) => {
                    collector.set_inputs(inputs)?;
                    collector
                }
                slot @ None => slot.insert(builder.clone().build(inputs)),
            };
            fruits.push(collector.collect_segment(
                weight.as_ref(),
                segment.ordinal,
                &segment.reader,
            )?);
        }
        let incremental = incremental.unwrap();
        assert_eq!(
            incremental.quantized_query_count(),
            if mixed { 2 } else { 1 }
        );
        assert_eq!(incremental.merge_fruits(fruits)?.results, combined.results);
        let mut union = Vec::new();
        for (ord, segment) in searcher.segment_readers().iter().enumerate() {
            let single = make_collector();
            let fruit = single.collect_segment(weight.as_ref(), ord as u32, segment)?;
            union.extend(single.merge_fruits(vec![fruit])?.results);
        }
        union.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        union.truncate(5);
        assert_eq!(combined.results, union);
    }
    Ok(())
}

#[test]
fn level_zero_flat_segment_remains_exact() -> crate::Result<()> {
    const DIM: usize = 64;
    let query = vec![0.05_f32; DIM];
    let expected = fixture_expected(&query, DIM, 3);
    let flat = build_flat_quantized_fixture(DIM)?;
    let reader = flat.reader()?;
    let searcher = reader.searcher();
    let field = flat.schema().get_field("embedding")?;
    let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
    assert!(vector_reader.index().is_none());
    assert!(vector_reader.quantization()?.is_none());

    let fruit = searcher.search(
        &AllQuery,
        &TopDocsByVectorSimilarityBuilder::new(field, query, 3)
            .with_max_scan_levels(0)
            .build_for_searcher(&searcher, &crate::query::AllQuery)
            .unwrap(),
    )?;
    assert_eq!(
        fruit
            .results
            .iter()
            .map(|&(score, address)| (score.to_bits(), address.doc_id))
            .collect::<Vec<_>>(),
        expected
    );
    let stats = &fruit.stats[0];
    assert_eq!(stats.exact_rows, 8, "{stats:?}");
    assert_eq!(stats.routing_visited_count, 0, "{stats:?}");
    assert_eq!(stats.clusters_probed(), 0, "{stats:?}");
    assert!(stats.layers.get(0).is_none(), "{stats:?}");
    Ok(())
}

#[test]
fn merge_quantization_matches_kernel_harness_and_is_reproducible() -> crate::Result<()> {
    let index = build_quantized_fixture(QUANT_FIXTURE_DIM, true)?;
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    let segment = &searcher.segment_readers()[0];
    let field = index.schema().get_field("embedding")?;
    let vector_reader = segment.vector_index(field)?;
    let ivf = vector_reader.index().expect("merged fixture must be IVF");
    assert_eq!(ivf.num_rows(), 8);
    let quantized = vector_reader
        .quantization()?
        .expect("configured IVF fixture must carry quantized slots");
    let (specs, grids) = (
        quantized.index_ctx().specs.clone(),
        quantized.index_ctx().grids.clone(),
    );
    let centroid_bytes = ivf.centroid_bytes()?;
    let centroid_stride = QUANT_FIXTURE_DIM * std::mem::size_of::<f32>();

    for cluster in 0..ivf.num_clusters() {
        let centroid = decode_row::<f32>(
            &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
            QUANT_FIXTURE_DIM,
        )?;
        let prepared = prepare_centroid(&centroid, &specs);
        let docs = vector_reader.cluster_doc_ids(cluster)?.unwrap();
        for (row, doc) in (0..ivf.cluster_num_rows(cluster)?).zip(docs) {
            let vector = decode_row::<f32>(
                &vector_reader.cluster_row_bytes(cluster, row, doc)?,
                QUANT_FIXTURE_DIM,
            )?;
            let mut expected_input = vector.clone();
            let mut workspace = quant::cascade::BatchEncodeWorkspace::new();
            let expected = quant::cascade::encode_batch_in_place_with_workspace(
                &mut expected_input,
                1,
                &prepared,
                &specs,
                &grids,
                &mut workspace,
                true,
            );
            assert_eq!(
                quantized
                    .residual_norm(BlockAddress::new(cluster as u32, row as u32))?
                    .to_bits(),
                l2_squared(&vector, &centroid).to_bits()
            );
            for (layer, stored) in quantized.layers().iter().enumerate() {
                let stored_codes =
                    stored.code_bytes(BlockAddress::new(cluster as u32, row as u32))?;
                assert_eq!(
                    stored_codes.len(),
                    QUANT_FIXTURE_DIM * usize::from(specs[layer].bits) / 8,
                    "divisible dimensions use exact byte strides"
                );
                assert_eq!(
                    stored_codes.as_slice(),
                    expected.layers[layer].codes.as_slice()
                );
                assert_eq!(
                    stored.scale(BlockAddress::new(cluster as u32, row as u32))?,
                    expected.layers[layer].scales[0]
                );
                assert_eq!(
                    stored
                        .gamma(BlockAddress::new(cluster as u32, row as u32))?
                        .to_bits(),
                    quant::model::f16::f16_to_f32(expected.layers[layer].gammas[0]).to_bits()
                );
                assert_eq!(
                    stored
                        .error_ratio(BlockAddress::new(cluster as u32, row as u32))?
                        .to_bits(),
                    quant::model::f16::f16_to_f32(expected.layers[layer].corrected_error_ratios[0])
                        .to_bits()
                );
                assert_eq!(
                    stored
                        .constant(BlockAddress::new(cluster as u32, row as u32))?
                        .expect("L2 fixture requires split constants")
                        .to_bits(),
                    expected.layers[layer].constants[0].to_bits()
                );
            }
        }
    }

    let query = vec![0.05_f32; QUANT_FIXTURE_DIM];
    let collector = TopDocsByVectorSimilarityBuilder::new(field, query.clone(), 3)
        .with_probe_params(ProbeParams {
            max_probe: usize::MAX,
            min_probe_clusters: 2,
            exact_plan: false,
            located: LocatedPath::Off,
            ..Default::default()
        })
        .build_for_searcher(&searcher, &crate::query::AllQuery)
        .unwrap();
    let quantized_fruit = searcher.search(&AllQuery, &collector)?;
    assert!(collector.quantized_query_count() > 0);
    assert_eq!(quantized_fruit.stats.len(), 1);
    let stats = &quantized_fruit.stats[0];
    let layer0 = stats.layers.get(0).expect("layer 0 must execute");
    let layer1 = stats.layers.get(1).expect("layer 1 must execute");
    assert!(layer0.scored() > 0, "{stats:?}");
    assert!(layer0.survivors() <= layer0.scored(), "{stats:?}");
    assert_eq!(
        layer1.scored(),
        layer0.survivors(),
        "the two-layer fixture refines every first-boundary survivor: {stats:?}"
    );
    assert!(layer1.survivors() <= layer0.survivors(), "{stats:?}");
    assert!(stats.exact_rows <= layer1.survivors(), "{stats:?}");
    let hits = quantized_fruit.results;
    let mut expected: Vec<(f32, u32)> = (0..8)
        .map(|doc| {
            let center = if doc < 4 { 0.0 } else { 1.0 };
            let vector: Vec<f32> = (0..QUANT_FIXTURE_DIM)
                .map(|coordinate| {
                    center + ((doc * QUANT_FIXTURE_DIM + coordinate) as f32 * 0.017).sin() * 0.1
                })
                .collect();
            (-l2_squared(&query, &vector), doc as u32)
        })
        .collect();
    expected.sort_unstable_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    assert_eq!(
        hits.iter()
            .map(|&(score, address)| (score.to_bits(), address.doc_id))
            .collect::<Vec<_>>(),
        expected[..3]
            .iter()
            .map(|&(score, doc)| (score.to_bits(), doc))
            .collect::<Vec<_>>()
    );

    let first = quantized_vec_file(&index)?;
    let second = quantized_vec_file(&build_quantized_fixture(QUANT_FIXTURE_DIM, true)?)?;
    assert_eq!(
        first, second,
        "fixed assignment and seeds must be byte-identical"
    );
    Ok(())
}

#[test]
fn quantized_exhaustive_search_probes_all_clusters() -> crate::Result<()> {
    let index = build_quantized_fixture_case(100, Metric::L2, &[1], true)?;
    let field = index.schema().get_field("embedding")?;
    let query = fixture_search_query(Metric::L2, 100);
    let searcher = index.reader()?.searcher();
    let collector = TopDocsByVectorSimilarityBuilder::new(field, query.clone(), 3)
        .with_probe_params(ProbeParams {
            max_probe: usize::MAX,
            min_probe_clusters: 1,
            exact_plan: false,
            located: LocatedPath::Off,
            ..Default::default()
        })
        .with_max_scan_levels(1)
        .build_for_searcher(&searcher, &crate::query::AllQuery)
        .unwrap();
    let fruit = searcher.search(&AllQuery, &collector)?;
    assert_eq!(fruit.stats.len(), 1);
    let stats = &fruit.stats[0];
    assert_eq!(
        fruit.results,
        fixture_exact_hits(&index, Metric::L2, &query, None, 3)?,
        "{stats:?}"
    );
    assert_eq!(stats.clusters_probed(), 2, "{stats:?}");
    Ok(())
}

#[test]
fn quantized_probe_terminates_at_ceiling() -> crate::Result<()> {
    let index = build_quantized_fixture_case(100, Metric::L2, &[1], true)?;
    let field = index.schema().get_field("embedding")?;
    let query = fixture_search_query(Metric::L2, 100);
    let searcher = index.reader()?.searcher();
    let collector = TopDocsByVectorSimilarityBuilder::new(field, query.clone(), 3)
        .with_probe_params(ProbeParams {
            max_probe: 1,
            exact_plan: false,
            located: LocatedPath::Off,
            ..Default::default()
        })
        .with_max_scan_levels(1)
        .build_for_searcher(&searcher, &crate::query::AllQuery)
        .unwrap();
    let fruit = searcher.search(&AllQuery, &collector)?;
    assert_eq!(fruit.stats.len(), 1);
    let stats = &fruit.stats[0];
    assert_eq!(
        fruit.results,
        fixture_exact_hits(&index, Metric::L2, &query, None, 3)?,
        "{stats:?}"
    );
    assert_eq!(
        stats.termination,
        crate::vector::ProbeTermination::Ceiling,
        "{stats:?}"
    );
    assert_eq!(stats.clusters_probed(), 1, "{stats:?}");
    assert_eq!(
        stats.vectors_visited,
        stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
        "{stats:?}"
    );
    Ok(())
}

#[test]
fn quantized_filtered_rows_are_not_charged() -> crate::Result<()> {
    let index = build_quantized_fixture_case(100, Metric::L2, &[1], true)?;
    let field = index.schema().get_field("embedding")?;
    let label = index.schema().get_field("label")?;
    let query = fixture_search_query(Metric::L2, 100);
    let searcher = index.reader()?.searcher();
    let keep = TermQuery::new(
        Term::from_field_text(label, "keep"),
        IndexRecordOption::Basic,
    );
    let params = ProbeParams {
        max_probe: usize::MAX,
        min_probe_clusters: 2,
        exact_plan: false,
        located: LocatedPath::Off,
        ..Default::default()
    };
    let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
    let ivf = vector_reader.index().expect("merged fixture must be IVF");
    let (_, n_avg, open_share) = params.resolved_work_budget(ivf.num_clusters(), ivf.num_rows())?;
    let row = (1.0 - open_share) / n_avg;
    // Request all fixture rows so both scans open every cluster.
    let collector = |filter: &dyn Query| {
        TopDocsByVectorSimilarityBuilder::new(field, query.clone(), 8)
            .with_probe_params(params.clone())
            .with_max_scan_levels(1)
            .build_for_searcher(&searcher, filter)
            .unwrap()
    };
    let unfiltered = searcher.search(&AllQuery, &collector(&AllQuery))?;
    let filtered = searcher.search(&keep, &collector(&keep))?;
    assert_eq!(
        unfiltered.results,
        fixture_exact_hits(&index, Metric::L2, &query, None, 8)?
    );
    assert_eq!(
        filtered.results,
        fixture_exact_hits(&index, Metric::L2, &query, Some(&keep), 8)?
    );
    assert_eq!(unfiltered.stats.len(), 1);
    assert_eq!(filtered.stats.len(), 1);
    let unfiltered = &unfiltered.stats[0];
    let filtered = &filtered.stats[0];
    assert_eq!(
        filtered.candidates_scored,
        fixture_filter_docs(&index, &keep)?.len(),
        "{filtered:?}"
    );
    let expected = (unfiltered.candidates_scored - filtered.candidates_scored) as f64 * row;
    assert!(
        ((unfiltered.work_charged - filtered.work_charged) as f64 - expected).abs() < 1e-5,
        "unfiltered={unfiltered:?}; filtered={filtered:?}; expected difference={expected}"
    );
    Ok(())
}

#[test]
fn quantized_top_n_fixture_matrix_matches_direct_exact_oracle() -> crate::Result<()> {
    const SCHEDULES: &[&[u8]] = &[&[1], &[1, 4], &[1, 1, 4], &[2, 4]];

    for metric in [Metric::Cosine, Metric::L2, Metric::Dot] {
        for &schedule in SCHEDULES {
            let dim = 100;

            let primary = build_quantized_fixture_case(dim, metric, schedule, true)?;
            assert_quantized_matrix_storage(&primary, metric, schedule)?;
            for scenario in [
                QuantizedMatrixScenario::None,
                QuantizedMatrixScenario::Filter,
            ] {
                for depth in 1..=schedule.len() {
                    run_quantized_matrix_query(&primary, metric, schedule, scenario, depth)?;
                }
            }

            let label = primary.schema().get_field("label")?;
            let mut writer: crate::IndexWriter<TantivyDocument> =
                primary.writer_with_num_threads(1, 30_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for doc in [0, 4] {
                writer.delete_term(Term::from_field_text(label, &format!("d{doc}")));
            }
            writer.commit()?;
            drop(writer);
            for depth in 1..=schedule.len() {
                run_quantized_matrix_query(
                    &primary,
                    metric,
                    schedule,
                    QuantizedMatrixScenario::Deletes,
                    depth,
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn general_dimension_quantized_bridge_at_d100() -> crate::Result<()> {
    const DIM: usize = 100;
    let index = build_quantized_fixture(DIM, true)?;
    let reader = index.reader().map_err(|error| {
        TantivyError::InternalError(format!("general-d reader open failed: {error}"))
    })?;
    reader.reload().map_err(|error| {
        TantivyError::InternalError(format!("general-d reader reload failed: {error}"))
    })?;
    let searcher = reader.searcher();
    let segment = &searcher.segment_readers()[0];
    let field = index.schema().get_field("embedding")?;
    let vector_reader = segment.vector_index(field).map_err(|error| {
        TantivyError::InternalError(format!("general-d vector reader failed: {error}"))
    })?;
    let ivf = vector_reader.index().expect("merged fixture must be IVF");
    let quantized = vector_reader
        .quantization()?
        .expect("configured IVF fixture must carry quantized slots");
    let (specs, grids) = (
        quantized.index_ctx().specs.clone(),
        quantized.index_ctx().grids.clone(),
    );
    let centroid_bytes = ivf.centroid_bytes()?;
    let centroid_stride = DIM * std::mem::size_of::<f32>();

    for cluster in 0..ivf.num_clusters() {
        let centroid = decode_row::<f32>(
            &centroid_bytes[cluster * centroid_stride..][..centroid_stride],
            DIM,
        )?;
        let prepared = prepare_centroid(&centroid, &specs);
        let docs = vector_reader.cluster_doc_ids(cluster)?.unwrap();
        for (row, doc) in (0..ivf.cluster_num_rows(cluster)?).zip(docs) {
            let vector =
                decode_row::<f32>(&vector_reader.cluster_row_bytes(cluster, row, doc)?, DIM)?;
            let residual: Vec<f32> = vector
                .iter()
                .zip(&centroid)
                .map(|(&value, &center)| value - center)
                .collect();
            let expected =
                quant::cascade::encode_layers(&residual, Some(&prepared), &specs, &grids);
            for (layer, stored) in quantized.layers().iter().enumerate() {
                assert_eq!(
                    stored
                        .code_bytes(BlockAddress::new(cluster as u32, row as u32))?
                        .as_slice(),
                    expected.codes[layer]
                );
                assert_eq!(
                    stored.scale(BlockAddress::new(cluster as u32, row as u32))?,
                    expected.scales[layer]
                );
                assert_eq!(
                    stored
                        .constant(BlockAddress::new(cluster as u32, row as u32))?
                        .expect("L2 fixture requires split constants")
                        .to_bits(),
                    expected.constants[layer].to_bits()
                );
            }
        }
    }

    let query = vec![0.05_f32; DIM];
    let quantized_hits = searcher
        .search(
            &AllQuery,
            &TopDocsByVectorSimilarityBuilder::new(field, query.clone(), 3)
                .with_probe_params(ProbeParams {
                    max_probe: usize::MAX,
                    min_probe_clusters: 2,
                    ..Default::default()
                })
                .build_for_searcher(&searcher, &crate::query::AllQuery)
                .unwrap(),
        )?
        .results;
    assert_eq!(
        quantized_hits
            .iter()
            .map(|&(score, address)| (score.to_bits(), address.doc_id))
            .collect::<Vec<_>>(),
        fixture_expected(&query, DIM, 3)
    );
    Ok(())
}

#[test]
fn l2_quantized_fixture_growth_matches_768_1_plus_4_ledger() -> crate::Result<()> {
    const DIM: usize = 768;
    const ROWS: usize = 8;
    let config = quant_fixture_config(DIM);
    assert_eq!(config.bytes_per_row(), 508);

    let plain = quantized_vec_file(&build_quantized_fixture(DIM, false)?)?;
    let quantized = quantized_vec_file(&build_quantized_fixture(DIM, true)?)?;
    let physical_growth = quantized.len() - plain.len();
    let logical_growth = ROWS * config.bytes_per_row();
    println!(
        "VECTOR_QUANTIZATION_SIZE dim={DIM} rows={ROWS} plain_bytes={} quantized_bytes={} \
         physical_growth={physical_growth} logical_growth={logical_growth}",
        plain.len(),
        quantized.len(),
    );
    assert!(physical_growth >= logical_growth);
    let opts = VectorOptions::new(DIM, Metric::L2);
    let plain_meta = VectorColMetadata::build_ivf(&opts, None)?;
    let quant_meta = VectorColMetadata::build_ivf(&opts, Some(&config))?;
    let flat_meta = VectorColMetadata::build_flat(&opts);
    let entry_len = |meta: &VectorColMetadata, blocks: usize| {
        use crate::vector::storage::blocks::{align_up, block_align, data_entry_len};
        data_entry_len(
            align_up(4 + meta.to_bytes().len(), block_align(&meta.slots()))
                + blocks * block_len(&meta.slots(), ROWS / blocks),
            blocks,
        )
    };
    // The quantized field adds its code columns in the cluster blocks, moves the rows into one
    // ExactRows block, and addresses them with a one-byte identity ExactIdMap.
    let expected_growth =
        entry_len(&quant_meta, 2) + entry_len(&flat_meta, 1) + 1 - entry_len(&plain_meta, 2);
    assert!(
        physical_growth >= expected_growth && physical_growth - expected_growth <= 24,
        "only the two extra composite directory entries may differ: physical {physical_growth} \
         expected {expected_growth}"
    );
    assert_eq!(logical_growth, ROWS * 508);
    Ok(())
}

#[test]
fn estimator_measurement_uses_production_path_and_is_centered() -> crate::Result<()> {
    const DIM: usize = 100;
    let index = build_quantized_fixture_with_schedule(DIM, true)?;
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    let field = index.schema().get_field("embedding")?;
    let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
    assert!(vector_reader.quantization()?.is_some());
    let queries = fixture_estimator_queries(DIM);
    let estimator_queries = queries
        .iter()
        .cloned()
        .map(|values| VectorEstimatorQuery {
            values,
            excluded_doc_id: None,
        })
        .collect::<Vec<_>>();
    let measurements = vector_reader
        .measure_estimator_queries(
            VectorEstimatorSource::Provided,
            &estimator_queries,
            1_000,
            None,
        )?
        .expect("quantized slots remain available to explicit diagnostics");
    assert_eq!(measurements.source(), VectorEstimatorSource::Provided);
    assert_eq!(measurements.sample_rows(), 8);
    assert_eq!(measurements.query_count(), queries.len() as u32);
    assert!(measurements
        .aggregate()
        .iter()
        .all(|depth| depth.sample_count == 8 * queries.len() as u64));
    for (depth, moments) in measurements.aggregate().iter().enumerate() {
        let bias = moments
            .bias()
            .expect("fixture must produce estimator errors");
        assert!(
            bias.abs() <= 0.3,
            "depth {} normalized estimator bias {bias} exceeds 0.3",
            depth + 1
        );
    }

    let fruit = searcher.search(
        &AllQuery,
        &TopDocsByVectorSimilarityBuilder::new(field, queries[0].clone(), 3)
            .with_probe_params(ProbeParams {
                max_probe: usize::MAX,
                min_probe_clusters: 2,
                exact_plan: false,
                located: LocatedPath::Off,
                ..Default::default()
            })
            .build_for_searcher(&searcher, &crate::query::AllQuery)
            .unwrap(),
    )?;
    assert!(fruit.stats[0].layers.get(0).is_some());
    assert!(fruit.stats[0].postings_row > 0);
    Ok(())
}

#[test]
fn estimator_samples_only_live_posting_rows() -> crate::Result<()> {
    const DIM: usize = 100;
    let index = build_quantized_fixture_with_schedule(DIM, true)?;
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    let segment = &searcher.segment_readers()[0];
    let field = index.schema().get_field("embedding")?;
    let vector_reader = segment.vector_index(field)?;
    let alive = crate::fastfield::AliveBitSet::for_test_from_deleted_docs(&[1, 3], 8);
    let live_posting_rows = block_rows(vector_reader.index().unwrap())?
        .into_iter()
        .filter(|&(block, row)| {
            alive.is_alive(
                vector_reader
                    .doc_id_at(BlockAddress::new(block as u32, row as u32))
                    .unwrap(),
            )
        })
        .count();
    assert_eq!(live_posting_rows, 6);

    let queries = fixture_estimator_queries(DIM);
    let measurements =
        estimator_measurements(vector_reader.as_ref(), &queries, usize::MAX, Some(&alive))?
            .unwrap();
    assert!(measurements
        .aggregate()
        .iter()
        .all(|depth| { depth.sample_count == (live_posting_rows * queries.len()) as u64 }));

    let bounded =
        estimator_measurements(vector_reader.as_ref(), &queries, 5, Some(&alive))?.unwrap();
    assert!(bounded
        .aggregate()
        .iter()
        .all(|depth| depth.sample_count == (5 * queries.len()) as u64));
    Ok(())
}

#[test]
fn held_out_estimator_excludes_the_source_row() -> crate::Result<()> {
    const DIM: usize = 100;
    let index = build_quantized_fixture_with_schedule(DIM, true)?;
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    let segment = &searcher.segment_readers()[0];
    let field = index.schema().get_field("embedding")?;
    let vector_reader = segment.vector_index(field)?;
    let queries = vector_reader
        .sample_estimator_pseudo_queries(1, segment.alive_bitset())?
        .expect("quantized fixture must support pseudo-query sampling");
    assert_eq!(queries.len(), 1);
    assert!(queries.iter().all(|query| query.excluded_doc_id.is_some()));

    let measurements = vector_reader
        .measure_estimator_queries(
            VectorEstimatorSource::HeldOut,
            &queries,
            usize::MAX,
            segment.alive_bitset(),
        )?
        .expect("quantized fixture must support estimator measurement");
    assert_eq!(measurements.source(), VectorEstimatorSource::HeldOut);
    assert_eq!(measurements.sample_rows(), 8);
    assert_eq!(measurements.query_count(), 1);
    assert!(measurements
        .aggregate()
        .iter()
        .all(|moments| moments.sample_count == 7));
    Ok(())
}
#[test]
fn layer_zero_io_matches_requested_page_spans() -> crate::Result<()> {
    use crate::vector::storage::io::test_support::{PagedDirectory, PAGE_BYTES};
    use crate::vector::VectorStage;
    let directory = PagedDirectory::default();
    let index =
        build_quantized_fixture_in_directory(64, Metric::L2, &[1], true, directory.clone())?;
    let field = index.schema().get_field("embedding")?;
    let searcher = index.reader()?.searcher();
    directory.reads.lock().unwrap().clear();
    directory.centroid_reads.lock().unwrap().clear();
    let result = searcher.search(
        &AllQuery,
        &TopDocsByVectorSimilarityBuilder::new(field, fixture_search_query(Metric::L2, 64), 16)
            .with_probe_params(ProbeParams {
                max_probe: usize::MAX,
                min_probe_clusters: 2,
                exact_plan: false,
                located: LocatedPath::Off,
                ..Default::default()
            })
            .build_for_searcher(&searcher, &crate::query::AllQuery)
            .unwrap(),
    )?;
    let reads = directory.reads.lock().unwrap();
    let centroid_reads = directory.centroid_reads.lock().unwrap();
    let ranges: Vec<_> = reads
        .iter()
        .chain(centroid_reads.iter())
        .filter_map(|(stage, range)| matches!(stage, VectorStage::LayerScan(0)).then_some(range))
        .collect();
    let io = result.stats[0].layers.get(0).unwrap().io;
    assert!(io.reads > 0);
    assert_eq!(io.reads, ranges.len() as u64);
    assert_eq!(
        io.bytes_read,
        ranges.iter().map(|r| r.len() as u64).sum::<u64>()
    );
    assert_eq!(
        io.storage_blocks,
        ranges
            .iter()
            .filter(|r| !r.is_empty())
            .map(|r| ((r.end - 1) / PAGE_BYTES - r.start / PAGE_BYTES + 1) as u64)
            .sum::<u64>()
    );
    Ok(())
}
mod storage_properties {
    include!("storage_properties.rs");
}

// --- Document-ordered rows ---

/// The vectors of a storage test, one of four one-hot clusters plus a signal; every 11th document
/// has none.
fn storage_vectors(docs: usize) -> Vec<Option<Vec<f32>>> {
    (0..docs)
        .map(|doc| {
            if doc % 11 == 4 {
                return None;
            }
            let mut rng = Lcg(doc as u64 * 31 + 7);
            let mut vector = rng.vector(64, 0.2);
            vector[doc % 4] += 1.0;
            Some(vector)
        })
        .collect()
}

fn storage_spec(layout: &str, metric: Metric, vectors: &[Option<Vec<f32>>]) -> FixtureSpec {
    let centroids: Vec<Vec<f32>> = (0..4)
        .map(|axis| {
            let mut centroid = vec![0.0; 64];
            centroid[axis] = 1.0;
            centroid
        })
        .collect();
    let present: Vec<Vec<f32>> = vectors.iter().flatten().cloned().collect();
    let absent: Vec<usize> = vectors
        .iter()
        .enumerate()
        .filter_map(|(doc, vector)| vector.is_none().then_some(doc))
        .collect();
    // Absent docs are written as docs without a vector, at their own ordinals.
    let mut dense: Vec<Vec<f32>> = Vec::with_capacity(vectors.len());
    let mut next = present.iter();
    for vector in vectors {
        dense.push(match vector {
            Some(_) => next.next().unwrap().clone(),
            None => vec![0.0; 64],
        });
    }
    let spec = match layout {
        "flat" => FixtureSpec::flat(metric, 64, dense.into_iter().map(Some).collect()),
        "plain" => FixtureSpec::ivf(metric, centroids, dense),
        "quantized" => FixtureSpec::ivf(metric, centroids, dense).schedule(&[1, 4]),
        _ => unreachable!(),
    };
    spec.without_vectors(move |ordinal| absent.contains(&ordinal))
}

/// Every representation returns each document's vector bytes as written: a flat field from its
/// rows, a plain clustered field from its cluster block, a quantized clustered field from its
/// document-ordered rows at the location its presence map ranks it to.
#[test]
fn document_vector_bytes_agree_across_representations() -> crate::Result<()> {
    let vectors = storage_vectors(300);
    for layout in ["flat", "plain", "quantized"] {
        let fixture = storage_spec(layout, Metric::L2, &vectors).build()?;
        let searcher = fixture.index.reader()?.searcher();
        let mut seen = 0;
        for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
            let reader = segment.vector_index(fixture.field)?;
            assert_eq!(reader.has_exact_rows(), layout == "quantized", "{layout}");
            for doc in 0..segment.max_doc() {
                let ordinal = fixture.ordinal(DocAddress::new(segment_ord as u32, doc))?;
                let bytes = reader.vector_bytes(doc)?;
                match &vectors[ordinal] {
                    None => {
                        assert!(bytes.is_none(), "{layout} doc {ordinal}");
                        assert!(!reader.contains(doc)?, "{layout} doc {ordinal}");
                        assert_eq!(reader.exact_location(doc)?, None, "{layout} doc {ordinal}");
                    }
                    Some(expected) => {
                        let bytes = bytes.expect("present vector");
                        assert_eq!(
                            decode_row::<f32>(&bytes, 64)?,
                            *expected,
                            "{layout} doc {ordinal}"
                        );
                        seen += 1;
                        let location = reader.row_location(doc)?.expect("located");
                        match layout {
                            "quantized" => {
                                let exact = reader.exact_location(doc)?.expect("exact row");
                                assert_eq!(reader.vector_bytes_for_row(exact)?, bytes);
                                let docs =
                                    reader.cluster_doc_ids(location.block as usize)?.unwrap();
                                assert_eq!(
                                    docs[location.row as usize], doc,
                                    "{layout} primary row"
                                );
                            }
                            "plain" => {
                                assert_eq!(reader.exact_location(doc)?, None);
                                assert_eq!(reader.vector_bytes_for_row(location)?, bytes);
                                assert_eq!(reader.doc_id_at(location)?, doc);
                            }
                            _ => {
                                assert_eq!(reader.exact_location(doc)?, Some(location));
                                assert_eq!(reader.vector_bytes_for_row(location)?, bytes);
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(seen, vectors.iter().flatten().count(), "{layout}");
    }
    Ok(())
}

/// The vectors every alive document of `index` resolves to, by ordinal.
fn vectors_by_ordinal(fixture: &Fixture) -> crate::Result<Vec<(usize, Option<Vec<f32>>)>> {
    let searcher = fixture.index.reader()?.searcher();
    let mut out = Vec::new();
    for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
        let reader = segment.vector_index(fixture.field)?;
        for doc in 0..segment.max_doc() {
            if segment
                .alive_bitset()
                .is_some_and(|alive| !alive.is_alive(doc))
            {
                continue;
            }
            let ordinal = fixture.ordinal(DocAddress::new(segment_ord as u32, doc))?;
            let vector = reader
                .vector_bytes(doc)?
                .map(|bytes| decode_row::<f32>(&bytes, 64))
                .transpose()?;
            out.push((ordinal, vector));
        }
    }
    out.sort_by_key(|(ordinal, _)| *ordinal);
    Ok(out)
}

/// Quantized segments round-trip every vector through every merge shape: two quantized segments
/// into one, a quantized segment with a flat one into one, and a quantized segment alone into a
/// flat segment once the clustering threshold is out of reach.
#[test]
fn quantized_segments_round_trip_through_every_merge_shape() -> crate::Result<()> {
    let vectors = storage_vectors(260);
    let expected: Vec<(usize, Option<Vec<f32>>)> = vectors
        .iter()
        .cloned()
        .enumerate()
        .filter(|(ordinal, _)| ordinal % 61 != 9)
        .collect();
    let formats = |fixture: &Fixture| -> crate::Result<Vec<bool>> {
        let searcher = fixture.index.reader()?.searcher();
        searcher
            .segment_readers()
            .iter()
            .map(|segment| Ok(segment.vector_index(fixture.field)?.has_exact_rows()))
            .collect()
    };
    for shape in ["quantized+quantized", "quantized+flat", "quantized->flat"] {
        let spec = storage_spec("quantized", Metric::L2, &vectors)
            .segments(if shape == "quantized+quantized" { 2 } else { 1 })
            .deletes(&[9, 70, 131, 192, 253]);
        let mut fixture = spec.build()?;
        let mut writer: IndexWriter = fixture.index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        if shape == "quantized+flat" {
            // Move the last 60 docs into a flat segment of their own: delete them from the
            // quantized segment and write them again.
            for ordinal in 200..260 {
                writer.delete_term(Term::from_field_text(fixture.label, &format!("d{ordinal}")));
            }
            writer.commit()?;
            for (ordinal, vector) in vectors.iter().enumerate().skip(200) {
                if ordinal % 61 == 9 {
                    continue;
                }
                let mut doc = TantivyDocument::new();
                doc.add_text(fixture.label, format!("d{ordinal}"));
                if let Some(vector) = vector {
                    doc.add_vector(fixture.field, vector.as_slice());
                }
                writer.add_document(doc)?;
            }
            writer.commit()?;
            assert_eq!(
                formats(&fixture)?.iter().filter(|exact| **exact).count(),
                1,
                "{shape}"
            );
            assert_eq!(formats(&fixture)?.len(), 2, "{shape}");
        } else if shape == "quantized->flat" {
            fixture.index.settings_mut().vector_clustering_threshold = usize::MAX;
            drop(writer);
            writer = fixture.index.writer_with_num_threads(1, 30_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
        }
        assert_eq!(vectors_by_ordinal(&fixture)?, expected, "{shape} before");
        writer
            .merge(&fixture.index.searchable_segment_ids()?)
            .wait()?;
        writer.wait_merging_threads()?;
        let merged = formats(&fixture)?;
        assert_eq!(merged.len(), 1, "{shape}");
        assert_eq!(merged[0], shape != "quantized->flat", "{shape}");
        assert_eq!(vectors_by_ordinal(&fixture)?, expected, "{shape} after");
        let query = Lcg(99).vector(64, 0.5);
        let fruit = fixture.search(&AllQuery, &query, 7, routed_exhaustive_params(4))?;
        assert_eq!(
            fruit.results,
            fixture.oracle(&AllQuery, &query, 7)?,
            "{shape}"
        );
    }
    Ok(())
}

/// A quantized cosine field's exact rows are the bytes its codes were built from: the clustered
/// merge re-normalizes each stored row as the encode pass does, while a flat merge keeps the
/// stored bytes. Stored rows are unit-norm only to within an ulp; rows concentrated on two axes
/// add their rounding errors up in the norm, so re-normalizing moves some of their bytes.
#[test]
fn exact_rows_carry_the_bytes_the_codes_were_built_from() -> crate::Result<()> {
    let vectors: Vec<Option<Vec<f32>>> = (0..300)
        .map(|doc| {
            if doc % 11 == 4 {
                return None;
            }
            let mut vector = Lcg(doc as u64 * 31 + 7).vector(64, 0.05);
            vector[doc % 4] += 0.7;
            vector[(doc + 1) % 4] += 0.7;
            Some(vector)
        })
        .collect();
    let options = VectorOptions::new(64, Metric::Cosine);
    let normalized = |mut bytes: Vec<u8>| {
        maybe_normalize_bytes(&options, &mut bytes);
        bytes
    };
    // Ingest normalizes each input once.
    let stored: Vec<Option<Vec<u8>>> = vectors
        .iter()
        .map(|vector| {
            vector
                .as_ref()
                .map(|vector| normalized(vector.iter().flat_map(|x| x.to_le_bytes()).collect()))
        })
        .collect();
    let encoded: Vec<Option<Vec<u8>>> = stored
        .iter()
        .map(|row| row.clone().map(normalized))
        .collect();
    assert!(
        stored
            .iter()
            .zip(&encoded)
            .any(|(stored, encoded)| stored != encoded),
        "some stored row re-normalizes to other bytes"
    );
    let bytes_by_ordinal = |fixture: &Fixture| -> crate::Result<Vec<Option<Vec<u8>>>> {
        let searcher = fixture.index.reader()?.searcher();
        let mut out = vec![None; vectors.len()];
        for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
            let reader = segment.vector_index(fixture.field)?;
            for doc in 0..segment.max_doc() {
                let ordinal = fixture.ordinal(DocAddress::new(segment_ord as u32, doc))?;
                out[ordinal] = reader
                    .vector_bytes(doc)?
                    .map(|bytes| bytes.as_slice().to_vec());
            }
        }
        Ok(out)
    };

    let flat = storage_spec("flat", Metric::Cosine, &vectors)
        .segments(2)
        .build()?;
    assert_eq!(bytes_by_ordinal(&flat)?, stored, "flat before the merge");
    let mut writer: IndexWriter = flat.index.writer_with_num_threads(1, 30_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    writer.merge(&flat.index.searchable_segment_ids()?).wait()?;
    writer.wait_merging_threads()?;
    assert_eq!(flat.index.searchable_segment_ids()?.len(), 1);
    assert_eq!(bytes_by_ordinal(&flat)?, stored, "flat after the merge");

    let quantized = storage_spec("quantized", Metric::Cosine, &vectors).build()?;
    let searcher = quantized.index.reader()?.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    assert!(searcher.segment_readers()[0]
        .vector_index(quantized.field)?
        .has_exact_rows());
    assert_eq!(
        bytes_by_ordinal(&quantized)?,
        encoded,
        "quantized exact rows"
    );
    Ok(())
}

/// A quantized clustered field whose composite lacks ExactIdMap or ExactRows is corruption at
/// open.
#[test]
fn quantized_field_requires_both_exact_entries() -> crate::Result<()> {
    use crate::directory::{CompositeFile, CompositeWrite, Directory, RamDirectory};
    use crate::vector::storage::header::{read_vector_header, HEADER_LEN};
    for dropped in [VectorEntry::ExactIdMap, VectorEntry::ExactRows] {
        let directory = RamDirectory::create();
        let index = build_quantized_fixture_in_directory(
            QUANT_FIXTURE_DIM,
            Metric::L2,
            &[1, 4],
            true,
            directory.clone(),
        )?;
        let field = index.schema().get_field("embedding")?;
        let segment = index.searchable_segments()?.remove(0);
        let path = segment.relative_path(SegmentComponent::Custom(VEC_EXT.into()));
        let file = segment.open_read(SegmentComponent::Custom(VEC_EXT.into()))?;
        let (_, body) = read_vector_header(&file)?;
        let composite = CompositeFile::open(&body)?;
        let mut rewritten = Vec::new();
        write_vector_header(&mut rewritten)?;
        let mut writer = CompositeWrite::wrap(&mut rewritten);
        for entry in [
            VectorEntry::Data,
            VectorEntry::ExactRows,
            VectorEntry::IdMap,
            VectorEntry::ExactIdMap,
        ] {
            if entry == dropped {
                continue;
            }
            let bytes = composite
                .open_read_with_idx(field, entry.index())
                .expect("entry")
                .read_bytes()?;
            if matches!(entry, VectorEntry::Data | VectorEntry::ExactRows) {
                writer.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
            }
            writer
                .for_field_with_idx(field, entry.index())
                .write_all(&bytes)?;
        }
        writer.close()?;
        directory
            .delete(&path)
            .map_err(|error| TantivyError::InternalError(error.to_string()))?;
        let mut write = segment.open_write(SegmentComponent::Custom(VEC_EXT.into()))?;
        write.write_all(&rewritten)?;
        write.terminate()?;
        let reader = crate::SegmentReader::open(&segment)?;
        let error = reader.vector_index(field).err().expect("missing entry");
        assert!(
            matches!(error, TantivyError::DataCorruption(_)),
            "{dropped:?}: {error:?}"
        );
    }
    Ok(())
}
