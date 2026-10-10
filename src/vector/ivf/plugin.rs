#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cascade::prepare_centroid;

    use crate::index::{IndexSettings, SegmentComponent};
    use crate::indexer::NoMergePolicy;
    use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
    use crate::schema::{
        Field, IndexRecordOption, Metric, Schema, Term, VectorOptions, STORED, STRING,
    };
    use crate::vector::blocks::block_len;
    use crate::vector::distance::l2_squared;
    use crate::vector::header::VectorEntry;
    use crate::vector::ivf::{decode_row, AdaptiveProbeParams};
    use crate::vector::metadata::{SlotType, VectorColMetadata};
    use crate::vector::prepared::QuantizedQueryCtx;
    use crate::vector::tests::ground_truth;
    use crate::vector::{
        CentroidProducer, IvfCentroids, IvfMatrix, RouterKind, TopDocsByVectorSimilarity,
        VectorEstimatorMeasurements, VectorEstimatorQuery, VectorEstimatorSource,
        VectorQuantizationConfig, VectorQuantizationLayer, ENTRY_ALIGN, VEC_EXT,
    };
    use crate::{Index, TantivyDocument, TantivyError};

    #[test]
    fn quantization_merge_source_has_no_estimator_analysis_entrypoint() {
        let production_source = include_str!("writer.rs");
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
            VectorColMetadata::build_ivf(&VectorOptions::new(100, Metric::Dot), Some(&config))
                .unwrap()
                .runtime();
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

    // Mixed field contracts retain exact entry lengths and aligned Data boundaries.
    fn check_two_field_entries(empty_second: bool) -> crate::Result<()> {
        use common::{HasLen, OwnedBytes};

        use crate::directory::{CompositeFile, FileHandle, FileSlice};
        use crate::vector::blocks::{block_align, data_entry_len, Blocks};
        use crate::vector::header::read_vector_header;
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
                vector_quantization: vec![config],
                ..Default::default()
            })
            .centroid_producer(Arc::new(QuantFixtureCentroids {
                dim: 65,
                metric: Metric::L2,
            }))
            .ivf_router(RouterKind::Rng)?
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
                &TopDocsByVectorSimilarity::new(field, vec![0.25; 65], 3).with_adaptive_params(
                    AdaptiveProbeParams {
                        max_probe_fraction: 1.0,
                        min_probe_clusters: 2,
                        ..Default::default()
                    },
                ),
            )?;
            assert_eq!(hits.results.len(), count.min(3));
            assert_eq!(vector.id_map_initialized(), count > 0);
            let ivf = vector.index().unwrap();
            for b in 0..ivf.num_clusters() {
                let range = ivf.cluster_range(b);
                let docs = vector.cluster_doc_ids(b).unwrap().unwrap();
                assert!(docs.windows(2).all(|pair| pair[0] < pair[1]));
                for (row, doc) in range.zip(docs) {
                    assert_eq!(vector.row_id(doc)?, Some(row));
                    let bytes = vector.vector_bytes_for_row(row)?;
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
            let rows = (0..ivf.num_clusters())
                .map(|b| ivf.cluster_range(b).start)
                .chain(std::iter::once(count))
                .collect();
            let data = composite
                .open_read_with_idx(field, VectorEntry::Data.index())
                .unwrap();
            let id_map = composite
                .open_read_with_idx(field, VectorEntry::IdMap.index())
                .unwrap();
            assert_eq!(id_map.len(), 1 + count * 4);
            let start = data.storage_block_ord(0).unwrap();
            assert_eq!(start % ENTRY_ALIGN, 0);
            assert_eq!(data.len() % ENTRY_ALIGN, 0);
            let blocks = Blocks::open(data.clone(), &opts, count, Some(rows))?;
            assert_eq!(
                data.len(),
                data_entry_len(
                    *blocks.block_starts.last().unwrap() as usize,
                    blocks.block_rows.len() - 1
                )
            );
            aligns.push(block_align(&blocks.slots));
            data_ends.push(start + data.len());
            id_starts.push(id_map.storage_block_ord(0).unwrap());
        }
        assert_ne!(aligns[0], aligns[1]);
        assert_eq!(data_ends.iter().max(), id_starts.iter().min());
        assert!(data_ends
            .iter()
            .all(|end| id_starts.iter().all(|start| end <= start)));
        Ok(())
    }

    // Empty quantized IVF fields still carry exact metadata-only Data and Explicit IdMap
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
                        for (row, doc) in ivf.cluster_range(cluster).zip(docs) {
                            rows_by_doc.insert(
                                doc,
                                (cluster, vectors.vector_bytes_for_row(row)?.to_vec()),
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
                    assert!(vectors.id_map_initialized());
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

    struct QuantFixtureCentroids {
        dim: usize,
        metric: Metric,
    }

    impl CentroidProducer for QuantFixtureCentroids {
        fn centroids(&self, _field: Field, options: &VectorOptions) -> crate::Result<IvfCentroids> {
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
                    .map(|coordinate| {
                        center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1
                    })
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
        build_quantized_fixture_case_with_router(dim, metric, schedule, quantized, RouterKind::Rng)
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
            RouterKind::Rng,
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
        let mut settings = IndexSettings::default();
        if quantized {
            settings.vector_quantization = vec![quant_fixture_config_for(dim, metric, schedule)];
        }
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .centroid_producer(Arc::new(QuantFixtureCentroids { dim, metric }))
            .ivf_router(router)?
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
        let field =
            schema_builder.add_vector_field("embedding", VectorOptions::new(dim, Metric::L2));
        let schema = schema_builder.build();
        let settings = IndexSettings {
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
                    .map(|coordinate| {
                        center + ((doc * dim + coordinate) as f32 * 0.017).sin() * 0.1
                    })
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
        stats: &crate::vector::backend::ProbeStats,
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
            if !stats.quantized_trace.scored_docs.contains(&doc) {
                "routing/admission miss"
            } else if stats
                .quantized_trace
                .boundary_docs
                .iter()
                .any(|survivors| !survivors.contains(&doc))
            {
                "band drop"
            } else if !stats.quantized_trace.rerank_docs.contains(&doc) {
                "rerank fetch bug"
            } else {
                "rerank scoring/order bug"
            }
        } else {
            "rerank scoring/order bug"
        };
        panic!(
            "{context}: {attribution}; actual={actual:?} expected={expected:?} trace={:?}",
            stats.quantized_trace
        );
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
            .quantization()
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
        for row in 0..ivf.num_rows() {
            assert!(quantized.residual_norm(row)?.is_finite());
            for (layer, stored) in quantized.layers().iter().enumerate() {
                assert!((1.0..=4.0).contains(&stored.gamma(row)?));
                let corrected_error = stored.error_ratio(row)?;
                assert!(corrected_error.is_finite() && corrected_error >= 0.0);
                assert_eq!(
                    stored.constant(row)?.is_some(),
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
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), TOP_N)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 2,
                ..Default::default()
            })
            .with_max_scan_levels(depth);
        let fruit = searcher.search(filter, &collector)?;
        assert_eq!(fruit.stats.len(), 1);
        let stats = &fruit.stats[0];
        let context =
            format!("metric={metric:?} schedule={schedule:?} scenario={scenario:?} depth={depth}");
        assert_matrix_results(&context, &fruit.results, &expected, stats);

        let layer0 = stats.layers.get(0).expect("layer 0 must execute");
        assert_eq!(
            stats.layer0_eligible,
            layer0.scored(),
            "{context}: layer 0 must score exactly the admitted eligible rows"
        );
        assert_eq!(
            stats.eligible_charged,
            layer0.scored(),
            "{context}: the probe budget must charge exactly the selected rows"
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
                "{context}: the unfiltered L2 matrix cell must prove that boundary 0 measurably \
                 drops at least one scored candidate: {stats:?}"
            );
        }
        assert_eq!(
            stats.quantized_trace.boundary_docs.len(),
            depth,
            "{context}: one identity snapshot per executed boundary"
        );
        for boundary in &stats.quantized_trace.boundary_docs {
            assert!(
                boundary
                    .iter()
                    .all(|doc| stats.quantized_trace.scored_docs.contains(doc)),
                "{context}: a later layer retained a row absent from layer-0 selection"
            );
        }
        assert!(
            stats.quantized_trace.rerank_docs.iter().all(|doc| stats
                .quantized_trace
                .boundary_docs
                .last()
                .is_some_and(|boundary| boundary.contains(doc))),
            "{context}: rerank read a row absent from the final selection"
        );
        for max_probe_fraction in [0.25, 1.0] {
            const ELIGIBILITY_TOP_N: usize = 8;
            let params = AdaptiveProbeParams {
                max_probe_fraction,
                min_probe_clusters: 2,
                ..Default::default()
            };
            let quantized = searcher.search(
                filter,
                &TopDocsByVectorSimilarity::new(field, query.clone(), ELIGIBILITY_TOP_N)
                    .with_adaptive_params(params.clone())
                    .with_max_scan_levels(depth),
            )?;
            let level0 = searcher.search(
                filter,
                &TopDocsByVectorSimilarity::new(field, query.clone(), ELIGIBILITY_TOP_N)
                    .with_adaptive_params(params)
                    .with_max_scan_levels(0),
            )?;
            assert_eq!(
                quantized.stats[0].quantized_trace.scored_docs,
                level0.stats[0].quantized_trace.scored_docs,
                "{context}: selected candidate set at probe fraction {max_probe_fraction}"
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
            .quantization()
            .expect("configured IVF fixture must carry quantized slots");
        let (specs, grids) = (
            quantized.index_ctx().specs.clone(),
            quantized.index_ctx().grids.clone(),
        );
        let query: Vec<f32> = (0..dim)
            .map(|coordinate| ((coordinate as f32 + 0.5) * 0.031).cos())
            .collect();
        let harness = cascade::prepare_split_query(&query, &specs, &grids, 4);
        let scan = QuantizedQueryCtx::new(Arc::clone(quantized.index_ctx()), query);

        for row in 0..ivf.num_rows() {
            let mut scan_sum = 0.0;
            let mut harness_sum = 0.0;
            for (layer, stored) in quantized.layers().iter().enumerate() {
                let codes = stored.code_bytes(row)?;
                let scale = stored.scale(row)?;
                let constant = stored.constant(row)?;
                let scan_estimate = scan.score_layer(layer, &codes, scale, constant)?;
                let harness_estimate = match constant {
                    Some(constant) => {
                        harness.score_layer(layer, &codes, scale, constant, specs[layer])
                    }
                    None => {
                        harness.score_layer_without_constant(layer, &codes, scale, specs[layer])
                    }
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
    fn settings_v3_builds_v4_and_readers_ignore_global_changes() -> crate::Result<()> {
        let mut index = build_quantized_fixture(QUANT_FIXTURE_DIM, true)?;
        let field = index.schema().get_field("embedding")?;
        let config = &index.settings().vector_quantization[0];
        assert_eq!(config.format_version, 3);
        config.validate(&VectorOptions::new(QUANT_FIXTURE_DIM, Metric::L2))?;
        assert_eq!(&quantized_vec_file(&index)?[..4], &[6, 0, 0, 0]);
        index.settings_mut().vector_quantization.clear();
        let searcher = index.reader()?.searcher();
        let vector = searcher.segment_readers()[0].vector_index(field)?;
        assert_eq!(vector.quantization().unwrap().layers().len(), 2);
        assert!(vector.quantization().unwrap().residual_norm(0)?.is_finite());
        Ok(())
    }

    #[test]
    fn level_zero_matches_unquantized_ivf() -> crate::Result<()> {
        const DIM: usize = 64;
        let query = vec![0.05_f32; DIM];
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.5,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let unquantized = build_quantized_fixture(DIM, false)?;
        let quantized = build_quantized_fixture(DIM, true)?;
        let field = unquantized.schema().get_field("embedding")?;
        let unquantized_fruit = unquantized.reader()?.searcher().search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(field, query.clone(), 3)
                .with_adaptive_params(params.clone()),
        )?;
        let level_zero_collector = TopDocsByVectorSimilarity::new(field, query, 3)
            .with_adaptive_params(params)
            .with_max_scan_levels(0);
        let quantized_reader = quantized.reader()?;
        let quantized_searcher = quantized_reader.searcher();
        assert!(quantized_searcher.segment_readers()[0]
            .vector_index(field)?
            .quantization()
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
        assert_eq!(level_zero.exact_rows_read, baseline.exact_rows_read);
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
        use crate::vector::{Partition, Quantizer, VectorColMetadata};
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
                    crate::vector::VectorNormPolicy::None
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
                TopDocsByVectorSimilarity::new(field, fixture_search_query(Metric::L2, 64), 5)
                    .with_adaptive_params(AdaptiveProbeParams {
                        max_probe_fraction: 1.0,
                        min_probe_clusters: 2,
                        ..Default::default()
                    })
            };
            let collector = make_collector();
            let combined = searcher.search(&AllQuery, &collector)?;
            assert_eq!(collector.quantized_query_count(), if mixed { 2 } else { 1 });
            let weight = AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))?;
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
    fn reused_collector_prepares_query_per_quantization_config() -> crate::Result<()> {
        const DIM: usize = 64;
        let field_of = |index: &Index| index.schema().get_field("embedding");
        let first = build_quantized_fixture_case(DIM, Metric::L2, &[1, 4], true)?;
        let second = build_quantized_fixture_case(DIM, Metric::L2, &[4], true)?;
        let params = AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 2,
            ..Default::default()
        };
        let query = vec![0.05_f32; DIM];
        let collector = |field| {
            TopDocsByVectorSimilarity::new(field, query.clone(), 3)
                .with_adaptive_params(params.clone())
        };

        let reused = collector(field_of(&first)?);
        first.reader()?.searcher().search(&AllQuery, &reused)?;
        assert_eq!(reused.quantized_query_count(), 1);
        let mut reused_fruit = second.reader()?.searcher().search(&AllQuery, &reused)?;
        let mut fresh_fruit = second
            .reader()?
            .searcher()
            .search(&AllQuery, &collector(field_of(&second)?))?;
        assert_eq!(reused.quantized_query_count(), 2);
        for stats in reused_fruit.stats.iter_mut().chain(&mut fresh_fruit.stats) {
            stats.clear_stage_timings();
        }

        assert_eq!(reused_fruit.results, fresh_fruit.results);
        assert_eq!(
            format!("{:?}", reused_fruit.stats),
            format!("{:?}", fresh_fruit.stats)
        );
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
        assert!(vector_reader.quantization().is_none());

        let fruit = searcher.search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(field, query, 3).with_max_scan_levels(0),
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
        assert_eq!(stats.exact_rows_read, 8, "{stats:?}");
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
            .quantization()
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
            for row in ivf.cluster_range(cluster) {
                let vector = decode_row::<f32>(
                    &vector_reader.vector_bytes_for_row(row)?,
                    QUANT_FIXTURE_DIM,
                )?;
                let mut expected_input = vector.clone();
                let mut workspace = cascade::BatchEncodeWorkspace::new();
                let expected = cascade::encode_batch_in_place_with_workspace(
                    &mut expected_input,
                    1,
                    &prepared,
                    &specs,
                    &grids,
                    &mut workspace,
                    true,
                );
                assert_eq!(
                    quantized.residual_norm(row)?.to_bits(),
                    l2_squared(&vector, &centroid).to_bits()
                );
                for (layer, stored) in quantized.layers().iter().enumerate() {
                    let stored_codes = stored.code_bytes(row)?;
                    assert_eq!(
                        stored_codes.len(),
                        QUANT_FIXTURE_DIM * usize::from(specs[layer].bits) / 8,
                        "divisible dimensions use exact byte strides"
                    );
                    assert_eq!(
                        stored_codes.as_slice(),
                        expected.layers[layer].codes.as_slice()
                    );
                    assert_eq!(stored.scale(row)?, expected.layers[layer].scales[0]);
                    assert_eq!(
                        stored.gamma(row)?.to_bits(),
                        quant_model::f16::f16_to_f32(expected.layers[layer].gammas[0]).to_bits()
                    );
                    assert_eq!(
                        stored.error_ratio(row)?.to_bits(),
                        quant_model::f16::f16_to_f32(
                            expected.layers[layer].corrected_error_ratios[0]
                        )
                        .to_bits()
                    );
                    assert_eq!(
                        stored
                            .constant(row)?
                            .expect("L2 fixture requires split constants")
                            .to_bits(),
                        expected.layers[layer].constants[0].to_bits()
                    );
                }
            }
        }

        let query = vec![0.05_f32; QUANT_FIXTURE_DIM];
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 3)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 2,
                ..Default::default()
            });
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
        assert!(stats.rerank_rows <= layer1.survivors(), "{stats:?}");
        assert_eq!(stats.exact_rows_read, stats.rerank_rows, "{stats:?}");
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
    fn quantized_bounds_gate_skips_provably_useless_cluster() -> crate::Result<()> {
        let index = build_quantized_fixture_case(100, Metric::L2, &[1], true)?;
        let field = index.schema().get_field("embedding")?;
        let query = fixture_search_query(Metric::L2, 100);
        let searcher = index.reader()?.searcher();
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 3)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 1,
                ..Default::default()
            })
            .with_max_scan_levels(1);
        let fruit = searcher.search(&AllQuery, &collector)?;
        assert_eq!(fruit.stats.len(), 1);
        let stats = &fruit.stats[0];
        assert_eq!(
            fruit.results,
            fixture_exact_hits(&index, Metric::L2, &query, None, 3)?,
            "{stats:?}"
        );
        assert_eq!(stats.bounds_skips, 1, "{stats:?}");
        assert_eq!(stats.bound_armed_count, 1, "{stats:?}");
        assert_eq!(stats.bound_armed_probe_sum, 0, "{stats:?}");
        assert_eq!(stats.clusters_probed(), 1, "{stats:?}");
        Ok(())
    }

    #[test]
    fn quantized_probe_terminates_at_ceiling() -> crate::Result<()> {
        let index = build_quantized_fixture_case(100, Metric::L2, &[1], true)?;
        let field = index.schema().get_field("embedding")?;
        let query = fixture_search_query(Metric::L2, 100);
        let searcher = index.reader()?.searcher();
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 3)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 0.1,
                min_probe_clusters: 1,
                ..Default::default()
            })
            .with_max_scan_levels(1);
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

    /// Under the stacked router the quantized loop runs APS: a loose
    /// recall target stops before the full budget is spent.
    #[test]
    fn quantized_probe_terminates_at_recall_target() -> crate::Result<()> {
        let index = build_quantized_fixture_case_with_router(
            100,
            Metric::L2,
            &[1],
            true,
            RouterKind::Stacked,
        )?;
        let field = index.schema().get_field("embedding")?;
        let query = fixture_search_query(Metric::L2, 100);
        let searcher = index.reader()?.searcher();
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 3)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 1,
                recall_target: 0.5,
                ..Default::default()
            })
            .with_max_scan_levels(1);
        let fruit = searcher.search(&AllQuery, &collector)?;
        let stats = &fruit.stats[0];
        assert_eq!(
            stats.termination,
            crate::vector::ProbeTermination::RecallTarget,
            "{stats:?}"
        );
        assert!(
            stats
                .recall_estimate
                .is_some_and(|estimate| estimate >= 0.5),
            "{stats:?}"
        );
        assert_eq!(
            fruit.results,
            fixture_exact_hits(&index, Metric::L2, &query, None, 3)?,
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
        let params = AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 2,
            ..Default::default()
        };
        let vector_reader = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vector_reader.index().expect("merged fixture must be IVF");
        let (_, n_avg, open_share) =
            params.resolved_work_budget(ivf.num_clusters(), ivf.num_docs())?;
        let row = (1.0 - open_share) / n_avg;
        // Request all fixture rows so both scans open every cluster.
        let collector = TopDocsByVectorSimilarity::new(field, query.clone(), 8)
            .with_adaptive_params(params)
            .with_max_scan_levels(1);
        let unfiltered = searcher.search(&AllQuery, &collector)?;
        let filtered = searcher.search(&keep, &collector)?;
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
            .quantization()
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
            for row in ivf.cluster_range(cluster) {
                let vector = decode_row::<f32>(&vector_reader.vector_bytes_for_row(row)?, DIM)?;
                let residual: Vec<f32> = vector
                    .iter()
                    .zip(&centroid)
                    .map(|(&value, &center)| value - center)
                    .collect();
                let expected = cascade::encode_layers(&residual, Some(&prepared), &specs, &grids);
                for (layer, stored) in quantized.layers().iter().enumerate() {
                    assert_eq!(stored.code_bytes(row)?.as_slice(), expected.codes[layer]);
                    assert_eq!(stored.scale(row)?, expected.scales[layer]);
                    assert_eq!(
                        stored
                            .constant(row)?
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
                &TopDocsByVectorSimilarity::new(field, query.clone(), 3).with_adaptive_params(
                    AdaptiveProbeParams {
                        max_probe_fraction: 1.0,
                        min_probe_clusters: 2,
                        ..Default::default()
                    },
                ),
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
        let entry_len = |meta: &VectorColMetadata| {
            use crate::vector::blocks::{align_up, block_align, data_entry_len};
            data_entry_len(
                align_up(4 + meta.to_bytes().len(), block_align(&meta.slots()))
                    + 2 * block_len(&meta.slots(), ROWS / 2),
                2,
            )
        };
        let expected_growth = entry_len(&quant_meta) - entry_len(&plain_meta);
        assert!(
            physical_growth.abs_diff(expected_growth) <= 8,
            "only composite footer varints may differ"
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
        assert!(vector_reader.quantization().is_some());
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
            &TopDocsByVectorSimilarity::new(field, queries[0].clone(), 3).with_adaptive_params(
                AdaptiveProbeParams {
                    max_probe_fraction: 1.0,
                    min_probe_clusters: 2,
                    ..Default::default()
                },
            ),
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
        let live_posting_rows = (0..vector_reader.index().unwrap().num_rows())
            .filter(|&row| alive.is_alive(vector_reader.doc_id_at(row).unwrap()))
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
        use crate::vector::storage_io::test_support::{PagedDirectory, PAGE_BYTES};
        use crate::vector::Stage;
        let directory = PagedDirectory::default();
        let index =
            build_quantized_fixture_in_directory(64, Metric::L2, &[1], true, directory.clone())?;
        let field = index.schema().get_field("embedding")?;
        let searcher = index.reader()?.searcher();
        directory.reads.lock().unwrap().clear();
        let result = searcher.search(
            &AllQuery,
            &TopDocsByVectorSimilarity::new(field, fixture_search_query(Metric::L2, 64), 16)
                .with_adaptive_params(AdaptiveProbeParams {
                    max_probe_fraction: 1.0,
                    min_probe_clusters: 2,
                    ..Default::default()
                }),
        )?;
        let reads = directory.reads.lock().unwrap();
        let ranges: Vec<_> = reads
            .iter()
            .filter_map(|(stage, range)| matches!(stage, Stage::LayerScan(0)).then_some(range))
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
}
