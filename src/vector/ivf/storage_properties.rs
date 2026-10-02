use super::*;
use crate::schema::Value;
use crate::vector::prepared::{corrected_quantized_estimate, ArithmeticError};
use crate::vector::storage_io::test_support::{PagedDirectory, PAGE_BYTES};

const DIM: usize = 64;
const DOCS: usize = 1200;

struct FixedClusters;
impl IvfClusterer for FixedClusters {
    fn training_sample_ratio(&self) -> f32 {
        1.0
    }
    fn train(&self, _: &VectorOptions, _: IvfTrainingVectors) -> crate::Result<IvfCentroids> {
        let mut values = vec![0.0; 4 * DIM];
        for cluster in 0..4 {
            values[cluster * DIM + cluster] = 1.0;
        }
        Ok(IvfCentroids::F32(IvfMatrix {
            values,
            rows: 4,
            dims: DIM,
        }))
    }
    fn assign(
        &self,
        _: &VectorOptions,
        vectors: IvfVectors<'_>,
        _: &IvfCentroids,
    ) -> crate::Result<Vec<u32>> {
        let IvfVectors::F32(vectors) = vectors;
        Ok(vectors
            .matrix
            .values
            .chunks_exact(DIM)
            .map(|row| (0..4).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap() as u32)
            .collect())
    }
}
fn present(doc: usize) -> bool {
    doc == 0 || !doc.is_multiple_of(97)
}
fn cluster_for(doc: usize) -> usize {
    if doc == 0 {
        1
    } else {
        2 + doc % 2
    }
}
fn input_vector(doc: usize) -> Vec<f32> {
    let mut row: Vec<f32> = (0..DIM)
        .map(|i| ((doc * DIM + i) as f32 * 0.071).sin() * 0.025)
        .collect();
    row[cluster_for(doc)] += 1.0;
    row
}
fn fixture(metric: Metric, schedule: &[u8]) -> crate::Result<(Index, PagedDirectory)> {
    fixture_with_deletes(metric, schedule, true)
}
fn fixture_with_deletes(
    metric: Metric,
    schedule: &[u8],
    deletes: bool,
) -> crate::Result<(Index, PagedDirectory)> {
    fixture_with_seed(metric, schedule, deletes, None)
}
fn fixture_with_seed(
    metric: Metric,
    schedule: &[u8],
    deletes: bool,
    seed: Option<u64>,
) -> crate::Result<(Index, PagedDirectory)> {
    let mut schema = Schema::builder();
    let vector = schema.add_vector_field("embedding", VectorOptions::new(DIM, metric));
    let label = schema.add_text_field("label", STRING | STORED);
    let ordinal = schema.add_u64_field("ordinal", STORED);
    let directory = PagedDirectory::default();
    let index = Index::builder()
        .schema(schema.build())
        .settings(IndexSettings {
            vector_clustering_threshold: 1,
            vector_quantization: if schedule.is_empty() {
                Vec::new()
            } else {
                let mut config = quant_fixture_config_for(DIM, metric, schedule);
                if let Some(seed) = seed {
                    for (layer, quant) in config.layers.iter_mut().enumerate() {
                        quant.seed = seed + layer as u64;
                    }
                }
                vec![config]
            },
            ..Default::default()
        })
        .ivf_clusterer(Arc::new(FixedClusters))
        .ivf_router(RouterKind::Exact)?
        .create(directory.clone())?;
    let mut writer = index.writer_with_num_threads(1, 30_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let mut segments = Vec::new();
    for doc in 0..DOCS {
        let mut document = TantivyDocument::new();
        document.add_u64(ordinal, doc as u64);
        document.add_text(label, format!("d{doc}"));
        if doc % 3 != 0 {
            document.add_text(label, "keep");
        }
        if present(doc) {
            document.add_text(label, "vector");
            document.add_vector(vector, &input_vector(doc));
        }
        writer.add_document(document)?;
        if doc == DOCS / 2 - 1 || doc == DOCS - 1 {
            writer.commit()?;
            for id in index.searchable_segment_ids()? {
                if !segments.contains(&id) {
                    segments.push(id);
                }
            }
        }
    }
    writer.merge(&segments).wait()?;
    for doc in [101, 202].into_iter().filter(|_| deletes) {
        writer.delete_term(Term::from_field_text(label, &format!("d{doc}")));
    }
    writer.commit()?;
    writer.wait_merging_threads()?;
    Ok((index, directory))
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn encoder_columns_and_read_paths() -> crate::Result<()> {
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        for schedule in [&[1][..], &[1, 4][..], &[1, 2, 4][..]] {
            verify_case(metric, schedule, false)?;
        }
    }
    Ok(())
}
#[test]
fn explicit_cluster_probes() -> crate::Result<()> {
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        for schedule in [&[1][..], &[1, 4][..], &[1, 2, 4][..]] {
            verify_case(metric, schedule, true)?;
        }
    }
    Ok(())
}
fn verify_case(metric: Metric, schedule: &[u8], probe: bool) -> crate::Result<()> {
    let (index, directory) = fixture(metric, schedule)?;
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    let segment = &searcher.segment_readers()[0];
    let field = index.schema().get_field("embedding")?;
    let ordinal = index.schema().get_field("ordinal")?;
    let label = index.schema().get_field("label")?;
    let vectors = segment.vector_index(field)?;
    let ivf = vectors.index().unwrap();
    let quantized = vectors.quantization().unwrap();
    let ctx = quantized.index_ctx();
    let metadata = vectors.metadata().unwrap();
    let slots = metadata.slots();
    let centroids = ivf.centroid_bytes()?;
    let originals: Vec<usize> = (0..segment.max_doc())
        .map(|doc| {
            let stored: TantivyDocument = searcher.doc(crate::DocAddress::new(0, doc)).unwrap();
            stored.get_first(ordinal).unwrap().as_u64().unwrap() as usize
        })
        .collect();
    let mut encodings = Vec::new();
    let mut row_docs = Vec::new();
    let mut expected_rows = vec![None; segment.max_doc() as usize];
    assert_eq!(
        ivf.cluster_range(0).len(),
        0,
        "cluster=0 row=all column=Rows empty fixture"
    );
    assert_eq!(
        ivf.cluster_range(1).len(),
        1,
        "cluster=1 row=all column=Rows singleton fixture"
    );
    for cluster in 0..ivf.num_clusters() {
        let rows = ivf.cluster_range(cluster);
        let row_bytes = vectors.read_cluster_rows(cluster)?;
        let mut docs = Vec::new();
        vectors.read_doc_ids(cluster, &mut docs)?;
        let centroid = floats(&centroids[cluster * DIM * 4..(cluster + 1) * DIM * 4]);
        let prepared = cascade::prepare_centroid(&centroid, &ctx.specs);
        let mut values = floats(&row_bytes);
        let encoded = cascade::encode_batch_in_place(
            &mut values,
            rows.len(),
            &prepared,
            &ctx.specs,
            &ctx.grids,
        );
        let written: Vec<_> = originals
            .iter()
            .enumerate()
            .filter_map(|(doc, &original)| {
                (present(original) && cluster_for(original) == cluster).then_some(doc as u32)
            })
            .collect();
        assert_eq!(
            docs, written,
            "{metric:?} {schedule:?} cluster={cluster} row=all column=DocIds"
        );
        for (local, &doc) in docs.iter().enumerate() {
            let row = rows.start + local;
            expected_rows[doc as usize] = Some(row);
            assert_eq!(
                vectors.row_id(doc)?,
                Some(row),
                "cluster={cluster} row={local} column=DocLocations"
            );
            assert_eq!(
                vectors.vector_bytes(doc)?.unwrap().as_slice(),
                &row_bytes[local * DIM * 4..(local + 1) * DIM * 4],
                "cluster={cluster} row={local} column=Rows locate"
            );
        }
        row_docs.extend(docs.iter().copied());
        for (column, slot) in slots.iter().enumerate() {
            let expected = match slot.slot_type {
                SlotType::Rows { .. } => row_bytes.to_vec(),
                SlotType::DocIds => docs.iter().flat_map(|doc| doc.to_le_bytes()).collect(),
                SlotType::ResidualNorms => f32_bytes(&encoded.residual_norms_squared),
                SlotType::QuantLayerCodes { layer, .. } => {
                    encoded.layers[layer as usize].codes.clone()
                }
                SlotType::QuantLayerScales { layer } => {
                    f32_bytes(&encoded.layers[layer as usize].scales)
                }
                SlotType::QuantLayerGammas { layer } => {
                    u16_bytes(&encoded.layers[layer as usize].gammas)
                }
                SlotType::QuantLayerErrors { layer } => {
                    u16_bytes(&encoded.layers[layer as usize].corrected_error_ratios)
                }
                SlotType::QuantLayerConstants { layer } => {
                    f32_bytes(&encoded.layers[layer as usize].constants)
                }
            };
            if rows.is_empty() {
                assert!(
                    expected.is_empty(),
                    "cluster={cluster} row=all column={column}"
                );
                continue;
            }
            let actual = quantized.layers()[0].read_column(column, rows.clone())?;
            assert_eq!(
                actual.len(),
                expected.len(),
                "cluster={cluster} row=all column={column} length"
            );
            for local in 0..rows.len() {
                let offset = local * slot.stride as usize;
                let bytes = offset..offset + slot.stride as usize;
                assert_eq!(
                    &actual[bytes.clone()],
                    &expected[bytes.clone()],
                    "{metric:?} {schedule:?} cluster={cluster} row={local} column={column} encoder"
                );
                let single = quantized.layers()[0]
                    .read_column(column, rows.start + local..rows.start + local + 1)?;
                assert_eq!(
                    single.as_slice(),
                    &expected[bytes],
                    "cluster={cluster} row={local} column={column} single-row"
                );
            }
        }
        if !rows.is_empty() {
            for (layer_idx, layer) in quantized.layers().iter().enumerate() {
                let band = layer.read_batch_in_block(cluster, rows.clone())?;
                let sidecar_columns = layer.read_sidecar(rows.clone())?;
                let sparse = layer.cluster_in_block(cluster)?;
                let sidecars = sparse.read_sidecar()?;
                let selected: Vec<_> = rows
                    .clone()
                    .filter(|row| row % 17 == 0 || *row == rows.start || *row == rows.end - 1)
                    .collect();
                let mut selected_scales = vec![0.0; selected.len()];
                let mut selected_gammas = vec![0.0; selected.len()];
                let mut selected_errors = vec![0.0; selected.len()];
                let mut selected_constants = if metric == Metric::L2 {
                    vec![0.0; selected.len()]
                } else {
                    Vec::new()
                };
                sidecars.decode_selected(
                    &selected,
                    &mut selected_scales,
                    &mut selected_gammas,
                    &mut selected_errors,
                    &mut selected_constants,
                )?;
                for (i, &row) in selected.iter().enumerate() {
                    let context = format!(
                        "cluster={cluster} row={} column=layer{layer_idx}",
                        row - rows.start
                    );
                    assert_eq!(
                        selected_scales[i].to_bits(),
                        band.scale(row)?.to_bits(),
                        "{context}.scales selected"
                    );
                    assert_eq!(
                        selected_gammas[i].to_bits(),
                        band.gamma(row)?.to_bits(),
                        "{context}.gammas selected"
                    );
                    assert_eq!(
                        selected_errors[i].to_bits(),
                        band.error_ratio(row)?.to_bits(),
                        "{context}.errors selected"
                    );
                    assert_eq!(
                        selected_constants.get(i).copied().map(f32::to_bits),
                        band.constant(row)?.map(f32::to_bits),
                        "{context}.constants selected"
                    );
                }
                let mut ranges = Vec::new();
                sparse.plan_codes(&selected, &mut ranges);
                let mut covered = Vec::new();
                for range in ranges {
                    let planned = sparse.read_codes(range.clone())?;
                    for row in selected.iter().copied().filter(|row| range.contains(row)) {
                        let start = (row - range.start) * layer.code_stride();
                        assert_eq!(
                            &planned[start..start + layer.code_stride()],
                            band.code_bytes(row)?,
                            "cluster={cluster} row={} column=layer{layer_idx}.codes sparse",
                            row - rows.start
                        );
                        covered.push(row);
                    }
                }
                assert_eq!(
                    covered, selected,
                    "cluster={cluster} row=selected column=layer{layer_idx}.codes coverage"
                );
                for row in rows.clone() {
                    let context = format!(
                        "cluster={cluster} row={} column=layer{layer_idx}",
                        row - rows.start
                    );
                    assert_eq!(
                        band.code_bytes(row)?,
                        layer.code_bytes(row)?.as_slice(),
                        "{context}.codes band"
                    );
                    for actual in [
                        band.scale(row)?,
                        sidecars.scale(row)?,
                        sidecar_columns.scale(row)?,
                    ] {
                        assert_eq!(
                            actual.to_bits(),
                            layer.scale(row)?.to_bits(),
                            "{context}.scales"
                        );
                    }
                    for actual in [
                        band.gamma(row)?,
                        sidecars.gamma(row)?,
                        sidecar_columns.gamma(row)?,
                    ] {
                        assert_eq!(
                            actual.to_bits(),
                            layer.gamma(row)?.to_bits(),
                            "{context}.gammas"
                        );
                    }
                    for actual in [
                        band.error_ratio(row)?,
                        sidecars.error_ratio(row)?,
                        sidecar_columns.error_ratio(row)?,
                    ] {
                        assert_eq!(
                            actual.to_bits(),
                            layer.error_ratio(row)?.to_bits(),
                            "{context}.errors"
                        );
                    }
                    assert_eq!(
                        band.constant(row)?.map(f32::to_bits),
                        layer.constant(row)?.map(f32::to_bits),
                        "{context}.constants"
                    );
                }
                if layer_idx == 0 {
                    assert_eq!(
                        band.residual_norms.as_ref().unwrap().as_slice(),
                        f32_bytes(&encoded.residual_norms_squared),
                        "cluster={cluster} row=all column=ResidualNorms band"
                    );
                }
            }
        }
        encodings.push(encoded);
    }
    for (doc, &original) in originals.iter().enumerate() {
        assert_eq!(
            vectors.row_id(doc as u32)?,
            expected_rows[doc],
            "cluster=none row=doc{doc} column=DocLocations"
        );
        if !present(original) {
            assert!(
                vectors.vector_bytes(doc as u32)?.is_none(),
                "cluster=none row=doc{doc} column=Rows absent"
            );
        }
    }
    assert!(
        directory
            .reads
            .lock()
            .unwrap()
            .iter()
            .any(|(_, range)| range.start / PAGE_BYTES != (range.end - 1) / PAGE_BYTES),
        "cluster=all row=all column=all fixture must cross pages"
    );
    if !probe {
        return Ok(());
    }
    let query = input_vector(42);
    let prepared = QuantizedQueryCtx::new(Arc::clone(ctx), query.clone());
    let exact_query = crate::vector::prepared::PreparedQuery::new(metric, Arc::new(query.clone()));
    for clusters in [vec![0, 1, 2, 3], vec![0, 1, 3]] {
        for filtered in [false, true] {
            let candidates = clusters
                .iter()
                .map(|&cluster| crate::vector::ivf::Candidate {
                    node: cluster as u32,
                    sim: metric.similarity_bytes::<f32>(
                        prepared.query(),
                        &centroids[cluster * DIM * 4..(cluster + 1) * DIM * 4],
                    ),
                })
                .collect();
            let keep = TermQuery::new(
                Term::from_field_text(label, "keep"),
                IndexRecordOption::Basic,
            );
            let filter: &dyn Query = if filtered { &keep } else { &AllQuery };
            let collector = TopDocsByVectorSimilarity::new(field, query.clone(), DOCS + 1)
                .with_adaptive_params(AdaptiveProbeParams {
                    max_probe_fraction: 1.0,
                    min_probe_clusters: 4,
                    ..Default::default()
                });
            let fruit = crate::vector::router::with_test_clusters(candidates, || {
                searcher.search(filter, &collector)
            })
            .unwrap_or_else(|error| {
                panic!(
                    "{metric:?} {schedule:?} cluster={clusters:?} row=all column=layer0 probe: \
                     {error}"
                )
            });
            let stats = &fruit.stats[0];
            assert!(
                matches!(
                    stats.routing,
                    Some(crate::vector::router::RouterMetrics::Exact { visited_count: 0 })
                ),
                "cluster={clusters:?} row=all column=router bypass"
            );
            let mut expected_docs = Vec::new();
            let mut pruned_dead = 0;
            let mut pruned_filter = 0;
            for &cluster in &clusters {
                for row in ivf.cluster_range(cluster) {
                    let doc = row_docs[row];
                    let original = originals[doc as usize];
                    let live = segment
                        .alive_bitset()
                        .is_none_or(|alive| alive.is_alive(doc));
                    // The row gate evaluates the filter before visibility.
                    if filtered && original % 3 == 0 {
                        pruned_filter += 1;
                    } else if !live {
                        pruned_dead += 1;
                    } else {
                        expected_docs.push(doc);
                    }
                }
            }
            expected_docs.sort_unstable();
            assert_eq!(
                stats.quantized_trace.scored_docs, expected_docs,
                "{metric:?} {schedule:?} cluster={clusters:?} row=all column=DocIds coverage \
                 filtered={filtered}"
            );
            assert_eq!(
                stats.pruned_dead, pruned_dead,
                "cluster={clusters:?} row=all column=DocIds dead"
            );
            assert_eq!(
                stats.pruned_filter, pruned_filter,
                "cluster={clusters:?} row=all column=DocIds filtered"
            );
            assert_eq!(stats.quantized_trace.estimates.len(), schedule.len());
            for (layer, trace) in stats.quantized_trace.estimates.iter().enumerate() {
                assert_eq!(
                    trace.len(),
                    expected_docs.len(),
                    "cluster={clusters:?} row=all column=layer{layer} coverage"
                );
                for &(row, doc, bits) in trace {
                    let cluster = (0..4)
                        .find(|&c| ivf.cluster_range(c).contains(&row))
                        .unwrap();
                    let local = row - ivf.cluster_range(cluster).start;
                    assert_eq!(
                        doc, row_docs[row],
                        "cluster={cluster} row={local} column=DocIds layer{layer}"
                    );
                    let encoded = &encodings[cluster];
                    let score = metric
                        .similarity_bytes::<f32>(
                            prepared.query(),
                            &centroids[cluster * DIM * 4..(cluster + 1) * DIM * 4],
                        )
                        .score();
                    let norm = encoded.residual_norms_squared[local];
                    let base = if metric == Metric::L2 {
                        score - norm
                    } else {
                        score
                    };
                    let mut raw = 0.0;
                    let mut arithmetic = ArithmeticError::default();
                    for l in 0..=layer {
                        let e = &encoded.layers[l];
                        let stride = quantized.layers()[l].code_stride();
                        raw = crate::vector::index_reader::diagnostic_advance_raw_prefix(
                            &prepared,
                            metric,
                            l,
                            &e.codes[local * stride..(local + 1) * stride],
                            e.scales[local],
                            (metric == Metric::L2).then(|| e.constants[local]),
                            raw,
                            score,
                            norm,
                            &mut arithmetic,
                        )?;
                    }
                    let estimate = corrected_quantized_estimate(
                        metric,
                        quant_model::f16::f16_to_f32(encoded.layers[layer].gammas[local]),
                        raw,
                        base,
                    );
                    assert_eq!(
                        bits,
                        estimate.to_bits(),
                        "{metric:?} {schedule:?} cluster={cluster} row={local} \
                         column=layer{layer}.estimate"
                    );
                }
            }
            let mut expected: Vec<_> = expected_docs
                .iter()
                .map(|&doc| {
                    let original = originals[doc as usize];
                    let mut bytes = f32_bytes(&input_vector(original));
                    let options = VectorOptions::new(DIM, metric);
                    // A flushed source and the merged row each obey the field's normalization rule.
                    maybe_normalize_bytes(&options, &mut bytes);
                    maybe_normalize_bytes(&options, &mut bytes);
                    (
                        exact_query.score_doc_bytes(&bytes),
                        crate::DocAddress::new(0, doc),
                    )
                })
                .collect();
            expected.sort_by(|(a, ad), (b, bd)| b.total_cmp(a).then_with(|| ad.cmp(bd)));
            assert_eq!(
                fruit.results.len(),
                expected.len(),
                "cluster={clusters:?} row=all column=Rows top-k length"
            );
            for (rank, (actual, expected)) in fruit.results.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.1, expected.1,
                    "cluster={clusters:?} row=rank{rank} column=Rows top-k"
                );
                assert_eq!(
                    actual.0.to_bits(),
                    expected.0.to_bits(),
                    "cluster={clusters:?} row=doc{} column=Rows exact score",
                    actual.1.doc_id
                );
            }
        }
    }
    Ok(())
}

#[test]
fn zero_row_cluster_skips_band_reads() -> crate::Result<()> {
    let (index, directory) = fixture(Metric::L2, &[1])?;
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    let field = index.schema().get_field("embedding")?;
    let vectors = searcher.segment_readers()[0].vector_index(field)?;
    let ivf = vectors.index().unwrap();
    assert_eq!(
        ivf.cluster_range(0).len(),
        0,
        "cluster=0 row=all column=Rows"
    );
    let query = input_vector(42);
    let centroid_bytes = ivf.centroid_bytes()?;
    let candidate = crate::vector::ivf::Candidate {
        node: 0,
        sim: Metric::L2.similarity_bytes::<f32>(&query, &centroid_bytes[..DIM * 4]),
    };
    let collector = TopDocsByVectorSimilarity::new(field, query, DOCS + 1).with_adaptive_params(
        AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 4,
            ..Default::default()
        },
    );
    directory.reads.lock().unwrap().clear();
    let fruit = crate::vector::router::with_test_clusters(vec![candidate], || {
        searcher.search(&AllQuery, &collector)
    })?;
    let stats = &fruit.stats[0];
    assert!(
        fruit.results.is_empty(),
        "cluster=0 row=all column=Rows results"
    );
    assert_eq!(
        stats.clusters_skipped_empty, 1,
        "cluster=0 row=all column=Rows empty counter"
    );
    assert_eq!(
        stats.postings_skipped, 1,
        "cluster=0 row=all column=Rows posting counter"
    );
    assert_eq!(
        stats.vectors_visited, 0,
        "cluster=0 row=all column=Rows visits"
    );
    assert_eq!(
        stats.layers.get(0).unwrap().io.reads,
        0,
        "cluster=0 row=all column=band0 reads"
    );
    assert!(
        !directory
            .reads
            .lock()
            .unwrap()
            .iter()
            .any(|(stage, _)| matches!(stage, crate::vector::Stage::LayerScan(0))),
        "cluster=0 row=all column=band0 requested ranges"
    );
    Ok(())
}

/// Captures one column request per nonempty cluster before the measured probe.
fn doc_column_ranges(
    vectors: &crate::vector::VectorIndexReader,
    directory: &PagedDirectory,
) -> crate::Result<Vec<Option<std::ops::Range<usize>>>> {
    let mut docs = Vec::new();
    let mut ranges = Vec::new();
    for cluster in 0..vectors.index().unwrap().num_clusters() {
        directory.reads.lock().unwrap().clear();
        vectors.read_doc_ids(cluster, &mut docs)?;
        let reads = directory.reads.lock().unwrap();
        assert_eq!(
            reads.len(),
            usize::from(!docs.is_empty()),
            "cluster={cluster} column=DocIds"
        );
        ranges.push(reads.first().map(|(_, range)| range.clone()));
    }
    directory.reads.lock().unwrap().clear();
    Ok(ranges)
}

#[test]
fn open_gate_reads_doc_ids_only_for_rerank() -> crate::Result<()> {
    use crate::vector::Stage;
    let (index, directory) = fixture_with_deletes(Metric::L2, &[1, 4], false)?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let segment = &searcher.segment_readers()[0];
    assert!(segment.alive_bitset().is_none());
    let field = index.schema().get_field("embedding")?;
    let label = index.schema().get_field("label")?;
    let vectors = segment.vector_index(field)?;
    let ranges = doc_column_ranges(&vectors, &directory)?;
    let collector = TopDocsByVectorSimilarity::new(field, input_vector(42), 3)
        .with_adaptive_params(AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 4,
            ..Default::default()
        });
    let open = searcher.search(&AllQuery, &collector)?;
    let reads = directory.reads.lock().unwrap().clone();
    let overlaps =
        |a: &std::ops::Range<usize>, b: &std::ops::Range<usize>| a.start < b.end && b.start < a.end;
    for (stage, range) in &reads {
        if matches!(stage, Stage::LayerScan(_)) {
            assert!(
                !ranges.iter().flatten().any(|docs| overlaps(range, docs)),
                "stage={stage:?} range={range:?} touches DocIds"
            );
        }
    }
    let rerank_docs = &open.stats[0].quantized_trace.rerank_docs;
    for (cluster, range) in ranges.iter().enumerate() {
        let Some(range) = range else {
            continue;
        };
        let mut docs = Vec::new();
        vectors.read_doc_ids(cluster, &mut docs)?;
        let expected = usize::from(docs.iter().any(|doc| rerank_docs.contains(doc)));
        let actual = reads
            .iter()
            .filter(|(stage, read)| matches!(stage, Stage::RerankFetch) && read == range)
            .count();
        assert_eq!(
            actual, expected,
            "cluster={cluster} column=DocIds rerank requests"
        );
        assert_eq!(
            reads.iter().filter(|(_, read)| read == range).count(),
            expected,
            "cluster={cluster} column=DocIds extra trace requests"
        );
    }
    for layer in 0..2 {
        let requested: Vec<_> = reads
            .iter()
            .filter(|(stage, _)| matches!(stage, Stage::LayerScan(l) if *l as usize == layer))
            .map(|(_, r)| r)
            .collect();
        let stats = open.stats[0].layers.get(layer).unwrap();
        assert_eq!(stats.io.reads, requested.len() as u64);
        assert_eq!(
            stats.io.bytes_read,
            requested.iter().map(|r| r.len() as u64).sum::<u64>()
        );
    }
    let all_vectors = TermQuery::new(
        Term::from_field_text(label, "vector"),
        IndexRecordOption::Basic,
    );
    let filtered = searcher.search(&all_vectors, &collector)?;
    assert_eq!(open.results, filtered.results);
    assert_eq!(
        open.stats[0].quantized_trace.scored_docs,
        filtered.stats[0].quantized_trace.scored_docs
    );
    assert_eq!(
        open.stats[0].quantized_trace.estimates,
        filtered.stats[0].quantized_trace.estimates
    );
    assert_eq!(
        open.stats[0].quantized_trace.boundary_docs,
        filtered.stats[0].quantized_trace.boundary_docs
    );
    Ok(())
}

#[test]
fn filtered_clusters_read_doc_ids_before_payload() -> crate::Result<()> {
    use crate::vector::Stage;
    for schedule in [&[][..], &[1, 4][..]] {
        let (index, directory) = fixture_with_deletes(Metric::L2, schedule, false)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let field = index.schema().get_field("embedding")?;
        let label = index.schema().get_field("label")?;
        let vectors = searcher.segment_readers()[0].vector_index(field)?;
        let ivf = vectors.index().unwrap();
        let doc_ranges = doc_column_ranges(&vectors, &directory)?;
        let mut row_ranges = Vec::new();
        for cluster in 0..ivf.num_clusters() {
            directory.reads.lock().unwrap().clear();
            vectors.read_cluster_rows(cluster)?;
            row_ranges.push(
                directory
                    .reads
                    .lock()
                    .unwrap()
                    .first()
                    .map(|(_, range)| range.clone()),
            );
        }
        let query = input_vector(0);
        let prepared =
            crate::vector::prepared::PreparedQuery::new(Metric::L2, Arc::new(query.clone()));
        let centroids = ivf.centroid_bytes()?;
        let candidates = (0..ivf.num_clusters())
            .map(|cluster| crate::vector::ivf::Candidate {
                node: cluster as u32,
                sim: Metric::L2.similarity_bytes::<f32>(
                    prepared.query(),
                    &centroids[cluster * DIM * 4..(cluster + 1) * DIM * 4],
                ),
            })
            .collect();
        let collector = TopDocsByVectorSimilarity::new(field, query, DOCS + 1)
            .with_adaptive_params(AdaptiveProbeParams {
                max_probe_fraction: 1.0,
                min_probe_clusters: 4,
                ..Default::default()
            });
        let filter = TermQuery::new(Term::from_field_text(label, "d0"), IndexRecordOption::Basic);
        directory.reads.lock().unwrap().clear();
        let result = crate::vector::router::with_test_clusters(candidates, || {
            searcher.search(&filter, &collector)
        })?;
        assert_eq!(result.results.len(), 1);
        let reads = directory.reads.lock().unwrap();
        let scans: Vec<_> = reads
            .iter()
            .filter(|(stage, _)| {
                if schedule.is_empty() {
                    matches!(stage, Stage::ExactScan)
                } else {
                    matches!(stage, Stage::LayerScan(0))
                }
            })
            .map(|(_, range)| range)
            .collect();
        for cluster in [2, 3] {
            let docs = doc_ranges[cluster].as_ref().unwrap();
            let start = row_ranges[cluster].as_ref().unwrap().start;
            let end = doc_ranges
                .iter()
                .skip(cluster + 1)
                .flatten()
                .next()
                .map_or(usize::MAX, |r| {
                    row_ranges[cluster + 1].as_ref().unwrap().start.min(r.start)
                });
            let cluster_reads: Vec<_> = scans
                .iter()
                .filter(|r| r.start >= start && r.start < end)
                .copied()
                .collect();
            assert_eq!(
                cluster_reads,
                vec![docs],
                "schedule={schedule:?} cluster={cluster} column=DocIds only"
            );
        }
        if schedule.is_empty() {
            let mut expected: Vec<_> = doc_ranges.iter().flatten().collect();
            expected.push(row_ranges[1].as_ref().unwrap());
            expected.sort_by_key(|range| range.start);
            let mut actual = scans.clone();
            actual.sort_by_key(|range| range.start);
            assert_eq!(
                actual, expected,
                "exact filter reads only DocIds and singleton survivor Rows"
            );
        }
    }
    Ok(())
}

#[test]
fn exact_filter_reads_only_survivor_pages() -> crate::Result<()> {
    use crate::query::BooleanQuery;
    use crate::vector::Stage;
    let (index, directory) = fixture_with_deletes(Metric::L2, &[], false)?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let segment = &searcher.segment_readers()[0];
    let field = index.schema().get_field("embedding")?;
    let label = index.schema().get_field("label")?;
    let vectors = segment.vector_index(field)?;
    let doc_ranges = doc_column_ranges(&vectors, &directory)?;
    let filter = BooleanQuery::union(
        ["d42", "d242"]
            .iter()
            .map(|name| {
                Box::new(TermQuery::new(
                    Term::from_field_text(label, name),
                    IndexRecordOption::Basic,
                )) as Box<dyn Query>
            })
            .collect(),
    );
    let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
    let mut selected = Vec::new();
    weight.for_each_no_score(segment, &mut |docs| selected.extend_from_slice(docs))?;
    assert_eq!(selected.len(), 2);
    let query = input_vector(42);
    let prepared = crate::vector::prepared::PreparedQuery::new(Metric::L2, Arc::new(query.clone()));
    let mut expected = Vec::new();
    let mut pages = std::collections::BTreeSet::new();
    for doc in selected {
        directory.reads.lock().unwrap().clear();
        let bytes = vectors.vector_bytes(doc)?.unwrap();
        expected.push((
            prepared.score_doc_bytes(&bytes),
            crate::DocAddress::new(0, doc),
        ));
        let reads = directory.reads.lock().unwrap();
        let (_, row) = reads.last().unwrap();
        pages.extend(row.start / PAGE_BYTES..=(row.end - 1) / PAGE_BYTES);
    }
    assert!(
        pages.len() >= 2,
        "survivor rows span distinct storage pages"
    );
    expected.sort_by(|(a, ad), (b, bd)| b.total_cmp(a).then_with(|| ad.cmp(bd)));
    let collector = TopDocsByVectorSimilarity::new(field, query, DOCS + 1).with_adaptive_params(
        AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 4,
            ..Default::default()
        },
    );
    directory.reads.lock().unwrap().clear();
    let result = searcher.search(&filter, &collector)?;
    assert_eq!(result.results, expected);
    let reads = directory.reads.lock().unwrap();
    let scans: Vec<_> = reads
        .iter()
        .filter(|(stage, _)| matches!(stage, Stage::ExactScan))
        .map(|(_, range)| range)
        .collect();
    for docs in doc_ranges.iter().flatten() {
        assert_eq!(
            scans.iter().filter(|range| **range == docs).count(),
            1,
            "DocIds read once"
        );
    }
    let mut payload_pages = std::collections::BTreeSet::new();
    for range in scans {
        if !doc_ranges.iter().flatten().any(|docs| docs == range) {
            payload_pages.extend(range.start / PAGE_BYTES..=(range.end - 1) / PAGE_BYTES);
        }
    }
    assert_eq!(
        payload_pages, pages,
        "Rows reads touch only selected-row pages"
    );
    Ok(())
}

#[test]
fn diagnostic_schedules_ignore_rotation_seeds() -> crate::Result<()> {
    use crate::vector::{QuantizerKind, VectorEstimatorQuery, VectorEstimatorSource};

    let measure = |seed| -> crate::Result<_> {
        let (index, _) = fixture_with_seed(Metric::Cosine, &[1, 4], false, Some(seed))?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segment = &searcher.segment_readers()[0];
        let vectors = segment.vector_index(index.schema().get_field("embedding")?)?;
        let rotation = vectors.metadata().unwrap().layers()[0].rotation();
        let queries = [VectorEstimatorQuery {
            values: input_vector(DOCS),
            excluded_doc_id: None,
        }];
        let audit = vectors
            .audit_error_queries(VectorEstimatorSource::Provided, &queries, 30, None)?
            .unwrap();
        let estimator = vectors
            .measure_estimator_queries(VectorEstimatorSource::Provided, &queries, 30, None)?
            .unwrap();
        assert_eq!(audit.schedule(), estimator.schedule());
        Ok((rotation, audit, estimator))
    };
    let (left_rotation, mut left_audit, mut left_estimator) = measure(7)?;
    let (right_rotation, right_audit, right_estimator) = measure(11)?;
    assert_ne!(left_rotation, right_rotation);
    assert_eq!(left_audit.schedule(), right_audit.schedule());
    assert_eq!(left_estimator.schedule(), right_estimator.schedule());
    assert_eq!(
        left_audit.schedule().layers(),
        &[(QuantizerKind::Sign, 1), (QuantizerKind::Grid, 4)]
    );
    left_estimator.merge(&right_estimator)?;
    left_audit.merge(&right_audit)?;
    Ok(())
}

#[test]
fn diagnostics_skip_empty_clusters() -> crate::Result<()> {
    use crate::vector::index_reader::{VectorEstimatorQuery, VectorEstimatorSource};
    let (index, _) = fixture(Metric::Cosine, &[1, 4])?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let segment = &searcher.segment_readers()[0];
    let vectors = segment.vector_index(index.schema().get_field("embedding")?)?;
    assert!(vectors.index().unwrap().cluster_range(0).is_empty());
    let queries = vectors
        .sample_estimator_pseudo_queries(3, segment.alive_bitset())?
        .unwrap();
    assert_eq!(queries.len(), 3);
    let audit = vectors
        .audit_error_queries(
            VectorEstimatorSource::HeldOut,
            &queries,
            30,
            segment.alive_bitset(),
        )?
        .unwrap();
    assert_eq!(
        audit.schedule().layers(),
        &[(crate::vector::QuantizerKind::Sign, 1), (crate::vector::QuantizerKind::Grid, 4)]
    );
    assert_eq!(audit.estimator.sample_rows(), 30);
    let estimates = vectors
        .measure_estimator_queries(
            VectorEstimatorSource::HeldOut,
            &queries,
            30,
            segment.alive_bitset(),
        )?
        .unwrap();
    assert_eq!(estimates, audit.estimator);
    let external: Vec<_> = (0..100)
        .map(|doc| VectorEstimatorQuery {
            values: input_vector(doc + DOCS),
            excluded_doc_id: None,
        })
        .collect();
    let cone = vectors
        .audit_error_cone(&external, segment.alive_bitset())?
        .unwrap();
    assert_eq!(cone.query_count, 100);
    assert_eq!(cone.depths.len(), 2);
    Ok(())
}

#[test]
fn merge_source_addresses_read_only_document_columns() -> crate::Result<()> {
    use crate::indexer::doc_id_mapping::{MappingType, SegmentDocIdMapping};
    let (index, directory) = fixture(Metric::Cosine, &[1, 4])?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let segments = searcher.segment_readers();
    let segment = &segments[0];
    let vectors = segment.vector_index(index.schema().get_field("embedding")?)?;
    let ranges = doc_column_ranges(&vectors, &directory)?;
    let docs: Vec<_> = (0..segment.max_doc())
        .filter(|&doc| {
            segment
                .alive_bitset()
                .is_none_or(|alive| alive.is_alive(doc))
        })
        .map(|doc| crate::DocAddress::new(0, doc))
        .collect();
    let mapping =
        SegmentDocIdMapping::new(docs.clone(), MappingType::StackedWithDeletes, vec![None]);
    let target = index.new_segment();
    let schema = index.schema();
    let context = crate::plugin::PluginMergeContext {
        readers: segments,
        doc_id_mapping: &mapping,
        target_segment: &target,
        schema: &schema,
        settings: index.settings(),
        ignore_store: false,
        cancel: &|| false,
    };
    directory.reads.lock().unwrap().clear();
    let rows = crate::vector::plugin::merge_source_rows(&context, &[Arc::clone(&vectors)])?;
    let reads = directory.reads.lock().unwrap().clone();
    let expected: Vec<_> = ranges.into_iter().flatten().collect();
    assert_eq!(
        reads
            .iter()
            .map(|(_, range)| range.clone())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(rows.len(), docs.len());
    for (source, target) in rows.into_iter().zip(docs) {
        match source {
            Some((segment, row)) => {
                assert_eq!(segment, 0);
                assert_eq!(
                    vectors.doc_id_at(row)?,
                    target.doc_id,
                    "row={row} column=DocIds"
                );
            }
            None => assert!(!vectors.contains(target.doc_id)?),
        }
    }
    Ok(())
}
