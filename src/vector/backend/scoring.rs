use super::*;

#[derive(Default)]
struct QuantizedScratch {
    kernel_scores: Vec<f32>,
    decoded_scales: Vec<f32>,
    decoded_gammas: Vec<f32>,
    decoded_error_ratios: Vec<f32>,
    decoded_constants: Vec<f32>,
    decoded_residual_norms: Vec<f32>,
    base_scores: Vec<f32>,
    estimate_scores: Vec<f32>,
    sigma_scores: Vec<f32>,
    residual_norm_squared_scores: Vec<f32>,
    sign_query_error_terms: Vec<f32>,
    arithmetic_variances: Vec<ArithmeticError>,
    selected_rows: Vec<usize>,
    indexed_row_offsets: Vec<usize>,
    survivor_read_ranges: Vec<Range<usize>>,
    survivor_block_scratch: Vec<(usize, usize)>,
    selection_offsets: Vec<usize>,
    cluster_docs: Vec<DocId>,
}

pub(super) struct QuantizedScorer {
    pub(super) scan: QuantizedScanCtx,
    scratch: QuantizedScratch,
}

impl QuantizedScorer {
    pub(super) fn new(max_doc: DocId, capacity: usize) -> Self {
        Self {
            scan: QuantizedScanCtx::new(max_doc, capacity),
            scratch: QuantizedScratch::default(),
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn score_cluster(
        &mut self,
        reader: &VectorIndexReader,
        query: &QuantizedQueryCtx,
        cluster: usize,
        sim: Similarity,
        rows: Range<usize>,
        selection: &Selection<'_>,
        docs: Option<&[DocId]>,
    ) -> crate::Result<()> {
        let layer = &reader.quantization().expect("quantized columns").layers()[0];
        let metric = reader.options().metric();
        let selected_count = selection.len(&rows);
        let Self { scan, scratch } = self;
        let QuantizedScratch {
            kernel_scores,
            decoded_scales,
            decoded_gammas,
            decoded_error_ratios,
            decoded_constants,
            decoded_residual_norms,
            base_scores,
            estimate_scores,
            sigma_scores,
            residual_norm_squared_scores,
            sign_query_error_terms,
            arithmetic_variances,
            selected_rows,
            indexed_row_offsets,
            survivor_read_ranges,
            ..
        } = scratch;
        let batch = layer.read_batch_in_block(cluster, rows.clone())?;
        let score_query_norm = query.score_query_norm(sim.score());
        score_layer(
            query,
            0,
            layer,
            Some((&batch, decoded_residual_norms)),
            Some(cluster),
            rows.clone(),
            selection,
            kernel_scores,
            decoded_scales,
            decoded_gammas,
            decoded_error_ratios,
            decoded_constants,
            survivor_read_ranges,
            selected_rows,
            indexed_row_offsets,
        )?;
        base_scores.resize(selected_count, 0.0);
        estimate_scores.resize(selected_count, 0.0);
        sigma_scores.resize(selected_count, 0.0);
        residual_norm_squared_scores.resize(selected_count, 0.0);
        sign_query_error_terms.resize(selected_count, 0.0);
        arithmetic_variances.resize(selected_count, ArithmeticError::default());
        let cluster_score = sim.score();
        combine_initial_decoded(
            metric,
            query.index.meta.field().dim as usize,
            kernel_scores,
            base_scores,
            estimate_scores,
            sigma_scores,
            residual_norm_squared_scores,
            sign_query_error_terms,
            arithmetic_variances,
            decoded_scales,
            decoded_gammas,
            decoded_error_ratios,
            decoded_constants,
            decoded_residual_norms,
            cluster_score,
            score_query_norm * score_query_norm,
            query.query_error_squared(0) as f32,
        );

        scan.set_cluster_query_norm(cluster, score_query_norm);
        scan.candidates.append_selected(
            rows.clone(),
            selection,
            docs,
            &base_scores[..selected_count],
            &kernel_scores[..selected_count],
            &estimate_scores[..selected_count],
            &sigma_scores[..selected_count],
            &residual_norm_squared_scores[..selected_count],
            &decoded_gammas[..selected_count],
            &sign_query_error_terms[..selected_count],
            &arithmetic_variances[..selected_count],
        );
        Ok(())
    }
    pub(super) fn refine(
        &mut self,
        reader: &VectorIndexReader,
        query: &QuantizedQueryCtx,
        layer_idx: usize,
        stats: &mut ProbeStats,
    ) -> crate::Result<()> {
        let index = reader.index().expect("IVF index");
        let quantized = reader.quantization().expect("quantized columns");
        let metric = reader.options().metric();
        let Self { scan, scratch } = self;
        let QuantizedScratch {
            kernel_scores,
            decoded_scales,
            decoded_gammas,
            decoded_error_ratios,
            decoded_constants,
            selected_rows,
            indexed_row_offsets,
            survivor_read_ranges,
            selection_offsets,
            ..
        } = scratch;
        stats.start_layer(layer_idx);
        let layer_start = Instant::now();
        let layer_stage = enter_vector_stage(Stage::LayerScan(layer_idx as u8));
        let layer = &quantized.layers()[layer_idx];
        let layer_scored = scan.candidates.len();
        if metric == Metric::Cosine {
            let query_norm = query.score_query_norm(0.0);
            for candidate_range in cosine_refinement_batches(scan.candidates.len()) {
                let candidate_start = candidate_range.start;
                let candidate_end = candidate_range.end;
                let first_row = scan.candidates.rows[candidate_start];
                let last_row = scan.candidates.rows[candidate_end - 1];
                let available_rows = first_row..last_row + 1;
                let selection = candidate_selection(
                    &scan.candidates.rows[candidate_range.clone()],
                    &available_rows,
                    selection_offsets,
                );
                let rows = score_layer(
                    query,
                    layer_idx,
                    layer,
                    None,
                    None,
                    available_rows,
                    &selection,
                    kernel_scores,
                    decoded_scales,
                    decoded_gammas,
                    decoded_error_ratios,
                    decoded_constants,
                    survivor_read_ranges,
                    selected_rows,
                    indexed_row_offsets,
                )?;
                let decoded_constants = if metric == Metric::L2 {
                    &decoded_constants[..rows]
                } else {
                    &[]
                };
                let sign_query_error_squared =
                    if matches!(query.index.specs[layer_idx].kind, cascade::LayerKind::Sign) {
                        query.query_error_squared(layer_idx) as f32
                    } else {
                        0.0
                    };
                combine_refinement_decoded(
                    metric,
                    query.index.meta.field().dim as usize,
                    &mut scan.candidates,
                    candidate_range,
                    &kernel_scores[..rows],
                    &decoded_scales[..rows],
                    &decoded_gammas[..rows],
                    &decoded_error_ratios[..rows],
                    decoded_constants,
                    query_norm * query_norm,
                    sign_query_error_squared,
                );
            }
        } else {
            let mut candidate_start = 0;
            let mut cluster = 0;
            while candidate_start < scan.candidates.len() {
                let first_row = scan.candidates.rows[candidate_start];
                while cluster < index.num_clusters()
                    && index.cluster_range(cluster).end <= first_row
                {
                    cluster += 1;
                }
                if cluster == index.num_clusters() {
                    return Err(TantivyError::DataCorruption(DataCorruption::comment_only(
                        format!("quantized survivor row {first_row} is outside IVF cluster ranges"),
                    )));
                }
                let cluster_rows = index.cluster_range(cluster);
                if first_row < cluster_rows.start {
                    return Err(TantivyError::DataCorruption(DataCorruption::comment_only(
                        format!(
                            "quantized survivor row {first_row} precedes cluster {cluster} range \
                             {cluster_rows:?}"
                        ),
                    )));
                }
                let mut candidate_end = candidate_start + 1;
                while candidate_end < scan.candidates.len()
                    && scan.candidates.rows[candidate_end] < cluster_rows.end
                {
                    candidate_end += 1;
                }
                let query_norm = scan.cluster_query_norm(cluster);
                let candidate_range = candidate_start..candidate_end;
                let selection = candidate_selection(
                    &scan.candidates.rows[candidate_range.clone()],
                    &cluster_rows,
                    selection_offsets,
                );
                let rows = score_layer(
                    query,
                    layer_idx,
                    layer,
                    None,
                    Some(cluster),
                    cluster_rows,
                    &selection,
                    kernel_scores,
                    decoded_scales,
                    decoded_gammas,
                    decoded_error_ratios,
                    decoded_constants,
                    survivor_read_ranges,
                    selected_rows,
                    indexed_row_offsets,
                )?;
                let decoded_constants = if metric == Metric::L2 {
                    &decoded_constants[..rows]
                } else {
                    &[]
                };
                let sign_query_error_squared =
                    if matches!(query.index.specs[layer_idx].kind, cascade::LayerKind::Sign) {
                        query.query_error_squared(layer_idx) as f32
                    } else {
                        0.0
                    };
                combine_refinement_decoded(
                    metric,
                    query.index.meta.field().dim as usize,
                    &mut scan.candidates,
                    candidate_range,
                    &kernel_scores[..rows],
                    &decoded_scales[..rows],
                    &decoded_gammas[..rows],
                    &decoded_error_ratios[..rows],
                    decoded_constants,
                    query_norm * query_norm,
                    sign_query_error_squared,
                );
                candidate_start = candidate_end;
            }
        }
        drop(layer_stage);
        stats.record_layer_scan(
            layer_idx,
            layer_scored,
            layer_start.elapsed().as_nanos() as u64,
        );
        #[cfg(test)]
        stats
            .quantized_trace
            .estimate_rows
            .push(scan.candidates.estimate_trace());
        Ok(())
    }
    pub(super) fn rerank<T: VectorElement>(
        &mut self,
        reader: &VectorIndexReader,
        query: &PreparedQuery<T>,
        stats: &mut ProbeStats,
        mut visit: impl FnMut(DocId, Score),
    ) -> crate::Result<()> {
        let index = reader.index().expect("IVF index");
        let Self { scan, scratch } = self;
        let QuantizedScratch {
            cluster_docs,
            survivor_read_ranges,
            survivor_block_scratch,
            ..
        } = scratch;
        let rerank_fetch_start = Instant::now();
        let rerank_fetch_stage = enter_vector_stage(Stage::RerankFetch);
        debug_assert!(
            scan.candidates
                .rows
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "boundary survivors are row-sorted"
        );
        let mut first = 0;
        let mut cluster = 0;
        while first < scan.candidates.len() {
            while index.cluster_range(cluster).end <= scan.candidates.rows[first] {
                cluster += 1;
            }
            let cluster_rows = index.cluster_range(cluster);
            let end = first
                + scan.candidates.rows[first..].partition_point(|&row| row < cluster_rows.end);
            if scan.candidates.docs[first..end].contains(&DocId::MAX) {
                reader.read_doc_ids(cluster, cluster_docs)?;
                for candidate in first..end {
                    if scan.candidates.docs[candidate] == DocId::MAX {
                        scan.candidates.docs[candidate] =
                            cluster_docs[scan.candidates.rows[candidate] - cluster_rows.start];
                    }
                }
            }
            first = end;
        }
        #[cfg(test)]
        {
            stats.quantized_trace.rerank_docs = scan.candidates.docs.clone();
            stats.quantized_trace.rerank_docs.sort_unstable();
        }
        let rerank_batch = reader.read_vector_rows_planned(
            &scan.candidates.rows,
            survivor_read_ranges,
            survivor_block_scratch,
        )?;
        drop(rerank_fetch_stage);
        stats.rerank_fetch_ns += rerank_fetch_start.elapsed().as_nanos() as u64;
        stats.rerank_rows += scan.candidates.len();

        // Row-sorted survivors own their document ids; the batch ordinal addresses both.
        for (candidate, (batch_row, bytes)) in rerank_batch.iter().enumerate() {
            debug_assert_eq!(scan.candidates.rows[candidate], batch_row);
            let doc = scan.candidates.docs[candidate];
            let score_start = Instant::now();
            {
                let _rerank_score_stage = enter_vector_stage(Stage::RerankScore);
                visit(doc, query.score_doc_bytes(bytes));
            }
            stats.rerank_score_ns += score_start.elapsed().as_nanos() as u64;
            stats.exact_rows_read += 1;
        }
        Ok(())
    }
}
