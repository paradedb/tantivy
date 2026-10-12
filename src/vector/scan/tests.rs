// ============================================================
// IVF `top_n` test gate.
//
// Built on top of `crate::vector::tests::TestVectorIndex` (the
// shared fixture) where the geometry fits — the 100-doc grid +
// selectivity-based labels covers oracle / filter / delete /
// overflow / zero-K. The handful of tests that need crafted point
// geometry (the trap case + the result-level candidate-floor
// demonstration) build a tiny IVF index inline via `build_inline_ivf`
// and an `InlineClusterer` that's compatible with the batched
// IvfClusterer trait.
// ============================================================
use std::cmp::Ordering;
use std::sync::Arc;

use cascade::{encode_batch_in_place, prepare_centroid};
use common::BitSet;
use quant_model::f16::f16_to_f32;

use crate::collector::TopDocs;
use crate::index::IndexSettings;
use crate::indexer::NoMergePolicy;
use crate::query::{
    AllQuery, BitSetDocSet, BooleanQuery, ConstScorer, EnableScoring, Explanation, Occur, Query,
    Scorer, TermQuery, Weight,
};
use crate::schema::{Field, IndexRecordOption, Metric, Schema, Term, STORED, STRING};
use crate::vector::backend::VectorBackend;
use crate::vector::interval::{LowerEndpoint, Threshold};
use crate::vector::ivf::{AdaptiveProbeParams, RecallEstimator};
use crate::vector::prepared::{
    corrected_quantized_estimate, quantized_model_sigma, ArithmeticError, PreparedQuery,
    QuantizedIndexCtx, QuantizedQueryCtx, VectorQuery,
};
use crate::vector::quantization::QUANTIZED_BOUNDARY_KAPPA;
use crate::vector::router::RouterMetrics;
use crate::vector::scan::columns::QuantizedCandidates;
use crate::vector::scan::estimate::{
    combine_initial_decoded, combine_refinement_decoded, cosine_refinement_batches,
};
use crate::vector::scan::QuantizedScanCtx;
use crate::vector::segment::plan::{build_segment_filter, SegmentFilter};
use crate::vector::source::controller::{
    fixed_probe_cost_rows, open_share, ProbeController, ProbeTermination, UnitPricing, WorkUnits,
};
use crate::vector::source::routed::{select_cluster_rows, RowGate};
use crate::vector::stats::{ProbeStats, QuantizedStageTrace};
use crate::vector::tests::{exhaustive_params, TestVectorIndex};
use crate::vector::{
    IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingVectors, IvfVectors,
    NeighborhoodGraphSearchMetrics, RouterKind, SearchTerminationReason, VectorDType,
    VectorOptions, VectorQuantizationConfig, VectorQuantizationLayer, VectorStorageFormat,
};
use crate::{DocAddress, DocId, Index, IndexWriter, Score, SegmentReader, TantivyDocument};

const FIXTURE_NUM_DOCS: usize = 100;

/// Number of centroids the shared fixture uses by default (the
/// 3×3 `grid2d::centroids()` grid). Used by tests that need an
/// "exhaustive" probe ceiling.
const DEFAULT_NUM_CENTROIDS: usize = 9;

#[test]
fn selected_doc_ids_follow_scores_and_defer_open_gate() {
    let mut candidates = QuantizedCandidates::with_capacity(8);
    let docs_pointer = candidates.docs.as_ptr();
    let mut offsets = Vec::with_capacity(4);
    let mut filter = BitSet::with_max_value(32);
    filter.insert(4);
    filter.insert(12);
    for (rows, docs, gate) in [
        (
            100..104,
            vec![4u32, 9, 12, 22],
            RowGate::FilterOnly(&filter),
        ),
        (200..202, vec![11u32, 13], RowGate::Open),
    ] {
        let (selection, visited, filtered, dead) =
            select_cluster_rows(&docs, rows.clone(), &gate, &mut offsets);
        assert_eq!(visited, docs.len());
        assert_eq!(filtered, docs.len() - 2);
        assert_eq!(dead, 0);
        let scores = [0.25, 0.75];
        candidates.append_selected(
            rows,
            &selection,
            (!matches!(gate, RowGate::Open)).then_some(docs.as_slice()),
            &scores,
            &scores,
            &scores,
            &scores,
            &scores,
            &scores,
            &scores,
            &[ArithmeticError::default(); 2],
        );
    }
    assert_eq!(candidates.rows, [100, 102, 200, 201]);
    assert_eq!(candidates.docs, [4, 12, DocId::MAX, DocId::MAX]);
    let mut trace = QuantizedStageTrace {
        scored_rows: candidates.rows.clone(),
        ..Default::default()
    };
    let mut mapping = vec![0; 202];
    for (row, doc) in [(100, 4), (102, 12), (200, 11), (201, 13)] {
        mapping[row] = doc;
    }
    assert_eq!(
        candidates
            .rows
            .iter()
            .map(|&row| mapping[row])
            .collect::<Vec<_>>(),
        [4, 12, 11, 13]
    );
    trace.translate(&mapping);
    assert_eq!(trace.scored_docs, [4, 11, 12, 13]);
    assert_eq!(candidates.estimates, [0.25, 0.75, 0.25, 0.75]);
    assert_eq!(candidates.docs.as_ptr(), docs_pointer);
    assert_eq!(offsets.capacity(), 4);
}

fn assert_diagnostic_prefix_matches_integrated(metric: Metric) {
    const DIM: usize = 64;
    let config = VectorQuantizationConfig::materialize(
        "embedding".to_string(),
        &VectorOptions::new(DIM, metric),
        vec![
            VectorQuantizationLayer { bits: 1, seed: 11 },
            VectorQuantizationLayer { bits: 4, seed: 22 },
        ],
    )
    .unwrap();
    let index = Arc::new(QuantizedIndexCtx::from_config(config).unwrap());
    let query = QuantizedQueryCtx::new(
        Arc::clone(&index),
        (0..DIM)
            .map(|coordinate| ((coordinate as f32 + 0.25) * 0.173).sin())
            .collect(),
    );
    let centroid: Vec<f32> = (0..DIM)
        .map(|coordinate| ((coordinate as f32 + 0.5) * 0.071).cos() * 0.1)
        .collect();
    let mut vector: Vec<f32> = centroid
        .iter()
        .enumerate()
        .map(|(coordinate, &center)| center + ((coordinate as f32 + 0.75) * 0.113).sin() * 0.2)
        .collect();
    let prepared_centroid = prepare_centroid(&centroid, &index.specs);
    let encoded = encode_batch_in_place(
        &mut vector,
        1,
        &prepared_centroid,
        &index.specs,
        &index.grids,
    );
    let residual_norm_squared = encoded.residual_norms_squared[0];
    let cluster_score = if metric == Metric::L2 { -1.75 } else { 0.25 };
    let score_query_norm = query.score_query_norm(cluster_score);
    let score_query_norm_squared = score_query_norm * score_query_norm;
    let base = if metric == Metric::L2 {
        cluster_score - residual_norm_squared
    } else {
        cluster_score
    };

    let first = &encoded.layers[0];
    let first_scale = first.scales[0];
    let first_gamma = f16_to_f32(first.gammas[0]);
    let first_error_ratio = f16_to_f32(first.corrected_error_ratios[0]);
    let first_constant = (metric == Metric::L2).then_some(first.constants[0]);
    let mut diagnostic_arithmetic = ArithmeticError::default();
    let diagnostic_first_raw = super::super::index_reader::diagnostic_advance_raw_prefix(
        &query,
        metric,
        0,
        &first.codes,
        first_scale,
        first_constant,
        0.0,
        cluster_score,
        residual_norm_squared,
        &mut diagnostic_arithmetic,
    )
    .unwrap();
    let diagnostic_first_estimate =
        corrected_quantized_estimate(metric, first_gamma, diagnostic_first_raw, base);
    let diagnostic_sign_query_error_term =
        first_scale * first_scale * query.query_error_squared(0) as f32;
    let diagnostic_first_sigma = quantized_model_sigma(
        metric,
        DIM,
        residual_norm_squared,
        first_error_ratio,
        first_gamma,
        score_query_norm_squared,
        diagnostic_sign_query_error_term,
    );
    let diagnostic_first_sigma = diagnostic_arithmetic.sigma(
        metric,
        diagnostic_first_sigma,
        first_gamma,
        diagnostic_first_raw,
        base,
    );

    let mut first_kernel = [0.0];
    query.score_layer_batch_unscaled(0, &first.codes, first.codes.len(), &mut first_kernel);
    let mut bases = [0.0];
    let mut estimates = [0.0];
    let mut sigmas = [0.0];
    let mut residual_norms = [0.0];
    let mut sign_query_error_terms = [0.0];
    let mut arithmetic = [ArithmeticError::default()];
    let first_constants = first_constant.map_or_else(Vec::new, |constant| vec![constant]);
    combine_initial_decoded(
        metric,
        DIM,
        &mut first_kernel,
        &mut bases,
        &mut estimates,
        &mut sigmas,
        &mut residual_norms,
        &mut sign_query_error_terms,
        &mut arithmetic,
        &[first_scale],
        &[first_gamma],
        &[first_error_ratio],
        &first_constants,
        &[residual_norm_squared],
        cluster_score,
        score_query_norm_squared,
        query.query_error_squared(0) as f32,
    );
    assert_eq!(diagnostic_first_raw.to_bits(), first_kernel[0].to_bits());
    assert_eq!(diagnostic_first_estimate.to_bits(), estimates[0].to_bits());
    assert_eq!(diagnostic_first_sigma.to_bits(), sigmas[0].to_bits());

    let mut candidates = QuantizedCandidates::with_capacity(1);
    candidates.push(
        0,
        0,
        bases[0],
        first_kernel[0],
        estimates[0],
        sigmas[0],
        residual_norms[0],
        first_gamma,
        sign_query_error_terms[0],
        arithmetic[0],
    );
    let refinement = &encoded.layers[1];
    let refinement_scale = refinement.scales[0];
    let refinement_gamma = f16_to_f32(refinement.gammas[0]);
    let refinement_error_ratio = f16_to_f32(refinement.corrected_error_ratios[0]);
    let refinement_constant = (metric == Metric::L2).then_some(refinement.constants[0]);
    let diagnostic_refined_raw = super::super::index_reader::diagnostic_advance_raw_prefix(
        &query,
        metric,
        1,
        &refinement.codes,
        refinement_scale,
        refinement_constant,
        diagnostic_first_raw,
        cluster_score,
        residual_norm_squared,
        &mut diagnostic_arithmetic,
    )
    .unwrap();
    let diagnostic_refined_estimate =
        corrected_quantized_estimate(metric, refinement_gamma, diagnostic_refined_raw, base);
    let diagnostic_refined_sigma = quantized_model_sigma(
        metric,
        DIM,
        residual_norm_squared,
        refinement_error_ratio,
        refinement_gamma,
        score_query_norm_squared,
        diagnostic_sign_query_error_term,
    );
    let diagnostic_refined_sigma = diagnostic_arithmetic.sigma(
        metric,
        diagnostic_refined_sigma,
        refinement_gamma,
        diagnostic_refined_raw,
        base,
    );

    let mut refinement_kernel = [0.0];
    query.score_layer_batch_unscaled(
        1,
        &refinement.codes,
        refinement.codes.len(),
        &mut refinement_kernel,
    );
    let refinement_constants = refinement_constant.map_or_else(Vec::new, |constant| vec![constant]);
    combine_refinement_decoded(
        metric,
        DIM,
        &mut candidates,
        0..1,
        &refinement_kernel,
        &[refinement_scale],
        &[refinement_gamma],
        &[refinement_error_ratio],
        &refinement_constants,
        score_query_norm_squared,
        query.query_error_squared(1) as f32,
    );
    assert_eq!(
        diagnostic_refined_raw.to_bits(),
        candidates.raw_prefixes[0].to_bits()
    );
    assert_eq!(
        diagnostic_refined_estimate.to_bits(),
        candidates.estimates[0].to_bits()
    );
    assert_eq!(
        diagnostic_refined_sigma.to_bits(),
        candidates.sigmas[0].to_bits()
    );
}

#[test]
fn diagnostic_prefixes_match_integrated_sign_grid_combines() {
    assert_diagnostic_prefix_matches_integrated(Metric::Cosine);
    assert_diagnostic_prefix_matches_integrated(Metric::L2);
}

fn push_test_candidate(
    scan: &mut QuantizedScanCtx,
    row: usize,
    doc: DocId,
    estimate: f32,
    sigma: f32,
) {
    scan.push(row, doc, 0.0, estimate, estimate, sigma, 1.0, 1.0, 0.0);
}

/// Run the full collector path with the given filter and adaptive
/// params. Returns the global top-K (already merged across
/// segments) in descending-score / (seg_ord, doc_id) order — the
/// same order `ground_truth::top_k` uses, so equality checks are
/// well-defined.
pub(in crate::vector) fn search(
    index: &Index,
    field: Field,
    filter: &dyn Query,
    query: Vec<f32>,
    k: usize,
    params: AdaptiveProbeParams,
) -> crate::Result<Vec<(Score, DocAddress)>> {
    let collector = TopDocs::with_limit(k)
        .order_by_similarity(field, query)
        .with_adaptive_params(params);
    Ok(index
        .reader()?
        .searcher()
        .search(filter, &collector)?
        .results)
}

/// Probe-stat helper: run `VectorBackend::top_n` against
/// the first segment of `index` and return (hits, stats).
/// The contracts are per-segment, so collecting from segment 0 is
/// what each assertion is talking about.
fn run_top_n(
    index: &Index,
    embed_field: Field,
    query: Vec<f32>,
    k: usize,
    params: AdaptiveProbeParams,
) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
    let searcher = index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    let weight = AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))?;
    let backend = VectorBackend::<f32>::for_segment(
        segment_reader,
        0,
        embed_field,
        VectorQuery::new(Arc::new(query), None),
        params,
    )?;
    assert!(
        segment_reader.vector_index(embed_field)?.index().is_some(),
        "expected IVF storage"
    );
    backend.top_n(weight.as_ref(), segment_reader, k)
}

// ---- Inline IVF builder for crafted-geometry tests ----
//
// The shared fixture's `grid2d::vectors` lays 100 deterministic
// points around a 3×3 grid; it doesn't expose a per-doc-vector
// override. The trap-case and result-level candidate-floor tests
// need points at specific coordinates, so they build a small IVF
// index inline via the helper below.
pub(in crate::vector) struct InlineClusterer {
    pub(in crate::vector) centroids: Vec<[f32; 2]>,
}

impl IvfClusterer for InlineClusterer {
    fn training_sample_ratio(&self) -> f32 {
        1.0
    }
    fn train(
        &self,
        options: &VectorOptions,
        _vectors: IvfTrainingVectors,
    ) -> crate::Result<IvfCentroids> {
        assert_eq!(options.dim(), 2);
        let num_centroids = self.centroids.len();
        Ok(IvfCentroids::F32(IvfMatrix {
            values: self
                .centroids
                .iter()
                .flat_map(|c| c.iter().copied())
                .collect(),
            rows: num_centroids,
            dims: 2,
        }))
    }
    fn assign(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        centroids: &IvfCentroids,
    ) -> crate::Result<Vec<u32>> {
        assert_eq!(options.dim(), 2);
        let IvfVectors::F32(vectors) = vectors;
        let IvfCentroids::F32(centroids) = centroids;
        Ok(vectors
            .matrix
            .values
            .chunks_exact(2)
            .map(|v| {
                let mut best = 0u32;
                let mut best_d2 = f32::INFINITY;
                for (i, c) in centroids.values.chunks_exact(2).enumerate() {
                    let dx = v[0] - c[0];
                    let dy = v[1] - c[1];
                    let d2 = dx * dx + dy * dy;
                    if d2 < best_d2 {
                        best = i as u32;
                        best_d2 = d2;
                    }
                }
                best
            })
            .collect())
    }
}

/// Build a single-IVF-segment index with the supplied centroids and
/// labelled docs. Splits docs across two commits so `merge_ivf`
/// has ≥ 2 source segments to consume. Returns the index plus the
/// `(embedding, label)` field handles.
pub(in crate::vector) fn build_inline_ivf(
    metric: Metric,
    centroids: &[[f32; 2]],
    docs: &[(&str, [f32; 2])],
) -> crate::Result<(Index, Field, Field)> {
    assert!(docs.len() >= 2, "need ≥ 2 docs for ≥ 2 source segments");
    let mut sb = Schema::builder();
    let embed_field = sb.add_vector_field(
        "embedding",
        VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
    );
    let label_field = sb.add_text_field("label", STRING | STORED);
    let schema = sb.build();

    let settings = IndexSettings {
        vector_clustering_threshold: 1,
        ..IndexSettings::default()
    };
    let index = Index::builder()
        .schema(schema)
        .settings(settings)
        .ivf_clusterer(Arc::new(InlineClusterer {
            centroids: centroids.to_vec(),
        }))
        .ivf_router(RouterKind::Stacked)?
        .create_in_ram()?;
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));

    let mid = docs.len() / 2;
    for chunk in [&docs[..mid.max(1)], &docs[mid.max(1)..]] {
        for (label, v) in chunk {
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, label);
            doc.add_vector(embed_field, v.as_slice());
            writer.add_document(doc)?;
        }
        writer.commit()?;
    }
    let segment_ids: Vec<_> = index.searchable_segment_ids()?.into_iter().collect();
    writer.merge(&segment_ids).wait()?;
    writer.wait_merging_threads()?;
    Ok((index, embed_field, label_field))
}

/// Decode a stored little-endian `[f32; 2]` row.
fn decode_2d(bytes: &[u8]) -> [f32; 2] {
    [
        f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
    ]
}

/// L2-nearest centroid with first-wins tie-break on strict `<` — the
/// same rule `InlineClusterer::assign` uses for the primary.
fn nearest_centroid(p: [f32; 2], centroids: &[[f32; 2]]) -> usize {
    let mut best = 0;
    let mut best_d2 = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let dx = p[0] - c[0];
        let dy = p[1] - c[1];
        let d2 = dx * dx + dy * dy;
        if d2 < best_d2 {
            best_d2 = d2;
            best = i;
        }
    }
    best
}

const DOCS_PER_CLUSTER: usize = 6;

pub(in crate::vector) fn multi_cluster_fixture() -> (Vec<[f32; 2]>, Vec<String>) {
    let centroids = vec![
        [0.0f32, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
        [0.0, 10.0],
        [10.0, 10.0],
        [20.0, 10.0],
    ];
    let labels = (0..centroids.len() * DOCS_PER_CLUSTER)
        .map(|i| format!("d{i}"))
        .collect();
    (centroids, labels)
}

pub(in crate::vector) fn multi_cluster_docs<'a>(
    centroids: &[[f32; 2]],
    labels: &'a [String],
) -> Vec<(&'a str, [f32; 2])> {
    (0..labels.len())
        .map(|i| {
            let c = centroids[i / DOCS_PER_CLUSTER];
            let off = (i % DOCS_PER_CLUSTER) as f32 * 0.01;
            (labels[i].as_str(), [c[0] + off, c[1] + off])
        })
        .collect()
}

/// Exhaustive probing on a multi-segment IVF index built by the
/// shared fixture must match the brute-force oracle. Sweep over
/// several queries and K values to cover ranking + drain edges.
#[test]
fn ivf_top_n_brute_force_oracle_l2() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .build()?;
    let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
    for query in [[0.5_f32, 0.5], [9.5, 9.5], [5.0, 0.0], [3.7, 11.2]] {
        for k in [1usize, 3, 6, 10] {
            let expected = index.ground_truth(query, k)?;
            let actual = search(
                &index.index,
                index.embedding_field(),
                &AllQuery,
                query.to_vec(),
                k,
                params.clone(),
            )?;
            assert_eq!(actual, expected, "L2 exhaustive query={query:?} k={k}");
        }
    }
    Ok(())
}

/// Same exhaustive correctness, confirming the metric threads
/// through generically.
#[test]
fn ivf_top_n_brute_force_oracle_cosine() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::Cosine)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .build()?;
    let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
    for query in [[1.0_f32, 0.0], [0.0, 1.0], [0.7, 0.3]] {
        for k in [1usize, 3, 6] {
            let expected = index.ground_truth(query, k)?;
            let actual = search(
                &index.index,
                index.embedding_field(),
                &AllQuery,
                query.to_vec(),
                k,
                params.clone(),
            )?;
            assert_eq!(actual, expected, "Cosine exhaustive query={query:?} k={k}");
        }
    }
    Ok(())
}

/// Exhaustive-probe correctness for Dot. EXHAUSTIVE-PROBE ONLY by
/// design: Dot isn't a metric (no triangle inequality), so the IVF
/// cluster-locality assumption is heuristic for unnormalized dot
/// and can break on high-magnitude vectors in a far cluster.
/// Adaptive Dot recall is a benchmark question, deferred. This
/// test confirms only that `Metric::Dot` threads through the
/// backend's full top_n loop and matches brute force when every
/// cluster is visited.
#[test]
fn ivf_top_n_brute_force_oracle_dot() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::Dot)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .build()?;
    let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
    for query in [[1.0_f32, 0.0], [2.0, 0.0], [0.5, -0.5]] {
        for k in [1usize, 3, 6] {
            let expected = index.ground_truth(query, k)?;
            let actual = search(
                &index.index,
                index.embedding_field(),
                &AllQuery,
                query.to_vec(),
                k,
                params.clone(),
            )?;
            assert_eq!(actual, expected, "Dot exhaustive query={query:?} k={k}");
        }
    }
    Ok(())
}

/// The trap: query closest to centroid A, true NN in cluster B.
/// Adaptive probing finds it; a 1-cluster probe ceiling must miss. Setup
/// assertions confirm the geometry is genuinely a trap before
/// the behavioral check — a slightly-off geometry could trivialize
/// the test. INLINE because the shared fixture's 100-doc grid
/// doesn't permit a single misplaced trap doc.
#[test]
fn ivf_top_n_trap_case() -> crate::Result<()> {
    let centroids = vec![[0.0_f32, 0.0], [10.0, 10.0]];
    // Two A-side docs far from the [1,1] query; a B-side trap
    // doc at [5, 5.01] just over the perpendicular bisector
    // (x+y=10) so it lands in cluster 1 yet is much closer to
    // the query than any A-side doc.
    let docs = [
        ("far_a", [0.0_f32, -10.0]),
        ("far_a", [-10.0, 0.0]),
        ("trap_b", [5.0, 5.01]),
        ("anchor_b", [10.0, 10.0]),
    ];
    let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
    let query = [1.0_f32, 1.0];

    // Setup assertions.
    //
    // (i) The trap doc is genuinely the true top-1 — without
    // this, "miss" and "find" would be indistinguishable.
    let oracle = ground_truth_top_k(&index, embed_field, Metric::L2, &query, 1)?;
    let trap_doc = stored_label_at(&index, label_field, oracle[0].1)?;
    assert_eq!(trap_doc, "trap_b", "true NN must be the trap doc");

    // (ii) Query's nearest centroid is A (the one at the origin).
    //
    // With the inline IVF building exactly one segment, segment 0
    // holds both centroids. We don't need to open the column
    // directly — the geometry says distance to A = √2 ≈ 1.41,
    // distance to B = √162 ≈ 12.73, so A wins decisively.

    // Behavioral check 1: a probe ceiling of 1 misses the trap.
    let one_probe = AdaptiveProbeParams {
        max_probe_fraction: 0.5,
        min_probe_clusters: 1,
        ..Default::default()
    };
    let hits1 = search(&index, embed_field, &AllQuery, query.to_vec(), 1, one_probe)?;
    assert_eq!(hits1.len(), 1);
    assert_ne!(
        stored_label_at(&index, label_field, hits1[0].1)?,
        "trap_b",
        "a 1-cluster probe ceiling should miss the trap (probes only cluster A)",
    );

    // Behavioral check 2: exhaustive probing finds it.
    let hits2 = search(
        &index,
        embed_field,
        &AllQuery,
        query.to_vec(),
        1,
        exhaustive_params(2),
    )?;
    assert_eq!(hits2.len(), 1);
    assert_eq!(
        stored_label_at(&index, label_field, hits2[0].1)?,
        "trap_b",
        "exhaustive probing should find the trap doc",
    );
    Ok(())
}

/// Filter selectivity: only docs in the filter set surface, and
/// the result equals the oracle restricted to that set. Uses the
/// shared fixture's `.selectivities(..)` to drop a "selectivity_0.1"
/// label on the first 10 of 100 docs.
#[test]
fn ivf_top_n_filter_selectivity() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .selectivities(&[0.1])
        .build()?;
    let filter = TermQuery::new(
        Term::from_field_text(index.label_field(), "selectivity_0.1"),
        IndexRecordOption::Basic,
    );
    let query = [0.5_f32, 0.5];
    let k = 5;
    // Oracle restricted to the filter set: brute force across the
    // whole index, then keep only the docs that carry the label.
    let filter_set = collect_filter_doc_set(&index.index, &filter)?;
    let mut restricted = ground_truth_top_k(
        &index.index,
        index.embedding_field(),
        Metric::L2,
        &query,
        FIXTURE_NUM_DOCS,
    )?;
    restricted.retain(|(_, addr)| filter_set.contains(addr));
    restricted.truncate(k);

    let actual = search(
        &index.index,
        index.embedding_field(),
        &filter,
        query.to_vec(),
        k,
        exhaustive_params(DEFAULT_NUM_CENTROIDS),
    )?;
    assert_eq!(actual, restricted);
    for (_, addr) in &actual {
        assert!(filter_set.contains(addr), "hit outside filter: {addr:?}");
    }
    Ok(())
}

/// Empty filter returns empty results, no panic.
#[test]
fn ivf_top_n_empty_filter() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .build()?;
    // No doc carries "absent" — the term query yields an empty
    // DocSet.
    let empty = TermQuery::new(
        Term::from_field_text(index.label_field(), "absent"),
        IndexRecordOption::Basic,
    );
    let hits = search(
        &index.index,
        index.embedding_field(),
        &empty,
        vec![0.0_f32, 0.0],
        5,
        exhaustive_params(DEFAULT_NUM_CENTROIDS),
    )?;
    assert!(hits.is_empty());
    Ok(())
}

/// K > total candidates: returns all docs in descending order,
/// no panic.
#[test]
fn ivf_top_n_k_exceeds_candidates() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .build()?;
    let query = [0.0_f32, 0.0];
    let big_k = FIXTURE_NUM_DOCS + 50;
    let expected = index.ground_truth(query, big_k)?;
    let actual = search(
        &index.index,
        index.embedding_field(),
        &AllQuery,
        query.to_vec(),
        big_k,
        exhaustive_params(DEFAULT_NUM_CENTROIDS),
    )?;
    assert_eq!(actual.len(), FIXTURE_NUM_DOCS);
    assert_eq!(actual, expected);
    Ok(())
}

/// Match-all filters, including a boolean that collapses to one, skip
/// the bitset drain; selective and empty filters still materialize.
#[test]
fn segment_filter_skips_bitset_for_match_all() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .selectivities(&[0.1])
        .build()?;
    let searcher = index.index.reader()?.searcher();
    let label = |text: &str| -> Box<dyn Query> {
        Box::new(TermQuery::new(
            Term::from_field_text(index.label_field(), text),
            IndexRecordOption::Basic,
        ))
    };
    let all_and_all = BooleanQuery::new(vec![
        (Occur::Must, Box::new(AllQuery) as Box<dyn Query>),
        (Occur::Should, Box::new(AllQuery)),
    ]);
    let all_or_label = BooleanQuery::new(vec![
        (Occur::Should, Box::new(AllQuery) as Box<dyn Query>),
        (Occur::Should, label("selectivity_0.1")),
    ]);
    let selective = label("selectivity_0.1");
    let missing = label("no_such_label");

    let mut saw_partial = false;
    for segment_reader in searcher.segment_readers() {
        let max_doc = segment_reader.max_doc();
        let weight_for =
            |query: &dyn Query| query.weight(EnableScoring::disabled_from_searcher(&searcher));
        let filter_for = |query: &dyn Query| -> crate::Result<SegmentFilter> {
            build_segment_filter(weight_for(query)?.as_ref(), segment_reader, max_doc)
        };

        assert!(matches!(filter_for(&AllQuery)?, SegmentFilter::All));
        assert!(matches!(filter_for(&all_and_all)?, SegmentFilter::All));
        assert!(matches!(filter_for(&all_or_label)?, SegmentFilter::All));
        assert!(filter_for(missing.as_ref())?.is_empty());

        let matching = weight_for(selective.as_ref())?.count(segment_reader)? as usize;
        match filter_for(selective.as_ref())? {
            SegmentFilter::All => assert_eq!(matching, max_doc as usize),
            SegmentFilter::Docs(docs) => {
                assert_eq!(docs.len(), matching);
                saw_partial |= matching > 0;
            }
        }
    }
    assert!(saw_partial, "fixture must exercise a partial filter");
    Ok(())
}

/// Deletes: a doc marked deleted must never appear, even if it
/// would otherwise rank top-K. Confirms the IVF backend's separate
/// alive-check (the filter bitmap doesn't carry delete info).
#[test]
fn ivf_top_n_respects_deletes() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .selectivities(&[0.1])
        .build()?;
    // Delete every doc carrying the 0.1-selectivity label — the
    // 10 docs nearest to the grid's origin centroid by
    // construction (they're inserted first).
    {
        let mut writer: IndexWriter = index.index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        writer.delete_term(Term::from_field_text(
            index.label_field(),
            "selectivity_0.1",
        ));
        writer.commit()?;
    }

    // Oracle restricted to the surviving docs.
    let query = [0.0_f32, 0.0];
    let searcher = index.index.reader()?.searcher();
    let mut alive_addrs = std::collections::HashSet::new();
    for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
        let alive = segment_reader.alive_bitset();
        for doc in 0..segment_reader.max_doc() {
            let is_alive = alive.is_none_or(|bs| bs.is_alive(doc));
            if is_alive {
                alive_addrs.insert(DocAddress::new(seg_ord as u32, doc));
            }
        }
    }
    assert!(
        alive_addrs.len() < FIXTURE_NUM_DOCS,
        "delete didn't remove anything (alive={})",
        alive_addrs.len(),
    );
    let k = 10;
    let mut expected = ground_truth_top_k(
        &index.index,
        index.embedding_field(),
        Metric::L2,
        &query,
        FIXTURE_NUM_DOCS,
    )?;
    expected.retain(|(_, addr)| alive_addrs.contains(addr));
    expected.truncate(k);

    let actual = search(
        &index.index,
        index.embedding_field(),
        &AllQuery,
        query.to_vec(),
        k,
        exhaustive_params(DEFAULT_NUM_CENTROIDS),
    )?;
    assert_eq!(actual, expected);
    for (_, addr) in &actual {
        assert!(
            alive_addrs.contains(addr),
            "deleted doc {addr:?} surfaced in results",
        );
    }
    Ok(())
}

/// `top_n == 0` returns empty without touching the column. The
/// collector layer rejects `TopDocs::with_limit(0)` before it
/// reaches the backend, so this test calls the backend directly
/// via the instrumented seam — the short-circuit lives in
/// `approximate_top_n`.
#[test]
fn ivf_top_n_zero_returns_empty() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .build()?;
    let (hits, stats) = run_top_n(
        &index.index,
        index.embedding_field(),
        vec![0.0_f32, 0.0],
        0,
        AdaptiveProbeParams::default(),
    )?;
    assert!(hits.is_empty());
    // Short-circuit fires before the probe loop, so no clusters
    // visited and no candidates scored.
    assert_eq!(stats.clusters_probed(), 0);
    assert_eq!(stats.candidates_scored, 0);
    Ok(())
}

/// Smoke for the instrumented seam: every centroid is probed under
/// exhaustive params, and candidates_scored ≤ total docs in the
/// inspected segment. Exhaustive params on a 9-centroid segment
/// visit all 9.
#[test]
fn ivf_top_n_collects_probe_stats() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .metric(Metric::L2)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .build()?;
    // k = 64 exceeds any segment's doc count, so the query bound
    // never arms and the bounds gate skips nothing - every cluster
    // is probed and every counter equality below is exact.
    let (_, stats) = run_top_n(
        &index.index,
        index.embedding_field(),
        vec![0.0_f32, 0.0],
        64,
        exhaustive_params(DEFAULT_NUM_CENTROIDS),
    )?;
    assert_eq!(stats.clusters_probed(), DEFAULT_NUM_CENTROIDS);
    // The first segment has docs distributed across all 9 clusters;
    // candidates_scored equals the segment's doc count under
    // exhaustive probe + AllQuery.
    let segment_doc_count =
        index.index.reader()?.searcher().segment_readers()[0].max_doc() as usize;
    assert_eq!(stats.candidates_scored, segment_doc_count);

    // Counter invariant: every touched doc lands in exactly one bucket.
    assert_eq!(
        stats.vectors_visited,
        stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
        "visited must equal filter+dead+scored ({stats:?})"
    );
    // Exhaustive params (unclamped ceiling, unsatisfiable floor)
    // drain the ranked list.
    assert_eq!(stats.termination, ProbeTermination::Exhausted);
    Ok(())
}

/// A `max_probe_fraction` resolving below the cluster count forces the
/// hard ceiling: the loop stops with `termination == Ceiling`, having
/// probed exactly the cap, and the counter invariant still holds. Uses the
/// deterministic `build_inline_ivf` fixture (fixed 6 centroids) so the
/// cutoff is stable.
#[test]
fn ivf_probe_stats_termination_ceiling() -> crate::Result<()> {
    let centroids = [
        [0.0f32, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
        [0.0, 10.0],
        [10.0, 10.0],
        [20.0, 10.0],
    ];
    let n_per = 6usize;
    let labels: Vec<String> = (0..centroids.len() * n_per)
        .map(|i| format!("d{i}"))
        .collect();
    let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
        .map(|i| {
            let c = centroids[i / n_per];
            let off = (i % n_per) as f32 * 0.01;
            (labels[i].as_str(), [c[0] + off, c[1] + off])
        })
        .collect();
    let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

    // Cap 1 → ceiling at the first probe; an unsatisfiable survivor
    // floor keeps the gate from firing first.
    let params = AdaptiveProbeParams {
        max_probe_fraction: 0.1,
        min_probe_clusters: 1,
        ..Default::default()
    };
    let (_, stats) = run_top_n(&index, embed_field, vec![10.0, 10.0], 3, params)?;
    assert_eq!(stats.termination, ProbeTermination::Ceiling);
    // Stopped at exactly the cap.
    assert_eq!(stats.clusters_probed(), 1);
    assert_eq!(
        stats.vectors_visited,
        stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
        "visited must equal filter+dead+scored ({stats:?})"
    );
    Ok(())
}

/// A single-centroid IVF merge skips the `.centroids` graph slot
/// (nothing to route between), so the reader must take the linear
/// fallback: rank the lone cluster without a graph and still return the
/// exact top-K.
#[test]
fn ivf_single_centroid_routes_without_graph() -> crate::Result<()> {
    let centroids = [[0.0f32, 0.0]];
    let labels: Vec<String> = (0..5).map(|i| format!("d{i}")).collect();
    let docs: Vec<(&str, [f32; 2])> = (0..5)
        .map(|i| (labels[i].as_str(), [i as f32 * 0.01, 0.0]))
        .collect();
    let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

    let searcher = index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    assert!(
        segment_reader.vector_index(embed_field)?.index().is_some(),
        "expected IVF storage"
    );

    let query = [0.0f32, 0.0];
    let k = 3;
    let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, k)?;
    let (hits, stats) = run_top_n(
        &index,
        embed_field,
        query.to_vec(),
        k,
        AdaptiveProbeParams::default(),
    )?;
    assert_eq!(hits, expected, "linear fallback must match the oracle");
    assert_eq!(stats.clusters_probed(), 1, "one cluster, one probe");
    Ok(())
}

/// When the probe ceiling is below the cluster count, cluster ranking
/// routes via the persisted RNG instead of scanning every centroid. With
/// 16 well-separated clusters and only 8 beam seeds, the router must
/// still navigate to the true nearest cluster: the routed top-K equals
/// the brute-force oracle, and the recorded navigation cost is the
/// beam-visited count, not a full scan of the ranked list.
#[test]
fn ivf_routed_ranking_matches_oracle_on_separated_clusters() -> crate::Result<()> {
    // 4×4 grid of well-separated centroids (spacing 10), 4 docs each,
    // tightly packed around their centroid so each query's true top-K
    // lives entirely in one cluster.
    let side = 4usize;
    let centroids: Vec<[f32; 2]> = (0..side * side)
        .map(|i| [(i % side) as f32 * 10.0, (i / side) as f32 * 10.0])
        .collect();
    let n_per = 4usize;
    let labels: Vec<String> = (0..centroids.len() * n_per)
        .map(|i| format!("d{i}"))
        .collect();
    let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
        .map(|i| {
            let c = centroids[i / n_per];
            let off = (i % n_per) as f32 * 0.01;
            (labels[i].as_str(), [c[0] + off, c[1] + off])
        })
        .collect();
    let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

    // The merged segment must carry the routing graph, and cap 2 (< 16
    // clusters) must engage it.
    let searcher = index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    let vec_reader = segment_reader.vector_index(embed_field)?;
    let ivf = vec_reader.index().expect("expected IVF segment");
    assert_eq!(ivf.num_clusters(), centroids.len());

    let params = AdaptiveProbeParams {
        max_probe_fraction: 0.1,
        min_probe_clusters: 1,
        ..Default::default()
    };
    let k = 3usize;
    for (ord, centroid) in centroids.iter().enumerate().step_by(3) {
        let query = [centroid[0] + 0.3, centroid[1] - 0.2];
        let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, k)?;
        let (hits, stats) = run_top_n(&index, embed_field, query.to_vec(), k, params.clone())?;
        assert_eq!(hits, expected, "routed top-{k} near centroid {ord}");
        assert!(
            stats.clusters_probed() <= 2,
            "cap 2 must bound the probes, got {}",
            stats.clusters_probed()
        );
    }
    Ok(())
}

/// Full-budget params with the gate parked - the shared configuration
/// for the budget properties.
fn budget_only_params() -> AdaptiveProbeParams {
    AdaptiveProbeParams {
        max_probe_fraction: 1.0,
        min_probe_clusters: 1,
        ..Default::default()
    }
}

/// An exhaustive scan charges `C*x + (1 - x)*N/n_avg = exactly C`
/// units - capacity is the cluster count, whatever the size skew.
#[test]
fn unit_normalization_exact() -> crate::Result<()> {
    // Uneven sizes on purpose: [5, 2, 2, 1] docs across 4 clusters.
    let centroids = vec![[0.0_f32, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]];
    let mut docs: Vec<(String, [f32; 2])> = Vec::new();
    for (c, count) in [(0usize, 5usize), (1, 2), (2, 2), (3, 1)] {
        for i in 0..count {
            docs.push((
                format!("d{c}_{i}"),
                [centroids[c][0] + i as f32 * 0.01, 0.0],
            ));
        }
    }
    let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
    let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
    let (_, stats) = run_top_n(
        &index,
        embed_field,
        vec![0.0, 0.0],
        11,
        budget_only_params(),
    )?;
    let c = centroids.len() as f32;
    assert_eq!(stats.termination, ProbeTermination::Exhausted);
    assert!(
        (stats.work_charged - c).abs() <= 1e-6 * c,
        "an exhaustive scan must charge exactly C units: {stats:?}"
    );
    Ok(())
}

/// Skew charges proportionally: a 30-doc cluster consumes most of the
/// budget a cluster-count budget would spread over five clusters, and
/// the stop point shifts - hand-verified numbers. Also pins the
/// overshoot bound: the final overrun is at most the last cluster's
/// charge.
#[test]
fn imbalance_charges_proportionally() -> crate::Result<()> {
    // Sizes [30, 2, 2, 2, 2, 2]: C = 6, N = 40, n_avg = 20/3.
    // Per-index x = 1.64/(1.64 + 20/3) ~ 0.1975; row_charge =
    // (1 - x)*3/20 ~ 0.1204. units_seg = 6x + (1 - x)*40/(20/3) =
    // 1.185 + 4.815 = 6.0 (identity). The query sits on the big
    // centroid, so it opens first: charge 0.1975 + 30*0.1204 ~ 3.809.
    // Each small cluster charges 0.1975 + 2*0.1204 ~ 0.438. At
    // f = 0.8 (budget 4.8): big -> 3.809, small -> 4.247, small ->
    // 4.685, small -> 5.123 >= 4.8 at the next boundary, so 4 clusters
    // are probed. A cluster-count budget at f = 0.8 would probe
    // ceil(4.8) = 5.
    let centroids = vec![
        [0.0_f32, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
        [30.0, 0.0],
        [40.0, 0.0],
        [50.0, 0.0],
    ];
    let mut docs: Vec<(String, [f32; 2])> = Vec::new();
    for (c, count) in [(0usize, 30usize), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)] {
        for i in 0..count {
            docs.push((
                format!("d{c}_{i}"),
                [centroids[c][0] + i as f32 * 0.001, 0.0],
            ));
        }
    }
    let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
    let params = AdaptiveProbeParams {
        max_probe_fraction: 0.8,
        ..budget_only_params()
    };
    let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
    let (_, stats) = run_top_n(&index, embed_field, vec![0.0, 0.0], 41, params)?;
    assert_eq!(stats.termination, ProbeTermination::Ceiling, "{stats:?}");
    assert_eq!(
        stats.clusters_probed(),
        4,
        "the big cluster eats the budget a count regime spreads over 5: {stats:?}"
    );
    assert!(
        (stats.work_charged - 5.123).abs() < 2e-3,
        "hand-computed spend: {stats:?}"
    );
    // Overshoot bound: the overrun is at most the last (small)
    // cluster's charge.
    let budget = 0.8f32 * 6.0;
    let last_charge = 0.1975 + 2.0 * 0.1204;
    assert!(
        stats.work_charged > budget && stats.work_charged - budget <= last_charge + 1e-4,
        "overshoot bounded by the last cluster's charge: {stats:?}"
    );
    Ok(())
}

/// A cluster's row charge tracks the rows it actually READ AND SCORED,
/// not the rows it walked past. Same fixture, same opens, two runs:
/// unfiltered (every row scored) and filtered to 3 of 22 docs. The
/// filtered run must charge 19 row-shares less - the rows the filter
/// rejected are never fetched, so they cost nothing. Under a
/// first-seen basis the two runs would charge identically, since both
/// walk all 22 rows and mark all 22 seen.
#[test]
fn filtered_rows_are_not_charged() -> crate::Result<()> {
    // Sizes [20, 2]: C = 2, N = 22, n_avg = 11.
    let centroids = vec![[0.0_f32, 0.0], [50.0, 0.0]];
    let mut docs: Vec<(String, [f32; 2])> = Vec::new();
    for (c, count) in [(0usize, 20usize), (1, 2)] {
        for i in 0..count {
            docs.push((
                format!("d{c}_{i}"),
                [centroids[c][0] + i as f32 * 0.001, 0.0],
            ));
        }
    }
    let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
    let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

    let n_avg = 22.0 / 2.0;
    let fixed = fixed_probe_cost_rows();
    let x = fixed / (fixed + n_avg);
    let row = (1.0 - x) / n_avg;

    // Unfiltered: every cluster opened, every row scored - the
    // identity's reference case, so exactly C = 2 units.
    let (_, full) = run_top_n(
        &index,
        embed_field,
        vec![0.0, 0.0],
        23,
        budget_only_params(),
    )?;
    assert_eq!(full.clusters_probed(), 2, "{full:?}");
    assert_eq!(full.candidates_scored, 22, "{full:?}");
    assert!(
        (full.work_charged as f64 - 2.0).abs() < 1e-5,
        "unfiltered exhaustive scan charges exactly C: {full:?}"
    );

    // Admit 3 docs of the big cluster; the other 19 rows are walked
    // and rejected, and cluster 1's 2 rows are walked and rejected.
    let searcher = index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    let admitted: Vec<DocId> = segment_reader
        .vector_index(embed_field)?
        .cluster_doc_ids(0)
        .unwrap()
        .expect("cluster 0 doc ids")
        .into_iter()
        .take(3)
        .collect();
    assert_eq!(admitted.len(), 3, "setup: need 3 admitted docs");
    let weight = FixedDocsWeight {
        max_doc: segment_reader.max_doc(),
        docs: admitted,
    };
    drop(searcher);

    let (_, filtered) = run_top_n_with_weight(
        &index,
        embed_field,
        vec![0.0, 0.0],
        23,
        budget_only_params(),
        &weight,
    )?;
    // Same opens, same rows walked - only the scored count differs.
    assert_eq!(filtered.clusters_probed(), 2, "{filtered:?}");
    assert_eq!(filtered.vectors_visited, 22, "{filtered:?}");
    assert_eq!(filtered.candidates_scored, 3, "{filtered:?}");

    let expected = 2.0 * x + 3.0 * row;
    assert!(
        (filtered.work_charged as f64 - expected).abs() < 1e-5,
        "charge must be 2 opens + 3 scored rows ({expected}): {filtered:?}"
    );
    // The 19 rejected rows cost exactly nothing.
    assert!(
        ((full.work_charged - filtered.work_charged) as f64 - 19.0 * row).abs() < 1e-5,
        "the filtered rows must account for the entire difference: full={full:?} \
         filtered={filtered:?}"
    );
    Ok(())
}

/// The controller stops at the first pull after the estimate reaches
/// the target; clusters covered while the heap fills carry no estimate
/// but still count once it arms.
#[test]
fn probe_controller_stops_at_recall_target() {
    let pricing = UnitPricing {
        budget: WorkUnits::new(100.0),
        open: WorkUnits::new(0.5),
        row: WorkUnits::new(0.01),
    };
    let query = [0.1f32, 0.0];
    let centroids = [[0.0f32, 0.0], [2.0, 0.0], [0.0, 2.0]];
    let rows: Vec<&[f32]> = centroids.iter().map(|row| &row[..]).collect();
    let estimator = RecallEstimator::from_rows(&query, rows, Metric::L2);
    let mut controller = ProbeController::new(pricing, centroids.len(), Some(estimator), 0.5);

    assert!(controller.admit());
    controller.cover(None).unwrap();
    assert!(controller.admit(), "no estimate while the heap fills");
    // A ball of radius 0.2 lies inside the nearest cell.
    controller.cover(Some(-(0.2f32 * 0.2))).unwrap();
    assert!(!controller.admit());

    let mut stats = ProbeStats::default();
    controller.finish(&mut stats);
    assert_eq!(stats.termination, ProbeTermination::RecallTarget);
    assert!(stats
        .recall_estimate
        .is_some_and(|estimate| estimate >= 0.5));
}

/// A stacked segment with a loose recall target stops on the estimate
/// before the (exhaustive) budget binds; target `1.0` leaves APS off.
#[test]
fn probe_stats_recall_target_stops_before_budget() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .build()?;
    let params = |recall_target| AdaptiveProbeParams {
        max_probe_fraction: 1.0,
        min_probe_clusters: 1,
        recall_target,
        ..Default::default()
    };

    let (_, aps) = run_top_n(
        &index.index,
        index.embedding_field(),
        vec![0.0_f32, 0.0],
        3,
        params(0.5),
    )?;
    assert_eq!(aps.termination, ProbeTermination::RecallTarget, "{aps:?}");
    assert!(
        aps.recall_estimate.is_some_and(|estimate| estimate >= 0.5),
        "{aps:?}"
    );

    let (_, off) = run_top_n(
        &index.index,
        index.embedding_field(),
        vec![0.0_f32, 0.0],
        3,
        params(1.0),
    )?;
    assert_ne!(off.termination, ProbeTermination::RecallTarget, "{off:?}");
    assert_eq!(off.recall_estimate, None, "{off:?}");
    assert!(aps.work_charged < off.work_charged, "{aps:?} {off:?}");
    assert_eq!(aps.work_budget, off.work_budget, "{aps:?} {off:?}");
    assert!(aps.work_charged < aps.work_budget, "{aps:?}");
    Ok(())
}

/// Dot has no query ball, so a stacked segment ignores the recall
/// target and scans to its budget.
#[test]
fn probe_stats_recall_target_ignored_for_dot() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .metric(Metric::Dot)
        .build()?;
    let (_, stats) = run_top_n(
        &index.index,
        index.embedding_field(),
        vec![1.0_f32, 0.0],
        3,
        AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 1,
            recall_target: 0.5,
            ..Default::default()
        },
    )?;
    assert_ne!(
        stats.termination,
        ProbeTermination::RecallTarget,
        "{stats:?}"
    );
    assert_eq!(stats.recall_estimate, None, "{stats:?}");
    Ok(())
}

/// APS is stacked-only: other routers ignore the recall target.
#[test]
fn probe_stats_recall_target_ignored_without_stacked_router() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .router(RouterKind::Exact)
        .build()?;
    let (_, stats) = run_top_n(
        &index.index,
        index.embedding_field(),
        vec![0.0_f32, 0.0],
        3,
        AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 1,
            recall_target: 0.5,
            ..Default::default()
        },
    )?;
    assert_eq!(stats.termination, ProbeTermination::Exhausted, "{stats:?}");
    assert_eq!(stats.recall_estimate, None, "{stats:?}");
    Ok(())
}

/// A budget below capacity binds, is attributed to the ceiling, and
/// overshoots by at most one cluster's charge - the boundary rule on
/// a real fixture rather than a hand-built one. The distance-ratio
/// gate is parked (floor unreachable), so the stop point under test
/// is the budget's alone. The exact router ranks every cluster; the
/// stacked router caps its ranking at `router_k`, which on this
/// fixture's empty clusters can run out before the budget binds.
#[test]
fn probe_stats_max_probe_fraction_ceiling() -> crate::Result<()> {
    let index = TestVectorIndex::builder(VectorDType::F32)
        .vector_storage_format(VectorStorageFormat::Ivf)
        .router(RouterKind::Exact)
        .build()?;
    let params = AdaptiveProbeParams {
        max_probe_fraction: 0.2,
        min_probe_clusters: 1,
        ..Default::default()
    };
    let searcher = index.index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    let vec_reader = segment_reader.vector_index(index.embedding_field())?;
    let ivf = vec_reader.index().expect("expected IVF storage");
    let (clusters, docs) = (ivf.num_clusters(), ivf.num_docs());
    let (budget, n_avg, x) = params.resolved_work_budget(clusters, docs)?;
    assert!(budget < clusters as f64, "setup: the budget must bind");
    drop(searcher);

    let (_, stats) = run_top_n(
        &index.index,
        index.embedding_field(),
        vec![0.0_f32, 0.0],
        3,
        params,
    )?;
    assert_eq!(stats.termination, ProbeTermination::Ceiling);
    assert!(
        (stats.work_budget as f64 - budget).abs() <= 1e-6 * budget,
        "the resolved budget is recorded: {stats:?}"
    );
    assert!(
        stats.clusters_probed() < clusters,
        "the budget must bind before exhaustion: {stats:?}"
    );
    assert!(
        stats.work_charged as f64 > budget,
        "the ceiling fires only once the budget is spent: {stats:?}"
    );
    // Overshoot bound: no single cluster can charge more than one
    // open plus every row in the segment.
    let max_cluster_charge = x + docs as f64 * (1.0 - x) / n_avg;
    assert!(
        stats.work_charged as f64 <= budget + max_cluster_charge + 1e-6,
        "overshoot is bounded by the last cluster's charge: {stats:?}"
    );
    Ok(())
}

/// `ProbeStats` round-trips through `serde_json` with the field names
/// callers rely on.
#[test]
fn probe_stats_serializes_to_json() {
    let mut stats = ProbeStats {
        rerank_io: super::super::VectorIoStats {
            reads: 2,
            bytes_read: 256,
            storage_blocks: 3,
        },
        candidates_scored: 10,
        layer0_eligible: 10,
        clusters_skipped_empty: 2,
        eligible_charged: 10,
        rerank_rows: 4,
        scan_init_ns: 75,
        non_vector_search_ns: 60,
        query_prep_ns: 125,
        routing_ns: 40,
        rerank_fetch_ns: 50,
        rerank_score_ns: 25,
        vectors_visited: 20,
        pruned_filter: 4,
        pruned_dead: 3,
        postings_row: 1,
        postings_skipped: 1,
        exact_rows_read: 0,
        bounds_skips: 2,
        termination: ProbeTermination::Ceiling,
        work_charged: 1.75,
        work_budget: 1.5,
        segment_rows: Some(100),
        segment_clusters: Some(5),
        ..Default::default()
    };
    stats.start_layer(0);
    stats.record_layer_scan(0, 10, 250);
    stats.record_boundary(0, 8, 50);
    stats.start_layer(1);
    stats.record_layer_scan(1, 8, 100);
    stats.record_boundary(1, 5, 25);
    stats.record_routing(RouterMetrics::Rng(NeighborhoodGraphSearchMetrics {
        visited_count: 7,
        expanded_count: 4,
        edges_scanned: 12,
        evictions: 1,
        result_count: 3,
        termination_reason: SearchTerminationReason::SearchConverged,
    }));
    stats.record_bound_armed(Some(1));

    let mut value = serde_json::to_value(&stats).expect("ProbeStats should serialize to JSON");
    let object = value.as_object_mut().unwrap();
    for key in [
        "layer0_sign_word_fallbacks",
        "layer0_reads",
        "layer0_bytes_read",
        "layer0_storage_blocks",
        "layer1_sign_word_fallbacks",
        "layer1_reads",
        "layer1_bytes_read",
        "layer1_storage_blocks",
    ] {
        assert_eq!(object.remove(key).unwrap(), 0);
    }
    assert_eq!(object.remove("rerank_reads").unwrap(), 2);
    assert_eq!(object.remove("rerank_bytes_read").unwrap(), 256);
    assert_eq!(object.remove("rerank_storage_blocks").unwrap(), 3);
    assert!(!object.contains_key("rerank_io"));
    assert_eq!(
        value,
        serde_json::json!({
            "candidates_scored": 10,
            "layer0_eligible": 10,
            "clusters_skipped_empty": 2,
            "eligible_charged": 10,
            "layer0_scan_ns": 250,
            "layer0_scored": 10,
            "layer0_survivors": 8,
            "boundary0_ns": 50,
            "layer1_scan_ns": 100,
            "layer1_scored": 8,
            "layer1_survivors": 5,
            "boundary1_ns": 25,
            "rerank_rows": 4,
            "scan_init_ns": 75,
            "non_vector_search_ns": 60,
            "query_prep_ns": 125,
            "routing_ns": 40,
            "rerank_fetch_ns": 50,
            "rerank_score_ns": 25,
            "vectors_visited": 20,
            "pruned_filter": 4,
            "pruned_dead": 3,
            "postings_row": 1,
            "postings_skipped": 1,
            "exact_rows_read": 0,
            "routing_visited_count": 7,
            "routing_graph_count": 1,
            "routing_graph_visited_count": 7,
            "routing_graph_expanded_count": 4,
            "routing_graph_edges_scanned": 12,
            "routing_graph_evictions": 1,
            "routing_graph_result_count": 3,
            "bounds_skips": 2,
            "bound_armed_count": 1,
            "bound_armed_probe_sum": 1,
            "termination": "Ceiling",
            "work_charged": 1.75,
            "work_budget": 1.5,
            "segment_rows": 100,
            "segment_clusters": 5
        })
    );
    assert_eq!(stats.clusters_probed(), 2);

    let mut exact_routing = ProbeStats::default();
    exact_routing.record_routing(RouterMetrics::Exact { visited_count: 7 });
    let exact_value =
        serde_json::to_value(&exact_routing).expect("ProbeStats should serialize to JSON");
    assert_eq!(exact_value["routing_visited_count"], 7);
    assert_eq!(exact_value["routing_graph_count"], 0);
    assert!(exact_value.get("routing").is_none());
    assert!(exact_value.get("layer0_scan_ns").is_none());
}

#[test]
fn quantized_boundary_kth_uses_the_row_tie_for_sigma() {
    let mut scan = QuantizedScanCtx::new(3, 3);
    scan.begin_cluster(2);
    for (row, score, sigma) in [(0, 10.0, 0.0), (1, 9.0, 1.0), (2, 9.0, 100.0)] {
        push_test_candidate(&mut scan, row, row as DocId, score, sigma);
    }
    scan.finish_cluster_bound_with_kappa(2.0);

    assert_eq!(
        scan.running_pessimistic_kth(2, 2.0),
        Some(Threshold(LowerEndpoint(7.0)))
    );
    assert_eq!(
        scan.pessimistic_kth(2, 2.0),
        Some(Threshold(LowerEndpoint(7.0)))
    );
}

/// The APS kth is the lowest point estimate in the lower-endpoint
/// top-k: above the pessimistic kth, and never above the true k-th
/// best estimate.
#[test]
fn running_estimate_kth_bounds_the_kth_estimate() {
    let mut scan = QuantizedScanCtx::new(3, 3);
    assert_eq!(scan.running_estimate_kth(2), None);
    scan.begin_cluster(2);
    // Lower endpoints at kappa 2: 10, 7, 4. The top-2 by lower endpoint
    // is rows 0 and 1; the top-2 by estimate is rows 2 and 0.
    for (row, score, sigma) in [(0, 10.0, 0.0), (1, 9.0, 1.0), (2, 12.0, 4.0)] {
        push_test_candidate(&mut scan, row, row as DocId, score, sigma);
    }
    scan.finish_cluster_bound_with_kappa(2.0);

    assert_eq!(
        scan.running_pessimistic_kth(2, 2.0),
        Some(Threshold(LowerEndpoint(7.0)))
    );
    assert_eq!(scan.running_estimate_kth(2), Some(9.0));
    assert_eq!(scan.running_estimate_kth(4), None);
}

#[test]
fn cluster_local_admission_kth_matches_full_partition() {
    const TOP_N: usize = 4;
    let mut scan = QuantizedScanCtx::new(12, 24);
    for cluster in 0..6 {
        scan.begin_cluster(TOP_N);
        for row in cluster * 4..cluster * 4 + 4 {
            push_test_candidate(
                &mut scan,
                row,
                (row % 12) as DocId,
                3.0 - row as f32 * 0.071 + (row as f32 * 0.37).sin() * 0.2,
                0.01 + (row % 5) as f32 * 0.003,
            );
        }
        scan.finish_cluster_bound_with_kappa(2.0);
        assert_eq!(
            scan.running_pessimistic_kth(TOP_N, 2.0),
            scan.pessimistic_kth(TOP_N, 2.0),
            "cluster={cluster}"
        );
    }
}

fn independent_admission_top(scan: &QuantizedScanCtx, top_n: usize) -> Vec<usize> {
    let mut indices = (0..scan.candidates.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        let left_estimate = scan.candidates.estimates[left] - 2.0 * scan.candidates.sigmas[left];
        let right_estimate = scan.candidates.estimates[right] - 2.0 * scan.candidates.sigmas[right];
        right_estimate
            .total_cmp(&left_estimate)
            .then(scan.candidates.rows[left].cmp(&scan.candidates.rows[right]))
    });
    indices.truncate(top_n);
    indices
}

#[test]
fn cluster_batch_selection_matches_independent_oracle() {
    let clusters: &[&[(usize, DocId, f32, f32)]] = &[
        &[(0, 0, 1.0, 0.01), (1, 1, 2.0, 0.02)],
        &[
            (2, 2, 9.0, 0.03),
            (3, 3, 8.0, 0.04),
            (4, 4, 7.0, 0.05),
            (5, 0, 6.0, 0.06),
            (6, 5, 5.0, 0.07),
        ],
        &[
            (7, 6, 0.0, 0.08),
            (8, 1, 10.0, 0.09),
            (9, 7, 1.0, 0.10),
            (10, 2, 9.0, 8.0),
            (11, 8, 2.0, 0.11),
        ],
    ];

    for top_n in [1, 3] {
        let mut scan = QuantizedScanCtx::new(16, 16);
        for (cluster, rows) in clusters.iter().enumerate() {
            scan.begin_cluster(top_n);
            for &(row, doc, estimate, sigma) in *rows {
                push_test_candidate(&mut scan, row, doc, estimate, sigma);
            }
            scan.finish_cluster_bound_with_kappa(2.0);

            let expected = independent_admission_top(&scan, top_n);
            let actual_rows = scan
                .bound_top
                .iter()
                .map(|&index| scan.candidates.rows[index])
                .collect::<Vec<_>>();
            let expected_rows = expected
                .iter()
                .map(|&index| scan.candidates.rows[index])
                .collect::<Vec<_>>();
            assert_eq!(
                actual_rows, expected_rows,
                "cluster={cluster}, top_n={top_n}"
            );

            let expected_kth = (expected.len() == top_n).then(|| {
                let index = expected[top_n - 1];
                Threshold(LowerEndpoint(
                    scan.candidates.estimates[index] - 2.0 * scan.candidates.sigmas[index],
                ))
            });
            assert_eq!(
                scan.running_pessimistic_kth(top_n, 2.0),
                expected_kth,
                "cluster={cluster}, top_n={top_n}"
            );
        }
    }
}

#[test]
fn cosine_refinement_batches_cross_clusters_and_cap_at_2048() {
    let rows = (0..2_049).collect::<Vec<_>>();
    let clusters = [0..700, 700..1_400, 1_400..2_100];
    let batches = cosine_refinement_batches(rows.len()).collect::<Vec<_>>();
    assert_eq!(batches, [0..2_048, 2_048..2_049]);
    let first_rows = rows[batches[0].clone()].iter().copied();
    for cluster in clusters {
        assert!(
            first_rows.clone().any(|row| cluster.contains(&row)),
            "one logical cosine batch must cross all three clusters"
        );
    }
}

#[test]
fn initial_l2_gamma_leaves_exact_base_unscaled() {
    let mut raw_prefixes = [2.0];
    let mut bases = [0.0];
    let mut estimates = [0.0];
    let mut sigmas = [0.0];
    let mut residual_norms_squared = [0.0];
    let mut sign_query_error_terms = [0.0];
    combine_initial_decoded(
        Metric::L2,
        1,
        &mut raw_prefixes,
        &mut bases,
        &mut estimates,
        &mut sigmas,
        &mut residual_norms_squared,
        &mut sign_query_error_terms,
        &mut [ArithmeticError::default()],
        &[3.0],
        &[2.0],
        &[0.5],
        &[5.0],
        &[7.0],
        10.0,
        11.0,
        13.0,
    );

    assert_eq!(bases, [3.0]);
    assert_eq!(raw_prefixes, [1.0]);
    assert_eq!(estimates, [7.0]);
    assert_eq!(residual_norms_squared, [7.0]);
    assert_eq!(sign_query_error_terms, [117.0]);
    let variance: f32 = 7.0 * 0.5 * 11.0 + 4.0 * 117.0;
    let expected_sigma = 2.0 * 1.15 * variance.sqrt();
    assert!((sigmas[0] - expected_sigma).abs() < 1e-5);
}

#[test]
fn l2_refinement_gamma_corrects_only_raw_prefix_and_keeps_state_local() {
    let mut candidates = QuantizedCandidates::with_capacity(2);
    candidates.push(
        0,
        10,
        10.0,
        2.0,
        14.0,
        0.0,
        9.0,
        1.0,
        0.0,
        ArithmeticError::default(),
    );
    candidates.push(
        1,
        11,
        20.0,
        1.0,
        22.0,
        0.0,
        4.0,
        1.0,
        1.0,
        ArithmeticError::default(),
    );

    combine_refinement_decoded(
        Metric::L2,
        1,
        &mut candidates,
        0..1,
        &[2.0],
        &[3.0],
        &[2.0],
        &[0.5],
        &[5.0],
        7.0,
        11.0,
    );
    combine_refinement_decoded(
        Metric::L2,
        1,
        &mut candidates,
        1..2,
        &[4.0],
        &[2.0],
        &[1.5],
        &[0.25],
        &[1.0],
        17.0,
        13.0,
    );

    assert_eq!(candidates.bases, [10.0, 20.0]);
    assert_eq!(candidates.raw_prefixes, [3.0, 8.0]);
    assert_eq!(candidates.estimates, [22.0, 44.0]);
    assert_eq!(candidates.residual_norm_squared, [9.0, 4.0]);
    assert_eq!(candidates.sign_query_error_terms, [99.0, 53.0]);
    let expected0 = 2.0 * 1.15 * 427.5_f32.sqrt();
    let expected1 = 2.0 * 1.15 * 136.25_f32.sqrt();
    assert!((candidates.sigmas[0] - expected0).abs() < 1e-5);
    assert!((candidates.sigmas[1] - expected1).abs() < 1e-5);
}

#[test]
fn survivor_sets_match_kernel_harness() {
    const CANDIDATES: usize = 40;
    const TOP_N: usize = 10;
    const KAPPA: f32 = QUANTIZED_BOUNDARY_KAPPA;

    let layer0_scores: Vec<f32> = (0..CANDIDATES)
        .map(|row| 2.0 - row as f32 * 0.09 + (row as f32 * 0.7).sin() * 0.08)
        .collect();
    let layer0_sigmas: Vec<f32> = (0..CANDIDATES)
        .map(|row| 0.03 + (row % 5) as f32 * 0.01)
        .collect();
    let refinements: Vec<f32> = (0..CANDIDATES)
        .map(|row| (row as f32 * 0.41).cos() * 0.06)
        .collect();
    let layer1_sigmas: Vec<f32> = (0..CANDIDATES)
        .map(|row| 0.012 + (row % 3) as f32 * 0.004)
        .collect();

    let lower: Vec<f32> = layer0_scores
        .iter()
        .zip(&layer0_sigmas)
        .map(|(&estimate, &sigma)| estimate - KAPPA * sigma)
        .collect();
    let (_, first_kth) = cascade::kth(&lower, TOP_N);
    let harness_first = cascade::band_filter(&layer0_scores, &layer0_sigmas, KAPPA, first_kth);

    let mut scan = QuantizedScanCtx::new(CANDIDATES as DocId, CANDIDATES);
    for row in 0..CANDIDATES {
        push_test_candidate(
            &mut scan,
            row,
            row as DocId,
            layer0_scores[row],
            layer0_sigmas[row],
        );
    }
    scan.band(TOP_N, KAPPA);
    let scan_first: Vec<u32> = scan.candidates.rows.iter().map(|&row| row as u32).collect();
    assert!(
        harness_first.len() < CANDIDATES,
        "first boundary must measurably filter"
    );
    assert_eq!(scan_first, harness_first, "layer-0 survivor set");

    let second_scores: Vec<f32> = harness_first
        .iter()
        .map(|&row| layer0_scores[row as usize] + refinements[row as usize])
        .collect();
    let second_sigmas: Vec<f32> = harness_first
        .iter()
        .map(|&row| layer1_sigmas[row as usize])
        .collect();
    let lower: Vec<f32> = second_scores
        .iter()
        .zip(&second_sigmas)
        .map(|(&estimate, &sigma)| estimate - KAPPA * sigma)
        .collect();
    let (_, second_kth) = cascade::kth(&lower, TOP_N);
    let harness_second_local =
        cascade::band_filter(&second_scores, &second_sigmas, KAPPA, second_kth);
    let harness_second: Vec<u32> = harness_second_local
        .iter()
        .map(|&local| harness_first[local as usize])
        .collect();

    for index in 0..scan.candidates.len() {
        let row = scan.candidates.rows[index];
        scan.candidates.estimates[index] += refinements[row];
        scan.candidates.sigmas[index] = layer1_sigmas[row];
    }
    scan.band(TOP_N, KAPPA);
    let scan_second: Vec<u32> = scan.candidates.rows.iter().map(|&row| row as u32).collect();
    assert!(
        harness_second.len() < harness_first.len(),
        "second boundary must measurably filter"
    );
    assert_eq!(scan_second, harness_second, "layer-1 survivor set");

    let mut exact_order: Vec<usize> = (0..CANDIDATES).collect();
    exact_order.sort_unstable_by(|&left, &right| {
        (layer0_scores[right] + refinements[right])
            .total_cmp(&(layer0_scores[left] + refinements[left]))
            .then_with(|| left.cmp(&right))
    });
    let recalled = exact_order[..TOP_N]
        .iter()
        .filter(|&&row| scan_second.contains(&(row as u32)))
        .count();
    assert_eq!(recalled, TOP_N, "candidate recall must match the harness");
}

#[test]
fn quantized_boundary_filters_with_finite_sigmas() {
    let mut scan = QuantizedScanCtx::new(3, 3);
    for (row, score) in [(0, 100.0), (1, 99.0), (2, -100.0)] {
        push_test_candidate(&mut scan, row, row as DocId, score, 0.1);
    }
    scan.band(1, 2.0);
    assert_eq!(scan.candidates.len(), 1);
    assert_eq!(scan.candidates.rows[0], 0);
}

/// A `Weight` over a fixed, hand-built doc-id set, so tests can hand
/// the backend an exact filter BitSet without routing it through a
/// real query.
struct FixedDocsWeight {
    max_doc: DocId,
    docs: Vec<DocId>,
}

impl Weight for FixedDocsWeight {
    fn scorer(&self, _reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let mut bs = BitSet::with_max_value(self.max_doc);
        for &doc in &self.docs {
            bs.insert(doc);
        }
        Ok(Box::new(ConstScorer::new(BitSetDocSet::from(bs), boost)))
    }

    fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> crate::Result<Explanation> {
        unreachable!("the vector backend never explains filter docs")
    }
}

/// Like [`run_top_n`] but with a caller-supplied filter
/// weight.
fn run_top_n_with_weight(
    index: &Index,
    embed_field: Field,
    query: Vec<f32>,
    k: usize,
    params: AdaptiveProbeParams,
    weight: &dyn Weight,
) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
    let searcher = index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    let backend = VectorBackend::<f32>::for_segment(
        segment_reader,
        0,
        embed_field,
        VectorQuery::new(Arc::new(query), None),
        params,
    )?;
    assert!(
        segment_reader.vector_index(embed_field)?.index().is_some(),
        "expected IVF storage"
    );
    backend.top_n(weight, segment_reader, k)
}

/// Every touched row lands in exactly one prune bucket.
fn assert_stats_identities(stats: &ProbeStats) {
    assert_eq!(
        stats.vectors_visited,
        stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
        "visited must equal filter+dead+scored ({stats:?})"
    );
}

/// Build a single-segment FLAT index (one commit, never merged past
/// the clustering threshold): `docs` are `(label, Some(vector))`, or
/// `(label, None)` for vectorless docs — mixing the two forces the
/// `Bitmap` id-map.
fn build_flat(
    dim: usize,
    docs: &[(&str, Option<Vec<f32>>)],
) -> crate::Result<(Index, Field, Field)> {
    let mut sb = Schema::builder();
    let embed_field = sb.add_vector_field(
        "embedding",
        VectorOptions::new(dim, Metric::L2).with_dtype(VectorDType::F32),
    );
    let label_field = sb.add_text_field("label", STRING | STORED);
    let index = Index::create_in_ram(sb.build());
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 30_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for (label, v) in docs {
        let mut doc = TantivyDocument::new();
        doc.add_text(label_field, label);
        if let Some(v) = v {
            doc.add_vector(embed_field, v.as_slice());
        }
        writer.add_document(doc)?;
    }
    writer.commit()?;
    Ok((index, embed_field, label_field))
}

/// Run the flat/exact path against `segment_reader` with a
/// caller-supplied filter weight.
fn run_exact_on_segment(
    segment_reader: &SegmentReader,
    embed_field: Field,
    query: Vec<f32>,
    k: usize,
    weight: &dyn Weight,
) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
    let backend = VectorBackend::<f32>::for_segment(
        segment_reader,
        0,
        embed_field,
        VectorQuery::new(Arc::new(query), None),
        AdaptiveProbeParams::default(),
    )?;
    assert!(
        segment_reader.vector_index(embed_field)?.index().is_none(),
        "expected flat storage"
    );
    backend.top_n(weight, segment_reader, k)
}

/// Filter-aware fetches across selectivities: on the multi-cluster
/// fixture, hand-built filters admitting {0, 1, 50, 100}% of docs
/// return every admitted doc exactly once, with both partition
/// identities intact. 0% admits nothing — the empty-filter
/// short-circuit returns before the probe loop, so zero clusters
/// probe and zero fetches happen (postings_skipped == clusters
/// probed == 0).
#[test]
fn filter_aware_fetch_across_selectivities() -> crate::Result<()> {
    let (centroids, labels) = multi_cluster_fixture();
    let docs = multi_cluster_docs(&centroids, &labels);
    let n = docs.len();
    let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
    let params = exhaustive_params(centroids.len());

    for pct in [0usize, 1, 50, 100] {
        // Admit the first ceil(pct% · n) doc ids: doc · 100 < pct · n.
        let weight = FixedDocsWeight {
            max_doc: n as DocId,
            docs: (0..n as DocId)
                .filter(|&doc| (doc as usize) * 100 < pct * n)
                .collect(),
        };
        let admitted = weight.docs.len();
        let (hits, stats) = run_top_n_with_weight(
            &index,
            embed_field,
            vec![10.0, 10.0],
            n,
            params.clone(),
            &weight,
        )?;

        assert_eq!(
            hits.len(),
            admitted,
            "{pct}%: every admitted doc returns exactly once"
        );
        assert_stats_identities(&stats);

        match pct {
            0 => {
                assert_eq!(stats.postings_row, 0, "0%: no fetches");
                assert_eq!(
                    stats.postings_skipped,
                    stats.clusters_probed(),
                    "0%: every probed cluster skips its fetch"
                );
            }
            1 => {
                // One admitted doc → exactly one cluster fetches (its
                // primary); every other probed cluster skips.
                assert_eq!(stats.postings_row, 1, "{stats:?}");
                assert_eq!(
                    stats.postings_skipped,
                    stats.clusters_probed() - 1,
                    "{stats:?}"
                );
            }
            _ => {
                assert!(stats.postings_row > 0, "{stats:?}");
            }
        }
    }
    Ok(())
}

/// Deletes are decided in the pre-pass: a cluster whose rows are all
/// dead yields zero survivors and fetches nothing.
#[test]
fn filter_aware_fetch_skips_all_dead_clusters() -> crate::Result<()> {
    let (centroids, labels) = multi_cluster_fixture();
    let docs = multi_cluster_docs(&centroids, &labels);
    let n = docs.len();
    // Cluster 0's rows are exactly its 6 primary docs — deleting
    // those leaves a fully-dead cluster.
    let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
    {
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for i in 0..DOCS_PER_CLUSTER {
            writer.delete_term(Term::from_field_text(label_field, &format!("d{i}")));
        }
        writer.commit()?;
    }

    let searcher = index.reader()?.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    let segment_reader = &searcher.segment_readers()[0];
    // Setup: the tombstones landed, and cluster 0 is exactly the
    // dead docs.
    let alive = segment_reader.alive_bitset().expect("deletes must land");
    let cluster0 = segment_reader
        .vector_index(embed_field)?
        .cluster_doc_ids(0)
        .unwrap()
        .expect("ivf cluster 0");
    assert_eq!(cluster0.len(), DOCS_PER_CLUSTER);
    assert!(
        cluster0.iter().all(|&doc| !alive.is_alive(doc)),
        "cluster 0 must be fully dead"
    );

    let max_doc = segment_reader.max_doc();
    let weight = FixedDocsWeight {
        max_doc,
        docs: (0..max_doc).collect(),
    };
    let params = exhaustive_params(centroids.len());
    let (hits, stats) =
        run_top_n_with_weight(&index, embed_field, vec![10.0, 10.0], n, params, &weight)?;

    assert_eq!(hits.len(), n - DOCS_PER_CLUSTER, "only alive docs surface");
    assert_eq!(
        stats.pruned_dead, DOCS_PER_CLUSTER,
        "every dead row prunes as dead: {stats:?}"
    );
    // The fully-dead cluster fetches nothing; the other five fetch.
    assert_eq!(stats.postings_skipped, 1, "{stats:?}");
    assert_eq!(stats.postings_row, centroids.len() - 1);
    assert_stats_identities(&stats);
    Ok(())
}

/// An empty cluster still counts as probed — the loop visits it and
/// takes the skip path: `postings_skipped` increments, nothing is
/// fetched, and the visited/prune counters don't move.
#[test]
fn empty_cluster_probed_but_fetch_skipped() -> crate::Result<()> {
    // No doc is nearest to the third centroid, so its cluster is
    // empty. The
    // empty centroid sits ON the query so its zero-radius bound can
    // never prove it useless - the bounds gate must not be what
    // skips it; the empty-fetch path is what's under test.
    let centroids = vec![[0.0f32, 0.0], [10.0, 0.0], [5.0, 0.1]];
    let labels: Vec<String> = (0..8).map(|i| format!("d{i}")).collect();
    let docs: Vec<(&str, [f32; 2])> = (0..8)
        .map(|i| {
            let c = centroids[i % 2];
            (labels[i].as_str(), [c[0] + (i / 2) as f32 * 0.01, c[1]])
        })
        .collect();
    let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

    let searcher = index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    let vec_reader = segment_reader.vector_index(embed_field)?;
    assert_eq!(
        vec_reader.cluster_sizes(),
        Some(vec![4, 4, 0]),
        "setup: cluster 2 must be empty"
    );

    let max_doc = segment_reader.max_doc();
    let weight = FixedDocsWeight {
        max_doc,
        docs: (0..max_doc).collect(),
    };
    let (hits, stats) = run_top_n_with_weight(
        &index,
        embed_field,
        vec![5.0, 0.0],
        8,
        exhaustive_params(centroids.len()),
        &weight,
    )?;
    assert_eq!(hits.len(), 8);
    assert_eq!(
        stats.clusters_probed(),
        centroids.len(),
        "the empty cluster still counts as probed: {stats:?}"
    );
    assert_eq!(
        stats.postings_skipped, 1,
        "the empty cluster fetched nothing: {stats:?}"
    );
    assert_eq!(stats.postings_row, 2);
    assert_eq!(
        stats.vectors_visited, 8,
        "an empty cluster contributes no visited rows"
    );
    assert_stats_identities(&stats);
    Ok(())
}

/// Flat-path behavior across filter selectivities: hand-built filters
/// admitting {0, 1, 50, 100}% of docs return every admitted doc
/// exactly once, the probe-loop counters stay zeroed, and the path
/// serves exactly one stride-sized row read per survivor.
#[test]
fn flat_exact_reads_one_row_per_survivor() -> crate::Result<()> {
    let n = 40usize;
    let labels: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
    let docs: Vec<(&str, Option<Vec<f32>>)> = (0..n)
        .map(|i| (labels[i].as_str(), Some(vec![i as f32 * 0.1, 1.0])))
        .collect();
    let (index, embed_field, _label) = build_flat(2, &docs)?;
    let searcher = index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    let max_doc = segment_reader.max_doc();
    assert_eq!(max_doc as usize, n, "one segment holding every doc");

    for pct in [0usize, 1, 50, 100] {
        let weight = FixedDocsWeight {
            max_doc,
            docs: (0..max_doc)
                .filter(|&doc| (doc as usize) * 100 < pct * n)
                .collect(),
        };
        let admitted = weight.docs.len();

        let (hits, stats) =
            run_exact_on_segment(segment_reader, embed_field, vec![0.0, 0.0], n, &weight)?;

        assert_eq!(hits.len(), admitted, "{pct}%");
        // The exact path fills only `exact_rows_read`; the probe-loop
        // fields must stay zeroed.
        assert_eq!(stats.vectors_visited, 0, "{pct}%: {stats:?}");
        assert_eq!(stats.candidates_scored, 0, "{pct}%: {stats:?}");
        assert_eq!(stats.clusters_probed(), 0, "{pct}%: {stats:?}");
        assert_eq!(
            stats.exact_rows_read, admitted,
            "{pct}%: one row read per survivor"
        );
        assert!(stats.exact_scan_ns.is_some(), "{pct}%: {stats:?}");
        assert!(stats.result_assembly_ns.is_some(), "{pct}%: {stats:?}");
    }
    Ok(())
}

/// Flat scan over a `Bitmap` id-map with tombstoned docs: vectorless
/// and deleted docs leave holes between surviving rows; every alive
/// doc with a vector is read and scored exactly once, matching the
/// brute-force oracle.
#[test]
fn flat_exact_handles_bitmap_holes_and_deletes() -> crate::Result<()> {
    let n = 30usize;
    let labels: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
    // Every third doc carries no vector at all → the id-map is Bitmap
    // and those docs own no row.
    let docs: Vec<(&str, Option<Vec<f32>>)> = (0..n)
        .map(|i| {
            let v = (i % 3 != 2).then(|| vec![i as f32 * 0.1, 1.0]);
            (labels[i].as_str(), v)
        })
        .collect();
    let (index, embed_field, label_field) = build_flat(2, &docs)?;
    // Tombstone a few vectored docs → alive holes between rows that
    // do exist.
    {
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for i in [0usize, 6, 7, 12] {
            writer.delete_term(Term::from_field_text(label_field, &format!("d{i}")));
        }
        writer.commit()?;
    }
    let searcher = index.reader()?.searcher();
    let segment_reader = &searcher.segment_readers()[0];
    let vec_reader = segment_reader.vector_index(embed_field)?;
    // Setup: mixed vector coverage (Bitmap id-map: fewer rows than
    // docs) and the tombstones landed.
    assert!(vec_reader.num_vectors() < segment_reader.max_doc() as usize);
    assert!(segment_reader.alive_bitset().is_some());

    let max_doc = segment_reader.max_doc();
    let weight = FixedDocsWeight {
        max_doc,
        docs: (0..max_doc).collect(),
    };
    let query = vec![0.0f32, 0.0];
    let (hits, stats) =
        run_exact_on_segment(segment_reader, embed_field, query.clone(), n, &weight)?;

    // Oracle: every alive doc with a vector, exactly once.
    let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, n)?;
    assert_eq!(hits, expected);

    let survivors = hits.len();
    assert!(survivors > 0, "fixture must leave survivors");
    assert_eq!(
        stats.exact_rows_read, survivors,
        "one row read per survivor: {stats:?}"
    );
    Ok(())
}

/// Compute a brute-force top-K with the same convention as the
/// shared fixture's `ground_truth::top_k`, but accepting any
/// `&Index` (the inline-built IVF index for crafted tests doesn't
/// have a `TestVectorIndex` wrapper).
fn ground_truth_top_k(
    index: &Index,
    vec_field: Field,
    metric: Metric,
    query: &[f32],
    top_k: usize,
) -> crate::Result<Vec<(Score, DocAddress)>> {
    let query = PreparedQuery::<f32>::new(metric, Arc::new(query.to_vec()));
    let searcher = index.reader()?.searcher();
    let mut scored = Vec::new();
    for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
        let vec_reader = segment_reader.vector_index(vec_field)?;
        let alive = segment_reader.alive_bitset();
        for doc in 0..segment_reader.max_doc() {
            if let Some(alive) = alive {
                if !alive.is_alive(doc) {
                    continue;
                }
            }
            if let Some(bytes) = vec_reader.vector_bytes(doc)? {
                scored.push((
                    query.score_doc_bytes(&bytes),
                    DocAddress::new(seg_ord as u32, doc),
                ));
            }
        }
    }
    scored.sort_by(|a: &(Score, DocAddress), b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.segment_ord.cmp(&b.1.segment_ord))
            .then(a.1.doc_id.cmp(&b.1.doc_id))
    });
    scored.truncate(top_k);
    Ok(scored)
}

/// Collect the set of `DocAddress`es that a `Query` admits, by
/// walking the per-segment weight. Used by the filter selectivity
/// test to build an oracle restricted to the filter set.
fn collect_filter_doc_set(
    index: &Index,
    filter: &dyn Query,
) -> crate::Result<std::collections::HashSet<DocAddress>> {
    let searcher = index.reader()?.searcher();
    let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
    let mut admitted = std::collections::HashSet::new();
    for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
        weight.for_each_no_score(segment_reader, &mut |docs| {
            for &d in docs {
                admitted.insert(DocAddress::new(seg_ord as u32, d));
            }
        })?;
    }
    Ok(admitted)
}

/// Read the stored label text at the given `DocAddress`.
/// Used by the trap-case + floor tests to identify docs by name
/// rather than relying on DocId (which the merger reassigns).
pub(in crate::vector) fn stored_label_at(
    index: &Index,
    label_field: Field,
    addr: DocAddress,
) -> crate::Result<String> {
    use crate::schema::Value;
    let searcher = index.reader()?.searcher();
    let doc = searcher.doc::<TantivyDocument>(addr)?;
    Ok(doc
        .get_first(label_field)
        .and_then(|v| Value::as_str(&v))
        .expect("stored label")
        .to_string())
}

/// The shared fixture's first centroid (top-left of the 3×3 grid).
fn grid2d_first_centroid() -> [f32; 2] {
    [0.0, 0.0]
}

// ==================================================================
// P5: the bounds gate
// ==================================================================
mod bounds_gate_tests {
    use super::*;
    use crate::vector::bounds::{HeapPeek, QueryBound, QueryBoundTracker};
    use crate::vector::{margin_ball_ball, margin_ball_halfspace, to_bound_space};

    /// A single-segment IVF index over `docs` with fixed `centroids`:
    /// ONE commit, then a single-segment merge — a multi-segment
    /// merge's source order varies across processes and permutes
    /// target doc ids, and these tests assert doc-id-level results.
    pub(super) fn single_segment_fixture(
        metric: Metric,
        centroids: &[[f32; 2]],
        docs: &[[f32; 2]],
    ) -> crate::Result<(Index, Field)> {
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
        );
        sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(InlineClusterer {
                centroids: centroids.to_vec(),
            }))
            .ivf_router(RouterKind::Stacked)?
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for (i, vector) in docs.iter().enumerate() {
            let mut doc = TantivyDocument::new();
            doc.add_text(index.schema().get_field("label").unwrap(), format!("d{i}"));
            doc.add_vector(embed_field, vector.as_slice());
            writer.add_document(doc)?;
        }
        writer.commit()?;
        let segment_ids = index.searchable_segment_ids()?;
        assert_eq!(segment_ids.len(), 1, "single flat segment");
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;
        Ok((index, embed_field))
    }

    /// Deterministic pseudo-random `f32` in `[-8, 8)` — a tiny LCG so
    /// the sweep needs no RNG dependency and every run replays.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 16.0
        }
        fn point(&mut self) -> [f32; 2] {
            loop {
                let p = [self.next_f32(), self.next_f32()];
                // Keep cosine's write normalization well-conditioned.
                if p[0] * p[0] + p[1] * p[1] > 0.25 {
                    return p;
                }
            }
        }
    }

    /// PROPERTY TEST — the home-cluster closure theorem. Random data x
    /// {L2, cosine, dot}; brute-force top-k vs gated top-k under a
    /// full budget. Asserts:
    ///
    /// (a) the result sets are identical — a wrong skip would lose a
    ///     true member;
    /// (b) no true member's HOME cluster is skippable even at the
    ///     FINAL (tightest) bound: `margin(t_final) >= 0`. The
    ///     running bound is never tighter than the final one (the kth
    ///     only improves), so this proves the home cluster probed at
    ///     every point of the scan — the bound fold covers every row,
    ///     so a qualifying row's cluster always fails the skip test.
    #[test]
    fn closure_no_true_member_skipped() -> crate::Result<()> {
        let k = 5;
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            for seed in [11u64, 29, 47] {
                let mut rng = Lcg(seed ^ (metric as u64) << 32);
                let centroids: Vec<[f32; 2]> = (0..5).map(|_| rng.point()).collect();
                let docs: Vec<[f32; 2]> = (0..40).map(|_| rng.point()).collect();
                let (index, field) = single_segment_fixture(metric, &centroids, &docs)?;
                let query: Vec<f32> = rng.point().to_vec();

                let brute = ground_truth_top_k(&index, field, metric, &query, k)?;
                let gated = search(
                    &index,
                    field,
                    &AllQuery,
                    query.clone(),
                    k,
                    exhaustive_params(centroids.len()),
                )?;
                assert_eq!(
                    gated, brute,
                    "{metric:?} seed {seed}: gated top-k must equal brute force"
                );

                // (b): the theorem, cluster by cluster. Homes are
                // recomputed with the clusterer's own rule: stored
                // (post-normalization) doc values against the RAW
                // trained centroids - the values `assign` saw. The
                // margin then runs against the STORED (normalized)
                // centroid, exactly as the gate does; the fold covers
                // members whatever rule assigned them, so the
                // triangle argument is assignment-rule-agnostic.
                let searcher = index.reader()?.searcher();
                let segment_reader = &searcher.segment_readers()[0];
                let vec_reader = segment_reader.vector_index(field)?;
                let ivf = vec_reader.index().expect("IVF segment");
                let bounds = ivf.bounds();
                let centroid_bytes = ivf.centroid_bytes()?;
                let stride = 2 * std::mem::size_of::<f32>();
                let kth_key = brute[k - 1].0;
                let t_final = to_bound_space(metric, kth_key);
                let q_norm = (query[0] * query[0] + query[1] * query[1]).sqrt();
                for &(_, addr) in &brute {
                    let stored = decode_2d(
                        &vec_reader
                            .vector_bytes(addr.doc_id)?
                            .expect("stored vector"),
                    );
                    let home = nearest_centroid(stored, &centroids);
                    let sim = Metric::similarity_bytes::<f32>(
                        metric,
                        &query,
                        &centroid_bytes[home * stride..(home + 1) * stride],
                    );
                    let r = bounds.ball_r(home);
                    let margin = match metric {
                        Metric::L2 | Metric::Cosine => {
                            margin_ball_ball(t_final, r, to_bound_space(metric, sim.score()))
                        }
                        Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t_final),
                    };
                    assert!(
                        margin >= 0.0,
                        "{metric:?} seed {seed}: true member {} home cluster {home} must never be \
                         skippable (margin {margin})",
                        addr.doc_id
                    );
                }
            }
        }
        Ok(())
    }

    /// Boundary ties, L2: a cluster whose margin is EXACTLY zero (all
    /// values powers of two — the arithmetic is exact) holds a doc
    /// tying the kth at d == t. Exact touch must PROBE: the tie doc
    /// is scored and the doc-id tie-break decides, identically to
    /// brute force. The far cluster is provably useless and skipped.
    #[test]
    fn boundary_tie_probes_l2() -> crate::Result<()> {
        let centroids = [[0.0f32, 4.0], [4.0, 0.0], [8.0, 0.0]];
        // d0 home A (r 2), d1 home C at margin-zero touch (r 2),
        // d2 home B (r 1, disjoint by 5 - strictly nearest B, no
        // assignment tie with C).
        let docs = [[0.0f32, 2.0], [2.0, 0.0], [7.0, 0.0]];
        let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs)?;
        let query = vec![0.0f32, 0.0];

        let brute = ground_truth_top_k(&index, field, Metric::L2, &query, 1)?;
        assert_eq!(
            brute[0].1.doc_id, 0,
            "tie at d = 2 breaks to the lower doc id"
        );

        let (hits, stats) = run_top_n(&index, field, query, 1, exhaustive_params(3))?;
        assert_eq!(
            hits, brute,
            "exact-touch cluster must probe, preserving the tie"
        );
        assert_eq!(
            stats.candidates_scored, 2,
            "both d = 2 docs are scored - the margin == 0 cluster probed"
        );
        assert_eq!(
            stats.clusters_probed(),
            2,
            "the disjoint cluster is skipped, the touching one is not"
        );
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        Ok(())
    }

    /// Boundary ties, dot: the second cluster's best possible score
    /// (q.c + ||q||*r) EQUALS the kth score — margin exactly zero,
    /// integer arithmetic. Exact touch probes; the tied doc wins on
    /// doc id exactly as brute force says.
    #[test]
    fn boundary_tie_probes_dot() -> crate::Result<()> {
        let centroids = [[2.0f32, 0.0], [4.0, 4.0]];
        // d0 = (3, 0) home c0, score 3; d1 = (3, 4) home c1, score 3.
        let docs = [[3.0f32, 0.0], [3.0, 4.0]];
        let (index, field) = single_segment_fixture(Metric::Dot, &centroids, &docs)?;
        let query = vec![1.0f32, 0.0];

        let brute = ground_truth_top_k(&index, field, Metric::Dot, &query, 1)?;
        assert_eq!(
            brute[0].1.doc_id, 0,
            "score tie at 3 breaks to the lower doc id"
        );

        let (hits, stats) = run_top_n(&index, field, query, 1, exhaustive_params(2))?;
        assert_eq!(hits, brute);
        assert_eq!(
            stats.candidates_scored, 2,
            "the margin == 0 cluster probed and scored its tied doc"
        );
        Ok(())
    }

    /// Forced skips charge exactly the open share: on well-separated
    /// clusters the scan probes one cluster and proves the other five
    /// useless, and the work charge equals
    /// `probed*x + skipped*x + scored*(1 - x)/n_avg` — free skips
    /// would break this identity.
    #[test]
    fn skips_charge_open_share() -> crate::Result<()> {
        let centroids: Vec<[f32; 2]> = vec![
            [1.0, 1.0],
            [11.0, 1.0],
            [21.0, 1.0],
            [1.0, 11.0],
            [11.0, 11.0],
            [21.0, 11.0],
        ];
        let docs: Vec<[f32; 2]> = (0..36)
            .map(|i| {
                let c = centroids[i / 6];
                let off = (i % 6) as f32 * 0.01;
                [c[0] + off, c[1] + off]
            })
            .collect();
        let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs)?;
        let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;

        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        assert_eq!(
            stats.clusters_probed(),
            1,
            "only the home cluster survives the margins: {stats:?}"
        );
        assert_eq!(stats.candidates_scored, 6);
        // All six ranked clusters were pulled (Exhausted); five were
        // passed over by the gate.
        let skipped = 6 - stats.clusters_probed();
        let n_avg = 36.0f64 / 6.0;
        let x = open_share(n_avg);
        let row = (1.0 - x) / n_avg;
        let expected =
            (stats.clusters_probed() + skipped) as f64 * x + stats.candidates_scored as f64 * row;
        assert!(
            (stats.work_charged as f64 - expected).abs() < 1e-5,
            "skips must charge the open share: expected {expected}, got {}",
            stats.work_charged
        );
        Ok(())
    }

    /// With k unreachable the heap never fills, the bound never arms,
    /// and NOTHING is skipped — every cluster probes and the work
    /// identity has no skip term.
    #[test]
    fn unarmed_never_skips() -> crate::Result<()> {
        let centroids: Vec<[f32; 2]> = vec![[1.0, 1.0], [11.0, 1.0], [21.0, 1.0]];
        let docs: Vec<[f32; 2]> = (0..9)
            .map(|i| {
                let c = centroids[i / 3];
                let off = (i % 3) as f32 * 0.01;
                [c[0] + off, c[1] + off]
            })
            .collect();
        let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs)?;
        // k = 100 > 9 docs: the heap can never hold k results.
        let (hits, stats) = run_top_n(&index, field, vec![0.2, 0.3], 100, exhaustive_params(3))?;
        assert_eq!(hits.len(), 9, "every doc is a hit at k > N");
        assert_eq!(
            stats.clusters_probed(),
            3,
            "unarmed, every cluster probes: {stats:?}"
        );
        assert_eq!(stats.candidates_scored, 9);
        let n_avg = 3.0f64;
        let x = open_share(n_avg);
        let row = (1.0 - x) / n_avg;
        let expected = 3.0 * x + 9.0 * row;
        assert!(
            (stats.work_charged as f64 - expected).abs() < 1e-5,
            "no skip term in the identity: expected {expected}, got {}",
            stats.work_charged
        );
        Ok(())
    }

    /// `t` tracks kth improvements in bound space per metric, and an
    /// unchanged kth leaves the bound untouched. The bound-space
    /// conversion runs inside the improvement branch only, so the
    /// cosine sqrt is paid per improvement — asserted here by value
    /// (the cached and recomputed paths are indistinguishable by
    /// construction when the key is unchanged).
    #[test]
    fn t_maintenance_per_metric() {
        // (metric, first kth key, expected t, improved key, expected t)
        let cases = [
            // L2 keys -d^2: d = 2, then d = 1.
            (Metric::L2, -4.0f32, 2.0f32, -1.0f32, 1.0f32),
            // Cosine keys cos: chord sqrt(2*(1-cos)).
            (Metric::Cosine, 0.5, 1.0, 0.875, 0.5),
            // Dot keys the score; identity.
            (Metric::Dot, 3.0, 3.0, 4.5, 4.5),
        ];
        for (metric, first, t_first, improved, t_improved) in cases {
            let mut tracker = QueryBoundTracker::new();
            assert_eq!(tracker.bound(), QueryBound::Filling);
            tracker.observe(metric, HeapPeek::Filling, 0);
            assert_eq!(
                tracker.bound(),
                QueryBound::Filling,
                "{metric:?}: no arm on Filling"
            );

            tracker.observe(metric, HeapPeek::Full { kth_key: first }, 1);
            assert_eq!(
                tracker.bound(),
                QueryBound::Armed { t: t_first },
                "{metric:?}: t from first kth"
            );
            // Unchanged kth: bound bit-identical.
            tracker.observe(metric, HeapPeek::Full { kth_key: first }, 2);
            assert_eq!(tracker.bound(), QueryBound::Armed { t: t_first });

            tracker.observe(metric, HeapPeek::Full { kth_key: improved }, 3);
            assert_eq!(
                tracker.bound(),
                QueryBound::Armed { t: t_improved },
                "{metric:?}: t tracks the improvement"
            );
        }
    }

    /// The armed index is the first probe at which the heap held k
    /// results, and never moves after.
    #[test]
    fn armed_index_recorded() {
        let mut tracker = QueryBoundTracker::new();
        tracker.observe(Metric::L2, HeapPeek::Filling, 0);
        tracker.observe(Metric::L2, HeapPeek::Filling, 1);
        assert_eq!(tracker.armed_at_probe(), None, "unarmed while filling");
        tracker.observe(Metric::L2, HeapPeek::Full { kth_key: -1.0 }, 2);
        assert_eq!(tracker.armed_at_probe(), Some(2), "arms at the first Full");
        tracker.observe(Metric::L2, HeapPeek::Full { kth_key: -0.5 }, 3);
        assert_eq!(
            tracker.armed_at_probe(),
            Some(2),
            "later improvements never move the armed index"
        );
    }
}

// ==================================================================
// P6: probe-stats telemetry
// ==================================================================
mod bounds_stats_tests {
    use super::bounds_gate_tests::single_segment_fixture;
    use super::*;

    /// The six-centroid separated fixture: probing the home cluster
    /// arms the bound and the other five clusters are provably
    /// useless.
    fn separated_fixture(metric: Metric) -> crate::Result<(Index, Field)> {
        let centroids: Vec<[f32; 2]> = vec![
            [1.0, 1.0],
            [11.0, 1.0],
            [21.0, 1.0],
            [1.0, 11.0],
            [11.0, 11.0],
            [21.0, 11.0],
        ];
        let docs: Vec<[f32; 2]> = (0..36)
            .map(|i| {
                let c = centroids[i / 6];
                let off = (i % 6) as f32 * 0.01;
                [c[0] + off, c[1] + off]
            })
            .collect();
        single_segment_fixture(metric, &centroids, &docs)
    }

    /// `bounds_skips` counts exactly the clusters the gate passed
    /// over: all ranked clusters minus the probed ones on an
    /// exhausted stream.
    #[test]
    fn skip_count_matches() -> crate::Result<()> {
        let (index, field) = separated_fixture(Metric::L2)?;
        let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        assert_eq!(stats.clusters_probed(), 1, "{stats:?}");
        assert_eq!(
            stats.bounds_skips, 5,
            "every non-home cluster is a counted skip: {stats:?}"
        );
        Ok(())
    }

    /// Unarmed (k > N): zero skips, and the armed index serializes
    /// as JSON null, not 0 - the harness's armed-share column
    /// depends on the null contract.
    #[test]
    fn armed_null_when_unarmed() -> crate::Result<()> {
        let (index, field) = separated_fixture(Metric::L2)?;
        let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 100, exhaustive_params(6))?;
        assert_eq!(stats.bounds_skips, 0);
        let value = serde_json::to_value(&stats).expect("ProbeStats serializes");
        assert_eq!(value["bound_armed_count"], 0);
        assert_eq!(value["bound_armed_probe_sum"], 0);
        Ok(())
    }

    /// The armed index is the tracker's recorded value: the heap
    /// fills inside the first probed cluster at k <= its size
    /// (index 0), and spans into the second at larger k (index 1).
    #[test]
    fn armed_index_value() -> crate::Result<()> {
        let (index, field) = separated_fixture(Metric::L2)?;
        let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;
        assert_eq!(stats.bound_armed_count, 1, "{stats:?}");
        assert_eq!(stats.bound_armed_probe_sum, 0, "{stats:?}");
        let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 10, exhaustive_params(6))?;
        assert_eq!(stats.bound_armed_count, 1, "{stats:?}");
        assert_eq!(
            stats.bound_armed_probe_sum, 1,
            "k = 10 needs the second probed cluster: {stats:?}"
        );
        assert!(
            stats.bounds_skips > 0,
            "armed late still skips the far tail"
        );
        Ok(())
    }
}
