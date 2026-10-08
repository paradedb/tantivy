//! Global versus per-segment search over identical shared-centroid indexes.
//! Run with `cargo bench --features superkmeans/accelerate --bench global_router`.
//! VECTOR_BENCH_{DOCS,DIM,CLUSTERS,QUERIES,ROUNDS} control the workload.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use common::HasLen;
use tantivy::collector::Collector;
use tantivy::directory::Directory;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::{AllQuery, EnableScoring, Query, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, Metric, Schema, VectorOptions, FAST, INDEXED};
use tantivy::vector::ivf::AdaptiveProbeParams;
use tantivy::vector::{
    CentroidProducer, IvfCentroids, IvfMatrix, PreparedQuery, RouterKind,
    TopDocsByVectorSimilarity, VectorQuantizationConfig, VectorQuantizationLayer,
    VectorSimilarityFruit,
};
use tantivy::{DocAddress, Index, IndexSettings, Searcher, TantivyDocument, Term};

const TOP_K: usize = 10;

struct Centroids(IvfMatrix<f32>);

impl CentroidProducer for Centroids {
    fn centroids(&self, _: Field, _: &VectorOptions) -> tantivy::Result<IvfCentroids> {
        Ok(IvfCentroids::F32(self.0.clone()))
    }
}

fn search(
    searcher: &Searcher,
    field: Field,
    query: &[f32],
    filter: &dyn Query,
    global: bool,
) -> tantivy::Result<VectorSimilarityFruit> {
    let collector = TopDocsByVectorSimilarity::new(field, query.to_vec(), TOP_K)
        .with_adaptive_params(AdaptiveProbeParams {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            router_recall_target: 1.0,
            recall_target: 1.0,
            ..Default::default()
        });
    if global {
        return searcher.search(filter, &collector);
    }
    let weight = filter.weight(EnableScoring::disabled_from_searcher(searcher))?;
    collector.check_schema(searcher.schema())?;
    let fruits = searcher
        .segment_readers()
        .iter()
        .enumerate()
        .map(|(ordinal, reader)| collector.collect_segment(weight.as_ref(), ordinal as u32, reader))
        .collect::<tantivy::Result<_>>()?;
    collector.merge_fruits(fruits)
}

fn oracle(
    searcher: &Searcher,
    field: Field,
    queries: &[Vec<f32>],
    metric: Metric,
    filtered: bool,
) -> tantivy::Result<Vec<Vec<DocAddress>>> {
    let mut rows = Vec::new();
    for (ordinal, reader) in searcher.segment_readers().iter().enumerate() {
        let kept = reader.fast_fields().u64("keep")?;
        let vectors = reader.vector_index(field)?;
        for doc in 0..reader.max_doc() {
            if !filtered || kept.first(doc) == Some(1) {
                rows.push((
                    DocAddress::new(ordinal as u32, doc),
                    vectors.vector_bytes(doc)?.unwrap(),
                ));
            }
        }
    }
    Ok(queries
        .iter()
        .map(|query| {
            let prepared = PreparedQuery::new(metric, Arc::new(query.clone()));
            let mut ranked: Vec<_> = rows
                .iter()
                .map(|(address, bytes)| (prepared.score_doc_bytes(bytes), *address))
                .collect();
            ranked.sort_unstable_by(|(a, da), (b, db)| b.total_cmp(a).then(da.cmp(db)));
            ranked
                .into_iter()
                .take(TOP_K)
                .map(|(_, address)| address)
                .collect()
        })
        .collect())
}

fn main() -> tantivy::Result<()> {
    let read_size = |name, default| {
        std::env::var(name)
            .map(|value| value.parse::<usize>().expect("positive integer"))
            .unwrap_or(default)
    };
    let docs = read_size("VECTOR_BENCH_DOCS", 16_384);
    let dim = read_size("VECTOR_BENCH_DIM", 64);
    let clusters = read_size("VECTOR_BENCH_CLUSTERS", 128);
    let num_queries = read_size("VECTOR_BENCH_QUERIES", 16);
    let rounds = read_size("VECTOR_BENCH_ROUNDS", 8);
    assert!(dim >= 64 && clusters > 0);
    assert!(docs >= 32 && docs % 32 == 0 && num_queries > 0 && rounds > 0);
    let mut rng = fastrand::Rng::with_seed(0x51_a7_09);
    let centroids: Vec<f32> = (0..clusters * dim).map(|_| rng.f32() * 8.0 - 4.0).collect();
    let vectors: Vec<Vec<f32>> = (0..docs)
        .map(|doc| {
            centroids[(doc % clusters) * dim..][..dim]
                .iter()
                .map(|center| center + (rng.f32() - 0.5) * 0.5)
                .collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|query| {
            centroids[(query * 17 % clusters) * dim..][..dim]
                .iter()
                .map(|center| center + (rng.f32() - 0.5) * 0.5)
                .collect()
        })
        .collect();

    for metric in [Metric::L2, Metric::Cosine] {
        let options = VectorOptions::new(dim, metric);
        let quantization = VectorQuantizationConfig::materialize(
            "embedding".into(),
            &options,
            vec![
                VectorQuantizationLayer { bits: 1, seed: 17 },
                VectorQuantizationLayer { bits: 4, seed: 31 },
            ],
        )?;
        for router in [RouterKind::Exact, RouterKind::Stacked] {
            for quantized in [false, true] {
                for segments in [1, 8, 32] {
                    let mut schema = Schema::builder();
                    let field = schema.add_vector_field("embedding", options.clone());
                    let id = schema.add_u64_field("id", FAST);
                    let keep = schema.add_u64_field("keep", INDEXED | FAST);
                    let started = Instant::now();
                    let index = Index::builder()
                        .schema(schema.build())
                        .settings(IndexSettings {
                            vector_quantization: if quantized {
                                vec![quantization.clone()]
                            } else {
                                vec![]
                            },
                            ..Default::default()
                        })
                        .centroid_producer(Arc::new(Centroids(IvfMatrix {
                            values: centroids.clone(),
                            rows: clusters,
                            dims: dim,
                        })))
                        .ivf_router(router)?
                        .create_in_ram()?;
                    let mut writer = index.writer_with_num_threads(1, 50_000_000)?;
                    writer.set_merge_policy(Box::new(NoMergePolicy));
                    for (doc_id, vector) in vectors.iter().enumerate() {
                        let mut doc = TantivyDocument::new();
                        doc.add_vector(field, vector);
                        doc.add_u64(id, doc_id as u64);
                        doc.add_u64(keep, u64::from((doc_id + doc_id / clusters) % 10 == 0));
                        writer.add_document(doc)?;
                        if (doc_id + 1) % (docs / segments) == 0 {
                            writer.commit()?;
                        }
                    }
                    let build_ms = started.elapsed().as_secs_f64() * 1000.0;
                    let searcher = index.reader()?.searcher();
                    assert_eq!(searcher.segment_readers().len(), segments);
                    let artifact = index.load_metas()?.centroid_index.unwrap();
                    let artifact_bytes = index.directory().open_read(&artifact.file_name)?.len();
                    for filtered in [false, true] {
                        let filter: Box<dyn Query> = if filtered {
                            Box::new(TermQuery::new(
                                Term::from_field_u64(keep, 1),
                                IndexRecordOption::Basic,
                            ))
                        } else {
                            Box::new(AllQuery)
                        };
                        let expected = oracle(&searcher, field, &queries, metric, filtered)?;
                        let mut samples = [Vec::new(), Vec::new()];
                        let mut recall = [0.0; 2];
                        let mut routes = [0_usize; 2];
                        let mut scored = [0_usize; 2];
                        for round in 0..=rounds {
                            for (q, query) in queries.iter().enumerate() {
                                for driver in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                                    let started = Instant::now();
                                    let fruit = black_box(search(
                                        &searcher,
                                        field,
                                        query,
                                        filter.as_ref(),
                                        driver == 0,
                                    )?);
                                    let micros = started.elapsed().as_secs_f64() * 1e6;
                                    if round == 0 {
                                        recall[driver] += fruit
                                            .results
                                            .iter()
                                            .filter(|(_, doc)| expected[q].contains(doc))
                                            .count()
                                            as f64
                                            / TOP_K as f64;
                                        routes[driver] += fruit
                                            .stats
                                            .iter()
                                            .filter(|stats| stats.routing.is_some())
                                            .count();
                                        scored[driver] += fruit
                                            .stats
                                            .iter()
                                            .map(|stats| stats.candidates_scored)
                                            .sum::<usize>();
                                    } else {
                                        samples[driver].push(micros);
                                    }
                                }
                            }
                        }
                        for driver in 0..2 {
                            samples[driver].sort_by(f64::total_cmp);
                            println!(
                                "{}",
                                serde_json::json!({
                                    "metric": format!("{metric:?}"), "router": router.to_string(),
                                    "quantized": quantized, "segments": segments, "docs": docs, "dim": dim, "clusters": clusters,
                                    "filtered": filtered, "driver": if driver == 0 { "global" } else { "per_segment" },
                                    "samples": samples[driver].len(),
                                    "median_us": samples[driver][samples[driver].len() / 2],
                                    "p95_us": samples[driver][samples[driver].len() * 95 / 100],
                                    "recall_at_10": recall[driver] / num_queries as f64,
                                    "routes_per_query": routes[driver] as f64 / num_queries as f64,
                                    "scored_per_query": scored[driver] as f64 / num_queries as f64,
                                    "build_ms": build_ms, "centroid_artifact_bytes": artifact_bytes,
                                })
                            );
                        }
                    }
                    let started = Instant::now();
                    writer.merge(&index.searchable_segment_ids()?).wait()?;
                    println!(
                        "{}",
                        serde_json::json!({
                            "metric": format!("{metric:?}"), "router": router.to_string(),
                            "quantized": quantized, "segments": segments, "docs": docs, "dim": dim, "clusters": clusters,
                            "merge_ms": started.elapsed().as_secs_f64() * 1000.0,
                        })
                    );
                }
            }
        }
    }
    Ok(())
}
