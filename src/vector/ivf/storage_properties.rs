use super::*;
use crate::vector::storage_io::test_support::PagedDirectory;

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
    let mut schema = Schema::builder();
    let vector = schema.add_vector_field("embedding", VectorOptions::new(DIM, metric));
    let label = schema.add_text_field("label", STRING | STORED);
    let ordinal = schema.add_u64_field("ordinal", STORED);
    let directory = PagedDirectory::default();
    let index = Index::builder()
        .schema(schema.build())
        .settings(IndexSettings {
            vector_clustering_threshold: 1,
            vector_quantization: vec![quant_fixture_config_for(DIM, metric, schedule)],
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
    for doc in [101, 202] {
        writer.delete_term(Term::from_field_text(label, &format!("d{doc}")));
    }
    writer.commit()?;
    writer.wait_merging_threads()?;
    Ok((index, directory))
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
