use std::sync::Arc;

use super::{ASSIGNMENTS_EXT, IVFVEC_EXT};
use crate::index::{SegmentComponent, SegmentId};
use crate::indexer::NoMergePolicy;
use crate::schema::{Schema, STORED, TEXT};
use crate::vector::meta::VECMETA_EXT;
use crate::vector::{
    IvfCentroids, IvfClusterer, IvfVector, IvfVectors, Metric, VectorColumn, VectorColumnReader,
    VectorOptions, VECTOR_PLUGIN_NAME,
};
use crate::{Index, IndexSettings, IndexWriter, TantivyDocument};

struct TestClusterer;

impl IvfClusterer for TestClusterer {
    fn centroid_ratio(&self) -> f32 {
        0.5
    }

    fn training_samples_per_centroid(&self) -> usize {
        2
    }

    fn train(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        num_centroids: usize,
    ) -> crate::Result<IvfCentroids> {
        assert_eq!(options.dim(), 2);
        assert_eq!(num_centroids, 2);
        match vectors {
            IvfVectors::F32(vectors) => assert!(!vectors.is_empty()),
        }
        Ok(IvfCentroids::F32(vec![vec![0.0, 0.0], vec![10.0, 10.0]]))
    }

    fn assign(
        &self,
        options: &VectorOptions,
        vector: IvfVector,
        centroids: &IvfCentroids,
    ) -> crate::Result<u32> {
        assert_eq!(options.dim(), 2);
        match centroids {
            IvfCentroids::F32(centroids) => assert_eq!(centroids.len(), 2),
        }
        let IvfVector::F32(vector) = vector;
        Ok(if vector.vector[0] < 5.0 { 0 } else { 1 })
    }
}

#[test]
fn test_merge_ivf_writes_meta_assignments_and_vec() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT | STORED);
    let vec_field = schema_builder.add_vector_field("embedding", VectorOptions::new(2, Metric::L2));
    let schema = schema_builder.build();
    let index = Index::builder()
        .schema(schema)
        .settings(IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        })
        .ivf_clusterer(Arc::new(TestClusterer))
        .create_in_ram()?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));

    let mut doc = TantivyDocument::new();
    doc.add_text(text_field, "low");
    doc.add_vector(vec_field, &[1.0f32, 1.0]);
    writer.add_document(doc)?;
    let mut doc = TantivyDocument::new();
    doc.add_text(text_field, "missing");
    writer.add_document(doc)?;
    writer.commit()?;

    let mut doc = TantivyDocument::new();
    doc.add_text(text_field, "high");
    doc.add_vector(vec_field, &[10.0f32, 10.0]);
    writer.add_document(doc)?;
    let mut doc = TantivyDocument::new();
    doc.add_text(text_field, "low");
    doc.add_vector(vec_field, &[2.0f32, 2.0]);
    writer.add_document(doc)?;
    writer.commit()?;

    let segment_ids: Vec<SegmentId> = index.searchable_segment_ids()?.into_iter().collect();
    assert_eq!(segment_ids.len(), 2);
    writer.merge(&segment_ids).wait()?;
    writer.wait_merging_threads()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let segments = searcher.segment_readers();
    assert_eq!(segments.len(), 1);
    let segment = &segments[0];
    assert!(
        segment
            .open_read(SegmentComponent::Custom(VECMETA_EXT.to_string()))
            .is_ok()
    );
    assert!(
        segment
            .open_read(SegmentComponent::Custom(ASSIGNMENTS_EXT.to_string()))
            .is_ok()
    );
    assert!(
        segment
            .open_read(SegmentComponent::Custom(IVFVEC_EXT.to_string()))
            .is_ok()
    );

    let vec_reader: Arc<crate::vector::VectorReader> = segment
        .plugin_reader::<crate::vector::VectorReader>(VECTOR_PLUGIN_NAME)?
        .expect("plugin reader");
    let column = vec_reader.open_column(vec_field)?;
    let VectorColumn::Ivf(column) = column else {
        panic!("expected IVF vector column");
    };

    assert_eq!(column.len(), 3);
    let low_docs = column.cluster_doc_ids(0)?.expect("low cluster");
    let high_docs = column.cluster_doc_ids(1)?.expect("high cluster");
    assert_eq!(low_docs.len(), 2);
    assert_eq!(high_docs.len(), 1);
    assert_eq!(column.centroid_bytes().len(), 16);
    assert_eq!(column.cluster_vector_bytes(0)?.len(), low_docs.len() * 8);
    assert_eq!(column.cluster_vector_bytes(1)?.len(), high_docs.len() * 8);
    let missing_doc = (0..segment.max_doc())
        .find(|&doc_id| !column.contains(doc_id))
        .expect("missing vector doc");
    assert!(column.vector_bytes_at(missing_doc).is_none());

    let decode = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    for &doc_id in low_docs {
        assert!(decode(column.vector_bytes_at(doc_id).expect("low vector"))[0] < 5.0);
    }
    for &doc_id in high_docs {
        assert!(decode(column.vector_bytes_at(doc_id).expect("high vector"))[0] >= 5.0);
    }
    Ok(())
}

#[cfg(feature = "mmap")]
#[test]
fn test_ivf_assignments_and_vec_components_are_lazy() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT | STORED);
    let vec_field = schema_builder.add_vector_field("embedding", VectorOptions::new(2, Metric::L2));
    let schema = schema_builder.build();
    let directory = crate::directory::MmapDirectory::create_from_tempdir()?;
    let directory_cache = directory.clone();
    let index = Index::builder()
        .schema(schema)
        .settings(IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        })
        .ivf_clusterer(Arc::new(TestClusterer))
        .open_or_create(directory)?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));

    let mut doc = TantivyDocument::new();
    doc.add_text(text_field, "low");
    doc.add_vector(vec_field, &[1.0f32, 1.0]);
    writer.add_document(doc)?;
    let mut doc = TantivyDocument::new();
    doc.add_text(text_field, "missing");
    writer.add_document(doc)?;
    writer.commit()?;

    let mut doc = TantivyDocument::new();
    doc.add_text(text_field, "high");
    doc.add_vector(vec_field, &[10.0f32, 10.0]);
    writer.add_document(doc)?;
    let mut doc = TantivyDocument::new();
    doc.add_text(text_field, "low");
    doc.add_vector(vec_field, &[2.0f32, 2.0]);
    writer.add_document(doc)?;
    writer.commit()?;

    let segment_ids: Vec<SegmentId> = index.searchable_segment_ids()?.into_iter().collect();
    writer.merge(&segment_ids).wait()?;
    writer.wait_merging_threads()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let segment = &searcher.segment_readers()[0];
    let vec_reader: Arc<crate::vector::VectorReader> = segment
        .plugin_reader::<crate::vector::VectorReader>(VECTOR_PLUGIN_NAME)?
        .expect("plugin reader");
    let column = vec_reader.open_column(vec_field)?;
    let VectorColumn::Ivf(column) = column else {
        panic!("expected IVF vector column");
    };

    let component_is_mapped = |extension| {
        directory_cache
            .get_cache_info()
            .mmapped
            .iter()
            .any(|path| path.extension().and_then(|ext| ext.to_str()) == Some(extension))
    };

    assert!(!component_is_mapped(ASSIGNMENTS_EXT));
    assert!(!component_is_mapped(IVFVEC_EXT));
    let low_docs = column.cluster_doc_ids(0)?.expect("low cluster");
    assert_eq!(low_docs.len(), 2);
    assert!(component_is_mapped(ASSIGNMENTS_EXT));
    assert!(!component_is_mapped(IVFVEC_EXT));
    assert_eq!(column.cluster_vector_bytes(0)?.len(), low_docs.len() * 8);
    assert!(component_is_mapped(IVFVEC_EXT));

    Ok(())
}
