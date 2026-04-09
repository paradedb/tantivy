use crate::index::SegmentReader;
use crate::indexer::doc_id_mapping::SegmentDocIdMapping;
use crate::schema::Field;
use crate::DocId;

/// Provides full-precision vectors for a batch of document IDs.
///
/// Tantivy only stores quantized (BQ) vectors. Clustering requires
/// full-precision vectors for centroid computation via k-means. The
/// downstream consumer (e.g. pg_search) implements this trait to fetch
/// vectors from its external store (e.g. Postgres heap).
///
/// Doc IDs are post-merge IDs. The implementor is responsible for
/// mapping these back to whatever external storage it uses.
pub trait VectorSampler: Send + Sync {
    fn sample_vectors(
        &self,
        field: Field,
        doc_ids: &[DocId],
    ) -> crate::Result<Vec<Option<Vec<f32>>>>;

    fn dims(&self, field: Field) -> usize;
}

/// Factory that creates a [`VectorSampler`] scoped to a specific merge.
///
/// The factory receives the source segment readers and doc ID mapping so it
/// can set up internal state for resolving post-merge doc IDs back to the
/// consumer's external vector store.
pub trait VectorSamplerFactory: Send + Sync {
    fn create_sampler(
        &self,
        readers: &[SegmentReader],
        doc_id_mapping: &SegmentDocIdMapping,
    ) -> crate::Result<Box<dyn VectorSampler>>;
}

#[cfg(test)]
pub(crate) mod test_utils {
    use std::sync::{Arc, Mutex};

    use super::*;

    pub struct InMemorySampler {
        vectors: Vec<Vec<f32>>,
    }

    impl VectorSampler for InMemorySampler {
        fn sample_vectors(
            &self,
            _field: Field,
            doc_ids: &[DocId],
        ) -> crate::Result<Vec<Option<Vec<f32>>>> {
            Ok(doc_ids
                .iter()
                .map(|&id| self.vectors.get(id as usize).cloned())
                .collect())
        }

        fn dims(&self, _field: Field) -> usize {
            self.vectors.first().map_or(0, |v| v.len())
        }
    }

    pub struct InMemorySamplerFactory {
        pub vectors: Arc<Mutex<Vec<Vec<f32>>>>,
    }

    impl VectorSamplerFactory for InMemorySamplerFactory {
        fn create_sampler(
            &self,
            _readers: &[SegmentReader],
            _doc_id_mapping: &SegmentDocIdMapping,
        ) -> crate::Result<Box<dyn VectorSampler>> {
            let vecs = self.vectors.lock().unwrap().clone();
            Ok(Box::new(InMemorySampler { vectors: vecs }))
        }
    }
}
