use std::sync::Arc;

use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::Candidate;
use crate::{DocId, Result};

const FILTER_BATCH_BYTES: usize = 4 * 1024 * 1024;
// Candidate keys, cluster scratch, match lists, and a transient DocIds read.
const FILTER_BATCH_DOCS: usize =
    FILTER_BATCH_BYTES / (size_of::<(u32, DocId)>() + 3 * size_of::<DocId>());

pub(crate) struct FilterBatch {
    docs: Vec<(u32, DocId)>,
    scratch: Vec<DocId>,
    oversized: Option<usize>,
}

impl FilterBatch {
    pub(crate) fn new() -> Self {
        Self {
            docs: Vec::with_capacity(FILTER_BATCH_DOCS),
            scratch: Vec::with_capacity(FILTER_BATCH_DOCS),
            oversized: None,
        }
    }

    pub(crate) fn fill(
        &mut self,
        candidates: impl Iterator<Item = (usize, Candidate)>,
        vectors: &[Arc<VectorIndexReader>],
        mut on_segment: impl FnMut(usize),
    ) -> Result<usize> {
        self.docs.clear();
        self.oversized = None;
        let mut clusters = 0;
        for (segment, candidate) in candidates {
            let reader = &vectors[segment];
            let rows = reader
                .index()
                .unwrap()
                .cluster_num_rows(candidate.node as usize)?;
            if rows > self.docs.capacity() - self.docs.len() {
                if clusters == 0 {
                    self.oversized = Some(segment);
                    clusters = 1;
                }
                break;
            }
            reader.read_doc_ids(candidate.node as usize, &mut self.scratch)?;
            on_segment(segment);
            self.docs
                .extend(self.scratch.iter().map(|&doc| (segment as u32, doc)));
            clusters += 1;
        }
        self.docs.sort_unstable();
        self.docs.dedup();
        Ok(clusters)
    }

    pub(crate) fn docs(&self, segment: usize) -> Option<&[(u32, DocId)]> {
        if self.oversized == Some(segment) {
            return None;
        }
        let start = self.docs.partition_point(|&(ord, _)| ord < segment as u32);
        let end = self.docs.partition_point(|&(ord, _)| ord <= segment as u32);
        Some(&self.docs[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;
    use crate::vector::test_support::FixtureSpec;

    #[test]
    fn batches_cover_clusters_and_bound_buffers() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 8, 8, 128)
            .replicated()
            .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let vectors = [searcher.segment_reader(0).vector_index(fixture.field)?];
        let ranked = fixture.candidates(&(0..8).collect::<Vec<_>>(), &[0.0; 8])?;
        let largest = (0..8)
            .map(|cluster| {
                vectors[0]
                    .index()
                    .unwrap()
                    .cluster_num_rows(cluster)
                    .unwrap()
            })
            .max()
            .unwrap();
        for capacity in [1, largest * 2] {
            let mut batch = FilterBatch {
                docs: Vec::with_capacity(capacity),
                scratch: Vec::with_capacity(capacity),
                oversized: None,
            };
            let mut cursor = 0;
            let mut batches = 0;
            while cursor < ranked.len() {
                let count =
                    batch.fill(ranked[cursor..].iter().map(|&c| (0, c)), &vectors, |_| {})?;
                assert!(count > 0);
                assert!(batch.docs.capacity() <= capacity);
                assert!(batch.scratch.capacity() <= capacity);
                if capacity == 1 {
                    assert!(batch.docs(0).is_none());
                } else {
                    assert!(batch
                        .docs(0)
                        .unwrap()
                        .windows(2)
                        .all(|pair| pair[0] < pair[1]));
                    let mut expected = Vec::new();
                    let mut scratch = Vec::new();
                    for c in &ranked[cursor..cursor + count] {
                        vectors[0].read_doc_ids(c.node as usize, &mut scratch)?;
                        expected.extend(scratch.iter().map(|&doc| (0, doc)));
                    }
                    expected.sort_unstable();
                    expected.dedup();
                    assert_eq!(batch.docs(0).unwrap(), expected);
                }
                cursor += count;
                batches += 1;
            }
            assert!(batches > 1);
        }
        Ok(())
    }
}
