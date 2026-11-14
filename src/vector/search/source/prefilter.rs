use std::sync::Arc;

use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::Candidate;
use crate::{DocId, Result};

const FILTER_BATCH_BYTES: usize = 4 * 1024 * 1024;
pub(crate) struct FilterBatch {
    docs: Vec<(u32, DocId)>,
    scratch: Vec<DocId>,
    oversized: Option<usize>,
    max_bytes: usize,
}

impl FilterBatch {
    pub(crate) fn new() -> Self {
        Self {
            docs: Vec::new(),
            scratch: Vec::new(),
            oversized: None,
            max_bytes: FILTER_BATCH_BYTES,
        }
    }

    pub(crate) fn fill(
        &mut self,
        candidates: impl Iterator<Item = (usize, Candidate)> + Clone,
        vectors: &[Arc<VectorIndexReader>],
        mut on_segment: impl FnMut(usize),
    ) -> Result<usize> {
        self.docs = Vec::new();
        self.scratch = Vec::new();
        self.oversized = None;
        let mut clusters = 0;
        let mut rows = 0usize;
        let mut largest_cluster = 0usize;
        for (segment, candidate) in candidates.clone() {
            let cluster_rows = vectors[segment]
                .index()
                .unwrap()
                .cluster_num_rows(candidate.node as usize)?;
            let next_rows = rows.saturating_add(cluster_rows);
            let next_largest = largest_cluster.max(cluster_rows);
            let candidate_bytes =
                next_rows.saturating_mul(size_of::<(u32, DocId)>() + size_of::<DocId>());
            let scratch_bytes = next_largest.saturating_mul(2 * size_of::<DocId>());
            if candidate_bytes.saturating_add(scratch_bytes) > self.max_bytes {
                if clusters == 0 {
                    self.oversized = Some(segment);
                    return Ok(1);
                }
                break;
            }
            rows = next_rows;
            largest_cluster = next_largest;
            clusters += 1;
        }
        self.docs.reserve_exact(rows);
        self.scratch.reserve_exact(largest_cluster);
        for (segment, candidate) in candidates.take(clusters) {
            vectors[segment].read_doc_ids(candidate.node as usize, &mut self.scratch)?;
            on_segment(segment);
            self.docs
                .extend(self.scratch.iter().map(|&doc| (segment as u32, doc)));
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
        let cluster_bytes = largest * (size_of::<(u32, DocId)>() + 3 * size_of::<DocId>());
        for max_bytes in [1, cluster_bytes * 2] {
            let mut batch = FilterBatch {
                max_bytes,
                ..FilterBatch::new()
            };
            let mut cursor = 0;
            let mut batches = 0;
            while cursor < ranked.len() {
                let count =
                    batch.fill(ranked[cursor..].iter().map(|&c| (0, c)), &vectors, |_| {})?;
                assert!(count > 0);
                let reserved = batch.docs.capacity()
                    * (size_of::<(u32, DocId)>() + size_of::<DocId>())
                    + batch.scratch.capacity() * 2 * size_of::<DocId>();
                assert!(reserved <= max_bytes);
                if max_bytes == 1 {
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
