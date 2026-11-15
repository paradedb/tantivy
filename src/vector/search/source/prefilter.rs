use std::sync::Arc;

use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::Candidate;
use crate::{DocId, Result};

pub(crate) const FILTER_BATCH_MAX_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const FILTER_BATCH_TARGET_DOC_BYTES: usize = FILTER_BATCH_MAX_BYTES;

#[derive(Debug, PartialEq)]
pub(crate) struct FilterBatch {
    pub(crate) clusters: usize,
    pub(crate) scratch_rows: usize,
}

pub(crate) fn plan_filter_batch(
    candidates: impl Iterator<Item = (usize, Candidate)>,
    vectors: &[Arc<VectorIndexReader>],
    target_doc_bytes: usize,
    max_bytes: usize,
) -> Result<Option<FilterBatch>> {
    let mut rows = 0usize;
    let mut batch = FilterBatch {
        clusters: 0,
        scratch_rows: 0,
    };
    for (segment, candidate) in candidates {
        let cluster_rows = vectors[segment]
            .index()
            .unwrap()
            .cluster_num_rows(candidate.node as usize)?;
        let next_rows = rows.saturating_add(cluster_rows);
        let largest_cluster = batch.scratch_rows.max(cluster_rows);
        let matches_bytes = next_rows.saturating_mul(size_of::<DocId>());
        let scratch_bytes = largest_cluster.saturating_mul(2 * size_of::<DocId>());
        if matches_bytes.saturating_add(scratch_bytes) > max_bytes {
            return Ok((batch.clusters > 0).then_some(batch));
        }
        if batch.clusters > 0 && matches_bytes > target_doc_bytes {
            break;
        }
        rows = next_rows;
        batch.clusters += 1;
        batch.scratch_rows = largest_cluster;
    }
    Ok(Some(batch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;
    use crate::vector::test_support::FixtureSpec;

    #[test]
    fn filter_batches_respect_target_and_buffer_limit() -> Result<()> {
        let fixture = FixtureSpec::clustered(Metric::L2, 8, 8, 128)
            .replicated()
            .build()?;
        let searcher = fixture.index.reader()?.searcher();
        let segment = searcher.segment_reader(0);
        let vectors = [segment.vector_index(fixture.field)?];
        let ranked = fixture.candidates(&(0..8).collect::<Vec<_>>(), &[0.0; 8])?;
        let sizes = (0..8)
            .map(|cluster| vectors[0].index().unwrap().cluster_num_rows(cluster))
            .collect::<Result<Vec<_>>>()?;
        let largest = *sizes.iter().max().unwrap();
        let required =
            sizes.iter().sum::<usize>() * size_of::<DocId>() + largest * 2 * size_of::<DocId>();
        let plan = |candidates: &[Candidate], target, budget| {
            plan_filter_batch(
                candidates.iter().map(|&candidate| (0, candidate)),
                &vectors,
                target,
                budget,
            )
        };
        assert_eq!(
            plan(&ranked, required, required)?,
            Some(FilterBatch {
                clusters: ranked.len(),
                scratch_rows: largest,
            })
        );
        let partial = plan(&ranked, required, required - 1)?.unwrap();
        assert!(partial.clusters > 0 && partial.clusters < ranked.len());
        for candidate in &ranked {
            let rows = sizes[candidate.node as usize];
            let required = rows * 3 * size_of::<DocId>();
            assert_eq!(
                plan(std::slice::from_ref(candidate), 1, required)?,
                Some(FilterBatch {
                    clusters: 1,
                    scratch_rows: rows,
                })
            );
            assert_eq!(
                plan(std::slice::from_ref(candidate), 1, required - 1)?,
                None
            );
        }
        let target = largest * size_of::<DocId>();
        let mut pending = ranked.as_slice();
        let mut batches = 0;
        while !pending.is_empty() {
            let batch = plan(pending, target, required)?.unwrap();
            let rows: usize = pending[..batch.clusters]
                .iter()
                .map(|candidate| sizes[candidate.node as usize])
                .sum();
            assert!(rows * size_of::<DocId>() <= target);
            pending = &pending[batch.clusters..];
            batches += 1;
        }
        assert!(batches > 1);
        Ok(())
    }
}
