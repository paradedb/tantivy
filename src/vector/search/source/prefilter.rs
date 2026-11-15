use std::sync::Arc;

use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::Candidate;
use crate::{DocId, Result};

pub(crate) const FILTER_BATCH_BYTES: usize = 4 * 1024 * 1024;

pub(crate) fn filter_batch_scratch(
    candidates: impl Iterator<Item = (usize, Candidate)>,
    vectors: &[Arc<VectorIndexReader>],
    max_doc: DocId,
    max_bytes: usize,
) -> Result<Option<usize>> {
    let bitmap_bytes = (max_doc as usize).div_ceil(u64::BITS as usize) * size_of::<u64>();
    let mut rows = 0usize;
    let mut largest_cluster = 0usize;
    for (segment, candidate) in candidates {
        let cluster_rows = vectors[segment]
            .index()
            .unwrap()
            .cluster_num_rows(candidate.node as usize)?;
        rows = rows.saturating_add(cluster_rows);
        largest_cluster = largest_cluster.max(cluster_rows);
        let matches_bytes = rows.saturating_mul(size_of::<DocId>());
        let scratch_bytes = largest_cluster.saturating_mul(2 * size_of::<DocId>());
        if bitmap_bytes
            .saturating_add(matches_bytes)
            .saturating_add(scratch_bytes)
            > max_bytes
        {
            return Ok(None);
        }
    }
    Ok(Some(largest_cluster))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;
    use crate::vector::test_support::FixtureSpec;

    #[test]
    fn filter_batch_falls_back_at_the_buffer_limit() -> Result<()> {
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
        let required = (segment.max_doc() as usize).div_ceil(u64::BITS as usize) * size_of::<u64>()
            + sizes.iter().sum::<usize>() * size_of::<DocId>()
            + largest * 2 * size_of::<DocId>();
        for (budget, expected) in [(required, Some(largest)), (required - 1, None), (1, None)] {
            assert_eq!(
                filter_batch_scratch(
                    ranked.iter().map(|&candidate| (0, candidate)),
                    &vectors,
                    segment.max_doc(),
                    budget,
                )?,
                expected,
            );
        }
        Ok(())
    }
}
