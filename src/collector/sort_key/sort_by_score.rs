use std::sync::Arc;

use super::shared_threshold::{AtomicSharedThreshold, SharedThreshold};
use crate::collector::sort_key::NaturalComparator;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer, TopNComputer};
use crate::{DocAddress, DocId, Score};

#[derive(Clone)]
pub struct SortBySimilarityScore {
    shared_threshold: Arc<dyn SharedThreshold<Score>>,
}

impl std::fmt::Debug for SortBySimilarityScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SortBySimilarityScore")
            .field("threshold", &self.shared_threshold.load())
            .finish()
    }
}

impl Default for SortBySimilarityScore {
    fn default() -> Self {
        Self {
            shared_threshold: Arc::new(AtomicSharedThreshold::default()),
        }
    }
}

impl SortBySimilarityScore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_shared_threshold(shared_threshold: Arc<dyn SharedThreshold<Score>>) -> Self {
        Self { shared_threshold }
    }
}

impl SortKeyComputer for SortBySimilarityScore {
    type SortKey = Score;

    type Child = SortBySimilarityScore;

    type Comparator = NaturalComparator;

    fn requires_scoring(&self) -> bool {
        true
    }

    fn shared_threshold(&self) -> Option<Arc<dyn SharedThreshold<<<Self as SortKeyComputer>::Child as SegmentSortKeyComputer>::SegmentSortKey>>> {
        Some(self.shared_threshold.clone())
    }

    fn segment_sort_key_computer(
        &self,
        _segment_reader: &crate::SegmentReader,
    ) -> crate::Result<Self::Child> {
        Ok(self.clone())
    }

    fn collect_segment_top_k(
        &self,
        k: usize,
        weight: &dyn crate::query::Weight,
        reader: &crate::SegmentReader,
        segment_ord: u32,
    ) -> crate::Result<Vec<(Self::SortKey, DocAddress)>> {
        let mut top_n: TopNComputer<Score, DocId, Self::Comparator> =
            TopNComputer::new_with_comparator(k, self.comparator());
        top_n.shared_threshold = self.shared_threshold();
        top_n.segment_ord = segment_ord;

        let initial_threshold = self.shared_threshold.load();
        top_n.threshold = Some(initial_threshold);

        if let Some(alive_bitset) = reader.alive_bitset() {
            weight.for_each_pruning(initial_threshold.0, reader, &mut |doc, score| {
                if alive_bitset.is_deleted(doc) {
                    return top_n.threshold.map(|t| t.0).unwrap_or(Score::MIN);
                }
                top_n.push(score, doc);
                top_n.threshold.map(|t| t.0).unwrap_or(Score::MIN)
            })?;
        } else {
            weight.for_each_pruning(initial_threshold.0, reader, &mut |doc, score| {
                top_n.push(score, doc);
                top_n.threshold.map(|t| t.0).unwrap_or(Score::MIN)
            })?;
        }

        let final_threshold = top_n.threshold.map(|t| t.0).unwrap_or(Score::MIN);
        self.shared_threshold.update(final_threshold, segment_ord);

        Ok(top_n
            .into_vec()
            .into_iter()
            .map(|cid| (cid.sort_key, DocAddress::new(segment_ord, cid.doc)))
            .collect())
    }
}

impl SegmentSortKeyComputer for SortBySimilarityScore {
    type SortKey = Score;
    type SegmentSortKey = Score;
    type SegmentComparator = NaturalComparator;

    #[inline(always)]
    fn segment_sort_key(&mut self, _doc: DocId, score: Score) -> Score {
        score
    }

    fn convert_segment_sort_key(&self, score: Score) -> Score {
        score
    }
}
