use std::sync::Arc;

use super::shared_threshold::{AtomicSharedThreshold, SharedThreshold};
use crate::collector::sort_key::NaturalComparator;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer, TopNComputer};
use crate::{DocAddress, DocId, Score};

const TRUNCATIONS_PER_SHARED_UPDATE: u32 = 2;

#[derive(Clone)]
pub struct SortBySimilarityScore {
    shared_threshold: Arc<dyn SharedThreshold>,
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

    pub fn with_shared_threshold(shared_threshold: Arc<dyn SharedThreshold>) -> Self {
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

        let shared = &self.shared_threshold;
        let initial_threshold = shared.load();
        let mut truncation_count: u32 = 0;

        if let Some(alive_bitset) = reader.alive_bitset() {
            let mut threshold = initial_threshold;
            top_n.threshold = Some(threshold);
            weight.for_each_pruning(threshold, reader, &mut |doc, score| {
                if alive_bitset.is_deleted(doc) {
                    return threshold;
                }
                top_n.push(score, doc);
                let new_threshold = top_n.threshold.unwrap_or(Score::MIN);
                if new_threshold > threshold {
                    threshold = new_threshold;
                    truncation_count += 1;
                    if truncation_count % TRUNCATIONS_PER_SHARED_UPDATE == 0 {
                        shared.update(threshold);
                        let global = shared.load();
                        if global > threshold {
                            threshold = global;
                            top_n.threshold = Some(threshold);
                        }
                    }
                }
                threshold
            })?;
        } else {
            let mut threshold = initial_threshold;
            top_n.threshold = Some(threshold);
            weight.for_each_pruning(threshold, reader, &mut |doc, score| {
                top_n.push(score, doc);
                let new_threshold = top_n.threshold.unwrap_or(Score::MIN);
                if new_threshold > threshold {
                    threshold = new_threshold;
                    truncation_count += 1;
                    if truncation_count % TRUNCATIONS_PER_SHARED_UPDATE == 0 {
                        shared.update(threshold);
                        let global = shared.load();
                        if global > threshold {
                            threshold = global;
                            top_n.threshold = Some(threshold);
                        }
                    }
                }
                threshold
            })?;
        }

        let final_threshold = top_n.threshold.unwrap_or(Score::MIN);
        shared.update(final_threshold);

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
