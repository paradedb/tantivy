use std::sync::{Arc, Mutex};

use super::shared_threshold::{AtomicSharedThreshold, SharedThresholdArcOpt};
use crate::collector::sort_key::NaturalComparator;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer, TopNComputer};
use crate::{DocAddress, DocId, Score, SegmentOrdinal};

thread_local! {
    static GLOBAL_TOPK_THRESHOLD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn set_global_topk_threshold(enabled: bool) {
    GLOBAL_TOPK_THRESHOLD.set(enabled);
}

struct PrefixTopK {
    k: usize,
    heap: TopNComputer<Score, DocAddress, NaturalComparator>,
}

#[derive(Clone)]
pub struct SortBySimilarityScore {
    shared_threshold: SharedThresholdArcOpt<Score>,
    prefix_top_k: Option<Arc<Mutex<Option<PrefixTopK>>>>,
}

impl std::fmt::Debug for SortBySimilarityScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SortBySimilarityScore")
            .field(
                "threshold",
                &self.shared_threshold.as_ref().and_then(|s| s.load()),
            )
            .finish()
    }
}

impl Default for SortBySimilarityScore {
    fn default() -> Self {
        Self::with_shared_threshold(Some(Arc::new(AtomicSharedThreshold::default())))
    }
}

impl SortBySimilarityScore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_shared_threshold(shared_threshold: SharedThresholdArcOpt<Score>) -> Self {
        Self {
            shared_threshold,
            prefix_top_k: GLOBAL_TOPK_THRESHOLD
                .get()
                .then(|| Arc::new(Mutex::new(None))),
        }
    }
}

impl SortKeyComputer for SortBySimilarityScore {
    type SortKey = Score;

    type Child = SortBySimilarityScore;

    type Comparator = NaturalComparator;

    fn requires_scoring(&self) -> bool {
        true
    }

    fn shared_threshold(
        &self,
    ) -> SharedThresholdArcOpt<
        <<Self as SortKeyComputer>::Child as SegmentSortKeyComputer>::SegmentSortKey,
    > {
        self.shared_threshold.clone()
    }

    fn segment_sort_key_computer(
        &self,
        _segment_reader: &crate::SegmentReader,
    ) -> crate::Result<Self::Child> {
        Ok(self.clone())
    }

    fn record_segment_top_k(&self, hits: &[(Score, DocAddress)], k: usize) {
        let (Some(prefix), Some(shared)) = (&self.prefix_top_k, &self.shared_threshold) else {
            return;
        };
        if k == 0 {
            return;
        }
        let mut prefix = prefix.lock().unwrap();
        if prefix.as_ref().is_none_or(|state| state.k != k) {
            *prefix = Some(PrefixTopK {
                k,
                heap: TopNComputer::new_with_comparator(k, NaturalComparator),
            });
        }
        let heap = &mut prefix.as_mut().unwrap().heap;
        for &(score, address) in hits {
            heap.push_unordered(score, address);
        }
        let Some(threshold) = heap.kth_best() else {
            return;
        };
        let mut current = shared.load();
        loop {
            if current.is_some_and(|(score, _)| score >= threshold) {
                return;
            }
            match shared.try_update(&current, (threshold, SegmentOrdinal::MAX)) {
                Ok(()) => return,
                Err(actual) => current = actual,
            }
        }
    }

    fn finish_top_k(&self) {
        if let Some(prefix) = &self.prefix_top_k {
            *prefix.lock().unwrap() = None;
        }
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

    fn supports_bm25_pruning(&self) -> bool {
        true
    }

    fn bm25_pruning_threshold(
        &self,
        threshold: &Score,
        segment_ord: SegmentOrdinal,
        threshold_ord: SegmentOrdinal,
    ) -> Option<Score> {
        if segment_ord < threshold_ord {
            Some(threshold.next_down())
        } else {
            Some(*threshold)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{set_global_topk_threshold, SortBySimilarityScore};
    use crate::collector::sort_key::SortByStaticFastValue;
    use crate::collector::{SortKeyComputer, TopDocs};
    use crate::merge_policy::NoMergePolicy;
    use crate::query::QueryParser;
    use crate::schema::{Schema, FAST, TEXT};
    use crate::{DocAddress, Index, Order, SegmentOrdinal};

    #[test]
    fn test_global_topk_prefix_threshold() {
        set_global_topk_threshold(true);
        let computer = SortBySimilarityScore::new();
        set_global_topk_threshold(false);
        computer.record_segment_top_k(&[], 0);
        assert_eq!(computer.shared_threshold().unwrap().load(), None);
        let first = [100.0, 90.0, 2.0, 1.0]
            .into_iter()
            .enumerate()
            .map(|(doc, score)| (score, DocAddress::new(0, doc as u32)))
            .collect::<Vec<_>>();
        computer.record_segment_top_k(&first, 4);
        assert_eq!(
            computer.shared_threshold().unwrap().load(),
            Some((1.0, SegmentOrdinal::MAX))
        );
        let second = [80.0, 70.0, 0.5, 0.25]
            .into_iter()
            .enumerate()
            .map(|(doc, score)| (score, DocAddress::new(1, doc as u32)))
            .collect::<Vec<_>>();
        computer.clone().record_segment_top_k(&second, 4);
        assert_eq!(
            computer.shared_threshold().unwrap().load(),
            Some((70.0, SegmentOrdinal::MAX))
        );
        computer.finish_top_k();
        assert!(computer
            .prefix_top_k
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .is_none());

        let disabled = SortBySimilarityScore::new();
        disabled.record_segment_top_k(&first, 4);
        assert_eq!(disabled.shared_threshold().unwrap().load(), None);
    }

    #[test]
    fn test_global_topk_prefix_results_ties_offsets_and_secondary_sort() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let text = schema.add_text_field("text", TEXT);
        let rank = schema.add_u64_field("rank", FAST);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for segment in 0..4 {
            for doc in 0..20 {
                let body = match (segment + doc) % 3 {
                    0 => "common alpha beta",
                    1 => "common alpha gamma",
                    _ => "common beta gamma",
                };
                writer.add_document(
                    crate::doc!(text => body, rank => (80 - segment * 20 - doc) as u64),
                )?;
            }
            writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 4);
        let parser = QueryParser::for_index(&index, vec![text]);
        for query_text in ["common", "alpha beta", "alpha gamma"] {
            let query = parser.parse_query(query_text)?;
            for limit in [1, 3, 10] {
                for offset in [0, 2, 7] {
                    set_global_topk_threshold(false);
                    let expected = searcher.search(
                        query.as_ref(),
                        &TopDocs::with_limit(limit)
                            .and_offset(offset)
                            .order_by_score(),
                    )?;
                    set_global_topk_threshold(true);
                    let actual = searcher.search(
                        query.as_ref(),
                        &TopDocs::with_limit(limit)
                            .and_offset(offset)
                            .order_by_score(),
                    )?;
                    assert_eq!(actual, expected);
                }
            }
            let mut secondary_results = Vec::new();
            for enabled in [false, true] {
                set_global_topk_threshold(enabled);
                let computer = (
                    SortBySimilarityScore::new(),
                    (SortByStaticFastValue::<u64>::for_field("rank"), Order::Asc),
                );
                secondary_results.push(searcher.search(
                    query.as_ref(),
                    &TopDocs::with_limit(10).and_offset(3).order_by(computer),
                )?);
            }
            assert_eq!(secondary_results[0], secondary_results[1]);
        }
        set_global_topk_threshold(false);
        Ok(())
    }
}
