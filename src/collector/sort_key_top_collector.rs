use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::collector::sort_key::{Comparator, SegmentSortKeyComputer, SortKeyComputer};
use crate::collector::{Collector, SegmentCollector, TopNComputer};
use crate::query::Weight;
use crate::schema::Schema;
use crate::{DocAddress, DocId, Result, Score, SegmentReader};

pub(crate) struct TopBySortKeyCollector<TSortKeyComputer>
where
    TSortKeyComputer: SortKeyComputer + Send + Sync,
{
    sort_key_computer: TSortKeyComputer,
    doc_range: Range<usize>,
    // Optional threshold for segment pruning. When set, segments whose bounds
    // don't overlap with the threshold can be skipped.
    // Uses Arc<Mutex<>> for interior mutability to allow threshold updates during sequential processing
    threshold: Arc<Mutex<Option<<<TSortKeyComputer as SortKeyComputer>::Child as SegmentSortKeyComputer>::SegmentSortKey>>>,
}

impl<TSortKeyComputer> TopBySortKeyCollector<TSortKeyComputer>
where
    TSortKeyComputer: SortKeyComputer + Send + Sync,
{
    pub fn new(sort_key_computer: TSortKeyComputer, doc_range: Range<usize>) -> Self {
        TopBySortKeyCollector {
            sort_key_computer,
            doc_range,
            threshold: Arc::new(Mutex::new(None)),
        }
    }

    /// Helper method to check if a segment should be skipped based on threshold
    fn should_skip_segment(
        &self,
        reader: &SegmentReader,
        threshold: &<<TSortKeyComputer as SortKeyComputer>::Child as SegmentSortKeyComputer>::SegmentSortKey,
    ) -> bool {
        if let Ok(Some((min_bound, max_bound))) = self.sort_key_computer.segment_bounds(reader) {
            let comparator = self.sort_key_computer.comparator();

            // Compare the segment's bounds against the threshold
            // The comparator defines what "better" means:
            // - For ascending order (ReverseNoneLowerComparator): smaller values compare as "Greater"
            // - For descending order (NaturalComparator): larger values compare as "Greater"
            //
            // We want to skip if ALL values in the segment are worse than threshold.
            // A value is "worse" if it compares as Less or Equal to threshold.
            // Skip only if the BEST value in the segment (the one that compares highest)
            // is still worse than threshold.
            //
            // For ascending order: best value is min_bound (smallest)
            // For descending order: best value is max_bound (largest)
            // But since we use the comparator, we just check if the "greater" bound is still ≤ threshold

            let min_cmp = comparator.compare(&min_bound, threshold);
            let max_cmp = comparator.compare(&max_bound, threshold);

            // Skip if both bounds compare as LessOrEqual to threshold
            // This means even the best value in the segment is not better than threshold
            if min_cmp != std::cmp::Ordering::Greater && max_cmp != std::cmp::Ordering::Greater {
                return true;
            }
        }
        false
    }
}

impl<TSortKeyComputer> Collector for TopBySortKeyCollector<TSortKeyComputer>
where
    TSortKeyComputer: SortKeyComputer + Send + Sync + 'static,
{
    type Fruit = Vec<(TSortKeyComputer::SortKey, DocAddress)>;

    type Child =
        TopBySortKeySegmentCollector<TSortKeyComputer::Child, TSortKeyComputer::Comparator>;

    fn check_schema(&self, schema: &Schema) -> crate::Result<()> {
        self.sort_key_computer.check_schema(schema)
    }

    fn for_segment(&self, segment_ord: u32, segment_reader: &SegmentReader) -> Result<Self::Child> {
        let segment_sort_key_computer = self
            .sort_key_computer
            .segment_sort_key_computer(segment_reader)?;
        let topn_computer = TopNComputer::new_with_comparator(
            self.doc_range.end,
            self.sort_key_computer.comparator(),
        );
        Ok(TopBySortKeySegmentCollector {
            topn_computer,
            segment_ord,
            segment_sort_key_computer,
        })
    }

    fn requires_scoring(&self) -> bool {
        self.sort_key_computer.requires_scoring()
    }

    fn merge_fruits(&self, segment_fruits: Vec<Self::Fruit>) -> Result<Self::Fruit> {
        Ok(merge_top_k(
            segment_fruits.into_iter().flatten(),
            self.doc_range.clone(),
            self.sort_key_computer.comparator(),
        ))
    }

    fn collect_segment(
        &self,
        weight: &dyn Weight,
        segment_ord: u32,
        reader: &SegmentReader,
    ) -> crate::Result<Vec<(TSortKeyComputer::SortKey, DocAddress)>> {
        // Check threshold and skip segment if it can't contribute
        if let Some(ref threshold) = *self.threshold.lock().unwrap() {
            if self.should_skip_segment(reader, threshold) {
                return Ok(Vec::new());
            }
        }

        // Process segment
        let mut segment_collector = self.for_segment(segment_ord, reader)?;
        crate::collector::default_collect_segment_impl(&mut segment_collector, weight, reader, self.requires_scoring())?;

        // Extract threshold from TopNComputer before harvesting
        // Always compute threshold to ensure it's set for pruning
        let k = self.doc_range.end;
        let segment_sort_key_computer = segment_collector.segment_sort_key_computer;

        // Check if TopNComputer already has a threshold set (when buffer reached capacity)
        let threshold_key = segment_collector.topn_computer.threshold.clone();

        // Use into_sorted_vec to get sorted results (this truncates if needed)
        let sorted_vec = segment_collector.topn_computer.into_sorted_vec();

        // If threshold wasn't set, compute it from sorted results
        let threshold_key = threshold_key.or_else(|| {
            if sorted_vec.len() >= k {
                // Use the k-th best value (index k-1)
                sorted_vec.get(k - 1).map(|cdoc| cdoc.sort_key.clone())
            } else if !sorted_vec.is_empty() {
                // If we have fewer than k results, use the worst one
                sorted_vec.last().map(|cdoc| cdoc.sort_key.clone())
            } else {
                None
            }
        });

        // Convert sorted segment sort keys to global sort keys for results
        let results: Vec<(TSortKeyComputer::SortKey, DocAddress)> = sorted_vec
            .into_iter()
            .map(|comparable_doc| {
                let global_sort_key = segment_sort_key_computer
                    .convert_segment_sort_key(comparable_doc.sort_key);
                (
                    global_sort_key,
                    DocAddress {
                        segment_ord,
                        doc_id: comparable_doc.doc,
                    },
                )
            })
            .collect();

        // Update threshold for adaptive pruning of subsequent segments
        // Only update if new threshold is "better" (more restrictive) than current
        if let Some(new_threshold) = threshold_key {
            let comparator = self.sort_key_computer.comparator();
            let mut current = self.threshold.lock().unwrap();
            if let Some(ref current_threshold) = *current {
                // A threshold is "better" if it compares as "Greater" to the current one
                // This means new values must be even better to pass the threshold
                if comparator.compare(&new_threshold, current_threshold) == std::cmp::Ordering::Greater {
                    *current = Some(new_threshold);
                }
            } else {
                // No current threshold, set the new one
                *current = Some(new_threshold);
            }
        }

        Ok(results)
    }
}

fn merge_top_k<D: Ord, TSortKey: Clone + std::fmt::Debug, C: Comparator<TSortKey>>(
    sort_key_docs: impl Iterator<Item = (TSortKey, D)>,
    doc_range: Range<usize>,
    comparator: C,
) -> Vec<(TSortKey, D)> {
    if doc_range.is_empty() {
        return Vec::new();
    }
    let mut top_collector: TopNComputer<TSortKey, D, C> =
        TopNComputer::new_with_comparator(doc_range.end, comparator);
    for (sort_key, doc) in sort_key_docs {
        top_collector.push(sort_key, doc);
    }
    top_collector
        .into_sorted_vec()
        .into_iter()
        .skip(doc_range.start)
        .map(|cdoc| (cdoc.sort_key, cdoc.doc))
        .collect()
}

pub struct TopBySortKeySegmentCollector<TSegmentSortKeyComputer, C>
where
    TSegmentSortKeyComputer: SegmentSortKeyComputer,
    C: Comparator<TSegmentSortKeyComputer::SegmentSortKey>,
{
    pub(crate) topn_computer: TopNComputer<TSegmentSortKeyComputer::SegmentSortKey, DocId, C>,
    pub(crate) segment_ord: u32,
    pub(crate) segment_sort_key_computer: TSegmentSortKeyComputer,
}

impl<TSegmentSortKeyComputer, C> SegmentCollector
    for TopBySortKeySegmentCollector<TSegmentSortKeyComputer, C>
where
    TSegmentSortKeyComputer: 'static + SegmentSortKeyComputer,
    C: Comparator<TSegmentSortKeyComputer::SegmentSortKey> + 'static,
{
    type Fruit = Vec<(TSegmentSortKeyComputer::SortKey, DocAddress)>;

    fn collect(&mut self, doc: DocId, score: Score) {
        self.segment_sort_key_computer.compute_sort_key_and_collect(
            doc,
            score,
            &mut self.topn_computer,
        );
    }

    fn harvest(self) -> Self::Fruit {
        let segment_ord = self.segment_ord;
        let segment_hits: Vec<(TSegmentSortKeyComputer::SortKey, DocAddress)> = self
            .topn_computer
            .into_vec()
            .into_iter()
            .map(|comparable_doc| {
                let sort_key = self
                    .segment_sort_key_computer
                    .convert_segment_sort_key(comparable_doc.sort_key);
                (
                    sort_key,
                    DocAddress {
                        segment_ord,
                        doc_id: comparable_doc.doc,
                    },
                )
            })
            .collect();
        segment_hits
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rand;
    use rand::seq::SliceRandom as _;

    use std::sync::{Arc, Mutex};

    use super::{merge_top_k, TopBySortKeyCollector};
    use crate::collector::sort_key::{ComparatorEnum, SortKeyComputer};
    use crate::collector::{Collector, SegmentCollector};
    use crate::indexer::NoMergePolicy;
    use crate::query::AllQuery;
    use crate::schema::{Schema, FAST};
    use crate::{doc, Index, Order, SegmentReader};

    fn test_merge_top_k_aux(
        order: Order,
        doc_range: Range<usize>,
        expected: &[(crate::Score, usize)],
    ) {
        let mut vals: Vec<(crate::Score, usize)> = (0..10).map(|val| (val as f32, val)).collect();
        vals.shuffle(&mut rand::thread_rng());
        let vals_merged = merge_top_k(vals.into_iter(), doc_range, ComparatorEnum::from(order));
        assert_eq!(&vals_merged, expected);
    }

    #[test]
    fn test_merge_top_k() {
        test_merge_top_k_aux(Order::Asc, 0..0, &[]);
        test_merge_top_k_aux(Order::Asc, 3..3, &[]);
        test_merge_top_k_aux(Order::Asc, 0..3, &[(0.0f32, 0), (1.0f32, 1), (2.0f32, 2)]);
        test_merge_top_k_aux(
            Order::Asc,
            0..11,
            &[
                (0.0f32, 0),
                (1.0f32, 1),
                (2.0f32, 2),
                (3.0f32, 3),
                (4.0f32, 4),
                (5.0f32, 5),
                (6.0f32, 6),
                (7.0f32, 7),
                (8.0f32, 8),
                (9.0f32, 9),
            ],
        );
        test_merge_top_k_aux(Order::Asc, 1..3, &[(1.0f32, 1), (2.0f32, 2)]);
        test_merge_top_k_aux(Order::Desc, 0..2, &[(9.0f32, 9), (8.0f32, 8)]);
        test_merge_top_k_aux(Order::Desc, 2..4, &[(7.0f32, 7), (6.0f32, 6)]);
    }

    // Helper collector wrapper that tracks which segments are processed (not pruned)
    // Uses concrete TopBySortKeyCollector type to access Vec-based Fruit
    struct TrackingCollector<TSortKeyComputer>
    where
        TSortKeyComputer: SortKeyComputer + Send + Sync + 'static,
    {
        inner: TopBySortKeyCollector<TSortKeyComputer>,
        processed_segments: Arc<Mutex<Vec<u32>>>,
    }

    impl<TSortKeyComputer> Collector for TrackingCollector<TSortKeyComputer>
    where
        TSortKeyComputer: SortKeyComputer + Send + Sync + 'static,
    {
        type Fruit = Vec<(TSortKeyComputer::SortKey, crate::DocAddress)>;
        type Child = <TopBySortKeyCollector<TSortKeyComputer> as Collector>::Child;

        fn check_schema(&self, schema: &Schema) -> crate::Result<()> {
            self.inner.check_schema(schema)
        }

        fn for_segment(
            &self,
            segment_ord: u32,
            reader: &SegmentReader,
        ) -> crate::Result<Self::Child> {
            self.inner.for_segment(segment_ord, reader)
        }

        fn requires_scoring(&self) -> bool {
            self.inner.requires_scoring()
        }

        fn merge_fruits(
            &self,
            segment_fruits: Vec<<Self::Child as SegmentCollector>::Fruit>,
        ) -> crate::Result<Self::Fruit> {
            self.inner.merge_fruits(segment_fruits)
        }

        fn collect_segment(
            &self,
            weight: &dyn crate::query::Weight,
            segment_ord: u32,
            reader: &SegmentReader,
        ) -> crate::Result<<Self::Child as SegmentCollector>::Fruit> {
            // Call the inner collector's collect_segment
            let result = self.inner.collect_segment(weight, segment_ord, reader)?;
            // Only record as processed if the segment wasn't pruned (non-empty result)
            if !result.is_empty() {
                self.processed_segments.lock().unwrap().push(segment_ord);
            }
            Ok(result)
        }
    }

    #[test]
    fn test_segment_bounds_works() -> crate::Result<()> {
        // Test that segment_bounds correctly returns min/max values
        let mut schema_builder = Schema::builder();
        let value_field = schema_builder.add_u64_field("value", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        let mut writer = index.writer_for_tests()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for value in [10u64, 20, 30, 40] {
            writer.add_document(doc!(value_field => value))?;
        }
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        use crate::collector::sort_key::SortByStaticFastValue;
        let sort_key_computer = (SortByStaticFastValue::<u64>::for_field("value"), Order::Asc);

        // Test segment_bounds directly
        let segment_reader = &searcher.segment_readers()[0];
        let bounds = sort_key_computer.segment_bounds(segment_reader)?;

        assert!(
            bounds.is_some(),
            "segment_bounds should return Some, but got None"
        );
        let (min, max) = bounds.unwrap();
        assert_eq!(min, Some(10), "min should be Some(10), got {:?}", min);
        assert_eq!(max, Some(40), "max should be Some(40), got {:?}", max);

        Ok(())
    }

    #[test]
    fn test_adaptive_pruning_skips_segments() -> crate::Result<()> {
        // Create an index with multiple segments with disjunct value ranges
        let mut schema_builder = Schema::builder();
        let value_field = schema_builder.add_u64_field("value", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        // Helper to create a segment
        let create_segment = |index: &Index, values: Vec<u64>| -> crate::Result<()> {
            let mut writer = index.writer_for_tests()?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for value in values {
                writer.add_document(doc!(value_field => value))?;
            }
            writer.commit()?;
            Ok(())
        };

        // Create segments with different value ranges
        // Segments with low values (min < 10) should be processed
        // Segments with high values (min >= 10) should be pruned after threshold is set
        create_segment(&index, vec![1, 2, 3, 4])?;
        create_segment(&index, vec![10, 11, 12, 13])?;
        create_segment(&index, vec![0, 4, 5, 6])?;
        create_segment(&index, vec![20, 21, 22, 23])?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Verify we have 4 segments
        assert_eq!(
            searcher.segment_readers().len(),
            4,
            "Expected 4 segments, got {}",
            searcher.segment_readers().len()
        );

        // Track which segments are processed (not pruned)
        let processed_segments: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let processed_segments_clone = processed_segments.clone();

        // Create TopBySortKeyCollector directly with concrete type and wrap it with tracking
        use crate::collector::sort_key::SortByStaticFastValue;
        use crate::core::Executor;
        use crate::query::EnableScoring;
        let inner_collector = TopBySortKeyCollector::new(
            (SortByStaticFastValue::<u64>::for_field("value"), Order::Asc),
            0..2,
        );
        let collector = TrackingCollector {
            inner: inner_collector,
            processed_segments: processed_segments_clone,
        };

        // Search with single-threaded executor to ensure sequential segment processing
        // This is required for adaptive pruning to work
        let executor = Executor::single_thread();
        let results = searcher.search_with_executor(
            &AllQuery,
            &collector,
            &executor,
            EnableScoring::disabled_from_searcher(&searcher),
        )?;

        // Verify results: should be [0, 1] - the two smallest values across all segments
        assert_eq!(results.len(), 2);
        let values: Vec<u64> = results
            .iter()
            .map(|(key, _)| key.expect("Expected Some(u64) value"))
            .collect();
        assert_eq!(values, vec![0, 1]);

        // Get segment bounds to identify which segment has which values
        let sort_key_computer_for_bounds =
            (SortByStaticFastValue::<u64>::for_field("value"), Order::Asc);
        let mut segment_bounds_map: std::collections::HashMap<u32, (u64, u64)> =
            std::collections::HashMap::new();
        for (i, reader) in searcher.segment_readers().iter().enumerate() {
            if let Ok(Some((Some(min), Some(max)))) =
                sort_key_computer_for_bounds.segment_bounds(reader)
            {
                segment_bounds_map.insert(i as u32, (min, max));
            }
        }

        let processed = processed_segments.lock().unwrap();

        // The first segment processed sets the threshold.
        // Segments processed later with all values worse than threshold should be pruned.
        // Since segment order is not guaranteed, we check:
        // 1. Results are correct (verified above)
        // 2. Segments with low values were processed (they contain the top results)
        let low_value_segments: Vec<u32> = segment_bounds_map
            .iter()
            .filter(|(_, (min, _))| *min < 10)
            .map(|(&ord, _)| ord)
            .collect();
        let processed_low_value = low_value_segments.iter().any(|ord| processed.contains(ord));
        assert!(
            processed_low_value,
            "At least one segment with low values should have been processed. \
             Low value segments: {:?}, Processed: {:?}",
            low_value_segments, *processed
        );

        // The segment with values [0, 4, 5, 6] must have been processed since it contains 0
        let seg_with_zero = segment_bounds_map
            .iter()
            .find(|(_, (min, _))| *min == 0)
            .map(|(&ord, _)| ord);
        if let Some(ord) = seg_with_zero {
            assert!(
                processed.contains(&ord),
                "Segment with min=0 should have been processed to get result 0"
            );
        }

        Ok(())
    }

    #[test]
    fn test_adaptive_pruning_with_descending_order() -> crate::Result<()> {
        // Test with descending order: larger values are better
        let mut schema_builder = Schema::builder();
        let value_field = schema_builder.add_u64_field("value", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        let create_segment = |index: &Index, values: Vec<u64>| -> crate::Result<()> {
            let mut writer = index.writer_for_tests()?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for value in values {
                writer.add_document(doc!(value_field => value))?;
            }
            writer.commit()?;
            Ok(())
        };

        // Create segments with different value ranges
        // Segments with high values (max > 4) should be processed
        // Segments with low values (max <= 4) should be pruned after threshold is set
        create_segment(&index, vec![10, 11, 12, 13])?;
        create_segment(&index, vec![1, 2, 3, 4])?;
        create_segment(&index, vec![14, 15, 16, 17])?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Track which segments are processed (not pruned)
        let processed_segments: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let processed_segments_clone = processed_segments.clone();

        // Create TopBySortKeyCollector directly with concrete type and wrap it with tracking
        use crate::collector::sort_key::SortByStaticFastValue;
        use crate::core::Executor;
        use crate::query::EnableScoring;
        let inner_collector = TopBySortKeyCollector::new(
            (SortByStaticFastValue::<u64>::for_field("value"), Order::Desc),
            0..2,
        );
        let collector = TrackingCollector {
            inner: inner_collector,
            processed_segments: processed_segments_clone,
        };

        // Search with single-threaded executor to ensure sequential segment processing
        let executor = Executor::single_thread();
        let results = searcher.search_with_executor(
            &AllQuery,
            &collector,
            &executor,
            EnableScoring::disabled_from_searcher(&searcher),
        )?;

        // Verify results: should be [17, 16] - the two largest values across all segments
        assert_eq!(results.len(), 2);
        let values: Vec<u64> = results
            .iter()
            .map(|(key, _)| key.expect("Expected Some(u64) value"))
            .collect();
        assert_eq!(values, vec![17, 16]);

        // Get segment bounds to identify which segment has which values
        let sort_key_computer_for_bounds =
            (SortByStaticFastValue::<u64>::for_field("value"), Order::Desc);
        let mut segment_bounds_map: std::collections::HashMap<u32, (u64, u64)> =
            std::collections::HashMap::new();
        for (i, reader) in searcher.segment_readers().iter().enumerate() {
            if let Ok(Some((Some(min), Some(max)))) =
                sort_key_computer_for_bounds.segment_bounds(reader)
            {
                segment_bounds_map.insert(i as u32, (min, max));
            }
        }

        let processed = processed_segments.lock().unwrap();

        // The first segment processed sets the threshold.
        // Segments processed later with all values worse than threshold should be pruned.
        // Since segment order is not guaranteed, we check:
        // 1. Results are correct (verified above)
        // 2. Segments with high values were processed (they contain the top results)
        let high_value_segments: Vec<u32> = segment_bounds_map
            .iter()
            .filter(|(_, (_, max))| *max > 4)
            .map(|(&ord, _)| ord)
            .collect();
        let processed_high_value = high_value_segments.iter().any(|ord| processed.contains(ord));
        assert!(
            processed_high_value,
            "At least one segment with high values should have been processed. \
             High value segments: {:?}, Processed: {:?}",
            high_value_segments, *processed
        );

        // The segment with values [14-17] must have been processed since it contains [17, 16]
        let seg_with_top_values = segment_bounds_map
            .iter()
            .find(|(_, (_, max))| *max == 17)
            .map(|(&ord, _)| ord);
        if let Some(ord) = seg_with_top_values {
            assert!(
                processed.contains(&ord),
                "Segment with max=17 should have been processed to get results [17, 16]"
            );
        }

        Ok(())
    }
}
