use std::marker::PhantomData;
use std::sync::Arc;

use columnar::Column;

use crate::collector::sort_key::ComparatorEnum;
use crate::collector::sort_key::shared_threshold::{RwLockSharedThresholdOptionU64, SharedThreshold};
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer, TopNComputer};
use crate::collector::sort_key_top_collector::TopBySortKeySegmentCollector;
use crate::fastfield::{FastFieldNotAvailableError, FastValue};
use crate::{DocAddress, DocId, Order, Score, SegmentReader};

const TRUNCATIONS_PER_SHARED_UPDATE: u32 = 2;

/// Sorts by a fast value (u64, i64, f64, bool).
///
/// The field must appear explicitly in the schema, with the right type, and declared as
/// a fast field..
///
/// If the field is multivalued, only the first value is considered.
///
/// Documents that do not have this value are still considered.
/// Their sort key will simply be `None`.
#[derive(Clone)]
pub struct SortByStaticFastValue<T: FastValue> {
    field: String,
    order: Order,
    shared_threshold: Arc<RwLockSharedThresholdOptionU64>,
    typ: PhantomData<T>,
}

impl<T: FastValue> std::fmt::Debug for SortByStaticFastValue<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SortByStaticFastValue")
            .field("field", &self.field)
            .field("order", &self.order)
            .finish()
    }
}

impl<T: FastValue> SortByStaticFastValue<T> {
    /// Creates a new `SortByStaticFastValue` instance for the given field and order.
    pub fn for_field_and_order(column_name: impl ToString, order: Order) -> SortByStaticFastValue<T> {
        Self {
            field: column_name.to_string(),
            order,
            shared_threshold: Arc::new(RwLockSharedThresholdOptionU64::new(order)),
            typ: PhantomData,
        }
    }

    /// Backwards compatibility / when order is ignored (assumed Asc).
    pub fn for_field(column_name: impl ToString) -> SortByStaticFastValue<T> {
        Self::for_field_and_order(column_name, Order::Asc)
    }
}

impl<T: FastValue> SortKeyComputer for SortByStaticFastValue<T> {
    type Child = SortByFastValueSegmentSortKeyComputer<T>;
    type SortKey = Option<T>;
    type Comparator = ComparatorEnum;

    fn check_schema(&self, schema: &crate::schema::Schema) -> crate::Result<()> {
        // At the segment sort key computer level, we rely on the u64 representation.
        // The mapping is monotonic, so it is sufficient to compute our top-K docs.
        let field = schema.get_field(&self.field)?;
        let field_entry = schema.get_field_entry(field);
        if !field_entry.is_fast() {
            return Err(crate::TantivyError::SchemaError(format!(
                "Field `{}` is not a fast field.",
                self.field,
            )));
        }
        let schema_type = field_entry.field_type().value_type();
        if schema_type != T::to_type() {
            return Err(crate::TantivyError::SchemaError(format!(
                "Field `{}` is of type {schema_type:?}, not of the type {:?}.",
                &self.field,
                T::to_type()
            )));
        }
        Ok(())
    }

    fn comparator(&self) -> Self::Comparator {
        self.order.into()
    }

    fn segment_sort_key_computer(
        &self,
        segment_reader: &SegmentReader,
    ) -> crate::Result<Self::Child> {
        let sort_column_opt = segment_reader.fast_fields().u64_lenient(&self.field)?;
        let (sort_column, _sort_column_type) =
            sort_column_opt.ok_or_else(|| FastFieldNotAvailableError {
                field_name: self.field.clone(),
            })?;
        Ok(SortByFastValueSegmentSortKeyComputer {
            sort_column,
            order: self.order,
            typ: PhantomData,
        })
    }

    fn collect_segment_top_k(
        &self,
        k: usize,
        weight: &dyn crate::query::Weight,
        reader: &crate::SegmentReader,
        segment_ord: u32,
    ) -> crate::Result<Vec<(Self::SortKey, DocAddress)>> {
        let segment_sort_key_computer = self.segment_sort_key_computer(reader)?;
        let mut top_n: TopNComputer<Option<u64>, DocId, Self::Comparator> =
            TopNComputer::new_with_comparator(k, self.comparator());

        let shared = &self.shared_threshold;
        let mut threshold = shared.load();
        
        // Only initialize top_n.threshold if it's strictly better than the worst value (None).
        // If we initialize it to None, TopNComputer will strictly reject equal elements (None),
        // preventing us from collecting them even when the buffer isn't full.
        if threshold.is_some() {
            top_n.threshold = Some(threshold);
        }
        let mut truncation_count: u32 = 0;

        let mut segment_top_key_collector = TopBySortKeySegmentCollector {
            topn_computer: top_n,
            segment_ord,
            segment_sort_key_computer,
        };

        let mut check_truncation = |collector: &mut TopBySortKeySegmentCollector<SortByFastValueSegmentSortKeyComputer<T>, Self::Comparator>| {
            let new_threshold_opt = collector.topn_computer.threshold;
            if let Some(new_thresh) = new_threshold_opt {
                let threshold_improved = match self.order {
                    Order::Asc => new_thresh < threshold,
                    Order::Desc => new_thresh > threshold,
                };
                if threshold_improved {
                    threshold = new_thresh;
                    truncation_count += 1;
                    if truncation_count % TRUNCATIONS_PER_SHARED_UPDATE == 0 {
                        shared.update(new_thresh);
                        let global = shared.load();
                        let global_improved = match self.order {
                            Order::Asc => global < threshold,
                            Order::Desc => global > threshold,
                        };
                        if global_improved {
                            threshold = global;
                            collector.topn_computer.threshold = Some(threshold);
                        }
                    }
                }
            }
        };

        use crate::collector::SegmentCollector;
        let with_scoring = self.requires_scoring();
        match (reader.alive_bitset(), with_scoring) {
            (Some(alive_bitset), true) => {
                weight.for_each(reader, &mut |doc, score| {
                    if alive_bitset.is_alive(doc) {
                        segment_top_key_collector.collect(doc, score);
                        check_truncation(&mut segment_top_key_collector);
                    }
                })?;
            }
            (Some(alive_bitset), false) => {
                weight.for_each_no_score(reader, &mut |docs| {
                    for doc in docs.iter().cloned() {
                        if alive_bitset.is_alive(doc) {
                            segment_top_key_collector.collect(doc, 0.0);
                            check_truncation(&mut segment_top_key_collector);
                        }
                    }
                })?;
            }
            (None, true) => {
                weight.for_each(reader, &mut |doc, score| {
                    segment_top_key_collector.collect(doc, score);
                    check_truncation(&mut segment_top_key_collector);
                })?;
            }
            (None, false) => {
                weight.for_each_no_score(reader, &mut |docs| {
                    for doc in docs.iter().cloned() {
                        segment_top_key_collector.collect(doc, 0.0);
                        check_truncation(&mut segment_top_key_collector);
                    }
                })?;
            }
        }

        if let Some(final_threshold) = segment_top_key_collector.topn_computer.threshold {
            shared.update(final_threshold);
        }

        let segment_hits: Vec<(Option<T>, DocAddress)> = segment_top_key_collector
            .topn_computer
            .into_vec()
            .into_iter()
            .map(|comparable_doc| {
                let sort_key = segment_top_key_collector
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
        Ok(segment_hits)
    }
}

pub struct SortByFastValueSegmentSortKeyComputer<T> {
    sort_column: Column<u64>,
    order: Order,
    typ: PhantomData<T>,
}

impl<T: FastValue> SegmentSortKeyComputer for SortByFastValueSegmentSortKeyComputer<T> {
    type SortKey = Option<T>;
    type SegmentSortKey = Option<u64>;
    type SegmentComparator = ComparatorEnum;

    fn segment_comparator(&self) -> Self::SegmentComparator {
        self.order.into()
    }

    #[inline(always)]
    fn segment_sort_key(&mut self, doc: DocId, _score: Score) -> Self::SegmentSortKey {
        self.sort_column.first(doc)
    }

    fn convert_segment_sort_key(&self, sort_key: Self::SegmentSortKey) -> Self::SortKey {
        sort_key.map(T::from_u64)
    }
}
