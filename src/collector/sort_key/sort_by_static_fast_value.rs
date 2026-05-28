use std::marker::PhantomData;
use std::sync::Arc;

use columnar::Column;

use crate::collector::sort_key::ComparatorEnum;
use crate::collector::sort_key::shared_threshold::{RwLockSharedThresholdOptionU64, SharedThreshold};
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer};
use crate::fastfield::{FastFieldNotAvailableError, FastValue};
use crate::{DocId, Order, Score, SegmentReader};

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

    fn create_shared_threshold(&self) -> Option<Arc<dyn SharedThreshold<<<Self as SortKeyComputer>::Child as SegmentSortKeyComputer>::SegmentSortKey>>> {
        let rwlock = Arc::new(RwLockSharedThresholdOptionU64::new(self.order));
        Some(rwlock)
    }

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
