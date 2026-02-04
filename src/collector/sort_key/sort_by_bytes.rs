use columnar::BytesColumn;

use crate::collector::sort_key::NaturalComparator;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer};
use crate::termdict::TermOrdinal;
use crate::{DocId, Score};

/// Sort by the first value of a bytes column.
///
/// If the field is multivalued, only the first value is considered.
///
/// Documents that do not have this value are still considered.
/// Their sort key will simply be `None`.
#[derive(Debug, Clone)]
pub struct SortByBytes {
    column_name: String,
}

impl SortByBytes {
    /// Creates a new sort by bytes sort key computer.
    pub fn for_field(column_name: impl ToString) -> Self {
        SortByBytes {
            column_name: column_name.to_string(),
        }
    }
}

impl SortKeyComputer for SortByBytes {
    type SortKey = Option<Vec<u8>>;
    type Child = ByBytesColumnSegmentSortKeyComputer;
    type Comparator = NaturalComparator;

    fn segment_sort_key_computer(
        &self,
        segment_reader: &crate::SegmentReader,
    ) -> crate::Result<Self::Child> {
        let bytes_column_opt = segment_reader.fast_fields().bytes(&self.column_name)?;
        Ok(ByBytesColumnSegmentSortKeyComputer { bytes_column_opt })
    }
}

/// Segment-level sort key computer for bytes columns.
pub struct ByBytesColumnSegmentSortKeyComputer {
    bytes_column_opt: Option<BytesColumn>,
}

impl SegmentSortKeyComputer for ByBytesColumnSegmentSortKeyComputer {
    type SortKey = Option<Vec<u8>>;
    type SegmentSortKey = Option<TermOrdinal>;
    type SegmentComparator = NaturalComparator;

    #[inline(always)]
    fn segment_sort_key(&mut self, doc: DocId, _score: Score) -> Option<TermOrdinal> {
        let bytes_column = self.bytes_column_opt.as_ref()?;
        bytes_column.ords().first(doc)
    }

    fn convert_segment_sort_key(&self, term_ord_opt: Option<TermOrdinal>) -> Option<Vec<u8>> {
        // TODO: Individual lookups to the dictionary like this are very likely to repeatedly
        // decompress the same blocks. See https://github.com/quickwit-oss/tantivy/issues/2776
        let term_ord = term_ord_opt?;
        let bytes_column = self.bytes_column_opt.as_ref()?;
        let mut bytes = Vec::new();
        bytes_column
            .dictionary()
            .ord_to_term(term_ord, &mut bytes)
            .ok()?;
        Some(bytes)
    }
}
