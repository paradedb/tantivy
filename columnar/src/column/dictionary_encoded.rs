use std::ops::Deref;
use std::sync::Arc;
use std::{fmt, io};

use sstable::{Dictionary, VoidSSTable};

use crate::RowId;
use crate::column::Column;

/// Dictionary encoded column.
///
/// The column simply gives access to a regular u64-column that, in
/// which the values are term-ordinals.
///
/// These ordinals are ids uniquely identify the bytes that are stored in
/// the column. These ordinals are small, and sorted in the same order
/// as the term_ord_column.
#[derive(Clone)]
pub struct BytesColumn {
    pub(crate) dictionary: Arc<Dictionary<VoidSSTable>>,
    pub(crate) term_ord_column: Column<u64>,
}

impl fmt::Debug for BytesColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesColumn")
            .field("term_ord_column", &self.term_ord_column)
            .finish()
    }
}

impl BytesColumn {
    pub fn empty(num_docs: u32) -> BytesColumn {
        BytesColumn {
            dictionary: Arc::new(Dictionary::empty()),
            term_ord_column: Column::build_empty_column(num_docs),
        }
    }

    /// Fills the given `output` buffer with the term associated to the ordinal `ord`.
    ///
    /// Returns `false` if the term does not exist (e.g. `term_ord` is greater or equal to the
    /// overll number of terms).
    pub fn ord_to_bytes(&self, ord: u64, output: &mut Vec<u8>) -> io::Result<bool> {
        self.dictionary.ord_to_term(ord, output)
    }

    /// Returns the number of rows in the column.
    pub fn num_rows(&self) -> RowId {
        self.term_ord_column.num_docs()
    }

    pub fn term_ords(&self, row_id: RowId) -> impl Iterator<Item = u64> + '_ {
        self.term_ord_column.values_for_doc(row_id)
    }

    /// Returns the column of ordinals
    pub fn ords(&self) -> &Column<u64> {
        &self.term_ord_column
    }

    pub fn num_terms(&self) -> usize {
        self.dictionary.num_terms()
    }

    pub fn dictionary(&self) -> &Dictionary<VoidSSTable> {
        self.dictionary.as_ref()
    }

    /// High-level convenience method: Finds the exact contiguous DocId range
    /// matching a range of byte slices in a sorted dictionary-encoded column.
    ///
    /// **Warning:** This method assumes the column is **sorted** by value.
    /// If the column is not sorted, the result is undefined and will be incorrect.
    ///
    /// The `docid_range` allows restricting the binary search space.
    ///
    /// # Panics
    ///
    /// Panics if the column cardinality is `Multivalued`.
    pub fn binary_search_range(
        &self,
        docid_range: Option<std::ops::Range<u32>>,
        lower_bound: std::ops::Bound<&[u8]>,
        upper_bound: std::ops::Bound<&[u8]>,
        is_descending: bool,
    ) -> io::Result<std::ops::Range<u32>> {
        use sstable::TermOrdHit;

        let dict = self.dictionary();
        let max_ord = dict.num_terms() as u64;

        if max_ord == 0 {
            return Ok(0..0);
        }

        // Resolve lower bound to a starting ordinal
        let start_ord = match lower_bound {
            std::ops::Bound::Included(key) => match dict.term_ord_or_next(key)? {
                TermOrdHit::Exact(ord) | TermOrdHit::Next(ord) => ord,
            },
            std::ops::Bound::Excluded(key) => match dict.term_ord_or_next(key)? {
                TermOrdHit::Exact(ord) => ord + 1,
                TermOrdHit::Next(ord) => ord,
            },
            std::ops::Bound::Unbounded => 0,
        };

        // Resolve upper bound to an ending ordinal
        let end_ord = match upper_bound {
            std::ops::Bound::Included(key) => match dict.term_ord_or_next(key)? {
                TermOrdHit::Exact(ord) => ord,
                TermOrdHit::Next(ord) => {
                    if ord == 0 {
                        return Ok(0..0);
                    }
                    ord - 1
                }
            },
            std::ops::Bound::Excluded(key) => match dict.term_ord_or_next(key)? {
                TermOrdHit::Exact(ord) | TermOrdHit::Next(ord) => {
                    if ord == 0 {
                        return Ok(0..0);
                    }
                    ord - 1
                }
            },
            std::ops::Bound::Unbounded => max_ord - 1,
        };

        if start_ord > end_ord {
            return Ok(0..0);
        }

        // Delegate to the numeric ordinal column
        Ok(self
            .ords()
            .binary_search_range(docid_range, &(start_ord..=end_ord), is_descending))
    }
}

#[derive(Clone)]
pub struct StrColumn(BytesColumn);

impl fmt::Debug for StrColumn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.term_ord_column)
    }
}

impl From<StrColumn> for BytesColumn {
    fn from(str_column: StrColumn) -> BytesColumn {
        str_column.0
    }
}

impl StrColumn {
    pub fn wrap(bytes_column: BytesColumn) -> StrColumn {
        StrColumn(bytes_column)
    }

    pub fn dictionary(&self) -> &Dictionary<VoidSSTable> {
        self.0.dictionary.as_ref()
    }

    /// Fills the buffer
    pub fn ord_to_str(&self, term_ord: u64, output: &mut String) -> io::Result<bool> {
        unsafe {
            let buf = output.as_mut_vec();
            if !self.0.dictionary.ord_to_term(term_ord, buf)? {
                return Ok(false);
            }
            // TODO consider remove checks if it hurts performance.
            if std::str::from_utf8(buf.as_slice()).is_err() {
                buf.clear();
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Not valid utf-8",
                ));
            }
        }
        Ok(true)
    }
}

impl Deref for StrColumn {
    type Target = BytesColumn;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
