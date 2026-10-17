//! Merge-time [`VectorSource`]: sampled rows for train, every row for assign.

use std::ops::Range;
use std::sync::Arc;

use crate::indexer::segment_updater::CancelSentinel;
use crate::vector::clustering::{
    decode_row_into, IvfMatrix, IvfTrainingBatch, IvfTrainingVectors, VectorSource,
};
use crate::vector::storage::quantized::ReadScratch;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::write::RowAddress;
use crate::{DocId, TantivyError};

/// Rows per planned read, bounding the raw bytes held while they decode.
const PLANNED_READ_ROWS: usize = 4_096;

/// Merge rows scanned in source storage order.
pub(super) struct MergeVectorSource<'a> {
    rows: Vec<(RowAddress, DocId)>,
    reader: RowReader<'a>,
    pub(super) reads: usize,
}

impl<'a> MergeVectorSource<'a> {
    pub(super) fn new(
        readers: &'a [Arc<VectorIndexReader>],
        rows: Vec<(RowAddress, DocId)>,
        dim: usize,
        cancel: &'a dyn CancelSentinel,
    ) -> Self {
        MergeVectorSource {
            rows,
            reader: RowReader {
                readers,
                dim,
                cancel,
                scratch: ReadScratch::default(),
                order: Vec::new(),
            },
            reads: 0,
        }
    }

    pub(super) fn rows(&self) -> &[(RowAddress, DocId)] {
        &self.rows
    }
}

/// Reads requested rows in storage order and decodes each into its request slot.
struct RowReader<'a> {
    readers: &'a [Arc<VectorIndexReader>],
    dim: usize,
    cancel: &'a dyn CancelSentinel,
    scratch: ReadScratch,
    /// Request slots sorted by source address.
    order: Vec<u32>,
}

impl RowReader<'_> {
    /// Decodes `requested` into a new batch in request order.
    fn load(&mut self, requested: &[(RowAddress, DocId)]) -> crate::Result<IvfTrainingVectors> {
        if self.cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
        let mut values = vec![0.0; requested.len() * self.dim];
        let mut doc_ids = vec![0; requested.len()];
        self.decode(requested, &mut values, &mut doc_ids)?;
        Ok(IvfTrainingVectors::F32(IvfTrainingBatch {
            doc_ids,
            matrix: IvfMatrix {
                values,
                rows: requested.len(),
                dims: self.dim,
            },
        }))
    }

    /// Fills `values` (`requested.len() * dim`) and `doc_ids` (`requested.len()`) in request
    /// order. Each segment's rows are fetched ascending through planned reads, so a request of
    /// scattered positions costs one forward pass per segment.
    fn decode(
        &mut self,
        requested: &[(RowAddress, DocId)],
        values: &mut [f32],
        doc_ids: &mut [DocId],
    ) -> crate::Result<()> {
        let dim = self.dim;
        debug_assert_eq!(values.len(), requested.len() * dim);
        debug_assert_eq!(doc_ids.len(), requested.len());
        let address = |slot: u32| requested[slot as usize].0;
        self.order.clear();
        self.order.extend(0..requested.len() as u32);
        self.order.sort_unstable_by_key(|&slot| address(slot));

        let mut remaining = self.order.as_slice();
        while let Some(&first) = remaining.first() {
            let segment_ord = address(first).segment_ord;
            let run_len =
                remaining.partition_point(|&slot| address(slot).segment_ord == segment_ord);
            let (run, rest) = remaining.split_at(run_len);
            remaining = rest;
            let reader = &self.readers[segment_ord as usize];
            for chunk in run.chunks(PLANNED_READ_ROWS) {
                if self.cancel.wants_cancel() {
                    return Err(TantivyError::Cancelled);
                }
                self.scratch.rows.clear();
                self.scratch
                    .rows
                    .extend(chunk.iter().map(|&slot| address(slot).row_id));
                self.scratch.rows.dedup();
                let batch = reader.read_vector_rows_planned(&mut self.scratch)?;
                let mut slots = chunk.iter().copied().peekable();
                for (row, bytes) in batch.iter() {
                    while let Some(slot) = slots.next_if(|&slot| address(slot).row_id == row) {
                        let slot = slot as usize;
                        decode_row_into::<f32>(bytes, &mut values[slot * dim..(slot + 1) * dim])?;
                        doc_ids[slot] = requested[slot].1;
                    }
                }
            }
        }
        Ok(())
    }
}

impl VectorSource for MergeVectorSource<'_> {
    fn num_rows(&self) -> usize {
        self.rows.len()
    }

    fn read_range(&mut self, range: Range<usize>) -> crate::Result<IvfTrainingVectors> {
        let Some(requested) = self.rows.get(range.clone()) else {
            return Err(TantivyError::InvalidArgument(format!(
                "vector source range {range:?} is out of bounds for {} rows",
                self.rows.len()
            )));
        };
        let batch = self.reader.load(requested)?;
        self.reads += requested.len();
        Ok(batch)
    }

    fn read_rows(&mut self, rows: &[u32]) -> crate::Result<IvfTrainingVectors> {
        if let Some(&row) = rows.iter().find(|&&row| row as usize >= self.rows.len()) {
            return Err(TantivyError::InvalidArgument(format!(
                "vector source row {row} is out of bounds for {} rows",
                self.rows.len()
            )));
        }
        let requested: Vec<(RowAddress, DocId)> =
            rows.iter().map(|&row| self.rows[row as usize]).collect();
        let batch = self.reader.load(&requested)?;
        self.reads += rows.len();
        Ok(batch)
    }

    fn wants_cancel(&self) -> bool {
        self.reader.cancel.wants_cancel()
    }
}
