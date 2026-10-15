//! Merge-time [`VectorSource`]: sampled rows for train, every row for assign.

use std::sync::Arc;

use crate::indexer::segment_updater::CancelSentinel;
use crate::vector::clustering::{
    decode_row_into, IvfMatrix, IvfMatrixView, IvfTrainingBatch, IvfTrainingVectors,
    IvfVectorBatch, IvfVectors, VectorSource,
};
use crate::vector::storage::quantized::ReadScratch;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::write::RowAddress;
use crate::{DocId, TantivyError};

/// Rows per planned read, bounding the raw bytes held while they decode.
const PLANNED_READ_ROWS: usize = 4_096;

/// Merge rows scanned in source storage order.
pub(super) struct MergeVectorSource<'a> {
    readers: &'a [Arc<VectorIndexReader>],
    rows: Vec<(RowAddress, DocId)>,
    dim: usize,
    cancel: &'a dyn CancelSentinel,
    pub(super) reads: usize,
    batch_values: Vec<f32>,
    batch_doc_ids: Vec<DocId>,
    reader: RowReader,
}

impl<'a> MergeVectorSource<'a> {
    pub(super) fn new(
        readers: &'a [Arc<VectorIndexReader>],
        rows: Vec<(RowAddress, DocId)>,
        dim: usize,
        cancel: &'a dyn CancelSentinel,
    ) -> Self {
        MergeVectorSource {
            readers,
            rows,
            dim,
            cancel,
            reads: 0,
            batch_values: Vec::new(),
            batch_doc_ids: Vec::new(),
            reader: RowReader::default(),
        }
    }

    pub(super) fn rows(&self) -> &[(RowAddress, DocId)] {
        &self.rows
    }
}

/// Reads requested rows in storage order and decodes each into its request slot.
#[derive(Default)]
struct RowReader {
    scratch: ReadScratch,
    /// Request slots sorted by source address.
    order: Vec<u32>,
}

impl RowReader {
    /// Fills `values` (`requested.len() * dim`) and `doc_ids` (`requested.len()`) in request
    /// order. Each segment's rows are fetched ascending through planned reads, so a request of
    /// scattered positions costs one forward pass per segment.
    fn decode(
        &mut self,
        readers: &[Arc<VectorIndexReader>],
        cancel: &dyn CancelSentinel,
        dim: usize,
        requested: &[(RowAddress, DocId)],
        values: &mut [f32],
        doc_ids: &mut [DocId],
    ) -> crate::Result<()> {
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
            let reader = &readers[segment_ord as usize];
            for chunk in run.chunks(PLANNED_READ_ROWS) {
                if cancel.wants_cancel() {
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

    fn for_each_batch(
        &mut self,
        batch_size: usize,
        f: &mut dyn FnMut(IvfVectors<'_>),
    ) -> crate::Result<()> {
        for batch in self.rows.chunks(batch_size.max(1)) {
            if self.cancel.wants_cancel() {
                return Err(TantivyError::Cancelled);
            }
            self.batch_values.resize(batch.len() * self.dim, 0.0);
            self.batch_doc_ids.resize(batch.len(), 0);
            self.reader.decode(
                self.readers,
                self.cancel,
                self.dim,
                batch,
                &mut self.batch_values,
                &mut self.batch_doc_ids,
            )?;
            self.reads += batch.len();
            f(IvfVectors::F32(IvfVectorBatch {
                doc_ids: &self.batch_doc_ids,
                matrix: IvfMatrixView {
                    values: &self.batch_values,
                    rows: batch.len(),
                    dims: self.dim,
                },
            }));
        }
        Ok(())
    }

    fn read_rows(&mut self, rows: &[u32]) -> crate::Result<IvfTrainingVectors> {
        if self.cancel.wants_cancel() {
            return Err(TantivyError::Cancelled);
        }
        if let Some(&row) = rows.iter().find(|&&row| row as usize >= self.rows.len()) {
            return Err(TantivyError::InvalidArgument(format!(
                "vector source row {row} is out of bounds for {} rows",
                self.rows.len()
            )));
        }
        let requested: Vec<(RowAddress, DocId)> =
            rows.iter().map(|&row| self.rows[row as usize]).collect();
        let mut values = vec![0.0; rows.len() * self.dim];
        let mut doc_ids = vec![0; rows.len()];
        self.reader.decode(
            self.readers,
            self.cancel,
            self.dim,
            &requested,
            &mut values,
            &mut doc_ids,
        )?;
        self.reads += rows.len();
        Ok(IvfTrainingVectors::F32(IvfTrainingBatch {
            doc_ids,
            matrix: IvfMatrix {
                values,
                rows: rows.len(),
                dims: self.dim,
            },
        }))
    }
}
