//! Merge-time [`VectorSource`]: sampled rows for train, every row for assign.

use std::sync::Arc;

use crate::indexer::segment_updater::CancelSentinel;
use crate::vector::clustering::{
    decode_row_append, IvfMatrix, IvfMatrixView, IvfTrainingBatch, IvfTrainingVectors,
    IvfVectorBatch, IvfVectors, VectorSource,
};
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::write::RowAddress;
use crate::{DocId, TantivyError};

/// Merge rows scanned in source storage order.
pub(super) struct MergeVectorSource<'a> {
    readers: &'a [Arc<VectorIndexReader>],
    rows: Vec<(RowAddress, DocId)>,
    dim: usize,
    cancel: &'a dyn CancelSentinel,
    pub(super) reads: usize,
    batch_values: Vec<f32>,
    batch_doc_ids: Vec<DocId>,
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
        }
    }

    pub(super) fn rows(&self) -> &[(RowAddress, DocId)] {
        &self.rows
    }
}

fn decode_source_rows(
    readers: &[Arc<VectorIndexReader>],
    dim: usize,
    rows: impl Iterator<Item = (RowAddress, DocId)>,
    values: &mut Vec<f32>,
    doc_ids: &mut Vec<DocId>,
) -> crate::Result<()> {
    for (source, target_doc_id) in rows {
        let bytes = readers[source.segment_ord as usize].vector_bytes_for_row(source.row_id)?;
        decode_row_append::<f32>(&bytes, dim, values)?;
        doc_ids.push(target_doc_id);
    }
    Ok(())
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
            self.batch_values.clear();
            self.batch_doc_ids.clear();
            decode_source_rows(
                self.readers,
                self.dim,
                batch.iter().copied(),
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
        let mut values = Vec::with_capacity(rows.len() * self.dim);
        let mut doc_ids = Vec::with_capacity(rows.len());
        decode_source_rows(
            self.readers,
            self.dim,
            rows.iter().map(|&row| self.rows[row as usize]),
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
