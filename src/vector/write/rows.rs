//! A field's full-precision rows streamed into a target segment's `.vec` ahead of its merge, and
//! the finished entry the merge trains and assigns from.

use std::ops::Range;
use std::path::PathBuf;

use common::HasLen;

use crate::directory::{CompositeWrite, Directory, FileSlice};
use crate::index::{Index, Segment, SegmentComponent};
use crate::schema::{Field, FieldType, VectorDType, VectorOptions};
use crate::vector::distance::maybe_normalize_bytes;
use crate::vector::storage::header::{write_vector_header, VectorEntry, HEADER_LEN, VEC_EXT};
use crate::vector::storage::id_map::IdMap;
use crate::vector::storage::reader::VectorIndexReader;
use crate::vector::write::flat::{FlatEntryWriter, RowAdmission};
use crate::vector::write::RowAddress;
use crate::vector::ENTRY_ALIGN;
use crate::{DocId, TantivyError};

/// One vector field's full-precision rows, appended to a target segment's `.vec` in document
/// order before the segment's merge.
pub struct VectorRowsWriter {
    field: Field,
    name: String,
    admission: RowAdmission,
    path: PathBuf,
    vec_write: CompositeWrite,
    /// Zero bytes between the file header and the rows entry.
    lead_pad: usize,
    /// The rows entry's start within the composite.
    entry_start: u64,
    entry: FlatEntryWriter,
    /// The documents with a row, as ascending ranges that do not touch.
    present: Vec<Range<DocId>>,
    /// The row being stored.
    row: Vec<u8>,
}

impl VectorRowsWriter {
    /// The field whose rows can be streamed: the index's only vector field, when it is f32 and
    /// quantized.
    pub fn field_for(index: &Index) -> Option<Field> {
        let schema = index.schema();
        let mut vector_fields = schema.fields().filter_map(|(field, entry)| {
            let FieldType::Vector(opts) = entry.field_type() else {
                return None;
            };
            Some((field, entry.name(), opts))
        });
        let (field, name, opts) = vector_fields.next()?;
        if vector_fields.next().is_some() {
            return None;
        }
        let quantized = index
            .settings()
            .vector_quantization
            .iter()
            .any(|config| config.field == name);
        // The merge decodes streamed rows as f32.
        let decodable = match opts.dtype() {
            VectorDType::F32 => true,
        };
        (decodable && quantized).then_some(field)
    }

    /// Opens `target`'s `.vec` and starts `field`'s rows.
    pub fn create(target: &Segment, field: Field) -> crate::Result<Self> {
        if Self::field_for(target.index()) != Some(field) {
            return Err(TantivyError::InvalidArgument(format!(
                "the rows of field {field:?} cannot be streamed into a segment of this index"
            )));
        }
        let schema = target.schema();
        let entry = schema.get_field_entry(field);
        let FieldType::Vector(opts) = entry.field_type() else {
            unreachable!("field_for returns a vector field")
        };
        let component = SegmentComponent::Custom(VEC_EXT.to_string());
        let path = target.relative_path(component.clone());
        let mut vec_file = target.open_write(component)?;
        write_vector_header(&mut vec_file)?;
        let mut vec_write = CompositeWrite::wrap(vec_file);
        let lead_pad = vec_write.align_next_field(ENTRY_ALIGN, HEADER_LEN)?;
        let entry_start = vec_write.written_bytes();
        let rows = FlatEntryWriter::start(vec_write.undeclared(), opts)?;
        Ok(Self {
            field,
            name: entry.name().to_string(),
            admission: RowAdmission::new(opts),
            path,
            vec_write,
            lead_pad,
            entry_start,
            entry: rows,
            present: Vec::new(),
            row: Vec::with_capacity(opts.bytes_per_vector()),
        })
    }

    /// Appends `doc`'s row, admitted and normalized as indexing the document would.
    ///
    /// Documents ascend. A document without a vector is skipped by not calling this.
    pub fn push(&mut self, doc: DocId, row: &[u8]) -> crate::Result<()> {
        if let Some(last) = self.present.last().filter(|last| doc < last.end) {
            return Err(TantivyError::InvalidArgument(format!(
                "vector row of doc {doc} in field '{}' follows the row of doc {}",
                self.name,
                last.end - 1
            )));
        }
        self.row.clear();
        self.admission.admit(&self.name, doc, row, &mut self.row)?;
        // The stored bytes are those the exact-rows pass of a merge leaves for a row a segment
        // writer admitted: the admitted row, normalized under the field's rule.
        maybe_normalize_bytes(self.admission.options(), &mut self.row);
        self.entry
            .push_rows(self.vec_write.undeclared(), &self.row)?;
        match self.present.last_mut() {
            Some(last) if last.end == doc => last.end = doc + 1,
            _ => self.present.push(doc..doc + 1),
        }
        Ok(())
    }

    /// Bytes appended to the file.
    pub fn bytes_written(&self) -> u64 {
        HEADER_LEN as u64 + self.vec_write.written_bytes()
    }

    /// Ends the rows entry and declares it as entry `kind` of a segment of `num_docs` documents.
    /// Returns the file, open for the segment's other entries, and the finished rows.
    pub(crate) fn finish(
        self,
        kind: VectorEntry,
        num_docs: u32,
    ) -> crate::Result<(CompositeWrite, StreamedRows)> {
        let Self {
            field,
            path,
            mut vec_write,
            lead_pad,
            entry_start,
            entry,
            present,
            ..
        } = self;
        if let Some(last) = present.last().filter(|last| last.end > num_docs) {
            return Err(TantivyError::InternalError(format!(
                "a vector row was streamed for doc {} of a segment of {num_docs} documents",
                last.end - 1
            )));
        }
        let num_rows = entry.num_rows();
        let entry_len = entry.finish(vec_write.undeclared())?;
        vec_write.declare_field_at(field, kind.index(), entry_start);
        let rows = StreamedRows {
            field,
            path,
            lead_pad,
            entry: entry_start..entry_start + entry_len,
            present,
            num_rows,
        };
        Ok((vec_write, rows))
    }
}

/// A finished rows entry at the head of a `.vec` that is still open for its other entries.
pub(crate) struct StreamedRows {
    pub(crate) field: Field,
    /// The `.vec` the entry was written to.
    pub(crate) path: PathBuf,
    /// Zero bytes between the file header and the entry.
    pub(crate) lead_pad: usize,
    /// The entry's bytes within the composite.
    entry: Range<u64>,
    present: Vec<Range<DocId>>,
    num_rows: usize,
}

impl StreamedRows {
    pub(crate) fn entry_len(&self) -> u64 {
        self.entry.end - self.entry.start
    }

    /// The documents with a row, ascending: row `r` is the row of the `r`-th document.
    pub(crate) fn present_docs(&self) -> Vec<DocId> {
        let mut docs = Vec::with_capacity(self.num_rows);
        docs.extend(self.present.iter().cloned().flatten());
        docs
    }

    /// Checkpoints `vec_write` and opens a flat reader over the entry in `directory`, for a
    /// segment of `num_docs` documents of which `present` have a row.
    pub(crate) fn open_reader(
        &self,
        vec_write: &mut CompositeWrite,
        directory: &dyn Directory,
        opts: &VectorOptions,
        num_docs: u32,
        present: &[DocId],
    ) -> crate::Result<VectorIndexReader> {
        vec_write.checkpoint()?;
        let written = HEADER_LEN + vec_write.written_bytes() as usize;
        let file = directory.open_unfinished(&self.path, written)?;
        debug_assert_eq!(file.len(), written);
        let entry = file
            .slice(HEADER_LEN + self.entry.start as usize..HEADER_LEN + self.entry.end as usize);
        let mut id_map = Vec::new();
        IdMap::serialize(present, num_docs, &mut id_map)?;
        let id_map = IdMap::open(FileSlice::from(id_map), num_docs)?;
        VectorIndexReader::flat(entry, opts.clone(), num_docs, id_map)
    }

    /// Each target document's row in the reader [`Self::open_reader`] returns, as the one source
    /// of a merge; `None` for a document without a vector.
    pub(crate) fn source_rows(
        reader: &VectorIndexReader,
        num_docs: u32,
        present: &[DocId],
    ) -> Vec<Option<RowAddress>> {
        let flat = reader
            .flat_rows()
            .expect("the streamed rows are a flat entry");
        let mut source_rows = vec![None; num_docs as usize];
        for (row, &doc) in present.iter().enumerate() {
            source_rows[doc as usize] = Some(RowAddress {
                segment_ord: 0,
                block_address: flat.address(row as u32),
            });
        }
        source_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::{AddOperation, SegmentWriter};
    use crate::schema::{Metric, Schema};
    use crate::vector::{VectorQuantizationConfig, VectorQuantizationLayer};
    use crate::{IndexSettings, TantivyDocument};

    /// `push` refuses a descending document, and a wrong length and a non-finite row with the
    /// errors indexing the document raises.
    #[test]
    fn push_refuses_what_indexing_refuses() -> crate::Result<()> {
        const DIM: usize = 64;
        let options = VectorOptions::new(DIM, Metric::L2);
        let mut schema = Schema::builder();
        let field = schema.add_vector_field("embedding", options.clone());
        let index = Index::builder()
            .schema(schema.build())
            .settings(IndexSettings {
                vector_quantization: vec![VectorQuantizationConfig::materialize(
                    "embedding".to_string(),
                    &options,
                    vec![VectorQuantizationLayer { bits: 1, seed: 7 }],
                )?],
                ..IndexSettings::default()
            })
            .create_in_ram()?;
        let bytes = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        let row = bytes(&[1.5; DIM]);
        // The error indexing raises for `refused` as document 3 of a segment.
        let indexed = |refused: &[u8]| -> String {
            let segment = index.new_segment();
            let mut writer = SegmentWriter::for_segment(15_000_000, segment, true).unwrap();
            let mut error = None;
            for doc in 0..4u64 {
                let mut document = TantivyDocument::new();
                document.add_field_value(field, if doc == 3 { refused } else { &row[..] });
                error = writer
                    .add_document(AddOperation {
                        opstamp: doc,
                        document,
                    })
                    .err();
            }
            error.expect("indexing refuses the row").to_string()
        };

        let mut rows = VectorRowsWriter::create(&index.new_segment(), field)?;
        rows.push(2, &row)?;
        let written = rows.bytes_written();
        for descending in [1, 2] {
            let refused = rows.push(descending, &row).unwrap_err();
            assert!(
                matches!(refused, TantivyError::InvalidArgument(_)),
                "{refused}"
            );
        }
        let mut non_finite = [1.5; DIM];
        non_finite[DIM - 1] = f32::NAN;
        for refused in [bytes(&[1.5; DIM - 1]), bytes(&non_finite)] {
            let error = rows.push(3, &refused).unwrap_err();
            assert_eq!(error.to_string(), indexed(&refused));
        }
        assert_eq!(rows.bytes_written(), written);
        rows.push(3, &row)?;
        assert_eq!(rows.bytes_written(), written + row.len() as u64);
        Ok(())
    }
}
