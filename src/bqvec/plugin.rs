use std::any::Any;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use common::file_slice::FileSlice;
use common::{HasLen, OwnedBytes};

use crate::directory::{CompositeFile, CompositeWrite};
use crate::index::SegmentComponent;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use crate::schema::document::{Document, Value};
use crate::schema::{Field, FieldType, Schema};
use crate::{DocId, Segment};

/// Per-field header length: [bytes_per_record: u32 LE] [num_records: u32 LE].
const HEADER_LEN: usize = 8;

fn component() -> SegmentComponent {
    SegmentComponent::Custom("bqvec".to_string())
}

/// Encode function that transforms raw f32 vectors into packed byte records.
pub type EncodeFn = Arc<dyn Fn(&[f32]) -> Vec<u8> + Send + Sync>;

/// Per-field configuration for vector encoding.
#[derive(Clone)]
struct VectorFieldConfig {
    field: Field,
    bytes_per_record: usize,
    encode_fn: EncodeFn,
}

/// Builder for constructing a [`BqVecPlugin`] with multi-field support.
pub struct BqVecPluginBuilder {
    fields: Vec<VectorFieldConfig>,
}

impl BqVecPluginBuilder {
    /// Register a vector field with its record size and encoding function.
    ///
    /// `encode_fn` transforms a `&[f32]` vector into a packed byte record of
    /// exactly `bytes_per_record` bytes.
    pub fn vector_field(
        mut self,
        field: Field,
        bytes_per_record: usize,
        encode_fn: EncodeFn,
    ) -> Self {
        assert!(bytes_per_record > 0, "bytes_per_record must be > 0");
        self.fields.push(VectorFieldConfig {
            field,
            bytes_per_record,
            encode_fn,
        });
        self
    }

    /// Build the plugin.
    pub fn build(self) -> BqVecPlugin {
        assert!(
            !self.fields.is_empty(),
            "at least one vector field required"
        );
        BqVecPlugin {
            fields: self.fields,
        }
    }
}

/// Multi-field fixed-size record storage plugin using CompositeFile.
///
/// Each registered vector field gets its own section in a single `.bqvec`
/// composite file. Records are produced by running a user-supplied encode
/// function on the raw f32 vectors extracted from documents.
pub struct BqVecPlugin {
    fields: Vec<VectorFieldConfig>,
}

impl BqVecPlugin {
    /// Start building a new multi-field plugin.
    pub fn builder() -> BqVecPluginBuilder {
        BqVecPluginBuilder { fields: Vec::new() }
    }
}

impl SegmentPlugin for BqVecPlugin {
    fn name(&self) -> &str {
        "bqvec"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["bqvec"]
    }

    fn create_writer(&self, _ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        let per_field: Vec<FieldWriterState> = self
            .fields
            .iter()
            .map(|cfg| FieldWriterState {
                field: cfg.field,
                bytes_per_record: cfg.bytes_per_record,
                encode_fn: cfg.encode_fn.clone(),
                buf: Vec::new(),
            })
            .collect();
        Ok(Box::new(BqVecPluginWriter { per_field }))
    }

    fn open_reader(&self, ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        let file_slice = ctx.segment_reader.open_read(component())?;
        let known_fields: Vec<Field> = self.fields.iter().map(|cfg| cfg.field).collect();
        BqVecPluginReader::open(file_slice, &known_fields)
            .map(|r| Arc::new(r) as Arc<dyn PluginReader>)
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        let source_readers: Vec<Arc<BqVecPluginReader>> = ctx
            .readers
            .iter()
            .map(|r| {
                r.plugin_reader::<BqVecPluginReader>("bqvec")
                    .and_then(|opt| {
                        opt.ok_or_else(|| {
                            crate::TantivyError::InternalError(
                                "bqvec reader missing during merge".into(),
                            )
                        })
                    })
            })
            .collect::<crate::Result<Vec<_>>>()?;

        let num_new_docs = ctx.doc_id_mapping.iter_old_doc_addrs().count();

        let write = ctx.target_segment.open_write(component())?;
        let mut composite = CompositeWrite::wrap(write);

        for cfg in &self.fields {
            let field = cfg.field;
            let bytes_per_record = cfg.bytes_per_record;

            let w = composite.for_field(field);

            // Write per-field header.
            w.write_all(&(bytes_per_record as u32).to_le_bytes())?;
            w.write_all(&(num_new_docs as u32).to_le_bytes())?;

            // Write records in new doc_id order.
            let zero = vec![0u8; bytes_per_record];
            for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                let reader = &source_readers[old_doc_addr.segment_ord as usize];
                if let Some(field_reader) = reader.field_reader(field) {
                    if (old_doc_addr.doc_id as usize) < field_reader.num_records() {
                        let rec = field_reader.record(old_doc_addr.doc_id)?;
                        w.write_all(&rec)?;
                    } else {
                        w.write_all(&zero)?;
                    }
                } else {
                    w.write_all(&zero)?;
                }
            }
            w.flush()?;
        }

        composite.close()?;
        Ok(())
    }
}

/// Per-field writer state held inside [`BqVecPluginWriter`].
struct FieldWriterState {
    field: Field,
    bytes_per_record: usize,
    encode_fn: EncodeFn,
    /// Flat buffer: `buf.len() == num_docs * bytes_per_record`.
    buf: Vec<u8>,
}

impl FieldWriterState {
    fn num_records(&self) -> usize {
        self.buf.len() / self.bytes_per_record
    }
}

/// Writer that accumulates fixed-size records during indexing.
///
/// Vectors are ingested from documents via [`ingest_vectors`](Self::ingest_vectors),
/// which extracts vector field values and runs the per-field encode function.
pub struct BqVecPluginWriter {
    per_field: Vec<FieldWriterState>,
}

impl BqVecPluginWriter {
    /// Extract vector field values from a document and encode them into per-field buffers.
    ///
    /// For each registered vector field, this method:
    /// 1. Looks up the field value in the document
    /// 2. If a vector value is found, runs the encode_fn and appends the result
    /// 3. If no vector value is found, appends zeros
    pub fn ingest_vectors<D: Document>(&mut self, doc: &D, schema: &Schema) {
        for state in &mut self.per_field {
            let mut found = false;
            for (field, value) in doc.iter_fields_and_values() {
                if field != state.field {
                    continue;
                }
                // Check the schema to confirm this is a vector field.
                let field_entry = schema.get_field_entry(field);
                if !matches!(field_entry.field_type(), FieldType::Vector(_)) {
                    continue;
                }
                let value = value.as_value();
                if let Some(vec_data) = value.as_leaf().and_then(|leaf| leaf.as_vector()) {
                    let encoded = (state.encode_fn)(vec_data);
                    debug_assert_eq!(
                        encoded.len(),
                        state.bytes_per_record,
                        "encode_fn returned {} bytes, expected {}",
                        encoded.len(),
                        state.bytes_per_record
                    );
                    state.buf.extend_from_slice(&encoded);
                    found = true;
                    break;
                }
            }
            if !found {
                // No vector value for this field — insert zeros.
                state
                    .buf
                    .resize(state.buf.len() + state.bytes_per_record, 0u8);
            }
        }
    }
}

impl PluginWriter for BqVecPluginWriter {
    fn serialize(
        &mut self,
        segment: &mut Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        let write = segment.open_write(component())?;
        let mut composite = CompositeWrite::wrap(write);

        for state in &self.per_field {
            let num_records = state.num_records();
            let w = composite.for_field(state.field);

            // Per-field header.
            w.write_all(&(state.bytes_per_record as u32).to_le_bytes())?;
            w.write_all(&(num_records as u32).to_le_bytes())?;

            if num_records > 0 {
                if let Some(mapping) = doc_id_map {
                    // Sorted index: write records in remapped order.
                    for old_doc_id in mapping.iter_old_doc_ids() {
                        let start = old_doc_id as usize * state.bytes_per_record;
                        w.write_all(&state.buf[start..start + state.bytes_per_record])?;
                    }
                } else {
                    w.write_all(&state.buf)?;
                }
            }
            w.flush()?;
        }

        composite.close()?;
        Ok(())
    }

    fn close(self: Box<Self>) -> crate::Result<()> {
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.per_field
            .iter()
            .map(|s| s.buf.capacity())
            .sum::<usize>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Per-field reader providing O(1) access to fixed-size records.
pub struct BqVecFieldReader {
    /// File slice positioned past the per-field header.
    data: FileSlice,
    num_records: usize,
    bytes_per_record: usize,
}

impl BqVecFieldReader {
    fn open(file_slice: FileSlice) -> crate::Result<Self> {
        if file_slice.len() < HEADER_LEN {
            return Err(crate::TantivyError::InternalError(format!(
                "bqvec field section too short: {} bytes",
                file_slice.len()
            )));
        }
        let header = file_slice.read_bytes_slice(0..HEADER_LEN)?;
        let bytes_per_record =
            u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let num_records = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

        let expected = HEADER_LEN + num_records * bytes_per_record;
        if file_slice.len() < expected {
            return Err(crate::TantivyError::InternalError(format!(
                "bqvec field section truncated: expected {expected} bytes, got {}",
                file_slice.len()
            )));
        }
        let data = file_slice.slice_from(HEADER_LEN);
        Ok(Self {
            data,
            num_records,
            bytes_per_record,
        })
    }

    /// Read the record for `doc_id`.
    ///
    /// Only the bytes for this single record are read from the underlying
    /// directory — no bulk load of the entire file.
    #[inline]
    pub fn record(&self, doc_id: DocId) -> std::io::Result<OwnedBytes> {
        let offset = (doc_id as usize) * self.bytes_per_record;
        self.data
            .read_bytes_slice(offset..offset + self.bytes_per_record)
    }

    /// Number of stored records.
    pub fn num_records(&self) -> usize {
        self.num_records
    }

    /// Bytes per record.
    pub fn bytes_per_record(&self) -> usize {
        self.bytes_per_record
    }
}

/// Reader providing per-field access to fixed-size records in a CompositeFile.
pub struct BqVecPluginReader {
    field_readers: HashMap<Field, BqVecFieldReader>,
}

impl BqVecPluginReader {
    fn open(file_slice: FileSlice, known_fields: &[Field]) -> crate::Result<Self> {
        let composite = CompositeFile::open(&file_slice)?;
        let mut field_readers = HashMap::new();
        for &field in known_fields {
            if let Some(field_slice) = composite.open_read(field) {
                let reader = BqVecFieldReader::open(field_slice)?;
                field_readers.insert(field, reader);
            }
        }
        Ok(Self { field_readers })
    }

    /// Get a reader for a specific field.
    pub fn field_reader(&self, field: Field) -> Option<&BqVecFieldReader> {
        self.field_readers.get(&field)
    }
}

impl PluginReader for BqVecPluginReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
