use std::any::Any;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};

use common::file_slice::FileSlice;
use common::{HasLen, OwnedBytes, TerminatingWrite};

use crate::index::SegmentComponent;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::{
    PluginMergeContext, PluginReader, PluginReaderContext, PluginWriter, PluginWriterContext,
    SegmentPlugin,
};
use crate::{DocId, Segment};

const HEADER_LEN: usize = 8;

fn component() -> SegmentComponent {
    SegmentComponent::Custom("bqvec".to_string())
}

/// Fixed-size record storage plugin.
///
/// Records are staged via [`stage_record`](Self::stage_record) before each
/// `IndexWriter::add_document` call. The plugin writer consumes the staged
/// record in its `add_document` callback.
pub struct BqVecPlugin {
    bytes_per_record: usize,
    staging: Mutex<VecDeque<Vec<u8>>>,
}

impl BqVecPlugin {
    /// Create a new plugin that stores `bytes_per_record` bytes per document.
    ///
    /// The record content is opaque — the caller decides the layout.
    pub fn new(bytes_per_record: usize) -> Self {
        assert!(bytes_per_record > 0, "bytes_per_record must be > 0");
        Self {
            bytes_per_record,
            staging: Mutex::new(VecDeque::new()),
        }
    }

    /// Stage a record for the next `add_document` call.
    ///
    /// The byte slice must be exactly [`bytes_per_record()`](Self::bytes_per_record)
    /// bytes long.
    pub fn stage_record(&self, record: Vec<u8>) {
        assert_eq!(
            record.len(),
            self.bytes_per_record,
            "expected {} bytes, got {}",
            self.bytes_per_record,
            record.len()
        );
        self.staging.lock().unwrap().push_back(record);
    }

    /// Number of bytes per record.
    pub fn bytes_per_record(&self) -> usize {
        self.bytes_per_record
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
        Ok(Box::new(BqVecPluginWriter {
            bytes_per_record: self.bytes_per_record,
            buf: Vec::new(),
            staging: self.staging.lock().unwrap().drain(..).collect(),
            staging_ref: &self.staging as *const Mutex<VecDeque<Vec<u8>>>,
        }))
    }

    fn open_reader(&self, ctx: &PluginReaderContext) -> crate::Result<Arc<dyn PluginReader>> {
        let file_slice = ctx.segment_reader.open_read(component())?;
        BqVecPluginReader::open(file_slice).map(|r| Arc::new(r) as Arc<dyn PluginReader>)
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        let bytes_per_record = self.bytes_per_record;

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

        let mut write = ctx.target_segment.open_write(component())?;

        // Header.
        write.write_all(&(bytes_per_record as u32).to_le_bytes())?;
        write.write_all(&(num_new_docs as u32).to_le_bytes())?;

        // Records in new doc_id order.
        let zero = vec![0u8; bytes_per_record];
        for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
            let reader = &source_readers[old_doc_addr.segment_ord as usize];
            if (old_doc_addr.doc_id as usize) < reader.num_records() {
                let rec = reader.record(old_doc_addr.doc_id)?;
                write.write_all(&rec)?;
            } else {
                write.write_all(&zero)?;
            }
        }

        TerminatingWrite::terminate(write)?;
        Ok(())
    }
}

/// Writer that accumulates fixed-size records during indexing.
///
/// Records arrive through the staging queue on [`BqVecPlugin`]. Each
/// `add_document` call pops one staged record (or inserts zeros).
pub struct BqVecPluginWriter {
    bytes_per_record: usize,
    /// Flat buffer: `buf.len() == num_docs * bytes_per_record`.
    buf: Vec<u8>,
    /// Local drain of staging queue at writer creation time (for records
    /// staged before the writer was created — rare but possible).
    staging: VecDeque<Vec<u8>>,
    /// Raw pointer to the plugin's staging mutex. SAFETY: the plugin (and its
    /// staging queue) outlives every writer it creates — the `Arc<dyn SegmentPlugin>`
    /// in `Index` keeps it alive for the entire index lifetime.
    staging_ref: *const Mutex<VecDeque<Vec<u8>>>,
}

// SAFETY: staging_ref points to data behind an Arc that outlives the writer.
unsafe impl Send for BqVecPluginWriter {}

impl BqVecPluginWriter {
    /// Number of records accumulated so far.
    pub fn num_records(&self) -> usize {
        self.buf.len() / self.bytes_per_record
    }

    fn pop_staged(&mut self) -> Option<Vec<u8>> {
        // Try local queue first, then shared.
        if let Some(v) = self.staging.pop_front() {
            return Some(v);
        }
        // SAFETY: see field doc.
        let mutex = unsafe { &*self.staging_ref };
        mutex.lock().unwrap().pop_front()
    }
}

impl PluginWriter for BqVecPluginWriter {
    fn add_document(&mut self, _doc_id: DocId) -> crate::Result<()> {
        if let Some(rec) = self.pop_staged() {
            debug_assert_eq!(rec.len(), self.bytes_per_record);
            self.buf.extend_from_slice(&rec);
        } else {
            // No record staged — insert zeros.
            self.buf
                .resize(self.buf.len() + self.bytes_per_record, 0u8);
        }
        Ok(())
    }

    fn serialize(
        &mut self,
        segment: &mut Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        let num_records = self.num_records();

        let mut write = segment.open_write(component())?;
        write.write_all(&(self.bytes_per_record as u32).to_le_bytes())?;
        write.write_all(&(num_records as u32).to_le_bytes())?;

        if num_records == 0 {
            TerminatingWrite::terminate(write)?;
            return Ok(());
        }

        if let Some(mapping) = doc_id_map {
            // Sorted index: write records in remapped order.
            for old_doc_id in mapping.iter_old_doc_ids() {
                let start = old_doc_id as usize * self.bytes_per_record;
                write.write_all(&self.buf[start..start + self.bytes_per_record])?;
            }
        } else {
            write.write_all(&self.buf)?;
        }

        TerminatingWrite::terminate(write)?;
        Ok(())
    }

    fn close(self: Box<Self>) -> crate::Result<()> {
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.buf.capacity() + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Reader providing O(1) access to fixed-size records.
///
/// Each [`record()`](Self::record) call reads only the requested byte range
/// from the underlying directory.
pub struct BqVecPluginReader {
    /// File slice positioned at the start of record data (past the header).
    data: FileSlice,
    num_records: usize,
    bytes_per_record: usize,
}

impl BqVecPluginReader {
    fn open(file_slice: FileSlice) -> crate::Result<Self> {
        if file_slice.len() < HEADER_LEN {
            return Err(crate::TantivyError::InternalError(format!(
                "bqvec file too short: {} bytes",
                file_slice.len()
            )));
        }
        // Read only the 8-byte header.
        let header = file_slice.read_bytes_slice(0..HEADER_LEN)?;
        let bytes_per_record =
            u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let num_records =
            u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

        let expected = HEADER_LEN + num_records * bytes_per_record;
        if file_slice.len() < expected {
            return Err(crate::TantivyError::InternalError(format!(
                "bqvec file truncated: expected {expected} bytes, got {}",
                file_slice.len()
            )));
        }
        // Slice past the header so record offsets are doc_id * bytes_per_record.
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

impl PluginReader for BqVecPluginReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::bqvec::{BqVecPlugin, BqVecPluginReader};
    use crate::plugin::SegmentPlugin;
    use crate::schema::{Schema, STORED, TEXT};
    use crate::{Index, IndexWriter};

    fn make_schema() -> (Schema, crate::schema::Field) {
        let mut builder = Schema::builder();
        let text = builder.add_text_field("text", TEXT | STORED);
        (builder.build(), text)
    }

    fn assert_record(reader: &BqVecPluginReader, doc_id: u32, expected: &[u8]) {
        let actual = reader.record(doc_id).unwrap();
        assert_eq!(&*actual, expected, "record mismatch for doc_id={doc_id}");
    }

    #[test]
    fn test_index_and_read() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let plugin = Arc::new(BqVecPlugin::new(8));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        plugin.stage_record(vec![0xFF; 8]);
        writer.add_document(crate::doc!(text_field => "hello"))?;

        plugin.stage_record(vec![0x00; 8]);
        writer.add_document(crate::doc!(text_field => "world"))?;

        plugin.stage_record(vec![0xAA; 8]);
        writer.add_document(crate::doc!(text_field => "foo"))?;

        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segments = searcher.segment_readers();
        assert_eq!(segments.len(), 1);

        let bq: Arc<BqVecPluginReader> = segments[0]
            .plugin_reader::<BqVecPluginReader>("bqvec")?
            .expect("bqvec reader should exist");

        assert_eq!(bq.num_records(), 3);
        assert_eq!(bq.bytes_per_record(), 8);

        assert_record(&bq, 0, &[0xFF; 8]);
        assert_record(&bq, 1, &[0x00; 8]);
        assert_record(&bq, 2, &[0xAA; 8]);

        Ok(())
    }

    #[test]
    fn test_staged_ordering() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let plugin = Arc::new(BqVecPlugin::new(2));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        plugin.stage_record(vec![0xAB, 0xCD]);
        plugin.stage_record(vec![0x00, 0x00]);
        plugin.stage_record(vec![0x12, 0x34]);

        writer.add_document(crate::doc!(text_field => "a"))?;
        writer.add_document(crate::doc!(text_field => "b"))?;
        writer.add_document(crate::doc!(text_field => "c"))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let seg = &searcher.segment_readers()[0];
        let bq: Arc<BqVecPluginReader> = seg
            .plugin_reader::<BqVecPluginReader>("bqvec")?
            .unwrap();

        assert_record(&bq, 0, &[0xAB, 0xCD]);
        assert_record(&bq, 1, &[0x00, 0x00]);
        assert_record(&bq, 2, &[0x12, 0x34]);

        Ok(())
    }

    #[test]
    fn test_auto_zero_fill() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let plugin = Arc::new(BqVecPlugin::new(2));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        writer.add_document(crate::doc!(text_field => "a"))?;
        writer.add_document(crate::doc!(text_field => "b"))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let seg = &searcher.segment_readers()[0];
        let bq: Arc<BqVecPluginReader> = seg
            .plugin_reader::<BqVecPluginReader>("bqvec")?
            .unwrap();

        assert_eq!(bq.num_records(), 2);
        assert_record(&bq, 0, &[0x00, 0x00]);
        assert_record(&bq, 1, &[0x00, 0x00]);

        Ok(())
    }

    #[test]
    fn test_merge() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let plugin = Arc::new(BqVecPlugin::new(4));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        plugin.stage_record(vec![0x11; 4]);
        writer.add_document(crate::doc!(text_field => "a"))?;
        plugin.stage_record(vec![0x22; 4]);
        writer.add_document(crate::doc!(text_field => "b"))?;
        writer.commit()?;

        plugin.stage_record(vec![0x33; 4]);
        writer.add_document(crate::doc!(text_field => "c"))?;
        writer.commit()?;

        let segment_ids = index.searchable_segment_ids()?;
        assert_eq!(segment_ids.len(), 2);
        writer.merge(&segment_ids).wait()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.num_docs(), 3);

        let segments = searcher.segment_readers();
        assert_eq!(segments.len(), 1);

        let bq: Arc<BqVecPluginReader> = segments[0]
            .plugin_reader::<BqVecPluginReader>("bqvec")?
            .unwrap();

        assert_eq!(bq.num_records(), 3);
        assert_record(&bq, 0, &[0x11; 4]);
        assert_record(&bq, 1, &[0x22; 4]);
        assert_record(&bq, 2, &[0x33; 4]);

        Ok(())
    }

    #[test]
    fn test_merge_with_deletes() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let plugin = Arc::new(BqVecPlugin::new(3));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        plugin.stage_record(vec![0xAA; 3]);
        writer.add_document(crate::doc!(text_field => "keep_a"))?;
        plugin.stage_record(vec![0xBB; 3]);
        writer.add_document(crate::doc!(text_field => "deleteme"))?;
        writer.commit()?;

        plugin.stage_record(vec![0xCC; 3]);
        writer.add_document(crate::doc!(text_field => "keep_b"))?;
        writer.commit()?;

        writer.delete_term(crate::Term::from_field_text(text_field, "deleteme"));
        writer.commit()?;

        let segment_ids = index.searchable_segment_ids()?;
        assert_eq!(segment_ids.len(), 2);
        writer.merge(&segment_ids).wait()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.num_docs(), 2);

        let segments = searcher.segment_readers();
        assert_eq!(segments.len(), 1);

        let bq: Arc<BqVecPluginReader> = segments[0]
            .plugin_reader::<BqVecPluginReader>("bqvec")?
            .unwrap();

        assert_eq!(bq.num_records(), 2);
        assert_record(&bq, 0, &[0xAA; 3]);
        assert_record(&bq, 1, &[0xCC; 3]);

        Ok(())
    }

    /// Test with a RaBitQ-style record: binary code + correction scalars.
    #[test]
    fn test_rabitq_record_layout() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        // 64-dim binary code (8 bytes) + norm f32 (4 bytes) + x_bar f32 (4 bytes) = 16 bytes
        let bytes_per_record = 8 + 4 + 4;
        let plugin = Arc::new(BqVecPlugin::new(bytes_per_record));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        // Build a record: binary code + norm + x_bar
        let mut rec0 = vec![0xAA; 8]; // binary code
        rec0.extend_from_slice(&1.5f32.to_le_bytes()); // norm
        rec0.extend_from_slice(&0.25f32.to_le_bytes()); // x_bar

        let mut rec1 = vec![0x55; 8];
        rec1.extend_from_slice(&2.0f32.to_le_bytes());
        rec1.extend_from_slice(&(-0.1f32).to_le_bytes());

        plugin.stage_record(rec0.clone());
        writer.add_document(crate::doc!(text_field => "doc0"))?;
        plugin.stage_record(rec1.clone());
        writer.add_document(crate::doc!(text_field => "doc1"))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let seg = &searcher.segment_readers()[0];
        let bq: Arc<BqVecPluginReader> = seg
            .plugin_reader::<BqVecPluginReader>("bqvec")?
            .unwrap();

        assert_eq!(bq.bytes_per_record(), 16);

        // Read back and parse the record.
        let data0 = bq.record(0)?;
        assert_eq!(&data0[..8], &[0xAA; 8]);
        let norm0 = f32::from_le_bytes([data0[8], data0[9], data0[10], data0[11]]);
        let xbar0 = f32::from_le_bytes([data0[12], data0[13], data0[14], data0[15]]);
        assert_eq!(norm0, 1.5);
        assert_eq!(xbar0, 0.25);

        let data1 = bq.record(1)?;
        assert_eq!(&data1[..8], &[0x55; 8]);
        let norm1 = f32::from_le_bytes([data1[8], data1[9], data1[10], data1[11]]);
        let xbar1 = f32::from_le_bytes([data1[12], data1[13], data1[14], data1[15]]);
        assert_eq!(norm1, 2.0);
        assert_eq!(xbar1, -0.1);

        Ok(())
    }

    #[test]
    fn test_plugin_metadata() {
        let plugin = BqVecPlugin::new(104);
        assert_eq!(plugin.name(), "bqvec");
        assert_eq!(plugin.extensions(), vec!["bqvec"]);
        assert_eq!(plugin.write_phase(), 2);
        assert_eq!(plugin.bytes_per_record(), 104);
    }
}
