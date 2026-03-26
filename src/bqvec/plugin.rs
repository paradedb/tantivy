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

/// Binary quantized vector storage plugin.
///
/// Vectors are staged via [`stage_vector`](Self::stage_vector) before each
/// `IndexWriter::add_document` call. The plugin writer consumes the staged
/// vector in its `add_document` callback.
pub struct BqVecPlugin {
    dimensions: usize,
    bytes_per_vector: usize,
    staging: Mutex<VecDeque<Vec<u8>>>,
}

impl BqVecPlugin {
    /// Create a new plugin for vectors with the given number of dimensions.
    ///
    /// `dimensions` must be a positive multiple of 8.
    pub fn new(dimensions: usize) -> Self {
        assert!(dimensions > 0, "dimensions must be > 0");
        assert!(
            dimensions % 8 == 0,
            "dimensions must be a multiple of 8, got {dimensions}"
        );
        let bytes_per_vector = dimensions / 8;
        Self {
            dimensions,
            bytes_per_vector,
            staging: Mutex::new(VecDeque::new()),
        }
    }

    /// Stage a binary quantized vector for the next `add_document` call.
    ///
    /// The byte slice must be exactly [`bytes_per_vector()`](Self::bytes_per_vector)
    /// bytes long.
    pub fn stage_vector(&self, bq_bytes: Vec<u8>) {
        assert_eq!(
            bq_bytes.len(),
            self.bytes_per_vector,
            "expected {} bytes, got {}",
            self.bytes_per_vector,
            bq_bytes.len()
        );
        self.staging.lock().unwrap().push_back(bq_bytes);
    }

    pub fn bytes_per_vector(&self) -> usize {
        self.bytes_per_vector
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
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
            dimensions: self.dimensions,
            bytes_per_vector: self.bytes_per_vector,
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
        let bytes_per_vector = self.bytes_per_vector;

        // Collect readers from source segments.
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
        write.write_all(&(self.dimensions as u32).to_le_bytes())?;
        write.write_all(&(num_new_docs as u32).to_le_bytes())?;

        // Vectors in new doc_id order.
        let zero = vec![0u8; bytes_per_vector];
        for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
            let reader = &source_readers[old_doc_addr.segment_ord as usize];
            if (old_doc_addr.doc_id as usize) < reader.num_vectors() {
                let vec_bytes = reader.vector(old_doc_addr.doc_id)?;
                write.write_all(&vec_bytes)?;
            } else {
                write.write_all(&zero)?;
            }
        }

        TerminatingWrite::terminate(write)?;
        Ok(())
    }
}

/// Writer that accumulates binary quantized vectors during indexing.
///
/// Vectors arrive through the staging queue on [`BqVecPlugin`]. Each
/// `add_document` call pops one staged vector (or inserts zeros).
pub struct BqVecPluginWriter {
    dimensions: usize,
    bytes_per_vector: usize,
    /// Flat buffer: `buf.len() == num_docs * bytes_per_vector`.
    buf: Vec<u8>,
    /// Local drain of staging queue at writer creation time (for vectors
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
    /// Number of vectors accumulated so far.
    pub fn num_vectors(&self) -> usize {
        self.buf.len() / self.bytes_per_vector
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
        if let Some(bq) = self.pop_staged() {
            debug_assert_eq!(bq.len(), self.bytes_per_vector);
            self.buf.extend_from_slice(&bq);
        } else {
            // No vector staged — insert zeros.
            self.buf
                .resize(self.buf.len() + self.bytes_per_vector, 0u8);
        }
        Ok(())
    }

    fn serialize(
        &mut self,
        segment: &mut Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        let num_vectors = self.num_vectors();
        if num_vectors == 0 {
            // Still write an empty file so the reader doesn't fail.
            let mut write = segment.open_write(component())?;
            write.write_all(&(self.dimensions as u32).to_le_bytes())?;
            write.write_all(&0u32.to_le_bytes())?;
            TerminatingWrite::terminate(write)?;
            return Ok(());
        }

        let mut write = segment.open_write(component())?;
        write.write_all(&(self.dimensions as u32).to_le_bytes())?;
        write.write_all(&(num_vectors as u32).to_le_bytes())?;

        if let Some(mapping) = doc_id_map {
            // Sorted index: write vectors in remapped order.
            for old_doc_id in mapping.iter_old_doc_ids() {
                let start = old_doc_id as usize * self.bytes_per_vector;
                write.write_all(&self.buf[start..start + self.bytes_per_vector])?;
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

pub struct BqVecPluginReader {
    /// File slice positioned at the start of vector data (past the header).
    /// Each `vector()` call reads only the requested range from the
    /// underlying directory, so a postgres-backed directory fetches only
    /// the blocks that contain the requested vector.
    data: FileSlice,
    dimensions: usize,
    num_vectors: usize,
    bytes_per_vector: usize,
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
        let dimensions =
            u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let num_vectors =
            u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let bytes_per_vector = dimensions / 8;

        let expected = HEADER_LEN + num_vectors * bytes_per_vector;
        if file_slice.len() < expected {
            return Err(crate::TantivyError::InternalError(format!(
                "bqvec file truncated: expected {expected} bytes, got {}",
                file_slice.len()
            )));
        }
        // Slice past the header so vector offsets are doc_id * bytes_per_vector.
        let data = file_slice.slice_from(HEADER_LEN);
        Ok(Self {
            data,
            dimensions,
            num_vectors,
            bytes_per_vector,
        })
    }

    /// Read the binary quantized vector for `doc_id`.
    ///
    /// Only the bytes for this single vector are read from the underlying
    /// directory — no bulk load of the entire file.
    #[inline]
    pub fn vector(&self, doc_id: DocId) -> std::io::Result<OwnedBytes> {
        let offset = (doc_id as usize) * self.bytes_per_vector;
        self.data.read_bytes_slice(offset..offset + self.bytes_per_vector)
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn bytes_per_vector(&self) -> usize {
        self.bytes_per_vector
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
    use crate::IndexWriter;
    use crate::Index;

    fn make_schema() -> (Schema, crate::schema::Field) {
        let mut builder = Schema::builder();
        let text = builder.add_text_field("text", TEXT | STORED);
        (builder.build(), text)
    }

    fn assert_vector(bq: &BqVecPluginReader, doc_id: u32, expected: &[u8]) {
        let actual = bq.vector(doc_id).unwrap();
        assert_eq!(&*actual, expected, "vector mismatch for doc_id={doc_id}");
    }

    #[test]
    fn test_bqvec_index_and_read() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let dims = 64;
        let plugin = Arc::new(BqVecPlugin::new(dims));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        plugin.stage_vector(vec![0xFF; 8]);
        writer.add_document(crate::doc!(text_field => "hello"))?;

        plugin.stage_vector(vec![0x00; 8]);
        writer.add_document(crate::doc!(text_field => "world"))?;

        plugin.stage_vector(vec![0xAA; 8]);
        writer.add_document(crate::doc!(text_field => "foo"))?;

        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segments = searcher.segment_readers();
        assert_eq!(segments.len(), 1);

        let bq: Arc<BqVecPluginReader> = segments[0]
            .plugin_reader::<BqVecPluginReader>("bqvec")?
            .expect("bqvec reader should exist");

        assert_eq!(bq.dimensions(), 64);
        assert_eq!(bq.num_vectors(), 3);
        assert_eq!(bq.bytes_per_vector(), 8);

        assert_vector(&bq, 0, &[0xFF; 8]);
        assert_vector(&bq, 1, &[0x00; 8]);
        assert_vector(&bq, 2, &[0xAA; 8]);

        Ok(())
    }

    #[test]
    fn test_bqvec_zero_fill_when_no_vector_staged() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let dims = 16;
        let plugin = Arc::new(BqVecPlugin::new(dims));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        plugin.stage_vector(vec![0xAB, 0xCD]);
        plugin.stage_vector(vec![0x00, 0x00]);
        plugin.stage_vector(vec![0x12, 0x34]);

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

        assert_vector(&bq, 0, &[0xAB, 0xCD]);
        assert_vector(&bq, 1, &[0x00, 0x00]);
        assert_vector(&bq, 2, &[0x12, 0x34]);

        Ok(())
    }

    #[test]
    fn test_bqvec_auto_zero_fill() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let dims = 16;
        let plugin = Arc::new(BqVecPlugin::new(dims));
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

        assert_eq!(bq.num_vectors(), 2);
        assert_vector(&bq, 0, &[0x00, 0x00]);
        assert_vector(&bq, 1, &[0x00, 0x00]);

        Ok(())
    }

    #[test]
    fn test_bqvec_merge() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let dims = 32;
        let plugin = Arc::new(BqVecPlugin::new(dims));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        plugin.stage_vector(vec![0x11; 4]);
        writer.add_document(crate::doc!(text_field => "a"))?;
        plugin.stage_vector(vec![0x22; 4]);
        writer.add_document(crate::doc!(text_field => "b"))?;
        writer.commit()?;

        plugin.stage_vector(vec![0x33; 4]);
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

        assert_eq!(bq.num_vectors(), 3);
        assert_vector(&bq, 0, &[0x11; 4]);
        assert_vector(&bq, 1, &[0x22; 4]);
        assert_vector(&bq, 2, &[0x33; 4]);

        Ok(())
    }

    #[test]
    fn test_bqvec_merge_with_deletes() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let dims = 24;
        let plugin = Arc::new(BqVecPlugin::new(dims));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        plugin.stage_vector(vec![0xAA; 3]);
        writer.add_document(crate::doc!(text_field => "keep_a"))?;
        plugin.stage_vector(vec![0xBB; 3]);
        writer.add_document(crate::doc!(text_field => "deleteme"))?;
        writer.commit()?;

        plugin.stage_vector(vec![0xCC; 3]);
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

        assert_eq!(bq.num_vectors(), 2);
        assert_vector(&bq, 0, &[0xAA; 3]);
        assert_vector(&bq, 1, &[0xCC; 3]);

        Ok(())
    }

    #[test]
    fn test_bqvec_large_dimensions() -> crate::Result<()> {
        let (schema, text_field) = make_schema();

        let dims = 768;
        let plugin = Arc::new(BqVecPlugin::new(dims));
        let index = Index::builder()
            .schema(schema)
            .plugin(plugin.clone() as Arc<dyn SegmentPlugin>)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;

        let vec0: Vec<u8> = (0..96).map(|i| i as u8).collect();
        let vec1: Vec<u8> = (0..96).map(|i| (255 - i) as u8).collect();

        plugin.stage_vector(vec0.clone());
        writer.add_document(crate::doc!(text_field => "doc0"))?;
        plugin.stage_vector(vec1.clone());
        writer.add_document(crate::doc!(text_field => "doc1"))?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let seg = &searcher.segment_readers()[0];
        let bq: Arc<BqVecPluginReader> = seg
            .plugin_reader::<BqVecPluginReader>("bqvec")?
            .unwrap();

        assert_eq!(bq.dimensions(), 768);
        assert_eq!(bq.bytes_per_vector(), 96);
        assert_vector(&bq, 0, &vec0);
        assert_vector(&bq, 1, &vec1);

        Ok(())
    }

    #[test]
    fn test_bqvec_plugin_metadata() {
        let plugin = BqVecPlugin::new(128);
        assert_eq!(plugin.name(), "bqvec");
        assert_eq!(plugin.extensions(), vec!["bqvec"]);
        assert_eq!(plugin.write_phase(), 2);
        assert_eq!(plugin.dimensions(), 128);
        assert_eq!(plugin.bytes_per_vector(), 16);
    }
}
