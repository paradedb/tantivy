use std::any::Any;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};

use common::{OwnedBytes, TerminatingWrite};

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

    /// Number of bytes per binary quantized vector (`dimensions / 8`).
    pub fn bytes_per_vector(&self) -> usize {
        self.bytes_per_vector
    }

    /// Number of dimensions.
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
        let data = file_slice.read_bytes()?;
        BqVecPluginReader::from_bytes(data).map(|r| Arc::new(r) as Arc<dyn PluginReader>)
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
                write.write_all(reader.vector(old_doc_addr.doc_id))?;
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

/// Reader providing O(1) access to binary quantized vectors via mmap.
pub struct BqVecPluginReader {
    /// Raw segment bytes (header + vectors). Backed by mmap.
    data: OwnedBytes,
    dimensions: usize,
    num_vectors: usize,
    bytes_per_vector: usize,
}

impl BqVecPluginReader {
    fn from_bytes(data: OwnedBytes) -> crate::Result<Self> {
        if data.len() < HEADER_LEN {
            return Err(crate::TantivyError::InternalError(format!(
                "bqvec file too short: {} bytes",
                data.len()
            )));
        }
        let dimensions =
            u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let num_vectors =
            u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let bytes_per_vector = dimensions / 8;

        let expected = HEADER_LEN + num_vectors * bytes_per_vector;
        if data.len() < expected {
            return Err(crate::TantivyError::InternalError(format!(
                "bqvec file truncated: expected {expected} bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            dimensions,
            num_vectors,
            bytes_per_vector,
        })
    }

    /// O(1) access to the binary quantized vector for `doc_id`.
    ///
    /// Returns a `bytes_per_vector`-length slice backed by mmap — no copies.
    #[inline]
    pub fn vector(&self, doc_id: DocId) -> &[u8] {
        let offset = HEADER_LEN + (doc_id as usize) * self.bytes_per_vector;
        &self.data[offset..offset + self.bytes_per_vector]
    }

    /// Number of dimensions (bits per vector).
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Number of stored vectors.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Bytes per vector (`dimensions / 8`).
    pub fn bytes_per_vector(&self) -> usize {
        self.bytes_per_vector
    }
}

impl PluginReader for BqVecPluginReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
