use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Mutex;

use common::TerminatingWrite;

use crate::directory::WritePtr;
use crate::fieldnorm::FieldNormsSerializer;
use crate::index::{Segment, SegmentComponent};
use crate::postings::InvertedIndexSerializer;
use crate::store::StoreWriter;
use crate::TantivyError;

/// Segment serializer is in charge of laying out on disk
/// the data accumulated and sorted by the `SegmentWriter`.
pub struct SegmentSerializer {
    segment: Segment,
    pub(crate) store_writer: StoreWriter,
    fast_field_write: WritePtr,
    fieldnorms_serializer: Option<FieldNormsSerializer>,
    postings_serializer: InvertedIndexSerializer,
}

impl SegmentSerializer {
    /// Creates a new `SegmentSerializer`.
    pub fn for_segment(mut segment: Segment) -> crate::Result<SegmentSerializer> {
        let settings = segment.index().settings().clone();
        let store_writer = {
            let store_write = segment.open_write(SegmentComponent::Store)?;
            StoreWriter::new(
                store_write,
                settings.docstore_compression,
                settings.docstore_blocksize,
                settings.docstore_compress_dedicated_thread,
            )?
        };

        let fast_field_write = segment.open_write(SegmentComponent::FastFields)?;

        let fieldnorms_write = segment.open_write(SegmentComponent::FieldNorms)?;
        let fieldnorms_serializer = FieldNormsSerializer::from_write(fieldnorms_write)?;

        let postings_serializer = InvertedIndexSerializer::open(&mut segment)?;
        Ok(SegmentSerializer {
            segment,
            store_writer,
            fast_field_write,
            fieldnorms_serializer: Some(fieldnorms_serializer),
            postings_serializer,
        })
    }

    /// The memory used (inclusive childs)
    pub fn mem_usage(&self) -> usize {
        self.store_writer.mem_usage()
    }

    pub fn segment(&self) -> &Segment {
        &self.segment
    }

    /// Accessor to the `PostingsSerializer`.
    pub fn get_postings_serializer(&mut self) -> &mut InvertedIndexSerializer {
        &mut self.postings_serializer
    }

    /// Accessor to the `FastFieldSerializer`.
    pub fn get_fast_field_write(&mut self) -> &mut WritePtr {
        &mut self.fast_field_write
    }

    /// Extract the field norm serializer.
    ///
    /// Note the fieldnorms serializer can only be extracted once.
    pub fn extract_fieldnorms_serializer(&mut self) -> Option<FieldNormsSerializer> {
        self.fieldnorms_serializer.take()
    }

    /// Accessor to the `StoreWriter`.
    pub fn get_store_writer(&mut self) -> &mut StoreWriter {
        &mut self.store_writer
    }

    /// Finalize the segment serialization.
    pub fn close(mut self) -> crate::Result<()> {
        if let Some(fieldnorms_serializer) = self.extract_fieldnorms_serializer() {
            fieldnorms_serializer.close()?;
        }
        self.fast_field_write.terminate()?;
        self.postings_serializer.close()?;
        self.store_writer.close()?;
        Ok(())
    }

    pub fn get_path_lookup_writer(&mut self) -> crate::Result<Box<dyn Write>> {
        // First, open the write object. Note that open_write(...) returns `Result<BufWriter<Box<dyn TerminatingWrite>>, OpenWriteError>`.
        // Next, map the error into TantivyError if needed, and then box the writer.
        let bufwriter = self
            .segment
            .open_write(SegmentComponent::PathLookup)
            .map_err(TantivyError::from)?;
        Ok(Box::new(bufwriter) as Box<dyn Write>)
    }

    /// Loads the path_lookup from the segment.
    pub fn load_path_lookup(&self) -> crate::Result<InMemoryPathLookup> {
        let path_data = self
            .segment
            .open_read(SegmentComponent::PathLookup)
            .map_err(TantivyError::from)?;

        // Read the file-slice into a Vec<u8>.
        let data = path_data.read_bytes()?;
        let path_map: HashMap<u32, String> = serde_json::from_slice(&data).map_err(|e| {
            crate::TantivyError::InternalError(format!("Invalid path lookup JSON: {e}"))
        })?;

        let mut in_mem_path_lookup = InMemoryPathLookup::new();
        for (id, path) in path_map {
            in_mem_path_lookup.insert_path(id, &path);
        }
        Ok(in_mem_path_lookup)
    }
}

/// A trait for path lookup: from a path_id (u32) => we get the actual string path (e.g. `address.city`)
pub trait PathLookup: Send + Sync {
    fn path_id_to_str(&self, path_id: u32) -> Option<String>;
    fn path_str_to_id(&self, path_str: &str) -> Option<u32>;
}

/// A thread‐safe in‐memory implementation that holds both path->id and id->path mappings.
/// For simplicity, we store them in `Mutex<HashMap<...>>`.
#[derive(Default)]
pub struct InMemoryPathLookup {
    /// Map from path_id => the actual path string
    pub id_to_path: Mutex<HashMap<u32, String>>,
    /// Map from path string => path_id (so we can re‐use existing IDs if the path repeats)
    pub path_to_id: Mutex<HashMap<String, u32>>,
    /// Next ID to assign
    pub next_id: std::sync::atomic::AtomicU32,
}

impl PathLookup for InMemoryPathLookup {
    fn path_id_to_str(&self, path_id: u32) -> Option<String> {
        println!(
            "[InMemoryPathLookup::path_id_to_str] Looking up path_id = {:?}",
            path_id
        );
        let map_read = self.id_to_path.lock().unwrap();
        let res = map_read.get(&path_id).cloned();
        println!(
            "[InMemoryPathLookup::path_id_to_str] Found = {:?}",
            res.as_ref().map(String::as_str)
        );
        res
    }

    fn path_str_to_id(&self, path_str: &str) -> Option<u32> {
        let map_read = self.path_to_id.lock().unwrap();
        let res = map_read.get(path_str).cloned();
        println!(
            "[InMemoryPathLookup::path_str_to_id] Found = {:?}",
            res.as_ref()
        );
        res
    }
}

impl InMemoryPathLookup {
    pub fn new() -> Self {
        println!("[InMemoryPathLookup::new] Creating new instance.");
        InMemoryPathLookup {
            id_to_path: Mutex::new(HashMap::new()),
            path_to_id: Mutex::new(HashMap::new()),
            next_id: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Assigns or retrieves an ID for the given path string.
    /// If the path was previously assigned an ID, that ID is re‐used.
    /// Otherwise, we generate a new ID.
    pub fn get_or_create_id_for_path(&self, path_str: &str) -> u32 {
        println!(
            "[InMemoryPathLookup::get_or_create_id_for_path] Called with path_str = {:?}",
            path_str
        );
        // Check if path already exists
        {
            let map_read = self.path_to_id.lock().unwrap();
            if let Some(&existing_id) = map_read.get(path_str) {
                println!(
                    "  [InMemoryPathLookup::get_or_create_id_for_path] path already assigned => {:?}",
                    existing_id
                );
                return existing_id;
            }
        }
        // Otherwise generate a new ID
        let new_id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        println!(
            "  [InMemoryPathLookup::get_or_create_id_for_path] path not found => generating new_id = {:?}",
            new_id
        );
        {
            let mut map_write = self.path_to_id.lock().unwrap();
            map_write.insert(path_str.to_string(), new_id);
        }
        let mut id_map_write = self.id_to_path.lock().unwrap();
        id_map_write.insert(new_id, path_str.to_string());
        println!(
            "  [InMemoryPathLookup::get_or_create_id_for_path] Inserted into maps => new_id = {:?}, path_str = {:?}",
            new_id, path_str
        );
        new_id
    }

    /// Optional convenience: store a known path_id => path_str directly
    /// (this is sometimes used if you already know your path_id).
    pub fn insert_path(&self, path_id: u32, path_str: &str) {
        println!(
            "[InMemoryPathLookup::insert_path] Called with path_id = {:?}, path_str = {:?}",
            path_id, path_str
        );
        let mut id_map = self.id_to_path.lock().unwrap();
        id_map.insert(path_id, path_str.to_string());
        let mut path_map = self.path_to_id.lock().unwrap();
        path_map.insert(path_str.to_string(), path_id);
        println!(
            "[InMemoryPathLookup::insert_path] Completed insertion => path_id = {:?}, path_str = {:?}",
            path_id, path_str
        );
    }
}
