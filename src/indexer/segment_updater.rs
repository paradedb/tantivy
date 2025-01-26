use std::borrow::BorrowMut;
use std::collections::HashSet;
use std::io::Write;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use rayon::{ThreadPool, ThreadPoolBuilder};

use super::segment_manager::SegmentManager;
use crate::core::META_FILEPATH;
use crate::directory::{Directory, DirectoryClone, DirectoryPanicHandler, GarbageCollectionResult};
use crate::fastfield::AliveBitSet;
use crate::index::{Index, IndexMeta, IndexSettings, Segment, SegmentId, SegmentMeta};
use crate::indexer::delete_queue::DeleteCursor;
use crate::indexer::index_writer::advance_deletes;
use crate::indexer::merge_operation::MergeOperationInventory;
use crate::indexer::merger::IndexMerger;
use crate::indexer::segment_manager::SegmentsStatus;
use crate::indexer::stamper::Stamper;
use crate::indexer::{
    DefaultMergePolicy, MergeCandidate, MergeOperation, MergePolicy, SegmentEntry,
    SegmentSerializer,
};
use crate::{FutureResult, Opstamp};

/// Save the index meta file.
/// This operation is atomic:
/// Either
/// - it fails, in which case an error is returned, and the `meta.json` remains untouched,
/// - it succeeds, and `meta.json` is written and flushed.
///
/// This method is not part of tantivy's public API
pub(crate) fn save_metas(
    metas: &IndexMeta,
    previous_metas: &IndexMeta,
    directory: &dyn Directory,
) -> crate::Result<()> {
    println!("save_metas: Starting to save metas.");
    info!("save metas");

    match directory.save_metas(metas, previous_metas, &mut ()) {
        Ok(_) => {
            println!("save_metas: Metas saved successfully using directory.save_metas.");
            Ok(())
        }
        Err(crate::TantivyError::InternalError(e)) => {
            println!("save_metas: Caught InternalError: {e:?} Attempting fallback save method.");
            let mut buffer = serde_json::to_vec_pretty(metas)?;
            // Just adding a new line at the end of the buffer.
            writeln!(&mut buffer)?;
            crate::fail_point!("save_metas", |msg| {
                println!("save_metas: Fail point triggered with message: {:?}", msg);
                Err(crate::TantivyError::from(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    msg.unwrap_or_else(|| "Undefined".to_string()),
                )))
            });
            println!("save_metas: Syncing directory after writing buffer.");
            directory.sync_directory()?;
            println!(
                "save_metas: Writing buffer to META_FILEPATH: {:?}",
                META_FILEPATH
            );
            directory.atomic_write(&META_FILEPATH, &buffer[..])?;
            debug!(
                "save_metas: Saved metas: {:?}",
                serde_json::to_string_pretty(&metas)
            );
            println!("save_metas: Fallback save method completed successfully.");
            Ok(())
        }
        Err(e) => {
            println!("save_metas: Encountered error: {:?}", e);
            Err(e)
        }
    }
}

// The segment update runner is in charge of processing all
//  of the `SegmentUpdate`s.
//
// All this processing happens on a single thread
// consuming a common queue.
//
// We voluntarily pass a merge_operation ref to guarantee that
// the merge_operation is alive during the process
#[derive(Clone)]
pub(crate) struct SegmentUpdater(Arc<InnerSegmentUpdater>);

impl Deref for SegmentUpdater {
    type Target = InnerSegmentUpdater;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn garbage_collect_files(
    segment_updater: SegmentUpdater,
) -> crate::Result<GarbageCollectionResult> {
    println!("garbage_collect_files: Starting garbage collection.");
    info!("Running garbage collection");
    let mut index = segment_updater.index.clone();
    println!(
        "garbage_collect_files: Initiating garbage collection on directory: {:?}",
        index.directory()
    );
    let result = index.directory_mut().garbage_collect(move || {
        println!("garbage_collect_files: Listing files for garbage collection.");
        segment_updater.list_files()
    })?;
    Ok(result)
}

/// Merges a list of segments the list of segment givens in the `segment_entries`.
/// This function happens in the calling thread and is computationally expensive.
fn merge(
    index: &Index,
    mut segment_entries: Vec<SegmentEntry>,
    target_opstamp: Opstamp,
) -> crate::Result<Option<SegmentEntry>> {
    println!(
        "merge: Starting merge with target_opstamp: {} and {} segment entries.",
        target_opstamp,
        segment_entries.len()
    );
    let num_docs: u64 = segment_entries
        .iter()
        .map(|segment| segment.meta().num_docs() as u64)
        .sum();
    println!("merge: Total number of documents to merge: {}", num_docs);
    if num_docs == 0 {
        println!("merge: No documents to merge. Exiting merge function.");
        return Ok(None);
    }

    // first we need to apply deletes to our segment.
    let merged_segment = index.new_segment();
    println!(
        "merge: Created new merged segment with ID: {:?}",
        merged_segment.id()
    );

    // First we apply all of the delete to the merged segment, up to the target opstamp.
    for (i, segment_entry) in segment_entries.iter_mut().enumerate() {
        println!(
            "merge: Applying deletes to segment {} with ID: {:?}",
            i,
            segment_entry.meta().id()
        );
        let segment = index.segment(segment_entry.meta().clone());
        advance_deletes(segment, segment_entry, target_opstamp)?;
        println!(
            "merge: Deletes applied to segment {} with ID: {:?}",
            i,
            segment_entry.meta().id()
        );
    }

    let delete_cursor = segment_entries[0].delete_cursor().clone();
    println!("merge: Cloned delete cursor from the first segment entry.");

    let segments: Vec<Segment> = segment_entries
        .iter()
        .map(|segment_entry| {
            println!(
                "merge: Retrieving segment for SegmentMeta ID: {:?}",
                segment_entry.meta().id()
            );
            index.segment(segment_entry.meta().clone())
        })
        .collect();
    println!("merge: Collected all segments for merging.");

    // An IndexMerger is like a "view" of our merged segments.
    let merger: IndexMerger = IndexMerger::open(index.schema(), &segments[..])?;
    println!("merge: Initialized IndexMerger.");

    // ... we just serialize this index merger in our new segment to merge the segments.
    let segment_serializer = SegmentSerializer::for_segment(merged_segment.clone())?;
    println!(
        "merge: Created SegmentSerializer for merged segment ID: {:?}",
        merged_segment.id()
    );

    let num_docs_written = merger.write(segment_serializer)?;
    println!(
        "merge: Written {} documents to the merged segment ID: {:?}",
        num_docs_written,
        merged_segment.id()
    );

    let merged_segment_id = merged_segment.id();
    println!(
        "merge: Merged segment ID is {:?}, number of documents written: {}",
        merged_segment_id, num_docs_written
    );

    let segment_meta = index.new_segment_meta(merged_segment_id, num_docs_written);
    println!(
        "merge: Created SegmentMeta for merged segment ID: {:?}",
        merged_segment_id
    );
    Ok(Some(SegmentEntry::new(segment_meta, delete_cursor, None)))
}

/// Advanced: Merges a list of segments from different indices in a new index.
///
/// Returns `TantivyError` if the indices list is empty or their
/// schemas don't match.
///
/// `output_directory`: is assumed to be empty.
///
/// # Warning
/// This function does NOT check or take the `IndexWriter` is running. It is not
/// meant to work if you have an `IndexWriter` running for the origin indices, or
/// the destination `Index`.
#[doc(hidden)]
pub fn merge_indices<T: Into<Box<dyn Directory>>>(
    indices: &[Index],
    output_directory: T,
) -> crate::Result<Index> {
    println!(
        "merge_indices: Starting to merge {} indices into directory ",
        indices.len(),
    );
    if indices.is_empty() {
        println!("merge_indices: No indices provided to merge. Returning error.");
        // If there are no indices to merge, there is no need to do anything.
        return Err(crate::TantivyError::InvalidArgument(
            "No indices given to merge".to_string(),
        ));
    }

    let target_settings = indices[0].settings().clone();
    println!(
        "merge_indices: Target index settings cloned from first index: {:?}",
        target_settings
    );

    // let's check that all of the indices have the same index settings
    if indices
        .iter()
        .skip(1)
        .any(|index| index.settings() != &target_settings)
    {
        println!(
            "merge_indices: Mismatch in index settings among provided indices. Returning error."
        );
        return Err(crate::TantivyError::InvalidArgument(
            "Attempt to merge indices with different index_settings".to_string(),
        ));
    }

    let mut segments: Vec<Segment> = Vec::new();
    for (i, index) in indices.iter().enumerate() {
        println!(
            "merge_indices: Collecting searchable segments from index {}.",
            i
        );
        segments.extend(index.searchable_segments()?);
        println!(
            "merge_indices: Collected {} segments from index {}.",
            segments.len(),
            i
        );
    }

    let non_filter = segments.iter().map(|_| None).collect::<Vec<_>>();
    println!(
        "merge_indices: Prepared non_filter vector with {} elements.",
        non_filter.len()
    );
    merge_filtered_segments(&segments, target_settings, non_filter, output_directory)
}

/// Advanced: Merges a list of segments from different indices in a new index.
/// Additionally, you can provide a delete bitset for each segment to ignore doc_ids.
///
/// Returns `TantivyError` if the indices list is empty or their
/// schemas don't match.
///
/// `output_directory`: is assumed to be empty.
///
/// # Warning
/// This function does NOT check or take the `IndexWriter` is running. It is not
/// meant to work if you have an `IndexWriter` running for the origin indices, or
/// the destination `Index`.
#[doc(hidden)]
pub fn merge_filtered_segments<T: Into<Box<dyn Directory>>>(
    segments: &[Segment],
    target_settings: IndexSettings,
    filter_doc_ids: Vec<Option<AliveBitSet>>,
    output_directory: T,
) -> crate::Result<Index> {
    println!(
        "merge_filtered_segments: Starting to merge {} segments into directory",
        segments.len(),
    );
    if segments.is_empty() {
        println!("merge_filtered_segments: No segments provided to merge. Returning error.");
        // If there are no indices to merge, there is no need to do anything.
        return Err(crate::TantivyError::InvalidArgument(
            "No segments given to merge".to_string(),
        ));
    }

    let target_schema = segments[0].schema();
    println!(
        "merge_filtered_segments: Target schema cloned from first segment: {:?}",
        target_schema
    );

    // let's check that all of the indices have the same schema
    if segments
        .iter()
        .skip(1)
        .any(|index| index.schema() != target_schema)
    {
        println!(
            "merge_filtered_segments: Mismatch in schemas among provided segments. Returning error."
        );
        return Err(crate::TantivyError::InvalidArgument(
            "Attempt to merge different schema indices".to_string(),
        ));
    }

    println!(
        "merge_filtered_segments: Creating new index with provided directory, schema, and settings."
    );
    let mut merged_index = Index::create(
        output_directory,
        target_schema.clone(),
        target_settings.clone(),
    )?;
    println!("merge_filtered_segments: Created new index. Initializing merged segment.");
    let merged_segment = merged_index.new_segment();
    let merged_segment_id = merged_segment.id();
    println!(
        "merge_filtered_segments: Created merged segment with ID: {:?}",
        merged_segment_id
    );
    let merger: IndexMerger =
        IndexMerger::open_with_custom_alive_set(merged_index.schema(), segments, filter_doc_ids)?;
    println!("merge_filtered_segments: Initialized IndexMerger with custom alive sets.");

    let segment_serializer = SegmentSerializer::for_segment(merged_segment)?;
    println!(
        "merge_filtered_segments: Created SegmentSerializer for merged segment ID: {:?}",
        merged_segment_id
    );
    let num_docs = merger.write(segment_serializer)?;
    println!(
        "merge_filtered_segments: Written {} documents to merged segment ID: {:?}",
        num_docs, merged_segment_id
    );

    let segment_meta = merged_index.new_segment_meta(merged_segment_id, num_docs);
    println!(
        "merge_filtered_segments: Created SegmentMeta for merged segment ID: {:?}",
        merged_segment_id
    );

    let stats = format!(
        "Segments Merge: [{}]",
        segments
            .iter()
            .fold(String::new(), |sum, current| format!(
                "{sum}{} ",
                current.meta().id().uuid_string()
            ))
            .trim_end()
    );
    println!("merge_filtered_segments: Created stats string: {}", stats);

    let index_meta = IndexMeta {
        index_settings: target_settings.clone(), /* index_settings of all segments should be the
                                                  * same */
        segments: vec![segment_meta],
        schema: target_schema.clone(),
        opstamp: 0u64,
        payload: Some(stats),
    };
    println!("merge_filtered_segments: Created IndexMeta with updated segments and payload.");

    // save the meta.json
    let segment_metas = segments
        .iter()
        .map(|segment| segment.meta().clone())
        .collect();
    let previous_meta = IndexMeta {
        index_settings: target_settings,
        segments: segment_metas,
        schema: target_schema,
        opstamp: 0u64,
        payload: None,
    };
    println!("merge_filtered_segments: Saving metas using save_metas function.");
    save_metas(&index_meta, &previous_meta, merged_index.directory_mut())?;
    println!("merge_filtered_segments: Metas saved successfully.");

    Ok(merged_index)
}

pub(crate) struct InnerSegmentUpdater {
    // we keep a copy of the current active IndexMeta to
    // avoid loading the file every time we need it in the
    // `SegmentUpdater`.
    //
    // This should be up to date as all update happen through
    // the unique active `SegmentUpdater`.
    active_index_meta: RwLock<Arc<IndexMeta>>,
    pool: ThreadPool,
    merge_thread_pool: ThreadPool,

    index: Index,
    segment_manager: SegmentManager,
    merge_policy: RwLock<Arc<dyn MergePolicy>>,
    killed: AtomicBool,
    stamper: Stamper,
    merge_operations: MergeOperationInventory,
}

impl SegmentUpdater {
    pub fn create(
        index: Index,
        stamper: Stamper,
        delete_cursor: &DeleteCursor,
        num_merge_threads: usize,
        panic_handler: Option<DirectoryPanicHandler>,
    ) -> crate::Result<SegmentUpdater> {
        println!(
            "SegmentUpdater::create: Initializing SegmentUpdater with {} merge threads.",
            num_merge_threads
        );
        let segments = index.searchable_segment_metas()?;
        println!(
            "SegmentUpdater::create: Retrieved {} searchable segment metas.",
            segments.len()
        );
        let segment_manager = SegmentManager::from_segments(segments, delete_cursor);
        let mut builder = ThreadPoolBuilder::new()
            .thread_name(|_| "segment_updater".to_string())
            .num_threads(1);

        if let Some(panic_handler) = panic_handler.as_ref() {
            let panic_handler = panic_handler.clone();
            builder = builder.panic_handler(move |any| {
                panic_handler(any);
            });
        }

        let pool = builder.build().map_err(|_| {
            crate::TantivyError::SystemError("Failed to spawn segment updater thread".to_string())
        })?;
        let mut builder = ThreadPoolBuilder::new()
            .thread_name(|i| format!("merge_thread_{i}"))
            .num_threads(num_merge_threads);
        if let Some(panic_handler) = panic_handler {
            let panic_handler = panic_handler.clone();
            builder = builder.panic_handler(move |any| {
                panic_handler(any);
            });
        }

        let merge_thread_pool = builder.build().map_err(|_| {
            crate::TantivyError::SystemError("Failed to spawn segment merging thread".to_string())
        })?;
        let index_meta = index.load_metas()?;
        println!(
            "SegmentUpdater::create: Loaded initial IndexMeta with opstamp: {}.",
            index_meta.opstamp
        );
        Ok(SegmentUpdater(Arc::new(InnerSegmentUpdater {
            active_index_meta: RwLock::new(Arc::new(index_meta)),
            pool,
            merge_thread_pool,
            index,
            segment_manager,
            merge_policy: RwLock::new(Arc::new(DefaultMergePolicy::default())),
            killed: AtomicBool::new(false),
            stamper,
            merge_operations: Default::default(),
        })))
    }

    pub fn get_merge_policy(&self) -> Arc<dyn MergePolicy> {
        println!("SegmentUpdater::get_merge_policy: Retrieving current merge policy.");
        self.merge_policy.read().unwrap().clone()
    }

    pub fn set_merge_policy(&self, merge_policy: Box<dyn MergePolicy>) {
        println!("SegmentUpdater::set_merge_policy: Setting a new merge policy.");
        let arc_merge_policy = Arc::from(merge_policy);
        *self.merge_policy.write().unwrap() = arc_merge_policy;
        println!("SegmentUpdater::set_merge_policy: New merge policy set successfully.");
    }

    fn schedule_task<T: 'static + Send, F: FnOnce() -> crate::Result<T> + 'static + Send>(
        &self,
        task: F,
    ) -> FutureResult<T> {
        println!("SegmentUpdater::schedule_task: Scheduling a new task.");
        if !self.is_alive() {
            println!(
                "SegmentUpdater::schedule_task: SegmentUpdater is killed. Cannot schedule task."
            );
            return crate::TantivyError::SystemError("Segment updater killed".to_string()).into();
        }
        let (scheduled_result, sender) = FutureResult::create(
            "A segment_updater future did not succeed. This should never happen.",
        );
        self.pool.spawn(|| {
            println!("SegmentUpdater::schedule_task: Executing scheduled task.");
            let task_result = task();
            let _ = sender.send(task_result);
        });
        println!("SegmentUpdater::schedule_task: Task scheduled successfully.");
        scheduled_result
    }

    pub fn schedule_add_segment(&self, segment_entry: SegmentEntry) -> FutureResult<()> {
        println!(
            "SegmentUpdater::schedule_add_segment: Scheduling addition of segment ID: {:?}",
            segment_entry.meta().id()
        );
        let segment_updater = self.clone();
        self.schedule_task(move || {
            println!(
                "SegmentUpdater::schedule_add_segment: Adding segment ID: {:?}",
                segment_entry.meta().id()
            );
            segment_updater
                .segment_manager
                .add_segment(segment_entry.clone());
            println!(
                "SegmentUpdater::schedule_add_segment: Segment ID: {:?} added successfully.",
                segment_entry.meta().id()
            );
            // mingy98: We don't need to consider merge options for every segment, just at the very
            // end segment_updater.consider_merge_options();
            Ok(())
        })
    }

    /// Orders `SegmentManager` to remove all segments
    pub(crate) fn remove_all_segments(&self) {
        println!("SegmentUpdater::remove_all_segments: Removing all segments.");
        self.segment_manager.remove_all_segments();
        println!("SegmentUpdater::remove_all_segments: All segments removed.");
    }

    pub fn kill(&mut self) {
        println!("SegmentUpdater::kill: Killing the SegmentUpdater.");
        self.killed.store(true, Ordering::Release);
        println!("SegmentUpdater::kill: SegmentUpdater killed.");
    }

    pub fn is_alive(&self) -> bool {
        let alive = !self.killed.load(Ordering::Acquire);
        println!(
            "SegmentUpdater::is_alive: SegmentUpdater is currently {}.",
            if alive { "alive" } else { "killed" }
        );
        alive
    }

    /// Apply deletes up to the target opstamp to all segments.
    ///
    /// The method returns copies of the segment entries,
    /// updated with the delete information.
    fn purge_deletes(&self, target_opstamp: Opstamp) -> crate::Result<Vec<SegmentEntry>> {
        println!(
            "SegmentUpdater::purge_deletes: Applying deletes up to opstamp: {}.",
            target_opstamp
        );
        let mut segment_entries = self.segment_manager.segment_entries();
        for (i, segment_entry) in segment_entries.iter_mut().enumerate() {
            println!(
                "SegmentUpdater::purge_deletes: Advancing deletes for segment {} with ID: {:?}",
                i,
                segment_entry.meta().id()
            );
            let segment = self.index.segment(segment_entry.meta().clone());
            advance_deletes(segment, segment_entry, target_opstamp)?;
            println!(
                "SegmentUpdater::purge_deletes: Deletes advanced for segment {} with ID: {:?}",
                i,
                segment_entry.meta().id()
            );
        }
        println!(
            "SegmentUpdater::purge_deletes: All deletes up to opstamp {} applied successfully.",
            target_opstamp
        );
        Ok(segment_entries)
    }

    pub fn save_metas(
        &self,
        opstamp: Opstamp,
        commit_message: Option<String>,
        previous_metas: &IndexMeta,
    ) -> crate::Result<()> {
        println!(
            "SegmentUpdater::save_metas: Saving metas with opstamp: {} and commit_message: {:?}",
            opstamp, commit_message
        );
        if self.is_alive() {
            let index = &self.index;
            let directory = index.directory();
            let mut committed_segment_metas = self.segment_manager.committed_segment_metas();
            println!(
                "SegmentUpdater::save_metas: Retrieved {} committed segment metas.",
                committed_segment_metas.len()
            );

            // We sort segment_readers by number of documents.
            // This is a heuristic to make multithreading more efficient.
            //
            // This is not done at the searcher level because I had a strange
            // use case in which I was dealing with a large static index,
            // dispatched over 5 SSD drives.
            //
            // A `UnionDirectory` makes it possible to read from these
            // 5 different drives and creates a meta.json on the fly.
            // In order to optimize the throughput, it creates a lasagna of segments
            // from the different drives.
            //
            // Segment 1 from disk 1, Segment 1 from disk 2, etc.
            committed_segment_metas.sort_by_key(|segment_meta| -(segment_meta.max_doc() as i32));
            println!(
                "SegmentUpdater::save_metas: Sorted committed segment metas by max_doc descending."
            );
            let index_meta = IndexMeta {
                index_settings: index.settings().clone(),
                segments: committed_segment_metas,
                schema: index.schema(),
                opstamp,
                payload: commit_message,
            };
            println!(
                "SegmentUpdater::save_metas: Created new IndexMeta with opstamp: {}.",
                opstamp
            );
            // TODO add context to the error.
            save_metas(
                &index_meta,
                &previous_metas,
                directory.box_clone().borrow_mut(),
            )?;
            println!("SegmentUpdater::save_metas: Metas saved successfully.");
            self.store_meta(&index_meta);
            println!("SegmentUpdater::save_metas: Active IndexMeta updated.");
        } else {
            println!("SegmentUpdater::save_metas: SegmentUpdater is killed. Skipping save_metas.");
        }
        Ok(())
    }

    pub fn schedule_garbage_collect(&self) -> FutureResult<GarbageCollectionResult> {
        println!("SegmentUpdater::schedule_garbage_collect: Scheduling garbage collection task.");
        let self_clone = self.clone();
        self.schedule_task(move || garbage_collect_files(self_clone))
    }

    /// List the files that are useful to the index.
    ///
    /// This does not include lock files, or files that are obsolete
    /// but have not yet been deleted by the garbage collector.
    fn list_files(&self) -> HashSet<PathBuf> {
        println!("SegmentUpdater::list_files: Listing all relevant files for the index.");
        let mut files: HashSet<PathBuf> = self
            .index
            .list_all_segment_metas()
            .into_iter()
            .flat_map(|segment_meta| {
                println!(
                    "SegmentUpdater::list_files: Listing files for SegmentMeta ID: {:?}",
                    segment_meta.id()
                );
                segment_meta.list_files()
            })
            .collect();
        files.insert(META_FILEPATH.to_path_buf());
        println!(
            "SegmentUpdater::list_files: Total files collected for garbage collection: {}",
            files.len()
        );
        files
    }

    pub(crate) fn schedule_commit(
        &self,
        opstamp: Opstamp,
        payload: Option<String>,
    ) -> FutureResult<Opstamp> {
        println!(
            "SegmentUpdater::schedule_commit: Scheduling commit with opstamp: {} and payload: {:?}",
            opstamp, payload
        );
        let segment_updater: SegmentUpdater = self.clone();
        self.schedule_task(move || {
            println!(
                "SegmentUpdater::schedule_commit: Applying deletes up to opstamp: {}.",
                opstamp
            );
            let segment_entries = segment_updater.purge_deletes(opstamp)?;
            println!(
                "SegmentUpdater::schedule_commit: Purged deletes. Committing {} segments.",
                segment_entries.len()
            );
            let previous_metas = segment_updater.load_meta();
            segment_updater.segment_manager.commit(segment_entries);
            println!("SegmentUpdater::schedule_commit: SegmentManager commit completed.");
            segment_updater.save_metas(opstamp, payload, &previous_metas)?;
            println!("SegmentUpdater::schedule_commit: Metas saved successfully.");
            let _ = garbage_collect_files(segment_updater.clone());
            println!(
                "SegmentUpdater::schedule_commit: Garbage collection triggered after commit."
            );

            let index_meta = segment_updater.load_meta();
            if let Some(new_merge_policy) = segment_updater
                .index
                .directory()
                .reconsider_merge_policy(&index_meta, &previous_metas)
            {
                println!(
                    "SegmentUpdater::schedule_commit: New merge policy detected. Updating merge policy."
                );
                segment_updater.set_merge_policy(new_merge_policy);
            } else {
                println!(
                    "SegmentUpdater::schedule_commit: No new merge policy detected."
                );
            }
            segment_updater.consider_merge_options();
            println!(
                "SegmentUpdater::schedule_commit: Considering merge options after commit."
            );
            Ok(opstamp)
        })
    }

    fn store_meta(&self, index_meta: &IndexMeta) {
        println!(
            "InnerSegmentUpdater::store_meta: Storing new IndexMeta with opstamp: {}.",
            index_meta.opstamp
        );
        *self.active_index_meta.write().unwrap() = Arc::new(index_meta.clone());
        println!("InnerSegmentUpdater::store_meta: Active IndexMeta updated successfully.");
    }

    fn load_meta(&self) -> Arc<IndexMeta> {
        let meta = self.active_index_meta.read().unwrap().clone();
        println!(
            "InnerSegmentUpdater::load_meta: Loaded IndexMeta with opstamp: {}.",
            meta.opstamp
        );
        meta
    }

    pub(crate) fn make_merge_operation(&self, segment_ids: &[SegmentId]) -> MergeOperation {
        println!(
            "InnerSegmentUpdater::make_merge_operation: Creating MergeOperation for segments: {:?}",
            segment_ids
        );
        let commit_opstamp = self.load_meta().opstamp;
        let merge_op =
            MergeOperation::new(&self.merge_operations, commit_opstamp, segment_ids.to_vec());
        println!(
            "InnerSegmentUpdater::make_merge_operation: MergeOperation created with target_opstamp: {}.",
            commit_opstamp
        );
        merge_op
    }

    // Starts a merge operation. This function will block until the merge operation is effectively
    // started. Note that it does not wait for the merge to terminate.
    // The calling thread should not be blocked for a long time, as this only involves waiting for the
    // `SegmentUpdater` queue which in turn only contains lightweight operations.
    //
    // The merge itself happens on a different thread.
    //
    // When successful, this function returns a `Future` for a `Result<SegmentMeta>` that represents
    // the actual outcome of the merge operation.
    //
    // It returns an error if for some reason the merge operation could not be started.
    //
    // At this point an error is not necessarily the sign of a malfunction.
    // (e.g. A rollback could have happened, between the instant when the merge operation was
    // suggested and the moment when it ended up being executed.)
    //
    // `segment_ids` is required to be non-empty.
    pub fn start_merge(
        &self,
        merge_operation: MergeOperation,
    ) -> FutureResult<Option<SegmentMeta>> {
        println!(
            "InnerSegmentUpdater::start_merge: Initiating merge operation for segments: {:?}",
            merge_operation.segment_ids()
        );
        assert!(
            !merge_operation.segment_ids().is_empty(),
            "Segment_ids cannot be empty."
        );

        let segment_updater = self.clone();
        let segment_entries: Vec<SegmentEntry> = match self
            .segment_manager
            .start_merge(merge_operation.segment_ids())
        {
            Ok(segment_entries) => {
                println!(
                    "InnerSegmentUpdater::start_merge: Started merge with {} segment entries.",
                    segment_entries.len()
                );
                segment_entries
            }
            Err(err) => {
                println!(
                    "InnerSegmentUpdater::start_merge: Failed to start merge operation: {:?}",
                    err
                );
                // If starting the merge fails, log a warning and return the error as a future.
                warn!(
                    "Starting the merge failed for the following reason. This is not fatal. {}",
                    err
                );
                return err.into();
            }
        };

        info!(
            "InnerSegmentUpdater::start_merge: Starting merge - {:?}",
            merge_operation.segment_ids()
        );

        let (scheduled_result, merging_future_send) =
            FutureResult::create("Merge operation failed.");

        self.merge_thread_pool.spawn(move || {
            println!(
                "InnerSegmentUpdater::start_merge: Merge thread spawned for segments: {:?}",
                merge_operation.segment_ids()
            );
            // The fact that `merge_operation` is moved here is important.
            // Its lifetime is used to track how many merging threads are currently running,
            // as well as which segment is currently in merge and therefore should not be
            // candidate for another merge.
            match merge(
                &segment_updater.index,
                segment_entries,
                merge_operation.target_opstamp(),
            ) {
                Ok(after_merge_segment_entry) => {
                    println!(
                        "InnerSegmentUpdater::start_merge: Merge completed successfully."
                    );
                    let res = segment_updater.end_merge(merge_operation, after_merge_segment_entry);
                    let _send_result = merging_future_send.send(res);
                }
                Err(merge_error) => {
                    println!(
                        "InnerSegmentUpdater::start_merge: Merge failed for segments: {:?} with error: {:?}",
                        merge_operation.segment_ids(),
                        merge_error
                    );
                    warn!(
                        "Merge of {:?} was cancelled: {:?}",
                        merge_operation.segment_ids(),
                        merge_error
                    );
                    if cfg!(test) {
                        panic!("{merge_error:?}");
                    }
                    let _send_result = merging_future_send.send(Err(merge_error));
                }
            }
        });

        println!("InnerSegmentUpdater::start_merge: Merge operation scheduled successfully.");
        scheduled_result
    }

    pub(crate) fn get_mergeable_segments(&self) -> (Vec<SegmentMeta>, Vec<SegmentMeta>) {
        println!("InnerSegmentUpdater::get_mergeable_segments: Retrieving mergeable segments.");
        let merge_segment_ids: HashSet<SegmentId> = self.merge_operations.segment_in_merge();
        println!(
            "InnerSegmentUpdater::get_mergeable_segments: Found {} segments currently in merge.",
            merge_segment_ids.len()
        );
        let committed_segments = self
            .segment_manager
            .get_mergeable_segments(&merge_segment_ids)
            .0;
        let uncommitted_segments = self
            .segment_manager
            .get_mergeable_segments(&merge_segment_ids)
            .1;
        println!(
            "InnerSegmentUpdater::get_mergeable_segments: Found {} committed and {} uncommitted mergeable segments.",
            committed_segments.len(),
            uncommitted_segments.len()
        );
        (committed_segments, uncommitted_segments)
    }

    fn consider_merge_options(&self) {
        println!("InnerSegmentUpdater::consider_merge_options: Evaluating merge options.");
        let (mut committed_segments, mut uncommitted_segments) = self.get_mergeable_segments();
        println!(
            "InnerSegmentUpdater::consider_merge_options: Retrieved {} committed and {} uncommitted segments.",
            committed_segments.len(),
            uncommitted_segments.len()
        );
        if committed_segments.len() == 1 && committed_segments[0].num_deleted_docs() == 0 {
            println!(
                "InnerSegmentUpdater::consider_merge_options: Only one committed segment with no deletes. Clearing committed_segments."
            );
            committed_segments.clear();
        }
        if uncommitted_segments.len() == 1 && uncommitted_segments[0].num_deleted_docs() == 0 {
            println!(
                "InnerSegmentUpdater::consider_merge_options: Only one uncommitted segment with no deletes. Clearing uncommitted_segments."
            );
            uncommitted_segments.clear();
        }

        // Committed segments cannot be merged with uncommitted_segments.
        // We therefore consider merges using these two sets of segments independently.
        let merge_policy = self.get_merge_policy();
        println!("InnerSegmentUpdater::consider_merge_options: Retrieved current merge policy.");

        let current_opstamp = self.stamper.stamp();
        println!(
            "InnerSegmentUpdater::consider_merge_options: Current opstamp is {}.",
            current_opstamp
        );

        let mut merge_candidates: Vec<MergeOperation> = merge_policy
            .compute_merge_candidates(&uncommitted_segments)
            .into_iter()
            .map(|merge_candidate| {
                println!(
                    "InnerSegmentUpdater::consider_merge_options: Creating MergeOperation for uncommitted segments: {:?}",
                    merge_candidate.0
                );
                MergeOperation::new(&self.merge_operations, current_opstamp, merge_candidate.0)
            })
            .collect();
        println!(
            "InnerSegmentUpdater::consider_merge_options: Found {} merge candidates from uncommitted segments.",
            merge_candidates.len()
        );

        let commit_opstamp = self.load_meta().opstamp;
        println!(
            "InnerSegmentUpdater::consider_merge_options: Commit opstamp loaded: {}.",
            commit_opstamp
        );
        let committed_merge_candidates = merge_policy
            .compute_merge_candidates(&committed_segments)
            .into_iter()
            .map(|merge_candidate: MergeCandidate| {
                println!(
                    "InnerSegmentUpdater::consider_merge_options: Creating MergeOperation for committed segments: {:?}",
                    merge_candidate.0
                );
                MergeOperation::new(&self.merge_operations, commit_opstamp, merge_candidate.0)
            });
        merge_candidates.extend(committed_merge_candidates);
        println!(
            "InnerSegmentUpdater::consider_merge_options: Total merge candidates after including committed segments: {}.",
            merge_candidates.len()
        );

        for merge_operation in merge_candidates {
            println!(
                "InnerSegmentUpdater::consider_merge_options: Initiating merge for segments: {:?}",
                merge_operation.segment_ids()
            );
            // If a merge cannot be started this is not a fatal error.
            // We do log a warning in `start_merge`.
            drop(self.start_merge(merge_operation));
            println!("InnerSegmentUpdater::consider_merge_options: Merge operation scheduled.");
        }
        println!(
            "InnerSegmentUpdater::consider_merge_options: Completed evaluating and scheduling merge options."
        );
    }

    /// Queues a `end_merge` in the segment updater and blocks until it is successfully processed.
    fn end_merge(
        &self,
        merge_operation: MergeOperation,
        mut after_merge_segment_entry: Option<SegmentEntry>,
    ) -> crate::Result<Option<SegmentMeta>> {
        println!(
            "InnerSegmentUpdater::end_merge: Finalizing merge operation for segments: {:?}",
            merge_operation.segment_ids()
        );
        let segment_updater = self.clone();
        let after_merge_segment_meta = after_merge_segment_entry
            .as_ref()
            .map(|after_merge_segment_entry| after_merge_segment_entry.meta().clone());
        self.schedule_task(move || {
            info!(
                "InnerSegmentUpdater::end_merge: Ending merge for segments: {:?}",
                after_merge_segment_entry.as_ref().map(|entry| entry.meta())
            );
            {
                if let Some(after_merge_segment_entry) = after_merge_segment_entry.as_mut() {
                    // Deletes and commits could have happened as we were merging.
                    // We need to make sure we are up to date with deletes before accepting the
                    // segment.
                    let mut delete_cursor = after_merge_segment_entry.delete_cursor().clone();
                    if let Some(delete_operation) = delete_cursor.get() {
                        let committed_opstamp = segment_updater.load_meta().opstamp;
                        if delete_operation.opstamp() < committed_opstamp {
                            // We are not up to date! Let's create a new tombstone file for our
                            // freshly create split.
                            let index = &segment_updater.index;
                            let segment = index.segment(after_merge_segment_entry.meta().clone());
                            if let Err(advance_deletes_err) = advance_deletes(
                                segment,
                                after_merge_segment_entry,
                                committed_opstamp,
                            ) {
                                println!(
                                    "InnerSegmentUpdater::end_merge: Failed to advance deletes for segment: {:?}. Error: {:?}",
                                    after_merge_segment_entry.meta().id(),
                                    advance_deletes_err
                                );
                                error!(
                                    "Merge of {:?} was cancelled (advancing deletes failed): {:?}",
                                    merge_operation.segment_ids(),
                                    advance_deletes_err
                                );
                                assert!(!cfg!(test), "Merge failed.");

                                // ... cancel merge
                                // `merge_operations` are tracked. As it is dropped, the
                                // the segment_ids will be available again for merge.
                                return Err(advance_deletes_err);
                            }
                            println!(
                                "InnerSegmentUpdater::end_merge: Deletes advanced successfully for segment: {:?}.",
                                after_merge_segment_entry.meta().id()
                            );
                        }
                    }
                }
                let previous_metas = segment_updater.load_meta();
                println!(
                    "InnerSegmentUpdater::end_merge: Loaded previous IndexMeta with opstamp: {}.",
                    previous_metas.opstamp
                );
                let segments_status = segment_updater
                    .segment_manager
                    .end_merge(merge_operation.segment_ids(), after_merge_segment_entry)?;

                if segments_status == SegmentsStatus::Committed {
                    println!(
                        "InnerSegmentUpdater::end_merge: SegmentsStatus::Committed. Saving metas."
                    );
                    segment_updater.save_metas(
                        previous_metas.opstamp,
                        previous_metas.payload.clone(),
                        &previous_metas,
                    )?;
                    println!("InnerSegmentUpdater::end_merge: Metas saved successfully.");
                }

                segment_updater.consider_merge_options();
                println!(
                    "InnerSegmentUpdater::end_merge: Considering further merge options post-merge."
                );
            } // we drop all possible handle to a now useless `SegmentMeta`.

            println!(
                "InnerSegmentUpdater::end_merge: Triggering garbage collection after merge."
            );
            let _ = garbage_collect_files(segment_updater);
            println!(
                "InnerSegmentUpdater::end_merge: Garbage collection triggered successfully."
            );
            Ok(())
        })
        .wait()?;
        println!("InnerSegmentUpdater::end_merge: Merge operation finalized successfully.");
        Ok(after_merge_segment_meta)
    }

    /// Wait for current merging threads.
    ///
    /// Upon termination of the current merging threads,
    /// merge opportunity may appear.
    ///
    /// We keep waiting until the merge policy judges that
    /// no opportunity is available.
    ///
    /// Note that it is not required to call this
    /// method in your application.
    /// Terminating your application without letting
    /// merge terminate is perfectly safe.
    ///
    /// Obsolete files will eventually be cleaned up
    /// by the directory garbage collector.
    pub fn wait_merging_thread(&self) -> crate::Result<()> {
        println!(
            "InnerSegmentUpdater::wait_merging_thread: Waiting for all merging threads to complete."
        );
        self.merge_operations.wait_until_empty();
        println!("InnerSegmentUpdater::wait_merging_thread: All merging threads have completed.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::merge_indices;
    use crate::collector::TopDocs;
    use crate::directory::RamDirectory;
    use crate::fastfield::AliveBitSet;
    use crate::indexer::merge_policy::tests::MergeWheneverPossible;
    use crate::indexer::merger::IndexMerger;
    use crate::indexer::segment_updater::merge_filtered_segments;
    use crate::query::QueryParser;
    use crate::schema::*;
    use crate::{Directory, DocAddress, Index, Segment};

    #[test]
    fn test_delete_during_merge() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema_builder.build());

        // writing the segment
        let mut index_writer = index.writer_for_tests()?;
        index_writer.set_merge_policy(Box::new(MergeWheneverPossible));

        for _ in 0..100 {
            index_writer.add_document(doc!(text_field=>"a"))?;
            index_writer.add_document(doc!(text_field=>"b"))?;
        }
        index_writer.commit()?;

        for _ in 0..100 {
            index_writer.add_document(doc!(text_field=>"c"))?;
            index_writer.add_document(doc!(text_field=>"d"))?;
        }
        index_writer.commit()?;

        index_writer.add_document(doc!(text_field=>"e"))?;
        index_writer.add_document(doc!(text_field=>"f"))?;
        index_writer.commit()?;

        let term = Term::from_field_text(text_field, "a");
        index_writer.delete_term(term);
        index_writer.commit()?;

        let reader = index.reader()?;
        assert_eq!(reader.searcher().num_docs(), 302);

        index_writer.wait_merging_threads()?;

        reader.reload()?;
        assert_eq!(reader.searcher().segment_readers().len(), 1);
        assert_eq!(reader.searcher().num_docs(), 302);
        Ok(())
    }

    #[test]
    fn delete_all_docs_min() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema_builder.build());

        // writing the segment
        let mut index_writer = index.writer_for_tests()?;

        for _ in 0..10 {
            index_writer.add_document(doc!(text_field=>"a"))?;
            index_writer.add_document(doc!(text_field=>"b"))?;
        }
        index_writer.commit()?;

        let _seg_ids = index.searchable_segment_ids()?;
        // In Tantivy upstream, this test results in 0 segments after delete.
        // However, due to our custom, visibility rules, we leave the segment.
        // See committed_segment_metas in segment_manager.rs.
        // assert!(!seg_ids.is_empty());

        let term = Term::from_field_text(text_field, "a");
        index_writer.delete_term(term);
        index_writer.commit()?;

        let term = Term::from_field_text(text_field, "b");
        index_writer.delete_term(term);
        index_writer.commit()?;

        index_writer.wait_merging_threads()?;

        let reader = index.reader()?;
        assert_eq!(reader.searcher().num_docs(), 0);

        let _seg_ids = index.searchable_segment_ids()?;
        // Skipped due to custom ParadeDB visibility rules.
        // assert!(seg_ids.is_empty());

        reader.reload()?;
        assert_eq!(reader.searcher().num_docs(), 0);
        // Skipped due to custom ParadeDB visibility rules.
        // assert!(index.searchable_segment_metas()?.is_empty());
        // assert!(reader.searcher().segment_readers().is_empty());

        Ok(())
    }

    #[test]
    fn delete_all_docs() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema_builder.build());

        // writing the segment
        let mut index_writer = index.writer_for_tests()?;

        for _ in 0..100 {
            index_writer.add_document(doc!(text_field=>"a"))?;
            index_writer.add_document(doc!(text_field=>"b"))?;
        }
        index_writer.commit()?;

        for _ in 0..100 {
            index_writer.add_document(doc!(text_field=>"c"))?;
            index_writer.add_document(doc!(text_field=>"d"))?;
        }
        index_writer.commit()?;

        index_writer.add_document(doc!(text_field=>"e"))?;
        index_writer.add_document(doc!(text_field=>"f"))?;
        index_writer.commit()?;

        let _seg_ids = index.searchable_segment_ids()?;
        // In Tantivy upstream, this test results in 0 segments after delete.
        // However, due to our custom, visibility rules, we leave the segment.
        // See committed_segment_metas in segment_manager.rs.
        // assert!(!seg_ids.is_empty());

        let term_vals = vec!["a", "b", "c", "d", "e", "f"];
        for term_val in term_vals {
            let term = Term::from_field_text(text_field, term_val);
            index_writer.delete_term(term);
            index_writer.commit()?;
        }

        index_writer.wait_merging_threads()?;

        let reader = index.reader()?;
        assert_eq!(reader.searcher().num_docs(), 0);

        let _seg_ids = index.searchable_segment_ids()?;
        // Skipped due to custom ParadeDB visibility rules.
        // assert!(seg_ids.is_empty());

        reader.reload()?;
        assert_eq!(reader.searcher().num_docs(), 0);
        // Skipped due to custom ParadeDB visibility rules.
        // assert!(index.searchable_segment_metas()?.is_empty());
        // assert!(reader.searcher().segment_readers().is_empty());

        Ok(())
    }

    #[test]
    fn test_remove_all_segments() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema_builder.build());

        // writing the segment
        let mut index_writer = index.writer_for_tests()?;
        for _ in 0..100 {
            index_writer.add_document(doc!(text_field=>"a"))?;
            index_writer.add_document(doc!(text_field=>"b"))?;
        }
        index_writer.commit()?;

        index_writer.segment_updater().remove_all_segments();
        let seg_vec = index_writer
            .segment_updater()
            .segment_manager
            .segment_entries();
        assert!(seg_vec.is_empty());
        Ok(())
    }

    #[test]
    fn test_merge_segments() -> crate::Result<()> {
        let mut indices = vec![];
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();

        for _ in 0..3 {
            let index = Index::create_in_ram(schema.clone());

            // writing two segments
            let mut index_writer = index.writer_for_tests()?;
            for _ in 0..100 {
                index_writer.add_document(doc!(text_field=>"fizz"))?;
                index_writer.add_document(doc!(text_field=>"buzz"))?;
            }
            index_writer.commit()?;

            for _ in 0..1000 {
                index_writer.add_document(doc!(text_field=>"foo"))?;
                index_writer.add_document(doc!(text_field=>"bar"))?;
            }
            index_writer.commit()?;
            indices.push(index);
        }

        assert_eq!(indices.len(), 3);
        let output_directory: Box<dyn Directory> = Box::<RamDirectory>::default();
        let index = merge_indices(&indices, output_directory)?;
        assert_eq!(index.schema(), schema);

        let segments = index.searchable_segments()?;
        assert_eq!(segments.len(), 1);

        let segment_metas = segments[0].meta();
        assert_eq!(segment_metas.num_deleted_docs(), 0);
        assert_eq!(segment_metas.num_docs(), 6600);
        Ok(())
    }

    #[test]
    fn test_merge_empty_indices_array() {
        let merge_result = merge_indices(&[], RamDirectory::default());
        assert!(merge_result.is_err());
    }

    #[test]
    fn test_merge_mismatched_schema() -> crate::Result<()> {
        let first_index = {
            let mut schema_builder = Schema::builder();
            let text_field = schema_builder.add_text_field("text", TEXT);
            let index = Index::create_in_ram(schema_builder.build());
            let mut index_writer = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field=>"some text"))?;
            index_writer.commit()?;
            index
        };

        let second_index = {
            let mut schema_builder = Schema::builder();
            let body_field = schema_builder.add_text_field("body", TEXT);
            let index = Index::create_in_ram(schema_builder.build());
            let mut index_writer = index.writer_for_tests()?;
            index_writer.add_document(doc!(body_field=>"some body"))?;
            index_writer.commit()?;
            index
        };

        // mismatched schema index list
        let result = merge_indices(&[first_index, second_index], RamDirectory::default());
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_merge_filtered_segments() -> crate::Result<()> {
        let first_index = {
            let mut schema_builder = Schema::builder();
            let text_field = schema_builder.add_text_field("text", TEXT);
            let index = Index::create_in_ram(schema_builder.build());
            let mut index_writer = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field=>"some text 1"))?;
            index_writer.add_document(doc!(text_field=>"some text 2"))?;
            index_writer.commit()?;
            index
        };

        let second_index = {
            let mut schema_builder = Schema::builder();
            let text_field = schema_builder.add_text_field("text", TEXT);
            let index = Index::create_in_ram(schema_builder.build());
            let mut index_writer = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field=>"some text 3"))?;
            index_writer.add_document(doc!(text_field=>"some text 4"))?;
            index_writer.delete_term(Term::from_field_text(text_field, "4"));

            index_writer.commit()?;
            index
        };

        let mut segments: Vec<Segment> = Vec::new();
        segments.extend(first_index.searchable_segments()?);
        segments.extend(second_index.searchable_segments()?);

        let target_settings = first_index.settings().clone();

        let filter_segment_1 = AliveBitSet::for_test_from_deleted_docs(&[1], 2);
        let filter_segment_2 = AliveBitSet::for_test_from_deleted_docs(&[0], 2);

        let filter_segments = vec![Some(filter_segment_1), Some(filter_segment_2)];

        let merged_index = merge_filtered_segments(
            &segments,
            target_settings,
            filter_segments,
            RamDirectory::default(),
        )?;

        let segments = merged_index.searchable_segments()?;
        assert_eq!(segments.len(), 1);

        let segment_metas = segments[0].meta();
        assert_eq!(segment_metas.num_deleted_docs(), 0);
        assert_eq!(segment_metas.num_docs(), 1);

        Ok(())
    }

    #[test]
    fn test_merge_single_filtered_segments() -> crate::Result<()> {
        let first_index = {
            let mut schema_builder = Schema::builder();
            let text_field = schema_builder.add_text_field("text", TEXT);
            let index = Index::create_in_ram(schema_builder.build());
            let mut index_writer = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field=>"test text"))?;
            index_writer.add_document(doc!(text_field=>"some text 2"))?;

            index_writer.add_document(doc!(text_field=>"some text 3"))?;
            index_writer.add_document(doc!(text_field=>"some text 4"))?;

            index_writer.delete_term(Term::from_field_text(text_field, "4"));

            index_writer.commit()?;
            index
        };

        let mut segments: Vec<Segment> = Vec::new();
        segments.extend(first_index.searchable_segments()?);

        let target_settings = first_index.settings().clone();

        let filter_segment = AliveBitSet::for_test_from_deleted_docs(&[0], 4);

        let filter_segments = vec![Some(filter_segment)];

        let index = merge_filtered_segments(
            &segments,
            target_settings,
            filter_segments,
            RamDirectory::default(),
        )?;

        let segments = index.searchable_segments()?;
        assert_eq!(segments.len(), 1);

        let segment_metas = segments[0].meta();
        assert_eq!(segment_metas.num_deleted_docs(), 0);
        assert_eq!(segment_metas.num_docs(), 2);

        let searcher = index.reader()?.searcher();
        {
            let text_field = index.schema().get_field("text").unwrap();

            let do_search = |term: &str| {
                let query = QueryParser::for_index(&index, vec![text_field])
                    .parse_query(term)
                    .unwrap();
                let top_docs: Vec<(f32, DocAddress)> =
                    searcher.search(&query, &TopDocs::with_limit(3)).unwrap();

                top_docs.iter().map(|el| el.1.doc_id).collect::<Vec<_>>()
            };

            assert_eq!(do_search("test"), vec![] as Vec<u32>);
            assert_eq!(do_search("text"), vec![0, 1]);
        }

        Ok(())
    }

    #[test]
    fn test_apply_doc_id_filter_in_merger() -> crate::Result<()> {
        let first_index = {
            let mut schema_builder = Schema::builder();
            let text_field = schema_builder.add_text_field("text", TEXT);
            let index = Index::create_in_ram(schema_builder.build());
            let mut index_writer = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field=>"some text 1"))?;
            index_writer.add_document(doc!(text_field=>"some text 2"))?;

            index_writer.add_document(doc!(text_field=>"some text 3"))?;
            index_writer.add_document(doc!(text_field=>"some text 4"))?;

            index_writer.delete_term(Term::from_field_text(text_field, "4"));

            index_writer.commit()?;
            index
        };

        let mut segments: Vec<Segment> = Vec::new();
        segments.extend(first_index.searchable_segments()?);

        let target_settings = first_index.settings().clone();
        {
            let filter_segment = AliveBitSet::for_test_from_deleted_docs(&[1], 4);
            let filter_segments = vec![Some(filter_segment)];
            let target_schema = segments[0].schema();
            let merged_index = Index::create(
                RamDirectory::default(),
                target_schema,
                target_settings.clone(),
            )?;
            let merger: IndexMerger = IndexMerger::open_with_custom_alive_set(
                merged_index.schema(),
                &segments[..],
                filter_segments,
            )?;

            let doc_ids_alive: Vec<_> = merger.readers[0].doc_ids_alive().collect();
            assert_eq!(doc_ids_alive, vec![0, 2]);
        }

        {
            let filter_segments = vec![None];
            let target_schema = segments[0].schema();
            let merged_index =
                Index::create(RamDirectory::default(), target_schema, target_settings)?;
            let merger: IndexMerger = IndexMerger::open_with_custom_alive_set(
                merged_index.schema(),
                &segments[..],
                filter_segments,
            )?;

            let doc_ids_alive: Vec<_> = merger.readers[0].doc_ids_alive().collect();
            assert_eq!(doc_ids_alive, vec![0, 1, 2]);
        }

        Ok(())
    }
}
