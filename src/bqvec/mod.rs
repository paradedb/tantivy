//! Fixed-size record storage as a [`SegmentPlugin`](crate::plugin::SegmentPlugin).
//!
//! Stores fixed-size byte records in a flat contiguous file with O(1) random
//! access by `DocId`. The on-disk layout is:
//!
//! ```text
//! [header: 8 bytes]
//!   bytes_per_record: u32 LE
//!   num_records:      u32 LE
//! [records: num_records × bytes_per_record]
//! ```
//!
//! Access: `record(doc_id) = data[8 + doc_id * bytes_per_record .. +bytes_per_record]`
//!
//! The record content is opaque — the caller decides what goes in each record.
//! For example, a RaBitQ record might pack:
//!
//! ```text
//! [binary_code: dims/8 bytes] [norm: f32] [x_bar: f32]
//! ```
//!
//! # Data ingestion
//!
//! Because [`PluginWriter::add_document`](crate::plugin::PluginWriter::add_document)
//! only receives a `DocId` (the `Document` trait is not dyn-compatible), records
//! are fed through a **staging queue** on the plugin itself:
//!
//! ```rust,ignore
//! // 768-dim binary code (96 bytes) + 2 f32 correction terms (8 bytes) = 104 bytes
//! let bqvec = Arc::new(BqVecPlugin::new(104));
//! let index = Index::builder().schema(schema).plugin(bqvec.clone()).create_in_ram()?;
//! let mut writer = index.writer_with_num_threads(1, 50_000_000)?;
//!
//! bqvec.stage_record(my_record_bytes);
//! writer.add_document(doc!(field => "hello"))?;
//! ```
//!
//! Each `stage_record` call enqueues one record; the next
//! `PluginWriter::add_document` call dequeues it. If no record is staged, a
//! zero-filled record is inserted.

mod plugin;

pub use self::plugin::{BqVecPlugin, BqVecPluginReader, BqVecPluginWriter};
