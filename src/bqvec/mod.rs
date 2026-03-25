//! Binary quantized vector storage as a [`SegmentPlugin`](crate::plugin::SegmentPlugin).
//!
//! Stores binary quantized vectors in a flat contiguous file with O(1) random
//! access by `DocId`. Each vector occupies exactly `dims / 8` bytes (one bit per
//! original dimension). The on-disk layout is:
//!
//! ```text
//! [header: 8 bytes]
//!   dimensions:  u32 LE
//!   num_vectors: u32 LE
//! [vectors: num_vectors × bytes_per_vector]
//! ```
//!
//! Access: `vector(doc_id) = data[8 + doc_id * bytes_per_vector .. +bytes_per_vector]`
//!
//! # Data ingestion
//!
//! Because [`PluginWriter::add_document`](crate::plugin::PluginWriter::add_document)
//! only receives a `DocId` (the `Document` trait is not dyn-compatible), vectors
//! are fed through a **staging queue** on the plugin itself:
//!
//! ```rust,ignore
//! let bqvec = Arc::new(BqVecPlugin::new(768));
//! let index = Index::builder().schema(schema).plugin(bqvec.clone()).create_in_ram()?;
//! let mut writer = index.writer_with_num_threads(1, 50_000_000)?;
//!
//! bqvec.stage_vector(my_bq_bytes);
//! writer.add_document(doc!(field => "hello"))?;
//! ```
//!
//! Each `stage_vector` call enqueues one vector; the next
//! `PluginWriter::add_document` call dequeues it. If no vector is staged, a
//! zero-vector is inserted.

mod plugin;

pub use self::plugin::{BqVecPlugin, BqVecPluginReader, BqVecPluginWriter};
