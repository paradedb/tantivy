//! Multi-field fixed-size record storage as a [`SegmentPlugin`](crate::plugin::SegmentPlugin).
//!
//! Stores fixed-size byte records per vector field in a single `.bqvec`
//! composite file. Each field gets its own section with O(1) random access
//! by `DocId`. The on-disk layout per field section is:
//!
//! ```text
//! [header: 8 bytes]
//!   bytes_per_record: u32 LE
//!   num_records:      u32 LE
//! [records: num_records x bytes_per_record]
//! ```
//!
//! Access: `record(doc_id) = section_data[8 + doc_id * bytes_per_record .. +bytes_per_record]`
//!
//! The record content is opaque — the caller decides what goes in each record
//! via the `encode_fn` provided at build time. For example, a RaBitQ record
//! might pack:
//!
//! ```text
//! [binary_code: dims/8 bytes] [norm: f32] [x_bar: f32]
//! ```
//!
//! # Data ingestion
//!
//! Vector data flows through documents. The `BqVecPluginWriter` extracts
//! vector field values from each document and encodes them using the
//! per-field `encode_fn`:
//!
//! ```rust,ignore
//! use std::sync::Arc;
//!
//! let bqvec = Arc::new(
//!     BqVecPlugin::builder()
//!         .vector_field(vec_field, 104, Arc::new(|v: &[f32]| rabitq::encode(v)))
//!         .build()
//! );
//! let index = Index::builder().schema(schema).plugin(bqvec.clone()).create_in_ram()?;
//! let mut writer = index.writer_with_num_threads(1, 50_000_000)?;
//!
//! writer.add_document(doc!(text_field => "hello", vec_field => vec![1.0f32; 768]))?;
//! ```

mod plugin;
#[cfg(test)]
mod tests;

pub use self::plugin::{
    BqVecFieldReader, BqVecPlugin, BqVecPluginBuilder, BqVecPluginReader, BqVecPluginWriter,
    EncodeFn,
};
pub(crate) use self::plugin::component as bqvec_component;
