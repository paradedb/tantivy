//! On-disk vector storage: file headers, block columns, metadata, cluster bounds, row maps, and
//! the readers over them. See `FORMAT.md`.

pub(crate) mod blocks;
pub(crate) mod bounds;
pub(crate) mod centroids;
pub(crate) mod header;
pub(crate) mod id_map;
pub(crate) mod io;
pub(crate) mod metadata;
pub(crate) mod quantized;
pub(crate) mod reader;

pub use io::VectorIoStats;
pub use reader::VectorIndexReader;
