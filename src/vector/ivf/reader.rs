use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::{io, mem};

use common::file_slice::DeferredFileSlice;
use common::{BinarySerializable, HasLen, OwnedBytes};

use super::{IvfFieldMeta, ASSIGNMENTS_EXT, IVFVEC_EXT};
use crate::directory::{CompositeFile, FileSlice};
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType};
use crate::vector::reader::VectorColumnReader;
use crate::vector::VectorOptions;
use crate::{DocId, SegmentReader, TantivyError};

pub struct IvfVecReader {
    meta: CompositeFile,
    assignments_file: FileSlice,
    vec_file: FileSlice,
    field_options: BTreeMap<Field, VectorOptions>,
}

impl IvfVecReader {
    pub(crate) fn open(
        segment_reader: &SegmentReader,
        meta_slice: FileSlice,
    ) -> crate::Result<Self> {
        let schema = segment_reader.schema();
        let mut field_options = BTreeMap::new();
        for (field, entry) in schema.fields() {
            if let FieldType::Vector(opts) = entry.field_type() {
                field_options.insert(field, opts.clone());
            }
        }
        let assignments_file =
            segment_reader.open_read(SegmentComponent::Custom(ASSIGNMENTS_EXT.to_string()))?;
        let vec_file =
            segment_reader.open_read(SegmentComponent::Custom(IVFVEC_EXT.to_string()))?;
        Ok(Self {
            meta: CompositeFile::open(&meta_slice)?,
            assignments_file,
            vec_file,
            field_options,
        })
    }
}

impl VectorColumnReader for IvfVecReader {
    type Column = IvfVectorColumn;

    fn open_column(&self, field: Field) -> crate::Result<IvfVectorColumn> {
        let options = self.field_options.get(&field).cloned().ok_or_else(|| {
            TantivyError::InvalidArgument(format!("field {field:?} is not a vector field"))
        })?;
        let meta_slice = self.meta.open_read(field).ok_or_else(|| {
            TantivyError::InternalError(format!(
                "no IVF vector metadata for vector field {field:?} in segment"
            ))
        })?;
        let assignments_file = self.assignments_file.clone();
        let assignments_slice = DeferredFileSlice::new(move || {
            let assignments = CompositeFile::open(&assignments_file)?;
            assignments.open_read(field).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("no IVF vector assignments for vector field {field:?} in segment"),
                )
            })
        });
        let vec_file = self.vec_file.clone();
        let vec_slice = DeferredFileSlice::new(move || {
            let vec = CompositeFile::open(&vec_file)?;
            vec.open_read(field).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("no IVF vector data for vector field {field:?} in segment"),
                )
            })
        });
        let meta = IvfFieldMeta::open(meta_slice, &options)?;
        let assignments = (0..meta.num_centroids).map(|_| OnceLock::new()).collect();
        Ok(IvfVectorColumn {
            meta,
            options,
            assignments_slice,
            assignments,
            vec_slice,
            vec_bytes: OnceLock::new(),
        })
    }

    fn count(&self, field: Field) -> crate::Result<usize> {
        let options = self.field_options.get(&field).ok_or_else(|| {
            TantivyError::InvalidArgument(format!("field {field:?} is not a vector field"))
        })?;
        let meta_slice = self.meta.open_read(field).ok_or_else(|| {
            TantivyError::InternalError(format!(
                "no IVF vector metadata for vector field {field:?} in segment"
            ))
        })?;
        Ok(IvfFieldMeta::open(meta_slice, options)?.num_vectors())
    }

    fn dim(&self, field: Field) -> crate::Result<usize> {
        self.field_options
            .get(&field)
            .map(VectorOptions::dim)
            .ok_or_else(|| {
                TantivyError::InvalidArgument(format!("field {field:?} is not a vector field"))
            })
    }
}

/// Per-segment, per-field IVF column view.
pub struct IvfVectorColumn {
    meta: IvfFieldMeta,
    options: VectorOptions,
    assignments_slice: DeferredFileSlice,
    assignments: Vec<OnceLock<Vec<DocId>>>,
    vec_slice: DeferredFileSlice,
    vec_bytes: OnceLock<OwnedBytes>,
}

impl IvfVectorColumn {
    pub fn dim(&self) -> usize {
        self.options.dim()
    }

    pub fn len(&self) -> usize {
        self.meta.num_vectors()
    }

    pub fn is_empty(&self) -> bool {
        self.meta.num_vectors() == 0
    }

    pub fn contains(&self, doc_id: DocId) -> bool {
        self.row_id(doc_id).ok().flatten().is_some()
    }

    pub fn vector_bytes_at(&self, doc_id: DocId) -> Option<&[u8]> {
        let row_id = self.row_id(doc_id).ok()??;
        let vec_bytes = if let Some(vec_bytes) = self.vec_bytes.get() {
            vec_bytes
        } else {
            let vec_bytes = self.vec_slice().ok()?.read_bytes().ok()?;
            let _ = self.vec_bytes.set(vec_bytes);
            self.vec_bytes.get()?
        };
        let stride = self.options.bytes_per_vector();
        let start = row_id.checked_mul(stride)?;
        let end = start.checked_add(stride)?;
        vec_bytes.get(start..end)
    }

    pub fn cluster_doc_ids(&self, cluster: usize) -> crate::Result<Option<&[DocId]>> {
        if cluster >= self.meta.num_centroids {
            return Ok(None);
        }
        if self.assignments[cluster].get().is_none() {
            let docs = self.read_cluster_doc_ids(cluster)?;
            let _ = self.assignments[cluster].set(docs);
        }
        Ok(self.assignments[cluster].get().map(Vec::as_slice))
    }

    pub fn centroid_bytes(&self) -> &[u8] {
        &self.meta.centroid_bytes
    }

    pub fn cluster_vector_bytes(&self, cluster: usize) -> crate::Result<OwnedBytes> {
        if cluster >= self.meta.num_centroids {
            return Err(TantivyError::InvalidArgument(format!(
                "cluster {cluster} is out of bounds"
            )));
        }
        let stride = self.options.bytes_per_vector();
        let start = (self.meta.cluster_offset(cluster) as usize)
            .checked_mul(stride)
            .ok_or_else(|| {
                TantivyError::InternalError("IVF cluster vector byte range overflow".to_string())
            })?;
        let end = (self.meta.cluster_offset(cluster + 1) as usize)
            .checked_mul(stride)
            .ok_or_else(|| {
                TantivyError::InternalError("IVF cluster vector byte range overflow".to_string())
            })?;
        Ok(self.vec_slice()?.read_bytes_slice(start..end)?)
    }

    fn vec_slice(&self) -> crate::Result<&FileSlice> {
        let vec_slice = self.vec_slice.open()?;
        let expected_vec_bytes = self
            .options
            .bytes_per_vector()
            .checked_mul(self.meta.num_vectors())
            .ok_or_else(|| {
                TantivyError::InternalError("IVF vector byte length overflow".to_string())
            })?;
        if vec_slice.len() != expected_vec_bytes {
            return Err(TantivyError::InternalError(
                "IVF vector byte length mismatch".to_string(),
            ));
        }
        Ok(vec_slice)
    }

    fn assignments_slice(&self) -> crate::Result<&FileSlice> {
        let assignments_slice = self.assignments_slice.open()?;
        let expected_assignment_bytes = self
            .meta
            .num_vectors()
            .checked_mul(mem::size_of::<DocId>())
            .ok_or_else(|| {
                TantivyError::InternalError("IVF assignment byte length overflow".to_string())
            })?;
        if assignments_slice.len() != expected_assignment_bytes {
            return Err(TantivyError::InternalError(
                "IVF assignment byte length mismatch".to_string(),
            ));
        }
        Ok(assignments_slice)
    }

    fn read_cluster_doc_ids(&self, cluster: usize) -> crate::Result<Vec<DocId>> {
        let start = (self.meta.cluster_offset(cluster) as usize)
            .checked_mul(mem::size_of::<DocId>())
            .ok_or_else(|| {
                TantivyError::InternalError("IVF assignment byte range overflow".to_string())
            })?;
        let end = (self.meta.cluster_offset(cluster + 1) as usize)
            .checked_mul(mem::size_of::<DocId>())
            .ok_or_else(|| {
                TantivyError::InternalError("IVF assignment byte range overflow".to_string())
            })?;
        let bytes = self.assignments_slice()?.read_bytes_slice(start..end)?;
        let mut reader = bytes.as_slice();
        let mut docs = Vec::with_capacity(bytes.len() / mem::size_of::<DocId>());
        while !reader.is_empty() {
            docs.push(DocId::deserialize(&mut reader)?);
        }
        if !docs.windows(2).all(|docs| docs[0] <= docs[1]) {
            return Err(TantivyError::InternalError(
                "IVF assignments are not sorted within cluster".to_string(),
            ));
        }
        Ok(docs)
    }

    fn row_id(&self, doc_id: DocId) -> crate::Result<Option<usize>> {
        for cluster in 0..self.meta.num_centroids {
            let start = self.meta.cluster_offset(cluster) as usize;
            let Some(docs) = self.cluster_doc_ids(cluster)? else {
                continue;
            };
            if let Ok(local_row_id) = docs.binary_search(&doc_id) {
                return Ok(Some(start + local_row_id));
            }
        }
        Ok(None)
    }
}
