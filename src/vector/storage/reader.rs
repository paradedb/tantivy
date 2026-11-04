//! Per-segment vector metadata, block columns and optional IVF routing.
//!
//! Metadata opens independently of search state. The first search-reader request
//! pins routing state. Block addresses are read and validated only when accessed.
//! Clustered IdMap entries open only for document lookups. See FORMAT.md.

use std::io;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use common::{HasLen, OwnedBytes};

use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::error::DataCorruption;
use crate::fastfield::AliveBitSet;
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType, VectorOptions};
use crate::vector::storage::blocks::{BlockMetadata, Blocks, RowOffsets};
use crate::vector::storage::centroids::{ClusterIndex, ClusterOffsets};
use crate::vector::storage::header::{
    read_centroid_header, read_vector_header, CentroidSlot, VectorEntry, CENTROIDS_EXT, VEC_EXT,
};
use crate::vector::storage::id_map::{DocLocation, IdMap};
use crate::vector::storage::io::{VectorRead, VectorReadKind};
use crate::vector::storage::metadata::{SlotType, VectorColMetadata};
use crate::vector::storage::quantized::{
    exact_read_cost, range_cost, QuantizedFieldReader, QuantizedIndexCtx, QuantizedLayerReader,
    ReadCost, ReadScratch,
};
use crate::{DocId, SegmentReader, TantivyError};

/// Storage-planned views of selected fp32 rows, borrowing their rows and chunks from the read's
/// scratch.
pub(crate) struct VectorRowBatch<'s> {
    rows: &'s [usize],
    chunks: &'s [VectorRowChunk],
    stride: usize,
}

/// One planned read: a run of rows and their bytes.
pub(crate) struct VectorRowChunk {
    rows: Range<usize>,
    bytes: OwnedBytes,
}

pub(crate) struct VectorRowBatchIter<'s> {
    rows: &'s [usize],
    chunks: &'s [VectorRowChunk],
    stride: usize,
    selected: usize,
    chunk: usize,
}

impl<'s> VectorRowBatch<'s> {
    pub(crate) fn iter(&self) -> VectorRowBatchIter<'s> {
        VectorRowBatchIter {
            rows: self.rows,
            chunks: self.chunks,
            stride: self.stride,
            selected: 0,
            chunk: 0,
        }
    }

    #[cfg(test)]
    fn read_count(&self) -> usize {
        self.chunks.len()
    }
}

impl<'s> Iterator for VectorRowBatchIter<'s> {
    type Item = (usize, &'s [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let &row = self.rows.get(self.selected)?;
        while self.chunks[self.chunk].rows.end <= row {
            self.chunk += 1;
        }
        let chunk = &self.chunks[self.chunk];
        debug_assert!(chunk.rows.contains(&row));
        let local = row - chunk.rows.start;
        let start = local * self.stride;
        self.selected += 1;
        Some((row, &chunk.bytes[start..start + self.stride]))
    }
}

/// Every vector field declares exactly the two format entries.
fn validate_vector_entries(composite: &CompositeFile, field: Field) -> crate::Result<()> {
    if composite.field_indices().any(|(_, idx)| idx > 1)
        || !composite
            .field_indices()
            .any(|(f, idx)| f == field && idx == VectorEntry::IdMap.index())
        || composite
            .open_read_with_idx(field, VectorEntry::Data.index())
            .is_none()
    {
        return Err(DataCorruption::comment_only(
            "vector field requires exactly IdMap and Data entries",
        )
        .into());
    }
    Ok(())
}

/// Light field metadata with one shared, fallible search-state initialization.
pub(crate) struct VectorFieldReader {
    options: VectorOptions,
    source: Option<VectorSource>,
    search: OnceLock<crate::Result<Arc<VectorIndexReader>>>,
}

#[derive(Clone)]
struct CentroidSlices {
    centroids: FileSlice,
    offsets: FileSlice,
    router: FileSlice,
}

struct VectorSource {
    metadata: BlockMetadata,
    composite: CompositeFile,
    field: Field,
    max_doc: DocId,
    centroid_slots: Option<CentroidSlices>,
}

/// Opens an addressing entry only when a document lookup needs it; failures are shared.
struct DeferredIdMap {
    source: Option<(CompositeFile, Field, DocId)>,
    value: OnceLock<crate::Result<IdMap>>,
}
impl DeferredIdMap {
    /// Stores a map whose bytes need no deferred entry access.
    fn ready(map: IdMap) -> Self {
        Self {
            source: None,
            value: OnceLock::from(Ok(map)),
        }
    }
    /// Initializes the entry once and checks its variant against the stored partition.
    fn get(&self, clustered: bool) -> crate::Result<&IdMap> {
        self.value
            .get_or_init(|| {
                let (composite, field, max_doc) =
                    self.source.as_ref().expect("deferred entry source");
                let entry = composite
                    .open_read_with_idx(*field, VectorEntry::IdMap.index())
                    .unwrap();
                let map = IdMap::open(entry, *max_doc)
                    .map_err(|e| DataCorruption::comment_only(e.to_string()))?;
                if matches!(map, IdMap::DocLocations(_)) != clustered {
                    return Err(DataCorruption::comment_only("partition/id-map mismatch").into());
                }
                Ok(map)
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

/// Per-(segment, field) vector reader: the row store plus, for IVF segments,
/// the routing index. See the module docs for the layout and the
/// pinned-vs-deferred split.
pub struct VectorIndexReader {
    max_doc: DocId,
    pub(crate) options: VectorOptions,
    /// Distinct documents with a vector.
    pub(crate) num_vectors: usize,
    /// `false` for the placeholder built by [`Self::empty`] — the segment has
    /// no vector data for this field at all.
    pub(crate) present: bool,
    /// Document addressing, deferred until a lookup requires the entry.
    id_map: DeferredIdMap,
    /// Deferred row-group columns and their validated geometry.
    rows_slice: Arc<Blocks>,
    pub(crate) index: Option<ClusterIndex>,
    pub(crate) quantization: Option<QuantizedFieldReader>,
}

impl VectorFieldReader {
    /// Opens `field`'s vector data in `segment_reader`'s segment. Returns the
    /// metadata placeholder when the segment has no `.vec` file. The vector header is
    /// validated before reading centroid data so unsupported formats have a typed error.
    pub(crate) fn open(segment_reader: &SegmentReader, field: Field) -> crate::Result<Self> {
        let entry = segment_reader.schema().get_field_entry(field);
        let options = match entry.field_type() {
            FieldType::Vector(opts) => opts.clone(),
            _ => {
                return Err(TantivyError::InvalidArgument(format!(
                    "field {:?} is not a vector field",
                    entry.name()
                )));
            }
        };

        let vec_file = match segment_reader.open_read(SegmentComponent::Custom(VEC_EXT.to_string()))
        {
            Ok(file) => file,
            Err(OpenReadError::FileDoesNotExist(_)) => {
                return Ok(Self {
                    options,
                    source: None,
                    search: OnceLock::new(),
                })
            }
            Err(err) => return Err(err.into()),
        };
        let (_, body) = read_vector_header(&vec_file)?;

        let centroid_slots = match segment_reader
            .open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string()))
        {
            Ok(file) => {
                let (_, body) = read_centroid_header(&file)?;
                let composite = CompositeFile::open(&body)?;
                match (
                    composite.open_read_with_idx(field, CentroidSlot::Centroids.index()),
                    composite.open_read_with_idx(field, CentroidSlot::Offsets.index()),
                    composite.open_read_with_idx(field, CentroidSlot::Router.index()),
                ) {
                    (Some(centroids), Some(offsets), Some(router)) => {
                        Some(CentroidSlices {
                            centroids,
                            offsets,
                            router,
                        })
                    }
                    (Some(_), None, _) => {
                        return Err(
                            DataCorruption::comment_only("missing cluster offsets slot").into()
                        );
                    }
                    (Some(_), _, None) => {
                        return Err(TantivyError::InternalError(format!(
                            "vector field {:?} has no router slot",
                            entry.name()
                        )));
                    }
                    (None, None, None) => None,
                    _ => {
                        return Err(DataCorruption::comment_only("incomplete centroid slots").into())
                    }
                }
            }
            Err(OpenReadError::FileDoesNotExist(_)) => None,
            Err(err) => return Err(err.into()),
        };

        let vec_composite = CompositeFile::open(&body)?;
        validate_vector_entries(&vec_composite, field)?;
        let data = vec_composite
            .open_read_with_idx(field, VectorEntry::Data.index())
            .unwrap();
        let metadata = BlockMetadata::open(data, &options, centroid_slots.is_some())?;
        Ok(Self {
            options,
            source: Some(VectorSource {
                metadata,
                composite: vec_composite,
                field,
                max_doc: segment_reader.max_doc(),
                centroid_slots,
            }),
            search: OnceLock::new(),
        })
    }

    #[cfg(test)]
    pub(crate) fn metadata(&self) -> Option<Arc<VectorColMetadata>> {
        self.source
            .as_ref()
            .map(|source| Arc::clone(&source.metadata.meta))
    }

    /// Initializes and validates search bytes once, including failures shared by concurrent
    /// callers.
    pub(crate) fn search_reader(&self) -> crate::Result<Arc<VectorIndexReader>> {
        self.search
            .get_or_init(|| VectorIndexReader::open(self).map(Arc::new))
            .clone()
    }
}

impl VectorIndexReader {
    fn open(field: &VectorFieldReader) -> crate::Result<Self> {
        let options = field.options.clone();
        let Some(source) = &field.source else {
            return Ok(Self::empty(options));
        };
        let centroid_slots = source.centroid_slots.clone();
        let id_map = DeferredIdMap {
            source: Some((source.composite.clone(), source.field, source.max_doc)),
            value: OnceLock::new(),
        };
        let row_offsets = match &centroid_slots {
            Some(slots) => RowOffsets::Clustered(ClusterOffsets::open(slots.offsets.clone())?),
            None => RowOffsets::Uniform {
                num_rows: id_map.get(false)?.num_rows() as usize,
            },
        };
        let rows_slice = Arc::new(Blocks::from_metadata(source.metadata.clone(), row_offsets)?);
        let index = if let Some(slots) = centroid_slots {
            Some(ClusterIndex::open(
                &options,
                slots.centroids,
                Arc::clone(&rows_slice),
                slots.router,
            )?)
        } else {
            None
        };
        let quantization = if let VectorColMetadata::Quantized { layers: quants, .. } =
            rows_slice.metadata.as_ref()
        {
            let mut layers = Vec::new();
            for (layer, quant) in quants.iter().enumerate() {
                let mut codes = None;
                let mut scales = None;
                let mut gammas = None;
                let mut errors = None;
                let mut constants = None;
                for (idx, slot) in rows_slice.schema.iter().enumerate() {
                    match &slot.slot_type {
                        SlotType::QuantLayerCodes { layer: l, .. } if *l as usize == layer => {
                            codes = Some(idx)
                        }
                        SlotType::QuantLayerScales { layer: l } if *l as usize == layer => {
                            scales = Some(idx)
                        }
                        SlotType::QuantLayerGammas { layer: l } if *l as usize == layer => {
                            gammas = Some(idx)
                        }
                        SlotType::QuantLayerErrors { layer: l } if *l as usize == layer => {
                            errors = Some(idx)
                        }
                        SlotType::QuantLayerConstants { layer: l } if *l as usize == layer => {
                            constants = Some(idx)
                        }
                        _ => (),
                    }
                }
                layers.push(QuantizedLayerReader {
                    blocks: Arc::clone(&rows_slice),
                    layer,
                    codes: codes.unwrap(),
                    scales: scales.unwrap(),
                    gammas: gammas.unwrap(),
                    errors: errors.unwrap(),
                    constants,
                    code_stride: quant.code_stride(options.dim()),
                    dim: options.dim(),
                    bits: quant.bits(),
                });
            }
            Some(QuantizedFieldReader {
                index_ctx: Arc::new(QuantizedIndexCtx::new(Arc::clone(&rows_slice.metadata))?),
                layers,
                blocks: Arc::clone(&rows_slice),
                norms: rows_slice
                    .schema
                    .iter()
                    .position(|slot| matches!(slot.slot_type, SlotType::ResidualNorms))
                    .expect("validated residual norms column"),
            })
        } else {
            None
        };

        let num_vectors = match &index {
            Some(index) => index.num_docs(),
            None => rows_slice.num_rows(),
        };
        Ok(Self {
            max_doc: source.max_doc,
            options,
            num_vectors,
            present: true,
            rows_slice,
            id_map,
            index,
            quantization,
        })
    }

    /// A reader with no rows and no routing index; every accessor behaves as an empty column.
    pub(crate) fn empty(options: VectorOptions) -> Self {
        let mut bytes = Vec::new();
        crate::vector::storage::blocks::write_metadata(
            &mut bytes,
            &VectorColMetadata::build_flat(&options),
        )
        .unwrap();
        let end = bytes.len();
        crate::vector::storage::blocks::BlockDirectory::new(end as u64)
            .finish(&mut bytes)
            .unwrap();
        let rows_slice = Arc::new(
            Blocks::open(
                FileSlice::from(bytes),
                &options,
                RowOffsets::Uniform { num_rows: 0 },
            )
            .unwrap(),
        );
        Self {
            max_doc: 0,
            options,
            num_vectors: 0,
            present: false,
            rows_slice,
            id_map: DeferredIdMap::ready(IdMap::Identity { num_docs: 0 }),
            index: None,
            quantization: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn id_map_initialized(&self) -> bool {
        self.id_map.value.get().is_some()
    }

    /// Returns this segment's stored field metadata, independent of index build settings.
    /// `None` means the segment has no stored vector data; an empty stored field still
    /// returns its metadata.
    pub(crate) fn metadata(&self) -> Option<&VectorColMetadata> {
        self.present.then_some(self.rows_slice.metadata.as_ref())
    }

    /// Returns the field options.
    pub fn options(&self) -> &VectorOptions {
        &self.options
    }

    /// Returns the vector dimension.
    pub fn dim(&self) -> usize {
        self.options.dim()
    }

    /// Number of distinct docs with a vector value.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Returns whether the field contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// The routing index, present iff the segment's rows are IVF-clustered; `None` means search
    /// scans the rows exactly.
    pub(crate) fn index(&self) -> Option<&ClusterIndex> {
        self.index.as_ref()
    }

    pub(crate) fn quantization(&self) -> Option<&QuantizedFieldReader> {
        self.quantization.as_ref()
    }

    /// Whether this segment contains the field's quantized storage slots. Field
    /// policy can enable quantization while a small flat segment stores no codes.
    pub fn has_quantized_storage(&self) -> bool {
        self.quantization.is_some()
    }

    /// Each cluster's posting-list size, in cluster order; `None` when the storage is not IVF.
    pub fn cluster_sizes(&self) -> crate::Result<Option<Vec<u32>>> {
        self.index
            .as_ref()
            .map(|index| {
                index
                    .cluster_sizes()
                    .map(|size| size.map(|n| n as u32))
                    .collect()
            })
            .transpose()
    }

    /// Each cluster's ball radius around its stored centroid, in cluster order, with
    /// `f32::INFINITY` for a saturated cluster; `None` when the storage is not IVF.
    pub fn cluster_radii(&self) -> Option<Vec<f32>> {
        self.index.as_ref().map(|index| {
            let bounds = index.bounds();
            (0..index.num_clusters())
                .map(|cluster| bounds.ball_r(cluster))
                .collect()
        })
    }

    /// Returns the number of live IVF posting rows.
    pub fn live_posting_row_count(&self, alive: Option<&AliveBitSet>) -> crate::Result<usize> {
        let Some(index) = &self.index else {
            return Ok(0);
        };
        let Some(alive) = alive else {
            return Ok(index.num_rows());
        };
        let mut docs = Vec::new();
        let mut count = 0;
        for cluster in 0..index.num_clusters() {
            self.read_doc_ids(cluster, &mut docs)?;
            count += docs.iter().filter(|&&doc| alive.is_alive(doc)).count();
        }
        Ok(count)
    }

    /// Returns whether the document has a vector, propagating storage errors.
    pub fn contains(&self, doc_id: DocId) -> crate::Result<bool> {
        Ok(self.row_id(doc_id)?.is_some())
    }

    /// Returns one document's raw little-endian vector bytes, or None if absent.
    ///
    /// # Errors
    ///
    /// Returns an error when the vector row cannot be read.
    pub fn vector_bytes(&self, doc_id: DocId) -> crate::Result<Option<OwnedBytes>> {
        if self.rows_slice.clustered() {
            let Some(location) = self
                .id_map
                .get(true)?
                .locate(doc_id, |cluster| {
                    self.rows_slice.row_range(cluster).map_err(io::Error::other)
                })
                .map_err(|e| DataCorruption::comment_only(e.to_string()))?
            else {
                return Ok(None);
            };
            let stride = self.options.bytes_per_vector();
            let start = location.local as usize * stride;
            return Ok(Some(
                self.rows_slice
                    .block(location.cluster as usize)?
                    .column(0)
                    .read_vector_bytes_slice(
                        start..start + stride,
                        VectorReadKind::FullPrecision,
                    )?,
            ));
        }
        let Some(row) = self.row_id(doc_id)? else {
            return Ok(None);
        };
        self.vector_bytes_for_row(row).map(Some)
    }

    /// Returns one dense vector row by row index without a document lookup.
    ///
    /// # Errors
    ///
    /// Returns an error when `row` is out of bounds or cannot be read.
    pub fn vector_bytes_for_row(&self, row: usize) -> crate::Result<OwnedBytes> {
        if row >= self.rows_slice.num_rows() {
            return Err(TantivyError::InvalidArgument(format!(
                "vector row {row} is out of bounds"
            )));
        }
        self.rows_slice.read_column(0, row..row + 1)
    }

    /// Fetches the increasing vector rows in `scratch.rows` through a storage-aware range plan. The
    /// batch borrows its rows and the read chunks from `scratch`.
    pub(crate) fn read_vector_rows_planned<'s>(
        &self,
        scratch: &'s mut ReadScratch,
    ) -> crate::Result<VectorRowBatch<'s>> {
        self.read_vector_rows(scratch, None)
    }
    pub(crate) fn read_vector_rows_in_cluster<'s>(
        &self,
        cluster: usize,
        scratch: &'s mut ReadScratch,
    ) -> crate::Result<VectorRowBatch<'s>> {
        self.read_vector_rows(scratch, Some(cluster))
    }
    fn read_vector_rows<'s>(
        &self,
        scratch: &'s mut ReadScratch,
        cluster: Option<usize>,
    ) -> crate::Result<VectorRowBatch<'s>> {
        let ReadScratch {
            rows,
            runs: read_ranges,
            blocks: block_scratch,
            chunks,
            ..
        } = scratch;
        let rows: &'s [usize] = rows;
        let num_rows = self.rows_slice.num_rows();
        if rows.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(TantivyError::InvalidArgument(
                "planned vector rows must be strictly increasing".to_string(),
            ));
        }
        let stride = self.options.bytes_per_vector();
        read_ranges.clear();
        chunks.clear();
        if let Some(&row) = rows.last() {
            if row >= num_rows {
                return Err(TantivyError::InvalidArgument(format!(
                    "vector row {row} is out of bounds"
                )));
            }
        } else {
            block_scratch.clear();
            return Ok(VectorRowBatch {
                rows,
                chunks,
                stride,
            });
        }

        let mut selected = rows;
        while let Some(&row) = selected.first() {
            let block = match cluster {
                Some(cluster) => {
                    let block = self.rows_slice.block(cluster)?;
                    block.check_rows(&(row..rows.last().unwrap() + 1))?;
                    block
                }
                None => self.rows_slice.block(self.rows_slice.block_of(row)?)?,
            };
            let first = block.row_range().start;
            let end = block.row_range().end;
            let count = selected.partition_point(|&r| r < end);
            let column = block.column(0);
            let range_start = read_ranges.len();
            QuantizedLayerReader::plan_slot_reads(
                &column,
                stride,
                first,
                first..end,
                &selected[..count],
                read_ranges,
                block_scratch,
            );
            // The plan already establishes the block and row origin; reads keep that address.
            for row_range in read_ranges[range_start..].iter().cloned() {
                let bytes = column.read_vector_bytes_slice(
                    (row_range.start - first) * stride..(row_range.end - first) * stride,
                    VectorReadKind::FullPrecision,
                )?;
                chunks.push(VectorRowChunk {
                    rows: row_range,
                    bytes,
                });
            }
            selected = &selected[count..];
        }
        Ok(VectorRowBatch {
            rows,
            chunks,
            stride,
        })
    }

    /// Reads a clustered document id from its Data column, or selects a flat bitmap row.
    pub fn doc_id_at(&self, row: usize) -> crate::Result<DocId> {
        if row >= self.num_vectors() {
            return Err(TantivyError::InvalidArgument(format!(
                "vector row {row} is out of bounds"
            )));
        }
        Ok(self.row_doc_ids(row..row + 1)?[0])
    }
    /// Reads and validates ascending document ids without opening the clustered IdMap.
    pub(crate) fn row_doc_ids(&self, rows: Range<usize>) -> crate::Result<Vec<DocId>> {
        if !self.rows_slice.clustered() {
            let map = self.id_map.get(false)?;
            return Ok(rows.map(|row| map.doc_at(row as u32)).collect());
        }
        let mut result = Vec::with_capacity(rows.len());
        let mut docs = Vec::new();
        let mut row = rows.start;
        while row < rows.end {
            let cluster = self.rows_slice.block_of(row)?;
            self.read_doc_ids(cluster, &mut docs)?;
            let block_rows = self.rows_slice.row_range(cluster)?;
            let first = block_rows.start;
            let end = rows.end.min(block_rows.end);
            result.extend_from_slice(&docs[row - first..end - first]);
            row = end;
        }
        Ok(result)
    }
    /// Returns sorted document ids assigned to a cluster, or None for an invalid cluster.
    pub fn cluster_doc_ids(&self, cluster: usize) -> crate::Result<Option<Vec<DocId>>> {
        let Some(index) = &self.index else {
            return Ok(None);
        };
        if cluster >= index.num_clusters() {
            return Ok(None);
        }
        let mut docs = Vec::new();
        self.read_doc_ids(cluster, &mut docs)?;
        Ok(Some(docs))
    }
    /// Resolves one document by direct location or flat bitmap rank, validating stored coordinates.
    pub(crate) fn row_id(&self, doc_id: DocId) -> crate::Result<Option<usize>> {
        let clustered = self.rows_slice.clustered();
        let map = self.id_map.get(clustered)?;
        if clustered {
            let location = map
                .locate(doc_id, |cluster| {
                    self.rows_slice.row_range(cluster).map_err(io::Error::other)
                })
                .map_err(|e| DataCorruption::comment_only(e.to_string()))?;
            location
                .map(|loc| {
                    Ok(self.rows_slice.row_range(loc.cluster as usize)?.start + loc.local as usize)
                })
                .transpose()
        } else {
            Ok(map.rank_if_exists(doc_id).map(|row| row as usize))
        }
    }
    /// Reads only a cluster's document column, validating bounds and ordering while decoding.
    pub(crate) fn read_doc_ids(&self, cluster: usize, out: &mut Vec<DocId>) -> crate::Result<()> {
        out.clear();
        let blocks = &self.rows_slice;
        if !blocks.clustered() || cluster >= blocks.num_blocks() {
            return Err(DataCorruption::comment_only("invalid DocIds cluster").into());
        }
        let block = blocks.block(cluster)?;
        if block.num_rows() == 0 {
            return Ok(());
        }
        let idx = blocks
            .schema
            .iter()
            .position(|slot| matches!(slot.slot_type, SlotType::DocIds))
            .ok_or_else(|| DataCorruption::comment_only("missing DocIds column"))?;
        let bytes = block.read_column(idx, block.row_range())?;
        out.reserve(block.num_rows());
        let mut previous = None;
        for bytes in bytes.chunks_exact(std::mem::size_of::<DocId>()) {
            let doc = DocId::from_le_bytes(bytes.try_into().unwrap());
            if doc >= self.max_doc || previous.is_some_and(|prev| prev >= doc) {
                return Err(DataCorruption::comment_only(format!(
                    "cluster {cluster} DocIds must ascend and be below max_doc {}",
                    self.max_doc
                ))
                .into());
            }
            out.push(doc);
            previous = Some(doc);
        }
        Ok(())
    }

    /// The cost of reading `cluster`'s full-precision rows at `offsets` exactly; `None` reads
    /// every row as one span.
    pub(crate) fn exact_cost(
        &self,
        cluster: usize,
        offsets: Option<&[u32]>,
    ) -> crate::Result<ReadCost> {
        let block = self.rows_slice.block(cluster)?;
        Ok(exact_read_cost(
            &block.column(0),
            self.options.bytes_per_vector(),
            offsets,
            block.num_rows(),
        ))
    }

    /// The cost of reading `cluster`'s centroid row.
    pub(crate) fn centroid_cost(&self, cluster: usize) -> crate::Result<ReadCost> {
        let index = self
            .index()
            .ok_or_else(|| DataCorruption::comment_only("centroid rows need an IVF index"))?;
        let row = index.centroid_row_slice(cluster)?;
        // Priced in the unit of the cluster's rows, whatever storage the centroids sit on.
        let unit = self
            .rows_slice
            .block(cluster)?
            .column(0)
            .storage_block_len();
        Ok(range_cost(&row, 0..row.len(), unit))
    }

    /// Locates ascending `docs` through the location map, appending `(doc, location)` for each
    /// one with a vector; an invalid map entry is `DataCorruption`.
    pub(crate) fn locate_docs(
        &self,
        docs: &[DocId],
        out: &mut Vec<(DocId, DocLocation)>,
    ) -> crate::Result<()> {
        self.id_map
            .get(true)?
            .locate_many(
                docs,
                |cluster| self.rows_slice.row_range(cluster).map_err(io::Error::other),
                out,
            )
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::InvalidData => {
                    DataCorruption::comment_only(error.to_string()).into()
                }
                _ => error.into(),
            })
    }

    /// Reads a cluster's full-precision rows without document ids or quantized columns.
    pub(crate) fn read_cluster_rows(&self, cluster: usize) -> crate::Result<OwnedBytes> {
        let block = self.rows_slice.block(cluster)?;
        block.read_column(0, block.row_range())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::directory::CompositeWrite;
    use crate::schema::Metric;
    use crate::vector::search::prepared::VectorQuery;
    use crate::vector::storage::io::test_support::BlockTrackedBytes;

    fn tracked_id_composite(ids: Vec<u8>, reads: &Arc<Mutex<Vec<Range<usize>>>>) -> CompositeFile {
        let mut bytes = Vec::new();
        let mut writer = crate::directory::CompositeWrite::wrap(&mut bytes);
        writer
            .for_field_with_idx(Field::from_field_id(0), VectorEntry::IdMap.index())
            .write_all(&ids)
            .unwrap();
        writer.close().unwrap();
        let composite = CompositeFile::open(&FileSlice::new(Arc::new(BlockTrackedBytes {
            bytes,
            reads: Arc::clone(reads),
            block_len: 16,
        })))
        .unwrap();
        reads.lock().unwrap().clear();
        composite
    }

    // Opening reads only the tag; a lookup reads exactly its eight-byte record.
    #[test]
    fn location_lookup_reads_one_record_and_defers_entry() -> crate::Result<()> {
        use crate::vector::storage::id_map::DocLocation;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut bytes = Vec::new();
        IdMap::serialize_locations(
            &[
                DocLocation::ABSENT,
                DocLocation {
                    cluster: 0,
                    local: 0,
                },
                DocLocation {
                    cluster: 0,
                    local: 1,
                },
            ],
            &mut bytes,
        )?;
        let map = DeferredIdMap {
            source: Some((
                tracked_id_composite(bytes, &reads),
                Field::from_field_id(0),
                3,
            )),
            value: OnceLock::new(),
        };
        assert!(reads.lock().unwrap().is_empty());
        let opened = map.get(true)?;
        assert_eq!(&*reads.lock().unwrap(), &[0..1]);
        reads.lock().unwrap().clear();
        assert_eq!(
            opened.locate(
                2,
                crate::vector::storage::id_map::tests::boundaries(&[0, 2])
            )?,
            Some(DocLocation {
                cluster: 0,
                local: 1
            })
        );
        assert_eq!(&*reads.lock().unwrap(), &[17..25]);
        Ok(())
    }

    #[test]
    fn doc_ids_reject_out_of_bounds_and_nonascending_values() -> crate::Result<()> {
        use crate::vector::storage::blocks::{
            block_len, column_range, pad, write_metadata, BlockDirectory,
        };
        for docs in [[0u32, 1, 3], [0, 0, 2], [1, 0, 2]] {
            let options = VectorOptions::new(2, Metric::L2);
            let meta = VectorColMetadata::build_ivf(&options, None)?;
            let slots = meta.slots();
            let mut data = Vec::new();
            let first = write_metadata(&mut data, &meta)?;
            for (idx, slot) in slots.iter().enumerate() {
                let range = column_range(&slots, docs.len(), idx);
                let padding = first + range.start - data.len();
                pad(&mut data, padding)?;
                if matches!(slot.slot_type, SlotType::DocIds) {
                    for doc in docs {
                        data.extend_from_slice(&doc.to_le_bytes());
                    }
                } else {
                    pad(&mut data, range.len())?;
                }
            }
            let padding = first + block_len(&slots, docs.len()) - data.len();
            pad(&mut data, padding)?;
            let mut directory = BlockDirectory::new(first as u64);
            directory.push(data.len() as u64);
            directory.finish(&mut data)?;
            let blocks = Arc::new(Blocks::open(
                FileSlice::from(data),
                &options,
                RowOffsets::clusters(&[0, docs.len() as u64]),
            )?);
            let reader = VectorIndexReader {
                max_doc: 3,
                options,
                num_vectors: 3,
                present: true,
                rows_slice: blocks,
                id_map: DeferredIdMap::ready(IdMap::Identity { num_docs: 3 }),
                index: None,
                quantization: None,
            };
            let mut decoded = Vec::new();
            let error = reader.read_doc_ids(0, &mut decoded).unwrap_err();
            assert!(
                matches!(error, TantivyError::DataCorruption(_)),
                "column=DocIds values={docs:?}: {error}"
            );
            assert!(matches!(
                reader.doc_id_at(0),
                Err(TantivyError::DataCorruption(_))
            ));
            assert!(matches!(
                reader.doc_id_at(3),
                Err(TantivyError::InvalidArgument(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn metadata_defers_search_bytes_and_shares_initialization() -> crate::Result<()> {
        let options = VectorOptions::new(3, Metric::L2);
        let mut data = Vec::new();
        let prefix = crate::vector::storage::blocks::write_metadata(
            &mut data,
            &VectorColMetadata::build_flat(&options),
        )?;
        data.extend([0; 12]);
        let end = data.len();
        let mut directory = crate::vector::storage::blocks::BlockDirectory::new(prefix as u64);
        directory.push(end as u64);
        directory.finish(&mut data)?;
        let footer = data.len() - 8;
        let directory_start = footer - 2 * 8;
        let mut ids = Vec::new();
        IdMap::serialize(&[1], 3, &mut ids)?;
        let data_reads = Arc::new(Mutex::new(Vec::new()));
        let id_reads = Arc::new(Mutex::new(Vec::new()));
        let metadata = BlockMetadata::open(
            FileSlice::new(Arc::new(BlockTrackedBytes {
                bytes: data,
                reads: Arc::clone(&data_reads),
                block_len: 16,
            })),
            &options,
            false,
        )?;
        let field = Arc::new(VectorFieldReader {
            options,
            source: Some(VectorSource {
                metadata,
                composite: tracked_id_composite(ids, &id_reads),
                field: Field::from_field_id(0),
                max_doc: 3,
                centroid_slots: None,
            }),
            search: OnceLock::new(),
        });
        assert!(field.metadata().is_some());
        assert!(field.search.get().is_none());
        assert!(id_reads.lock().unwrap().is_empty());
        assert!(data_reads.lock().unwrap().iter().all(|r| r.end <= prefix));
        let readers = std::thread::scope(|scope| {
            (0..8)
                .map(|_| scope.spawn(|| field.search_reader().unwrap()))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(readers.iter().all(|r| Arc::ptr_eq(r, &readers[0])));
        let reads = data_reads.lock().unwrap();
        assert_eq!(
            reads.iter().filter(|r| **r == (footer..footer + 8)).count(),
            1
        );
        assert_eq!(
            reads
                .iter()
                .filter(|r| r.start < footer && r.end > directory_start)
                .count(),
            0
        );
        drop(reads);
        assert_eq!(
            id_reads
                .lock()
                .unwrap()
                .iter()
                .filter(|r| **r == (0..1))
                .count(),
            1
        );
        assert_eq!(readers[0].num_vectors(), 1);
        assert_eq!(readers[0].row_id(1)?, Some(0));
        Ok(())
    }

    #[test]
    fn lazy_search_rejects_corrupt_geometry_and_caches_error() -> crate::Result<()> {
        let options = VectorOptions::new(3, Metric::L2);
        let mut data = Vec::new();
        crate::vector::storage::blocks::write_metadata(
            &mut data,
            &VectorColMetadata::build_flat(&options),
        )?;
        let end = data.len();
        crate::vector::storage::blocks::BlockDirectory::new(end as u64).finish(&mut data)?;
        let id_reads = Arc::new(Mutex::new(Vec::new()));
        let field = VectorFieldReader {
            source: Some(VectorSource {
                metadata: BlockMetadata::open(FileSlice::from(data), &options, false)?,
                composite: tracked_id_composite(vec![0], &id_reads),
                field: Field::from_field_id(0),
                max_doc: 1,
                centroid_slots: None,
            }),
            options,
            search: OnceLock::new(),
        };
        assert!(field.metadata().is_some());
        for _ in 0..2 {
            assert!(matches!(
                field.search_reader(),
                Err(TantivyError::DataCorruption(_))
            ));
        }
        assert_eq!(id_reads.lock().unwrap().len(), 1);
        Ok(())
    }

    #[test]
    fn planned_vector_rows_coalesce_reads_and_preserve_exact_scores() -> crate::Result<()> {
        const DIM: usize = 2;
        const ROWS: usize = 16;
        let row_bytes = (0..ROWS)
            .flat_map(|row| {
                let values = [row as f32 + 0.25, row as f32 * -0.5 + 1.0];
                values.into_iter().flat_map(f32::to_le_bytes)
            })
            .collect::<Vec<_>>();
        let expected_bytes = row_bytes.clone();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let opts = VectorOptions::new(DIM, Metric::Dot);
        let mut data = Vec::new();
        let prefix = crate::vector::storage::blocks::write_metadata(
            &mut data,
            &VectorColMetadata::build_flat(&opts),
        )?;
        data.extend_from_slice(&expected_bytes);
        let mut directory = crate::vector::storage::blocks::BlockDirectory::new(prefix as u64);
        directory.push(data.len() as u64);
        directory.finish(&mut data)?;
        let storage = Arc::new(BlockTrackedBytes {
            bytes: data,
            reads: Arc::clone(&reads),
            block_len: 32,
        });
        let blocks = Arc::new(Blocks::open(
            FileSlice::new(storage),
            &opts,
            RowOffsets::Uniform { num_rows: ROWS },
        )?);
        reads.lock().unwrap().clear();
        let reader = VectorIndexReader {
            max_doc: ROWS as DocId,
            options: VectorOptions::new(DIM, Metric::Dot),
            num_vectors: ROWS,
            present: true,
            id_map: DeferredIdMap::ready(IdMap::Identity {
                num_docs: ROWS as u32,
            }),
            rows_slice: blocks,
            index: None,
            quantization: None,
        };
        let selected = [1, 2, 10, 11];
        let mut scratch = ReadScratch::default();
        scratch.rows.extend(selected);
        let batch = reader.read_vector_rows_planned(&mut scratch)?;

        assert_eq!(batch.read_count(), 2);
        assert!(batch.read_count() < selected.len());

        let query = VectorQuery::<f32>::new(Metric::Dot, Arc::new(vec![2.0, -0.5]));
        let actual = batch
            .iter()
            .map(|(row, bytes)| (row, query.score_row(bytes).score().to_bits()))
            .collect::<Vec<_>>();
        assert_eq!(scratch.runs, [1..3, 10..12]);
        let blocks_end = prefix + expected_bytes.len();
        let offsets_start = crate::vector::storage::blocks::align_up(blocks_end, 8);
        assert_eq!(
            &*reads.lock().unwrap(),
            &[
                // One offset pair and terminal alignment check, then the two planned runs.
                offsets_start..offsets_start + 16,
                blocks_end..offsets_start,
                prefix + 8..prefix + 24,
                prefix + 80..prefix + 96,
            ]
        );
        let stride = DIM * std::mem::size_of::<f32>();
        let expected = selected
            .iter()
            .map(|&row| {
                let bytes = &expected_bytes[row * stride..(row + 1) * stride];
                (row, query.score_row(bytes).score().to_bits())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    // Missing entries and unknown entry indices are corruption regardless of field order.
    #[test]
    fn vector_entries_require_exact_pair() -> crate::Result<()> {
        let field = Field::from_field_id(0);
        for entries in [vec![0], vec![1], vec![0, 1, 2], vec![1, 0]] {
            let mut bytes = Vec::new();
            let mut writer = CompositeWrite::wrap(&mut bytes);
            for &idx in &entries {
                writer.for_field_with_idx(field, idx).write_all(&[0])?;
            }
            writer.close()?;
            let composite = CompositeFile::open(&FileSlice::from(bytes))?;
            assert_eq!(
                validate_vector_entries(&composite, field).is_ok(),
                entries == [1, 0]
            );
        }
        Ok(())
    }
}
