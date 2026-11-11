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
use crate::vector::storage::blocks::{Block, BlockMetadata, Blocks};
use crate::vector::storage::centroids::ClusterIndex;
use crate::vector::storage::header::{
    read_centroid_header, read_vector_header, CentroidSlot, VectorEntry, CENTROIDS_EXT, VEC_EXT,
};
use crate::vector::storage::id_map::IdMap;
use crate::vector::storage::io::{VectorRead, VectorReadKind};
use crate::vector::storage::metadata::{SlotType, VectorColMetadata};
use crate::vector::storage::quantized::{
    exact_read_cost, range_cost, QuantizedFieldReader, QuantizedIndexCtx, QuantizedLayerReader,
    ReadCost, ReadScratch,
};
use crate::vector::BlockAddress;
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

/// Stored vector counts, available without opening routing or row payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorIndexStats {
    /// Distinct documents with a vector value.
    pub num_vectors: usize,
    /// IVF cluster count, or `None` for flat storage.
    pub num_centroids: Option<usize>,
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
    quantization: OnceLock<crate::Result<Option<QuantizedFieldReader>>>,
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
                if composite
                    .field_indices()
                    .any(|(f, slot)| f == field && !matches!(slot, 0 | 2))
                {
                    return Err(DataCorruption::comment_only("unexpected centroid slot").into());
                }
                match (
                    composite.open_read_with_idx(field, CentroidSlot::Centroids.index()),
                    composite.open_read_with_idx(field, CentroidSlot::Router.index()),
                ) {
                    (Some(centroids), Some(router)) => Some(CentroidSlices { centroids, router }),
                    (Some(_), None) => {
                        return Err(TantivyError::InternalError(format!(
                            "vector field {:?} has no router slot",
                            entry.name()
                        )));
                    }
                    (None, None) => None,
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

    pub(crate) fn stats(&self) -> crate::Result<Option<VectorIndexStats>> {
        let Some(source) = &self.source else {
            return Ok(None);
        };
        let blocks = Blocks::from_metadata(source.metadata.clone())?;
        let stats = if let Some(slots) = &source.centroid_slots {
            let (num_centroids, num_vectors) =
                ClusterIndex::read_counts(&self.options, &slots.centroids)?;
            if num_centroids != blocks.num_blocks() || num_vectors > blocks.num_rows() {
                return Err(DataCorruption::comment_only("invalid IVF vector counts").into());
            }
            VectorIndexStats {
                num_vectors,
                num_centroids: Some(num_centroids),
            }
        } else {
            VectorIndexStats {
                num_vectors: blocks.num_rows(),
                num_centroids: None,
            }
        };
        Ok(Some(stats))
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
        let rows_slice = Arc::new(Blocks::from_metadata(source.metadata.clone())?);
        let index = if let Some(slots) = centroid_slots {
            Some(ClusterIndex::open(
                &options,
                slots.centroids,
                rows_slice.block_offsets(),
                slots.router,
            )?)
        } else {
            if rows_slice.num_rows() != id_map.get(false)?.num_rows() as usize {
                return Err(DataCorruption::comment_only("flat vector row count mismatch").into());
            }
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
            quantization: OnceLock::new(),
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
        crate::vector::storage::blocks::BlockOffsetsWriter::new(end as u64)
            .finish(&mut bytes)
            .unwrap();
        let rows_slice = Arc::new(Blocks::open(FileSlice::from(bytes), &options, false).unwrap());
        Self {
            max_doc: 0,
            options,
            num_vectors: 0,
            present: false,
            rows_slice,
            id_map: DeferredIdMap::ready(IdMap::Identity { num_docs: 0 }),
            index: None,
            quantization: OnceLock::new(),
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
    pub(crate) fn num_blocks(&self) -> usize {
        self.rows_slice.num_blocks()
    }
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

    pub(crate) fn quantization(&self) -> crate::Result<Option<&QuantizedFieldReader>> {
        self.quantization
            .get_or_init(|| self.open_quantization())
            .as_ref()
            .map(Option::as_ref)
            .map_err(Clone::clone)
    }

    fn open_quantization(&self) -> crate::Result<Option<QuantizedFieldReader>> {
        let quantization = if let VectorColMetadata::Quantized { layers: quants, .. } =
            self.rows_slice.metadata.as_ref()
        {
            let mut layers = Vec::new();
            for (layer, quant) in quants.iter().enumerate() {
                let mut codes = None;
                let mut scales = None;
                let mut gammas = None;
                let mut errors = None;
                let mut constants = None;
                for (idx, slot) in self.rows_slice.schema.iter().enumerate() {
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
                    blocks: Arc::clone(&self.rows_slice),
                    layer,
                    codes: codes.unwrap(),
                    scales: scales.unwrap(),
                    gammas: gammas.unwrap(),
                    errors: errors.unwrap(),
                    constants,
                    code_stride: quant.code_stride(self.options.dim()),
                    dim: self.options.dim(),
                    bits: quant.bits(),
                });
            }
            Some(QuantizedFieldReader {
                index_ctx: Arc::new(QuantizedIndexCtx::new(Arc::clone(
                    &self.rows_slice.metadata,
                ))?),
                layers,
                blocks: Arc::clone(&self.rows_slice),
                norms: self
                    .rows_slice
                    .schema
                    .iter()
                    .position(|slot| matches!(slot.slot_type, SlotType::ResidualNorms))
                    .expect("validated residual norms column"),
            })
        } else {
            None
        };

        Ok(quantization)
    }

    #[cfg(test)]
    pub(crate) fn quantization_initialized(&self) -> bool {
        self.quantization.get().is_some()
    }

    /// Whether this segment contains the field's quantized storage slots. Field
    /// policy can enable quantization while a small flat segment stores no codes.
    pub fn has_quantized_storage(&self) -> bool {
        matches!(
            self.rows_slice.metadata.as_ref(),
            VectorColMetadata::Quantized { .. }
        )
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
        Ok(self.row_location(doc_id)?.is_some())
    }

    /// Reads a document's vector through its stored block-local location.
    pub fn vector_bytes(&self, doc_id: DocId) -> crate::Result<Option<OwnedBytes>> {
        self.row_location(doc_id)?
            .map(|address| self.vector_bytes_for_row(address))
            .transpose()
    }

    /// Reads one full-precision vector at a block-local row.
    pub fn vector_bytes_for_row(&self, address: BlockAddress) -> crate::Result<OwnedBytes> {
        let (block, row) = (address.block as usize, address.row as usize);
        if block >= self.num_blocks() {
            return Err(TantivyError::InvalidArgument(
                "vector block is out of bounds".into(),
            ));
        }
        let view = self.rows_slice.block(block)?;
        if row >= view.num_rows() {
            return Err(TantivyError::InvalidArgument(
                "vector row is out of bounds".into(),
            ));
        }
        view.read_column(0, row..row + 1)
    }

    /// Fetches increasing local rows from one known block using storage-aware range reads.
    pub(crate) fn read_vector_rows_planned<'s>(
        &self,
        block: usize,
        scratch: &'s mut ReadScratch,
    ) -> crate::Result<VectorRowBatch<'s>> {
        self.read_vector_rows_in_cluster(block, scratch)
    }
    pub(crate) fn read_vector_rows_in_cluster<'s>(
        &self,
        cluster: usize,
        scratch: &'s mut ReadScratch,
    ) -> crate::Result<VectorRowBatch<'s>> {
        let ReadScratch {
            rows,
            runs: ranges,
            blocks: block_scratch,
            chunks,
            ..
        } = scratch;
        if rows.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(TantivyError::InvalidArgument(
                "planned vector rows must be strictly increasing".to_string(),
            ));
        }
        let stride = self.options.bytes_per_vector();
        ranges.clear();
        chunks.clear();
        block_scratch.clear();
        if let Some(&last) = rows.last() {
            let block = self.rows_slice.block(cluster)?;
            block.check_rows(&(rows[0]..last + 1))?;
            let column = block.column(0);
            QuantizedLayerReader::plan_slot_reads(
                &column,
                stride,
                0,
                block.row_range(),
                rows,
                ranges,
                block_scratch,
            );
            for range in ranges.iter().cloned() {
                let bytes = column.read_vector_bytes_slice(
                    range.start * stride..range.end * stride,
                    VectorReadKind::FullPrecision,
                )?;
                chunks.push(VectorRowChunk { rows: range, bytes });
            }
        }
        Ok(VectorRowBatch {
            rows,
            chunks,
            stride,
        })
    }

    pub fn doc_id_at(&self, address: BlockAddress) -> crate::Result<DocId> {
        let (block, row) = (address.block as usize, address.row as usize);
        if block >= self.num_blocks() {
            return Err(TantivyError::InvalidArgument(
                "vector block is out of bounds".into(),
            ));
        }
        let view = self.rows_slice.block(block)?;
        if row >= view.num_rows() {
            return Err(TantivyError::InvalidArgument(
                "vector row is out of bounds".into(),
            ));
        }
        let mut docs = Vec::new();
        self.read_doc_ids_from_block(block, &view, &mut docs)?;
        Ok(docs[row])
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
    /// Resolves a document to a block and local row without global row numbering.
    pub(crate) fn row_location(&self, doc_id: DocId) -> crate::Result<Option<BlockAddress>> {
        if doc_id >= self.max_doc {
            return Ok(None);
        }
        let clustered = self.rows_slice.clustered();
        let map = self.id_map.get(clustered)?;
        if clustered {
            let location = map
                .locate(doc_id)
                .map_err(|e| DataCorruption::comment_only(e.to_string()))?;
            if let Some(address) = location {
                self.validate_location(address)?;
            }
            Ok(location)
        } else {
            let crate::vector::storage::metadata::Partition::Uniform { rows_per_block } =
                self.rows_slice.metadata.field().partition
            else {
                unreachable!()
            };
            Ok(map
                .rank_if_exists(doc_id)
                .map(|row| BlockAddress::new(row / rows_per_block, row % rows_per_block)))
        }
    }

    /// Checks one address against its block's stored row count.
    fn validate_location(&self, address: BlockAddress) -> crate::Result<()> {
        let rows = self.rows_slice.row_range(address.block as usize)?;
        if !rows.contains(&(address.row as usize)) {
            return Err(
                DataCorruption::comment_only("document location is outside its block").into(),
            );
        }
        Ok(())
    }
    /// Reads a block's document ids from its column or the flat presence map.
    pub(crate) fn read_doc_ids(&self, cluster: usize, out: &mut Vec<DocId>) -> crate::Result<()> {
        let block = self.rows_slice.block(cluster)?;
        self.read_doc_ids_from_block(cluster, &block, out)
    }

    fn read_doc_ids_from_block(
        &self,
        cluster: usize,
        block: &Block<'_>,
        out: &mut Vec<DocId>,
    ) -> crate::Result<()> {
        out.clear();
        let blocks = &self.rows_slice;
        if !blocks.clustered() {
            let crate::vector::storage::metadata::Partition::Uniform { rows_per_block } =
                blocks.metadata.field().partition
            else {
                unreachable!()
            };
            let first = cluster * rows_per_block as usize;
            let map = self.id_map.get(false)?;
            out.extend((first..first + block.num_rows()).map(|row| map.doc_at(row as u32)));
            return Ok(());
        }

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

    /// Appends locations for ascending `docs` in block/row order, validating each block once.
    /// An invalid map entry is `DataCorruption`.
    pub(crate) fn locate_docs(
        &self,
        docs: &[DocId],
        out: &mut Vec<(DocId, BlockAddress)>,
    ) -> crate::Result<()> {
        let start = out.len();
        if let Err(error) = self.id_map.get(true)?.locate_many(docs, out) {
            out.truncate(start);
            return Err(match error.kind() {
                io::ErrorKind::InvalidData => {
                    DataCorruption::comment_only(error.to_string()).into()
                }
                _ => error.into(),
            });
        }
        out[start..].sort_unstable_by_key(|&(_, address)| address);
        for group in out[start..].chunk_by(|a, b| a.1.block == b.1.block) {
            // Row order makes the last row sufficient to validate the entire block group.
            if let Err(error) = self.validate_location(group[group.len() - 1].1) {
                out.truncate(start);
                return Err(error);
            }
        }
        Ok(())
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
    use crate::vector::storage::io::test_support::{BlockTrackedBytes, TrackedReads};

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
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut bytes = Vec::new();
        let mut locations = IdMap::empty_locations(3);
        locations[1] = BlockAddress::new(0, 0);
        locations[2] = BlockAddress::new(0, 1);
        IdMap::serialize_locations(&locations, &mut bytes)?;
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
        assert_eq!(opened.locate(2)?, Some(BlockAddress::new(0, 1)));
        assert_eq!(&*reads.lock().unwrap(), &[17..25]);
        Ok(())
    }

    /// A metadata-only fixture: location validation must never read block payloads.
    fn location_reader(
        locations: &[BlockAddress],
        counts: &[usize],
    ) -> crate::Result<(VectorIndexReader, TrackedReads, usize)> {
        use crate::vector::storage::blocks::{block_len, pad, write_metadata, BlockOffsetsWriter};
        let options = VectorOptions::new(2, Metric::L2);
        let metadata = VectorColMetadata::build_ivf(&options, None)?;
        let mut data = Vec::new();
        write_metadata(&mut data, &metadata)?;
        let mut offsets = BlockOffsetsWriter::new(data.len() as u64);
        for &count in counts {
            pad(&mut data, block_len(&metadata.slots(), count))?;
            offsets.push(data.len() as u64, count);
        }
        offsets.finish(&mut data)?;
        let offsets_start = data.len() - 16 - 16 * (counts.len() + 1);
        let reads = TrackedReads::default();
        let blocks = Blocks::open(
            FileSlice::new(Arc::new(BlockTrackedBytes {
                bytes: data,
                reads: Arc::clone(&reads),
                block_len: 8192,
            })),
            &options,
            true,
        )?;
        let mut ids = Vec::new();
        IdMap::serialize_locations(locations, &mut ids)?;
        let max_doc = locations.len() as u32;
        let reader = VectorIndexReader {
            max_doc,
            options,
            num_vectors: counts.iter().sum(),
            present: true,
            id_map: DeferredIdMap::ready(IdMap::open(FileSlice::from(ids), max_doc)?),
            rows_slice: Arc::new(blocks),
            index: None,
            quantization: OnceLock::new(),
        };
        reads.lock().unwrap().clear();
        Ok((reader, reads, offsets_start))
    }

    #[test]
    fn locate_docs_validates_distinct_blocks_in_storage_order() -> crate::Result<()> {
        let expected = [
            (1, BlockAddress::new(0, 0)),
            (4, BlockAddress::new(0, 1)),
            (7, BlockAddress::new(0, 2)),
            (3, BlockAddress::new(1, 0)),
            (6, BlockAddress::new(1, 1)),
            (0, BlockAddress::new(2, 0)),
            (5, BlockAddress::new(2, 1)),
            (8, BlockAddress::new(2, 2)),
        ];
        let mut locations = IdMap::empty_locations(10);
        for &(doc, address) in &expected {
            locations[doc as usize] = address;
        }
        let (reader, reads, start) = location_reader(&locations, &[3, 2, 3])?;
        for docs in [
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99],
            vec![0, 2, 5, 8],
            vec![2, 9, 99],
            Vec::new(),
        ] {
            reads.lock().unwrap().clear();
            let prefix = (100, BlockAddress::new(100, 100));
            let mut located = vec![prefix];
            reader.locate_docs(&docs, &mut located)?;
            assert!(!reader.quantization_initialized());
            let expected: Vec<_> = expected
                .iter()
                .copied()
                .filter(|(doc, _)| docs.binary_search(doc).is_ok())
                .collect();
            assert_eq!(located[0], prefix);
            assert_eq!(&located[1..], expected);
            let blocks: std::collections::BTreeSet<_> =
                expected.iter().map(|(_, address)| address.block).collect();
            let expected_reads: Vec<_> = blocks
                .into_iter()
                .map(|block| {
                    let at = start + block as usize * 16;
                    at..at + 24
                })
                .collect();
            assert_eq!(*reads.lock().unwrap(), expected_reads, "docs={docs:?}");
        }
        Ok(())
    }

    #[test]
    fn location_bounds_are_checked_by_single_and_grouped_lookups() -> crate::Result<()> {
        for (count, invalid) in [
            (3, BlockAddress::new(0, 3)),
            (3, BlockAddress::new(1, 0)),
            (3, BlockAddress::new(0, u32::MAX)),
            (3, BlockAddress::new(u32::MAX, 1)),
            (0, BlockAddress::new(0, 0)),
        ] {
            let mut locations = IdMap::empty_locations(3);
            locations[0] = BlockAddress::new(0, 0);
            locations[1] = invalid;
            let (reader, _, _) = location_reader(&locations, &[count])?;
            assert!(matches!(
                reader.row_location(1),
                Err(TantivyError::DataCorruption(_))
            ));
            let prefix = (100, BlockAddress::new(100, 100));
            let mut located = vec![prefix];
            assert!(matches!(
                reader.locate_docs(&[0, 1], &mut located),
                Err(TantivyError::DataCorruption(_))
            ));
            assert_eq!(located, [prefix]);
        }
        Ok(())
    }

    #[test]
    fn doc_ids_reject_out_of_bounds_and_nonascending_values() -> crate::Result<()> {
        use crate::vector::storage::blocks::{
            block_len, column_range, pad, write_metadata, BlockOffsetsWriter,
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
            let mut directory = BlockOffsetsWriter::new(first as u64);
            directory.push(data.len() as u64, docs.len());
            directory.finish(&mut data)?;
            let blocks = Arc::new(Blocks::open(FileSlice::from(data), &options, true)?);
            let reader = VectorIndexReader {
                max_doc: 3,
                options,
                num_vectors: 3,
                present: true,
                rows_slice: blocks,
                id_map: DeferredIdMap::ready(IdMap::Identity { num_docs: 3 }),
                index: None,
                quantization: OnceLock::new(),
            };
            let mut decoded = Vec::new();
            let error = reader.read_doc_ids(0, &mut decoded).unwrap_err();
            assert!(
                matches!(error, TantivyError::DataCorruption(_)),
                "column=DocIds values={docs:?}: {error}"
            );
            assert!(matches!(
                reader.doc_id_at(BlockAddress::new(0, 0)),
                Err(TantivyError::DataCorruption(_))
            ));
            assert!(matches!(
                reader.doc_id_at(BlockAddress::new(0, 3)),
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
        let mut directory = crate::vector::storage::blocks::BlockOffsetsWriter::new(prefix as u64);
        directory.push(end as u64, 1);
        directory.finish(&mut data)?;
        let footer = data.len() - 32;
        let directory_start = data.len() - 16 - 2 * 16;
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
        assert_eq!(
            field.stats()?,
            Some(VectorIndexStats {
                num_vectors: 1,
                num_centroids: None
            })
        );
        assert!(field.search.get().is_none());
        assert!(id_reads.lock().unwrap().is_empty());
        data_reads.lock().unwrap().clear();
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
            reads
                .iter()
                .filter(|r| **r == (footer..footer + 32))
                .count(),
            1
        );
        assert_eq!(
            reads
                .iter()
                .filter(|r| r.start < directory_start && r.end > prefix)
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
        assert_eq!(readers[0].row_location(1)?, Some(BlockAddress::new(0, 0)));
        Ok(())
    }

    #[test]
    fn ivf_stats_read_counts_without_search_payloads() -> crate::Result<()> {
        use crate::vector::storage::blocks::{block_len, pad, write_metadata, BlockOffsetsWriter};

        for clusters in [1usize, 4096] {
            let options = VectorOptions::new(2, Metric::L2);
            let meta = VectorColMetadata::build_ivf(&options, None)?;
            let mut data = Vec::new();
            write_metadata(&mut data, &meta)?;
            let mut offsets = BlockOffsetsWriter::new(data.len() as u64);
            for _ in 0..clusters {
                pad(&mut data, block_len(&meta.slots(), 2))?;
                offsets.push(data.len() as u64, 2);
            }
            offsets.finish(&mut data)?;
            let footer = data.len() - 32;
            let data_reads = TrackedReads::default();
            let centroid_reads = TrackedReads::default();
            let router_reads = TrackedReads::default();
            let id_reads = TrackedReads::default();
            let mut centroids = Vec::new();
            ClusterIndex::serialize_centroids(
                clusters,
                clusters,
                &vec![0; clusters * 8],
                &options,
                &mut centroids,
            )?;
            let field = VectorFieldReader {
                options: options.clone(),
                source: Some(VectorSource {
                    metadata: BlockMetadata::open(
                        FileSlice::new(Arc::new(BlockTrackedBytes {
                            bytes: data,
                            reads: Arc::clone(&data_reads),
                            block_len: 16,
                        })),
                        &options,
                        true,
                    )?,
                    composite: tracked_id_composite(vec![0], &id_reads),
                    field: Field::from_field_id(0),
                    max_doc: clusters as u32,
                    centroid_slots: Some(CentroidSlices {
                        centroids: FileSlice::new(Arc::new(BlockTrackedBytes {
                            bytes: centroids,
                            reads: Arc::clone(&centroid_reads),
                            block_len: 16,
                        })),
                        router: FileSlice::new(Arc::new(BlockTrackedBytes {
                            bytes: vec![0],
                            reads: Arc::clone(&router_reads),
                            block_len: 16,
                        })),
                    }),
                }),
                search: OnceLock::new(),
            };
            data_reads.lock().unwrap().clear();
            assert_eq!(
                field.stats()?,
                Some(VectorIndexStats {
                    num_vectors: clusters,
                    num_centroids: Some(clusters),
                })
            );
            assert_eq!(*data_reads.lock().unwrap(), [footer..footer + 32]);
            assert_eq!(*centroid_reads.lock().unwrap(), [0..8]);
            assert!(router_reads.lock().unwrap().is_empty());
            assert!(id_reads.lock().unwrap().is_empty());
            assert!(field.search.get().is_none());
        }
        let missing = VectorFieldReader {
            options: VectorOptions::new(2, Metric::L2),
            source: None,
            search: OnceLock::new(),
        };
        assert_eq!(missing.stats()?, None);
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
        crate::vector::storage::blocks::BlockOffsetsWriter::new(end as u64).finish(&mut data)?;
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
        let mut directory = crate::vector::storage::blocks::BlockOffsetsWriter::new(prefix as u64);
        directory.push(data.len() as u64, ROWS);
        directory.finish(&mut data)?;
        let storage = Arc::new(BlockTrackedBytes {
            bytes: data,
            reads: Arc::clone(&reads),
            block_len: 32,
        });
        let blocks = Arc::new(Blocks::open(FileSlice::new(storage), &opts, false)?);
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
            quantization: OnceLock::new(),
        };
        let selected = [1, 2, 10, 11];
        let mut scratch = ReadScratch::default();
        scratch.rows.extend(selected);
        let batch = reader.read_vector_rows_planned(0, &mut scratch)?;

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
                offsets_start..offsets_start + 24,
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
