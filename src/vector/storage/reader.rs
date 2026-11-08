//! Per-segment vector metadata, block columns and optional IVF routing.
//!
//! Metadata opens independently of search state. The first search-reader request
//! pins routing state. Block addresses are read and validated only when accessed.
//! Clustered IdMap entries open only for document lookups. A quantized clustered field keeps
//! its full-precision rows in document order in its `ExactRows` entry; one set of flat-row
//! primitives, [`FlatRows`], serves that entry and flat fields alike. See FORMAT.md.

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
use crate::vector::storage::metadata::{Partition, SlotType, VectorColMetadata};
use crate::vector::storage::quantized::{
    exact_read_cost, push_consecutive_runs, range_cost, rows_read_cost, QuantizedFieldReader,
    QuantizedIndexCtx, QuantizedLayerReader, ReadCost, ReadScratch,
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

/// A flat or plain clustered field declares exactly the IdMap and Data entries; a quantized
/// clustered field also declares exactly the ExactIdMap and ExactRows entries.
fn validate_vector_entries(
    composite: &CompositeFile,
    field: Field,
    exact_rows: bool,
) -> crate::Result<()> {
    let declared = |entry: VectorEntry| {
        composite
            .field_indices()
            .any(|(f, idx)| f == field && idx == entry.index())
    };
    let last = if exact_rows {
        VectorEntry::ExactRows.index()
    } else {
        VectorEntry::Data.index()
    };
    let exact_declared = declared(VectorEntry::ExactIdMap) && declared(VectorEntry::ExactRows);
    if composite
        .field_indices()
        .any(|(f, idx)| f == field && idx > last)
        || !declared(VectorEntry::IdMap)
        || !declared(VectorEntry::Data)
        || exact_declared != exact_rows
    {
        return Err(DataCorruption::comment_only(if exact_rows {
            "quantized clustered vector field requires exactly IdMap, Data, ExactIdMap and \
             ExactRows entries"
        } else {
            "vector field requires exactly IdMap and Data entries"
        })
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
    /// The deferred full-precision entry of a quantized clustered field.
    exact: Option<FileSlice>,
}

/// The entry a deferred addressing map is read from.
struct DeferredIdMapSource {
    composite: CompositeFile,
    field: Field,
    max_doc: DocId,
    entry: VectorEntry,
}

/// Opens an addressing entry only when a document lookup needs it; failures are shared.
struct DeferredIdMap {
    source: Option<DeferredIdMapSource>,
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
    /// Defers `entry` of `field` in `composite`.
    fn deferred(
        composite: CompositeFile,
        field: Field,
        max_doc: DocId,
        entry: VectorEntry,
    ) -> Self {
        Self {
            source: Some(DeferredIdMapSource {
                composite,
                field,
                max_doc,
                entry,
            }),
            value: OnceLock::new(),
        }
    }
    /// Initializes the entry once and checks its variant against the stored partition.
    fn get(&self, clustered: bool) -> crate::Result<&IdMap> {
        self.value
            .get_or_init(|| {
                let DeferredIdMapSource {
                    composite,
                    field,
                    max_doc,
                    entry,
                } = self.source.as_ref().expect("deferred entry source");
                let entry = composite.open_read_with_idx(*field, entry.index()).unwrap();
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
    /// Document addressing, deferred until a lookup requires the entry. Clustered storage maps a
    /// document to its primary row.
    id_map: DeferredIdMap,
    /// Deferred row-group columns and their validated geometry.
    rows_slice: Arc<Blocks>,
    pub(crate) index: Option<ClusterIndex>,
    /// The document-ordered full-precision rows of a quantized clustered field.
    exact: Option<ExactRows>,
    quantization: OnceLock<crate::Result<Option<QuantizedFieldReader>>>,
}

/// A quantized clustered field's full-precision rows in document order: flat blocks and the
/// presence map that ranks a document to its row.
struct ExactRows {
    source: FileSlice,
    blocks: OnceLock<crate::Result<Blocks>>,
    id_map: DeferredIdMap,
}

/// Document-ordered full-precision rows in flat blocks: a flat field's Data entry, or a quantized
/// clustered field's ExactRows entry. Every flat row access goes through here.
pub(crate) struct FlatRows<'a> {
    blocks: &'a Blocks,
    id_map: &'a DeferredIdMap,
}

impl FlatRows<'_> {
    fn rows_per_block(&self) -> u32 {
        let Partition::Uniform { rows_per_block } = self.blocks.metadata.field().partition else {
            unreachable!("flat rows have a uniform partition")
        };
        rows_per_block
    }

    fn map(&self) -> crate::Result<&IdMap> {
        self.id_map.get(false)
    }

    /// The block and local row of flat row `row`.
    pub(crate) fn address(&self, row: u32) -> BlockAddress {
        let rows_per_block = self.rows_per_block();
        BlockAddress::new(row / rows_per_block, row % rows_per_block)
    }

    /// Rows stored, every document with a vector once.
    pub(crate) fn num_rows(&self) -> usize {
        self.blocks.num_rows()
    }

    /// The location of `doc`'s row, by presence rank.
    pub(crate) fn location(&self, doc: DocId) -> crate::Result<Option<BlockAddress>> {
        Ok(self.map()?.rank_if_exists(doc).map(|row| self.address(row)))
    }

    /// The document of flat row `row`.
    pub(crate) fn doc_at(&self, row: u32) -> crate::Result<DocId> {
        Ok(self.map()?.doc_at(row))
    }

    /// The flat rows of ascending `docs`, which must all have one; ascending as well.
    fn rows_of(&self, docs: &[DocId], out: &mut Vec<usize>) -> crate::Result<()> {
        debug_assert!(docs.windows(2).all(|pair| pair[0] < pair[1]));
        let map = self.map()?;
        out.clear();
        for &doc in docs {
            let row = map.rank_if_exists(doc).ok_or_else(|| {
                DataCorruption::comment_only(format!("document {doc} has no full-precision row"))
            })?;
            out.push(row as usize);
        }
        Ok(())
    }

    /// The rows as one contiguous span, row `r` at `r * stride`.
    fn span(&self, stride: usize) -> &FileSlice {
        debug_assert_eq!(stride, self.blocks.schema[0].stride as usize);
        self.blocks
            .flat_rows()
            .expect("flat rows have a uniform partition")
    }

    /// Reads the rows of ascending `docs` as one range read per run of consecutive rows,
    /// addressed by row within the contiguous span; the batch yields them in document order,
    /// keyed by flat row.
    pub(crate) fn read_docs<'s>(
        &self,
        docs: &[DocId],
        stride: usize,
        scratch: &'s mut ReadScratch,
    ) -> crate::Result<VectorRowBatch<'s>> {
        let ReadScratch {
            rows, runs, chunks, ..
        } = scratch;
        runs.clear();
        chunks.clear();
        self.rows_of(docs, rows)?;
        let span = self.span(stride);
        push_consecutive_runs(rows, runs);
        for range in runs.iter().cloned() {
            let bytes = span.read_vector_bytes_slice(
                range.start * stride..range.end * stride,
                VectorReadKind::FullPrecision,
            )?;
            chunks.push(VectorRowChunk { rows: range, bytes });
        }
        Ok(VectorRowBatch {
            rows,
            chunks,
            stride,
        })
    }

    /// The cost of reading the rows of ascending `docs`: the distinct storage blocks they touch,
    /// or their bytes without block geometry, from the rows' arithmetic offsets within the span.
    /// Reads nothing. Uses `scratch.offsets`.
    pub(crate) fn docs_cost(
        &self,
        docs: &[DocId],
        stride: usize,
        scratch: &mut ReadScratch,
    ) -> crate::Result<ReadCost> {
        let rows = &mut scratch.offsets;
        self.rows_of(docs, rows)?;
        Ok(rows_read_cost(
            self.span(stride),
            stride,
            rows.iter().copied(),
            rows.len(),
        ))
    }
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
        let data = vec_composite
            .open_read_with_idx(field, VectorEntry::Data.index())
            .ok_or_else(|| DataCorruption::comment_only("vector field has no Data entry"))?;
        let metadata = BlockMetadata::open(data, &options, centroid_slots.is_some())?;
        let exact_rows = metadata.meta.has_exact_rows();
        validate_vector_entries(&vec_composite, field, exact_rows)?;
        let exact = if exact_rows {
            let entry = vec_composite
                .open_read_with_idx(field, VectorEntry::ExactRows.index())
                .expect("validated ExactRows entry");
            Some(entry)
        } else {
            None
        };
        Ok(Self {
            options,
            source: Some(VectorSource {
                metadata,
                composite: vec_composite,
                field,
                max_doc: segment_reader.max_doc(),
                centroid_slots,
                exact,
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
        let id_map = DeferredIdMap::deferred(
            source.composite.clone(),
            source.field,
            source.max_doc,
            VectorEntry::IdMap,
        );
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
        // The document-ordered rows hold every document once: as many as the routing index counts
        // and as the presence map ranks.
        let exact = source.exact.as_ref().map(|entry| ExactRows {
            source: entry.clone(),
            blocks: OnceLock::new(),
            id_map: DeferredIdMap::deferred(
                source.composite.clone(),
                source.field,
                source.max_doc,
                VectorEntry::ExactIdMap,
            ),
        });
        Ok(Self {
            max_doc: source.max_doc,
            options,
            num_vectors,
            present: true,
            rows_slice,
            id_map,
            index,
            exact,
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
            exact: None,
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

    /// Whether the full-precision rows live in their own document-ordered entry: quantized
    /// clustered storage.
    pub(crate) fn has_exact_rows(&self) -> bool {
        self.exact.is_some()
    }

    /// Whether a document may have a row in two clusters.
    pub(crate) fn spilled(&self) -> bool {
        self.rows_slice.metadata.spilled()
    }

    /// One past the segment's largest document id.
    pub(crate) fn max_doc(&self) -> DocId {
        self.max_doc
    }

    /// The document-ordered full-precision rows: a flat field's blocks, or a quantized clustered
    /// field's `ExactRows`. `None` for plain clustered storage, whose rows sit in the cluster
    /// blocks.
    pub(crate) fn flat_rows(&self) -> crate::Result<Option<FlatRows<'_>>> {
        match &self.exact {
            Some(exact) => Ok(Some(FlatRows {
                blocks: self.full_precision_blocks()?,
                id_map: &exact.id_map,
            })),
            None if !self.rows_slice.clustered() => Ok(Some(FlatRows {
                blocks: &self.rows_slice,
                id_map: &self.id_map,
            })),
            None => Ok(None),
        }
    }

    fn full_precision_blocks(&self) -> crate::Result<&Blocks> {
        match &self.exact {
            Some(exact) => exact
                .blocks
                .get_or_init(|| {
                    let _init = crate::vector::telemetry::enter_vector_stage(
                        crate::vector::VectorStage::ScanInit,
                    );
                    let metadata = BlockMetadata::open(exact.source.clone(), &self.options, false)?;
                    let blocks = Blocks::from_metadata(metadata)?;
                    if blocks.num_rows() != self.num_vectors
                        || blocks.num_rows() != exact.id_map.get(false)?.num_rows() as usize
                    {
                        return Err(DataCorruption::comment_only(
                            "exact vector row count mismatch",
                        )
                        .into());
                    }
                    Ok(blocks)
                })
                .as_ref()
                .map_err(Clone::clone),
            None => Ok(&self.rows_slice),
        }
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

    /// Each cluster's row count, in cluster order; `None` when the storage is not IVF. Rows, not
    /// documents: a document assigned to two clusters counts in both.
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

    /// Returns the number of live IVF posting rows: the work an exhaustive scan does, counting a
    /// document once per cluster it is assigned to, not the number of live documents.
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

    /// The location of `doc`'s full-precision row in the document-ordered blocks: a flat field's
    /// row, or a quantized clustered field's row in `ExactRows`. `None` for plain clustered
    /// storage, whose rows sit at the document's cluster location, and for documents without a
    /// vector.
    pub(crate) fn exact_location(&self, doc_id: DocId) -> crate::Result<Option<BlockAddress>> {
        if doc_id >= self.max_doc {
            return Ok(None);
        }
        match self.flat_rows()? {
            Some(flat) => flat.location(doc_id),
            None => Ok(None),
        }
    }

    /// Reads a document's vector: from the document-ordered rows where the field has them, else
    /// through its stored cluster location.
    pub fn vector_bytes(&self, doc_id: DocId) -> crate::Result<Option<OwnedBytes>> {
        let location = if self.has_exact_rows() || !self.rows_slice.clustered() {
            self.exact_location(doc_id)?
        } else {
            self.row_location(doc_id)?
        };
        location
            .map(|address| self.vector_bytes_for_row(address))
            .transpose()
    }

    /// Reads one full-precision vector at a block-local row of the blocks that hold the rows:
    /// `ExactRows` for quantized clustered storage, the Data blocks otherwise.
    pub fn vector_bytes_for_row(&self, address: BlockAddress) -> crate::Result<OwnedBytes> {
        let blocks = self.full_precision_blocks()?;
        let (block, row) = (address.block as usize, address.row as usize);
        if block >= blocks.num_blocks() {
            return Err(TantivyError::InvalidArgument(
                "vector block is out of bounds".into(),
            ));
        }
        let view = blocks.block(block)?;
        if row >= view.num_rows() {
            return Err(TantivyError::InvalidArgument(
                "vector row is out of bounds".into(),
            ));
        }
        view.read_column(0, row..row + 1)
    }

    /// The full-precision row at `(cluster, row)` of a clustered field, whose document is `doc`:
    /// from the cluster block for plain storage, from `ExactRows` for quantized storage.
    pub(crate) fn cluster_row_bytes(
        &self,
        cluster: usize,
        row: usize,
        doc: DocId,
    ) -> crate::Result<OwnedBytes> {
        if self.exact.is_none() {
            return self.vector_bytes_for_row(BlockAddress::new(cluster as u32, row as u32));
        }
        self.vector_bytes(doc)?.ok_or_else(|| {
            DataCorruption::comment_only(format!(
                "cluster {cluster} row {row} names document {doc}, which has no full-precision row"
            ))
            .into()
        })
    }

    /// Reads the rows of ascending `docs` from the document-ordered rows through storage-aware
    /// range reads; the batch yields them in document order.
    pub(crate) fn read_exact_docs<'s>(
        &self,
        docs: &[DocId],
        scratch: &'s mut ReadScratch,
    ) -> crate::Result<VectorRowBatch<'s>> {
        let flat = self.flat_rows()?.ok_or_else(|| {
            DataCorruption::comment_only("document-ordered rows need a flat entry")
        })?;
        flat.read_docs(docs, self.options.bytes_per_vector(), scratch)
    }

    /// An upper bound on the cost of reading `rows` document-ordered rows whose documents are not
    /// in hand: each row spans at most one storage block more than its stride fills, or its bytes
    /// without block geometry.
    pub(crate) fn exact_rows_cost_bound(&self, rows: usize) -> ReadCost {
        let stride = self.options.bytes_per_vector();
        let block_len = match &self.exact {
            Some(exact) => exact.source.storage_block_len(),
            None => self.rows_slice.storage_block_len(),
        };
        match block_len {
            Some(block_len) => ReadCost(rows * (stride.div_ceil(block_len) + 1)),
            None => ReadCost(rows * stride),
        }
    }

    /// The cost of [`Self::read_exact_docs`] for ascending `docs`: the distinct storage blocks of
    /// `ExactRows` they touch, or their bytes without block geometry.
    pub(crate) fn exact_docs_cost(
        &self,
        docs: &[DocId],
        scratch: &mut ReadScratch,
    ) -> crate::Result<ReadCost> {
        let flat = self.flat_rows()?.ok_or_else(|| {
            DataCorruption::comment_only("document-ordered rows need a flat entry")
        })?;
        flat.docs_cost(docs, self.options.bytes_per_vector(), scratch)
    }

    /// Fetches increasing local rows from one known block of the blocks that hold the
    /// full-precision rows, using storage-aware range reads.
    pub(crate) fn read_vector_rows_planned<'s>(
        &self,
        block: usize,
        scratch: &'s mut ReadScratch,
    ) -> crate::Result<VectorRowBatch<'s>> {
        read_rows_planned_in(
            self.full_precision_blocks()?,
            block,
            self.options.bytes_per_vector(),
            scratch,
        )
    }

    /// Fetches increasing local rows of a plain cluster block; quantized clustered storage keeps
    /// its rows in `ExactRows` and never calls this.
    pub(crate) fn read_vector_rows_in_cluster<'s>(
        &self,
        cluster: usize,
        scratch: &'s mut ReadScratch,
    ) -> crate::Result<VectorRowBatch<'s>> {
        debug_assert!(self.exact.is_none(), "cluster blocks hold no rows");
        read_rows_planned_in(
            &self.rows_slice,
            cluster,
            self.options.bytes_per_vector(),
            scratch,
        )
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
    /// Resolves a document to a block and local row of the Data blocks without global row
    /// numbering: its primary cluster row for clustered storage, its flat row otherwise.
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
            FlatRows {
                blocks: &self.rows_slice,
                id_map: &self.id_map,
            }
            .location(doc_id)
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
            let flat = FlatRows {
                blocks,
                id_map: &self.id_map,
            };
            let first = cluster * flat.rows_per_block() as usize;
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

    /// The cost of reading a plain cluster's full-precision rows at `offsets` exactly; `None`
    /// reads every row as one span. Quantized clustered storage prices [`Self::exact_docs_cost`]
    /// instead.
    pub(crate) fn exact_cost(
        &self,
        cluster: usize,
        offsets: Option<&[u32]>,
    ) -> crate::Result<ReadCost> {
        debug_assert!(self.exact.is_none(), "cluster blocks hold no rows");
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

    /// Reads a plain cluster's full-precision rows without document ids; quantized clustered
    /// storage keeps its rows in `ExactRows` and never calls this.
    pub(crate) fn read_cluster_rows(&self, cluster: usize) -> crate::Result<OwnedBytes> {
        debug_assert!(self.exact.is_none(), "cluster blocks hold no rows");
        let block = self.rows_slice.block(cluster)?;
        block.read_column(0, block.row_range())
    }
}

/// Fetches increasing local `scratch.rows` of `block` in `blocks` through storage-aware range
/// reads of its Rows column, the first column of every block that holds rows.
fn read_rows_planned_in<'s>(
    blocks: &Blocks,
    block: usize,
    stride: usize,
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
    ranges.clear();
    chunks.clear();
    block_scratch.clear();
    if let Some(&last) = rows.last() {
        let view = blocks.block(block)?;
        view.check_rows(&(rows[0]..last + 1))?;
        let column = view.column(0);
        QuantizedLayerReader::plan_slot_reads(
            &column,
            stride,
            0,
            view.row_range(),
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
        let map = DeferredIdMap::deferred(
            tracked_id_composite(bytes, &reads),
            Field::from_field_id(0),
            3,
            VectorEntry::IdMap,
        );
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
            exact: None,
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
                exact: None,
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
    fn routing_defers_exact_entries_and_shares_their_validation() -> crate::Result<()> {
        use crate::vector::test_support::FixtureSpec;
        let fixture = FixtureSpec::clustered(Metric::L2, 64, 4, 32)
            .schedule(&[1, 1])
            .spill()
            .build()?;
        let segment = fixture.index.searchable_segments()?.remove(0);
        let segment = SegmentReader::open(&segment)?;
        let mut field = VectorFieldReader::open(&segment, fixture.field)?;
        let source = field.source.as_mut().unwrap();
        let bytes = source.exact.as_ref().unwrap().read_bytes()?.to_vec();
        let reads = Arc::new(Mutex::new(Vec::new()));
        source.exact = Some(FileSlice::new(Arc::new(BlockTrackedBytes {
            bytes,
            reads: Arc::clone(&reads),
            block_len: 16,
        })));
        let reader = Arc::new(VectorIndexReader::open(&field)?);
        assert_eq!(reader.index().unwrap().num_docs(), 32);
        reader.exact_rows_cost_bound(10);
        let exact = reader.exact.as_ref().unwrap();
        assert!(exact.blocks.get().is_none());
        assert!(exact.id_map.value.get().is_none());
        assert!(reads.lock().unwrap().is_empty());
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| reader.full_precision_blocks().unwrap()))
                .collect();
            let blocks: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            assert!(blocks.iter().all(|b| std::ptr::eq(*b, blocks[0])));
        });
        let metadata_reads = reads.lock().unwrap().len();
        assert!(metadata_reads > 0);
        assert!(exact.id_map.value.get().is_some());
        assert_eq!(reader.flat_rows()?.unwrap().num_rows(), 32);
        assert_eq!(reads.lock().unwrap().len(), metadata_reads);
        assert!(reader.vector_bytes(0)?.is_some());
        assert!(exact.id_map.value.get().is_some());

        let mut mismatch = VectorIndexReader::open(&field)?;
        mismatch.num_vectors += 1;
        for _ in 0..2 {
            assert!(matches!(
                mismatch.full_precision_blocks(),
                Err(TantivyError::DataCorruption(_))
            ));
        }
        let exact = mismatch.exact.as_ref().unwrap();
        assert!(matches!(
            exact.blocks.get(),
            Some(Err(TantivyError::DataCorruption(_)))
        ));

        let mut mismatch = VectorIndexReader::open(&field)?;
        let mut ids = Vec::new();
        IdMap::serialize(&[0], 32, &mut ids)?;
        mismatch.exact.as_mut().unwrap().id_map =
            DeferredIdMap::ready(IdMap::open(FileSlice::from(ids), 32)?);
        assert!(matches!(
            mismatch.vector_bytes_for_row(BlockAddress::new(0, 0)),
            Err(TantivyError::DataCorruption(_))
        ));
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
                exact: None,
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
                exact: None,
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
            exact: None,
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

    // Missing entries and unknown entry indices are corruption regardless of field order: a flat
    // or plain field declares exactly entries 0 and 1, a quantized clustered field exactly 0 to 3.
    #[test]
    fn vector_entries_require_exact_set() -> crate::Result<()> {
        let field = Field::from_field_id(0);
        for (entries, plain_ok, exact_ok) in [
            (vec![0], false, false),
            (vec![1], false, false),
            (vec![0, 1, 2], false, false),
            (vec![0, 1, 3], false, false),
            (vec![1, 0], true, false),
            (vec![1, 3, 0, 2], false, true),
            (vec![0, 1, 2, 3, 4], false, false),
        ] {
            let mut bytes = Vec::new();
            let mut writer = CompositeWrite::wrap(&mut bytes);
            for &idx in &entries {
                writer.for_field_with_idx(field, idx).write_all(&[0])?;
            }
            writer.close()?;
            let composite = CompositeFile::open(&FileSlice::from(bytes))?;
            assert_eq!(
                validate_vector_entries(&composite, field, false).is_ok(),
                plain_ok,
                "{entries:?}"
            );
            assert_eq!(
                validate_vector_entries(&composite, field, true).is_ok(),
                exact_ok,
                "{entries:?}"
            );
        }
        Ok(())
    }
}
