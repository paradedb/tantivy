//! Per-(segment, field) vector reader, modeled on
//! [`InvertedIndexReader`](crate::index::InvertedIndexReader).
//!
//! One [`VectorIndexReader`] serves one vector field of one segment, opened
//! (and cached) via
//! [`SegmentReader::vector_index`](crate::SegmentReader::vector_index). Small
//! routing state is parsed once and pinned in memory, while the bulk payload
//! stays behind [`FileSlice`]s and is fetched with ranged reads at query time.
//!
//! The reader is "store + optional index":
//! - The **store** is the segment's `.vec` composite: slot `[0]` is the row→doc_id [`IdMap`], slot
//!   `[1]` the dense vector rows (deferred).
//! - The **index** ([`IvfIndex`]) exists if the segment was merged with IVF clustering, and is
//!   loaded from the sibling `.centroids` composite. It tells a query which clusters — contiguous
//!   row ranges of slot `[1]` — to probe. Without it, search falls back to an exact scan.
//!
//! The pairing is an invariant of the write path (`VectorPlugin`): the IVF
//! merge writes cluster-sorted rows + an `Explicit` id-map + `.centroids`;
//! every other writer produces doc-ordered rows + `Identity`/`Bitmap` and no
//! sidecar. [`VectorIndexReader::open`] validates the two signals agree.

use std::cmp::Ordering;

use common::{HasLen, OwnedBytes};

use super::flat::IdMap;
use super::header::{
    read_centroid_header, read_vector_header, CentroidSlot, VectorFileVersion, VectorSlot,
};
use super::ivf::{IvfIndex, CENTROIDS_EXT};
use super::quantization::{VectorQuantizationConfig, QUANTIZED_RESIDUAL_NORM_STRIDE};
use super::VEC_EXT;
use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::error::DataCorruption;
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType, VectorOptions};
use crate::{DocId, SegmentReader, TantivyError};

/// Which on-disk layout a segment's vector data uses, surfaced through
/// [`VectorInfo`] for tooling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorStorageFormat {
    Flat,
    Ivf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorInfo {
    pub format: VectorStorageFormat,
    /// Distinct documents with a vector in this field. The per-cluster
    /// numbers (`cluster_stats`, [`VectorIndexReader::cluster_sizes`]) count
    /// posting rows, so with legacy V2 replication their sum could exceed
    /// `num_vectors`.
    pub num_vectors: usize,
    pub num_centroids: Option<usize>,
    pub cluster_stats: Option<VectorClusterStats>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorClusterStats {
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub avg_cluster_size: f64,
    pub empty_clusters: usize,
}

fn logical_slice(
    slice: FileSlice,
    logical_len: usize,
    description: &str,
) -> crate::Result<FileSlice> {
    let physical_len = slice.len();
    if physical_len < logical_len || physical_len - logical_len >= 64 {
        return Err(DataCorruption::comment_only(format!(
            "{description} physical length {physical_len} does not contain logical length \
             {logical_len} plus at most 63 alignment bytes"
        ))
        .into());
    }
    if physical_len != logical_len {
        let trailer = slice.slice(logical_len..physical_len).read_bytes()?;
        if trailer.iter().any(|&byte| byte != 0) {
            return Err(DataCorruption::comment_only(format!(
                "{description} has a non-zero alignment trailer"
            ))
            .into());
        }
    }
    Ok(slice.slice_to(logical_len))
}

fn validate_vector_slot_map(composite: &CompositeFile) -> crate::Result<()> {
    if let Some((field, slot)) = composite
        .field_indices()
        .find(|&(_, slot)| slot >= VectorSlot::COUNT)
    {
        return Err(DataCorruption::comment_only(format!(
            "vector composite field {field:?} declares slot {slot}; format {} permits only slots \
             0..{}",
            super::header::VECTOR_FILE_FORMAT_VERSION,
            VectorSlot::COUNT - 1
        ))
        .into());
    }
    Ok(())
}

fn validate_quantization_file_format(
    config: Option<&VectorQuantizationConfig>,
    vector_file_format: VectorFileVersion,
    field_name: &str,
) -> crate::Result<()> {
    if let Some(config) = config {
        if config.format_version != vector_file_format as u32 {
            return Err(DataCorruption::comment_only(format!(
                "vector field {field_name:?} settings format version {} does not match `.vec` \
                 format version {}; rebuild required",
                config.format_version, vector_file_format as u32
            ))
            .into());
        }
    }
    Ok(())
}

/// Per-(segment, field) vector reader: the row store plus, for IVF segments,
/// the routing index. See the module docs for the layout and the
/// pinned-vs-deferred split.
pub struct VectorIndexReader {
    options: VectorOptions,
    /// Distinct docs with a vector (the IdMap row count for flat storage; the
    /// persisted doc count for IVF, whose row total legacy V2 replication
    /// could inflate).
    num_vectors: usize,
    /// `false` for the placeholder built by [`Self::empty`] — the segment has
    /// no vector data for this field at all.
    present: bool,
    /// `.vec` slot `[0]`
    id_map: IdMap,
    /// `.vec` slot `[1]`: the dense vector rows. Never materialized whole;
    /// queries fetch per-cluster (or per-doc) ranges.
    rows_slice: FileSlice,
    index: Option<IvfIndex>,
}

impl VectorIndexReader {
    /// Opens `field`'s vector data in `segment_reader`'s segment. Returns the
    /// [`empty`](Self::empty) placeholder when the segment carries no vector
    /// data for the field (no `.vec` file, or the field has no slots in it),
    /// mirroring `SegmentReader::inverted_index`.
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
            Err(OpenReadError::FileDoesNotExist(_)) => return Ok(Self::empty(options)),
            Err(err) => return Err(err.into()),
        };
        let (vector_file_format, body) = read_vector_header(&vec_file)?;
        let vec_composite = CompositeFile::open(&body)?;
        validate_vector_slot_map(&vec_composite)?;
        let (Some(id_map_slice), Some(rows_slice)) = (
            vec_composite.open_read_with_idx(field, VectorSlot::IdMap.index()),
            vec_composite.open_read_with_idx(field, VectorSlot::Rows.index()),
        ) else {
            return Ok(Self::empty(options));
        };
        let id_map = IdMap::open(id_map_slice, segment_reader.max_doc())?;
        let quantization_config = segment_reader
            .index_settings()
            .vector_quantization
            .iter()
            .find(|config| config.field == entry.name())
            .cloned();
        validate_quantization_file_format(
            quantization_config.as_ref(),
            vector_file_format,
            entry.name(),
        )?;

        let centroid_slots = match segment_reader
            .open_read(SegmentComponent::Custom(CENTROIDS_EXT.to_string()))
        {
            Ok(file) => {
                let (centroids_version, body) = read_centroid_header(&file)?;
                let composite = CompositeFile::open(&body)?;
                match (
                    composite.open_read_with_idx(field, CentroidSlot::Centroids.index()),
                    composite.open_read_with_idx(field, CentroidSlot::Offsets.index()),
                    composite.open_read_with_idx(field, CentroidSlot::Bounds.index()),
                ) {
                    // Slot [2] (the router) stays optional: the write
                    // side skips it for degenerate centroid counts. Slot
                    // [3] (bounds) is read whenever present; when absent,
                    // a V1 file simply predates it — `IvfIndex::open`
                    // synthesizes SATURATED bounds — while a V2+ file
                    // without it is corrupt, not old.
                    (Some(centroids), Some(offsets), bounds) => {
                        if bounds.is_none() && centroids_version >= VectorFileVersion::V2 {
                            return Err(TantivyError::InternalError(format!(
                                "vector field {:?} has a V2 `.centroids` file with no bounds slot",
                                entry.name()
                            )));
                        }
                        Some((
                            centroids_version,
                            centroids,
                            offsets,
                            composite.open_read_with_idx(field, CentroidSlot::Router.index()),
                            bounds,
                        ))
                    }
                    _ => None,
                }
            }
            Err(OpenReadError::FileDoesNotExist(_)) => None,
            Err(err) => return Err(err.into()),
        };

        // The id-map variant and the `.centroids` sidecar are two signals of
        // one write-path decision; a mismatch means a corrupt segment, never a
        // fallback.
        let index = match (&id_map, centroid_slots) {
            (IdMap::Explicit(_), Some((version, centroids, offsets, router, bounds))) => Some(
                IvfIndex::open(version, &options, centroids, offsets, router, bounds)?,
            ),
            (IdMap::Explicit(_), None) => {
                return Err(TantivyError::InternalError(format!(
                    "vector field {:?} has cluster-sorted rows but no `.centroids` data",
                    entry.name()
                )));
            }
            (_, Some(_)) => {
                return Err(TantivyError::InternalError(format!(
                    "vector field {:?} has `.centroids` data but doc-ordered rows",
                    entry.name()
                )));
            }
            (_, None) => None,
        };

        let num_rows = id_map.num_rows() as usize;
        if let Some(index) = &index {
            if index.num_rows() != num_rows {
                return Err(TantivyError::InternalError(
                    "IVF id-map length does not match the cluster offsets".to_string(),
                ));
            }
        }
        let rows_slice = logical_slice(
            rows_slice,
            num_rows * options.bytes_per_vector(),
            &format!("vector field {:?} rows", entry.name()),
        )?;

        // T2 deliberately opens only the unquantized scan inputs. T3 replaces
        // this local validation with the complete quantized-layer readers.
        let residual_norm_slot =
            vec_composite.open_read_with_idx(field, VectorSlot::ResidualNorms.index());
        let _residual_norms = match (&index, quantization_config, residual_norm_slot) {
            (Some(_), Some(_), Some(slice)) => Some(logical_slice(
                slice,
                num_rows * QUANTIZED_RESIDUAL_NORM_STRIDE,
                &format!("vector field {:?} residual squared norms", entry.name()),
            )?),
            (Some(_), Some(_), None) => {
                return Err(DataCorruption::comment_only(format!(
                    "quantized vector field {:?} is missing residual squared norm slot 2",
                    entry.name()
                ))
                .into());
            }
            (Some(_), None, Some(_)) => {
                return Err(DataCorruption::comment_only(format!(
                    "vector field {:?} carries quantized slots without index metadata",
                    entry.name()
                ))
                .into());
            }
            (None, _, Some(_)) => {
                return Err(DataCorruption::comment_only(format!(
                    "flat vector field {:?} unexpectedly carries quantized slots",
                    entry.name()
                ))
                .into());
            }
            _ => None,
        };

        let num_vectors = match &index {
            Some(index) => index.num_docs(),
            None => num_rows,
        };
        Ok(Self {
            options,
            num_vectors,
            present: true,
            rows_slice,
            id_map,
            index,
        })
    }

    /// The no-data placeholder: zero vectors, no index. Every accessor
    /// behaves as an empty column, so callers never branch on presence.
    pub(crate) fn empty(options: VectorOptions) -> Self {
        Self {
            options,
            num_vectors: 0,
            present: false,
            rows_slice: FileSlice::empty(),
            id_map: IdMap::Identity { num_docs: 0 },
            index: None,
        }
    }

    pub fn options(&self) -> &VectorOptions {
        &self.options
    }

    pub fn dim(&self) -> usize {
        self.options.dim()
    }

    /// Number of distinct docs with a vector value.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// The routing index, present iff the segment's rows are IVF-clustered.
    /// `None` means search must scan the rows exactly.
    pub fn index(&self) -> Option<&IvfIndex> {
        self.index.as_ref()
    }

    /// Storage info for tooling; `None` if the segment has no vector data for
    /// the field.
    pub fn info(&self) -> Option<VectorInfo> {
        if !self.present {
            return None;
        }
        let Some(index) = &self.index else {
            return Some(VectorInfo {
                format: VectorStorageFormat::Flat,
                num_vectors: self.num_vectors,
                num_centroids: None,
                cluster_stats: None,
            });
        };
        let mut empty_clusters = 0;
        let mut min_cluster_size = usize::MAX;
        let mut max_cluster_size = 0;
        let mut total_cluster_size = 0;
        for cluster_size in index.cluster_sizes() {
            empty_clusters += usize::from(cluster_size == 0);
            min_cluster_size = min_cluster_size.min(cluster_size);
            max_cluster_size = max_cluster_size.max(cluster_size);
            total_cluster_size += cluster_size;
        }
        let num_centroids = index.num_clusters();
        let avg_cluster_size = if num_centroids == 0 {
            0.0
        } else {
            total_cluster_size as f64 / num_centroids as f64
        };
        let min_cluster_size = if num_centroids == 0 {
            0
        } else {
            min_cluster_size
        };
        Some(VectorInfo {
            format: VectorStorageFormat::Ivf,
            num_vectors: self.num_vectors,
            num_centroids: Some(num_centroids),
            cluster_stats: Some(VectorClusterStats {
                min_cluster_size,
                max_cluster_size,
                avg_cluster_size,
                empty_clusters,
            }),
        })
    }

    /// Per-cluster posting-list sizes in cluster order — the distribution
    /// behind [`Self::info`]'s aggregate cluster stats. `None` when the
    /// field's storage is not IVF.
    pub fn cluster_sizes(&self) -> Option<Vec<u32>> {
        self.index
            .as_ref()
            .map(|index| index.cluster_sizes().map(|size| size as u32).collect())
    }

    /// `true` if `doc_id` has a stored vector.
    pub fn contains(&self, doc_id: DocId) -> bool {
        self.row_id(doc_id).is_some()
    }

    /// The raw little-endian bytes of `doc_id`'s vector, fetched with one
    /// stride-sized ranged read; `None` if the doc has no vector.
    pub fn vector_bytes(&self, doc_id: DocId) -> crate::Result<Option<OwnedBytes>> {
        let Some(row) = self.row_id(doc_id) else {
            return Ok(None);
        };
        self.vector_bytes_for_row(row).map(Some)
    }

    /// The raw bytes of the single vector row at `row` of the dense rows
    /// slot, fetched with one stride-sized ranged read
    /// (`row * stride..(row + 1) * stride`). The caller resolves `row`
    /// beforehand (e.g. from a cluster's row range), so no doc→row lookup
    /// happens here.
    pub fn vector_bytes_for_row(&self, row: usize) -> crate::Result<OwnedBytes> {
        if row >= self.id_map.num_rows() as usize {
            return Err(TantivyError::InvalidArgument(format!(
                "vector row {row} is out of bounds"
            )));
        }
        let stride = self.options.bytes_per_vector();
        let bytes = self
            .rows_slice
            .slice(row * stride..(row + 1) * stride)
            .read_bytes()?;
        Ok(bytes)
    }

    /// The doc id stored at `row` of the cluster-sorted permutation, decoded
    /// on demand from the pinned `Explicit` id-map. IVF storage only.
    #[inline]
    pub fn doc_id_at(&self, row: usize) -> DocId {
        let IdMap::Explicit(bytes) = &self.id_map else {
            unreachable!("doc_id_at is only meaningful for cluster-sorted (IVF) storage");
        };
        let start = row * std::mem::size_of::<DocId>();
        DocId::from_le_bytes(
            bytes[start..start + std::mem::size_of::<DocId>()]
                .try_into()
                .unwrap(),
        )
    }

    /// The doc ids assigned to `cluster`, ascending; `None` if the storage is
    /// not IVF or `cluster` is out of bounds.
    pub fn cluster_doc_ids(&self, cluster: usize) -> Option<Vec<DocId>> {
        let index = self.index.as_ref()?;
        if cluster >= index.num_clusters() {
            return None;
        }
        Some(
            index
                .cluster_range(cluster)
                .map(|row| self.doc_id_at(row))
                .collect(),
        )
    }

    /// Doc → dense row. For clustered storage, rows are cluster-sorted and
    /// ascending by doc id within each cluster, so this scans clusters and
    /// binary-searches each one over the pinned id-map bytes. For the flat
    /// id-maps (`Identity`/`Bitmap`) the mapping is strictly ascending in
    /// doc id — the property the exact path's run builder leans on.
    pub(crate) fn row_id(&self, doc_id: DocId) -> Option<usize> {
        match &self.id_map {
            IdMap::Identity { num_docs } => (doc_id < *num_docs).then_some(doc_id as usize),
            IdMap::Bitmap(_) => self.id_map.rank_if_exists(doc_id).map(|row| row as usize),
            IdMap::Explicit(_) => {
                let index = self.index.as_ref()?;
                for cluster in 0..index.num_clusters() {
                    let rows = index.cluster_range(cluster);
                    let mut lo = rows.start;
                    let mut hi = rows.end;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        match self.doc_id_at(mid).cmp(&doc_id) {
                            Ordering::Less => lo = mid + 1,
                            Ordering::Greater => hi = mid,
                            Ordering::Equal => return Some(mid),
                        }
                    }
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::directory::CompositeWrite;
    use crate::vector::header::{read_vector_header, write_vector_header};

    #[test]
    fn aligned_rows_and_residual_norms_are_trimmed_to_logical_lengths() -> crate::Result<()> {
        let field = Field::from_field_id(0);
        let mut bytes = Vec::new();
        write_vector_header(&mut bytes)?;
        {
            let mut writer = CompositeWrite::wrap(&mut bytes);
            let rows = writer.for_field_with_idx(field, VectorSlot::Rows.index());
            rows.write_all(&[1, 2, 3, 4])?;
            rows.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(1), 0)
                .write_all(&[9])?;

            let norms = writer.for_field_with_idx(field, VectorSlot::ResidualNorms.index());
            norms.write_all(&2.5_f32.to_le_bytes())?;
            norms.write_all(&[0; 63])?;
            writer
                .for_field_with_idx(Field::from_field_id(2), 0)
                .write_all(&[9])?;
            writer.close()?;
        }

        let (_, body) = read_vector_header(&FileSlice::from(bytes))?;
        let composite = CompositeFile::open(&body)?;
        let rows = logical_slice(
            composite
                .open_read_with_idx(field, VectorSlot::Rows.index())
                .unwrap(),
            4,
            "rows fixture",
        )?
        .read_bytes()?;
        let norms = logical_slice(
            composite
                .open_read_with_idx(field, VectorSlot::ResidualNorms.index())
                .unwrap(),
            4,
            "residual norms fixture",
        )?
        .read_bytes()?;

        assert_eq!(rows.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(
            f32::from_le_bytes(norms.as_slice().try_into().unwrap()),
            2.5
        );
        Ok(())
    }

    #[test]
    fn vector_format_rejects_slots_outside_the_fixed_map() -> crate::Result<()> {
        let field = Field::from_field_id(0);
        let mut bytes = Vec::new();
        write_vector_header(&mut bytes)?;
        {
            let mut writer = CompositeWrite::wrap(&mut bytes);
            writer
                .for_field_with_idx(field, VectorSlot::COUNT)
                .write_all(&[1])?;
            writer.close()?;
        }

        let (_, body) = read_vector_header(&FileSlice::from(bytes))?;
        let composite = CompositeFile::open(&body)?;
        let error = validate_vector_slot_map(&composite).unwrap_err();
        assert!(error.to_string().contains("permits only slots 0..11"));
        Ok(())
    }

    #[test]
    fn quantization_settings_format_must_match_vector_file() -> crate::Result<()> {
        let options = VectorOptions::new(64, crate::schema::Metric::Dot);
        let mut config = VectorQuantizationConfig::materialize(
            "embedding".to_string(),
            &options,
            vec![super::super::quantization::VectorQuantizationLayer { bits: 1, seed: 7 }],
        )?;
        config.format_version = VectorFileVersion::V2 as u32;

        let error =
            validate_quantization_file_format(Some(&config), VectorFileVersion::V3, "embedding")
                .unwrap_err();
        assert!(error.to_string().contains("rebuild required"));
        Ok(())
    }

    #[test]
    fn nonzero_alignment_trailer_is_corruption() {
        let slice = FileSlice::from(vec![1, 2, 3, 4, 0, 7]);
        assert!(logical_slice(slice, 4, "bad fixture").is_err());
    }
}
