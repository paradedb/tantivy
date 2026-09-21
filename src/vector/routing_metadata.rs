use std::io::{self, Write};
use std::sync::{Arc, OnceLock};

use common::{BinarySerializable, HasLen, OwnedBytes, VInt};

use super::{VectorIndexMetadata, VMETA_EXT};
use crate::directory::error::OpenReadError;
use crate::directory::{CompositeFile, FileSlice};
use crate::index::SegmentComponent;
use crate::schema::{Field, FieldType, VectorOptions};
use crate::{SegmentReader, TantivyError};

const MAGIC: &[u8; 4] = b"VMT1";
const SUMMARY_LEN: usize = 24;
const PAGE_LEN: usize = 4096;

pub(crate) fn write_header<W: Write + ?Sized>(out: &mut W) -> io::Result<()> {
    out.write_all(MAGIC)
}

pub(crate) fn serialize_field<W: Write + ?Sized>(
    out: &mut W,
    num_docs: usize,
    offsets: Option<&[u64]>,
) -> io::Result<()> {
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid routing metadata counts",
        )
    };
    let docs = u32::try_from(num_docs).map_err(|_| invalid())?;
    let (clusters, rows, nonempty, kind) = if let Some(offsets) = offsets {
        if offsets.first() != Some(&0) {
            return Err(invalid());
        }
        let clusters = u32::try_from(offsets.len() - 1).map_err(|_| invalid())?;
        if clusters == 0 {
            return Err(invalid());
        }
        let mut nonempty = 0u32;
        for pair in offsets.windows(2) {
            let rows = pair[1].checked_sub(pair[0]).ok_or_else(invalid)?;
            u32::try_from(rows).map_err(|_| invalid())?;
            if rows > u64::from(docs) {
                return Err(invalid());
            }
            nonempty += u32::from(rows > 0);
        }
        let rows = *offsets.last().unwrap();
        if rows < u64::from(docs) {
            return Err(invalid());
        }
        (clusters, rows, nonempty, 1u32)
    } else {
        (0, u64::from(docs), 0, 0)
    };
    docs.serialize(out)?;
    clusters.serialize(out)?;
    rows.serialize(out)?;
    nonempty.serialize(out)?;
    kind.serialize(out)?;
    if let Some(offsets) = offsets {
        for pair in offsets.windows(2) {
            ((pair[1] - pair[0]) as u32).serialize(out)?;
        }
    }
    Ok(())
}

#[derive(Default)]
struct Summary {
    docs: usize,
    clusters: usize,
    rows: usize,
    nonempty: usize,
    clustered: bool,
}

pub(crate) struct VectorRoutingMetadata {
    options: VectorOptions,
    summary: Summary,
    counts: FileSlice,
    pages: Vec<OnceLock<OwnedBytes>>,
    legacy: Option<Arc<VectorIndexMetadata>>,
}

impl VectorRoutingMetadata {
    pub(crate) fn open(segment: &SegmentReader, field: Field) -> crate::Result<Self> {
        let entry = segment.schema().get_field_entry(field);
        let options = match entry.field_type() {
            FieldType::Vector(options) => options.clone(),
            _ => {
                return Err(TantivyError::InvalidArgument(format!(
                    "field {:?} is not a vector field",
                    entry.name()
                )));
            }
        };
        match segment.open_read(SegmentComponent::Custom(VMETA_EXT.to_string())) {
            Ok(file) => {
                let metadata = Self::open_file(options, file, field)?;
                if metadata.num_docs() > segment.max_doc() as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "routing document count exceeds segment size",
                    )
                    .into());
                }
                Ok(metadata)
            }
            Err(OpenReadError::FileDoesNotExist(_)) => {
                let legacy = segment.vector_index_metadata(field)?;
                let summary = match legacy.clusters() {
                    Some(clusters) => Summary {
                        docs: clusters.num_docs(),
                        clusters: clusters.num_clusters(),
                        rows: clusters.num_rows(),
                        nonempty: clusters.num_non_empty_clusters(),
                        clustered: true,
                    },
                    None => Summary {
                        docs: legacy.num_vectors(),
                        rows: legacy.num_vectors(),
                        ..Default::default()
                    },
                };
                Ok(Self {
                    options,
                    summary,
                    counts: FileSlice::empty(),
                    pages: Vec::new(),
                    legacy: Some(legacy),
                })
            }
            Err(error) => Err(error.into()),
        }
    }

    fn open_file(options: VectorOptions, file: FileSlice, field: Field) -> io::Result<Self> {
        if file.len() < MAGIC.len() + 5
            || file.slice_to(MAGIC.len()).read_bytes()?.as_slice() != MAGIC
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid routing metadata header",
            ));
        }
        let body = file.slice_from(MAGIC.len());
        let footer_len = body.slice_from(body.len() - 4).read_bytes()?;
        let footer_len = u32::deserialize(&mut footer_len.as_slice())? as usize;
        if footer_len > body.len() - 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid routing metadata footer",
            ));
        }
        let footer_start = body.len() - 4 - footer_len;
        let footer = body.slice(footer_start..body.len() - 4).read_bytes()?;
        let invalid_footer = || {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid routing metadata footer",
            )
        };
        if footer.len() != footer_len {
            return Err(invalid_footer());
        }
        let read_vint = |input: &mut &[u8]| -> io::Result<u64> {
            let last = input
                .iter()
                .take(10)
                .position(|byte| byte & 0x80 != 0)
                .ok_or_else(invalid_footer)?;
            if last == 9 && input[last] & 0x7f > 1 {
                return Err(invalid_footer());
            }
            Ok(VInt::deserialize(input)?.0)
        };
        let mut input = footer.as_slice();
        let fields = usize::try_from(read_vint(&mut input)?).map_err(|_| invalid_footer())?;
        if fields > input.len() / 6 || (fields == 0 && footer_start != 0) {
            return Err(invalid_footer());
        }
        let mut offset = 0usize;
        let mut seen = Vec::with_capacity(fields);
        for i in 0..fields {
            let delta = usize::try_from(read_vint(&mut input)?).map_err(|_| invalid_footer())?;
            offset = offset.checked_add(delta).ok_or_else(invalid_footer)?;
            let field = Field::deserialize(&mut input)?;
            if offset > footer_start
                || (i == 0 && offset != 0)
                || read_vint(&mut input)? != 0
                || seen.contains(&field)
            {
                return Err(invalid_footer());
            }
            seen.push(field);
        }
        if !input.is_empty() {
            return Err(invalid_footer());
        }
        let composite = CompositeFile::open(&body)?;
        Self::open_field(options, composite.open_read_with_idx(field, 0))
    }

    fn open_field(options: VectorOptions, field: Option<FileSlice>) -> io::Result<Self> {
        let Some(field) = field else {
            return Ok(Self {
                options,
                summary: Summary::default(),
                counts: FileSlice::empty(),
                pages: Vec::new(),
                legacy: None,
            });
        };
        let invalid = || {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid routing metadata summary",
            )
        };
        if field.len() < SUMMARY_LEN {
            return Err(invalid());
        }
        let bytes = field.slice_to(SUMMARY_LEN).read_bytes()?;
        if bytes.len() != SUMMARY_LEN {
            return Err(invalid());
        }
        let mut input = bytes.as_slice();
        let docs = u32::deserialize(&mut input)? as usize;
        let clusters = u32::deserialize(&mut input)? as usize;
        let rows = usize::try_from(u64::deserialize(&mut input)?).map_err(|_| invalid())?;
        let nonempty = u32::deserialize(&mut input)? as usize;
        let kind = u32::deserialize(&mut input)?;
        let count_bytes = clusters.checked_mul(4).ok_or_else(invalid)?;
        if field.len().checked_sub(SUMMARY_LEN) != Some(count_bytes)
            || kind > 1
            || docs > rows
            || nonempty > clusters
            || nonempty > rows
            || (kind == 0 && (clusters != 0 || nonempty != 0 || rows != docs))
            || (kind == 1 && clusters == 0)
            || (kind == 1 && ((rows == 0) != (nonempty == 0)))
            || (kind == 1 && rows as u64 > clusters as u64 * docs as u64)
        {
            return Err(invalid());
        }
        Ok(Self {
            options,
            summary: Summary {
                docs,
                clusters,
                rows,
                nonempty,
                clustered: kind == 1,
            },
            counts: field.slice_from(SUMMARY_LEN),
            pages: (0..count_bytes.div_ceil(PAGE_LEN))
                .map(|_| OnceLock::new())
                .collect(),
            legacy: None,
        })
    }

    pub(crate) fn options(&self) -> &VectorOptions {
        &self.options
    }
    pub(crate) fn is_clustered(&self) -> bool {
        self.summary.clustered
    }
    pub(crate) fn num_docs(&self) -> usize {
        self.summary.docs
    }
    pub(crate) fn num_rows(&self) -> usize {
        self.summary.rows
    }
    pub(crate) fn num_clusters(&self) -> usize {
        self.summary.clusters
    }
    pub(crate) fn num_non_empty_clusters(&self) -> usize {
        self.summary.nonempty
    }

    pub(crate) fn cluster_rows(&self, cluster: usize) -> crate::Result<Option<usize>> {
        if !self.is_clustered() || cluster >= self.num_clusters() {
            return Err(TantivyError::InvalidArgument(
                "routing cluster out of bounds".into(),
            ));
        }
        if let Some(legacy) = &self.legacy {
            return Ok(legacy
                .clusters()
                .unwrap()
                .non_empty_cluster_range(cluster)
                .map(|range| range.len()));
        }
        let offset = cluster * 4;
        let page = offset / PAGE_LEN;
        let bytes = if let Some(bytes) = self.pages[page].get() {
            bytes
        } else {
            let start = page * PAGE_LEN;
            let end = (start + PAGE_LEN).min(self.counts.len());
            let bytes = self.counts.slice(start..end).read_bytes()?;
            if bytes.len() != end - start {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "routing count page length mismatch",
                )
                .into());
            }
            let owned = OwnedBytes::new(bytes.as_slice().to_vec());
            let _ = self.pages[page].set(owned);
            self.pages[page].get().unwrap()
        };
        let start = offset % PAGE_LEN;
        let rows = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()) as usize;
        if rows > self.num_docs() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "routing cluster count exceeds document count",
            )
            .into());
        }
        Ok((rows > 0).then_some(rows))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::Mutex;

    use super::*;
    use crate::directory::{CompositeWrite, FileHandle};
    use crate::schema::Metric;

    #[derive(Debug)]
    struct CountedFile {
        bytes: Vec<u8>,
        reads: Mutex<Vec<Range<usize>>>,
        fault: AtomicU8,
    }

    impl HasLen for CountedFile {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl FileHandle for CountedFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            self.reads.lock().unwrap().push(range.clone());
            let mut bytes = self.bytes[range.clone()].to_vec();
            if range.start >= SUMMARY_LEN {
                match self.fault.load(Ordering::Relaxed) {
                    1 => {
                        bytes.pop();
                    }
                    2 => return Err(io::Error::other("injected count read error")),
                    3 => bytes.push(0),
                    _ => {}
                }
            }
            Ok(OwnedBytes::new(bytes))
        }
    }

    fn field_bytes(docs: usize, offsets: Option<&[u64]>) -> Vec<u8> {
        let mut bytes = Vec::new();
        serialize_field(&mut bytes, docs, offsets).unwrap();
        bytes
    }

    #[test]
    fn routing_metadata_counts_are_lazy_cached_and_owned() -> crate::Result<()> {
        let offsets: Vec<_> = (0..=2051).map(|i| (i / 3) as u64).collect();
        let file = Arc::new(CountedFile {
            bytes: field_bytes(683, Some(&offsets)),
            reads: Mutex::new(Vec::new()),
            fault: AtomicU8::new(0),
        });
        let metadata = VectorRoutingMetadata::open_field(
            VectorOptions::new(2, Metric::L2),
            Some(FileSlice::new(file.clone())),
        )?;
        assert_eq!(*file.reads.lock().unwrap(), vec![0..SUMMARY_LEN]);
        assert_eq!(metadata.num_docs(), 683);
        assert_eq!(metadata.num_rows(), 683);
        assert_eq!(metadata.num_clusters(), 2051);
        assert_eq!(metadata.num_non_empty_clusters(), 683);
        assert!(metadata.pages.iter().all(|page| page.get().is_none()));
        for id in [1023, 3, 1024, 2049, 2050, 1025, 0] {
            let rows = (offsets[id + 1] - offsets[id]) as usize;
            assert_eq!(metadata.cluster_rows(id)?, (rows > 0).then_some(rows));
        }
        assert_eq!(
            *file.reads.lock().unwrap(),
            vec![
                0..SUMMARY_LEN,
                SUMMARY_LEN..SUMMARY_LEN + PAGE_LEN,
                SUMMARY_LEN + PAGE_LEN..SUMMARY_LEN + PAGE_LEN * 2,
                SUMMARY_LEN + PAGE_LEN * 2..file.len(),
            ]
        );
        file.fault.store(2, Ordering::Relaxed);
        for id in 0..2051 {
            let rows = (offsets[id + 1] - offsets[id]) as usize;
            assert_eq!(metadata.cluster_rows(id)?, (rows > 0).then_some(rows));
        }
        assert_eq!(file.reads.lock().unwrap().len(), 4);
        assert!(metadata.cluster_rows(2051).is_err());
        assert_eq!(
            metadata
                .pages
                .iter()
                .map(|page| page.get().unwrap().len())
                .sum::<usize>(),
            2051 * 4
        );
        Ok(())
    }

    #[test]
    fn routing_metadata_count_read_failures_are_not_cached() -> crate::Result<()> {
        for fault in [1, 2, 3] {
            let file = Arc::new(CountedFile {
                bytes: field_bytes(2, Some(&[0, 1, 2])),
                reads: Mutex::new(Vec::new()),
                fault: AtomicU8::new(fault),
            });
            let metadata = VectorRoutingMetadata::open_field(
                VectorOptions::new(2, Metric::L2),
                Some(FileSlice::new(file.clone())),
            )?;
            assert!(metadata.cluster_rows(0).is_err());
            assert!(metadata.pages[0].get().is_none());
            file.fault.store(0, Ordering::Relaxed);
            assert_eq!(metadata.cluster_rows(0)?, Some(1));
            assert_eq!(file.reads.lock().unwrap().len(), 3);
        }
        Ok(())
    }

    #[test]
    fn routing_metadata_format_validates_summaries_and_counts() -> crate::Result<()> {
        let options = VectorOptions::new(2, Metric::L2);
        let field = Field::from_field_id(2);
        let mut bytes = Vec::new();
        write_header(&mut bytes)?;
        let mut composite = CompositeWrite::wrap(&mut bytes);
        serialize_field(
            composite.for_field_with_idx(field, 0),
            3,
            Some(&[0, 0, 2, 5]),
        )?;
        let flat = Field::from_field_id(4);
        serialize_field(composite.for_field_with_idx(flat, 0), 7, None)?;
        composite.close()?;
        let counted = Arc::new(CountedFile {
            bytes: bytes.clone(),
            reads: Mutex::new(Vec::new()),
            fault: AtomicU8::new(0),
        });
        let summary_only = VectorRoutingMetadata::open_file(
            options.clone(),
            FileSlice::new(counted.clone()),
            field,
        )?;
        assert!(summary_only.pages.iter().all(|page| page.get().is_none()));
        let count_range = MAGIC.len() + SUMMARY_LEN..MAGIC.len() + SUMMARY_LEN + 12;
        assert!(counted
            .reads
            .lock()
            .unwrap()
            .iter()
            .all(|range| { range.end <= count_range.start || range.start >= count_range.end }));
        let file = FileSlice::from(bytes.clone());
        let clustered = VectorRoutingMetadata::open_file(options.clone(), file.clone(), field)?;
        assert!(clustered.is_clustered());
        assert_eq!(clustered.num_rows(), 5);
        assert_eq!(clustered.num_non_empty_clusters(), 2);
        assert_eq!(clustered.cluster_rows(0)?, None);
        assert_eq!(clustered.cluster_rows(1)?, Some(2));
        assert_eq!(clustered.cluster_rows(2)?, Some(3));
        let flat = VectorRoutingMetadata::open_file(options.clone(), file.clone(), flat)?;
        assert!(!flat.is_clustered());
        assert_eq!(flat.num_docs(), 7);
        assert_eq!(flat.num_rows(), 7);
        assert!(flat.pages.is_empty());
        let missing =
            VectorRoutingMetadata::open_file(options.clone(), file, Field::from_field_id(8))?;
        assert!(!missing.is_clustered());
        assert_eq!(missing.num_docs(), 0);
        for bad in [vec![], b"VMT1".to_vec(), vec![0; 12], {
            let mut bad = bytes.clone();
            let end = bad.len();
            bad[end - 4..].fill(255);
            bad
        }] {
            assert!(
                VectorRoutingMetadata::open_file(options.clone(), FileSlice::from(bad), field)
                    .is_err()
            );
        }
        for (delta, idx) in [(1u64, 0u64), (u64::MAX, 0), (0, 1)] {
            let mut bad = MAGIC.to_vec();
            bad.extend(field_bytes(3, None));
            let mut footer = Vec::new();
            VInt(1).serialize(&mut footer)?;
            VInt(delta).serialize(&mut footer)?;
            field.serialize(&mut footer)?;
            VInt(idx).serialize(&mut footer)?;
            bad.extend(&footer);
            (footer.len() as u32).serialize(&mut bad)?;
            assert!(
                VectorRoutingMetadata::open_file(options.clone(), FileSlice::from(bad), field,)
                    .is_err()
            );
        }
        for token in 0..3 {
            for malformed in [
                vec![0; 11],
                {
                    let mut bytes = vec![0; 10];
                    bytes.push(0x80);
                    bytes
                },
                {
                    let mut bytes = vec![0; 9];
                    bytes.push(0x82);
                    bytes
                },
            ] {
                let mut footer = Vec::new();
                for (i, value) in [1, 0, 0].into_iter().enumerate() {
                    if i == 2 {
                        field.serialize(&mut footer)?;
                    }
                    if i == token {
                        footer.extend_from_slice(&malformed);
                    } else {
                        VInt(value).serialize(&mut footer)?;
                    }
                }
                let mut bad = MAGIC.to_vec();
                bad.extend(field_bytes(3, None));
                bad.extend(&footer);
                (footer.len() as u32).serialize(&mut bad)?;
                assert!(VectorRoutingMetadata::open_file(
                    options.clone(),
                    FileSlice::from(bad),
                    field,
                )
                .is_err());
            }
        }
        let valid = field_bytes(3, Some(&[0, 0, 2, 5]));
        for (start, replacement) in [(0, 6u32), (4, 4), (16, 4), (20, 2)] {
            let mut bad = valid.clone();
            bad[start..start + 4].copy_from_slice(&replacement.to_le_bytes());
            assert!(
                VectorRoutingMetadata::open_field(options.clone(), Some(FileSlice::from(bad)))
                    .is_err()
            );
        }
        for bad in [valid[..23].to_vec(), valid[..valid.len() - 1].to_vec(), {
            let mut bad = valid.clone();
            bad.push(0);
            bad
        }] {
            assert!(
                VectorRoutingMetadata::open_field(options.clone(), Some(FileSlice::from(bad)))
                    .is_err()
            );
        }
        let mut bad_count = valid;
        bad_count[SUMMARY_LEN..SUMMARY_LEN + 4].copy_from_slice(&4u32.to_le_bytes());
        let metadata =
            VectorRoutingMetadata::open_field(options, Some(FileSlice::from(bad_count)))?;
        assert!(metadata.cluster_rows(0).is_err());
        for offsets in [
            &[][..],
            &[0][..],
            &[1, 2][..],
            &[0, 2, 1][..],
            &[0, 4][..],
            &[0, u64::from(u32::MAX) + 1][..],
        ] {
            assert!(serialize_field(&mut Vec::new(), 3, Some(offsets)).is_err());
        }
        Ok(())
    }
}
