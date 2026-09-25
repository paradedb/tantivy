use std::io::{self, Write};
use std::sync::{Arc, OnceLock};

use common::file_slice::FileSlice;
use common::{CountingWriter, HasLen, VInt};

use super::serialize_column_mappable_to_u64;
use crate::column_index::{
    OptionalIndex, SelectCursor, SerializableColumnIndex, Set, open_optional_index,
    serialize_optional_index,
};
use crate::column_values::{
    CodecType, load_u64_based_column_values, serialize_u64_based_column_values,
};
use crate::iterable::Iterable;
use crate::{Column, ColumnIndex, ColumnValues, MonotonicallyMappableToU64};

// The index payload is an ordinary nullable index; missing rows repeat the preceding value.
pub(super) const INDEX_CODE: u8 = 3;

/// A required column stored as nullable run starts and one value per run.
/// Ordinary column access expands repeats through rank, without materializing them.
pub struct RunLengthColumn<T: PartialOrd> {
    starts: OnceLock<OptionalIndex>,
    index: FileSlice,
    num_docs: u32,
    values: Arc<dyn ColumnValues<T>>,
}

impl<T: PartialOrd + 'static> RunLengthColumn<T> {
    /// Non-null document positions mark the start of each run.
    pub fn starts(&self) -> &OptionalIndex {
        self.starts.get_or_init(|| {
            let starts = open_optional_index(self.index.clone()).expect("read run starts");
            assert!(
                starts.num_non_nulls() == self.values.num_vals(),
                "invalid run-length column"
            );
            starts
        })
    }

    /// Dense values have the same ordinals as the non-null run starts.
    pub fn values(&self) -> &Arc<dyn ColumnValues<T>> {
        &self.values
    }
}

impl<T: MonotonicallyMappableToU64> ColumnValues<T> for RunLengthColumn<T> {
    fn get_val(&self, doc: u32) -> T {
        assert!(doc < self.num_vals());
        self.values.get_val(self.starts().rank(doc + 1) - 1)
    }

    fn get_u32_vals(&self, docs: &[u32], output: &mut [u32]) {
        assert_eq!(docs.len(), output.len());
        let mut cursor = self.starts().select_cursor();
        let mut pos = 0;
        while pos < docs.len() {
            assert!(docs[pos] < self.num_vals());
            let ordinal = self.starts().rank(docs[pos] + 1) - 1;
            let end = if ordinal + 1 < self.values.num_vals() {
                cursor.select(ordinal + 1)
            } else {
                self.num_vals()
            };
            let value = u32::try_from(self.values.get_val(ordinal).to_u64())
                .expect("run value exceeds u32");
            while pos < docs.len() && docs[pos] < end {
                output[pos] = value;
                pos += 1;
            }
        }
    }

    fn get_range(&self, start: u64, output: &mut [T]) {
        assert!(start + output.len() as u64 <= u64::from(self.num_vals()));
        if output.is_empty() {
            return;
        }
        let mut ordinal = self.starts().rank(start as u32 + 1) - 1;
        let mut cursor = self.starts().select_cursor();
        let mut pos = 0;
        while pos < output.len() {
            let end = if ordinal + 1 < self.values.num_vals() {
                cursor.select(ordinal + 1)
            } else {
                self.num_vals()
            };
            let limit = (u64::from(end) - start).min(output.len() as u64) as usize;
            output[pos..limit].fill(self.values.get_val(ordinal));
            pos = limit;
            ordinal += 1;
        }
    }

    fn min_value(&self) -> T {
        self.values.min_value()
    }
    fn max_value(&self) -> T {
        self.values.max_value()
    }
    fn num_vals(&self) -> u32 {
        self.num_docs
    }

    fn iter(&self) -> Box<dyn Iterator<Item = T> + '_> {
        let mut cursor = self.starts().select_cursor();
        let mut start = 0;
        Box::new(self.values.iter().enumerate().flat_map(move |(i, value)| {
            let end = if i + 1 < self.values.num_vals() as usize {
                cursor.select(i as u32 + 1)
            } else {
                self.num_vals()
            };
            let count = end - start;
            start = end;
            std::iter::repeat_n(value, count as usize)
        }))
    }
}

struct Runs<'a>(&'a dyn Iterable<u64>);
impl Runs<'_> {
    /// Replays the input in final document order, retaining only the previous value.
    fn iter(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        let mut previous = None;
        self.0
            .boxed_iter()
            .enumerate()
            .filter_map(move |(row, value)| {
                if previous == Some(value) {
                    return None;
                }
                previous = Some(value);
                Some((row as u32, value))
            })
    }
}
impl Iterable<u32> for Runs<'_> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        Box::new(self.iter().map(|(row, _)| row))
    }
}
impl Iterable<u64> for Runs<'_> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        Box::new(self.iter().map(|(_, value)| value))
    }
}

/// Serializes the existing nullable index and codecs over a replayable, required column.
pub(crate) fn serialize_run_length_column(
    num_docs: u32,
    values: &dyn Iterable<u64>,
    codecs: &[CodecType],
    output: &mut impl Write,
) -> io::Result<()> {
    if num_docs == 0 {
        return serialize_column_mappable_to_u64::<u64>(
            SerializableColumnIndex::Full,
            &&[][..],
            codecs,
            output,
        );
    }
    let runs = Runs(values);
    let mut index_writer = CountingWriter::wrap(&mut *output);
    index_writer.write_all(&[INDEX_CODE])?;
    serialize_optional_index(&runs, num_docs, &mut index_writer)?;
    let index_len = u32::try_from(index_writer.written_bytes())
        .map_err(|_| io::Error::other("run index exceeds u32"))?;
    serialize_u64_based_column_values::<u64>(&runs, codecs, output)?;
    output.write_all(&index_len.to_le_bytes())
}

/// Restores ordinary full-column semantics while retaining access to the compact representation.
pub(super) fn open<T: MonotonicallyMappableToU64>(
    index: FileSlice,
    values: FileSlice,
) -> io::Result<Column<T>> {
    let mut header = index.slice(..index.len().min(10)).read_bytes()?;
    let num_docs = u32::try_from(VInt::deserialize_u64(&mut header)?)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid run document count"))?;
    let values = load_u64_based_column_values(values)?;
    if num_docs == 0 || values.num_vals() == 0 || values.num_vals() > num_docs {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid run-length column",
        ));
    }
    Ok(Column {
        index: ColumnIndex::Full,
        values: Arc::new(RunLengthColumn {
            starts: OnceLock::new(),
            index,
            num_docs,
            values,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Cardinality, ColumnType, ColumnarReader, ColumnarWriter, DynamicColumn, RowAddr,
        ShuffleMergeOrder, StackMergeOrder, merge_columnar_with_run_length,
    };

    fn write(values: &[u64], encoded: bool, mapping: Option<&[u32]>) -> ColumnarReader {
        let mut writer = ColumnarWriter::default();
        writer.record_column_type("block", ColumnType::U64, false);
        if encoded {
            writer.set_run_length_columns(&["block".into()]);
        }
        for (doc, &value) in values.iter().enumerate() {
            writer.record_numerical(doc as u32, "block", value);
        }
        let mut bytes = Vec::new();
        writer
            .serialize(
                values.len() as u32,
                mapping,
                &crate::DEFAULT_CODEC_TYPES,
                &mut bytes,
            )
            .unwrap();
        ColumnarReader::open(bytes).unwrap()
    }

    fn read(reader: &ColumnarReader) -> Column<u64> {
        let DynamicColumn::U64(column) = reader.read_columns("block").unwrap()[0].open().unwrap()
        else {
            panic!("expected u64")
        };
        column
    }

    fn check(column: &Column<u64>, expected: &[u64]) {
        assert_eq!(column.get_cardinality(), Cardinality::Full);
        assert_eq!(column.num_docs() as usize, expected.len());
        assert_eq!(column.values.iter().collect::<Vec<_>>(), expected);
        for (doc, &value) in expected.iter().enumerate() {
            assert_eq!(column.first(doc as u32), Some(value));
        }
        let docs: Vec<_> = (0..expected.len() as u32)
            .step_by(97)
            .flat_map(|d| [d, d])
            .collect();
        let mut out = vec![0; docs.len()];
        column.u32_vals(&docs, &mut out);
        assert_eq!(
            out,
            docs.iter()
                .map(|&d| expected[d as usize] as u32)
                .collect::<Vec<_>>()
        );
        for start in [0, expected.len() / 2, expected.len().saturating_sub(5)] {
            let len = (expected.len() - start).min(1024);
            let mut out = vec![0; len];
            column.values.get_range(start as u64, &mut out);
            assert_eq!(out, expected[start..start + len]);
        }
        let mut hits = Vec::new();
        column.get_docids_for_value_range(10..=11, 1..expected.len() as u32, &mut hits);
        assert_eq!(
            hits,
            expected
                .iter()
                .enumerate()
                .skip(1)
                .filter(|(_, v)| (10..=11).contains(*v))
                .map(|(i, _)| i as u32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn nullable_runs_roundtrip_and_chunk_boundaries() {
        for len in [1, 7, 65535, 65536, 65537, 131073] {
            for per_page in [1, 3, 291, 70000] {
                let values: Vec<_> = (0..len).map(|d| (d / per_page) as u64 + 10).collect();
                let reader = write(&values, true, None);
                let column = read(&reader);
                check(&column, &values);
                let runs = column.run_length().unwrap();
                assert_eq!(
                    runs.values().num_vals() as usize,
                    (len as usize).div_ceil(per_page as usize)
                );
                let mut select = runs.starts().select_cursor();
                for i in 0..runs.values().num_vals() {
                    assert_eq!(select.select(i), i * per_page as u32);
                }
            }
        }
    }

    #[test]
    fn nullable_runs_follow_final_document_order() {
        let values = [15, 11, 10, 11, 15, 10, 11];
        let mapping = [5, 2, 0, 3, 6, 1, 4];
        check(
            &read(&write(&values, true, Some(&mapping))),
            &[10, 10, 11, 11, 11, 15, 15],
        );
        let reverse: Vec<_> = (0..7).rev().collect();
        check(
            &read(&write(&[10, 10, 11, 11, 11, 15, 15], true, Some(&reverse))),
            &[15, 15, 11, 11, 11, 10, 10],
        );
    }

    #[test]
    fn nullable_runs_merge_after_deleting_run_starts() {
        let a = write(&[10, 10, 11, 11, 11, 15, 15], true, None);
        let b = write(&[10, 10, 12, 12, 15], false, None);
        let selected = [(0, 1), (1, 1), (0, 3), (0, 4), (1, 3), (0, 6), (1, 4)];
        let order = ShuffleMergeOrder::for_test(
            &[7, 5],
            selected
                .into_iter()
                .map(|(segment_ord, row_id)| RowAddr {
                    segment_ord,
                    row_id,
                })
                .collect(),
        );
        let mut output = Vec::new();
        merge_columnar_with_run_length(
            &[&a, &b],
            &[("block".into(), ColumnType::U64)],
            order.into(),
            &crate::DEFAULT_CODEC_TYPES,
            &["block".into()],
            &mut output,
            || false,
        )
        .unwrap();
        let merged = ColumnarReader::open(output).unwrap();
        check(&read(&merged), &[10, 10, 11, 11, 12, 15, 15]);
        let runs = read(&merged);
        assert_eq!(
            runs.run_length()
                .unwrap()
                .values()
                .iter()
                .collect::<Vec<_>>(),
            [10, 11, 12, 15]
        );
        let mut output = Vec::new();
        merge_columnar_with_run_length(
            &[&merged, &a],
            &[("block".into(), ColumnType::U64)],
            StackMergeOrder::stack(&[&merged, &a]).into(),
            &crate::DEFAULT_CODEC_TYPES,
            &["block".into()],
            &mut output,
            || false,
        )
        .unwrap();
        check(
            &read(&ColumnarReader::open(output).unwrap()),
            &[10, 10, 11, 11, 12, 15, 15, 10, 10, 11, 11, 11, 15, 15],
        );
    }
}
