//! Encoded rows between the assign pass and the Data entry: fixed-width records sorted by
//! (cluster, document), buffered in memory and spilled to temporary files as sorted runs, then
//! merged back in that order.
//!
//! A record is the cluster (`u32`), the target document (`u32`) and the primary flag (`u8`),
//! little-endian, followed by the row's bytes for every non-DocIds slot of the block, in slot
//! order.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use crate::directory::{Directory, TempFilePtr};
use crate::vector::storage::metadata::{Slot, SlotType};
use crate::DocId;

/// Bytes of a record before its slot bytes: cluster, document, primary flag.
const RECORD_HEADER: usize = 9;

/// Most bytes one spilled run buffers while the merge reads it.
const RUN_READ_BYTES: usize = 256 << 10;

/// Most bytes buffered while a run is written.
const RUN_WRITE_BYTES: usize = 1 << 20;

/// Bookkeeping per buffered record while it is sorted: its key and position.
pub(super) const RECORD_SORT_BYTES: usize = 16;

/// Where each slot's row bytes sit in a record.
pub(super) struct RecordLayout {
    pub(super) len: usize,
    /// `(slot index, offset, stride)` of every slot a record carries, in slot order.
    pub(super) fields: Vec<(usize, usize, usize)>,
    pub(super) doc_ids_slot: usize,
}

impl RecordLayout {
    /// The layout for blocks of `slots`: every slot but DocIds, whose value is the header's
    /// document. Blocks that hold rows have no record layout.
    pub(super) fn new(slots: &[Slot]) -> Self {
        let mut len = RECORD_HEADER;
        let mut fields = Vec::new();
        let mut doc_ids_slot = None;
        for (idx, slot) in slots.iter().enumerate() {
            match slot.slot_type {
                SlotType::DocIds => doc_ids_slot = Some(idx),
                SlotType::Rows { .. } => panic!("encoded records never carry rows"),
                _ => {
                    fields.push((idx, len, slot.stride as usize));
                    len += slot.stride as usize;
                }
            }
        }
        RecordLayout {
            len,
            fields,
            doc_ids_slot: doc_ids_slot.expect("clustered blocks carry document ids"),
        }
    }
}

/// What a record says about its row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RecordHeader {
    pub(super) cluster: u32,
    pub(super) doc: DocId,
    /// Whether this is the document's primary row, the one its location points at.
    pub(super) primary: bool,
}

impl RecordHeader {
    pub(super) fn write(self, record: &mut [u8]) {
        record[..4].copy_from_slice(&self.cluster.to_le_bytes());
        record[4..8].copy_from_slice(&self.doc.to_le_bytes());
        record[8] = u8::from(self.primary);
    }

    pub(super) fn read(record: &[u8]) -> Self {
        let word = |at: usize| u32::from_le_bytes(record[at..at + 4].try_into().unwrap());
        RecordHeader {
            cluster: word(0),
            doc: word(4),
            primary: record[8] != 0,
        }
    }
}

/// A record's position in (cluster, document) order.
fn key(record: &[u8]) -> u64 {
    let RecordHeader { cluster, doc, .. } = RecordHeader::read(record);
    (u64::from(cluster) << 32) | u64::from(doc)
}

/// Records buffered in memory, spilled as sorted runs once the buffer is full.
pub(super) struct RecordRuns<'d> {
    directory: &'d dyn Directory,
    record_len: usize,
    /// Records the buffer holds before it spills.
    capacity: usize,
    records: Vec<u8>,
    /// Spilled runs, each a temporary file and its record count.
    runs: Vec<(TempFilePtr, usize)>,
}

impl<'d> RecordRuns<'d> {
    pub(super) fn new(directory: &'d dyn Directory, record_len: usize, capacity: usize) -> Self {
        RecordRuns {
            directory,
            record_len,
            capacity: capacity.max(1),
            records: Vec::new(),
            runs: Vec::new(),
        }
    }

    fn buffered(&self) -> usize {
        self.records.len() / self.record_len
    }

    /// Room for `n` more records, zeroed, after spilling the buffer as a run when they would not
    /// fit beside it. The caller writes every record of the returned slice.
    pub(super) fn append(&mut self, n: usize) -> io::Result<&mut [u8]> {
        if self.buffered() > 0 && self.buffered() + n > self.capacity {
            self.spill()?;
        }
        let start = self.records.len();
        self.records.resize(start + n * self.record_len, 0);
        Ok(&mut self.records[start..])
    }

    /// The buffered records' positions in (cluster, document) order.
    fn sorted_order(&self) -> Vec<u32> {
        let mut keyed: Vec<(u64, u32)> = self
            .records
            .chunks_exact(self.record_len)
            .enumerate()
            .map(|(at, record)| (key(record), at as u32))
            .collect();
        keyed.sort_unstable();
        keyed.into_iter().map(|(_, at)| at).collect()
    }

    /// Writes the buffer, sorted, to a temporary file of its own and empties it.
    fn spill(&mut self) -> io::Result<()> {
        let order = self.sorted_order();
        let mut file = self.directory.open_temp_file()?;
        {
            let mut writer =
                BufWriter::with_capacity(RUN_WRITE_BYTES.min(self.records.len()), &mut file);
            for &at in &order {
                let at = at as usize * self.record_len;
                writer.write_all(&self.records[at..at + self.record_len])?;
            }
            writer.flush()?;
        }
        self.runs.push((file, order.len()));
        self.records.clear();
        Ok(())
    }

    /// Every record in (cluster, document) order: the spilled runs and the records still
    /// buffered, which stay in memory.
    pub(super) fn into_merged(self) -> io::Result<MergedRecords> {
        let record_len = self.record_len;
        let order = self.sorted_order();
        let mut sources = Vec::with_capacity(self.runs.len() + 1);
        for (mut file, records) in self.runs {
            file.seek(SeekFrom::Start(0))?;
            let capacity = RUN_READ_BYTES.min(records * record_len).max(record_len);
            sources.push(Source::Run {
                reader: BufReader::with_capacity(capacity, file),
                remaining: records,
                record: vec![0; record_len],
            });
        }
        sources.push(Source::Memory {
            records: self.records,
            order,
            next: 0,
        });
        let mut merged = MergedRecords {
            record_len,
            sources,
            heap: BinaryHeap::new(),
            current: vec![0; record_len],
        };
        for source in 0..merged.sources.len() {
            merged.advance(source)?;
        }
        Ok(merged)
    }
}

/// One sorted input of the merge.
enum Source {
    Run {
        reader: BufReader<TempFilePtr>,
        /// Records not yet read.
        remaining: usize,
        /// The record read last.
        record: Vec<u8>,
    },
    Memory {
        records: Vec<u8>,
        order: Vec<u32>,
        /// Position in `order` of the next record.
        next: usize,
    },
}

/// The k-way merge of sorted runs.
pub(super) struct MergedRecords {
    record_len: usize,
    sources: Vec<Source>,
    /// The key of each source's current record, smallest first.
    heap: BinaryHeap<Reverse<(u64, usize)>>,
    current: Vec<u8>,
}

impl MergedRecords {
    /// Moves `source` to its next record and queues it; an exhausted source leaves the merge.
    fn advance(&mut self, source: usize) -> io::Result<()> {
        let record_len = self.record_len;
        let key = match &mut self.sources[source] {
            Source::Run {
                reader,
                remaining,
                record,
            } => {
                if *remaining == 0 {
                    return Ok(());
                }
                reader.read_exact(record)?;
                *remaining -= 1;
                key(record)
            }
            Source::Memory {
                records,
                order,
                next,
            } => {
                let Some(&at) = order.get(*next) else {
                    return Ok(());
                };
                let at = at as usize * record_len;
                key(&records[at..at + record_len])
            }
        };
        self.heap.push(Reverse((key, source)));
        Ok(())
    }

    /// The next record in (cluster, document) order, or `None` once every source is drained.
    pub(super) fn next_record(&mut self) -> io::Result<Option<&[u8]>> {
        let Some(Reverse((_, source))) = self.heap.pop() else {
            return Ok(None);
        };
        let record_len = self.record_len;
        match &mut self.sources[source] {
            Source::Run { record, .. } => self.current.copy_from_slice(record),
            Source::Memory {
                records,
                order,
                next,
            } => {
                let at = order[*next] as usize * record_len;
                self.current.copy_from_slice(&records[at..at + record_len]);
                *next += 1;
            }
        }
        self.advance(source)?;
        Ok(Some(&self.current))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::RamDirectory;

    /// Records come back from the merge in (cluster, document) order, each once with its bytes,
    /// whether they stayed in memory or were spilled across many runs.
    #[test]
    fn merge_yields_every_record_in_key_order() -> io::Result<()> {
        const RECORD_LEN: usize = RECORD_HEADER + 3;
        let records: Vec<(RecordHeader, [u8; 3])> = (0..500u32)
            .map(|i| {
                let header = RecordHeader {
                    cluster: i * 7_919 % 13,
                    doc: i * 104_729 % 1_009,
                    primary: i % 3 == 0,
                };
                (header, [i as u8, (i >> 8) as u8, 0xA5])
            })
            .collect();
        let mut expected = records.clone();
        expected.sort_by_key(|(header, _)| (header.cluster, header.doc));
        let directory = RamDirectory::default();
        for (capacity, chunk, runs) in [(1_000, 64, 0), (37, 5, 14), (1, 1, 499)] {
            let mut buffer = RecordRuns::new(&directory, RECORD_LEN, capacity);
            for group in records.chunks(chunk) {
                let region = buffer.append(group.len())?;
                for ((header, payload), record) in
                    group.iter().zip(region.chunks_exact_mut(RECORD_LEN))
                {
                    header.write(record);
                    record[RECORD_HEADER..].copy_from_slice(payload);
                }
            }
            assert_eq!(buffer.runs.len(), runs, "capacity {capacity}");
            let mut merged = buffer.into_merged()?;
            let mut actual = Vec::new();
            while let Some(record) = merged.next_record()? {
                actual.push((
                    RecordHeader::read(record),
                    <[u8; 3]>::try_from(&record[RECORD_HEADER..]).unwrap(),
                ));
            }
            assert_eq!(actual, expected, "capacity {capacity}");
        }
        Ok(())
    }
}
