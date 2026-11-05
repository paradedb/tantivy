//! The `.syn` component of a segment.
//!
//! The file starts with the group section, which holds the alternatives of each rule once. Then
//! comes an sstable from the analyzed tokens of a key to the groups that expand it, and a footer.
//! A key points into the group section, so the keys of one equivalence rule share one copy of
//! its alternatives, the layout of Lucene's `SynonymMap`.

use std::io::{self, Write};

use common::file_slice::FileSlice;
use common::{HasLen, OwnedBytes};
use sstable::value::{ValueReader, ValueWriter};
use sstable::{Dictionary, SSTable};

use crate::DocId;

/// The extension of the synonym component of a segment.
pub const SYNONYM_EXT: &str = "syn";

const FORMAT_VERSION: u8 = 1;

/// The length of the group section as a `u64`, the longest key in tokens as a `u32`, then the
/// format version as a `u8`.
const FOOTER_LEN: usize = 13;

/// The footer `sstable::Dictionary::open` reads from the end of its slice.
const SSTABLE_FOOTER_LEN: usize = 20;

/// Joins the tokens of a key. A rule cannot contain this byte, so two keys never join to the
/// same bytes.
const KEY_SEPARATOR: u8 = 0;

pub(crate) fn encode_key(tokens: &[String]) -> Vec<u8> {
    tokens
        .iter()
        .map(String::as_bytes)
        .collect::<Vec<_>>()
        .join(&KEY_SEPARATOR)
}

fn key_tokens(key: &[u8]) -> usize {
    key.iter().filter(|&&byte| byte == KEY_SEPARATOR).count() + 1
}

/// Where the alternatives of one rule are in the group section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupRef {
    /// The rule's document, so a lookup skips a deleted rule without reading its group.
    pub(crate) doc: DocId,
    offset: u64,
    len: u64,
}

/// The alternatives that one rule gives a key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleAlternatives {
    /// The rule's document.
    pub doc: DocId,
    /// Each alternative is a list of analyzed tokens.
    pub alternatives: Vec<Vec<String>>,
}

/// The groups of one key, in document order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SynonymSSTable;

impl SSTable for SynonymSSTable {
    type Value = Vec<GroupRef>;
    type ValueReader = GroupRefsReader;
    type ValueWriter = GroupRefsWriter;
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn damaged(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("damaged synonym dictionary: {what}"),
    )
}

fn get_varint(data: &mut &[u8]) -> io::Result<u64> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let (&byte, rest) = data
            .split_first()
            .ok_or_else(|| damaged("truncated varint"))?;
        *data = rest;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift > 63 {
            return Err(damaged("varint too long"));
        }
    }
}

fn get_count(data: &mut &[u8]) -> io::Result<usize> {
    let count = get_varint(data)?;
    usize::try_from(count).map_err(|_| damaged("count too large"))
}

fn get_doc(data: &mut &[u8]) -> io::Result<DocId> {
    DocId::try_from(get_varint(data)?).map_err(|_| damaged("doc id out of range"))
}

fn encode_alternatives(alternatives: &[Vec<String>], out: &mut Vec<u8>) {
    put_varint(out, alternatives.len() as u64);
    for alternative in alternatives {
        put_varint(out, alternative.len() as u64);
        for token in alternative {
            put_varint(out, token.len() as u64);
            out.extend_from_slice(token.as_bytes());
        }
    }
}

fn decode_alternatives(mut data: &[u8]) -> io::Result<Vec<Vec<String>>> {
    // Each reservation is capped by the bytes left, so a damaged count cannot ask for an
    // arbitrary allocation.
    let num_alternatives = get_count(&mut data)?;
    let mut alternatives = Vec::with_capacity(num_alternatives.min(data.len()));
    for _ in 0..num_alternatives {
        let num_tokens = get_count(&mut data)?;
        let mut alternative = Vec::with_capacity(num_tokens.min(data.len()));
        for _ in 0..num_tokens {
            let len = get_count(&mut data)?;
            if len > data.len() {
                return Err(damaged("truncated token"));
            }
            let (token, rest) = data.split_at(len);
            data = rest;
            let token =
                String::from_utf8(token.to_vec()).map_err(|_| damaged("token is not UTF-8"))?;
            alternative.push(token);
        }
        alternatives.push(alternative);
    }
    if !data.is_empty() {
        return Err(damaged("trailing bytes in a group"));
    }
    Ok(alternatives)
}

#[derive(Default)]
pub(crate) struct GroupRefsWriter {
    num_values: u64,
    buffer: Vec<u8>,
}

impl ValueWriter for GroupRefsWriter {
    type Value = Vec<GroupRef>;

    fn write(&mut self, groups: &Vec<GroupRef>) {
        self.num_values += 1;
        put_varint(&mut self.buffer, groups.len() as u64);
        for group in groups {
            put_varint(&mut self.buffer, u64::from(group.doc));
            put_varint(&mut self.buffer, group.offset);
            put_varint(&mut self.buffer, group.len);
        }
    }

    fn serialize_block(&self, output: &mut Vec<u8>) {
        put_varint(output, self.num_values);
        output.extend_from_slice(&self.buffer);
    }

    fn clear(&mut self) {
        self.num_values = 0;
        self.buffer.clear();
    }
}

/// A damaged block can have fewer values than keys. Its extra keys read as having no groups.
static NO_GROUPS: Vec<GroupRef> = Vec::new();

#[derive(Default)]
pub(crate) struct GroupRefsReader {
    values: Vec<Vec<GroupRef>>,
}

impl ValueReader for GroupRefsReader {
    type Value = Vec<GroupRef>;

    fn value(&self, idx: usize) -> &Vec<GroupRef> {
        self.values.get(idx).unwrap_or(&NO_GROUPS)
    }

    fn load(&mut self, mut data: &[u8]) -> io::Result<usize> {
        let original_len = data.len();
        self.values.clear();
        let num_values = get_count(&mut data)?;
        self.values.reserve(num_values.min(data.len()));
        for _ in 0..num_values {
            let num_groups = get_count(&mut data)?;
            let mut groups = Vec::with_capacity(num_groups.min(data.len()));
            for _ in 0..num_groups {
                let doc = get_doc(&mut data)?;
                let offset = get_varint(&mut data)?;
                let len = get_varint(&mut data)?;
                groups.push(GroupRef { doc, offset, len });
            }
            self.values.push(groups);
        }
        Ok(original_len - data.len())
    }
}

/// Writes the group section of a `.syn` component. The keys come after it, through
/// [`SynonymKeysWriter`].
pub(crate) struct SynonymFileWriter<W: Write> {
    wrt: W,
    groups_len: u64,
    buffer: Vec<u8>,
}

impl<W: Write> SynonymFileWriter<W> {
    pub(crate) fn new(wrt: W) -> Self {
        SynonymFileWriter {
            wrt,
            groups_len: 0,
            buffer: Vec::new(),
        }
    }

    pub(crate) fn add_group(
        &mut self,
        doc: DocId,
        alternatives: &[Vec<String>],
    ) -> io::Result<GroupRef> {
        let mut payload = std::mem::take(&mut self.buffer);
        payload.clear();
        encode_alternatives(alternatives, &mut payload);
        let group = self.add_encoded_group(doc, &payload);
        self.buffer = payload;
        group
    }

    /// Adds a group in its encoded form, so a merge copies the bytes without decoding them.
    pub(crate) fn add_encoded_group(&mut self, doc: DocId, payload: &[u8]) -> io::Result<GroupRef> {
        let mut header = Vec::with_capacity(20);
        put_varint(&mut header, u64::from(doc));
        put_varint(&mut header, payload.len() as u64);
        self.wrt.write_all(&header)?;
        self.wrt.write_all(payload)?;
        let offset = self.groups_len + header.len() as u64;
        self.groups_len = offset + payload.len() as u64;
        Ok(GroupRef {
            doc,
            offset,
            len: payload.len() as u64,
        })
    }

    pub(crate) fn into_keys(self) -> SynonymKeysWriter<W> {
        SynonymKeysWriter {
            sstable: SynonymSSTable::writer(self.wrt),
            groups_len: self.groups_len,
            max_key_tokens: 0,
            last_key: None,
        }
    }
}

/// Writes the keys of a `.syn` component, in byte order of their joined form.
pub(crate) struct SynonymKeysWriter<W: Write> {
    sstable: sstable::Writer<W, GroupRefsWriter>,
    groups_len: u64,
    max_key_tokens: usize,
    last_key: Option<Vec<u8>>,
}

impl<W: Write> SynonymKeysWriter<W> {
    /// Inserts a key as joined by [`encode_key`]. A key out of order returns an error, because a
    /// merge reads its keys from files that can be damaged and the sstable writer panics on one.
    pub(crate) fn insert(&mut self, key: &[u8], groups: &Vec<GroupRef>) -> io::Result<()> {
        if self
            .last_key
            .as_deref()
            .is_some_and(|last_key| last_key >= key)
        {
            return Err(damaged("keys out of order"));
        }
        self.last_key = Some(key.to_vec());
        self.max_key_tokens = self.max_key_tokens.max(key_tokens(key));
        self.sstable.insert(key, groups)
    }

    pub(crate) fn finish(self) -> io::Result<W> {
        let max_key_tokens = u32::try_from(self.max_key_tokens)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "synonym key too long"))?;
        let mut wrt = self.sstable.finish()?;
        wrt.write_all(&self.groups_len.to_le_bytes())?;
        wrt.write_all(&max_key_tokens.to_le_bytes())?;
        wrt.write_all(&[FORMAT_VERSION])?;
        Ok(wrt)
    }
}

/// The synonym component of one segment, read by range from its `FileSlice`, so a lookup reads
/// a few blocks and no reader holds the whole dictionary in memory.
#[derive(Clone, Debug)]
pub struct SynonymDictionary {
    groups: FileSlice,
    keys: Dictionary<SynonymSSTable>,
    max_key_tokens: usize,
}

impl SynonymDictionary {
    /// Opens a `.syn` component.
    pub fn open(file: FileSlice) -> io::Result<Self> {
        if file.len() < FOOTER_LEN {
            return Err(damaged("file shorter than its footer"));
        }
        let (body, footer) = file.split_from_end(FOOTER_LEN);
        let footer = footer.read_bytes()?;
        let version = footer[FOOTER_LEN - 1];
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown synonym dictionary format {version}"),
            ));
        }
        let groups_len = u64::from_le_bytes(footer[..8].try_into().unwrap());
        let max_key_tokens = u32::from_le_bytes(footer[8..12].try_into().unwrap()) as usize;
        let groups_len = usize::try_from(groups_len)
            .ok()
            .filter(|&groups_len| {
                groups_len
                    .checked_add(SSTABLE_FOOTER_LEN)
                    .is_some_and(|min_len| min_len <= body.len())
            })
            .ok_or_else(|| damaged("group section past the end of the file"))?;
        let (groups, keys) = body.split(groups_len);
        Ok(SynonymDictionary {
            groups,
            keys: Dictionary::open(keys)?,
            max_key_tokens,
        })
    }

    /// The most tokens in one key. An expansion tries no longer run at a position.
    pub fn max_key_tokens(&self) -> usize {
        self.max_key_tokens
    }

    /// The number of keys, deleted rules included.
    pub fn num_keys(&self) -> usize {
        self.keys.num_terms()
    }

    /// The rules that expand `key`, in document order, deleted ones included.
    pub fn get(&self, key: &[String]) -> io::Result<Option<Vec<RuleAlternatives>>> {
        let Some(groups) = self.groups_of(key)? else {
            return Ok(None);
        };
        let mut rules = Vec::with_capacity(groups.len());
        for group in &groups {
            rules.push(RuleAlternatives {
                doc: group.doc,
                alternatives: self.alternatives(group)?,
            });
        }
        Ok(Some(rules))
    }

    pub(crate) fn groups_of(&self, key: &[String]) -> io::Result<Option<Vec<GroupRef>>> {
        self.keys.get(encode_key(key))
    }

    pub(crate) fn alternatives(&self, group: &GroupRef) -> io::Result<Vec<Vec<String>>> {
        let range = usize::try_from(group.offset)
            .ok()
            .zip(usize::try_from(group.len).ok())
            .and_then(|(start, len)| Some(start..start.checked_add(len)?))
            .filter(|range| range.end <= self.groups.len())
            .ok_or_else(|| damaged("group past the end of its section"))?;
        decode_alternatives(&self.groups.read_bytes_slice(range)?)
    }

    /// Reads the whole group section, for a merge.
    pub(crate) fn read_groups(&self) -> io::Result<OwnedBytes> {
        self.groups.read_bytes()
    }

    /// Reads the whole sstable, for a merge that streams every key.
    pub(crate) fn read_keys(&self) -> io::Result<sstable::Reader<GroupRefsReader>> {
        Ok(SynonymSSTable::reader(
            self.keys.sstable_slice.read_bytes()?,
        ))
    }
}

/// One group of a group section, as [`groups`] returns it.
pub(crate) struct EncodedGroup<'a> {
    pub(crate) doc: DocId,
    /// Where the group is in its section, the offset a [`GroupRef`] holds.
    pub(crate) offset: u64,
    pub(crate) payload: &'a [u8],
}

impl GroupRef {
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }
}

/// Walks a group section in file order.
pub(crate) fn groups(section: &[u8]) -> impl Iterator<Item = io::Result<EncodedGroup<'_>>> + '_ {
    let mut rest = section;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let group = (|| {
            let doc = get_doc(&mut rest)?;
            let len = get_count(&mut rest)?;
            if len > rest.len() {
                return Err(damaged("truncated group"));
            }
            let offset = (section.len() - rest.len()) as u64;
            let (payload, after) = rest.split_at(len);
            rest = after;
            Ok(EncodedGroup {
                doc,
                offset,
                payload,
            })
        })();
        if group.is_err() {
            rest = &[];
        }
        Some(group)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::ops::Range;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use common::file_slice::FileHandle;

    use super::*;

    fn strings(terms: &[&str]) -> Vec<String> {
        terms.iter().map(|term| term.to_string()).collect()
    }

    fn alternatives(alternatives: &[&[&str]]) -> Vec<Vec<String>> {
        alternatives
            .iter()
            .map(|alternative| strings(alternative))
            .collect()
    }

    /// A rule's document, its keys and its alternatives.
    type TestRule<'a> = (DocId, &'a [&'a [&'a str]], &'a [&'a [&'a str]]);

    /// Writes one group per rule, and points each key at the groups of the rules that list it.
    fn build(rules: &[TestRule]) -> Vec<u8> {
        let mut file = SynonymFileWriter::new(Vec::new());
        let mut keys: Vec<(Vec<u8>, Vec<GroupRef>)> = Vec::new();
        for &(doc, rule_keys, rule_alternatives) in rules {
            let group = file
                .add_group(doc, &alternatives(rule_alternatives))
                .unwrap();
            for key in rule_keys {
                let key = encode_key(&strings(key));
                match keys.iter_mut().find(|(existing, _)| *existing == key) {
                    Some((_, groups)) => groups.push(group),
                    None => keys.push((key, vec![group])),
                }
            }
        }
        keys.sort_by(|left, right| left.0.cmp(&right.0));
        let mut writer = file.into_keys();
        for (key, groups) in &keys {
            writer.insert(key, groups).unwrap();
        }
        writer.finish().unwrap()
    }

    fn open(bytes: Vec<u8>) -> io::Result<SynonymDictionary> {
        SynonymDictionary::open(FileSlice::from(bytes))
    }

    #[test]
    fn values_round_trip() {
        let dictionary = open(build(&[
            (0, &[&["couch"], &["sofa"]], &[&["couch"], &["sofa"]]),
            (3, &[&["run", "shoe"]], &[&["run", "shoe"], &["sneaker"]]),
            (7, &[&["sofa"]], &[&["settee"]]),
            (u32::MAX, &[&["日本"]], &[&["にほん"]]),
        ]))
        .unwrap();
        assert_eq!(dictionary.num_keys(), 4);
        assert_eq!(dictionary.max_key_tokens(), 2);
        let rule = |doc, alts: &[&[&str]]| RuleAlternatives {
            doc,
            alternatives: alternatives(alts),
        };
        assert_eq!(
            dictionary.get(&strings(&["sofa"])).unwrap(),
            Some(vec![
                rule(0, &[&["couch"], &["sofa"]]),
                rule(7, &[&["settee"]])
            ])
        );
        assert_eq!(
            dictionary.get(&strings(&["run", "shoe"])).unwrap(),
            Some(vec![rule(3, &[&["run", "shoe"], &["sneaker"]])])
        );
        assert_eq!(
            dictionary.get(&strings(&["日本"])).unwrap(),
            Some(vec![rule(u32::MAX, &[&["にほん"]])])
        );
        assert_eq!(dictionary.get(&strings(&["run"])).unwrap(), None);
        assert_eq!(dictionary.get(&strings(&["shoe"])).unwrap(), None);
    }

    #[test]
    fn the_keys_of_one_rule_share_one_group() {
        let dictionary = open(build(&[(
            0,
            &[&["couch"], &["sofa"], &["divan"]],
            &[&["couch"], &["sofa"], &["divan"]],
        )]))
        .unwrap();
        let couch = dictionary.groups_of(&strings(&["couch"])).unwrap();
        assert_eq!(couch, dictionary.groups_of(&strings(&["sofa"])).unwrap());
        assert_eq!(couch, dictionary.groups_of(&strings(&["divan"])).unwrap());
        assert_eq!(couch.unwrap().len(), 1);
    }

    #[test]
    fn a_damaged_file_is_an_error() {
        let bytes = build(&[(0, &[&["sofa"]], &[&["couch"]])]);
        for len in [0, 3, FOOTER_LEN, FOOTER_LEN + SSTABLE_FOOTER_LEN - 1] {
            let mut short = vec![0u8; len];
            if len > 0 {
                *short.last_mut().unwrap() = FORMAT_VERSION;
            }
            assert!(open(short).is_err(), "{len} bytes opened");
        }
        let mut wrong_version = bytes.clone();
        *wrong_version.last_mut().unwrap() = FORMAT_VERSION + 1;
        assert_eq!(
            open(wrong_version).unwrap_err().to_string(),
            "unknown synonym dictionary format 2"
        );
        let mut long_groups = bytes.clone();
        let footer = long_groups.len() - FOOTER_LEN;
        long_groups[footer..footer + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(open(long_groups).is_err());

        let dictionary = open(bytes).unwrap();
        let group = dictionary.groups_of(&strings(&["sofa"])).unwrap().unwrap()[0];
        let past_end = GroupRef {
            offset: group.offset + 1_000,
            ..group
        };
        assert!(dictionary.alternatives(&past_end).is_err());
        let short = GroupRef {
            len: group.len - 1,
            ..group
        };
        assert!(dictionary.alternatives(&short).is_err());
    }

    #[test]
    fn a_damaged_group_is_an_error() {
        // A count of alternatives past the end of the group.
        assert!(decode_alternatives(&[0xff, 0x01]).is_err());
        // A token length past the end of the group.
        assert!(decode_alternatives(&[1, 1, 9, b'a']).is_err());
        // A varint that never ends.
        assert!(decode_alternatives(&[0xff; 12]).is_err());
        // Not UTF-8.
        assert!(decode_alternatives(&[1, 1, 1, 0xff]).is_err());
        // Bytes after the last alternative.
        assert!(decode_alternatives(&[1, 1, 1, b'a', 0]).is_err());
        assert_eq!(
            decode_alternatives(&[1, 1, 1, b'a']).unwrap(),
            vec![strings(&["a"])]
        );
    }

    #[test]
    fn the_value_reader_returns_the_bytes_it_read() {
        let group = GroupRef {
            doc: 2,
            offset: 300,
            len: 17,
        };
        let mut writer = GroupRefsWriter::default();
        writer.write(&vec![group, group]);
        writer.write(&vec![]);
        let mut block = Vec::new();
        writer.serialize_block(&mut block);
        let len = block.len();
        block.extend_from_slice(b"next block");
        let mut reader = GroupRefsReader::default();
        assert_eq!(reader.load(&block).unwrap(), len);
        assert_eq!(reader.value(0), &vec![group, group]);
        assert!(reader.value(1).is_empty());
        // A damaged block with fewer values than keys reads as no groups.
        assert!(reader.value(2).is_empty());
        assert!(reader.load(&[1, 1, 0xff]).is_err());
    }

    /// A file with no in-memory slice, so the sstable index takes its paged path, the way it
    /// does on Postgres blocks.
    #[derive(Debug)]
    struct CountingFile {
        bytes: OwnedBytes,
        read: Arc<AtomicUsize>,
    }

    impl HasLen for CountingFile {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl FileHandle for CountingFile {
        fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
            self.read.fetch_add(range.len(), Ordering::Relaxed);
            Ok(self.bytes.slice(range))
        }
    }

    #[test]
    fn a_lookup_in_a_large_dictionary_reads_a_few_blocks() {
        const NUM_RULES: usize = 100_000;
        let mut file = SynonymFileWriter::new(Vec::new());
        let mut keys = Vec::with_capacity(NUM_RULES);
        for i in 0..NUM_RULES {
            let key = format!("term{i:06}");
            let group = file
                .add_group(i as DocId, &[vec![format!("syn{i:06}")], vec![key.clone()]])
                .unwrap();
            keys.push((encode_key(&[key]), vec![group]));
        }
        let mut writer = file.into_keys();
        for (key, groups) in &keys {
            writer.insert(key, groups).unwrap();
        }
        let bytes = writer.finish().unwrap();

        let read = Arc::new(AtomicUsize::new(0));
        let file = FileSlice::new(Arc::new(CountingFile {
            bytes: OwnedBytes::new(bytes.clone()),
            read: read.clone(),
        }));
        let dictionary = SynonymDictionary::open(file).unwrap();
        assert_eq!(dictionary.num_keys(), NUM_RULES);

        for i in [0, 4_242, NUM_RULES / 2, NUM_RULES - 1] {
            read.store(0, Ordering::Relaxed);
            let key = format!("term{i:06}");
            let rules = dictionary.get(std::slice::from_ref(&key)).unwrap().unwrap();
            assert_eq!(
                rules[0].alternatives,
                vec![vec![format!("syn{i:06}")], vec![key]]
            );
            let bytes_read = read.load(Ordering::Relaxed);
            assert!(
                bytes_read * 50 < bytes.len(),
                "a lookup read {bytes_read} of {} bytes",
                bytes.len()
            );
        }
        assert_eq!(dictionary.get(&strings(&["term1"])).unwrap(), None);
    }
}
