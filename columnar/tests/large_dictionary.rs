//! A merged string column whose dictionary passes 4 GiB must reopen.
//!
//! The test writes about 9 GB of scratch files and takes a minute in release mode, so it is
//! ignored by default:
//!
//! ```sh
//! cargo test -p tantivy-columnar --release --test large_dictionary -- --ignored
//! ```

use std::fs::File;
use std::io::BufWriter;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::file_slice::{FileHandle, FileSlice};
use common::{HasLen, OwnedBytes};
use tantivy_columnar::{
    ColumnarReader, ColumnarWriter, DEFAULT_CODEC_TYPES, DynamicColumn, StackMergeOrder,
    merge_columnar,
};

const NUM_INPUTS: u64 = 4;
const DOCS_PER_INPUT: u32 = 4_500_000;
const KEY_LEN: usize = 256;

/// Reads with `pread`, so the merge never holds a whole input in memory.
#[derive(Debug)]
struct PreadFile {
    file: File,
    len: usize,
}

impl HasLen for PreadFile {
    fn len(&self) -> usize {
        self.len
    }
}

impl FileHandle for PreadFile {
    fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
        let mut buf = vec![0u8; range.len()];
        if !buf.is_empty() {
            self.file.read_exact_at(&mut buf, range.start as u64)?;
        }
        Ok(OwnedBytes::new(buf))
    }
}

fn open_file(path: &Path) -> FileSlice {
    let file = File::open(path).unwrap();
    let len = file.metadata().unwrap().len() as usize;
    FileSlice::new(Arc::new(PreadFile { file, len }))
}

/// Keys are random letters behind a unique prefix, so prefix compression saves nothing.
fn key(input: u64, doc: u32, buf: &mut [u8], state: &mut u64) -> String {
    for chunk in buf.chunks_mut(8) {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        for (byte, random) in chunk.iter_mut().zip(state.to_le_bytes()) {
            *byte = b'a' + (random % 26);
        }
    }
    let prefix = format!("{input:02}{doc:010}");
    buf[..prefix.len()].copy_from_slice(prefix.as_bytes());
    String::from_utf8(buf.to_vec()).unwrap()
}

fn write_input(dir: &Path, input: u64) -> PathBuf {
    let mut writer = ColumnarWriter::default();
    let mut buf = vec![0u8; KEY_LEN];
    let mut state = 0x9E37_79B9_7F4A_7C15 ^ (input << 32);
    for doc in 0..DOCS_PER_INPUT {
        writer.record_str(doc, "s", &key(input, doc, &mut buf, &mut state));
    }
    let path = dir.join(format!("input{input}.columnar"));
    let mut out = BufWriter::new(File::create(&path).unwrap());
    writer
        .serialize(DOCS_PER_INPUT, None, &DEFAULT_CODEC_TYPES, &mut out)
        .unwrap();
    path
}

#[test]
#[ignore]
fn merged_dictionary_past_4gib_reopens() {
    let dir = std::env::temp_dir().join(format!(
        "tantivy-columnar-large-dictionary-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let inputs: Vec<PathBuf> = (0..NUM_INPUTS)
        .map(|input| write_input(&dir, input))
        .collect();
    let readers: Vec<ColumnarReader> = inputs
        .iter()
        .map(|path| ColumnarReader::open(open_file(path)).unwrap())
        .collect();
    let readers: Vec<&ColumnarReader> = readers.iter().collect();
    let merged_path = dir.join("merged.columnar");
    let mut out = BufWriter::new(File::create(&merged_path).unwrap());
    merge_columnar(
        &readers,
        &[],
        StackMergeOrder::stack(&readers).into(),
        &DEFAULT_CODEC_TYPES,
        &mut out,
        || false,
    )
    .unwrap();
    drop(out);
    drop(readers);

    let merged = ColumnarReader::open(open_file(&merged_path)).unwrap();
    let handles = merged.read_columns("s").unwrap();
    let DynamicColumn::Str(column) = handles[0].open().unwrap() else {
        panic!("expected a str column");
    };
    let num_terms = u64::from(NUM_INPUTS as u32 * DOCS_PER_INPUT);
    assert_eq!(column.dictionary().num_terms() as u64, num_terms);
    assert!(column.dictionary().num_bytes().get_bytes() > u64::from(u32::MAX));
    let mut last_key = Vec::new();
    column
        .dictionary()
        .ord_to_term(num_terms - 1, &mut last_key)
        .unwrap();
    let mut buf = vec![0u8; KEY_LEN];
    let mut state = 0x9E37_79B9_7F4A_7C15 ^ ((NUM_INPUTS - 1) << 32);
    let mut expected = String::new();
    for doc in 0..DOCS_PER_INPUT {
        expected = key(NUM_INPUTS - 1, doc, &mut buf, &mut state);
    }
    assert_eq!(last_key, expected.as_bytes());
    std::fs::remove_dir_all(&dir).unwrap();
}
