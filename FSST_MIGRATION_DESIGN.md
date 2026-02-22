# FSST Migration Design for `tantivy_sstable`

## Goal
Migrate the generic block compression in the `sstable` subcrate from `zstd` to `fsst` for faster string compression and decompression.

## Core Design: Global Dictionary via Caller-Provided Training
Because `fsst` uses a dictionary of up to ~2.3KB, storing a local dictionary per 4KB block would add unacceptable overhead. We must use a **Global Dictionary** approach where a single FSST dictionary is trained and applied to the entire SSTable, and serialized once in the SSTable footer.

To ensure the dictionary is highly representative without buffering data internally or disrupting streaming merges, we will require the *callers* of `sstable::Writer` to provide a trained `fsst::Compressor` (or the raw training sample) when constructing the writer.

### 1. `sstable` Crate Changes
- **Dependencies:** Remove `zstd` and `zstd-compression`. Add `fsst = "0.1"`.
- **`Writer` API:** Update `SSTable::writer` and `Dictionary::builder` to optionally accept an `fsst::Compressor` (or build one from a provided `&[&[u8]]` sample). If no sample/compressor is provided, the blocks will not be compressed (or we can make compression mandatory by requiring it). We'll add a method `Dictionary::builder_with_sample(wrt: W, sample: &[&[u8]])`.
- **`DeltaWriter`:** 
  - Store the `Compressor` (if any).
  - When flushing a block, use `compress_into` with a reusable buffer instead of `zstd`.
- **Format Changes (`dictionary.rs`, `lib.rs`):**
  - Increment `SSTABLE_VERSION` to `4`.
  - When `Writer::finish()` is called, if FSST compression was used, serialize the FSST dictionary (raw `u64` symbols and `u8` lengths) and append it to the file just before the existing footer.
  - Add an 8-byte `fsst_dict_offset` to the footer. If it's `0`, no compression is used.
- **`Reader` API:**
  - `Dictionary::open()`: Read the footer. If version >= 4 and `fsst_dict_offset` != 0, seek to that offset, read the dictionary, and reconstruct an `Arc<fsst::Decompressor>`.
  - `BlockReader`: Hold a clone of the `Arc<fsst::Decompressor>`. When reading a compressed block, use `decompress_into` with a properly sized reusable buffer (accounting for the `+7` padding requirement for `fsst`'s safety).

### 2. Indexing Phase (In-Memory `Vec` Sampling)
During fresh indexing, terms are collected in memory before being serialized.
- **`src/postings/postings_writer.rs`:** In `serialize_postings`, terms are collected into a `term_offsets` Vec and sorted. We will take a uniform random sample (e.g., every Nth term, up to ~64KB of strings) from this Vec and use it to initialize the `sstable::Writer`.
- **`columnar/src/dictionary.rs`:** In `DictionaryBuilder::serialize`, terms are collected into a `terms` Vec and sorted. We will take a uniform random sample from this Vec and use it to initialize the `sstable::Writer`.

### 3. Merging Phase (Dictionary Iteration Sampling)
During segment merging, terms are streamed and not fully buffered in memory.
- **`src/indexer/merger.rs` & `columnar/src/columnar/merge/merge_dict_column.rs`:** 
  - Before starting the `TermMerger` stream, we will inspect the input segment dictionaries.
  - We know the total number of terms (`num_terms()`).
  - We will determine a sample size (e.g., 1,000 terms) and distribute it proportionally across the input segments.
  - For each segment, we generate a sorted list of target ordinals (e.g., using a uniform stride).
  - We will use `Dictionary::sorted_ords_to_term_cb(ords, callback)` to efficiently extract these specific terms directly from the blocks into a flat `Vec<Vec<u8>>` buffer.
  - We use this buffer to train the `fsst::Compressor` and pass it to the `sstable::Writer`. This provides perfect sampling without interrupting the streaming merge.
