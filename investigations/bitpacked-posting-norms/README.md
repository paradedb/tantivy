# Bitpacked posting-local fieldnorms

Branch: `codex/bitpacked-posting-norms`, based on raw posting-local prototype `1ac97127c`. Experiment date: 2026-09-21.

Fixed-width compression preserves O(1) access and identical scores, but this separate-component layout is not a query-performance improvement. It saves 9.10% of norm-stream bytes overall; the six measured queries use 50–69 more shared-buffer hits and have approximately 3–6% higher median execution times than raw posting-local norms.

## Format

The value stored per posting is Tantivy's existing quantized one-byte `fieldnorm_id`, not an exact document length. Compression is lossless over these IDs. Values are in a separate stream in posting order, not interleaved with document IDs and term frequencies.

Each field's logical stream is divided into 128-value blocks, which can cross term boundaries. Each block stores a minimum byte and bit-width byte, followed by fixed-width unsigned differences from that shared minimum. Width zero represents a constant block; width eight stores raw values. The final block uses its actual length. Access uses `minimum + unpack(i * width)`, without decoding preceding values.

An offset directory uses four-byte offsets, or eight-byte offsets for payloads exceeding 4 GiB. A 21-byte footer stores logical length, directory start, offset width, and `PNB1` magic. Optional `.bpnorm` components coexist with raw `.pnorm` components, preserving original term offsets and permitting same-index A/B comparisons. Missing packed components fall back to raw norms.

Metadata opens lazily and is cached per field source. Readers buffer directory and payload bytes separately. Read-ahead is bounded to the queried term's logical range; an initial unrestricted implementation fetched unnecessary neighboring terms and was corrected before the final measurements.

## Storage

| Stream | Raw bytes | Packed bytes | Reduction |
|---|---:|---:|---:|
| title | 31,415,250 | 17,418,836 | 44.55% |
| text | 1,159,953,217 | 1,065,560,578 | 8.14% |
| All components, including framing | 1,191,369,357 | 1,082,981,764 | 9.10% |

Text accounts for 97.36% of the raw bytes. Payloads average approximately 4.06 bits/value for title and 6.97 for text, excluding headers, directories, and footers. No HN block needed the eight-bit fallback; seven text blocks were constant. Headers and directories can make other inputs larger than raw storage.

The diagnostic index retains both copies; its actual total size has not decreased. The savings above describe replacing the raw component with the packed component.

## Query measurements

Same full HN index, 28,737,557 documents, ten segments, serial execution, identical postings and pruning metadata. Single-term queries use `database`; AND/OR queries combine `postgres` and `database`. Each mode received five warmups and twenty alternating measured runs. All measured shared reads were zero.

| Query | Total shared hits, raw → packed | Norm-stream hits, raw → packed | Median ms, raw → packed |
|---|---:|---:|---:|
| title single | 428 → 487 | 62 → 121 | 0.7320 → 0.7665 |
| title AND | 492 → 561 | 72 → 141 | 1.0055 → 1.0385 |
| title OR | 492 → 561 | 72 → 141 | 1.0710 → 1.1120 |
| text single | 542 → 592 | 77 → 127 | 1.2505 → 1.3020 |
| text AND | 635 → 695 | 90 → 150 | 1.3390 → 1.3995 |
| text OR | 635 → 695 | 90 → 150 | 1.6365 → 1.7360 |

Both modes additionally touch 30 global fieldnorm metadata buffers, with **zero global per-document fieldnorm reads**. Logical norm lookup counts are identical between modes. Thus the locality benefit of posting-local norms survives compression.

Extra directory/footer reads contribute to the regression. Physical PostgreSQL block-map setup also costs 81 hits for the packed components versus 40 for raw components. This allocation/layout difference is a confound: the observed regression cannot all be attributed to bit-unpacking CPU. Clipping read-ahead removed 14–40 hits per query relative to the initial packed reader, but did not eliminate this overhead.

Final timings were collected after an independent compilation on the machine finished. Earlier contended measurements are excluded. These small warm serial measurements do not establish behavior under cold I/O or concurrency.

A next experiment could place compression offsets/descriptors in already-read postings metadata to avoid the separate directory cost. That change is not implemented here.

## Validation

- Native release suite: 1,503 passed, zero failed, 20 ignored, two pre-existing release argument-check tests filtered.
- Packed tests cover widths 0–8, partial and multiple blocks, arbitrary write chunks, reverse/random access, clones, clipped windows, invalid inputs, and bounds.
- Integration coverage includes scores, seeks/resets, merges/deletions, and no norm reads for unscored or fully pruned queries.
- All six SQL cases match exhaustive raw-versus-packed score digests and top scores. Unscored and count queries perform zero norm lookups.

```sh
cargo test --release --features subblock-pruning,bitpacked-posting-norms --lib -- --skip query::disjunction::tests::test_arg_check1 --skip query::disjunction::tests::test_arg_check2
uv run --with 'psycopg[binary]' python compare-sql.py
uv run --with 'psycopg[binary]' python verify-scores.py
```

Run SQL scripts from this directory. The comparison writes `packed-comparison-results.json`, which the correctness script consumes; the archived original is compressed alongside this report.

## Persistent instance and provenance

Connection: `postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable`.

Packed mode is enabled by default. Switch in the same SQL session before running EXPLAIN:

```sql
SELECT diagnostic_posting_norms(true);
SELECT diagnostic_packed_posting_norms(false); -- raw
SELECT diagnostic_packed_posting_norms(true);  -- packed
```

All ten segments have both norm components. The diagnostic converter added packed streams from existing raw bytes without rebuilding postings or changing segments. It is an experimental helper, not a production migration mechanism. Do not restore the previous binary or index as cleanup; the user is inspecting this instance. Port 29949 is the untouched baseline.

Installed binary SHA256: `eb0cfec0a42e6203218f53f38d1b2a9d901b85ac4a3cc0c01a924ff2c819cce3`.

Binary archive: `/Users/mingying/benchmarker/bm25-io-investigation-20260921/bitpacked-norms/pg_search-clipped.dylib`.

`pg-search-packed-adapter.patch.gz` captures the PG adapter changes against the pre-experiment diagnostic source. That source is under `/Users/mingying/benchmarker/bm25-io-investigation-20260921/source`; its isolated Tantivy dependency is under the sibling `posting-norms/tantivy`. The latter preserves its older unrelated vector implementation. Build logs and original artifacts remain in the sibling `bitpacked-norms` directory.

`results.json` contains the compact final comparison. Full final and initial run data, final plans, storage/conversion statistics, exhaustive correctness results, native test output, and SQL scripts are saved alongside this report.
