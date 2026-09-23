# Buffered lazy postings

This change builds on [MaxScore](https://github.com/paradedb/tantivy/pull/243). For a postings
slice larger than 32 KiB, the cursor reads its header and skip directory, then
fetches payload only for blocks execution visits, using 32 KiB read-ahead. Lists
at most 32 KiB (and lists below one 128-posting block) retain a single eager read.
This policy is automatic, with no environment controls or index format change.

For example, WAND may skip from postings block 0 to block 512. Previously,
opening the list had already read every compressed payload byte. The lazy cursor
reads metadata and the payload around the two visited blocks; skipped bytes
outside the read-ahead windows need not be requested from storage. Physical page
savings depend on layout and the directory implementation. A mmap directory does
not pay the same copy/buffer costs as PostgreSQL.

The cursor retains its current OwnedBytes buffer and decodes directly from it.
There is no per-block slice-owner clone. This reduced larger full-scan CPU ratios
from roughly 1.24x eager to about 1.01x in the copy-backed microbenchmark.

## Buffer-size sweep

```sh
cargo test --release --features quickwit --lib benchmark_lazy_read_cutovers -- --ignored --nocapture --test-threads=1
```

The benchmark varies 4,096–1,048,576 postings, visiting
every 1/8/64/512th compressed block, and eager/8/32/128/512 KiB read-ahead.
Each FileHandle read copies its requested bytes, records calls/bytes, and models
8 KiB page acquisitions. **Those modeled page counts are not EXPLAIN BUFFERS.**
Five rotated 8 ms rounds are reported as median ns/query, including cursor open.
The data and layout are synthetic; hot mmap reads and real storage have different
costs. Large scans may still acquire a page more than once at buffer boundaries.

For the 1,048,576-posting list, 32 KiB reads took 1.01x eager on a full scan and
0.73x eager when visiting every 512th block, requesting about 43% of payload/list
bytes. An 8 KiB buffer reduced skipped-query bytes further but increased modeled
full-scan page acquisitions to about 2x eager; 128/512 KiB increasingly read past
skipped blocks. 32 KiB is the measured compromise, not a guarantee of fewer
buffers for every query. Smaller lists stay eager to avoid extra header/read calls.

Correctness coverage compares eager and lazy scans/seeks/clones for empty lists,
127/128/129-posting boundaries, dense and sparse docs, frequencies, Basic/WithFreqs/
positions schemas, and several buffer sizes. Runtime read failures propagate at
open or follow the existing cursor's infallible-iteration error behavior later.
