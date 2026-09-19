# Reproduce sorted-merge memory amplification

The shuffled postings merge previously copied every document's positions for an entire term before sorting by mapped document ID. With common terms and many positions, its live heap grew with documents times positions, independently of the indexing writer budget. The streaming implementation keeps one postings cursor per segment and reuses a positions buffer.

From the repository root:

```sh
bash run-memory-matrix.sh > results.jsonl
python3 check-merge-memory.py results.jsonl
cargo test --release --test streaming_merge --no-default-features --features mmap,quickwit,paradedb
```

The example builds four batches of documents with disjoint or interleaved sort keys, disables automatic merging, and runs one foreground merge with a counting allocator. Flushes can produce more than four input segments. Every case checks document count, sorted keys, and positive/negative phrase matches. The memory check compares six shuffled cases with their stacked controls; it fails on the original implementation and passes with the fix. Figures are live Rust allocation growth during merging, not process RSS.

On Linux aarch64, 80,000 documents with 512 positions per common term required 177.98 MiB of additional live heap before the fix and 17.71 MiB afterward. The writer budget was 15,000,000 bytes in both cases. The original copied positions alone occupied 163,840,000 bytes, plus 4,194,304 bytes of tuple-vector capacity. The standalone matrix has twelve cases; streaming_merge covers both sort directions, all three postings record modes, missing/duplicate sort keys, and deleted documents.

A paired PostgreSQL 17 concurrent-reindex workload also reduced median peak backend anonymous RSS from 126.35 to 51.52 MiB over three repeated runs; a default-sort control stayed around 42.4 MiB. That is a synthetic forced-overlap case, not proof of any production OOM's cause. Whole-term serialization buffers, document maps, and exceptionally large individual documents still consume memory; this is not a total merge-memory cap.
