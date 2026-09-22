# Inline compressed posting-local norms

Experiment date: 2026-09-21. Branch `codex/inline-packed-posting-norms`, based on `4cc68206c`.

## Format and implementation

Each term's quantized fieldnorm IDs are stored with its postings. The ten-byte format marker replaces the previous marker, and a variable-length encoded payload length replaces the old eight-byte sidecar offset. The document frequency supplies the number of norms.

The encoder chooses the smallest representation:

- Raw: one mode byte followed by the original norm bytes.
- One fixed-width frame: bit width, minimum ID, and packed differences from that shared minimum. Width zero represents a constant list.
- Frames of up to 128 values: a mode byte, four-byte offsets for every block after the first, then width/minimum/payload frames. Block count comes from document frequency; the first offset is implicitly zero. No packed-file footer or external directory is required.

This is lossless compression of the existing quantized IDs, not exact document lengths. Access is O(1), with no successive-value delta chain. Raw fallback caps codec size, and removing the old offset helps short lists even when bitpacking itself does not save bytes.

`inline-posting-norms` enables actual index serialization in this format. It implies `posting-norms` for gathering the values and suppresses `.pnorm` and `.bpnorm` creation. The global `.fieldnorm` component remains. The writer emits the norm header after collecting the tail's norms, without an additional copy of the postings body.

Scored reads load the compressed norm payload with postings and decode individual values on demand. Reads requesting only document IDs probe the header, skip the norm payload, and retain a lazy file slice. If such a caller later requests a norm, it loads that term's norm payload once. Both paths share the existing postings file handle and avoid opening a norm sidecar or its physical block map.

The converter rewrote the HN postings and term dictionaries while preserving all ten segments, document IDs, TFs, pruning metadata, raw norm values, and global scoring statistics. Native serialization and the converter produce the same format. The old components and norm streams are retained for A/B tests.

## Storage

| Representation | Raw posting-local format | Inline format | Reduction |
|---|---:|---:|---:|
| Title norms including term headers | 55,239,798 B | 33,335,083 B | 39.65% |
| Text norms including term headers | 1,326,020,875 B | 1,127,605,661 B | 14.96% |
| All norms including term headers | 1,381,260,673 B | 1,160,940,744 B | 15.95% |
| All active index-component data | 6,779,804,623 B | 6,559,657,814 B | 3.25% |

The net saving is **220,146,809 bytes (about 210 MiB)**. Postings grow by 971,048,548 bytes and term dictionaries by 174,000 bytes, replacing 1,191,369,357 bytes of raw norm sidecars. This includes framing and dictionary effects; it is not a payload-only estimate.

The per-field norm figures include the previous 18-byte norm header for every title/text term. All terms in those two HN fields have term frequencies. The whole-index figures sum the actual serialized component lengths and exclude unused A/B copies. They do not include PostgreSQL page padding, physical block maps, WAL, or free space. The diagnostic relation retains original files, previous experiment copies, and norm sidecars; its physical `pg_relation_size` has not shrunk. New indexes written with this feature have no norm sidecars.

## Validation and reproduction

- Full native release suite: 1,507 passed, zero failed, 20 ignored, two pre-existing release argument-check tests filtered.
- Codec tests cover raw/constant/frame/block modes, widths, partial blocks, random/reverse access, malformed framing, cloning, and removal of the old offset.
- Integration tests compare scores, seek/reset in both frequency and document-only modes, deletions, and merges. They assert that newly serialized inline indexes have neither norm sidecar.
- A tracked file test proves that document-only reads skip the norm payload and that later scoring loads it once without opening a sidecar.
- The SQL smoke test creates an actual inline index with no norm sidecars and compares all 600 scores against global fieldnorm scoring.

```sh
cargo test --release --features subblock-pruning,bitpacked-posting-norms,inline-posting-norms --lib -- --skip query::disjunction::tests::test_arg_check1 --skip query::disjunction::tests::test_arg_check2
uv run --with 'psycopg[binary]' python compare.py
uv run --with 'psycopg[binary]' python verify-scores.py
uv run --with 'psycopg[binary]' python guards.py
python analyze-storage.py
```

`compare.py` and `guards.py` switch the original/inline component references and leave inline components selected. `verify-scores.py` compares inline scoring with global fieldnorm scoring on the inline index. Run scripts from this directory; archived full run data is compressed and the scripts create uncompressed output as needed. Do not rerun `convert.py` as ordinary verification: it writes new physical component copies.

## Persistent instance

Connection: `postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable`.

The inline components are selected by default after the comparison scripts finish. On inline postings, `diagnostic_packed_posting_norms(false)` does not select the old raw representation: that switch only controls sidecar-based formats. Select original components using the owner-only helper and the exact references in `conversion.json` to reproduce the original raw baseline. `compare.py` shows the calls. References are valid only for this index before rebuilding it.

`diagnostic_posting_norms(false)` still forces global norm scoring for correctness checks; `true` restores posting-local scoring. A pre-inline binary cannot read the new postings, so select original components before any binary rollback. Do not automatically restore this experiment: the user is inspecting the persistent instance.

Port 29949 and the other independent experiments were not modified. PG source changes relative to the preceding experiment are archived in `pg-adapter.patch.gz`; root Tantivy changes are on this branch. The isolated PG dependency preserves its older unrelated vector implementation.

## Final query results

Thirty warm measured iterations per mode after five warmups, serial execution, with mode order rotated/reversed. Same HN documents, term matches, posting IDs, frequencies, and pruning summaries. All measured shared reads are zero.

| Query | Raw → inline shared hits | Raw → inline median ms |
|---|---:|---:|
| title-single | 428 → 309 | 0.7570 → 0.6615 |
| title-and | 492 → 365 | 0.9990 → 0.8940 |
| title-or | 492 → 365 | 1.0580 → 0.9520 |
| text-single | 542 → 426 | 1.2425 → 1.1260 |
| text-and | 635 → 504 | 1.3640 → 1.2445 |
| text-or | 635 → 504 | 1.6260 → 1.5180 |

All six cases preserve exactly the selected rows/scores and logical norm lookup counts. Exhaustive score digests match global-fieldnorm scoring, and selected top scores match exhaustive top scores. Inline mode has zero norm-sidecar reads and zero global per-document norm reads. The 30 remaining global fieldnorm hits are metadata.

The rewritten physical layout is a confound for latency and total-buffer comparisons: in the title/database query, postings block-map hits fall 70 → 20 and dictionary reads fall 38 → 30. Of the 119-hit improvement, 58 are these allocation/boundary effects. The remaining accounting removes 62 raw norm-file hits and adds one postings-payload hit. Do not attribute the entire measured improvement to compression. Storage savings use actual component lengths and are unaffected by that allocation confound.

## Guard queries and limitations

Ten measured runs after two warmups, alternating original raw and inline components. Every guard returns identical rows/scores or counts. Unscored/count cases have zero logical norm lookups. The document-only path uses an up-to-20-byte prefix probe and skips the bulk norm payload; a shared PostgreSQL page can still contain norm bytes.

| Guard | Raw → inline shared hits | Raw → inline median ms |
|---|---:|---:|
| title-database-unscored | 103 → 98 | 0.1470 → 0.1445 |
| title-database-count | 13297 → 13240 | 7.0505 → 6.9160 |
| title-the-unscored | 106 → 103 | 0.1345 → 0.1355 |
| title-the-count | 17083 → 17039 | 32.5265 → 32.4210 |
| title-the-top10 | 738 → 577 | 4.2780 → 4.3425 |
| title-the-database-top10 | 775 → 603 | 1.6470 → 1.4785 |
| text-database-unscored | 102 → 97 | 0.1275 → 0.1270 |
| text-database-count | 16843 → 16800 | 18.6470 → 18.7210 |
| text-the-unscored | 360 → 356 | 0.2700 → 0.2730 |
| text-the-count | 11266 → 11225 | 204.5945 → 204.6355 |
| text-the-top10 | 12213 → 11840 | 93.3625 → 87.4620 |
| text-the-database-top10 | 10715 → 11969 | 12.3615 → 13.0130 |

The measured remaining tradeoff is a scored conjunction with a very common term: `text === 'the' AND text === 'database'` reads 11.7% more buffers and has a 5.3% higher median latency. Scored reads still load that long common term's full compressed norm payload, even though only the intersection is scored. A future reader could buffer long scored norm payloads lazily through the same postings file handle, preserving the storage format and avoiding a separate block map. That extension is not implemented here. The results are warm serial measurements, not cold-I/O or concurrency results.

This version meets the storage goal on the six target queries, while the common-term conjunction remains a real exception to unchanged query costs.

## Binary provenance

Final installed binary SHA256: `b5a70efdcb1bcef500ef51098225ba330bd689f3cdb52fde27154aa14c6d3a04`.

Archive: `/Users/mingying/benchmarker/bm25-io-investigation-20260921/inline-norms/pg_search-inline-final.dylib`.

Initial eager-reader results are retained separately for comparison. The final binary includes document-only payload skipping and the writer's temporary-copy removal. Original logs, binaries, and artifacts remain in the same `inline-norms` directory.
