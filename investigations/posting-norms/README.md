# Posting-local norms audit

Branch: `codex/lazy-posting-norms`, based on saved pruning prototype `codex/subblock-pruning-prototype` (`dc193cf838a88cc06db4ef86915adebd463e0ae2`).

The diagnostic database on port 29950 has the full rebuilt `public.hn_items_idx` committed and the audited binary installed. Leave both in place for direct user testing. Baseline port 29949 is unchanged.

## Controlled SQL comparison

Same full index, ten segments, serial execution, identical subblock pruning enabled in both modes. Switch only `diagnostic_posting_norms(false/true)`. Twenty measured alternating warm runs per mode after five warmups; all measured shared reads were zero. Shared hits count buffer acquisitions, not distinct pages or physical disk reads.

Single-term query is `database`; AND/OR combine `postgres` and `database` using explicit SQL AND/OR. Each selects `id, title, by, score`, orders by `pdb.score(id) DESC`, and limits to 10. Plans verify the intended term/must/should structure.

| Query | Norm hits global → local | Total hits global → local | Median execution ms global → local | Logical norm lookups (both) |
|---|---:|---:|---:|---:|
| title-single | 716 → 92 | 1052 → 428 | 1.192 → 0.694 | 2038 |
| title-and | 1655 → 102 | 2045 → 492 | 1.944 → 0.972 | 2380 |
| title-or | 2384 → 102 | 2774 → 492 | 2.848 → 1.023 | 7297 |
| text-single | 3238 → 107 | 3673 → 542 | 6.383 → 1.200 | 25890 |
| text-and | 3522 → 120 | 4037 → 635 | 8.120 → 1.308 | 34555 |
| text-or | 2645 → 120 | 3160 → 635 | 4.766 → 1.632 | 14166 |

Local norm hits include **both** retained global-fieldnorm metadata and the new posting-norm stream. Every local query has zero `fieldnorm/read_byte` calls. Remaining global-fieldnorm hits are 20 block-map plus 10 file reads; posting-norm hits are 40 block-map plus 22–50 file reads. File-read buckets include file/composite metadata as well as term data.

## Audit and changes

- Norms are the original quantized bytes copied in posting order, addressed by `(doc_freq - remaining_docs) + in_block_offset`. Seek/reset and merge retain this correspondence. Top-K pruning and candidate membership do not change.
- Norm files open only on first scoring lookup. Each active term retains at most 8 KiB of norm payload; forward gaps can skip chunks. Clones retain the current owned buffer. A rejected block does not read norms. The first surviving candidate can prefetch norms for neighboring candidates that are later rejected.
- Fixed disabled scoring: detach the term-local reader when `TermWeight.scoring_enabled` is false, preserving the existing constant-norm path. SQL LIMIT without score ordering and COUNT now perform zero local norm lookups for all six shapes.
- Added `BufferedFileSlice::read_byte`: cache hits return a byte directly without creating/dropping an OwnedBytes slice and its reference-counted owner. Cache misses reuse existing buffering and bounds checks. This is a CPU-path improvement; the matrix does not isolate a causal latency gain from this change alone.
- Audited union scoring through TermScorer and the batched intersection leader through posting-ordinal access. Phrase-specific scorers still use global fieldnorms and were outside this experiment.
- Regenerated the initial experimental format-9 fixture and added a scored reopen regression test. The old generated fixture predated correct registration of the pnorm component for retention.

## Correctness

Full native suite: 1,500 passed, zero failures, 20 ignored, two pre-existing release-only disjunction argument-assertion tests filtered. Additional format-9 reopen test passed. Targeted norm tests cover lazy reads, cached reads/clones, bounds errors, 8 KiB boundaries and tail buffers, single/AND/OR scoring, disabled scoring, all-block rejection, seek/reset, deletion, and merge across 26,000 documents.

Exhaustive SQL compared sorted `(id, float32 score)` streams via SHA-256 in both modes, and checked every returned top-10 score against the exhaustive reference. Match counts: title-single: 10,980, title-and: 143, title-or: 13,074, text-single: 140,613, text-and: 6,187, text-or: 166,965. All comparisons passed. The score reference matrix was collected before the CPU-only audit fixes, so this also checks that the fixes preserve scored results.

## Costs and remaining opportunities

- The full rebuilt index has 1,191,369,357 bytes (1.11 GiB) of posting-norm streams, compared with 57,475,554 bytes of global fieldnorm streams that remain for fallback. This is one byte per term/document posting, not one byte per document. It also adds an 18-byte term header (magic + offset) for each frequency-bearing term in each segment. The full index relation is 13,220,569,088 bytes; its difference from the original index is not a controlled measurement of this feature alone.
- Build/merge buffers one norm byte per posting for the current term, then writes its stream. The vector capacity is reused across terms. This adds build memory and write amplification.
- 8 KiB buffering is in logical norm bytes, not aligned PostgreSQL pages. A chunk can span multiple physical pages; postings and norm files also have different metadata. Exact page-count equality is not expected.
- Compression and compact term-offset framing could reduce storage; page-aware buffering could reduce boundary rereads. Removing the remaining global-fieldnorm metadata setup would require lazy reader setup. None was necessary to collapse scattered reads in this matrix, and none is claimed as implemented.
- This remains a feature-gated format prototype with diagnostic per-lookup counters. Broad cold-cache, high-concurrency, common-term, phrase, and production storage-format evaluation remain outside these six query shapes.

## Reproduce

```sh
psql 'postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable'
```

```sql
SELECT diagnostic_posting_norms(true); -- default in a fresh backend
EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, TIMING OFF)
SELECT id, title, by, score FROM hn_items
WHERE title === 'database' ORDER BY pdb.score(id) DESC LIMIT 10;
```

`audit-matrix.py audit-after` writes fresh measurements without rebuilding. `audit-correctness.py` validates the matrix. Scripts use psycopg and can run through `uv run --with 'psycopg[binary]' python ...`. The compressed PostgreSQL adapter patch (`pg-search-adapter.patch.gz`) is relative to the previously instrumented diagnostic source; its Cargo paths point to the isolated Tantivy copy and must be adjusted elsewhere. Production dependencies are unchanged by default; serialization requires `posting-norms` and this run also uses `subblock-pruning`. Raw plans/runs are retained in compressed JSON files.
