# Packed norm directories embedded in postings

2026-09-21. Branch `codex/postings-norm-descriptors`, based on `97a803f15`.

Embedding directory entries removes 12–30 norm-stream buffer hits per query, but this format is not a net performance win over raw posting-local norms. The large remaining cost is opening the separate compressed file's physical block map. Rewriting the postings introduced another physical-layout confound, measured separately below.

## Change

For posting lists with at least 128 documents, the diagnostic converter copies their packed-norm offset directory into the existing postings header. It preserves the exact postings, pruning summaries, raw norm offsets, compressed payload, document IDs, segments, and BM25 parameters. Lists below 128 keep the previous format.

The new header contains a ten-byte marker, logical field norm count, payload boundary, offset width, directory length, and directory entries. Opening postings reads those entries as part of the existing postings read. On the first norm lookup, the reader opens the compressed component and reads only the needed payload. It bypasses the packed footer and on-disk directory. Width/minimum bytes remain with the compressed blocks. Random access remains O(1), and values remain the original quantized fieldnorm IDs.

This is a post-build conversion prototype. Normal index serialization still writes the prior format; a REINDEX does not automatically embed directories. New readers support both formats. The old binary cannot read the converted postings; switch to the saved original components before any binary rollback.

## Controlled comparison

Full HN index: 28,737,557 documents, ten unchanged segments. Serial execution. Five warmups and thirty measured iterations per mode, rotating and reversing mode order. Six queries: title/text `database`, AND of `postgres` and `database`, and OR of those terms. Every measured shared read count is zero.

Five modes distinguish the directory optimization from rewriting the physical files: original raw; original packed; rewritten raw; rewritten packed with external directory; rewritten packed with embedded directory. The three rewritten modes use exactly the same files and page layout.

| Query | Rewritten raw hits | Rewritten external-directory packed hits | Embedded-directory packed hits | External → embedded median ms |
|---|---:|---:|---:|---:|
| title single | 512 | 571 | 551 | 0.8065 → 0.8075 |
| title AND | 576 | 645 | 615 | 1.0755 → 1.0485 |
| title OR | 576 | 645 | 615 | 1.1240 → 1.0975 |
| text single | 627 | 677 | 665 | 1.3220 → 1.3285 |
| text AND | 720 | 780 | 758 | 1.4505 → 1.4335 |
| text OR | 720 | 780 | 758 | 1.7320 → 1.7150 |

The directory optimization lowers hits consistently but does not consistently improve timing. Treat the small warm timing differences as descriptive; these runs do not establish cold-I/O or concurrency behavior.

Original raw title/database remains 428 hits, and original packed remains 487. Rewriting the postings increased the title query's postings block-map hits from 70 to 150 and term-dictionary map hits from 20 to 27; other page-boundary changes offset part of this, for an 84-hit net increase in raw mode (428 → 512). This is a physical-layout effect, not the cost of decoding the embedded directory. It is why the same-layout controls are necessary.

## Where norm hits remain

| Query | Raw norm-file reads, excluding block map | Embedded packed norm-file reads, excluding block map | Raw / packed block-map hits |
|---|---:|---:|---:|
| title single | 22 | 20 | 40 / 81 |
| title AND / OR | 32 | 30 | 40 / 81 |
| text single | 37 | 34 | 40 / 81 |
| text AND / OR | 50 | 47 | 40 / 81 |

The file-read counts include component framing as well as payload. Embedding removes the separate packed footer/directory accesses, but opening `.bpnorm` still loads its physical block map. Compression saves only two or three remaining norm-file page hits for these queries. The larger map more than cancels those savings. The raw and packed maps refer to different physical allocations; the 41-hit difference is not inherent to bitpacking.

Both modes also have 30 global fieldnorm metadata hits and zero global per-document norm reads. Logical norm lookup counts and selected rows/scores match in every mode.

## Storage and validation

Embedding adds 46,491,343 bytes to postings and 2,792 bytes to the rebuilt term dictionaries. The compressed payload is unchanged. The prototype retains the external directories for fallback and A/B testing, so this is duplicated metadata. Relative to replacing raw streams with the packed stream plus this extra metadata, the previous 108,387,593-byte saving becomes 61,893,458 bytes (about 5.2% of raw norm-stream size). Actual live storage is larger because both original and rewritten components and both norm streams are retained.

- Native release suite: 1,504 passed, zero failed, 20 ignored, two pre-existing release argument-check tests filtered.
- New tests cover all bit widths, shared block boundaries, partial tails, reverse/random access, cloning, malformed headers, idempotent rewriting, and confirm that the embedded path never opens the packed footer.
- A 600-row SQL smoke test preserves every score and verifies switching old/new components in both directions.
- All six HN queries pass exhaustive raw-versus-embedded score digests and top-score checks. Unscored/count queries have zero logical norm reads.
- The full HN conversion took roughly 65 seconds; no postings or documents were reindexed.

## Reproduction and persistent instance

Connection: `postgresql://mingying@127.0.0.1:29950/hn_benchmark?sslmode=disable`.

All ten segments currently select the rewritten components. New backends default to embedded packed mode. In the same session before EXPLAIN:

```sql
SELECT diagnostic_posting_norms(true);
SELECT diagnostic_packed_posting_norms(true);
SELECT diagnostic_embedded_norm_directory(true);  -- embedded directory
SELECT diagnostic_embedded_norm_directory(false); -- external directory
SELECT diagnostic_packed_posting_norms(false);    -- raw norms
```

Raw mode on the rewritten index has 512 title/database hits; use the saved original components to reproduce the original 428-hit baseline. `conversion.json` records both sets of component references, and `compare.py` demonstrates switching them with the owner-only diagnostic function. These references are specific to this index and must not be used on another database or after rebuilding it.

```sh
cargo test --release --features subblock-pruning,bitpacked-posting-norms --lib -- --skip query::disjunction::tests::test_arg_check1 --skip query::disjunction::tests::test_arg_check2
uv run --with 'psycopg[binary]' python compare.py
uv run --with 'psycopg[binary]' python verify-scores.py
```

The comparison selects the rewritten components when finished. The correctness script consumes its `comparison.json`; the original captured run data is archived as `comparison.json.gz`. Do not rerun `convert.py` as routine verification: it creates new physical copies.

Installed binary SHA256: `12cba698ff92d3839da5e5a8ea3e54660b20a2d3d473639f75371fb2d9590f2d`.

Archive: `/Users/mingying/benchmarker/bm25-io-investigation-20260921/norm-descriptors/pg_search-descriptors.dylib`. The PG adapter diff against the previous experiment is `pg-adapter.patch.gz`. Build logs and full artifacts remain in that directory. Port 29949 was not modified. The temporary change to the diagnostic instance's WAL-size setting was reset after conversion.

## Next experiment suggested by the result

To remove the dominant remaining cost, test placing compressed norm payloads with the already-read postings, not just their directories. That avoids opening a separate file and block map, but makes compressed norm bytes eager when postings are read. A size threshold could limit overfetch for very long lists. It needs a same-layout control and explicit measurement of extra postings bytes, map costs, unscored queries, and lists pruned before scoring. This variant is not implemented here.
