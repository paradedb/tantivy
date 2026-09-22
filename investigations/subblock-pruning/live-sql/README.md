# Live PostgreSQL fieldnorm pruning experiment

Measured on September 21, 2026, against the local diagnostic PostgreSQL 18 instance on port 29950. Port 29949 was not changed. These results come from actual PostgreSQL execution and ReadBuffer accounting.

```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, TIMING OFF)
SELECT id, title, by, score
FROM hn_items
WHERE title === 'database'
ORDER BY pdb.score(id) DESC
LIMIT 10;
```

## Controlled result

Both modes used the same temporary index on `(id, title)`, with 10 segments and all 28,737,557 table rows. Index size was 910,811,136 bytes. Query execution was serial, matching the original plan. Pruning was alternated off/on within one backend; five warmups per mode were discarded, followed by 20 measured runs per mode.

| Metric | Pruning off | Pruning on | Reduction |
|---|---:|---:|---:|
| Fieldnorm shared buffer hits | 1,721 | 745 | 56.7% |
| Total shared buffer hits | 1,957 | 981 | 49.9% |
| Logical fieldnorm value reads | 5,560 | 2,135 | 61.6% |
| Median execution time | 1.9495 ms | 1.066 ms | 45.3% |
| Distinct fieldnorm value pages | 1,607 | 715 | 55.5% |

Buffer counts were identical across all 20 measured runs in each mode; shared reads were zero. Timings are warm, instrumented local measurements, not concurrency benchmarks.

All 976 eliminated buffer hits were in the fieldnorm component. Every other component was unchanged: fast 41, heap 10, postings 53, metadata 31, positions 41, term dictionary 60. Fieldnorm hits include 30 fixed footer/block-map acquisitions in both modes; fieldnorm value acquisitions fell from 1,691 to 715.

Returned IDs, ordering, projected values, and BM25 scores were identical off/on. A materialized CTE scored all 10,980 matches without top-K pruning: both modes returned the same top-10 score sequence as that exhaustive result, and each returned row's score matched. This validation permits equivalent choices among ties at the cutoff.

## What was tested

The prototype writes a conservative pair of bytes for each 16-posting group: the minimum fieldnorm ID and an upper encoding of maximum term frequency. It includes the final partial block. Query-time bounds can reject a group before fetching candidate fieldnorms; whole-block bounds can also be computed from these summaries without scanning tail norms.

The diagnostic build exposes a temporary SQL setter for a backend-local pruning flag. Both modes parse the same format and load its metadata. Off uses the legacy block-bound path and the existing norm-zero candidate TF filter; on also uses the new summaries. Thus this experiment isolates pruning savings, but does not measure the new format's storage or metadata-read overhead against an otherwise identical index written in the old format.

The original all-column index consistently used 1,539 fieldnorm hits and 1,864 total hits before the experiment, with the new binary reading the old format, and again after restoration. Rebuilding changes segment/doc layout, so **1,721 → 745**, rather than **1,539 → 745**, is the controlled effect of enabling pruning.

This is a prototype, not a format ready for general deployment. Separate standalone text experiments in `text.json` regressed with these independent extrema; the positive live result here is specifically for the requested title/database query.

## Artifacts and restoration

- `sql-toggle.json`: all live off/on plans, rows, timing samples, and exhaustive validation.
- `sql-toggle-off-plan.txt`, `sql-toggle-on-plan.txt`: readable plans for the exact SQL above.
- `sql-original.json`, `sql-original-new-binary.json`, `sql-original-restored.json`: original-index controls.
- `measure-sql.py`: measurement and correctness script; temporary index replacement is inside `transaction(force_rollback=True)`.
- `setup.sql`: temporary SQL wrapper definition, needed only with the prototype binary loaded.
- `tantivy/`: isolated diagnostic Tantivy source, based on the previously pinned dependency plus prototype postings changes and the query-time switch.
- `prototype-pg-search/`: dependency overrides, lockfile, and wrapper source used in the diagnostic build.
- `pg_search.dylib`, `pg-build.log`: built prototype and build log.
- `before/`: original diagnostic binary, manifests, and wrapper source.
- `provenance.json`: source revisions and binary hashes, including restoration checks.

ParadeDB rejects two BM25 indexes on one table. The measurement therefore temporarily dropped the original index and created the title index inside one transaction, ran both modes, and rolled back. The rollback restored the original 9,185 MB index and removed the experimental index. The temporary SQL function was dropped, original diagnostic binary and source manifests restored, and the original query rechecked. Both servers are running; their installed binary hashes match those captured before the experiment.

To repeat, the diagnostic server must first load the saved prototype binary and register `setup.sql`; then run `uv run --with 'psycopg[binary]' python measure-sql.py toggle`. The script holds a table lock during the temporary replacement and always rolls it back. Restore the original diagnostic binary and remove the wrapper afterward. The root Tantivy worktree retains the prototype source changes, with `subblock-pruning` disabled by default.
