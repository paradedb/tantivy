# `term_set_queries` bench captures

Two live capture files live alongside the bench source:

| File | Tier | Contents |
|---|---|---|
| `term_set_queries.full-tier.txt` | full | `TERM_SET_BENCH_TIER=full` against current branch HEAD. 18 (N × kind × sort) groups, ~360 cells. |
| `term_set_queries.threshold-tier.txt` | threshold | `TERM_SET_BENCH_TIER=threshold` against current HEAD. LowFk crossover-region cells (K/N ∈ [0.002, 0.01]). |

## Refresh

```bash
TERM_SET_BENCH_TIER=full cargo bench --bench term_set_queries 2>&1 \
    | tee benches/term_set_queries.full-tier.txt           # ~30 minutes
TERM_SET_BENCH_TIER=threshold cargo bench --bench term_set_queries 2>&1 \
    | tee benches/term_set_queries.threshold-tier.txt      # ~1 minute
```

Re-run when the gallop algorithm, smart-seek, or `TermSetStrategyConfig::default()` changes meaningfully. Don't refresh on comment-only or test-only changes.
