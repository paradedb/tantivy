# `term_set_queries` bench captures over time

These files are the on-disk record of each round of bench measurements that
informed a decision in paradedb/paradedb#4895. The naming convention encodes
which point in the branch history each capture predates, so a future reader
inspecting the `benches/` directory can tell at a glance which numbers are
"live for the current code" vs "historical evidence behind a decision."

## Live captures (refreshed when the gallop algorithm changes)

| File | Tier | What it contains |
|---|---|---|
| `term_set_queries.full-tier.txt` | full | Most recent `TERM_SET_BENCH_TIER=full` run against the current branch HEAD. 18 (N × kind × sort) groups, ~360 cells. Refresh after any change to the gallop strategy's algorithm. |
| `term_set_queries.threshold-tier.txt` | threshold | Most recent `TERM_SET_BENCH_TIER=threshold` run against current HEAD. 16 LowFk crossover-region cells (K/N ∈ [0.001, 0.01]). Refresh on the same trigger as full-tier. |

## Historical captures (audit trail; do not delete)

| File | Captured against | Decision it justified |
|---|---|---|
| `term_set_queries.full-tier.original.txt` | Pre-threshold-tier merge (Step 6 only) | Original Step 6 capture before the threshold tier was added. Kept as the unmerged base; the live full-tier file at the time of capture appended the threshold-tier output below a separator block. |
| `term_set_queries.full-tier.pre-followup-d.txt` | Pre-Follow-up-D wire-in (parent of commit `953df2cb`) | Step 6's threshold-range identification (K/N ∈ [0.01, 0.05]) and the `gallop_max_density = 1/200` default in Step 7. Numbers reflect plain binary-search galloping, NOT true galloping. |
| `term_set_queries.threshold-tier.pre-followup-d.txt` | Same as above | Step 7's targeted LowFk crossover capture that pinned the `1/200` default at the K/N = 0.005 data point with worst-case 1.74× margin at N=10M. |

## When to refresh

Re-run the live captures when any of these change:

- The gallop algorithm itself (per-term search, range emission, cursor advance).
- The smart-seek implementation on `TermSetDocSet`.
- `TermSetStrategyConfig::default()` if the change affects the matrix bench's
  `cfg_force_gallop()` / `cfg_force_linear()` definitions.
- The corpus build path in `build_corpus()` (e.g., a different schema option
  that affects column layout).

Don't refresh on:

- Code reorganization that doesn't change runtime behavior.
- Comment-only changes.
- Test-only changes outside `benches/`.

## How to refresh

```bash
cd ~/Documents/Workplace/tantivy
TERM_SET_BENCH_TIER=full cargo bench --bench term_set_queries 2>&1 \
    | tee benches/term_set_queries.full-tier.txt
TERM_SET_BENCH_TIER=threshold cargo bench --bench term_set_queries 2>&1 \
    | tee benches/term_set_queries.threshold-tier.txt
```

Full tier ~30 minutes wall-clock (six N=50M corpus builds dominate); threshold
tier ~1 minute. After refreshing, commit the live capture files. If the
refresh predates a meaningful change to the gallop algorithm (i.e., a future
follow-up), rename the prior live captures to a labeled archival name
(`*-tier.<short-description>.txt` or `*-tier.before-<change>.txt`) before
running the new bench, and add an entry to the historical-captures table
above.
