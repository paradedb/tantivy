# HN postings and Block-Max experiments

This branch records independently switchable research prototypes, not a production rollout. It builds on the buffered lazy-postings experiment. Runtime controls in `src/postings/mod.rs` default to baseline behavior. Optional `postings-diagnostics` counters support the matching pg_search instrumentation.

Included mechanisms:

- Lazy position payloads, exact phrase anchors, and adaptive membership-first conjunctions.
- Candidate term-frequency and phrase-score bounds; experimental deferred seeks and segment/suffix maxima.
- A stronger completed-prefix top-K cutoff, with integration-side serial-only eligibility.
- A provider for exact norm bytes in posting order, with native fieldnorm fallback.
- Dense-term score suppression and deferred zero-weight reader creation for eligible pure-OR top-K. Matching is preserved; ranking intentionally changes when the density policy is enabled.

PG17 HN measurements show roughly 4–7x fewer whole-query buffer loads for selective AND, 8.36x for one nonempty phrase, and 3.2x for sparse technical OR with exact norm locality. Common phrases remain a negative control. The dense-term policy plus norms saves 4.41–8.94x reads on five public OR strings, with changed scores/results relative to full BM25. Bounds and skipped compressed blocks often fail to remove physical pages; this branch does not establish a general tenfold exact ranked-OR improvement.

The norm provider is an experimental integration boundary, not a versioned production codec. Stored block-max pairs have an existing conservativeness caveat when global average document length changes; segment/suffix experiments must not be promoted without resolving it. The global-prefix experiment can alter float addition order by one ULP, and parallel MVCC/retry completeness is not established. All such controls remain opt-in.

Recorded validation: Block-WAND filter 16 passed / 2 preexisting ignored; full term-query filter 28 passed / 1 preexisting ignored; focused dense opening 5 passed; V5 phrase filter 31 passed / 1 preexisting ignored. Earlier lazy-position, exact-anchor, candidate-bound, norm-provider, and generic-intersection filters also passed. PostgreSQL result/score comparisons and short benchmark measurements are retained with the corresponding ParadeDB experiment.
