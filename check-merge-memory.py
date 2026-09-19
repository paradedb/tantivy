#!/usr/bin/env python3
"""Check the streaming merge's incremental heap against its stacked control.

Run the 12-case merge_memory matrix in fresh processes, then pass its JSONL here.
The bound is deliberately relative: it tolerates allocator/background overhead
while rejecting the original documents-times-positions amplification.
"""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
cases = {(r['docs'], r['repetitions'], r['mode']): r for r in rows}
assert len(cases) == len(rows) == 12, 'Expected twelve distinct matrix cases'
failures = []
for docs in (40000, 80000):
    for repetitions in (8, 128, 512):
        control = cases[(docs, repetitions, 'disjoint')]
        shuffled = cases[(docs, repetitions, 'interleaved')]
        assert control['correctness'] == shuffled['correctness'] == 'pass'
        assert control['input_segments'] == shuffled['input_segments']
        actual = shuffled['merge_peak_above_baseline_bytes']
        limit = 2 * control['merge_peak_above_baseline_bytes'] + 1024 * 1024
        if actual > limit:
            failures.append(f'{docs=} {repetitions=}: {actual} > {limit} bytes')
if failures:
    raise SystemExit('\n'.join(failures))
print('PASS: all six shuffled merges stay within 2x stacked heap growth + 1 MiB')
