#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CARGO_PROFILE_RELEASE_DEBUG=1
cargo build --release --example merge_memory --no-default-features \
  --features mmap,columnar-zstd-compression,lz4-compression,paradedb,quickwit,stemmer,stopwords
target_dir=${CARGO_TARGET_DIR:-target}
for n in 10000 20000; do
  for mode in disjoint interleaved; do
    for reps in 8 128 512; do
      "$target_dir/release/examples/merge_memory" "$mode" "$n" "$reps"
    done
  done
done
