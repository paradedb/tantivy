#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

packages="-p fht -p sign-plane -p grid-plane -p quant-model -p cascade"

# shellcheck disable=SC2086 # the package list is intentionally word-split.
cargo +nightly fmt --check $packages
# shellcheck disable=SC2086
cargo clippy $packages -- -D warnings
# shellcheck disable=SC2086
cargo test $packages --release
