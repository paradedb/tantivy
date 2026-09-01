#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

if [ "$(uname -m)" != "x86_64" ]; then
    echo "x86 asm expectations require an x86_64 host" >&2
    exit 2
fi

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/quant-kernels-asm}"
# Thin LTO defers loop vectorization to the final link, while this check asks
# rustc for pre-link assembly. Disable LTO here so the inspected artifact is
# the optimized machine code whose instructions the expectations validate.
export CARGO_PROFILE_RELEASE_LTO=false

cargo rustc -p sign-plane --release --lib -- --emit=asm
cargo rustc -p fht --release --lib -- --emit=asm

sign_asm=$(find "$CARGO_TARGET_DIR/release/deps" -name 'sign_plane-*.s' -print | head -n 1)
fht_asm=$(find "$CARGO_TARGET_DIR/release/deps" -name 'fht-*.s' -print | head -n 1)

if [ -z "$sign_asm" ] || ! grep -Eq '\bpopcnt(l|q)?\b' "$sign_asm"; then
    echo "sign-plane assembly does not contain a hardware popcnt instruction" >&2
    exit 1
fi

if [ -z "$fht_asm" ] || ! grep -Eq '\b(v?addps|v?subps)\b' "$fht_asm"; then
    echo "fht assembly does not contain packed float butterfly instructions" >&2
    exit 1
fi
