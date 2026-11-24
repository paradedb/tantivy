#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

if [ "$(uname -m)" != "x86_64" ]; then
    echo "x86 asm expectations require an x86_64 host" >&2
    exit 2
fi

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/quant-asm}"
# Thin LTO defers loop vectorization to the final link, while this check asks
# rustc for pre-link assembly. Disable LTO here so the inspected artifact is
# the optimized machine code whose instructions the expectations validate.
export CARGO_PROFILE_RELEASE_LTO=false

# Prints the functions whose mangled names start with `tantivy_quant::$2`, each from its label to
# its `.cfi_endproc` or the next symbol label. Mach-O prefixes symbols with an extra `_`.
module_functions() {
    awk -v start="^_?_ZN13tantivy_quant$2[^:]*:" '
        in_fn && (/\.cfi_endproc/ || (/^[A-Za-z_$][^:]*:/ && !/^[Ll]/)) { in_fn = 0 }
        $0 ~ start { in_fn = 1 }
        in_fn { print }
    ' "$1"
}

cargo rustc -p tantivy-quant --release --lib -- --emit=asm

quant_asm=$(find "$CARGO_TARGET_DIR/release/deps" -name 'tantivy_quant-*.s' -print | head -n 1)

if [ -z "$quant_asm" ]; then
    echo "no tantivy_quant assembly was emitted" >&2
    exit 1
fi

sign_asm="$CARGO_TARGET_DIR/tantivy_quant-sign.s"
module_functions "$quant_asm" 4sign > "$sign_asm"
if [ ! -s "$sign_asm" ]; then
    echo "no tantivy_quant::sign functions found in $quant_asm" >&2
    exit 1
fi
if ! grep -Eq '\bpopcnt(l|q)?\b' "$sign_asm"; then
    echo "tantivy_quant::sign assembly does not contain a hardware popcnt instruction" >&2
    exit 1
fi

rotation_asm="$CARGO_TARGET_DIR/tantivy_quant-rotation.s"
module_functions "$quant_asm" 8rotation > "$rotation_asm"
if [ ! -s "$rotation_asm" ]; then
    echo "no tantivy_quant::rotation functions found in $quant_asm" >&2
    exit 1
fi
if ! grep -Eq '\b(v?addps|v?subps)\b' "$rotation_asm"; then
    echo "tantivy_quant::rotation assembly does not contain packed float butterfly instructions" >&2
    exit 1
fi
