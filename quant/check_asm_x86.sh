#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/quant-asm}"
# Thin LTO defers loop vectorization to the final link, while this check asks
# rustc for pre-link assembly. Disable LTO here so the inspected artifact is
# the optimized machine code whose instructions the expectations validate.
export CARGO_PROFILE_RELEASE_LTO=false

# On an x86_64 host the artifacts build natively; elsewhere they cross-compile
# to x86_64-unknown-linux-gnu, which only emits assembly and never links.
if [ "$(uname -m)" = "x86_64" ]; then
    target_args=""
    deps_dir="$CARGO_TARGET_DIR/release/deps"
    tuned_cpu="native"
else
    target_args="--target x86_64-unknown-linux-gnu"
    deps_dir="$CARGO_TARGET_DIR/x86_64-unknown-linux-gnu/release/deps"
    tuned_cpu="x86-64-v3"
fi

# Prints the functions whose mangled names start with `tantivy_quant::$2`, each from its label to
# its `.cfi_endproc` or the next symbol label. Mach-O prefixes symbols with an extra `_`.
module_functions() {
    awk -v start="^_?_ZN13tantivy_quant$2[^:]*:" '
        in_fn && (/\.cfi_endproc/ || (/^[A-Za-z_$][^:]*:/ && !/^[Ll]/)) { in_fn = 0 }
        $0 ~ start { in_fn = 1 }
        in_fn { print }
    ' "$1"
}

emit_asm() {
    rm -f "$deps_dir"/tantivy_quant-*.s
    # shellcheck disable=SC2086
    RUSTFLAGS="$1" cargo rustc -p tantivy-quant --release --lib $target_args -- --emit=asm
    find "$deps_dir" -name 'tantivy_quant-*.s' -print | head -n 1
}

# Pass 1: default flags, which is what shipped packages build with. The popcnt
# requirement holds only inside the runtime-dispatched variants.
quant_asm=$(emit_asm "")

if [ -z "$quant_asm" ]; then
    echo "no tantivy_quant assembly was emitted" >&2
    exit 1
fi

dispatched_asm="$CARGO_TARGET_DIR/tantivy_quant-sign-dispatched.s"
module_functions "$quant_asm" 4sign3x86 > "$dispatched_asm"
if [ ! -s "$dispatched_asm" ]; then
    echo "no tantivy_quant::sign::x86 dispatched variants found in $quant_asm" >&2
    exit 1
fi
if ! grep -Eq '\bpopcnt(l|q)?\b' "$dispatched_asm"; then
    echo "dispatched tantivy_quant::sign::x86 variants do not contain a hardware popcnt" >&2
    exit 1
fi

# Prints the vpopcntq count of the sign::x86 function named by mangled segment $1.
vpopcntq_count() {
    variant_asm="$CARGO_TARGET_DIR/tantivy_quant-sign-x86-$1.s"
    module_functions "$quant_asm" "4sign3x86$1" > "$variant_asm"
    if [ ! -s "$variant_asm" ]; then
        echo "no tantivy_quant::sign::x86 $1 found in $quant_asm" >&2
        exit 1
    fi
    grep -Ec '\bvpopcntq\b' "$variant_asm" || true
}

# The AVX-512 tier exists for vectorized VPOPCNTQ, so the batch and indexed variants must
# contain it. The single-row variant is reported, not required.
batch_count=$(vpopcntq_count 12batch_avx512)
indexed_count=$(vpopcntq_count 14indexed_avx512)
single_count=$(vpopcntq_count 13single_avx512)
echo "vpopcntq in sign::x86 AVX-512 variants: batch $batch_count, indexed $indexed_count, single $single_count"
if [ "$batch_count" -eq 0 ] || [ "$indexed_count" -eq 0 ]; then
    echo "tantivy_quant::sign::x86 batch_avx512 and indexed_avx512 must contain vpopcntq" >&2
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

# Pass 2: a tuned CPU (native on x86 hosts), where popcnt must appear in the
# sign module at large.
quant_asm=$(emit_asm "-C target-cpu=$tuned_cpu")

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

module_functions "$quant_asm" 8rotation > "$rotation_asm"
if [ ! -s "$rotation_asm" ]; then
    echo "no tantivy_quant::rotation functions found in $quant_asm" >&2
    exit 1
fi
if ! grep -Eq '\b(v?addps|v?subps)\b' "$rotation_asm"; then
    echo "tantivy_quant::rotation assembly does not contain packed float butterfly instructions" >&2
    exit 1
fi

echo "quant x86 asm expectations hold (default + $tuned_cpu)"
