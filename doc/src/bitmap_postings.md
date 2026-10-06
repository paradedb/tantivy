# Optional posting bitmaps

Posting bitmaps accelerate unscored membership queries and counts over common terms.
They are an optional built-in inverted-index component, alongside posting norms.
The ordinary postings, interleaved frequencies, and positions remain available for
scoring, positional queries, and sparse terms.

## Configuration and format

Enable the component on a text field's `TextFieldIndexing`:

```rust
let indexing = TEXT.get_indexing_options().unwrap().clone()
    .set_bitmap_postings(true);
let field = schema.add_text_field("body", TEXT.set_indexing_options(indexing));
```

JSON text indexing accepts the same option. Existing schemas default to disabled.
`IndexSettings::bitmap_postings` supplies these defaults:

| Setting | Default | Meaning |
| --- | --- | --- |
| `min_density_percent` | 10 | Minimum `doc_freq / max_doc`, between 1 and 100 |
| `min_docs` | 128 | Minimum term document frequency |
| `max_bytes_per_segment` | 64 MiB | Shared payload budget across enabled fields |
| `use_for_queries` | true | Allow readers to use available bitmaps |

Universal terms use the existing all-documents shortcut. Each other eligible term
uses `8 * ceil(max_doc / 64)` bytes of little-endian bitmap words in `.bmap`.
Selection is density-based, with a storage budget; it does not assume a bitmap is
smaller than compressed postings. The budget is consumed in serialization order.
Terms that do not fit retain only their ordinary postings. The budget excludes
component headers and term-dictionary metadata.

Term metadata stores an optional offset. Readers accept existing V1/V2 term
dictionaries; dictionaries with bitmap offsets use V3. Older binaries cannot read
V3 dictionaries. Flush and merge regenerate bitmaps from the remapped postings,
including deletion compaction. Enabling the option does not retrofit old segments
until they are rewritten. Disabling `use_for_queries` before constructing a reader
leaves the stored component intact, providing a comparison against the same index.

## Query execution

Unscored term scorers expose bitmap windows of 1,024 document IDs. Union,
intersection, and exclusion can combine these windows without enumerating every
matching document between operators. Word loops let the compiler vectorize dense
AND, OR, and AND-NOT operations without requiring architecture-specific intrinsics.

Sparse intersections still drive the existing iterator intersection and probe the
dense bitmap. The current intersection heuristic chooses block composition when
at least one child exposes fast bitmap blocks and every child estimates at least
`num_docs / 32` matches. This execution heuristic is separate from the persisted
bitmap eligibility threshold. Inaccurate size estimates can affect performance,
but not membership.

Other producers can supply words directly or use the default document-to-bitmap
adapter. This includes fast-field ranges, existing in-memory bitsets, and term
expansions. Phrases and minimum-should-match queries retain their membership
semantics through existing scorers. Scored queries retain their ordinary paths.

The unscored collector path accepts either document batches or bitmap windows.
Tantivy's alive mask is applied before collection. `Count`, match-all filter counts
without subaggregations, and value counts on full single-valued columns can count
bits directly. Other collectors enumerate through the default adapter. Collector
wrappers must forward `collect_bitmap` to preserve this optimization.

An embedding database remains responsible for its own visibility rules. An
all-visible segment can forward windows to its child collector; otherwise the
wrapper can enumerate into its existing visibility batches. The persistent term
bitmap is not a transaction visibility cache.

## Reproducing the benchmark

```sh
cargo run --release --no-default-features --features quickwit,lz4-compression \
  --example bitmap_counts -- 200000 101 10
```

The arguments are document count, measured rounds, and density cutoff percentage.
Optional trailing arguments select an exact query, variant, and collector, for
example `"a OR b" bitmap_enabled wrapped` for a focused CPU profile.

The benchmark generates deterministic, unsorted data: terms near 50%, 33%, 12%,
6%, and 0.1% frequency, plus a numeric fast field. It compares an ordinary index,
a bitmap index with reads disabled, and the same bitmap index with reads enabled.
Three warmup rounds precede interleaved variant measurements. JSON lines report
build time, component bytes, median/p95 latency, counts, and root bitmap versus
document batches. `wrapped` uses `Some(Count)` to exercise the generic collector
path through a wrapper; it does not simulate PostgreSQL visibility costs.

One warm, single-segment run on an Apple M5 with Rust 1.94.1, 200,000 documents,
101 rounds, and the 10% cutoff produced these wrapped-count medians:

| Query | Ordinary (us) | Bitmap reads off (us) | Bitmap reads on (us) |
| --- | ---: | ---: | ---: |
| `a OR b` | 674.5 | 674.5 | 18.0 |
| `a AND b` | 1842.4 | 1603.4 | 34.7 |
| `(a OR b) AND c` | 1773.7 | 1724.1 | 50.8 |
| `(a OR b) -c` | 1711.0 | 1661.7 | 49.1 |
| `rare AND a` | 26.3 | 26.3 | 5.9 |
| `rare AND medium` | 12.6 | 12.7 | 12.7 |
| `a AND number:[100 TO 1000]` | 1215.2 | 1215.5 | 216.2 |
| `a AND number:[100 TO 100000]` | 1875.2 | 1859.3 | 498.5 |

Dense OR emitted 196 bitmap batches and zero document batches, versus 132,625
enumerated documents without bitmap reads. Separate native samples showed the
old buffered-union refill and enumeration loops absent from the enabled run;
time moved to bitmap reads, word composition, and counting. ARM64 disassembly
contained vector `and`, `orr`, and `bic` operations. The sparse intersection
`rare AND a` retained document iteration, returning 112 candidates.

Total reported segment storage grew from 1,641,682 to 1,716,715 bytes (4.6%), of
which 75,000 bytes were bitmap payload. Top-10 medians on the same query matrix
were within about 3.1% of the ordinary index. Build times were 134 and 136 ms in
this run; a single build pair is not a throughput estimate.

A cutoff sweep wrote 100,000 / 75,000 / 50,000 bitmap bytes at 5% / 10% / 20%.
For `medium AND c`, within-run speedups over reads disabled were 6.5x / 3.0x / 1.0x.
The 5% setting also accelerated `rare AND medium` by 2.4x. These results support
keeping the cutoff configurable; 10% is a conservative starting default, not a
universal optimum. Runs shared a development machine, so compare variants within
a run and remeasure on the target workload.

## Verification

The regression suite covers storage budgets, optional components, remapped and
merged segments, deletes, both term-dictionary backends, posting-norm coexistence,
cursor/property tests, mixed sparse/dense Boolean trees, ranges, term expansions,
phrases, minimum-should-match, and scored-result equivalence. Mechanism tests fail
if nested bitmap operators enumerate their leaves or a bitmap count collector is
fed individual documents. Nullable and multivalued counts retain their fallback.

```sh
cargo test --lib --no-default-features bitmap
cargo test --lib --no-default-features termdict
cargo test -p tantivy-common union_words
cargo test --lib --no-default-features --features quickwit -- \
  --skip l2_translation_preserves_ranking_and_conservatively_widens_survivors
```

The final command excludes an unrelated long-running vector test. The synthetic
results establish removal of the targeted enumeration work; production claims
still require the original count workload, realistic segment counts, cold reads,
index/merge throughput, and database visibility costs.
