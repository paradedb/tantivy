# Vector storage format

## Segment

A Tantivy index consists of segments. Each segment contains its own documents,
vector fields, and vector index. The schema defines each field's dimension and
distance metric; the segment stores the metadata needed to decode its values.

```text
+----------------------------------------------------------------------+
| One segment                                                          |
+----------------------------------------------------------------------+
| <segment-id>.vec                                                     |
| +----------------------------------------------------------------+   |
| | Vector field A: full vectors, IdMap, metadata, ...             |   |
| +----------------------------------------------------------------+   |
| | Vector field B: full vectors, IdMap, metadata, ...             |   |
| +----------------------------------------------------------------+   |
+----------------------------------------------------------------------+
| <segment-id>.centroids (present for clustered storage)               |
| +----------------------------------------------------------------+   |
| | Vector field A: cluster centers, counts, bounds, router        |   |
| +----------------------------------------------------------------+   |
| | Vector field B: cluster centers, counts, bounds, router        |   |
| +----------------------------------------------------------------+   |
+----------------------------------------------------------------------+
| Other files: documents, postings, deletion visibility, ...           |
+----------------------------------------------------------------------+
```

The sidecar's extension is `.centroids` (plural). Both files are shared by all
vector fields in the segment. Each field has its own entries within each file.

Initial indexing writes flat segments. A merge chooses flat or clustered output
using the target document count and `vector_clustering_threshold`. A clustered
merge trains centers, assigns vectors to clusters, and writes both files.
Quantization is configured per field and adds compact representations alongside
the full-precision vectors.

| Storage mode | Vector order within a field | Block contents | `.centroids` |
|---|---|---|---|
| Flat | Document ID order, skipping missing vectors | Full vectors | Absent |
| Clustered, plain | Cluster order, then document ID | Full vectors and document IDs | Present |
| Clustered, quantized | Cluster order, then document ID | Full vectors, document IDs, and quantization columns | Present |

Segment vector files are immutable once written. Inserts produce new segments.
Deletion visibility is stored separately; a merge omits deleted documents and
writes new vector files.

This document describes the implemented `.vec` **V5** and `.centroids` **V3**
formats. Cluster row offsets live only in `.centroids`; `.vec` stores block byte
offsets. All multibyte numbers below are little-endian. Ranges use an exclusive
end: `[start, end)`.
Diagrams show grouping and byte order; box sizes are not proportional to byte
sizes. Borders and labels do not represent extra headers on disk.

## `.vec`: vector data for each field

Each vector field has two composite entries: **IdMap** and **Data**. Data holds
the field metadata, vector blocks, and a directory describing those blocks.

```text
<segment-id>.vec: logical contents of one field
+----------------------------------------------------------------------+
| IdMap (entry 0)                                                      |
|   Document ID -> vector location                                     |
+----------------------------------------------------------------------+
| Data (entry 1)                                                       |
| +----------------------------------------------------------------+   |
| | Metadata: dimension, metric, column layout, ...                |   |
| +----------------------------------------------------------------+   |
| | Block 0: vectors | doc_ids | codes | ...                       |   |
| | Block 1: vectors | doc_ids | codes | ...                       |   |
| | ...                                                            |   |
| +----------------------------------------------------------------+   |
| | byte_offsets: u64[B+1]   start of each block, in bytes         |   |
| +----------------------------------------------------------------+   |
| | num_blocks:   u64        B                                     |   |
| +----------------------------------------------------------------+   |
+----------------------------------------------------------------------+
```

The writer records `byte_offsets` in `BlockDirectory`. The reader keeps that
array as a lazy file slice. For clustered fields, row boundaries come from the
field's cluster-offset slot in `.centroids`. Flat fields derive row boundaries
from their fixed block size and IdMap row count.

### File framing and field entries

The diagram above groups entries by field. The physical writer places **all
Data entries before all IdMap entries**, followed by the composite directory:

```text
<segment-id>.vec: bytes from start to end
+----------------------------------------------------------------------+
| Version: u32 = 5                                                     |
+----------------------------------------------------------------------+
| Zero padding to an absolute 8-byte boundary                          |
+----------------------------------------------------------------------+
| Data(field A)                                                        |
+----------------------------------------------------------------------+
| Data(field B)                                                        |
| ... remaining Data entries                                           |
+----------------------------------------------------------------------+
| IdMap(field A)                                                       |
+----------------------------------------------------------------------+
| IdMap(field B)                                                       |
| ... remaining IdMap entries                                          |
+----------------------------------------------------------------------+
| Composite directory: (field, entry index) -> entry location          |
+----------------------------------------------------------------------+
| Composite directory length: u32                                      |
+----------------------------------------------------------------------+
| Standard Tantivy checksum/version footer: 24 bytes                   |
+----------------------------------------------------------------------+
```

Entry indices are lookup keys, not physical positions. Each field must declare
exactly entry 0 (IdMap) and entry 1 (Data); missing or additional entry indices
are corruption. Data entries start and end on absolute 8-byte boundaries.
IdMaps need no alignment or inter-entry padding.

### IdMap: document to vector location

A **document ID** identifies a document within its segment. A **row number**
identifies a vector within one field's stored vector sequence. Documents without
a vector have no row. In clustered storage, rows are reordered by cluster, so a
document ID generally differs from its row number.

IdMap begins with a one-byte tag:

| Tag | Encoding | Payload after the tag | Lookup |
|---|---|---|---|
| 0 | Identity | None | Row number equals document ID |
| 1 | Bitmap | Columnar OptionalIndex encoding of documents with vectors | Bitmap rank gives the row number |
| 3 | DocLocations | One `(cluster: u32, local: u32)` per segment document | Cluster index and position within that cluster |

All other tags, including 2, are rejected. Flat fields use Identity when every
document has a vector, otherwise Bitmap. Identity occupies just one byte; the
document count comes from the segment.

Clustered fields use DocLocations:

```text
IdMap entry: clustered encoding
+----------------------------------------------------------------------+
| Tag: u8 = 3                                                          |
+----------------------------------------------------------------------+
| Document 0:         [cluster: u32][local: u32]   8 bytes             |
+----------------------------------------------------------------------+
| Document 1:         [cluster: u32][local: u32]   8 bytes             |
+----------------------------------------------------------------------+
| ...                                                                  |
+----------------------------------------------------------------------+
| Document max_doc-1: [cluster: u32][local: u32]   8 bytes             |
+----------------------------------------------------------------------+
```

The entry occupies `1 + 8 * max_doc` bytes, including records for documents
without vectors. `max_doc` includes deleted documents until a merge removes
them. Record `d` begins at byte `1 + 8*d` within IdMap. Missing vectors use
`(u32::MAX, 0)`, so the cluster count must be less than `u32::MAX`.

Opening validates the table length. Individual lookups validate the cluster
index and local position. The clustered table is read lazily for document
lookups; scans use the reverse mapping in each block's DocIds column.

### Data: metadata, blocks, and directory

The exact order within each field's Data entry is:

```text
One field's Data entry: bytes from start to end
+----------------------------------------------------------------------+
| metadata_length: u32                                                 |
| metadata: bytes[metadata_length]                                     |
+----------------------------------------------------------------------+
| Zero padding to block alignment                                      |
+----------------------------------------------------------------------+
| Block 0: columns for this block's vectors                            |
+----------------------------------------------------------------------+
| Block 1: columns for this block's vectors                            |
| ...                                                                  |
+----------------------------------------------------------------------+
| Block B-1: columns for this block's vectors                          |
+----------------------------------------------------------------------+
| Zero padding to 8 bytes                                              |
+----------------------------------------------------------------------+
| byte_offsets: u64[B+1]   [start 0][start 1] ... [directory start]    |
+----------------------------------------------------------------------+
| num_blocks: u64 = B     final 8 bytes of the Data entry              |
+----------------------------------------------------------------------+
```

Padding here is relative to the Data entry start. No per-block header or
per-column offset table is written between the blocks.

#### Metadata

Metadata records the field's dimensions, element type, metric, normalization
policy, and partitioning. Quantized fields also record the ordered quantization
layers, code widths, rotations, seeds, and reconstruction/error models.

`Partition::Clusters` requires a DocLocations IdMap and centroid data.
`Partition::Uniform { rows_per_block }` requires a nonzero block size, an
Identity or Bitmap IdMap, and no centroid data for the field.

These values determine the column order, alignment, and bytes per vector in
each column. They are shared by every block in this field. The
[metadata grammar](#metadata-grammar) specifies their encoding.

Readers use the segment's persisted metadata. Changing index settings changes
the target for future builds; it does not change how existing segments decode.

#### Block directory

For clustered block `b`, `.vec` gives its location and `.centroids` gives its
vector count:

```text
block_start = byte_offsets[b]             relative to the Data entry start
block_end   = byte_offsets[b + 1]
count       = cluster_offsets[b + 1] - cluster_offsets[b]  clustered only
```

For clustered fields, **block index equals cluster index**. For flat fields,
blocks are consecutive groups of up to 16,384 vectors. Flat block `b` contains
rows `[b*r, min((b+1)*r, total_rows))`, where `r = rows_per_block`.

The byte-offset array has an extra terminal boundary pointing to the directory,
after any padding at the end of the block area. Empty clusters consume no column
bytes but retain their byte and row boundaries. An empty flat field has zero
blocks and a single byte-offset sentinel. Flat fields need no stored row-offset
array: the IdMap supplies the total row count and metadata supplies `r`.

The trailing `num_blocks` locates the directory without scanning block contents.
With `entry_end` relative to the Data entry start:

```text
directory_start = entry_end - 8 - 8 * (B + 1)
```

Search-reader initialization reads the block-count footer and checks directory
framing without loading the byte-offset array. For clustered fields, it opens
the `.centroids` offset slot and reads only its first and final values. The first
must be zero; the final value supplies the total row count. An empty field also
validates its sole byte-offset sentinel and padding.

The first access to a cluster reads two adjacent `u64` byte offsets from `.vec`
and two adjacent `u64` row offsets from `.centroids`: two 16-byte reads. It checks
ordered boundaries, alignment, row-count limits, and that the calculated columns
fit the stored block size. Accessing the final block also checks the terminal
byte boundary and zero padding before the directory. The combined address is
cached for subsequent column accesses. Invalid boundaries in an unvisited
cluster are detected when that cluster is first accessed.

### Block: a collection of vectors stored as columns

A block is a logical collection of vectors. In clustered storage, it contains
all vectors assigned to one cluster. Its byte size varies with the vector count
and field configuration. It is not tied to a PostgreSQL page or disk sector,
and it provides no additional atomic-read or atomic-write guarantee.

Each block stores one complete column before the next column begins. Here `N`
is the number of vectors in this block; alignment padding is omitted:

```text
Flat, plain block
+----------------------------------------------------------------------+
| Rows: [v0][v1] ... [vN-1]                                            |
+----------------------------------------------------------------------+

Clustered, plain block
+----------------------------------------------------------------------+
| Rows:   [v0][v1] ... [vN-1]                                          |
+----------------------------------------------------------------------+
| DocIds: [d0][d1] ... [dN-1]                                          |
+----------------------------------------------------------------------+

Clustered, quantized block: bytes from start to end
+----------------------------------------------------------------------+
| Rows:          [v0][v1] ... [vN-1]                                   |
+----------------------------------------------------------------------+
| DocIds:        [d0][d1] ... [dN-1]                                   |
+----------------------------------------------------------------------+
| ResidualNorms: [n0][n1] ... [nN-1]                                   |
+----------------------------------------------------------------------+
| Layer 0 Codes:     [code0][code1] ... [codeN-1]                      |
+----------------------------------------------------------------------+
| Layer 0 Scales:    [s0][s1] ... [sN-1]                               |
+----------------------------------------------------------------------+
| Layer 0 Gammas:    [g0][g1] ... [gN-1]                               |
+----------------------------------------------------------------------+
| Layer 0 Errors:    [e0][e1] ... [eN-1]                               |
+----------------------------------------------------------------------+
| Layer 0 Constants (L2 only): [k0][k1] ... [kN-1]                     |
+----------------------------------------------------------------------+
| Layer 1: Codes, Scales, Gammas, Errors, Constants (L2 only)          |
| ... remaining layers, each with the same column order                |
+----------------------------------------------------------------------+
```

All columns use the same local row order. For example, the entries at local
row 7 in Rows, DocIds, and Codes describe the same member of the cluster.
Full vectors remain present when quantization is enabled.

#### Columns and their encodings

A column's **stride** is its bytes per vector. If the block contains `count`
vectors, the column contains `count * stride` bytes.

| Column in write order | Decoder element | Stride | Meaning |
|---|---|---:|---|
| Rows | F32 | `4 * dim` | Full-precision vector values |
| DocIds, clustered only | U32 | 4 | Segment document ID for each vector |
| ResidualNorms, quantized only | F32 | 4 | Squared norm of each vector's residual from its centroid |
| QuantLayerCodes(l) | U64 for SignPlane; U8 for GridPlane | Quantizer code stride | Compact residual representation |
| QuantLayerScales(l) | F32 | 4 | Scale for score reconstruction |
| QuantLayerGammas(l) | F16 | 2 | Correction factor |
| QuantLayerErrors(l) | F16 | 2 | Error information used for pruning |
| QuantLayerConstants(l), L2 only | F32 | 4 | Additional L2 scoring term |

ResidualNorms appears once. The layer columns repeat in ascending layer order;
quantized fields contain one to three layers.

**Rows** stores each vector's coordinates together:

```text
Rows column: vectors laid out one after another
+----------------------------------------------------------------------+
| v0: [coordinate 0: f32] ... [coordinate dim-1: f32]                  |
+----------------------------------------------------------------------+
| v1: [coordinate 0: f32] ... [coordinate dim-1: f32]                  |
+----------------------------------------------------------------------+
| ...                                                                  |
+----------------------------------------------------------------------+
| vN-1: [coordinate 0: f32] ... [coordinate dim-1: f32]                |
+----------------------------------------------------------------------+
```

The current element type is F32. Cosine vectors are normalized when written, so
full precision refers to the stored representation, not necessarily the original
input bytes. Flat blocks have no other columns, making their vector bytes
contiguous across block boundaries.

**DocIds** contains strictly ascending `u32` document IDs within each cluster,
all below `max_doc`. It answers which document owns a local vector row, the
reverse of IdMap's document-to-location lookup.

**Codes** use `8 * ceil(dim / 64)` bytes per vector for one-bit SignPlane.
GridPlane uses two to four bits per dimension and
`8 * ceil(dim * bits / 64)` bytes per vector. Unused code tail bits must be zero.
The scoring columns use binary32 or binary16 as shown in the table.

#### Calculating a column's offset

Column offsets are calculated separately for each block. They depend on the
block's vector count; they are not fixed constants and are not stored on disk.
The reader derives a short ordered list of columns and strides from the field
metadata.

```text
position = 0
for each column in layout order:
    start = align_up(position, column.element_size)
    end = start + count * column.stride
    position = end

column file range = Data entry start + block_start + [start, end)
```

Computing the ranges of preceding columns requires only arithmetic. Their bytes
do not need to be read. Within a column, local row `i` starts at
`column_start + i * stride`. The reader also checks the requested range against
the next block's byte offset.

For example, use four-dimensional F32 vectors, one SignPlane layer, and dot
product scoring. The codes occupy one 8-byte word per vector:

```text
Block with 3 vectors                  Block with 5 vectors
  0 +----------------------------+      0 +----------------------------+
    | Rows: 48 bytes             |        | Rows: 80 bytes             |
 48 +----------------------------+     80 +----------------------------+
    | DocIds: 12 bytes           |        | DocIds: 20 bytes           |
 60 +----------------------------+    100 +----------------------------+
    | ResidualNorms: 12 bytes    |        | ResidualNorms: 20 bytes    |
 72 +----------------------------+    120 +----------------------------+
    | Codes: 24 bytes            |        | Codes: 40 bytes            |
 96 +----------------------------+    160 +----------------------------+
    | Scales: 12 bytes           |        | Scales: 20 bytes           |
108 +----------------------------+    180 +----------------------------+
    | Gammas: 6 bytes            |        | Gammas: 10 bytes           |
114 +----------------------------+    190 +----------------------------+
    | Errors: 6 bytes            |        | Errors: 10 bytes           |
120 +----------------------------+    200 +----------------------------+
```

These offsets are relative to the block start. This example needs no padding.
Other dimensions, layer schedules, or metrics can insert alignment padding.

#### Alignment

Each column starts at a multiple of its decoder element size relative to the
block start. Every stride is a multiple of that size. The block ends on a
multiple of its largest decoder element: 4 bytes for plain storage, 8 bytes when
SignPlane is present. Empty blocks consume no column bytes. All padding is zero.

Sizes come from fixed-width representations, independent of ABI alignment; F16
uses the size of `u16`. Every element size is a power of two. The exported
`MAX_ELEM_BYTES` is the maximum size and must not exceed `ENTRY_ALIGN`, which is
fixed at 8 bytes. No alignment value is stored in the file. File alignment does
not guarantee an aligned memory address; decoders support unaligned bytes.

## `.centroids`: clustering data for each field

This file describes the clusters whose vector data is in `.vec`. A centroid is
a representative vector for a cluster. Each field has four composite slots:

```text
<segment-id>.centroids: bytes from start to end
+----------------------------------------------------------------------+
| Version: u32 = 3                                                     |
+----------------------------------------------------------------------+
| Field A                                                              |
| +----------------------------------------------------------------+   |
| | Slot 0: counts and full-precision cluster centers              |   |
| +----------------------------------------------------------------+   |
| | Slot 1: cluster_offsets, cumulative vector counts              |   |
| +----------------------------------------------------------------+   |
| | Slot 3: bounds, a radius around each cluster center            |   |
| +----------------------------------------------------------------+   |
| | Slot 2: router, navigation among cluster centers               |   |
| +----------------------------------------------------------------+   |
+----------------------------------------------------------------------+
| Field B: the same four slots                                         |
| ... remaining fields                                                 |
+----------------------------------------------------------------------+
| Composite directory: (field, slot index) -> entry location           |
+----------------------------------------------------------------------+
| Composite directory length: u32                                      |
+----------------------------------------------------------------------+
| Standard Tantivy checksum/version footer: 24 bytes                   |
+----------------------------------------------------------------------+
```

The writer emits slots in the order **0, 1, 3, 2** for each field. Slot indices
are lookup keys. The file begins with `version: u32 = 3` and ends with the same
composite directory, directory-length word, and standard footer used by `.vec`.

The cluster index is shared across both files: center `c`, offsets `c` and
`c+1`, bound `c`, and `.vec` block `c` all refer to the same cluster. Stacked
routing can reorder clusters during construction; all these structures use
the resulting order.

### Slot 0: counts and centroid vectors

```text
Slot 0: counts and centroid vectors
+----------------------------------------------------------------------+
| num_centroids: u32 = K                                               |
+----------------------------------------------------------------------+
| num_docs: u32          documents with a vector in this field         |
+----------------------------------------------------------------------+
| centroids: f32[K * dim]                                              |
|   [center 0: dim floats][center 1: dim floats] ... [center K-1]      |
+----------------------------------------------------------------------+
```

Each center uses `4 * dim` bytes and the field's normalization policy. The counts
occupy 8 bytes; the remaining slot length must be exactly `K * 4 * dim`. The
current writer assigns each vector-bearing document to one cluster.

### Slot 1: cluster row offsets

This slot contains `cluster_offsets: u64[K+1]`. Each entry is the number of
vectors preceding that cluster, so adjacent entries give its vector count:

```text
count = cluster_offsets[c + 1] - cluster_offsets[c]
```

These are row offsets, not byte locations. Together with the byte offsets in
`.vec`, they describe each cluster's position and count:

```text
                                    Cluster 0  Cluster 1  End
                                  +----------+----------+----------+
.centroids cluster_offsets (u64)  | 0        | 3        | 8        |
                                  +----------+----------+----------+
.vec byte_offsets (u64)           | start 0  | start 1  | end      |
                                  +----------+----------+----------+
```

Cluster 0 has `3 - 0 = 3` vectors; cluster 1 has `8 - 3 = 5`. The byte offsets
locate their blocks, and these counts determine the column offsets within them.
The slot must contain exactly `K+1` offsets and agree with the `.vec` block count.
Offsets are read lazily and must be nondecreasing and no larger than the final
row count, which is limited to `u32::MAX` by vector row addressing.

V4 duplicated these counts in a `.vec` `row_offsets: u32[K+1]` array. V5 removes
that array. The `.centroids` slot keeps its V3 encoding.

### Slot 3: cluster bounds

```text
Slot 3: cluster bounds
+----------------------------------------------------------------------+
| bound_kind: u8 = 0     Ball                                          |
+----------------------------------------------------------------------+
| radii: f32[K]         [radius 0][radius 1] ... [radius K-1]          |
+----------------------------------------------------------------------+
```

A radius covers the cluster's stored members around its stored center, using
the Euclidean residual norm. Search uses it to determine when a cluster cannot
improve the current results. Infinite radii disable that skip. Negative bounds
are rejected. The slot occupies `1 + 4*K` bytes.

### Slot 2: router

```text
Slot 2: router
+----------------------------------------------------------------------+
| router_kind: u8       1 = Stacked, 2 = Exact                         |
+----------------------------------------------------------------------+
| router_payload       variable length; empty for Exact                |
+----------------------------------------------------------------------+
```

The default Stacked router has tag 1. It builds a hierarchy over the field's
cluster centers, with its own higher-level centers and member ranges. The
payload encodes levels from the bottom routing level to the top:

```text
centroid_rows_offset: u64                  relative to router payload start
num_levels: u32
for each routing level:
    num_lists: u32
    member_ranges: (start:u64, end:u64)[num_lists]
for each routing level:
    centers: f32[num_lists * dim]
```

The bottom level's members are the centers already in slot 0; those vectors are
not duplicated in the router payload. The routing configuration itself is not
serialized. The offsets above describe routing membership, distinct from slot
1's mapping from clusters to document-vector rows.

The Exact router has tag 2 and no payload. It scores the centers from slot 0
directly. Other router tags are rejected. Readers follow the persisted router
kind, regardless of the settings used for future builds.

## Following a document lookup

Suppose a segment has five documents, document 2 has no vector, and clustering
assigns documents 0 and 3 to cluster 0, and documents 1 and 4 to cluster 1:

```text
.vec: two clusters for one field
+----------------------------------------------------------------------+
| Block 0 = cluster 0                                                  |
| +----------------------------------------------------------------+   |
| | Rows: [v0][v3]                                                 |   |
| +----------------------------------------------------------------+   |
| | DocIds: [0][3]                                                 |   |
| +----------------------------------------------------------------+   |
| | Quantization columns, when enabled                             |   |
| +----------------------------------------------------------------+   |
+----------------------------------------------------------------------+
| Block 1 = cluster 1                                                  |
| +----------------------------------------------------------------+   |
| | Rows: [v1][v4]                                                 |   |
| +----------------------------------------------------------------+   |
| | DocIds: [1][4]                                                 |   |
| +----------------------------------------------------------------+   |
| | Quantization columns, when enabled                             |   |
| +----------------------------------------------------------------+   |
+----------------------------------------------------------------------+

IdMap entry
+----------------------------------------------------------------------+
| doc 0 -> (cluster 0, local 0)                                        |
+----------------------------------------------------------------------+
| doc 1 -> (cluster 1, local 0)                                        |
+----------------------------------------------------------------------+
| doc 2 -> absent                                                      |
+----------------------------------------------------------------------+
| doc 3 -> (cluster 0, local 1)                                        |
+----------------------------------------------------------------------+
| doc 4 -> (cluster 1, local 1)                                        |
+----------------------------------------------------------------------+

.centroids cluster_offsets = [0, 2, 4]
```

To retrieve document 4, IdMap supplies cluster 1 and local position 1. The
directory supplies that cluster's starting byte and count of 2. Rows is the
first column, so the requested vector starts at:

```text
Data entry start + byte_offsets[1] + 1 * (4 * dim)
```

To scan cluster 1's codes instead, the reader computes the Codes column range
using count 2 and the field metadata. It can read that range without reading
Rows or DocIds. Resolving a result back to a document uses DocIds at the same
local position.

## Composite directory and standard footer

Both files use [CompositeFile](../../directory/composite_file.rs) to locate
field entries. This is separate from the block directory inside a `.vec` Data
entry. Composite offsets are relative to the composite body, after the file's
4-byte vector-format header.

```text
number_of_entries: VInt
for each entry in physical order:
    delta_from_previous_entry_start: VInt
    field_id: u32
    entry_index: VInt
directory_length: u32
```

VInt is Tantivy's variable-length unsigned integer encoding. The first delta
is relative to composite offset zero. An entry ends at the next entry's start;
the last entry ends at the start of this directory.

The managed directory appends a separate 24-byte
[standard footer](../../directory/footer.rs): library major/minor/patch
versions, index-format version, CRC32, and magic number 1337, each a `u32`.
This framing is independent of the vector file versions and per-field metadata.

## Read behavior

The independently addressable columns allow search to fetch the representation
needed at each stage. Full vectors are read for exact scoring. DocIds can be
read first to apply filters and deletion visibility; a cluster with no surviving
rows then needs no vector payload read. Without filtering or deletions,
quantized search defers DocIds until the exact phase resolves final candidates.
Exact scans without a row gate resolve DocIds only when a score reaches heap
admission.

A **band** is a contiguous group of quantization columns read together. Band 0
spans ResidualNorms through the final layer-0 column. Each higher band spans
that layer's codes through its last column. A band uses one logical range read
and exposes its columns as views; Rows and DocIds do not belong to any band.

Sparse code reads group selected rows only while their storage-page spans
overlap. Adjacent disjoint pages start separate reads, and each range ends at
its last selected row. Scales through the last sidecar or constants column use
one span per touched cluster. Codes are not coalesced with those sidecars.
Multi-page reads may copy under paged storage, so fewer logical reads do not
necessarily mean less work. A row-range read from one column cannot cross block
boundaries; sorted row requests are split into their containing blocks.

The writer streams full vector rows immediately and buffers encoded columns
for one cluster, then flushes them in layout order. The reader materializes
bounds and router topology. Block byte offsets, cluster row offsets, and centroid
vectors remain available through lazy range reads; only accessed block addresses
are cached. ANN carries cluster IDs through exact reranking, avoiding a search
through unrelated clusters. Row-only merge and diagnostic APIs use lazy binary
search when they need to resolve a global row to a cluster.

## Metadata grammar

All multibyte fields are little-endian. Tags use explicit maps, independent of
Rust discriminants. The `.vec` version selects the grammar; metadata has no
additional version field.

```
meta       := repr:u8 field [n_layers:u8 quantizer{n_layers} if repr=1]
field      := dim:u32 dtype:u8 metric:u8 norm_policy:u8 partition
partition  := kind:u8 [rows_per_block:u32 if kind=1]
quantizer  := kind:u8 payload
  SignPlane := rotation rho_model:f64
  GridPlane := bits:u8 rotation rho_model:f64
               n_points:u16 points:f32{n_points}
rotation   := kind:u8 [seed:u64 if kind=1]
```

| Tag domain | Frozen values |
|---|---|
| repr | 0 Plain, 1 Quantized |
| dtype | 0 F32 |
| metric | 0 L2, 1 Dot, 2 Cosine |
| norm_policy | 0 None, 1 UnitL2 |
| partition | 0 Clusters, 1 Uniform |
| quantizer | 0 SignPlane, 1 GridPlane |
| rotation | 0 None, 1 SeededFhtChaCha8 |

Unknown tags fail at open. SignPlane is the one-bit popcount contract without
grid points. GridPlane permits bits 2 through 4 and exactly `1 << bits` points.
SeededFhtChaCha8 pins FHT construction, ChaCha8, and `rand_core::seed_from_u64`.
Quantized fields contain one to three layers. Dimension, dtype, metric and
normalization must match the schema. Parsing rejects truncated records,
trailing metadata bytes, invalid geometry, and entry-length mismatches as
DataCorruption before payload access.

## Versioning rules

The current `.vec` reader supports V5; earlier vector formats require rebuilding.
The `.centroids` reader uses V3 for its required bounds and tagged router.
`VectorQuantizationConfig.format_version = 3` identifies the independent index
settings grammar and specifies a build target, not a segment file version.

1. Stored values need no version: grid points, rho_model, block size,
   and IdMap data are self-describing inputs.
2. The `.vec` version alone versions byte grammar, entry framing, and metadata
   encoding. New tag values need no bump; adding fields to an existing record does.
3. Quantizer contracts and seed-derived rotation semantics are pinned by enum
   tags. Semantic changes add a variant and retain the existing decode arm.
4. `slots()` and `layer_slots()` are format semantics, including ordering,
   element widths, strides, and band ownership. Golden tests pin these lists.
5. Golden bytes pin encoder columns and transforms for each quantizer and rotation.
6. Query policy remains code: kappa, statistical-width multiplier and SIGN_QUERY_BITS.
7. Retiring a decode arm is deliberate; segments using retired tags fail with
   “rebuild required.”

## Query decoding and scoring contracts

The following contracts govern interpretation of the stored quantization data.

### Prepared-query identity

Quantized metadata compares and hashes `(dim, metric, layers)`. A quantizer's
identity includes its tag, width, rotation tag and seed, and model bits. Grid
identity is `(rho_model.to_bits(), points.map(to_bits))`. Quantizer equality and
hashing include every field, with floating-point values compared by bit pattern.
Signed zero and all floating-point bit distinctions are preserved.

Plain metadata compares by variant only and is never a prepared-query key.
Normalization, dtype and partition do not affect preparation. Full
storage identity, including round trips, compares serialized metadata bytes.
A collector caches one OnceLock cell per metadata key and releases its map lock
before preparation. Read paths use the segment's metadata, never global build
settings.

### Threshold and policy

All scores use higher-is-better score space. At each layer, and while admitting
clusters, the threshold is the **k-th largest lower endpoint**
`estimate - kappa * sigma`. It is not the lower endpoint of the k-th point
estimate. A candidate survives when its upper endpoint reaches the threshold.
Consequently, enclosing intervals preserve the true top-k; widening any interval
or increasing kappa cannot remove an existing survivor. Statistical intervals
are not guaranteed to enclose every true score, so exact scoring of the survivors alone
does not guarantee recall parity.

The policy constants are uniform `kappa = 2.5`, statistical-width multiplier
`1.15`, and serialized gamma clamp `[1, 4]`. None is fitted per dataset, layer,
or query. The lower-endpoint threshold correction is independent of kappa.

### Arithmetic uncertainty

For separately rounded terms combined as a sum or difference, the additional
analytical term is `delta = c * f32::EPSILON * (abs(a) + abs(b))`. Independent
contributions are accumulated in quadrature, separately from the statistical
model; there is no empirical multiplier. The rounding-count constants are
documented at the expressions in `prepared.rs`:

- Initial L2 split subtraction: `c = 2` for operand rounding and fused subtraction.
- L2 refinement also includes the separate `prefix - constant` subtraction with
  `c = 1`.
- Initial dot/cosine scaling uses `c = 1`; refinement scaling and addition use
  `c = 2`.
- The L2 exact-base subtraction and final fused base-plus-corrected-residual
  expression each use `c = 1`, including cosine's base-plus-residual sum.

Raw-prefix variance is propagated through the current gamma correction before
combining with the model width. Refinement preserves previous arithmetic error.
Thus L2 at `query = centroid` still has positive uncertainty when cancellation
is possible, even though its data-model width vanishes.

Large common offsets can widen L2 arithmetic bounds and retain extra candidates.
Translation tests use exactly representable input translations and require the
same final ranking, true top-k survival at every boundary, and a translated
survivor set containing the original set. Centroid-relative L2 scoring remains
a follow-up to reduce this conservative cost on offset-heavy data.

## Implementation references

| Responsibility | Source |
|---|---|
| File versions and entry indices | [header.rs](header.rs) |
| Metadata and derived column layout | [metadata.rs](metadata.rs) |
| Block directory and column offsets | [blocks.rs](blocks.rs) |
| Document addressing | [id_map.rs](id_map.rs) |
| Centroid slots and cluster reader | [centroids.rs](centroids.rs) |
| Vector reader | [reader.rs](reader.rs) |
| Flat writer | [write/flat.rs](../write/flat.rs) |
| Clustered block writer | [write/ivf/encode.rs](../write/ivf/encode.rs) |
| Centroid slot writer | [write/ivf/write.rs](../write/ivf/write.rs) |
| Router payload | [router/stacked/levels.rs](../router/stacked/levels.rs) |
