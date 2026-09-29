# Vector storage tradeoffs

Sparse code reads group selected rows only while their storage-page spans overlap.
Adjacent disjoint pages start separate groups.
Each range begins at its first selected row and ends after its last selected row;
a row spanning two pages connects overlapping spans on both pages. Available cluster
boundaries do not expand a code range. Storage without page geometry groups only
consecutive selected rows.

Sparse reads consume ordered code ranges directly using reusable row-range and
selection scratch. Each touched cluster's sidecar span, from scales through
errors or constants, is pinned with one read and decoded together. The span
avoids per-column planning, request/view allocation and sorting. Code reads
remain separate because paged storage copies multi-page requests; equal page
counts do not make a wider code range free.

The cluster-major layout trades cross-cluster page sharing for locality within a
cluster. Additional distinct pages at later layers and the loss of cross-cluster
sharing at layer zero are
accepted layout costs. Layer 0 can still use fewer read calls and buffer
acquisitions because its band is contiguous within each block.

Cold layer 1 benefits from band-0/band-1 adjacency within a block: processing band
0 can load a shared boundary page before band 1 needs it. A cold query starts
with cold PostgreSQL and OS caches; later stages can reuse pages loaded by
earlier stages of that query.

Multi-cluster band packing is a possible future format change. It is not planned.

Clustered document ids reside beside exact rows and the first scan band. Exact
and quantized scans acquire ids with their scoring bytes, while document lookups
read a single cluster/local pair from a lazily opened location table. Merge
source addressing comes from block traversal and the merge document mapping.
