# Tantivy Max Score API

This document describes Tantivy's Max Score API (`Weight::index_max_score`, `Weight::max_score`, and corresponding `Query` helpers).

The API provides sound, $O(1)$ upper bounds on query scores across the entire index and within individual segments. It is designed for multi-index joins (e.g., ParadeDB Top-K score joins), multi-query aggregations, and speculative pruning pipelines.

---

## 1. Motivation

In aggregate scoring queries where the final document score is a function of multiple queries (e.g., $S_{total} = S_A + S_B$ across distinct tables or indexes), evaluating both sides exhaustively is prohibitively expensive.

To enable early-pruning algorithms like Block-Max WAND (BMW) across joins, one search stream requires an upper bound $M_B$ on the maximum possible score the other query can produce:
$$\text{threshold}_A = \theta_{total} - M_B$$

The Max Score API exposes these upper bounds without needing to decode posting lists or iterate documents.

---

## 2. API Surface

### Trait `Weight`

Defined in `src/query/weight.rs`:

```rust
pub trait Weight: Send + Sync + 'static {
    // ... existing methods ...

    /// Returns an upper bound on the maximum score any document in the given segment
    /// can achieve for this query/weight and boost factor.
    ///
    /// The default implementation delegates to `self.index_max_score(boost)`.
    fn max_score(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Score> {
        self.index_max_score(boost)
    }

    /// Returns an upper bound on the maximum score across all segments in the index.
    ///
    /// The default implementation returns `Score::MAX` as a sound, conservative fallback.
    fn index_max_score(&self, boost: Score) -> crate::Result<Score> {
        Ok(Score::MAX)
    }
}
```

### Trait `Query` Helpers

Defined in `src/query/query.rs`:

```rust
pub trait Query: /* ... */ {
    /// Returns an upper bound on the maximum score across all segments in the index.
    fn index_max_score(&self, searcher: &Searcher) -> crate::Result<Score> {
        let weight = self.weight(EnableScoring::enabled_from_searcher(searcher))?;
        weight.index_max_score(1.0)
    }

    /// Returns an upper bound on the maximum score any document in the given segment
    /// can achieve for this query.
    fn max_score(
        &self,
        searcher: &Searcher,
        segment_reader: &SegmentReader,
    ) -> crate::Result<Score> {
        let weight = self.weight(EnableScoring::enabled_from_searcher(searcher))?;
        weight.max_score(segment_reader, 1.0)
    }
}
```

---

## 3. Query Type Implementations & Soundness Guarantees

| Query / Weight Type | `index_max_score(boost)` | `max_score(reader, boost)` |
| :--- | :--- | :--- |
| **`TermWeight`** | $\text{IDF} \cdot (k_1 + 1.0) \cdot \text{boost}$ | `0.0` if term is absent in segment dictionary; otherwise `index_max_score` |
| **`BooleanWeight`** | Combines positive clauses (`Must`, `Should`) via `ScoreCombiner`. `Filter` and `MustNot` clauses contribute `0.0`. | Per-segment combination; clauses missing in segment contribute `0.0` |
| **`DisjunctionMaxQuery`** | $M_{\max} + \lambda \sum_{i \ne \max} M_i$ (where $\lambda$ is tie breaker) | Per-segment disjunction max combination |
| **`BoostWeight`** | `inner_weight.index_max_score(boost * self.boost)` | `inner_weight.max_score(reader, boost * self.boost)` |
| **`PhraseWeight`** | $\text{IDF}_{\text{phrase}} \cdot (k_1 + 1.0) \cdot \text{boost}$ | `0.0` if any constituent term is absent in segment dictionary; otherwise `index_max_score` |
| **`ConstScoreQuery`** | `self.score * boost` | `self.score * boost` |
| **`AllQuery`** | `1.0 * boost` | `1.0 * boost` |
| **`EmptyQuery`** | `0.0` | `0.0` |
| **`ExistsQuery`** / **`TermSetQuery`** / **`RangeQuery`** | `1.0 * boost` | `1.0 * boost` |
| **Custom / Unknown `Weight`** | `Score::MAX` (safe fallback) | `Score::MAX` (safe fallback) |

### Mathematical Soundness for BM25

Tantivy's BM25 formula calculates score as:
$$\text{Score}(D, Q) = \text{IDF} \cdot \frac{\text{tf} \cdot (k_1 + 1.0)}{\text{tf} + k_1 \cdot \left(1.0 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Because $\frac{\text{tf}}{\text{tf} + \text{norm}} \le 1.0$ for all document lengths $|D| \ge 0$ and term frequencies $\text{tf} \ge 0$, $\text{IDF} \cdot (k_1 + 1.0)$ is a strict mathematical supremum.

---

## 4. Usage Examples

### Example A: Query-Level Static Index Upper Bound

```rust
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::Index;

let reader = index.reader()?;
let searcher = reader.searcher();

let query = query_parser.parse_query("title:tantivy AND body:search")?;

// O(1) global upper bound across all segments
let global_max_score: f32 = query.index_max_score(&searcher)?;
println!("Maximum possible score for query: {}", global_max_score);
```

### Example B: Segment-Level Bound with Non-Matching Pruning

```rust
let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;

for segment_reader in searcher.segment_readers() {
    let seg_max_score = weight.max_score(segment_reader, 1.0)?;
    if seg_max_score == 0.0 {
        // Fast path: None of the required terms exist in this segment.
        // Entire segment can be skipped without opening postings or evaluating filters.
        continue;
    }

    // Use seg_max_score to establish dynamic thresholds for joined relations
    let mut scorer = weight.scorer(segment_reader, 1.0)?;
    // ...
}
```

### Example C: Cross-Table Top-K Join Threshold Calculation

```rust
// Query A runs on Index A, Query B runs on Index B.
// Total score = Score_A + Score_B.
let max_b = query_b.index_max_score(&searcher_b)?;

// If our current K-th global score candidate is current_top_k_score:
let threshold_a = (current_top_k_score - max_b).max(0.0);

// Use threshold_a to prune postings in Index A using PruningScorer / Block-Max WAND:
let mut scorer_a = weight_a.pruning_scorer(segment_reader_a, 1.0, threshold_a)?;
```

---

## 5. Future Roadmap: Dynamic Intra-Segment Bounds (Proposal B)

The current API (Proposal A) provides **static** bounds per query and per segment.

A future extension tracked in `src/query/scorer.rs` (`TODO(Proposal B)`) introduces dynamic remaining maximum score tracking directly on `PruningScorer`:

```rust
pub trait PruningScorer: Scorer {
    fn set_threshold(&mut self, score: Score);

    // Dynamic upper bound of all unvisited documents in the current segment:
    fn max_score(&self) -> Score { Score::MAX }
}
```

### What Proposal B Enables Over Proposal A:
1. **Dynamic Threshold Tightening During Joins**: As an index scan advances past its highest-scoring blocks, its remaining maximum score drops dynamically ($M_B(t) < M_B(0)$). This allows the partner stream's pruning threshold $\theta_A(t) = \theta_{total} - M_B(t)$ to rise in real time, skipping more blocks via Block-Max WAND.
2. **Early Segment Scan Termination**: If $M_B(t) + M_A < \theta_{total}$, the remaining document stream in the segment can be aborted immediately.
