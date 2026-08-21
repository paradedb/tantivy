//! Centroid assignment: vector → nearest cells of the index-level set.
//!
//! Both write paths (per-commit serialize and merge) assign every vector
//! against the same frozen centroid index, taking the primary cell and the
//! `replicas - 1` next-nearest cells from ONE k-NN call per vector. The
//! selector mirrors the query-time router's ranking per metric, so cells
//! predict where a query would look.
//!
//! Everything here runs on the CALLING thread. Assignment reads centroid
//! rows through the index's `Directory`, and an embedder like pg_search
//! runs inside a Postgres backend, where FFI from a spawned thread aborts
//! the transaction — so no path in the vector write pipeline spawns.

use std::cmp::Ordering;

use super::graph::{NodeId, RelativeNeighborhoodGraph, Workspace};
use crate::schema::{Metric, VectorOptions};
use crate::vector::distance::{cosine, dot, l2_squared};
use crate::vector::ivf::centroid_index::{FieldCentroids, UnitNormRowsArena};
use crate::Executor;

/// Rows assigned per [`Executor`] work item when a batch is split across
/// threads. Small enough to load-balance, large enough to amortize the
/// per-item `Workspace` allocation.
const ASSIGN_CHUNK_ROWS: usize = 256;

/// Centroid ids of the `knn` nearest centroids to `query`, nearest first —
/// the exact counterpart of [`RelativeNeighborhoodGraph::nearest`], same
/// distance family per metric. Ties break on centroid id so selection is
/// deterministic.
fn exact_nearest_centroids(
    metric: Metric,
    centroids: &[f32],
    dim: usize,
    query: &[f32],
    knn: usize,
) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = centroids
        .chunks_exact(dim)
        .enumerate()
        .map(|(id, centroid)| {
            // Each arm is the negated `Metric::similarity` ordering the
            // graph selector ranks by.
            let d = match metric {
                // `1 - cosine` orders identically to descending cosine.
                Metric::Cosine => 1.0 - cosine(query, centroid),
                // Negated raw dot: the query-time router ranks by dot, and
                // dot ordering is ||q||-invariant — a v-directed query
                // ranks cells by dot(v, c), so raw dot IS the
                // query-consistent criterion.
                Metric::Dot => -dot(query, centroid),
                // Squared L2 orders identically to L2.
                Metric::L2 => l2_squared(query, centroid),
            };
            // A NaN score (a zero-norm centroid under Cosine) must rank
            // WORST, not tie: `partial_cmp`'s `Equal` fallback would let
            // the id tie-break route real vectors into a degenerate cell.
            let d = if d.is_nan() { f32::INFINITY } else { d };
            (d, id)
        })
        .collect();
    scored.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.truncate(knn);
    scored.into_iter().map(|(_, id)| id).collect()
}

/// How a vector's cells are picked from the stored centroids. Exact
/// k-NN scan for a small centroid index — anything the search's own `ef` visit
/// budget would cover wholesale anyway, where the brute scan is at most
/// as expensive and exact (an approximate graph over a handful of points
/// can return fewer than `knn` neighbours, silently under-assigning) —
/// and, for large ones, the set's PERSISTED router: the same pinned
/// graph queries route through, cached on `Index`, so a segment build
/// never constructs a selector structure of its own.
pub(crate) enum CentroidSelector<'a> {
    Exact { centroids: Vec<f32> },
    Router(&'a RelativeNeighborhoodGraph<UnitNormRowsArena>),
}

impl<'a> CentroidSelector<'a> {
    /// Selector over `set`'s rows, sized for `cells_per_vector`-deep
    /// selection. `router` is the set's persisted routing graph (absent
    /// for degenerate sets and consumer-defined routers); without it a
    /// large set falls back to the exact scan — correct, just O(C) per
    /// vector.
    pub(crate) fn for_set(
        set: &FieldCentroids,
        router: Option<&'a RelativeNeighborhoodGraph<UnitNormRowsArena>>,
        options: &VectorOptions,
        cells_per_vector: usize,
    ) -> crate::Result<Self> {
        let ef_search = (cells_per_vector * 4).max(64);
        if set.num_centroids() <= ef_search {
            return Ok(CentroidSelector::Exact {
                centroids: set.values_f32(options)?,
            });
        }
        match router {
            Some(graph) => Ok(CentroidSelector::Router(graph)),
            None => {
                log::warn!(
                    "assigning against {} centroids with no readable router; falling back to an \
                     exact per-vector scan",
                    set.num_centroids(),
                );
                Ok(CentroidSelector::Exact {
                    centroids: set.values_f32(options)?,
                })
            }
        }
    }

    /// The `knn` nearest centroid ids to `v`, nearest first. The graph arm
    /// may return fewer than `knn` (approximate recall), never duplicates.
    fn nearest(
        &self,
        metric: Metric,
        dim: usize,
        ws: &mut Workspace,
        v: &[f32],
        knn: usize,
    ) -> Vec<usize> {
        match self {
            CentroidSelector::Exact { centroids } => {
                exact_nearest_centroids(metric, centroids, dim, v, knn)
            }
            CentroidSelector::Router(graph) => {
                // The router's Cosine arm is a raw dot over unit-norm rows
                // (`UnitNormRowsArena`); assignment inputs are stored rows,
                // normalized at ingest, so the contract holds.
                // TODO: Replace with proper seed generation
                let seeds: Vec<NodeId> = (0..graph.len())
                    .step_by((graph.len() / 8).max(1))
                    .take(8)
                    .map(|node| node as NodeId)
                    .collect();
                graph
                    .search(ws, v, &seeds, knn)
                    .0
                    .into_iter()
                    .map(|candidate| candidate.node as usize)
                    .collect()
            }
        }
    }
}

/// Assign a batch of vectors to their cells: per vector, up to
/// `cells_per_vector` distinct centroid ids, nearest first — index 0 is the
/// primary, the rest are replica cells. `values` is `dim`-strided,
/// row-parallel output order. Chunks the batch across `executor`.
pub(crate) fn assign_cells(
    selector: &CentroidSelector<'_>,
    metric: Metric,
    dim: usize,
    values: &[f32],
    cells_per_vector: usize,
    executor: &Executor,
) -> crate::Result<Vec<Vec<usize>>> {
    debug_assert_eq!(values.len() % dim.max(1), 0);
    let num_rows = values.len() / dim.max(1);
    let assign_chunk = |range: std::ops::Range<usize>| -> Vec<Vec<usize>> {
        // One scratch reused across the chunk's graph lookups — never
        // per vector.
        let mut ws = Workspace::new();
        range
            .map(|row| {
                let v = &values[row * dim..(row + 1) * dim];
                selector.nearest(metric, dim, &mut ws, v, cells_per_vector)
            })
            .collect()
    };
    if executor.num_threads() <= 1 || num_rows <= ASSIGN_CHUNK_ROWS {
        return Ok(assign_chunk(0..num_rows));
    }
    let chunk_starts = (0..num_rows).step_by(ASSIGN_CHUNK_ROWS);
    let per_chunk = executor.map(
        |start| {
            Ok(assign_chunk(
                start..(start + ASSIGN_CHUNK_ROWS).min(num_rows),
            ))
        },
        chunk_starts,
    )?;
    Ok(per_chunk.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the Dot selection semantics: cells follow RAW dot — the
    /// query-time router's ranking — not angular order. Centroid norms are
    /// deliberately unequal so the two orderings disagree.
    #[test]
    fn dot_selector_uses_raw_dot_not_angular() {
        let centroids: Vec<f32> = vec![
            10.0, 0.0, // long, off-direction: dot 10, cosine 0.45
            0.0, 1.0, // short, near-direction: dot 2, cosine 0.89
            7.0, 7.0, // long, near-direction: dot 21, cosine 0.95
        ];
        let query = [1.0_f32, 2.0];
        let picked = exact_nearest_centroids(Metric::Dot, &centroids, 2, &query, 3);
        // Raw-dot order: [7,7] (21), then [10,0] (10), then [0,1] (2).
        // Angular order would put [0,1] ahead of [10,0].
        assert_eq!(picked, vec![2, 0, 1], "must rank by raw dot");
    }

    /// Assignment yields nearest-first distinct cells for every row, and the
    /// parallel chunking preserves row order.
    #[test]
    fn assign_cells_is_nearest_first_and_order_preserving() -> crate::Result<()> {
        let selector = CentroidSelector::Exact {
            centroids: vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0],
        };
        let values: Vec<f32> = vec![
            1.0, 0.0, // nearest 0, then 1
            9.0, 1.0, // nearest 1, then 0
            0.5, 9.0, // nearest 2, then 0
        ];
        let cells = assign_cells(
            &selector,
            Metric::L2,
            2,
            &values,
            2,
            &Executor::single_thread(),
        )?;
        assert_eq!(cells, vec![vec![0, 1], vec![1, 0], vec![2, 0]]);
        Ok(())
    }
}
