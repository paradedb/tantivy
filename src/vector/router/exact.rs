//! The exact router: scores every centroid against the query and ranks them all.

use std::io::{self, Write};

use common::HasLen;

use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::clustering::IvfCentroids;
use crate::vector::router::arena::VectorArena;
use crate::vector::router::candidate::Candidate;
use crate::vector::router::stacked::levels::{InMemoryStore, LazyStore};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::VectorElement;

pub(crate) struct ExactRouter<S> {
    centroids: S,
    num_centroids: usize,
    dim: usize,
    metric: Metric,
}

impl<S: VectorArena<Elem = f32>> ExactRouter<S> {
    pub(super) fn rank<T: VectorElement>(&self, query: &VectorQuery<T>) -> Ranking {
        assert_eq!(query.widened().len(), self.dim);
        debug_assert_eq!(query.metric(), self.metric);
        let mut scores = Vec::with_capacity(self.num_centroids);
        self.centroids.batch_similarity(
            query.widened(),
            query.metric(),
            query.routing_norm(),
            0..self.num_centroids,
            &mut scores,
        );
        let mut ranked = scores
            .into_iter()
            .enumerate()
            .map(|(cluster, sim)| Candidate {
                sim,
                node: cluster as u32,
            })
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|a, b| b.cmp(a));
        Ranking {
            ranked: ranked.into_iter(),
            visited_count: self.num_centroids,
        }
    }

    pub(super) fn serialize_payload<W: Write + ?Sized>(&self, _out: &mut W) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn build(
    options: &VectorOptions,
    centroids: &IvfCentroids,
) -> ExactRouter<InMemoryStore> {
    let IvfCentroids::F32(matrix) = centroids;
    ExactRouter {
        centroids: InMemoryStore::new(matrix.values.clone(), options.dim()),
        num_centroids: matrix.rows,
        dim: options.dim(),
        metric: options.metric(),
    }
}

pub(super) fn open(
    payload: FileSlice,
    centroids: FileSlice,
    options: &VectorOptions,
) -> crate::Result<ExactRouter<LazyStore>> {
    if !payload.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "exact router payload must be empty",
        )
        .into());
    }
    let num_centroids = centroids
        .len()
        .checked_div(options.bytes_per_vector())
        .unwrap_or(0);
    Ok(ExactRouter {
        centroids: LazyStore::new(centroids, options.dim()),
        num_centroids,
        dim: options.dim(),
        metric: options.metric(),
    })
}

pub(crate) struct Ranking {
    pub(super) ranked: std::vec::IntoIter<Candidate>,
    visited_count: usize,
}

impl Ranking {
    pub(super) fn visited_count(&self) -> usize {
        self.visited_count
    }
}

impl Iterator for Ranking {
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        self.ranked.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::clustering::IvfMatrix;

    #[test]
    fn batch_ranking_matches_single_row_scoring() -> crate::Result<()> {
        let matrix: IvfMatrix<f32> = IvfMatrix {
            values: vec![
                1.0, -2.0, 3.0, 0.0, 0.0, 0.0, -4.0, 5.0, -6.0, 1.0, -2.0, 3.0,
            ],
            rows: 4,
            dims: 3,
        };
        let bytes: Vec<_> = matrix.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let options = VectorOptions::new(matrix.dims, metric);
            let router = build(&options, &IvfCentroids::F32(matrix.clone()));
            let opened = open(
                FileSlice::from(Vec::new()),
                FileSlice::from(bytes.clone()),
                &options,
            )?;
            for query in [[1.5, -0.5, 2.0], [0.0, 0.0, 0.0]] {
                let mut expected: Vec<_> = matrix
                    .values
                    .chunks_exact(matrix.dims)
                    .enumerate()
                    .map(|(node, row)| Candidate {
                        sim: metric.similarity(&query, row),
                        node: node as u32,
                    })
                    .collect();
                expected.sort_unstable_by(|a, b| b.cmp(a));
                let query = VectorQuery::new(metric, std::sync::Arc::new(query.to_vec()));
                assert_eq!(router.rank(&query).collect::<Vec<_>>(), expected);
                assert_eq!(opened.rank(&query).collect::<Vec<_>>(), expected);
            }
        }
        Ok(())
    }
}
