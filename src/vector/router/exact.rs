use std::io::{self, Write};

use common::HasLen;

use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::ivf::{Candidate, InMemoryStore, LazyStore};
use crate::vector::{IvfCentroids, VectorArena};

pub(crate) struct ExactRouter<S> {
    centroids: S,
    num_centroids: usize,
    dim: usize,
    metric: Metric,
}

impl<S: VectorArena<Elem = f32>> ExactRouter<S> {
    pub(super) fn rank(&self, query: &[f32]) -> Ranking {
        let mut ranked = (0..self.num_centroids)
            .map(|cluster| Candidate {
                sim: self
                    .centroids
                    .similarity(self.metric, self.dim, cluster as u32, query),
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
    ranked: std::vec::IntoIter<Candidate>,
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
