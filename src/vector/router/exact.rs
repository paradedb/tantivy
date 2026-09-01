use std::io::{self, Write};

use common::HasLen;

use super::{Router, RouterDescriptor};
use crate::directory::FileSlice;
use crate::schema::Metric;
use crate::vector::header::VectorFileVersion;
use crate::vector::ivf::graph::Candidate;
use crate::vector::{FileSliceArena, IvfCentroids, VectorArena, VectorOptions};

const EXACT_ROUTER_ID: &str = "tantivy.exact";

pub struct ExactRouter<S> {
    centroids: S,
    num_centroids: usize,
    dim: usize,
    metric: Metric,
}

pub type LazyExactRouter = ExactRouter<FileSliceArena<f32>>;

impl<S> Router for ExactRouter<S>
where S: VectorArena<Elem = f32> + Send + Sync + 'static
{
    fn router_descriptor() -> RouterDescriptor {
        RouterDescriptor::new(EXACT_ROUTER_ID, VectorFileVersion::V3)
    }

    fn build_router(
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Box<dyn Router>> {
        let IvfCentroids::F32(matrix) = centroids;
        Ok(Box::new(ExactRouter::<Vec<f32>> {
            centroids: matrix.values.clone(),
            num_centroids: matrix.rows,
            dim: options.dim(),
            metric: options.metric(),
        }))
    }

    fn deserialize(
        payload: FileSlice,
        centroids: FileSlice,
        options: &VectorOptions,
    ) -> crate::Result<Box<dyn Router>> {
        if payload.len() != 0 {
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
        Ok(Box::new(LazyExactRouter {
            centroids: FileSliceArena::new(centroids),
            num_centroids,
            dim: options.dim(),
            metric: options.metric(),
        }))
    }

    fn rank<'a>(
        &'a self,
        query: &'a [f32],
        _metric: Metric,
    ) -> Box<dyn Iterator<Item = Candidate> + 'a> {
        let mut ranked = (0..self.num_centroids)
            .map(|cluster| Candidate {
                sim: self
                    .centroids
                    .similarity(self.metric, self.dim, cluster as u32, query),
                node: cluster as u32,
            })
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|a, b| b.cmp(a));
        Box::new(ranked.into_iter())
    }

    fn serialize_payload(&self, _out: &mut dyn Write) -> io::Result<()> {
        Ok(())
    }
}
