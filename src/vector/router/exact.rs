use std::io::{self, Write};

use common::HasLen;

use super::{
    require_version, EagerRouterRanking, Router, RouterOpenContext, RouterRanking,
    RouterSearchContext,
};
use crate::directory::FileSlice;
use crate::schema::Metric;
use crate::vector::header::VectorFileVersion;
use crate::vector::ivf::graph::{Candidate, Workspace};
use crate::vector::{BuiltRouter, FileSliceArena, IvfCentroids, VectorArena, VectorOptions};

const EXACT_ROUTER_ID: &str = "tantivy.exact";
const EXACT_ROUTER_VERSION: u32 = 1;

pub(crate) struct ExactRouter<S> {
    centroids: S,
    num_centroids: usize,
    dim: usize,
    metric: Metric,
}

pub(crate) type LazyExactRouter = ExactRouter<FileSliceArena<f32>>;

impl ExactRouter<FileSliceArena<f32>> {
    pub(crate) fn new(context: &RouterOpenContext) -> Self {
        Self {
            centroids: FileSliceArena::new(context.centroids().clone()),
            num_centroids: context
                .centroids()
                .len()
                .checked_div(context.options().bytes_per_vector())
                .unwrap_or(0),
            dim: context.options().dim(),
            metric: context.options().metric(),
        }
    }
}

impl ExactRouter<Vec<f32>> {
    fn from_centroids(options: &VectorOptions, centroids: &IvfCentroids) -> Self {
        let IvfCentroids::F32(matrix) = centroids;
        Self {
            centroids: matrix.values.clone(),
            num_centroids: matrix.rows,
            dim: options.dim(),
            metric: options.metric(),
        }
    }
}

impl<S> Router for ExactRouter<S>
where
    S: VectorArena<Elem = f32> + Send + Sync + 'static,
{
    fn build_router(
        options: &VectorOptions,
        centroids: &IvfCentroids,
    ) -> crate::Result<BuiltRouter> {
        Ok(BuiltRouter::new(ExactRouter::<Vec<f32>>::from_centroids(
            options, centroids,
        )))
    }

    fn id(&self) -> &'static str {
        EXACT_ROUTER_ID
    }

    fn vector_file_version(&self) -> VectorFileVersion {
        VectorFileVersion::V3
    }

    fn format_version(&self) -> u32 {
        EXACT_ROUTER_VERSION
    }

    fn deserialize(
        format_version: u32,
        payload: FileSlice,
        context: &RouterOpenContext,
    ) -> crate::Result<Box<dyn Router>> {
        require_version(EXACT_ROUTER_ID, format_version, EXACT_ROUTER_VERSION)?;
        if payload.len() != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "exact router payload must be empty",
            )
            .into());
        }
        Ok(Box::new(LazyExactRouter::new(context)))
    }

    fn rank<'a>(
        &'a self,
        _workspace: &'a mut Workspace,
        query: &'a [f32],
        _context: RouterSearchContext,
    ) -> Box<dyn RouterRanking + 'a> {
        let mut ranked = (0..self.num_centroids)
            .map(|cluster| Candidate {
                sim: self
                    .centroids
                    .similarity(self.metric, self.dim, cluster as u32, query),
                node: cluster as u32,
            })
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|a, b| b.cmp(a));
        Box::new(EagerRouterRanking::new(ranked, self.num_centroids))
    }

    fn serialize_payload(&self, _out: &mut dyn Write) -> io::Result<()> {
        Ok(())
    }
}
