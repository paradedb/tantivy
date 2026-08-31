use std::io::{self, Write};

use super::{
    require_version, EagerRouterRanking, Router, RouterOpenContext, RouterRanking,
    RouterSearchContext,
};
use crate::directory::FileSlice;
use crate::schema::VectorOptions;
use crate::vector::header::VectorFileVersion;
use crate::vector::ivf::graph::{Candidate, Workspace};
use crate::vector::ivf::{
    BuiltRouter, InMemoryStackedIvf, IvfCentroids, IvfConfig as StackedIvfConfig, IvfIndexBuilder,
    LazyStackedIvf, MultiLevelIvf, SuperKMeansLevelClusterer,
};
use crate::vector::VectorArena;

const STACKED_ROUTER_VERSION: u32 = 1;
const STACKED_ROUTER_ID: &str = "tantivy.stacked-ivf";

pub(crate) fn build_default_stacked_router(
    options: &VectorOptions,
    centroids: &IvfCentroids,
) -> BuiltRouter {
    let IvfCentroids::F32(matrix) = centroids;
    let clusterer = SuperKMeansLevelClusterer::default();
    let (index, permutation) = IvfIndexBuilder::new(
        matrix.values.clone(),
        matrix.rows,
        options.dim(),
        &clusterer,
        StackedIvfConfig::default(),
    )
    .build();
    BuiltRouter::new(index).with_centroid_permutation(permutation)
}

fn deserialize_stacked_router(
    format_version: u32,
    payload: FileSlice,
    context: &RouterOpenContext,
) -> crate::Result<Box<dyn Router>> {
    require_version(STACKED_ROUTER_ID, format_version, STACKED_ROUTER_VERSION)?;
    Ok(Box::new(LazyStackedIvf::open(
        payload,
        context.centroids().clone(),
        context.options().dim(),
        StackedIvfConfig::default(),
    )?))
}

fn rank_stacked<'a, C, M>(
    index: &MultiLevelIvf<C, M>,
    query: &[f32],
    context: RouterSearchContext,
) -> Box<dyn RouterRanking + 'a>
where
    C: VectorArena<Elem = f32>,
    M: VectorArena<Elem = f32>,
{
    let ranked = index
        .search(query, context.num_centroids(), 1.0, context.metric())
        .into_iter()
        .map(|candidate| Candidate {
            sim: candidate.sim,
            node: candidate.node.0,
        })
        .collect::<Vec<_>>();
    let visited_count = ranked.len();
    Box::new(EagerRouterRanking::new(ranked, visited_count))
}

impl Router for InMemoryStackedIvf {
    fn id(&self) -> &'static str {
        STACKED_ROUTER_ID
    }

    fn vector_file_version(&self) -> VectorFileVersion {
        VectorFileVersion::V3
    }

    fn format_version(&self) -> u32 {
        STACKED_ROUTER_VERSION
    }

    fn deserialize(
        format_version: u32,
        payload: FileSlice,
        context: &RouterOpenContext,
    ) -> crate::Result<Box<dyn Router>> {
        deserialize_stacked_router(format_version, payload, context)
    }

    fn rank<'a>(
        &'a self,
        _workspace: &'a mut Workspace,
        query: &'a [f32],
        context: RouterSearchContext,
    ) -> Box<dyn RouterRanking + 'a> {
        rank_stacked(self, query, context)
    }

    fn serialize_payload(&self, out: &mut dyn Write) -> io::Result<()> {
        self.serialize_router_payload(out)
    }
}

impl Router for LazyStackedIvf {
    fn id(&self) -> &'static str {
        STACKED_ROUTER_ID
    }

    fn vector_file_version(&self) -> VectorFileVersion {
        VectorFileVersion::V3
    }

    fn format_version(&self) -> u32 {
        STACKED_ROUTER_VERSION
    }

    fn deserialize(
        format_version: u32,
        payload: FileSlice,
        context: &RouterOpenContext,
    ) -> crate::Result<Box<dyn Router>> {
        deserialize_stacked_router(format_version, payload, context)
    }

    fn rank<'a>(
        &'a self,
        _workspace: &'a mut Workspace,
        query: &'a [f32],
        context: RouterSearchContext,
    ) -> Box<dyn RouterRanking + 'a> {
        rank_stacked(self, query, context)
    }

    fn serialize_payload(&self, out: &mut dyn Write) -> io::Result<()> {
        self.serialize_router_payload(out)
    }
}
