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
    InMemoryStackedIvf, IvfCentroids, IvfConfig as StackedIvfConfig, IvfIndexBuilder,
    LazyStackedIvf, MultiLevelIvf, SuperKMeansLevelClusterer,
};
use crate::vector::VectorArena;
use crate::TantivyError;

const STACKED_ROUTER_VERSION: u32 = 1;
const STACKED_ROUTER_ID: &str = "tantivy.stacked-ivf";

fn build_stacked_router(
    options: &VectorOptions,
    centroids: &mut IvfCentroids,
) -> crate::Result<Box<dyn Router>> {
    let IvfCentroids::F32(matrix) = &*centroids;
    let clusterer = SuperKMeansLevelClusterer::default();
    let (index, permutation) = IvfIndexBuilder::new(
        matrix.values.clone(),
        matrix.rows,
        options.dim(),
        &clusterer,
        StackedIvfConfig::default(),
    )
    .build();
    let IvfCentroids::F32(matrix) = centroids;
    if permutation.len() != matrix.rows {
        return Err(TantivyError::InvalidArgument(format!(
            "stacked router returned a permutation over {} centroids, expected {}",
            permutation.len(),
            matrix.rows
        )));
    }
    let mut values = vec![0.0f32; matrix.values.len()];
    let mut seen = vec![false; matrix.rows];
    for (old, &new) in permutation.iter().enumerate() {
        let new = new as usize;
        if new >= matrix.rows || seen[new] {
            return Err(TantivyError::InvalidArgument(
                "stacked router centroid permutation is not a bijection".to_string(),
            ));
        }
        seen[new] = true;
        values[new * matrix.dims..(new + 1) * matrix.dims]
            .copy_from_slice(&matrix.values[old * matrix.dims..(old + 1) * matrix.dims]);
    }
    matrix.values = values;
    Ok(Box::new(index))
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
    fn build_router(
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Box<dyn Router>> {
        build_stacked_router(options, centroids)
    }

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
    fn build_router(
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Box<dyn Router>> {
        build_stacked_router(options, centroids)
    }

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
