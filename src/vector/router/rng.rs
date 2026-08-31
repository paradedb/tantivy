use std::io::{self, Write};

use super::{
    open_enveloped_router, require_version, IvfSearchMetrics, Router, RouterOpenContext,
    RouterRanking, RouterSearchContext,
};
use crate::directory::FileSlice;
use crate::schema::VectorDType;
use crate::vector::header::VectorFileVersion;
use crate::vector::ivf::graph::{
    NeighborhoodGraphConfig, RelativeNeighborhoodGraph, SearchIterator, Workspace,
};
use crate::vector::{FileSliceArena, VectorArena};

const GRAPH_ROUTER_VERSION: u32 = 1;
const GRAPH_ROUTER_ID: &str = "tantivy.relative-neighborhood-graph";

impl<S> Router for RelativeNeighborhoodGraph<S>
where S: VectorArena<Elem = f32> + Send + Sync
{
    fn id(&self) -> &'static str {
        GRAPH_ROUTER_ID
    }

    fn vector_file_version(&self) -> VectorFileVersion {
        VectorFileVersion::V3
    }

    fn format_version(&self) -> u32 {
        GRAPH_ROUTER_VERSION
    }

    fn deserialize(
        format_version: u32,
        payload: FileSlice,
        context: &RouterOpenContext,
    ) -> crate::Result<Box<dyn Router>> {
        require_version(GRAPH_ROUTER_ID, format_version, GRAPH_ROUTER_VERSION)?;
        let vectors = match context.options().dtype() {
            VectorDType::F32 => FileSliceArena::<f32>::new(context.centroids().clone()),
        };
        let adjacency = payload.read_bytes()?;
        Ok(Box::new(RelativeNeighborhoodGraph::open(
            &adjacency,
            vectors,
            context.options().dim(),
            context.options().metric(),
            NeighborhoodGraphConfig::default(),
        )?))
    }

    fn open_router(
        file_version: VectorFileVersion,
        slot: FileSlice,
        context: &RouterOpenContext,
    ) -> crate::Result<Box<dyn Router>> {
        if file_version == VectorFileVersion::V2 {
            return Self::deserialize(GRAPH_ROUTER_VERSION, slot, context);
        }
        open_enveloped_router::<Self>(file_version, slot, context)
    }

    fn rank<'a>(
        &'a self,
        workspace: &'a mut Workspace,
        query: &'a [f32],
        _context: RouterSearchContext,
    ) -> Box<dyn RouterRanking + 'a> {
        let seeds = (0..self.len())
            .step_by((self.len() / 8).max(1))
            .take(8)
            .map(|node| node as u32)
            .collect::<Vec<_>>();
        Box::new(self.search_iter(workspace, query, &seeds))
    }

    fn serialize_payload(&self, out: &mut dyn Write) -> io::Result<()> {
        self.serialize_adjacency(out)
    }
}

impl<S: VectorArena> RouterRanking for SearchIterator<'_, '_, S, true> {
    fn metrics(&self) -> IvfSearchMetrics {
        let graph = SearchIterator::metrics(self);
        IvfSearchMetrics {
            visited_count: graph.visited_count,
            graph: Some(graph),
        }
    }
}
