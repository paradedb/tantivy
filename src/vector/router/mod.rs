use std::io::{self, Write};
use std::marker::PhantomData;
use std::sync::Arc;

use common::{BinarySerializable, HasLen};

use super::ivf::graph::{Candidate, NeighborhoodGraphSearchMetrics, Workspace};
use super::ivf::IvfCentroids;
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::header::VectorFileVersion;

mod exact;
mod rng;
mod stacked;

pub(crate) use exact::LazyExactRouter;

const ROUTER_MAGIC: &[u8; 8] = b"TVROUTER";
const ROUTER_HEADER_LEN: usize = ROUTER_MAGIC.len() + size_of::<u16>() + size_of::<u32>();

pub struct RouterOpenContext {
    centroids: FileSlice,
    options: VectorOptions,
}

impl RouterOpenContext {
    pub(crate) fn new(centroids: FileSlice, options: VectorOptions) -> Self {
        Self { centroids, options }
    }

    pub fn centroids(&self) -> &FileSlice {
        &self.centroids
    }

    pub fn options(&self) -> &VectorOptions {
        &self.options
    }
}

#[derive(Clone, Copy)]
pub struct RouterSearchContext {
    num_centroids: usize,
    metric: Metric,
}

impl RouterSearchContext {
    pub(crate) fn new(num_centroids: usize, metric: Metric) -> Self {
        Self {
            num_centroids,
            metric,
        }
    }

    pub fn num_centroids(self) -> usize {
        self.num_centroids
    }

    pub fn metric(self) -> Metric {
        self.metric
    }
}

/// Builds and routes an IVF index and owns its persisted format.
///
/// `serialize` wraps the router payload in an envelope containing its ID and
/// format version. `open_router` removes and validates that envelope before
/// delegating the payload to `deserialize`.
pub trait Router: Send + Sync {
    /// Builds a router over the supplied centroids, which may be empty.
    /// Implementations may reorder rows in place but must preserve the matrix shape.
    fn build_router(
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Box<dyn Router>>
    where
        Self: Sized;

    fn id(&self) -> &'static str;

    fn vector_file_version(&self) -> VectorFileVersion;

    fn format_version(&self) -> u32;

    fn deserialize(
        format_version: u32,
        payload: FileSlice,
        context: &RouterOpenContext,
    ) -> crate::Result<Box<dyn Router>>
    where
        Self: Sized;

    fn open_router(
        file_version: VectorFileVersion,
        slot: FileSlice,
        context: &RouterOpenContext,
    ) -> crate::Result<Box<dyn Router>>
    where
        Self: Sized,
    {
        open_enveloped_router::<Self>(file_version, slot, context)
    }

    fn rank<'a>(
        &'a self,
        workspace: &'a mut Workspace,
        query: &'a [f32],
        context: RouterSearchContext,
    ) -> Box<dyn RouterRanking + 'a>;

    fn serialize_payload(&self, out: &mut dyn Write) -> io::Result<()>;

    fn serialize(&self, out: &mut dyn Write) -> io::Result<()> {
        write_router_header(self.id(), self.format_version(), out)?;
        self.serialize_payload(out)
    }
}

pub trait RouterFactory: Send + Sync {
    fn build(
        &self,
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Box<dyn Router>>;

    fn open(
        &self,
        file_version: VectorFileVersion,
        slot: FileSlice,
        context: &RouterOpenContext,
    ) -> crate::Result<Box<dyn Router>>;
}

struct RouterFactoryFor<R>(PhantomData<fn() -> R>);

impl<R> RouterFactoryFor<R> {
    fn new() -> Self {
        Self(PhantomData)
    }
}

impl<R: Router + 'static> RouterFactory for RouterFactoryFor<R> {
    fn build(
        &self,
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Box<dyn Router>> {
        R::build_router(options, centroids)
    }

    fn open(
        &self,
        file_version: VectorFileVersion,
        slot: FileSlice,
        context: &RouterOpenContext,
    ) -> crate::Result<Box<dyn Router>> {
        R::open_router(file_version, slot, context)
    }
}

pub(crate) fn router_factory_for<R: Router + 'static>() -> Arc<dyn RouterFactory> {
    Arc::new(RouterFactoryFor::<R>::new())
}

pub trait RouterRanking: Iterator<Item = Candidate> {
    fn metrics(&self) -> IvfSearchMetrics;
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct IvfSearchMetrics {
    pub visited_count: usize,
    pub graph: Option<NeighborhoodGraphSearchMetrics>,
}

fn write_router_header(id: &str, version: u32, out: &mut dyn Write) -> io::Result<()> {
    let id_len = u16::try_from(id.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "router ID exceeds u16"))?;
    out.write_all(ROUTER_MAGIC)?;
    id_len.serialize(out)?;
    version.serialize(out)?;
    out.write_all(id.as_bytes())
}

fn open_enveloped_router<R: Router>(
    file_version: VectorFileVersion,
    slot: FileSlice,
    context: &RouterOpenContext,
) -> crate::Result<Box<dyn Router>> {
    let (id, format_version, payload_offset) = read_router_header(&slot)?;
    let router = R::deserialize(format_version, slot.slice_from(payload_offset), context)?;
    if router.id() != id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "router opener {} does not match persisted router {id}",
                router.id()
            ),
        )
        .into());
    }
    if router.vector_file_version() != file_version {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "router {} requires vector file version {:?}, found {:?}",
                router.id(),
                router.vector_file_version(),
                file_version
            ),
        )
        .into());
    }
    Ok(router)
}

fn read_router_header(slot: &FileSlice) -> crate::Result<(String, u32, usize)> {
    if slot.len() < ROUTER_HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "router slot is shorter than its header",
        )
        .into());
    }
    let header = slot.slice_to(ROUTER_HEADER_LEN).read_bytes()?;
    if &header[..ROUTER_MAGIC.len()] != ROUTER_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid router magic").into());
    }
    let mut cursor = &header[ROUTER_MAGIC.len()..];
    let id_len = u16::deserialize(&mut cursor)? as usize;
    let version = u32::deserialize(&mut cursor)?;
    let payload_offset = ROUTER_HEADER_LEN
        .checked_add(id_len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "router header overflow"))?;
    if payload_offset > slot.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "router slot is shorter than its declared ID",
        )
        .into());
    }
    let id_bytes = slot.slice(ROUTER_HEADER_LEN..payload_offset).read_bytes()?;
    let id = std::str::from_utf8(&id_bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "router ID is not UTF-8"))?
        .to_owned();
    Ok((id, version, payload_offset))
}

fn require_version(id: &str, actual: u32, expected: u32) -> crate::Result<()> {
    if actual != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported {id} router format version: {actual}"),
        )
        .into());
    }
    Ok(())
}

struct EagerRouterRanking {
    ranked: std::vec::IntoIter<Candidate>,
    visited_count: usize,
}

impl EagerRouterRanking {
    fn new(ranked: Vec<Candidate>, visited_count: usize) -> Self {
        Self {
            ranked: ranked.into_iter(),
            visited_count,
        }
    }
}

impl Iterator for EagerRouterRanking {
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        self.ranked.next()
    }
}

impl RouterRanking for EagerRouterRanking {
    fn metrics(&self) -> IvfSearchMetrics {
        IvfSearchMetrics {
            visited_count: self.visited_count,
            graph: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::vector::Similarity;

    struct TestRouter {
        cluster: u32,
    }

    impl Router for TestRouter {
        fn build_router(
            _options: &VectorOptions,
            _centroids: &mut IvfCentroids,
        ) -> crate::Result<Box<dyn Router>> {
            Ok(Box::new(TestRouter { cluster: 0 }))
        }

        fn id(&self) -> &'static str {
            "test.router"
        }

        fn vector_file_version(&self) -> VectorFileVersion {
            VectorFileVersion::V3
        }

        fn format_version(&self) -> u32 {
            7
        }

        fn deserialize(
            format_version: u32,
            payload: FileSlice,
            _context: &RouterOpenContext,
        ) -> crate::Result<Box<dyn Router>> {
            require_version("test.router", format_version, 7)?;
            let bytes = payload.read_bytes()?;
            let mut cursor = bytes.as_slice();
            let cluster = u32::deserialize(&mut cursor)?;
            if !cursor.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "test router payload has trailing bytes",
                )
                .into());
            }
            Ok(Box::new(TestRouter { cluster }))
        }

        fn rank<'a>(
            &'a self,
            _workspace: &'a mut Workspace,
            _query: &'a [f32],
            _context: RouterSearchContext,
        ) -> Box<dyn RouterRanking + 'a> {
            Box::new(EagerRouterRanking::new(
                vec![Candidate {
                    sim: Similarity::new(1.0),
                    node: self.cluster,
                }],
                1,
            ))
        }

        fn serialize_payload(&self, out: &mut dyn Write) -> io::Result<()> {
            self.cluster.serialize(out)
        }
    }

    struct CountingRouterFactory {
        opens: Arc<AtomicUsize>,
    }

    impl RouterFactory for CountingRouterFactory {
        fn build(
            &self,
            options: &VectorOptions,
            centroids: &mut IvfCentroids,
        ) -> crate::Result<Box<dyn Router>> {
            TestRouter::build_router(options, centroids)
        }

        fn open(
            &self,
            file_version: VectorFileVersion,
            slot: FileSlice,
            context: &RouterOpenContext,
        ) -> crate::Result<Box<dyn Router>> {
            self.opens.fetch_add(1, Ordering::Relaxed);
            TestRouter::open_router(file_version, slot, context)
        }
    }

    fn test_context() -> RouterOpenContext {
        RouterOpenContext::new(FileSlice::empty(), VectorOptions::new(1, Metric::L2))
    }

    #[test]
    fn caller_selected_router_roundtrips_without_core_dispatch() -> crate::Result<()> {
        let router = TestRouter { cluster: 42 };
        let mut bytes = Vec::new();
        router.serialize(&mut bytes)?;

        let slot = FileSlice::from(bytes);
        let opened = router_factory_for::<TestRouter>().open(
            VectorFileVersion::V3,
            slot,
            &test_context(),
        )?;
        assert_eq!(opened.id(), "test.router");
        assert_eq!(opened.vector_file_version(), VectorFileVersion::V3);
        assert_eq!(opened.format_version(), 7);

        let mut workspace = Workspace::new();
        let mut ranked = opened.rank(
            &mut workspace,
            &[0.0],
            RouterSearchContext::new(1, Metric::L2),
        );
        assert_eq!(ranked.next().map(|candidate| candidate.node), Some(42));
        assert_eq!(ranked.metrics().visited_count, 1);
        Ok(())
    }

    #[test]
    fn stateful_factory_opens_the_selected_router() -> crate::Result<()> {
        let router = TestRouter { cluster: 42 };
        let mut bytes = Vec::new();
        router.serialize(&mut bytes)?;

        let opens = Arc::new(AtomicUsize::new(0));
        let factory: Arc<dyn RouterFactory> = Arc::new(CountingRouterFactory {
            opens: opens.clone(),
        });
        let opened = factory.open(
            VectorFileVersion::V3,
            FileSlice::from(bytes),
            &test_context(),
        )?;

        assert_eq!(opened.id(), "test.router");
        assert_eq!(opens.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn selected_router_rejects_a_different_envelope_id() {
        let mut bytes = Vec::new();
        write_router_header("test.missing-router", 7, &mut bytes).unwrap();
        42u32.serialize(&mut bytes).unwrap();
        let error = TestRouter::open_router(
            VectorFileVersion::V3,
            FileSlice::from(bytes),
            &test_context(),
        )
        .err()
        .expect("a different router ID must fail");
        assert!(error.to_string().contains("test.missing-router"));
    }

    #[test]
    fn selected_router_rejects_incompatible_vector_file_version() {
        let router = TestRouter { cluster: 42 };
        let mut bytes = Vec::new();
        router.serialize(&mut bytes).unwrap();
        let slot = FileSlice::from(bytes);
        let error = TestRouter::open_router(VectorFileVersion::V2, slot, &test_context())
            .err()
            .expect("incompatible vector file version must fail");
        assert!(error
            .to_string()
            .contains("requires vector file version V3"));
    }
}
