use std::io::{self, Write};

use common::{BinarySerializable, HasLen};

use super::ivf::graph::{Candidate, NeighborhoodGraphSearchMetrics, Workspace};
use super::ivf::IvfCentroids;
use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::header::VectorFileVersion;

mod exact;
mod rng;
mod stacked;

pub use exact::LazyExactRouter;

const ROUTER_HEADER_LEN: usize = size_of::<u16>();

/// The persisted identity and format compatibility of a router family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterDescriptor {
    id: &'static str,
    vector_file_version: VectorFileVersion,
}

impl RouterDescriptor {
    pub const fn new(id: &'static str, vector_file_version: VectorFileVersion) -> Self {
        Self {
            id,
            vector_file_version,
        }
    }

    pub fn id(self) -> &'static str {
        self.id
    }

    pub fn vector_file_version(self) -> VectorFileVersion {
        self.vector_file_version
    }

    pub(crate) fn validate(self) -> crate::Result<()> {
        if self.id.is_empty() {
            return Err(crate::TantivyError::InvalidArgument(
                "router ID cannot be empty".to_string(),
            ));
        }
        if u16::try_from(self.id.len()).is_err() {
            return Err(crate::TantivyError::InvalidArgument(format!(
                "router ID exceeds u16: {}",
                self.id
            )));
        }
        Ok(())
    }
}

pub(crate) trait ErasedRouterDescriptor {
    fn erased_descriptor(&self) -> RouterDescriptor;
}

/// Builds and routes an IVF index and owns its persisted format.
///
/// `serialize` prefixes the router payload with its ID. The configured router
/// checks that ID before opening the payload.
#[allow(private_bounds)]
pub trait Router: ErasedRouterDescriptor + Send + Sync + 'static {
    /// Return this router's persisted identity and compatible vector file version.
    fn router_descriptor() -> RouterDescriptor
    where Self: Sized;

    fn descriptor(&self) -> RouterDescriptor {
        ErasedRouterDescriptor::erased_descriptor(self)
    }

    fn id(&self) -> &'static str {
        self.descriptor().id()
    }

    fn vector_file_version(&self) -> VectorFileVersion {
        self.descriptor().vector_file_version()
    }

    /// Builds a router over the supplied centroids, which may be empty.
    /// Implementations may reorder rows in place but must preserve the matrix shape.
    fn build_router(
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Box<dyn Router>>
    where
        Self: Sized;

    fn deserialize(
        payload: FileSlice,
        centroids: FileSlice,
        options: &VectorOptions,
    ) -> crate::Result<Box<dyn Router>>
    where
        Self: Sized;

    /// Rank centroids and update `metrics` as the returned iterator advances.
    fn rank<'a>(
        &'a self,
        workspace: &'a mut Workspace,
        query: &'a [f32],
        metric: Metric,
        metrics: &'a mut IvfSearchMetrics,
    ) -> Box<dyn Iterator<Item = Candidate> + 'a>;

    fn serialize_payload(&self, out: &mut dyn Write) -> io::Result<()>;

    fn serialize(&self, out: &mut dyn Write) -> io::Result<()> {
        let descriptor = self.descriptor();
        write_router_header(descriptor.id(), out)?;
        self.serialize_payload(out)
    }
}

impl<R: Router> ErasedRouterDescriptor for R {
    fn erased_descriptor(&self) -> RouterDescriptor {
        R::router_descriptor()
    }
}

/// Type-erased constructors for the router type selected on an index.
/// `Router` represents an existing router, but its `Self: Sized` build and
/// deserialize functions cannot be called through `dyn Router` before one exists.
#[derive(Clone, Copy)]
pub(crate) struct RouterBinding {
    descriptor: RouterDescriptor,
    build:
        fn(options: &VectorOptions, centroids: &mut IvfCentroids) -> crate::Result<Box<dyn Router>>,
    deserialize: fn(
        payload: FileSlice,
        centroids: FileSlice,
        options: &VectorOptions,
    ) -> crate::Result<Box<dyn Router>>,
}

impl RouterBinding {
    pub(crate) fn new<R: Router>() -> Self {
        Self {
            descriptor: R::router_descriptor(),
            build: R::build_router,
            deserialize: R::deserialize,
        }
    }

    pub(crate) fn descriptor(&self) -> RouterDescriptor {
        self.descriptor
    }

    pub(crate) fn build(
        &self,
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
    ) -> crate::Result<Box<dyn Router>> {
        (self.build)(options, centroids)
    }

    pub(crate) fn open(
        &self,
        file_version: VectorFileVersion,
        slot: FileSlice,
        centroids: FileSlice,
        options: &VectorOptions,
    ) -> crate::Result<Box<dyn Router>> {
        let descriptor = self.descriptor();
        if descriptor.vector_file_version() != file_version {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "router {} requires vector file version {:?}, found {:?}",
                    descriptor.id(),
                    descriptor.vector_file_version(),
                    file_version
                ),
            )
            .into());
        }
        if slot.len() < ROUTER_HEADER_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "router slot is shorter than its header",
            )
            .into());
        }
        let header = slot.slice_to(ROUTER_HEADER_LEN).read_bytes()?;
        let mut cursor = header.as_slice();
        let id_len = u16::deserialize(&mut cursor)? as usize;
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
        let persisted_id = std::str::from_utf8(&id_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "router ID is not UTF-8"))?;
        if persisted_id != descriptor.id() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "configured router {} does not match persisted router {persisted_id}",
                    descriptor.id()
                ),
            )
            .into());
        }
        let router = (self.deserialize)(slot.slice_from(payload_offset), centroids, options)?;
        if router.descriptor() != descriptor {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "configured router {} opened router {}",
                    descriptor.id(),
                    router.id()
                ),
            )
            .into());
        }
        Ok(router)
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct IvfSearchMetrics {
    pub visited_count: usize,
    pub graph: Option<NeighborhoodGraphSearchMetrics>,
}

fn write_router_header(id: &str, out: &mut dyn Write) -> io::Result<()> {
    if id.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "router ID cannot be empty",
        ));
    }
    let id_len = u16::try_from(id.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "router ID exceeds u16"))?;
    id_len.serialize(out)?;
    out.write_all(id.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Similarity;

    struct TestRouter<const KIND: u8> {
        cluster: u32,
    }

    impl<const KIND: u8> Router for TestRouter<KIND> {
        fn router_descriptor() -> RouterDescriptor {
            let id = match KIND {
                0 => "test.primary",
                1 => "test.secondary",
                _ => "test.unknown",
            };
            RouterDescriptor::new(id, VectorFileVersion::V3)
        }

        fn build_router(
            _options: &VectorOptions,
            _centroids: &mut IvfCentroids,
        ) -> crate::Result<Box<dyn Router>> {
            Ok(Box::new(TestRouter::<KIND> { cluster: 0 }))
        }

        fn deserialize(
            payload: FileSlice,
            _centroids: FileSlice,
            _options: &VectorOptions,
        ) -> crate::Result<Box<dyn Router>> {
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
            Ok(Box::new(TestRouter::<KIND> { cluster }))
        }

        fn rank<'a>(
            &'a self,
            _workspace: &'a mut Workspace,
            _query: &'a [f32],
            _metric: Metric,
            metrics: &'a mut IvfSearchMetrics,
        ) -> Box<dyn Iterator<Item = Candidate> + 'a> {
            *metrics = IvfSearchMetrics {
                visited_count: 1,
                graph: None,
            };
            Box::new(
                vec![Candidate {
                    sim: Similarity::new(1.0),
                    node: self.cluster,
                }]
                .into_iter(),
            )
        }

        fn serialize_payload(&self, out: &mut dyn Write) -> io::Result<()> {
            self.cluster.serialize(out)
        }
    }

    #[test]
    fn configured_router_opens_matching_payload() -> crate::Result<()> {
        let router = TestRouter::<0> { cluster: 42 };
        let mut bytes = Vec::new();
        router.serialize(&mut bytes)?;
        let binding = RouterBinding::new::<TestRouter<0>>();
        let options = VectorOptions::new(1, Metric::L2);
        let opened = binding.open(
            VectorFileVersion::V3,
            FileSlice::from(bytes),
            FileSlice::empty(),
            &options,
        )?;
        let mut workspace = Workspace::new();
        let mut metrics = IvfSearchMetrics::default();
        let mut ranking = opened.rank(&mut workspace, &[0.0], Metric::L2, &mut metrics);
        assert_eq!(ranking.next().unwrap().node, 42);
        drop(ranking);
        assert_eq!(metrics.visited_count, 1);
        Ok(())
    }

    #[test]
    fn configured_router_rejects_a_different_persisted_router() {
        let router = TestRouter::<1> { cluster: 42 };
        let mut bytes = Vec::new();
        router.serialize(&mut bytes).unwrap();
        let options = VectorOptions::new(1, Metric::L2);
        let error = RouterBinding::new::<TestRouter<0>>()
            .open(
                VectorFileVersion::V3,
                FileSlice::from(bytes),
                FileSlice::empty(),
                &options,
            )
            .err()
            .expect("a different persisted router must fail");
        assert!(error.to_string().contains(
            "configured router test.primary does not match persisted router test.secondary"
        ));
    }

    #[test]
    fn pre_v3_router_format_is_rejected() {
        let options = VectorOptions::new(1, Metric::L2);
        let error = RouterBinding::new::<TestRouter<0>>()
            .open(
                VectorFileVersion::V2,
                FileSlice::empty(),
                FileSlice::empty(),
                &options,
            )
            .err()
            .expect("pre-V3 router formats must fail");
        assert!(error
            .to_string()
            .contains("requires vector file version V3"));
    }
}
