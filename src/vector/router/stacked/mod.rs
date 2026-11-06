//! The stacked router: a stacked IVF over a segment's centroids, ranked with
//! [`StackedIvf::lists_to_open`] and stopped by [`aps`].

pub(crate) mod aps;
pub(crate) mod levels;
pub(crate) mod search;

use std::sync::Arc;

use crate::directory::FileSlice;
use crate::schema::VectorOptions;
use crate::vector::clustering::IvfCentroids;
use crate::vector::router::arena::VectorArena;
use crate::vector::router::source::{EncodingContext, VectorSet, VectorSource, ROUTER_SEED};
use crate::vector::router::stacked::levels::{
    serialize_stacked_index, InMemoryStore, LazyStore, StackedIvf, StackedIvfBuilder,
    StackedIvfConfig, StackedSearchStats, SuperKMeansLevelClusterer, PARENT_NPROBE_FRACTION,
};
use crate::vector::router::{RouterIter, RouterKind, RouterMetrics, RoutingParams};
use crate::vector::search::prepared::VectorQuery;
use crate::vector::storage::quantized::sign::SignVectors;
use crate::vector::VectorElement;
use crate::TantivyError;

/// The router's [`StackedIvfConfig`]. The segment's own IVF (`.vec` rows under
/// `.centroids`) is L0 and is probed by the caller's work budget; every
/// level of this router sits above it and is a parent, so the bottom router
/// level uses [`PARENT_NPROBE_FRACTION`] too rather than the standalone
/// L0 default. Search opens [`StackedIvf::lists_to_open`] lists (25% floor,
/// ~2x `k` member beam, 100% cap). Each level uses `nlist = ceil(sqrt(n))`;
/// parents are added while the top has at least 256 lists. The config is
/// not persisted, so open must agree with build only on what search reads.
pub(super) fn router_config() -> StackedIvfConfig {
    StackedIvfConfig {
        nprobe_fraction: PARENT_NPROBE_FRACTION,
    }
}

/// Builds the stack and reorders `centroids` into router order; also returns that permutation,
/// `perm[old] = new`.
pub(super) fn build(
    options: &VectorOptions,
    centroids: &mut IvfCentroids,
) -> crate::Result<(StackedIvf<InMemoryStore>, Vec<u32>)> {
    let IvfCentroids::F32(matrix) = &*centroids;
    let clusterer = SuperKMeansLevelClusterer::for_metric(options.metric());
    let (index, permutation) = StackedIvfBuilder::new(
        matrix.values.clone(),
        matrix.rows,
        options.dim(),
        &clusterer,
        router_config(),
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
    Ok((index, permutation))
}

pub(super) fn open(
    payload: FileSlice,
    centroids: FileSlice,
    options: &VectorOptions,
) -> crate::Result<StackedIvf<LazyStore>> {
    Ok(StackedIvf::<LazyStore>::open(
        payload,
        centroids,
        options.dim(),
        router_config(),
    )?)
}

/// One stacked algorithm, independently configured vector banks and a shared
/// encoding context carried by the banks that use compressed representations.
pub(crate) struct StackedRouter<S> {
    pub(crate) level: StackedIvf<VectorSet<S>>,
    pub(crate) dim: usize,
    pub(crate) metric: crate::schema::Metric,
}

impl StackedRouter<InMemoryStore> {
    pub(crate) fn build(
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
        compressed: bool,
    ) -> crate::Result<Self> {
        Ok(Self::build_with_permutation(options, centroids, compressed)?.0)
    }

    /// [`Self::build`], also returning how it reordered `centroids`: `perm[old] = new`.
    pub(crate) fn build_with_permutation(
        options: &VectorOptions,
        centroids: &mut IvfCentroids,
        compressed: bool,
    ) -> crate::Result<(Self, Vec<u32>)> {
        if options.dim() == 0 || options.dim() > u32::MAX as usize {
            return Err(TantivyError::InvalidArgument(
                "routing requires a positive supported dimension".into(),
            ));
        }
        let (index, permutation) = build(options, centroids)?;
        Ok((Self::from_index(index, options, compressed)?, permutation))
    }

    pub(crate) fn from_index(
        index: StackedIvf<InMemoryStore>,
        options: &VectorOptions,
        compressed: bool,
    ) -> crate::Result<Self> {
        let (config, topology, banks) = index.into_vector_sets();
        let context =
            compressed.then(|| Arc::new(EncodingContext::new(options.dim(), ROUTER_SEED)));
        let mut sources = Vec::with_capacity(banks.len());
        for (bank, raw) in banks.into_iter().enumerate() {
            let source = if let Some(context) = &context {
                let codes = Arc::new(SignVectors::encode(&raw, options.dim(), &context.rotation)?);
                VectorSet::Encoded {
                    codes,
                    context: context.clone(),
                    // Preserve the version-1 policy: only final members have
                    // raw rows and receive exact refinement.
                    raw: (bank == 0).then(|| Arc::new(raw)),
                }
            } else {
                VectorSet::Raw(Arc::new(raw))
            };
            sources.push(source);
        }
        let level = StackedIvf::from_vector_sets(topology, sources, config, |source| {
            source.num_vectors(options.dim())
        })?;
        Self::new(level, options)
    }

    pub(crate) fn serialize(
        &self,
        out: &mut (impl std::io::Write + ?Sized),
    ) -> std::io::Result<()> {
        if self.level.vectors.compressed().is_some() {
            super::quantized::serialize(self, out)
        } else {
            serialize_stacked_index(&self.level, out)
        }
    }
}

impl StackedRouter<LazyStore> {
    pub(crate) fn open_raw(
        payload: FileSlice,
        rows: FileSlice,
        options: &VectorOptions,
    ) -> crate::Result<Self> {
        let (config, topology, banks) = open(payload, rows, options)?.into_vector_sets();
        let sources = banks
            .into_iter()
            .map(|raw| VectorSet::Raw(Arc::new(raw)))
            .collect();
        let level = StackedIvf::from_vector_sets(topology, sources, config, |source| {
            source.num_vectors(options.dim())
        })?;
        Self::new(level, options)
    }
}

impl<S: VectorArena<Elem = f32>> StackedRouter<S> {
    /// Validate capabilities before constructing a final-routing pipeline. All
    /// encoded banks share the router's persisted origin/rotation context.
    pub(crate) fn new(
        level: StackedIvf<VectorSet<S>>,
        options: &VectorOptions,
    ) -> crate::Result<Self> {
        if level.vectors.raw().is_none() {
            return Err(TantivyError::InvalidArgument(
                "final routing requires raw member vectors".into(),
            ));
        }
        let mut seed = None;
        for source in
            std::iter::once(&level.vectors).chain(level.levels().map(|level| &level.centroids))
        {
            if let Some((_, context)) = source.compressed() {
                if context.rotation.dimension() != options.dim()
                    || seed.is_some_and(|seed| seed != context.seed)
                {
                    return Err(TantivyError::InvalidArgument(
                        "router banks require one shared encoding context".into(),
                    ));
                }
                seed = Some(context.seed);
                if let Some(raw) = source.raw() {
                    if raw.num_vectors(options.dim()) != source.num_vectors(options.dim()) {
                        return Err(TantivyError::InvalidArgument(
                            "raw and encoded member identities differ".into(),
                        ));
                    }
                }
            }
        }
        Ok(Self {
            level,
            dim: options.dim(),
            metric: options.metric(),
        })
    }

    /// The public/persisted compatibility identifier; runtime dispatch is stacked
    /// for both vector representations.
    pub(crate) fn kind(&self) -> RouterKind {
        if self.level.vectors.compressed().is_some() {
            RouterKind::QuantizedStacked
        } else {
            RouterKind::Stacked
        }
    }

    pub(crate) fn level_sizes(&self) -> Vec<usize> {
        self.level
            .levels()
            .map(|level| level.offsets.len())
            .collect()
    }

    pub(crate) fn rank<T: VectorElement>(
        &self,
        query: &VectorQuery<T>,
        params: RoutingParams,
    ) -> RouterIter {
        assert_eq!(query.widened().len(), self.dim);
        debug_assert_eq!(query.metric(), self.metric);
        rank(&self.level, query, params)
    }
}

/// The result contract is independent of the precision pipeline that produced it.
pub(super) fn rank<S: VectorSource, T: VectorElement>(
    level: &StackedIvf<S>,
    query: &VectorQuery<T>,
    params: RoutingParams,
) -> RouterIter {
    let compressed = level.vectors.compressed().is_some();
    let (ranked, stats) = super::search::rank(level, query, params);
    let candidate_count = ranked.len();
    let metrics = routing_metrics(compressed, candidate_count, stats);
    RouterIter::new(ranked, metrics)
}

pub(super) fn routing_metrics(
    compressed: bool,
    candidate_count: usize,
    stats: StackedSearchStats,
) -> RouterMetrics {
    if compressed {
        RouterMetrics::QuantizedStacked {
            candidate_count,
            lists_scanned: stats.lists_scanned,
            approximate_scored: stats.approximate_scored,
            approximate_members_scored: stats.leaf_approximate_scored,
            exact_scored: stats.leaf_exact_scored,
        }
    } else {
        RouterMetrics::Stacked {
            candidate_count,
            lists_scanned: stats.lists_scanned,
            members_scored: stats.members_scored,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;

    #[test]
    fn construction_rejects_unavailable_refinement_and_inconsistent_banks() -> crate::Result<()> {
        let dim = 8;
        let encoded = |rows: usize,
                       raw_rows: Option<usize>,
                       context: Arc<EncodingContext>|
         -> crate::Result<_> {
            Ok(VectorSet::Encoded {
                codes: Arc::new(SignVectors::encode(
                    &vec![1.0; rows * dim],
                    dim,
                    &context.rotation,
                )?),
                context,
                raw: raw_rows.map(|rows| Arc::new(InMemoryStore::new(vec![1.0; rows * dim], dim))),
            })
        };
        let context = Arc::new(EncodingContext::new(dim, ROUTER_SEED));
        let options = VectorOptions::new(dim, Metric::L2);
        let level = |vectors, centroids| StackedIvf {
            config: Default::default(),
            parent: None,
            offsets: vec![(0, 2)],
            vectors,
            centroids,
        };
        let root = || VectorSet::Raw(Arc::new(InMemoryStore::new(vec![1.0; dim], dim)));
        assert!(
            StackedRouter::new(level(encoded(2, None, context.clone())?, root()), &options)
                .is_err()
        );
        assert!(StackedRouter::new(
            level(encoded(2, Some(3), context.clone())?, root()),
            &options
        )
        .is_err());
        let other_seed = Arc::new(EncodingContext::new(dim, ROUTER_SEED + 1));
        assert!(StackedRouter::new(
            level(
                encoded(2, Some(2), context.clone())?,
                encoded(1, None, other_seed)?,
            ),
            &options
        )
        .is_err());
        // A shared context with the wrong dimension must fail before query preparation.
        let wrong_dim = Arc::new(EncodingContext::new(dim * 2, ROUTER_SEED));
        let mut members = encoded(2, Some(2), context)?;
        if let VectorSet::Encoded { context, .. } = &mut members {
            *context = wrong_dim;
        }
        assert!(StackedRouter::new(level(members, root()), &options).is_err());
        Ok(())
    }
}
