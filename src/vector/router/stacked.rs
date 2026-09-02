use crate::directory::FileSlice;
use crate::schema::{Metric, VectorOptions};
use crate::vector::ivf::{
    Candidate, ClusterId, InMemoryStackedIvf, IvfConfig, IvfIndexBuilder, LazyStackedIvf,
    SuperKMeansLevelClusterer,
};
use crate::vector::IvfCentroids;
use crate::TantivyError;

pub(super) fn build(
    options: &VectorOptions,
    centroids: &mut IvfCentroids,
) -> crate::Result<InMemoryStackedIvf> {
    let IvfCentroids::F32(matrix) = &*centroids;
    let clusterer = SuperKMeansLevelClusterer::default();
    let (index, permutation) = IvfIndexBuilder::new(
        matrix.values.clone(),
        matrix.rows,
        options.dim(),
        &clusterer,
        IvfConfig::default(),
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
    Ok(index)
}

pub(super) fn open(
    payload: FileSlice,
    centroids: FileSlice,
    options: &VectorOptions,
) -> crate::Result<LazyStackedIvf> {
    Ok(LazyStackedIvf::open(
        payload,
        centroids,
        options.dim(),
        IvfConfig::default(),
    )?)
}

pub(super) fn rank(index: &LazyStackedIvf, query: &[f32], metric: Metric) -> Ranking {
    let ranked = index.search(query, index.nlist(), 1.0, metric);
    let candidate_count = ranked.len();
    Ranking {
        ranked: ranked.into_iter(),
        candidate_count,
    }
}

pub(crate) struct Ranking {
    ranked: std::vec::IntoIter<Candidate<ClusterId>>,
    candidate_count: usize,
}

impl Ranking {
    pub(super) fn candidate_count(&self) -> usize {
        self.candidate_count
    }
}

impl Iterator for Ranking {
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        self.ranked.next().map(|candidate| Candidate {
            sim: candidate.sim,
            node: candidate.node.0,
        })
    }
}
