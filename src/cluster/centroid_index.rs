use hnsw_rs::prelude::*;
use parking_lot::RwLock;

use crate::rabitq::math::l2_distance_sqr;

pub struct CentroidIndex {
    pub(crate) centroid_ids: Vec<u32>,
    pub(crate) centroids: Vec<Vec<f32>>,
    hnsw_cache: RwLock<Option<Hnsw<'static, f32, DistL2>>>,
}

// SAFETY: The Hnsw graph references data owned by `centroids` which is pinned
// for the lifetime of the struct. The RwLock ensures thread-safe access.
unsafe impl Send for CentroidIndex {}
unsafe impl Sync for CentroidIndex {}

impl CentroidIndex {
    pub fn build(centroids: Vec<Vec<f32>>, centroid_ids: Vec<u32>) -> Self {
        assert_eq!(centroids.len(), centroid_ids.len());
        Self {
            centroid_ids,
            centroids,
            hnsw_cache: RwLock::new(None),
        }
    }

    fn ensure_hnsw_built(&self) {
        {
            let cache = self.hnsw_cache.read();
            if cache.is_some() {
                return;
            }
        }

        let mut cache = self.hnsw_cache.write();
        if cache.is_some() {
            return;
        }

        let max_nb_connection = 32;
        let ef_construction = 200;
        let max_layer = 16;

        let mut hnsw = Hnsw::<f32, DistL2>::new(
            max_nb_connection,
            self.centroids.len(),
            max_layer,
            ef_construction,
            DistL2 {},
        );

        // SAFETY: We transmute the lifetime of the centroid references to 'static.
        // This is safe because `self.centroids` is owned by the same struct and will
        // not be moved or dropped while the HNSW index exists. The `hnsw_cache` is
        // invalidated when the struct is dropped.
        let data_with_id: Vec<(&Vec<f32>, usize)> = unsafe {
            std::mem::transmute(
                self.centroids
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v, i))
                    .collect::<Vec<_>>(),
            )
        };
        hnsw.parallel_insert(&data_with_id);
        hnsw.set_searching_mode(true);

        *cache = Some(hnsw);
    }

    pub fn search(&self, query: &[f32], ef_search: usize) -> Vec<(u32, f32)> {
        let n = self.centroid_ids.len();

        if n < 2 {
            return self.brute_force_search(query, ef_search);
        }

        self.ensure_hnsw_built();

        let cache = self.hnsw_cache.read();
        let hnsw = cache.as_ref().unwrap();
        let limit = ef_search.min(n);

        let neighbors = hnsw.search(query, limit, limit);
        neighbors
            .iter()
            .map(|neighbor| (self.centroid_ids[neighbor.d_id], neighbor.distance))
            .collect()
    }

    fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| (self.centroid_ids[idx], l2_distance_sqr(query, centroid)))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    pub fn len(&self) -> usize {
        self.centroid_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.centroid_ids.is_empty()
    }

    pub fn dimension(&self) -> Option<usize> {
        self.centroids.first().map(|v| v.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_finds_nearest_centroid() {
        let centroids = vec![
            vec![0.0, 0.0],
            vec![10.0, 10.0],
            vec![20.0, 20.0],
        ];
        let ids = vec![100, 200, 300];
        let index = CentroidIndex::build(centroids, ids);

        let results = index.search(&[1.0, 1.0], 3);
        assert_eq!(results[0].0, 100);

        let results = index.search(&[19.0, 19.0], 3);
        assert_eq!(results[0].0, 300);
    }

    #[test]
    fn brute_force_fallback_single_centroid() {
        let centroids = vec![vec![5.0, 5.0]];
        let ids = vec![42];
        let index = CentroidIndex::build(centroids, ids);

        let results = index.search(&[0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn search_clamps_ef_search() {
        let centroids = vec![
            vec![0.0, 0.0],
            vec![10.0, 10.0],
        ];
        let ids = vec![1, 2];
        let index = CentroidIndex::build(centroids, ids);

        let results = index.search(&[0.0, 0.0], 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn kmeans_then_centroid_index() {
        use crate::cluster::kmeans::{run_kmeans_with_config, KMeansConfig};

        let mut data = Vec::new();
        for _ in 0..50 {
            data.push(vec![0.0, 0.0]);
            data.push(vec![100.0, 100.0]);
        }

        let config = KMeansConfig {
            niter: 20,
            nredo: 1,
            seed: 123,
            ..Default::default()
        };
        let result = run_kmeans_with_config(&data, 2, config);
        let ids: Vec<u32> = (0..result.centroids.len() as u32).collect();
        let index = CentroidIndex::build(result.centroids, ids);

        let results = index.search(&[1.0, 1.0], 2);
        let nearest = &index.centroids[results[0].0 as usize];
        assert!(nearest[0] < 50.0 && nearest[1] < 50.0);
    }
}
