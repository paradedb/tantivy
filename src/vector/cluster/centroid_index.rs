use usearch::ffi::{IndexOptions, MetricKind, ScalarKind};
use usearch::Index;

use crate::vector::rabitq::math::l2_distance_sqr;
use crate::vector::rabitq::Metric;

fn metric_to_usearch(metric: Metric) -> MetricKind {
    match metric {
        Metric::L2 => MetricKind::L2sq,
        Metric::InnerProduct => MetricKind::IP,
    }
}

fn make_options(dims: usize, metric: MetricKind) -> IndexOptions {
    IndexOptions {
        dimensions: dims,
        metric,
        quantization: ScalarKind::F32,
        connectivity: 32,
        expansion_add: 200,
        expansion_search: 0,
        multi: false,
    }
}

pub struct CentroidIndex {
    pub(crate) centroid_ids: Vec<u32>,
    centroids: Vec<Vec<f32>>,
    index: Index,
    dims: usize,
    metric: MetricKind,
}

impl CentroidIndex {
    pub fn build(centroids: Vec<Vec<f32>>, centroid_ids: Vec<u32>, metric: Metric) -> Self {
        assert_eq!(centroids.len(), centroid_ids.len());
        let dims = centroids.first().map_or(0, |v| v.len());
        let metric = metric_to_usearch(metric);

        let options = make_options(dims, metric);
        let index = Index::new(&options).expect("failed to create usearch index");

        if !centroids.is_empty() {
            index
                .reserve(centroids.len())
                .expect("failed to reserve capacity");
            for (i, centroid) in centroids.iter().enumerate() {
                index
                    .add(i as u64, centroid.as_slice())
                    .expect("failed to add centroid");
            }
        }

        Self {
            centroid_ids,
            centroids,
            index,
            dims,
            metric,
        }
    }

    pub fn search(&self, query: &[f32], ef_search: usize) -> Vec<(u32, f32)> {
        let n = self.centroid_ids.len();
        if n == 0 {
            return vec![];
        }
        // For small cluster counts, brute-force is faster than HNSW
        // due to graph traversal overhead. For larger counts, HNSW
        // visits fewer nodes (O(log n) vs O(n)).
        if n <= 1024 {
            return self.brute_force_search(query, ef_search.min(n));
        }

        let limit = ef_search.min(n);
        let results = self.index.search(query, limit).expect("search failed");

        results
            .keys
            .iter()
            .zip(results.distances.iter())
            .map(|(&key, &dist)| (self.centroid_ids[key as usize], dist))
            .collect()
    }

    fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let n = self.centroid_ids.len();
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            results.push((self.centroid_ids[i], l2_distance_sqr(query, &self.centroids[i])));
        }

        if k < n {
            results.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
            results.truncate(k);
        }
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    pub fn save_to_bytes(&self) -> crate::Result<Vec<u8>> {
        let len = self.index.serialized_length();
        let mut buf = vec![0u8; len];
        self.index.save_to_buffer(&mut buf).map_err(|e| {
            crate::TantivyError::InternalError(format!("usearch save failed: {e}"))
        })?;
        Ok(buf)
    }

    pub fn load_from_bytes(
        bytes: &[u8],
        centroid_ids: Vec<u32>,
        dims: usize,
        metric: Metric,
    ) -> crate::Result<Self> {
        let metric = metric_to_usearch(metric);
        let options = make_options(dims, metric);
        let index = Index::new(&options).map_err(|e| {
            crate::TantivyError::InternalError(format!("usearch index create failed: {e}"))
        })?;
        index.load_from_buffer(bytes).map_err(|e| {
            crate::TantivyError::InternalError(format!("usearch load failed: {e}"))
        })?;

        // Extract centroids from usearch index for brute-force path
        let centroids: Vec<Vec<f32>> = (0..centroid_ids.len())
            .map(|i| {
                let mut v = vec![0.0f32; dims];
                index.get(i as u64, &mut v).expect("failed to get centroid");
                v
            })
            .collect();

        Ok(Self {
            centroid_ids,
            centroids,
            index,
            dims,
            metric,
        })
    }

    pub fn len(&self) -> usize {
        self.centroid_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.centroid_ids.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dims
    }

    pub fn metric(&self) -> MetricKind {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_finds_nearest_centroid() {
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0], vec![20.0, 20.0]];
        let ids = vec![100, 200, 300];
        let index = CentroidIndex::build(centroids, ids, Metric::L2);

        let results = index.search(&[1.0, 1.0], 3);
        assert_eq!(results[0].0, 100);

        let results = index.search(&[19.0, 19.0], 3);
        assert_eq!(results[0].0, 300);
    }

    #[test]
    fn brute_force_fallback_single_centroid() {
        let centroids = vec![vec![5.0, 5.0]];
        let ids = vec![42];
        let index = CentroidIndex::build(centroids, ids, Metric::L2);

        let results = index.search(&[0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn search_clamps_ef_search() {
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let ids = vec![1, 2];
        let index = CentroidIndex::build(centroids, ids, Metric::L2);

        let results = index.search(&[0.0, 0.0], 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn kmeans_then_centroid_index() {
        use crate::vector::cluster::kmeans::{run_kmeans_with_config, KMeansConfig};

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
        let index = CentroidIndex::build(result.centroids, ids, Metric::L2);

        let results = index.search(&[1.0, 1.0], 2);
        assert!(results[0].1 < results[1].1);
    }

    #[test]
    fn save_load_roundtrip() {
        let centroids = vec![
            vec![0.0, 0.0, 0.0],
            vec![10.0, 10.0, 10.0],
            vec![20.0, 20.0, 20.0],
        ];
        let ids = vec![100, 200, 300];
        let index = CentroidIndex::build(centroids, ids.clone(), Metric::L2);

        let bytes = index.save_to_bytes().unwrap();
        assert!(!bytes.is_empty());

        let loaded = CentroidIndex::load_from_bytes(&bytes, ids, 3, Metric::L2).unwrap();

        let results_orig = index.search(&[1.0, 1.0, 1.0], 3);
        let results_loaded = loaded.search(&[1.0, 1.0, 1.0], 3);

        assert_eq!(results_orig[0].0, results_loaded[0].0);
        assert_eq!(results_orig.len(), results_loaded.len());
    }
}
