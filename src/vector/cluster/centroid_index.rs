use crate::vector::rabitq::math::l2_distance_sqr;
use crate::vector::rabitq::Metric;

pub struct CentroidIndex {
    pub(crate) centroid_ids: Vec<u32>,
    centroids: Vec<Vec<f32>>,
    dims: usize,
}

impl CentroidIndex {
    pub fn build(centroids: Vec<Vec<f32>>, centroid_ids: Vec<u32>, _metric: Metric) -> Self {
        assert_eq!(centroids.len(), centroid_ids.len());
        let dims = centroids.first().map_or(0, |v| v.len());
        Self {
            centroid_ids,
            centroids,
            dims,
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let n = self.centroid_ids.len();
        if n == 0 {
            return vec![];
        }
        let k = k.min(n);

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
        let mut buf = Vec::new();
        let n = self.centroids.len() as u32;
        buf.extend_from_slice(&n.to_le_bytes());
        buf.extend_from_slice(&(self.dims as u32).to_le_bytes());

        for centroid in &self.centroids {
            for &v in centroid {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        Ok(buf)
    }

    pub fn load_from_bytes(
        bytes: &[u8],
        centroid_ids: Vec<u32>,
        dims: usize,
        _metric: Metric,
    ) -> crate::Result<Self> {
        let mut pos = 0;
        let n = u32::from_le_bytes([bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]]) as usize;
        pos += 4;
        let _dims = u32::from_le_bytes([bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]]) as usize;
        pos += 4;

        let mut centroids = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v = Vec::with_capacity(dims);
            for _ in 0..dims {
                let f = f32::from_le_bytes([bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]]);
                pos += 4;
                v.push(f);
            }
            centroids.push(v);
        }

        Ok(Self {
            centroid_ids,
            centroids,
            dims,
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
    fn single_centroid() {
        let centroids = vec![vec![5.0, 5.0]];
        let ids = vec![42];
        let index = CentroidIndex::build(centroids, ids, Metric::L2);

        let results = index.search(&[0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn search_clamps_k() {
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
