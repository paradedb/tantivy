use crate::vector::math::l2_distance_sqr;
use crate::vector::Metric;

/// Full-precision centroid storage for brute-force probe selection.
///
/// Each centroid is stored as `dims` consecutive `f32` values. The previous
/// implementation used per-centroid int8 quantization (one f32 scale +
/// `dims` i8 values, ~4× smaller); this is the experimental f32 variant.
///
/// File-on-disk size for `dims=768`:
///   - i8 storage:  4 (scale) + 768 = 772 bytes per centroid
///   - f32 storage: 768 * 4 = 3072 bytes per centroid (~4× larger)
pub struct CentroidIndex {
    pub(crate) centroid_ids: Vec<u32>,
    /// Flat layout: centroid `i` occupies `data[i*dims..(i+1)*dims]`.
    data: Vec<f32>,
    dims: usize,
}

impl CentroidIndex {
    pub fn build(centroids: Vec<Vec<f32>>, centroid_ids: Vec<u32>, _metric: Metric) -> Self {
        assert_eq!(centroids.len(), centroid_ids.len());
        let dims = centroids.first().map_or(0, |v| v.len());
        let mut data = Vec::with_capacity(centroids.len() * dims);
        for c in &centroids {
            data.extend_from_slice(c);
        }
        Self {
            centroid_ids,
            data,
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
            let centroid = &self.data[i * self.dims..(i + 1) * self.dims];
            results.push((self.centroid_ids[i], l2_distance_sqr(query, centroid)));
        }

        if k < n {
            results.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
            results.truncate(k);
        }
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    /// On-disk format:
    ///   u32 n
    ///   u32 dims
    ///   f32[n*dims] values
    pub fn save_to_bytes(&self) -> crate::Result<Vec<u8>> {
        let n = self.centroid_ids.len() as u32;
        let mut buf = Vec::with_capacity(8 + self.data.len() * std::mem::size_of::<f32>());
        buf.extend_from_slice(&n.to_le_bytes());
        buf.extend_from_slice(&(self.dims as u32).to_le_bytes());
        // SAFETY: f32 slice → bytes
        let data_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        };
        buf.extend_from_slice(data_bytes);
        Ok(buf)
    }

    pub fn load_from_bytes(
        bytes: &[u8],
        centroid_ids: Vec<u32>,
        dims: usize,
        _metric: Metric,
    ) -> crate::Result<Self> {
        let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let _stored_dims = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let mut p = 8;

        let total = n * dims;
        let mut data = Vec::with_capacity(total);
        for _ in 0..total {
            data.push(f32::from_le_bytes([
                bytes[p],
                bytes[p + 1],
                bytes[p + 2],
                bytes[p + 3],
            ]));
            p += 4;
        }

        Ok(Self {
            centroid_ids,
            data,
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

        // Full precision: round-trip is exact.
        for (a, b) in results_orig.iter().zip(results_loaded.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    fn high_dim_nearest() {
        let mut centroids = Vec::new();
        for i in 0..10 {
            centroids.push((0..768).map(|d| (i * 100 + d) as f32 * 0.001).collect());
        }
        let ids: Vec<u32> = (0..10).collect();
        let index = CentroidIndex::build(centroids.clone(), ids, Metric::L2);

        let query: Vec<f32> = (0..768).map(|d| (5 * 100 + d) as f32 * 0.001).collect();
        let results = index.search(&query, 3);
        assert_eq!(results[0].0, 5);
    }
}
