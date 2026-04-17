use crate::vector::rabitq::distance::RaBitQQuery;
use crate::vector::rabitq::rotation::DynamicRotator;
use crate::vector::rabitq::{self, Metric, RabitqConfig};

/// RaBitQ-quantized centroid storage.
///
/// Each centroid is stored as a full RaBitQ record (binary code + extended
/// bits + scalars) produced by the same encoder used for document vectors.
/// This makes the quantization scheme rotation-based — random rotation
/// decorrelates dimensions so low-bit quantization preserves distance
/// ranking much more robustly than naive per-dim integer quantization.
///
/// Per-centroid on-disk sizes at `dims=768, ex_bits=2`:
///   f32 storage:        3072 bytes per centroid
///   i8 + scale:          772 bytes per centroid
///   RaBitQ (1 + 2 bits): ~320 bytes per centroid (binary 96 + ex 192 + 32 scalars)
///
/// Search reuses `RaBitQQuery::estimate_distance_from_record` — the same
/// distance-estimation path used for doc batches. Caller supplies the
/// pre-built `RaBitQQuery` (already cached per query), so centroid search
/// has no additional per-query setup cost.
pub struct CentroidIndex {
    pub(crate) centroid_ids: Vec<u32>,
    /// Concatenated RaBitQ records, one per centroid.
    records: Vec<u8>,
    bytes_per_record: usize,
    padded_dims: usize,
    ex_bits: usize,
    dims: usize,
}

impl CentroidIndex {
    pub fn build(
        centroids: Vec<Vec<f32>>,
        centroid_ids: Vec<u32>,
        rotator: &DynamicRotator,
        ex_bits: usize,
        metric: Metric,
    ) -> Self {
        assert_eq!(centroids.len(), centroid_ids.len());
        let dims = centroids.first().map_or(0, |v| v.len());
        let padded_dims = rotator.padded_dim();
        let bpr = rabitq::bytes_per_record(padded_dims, ex_bits);
        let config = RabitqConfig::new(ex_bits + 1);
        let n = centroids.len();

        let zero = vec![0.0f32; dims];
        let mut records = Vec::with_capacity(n * bpr);
        for c in &centroids {
            let rec = rabitq::encode(rotator, &config, metric, c, &zero);
            debug_assert_eq!(rec.len(), bpr);
            records.extend_from_slice(&rec);
        }

        Self {
            centroid_ids,
            records,
            bytes_per_record: bpr,
            padded_dims,
            ex_bits,
            dims,
        }
    }

    /// Score every centroid against `query` and return the `k` nearest as
    /// `(centroid_id, estimated_distance)` ascending by distance.
    ///
    /// The caller supplies a ready-to-use `RaBitQQuery` built with the same
    /// rotator, ex_bits, and metric as `build` received — typically cached
    /// once per search-query and reused across every window's centroid index
    /// and every doc-vector batch scan in the same query.
    pub fn search(&self, query: &RaBitQQuery, k: usize) -> Vec<(u32, f32)> {
        let n = self.centroid_ids.len();
        if n == 0 {
            return vec![];
        }
        let k = k.min(n);

        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * self.bytes_per_record;
            let rec = &self.records[off..off + self.bytes_per_record];
            let d = query.estimate_distance_from_record(rec, self.padded_dims, 0.0);
            results.push((self.centroid_ids[i], d));
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
    ///   u32 padded_dims
    ///   u32 ex_bits
    ///   u32 dims
    ///   u8[n * bytes_per_record] concatenated RaBitQ records
    pub fn save_to_bytes(&self) -> crate::Result<Vec<u8>> {
        let n = self.centroid_ids.len() as u32;
        let mut buf = Vec::with_capacity(16 + self.records.len());
        buf.extend_from_slice(&n.to_le_bytes());
        buf.extend_from_slice(&(self.padded_dims as u32).to_le_bytes());
        buf.extend_from_slice(&(self.ex_bits as u32).to_le_bytes());
        buf.extend_from_slice(&(self.dims as u32).to_le_bytes());
        buf.extend_from_slice(&self.records);
        Ok(buf)
    }

    pub fn load_from_bytes(
        bytes: &[u8],
        centroid_ids: Vec<u32>,
        _dims_hint: usize,
        _metric: Metric,
    ) -> crate::Result<Self> {
        let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let padded_dims =
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let ex_bits =
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let dims =
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let bpr = rabitq::bytes_per_record(padded_dims, ex_bits);
        let total = n * bpr;
        let records = bytes[16..16 + total].to_vec();

        Ok(Self {
            centroid_ids,
            records,
            bytes_per_record: bpr,
            padded_dims,
            ex_bits,
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
    use crate::vector::rabitq::rotation::RotatorType;

    fn make_rotator(dims: usize) -> DynamicRotator {
        DynamicRotator::new(dims, RotatorType::FhtKacRotator, 42)
    }

    fn make_query(rotator: &DynamicRotator, query: &[f32], ex_bits: usize) -> RaBitQQuery {
        rabitq::prepare_query(rotator, query, ex_bits, Metric::L2)
    }

    #[test]
    fn search_finds_nearest_centroid() {
        let rotator = make_rotator(4);
        let centroids = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![10.0, 10.0, 10.0, 10.0],
            vec![20.0, 20.0, 20.0, 20.0],
        ];
        let ids = vec![100, 200, 300];
        let index = CentroidIndex::build(centroids, ids, &rotator, 2, Metric::L2);

        let q = vec![1.0, 1.0, 1.0, 1.0];
        let query = make_query(&rotator, &q, 2);
        let results = index.search(&query, 3);
        assert_eq!(results[0].0, 100);

        let q2 = vec![19.0, 19.0, 19.0, 19.0];
        let query2 = make_query(&rotator, &q2, 2);
        let results = index.search(&query2, 3);
        assert_eq!(results[0].0, 300);
    }

    #[test]
    fn single_centroid() {
        let rotator = make_rotator(4);
        let centroids = vec![vec![5.0, 5.0, 5.0, 5.0]];
        let ids = vec![42];
        let index = CentroidIndex::build(centroids, ids, &rotator, 2, Metric::L2);

        let q = vec![0.0, 0.0, 0.0, 0.0];
        let query = make_query(&rotator, &q, 2);
        let results = index.search(&query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn search_clamps_k() {
        let rotator = make_rotator(4);
        let centroids = vec![vec![0.0; 4], vec![10.0; 4]];
        let ids = vec![1, 2];
        let index = CentroidIndex::build(centroids, ids, &rotator, 2, Metric::L2);

        let q = vec![0.0; 4];
        let query = make_query(&rotator, &q, 2);
        let results = index.search(&query, 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn save_load_roundtrip() {
        let rotator = make_rotator(8);
        let centroids = vec![
            vec![0.0; 8],
            (0..8).map(|i| i as f32).collect(),
            (0..8).map(|i| (i * 2) as f32).collect(),
        ];
        let ids = vec![100, 200, 300];
        let index =
            CentroidIndex::build(centroids, ids.clone(), &rotator, 2, Metric::L2);

        let bytes = index.save_to_bytes().unwrap();
        assert!(!bytes.is_empty());

        let loaded =
            CentroidIndex::load_from_bytes(&bytes, ids, 8, Metric::L2).unwrap();

        let q: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let query = make_query(&rotator, &q, 2);
        let results_orig = index.search(&query, 3);
        let results_loaded = loaded.search(&query, 3);

        assert_eq!(results_orig[0].0, results_loaded[0].0);
        assert_eq!(results_orig.len(), results_loaded.len());
    }

    #[test]
    fn quantization_preserves_nearest() {
        // High-dim test: quantization noise shouldn't flip nearest results
        // when the query is drawn from one of the centroids.
        let dims = 768;
        let rotator = make_rotator(dims);
        let mut centroids = Vec::new();
        for i in 0..10 {
            centroids.push((0..dims).map(|d| (i * 100 + d) as f32 * 0.001).collect());
        }
        let ids: Vec<u32> = (0..10).collect();
        let index = CentroidIndex::build(centroids.clone(), ids, &rotator, 2, Metric::L2);

        let q: Vec<f32> = (0..dims).map(|d| (5 * 100 + d) as f32 * 0.001).collect();
        let query = make_query(&rotator, &q, 2);
        let results = index.search(&query, 3);
        // Nearest should be centroid 5 (which we constructed the query from).
        assert_eq!(results[0].0, 5);
    }
}
