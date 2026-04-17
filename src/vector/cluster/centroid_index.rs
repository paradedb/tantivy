use crate::vector::rabitq::math::l2_distance_sqr;
use crate::vector::rabitq::Metric;

/// Per-centroid int4 quantization.
///
/// Each centroid is stored as a `f32` scale plus `dims` signed 4-bit values
/// in the range `[-8, 7]`, packed two per byte (low nibble = dim 2*i,
/// high nibble = dim 2*i+1). The original f32 is recovered as
/// `i4_value as f32 * scale`. With one scale per centroid the worst
/// quantization error is `max_abs(centroid) / 7`, ~18× coarser than the
/// i8 scheme but still typically well inside k-means cluster boundaries —
/// recall impact measured empirically per index.
///
/// File-on-disk size for `dims=768`:
///   - f32 storage: 768 * 4 = 3072 bytes per centroid
///   - i8 storage:  4 (scale) + 768 = 772 bytes per centroid
///   - i4 storage:  4 (scale) + 384 = 388 bytes per centroid
///   - ~2× smaller than i8, ~8× smaller than raw f32.
///
/// Requires `dims` to be even.
pub struct CentroidIndex {
    pub(crate) centroid_ids: Vec<u32>,
    /// One scale per centroid.
    scales: Vec<f32>,
    /// Packed i4 layout: centroid `i` occupies `data[i*(dims/2)..(i+1)*(dims/2)]`.
    /// Within each byte, the low nibble is the smaller-index dim.
    data: Vec<u8>,
    dims: usize,
}

#[inline(always)]
fn pack_i4_pair(a: i8, b: i8) -> u8 {
    ((a as u8) & 0x0F) | (((b as u8) & 0x0F) << 4)
}

#[inline(always)]
fn unpack_i4_pair(byte: u8) -> (i8, i8) {
    // Sign-extend 4-bit values: if the high bit of the nibble is set, OR in 0xF0.
    let lo_nib = byte & 0x0F;
    let hi_nib = (byte >> 4) & 0x0F;
    let lo = if lo_nib & 0x08 != 0 { (lo_nib | 0xF0) as i8 } else { lo_nib as i8 };
    let hi = if hi_nib & 0x08 != 0 { (hi_nib | 0xF0) as i8 } else { hi_nib as i8 };
    (lo, hi)
}

impl CentroidIndex {
    pub fn build(centroids: Vec<Vec<f32>>, centroid_ids: Vec<u32>, _metric: Metric) -> Self {
        assert_eq!(centroids.len(), centroid_ids.len());
        let dims = centroids.first().map_or(0, |v| v.len());
        assert!(dims % 2 == 0, "CentroidIndex requires even dims for i4 packing; got {dims}");
        let n = centroids.len();
        let packed_per_centroid = dims / 2;
        let mut scales = Vec::with_capacity(n);
        let mut data = Vec::with_capacity(n * packed_per_centroid);
        for c in &centroids {
            let max_abs = c.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales.push(scale);
            let inv = 1.0 / scale;
            for chunk in c.chunks_exact(2) {
                let q0 = (chunk[0] * inv).round().clamp(-8.0, 7.0) as i8;
                let q1 = (chunk[1] * inv).round().clamp(-8.0, 7.0) as i8;
                data.push(pack_i4_pair(q0, q1));
            }
        }
        Self {
            centroid_ids,
            scales,
            data,
            dims,
        }
    }

    /// Dequantize centroid `i` into the provided f32 buffer.
    fn dequantize(&self, i: usize, out: &mut [f32]) {
        let scale = self.scales[i];
        let packed_per_centroid = self.dims / 2;
        let src = &self.data[i * packed_per_centroid..(i + 1) * packed_per_centroid];
        for (idx, &byte) in src.iter().enumerate() {
            let (lo, hi) = unpack_i4_pair(byte);
            out[idx * 2] = (lo as f32) * scale;
            out[idx * 2 + 1] = (hi as f32) * scale;
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let n = self.centroid_ids.len();
        if n == 0 {
            return vec![];
        }
        let k = k.min(n);

        let mut buf = vec![0.0f32; self.dims];
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            self.dequantize(i, &mut buf);
            results.push((self.centroid_ids[i], l2_distance_sqr(query, &buf)));
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
    ///   f32[n] scales
    ///   u8[n * dims/2] packed i4 values (low nibble = dim 2i, high = dim 2i+1)
    pub fn save_to_bytes(&self) -> crate::Result<Vec<u8>> {
        let n = self.centroid_ids.len() as u32;
        let mut buf = Vec::with_capacity(8 + self.scales.len() * 4 + self.data.len());
        buf.extend_from_slice(&n.to_le_bytes());
        buf.extend_from_slice(&(self.dims as u32).to_le_bytes());
        let scale_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.scales.as_ptr() as *const u8,
                self.scales.len() * std::mem::size_of::<f32>(),
            )
        };
        buf.extend_from_slice(scale_bytes);
        buf.extend_from_slice(&self.data);
        Ok(buf)
    }

    pub fn load_from_bytes(
        bytes: &[u8],
        centroid_ids: Vec<u32>,
        dims: usize,
        _metric: Metric,
    ) -> crate::Result<Self> {
        let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let _stored_dims =
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let mut p = 8;

        let mut scales = Vec::with_capacity(n);
        for _ in 0..n {
            scales.push(f32::from_le_bytes([
                bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3],
            ]));
            p += 4;
        }

        let packed_per_centroid = dims / 2;
        let total = n * packed_per_centroid;
        let data = bytes[p..p + total].to_vec();

        Ok(Self {
            centroid_ids,
            scales,
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
            vec![0.0, 0.0, 0.0, 0.0],
            vec![5.0, 5.0, 5.0, 5.0],
            vec![7.0, 7.0, 7.0, 7.0],
        ];
        let ids = vec![100, 200, 300];
        let index = CentroidIndex::build(centroids, ids.clone(), Metric::L2);

        let bytes = index.save_to_bytes().unwrap();
        assert!(!bytes.is_empty());

        let loaded = CentroidIndex::load_from_bytes(&bytes, ids, 4, Metric::L2).unwrap();

        let results_orig = index.search(&[1.0, 1.0, 1.0, 1.0], 3);
        let results_loaded = loaded.search(&[1.0, 1.0, 1.0, 1.0], 3);

        assert_eq!(results_orig[0].0, results_loaded[0].0);
        assert_eq!(results_orig.len(), results_loaded.len());
    }

    #[test]
    fn quantization_preserves_nearest() {
        // High-dim test that quantization noise doesn't flip nearest results.
        let mut centroids = Vec::new();
        for i in 0..10 {
            centroids.push((0..768).map(|d| (i * 100 + d) as f32 * 0.001).collect());
        }
        let ids: Vec<u32> = (0..10).collect();
        let index = CentroidIndex::build(centroids.clone(), ids, Metric::L2);

        let query: Vec<f32> = (0..768).map(|d| (5 * 100 + d) as f32 * 0.001).collect();
        let results = index.search(&query, 3);
        // Nearest should be centroid 5 (which we constructed query from).
        assert_eq!(results[0].0, 5);
    }
}
