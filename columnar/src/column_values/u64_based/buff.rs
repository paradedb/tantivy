//! BUFF (Byte-sliced) codec for u64-based column values.
//!
//! This codec implements byte-sliced storage inspired by the BUFF paper:
//! "Decomposed Bounded Floats for Fast Compression and Queries" (VLDB 2021).
//!
//! The key innovation is storing values in a byte-sliced format where each byte
//! position across all values is stored contiguously. This enables:
//! - Efficient compression through byte-level delta encoding
//! - Fast predicate evaluation by examining only relevant bytes
//! - SIMD-friendly memory access patterns

use std::io::{self, Write};
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;

use buff_rs::BuffCodec as ExternalBuffCodec;
use common::file_slice::FileSlice;
use common::{BinarySerializable, OwnedBytes};

use crate::ColumnValues;
use crate::column_values::u64_based::{ColumnCodec, ColumnCodecEstimator, ColumnStats};

/// Default scale for BUFF encoding (3 decimal places).
/// This determines the precision of the encoding.
const DEFAULT_SCALE: usize = 1000;

/// BUFF codec reader for accessing compressed column values.
#[derive(Clone)]
pub struct BuffReader {
    /// The compressed data.
    data: OwnedBytes,
    /// Column statistics (min, max, gcd, num_rows).
    stats: ColumnStats,
    /// The BUFF codec used for decoding.
    buff_codec: ExternalBuffCodec,
    /// Cached decoded values (lazily populated).
    decoded: Arc<std::sync::OnceLock<Vec<u64>>>,
}

impl BuffReader {
    /// Decode all values from the compressed data.
    fn decode_all(&self) -> io::Result<Vec<u64>> {
        if self.stats.num_rows == 0 {
            return Ok(Vec::new());
        }

        // Decode using buff-rs
        let f64_values = self
            .buff_codec
            .decode(&self.data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        // Convert f64 back to u64
        // The values were stored as f64 = (u64_val - min_value) / gcd
        // So we reverse: u64_val = f64 * gcd + min_value
        let gcd = self.stats.gcd.get();
        let min_value = self.stats.min_value;

        Ok(f64_values
            .into_iter()
            .map(|f| {
                let scaled = (f.round() as u64).saturating_mul(gcd);
                scaled.saturating_add(min_value)
            })
            .collect())
    }

    /// Get decoded values, using cache if available.
    fn get_decoded(&self) -> &[u64] {
        self.decoded
            .get_or_init(|| self.decode_all().unwrap_or_default())
    }
}

impl ColumnValues for BuffReader {
    #[inline]
    fn get_val(&self, idx: u32) -> u64 {
        self.get_decoded()[idx as usize]
    }

    fn get_vals(&self, indexes: &[u32], output: &mut [u64]) {
        let decoded = self.get_decoded();
        for (out, &idx) in output.iter_mut().zip(indexes.iter()) {
            *out = decoded[idx as usize];
        }
    }

    #[inline]
    fn min_value(&self) -> u64 {
        self.stats.min_value
    }

    #[inline]
    fn max_value(&self) -> u64 {
        self.stats.max_value
    }

    #[inline]
    fn num_vals(&self) -> u32 {
        self.stats.num_rows
    }

    fn get_row_ids_for_value_range(
        &self,
        range: RangeInclusive<u64>,
        doc_id_range: Range<u32>,
        positions: &mut Vec<u32>,
    ) {
        positions.clear();
        let decoded = self.get_decoded();
        for doc_id in doc_id_range {
            let val = decoded[doc_id as usize];
            if range.contains(&val) {
                positions.push(doc_id);
            }
        }
    }
}

/// Estimator for the BUFF codec.
///
/// Collects values during the first pass and estimates the compressed size.
#[derive(Default)]
pub struct BuffCodecEstimator {
    /// Collected values for size estimation.
    values: Vec<u64>,
}

impl ColumnCodecEstimator for BuffCodecEstimator {
    fn collect(&mut self, value: u64) {
        self.values.push(value);
    }

    fn finalize(&mut self) {
        // No additional finalization needed
    }

    fn estimate(&self, stats: &ColumnStats) -> Option<u64> {
        if stats.num_rows == 0 {
            return Some(stats.num_bytes());
        }

        // Convert values to f64 for BUFF encoding
        let gcd = stats.gcd.get();
        let min_value = stats.min_value;

        let f64_values: Vec<f64> = self
            .values
            .iter()
            .map(|&v| ((v - min_value) / gcd) as f64)
            .collect();

        // Try encoding to estimate size
        let buff_codec = ExternalBuffCodec::new(DEFAULT_SCALE);
        let encoded = buff_codec.encode(&f64_values).ok()?;

        // Total size = stats header + BUFF encoded data
        Some(stats.num_bytes() + encoded.len() as u64)
    }

    fn serialize(
        &self,
        stats: &ColumnStats,
        vals: &mut dyn Iterator<Item = u64>,
        wrt: &mut dyn Write,
    ) -> io::Result<()> {
        // Write stats header
        stats.serialize(wrt)?;

        if stats.num_rows == 0 {
            return Ok(());
        }

        // Collect all values
        let values: Vec<u64> = vals.collect();

        // Convert to f64 (normalized by GCD and shifted by min)
        let gcd = stats.gcd.get();
        let min_value = stats.min_value;

        let f64_values: Vec<f64> = values
            .iter()
            .map(|&v| ((v - min_value) / gcd) as f64)
            .collect();

        // Encode with BUFF
        let buff_codec = ExternalBuffCodec::new(DEFAULT_SCALE);
        let encoded = buff_codec
            .encode(&f64_values)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        // Write encoded data
        wrt.write_all(&encoded)?;

        Ok(())
    }
}

/// BUFF codec for u64-based column values.
pub struct BuffCodec;

impl ColumnCodec for BuffCodec {
    type ColumnValues = BuffReader;
    type Estimator = BuffCodecEstimator;

    fn load(file_slice: FileSlice) -> io::Result<Self::ColumnValues> {
        let (stats, data_slice) = ColumnStats::deserialize_from_tail(file_slice)?;

        let data = data_slice.read_bytes()?;

        Ok(BuffReader {
            data,
            stats,
            buff_codec: ExternalBuffCodec::new(DEFAULT_SCALE),
            decoded: Arc::new(std::sync::OnceLock::new()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column_values::u64_based::tests::create_and_validate;

    #[test]
    fn test_buff_simple() {
        create_and_validate::<BuffCodec>(&[4, 3, 12], "simple");
    }

    #[test]
    fn test_buff_with_gcd() {
        create_and_validate::<BuffCodec>(&[1000, 2000, 3000], "gcd");
    }

    #[test]
    fn test_buff_single_value() {
        create_and_validate::<BuffCodec>(&[42], "single");
    }

    #[test]
    fn test_buff_constant() {
        create_and_validate::<BuffCodec>(&[100; 50], "constant");
    }

    #[test]
    fn test_buff_ascending() {
        let data: Vec<u64> = (0..100).collect();
        create_and_validate::<BuffCodec>(&data, "ascending");
    }

    #[test]
    fn test_buff_descending() {
        let data: Vec<u64> = (0..100).rev().collect();
        create_and_validate::<BuffCodec>(&data, "descending");
    }

    #[test]
    fn test_buff_large_values() {
        create_and_validate::<BuffCodec>(
            &[u64::MAX / 2, u64::MAX / 2 + 1, u64::MAX / 2 + 2],
            "large",
        );
    }
}
