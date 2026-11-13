#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::array::from_fn;

#[cfg(target_arch = "aarch64")]
use super::finish_asym;
use super::{estimate_asym_batch_unscaled, QueryPlanes};

const QUERY_BITS: usize = 4;
const GROUP_BITS: usize = 4;
const TABLE_SIZE: usize = 1 << GROUP_BITS;
#[cfg(target_arch = "aarch64")]
const GROUP_MASK: u8 = (TABLE_SIZE - 1) as u8;
const TILE_ROWS: usize = 16;
#[cfg(target_arch = "aarch64")]
const HALF_TILE_ROWS: usize = TILE_ROWS / 2;
#[cfg(target_arch = "aarch64")]
const WORD_BYTES: usize = size_of::<u64>();
const WORD_BITS: usize = u64::BITS as usize;
const MAX_QUERY_CODE: usize = (1 << QUERY_BITS) - 1;
const MAX_WORDS: usize = u16::MAX as usize / (WORD_BITS * MAX_QUERY_CODE);

/// Exact integer subset-sum tables for dense sign scoring.
pub struct FastScanQuery {
    query: QueryPlanes,
    tables: Vec<[u8; TABLE_SIZE]>,
}

impl FastScanQuery {
    /// Number of rows in one SIMD tile.
    pub const MIN_ROWS: usize = TILE_ROWS;

    /// Returns a prepared query when the CPU, query width and accumulator range are supported.
    pub fn new(query: &QueryPlanes) -> Option<Self> {
        if !cfg!(all(target_arch = "aarch64", target_endian = "little"))
            || query.planes.len() != QUERY_BITS
            || query.d.div_ceil(WORD_BITS) > MAX_WORDS
        {
            return None;
        }
        Some(Self {
            query: query.clone(),
            tables: build_tables(query),
        })
    }

    /// Scores dense rows with an in-register transpose, using popcount for the final partial tile.
    ///
    /// # Panics
    ///
    /// Panics when the row stride or output shape does not match the prepared query.
    pub fn score_batch(&self, rows: &[u64], words_per_row: usize, out: &mut [f32]) {
        assert_eq!(words_per_row, self.query.d.div_ceil(WORD_BITS));
        assert_eq!(rows.len(), out.len() * words_per_row);
        assert_eq!(self.tables.len(), words_per_row * WORD_BITS / GROUP_BITS);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is available on aarch64 and the row shapes are checked above.
        let scored = unsafe { score_neon(rows, words_per_row, &self.query, &self.tables, out) };
        #[cfg(not(target_arch = "aarch64"))]
        let scored = 0;
        estimate_asym_batch_unscaled(
            &rows[scored * words_per_row..],
            words_per_row,
            &self.query,
            &mut out[scored..],
        );
    }
}

fn build_tables(query: &QueryPlanes) -> Vec<[u8; TABLE_SIZE]> {
    (0..query.d.div_ceil(WORD_BITS) * WORD_BITS / GROUP_BITS)
        .map(|group| {
            let mut values = [0_u8; GROUP_BITS];
            for (offset, value) in values.iter_mut().enumerate() {
                let coordinate = group * GROUP_BITS + offset;
                for (bit, plane) in query.planes.iter().enumerate() {
                    *value |= (((plane[coordinate / WORD_BITS] >> (coordinate % WORD_BITS)) & 1)
                        as u8)
                        << bit;
                }
            }
            from_fn(|signs| {
                values
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| signs & (1 << bit) != 0)
                    .map(|(_, &value)| value)
                    .sum()
            })
        })
        .collect()
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose(rows: [uint8x8_t; HALF_TILE_ROWS]) -> [uint8x8_t; WORD_BYTES] {
    let a = vtrn_u8(rows[0], rows[1]);
    let b = vtrn_u8(rows[2], rows[3]);
    let c = vtrn_u8(rows[4], rows[5]);
    let d = vtrn_u8(rows[6], rows[7]);
    let e = vtrn_u16(vreinterpret_u16_u8(a.0), vreinterpret_u16_u8(b.0));
    let f = vtrn_u16(vreinterpret_u16_u8(a.1), vreinterpret_u16_u8(b.1));
    let g = vtrn_u16(vreinterpret_u16_u8(c.0), vreinterpret_u16_u8(d.0));
    let h = vtrn_u16(vreinterpret_u16_u8(c.1), vreinterpret_u16_u8(d.1));
    let i = vtrn_u32(vreinterpret_u32_u16(e.0), vreinterpret_u32_u16(g.0));
    let j = vtrn_u32(vreinterpret_u32_u16(f.0), vreinterpret_u32_u16(h.0));
    let k = vtrn_u32(vreinterpret_u32_u16(e.1), vreinterpret_u32_u16(g.1));
    let l = vtrn_u32(vreinterpret_u32_u16(f.1), vreinterpret_u32_u16(h.1));
    [
        vreinterpret_u8_u32(i.0),
        vreinterpret_u8_u32(j.0),
        vreinterpret_u8_u32(k.0),
        vreinterpret_u8_u32(l.0),
        vreinterpret_u8_u32(i.1),
        vreinterpret_u8_u32(j.1),
        vreinterpret_u8_u32(k.1),
        vreinterpret_u8_u32(l.1),
    ]
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn score_neon(
    rows: &[u64],
    stride: usize,
    query: &QueryPlanes,
    tables: &[[u8; TABLE_SIZE]],
    out: &mut [f32],
) -> usize {
    let scored = out.len() / TILE_ROWS * TILE_ROWS;
    for row in (0..scored).step_by(TILE_ROWS) {
        let mut counts = [vdupq_n_u16(0); 2];
        let mut sums = [vdupq_n_u16(0); 2];
        for word in 0..stride {
            let lower = transpose(from_fn(|lane| {
                vld1_u8(rows.as_ptr().add((row + lane) * stride + word).cast())
            }));
            let upper = transpose(from_fn(|lane| {
                vld1_u8(
                    rows.as_ptr()
                        .add((row + lane + HALF_TILE_ROWS) * stride + word)
                        .cast(),
                )
            }));
            for byte in 0..WORD_BYTES {
                let signs = vcombine_u8(lower[byte], upper[byte]);
                let table = (word * WORD_BYTES + byte) * 2;
                let low = vqtbl1q_u8(
                    vld1q_u8(tables[table].as_ptr()),
                    vandq_u8(signs, vdupq_n_u8(GROUP_MASK)),
                );
                let high = vqtbl1q_u8(
                    vld1q_u8(tables[table + 1].as_ptr()),
                    vshrq_n_u8::<{ GROUP_BITS as i32 }>(signs),
                );
                let values = vaddq_u8(low, high);
                let positives = vcntq_u8(signs);
                sums[0] = vaddw_u8(sums[0], vget_low_u8(values));
                sums[1] = vaddw_u8(sums[1], vget_high_u8(values));
                counts[0] = vaddw_u8(counts[0], vget_low_u8(positives));
                counts[1] = vaddw_u8(counts[1], vget_high_u8(positives));
            }
        }
        let mut positives = [0_u16; TILE_ROWS];
        let mut weighted = [0_u16; TILE_ROWS];
        vst1q_u16(positives.as_mut_ptr(), counts[0]);
        vst1q_u16(positives.as_mut_ptr().add(HALF_TILE_ROWS), counts[1]);
        vst1q_u16(weighted.as_mut_ptr(), sums[0]);
        vst1q_u16(weighted.as_mut_ptr().add(HALF_TILE_ROWS), sums[1]);
        for lane in 0..TILE_ROWS {
            out[row + lane] =
                finish_asym(u32::from(positives[lane]), u64::from(weighted[lane]), query);
        }
    }
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sign::prepare_query;

    #[test]
    fn fastscan_matches_popcount_with_tails_and_fallbacks() {
        for d in [
            1,
            63,
            64,
            65,
            100,
            768,
            1024,
            MAX_WORDS * WORD_BITS,
            (MAX_WORDS + 1) * WORD_BITS,
        ] {
            for query_bits in [1, 4, 8] {
                let query = prepare_query(
                    &(0..d).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>(),
                    query_bits,
                );
                let fastscan = FastScanQuery::new(&query);
                assert_eq!(
                    fastscan.is_some(),
                    cfg!(all(target_arch = "aarch64", target_endian = "little"))
                        && query_bits == QUERY_BITS as u8
                        && d.div_ceil(WORD_BITS) <= MAX_WORDS,
                );
                let stride = d.div_ceil(WORD_BITS);
                for count in [0, 1, TILE_ROWS - 1, TILE_ROWS, TILE_ROWS + 1, 100] {
                    let rows: Vec<u64> = (0..count * stride)
                        .map(|i| {
                            let value = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                            if i % stride == stride - 1 && d % WORD_BITS != 0 {
                                value & ((1 << (d % WORD_BITS)) - 1)
                            } else {
                                value
                            }
                        })
                        .collect();
                    let mut expected = vec![0.0; count];
                    let mut actual = expected.clone();
                    estimate_asym_batch_unscaled(&rows, stride, &query, &mut expected);
                    if let Some(fastscan) = &fastscan {
                        fastscan.score_batch(&rows, stride, &mut actual);
                    } else {
                        estimate_asym_batch_unscaled(&rows, stride, &query, &mut actual);
                    }
                    assert_eq!(actual, expected, "d={d}, bits={query_bits}, rows={count}");
                }
            }
        }
    }

    #[test]
    fn maximum_accumulators_and_constant_queries_match_popcount() {
        let d = MAX_WORDS * WORD_BITS;
        for constant in [false, true] {
            let mut values = vec![1.0; d];
            if !constant {
                values[0] = -1.0;
            }
            let query = prepare_query(&values, QUERY_BITS as u8);
            let Some(fastscan) = FastScanQuery::new(&query) else {
                continue;
            };
            for word in [0, u64::MAX] {
                let rows = vec![word; TILE_ROWS * MAX_WORDS];
                let mut actual = vec![0.0; TILE_ROWS];
                let mut expected = actual.clone();
                fastscan.score_batch(&rows, MAX_WORDS, &mut actual);
                estimate_asym_batch_unscaled(&rows, MAX_WORDS, &query, &mut expected);
                assert_eq!(actual, expected);
            }
        }
    }
}
