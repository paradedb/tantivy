//! Sign-plane encoding and popcount scoring kernels.

use quant_model::f16::{f16_to_f32, f32_to_f16};

#[derive(Clone, Debug)]
pub struct QueryPlanes {
    pub planes: Vec<Vec<u64>>,
    pub lo: f32,
    pub delta: f32,
    pub sum_codes: u64,
    d: usize,
}

fn packed_words(d: usize) -> usize {
    assert!(d > 0);
    d.div_ceil(64)
}

fn tail_is_zero(words: &[u64], d: usize) -> bool {
    if d == 0 || words.len() != packed_words(d) {
        return false;
    }
    let tail = d % 64;
    tail == 0 || words.last().is_some_and(|word| word >> tail == 0)
}

pub fn pack(y: &[f32], out: &mut [u64]) {
    assert!(!y.is_empty());
    assert_eq!(out.len(), packed_words(y.len()));
    out.fill(0);
    for (i, &value) in y.iter().enumerate() {
        if value > 0.0 {
            out[i / 64] |= 1_u64 << (i % 64);
        }
    }
    debug_assert!(tail_is_zero(out, y.len()));
}

pub fn unpack(bits: &[u64], d: usize) -> Vec<f32> {
    assert!(d > 0);
    assert_eq!(bits.len(), packed_words(d));
    debug_assert!(tail_is_zero(bits, d));
    (0..d)
        .map(|i| {
            if bits[i / 64] & (1_u64 << (i % 64)) != 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Encode signs and return the mean-absolute-value scale after f16 rounding.
pub fn encode(y: &[f32], out_bits: &mut [u64]) -> u16 {
    assert!(!y.is_empty());
    assert_eq!(out_bits.len(), packed_words(y.len()));
    pack(y, out_bits);
    if y.iter().all(|&value| value == 0.0) {
        return 0;
    }
    let scale = y.iter().map(|value| value.abs()).sum::<f32>() / y.len() as f32;
    f32_to_f16(scale)
}

/// Hamming distance between two sign-code vectors.
pub fn score_sym(x: &[u64], q: &[u64]) -> u32 {
    assert!(!x.is_empty());
    assert_eq!(x.len(), q.len());
    x.iter().zip(q).map(|(&a, &b)| (a ^ b).count_ones()).sum()
}

pub fn prepare_query(u: &[f32], bq: u8) -> QueryPlanes {
    assert!(!u.is_empty());
    assert!((1..=8).contains(&bq));
    debug_assert!(u.iter().all(|value| value.is_finite()));

    let lo = u.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = u.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let levels = (1_u16 << bq) - 1;
    let delta = if hi == lo {
        0.0
    } else {
        (hi - lo) / f32::from(levels)
    };
    let mut planes = vec![vec![0_u64; packed_words(u.len())]; bq as usize];
    let mut sum_codes = 0_u64;

    if delta != 0.0 {
        for (i, &value) in u.iter().enumerate() {
            let code = (((value - lo) / delta).round() as u16).min(levels);
            sum_codes += u64::from(code);
            for bit in 0..bq {
                if code & (1_u16 << bit) != 0 {
                    planes[bit as usize][i / 64] |= 1_u64 << (i % 64);
                }
            }
        }
    }

    debug_assert!(planes.iter().all(|plane| tail_is_zero(plane, u.len())));
    QueryPlanes {
        planes,
        lo,
        delta,
        sum_codes,
        d: u.len(),
    }
}

/// Return positive-sign count `P` and query-code sum over positive signs `S`.
pub fn score_asym(x: &[u64], q: &QueryPlanes) -> (u32, u64) {
    assert!(!x.is_empty());
    assert!(!q.planes.is_empty());
    assert!(q.planes.iter().all(|plane| plane.len() == x.len()));
    debug_assert!(tail_is_zero(x, q.d));
    debug_assert!(q.planes.iter().all(|plane| tail_is_zero(plane, q.d)));
    let positives = x.iter().map(|word| word.count_ones()).sum();
    let weighted_sum = q
        .planes
        .iter()
        .enumerate()
        .map(|(bit, plane)| {
            let intersections: u64 = x
                .iter()
                .zip(plane)
                .map(|(&signs, &query)| u64::from((signs & query).count_ones()))
                .sum();
            intersections << bit
        })
        .sum();
    (positives, weighted_sum)
}

pub fn estimate_asym(x: &[u64], scale: u16, q: &QueryPlanes) -> f32 {
    let (positives, weighted_sum) = score_asym(x, q);
    let d = q.d as i64;
    let sign_sum = i64::from(positives) * 2 - d;
    let code_sum = i128::from(weighted_sum) * 2 - i128::from(q.sum_codes);
    f16_to_f32(scale) * (q.lo * sign_sum as f32 + q.delta * code_sum as f32)
}

pub fn estimate_fp(x: &[u64], scale: u16, query: &[f32]) -> f32 {
    assert!(!query.is_empty());
    assert_eq!(x.len(), packed_words(query.len()));
    debug_assert!(tail_is_zero(x, query.len()));
    let signed_dot: f32 = query
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            if x[i / 64] & (1_u64 << (i % 64)) != 0 {
                value
            } else {
                -value
            }
        })
        .sum();
    f16_to_f32(scale) * signed_dot
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;

    #[test]
    fn packing_round_trip() {
        let values: Vec<f32> = (0..128)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut packed = [0_u64; 2];
        pack(&values, &mut packed);
        assert_eq!(unpack(&packed, 128), values);
    }

    #[test]
    fn odd_dimension_round_trip_and_zero_tail() {
        for d in [65, 100, 300, 769] {
            let values: Vec<f32> = (0..d)
                .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
                .collect();
            let mut packed = vec![u64::MAX; packed_words(d)];
            pack(&values, &mut packed);
            assert!(tail_is_zero(&packed, d), "d={d}: {packed:x?}");
            assert_eq!(unpack(&packed, d), values);

            let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.13).sin()).collect();
            let prepared = prepare_query(&query, 4);
            assert!(
                prepared.planes.iter().all(|plane| tail_is_zero(plane, d)),
                "d={d}"
            );
            let scale = f32_to_f16(0.25);
            let direct = estimate_fp(&packed, scale, &query);
            let asymmetric = estimate_asym(&packed, scale, &prepared);
            let quantized_query: Vec<f32> = (0..d)
                .map(|i| {
                    let code: u16 = prepared
                        .planes
                        .iter()
                        .enumerate()
                        .map(|(bit, plane)| (((plane[i / 64] >> (i % 64)) & 1) as u16) << bit)
                        .sum();
                    prepared.lo + prepared.delta * f32::from(code)
                })
                .collect();
            let quantized_direct = estimate_fp(&packed, scale, &quantized_query);
            assert!((asymmetric - quantized_direct).abs() < 2e-4, "d={d}");
            assert!(direct.is_finite());

            let mut query_signs = vec![0_u64; packed_words(d)];
            let query_scale = encode(&query, &mut query_signs);
            let decoded_dot = dot(
                &unpack(&packed, d)
                    .into_iter()
                    .map(|sign| sign * f16_to_f32(scale))
                    .collect::<Vec<_>>(),
                &unpack(&query_signs, d)
                    .into_iter()
                    .map(|sign| sign * f16_to_f32(query_scale))
                    .collect::<Vec<_>>(),
            );
            let sign_sum = d as i64 - 2 * i64::from(score_sym(&packed, &query_signs));
            let popcount = f16_to_f32(scale) * f16_to_f32(query_scale) * sign_sum as f32;
            assert!((decoded_dot - popcount).abs() < 1e-5, "d={d}");
        }
    }

    #[test]
    fn symmetric_hamming_score() {
        assert_eq!(score_sym(&[0, u64::MAX], &[u64::MAX, u64::MAX]), 64);
    }

    #[test]
    fn decode_then_dot_matches_symmetric_estimate() {
        let d = 768;
        let mut rng = ChaCha8Rng::seed_from_u64(0x0044_4543_4f44_4501);
        for _ in 0..100 {
            let data = random_unit(&mut rng, d);
            let query = random_unit(&mut rng, d);
            let mut data_bits = vec![0_u64; packed_words(d)];
            let mut query_bits = vec![0_u64; packed_words(d)];
            let data_scale = f16_to_f32(encode(&data, &mut data_bits));
            let query_scale = f16_to_f32(encode(&query, &mut query_bits));
            let decoded_data: Vec<f32> = unpack(&data_bits, d)
                .into_iter()
                .map(|sign| sign * data_scale)
                .collect();
            let decoded_query: Vec<f32> = unpack(&query_bits, d)
                .into_iter()
                .map(|sign| sign * query_scale)
                .collect();
            let decoded_dot = dot(&decoded_data, &decoded_query);
            let sign_sum = d as i64 - 2 * i64::from(score_sym(&data_bits, &query_bits));
            let popcount_estimate = data_scale * query_scale * sign_sum as f32;
            let absolute = (decoded_dot - popcount_estimate).abs();
            let relative = absolute / popcount_estimate.abs().max(f32::MIN_POSITIVE);
            assert!(
                absolute <= 1e-7 || relative <= 1e-5,
                "decoded={decoded_dot}, popcount={popcount_estimate}, absolute={absolute}, \
                 relative={relative}"
            );
        }
    }

    #[test]
    fn formula_exactness_and_constant_query() {
        let signs: Vec<f32> = (0..128)
            .map(|i| if i % 5 < 2 { 1.0 } else { -1.0 })
            .collect();
        let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut bits = [0_u64; 2];
        pack(&signs, &mut bits);
        let prepared = prepare_query(&query, 4);
        let (positives, sum) = score_asym(&bits, &prepared);
        let combined = prepared.lo * (2_i64 * i64::from(positives) - 128) as f32
            + prepared.delta * (2_i128 * i128::from(sum) - i128::from(prepared.sum_codes)) as f32;
        let direct: f32 = signs
            .iter()
            .enumerate()
            .map(|(i, &sign)| {
                let code: u16 = prepared
                    .planes
                    .iter()
                    .enumerate()
                    .map(|(bit, plane)| (((plane[i / 64] >> (i % 64)) & 1) as u16) << bit)
                    .sum();
                sign * (prepared.lo + prepared.delta * f32::from(code))
            })
            .sum();
        assert!((combined - direct).abs() < 2e-4, "{combined} != {direct}");

        let constant = vec![0.375; 128];
        let prepared = prepare_query(&constant, 4);
        assert_eq!(prepared.delta, 0.0);
        assert_eq!(prepared.sum_codes, 0);
        assert!(prepared.planes.iter().flatten().all(|&word| word == 0));
        let scale = f32_to_f16(0.25);
        let estimated = estimate_asym(&bits, scale, &prepared);
        let direct: f32 = signs.iter().map(|&sign| sign * 0.375 * 0.25).sum();
        assert!((estimated - direct).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_is_canonical() {
        let mut bits = [u64::MAX; 2];
        let scale = encode(&[0.0; 128], &mut bits);
        assert_eq!(scale, 0);
        assert_eq!(bits, [0, 0]);
        assert_eq!(estimate_fp(&bits, scale, &[0.25; 128]), 0.0);
        assert_eq!(
            estimate_asym(&bits, scale, &prepare_query(&[0.25; 128], 4)),
            0.0
        );
    }

    #[test]
    fn sign_scale_matches_reference_reconstruction() {
        let d = 768;
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let mut encoded_error = 0.0_f64;
        let mut reference_error = 0.0_f64;
        let mut energy = 0.0_f64;
        for _ in 0..1_000 {
            let vector = random_unit(&mut rng, d);
            let mut bits = vec![0_u64; packed_words(d)];
            let scale = f16_to_f32(encode(&vector, &mut bits));
            let reference_scale = f16_to_f32(f32_to_f16(
                vector.iter().map(|value| value.abs()).sum::<f32>() / d as f32,
            ));
            for &value in &vector {
                let sign = if value > 0.0 { 1.0 } else { -1.0 };
                encoded_error += f64::from(value - scale * sign).powi(2);
                reference_error += f64::from(value - reference_scale * sign).powi(2);
                energy += f64::from(value).powi(2);
            }
        }
        let encoded_rho = (encoded_error / energy).sqrt();
        let reference_rho = (reference_error / energy).sqrt();
        assert!((encoded_rho - reference_rho).abs() < 1e-6);
    }

    #[test]
    fn statistical_estimators() {
        let d = 768;
        let mut rng = ChaCha8Rng::seed_from_u64(0x51_47_4e);
        let queries: Vec<Vec<f32>> = (0..100).map(|_| random_unit(&mut rng, d)).collect();
        let q4: Vec<QueryPlanes> = queries
            .iter()
            .map(|query| prepare_query(query, 4))
            .collect();
        let q1: Vec<QueryPlanes> = queries
            .iter()
            .map(|query| prepare_query(query, 1))
            .collect();
        let mut truth_energy = 0.0_f64;
        let mut fp_error = 0.0_f64;
        let mut q4_error = 0.0_f64;
        let mut q1_error = 0.0_f64;
        for _ in 0..1_000 {
            let vector = random_unit(&mut rng, d);
            let mut bits = vec![0_u64; packed_words(d)];
            let scale = encode(&vector, &mut bits);
            for ((query, q4), q1) in queries.iter().zip(&q4).zip(&q1) {
                let truth = dot(query, &vector);
                truth_energy += f64::from(truth).powi(2);
                fp_error += f64::from(estimate_fp(&bits, scale, query) - truth).powi(2);
                q4_error += f64::from(estimate_asym(&bits, scale, q4) - truth).powi(2);
                q1_error += f64::from(estimate_asym(&bits, scale, q1) - truth).powi(2);
            }
        }
        let fp_rho = (fp_error / truth_energy).sqrt();
        let q4_rho = (q4_error / truth_energy).sqrt();
        let q1_rho = (q1_error / truth_energy).sqrt();
        assert!(
            (fp_rho - 0.6025).abs() <= 0.010,
            "fp={fp_rho}, q4={q4_rho}, q1={q1_rho}"
        );
        assert!(q4_rho <= fp_rho * 1.05, "fp={fp_rho}, q4={q4_rho}");
        assert!(q1_rho > q4_rho, "q1={q1_rho}, q4={q4_rho}");
    }

    fn random_unit(rng: &mut ChaCha8Rng, d: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(d);
        while values.len() < d {
            let u1 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            let u2 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * u2;
            values.push((radius * angle.cos()) as f32);
            if values.len() < d {
                values.push((radius * angle.sin()) as f32);
            }
        }
        let norm = values
            .iter()
            .map(|&value| value * value)
            .sum::<f32>()
            .sqrt();
        for value in &mut values {
            *value /= norm;
        }
        values
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }
}
