#[inline]
pub(super) fn score_batch(
    weight: f32,
    cache: &[f32; 256],
    fieldnorms: &[u8],
    freqs: &[u32],
    scores: &mut [f32],
) {
    assert_eq!(fieldnorms.len(), freqs.len());
    assert_eq!(freqs.len(), scores.len());
    #[cfg(target_arch = "x86_64")]
    if scores.len() >= 8 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is available and all slices have equal lengths.
        unsafe { score_avx2(weight, cache, fieldnorms, freqs, scores) };
        return;
    }
    score_portable(weight, cache, fieldnorms, freqs, scores);
}

#[inline]
fn score_portable(
    weight: f32,
    cache: &[f32; 256],
    fieldnorms: &[u8],
    freqs: &[u32],
    scores: &mut [f32],
) {
    for ((score, &fieldnorm), &freq) in scores.iter_mut().zip(fieldnorms).zip(freqs) {
        let freq = freq as f32;
        *score = weight * (freq / (freq + cache[fieldnorm as usize]));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn score_avx2(
    weight: f32,
    cache: &[f32; 256],
    fieldnorms: &[u8],
    freqs: &[u32],
    scores: &mut [f32],
) {
    use std::arch::x86_64::*;

    let vector_weight = _mm256_set1_ps(weight);
    let low_mask = _mm256_set1_epi32(0xffff);
    let high_scale = _mm256_set1_ps(65536.0);
    let vector_len = scores.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: Each load/store covers eight in-bounds elements; u8 indices fit the cache.
        unsafe {
            let ids = _mm256_cvtepu8_epi32(_mm_loadl_epi64(fieldnorms.as_ptr().add(offset).cast()));
            let norms = _mm256_i32gather_ps::<4>(cache.as_ptr(), ids);
            let integers = _mm256_loadu_si256(freqs.as_ptr().add(offset).cast());
            // Exact 16-bit halves preserve unsigned-to-float rounding across the full u32 range.
            let low = _mm256_cvtepi32_ps(_mm256_and_si256(integers, low_mask));
            let high = _mm256_cvtepi32_ps(_mm256_srli_epi32::<16>(integers));
            let freq = _mm256_add_ps(_mm256_mul_ps(high, high_scale), low);
            let score = _mm256_mul_ps(
                vector_weight,
                _mm256_div_ps(freq, _mm256_add_ps(freq, norms)),
            );
            _mm256_storeu_ps(scores.as_mut_ptr().add(offset), score);
        }
    }
    score_portable(
        weight,
        cache,
        &fieldnorms[vector_len..],
        &freqs[vector_len..],
        &mut scores[vector_len..],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    type Scorer = fn(f32, &[f32; 256], &[u8], &[u32], &mut [f32]);

    fn check_scores(scorer: Scorer) {
        let cache = std::array::from_fn(|i| (i as f32 + 0.25) * (i as f32 + 0.5));
        let fieldnorms: Vec<u8> = (0..272).map(|i| i as u8).collect();
        let edge_freqs = [
            0,
            1,
            2,
            127,
            128,
            65535,
            65536,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            (1 << 24) + 3,
            i32::MAX as u32,
            1 << 31,
            (1 << 31) + 1,
            u32::MAX - 128,
            u32::MAX,
        ];
        for random in [false, true] {
            let freqs: Vec<u32> = (0..272)
                .map(|i| {
                    if random {
                        (i as u32).wrapping_mul(2654435761)
                    } else {
                        edge_freqs[i % edge_freqs.len()]
                    }
                })
                .collect();
            for weight in [0.0, -0.0, 0.25, 2.2, -3.0, f32::MIN_POSITIVE, 1e30] {
                for offset in 0..8 {
                    for len in 0..=264 {
                        let end = offset + len;
                        let mut scores = vec![f32::NEG_INFINITY; len + 2];
                        scorer(
                            weight,
                            &cache,
                            &fieldnorms[offset..end],
                            &freqs[offset..end],
                            &mut scores[1..len + 1],
                        );
                        assert_eq!(scores[0], f32::NEG_INFINITY);
                        assert_eq!(scores[len + 1], f32::NEG_INFINITY);
                        for i in 0..len {
                            let freq = freqs[offset + i] as f32;
                            let expected =
                                weight * (freq / (freq + cache[fieldnorms[offset + i] as usize]));
                            assert_eq!(
                                scores[i + 1].to_bits(),
                                expected.to_bits(),
                                "offset={offset}, len={len}, i={i}, weight={weight}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn portable_matches_scalar() {
        check_scores(score_portable);
    }

    #[test]
    fn dispatch_matches_scalar() {
        check_scores(score_batch);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 unavailable; kernel not executed");
            return;
        }
        check_scores(|weight, cache, norms, freqs, scores| unsafe {
            score_avx2(weight, cache, norms, freqs, scores)
        });
    }

    #[test]
    #[should_panic]
    fn mismatched_norms_rejected() {
        score_batch(1.0, &[1.0; 256], &[0; 7], &[1; 8], &mut [0.0; 8]);
    }

    #[test]
    #[should_panic]
    fn mismatched_output_rejected() {
        score_batch(1.0, &[1.0; 256], &[0; 8], &[1; 8], &mut [0.0; 7]);
    }
}
