//! Runtime-dispatched operations over contiguous membership bitmap words.

use std::sync::atomic::{AtomicU8, Ordering};

use common::TinySet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum Implementation {
    #[cfg(target_arch = "x86_64")]
    Avx512 = 0,
    #[cfg(target_arch = "x86_64")]
    Avx2 = 1,
    #[cfg(target_arch = "aarch64")]
    Neon = 2,
    Scalar = 3,
}

static IMPLEMENTATION: AtomicU8 = AtomicU8::new(u8::MAX);

#[inline]
fn implementation() -> Implementation {
    let selected = IMPLEMENTATION.load(Ordering::Relaxed);
    if selected != u8::MAX {
        return Implementation::from_u8(selected);
    }
    let selected = Implementation::detect();
    IMPLEMENTATION.store(selected as u8, Ordering::Relaxed);
    selected
}

impl Implementation {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
            {
                return Self::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self::Neon;
        }
        #[allow(unreachable_code)]
        Self::Scalar
    }

    fn from_u8(value: u8) -> Self {
        match value {
            #[cfg(target_arch = "x86_64")]
            value if value == Self::Avx512 as u8 => Self::Avx512,
            #[cfg(target_arch = "x86_64")]
            value if value == Self::Avx2 as u8 => Self::Avx2,
            #[cfg(target_arch = "aarch64")]
            value if value == Self::Neon as u8 => Self::Neon,
            _ => Self::Scalar,
        }
    }
}

#[inline]
pub(crate) fn union_in_place(left: &mut [TinySet], right: &[TinySet]) {
    assert_eq!(left.len(), right.len());
    let left = TinySet::as_u64_slice_mut(left);
    let right = TinySet::as_u64_slice(right);
    match implementation() {
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => unsafe { x86::union_avx512(left, right) },
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => unsafe { x86::union_avx2(left, right) },
        #[cfg(target_arch = "aarch64")]
        Implementation::Neon => unsafe { arm::union_neon(left, right) },
        Implementation::Scalar => scalar_union(left, right),
    }
}

#[inline]
pub(crate) fn intersect_in_place(left: &mut [TinySet], right: &[TinySet]) -> bool {
    assert_eq!(left.len(), right.len());
    let left = TinySet::as_u64_slice_mut(left);
    let right = TinySet::as_u64_slice(right);
    match implementation() {
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => unsafe { x86::intersect_avx512(left, right) },
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => unsafe { x86::intersect_avx2(left, right) },
        #[cfg(target_arch = "aarch64")]
        Implementation::Neon => unsafe { arm::intersect_neon(left, right) },
        Implementation::Scalar => scalar_intersect(left, right),
    }
}

#[inline]
pub(crate) fn exclude_in_place(left: &mut [TinySet], right: &[TinySet]) -> bool {
    assert_eq!(left.len(), right.len());
    let left = TinySet::as_u64_slice_mut(left);
    let right = TinySet::as_u64_slice(right);
    match implementation() {
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => unsafe { x86::exclude_avx512(left, right) },
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => unsafe { x86::exclude_avx2(left, right) },
        #[cfg(target_arch = "aarch64")]
        Implementation::Neon => unsafe { arm::exclude_neon(left, right) },
        Implementation::Scalar => scalar_exclude(left, right),
    }
}

#[inline]
pub(crate) fn count(words: &[TinySet]) -> u32 {
    let words = TinySet::as_u64_slice(words);
    match implementation() {
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => unsafe { x86::count_avx512(words) },
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => unsafe { x86::count_avx2(words) },
        #[cfg(target_arch = "aarch64")]
        Implementation::Neon => unsafe { arm::count_neon(words) },
        Implementation::Scalar => scalar_count(words),
    }
}

#[inline]
pub(crate) fn count_intersection(left: &[TinySet], right: &[TinySet]) -> u32 {
    assert_eq!(left.len(), right.len());
    let left = TinySet::as_u64_slice(left);
    let right = TinySet::as_u64_slice(right);
    match implementation() {
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => unsafe { x86::count_intersection_avx512(left, right) },
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => unsafe { x86::count_intersection_avx2(left, right) },
        #[cfg(target_arch = "aarch64")]
        Implementation::Neon => unsafe { arm::count_intersection_neon(left, right) },
        Implementation::Scalar => scalar_count_intersection(left, right),
    }
}

#[inline]
fn scalar_union(left: &mut [u64], right: &[u64]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left |= *right;
    }
}

#[inline]
fn scalar_intersect(left: &mut [u64], right: &[u64]) -> bool {
    let mut any = 0;
    for (left, right) in left.iter_mut().zip(right) {
        *left &= *right;
        any |= *left;
    }
    any == 0
}

#[inline]
fn scalar_exclude(left: &mut [u64], right: &[u64]) -> bool {
    let mut any = 0;
    for (left, right) in left.iter_mut().zip(right) {
        *left &= !*right;
        any |= *left;
    }
    any == 0
}

#[inline]
fn scalar_count(words: &[u64]) -> u32 {
    words.iter().map(|word| word.count_ones()).sum()
}

#[inline]
fn scalar_count_intersection(left: &[u64], right: &[u64]) -> u32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left & right).count_ones())
        .sum()
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn union_avx512(left: &mut [u64], right: &[u64]) {
        let chunks = left.len() / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let a = unsafe { _mm512_loadu_si512(left.as_ptr().add(offset).cast()) };
            let b = unsafe { _mm512_loadu_si512(right.as_ptr().add(offset).cast()) };
            unsafe {
                _mm512_storeu_si512(left.as_mut_ptr().add(offset).cast(), _mm512_or_si512(a, b))
            };
        }
        super::scalar_union(&mut left[chunks * 8..], &right[chunks * 8..]);
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn intersect_avx512(left: &mut [u64], right: &[u64]) -> bool {
        let chunks = left.len() / 8;
        let mut any = _mm512_setzero_si512();
        for i in 0..chunks {
            let offset = i * 8;
            let a = unsafe { _mm512_loadu_si512(left.as_ptr().add(offset).cast()) };
            let b = unsafe { _mm512_loadu_si512(right.as_ptr().add(offset).cast()) };
            let value = _mm512_and_si512(a, b);
            unsafe { _mm512_storeu_si512(left.as_mut_ptr().add(offset).cast(), value) };
            any = _mm512_or_si512(any, value);
        }
        let tail_empty = super::scalar_intersect(&mut left[chunks * 8..], &right[chunks * 8..]);
        _mm512_test_epi64_mask(any, any) == 0 && tail_empty
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn exclude_avx512(left: &mut [u64], right: &[u64]) -> bool {
        let chunks = left.len() / 8;
        let mut any = _mm512_setzero_si512();
        for i in 0..chunks {
            let offset = i * 8;
            let a = unsafe { _mm512_loadu_si512(left.as_ptr().add(offset).cast()) };
            let b = unsafe { _mm512_loadu_si512(right.as_ptr().add(offset).cast()) };
            let value = _mm512_andnot_si512(b, a);
            unsafe { _mm512_storeu_si512(left.as_mut_ptr().add(offset).cast(), value) };
            any = _mm512_or_si512(any, value);
        }
        let tail_empty = super::scalar_exclude(&mut left[chunks * 8..], &right[chunks * 8..]);
        _mm512_test_epi64_mask(any, any) == 0 && tail_empty
    }

    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub(super) unsafe fn count_avx512(words: &[u64]) -> u32 {
        let chunks = words.len() / 8;
        let mut counts = _mm512_setzero_si512();
        for i in 0..chunks {
            let value = unsafe { _mm512_loadu_si512(words.as_ptr().add(i * 8).cast()) };
            counts = _mm512_add_epi64(counts, _mm512_popcnt_epi64(value));
        }
        let mut lanes = [0u64; 8];
        unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast(), counts) };
        lanes.iter().sum::<u64>() as u32 + super::scalar_count(&words[chunks * 8..])
    }

    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub(super) unsafe fn count_intersection_avx512(left: &[u64], right: &[u64]) -> u32 {
        let chunks = left.len() / 8;
        let mut counts = _mm512_setzero_si512();
        for i in 0..chunks {
            let offset = i * 8;
            let a = unsafe { _mm512_loadu_si512(left.as_ptr().add(offset).cast()) };
            let b = unsafe { _mm512_loadu_si512(right.as_ptr().add(offset).cast()) };
            counts = _mm512_add_epi64(counts, _mm512_popcnt_epi64(_mm512_and_si512(a, b)));
        }
        let mut lanes = [0u64; 8];
        unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast(), counts) };
        lanes.iter().sum::<u64>() as u32
            + super::scalar_count_intersection(&left[chunks * 8..], &right[chunks * 8..])
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn union_avx2(left: &mut [u64], right: &[u64]) {
        let chunks = left.len() / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let a = unsafe { _mm256_loadu_si256(left.as_ptr().add(offset).cast()) };
            let b = unsafe { _mm256_loadu_si256(right.as_ptr().add(offset).cast()) };
            unsafe {
                _mm256_storeu_si256(left.as_mut_ptr().add(offset).cast(), _mm256_or_si256(a, b))
            };
        }
        super::scalar_union(&mut left[chunks * 4..], &right[chunks * 4..]);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn intersect_avx2(left: &mut [u64], right: &[u64]) -> bool {
        let chunks = left.len() / 4;
        let mut any = _mm256_setzero_si256();
        for i in 0..chunks {
            let offset = i * 4;
            let a = unsafe { _mm256_loadu_si256(left.as_ptr().add(offset).cast()) };
            let b = unsafe { _mm256_loadu_si256(right.as_ptr().add(offset).cast()) };
            let value = _mm256_and_si256(a, b);
            unsafe { _mm256_storeu_si256(left.as_mut_ptr().add(offset).cast(), value) };
            any = _mm256_or_si256(any, value);
        }
        let tail_empty = super::scalar_intersect(&mut left[chunks * 4..], &right[chunks * 4..]);
        _mm256_testz_si256(any, any) != 0 && tail_empty
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn exclude_avx2(left: &mut [u64], right: &[u64]) -> bool {
        let chunks = left.len() / 4;
        let mut any = _mm256_setzero_si256();
        for i in 0..chunks {
            let offset = i * 4;
            let a = unsafe { _mm256_loadu_si256(left.as_ptr().add(offset).cast()) };
            let b = unsafe { _mm256_loadu_si256(right.as_ptr().add(offset).cast()) };
            let value = _mm256_andnot_si256(b, a);
            unsafe { _mm256_storeu_si256(left.as_mut_ptr().add(offset).cast(), value) };
            any = _mm256_or_si256(any, value);
        }
        let tail_empty = super::scalar_exclude(&mut left[chunks * 4..], &right[chunks * 4..]);
        _mm256_testz_si256(any, any) != 0 && tail_empty
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn count_avx2(words: &[u64]) -> u32 {
        let chunks = words.len() / 4;
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let low_mask = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        let mut counts = zero;
        for i in 0..chunks {
            let value = unsafe { _mm256_loadu_si256(words.as_ptr().add(i * 4).cast()) };
            let low = _mm256_and_si256(value, low_mask);
            let high = _mm256_and_si256(_mm256_srli_epi16(value, 4), low_mask);
            let bytes = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, low),
                _mm256_shuffle_epi8(lookup, high),
            );
            counts = _mm256_add_epi64(counts, _mm256_sad_epu8(bytes, zero));
        }
        let mut lanes = [0u64; 4];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), counts) };
        lanes.iter().sum::<u64>() as u32 + super::scalar_count(&words[chunks * 4..])
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn count_intersection_avx2(left: &[u64], right: &[u64]) -> u32 {
        let chunks = left.len() / 4;
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let low_mask = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        let mut counts = zero;
        for i in 0..chunks {
            let offset = i * 4;
            let a = unsafe { _mm256_loadu_si256(left.as_ptr().add(offset).cast()) };
            let b = unsafe { _mm256_loadu_si256(right.as_ptr().add(offset).cast()) };
            let value = _mm256_and_si256(a, b);
            let low = _mm256_and_si256(value, low_mask);
            let high = _mm256_and_si256(_mm256_srli_epi16(value, 4), low_mask);
            let bytes = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, low),
                _mm256_shuffle_epi8(lookup, high),
            );
            counts = _mm256_add_epi64(counts, _mm256_sad_epu8(bytes, zero));
        }
        let mut lanes = [0u64; 4];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), counts) };
        lanes.iter().sum::<u64>() as u32
            + super::scalar_count_intersection(&left[chunks * 4..], &right[chunks * 4..])
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn union_neon(left: &mut [u64], right: &[u64]) {
        let chunks = left.len() / 2;
        for i in 0..chunks {
            let offset = i * 2;
            let a = unsafe { vld1q_u64(left.as_ptr().add(offset)) };
            let b = unsafe { vld1q_u64(right.as_ptr().add(offset)) };
            unsafe { vst1q_u64(left.as_mut_ptr().add(offset), vorrq_u64(a, b)) };
        }
        super::scalar_union(&mut left[chunks * 2..], &right[chunks * 2..]);
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn intersect_neon(left: &mut [u64], right: &[u64]) -> bool {
        let chunks = left.len() / 2;
        let mut any = vdupq_n_u64(0);
        for i in 0..chunks {
            let offset = i * 2;
            let a = unsafe { vld1q_u64(left.as_ptr().add(offset)) };
            let b = unsafe { vld1q_u64(right.as_ptr().add(offset)) };
            let value = vandq_u64(a, b);
            unsafe { vst1q_u64(left.as_mut_ptr().add(offset), value) };
            any = vorrq_u64(any, value);
        }
        let tail_empty = super::scalar_intersect(&mut left[chunks * 2..], &right[chunks * 2..]);
        vmaxvq_u8(vreinterpretq_u8_u64(any)) == 0 && tail_empty
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn exclude_neon(left: &mut [u64], right: &[u64]) -> bool {
        let chunks = left.len() / 2;
        let mut any = vdupq_n_u64(0);
        for i in 0..chunks {
            let offset = i * 2;
            let a = unsafe { vld1q_u64(left.as_ptr().add(offset)) };
            let b = unsafe { vld1q_u64(right.as_ptr().add(offset)) };
            let value = vbicq_u64(a, b);
            unsafe { vst1q_u64(left.as_mut_ptr().add(offset), value) };
            any = vorrq_u64(any, value);
        }
        let tail_empty = super::scalar_exclude(&mut left[chunks * 2..], &right[chunks * 2..]);
        vmaxvq_u8(vreinterpretq_u8_u64(any)) == 0 && tail_empty
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn count_neon(words: &[u64]) -> u32 {
        let chunks = words.len() / 2;
        let mut count = 0u32;
        for i in 0..chunks {
            let value = unsafe { vld1q_u8(words.as_ptr().add(i * 2).cast()) };
            count += vaddlvq_u8(vcntq_u8(value)) as u32;
        }
        count + super::scalar_count(&words[chunks * 2..])
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn count_intersection_neon(left: &[u64], right: &[u64]) -> u32 {
        let chunks = left.len() / 2;
        let mut count = 0u32;
        for i in 0..chunks {
            let offset = i * 2;
            let a = unsafe { vld1q_u64(left.as_ptr().add(offset)) };
            let b = unsafe { vld1q_u64(right.as_ptr().add(offset)) };
            count += vaddlvq_u8(vcntq_u8(vreinterpretq_u8_u64(vandq_u64(a, b)))) as u32;
        }
        count + super::scalar_count_intersection(&left[chunks * 2..], &right[chunks * 2..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(values: &[u64]) -> Vec<TinySet> {
        values
            .iter()
            .map(|word| TinySet::deserialize(word.to_le_bytes()))
            .collect()
    }

    #[test]
    fn dispatched_operations_match_scalar() {
        let input: Vec<_> = (0..259)
            .map(|i| {
                (i as u64)
                    .wrapping_mul(0x9e3779b97f4a7c15)
                    .rotate_left(i % 64)
            })
            .collect();
        let other: Vec<_> = (0..259)
            .map(|i| {
                (!(i as u64))
                    .wrapping_mul(0xd6e8feb86659fd93)
                    .rotate_right(i % 64)
            })
            .collect();
        let right = words(&other);

        let mut actual = words(&input);
        union_in_place(&mut actual, &right);
        let expected: Vec<_> = input.iter().zip(&other).map(|(a, b)| a | b).collect();
        assert_eq!(TinySet::as_u64_slice(&actual), expected);

        let mut actual = words(&input);
        let empty = intersect_in_place(&mut actual, &right);
        let expected: Vec<_> = input.iter().zip(&other).map(|(a, b)| a & b).collect();
        assert_eq!(TinySet::as_u64_slice(&actual), expected);
        assert_eq!(empty, expected.iter().all(|word| *word == 0));

        let mut actual = words(&input);
        let empty = exclude_in_place(&mut actual, &right);
        let expected: Vec<_> = input.iter().zip(&other).map(|(a, b)| a & !b).collect();
        assert_eq!(TinySet::as_u64_slice(&actual), expected);
        assert_eq!(empty, expected.iter().all(|word| *word == 0));
        assert_eq!(
            count(&words(&input)),
            input.iter().map(|word| word.count_ones()).sum::<u32>()
        );
        assert_eq!(
            count_intersection(&words(&input), &right),
            input
                .iter()
                .zip(&other)
                .map(|(left, right)| (left & right).count_ones())
                .sum::<u32>()
        );
    }
}
