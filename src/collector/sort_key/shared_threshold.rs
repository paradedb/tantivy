use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::collector::sort_key::{Comparator, ComparatorEnum};
use crate::{Order, Score};

pub type SharedThresholdArc<T> = Arc<dyn SharedThreshold<T>>;
pub type SharedThresholdArcOpt<T> = Option<SharedThresholdArc<T>>;

/// A trait for sharing a search threshold across multiple threads or segments.
///
/// Implementations of this trait must be thread-safe as they are typically wrapped in an [`Arc`].
/// The threshold is used to prune documents that cannot possibly compete with the top-K
/// documents already found in other segments.
pub trait SharedThreshold<T>: Send + Sync {
    /// Loads the current shared threshold and its associated segment ordinal.
    ///
    /// Among documents with the same sort key, we favor those from segments with a lower ordinal.
    /// This is consistent with the tie-breaking behavior of [`DocAddress`], which ensures
    /// stable sorting across multiple segments.
    fn load(&self) -> (T, u32);

    /// Conditionally updates the shared threshold if `new_threshold` is more restrictive.
    ///
    /// Returns the most restrictive threshold currently known after the update attempt (which
    /// will be `new_threshold` if the update succeeded, or a pre-existing strictly better
    /// threshold if it failed).
    fn update(&self, new_threshold: T, segment_ord: u32) -> (T, u32);
}

pub struct NoopSharedThreshold<T> {
    noop_value: T,
}

impl<T: Clone + Send + Sync> NoopSharedThreshold<T> {
    pub fn new(noop_value: T) -> Self {
        Self { noop_value }
    }
}

impl<T: Clone + Send + Sync> SharedThreshold<T> for NoopSharedThreshold<T> {
    fn load(&self) -> (T, u32) {
        (self.noop_value.clone(), 0)
    }

    fn update(&self, new_threshold: T, segment_ord: u32) -> (T, u32) {
        (new_threshold, segment_ord)
    }
}

#[inline]
fn f32_to_ordered_u32(f: f32) -> u32 {
    let bits = f.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

#[inline]
fn ordered_u32_to_f32(u: u32) -> f32 {
    let bits = if u & 0x8000_0000 != 0 {
        u ^ 0x8000_0000
    } else {
        !u
    };
    f32::from_bits(bits)
}

/// Packs an f32 Score and a segment ordinal into a u64 for lock-free atomic max operations.
/// We want to maximize the score, but *minimize* the segment ordinal to favor lower segments
/// in tie-breakers. So we pack the inverted segment ordinal into the lower 32 bits.
#[inline(always)]
fn pack_score_and_ord(score: Score, segment_ord: u32) -> u64 {
    let top = f32_to_ordered_u32(score) as u64;
    let bottom = (!segment_ord) as u64;
    (top << 32) | bottom
}

#[inline(always)]
fn unpack_score_and_ord(val: u64) -> (Score, u32) {
    let score = ordered_u32_to_f32((val >> 32) as u32);
    let segment_ord = !(val as u32);
    (score, segment_ord)
}

pub struct AtomicSharedThreshold {
    value: AtomicU64,
}

impl Default for AtomicSharedThreshold {
    fn default() -> Self {
        Self {
            value: AtomicU64::new(pack_score_and_ord(Score::MIN, u32::MAX)),
        }
    }
}

impl SharedThreshold<Score> for AtomicSharedThreshold {
    fn load(&self) -> (Score, u32) {
        unpack_score_and_ord(self.value.load(Ordering::Relaxed))
    }

    fn update(&self, new_threshold: Score, segment_ord: u32) -> (Score, u32) {
        let new_packed = pack_score_and_ord(new_threshold, segment_ord);
        let mut current = self.value.load(Ordering::Relaxed);
        loop {
            if new_packed <= current {
                return unpack_score_and_ord(current);
            }
            match self.value.compare_exchange_weak(
                current,
                new_packed,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return (new_threshold, segment_ord),
                Err(actual) => current = actual,
            }
        }
    }
}

/// A shared threshold for `Option<u64>` values.
///
/// In both `NaturalComparator` and `ReverseNoneIsLowerComparator`, `None` is the worst value
/// (it appears last in top docs). So the initial threshold is `None`.
///
/// Since `AtomicU64` cannot cleanly pack `Option<u64>` without losing a state, and threshold
/// updates are very rare compared to reads, we use a `RwLock<(Option<u64>, u32)>`.
pub struct RwLockSharedThresholdOptionU64 {
    value: RwLock<(Option<u64>, u32)>,
    order: Order,
}

impl RwLockSharedThresholdOptionU64 {
    pub fn new(order: Order) -> Self {
        Self {
            value: RwLock::new((None, u32::MAX)),
            order,
        }
    }

    // helper to compare
    fn is_better(&self, new: &Option<u64>, new_ord: u32, old: &Option<u64>, old_ord: u32) -> bool {
        let cmp_enum = ComparatorEnum::from(self.order);
        match cmp_enum.compare(new, old) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => new_ord < old_ord,
        }
    }
}

impl SharedThreshold<Option<u64>> for RwLockSharedThresholdOptionU64 {
    fn load(&self) -> (Option<u64>, u32) {
        *self.value.read().unwrap()
    }

    fn update(&self, new_threshold: Option<u64>, segment_ord: u32) -> (Option<u64>, u32) {
        let current = *self.value.read().unwrap();
        if self.is_better(&new_threshold, segment_ord, &current.0, current.1) {
            if let Ok(mut write_guard) = self.value.write() {
                if self.is_better(&new_threshold, segment_ord, &write_guard.0, write_guard.1) {
                    *write_guard = (new_threshold, segment_ord);
                    return (new_threshold, segment_ord);
                } else {
                    return *write_guard;
                }
            }
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_ordered_roundtrip() {
        let values = [Score::MIN, -1.0, -0.0, 0.0, 0.5, 1.0, 42.0, Score::MAX];
        for &v in &values {
            let u = f32_to_ordered_u32(v);
            let back = ordered_u32_to_f32(u);
            assert_eq!(v.to_bits(), back.to_bits(), "roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_f32_ordering_preserved() {
        let sorted = [-100.0f32, -1.0, -0.001, 0.0, 0.001, 1.0, 100.0];
        for w in sorted.windows(2) {
            assert!(
                f32_to_ordered_u32(w[0]) < f32_to_ordered_u32(w[1]),
                "{} should map below {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_pack_score_and_ord() {
        let packed1 = pack_score_and_ord(1.0, 5);
        let packed2 = pack_score_and_ord(1.0, 2); // ord 2 is better than ord 5
        let packed3 = pack_score_and_ord(2.0, 10); // score 2.0 is better than score 1.0

        assert!(packed2 > packed1);
        assert!(packed3 > packed2);

        assert_eq!(unpack_score_and_ord(packed1), (1.0, 5));
        assert_eq!(unpack_score_and_ord(packed2), (1.0, 2));
        assert_eq!(unpack_score_and_ord(packed3), (2.0, 10));
    }

    #[test]
    fn test_atomic_shared_threshold() {
        let t = AtomicSharedThreshold::default();
        assert_eq!(t.load(), (Score::MIN, u32::MAX));

        t.update(0.5, 5);
        assert_eq!(t.load(), (0.5, 5));

        t.update(0.3, 2);
        assert_eq!(t.load(), (0.5, 5));

        t.update(0.5, 2); // Same score, better ord
        assert_eq!(t.load(), (0.5, 2));

        t.update(0.9, 10);
        assert_eq!(t.load(), (0.9, 10));
    }

    #[test]
    fn test_rwlock_shared_threshold_option_u64_asc() {
        let t = RwLockSharedThresholdOptionU64::new(Order::Asc);
        assert_eq!(t.load(), (None, u32::MAX));

        t.update(Some(100), 5);
        assert_eq!(t.load(), (Some(100), 5));

        t.update(Some(200), 2); // 200 > 100, worse
        assert_eq!(t.load(), (Some(100), 5));

        t.update(Some(100), 2); // Same score, better ord
        assert_eq!(t.load(), (Some(100), 2));

        t.update(None, 1); // None is strictly smaller than Some(100) -> Wait, None is worse.
        assert_eq!(t.load(), (Some(100), 2));

        t.update(Some(0), 10); // Some(0) > None, better
        assert_eq!(t.load(), (Some(0), 10));
    }

    #[test]
    fn test_rwlock_shared_threshold_option_u64_desc() {
        let t = RwLockSharedThresholdOptionU64::new(Order::Desc);
        assert_eq!(t.load(), (None, u32::MAX));

        t.update(Some(100), 5);
        assert_eq!(t.load(), (Some(100), 5));

        t.update(Some(50), 2); // 50 < 100, worse
        assert_eq!(t.load(), (Some(100), 5));

        t.update(Some(100), 2); // Same score, better ord
        assert_eq!(t.load(), (Some(100), 2));

        t.update(Some(200), 10); // 200 > 100
        assert_eq!(t.load(), (Some(200), 10));

        t.update(None, 1); // None < Some(200), worse
        assert_eq!(t.load(), (Some(200), 10));
    }
}
