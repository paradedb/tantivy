use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::{Order, Score};

pub trait SharedThreshold<T>: Send + Sync {
    fn load(&self) -> T;
    fn update(&self, new_threshold: T);
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
    fn load(&self) -> T {
        self.noop_value.clone()
    }

    fn update(&self, _new_threshold: T) {}
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

pub struct AtomicSharedThreshold {
    value: AtomicU32,
}

impl Default for AtomicSharedThreshold {
    fn default() -> Self {
        Self {
            value: AtomicU32::new(f32_to_ordered_u32(Score::MIN)),
        }
    }
}

impl SharedThreshold<Score> for AtomicSharedThreshold {
    fn load(&self) -> Score {
        ordered_u32_to_f32(self.value.load(Ordering::Relaxed))
    }

    fn update(&self, new_threshold: Score) {
        let new_ordered = f32_to_ordered_u32(new_threshold);
        let mut current = self.value.load(Ordering::Relaxed);
        loop {
            if new_ordered <= current {
                break;
            }
            match self.value.compare_exchange_weak(
                current,
                new_ordered,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

pub struct AtomicSharedThresholdU64 {
    value: AtomicU64,
    order: Order,
}

impl AtomicSharedThresholdU64 {
    pub fn new(order: Order) -> Self {
        let init_val = match order {
            Order::Asc => u64::MAX,
            Order::Desc => u64::MIN,
        };
        Self {
            value: AtomicU64::new(init_val),
            order,
        }
    }
}

impl SharedThreshold<u64> for AtomicSharedThresholdU64 {
    fn load(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    fn update(&self, new_threshold: u64) {
        let mut current = self.value.load(Ordering::Relaxed);
        loop {
            let should_update = match self.order {
                Order::Asc => new_threshold < current,
                Order::Desc => new_threshold > current,
            };
            if !should_update {
                break;
            }
            match self.value.compare_exchange_weak(
                current,
                new_threshold,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

/// A shared threshold for `Option<u64>` values.
///
/// In `NaturalComparator`, `None` is strictly smaller than any `Some(_)`.
/// * When sorting Ascending, the worst value is `Some(MAX)`. `None` is the most restrictive threshold.
/// * When sorting Descending, the worst value is `None`. `Some(MAX)` is the most restrictive threshold.
///
/// Since `AtomicU64` cannot cleanly pack `Option<u64>` without losing a state, and threshold updates
/// are very rare compared to reads, we use a `RwLock<Option<u64>>`.
use std::sync::RwLock;

pub struct RwLockSharedThresholdOptionU64 {
    value: RwLock<Option<u64>>,
    order: Order,
}

impl RwLockSharedThresholdOptionU64 {
    pub fn new(order: Order) -> Self {
        // Initial "worst" threshold
        let init_val = match order {
            Order::Asc => Some(u64::MAX),
            // For descending order, `None` is the smallest value (which is worst because we want largest).
            Order::Desc => None,
        };
        Self {
            value: RwLock::new(init_val),
            order,
        }
    }
    
    // helper to compare
    fn is_better(&self, new: &Option<u64>, old: &Option<u64>) -> bool {
        // Natural ordering: None < Some(_)
        match self.order {
            Order::Asc => {
                // We want smaller values, so `new` is better if `new < old`.
                new < old
            }
            Order::Desc => {
                // We want larger values, so `new` is better if `new > old`.
                new > old
            }
        }
    }
}

impl SharedThreshold<Option<u64>> for RwLockSharedThresholdOptionU64 {
    fn load(&self) -> Option<u64> {
        *self.value.read().unwrap()
    }

    fn update(&self, new_threshold: Option<u64>) {
        let current = *self.value.read().unwrap();
        if self.is_better(&new_threshold, &current) {
            if let Ok(mut write_guard) = self.value.write() {
                if self.is_better(&new_threshold, &write_guard) {
                    *write_guard = new_threshold;
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_ordered_roundtrip() {
        let values = [
            Score::MIN,
            -1.0,
            -0.0,
            0.0,
            0.5,
            1.0,
            42.0,
            Score::MAX,
        ];
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
    fn test_atomic_shared_threshold() {
        let t = AtomicSharedThreshold::default();
        assert_eq!(t.load(), Score::MIN);

        t.update(0.5);
        assert_eq!(t.load(), 0.5);

        t.update(0.3);
        assert_eq!(t.load(), 0.5);

        t.update(0.9);
        assert_eq!(t.load(), 0.9);
    }

    #[test]
    fn test_atomic_shared_threshold_u64_asc() {
        let t = AtomicSharedThresholdU64::new(Order::Asc);
        assert_eq!(t.load(), u64::MAX);

        t.update(100);
        assert_eq!(t.load(), 100);

        t.update(200); // 200 > 100, so it's worse in Asc
        assert_eq!(t.load(), 100);

        t.update(50);
        assert_eq!(t.load(), 50);
    }

    #[test]
    fn test_atomic_shared_threshold_u64_desc() {
        let t = AtomicSharedThresholdU64::new(Order::Desc);
        assert_eq!(t.load(), u64::MIN);

        t.update(100);
        assert_eq!(t.load(), 100);

        t.update(50); // 50 < 100, so it's worse in Desc
        assert_eq!(t.load(), 100);

        t.update(200);
        assert_eq!(t.load(), 200);
    }

    #[test]
    fn test_rwlock_shared_threshold_option_u64_asc() {
        let t = RwLockSharedThresholdOptionU64::new(Order::Asc);
        assert_eq!(t.load(), Some(u64::MAX));

        t.update(Some(100));
        assert_eq!(t.load(), Some(100));

        t.update(Some(200)); // 200 > 100, worse
        assert_eq!(t.load(), Some(100));

        t.update(None); // None is strictly smaller than Some(100)
        assert_eq!(t.load(), None);

        t.update(Some(0)); // Some(0) > None, worse
        assert_eq!(t.load(), None);
    }

    #[test]
    fn test_rwlock_shared_threshold_option_u64_desc() {
        let t = RwLockSharedThresholdOptionU64::new(Order::Desc);
        assert_eq!(t.load(), None);

        t.update(Some(100));
        assert_eq!(t.load(), Some(100));

        t.update(Some(50)); // 50 < 100, worse
        assert_eq!(t.load(), Some(100));

        t.update(Some(200)); // 200 > 100
        assert_eq!(t.load(), Some(200));

        t.update(None); // None < Some(200), worse
        assert_eq!(t.load(), Some(200));
    }
}
