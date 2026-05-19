use std::sync::atomic::{AtomicU32, Ordering};

use crate::Score;

pub trait SharedThreshold: Send + Sync {
    fn load(&self) -> Score;
    fn update(&self, new_threshold: Score);
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

impl SharedThreshold for AtomicSharedThreshold {
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
}
