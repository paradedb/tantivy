//! Cached CPU-feature tiers for runtime-dispatched kernels.

#[cfg(target_arch = "x86_64")]
pub(crate) use x86::{tier, Tier};

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::sync::atomic::{AtomicU8, Ordering};

    /// Feature tiers the dispatched x86 kernels compile for, in preference order.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum Tier {
        /// Baseline x86-64: software popcount and software binary16 conversion.
        Baseline,
        /// Hardware POPCNT.
        Popcnt,
        /// AVX2 vectorization, hardware POPCNT and F16C conversions.
        Avx2,
        /// AVX-512 VPOPCNTDQ on top of the AVX2 tier's features.
        Avx512,
    }

    /// Returns the detected tier, probing CPU features once per process.
    #[inline]
    pub(crate) fn tier() -> Tier {
        static TIER: AtomicU8 = AtomicU8::new(u8::MAX);
        match TIER.load(Ordering::Relaxed) {
            0 => Tier::Baseline,
            1 => Tier::Popcnt,
            2 => Tier::Avx2,
            3 => Tier::Avx512,
            _ => {
                let detected = detect();
                TIER.store(detected as u8, Ordering::Relaxed);
                detected
            }
        }
    }

    fn detect() -> Tier {
        let avx2 = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("f16c");
        if avx2
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
        {
            Tier::Avx512
        } else if avx2 {
            Tier::Avx2
        } else if std::arch::is_x86_feature_detected!("popcnt") {
            Tier::Popcnt
        } else {
            Tier::Baseline
        }
    }
}
