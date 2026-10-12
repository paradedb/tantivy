//! Score-space estimates, interval endpoints and thresholds.

/// A point estimate in score space. Pruning only compares its typed endpoints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::vector) struct Estimate(pub(in crate::vector) f32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::vector) struct LowerEndpoint(pub(in crate::vector) f32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::vector) struct UpperEndpoint(pub(in crate::vector) f32);

/// The k-th largest lower endpoint in score space (larger is better).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::vector) struct Threshold(pub(in crate::vector) LowerEndpoint);

impl Estimate {
    pub(in crate::vector) fn lower(self, sigma: f32, kappa: f32) -> LowerEndpoint {
        debug_assert!(sigma >= 0.0 && kappa >= 0.0);
        LowerEndpoint(self.0 - kappa * sigma)
    }

    pub(in crate::vector) fn upper(self, sigma: f32, kappa: f32) -> UpperEndpoint {
        debug_assert!(sigma >= 0.0 && kappa >= 0.0);
        UpperEndpoint(self.0 + kappa * sigma)
    }
}

impl Threshold {
    pub(in crate::vector) fn admits(self, upper: UpperEndpoint) -> bool {
        upper.0 >= self.0 .0
    }
}
