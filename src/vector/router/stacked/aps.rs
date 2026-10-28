//! When a ranked list walk should stop opening further lists.
//!
//! [`Aps`] is that decision. The walk only calls [`Aps::observe`]. Which
//! estimate produces the decision — today a patience window, later a
//! replacement — stays in this module.

use crate::vector::distance::Similarity;

/// Consecutive lists that leave a full size-`k` heap unchanged before the
/// current fallback stops. The shortlist from [`super::levels::StackedIvf::lists_to_open`]
/// is still the hard cap.
const MEMBER_SCAN_PATIENCE: usize = 5;

/// Decides whether the list just scanned is the last one to open.
pub(crate) trait Aps {
    /// `kth` is the worst similarity in a full size-`k` heap, `None` until the heap fills.
    /// `true` stops the walk.
    fn observe(&mut self, kth: Option<Similarity>) -> bool;
}

/// Builds the stop for one level.
pub(crate) fn open() -> Box<dyn Aps> {
    Box::new(Patience::new(MEMBER_SCAN_PATIENCE))
}

/// Stops after `window` lists in a row leave the full heap's worst similarity unchanged.
struct Patience {
    window: usize,
    misses: usize,
    last: Option<Similarity>,
}

impl Patience {
    fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            misses: 0,
            last: None,
        }
    }
}

impl Aps for Patience {
    fn observe(&mut self, kth: Option<Similarity>) -> bool {
        let Some(kth) = kth else {
            return false;
        };
        match self.last {
            None => self.last = Some(kth),
            Some(previous) if previous == kth => {
                self.misses += 1;
                if self.misses >= self.window {
                    return true;
                }
            }
            Some(_) => {
                self.misses = 0;
                self.last = Some(kth);
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::distance::Similarity;

    fn sim(bits: u32) -> Similarity {
        Similarity::new(f32::from_bits(bits))
    }

    #[test]
    fn patience_stops_after_a_run_of_unchanged_kth() {
        let mut stop = Patience::new(3);
        let a = sim(1);
        let b = sim(2);
        assert!(!stop.observe(None));
        assert!(!stop.observe(Some(a)));
        assert!(!stop.observe(Some(a)));
        assert!(!stop.observe(Some(a)));
        assert!(stop.observe(Some(a)));
        let mut stop = Patience::new(2);
        assert!(!stop.observe(Some(a)));
        assert!(!stop.observe(Some(b)));
        assert!(!stop.observe(Some(b)));
        assert!(stop.observe(Some(b)));
    }
}
