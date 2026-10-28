//! When a ranked list walk should stop opening further lists.
//!
//! [`Aps`] is that decision. The walk only calls [`Aps::observe`]. Which
//! estimate produces the decision — today a patience window, later a
//! replacement — stays in this module.

use crate::vector::distance::Similarity;

/// Members scored with an unchanged full size-`k` heap, as a multiple of `k`,
/// before patience stops. Counted in members (not lists) so √n list sizing
/// does not make the same constant a different share of the shortlist at
/// 1M vs 10M.
const PATIENCE_K_MULTIPLIER: f32 = 5.0;

/// Decides whether the list just scanned is the last one to open.
pub(crate) trait Aps {
    /// `kth` is the worst similarity in a full size-`k` heap, `None` until the
    /// heap fills. `members` is how many rows the list just scanned contributed.
    /// `true` stops the walk.
    fn observe(&mut self, kth: Option<Similarity>, members: usize) -> bool;
}

/// Builds the stop for one level. The patience window is `⌈PATIENCE_K_MULTIPLIER × k⌉`
/// members without a heap improvement.
pub(crate) fn open(k: usize) -> Box<dyn Aps> {
    let window = ((k as f32 * PATIENCE_K_MULTIPLIER).ceil() as usize).max(1);
    Box::new(Patience::new(window))
}

/// Stops after `window` members in a row leave the full heap's worst similarity
/// unchanged. Lists are still scanned whole; the check runs at list boundaries.
struct Patience {
    window: usize,
    /// Members scored since the last kth change (while the heap is full).
    stagnant: usize,
    last: Option<Similarity>,
}

impl Patience {
    fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            stagnant: 0,
            last: None,
        }
    }
}

impl Aps for Patience {
    fn observe(&mut self, kth: Option<Similarity>, members: usize) -> bool {
        let Some(kth) = kth else {
            // Heap still filling: improvements are structural, not stagnation.
            self.stagnant = 0;
            self.last = None;
            return false;
        };
        match self.last {
            None => {
                // Heap just filled (or first full observation): that list was progress.
                self.last = Some(kth);
                self.stagnant = 0;
            }
            Some(previous) if previous == kth => {
                self.stagnant = self.stagnant.saturating_add(members);
            }
            Some(_) => {
                // Heap improved: restart the window after this list.
                self.last = Some(kth);
                self.stagnant = 0;
            }
        }
        self.stagnant >= self.window
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
    fn patience_window_scales_with_k() {
        // 5 × k members.
        assert_eq!(((8f32 * PATIENCE_K_MULTIPLIER).ceil() as usize), 40);
    }

    #[test]
    fn patience_stops_after_stagnant_members() {
        let mut stop = Patience::new(10);
        let a = sim(1);
        let b = sim(2);
        assert!(!stop.observe(None, 100));
        assert!(!stop.observe(Some(a), 4)); // first full: stagnant = 0
        assert!(!stop.observe(Some(a), 4));
        assert!(!stop.observe(Some(a), 4));
        assert!(stop.observe(Some(a), 2)); // 4+4+2 = 10
        let mut stop = Patience::new(5);
        assert!(!stop.observe(Some(a), 3)); // first full
        assert!(!stop.observe(Some(b), 3)); // improved: stagnant = 0
        assert!(!stop.observe(Some(b), 4));
        assert!(stop.observe(Some(b), 1)); // 4+1 = 5
    }
}
