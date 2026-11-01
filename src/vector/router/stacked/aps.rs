//! When a ranked list walk should stop opening further lists.
//!
//! [`Aps`] is that decision. The walk only calls [`Aps::observe`]. Which
//! estimate produces the decision — today a patience window, later a
//! replacement — stays in this module.

use crate::vector::distance::Similarity;

/// Consecutive lists that leave a full size-`k` heap unchanged before the
/// walk stops. Five is the value measured on cohere-1m and cohere-10m.
/// Each list is scanned whole and is already larger than a `k`-sized row
/// budget, so the window is in lists, not rows, and does not grow with `k`
/// or `nlist`.
const PATIENCE_LISTS: usize = 5;

/// Decides whether the list just scanned is the last one to open.
pub(crate) trait Aps {
    /// `kth` is the worst similarity in a full size-`k` heap, `None` until the
    /// heap fills. `true` stops the walk. One call is one list.
    fn observe(&mut self, kth: Option<Similarity>) -> bool;
}

/// Builds the stop for one level. The patience window is [`PATIENCE_LISTS`]
/// lists without a heap improvement.
pub(crate) fn open() -> Box<dyn Aps> {
    Box::new(Patience::new(PATIENCE_LISTS))
}

/// Stops after `window` lists in a row leave the full heap's worst similarity
/// unchanged. One `observe` call is one list.
struct Patience {
    window: usize,
    /// Lists scored since the last kth change (while the heap is full).
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
    fn observe(&mut self, kth: Option<Similarity>) -> bool {
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
                self.stagnant = self.stagnant.saturating_add(1);
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
    fn patience_stops_after_five_quiet_lists() {
        let mut stop = Patience::new(PATIENCE_LISTS);
        let a = sim(1);
        let b = sim(2);
        assert!(!stop.observe(None));
        assert!(!stop.observe(Some(a))); // first full: stagnant = 0
        for _ in 0..4 {
            assert!(!stop.observe(Some(a)));
        }
        assert!(stop.observe(Some(a))); // 5 quiet lists
        let mut stop = Patience::new(PATIENCE_LISTS);
        assert!(!stop.observe(Some(a))); // first full
        assert!(!stop.observe(Some(b))); // improved: stagnant = 0
        for _ in 0..4 {
            assert!(!stop.observe(Some(b)));
        }
        assert!(stop.observe(Some(b)));
    }
}
