use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::candidate::Candidate;
use crate::vector::distance::Similarity;

/// Members scored with an unchanged full size-`k` heap, as a multiple of `k`,
/// before patience stops. Counted in members (not lists) so √n list sizing
/// does not make the same constant a different share of the shortlist at
/// 1M vs 10M.
const PATIENCE_K_MULTIPLIER: f32 = 5.0;

/// Tracks the best members at one router level and stops opening lists when
/// they stop improving. Segment and global routing share the same patience rule.
pub(crate) struct AdaptiveProbe<N: Ord> {
    k: usize,
    best: BinaryHeap<Reverse<Candidate<N>>>,
    patience: Patience,
}

impl<N: Ord> AdaptiveProbe<N> {
    pub(crate) fn new(k: usize) -> Self {
        let k = k.max(1);
        let window = ((k as f32 * PATIENCE_K_MULTIPLIER).ceil() as usize).max(1);
        Self {
            k,
            best: BinaryHeap::new(),
            patience: Patience::new(window),
        }
    }

    pub(crate) fn push(&mut self, candidate: Candidate<N>) {
        if self.best.len() < self.k {
            self.best.push(Reverse(candidate));
        } else if let Some(mut worst) = self.best.peek_mut() {
            if candidate > worst.0 {
                *worst = Reverse(candidate);
            }
        }
    }

    /// Call after scoring a complete list, counting its members even if none entered the heap.
    pub(crate) fn should_stop(&mut self, members: usize) -> bool {
        let kth = self
            .best
            .peek()
            .filter(|_| self.best.len() >= self.k)
            .map(|worst| worst.0.sim);
        self.patience.observe(kth, members)
    }

    pub(crate) fn into_sorted_vec(self) -> Vec<Candidate<N>> {
        self.best
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(candidate)| candidate)
            .collect()
    }
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

    fn sim(bits: u32) -> Similarity {
        Similarity::new(f32::from_bits(bits))
    }

    #[test]
    fn probe_waits_for_full_heap_and_resets_after_improvement() {
        let mut probe = AdaptiveProbe::new(2);
        probe.push(Candidate {
            sim: Similarity::new(10.0),
            node: 0,
        });
        assert!(!probe.should_stop(100));
        probe.push(Candidate {
            sim: Similarity::new(20.0),
            node: 1,
        });
        assert!(!probe.should_stop(1));
        probe.push(Candidate {
            sim: Similarity::new(5.0),
            node: 2,
        });
        assert!(!probe.should_stop(9));
        probe.push(Candidate {
            sim: Similarity::new(30.0),
            node: 3,
        });
        assert!(!probe.should_stop(1));
        assert!(!probe.should_stop(9));
        assert!(probe.should_stop(1));
        assert_eq!(
            probe
                .into_sorted_vec()
                .iter()
                .map(|c| c.node)
                .collect::<Vec<_>>(),
            vec![3, 1]
        );
    }

    #[test]
    fn probe_preserves_candidate_order_and_ties() {
        let candidates: Vec<_> = (0..100)
            .map(|node| Candidate {
                sim: Similarity::new((node % 7) as f32),
                node,
            })
            .collect();
        let mut expected = candidates.clone();
        expected.sort_unstable_by(|a, b| b.cmp(a));
        expected.truncate(12);
        let mut probe = AdaptiveProbe::new(12);
        for candidate in candidates {
            probe.push(candidate);
        }
        assert_eq!(probe.into_sorted_vec(), expected);
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
