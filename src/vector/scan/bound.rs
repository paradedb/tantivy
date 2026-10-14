//! The scan's pruning bound: the k-th largest lower endpoint over quantized intervals and exact
//! scores.

/// Orders `(lower, row, estimate)` entries by lower endpoint descending, then row ascending.
#[inline]
pub(crate) fn lower_endpoint_order(
    a: &(f32, usize, f32),
    b: &(f32, usize, f32),
) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// The scan's one pruning bound: the k-th largest lower endpoint over the live candidates. A
/// full-precision segment scores every row exactly, so its bound is the k-th exact score; a
/// quantized segment bounds its quantized intervals until the exact phase scores the survivors.
/// Only the top `k` can affect it, so each side keeps at most `k` entries.
#[derive(Clone)]
pub(crate) struct KthBound {
    pub(super) k: usize,
    /// Top-k exact scores, descending. Exact rows never move.
    exact: Vec<f32>,
    /// Top-k quantized entries admitted since the last boundary, as `(lower, row, estimate)`,
    /// ordered by [`lower_endpoint_order`].
    pub(super) running: Vec<(f32, usize, f32)>,
    merge: Vec<(f32, usize, f32)>,
}

impl KthBound {
    pub(super) fn new(k: usize) -> Self {
        Self {
            k,
            exact: Vec::with_capacity(k),
            running: Vec::with_capacity(k),
            merge: Vec::with_capacity(k.saturating_mul(2)),
        }
    }

    /// Records an exactly scored row.
    pub(super) fn push_exact(&mut self, score: f32) {
        if self.k == 0
            || (self.exact.len() == self.k
                && score.total_cmp(self.exact.last().unwrap()) != std::cmp::Ordering::Greater)
        {
            return;
        }
        let at = self
            .exact
            .partition_point(|kept| kept.total_cmp(&score) != std::cmp::Ordering::Less);
        self.exact.insert(at, score);
        self.exact.truncate(self.k);
    }

    /// The k-th exact score: no row scoring below it can enter the top k.
    #[inline]
    pub(super) fn exact_kth(&self) -> Option<f32> {
        (self.k > 0 && self.exact.len() == self.k).then(|| self.exact[self.k - 1])
    }

    /// Merges one cluster's top-k entries into the running top-k.
    pub(super) fn merge_running(&mut self, cluster_top: &[(f32, usize, f32)]) {
        self.merge.clear();
        self.merge.extend_from_slice(&self.running);
        self.merge.extend_from_slice(cluster_top);
        self.merge.sort_unstable_by(lower_endpoint_order);
        self.merge.truncate(self.k);
        std::mem::swap(&mut self.running, &mut self.merge);
    }

    /// The bounds gate's threshold: the k-th exact score once rows are scored exactly, otherwise
    /// the k-th running lower endpoint; `None` until `k` are in.
    pub(crate) fn running_lower(&self) -> Option<f32> {
        if !self.exact.is_empty() {
            return self.exact_kth();
        }
        (self.k > 0 && self.running.len() >= self.k).then(|| self.running[self.k - 1].0)
    }

    /// APS cover's threshold: the k-th exact score once rows are scored exactly, otherwise the
    /// smallest running point estimate; `None` until `k` are in. `k` rows score at least this, so
    /// it never exceeds the k-th best estimate; it carries no confidence margin, so it must not
    /// drive pruning.
    pub(crate) fn running_estimate(&self) -> Option<f32> {
        if self.k == 0 {
            return None;
        }
        if !self.exact.is_empty() {
            return self.exact_kth();
        }
        if self.running.len() < self.k {
            return None;
        }
        self.running
            .iter()
            .map(|&(_, _, estimate)| estimate)
            .min_by(f32::total_cmp)
    }

    /// The boundary threshold: the k-th largest of `candidate_lowers`, one per live quantized
    /// candidate, reordered in place. Clears the running top-k, which describes the candidates
    /// before the boundary. Boundaries precede every exact score.
    pub(super) fn boundary_lower(&mut self, candidate_lowers: &mut [f32]) -> Option<f32> {
        debug_assert!(self.exact.is_empty(), "boundaries precede exact scoring");
        self.running.clear();
        let k = self.k;
        if k == 0 || candidate_lowers.len() < k {
            return None;
        }
        let (_, kth, _) = candidate_lowers.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
        Some(*kth)
    }
}
