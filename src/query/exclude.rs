use crate::docset::{DocSet, SeekDangerResult, TERMINATED};
use crate::query::Scorer;
use crate::{DocId, Score};

/// An exclusion set is a set of documents
/// that should be excluded from a given DocSet.
///
/// It can be a single DocSet, or a Vec of DocSets.
pub trait ExclusionSet: Send {
    /// Returns `true` if the given `doc` is in the exclusion set.
    fn contains(&mut self, doc: DocId) -> bool;
}

impl<TDocSet: DocSet> ExclusionSet for TDocSet {
    #[inline]
    fn contains(&mut self, doc: DocId) -> bool {
        self.seek_danger(doc) == SeekDangerResult::Found
    }
}

impl<TDocSet: DocSet> ExclusionSet for Vec<TDocSet> {
    #[inline]
    fn contains(&mut self, doc: DocId) -> bool {
        for docset in self.iter_mut() {
            if docset.seek_danger(doc) == SeekDangerResult::Found {
                return true;
            }
        }
        false
    }
}

/// Filters a given `DocSet` by removing the docs from an exclusion set.
///
/// The excluding docsets have no impact on scoring.
pub struct Exclude<TDocSet, TExclusionSet> {
    underlying_docset: TDocSet,
    exclusion_set: TExclusionSet,
}

impl<TDocSet, TExclusionSet> Exclude<TDocSet, TExclusionSet>
where
    TDocSet: DocSet,
    TExclusionSet: ExclusionSet,
{
    /// Creates a new `ExcludeScorer`
    pub fn new(
        mut underlying_docset: TDocSet,
        mut exclusion_set: TExclusionSet,
    ) -> Exclude<TDocSet, TExclusionSet> {
        while underlying_docset.doc() != TERMINATED {
            let target = underlying_docset.doc();
            if !exclusion_set.contains(target) {
                break;
            }
            underlying_docset.advance();
        }
        Exclude {
            underlying_docset,
            exclusion_set,
        }
    }
}

impl<TDocSet, TExclusionSet> DocSet for Exclude<TDocSet, TExclusionSet>
where
    TDocSet: DocSet,
    TExclusionSet: ExclusionSet,
{
    fn advance(&mut self) -> DocId {
        loop {
            let candidate = self.underlying_docset.advance();
            if candidate == TERMINATED {
                return TERMINATED;
            }
            if !self.exclusion_set.contains(candidate) {
                return candidate;
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let candidate = self.underlying_docset.seek(target);
        if candidate == TERMINATED {
            return TERMINATED;
        }
        if !self.exclusion_set.contains(candidate) {
            return candidate;
        }
        self.advance()
    }

    /// One lookup in the exclusion set settles the target. `seek` would walk the underlying
    /// docset through every excluded doc, and a filter under block-WAND is probed once per
    /// candidate.
    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        if target >= TERMINATED {
            debug_assert!(target == TERMINATED);
            return SeekDangerResult::SeekLowerBound(target);
        }
        match self.underlying_docset.seek_danger(target) {
            SeekDangerResult::Found if self.exclusion_set.contains(target) => {
                // The next doc we can accept is strictly after the excluded target.
                SeekDangerResult::SeekLowerBound(target + 1)
            }
            // Excluding docs never adds any, so the underlying lower bound holds for us too.
            result => result,
        }
    }

    /// With no underlying docs in the range there is nothing to exclude. With some, the answer
    /// needs the walk this type avoids, so it stays unknown.
    fn is_empty_in_range(&mut self, start: DocId, end: DocId) -> bool {
        self.underlying_docset.is_empty_in_range(start, end)
    }

    fn doc(&self) -> DocId {
        self.underlying_docset.doc()
    }

    /// `.size_hint()` directly returns the size
    /// of the underlying docset without taking in account
    /// the fact that docs might be deleted.
    fn size_hint(&self) -> u32 {
        self.underlying_docset.size_hint()
    }
}

impl<TScorer, TExclusionSet> Scorer for Exclude<TScorer, TExclusionSet>
where
    TScorer: Scorer,
    TExclusionSet: ExclusionSet + 'static,
{
    #[inline]
    fn score(&mut self) -> Score {
        self.underlying_docset.score()
    }

    #[inline]
    fn constant_score(&self) -> Option<Score> {
        self.underlying_docset.constant_score()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::postings::tests::test_skip_against_unoptimized;
    use crate::query::VecDocSet;
    use crate::tests::sample_with_seed;

    #[test]
    fn test_exclude() {
        let mut exclude_scorer = Exclude::new(
            VecDocSet::from(vec![1, 2, 5, 8, 10, 15, 24]),
            VecDocSet::from(vec![1, 2, 3, 10, 16, 24]),
        );
        let mut els = vec![];
        while exclude_scorer.doc() != TERMINATED {
            els.push(exclude_scorer.doc());
            exclude_scorer.advance();
        }
        assert_eq!(els, vec![5, 8, 15]);
    }

    #[test]
    fn test_exclude_skip() {
        test_skip_against_unoptimized(
            || {
                Box::new(Exclude::new(
                    VecDocSet::from(vec![1, 2, 5, 8, 10, 15, 24]),
                    VecDocSet::from(vec![1, 2, 3, 10, 16, 24]),
                ))
            },
            vec![5, 8, 10, 15, 24],
        );
    }

    #[test]
    fn test_exclude_skip_random() {
        let sample_include = sample_with_seed(10_000, 0.1, 1);
        let sample_exclude = sample_with_seed(10_000, 0.05, 2);
        let sample_skip = sample_with_seed(10_000, 0.005, 3);
        test_skip_against_unoptimized(
            || {
                Box::new(Exclude::new(
                    VecDocSet::from(sample_include.clone()),
                    VecDocSet::from(sample_exclude.clone()),
                ))
            },
            sample_skip,
        );
    }

    /// Counts the `advance` calls on a wrapped docset, to show when a walk happens.
    struct CountingDocSet {
        inner: VecDocSet,
        advances: usize,
    }

    impl DocSet for CountingDocSet {
        fn advance(&mut self) -> DocId {
            self.advances += 1;
            self.inner.advance()
        }

        fn seek(&mut self, target: DocId) -> DocId {
            self.inner.seek(target)
        }

        fn doc(&self) -> DocId {
            self.inner.doc()
        }

        fn size_hint(&self) -> u32 {
            self.inner.size_hint()
        }
    }

    fn exclude_with_counts(
        include: Vec<DocId>,
        exclude: Vec<DocId>,
    ) -> Exclude<CountingDocSet, VecDocSet> {
        Exclude::new(
            CountingDocSet {
                inner: VecDocSet::from(include),
                advances: 0,
            },
            VecDocSet::from(exclude),
        )
    }

    #[test]
    fn seek_danger_found_leaves_valid_state() {
        let mut exclude =
            exclude_with_counts(vec![1, 2, 5, 8, 10, 15, 24], vec![1, 2, 3, 10, 16, 24]);
        assert_eq!(exclude.seek_danger(5), SeekDangerResult::Found);
        assert_eq!(exclude.doc(), 5);
        assert_eq!(exclude.advance(), 8);
    }

    #[test]
    fn seek_danger_on_an_excluded_doc_does_not_walk() {
        // Docs 10 to 19 are all excluded, so a seek to 10 would step through every one of them.
        let include: Vec<DocId> = (0..30).collect();
        let exclude: Vec<DocId> = (10..20).collect();
        let mut exclude = exclude_with_counts(include, exclude);
        let advances_before = exclude.underlying_docset.advances;

        assert_eq!(
            exclude.seek_danger(10),
            SeekDangerResult::SeekLowerBound(11)
        );
        assert_eq!(exclude.underlying_docset.advances, advances_before);

        // Later probes recover. The first kept doc is found in a valid state.
        assert_eq!(
            exclude.seek_danger(19),
            SeekDangerResult::SeekLowerBound(20)
        );
        assert_eq!(exclude.seek_danger(20), SeekDangerResult::Found);
        assert_eq!(exclude.doc(), 20);
        assert_eq!(exclude.advance(), 21);
    }

    #[test]
    fn seek_danger_passes_the_underlying_lower_bound_through() {
        let mut exclude =
            exclude_with_counts(vec![1, 2, 5, 8, 10, 15, 24], vec![1, 2, 3, 10, 16, 24]);
        // 6 is not in the underlying docset, whose own lower bound is its next doc, 8.
        assert_eq!(exclude.seek_danger(6), SeekDangerResult::SeekLowerBound(8));
        assert_eq!(
            exclude.seek_danger(TERMINATED),
            SeekDangerResult::SeekLowerBound(TERMINATED)
        );
    }

    #[test]
    fn seek_danger_matches_seek() {
        let include = vec![1, 2, 5, 8, 10, 15, 24];
        let exclude = vec![1, 2, 3, 10, 16, 24];
        for target in 0..30 {
            let mut fresh = Exclude::new(
                VecDocSet::from(include.clone()),
                VecDocSet::from(exclude.clone()),
            );
            // `seek` must not move backwards, and the leading excluded docs are already skipped.
            if target < fresh.doc() {
                continue;
            }
            let next = fresh.seek(target);
            let mut probed = Exclude::new(
                VecDocSet::from(include.clone()),
                VecDocSet::from(exclude.clone()),
            );
            match probed.seek_danger(target) {
                SeekDangerResult::Found => assert_eq!(next, target, "target {target}"),
                SeekDangerResult::SeekLowerBound(bound) => {
                    assert_ne!(next, target, "target {target}");
                    assert!(
                        bound > target && bound <= next,
                        "target {target}: bound {bound}, next {next}"
                    );
                }
            }
        }
    }

    #[test]
    fn is_empty_in_range_follows_the_underlying_docset() {
        // A fresh docset per probe, since `seek_danger` targets must increase within one docset.
        let fresh = || exclude_with_counts(vec![0, 5, 8], vec![5]);
        assert!(fresh().is_empty_in_range(6, 7));
        assert!(fresh().is_empty_in_range(9, 20));
        // 5 is excluded, but telling needs the walk, so the range is not known to be empty.
        assert!(!fresh().is_empty_in_range(5, 5));
        assert!(!fresh().is_empty_in_range(0, 8));
    }
}
