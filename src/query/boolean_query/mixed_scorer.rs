use super::block_maxscore::BlockMaxScorer;
use crate::docset::SeekDangerResult;
use crate::postings::SegmentPostings;
use crate::query::phrase_query::PhraseScorer;
use crate::query::scorer::PruningScorer;
use crate::query::{Scorer, TermScorer};
use crate::{DocId, DocSet, Score, TERMINATED};

pub(super) enum MixedScorer {
    Term(TermScorer),
    Phrase(DeferredPhraseScorer, Box<TermScorer>),
}

pub(super) struct DeferredPhraseScorer {
    scorer: PhraseScorer<SegmentPostings>,
    doc: DocId,
    verified: bool,
    size_hint: u32,
    cost: u64,
}

impl MixedScorer {
    pub(super) fn from_phrase(scorer: PhraseScorer<SegmentPostings>) -> Self {
        assert!(scorer.global_score_bound().is_some());
        let bound = Box::new(scorer.block_bound_scorer());
        let phrase = DeferredPhraseScorer {
            doc: scorer.doc(),
            verified: true,
            size_hint: scorer.size_hint(),
            cost: scorer.cost(),
            scorer,
        };
        Self::Phrase(phrase, bound)
    }
}

impl DeferredPhraseScorer {
    fn matches(&mut self, target: DocId) -> bool {
        if target < self.doc {
            return false;
        }
        if target == TERMINATED {
            self.doc = TERMINATED;
            self.verified = false;
            return false;
        }
        if self.verified && target == self.doc {
            return true;
        }
        match self.scorer.seek_danger(target) {
            SeekDangerResult::Found => {
                self.doc = target;
                self.verified = true;
                true
            }
            SeekDangerResult::SeekLowerBound(next) => {
                self.doc = next;
                self.verified = false;
                false
            }
        }
    }
}

impl DocSet for DeferredPhraseScorer {
    fn advance(&mut self) -> DocId {
        if self.doc == TERMINATED {
            return TERMINATED;
        }
        if !self.verified {
            return self.seek(self.doc);
        }
        self.doc = self.scorer.advance();
        self.doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let mut target = target.max(self.doc);
        while target < TERMINATED {
            if self.matches(target) {
                return self.doc;
            }
            target = self.doc;
        }
        self.doc = TERMINATED;
        self.doc
    }

    fn doc(&self) -> DocId {
        self.doc
    }

    fn size_hint(&self) -> u32 {
        self.size_hint
    }

    fn cost(&self) -> u64 {
        self.cost
    }
}

impl Scorer for DeferredPhraseScorer {
    fn score(&mut self) -> Score {
        debug_assert!(self.verified && self.doc != TERMINATED);
        self.scorer.score()
    }
}

impl DocSet for MixedScorer {
    #[inline]
    fn advance(&mut self) -> DocId {
        match self {
            Self::Term(scorer) => scorer.advance(),
            Self::Phrase(scorer, _) => scorer.advance(),
        }
    }

    #[inline]
    fn seek(&mut self, target: DocId) -> DocId {
        match self {
            Self::Term(scorer) => scorer.seek(target),
            Self::Phrase(scorer, _) => scorer.seek(target),
        }
    }

    #[inline]
    fn doc(&self) -> DocId {
        match self {
            Self::Term(scorer) => scorer.doc(),
            Self::Phrase(scorer, _) => scorer.doc(),
        }
    }

    fn size_hint(&self) -> u32 {
        match self {
            Self::Term(scorer) => scorer.size_hint(),
            Self::Phrase(scorer, _) => scorer.size_hint(),
        }
    }

    fn cost(&self) -> u64 {
        match self {
            Self::Term(scorer) => scorer.cost(),
            Self::Phrase(scorer, _) => scorer.cost(),
        }
    }
}

impl Scorer for MixedScorer {
    #[inline]
    fn score(&mut self) -> Score {
        match self {
            Self::Term(scorer) => scorer.score(),
            Self::Phrase(scorer, _) => scorer.score(),
        }
    }
}

impl BlockMaxScorer for MixedScorer {
    #[inline]
    fn seek_block(&mut self, target: DocId) {
        match self {
            Self::Term(scorer) => scorer.seek_block(target),
            Self::Phrase(_, bound) => bound.seek_block(target),
        }
    }

    #[inline]
    fn block_max_score_up_to(&mut self, target: DocId) -> (Score, DocId) {
        match self {
            Self::Term(scorer) => scorer.block_max_score_up_to(target),
            Self::Phrase(_, bound) => bound.block_max_score_up_to(target),
        }
    }

    #[inline]
    fn block_score_hint(&self) -> Score {
        match self {
            Self::Term(scorer) => scorer.block_score_hint(),
            Self::Phrase(_, bound) => bound.block_score_hint(),
        }
    }

    #[inline]
    fn refine_block_max_score(&mut self) -> Score {
        match self {
            Self::Term(scorer) => scorer.refine_block_max_score(),
            Self::Phrase(_, bound) => bound.refine_block_max_score(),
        }
    }

    #[inline]
    fn for_each_score_until(&mut self, end: DocId, mut callback: impl FnMut(DocId, Score)) {
        match self {
            Self::Term(scorer) => scorer.for_each_score_until(end, callback),
            Self::Phrase(scorer, _) => {
                // A deferred phrase check can leave the current document unverified.
                scorer.seek(scorer.doc());
                while scorer.doc() < end {
                    callback(scorer.doc(), scorer.score());
                    scorer.advance();
                }
            }
        }
    }

    #[inline]
    fn score_at(&mut self, doc: DocId) -> Option<Score> {
        match self {
            Self::Term(scorer) => scorer.score_at(doc),
            Self::Phrase(scorer, _) => scorer.matches(doc).then(|| scorer.score()),
        }
    }
}

pub(super) struct TermPhraseIntersectionScorer {
    term: TermScorer,
    phrase: PhraseScorer<SegmentPostings>,
    phrase_bound: Score,
    threshold: Score,
    next_doc: DocId,
    current: (DocId, Score),
    size_hint: u32,
}

impl TermPhraseIntersectionScorer {
    pub(super) fn new(
        term: TermScorer,
        phrase: PhraseScorer<SegmentPostings>,
        threshold: Score,
    ) -> Self {
        let next_doc = term.doc().max(phrase.doc());
        let phrase_bound = phrase.global_score_bound().unwrap();
        let size_hint = term.size_hint().min(phrase.size_hint());
        let mut scorer = Self {
            term,
            phrase,
            phrase_bound,
            threshold,
            next_doc,
            current: (TERMINATED, Score::MIN),
            size_hint,
        };
        scorer.advance();
        scorer
    }
}

impl DocSet for TermPhraseIntersectionScorer {
    fn advance(&mut self) -> DocId {
        while self.next_doc < TERMINATED {
            let doc = self.term.seek(self.next_doc);
            if doc == TERMINATED {
                break;
            }
            self.next_doc = doc + 1;
            let term_score = self.term.score();
            if term_score + self.phrase_bound <= self.threshold {
                continue;
            }
            self.phrase
                .set_threshold((self.threshold - term_score).next_down());
            match self.phrase.seek_danger(doc) {
                SeekDangerResult::Found => {
                    let score = term_score + self.phrase.score();
                    if score > self.threshold {
                        self.current = (doc, score);
                        return doc;
                    }
                }
                SeekDangerResult::SeekLowerBound(next) => self.next_doc = next,
            }
        }
        self.next_doc = TERMINATED;
        self.current = (TERMINATED, Score::MIN);
        TERMINATED
    }

    fn doc(&self) -> DocId {
        self.current.0
    }

    fn size_hint(&self) -> u32 {
        self.size_hint
    }
}

impl Scorer for TermPhraseIntersectionScorer {
    fn score(&mut self) -> Score {
        self.current.1
    }
}

impl PruningScorer for TermPhraseIntersectionScorer {
    fn set_threshold(&mut self, threshold: Score) {
        self.threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{EnableScoring, QueryParser};
    use crate::schema::{Schema, TEXT};
    use crate::Index;

    #[test]
    fn test_deferred_phrase_checks_and_scoring_windows() -> crate::Result<()> {
        for pnorms in [false, true] {
            let mut schema = Schema::builder();
            let options = TEXT.set_indexing_options(
                TEXT.get_indexing_options()
                    .unwrap()
                    .clone()
                    .set_pnorms(pnorms),
            );
            let field = schema.add_text_field("text", options);
            let index = Index::create_in_ram(schema.build());
            let mut writer = index.writer_for_tests()?;
            for text in ["a x b", "a b", "x", "a x b", "a b a b", "a x b"] {
                writer.add_document(doc!(field => text))?;
            }
            for ordinal in 0..20_000 {
                let text = if ordinal % 7 == 0 {
                    "a b ".repeat(1 + ordinal % 11)
                } else if ordinal % 5 == 0 {
                    "a x b ".repeat(1 + ordinal % 101)
                } else {
                    "x".to_owned()
                };
                writer.add_document(doc!(field => text))?;
            }
            writer.commit()?;
            drop(writer);
            let searcher = index.reader()?.searcher();
            let query = QueryParser::for_index(&index, vec![field]).parse_query("\"a b\"")?;
            let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
            let reader = searcher.segment_reader(0);
            let mut expected = Vec::new();
            let mut ordinary = weight.scorer(reader, 1.0)?;
            while ordinary.doc() != TERMINATED {
                expected.push((ordinary.doc(), ordinary.score()));
                ordinary.advance();
            }
            let mut before_first = weight.scorer(reader, 1.0)?;
            assert_eq!(
                before_first.seek_danger(0),
                SeekDangerResult::SeekLowerBound(1)
            );
            assert_eq!(before_first.seek_danger(1), SeekDangerResult::Found);
            assert_eq!(before_first.score(), expected[0].1);
            for unknown_average in [false, true] {
                for candidate in [
                    0, 1, 2, 3, 4, 5, 4095, 4096, 8191, 8192, 16383, 19000, TERMINATED,
                ] {
                    let phrase = weight
                        .scorer(reader, 1.0)?
                        .downcast::<PhraseScorer<SegmentPostings>>()
                        .map_err(|_| ())
                        .unwrap();
                    let phrase = if unknown_average {
                        (*phrase).with_indexing_average(Score::NAN)
                    } else {
                        *phrase
                    };
                    let mut scorer = MixedScorer::from_phrase(phrase);
                    let logical_doc = scorer.doc();
                    scorer.seek_block(candidate);
                    let (bound, end) = scorer
                        .block_max_score_up_to(candidate.saturating_add(8191).min(TERMINATED));
                    assert_eq!(scorer.doc(), logical_doc);
                    for &(doc, score) in &expected {
                        if doc >= candidate && doc <= end {
                            assert!(score <= bound, "doc={doc}, score={score}, bound={bound}");
                        }
                    }
                    let refined = scorer.refine_block_max_score();
                    assert!(refined.is_finite());
                    assert_eq!(scorer.doc(), logical_doc);
                    let reference = expected
                        .iter()
                        .find(|&&(doc, _)| doc == candidate)
                        .map(|&(_, score)| score);
                    assert_eq!(scorer.score_at(candidate), reference);
                    assert_eq!(scorer.score_at(candidate), reference);
                    let logical_doc = scorer.doc();
                    scorer.seek_block(candidate.saturating_add(1024).min(TERMINATED));
                    assert_eq!(scorer.doc(), logical_doc);
                    let mut actual = Vec::new();
                    scorer.for_each_score_until(TERMINATED, |doc, score| actual.push((doc, score)));
                    assert_eq!(
                        actual,
                        expected
                            .iter()
                            .copied()
                            .filter(|&(doc, _)| doc >= candidate)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
        Ok(())
    }
}
