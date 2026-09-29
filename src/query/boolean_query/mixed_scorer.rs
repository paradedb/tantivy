use super::block_maxscore::BlockMaxScorer;
use crate::docset::SeekDangerResult;
use crate::postings::SegmentPostings;
use crate::query::phrase_query::PhraseScorer;
use crate::query::{Scorer, TermScorer};
use crate::{DocId, DocSet, Score, TERMINATED};

pub(super) enum MixedScorer {
    Term(TermScorer),
    Phrase(DeferredPhraseScorer, Score),
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
        let bound = scorer.global_score_bound().unwrap();
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
        if let Self::Term(scorer) = self {
            scorer.seek_block(target);
        }
    }

    #[inline]
    fn block_max_score_up_to(&mut self, target: DocId) -> (Score, DocId) {
        match self {
            Self::Term(scorer) => scorer.block_max_score_up_to(target),
            Self::Phrase(_, bound) => (*bound, TERMINATED),
        }
    }

    #[inline]
    fn block_score_hint(&self) -> Score {
        match self {
            Self::Term(scorer) => scorer.block_score_hint(),
            Self::Phrase(_, bound) => *bound,
        }
    }

    #[inline]
    fn refine_block_max_score(&mut self) -> Score {
        match self {
            Self::Term(scorer) => scorer.refine_block_max_score(),
            Self::Phrase(_, bound) => *bound,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{EnableScoring, QueryParser};
    use crate::schema::{Schema, TEXT};
    use crate::Index;

    #[test]
    fn test_deferred_phrase_checks_and_scoring_windows() -> crate::Result<()> {
        let mut schema = Schema::builder();
        let field = schema.add_text_field("text", TEXT);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        for text in ["a x b", "a b", "x", "a x b", "a b a b", "a x b"] {
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
        for candidate in [0, 1, 2, 3, 4, 5, TERMINATED] {
            let phrase = weight
                .scorer(reader, 1.0)?
                .downcast::<PhraseScorer<SegmentPostings>>()
                .map_err(|_| ())
                .unwrap();
            let mut scorer = MixedScorer::from_phrase(*phrase);
            let reference = expected
                .iter()
                .find(|&&(doc, _)| doc == candidate)
                .map(|&(_, score)| score);
            assert_eq!(scorer.score_at(candidate), reference);
            assert_eq!(scorer.score_at(candidate), reference);
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
        Ok(())
    }
}
