use super::block_maxscore::BlockMaxScorer;
use crate::postings::SegmentPostings;
use crate::query::phrase_query::PhraseScorer;
use crate::query::{Scorer, TermScorer};
use crate::{DocId, DocSet, Score, TERMINATED};

pub(super) enum MixedScorer {
    Term(TermScorer),
    Phrase(PhraseScorer<SegmentPostings>, Score),
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
                while scorer.doc() < end {
                    callback(scorer.doc(), scorer.score());
                    scorer.advance();
                }
            }
        }
    }
}
