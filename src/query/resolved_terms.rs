use std::collections::BTreeMap;
use std::sync::Arc;

use crate::index::SegmentId;
use crate::postings::TermInfo;
use crate::query::{Bm25StatisticsProvider, EnableScoring};
use crate::schema::Field;
use crate::{Searcher, Term};

#[cfg(all(test, feature = "quickwit"))]
mod tests;

pub(crate) struct ResolvedTerms {
    pub doc_freqs: BTreeMap<Term, u64>,
    pub term_infos: BTreeMap<Term, Arc<BTreeMap<SegmentId, Option<TermInfo>>>>,
}

impl ResolvedTerms {
    pub fn for_scoring<'a>(
        enable_scoring: EnableScoring<'_>,
        terms: impl IntoIterator<Item = &'a Term>,
    ) -> crate::Result<Option<Self>> {
        match enable_scoring {
            EnableScoring::Enabled {
                statistics_provider,
                ..
            } => statistics_provider
                .local_searcher()
                .map(|searcher| Self::new(searcher, terms))
                .transpose(),
            EnableScoring::Disabled { .. } => Ok(None),
        }
    }

    pub fn new<'a>(
        searcher: &Searcher,
        terms: impl IntoIterator<Item = &'a Term>,
    ) -> crate::Result<Self> {
        let mut terms: Vec<_> = terms.into_iter().cloned().collect();
        terms.sort_unstable();
        terms.dedup();
        let mut doc_freqs = vec![0; terms.len()];
        let mut infos = vec![BTreeMap::new(); terms.len()];
        let mut start = 0;
        while start < terms.len() {
            let field = terms[start].field();
            let end = start + terms[start..].partition_point(|term| term.field() == field);
            for segment in searcher.segment_readers() {
                let reader = segment.inverted_index(field)?;
                let mut segment_infos = vec![None; end - start];
                for (term, info) in terms[start..end].iter().zip(&mut segment_infos) {
                    *info = reader.get_term_info(term)?;
                }
                for ((doc_freq, segments), info) in doc_freqs[start..end]
                    .iter_mut()
                    .zip(&mut infos[start..end])
                    .zip(segment_infos)
                {
                    *doc_freq += info.as_ref().map_or(0, |info| u64::from(info.doc_freq));
                    segments.insert(segment.segment_id(), info);
                }
            }
            start = end;
        }
        Ok(Self {
            doc_freqs: terms.iter().cloned().zip(doc_freqs).collect(),
            term_infos: terms
                .into_iter()
                .zip(infos.into_iter().map(Arc::new))
                .collect(),
        })
    }
}

pub(crate) struct ResolvedStatistics<'a> {
    pub terms: Option<&'a ResolvedTerms>,
    pub provider: &'a dyn Bm25StatisticsProvider,
}

impl Bm25StatisticsProvider for ResolvedStatistics<'_> {
    fn total_num_tokens(&self, field: Field) -> crate::Result<u64> {
        self.provider.total_num_tokens(field)
    }

    fn total_num_docs(&self) -> crate::Result<u64> {
        self.provider.total_num_docs()
    }

    fn doc_freq(&self, term: &Term) -> crate::Result<u64> {
        match self.terms.and_then(|terms| terms.doc_freqs.get(term)) {
            Some(&doc_freq) => Ok(doc_freq),
            None => self.provider.doc_freq(term),
        }
    }

    fn bm25_params(&self, field: Field) -> crate::index::Bm25Params {
        self.provider.bm25_params(field)
    }
}
