use std::sync::Arc;

use crate::index::InvertedIndexReader;
use crate::postings::TermInfo;
use crate::query::Bm25StatisticsProvider;
use crate::schema::Field;
use crate::{Searcher, Term};

#[cfg(all(test, feature = "quickwit"))]
mod tests;

pub(crate) struct ResolvedTermInfo {
    pub doc_freq: u64,
    segments: Vec<(Arc<InvertedIndexReader>, Option<TermInfo>)>,
}

impl ResolvedTermInfo {
    pub fn get(&self, reader: &Arc<InvertedIndexReader>) -> Option<Option<&TermInfo>> {
        self.segments
            .iter()
            .find(|(source, _)| Arc::ptr_eq(source, reader))
            .map(|(_, info)| info.as_ref())
    }
}

pub(crate) struct ResolvedTerms {
    terms: Vec<(Term, Arc<ResolvedTermInfo>)>,
}

impl ResolvedTerms {
    pub fn new<'a>(
        searcher: &Searcher,
        terms: impl IntoIterator<Item = &'a Term>,
    ) -> crate::Result<Self> {
        let mut terms: Vec<_> = terms.into_iter().cloned().collect();
        terms.sort_unstable();
        terms.dedup();
        let mut infos: Vec<_> = terms
            .iter()
            .map(|_| ResolvedTermInfo {
                doc_freq: 0,
                segments: Vec::with_capacity(searcher.segment_readers().len()),
            })
            .collect();
        let mut start = 0;
        while start < terms.len() {
            let field = terms[start].field();
            let end = start + terms[start..].partition_point(|term| term.field() == field);
            for segment in searcher.segment_readers() {
                let reader = segment.inverted_index(field)?;
                let mut segment_infos = vec![None; end - start];
                if end - start == 1 {
                    segment_infos[0] = reader.get_term_info(&terms[start])?;
                } else {
                    #[cfg(feature = "quickwit")]
                    {
                        let keys: Vec<_> = terms[start..end]
                            .iter()
                            .map(Term::serialized_value_bytes)
                            .collect();
                        let keys = crate::termdict::SortedTermSlice::new_assume_sorted(&keys);
                        for entry in reader.terms().batch_term_info_exact(keys) {
                            let (index, info) = entry?;
                            segment_infos[index] = Some(info);
                        }
                    }
                    #[cfg(not(feature = "quickwit"))]
                    for (term, info) in terms[start..end].iter().zip(&mut segment_infos) {
                        *info = reader.get_term_info(term)?;
                    }
                }
                for (resolved, info) in infos[start..end].iter_mut().zip(segment_infos) {
                    resolved.doc_freq += info.as_ref().map_or(0, |info| u64::from(info.doc_freq));
                    resolved.segments.push((Arc::clone(&reader), info));
                }
            }
            start = end;
        }
        Ok(Self {
            terms: terms
                .into_iter()
                .zip(infos.into_iter().map(Arc::new))
                .collect(),
        })
    }

    pub fn get(&self, term: &Term) -> Option<&Arc<ResolvedTermInfo>> {
        self.terms
            .binary_search_by(|(key, _)| key.cmp(term))
            .ok()
            .map(|index| &self.terms[index].1)
    }
}

pub(crate) struct ResolvedStatistics<'a> {
    pub terms: &'a ResolvedTerms,
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
        match self.terms.get(term) {
            Some(info) => Ok(info.doc_freq),
            None => self.provider.doc_freq(term),
        }
    }

    fn bm25_params(&self, field: Field) -> crate::index::Bm25Params {
        self.provider.bm25_params(field)
    }
}
