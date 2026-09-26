use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::fieldnorm::FieldNormReader;
use crate::index::Bm25Params;
use crate::postings::ResolvedTermInfo;
use crate::query::{EnableScoring, Explanation};
use crate::schema::Field;
use crate::{Score, Searcher, Term};

/// Provides the corpus-level statistics needed by BM25 scoring.
///
/// The standard implementation is [`Searcher`], but you can implement this
/// trait on your own type to supply custom statistics (e.g. cluster-wide
/// counts in a distributed setting).
pub trait Bm25StatisticsProvider {
    /// Returns the total number of tokens indexed for `field` across all
    /// segments.
    fn total_num_tokens(&self, field: Field) -> crate::Result<u64>;

    /// Returns the total number of documents in the index.
    fn total_num_docs(&self) -> crate::Result<u64>;

    /// Returns the number of documents containing `term`.
    fn doc_freq(&self, term: &Term) -> crate::Result<u64>;

    /// Returns the BM25 parameters (`k1`, `b`) for `field`.
    ///
    /// Defaults to [`Bm25Params::DEFAULT`] (`k1 = 1.2`, `b = 0.75`).
    fn bm25_params(&self, _field: Field) -> Bm25Params {
        Bm25Params::default()
    }
}

impl Bm25StatisticsProvider for Searcher {
    fn total_num_tokens(&self, field: Field) -> crate::Result<u64> {
        let mut total_num_tokens = 0u64;

        for segment_reader in self.segment_readers() {
            let inverted_index = segment_reader.inverted_index(field)?;
            total_num_tokens += inverted_index.total_num_tokens();
        }
        Ok(total_num_tokens)
    }

    fn total_num_docs(&self) -> crate::Result<u64> {
        let mut total_num_docs = 0u64;

        for segment_reader in self.segment_readers() {
            total_num_docs += u64::from(segment_reader.max_doc());
        }
        Ok(total_num_docs)
    }

    fn doc_freq(&self, term: &Term) -> crate::Result<u64> {
        self.doc_freq(term)
    }

    fn bm25_params(&self, field: Field) -> Bm25Params {
        self.schema()
            .get_field_entry(field)
            .field_type()
            .bm25_params()
            .unwrap_or_default()
    }
}

pub(crate) struct ResolvedTerms {
    pub term_infos: FxHashMap<Term, ResolvedTermInfo>,
}

impl ResolvedTerms {
    pub fn for_scoring<'a>(
        enable_scoring: EnableScoring<'_>,
        terms: impl IntoIterator<Item = &'a Term>,
    ) -> crate::Result<Option<Self>> {
        match enable_scoring {
            EnableScoring::Enabled {
                searcher,
                use_local_statistics: true,
                ..
            } if cfg!(feature = "quickwit") => Self::new(searcher, terms).map(Some),
            _ => Ok(None),
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
        let mut infos: Vec<_> = (0..terms.len())
            .map(|_| {
                FxHashMap::with_capacity_and_hasher(
                    searcher.segment_readers().len(),
                    Default::default(),
                )
            })
            .collect();
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
            term_infos: terms
                .into_iter()
                .zip(
                    doc_freqs
                        .into_iter()
                        .zip(infos)
                        .map(|(doc_freq, segments)| ResolvedTermInfo {
                            doc_freq,
                            segments: Some(Arc::new(segments)),
                        }),
                )
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
        match self.terms.and_then(|terms| terms.term_infos.get(term)) {
            Some(info) => Ok(info.doc_freq),
            None => self.provider.doc_freq(term),
        }
    }

    fn bm25_params(&self, field: Field) -> crate::index::Bm25Params {
        self.provider.bm25_params(field)
    }
}

pub(crate) fn idf(doc_freq: u64, doc_count: u64) -> Score {
    assert!(doc_count >= doc_freq, "{doc_count} >= {doc_freq}");
    let x = ((doc_count - doc_freq) as Score + 0.5) / (doc_freq as Score + 0.5);
    (1.0 + x).ln()
}

fn cached_tf_component(fieldnorm: u32, average_fieldnorm: Score, k1: Score, b: Score) -> Score {
    k1 * (1.0 - b + b * fieldnorm as Score / average_fieldnorm)
}

fn compute_tf_cache(average_fieldnorm: Score, k1: Score, b: Score) -> Arc<[Score; 256]> {
    let mut cache: [Score; 256] = [0.0; 256];
    for (fieldnorm_id, cache_mut) in cache.iter_mut().enumerate() {
        let fieldnorm = FieldNormReader::id_to_fieldnorm(fieldnorm_id as u8);
        *cache_mut = cached_tf_component(fieldnorm, average_fieldnorm, k1, b);
    }
    Arc::new(cache)
}

#[derive(Clone)]
pub struct Bm25Weight {
    idf_explain: Option<Explanation>,
    weight: Score,
    cache: Arc<[Score; 256]>,
    average_fieldnorm: Score,
    params: Bm25Params,
}

impl Bm25Weight {
    pub fn boost_by(&self, boost: Score) -> Bm25Weight {
        if boost == 1.0f32 {
            return self.clone();
        }
        Bm25Weight {
            idf_explain: self.idf_explain.clone(),
            weight: self.weight * boost,
            cache: self.cache.clone(),
            average_fieldnorm: self.average_fieldnorm,
            params: self.params,
        }
    }

    pub fn for_terms(
        statistics: &dyn Bm25StatisticsProvider,
        terms: &[Term],
    ) -> crate::Result<Bm25Weight> {
        assert!(!terms.is_empty(), "Bm25 requires at least one term");
        let field = terms[0].field();
        for term in &terms[1..] {
            assert_eq!(
                term.field(),
                field,
                "All terms must belong to the same field."
            );
        }

        let total_num_tokens = statistics.total_num_tokens(field)?;
        let total_num_docs = statistics.total_num_docs()?;
        let average_fieldnorm = total_num_tokens as Score / total_num_docs as Score;
        let params = statistics.bm25_params(field);

        if terms.len() == 1 {
            let term_doc_freq = statistics.doc_freq(&terms[0])?;
            Ok(Bm25Weight::for_one_term(
                term_doc_freq,
                total_num_docs,
                average_fieldnorm,
                params,
            ))
        } else {
            let mut idf_sum: Score = 0.0;
            for term in terms {
                let term_doc_freq = statistics.doc_freq(term)?;
                idf_sum += idf(term_doc_freq, total_num_docs);
            }
            let idf_explain = Explanation::new("idf", idf_sum);
            Ok(Bm25Weight::new(idf_explain, average_fieldnorm, params))
        }
    }

    pub fn for_one_term(
        term_doc_freq: u64,
        total_num_docs: u64,
        avg_fieldnorm: Score,
        params: Bm25Params,
    ) -> Bm25Weight {
        let idf = idf(term_doc_freq, total_num_docs);
        let mut idf_explain =
            Explanation::new("idf, computed as log(1 + (N - n + 0.5) / (n + 0.5))", idf);
        idf_explain.add_const(
            "n, number of docs containing this term",
            term_doc_freq as Score,
        );
        idf_explain.add_const("N, total number of docs", total_num_docs as Score);
        Bm25Weight::new(idf_explain, avg_fieldnorm, params)
    }

    pub fn for_one_term_without_explain(
        term_doc_freq: u64,
        total_num_docs: u64,
        avg_fieldnorm: Score,
        params: Bm25Params,
    ) -> Bm25Weight {
        let idf = idf(term_doc_freq, total_num_docs);
        Bm25Weight::new_without_explain(idf, avg_fieldnorm, params)
    }

    pub(crate) fn new(
        idf_explain: Explanation,
        average_fieldnorm: Score,
        params: Bm25Params,
    ) -> Bm25Weight {
        let weight = idf_explain.value() * (1.0 + params.k1());
        Bm25Weight {
            idf_explain: Some(idf_explain),
            weight,
            cache: compute_tf_cache(average_fieldnorm, params.k1(), params.b()),
            average_fieldnorm,
            params,
        }
    }

    pub(crate) fn new_without_explain(
        idf: f32,
        average_fieldnorm: Score,
        params: Bm25Params,
    ) -> Bm25Weight {
        let weight = idf * (1.0 + params.k1());
        Bm25Weight {
            idf_explain: None,
            weight,
            cache: compute_tf_cache(average_fieldnorm, params.k1(), params.b()),
            average_fieldnorm,
            params,
        }
    }

    #[inline]
    pub fn score(&self, fieldnorm_id: u8, term_freq: u32) -> Score {
        self.weight * self.tf_factor(fieldnorm_id, term_freq)
    }

    pub fn max_score(&self) -> Score {
        self.score(255u8, 2_013_265_944)
    }

    #[inline]
    pub(crate) fn tf_factor(&self, fieldnorm_id: u8, term_freq: u32) -> Score {
        let term_freq = term_freq as Score;
        let norm = self.cache[fieldnorm_id as usize];
        term_freq / (term_freq + norm)
    }

    pub fn explain(&self, fieldnorm_id: u8, term_freq: u32) -> Explanation {
        let score = self.score(fieldnorm_id, term_freq);

        let norm = self.cache[fieldnorm_id as usize];
        let term_freq = term_freq as Score;
        let right_factor = term_freq / (term_freq + norm);

        let mut tf_explanation = Explanation::new(
            "freq / (freq + k1 * (1 - b + b * dl / avgdl))",
            right_factor,
        );

        tf_explanation.add_const("freq, occurrences of term within document", term_freq);
        tf_explanation.add_const("k1, term saturation parameter", self.params.k1());
        tf_explanation.add_const("b, length normalization parameter", self.params.b());
        tf_explanation.add_const(
            "dl, length of field",
            FieldNormReader::id_to_fieldnorm(fieldnorm_id) as Score,
        );
        tf_explanation.add_const("avgdl, average length of field", self.average_fieldnorm);

        let mut explanation = Explanation::new("TermQuery, product of...", score);
        explanation.add_detail(Explanation::new("(K1+1)", self.params.k1() + 1.0));
        if let Some(idf_explain) = &self.idf_explain {
            explanation.add_detail(idf_explain.clone());
        }
        explanation.add_detail(tf_explanation);
        explanation
    }
}

#[cfg(test)]
mod tests {

    use super::idf;
    use crate::{assert_nearly_equals, Score};

    #[test]
    fn test_idf() {
        let score: Score = 2.0;
        assert_nearly_equals!(idf(1, 2), score.ln());
    }

    #[test]
    fn test_custom_bm25_params_produce_different_scores() {
        use super::Bm25Weight;
        use crate::index::Bm25Params;

        let default_params = Bm25Params::default();
        let custom_params = Bm25Params::new(2.0, 0.3);

        let w_default = Bm25Weight::for_one_term(10, 100, 50.0, default_params);
        let w_custom = Bm25Weight::for_one_term(10, 100, 50.0, custom_params);

        let fieldnorm_id = 10u8;
        let term_freq = 5u32;

        let score_default = w_default.score(fieldnorm_id, term_freq);
        let score_custom = w_custom.score(fieldnorm_id, term_freq);

        assert!(
            (score_default - score_custom).abs() > 1e-6,
            "Custom k1/b should produce different scores: default={score_default}, \
             custom={score_custom}"
        );
    }

    #[test]
    #[should_panic(expected = "k1 must be non-negative")]
    fn test_bm25_params_rejects_negative_k1() {
        use crate::index::Bm25Params;
        Bm25Params::new(-1.0, 0.75);
    }

    #[test]
    #[should_panic(expected = "b must be in [0, 1]")]
    fn test_bm25_params_rejects_b_out_of_range() {
        use crate::index::Bm25Params;
        Bm25Params::new(1.2, 1.5);
    }

    #[cfg(feature = "quickwit")]
    #[test]
    fn resolved_terms_preserve_scores_and_statistics_fallback() -> crate::Result<()> {
        use super::{Bm25StatisticsProvider, ResolvedTerms};
        use crate::collector::TopDocs;
        use crate::indexer::NoMergePolicy;
        use crate::query::{
            BooleanQuery, BoostQuery, EnableScoring, Occur, PhraseQuery, Query, TermQuery,
        };
        use crate::schema::{Field, IndexRecordOption, Schema, TEXT};
        use crate::{Index, IndexWriter, Searcher, Term};

        struct CustomStatistics<'a>(&'a Searcher, Option<u64>);
        impl Bm25StatisticsProvider for CustomStatistics<'_> {
            fn total_num_tokens(&self, field: Field) -> crate::Result<u64> {
                Bm25StatisticsProvider::total_num_tokens(self.0, field)
            }
            fn total_num_docs(&self) -> crate::Result<u64> {
                Bm25StatisticsProvider::total_num_docs(self.0)
            }
            fn doc_freq(&self, term: &Term) -> crate::Result<u64> {
                match self.1 {
                    Some(doc_freq) => Ok(doc_freq),
                    None => self.0.doc_freq(term),
                }
            }
        }

        let mut schema = Schema::builder();
        let first = schema.add_text_field("first", TEXT);
        let second = schema.add_text_field("second", TEXT);
        let index = Index::create_in_ram(schema.build());
        let mut writer: IndexWriter = index.writer_for_tests()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for text in ["rust rust memory", "memory safety"] {
            writer.add_document(doc!(first => text, second => "rust"))?;
            writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        let terms = [
            Term::from_field_text(first, "rust"),
            Term::from_field_text(first, "missing"),
            Term::from_field_text(second, "rust"),
        ];
        let resolved = ResolvedTerms::new(&searcher, terms.iter().chain([&terms[0]]))?;
        assert_eq!(resolved.term_infos.len(), terms.len());
        for term in &terms {
            assert_eq!(resolved.term_infos[term].doc_freq, searcher.doc_freq(term)?);
            for segment in searcher.segment_readers() {
                assert_eq!(
                    resolved.term_infos[term].get(segment, term)?,
                    segment.inverted_index(term.field())?.get_term_info(term)?
                );
            }
        }
        let term_query = |term: &Term| -> Box<dyn Query> {
            Box::new(TermQuery::new(term.clone(), IndexRecordOption::WithFreqs))
        };
        let queries: Vec<Box<dyn Query>> = vec![
            term_query(&terms[0]),
            term_query(&terms[1]),
            Box::new(PhraseQuery::new(vec![terms[0].clone(), terms[0].clone()])),
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, term_query(&terms[0])),
                (
                    Occur::Should,
                    Box::new(BoostQuery::new(term_query(&terms[0]), 2.0)),
                ),
                (Occur::Should, term_query(&terms[2])),
                (Occur::MustNot, term_query(&terms[1])),
            ])),
        ];
        let custom = CustomStatistics(&searcher, None);
        let collector = TopDocs::with_limit(10).order_by_score();
        for query in queries {
            assert_eq!(
                searcher.search(query.as_ref(), &collector)?,
                searcher.search_with_statistics_provider(query.as_ref(), &collector, &custom)?
            );
        }
        let query = term_query(&terms[0]);
        assert_ne!(
            searcher.search(query.as_ref(), &collector)?,
            searcher.search_with_statistics_provider(
                query.as_ref(),
                &collector,
                &CustomStatistics(&searcher, Some(0)),
            )?
        );
        assert!(ResolvedTerms::for_scoring(
            EnableScoring::enabled_from_searcher(&searcher),
            &terms
        )?
        .is_some());
        assert!(ResolvedTerms::for_scoring(
            EnableScoring::disabled_from_searcher(&searcher),
            &terms
        )?
        .is_none());
        assert!(ResolvedTerms::for_scoring(
            EnableScoring::enabled_from_statistics_provider(&custom, &searcher),
            &terms
        )?
        .is_none());
        Ok(())
    }
}
