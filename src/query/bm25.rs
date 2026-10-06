use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::fieldnorm::FieldNormReader;
use crate::index::Bm25Params;
use crate::query::Explanation;
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

    /// Returns document frequencies in input order, including repeated terms.
    /// Defaults to looking up each term separately.
    fn doc_freqs(&self, terms: &[&Term]) -> crate::Result<Vec<u64>> {
        terms.iter().map(|term| self.doc_freq(term)).collect()
    }

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

    #[cfg(feature = "quickwit")]
    fn doc_freqs(&self, terms: &[&Term]) -> crate::Result<Vec<u64>> {
        let mut sorted_terms = terms.to_vec();
        sorted_terms.sort_unstable();
        sorted_terms.dedup();
        let mut doc_freqs = vec![0u64; sorted_terms.len()];
        let mut start = 0;
        while start < sorted_terms.len() {
            let field = sorted_terms[start].field();
            let end = start + sorted_terms[start..].partition_point(|term| term.field() == field);
            let keys: Vec<_> = sorted_terms[start..end]
                .iter()
                .map(|term| term.serialized_value_bytes())
                .collect();
            // Term ordering includes a type tag that dictionary keys omit.
            let batch_lookup =
                keys.len() > 1 && crate::termdict::SortedTermSlice::new(&keys).is_some();
            for segment in self.segment_readers() {
                let reader = segment.inverted_index(field)?;
                if batch_lookup {
                    let keys = crate::termdict::SortedTermSlice::new_assume_sorted(&keys);
                    for (index, info) in reader.get_term_infos(keys)?.into_iter().enumerate() {
                        if let Some(info) = info {
                            doc_freqs[start + index] += u64::from(info.doc_freq);
                        }
                    }
                } else {
                    for (term, doc_freq) in sorted_terms[start..end]
                        .iter()
                        .zip(&mut doc_freqs[start..end])
                    {
                        *doc_freq += u64::from(reader.doc_freq(term)?);
                    }
                }
            }
            start = end;
        }
        Ok(terms
            .iter()
            .map(|term| doc_freqs[sorted_terms.binary_search(term).unwrap()])
            .collect())
    }

    fn bm25_params(&self, field: Field) -> Bm25Params {
        self.schema()
            .get_field_entry(field)
            .field_type()
            .bm25_params()
            .unwrap_or_default()
    }
}

pub(crate) struct BatchedStatistics<'a> {
    provider: &'a dyn Bm25StatisticsProvider,
    doc_freqs: FxHashMap<&'a Term, u64>,
}

impl<'a> BatchedStatistics<'a> {
    pub fn new(
        provider: &'a dyn Bm25StatisticsProvider,
        terms: &[&'a Term],
    ) -> crate::Result<Self> {
        let doc_freqs = terms
            .iter()
            .copied()
            .zip(provider.doc_freqs(terms)?)
            .collect();
        Ok(Self {
            provider,
            doc_freqs,
        })
    }
}

impl Bm25StatisticsProvider for BatchedStatistics<'_> {
    fn total_num_tokens(&self, field: Field) -> crate::Result<u64> {
        self.provider.total_num_tokens(field)
    }

    fn total_num_docs(&self) -> crate::Result<u64> {
        self.provider.total_num_docs()
    }

    fn doc_freq(&self, term: &Term) -> crate::Result<u64> {
        match self.doc_freqs.get(term) {
            Some(&doc_freq) => Ok(doc_freq),
            None => self.provider.doc_freq(term),
        }
    }

    fn bm25_params(&self, field: Field) -> Bm25Params {
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
    norm_const: Score,
    norm_factor: Score,
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
            norm_const: self.norm_const,
            norm_factor: self.norm_factor,
        }
    }

    pub(crate) fn for_phrase_pruning(&self, indexing_average: Score) -> Option<Self> {
        if !self.weight.is_finite()
            || self.weight < 0.0
            || !indexing_average.is_finite()
            || indexing_average <= 0.0
            || !self.average_fieldnorm.is_finite()
            || self.average_fieldnorm <= 0.0
        {
            return None;
        }
        let mut bound = self.boost_by(1.0 + 4.0 * Score::EPSILON);
        if !bound.weight.is_finite() {
            return None;
        }
        bound.cache = compute_tf_cache(indexing_average, self.params.k1(), self.params.b());
        let scale = (indexing_average / self.average_fieldnorm).min(1.0);
        // Uniform scaling preserves the index-time winner and bounds the query-time TF factor.
        for value in Arc::make_mut(&mut bound.cache) {
            *value *= scale;
            if !value.is_finite() || *value < 0.0 {
                return None;
            }
        }
        Some(bound)
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
            for term_doc_freq in statistics.doc_freqs(&terms.iter().collect::<Vec<_>>())? {
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
        let norm_const = params.k1() * (1.0 - params.b());
        let norm_factor = if average_fieldnorm > 0.0 {
            (params.k1() * params.b()) / average_fieldnorm
        } else {
            0.0
        };
        Bm25Weight {
            idf_explain: Some(idf_explain),
            weight,
            cache: compute_tf_cache(average_fieldnorm, params.k1(), params.b()),
            average_fieldnorm,
            params,
            norm_const,
            norm_factor,
        }
    }

    pub(crate) fn new_without_explain(
        idf: f32,
        average_fieldnorm: Score,
        params: Bm25Params,
    ) -> Bm25Weight {
        let weight = idf * (1.0 + params.k1());
        let norm_const = params.k1() * (1.0 - params.b());
        let norm_factor = if average_fieldnorm > 0.0 {
            (params.k1() * params.b()) / average_fieldnorm
        } else {
            0.0
        };
        Bm25Weight {
            idf_explain: None,
            weight,
            cache: compute_tf_cache(average_fieldnorm, params.k1(), params.b()),
            average_fieldnorm,
            params,
            norm_const,
            norm_factor,
        }
    }

    #[inline]
    pub fn norm_const(&self) -> Score {
        self.norm_const
    }

    #[inline]
    pub fn norm_factor(&self) -> Score {
        self.norm_factor
    }

    #[inline]
    pub fn weight(&self) -> Score {
        self.weight
    }

    #[inline]
    pub fn score(&self, fieldnorm_id: u8, term_freq: u32) -> Score {
        self.weight * self.tf_factor(fieldnorm_id, term_freq)
    }

    #[inline]
    pub fn score_fieldnorm(&self, fieldnorm: u32, term_freq: u32) -> Score {
        let norm = self.norm_const + self.norm_factor * (fieldnorm as Score);
        let term_freq = term_freq as Score;
        self.weight * (term_freq / (term_freq + norm))
    }

    /// Computes BM25 scores for a block of `(freqs, norms)` pairs, storing the results in `scores`,
    /// and returns the maximum score across the evaluated elements.
    #[inline]
    pub fn compute_block_scores(
        &self,
        freqs: &[u32],
        norms: &[u32],
        scores: &mut [Score],
    ) -> Score {
        let len = freqs.len().min(norms.len()).min(scores.len());
        if len == 0 {
            return 0.0;
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.compute_block_scores_neon(&freqs[..len], &norms[..len], &mut scores[..len])
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.compute_block_scores_sse2(&freqs[..len], &norms[..len], &mut scores[..len])
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            self.compute_block_scores_scalar(&freqs[..len], &norms[..len], &mut scores[..len])
        }
    }

    #[inline]
    #[cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]
    pub(crate) fn compute_block_scores_scalar(
        &self,
        freqs: &[u32],
        norms: &[u32],
        scores: &mut [Score],
    ) -> Score {
        let len = freqs.len().min(norms.len()).min(scores.len());
        let mut max_score = 0.0f32;
        for i in 0..len {
            let s = self.score_fieldnorm(norms[i], freqs[i]);
            scores[i] = s;
            max_score = max_score.max(s);
        }
        max_score
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn compute_block_scores_neon(
        &self,
        freqs: &[u32],
        norms: &[u32],
        scores: &mut [Score],
    ) -> Score {
        use core::arch::aarch64::*;

        let len = freqs.len().min(norms.len()).min(scores.len());
        let v_norm_const = vdupq_n_f32(self.norm_const);
        let v_norm_factor = vdupq_n_f32(self.norm_factor);
        let v_weight = vdupq_n_f32(self.weight);
        let mut v_max = vdupq_n_f32(0.0f32);

        let chunks = len / 4;
        let freqs_ptr = freqs.as_ptr();
        let norms_ptr = norms.as_ptr();
        let scores_ptr = scores.as_mut_ptr();

        for c in 0..chunks {
            let i = c * 4;
            let tf_u32 = vld1q_u32(freqs_ptr.add(i));
            let norm_u32 = vld1q_u32(norms_ptr.add(i));
            let tf = vcvtq_f32_u32(tf_u32);
            let norm_f = vcvtq_f32_u32(norm_u32);
            let norm = vaddq_f32(v_norm_const, vmulq_f32(v_norm_factor, norm_f));
            let denom = vaddq_f32(tf, norm);
            let ratio = vdivq_f32(tf, denom);
            let s = vmulq_f32(v_weight, ratio);
            vst1q_f32(scores_ptr.add(i), s);
            v_max = vmaxq_f32(v_max, s);
        }

        let mut max_score = if chunks > 0 {
            vmaxvq_f32(v_max)
        } else {
            0.0f32
        };

        for i in (chunks * 4)..len {
            let s = self.score_fieldnorm(norms[i], freqs[i]);
            scores[i] = s;
            max_score = max_score.max(s);
        }

        max_score
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn compute_block_scores_sse2(
        &self,
        freqs: &[u32],
        norms: &[u32],
        scores: &mut [Score],
    ) -> Score {
        use core::arch::x86_64::*;

        let len = freqs.len().min(norms.len()).min(scores.len());
        let v_norm_const = _mm_set1_ps(self.norm_const);
        let v_norm_factor = _mm_set1_ps(self.norm_factor);
        let v_weight = _mm_set1_ps(self.weight);
        let mut v_max = _mm_set1_ps(0.0f32);

        let chunks = len / 4;
        let freqs_ptr = freqs.as_ptr();
        let norms_ptr = norms.as_ptr();
        let scores_ptr = scores.as_mut_ptr();

        for c in 0..chunks {
            let i = c * 4;
            let tf_raw = _mm_loadu_si128(freqs_ptr.add(i) as *const __m128i);
            let norm_raw = _mm_loadu_si128(norms_ptr.add(i) as *const __m128i);
            let tf = _mm_cvtepi32_ps(tf_raw);
            let norm_f = _mm_cvtepi32_ps(norm_raw);
            let norm = _mm_add_ps(v_norm_const, _mm_mul_ps(v_norm_factor, norm_f));
            let denom = _mm_add_ps(tf, norm);
            let ratio = _mm_div_ps(tf, denom);
            let s = _mm_mul_ps(v_weight, ratio);
            _mm_storeu_ps(scores_ptr.add(i), s);
            v_max = _mm_max_ps(v_max, s);
        }

        let mut max_score = if chunks > 0 {
            let mut max_arr = [0.0f32; 4];
            _mm_storeu_ps(max_arr.as_mut_ptr(), v_max);
            max_arr[0].max(max_arr[1]).max(max_arr[2]).max(max_arr[3])
        } else {
            0.0f32
        };

        for i in (chunks * 4)..len {
            let s = self.score_fieldnorm(norms[i], freqs[i]);
            scores[i] = s;
            max_score = max_score.max(s);
        }

        max_score
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
    fn phrase_pruning_rejects_unsupported_weights() {
        let weight = super::Bm25Weight::for_one_term(10, 100, 20.0, crate::Bm25Params::default());
        for average in [0.0, -1.0, Score::NAN, Score::INFINITY] {
            assert!(weight.for_phrase_pruning(average).is_none());
        }
        for boost in [-1.0, Score::NAN, Score::INFINITY] {
            assert!(weight.boost_by(boost).for_phrase_pruning(10.0).is_none());
        }
        assert!(weight.boost_by(0.0).for_phrase_pruning(10.0).is_some());
    }

    proptest::proptest! {
        #[test]
        fn phrase_block_bound_covers_different_segment_averages(
            pairs in proptest::collection::vec((0u8..=255, 1u32..1000), 1..129),
            index_average in 1.0f32..2000.0,
            query_average in 1.0f32..2000.0,
            k1 in 0.0f32..4.0,
            b in 0.0f32..1.0,
        ) {
            use super::Bm25Weight;
            use crate::postings::skip::{decode_block_wand_max_tf, encode_block_wand_max_tf};

            let params = crate::Bm25Params::new(k1, b);
            let index = Bm25Weight::for_one_term(10, 100, index_average, params);
            let query = Bm25Weight::for_one_term(10, 100, query_average, params);
            let &(norm, freq) = pairs.iter().max_by(|&&(a, af), &&(b, bf)| {
                index.tf_factor(a, af).total_cmp(&index.tf_factor(b, bf))
            }).unwrap();
            let stored_freq = decode_block_wand_max_tf(encode_block_wand_max_tf(freq));
            let bound = query.for_phrase_pruning(index_average).unwrap().score(norm, stored_freq);
            for (norm, freq) in pairs {
                proptest::prop_assert!(bound >= query.score(norm, freq));
            }
        }
    }

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

    #[test]
    fn test_compute_block_scores() {
        use super::Bm25Weight;
        use crate::index::Bm25Params;

        let weight = Bm25Weight::for_one_term(10, 1000, 45.0, Bm25Params::default());

        // Test lengths from 0 to 128
        let freqs: Vec<u32> = (0..128).map(|i| (i % 7) + 1).collect();
        let norms: Vec<u32> = (0..128).map(|i| ((i * 13) % 150) + 1).collect();

        for len in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 63, 64, 127, 128] {
            let mut scores = vec![0.0f32; len];
            let mut scalar_scores = vec![0.0f32; len];

            let max_score = weight.compute_block_scores(&freqs[..len], &norms[..len], &mut scores);
            let scalar_max = weight.compute_block_scores_scalar(
                &freqs[..len],
                &norms[..len],
                &mut scalar_scores,
            );

            assert!((max_score - scalar_max).abs() < 1e-6);

            let mut expected_max = 0.0f32;
            for i in 0..len {
                let expected = weight.score_fieldnorm(norms[i], freqs[i]);
                assert!((scores[i] - expected).abs() < 1e-6);
                assert!((scalar_scores[i] - expected).abs() < 1e-6);
                expected_max = expected_max.max(expected);
            }
            assert!((max_score - expected_max).abs() < 1e-6);
        }
    }
}
