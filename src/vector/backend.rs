//! Per-segment vector search execution.
//! Supports flat scans and routed quantized scans.

use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use common::BitSet;
use quant_model::f16::f16_to_f32;

use super::bounds::{
    bounds_verdict, margin_ball_ball, margin_ball_halfspace, to_bound_space, HeapPeek, QueryBound,
    QueryBoundTracker, Verdict,
};
use super::distance::norm_squared_wide;
use super::index_reader::{QuantizedFieldReader, QuantizedLayerReader, VectorIndexReader};
use super::ivf::{AdaptiveProbeParams, Candidate, IvfIndex, IvfSearchMetrics, Workspace};
use super::prepared::{
    corrected_quantized_estimate, initial_dot_raw_prefix, initial_l2_raw_prefix,
    quantized_model_sigma, refine_dot_raw_prefix, refine_l2_raw_prefix, PreparedQuery,
    QuantizedQueryCache, QuantizedQueryCtx,
};
use super::quantization::QUANTIZED_BOUNDARY_KAPPA;
use super::tie_break::NoTieBreak;
use super::VectorElement;
use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, TopNComputer};
use crate::error::DataCorruption;
use crate::fastfield::AliveBitSet;
use crate::query::Weight;
use crate::schema::{Field, Metric};
use crate::{DocAddress, DocId, Score, SegmentOrdinal, SegmentReader, TantivyError};

/// The settled result.
type TieBreakHits<K> = Vec<(
    (Score, <K as SegmentSortKeyComputer>::SegmentSortKey),
    DocAddress,
)>;

/// The in-flight accumulator.
type TieBreakHeap<K, CTail> = TopNComputer<
    (Score, <K as SegmentSortKeyComputer>::SegmentSortKey),
    DocId,
    (NaturalComparator, CTail),
>;

/// Per-segment vector search state.
pub struct VectorBackend<T: VectorElement> {
    reader: Arc<VectorIndexReader>,
    query: Arc<PreparedQuery<T>>,
    quantized_query: Option<Arc<QuantizedQueryCtx>>,
    adaptive: AdaptiveProbeParams,
    segment_ord: SegmentOrdinal,
}

impl<T: VectorElement> VectorBackend<T> {
    /// Prepares a segment vector backend.
    pub(crate) fn for_segment(
        segment_reader: &SegmentReader,
        segment_ord: SegmentOrdinal,
        field: Field,
        query: Arc<Vec<T>>,
        quantized_queries: &QuantizedQueryCache,
        adaptive: AdaptiveProbeParams,
        max_scan_levels: usize,
    ) -> crate::Result<Self> {
        let reader = segment_reader.vector_index(field)?;
        let quantized_query = if max_scan_levels == 0 {
            None
        } else if let Some(quantized) = reader.quantization() {
            let index_ctx = quantized.index_ctx()?;
            let active_layers = max_scan_levels.min(index_ctx.specs.len());
            Some(quantized_queries.resolve(index_ctx, query.as_slice(), active_layers))
        } else {
            None
        };
        let query = Arc::new(PreparedQuery::<T>::new(reader.options().metric(), query));
        Ok(Self {
            reader,
            query,
            quantized_query,
            adaptive,
            segment_ord,
        })
    }

    /// Returns the segment's top vector matches and probe statistics.
    ///
    /// # Errors
    ///
    /// Returns an error when segment vector data cannot be opened or scored.
    pub fn top_n(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let (hits, stats) = self.top_n_by(
            weight,
            segment_reader,
            top_n,
            &mut NoTieBreak,
            NaturalComparator,
        )?;
        Ok((
            hits.into_iter()
                .map(|((score, ()), address)| (score, address))
                .collect(),
            stats,
        ))
    }

    /// Returns top matches using a secondary sort key.
    ///
    /// # Errors
    ///
    /// Returns an error when segment vector data cannot be opened or scored.
    pub fn top_n_by<K, CTail>(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
    ) -> crate::Result<(TieBreakHits<K>, ProbeStats)>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let mut stats = ProbeStats::default();
        let hits = match self.reader.index() {
            None => self.exact_top_n(
                weight,
                segment_reader,
                top_n,
                tie_break,
                tie_comparator,
                &mut stats,
            )?,
            Some(index) => match &self.quantized_query {
                Some(quantized_query) => self.quantized_top_n(
                    index,
                    quantized_query,
                    weight,
                    segment_reader,
                    top_n,
                    tie_break,
                    tie_comparator,
                    &mut stats,
                )?,
                None => self.approximate_top_n(
                    index,
                    weight,
                    segment_reader,
                    top_n,
                    tie_break,
                    tie_comparator,
                    &mut stats,
                )?,
            },
        };
        Ok((hits, stats))
    }

    /// Scans full-precision rows matching a filter.
    fn exact_top_n<K, CTail>(
        &self,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let mut topn = TopNComputer::with_comparator(top_n, (NaturalComparator, tie_comparator));
        let alive = segment_reader.alive_bitset();
        let mut rows_read = 0usize;
        let mut read_err: Option<TantivyError> = None;
        weight.for_each_no_score(segment_reader, &mut |docs| {
            if read_err.is_some() {
                return;
            }
            for &doc in docs {
                if let Some(bs) = alive {
                    if !bs.is_alive(doc) {
                        continue;
                    }
                }
                let Some(row) = self.reader.row_id(doc) else {
                    continue;
                };
                match self.reader.vector_bytes_for_row(row) {
                    Ok(vbytes) => {
                        rows_read += 1;
                        let score = self.query.score_doc_bytes(&vbytes);
                        if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                            topn.push(key, doc);
                        }
                    }
                    Err(err) => {
                        read_err = Some(err);
                        return;
                    }
                }
            }
        })?;
        if let Some(err) = read_err {
            return Err(err);
        }
        stats.exact_rows_read += rows_read;
        let segment_ord = self.segment_ord;
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect();
        Ok(hits)
    }
}

/// Builds a competitive candidate's composite heap key.
#[inline(always)]
fn tie_break_key<K, CTail>(
    topn: &TieBreakHeap<K, CTail>,
    tie_break: &mut K,
    score: Score,
    doc: DocId,
) -> Option<(Score, K::SegmentSortKey)>
where
    K: SegmentSortKeyComputer,
    CTail: Comparator<K::SegmentSortKey>,
{
    if let Some(((threshold_score, _), _)) = &topn.threshold {
        if score < *threshold_score {
            return None;
        }
    }
    Some((score, tie_break.segment_sort_key(doc, score)))
}

/// How the probe loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
pub enum ProbeTermination {
    /// The work-unit probe budget was spent - the probe ceiling.
    Ceiling,
    /// The centroid stream was exhausted.
    #[default]
    Exhausted,
}

/// Candidate identities at quantized stage boundaries.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct QuantizedStageTrace {
    pub(crate) scored_docs: Vec<DocId>,
    pub(crate) boundary_docs: Vec<Vec<DocId>>,
    pub(crate) rerank_docs: Vec<DocId>,
}

#[derive(Debug, Default, serde::Serialize)]
/// Counters for one vector probe.
pub struct ProbeStats {
    /// Rows scored by the active path.
    pub candidates_scored: usize,
    #[cfg(test)]
    #[serde(skip)]
    pub(crate) quantized_trace: QuantizedStageTrace,
    /// Documents visited before pruning.
    pub vectors_visited: usize,
    /// Touched docs rejected by `filter.contains`.
    pub pruned_filter: usize,
    /// Touched docs rejected by `is_alive`.
    pub pruned_dead: usize,
    /// Probed clusters with fetched posting rows.
    pub postings_row: usize,
    /// Probed clusters without fetched posting rows.
    pub postings_skipped: usize,
    /// Full-precision row reads performed by the exact path.
    pub exact_rows_read: usize,
    /// Cluster-routing counters.
    pub routing: IvfSearchMetrics,
    /// Clusters rejected by the bounds gate.
    pub bounds_skips: u32,
    /// Probe index at which the bound armed.
    pub bound_armed_at_probe: Option<u32>,
    /// How the probe loop terminated. Per-segment; does not sum.
    pub termination: ProbeTermination,
    /// Work units charged by the probe loop.
    pub work_charged: f32,
}

impl ProbeStats {
    fn record_routing(&mut self, routing: IvfSearchMetrics) {
        self.routing = routing;
    }

    fn record_bound_armed(&mut self, at_probe: Option<u32>) {
        self.bound_armed_at_probe = at_probe;
    }

    /// Returns the probed-cluster count.
    #[inline]
    pub fn clusters_probed(&self) -> usize {
        self.postings_row + self.postings_skipped
    }
}

/// Fixed cluster-open cost in row-work units.
pub const DEFAULT_FIXED_PROBE_COST_ROWS: f64 = 1.64;

/// Current fixed cluster-open cost stored as binary64 bits.
static FIXED_PROBE_COST_ROWS_BITS: AtomicU64 =
    AtomicU64::new(DEFAULT_FIXED_PROBE_COST_ROWS.to_bits());

/// Sets the fixed cluster-open cost in row-work units.
pub fn set_fixed_probe_cost_rows(v: f64) {
    let v = if v.is_finite() && v > 0.0 {
        v
    } else {
        DEFAULT_FIXED_PROBE_COST_ROWS
    };
    FIXED_PROBE_COST_ROWS_BITS.store(v.to_bits(), Relaxed);
}

/// Returns the fixed cluster-open cost in row-work units.
pub(crate) fn fixed_probe_cost_rows() -> f64 {
    f64::from_bits(FIXED_PROBE_COST_ROWS_BITS.load(Relaxed))
}

/// Returns the cluster-open share of average posting work.
pub(crate) fn open_share(n_avg: f64) -> f64 {
    let fixed = fixed_probe_cost_rows();
    (fixed / (fixed + n_avg.max(0.0))).min(0.5)
}

/// Probe work measured in average-cluster units.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
pub struct WorkUnits(f64);

impl WorkUnits {
    /// No work.
    pub const ZERO: WorkUnits = WorkUnits(0.0);

    /// Wraps an amount in work units.
    #[inline]
    pub fn new(units: f64) -> WorkUnits {
        WorkUnits(units)
    }

    /// Returns the binary64 amount.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }

    /// Returns the amount narrowed to binary32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32
    }
}

impl std::ops::Add for WorkUnits {
    type Output = WorkUnits;
    #[inline]
    fn add(self, rhs: WorkUnits) -> WorkUnits {
        WorkUnits(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for WorkUnits {
    #[inline]
    fn add_assign(&mut self, rhs: WorkUnits) {
        self.0 += rhs.0;
    }
}

impl std::ops::Mul<f64> for WorkUnits {
    type Output = WorkUnits;
    /// Scales work by a count.
    #[inline]
    fn mul(self, rhs: f64) -> WorkUnits {
        WorkUnits(self.0 * rhs)
    }
}

/// Per-segment probe budget and event prices.
#[derive(Clone, Copy, Debug)]
struct UnitPricing {
    /// Work this segment may spend before the ceiling binds.
    budget: WorkUnits,
    /// The per-index open share `x`: what opening one cluster costs.
    open: WorkUnits,
    /// `(1 - x)/n_avg`: what one scored row costs.
    row: WorkUnits,
}

/// One row surviving the cluster pre-pass.
#[derive(Clone, Copy)]
struct Survivor {
    row: usize,
    doc: DocId,
}

/// One row surviving a quantized boundary.
#[derive(Clone, Copy)]
struct QuantizedCandidate {
    row: usize,
    doc: DocId,
    base: f32,
    raw_prefix: f32,
    estimate: f32,
    sigma: f32,
    residual_norm_squared: f32,
    gamma: f32,
    sign_query_error_term: f32,
}

/// Storage-row selection resolved before a quantized layer is read.
enum Selection<'a> {
    All,
    Rows(&'a [usize]),
    None,
}

impl Selection<'_> {
    #[inline]
    fn len(&self, rows: &Range<usize>) -> usize {
        match self {
            Self::All => rows.len(),
            Self::Rows(offsets) => offsets.len(),
            Self::None => 0,
        }
    }
}

/// Row-parallel quantized scan columns.
struct QuantizedCandidates {
    rows: Vec<usize>,
    docs: Vec<DocId>,
    bases: Vec<f32>,
    raw_prefixes: Vec<f32>,
    estimates: Vec<f32>,
    sigmas: Vec<f32>,
    residual_norm_squared: Vec<f32>,
    gammas: Vec<f32>,
    sign_query_error_terms: Vec<f32>,
}

impl QuantizedCandidates {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            rows: Vec::with_capacity(capacity),
            docs: Vec::with_capacity(capacity),
            bases: Vec::with_capacity(capacity),
            raw_prefixes: Vec::with_capacity(capacity),
            estimates: Vec::with_capacity(capacity),
            sigmas: Vec::with_capacity(capacity),
            residual_norm_squared: Vec::with_capacity(capacity),
            gammas: Vec::with_capacity(capacity),
            sign_query_error_terms: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.rows.len()
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn push(
        &mut self,
        row: usize,
        doc: DocId,
        base: f32,
        raw_prefix: f32,
        estimate: f32,
        sigma: f32,
        residual_norm_squared: f32,
        gamma: f32,
        sign_query_error_term: f32,
    ) {
        self.rows.push(row);
        self.docs.push(doc);
        self.bases.push(base);
        self.raw_prefixes.push(raw_prefix);
        self.estimates.push(estimate);
        self.sigmas.push(sigma);
        self.residual_norm_squared.push(residual_norm_squared);
        self.gammas.push(gamma);
        self.sign_query_error_terms.push(sign_query_error_term);
    }

    #[allow(clippy::too_many_arguments)]
    fn append_selected(
        &mut self,
        rows: Range<usize>,
        selection: &Selection<'_>,
        docs: &[DocId],
        bases: &[f32],
        raw_prefixes: &[f32],
        estimates: &[f32],
        sigmas: &[f32],
        residual_norms_squared: &[f32],
        gammas: &[f32],
        sign_query_error_terms: &[f32],
    ) {
        let len = selection.len(&rows);
        debug_assert_eq!(docs.len(), len);
        debug_assert_eq!(bases.len(), len);
        debug_assert_eq!(raw_prefixes.len(), len);
        debug_assert_eq!(estimates.len(), len);
        debug_assert_eq!(sigmas.len(), len);
        debug_assert_eq!(residual_norms_squared.len(), len);
        debug_assert_eq!(gammas.len(), len);
        debug_assert_eq!(sign_query_error_terms.len(), len);
        match selection {
            Selection::All => self.rows.extend(rows),
            Selection::Rows(offsets) => self.rows.extend(offsets.iter().map(|&offset| {
                debug_assert!(offset < rows.len());
                rows.start + offset
            })),
            Selection::None => unreachable!("empty selections are skipped before scoring"),
        }
        self.docs.extend_from_slice(docs);
        self.bases.extend_from_slice(bases);
        self.raw_prefixes.extend_from_slice(raw_prefixes);
        self.estimates.extend_from_slice(estimates);
        self.sigmas.extend_from_slice(sigmas);
        self.residual_norm_squared
            .extend_from_slice(residual_norms_squared);
        self.gammas.extend_from_slice(gammas);
        self.sign_query_error_terms
            .extend_from_slice(sign_query_error_terms);
    }

    #[inline(always)]
    fn estimate(&self, index: usize) -> f32 {
        self.estimates[index]
    }

    fn materialize(&self, index: usize) -> QuantizedCandidate {
        QuantizedCandidate {
            row: self.rows[index],
            doc: self.docs[index],
            base: self.bases[index],
            raw_prefix: self.raw_prefixes[index],
            estimate: self.estimates[index],
            sigma: self.sigmas[index],
            residual_norm_squared: self.residual_norm_squared[index],
            gamma: self.gammas[index],
            sign_query_error_term: self.sign_query_error_terms[index],
        }
    }

    fn replace_with_boundary_survivors(&mut self, survivors: &[QuantizedCandidate]) {
        self.rows.clear();
        self.docs.clear();
        self.bases.clear();
        self.raw_prefixes.clear();
        self.estimates.clear();
        self.sigmas.clear();
        self.residual_norm_squared.clear();
        self.gammas.clear();
        self.sign_query_error_terms.clear();
        self.rows.reserve(survivors.len());
        self.docs.reserve(survivors.len());
        self.bases.reserve(survivors.len());
        self.raw_prefixes.reserve(survivors.len());
        self.estimates.reserve(survivors.len());
        self.sigmas.reserve(survivors.len());
        self.residual_norm_squared.reserve(survivors.len());
        self.gammas.reserve(survivors.len());
        self.sign_query_error_terms.reserve(survivors.len());
        for survivor in survivors {
            self.push(
                survivor.row,
                survivor.doc,
                survivor.base,
                survivor.raw_prefix,
                survivor.estimate,
                survivor.sigma,
                survivor.residual_norm_squared,
                survivor.gamma,
                survivor.sign_query_error_term,
            );
        }
    }
}

/// Decodes a binary16 run into binary32 scratch storage.
#[inline(always)]
fn decode_f16s(values: &[u8], decoded: &mut Vec<f32>) {
    assert_eq!(values.len() % std::mem::size_of::<u16>(), 0);
    decoded.resize(values.len() / std::mem::size_of::<u16>(), 0.0);
    for (out, bytes) in decoded.iter_mut().zip(values.chunks_exact(2)) {
        let bits = bytes[0] as u16 | (bytes[1] as u16) << 8;
        *out = f16_to_f32(bits);
    }
}

/// Decodes a little-endian binary32 run.
#[inline(always)]
fn decode_f32s(bytes: &[u8], decoded: &mut Vec<f32>) {
    assert_eq!(bytes.len() % std::mem::size_of::<f32>(), 0);
    decoded.resize(bytes.len() / std::mem::size_of::<f32>(), 0.0);
    for (out, bytes) in decoded.iter_mut().zip(bytes.chunks_exact(4)) {
        let bits = bytes[0] as u32
            | (bytes[1] as u32) << 8
            | (bytes[2] as u32) << 16
            | (bytes[3] as u32) << 24;
        *out = f32::from_bits(bits);
    }
}

/// Computes corrected-reconstruction uncertainty.
#[inline(always)]
fn fill_gamma_sigmas(
    sigmas: &mut [f32],
    residual_norms_squared: &[f32],
    gammas: &[f32],
    error_ratios: &[f32],
    sign_query_error_terms: &[f32],
    score_query_norm_squared: f32,
    dimension: usize,
    metric: Metric,
) {
    debug_assert_eq!(sigmas.len(), residual_norms_squared.len());
    debug_assert_eq!(sigmas.len(), gammas.len());
    debug_assert_eq!(sigmas.len(), error_ratios.len());
    debug_assert_eq!(sigmas.len(), sign_query_error_terms.len());
    for ((((sigma, &residual_norm_squared), &gamma), &error_ratio), &sign_query_error_term) in
        sigmas
            .iter_mut()
            .zip(residual_norms_squared)
            .zip(gammas)
            .zip(error_ratios)
            .zip(sign_query_error_terms)
    {
        *sigma = quantized_model_sigma(
            metric,
            dimension,
            residual_norm_squared,
            error_ratio,
            gamma,
            score_query_norm_squared,
            sign_query_error_term,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn combine_initial_decoded(
    metric: Metric,
    dimension: usize,
    kernel_scores: &mut [f32],
    bases: &mut [f32],
    estimates: &mut [f32],
    sigmas: &mut [f32],
    residual_norms_squared: &mut [f32],
    sign_query_error_terms: &mut [f32],
    decoded_scales: &[f32],
    decoded_gammas: &[f32],
    decoded_error_ratios: &[f32],
    decoded_constants: &[f32],
    decoded_residual_norms: &[f32],
    cluster_score: f32,
    score_query_norm_squared: f32,
    sign_query_error_squared: f32,
) {
    debug_assert_eq!(kernel_scores.len(), decoded_scales.len());
    debug_assert_eq!(kernel_scores.len(), decoded_gammas.len());
    debug_assert_eq!(kernel_scores.len(), decoded_error_ratios.len());
    debug_assert_eq!(kernel_scores.len(), decoded_residual_norms.len());
    debug_assert_eq!(bases.len(), decoded_scales.len());
    debug_assert_eq!(estimates.len(), decoded_scales.len());
    debug_assert_eq!(sigmas.len(), decoded_scales.len());
    debug_assert_eq!(residual_norms_squared.len(), decoded_scales.len());
    debug_assert_eq!(sign_query_error_terms.len(), decoded_scales.len());
    match metric {
        Metric::L2 => {
            debug_assert_eq!(decoded_constants.len(), decoded_scales.len());
            for (
                (
                    (
                        (
                            (((raw_prefix, base), estimate), residual_norm_squared),
                            sign_query_error_term,
                        ),
                        &scale,
                    ),
                    &gamma,
                ),
                (&constant, &residual_norm_sq),
            ) in kernel_scores
                .iter_mut()
                .zip(bases.iter_mut())
                .zip(estimates.iter_mut())
                .zip(residual_norms_squared.iter_mut())
                .zip(sign_query_error_terms.iter_mut())
                .zip(decoded_scales)
                .zip(decoded_gammas)
                .zip(decoded_constants.iter().zip(decoded_residual_norms))
            {
                *raw_prefix = initial_l2_raw_prefix(*raw_prefix, scale, constant);
                *base = cluster_score - residual_norm_sq;
                *estimate = corrected_quantized_estimate(metric, gamma, *raw_prefix, *base);
                *residual_norm_squared = residual_norm_sq;
                *sign_query_error_term = scale * scale * sign_query_error_squared;
            }
        }
        Metric::Dot | Metric::Cosine => {
            for (
                (
                    (
                        (
                            (((raw_prefix, base), estimate), residual_norm_squared),
                            sign_query_error_term,
                        ),
                        &scale,
                    ),
                    &gamma,
                ),
                &residual_norm_sq,
            ) in kernel_scores
                .iter_mut()
                .zip(bases.iter_mut())
                .zip(estimates.iter_mut())
                .zip(residual_norms_squared.iter_mut())
                .zip(sign_query_error_terms.iter_mut())
                .zip(decoded_scales)
                .zip(decoded_gammas)
                .zip(decoded_residual_norms)
            {
                *raw_prefix = initial_dot_raw_prefix(*raw_prefix, scale);
                *base = cluster_score;
                *estimate = corrected_quantized_estimate(metric, gamma, *raw_prefix, *base);
                *residual_norm_squared = residual_norm_sq;
                *sign_query_error_term = scale * scale * sign_query_error_squared;
            }
        }
    }
    fill_gamma_sigmas(
        sigmas,
        residual_norms_squared,
        decoded_gammas,
        decoded_error_ratios,
        sign_query_error_terms,
        score_query_norm_squared,
        dimension,
        metric,
    );
}

/// Runs the complete layer-0 cluster scoring shape.
#[cfg(feature = "quantization-bench")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn quantization_bench_layer0_cosine_cluster(
    dimension: usize,
    prepared: &cascade::PreparedSplitQuery,
    spec: cascade::LayerSpec,
    codes: &[u8],
    code_stride: usize,
    scales: &[u8],
    gammas: &[u8],
    error_ratios: &[u8],
    residual_norms: &[u8],
    cluster_score: f32,
    score_query_norm_squared: f32,
    sign_query_error_squared: f32,
    kernel_scores: &mut Vec<f32>,
    decoded_scales: &mut Vec<f32>,
    decoded_gammas: &mut Vec<f32>,
    decoded_error_ratios: &mut Vec<f32>,
    decoded_residual_norms: &mut Vec<f32>,
    bases: &mut Vec<f32>,
    estimates: &mut Vec<f32>,
    sigmas: &mut Vec<f32>,
    residual_norms_squared: &mut Vec<f32>,
    sign_query_error_terms: &mut Vec<f32>,
) -> f32 {
    let rows = scales.len() / std::mem::size_of::<f32>();
    assert_eq!(codes.len(), rows * code_stride);
    assert_eq!(gammas.len(), rows * std::mem::size_of::<u16>());
    assert_eq!(error_ratios.len(), rows * std::mem::size_of::<u16>());
    assert_eq!(residual_norms.len(), rows * std::mem::size_of::<f32>());
    kernel_scores.resize(rows, 0.0);
    prepared.score_layer_batch_unscaled(0, codes, code_stride, spec, kernel_scores);
    decode_f32s(scales, decoded_scales);
    finish_quantization_bench_layer0_cosine_cluster(
        dimension,
        rows,
        gammas,
        error_ratios,
        residual_norms,
        cluster_score,
        score_query_norm_squared,
        sign_query_error_squared,
        kernel_scores,
        decoded_scales,
        decoded_gammas,
        decoded_error_ratios,
        decoded_residual_norms,
        bases,
        estimates,
        sigmas,
        residual_norms_squared,
        sign_query_error_terms,
    )
}

#[cfg(feature = "quantization-bench")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn finish_quantization_bench_layer0_cosine_cluster(
    dimension: usize,
    rows: usize,
    gammas: &[u8],
    error_ratios: &[u8],
    residual_norms: &[u8],
    cluster_score: f32,
    score_query_norm_squared: f32,
    sign_query_error_squared: f32,
    kernel_scores: &mut Vec<f32>,
    decoded_scales: &mut Vec<f32>,
    decoded_gammas: &mut Vec<f32>,
    decoded_error_ratios: &mut Vec<f32>,
    decoded_residual_norms: &mut Vec<f32>,
    bases: &mut Vec<f32>,
    estimates: &mut Vec<f32>,
    sigmas: &mut Vec<f32>,
    residual_norms_squared: &mut Vec<f32>,
    sign_query_error_terms: &mut Vec<f32>,
) -> f32 {
    decode_f16s(gammas, decoded_gammas);
    decode_f16s(error_ratios, decoded_error_ratios);
    decode_f32s(residual_norms, decoded_residual_norms);
    bases.resize(rows, 0.0);
    estimates.resize(rows, 0.0);
    sigmas.resize(rows, 0.0);
    residual_norms_squared.resize(rows, 0.0);
    sign_query_error_terms.resize(rows, 0.0);
    combine_initial_decoded(
        Metric::Cosine,
        dimension,
        kernel_scores,
        bases,
        estimates,
        sigmas,
        residual_norms_squared,
        sign_query_error_terms,
        decoded_scales,
        decoded_gammas,
        decoded_error_ratios,
        &[],
        decoded_residual_norms,
        cluster_score,
        score_query_norm_squared,
        sign_query_error_squared,
    );

    kernel_scores[rows - 1] + estimates[rows - 1] + sigmas[rows - 1]
}

/// Runs layer-0 scoring with binary16 scale decoding.
#[cfg(feature = "quantization-bench")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn quantization_bench_layer0_cosine_cluster_f16_scales(
    dimension: usize,
    prepared: &cascade::PreparedSplitQuery,
    spec: cascade::LayerSpec,
    codes: &[u8],
    code_stride: usize,
    scales: &[u8],
    gammas: &[u8],
    error_ratios: &[u8],
    residual_norms: &[u8],
    cluster_score: f32,
    score_query_norm_squared: f32,
    sign_query_error_squared: f32,
    kernel_scores: &mut Vec<f32>,
    decoded_scales: &mut Vec<f32>,
    decoded_gammas: &mut Vec<f32>,
    decoded_error_ratios: &mut Vec<f32>,
    decoded_residual_norms: &mut Vec<f32>,
    bases: &mut Vec<f32>,
    estimates: &mut Vec<f32>,
    sigmas: &mut Vec<f32>,
    residual_norms_squared: &mut Vec<f32>,
    sign_query_error_terms: &mut Vec<f32>,
) -> f32 {
    let rows = scales.len() / std::mem::size_of::<u16>();
    assert_eq!(codes.len(), rows * code_stride);
    assert_eq!(gammas.len(), rows * std::mem::size_of::<u16>());
    assert_eq!(error_ratios.len(), rows * std::mem::size_of::<u16>());
    assert_eq!(residual_norms.len(), rows * std::mem::size_of::<f32>());
    kernel_scores.resize(rows, 0.0);
    prepared.score_layer_batch_unscaled(0, codes, code_stride, spec, kernel_scores);
    decode_f16s(scales, decoded_scales);
    finish_quantization_bench_layer0_cosine_cluster(
        dimension,
        rows,
        gammas,
        error_ratios,
        residual_norms,
        cluster_score,
        score_query_norm_squared,
        sign_query_error_squared,
        kernel_scores,
        decoded_scales,
        decoded_gammas,
        decoded_error_ratios,
        decoded_residual_norms,
        bases,
        estimates,
        sigmas,
        residual_norms_squared,
        sign_query_error_terms,
    )
}

const COSINE_REFINEMENT_BATCH_ROWS: usize = 2_048;

fn cosine_refinement_batches(row_count: usize) -> impl Iterator<Item = Range<usize>> {
    (0..row_count)
        .step_by(COSINE_REFINEMENT_BATCH_ROWS)
        .map(move |start| start..(start + COSINE_REFINEMENT_BATCH_ROWS).min(row_count))
}

#[inline(always)]
fn combine_refinement_decoded(
    metric: Metric,
    dimension: usize,
    candidates: &mut QuantizedCandidates,
    candidate_range: Range<usize>,
    kernel_scores: &[f32],
    decoded_scales: &[f32],
    decoded_gammas: &[f32],
    decoded_error_ratios: &[f32],
    decoded_constants: &[f32],
    score_query_norm_squared: f32,
    sign_query_error_squared: f32,
) {
    let rows = candidate_range.len();
    debug_assert_eq!(kernel_scores.len(), rows);
    debug_assert_eq!(decoded_scales.len(), rows);
    debug_assert_eq!(decoded_gammas.len(), rows);
    debug_assert_eq!(decoded_error_ratios.len(), rows);
    let bases = &candidates.bases[candidate_range.clone()];
    let raw_prefixes = &mut candidates.raw_prefixes[candidate_range.clone()];
    let estimates = &mut candidates.estimates[candidate_range.clone()];
    let residual_norms_squared = &candidates.residual_norm_squared[candidate_range.clone()];
    let current_gammas = &mut candidates.gammas[candidate_range.clone()];
    let sign_query_error_terms = &mut candidates.sign_query_error_terms[candidate_range.clone()];
    match metric {
        Metric::L2 => {
            debug_assert_eq!(decoded_constants.len(), rows);
            for (
                (
                    (
                        ((((raw_prefix, estimate), current_gamma), sign_query_error_term), &base),
                        &kernel_score,
                    ),
                    (&scale, &gamma),
                ),
                &constant,
            ) in raw_prefixes
                .iter_mut()
                .zip(estimates.iter_mut())
                .zip(current_gammas.iter_mut())
                .zip(sign_query_error_terms.iter_mut())
                .zip(bases)
                .zip(kernel_scores)
                .zip(decoded_scales.iter().zip(decoded_gammas))
                .zip(decoded_constants)
            {
                *current_gamma = gamma;
                *sign_query_error_term += scale * scale * sign_query_error_squared;
                *raw_prefix = refine_l2_raw_prefix(*raw_prefix, kernel_score, scale, constant);
                *estimate = corrected_quantized_estimate(metric, gamma, *raw_prefix, base);
            }
        }
        Metric::Dot | Metric::Cosine => {
            for (
                (
                    ((((raw_prefix, estimate), current_gamma), sign_query_error_term), &base),
                    &kernel_score,
                ),
                (&scale, &gamma),
            ) in raw_prefixes
                .iter_mut()
                .zip(estimates.iter_mut())
                .zip(current_gammas.iter_mut())
                .zip(sign_query_error_terms.iter_mut())
                .zip(bases)
                .zip(kernel_scores)
                .zip(decoded_scales.iter().zip(decoded_gammas))
            {
                *current_gamma = gamma;
                *sign_query_error_term += scale * scale * sign_query_error_squared;
                *raw_prefix = refine_dot_raw_prefix(*raw_prefix, kernel_score, scale);
                *estimate = corrected_quantized_estimate(metric, gamma, *raw_prefix, base);
            }
        }
    }
    fill_gamma_sigmas(
        &mut candidates.sigmas[candidate_range],
        residual_norms_squared,
        current_gammas,
        decoded_error_ratios,
        sign_query_error_terms,
        score_query_norm_squared,
        dimension,
        metric,
    );
}

/// Reads and scores one selected layer range.
#[inline(always)]
fn score_layer(
    query: &QuantizedQueryCtx,
    layer_idx: usize,
    metric: Metric,
    layer: &QuantizedLayerReader,
    rows: Range<usize>,
    selection: &Selection<'_>,
    kernel_scores: &mut Vec<f32>,
    decoded_scales: &mut Vec<f32>,
    decoded_gammas: &mut Vec<f32>,
    decoded_error_ratios: &mut Vec<f32>,
    decoded_constants: &mut Vec<f32>,
    read_ranges: &mut Vec<Range<usize>>,
    block_scratch: &mut Vec<(usize, usize)>,
    selected_rows: &mut Vec<usize>,
    row_offsets: &mut Vec<usize>,
) -> crate::Result<usize> {
    let selected_count = selection.len(&rows);
    if selected_count == 0 {
        unreachable!("empty selections are skipped before scoring");
    }
    kernel_scores.resize(selected_count, 0.0);
    decoded_scales.resize(selected_count, 0.0);
    decoded_gammas.resize(selected_count, 0.0);
    decoded_error_ratios.resize(selected_count, 0.0);
    if metric == Metric::L2 {
        decoded_constants.resize(selected_count, 0.0);
    }

    if matches!(selection, Selection::All) {
        let batch = layer.read_batch(rows)?;
        query.score_layer_batch_unscaled(
            layer_idx,
            batch.codes(),
            batch.code_stride(),
            &mut kernel_scores[..selected_count],
        );
        decode_f32s(batch.scales(), decoded_scales);
        decode_f16s(batch.gammas(), decoded_gammas);
        decode_f16s(batch.error_ratios(), decoded_error_ratios);
        if metric == Metric::L2 {
            let constants = batch.constants().ok_or_else(|| {
                TantivyError::DataCorruption(DataCorruption::comment_only(
                    "quantized L2 field is missing a constants slot",
                ))
            })?;
            decode_f32s(constants, decoded_constants);
        }
        return Ok(selected_count);
    }

    let Selection::Rows(offsets) = selection else {
        unreachable!("empty selections are skipped before scoring");
    };
    debug_assert!(offsets.windows(2).all(|pair| pair[0] < pair[1]));
    debug_assert!(offsets.iter().all(|&offset| offset < rows.len()));
    selected_rows.clear();
    selected_rows.extend(offsets.iter().map(|&offset| rows.start + offset));

    layer.plan_code_reads(rows.clone(), selected_rows, read_ranges, block_scratch);
    let mut selected_start = 0usize;
    for read_range in read_ranges.iter().cloned() {
        let mut selected_end = selected_start;
        while selected_end < selected_count && selected_rows[selected_end] < read_range.end {
            debug_assert!(selected_rows[selected_end] >= read_range.start);
            selected_end += 1;
        }
        row_offsets.clear();
        row_offsets.extend(
            selected_rows[selected_start..selected_end]
                .iter()
                .map(|&row| row - read_range.start),
        );
        let codes = layer.read_codes(read_range)?;
        query.score_layer_batch_unscaled_indexed(
            layer_idx,
            codes.as_slice(),
            layer.code_stride(),
            row_offsets,
            &mut kernel_scores[selected_start..selected_end],
        );
        selected_start = selected_end;
    }
    debug_assert_eq!(selected_start, selected_count);

    layer.plan_sidecar_reads(rows.clone(), selected_rows, read_ranges, block_scratch);
    selected_start = 0;
    for read_range in read_ranges.iter().cloned() {
        let sidecar = layer.read_sidecar(read_range.clone())?;
        while selected_start < selected_count && selected_rows[selected_start] < read_range.end {
            let row = selected_rows[selected_start];
            debug_assert!(row >= read_range.start);
            let scale_offset = (row - read_range.start) * std::mem::size_of::<f32>();
            let scale_bits = sidecar.scales()[scale_offset] as u32
                | (sidecar.scales()[scale_offset + 1] as u32) << 8
                | (sidecar.scales()[scale_offset + 2] as u32) << 16
                | (sidecar.scales()[scale_offset + 3] as u32) << 24;
            let gamma_offset = (row - read_range.start) * std::mem::size_of::<u16>();
            let gamma_bits = sidecar.gammas()[gamma_offset] as u16
                | (sidecar.gammas()[gamma_offset + 1] as u16) << 8;
            let error_ratio_bits = sidecar.error_ratios()[gamma_offset] as u16
                | (sidecar.error_ratios()[gamma_offset + 1] as u16) << 8;
            decoded_scales[selected_start] = f32::from_bits(scale_bits);
            decoded_gammas[selected_start] = f16_to_f32(gamma_bits);
            decoded_error_ratios[selected_start] = f16_to_f32(error_ratio_bits);
            selected_start += 1;
        }
    }
    debug_assert_eq!(selected_start, selected_count);

    if metric == Metric::L2 {
        layer.plan_constant_reads(rows, selected_rows, read_ranges, block_scratch)?;
        selected_start = 0;
        for read_range in read_ranges.iter().cloned() {
            let constants = layer.read_constants(read_range.clone())?.ok_or_else(|| {
                TantivyError::DataCorruption(DataCorruption::comment_only(
                    "quantized L2 field is missing a constants slot",
                ))
            })?;
            while selected_start < selected_count && selected_rows[selected_start] < read_range.end
            {
                let row = selected_rows[selected_start];
                debug_assert!(row >= read_range.start);
                let offset = (row - read_range.start) * std::mem::size_of::<f32>();
                let bits = constants[offset] as u32
                    | (constants[offset + 1] as u32) << 8
                    | (constants[offset + 2] as u32) << 16
                    | (constants[offset + 3] as u32) << 24;
                decoded_constants[selected_start] = f32::from_bits(bits);
                selected_start += 1;
            }
        }
        debug_assert_eq!(selected_start, selected_count);
    }
    Ok(selected_count)
}

fn decode_selected_residual_norms(
    quantized: &QuantizedFieldReader,
    rows: Range<usize>,
    selection: &Selection<'_>,
    decoded: &mut Vec<f32>,
    read_ranges: &mut Vec<Range<usize>>,
    block_scratch: &mut Vec<(usize, usize)>,
    selected_rows: &mut Vec<usize>,
) -> crate::Result<()> {
    let selected_count = selection.len(&rows);
    decoded.resize(selected_count, 0.0);
    if matches!(selection, Selection::All) {
        let residual_norms = quantized.read_residual_norm_batch(rows)?;
        decode_f32s(residual_norms.as_bytes(), decoded);
        return Ok(());
    }

    let Selection::Rows(offsets) = selection else {
        unreachable!("empty selections are skipped before scoring");
    };
    selected_rows.clear();
    selected_rows.extend(offsets.iter().map(|&offset| rows.start + offset));
    quantized.plan_residual_norm_reads(rows, selected_rows, read_ranges, block_scratch);
    let mut selected_start = 0usize;
    for read_range in read_ranges.iter().cloned() {
        let residual_norms = quantized.read_residual_norms(read_range.clone())?;
        while selected_start < selected_count && selected_rows[selected_start] < read_range.end {
            let row = selected_rows[selected_start];
            debug_assert!(row >= read_range.start);
            let offset = (row - read_range.start) * std::mem::size_of::<f32>();
            decoded[selected_start] = f32::from_le_bytes(
                residual_norms[offset..offset + std::mem::size_of::<f32>()]
                    .try_into()
                    .unwrap(),
            );
            selected_start += 1;
        }
    }
    debug_assert_eq!(selected_start, selected_count);
    Ok(())
}

struct QuantizedScanCtx {
    candidates: QuantizedCandidates,
    boundary_scratch: Vec<QuantizedCandidate>,
    /// Query-residual norms by cluster.
    cluster_query_norms: Vec<f32>,
    /// Running top document estimates.
    bound_top: Vec<usize>,
    /// Cluster-local selection scratch.
    cluster_top: Vec<usize>,
    cluster_top_n: usize,
    cluster_start: Option<usize>,
    /// Top-k merge scratch.
    bound_merge: Vec<usize>,
    kth_scratch: Vec<usize>,
    work_spent: WorkUnits,
}

impl QuantizedScanCtx {
    fn new(max_doc: DocId, candidate_capacity: usize) -> Self {
        let distinct_capacity = candidate_capacity.min(max_doc as usize);
        Self {
            candidates: QuantizedCandidates::with_capacity(candidate_capacity),
            boundary_scratch: Vec::with_capacity(candidate_capacity),
            cluster_query_norms: Vec::new(),
            bound_top: Vec::new(),
            cluster_top: Vec::new(),
            cluster_top_n: 0,
            cluster_start: None,
            bound_merge: Vec::new(),
            kth_scratch: Vec::with_capacity(distinct_capacity),
            work_spent: WorkUnits::ZERO,
        }
    }

    fn begin_cluster(&mut self, top_n: usize) {
        debug_assert!(self.cluster_start.is_none());
        debug_assert!(self.cluster_top.is_empty());
        self.cluster_top.reserve(top_n);
        self.bound_top.reserve(top_n);
        self.bound_merge.reserve(top_n.saturating_mul(2));
        self.cluster_top_n = top_n;
        self.cluster_start = Some(self.candidates.len());
    }

    /// Appends an eligible row to scan columns.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn push(
        &mut self,
        row: usize,
        doc: DocId,
        base: f32,
        raw_prefix: f32,
        estimate: f32,
        sigma: f32,
        residual_norm_squared: f32,
        gamma: f32,
        sign_query_error_term: f32,
    ) {
        self.candidates.push(
            row,
            doc,
            base,
            raw_prefix,
            estimate,
            sigma,
            residual_norm_squared,
            gamma,
            sign_query_error_term,
        );
    }

    fn set_cluster_query_norm(&mut self, cluster: usize, query_norm: f32) {
        if self.cluster_query_norms.len() <= cluster {
            self.cluster_query_norms.resize(cluster + 1, f32::NAN);
        }
        self.cluster_query_norms[cluster] = query_norm;
    }

    fn cluster_query_norm(&self, cluster: usize) -> f32 {
        let query_norm = self.cluster_query_norms[cluster];
        debug_assert!(query_norm.is_finite());
        query_norm
    }

    /// Merges one cluster into the running admission top-k.
    fn finish_cluster_bound(&mut self) {
        let cluster_start = self
            .cluster_start
            .take()
            .expect("finish_cluster_bound requires begin_cluster");
        let top_n = std::mem::take(&mut self.cluster_top_n);
        if top_n == 0 || cluster_start >= self.candidates.len() {
            self.cluster_top.clear();
            return;
        }

        self.cluster_top.clear();
        for index in cluster_start..self.candidates.len() {
            if self.cluster_top.len() < top_n {
                self.cluster_top.push(index);
                if self.cluster_top.len() == top_n {
                    self.cluster_top
                        .sort_unstable_by(|&a, &b| candidate_order(&self.candidates, a, b));
                }
                continue;
            }
            let tracked_min = *self.cluster_top.last().unwrap();
            if !candidate_precedes(&self.candidates, index, tracked_min) {
                continue;
            }
            let insert_at = self
                .cluster_top
                .partition_point(|&kept| candidate_precedes(&self.candidates, kept, index));
            self.cluster_top.insert(insert_at, index);
            self.cluster_top.pop();
        }
        if self.cluster_top.len() < top_n {
            self.cluster_top
                .sort_unstable_by(|&a, &b| candidate_order(&self.candidates, a, b));
        }

        self.bound_merge.clear();
        self.bound_merge.extend_from_slice(&self.bound_top);
        for &index in &self.cluster_top {
            self.bound_merge.push(index);
        }
        self.cluster_top.clear();
        self.bound_merge
            .sort_unstable_by(|&a, &b| candidate_order(&self.candidates, a, b));
        self.bound_merge.truncate(top_n);
        std::mem::swap(&mut self.bound_top, &mut self.bound_merge);
    }

    fn running_pessimistic_kth(&self, top_n: usize, kappa: f32) -> Option<f32> {
        if top_n == 0 || self.bound_top.len() < top_n {
            return None;
        }
        let index = *self.bound_top.last().unwrap();
        Some(self.candidates.estimate(index) - kappa * self.candidates.sigmas[index])
    }

    /// The k-th estimate widened pessimistically by its σ.
    fn pessimistic_kth(&mut self, top_n: usize, kappa: f32) -> Option<f32> {
        debug_assert!(
            self.candidates
                .estimates
                .iter()
                .zip(&self.candidates.sigmas)
                .all(|(&estimate, &sigma)| estimate.is_finite() && sigma.is_finite()),
            "quantized boundary inputs must be finite"
        );
        if top_n == 0 || self.candidates.len() < top_n {
            return None;
        }
        self.kth_scratch.clear();
        self.kth_scratch.extend(0..self.candidates.len());
        let (_, selected, _) = self
            .kth_scratch
            .select_nth_unstable_by(top_n - 1, |&a, &b| candidate_order(&self.candidates, a, b));
        let index = *selected;
        Some(self.candidates.estimate(index) - kappa * self.candidates.sigmas[index])
    }

    fn band(&mut self, top_n: usize, kappa: f32) {
        let pessimistic_kth = self.pessimistic_kth(top_n, kappa);
        self.boundary_scratch.clear();
        for index in 0..self.candidates.len() {
            if pessimistic_kth.is_none_or(|kth| {
                self.candidates.estimate(index) + kappa * self.candidates.sigmas[index] >= kth
            }) {
                self.boundary_scratch
                    .push(self.candidates.materialize(index));
            }
        }
        self.boundary_scratch
            .sort_unstable_by_key(|candidate| candidate.row);
        self.candidates
            .replace_with_boundary_survivors(&self.boundary_scratch);
    }
}

#[cfg(test)]
fn candidate_docs(candidates: &QuantizedCandidates) -> Vec<DocId> {
    let mut docs = candidates.docs.clone();
    docs.sort_unstable();
    docs
}

fn select_cluster_rows<'a>(
    reader: &VectorIndexReader,
    rows: Range<usize>,
    eligibility: &BitSet,
    filter: &BitSet,
    filter_is_all: bool,
    alive: Option<&AliveBitSet>,
    offsets: &'a mut Vec<usize>,
    docs: &mut Vec<DocId>,
) -> (Selection<'a>, usize, usize, usize) {
    offsets.clear();
    docs.clear();
    let visited = rows.len();
    if filter_is_all && alive.is_none() {
        docs.extend(rows.map(|row| reader.doc_id_at(row)));
        return (Selection::All, visited, 0, 0);
    }

    let mut pruned_filter = 0usize;
    let mut pruned_dead = 0usize;
    for (offset, row) in rows.enumerate() {
        let doc = reader.doc_id_at(row);
        if !eligibility.contains(doc) {
            if !filter.contains(doc) {
                pruned_filter += 1;
            } else {
                debug_assert!(alive.is_some_and(|alive| !alive.is_alive(doc)));
                pruned_dead += 1;
            }
            continue;
        }
        offsets.push(offset);
        docs.push(doc);
    }
    if offsets.is_empty() {
        (Selection::None, visited, pruned_filter, pruned_dead)
    } else {
        (
            Selection::Rows(offsets),
            visited,
            pruned_filter,
            pruned_dead,
        )
    }
}

fn candidate_selection<'a>(
    candidate_rows: &[usize],
    available_rows: &Range<usize>,
    offsets: &'a mut Vec<usize>,
) -> Selection<'a> {
    debug_assert!(!candidate_rows.is_empty());
    debug_assert!(candidate_rows.windows(2).all(|pair| pair[0] < pair[1]));
    debug_assert!(candidate_rows
        .iter()
        .all(|row| available_rows.contains(row)));
    offsets.clear();
    offsets.extend(candidate_rows.iter().map(|&row| row - available_rows.start));
    Selection::Rows(offsets)
}

#[inline]
fn candidate_order(candidates: &QuantizedCandidates, a: usize, b: usize) -> std::cmp::Ordering {
    candidates
        .estimate(b)
        .total_cmp(&candidates.estimate(a))
        .then(candidates.rows[a].cmp(&candidates.rows[b]))
}

#[inline]
fn candidate_precedes(candidates: &QuantizedCandidates, a: usize, b: usize) -> bool {
    candidate_order(candidates, a, b).is_lt()
}

impl<T: VectorElement> VectorBackend<T> {
    #[allow(clippy::too_many_arguments)]
    fn quantized_top_n<K, CTail>(
        &self,
        index: &IvfIndex,
        query: &QuantizedQueryCtx,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        if top_n == 0 || segment_reader.max_doc() == 0 || index.num_clusters() == 0 {
            return Ok(Vec::new());
        }
        let max_doc = segment_reader.max_doc();
        let filter = build_filter_bitset(weight, segment_reader, max_doc)?;
        let alive = segment_reader.alive_bitset();
        let eligibility = alive.map(|alive| {
            let mut eligibility = filter.clone();
            eligibility.intersect_update(alive.bitset());
            eligibility
        });
        if filter.len() == 0 {
            return Ok(Vec::new());
        }
        let filter_is_all = filter.len() == max_doc as usize;
        let scan_levels = query.active_layers();
        let quantized = self
            .reader
            .quantization()
            .expect("quantized query requires quantized slots");
        let (work_budget, n_avg, x) = self
            .adaptive
            .resolved_work_budget(index.num_clusters(), index.num_docs())?;
        let pricing = UnitPricing {
            budget: WorkUnits::new(work_budget),
            open: WorkUnits::new(x),
            row: WorkUnits::new((1.0 - x) / n_avg),
        };
        let mut routing_ws = Workspace::new();
        let candidate_capacity =
            ((pricing.budget.get() / pricing.row.get()).ceil() as usize).min(index.num_rows());
        let mut scan = QuantizedScanCtx::new(max_doc, candidate_capacity);
        let mut postings_row = 0usize;
        let mut postings_skipped = 0usize;
        let bounds = index.bounds();
        let metric = query.index.config.metric;
        let q_norm = norm_squared_wide(query.query()).sqrt() as f32;
        let mut bounds_skips = 0u32;
        let mut armed_probe = None;
        let mut cluster_docs = Vec::new();
        let mut selection_offsets = Vec::new();
        let mut kernel_scores = Vec::new();
        let mut decoded_scales = Vec::new();
        let mut decoded_gammas = Vec::new();
        let mut decoded_error_ratios = Vec::new();
        let mut decoded_constants = Vec::new();
        let mut decoded_residual_norms = Vec::new();
        let mut base_scores = Vec::new();
        let mut estimate_scores = Vec::new();
        let mut sigma_scores = Vec::new();
        let mut residual_norm_squared_scores = Vec::new();
        let mut sign_query_error_terms = Vec::new();
        let mut selected_rows = Vec::new();
        let mut indexed_row_offsets = Vec::new();
        let mut survivor_read_ranges = Vec::new();
        let mut survivor_block_scratch = Vec::new();
        let mut ranked = index.rank_clusters(&mut routing_ws, query.query());

        loop {
            let next = ranked.next();
            let Some(Candidate { sim, node }) = next else {
                break;
            };
            if scan.work_spent >= pricing.budget {
                stats.termination = ProbeTermination::Ceiling;
                break;
            }
            let cluster = node as usize;
            let query_bound = scan
                .running_pessimistic_kth(top_n, QUANTIZED_BOUNDARY_KAPPA)
                .map_or(QueryBound::Filling, |score| QueryBound::Armed {
                    t: to_bound_space(metric, score),
                });
            if armed_probe.is_none() && matches!(query_bound, QueryBound::Armed { .. }) {
                armed_probe = Some((postings_row + postings_skipped).saturating_sub(1) as u32);
            }
            let verdict = bounds_verdict(query_bound, || {
                let QueryBound::Armed { t } = query_bound else {
                    return f32::INFINITY;
                };
                let r = bounds.ball_r(cluster);
                match metric {
                    Metric::L2 | Metric::Cosine => {
                        margin_ball_ball(t, r, to_bound_space(metric, sim.score()))
                    }
                    Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t),
                }
            });
            if verdict == Verdict::Skip {
                scan.work_spent += pricing.open;
                bounds_skips += 1;
                continue;
            }
            scan.work_spent += pricing.open;
            let rows = index.cluster_range(cluster);
            let (selection, visited, pruned_filter, pruned_dead) = select_cluster_rows(
                &self.reader,
                rows.clone(),
                eligibility.as_ref().unwrap_or(&filter),
                &filter,
                filter_is_all,
                alive,
                &mut selection_offsets,
                &mut cluster_docs,
            );
            stats.vectors_visited += visited;
            stats.pruned_filter += pruned_filter;
            stats.pruned_dead += pruned_dead;
            let selected_count = selection.len(&rows);
            if selected_count == 0 {
                postings_skipped += 1;
                continue;
            }

            let score_query_norm = query.score_query_norm(sim.score());
            let layer = &quantized.layers()[0];
            score_layer(
                query,
                0,
                metric,
                layer,
                rows.clone(),
                &selection,
                &mut kernel_scores,
                &mut decoded_scales,
                &mut decoded_gammas,
                &mut decoded_error_ratios,
                &mut decoded_constants,
                &mut survivor_read_ranges,
                &mut survivor_block_scratch,
                &mut selected_rows,
                &mut indexed_row_offsets,
            )?;
            decode_selected_residual_norms(
                quantized,
                rows.clone(),
                &selection,
                &mut decoded_residual_norms,
                &mut survivor_read_ranges,
                &mut survivor_block_scratch,
                &mut selected_rows,
            )?;
            base_scores.resize(selected_count, 0.0);
            estimate_scores.resize(selected_count, 0.0);
            sigma_scores.resize(selected_count, 0.0);
            residual_norm_squared_scores.resize(selected_count, 0.0);
            sign_query_error_terms.resize(selected_count, 0.0);
            let cluster_score = sim.score();
            combine_initial_decoded(
                metric,
                query.index.config.dim,
                &mut kernel_scores,
                &mut base_scores,
                &mut estimate_scores,
                &mut sigma_scores,
                &mut residual_norm_squared_scores,
                &mut sign_query_error_terms,
                &decoded_scales,
                &decoded_gammas,
                &decoded_error_ratios,
                &decoded_constants,
                &decoded_residual_norms,
                cluster_score,
                score_query_norm * score_query_norm,
                query.query_error_squared(0) as f32,
            );

            scan.set_cluster_query_norm(cluster, score_query_norm);
            scan.begin_cluster(top_n);
            scan.candidates.append_selected(
                rows.clone(),
                &selection,
                &cluster_docs,
                &base_scores[..selected_count],
                &kernel_scores[..selected_count],
                &estimate_scores[..selected_count],
                &sigma_scores[..selected_count],
                &residual_norm_squared_scores[..selected_count],
                &decoded_gammas[..selected_count],
                &sign_query_error_terms[..selected_count],
            );
            scan.finish_cluster_bound();
            scan.work_spent += pricing.row * selected_count as f64;
            postings_row += 1;
        }
        stats.record_routing(ranked.metrics());
        stats.postings_row += postings_row;
        stats.postings_skipped += postings_skipped;
        stats.candidates_scored += scan.candidates.len();
        stats.bounds_skips += bounds_skips;
        stats.record_bound_armed(armed_probe);
        stats.work_charged += scan.work_spent.to_f32();
        #[cfg(test)]
        {
            stats.quantized_trace.scored_docs = candidate_docs(&scan.candidates);
        }

        scan.band(top_n, QUANTIZED_BOUNDARY_KAPPA);
        #[cfg(test)]
        stats
            .quantized_trace
            .boundary_docs
            .push(candidate_docs(&scan.candidates));
        for layer_idx in 1..scan_levels {
            let layer = &quantized.layers()[layer_idx];
            if metric == Metric::Cosine {
                let query_norm = query.score_query_norm(0.0);
                for candidate_range in cosine_refinement_batches(scan.candidates.len()) {
                    let candidate_start = candidate_range.start;
                    let candidate_end = candidate_range.end;
                    let first_row = scan.candidates.rows[candidate_start];
                    let last_row = scan.candidates.rows[candidate_end - 1];
                    let available_rows = first_row..last_row + 1;
                    let selection = candidate_selection(
                        &scan.candidates.rows[candidate_range.clone()],
                        &available_rows,
                        &mut selection_offsets,
                    );
                    let rows = score_layer(
                        query,
                        layer_idx,
                        metric,
                        layer,
                        available_rows,
                        &selection,
                        &mut kernel_scores,
                        &mut decoded_scales,
                        &mut decoded_gammas,
                        &mut decoded_error_ratios,
                        &mut decoded_constants,
                        &mut survivor_read_ranges,
                        &mut survivor_block_scratch,
                        &mut selected_rows,
                        &mut indexed_row_offsets,
                    )?;
                    let decoded_constants = if metric == Metric::L2 {
                        &decoded_constants[..rows]
                    } else {
                        &[]
                    };
                    let sign_query_error_squared = if query.index.specs[layer_idx].bits == 1 {
                        query.query_error_squared(layer_idx) as f32
                    } else {
                        0.0
                    };
                    combine_refinement_decoded(
                        metric,
                        query.index.config.dim,
                        &mut scan.candidates,
                        candidate_range,
                        &kernel_scores[..rows],
                        &decoded_scales[..rows],
                        &decoded_gammas[..rows],
                        &decoded_error_ratios[..rows],
                        decoded_constants,
                        query_norm * query_norm,
                        sign_query_error_squared,
                    );
                }
            } else {
                let mut candidate_start = 0;
                let mut cluster = 0;
                while candidate_start < scan.candidates.len() {
                    let first_row = scan.candidates.rows[candidate_start];
                    while cluster < index.num_clusters()
                        && index.cluster_range(cluster).end <= first_row
                    {
                        cluster += 1;
                    }
                    if cluster == index.num_clusters() {
                        return Err(TantivyError::DataCorruption(DataCorruption::comment_only(
                            format!(
                                "quantized survivor row {first_row} is outside IVF cluster ranges"
                            ),
                        )));
                    }
                    let cluster_rows = index.cluster_range(cluster);
                    if first_row < cluster_rows.start {
                        return Err(TantivyError::DataCorruption(DataCorruption::comment_only(
                            format!(
                                "quantized survivor row {first_row} precedes cluster {cluster} \
                                 range {cluster_rows:?}"
                            ),
                        )));
                    }
                    let mut candidate_end = candidate_start + 1;
                    while candidate_end < scan.candidates.len()
                        && scan.candidates.rows[candidate_end] < cluster_rows.end
                    {
                        candidate_end += 1;
                    }
                    let query_norm = scan.cluster_query_norm(cluster);
                    let candidate_range = candidate_start..candidate_end;
                    let selection = candidate_selection(
                        &scan.candidates.rows[candidate_range.clone()],
                        &cluster_rows,
                        &mut selection_offsets,
                    );
                    let rows = score_layer(
                        query,
                        layer_idx,
                        metric,
                        layer,
                        cluster_rows,
                        &selection,
                        &mut kernel_scores,
                        &mut decoded_scales,
                        &mut decoded_gammas,
                        &mut decoded_error_ratios,
                        &mut decoded_constants,
                        &mut survivor_read_ranges,
                        &mut survivor_block_scratch,
                        &mut selected_rows,
                        &mut indexed_row_offsets,
                    )?;
                    let decoded_constants = if metric == Metric::L2 {
                        &decoded_constants[..rows]
                    } else {
                        &[]
                    };
                    let sign_query_error_squared = if query.index.specs[layer_idx].bits == 1 {
                        query.query_error_squared(layer_idx) as f32
                    } else {
                        0.0
                    };
                    combine_refinement_decoded(
                        metric,
                        query.index.config.dim,
                        &mut scan.candidates,
                        candidate_range,
                        &kernel_scores[..rows],
                        &decoded_scales[..rows],
                        &decoded_gammas[..rows],
                        &decoded_error_ratios[..rows],
                        decoded_constants,
                        query_norm * query_norm,
                        sign_query_error_squared,
                    );
                    candidate_start = candidate_end;
                }
            }
            scan.band(top_n, QUANTIZED_BOUNDARY_KAPPA);
            #[cfg(test)]
            stats
                .quantized_trace
                .boundary_docs
                .push(candidate_docs(&scan.candidates));
        }

        let mut rerank = Vec::with_capacity(scan.candidates.len());
        rerank.extend(
            scan.candidates
                .rows
                .iter()
                .copied()
                .zip(scan.candidates.docs.iter().copied()),
        );
        rerank.sort_unstable_by_key(|&(row, _)| row);
        #[cfg(test)]
        {
            stats.quantized_trace.rerank_docs = rerank.iter().map(|&(_, doc)| doc).collect();
            stats.quantized_trace.rerank_docs.sort_unstable();
        }
        let rerank_rows = rerank.iter().map(|&(row, _)| row).collect::<Vec<_>>();
        let rerank_batch = self.reader.read_vector_rows_planned(
            &rerank_rows,
            &mut survivor_read_ranges,
            &mut survivor_block_scratch,
        )?;
        let mut topn = TopNComputer::with_comparator(top_n, (NaturalComparator, tie_comparator));
        for ((row, doc), (batch_row, bytes)) in rerank.into_iter().zip(rerank_batch.iter()) {
            debug_assert_eq!(row, batch_row);
            let score = self.query.score_doc_bytes(bytes);
            if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                topn.push_unordered(key, doc);
            }
            stats.exact_rows_read += 1;
        }
        let segment_ord = self.segment_ord;
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| {
                (
                    candidate.sort_key,
                    DocAddress::new(segment_ord, candidate.doc),
                )
            })
            .collect();
        Ok(hits)
    }

    /// Returns the top matches from an IVF probe.
    #[allow(clippy::too_many_arguments)]
    fn approximate_top_n<K, CTail>(
        &self,
        index: &IvfIndex,
        weight: &dyn Weight,
        segment_reader: &SegmentReader,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHits<K>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        if top_n == 0 {
            return Ok(Vec::new());
        }
        let max_doc = segment_reader.max_doc();
        if max_doc == 0 {
            return Ok(Vec::new());
        }

        let filter = build_filter_bitset(weight, segment_reader, max_doc)?;
        if filter.len() == 0 {
            return Ok(Vec::new());
        }
        let alive = segment_reader.alive_bitset();

        let num_centroids = index.num_clusters();
        if num_centroids == 0 {
            return Ok(Vec::new());
        }
        let (work_budget, n_avg, x) = self
            .adaptive
            .resolved_work_budget(num_centroids, index.num_docs())?;
        debug_assert!(n_avg > 0.0);
        let pricing = UnitPricing {
            budget: WorkUnits::new(work_budget),
            open: WorkUnits::new(x),
            row: WorkUnits::new((1.0 - x) / n_avg),
        };

        let query_f32: Vec<f32> = self.query.query().iter().map(|e| e.to_f32()).collect();
        let mut routing_ws = Workspace::new();
        let mut ranked = index.rank_clusters(&mut routing_ws, &query_f32);
        let topn = self.scan_clusters(
            index,
            &mut ranked,
            pricing,
            &filter,
            alive,
            top_n,
            tie_break,
            tie_comparator,
            &query_f32,
            stats,
        )?;

        stats.record_routing(ranked.metrics());

        let segment_ord = self.segment_ord;
        let hits = topn
            .into_sorted_vec()
            .into_iter()
            .map(|cd| (cd.sort_key, DocAddress::new(segment_ord, cd.doc)))
            .collect();
        Ok(hits)
    }

    /// Probes ranked clusters under bounds and work-budget gates.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn scan_clusters<K, CTail>(
        &self,
        index: &IvfIndex,
        ranked: &mut impl Iterator<Item = Candidate>,
        pricing: UnitPricing,
        filter: &BitSet,
        alive: Option<&AliveBitSet>,
        top_n: usize,
        tie_break: &mut K,
        tie_comparator: CTail,
        routing_query: &[f32],
        stats: &mut ProbeStats,
    ) -> crate::Result<TieBreakHeap<K, CTail>>
    where
        K: SegmentSortKeyComputer,
        CTail: Comparator<K::SegmentSortKey>,
    {
        let mut topn = TopNComputer::with_comparator(top_n, (NaturalComparator, tie_comparator));
        let mut candidates = 0usize;
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut postings_row = 0usize;
        let mut postings_skipped = 0usize;
        let mut bounds_skips = 0u32;
        let mut termination = ProbeTermination::Exhausted;
        let metric = self.query.metric();
        let mut bound_tracker = QueryBoundTracker::new();
        let q_norm = norm_squared_wide(self.query.query()).sqrt() as f32;
        let bounds = index.bounds();
        let mut survivors: Vec<Survivor> = Vec::new();
        let mut work_spent = WorkUnits::ZERO;
        let work_budget = pricing.budget;

        loop {
            let next = ranked.next();
            let Some(Candidate { sim, node: cluster }) = next else {
                break;
            };
            if work_spent >= work_budget {
                termination = ProbeTermination::Ceiling;
                break;
            }
            let cluster = cluster as usize;

            let qb = bound_tracker.bound();
            let verdict = bounds_verdict(qb, || {
                let QueryBound::Armed { t } = qb else {
                    return f32::INFINITY;
                };
                #[cfg(debug_assertions)]
                {
                    let stride = self.reader.options().bytes_per_vector();
                    let centroid_bytes = index.centroid_bytes().expect("readable centroid rows");
                    let exact = metric.similarity_bytes::<f32>(
                        routing_query,
                        &centroid_bytes[cluster * stride..(cluster + 1) * stride],
                    );
                    debug_assert_eq!(
                        sim, exact,
                        "routing stream key must be the exact centroid similarity"
                    );
                }
                let r = bounds.ball_r(cluster);
                match metric {
                    Metric::L2 | Metric::Cosine => {
                        margin_ball_ball(t, r, to_bound_space(metric, sim.score()))
                    }
                    Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t),
                }
            });
            if let Verdict::Skip = verdict {
                work_spent += pricing.open;
                bounds_skips += 1;
                continue;
            }

            work_spent += pricing.open;

            let rows = index.cluster_range(cluster);

            let (v, pf, pd, scored_rows) =
                self.collect_cluster_survivors(rows, filter, alive, &mut survivors);
            visited += v;
            pruned_filter += pf;
            pruned_dead += pd;

            work_spent += pricing.row * scored_rows as f64;

            if survivors.is_empty() {
                postings_skipped += 1;
            } else {
                postings_row += 1;
                #[cfg(test)]
                stats
                    .quantized_trace
                    .scored_docs
                    .extend(survivors.iter().map(|survivor| survivor.doc));
                for &Survivor { row, doc } in &survivors {
                    let vbytes = self.reader.vector_bytes_for_row(row)?;
                    let score = self.query.score_doc_bytes(&vbytes);
                    if let Some(key) = tie_break_key(&topn, tie_break, score, doc) {
                        topn.push_unordered(key, doc);
                    }
                }
            }
            candidates += survivors.len();

            let probe_idx = (postings_row + postings_skipped - 1) as u32;
            let peek = HeapPeek::from_kth(topn.kth_best().map(|(score, _tie)| score));
            bound_tracker.observe(metric, peek, probe_idx);
        }
        debug_assert!(
            bound_tracker.armed_at_probe().is_some()
                == matches!(bound_tracker.bound(), QueryBound::Armed { .. })
        );

        stats.vectors_visited += visited;
        stats.pruned_filter += pruned_filter;
        stats.pruned_dead += pruned_dead;
        stats.postings_row += postings_row;
        stats.postings_skipped += postings_skipped;
        stats.candidates_scored += candidates;
        stats.bounds_skips += bounds_skips;
        stats.record_bound_armed(bound_tracker.armed_at_probe());
        stats.termination = termination;
        stats.work_charged += work_spent.to_f32();
        #[cfg(test)]
        {
            stats.quantized_trace.scored_docs.sort_unstable();
        }

        Ok(topn)
    }

    /// Collects scoreable rows from one cluster into `survivors`.
    #[inline(never)]
    fn collect_cluster_survivors(
        &self,
        rows: Range<usize>,
        filter: &BitSet,
        alive: Option<&AliveBitSet>,
        survivors: &mut Vec<Survivor>,
    ) -> (usize, usize, usize, usize) {
        survivors.clear();
        let mut visited = 0usize;
        let mut pruned_filter = 0usize;
        let mut pruned_dead = 0usize;
        let mut scored_rows = 0usize;
        for row in rows {
            let doc = self.reader.doc_id_at(row);
            visited += 1;
            if !filter.contains(doc) {
                pruned_filter += 1;
                continue;
            }
            if let Some(bs) = alive {
                if !bs.is_alive(doc) {
                    pruned_dead += 1;
                    continue;
                }
            }
            survivors.push(Survivor { row, doc });
            scored_rows += 1;
        }
        (visited, pruned_filter, pruned_dead, scored_rows)
    }
}

/// Materializes a filter doc set as a dense eligibility bitset.
#[inline(never)]
fn build_filter_bitset(
    weight: &dyn Weight,
    segment_reader: &SegmentReader,
    max_doc: DocId,
) -> crate::Result<BitSet> {
    let mut filter = BitSet::with_max_value(max_doc);
    weight.for_each_no_score(segment_reader, &mut |docs| {
        for &doc in docs {
            filter.insert(doc);
        }
    })?;
    Ok(filter)
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;
    use crate::collector::TopDocs;
    use crate::index::IndexSettings;
    use crate::indexer::NoMergePolicy;
    use crate::query::{
        AllQuery, BitSetDocSet, ConstScorer, EnableScoring, Explanation, Query, Scorer, TermQuery,
    };
    use crate::schema::{IndexRecordOption, Schema, Term, STORED, STRING};
    use crate::vector::tests::{exhaustive_params, TestVectorIndex};
    use crate::vector::{
        IvfCentroids, IvfClusterer, IvfMatrix, IvfTrainingVectors, IvfVectors, VectorClusterStats,
        VectorDType, VectorInfo, VectorOptions, VectorStorageFormat,
    };
    use crate::{Index, IndexWriter, TantivyDocument};

    const FIXTURE_NUM_DOCS: usize = 100;
    const DEFAULT_NUM_CENTROIDS: usize = 9;

    fn push_test_candidate(
        scan: &mut QuantizedScanCtx,
        row: usize,
        doc: DocId,
        estimate: f32,
        sigma: f32,
    ) {
        scan.push(row, doc, 0.0, estimate, estimate, sigma, 1.0, 1.0, 0.0);
    }

    fn search(
        index: &Index,
        field: Field,
        filter: &dyn Query,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let collector = TopDocs::with_limit(k)
            .order_by_similarity(field, query)
            .with_adaptive_params(params);
        Ok(index
            .reader()?
            .searcher()
            .search(filter, &collector)?
            .results)
    }

    fn run_top_n(
        index: &Index,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let weight = AllQuery.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let quantized_queries = QuantizedQueryCache::default();
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            &quantized_queries,
            params,
            usize::MAX,
        )?;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_some(),
            "expected IVF storage"
        );
        backend.top_n(weight.as_ref(), segment_reader, k)
    }

    struct InlineClusterer {
        centroids: Vec<[f32; 2]>,
    }

    impl IvfClusterer for InlineClusterer {
        fn training_sample_ratio(&self) -> f32 {
            1.0
        }
        fn train(
            &self,
            options: &VectorOptions,
            _vectors: IvfTrainingVectors,
        ) -> crate::Result<IvfCentroids> {
            assert_eq!(options.dim(), 2);
            let num_centroids = self.centroids.len();
            Ok(IvfCentroids::F32(IvfMatrix {
                values: self
                    .centroids
                    .iter()
                    .flat_map(|c| c.iter().copied())
                    .collect(),
                rows: num_centroids,
                dims: 2,
            }))
        }
        fn assign(
            &self,
            options: &VectorOptions,
            vectors: IvfVectors<'_>,
            centroids: &IvfCentroids,
        ) -> crate::Result<Vec<u32>> {
            assert_eq!(options.dim(), 2);
            let IvfVectors::F32(vectors) = vectors;
            let IvfCentroids::F32(centroids) = centroids;
            Ok(vectors
                .matrix
                .values
                .chunks_exact(2)
                .map(|v| {
                    let mut best = 0u32;
                    let mut best_d2 = f32::INFINITY;
                    for (i, c) in centroids.values.chunks_exact(2).enumerate() {
                        let dx = v[0] - c[0];
                        let dy = v[1] - c[1];
                        let d2 = dx * dx + dy * dy;
                        if d2 < best_d2 {
                            best = i as u32;
                            best_d2 = d2;
                        }
                    }
                    best
                })
                .collect())
        }
    }

    fn build_inline_ivf(
        metric: Metric,
        centroids: &[[f32; 2]],
        docs: &[(&str, [f32; 2])],
    ) -> crate::Result<(Index, Field, Field)> {
        assert!(docs.len() >= 2, "need ≥ 2 docs for ≥ 2 source segments");
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();

        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(InlineClusterer {
                centroids: centroids.to_vec(),
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        let mid = docs.len() / 2;
        for chunk in [&docs[..mid.max(1)], &docs[mid.max(1)..]] {
            for (label, v) in chunk {
                let mut doc = TantivyDocument::new();
                doc.add_text(label_field, label);
                doc.add_vector(embed_field, v.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }
        let segment_ids: Vec<_> = index.searchable_segment_ids()?.into_iter().collect();
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;
        Ok((index, embed_field, label_field))
    }

    fn decode_2d(bytes: &[u8]) -> [f32; 2] {
        [
            f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        ]
    }

    fn nearest_centroid(p: [f32; 2], centroids: &[[f32; 2]]) -> usize {
        let mut best = 0;
        let mut best_d2 = f32::INFINITY;
        for (i, c) in centroids.iter().enumerate() {
            let dx = p[0] - c[0];
            let dy = p[1] - c[1];
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = i;
            }
        }
        best
    }

    const DOCS_PER_CLUSTER: usize = 6;

    fn multi_cluster_fixture() -> (Vec<[f32; 2]>, Vec<String>) {
        let centroids = vec![
            [0.0f32, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 10.0],
        ];
        let labels = (0..centroids.len() * DOCS_PER_CLUSTER)
            .map(|i| format!("d{i}"))
            .collect();
        (centroids, labels)
    }

    fn multi_cluster_docs<'a>(
        centroids: &[[f32; 2]],
        labels: &'a [String],
    ) -> Vec<(&'a str, [f32; 2])> {
        (0..labels.len())
            .map(|i| {
                let c = centroids[i / DOCS_PER_CLUSTER];
                let off = (i % DOCS_PER_CLUSTER) as f32 * 0.01;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect()
    }

    /// Merging flat segments with deletes past the clustering threshold: rows
    /// written for since-deleted docs still count toward the sources'
    /// `count()` (tombstones don't rewrite `.vec`), so the alive-doc merge
    /// iteration legitimately comes up short of `vector_count`. The merge
    /// must tolerate that, and the resulting IVF segment must hold — and
    /// count — the alive docs only.
    #[test]
    fn merge_flat_segments_with_deletes_into_ivf() -> crate::Result<()> {
        let (centroids, labels) = multi_cluster_fixture();
        let docs = multi_cluster_docs(&centroids, &labels);
        let n = docs.len();

        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(InlineClusterer {
                centroids: centroids.clone(),
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let mid = n / 2;
        for chunk in [&docs[..mid], &docs[mid..]] {
            for (label, v) in chunk {
                let mut doc = TantivyDocument::new();
                doc.add_text(label_field, label);
                doc.add_vector(embed_field, v.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }

        let deleted = ["d0", "d7", "d35"];
        for label in deleted {
            writer.delete_term(Term::from_field_text(label_field, label));
        }
        writer.commit()?;
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        let alive = n - deleted.len();
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1, "one merged segment");
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        let info = vec_reader.info().expect("vector info");
        assert_eq!(info.format, VectorStorageFormat::Ivf, "merge must cluster");
        assert_eq!(info.num_vectors, alive, "deleted docs must not be counted");
        assert_eq!(vec_reader.num_vectors(), alive);

        let hits = search(
            &index,
            embed_field,
            &AllQuery,
            vec![10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?;
        assert_eq!(hits.len(), alive, "exhaustive top-N must return alive docs");
        let mut seen_labels = std::collections::HashSet::new();
        for (_, addr) in &hits {
            let label = stored_label_at(&index, label_field, *addr)?;
            assert!(
                !deleted.contains(&label.as_str()),
                "deleted doc {label} surfaced in results"
            );
            assert!(seen_labels.insert(label), "duplicate doc in results");
        }
        Ok(())
    }

    #[test]
    fn merge_deleting_every_doc_of_one_field_writes_empty_ivf() -> crate::Result<()> {
        let (centroids, labels) = multi_cluster_fixture();
        let docs = multi_cluster_docs(&centroids, &labels);
        let n = docs.len();

        let mut sb = Schema::builder();
        let doomed_field = sb.add_vector_field(
            "embedding_doomed",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let kept_field = sb.add_vector_field(
            "embedding_kept",
            VectorOptions::new(2, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let schema = sb.build();
        let settings = IndexSettings {
            vector_clustering_threshold: 1,
            ..IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .ivf_clusterer(Arc::new(InlineClusterer {
                centroids: centroids.clone(),
            }))
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        let mid = n / 2;
        for (i, (label, v)) in docs.iter().enumerate() {
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, label);
            let field = if i % 2 == 0 { doomed_field } else { kept_field };
            doc.add_vector(field, v.as_slice());
            writer.add_document(doc)?;
            if i + 1 == mid {
                writer.commit()?;
            }
        }
        writer.commit()?;

        for (i, (label, _)) in docs.iter().enumerate() {
            if i % 2 == 0 {
                writer.delete_term(Term::from_field_text(label_field, label));
            }
        }
        writer.commit()?;
        let segment_ids = index.searchable_segment_ids()?;
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1, "one merged segment");
        let segment_reader = &searcher.segment_readers()[0];

        let vec_reader = segment_reader.vector_index(doomed_field)?;
        assert_eq!(vec_reader.num_vectors(), 0);
        let info = vec_reader.info().expect("vector info");
        assert_eq!(
            info,
            VectorInfo {
                format: VectorStorageFormat::Ivf,
                num_vectors: 0,
                num_centroids: Some(0),
                cluster_stats: Some(VectorClusterStats {
                    min_cluster_size: 0,
                    max_cluster_size: 0,
                    avg_cluster_size: 0.0,
                    empty_clusters: 0,
                }),
            },
        );
        let ivf = vec_reader.index().expect("expected IVF storage");
        assert!(vec_reader.is_empty(), "no rows in the emptied field");
        assert_eq!(ivf.num_rows(), 0);
        assert_eq!(ivf.num_clusters(), 0);

        let kept_count = n / 2;
        assert_eq!(
            segment_reader.vector_index(kept_field)?.num_vectors(),
            kept_count
        );
        let hits = search(
            &index,
            kept_field,
            &AllQuery,
            vec![10.0, 10.0],
            n,
            exhaustive_params(centroids.len()),
        )?;
        assert_eq!(hits.len(), kept_count, "kept field returns alive docs");
        Ok(())
    }

    struct CaptureLogger;
    static CAPTURED_IVF_BUILD: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    impl log::Log for CaptureLogger {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.target() == "paradedb::ivf_build"
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                CAPTURED_IVF_BUILD
                    .lock()
                    .unwrap()
                    .push(format!("{}", r.args()));
            }
        }
        fn flush(&self) {}
    }
    static CAPTURE_LOGGER: CaptureLogger = CaptureLogger;

    /// The merge emits one parseable `ivf_build timings_ms ...` line per
    /// field. Builds a larger index so the phase timings are measurable,
    /// captures the line, and prints it (run with `--nocapture`) so we can
    /// see where build time goes.
    #[test]
    fn ivf_build_emits_timings_log() -> crate::Result<()> {
        let _ = log::set_logger(&CAPTURE_LOGGER);
        log::set_max_level(log::LevelFilter::Info);

        let mut centroids: Vec<[f32; 2]> = Vec::new();
        for x in 0..20 {
            for y in 0..10 {
                centroids.push([x as f32 * 10.0, y as f32 * 10.0]);
            }
        }
        let n_per = 25usize;
        let labels: Vec<String> = (0..centroids.len() * n_per)
            .map(|i| format!("d{i}"))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
            .map(|i| {
                let c = centroids[i / n_per];
                let off = (i % n_per) as f32 * 0.05;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect();

        let before = CAPTURED_IVF_BUILD.lock().unwrap().len();
        let _ = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let lines: Vec<String> = CAPTURED_IVF_BUILD.lock().unwrap()[before..].to_vec();
        let line = lines
            .iter()
            .find(|l| l.contains("ivf_build timings_ms") && l.contains("centroids=200"))
            .expect("expected an ivf_build timings line for the 200-centroid build");
        assert!(line.contains("train="));
        assert!(line.contains("posting_write="));
        eprintln!("IVF_BUILD_SAMPLE {line}");
        Ok(())
    }

    #[test]
    fn ivf_top_n_brute_force_oracle_l2() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
        for query in [[0.5_f32, 0.5], [9.5, 9.5], [5.0, 0.0], [3.7, 11.2]] {
            for k in [1usize, 3, 6, 10] {
                let expected = index.ground_truth(query, k)?;
                let actual = search(
                    &index.index,
                    index.embedding_field(),
                    &AllQuery,
                    query.to_vec(),
                    k,
                    params.clone(),
                )?;
                assert_eq!(actual, expected, "L2 exhaustive query={query:?} k={k}");
            }
        }
        Ok(())
    }

    #[test]
    fn ivf_top_n_brute_force_oracle_cosine() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::Cosine)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
        for query in [[1.0_f32, 0.0], [0.0, 1.0], [0.7, 0.3]] {
            for k in [1usize, 3, 6] {
                let expected = index.ground_truth(query, k)?;
                let actual = search(
                    &index.index,
                    index.embedding_field(),
                    &AllQuery,
                    query.to_vec(),
                    k,
                    params.clone(),
                )?;
                assert_eq!(actual, expected, "Cosine exhaustive query={query:?} k={k}");
            }
        }
        Ok(())
    }

    #[test]
    fn ivf_top_n_brute_force_oracle_dot() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::Dot)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = exhaustive_params(DEFAULT_NUM_CENTROIDS);
        for query in [[1.0_f32, 0.0], [2.0, 0.0], [0.5, -0.5]] {
            for k in [1usize, 3, 6] {
                let expected = index.ground_truth(query, k)?;
                let actual = search(
                    &index.index,
                    index.embedding_field(),
                    &AllQuery,
                    query.to_vec(),
                    k,
                    params.clone(),
                )?;
                assert_eq!(actual, expected, "Dot exhaustive query={query:?} k={k}");
            }
        }
        Ok(())
    }

    #[test]
    fn ivf_top_n_trap_case() -> crate::Result<()> {
        let centroids = vec![[0.0_f32, 0.0], [10.0, 10.0]];
        let docs = [
            ("far_a", [0.0_f32, -10.0]),
            ("far_a", [-10.0, 0.0]),
            ("trap_b", [5.0, 5.01]),
            ("anchor_b", [10.0, 10.0]),
        ];
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let query = [1.0_f32, 1.0];

        let oracle = ground_truth_top_k(&index, embed_field, Metric::L2, &query, 1)?;
        let trap_doc = stored_label_at(&index, label_field, oracle[0].1)?;
        assert_eq!(trap_doc, "trap_b", "true NN must be the trap doc");

        let one_probe = AdaptiveProbeParams {
            max_probe_fraction: 0.5,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let hits1 = search(&index, embed_field, &AllQuery, query.to_vec(), 1, one_probe)?;
        assert_eq!(hits1.len(), 1);
        assert_ne!(
            stored_label_at(&index, label_field, hits1[0].1)?,
            "trap_b",
            "a 1-cluster probe ceiling should miss the trap (probes only cluster A)",
        );

        let hits2 = search(
            &index,
            embed_field,
            &AllQuery,
            query.to_vec(),
            1,
            exhaustive_params(2),
        )?;
        assert_eq!(hits2.len(), 1);
        assert_eq!(
            stored_label_at(&index, label_field, hits2[0].1)?,
            "trap_b",
            "exhaustive probing should find the trap doc",
        );
        Ok(())
    }

    #[test]
    fn ivf_top_n_filter_selectivity() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .selectivities(&[0.1])
            .build()?;
        let filter = TermQuery::new(
            Term::from_field_text(index.label_field(), "selectivity_0.1"),
            IndexRecordOption::Basic,
        );
        let query = [0.5_f32, 0.5];
        let k = 5;
        let filter_set = collect_filter_doc_set(&index.index, &filter)?;
        let mut restricted = ground_truth_top_k(
            &index.index,
            index.embedding_field(),
            Metric::L2,
            &query,
            FIXTURE_NUM_DOCS,
        )?;
        restricted.retain(|(_, addr)| filter_set.contains(addr));
        restricted.truncate(k);

        let actual = search(
            &index.index,
            index.embedding_field(),
            &filter,
            query.to_vec(),
            k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(actual, restricted);
        for (_, addr) in &actual {
            assert!(filter_set.contains(addr), "hit outside filter: {addr:?}");
        }
        Ok(())
    }

    #[test]
    fn ivf_top_n_empty_filter() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let empty = TermQuery::new(
            Term::from_field_text(index.label_field(), "absent"),
            IndexRecordOption::Basic,
        );
        let hits = search(
            &index.index,
            index.embedding_field(),
            &empty,
            vec![0.0_f32, 0.0],
            5,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert!(hits.is_empty());
        Ok(())
    }

    #[test]
    fn ivf_top_n_k_exceeds_candidates() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let query = [0.0_f32, 0.0];
        let big_k = FIXTURE_NUM_DOCS + 50;
        let expected = index.ground_truth(query, big_k)?;
        let actual = search(
            &index.index,
            index.embedding_field(),
            &AllQuery,
            query.to_vec(),
            big_k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(actual.len(), FIXTURE_NUM_DOCS);
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn ivf_top_n_respects_deletes() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .selectivities(&[0.1])
            .build()?;
        {
            let mut writer: IndexWriter = index.index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            writer.delete_term(Term::from_field_text(
                index.label_field(),
                "selectivity_0.1",
            ));
            writer.commit()?;
        }

        let query = [0.0_f32, 0.0];
        let searcher = index.index.reader()?.searcher();
        let mut alive_addrs = std::collections::HashSet::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            let alive = segment_reader.alive_bitset();
            for doc in 0..segment_reader.max_doc() {
                let is_alive = alive.is_none_or(|bs| bs.is_alive(doc));
                if is_alive {
                    alive_addrs.insert(DocAddress::new(seg_ord as u32, doc));
                }
            }
        }
        assert!(
            alive_addrs.len() < FIXTURE_NUM_DOCS,
            "delete didn't remove anything (alive={})",
            alive_addrs.len(),
        );
        let k = 10;
        let mut expected = ground_truth_top_k(
            &index.index,
            index.embedding_field(),
            Metric::L2,
            &query,
            FIXTURE_NUM_DOCS,
        )?;
        expected.retain(|(_, addr)| alive_addrs.contains(addr));
        expected.truncate(k);

        let actual = search(
            &index.index,
            index.embedding_field(),
            &AllQuery,
            query.to_vec(),
            k,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(actual, expected);
        for (_, addr) in &actual {
            assert!(
                alive_addrs.contains(addr),
                "deleted doc {addr:?} surfaced in results",
            );
        }
        Ok(())
    }
    #[test]
    fn ivf_top_n_zero_returns_empty() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let (hits, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            0,
            AdaptiveProbeParams::default(),
        )?;
        assert!(hits.is_empty());
        assert_eq!(stats.clusters_probed(), 0);
        assert_eq!(stats.candidates_scored, 0);
        Ok(())
    }

    #[test]
    fn ivf_top_n_collects_probe_stats() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let (_, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            64,
            exhaustive_params(DEFAULT_NUM_CENTROIDS),
        )?;
        assert_eq!(stats.clusters_probed(), DEFAULT_NUM_CENTROIDS);
        let segment_doc_count =
            index.index.reader()?.searcher().segment_readers()[0].max_doc() as usize;
        assert_eq!(stats.candidates_scored, segment_doc_count);

        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
            "visited must equal filter+dead+scored ({stats:?})"
        );
        assert_eq!(stats.routing.visited_count, DEFAULT_NUM_CENTROIDS);
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        Ok(())
    }

    #[test]
    fn ivf_probe_stats_termination_ceiling() -> crate::Result<()> {
        let centroids = [
            [0.0f32, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 10.0],
        ];
        let n_per = 6usize;
        let labels: Vec<String> = (0..centroids.len() * n_per)
            .map(|i| format!("d{i}"))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
            .map(|i| {
                let c = centroids[i / n_per];
                let off = (i % n_per) as f32 * 0.01;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let (_, stats) = run_top_n(&index, embed_field, vec![10.0, 10.0], 3, params)?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert_eq!(stats.clusters_probed(), 1);
        assert_eq!(stats.routing.visited_count, centroids.len());
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
            "visited must equal filter+dead+scored ({stats:?})"
        );
        Ok(())
    }

    #[test]
    fn ivf_single_centroid_routes_without_graph() -> crate::Result<()> {
        let centroids = [[0.0f32, 0.0]];
        let labels: Vec<String> = (0..5).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, [f32; 2])> = (0..5)
            .map(|i| (labels[i].as_str(), [i as f32 * 0.01, 0.0]))
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_some(),
            "expected IVF storage"
        );

        let query = [0.0f32, 0.0];
        let k = 3;
        let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, k)?;
        let (hits, stats) = run_top_n(
            &index,
            embed_field,
            query.to_vec(),
            k,
            AdaptiveProbeParams::default(),
        )?;
        assert_eq!(hits, expected, "linear fallback must match the oracle");
        assert_eq!(stats.clusters_probed(), 1, "one cluster, one probe");
        assert_eq!(stats.routing.visited_count, 1);
        Ok(())
    }

    #[test]
    fn ivf_routed_ranking_matches_oracle_on_separated_clusters() -> crate::Result<()> {
        let side = 4usize;
        let centroids: Vec<[f32; 2]> = (0..side * side)
            .map(|i| [(i % side) as f32 * 10.0, (i / side) as f32 * 10.0])
            .collect();
        let n_per = 4usize;
        let labels: Vec<String> = (0..centroids.len() * n_per)
            .map(|i| format!("d{i}"))
            .collect();
        let docs: Vec<(&str, [f32; 2])> = (0..centroids.len() * n_per)
            .map(|i| {
                let c = centroids[i / n_per];
                let off = (i % n_per) as f32 * 0.01;
                (labels[i].as_str(), [c[0] + off, c[1] + off])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        let ivf = vec_reader.index().expect("expected IVF segment");
        assert_eq!(ivf.num_clusters(), centroids.len());

        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.1,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let k = 3usize;
        for (ord, centroid) in centroids.iter().enumerate().step_by(3) {
            let query = [centroid[0] + 0.3, centroid[1] - 0.2];
            let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, k)?;
            let (hits, stats) = run_top_n(&index, embed_field, query.to_vec(), k, params.clone())?;
            assert_eq!(hits, expected, "routed top-{k} near centroid {ord}");
            assert!(
                stats.clusters_probed() <= 2,
                "cap 2 must bound the probes, got {}",
                stats.clusters_probed()
            );
            assert!(
                stats.routing.visited_count <= centroids.len(),
                "navigation cost is the beam-visited count"
            );
        }
        Ok(())
    }

    #[test]
    fn ivf_cluster_sizes_match_vector_info() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let field = index.embedding_field();
        let searcher = index.index.reader()?.searcher();

        let mut segments_checked = 0;
        for segment_reader in searcher.segment_readers() {
            let vec_reader = segment_reader.vector_index(field)?;
            let sizes = vec_reader
                .cluster_sizes()
                .expect("ivf segment exposes cluster sizes");
            let info = vec_reader.info().expect("vector info");
            assert_eq!(info.format, VectorStorageFormat::Ivf);
            let stats = info.cluster_stats.expect("ivf cluster stats");

            assert_eq!(sizes.len(), info.num_centroids.expect("ivf centroids"));
            let sum: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
            let min = sizes.iter().copied().min().unwrap() as usize;
            let max = sizes.iter().copied().max().unwrap() as usize;
            let empty = sizes.iter().filter(|&&s| s == 0).count();
            let avg = sum as f64 / sizes.len() as f64;

            let ivf = vec_reader.index().expect("expected IVF segment");
            assert_eq!(sum as usize, ivf.num_rows(), "sizes sum to rows");
            // ...and the shared fixture has no deletes, so the row total
            // coincides with the distinct-doc `num_vectors`.
            assert_eq!(sum as usize, info.num_vectors, "rows == docs");
            assert_eq!(min, stats.min_cluster_size, "min");
            assert_eq!(max, stats.max_cluster_size, "max");
            assert_eq!(empty, stats.empty_clusters, "empty");
            assert!(
                (avg - stats.avg_cluster_size).abs() < 1e-9,
                "avg {avg} vs {}",
                stats.avg_cluster_size
            );
            segments_checked += 1;
        }
        assert!(
            segments_checked > 0,
            "fixture must produce >= 1 IVF segment"
        );

        let flat = TestVectorIndex::builder(VectorDType::F32)
            .metric(Metric::L2)
            .vector_storage_format(VectorStorageFormat::Flat)
            .build()?;
        let flat_field = flat.embedding_field();
        let flat_searcher = flat.index.reader()?.searcher();
        for segment_reader in flat_searcher.segment_readers() {
            assert!(
                segment_reader
                    .vector_index(flat_field)?
                    .cluster_sizes()
                    .is_none(),
                "flat segments expose no cluster sizes"
            );
        }
        Ok(())
    }

    fn budget_only_params() -> AdaptiveProbeParams {
        AdaptiveProbeParams {
            max_probe_fraction: 1.0,
            min_probe_clusters: 1,
            ..Default::default()
        }
    }

    #[test]
    fn unit_normalization_exact() -> crate::Result<()> {
        let centroids = vec![[0.0_f32, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]];
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for (c, count) in [(0usize, 5usize), (1, 2), (2, 2), (3, 1)] {
            for i in 0..count {
                docs.push((
                    format!("d{c}_{i}"),
                    [centroids[c][0] + i as f32 * 0.01, 0.0],
                ));
            }
        }
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let (_, stats) = run_top_n(
            &index,
            embed_field,
            vec![0.0, 0.0],
            11,
            budget_only_params(),
        )?;
        let c = centroids.len() as f32;
        assert_eq!(stats.termination, ProbeTermination::Exhausted);
        assert!(
            (stats.work_charged - c).abs() <= 1e-6 * c,
            "an exhaustive scan must charge exactly C units: {stats:?}"
        );
        Ok(())
    }

    /// Skew charges proportionally: a 30-doc cluster consumes most of the
    /// budget a cluster-count budget would spread over five clusters, and
    /// the stop point shifts - hand-verified numbers. Also pins the
    /// overshoot bound: the final overrun is at most the last cluster's
    /// charge.
    #[test]
    fn imbalance_charges_proportionally() -> crate::Result<()> {
        let centroids = vec![
            [0.0_f32, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
            [50.0, 0.0],
        ];
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for (c, count) in [(0usize, 30usize), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)] {
            for i in 0..count {
                docs.push((
                    format!("d{c}_{i}"),
                    [centroids[c][0] + i as f32 * 0.001, 0.0],
                ));
            }
        }
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.8,
            ..budget_only_params()
        };
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let (_, stats) = run_top_n(&index, embed_field, vec![0.0, 0.0], 41, params)?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling, "{stats:?}");
        assert_eq!(
            stats.clusters_probed(),
            4,
            "the big cluster eats the budget a count regime spreads over 5: {stats:?}"
        );
        assert!(
            (stats.work_charged - 5.123).abs() < 2e-3,
            "hand-computed spend: {stats:?}"
        );
        let budget = 0.8f32 * 6.0;
        let last_charge = 0.1975 + 2.0 * 0.1204;
        assert!(
            stats.work_charged > budget && stats.work_charged - budget <= last_charge + 1e-4,
            "overshoot bounded by the last cluster's charge: {stats:?}"
        );
        Ok(())
    }

    #[test]
    fn filtered_rows_are_not_charged() -> crate::Result<()> {
        let centroids = vec![[0.0_f32, 0.0], [50.0, 0.0]];
        let mut docs: Vec<(String, [f32; 2])> = Vec::new();
        for (c, count) in [(0usize, 20usize), (1, 2)] {
            for i in 0..count {
                docs.push((
                    format!("d{c}_{i}"),
                    [centroids[c][0] + i as f32 * 0.001, 0.0],
                ));
            }
        }
        let docs: Vec<(&str, [f32; 2])> = docs.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

        let n_avg = 22.0 / 2.0;
        let fixed = fixed_probe_cost_rows();
        let x = fixed / (fixed + n_avg);
        let row = (1.0 - x) / n_avg;

        let (_, full) = run_top_n(
            &index,
            embed_field,
            vec![0.0, 0.0],
            23,
            budget_only_params(),
        )?;
        assert_eq!(full.clusters_probed(), 2, "{full:?}");
        assert_eq!(full.candidates_scored, 22, "{full:?}");
        assert!(
            (full.work_charged as f64 - 2.0).abs() < 1e-5,
            "unfiltered exhaustive scan charges exactly C: {full:?}"
        );

        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let admitted: Vec<DocId> = segment_reader
            .vector_index(embed_field)?
            .cluster_doc_ids(0)
            .expect("cluster 0 doc ids")
            .into_iter()
            .take(3)
            .collect();
        assert_eq!(admitted.len(), 3, "setup: need 3 admitted docs");
        let weight = FixedDocsWeight {
            max_doc: segment_reader.max_doc(),
            docs: admitted,
        };
        drop(searcher);

        let (_, filtered) = run_top_n_with_weight(
            &index,
            embed_field,
            vec![0.0, 0.0],
            23,
            budget_only_params(),
            &weight,
        )?;
        assert_eq!(filtered.clusters_probed(), 2, "{filtered:?}");
        assert_eq!(filtered.vectors_visited, 22, "{filtered:?}");
        assert_eq!(filtered.candidates_scored, 3, "{filtered:?}");

        let expected = 2.0 * x + 3.0 * row;
        assert!(
            (filtered.work_charged as f64 - expected).abs() < 1e-5,
            "charge must be 2 opens + 3 scored rows ({expected}): {filtered:?}"
        );
        assert!(
            ((full.work_charged - filtered.work_charged) as f64 - 19.0 * row).abs() < 1e-5,
            "the filtered rows must account for the entire difference: full={full:?} \
             filtered={filtered:?}"
        );
        Ok(())
    }

    #[test]
    fn probe_stats_max_probe_fraction_ceiling() -> crate::Result<()> {
        let index = TestVectorIndex::builder(VectorDType::F32)
            .vector_storage_format(VectorStorageFormat::Ivf)
            .build()?;
        let params = AdaptiveProbeParams {
            max_probe_fraction: 0.2,
            min_probe_clusters: 1,
            ..Default::default()
        };
        let searcher = index.index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(index.embedding_field())?;
        let ivf = vec_reader.index().expect("expected IVF storage");
        let (clusters, docs) = (ivf.num_clusters(), ivf.num_docs());
        let (budget, n_avg, x) = params.resolved_work_budget(clusters, docs)?;
        assert!(budget < clusters as f64, "setup: the budget must bind");
        drop(searcher);

        let (_, stats) = run_top_n(
            &index.index,
            index.embedding_field(),
            vec![0.0_f32, 0.0],
            3,
            params,
        )?;
        assert_eq!(stats.termination, ProbeTermination::Ceiling);
        assert!(
            stats.clusters_probed() < clusters,
            "the budget must bind before exhaustion: {stats:?}"
        );
        assert!(
            stats.work_charged as f64 > budget,
            "the ceiling fires only once the budget is spent: {stats:?}"
        );
        let max_cluster_charge = x + docs as f64 * (1.0 - x) / n_avg;
        assert!(
            stats.work_charged as f64 <= budget + max_cluster_charge + 1e-6,
            "overshoot is bounded by the last cluster's charge: {stats:?}"
        );
        Ok(())
    }

    #[test]
    fn quantized_boundary_kth_uses_the_row_tie_for_sigma() {
        let mut scan = QuantizedScanCtx::new(3, 3);
        scan.begin_cluster(2);
        for (row, score, sigma) in [(0, 10.0, 0.0), (1, 9.0, 1.0), (2, 9.0, 100.0)] {
            push_test_candidate(&mut scan, row, row as DocId, score, sigma);
        }
        scan.finish_cluster_bound();

        assert_eq!(scan.running_pessimistic_kth(2, 2.0), Some(7.0));
        assert_eq!(scan.pessimistic_kth(2, 2.0), Some(7.0));
    }

    #[test]
    fn cluster_local_admission_kth_matches_full_partition() {
        const TOP_N: usize = 4;
        let mut scan = QuantizedScanCtx::new(12, 24);
        for cluster in 0..6 {
            scan.begin_cluster(TOP_N);
            for row in cluster * 4..cluster * 4 + 4 {
                push_test_candidate(
                    &mut scan,
                    row,
                    (row % 12) as DocId,
                    3.0 - row as f32 * 0.071 + (row as f32 * 0.37).sin() * 0.2,
                    0.01 + (row % 5) as f32 * 0.003,
                );
            }
            scan.finish_cluster_bound();
            assert_eq!(
                scan.running_pessimistic_kth(TOP_N, 2.0),
                scan.pessimistic_kth(TOP_N, 2.0),
                "cluster={cluster}"
            );
        }
    }

    fn independent_admission_top(scan: &QuantizedScanCtx, top_n: usize) -> Vec<usize> {
        let mut indices = (0..scan.candidates.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|&left, &right| {
            let left_estimate = scan.candidates.estimates[left];
            let right_estimate = scan.candidates.estimates[right];
            right_estimate
                .total_cmp(&left_estimate)
                .then(scan.candidates.rows[left].cmp(&scan.candidates.rows[right]))
        });
        indices.truncate(top_n);
        indices
    }

    #[test]
    fn cluster_batch_selection_matches_independent_oracle() {
        let clusters: &[&[(usize, DocId, f32, f32)]] = &[
            &[(0, 0, 1.0, 0.01), (1, 1, 2.0, 0.02)],
            &[
                (2, 2, 9.0, 0.03),
                (3, 3, 8.0, 0.04),
                (4, 4, 7.0, 0.05),
                (5, 0, 6.0, 0.06),
                (6, 5, 5.0, 0.07),
            ],
            &[
                (7, 6, 0.0, 0.08),
                (8, 1, 10.0, 0.09),
                (9, 7, 1.0, 0.10),
                (10, 2, 9.0, 8.0),
                (11, 8, 2.0, 0.11),
            ],
        ];

        for top_n in [1, 3] {
            let mut scan = QuantizedScanCtx::new(16, 16);
            for (cluster, rows) in clusters.iter().enumerate() {
                scan.begin_cluster(top_n);
                for &(row, doc, estimate, sigma) in *rows {
                    push_test_candidate(&mut scan, row, doc, estimate, sigma);
                }
                scan.finish_cluster_bound();

                let expected = independent_admission_top(&scan, top_n);
                let actual_rows = scan
                    .bound_top
                    .iter()
                    .map(|&index| scan.candidates.rows[index])
                    .collect::<Vec<_>>();
                let expected_rows = expected
                    .iter()
                    .map(|&index| scan.candidates.rows[index])
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual_rows, expected_rows,
                    "cluster={cluster}, top_n={top_n}"
                );

                let expected_kth = (expected.len() == top_n).then(|| {
                    let index = expected[top_n - 1];
                    scan.candidates.estimates[index] - 2.0 * scan.candidates.sigmas[index]
                });
                assert_eq!(
                    scan.running_pessimistic_kth(top_n, 2.0),
                    expected_kth,
                    "cluster={cluster}, top_n={top_n}"
                );
            }
        }
    }

    #[test]
    fn cosine_refinement_batches_cross_clusters_and_cap_at_2048() {
        let rows = (0..2_049).collect::<Vec<_>>();
        let clusters = [0..700, 700..1_400, 1_400..2_100];
        let batches = cosine_refinement_batches(rows.len()).collect::<Vec<_>>();
        assert_eq!(batches, [0..2_048, 2_048..2_049]);
        let first_rows = rows[batches[0].clone()].iter().copied();
        for cluster in clusters {
            assert!(
                first_rows.clone().any(|row| cluster.contains(&row)),
                "one logical cosine batch must cross all three clusters"
            );
        }
    }

    #[test]
    fn initial_l2_gamma_leaves_exact_base_unscaled() {
        let mut raw_prefixes = [2.0];
        let mut bases = [0.0];
        let mut estimates = [0.0];
        let mut sigmas = [0.0];
        let mut residual_norms_squared = [0.0];
        let mut sign_query_error_terms = [0.0];
        combine_initial_decoded(
            Metric::L2,
            1,
            &mut raw_prefixes,
            &mut bases,
            &mut estimates,
            &mut sigmas,
            &mut residual_norms_squared,
            &mut sign_query_error_terms,
            &[3.0],
            &[2.0],
            &[0.5],
            &[5.0],
            &[7.0],
            10.0,
            11.0,
            13.0,
        );

        assert_eq!(bases, [3.0]);
        assert_eq!(raw_prefixes, [1.0]);
        assert_eq!(estimates, [7.0]);
        assert_eq!(residual_norms_squared, [7.0]);
        assert_eq!(sign_query_error_terms, [117.0]);
        let variance: f32 = 7.0 * 0.5 * 11.0 + 4.0 * 117.0;
        let expected_sigma = 2.0 * 1.15 * variance.sqrt();
        assert!((sigmas[0] - expected_sigma).abs() < 1e-5);
    }

    #[test]
    fn l2_refinement_gamma_corrects_only_raw_prefix_and_keeps_state_local() {
        let mut candidates = QuantizedCandidates::with_capacity(2);
        candidates.push(0, 10, 10.0, 2.0, 14.0, 0.0, 9.0, 1.0, 0.0);
        candidates.push(1, 11, 20.0, 1.0, 22.0, 0.0, 4.0, 1.0, 1.0);

        combine_refinement_decoded(
            Metric::L2,
            1,
            &mut candidates,
            0..1,
            &[2.0],
            &[3.0],
            &[2.0],
            &[0.5],
            &[5.0],
            7.0,
            11.0,
        );
        combine_refinement_decoded(
            Metric::L2,
            1,
            &mut candidates,
            1..2,
            &[4.0],
            &[2.0],
            &[1.5],
            &[0.25],
            &[1.0],
            17.0,
            13.0,
        );

        assert_eq!(candidates.bases, [10.0, 20.0]);
        assert_eq!(candidates.raw_prefixes, [3.0, 8.0]);
        assert_eq!(candidates.estimates, [22.0, 44.0]);
        assert_eq!(candidates.residual_norm_squared, [9.0, 4.0]);
        assert_eq!(candidates.sign_query_error_terms, [99.0, 53.0]);
        let expected0 = 2.0 * 1.15 * 427.5_f32.sqrt();
        let expected1 = 2.0 * 1.15 * 136.25_f32.sqrt();
        assert!((candidates.sigmas[0] - expected0).abs() < 1e-5);
        assert!((candidates.sigmas[1] - expected1).abs() < 1e-5);
    }

    #[test]
    fn survivor_sets_match_kernel_harness() {
        const CANDIDATES: usize = 40;
        const TOP_N: usize = 10;
        const KAPPA: f32 = QUANTIZED_BOUNDARY_KAPPA;

        let layer0_scores: Vec<f32> = (0..CANDIDATES)
            .map(|row| 2.0 - row as f32 * 0.09 + (row as f32 * 0.7).sin() * 0.08)
            .collect();
        let layer0_sigmas: Vec<f32> = (0..CANDIDATES)
            .map(|row| 0.03 + (row % 5) as f32 * 0.01)
            .collect();
        let refinements: Vec<f32> = (0..CANDIDATES)
            .map(|row| (row as f32 * 0.41).cos() * 0.06)
            .collect();
        let layer1_sigmas: Vec<f32> = (0..CANDIDATES)
            .map(|row| 0.012 + (row % 3) as f32 * 0.004)
            .collect();

        let (first_kth_index, first_kth) = cascade::kth(&layer0_scores, TOP_N);
        let harness_first = cascade::band_filter(
            &layer0_scores,
            &layer0_sigmas,
            KAPPA,
            first_kth - KAPPA * layer0_sigmas[first_kth_index],
        );

        let mut scan = QuantizedScanCtx::new(CANDIDATES as DocId, CANDIDATES);
        for row in 0..CANDIDATES {
            push_test_candidate(
                &mut scan,
                row,
                row as DocId,
                layer0_scores[row],
                layer0_sigmas[row],
            );
        }
        scan.band(TOP_N, KAPPA);
        let scan_first: Vec<u32> = scan.candidates.rows.iter().map(|&row| row as u32).collect();
        assert!(
            harness_first.len() < CANDIDATES,
            "first boundary must measurably filter"
        );
        assert_eq!(scan_first, harness_first, "layer-0 survivor set");

        let second_scores: Vec<f32> = harness_first
            .iter()
            .map(|&row| layer0_scores[row as usize] + refinements[row as usize])
            .collect();
        let second_sigmas: Vec<f32> = harness_first
            .iter()
            .map(|&row| layer1_sigmas[row as usize])
            .collect();
        let (second_kth_index, second_kth) = cascade::kth(&second_scores, TOP_N);
        let harness_second_local = cascade::band_filter(
            &second_scores,
            &second_sigmas,
            KAPPA,
            second_kth - KAPPA * second_sigmas[second_kth_index],
        );
        let harness_second: Vec<u32> = harness_second_local
            .iter()
            .map(|&local| harness_first[local as usize])
            .collect();

        for index in 0..scan.candidates.len() {
            let row = scan.candidates.rows[index];
            scan.candidates.estimates[index] += refinements[row];
            scan.candidates.sigmas[index] = layer1_sigmas[row];
        }
        scan.band(TOP_N, KAPPA);
        let scan_second: Vec<u32> = scan.candidates.rows.iter().map(|&row| row as u32).collect();
        assert!(
            harness_second.len() < harness_first.len(),
            "second boundary must measurably filter"
        );
        assert_eq!(scan_second, harness_second, "layer-1 survivor set");

        let mut exact_order: Vec<usize> = (0..CANDIDATES).collect();
        exact_order.sort_unstable_by(|&left, &right| {
            (layer0_scores[right] + refinements[right])
                .total_cmp(&(layer0_scores[left] + refinements[left]))
                .then_with(|| left.cmp(&right))
        });
        let recalled = exact_order[..TOP_N]
            .iter()
            .filter(|&&row| scan_second.contains(&(row as u32)))
            .count();
        assert_eq!(recalled, TOP_N, "candidate recall must match the harness");
    }

    #[test]
    fn quantized_boundary_filters_with_finite_sigmas() {
        let mut scan = QuantizedScanCtx::new(3, 3);
        for (row, score) in [(0, 100.0), (1, 99.0), (2, -100.0)] {
            push_test_candidate(&mut scan, row, row as DocId, score, 0.1);
        }
        scan.band(1, 2.0);
        assert_eq!(scan.candidates.len(), 1);
        assert_eq!(scan.candidates.rows[0], 0);
    }

    struct FixedDocsWeight {
        max_doc: DocId,
        docs: Vec<DocId>,
    }

    impl Weight for FixedDocsWeight {
        fn scorer(&self, _reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
            let mut bs = BitSet::with_max_value(self.max_doc);
            for &doc in &self.docs {
                bs.insert(doc);
            }
            Ok(Box::new(ConstScorer::new(BitSetDocSet::from(bs), boost)))
        }

        fn explain(&self, _reader: &SegmentReader, _doc: DocId) -> crate::Result<Explanation> {
            unreachable!("the vector backend never explains filter docs")
        }
    }

    fn run_top_n_with_weight(
        index: &Index,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        params: AdaptiveProbeParams,
        weight: &dyn Weight,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let quantized_queries = QuantizedQueryCache::default();
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            &quantized_queries,
            params,
            usize::MAX,
        )?;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_some(),
            "expected IVF storage"
        );
        backend.top_n(weight, segment_reader, k)
    }

    fn assert_stats_identities(stats: &ProbeStats) {
        assert_eq!(
            stats.vectors_visited,
            stats.pruned_filter + stats.pruned_dead + stats.candidates_scored,
            "visited must equal filter+dead+scored ({stats:?})"
        );
    }

    fn build_flat(
        dim: usize,
        docs: &[(&str, Option<Vec<f32>>)],
    ) -> crate::Result<(Index, Field, Field)> {
        let mut sb = Schema::builder();
        let embed_field = sb.add_vector_field(
            "embedding",
            VectorOptions::new(dim, Metric::L2).with_dtype(VectorDType::F32),
        );
        let label_field = sb.add_text_field("label", STRING | STORED);
        let index = Index::create_in_ram(sb.build());
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 30_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for (label, v) in docs {
            let mut doc = TantivyDocument::new();
            doc.add_text(label_field, label);
            if let Some(v) = v {
                doc.add_vector(embed_field, v.as_slice());
            }
            writer.add_document(doc)?;
        }
        writer.commit()?;
        Ok((index, embed_field, label_field))
    }

    fn run_exact_on_segment(
        segment_reader: &SegmentReader,
        embed_field: Field,
        query: Vec<f32>,
        k: usize,
        weight: &dyn Weight,
    ) -> crate::Result<(Vec<(Score, DocAddress)>, ProbeStats)> {
        let quantized_queries = QuantizedQueryCache::default();
        let backend = VectorBackend::<f32>::for_segment(
            segment_reader,
            0,
            embed_field,
            Arc::new(query),
            &quantized_queries,
            AdaptiveProbeParams::default(),
            usize::MAX,
        )?;
        assert!(
            segment_reader.vector_index(embed_field)?.index().is_none(),
            "expected flat storage"
        );
        backend.top_n(weight, segment_reader, k)
    }

    /// Filter-aware fetches across selectivities: on the multi-cluster
    /// fixture, hand-built filters admitting {0, 1, 50, 100}% of docs
    /// return every admitted doc exactly once, with both partition
    /// identities intact. 0% admits nothing — the empty-filter
    /// short-circuit returns before the probe loop, so zero clusters
    /// probe and zero fetches happen (postings_skipped == clusters
    /// probed == 0).
    #[test]
    fn filter_aware_fetch_across_selectivities() -> crate::Result<()> {
        let (centroids, labels) = multi_cluster_fixture();
        let docs = multi_cluster_docs(&centroids, &labels);
        let n = docs.len();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        let params = exhaustive_params(centroids.len());

        for pct in [0usize, 1, 50, 100] {
            let weight = FixedDocsWeight {
                max_doc: n as DocId,
                docs: (0..n as DocId)
                    .filter(|&doc| (doc as usize) * 100 < pct * n)
                    .collect(),
            };
            let admitted = weight.docs.len();
            let (hits, stats) = run_top_n_with_weight(
                &index,
                embed_field,
                vec![10.0, 10.0],
                n,
                params.clone(),
                &weight,
            )?;

            assert_eq!(
                hits.len(),
                admitted,
                "{pct}%: every admitted doc returns exactly once"
            );
            assert_stats_identities(&stats);

            match pct {
                0 => {
                    assert_eq!(stats.postings_row, 0, "0%: no fetches");
                    assert_eq!(
                        stats.postings_skipped,
                        stats.clusters_probed(),
                        "0%: every probed cluster skips its fetch"
                    );
                }
                1 => {
                    // One admitted doc → exactly one cluster fetches (its
                    // primary); every other probed cluster skips.
                    assert_eq!(stats.postings_row, 1, "{stats:?}");
                    assert_eq!(
                        stats.postings_skipped,
                        stats.clusters_probed() - 1,
                        "{stats:?}"
                    );
                }
                _ => {
                    assert!(stats.postings_row > 0, "{stats:?}");
                }
            }
        }
        Ok(())
    }

    /// Deletes are decided in the pre-pass: a cluster whose rows are all
    /// dead yields zero survivors and fetches nothing.
    #[test]
    fn filter_aware_fetch_skips_all_dead_clusters() -> crate::Result<()> {
        let (centroids, labels) = multi_cluster_fixture();
        let docs = multi_cluster_docs(&centroids, &labels);
        let n = docs.len();
        // Cluster 0's rows are exactly its 6 primary docs — deleting
        // those leaves a fully-dead cluster.
        let (index, embed_field, label_field) = build_inline_ivf(Metric::L2, &centroids, &docs)?;
        {
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for i in 0..DOCS_PER_CLUSTER {
                writer.delete_term(Term::from_field_text(label_field, &format!("d{i}")));
            }
            writer.commit()?;
        }

        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let segment_reader = &searcher.segment_readers()[0];
        let alive = segment_reader.alive_bitset().expect("deletes must land");
        let cluster0 = segment_reader
            .vector_index(embed_field)?
            .cluster_doc_ids(0)
            .expect("ivf cluster 0");
        assert_eq!(cluster0.len(), DOCS_PER_CLUSTER);
        assert!(
            cluster0.iter().all(|&doc| !alive.is_alive(doc)),
            "cluster 0 must be fully dead"
        );

        let max_doc = segment_reader.max_doc();
        let weight = FixedDocsWeight {
            max_doc,
            docs: (0..max_doc).collect(),
        };
        let params = exhaustive_params(centroids.len());
        let (hits, stats) =
            run_top_n_with_weight(&index, embed_field, vec![10.0, 10.0], n, params, &weight)?;

        assert_eq!(hits.len(), n - DOCS_PER_CLUSTER, "only alive docs surface");
        assert_eq!(
            stats.pruned_dead, DOCS_PER_CLUSTER,
            "every dead row prunes as dead: {stats:?}"
        );
        assert_eq!(stats.postings_skipped, 1, "{stats:?}");
        assert_eq!(stats.postings_row, centroids.len() - 1);
        assert_stats_identities(&stats);
        Ok(())
    }

    #[test]
    fn empty_cluster_probed_but_fetch_skipped() -> crate::Result<()> {
        // No doc is nearest to the third centroid, so its cluster is
        // empty. The
        // empty centroid sits ON the query so its zero-radius bound can
        // never prove it useless - the bounds gate must not be what
        // skips it; the empty-fetch path is what's under test.
        let centroids = vec![[0.0f32, 0.0], [10.0, 0.0], [5.0, 0.1]];
        let labels: Vec<String> = (0..8).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, [f32; 2])> = (0..8)
            .map(|i| {
                let c = centroids[i % 2];
                (labels[i].as_str(), [c[0] + (i / 2) as f32 * 0.01, c[1]])
            })
            .collect();
        let (index, embed_field, _label) = build_inline_ivf(Metric::L2, &centroids, &docs)?;

        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        assert_eq!(
            vec_reader.cluster_sizes(),
            Some(vec![4, 4, 0]),
            "setup: cluster 2 must be empty"
        );

        let max_doc = segment_reader.max_doc();
        let weight = FixedDocsWeight {
            max_doc,
            docs: (0..max_doc).collect(),
        };
        let (hits, stats) = run_top_n_with_weight(
            &index,
            embed_field,
            vec![5.0, 0.0],
            8,
            exhaustive_params(centroids.len()),
            &weight,
        )?;
        assert_eq!(hits.len(), 8);
        assert_eq!(
            stats.clusters_probed(),
            centroids.len(),
            "the empty cluster still counts as probed: {stats:?}"
        );
        assert_eq!(
            stats.postings_skipped, 1,
            "the empty cluster fetched nothing: {stats:?}"
        );
        assert_eq!(stats.postings_row, 2);
        assert_eq!(
            stats.vectors_visited, 8,
            "an empty cluster contributes no visited rows"
        );
        assert_stats_identities(&stats);
        Ok(())
    }

    #[test]
    fn flat_exact_reads_one_row_per_survivor() -> crate::Result<()> {
        let n = 40usize;
        let labels: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, Option<Vec<f32>>)> = (0..n)
            .map(|i| (labels[i].as_str(), Some(vec![i as f32 * 0.1, 1.0])))
            .collect();
        let (index, embed_field, _label) = build_flat(2, &docs)?;
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let max_doc = segment_reader.max_doc();
        assert_eq!(max_doc as usize, n, "one segment holding every doc");

        for pct in [0usize, 1, 50, 100] {
            let weight = FixedDocsWeight {
                max_doc,
                docs: (0..max_doc)
                    .filter(|&doc| (doc as usize) * 100 < pct * n)
                    .collect(),
            };
            let admitted = weight.docs.len();

            let (hits, stats) =
                run_exact_on_segment(segment_reader, embed_field, vec![0.0, 0.0], n, &weight)?;

            assert_eq!(hits.len(), admitted, "{pct}%");
            assert_eq!(stats.vectors_visited, 0, "{pct}%: {stats:?}");
            assert_eq!(stats.candidates_scored, 0, "{pct}%: {stats:?}");
            assert_eq!(stats.clusters_probed(), 0, "{pct}%: {stats:?}");
            assert_eq!(
                stats.exact_rows_read, admitted,
                "{pct}%: one row read per survivor"
            );
        }
        Ok(())
    }

    #[test]
    fn flat_exact_handles_bitmap_holes_and_deletes() -> crate::Result<()> {
        let n = 30usize;
        let labels: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
        let docs: Vec<(&str, Option<Vec<f32>>)> = (0..n)
            .map(|i| {
                let v = (i % 3 != 2).then(|| vec![i as f32 * 0.1, 1.0]);
                (labels[i].as_str(), v)
            })
            .collect();
        let (index, embed_field, label_field) = build_flat(2, &docs)?;
        {
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for i in [0usize, 6, 7, 12] {
                writer.delete_term(Term::from_field_text(label_field, &format!("d{i}")));
            }
            writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        let segment_reader = &searcher.segment_readers()[0];
        let vec_reader = segment_reader.vector_index(embed_field)?;
        assert!(vec_reader.num_vectors() < segment_reader.max_doc() as usize);
        assert!(segment_reader.alive_bitset().is_some());

        let max_doc = segment_reader.max_doc();
        let weight = FixedDocsWeight {
            max_doc,
            docs: (0..max_doc).collect(),
        };
        let query = vec![0.0f32, 0.0];
        let (hits, stats) =
            run_exact_on_segment(segment_reader, embed_field, query.clone(), n, &weight)?;

        let expected = ground_truth_top_k(&index, embed_field, Metric::L2, &query, n)?;
        assert_eq!(hits, expected);

        let survivors = hits.len();
        assert!(survivors > 0, "fixture must leave survivors");
        assert_eq!(
            stats.exact_rows_read, survivors,
            "one row read per survivor: {stats:?}"
        );
        Ok(())
    }

    fn ground_truth_top_k(
        index: &Index,
        vec_field: Field,
        metric: Metric,
        query: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(Score, DocAddress)>> {
        let query = PreparedQuery::<f32>::new(metric, Arc::new(query.to_vec()));
        let searcher = index.reader()?.searcher();
        let mut scored = Vec::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            let vec_reader = segment_reader.vector_index(vec_field)?;
            let alive = segment_reader.alive_bitset();
            for doc in 0..segment_reader.max_doc() {
                if let Some(alive) = alive {
                    if !alive.is_alive(doc) {
                        continue;
                    }
                }
                if let Some(bytes) = vec_reader.vector_bytes(doc)? {
                    scored.push((
                        query.score_doc_bytes(&bytes),
                        DocAddress::new(seg_ord as u32, doc),
                    ));
                }
            }
        }
        scored.sort_by(|a: &(Score, DocAddress), b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.segment_ord.cmp(&b.1.segment_ord))
                .then(a.1.doc_id.cmp(&b.1.doc_id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    fn collect_filter_doc_set(
        index: &Index,
        filter: &dyn Query,
    ) -> crate::Result<std::collections::HashSet<DocAddress>> {
        let searcher = index.reader()?.searcher();
        let weight = filter.weight(EnableScoring::disabled_from_searcher(&searcher))?;
        let mut admitted = std::collections::HashSet::new();
        for (seg_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            weight.for_each_no_score(segment_reader, &mut |docs| {
                for &d in docs {
                    admitted.insert(DocAddress::new(seg_ord as u32, d));
                }
            })?;
        }
        Ok(admitted)
    }

    fn stored_label_at(
        index: &Index,
        label_field: Field,
        addr: DocAddress,
    ) -> crate::Result<String> {
        use crate::schema::Value;
        let searcher = index.reader()?.searcher();
        let doc = searcher.doc::<TantivyDocument>(addr)?;
        Ok(doc
            .get_first(label_field)
            .and_then(|v| Value::as_str(&v))
            .expect("stored label")
            .to_string())
    }

    fn grid2d_first_centroid() -> [f32; 2] {
        [0.0, 0.0]
    }

    mod bounds_gate_tests {
        use super::*;
        use crate::vector::bounds::{HeapPeek, QueryBound, QueryBoundTracker};
        use crate::vector::{margin_ball_ball, margin_ball_halfspace, to_bound_space};

        pub(super) fn single_segment_fixture(
            metric: Metric,
            centroids: &[[f32; 2]],
            docs: &[[f32; 2]],
        ) -> crate::Result<(Index, Field)> {
            let mut sb = Schema::builder();
            let embed_field = sb.add_vector_field(
                "embedding",
                VectorOptions::new(2, metric).with_dtype(VectorDType::F32),
            );
            sb.add_text_field("label", STRING | STORED);
            let schema = sb.build();
            let settings = IndexSettings {
                vector_clustering_threshold: 1,
                ..IndexSettings::default()
            };
            let index = Index::builder()
                .schema(schema)
                .settings(settings)
                .ivf_clusterer(Arc::new(InlineClusterer {
                    centroids: centroids.to_vec(),
                }))
                .create_in_ram()?;
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000)?;
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for (i, vector) in docs.iter().enumerate() {
                let mut doc = TantivyDocument::new();
                doc.add_text(index.schema().get_field("label").unwrap(), format!("d{i}"));
                doc.add_vector(embed_field, vector.as_slice());
                writer.add_document(doc)?;
            }
            writer.commit()?;
            let segment_ids = index.searchable_segment_ids()?;
            assert_eq!(segment_ids.len(), 1, "single flat segment");
            writer.merge(&segment_ids).wait()?;
            writer.wait_merging_threads()?;
            Ok((index, embed_field))
        }

        struct Lcg(u64);
        impl Lcg {
            fn next_f32(&mut self) -> f32 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 16.0
            }
            fn point(&mut self) -> [f32; 2] {
                loop {
                    let p = [self.next_f32(), self.next_f32()];
                    if p[0] * p[0] + p[1] * p[1] > 0.25 {
                        return p;
                    }
                }
            }
        }

        /// PROPERTY TEST — the home-cluster closure theorem. Random data x
        /// {L2, cosine, dot}; brute-force top-k vs gated top-k under a
        /// full budget. Asserts:
        ///
        /// (a) the result sets are identical — a wrong skip would lose a
        ///     true member;
        /// (b) no true member's HOME cluster is skippable even at the
        ///     FINAL (tightest) bound: `margin(t_final) >= 0`. The
        ///     running bound is never tighter than the final one (the kth
        ///     only improves), so this proves the home cluster probed at
        ///     every point of the scan — the bound fold covers every row,
        ///     so a qualifying row's cluster always fails the skip test.
        #[test]
        fn closure_no_true_member_skipped() -> crate::Result<()> {
            let k = 5;
            for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
                for seed in [11u64, 29, 47] {
                    let mut rng = Lcg(seed ^ (metric as u64) << 32);
                    let centroids: Vec<[f32; 2]> = (0..5).map(|_| rng.point()).collect();
                    let docs: Vec<[f32; 2]> = (0..40).map(|_| rng.point()).collect();
                    let (index, field) = single_segment_fixture(metric, &centroids, &docs)?;
                    let query: Vec<f32> = rng.point().to_vec();

                    let brute = ground_truth_top_k(&index, field, metric, &query, k)?;
                    let gated = search(
                        &index,
                        field,
                        &AllQuery,
                        query.clone(),
                        k,
                        exhaustive_params(centroids.len()),
                    )?;
                    assert_eq!(
                        gated, brute,
                        "{metric:?} seed {seed}: gated top-k must equal brute force"
                    );

                    let searcher = index.reader()?.searcher();
                    let segment_reader = &searcher.segment_readers()[0];
                    let vec_reader = segment_reader.vector_index(field)?;
                    let ivf = vec_reader.index().expect("IVF segment");
                    let bounds = ivf.bounds();
                    let centroid_bytes = ivf.centroid_bytes()?;
                    let stride = 2 * std::mem::size_of::<f32>();
                    let kth_key = brute[k - 1].0;
                    let t_final = to_bound_space(metric, kth_key);
                    let q_norm = (query[0] * query[0] + query[1] * query[1]).sqrt();
                    for &(_, addr) in &brute {
                        let stored = decode_2d(
                            &vec_reader
                                .vector_bytes(addr.doc_id)?
                                .expect("stored vector"),
                        );
                        let home = nearest_centroid(stored, &centroids);
                        let sim = Metric::similarity_bytes::<f32>(
                            metric,
                            &query,
                            &centroid_bytes[home * stride..(home + 1) * stride],
                        );
                        let r = bounds.ball_r(home);
                        let margin = match metric {
                            Metric::L2 | Metric::Cosine => {
                                margin_ball_ball(t_final, r, to_bound_space(metric, sim.score()))
                            }
                            Metric::Dot => margin_ball_halfspace(sim.score(), q_norm, r, t_final),
                        };
                        assert!(
                            margin >= 0.0,
                            "{metric:?} seed {seed}: true member {} home cluster {home} must \
                             never be skippable (margin {margin})",
                            addr.doc_id
                        );
                    }
                }
            }
            Ok(())
        }

        #[test]
        fn boundary_tie_probes_l2() -> crate::Result<()> {
            let centroids = [[0.0f32, 4.0], [4.0, 0.0], [8.0, 0.0]];
            let docs = [[0.0f32, 2.0], [2.0, 0.0], [7.0, 0.0]];
            let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs)?;
            let query = vec![0.0f32, 0.0];

            let brute = ground_truth_top_k(&index, field, Metric::L2, &query, 1)?;
            assert_eq!(
                brute[0].1.doc_id, 0,
                "tie at d = 2 breaks to the lower doc id"
            );

            let (hits, stats) = run_top_n(&index, field, query, 1, exhaustive_params(3))?;
            assert_eq!(
                hits, brute,
                "exact-touch cluster must probe, preserving the tie"
            );
            assert_eq!(
                stats.candidates_scored, 2,
                "both d = 2 docs are scored - the margin == 0 cluster probed"
            );
            assert_eq!(
                stats.clusters_probed(),
                2,
                "the disjoint cluster is skipped, the touching one is not"
            );
            assert_eq!(stats.termination, ProbeTermination::Exhausted);
            Ok(())
        }

        #[test]
        fn boundary_tie_probes_dot() -> crate::Result<()> {
            let centroids = [[2.0f32, 0.0], [4.0, 4.0]];
            let docs = [[3.0f32, 0.0], [3.0, 4.0]];
            let (index, field) = single_segment_fixture(Metric::Dot, &centroids, &docs)?;
            let query = vec![1.0f32, 0.0];

            let brute = ground_truth_top_k(&index, field, Metric::Dot, &query, 1)?;
            assert_eq!(
                brute[0].1.doc_id, 0,
                "score tie at 3 breaks to the lower doc id"
            );

            let (hits, stats) = run_top_n(&index, field, query, 1, exhaustive_params(2))?;
            assert_eq!(hits, brute);
            assert_eq!(
                stats.candidates_scored, 2,
                "the margin == 0 cluster probed and scored its tied doc"
            );
            Ok(())
        }

        #[test]
        fn skips_charge_open_share() -> crate::Result<()> {
            let centroids: Vec<[f32; 2]> = vec![
                [1.0, 1.0],
                [11.0, 1.0],
                [21.0, 1.0],
                [1.0, 11.0],
                [11.0, 11.0],
                [21.0, 11.0],
            ];
            let docs: Vec<[f32; 2]> = (0..36)
                .map(|i| {
                    let c = centroids[i / 6];
                    let off = (i % 6) as f32 * 0.01;
                    [c[0] + off, c[1] + off]
                })
                .collect();
            let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;

            assert_eq!(stats.termination, ProbeTermination::Exhausted);
            assert_eq!(
                stats.clusters_probed(),
                1,
                "only the home cluster survives the margins: {stats:?}"
            );
            assert_eq!(stats.candidates_scored, 6);
            let skipped = 6 - stats.clusters_probed();
            let n_avg = 36.0f64 / 6.0;
            let x = open_share(n_avg);
            let row = (1.0 - x) / n_avg;
            let expected = (stats.clusters_probed() + skipped) as f64 * x
                + stats.candidates_scored as f64 * row;
            assert!(
                (stats.work_charged as f64 - expected).abs() < 1e-5,
                "skips must charge the open share: expected {expected}, got {}",
                stats.work_charged
            );
            Ok(())
        }

        #[test]
        fn unarmed_never_skips() -> crate::Result<()> {
            let centroids: Vec<[f32; 2]> = vec![[1.0, 1.0], [11.0, 1.0], [21.0, 1.0]];
            let docs: Vec<[f32; 2]> = (0..9)
                .map(|i| {
                    let c = centroids[i / 3];
                    let off = (i % 3) as f32 * 0.01;
                    [c[0] + off, c[1] + off]
                })
                .collect();
            let (index, field) = single_segment_fixture(Metric::L2, &centroids, &docs)?;
            // k = 100 > 9 docs: the heap can never hold k results.
            let (hits, stats) =
                run_top_n(&index, field, vec![0.2, 0.3], 100, exhaustive_params(3))?;
            assert_eq!(hits.len(), 9, "every doc is a hit at k > N");
            assert_eq!(
                stats.clusters_probed(),
                3,
                "unarmed, every cluster probes: {stats:?}"
            );
            assert_eq!(stats.candidates_scored, 9);
            let n_avg = 3.0f64;
            let x = open_share(n_avg);
            let row = (1.0 - x) / n_avg;
            let expected = 3.0 * x + 9.0 * row;
            assert!(
                (stats.work_charged as f64 - expected).abs() < 1e-5,
                "no skip term in the identity: expected {expected}, got {}",
                stats.work_charged
            );
            Ok(())
        }

        #[test]
        fn t_maintenance_per_metric() {
            let cases = [
                (Metric::L2, -4.0f32, 2.0f32, -1.0f32, 1.0f32),
                (Metric::Cosine, 0.5, 1.0, 0.875, 0.5),
                (Metric::Dot, 3.0, 3.0, 4.5, 4.5),
            ];
            for (metric, first, t_first, improved, t_improved) in cases {
                let mut tracker = QueryBoundTracker::new();
                assert_eq!(tracker.bound(), QueryBound::Filling);
                tracker.observe(metric, HeapPeek::Filling, 0);
                assert_eq!(
                    tracker.bound(),
                    QueryBound::Filling,
                    "{metric:?}: no arm on Filling"
                );

                tracker.observe(metric, HeapPeek::Full { kth_key: first }, 1);
                assert_eq!(
                    tracker.bound(),
                    QueryBound::Armed { t: t_first },
                    "{metric:?}: t from first kth"
                );
                tracker.observe(metric, HeapPeek::Full { kth_key: first }, 2);
                assert_eq!(tracker.bound(), QueryBound::Armed { t: t_first });

                tracker.observe(metric, HeapPeek::Full { kth_key: improved }, 3);
                assert_eq!(
                    tracker.bound(),
                    QueryBound::Armed { t: t_improved },
                    "{metric:?}: t tracks the improvement"
                );
            }
        }

        #[test]
        fn armed_index_recorded() {
            let mut tracker = QueryBoundTracker::new();
            tracker.observe(Metric::L2, HeapPeek::Filling, 0);
            tracker.observe(Metric::L2, HeapPeek::Filling, 1);
            assert_eq!(tracker.armed_at_probe(), None, "unarmed while filling");
            tracker.observe(Metric::L2, HeapPeek::Full { kth_key: -1.0 }, 2);
            assert_eq!(tracker.armed_at_probe(), Some(2), "arms at the first Full");
            tracker.observe(Metric::L2, HeapPeek::Full { kth_key: -0.5 }, 3);
            assert_eq!(
                tracker.armed_at_probe(),
                Some(2),
                "later improvements never move the armed index"
            );
        }
    }

    mod bounds_stats_tests {
        use super::bounds_gate_tests::single_segment_fixture;
        use super::*;

        fn separated_fixture(metric: Metric) -> crate::Result<(Index, Field)> {
            let centroids: Vec<[f32; 2]> = vec![
                [1.0, 1.0],
                [11.0, 1.0],
                [21.0, 1.0],
                [1.0, 11.0],
                [11.0, 11.0],
                [21.0, 11.0],
            ];
            let docs: Vec<[f32; 2]> = (0..36)
                .map(|i| {
                    let c = centroids[i / 6];
                    let off = (i % 6) as f32 * 0.01;
                    [c[0] + off, c[1] + off]
                })
                .collect();
            single_segment_fixture(metric, &centroids, &docs)
        }

        #[test]
        fn skip_count_matches() -> crate::Result<()> {
            let (index, field) = separated_fixture(Metric::L2)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;
            assert_eq!(stats.termination, ProbeTermination::Exhausted);
            assert_eq!(stats.clusters_probed(), 1, "{stats:?}");
            assert_eq!(
                stats.bounds_skips, 5,
                "every non-home cluster is a counted skip: {stats:?}"
            );
            Ok(())
        }

        #[test]
        fn armed_null_when_unarmed() -> crate::Result<()> {
            let (index, field) = separated_fixture(Metric::L2)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 100, exhaustive_params(6))?;
            assert_eq!(stats.bounds_skips, 0);
            assert_eq!(stats.bound_armed_at_probe, None);
            let value = serde_json::to_value(&stats).expect("ProbeStats serializes");
            assert!(value["bound_armed_at_probe"].is_null());
            Ok(())
        }

        #[test]
        fn armed_index_value() -> crate::Result<()> {
            let (index, field) = separated_fixture(Metric::L2)?;
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 5, exhaustive_params(6))?;
            assert_eq!(stats.bound_armed_at_probe, Some(0), "{stats:?}");
            let (_, stats) = run_top_n(&index, field, vec![0.2, 0.3], 10, exhaustive_params(6))?;
            assert_eq!(
                stats.bound_armed_at_probe,
                Some(1),
                "k = 10 needs the second probed cluster: {stats:?}"
            );
            assert!(
                stats.bounds_skips > 0,
                "armed late still skips the far tail"
            );
            Ok(())
        }
    }
}
