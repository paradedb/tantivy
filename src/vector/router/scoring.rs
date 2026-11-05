//! Precision operations over vector banks. Sources expose data; these functions
//! prepare queries and score batches without owning retention or stopping.

use std::cell::{Ref, RefCell};
use std::collections::BTreeMap;
use std::ops::Range;

use quant::cascade::{LayerKind, LayerSpec, Rotation};

use super::arena::VectorArena;
use super::candidate::Candidate;
use super::search::ScoreBuffer;
use super::source::{EncodingContext, VectorSource};
use super::stacked::levels::ClusterId;
use crate::schema::Metric;
use crate::vector::distance::Similarity;
use crate::vector::search::prepared::{QuantizedVectorQuery, VectorQuery};
use crate::vector::VectorElement;

/// One base query, prepared lazily once for each rotation seed used by its segments.
pub(crate) struct ScoringQuery<'q, T: VectorElement> {
    pub(crate) query: &'q VectorQuery<T>,
    encoded: RefCell<BTreeMap<u64, QuantizedVectorQuery>>,
}

impl<'q, T: VectorElement> ScoringQuery<'q, T> {
    pub(crate) fn new(query: &'q VectorQuery<T>) -> Self {
        Self {
            query,
            encoded: RefCell::new(BTreeMap::new()),
        }
    }
    fn prepared(&self, encoding: &EncodingContext) -> Ref<'_, QuantizedVectorQuery> {
        self.encoded
            .borrow_mut()
            .entry(encoding.seed)
            .or_insert_with(|| QuantizedVectorQuery::for_sign(self.query, &encoding.rotation));
        Ref::map(self.encoded.borrow(), |prepared| &prepared[&encoding.seed])
    }
}

/// Produce scores for addressed rows in the source's initial representation.
/// Returns whether those scores are approximate. Identities remain unchanged.
pub(crate) fn score_initial<S: VectorSource, T: VectorElement>(
    source: &S,
    query: &ScoringQuery<'_, T>,
    rows: Range<usize>,
    out: &mut ScoreBuffer<ClusterId>,
) -> bool {
    out.candidates.clear();
    out.similarities.clear();
    out.estimates.clear();
    let approximate = source.compressed().is_some();
    if rows.is_empty() {
        return approximate;
    }
    if let Some((codes, context)) = source.compressed() {
        let prepared = query.prepared(context);
        let batch = codes
            .read(rows.clone())
            .expect("failed to read quantized vector rows");
        out.estimates.resize(rows.len(), 0.0);
        prepared.codes.score_layer_batch_unscaled(
            0,
            batch.codes(),
            query.query.widened().len().div_ceil(64) * 8,
            LayerSpec {
                kind: LayerKind::Sign,
                bits: 1,
                rotation: Rotation::None,
            },
            &mut out.estimates,
        );
        for (i, &estimate) in out.estimates.iter().enumerate() {
            let (coefficient, norm_squared) =
                batch.sidecar(i).expect("invalid quantized vector sidecars");
            let dot = estimate * coefficient;
            let score = match prepared.metric {
                Metric::L2 => -(prepared.norm_squared + norm_squared - 2.0 * dot),
                Metric::Dot => dot,
                Metric::Cosine => {
                    let denominator = prepared.norm * norm_squared.sqrt();
                    if denominator == 0.0 {
                        0.0
                    } else {
                        dot / denominator
                    }
                }
            };
            out.similarities.push(Similarity::new(score));
        }
    } else {
        source
            .raw()
            .expect("a vector source must have a representation")
            .batch_similarity(
                query.query.widened(),
                query.query.metric(),
                query.query.routing_norm(),
                rows.clone(),
                &mut out.similarities,
            );
    }
    out.candidates
        .extend(rows.zip(&out.similarities).map(|(row, &sim)| Candidate {
            node: ClusterId::from(row),
            sim,
        }));
    approximate
}
