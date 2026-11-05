//! Precision operations over vector banks. Sources expose data; these functions
//! prepare queries and score batches without owning retention or stopping.

use std::ops::Range;

use quant::cascade::{LayerKind, LayerSpec, Rotation};

use super::arena::VectorArena;
use super::candidate::Candidate;
use super::search::ScoreBuffer;
use super::source::VectorSource;
use super::stacked::levels::ClusterId;
use crate::schema::Metric;
use crate::vector::distance::Similarity;
use crate::vector::search::prepared::VectorQuery;
use crate::vector::VectorElement;

/// Produce scores for addressed rows in the source's initial representation.
/// Returns whether those scores are approximate. Identities remain unchanged.
pub(crate) fn score_initial<S: VectorSource, T: VectorElement>(
    source: &S,
    query: &VectorQuery<T>,
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
        let prepared = query.quantized_router_query(context.seed, &context.rotation);
        let batch = codes
            .read(rows.clone())
            .expect("failed to read quantized vector rows");
        out.estimates.resize(rows.len(), 0.0);
        prepared.codes.score_layer_batch_unscaled(
            0,
            batch.codes(),
            query.widened().len().div_ceil(64) * 8,
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
            out.candidates.push(Candidate {
                node: ClusterId::from(rows.start + i),
                sim: Similarity::new(score),
            });
        }
    } else {
        source
            .raw()
            .expect("a vector source must have a representation")
            .batch_similarity(
                query.widened(),
                query.metric(),
                query.routing_norm(),
                rows.clone(),
                &mut out.similarities,
            );
        out.candidates
            .extend(rows.zip(&out.similarities).map(|(row, &sim)| Candidate {
                node: ClusterId::from(row),
                sim,
            }));
    }
    approximate
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::vector::router::source::{EncodingContext, VectorSet, ROUTER_SEED};
    use crate::vector::router::stacked::levels::InMemoryStore;
    use crate::vector::storage::quantized::sign::SignVectors;

    #[test]
    fn quantized_batches_emit_addressed_candidates_without_similarity_scratch() -> crate::Result<()>
    {
        let dim = 8;
        let values: Vec<f32> = [0.0, 1.0, -2.0, 3.0]
            .into_iter()
            .flat_map(|v| vec![v; dim])
            .collect();
        let raw = Arc::new(InMemoryStore::new(values.clone(), dim));
        let context = Arc::new(EncodingContext::new(dim, ROUTER_SEED));
        let encoded = VectorSet::Encoded {
            codes: Arc::new(SignVectors::encode(&values, dim, &context.rotation)?),
            context,
            raw: Some(raw.clone()),
        };
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let query = VectorQuery::new(metric, Arc::new(vec![0.0; dim]));
            let mut buffer = ScoreBuffer::default();
            assert!(!score_initial(raw.as_ref(), &query, 0..4, &mut buffer));
            assert_eq!(buffer.similarities.len(), 4);
            assert!(score_initial(&encoded, &query, 1..3, &mut buffer));
            assert!(buffer.similarities.is_empty());
            let expected: Vec<_> = (1..3)
                .map(|row| Candidate {
                    node: ClusterId::from(row),
                    sim: metric.similarity(query.widened(), &values[row * dim..][..dim]),
                })
                .collect();
            assert_eq!(buffer.candidates, expected);
            assert!(score_initial(&encoded, &query, 2..2, &mut buffer));
            assert!(buffer.candidates.is_empty());
            assert!(buffer.estimates.is_empty());
            assert!(buffer.similarities.is_empty());
        }
        Ok(())
    }
}
