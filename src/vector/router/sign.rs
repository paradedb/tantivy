//! Persisted one-bit vector rows and their scalar distance estimator. No search
//! policy lives here: codes can score centroids or members at any level.

use std::io::{self, Write};
use std::ops::Range;

use common::BinarySerializable;
use quant::rotation::Rotation;
use quant::sign::{estimate_asym_batch_unscaled, pack, prepare_query, QueryPlanes};

use super::candidate::Candidate;
use super::search::ScoreBuffer;
use crate::schema::Metric;
use crate::vector::distance::Similarity;
use crate::vector::quantization::SIGN_QUERY_BITS;
use crate::vector::search::prepared::VectorQuery;
use crate::vector::VectorElement;

/// The format fixes zero centering, a seeded orthogonal rotation, and the
/// asymmetric sign estimator. Raw vectors are kept separately for exact reads.
pub(crate) struct SignVectors {
    dim: usize,
    codes: Vec<u64>,
    /// ||x||² / <Rx, sign(Rx)>. Unlike reconstruction's mean-absolute
    /// scale, this corrects the systematic shrinkage of estimated dot products.
    coefficients: Vec<f32>,
    norms_squared: Vec<f32>,
}

pub(crate) struct SignQuery {
    planes: QueryPlanes,
    norm_squared: f32,
    norm: f32,
    metric: Metric,
}

impl SignQuery {
    pub(crate) fn new<T: VectorElement>(query: &VectorQuery<T>, rotation: &Rotation) -> Self {
        let mut rotated = query.widened().to_vec();
        rotation.apply(&mut rotated);
        Self {
            planes: prepare_query(&rotated, SIGN_QUERY_BITS),
            norm_squared: query.widened().iter().map(|x| x * x).sum(),
            norm: query.routing_norm(),
            metric: query.metric(),
        }
    }
}

pub(crate) fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

impl SignVectors {
    pub(crate) fn encode(values: &[f32], dim: usize, rotation: &Rotation) -> io::Result<Self> {
        if dim == 0 || values.len() % dim != 0 {
            return Err(invalid("invalid sign-vector matrix shape"));
        }
        let rows = values.len() / dim;
        let words = dim.div_ceil(64);
        let mut codes = vec![
            0;
            rows.checked_mul(words)
                .ok_or_else(|| invalid("sign code size overflow"))?
        ];
        let mut coefficients = Vec::with_capacity(rows);
        let mut norms_squared = Vec::with_capacity(rows);
        let mut rotated = vec![0.0; dim];
        let mut scratch = vec![0.0; dim];
        for (row, values) in values.chunks_exact(dim).enumerate() {
            if values.iter().any(|x| !x.is_finite()) {
                return Err(invalid("sign-vector inputs must be finite"));
            }
            let norm_squared = values.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
            rotated.copy_from_slice(values);
            rotation.apply_with_scratch(&mut rotated, &mut scratch);
            let sum_abs = rotated.iter().map(|x| f64::from(x.abs())).sum::<f64>();
            let coefficient = if sum_abs == 0.0 || norm_squared as f32 == 0.0 {
                0.0
            } else {
                norm_squared / sum_abs
            } as f32;
            let norm_squared = norm_squared as f32;
            if !coefficient.is_finite() || !norm_squared.is_finite() {
                return Err(invalid("sign-vector norm exceeds the estimator's range"));
            }
            pack(&rotated, &mut codes[row * words..(row + 1) * words]);
            coefficients.push(coefficient);
            norms_squared.push(norm_squared);
        }
        Ok(Self {
            dim,
            codes,
            coefficients,
            norms_squared,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.coefficients.len()
    }

    pub(crate) fn score<I>(
        &self,
        rows: Range<usize>,
        query: &SignQuery,
        out: &mut ScoreBuffer<I>,
        mut identify: impl FnMut(usize) -> I,
    ) {
        let words = self.dim.div_ceil(64);
        out.estimates.resize(rows.len(), 0.0);
        estimate_asym_batch_unscaled(
            &self.codes[rows.start * words..rows.end * words],
            words,
            &query.planes,
            &mut out.estimates,
        );
        out.candidates.clear();
        out.candidates
            .extend(rows.zip(&out.estimates).map(|(row, &estimate)| {
                let dot = estimate * self.coefficients[row];
                let score = match query.metric {
                    Metric::L2 => -(query.norm_squared + self.norms_squared[row] - 2.0 * dot),
                    Metric::Dot => dot,
                    Metric::Cosine => {
                        let denominator = query.norm * self.norms_squared[row].sqrt();
                        if denominator == 0.0 {
                            0.0
                        } else {
                            dot / denominator
                        }
                    }
                };
                Candidate {
                    sim: Similarity::new(score),
                    node: identify(row),
                }
            }));
    }

    /// Columnar, fixed-stride representation. The containing format supplies
    /// dimension and rotation; all representations share canonical row numbers.
    pub(crate) fn serialize(&self, out: &mut (impl Write + ?Sized)) -> io::Result<()> {
        u32::try_from(self.len())
            .map_err(|_| invalid("too many sign rows"))?
            .serialize(out)?;
        for word in &self.codes {
            word.serialize(out)?;
        }
        for coefficient in &self.coefficients {
            coefficient.serialize(out)?;
        }
        for norm in &self.norms_squared {
            norm.serialize(out)?;
        }
        Ok(())
    }

    pub(crate) fn deserialize(input: &mut &[u8], dim: usize) -> io::Result<Self> {
        if dim == 0 {
            return Err(invalid("sign-vector dimension must be positive"));
        }
        let rows = u32::deserialize(input)? as usize;
        let words = dim.div_ceil(64);
        let count = rows
            .checked_mul(words)
            .ok_or_else(|| invalid("sign code size overflow"))?;
        let bytes = count
            .checked_mul(8)
            .and_then(|n| rows.checked_mul(8).and_then(|m| n.checked_add(m)))
            .ok_or_else(|| invalid("sign-vector size overflow"))?;
        if input.len() < bytes {
            return Err(invalid("truncated sign-vector columns"));
        }
        let codes = (0..count)
            .map(|_| u64::deserialize(input))
            .collect::<io::Result<Vec<_>>>()?;
        let coefficients = (0..rows)
            .map(|_| f32::deserialize(input))
            .collect::<io::Result<Vec<_>>>()?;
        let norms_squared = (0..rows)
            .map(|_| f32::deserialize(input))
            .collect::<io::Result<Vec<_>>>()?;
        if coefficients
            .iter()
            .chain(&norms_squared)
            .any(|x| !x.is_finite() || *x < 0.0)
            || coefficients
                .iter()
                .zip(&norms_squared)
                .any(|(&scale, &norm)| (scale == 0.0) != (norm == 0.0))
        {
            return Err(invalid("invalid sign-vector sidecars"));
        }
        if dim % 64 != 0
            && codes
                .chunks_exact(words)
                .any(|row| row[words - 1] >> (dim % 64) != 0)
        {
            return Err(invalid("nonzero sign-code tail padding"));
        }
        Ok(Self {
            dim,
            codes,
            coefficients,
            norms_squared,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::vector::router::stacked::levels::ClusterId;

    #[test]
    fn sign_columns_roundtrip_and_validate_framing_and_tail_bits() -> io::Result<()> {
        let dim = 65;
        let rotation = Rotation::new(dim, 7);
        let values: Vec<_> = (0..dim * 3).map(|i| (i as f32 * 0.37).sin()).collect();
        let vectors = SignVectors::encode(&values, dim, &rotation)?;
        let mut bytes = Vec::new();
        vectors.serialize(&mut bytes)?;
        let opened = SignVectors::deserialize(&mut bytes.as_slice(), dim)?;
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let query = VectorQuery::new(metric, Arc::new(values[..dim].to_vec()));
            let query = SignQuery::new(&query, &rotation);
            let mut before = ScoreBuffer::default();
            let mut after = ScoreBuffer::default();
            vectors.score(0..3, &query, &mut before, ClusterId::from);
            opened.score(0..3, &query, &mut after, ClusterId::from);
            assert_eq!(before.candidates, after.candidates);
            assert!(after.candidates.iter().all(|c| c.sim.score().is_finite()));
        }
        for end in 0..bytes.len() {
            assert!(SignVectors::deserialize(&mut &bytes[..end], dim).is_err());
        }
        let mut bad_tail = bytes.clone();
        bad_tail[4 + 8 + 7] |= 0x80;
        assert!(SignVectors::deserialize(&mut bad_tail.as_slice(), dim).is_err());
        let scales_start = 4 + 3 * dim.div_ceil(64) * 8;
        bytes[scales_start..scales_start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(SignVectors::deserialize(&mut bytes.as_slice(), dim).is_err());
        Ok(())
    }

    #[test]
    fn zero_vectors_and_zero_queries_have_finite_estimates() -> io::Result<()> {
        let rotation = Rotation::new(64, 17);
        let vectors = SignVectors::encode(&vec![0.0; 128], 64, &rotation)?;
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let query = VectorQuery::new(metric, Arc::new(vec![0.0; 64]));
            let query = SignQuery::new(&query, &rotation);
            let mut out = ScoreBuffer::default();
            vectors.score(0..2, &query, &mut out, ClusterId::from);
            assert!(out.candidates.iter().all(|c| c.sim.score() == 0.0));
        }
        Ok(())
    }
}
