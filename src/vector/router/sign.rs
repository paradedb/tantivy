//! Persisted one-bit vector rows and their scalar distance estimator. No search
//! policy lives here: codes can score centroids or members at any level.

use std::io::{self, Write};
use std::ops::Range;

use common::{BinarySerializable, HasLen};
use quant::rotation::Rotation;
use quant::sign::{
    estimate_asym_batch_unscaled, pack, prepare_query, words_from_le_bytes, QueryPlanes,
};

use super::candidate::Candidate;
use super::search::ScoreBuffer;
use crate::directory::FileSlice;
use crate::schema::Metric;
use crate::vector::distance::Similarity;
use crate::vector::quantization::SIGN_QUERY_BITS;
use crate::vector::search::prepared::VectorQuery;
use crate::vector::VectorElement;

/// The format fixes zero centering, a seeded orthogonal rotation, and the
/// asymmetric sign estimator. Raw vectors are kept separately for exact reads.
pub(crate) struct SignVectors {
    dim: usize,
    rows: usize,
    storage: SignStorage,
}

enum SignStorage {
    Owned {
        codes: Vec<u64>,
        /// ||x||² / <Rx, sign(Rx)>. Unlike reconstruction's mean-absolute
        /// scale, this corrects the systematic shrinkage of estimated dot products.
        coefficients: Vec<f32>,
        norms_squared: Vec<f32>,
    },
    /// Each column is addressed without reading its contents at open.
    File {
        codes: FileSlice,
        coefficients: FileSlice,
        norms_squared: FileSlice,
    },
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
            rows,
            storage: SignStorage::Owned {
                codes,
                coefficients,
                norms_squared,
            },
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.rows
    }

    pub(crate) fn score<I>(
        &self,
        rows: Range<usize>,
        query: &SignQuery,
        out: &mut ScoreBuffer<I>,
        identify: impl FnMut(usize) -> I,
    ) {
        self.try_score(rows, query, out, identify)
            .expect("failed to read quantized router rows");
    }

    fn try_score<I>(
        &self,
        rows: Range<usize>,
        query: &SignQuery,
        out: &mut ScoreBuffer<I>,
        identify: impl FnMut(usize) -> I,
    ) -> io::Result<()> {
        assert!(rows.start <= rows.end && rows.end <= self.rows);
        out.candidates.clear();
        out.estimates.clear();
        if rows.is_empty() {
            return Ok(());
        }
        let words = self.dim.div_ceil(64);
        match &self.storage {
            SignStorage::Owned {
                codes,
                coefficients,
                norms_squared,
            } => self.score_codes(
                rows.clone(),
                &codes[rows.start * words..rows.end * words],
                query,
                out,
                identify,
                |i| (coefficients[rows.start + i], norms_squared[rows.start + i]),
            ),
            SignStorage::File {
                codes,
                coefficients,
                norms_squared,
            } => {
                let codes = codes
                    .slice(rows.start * words * 8..rows.end * words * 8)
                    .read_bytes()?;
                let coefficients = coefficients
                    .slice(rows.start * 4..rows.end * 4)
                    .read_bytes()?;
                let norms = norms_squared
                    .slice(rows.start * 4..rows.end * 4)
                    .read_bytes()?;
                let words = words_from_le_bytes(&codes);
                self.score_codes(rows, &words, query, out, identify, |i| {
                    let at = i * 4;
                    (
                        f32::from_le_bytes(coefficients[at..at + 4].try_into().unwrap()),
                        f32::from_le_bytes(norms[at..at + 4].try_into().unwrap()),
                    )
                })
            }
        }
    }

    fn score_codes<I>(
        &self,
        rows: Range<usize>,
        codes: &[u64],
        query: &SignQuery,
        out: &mut ScoreBuffer<I>,
        mut identify: impl FnMut(usize) -> I,
        sidecars: impl Fn(usize) -> (f32, f32),
    ) -> io::Result<()> {
        let words = self.dim.div_ceil(64);
        if self.dim % 64 != 0
            && codes
                .chunks_exact(words)
                .any(|row| row[words - 1] >> (self.dim % 64) != 0)
        {
            return Err(invalid("nonzero sign-code tail padding"));
        }
        out.estimates.resize(rows.len(), 0.0);
        estimate_asym_batch_unscaled(codes, words, &query.planes, &mut out.estimates);
        for (i, (row, &estimate)) in rows.zip(&out.estimates).enumerate() {
            let (coefficient, norm_squared) = sidecars(i);
            if !coefficient.is_finite()
                || coefficient < 0.0
                || !norm_squared.is_finite()
                || norm_squared < 0.0
                || (coefficient == 0.0) != (norm_squared == 0.0)
            {
                return Err(invalid("invalid sign-vector sidecars"));
            }
            let dot = estimate * coefficient;
            let score = match query.metric {
                Metric::L2 => -(query.norm_squared + norm_squared - 2.0 * dot),
                Metric::Dot => dot,
                Metric::Cosine => {
                    let denominator = query.norm * norm_squared.sqrt();
                    if denominator == 0.0 {
                        0.0
                    } else {
                        dot / denominator
                    }
                }
            };
            out.candidates.push(Candidate {
                sim: Similarity::new(score),
                node: identify(row),
            });
        }
        Ok(())
    }

    /// Columnar, fixed-stride representation. The containing format supplies
    /// dimension and rotation; all representations share canonical row numbers.
    pub(crate) fn serialize(&self, out: &mut (impl Write + ?Sized)) -> io::Result<()> {
        u32::try_from(self.len())
            .map_err(|_| invalid("too many sign rows"))?
            .serialize(out)?;
        match &self.storage {
            SignStorage::Owned {
                codes,
                coefficients,
                norms_squared,
            } => {
                for word in codes {
                    word.serialize(out)?;
                }
                for coefficient in coefficients {
                    coefficient.serialize(out)?;
                }
                for norm in norms_squared {
                    norm.serialize(out)?;
                }
            }
            SignStorage::File {
                codes,
                coefficients,
                norms_squared,
            } => {
                for column in [codes, coefficients, norms_squared] {
                    out.write_all(&column.read_bytes()?)?;
                }
            }
        }
        Ok(())
    }

    /// Reads only the row count and validates framing. Code and sidecar validity
    /// is checked when a scoring batch reads those rows, never at open.
    pub(crate) fn open(input: &mut FileSlice, dim: usize) -> io::Result<Self> {
        if dim == 0 || input.len() < 4 {
            return Err(invalid("missing sign-vector header or dimension"));
        }
        let header = input.slice_to(4).read_bytes()?;
        let rows = u32::deserialize(&mut header.as_slice())? as usize;
        let code_bytes = rows
            .checked_mul(dim.div_ceil(64))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| invalid("sign code size overflow"))?;
        let sidecar_bytes = rows
            .checked_mul(4)
            .ok_or_else(|| invalid("sign sidecar size overflow"))?;
        let total = sidecar_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(code_bytes))
            .ok_or_else(|| invalid("sign-vector size overflow"))?;
        let data = input.slice_from(4);
        if data.len() < total {
            return Err(invalid("truncated sign-vector columns"));
        }
        let storage = SignStorage::File {
            codes: data.slice_to(code_bytes),
            coefficients: data.slice(code_bytes..code_bytes + sidecar_bytes),
            norms_squared: data.slice(code_bytes + sidecar_bytes..total),
        };
        *input = data.slice_from(total);
        Ok(Self { dim, rows, storage })
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
        let opened = SignVectors::open(&mut FileSlice::from(bytes.clone()), dim)?;
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
            assert!(SignVectors::open(&mut FileSlice::from(bytes[..end].to_vec()), dim).is_err());
        }
        let mut bad_tail = bytes.clone();
        bad_tail[4 + 8 + 7] |= 0x80;
        let corrupt = SignVectors::open(&mut FileSlice::from(bad_tail), dim)?;
        let query = SignQuery::new(
            &VectorQuery::new(Metric::Dot, Arc::new(values[..dim].to_vec())),
            &rotation,
        );
        let mut scores = ScoreBuffer::default();
        // Invalid bytes outside the selected range are not read or validated.
        corrupt.try_score(1..3, &query, &mut scores, ClusterId::from)?;
        assert!(corrupt
            .try_score(0..1, &query, &mut scores, ClusterId::from)
            .is_err());
        let scales_start = 4 + 3 * dim.div_ceil(64) * 8;
        bytes[scales_start..scales_start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let corrupt = SignVectors::open(&mut FileSlice::from(bytes), dim)?;
        corrupt.try_score(1..3, &query, &mut scores, ClusterId::from)?;
        assert!(corrupt
            .try_score(0..1, &query, &mut scores, ClusterId::from)
            .is_err());
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
