//! Persisted one-bit vector rows and their scalar sidecars. Reads expose and
//! validate representations without query preparation or scoring policy.

use std::io::{self, Write};
use std::ops::Range;

use common::{BinarySerializable, HasLen, OwnedBytes};
use quant::cascade::QueryRotationPlan;
use quant::sign::encode_unbiased;

use crate::directory::FileSlice;

/// The format fixes zero centering, a seeded orthogonal rotation, and the
/// asymmetric sign estimator. Raw vectors are kept separately for exact reads.
pub(crate) struct SignVectors {
    dim: usize,
    rows: usize,
    storage: SignStorage,
}

enum SignStorage {
    Owned {
        codes: Vec<u8>,
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

pub(crate) fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

impl SignVectors {
    pub(crate) fn encode(values: &[f32], dim: usize, plan: &QueryRotationPlan) -> io::Result<Self> {
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
            plan.rotate_layer_in_place(0, &mut rotated, &mut scratch);
            let coefficient = encode_unbiased(
                &rotated,
                norm_squared,
                &mut codes[row * words..(row + 1) * words],
            );
            let norm_squared = norm_squared as f32;
            if !coefficient.is_finite() || !norm_squared.is_finite() {
                return Err(invalid("sign-vector norm exceeds the estimator's range"));
            }
            coefficients.push(coefficient);
            norms_squared.push(norm_squared);
        }
        Ok(Self {
            dim,
            rows,
            storage: SignStorage::Owned {
                codes: codes.into_iter().flat_map(u64::to_le_bytes).collect(),
                coefficients,
                norms_squared,
            },
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.rows
    }

    /// Pins only the selected code and sidecar ranges. Validation is deferred
    /// until those rows are read; no query or search policy is needed here.
    pub(crate) fn read(&self, rows: Range<usize>) -> io::Result<SignBatch<'_>> {
        assert!(rows.start <= rows.end && rows.end <= self.rows);
        if rows.is_empty() {
            return Ok(SignBatch::Borrowed {
                codes: &[],
                coefficients: &[],
                norms: &[],
            });
        }
        let words = self.dim.div_ceil(64);
        let batch = match &self.storage {
            SignStorage::Owned {
                codes,
                coefficients,
                norms_squared,
            } => SignBatch::Borrowed {
                codes: &codes[rows.start * words * 8..rows.end * words * 8],
                coefficients: &coefficients[rows.clone()],
                norms: &norms_squared[rows.clone()],
            },
            SignStorage::File {
                codes,
                coefficients,
                norms_squared,
            } => SignBatch::Pinned {
                codes: codes
                    .slice(rows.start * words * 8..rows.end * words * 8)
                    .read_bytes()?,
                coefficients: coefficients
                    .slice(rows.start * 4..rows.end * 4)
                    .read_bytes()?,
                norms: norms_squared
                    .slice(rows.start * 4..rows.end * 4)
                    .read_bytes()?,
            },
        };
        if self.dim % 64 != 0
            && batch.codes().chunks_exact(words * 8).any(|code| {
                !crate::vector::quantization::quantized_code_tail_is_zero(code, self.dim, 1)
            })
        {
            return Err(invalid("nonzero sign-code tail padding"));
        }
        Ok(batch)
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
                out.write_all(codes)?;
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

/// A representation batch, borrowing owned rows or pinning selected file bytes.
pub(crate) enum SignBatch<'a> {
    Borrowed {
        codes: &'a [u8],
        coefficients: &'a [f32],
        norms: &'a [f32],
    },
    Pinned {
        codes: OwnedBytes,
        coefficients: OwnedBytes,
        norms: OwnedBytes,
    },
}

impl SignBatch<'_> {
    pub(crate) fn codes(&self) -> &[u8] {
        match self {
            Self::Borrowed { codes, .. } => codes,
            Self::Pinned { codes, .. } => codes,
        }
    }
    /// Decode and validate the addressed sidecars when consumed, so scoring
    /// need not traverse these columns a second time.
    #[inline]
    pub(crate) fn sidecar(&self, row: usize) -> io::Result<(f32, f32)> {
        let (coefficient, norm) = match self {
            Self::Borrowed {
                coefficients,
                norms,
                ..
            } => (coefficients[row], norms[row]),
            Self::Pinned {
                coefficients,
                norms,
                ..
            } => {
                let at = row * 4;
                (
                    f32::from_le_bytes(coefficients[at..at + 4].try_into().unwrap()),
                    f32::from_le_bytes(norms[at..at + 4].try_into().unwrap()),
                )
            }
        };
        if !coefficient.is_finite()
            || coefficient < 0.0
            || !norm.is_finite()
            || norm < 0.0
            || (coefficient == 0.0) != (norm == 0.0)
        {
            return Err(invalid("invalid sign-vector sidecars"));
        }
        Ok((coefficient, norm))
    }
}

#[cfg(test)]
mod tests {
    use quant::cascade::{LayerKind, LayerSpec, Rotation};

    use super::*;

    fn rotation_plan(dim: usize, seed: u64) -> QueryRotationPlan {
        QueryRotationPlan::new(
            dim,
            &[LayerSpec {
                kind: LayerKind::Sign,
                bits: 1,
                rotation: Rotation::SeededFhtChaCha8 { seed },
            }],
        )
    }

    #[test]
    fn sign_columns_roundtrip_and_validate_framing_and_tail_bits() -> io::Result<()> {
        let dim = 65;
        let rotation = rotation_plan(dim, 7);
        let values: Vec<_> = (0..dim * 3).map(|i| (i as f32 * 0.37).sin()).collect();
        let vectors = SignVectors::encode(&values, dim, &rotation)?;
        let mut bytes = Vec::new();
        vectors.serialize(&mut bytes)?;
        let opened = SignVectors::open(&mut FileSlice::from(bytes.clone()), dim)?;
        let before = vectors.read(0..3)?;
        let after = opened.read(0..3)?;
        assert_eq!(before.codes(), after.codes());
        for i in 0..3 {
            assert_eq!(before.sidecar(i)?, after.sidecar(i)?);
        }
        for end in 0..bytes.len() {
            assert!(SignVectors::open(&mut FileSlice::from(bytes[..end].to_vec()), dim).is_err());
        }
        let mut bad_tail = bytes.clone();
        bad_tail[4 + 8 + 7] |= 0x80;
        let corrupt = SignVectors::open(&mut FileSlice::from(bad_tail), dim)?;
        corrupt.read(1..3)?;
        assert!(corrupt.read(0..1).is_err());
        let scales_start = 4 + 3 * dim.div_ceil(64) * 8;
        bytes[scales_start..scales_start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let corrupt = SignVectors::open(&mut FileSlice::from(bytes), dim)?;
        corrupt.read(1..3)?.sidecar(0)?;
        assert!(corrupt.read(0..1)?.sidecar(0).is_err());
        Ok(())
    }

    #[test]
    fn zero_vector_storage_preserves_zero_codes_and_sidecars() -> io::Result<()> {
        let vectors = SignVectors::encode(&vec![0.0; 128], 64, &rotation_plan(64, 17))?;
        let batch = vectors.read(0..2)?;
        assert!(batch.codes().iter().all(|&b| b == 0));
        assert_eq!(batch.sidecar(0)?, (0.0, 0.0));
        assert_eq!(batch.sidecar(1)?, (0.0, 0.0));
        Ok(())
    }
}
