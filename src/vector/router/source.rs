//! Vector-bank representations and their shared encoding context. No query,
//! candidate-retention, or traversal state lives in a source.

use std::sync::Arc;

use quant::cascade::{LayerKind, LayerSpec, QueryRotationPlan, Rotation};

use super::arena::VectorArena;
use super::stacked::levels::SerializableStore;
use crate::vector::storage::quantized::sign::SignVectors;

pub(crate) const ROUTER_SEED: u64 = 0x726f_7574_6572_0001;

/// Version 1 uses one zero origin and one rotation throughout a segment router.
/// Levels independently select a representation without changing coordinates.
pub(crate) struct EncodingContext {
    pub(crate) seed: u64,
    pub(crate) rotation: QueryRotationPlan,
}

impl EncodingContext {
    pub(crate) fn new(dim: usize, seed: u64) -> Self {
        Self {
            seed,
            rotation: QueryRotationPlan::new(
                dim,
                &[LayerSpec {
                    kind: LayerKind::Sign,
                    bits: 1,
                    rotation: Rotation::SeededFhtChaCha8 { seed },
                }],
            ),
        }
    }
}

/// Available representations of the same addressed rows. Legacy internal banks
/// have codes only; new exact-refinement operations must require raw rows.
pub(crate) enum VectorSet<S> {
    Raw(Arc<S>),
    Encoded {
        codes: Arc<SignVectors>,
        context: Arc<EncodingContext>,
        raw: Option<Arc<S>>,
    },
}

impl<S> Clone for VectorSet<S> {
    fn clone(&self) -> Self {
        match self {
            Self::Raw(raw) => Self::Raw(raw.clone()),
            Self::Encoded {
                codes,
                context,
                raw,
            } => Self::Encoded {
                codes: codes.clone(),
                context: context.clone(),
                raw: raw.clone(),
            },
        }
    }
}

/// Access the available representations of one vector bank.
pub(crate) trait VectorSource {
    type Raw: VectorArena<Elem = f32>;
    fn num_vectors(&self, dim: usize) -> usize;
    fn raw(&self) -> Option<&Self::Raw>;
    fn compressed(&self) -> Option<(&SignVectors, &EncodingContext)>;
}

impl<S: VectorArena<Elem = f32>> VectorSource for S {
    type Raw = S;
    fn num_vectors(&self, dim: usize) -> usize {
        VectorArena::num_vectors(self, dim)
    }
    fn raw(&self) -> Option<&S> {
        Some(self)
    }
    fn compressed(&self) -> Option<(&SignVectors, &EncodingContext)> {
        None
    }
}

impl<S: VectorArena<Elem = f32>> VectorSource for VectorSet<S> {
    type Raw = S;
    fn num_vectors(&self, dim: usize) -> usize {
        match self {
            VectorSet::Raw(raw) => VectorArena::num_vectors(raw.as_ref(), dim),
            VectorSet::Encoded { codes, .. } => codes.len(),
        }
    }
    fn raw(&self) -> Option<&S> {
        match self {
            VectorSet::Raw(raw) => Some(raw.as_ref()),
            VectorSet::Encoded { raw, .. } => raw.as_deref(),
        }
    }
    fn compressed(&self) -> Option<(&SignVectors, &EncodingContext)> {
        match self {
            VectorSet::Raw(_) => None,
            VectorSet::Encoded { codes, context, .. } => Some((codes, context)),
        }
    }
}

impl<S: SerializableStore> SerializableStore for VectorSet<S> {
    fn len(&self) -> usize {
        match self {
            VectorSet::Raw(raw) => raw.len(),
            VectorSet::Encoded { codes, .. } => codes.len(),
        }
    }
    fn serialize_rows<W: std::io::Write + ?Sized>(&self, out: &mut W) -> std::io::Result<()> {
        match self {
            VectorSet::Raw(raw) => raw.serialize_rows(out),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "full-precision router format requires raw vector banks",
            )),
        }
    }
}
