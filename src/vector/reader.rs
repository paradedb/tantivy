//! Per-segment vector reader.
//!
//! Composes a [`FlatVecReader`] and an [`IvfVecReader`] under a single
//! plugin name (`"vectors"`). Callers — primarily
//! [`VectorBackend::for_segment`](super::backend::VectorBackend::for_segment)
//! and the flat-format merge routine — ask for the format they want
//! via [`VectorReader::open_flat_column`] / [`VectorReader::open_ivf_column`].

use std::any::Any;

use super::flat::{FlatVecReader, VectorColumn};
use super::ivf::{IvfVecReader, IvfVectorColumn};
use crate::plugin::{PluginReader, PluginReaderContext};
use crate::schema::Field;

pub struct VectorReader {
    flat: FlatVecReader,
    ivf: IvfVecReader,
}

impl VectorReader {
    pub(crate) fn open(ctx: &PluginReaderContext) -> crate::Result<Self> {
        Ok(Self {
            flat: FlatVecReader::open(ctx)?,
            ivf: IvfVecReader::stub(),
        })
    }

    /// Per-field flat view. Returns `None` if the segment has no
    /// flatvec file or the field isn't vector-typed.
    pub fn open_flat_column(&self, field: Field) -> Option<VectorColumn> {
        self.flat.open_column(field)
    }

    /// Per-field IVF view. Returns `None` if the segment has no
    /// ivfvec file (which today is always — the IVF reader is still a stub).
    pub fn open_ivf_column(&self, field: Field) -> Option<IvfVectorColumn> {
        self.ivf.open_column(field)
    }

    /// Declared dimension for a vector field, if any.
    pub fn dim(&self, field: Field) -> Option<usize> {
        self.flat.dim(field)
    }
}

impl PluginReader for VectorReader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
