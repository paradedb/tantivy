//! Vector element types, their accumulators, and the element sizes behind column and entry
//! alignment.

use std::io;
use std::mem::size_of;

use crate::schema::VectorDType;

/// Element consumed by a column decoder; F16 uses a binary16 bit representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ElemType {
    U8,
    F16,
    F32,
    U32,
    U64,
}
impl ElemType {
    /// Complete element vocabulary used to derive the maximum entry alignment.
    pub(crate) const ALL: [Self; 5] = [Self::U8, Self::F16, Self::F32, Self::U32, Self::U64];

    /// Serialized element width, independent of the target's ABI alignment.
    pub(crate) const fn size(self) -> usize {
        match self {
            Self::U8 => size_of::<u8>(),
            Self::F16 => size_of::<u16>(),
            Self::F32 => size_of::<f32>(),
            Self::U32 => size_of::<u32>(),
            Self::U64 => size_of::<u64>(),
        }
    }
}

const fn max_elem_bytes() -> usize {
    let mut largest = 0;
    let mut i = 0;
    while i < ElemType::ALL.len() {
        let size = ElemType::ALL[i].size();
        assert!(size.is_power_of_two());
        if size > largest {
            largest = size;
        }
        i += 1;
    }
    largest
}

/// Maximum decoder element size, bounding column and block alignment.
/// Storage providers can use this to check page-data starts and usable page lengths.
pub(crate) const MAX_ELEM_BYTES: usize = max_elem_bytes();

/// Fixed alignment of Data entries and their block directories.
pub const ENTRY_ALIGN: usize = 8;
const _: () = assert!(MAX_ELEM_BYTES <= ENTRY_ALIGN);

// --- Element traits ---

/// Seals [`Accumulator`] and [`VectorElement`]: the element types are the ones [`VectorDType`]
/// lists.
mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// The wide accumulator a reduction kernel sums an element type's squares in; sealed.
///
/// `to_f64` is the lossy read-out at the end of a reduction.
pub trait Accumulator: sealed::Sealed + Copy + Send + Sync + 'static {
    const ZERO: Self;
    fn add(self, rhs: Self) -> Self;
    fn to_f64(self) -> f64;
}

impl Accumulator for f64 {
    const ZERO: Self = 0.0;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
}

/// A vector element type with the primitives the storage layer and the distance kernels need;
/// sealed, one per [`VectorDType`].
///
/// `squared_diff` and `product` return `f32` so kernels share one accumulator across element
/// types; the norm kernels accumulate in [`Self::Acc`] through [`Self::mul_wide`].
pub trait VectorElement: sealed::Sealed + Copy + Send + Sync + 'static {
    const DTYPE: VectorDType;
    const SIZE_BYTES: usize;

    /// Accumulator for this element's squares in reduction kernels.
    /// Each element type declares how much headroom it needs: f32 -> f64
    /// (squaring doubles the exponent, so |v| > ~1.8e19 overflows an f32
    /// product; f64 makes any sum of finite-f32 squares finite).
    type Acc: Accumulator;

    fn encode_le<W: io::Write + ?Sized>(&self, buf: &mut W) -> io::Result<()>;

    /// Decode one element from its little-endian byte representation.
    /// `bytes.len()` must be `SIZE_BYTES`.
    fn decode_le(bytes: &[u8]) -> Self;

    /// `(a - b)^2` promoted to `f32` for accumulator-friendly distance
    /// computation. For `f32` this is the obvious arithmetic; for
    /// quantized types it may promote through a wider integer first.
    fn squared_diff(a: Self, b: Self) -> f32;

    /// `a * b` promoted to `f32`. Same rationale as `squared_diff`.
    fn product(a: Self, b: Self) -> f32;

    /// Widen-THEN-multiply: the cast happens before the square so the
    /// product cannot overflow the narrow type. One operation, so the
    /// ordering invariant is unforgettable per element type.
    fn mul_wide(a: Self, b: Self) -> Self::Acc;

    /// Lossless widening to `f32`.
    fn to_f32(self) -> f32;

    /// Narrowing from `f32`; used at normalization write-back where
    /// values are already `<= 1`.
    fn from_f32(v: f32) -> Self;
}

impl VectorElement for f32 {
    const DTYPE: VectorDType = VectorDType::F32;
    const SIZE_BYTES: usize = 4;

    type Acc = f64;

    #[inline(always)]
    fn encode_le<W: io::Write + ?Sized>(&self, buf: &mut W) -> io::Result<()> {
        buf.write_all(&self.to_le_bytes())
    }

    #[inline(always)]
    fn decode_le(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[inline(always)]
    fn squared_diff(a: Self, b: Self) -> f32 {
        let d = a - b;
        d * d
    }

    #[inline(always)]
    fn product(a: Self, b: Self) -> f32 {
        a * b
    }

    #[inline(always)]
    fn mul_wide(a: Self, b: Self) -> f64 {
        (a as f64) * (b as f64)
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }
}
