//! Decoder element sizes define the portable column and entry alignment rules.
use std::mem::size_of;

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
pub const MAX_ELEM_BYTES: usize = max_elem_bytes();
