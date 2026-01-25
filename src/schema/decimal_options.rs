//! Configuration options for decimal fields.
//!
//! This module provides `DecimalOptions`, which defines how arbitrary precision
//! decimal fields should be handled by Tantivy, including precision and scale
//! constraints similar to SQL NUMERIC(p, s).
//!
//! ## PostgreSQL Compatibility
//!
//! Supports the full PostgreSQL NUMERIC specification:
//! - Up to 131,072 digits before the decimal point
//! - Up to 16,383 digits after the decimal point
//! - Negative scale (rounds to left of decimal point)
//! - Special values: Infinity, -Infinity, NaN

use std::ops::BitOr;

use serde::{Deserialize, Serialize};

use super::flags::{FastFlag, IndexedFlag, SchemaFlagList, StoredFlag};

/// Maximum supported precision (total number of significant digits).
pub const MAX_PRECISION: u32 = 131072;
/// Maximum supported scale (digits after decimal point).
/// PostgreSQL allows up to 16383.
pub const MAX_SCALE: i32 = 16383;
/// Minimum supported scale (negative scale rounds to left of decimal).
/// PostgreSQL allows down to -1000.
pub const MIN_SCALE: i32 = -1000;

/// Maximum precision that can be stored using Decimal64 (fixed 8 bytes with embedded scale).
/// Decimal64 uses 8 bits for scale and 56 bits for signed value.
/// 56-bit signed range: -2^55 to 2^55-1 ≈ ±3.6 × 10^16
/// So we can safely store up to 16 significant digits.
pub const MAX_I64_PRECISION: u32 = 16;

/// Define how a decimal field should be handled by tantivy.
///
/// Supports PostgreSQL-style precision and scale:
/// - `NUMERIC` - unlimited precision (precision=None, scale=None)
/// - `NUMERIC(p)` - precision only (scale defaults to 0)
/// - `NUMERIC(p, s)` - both specified (scale can be negative)
///
/// Negative scale rounds to the left of the decimal point:
/// - `NUMERIC(2, -3)` rounds to nearest 1000, stores values like 12000, 99000
///
/// Values exceeding precision/scale will be truncated/rounded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DecimalOptions {
    /// Maximum total number of significant digits.
    /// None means unlimited (up to MAX_PRECISION).
    #[serde(default)]
    precision: Option<u32>,
    /// Number of digits after the decimal point.
    /// Negative values round to the left of the decimal point.
    /// None means unlimited (up to MAX_SCALE).
    #[serde(default)]
    scale: Option<i32>,
    /// Whether the field is indexed.
    #[serde(default)]
    indexed: bool,
    /// Whether field norms are stored.
    #[serde(default)]
    fieldnorms: bool,
    /// Whether the field is a fast field.
    #[serde(default)]
    fast: bool,
    /// Whether the field is stored.
    #[serde(default)]
    stored: bool,
}

impl DecimalOptions {
    /// Creates a new `DecimalOptions` with unlimited precision and scale.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates options with specified precision (like NUMERIC(p)).
    /// Scale defaults to 0.
    pub fn with_precision(precision: u32) -> Self {
        DecimalOptions {
            precision: Some(precision.min(MAX_PRECISION)),
            scale: Some(0),
            ..Default::default()
        }
    }

    /// Creates options with specified precision and scale (like NUMERIC(p, s)).
    ///
    /// # Arguments
    ///
    /// * `precision` - Maximum total significant digits
    /// * `scale` - Digits after decimal point (negative values round to left of decimal)
    pub fn with_precision_and_scale(precision: u32, scale: i32) -> Self {
        DecimalOptions {
            precision: Some(precision.min(MAX_PRECISION)),
            scale: Some(scale.clamp(MIN_SCALE, MAX_SCALE)),
            ..Default::default()
        }
    }

    /// Returns the precision limit, if any.
    #[inline]
    pub fn precision(&self) -> Option<u32> {
        self.precision
    }

    /// Returns the scale limit, if any.
    /// Negative values indicate rounding to the left of the decimal point.
    #[inline]
    pub fn scale(&self) -> Option<i32> {
        self.scale
    }

    /// Returns true if the value is indexed.
    #[inline]
    pub fn is_indexed(&self) -> bool {
        self.indexed
    }

    /// Returns true if and only if the value has fieldnorms.
    #[inline]
    pub fn fieldnorms(&self) -> bool {
        self.fieldnorms
    }

    /// Returns true if the value is a fast field.
    #[inline]
    pub fn is_fast(&self) -> bool {
        self.fast
    }

    /// Returns true if the value is stored.
    #[inline]
    pub fn is_stored(&self) -> bool {
        self.stored
    }

    /// Returns true if the decimal values with this precision/scale can be
    /// safely stored using Decimal64 (8 bytes with embedded scale).
    ///
    /// Decimal64 uses 56 bits for the value, which can hold ~3.6×10^16,
    /// meaning we can safely store decimals with up to 16 significant digits.
    /// If the precision exceeds this, the decimal should be stored as bytes instead.
    ///
    /// # Returns
    ///
    /// - `true` if both precision and scale are defined AND precision <= 16
    /// - `false` if precision is None, scale is None, or precision > 16
    #[inline]
    pub fn fits_in_i64(&self) -> bool {
        match (self.precision, self.scale) {
            (Some(precision), Some(_scale)) => precision <= MAX_I64_PRECISION,
            _ => false,
        }
    }

    /// Set the precision limit.
    ///
    /// # Arguments
    ///
    /// * `precision` - Maximum total number of significant digits
    #[must_use]
    pub fn set_precision(mut self, precision: u32) -> DecimalOptions {
        self.precision = Some(precision.min(MAX_PRECISION));
        self
    }

    /// Set the scale limit.
    ///
    /// # Arguments
    ///
    /// * `scale` - Number of digits after decimal point (negative rounds to left)
    #[must_use]
    pub fn set_scale(mut self, scale: i32) -> DecimalOptions {
        self.scale = Some(scale.clamp(MIN_SCALE, MAX_SCALE));
        self
    }

    /// Set the field as indexed.
    ///
    /// Setting a decimal as indexed will generate
    /// a posting list for each unique value.
    #[must_use]
    pub fn set_indexed(mut self) -> DecimalOptions {
        self.indexed = true;
        self
    }

    /// Set the field as having fieldnorms.
    ///
    /// Setting a decimal as having fieldnorms will generate
    /// the fieldnorm data for it.
    #[must_use]
    pub fn set_fieldnorms(mut self) -> DecimalOptions {
        self.fieldnorms = true;
        self
    }

    /// Set the field as a fast field.
    ///
    /// Fast fields are designed for random access.
    #[must_use]
    pub fn set_fast(mut self) -> DecimalOptions {
        self.fast = true;
        self
    }

    /// Set the field as stored.
    ///
    /// Only the fields that are set as *stored* are
    /// persisted into the Tantivy's store.
    #[must_use]
    pub fn set_stored(mut self) -> DecimalOptions {
        self.stored = true;
        self
    }
}

impl<T: Into<DecimalOptions>> BitOr<T> for DecimalOptions {
    type Output = DecimalOptions;

    fn bitor(self, other: T) -> DecimalOptions {
        let other = other.into();
        DecimalOptions {
            // Precision/scale come from explicit configuration, not flag composition.
            // Keep self's values if set, otherwise take other's.
            precision: self.precision.or(other.precision),
            scale: self.scale.or(other.scale),
            indexed: self.indexed | other.indexed,
            fieldnorms: self.fieldnorms | other.fieldnorms,
            stored: self.stored | other.stored,
            fast: self.fast | other.fast,
        }
    }
}

impl From<()> for DecimalOptions {
    fn from(_: ()) -> Self {
        Self::default()
    }
}

impl From<FastFlag> for DecimalOptions {
    fn from(_: FastFlag) -> Self {
        DecimalOptions {
            precision: None,
            scale: None,
            indexed: false,
            fieldnorms: false,
            stored: false,
            fast: true,
        }
    }
}

impl From<StoredFlag> for DecimalOptions {
    fn from(_: StoredFlag) -> Self {
        DecimalOptions {
            precision: None,
            scale: None,
            indexed: false,
            fieldnorms: false,
            stored: true,
            fast: false,
        }
    }
}

impl From<IndexedFlag> for DecimalOptions {
    fn from(_: IndexedFlag) -> Self {
        DecimalOptions {
            precision: None,
            scale: None,
            indexed: true,
            fieldnorms: true,
            stored: false,
            fast: false,
        }
    }
}

impl<Head, Tail> From<SchemaFlagList<Head, Tail>> for DecimalOptions
where
    Head: Clone,
    Tail: Clone,
    Self: BitOr<Output = Self> + From<Head> + From<Tail>,
{
    fn from(head_tail: SchemaFlagList<Head, Tail>) -> Self {
        Self::from(head_tail.head) | Self::from(head_tail.tail)
    }
}

#[cfg(test)]
mod tests {
    use crate::schema::{DecimalOptions, FAST, INDEXED, STORED};

    #[test]
    fn test_decimal_options_default() {
        let opts = DecimalOptions::default();
        assert!(!opts.is_indexed());
        assert!(!opts.is_fast());
        assert!(!opts.is_stored());
        assert!(!opts.fieldnorms());
        assert!(opts.precision().is_none());
        assert!(opts.scale().is_none());
    }

    #[test]
    fn test_decimal_options_with_precision() {
        let opts = DecimalOptions::with_precision(10);
        assert_eq!(opts.precision(), Some(10));
        assert_eq!(opts.scale(), Some(0));
    }

    #[test]
    fn test_decimal_options_with_precision_and_scale() {
        let opts = DecimalOptions::with_precision_and_scale(10, 2);
        assert_eq!(opts.precision(), Some(10));
        assert_eq!(opts.scale(), Some(2));
    }

    #[test]
    fn test_decimal_options_with_negative_scale() {
        // PostgreSQL supports negative scale (rounds to left of decimal)
        let opts = DecimalOptions::with_precision_and_scale(5, -3);
        assert_eq!(opts.precision(), Some(5));
        assert_eq!(opts.scale(), Some(-3));
    }

    #[test]
    fn test_decimal_options_flags() {
        assert_eq!(DecimalOptions::default().set_fast(), FAST.into());
        assert_eq!(
            DecimalOptions::default().set_indexed().set_fieldnorms(),
            INDEXED.into()
        );
        assert_eq!(DecimalOptions::default().set_stored(), STORED.into());
    }

    #[test]
    fn test_decimal_options_flag_composition() {
        assert_eq!(
            DecimalOptions::default().set_fast().set_stored(),
            (FAST | STORED).into()
        );
        assert_eq!(
            DecimalOptions::default()
                .set_indexed()
                .set_fieldnorms()
                .set_fast(),
            (INDEXED | FAST).into()
        );
        assert_eq!(
            DecimalOptions::default()
                .set_stored()
                .set_fieldnorms()
                .set_indexed(),
            (STORED | INDEXED).into()
        );
    }

    #[test]
    fn test_decimal_options_setters() {
        assert!(!DecimalOptions::default().is_stored());
        assert!(!DecimalOptions::default().is_fast());
        assert!(!DecimalOptions::default().is_indexed());
        assert!(!DecimalOptions::default().fieldnorms());
        assert!(DecimalOptions::default().set_stored().is_stored());
        assert!(DecimalOptions::default().set_fast().is_fast());
        assert!(DecimalOptions::default().set_indexed().is_indexed());
        assert!(DecimalOptions::default().set_fieldnorms().fieldnorms());
    }

    #[test]
    fn test_decimal_options_precision_scale_setters() {
        let opts = DecimalOptions::default().set_precision(15).set_scale(3);
        assert_eq!(opts.precision(), Some(15));
        assert_eq!(opts.scale(), Some(3));

        // Test negative scale setter
        let opts = DecimalOptions::default().set_precision(5).set_scale(-2);
        assert_eq!(opts.precision(), Some(5));
        assert_eq!(opts.scale(), Some(-2));
    }

    #[test]
    fn test_decimal_options_serialization() {
        let opts = DecimalOptions::with_precision_and_scale(10, 2)
            .set_indexed()
            .set_fast()
            .set_stored();

        let json = serde_json::to_string(&opts).unwrap();
        let deserialized: DecimalOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, deserialized);
    }

    #[test]
    fn test_decimal_options_negative_scale_serialization() {
        let opts = DecimalOptions::with_precision_and_scale(5, -3)
            .set_indexed()
            .set_stored();

        let json = serde_json::to_string(&opts).unwrap();
        let deserialized: DecimalOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, deserialized);
        assert_eq!(deserialized.scale(), Some(-3));
    }

    #[test]
    fn test_decimal_options_deser_with_defaults() {
        // All fields use serde defaults when missing
        let json = r#"{}"#;
        let opts: DecimalOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts, DecimalOptions::default());
    }

    #[test]
    fn test_decimal_options_deser_partial() {
        let json = r#"{
            "precision": 10,
            "scale": 2,
            "indexed": true
        }"#;
        let opts: DecimalOptions = serde_json::from_str(json).unwrap();
        assert_eq!(
            &opts,
            &DecimalOptions {
                precision: Some(10),
                scale: Some(2),
                indexed: true,
                fieldnorms: false,
                fast: false,
                stored: false
            }
        );
    }

    #[test]
    fn test_decimal_options_fits_in_i64() {
        // Small precision (≤16) with scale should fit in Decimal64
        let opts = DecimalOptions::with_precision_and_scale(10, 2);
        assert!(opts.fits_in_i64());

        let opts = DecimalOptions::with_precision_and_scale(16, 5);
        assert!(opts.fits_in_i64());

        // Precision exactly at boundary (16 digits)
        let opts = DecimalOptions::with_precision_and_scale(16, 0);
        assert!(opts.fits_in_i64());

        // Large precision (>16) should NOT fit in Decimal64
        let opts = DecimalOptions::with_precision_and_scale(17, 2);
        assert!(!opts.fits_in_i64());

        let opts = DecimalOptions::with_precision_and_scale(30, 10);
        assert!(!opts.fits_in_i64());

        // Unlimited precision (None) should NOT fit in Decimal64
        let opts = DecimalOptions::default();
        assert!(!opts.fits_in_i64());

        // Precision without scale should NOT fit in Decimal64 (scale is required)
        let opts = DecimalOptions {
            precision: Some(10),
            scale: None,
            ..Default::default()
        };
        assert!(!opts.fits_in_i64());

        // Negative scale still counts - precision is what matters
        let opts = DecimalOptions::with_precision_and_scale(15, -3);
        assert!(opts.fits_in_i64());
    }
}
