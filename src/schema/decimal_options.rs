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
#[serde(from = "DecimalOptionsDeser")]
pub struct DecimalOptions {
    /// Maximum total number of significant digits.
    /// None means unlimited (up to MAX_PRECISION).
    precision: Option<u32>,
    /// Number of digits after the decimal point.
    /// Negative values round to the left of the decimal point.
    /// None means unlimited (up to MAX_SCALE).
    scale: Option<i32>,
    /// Whether the field is indexed.
    indexed: bool,
    /// Whether field norms are stored.
    fieldnorms: bool,
    /// Whether the field is a fast field.
    fast: bool,
    /// Whether the field is stored.
    stored: bool,
}

/// For backward compatibility, interpret missing fieldnorms as true if indexed.
#[derive(Deserialize)]
struct DecimalOptionsDeser {
    #[serde(default)]
    precision: Option<u32>,
    #[serde(default)]
    scale: Option<i32>,
    indexed: bool,
    #[serde(default)]
    fieldnorms: Option<bool>,
    fast: bool,
    stored: bool,
}

impl From<DecimalOptionsDeser> for DecimalOptions {
    fn from(deser: DecimalOptionsDeser) -> Self {
        DecimalOptions {
            precision: deser.precision,
            scale: deser.scale,
            indexed: deser.indexed,
            fieldnorms: deser.fieldnorms.unwrap_or(deser.indexed),
            fast: deser.fast,
            stored: deser.stored,
        }
    }
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
            // For precision/scale, take the more restrictive value if both are set
            // For precision: smaller is more restrictive
            // For scale: the one closer to zero is more restrictive (allows less precision)
            precision: match (self.precision, other.precision) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
            scale: match (self.scale, other.scale) {
                (Some(a), Some(b)) => {
                    // Take the scale that's closer to zero (more restrictive)
                    if a.abs() <= b.abs() {
                        Some(a)
                    } else {
                        Some(b)
                    }
                }
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
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
    fn test_decimal_options_deser_if_fieldnorm_missing_indexed_true() {
        let json = r#"{
            "precision": 10,
            "scale": 2,
            "indexed": true,
            "fast": false,
            "stored": false
        }"#;
        let opts: DecimalOptions = serde_json::from_str(json).unwrap();
        assert_eq!(
            &opts,
            &DecimalOptions {
                precision: Some(10),
                scale: Some(2),
                indexed: true,
                fieldnorms: true,
                fast: false,
                stored: false
            }
        );
    }

    #[test]
    fn test_decimal_options_deser_if_fieldnorm_missing_indexed_false() {
        let json = r#"{
            "indexed": false,
            "stored": false,
            "fast": false
        }"#;
        let opts: DecimalOptions = serde_json::from_str(json).unwrap();
        assert_eq!(
            &opts,
            &DecimalOptions {
                precision: None,
                scale: None,
                indexed: false,
                fieldnorms: false,
                fast: false,
                stored: false
            }
        );
    }
}
