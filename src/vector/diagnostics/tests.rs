//! Diagnostics tests: measurement merges and moments.

use super::*;
use crate::vector::storage::metadata::{QuantizationSchedule, QuantizerKind};

#[test]
fn estimator_merge_sums_rows_and_preserves_query_count() {
    let measurement = |sample_rows, query_count, value| VectorEstimatorMeasurements {
        source: VectorEstimatorSource::Provided,
        schedule: QuantizationSchedule::new(vec![(QuantizerKind::Sign, 1)]),
        aggregate: vec![VectorEstimatorMoments {
            sample_count: 1,
            normalized_error_sum: value,
            normalized_error_squared_sum: value * value,
        }],
        per_query: vec![
            vec![VectorEstimatorMoments {
                sample_count: 1,
                normalized_error_sum: value,
                normalized_error_squared_sum: value * value,
            }];
            query_count as usize
        ],
        sample_rows,
        query_count,
    };
    let mut aggregate = measurement(7, 2, 1.0);
    aggregate.merge(&measurement(5, 2, 3.0)).unwrap();
    assert_eq!(aggregate.sample_rows(), 12);
    assert_eq!(aggregate.query_count(), 2);
    assert_eq!(aggregate.aggregate()[0].sample_count, 2);
    assert_eq!(aggregate.aggregate()[0].bias(), Some(2.0));
    assert!(aggregate.merge(&measurement(1, 3, 1.0)).is_err());
    let mut held_out = measurement(1, 2, 1.0);
    held_out.source = VectorEstimatorSource::HeldOut;
    assert!(aggregate.merge(&held_out).is_err());
}

#[test]
fn measurement_merges_reject_different_schedules() {
    use QuantizerKind::{Grid, Sign};

    let measurement = |schedule: &[(QuantizerKind, u8)]| VectorEstimatorMeasurements {
        source: VectorEstimatorSource::Provided,
        schedule: QuantizationSchedule::new(schedule.to_vec()),
        aggregate: vec![VectorEstimatorMoments::default(); schedule.len()],
        per_query: vec![vec![VectorEstimatorMoments::default(); schedule.len()]],
        sample_rows: 0,
        query_count: 1,
    };
    for (left, right, expected) in [
        (
            vec![(Sign, 1)],
            vec![(Sign, 1), (Grid, 4)],
            "different schedules: [sign:1] and [sign:1, grid:4]",
        ),
        (
            vec![(Sign, 1), (Grid, 4)],
            vec![(Grid, 2), (Grid, 4)],
            "different schedules: [sign:1, grid:4] and [grid:2, grid:4]",
        ),
        (
            vec![(Sign, 1)],
            vec![(Grid, 1)],
            "different schedules: [sign:1] and [grid:1]",
        ),
    ] {
        let a = measurement(&left);
        let b = measurement(&right);
        assert!(a
            .clone()
            .merge(&b)
            .unwrap_err()
            .to_string()
            .contains(expected));
        let mut audit = VectorErrorAuditMeasurements {
            source: VectorEstimatorSource::Provided,
            depths: vec![VectorErrorDepthMeasurements::default(); left.len()],
            estimator: a,
        };
        assert_eq!(audit.schedule(), audit.estimator.schedule());
        let other = VectorErrorAuditMeasurements {
            source: VectorEstimatorSource::Provided,
            depths: vec![VectorErrorDepthMeasurements::default(); right.len()],
            estimator: b,
        };
        assert!(audit
            .merge(&other)
            .unwrap_err()
            .to_string()
            .contains(expected));
    }
}

#[test]
fn audit_moments_retain_exact_mergeable_quantiles() {
    let mut left = VectorAuditMoments::default();
    for value in [-4.0, 1.0, 2.0] {
        left.observe(value);
    }
    let mut right = VectorAuditMoments::default();
    for value in [3.0, 5.0] {
        right.observe(value);
    }
    left.merge(&right);
    assert_eq!(left.p50(), Some(2.0));
    assert_eq!(left.p95(), Some(5.0));
    assert_eq!(left.p99(), Some(5.0));
    assert_eq!(left.p99_abs(), Some(5.0));
    assert_eq!(left.max_abs(), Some(5.0));
}
