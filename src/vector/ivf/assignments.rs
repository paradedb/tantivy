//! Assign vectors to the shared centroid matrix in bounded SGEMM batches.

use superkmeans::gemm::sgemm_row_major_b_transposed;

use crate::schema::{Metric, VectorOptions};
use crate::vector::distance::norm_squared;
use crate::vector::Similarity;

pub(super) const ASSIGN_BATCH_SIZE: usize = 256;
const CENTROID_TILE: usize = 1024;

fn best_score_position(scores: &[f32]) -> usize {
    let ordered = |score: f32| {
        let bits = score.to_bits() as i32;
        bits ^ (((bits >> 31) as u32) >> 1) as i32
    };
    let best = scores.iter().copied().map(ordered).max().unwrap();
    scores
        .iter()
        .position(|&score| ordered(score) == best)
        .unwrap()
}

pub(super) struct BatchAssigner {
    centroids: Vec<f32>,
    centroid_norms: Vec<f32>,
    dim: usize,
    metric: Metric,
    scores: Vec<f32>,
}

impl BatchAssigner {
    pub(super) fn new(centroids: Vec<f32>, options: &VectorOptions) -> Self {
        let dim = options.dim();
        assert!(dim > 0 && !centroids.is_empty() && centroids.len() % dim == 0);
        let count = centroids.len() / dim;
        let centroid_norms = match options.metric() {
            Metric::L2 => superkmeans::squared_norms(&centroids, count, dim),
            Metric::Cosine => centroids
                .chunks_exact(dim)
                .map(|row| norm_squared(row).sqrt())
                .collect(),
            Metric::Dot => Vec::new(),
        };
        Self {
            centroids,
            centroid_norms,
            dim,
            metric: options.metric(),
            scores: Vec::with_capacity(ASSIGN_BATCH_SIZE * count.min(CENTROID_TILE)),
        }
    }

    pub(super) fn assign(&mut self, values: &[f32]) -> Vec<usize> {
        assert_eq!(values.len() % self.dim, 0);
        let count = self.centroids.len() / self.dim;
        self.scores.resize(
            (values.len() / self.dim).min(ASSIGN_BATCH_SIZE) * count.min(CENTROID_TILE),
            0.0,
        );
        let mut assignments = Vec::with_capacity(values.len() / self.dim);
        for rows in values.chunks(ASSIGN_BATCH_SIZE * self.dim) {
            let row_count = rows.len() / self.dim;
            let row_norms = match self.metric {
                Metric::L2 => superkmeans::squared_norms(rows, row_count, self.dim),
                Metric::Cosine => rows
                    .chunks_exact(self.dim)
                    .map(|row| norm_squared(row).sqrt())
                    .collect(),
                Metric::Dot => Vec::new(),
            };
            let mut nearest = vec![(Similarity::WORST, 0); row_count];
            for (tile, centroids) in self.centroids.chunks(CENTROID_TILE * self.dim).enumerate() {
                let centroid_count = centroids.len() / self.dim;
                sgemm_row_major_b_transposed(
                    row_count,
                    self.dim,
                    centroid_count,
                    rows,
                    centroids,
                    &mut self.scores[..row_count * centroid_count],
                );
                for (row, best) in nearest.iter_mut().enumerate() {
                    let scores = &mut self.scores[row * centroid_count..(row + 1) * centroid_count];
                    match self.metric {
                        Metric::L2 => {
                            for (col, score) in scores.iter_mut().enumerate() {
                                let norms = row_norms[row]
                                    + self.centroid_norms[tile * CENTROID_TILE + col];
                                let distance = norms - 2.0 * *score;
                                let error = (norms + 2.0 * score.abs())
                                    * (2.0 * self.dim as f32 * f32::EPSILON);
                                // Near cancellation or overflow, use the direct L2 kernel.
                                *score = if !distance.is_finite() || distance <= error {
                                    Metric::L2
                                        .similarity(
                                            &rows[row * self.dim..(row + 1) * self.dim],
                                            &centroids[col * self.dim..(col + 1) * self.dim],
                                        )
                                        .score()
                                } else {
                                    -distance
                                };
                            }
                        }
                        Metric::Cosine => {
                            for (col, score) in scores.iter_mut().enumerate() {
                                let norm = row_norms[row]
                                    * self.centroid_norms[tile * CENTROID_TILE + col];
                                *score = if norm == 0.0 { 0.0 } else { *score / norm };
                            }
                        }
                        Metric::Dot => {}
                    }
                    let col = best_score_position(scores);
                    let score = Similarity::new(scores[col]);
                    if tile == 0 || score > best.0 {
                        *best = (score, tile * CENTROID_TILE + col);
                    }
                }
            }
            assignments.extend(nearest.into_iter().map(|(_, centroid)| centroid));
        }
        assignments
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn batches_match_scalar_assignment_across_tiles_and_metrics() {
        let dim = 9;
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let centroids: Vec<f32> = (0..(CENTROID_TILE + 7) * dim)
                .map(|_| rng.random_range(-20..=20) as f32)
                .collect();
            let values: Vec<f32> = (0..(ASSIGN_BATCH_SIZE + 3) * dim)
                .map(|_| rng.random_range(-20..=20) as f32)
                .collect();
            let expected: Vec<_> = values
                .chunks_exact(dim)
                .map(|row| {
                    centroids
                        .chunks_exact(dim)
                        .enumerate()
                        .map(|(id, centroid)| (id, metric.similarity(row, centroid)))
                        .max_by(|(a, sa), (b, sb)| sa.cmp(sb).then_with(|| b.cmp(a)))
                        .unwrap()
                        .0
                })
                .collect();
            let mut assigner = BatchAssigner::new(centroids, &VectorOptions::new(dim, metric));
            assert_eq!(assigner.assign(&values), expected, "{metric:?}");
            assert_eq!(assigner.assign(&values[..dim]), expected[..1]);
            assert!(assigner.assign(&[]).is_empty());
            assert!(assigner.scores.capacity() <= ASSIGN_BATCH_SIZE * CENTROID_TILE);
        }
    }

    #[test]
    fn total_order_and_first_tie() {
        let values = [
            f32::from_bits(0xffc00001),
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f32::INFINITY,
            f32::from_bits(0x7fc00001),
        ];
        for &a in &values {
            for &b in &values {
                for &c in &values {
                    let scores = [a, b, c, a];
                    let expected = (0..scores.len())
                        .max_by(|&i, &j| scores[i].total_cmp(&scores[j]).then_with(|| j.cmp(&i)))
                        .unwrap();
                    assert_eq!(best_score_position(&scores), expected);
                }
            }
        }
        let mut centroids = vec![0.0; (CENTROID_TILE + 2) * 2];
        centroids[..2].copy_from_slice(&[10.0, 0.0]);
        centroids[CENTROID_TILE * 2..].copy_from_slice(&[1.0, 0.0, 10.0, 0.0]);
        let mut assigner = BatchAssigner::new(centroids, &VectorOptions::new(2, Metric::Dot));
        assert_eq!(assigner.assign(&[1.0, 0.0, 0.0, 0.0]), [0, 0]);
    }

    #[test]
    fn metric_semantics_and_zero_norms() {
        let centroids = vec![10.0, 0.0, 0.0, 1.0, 7.0, 7.0, 0.0, 0.0];
        for (metric, expected) in [(Metric::Dot, 2), (Metric::Cosine, 2), (Metric::L2, 1)] {
            let mut assigner =
                BatchAssigner::new(centroids.clone(), &VectorOptions::new(2, metric));
            assert_eq!(assigner.assign(&[1.0, 2.0]), [expected]);
            if metric != Metric::L2 {
                assert_eq!(assigner.assign(&[0.0, 0.0]), [0]);
            }
        }
        let mut assigner = BatchAssigner::new(
            vec![0.0, 0.0, -1.0, 0.0],
            &VectorOptions::new(2, Metric::Cosine),
        );
        assert_eq!(assigner.assign(&[1.0, 0.0]), [0]);
    }

    #[test]
    fn l2_cancellation_and_overflow() {
        let mut assigner = BatchAssigner::new(
            vec![1e6, 1e6, 1e6 + 1.0, 1e6],
            &VectorOptions::new(2, Metric::L2),
        );
        assert_eq!(assigner.assign(&[1e6 + 1.0, 1e6]), [1]);
        assert_eq!(assigner.assign(&[3e38, 3e38]), [0]);
    }
}
