//! Assign vectors to the shared centroid matrix in bounded SGEMM batches.

use superkmeans::gemm::sgemm_row_major_b_transposed;

use crate::collector::sort_key::NaturalComparator;
use crate::collector::TopNComputer;
use crate::schema::{Metric, VectorOptions};
use crate::vector::Similarity;

const ASSIGN_ROW_TILE: usize = 256;
const ASSIGN_CENTROID_TILE: usize = 1024;

fn best_score_position(scores: &[f32]) -> usize {
    let ordered = |score: f32| {
        let bits = score.to_bits() as i32;
        bits ^ (((bits >> 31) as u32) >> 1) as i32
    };
    // An integer maximum vectorizes while preserving Similarity's total order.
    let best = scores.iter().copied().map(ordered).max().unwrap();
    scores
        .iter()
        .position(|&score| ordered(score) == best)
        .unwrap()
}

pub(crate) struct BatchAssigner {
    centroids: Vec<f32>,
    centroid_norms: Vec<f32>,
    dim: usize,
    metric: Metric,
    scores: Vec<f32>,
}

impl BatchAssigner {
    pub(crate) fn new(centroids: Vec<f32>, options: &VectorOptions) -> Self {
        let dim = options.dim();
        assert!(dim > 0 && !centroids.is_empty() && centroids.len() % dim == 0);
        let centroid_norms = if options.metric() == Metric::L2 {
            superkmeans::squared_norms(&centroids, centroids.len() / dim, dim)
        } else {
            Vec::new()
        };
        Self {
            centroids,
            centroid_norms,
            dim,
            metric: options.metric(),
            scores: Vec::new(),
        }
    }

    pub(crate) fn assign_cells(
        &mut self,
        values: &[f32],
        cells_per_vector: usize,
    ) -> Vec<Vec<usize>> {
        assert_eq!(values.len() % self.dim, 0);
        let num_centroids = self.centroids.len() / self.dim;
        assert!((1..=num_centroids).contains(&cells_per_vector));
        let num_rows = values.len() / self.dim;
        let row_norms = if self.metric == Metric::L2 {
            superkmeans::squared_norms(values, num_rows, self.dim)
        } else {
            Vec::new()
        };
        self.scores.resize(
            num_rows.min(ASSIGN_ROW_TILE) * num_centroids.min(ASSIGN_CENTROID_TILE),
            0.0,
        );
        let mut assignments = Vec::with_capacity(num_rows);
        for (row_tile, rows) in values.chunks(ASSIGN_ROW_TILE * self.dim).enumerate() {
            let row_count = rows.len() / self.dim;
            let mut nearest: Vec<_> = (0..row_count)
                .map(|_| TopNComputer::new_with_comparator(cells_per_vector, NaturalComparator))
                .collect();
            for (tile, centroids) in self
                .centroids
                .chunks(ASSIGN_CENTROID_TILE * self.dim)
                .enumerate()
            {
                let centroid_count = centroids.len() / self.dim;
                sgemm_row_major_b_transposed(
                    row_count,
                    self.dim,
                    centroid_count,
                    rows,
                    centroids,
                    &mut self.scores[..row_count * centroid_count],
                );
                for (row, top) in nearest.iter_mut().enumerate() {
                    if cells_per_vector == 1 && self.metric != Metric::L2 {
                        let scores = &self.scores[row * centroid_count..(row + 1) * centroid_count];
                        let col = best_score_position(scores);
                        top.push(
                            Similarity::new(scores[col]),
                            tile * ASSIGN_CENTROID_TILE + col,
                        );
                        continue;
                    }
                    for col in 0..centroid_count {
                        let centroid = tile * ASSIGN_CENTROID_TILE + col;
                        let dot = self.scores[row * centroid_count + col];
                        let score = match self.metric {
                            Metric::L2 => {
                                let norms = row_norms[row_tile * ASSIGN_ROW_TILE + row]
                                    + self.centroid_norms[centroid];
                                let distance = norms - 2.0 * dot;
                                let error = (norms + 2.0 * dot.abs())
                                    * (2.0 * self.dim as f32 * f32::EPSILON);
                                if !distance.is_finite() || distance <= error {
                                    Metric::L2
                                        .similarity(
                                            &rows[row * self.dim..(row + 1) * self.dim],
                                            &centroids[col * self.dim..(col + 1) * self.dim],
                                        )
                                        .score()
                                } else {
                                    -distance
                                }
                            }
                            // Cosine rows and centroids are normalized before assignment.
                            Metric::Cosine | Metric::Dot => dot,
                        };
                        top.push(Similarity::new(score), centroid);
                    }
                }
            }
            assignments.extend(nearest.into_iter().map(|top| {
                top.into_sorted_vec()
                    .into_iter()
                    .map(|hit| hit.doc)
                    .collect()
            }));
        }
        assignments
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn best_score_preserves_total_order_and_first_tie() {
        let values = [
            f32::from_bits(0xffc00001),
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f32::INFINITY,
            f32::from_bits(0x7fc00001),
            f32::from_bits(0x7fc00002),
        ];
        for &a in &values {
            for &b in &values {
                for &c in &values {
                    let scores = [a, b, c, a];
                    let mut top = TopNComputer::new_with_comparator(1, NaturalComparator);
                    for (id, &score) in scores.iter().enumerate() {
                        top.push(Similarity::new(score), id);
                    }
                    let expected = top.into_sorted_vec();
                    let actual = best_score_position(&scores);
                    assert_eq!(actual, expected[0].doc);
                    assert_eq!(
                        scores[actual].to_bits(),
                        expected[0].sort_key.score().to_bits()
                    );
                }
            }
        }
    }

    /// Pins the Dot selection semantics: cells follow RAW dot — the
    /// query-time router's ranking — not angular order. Centroid norms are
    /// deliberately unequal so the two orderings disagree.
    #[test]
    fn dot_selector_uses_raw_dot_not_angular() {
        let centroids: Vec<f32> = vec![
            10.0, 0.0, // long, off-direction: dot 10, cosine 0.45
            0.0, 1.0, // short, near-direction: dot 2, cosine 0.89
            7.0, 7.0, // long, near-direction: dot 21, cosine 0.95
        ];
        let mut assigner = BatchAssigner::new(centroids, &VectorOptions::new(2, Metric::Dot));
        let picked = assigner.assign_cells(&[1.0_f32, 2.0], 3).remove(0);
        // Raw-dot order: [7,7] (21), then [10,0] (10), then [0,1] (2).
        // Angular order would put [0,1] ahead of [10,0].
        assert_eq!(picked, vec![2, 0, 1], "must rank by raw dot");
    }

    #[test]
    fn assign_cells_is_nearest_first_and_order_preserving() {
        let mut assigner = BatchAssigner::new(
            vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0],
            &VectorOptions::new(2, Metric::L2),
        );
        let values: Vec<f32> = vec![
            1.0, 0.0, // nearest 0, then 1
            9.0, 1.0, // nearest 1, then 0
            0.5, 9.0, // nearest 2, then 0
        ];
        let cells = assigner.assign_cells(&values, 2);
        assert_eq!(cells, vec![vec![0, 1], vec![1, 0], vec![2, 0]]);
    }

    #[test]
    fn batches_match_scalar_assignment_across_tiles_and_metrics() {
        let dim = 9;
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let mut centroids: Vec<f32> = (0..(ASSIGN_CENTROID_TILE + 7) * dim)
                .map(|_| rng.random_range(-20..=20) as f32)
                .collect();
            let mut values: Vec<f32> = (0..(ASSIGN_ROW_TILE + 3) * dim)
                .map(|_| rng.random_range(-20..=20) as f32)
                .collect();
            if metric == Metric::Cosine {
                for row in centroids
                    .chunks_exact_mut(dim)
                    .chain(values.chunks_exact_mut(dim))
                {
                    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                    for x in row {
                        *x /= norm;
                    }
                }
            }
            let mut assigner =
                BatchAssigner::new(centroids.clone(), &VectorOptions::new(dim, metric));
            for replicas in [1, 3] {
                let assigned = assigner.assign_cells(&values, replicas);
                for (row, actual) in values.chunks_exact(dim).zip(assigned) {
                    let mut expected: Vec<_> = centroids
                        .chunks_exact(dim)
                        .enumerate()
                        .map(|(id, centroid)| (metric.similarity(row, centroid), id))
                        .collect();
                    expected.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                    assert_eq!(
                        actual,
                        expected[..replicas].iter().map(|c| c.1).collect::<Vec<_>>(),
                        "{metric:?}"
                    );
                }
            }
            assert!(assigner.assign_cells(&[], 1).is_empty());
            assert_eq!(assigner.assign_cells(&values[..dim], 1).len(), 1);
            assert!(assigner.scores.capacity() <= ASSIGN_ROW_TILE * ASSIGN_CENTROID_TILE);
        }
    }

    #[test]
    fn ties_and_positive_dot_scores_across_centroid_tiles() {
        let mut centroids = vec![0.0; (ASSIGN_CENTROID_TILE + 2) * 2];
        centroids[..2].copy_from_slice(&[10.0, 0.0]);
        centroids[ASSIGN_CENTROID_TILE * 2..].copy_from_slice(&[1.0, 0.0, 10.0, 0.0]);
        let mut assigner = BatchAssigner::new(centroids, &VectorOptions::new(2, Metric::Dot));
        assert_eq!(
            assigner.assign_cells(&[1.0, 0.0], 2),
            vec![vec![0, ASSIGN_CENTROID_TILE + 1]]
        );
        assert_eq!(assigner.assign_cells(&[1.0, 0.0], 1), vec![vec![0]]);
        assert_eq!(assigner.assign_cells(&[0.0, 0.0], 1), vec![vec![0]]);
        assert_eq!(assigner.assign_cells(&[0.0, 0.0], 3), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn l2_assignment_handles_cancellation_and_overflow() {
        let mut assigner = BatchAssigner::new(
            vec![1e6, 1e6, 1e6 + 1.0, 1e6],
            &VectorOptions::new(2, Metric::L2),
        );
        assert_eq!(assigner.assign_cells(&[1e6 + 1.0, 1e6], 1), vec![vec![1]]);
        assert_eq!(assigner.assign_cells(&[3e38, 3e38], 2), vec![vec![0, 1]]);
    }
}
