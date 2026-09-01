use std::collections::HashSet;
use std::time::Instant;

use cascade::{
    band_filter, encode_layers, estimate_prepared_fp, kth, prepare_fp_query,
    reconstruct_first_space, LayerSpec,
};
use fht::Rotation;
use grid_plane::{build_lut, estimate as estimate_grid};
use quant_model::{build_grid, empirical_sigma, measure_cal, sigma_from_rho, Grid, DEFAULT_CAL};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use sign_plane::{encode as encode_sign, estimate_asym, estimate_fp, prepare_query, QueryPlanes};

const D: usize = 768;
const VECTOR_COUNT: usize = 1_000;

fn main() {
    println!(
        "MEASURE date=2026-08-20 arch={} os={}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    measure_calibration();
    measure_sign_rows();
    measure_layered();
    measure_survivors();
}

fn measure_calibration() {
    for d in [128, 768, 1536] {
        let cal = measure_cal(d, 4, VECTOR_COUNT);
        println!("CAL d={d} bits=4 n={VECTOR_COUNT} cal={cal:.6}");
    }
}

fn measure_sign_rows() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x0051_5349_474e_0001);
    let queries: Vec<Vec<f32>> = (0..100).map(|_| random_unit(&mut rng, D, 1.0)).collect();
    let q4: Vec<QueryPlanes> = queries
        .iter()
        .map(|query| prepare_query(query, 4))
        .collect();
    let q1: Vec<QueryPlanes> = queries
        .iter()
        .map(|query| prepare_query(query, 1))
        .collect();
    let mut truth_energy = 0.0_f64;
    let mut fp_error = 0.0_f64;
    let mut q4_error = 0.0_f64;
    let mut q1_error = 0.0_f64;
    for _ in 0..VECTOR_COUNT {
        let vector = random_unit(&mut rng, D, 1.0);
        let mut words = vec![0_u64; D / 64];
        let scale = encode_sign(&vector, &mut words);
        for ((query, q4), q1) in queries.iter().zip(&q4).zip(&q1) {
            let truth = dot(query, &vector);
            truth_energy += f64::from(truth).powi(2);
            fp_error += f64::from(estimate_fp(&words, scale, query) - truth).powi(2);
            q4_error += f64::from(estimate_asym(&words, scale, q4) - truth).powi(2);
            q1_error += f64::from(estimate_asym(&words, scale, q1) - truth).powi(2);
        }
    }
    println!(
        "SIGN_RHO d={D} n={VECTOR_COUNT} batch=100 fp={:.6} bq4={:.6} bq1={:.6}",
        (fp_error / truth_energy).sqrt(),
        (q4_error / truth_energy).sqrt(),
        (q1_error / truth_energy).sqrt()
    );
}

fn measure_layered() {
    let specs = specs();
    let grids = grids();
    let mut rng = ChaCha8Rng::seed_from_u64(0x004d_4c41_5945_5201);
    let queries: Vec<Vec<f32>> = (0..32).map(|_| random_unit(&mut rng, D, 1.0)).collect();
    let prepared: Vec<_> = queries
        .iter()
        .map(|query| prepare_fp_query(query, &specs))
        .collect();
    let mut error_energy = 0.0_f64;
    let mut signal_energy = 0.0_f64;
    let mut estimates = Vec::with_capacity(32 * VECTOR_COUNT);
    let mut truths = Vec::with_capacity(32 * VECTOR_COUNT);
    for _ in 0..VECTOR_COUNT {
        let vector = random_unit(&mut rng, D, 1.0);
        let encoded = encode_layers(&vector, None, &specs, &grids);
        let mut first_space = vector.clone();
        Rotation::new(D, specs[0].seed).apply(&mut first_space);
        let reconstructed = reconstruct_first_space(&encoded, &specs, &grids, D);
        for (&actual, estimated) in first_space.iter().zip(reconstructed) {
            error_energy += f64::from(actual - estimated).powi(2);
            signal_energy += f64::from(actual).powi(2);
        }
        for (query, prepared) in queries.iter().zip(&prepared) {
            estimates.push(estimate_prepared_fp(&encoded, prepared, &specs, &grids, D));
            truths.push(dot(query, &vector));
        }
    }
    let rho = (error_energy / signal_energy).sqrt();
    let empirical = empirical_sigma(&estimates, &truths);
    let model = sigma_from_rho(grids[0].rho_model * grids[1].rho_model, D, DEFAULT_CAL);
    println!(
        "LAYERED d={D} n={VECTOR_COUNT} rho={rho:.6} empirical_sigma={empirical:.8} \
         model_sigma={model:.8} ratio={:.6}",
        empirical / model
    );
}

struct Cluster {
    start: usize,
    end: usize,
    q1: QueryPlanes,
    lut2: Vec<f32>,
}

fn measure_survivors() {
    const CLUSTERS: usize = 500;
    const PER_CLUSTER: usize = 100;
    const COUNT: usize = CLUSTERS * PER_CLUSTER;
    const K: usize = 10;

    let specs = specs();
    let grids = grids();
    let mut rng = ChaCha8Rng::seed_from_u64(0x0053_5552_5649_5645);
    let mut clusters = Vec::with_capacity(CLUSTERS);
    let mut candidates = Vec::with_capacity(COUNT);
    let mut truths = Vec::with_capacity(COUNT);
    for cluster_index in 0..CLUSTERS {
        let query_residual = random_unit(&mut rng, D, 1.0);
        let mut u1 = query_residual.clone();
        Rotation::new(D, specs[0].seed).apply(&mut u1);
        let q1 = prepare_query(&u1, 4);
        let mut u2 = u1;
        Rotation::new(D, specs[1].seed).apply(&mut u2);
        let lut2 = build_lut(&u2, &grids[1].points, 4);
        let start = cluster_index * PER_CLUSTER;
        for _ in 0..PER_CLUSTER {
            let residual = random_unit(&mut rng, D, 0.3);
            truths.push(dot(&query_residual, &residual));
            candidates.push(encode_layers(&residual, None, &specs, &grids));
        }
        clusters.push(Cluster {
            start,
            end: start + PER_CLUSTER,
            q1,
            lut2,
        });
    }
    let mut exact_indices: Vec<usize> = (0..COUNT).collect();
    exact_indices.sort_unstable_by(|&a, &b| truths[b].total_cmp(&truths[a]));
    let exact_top: HashSet<u32> = exact_indices[..K]
        .iter()
        .map(|&index| index as u32)
        .collect();

    let sigma1 = (grids[0].rho_model * 0.3 / (D as f64).sqrt()) as f32;
    let sigma2 = (grids[0].rho_model * grids[1].rho_model * 0.3 / (D as f64).sqrt()) as f32;
    for kappa in [2.0_f32, 3.0, 4.0] {
        let started = Instant::now();
        let mut first_scores = vec![0.0_f32; COUNT];
        for cluster in &clusters {
            for index in cluster.start..cluster.end {
                let words = bytes_to_words(&candidates[index].codes[0]);
                first_scores[index] =
                    estimate_asym(&words, candidates[index].scales[0], &cluster.q1);
            }
        }
        let first_sigmas = vec![sigma1; COUNT];
        let (first_kth_index, first_kth) = kth(&first_scores, K);
        let first_survivors = band_filter(
            &first_scores,
            &first_sigmas,
            kappa,
            first_kth - kappa * first_sigmas[first_kth_index],
        );

        let mut second_scores = Vec::with_capacity(first_survivors.len());
        for &global_index in &first_survivors {
            let index = global_index as usize;
            let cluster = &clusters[index / PER_CLUSTER];
            let second = estimate_grid(
                &candidates[index].codes[1],
                candidates[index].scales[1],
                &cluster.lut2,
                D,
                4,
            );
            second_scores.push(first_scores[index] + second);
        }
        let second_sigmas = vec![sigma2; second_scores.len()];
        let (second_kth_index, second_kth) = kth(&second_scores, K);
        let local_final = band_filter(
            &second_scores,
            &second_sigmas,
            kappa,
            second_kth - kappa * second_sigmas[second_kth_index],
        );
        let final_survivors: HashSet<u32> = local_final
            .iter()
            .map(|&local| first_survivors[local as usize])
            .collect();
        let recall = exact_top.intersection(&final_survivors).count() as f64 / K as f64;
        if kappa == 3.0 {
            assert!(recall >= 0.99, "κ=3 candidate recall was {recall}");
        }
        let elapsed = started.elapsed();
        println!(
            "SURVIVORS d={D} candidates={COUNT} k={K} kappa={kappa:.0} s1={:.6} s2={:.6} \
             recall={recall:.3} ns_per_candidate={:.3}",
            first_survivors.len() as f64 / COUNT as f64,
            final_survivors.len() as f64 / COUNT as f64,
            elapsed.as_nanos() as f64 / COUNT as f64
        );
    }
}

fn specs() -> [LayerSpec; 2] {
    [
        LayerSpec {
            bits: 1,
            seed: 11,
            rotate: true,
        },
        LayerSpec {
            bits: 4,
            seed: 22,
            rotate: true,
        },
    ]
}

fn grids() -> [Grid; 2] {
    [build_grid(D, 1), build_grid(D, 4)]
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight bytes")))
        .collect()
}

fn random_unit(rng: &mut ChaCha8Rng, d: usize, norm: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(d);
    while values.len() < d {
        let u1 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
        let u2 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        values.push((radius * angle.cos()) as f32);
        if values.len() < d {
            values.push((radius * angle.sin()) as f32);
        }
    }
    let current = values
        .iter()
        .map(|&value| value * value)
        .sum::<f32>()
        .sqrt();
    for value in &mut values {
        *value *= norm / current;
    }
    values
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
