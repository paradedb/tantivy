use std::collections::{BTreeMap, HashSet};
use std::hint::black_box;
use std::time::{Duration, Instant};

use cascade::{
    band_filter, encode_layers, kth, prepare_centroid, prepare_fp_query, Encoded, LayerSpec,
};
use fht::Rotation;
use grid_plane::{
    build_lut, decode as decode_grid, encode as encode_grid, estimate as estimate_grid,
};
use quant_model::f16::{f16_to_f32, f32_to_f16};
use quant_model::{build_grid, Grid};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use sign_plane::{
    encode as encode_sign, estimate_asym, prepare_query, unpack as unpack_sign, QueryPlanes,
};

const D: usize = 768;
const CLUSTERS: usize = 500;
const PER_CLUSTER: usize = 100;
const CANDIDATES: usize = CLUSTERS * PER_CLUSTER;
const K: usize = 10;
const RADIUS: f32 = 0.8;
const QUERY_SEEDS: usize = 200;
const PLANTED_SEEDS: usize = 100;
const SIGN_RHO: f64 = 0.602469;

#[derive(Clone)]
struct Schedule {
    label: &'static str,
    specs: Vec<LayerSpec>,
    grids: Vec<Grid>,
    rho: f64,
}

struct PreparedSegmentQuery {
    first_sign: QueryPlanes,
    second_sign: Option<QueryPlanes>,
    second_lut: Option<Vec<f32>>,
}

struct CascadeData {
    query: PreparedSegmentQuery,
    candidates: Vec<Encoded>,
    truths: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct Observation {
    recall: f64,
    s1: f64,
    s2: f64,
    ns_per_candidate: f64,
}

#[derive(Clone, Copy, Debug)]
struct Aggregate {
    mean_recall: f64,
    p5_recall: f64,
    min_recall: f64,
    misses: usize,
    mean_s1: f64,
    mean_s2: f64,
    mean_ns: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ConfigKey {
    k1: usize,
    k2: usize,
}

impl ConfigKey {
    fn new(k1: f32, k2: f32) -> Self {
        Self {
            k1: (k1 * 100.0).round() as usize,
            k2: (k2 * 100.0).round() as usize,
        }
    }

    fn values(self) -> (f32, f32) {
        (self.k1 as f32 / 100.0, self.k2 as f32 / 100.0)
    }
}

fn main() {
    let schedules = schedules();
    let fp_dot_ns = bench_fp_dot();
    table_e5(&schedules);
    table_e8(&schedules[0]);

    let mut observations: BTreeMap<(&'static str, ConfigKey), Vec<Observation>> = BTreeMap::new();
    collect_e6_e7(&schedules[..2], &mut observations);
    table_e(&schedules[..2], &observations);
    let pareto = table_f(&schedules[..2], &observations);
    table_g(&schedules[..2], &pareto);

    collect_e10_schedule(&schedules[2], &mut observations);
    table_h(&schedules, &observations, fp_dot_ns);
}

fn schedules() -> Vec<Schedule> {
    vec![
        schedule("[1,4]", 4, 0.058493),
        schedule("[1,1]", 1, 0.363015),
        schedule("[1,2]", 2, 0.206326),
    ]
}

fn schedule(label: &'static str, second_bits: u8, rho: f64) -> Schedule {
    Schedule {
        label,
        specs: vec![spec(1, 11, true), spec(second_bits, 22, true)],
        grids: vec![build_grid(D, 1), build_grid(D, second_bits)],
        rho,
    }
}

fn table_e5(schedules: &[Schedule]) {
    let rotated = &schedules[0];
    let norot = Schedule {
        label: "[1,4]-norot",
        specs: vec![spec(1, 11, true), spec(4, 22, false)],
        grids: rotated.grids.clone(),
        rho: 0.074927,
    };
    let components = encode_components(rotated);
    let component_sum: f64 = components.iter().map(|(_, value)| value).sum();
    let total = bench_cluster_encode(rotated);
    let relative = (component_sum - total).abs() / total;
    assert!(
        relative <= 0.15,
        "encode component sum {component_sum:.3} ns differs from total {total:.3} ns by {:.2}%",
        relative * 100.0
    );
    let norot_total = bench_cluster_encode(&norot);

    println!("## E5 encode component breakdown");
    println!();
    println!("| component | ns/vector |");
    println!("|---|---:|");
    for (label, value) in components {
        println!("| {label} | {value:.3} |");
    }
    println!("| components sum | {component_sum:.3} |");
    println!("| total | {total:.3} |");
    println!();
    println!("## Table C — A.2 encode remeasurement");
    println!();
    println!("| schedule | encode ns/v |");
    println!("|---|---:|");
    println!("| [1,4] | {total:.3} |");
    println!("| [1,4]-norot | {norot_total:.3} |");
    println!();
}

fn encode_components(schedule: &Schedule) -> Vec<(&'static str, f64)> {
    let mut rng = ChaCha8Rng::seed_from_u64(0x0045_3543_4f4d_5001);
    let centroid = random_unit(&mut rng, D, 0.7);
    let vectors: Vec<Vec<f32>> = (0..64).map(|_| random_unit(&mut rng, D, RADIUS)).collect();
    let r1 = Rotation::new(D, schedule.specs[0].seed);
    let r2 = Rotation::new(D, schedule.specs[1].seed);

    let rotated_first: Vec<Vec<f32>> = vectors
        .iter()
        .map(|vector| {
            let mut value = vector.clone();
            r1.apply(&mut value);
            value
        })
        .collect();
    let after_sign: Vec<Vec<f32>> = rotated_first
        .iter()
        .map(|value| {
            let mut residual = value.clone();
            let mut words = vec![0_u64; D / 64];
            let scale = f16_to_f32(encode_sign(&residual, &mut words));
            for (coordinate, sign) in residual.iter_mut().zip(unpack_sign(&words, D)) {
                *coordinate -= scale * sign;
            }
            residual
        })
        .collect();
    let rotated_second: Vec<Vec<f32>> = after_sign
        .iter()
        .map(|value| {
            let mut value = value.clone();
            r2.apply(&mut value);
            value
        })
        .collect();

    let rotation = bench_indexed(&vectors, |vector| {
        let mut value = vector.clone();
        r1.apply(&mut value);
        r2.apply(&mut value);
        value
    });
    let sign = bench_indexed(&rotated_first, |value| {
        let mut residual = value.clone();
        let mut words = vec![0_u64; D / 64];
        let scale = f16_to_f32(encode_sign(&residual, &mut words));
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        for (coordinate, sign) in residual.iter_mut().zip(unpack_sign(&words, D)) {
            *coordinate -= scale * sign;
        }
        (residual, bytes)
    });
    let grid = bench_indexed(&rotated_second, |value| {
        let mut residual = value.clone();
        let mut codes = vec![0_u8; grid_plane::packed_len(D, 4)];
        let scale = encode_grid(&residual, &schedule.grids[1].points, 4, &mut codes);
        for (coordinate, reconstruction) in
            residual
                .iter_mut()
                .zip(decode_grid(&codes, &schedule.grids[1].points, D, 4, scale))
        {
            *coordinate -= reconstruction;
        }
        (residual, codes)
    });

    let mut c1 = centroid.clone();
    r1.apply(&mut c1);
    let mut c2 = c1.clone();
    r2.apply(&mut c2);
    let constants = bench_indexed(&rotated_second, |value| {
        let first = dot(&c1, &rotated_first[0]);
        let second = dot(&c2, value);
        first + second
    });
    let f16_scales = bench_ns(|| {
        let a = f32_to_f16(black_box(0.024_f32));
        let b = f32_to_f16(black_box(0.011_f32));
        (a, b)
    });
    vec![
        ("rotation", rotation),
        ("sign encode + pack", sign),
        ("grid encode + pack", grid),
        ("constants", constants),
        ("f16 scales", f16_scales),
    ]
}

fn bench_cluster_encode(schedule: &Schedule) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(0x0045_3542_454e_4348);
    let centroid = random_unit(&mut rng, D, 0.7);
    let context = prepare_centroid(&centroid, &schedule.specs);
    let vectors: Vec<Vec<f32>> = (0..64).map(|_| random_unit(&mut rng, D, RADIUS)).collect();
    bench_indexed(&vectors, |vector| {
        encode_layers(vector, Some(&context), &schedule.specs, &schedule.grids)
    })
}

fn table_e8(schedule: &Schedule) {
    let mut rng = ChaCha8Rng::seed_from_u64(0x0045_3853_504c_4954);
    let mut truth_energy = 0.0_f64;
    let mut fp_error = 0.0_f64;
    let mut bq4_error = 0.0_f64;
    let one_spec = [schedule.specs[0]];
    let one_grid = [schedule.grids[0].clone()];
    for _ in 0..100 {
        let query = random_unit(&mut rng, D, 1.0);
        let mut rotated_query = query.clone();
        Rotation::new(D, one_spec[0].seed).apply(&mut rotated_query);
        let fp_query = prepare_fp_query(&query, &one_spec);
        let bq4_query = prepare_query(&rotated_query, 4);
        for _ in 0..10 {
            let residual_query = random_unit(&mut rng, D, 1.0);
            let centroid: Vec<f32> = query
                .iter()
                .zip(&residual_query)
                .map(|(&q, &u)| q - u)
                .collect();
            let context = prepare_centroid(&centroid, &one_spec);
            for _ in 0..100 {
                let residual = random_unit(&mut rng, D, RADIUS);
                let encoded = encode_layers(&residual, Some(&context), &one_spec, &one_grid);
                let words = bytes_to_words(&encoded.codes[0]);
                let truth = dot(&residual_query, &residual);
                let fp = cascade::estimate_prepared_fp_split(
                    &encoded, &fp_query, &one_spec, &one_grid, D,
                );
                let bq4 =
                    estimate_asym(&words, encoded.scales[0], &bq4_query) - encoded.constants[0];
                truth_energy += f64::from(truth).powi(2);
                fp_error += f64::from(fp - truth).powi(2);
                bq4_error += f64::from(bq4 - truth).powi(2);
            }
        }
    }
    println!("## E8 split-form estimator rows");
    println!();
    println!("| query side | d | N | ρ |");
    println!("|---|---:|---:|---:|");
    println!(
        "| fp32 reference | {D} | 1000 × 100 queries | {:.6} |",
        (fp_error / truth_energy).sqrt()
    );
    println!(
        "| B_q=4 asymmetric | {D} | 1000 × 100 queries | {:.6} |",
        (bq4_error / truth_energy).sqrt()
    );
    println!();
}

fn collect_e6_e7(
    schedules: &[Schedule],
    observations: &mut BTreeMap<(&'static str, ConfigKey), Vec<Observation>>,
) {
    let e6_kappas = [2.0_f32, 2.25, 2.5, 2.75, 3.0];
    let e7_k1 = e6_kappas;
    let e7_k2 = [3.0_f32, 3.5, 4.0];
    for schedule in schedules {
        for seed in 0..QUERY_SEEDS {
            let data = build_cascade_data(schedule, seed as u64, None);
            for kappa in e6_kappas {
                push_observation(observations, schedule, kappa, kappa, &data);
            }
            for k1 in e7_k1 {
                for k2 in e7_k2 {
                    if (k1 - 3.0).abs() < f32::EPSILON && (k2 - 3.0).abs() < f32::EPSILON {
                        continue;
                    }
                    push_observation(observations, schedule, k1, k2, &data);
                }
            }
        }
    }
}

fn push_observation(
    observations: &mut BTreeMap<(&'static str, ConfigKey), Vec<Observation>>,
    schedule: &Schedule,
    k1: f32,
    k2: f32,
    data: &CascadeData,
) {
    observations
        .entry((schedule.label, ConfigKey::new(k1, k2)))
        .or_default()
        .push(run_cascade(schedule, data, k1, k2));
}

fn table_e(
    schedules: &[Schedule],
    observations: &BTreeMap<(&'static str, ConfigKey), Vec<Observation>>,
) {
    println!("## Table E");
    println!();
    println!(
        "| schedule | κ | mean recall | p5 recall | min recall | queries-with-miss /200 | s1 | s2 \
         |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for schedule in schedules {
        for kappa in [2.0_f32, 2.25, 2.5, 2.75, 3.0] {
            let aggregate =
                aggregate(&observations[&(schedule.label, ConfigKey::new(kappa, kappa))]);
            println!(
                "| {} | {:.2} | {:.6} | {:.6} | {:.6} | {}/200 | {:.6} | {:.6} |",
                schedule.label,
                kappa,
                aggregate.mean_recall,
                aggregate.p5_recall,
                aggregate.min_recall,
                aggregate.misses,
                aggregate.mean_s1,
                aggregate.mean_s2
            );
        }
    }
    println!();
}

fn table_f(
    schedules: &[Schedule],
    observations: &BTreeMap<(&'static str, ConfigKey), Vec<Observation>>,
) -> BTreeMap<&'static str, ConfigKey> {
    let keys: Vec<ConfigKey> = [2.0_f32, 2.25, 2.5, 2.75, 3.0]
        .into_iter()
        .flat_map(|k1| [3.0_f32, 3.5, 4.0].map(|k2| ConfigKey::new(k1, k2)))
        .collect();
    println!("## Table F");
    println!();
    println!(
        "| schedule | κ1 | κ2 | mean recall | p5 recall | s1 | s2 | end-to-end ns/candidate |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for schedule in schedules {
        let rows: Vec<(ConfigKey, Aggregate)> = keys
            .iter()
            .copied()
            .map(|key| (key, aggregate(&observations[&(schedule.label, key)])))
            .collect();
        for (key, row) in &rows {
            let (k1, k2) = key.values();
            println!(
                "| {} | {k1:.2} | {k2:.1} | {:.6} | {:.6} | {:.6} | {:.6} | {:.3} |",
                schedule.label,
                row.mean_recall,
                row.p5_recall,
                row.mean_s1,
                row.mean_s2,
                row.mean_ns
            );
        }
    }
    println!();
    println!("| Pareto schedule | κ1 | κ2 | mean recall | end-to-end ns/candidate |");
    println!("|---|---:|---:|---:|---:|");
    let mut selected = BTreeMap::new();
    for schedule in schedules {
        let rows: Vec<(ConfigKey, Aggregate)> = keys
            .iter()
            .copied()
            .map(|key| (key, aggregate(&observations[&(schedule.label, key)])))
            .collect();
        let frontier: Vec<(ConfigKey, Aggregate)> = rows
            .iter()
            .copied()
            .filter(|(_, candidate)| {
                !rows.iter().any(|(_, other)| {
                    other.mean_recall >= candidate.mean_recall
                        && other.mean_ns <= candidate.mean_ns
                        && (other.mean_recall > candidate.mean_recall
                            || other.mean_ns < candidate.mean_ns)
                })
            })
            .collect();
        for (key, row) in &frontier {
            let (k1, k2) = key.values();
            println!(
                "| {} | {k1:.2} | {k2:.1} | {:.6} | {:.3} |",
                schedule.label, row.mean_recall, row.mean_ns
            );
        }
        let best = frontier
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.mean_recall
                    .total_cmp(&b.mean_recall)
                    .then_with(|| b.mean_ns.total_cmp(&a.mean_ns))
            })
            .expect("non-empty Pareto frontier")
            .0;
        selected.insert(schedule.label, best);
    }
    println!();
    selected
}

fn table_g(schedules: &[Schedule], pareto: &BTreeMap<&'static str, ConfigKey>) {
    println!("## Table G");
    println!();
    println!("| z | schedule | config | mean recall | s1 | s2 |");
    println!("|---:|---|---|---:|---:|---:|");
    for z in [2.0_f32, 4.0, 8.0] {
        for schedule in schedules {
            let baseline = ConfigKey::new(3.0, 3.0);
            let best = pareto[schedule.label];
            let configs = [("κ1=3, κ2=3", baseline), ("Pareto max-recall", best)];
            let mut samples = [
                Vec::with_capacity(PLANTED_SEEDS),
                Vec::with_capacity(PLANTED_SEEDS),
            ];
            for seed in 0..PLANTED_SEEDS {
                let data = build_cascade_data(schedule, seed as u64, Some(z));
                for (index, (_, config)) in configs.iter().enumerate() {
                    let (k1, k2) = config.values();
                    samples[index].push(run_cascade(schedule, &data, k1, k2));
                }
            }
            for (index, (label, config)) in configs.into_iter().enumerate() {
                let (k1, k2) = config.values();
                let aggregate = aggregate(&samples[index]);
                println!(
                    "| {z:.0} | {} | {label} ({k1:.2},{k2:.1}) | {:.6} | {:.6} | {:.6} |",
                    schedule.label, aggregate.mean_recall, aggregate.mean_s1, aggregate.mean_s2
                );
            }
        }
    }
    println!();
}

fn collect_e10_schedule(
    schedule: &Schedule,
    observations: &mut BTreeMap<(&'static str, ConfigKey), Vec<Observation>>,
) {
    for seed in 0..QUERY_SEEDS {
        let data = build_cascade_data(schedule, seed as u64, None);
        for kappa in [2.0_f32, 3.0] {
            push_observation(observations, schedule, kappa, kappa, &data);
        }
    }
}

fn table_h(
    schedules: &[Schedule],
    observations: &BTreeMap<(&'static str, ConfigKey), Vec<Observation>>,
    fp_dot_ns: f64,
) {
    println!("## Table H");
    println!();
    println!(
        "| schedule | κ | R | plane bytes | rerank bytes | total bytes | grand compute µs | code \
         bits/dim stored |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for schedule in schedules {
        for kappa in [2.0_f32, 3.0] {
            let row = aggregate(&observations[&(schedule.label, ConfigKey::new(kappa, kappa))]);
            let rerank_count = row.mean_s2 * CANDIDATES as f64;
            let plane1_bytes = CANDIDATES as f64 * D as f64 / 8.0;
            let plane2_bytes =
                row.mean_s1 * CANDIDATES as f64 * D as f64 * f64::from(schedule.specs[1].bits)
                    / 8.0;
            let plane_bytes = plane1_bytes + plane2_bytes;
            let rerank_bytes = rerank_count * D as f64 * 4.0;
            let total_bytes = plane_bytes + rerank_bytes;
            let grand_compute =
                (row.mean_ns * CANDIDATES as f64 + rerank_count * fp_dot_ns) / 1_000.0;
            let stored_bits: u8 = schedule.specs.iter().map(|spec| spec.bits).sum();
            println!(
                "| {} | {kappa:.0} | {rerank_count:.3} | {plane_bytes:.0} | {rerank_bytes:.0} | \
                 {total_bytes:.0} | {grand_compute:.3} | {stored_bits} |",
                schedule.label
            );
        }
    }
    println!();
}

fn build_cascade_data(schedule: &Schedule, query_seed: u64, plant_z: Option<f32>) -> CascadeData {
    let seed = 0x0041_3251_5545_5259 ^ query_seed.wrapping_mul(0x9e37_79b9);
    let (query_vector, natural_kth, population_sigma) = if plant_z.is_some() {
        let mut first_pass = ChaCha8Rng::seed_from_u64(seed);
        let query = random_unit(&mut first_pass, D, 1.0);
        let mut truths = Vec::with_capacity(CANDIDATES);
        for _ in 0..CLUSTERS {
            let residual_query = random_unit(&mut first_pass, D, 1.0);
            for _ in 0..PER_CLUSTER {
                let residual = random_unit(&mut first_pass, D, RADIUS);
                truths.push(dot(&residual_query, &residual));
            }
        }
        let (_, natural_kth) = kth(&truths, K);
        let mean = truths.iter().map(|&value| f64::from(value)).sum::<f64>() / truths.len() as f64;
        let variance = truths
            .iter()
            .map(|&value| (f64::from(value) - mean).powi(2))
            .sum::<f64>()
            / truths.len() as f64;
        (query, natural_kth, variance.sqrt() as f32)
    } else {
        (Vec::new(), 0.0, 0.0)
    };

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let query = random_unit(&mut rng, D, 1.0);
    if plant_z.is_some() {
        debug_assert_eq!(query, query_vector);
    }
    let prepared_query = prepare_segment_query(&query, schedule);
    let mut candidates = Vec::with_capacity(CANDIDATES);
    let mut truths = Vec::with_capacity(CANDIDATES);

    for cluster in 0..CLUSTERS {
        let residual_query = random_unit(&mut rng, D, 1.0);
        let centroid: Vec<f32> = query
            .iter()
            .zip(&residual_query)
            .map(|(&q, &u)| q - u)
            .collect();
        let context = prepare_centroid(&centroid, &schedule.specs);
        for member in 0..PER_CLUSTER {
            let mut residual = random_unit(&mut rng, D, RADIUS);
            let index = cluster * PER_CLUSTER + member;
            if let Some(z) = plant_z {
                if cluster < K && member == 0 {
                    let current = dot(&residual_query, &residual);
                    let target = natural_kth + z * population_sigma;
                    for (value, &direction) in residual.iter_mut().zip(&residual_query) {
                        *value += (target - current) * direction;
                    }
                }
            }
            truths.push(dot(&residual_query, &residual));
            candidates.push(encode_layers(
                &residual,
                Some(&context),
                &schedule.specs,
                &schedule.grids,
            ));
            debug_assert_eq!(candidates.len(), index + 1);
        }
    }
    CascadeData {
        query: prepared_query,
        candidates,
        truths,
    }
}

fn prepare_segment_query(query: &[f32], schedule: &Schedule) -> PreparedSegmentQuery {
    let mut current = query.to_vec();
    let mut first_sign = None;
    let mut second_sign = None;
    let mut second_lut = None;
    for (layer, (spec, grid)) in schedule.specs.iter().zip(&schedule.grids).enumerate() {
        if layer == 0 || spec.rotate {
            Rotation::new(D, spec.seed).apply(&mut current);
        }
        if layer == 0 {
            first_sign = Some(prepare_query(&current, 4));
        } else if spec.bits == 1 {
            second_sign = Some(prepare_query(&current, 4));
        } else {
            second_lut = Some(build_lut(&current, &grid.points, spec.bits));
        }
    }
    PreparedSegmentQuery {
        first_sign: first_sign.expect("sign first layer"),
        second_sign,
        second_lut,
    }
}

fn run_cascade(schedule: &Schedule, data: &CascadeData, k1: f32, k2: f32) -> Observation {
    let mut exact_indices: Vec<usize> = (0..CANDIDATES).collect();
    exact_indices.sort_unstable_by(|&a, &b| data.truths[b].total_cmp(&data.truths[a]));
    let exact_top: HashSet<u32> = exact_indices[..K]
        .iter()
        .map(|&index| index as u32)
        .collect();
    let started = Instant::now();

    let mut first_scores = Vec::with_capacity(CANDIDATES);
    for encoded in &data.candidates {
        let words = bytes_to_words(&encoded.codes[0]);
        first_scores.push(
            estimate_asym(&words, encoded.scales[0], &data.query.first_sign) - encoded.constants[0],
        );
    }
    let sigma1 = (SIGN_RHO * f64::from(RADIUS) / (D as f64).sqrt()) as f32;
    let first_sigmas = vec![sigma1; CANDIDATES];
    let (first_index, first_value) = kth(&first_scores, K);
    let first_survivors = band_filter(
        &first_scores,
        &first_sigmas,
        k1,
        first_value - k1 * first_sigmas[first_index],
    );

    let mut second_scores = Vec::with_capacity(first_survivors.len());
    for &global in &first_survivors {
        let index = global as usize;
        let encoded = &data.candidates[index];
        let second = if schedule.specs[1].bits == 1 {
            let words = bytes_to_words(&encoded.codes[1]);
            estimate_asym(
                &words,
                encoded.scales[1],
                data.query
                    .second_sign
                    .as_ref()
                    .expect("second-layer sign query"),
            )
        } else {
            estimate_grid(
                &encoded.codes[1],
                encoded.scales[1],
                data.query.second_lut.as_ref().expect("second-layer LUT"),
                D,
                schedule.specs[1].bits,
            )
        } - encoded.constants[1];
        second_scores.push(first_scores[index] + second);
    }
    let sigma2 = (schedule.rho * f64::from(RADIUS) / (D as f64).sqrt()) as f32;
    let second_sigmas = vec![sigma2; second_scores.len()];
    let (second_index, second_value) = kth(&second_scores, K);
    let local = band_filter(
        &second_scores,
        &second_sigmas,
        k2,
        second_value - k2 * second_sigmas[second_index],
    );
    let final_set: HashSet<u32> = local
        .iter()
        .map(|&local_index| first_survivors[local_index as usize])
        .collect();
    Observation {
        recall: exact_top.intersection(&final_set).count() as f64 / K as f64,
        s1: first_survivors.len() as f64 / CANDIDATES as f64,
        s2: final_set.len() as f64 / CANDIDATES as f64,
        ns_per_candidate: started.elapsed().as_nanos() as f64 / CANDIDATES as f64,
    }
}

fn aggregate(values: &[Observation]) -> Aggregate {
    assert!(!values.is_empty());
    let mut recalls: Vec<f64> = values.iter().map(|row| row.recall).collect();
    recalls.sort_by(f64::total_cmp);
    let sum = |select: fn(&Observation) -> f64| {
        values.iter().map(select).sum::<f64>() / values.len() as f64
    };
    Aggregate {
        mean_recall: sum(|row| row.recall),
        p5_recall: recalls[(values.len() * 5).div_ceil(100) - 1],
        min_recall: recalls[0],
        misses: values.iter().filter(|row| row.recall < 1.0).count(),
        mean_s1: sum(|row| row.s1),
        mean_s2: sum(|row| row.s2),
        mean_ns: sum(|row| row.ns_per_candidate),
    }
}

fn bench_fp_dot() -> f64 {
    let a: Vec<f32> = (0..D).map(|i| (i as f32 * 0.013).sin()).collect();
    let b: Vec<f32> = (0..D).map(|i| (i as f32 * 0.017).cos()).collect();
    bench_ns(|| dot(black_box(&a), black_box(&b)))
}

fn bench_indexed<T, F, O>(values: &[T], mut operation: F) -> f64
where F: FnMut(&T) -> O {
    let mut index = 0;
    bench_ns(|| {
        let result = operation(&values[index]);
        index = (index + 1) % values.len();
        result
    })
}

fn bench_ns<F, T>(mut operation: F) -> f64
where F: FnMut() -> T {
    for _ in 0..16 {
        black_box(operation());
    }
    let target = Duration::from_millis(80);
    let mut iterations = 8_u64;
    loop {
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(operation());
        }
        let elapsed = started.elapsed();
        if elapsed >= target {
            return elapsed.as_nanos() as f64 / iterations as f64;
        }
        iterations = iterations.saturating_mul(2);
    }
}

fn spec(bits: u8, seed: u64, rotate: bool) -> LayerSpec {
    LayerSpec { bits, seed, rotate }
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
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
