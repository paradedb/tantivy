use std::collections::{BTreeMap, HashSet};
use std::hint::black_box;
use std::time::{Duration, Instant};

use cascade::{
    band_filter, encode_layers, estimate_prepared_fp, kth, prepare_fp_query,
    reconstruct_first_space, Encoded, LayerSpec,
};
use fht::Rotation;
use grid_plane::{build_lut, encode as encode_grid, estimate as estimate_grid, packed_len, score};
use quant_model::f16::f16_to_f32;
use quant_model::{build_grid, empirical_sigma, sigma_from_rho, Grid, DEFAULT_CAL};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use sign_plane::{
    encode as encode_sign, estimate_asym, prepare_query, score_asym, unpack, QueryPlanes,
};

const D: usize = 768;
const N: usize = 1_000;
const QUERY_BATCH: usize = 32;
const SIGN_SYM_NS: f64 = 2.588;
const SIGN_ASYM_BQ4_NS: f64 = 13.869;
const GRID_B4_NS: f64 = 506.91;

#[derive(Clone)]
struct QuantResult {
    rho: f64,
    sigma_ratio: Option<f64>,
}

#[derive(Clone)]
struct Schedule {
    label: &'static str,
    specs: Vec<LayerSpec>,
    grids: Vec<Grid>,
    rho: f64,
}

struct ClusterQuery {
    start: usize,
    end: usize,
    sign: Option<QueryPlanes>,
    first_lut: Option<Vec<f32>>,
    second_sign: Option<QueryPlanes>,
    second_lut: Option<Vec<f32>>,
}

struct CascadeData {
    clusters: Vec<ClusterQuery>,
    candidates: Vec<Encoded>,
    truths: Vec<f32>,
}

struct CascadeRow {
    s1: Option<f64>,
    s2: f64,
    recall: f64,
    ns_per_candidate: f64,
}

fn main() {
    let mut grids = BTreeMap::new();
    for bits in 1..=4 {
        grids.insert(bits, build_grid(D, bits));
    }

    table_a(&grids);
    let (monolithic, layered) = table_b(&grids);
    let norot = table_c(&grids, &monolithic, &layered);
    table_d(&grids, &monolithic, &layered, &norot);
}

fn table_a(grids: &BTreeMap<u8, Grid>) {
    let fp_rows: Vec<(usize, f64)> = [128, 768, 1536]
        .into_iter()
        .map(|d| {
            let data: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013).sin()).collect();
            let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.017).cos()).collect();
            let ns = bench_ns(|| fp_dot(black_box(&data), black_box(&query)));
            (d, ns)
        })
        .collect();

    let data: Vec<f32> = (0..D).map(|i| (i as f32 * 0.013).sin()).collect();
    let query: Vec<f32> = (0..D).map(|i| (i as f32 * 0.017).cos()).collect();
    let mut data_bits = vec![0_u64; D / 64];
    let mut query_bits = vec![0_u64; D / 64];
    let data_scale = f16_to_f32(encode_sign(&data, &mut data_bits));
    let query_scale = f16_to_f32(encode_sign(&query, &mut query_bits));
    let float_query: Vec<f32> = unpack(&query_bits, D)
        .into_iter()
        .map(|sign| sign * query_scale)
        .collect();
    let decode_ns = bench_ns(|| {
        let decoded: Vec<f32> = unpack(black_box(&data_bits), D)
            .into_iter()
            .map(|sign| sign * data_scale)
            .collect();
        fp_dot(black_box(&decoded), black_box(&float_query))
    });

    let grid = &grids[&4];
    let mut codes = vec![0; packed_len(D, 4)];
    encode_grid(&data, &grid.points, 4, &mut codes);
    let lut = build_lut(&query, &grid.points, 4);
    black_box(score(&codes, &lut, D, 4));

    println!("## Table A");
    println!();
    println!("| kernel | d | ns/vector |");
    println!("|---|---:|---:|");
    for (d, ns) in fp_rows {
        println!("| fp_dot | {d} | {ns:.3} |");
    }
    println!("| decode_then_dot | {D} | {decode_ns:.3} |");
    println!("| sign symmetric | {D} | {SIGN_SYM_NS:.3} |");
    println!("| sign asymmetric B_q=4 | {D} | {SIGN_ASYM_BQ4_NS:.3} |");
    println!("| grid score b=4 | {D} | {GRID_B4_NS:.3} |");
    println!();
    println!(
        "decode_then_dot / sign symmetric: {:.3}x",
        decode_ns / SIGN_SYM_NS
    );
    println!();
}

fn table_b(grids: &BTreeMap<u8, Grid>) -> (BTreeMap<u8, QuantResult>, BTreeMap<u8, QuantResult>) {
    let mut monolithic = BTreeMap::new();
    for bits in 1..=4 {
        let specs = vec![spec(bits, 11, true)];
        let schedule_grids = vec![grids[&bits].clone()];
        monolithic.insert(bits, measure_quantization(&specs, &schedule_grids, None));
    }

    let mut layered = BTreeMap::new();
    for bits in 1..=4 {
        let product = monolithic[&1].rho * monolithic[&bits].rho;
        let specs = vec![spec(1, 11, true), spec(bits, 22, true)];
        let schedule_grids = vec![grids[&1].clone(), grids[&bits].clone()];
        layered.insert(
            bits,
            measure_quantization(&specs, &schedule_grids, Some(product)),
        );
    }

    let plane_scores = measure_plane_scores(grids);

    println!("## Table B");
    println!();
    println!(
        "| schedule | bits total | ρ measured | ρ product-model | σ ratio | plane-2 score ns/v |"
    );
    println!("|---|---:|---:|---:|---:|---:|");
    for bits in 1..=4 {
        let result = &monolithic[&bits];
        println!("| [{bits}] | {bits} | {:.6} | — | — | — |", result.rho);
    }
    for bits in 1..=4 {
        let result = &layered[&bits];
        let product = monolithic[&1].rho * monolithic[&bits].rho;
        println!(
            "| [1,{bits}] | {} | {:.6} | {:.6} | {:.6} | {:.3} |",
            1 + bits,
            result.rho,
            product,
            result.sigma_ratio.expect("layered sigma ratio"),
            plane_scores[&bits]
        );
    }
    println!();
    (monolithic, layered)
}

fn table_c(
    grids: &BTreeMap<u8, Grid>,
    monolithic: &BTreeMap<u8, QuantResult>,
    layered: &BTreeMap<u8, QuantResult>,
) -> BTreeMap<u8, QuantResult> {
    let mut norot = BTreeMap::new();
    for bits in [2, 4] {
        let product = monolithic[&1].rho * monolithic[&bits].rho;
        let specs = vec![spec(1, 11, true), spec(bits, 22, false)];
        let schedule_grids = vec![grids[&1].clone(), grids[&bits].clone()];
        norot.insert(
            bits,
            measure_quantization(&specs, &schedule_grids, Some(product)),
        );
    }

    let vectors = fixed_vectors(64, 0x0045_4e43_4f44_4501, 1.0);
    let rotated_specs = vec![spec(1, 11, true), spec(4, 22, true)];
    let norot_specs = vec![spec(1, 11, true), spec(4, 22, false)];
    let schedule_grids = vec![grids[&1].clone(), grids[&4].clone()];
    let rotated_encode_ns = bench_encode(&vectors, &rotated_specs, &schedule_grids);
    let norot_encode_ns = bench_encode(&vectors, &norot_specs, &schedule_grids);

    println!("## Table C");
    println!();
    println!("| schedule | ρ measured | ρ product-model | σ ratio | encode ns/v |");
    println!("|---|---:|---:|---:|---:|");
    let rotated = &layered[&4];
    let rotated_product = monolithic[&1].rho * monolithic[&4].rho;
    println!(
        "| [1,4] | {:.6} | {:.6} | {:.6} | {:.3} |",
        rotated.rho,
        rotated_product,
        rotated.sigma_ratio.expect("layered sigma ratio"),
        rotated_encode_ns
    );
    for bits in [2, 4] {
        let result = &norot[&bits];
        let product = monolithic[&1].rho * monolithic[&bits].rho;
        let encode = if bits == 4 {
            format!("{norot_encode_ns:.3}")
        } else {
            "—".to_string()
        };
        println!(
            "| [1,{bits}]-norot | {:.6} | {:.6} | {:.6} | {encode} |",
            result.rho,
            product,
            result.sigma_ratio.expect("norot sigma ratio")
        );
    }
    println!();
    norot
}

fn table_d(
    grids: &BTreeMap<u8, Grid>,
    monolithic: &BTreeMap<u8, QuantResult>,
    layered: &BTreeMap<u8, QuantResult>,
    norot: &BTreeMap<u8, QuantResult>,
) {
    let schedules = vec![
        Schedule {
            label: "[1,1]",
            specs: vec![spec(1, 11, true), spec(1, 22, true)],
            grids: vec![grids[&1].clone(), grids[&1].clone()],
            rho: layered[&1].rho,
        },
        Schedule {
            label: "[1,2]",
            specs: vec![spec(1, 11, true), spec(2, 22, true)],
            grids: vec![grids[&1].clone(), grids[&2].clone()],
            rho: layered[&2].rho,
        },
        Schedule {
            label: "[1,4]",
            specs: vec![spec(1, 11, true), spec(4, 22, true)],
            grids: vec![grids[&1].clone(), grids[&4].clone()],
            rho: layered[&4].rho,
        },
        Schedule {
            label: "[1,4]-norot",
            specs: vec![spec(1, 11, true), spec(4, 22, false)],
            grids: vec![grids[&1].clone(), grids[&4].clone()],
            rho: norot[&4].rho,
        },
        Schedule {
            label: "[4]",
            specs: vec![spec(4, 11, true)],
            grids: vec![grids[&4].clone()],
            rho: monolithic[&4].rho,
        },
    ];

    println!("## Table D");
    println!();
    println!("| schedule | r | kappa | s1 | s2 | candidate recall | end-to-end ns/candidate |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for schedule in schedules {
        for radius in [0.3_f32, 0.8] {
            let data = build_cascade_data(&schedule, radius);
            for kappa in [2.0_f32, 3.0] {
                let row = run_cascade(&schedule, &data, radius, kappa, monolithic[&1].rho);
                let s1 = row
                    .s1
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "—".to_string());
                println!(
                    "| {} | {:.1} | {:.0} | {} | {:.6} | {:.3} | {:.3} |",
                    schedule.label, radius, kappa, s1, row.s2, row.recall, row.ns_per_candidate
                );
            }
        }
    }
}

fn measure_quantization(
    specs: &[LayerSpec],
    grids: &[Grid],
    product_model: Option<f64>,
) -> QuantResult {
    let mut rng = ChaCha8Rng::seed_from_u64(0x0053_5745_4550_0001);
    let queries: Vec<Vec<f32>> = (0..QUERY_BATCH)
        .map(|_| random_unit(&mut rng, D, 1.0))
        .collect();
    let prepared: Vec<_> = product_model
        .map(|_| {
            queries
                .iter()
                .map(|query| prepare_fp_query(query, specs))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut error_energy = 0.0_f64;
    let mut signal_energy = 0.0_f64;
    let mut estimates = Vec::with_capacity(QUERY_BATCH * N);
    let mut truths = Vec::with_capacity(QUERY_BATCH * N);

    for _ in 0..N {
        let vector = random_unit(&mut rng, D, 1.0);
        let encoded = encode_layers(&vector, None, specs, grids);
        let mut first_space = vector.clone();
        Rotation::new(D, specs[0].seed).apply(&mut first_space);
        let reconstructed = reconstruct_first_space(&encoded, specs, grids, D);
        for (&actual, estimated) in first_space.iter().zip(reconstructed) {
            error_energy += f64::from(actual - estimated).powi(2);
            signal_energy += f64::from(actual).powi(2);
        }
        if product_model.is_some() {
            for (query, prepared) in queries.iter().zip(&prepared) {
                estimates.push(estimate_prepared_fp(&encoded, prepared, specs, grids, D));
                truths.push(fp_dot(query, &vector));
            }
        }
    }
    let rho = (error_energy / signal_energy).sqrt();
    let sigma_ratio = product_model.map(|product| {
        empirical_sigma(&estimates, &truths) / sigma_from_rho(product, D, DEFAULT_CAL)
    });
    QuantResult { rho, sigma_ratio }
}

fn measure_plane_scores(grids: &BTreeMap<u8, Grid>) -> BTreeMap<u8, f64> {
    let data: Vec<f32> = (0..D).map(|i| (i as f32 * 0.013).sin()).collect();
    let query: Vec<f32> = (0..D).map(|i| (i as f32 * 0.017).cos()).collect();
    let mut result = BTreeMap::new();

    let mut words = vec![0_u64; D / 64];
    encode_sign(&data, &mut words);
    let prepared = prepare_query(&query, 4);
    result.insert(
        1,
        bench_ns(|| score_asym(black_box(&words), black_box(&prepared))),
    );

    for bits in 2..=4 {
        let grid = &grids[&bits];
        let mut codes = vec![0; packed_len(D, bits)];
        encode_grid(&data, &grid.points, bits, &mut codes);
        let lut = build_lut(&query, &grid.points, bits);
        result.insert(
            bits,
            bench_ns(|| score(black_box(&codes), black_box(&lut), D, bits)),
        );
    }
    result
}

fn bench_encode(vectors: &[Vec<f32>], specs: &[LayerSpec], grids: &[Grid]) -> f64 {
    let mut index = 0;
    bench_ns(|| {
        let encoded = encode_layers(&vectors[index], None, specs, grids);
        index = (index + 1) % vectors.len();
        encoded
    })
}

fn build_cascade_data(schedule: &Schedule, radius: f32) -> CascadeData {
    const CLUSTERS: usize = 500;
    const PER_CLUSTER: usize = 100;
    let mut rng = ChaCha8Rng::seed_from_u64(0x0044_4154_4100_0000);
    let mut clusters = Vec::with_capacity(CLUSTERS);
    let mut candidates = Vec::with_capacity(CLUSTERS * PER_CLUSTER);
    let mut truths = Vec::with_capacity(CLUSTERS * PER_CLUSTER);

    for cluster_index in 0..CLUSTERS {
        let query_residual = random_unit(&mut rng, D, 1.0);
        let mut layer_query = query_residual.clone();
        let mut sign = None;
        let mut first_lut = None;
        let mut second_sign = None;
        let mut second_lut = None;
        for (layer, (spec, grid)) in schedule.specs.iter().zip(&schedule.grids).enumerate() {
            if layer == 0 || spec.rotate {
                Rotation::new(D, spec.seed).apply(&mut layer_query);
            }
            if layer == 0 && spec.bits == 1 {
                sign = Some(prepare_query(&layer_query, 4));
            } else if layer == 0 {
                first_lut = Some(build_lut(&layer_query, &grid.points, spec.bits));
            } else if spec.bits == 1 {
                second_sign = Some(prepare_query(&layer_query, 4));
            } else {
                second_lut = Some(build_lut(&layer_query, &grid.points, spec.bits));
            }
        }
        let start = cluster_index * PER_CLUSTER;
        for _ in 0..PER_CLUSTER {
            let residual = random_unit(&mut rng, D, radius);
            truths.push(fp_dot(&query_residual, &residual));
            candidates.push(encode_layers(
                &residual,
                None,
                &schedule.specs,
                &schedule.grids,
            ));
        }
        clusters.push(ClusterQuery {
            start,
            end: start + PER_CLUSTER,
            sign,
            first_lut,
            second_sign,
            second_lut,
        });
    }
    CascadeData {
        clusters,
        candidates,
        truths,
    }
}

fn run_cascade(
    schedule: &Schedule,
    data: &CascadeData,
    radius: f32,
    kappa: f32,
    sign_rho: f64,
) -> CascadeRow {
    const K: usize = 10;
    let count = data.candidates.len();
    let mut exact_indices: Vec<usize> = (0..count).collect();
    exact_indices.sort_unstable_by(|&a, &b| data.truths[b].total_cmp(&data.truths[a]));
    let exact_top: HashSet<u32> = exact_indices[..K]
        .iter()
        .map(|&index| index as u32)
        .collect();
    let started = Instant::now();

    if schedule.specs.len() == 1 {
        let mut scores = vec![0.0_f32; count];
        for cluster in &data.clusters {
            let lut = cluster.first_lut.as_ref().expect("monolithic LUT");
            for (index, score_out) in scores
                .iter_mut()
                .enumerate()
                .take(cluster.end)
                .skip(cluster.start)
            {
                *score_out = estimate_grid(
                    &data.candidates[index].codes[0],
                    data.candidates[index].scales[0],
                    lut,
                    D,
                    schedule.specs[0].bits,
                );
            }
        }
        let sigma = (schedule.rho * f64::from(radius) / (D as f64).sqrt()) as f32;
        let sigmas = vec![sigma; count];
        let (index, value) = kth(&scores, K);
        let survivors = band_filter(&scores, &sigmas, kappa, value - kappa * sigmas[index]);
        let survivor_set: HashSet<u32> = survivors.iter().copied().collect();
        return CascadeRow {
            s1: None,
            s2: survivors.len() as f64 / count as f64,
            recall: exact_top.intersection(&survivor_set).count() as f64 / K as f64,
            ns_per_candidate: started.elapsed().as_nanos() as f64 / count as f64,
        };
    }

    let mut first_scores = vec![0.0_f32; count];
    for cluster in &data.clusters {
        let query = cluster.sign.as_ref().expect("sign query");
        for (index, score_out) in first_scores
            .iter_mut()
            .enumerate()
            .take(cluster.end)
            .skip(cluster.start)
        {
            let words = bytes_to_words(&data.candidates[index].codes[0]);
            *score_out = estimate_asym(&words, data.candidates[index].scales[0], query);
        }
    }
    let sigma1 = (sign_rho * f64::from(radius) / (D as f64).sqrt()) as f32;
    let first_sigmas = vec![sigma1; count];
    let (first_index, first_value) = kth(&first_scores, K);
    let first_survivors = band_filter(
        &first_scores,
        &first_sigmas,
        kappa,
        first_value - kappa * first_sigmas[first_index],
    );

    let mut second_scores = Vec::with_capacity(first_survivors.len());
    for &global in &first_survivors {
        let index = global as usize;
        let cluster = &data.clusters[index / 100];
        let second = if schedule.specs[1].bits == 1 {
            let words = bytes_to_words(&data.candidates[index].codes[1]);
            estimate_asym(
                &words,
                data.candidates[index].scales[1],
                cluster
                    .second_sign
                    .as_ref()
                    .expect("second-layer sign query"),
            )
        } else {
            estimate_grid(
                &data.candidates[index].codes[1],
                data.candidates[index].scales[1],
                cluster.second_lut.as_ref().expect("second-layer LUT"),
                D,
                schedule.specs[1].bits,
            )
        };
        second_scores.push(first_scores[index] + second);
    }
    let sigma2 = (schedule.rho * f64::from(radius) / (D as f64).sqrt()) as f32;
    let second_sigmas = vec![sigma2; second_scores.len()];
    let (second_index, second_value) = kth(&second_scores, K);
    let local = band_filter(
        &second_scores,
        &second_sigmas,
        kappa,
        second_value - kappa * second_sigmas[second_index],
    );
    let final_set: HashSet<u32> = local
        .iter()
        .map(|&local_index| first_survivors[local_index as usize])
        .collect();
    CascadeRow {
        s1: Some(first_survivors.len() as f64 / count as f64),
        s2: final_set.len() as f64 / count as f64,
        recall: exact_top.intersection(&final_set).count() as f64 / K as f64,
        ns_per_candidate: started.elapsed().as_nanos() as f64 / count as f64,
    }
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

fn fixed_vectors(count: usize, seed: u64, norm: f32) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count).map(|_| random_unit(&mut rng, D, norm)).collect()
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

fn fp_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
