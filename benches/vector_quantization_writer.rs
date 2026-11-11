use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use quant::cascade::{
    encode_batch_in_place, encode_batch_in_place_with_workspace, prepare_centroid,
    BatchEncodeWorkspace, LayerSpec,
};
use quant::model::build_grid;

const DIM: usize = 768;
const ROWS: usize = 100;

const CLUSTER_DIM: usize = 1024;
const CLUSTER_ROWS: usize = 200;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Standard normal samples from two uniforms in (0, 1] (Box-Muller).
fn gaussian_pair(state: &mut u64) -> (f32, f32) {
    let unit = |state: &mut u64| ((lcg_next(state) >> 40) as f64 + 1.0) / (1_u64 << 24) as f64;
    let radius = (-2.0 * unit(state).ln()).sqrt();
    let angle = std::f64::consts::TAU * unit(state);
    ((radius * angle.cos()) as f32, (radius * angle.sin()) as f32)
}

fn gaussian_vec(state: &mut u64, len: usize, sigma: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(len + 1);
    while values.len() < len {
        let (first, second) = gaussian_pair(state);
        values.push(first * sigma);
        values.push(second * sigma);
    }
    values.truncate(len);
    values
}

/// Encodes one cluster tile per two-layer schedule, both layers rotated: Sign then Sign (the
/// default `[1,1]`) and Sign then Grid 4 bits (`[1,4]`). The centroid and workspace are prepared
/// outside the timed region, as the IVF writer prepares them once per cluster.
fn cluster_tile_encode(c: &mut Criterion) {
    let layer = |kind, bits, seed| LayerSpec {
        kind,
        bits,
        rotation: quant::cascade::Rotation::SeededFhtChaCha8 { seed },
    };
    let sign = quant::cascade::LayerKind::Sign;
    let grid = quant::cascade::LayerKind::Grid;
    let schedules = [
        (
            "1_plus_1_d1024_cluster200",
            [layer(sign, 1, 0x1111), layer(sign, 1, 0x2222)],
        ),
        (
            "1_plus_4_d1024_cluster200",
            [layer(sign, 1, 0x1111), layer(grid, 4, 0x2222)],
        ),
    ];
    let mut state = 0x5eed_c1a5_7e40_0001_u64;
    let coordinate_sigma = 1.0 / (CLUSTER_DIM as f32).sqrt();
    let centroid = gaussian_vec(&mut state, CLUSTER_DIM, coordinate_sigma);
    let tile: Vec<f32> = (0..CLUSTER_ROWS)
        .flat_map(|_| {
            gaussian_vec(&mut state, CLUSTER_DIM, 0.5 * coordinate_sigma)
                .into_iter()
                .zip(&centroid)
                .map(|(offset, &center)| center + offset)
                .collect::<Vec<_>>()
        })
        .collect();

    let mut group = c.benchmark_group("vector_quantization_writer");
    group.throughput(Throughput::Elements(CLUSTER_ROWS as u64));
    for (name, specs) in schedules {
        let grids = specs.map(|spec| build_grid(CLUSTER_DIM, spec.bits));
        let prepared = prepare_centroid(&centroid, &specs);
        let mut workspace = BatchEncodeWorkspace::with_capacity(CLUSTER_DIM, CLUSTER_ROWS, &specs);
        for (variant, compute_constants) in [("constants_off", false), ("constants_on", true)] {
            group.bench_function(BenchmarkId::new(name, variant), |b| {
                b.iter_batched(
                    || tile.clone(),
                    |mut tile| {
                        black_box(encode_batch_in_place_with_workspace(
                            &mut tile,
                            CLUSTER_ROWS,
                            &prepared,
                            &specs,
                            &grids,
                            &mut workspace,
                            compute_constants,
                        ));
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

fn writer_context_encode(c: &mut Criterion) {
    let specs = [
        LayerSpec {
            kind: quant::cascade::LayerKind::Sign,
            bits: 1,
            rotation: quant::cascade::Rotation::SeededFhtChaCha8 { seed: 0x1111 },
        },
        LayerSpec {
            kind: quant::cascade::LayerKind::Grid,
            bits: 4,
            rotation: quant::cascade::Rotation::SeededFhtChaCha8 { seed: 0x2222 },
        },
    ];
    let grids = [build_grid(DIM, 1), build_grid(DIM, 4)];
    let centroid: Vec<f32> = (0..DIM)
        .map(|coordinate| (coordinate as f32 * 0.013).sin() * 0.1)
        .collect();
    let vectors: Vec<f32> = (0..ROWS)
        .flat_map(|row| {
            centroid
                .iter()
                .enumerate()
                .map(move |(coordinate, &center)| {
                    center + ((row * DIM + coordinate) as f32 * 0.017).cos() * 0.2
                })
        })
        .collect();

    let mut group = c.benchmark_group("vector_quantization_writer");
    group.throughput(Throughput::Elements(ROWS as u64));
    group.bench_function("1_plus_4_d768_cluster100", |b| {
        b.iter_batched(
            || vectors.clone(),
            |mut tile| {
                let prepared = prepare_centroid(black_box(&centroid), black_box(&specs));
                black_box(encode_batch_in_place(
                    &mut tile, ROWS, &prepared, &specs, &grids,
                ))
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, writer_context_encode, cluster_tile_encode);
criterion_main!(benches);
