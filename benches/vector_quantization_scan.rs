//! Integrated plane-1 scan-shape benchmark.
//!
//! Each iteration advances to the next 100-row posting and runs the production
//! sign batch kernel, LE-f16 scale decode, score/bias FMA pass, and sigma pass.
//! Query preparation and scratch allocation happen outside the timed loop.

use std::hint::black_box;

use cascade::{prepare_split_query, LayerSpec};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use quant_model::build_grid;
use quant_model::f16::f32_to_f16;
use tantivy::vector::quantization_bench_plane1_cosine_cluster;

const DIM: usize = 1_024;
const ROWS_PER_CLUSTER: usize = 100;
const CLUSTERS: usize = 256;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn plane1_integrated_shape(c: &mut Criterion) {
    let spec = LayerSpec {
        bits: 1,
        seed: 0x51_91_14,
        rotate: true,
    };
    let grid = build_grid(DIM, 1);
    let query: Vec<f32> = (0..DIM)
        .map(|coordinate| (coordinate as f32 * 0.013).sin())
        .collect();
    let prepared = prepare_split_query(&query, &[spec], &[grid], 4);
    let code_stride = DIM.div_ceil(64) * std::mem::size_of::<u64>();
    let cluster_code_bytes = ROWS_PER_CLUSTER * code_stride;
    let cluster_scale_bytes = ROWS_PER_CLUSTER * std::mem::size_of::<u16>();

    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut codes = vec![0_u8; CLUSTERS * cluster_code_bytes];
    for chunk in codes.chunks_exact_mut(8) {
        chunk.copy_from_slice(&lcg_next(&mut state).to_le_bytes());
    }
    let mut scales = vec![0_u8; CLUSTERS * cluster_scale_bytes];
    for chunk in scales.chunks_exact_mut(2) {
        let value = 0.01 + ((lcg_next(&mut state) >> 40) as f32 / (1_u32 << 24) as f32) * 0.2;
        chunk.copy_from_slice(&f32_to_f16(value).to_le_bytes());
    }
    let cluster_scores: Vec<f32> = (0..CLUSTERS)
        .map(|cluster| (cluster as f32 * 0.071).cos())
        .collect();

    let mut kernel_scores = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut decoded_scales = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut bias_scores = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut sigma_scores = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut next_cluster = 0_usize;

    // Allocate and size every scratch stream before Criterion starts timing.
    let _ = quantization_bench_plane1_cosine_cluster(
        &prepared,
        spec,
        &codes[..cluster_code_bytes],
        code_stride,
        &scales[..cluster_scale_bytes],
        cluster_scores[0],
        -0.04,
        0.12,
        &mut kernel_scores,
        &mut decoded_scales,
        &mut bias_scores,
        &mut sigma_scores,
    );

    let mut group = c.benchmark_group("vector_quantization_scan");
    group.throughput(Throughput::Elements(ROWS_PER_CLUSTER as u64));
    group.bench_function("plane1_cosine_d1024_cluster100", |b| {
        b.iter(|| {
            let cluster = next_cluster;
            next_cluster = (next_cluster + 1) % CLUSTERS;
            let code_start = cluster * cluster_code_bytes;
            let scale_start = cluster * cluster_scale_bytes;
            black_box(quantization_bench_plane1_cosine_cluster(
                black_box(&prepared),
                spec,
                black_box(&codes[code_start..code_start + cluster_code_bytes]),
                code_stride,
                black_box(&scales[scale_start..scale_start + cluster_scale_bytes]),
                cluster_scores[cluster],
                -0.04,
                0.12,
                &mut kernel_scores,
                &mut decoded_scales,
                &mut bias_scores,
                &mut sigma_scores,
            ))
        })
    });
    group.finish();
}

criterion_group!(benches, plane1_integrated_shape);
criterion_main!(benches);
