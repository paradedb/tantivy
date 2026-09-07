//! Integrated layer-0 scan-shape benchmark.
//!
//! Each iteration advances to the next 100-row posting and runs the production
//! sign batch kernel, sidecar decode, corrected-score FMA, and analytical
//! sigma pass.
//! Query preparation and scratch allocation happen outside the timed loop.

use std::hint::black_box;

use cascade::{audit_split_query_layer_error_squared, prepare_split_query, LayerSpec};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use quant_model::build_grid;
use quant_model::f16::f32_to_f16;
use tantivy::vector::{
    quantization_bench_layer0_cosine_cluster, quantization_bench_layer0_cosine_cluster_f16_scales,
};

const DIM: usize = 1_024;
const ROWS_PER_CLUSTER: usize = 100;
const CLUSTERS: usize = 256;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn layer0_integrated_shape(c: &mut Criterion) {
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
    let query_norm_squared = query.iter().map(|value| value * value).sum::<f32>();
    let query_error_squared =
        audit_split_query_layer_error_squared(&query, &[spec], &[build_grid(DIM, 1)], 4)[0] as f32;
    let code_stride = DIM.div_ceil(64) * std::mem::size_of::<u64>();
    let cluster_code_bytes = ROWS_PER_CLUSTER * code_stride;
    let cluster_scale_f32_bytes = ROWS_PER_CLUSTER * std::mem::size_of::<f32>();
    let cluster_scale_f16_bytes = ROWS_PER_CLUSTER * std::mem::size_of::<u16>();
    let cluster_gamma_bytes = ROWS_PER_CLUSTER * std::mem::size_of::<u16>();

    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut codes = vec![0_u8; CLUSTERS * cluster_code_bytes];
    for chunk in codes.chunks_exact_mut(8) {
        chunk.copy_from_slice(&lcg_next(&mut state).to_le_bytes());
    }
    let mut scales_f32 = vec![0_u8; CLUSTERS * cluster_scale_f32_bytes];
    let mut scales_f16 = vec![0_u8; CLUSTERS * cluster_scale_f16_bytes];
    for (f32_chunk, f16_chunk) in scales_f32
        .chunks_exact_mut(4)
        .zip(scales_f16.chunks_exact_mut(2))
    {
        let value = 0.01 + ((lcg_next(&mut state) >> 40) as f32 / (1_u32 << 24) as f32) * 0.2;
        f32_chunk.copy_from_slice(&value.to_le_bytes());
        f16_chunk.copy_from_slice(&f32_to_f16(value).to_le_bytes());
    }
    let mut gammas = vec![0_u8; CLUSTERS * cluster_gamma_bytes];
    for chunk in gammas.chunks_exact_mut(2) {
        let value = 1.45 + ((lcg_next(&mut state) >> 40) as f32 / (1_u32 << 24) as f32) * 0.25;
        chunk.copy_from_slice(&f32_to_f16(value).to_le_bytes());
    }
    let mut error_ratios = vec![0_u8; CLUSTERS * cluster_gamma_bytes];
    for chunk in error_ratios.chunks_exact_mut(2) {
        let value = 0.25 + ((lcg_next(&mut state) >> 40) as f32 / (1_u32 << 24) as f32) * 0.5;
        chunk.copy_from_slice(&f32_to_f16(value).to_le_bytes());
    }
    let mut residual_norms = vec![0_u8; CLUSTERS * cluster_scale_f32_bytes];
    for chunk in residual_norms.chunks_exact_mut(4) {
        let value = 0.5 + ((lcg_next(&mut state) >> 40) as f32 / (1_u32 << 24) as f32) * 1.5;
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    let cluster_scores: Vec<f32> = (0..CLUSTERS)
        .map(|cluster| (cluster as f32 * 0.071).cos())
        .collect();

    let mut kernel_scores = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut decoded_scales = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut decoded_gammas = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut decoded_error_ratios = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut decoded_residual_norms = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut bases = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut estimates = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut sigmas = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut residual_norms_squared = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut sign_query_error_terms = Vec::with_capacity(ROWS_PER_CLUSTER);
    let mut next_cluster = 0_usize;

    // Allocate and size every scratch stream before Criterion starts timing.
    let _ = quantization_bench_layer0_cosine_cluster(
        DIM,
        &prepared,
        spec,
        &codes[..cluster_code_bytes],
        code_stride,
        &scales_f32[..cluster_scale_f32_bytes],
        &gammas[..cluster_gamma_bytes],
        &error_ratios[..cluster_gamma_bytes],
        &residual_norms[..cluster_scale_f32_bytes],
        cluster_scores[0],
        query_norm_squared,
        query_error_squared,
        &mut kernel_scores,
        &mut decoded_scales,
        &mut decoded_gammas,
        &mut decoded_error_ratios,
        &mut decoded_residual_norms,
        &mut bases,
        &mut estimates,
        &mut sigmas,
        &mut residual_norms_squared,
        &mut sign_query_error_terms,
    );
    let _ = quantization_bench_layer0_cosine_cluster_f16_scales(
        DIM,
        &prepared,
        spec,
        &codes[..cluster_code_bytes],
        code_stride,
        &scales_f16[..cluster_scale_f16_bytes],
        &gammas[..cluster_gamma_bytes],
        &error_ratios[..cluster_gamma_bytes],
        &residual_norms[..cluster_scale_f32_bytes],
        cluster_scores[0],
        query_norm_squared,
        query_error_squared,
        &mut kernel_scores,
        &mut decoded_scales,
        &mut decoded_gammas,
        &mut decoded_error_ratios,
        &mut decoded_residual_norms,
        &mut bases,
        &mut estimates,
        &mut sigmas,
        &mut residual_norms_squared,
        &mut sign_query_error_terms,
    );

    let mut group = c.benchmark_group("vector_quantization_scan");
    group.throughput(Throughput::Elements(ROWS_PER_CLUSTER as u64));
    group.bench_function("layer0_cosine_d1024_cluster100", |b| {
        b.iter(|| {
            let cluster = next_cluster;
            next_cluster = (next_cluster + 1) % CLUSTERS;
            let code_start = cluster * cluster_code_bytes;
            let scale_start = cluster * cluster_scale_f32_bytes;
            let gamma_start = cluster * cluster_gamma_bytes;
            let residual_norm_start = cluster * cluster_scale_f32_bytes;
            black_box(quantization_bench_layer0_cosine_cluster(
                DIM,
                black_box(&prepared),
                spec,
                black_box(&codes[code_start..code_start + cluster_code_bytes]),
                code_stride,
                black_box(&scales_f32[scale_start..scale_start + cluster_scale_f32_bytes]),
                black_box(&gammas[gamma_start..gamma_start + cluster_gamma_bytes]),
                black_box(&error_ratios[gamma_start..gamma_start + cluster_gamma_bytes]),
                black_box(
                    &residual_norms
                        [residual_norm_start..residual_norm_start + cluster_scale_f32_bytes],
                ),
                cluster_scores[cluster],
                query_norm_squared,
                query_error_squared,
                &mut kernel_scores,
                &mut decoded_scales,
                &mut decoded_gammas,
                &mut decoded_error_ratios,
                &mut decoded_residual_norms,
                &mut bases,
                &mut estimates,
                &mut sigmas,
                &mut residual_norms_squared,
                &mut sign_query_error_terms,
            ))
        })
    });
    let mut next_cluster_f16 = 0_usize;
    group.bench_function("layer0_cosine_d1024_cluster100_scale_f16", |b| {
        b.iter(|| {
            let cluster = next_cluster_f16;
            next_cluster_f16 = (next_cluster_f16 + 1) % CLUSTERS;
            let code_start = cluster * cluster_code_bytes;
            let scale_start = cluster * cluster_scale_f16_bytes;
            let gamma_start = cluster * cluster_gamma_bytes;
            let residual_norm_start = cluster * cluster_scale_f32_bytes;
            black_box(quantization_bench_layer0_cosine_cluster_f16_scales(
                DIM,
                black_box(&prepared),
                spec,
                black_box(&codes[code_start..code_start + cluster_code_bytes]),
                code_stride,
                black_box(&scales_f16[scale_start..scale_start + cluster_scale_f16_bytes]),
                black_box(&gammas[gamma_start..gamma_start + cluster_gamma_bytes]),
                black_box(&error_ratios[gamma_start..gamma_start + cluster_gamma_bytes]),
                black_box(
                    &residual_norms
                        [residual_norm_start..residual_norm_start + cluster_scale_f32_bytes],
                ),
                cluster_scores[cluster],
                query_norm_squared,
                query_error_squared,
                &mut kernel_scores,
                &mut decoded_scales,
                &mut decoded_gammas,
                &mut decoded_error_ratios,
                &mut decoded_residual_norms,
                &mut bases,
                &mut estimates,
                &mut sigmas,
                &mut residual_norms_squared,
                &mut sign_query_error_terms,
            ))
        })
    });
    group.finish();
}

criterion_group!(benches, layer0_integrated_shape);
criterion_main!(benches);
