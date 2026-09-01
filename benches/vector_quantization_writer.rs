use std::hint::black_box;

use cascade::{encode_batch_in_place, prepare_centroid, LayerSpec};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use quant_model::build_grid;

const DIM: usize = 768;
const ROWS: usize = 100;

fn writer_context_encode(c: &mut Criterion) {
    let specs = [
        LayerSpec {
            bits: 1,
            seed: 0x1111,
            rotate: true,
        },
        LayerSpec {
            bits: 4,
            seed: 0x2222,
            rotate: true,
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

criterion_group!(benches, writer_context_encode);
criterion_main!(benches);
