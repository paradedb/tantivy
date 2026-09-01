use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grid_plane::{build_lut, encode, packed_len, score};
use quant_model::build_grid;

fn bench_kernels(c: &mut Criterion) {
    let d = 768;
    let vector: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013).sin()).collect();
    let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.017).cos()).collect();
    let mut group = c.benchmark_group("grid_d768");
    for bits in [2, 3, 4] {
        let grid = build_grid(d, bits);
        let mut codes = vec![0; packed_len(d, bits)];
        encode(&vector, &grid.points, bits, &mut codes);
        let lut = build_lut(&query, &grid.points, bits);
        group.bench_with_input(BenchmarkId::new("encode", bits), &bits, |b, _| {
            b.iter(|| {
                encode(
                    black_box(&vector),
                    black_box(&grid.points),
                    bits,
                    black_box(&mut codes),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("score", bits), &bits, |b, _| {
            b.iter(|| score(black_box(&codes), black_box(&lut), d, bits))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
