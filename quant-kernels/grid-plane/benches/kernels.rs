use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use grid_plane::{
    build_lut, build_packed_lut_4, encode, packed_len, score, score_batch, score_batch_packed_4,
    score_batch_packed_4_indexed,
};
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
        let rows = 32;
        let batch_codes = codes.repeat(rows);
        let mut batch_out = vec![0.0; rows];
        group.bench_with_input(BenchmarkId::new("score_batch_32", bits), &bits, |b, _| {
            b.iter(|| {
                score_batch(
                    black_box(&batch_codes),
                    codes.len(),
                    black_box(&lut),
                    d,
                    bits,
                    black_box(&mut batch_out),
                )
            })
        });
        if bits == 4 {
            let packed_lut = build_packed_lut_4(&lut, d);
            group.bench_with_input(
                BenchmarkId::new("score_batch_packed_32", bits),
                &bits,
                |b, _| {
                    b.iter(|| {
                        score_batch_packed_4(
                            black_box(&batch_codes),
                            codes.len(),
                            black_box(&packed_lut),
                            d,
                            black_box(&mut batch_out),
                        )
                    })
                },
            );
            let sparse_codes = codes.repeat(rows * 3);
            let row_offsets: Vec<usize> = (0..rows).map(|row| row * 3).collect();
            group.bench_with_input(
                BenchmarkId::new("score_batch_packed_indexed_32", bits),
                &bits,
                |b, _| {
                    b.iter(|| {
                        score_batch_packed_4_indexed(
                            black_box(&sparse_codes),
                            codes.len(),
                            black_box(&row_offsets),
                            black_box(&packed_lut),
                            d,
                            black_box(&mut batch_out),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
