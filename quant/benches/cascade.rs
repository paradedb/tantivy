use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tantivy_quant::cascade::{band_filter, kth};

fn bench_boundaries(c: &mut Criterion) {
    let mut group = c.benchmark_group("boundary_ops");
    for count in [50_000, 500_000] {
        let scores: Vec<f32> = (0..count).map(|i| ((i as f32) * 0.001).sin()).collect();
        let sigmas = vec![0.05; count];
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                let (index, value) = kth(black_box(&scores), 10);
                band_filter(
                    black_box(&scores),
                    black_box(&sigmas),
                    3.0,
                    value - sigmas[index],
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_boundaries);
criterion_main!(benches);
