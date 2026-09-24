use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fht::Rotation;

fn bench_rotation(c: &mut Criterion) {
    let mut group = c.benchmark_group("fht_apply");
    for d in [128, 768, 1536] {
        let rotation = Rotation::new(d, 42);
        let input = vec![0.25; d];
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            b.iter_batched(
                || input.clone(),
                |mut vector| rotation.apply(black_box(&mut vector)),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rotation);
criterion_main!(benches);
