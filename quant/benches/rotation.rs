use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tantivy_quant::rotation::{HadamardKernel, Rotation};

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

/// Times each Hadamard kernel this CPU has over the block decomposition of `d`, whichever kernel
/// the dispatcher would pick.
fn bench_hadamard_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("hadamard");
    for d in [768, 1024, 1536] {
        let input: Vec<f32> = (0..d).map(|i| ((i as f32 + 1.0) * 0.731).sin()).collect();
        for kernel in HadamardKernel::ALL
            .into_iter()
            .filter(|kernel| kernel.is_available())
        {
            group.bench_with_input(BenchmarkId::new(kernel.name(), d), &d, |b, _| {
                b.iter_batched(
                    || input.clone(),
                    |mut vector| kernel.block_hadamard(black_box(&mut vector)),
                    criterion::BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_rotation, bench_hadamard_kernels);
criterion_main!(benches);
