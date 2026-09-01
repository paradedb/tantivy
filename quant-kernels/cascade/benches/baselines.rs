use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quant_model::f16::f16_to_f32;
use sign_plane::{encode, unpack};

fn fp_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn bench_baselines(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp_dot");
    for d in [128, 768, 1536] {
        let data: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013).sin()).collect();
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.017).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            b.iter(|| fp_dot(black_box(&data), black_box(&query)))
        });
    }
    group.finish();

    let d = 768;
    let data: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013).sin()).collect();
    let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.017).cos()).collect();
    let mut bits = vec![0_u64; d / 64];
    let scale = f16_to_f32(encode(&data, &mut bits));
    c.bench_function("decode_then_dot_d768", |b| {
        b.iter(|| {
            let decoded: Vec<f32> = unpack(black_box(&bits), d)
                .into_iter()
                .map(|sign| sign * scale)
                .collect();
            fp_dot(black_box(&decoded), black_box(&query))
        })
    });
}

criterion_group!(benches, bench_baselines);
criterion_main!(benches);
