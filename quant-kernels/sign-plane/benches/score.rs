use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sign_plane::{prepare_query, score_asym, score_sym};

fn bench_scores(c: &mut Criterion) {
    let d = 768;
    let x = vec![0xaaaa_5555_ffff_0000_u64; d / 64];
    let q = vec![0x1234_5678_9abc_def0_u64; d / 64];
    let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.01).sin()).collect();
    let prepared = prepare_query(&query, 4);
    c.bench_function("sign_score_sym_d768", |b| {
        b.iter(|| score_sym(black_box(&x), black_box(&q)))
    });
    c.bench_function("sign_score_asym_bq4_d768", |b| {
        b.iter(|| score_asym(black_box(&x), black_box(&prepared)))
    });
}

criterion_group!(benches, bench_scores);
criterion_main!(benches);
