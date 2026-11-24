use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tantivy_quant::cascade::{
    band_filter, kth, prepare_split_query_with_plan, LayerKind, LayerSpec, QueryRotationPlan,
    Rotation,
};
use tantivy_quant::model::build_grid;

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

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Standard normal samples from two uniforms in (0, 1] (Box-Muller).
fn gaussian_pair(state: &mut u64) -> (f32, f32) {
    let unit = |state: &mut u64| ((lcg_next(state) >> 40) as f64 + 1.0) / (1_u64 << 24) as f64;
    let radius = (-2.0 * unit(state).ln()).sqrt();
    let angle = std::f64::consts::TAU * unit(state);
    ((radius * angle.cos()) as f32, (radius * angle.sin()) as f32)
}

/// Per-query preparation with four query bits for each two-layer schedule: both rotations, the
/// sign bitplanes and, for Grid 4 bits, the packed four-bit lookup table.
fn bench_query_prep(c: &mut Criterion) {
    const D: usize = 1024;
    const QUERIES: usize = 16;
    let layer = |kind, bits, seed| LayerSpec {
        kind,
        bits,
        rotation: Rotation::SeededFhtChaCha8 { seed },
    };
    let schedules = [
        (
            "sign4_sign4_d1024",
            [
                layer(LayerKind::Sign, 1, 0x1111),
                layer(LayerKind::Sign, 1, 0x2222),
            ],
        ),
        (
            "sign4_grid4_d1024",
            [
                layer(LayerKind::Sign, 1, 0x1111),
                layer(LayerKind::Grid, 4, 0x2222),
            ],
        ),
    ];
    let mut state = 0x5eed_9e7a_0000_0001_u64;
    let sigma = 1.0 / (D as f32).sqrt();
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|_| {
            let mut query = Vec::with_capacity(D);
            while query.len() < D {
                let (first, second) = gaussian_pair(&mut state);
                query.push(first * sigma);
                query.push(second * sigma);
            }
            query
        })
        .collect();

    let mut group = c.benchmark_group("query_prep");
    for (name, specs) in schedules {
        let grids = specs.map(|spec| build_grid(D, spec.bits));
        let plan = QueryRotationPlan::new(D, &specs);
        let mut next = 0;
        group.bench_function(name, |b| {
            b.iter(|| {
                let query = &queries[next % QUERIES];
                next += 1;
                black_box(prepare_split_query_with_plan(
                    black_box(query),
                    &plan,
                    &grids,
                    4,
                ))
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_boundaries, bench_query_prep);
criterion_main!(benches);
