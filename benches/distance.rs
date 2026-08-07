// Benchmarks the SIMD distance kernels in src/vector/distance.rs.
//
// What's measured:
// - Pairwise f32:    l2_squared / dot / cosine on two f32 slices
// - Pairwise bytes:  query is f32, doc side is LE bytes (segment shape)
//
// Dimension sweep covers common embedding widths:
//   128 (word2vec) / 384 (MiniLM) / 768 (BERT-base) / 1024 (e5-large)
//   1536 (OpenAI ada-002) / 3072 (text-embedding-3-large)
//
// Throughput is reported in bytes/sec so criterion prints MB/s, which
// is directly comparable against:
//   - L1 bandwidth   (~1 TB/s on modern cores)
//   - FMA peak       (AVX-512: 64 FLOPS/cycle/core; NEON Apple M-series: 16-32 FLOPS/cycle/core
//     depending on generation)
//
// Run with:  cargo bench --bench distance

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tantivy::vector::{
    cosine, cosine_bytes, cosine_sq4, dot, dot_bytes, dot_sq4, l2_squared, l2_squared_bytes,
    l2_squared_sq4, Sq4Params,
};

const DIMS: &[usize] = &[1024, 2048];

/// Guaranteed-scalar dot product. Serves as the "no SIMD at all"
/// baseline so we can see how much the autovectorizer is buying us.
///
/// `black_box` on the accumulator each iteration is the trick: it
/// makes LLVM treat `acc` as opaque between iterations, which prevents
/// the loop vectorizer from recognizing the reduction pattern and
/// emitting SIMD.
#[inline(never)]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc = black_box(acc + a[i] * b[i]);
    }
    acc
}

/// Deterministic pseudo-random f32 values in [-1, 1). A bare LCG is
/// enough — we just need non-zero, non-equal data the optimizer can't
/// fold to a constant.
fn make_f32(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn f32_to_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bench_pairwise_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise_f32");
    for &dim in DIMS {
        let a = make_f32(dim, 1);
        let b = make_f32(dim, 2);

        // Sanity check: the scalar baseline must agree with the
        // autovectorized kernel within fp tolerance. Different reduction
        // orders mean we can't expect bit-exact equality.
        {
            let autovec = dot(&a, &b);
            let scalar = dot_scalar(&a, &b);
            let tol = 1e-3 * autovec.abs().max(1.0);
            assert!(
                (autovec - scalar).abs() < tol,
                "dot_scalar mismatch at dim={dim}: autovec={autovec} scalar={scalar}"
            );
        }

        // 2 vectors * 4 B/element.
        group.throughput(Throughput::Bytes((dim * 8) as u64));
        group.bench_with_input(BenchmarkId::new("dot_scalar", dim), &dim, |bn, _| {
            bn.iter(|| dot_scalar(black_box(&a), black_box(&b)))
        });
        group.bench_with_input(BenchmarkId::new("dot", dim), &dim, |bn, _| {
            bn.iter(|| dot(black_box(&a), black_box(&b)))
        });
        group.bench_with_input(BenchmarkId::new("l2_squared", dim), &dim, |bn, _| {
            bn.iter(|| l2_squared(black_box(&a), black_box(&b)))
        });
        group.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bn, _| {
            bn.iter(|| cosine(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

fn bench_pairwise_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise_bytes");
    for &dim in DIMS {
        let q = make_f32(dim, 1);
        let d = make_f32(dim, 2);
        let d_bytes = f32_to_le_bytes(&d);
        group.throughput(Throughput::Bytes((dim * 8) as u64));
        group.bench_with_input(BenchmarkId::new("l2_squared_bytes", dim), &dim, |bn, _| {
            bn.iter(|| l2_squared_bytes::<f32>(black_box(&q), black_box(&d_bytes)))
        });
        group.bench_with_input(BenchmarkId::new("dot_bytes", dim), &dim, |bn, _| {
            bn.iter(|| dot_bytes::<f32>(black_box(&q), black_box(&d_bytes)))
        });
        group.bench_with_input(BenchmarkId::new("cosine_bytes", dim), &dim, |bn, _| {
            bn.iter(|| cosine_bytes::<f32>(black_box(&q), black_box(&d_bytes)))
        });
    }
    group.finish();
}

/// f32 query × packed SQ4 row — the centroid-routing shape from V3 on.
/// Throughput counts the bytes actually touched (4 B/dim query + 0.5
/// B/dim codes), so MB/s is comparable against the byte kernels' cache
/// pressure, not their arithmetic.
///
/// The `*_sq4` cases are the one-shot reference kernels; the `adc_*`
/// cases hold a prepared [`Sq4Scorer`] across scores — the shape routing
/// actually runs, with the per-query prep amortized away.
fn bench_pairwise_sq4(c: &mut Criterion) {
    use tantivy::vector::{ArenaScorer, Sq4Arena, VectorArena};

    let mut group = c.benchmark_group("pairwise_sq4");
    for &dim in DIMS {
        let q = make_f32(dim, 1);
        let d = make_f32(dim, 2);
        let params = Sq4Params::fit(&d, dim);
        let mut codes = Vec::new();
        params.quantize_row_into(&d, &mut codes);
        let norms = tantivy::vector::sq4_row_norms(&codes, &params);
        let arena = Sq4Arena::new(&codes, &params, &norms);
        group.throughput(Throughput::Bytes((dim * 4 + dim / 2) as u64));
        group.bench_with_input(BenchmarkId::new("l2_squared_sq4", dim), &dim, |bn, _| {
            bn.iter(|| l2_squared_sq4(black_box(&q), black_box(&codes), black_box(&params)))
        });
        group.bench_with_input(BenchmarkId::new("dot_sq4", dim), &dim, |bn, _| {
            bn.iter(|| dot_sq4(black_box(&q), black_box(&codes), black_box(&params)))
        });
        group.bench_with_input(BenchmarkId::new("cosine_sq4", dim), &dim, |bn, _| {
            bn.iter(|| cosine_sq4(black_box(&q), black_box(&codes), black_box(&params)))
        });
        for metric in [
            tantivy::schema::Metric::L2,
            tantivy::schema::Metric::Dot,
            tantivy::schema::Metric::Cosine,
        ] {
            let scorer = arena.scorer(metric, dim, &q);
            let name = format!("adc_{metric:?}").to_lowercase();
            group.bench_with_input(BenchmarkId::new(name, dim), &dim, |bn, _| {
                bn.iter(|| black_box(&scorer).score(black_box(0)))
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(100);
    targets = bench_pairwise_f32, bench_pairwise_bytes, bench_pairwise_sq4
}
criterion_main!(benches);
