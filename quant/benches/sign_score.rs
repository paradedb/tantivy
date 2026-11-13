use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tantivy_quant::sign::fastscan::FastScanQuery;
use tantivy_quant::sign::{estimate_asym_batch_unscaled, prepare_query, score_asym, score_sym};

const QUERY_BITS: u8 = 4;
const SCAN_DIMENSIONS: [usize; 3] = [128, 768, 1024];
const SCAN_ROWS: [usize; 3] = [32, 100, 1024];

fn bench_dense_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("sign_dense_scan");
    for d in SCAN_DIMENSIONS {
        let stride = d.div_ceil(u64::BITS as usize);
        let query = prepare_query(
            &(0..d).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>(),
            QUERY_BITS,
        );
        let fastscan = FastScanQuery::new(&query);
        for count in SCAN_ROWS {
            let rows: Vec<u64> = (0..count * stride)
                .map(|i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                .collect();
            let mut out = vec![0.0; count];
            group.throughput(Throughput::Elements(count as u64));
            let shape = format!("d{d}_rows{count}");
            group.bench_function(BenchmarkId::new("popcount", &shape), |b| {
                b.iter(|| {
                    estimate_asym_batch_unscaled(
                        black_box(&rows),
                        stride,
                        black_box(&query),
                        black_box(&mut out),
                    );
                    black_box(&out);
                })
            });
            if let Some(fastscan) = &fastscan {
                group.bench_function(BenchmarkId::new("fastscan_transpose", &shape), |b| {
                    b.iter(|| {
                        fastscan.score_batch(black_box(&rows), stride, black_box(&mut out));
                        black_box(&out);
                    })
                });
            }
        }
    }
    group.finish();
    for d in SCAN_DIMENSIONS {
        let query = prepare_query(
            &(0..d).map(|i| (i as f32 * 0.019).sin()).collect::<Vec<_>>(),
            QUERY_BITS,
        );
        c.bench_function(&format!("sign_fastscan_prepare_d{d}"), |b| {
            b.iter(|| FastScanQuery::new(black_box(&query)))
        });
    }
}

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

criterion_group!(benches, bench_scores, bench_dense_scan);
criterion_main!(benches);
