use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tantivy_quant::cascade::{prepare_split_query, LayerKind, LayerSpec, Rotation};
use tantivy_quant::model::build_grid;
use tantivy_quant::sign::{prepare_query, score_asym, score_sym};

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

/// One cluster's sign layer scored from bytes that start on and off a `u64` boundary, as a
/// storage read hands them over.
fn bench_batch_alignment(c: &mut Criterion) {
    let d = 768;
    let rows = 800;
    let stride = d / 8;
    let spec = LayerSpec {
        kind: LayerKind::Sign,
        bits: 1,
        rotation: Rotation::SeededFhtChaCha8 { seed: 11 },
    };
    let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.01).sin()).collect();
    let prepared = prepare_split_query(&query, &[spec], &[build_grid(d, 1)], 4);
    let codes: Vec<u8> = (0..rows * stride).map(|i| (i * 37 % 251) as u8).collect();
    let row_offsets: Vec<usize> = (0..rows).step_by(2).collect();
    let mut storage = vec![0_u8; codes.len() + 8];
    let mut dense = vec![0.0_f32; rows];
    let mut sparse = vec![0.0_f32; row_offsets.len()];
    for (name, aligned) in [("aligned", true), ("unaligned", false)] {
        let offset = (0..8)
            .find(|offset| ((storage.as_ptr() as usize + offset) % 8 == 0) == aligned)
            .unwrap();
        storage[offset..offset + codes.len()].copy_from_slice(&codes);
        let placed = &storage[offset..offset + codes.len()];
        c.bench_function(&format!("sign_batch_bq4_d768_rows800_{name}"), |b| {
            b.iter(|| {
                prepared.score_layer_batch_unscaled(0, black_box(placed), stride, spec, &mut dense)
            })
        });
        c.bench_function(
            &format!("sign_batch_indexed_bq4_d768_rows400_{name}"),
            |b| {
                b.iter(|| {
                    prepared.score_layer_batch_unscaled_indexed(
                        0,
                        black_box(placed),
                        stride,
                        black_box(&row_offsets),
                        spec,
                        &mut sparse,
                    )
                })
            },
        );
    }
}

criterion_group!(benches, bench_scores, bench_batch_alignment);
criterion_main!(benches);
