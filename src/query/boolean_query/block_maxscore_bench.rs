fn benchmark_pruning(scorers: Vec<TermScorer>, k: usize, mode: usize) -> Vec<Score> {
    let mut heap: BinaryHeap<Float> = BinaryHeap::with_capacity(k + 1);
    let mut callback = |_: DocId, score: Score| {
        heap.push(Float(score));
        if heap.len() > k {
            heap.pop();
        }
        if heap.len() == k {
            heap.peek().unwrap().0
        } else {
            Score::MIN
        }
    };
    match mode {
        0 => super::block_wand(scorers, Score::MIN, &mut callback),
        1 => super::super::block_maxscore::block_maxscore::<false>(
            scorers,
            Score::MIN,
            8192,
            &mut callback,
        ),
        2 => super::super::block_maxscore::block_maxscore::<true>(
            scorers,
            Score::MIN,
            8192,
            &mut callback,
        ),
        3 => scorers
            .into_iter()
            .next()
            .unwrap()
            .for_each_pruning_batch(Score::MIN, &mut callback),
        4 => scorers
            .into_iter()
            .next()
            .unwrap()
            .for_each_pruning_batch_size::<8, 8>(Score::MIN, &mut callback),
        5 => scorers
            .into_iter()
            .next()
            .unwrap()
            .for_each_pruning_batch_size::<16, 16>(Score::MIN, &mut callback),
        6 => scorers
            .into_iter()
            .next()
            .unwrap()
            .for_each_pruning_batch_size::<64, 64>(Score::MIN, &mut callback),
        7 => scorers
            .into_iter()
            .next()
            .unwrap()
            .for_each_pruning_batch_size::<128, 128>(Score::MIN, &mut callback),
        _ => unreachable!(),
    }
    let mut scores: Vec<_> = heap.into_iter().map(|v| v.0).collect();
    scores.sort_unstable_by(Score::total_cmp);
    scores
}

#[test]
#[ignore = "release-mode cutover benchmark; prints CSV"]
fn benchmark_scheduler_cutovers() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    println!("CUTOVER,max_doc,terms,df_per_term,layout,k,mode,median_ns");
    for max_doc in [65536u32, 1048576] {
        let mut rng = fastrand::Rng::with_seed(123);
        let norms: Vec<u32> = (0..max_doc).map(|_| rng.u32(64..321)).collect();
        let avg = norms.iter().map(|&v| v as u64).sum::<u64>() as f32 / max_doc as f32;
        for terms in [2, 3, 5, 10] {
            for df in [32u32, 128, 512, 2048, 8192] {
                for clustered in [false, true] {
                    let span = if clustered { max_doc / 8 } else { max_doc };
                    let step = span / df;
                    let scorers: Vec<_> = (0..terms)
                        .map(|_| {
                            let phase = rng.u32(0..step);
                            let postings: Vec<_> = (0..df)
                                .map(|i| (i * step + phase, rng.u32(1..17)))
                                .collect();
                            let weight = Bm25Weight::for_one_term(
                                df as u64,
                                max_doc as u64,
                                avg,
                                Bm25Params::default(),
                            );
                            {
                                let norm_reader =
                                    crate::fieldnorm::FieldNormReader::for_test(&norms);
                                let mut scorer =
                                    TermScorer::create_for_test(&postings, &norms, weight);
                                scorer.block_cursor().set_posting_norms_for_test(
                                    postings
                                        .iter()
                                        .map(|&(doc, _)| norm_reader.fieldnorm_id(doc))
                                        .collect(),
                                );
                                scorer
                            }
                        })
                        .collect();
                    for k in [10, 100] {
                        let expected = benchmark_pruning(scorers.clone(), k, 0);
                        let mut times = [Vec::new(), Vec::new(), Vec::new()];
                        for round in 0..5 {
                            for turn in 0..3 {
                                let mode = (round + turn) % 3;
                                let actual = benchmark_pruning(scorers.clone(), k, mode);
                                assert_eq!(actual.len(), expected.len());
                                assert!(actual
                                    .iter()
                                    .zip(&expected)
                                    .all(|(&a, &b)| nearly_equals(a, b)));
                                let start = Instant::now();
                                let mut n = 0u32;
                                while start.elapsed() < Duration::from_millis(8) {
                                    black_box(benchmark_pruning(
                                        black_box(scorers.clone()),
                                        k,
                                        mode,
                                    ));
                                    n += 1;
                                }
                                times[mode].push(start.elapsed().as_nanos() as u64 / u64::from(n));
                            }
                        }
                        for (mode, samples) in times.iter_mut().enumerate() {
                            samples.sort_unstable();
                            println!(
                                "CUTOVER,{max_doc},{terms},{df},{clustered},{k},{mode},{}",
                                samples[2]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "release-mode cutover benchmark; prints CSV"]
fn benchmark_single_term_cutovers() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    println!("SINGLE,postings,k,mode,median_ns");
    for count in [1u32, 4, 8, 16, 32, 64, 127, 128, 129, 256, 1024, 8192] {
        let norms: Vec<_> = (0..count).map(|i| 64 + i % 257).collect();
        let avg = norms.iter().map(|&v| v as u64).sum::<u64>() as f32 / count as f32;
        let postings: Vec<_> = (0..count).map(|i| (i, 1 + (i * 137) % 16)).collect();
        let weight =
            Bm25Weight::for_one_term(count as u64, count as u64, avg, Bm25Params::default());
        let norm_reader = crate::fieldnorm::FieldNormReader::for_test(&norms);
        let mut scorer = TermScorer::create_for_test(&postings, &norms, weight);
        scorer.block_cursor().set_posting_norms_for_test(
            postings
                .iter()
                .map(|&(doc, _)| norm_reader.fieldnorm_id(doc))
                .collect(),
        );
        let scorers = vec![scorer];
        for k in [1, 10, 100] {
            let expected = benchmark_pruning(scorers.clone(), k, 0);
            let modes = [0, 4, 5, 3, 6, 7];
            let mut times: Vec<Vec<u64>> = modes.iter().map(|_| Vec::new()).collect();
            for round in 0..7 {
                for turn in 0..modes.len() {
                    let mode = (round + turn) % modes.len();
                    assert_eq!(benchmark_pruning(scorers.clone(), k, modes[mode]), expected);
                    let start = Instant::now();
                    let mut n = 0u32;
                    while start.elapsed() < Duration::from_millis(10) {
                        black_box(benchmark_pruning(
                            black_box(scorers.clone()),
                            k,
                            modes[mode],
                        ));
                        n += 1;
                    }
                    times[mode].push(start.elapsed().as_nanos() as u64 / u64::from(n));
                }
            }
            for (mode, samples) in times.iter_mut().enumerate() {
                samples.sort_unstable();
                println!("SINGLE,{count},{k},{},{}", modes[mode], samples[3]);
            }
        }
    }
}

#[test]
#[ignore = "release-mode cutover benchmark; prints CSV"]
fn benchmark_scheduler_boundary() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    println!("BOUNDARY,max_doc,terms,df_per_term,layout,k,mode,median_ns");
    for max_doc in [4096u32, 65536, 1048576] {
        let mut rng = fastrand::Rng::with_seed(123);
        let norms: Vec<u32> = (0..max_doc).map(|_| rng.u32(64..321)).collect();
        let avg = norms.iter().map(|&v| v as u64).sum::<u64>() as f32 / max_doc as f32;
        for terms in [2, 5, 10] {
            for denominator in [128u32, 256, 512, 1024] {
                let df = (max_doc / (terms * denominator)).max(1);
                for clustered in [false, true] {
                    let span = max_doc;
                    let step = span / df;
                    let scorers: Vec<_> = (0..terms)
                        .map(|_| {
                            let phase = rng.u32(0..step);
                            let postings: Vec<_> = (0..df)
                                .map(|i| {
                                    (
                                        if clustered && i == 0 {
                                            0
                                        } else {
                                            i * step + phase
                                        },
                                        if clustered {
                                            if i == 0 {
                                                512
                                            } else {
                                                1
                                            }
                                        } else {
                                            rng.u32(1..17)
                                        },
                                    )
                                })
                                .collect();
                            let weight = Bm25Weight::for_one_term(
                                df as u64,
                                max_doc as u64,
                                avg,
                                Bm25Params::default(),
                            );
                            {
                                let norm_reader =
                                    crate::fieldnorm::FieldNormReader::for_test(&norms);
                                let mut scorer =
                                    TermScorer::create_for_test(&postings, &norms, weight);
                                scorer.block_cursor().set_posting_norms_for_test(
                                    postings
                                        .iter()
                                        .map(|&(doc, _)| norm_reader.fieldnorm_id(doc))
                                        .collect(),
                                );
                                scorer
                            }
                        })
                        .collect();
                    for k in [10, 100] {
                        let expected = benchmark_pruning(scorers.clone(), k, 0);
                        let mut times = [Vec::new(), Vec::new(), Vec::new()];
                        for round in 0..5 {
                            for turn in 0..3 {
                                let mode = (round + turn) % 3;
                                let actual = benchmark_pruning(scorers.clone(), k, mode);
                                assert_eq!(actual.len(), expected.len());
                                assert!(actual
                                    .iter()
                                    .zip(&expected)
                                    .all(|(&a, &b)| nearly_equals(a, b)));
                                let start = Instant::now();
                                let mut n = 0u32;
                                while start.elapsed() < Duration::from_millis(8) {
                                    black_box(benchmark_pruning(
                                        black_box(scorers.clone()),
                                        k,
                                        mode,
                                    ));
                                    n += 1;
                                }
                                times[mode].push(start.elapsed().as_nanos() as u64 / u64::from(n));
                            }
                        }
                        for (mode, samples) in times.iter_mut().enumerate() {
                            samples.sort_unstable();
                            println!(
                                "BOUNDARY,{max_doc},{terms},{df},{clustered},{k},{mode},{}",
                                samples[2]
                            );
                        }
                    }
                }
            }
        }
    }
}
