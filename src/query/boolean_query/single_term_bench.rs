fn single_term_top_scores(scorers: Vec<TermScorer>, k: usize, batch: bool) -> Vec<Score> {
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
    if batch {
        scorers
            .into_iter()
            .next()
            .unwrap()
            .for_each_pruning_batch(Score::MIN, &mut callback);
    } else {
        super::block_wand_single_scorer(
            scorers.into_iter().next().unwrap(),
            Score::MIN,
            &mut callback,
        );
    }
    let mut scores: Vec<_> = heap.into_iter().map(|v| v.0).collect();
    scores.sort_unstable_by(Score::total_cmp);
    scores
}

#[test]
fn test_single_term_batch_boundaries() {
    for count in [1u32, 8, 9, 127, 128, 129, 255, 256, 257] {
        for stride in [1, 64] {
            let norms: Vec<_> = (0..count * stride).map(|i| 64 + i % 257).collect();
            let avg = norms.iter().map(|&n| n as u64).sum::<u64>() as f32 / norms.len() as f32;
            let postings: Vec<_> = (0..count)
                .map(|i| (i * stride, 1 + (i * 137) % 16))
                .collect();
            let weight = Bm25Weight::for_one_term(
                count as u64,
                norms.len() as u64,
                avg,
                Bm25Params::default(),
            );
            let scorer = TermScorer::create_for_test(&postings, &norms, weight);
            for start in [0, count / 2, count - 1] {
                let mut scorer = scorer.clone();
                scorer.seek(start * stride);
                for k in [1, 10, 100] {
                    assert_eq!(
                        single_term_top_scores(vec![scorer.clone()], k, true),
                        single_term_top_scores(vec![scorer.clone()], k, false),
                        "count={count}, stride={stride}, start={start}, k={k}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "release-mode benchmark"]
fn benchmark_single_term_batch() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    println!("SINGLE,postings,stride,k,batch,median_ns");
    for stride in [1, 64] {
        for count in [1u32, 4, 8, 16, 32, 64, 127, 128, 129, 256, 1024, 8192] {
            let norms: Vec<_> = (0..count * stride).map(|i| 64 + i % 257).collect();
            let avg = norms.iter().map(|&v| v as u64).sum::<u64>() as f32 / norms.len() as f32;
            let postings: Vec<_> = (0..count)
                .map(|i| (i * stride, 1 + (i * 137) % 16))
                .collect();
            let weight = Bm25Weight::for_one_term(
                count as u64,
                norms.len() as u64,
                avg,
                Bm25Params::default(),
            );
            let scorer = TermScorer::create_for_test(&postings, &norms, weight);
            let scorers = vec![scorer];
            for k in [1, 10, 100] {
                let expected = single_term_top_scores(scorers.clone(), k, false);
                let modes = [false, true];
                let mut times: Vec<Vec<u64>> = modes.iter().map(|_| Vec::new()).collect();
                for round in 0..7 {
                    for turn in 0..modes.len() {
                        let mode = (round + turn) % modes.len();
                        assert_eq!(
                            single_term_top_scores(scorers.clone(), k, modes[mode]),
                            expected
                        );
                        let start = Instant::now();
                        let mut n = 0u32;
                        while start.elapsed() < Duration::from_millis(10) {
                            black_box(single_term_top_scores(
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
                    println!("SINGLE,{count},{stride},{k},{},{}", modes[mode], samples[3]);
                }
            }
        }
    }
}
