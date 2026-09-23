#[test]
#[ignore = "release-mode read-policy benchmark; prints CSV"]
fn benchmark_lazy_read_cutovers() -> crate::Result<()> {
    use std::hint::black_box;
    use std::ops::Range;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use crate::directory::{FileHandle, FileSlice, OwnedBytes};
    use crate::postings::serializer::PostingsSerializer;

    #[derive(Debug)]
    struct CopiedFile {
        bytes: Vec<u8>,
        calls: AtomicUsize,
        copied: AtomicUsize,
        pages: AtomicUsize,
    }
    impl HasLen for CopiedFile {
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }
    impl FileHandle for CopiedFile {
        fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.copied.fetch_add(range.len(), Ordering::Relaxed);
            self.pages.fetch_add(
                range.end.div_ceil(8192) - range.start / 8192,
                Ordering::Relaxed,
            );
            Ok(OwnedBytes::new(self.bytes[range].to_vec()))
        }
    }
    println!(
        "LAZY,postings,file_bytes,stride_blocks,buffer,median_ns,calls,copied_bytes,\
         page_acquisitions"
    );
    for count in [4096u32, 16384, 32768, 65536, 262144, 1048576] {
        let mut bytes = Vec::new();
        let mut serializer = PostingsSerializer::new(
            100.0,
            IndexRecordOption::WithFreqs,
            None,
            crate::Bm25Params::default(),
        );
        serializer.new_term(count, true);
        for i in 0..count {
            serializer.write_doc(i * 17, 1 + (i * 137) % 32);
        }
        serializer.close_term(count, &mut bytes)?;
        let file = Arc::new(CopiedFile {
            bytes,
            calls: AtomicUsize::new(0),
            copied: AtomicUsize::new(0),
            pages: AtomicUsize::new(0),
        });
        for stride in [1u32, 8, 64, 512] {
            let buffers = [0, 8192, 32768, 131072, 524288];
            let mut samples: Vec<Vec<u64>> = buffers.iter().map(|_| Vec::new()).collect();
            let mut counts = vec![(0, 0, 0); buffers.len()];
            for round in 0..5 {
                for turn in 0..buffers.len() {
                    let mode = (round + turn) % buffers.len();
                    let run = || -> crate::Result<()> {
                        let mut reader = BlockSegmentPostings::open_from_file(
                            count,
                            FileSlice::new(file.clone()),
                            IndexRecordOption::WithFreqs,
                            IndexRecordOption::WithFreqs,
                            buffers[mode],
                        )?;
                        for ordinal in (0..count).step_by((128 * stride) as usize) {
                            let within = reader.seek(ordinal * 17);
                            assert_eq!(black_box(reader.doc(within)), ordinal * 17);
                            black_box(reader.freqs()[within]);
                        }
                        Ok(())
                    };
                    file.calls.store(0, Ordering::Relaxed);
                    file.copied.store(0, Ordering::Relaxed);
                    file.pages.store(0, Ordering::Relaxed);
                    run()?;
                    counts[mode] = (
                        file.calls.load(Ordering::Relaxed),
                        file.copied.load(Ordering::Relaxed),
                        file.pages.load(Ordering::Relaxed),
                    );
                    let start = Instant::now();
                    let mut n = 0u32;
                    while start.elapsed() < Duration::from_millis(8) {
                        run()?;
                        n += 1;
                    }
                    samples[mode].push(start.elapsed().as_nanos() as u64 / u64::from(n));
                }
            }
            for (mode, sample) in samples.iter_mut().enumerate() {
                sample.sort_unstable();
                let (calls, copied, pages) = counts[mode];
                println!(
                    "LAZY,{count},{},{stride},{},{},{calls},{copied},{pages}",
                    file.len(),
                    buffers[mode],
                    sample[2]
                );
            }
        }
    }
    Ok(())
}
