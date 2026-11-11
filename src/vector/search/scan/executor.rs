use std::ops::Range;

use serde::{Deserialize, Serialize};

use super::layer_read::{decode_layer, Decoded, LayerRead, LayerSpec};
use super::plan::price;
use crate::vector::search::cluster_source::ClusterRows;
use crate::vector::storage::quantized::ReadScratch;
use crate::SegmentOrdinal;

/// Selected rows of one cluster's first quantized layer. Tasks never change probing state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorScanTask {
    pub segment: SegmentOrdinal,
    pub cluster: usize,
    pub rows: Range<usize>,
    pub offsets: Option<Vec<u32>>,
    pub sparse: bool,
}

/// Scored columns and storage accounting returned by a scan task.
#[derive(Default, Serialize, Deserialize)]
pub struct VectorScanOutput {
    pub(crate) decoded: Decoded,
    pub(crate) totals: crate::vector::telemetry::StageTotals,
}

/// Caller-owned transport for a cooperative vector scan. All participants supply the same
/// segments, query and executor; exactly one runs the global probing loop. Helpers only
/// score selected rows. `resolve` must return columns in task row order and make progress with no
/// helpers (for example, when a database launches fewer workers than requested).
pub trait VectorScanExecutor: Send + Sync {
    /// Elects the coordinator once for this scan.
    fn start(&self) -> crate::Result<bool>;
    /// Maximum number of clusters in flight, including completed results not yet consumed.
    fn capacity(&self) -> usize;
    /// Enqueues one cluster. The coordinator never exceeds `capacity` outstanding tasks.
    fn submit(&self, task: &VectorScanTask) -> crate::Result<usize>;
    /// Retrieves a task's result, helping compute queued tasks while waiting.
    fn resolve(
        &self,
        ticket: usize,
        compute: &mut dyn FnMut(&VectorScanTask) -> crate::Result<VectorScanOutput>,
    ) -> crate::Result<VectorScanOutput>;
    /// Services tasks until the coordinator finishes. Implementations must propagate errors and
    /// cancellation, including failures in another participant.
    fn work(
        &self,
        compute: &mut dyn FnMut(&VectorScanTask) -> crate::Result<VectorScanOutput>,
    ) -> crate::Result<()>;
    /// Releases helpers on both success and failure. Must be idempotent.
    fn finish(&self) -> crate::Result<()>;
}

pub(crate) fn decode_task(
    spec: &LayerSpec<'_>,
    task: &VectorScanTask,
) -> crate::Result<VectorScanOutput> {
    let (decoded, totals) = crate::vector::telemetry::measure(|| {
        let _stage =
            crate::vector::telemetry::enter_vector_stage(crate::vector::VectorStage::LayerScan(0));
        let selected = ClusterRows {
            cluster: task.cluster,
            rows: task.rows.clone(),
            offsets: task.offsets.as_deref(),
            docs: None,
            centroid: None,
        };
        let mut scratch = ReadScratch::default();
        let read = if task.sparse {
            price(spec, None, &selected, &mut scratch)?;
            LayerRead::Sparse
        } else {
            LayerRead::Band
        };
        let mut decoded = Decoded::default();
        decode_layer(spec, &selected, read, &mut decoded, &mut scratch)?;
        Ok::<_, crate::TantivyError>(decoded)
    });
    Ok(VectorScanOutput {
        decoded: decoded?,
        totals,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Barrier, Condvar, Mutex};

    use super::*;

    #[derive(Default)]
    struct State {
        started: bool,
        finished: bool,
        next: usize,
        jobs: VecDeque<(usize, VectorScanTask)>,
        results: HashMap<usize, crate::Result<VectorScanOutput>>,
    }

    pub(crate) struct ThreadExecutor {
        state: Mutex<State>,
        cv: Condvar,
        start: Barrier,
    }

    impl ThreadExecutor {
        pub(crate) fn new(participants: usize) -> Self {
            Self {
                state: Mutex::new(State::default()),
                cv: Condvar::new(),
                start: Barrier::new(participants),
            }
        }

        fn compute_one(
            &self,
            compute: &mut dyn FnMut(&VectorScanTask) -> crate::Result<VectorScanOutput>,
        ) -> bool {
            let Some((id, task)) = self.state.lock().unwrap().jobs.pop_back() else {
                return false;
            };
            let result = compute(&task);
            self.state.lock().unwrap().results.insert(id, result);
            self.cv.notify_all();
            true
        }
    }

    impl VectorScanExecutor for ThreadExecutor {
        fn start(&self) -> crate::Result<bool> {
            let coordinator = {
                let mut state = self.state.lock().unwrap();
                let first = !state.started;
                state.started = true;
                first
            };
            self.start.wait();
            Ok(coordinator)
        }

        fn capacity(&self) -> usize {
            8
        }

        fn submit(&self, task: &VectorScanTask) -> crate::Result<usize> {
            let mut state = self.state.lock().unwrap();
            let id = state.next;
            state.next += 1;
            state.jobs.push_back((id, task.clone()));
            self.cv.notify_all();
            Ok(id)
        }

        fn resolve(
            &self,
            ticket: usize,
            compute: &mut dyn FnMut(&VectorScanTask) -> crate::Result<VectorScanOutput>,
        ) -> crate::Result<VectorScanOutput> {
            loop {
                if let Some(result) = self.state.lock().unwrap().results.remove(&ticket) {
                    return result;
                }
                if self.compute_one(compute) {
                    continue;
                }
                let mut state = self.state.lock().unwrap();
                if state.results.contains_key(&ticket) {
                    continue;
                }
                state = self.cv.wait(state).unwrap();
                drop(state);
            }
        }

        fn work(
            &self,
            compute: &mut dyn FnMut(&VectorScanTask) -> crate::Result<VectorScanOutput>,
        ) -> crate::Result<()> {
            loop {
                if self.state.lock().unwrap().finished {
                    return Ok(());
                }
                if self.compute_one(compute) {
                    continue;
                }
                let mut state = self.state.lock().unwrap();
                if !state.finished && state.jobs.is_empty() {
                    state = self.cv.wait(state).unwrap();
                }
                drop(state);
            }
        }

        fn finish(&self) -> crate::Result<()> {
            self.state.lock().unwrap().finished = true;
            self.cv.notify_all();
            Ok(())
        }
    }

    #[test]
    fn helper_errors_release_waiters() {
        let executor = ThreadExecutor::new(4);
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        let mut fail = |_: &VectorScanTask| {
                            Err(crate::TantivyError::SystemError("decode failed".into()))
                        };
                        if executor.start().unwrap() {
                            let ticket = executor
                                .submit(&VectorScanTask {
                                    segment: 0,
                                    cluster: 0,
                                    rows: 0..256,
                                    offsets: None,
                                    sparse: false,
                                })
                                .unwrap();
                            let result = executor.resolve(ticket, &mut fail);
                            executor.finish().unwrap();
                            assert!(result.err().unwrap().to_string().contains("decode failed"));
                        } else {
                            executor.work(&mut fail).unwrap();
                        }
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
        });
    }
}
