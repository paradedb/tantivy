//! Deterministic foreground-merge allocation reproduction.
//! Usage: merge_memory <disjoint|interleaved> <docs-per-segment> <repetitions>
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use tantivy::collector::Count;
use tantivy::indexer::{IndexWriterOptions, NoMergePolicy};
use tantivy::query::PhraseQuery;
use tantivy::schema::{Schema, FAST, TEXT};
use tantivy::{doc, Index, IndexSettings, IndexSortByField, IndexWriter, Order, Term};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn allocated(n: usize) {
    let live = LIVE.fetch_add(n, SeqCst) + n;
    PEAK.fetch_max(live, SeqCst);
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            allocated(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            allocated(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        LIVE.fetch_sub(l.size(), SeqCst);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = System.realloc(p, l, n);
        if !q.is_null() {
            if n >= l.size() {
                allocated(n - l.size());
            } else {
                LIVE.fetch_sub(l.size() - n, SeqCst);
            }
        }
        q
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
fn main() -> tantivy::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: merge_memory disjoint|interleaved docs-per-segment repetitions"
    );
    let mode = &args[1];
    assert!(mode == "disjoint" || mode == "interleaved");
    let n: usize = args[2].parse().unwrap();
    let reps: usize = args[3].parse().unwrap();
    assert!(n > 1 && reps > 0);
    let mut schema = Schema::builder();
    let key = schema.add_u64_field("key", FAST);
    let text = schema.add_text_field("text", TEXT);
    let dir = tempfile::tempdir()?;
    let index = Index::builder()
        .schema(schema.build())
        .settings(IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "key".into(),
                order: Order::Asc,
            }),
            ..Default::default()
        })
        .create_in_dir(dir.path())?;
    let budget = 15_000_000;
    let mut writer: IndexWriter = index.writer_with_num_threads(1, budget)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let content = "common anchor ".repeat(reps);
    for segment in 0..4 {
        for row in 0..n {
            let id = if mode == "interleaved" {
                row * 4 + segment
            } else {
                segment * n + row
            };
            writer.add_document(doc!(key => id as u64, text => content.as_str()))?;
        }
        writer.commit()?;
    }
    drop(writer);
    let ids = index.searchable_segment_ids()?;
    assert!(ids.len() >= 4);
    // A fresh writer has no indexing work in flight when the counter is reset.
    let mut writer: IndexWriter = index.writer_with_options(
        IndexWriterOptions::builder()
            .memory_budget_per_thread(budget)
            .num_worker_threads(0)
            .num_merge_threads(0)
            .build(),
    )?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let baseline = LIVE.load(SeqCst);
    PEAK.store(baseline, SeqCst);
    let start = std::time::Instant::now();
    writer.merge_foreground(&ids, true)?;
    let elapsed = start.elapsed();
    let peak = PEAK.load(SeqCst);
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    assert_eq!(searcher.num_docs(), (4 * n) as u64);
    let phrase = PhraseQuery::new(vec![
        Term::from_field_text(text, "common"),
        Term::from_field_text(text, "anchor"),
    ]);
    assert_eq!(searcher.search(&phrase, &Count)?, 4 * n);
    let absent = PhraseQuery::new(vec![
        Term::from_field_text(text, "common"),
        Term::from_field_text(text, "common"),
    ]);
    assert_eq!(searcher.search(&absent, &Count)?, 0);
    let keys = searcher.segment_readers()[0].fast_fields().u64("key")?;
    for id in 0..(4 * n) as u32 {
        assert_eq!(keys.first(id), Some(id as u64));
    }
    println!("{{\"mode\":\"{}\",\"docs\":{},\"repetitions\":{},\"input_segments\":{},\"writer_budget_bytes\":{},\"baseline_bytes\":{},\"merge_peak_live_bytes\":{},\"merge_peak_above_baseline_bytes\":{},\"merge_ms\":{},\"correctness\":\"pass\"}}",mode,4*n,reps,ids.len(),budget,baseline,peak,peak.saturating_sub(baseline),elapsed.as_millis());
    Ok(())
}
