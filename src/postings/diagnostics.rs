//! Optional per-thread counters for short postings experiments.

use std::cell::{Cell, RefCell};
use std::io;

use common::VInt;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default)]
pub enum ReadKind {
    #[default]
    Other,
    Eager {
        doc_freq: u32,
    },
    Header,
    Skips,
    Payload,
}

#[derive(Default, Debug, Serialize)]
pub struct Counters {
    pub candidate_bounds_tested: u64,
    pub candidate_bounds_rejected: u64,
    pub candidate_fieldnorm_lookups_avoided: u64,
    pub intersection_membership_rejections: u64,
    pub intersection_membership_matches: u64,
    pub lists_opened: u64,
    pub union_rejection_seeks: u64,
    pub union_rejection_loads: u64,
    pub union_rejection_loads_scored: u64,
    pub blocks_available: u64,
    pub blocks_decoded: u64,
    pub blocks_seek_skipped: u64,
    pub payload_bytes_available: u64,
}

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static READ_KIND: Cell<ReadKind> = const { Cell::new(ReadKind::Other) };
    static COUNTERS: RefCell<Counters> = RefCell::default();
}

pub struct ReadGuard(ReadKind);

impl ReadGuard {
    pub fn enter(kind: ReadKind) -> Self {
        Self(READ_KIND.replace(kind))
    }
}

impl Drop for ReadGuard {
    fn drop(&mut self) {
        READ_KIND.set(self.0);
    }
}

pub fn read_kind() -> ReadKind {
    READ_KIND.get()
}

pub fn reset(enabled: bool) {
    ENABLED.set(enabled);
    COUNTERS.take();
}

pub fn take() -> Counters {
    COUNTERS.take()
}

pub(super) fn opened(doc_freq: u32, payload_len: usize) {
    if ENABLED.get() {
        COUNTERS.with_borrow_mut(|c| {
            c.lists_opened += 1;
            c.blocks_available += u64::from(doc_freq).div_ceil(128);
            c.payload_bytes_available += payload_len as u64;
        });
    }
}

pub(super) fn decoded(remaining_docs: u32) {
    if ENABLED.get() && remaining_docs != 0 {
        COUNTERS.with_borrow_mut(|c| c.blocks_decoded += 1);
    }
}

pub(super) fn seek_skipped(before: u32, after: u32, was_loaded: bool) {
    if ENABLED.get() {
        let crossed = (before / 128).saturating_sub(after / 128);
        let skipped = crossed.saturating_sub(u32::from(was_loaded));
        COUNTERS.with_borrow_mut(|c| c.blocks_seek_skipped += u64::from(skipped));
    }
}

pub fn metadata_len(doc_freq: u32, mut bytes: &[u8]) -> io::Result<usize> {
    if doc_freq < 128 {
        return Ok(0);
    }
    let original_len = bytes.len();
    let skip_len = VInt::deserialize_u64(&mut bytes)? as usize;
    let prefix_len = original_len - bytes.len();
    prefix_len
        .checked_add(skip_len)
        .filter(|end| *end <= original_len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "Invalid postings skip length"))
}

pub(crate) fn decoded_count() -> u64 {
    COUNTERS.with_borrow(|c| c.blocks_decoded)
}

pub(crate) fn rejection_seek(before: u64) -> bool {
    if !ENABLED.get() {
        return false;
    }
    COUNTERS.with_borrow_mut(|c| {
        c.union_rejection_seeks += 1;
        let loaded = c.blocks_decoded - before;
        c.union_rejection_loads += loaded;
        loaded != 0
    })
}

pub(crate) fn rejection_block_scored() {
    if ENABLED.get() {
        COUNTERS.with_borrow_mut(|c| c.union_rejection_loads_scored += 1);
    }
}

pub(crate) fn intersection_membership(matched: bool) {
    if ENABLED.get() {
        COUNTERS.with_borrow_mut(|c| {
            c.intersection_membership_matches += u64::from(matched);
            c.intersection_membership_rejections += u64::from(!matched);
        });
    }
}

pub(crate) fn candidate_bound(rejected: bool, terms: usize) {
    if ENABLED.get() {
        COUNTERS.with_borrow_mut(|c| {
            c.candidate_bounds_tested += 1;
            c.candidate_bounds_rejected += u64::from(rejected);
            if rejected {
                c.candidate_fieldnorm_lookups_avoided += terms as u64;
            }
        });
    }
}
