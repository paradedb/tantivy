//! A host that embeds tantivy in a process with its own interrupt handling (Postgres, for one)
//! registers a check here. Long walks call it every few thousand steps, so a pending cancel
//! lands inside the walk instead of after it.

use std::sync::atomic::{AtomicPtr, Ordering};

static CHECK: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// Registers the check. It must be safe to call from every thread that runs a walk.
pub fn set_interrupt_check(check: fn()) {
    CHECK.store(check as *mut (), Ordering::Relaxed);
}

/// Runs the registered check, if there is one. The pointer publishes no data of its own, so a
/// relaxed load is a plain load.
#[inline]
pub fn check_for_interrupt() {
    let check = CHECK.load(Ordering::Relaxed);
    if !check.is_null() {
        // SAFETY: the pointer was made from a `fn()` in `set_interrupt_check`.
        let check: fn() = unsafe { std::mem::transmute::<*mut (), fn()>(check) };
        check();
    }
}
