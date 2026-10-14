//! Cluster sources: which clusters a segment scan admits, in what order, and when to stop.

pub(crate) mod controller;
pub(crate) mod router;
#[cfg(test)]
mod tests;

use router::RouterReport;

/// A finished source's report.
pub(crate) enum SourceReport {
    Router(RouterReport),
}
