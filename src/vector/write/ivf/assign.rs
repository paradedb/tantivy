//! The assign step: every vector to its primary centroid and, when the field replicates, to a
//! second one, and the rows the Data entry's cluster blocks are written from.

use std::time::{Duration, Instant};

use itertools::Itertools;

use super::encode::RowEncoder;
use super::records::MergedRecords;
use super::source::MergeVectorSource;
use super::{AssignedVector, FieldMerge, IvfBuildTimings};
use crate::vector::clustering::soar::SOAR_LAMBDA;
use crate::vector::clustering::{AssignedBatch, AssignmentShortlist, IvfCentroids, IvfClusterer};
use crate::vector::router::{BuiltRouter, RouterKind, MIN_ROUTED_ASSIGNMENT_CENTROIDS};
use crate::vector::write::RowAddress;
use crate::{DocId, TantivyError};

/// The rows of a field's cluster blocks.
pub(super) enum ClusterRows {
    /// Plain storage: every assigned row sorted by (cluster, target doc) and the clusters' row
    /// offsets, a prefix sum of `num_centroids + 1` entries; the rows are read from the sources.
    Addresses {
        assigned_vectors: Vec<AssignedVector>,
        cluster_offsets: Vec<u64>,
    },
    /// Quantized storage: the rows the assign pass encoded, in (cluster, target doc) order.
    Encoded(MergedRecords),
}

/// A field's assignment.
pub(super) struct Assigned {
    pub(super) rows: ClusterRows,
    /// Documents with a primary row.
    pub(super) primaries: usize,
    /// Rows, replicas included.
    pub(super) total: usize,
    /// Time inside the assign pass spent encoding rows and writing their records: parts of
    /// `other`, not of `assign`.
    pub(super) encode_and_write: Duration,
}

/// What the assign pass does with each batch: collect a plain field's rows, or encode a quantized
/// field's.
enum Sink<'a> {
    Collect(Vec<AssignedVector>),
    Encode(Box<RowEncoder<'a>>),
}

/// Assigns each of the field's vectors to one of `centroids`, and to a second one when the field
/// replicates, reading the sources in storage order through a [`MergeVectorSource`]. The clusterer
/// assigns through the router's shortlist when the router is quantized and large enough, else
/// exhaustively. A quantized field's rows are encoded against their clusters while each batch is
/// still decoded, so its Data entry reads no source row.
pub(super) fn assign<'a>(
    merge: &FieldMerge<'a>,
    clusterer: &dyn IvfClusterer,
    source_rows: &[Option<RowAddress>],
    centroids: &IvfCentroids,
    centroid_bytes: &'a [u8],
    router: &BuiltRouter,
    timings: &mut IvfBuildTimings,
) -> crate::Result<Assigned> {
    let opts = merge.opts;
    let IvfCentroids::F32(centroid_matrix) = centroids;
    let num_centroids = centroid_matrix.rows;
    // Assignment is per vector, so the source follows storage order.
    let assign_sources: Vec<(RowAddress, DocId)> = source_rows
        .iter()
        .enumerate()
        .filter_map(|(doc_id, row)| row.map(|row| (row, doc_id as DocId)))
        .sorted()
        .collect();
    let num_rows = assign_sources.len();

    let mut sink = match merge.quantization {
        Some(_) => Sink::Encode(Box::new(RowEncoder::new(merge, centroid_bytes)?)),
        None => Sink::Collect(Vec::with_capacity(num_rows)),
    };
    let mut source = MergeVectorSource::new(
        &merge.readers,
        &assign_sources,
        opts.dim(),
        merge.ctx.cancel,
    );
    let lambda = (merge.replication >= 2).then_some(SOAR_LAMBDA);
    let routed = router.kind() == RouterKind::QuantizedStacked
        && num_centroids >= MIN_ROUTED_ASSIGNMENT_CENTROIDS;
    let candidates = |vector: &[f32]| router.assignment_candidates(vector);
    let candidates: AssignmentShortlist = &candidates;
    let mut assigned = 0;
    clusterer.assign_batches(
        opts,
        &mut source,
        centroids,
        lambda,
        routed.then_some(candidates),
        &mut |batch: AssignedBatch| {
            check_batch(&batch, assigned, num_rows, opts.dim(), num_centroids)?;
            let docs = &assign_sources[batch.rows.clone()];
            match &mut sink {
                Sink::Collect(assigned_vectors) => collect(assigned_vectors, &batch, docs),
                Sink::Encode(encoder) => encoder.accept(&batch, docs)?,
            }
            assigned = batch.rows.end;
            Ok(())
        },
    )?;
    timings.assign_reads += source.reads;
    if assigned != num_rows {
        return Err(TantivyError::InvalidArgument(format!(
            "IvfClusterer assigned {assigned} clusters for {num_rows} vectors"
        )));
    }

    Ok(match sink {
        Sink::Collect(mut assigned_vectors) => {
            let primaries = num_rows;
            let total = assigned_vectors.len();
            let cluster_offsets = cluster_offsets(&mut assigned_vectors, num_centroids);
            Assigned {
                rows: ClusterRows::Addresses {
                    assigned_vectors,
                    cluster_offsets,
                },
                primaries,
                total,
                encode_and_write: Duration::ZERO,
            }
        }
        Sink::Encode(encoder) => {
            let (primaries, total) = (encoder.primaries, encoder.total);
            timings.encode += encoder.encode;
            timings.data_write += encoder.spill;
            let finish_start = Instant::now();
            let encode_and_write = encoder.encode + encoder.spill;
            let records = encoder.finish()?;
            let finish = finish_start.elapsed();
            timings.data_write += finish;
            Assigned {
                rows: ClusterRows::Encoded(records),
                primaries,
                total,
                encode_and_write: encode_and_write + finish,
            }
        }
    })
}

/// Checks that a batch continues the scan at `assigned`, is shaped as its rows, and names valid,
/// distinct clusters.
fn check_batch(
    batch: &AssignedBatch,
    assigned: usize,
    num_rows: usize,
    dim: usize,
    num_centroids: usize,
) -> crate::Result<()> {
    let rows = batch.rows.len();
    if batch.rows.start != assigned
        || batch.rows.end > num_rows
        || batch.pairs.len() != rows
        || batch.vectors.rows != rows
        || batch.vectors.dims != dim
        || batch.vectors.values.len() != rows * dim
    {
        return Err(TantivyError::InvalidArgument(format!(
            "IvfClusterer handed over rows {:?} with {} clusters and {} vectors of dimension {} \
             after {assigned} of {num_rows} rows",
            batch.rows,
            batch.pairs.len(),
            batch.vectors.rows,
            batch.vectors.dims
        )));
    }
    let check = |cluster: u32| -> crate::Result<()> {
        if cluster as usize >= num_centroids {
            return Err(TantivyError::InvalidArgument(format!(
                "IvfClusterer assigned vector to cluster {cluster}, but only {num_centroids} \
                 centroids were trained"
            )));
        }
        Ok(())
    };
    for &(primary, secondary) in batch.pairs {
        check(primary)?;
        if let Some(secondary) = secondary {
            check(secondary)?;
            if secondary == primary {
                return Err(TantivyError::InvalidArgument(format!(
                    "IvfClusterer replicated a vector into its own cluster {primary}"
                )));
            }
        }
    }
    Ok(())
}

/// Appends a plain field's assigned rows: a replicated vector yields two entries with the same
/// source and document, the primary first.
fn collect(
    assigned_vectors: &mut Vec<AssignedVector>,
    batch: &AssignedBatch,
    docs: &[(RowAddress, DocId)],
) {
    for (&(primary, secondary), &(source, target_doc_id)) in batch.pairs.iter().zip(docs) {
        assigned_vectors.push(AssignedVector {
            cluster: primary as usize,
            target_doc_id,
            source,
            primary: true,
        });
        if let Some(secondary) = secondary {
            assigned_vectors.push(AssignedVector {
                cluster: secondary as usize,
                target_doc_id,
                source,
                primary: false,
            });
        }
    }
}

/// Sorts `assigned` by (cluster, target doc) and returns the clusters' row offsets: a prefix sum
/// of `num_centroids + 1` entries. A document appears at most once per cluster, so each block's
/// documents stay strictly ascending.
fn cluster_offsets(assigned: &mut [AssignedVector], num_centroids: usize) -> Vec<u64> {
    let mut cluster_counts = vec![0usize; num_centroids];
    for assigned_vector in assigned.iter() {
        cluster_counts[assigned_vector.cluster] += 1;
    }

    assigned.sort_unstable_by_key(|vector| (vector.cluster, vector.target_doc_id));

    let mut cluster_offsets: Vec<u64> = Vec::with_capacity(num_centroids + 1);
    let mut next_offset = 0u64;
    cluster_offsets.push(next_offset);
    for cluster_count in cluster_counts {
        next_offset += cluster_count as u64;
        cluster_offsets.push(next_offset);
    }
    cluster_offsets
}
