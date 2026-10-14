//! Read planning: a cluster's selected rows are read the cheapest way their storage allows, priced
//! by the storage blocks each read touches. A plan changes reads, never scores.

use super::layer_read::{LayerRead, LayerSpec};
use crate::vector::cluster_source::ClusterRows;
use crate::vector::index_reader::{ReadCost, ReadScratch};

// --- Costs ---

/// The ways to read a cluster's selected rows at a quantized layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayerCosts {
    /// The sparse read; `None` for a whole-cluster selection.
    pub(crate) sparse: Option<ReadCost>,
    /// The whole band.
    pub(crate) band: ReadCost,
}

/// Every way to read a cluster's selected rows, priced from storage geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadCosts {
    pub(crate) layer: LayerCosts,
}

// --- Planning ---

/// How a cluster's selected rows are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Read {
    Layer(LayerRead),
}

/// Reads forced by a test; each applies only where the rows have that read.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PlanOverride {
    /// The quantized read of every cluster.
    pub(crate) read: Option<LayerRead>,
}

#[cfg(test)]
impl PlanOverride {
    /// Every cluster read as `read`.
    pub(crate) fn read(read: LayerRead) -> Self {
        Self { read: Some(read) }
    }
}

/// Sparse only when strictly cheaper than the band; a tie reads the band in one request.
pub(crate) fn plan(costs: &ReadCosts) -> Read {
    match costs.layer.sparse {
        Some(sparse) if sparse < costs.layer.band => Read::Layer(LayerRead::Sparse),
        _ => Read::Layer(LayerRead::Band),
    }
}

/// [`plan`], unless `forced` names a read the rows have.
#[cfg(test)]
pub(crate) fn forced_plan(costs: &ReadCosts, forced: Option<PlanOverride>) -> Read {
    match forced.and_then(|forced| forced.read) {
        Some(LayerRead::Band) => Read::Layer(LayerRead::Band),
        Some(LayerRead::Sparse) if costs.layer.sparse.is_some() => Read::Layer(LayerRead::Sparse),
        _ => plan(costs),
    }
}

/// Prices the rows of `selected` at `spec`'s layer. A sparse price leaves its rows and code runs in
/// `scratch` for the read.
pub(crate) fn price(
    spec: &LayerSpec<'_>,
    selected: &ClusterRows<'_>,
    scratch: &mut ReadScratch,
) -> crate::Result<ReadCosts> {
    let band = spec.reader.band_cost(selected.cluster)?;
    let sparse = selected
        .offsets
        .map(|offsets| spec.reader.sparse_cost(selected.cluster, offsets, scratch))
        .transpose()?;
    Ok(ReadCosts {
        layer: LayerCosts { sparse, band },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;
    use crate::vector::test_support::FixtureSpec;

    fn costs(band: usize, sparse: Option<usize>) -> ReadCosts {
        ReadCosts {
            layer: LayerCosts {
                sparse: sparse.map(ReadCost),
                band: ReadCost(band),
            },
        }
    }

    /// Sparse only when strictly cheaper than the band, and never for a whole-cluster selection;
    /// an override applies only where its read exists.
    #[test]
    fn plan_from_costs() {
        let band = Read::Layer(LayerRead::Band);
        let sparse = Read::Layer(LayerRead::Sparse);
        let cases = [
            ("cheaper sparse", costs(5, Some(4)), None, sparse),
            ("a tie reads the band", costs(5, Some(5)), None, band),
            ("dearer sparse", costs(5, Some(6)), None, band),
            ("every row selected", costs(5, None), None, band),
            ("large counts", costs(1000, Some(999)), None, sparse),
            (
                "forced band",
                costs(5, Some(1)),
                Some(PlanOverride::read(LayerRead::Band)),
                band,
            ),
            (
                "forced sparse",
                costs(5, Some(9)),
                Some(PlanOverride::read(LayerRead::Sparse)),
                sparse,
            ),
            (
                "no sparse read to force",
                costs(5, None),
                Some(PlanOverride::read(LayerRead::Sparse)),
                band,
            ),
        ];
        for (name, costs, forced, expected) in cases {
            assert_eq!(forced_plan(&costs, forced), expected, "{name}");
            if forced.is_none() {
                assert_eq!(plan(&costs), expected, "{name}");
            }
        }
    }

    /// Costs follow storage geometry, priced by the reader alone: in bytes in RAM and in blocks on
    /// pages, one row reads sparsely, nine rows in ten read the band on pages, and every row
    /// never reads sparsely.
    #[test]
    fn costs_from_geometry() -> crate::Result<()> {
        let ram = FixtureSpec::clustered(Metric::L2, 64, 4, 400)
            .schedule(&[1, 4])
            .build()?;
        let paged = FixtureSpec::clustered(Metric::L2, 1024, 2, 1200)
            .schedule(&[1, 4])
            .paged()
            .build()?;
        for (storage, fixture) in [("ram", &ram), ("paged", &paged)] {
            let searcher = fixture.index.reader()?.searcher();
            let reader = searcher.segment_readers()[0].vector_index(fixture.field)?;
            let rows = reader.index().expect("IVF segment").cluster_range(0).len();
            let one = vec![0];
            let dense: Vec<usize> = (0..rows).filter(|row| row % 10 != 0).collect();
            let every: Vec<usize> = (0..rows).collect();
            for (layer, layer_reader) in reader.quantization().unwrap().layers().iter().enumerate()
            {
                let band_cost = layer_reader.band_cost(0)?;
                let cost = |offsets: Option<&[usize]>| -> crate::Result<ReadCosts> {
                    let sparse = offsets
                        .map(|offsets| {
                            layer_reader.sparse_cost(0, offsets, &mut ReadScratch::default())
                        })
                        .transpose()?;
                    Ok(ReadCosts {
                        layer: LayerCosts {
                            sparse,
                            band: band_cost,
                        },
                    })
                };
                let sparse = Read::Layer(LayerRead::Sparse);
                let band = Read::Layer(LayerRead::Band);
                let cases: [(&str, Option<&[usize]>, Read); 4] = [
                    ("whole cluster", None, band),
                    ("one row", Some(&one), sparse),
                    (
                        "nine rows in ten",
                        Some(&dense),
                        if storage == "paged" { band } else { sparse },
                    ),
                    ("every row", Some(&every), band),
                ];
                for (name, offsets, expected) in cases {
                    let costs = cost(offsets)?;
                    assert_eq!(
                        plan(&costs),
                        expected,
                        "{storage} layer {layer} {name}: {costs:?}"
                    );
                }
                let code_bytes = rows * layer_reader.code_stride();
                if storage == "ram" {
                    assert!(
                        band_cost.0 > code_bytes,
                        "ram costs count bytes: {band_cost:?}"
                    );
                } else {
                    assert!(
                        band_cost.0 <= code_bytes.div_ceil(8192) + 4,
                        "paged costs count blocks: {band_cost:?}"
                    );
                }
            }
        }
        Ok(())
    }
}
