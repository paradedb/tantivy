//! Read planning: a cluster's selected rows are read the cheapest way their storage allows, priced
//! by the storage blocks each read touches. A plan changes reads, never scores.

use crate::vector::search::cluster_source::ClusterRows;
use crate::vector::search::scan::layer_read::{LayerRead, LayerSpec};
use crate::vector::storage::quantized::{ReadCost, ReadScratch};
use crate::vector::storage::reader::VectorIndexReader;

// --- Costs ---

/// The ways to read a cluster's selected rows at a quantized layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayerCosts {
    /// The sparse read; `None` for a whole-cluster selection.
    pub(crate) sparse: Option<ReadCost>,
    /// The whole band.
    pub(crate) band: ReadCost,
    /// The centroid row either quantized read needs first: non-zero only for rows without a
    /// centroid score, with the exact plan on.
    pub(crate) centroid: ReadCost,
}

impl LayerCosts {
    /// The cheaper of the two quantized reads.
    fn cheapest(&self) -> ReadCost {
        self.sparse
            .map_or(self.band, |sparse| sparse.min(self.band))
    }
}

/// Every way to read a cluster's selected rows, priced from storage geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadCosts {
    /// The exact read of the selected rows; `None` with the exact plan off.
    pub(crate) exact: Option<ReadCost>,
    pub(crate) layer: LayerCosts,
}

// --- Planning ---

/// How a cluster's selected rows are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Read {
    /// The selected rows read and scored at full precision; they finish here.
    Exact,
    Layer(LayerRead),
}

/// Whether a test scores the rows of `(cluster, layer)` exactly.
#[cfg(test)]
pub(crate) type ExactRule = fn(usize, usize) -> bool;

/// Reads forced by a test; each applies only where the rows have that read.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PlanOverride {
    /// The quantized read of every cluster.
    pub(crate) read: Option<LayerRead>,
    /// The clusters scored exactly; overrides `read`.
    pub(crate) exact: Option<ExactRule>,
}

#[cfg(test)]
impl PlanOverride {
    /// Every cluster read as `read`.
    pub(crate) fn read(read: LayerRead) -> Self {
        Self {
            read: Some(read),
            exact: None,
        }
    }

    /// The clusters `rule` selects scored exactly; the layout plans the rest.
    pub(crate) fn exact(rule: ExactRule) -> Self {
        Self {
            read: None,
            exact: Some(rule),
        }
    }
}

/// Exact only when strictly cheaper than the cheaper quantized read with its centroid row, so
/// ties stay quantized; then sparse only when strictly cheaper than the band, so a tie reads the
/// band in one request.
pub(crate) fn plan(costs: &ReadCosts) -> Read {
    let layer = &costs.layer;
    if costs
        .exact
        .is_some_and(|exact| exact < layer.cheapest() + layer.centroid)
    {
        return Read::Exact;
    }
    match costs.layer.sparse {
        Some(sparse) if sparse < costs.layer.band => Read::Layer(LayerRead::Sparse),
        _ => Read::Layer(LayerRead::Band),
    }
}

/// [`plan`], unless `forced` names a read the rows of `(cluster, layer)` have.
#[cfg(test)]
pub(crate) fn forced_plan(
    costs: &ReadCosts,
    forced: Option<PlanOverride>,
    cluster: usize,
    layer: usize,
) -> Read {
    let Some(forced) = forced else {
        return plan(costs);
    };
    if forced.exact.is_some_and(|rule| rule(cluster, layer)) {
        return Read::Exact;
    }
    match forced.read {
        Some(LayerRead::Band) => Read::Layer(LayerRead::Band),
        Some(LayerRead::Sparse) if costs.layer.sparse.is_some() => Read::Layer(LayerRead::Sparse),
        _ => plan(costs),
    }
}

/// Prices the rows of `selected` at `spec`'s layer, and exactly through `exact` when the exact plan
/// is on. A sparse price leaves its rows and code runs in `scratch` for the read.
pub(crate) fn price(
    spec: &LayerSpec<'_>,
    exact: Option<&VectorIndexReader>,
    selected: &ClusterRows<'_>,
    scratch: &mut ReadScratch,
) -> crate::Result<ReadCosts> {
    let centroid = match (exact, selected.centroid) {
        (Some(reader), None) => reader.centroid_cost(selected.cluster)?,
        _ => ReadCost(0),
    };
    let exact = exact
        .map(|reader| reader.exact_cost(selected.cluster, selected.offsets))
        .transpose()?;
    let band = spec.reader.band_cost(selected.cluster)?;
    let sparse = selected
        .offsets
        .map(|offsets| spec.reader.sparse_cost(selected.cluster, offsets, scratch))
        .transpose()?;
    Ok(ReadCosts {
        exact,
        layer: LayerCosts {
            sparse,
            band,
            centroid,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Metric;
    use crate::vector::test_support::{first_segment, FixtureSpec};

    fn costs(band: usize, sparse: Option<usize>) -> ReadCosts {
        ReadCosts {
            exact: None,
            layer: LayerCosts {
                sparse: sparse.map(ReadCost),
                band: ReadCost(band),
                centroid: ReadCost(0),
            },
        }
    }

    fn with_exact(exact: usize, costs: ReadCosts) -> ReadCosts {
        ReadCosts {
            exact: Some(ReadCost(exact)),
            ..costs
        }
    }

    /// Exact only when strictly cheaper than the cheaper quantized read, then sparse only when
    /// strictly cheaper than the band; an override applies only where its read exists.
    #[test]
    fn plan_from_costs() {
        let band = Read::Layer(LayerRead::Band);
        let sparse = Read::Layer(LayerRead::Sparse);
        let exact = Read::Exact;
        let read = |read| Some(PlanOverride::read(read));
        let odd_exact = Some(PlanOverride::exact(|cluster, _| cluster % 2 == 1));
        let cases = [
            ("cheaper sparse", costs(5, Some(4)), None, sparse),
            ("a tie reads the band", costs(5, Some(5)), None, band),
            ("dearer sparse", costs(5, Some(6)), None, band),
            ("every row selected", costs(5, None), None, band),
            ("large counts", costs(1000, Some(999)), None, sparse),
            (
                "cheaper exact",
                with_exact(3, costs(5, Some(4))),
                None,
                exact,
            ),
            (
                "exact ties the cheaper read",
                with_exact(4, costs(5, Some(4))),
                None,
                sparse,
            ),
            (
                "dearer exact",
                with_exact(5, costs(5, Some(4))),
                None,
                sparse,
            ),
            (
                "exact under a lone band",
                with_exact(9, costs(10, None)),
                None,
                exact,
            ),
            (
                "exact ties a lone band",
                with_exact(10, costs(10, None)),
                None,
                band,
            ),
            (
                "the band is the cheaper read",
                with_exact(5, costs(4, Some(6))),
                None,
                band,
            ),
            (
                "forced band",
                costs(5, Some(1)),
                read(LayerRead::Band),
                band,
            ),
            (
                "forced sparse",
                costs(5, Some(9)),
                read(LayerRead::Sparse),
                sparse,
            ),
            (
                "no sparse read to force",
                costs(5, None),
                read(LayerRead::Sparse),
                band,
            ),
            ("forced exact", costs(5, Some(4)), odd_exact, exact),
        ];
        for (name, costs, forced, expected) in cases {
            assert_eq!(forced_plan(&costs, forced, 1, 0), expected, "{name}");
            if forced.is_none() {
                assert_eq!(plan(&costs), expected, "{name}");
            }
        }
        let layout = forced_plan(&costs(5, Some(4)), odd_exact, 2, 0);
        assert_eq!(
            layout, sparse,
            "the rule leaves other clusters to the layout"
        );
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
            let reader = first_segment(&searcher).1.vector_index(fixture.field)?;
            let rows = reader.index().expect("IVF segment").cluster_range(0).len();
            let one = vec![0];
            let dense: Vec<u32> = (0..rows as u32).filter(|row| row % 10 != 0).collect();
            let every: Vec<u32> = (0..rows as u32).collect();
            for (layer, layer_reader) in reader.quantization().unwrap().layers().iter().enumerate()
            {
                let band_cost = layer_reader.band_cost(0)?;
                let cost = |offsets: Option<&[u32]>| -> crate::Result<ReadCosts> {
                    let sparse = offsets
                        .map(|offsets| {
                            layer_reader.sparse_cost(0, offsets, &mut ReadScratch::default())
                        })
                        .transpose()?;
                    Ok(ReadCosts {
                        exact: None,
                        layer: LayerCosts {
                            sparse,
                            band: band_cost,
                            centroid: ReadCost(0),
                        },
                    })
                };
                let sparse = Read::Layer(LayerRead::Sparse);
                let band = Read::Layer(LayerRead::Band);
                let cases: [(&str, Option<&[u32]>, Read); 4] = [
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
