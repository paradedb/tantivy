use std::cmp::Ordering;
use std::marker::PhantomData;
use std::sync::Arc;

use columnar::{ColumnType, ColumnValues, DynamicColumn};

use crate::collector::top_collector::{TopCollector, TopSegmentCollector};
use crate::collector::{Collector, SegmentCollector};
use crate::fastfield::{FastFieldNotAvailableError, FastValue};
use crate::{DateTime, DocAddress, DocId, Order, Score, SegmentReader};

pub trait Feature: Clone + Sync + Send + 'static {
    type Output: Clone + PartialOrd + Sync + Send + 'static;
    type SegmentOutput: Clone + PartialOrd + Sync + Send + 'static;

    const IS_SCORE: bool;

    fn open(&self, segment_reader: &SegmentReader, order: Order) -> crate::Result<FeatureColumn>;
    fn get(column: &FeatureColumn, order: Order, doc: DocId, score: Score) -> Self::SegmentOutput;
    fn decode(
        column: &FeatureColumn,
        order: Order,
        segment_output: impl Iterator<Item = Self::SegmentOutput> + DoubleEndedIterator,
    ) -> Vec<Self::Output>;
}

#[derive(Clone)]
pub struct ScoreFeature;

impl Feature for ScoreFeature {
    type Output = Score;
    type SegmentOutput = Score;

    const IS_SCORE: bool = true;

    fn open(&self, _segment_reader: &SegmentReader, _order: Order) -> crate::Result<FeatureColumn> {
        Ok(FeatureColumn::Score)
    }

    fn get(
        _column: &FeatureColumn,
        order: Order,
        _doc: DocId,
        score: Score,
    ) -> Self::SegmentOutput {
        if order.is_asc() {
            -score
        } else {
            score
        }
    }

    fn decode(
        _column: &FeatureColumn,
        order: Order,
        segment_output: impl Iterator<Item = Self::SegmentOutput> + DoubleEndedIterator,
    ) -> Vec<Self::Output> {
        if order.is_asc() {
            segment_output.map(|score| -score).collect()
        } else {
            segment_output.collect()
        }
    }
}

#[derive(Clone)]
pub struct FieldFeature<T>
where T: Clone + PartialOrd + Sync + Send + 'static
{
    field: String,
    field_type: ColumnType,
    _output_type: PhantomData<T>,
}

impl FieldFeature<String> {
    pub fn string(field: impl AsRef<str>) -> Self {
        Self {
            field: field.as_ref().to_owned(),
            field_type: ColumnType::Str,
            _output_type: PhantomData,
        }
    }
}

impl Feature for FieldFeature<String> {
    type Output = String;
    type SegmentOutput = u64;

    const IS_SCORE: bool = false;

    fn open(&self, segment_reader: &SegmentReader, order: Order) -> crate::Result<FeatureColumn> {
        FeatureColumn::open_field(segment_reader, &self.field, self.field_type, order)
    }

    fn get(column: &FeatureColumn, order: Order, doc: DocId, _score: Score) -> Self::SegmentOutput {
        let FeatureColumn::Field(_, sort_column) = column else {
            panic!("Field columns were not aligned to field definitions.");
        };
        let value = sort_column.get_val(doc);
        if order.is_desc() {
            value
        } else {
            u64::MAX - value
        }
    }

    fn decode(
        column: &FeatureColumn,
        order: Order,
        segment_output: impl Iterator<Item = Self::SegmentOutput> + DoubleEndedIterator,
    ) -> Vec<Self::Output> {
        let FeatureColumn::Field(DynamicColumn::Str(ff), _) = column else {
            todo!("Support more than strings!");
        };

        // Collect terms.
        let mut terms = Vec::with_capacity(segment_output.size_hint().0);
        let result = if order.is_asc() {
            ff.dictionary().sorted_ords_to_term_cb(
                segment_output.map(|term_ord| u64::MAX - term_ord),
                |term| {
                    terms.push(
                        std::str::from_utf8(term)
                            .expect("Failed to decode term as unicode")
                            .to_owned(),
                    );
                    Ok(())
                },
            )
        } else {
            ff.dictionary()
                .sorted_ords_to_term_cb(segment_output.rev(), |term| {
                    terms.push(
                        std::str::from_utf8(term)
                            .expect("Failed to decode term as unicode")
                            .to_owned(),
                    );
                    Ok(())
                })
        };

        assert!(
            result.expect("Failed to read terms from term dictionary"),
            "Not all terms were matched in segment."
        );

        if order.is_desc() {
            terms.reverse()
        }
        terms
    }
}

impl FieldFeature<u64> {
    pub fn u64(field: impl AsRef<str>) -> Self {
        Self {
            field: field.as_ref().to_owned(),
            field_type: ColumnType::U64,
            _output_type: PhantomData,
        }
    }
}

impl FieldFeature<i64> {
    pub fn i64(field: impl AsRef<str>) -> Self {
        Self {
            field: field.as_ref().to_owned(),
            field_type: ColumnType::I64,
            _output_type: PhantomData,
        }
    }
}

impl FieldFeature<f64> {
    pub fn f64(field: impl AsRef<str>) -> Self {
        Self {
            field: field.as_ref().to_owned(),
            field_type: ColumnType::F64,
            _output_type: PhantomData,
        }
    }
}

impl FieldFeature<bool> {
    pub fn bool(field: impl AsRef<str>) -> Self {
        Self {
            field: field.as_ref().to_owned(),
            field_type: ColumnType::Bool,
            _output_type: PhantomData,
        }
    }
}

impl FieldFeature<DateTime> {
    pub fn datetime(field: impl AsRef<str>) -> Self {
        Self {
            field: field.as_ref().to_owned(),
            field_type: ColumnType::DateTime,
            _output_type: PhantomData,
        }
    }
}

impl<F: FastValue> Feature for FieldFeature<F> {
    type Output = F;
    type SegmentOutput = u64;

    const IS_SCORE: bool = false;

    fn open(&self, segment_reader: &SegmentReader, order: Order) -> crate::Result<FeatureColumn> {
        FeatureColumn::open_field(segment_reader, &self.field, self.field_type, order)
    }

    fn get(column: &FeatureColumn, order: Order, doc: DocId, _score: Score) -> Self::SegmentOutput {
        let FeatureColumn::Field(_, sort_column) = column else {
            panic!("Field columns were not aligned to field definitions.");
        };
        let value = sort_column.get_val(doc);
        if order.is_desc() {
            value
        } else {
            u64::MAX - value
        }
    }

    fn decode(
        _column: &FeatureColumn,
        order: Order,
        segment_output: impl Iterator<Item = Self::SegmentOutput> + DoubleEndedIterator,
    ) -> Vec<Self::Output> {
        if order.is_desc() {
            segment_output.map(F::from_u64).collect()
        } else {
            segment_output.map(|o| F::from_u64(u64::MAX - o)).collect()
        }
    }
}

pub enum FeatureColumn {
    Score,
    Field(DynamicColumn, Arc<dyn ColumnValues<u64>>),
}

impl FeatureColumn {
    fn open_field(
        segment_reader: &SegmentReader,
        field: &str,
        field_type: ColumnType,
        order: Order,
    ) -> crate::Result<Self> {
        // We interpret this field as u64, regardless of its type, that way,
        // we avoid needless conversion. Regardless of the fast field type, the
        // mapping is monotonic, so it is sufficient to compute our top-K docs.
        //
        // The conversion will then happen only on the top-K docs for each segment.
        let sort_column_opt = segment_reader.fast_fields().u64_lenient(field)?;
        let (sort_column, _sort_column_type) =
            sort_column_opt.ok_or_else(|| FastFieldNotAvailableError {
                field_name: field.to_owned(),
            })?;

        let dynamic_column = segment_reader
            .fast_fields()
            .dynamic_column_handle(field, field_type)?
            .ok_or_else(|| FastFieldNotAvailableError {
                field_name: field.to_owned(),
            })?
            .open()?;
        let mut default_value = 0u64;
        if order.is_asc() {
            default_value = u64::MAX;
        }
        Ok(FeatureColumn::Field(
            dynamic_column,
            sort_column.first_or_default_col(default_value),
        ))
    }
}

pub trait TopOrderable: Clone + Sync + Send + 'static {
    type Output: Clone + PartialOrd + Sync + Send + 'static;
    type SegmentOutput: Clone + PartialOrd + Sync + Send + 'static;

    fn requires_scoring(&self) -> bool;

    fn feature_columns(
        &self,
        segment_reader: &SegmentReader,
    ) -> impl Iterator<Item = crate::Result<(FeatureColumn, Order)>>;

    fn segment_score(
        &mut self,
        features: &Vec<(FeatureColumn, Order)>,
        doc: DocId,
        score: Score,
    ) -> Self::SegmentOutput;

    fn decode(
        &self,
        features: &Vec<(FeatureColumn, Order)>,
        segment_output: Vec<(Self::SegmentOutput, DocAddress)>,
    ) -> Vec<(Self::Output, DocAddress)>;

    fn compare(&self, a: &(Self::Output, DocAddress), b: &(Self::Output, DocAddress)) -> bool;
}

pub struct TopOrderableSegmentCollector<O: TopOrderable> {
    segment_collector: TopSegmentCollector<O::SegmentOutput>,
    orderable: O,
    features: Vec<(FeatureColumn, Order)>,
}

impl<O: TopOrderable> SegmentCollector for TopOrderableSegmentCollector<O> {
    type Fruit = Vec<(O::Output, DocAddress)>;

    fn collect(&mut self, doc: DocId, score: Score) {
        let score = self.orderable.segment_score(&self.features, doc, score);
        self.segment_collector.collect(doc, score);
    }

    fn harvest(self) -> Vec<(O::Output, DocAddress)> {
        let harvested = self.segment_collector.harvest();
        self.orderable.decode(&self.features, harvested)
    }
}

pub(crate) struct TopOrderableCollector<O: TopOrderable> {
    orderable: O,
    // TODO: The type signature of `TopCollector` does not matter, because we only use it for
    // segments, and the type is chosen at `for_segment` time.
    collector: TopCollector<()>,
}

impl<O: TopOrderable> TopOrderableCollector<O> {
    pub(crate) fn new(orderable: O, collector: TopCollector<()>) -> TopOrderableCollector<O> {
        Self {
            orderable,
            collector,
        }
    }
}

impl<O: TopOrderable> Collector for TopOrderableCollector<O> {
    type Fruit = Vec<(O::Output, DocAddress)>;

    type Child = TopOrderableSegmentCollector<O>;

    fn for_segment(
        &self,
        segment_local_id: u32,
        segment_reader: &SegmentReader,
    ) -> crate::Result<Self::Child> {
        let segment_collector = self
            .collector
            .for_segment::<O::SegmentOutput>(segment_local_id, segment_reader);
        Ok(TopOrderableSegmentCollector {
            segment_collector,
            orderable: self.orderable.clone(),
            features: self
                .orderable
                .feature_columns(segment_reader)
                .collect::<crate::Result<Vec<_>>>()?,
        })
    }

    fn requires_scoring(&self) -> bool {
        self.orderable.requires_scoring()
    }

    fn merge_fruits(&self, segment_fruits: Vec<Self::Fruit>) -> crate::Result<Self::Fruit> {
        let merged = itertools::kmerge_by(
            segment_fruits,
            |a: &(O::Output, DocAddress), b: &(O::Output, DocAddress)| self.orderable.compare(a, b),
        )
        .skip(self.collector.offset)
        .take(self.collector.limit)
        .collect();
        Ok(merged)
    }
}

macro_rules! impl_top_orderable {
    ( $( ($T:ident, $idx:tt) ),+ ) => {
        impl<$($T: Feature),+> TopOrderable for ( $(($T, Order)),+ ,) {
            type Output = ( $($T::Output),+ ,);
            type SegmentOutput = ( $($T::SegmentOutput),+ ,);

            fn requires_scoring(&self) -> bool {
                // Returns true if any of the features require scoring.
                false $(|| $T::IS_SCORE)*
            }

            fn feature_columns(
                &self,
                segment_reader: &SegmentReader,
            ) -> impl Iterator<Item = crate::Result<(FeatureColumn, Order)>> {
                // Collects all feature columns from the tuple elements.
                [
                    $(
                        self.$idx.0.open(segment_reader, self.$idx.1.clone()).map(|fc| (fc, self.$idx.1.clone()))
                    ),+
                ]
                .into_iter()
            }

            fn segment_score(
                &mut self,
                features: &Vec<(FeatureColumn, Order)>,
                doc: DocId,
                score: Score,
            ) -> Self::SegmentOutput {
                // Scores the document for each feature and returns a tuple of segment outputs.
                (
                    $(
                        $T::get(&features[$idx].0, features[$idx].1.clone(), doc, score)
                    ),+
                    ,
                )
            }

            fn decode(
                &self,
                features: &Vec<(FeatureColumn, Order)>,
                segment_output: Vec<(Self::SegmentOutput, DocAddress)>,
            ) -> Vec<(Self::Output, DocAddress)> {
                // Decode each feature's values separately.
                $(
                    paste::paste! {
                        let mut [<decoded_values_ $idx>] = $T::decode(
                            &features[$idx].0,
                            features[$idx].1.clone(),
                            segment_output.iter().map(|(v, _)| v.$idx.clone()),
                        ).into_iter();
                    }
                )*

                // Zip the decoded values and doc addresses back together.
                let mut result = Vec::with_capacity(segment_output.len());
                for (_, doc_address) in segment_output {
                    let output_tuple = (
                        $(
                            paste::paste! {
                                [<decoded_values_ $idx>].next().unwrap()
                            }
                        ),+
                        ,
                    );
                    result.push((output_tuple, doc_address));
                }
                result
            }

            fn compare(&self, a: &(Self::Output, DocAddress), b: &(Self::Output, DocAddress)) -> bool {
                // Perform lexicographical comparison on the tuple elements.
                $(
                    if self.$idx.1.is_asc() {
                        match (a.0).$idx.partial_cmp(&(b.0).$idx) {
                            Some(Ordering::Less) => return true,
                            Some(Ordering::Greater) => return false,
                            Some(Ordering::Equal) | None => {} // Fall through
                        }
                    } else {
                        match (a.0).$idx.partial_cmp(&(b.0).$idx) {
                            Some(Ordering::Less) => return false,
                            Some(Ordering::Greater) => return true,
                            Some(Ordering::Equal) | None => {} // Fall through
                        }
                    }
                )*

                // Tie-breaker: DocAddress is always compared ascending.
                a.1 < b.1
            }
        }
    };
}

impl_top_orderable! { (F1, 0) }
impl_top_orderable! { (F1, 0), (F2, 1) }
impl_top_orderable! { (F1, 0), (F2, 1), (F3, 2) }

#[cfg(test)]
mod tests {
    use super::{FieldFeature, ScoreFeature};
    use crate::collector::TopDocs;
    use crate::query::{AllQuery, QueryParser};
    use crate::schema::{Schema, FAST, TEXT};
    use crate::{DocAddress, Document, Index, Order, Score};

    fn make_index() -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let city = schema_builder.add_text_field("city", TEXT | FAST);
        let catchphrase = schema_builder.add_text_field("catchphrase", TEXT);
        let altitude = schema_builder.add_f64_field("altitude", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        fn create_segment(index: &Index, docs: Vec<impl Document>) -> crate::Result<()> {
            let mut index_writer = index.writer_for_tests()?;
            for doc in docs {
                index_writer.add_document(doc)?;
            }
            index_writer.commit()?;
            Ok(())
        }

        create_segment(
            &index,
            vec![
                doc!(
                    city => "austin",
                    catchphrase => "Hills, Barbeque, Glow",
                    altitude => 149.0,
                ),
                doc!(
                    city => "greenville",
                    catchphrase => "Grow, Glow, Glow",
                    altitude => 27.0,
                ),
            ],
        )?;
        create_segment(
            &index,
            vec![doc!(
                city => "tokyo",
                catchphrase => "Glow, Glow, Glow",
                altitude => 40.0,
            )],
        )?;
        Ok(index)
    }

    #[test]
    fn test_order_by_string() -> crate::Result<()> {
        let index = make_index()?;

        fn query(
            index: &Index,
            order: Order,
            limit: usize,
            offset: usize,
        ) -> crate::Result<Vec<(String, DocAddress)>> {
            let searcher = index.reader()?.searcher();
            let top_collector = TopDocs::with_limit(limit)
                .and_offset(offset)
                .order_by(((FieldFeature::string("city"), order),));
            Ok(searcher
                .search(&AllQuery, &top_collector)?
                .into_iter()
                .map(|((s,), doc)| (s, doc))
                .collect())
        }

        assert_eq!(
            &query(&index, Order::Asc, 3, 0)?,
            &[
                ("austin".to_owned(), DocAddress::new(0, 0)),
                ("greenville".to_owned(), DocAddress::new(0, 1)),
                ("tokyo".to_owned(), DocAddress::new(1, 0)),
            ]
        );

        assert_eq!(
            &query(&index, Order::Asc, 2, 1)?,
            &[
                ("greenville".to_owned(), DocAddress::new(0, 1)),
                ("tokyo".to_owned(), DocAddress::new(1, 0)),
            ]
        );

        assert_eq!(
            &query(&index, Order::Desc, 3, 0)?,
            &[
                ("tokyo".to_owned(), DocAddress::new(1, 0)),
                ("greenville".to_owned(), DocAddress::new(0, 1)),
                ("austin".to_owned(), DocAddress::new(0, 0)),
            ]
        );

        assert_eq!(
            &query(&index, Order::Desc, 2, 1)?,
            &[
                ("greenville".to_owned(), DocAddress::new(0, 1)),
                ("austin".to_owned(), DocAddress::new(0, 0)),
            ]
        );
        Ok(())
    }

    #[test]
    fn test_order_by_f64() -> crate::Result<()> {
        let index = make_index()?;

        fn query(index: &Index, order: Order) -> crate::Result<Vec<(f64, DocAddress)>> {
            let searcher = index.reader()?.searcher();
            let top_collector =
                TopDocs::with_limit(3).order_by(((FieldFeature::f64("altitude"), order),));
            Ok(searcher
                .search(&AllQuery, &top_collector)?
                .into_iter()
                .map(|((f,), doc)| (f, doc))
                .collect())
        }

        assert_eq!(
            &query(&index, Order::Asc)?,
            &[
                (27.0, DocAddress::new(0, 1)),
                (40.0, DocAddress::new(1, 0)),
                (149.0, DocAddress::new(0, 0)),
            ]
        );

        assert_eq!(
            &query(&index, Order::Desc)?,
            &[
                (149.0, DocAddress::new(0, 0)),
                (40.0, DocAddress::new(1, 0)),
                (27.0, DocAddress::new(0, 1)),
            ]
        );

        Ok(())
    }

    #[test]
    fn test_order_by_score() -> crate::Result<()> {
        let index = make_index()?;

        fn query(index: &Index, order: Order) -> crate::Result<Vec<((Score,), DocAddress)>> {
            let searcher = index.reader()?.searcher();
            let top_collector = TopDocs::with_limit(3).order_by(((ScoreFeature, order),));
            let field = index.schema().get_field("catchphrase").unwrap();
            let query_parser = QueryParser::for_index(&index, vec![field]);
            let text_query = query_parser.parse_query("glow")?;
            searcher.search(&text_query, &top_collector)
        }

        assert_eq!(
            &query(&index, Order::Asc)?,
            &[
                ((0.13353144,), DocAddress::new(0, 0)),
                ((0.18360573,), DocAddress::new(0, 1)),
                ((0.20983513,), DocAddress::new(1, 0)),
            ]
        );

        assert_eq!(
            &query(&index, Order::Desc)?,
            &[
                ((0.20983513,), DocAddress::new(1, 0)),
                ((0.18360573,), DocAddress::new(0, 1)),
                ((0.13353144,), DocAddress::new(0, 0)),
            ]
        );
        Ok(())
    }

    #[test]
    fn test_order_by_score_then_string() -> crate::Result<()> {
        let index = make_index()?;

        fn query(
            index: &Index,
            score_order: Order,
            city_order: Order,
        ) -> crate::Result<Vec<((Score, String), DocAddress)>> {
            let searcher = index.reader()?.searcher();
            let top_collector = TopDocs::with_limit(3).order_by((
                (ScoreFeature, score_order),
                (FieldFeature::string("city"), city_order),
            ));
            searcher.search(&AllQuery, &top_collector)
        }

        assert_eq!(
            &query(&index, Order::Asc, Order::Asc)?,
            &[
                ((1.0, "austin".to_owned()), DocAddress::new(0, 0)),
                ((1.0, "greenville".to_owned()), DocAddress::new(0, 1)),
                ((1.0, "tokyo".to_owned()), DocAddress::new(1, 0)),
            ]
        );

        assert_eq!(
            &query(&index, Order::Asc, Order::Desc)?,
            &[
                ((1.0, "tokyo".to_owned()), DocAddress::new(1, 0)),
                ((1.0, "greenville".to_owned()), DocAddress::new(0, 1)),
                ((1.0, "austin".to_owned()), DocAddress::new(0, 0)),
            ]
        );
        Ok(())
    }
}
