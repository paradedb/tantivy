use common::TinySet;

use crate::docset::{BLOCK_NUM_TINYBITSETS, BLOCK_WINDOW};
use crate::query::size_hint::{estimate_intersection, estimate_union};
use crate::query::Scorer;
use crate::{DocId, DocSet, Score, TERMINATED};

type Block = [TinySet; BLOCK_NUM_TINYBITSETS];

#[derive(Clone, Copy)]
pub(crate) enum BitmapOperation {
    Union,
    Intersection,
    Exclude,
}

pub(crate) struct BitmapCombination {
    children: Vec<Box<dyn Scorer>>,
    operation: BitmapOperation,
    mask: Block,
    base: DocId,
    doc: DocId,
    size_hint: u32,
    cost: u64,
}

impl BitmapCombination {
    pub(crate) fn new(
        children: Vec<Box<dyn Scorer>>,
        operation: BitmapOperation,
        num_docs: u32,
    ) -> Self {
        assert!(!children.is_empty());
        let sizes = children.iter().map(|child| child.size_hint());
        let size_hint = match operation {
            BitmapOperation::Union => estimate_union(sizes, num_docs),
            BitmapOperation::Intersection => estimate_intersection(sizes, num_docs),
            BitmapOperation::Exclude => children[0].size_hint(),
        };
        let cost = match operation {
            BitmapOperation::Union => children.iter().map(|child| child.cost()).sum(),
            BitmapOperation::Intersection => {
                children.iter().map(|child| child.cost()).min().unwrap()
            }
            BitmapOperation::Exclude => children[0].cost(),
        };
        let mut result = Self {
            children,
            operation,
            mask: [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS],
            base: 0,
            doc: 0,
            size_hint,
            cost,
        };
        result.refill(0);
        result
    }

    fn next_base(&self) -> DocId {
        match self.operation {
            BitmapOperation::Union => self.children.iter().map(|child| child.doc()).min().unwrap(),
            BitmapOperation::Intersection => {
                self.children.iter().map(|child| child.doc()).max().unwrap()
            }
            BitmapOperation::Exclude => self.children[0].doc(),
        }
    }

    fn position(&mut self, target: DocId) -> bool {
        for (i, word) in self.mask.iter_mut().enumerate() {
            let word_base = self.base + i as u32 * 64;
            if target >= word_base + 64 {
                *word = TinySet::EMPTY;
            } else if target > word_base {
                *word = word.intersect(TinySet::range_greater_or_equal(target - word_base));
            }
            if let Some(bit) = (*word).into_iter().next() {
                self.doc = word_base + bit;
                return true;
            }
        }
        false
    }

    fn refill(&mut self, target: DocId) -> DocId {
        loop {
            let next = self.next_base().max(target);
            if next >= TERMINATED {
                self.doc = TERMINATED;
                return self.doc;
            }
            self.base = next / BLOCK_WINDOW * BLOCK_WINDOW;
            self.mask.fill(TinySet::EMPTY);
            self.children[0].fill_bitset_block(self.base, &mut self.mask);
            for child in &mut self.children[1..] {
                let mut other = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
                child.fill_bitset_block(self.base, &mut other);
                match self.operation {
                    BitmapOperation::Union => {
                        crate::docset::union_bitset_blocks(&mut self.mask, &other);
                    }
                    BitmapOperation::Intersection => {
                        super::intersection::and_blocks_and_return_is_empty(&mut self.mask, &other);
                    }
                    BitmapOperation::Exclude => {
                        for (word, other) in self.mask.iter_mut().zip(other) {
                            let bits = word.into_u64() & !other.into_u64();
                            *word = TinySet::deserialize(bits.to_le_bytes());
                        }
                    }
                }
            }
            if self.position(target) {
                return self.doc;
            }
        }
    }
}

impl DocSet for BitmapCombination {
    fn advance(&mut self) -> DocId {
        self.seek(self.doc.saturating_add(1))
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if target <= self.doc {
            return self.doc;
        }
        if self.position(target) {
            return self.doc;
        }
        self.refill(target)
    }

    fn doc(&self) -> DocId {
        self.doc
    }
    fn size_hint(&self) -> u32 {
        self.size_hint
    }
    fn cost(&self) -> u64 {
        self.cost
    }
    fn has_fast_bitset(&self) -> bool {
        true
    }

    fn fill_bitset_block(&mut self, base: DocId, mask: &mut Block) -> DocId {
        if self.doc < base {
            self.seek(base);
        }
        let horizon = base.saturating_add(BLOCK_WINDOW).min(TERMINATED);
        while self.doc < horizon {
            if base == self.base {
                crate::docset::union_bitset_blocks(mask, &self.mask);
                return self.seek(horizon);
            }
            for (i, word) in self.mask.iter().enumerate() {
                let word_base = self.base + i as u32 * 64;
                if word_base >= horizon || word_base + 64 <= base {
                    continue;
                }
                let mut bits = word.into_u64();
                if word_base < base {
                    bits >>= base - word_base;
                    mask[0] = mask[0].union(TinySet::deserialize(bits.to_le_bytes()));
                } else {
                    let delta = word_base - base;
                    let bucket = delta as usize / 64;
                    let shift = delta % 64;
                    mask[bucket] =
                        mask[bucket].union(TinySet::deserialize((bits << shift).to_le_bytes()));
                    if shift != 0 && bucket + 1 < mask.len() {
                        mask[bucket + 1] = mask[bucket + 1]
                            .union(TinySet::deserialize((bits >> (64 - shift)).to_le_bytes()));
                    }
                }
            }
            self.seek((self.base + BLOCK_WINDOW).min(horizon));
        }
        self.doc
    }
}

impl Scorer for BitmapCombination {
    fn score(&mut self) -> Score {
        1.0
    }
    fn constant_score(&self) -> Option<Score> {
        Some(1.0)
    }
}

#[cfg(test)]
mod tests {
    use common::BitSet;

    use super::*;
    use crate::query::{BitSetDocSet, ConstScorer, VecDocSet};

    fn leaf(docs: Vec<u32>, dense: bool) -> Box<dyn Scorer> {
        if dense {
            let mut bits = BitSet::with_max_value(8007);
            for doc in docs {
                bits.insert(doc);
            }
            Box::new(ConstScorer::new(BitSetDocSet::from(bits), 1.0))
        } else {
            Box::new(ConstScorer::new(VecDocSet::from(docs), 1.0))
        }
    }

    fn combined(
        a: &[u32],
        b: &[u32],
        operation: BitmapOperation,
        dense: bool,
    ) -> BitmapCombination {
        BitmapCombination::new(
            vec![leaf(a.to_vec(), true), leaf(b.to_vec(), dense)],
            operation,
            8007,
        )
    }

    #[test]
    fn bitmap_combinations_preserve_iteration_and_partial_windows() {
        let a: Vec<_> = (0..8007).filter(|doc| doc % 7 < 3).collect();
        let b: Vec<_> = (0..8007).filter(|doc| doc % 11 < 2).collect();
        for dense in [false, true] {
            for operation in [
                BitmapOperation::Union,
                BitmapOperation::Intersection,
                BitmapOperation::Exclude,
            ] {
                let expected: Vec<_> = (0..8007)
                    .filter(|doc| match operation {
                        BitmapOperation::Union => a.contains(doc) || b.contains(doc),
                        BitmapOperation::Intersection => a.contains(doc) && b.contains(doc),
                        BitmapOperation::Exclude => a.contains(doc) && !b.contains(doc),
                    })
                    .collect();
                let mut scorer = combined(&a, &b, operation, dense);
                for &doc in &expected {
                    assert_eq!(scorer.doc(), doc);
                    scorer.advance();
                }
                assert_eq!(scorer.doc(), TERMINATED);
                assert_eq!(scorer.advance(), TERMINATED);
                for start in [0, 1, 63, 64, 1019, 4103, 7901] {
                    let mut scorer = combined(&a, &b, operation, dense);
                    let positioned = scorer.seek(start + 17);
                    let mut results = Vec::new();
                    let mut base = start;
                    while scorer.doc() < TERMINATED {
                        let mut mask = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
                        scorer.fill_bitset_block(base, &mut mask);
                        results.extend(mask.into_iter().enumerate().flat_map(|(i, word)| {
                            word.into_iter().map(move |bit| base + i as u32 * 64 + bit)
                        }));
                        base += BLOCK_WINDOW;
                    }
                    assert_eq!(
                        results,
                        expected
                            .iter()
                            .copied()
                            .filter(|doc| *doc >= positioned)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn bitmap_boolean_queries_match_postings() -> crate::Result<()> {
        use crate::query::{EnableScoring, QueryParser};
        use crate::schema::{Schema, FAST, TEXT};
        use crate::Index;
        let mut schema = Schema::builder();
        let text = schema.add_text_field(
            "text",
            TEXT.set_indexing_options(
                TEXT.get_indexing_options()
                    .unwrap()
                    .clone()
                    .set_bitmap_postings(true),
            ),
        );
        let number = schema.add_u64_field("number", FAST);
        let mut index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_for_tests()?;
        for doc in 0..8007 {
            let mut terms = vec!["all"];
            if doc % 2 == 0 {
                terms.push("a");
            }
            if doc % 3 == 0 {
                terms.push("b");
            }
            if doc % 7 == 0 {
                terms.push("c");
            }
            if doc % 997 == 0 {
                terms.push("rare");
            }
            writer.add_document(doc!(text => terms.join(" "), number => doc as u64))?;
        }
        writer.commit()?;
        let parser = QueryParser::for_index(&index, vec![text]);
        let mut queries: Vec<Box<dyn crate::query::Query>> = [
            "a OR b",
            "a AND b",
            "rare AND a",
            "rare OR a",
            "(a OR b) AND c",
            "(a AND b) OR c",
            "a -b",
            "(a OR b) -c",
            "all -a",
            "a AND number:[101 TO 4321]",
            "a OR number:[101 TO 4321]",
            "a AND \"b c\"",
            "a AND missing",
        ]
        .into_iter()
        .map(|query| parser.parse_query(query))
        .collect::<Result<Vec<_>, _>>()?;
        use crate::query::{
            BooleanQuery, FuzzyTermQuery, Occur, RegexQuery, TermQuery, TermSetQuery,
        };
        use crate::schema::IndexRecordOption;
        use crate::Term;
        let terms: Vec<_> = ["a", "b", "c"]
            .into_iter()
            .map(|word| Term::from_field_text(text, word))
            .collect();
        queries.push(Box::new(BooleanQuery::with_minimum_required_clauses(
            terms
                .iter()
                .map(|term| {
                    (
                        Occur::Should,
                        Box::new(TermQuery::new(term.clone(), IndexRecordOption::Basic))
                            as Box<dyn crate::query::Query>,
                    )
                })
                .collect(),
            2,
        )));
        queries.push(Box::new(RegexQuery::from_pattern("[ab]", text)?));
        queries.push(Box::new(FuzzyTermQuery::new(
            Term::from_field_text(text, "a"),
            1,
            false,
        )));
        queries.push(Box::new(TermSetQuery::new(terms)));
        for query in queries {
            let mut results = Vec::new();
            let mut scored_results = Vec::new();
            for enabled in [false, true] {
                index.settings_mut().bitmap_postings.use_for_queries = enabled;
                let searcher = index.reader()?.searcher();
                scored_results.push(searcher.search(
                    query.as_ref(),
                    &crate::collector::TopDocs::with_limit(10).order_by_score(),
                )?);
                let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
                let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
                let mut docs = Vec::new();
                while scorer.doc() != TERMINATED {
                    docs.push(scorer.doc());
                    scorer.advance();
                }
                let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
                let mut blocks = Vec::new();
                while scorer.doc() != TERMINATED {
                    let base = scorer.doc() / BLOCK_WINDOW * BLOCK_WINDOW;
                    let mut mask = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
                    scorer.fill_bitset_block(base, &mut mask);
                    blocks.extend(mask.into_iter().enumerate().flat_map(|(i, word)| {
                        word.into_iter().map(move |bit| base + i as u32 * 64 + bit)
                    }));
                }
                assert_eq!(docs, blocks, "{query:?}");
                results.push(docs);
            }
            assert_eq!(results[0], results[1], "{query:?}");
            assert_eq!(scored_results[0], scored_results[1], "scored {query:?}");
        }
        Ok(())
    }

    struct BlocksOnly(BitSetDocSet);
    impl DocSet for BlocksOnly {
        fn doc(&self) -> DocId {
            self.0.doc()
        }
        fn size_hint(&self) -> u32 {
            self.0.size_hint()
        }
        fn advance(&mut self) -> DocId {
            panic!("intermediate bitmap enumerated")
        }
        fn has_fast_bitset(&self) -> bool {
            true
        }
        fn fill_bitset_block(&mut self, base: DocId, mask: &mut Block) -> DocId {
            self.0.fill_bitset_block(base, mask)
        }
    }

    #[test]
    fn nested_bitmap_operators_do_not_enumerate_leaves() {
        let leaf = |divisor| {
            let mut bits = BitSet::with_max_value(8007);
            for doc in (0..8007).step_by(divisor) {
                bits.insert(doc);
            }
            Box::new(ConstScorer::new(BlocksOnly(BitSetDocSet::from(bits)), 1.0)) as Box<dyn Scorer>
        };
        let union = BitmapCombination::new(vec![leaf(2), leaf(3)], BitmapOperation::Union, 8007);
        let intersection = BitmapCombination::new(
            vec![Box::new(union), leaf(5)],
            BitmapOperation::Intersection,
            8007,
        );
        let mut exclude = BitmapCombination::new(
            vec![Box::new(intersection), leaf(7)],
            BitmapOperation::Exclude,
            8007,
        );
        let mut count = 0;
        while exclude.doc() != TERMINATED {
            let base = exclude.doc() / BLOCK_WINDOW * BLOCK_WINDOW;
            let mut mask = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
            exclude.fill_bitset_block(base, &mut mask);
            count += mask.iter().map(|word| word.len()).sum::<u32>();
        }
        assert_eq!(
            count,
            (0..8007)
                .filter(|doc| (doc % 2 == 0 || doc % 3 == 0) && doc % 5 == 0 && doc % 7 != 0)
                .count() as u32
        );
    }
}
