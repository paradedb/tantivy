//! Combines query matches one 1,024-document window at a time, keeping the result
//! as bits so count collectors can count set bits without visiting every document.
//! Each window has sixteen 64-bit words (`TinySet`s). For example, in the window
//! `[1024, 2048)`, bit 3 of the first word represents document 1027 in this segment.
//!
//! Every child supplies its matches for the current window through
//! `fill_bitset_block`. How much work that takes depends on the child:
//!
//! - Dense + dense: both children supply bitmap words directly. We combine those words with bitwise
//!   OR for a union, or AND for an intersection.
//! - Sparse + dense: the sparse child reads its ordinary postings and sets bits for the matching
//!   document IDs in this window. The dense child supplies its bitmap words directly. We then use
//!   the same OR or AND operations. Only the current window is converted; we do not build a bitmap
//!   for the whole sparse list.
//! - Sparse + sparse: both children could fill masks from their document IDs, but Boolean query
//!   selection normally keeps these queries on ordinary postings scorers. This scorer is chosen
//!   when at least one child can supply a bitmap.
//!
//! As a small example, write the set bits as document IDs: A = {1, 2, 4} and
//! B = {2, 3}. A OR B produces {1, 2, 3, 4}; A AND B produces {2}; A NOT B
//! produces {1, 4}. Exclusion starts with the first child's mask and clears bits
//! found in any of the remaining children. Children can also be nested queries
//! or filters; they use the same window interface.
//!
//! A union skips children whose next match is beyond this window and removes
//! exhausted children after including their last bits. An intersection tries
//! children with lower estimated cost first, breaking ties by estimated match
//! count. Intersections and exclusions stop filling a window once no bits remain.
//! The children's next document IDs tell us which window to visit next.
//!
//! Query selection can still choose ordinary, candidate-driven evaluation. For
//! example, a very rare term AND a dense term can probe the dense bitmap only at
//! the rare term's document IDs. Likewise, `selective -"of the"` keeps
//! candidate-driven phrase evaluation to avoid scanning phrase matches throughout
//! every window.
//!
//! The result supports both ordinary document iteration and passing a whole mask
//! to another bitmap-aware scorer or collector. The optimized collection path is
//! currently used for counts; these masks describe matches, with document
//! visibility checks handled downstream.

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
        mut children: Vec<Box<dyn Scorer>>,
        operation: BitmapOperation,
        num_docs: u32,
    ) -> Self {
        assert!(!children.is_empty());
        if matches!(operation, BitmapOperation::Intersection) {
            children.sort_by_key(|child| (child.cost(), child.size_hint()));
        }
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
            BitmapOperation::Union => self
                .children
                .iter()
                .map(|child| child.doc())
                .min()
                .unwrap_or(TERMINATED),
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
            if matches!(self.operation, BitmapOperation::Union) {
                let horizon = self.base.saturating_add(BLOCK_WINDOW).min(TERMINATED);
                self.children.retain_mut(|child| {
                    if child.doc() < horizon {
                        child.fill_bitset_block(self.base, &mut self.mask);
                    }
                    child.doc() != TERMINATED
                });
                if self.position(target) {
                    return self.doc;
                }
                continue;
            }
            self.children[0].fill_bitset_block(self.base, &mut self.mask);
            let mut empty = self.mask.iter().all(|word| word.is_empty());
            for child in &mut self.children[1..] {
                if empty {
                    break;
                }
                let mut other = [TinySet::EMPTY; BLOCK_NUM_TINYBITSETS];
                child.fill_bitset_block(self.base, &mut other);
                match self.operation {
                    BitmapOperation::Union => unreachable!(),
                    BitmapOperation::Intersection => {
                        empty = super::intersection::and_blocks_and_return_is_empty(
                            &mut self.mask,
                            &other,
                        );
                    }
                    BitmapOperation::Exclude => {
                        empty = true;
                        for (word, other) in self.mask.iter_mut().zip(other) {
                            let bits = word.into_u64() & !other.into_u64();
                            *word = TinySet::deserialize(bits.to_le_bytes());
                            empty &= bits == 0;
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

    struct TrackedScorer {
        docs: VecDocSet,
        cost: u64,
        fills: std::sync::Arc<std::sync::Mutex<Vec<DocId>>>,
    }

    impl DocSet for TrackedScorer {
        fn advance(&mut self) -> DocId {
            self.docs.advance()
        }
        fn seek(&mut self, target: DocId) -> DocId {
            self.docs.seek(target)
        }
        fn doc(&self) -> DocId {
            self.docs.doc()
        }
        fn size_hint(&self) -> u32 {
            self.docs.size_hint()
        }
        fn cost(&self) -> u64 {
            self.cost
        }
        fn fill_bitset_block(&mut self, base: DocId, mask: &mut Block) -> DocId {
            self.fills.lock().unwrap().push(base);
            self.docs.fill_bitset_block(base, mask)
        }
    }

    impl Scorer for TrackedScorer {
        fn score(&mut self) -> Score {
            1.0
        }
    }

    #[test]
    fn union_retires_children_after_their_last_mask_and_skips_future_windows() {
        let fills: Vec<_> = (0..3)
            .map(|_| std::sync::Arc::new(std::sync::Mutex::new(Vec::new())))
            .collect();
        let children = [vec![1], vec![2, 1025, 2049], vec![4097]]
            .into_iter()
            .enumerate()
            .map(|(i, docs)| {
                Box::new(TrackedScorer {
                    docs: VecDocSet::from(docs),
                    cost: 1,
                    fills: fills[i].clone(),
                }) as Box<dyn Scorer>
            })
            .collect();
        let mut scorer = BitmapCombination::new(children, BitmapOperation::Union, 8007);
        assert_eq!(scorer.children.len(), 2);
        for doc in [1, 2, 1025, 2049, 4097] {
            assert_eq!(scorer.doc(), doc);
            scorer.advance();
        }
        assert!(scorer.children.is_empty());
        assert_eq!(scorer.advance(), TERMINATED);
        assert_eq!(*fills[0].lock().unwrap(), vec![0]);
        assert_eq!(*fills[1].lock().unwrap(), vec![0, 1024, 2048]);
        assert_eq!(*fills[2].lock().unwrap(), vec![4096]);
    }

    #[test]
    fn empty_windows_skip_children_and_preserve_progress() {
        for operation in [BitmapOperation::Intersection, BitmapOperation::Exclude] {
            let intersection = matches!(operation, BitmapOperation::Intersection);
            let documents = [
                vec![1, 1025, 2049, 4097],
                if intersection {
                    vec![2, 1026, 2049, 4097]
                } else {
                    vec![1, 1025, 4097]
                },
                if intersection {
                    vec![3, 1027, 2049, 4097]
                } else {
                    vec![3, 1027, 4097]
                },
                if intersection {
                    vec![4, 1028, 2049, 4097]
                } else {
                    vec![4, 1028, 4097]
                },
            ];
            let fills: Vec<_> = (0..4)
                .map(|_| std::sync::Arc::new(std::sync::Mutex::new(Vec::new())))
                .collect();
            let mut children: Vec<Box<dyn Scorer>> = documents
                .into_iter()
                .enumerate()
                .map(|(i, docs)| {
                    Box::new(TrackedScorer {
                        docs: VecDocSet::from(docs),
                        cost: i as u64,
                        fills: fills[i].clone(),
                    }) as Box<dyn Scorer>
                })
                .collect();
            if intersection {
                children.reverse();
            }
            let mut scorer = BitmapCombination::new(children, operation, 8007);
            assert_eq!(scorer.doc(), 2049);
            assert_eq!(*fills[2].lock().unwrap(), vec![2048]);
            assert_eq!(*fills[3].lock().unwrap(), vec![2048]);
            assert_eq!(
                scorer.advance(),
                if intersection { 4097 } else { TERMINATED }
            );
            assert_eq!(scorer.advance(), TERMINATED);
            assert_eq!(*fills[0].lock().unwrap(), vec![0, 1024, 2048, 4096]);
            assert_eq!(
                *fills[3].lock().unwrap(),
                if intersection {
                    vec![2048, 4096]
                } else {
                    vec![2048]
                }
            );
        }
    }

    #[test]
    fn empty_first_child_skips_the_remaining_children() {
        let fills = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut scorer = BitmapCombination::new(
            vec![
                Box::new(TrackedScorer {
                    docs: VecDocSet::from(vec![1, 4097]),
                    cost: 0,
                    fills: Default::default(),
                }),
                Box::new(TrackedScorer {
                    docs: VecDocSet::from(vec![2049, 4097]),
                    cost: 1,
                    fills: fills.clone(),
                }),
            ],
            BitmapOperation::Intersection,
            8007,
        );
        assert_eq!(scorer.doc(), 4097);
        assert_eq!(*fills.lock().unwrap(), vec![4096]);
        assert_eq!(scorer.advance(), TERMINATED);
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
