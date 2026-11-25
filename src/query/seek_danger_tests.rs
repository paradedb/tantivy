//! Shared contract tests for query-side `seek_danger` implementations.
//!
//! When adding an override, add its query/scorer to `Kind`, `IMPLEMENTATIONS`, and
//! `Expr::query` (or `scorer`). Test it both alone and nested: a miss may leave a
//! child invalid even when its parent returns Found.
//!
//! Fork-only overrides that are not yet wired into the expression generator are
//! listed in `FORK_INVENTORY` so the inventory guard stays accurate.

use std::collections::BTreeMap;
use std::ops::Bound;
use std::path::Path;
use std::sync::OnceLock;

use common::TinySet;
use proptest::prelude::*;

use super::*;
use crate::docset::{fill_all_candidates_mask, SeekDangerResult};
use crate::schema::{Field, IndexRecordOption, Schema, FAST, TEXT};
use crate::{DocId, DocSet, Index, Searcher, Term, TERMINATED};

#[derive(Clone, Copy, Debug)]
enum Kind {
    Intersection,
    BatchFiltered,
    BufferedUnion,
    Boost,
    ConstScore,
    RequiredOptional,
    Exclude,
    BatchExclude,
    Phrase,
    PhrasePrefix,
    FastFieldRange,
    PredicateColumn,
    CachedDictPredicate,
    ScorerWrapper,
}

// One entry per override, not merely per source file. The inventory test below
// also catches a second override added to an already covered file.
// An empty path denotes a query using the default implementation.
macro_rules! implementations {
    ($(($path:literal, $kind:ident, $test:ident)),* $(,)?) => {
        const IMPLEMENTATIONS: &[(&str, Kind)] = &[$(($path, Kind::$kind)),*];
        $(proptest! {
            #![proptest_config(ProptestConfig::with_cases(32))]
            #[test]
            fn $test(
                left in expressions(),
                right in expressions(),
                random_targets in prop::collection::vec(0u32..12000, 0..80),
                scoring in any::<bool>(),
                sparse_seed in any::<u64>(),
            ) {
                let mut targets = random_targets;
                targets.extend([0, 1, 4095, 4096, 4097, 6000, 10200, 11000, 11010, 11101, 11102, TERMINATED]);
                targets.sort_unstable();
                targets.dedup();
                // Exercise the implementation with simple, nonempty children
                // as well as with randomly nested queries.
                for (left, right) in [(Expr::Term(2, false), Expr::Term(2, true)), (left, right)] {
                    let expr = Expr::Node(Kind::$kind, Box::new(left), Box::new(right));
                    check(|| scorer(&expr, scoring), &targets);
                    check_batch(|| scorer(&expr, scoring), &targets, sparse_seed);
                }
            }
        })*
    };
}

implementations! {
    ("intersection.rs", Intersection, query_seek_danger_intersection),
    ("intersection.rs", BatchFiltered, query_seek_danger_batch_filtered),
    ("", BufferedUnion, query_seek_danger_buffered_union),
    ("boost_query.rs", Boost, query_seek_danger_boost),
    ("const_score_query.rs", ConstScore, query_seek_danger_const_score),
    ("reqopt_scorer.rs", RequiredOptional, query_seek_danger_required_optional),
    ("exclude.rs", Exclude, query_seek_danger_exclude),
    ("exclude.rs", BatchExclude, query_seek_danger_batch_exclude),
    ("phrase_query/phrase_scorer.rs", Phrase, query_seek_danger_phrase),
    ("phrase_prefix_query/phrase_prefix_scorer.rs", PhrasePrefix, query_seek_danger_phrase_prefix),
    ("range_query/fast_field_range_doc_set.rs", FastFieldRange, query_seek_danger_fast_field_range),
    ("term_set_query/term_predicate.rs", PredicateColumn, query_seek_danger_predicate_column),
    ("term_set_query/term_predicate.rs", CachedDictPredicate, query_seek_danger_cached_dict_predicate),
    ("disjunction.rs", ScorerWrapper, query_seek_danger_scorer_wrapper),
}

// Overrides not yet covered by the expression generator above.
const FORK_INVENTORY: &[(&str, usize)] = &[("query/term_set_query/term_set_weight.rs", 1)];

#[derive(Clone, Debug)]
enum Expr {
    Term(usize, bool),
    Node(Kind, Box<Expr>, Box<Expr>),
}

#[derive(Debug)]
struct ModuloPredicate;

impl TermPredicate for ModuloPredicate {
    fn filter_bytes(&self, data: &[u8], offsets: &[u32], out: &mut [common::TinySet]) {
        for i in 0..offsets.len().saturating_sub(1) {
            let slice = &data[offsets[i] as usize..offsets[i + 1] as usize];
            if slice.last().is_some_and(|b| b % 2 == 0) {
                out[i / 64].insert_mut((i % 64) as u32);
            }
        }
    }

    fn filter_monotonic_u64s(
        &self,
        _col_type: columnar::ColumnType,
        values: &[u64],
        out: &mut [common::TinySet],
    ) {
        for (i, &v) in values.iter().enumerate() {
            if (3..8).contains(&v) {
                out[i / 64].insert_mut((i % 64) as u32);
            }
        }
    }

    fn probe_cost(&self) -> crate::ProbeCost {
        crate::ProbeCost::new(crate::ProbeSetupCost(300), crate::ProbePerDocCost(10))
    }
}

impl Expr {
    fn query(&self, fields: &[Field; 3], number: Field, dict_str: Field) -> Box<dyn Query> {
        let term =
            |field, order| Term::from_field_text(field, if order { "order" } else { "service" });
        let Expr::Node(kind, left, right) = self else {
            let Expr::Term(field, order) = self else {
                unreachable!()
            };
            return Box::new(TermQuery::new(
                term(fields[*field], *order),
                IndexRecordOption::WithFreqs,
            ));
        };
        let left = left.query(fields, number, dict_str);
        let right = right.query(fields, number, dict_str);
        match kind {
            Kind::Intersection => Box::new(BooleanQuery::intersection(vec![left, right])),
            Kind::BatchFiltered => Box::new(BooleanQuery::intersection(vec![
                left,
                Box::new(TermPredicateQuery::new(
                    number,
                    std::sync::Arc::new(ModuloPredicate),
                )),
            ])),
            Kind::BufferedUnion => Box::new(DisjunctionMaxQuery::with_tie_breaker(
                vec![left, right],
                0.3,
            )),
            Kind::Boost => Box::new(BoostQuery::new(left, 2.0)),
            Kind::ConstScore => Box::new(ConstScoreQuery::new(left, 3.0)),
            Kind::RequiredOptional => Box::new(BooleanQuery::new(vec![
                (Occur::Must, left),
                (Occur::Should, right),
            ])),
            Kind::Exclude => Box::new(BooleanQuery::new(vec![
                (Occur::Must, left),
                (Occur::MustNot, right),
            ])),
            Kind::BatchExclude => Box::new(BooleanQuery::new(vec![
                (Occur::Must, left),
                (
                    Occur::MustNot,
                    Box::new(TermPredicateQuery::new(
                        number,
                        std::sync::Arc::new(ModuloPredicate),
                    )),
                ),
            ])),
            Kind::Phrase => Box::new(PhraseQuery::new(vec![
                term(fields[2], false),
                term(fields[2], true),
            ])),
            Kind::PhrasePrefix => Box::new(PhrasePrefixQuery::new(vec![
                term(fields[2], false),
                Term::from_field_text(fields[2], "ord"),
            ])),
            Kind::FastFieldRange => Box::new(RangeQuery::new(
                Bound::Included(Term::from_field_u64(number, 3)),
                Bound::Excluded(Term::from_field_u64(number, 8)),
            )),
            Kind::PredicateColumn => Box::new(TermPredicateQuery::new(
                number,
                std::sync::Arc::new(ModuloPredicate),
            )),
            Kind::CachedDictPredicate => {
                // Intersect with the selective `fields[1]: "order"` term (2 matches out of 11,102)
                // so `BooleanWeight` routes `dict_str` onto `CachedDictPredicateDocSet`.
                let rare = Box::new(TermQuery::new(
                    term(fields[1], true),
                    IndexRecordOption::WithFreqs,
                ));
                let pred = Box::new(
                    TermPredicateQuery::new(dict_str, std::sync::Arc::new(ModuloPredicate))
                        .with_exact_postings(ExactPostings::Unavailable),
                );
                Box::new(BooleanQuery::intersection(vec![rare, pred]))
            }
            // ScorerWrapper is private to Disjunction and its override is not
            // called by Disjunction's normal traversal. Exercise it directly.
            Kind::ScorerWrapper => left,
        }
    }
}

fn expressions() -> impl Strategy<Value = Expr> {
    (0usize..3, any::<bool>())
        .prop_map(|(field, order)| Expr::Term(field, order))
        .prop_recursive(3, 16, 2, |inner| {
            // ScorerWrapper is exercised directly as a root, not a Query.
            let kinds: Vec<_> = IMPLEMENTATIONS
                .iter()
                .map(|&(_, kind)| kind)
                .filter(|kind| !matches!(kind, Kind::ScorerWrapper))
                .collect();
            (prop::sample::select(kinds), inner.clone(), inner)
                .prop_map(|(kind, left, right)| Expr::Node(kind, Box::new(left), Box::new(right)))
        })
}

struct Fixture {
    searcher: Searcher,
    fields: [Field; 3],
    number: Field,
    dict_str: Field,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let mut schema = Schema::builder();
        let fields = [
            schema.add_text_field("content", TEXT),
            schema.add_text_field("title", TEXT),
            schema.add_text_field("mixed", TEXT),
        ];
        let number = schema.add_u64_field("number", FAST);
        let dict_str = schema.add_text_field("dict_str", crate::schema::STRING | FAST);
        let index = Index::create_in_ram(schema.build());
        let mut writer = index.writer_with_num_threads(1, 50_000_000).unwrap();
        for id in 0..11102u64 {
            // Preserve the issue #3086 layout in content/title. A third field
            // supplies dense matches, phrase misses, and prefix expansions.
            let (content, title) = match id {
                0 | 10200 | 11101 => ("service order", ""),
                6000 | 11000 => ("nothing here", "service order"),
                11010 => ("purchase order", ""),
                _ => ("service team meeting", ""),
            };
            let mixed = [
                "service order",
                "order service",
                "service ordering",
                "service",
                "order",
                "nothing",
            ][(id % 6) as usize];
            let s = format!("entry-{id:06}-padding-block");
            writer
                .add_document(doc!(
                    fields[0] => content,
                    fields[1] => title,
                    fields[2] => mixed,
                    number => id % 17,
                    dict_str => s,
                ))
                .unwrap();
        }
        writer.commit().unwrap();
        let searcher = index.reader().unwrap().searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        Fixture {
            searcher,
            fields,
            number,
            dict_str,
        }
    })
}

fn scorer(expr: &Expr, scoring: bool) -> Box<dyn Scorer> {
    let fixture = fixture();
    let query = expr.query(&fixture.fields, fixture.number, fixture.dict_str);
    let enable = if scoring {
        EnableScoring::enabled_from_searcher(&fixture.searcher)
    } else {
        EnableScoring::disabled_from_searcher(&fixture.searcher)
    };
    let scorer = query
        .weight(enable)
        .unwrap()
        .scorer(fixture.searcher.segment_reader(0), 1.0)
        .unwrap();
    if matches!(expr, Expr::Node(Kind::ScorerWrapper, _, _)) {
        Box::new(ConstScorer::new(
            super::disjunction::seek_danger_test_wrapper(scorer),
            1.0,
        ))
    } else {
        scorer
    }
}

/// The oracle uses only advance. After a miss, the tested scorer receives ONLY
/// seek_danger, with strictly increasing targets; even doc() is forbidden.
fn check(mut make: impl FnMut() -> Box<dyn Scorer>, targets: &[DocId]) {
    let mut reference = make();
    let mut expected = Vec::new();
    while reference.doc() != TERMINATED {
        expected.push((reference.doc(), reference.score()));
        reference.advance();
    }
    let mut tested = make();
    let mut lower_bound = tested.doc();
    let mut previous_target = None;
    for &requested_target in targets {
        // Respect the returned lower bound, just like an intersection driver.
        // In particular, do not probe behind a scorer that has sought ahead.
        let target = requested_target.max(lower_bound);
        if previous_target == Some(target) {
            continue;
        }
        previous_target = Some(target);
        let next = expected.partition_point(|&(doc, _)| doc < target);
        let next_doc = expected.get(next).map_or(TERMINATED, |&(doc, _)| doc);
        // Exercise both forwarding implementations in docset.rs as well:
        // &mut dyn DocSet -> Box<dyn Scorer> -> concrete scorer.
        let mut borrowed: &mut dyn DocSet = &mut tested;
        match DocSet::seek_danger(&mut borrowed, target) {
            SeekDangerResult::Found => {
                assert_ne!(target, TERMINATED);
                assert_eq!(next_doc, target, "false positive at {target}");
                assert_eq!(tested.doc(), target);
                let score = tested.score();
                let expected_score = expected[next].1;
                assert!(
                    (score - expected_score).abs() <= 1e-5 * expected_score.abs().max(1.0),
                    "score at {target}: {score} != {expected_score}"
                );
            }
            SeekDangerResult::SeekLowerBound(bound) => {
                assert!(
                    target == TERMINATED || bound > target,
                    "non-increasing bound {bound} at {target}"
                );
                assert!(bound <= next_doc, "bound {bound} skips match {next_doc}");
                assert!(
                    next_doc != target || target == TERMINATED,
                    "missed match {target}"
                );
                lower_bound = bound;
            }
        }
    }
}

/// Oracle for `seek_danger_batch`: verifies in-place AND semantics (both all-active and sparse
/// pre-filtered masks), skip-horizon bounds, empty batches, multi-chunk monotonic probing, and
/// probing after advancing into a buffered scorer.
fn check_batch(mut make: impl FnMut() -> Box<dyn Scorer>, targets: &[DocId], sparse_seed: u64) {
    let mut reference = make();
    let mut expected = Vec::new();
    while reference.doc() != TERMINATED {
        expected.push(reference.doc());
        reference.advance();
    }
    let valid_targets: Vec<DocId> = targets
        .iter()
        .copied()
        .filter(|&d| d < TERMINATED)
        .collect();

    // Test both all-active and sparse pre-filtered masks across varying chunk sizes,
    // and also test a partially-advanced scorer (straddling any internal buffer).
    for sparse in [false, true] {
        for advance_first in [false, true] {
            let mut tested = make();
            assert_eq!(tested.seek_danger_batch(&[], &mut []), 0);
            if advance_first && tested.doc() != TERMINATED {
                tested.advance();
            }
            let mut lower_bound = tested.doc();
            let mut probe_docs = valid_targets.clone();
            if lower_bound < TERMINATED {
                probe_docs.push(lower_bound);
                if lower_bound + 1 < TERMINATED {
                    probe_docs.push(lower_bound + 1);
                }
                probe_docs.sort_unstable();
                probe_docs.dedup();
            }

            let chunk_sizes = if advance_first {
                &[probe_docs.len().max(1)][..]
            } else {
                &[1usize, 7, 64, 65, 130][..]
            };
            let mut pos = 0usize;
            let mut chunk_idx = 0usize;
            while pos < probe_docs.len() {
                let size = chunk_sizes[chunk_idx % chunk_sizes.len()];
                chunk_idx += 1;
                let end = (pos + size).min(probe_docs.len());
                let chunk: Vec<DocId> = probe_docs[pos..end]
                    .iter()
                    .copied()
                    .filter(|&d| d >= lower_bound)
                    .collect();
                pos = end;
                if chunk.is_empty() {
                    continue;
                }
                let words = chunk.len().div_ceil(64);
                let mut mask = vec![TinySet::EMPTY; words];
                if sparse {
                    for i in 0..chunk.len() {
                        let hash = sparse_seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add((chunk_idx as u64) << 16)
                            .wrapping_add(i as u64);
                        if (hash >> 3) & 1 == 0 {
                            mask[i / 64].insert_mut((i % 64) as u32);
                        }
                    }
                } else {
                    fill_all_candidates_mask(chunk.len(), &mut mask);
                }
                let initial_mask = mask.clone();
                let mut borrowed: &mut dyn DocSet = &mut tested;
                let next_bound = DocSet::seek_danger_batch(&mut borrowed, &chunk, &mut mask);

                for (i, &doc) in chunk.iter().enumerate() {
                    let was_active = initial_mask[i / 64].contains((i % 64) as u32);
                    let got = mask[i / 64].contains((i % 64) as u32);
                    let want = was_active && expected.binary_search(&doc).is_ok();
                    assert_eq!(
                        got, want,
                        "seek_danger_batch mismatch at doc {doc} (sparse={sparse}, \
                         advance_first={advance_first})"
                    );
                }
                let rem = chunk.len() % 64;
                if rem > 0 {
                    assert_eq!(
                        mask[words - 1].into_u64() >> rem,
                        0,
                        "trailing bits beyond chunk.len() must remain 0"
                    );
                }
                let last = *chunk.last().unwrap();
                assert!(
                    next_bound > last || next_bound == TERMINATED,
                    "non-increasing next_bound {next_bound} after {last}"
                );
                let next_idx = expected.partition_point(|&d| d <= last);
                let next_expected = expected.get(next_idx).copied().unwrap_or(TERMINATED);
                assert!(
                    next_bound <= next_expected,
                    "next_bound {next_bound} skips match {next_expected} after {last}"
                );
                lower_bound = lower_bound.max(next_bound);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn buffered_union_invalid_child_regression(offset in 0u32..1000, extra_gap in 0u32..1000) {
        // Same state transitions as #3086, without indexing 11k documents per
        // case. Both seeks are beyond the buffered horizon. The first child
        // misses the second target and leaves a one-term-only doc in the new
        // horizon; the second child finds the target.
        let first = offset + 6000;
        let second = first + 5000 + extra_gap;
        let false_match = second + 10;
        let last = second + 101;
        check(|| {
            let intersection = |a, b| -> Box<dyn Scorer> {
                Box::new(Intersection::new(vec![
                    ConstScorer::new(VecDocSet::from(a), 1.0),
                    ConstScorer::new(VecDocSet::from(b), 1.0),
                ], last + 3))
            };
            Box::new(BufferedUnionScorer::build(vec![
                intersection(vec![offset, first + 4200, false_match, last], vec![offset, first + 4200, last, last + 1, last + 2]),
                intersection(vec![first, second], vec![first, second]),
            ], SumCombiner::default, last + 3))
        }, &[first, second, false_match, last, TERMINATED]);
    }
}

#[test]
fn seek_danger_implementation_inventory() {
    fn visit(root: &Path, dir: &Path, pattern: &regex::Regex, found: &mut BTreeMap<String, usize>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(root, &path, pattern, found);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = std::fs::read_to_string(&path).unwrap();
                let count = pattern.find_iter(&source).count();
                if count != 0 {
                    found.insert(
                        path.strip_prefix(root)
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .replace('\\', "/"),
                        count,
                    );
                }
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found = BTreeMap::new();
    let pattern = regex::Regex::new(r"fn\s+seek_danger\s*\(").unwrap();
    visit(&root, &root, &pattern, &mut found);
    let mut covered = BTreeMap::new();
    for &(path, _) in IMPLEMENTATIONS {
        if !path.is_empty() {
            *covered.entry(format!("query/{path}")).or_insert(0) += 1;
        }
    }
    for &(path, count) in FORK_INVENTORY {
        covered.insert(path.to_string(), count);
    }
    // Trait default (used by TermScorer) and the Box / &mut / UnpositionedScorer forwarding impls.
    covered.insert("docset.rs".to_string(), 3);
    covered.insert("query/scorer.rs".to_string(), 1);
    // Test-only spies for intersection unit tests.
    *covered
        .entry("query/intersection.rs".to_string())
        .or_insert(0) += 2;
    assert_eq!(
        found, covered,
        "Register new seek_danger implementations in seek_danger_tests.rs and add their \
         query/scorer to the generator"
    );
}

#[test]
fn seek_danger_batch_implementation_inventory() {
    fn visit(root: &Path, dir: &Path, pattern: &regex::Regex, found: &mut BTreeMap<String, usize>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(root, &path, pattern, found);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = std::fs::read_to_string(&path).unwrap();
                let count = pattern.find_iter(&source).count();
                if count != 0 {
                    found.insert(
                        path.strip_prefix(root)
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .replace('\\', "/"),
                        count,
                    );
                }
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found = BTreeMap::new();
    let pattern = regex::Regex::new(r"fn\s+seek_danger_batch\s*\(").unwrap();
    visit(&root, &root, &pattern, &mut found);
    let expected: BTreeMap<String, usize> = [
        ("docset.rs", 3),
        ("query/boost_query.rs", 1),
        ("query/const_score_query.rs", 1),
        ("query/exclude.rs", 2),
        ("query/intersection.rs", 2),
        ("query/reqopt_scorer.rs", 1),
        // Forwarding wrapper `UnpositionedScorer`.
        ("query/scorer.rs", 1),
        ("query/term_set_query/term_predicate.rs", 2),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    assert_eq!(
        found, expected,
        "Register new seek_danger_batch implementations in seek_danger_tests.rs"
    );
}
