use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::{Rng, SeedableRng};

use crate::docset::{DocSet, TERMINATED};
use crate::indexer::NoMergePolicy;
use crate::query::term_query::TermScorer;
use crate::query::{EnableScoring, TermQuery, Weight};
use crate::schema::{Field, IndexRecordOption, Schema, STRING, TEXT};
use crate::{Index, IndexWriter, Searcher, SegmentReader, Term};

const SEGMENT_DOCS: u32 = 20_000;
const NUM_SEGMENTS: u32 = 2;
const NUM_DOCS: u32 = SEGMENT_DOCS * NUM_SEGMENTS;

const DENSE_WORD: &str = "dense";
const HUNDRED_WORD: &str = "hundred";
const HUNDRED_DOCS_PER_SEGMENT: usize = 100;
const DENSITY_WORDS: [(&str, f64); 5] = [
    (DENSE_WORD, 0.6),
    ("common", 0.2),
    ("medium", 0.05),
    ("rare", 0.002),
    ("rarer", 0.001),
];
const ABSENT_WORD: &str = "absent";

#[derive(Clone, Copy, Debug, PartialEq)]
enum DeletePattern {
    None,
    ScatteredSparse,
    TwoRuns,
    ManyRuns,
    ScatteredHeavy,
}

impl DeletePattern {
    const ALL: [DeletePattern; 5] = [
        DeletePattern::None,
        DeletePattern::ScatteredSparse,
        DeletePattern::TwoRuns,
        DeletePattern::ManyRuns,
        DeletePattern::ScatteredHeavy,
    ];

    fn is_clustered(self) -> bool {
        matches!(self, DeletePattern::TwoRuns | DeletePattern::ManyRuns)
    }

    /// Global ids to delete; ids `[k * SEGMENT_DOCS, (k + 1) * SEGMENT_DOCS)` live in segment `k`.
    fn ids(self, rng: &mut StdRng) -> Vec<u32> {
        let scattered = |rng: &mut StdRng, fraction: f64| {
            let amount = (NUM_DOCS as f64 * fraction).round() as usize;
            sample(rng, NUM_DOCS as usize, amount)
                .into_iter()
                .map(|id| id as u32)
                .collect()
        };
        match self {
            DeletePattern::None => Vec::new(),
            DeletePattern::ScatteredSparse => scattered(rng, 0.002),
            DeletePattern::TwoRuns => (5_000..5_200)
                .chain(SEGMENT_DOCS + 11_000..SEGMENT_DOCS + 11_050)
                .collect(),
            DeletePattern::ManyRuns => (0..200u32)
                .flat_map(|run| run * 200..run * 200 + 100)
                .collect(),
            DeletePattern::ScatteredHeavy => scattered(rng, 0.3),
        }
    }
}

struct Fixture {
    searcher: Searcher,
    text_field: Field,
    num_deleted: u32,
}

fn build_index(seed: u64, pattern: DeletePattern) -> crate::Result<Fixture> {
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT);
    let id_field = schema_builder.add_text_field("id", STRING);
    let index = Index::create_in_ram(schema_builder.build());
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let mut rng = StdRng::seed_from_u64(seed);
    for segment in 0..NUM_SEGMENTS {
        let hundred_docs =
            sample(&mut rng, SEGMENT_DOCS as usize, HUNDRED_DOCS_PER_SEGMENT).into_vec();
        for local in 0..SEGMENT_DOCS {
            let mut words: Vec<&str> = DENSITY_WORDS
                .iter()
                .filter(|&&(_, density)| rng.random_bool(density))
                .map(|&(word, _)| word)
                .collect();
            if hundred_docs.contains(&(local as usize)) {
                words.push(HUNDRED_WORD);
            }
            words.extend(std::iter::repeat_n("pad", rng.random_range(1..4)));
            let id = segment * SEGMENT_DOCS + local;
            writer.add_document(doc!(text_field => words.join(" "), id_field => id.to_string()))?;
        }
        writer.commit()?;
    }
    let deleted_ids = pattern.ids(&mut rng);
    for &id in &deleted_ids {
        writer.delete_term(Term::from_field_text(id_field, &id.to_string()));
    }
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    assert_eq!(searcher.segment_readers().len(), NUM_SEGMENTS as usize);
    for reader in searcher.segment_readers() {
        assert_eq!(reader.max_doc(), SEGMENT_DOCS);
    }
    Ok(Fixture {
        searcher,
        text_field,
        num_deleted: deleted_ids.len() as u32,
    })
}

fn drained_live_count(weight: &dyn Weight, reader: &SegmentReader) -> crate::Result<u32> {
    let mut scorer = weight.scorer(reader, 1.0)?;
    let alive_bitset = reader.alive_bitset();
    let mut live = 0u32;
    let mut doc = scorer.doc();
    while doc != TERMINATED {
        if alive_bitset.is_none_or(|alive| alive.is_alive(doc)) {
            live += 1;
        }
        doc = scorer.advance();
    }
    Ok(live)
}

fn gate_uses_block_loop(scorer: &TermScorer, reader: &SegmentReader) -> bool {
    TermScorer::count_alive_uses_block_loop(scorer.doc_freq(), reader.max_doc())
}

#[test]
fn test_term_count_matches_drained_live_count() -> crate::Result<()> {
    let words: Vec<&str> = DENSITY_WORDS
        .iter()
        .map(|&(word, _)| word)
        .chain([HUNDRED_WORD, ABSENT_WORD])
        .collect();
    for seed in 0..3u64 {
        for pattern in DeletePattern::ALL {
            let fixture = build_index(seed, pattern)?;
            let searcher = &fixture.searcher;
            let num_deleted: u32 = searcher
                .segment_readers()
                .iter()
                .map(SegmentReader::num_deleted_docs)
                .sum();
            assert_eq!(num_deleted, fixture.num_deleted, "{pattern:?}");
            for scoring in [
                EnableScoring::enabled_from_searcher(searcher),
                EnableScoring::disabled_from_searcher(searcher),
            ] {
                let scoring_enabled = scoring.is_scoring_enabled();
                for &word in &words {
                    let term = Term::from_field_text(fixture.text_field, word);
                    let weight = TermQuery::new(term, IndexRecordOption::WithFreqs)
                        .specialized_weight(scoring)?;
                    for reader in searcher.segment_readers() {
                        let context = format!(
                            "seed={seed} pattern={pattern:?} scoring={scoring_enabled} \
                             word={word} segment={:?}",
                            reader.segment_id()
                        );
                        assert_eq!(
                            weight.count(reader)?,
                            drained_live_count(&weight, reader)?,
                            "{context}"
                        );
                        let Some(term_scorer) = weight.term_scorer_for_test(reader, 1.0)? else {
                            assert_eq!(word, ABSENT_WORD, "{context}");
                            continue;
                        };
                        if word == DENSE_WORD && pattern.is_clustered() {
                            assert!(gate_uses_block_loop(&term_scorer, reader), "{context}");
                        }
                        if word == HUNDRED_WORD {
                            assert_eq!(
                                term_scorer.doc_freq(),
                                HUNDRED_DOCS_PER_SEGMENT as u32,
                                "{context}"
                            );
                            assert!(!gate_uses_block_loop(&term_scorer, reader), "{context}");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
