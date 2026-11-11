//! Metric and dtype handling stays exhaustive: no `==`, `!=` or `matches!` on a `Metric` or
//! `VectorDType` variant in `src/vector` or `src/schema`. Each decision goes through a named fact,
//! such as `Metric::scores_by_distance`, or an exhaustive `match`, so a new variant must decide
//! everywhere it matters.

use std::path::{Path, PathBuf};

const ENUMS: [&str; 2] = ["Metric", "VectorDType"];

/// The enums' own definitions and this scan.
const EXEMPT: [&str; 2] = [
    "src/schema/vector_options.rs",
    "src/vector/exhaustive_metric_tests.rs",
];

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Whether a `::`-separated path names a variant of one of the enums.
fn names_variant(path: &str) -> bool {
    let segments: Vec<&str> = path.split("::").collect();
    segments
        .windows(2)
        .any(|pair| ENUMS.contains(&pair[0]) && !pair[1].is_empty())
}

/// The path that starts at `at`, after any whitespace.
fn path_after(text: &str, at: usize) -> &str {
    let rest = text[at..].trim_start();
    let end = rest
        .find(|c: char| !(is_ident(c) || c == ':'))
        .unwrap_or(rest.len());
    &rest[..end]
}

/// The path that ends at `at`, before any whitespace.
fn path_before(text: &str, at: usize) -> &str {
    let head = text[..at].trim_end();
    let start = head
        .rfind(|c: char| !(is_ident(c) || c == ':'))
        .map_or(0, |index| index + 1);
    &head[start..]
}

/// The 1-based lines where `text` compares a variant of either enum with `==` or `!=`, or names
/// one inside `matches!`, across line breaks.
fn violations(text: &str) -> Vec<usize> {
    let line = |at: usize| text[..at].matches('\n').count() + 1;
    let mut found = Vec::new();
    for operator in ["==", "!="] {
        for (at, _) in text.match_indices(operator) {
            if names_variant(path_after(text, at + operator.len()))
                || names_variant(path_before(text, at))
            {
                found.push(line(at));
            }
        }
    }
    for (at, _) in text.match_indices("matches!") {
        let Some(open) = text[at..].find('(').map(|offset| at + offset) else {
            continue;
        };
        let mut depth = 0;
        let mut close = text.len();
        for (offset, c) in text[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = open + offset;
                        break;
                    }
                }
                _ => {}
            }
        }
        let body = &text[open..close];
        let names_enum = ENUMS.iter().any(|name| {
            body.match_indices(&format!("{name}::"))
                .any(|(index, _)| !body[..index].ends_with(is_ident))
        });
        if names_enum {
            found.push(line(at));
        }
    }
    found.sort_unstable();
    found
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn metric_and_dtype_decisions_are_exhaustive() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for dir in ["src/vector", "src/schema"] {
        rust_files(&root.join(dir), &mut files);
    }
    files.sort();
    let mut found = Vec::new();
    for path in files {
        let relative = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if EXEMPT.contains(&relative.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        found.extend(
            violations(&text)
                .into_iter()
                .map(|line| format!("{relative}:{line}")),
        );
    }
    assert!(
        found.is_empty(),
        "decide on Metric or VectorDType through a named fact or an exhaustive match:\n{}",
        found.join("\n")
    );
}

#[test]
fn the_scan_finds_each_form() {
    assert_eq!(violations("let l2 = metric == Metric::L2;\n"), [1]);
    assert_eq!(
        violations(
            "fn f(dtype: VectorDType) -> bool {\n    dtype\n        != \
             crate::schema::VectorDType::F32\n}\n"
        ),
        [3]
    );
    assert_eq!(
        violations("let angular = matches!(\n    metric,\n    Metric::Cosine | Metric::Dot\n);\n"),
        [1]
    );
    assert!(violations(
        "let l2 = metric.scores_by_distance();\nmatch dtype {\n    VectorDType::F32 => {}\n}\n"
    )
    .is_empty());
}
