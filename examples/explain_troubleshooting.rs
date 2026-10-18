use std::path::{Path, PathBuf};

use cofferdb::{
    parse_json, Database, DatabaseConfig, Error, ErrorCode, JsonValue, OnConflict,
    QueryEmbeddingInput, ResourceProfile, Result, SearchWeights,
};

const MODEL: &str = "synthetic-demo-2d-v1";

fn demonstrate(path: &Path) -> Result<JsonValue> {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/data");
    let db = Database::create(path, DatabaseConfig::default())?;
    for name in ["markdown", "code"] {
        db.import_manifest(
            fixtures.join(name).join("import_manifest.json"),
            OnConflict::Error,
        )?;
    }
    drop(db);
    let db = Database::open_read_only(path)?;
    let mut runs = Vec::new();
    for (label, collection, text, vector, filter, weights, profile) in [
        (
            "markdown_hybrid",
            "notes",
            "write ahead",
            [1.0, 0.0],
            r#"{"doc.source":"markdown"}"#,
            [0.5, 0.5],
            ResourceProfile::Balanced,
        ),
        (
            "code_lexical_priority",
            "code",
            "page cache",
            [0.0, 1.0],
            r#"{"doc.language":"rust"}"#,
            [0.9, 0.1],
            ResourceProfile::Balanced,
        ),
        (
            "code_vector_priority",
            "code",
            "page cache",
            [0.0, 1.0],
            r#"{"doc.language":"rust"}"#,
            [0.1, 0.9],
            ResourceProfile::Balanced,
        ),
        (
            "code_expanded",
            "code",
            "page cache",
            [0.0, 1.0],
            r#"{"doc.language":"rust"}"#,
            [0.1, 0.9],
            ResourceProfile::Expanded,
        ),
        (
            "code_constrained",
            "code",
            "page cache",
            [0.0, 1.0],
            r#"{"doc.language":"rust"}"#,
            [0.1, 0.9],
            ResourceProfile::Constrained,
        ),
    ] {
        let result = db
            .collection(collection)?
            .search()
            .text(text)
            .query_embedding(QueryEmbeddingInput {
                model_name: Some(MODEL.into()),
                dim: 2,
                vector: vector.to_vec(),
            })
            .filter(parse_json(filter)?)
            .weights(SearchWeights::new(weights[0], weights[1], 0.0)?)
            .resource_profile(profile)
            .top_k(2)
            .execute()?;
        runs.push(JsonValue::Object(
            [
                ("step".into(), JsonValue::String(label.into())),
                ("result".into(), result.to_json()),
            ]
            .into_iter()
            .collect(),
        ));
    }
    Ok(JsonValue::Object([
        ("example_version".into(), JsonValue::String("1".into())),
        ("embedding_notice".into(), JsonValue::String(
            "Hand-authored 2D fixture vectors demonstrate ranking mechanics, not semantic retrieval quality or profile performance.".into(),
        )),
        ("runs".into(), JsonValue::Array(runs)),
    ].into_iter().collect()))
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().filter(|path| !path.is_empty()).ok_or_else(|| {
        Error::new(
            ErrorCode::CliUsage,
            "usage: explain_troubleshooting <new-database.coffer>",
        )
    })?;
    if args.next().is_some() {
        return Err(Error::new(
            ErrorCode::CliUsage,
            "expected one new database path",
        ));
    }
    println!("{}", demonstrate(Path::new(&path))?.to_pretty_string());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "cofferdb-explain-demo-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn ranking_changes_are_explained_and_filtered_after_reopen() {
        let fixture = Fixture::new();
        let path = fixture.0.join("demo.coffer");
        let report = demonstrate(&path).unwrap();
        let wire = report.to_pretty_string();
        assert_eq!(parse_json(&wire).unwrap(), report);
        let runs = report.get("runs").unwrap().as_array().unwrap();
        assert_eq!(runs.len(), 5);
        for (index, expected) in [
            "wal#commit",
            "storage#cache",
            "storage#recovery",
            "storage#recovery",
            "storage#recovery",
        ]
        .into_iter()
        .enumerate()
        {
            let result = runs[index].get("result").unwrap();
            let rows = result.get("results").unwrap().as_array().unwrap();
            assert_eq!(rows.len(), 2);
            assert_eq!(
                rows[0].get("chunk_external_id").and_then(JsonValue::as_str),
                Some(expected)
            );
            let context = result.get("query_context").unwrap();
            assert_eq!(
                context.get("index_source").and_then(JsonValue::as_str),
                Some("persistent_checkpoint")
            );
            let (profile, limit) = match index {
                3 => ("expanded", 500.0),
                4 => ("constrained", 50.0),
                _ => ("balanced", 200.0),
            };
            assert_eq!(
                result.get("profile").and_then(JsonValue::as_str),
                Some(profile)
            );
            let limits = context.get("candidate_limits").unwrap();
            for branch in ["bm25", "vector", "fusion"] {
                assert_eq!(limits.get(branch).and_then(JsonValue::as_f64), Some(limit));
            }
            for row in rows {
                assert_ne!(
                    row.get("chunk_external_id").and_then(JsonValue::as_str),
                    Some("python#cache")
                );
                let explain = row.get("explain").unwrap();
                assert!(explain.get("bm25").is_some());
                let scores = explain.get("scores").unwrap();
                let weights = explain.get("weights").unwrap().get("effective").unwrap();
                let number =
                    |value: &JsonValue, key| value.get(key).and_then(JsonValue::as_f64).unwrap();
                let calculated = number(weights, "bm25") * number(scores, "bm25_norm")
                    + number(weights, "vector") * number(scores, "vector_norm");
                assert!((calculated - number(row, "final_score")).abs() < 1e-12);
                assert_eq!(
                    explain.get("fusion").and_then(JsonValue::as_str),
                    Some("weighted_sum")
                );
                assert_eq!(
                    explain.get("distance_metric").and_then(JsonValue::as_str),
                    Some("cosine")
                );
                assert_eq!(
                    number(explain.get("execution_stats").unwrap(), "distance_ops"),
                    4.0
                );
                assert_eq!(
                    number(explain.get("execution_stats").unwrap(), "vector_scan_count"),
                    // The notes collection has two vectors. The code collection
                    // checks all three, then its filter excludes the Python one.
                    if index == 0 { 2.0 } else { 3.0 }
                );
                assert_eq!(
                    explain
                        .get("filter")
                        .unwrap()
                        .get("matched")
                        .and_then(JsonValue::as_bool),
                    Some(true)
                );
            }
        }
        let before = fs::read(&path).unwrap();
        let db = Database::open_read_only(&path).unwrap();
        let code = db.collection("code").unwrap();
        let unfiltered = code.search().text("page cache").execute().unwrap();
        assert_eq!(unfiltered.results[0].chunk_external_id, "python#cache");
        for (id, line) in [("storage#cache", 1), ("storage#recovery", 2)] {
            let chunk = code.get_chunk_by_external_id(id).unwrap();
            let span = chunk.source_span.unwrap();
            assert_eq!(span.line_start, Some(line));
            assert_eq!(span.line_end, Some(line));
        }
        assert_eq!(code.stats().unwrap().embedding_count, 3);
        assert_eq!(
            db.collection("notes")
                .unwrap()
                .stats()
                .unwrap()
                .embedding_count,
            2
        );
        assert_eq!(fs::read(&path).unwrap(), before);
    }

    #[test]
    fn existing_database_is_never_overwritten() {
        let fixture = Fixture::new();
        let path = fixture.0.join("existing.coffer");
        fs::write(&path, b"existing user content").unwrap();
        assert!(demonstrate(&path).is_err());
        assert_eq!(fs::read(path).unwrap(), b"existing user content");
    }
}
