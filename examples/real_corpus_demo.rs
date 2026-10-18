//! Three source-backed demonstrations. Run with a new database path:
//! cargo run --example real_corpus_demo -- /tmp/real-corpus-demo.coffer
//! Pinned upstream source snapshots and precomputed model vectors are bundled.
//! No Python, model download, AST extraction, or external service is required.

use std::path::{Path, PathBuf};

use cofferdb::{
    parse_json, Database, DatabaseConfig, Error, ErrorCode, JsonValue, OnConflict,
    QueryEmbeddingInput, ResourceProfile, Result, SearchWeights,
};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/data/real_corpus")
}

fn invalid_fixture() -> Error {
    Error::new(ErrorCode::InvalidJsonlSchema, "invalid bundled demo query")
}

fn load_query(name: &str) -> Result<(String, QueryEmbeddingInput)> {
    let wire = std::fs::read_to_string(fixtures().join(name))?;
    let value = parse_json(&wire)?;
    let text = value
        .get("text")
        .and_then(JsonValue::as_str)
        .ok_or_else(invalid_fixture)?;
    let embedding = value.get("embedding").ok_or_else(invalid_fixture)?;
    let model = embedding
        .get("model_name")
        .and_then(JsonValue::as_str)
        .ok_or_else(invalid_fixture)?;
    let dim = embedding
        .get("dim")
        .and_then(JsonValue::as_f64)
        .ok_or_else(invalid_fixture)?;
    let values = embedding
        .get("vector")
        .and_then(JsonValue::as_array)
        .ok_or_else(invalid_fixture)?;
    if dim != 1024.0 || values.len() != 1024 {
        return Err(invalid_fixture());
    }
    let vector = values
        .iter()
        .map(|value| {
            let n = value.as_f64().ok_or_else(invalid_fixture)?;
            let f = n as f32;
            if !n.is_finite() || !f.is_finite() {
                return Err(invalid_fixture());
            }
            Ok(f)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((
        text.into(),
        QueryEmbeddingInput {
            model_name: Some(model.into()),
            dim: 1024,
            vector,
        },
    ))
}

fn demonstrate(path: &Path) -> Result<JsonValue> {
    let db = Database::create(path, DatabaseConfig::default())?;
    for name in ["markdown", "code"] {
        db.import_manifest(
            fixtures().join(name).join("import_manifest.json"),
            OnConflict::Error,
        )?;
    }
    drop(db);
    let db = Database::open_read_only(path)?;
    let mut runs = Vec::new();
    for (step, collection, query, filter, bm25, vector, profile) in [
        (
            "markdown_knowledge_base",
            "real_notes",
            "markdown-query.json",
            r#"{"doc.lang":"markdown"}"#,
            0.5,
            0.5,
            ResourceProfile::Balanced,
        ),
        (
            "rust_code_lexical_priority",
            "real_code",
            "code-query.json",
            r#"{"doc.lang":"rust"}"#,
            0.9,
            0.1,
            ResourceProfile::Balanced,
        ),
        (
            "rust_code_vector_priority",
            "real_code",
            "code-query.json",
            r#"{"doc.lang":"rust"}"#,
            0.1,
            0.9,
            ResourceProfile::Balanced,
        ),
        (
            "rag_debug_balanced",
            "real_code",
            "code-query.json",
            r#"{"doc.lang":"rust"}"#,
            0.5,
            0.5,
            ResourceProfile::Balanced,
        ),
        (
            "rag_debug_expanded",
            "real_code",
            "code-query.json",
            r#"{"doc.lang":"rust"}"#,
            0.5,
            0.5,
            ResourceProfile::Expanded,
        ),
        (
            "rag_debug_constrained",
            "real_code",
            "code-query.json",
            r#"{"doc.lang":"rust"}"#,
            0.5,
            0.5,
            ResourceProfile::Constrained,
        ),
    ] {
        let (text, embedding) = load_query(query)?;
        let result = db
            .collection(collection)?
            .search()
            .text(&text)
            .query_embedding(embedding)
            .filter(parse_json(filter)?)
            .weights(SearchWeights::new(bm25, vector, 0.0)?)
            .resource_profile(profile)
            .top_k(10)
            .execute()?;
        runs.push(JsonValue::Object(
            [
                ("step".into(), JsonValue::String(step.into())),
                ("query".into(), JsonValue::String(text)),
                ("result".into(), result.to_json()),
            ]
            .into_iter()
            .collect(),
        ));
    }
    Ok(JsonValue::Object([
        ("example_version".into(), JsonValue::String("2".into())),
        ("source_commit".into(), JsonValue::String("dcc14f912bf8790792a2df082eddc82ab2539229".into())),
        ("notice".into(), JsonValue::String(
            "Two pinned Rust by Example Markdown sources, two document chunks, five unchanged fenced Rust snippets and matching precomputed BGE-M3 vectors. Source snapshots and upstream licenses are bundled under source_snapshot. These small examples demonstrate import, filtering, weighting and explain; they are not a representative relevance benchmark and do not establish a winning profile. Snippets are retrieval inputs and may contain intentional upstream compile errors or panics. Database defaults are unchanged.".into())),
        ("runs".into(), JsonValue::Array(runs)),
    ].into_iter().collect()))
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().filter(|path| !path.is_empty()).ok_or_else(|| {
        Error::new(
            ErrorCode::CliUsage,
            "usage: real_corpus_demo <new-database.coffer>",
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
                "cofferdb-real-demo-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
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
    fn all_imported_chunks_match_original_git_bytes_after_reopen() {
        let fixture = Fixture::new();
        let path = fixture.0.join("source.coffer");
        demonstrate(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let db = Database::open_read_only(&path).unwrap();
        let mut checked = 0;
        for (name, collection, lang) in [
            ("markdown", "real_notes", "markdown"),
            ("code", "real_code", "rust"),
        ] {
            let col = db.collection(collection).unwrap();
            for wire in fs::read_to_string(fixtures().join(name).join("chunks.jsonl"))
                .unwrap()
                .lines()
            {
                let row = parse_json(wire).unwrap();
                let id = row.get("chunk_external_id").unwrap().as_str().unwrap();
                let chunk = col.get_chunk_by_external_id(id).unwrap();
                let document = col
                    .get_document_by_external_id(
                        row.get("document_external_id").unwrap().as_str().unwrap(),
                    )
                    .unwrap();
                let snapshot = document.source_uri.as_ref().unwrap();
                let original = fs::read(fixtures().join(snapshot)).unwrap();
                assert_eq!(
                    chunk
                        .metadata
                        .get("source_span_coordinate_system")
                        .unwrap()
                        .as_str(),
                    Some("original UTF-8 bytes in document.source_uri")
                );
                assert!(chunk.metadata.get("source_coordinate_system").is_none());
                let span = chunk.source_span.as_ref().unwrap();
                let start = span.start.unwrap() as usize;
                let end = span.end.unwrap() as usize;
                assert_eq!(&original[start..end], chunk.text.as_bytes());
                assert_eq!(
                    span.line_start.unwrap() as usize,
                    original[..start].iter().filter(|b| **b == b'\n').count() + 1
                );
                let end_line = original[..end].iter().filter(|b| **b == b'\n').count()
                    + usize::from(end > 0 && original[end - 1] != b'\n');
                assert_eq!(span.line_end.unwrap() as usize, end_line);
                assert_eq!(document.metadata.get("lang").unwrap().as_str(), Some(lang));
                assert_eq!(chunk.metadata.get("path"), document.metadata.get("path"));
                checked += 1;
            }
        }
        assert_eq!(checked, 7);
        drop(db);
        assert_eq!(before, fs::read(&path).unwrap());
    }

    #[test]
    fn real_weights_change_ranking_and_explain_matches_scores_and_sources() {
        let fixture = Fixture::new();
        let report = demonstrate(&fixture.0.join("ranking.coffer")).unwrap();
        let wire = report.to_pretty_string();
        assert_eq!(report, parse_json(&wire).unwrap());
        let runs = report.get("runs").unwrap().as_array().unwrap();
        assert_eq!(runs.len(), 6);
        let ranked = |run: &JsonValue| -> Vec<String> {
            run.get("result")
                .unwrap()
                .get("results")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    row.get("chunk_external_id")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .into()
                })
                .collect()
        };
        assert_ne!(ranked(&runs[1]), ranked(&runs[2]));
        for (i, run) in runs.iter().enumerate() {
            let result = run.get("result").unwrap();
            let context = result.get("query_context").unwrap();
            assert_eq!(
                context.get("index_source").unwrap().as_str(),
                Some("persistent_checkpoint")
            );
            let limit = if i == 4 {
                500.0
            } else if i == 5 {
                50.0
            } else {
                200.0
            };
            for name in ["bm25", "vector", "fusion"] {
                assert_eq!(
                    context
                        .get("candidate_limits")
                        .unwrap()
                        .get(name)
                        .unwrap()
                        .as_f64(),
                    Some(limit)
                );
            }
            let rows = result.get("results").unwrap().as_array().unwrap();
            assert_eq!(rows.len(), if i == 0 { 2 } else { 5 });
            for row in rows {
                let explain = row.get("explain").unwrap();
                let source = explain.get("source").unwrap();
                let uri = source.get("source_uri").unwrap().as_str().unwrap();
                assert!(fixtures().join(uri).is_file());
                let boundary = explain.get("chunk_boundary").unwrap();
                assert!(
                    boundary.get("end").unwrap().as_f64() > boundary.get("start").unwrap().as_f64()
                );
                assert!(boundary.get("line_start").unwrap().as_f64().unwrap() >= 1.0);
                let scores = explain.get("scores").unwrap();
                let weights = explain.get("weights").unwrap().get("effective").unwrap();
                let expected = scores.get("bm25_norm").unwrap().as_f64().unwrap()
                    * weights.get("bm25").unwrap().as_f64().unwrap()
                    + scores.get("vector_norm").unwrap().as_f64().unwrap()
                        * weights.get("vector").unwrap().as_f64().unwrap();
                assert!(
                    (row.get("final_score").unwrap().as_f64().unwrap() - expected).abs() <= 1e-12
                );
                assert_eq!(
                    row.get("document_metadata")
                        .unwrap()
                        .get("lang")
                        .unwrap()
                        .as_str(),
                    Some(if i == 0 { "markdown" } else { "rust" })
                );
            }
        }
    }
}
