use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use cofferdb::{
    parse_json, ChunkInput, CollectionConfig, Database, DatabaseConfig, DistanceMetric,
    DocumentInput, Error, ErrorCode, JsonValue, MetadataBoostConfig, OnConflict, TokenizerConfig,
    VectorIndexConfig,
};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cofferdb-json-depth-{}-{}",
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

fn assert_depth_error(err: &Error) {
    let kind = err
        .detail()
        .iter()
        .find(|(key, _)| key == "json_input_kind")
        .map(|(_, value)| value.as_str())
        .unwrap();
    let expected = match kind {
        "filter" => ErrorCode::InvalidMetadataFilter,
        "metadata_boost" => ErrorCode::InvalidMetadataBoost,
        "stored_json" => ErrorCode::RecoveryError,
        _ => ErrorCode::InvalidJsonlSchema,
    };
    assert_eq!(err.code(), expected);
    assert!(err
        .detail()
        .iter()
        .any(|(key, value)| key == "json_limit_reason" && value == "depth_exceeded"));
    assert!(err
        .detail()
        .iter()
        .any(|(key, value)| key == "limit" && value == "128"));
}

fn nested_value(depth: usize) -> JsonValue {
    let mut value = JsonValue::Null;
    for n in 0..depth {
        value = if n % 2 == 0 {
            JsonValue::Array(vec![value])
        } else {
            JsonValue::Object([("v".into(), value)].into_iter().collect())
        };
    }
    value
}

#[test]
fn direct_rust_values_are_rejected_and_released_without_aborting_host() {
    const WORKER: &str = "COFFERDB_JSON_DEPTH_TEST_WORKER";
    if std::env::var_os(WORKER).is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "direct_rust_values_are_rejected_and_released_without_aborting_host",
                "--nocapture",
            ])
            .env(WORKER, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "status={:?}, stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let fixture = Fixture::new();
    let path = fixture.0.join("data.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            name: "notes".into(),
            tokenizer: TokenizerConfig::default(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
    let before = fs::read(&path).unwrap();
    let value = nested_value(30_000);
    let copy = value.clone();
    assert!(value == copy);
    assert_eq!(format!("{value:?}"), "JsonValue(<invalid or over-depth>)");
    assert_depth_error(&value.try_to_compact_string().unwrap_err());
    assert!(value.validate_finite_numbers().is_ok());
    assert!(value.to_compact_string().len() > 60_000);
    assert_depth_error(
        &collection
            .insert_document(DocumentInput {
                external_id: "bad-document".into(),
                title: None,
                source_uri: None,
                metadata: copy,
                chunks: vec![],
            })
            .unwrap_err(),
    );
    assert_depth_error(
        &collection
            .insert_document(DocumentInput {
                external_id: "bad-chunk".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: vec![ChunkInput {
                    chunk_external_id: "bad-chunk#1".into(),
                    text: "text".into(),
                    ordinal: 0,
                    metadata: value.clone(),
                    source_span: None,
                }],
            })
            .unwrap_err(),
    );
    assert_depth_error(
        &collection
            .search()
            .text("query")
            .filter(value.clone())
            .execute()
            .unwrap_err(),
    );
    assert_depth_error(&MetadataBoostConfig::parse(value.clone()).unwrap_err());
    assert_depth_error(
        &db.why_not_debug(
            "notes",
            "missing",
            Some("query".into()),
            None,
            value.clone(),
            None,
            false,
            1,
            None,
        )
        .unwrap_err(),
    );
    assert_depth_error(&TokenizerConfig::from_json(&value).unwrap_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(db.stats().unwrap().document_count, 0);
    drop(value);
    drop(collection);
    drop(db);
}

#[test]
fn cli_rejects_deep_filter_boost_and_embedding_with_machine_readable_errors() {
    let fixture = Fixture::new();
    let missing = fixture.0.join("missing.coffer");
    for raw in [
        format!("{}0{}", "[".repeat(30_000), "]".repeat(30_000)),
        "[".repeat(30_000),
        format!("{}0{}", "{\"v\":".repeat(129), "}".repeat(129)),
    ] {
        fs::write(fixture.0.join("input.json"), &raw).unwrap();
        for flag in ["--filter", "--metadata-boost", "--embedding"] {
            if flag == "--embedding" {
                fs::write(
                    fixture.0.join("input.json"),
                    format!("{{\"vector\":{raw}}}"),
                )
                .unwrap();
            } else {
                fs::write(fixture.0.join("input.json"), &raw).unwrap();
            }
            let value = if flag == "--filter" {
                raw.clone()
            } else {
                fixture.0.join("input.json").to_str().unwrap().to_owned()
            };
            for command in ["search", "why-not"] {
                let mut child = Command::new(env!("CARGO_BIN_EXE_coffer"));
                child.arg(command).arg(&missing);
                if command == "search" {
                    child.arg("query");
                } else {
                    child.args(["--query", "query", "--chunk-id", "missing"]);
                }
                let output = child
                    .args(["--collection", "notes", flag, &value, "--json"])
                    .output()
                    .unwrap();
                assert_eq!(
                    output.status.code(),
                    Some(1),
                    "{flag}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert!(output.stdout.is_empty());
                let envelope = parse_json(&String::from_utf8(output.stderr).unwrap()).unwrap();
                let error = envelope.get("error").unwrap();
                assert_eq!(
                    error.get("code").and_then(JsonValue::as_str),
                    Some(match flag {
                        "--filter" => "invalid_metadata_filter",
                        "--metadata-boost" => "invalid_metadata_boost",
                        _ => "invalid_jsonl_schema",
                    })
                );
                let details = error.get("detail").unwrap();
                assert_eq!(
                    details.get("json_limit_reason").and_then(JsonValue::as_str),
                    Some("depth_exceeded"),
                    "{flag}: {error:?}"
                );
                assert_eq!(
                    details.get("limit").and_then(JsonValue::as_f64),
                    Some(128.0)
                );
                assert!(!missing.exists());
            }
        }
    }
}

#[test]
fn manifest_and_jsonl_depth_failures_commit_no_main_data() {
    let fixture = Fixture::new();
    let path = fixture.0.join("data.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let before = fs::read(&path).unwrap();
    let manifest = fixture.0.join("import.json");
    fs::write(&manifest, "[".repeat(30_000)).unwrap();
    assert_depth_error(
        &db.import_manifest(&manifest, OnConflict::Error)
            .unwrap_err(),
    );
    let raw = format!("{}0{}", "[".repeat(30_000), "]".repeat(30_000));
    let valid_manifest = r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#;
    fs::write(&manifest, valid_manifest).unwrap();
    fs::write(
        fixture.0.join("documents.jsonl"),
        format!("{{\"external_id\":\"doc\",\"metadata\":{raw}}}\n"),
    )
    .unwrap();
    fs::write(fixture.0.join("chunks.jsonl"), "").unwrap();
    let err = db
        .import_manifest(&manifest, OnConflict::Error)
        .unwrap_err();
    assert_depth_error(&err);
    assert!(err
        .detail()
        .iter()
        .any(|(key, value)| key == "line" && value == "1"));
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(db.list_collections().unwrap().is_empty());
}

#[test]
fn maximum_row_depth_metadata_survives_binary_reopen_export_and_reimport() {
    let fixture = Fixture::new();
    let path = fixture.0.join("data.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let config = CollectionConfig {
        name: "notes".into(),
        tokenizer: TokenizerConfig::default(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: None,
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    };
    let collection = db.create_collection(config).unwrap();
    // The JSONL wrapper must not consume any of the metadata's 128 levels.
    let metadata = JsonValue::Object([("v".into(), nested_value(127))].into_iter().collect());
    collection
        .insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: metadata.clone(),
            chunks: vec![ChunkInput {
                chunk_external_id: "chunk".into(),
                text: "searchable text".into(),
                ordinal: 0,
                metadata: metadata.clone(),
                source_span: None,
            }],
        })
        .unwrap();
    let before = fs::read(&path).unwrap();
    let too_deep = JsonValue::Object([("v".into(), nested_value(128))].into_iter().collect());
    assert_depth_error(
        &collection
            .insert_document(DocumentInput {
                external_id: "too-deep".into(),
                title: None,
                source_uri: None,
                metadata: too_deep,
                chunks: vec![],
            })
            .unwrap_err(),
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    drop(collection);
    drop(db);
    let db = Database::open(&path).unwrap();
    let output = fixture.0.join("export");
    db.export_collection_jsonl("notes", &output).unwrap();
    let copy = Database::create(fixture.0.join("copy.coffer"), DatabaseConfig::default()).unwrap();
    assert_eq!(
        copy.import_manifest(output.join("export_manifest.json"), OnConflict::Error)
            .unwrap()
            .chunks_inserted,
        1
    );
}
