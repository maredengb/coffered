use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use cofferdb::{
    parse_json, CheckOptions, CollectionConfig, CompactOptions, Database, DatabaseConfig,
    DatabaseOpenOptions, DistanceMetric, ErrorCode, JsonValue, OnConflict, QueryEmbeddingInput,
    SearchResults, TokenizerConfig, VectorIndexConfig,
};

#[path = "support/error_contract.rs"]
#[allow(dead_code)]
mod error_contract;

// Cargo's target runner covers the test executable, but not CLI subprocesses.
fn coffer_command() -> Command {
    let binary = env!("CARGO_BIN_EXE_coffer");
    if let Some(value) = std::env::var_os("COFFERDB_TEST_CLI_RUNNER") {
        let value = parse_json(value.to_str().expect("test runner must be UTF-8"))
            .expect("test runner must be a JSON argv array");
        let argv = value.as_array().expect("test runner must be a JSON array");
        let (program, arguments) = argv
            .split_first()
            .expect("test runner argv must not be empty");
        let program = program
            .as_str()
            .expect("test runner program must be a string");
        assert!(!program.is_empty(), "test runner program must not be empty");
        let mut command = Command::new(program);
        for argument in arguments {
            command.arg(
                argument
                    .as_str()
                    .expect("test runner arguments must be strings"),
            );
        }
        command.arg(binary);
        return command;
    }
    Command::new(binary)
}

const METRICS: [DistanceMetric; 3] = [
    DistanceMetric::Cosine,
    DistanceMetric::DotProduct,
    DistanceMetric::Euclidean,
];

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = loop {
            let path = std::env::temp_dir().join(format!(
                "cofferdb-vector-manifest-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("cannot create fixture: {error}"),
            }
        };
        fs::write(
            directory.join("documents.jsonl"),
            r#"{"external_id":"doc","metadata":{}}"#,
        )
        .unwrap();
        fs::write(directory.join("chunks.jsonl"), concat!(
            "{\"document_external_id\":\"doc\",\"chunk_external_id\":\"c1\",\"ordinal\":0,\"text\":\"alpha alpha\",\"metadata\":{}}\n",
            "{\"document_external_id\":\"doc\",\"chunk_external_id\":\"c2\",\"ordinal\":1,\"text\":\"beta\",\"metadata\":{}}\n"
        )).unwrap();
        fs::write(directory.join("empty.jsonl"), "").unwrap();
        let fixture = Self(directory);
        fixture.embeddings(false);
        fixture
    }

    fn embeddings(&self, raw: bool) {
        let vector = if raw { "[3,0]" } else { "[0.6,0.8]" };
        fs::write(self.0.join("embeddings.jsonl"), format!(
            "{{\"chunk_external_id\":\"c1\",\"model_name\":\"m\",\"dim\":2,\"vector\":{vector}}}\n{{\"chunk_external_id\":\"c2\",\"model_name\":\"m\",\"dim\":2,\"vector\":[0,1]}}\n"
        )).unwrap();
    }

    fn manifest(
        &self,
        flag: Option<JsonValue>,
        metric: DistanceMetric,
        embedding_file: Option<&str>,
    ) -> PathBuf {
        let mut value = parse_json(r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
        let object = value.as_object_mut().unwrap();
        object.insert(
            "distance_metric".into(),
            JsonValue::String(metric.as_str().into()),
        );
        if let Some(flag) = flag {
            object.insert("vectors_normalized".into(), flag);
        }
        if let Some(file) = embedding_file {
            object.insert("embeddings".into(), JsonValue::String(file.into()));
        }
        let path = self.0.join("manifest.json");
        fs::write(&path, value.to_compact_string()).unwrap();
        path
    }

    fn config(&self, db: &Database) -> cofferdb::CollectionInfo {
        db.list_collections()
            .unwrap()
            .into_iter()
            .find(|info| info.name == "notes")
            .unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn config(name: &str, normalized: bool) -> CollectionConfig {
    CollectionConfig {
        name: name.into(),
        tokenizer: TokenizerConfig::default_en(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: None,
        vectors_normalized: normalized,
        vector_index: VectorIndexConfig::Exact,
    }
}

fn assert_flag_error(error: &cofferdb::Error) {
    assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema, "{error}");
    let value = error.to_json();
    error_contract::validate_error(&value).unwrap();
    assert_eq!(
        value.get("recoverable").and_then(JsonValue::as_bool),
        Some(true)
    );
    assert_eq!(
        value
            .get("detail")
            .unwrap()
            .get("field")
            .and_then(JsonValue::as_str),
        Some("vectors_normalized")
    );
}

#[test]
fn manifest_flag_creation_inheritance_and_explicit_matching_use_collection_not_database_existence()
{
    for flag in [None, Some(false), Some(true)] {
        for metric in METRICS {
            let f = Fixture::new();
            let db =
                Database::create(f.0.join("source.coffer"), DatabaseConfig::default()).unwrap();
            db.create_collection(config("unrelated", true)).unwrap();
            let manifest = f.manifest(flag.map(JsonValue::Bool), metric, Some("embeddings.jsonl"));
            db.import_manifest(&manifest, OnConflict::Error).unwrap();
            let expected = flag.unwrap_or(false);
            assert_eq!(f.config(&db).vectors_normalized, expected);
            let manifest = f.manifest(None, metric, Some("embeddings.jsonl"));
            let mut value = parse_json(&fs::read_to_string(&manifest).unwrap()).unwrap();
            value.as_object_mut().unwrap().remove("embedding_dim");
            value.as_object_mut().unwrap().remove("distance_metric");
            fs::write(&manifest, value.to_compact_string()).unwrap();
            let summary = db.import_manifest(&manifest, OnConflict::Skip).unwrap();
            assert_eq!(summary.embeddings_skipped, 2);
            assert_eq!(f.config(&db).vectors_normalized, expected);
            let manifest = f.manifest(
                Some(JsonValue::Bool(expected)),
                metric,
                Some("embeddings.jsonl"),
            );
            db.import_manifest(&manifest, OnConflict::Skip).unwrap();
            assert_eq!(f.config(&db).vectors_normalized, expected);
        }
    }
}

#[test]
fn manifest_flag_conflicts_refuse_all_policies_and_empty_vector_sources_before_any_commit() {
    for normalized in [false, true] {
        for metric in METRICS {
            for conflict in [OnConflict::Error, OnConflict::Skip, OnConflict::Replace] {
                for embedding_file in [None, Some("empty.jsonl"), Some("embeddings.jsonl")] {
                    let f = Fixture::new();
                    let path = f.0.join("source.coffer");
                    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
                    let mut target = config("notes", normalized);
                    target.distance_metric = metric;
                    db.create_collection(target).unwrap();
                    db.import_manifest(
                        f.manifest(None, metric, Some("embeddings.jsonl")),
                        OnConflict::Error,
                    )
                    .unwrap();
                    let before = fs::read(&path).unwrap();
                    let error = db
                        .import_manifest(
                            f.manifest(Some(JsonValue::Bool(!normalized)), metric, embedding_file),
                            conflict,
                        )
                        .unwrap_err();
                    assert_flag_error(&error);
                    assert!(error.message().contains("does not match"));
                    assert_eq!(fs::read(&path).unwrap(), before);
                    assert_eq!(f.config(&db).vectors_normalized, normalized);
                    assert_eq!(
                        db.collection("notes")
                            .unwrap()
                            .stats()
                            .unwrap()
                            .embedding_count,
                        2
                    );
                }
            }
        }
    }
}

#[test]
fn successful_replace_cascade_keeps_the_flag_with_or_without_replacement_embeddings() {
    for flag in [None, Some(JsonValue::Bool(true))] {
        for embedding_file in [None, Some("empty.jsonl"), Some("embeddings.jsonl")] {
            let f = Fixture::new();
            let path = f.0.join("source.coffer");
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            db.import_manifest(
                f.manifest(
                    Some(JsonValue::Bool(true)),
                    DistanceMetric::Cosine,
                    Some("embeddings.jsonl"),
                ),
                OnConflict::Error,
            )
            .unwrap();
            let chunks = fs::read_to_string(f.0.join("chunks.jsonl")).unwrap();
            fs::write(
                f.0.join("chunks.jsonl"),
                chunks.replace("alpha alpha", "replaced alpha"),
            )
            .unwrap();
            let result = db
                .import_manifest(
                    f.manifest(flag.clone(), DistanceMetric::Cosine, embedding_file),
                    OnConflict::Replace,
                )
                .unwrap();
            assert_eq!(result.documents_replaced, 1);
            let collection = db.collection("notes").unwrap();
            assert_eq!(
                collection.get_chunk_by_external_id("c1").unwrap().text,
                "replaced alpha"
            );
            let expected = if embedding_file == Some("embeddings.jsonl") {
                2
            } else {
                0
            };
            assert_eq!(collection.stats().unwrap().embedding_count, expected);
            assert!(f.config(&db).vectors_normalized);
            db.compact(CompactOptions::default()).unwrap();
            db.close().unwrap();
            let reopened = Database::open(&path).unwrap();
            assert!(f.config(&reopened).vectors_normalized);
            assert_eq!(
                reopened
                    .collection("notes")
                    .unwrap()
                    .stats()
                    .unwrap()
                    .embedding_count,
                expected
            );
        }
    }
}

#[test]
fn valid_manifest_flags_do_not_override_actual_json_limits_or_input_io_failures() {
    for normalized in [false, true] {
        for json_limit in [false, true] {
            let f = Fixture::new();
            let path = f.0.join("source.coffer");
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            db.create_collection(config("notes", normalized)).unwrap();
            let manifest = f.manifest(
                Some(JsonValue::Bool(normalized)),
                DistanceMetric::Cosine,
                None,
            );
            if json_limit {
                fs::write(
                    f.0.join("documents.jsonl"),
                    format!(
                        "{{\"external_id\":\"doc\",\"metadata\":{{\"nested\":{}0{}}}}}",
                        "[".repeat(129),
                        "]".repeat(129)
                    ),
                )
                .unwrap();
            } else {
                fs::remove_file(f.0.join("documents.jsonl")).unwrap();
            }
            let before = fs::read(&path).unwrap();
            let error = db.import_manifest(&manifest, OnConflict::Skip).unwrap_err();
            assert_eq!(
                error.code(),
                if json_limit {
                    ErrorCode::InvalidJsonlSchema
                } else {
                    ErrorCode::IoError
                }
            );
            let value = error.to_json();
            error_contract::validate_error(&value).unwrap();
            let missing_flag = f.manifest(None, DistanceMetric::Cosine, None);
            let original_error = db
                .import_manifest(&missing_flag, OnConflict::Skip)
                .unwrap_err();
            let original = original_error.to_json();
            assert_eq!(error.code(), original_error.code());
            assert_eq!(
                value.get("suggested_action"),
                original.get("suggested_action")
            );
            assert_eq!(value.get("recoverable"), original.get("recoverable"));
            if json_limit {
                assert_eq!(
                    value
                        .get("detail")
                        .unwrap()
                        .get("json_limit_reason")
                        .and_then(JsonValue::as_str),
                    Some("depth_exceeded")
                );
            }
            assert_ne!(
                value
                    .get("detail")
                    .and_then(|detail| detail.get("field"))
                    .and_then(JsonValue::as_str),
                Some("vectors_normalized")
            );
            assert!(!value
                .get("suggested_action")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .contains("do not rewrite existing vectors"));
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(
                db.collection("notes")
                    .unwrap()
                    .stats()
                    .unwrap()
                    .document_count,
                0
            );
            assert_eq!(f.config(&db).vectors_normalized, normalized);
        }
    }
}

#[test]
fn manifest_flag_null_and_other_non_booleans_are_not_defaults_for_new_or_existing_collections() {
    for normalized in [None, Some(false), Some(true)] {
        for invalid in ["null", "0", "1", "\"true\"", "[]", "{}"] {
            let f = Fixture::new();
            let path = f.0.join("source.coffer");
            let db = Database::create(&path, DatabaseConfig::default()).unwrap();
            if let Some(normalized) = normalized {
                db.create_collection(config("notes", normalized)).unwrap();
            }
            let before = fs::read(&path).unwrap();
            let error = db
                .import_manifest(
                    f.manifest(
                        Some(parse_json(invalid).unwrap()),
                        DistanceMetric::Cosine,
                        None,
                    ),
                    OnConflict::Skip,
                )
                .unwrap_err();
            assert_flag_error(&error);
            assert!(error.message().contains("must be a boolean"));
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(
                db.list_collections().unwrap().len(),
                usize::from(normalized.is_some())
            );
        }
    }
}

#[test]
fn clearing_compacting_and_reopening_does_not_reset_the_persistent_flag() {
    let f = Fixture::new();
    let path = f.0.join("source.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    db.import_manifest(
        f.manifest(
            Some(JsonValue::Bool(true)),
            DistanceMetric::Cosine,
            Some("embeddings.jsonl"),
        ),
        OnConflict::Error,
    )
    .unwrap();
    let collection = db.collection("notes").unwrap();
    let document = collection.get_document_by_external_id("doc").unwrap();
    collection.delete_document(&document.document_id).unwrap();
    assert_eq!(collection.stats().unwrap().embedding_count, 0);
    db.compact(CompactOptions::default()).unwrap();
    db.close().unwrap();
    let db = Database::open(&path).unwrap();
    assert!(f.config(&db).vectors_normalized);
    let before = fs::read(&path).unwrap();
    assert_flag_error(
        &db.import_manifest(
            f.manifest(Some(JsonValue::Bool(false)), DistanceMetric::Cosine, None),
            OnConflict::Replace,
        )
        .unwrap_err(),
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    db.import_manifest(
        f.manifest(None, DistanceMetric::Cosine, None),
        OnConflict::Error,
    )
    .unwrap();
    assert!(f.config(&db).vectors_normalized);
}

#[test]
fn real_legacy_mixed_models_remain_diagnosable_exportable_and_explicitly_cleanable() {
    // Produced through the f4c6730 public API, before model admission was enforced.
    let hex = include_str!("fixtures/mixed-model-f4c6730.hex")
        .split_whitespace()
        .collect::<String>();
    assert_eq!(hex.len() % 2, 0);
    let bytes: Vec<u8> = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    let f = Fixture::new();
    let path = f.0.join("historical.coffer");
    fs::write(&path, &bytes).unwrap();
    let db = Database::open(&path).unwrap();
    let collection = db.collection("notes").unwrap();
    assert_eq!(collection.stats().unwrap().embedding_count, 2);
    assert_eq!(
        collection
            .search()
            .text("retrieval")
            .execute()
            .unwrap()
            .results
            .len(),
        2
    );
    assert!(f.config(&db).vectors_normalized);
    let report = db
        .check(CheckOptions {
            verify_indexes: true,
        })
        .unwrap();
    let issue = report
        .issues
        .iter()
        .find(|issue| issue.code == "embedding_model_mismatch")
        .unwrap();
    assert_eq!(
        (
            &*issue.stream,
            &*issue.visibility_impact,
            issue.repair_available
        ),
        ("critical", "none", false)
    );
    assert!(issue.offset.unwrap() < bytes.len() as u64);
    assert_eq!(
        collection
            .search()
            .vector(vec![1.0, 0.0])
            .execute()
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert_eq!(
        db.compact(CompactOptions::default()).unwrap_err().code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert_eq!(
        db.rebuild_index(None).unwrap_err().code(),
        ErrorCode::EmbeddingModelMismatch
    );
    let output = coffer_command()
        .arg("check")
        .arg(&path)
        .args(["--verify-indexes", "--json"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let checked = parse_json(&String::from_utf8(output.stdout).unwrap()).unwrap();
    assert_eq!(checked.get("ok"), Some(&JsonValue::Bool(true)));
    let checked_result = checked.get("result").unwrap();
    assert_eq!(
        checked_result.get("status").and_then(JsonValue::as_str),
        Some("error")
    );
    assert_eq!(
        checked_result
            .get("last_valid_commit")
            .and_then(JsonValue::as_f64),
        Some(3.0)
    );
    assert_eq!(
        checked_result
            .get("result")
            .unwrap()
            .get("live_chunks")
            .and_then(JsonValue::as_f64),
        Some(2.0)
    );
    let exported = f.0.join("rescued");
    db.export_collection_jsonl("notes", &exported).unwrap();
    let manifest =
        parse_json(&fs::read_to_string(exported.join("export_manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest.get("vectors_normalized"),
        Some(&JsonValue::Bool(true))
    );
    let names = fs::read_to_string(exported.join("embeddings.jsonl"))
        .unwrap()
        .lines()
        .map(|line| {
            parse_json(line)
                .unwrap()
                .get("model_name")
                .and_then(JsonValue::as_str)
                .unwrap()
                .to_owned()
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        names,
        ["M".to_owned(), "N".to_owned()].into_iter().collect()
    );
    assert_eq!(fs::read(&path).unwrap(), bytes);
    let restored_path = f.0.join("restored.coffer");
    let restored = Database::create(&restored_path, DatabaseConfig::default()).unwrap();
    let empty = fs::read(&restored_path).unwrap();
    assert_eq!(
        restored
            .import_manifest(exported.join("export_manifest.json"), OnConflict::Error)
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    assert_eq!(fs::read(&restored_path).unwrap(), empty);
    assert!(restored.list_collections().unwrap().is_empty());
    let reader = Database::open_with_options(
        &path,
        DatabaseOpenOptions {
            read_only: true,
            ..Default::default()
        },
    )
    .unwrap();
    collection
        .delete_chunk(&collection.get_chunk_by_external_id("c1").unwrap().chunk_id)
        .unwrap();
    assert_eq!(
        collection
            .search()
            .vector(vec![1.0, 0.0])
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
    assert_eq!(
        reader
            .collection("notes")
            .unwrap()
            .search()
            .vector(vec![1.0, 0.0])
            .execute()
            .unwrap_err()
            .code(),
        ErrorCode::EmbeddingModelMismatch
    );
    db.compact(CompactOptions::default()).unwrap();
    db.close().unwrap();
    let reopened = Database::open(&path).unwrap();
    assert!(f.config(&reopened).vectors_normalized);
    assert!(reopened
        .check(CheckOptions::default())
        .unwrap()
        .issues
        .iter()
        .all(|issue| issue.code != "embedding_model_mismatch"));
    assert_eq!(
        reopened
            .collection("notes")
            .unwrap()
            .search()
            .vector(vec![1.0, 0.0])
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
}

fn compare_results(source: &SearchResults, restored: &SearchResults) {
    assert_eq!(source.results.len(), restored.results.len());
    for (left, right) in source.results.iter().zip(&restored.results) {
        assert_eq!(left.chunk_external_id, right.chunk_external_id);
        assert_eq!(left.rank, right.rank);
        for (a, b) in [
            (left.vector_score, right.vector_score),
            (
                left.normalized_scores.vector_norm,
                right.normalized_scores.vector_norm,
            ),
            (Some(left.final_score), Some(right.final_score)),
        ] {
            match (a, b) {
                (Some(a), Some(b)) => assert!((a - b).abs() <= 1e-12),
                (None, None) => {}
                _ => panic!("score availability changed"),
            }
        }
        for field in ["distance_metric", "normalization_method", "vector_raw_kind"] {
            assert_eq!(left.explain.get(field), right.explain.get(field), "{field}");
        }
    }
}

#[test]
fn export_reimport_preserves_cosine_and_dot_product_scores_without_rewriting_vectors() {
    for normalized in [false, true] {
        for metric in [DistanceMetric::Cosine, DistanceMetric::DotProduct] {
            let f = Fixture::new();
            let raw = metric == DistanceMetric::DotProduct && !normalized;
            f.embeddings(raw);
            let source =
                Database::create(f.0.join("source.coffer"), DatabaseConfig::default()).unwrap();
            source
                .import_manifest(
                    f.manifest(
                        Some(JsonValue::Bool(normalized)),
                        metric,
                        Some("embeddings.jsonl"),
                    ),
                    OnConflict::Error,
                )
                .unwrap();
            source.compact(CompactOptions::default()).unwrap();
            let exported = f.0.join("exported");
            source.export_collection_jsonl("notes", &exported).unwrap();
            let manifest =
                parse_json(&fs::read_to_string(exported.join("export_manifest.json")).unwrap())
                    .unwrap();
            assert_eq!(
                manifest.get("vectors_normalized"),
                Some(&JsonValue::Bool(normalized))
            );
            assert_eq!(
                manifest.get("embedding_dim").and_then(JsonValue::as_f64),
                Some(2.0)
            );
            assert_eq!(
                manifest.get("distance_metric").and_then(JsonValue::as_str),
                Some(metric.as_str())
            );
            let restored =
                Database::create(f.0.join("restored.coffer"), DatabaseConfig::default()).unwrap();
            restored
                .create_collection(config("unrelated", false))
                .unwrap();
            restored
                .import_manifest(exported.join("export_manifest.json"), OnConflict::Error)
                .unwrap();
            assert_eq!(f.config(&restored).vectors_normalized, normalized);
            let query = QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: if raw { vec![2.0, 0.0] } else { vec![1.0, 0.0] },
            };
            for text in [None, Some("alpha")] {
                let search = |db: &Database| {
                    let builder = db
                        .collection("notes")
                        .unwrap()
                        .search()
                        .query_embedding(query.clone());
                    match text {
                        Some(text) => builder.text(text).execute(),
                        None => builder.execute(),
                    }
                };
                match (search(&source), search(&restored)) {
                    (Ok(left), Ok(right)) => {
                        assert_eq!(left.results.len(), 2);
                        compare_results(&left, &right);
                    }
                    (Err(left), Err(right)) if text.is_some() && raw => {
                        assert_eq!(left.code(), right.code())
                    }
                    (left, right) => panic!("round-trip search mismatch: {left:?} / {right:?}"),
                }
            }
            let second = f.0.join("second-export");
            restored.export_collection_jsonl("notes", &second).unwrap();
            assert_eq!(
                fs::read(exported.join("embeddings.jsonl")).unwrap(),
                fs::read(second.join("embeddings.jsonl")).unwrap()
            );
        }
    }
}

#[test]
fn empty_collection_export_and_cli_refusals_keep_the_flag_and_existing_error_envelope() {
    for normalized in [false, true] {
        let f = Fixture::new();
        let path = f.0.join("source.coffer");
        let db = Database::create(&path, DatabaseConfig::default()).unwrap();
        db.create_collection(config("notes", normalized)).unwrap();
        let exported = f.0.join("exported");
        db.export_collection_jsonl("notes", &exported).unwrap();
        let manifest =
            parse_json(&fs::read_to_string(exported.join("export_manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            manifest.get("vectors_normalized"),
            Some(&JsonValue::Bool(normalized))
        );
        assert!(fs::read(exported.join("embeddings.jsonl"))
            .unwrap()
            .is_empty());
        let restored =
            Database::create(f.0.join("restored.coffer"), DatabaseConfig::default()).unwrap();
        restored
            .import_manifest(exported.join("export_manifest.json"), OnConflict::Error)
            .unwrap();
        assert_eq!(f.config(&restored).vectors_normalized, normalized);
        assert_eq!(
            restored
                .collection("notes")
                .unwrap()
                .stats()
                .unwrap()
                .embedding_count,
            0
        );
        db.close().unwrap();
        for flag in [JsonValue::Null, JsonValue::Bool(!normalized)] {
            let manifest = f.manifest(Some(flag), DistanceMetric::Cosine, None);
            let before = fs::read(&path).unwrap();
            let output = coffer_command()
                .arg("import")
                .arg(&path)
                .arg("--manifest")
                .arg(&manifest)
                .args(["--on-conflict", "skip", "--json"])
                .output()
                .unwrap();
            assert_eq!(
                output.status.code(),
                Some(1),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.stdout.is_empty());
            let value = parse_json(&String::from_utf8(output.stderr).unwrap()).unwrap();
            let error = value.get("error").unwrap();
            error_contract::validate_error(error).unwrap();
            assert_eq!(
                error.get("code").and_then(JsonValue::as_str),
                Some("invalid_jsonl_schema")
            );
            assert_eq!(
                error
                    .get("detail")
                    .unwrap()
                    .get("field")
                    .and_then(JsonValue::as_str),
                Some("vectors_normalized")
            );
            assert_eq!(fs::read(&path).unwrap(), before);
        }
    }
}
