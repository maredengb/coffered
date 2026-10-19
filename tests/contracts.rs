use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use cofferdb::{
    parse_json, ChunkInput, CollectionConfig, CompactOptions, Database, DatabaseConfig,
    DatabaseOpenOptions, DistanceMetric, DocumentInput, EmbeddingInput, ErrorCode, JsonValue,
    MetadataBoostConfig, OnConflict, QueryEmbeddingInput, ResourceProfile, TokenizerConfig,
    VectorIndexConfig,
};

#[path = "support/error_contract.rs"]
mod error_contract;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = loop {
            let path = std::env::temp_dir().join(format!(
                "cofferdb-contracts-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("cannot create test directory: {error}"),
            }
        };
        fs::write(
            path.join("documents.jsonl"),
            r#"{"external_id":"doc","metadata":{}}"#,
        )
        .unwrap();
        fs::write(path.join("chunks.jsonl"), r#"{"document_external_id":"doc","chunk_external_id":"doc#1","ordinal":0,"text":"the WalRecovery page_cache src/storage.rs CAFÉ 中文","metadata":{}}"#).unwrap();
        let manifest = r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#;
        fs::write(path.join("import.json"), manifest).unwrap();
        Self(path)
    }

    fn db(&self) -> PathBuf {
        self.0.join("test.coffer")
    }
    fn manifest(&self) -> PathBuf {
        self.0.join("import.json")
    }

    fn set_tokenizer(&self, tokenizer: JsonValue) {
        let mut manifest = parse_json(&fs::read_to_string(self.manifest()).unwrap()).unwrap();
        manifest
            .as_object_mut()
            .unwrap()
            .insert("tokenizer".into(), tokenizer);
        fs::write(self.manifest(), manifest.to_compact_string()).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn config(name: &str) -> CollectionConfig {
    CollectionConfig {
        name: name.into(),
        tokenizer: TokenizerConfig::default_en(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: None,
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    }
}

fn insert(db: &Database, id: &str) {
    db.collection("notes")
        .unwrap()
        .insert_document(DocumentInput {
            external_id: id.into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: vec![ChunkInput {
                chunk_external_id: id.into(),
                text: "WalRecovery page_cache".into(),
                ordinal: 0,
                metadata: JsonValue::object(),
                source_span: None,
            }],
        })
        .unwrap();
}

#[test]
fn legal_manifest_above_one_mib_imports_through_api_and_cli_without_changing_inputs() {
    for through_cli in [false, true] {
        let fixture = Fixture::new();
        let compact = fs::read(fixture.manifest()).unwrap();
        let mut wire = vec![b' '; 1024 * 1024 + 1];
        wire.extend_from_slice(&compact);
        fs::write(fixture.manifest(), &wire).unwrap();
        let documents = fs::read(fixture.0.join("documents.jsonl")).unwrap();
        let chunks = fs::read(fixture.0.join("chunks.jsonl")).unwrap();
        if through_cli {
            let database_path = fixture.db();
            let manifest_path = fixture.manifest();
            for args in [
                vec!["init", database_path.to_str().unwrap()],
                vec![
                    "import",
                    database_path.to_str().unwrap(),
                    "--manifest",
                    manifest_path.to_str().unwrap(),
                ],
            ] {
                let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
                    .args(&args)
                    .arg("--json")
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert_eq!(
                    parse_json(&String::from_utf8(output.stdout).unwrap())
                        .unwrap()
                        .get("ok"),
                    Some(&JsonValue::Bool(true))
                );
            }
        } else {
            let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
            db.import_manifest(fixture.manifest(), OnConflict::Error)
                .unwrap();
            db.close().unwrap();
        }
        assert_eq!(fs::read(fixture.manifest()).unwrap(), wire);
        assert_eq!(
            fs::read(fixture.0.join("documents.jsonl")).unwrap(),
            documents
        );
        assert_eq!(fs::read(fixture.0.join("chunks.jsonl")).unwrap(), chunks);
        let db = Database::open_read_only(fixture.db()).unwrap();
        let collection = db.collection("notes").unwrap();
        let stats = collection.stats().unwrap();
        assert_eq!((stats.document_count, stats.chunk_count), (1, 1));
        assert_eq!(
            collection
                .get_document_by_external_id("doc")
                .unwrap()
                .external_id,
            "doc"
        );
        let chunk = collection.get_chunk_by_external_id("doc#1").unwrap();
        assert_eq!(
            chunk.text,
            "the WalRecovery page_cache src/storage.rs CAFÉ 中文"
        );
        let export = fixture.0.join("export");
        db.export_collection_jsonl("notes", &export).unwrap();
        db.close().unwrap();
        let copy =
            Database::create(fixture.0.join("copy.coffer"), DatabaseConfig::default()).unwrap();
        copy.import_manifest(export.join("export_manifest.json"), OnConflict::Error)
            .unwrap();
        assert_eq!(
            copy.collection("notes")
                .unwrap()
                .get_chunk_by_external_id("doc#1")
                .unwrap()
                .text,
            chunk.text
        );
        copy.close().unwrap();
    }
}

#[test]
fn tokenizer_and_metadata_profiles_do_not_use_resource_preset_validation() {
    let metadata = parse_json(
        r#"{"profile":"application-specific-preset","resource_profile":"user-defined-category"}"#,
    )
    .unwrap();
    let tokenizer = TokenizerConfig::default_en().to_json();
    assert_eq!(
        tokenizer.get("profile").and_then(JsonValue::as_str),
        Some("latin_code_default")
    );
    for through_cli in [false, true] {
        let fixture = Fixture::new();
        fixture.set_tokenizer(tokenizer.clone());
        for name in ["documents.jsonl", "chunks.jsonl"] {
            let path = fixture.0.join(name);
            let mut value = parse_json(&fs::read_to_string(&path).unwrap()).unwrap();
            value
                .as_object_mut()
                .unwrap()
                .insert("metadata".into(), metadata.clone());
            fs::write(path, value.to_compact_string()).unwrap();
        }
        let inputs = ["import.json", "documents.jsonl", "chunks.jsonl"].map(|name| {
            (
                fixture.0.join(name),
                fs::read(fixture.0.join(name)).unwrap(),
            )
        });
        if through_cli {
            for args in [
                vec!["init".into(), fixture.db().to_str().unwrap().to_owned()],
                vec![
                    "import".into(),
                    fixture.db().to_str().unwrap().to_owned(),
                    "--manifest".into(),
                    fixture.manifest().to_str().unwrap().to_owned(),
                ],
            ] {
                let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
                    .args(args)
                    .arg("--json")
                    .output()
                    .unwrap();
                assert!(output.status.success(), "{output:?}");
                assert_eq!(
                    parse_json(std::str::from_utf8(&output.stdout).unwrap())
                        .unwrap()
                        .get("ok"),
                    Some(&JsonValue::Bool(true))
                );
            }
        } else {
            let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
            db.import_manifest(fixture.manifest(), OnConflict::Error)
                .unwrap();
            db.close().unwrap();
        }
        for (path, before) in inputs {
            assert_eq!(fs::read(path).unwrap(), before);
        }
        let db = Database::open_read_only(fixture.db()).unwrap();
        let collection = db.collection("notes").unwrap();
        assert_eq!(
            collection
                .get_document_by_external_id("doc")
                .unwrap()
                .metadata,
            metadata
        );
        assert_eq!(
            collection
                .get_chunk_by_external_id("doc#1")
                .unwrap()
                .metadata,
            metadata
        );
        let export = fixture.0.join("profile-export");
        db.export_collection_jsonl("notes", &export).unwrap();
        db.close().unwrap();
        let manifest =
            parse_json(&fs::read_to_string(export.join("export_manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest.get("tokenizer"), Some(&tokenizer));
        let copy =
            Database::create(fixture.0.join("copy.coffer"), DatabaseConfig::default()).unwrap();
        copy.import_manifest(export.join("export_manifest.json"), OnConflict::Error)
            .unwrap();
        let collection = copy.collection("notes").unwrap();
        assert_eq!(
            collection
                .get_document_by_external_id("doc")
                .unwrap()
                .metadata,
            metadata
        );
        assert_eq!(
            collection
                .get_chunk_by_external_id("doc#1")
                .unwrap()
                .metadata,
            metadata
        );
        copy.close().unwrap();
    }
}

#[test]
fn memory_metadata_template_filters_and_round_trips_without_automatic_lifecycle() {
    let fixture = Fixture::new();
    let template = parse_json(include_str!(
        "../examples/data/memory/metadata-template.json"
    ))
    .unwrap();
    assert_eq!(template.as_object().unwrap().len(), 10);
    let mut other = template.clone();
    for (key, value) in [
        ("type", "event"),
        ("source", "tool"),
        ("scope", "session"),
        ("project_id", "other-project"),
        ("session_id", "other-session"),
    ] {
        other
            .as_object_mut()
            .unwrap()
            .insert(key.into(), JsonValue::String(value.into()));
    }
    // Expiration is stored metadata, with no implicit TTL or default rank boost.
    other.as_object_mut().unwrap().insert(
        "expires_at".into(),
        JsonValue::String("2020-01-01T00:00:00Z".into()),
    );
    let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    let memory = db.create_collection(config("memory")).unwrap();
    memory
        .insert_document(DocumentInput {
            external_id: "entry".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: [template.clone(), other.clone()]
                .into_iter()
                .enumerate()
                .map(|(i, metadata)| ChunkInput {
                    chunk_external_id: format!("entry#{i}"),
                    text: "deployment fact".into(),
                    ordinal: i,
                    metadata,
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
    db.close().unwrap();
    let reopened = Database::open_read_only(fixture.db()).unwrap();
    let memory = reopened.collection("memory").unwrap();
    for key in ["type", "source", "scope", "project_id", "session_id"] {
        let filter = parse_json(&format!(
            "{{\"chunk.{key}\":{}}}",
            template.get(key).unwrap().to_compact_string()
        ))
        .unwrap();
        let hits = memory
            .search()
            .text("deployment")
            .filter(filter)
            .execute()
            .unwrap();
        assert_eq!(
            hits.results
                .iter()
                .map(|hit| hit.chunk_external_id.as_str())
                .collect::<Vec<_>>(),
            ["entry#0"]
        );
    }
    assert_eq!(
        memory
            .search()
            .text("deployment")
            .execute()
            .unwrap()
            .results
            .len(),
        2
    );
    assert_eq!(
        memory.get_chunk_by_external_id("entry#0").unwrap().metadata,
        template
    );
    assert_eq!(
        memory.get_chunk_by_external_id("entry#1").unwrap().metadata,
        other
    );
    let export = fixture.0.join("memory-export");
    reopened.export_collection_jsonl("memory", &export).unwrap();
    reopened.close().unwrap();
    let copy = Database::create(
        fixture.0.join("memory-copy.coffer"),
        DatabaseConfig::default(),
    )
    .unwrap();
    copy.import_manifest(export.join("export_manifest.json"), OnConflict::Error)
        .unwrap();
    let memory = copy.collection("memory").unwrap();
    assert_eq!(
        memory.get_chunk_by_external_id("entry#0").unwrap().metadata,
        template
    );
    assert_eq!(
        memory.get_chunk_by_external_id("entry#1").unwrap().metadata,
        other
    );
    assert_eq!(
        memory
            .search()
            .text("deployment")
            .execute()
            .unwrap()
            .results
            .len(),
        2
    );
    copy.close().unwrap();
}

fn cli(args: &[&str]) -> JsonValue {
    let mut arguments = vec!["coffer".to_owned()];
    arguments.extend(args.iter().map(|value| (*value).to_owned()));
    arguments.push("--json".into());
    let output = cofferdb::cli::run(arguments).unwrap();
    let envelope = parse_json(&output).unwrap();
    for warning in envelope.get("warnings").unwrap().as_array().unwrap() {
        fields(warning, &["code", "severity", "path", "message", "details"]);
        assert!(warning.get("details").unwrap().as_object().is_some());
    }
    for field in [
        "schema_version",
        "command",
        "ok",
        "warnings",
        "duration_ms",
        "result",
    ] {
        assert!(envelope.get(field).is_some(), "missing envelope {field}");
    }
    envelope.get("result").unwrap().clone()
}

fn fields(value: &JsonValue, fields: &[&str]) {
    for field in fields {
        assert!(value.get(field).is_some(), "missing {field} in {value:?}");
    }
}

fn query_option_error(
    fixture: &Fixture,
    flag: &str,
    value: &str,
    code: &str,
    reason_key: &str,
    reason: &str,
) {
    for command in ["search", "why-not"] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_coffer"));
        child
            .arg(command)
            .arg(fixture.db())
            .args(["--collection", "notes"]);
        if command == "search" {
            child.arg("private-query-text");
        } else {
            child.args([
                "--query",
                "private-query-text",
                "--chunk-external-id",
                "doc#1",
            ]);
        }
        let output = child.args([flag, value, "--json"]).output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(1),
            "{command} {reason_key}={reason}"
        );
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr).unwrap();
        for private in [
            "private-query-text",
            "private-weight-value",
            "private-boost-value",
        ] {
            assert!(!stderr.contains(private), "input leaked: {stderr}");
        }
        let envelope = parse_json(&stderr).unwrap();
        assert_eq!(envelope.get("ok").and_then(JsonValue::as_bool), Some(false));
        let error = envelope.get("error").unwrap();
        assert_eq!(error.get("code").and_then(JsonValue::as_str), Some(code));
        assert_eq!(
            error
                .get("detail")
                .unwrap()
                .get(reason_key)
                .and_then(JsonValue::as_str),
            Some(reason),
            "{command}: {stderr}"
        );
    }
}

#[cfg(unix)]
#[test]
fn retained_compact_alias_refuses_writers_and_maintenance_without_mutation() {
    for durability in [cofferdb::Durability::Fast, cofferdb::Durability::Commit] {
        let fixture = Fixture::new();
        let path = fixture.db();
        let alias = fixture.0.join("old-alias.coffer");
        let retained_alias = fixture.0.join("retained-generation.coffer");
        let writer = Database::create(
            &path,
            DatabaseConfig {
                durability,
                ..Default::default()
            },
        )
        .unwrap();
        writer.create_collection(config("notes")).unwrap();
        insert(&writer, "first");
        fs::hard_link(&path, &alias).unwrap();
        fs::hard_link(&path, &retained_alias).unwrap();
        let reader = Database::open_read_only(&alias).unwrap();
        let expected = reader
            .collection("notes")
            .unwrap()
            .search()
            .text("WalRecovery")
            .execute()
            .unwrap();
        writer.compact(CompactOptions::default()).unwrap();
        let old_bytes = fs::read(&alias).unwrap();
        let new_bytes = fs::read(&path).unwrap();
        let assert_refused = |error: cofferdb::Error| {
            assert_eq!(error.code(), ErrorCode::RecoveryError);
            let json = error.to_json();
            error_contract::validate_writer_reopen(&json).unwrap();
            assert_eq!(json.get("recoverable"), Some(&JsonValue::Bool(true)));
            assert_eq!(
                json.get("detail")
                    .unwrap()
                    .get("reason")
                    .and_then(JsonValue::as_str),
                Some("writer_reopen_required")
            );
            let action = json
                .get("suggested_action")
                .and_then(JsonValue::as_str)
                .unwrap();
            assert!(action.contains("current") && action.contains("reopen"));
            assert!(!json.to_compact_string().contains(alias.to_str().unwrap()));
            assert_eq!(fs::read(&alias).unwrap(), old_bytes);
            assert_eq!(fs::read(&path).unwrap(), new_bytes);
        };
        for closed in [false, true] {
            if closed {
                writer.clone().close().unwrap();
            }
            assert_refused(Database::open(&alias).err().unwrap());
            assert_refused(Database::repair_truncate_tail(&alias).unwrap_err());
            for (command, flags) in [
                ("compact", vec![]),
                ("repair", vec!["--truncate-tail"]),
                ("rebuild-index", vec!["--collection", "notes"]),
            ] {
                let args = ["coffer", command, alias.to_str().unwrap()]
                    .into_iter()
                    .chain(flags)
                    .map(str::to_owned)
                    .collect();
                let error = cofferdb::cli::run(args).unwrap_err();
                let output = cofferdb::cli::error_output(Some(command), &error);
                let envelope = parse_json(&output).unwrap();
                assert_eq!(
                    envelope.get("schema_version").and_then(JsonValue::as_str),
                    Some("0.1")
                );
                assert_eq!(envelope.get("ok"), Some(&JsonValue::Bool(false)));
                assert_eq!(envelope.get("error"), Some(&error.to_json()));
                assert!(!output.contains(alias.to_str().unwrap()));
                assert_refused(error);
            }
            let alias_reader = Database::open_read_only(&alias).unwrap();
            assert_eq!(alias_reader.stats().unwrap().chunk_count, 1);
            alias_reader.close().unwrap();
        }
        let current = Database::open(&path).unwrap();
        insert(&current, "second");
        assert_eq!(current.stats().unwrap().chunk_count, 2);
        let after = reader
            .collection("notes")
            .unwrap()
            .search()
            .text("WalRecovery")
            .execute()
            .unwrap();
        assert_eq!(after.results.len(), expected.results.len());
        assert_eq!(after.results[0].chunk_external_id, "first");
        assert_eq!(
            after.results[0].final_score.to_bits(),
            expected.results[0].final_score.to_bits()
        );
        current.close().unwrap();

        // Reusing the alias path for the current identity must not inherit a ban.
        fs::remove_file(&alias).unwrap();
        fs::hard_link(&path, &alias).unwrap();
        let reused = Database::open(&alias).unwrap();
        assert_eq!(reused.stats().unwrap().chunk_count, 2);
        reused.close().unwrap();

        assert_eq!(
            Database::open(&retained_alias).err().unwrap().code(),
            ErrorCode::RecoveryError
        );
        reader.close().unwrap();
        // The actual old identity is no longer banned after its final local reader closes.
        let old_writer = Database::open(&retained_alias).unwrap();
        assert_eq!(old_writer.stats().unwrap().chunk_count, 1);
        old_writer.close().unwrap();
        assert_eq!(fs::read(&retained_alias).unwrap(), old_bytes);
    }
}

#[cfg(unix)]
#[test]
fn retained_alias_repair_refuses_before_scanning_even_if_external_bytes_are_invalid() {
    let fixture = Fixture::new();
    let path = fixture.db();
    let alias = fixture.0.join("old-alias.coffer");
    let writer = Database::create(&path, DatabaseConfig::default()).unwrap();
    writer.create_collection(config("notes")).unwrap();
    insert(&writer, "first");
    fs::hard_link(&path, &alias).unwrap();
    let reader = Database::open_read_only(&alias).unwrap();
    writer.compact(CompactOptions::default()).unwrap();
    writer.close().unwrap();
    let current_bytes = fs::read(&path).unwrap();
    // External writes are outside the policy, but must not bypass known-owner admission.
    fs::write(&alias, b"externally damaged header").unwrap();
    let before = fs::read(&alias).unwrap();
    let error = Database::repair_truncate_tail(&alias).unwrap_err();
    assert_eq!(error.code(), ErrorCode::RecoveryError);
    assert!(error
        .detail()
        .contains(&("reason".into(), "writer_reopen_required".into())));
    assert_eq!(fs::read(&alias).unwrap(), before);
    assert_eq!(fs::read(&path).unwrap(), current_bytes);
    reader.close().unwrap();
    assert_eq!(
        Database::open(&alias).err().unwrap().code(),
        ErrorCode::FileFormatError
    );
}

#[cfg(unix)]
#[test]
fn retained_alias_repair_never_truncates_an_actually_repairable_tail() {
    let fixture = Fixture::new();
    let path = fixture.db();
    let alias = fixture.0.join("old-alias.coffer");
    let writer = Database::create(&path, DatabaseConfig::default()).unwrap();
    writer.create_collection(config("notes")).unwrap();
    insert(&writer, "first");
    insert(&writer, "second");
    fs::hard_link(&path, &alias).unwrap();
    let reader = Database::open_read_only(&alias).unwrap();
    let snapshot = || {
        reader
            .collection("notes")
            .unwrap()
            .search()
            .text("WalRecovery")
            .execute()
            .unwrap()
            .results
            .into_iter()
            .map(|row| (row.chunk_external_id, row.final_score.to_bits()))
            .collect::<Vec<_>>()
    };
    let expected = snapshot();
    assert_eq!(expected.len(), 2);
    writer.compact(CompactOptions::default()).unwrap();
    writer.close().unwrap();
    let current_bytes = fs::read(&path).unwrap();
    let mut old_bytes = fs::read(&alias).unwrap();
    old_bytes.extend_from_slice(b"externally appended invalid tail");
    fs::write(&alias, &old_bytes).unwrap();
    let error = Database::repair_truncate_tail(&alias).unwrap_err();
    assert_eq!(error.code(), ErrorCode::RecoveryError);
    assert!(error
        .detail()
        .contains(&("reason".into(), "writer_reopen_required".into())));
    assert_eq!(fs::read(&alias).unwrap(), old_bytes);
    assert_eq!(fs::read(&path).unwrap(), current_bytes);
    assert_eq!(snapshot(), expected);
    reader.close().unwrap();
    let repaired = Database::repair_truncate_tail(&alias).unwrap();
    assert!(repaired.repaired && repaired.bytes_removed > 0);
    assert!(fs::metadata(&alias).unwrap().len() < old_bytes.len() as u64);
    assert_eq!(fs::read(&path).unwrap(), current_bytes);
    let reopened = Database::open(&alias).unwrap();
    let results = reopened
        .collection("notes")
        .unwrap()
        .search()
        .text("WalRecovery")
        .execute()
        .unwrap()
        .results
        .into_iter()
        .map(|row| (row.chunk_external_id, row.final_score.to_bits()))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
    reopened.close().unwrap();
}

#[cfg(unix)]
#[test]
fn concurrent_alias_writer_open_and_compact_never_gain_old_file_write_access() {
    let fixture = Fixture::new();
    let path = fixture.db();
    let alias = fixture.0.join("old-alias.coffer");
    let writer = Database::create(&path, DatabaseConfig::default()).unwrap();
    writer.create_collection(config("notes")).unwrap();
    insert(&writer, "first");
    fs::hard_link(&path, &alias).unwrap();
    let reader = Database::open_read_only(&alias).unwrap();
    let old_bytes = fs::read(&alias).unwrap();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let compact = scope.spawn(|| {
            barrier.wait();
            writer.compact(CompactOptions::default()).unwrap();
        });
        barrier.wait();
        let error = Database::open(&alias).err().unwrap();
        assert!(matches!(
            error.code(),
            ErrorCode::LockContention | ErrorCode::RecoveryError
        ));
        compact.join().unwrap();
    });
    assert_eq!(fs::read(&alias).unwrap(), old_bytes);
    assert_eq!(
        Database::open(&alias).err().unwrap().code(),
        ErrorCode::RecoveryError
    );
    assert_eq!(reader.stats().unwrap().chunk_count, 1);
    writer.close().unwrap();
    reader.close().unwrap();
}

#[cfg(unix)]
#[test]
fn concurrent_final_retained_reader_close_and_alias_open_have_a_safe_admission_boundary() {
    let fixture = Fixture::new();
    let path = fixture.db();
    let alias = fixture.0.join("old-alias.coffer");
    let writer = Database::create(&path, DatabaseConfig::default()).unwrap();
    writer.create_collection(config("notes")).unwrap();
    insert(&writer, "first");
    fs::hard_link(&path, &alias).unwrap();
    let reader = Database::open_read_only(&alias).unwrap();
    writer.compact(CompactOptions::default()).unwrap();
    writer.close().unwrap();
    let old_bytes = fs::read(&alias).unwrap();
    let new_bytes = fs::read(&path).unwrap();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let close = scope.spawn(|| {
            barrier.wait();
            reader.close().unwrap();
        });
        barrier.wait();
        match Database::open(&alias) {
            Ok(writer) => {
                assert_eq!(writer.stats().unwrap().chunk_count, 1);
                writer.close().unwrap();
            }
            Err(error) => {
                assert_eq!(error.code(), ErrorCode::RecoveryError);
                assert!(error
                    .detail()
                    .contains(&("reason".into(), "writer_reopen_required".into())));
            }
        }
        close.join().unwrap();
    });
    Database::open(&alias).unwrap().close().unwrap();
    assert_eq!(fs::read(&alias).unwrap(), old_bytes);
    assert_eq!(fs::read(&path).unwrap(), new_bytes);
}

#[test]
fn real_cli_explain_weight_aliases_match_effective_weights_and_final_score() {
    let fixture = Fixture::new();
    let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    db.import_manifest(fixture.manifest(), OnConflict::Error)
        .unwrap();
    db.collection("notes")
        .unwrap()
        .put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "fixture".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
    drop(db);
    let embedding_path = fixture.0.join("query_embedding.json");
    fs::write(
        &embedding_path,
        r#"{"model_name":"fixture","dim":2,"vector":[1,0]}"#,
    )
    .unwrap();
    let before = fs::read(fixture.db()).unwrap();
    for profile in ["balanced", "constrained", "expanded"] {
        for mode in ["text", "vector", "hybrid", "vector-search"] {
            for override_weights in [false, true] {
                if mode == "vector-search" && override_weights {
                    continue;
                }
                let mut command = Command::new(env!("CARGO_BIN_EXE_coffer"));
                command
                    .arg(if mode == "vector-search" {
                        "vector-search"
                    } else {
                        "search"
                    })
                    .arg(fixture.db())
                    .args(["--collection", "notes", "--profile", profile, "--json"]);
                if mode == "text" || mode == "hybrid" {
                    command.arg("recovery");
                }
                if mode != "text" {
                    command.arg("--embedding").arg(&embedding_path);
                }
                if override_weights {
                    command.args(["--weights", r#"{"bm25":0.25,"vector":0.5,"metadata":0.25}"#]);
                }
                let output = command.output().unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let envelope = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
                let result = envelope.get("result").unwrap();
                assert_eq!(
                    result
                        .get("query_context")
                        .unwrap()
                        .get("profile")
                        .and_then(JsonValue::as_str),
                    Some(profile)
                );
                let hits = result.get("results").unwrap().as_array().unwrap();
                assert_eq!(hits.len(), 1);
                let explain = hits[0].get("explain").unwrap();
                assert_eq!(
                    explain.get("resource_profile").and_then(JsonValue::as_str),
                    Some(profile)
                );
                let weights = explain.get("weights").unwrap();
                let effective = weights.get("effective").unwrap();
                assert_eq!(weights.as_object().unwrap().len(), 7);
                let mut score = 0.0;
                for (field, score_field) in [
                    ("bm25", "bm25_norm"),
                    ("vector", "vector_norm"),
                    ("metadata", "metadata_boost"),
                ] {
                    assert_eq!(weights.get(field), effective.get(field));
                    score += effective.get(field).unwrap().as_f64().unwrap()
                        * explain
                            .get("scores")
                            .unwrap()
                            .get(score_field)
                            .unwrap()
                            .as_f64()
                            .unwrap_or(0.0);
                }
                assert!(
                    (score - hits[0].get("final_score").unwrap().as_f64().unwrap()).abs() < 1e-12
                );
                assert_eq!(
                    weights.get("source").and_then(JsonValue::as_str),
                    Some(if override_weights {
                        "user_override"
                    } else {
                        "default_no_boost"
                    })
                );
                let context = result.get("query_context").unwrap().get("weights").unwrap();
                for field in ["requested", "effective", "source", "renormalization_reason"] {
                    assert_eq!(weights.get(field), context.get(field));
                }
                if override_weights {
                    assert_ne!(weights.get("requested"), weights.get("effective"));
                    assert!(weights
                        .get("renormalization_reason")
                        .unwrap()
                        .as_str()
                        .is_some());
                } else {
                    assert_eq!(
                        weights.get("renormalization_reason"),
                        Some(&JsonValue::Null)
                    );
                }
            }
        }
    }
    assert_eq!(fs::read(fixture.db()).unwrap(), before);
}

#[test]
fn real_cli_rejects_unsupported_profiles_before_search_or_benchmark_execution() {
    let fixture = Fixture::new();
    let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    db.import_manifest(fixture.manifest(), OnConflict::Error)
        .unwrap();
    db.collection("notes")
        .unwrap()
        .put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "fixture".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
    db.close().unwrap();
    let embedding = fixture.0.join("query.json");
    fs::write(
        &embedding,
        r#"{"model_name":"fixture","dim":2,"vector":[1,0]}"#,
    )
    .unwrap();
    let report = fixture.0.join("result.json");
    fs::write(&report, "preserve existing report").unwrap();
    let before = fs::read(fixture.db()).unwrap();
    let missing_db = fixture.0.join("must-not-be-created.coffer");
    let inputs: Vec<_> = [
        fixture.manifest(),
        fixture.0.join("documents.jsonl"),
        fixture.0.join("chunks.jsonl"),
        embedding.clone(),
    ]
    .into_iter()
    .map(|path| {
        let bytes = fs::read(&path).unwrap();
        (path, bytes)
    })
    .collect();
    let command_for = |operation: &str, path: &std::path::Path, profile: &str| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_coffer"));
        command
            .arg(operation)
            .arg(path)
            .args(["--profile", profile, "--json"]);
        match operation {
            "search" => {
                command.args(["--collection", "notes", "recovery"]);
            }
            "vector-search" => {
                command
                    .args(["--collection", "notes", "--embedding"])
                    .arg(&embedding);
            }
            "why-not" => {
                command.args([
                    "--collection",
                    "notes",
                    "--chunk-id",
                    "doc#1",
                    "--query",
                    "recovery",
                ]);
            }
            "benchmark" => {
                // Also retain the early-rejection check for unavailable inputs.
                // Legal benchmark input controls are in benchmark_cli.rs.
                command
                    .arg("--dataset")
                    .arg(fixture.0.join("missing-dataset"))
                    .arg("--output")
                    .arg(&report);
            }
            _ => unreachable!(),
        }
        command
    };
    for profile in ["expanded", "balanced", "constrained"] {
        for operation in ["search", "vector-search", "why-not"] {
            let output = command_for(operation, &fixture.db(), profile)
                .output()
                .unwrap();
            assert_eq!(
                output.status.code(),
                Some(0),
                "{operation}/{profile}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.stderr.is_empty());
            let envelope = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
            assert_eq!(envelope.get("ok"), Some(&JsonValue::Bool(true)));
            assert_eq!(
                envelope
                    .get("result")
                    .unwrap()
                    .get("query_context")
                    .unwrap()
                    .get("profile")
                    .and_then(JsonValue::as_str),
                Some(profile)
            );
            assert_eq!(fs::read(fixture.db()).unwrap(), before);
        }
    }
    for profile in [
        "quality-first",
        "quality_first",
        "resource-first",
        "resource_first",
        "performance",
        "eco",
        "QualityFirst",
        "ResourceFirst",
        "Expanded",
        "Balanced",
        "Constrained",
        "EXPANDED",
        "BALANCED",
        "CONSTRAINED",
        " expanded",
        "constrained ",
        "expanded-budget",
        "",
        "unknown",
    ] {
        for path in [fixture.db(), missing_db.clone()] {
            for operation in ["search", "vector-search", "why-not", "benchmark"] {
                let output = command_for(operation, &path, profile).output().unwrap();
                assert_eq!(output.status.code(), Some(1), "{operation}/{profile:?}");
                assert!(output.stdout.is_empty());
                let envelope = parse_json(std::str::from_utf8(&output.stderr).unwrap()).unwrap();
                let error = envelope.get("error").unwrap();
                error_contract::validate_error(error).unwrap();
                assert_eq!(
                    error.get("code").and_then(JsonValue::as_str),
                    Some("cli_usage")
                );
                assert_eq!(
                    error
                        .get("detail")
                        .unwrap()
                        .get("actual")
                        .and_then(JsonValue::as_str),
                    Some(profile)
                );
                assert_eq!(fs::read(fixture.db()).unwrap(), before);
                for (input, bytes) in &inputs {
                    assert_eq!(&fs::read(input).unwrap(), bytes, "{}", input.display());
                }
                assert!(!missing_db.exists());
                assert_eq!(
                    fs::read_to_string(&report).unwrap(),
                    "preserve existing report"
                );
            }
        }
    }
}

#[test]
fn real_cli_why_not_and_profile_help_use_current_names_and_default() {
    let fixture = Fixture::new();
    let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    db.import_manifest(fixture.manifest(), OnConflict::Error)
        .unwrap();
    db.close().unwrap();
    let before = fs::read(fixture.db()).unwrap();
    for profile in [
        None,
        Some("expanded"),
        Some("balanced"),
        Some("constrained"),
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_coffer"));
        command.arg("why-not").arg(fixture.db()).args([
            "--collection",
            "notes",
            "--chunk-id",
            "doc#1",
            "--query",
            "recovery",
            "--json",
        ]);
        if let Some(profile) = profile {
            command.args(["--profile", profile]);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let envelope = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
        let result = envelope.get("result").unwrap();
        let expected = profile.unwrap_or("balanced");
        assert_eq!(
            result.get("profile").and_then(JsonValue::as_str),
            Some(expected)
        );
        assert_eq!(
            result
                .get("query_context")
                .unwrap()
                .get("profile")
                .and_then(JsonValue::as_str),
            Some(expected)
        );
    }
    let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .args(["profile-help", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let envelope = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
    let result = envelope.get("result").unwrap();
    assert_eq!(
        result.get("default").and_then(JsonValue::as_str),
        Some("balanced")
    );
    let profiles = result.get("profiles").unwrap().as_array().unwrap();
    assert_eq!(profiles.len(), 3);
    for (item, name) in profiles.iter().zip(["expanded", "balanced", "constrained"]) {
        assert_eq!(item.get("cli").and_then(JsonValue::as_str), Some(name));
        assert_eq!(item.get("json").and_then(JsonValue::as_str), Some(name));
        assert_eq!(item.get("alias"), Some(&JsonValue::Null));
    }
    let help = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .arg("benchmark")
        .arg(fixture.0.join("not-created.coffer"))
        .args(["--profile", "help", "--json"])
        .output()
        .unwrap();
    assert!(help.status.success());
    assert_eq!(
        parse_json(std::str::from_utf8(&help.stdout).unwrap())
            .unwrap()
            .get("result"),
        Some(result)
    );
    assert!(!fixture.0.join("not-created.coffer").exists());
    assert_eq!(fs::read(fixture.db()).unwrap(), before);
}

#[test]
fn six_default_weight_rows_use_the_default_path_across_rust_and_cli() {
    let fixture = Fixture::new();
    fs::write(
        fixture.0.join("documents.jsonl"),
        r#"{"external_id":"doc","metadata":{"kind":"reference"}}"#,
    )
    .unwrap();
    let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    db.import_manifest(fixture.manifest(), OnConflict::Error)
        .unwrap();
    db.collection("notes")
        .unwrap()
        .put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "fixture".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
    db.close().unwrap();
    let embedding_path = fixture.0.join("query_embedding.json");
    fs::write(
        &embedding_path,
        r#"{"model_name":"fixture","dim":2,"vector":[1,0]}"#,
    )
    .unwrap();
    let boost_json = r#"{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{"when":{"doc.kind":"reference"},"boost":0.4}]}"#;
    let before = fs::read(fixture.db()).unwrap();
    let db = Database::open_read_only(fixture.db()).unwrap();
    let collection = db.collection("notes").unwrap();
    // Independent constants from requirements 8.9, not default_for() output.
    for (text, vector, boost, expected) in [
        (true, true, false, [0.5, 0.5, 0.0]),
        (true, false, false, [1.0, 0.0, 0.0]),
        (false, true, false, [0.0, 1.0, 0.0]),
        (true, true, true, [0.45, 0.5, 0.05]),
        (true, false, true, [0.95, 0.0, 0.05]),
        (false, true, true, [0.0, 0.95, 0.05]),
    ] {
        let weights = parse_json(&format!(
            r#"{{"bm25":{},"vector":{},"metadata":{}}}"#,
            expected[0], expected[1], expected[2]
        ))
        .unwrap();
        for profile in [
            ResourceProfile::Balanced,
            ResourceProfile::Expanded,
            ResourceProfile::Constrained,
        ] {
            // Equal numbers supplied explicitly must still report user_override.
            for explicit in [false, true] {
                let mut builder = collection.search().resource_profile(profile).top_k(10);
                let mut command = Command::new(env!("CARGO_BIN_EXE_coffer"));
                command.arg("search").arg(fixture.db()).args([
                    "--collection",
                    "notes",
                    "--profile",
                    profile.as_cli_str(),
                    "--json",
                ]);
                if text {
                    builder = builder.text("recovery");
                    command.arg("recovery");
                }
                if vector {
                    builder =
                        builder.query_embedding(QueryEmbeddingInput::from_vector(vec![1.0, 0.0]));
                    command.arg("--embedding").arg(&embedding_path);
                }
                if boost {
                    builder = builder.metadata_boost(
                        MetadataBoostConfig::parse(parse_json(boost_json).unwrap()).unwrap(),
                    );
                    command.args(["--metadata-boost", boost_json]);
                }
                if explicit {
                    builder = builder.weights(
                        cofferdb::SearchWeights::new(expected[0], expected[1], expected[2])
                            .unwrap(),
                    );
                    command.arg("--weights").arg(weights.to_compact_string());
                }
                let rust_result = builder.execute().unwrap().to_json();
                let output = command.output().unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let envelope = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
                for result in [&rust_result, envelope.get("result").unwrap()] {
                    let context = result.get("query_context").unwrap().get("weights").unwrap();
                    assert_eq!(context.get("requested"), Some(&weights));
                    assert_eq!(context.get("effective"), Some(&weights));
                    let source = if explicit {
                        "user_override"
                    } else if boost {
                        "default_with_boost"
                    } else {
                        "default_no_boost"
                    };
                    assert_eq!(
                        context.get("source").and_then(JsonValue::as_str),
                        Some(source)
                    );
                    assert_eq!(
                        context.get("renormalization_reason"),
                        Some(&JsonValue::Null)
                    );
                    let hits = result.get("results").unwrap().as_array().unwrap();
                    assert_eq!(hits.len(), 1);
                    let explain = hits[0].get("explain").unwrap();
                    let actual_weights = explain.get("weights").unwrap();
                    for field in ["requested", "effective", "source", "renormalization_reason"] {
                        assert_eq!(actual_weights.get(field), context.get(field));
                    }
                    for field in ["bm25", "vector", "metadata"] {
                        assert_eq!(actual_weights.get(field), weights.get(field));
                    }
                    // Single matching chunk: BM25 norm = 1, cosine norm = 1;
                    // the matching metadata rule contributes 0.4 before weighting.
                    let expected_score = expected[0] + expected[1] + expected[2] * 0.4;
                    assert!(
                        (hits[0].get("final_score").unwrap().as_f64().unwrap() - expected_score)
                            .abs()
                            < 1e-12
                    );
                    assert!(
                        (explain
                            .get("scores")
                            .unwrap()
                            .get("final_score")
                            .unwrap()
                            .as_f64()
                            .unwrap()
                            - expected_score)
                            .abs()
                            < 1e-12
                    );
                }
            }
        }
    }
    drop(collection);
    db.close().unwrap();
    assert_eq!(fs::read(fixture.db()).unwrap(), before);
}

#[test]
fn query_option_weights_reasons_cover_cli_schema_syntax_and_json_limits() {
    let fixture = Fixture::new();
    let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    db.import_manifest(fixture.manifest(), OnConflict::Error)
        .unwrap();
    drop(db);
    let before = fs::read(fixture.db()).unwrap();
    for (value, reason) in [
        ("{", "invalid_json"),
        (r#"{"bm25":1e400}"#, "invalid_json"),
        (r#"{"bm25":1,"bm25":1}"#, "invalid_json"),
        ("null", "invalid_type"),
        ("[]", "invalid_type"),
        ("{}", "missing_required_field"),
        (r#"{"bm25":1}"#, "missing_required_field"),
        (r#"{"bm25":1,"vector":0}"#, "missing_required_field"),
        (
            r#"{"bm25":"private-weight-value","vector":0,"metadata":0}"#,
            "invalid_type",
        ),
        (r#"{"bm25":1,"vector":false,"metadata":0}"#, "invalid_type"),
        (r#"{"bm25":1,"vector":0,"metadata":null}"#, "invalid_type"),
        (
            r#"{"bm25":1,"vector":0,"metadata":0,"debug":true}"#,
            "unsupported_field",
        ),
        (
            r#"{"bm25":-1,"vector":2,"metadata":0}"#,
            "finite_non_negative",
        ),
        (r#"{"bm25":0,"vector":0,"metadata":0}"#, "all_zero"),
        (r#"{"bm25":1,"vector":1,"metadata":0}"#, "sum_mismatch"),
        (
            r#"{"bm25":0,"vector":1,"metadata":0}"#,
            "effective_sum_zero",
        ),
        (
            r#"{"bm25":0,"vector":0,"metadata":1}"#,
            "effective_sum_zero",
        ),
    ] {
        query_option_error(
            &fixture,
            "--weights",
            value,
            "invalid_search_weights",
            "weights_reason",
            reason,
        );
    }
    let too_deep = format!("{}0{}", "[".repeat(129), "]".repeat(129));
    query_option_error(
        &fixture,
        "--weights",
        &too_deep,
        "invalid_search_weights",
        "weights_reason",
        "json_processing_limit",
    );
    assert_eq!(fs::read(fixture.db()).unwrap(), before);
    let result = cli(&[
        "search",
        fixture.db().to_str().unwrap(),
        "recovery",
        "--collection",
        "notes",
        "--weights",
        r#"{"bm25":1,"vector":0,"metadata":0}"#,
    ]);
    assert_eq!(result.get("results").unwrap().as_array().unwrap().len(), 1);
}

#[test]
fn query_option_boost_reasons_agree_across_rust_inline_and_file_inputs() {
    let fixture = Fixture::new();
    let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    db.import_manifest(fixture.manifest(), OnConflict::Error)
        .unwrap();
    drop(db);
    let before = fs::read(fixture.db()).unwrap();
    let valid = parse_json(r#"{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{"when":{"chunk.kind":"rust"},"boost":0.5}]}"#).unwrap();
    let file = fixture.0.join("boost.json");
    for (field, replacement, reason) in [
        ("schema_version", None, "missing_required_field"),
        ("schema_version", Some(r#""0.2""#), "unsupported_version"),
        ("schema_version", Some("null"), "unsupported_version"),
        ("combine", None, "missing_required_field"),
        ("combine", Some(r#""product""#), "unsupported_combine"),
        ("combine", Some("0"), "unsupported_combine"),
        ("max_boost", None, "missing_required_field"),
        (
            "max_boost",
            Some(r#""private-boost-value""#),
            "invalid_type",
        ),
        ("max_boost", Some("-1"), "finite_non_negative"),
        ("debug", Some("true"), "unsupported_field"),
        ("rules", None, "missing_required_field"),
        ("rules", Some("null"), "invalid_type"),
        ("rules", Some("[]"), "empty"),
        ("rules", Some("[null]"), "invalid_type"),
        (
            "rules",
            Some(r#"[{"when":{"chunk.kind":"rust"},"boost":1,"debug":true}]"#),
            "unsupported_field",
        ),
        ("rules", Some(r#"[{"boost":1}]"#), "missing_required_field"),
        (
            "rules",
            Some(r#"[{"when":null,"boost":1}]"#),
            "empty_filter",
        ),
        (
            "rules",
            Some(r#"[{"when":[],"boost":1}]"#),
            "invalid_filter",
        ),
        ("rules", Some(r#"[{"when":{},"boost":1}]"#), "empty_filter"),
        (
            "rules",
            Some(r#"[{"when":{"chunk.kind":{"$eq":{"secret":"private-boost-value"}}},"boost":1}]"#),
            "invalid_filter",
        ),
        (
            "rules",
            Some(r#"[{"when":{"chunk.kind":"rust"}}]"#),
            "missing_required_field",
        ),
        (
            "rules",
            Some(r#"[{"when":{"chunk.kind":"rust"},"boost":null}]"#),
            "invalid_type",
        ),
        (
            "rules",
            Some(r#"[{"when":{"chunk.kind":"rust"},"boost":-1}]"#),
            "finite_non_negative",
        ),
    ] {
        let mut value = valid.clone();
        let object = value.as_object_mut().unwrap();
        if let Some(replacement) = replacement {
            object.insert(field.into(), parse_json(replacement).unwrap());
        } else {
            object.remove(field);
        }
        let error = MetadataBoostConfig::parse(value.clone()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataBoost);
        assert_eq!(
            error
                .to_json()
                .get("detail")
                .unwrap()
                .get("metadata_boost_reason")
                .and_then(JsonValue::as_str),
            Some(reason)
        );
        let wire = value.to_compact_string();
        fs::write(&file, &wire).unwrap();
        for input in [wire.as_str(), file.to_str().unwrap()] {
            query_option_error(
                &fixture,
                "--metadata-boost",
                input,
                "invalid_metadata_boost",
                "metadata_boost_reason",
                reason,
            );
        }
    }
    for (wire, reason) in [
        ("null".to_owned(), "invalid_type"),
        ("[]".to_owned(), "invalid_type"),
        ("{bad json".to_owned(), "invalid_json"),
        (
            format!(r#"{{"rules":{}0{}}}"#, "[".repeat(129), "]".repeat(129)),
            "json_processing_limit",
        ),
    ] {
        fs::write(&file, &wire).unwrap();
        query_option_error(
            &fixture,
            "--metadata-boost",
            file.to_str().unwrap(),
            "invalid_metadata_boost",
            "metadata_boost_reason",
            reason,
        );
        if wire.starts_with('{') {
            query_option_error(
                &fixture,
                "--metadata-boost",
                &wire,
                "invalid_metadata_boost",
                "metadata_boost_reason",
                reason,
            );
        } else {
            let error = MetadataBoostConfig::parse(parse_json(&wire).unwrap()).unwrap_err();
            assert_eq!(
                error
                    .to_json()
                    .get("detail")
                    .unwrap()
                    .get("metadata_boost_reason")
                    .and_then(JsonValue::as_str),
                Some(reason)
            );
        }
    }
    assert_eq!(fs::read(fixture.db()).unwrap(), before);
    assert_eq!(MetadataBoostConfig::parse(valid).unwrap().rule_count(), 1);
}

#[test]
fn tokenizer_creation_inputs_reopen_compact_export_and_query_contracts() {
    let fixture = Fixture::new();
    let expected = TokenizerConfig::default_en().to_json();
    let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    db.create_collection(config("api")).unwrap();
    let summary = db
        .import_manifest(fixture.manifest(), OnConflict::Error)
        .unwrap();
    assert_eq!(summary.to_json().get("tokenizer"), Some(&expected));
    assert_eq!(
        summary.to_json().get("skipped_files"),
        Some(&JsonValue::Array(Vec::new()))
    );
    db.collection("notes")
        .unwrap()
        .put_embeddings(vec![EmbeddingInput {
            chunk_external_id: "doc#1".into(),
            model_name: "m".into(),
            dim: 2,
            vector: vec![1.0, 0.0],
        }])
        .unwrap();
    drop(db);
    let path = fixture.db();
    let path = path.to_str().unwrap();
    let created = cli(&[
        "create-collection",
        path,
        "cli",
        "--dim",
        "2",
        "--tokenizer",
        "default_en",
    ]);
    assert_eq!(created.get("tokenizer"), Some(&expected));
    let db = Database::open(fixture.db()).unwrap();
    for info in db.list_collections().unwrap() {
        assert_eq!(info.tokenizer.to_json(), expected);
    }
    let col = db.collection("notes").unwrap();
    for query in [
        "wal", "recovery", "page", "cache", "storage", "rs", "café", "中文",
    ] {
        let results = col.search().text(query).execute().unwrap();
        assert_eq!(results.results.len(), 1, "{query}");
        assert_eq!(
            results.query_context.to_json().get("tokenizer"),
            Some(&expected)
        );
        let explain = &results.results[0].explain;
        assert_eq!(explain.get("tokenizer"), Some(&expected));
        let terms = explain
            .get("bm25")
            .unwrap()
            .get("terms")
            .unwrap()
            .as_array()
            .unwrap();
        let contribution: f64 = terms
            .iter()
            .map(|term| {
                fields(
                    term,
                    &[
                        "term",
                        "tf",
                        "df",
                        "chunk_length",
                        "avg_chunk_length",
                        "raw_contribution",
                    ],
                );
                assert_eq!(term.get("df").and_then(JsonValue::as_f64), Some(1.0));
                term.get("raw_contribution")
                    .and_then(JsonValue::as_f64)
                    .unwrap()
            })
            .sum();
        assert!((contribution - results.results[0].bm25_score.unwrap()).abs() < 1e-9);
    }
    assert!(col
        .search()
        .text("the and with")
        .execute()
        .unwrap()
        .results
        .is_empty());
    let vector = col
        .search()
        .query_embedding(QueryEmbeddingInput {
            model_name: Some("m".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        })
        .execute()
        .unwrap();
    assert_eq!(
        vector.query_context.to_json().get("tokenizer"),
        Some(&JsonValue::Null)
    );
    db.compact(CompactOptions {
        rebuild_indexes: true,
    })
    .unwrap();
    let out = fixture.0.join("export");
    db.export_collection_jsonl("notes", &out).unwrap();
    drop(col);
    drop(db);
    let manifest =
        parse_json(&fs::read_to_string(out.join("export_manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest.get("tokenizer"), Some(&expected));
    let roundtrip = Database::create(
        fixture.0.join("roundtrip.coffer"),
        DatabaseConfig::default(),
    )
    .unwrap();
    roundtrip
        .import_manifest(out.join("export_manifest.json"), OnConflict::Error)
        .unwrap();
    assert_eq!(
        roundtrip
            .collection("notes")
            .unwrap()
            .search()
            .text("page recovery")
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
}

#[test]
fn tokenizer_invalid_identity_fails_before_any_write_for_new_and_existing_collection() {
    let fixture = Fixture::new();
    let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    let mut invalid = vec![JsonValue::Null];
    for field in ["name", "version", "profile", "language_hint", "config"] {
        let mut value = TokenizerConfig::default().to_json();
        value.as_object_mut().unwrap().remove(field);
        invalid.push(value);
    }
    let mut custom = TokenizerConfig::default().to_json();
    custom
        .as_object_mut()
        .unwrap()
        .get_mut("config")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert("stop_words".into(), JsonValue::Array(vec![]));
    invalid.push(custom);
    for existing in [false, true] {
        if existing {
            db.create_collection(config("notes")).unwrap();
        }
        for value in &invalid {
            fixture.set_tokenizer(value.clone());
            let before = fs::read(fixture.db()).unwrap();
            assert_eq!(
                db.import_manifest(fixture.manifest(), OnConflict::Error)
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidJsonlSchema
            );
            assert_eq!(fs::read(fixture.db()).unwrap(), before);
        }
    }
}

#[test]
fn reader_snapshot_coexists_with_writer_and_survives_compact() {
    let fixture = Fixture::new();
    let writer = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    writer.create_collection(config("notes")).unwrap();
    insert(&writer, "first");
    let reader = Database::open_read_only(fixture.db()).unwrap();
    assert_eq!(
        Database::open(fixture.db()).err().unwrap().code(),
        ErrorCode::LockContention
    );
    insert(&writer, "second");
    assert_eq!(
        reader
            .collection("notes")
            .unwrap()
            .search()
            .text("recovery")
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
    let latest = Database::open_read_only(fixture.db()).unwrap();
    assert_eq!(
        latest
            .collection("notes")
            .unwrap()
            .search()
            .text("recovery")
            .execute()
            .unwrap()
            .results
            .len(),
        2
    );
    assert_eq!(
        reader
            .create_collection(config("denied"))
            .err()
            .unwrap()
            .code(),
        ErrorCode::PermissionDenied
    );
    assert_eq!(
        reader
            .compact(CompactOptions::default())
            .unwrap_err()
            .code(),
        ErrorCode::PermissionDenied
    );
    writer
        .compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
    assert_eq!(
        reader
            .collection("notes")
            .unwrap()
            .get_chunk_by_external_id("first")
            .unwrap()
            .text,
        "WalRecovery page_cache"
    );
    assert_eq!(
        reader
            .collection("notes")
            .unwrap()
            .search()
            .text("cache")
            .execute()
            .unwrap()
            .results
            .len(),
        1
    );
    let path = fixture.db();
    assert_eq!(
        cli(&["stats", path.to_str().unwrap()])
            .get("document_count")
            .and_then(JsonValue::as_f64),
        Some(2.0)
    );
}

#[test]
fn cli_json_required_fields_cover_p0_lifecycle_and_read_only_operations() {
    let fixture = Fixture::new();
    let path = fixture.db();
    let path = path.to_str().unwrap();
    fields(
        &cli(&["init", path]),
        &["path", "format_version", "created"],
    );
    fields(
        &cli(&[
            "import",
            path,
            "--manifest",
            fixture.manifest().to_str().unwrap(),
        ]),
        &[
            "documents_imported",
            "chunks_imported",
            "embeddings_imported",
            "skipped",
            "skipped_files",
            "replaced",
            "errors",
            "manifest",
            "tokenizer",
            "durability",
        ],
    );
    fields(&cli(&["list-collections", path]), &["collections"]);
    let statistics = [
        "document_count",
        "chunk_count",
        "embedding_count",
        "deleted_count",
        "index_size_bytes",
        "data_size_bytes",
        "last_commit_id",
        "format_version",
        "active_profile",
        "cache_budget_mb",
        "background_compaction_enabled",
        "pending_compaction_bytes",
        "index_stats",
    ];
    fields(
        &cli(&[
            "collection-stats",
            path,
            "--collection",
            "notes",
            "--include-index",
        ]),
        &statistics,
    );
    fields(&cli(&["stats", path, "--include-index"]), &statistics);
    fields(
        &cli(&["search", path, "--collection", "notes", "recovery"]),
        &[
            "query",
            "top_k",
            "results",
            "profile",
            "candidate_counts",
            "query_context",
        ],
    );
    fields(
        &cli(&[
            "get-document",
            path,
            "--collection",
            "notes",
            "--external-id",
            "doc",
            "--include-chunks",
        ]),
        &[
            "document",
            "document_id",
            "external_id",
            "chunks_count",
            "chunks",
        ],
    );
    fields(
        &cli(&[
            "get-chunk",
            path,
            "--collection",
            "notes",
            "--chunk-external-id",
            "doc#1",
        ]),
        &[
            "chunk",
            "chunk_id",
            "chunk_external_id",
            "document",
            "document_id",
            "external_id",
            "embedding_present",
        ],
    );
    fields(
        &cli(&["check", path]),
        &[
            "format_version",
            "last_valid_commit",
            "issues",
            "repair_available",
        ],
    );
    fields(
        &cli(&[
            "export",
            path,
            fixture.0.join("out").to_str().unwrap(),
            "--collection",
            "notes",
            "--format",
            "jsonl",
        ]),
        &[
            "output_dir",
            "collection",
            "documents_exported",
            "chunks_exported",
            "embeddings_exported",
            "manifest",
        ],
    );
    fields(
        &cli(&[
            "delete-chunk",
            path,
            "--collection",
            "notes",
            "--chunk-external-id",
            "doc#1",
        ]),
        &[
            "chunk_id",
            "chunk_external_id",
            "embedding_deleted",
            "tombstone_id",
        ],
    );
    fields(
        &cli(&[
            "delete-document",
            path,
            "--collection",
            "notes",
            "--external-id",
            "doc",
        ]),
        &[
            "document_id",
            "external_id",
            "chunks_deleted",
            "embeddings_deleted",
            "tombstone_id",
        ],
    );
    fields(
        &cli(&["compact", path]),
        &[
            "before_file_size_bytes",
            "after_file_size_bytes",
            "records_removed",
            "tombstones_removed",
            "segments_merged",
        ],
    );
    fields(
        &cli(&["drop-collection", path, "--collection", "notes"]),
        &[
            "collection",
            "documents_deleted",
            "chunks_deleted",
            "embeddings_deleted",
            "tombstone_id",
        ],
    );
}

#[test]
fn cli_human_output_and_invalid_arguments_do_not_write() {
    let fixture = Fixture::new();
    let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .arg("init")
        .arg(fixture.db())
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8(output.stdout)
        .unwrap()
        .starts_with("init\n"));
    let before = fs::read(fixture.db()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .arg("import")
        .arg(fixture.db())
        .arg("--manifest")
        .arg(fixture.manifest())
        .arg("--unknown")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .starts_with("cli_usage:"));
    assert_eq!(fs::read(fixture.db()).unwrap(), before);
}

#[test]
fn jsonl_missing_metadata_and_invalid_utf8_are_atomic() {
    let fixture = Fixture::new();
    let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
    fs::write(
        fixture.0.join("documents.jsonl"),
        r#"{"external_id":"doc"}"#,
    )
    .unwrap();
    assert_eq!(
        db.import_manifest(fixture.manifest(), OnConflict::Error)
            .unwrap_err()
            .code(),
        ErrorCode::MissingRequiredField
    );
    assert!(db.list_collections().unwrap().is_empty());
    fs::write(
        fixture.0.join("documents.jsonl"),
        b"{\"external_id\":\"doc\",\"metadata\":{}}\xff",
    )
    .unwrap();
    let err = db
        .import_manifest(fixture.manifest(), OnConflict::Error)
        .unwrap_err();
    assert!(err.detail().iter().any(|(field, _)| field == "path"));
    assert!(err
        .detail()
        .iter()
        .any(|(field, value)| field == "line" && value == "1"));
    assert!(db.list_collections().unwrap().is_empty());
}

#[test]
fn profile_defaults_and_zero_cache_import_use_independent_working_memory() {
    let fixture = Fixture::new();
    let db = Database::create(
        fixture.db(),
        DatabaseConfig {
            resource_profile: ResourceProfile::Constrained,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(db.stats().unwrap().cache_budget_mb, 64);
    drop(db);
    let db = Database::open_with_options(
        fixture.db(),
        DatabaseOpenOptions {
            cache_budget_mb: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let before = fs::read(fixture.db()).unwrap();
    let summary = db
        .import_manifest(fixture.manifest(), OnConflict::Error)
        .unwrap();
    assert_eq!(
        (
            summary.documents_inserted,
            summary.chunks_inserted,
            summary.embeddings_inserted
        ),
        (1, 1, 0)
    );
    assert_eq!(db.stats().unwrap().cache_budget_mb, 0);
    let notes = db.collection("notes").unwrap();
    let stats = notes.stats().unwrap();
    assert_eq!(
        (
            stats.document_count,
            stats.chunk_count,
            stats.embedding_count
        ),
        (1, 1, 0)
    );
    let hits = notes.search().text("WalRecovery").execute().unwrap();
    assert_eq!(hits.results.len(), 1);
    assert_eq!(hits.results[0].chunk_external_id, "doc#1");
    drop((hits, notes));
    let committed = fs::read(fixture.db()).unwrap();
    assert_ne!(committed, before);
    assert_eq!(
        db.import_manifest(fixture.manifest(), OnConflict::Error)
            .unwrap_err()
            .code(),
        ErrorCode::DuplicateExternalId
    );
    assert_eq!(fs::read(fixture.db()).unwrap(), committed);
    drop(db);
    let db = Database::open_with_options(
        fixture.db(),
        DatabaseOpenOptions {
            resource_profile: Some(ResourceProfile::Expanded),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(db.stats().unwrap().cache_budget_mb, 512);
    assert_eq!(
        db.collection("notes").unwrap().stats().unwrap().chunk_count,
        1
    );
    assert_eq!(
        Database::open_with_options(
            fixture.db(),
            DatabaseOpenOptions {
                background_compaction: Some(true),
                ..Default::default()
            }
        )
        .err()
        .unwrap()
        .code(),
        ErrorCode::UnsupportedFeatureFlag
    );
}

#[test]
fn embedding_cli_validates_before_commit_and_persists_batched_conflict_results() {
    const CHUNKS: usize = 1_001;

    fn row(id: &str, vector: &str) -> String {
        format!(
            "{{\"chunk_external_id\":\"{id}\",\"model_name\":\"m\",\"dim\":2,\"vector\":{vector}}}\n"
        )
    }

    fn run(fixture: &Fixture, durability: &str, conflict: Option<&str>, exit: i32) -> JsonValue {
        let mut command = Command::new(env!("CARGO_BIN_EXE_coffer"));
        command
            .arg("import-embeddings")
            .arg(fixture.db())
            .arg(fixture.0.join("embeddings.jsonl"))
            .args([
                "--collection",
                "notes",
                "--durability",
                durability,
                "--json",
            ]);
        if let Some(conflict) = conflict {
            command.args(["--on-conflict", conflict]);
        }
        let output = command.output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(exit),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let (json, other) = if exit == 0 {
            (&output.stdout, &output.stderr)
        } else {
            (&output.stderr, &output.stdout)
        };
        assert!(
            other.is_empty(),
            "unexpected CLI output on the other stream"
        );
        let value = parse_json(std::str::from_utf8(json).unwrap()).unwrap();
        assert_eq!(value.get("ok"), Some(&JsonValue::Bool(exit == 0)));
        assert_eq!(
            value.get("command").and_then(JsonValue::as_str),
            Some("import-embeddings")
        );
        value
    }

    fn winner(db: &Database) -> String {
        let result = db
            .collection("notes")
            .unwrap()
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("m".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .top_k(1)
            .execute()
            .unwrap();
        assert_eq!(
            result.query_context.execution_stats.vector_scan_count,
            CHUNKS
        );
        assert_eq!(result.results[0].vector_score, Some(1.0));
        result.results[0].chunk_external_id.clone()
    }

    for durability in ["commit", "fast"] {
        let fixture = Fixture::new();
        let db = Database::create(fixture.db(), DatabaseConfig::default()).unwrap();
        let mut collection_config = config("notes");
        collection_config.embedding_model = Some("m".into());
        let collection = db.create_collection(collection_config).unwrap();
        collection
            .insert_document(DocumentInput {
                external_id: "document".into(),
                title: None,
                source_uri: None,
                metadata: JsonValue::object(),
                chunks: (0..CHUNKS)
                    .map(|ordinal| ChunkInput {
                        chunk_external_id: format!("c{ordinal}"),
                        ordinal,
                        text: "topic retrieval".into(),
                        metadata: JsonValue::object(),
                        source_span: None,
                    })
                    .collect(),
            })
            .unwrap();
        drop(collection);
        let initial_commit = db.stats().unwrap().last_commit_id;
        db.close().unwrap();
        let input = fixture.0.join("embeddings.jsonl");
        fs::write(
            &input,
            (0..CHUNKS)
                .map(|index| {
                    row(
                        &format!("c{index}"),
                        if index == 0 { "[1,0]" } else { "[0,1]" },
                    )
                })
                .collect::<String>(),
        )
        .unwrap();
        let value = run(&fixture, durability, Some("error"), 0);
        let summary = value.get("result").unwrap();
        assert_eq!(
            summary.get("durability").and_then(JsonValue::as_str),
            Some(durability)
        );
        assert_eq!(
            summary
                .get("embeddings_inserted")
                .and_then(JsonValue::as_f64),
            Some(CHUNKS as f64)
        );
        let db = Database::open_read_only(fixture.db()).unwrap();
        assert!(db.stats().unwrap().last_commit_id > initial_commit + 1);
        assert_eq!(
            db.collection("notes")
                .unwrap()
                .stats()
                .unwrap()
                .embedding_count,
            CHUNKS
        );
        assert_eq!(winner(&db), "c0");
        db.close().unwrap();

        let before = fs::read(fixture.db()).unwrap();
        let replacements = (0..CHUNKS)
            .map(|index| row(&format!("c{index}"), "[0,1]"))
            .collect::<String>();
        // A valid prefix larger than a commit batch still cannot escape tail validation.
        for (tail, conflict, code) in [
            ("not JSON\n".to_owned(), None, "invalid_jsonl_schema"),
            (row("missing", "[1,0]"), None, "chunk_not_found"),
            (row("missing", "[1,0]"), Some("skip"), "chunk_not_found"),
            (row("c0", "[1,0]"), Some("skip"), "duplicate_external_id"),
        ] {
            fs::write(&input, replacements.clone() + &tail).unwrap();
            let value = run(&fixture, durability, conflict, 1);
            let error = value.get("error").unwrap();
            error_contract::validate_error(error).unwrap();
            assert_eq!(error.get("code").and_then(JsonValue::as_str), Some(code));
            assert_eq!(fs::read(fixture.db()).unwrap(), before);
            assert!(!value.to_compact_string().contains("not JSON"));
        }

        fs::write(&input, replacements).unwrap();
        let value = run(&fixture, durability, Some("skip"), 0);
        assert_eq!(
            value
                .get("result")
                .unwrap()
                .get("embeddings_skipped")
                .and_then(JsonValue::as_f64),
            Some(CHUNKS as f64)
        );
        assert_eq!(fs::read(fixture.db()).unwrap(), before);

        fs::write(
            &input,
            (0..CHUNKS)
                .map(|index| {
                    row(
                        &format!("c{index}"),
                        if index + 1 == CHUNKS {
                            "[1,0]"
                        } else {
                            "[0,1]"
                        },
                    )
                })
                .collect::<String>(),
        )
        .unwrap();
        let value = run(&fixture, durability, None, 0);
        assert_eq!(
            value
                .get("result")
                .unwrap()
                .get("embeddings_replaced")
                .and_then(JsonValue::as_f64),
            Some(CHUNKS as f64)
        );
        let db = Database::open(fixture.db()).unwrap();
        assert_eq!(winner(&db), "c1000");
        assert_eq!(
            db.collection("notes")
                .unwrap()
                .stats()
                .unwrap()
                .embedding_count,
            CHUNKS
        );
        db.compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
        assert_eq!(winner(&db), "c1000");
        db.close().unwrap();
        let db = Database::open_read_only(fixture.db()).unwrap();
        assert_eq!(winner(&db), "c1000");
        db.close().unwrap();
    }
}
