use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use cofferdb::{
    parse_json, CollectionConfig, Database, DatabaseConfig, DistanceMetric, JsonValue, OnConflict,
    QueryEmbeddingInput, VectorIndexConfig,
};

fn temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("cofferdb-e2e-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn uncached_vector_checkpoint_rewrite_preserves_reader_snapshots_and_maintenance_results() {
    use cofferdb::{
        CheckOptions, ChunkInput, CompactOptions, DatabaseOpenOptions, DocumentInput,
        EmbeddingInput,
    };

    const DIM: usize = 1025;
    let config = |name: &str| CollectionConfig {
        name: name.into(),
        tokenizer: Default::default(),
        embedding_dim: DIM,
        distance_metric: DistanceMetric::DotProduct,
        embedding_model: Some("test-model".into()),
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    };
    let dir = temp_dir("uncached-vector-maintenance");
    let path = dir.join("data.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let collection = db.create_collection(config("vectors")).unwrap();
    collection
        .insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: (0..64)
                .map(|index| ChunkInput {
                    chunk_external_id: format!("chunk-{index:02}"),
                    ordinal: index,
                    text: "shared recovery checkpoint".into(),
                    metadata: JsonValue::object(),
                    source_span: None,
                })
                .collect(),
        })
        .unwrap();
    collection
        .put_embeddings(
            (0..64)
                .map(|index| {
                    let mut vector = vec![0.0; DIM];
                    vector[0] = (index + 1) as f32 / 64.0;
                    EmbeddingInput {
                        chunk_external_id: format!("chunk-{index:02}"),
                        model_name: "test-model".into(),
                        dim: DIM,
                        vector,
                    }
                })
                .collect(),
        )
        .unwrap();
    drop(collection);
    drop(db);

    let options = |read_only| DatabaseOpenOptions {
        read_only,
        cache_budget_mb: Some(1),
        ..Default::default()
    };
    let query = |db: &Database, source: &str| {
        let mut vector = vec![0.0; DIM];
        vector[0] = 1.0;
        let results = db
            .collection("vectors")
            .unwrap()
            .search()
            .query_embedding(QueryEmbeddingInput {
                model_name: Some("test-model".into()),
                dim: DIM,
                vector,
            })
            .top_k(7)
            .execute()
            .unwrap();
        assert_eq!(results.query_context.index_source, source);
        assert_eq!(results.query_context.execution_stats.vector_scan_count, 64);
        assert_eq!(
            results
                .results
                .iter()
                .map(|hit| hit.chunk_external_id.as_str())
                .collect::<Vec<_>>(),
            ["chunk-63", "chunk-62", "chunk-61", "chunk-60", "chunk-59", "chunk-58", "chunk-57"]
        );
        results
            .results
            .into_iter()
            .map(|hit| (hit.chunk_external_id, hit.final_score.to_bits()))
            .collect::<Vec<_>>()
    };
    // 64 * 1025 f32 values alone exceed the 1-MiB cache's 1/8 decoded admission target.
    let writer = Database::open_with_options(&path, options(false)).unwrap();
    let reader = Database::open_with_options(&path, options(true)).unwrap();
    let expected = query(&writer, "persistent_checkpoint");
    assert_eq!(query(&reader, "persistent_checkpoint"), expected);
    writer.create_collection(config("empty")).unwrap();
    assert_eq!(writer.list_collections().unwrap().len(), 2);
    assert_eq!(reader.list_collections().unwrap().len(), 1);
    assert_eq!(query(&writer, "persistent_checkpoint"), expected);
    assert_eq!(query(&reader, "transient_rebuild"), expected);

    writer.rebuild_index(None).unwrap();
    assert_eq!(query(&writer, "persistent_checkpoint"), expected);
    let report = writer
        .check(CheckOptions {
            verify_indexes: true,
        })
        .unwrap();
    assert_eq!(report.status, "ok");
    assert!(report.issues.is_empty());
    writer
        .compact(CompactOptions {
            rebuild_indexes: true,
        })
        .unwrap();
    assert_eq!(query(&writer, "persistent_checkpoint"), expected);
    assert_eq!(query(&reader, "transient_rebuild"), expected);
    drop(writer);
    let reopened = Database::open_with_options(&path, options(true)).unwrap();
    assert_eq!(reopened.list_collections().unwrap().len(), 2);
    assert_eq!(query(&reopened, "persistent_checkpoint"), expected);
    assert_eq!(query(&reader, "transient_rebuild"), expected);
    drop(reopened);
    drop(reader);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn index_segment_counts_follow_checkpoint_availability_even_without_terms() {
    use cofferdb::{CheckOptions, ChunkInput, CompactOptions, DocumentInput};

    fn verify(db: &Database, path: &std::path::Path, ready: bool) {
        let stats = db.index_stats().unwrap();
        let segments = if ready { 2 } else { 0 };
        assert_eq!(stats.text_segments, segments);
        assert_eq!(stats.vector_segments, segments);
        assert_eq!(stats.rebuild_needed, !ready);
        assert_eq!(stats.last_index_commit_id.is_some(), ready);
        for name in ["empty", "stopwords"] {
            let stats = db.collection(name).unwrap().stats().unwrap();
            assert_eq!(stats.text_index_terms, 0);
            assert_eq!(stats.text_index_segments, usize::from(ready));
        }
        let before = fs::read(path).unwrap();
        for command in ["stats", "collection-stats"] {
            let mut cli = Command::new(env!("CARGO_BIN_EXE_coffer"));
            cli.arg(command)
                .arg(path)
                .args(["--include-index", "--json"]);
            if command == "collection-stats" {
                cli.args(["--collection", "empty"]);
            }
            let output = cli.output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let envelope = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
            let result = envelope.get("result").unwrap();
            for field in ["text_segments", "vector_segments"] {
                assert_eq!(
                    result
                        .get("index_stats")
                        .unwrap()
                        .get(field)
                        .unwrap()
                        .as_f64(),
                    Some(segments as f64)
                );
            }
            if command == "collection-stats" {
                assert_eq!(
                    result.get("text_index_segments").unwrap().as_f64(),
                    Some(f64::from(ready))
                );
            }
        }
        assert_eq!(fs::read(path).unwrap(), before);
    }

    let dir = temp_dir("segment-statistics");
    let path = dir.join("data.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    for name in ["empty", "stopwords"] {
        let collection = db
            .create_collection(CollectionConfig {
                name: name.into(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
                tokenizer: Default::default(),
            })
            .unwrap();
        if name == "stopwords" {
            collection
                .insert_document(DocumentInput {
                    external_id: "doc".into(),
                    title: None,
                    source_uri: None,
                    metadata: JsonValue::object(),
                    chunks: vec![ChunkInput {
                        chunk_external_id: "chunk".into(),
                        text: "the and with".into(),
                        ordinal: 0,
                        metadata: JsonValue::object(),
                        source_span: None,
                    }],
                })
                .unwrap();
        }
    }
    verify(&db, &path, true);
    let end = db
        .check(CheckOptions::default())
        .unwrap()
        .critical_end_offset as usize;
    drop(db);
    let original = fs::read(&path).unwrap();
    for corrupt in [false, true] {
        let mut bytes = original.clone();
        if corrupt {
            *bytes.last_mut().unwrap() ^= 1;
        } else {
            bytes.truncate(end);
        }
        fs::write(&path, bytes).unwrap();
        let db = Database::open(&path).unwrap();
        verify(&db, &path, false);
        let result = db
            .collection("stopwords")
            .unwrap()
            .search()
            .text("the")
            .execute()
            .unwrap();
        assert!(result.results.is_empty());
        assert_eq!(result.query_context.index_source, "transient_rebuild");
        verify(&db, &path, false);
        db.rebuild_index(None).unwrap();
        verify(&db, &path, true);
        db.compact(CompactOptions::default()).unwrap();
        verify(&db, &path, false);
        drop(db);
        let db = Database::open_read_only(&path).unwrap();
        verify(&db, &path, false);
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn collection_term_statistics_are_scoped_across_writes_deletes_and_reopens() {
    use cofferdb::{ChunkInput, CompactOptions, DocumentInput};

    fn verify(db: &Database, path: &std::path::Path, expected: [usize; 3]) {
        let before = fs::read(path).unwrap();
        for (name, count) in ["first", "second", "empty"].into_iter().zip(expected) {
            assert_eq!(
                db.collection(name)
                    .unwrap()
                    .stats()
                    .unwrap()
                    .text_index_terms,
                count
            );
            for include_index in [false, true] {
                let mut command = Command::new(env!("CARGO_BIN_EXE_coffer"));
                command
                    .arg("collection-stats")
                    .arg(path)
                    .args(["--collection", name, "--json"]);
                if include_index {
                    command.arg("--include-index");
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
                    result.get("collection").and_then(JsonValue::as_str),
                    Some(name)
                );
                assert_eq!(
                    result.get("text_index_terms").and_then(JsonValue::as_f64),
                    Some(count as f64)
                );
                assert_eq!(result.get("index_stats").is_some(), include_index);
            }
        }
        assert_eq!(fs::read(path).unwrap(), before);
    }

    let input = |id: &str, text: &str| DocumentInput {
        external_id: id.into(),
        title: None,
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: vec![ChunkInput {
            chunk_external_id: id.into(),
            text: text.into(),
            ordinal: 0,
            metadata: JsonValue::object(),
            source_span: None,
        }],
    };
    let dir = temp_dir("collection-term-counts");
    let path = dir.join("data.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    for (name, text) in [
        ("first", "apples bananas apples"),
        ("second", "bananas cherries dates"),
        ("empty", ""),
    ] {
        let collection = db
            .create_collection(CollectionConfig {
                name: name.into(),
                tokenizer: Default::default(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .unwrap();
        if !text.is_empty() {
            collection.insert_document(input("doc", text)).unwrap();
        }
    }
    verify(&db, &path, [2, 3, 0]);
    db.collection("first")
        .unwrap()
        .insert_document(input("extra", "figs grapes figs"))
        .unwrap();
    verify(&db, &path, [4, 3, 0]);
    drop(db);
    let db = Database::open(&path).unwrap();
    verify(&db, &path, [4, 3, 0]);
    let first = db.collection("first").unwrap();
    let extra = first.get_document_by_external_id("extra").unwrap();
    first.delete_document(&extra.document_id).unwrap();
    verify(&db, &path, [2, 3, 0]);
    let original = first.get_document_by_external_id("doc").unwrap();
    first.delete_document(&original.document_id).unwrap();
    verify(&db, &path, [0, 3, 0]);
    db.compact(CompactOptions {
        rebuild_indexes: true,
    })
    .unwrap();
    verify(&db, &path, [0, 3, 0]);
    drop((first, db));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_lock_contention_uses_json_error_and_exit_code() {
    let dir = temp_dir("lock-contention");
    let db_path = dir.join("locked.coffer");
    let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .arg("create-collection")
        .arg(&db_path)
        .args(["other", "--dim", "2"])
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(4));
    let stderr = String::from_utf8(output.stderr).unwrap();
    let json = parse_json(&stderr).unwrap();
    assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(false));
    let error = json.get("error").and_then(JsonValue::as_object).unwrap();
    assert_eq!(
        error.get("code").and_then(JsonValue::as_str),
        Some("lock_contention")
    );
    assert_eq!(
        error.get("suggested_action").and_then(JsonValue::as_str),
        Some("retry after the active writer finishes")
    );
    drop(db);
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn writer_lock_covers_file_aliases_and_follows_compact_publication() {
    use cofferdb::{CompactOptions, DatabaseOpenOptions, ErrorCode};
    use std::os::unix::fs::symlink;

    let dir = temp_dir("writer-aliases");
    let path = dir.join("data.coffer");
    let symbolic = dir.join("symbolic.coffer");
    let hard = dir.join("hard.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    symlink(&path, &symbolic).unwrap();
    fs::hard_link(&path, &hard).unwrap();
    let denied = |alias: &std::path::Path| {
        let error = match Database::open(alias) {
            Ok(_) => panic!("second writer accepted for {}", alias.display()),
            Err(error) => error,
        };
        assert_eq!(error.code(), ErrorCode::LockContention);
        let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
            .arg("create-collection")
            .arg(alias)
            .args(["other", "--dim", "2", "--json"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(4));
        let error = parse_json(std::str::from_utf8(&output.stderr).unwrap()).unwrap();
        assert_eq!(
            error
                .get("error")
                .unwrap()
                .get("code")
                .and_then(JsonValue::as_str),
            Some("lock_contention")
        );
        let reader = Database::open_with_options(
            alias,
            DatabaseOpenOptions {
                read_only: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(reader.stats().unwrap().document_count, 0);
    };
    let before = fs::read(&path).unwrap();
    for alias in [&path, &symbolic, &hard] {
        denied(alias);
    }
    assert_eq!(fs::read(&path).unwrap(), before);
    drop(db);
    let writer = Database::open(&symbolic).unwrap();
    for round in 0..3 {
        writer.compact(CompactOptions::default()).unwrap();
        assert!(fs::symlink_metadata(&symbolic)
            .unwrap()
            .file_type()
            .is_symlink());
        let published_alias = dir.join(format!("published-{round}.coffer"));
        fs::hard_link(&path, &published_alias).unwrap();
        for alias in [&path, &symbolic, &published_alias] {
            denied(alias);
        }
        fs::remove_file(published_alias).unwrap();
    }
    // Rename replaces one directory entry; an old hard link remains an independent snapshot.
    assert_eq!(fs::read(&hard).unwrap(), before);
    drop(writer);
    for alias in [&path, &symbolic, &hard] {
        drop(Database::open(alias).unwrap());
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_check_issues_use_json_output_and_exit_code() {
    let dir = temp_dir("check-issues-exit-code");
    let db_path = dir.join("warning.coffer");
    let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
    db.create_collection(CollectionConfig {
        tokenizer: Default::default(),
        name: "notes".into(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: None,
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
    })
    .unwrap();
    drop(db);
    fs::OpenOptions::new()
        .append(true)
        .open(&db_path)
        .unwrap()
        .write_all(b"corrupt-tail")
        .unwrap();
    let before = fs::read(&db_path).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .arg("check")
        .arg(&db_path)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let json = parse_json(&stdout).unwrap();
    assert_eq!(json.get("ok").and_then(JsonValue::as_bool), Some(true));
    let result = json.get("result").and_then(JsonValue::as_object).unwrap();
    assert_eq!(
        result.get("status").and_then(JsonValue::as_str),
        Some("warning")
    );
    assert_eq!(
        result.get("repair_available").and_then(JsonValue::as_bool),
        Some(true)
    );
    assert_eq!(fs::read(&db_path).unwrap(), before);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn cli_deletions_survive_export_compaction_and_collection_name_reservation() {
    use cofferdb::ErrorCode;

    let dir = temp_dir("cli-delete-lifecycle");
    let path = dir.join("data.coffer");
    let run = |command: &str, args: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
            .arg(command)
            .arg(&path)
            .args(args)
            .arg("--json")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let envelope = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
        assert_eq!(envelope.get("ok").and_then(JsonValue::as_bool), Some(true));
        envelope.get("result").unwrap().clone()
    };
    let verify = |documents, expected: &[&str], export_name: &str| {
        let db = Database::open_read_only(&path).unwrap();
        let collection = db.collection("code").unwrap();
        let stats = collection.stats().unwrap();
        assert_eq!(stats.document_count, documents);
        assert_eq!(stats.chunk_count, expected.len());
        assert_eq!(stats.embedding_count, expected.len());
        let mut hits: Vec<_> = collection
            .search()
            .text("recovery recover")
            .top_k(10)
            .execute()
            .unwrap()
            .results
            .into_iter()
            .map(|r| r.chunk_external_id)
            .collect();
        hits.sort();
        assert_eq!(hits, expected);
        let output = dir.join(export_name);
        let summary = db.export_collection_jsonl("code", &output).unwrap();
        assert_eq!(
            (summary.documents, summary.chunks, summary.embeddings),
            (documents, expected.len(), expected.len())
        );
        for file in ["chunks.jsonl", "embeddings.jsonl"] {
            let mut ids: Vec<_> = fs::read_to_string(output.join(file))
                .unwrap()
                .lines()
                .map(|line| {
                    parse_json(line)
                        .unwrap()
                        .get("chunk_external_id")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .to_owned()
                })
                .collect();
            ids.sort();
            assert_eq!(ids, expected);
        }
    };
    run("init", &[]);
    let manifest =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/data/code/import_manifest.json");
    run("import", &["--manifest", manifest.to_str().unwrap()]);
    let deleted = run(
        "delete-chunk",
        &[
            "--collection",
            "code",
            "--chunk-external-id",
            "storage#cache",
        ],
    );
    assert_eq!(
        deleted
            .get("embedding_deleted")
            .and_then(JsonValue::as_bool),
        Some(true)
    );
    verify(2, &["python#cache", "storage#recovery"], "after-chunk");
    let reader = Database::open_read_only(&path).unwrap();
    let old = reader.collection("code").unwrap();
    assert_eq!(
        old.get_chunk_by_external_id("storage#cache")
            .unwrap_err()
            .code(),
        ErrorCode::ChunkNotFound
    );
    let deleted = run(
        "delete-document",
        &["--collection", "code", "--external-id", "storage.rs"],
    );
    for field in ["chunks_deleted", "embeddings_deleted"] {
        assert_eq!(deleted.get(field).and_then(JsonValue::as_f64), Some(1.0));
    }
    verify(1, &["python#cache"], "after-document");
    run("compact", &[]);
    verify(1, &["python#cache"], "after-compact");
    assert!(old.get_document_by_external_id("storage.rs").is_ok());
    assert!(old.get_chunk_by_external_id("storage#recovery").is_ok());
    drop(old);
    drop(reader);
    let dropped = run("drop-collection", &["--collection", "code"]);
    for field in ["documents_deleted", "chunks_deleted", "embeddings_deleted"] {
        assert_eq!(dropped.get(field).and_then(JsonValue::as_f64), Some(1.0));
    }
    for _ in 0..2 {
        run("compact", &[]);
        let db = Database::open(&path).unwrap();
        let stats = db.stats().unwrap();
        assert_eq!(
            (
                stats.collection_count,
                stats.document_count,
                stats.chunk_count,
                stats.embedding_count
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(
            db.collection("code").err().unwrap().code(),
            ErrorCode::CollectionNotFound
        );
        assert_eq!(
            db.create_collection(CollectionConfig {
                name: "code".into(),
                tokenizer: Default::default(),
                embedding_dim: 2,
                distance_metric: DistanceMetric::Cosine,
                embedding_model: None,
                vectors_normalized: false,
                vector_index: VectorIndexConfig::Exact,
            })
            .err()
            .unwrap()
            .code(),
            ErrorCode::CollectionDeletedNameReserved
        );
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn manifest_import_search_and_export_roundtrip_files() {
    let dir = temp_dir("manifest");
    let db_path = dir.join("notes.coffer");
    fs::write(
        dir.join("documents.jsonl"),
        r#"{"external_id":"doc-1","title":"WAL","source_uri":"docs/wal.md","metadata":{"path":"docs/wal.md","source":"markdown"}}"#,
    )
    .unwrap();
    fs::write(
        dir.join("chunks.jsonl"),
        r#"{"document_external_id":"doc-1","chunk_external_id":"doc-1#1","ordinal":1,"text":"write ahead log recovery explains committed batches","metadata":{"lang":"rust","byte_start":0,"byte_end":52,"line_start":1,"line_end":1}}"#,
    )
    .unwrap();
    fs::write(
        dir.join("embeddings.jsonl"),
        r#"{"chunk_external_id":"doc-1#1","model_name":"test-model","dim":2,"vector":[1.0,0.0]}"#,
    )
    .unwrap();
    fs::write(
        dir.join("import_manifest.json"),
        r#"{"schema_version":"0.1","dataset_id":"demo","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":"embeddings.jsonl","embedding_model":"test-model","embedding_dim":2,"distance_metric":"cosine"}"#,
    )
    .unwrap();

    let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
    let summary = db
        .import_manifest(dir.join("import_manifest.json"), OnConflict::Error)
        .unwrap();
    assert_eq!(summary.documents_inserted, 1);
    assert_eq!(summary.chunks_inserted, 1);
    assert_eq!(summary.embeddings_inserted, 1);

    let results = db
        .collection("notes")
        .unwrap()
        .search()
        .text("wal recovery")
        .query_embedding(QueryEmbeddingInput {
            model_name: Some("test-model".into()),
            dim: 2,
            vector: vec![1.0, 0.0],
        })
        .top_k(3)
        .execute()
        .unwrap();
    assert_eq!(results.results[0].chunk_external_id, "doc-1#1");
    assert!(matches!(results.results[0].explain, JsonValue::Object(_)));

    let export_dir = dir.join("export");
    let export = db.export_collection_jsonl("notes", &export_dir).unwrap();
    assert_eq!(export.documents, 1);
    let export_manifest = export_dir.join("export_manifest.json");
    assert!(export_manifest.exists());
    assert!(export.documents_sha256.len() == 64);
    let exported_manifest_json =
        parse_json(&fs::read_to_string(&export_manifest).unwrap()).unwrap();
    assert_eq!(
        exported_manifest_json
            .get("embedding_dim")
            .and_then(JsonValue::as_f64),
        Some(2.0)
    );
    assert_eq!(
        exported_manifest_json
            .get("distance_metric")
            .and_then(JsonValue::as_str),
        Some("cosine")
    );
    assert_eq!(
        exported_manifest_json
            .get("embedding_model")
            .and_then(JsonValue::as_str),
        Some("test-model")
    );

    let roundtrip_db_path = dir.join("roundtrip.coffer");
    let roundtrip = Database::create(&roundtrip_db_path, DatabaseConfig::default()).unwrap();
    let roundtrip_summary = roundtrip
        .import_manifest(&export_manifest, OnConflict::Error)
        .unwrap();
    assert_eq!(roundtrip_summary.documents_inserted, 1);
    assert_eq!(roundtrip_summary.chunks_inserted, 1);
    assert_eq!(roundtrip_summary.embeddings_inserted, 1);
    let roundtrip_results = roundtrip
        .collection("notes")
        .unwrap()
        .search()
        .text("wal recovery")
        .top_k(1)
        .execute()
        .unwrap();
    assert_eq!(roundtrip_results.results[0].chunk_external_id, "doc-1#1");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn tombstoned_chunks_are_not_searchable() {
    let dir = temp_dir("delete");
    let db_path = dir.join("delete.coffer");
    let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
    let col = db
        .create_collection(CollectionConfig {
            tokenizer: Default::default(),
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
    col.insert_document(cofferdb::DocumentInput {
        external_id: "doc".into(),
        title: None,
        source_uri: None,
        metadata: parse_json("{}").unwrap(),
        chunks: vec![cofferdb::ChunkInput {
            chunk_external_id: "doc#1".into(),
            text: "local rag retrieval".into(),
            ordinal: 1,
            metadata: parse_json("{}").unwrap(),
            source_span: None,
        }],
    })
    .unwrap();
    let chunk = col.get_chunk_by_external_id("doc#1").unwrap();
    col.delete_chunk(&chunk.chunk_id).unwrap();
    let results = col.search().text("retrieval").top_k(5).execute().unwrap();
    assert!(results.results.is_empty());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn manifest_import_failure_leaves_no_visible_collection() {
    let dir = temp_dir("manifest-atomic");
    let db_path = dir.join("atomic.coffer");
    fs::write(
        dir.join("documents.jsonl"),
        r#"{"external_id":"doc-1","metadata":{}}"#,
    )
    .unwrap();
    fs::write(
        dir.join("chunks.jsonl"),
        r#"{"document_external_id":"doc-1","chunk_external_id":"doc-1#1","ordinal":1,"text":"valid chunk","metadata":{}}"#,
    )
    .unwrap();
    fs::write(
        dir.join("embeddings.jsonl"),
        r#"{"chunk_external_id":"missing#1","model_name":"test-model","dim":2,"vector":[1.0,0.0]}"#,
    )
    .unwrap();
    fs::write(
        dir.join("import_manifest.json"),
        r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":"embeddings.jsonl","embedding_model":"test-model","embedding_dim":2,"distance_metric":"cosine"}"#,
    )
    .unwrap();

    let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
    let err = db
        .import_manifest(dir.join("import_manifest.json"), OnConflict::Error)
        .unwrap_err();
    assert_eq!(err.code(), cofferdb::ErrorCode::InvalidJsonlSchema);
    assert!(err.message().contains("use import-embeddings"));
    assert!(db.list_collections().unwrap().is_empty());
    let err = match db.collection("notes") {
        Ok(_) => panic!("failed import should not create the collection"),
        Err(err) => err,
    };
    assert_eq!(err.code(), cofferdb::ErrorCode::CollectionNotFound);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn replace_import_failure_keeps_existing_document_visible() {
    let dir = temp_dir("replace-atomic");
    let db_path = dir.join("replace.coffer");
    fs::write(
        dir.join("documents.jsonl"),
        r#"{"external_id":"doc-1","metadata":{}}"#,
    )
    .unwrap();
    fs::write(
        dir.join("chunks.jsonl"),
        r#"{"document_external_id":"doc-1","chunk_external_id":"doc-1#old","ordinal":1,"text":"old stable chunk","metadata":{}}"#,
    )
    .unwrap();
    fs::write(
        dir.join("import_manifest.json"),
        r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_model":null,"embedding_dim":2,"distance_metric":"cosine"}"#,
    )
    .unwrap();

    let db = Database::create(&db_path, DatabaseConfig::default()).unwrap();
    db.import_manifest(dir.join("import_manifest.json"), OnConflict::Error)
        .unwrap();

    fs::write(
        dir.join("chunks.jsonl"),
        r#"{"document_external_id":"doc-1","chunk_external_id":"doc-1#new","ordinal":1,"text":"new replacement chunk","metadata":{}}"#,
    )
    .unwrap();
    fs::write(
        dir.join("embeddings.jsonl"),
        concat!(
            r#"{"chunk_external_id":"doc-1#new","model_name":"m","dim":2,"vector":[1.0,0.0]}"#,
            "\n",
            r#"{"chunk_external_id":"doc-1#new","model_name":"m","dim":2,"vector":[0.0,1.0]}"#
        ),
    )
    .unwrap();
    fs::write(
        dir.join("import_manifest.json"),
        r#"{"schema_version":"0.1","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":"embeddings.jsonl","embedding_model":null,"embedding_dim":2,"distance_metric":"cosine"}"#,
    )
    .unwrap();

    assert!(db
        .import_manifest(dir.join("import_manifest.json"), OnConflict::Replace)
        .is_err());
    let col = db.collection("notes").unwrap();
    let doc = col.get_document_by_external_id("doc-1").unwrap();
    let chunks = col.list_chunks(&doc.document_id).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].chunk_external_id, "doc-1#old");
    assert_eq!(chunks[0].text, "old stable chunk");
    let _ = fs::remove_dir_all(dir);
}
