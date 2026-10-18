use super::*;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn directory(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "cofferdb-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&path).unwrap();
    path
}

fn fixture(root: &Path, label: &str) -> PathBuf {
    let dir = root.join(label);
    fs::create_dir(&dir).unwrap();
    fs::write(dir.join("documents.jsonl"), format!(
        "{{\"external_id\":\"{label}\",\"title\":\"Record {label}\",\"source_uri\":\"records/{label}.txt\",\"metadata\":{{\"tag\":\"{label}\"}}}}\n"
    )).unwrap();
    fs::write(dir.join("chunks.jsonl"), format!(
        "{{\"document_external_id\":\"{label}\",\"chunk_external_id\":\"chunk-{label}\",\"ordinal\":7,\"text\":\"shared durable record {label}\",\"metadata\":{{\"tag\":\"{label}\"}},\"source_span\":{{\"start\":10,\"end\":40,\"line_start\":2,\"line_end\":3}}}}\n"
    )).unwrap();
    fs::write(dir.join("embeddings.jsonl"), format!(
        "{{\"chunk_external_id\":\"chunk-{label}\",\"model_name\":\"synthetic-2d\",\"dim\":2,\"vector\":[1,0]}}\n"
    )).unwrap();
    let path = dir.join("manifest.json");
    fs::write(&path, r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"embedding_model":"synthetic-2d","distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl","embeddings":"embeddings.jsonl"}"#).unwrap();
    path
}

fn import_one(db: &Database, path: &Path) {
    let before = db.stats().unwrap().last_commit_id;
    let summary = db.import_manifest(path, OnConflict::Error).unwrap();
    assert_eq!(
        (
            summary.documents_inserted,
            summary.chunks_inserted,
            summary.embeddings_inserted
        ),
        (1, 1, 1)
    );
    // This frozen small import must remain one transaction, not three independent writes.
    assert_eq!(db.stats().unwrap().last_commit_id, before + 1);
}

type QueryRows = Vec<Vec<(String, String, String, u64, Option<u64>, Option<u64>)>>;

fn snapshot(db: &Database, labels: &[&str]) -> QueryRows {
    let collection = db.collection("notes").unwrap();
    let stats = collection.stats().unwrap();
    assert_eq!(
        (
            stats.document_count,
            stats.chunk_count,
            stats.embedding_count
        ),
        (labels.len(), labels.len(), labels.len())
    );
    for label in labels {
        let doc = collection.get_document_by_external_id(label).unwrap();
        let chunk = collection
            .get_chunk_by_external_id(&format!("chunk-{label}"))
            .unwrap();
        assert_eq!(doc.external_id, *label);
        assert_eq!(doc.collection_id, collection.collection_id());
        assert_eq!(
            doc.title.as_deref(),
            Some(format!("Record {label}").as_str())
        );
        assert_eq!(chunk.document_id, doc.document_id);
        assert_eq!(
            doc.source_uri.as_deref(),
            Some(format!("records/{label}.txt").as_str())
        );
        assert_eq!(chunk.collection_id, doc.collection_id);
        assert_eq!(chunk.chunk_external_id, format!("chunk-{label}"));
        assert_eq!(chunk.ordinal, 7);
        assert_eq!(
            chunk.source_span,
            Some(crate::model::SourceSpan {
                start: Some(10),
                end: Some(40),
                line_start: Some(2),
                line_end: Some(3)
            })
        );
        assert_eq!(chunk.text, format!("shared durable record {label}"));
        assert_eq!(
            doc.metadata,
            parse(&format!("{{\"tag\":\"{label}\"}}")).unwrap()
        );
        assert_eq!(chunk.metadata, doc.metadata);
    }
    let mut output = Vec::new();
    for mode in 0..3 {
        let mut query = collection.search().top_k(10);
        if mode != 1 {
            query = query.text("shared");
        }
        if mode != 0 {
            query = query.query_embedding(QueryEmbeddingInput {
                model_name: Some("synthetic-2d".into()),
                dim: 2,
                vector: vec![1.0, 0.0],
            });
        }
        let result = query.execute().unwrap();
        let mut ids = result
            .results
            .iter()
            .map(|r| r.document_external_id.as_str())
            .collect::<Vec<_>>();
        ids.sort_unstable();
        assert_eq!(ids, labels);
        output.push(
            result
                .results
                .into_iter()
                .map(|r| {
                    (
                        r.document_external_id,
                        r.chunk_external_id,
                        r.snippet,
                        r.final_score.to_bits(),
                        r.bm25_score.map(f64::to_bits),
                        r.vector_score.map(f64::to_bits),
                    )
                })
                .collect(),
        );
    }
    output
}

fn assert_exported_objects(db: &Database, out: &Path, labels: &[&str]) {
    let summary = db.export_collection_jsonl("notes", out).unwrap();
    assert_eq!(
        (summary.documents, summary.chunks, summary.embeddings),
        (labels.len(), labels.len(), labels.len())
    );
    for (name, key) in [
        ("documents", "external_id"),
        ("chunks", "chunk_external_id"),
    ] {
        let input = fs::read_to_string(out.join(format!("{name}.jsonl"))).unwrap();
        let rows = input
            .lines()
            .map(|line| parse(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), labels.len());
        for label in labels {
            let id = if name == "documents" {
                label.to_string()
            } else {
                format!("chunk-{label}")
            };
            let row = rows
                .iter()
                .find(|row| row.get(key).and_then(JsonValue::as_str) == Some(id.as_str()))
                .unwrap();
            let expected = parse(
                &fs::read_to_string(
                    out.parent()
                        .unwrap()
                        .join(label)
                        .join(format!("{name}.jsonl")),
                )
                .unwrap(),
            )
            .unwrap();
            for (key, value) in expected.as_object().unwrap() {
                assert_eq!(row.get(key), Some(value), "{name}/{id}/{key}");
            }
        }
    }
    let input = fs::read_to_string(out.join("embeddings.jsonl")).unwrap();
    let mut ids = Vec::new();
    for line in input.lines() {
        let row = parse(line).unwrap();
        assert_eq!(row.get("vector"), Some(&parse("[1,0]").unwrap()));
        assert_eq!(row.get("dim"), Some(&parse("2").unwrap()));
        assert_eq!(
            row.get("model_name").and_then(JsonValue::as_str),
            Some("synthetic-2d")
        );
        ids.push(
            row.get("chunk_external_id")
                .and_then(JsonValue::as_str)
                .unwrap()
                .to_string(),
        );
    }
    ids.sort();
    assert_eq!(
        ids,
        labels
            .iter()
            .map(|label| format!("chunk-{label}"))
            .collect::<Vec<_>>()
    );
}

fn core_flow(root: &Path) -> QueryRows {
    let input = fixture(root, "a");
    let path = root.join("data.coffer");
    let db = Database::create(
        &path,
        DatabaseConfig {
            background_compaction: false,
            ..Default::default()
        },
    )
    .unwrap();
    import_one(&db, &input);
    let expected = snapshot(&db, &["a"]);
    assert_eq!(
        db.check(CheckOptions {
            verify_indexes: true
        })
        .unwrap()
        .status,
        "ok"
    );
    db.compact(CompactOptions::default()).unwrap();
    assert_eq!(snapshot(&db, &["a"]), expected);
    assert_exported_objects(&db, &root.join("export"), &["a"]);
    db.close().unwrap();
    let original = fs::read(&path).unwrap();
    let read = Database::open_read_only(&path).unwrap();
    assert_eq!(snapshot(&read, &["a"]), expected);
    assert_eq!(
        read.check(CheckOptions {
            verify_indexes: true
        })
        .unwrap()
        .status,
        "ok"
    );
    read.close().unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
    let copy = Database::create(
        root.join("copy.coffer"),
        DatabaseConfig {
            background_compaction: false,
            ..Default::default()
        },
    )
    .unwrap();
    import_one(&copy, &root.join("export/export_manifest.json"));
    assert_eq!(snapshot(&copy, &["a"]), expected);
    assert_exported_objects(&copy, &root.join("copy-export"), &["a"]);
    copy.close().unwrap();
    expected
}

#[test]
fn software_crc_dispatch_preserves_public_core_flow() {
    use crate::hash::test_backend::{counts, SoftwareOnly};
    let normal = directory("crc-normal");
    let software = directory("crc-software");
    let expected = core_flow(&normal);
    let before = counts();
    {
        let _guard = SoftwareOnly::new();
        assert_eq!(core_flow(&software), expected);
        let after = counts();
        assert!(after.0 > before.0);
        assert_eq!(after.1, before.1);
        eprintln!(
            "software CRC core flow: software_calls={}, hardware_calls=0",
            after.0 - before.0
        );
    }
    fs::remove_dir_all(normal).unwrap();
    fs::remove_dir_all(software).unwrap();
}

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[ignore = "subprocess worker for public_import_crash_windows_preserve_complete_objects"]
fn public_import_crash_worker() {
    let path = std::env::var_os("COFFERDB_TYPED_CRASH_DATABASE").unwrap();
    let manifest = std::env::var_os("COFFERDB_TYPED_CRASH_MANIFEST").unwrap();
    let db = Database::open_with_options(
        path,
        DatabaseOpenOptions {
            durability: Some(Durability::Commit),
            background_compaction: Some(false),
            ..Default::default()
        },
    )
    .unwrap();
    import_one(&db, Path::new(&manifest));
    if std::env::var("COFFERDB_TEST_CRASH_STAGE").unwrap() == "api_returned" {
        fs::write(
            std::env::var_os("COFFERDB_TEST_CRASH_MARKER").unwrap(),
            "api_returned",
        )
        .unwrap();
        loop {
            std::thread::park();
        }
    }
    panic!("the selected crash barrier was not reached");
}

#[test]
fn public_import_crash_windows_preserve_complete_objects() {
    let root = directory("typed-import-crash");
    let a = fixture(&root, "a");
    let b = fixture(&root, "b");
    let baseline = root.join("baseline.coffer");
    let db = Database::create(
        &baseline,
        DatabaseConfig {
            durability: Durability::Commit,
            background_compaction: false,
            ..Default::default()
        },
    )
    .unwrap();
    import_one(&db, &a);
    let a_commit = db.stats().unwrap().last_commit_id;
    let expected_a = snapshot(&db, &["a"]);
    db.close().unwrap();
    let control = root.join("control.coffer");
    fs::copy(&baseline, &control).unwrap();
    let db = Database::open(&control).unwrap();
    import_one(&db, &b);
    let expected_ab = snapshot(&db, &["a", "b"]);
    db.close().unwrap();
    for (stage, committed) in [
        ("batch_before_commit", false),
        ("commit_payload_partial", false),
        ("commit_synced", true),
        ("api_returned", true),
    ] {
        let path = root.join(format!("{stage}.coffer"));
        let marker = root.join(format!("{stage}.marker"));
        fs::copy(&baseline, &path).unwrap();
        let stdout = File::create(root.join(format!("{stage}.stdout"))).unwrap();
        let stderr = File::create(root.join(format!("{stage}.stderr"))).unwrap();
        let mut child = OwnedChild(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "db::tests::recovery_crc::public_import_crash_worker",
                    "--ignored",
                    "--nocapture",
                ])
                .env("COFFERDB_TYPED_CRASH_DATABASE", &path)
                .env("COFFERDB_TYPED_CRASH_MANIFEST", &b)
                .env("COFFERDB_TEST_CRASH_STAGE", stage)
                .env("COFFERDB_TEST_CRASH_MARKER", &marker)
                .stdin(Stdio::null())
                .stdout(stdout)
                .stderr(stderr)
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(15);
        while !marker.exists() {
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "worker exited before {stage}; evidence: {}",
                root.display()
            );
            assert!(
                Instant::now() < deadline,
                "barrier timeout {stage}; evidence: {}",
                root.display()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(fs::read_to_string(&marker).unwrap(), stage);
        assert!(child.0.try_wait().unwrap().is_none());
        child.0.kill().unwrap();
        assert!(!child.0.wait().unwrap().success());
        let bytes = fs::read(&path).unwrap();
        let db = Database::open_read_only(&path).unwrap();
        let labels: &[&str] = if committed { &["a", "b"] } else { &["a"] };
        assert_eq!(
            snapshot(&db, labels),
            if committed {
                expected_ab.clone()
            } else {
                expected_a.clone()
            }
        );
        if !committed {
            let collection = db.collection("notes").unwrap();
            assert!(collection.get_document_by_external_id("b").is_err());
            assert!(collection.get_chunk_by_external_id("chunk-b").is_err());
        }
        let check = db
            .check(CheckOptions {
                verify_indexes: true,
            })
            .unwrap();
        assert_eq!(check.last_valid_commit, a_commit + u64::from(committed));
        assert_eq!(
            (check.live_documents, check.live_chunks),
            (labels.len(), labels.len())
        );
        if committed {
            assert_eq!(check.status, "ok", "{stage}: {check:?}");
            assert!(check.issues.is_empty(), "{stage}: {check:?}");
        } else {
            assert_eq!(check.status, "warning", "{stage}: {check:?}");
            assert!(!check.issues.is_empty(), "{stage}: {check:?}");
            for issue in &check.issues {
                assert_eq!(issue.severity, "warning");
                assert_eq!(issue.stream, "critical");
                assert_eq!(issue.visibility_impact, "tail_only");
                assert!(
                    matches!(issue.code.as_str(), "uncommitted_batch" | "corrupt_tail"),
                    "{stage}: {issue:?}"
                );
            }
        }
        if stage == "api_returned" {
            assert_eq!(check.index_check, "ok", "{check:?}");
            assert_eq!(check.index_check_reason, None);
        } else {
            assert_eq!(check.index_check, "skipped", "{check:?}");
            assert_eq!(
                check.index_check_reason.as_deref(),
                Some("persistent_index_checkpoint_missing")
            );
        }
        assert_exported_objects(&db, &root.join(format!("{stage}-export")), labels);
        db.close().unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        eprintln!("public import crash window {stage}: committed={committed}, last_valid_commit={}, complete_objects={}, readonly_bytes_unchanged=true", a_commit + u64::from(committed), labels.len());
    }
    fs::remove_dir_all(root).unwrap();
}
