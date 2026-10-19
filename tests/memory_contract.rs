use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use cofferdb::{
    parse_json, ChunkInput, CollectionConfig, Database, DatabaseConfig, DistanceMetric,
    DocumentInput, JsonValue, OnConflict, TokenizerConfig, VectorIndexConfig,
};

const FIELDS: [&str; 5] = ["type", "source", "scope", "project_id", "session_id"];
const OTHER: [&str; 5] = ["event", "tool", "session", "other-project", "other-session"];

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "cofferdb-memory-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("cannot create fixture: {error}"),
            }
        }
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Copy, Debug)]
enum Input {
    Rust,
    Jsonl,
}

struct Record {
    document: String,
    chunk: String,
    text: String,
    metadata: JsonValue,
}

fn records() -> Vec<Record> {
    let template = parse_json(include_str!(
        "../examples/data/memory/metadata-template.json"
    ))
    .unwrap();
    assert_eq!(template.as_object().unwrap().len(), 10);
    let mut values = vec![template.clone()];
    for (field, other) in FIELDS.into_iter().zip(OTHER) {
        let mut changed = template.clone();
        changed
            .as_object_mut()
            .unwrap()
            .insert(field.into(), JsonValue::String(other.into()));
        values.push(changed);
        let mut missing = template.clone();
        missing.as_object_mut().unwrap().remove(field);
        values.push(missing);
    }
    let mut alternate = template;
    for (field, other) in FIELDS.into_iter().zip(OTHER) {
        alternate
            .as_object_mut()
            .unwrap()
            .insert(field.into(), JsonValue::String(other.into()));
    }
    for (field, value) in [
        ("importance", JsonValue::Number(0.25)),
        (
            "expires_at",
            JsonValue::String("2020-01-01T00:00:00Z".into()),
        ),
        ("supersedes", JsonValue::String("entry-00#0".into())),
    ] {
        alternate
            .as_object_mut()
            .unwrap()
            .insert(field.into(), value);
    }
    values.push(alternate);
    values
        .into_iter()
        .enumerate()
        .map(|(index, metadata)| Record {
            document: format!("entry-{index:02}"),
            chunk: format!("entry-{index:02}#0"),
            text: format!("deployment memory fact {index}"),
            metadata,
        })
        .collect()
}

fn object(pairs: impl IntoIterator<Item = (String, JsonValue)>) -> JsonValue {
    JsonValue::Object(pairs.into_iter().collect())
}

fn document_metadata() -> JsonValue {
    parse_json(r#"{"source":"notebook","revision":1,"optional":null}"#).unwrap()
}

fn cli(args: &[&str]) -> JsonValue {
    let output = Command::new(env!("CARGO_BIN_EXE_coffer"))
        .args(args)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let response = parse_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
    assert_eq!(response.get("ok"), Some(&JsonValue::Bool(true)));
    response.get("result").unwrap().clone()
}

fn write_manifest(directory: &Path, rows: &[Record]) -> PathBuf {
    let documents = rows
        .iter()
        .map(|row| {
            object([
                (
                    "external_id".into(),
                    JsonValue::String(row.document.clone()),
                ),
                ("metadata".into(), document_metadata()),
            ])
            .to_compact_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let chunks = rows
        .iter()
        .map(|row| {
            object([
                (
                    "document_external_id".into(),
                    JsonValue::String(row.document.clone()),
                ),
                (
                    "chunk_external_id".into(),
                    JsonValue::String(row.chunk.clone()),
                ),
                ("ordinal".into(), JsonValue::Number(0.0)),
                ("text".into(), JsonValue::String(row.text.clone())),
                ("metadata".into(), row.metadata.clone()),
            ])
            .to_compact_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(directory.join("documents.jsonl"), documents + "\n").unwrap();
    fs::write(directory.join("chunks.jsonl"), chunks + "\n").unwrap();
    let manifest = directory.join("manifest.json");
    fs::write(&manifest, r#"{"schema_version":"0.1","collection":"memory","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
    manifest
}

fn import(path: &Path, manifest: &Path, input: Input) -> Database {
    match input {
        Input::Rust => {
            let db = Database::create(path, DatabaseConfig::default()).unwrap();
            db.import_manifest(manifest, OnConflict::Error).unwrap();
            db
        }
        Input::Jsonl => {
            cli(&["init", path.to_str().unwrap()]);
            cli(&[
                "import",
                path.to_str().unwrap(),
                "--manifest",
                manifest.to_str().unwrap(),
            ]);
            Database::open_read_only(path).unwrap()
        }
    }
}

fn prepare(directory: &Path, rows: &[Record], input: Input) -> Database {
    let path = directory.join("original.coffer");
    if let Input::Jsonl = input {
        return import(&path, &write_manifest(directory, rows), input);
    }
    let db = Database::create(path, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            name: "memory".into(),
            tokenizer: TokenizerConfig::default_en(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
    for row in rows {
        collection
            .insert_document(DocumentInput {
                external_id: row.document.clone(),
                title: None,
                source_uri: None,
                metadata: document_metadata(),
                chunks: vec![ChunkInput {
                    chunk_external_id: row.chunk.clone(),
                    text: row.text.clone(),
                    ordinal: 0,
                    metadata: row.metadata.clone(),
                    source_span: None,
                }],
            })
            .unwrap();
    }
    db
}

fn assert_records(db: &Database, rows: &[Record]) {
    let collection = db.collection("memory").unwrap();
    let stats = collection.stats().unwrap();
    assert_eq!(stats.document_count, rows.len());
    assert_eq!(stats.chunk_count, rows.len());
    for row in rows {
        let document = collection
            .get_document_by_external_id(&row.document)
            .unwrap();
        assert_eq!(document.external_id, row.document);
        assert_eq!(document.metadata, document_metadata());
        let chunks = collection.list_chunks(&document.document_id).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_external_id, row.chunk);
        let chunk = collection.get_chunk_by_external_id(&row.chunk).unwrap();
        assert_eq!(chunk.document_id, document.document_id);
        assert_eq!(chunk.chunk_external_id, row.chunk);
        assert_eq!(chunk.text, row.text);
        assert_eq!(chunk.ordinal, 0);
        assert_eq!(chunk.metadata, row.metadata);
    }
}

fn assert_cli_records(path: &Path, rows: &[Record]) {
    for row in rows {
        let result = cli(&[
            "get-document",
            path.to_str().unwrap(),
            "--collection",
            "memory",
            "--external-id",
            &row.document,
        ]);
        let document = result.get("document").unwrap();
        assert_eq!(
            document.get("external_id").and_then(JsonValue::as_str),
            Some(row.document.as_str())
        );
        assert_eq!(document.get("metadata"), Some(&document_metadata()));
        let result = cli(&[
            "get-chunk",
            path.to_str().unwrap(),
            "--collection",
            "memory",
            "--chunk-external-id",
            &row.chunk,
        ]);
        let chunk = result.get("chunk").unwrap();
        assert_eq!(
            chunk.get("chunk_external_id").and_then(JsonValue::as_str),
            Some(row.chunk.as_str())
        );
        assert_eq!(
            chunk.get("text").and_then(JsonValue::as_str),
            Some(row.text.as_str())
        );
        assert_eq!(chunk.get("metadata"), Some(&row.metadata));
        assert_eq!(
            result.get("external_id").and_then(JsonValue::as_str),
            Some(row.document.as_str())
        );
    }
}

fn assert_search(db: &Database, path: &Path, rows: &[Record], terms: &[(&str, JsonValue)]) {
    // The oracle reads the input records directly; it does not call the filter engine.
    let expected = rows
        .iter()
        .filter(|row| {
            terms
                .iter()
                .all(|(field, value)| row.metadata.get(field) == Some(value))
        })
        .map(|row| row.chunk.clone())
        .collect::<BTreeSet<_>>();
    let filter = object(
        terms
            .iter()
            .map(|(field, value)| (format!("chunk.{field}"), value.clone())),
    );
    let result = db
        .collection("memory")
        .unwrap()
        .search()
        .text("deployment")
        .top_k(rows.len() + 1)
        .filter(filter.clone())
        .execute()
        .unwrap();
    assert_eq!(result.results.len(), expected.len(), "Rust {filter:?}");
    assert_eq!(
        result
            .results
            .into_iter()
            .map(|hit| hit.chunk_external_id)
            .collect::<BTreeSet<_>>(),
        expected,
        "Rust {filter:?}"
    );
    let result = cli(&[
        "search",
        path.to_str().unwrap(),
        "deployment",
        "--collection",
        "memory",
        "--top-k",
        &(rows.len() + 1).to_string(),
        "--filter",
        &filter.to_compact_string(),
    ]);
    let hits = result.get("results").unwrap().as_array().unwrap();
    assert_eq!(hits.len(), expected.len(), "CLI {filter:?}");
    assert_eq!(
        hits.iter()
            .map(|hit| hit
                .get("chunk_external_id")
                .unwrap()
                .as_str()
                .unwrap()
                .to_owned())
            .collect::<BTreeSet<_>>(),
        expected,
        "CLI {filter:?}"
    );
}

fn assert_filters(db: &Database, path: &Path, rows: &[Record]) {
    assert_search(db, path, rows, &[]);
    for (field, other) in FIELDS.into_iter().zip(OTHER) {
        assert!(rows.iter().any(|row| row.metadata.get(field).is_none()));
        for value in [
            rows[0].metadata.get(field).unwrap().clone(),
            JsonValue::String(other.into()),
            JsonValue::String("no-such-value".into()),
        ] {
            assert_search(db, path, rows, &[(field, value)]);
        }
    }
    for project in [
        rows[0].metadata.get("project_id").unwrap().clone(),
        JsonValue::String("other-project".into()),
        JsonValue::String("no-such-project".into()),
    ] {
        for session in [
            rows[0].metadata.get("session_id").unwrap().clone(),
            JsonValue::String("other-session".into()),
        ] {
            assert_search(
                db,
                path,
                rows,
                &[("project_id", project.clone()), ("session_id", session)],
            );
        }
    }
    // Searches must not update timestamps or remove expired/superseded records.
    assert_records(db, rows);
}

#[test]
fn memory_rust_and_jsonl_preserve_metadata_before_and_after_reopen() {
    let rows = records();
    for input in [Input::Rust, Input::Jsonl] {
        let temp = Temp::new();
        let path = temp.0.join("original.coffer");
        let db = prepare(&temp.0, &rows, input);
        assert_records(&db, &rows);
        assert_search(&db, &path, &rows, &[]);
        db.close().unwrap();
        let reopened = Database::open_read_only(&path).unwrap();
        assert_records(&reopened, &rows);
        assert_cli_records(&path, &rows);
        assert_search(&reopened, &path, &rows, &[]);
        assert_records(&reopened, &rows);
        reopened.close().unwrap();
    }
}

#[test]
fn memory_filters_match_independent_rust_and_cli_oracles() {
    let rows = records();
    for input in [Input::Rust, Input::Jsonl] {
        let temp = Temp::new();
        prepare(&temp.0, &rows, input).close().unwrap();
        let path = temp.0.join("original.coffer");
        let reopened = Database::open_read_only(&path).unwrap();
        assert_filters(&reopened, &path, &rows);
        reopened.close().unwrap();
    }
}

#[test]
fn memory_round_trip_preserves_objects_and_reopened_filters() {
    let rows = records();
    for input in [Input::Rust, Input::Jsonl] {
        let temp = Temp::new();
        let path = temp.0.join("original.coffer");
        let original = prepare(&temp.0, &rows, input);
        assert_records(&original, &rows);
        assert_filters(&original, &path, &rows);
        let export = temp.0.join("export");
        match input {
            Input::Rust => {
                original.export_collection_jsonl("memory", &export).unwrap();
            }
            Input::Jsonl => {
                cli(&[
                    "export",
                    path.to_str().unwrap(),
                    export.to_str().unwrap(),
                    "--collection",
                    "memory",
                    "--format",
                    "jsonl",
                ]);
            }
        }
        original.close().unwrap();
        let copy_path = temp.0.join("copy.coffer");
        let copy = import(&copy_path, &export.join("export_manifest.json"), input);
        assert_records(&copy, &rows);
        copy.close().unwrap();
        let reopened = Database::open_read_only(&copy_path).unwrap();
        assert_records(&reopened, &rows);
        assert_cli_records(&copy_path, &rows);
        assert_filters(&reopened, &copy_path, &rows);
        reopened.close().unwrap();
    }
}
