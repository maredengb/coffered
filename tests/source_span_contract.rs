use std::fs;

use cofferdb::{
    parse_json, ChunkInput, CollectionConfig, CompactOptions, Database, DatabaseConfig,
    DistanceMetric, DocumentInput, JsonValue, OnConflict, SourceSpan, VectorIndexConfig,
};

fn metadata() -> JsonValue {
    parse_json(r#"{"byte_start":99,"line_start":0,"private":"preserve me"}"#).unwrap()
}

fn offset_json(value: Option<u64>) -> JsonValue {
    match value {
        Some(value) if value > 9_007_199_254_740_991 => JsonValue::String(value.to_string()),
        Some(value) => JsonValue::Number(value as f64),
        None => JsonValue::Null,
    }
}

fn assert_spans(db: &Database, expected: &[Option<SourceSpan>], expected_metadata: &JsonValue) {
    let collection = db.collection("notes").unwrap();
    let results = collection
        .search()
        .text("boundary")
        .top_k(20)
        .execute()
        .unwrap();
    assert_eq!(results.results.len(), expected.len());
    for (index, span) in expected.iter().enumerate() {
        let external_id = format!("doc#{index}");
        let chunk = collection.get_chunk_by_external_id(&external_id).unwrap();
        assert_eq!(&chunk.source_span, span, "{external_id}");
        assert_eq!(&chunk.metadata, expected_metadata);
        let result = results
            .results
            .iter()
            .find(|row| row.chunk_external_id == external_id)
            .unwrap();
        let boundary = result
            .explain
            .get("chunk_boundary")
            .unwrap()
            .as_object()
            .unwrap();
        assert_eq!(
            boundary.get("start"),
            Some(&offset_json(span.as_ref().and_then(|span| span.start)))
        );
        assert_eq!(
            boundary.get("end"),
            Some(&offset_json(span.as_ref().and_then(|span| span.end)))
        );
        assert_eq!(
            boundary.contains_key("line_start"),
            span.as_ref().is_some_and(|span| span.line_start.is_some())
        );
        assert_eq!(
            boundary.contains_key("line_end"),
            boundary.contains_key("line_start")
        );
    }
}

#[test]
fn rust_explicit_unknown_spans_survive_reopen_compact_export_and_reimport() {
    let dir = std::env::temp_dir().join(format!("cofferdb-span-rust-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("data.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let collection = db
        .create_collection(CollectionConfig {
            name: "notes".into(),
            embedding_dim: 2,
            distance_metric: DistanceMetric::Cosine,
            embedding_model: None,
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
            tokenizer: Default::default(),
        })
        .unwrap();
    let inputs = [
        SourceSpan::default(),
        SourceSpan {
            start: Some(0),
            end: Some(0),
            line_start: None,
            line_end: None,
        },
        SourceSpan {
            start: None,
            end: None,
            line_start: Some(2),
            line_end: Some(7),
        },
        SourceSpan {
            start: Some(4),
            end: Some(12),
            line_start: Some(2),
            line_end: Some(7),
        },
        SourceSpan {
            start: Some(9_007_199_254_740_991),
            end: Some(9_007_199_254_740_993),
            line_start: Some(u32::MAX),
            line_end: Some(u32::MAX),
        },
        SourceSpan {
            start: Some(u64::MAX - 1),
            end: Some(u64::MAX),
            line_start: None,
            line_end: None,
        },
    ];
    let expected = inputs
        .iter()
        .map(|span| (span != &SourceSpan::default()).then_some(span.clone()))
        .collect::<Vec<_>>();
    collection
        .insert_document(DocumentInput {
            external_id: "doc".into(),
            title: None,
            source_uri: None,
            metadata: JsonValue::object(),
            chunks: inputs
                .into_iter()
                .enumerate()
                .map(|(index, span)| ChunkInput {
                    chunk_external_id: format!("doc#{index}"),
                    text: "source boundary".into(),
                    ordinal: index,
                    metadata: metadata(),
                    source_span: Some(span),
                })
                .collect(),
        })
        .unwrap();
    assert_spans(&db, &expected, &metadata());
    drop(collection);
    drop(db);
    let db = Database::open(&path).unwrap();
    assert_spans(&db, &expected, &metadata());
    db.compact(CompactOptions {
        rebuild_indexes: true,
    })
    .unwrap();
    assert_spans(&db, &expected, &metadata());
    let output = dir.join("export");
    db.export_collection_jsonl("notes", &output).unwrap();
    let restored =
        Database::create(dir.join("restored.coffer"), DatabaseConfig::default()).unwrap();
    restored
        .import_manifest(output.join("export_manifest.json"), OnConflict::Error)
        .unwrap();
    assert_spans(&restored, &expected, &metadata());
    drop(restored);
    drop(db);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn decimal_offsets_and_metadata_aliases_preserve_full_u64_range() {
    let dir = std::env::temp_dir().join(format!("cofferdb-span-u64-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("documents.jsonl"),
        r#"{"external_id":"doc","metadata":{}}"#,
    )
    .unwrap();
    let metadata = parse_json(r#"{"byte_start":"18446744073709551614","byte_end":"18446744073709551615","line_start":4294967295,"line_end":4294967295}"#).unwrap();
    let row = format!(
        r#"{{"document_external_id":"doc","chunk_external_id":"doc#0","ordinal":0,"text":"source boundary","metadata":{}}}"#,
        metadata.to_compact_string()
    );
    fs::write(dir.join("chunks.jsonl"), row).unwrap();
    fs::write(dir.join("manifest.json"), r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
    let path = dir.join("data.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    db.import_manifest(dir.join("manifest.json"), OnConflict::Error)
        .unwrap();
    let expected = [Some(SourceSpan {
        start: Some(u64::MAX - 1),
        end: Some(u64::MAX),
        line_start: Some(u32::MAX),
        line_end: Some(u32::MAX),
    })];
    assert_spans(&db, &expected, &metadata);
    drop(db);
    let db = Database::open(&path).unwrap();
    db.compact(CompactOptions {
        rebuild_indexes: true,
    })
    .unwrap();
    let output = dir.join("export");
    db.export_collection_jsonl("notes", &output).unwrap();
    let exported = fs::read_to_string(output.join("chunks.jsonl")).unwrap();
    let span = parse_json(exported.trim())
        .unwrap()
        .get("source_span")
        .unwrap()
        .clone();
    assert_eq!(
        span.get("start"),
        Some(&JsonValue::String((u64::MAX - 1).to_string()))
    );
    assert_eq!(
        span.get("end"),
        Some(&JsonValue::String(u64::MAX.to_string()))
    );
    let restored =
        Database::create(dir.join("restored.coffer"), DatabaseConfig::default()).unwrap();
    restored
        .import_manifest(output.join("export_manifest.json"), OnConflict::Error)
        .unwrap();
    assert_spans(&restored, &expected, &metadata);
    drop(restored);
    drop(db);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn jsonl_null_and_empty_spans_override_invalid_metadata_aliases() {
    let dir = std::env::temp_dir().join(format!("cofferdb-span-jsonl-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("documents.jsonl"),
        r#"{"external_id":"doc","metadata":{}}"#,
    )
    .unwrap();
    let rows = [
        "null",
        "{}",
        r#"{"start":null,"end":null,"line_start":null,"line_end":null}"#,
    ];
    fs::write(dir.join("chunks.jsonl"), rows.iter().enumerate().map(|(index, span)| format!(r#"{{"document_external_id":"doc","chunk_external_id":"doc#{index}","ordinal":{index},"text":"source boundary","metadata":{},"source_span":{span}}}"#, metadata().to_compact_string())).collect::<Vec<_>>().join("\n")).unwrap();
    fs::write(dir.join("manifest.json"), r#"{"schema_version":"0.1","collection":"notes","embedding_dim":2,"distance_metric":"cosine","documents":"documents.jsonl","chunks":"chunks.jsonl"}"#).unwrap();
    let db = Database::create(dir.join("data.coffer"), DatabaseConfig::default()).unwrap();
    let imported = db
        .import_manifest(dir.join("manifest.json"), OnConflict::Error)
        .unwrap();
    assert_eq!(
        imported
            .warnings
            .iter()
            .filter(|warning| warning.code == "source_span_metadata_conflict")
            .count(),
        3
    );
    assert_spans(&db, &[None, None, None], &metadata());
    let output = dir.join("export");
    db.export_collection_jsonl("notes", &output).unwrap();
    let restored =
        Database::create(dir.join("restored.coffer"), DatabaseConfig::default()).unwrap();
    restored
        .import_manifest(output.join("export_manifest.json"), OnConflict::Error)
        .unwrap();
    assert_spans(&restored, &[None, None, None], &metadata());
    drop(restored);
    drop(db);
    fs::remove_dir_all(dir).unwrap();
}
