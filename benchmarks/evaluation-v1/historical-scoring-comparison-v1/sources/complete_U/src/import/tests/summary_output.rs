use super::*;

#[test]
fn measured_summary_keeps_warnings_and_dataset_credit_after_database_close() {
    let directory = temp_dir("summary-warning-owner");
    let path = directory.join("data.coffer");
    let manifest = directory.join("manifest.json");
    let dataset_id = "d".repeat(8193);
    fs::write(&manifest, format!(r#"{{"schema_version":"0.1","dataset_id":"{dataset_id}","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine"}}"#)).unwrap();
    fs::write(
        directory.join("documents.jsonl"),
        r#"{"external_id":"doc","metadata":{}}"#,
    )
    .unwrap();
    let chunks = (0..9).map(|n| format!(r#"{{"document_external_id":"doc","chunk_external_id":"c{n}","ordinal":{n},"text":"needle","metadata":{{"byte_start":0,"byte_end":1}},"source_span":{{"start":1,"end":2}}}}
"#)).collect::<String>();
    fs::write(directory.join("chunks.jsonl"), chunks).unwrap();
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let context = db.json_context();
    let measured = db
        .import_manifest_measured_in(
            &manifest,
            OnConflict::Error,
            ImportIndexBuildMode::DeferredFullCheckpoint,
            context.clone(),
        )
        .unwrap();
    let bytes = measured._work.reserved_bytes();
    assert_eq!(measured.summary.warnings.len(), 11);
    assert_eq!(
        measured.summary.warnings[0].field,
        "hashes.documents_sha256"
    );
    assert_eq!(measured.summary.warnings[1].field, "hashes.chunks_sha256");
    assert!(measured.summary.warnings[2..]
        .iter()
        .all(|warning| warning.code == "source_span_metadata_conflict"));
    assert_eq!(
        measured.summary.dataset_id.as_deref(),
        Some(dataset_id.as_str())
    );
    drop(db);
    let limit = crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
    assert_eq!(context.available_workspace_bytes(), limit - bytes);
    let expected = measured.summary.to_json();
    let mut work = context.reservation();
    let value = measured
        .summary
        .build_json(&JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson))
        .unwrap();
    assert_eq!(value, expected);
    drop(measured);
    assert_eq!(
        context.available_workspace_bytes(),
        limit - work.reserved_bytes()
    );
    drop(value);
    drop(work);
    assert_eq!(context.available_workspace_bytes(), limit);
    let db = Database::open(&path).unwrap();
    let context = db.json_context();
    let public = db.import_manifest(&manifest, OnConflict::Skip).unwrap();
    assert_eq!(public.warnings.len(), 11);
    drop(db);
    assert_eq!(context.available_workspace_bytes(), limit);
    assert_eq!(public.warnings[2].field, "source_span");
    fs::remove_dir_all(directory).unwrap();
}
