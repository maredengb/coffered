use super::*;
use crate::json::JsonProcessingPolicyV1;

fn fixture(label: &str) -> (PathBuf, PathBuf) {
    let path = temp_path(label);
    let directory = path.with_extension("input");
    fs::create_dir(&directory).unwrap();
    let manifest = directory.join("manifest.json");
    fs::write(&manifest, r#"{"schema_version":"0.1","dataset_id":"cli-warning-data","collection":"notes","documents":"documents.jsonl","chunks":"chunks.jsonl","embedding_dim":2,"distance_metric":"cosine"}"#).unwrap();
    fs::write(
        directory.join("documents.jsonl"),
        r#"{"external_id":"doc","metadata":{}}"#,
    )
    .unwrap();
    let chunks = (0..17).map(|n| format!(r#"{{"document_external_id":"doc","chunk_external_id":"c{n}","ordinal":{n},"text":"needle","metadata":{{"byte_start":0,"byte_end":1}},"source_span":{{"start":1,"end":2}}}}
"#)).collect::<String>();
    fs::write(directory.join("chunks.jsonl"), chunks).unwrap();
    (path, manifest)
}

fn import(path: &std::path::Path, manifest: &std::path::Path) -> CommandResult {
    run_command(
        "import",
        Args::new(vec![
            path.display().to_string(),
            "--manifest".into(),
            manifest.display().to_string(),
            "--on-conflict".into(),
            "skip".into(),
        ]),
    )
    .unwrap()
}

#[test]
fn import_retained_output_preserves_complete_warnings_and_legacy_wire() {
    let (path, manifest) = fixture("import-output-wire");
    let result = import(&path, &manifest);
    let (_, work, _) = result.retained.as_ref().unwrap();
    let context = result.retained.as_ref().unwrap().0.clone();
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.reserved_bytes()
    );
    let expected_summary = {
        let db = Database::open(&path).unwrap();
        let summary = db.import_manifest(&manifest, OnConflict::Skip).unwrap();
        assert_eq!(summary.warnings.len(), 19);
        summary
    };
    let warnings = result.value.get("warnings").unwrap().as_array().unwrap();
    assert_eq!(warnings.len(), 19);
    assert_eq!(
        warnings,
        expected_summary
            .to_json()
            .get("warnings")
            .unwrap()
            .as_array()
            .unwrap()
    );
    assert_eq!(
        warnings[2].get("code").and_then(JsonValue::as_str),
        Some("source_span_metadata_conflict")
    );
    let expected = cli_envelope("import", true, result.value.clone(), None, 7);
    let before = fs::read(&path).unwrap();
    let actual = success_output("import", result, 7).unwrap();
    assert_eq!(actual, expected);
    let envelope = parse(&actual).unwrap();
    assert_eq!(
        envelope.get("warnings").unwrap().as_array().unwrap().len(),
        19
    );
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::remove_file(path).unwrap();
    fs::remove_dir_all(manifest.parent().unwrap()).unwrap();
}

#[test]
fn failed_import_output_does_not_replay_committed_import_or_leak_credit() {
    let (path, manifest) = fixture("import-output-failure");
    for native in [false, true] {
        let result = import(&path, &manifest);
        let context = result.retained.as_ref().unwrap().0.clone();
        let before = fs::read(&path).unwrap();
        let mut held = context.workspace_reservation();
        let error = if native {
            crate::test_alloc::fail_byte_allocation_once(
                "source_span_metadata_conflict".len(),
                || success_output("import", result, 7),
            )
            .unwrap_err()
        } else {
            let occupied = context.available_workspace_bytes() - 1;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            success_output("import", result, 7).unwrap_err()
        };
        assert_eq!(
            error.code(),
            if native {
                ErrorCode::IoError
            } else {
                ErrorCode::RecoveryError
            }
        );
        if !native {
            assert!(error
                .detail()
                .contains(&("json_limit_reason".into(), "work_budget_exceeded".into())));
        }
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - held.reserved_bytes()
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        let db = Database::open_read_only(&path).unwrap();
        assert_eq!(db.stats().unwrap().chunk_count, 17);
        drop(db);
        drop(held);
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
    }
    fs::remove_file(path).unwrap();
    fs::remove_dir_all(manifest.parent().unwrap()).unwrap();
}

#[test]
fn failed_summary_conversion_keeps_its_owner_and_the_committed_database() {
    let (path, manifest) = fixture("import-summary-conversion-failure");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let context = db.json_context();
    let imported = db
        .import_manifest_measured_in(
            &manifest,
            OnConflict::Error,
            crate::db::ImportIndexBuildMode::DeferredFullCheckpoint,
            context.clone(),
        )
        .unwrap();
    let before = fs::read(&path).unwrap();
    let available = context.available_workspace_bytes();
    let old = imported.summary.warnings.clone();
    for native in [false, true] {
        let mut held = context.workspace_reservation();
        let error = if native {
            crate::test_alloc::fail_byte_allocation_once(
                "source_span_metadata_conflict".len(),
                || {
                    import_output_in(
                        &db,
                        &imported.summary,
                        manifest.to_str().unwrap(),
                        Durability::Fast,
                        &context,
                    )
                },
            )
            .err()
            .unwrap()
        } else {
            let occupied = available - 1;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            import_output_in(
                &db,
                &imported.summary,
                manifest.to_str().unwrap(),
                Durability::Fast,
                &context,
            )
            .err()
            .unwrap()
        };
        assert_eq!(
            error.code(),
            if native {
                ErrorCode::IoError
            } else {
                ErrorCode::RecoveryError
            }
        );
        assert_eq!(imported.summary.warnings, old);
        if !native {
            assert!(error
                .detail()
                .contains(&("json_limit_reason".into(), "work_budget_exceeded".into())));
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(
            context.available_workspace_bytes(),
            available - held.reserved_bytes()
        );
    }
    drop(imported);
    assert_eq!(db.stats().unwrap().chunk_count, 17);
    drop(db);
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
    fs::remove_file(path).unwrap();
    fs::remove_dir_all(manifest.parent().unwrap()).unwrap();
}
