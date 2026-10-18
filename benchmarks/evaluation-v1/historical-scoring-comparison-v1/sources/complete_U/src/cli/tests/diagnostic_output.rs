use super::*;
use crate::db::{CheckReport, RepairSummary, SnapshotSummary};
use crate::import::ExportSummary;
use crate::json::JsonProcessingPolicyV1;
use crate::storage::{CheckIssue, IndexSuffixReport};

#[test]
fn diagnostic_and_export_field_tables_match_independent_legacy_oracles() {
    let repair = RepairSummary {
        schema_version: "0.1".into(),
        action: "truncate-tail".into(),
        repaired: true,
        before_file_size: 10,
        after_file_size: 7,
        bytes_removed: 3,
        critical_end_offset: 7,
        issues_repaired: 1,
    };
    let snapshot = SnapshotSummary {
        schema_version: "0.1".into(),
        source_path: "source".into(),
        output_path: "dest".into(),
        bytes_copied: 7,
        last_valid_commit: 3,
        critical_end_offset: 5,
        index_suffix_status: "persistent_checkpoint".into(),
    };
    let export = ExportSummary {
        documents: 3,
        chunks: 5,
        embeddings: 5,
        manifest: "export_manifest.json".into(),
        documents_sha256: "documents".into(),
        chunks_sha256: "chunks".into(),
        embeddings_sha256: "embeddings".into(),
    };
    let mut report = CheckReport {
        schema_version: "0.1".into(),
        path: "source".into(),
        status: "warning".into(),
        format_version: 1,
        last_valid_commit: 9_007_199_254_740_992,
        critical_end_offset: 7,
        index_suffix_status: "rebuild_needed".into(),
        index_suffix: None,
        issues: vec![CheckIssue {
            severity: "warning".into(),
            code: "corrupt_suffix".into(),
            message: "broken \"suffix\"".into(),
            stream: "index".into(),
            visibility_impact: "none".into(),
            offset: Some(7),
            extent_id: None,
            record_type: Some("Index".into()),
            batch_id: Some(9_007_199_254_740_992),
            repair_available: true,
            suggested_command: Some("coffer rebuild-index source".into()),
        }],
        repair_available: true,
        index_check: "completed".into(),
        index_check_reason: None,
        live_documents: 3,
        live_chunks: 5,
        tombstones: 2,
    };
    let context = JsonOperationContext::standalone();
    for (value, host, expected) in [
        (
            retained_output_in(context.clone(), |b| repair.build_json(b)).unwrap(),
            repair.to_json(),
            r#"{"schema_version":"0.1","action":"truncate-tail","repaired":true,"before_file_size":10,"before_file_size_bytes":10,"after_file_size":7,"after_file_size_bytes":7,"bytes_removed":3,"critical_end_offset":7,"issues_repaired":1}"#,
        ),
        (
            retained_output_in(context.clone(), |b| snapshot.build_json(b)).unwrap(),
            snapshot.to_json(),
            r#"{"schema_version":"0.1","source_path":"source","output_path":"dest","bytes_copied":7,"last_valid_commit":3,"critical_end_offset":5,"index_suffix_status":"persistent_checkpoint"}"#,
        ),
        (
            retained_output_in(context.clone(), |b| export.build_json(b)).unwrap(),
            export.to_json(),
            r#"{"documents":3,"documents_exported":3,"chunks":5,"chunks_exported":5,"embeddings":5,"embeddings_exported":5,"manifest":"export_manifest.json","documents_sha256":"documents","chunks_sha256":"chunks","embeddings_sha256":"embeddings"}"#,
        ),
        (
            retained_output_in(context.clone(), |b| report.build_json(b)).unwrap(),
            report.to_json(),
            r#"{"schema_version":"0.1","path":"source","status":"warning","format_version":1,"last_valid_commit":9007199254740992,"critical_end_offset":7,"index_suffix_status":"rebuild_needed","index_suffix":null,"issues":[{"severity":"warning","code":"corrupt_suffix","message":"broken \"suffix\"","stream":"index","visibility_impact":"none","offset":7,"extent_id":null,"record_type":"Index","batch_id":9007199254740992,"repair_available":true,"repairable":true,"suggested_command":"coffer rebuild-index source"}],"repair_available":true,"index_check":"completed","index_check_reason":null,"result":{"live_documents":3,"live_chunks":5,"tombstones":2}}"#,
        ),
    ] {
        assert_eq!(value.value, parse(expected).unwrap());
        assert_eq!(host, parse(expected).unwrap());
        let measured = value
            .value
            .reserve_retained_in(JsonInputKind::StoredJson, &context)
            .unwrap();
        assert_eq!(
            value.retained.as_ref().unwrap().1.reserved_bytes(),
            measured.reserved_bytes()
        );
        drop(measured);
        drop(value);
    }
    let suffix = IndexSuffixReport {
        status: "persistent_checkpoint".into(),
        offset: 7,
        checkpoint_generation: 2,
        covered_critical_end_offset: 7,
        covered_last_batch_id: 3,
        covered_last_commit_sequence_no: 9_007_199_254_740_992,
        coverage_hash: u32::MAX,
        collection_count: 1,
        live_document_count: 3,
        live_chunk_count: 5,
        live_embedding_count: 5,
        text_segment_count: 2,
        vector_segment_count: 1,
        index_record_count: 3,
        index_extent_count: 4,
        payload_len: 3,
        build_mode: "full_visible_snapshot".into(),
    };
    let expected = parse(r#"{"status":"persistent_checkpoint","offset":7,"checkpoint_generation":2,"covered_critical_end_offset":7,"covered_last_batch_id":3,"covered_last_commit_sequence_no":9007199254740992,"coverage_hash":4294967295,"collection_count":1,"live_document_count":3,"live_chunk_count":5,"live_embedding_count":5,"text_segment_count":2,"vector_segment_count":1,"index_record_count":3,"index_extent_count":4,"payload_len":3,"total_size_bytes":302,"build_mode":"full_visible_snapshot"}"#).unwrap();
    assert_eq!(suffix.to_json(), expected);
    report.index_suffix = Some(suffix);
    let old = retained_output_in(context.clone(), |b| report.build_json(b)).unwrap();
    assert_eq!(old.value.get("index_suffix"), Some(&expected));
    let available = context.available_workspace_bytes();
    report.issues[0].message = "x".repeat(65_539);
    let error = crate::test_alloc::fail_byte_allocation_once(65_539, || {
        retained_output_in(context.clone(), |b| report.build_json(b))
    })
    .err()
    .unwrap();
    assert_eq!(error.code(), ErrorCode::IoError);
    assert_eq!(old.value.get("index_suffix"), Some(&expected));
    assert_eq!(context.available_workspace_bytes(), available);
    drop(old);
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
}

fn fixture(label: &str) -> PathBuf {
    let path = temp_path(label);
    run(vec![
        "coffer".into(),
        "init".into(),
        path.display().to_string(),
    ])
    .unwrap();
    run(vec![
        "coffer".into(),
        "create-collection".into(),
        path.display().to_string(),
        "notes".into(),
        "--dim".into(),
        "2".into(),
    ])
    .unwrap();
    path
}

#[test]
fn diagnostic_export_outputs_survive_close_and_refusal_does_not_republish() {
    use std::io::Write;
    let path = fixture("diagnostic-output-source");
    let snapshot_path = temp_path("diagnostic-output-snapshot");
    let export_path = temp_path("diagnostic-output-export");
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"broken-index-suffix")
        .unwrap();
    let commands = [
        ("check", vec![path.display().to_string()]),
        (
            "repair",
            vec![path.display().to_string(), "--truncate-tail".into()],
        ),
        (
            "snapshot",
            vec![
                path.display().to_string(),
                snapshot_path.display().to_string(),
            ],
        ),
        (
            "export",
            vec![
                path.display().to_string(),
                export_path.display().to_string(),
                "--collection".into(),
                "notes".into(),
            ],
        ),
    ];
    for (command, args) in commands {
        let result = run_command(command, Args::new(args)).unwrap();
        let context = result.retained.as_ref().unwrap().0.clone();
        let expected = cli_envelope(command, true, result.value.clone(), None, 7);
        let before = fs::read(&path).unwrap();
        let snapshot_before = snapshot_path
            .exists()
            .then(|| fs::read(&snapshot_path).unwrap());
        let export_before = export_path
            .exists()
            .then(|| fs::read(export_path.join("export_manifest.json")).unwrap());
        assert!(context.available_workspace_bytes() < JsonProcessingPolicyV1::WORK_BUDGET_BYTES);
        // An independently retained old output remains valid after the new output fails.
        let old =
            retained_output_in(context.clone(), |b| b.copy_validated_value(&result.value)).unwrap();
        assert_eq!(success_output(command, result, 7).unwrap(), expected);
        let mut held = context.workspace_reservation();
        let available = context.available_workspace_bytes();
        held.grow_to(available - 1, available - 1, "operation_scratch")
            .unwrap();
        let error = success_output(command, old, 7).unwrap_err();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert!(error
            .detail()
            .contains(&("json_limit_reason".into(), "work_budget_exceeded".into())));
        assert_eq!(fs::read(&path).unwrap(), before);
        if let Some(before) = snapshot_before {
            assert_eq!(fs::read(&snapshot_path).unwrap(), before);
        }
        if let Some(before) = export_before {
            assert_eq!(
                fs::read(export_path.join("export_manifest.json")).unwrap(),
                before
            );
        }
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - held.reserved_bytes()
        );
        drop(held);
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
        drop(Database::open_read_only(&path).unwrap());
    }
    fs::remove_file(path).unwrap();
    fs::remove_file(snapshot_path).unwrap();
    fs::remove_dir_all(export_path).unwrap();
}

#[test]
fn unexpected_flags_are_rejected_before_diagnosis_or_export_side_effects() {
    let missing = temp_path("diagnostic-missing-source");
    let out = temp_path("diagnostic-unused-export");
    for (command, args) in [
        (
            "check",
            vec![missing.display().to_string(), "--unexpected".into()],
        ),
        (
            "export",
            vec![
                missing.display().to_string(),
                out.display().to_string(),
                "--collection".into(),
                "notes".into(),
                "--unexpected".into(),
            ],
        ),
    ] {
        assert_eq!(
            run_command(command, Args::new(args)).err().unwrap().code(),
            ErrorCode::CliUsage
        );
        assert!(!missing.exists());
        assert!(!out.exists());
    }
    let path = fixture("diagnostic-unknown-flag-source");
    let before = fs::read(&path).unwrap();
    let error = run_command(
        "export",
        Args::new(vec![
            path.display().to_string(),
            out.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--unexpected".into(),
        ]),
    )
    .err()
    .unwrap();
    assert_eq!(error.code(), ErrorCode::CliUsage);
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(!out.exists());
    fs::remove_file(path).unwrap();
}
