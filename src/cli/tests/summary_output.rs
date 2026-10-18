use super::*;
use crate::db::{CompactSummary, RebuildIndexSummary};
use crate::json::JsonProcessingPolicyV1;

fn compact() -> CompactSummary {
    CompactSummary {
        before_file_size: 9123,
        after_file_size: 8123,
        records_removed: 17,
        tombstones_removed: 3,
        segments_merged: 2,
        index_rebuild: "completed".into(),
        index_rebuild_reason: None,
        duration_ms: 9,
    }
}

fn rebuild(collection: Option<String>) -> RebuildIndexSummary {
    RebuildIndexSummary {
        schema_version: "0.1".into(),
        collection,
        rebuild_scope: "full_visible_snapshot".into(),
        index_suffix_status: "persistent_checkpoint".into(),
        index_record_count: 7,
        index_extent_count: 6,
        text_segment_count: 2,
        vector_segment_count: 1,
        rebuild_needed: false,
        last_index_commit_id: Some(9_007_199_254_740_992),
        duration_ms: 9,
    }
}

#[test]
fn fixed_summary_field_tables_preserve_aliases_null_and_number_semantics() {
    let context = JsonOperationContext::standalone();
    let c = compact();
    let expected = parse(r#"{"before_file_size":9123,"before_file_size_bytes":9123,"after_file_size":8123,"after_file_size_bytes":8123,"records_removed":17,"tombstones_removed":3,"segments_merged":2,"index_rebuild":"completed","index_rebuild_reason":null,"duration_ms":9}"#).unwrap();
    let (value, work) = build_output_in(&context, |builder| c.build_json(builder)).unwrap();
    assert_eq!(value, expected);
    assert_eq!(c.to_json(), expected);
    drop(value);
    drop(work);
    for collection in [None, Some("quoted \"name\"".into())] {
        let r = rebuild(collection.clone());
        let (value, work) = build_output_in(&context, |builder| r.build_json(builder)).unwrap();
        let mut expected = parse(r#"{"schema_version":"0.1","collection":null,"rebuild_scope":"full_visible_snapshot","index_suffix_status":"persistent_checkpoint","index_record_count":7,"index_extent_count":6,"text_segment_count":2,"vector_segment_count":1,"rebuild_needed":false,"last_index_commit_id":0,"duration_ms":9}"#).unwrap();
        expected.as_object_mut().unwrap().insert(
            "collection".into(),
            collection
                .clone()
                .map(JsonValue::String)
                .unwrap_or_default(),
        );
        expected.as_object_mut().unwrap().insert(
            "last_index_commit_id".into(),
            JsonValue::Number(9_007_199_254_740_992.0),
        );
        assert_eq!(value, expected);
        assert_eq!(r.to_json(), expected);
        assert_eq!(
            value
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec![
                "collection",
                "duration_ms",
                "index_extent_count",
                "index_record_count",
                "index_suffix_status",
                "last_index_commit_id",
                "rebuild_needed",
                "rebuild_scope",
                "schema_version",
                "text_segment_count",
                "vector_segment_count",
            ]
        );
        assert_eq!(
            value.get("collection"),
            Some(&collection.map(JsonValue::String).unwrap_or_default())
        );
        assert_eq!(
            value.get("last_index_commit_id"),
            Some(&JsonValue::Number(9_007_199_254_740_992.0))
        );
        assert_eq!(
            value.get("index_record_count"),
            Some(&JsonValue::Number(7.0))
        );
        drop(value);
        drop(work);
    }
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
}

#[test]
fn maintenance_output_owner_survives_close_and_output_refusal_preserves_publication() {
    let path = temp_path("maintenance-summary-command");
    drop(Database::create(&path, DatabaseConfig::default()).unwrap());
    for command in ["compact", "rebuild-index"] {
        let result = run_command(command, Args::new(vec![path.display().to_string()])).unwrap();
        let (context, work, _) = result.retained.as_ref().unwrap();
        let context = context.clone();
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.reserved_bytes()
        );
        let expected = cli_envelope(command, true, result.value.clone(), None, 7);
        let before = fs::read(&path).unwrap();
        assert_eq!(success_output(command, result, 7).unwrap(), expected);
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
        assert_eq!(fs::read(&path).unwrap(), before);

        let result = run_command(command, Args::new(vec![path.display().to_string()])).unwrap();
        let context = result.retained.as_ref().unwrap().0.clone();
        let before = fs::read(&path).unwrap();
        let mut held = context.workspace_reservation();
        let occupied = context.available_workspace_bytes() - 1;
        held.grow_to(occupied, occupied, "operation_scratch")
            .unwrap();
        let error = success_output(command, result, 7).unwrap_err();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert!(error
            .detail()
            .contains(&("json_limit_reason".into(), "work_budget_exceeded".into())));
        assert_eq!(fs::read(&path).unwrap(), before);
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
}

#[test]
fn partial_summary_conversion_failure_preserves_an_older_output_owner() {
    let path = temp_path("maintenance-summary-conversion");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let old = retained_summary_output(&db, |builder| compact().build_json(builder)).unwrap();
    let context = old.retained.as_ref().unwrap().0.clone();
    let available = context.available_workspace_bytes();
    let expected = old.value.clone();
    let summary = rebuild(Some("\"".repeat(65_539)));
    let before = fs::read(&path).unwrap();
    for native in [false, true] {
        let mut held = context.workspace_reservation();
        let error = if native {
            crate::test_alloc::fail_byte_allocation_once(65_539, || {
                retained_summary_output(&db, |builder| summary.build_json(builder))
            })
            .err()
            .unwrap()
        } else {
            held.grow_to(available - 1, available - 1, "operation_scratch")
                .unwrap();
            retained_summary_output(&db, |builder| summary.build_json(builder))
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
        assert_eq!(old.value, expected);
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(
            context.available_workspace_bytes(),
            available - held.reserved_bytes()
        );
    }
    drop(old);
    drop(db);
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
    fs::remove_file(path).unwrap();
}
