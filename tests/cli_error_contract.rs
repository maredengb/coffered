use cofferdb::{parse_json, JsonObject, JsonValue};

#[path = "support/error_contract.rs"]
mod contract;

fn detail(value: &mut JsonValue) -> &mut JsonObject {
    value
        .as_object_mut()
        .unwrap()
        .get_mut("detail")
        .unwrap()
        .as_object_mut()
        .unwrap()
}

fn json_limit(kind: &str, code: &str) -> JsonValue {
    let advice = if kind == "stored_json" {
        "preserve the original file and use a compatible version for explicit migration"
    } else {
        "reduce the JSON value or processing footprint and retry"
    };
    let mut value = parse_json(&format!(
        r#"{{"code":"{code}","message":"controlled refusal","recoverable":{},"suggested_action":"{advice}","detail":{{"json_limit_reason":"depth_exceeded","json_input_kind":"{kind}","limit":128,"limit_unit":"container_levels","actual":129}}}}"#,
        kind != "stored_json"
    ))
    .unwrap();
    match kind {
        "weights" => {
            detail(&mut value).insert(
                "weights_reason".into(),
                JsonValue::String("json_processing_limit".into()),
            );
        }
        "metadata_boost" => {
            detail(&mut value).insert(
                "metadata_boost_reason".into(),
                JsonValue::String("json_processing_limit".into()),
            );
        }
        _ => {}
    }
    value
}

#[test]
fn json_contract_has_twelve_fixed_kinds_conditional_fields_and_optional_actual() {
    for (kind, code) in [
        ("manifest", "invalid_jsonl_schema"),
        ("collection_config", "invalid_jsonl_schema"),
        ("runtime_config", "invalid_jsonl_schema"),
        ("document_metadata", "invalid_jsonl_schema"),
        ("chunk_metadata", "invalid_jsonl_schema"),
        ("query_embedding", "invalid_jsonl_schema"),
        ("jsonl_record", "invalid_jsonl_schema"),
        ("benchmark_artifact", "invalid_jsonl_schema"),
        ("filter", "invalid_metadata_filter"),
        ("weights", "invalid_search_weights"),
        ("metadata_boost", "invalid_metadata_boost"),
        ("stored_json", "recovery_error"),
    ] {
        for (reason, limit, unit) in [
            ("depth_exceeded", 128, "container_levels"),
            ("unit_bytes_exceeded", 8_388_608, "bytes"),
            ("work_budget_exceeded", 4096, "bytes"),
        ] {
            let mut valid = json_limit(kind, code);
            let fields = detail(&mut valid);
            fields.insert("json_limit_reason".into(), JsonValue::String(reason.into()));
            fields.insert("limit".into(), JsonValue::Number(limit as f64));
            fields.insert("limit_unit".into(), JsonValue::String(unit.into()));
            fields.insert("actual".into(), JsonValue::Number((limit + 1) as f64));
            contract::validate_error(&valid).unwrap();
            for field in [
                "json_limit_reason",
                "json_input_kind",
                "limit",
                "limit_unit",
            ] {
                for wrong_type in [false, true] {
                    let mut invalid = valid.clone();
                    if wrong_type {
                        detail(&mut invalid).insert(field.into(), JsonValue::Bool(false));
                    } else {
                        detail(&mut invalid).remove(field);
                    }
                    assert!(
                        contract::validate_error(&invalid).is_err(),
                        "{kind} {field}"
                    );
                }
            }
            let mut wrong_code = valid.clone();
            wrong_code
                .as_object_mut()
                .unwrap()
                .insert("code".into(), JsonValue::String("io_error".into()));
            assert!(contract::validate_error(&wrong_code).is_err());
            if let Some(field) = match kind {
                "weights" => Some("weights_reason"),
                "metadata_boost" => Some("metadata_boost_reason"),
                _ => None,
            } {
                let mut missing_reason = valid.clone();
                detail(&mut missing_reason).remove(field);
                assert!(contract::validate_error(&missing_reason).is_err());
                let mut missing_limits = valid.clone();
                for field in [
                    "json_limit_reason",
                    "json_input_kind",
                    "limit",
                    "limit_unit",
                ] {
                    detail(&mut missing_limits).remove(field);
                }
                assert!(contract::validate_error(&missing_limits).is_err());
            }
            detail(&mut valid).remove("actual");
            detail(&mut valid).insert(
                "future_detail".into(),
                JsonValue::String("permitted extension".into()),
            );
            contract::validate_error(&valid).unwrap();
        }
    }
}

#[test]
fn json_contract_does_not_turn_current_recoverability_into_a_new_policy() {
    for (kind, code) in [
        ("filter", "invalid_metadata_filter"),
        ("stored_json", "recovery_error"),
    ] {
        for recoverable in [false, true] {
            let mut value = json_limit(kind, code);
            value
                .as_object_mut()
                .unwrap()
                .insert("recoverable".into(), JsonValue::Bool(recoverable));
            contract::validate_error(&value).unwrap();
        }
    }
}

#[test]
fn json_contract_rejects_unknown_enums_inexact_integer_types_and_wrong_limits() {
    let valid = json_limit("filter", "invalid_metadata_filter");
    for (field, replacement) in [
        ("json_input_kind", "\"new_unconfirmed_kind\""),
        ("json_limit_reason", "\"truncated\""),
        ("limit_unit", "\"levels\""),
        ("limit", "127"),
        ("limit", "\"128\""),
        ("actual", "null"),
        ("actual", "-1"),
        ("actual", "129.5"),
        ("actual", "128"),
        ("actual", "18446744073709551616"),
    ] {
        let mut invalid = valid.clone();
        detail(&mut invalid).insert(field.into(), parse_json(replacement).unwrap());
        assert!(
            contract::validate_error(&invalid).is_err(),
            "{field} {replacement}"
        );
    }
    let mut work = valid;
    detail(&mut work).insert(
        "json_limit_reason".into(),
        JsonValue::String("work_budget_exceeded".into()),
    );
    detail(&mut work).insert("limit_unit".into(), JsonValue::String("bytes".into()));
    detail(&mut work).insert("limit".into(), JsonValue::Number(33_554_433.0));
    assert!(contract::validate_error(&work).is_err());
    detail(&mut work).insert("limit".into(), JsonValue::Number(0.0));
    detail(&mut work).remove("actual");
    contract::validate_error(&work).unwrap();
}

fn resource(resource: &str, limit: u64, requested: u128) -> JsonValue {
    parse_json(&format!(
        r#"{{"code":"resource_exhausted","message":"budget refusal","recoverable":true,"suggested_action":"release resources and verify committed progress","detail":{{"resource":"{resource}","operation":"new_stage_2","limit_bytes":"{limit}","requested_bytes":"{requested}"}}}}"#
    ))
    .unwrap()
}

#[test]
fn resource_contract_keeps_exact_decimal_ranges_and_open_operation_labels() {
    for name in ["memory", "catalog", "temporary_disk"] {
        for limit in [0, 1, (1 << 53) + 1, u64::MAX] {
            for requested in [u128::from(limit) + 1, u128::MAX] {
                let valid = resource(name, limit, requested);
                contract::validate_error(&valid).unwrap();
                for field in ["resource", "operation", "limit_bytes", "requested_bytes"] {
                    for wrong_type in [false, true] {
                        let mut invalid = valid.clone();
                        if wrong_type {
                            detail(&mut invalid).insert(field.into(), JsonValue::Number(1.0));
                        } else {
                            detail(&mut invalid).remove(field);
                        }
                        assert!(
                            contract::validate_error(&invalid).is_err(),
                            "{name} {field}"
                        );
                    }
                }
            }
        }
    }
    let valid = resource("memory", 1, 2);
    for (field, text) in [
        ("resource", "cache"),
        ("operation", ""),
        ("operation", "2_stage"),
        ("operation", "Stage"),
        ("operation", "stage/path"),
        ("operation", "stage\u{540d}"),
        ("limit_bytes", "18446744073709551616"),
        ("requested_bytes", "340282366920938463463374607431768211456"),
        ("requested_bytes", "1"),
        ("requested_bytes", "0"),
        ("requested_bytes", "02"),
        ("requested_bytes", "-2"),
        ("requested_bytes", "+2"),
        ("requested_bytes", "2.0"),
        ("requested_bytes", "2e0"),
        ("requested_bytes", "2 bytes"),
    ] {
        let mut invalid = valid.clone();
        detail(&mut invalid).insert(field.into(), JsonValue::String(text.into()));
        assert!(
            contract::validate_error(&invalid).is_err(),
            "{field} {text}"
        );
    }
}

#[test]
fn writer_reopen_is_conditional_and_not_required_on_other_recovery_errors() {
    let valid = parse_json(r#"{"code":"recovery_error","message":"retained file","recoverable":true,"suggested_action":"close outdated writer, confirm current file and reopen","detail":{"reason":"writer_reopen_required"}}"#).unwrap();
    contract::validate_writer_reopen(&valid).unwrap();
    for (field, replacement) in [
        ("code", "\"file_format_error\""),
        ("recoverable", "false"),
        ("suggested_action", "null"),
        ("suggested_action", "\"\""),
    ] {
        let mut invalid = valid.clone();
        invalid
            .as_object_mut()
            .unwrap()
            .insert(field.into(), parse_json(replacement).unwrap());
        assert!(contract::validate_writer_reopen(&invalid).is_err());
    }
    let mut missing = valid.clone();
    detail(&mut missing).remove("reason");
    assert!(contract::validate_writer_reopen(&missing).is_err());
    missing.as_object_mut().unwrap().insert(
        "reason".into(),
        JsonValue::String("writer_reopen_required".into()),
    );
    assert!(contract::validate_writer_reopen(&missing).is_err());
    let ordinary = parse_json(r#"{"code":"recovery_error","message":"other failure","recoverable":false,"suggested_action":null}"#).unwrap();
    contract::validate_error(&ordinary).unwrap();
    let stored = json_limit("stored_json", "recovery_error");
    contract::validate_error(&stored).unwrap();
    assert!(contract::validate_writer_reopen(&ordinary).is_err());
    assert!(contract::validate_writer_reopen(&stored).is_err());
}

#[test]
fn model_conflicts_require_explicit_retry_semantics_without_new_detail_fields() {
    for message in ["input model mismatch", "conflicting visible models"] {
        let valid =
            cofferdb::Error::new(cofferdb::ErrorCode::EmbeddingModelMismatch, message).to_json();
        contract::validate_error(&valid).unwrap();
        assert_eq!(valid.get("recoverable"), Some(&JsonValue::Bool(true)));
        assert_eq!(
            valid.get("suggested_action").and_then(JsonValue::as_str),
            Some("use the collection model, or explicitly export and clean up conflicting visible embeddings; rebuilding indexes or truncating the file does not change their models")
        );
        let mut invalid = valid.clone();
        invalid
            .as_object_mut()
            .unwrap()
            .insert("recoverable".into(), JsonValue::Bool(false));
        assert!(contract::validate_error(&invalid).is_err());
    }
}

#[test]
fn ordinary_suggested_action_is_required_nullable_and_not_an_encoded_null() {
    for code in [
        cofferdb::ErrorCode::CliUsage,
        cofferdb::ErrorCode::IoError,
        cofferdb::ErrorCode::RecoveryError,
    ] {
        let actual =
            cofferdb::Error::new(code, "invalid input or local operation failed").to_json();
        assert_eq!(actual.get("suggested_action"), Some(&JsonValue::Null));
        contract::validate_error(&actual).unwrap();

        let mut with_advice = actual.clone();
        with_advice.as_object_mut().unwrap().insert(
            "suggested_action".into(),
            JsonValue::String("correct the input before retrying the operation".into()),
        );
        contract::validate_error(&with_advice).unwrap();

        let mut missing = actual.clone();
        missing.as_object_mut().unwrap().remove("suggested_action");
        assert!(contract::validate_error(&missing).is_err());

        for invalid in [
            JsonValue::String(String::new()),
            JsonValue::String(" \t\r\n".into()),
            JsonValue::String("null".into()),
            JsonValue::String(" null ".into()),
            JsonValue::Number(1.0),
            JsonValue::Bool(false),
            JsonValue::Array(Vec::new()),
            JsonValue::object(),
        ] {
            let mut value = actual.clone();
            value
                .as_object_mut()
                .unwrap()
                .insert("suggested_action".into(), invalid.clone());
            assert!(
                contract::validate_error(&value).is_err(),
                "ordinary {code:?} must reject {invalid:?}"
            );
        }
    }
}

#[test]
fn nullable_ordinary_advice_does_not_waive_required_recovery_advice() {
    let writer = parse_json(r#"{"code":"recovery_error","message":"retained file","recoverable":true,"suggested_action":"close outdated writer, confirm current file and reopen","detail":{"reason":"writer_reopen_required"}}"#).unwrap();
    for actual in [
        writer,
        resource("memory", 1, 2),
        json_limit("manifest", "invalid_jsonl_schema"),
        json_limit("stored_json", "recovery_error"),
    ] {
        contract::validate_error(&actual).unwrap();
        for invalid in [JsonValue::Null, JsonValue::String(" \t".into())] {
            let mut value = actual.clone();
            value
                .as_object_mut()
                .unwrap()
                .insert("suggested_action".into(), invalid);
            assert!(contract::validate_error(&value).is_err(), "{value:?}");
        }
    }
}

#[test]
fn mandatory_recovery_advice_checks_actions_and_rejects_destructive_suffixes() {
    let writer = parse_json(r#"{"code":"recovery_error","message":"retained file","recoverable":true,"suggested_action":"close outdated writer, confirm current file and reopen","detail":{"reason":"writer_reopen_required"}}"#).unwrap();
    for (actual, alternatives, missing_actions) in [
        (
            writer,
            [
                "release the outdated writer; verify the current database path; reopen a writer",
                "close the stale writer, check the current main file, and open a fresh writer",
            ],
            [
                "confirm the current file and reopen the writer",
                "close the outdated writer and reopen it",
                "close the writer and confirm the current file",
            ],
        ),
        (
            resource("memory", 1, 2),
            [
                "free memory or reduce work; check committed progress before retrying writes",
                "release resources, verify committed progress, then retry only outstanding writes",
            ],
            [
                "verify committed progress before retrying writes",
                "release resources then retry",
                "release resources and retry the whole import unchanged",
            ],
        ),
        (
            json_limit("stored_json", "recovery_error"),
            [
                "keep the original file; use a compatible version for migration",
                "preserve the original database and migrate explicitly using a compatible version",
            ],
            [
                "migrate using a compatible version",
                "preserve the original file and retry unchanged",
                "preserve the original file and migrate using the current version",
            ],
        ),
        (
            json_limit("manifest", "invalid_jsonl_schema"),
            [
                "reduce the JSON input or processing footprint before retrying",
                "shrink the input and retry after correcting the JSON value",
            ],
            [
                "retry unchanged",
                "switch to another version",
                "delete the database and retry",
            ],
        ),
    ] {
        contract::validate_error(&actual).unwrap();
        let current = actual
            .get("suggested_action")
            .and_then(JsonValue::as_str)
            .unwrap();
        for advice in alternatives.into_iter().map(str::to_owned).chain([
            format!("{current}; do not truncate or delete the database"),
            format!("{current}; do not retry the whole import unchanged"),
            format!("{current}; do not replay the whole import unchanged"),
            format!("{current} and do not retry the whole import unchanged"),
            format!("{current} and do not replay the whole import unchanged"),
        ]) {
            let mut value = actual.clone();
            value
                .as_object_mut()
                .unwrap()
                .insert("suggested_action".into(), JsonValue::String(advice));
            contract::validate_error(&value).unwrap();
        }
        for advice in missing_actions.into_iter().map(str::to_owned).chain([
            "truncate the current database file and retry the whole import unchanged".into(),
            format!("{current}; truncate the current database file"),
            format!("{current}; replay the whole import unchanged"),
            format!("{current}; do not truncate the database, but delete the main file"),
            format!("{current}; do not hesitate to delete the main database file"),
            format!("{current}; do not forget to truncate the current database file"),
            format!("{current}; without delay delete the current main database file"),
        ]) {
            let mut value = actual.clone();
            value
                .as_object_mut()
                .unwrap()
                .insert("suggested_action".into(), JsonValue::String(advice.clone()));
            assert!(
                contract::validate_error(&value).is_err(),
                "unsafe or incomplete advice: {advice}"
            );
        }
    }
}
