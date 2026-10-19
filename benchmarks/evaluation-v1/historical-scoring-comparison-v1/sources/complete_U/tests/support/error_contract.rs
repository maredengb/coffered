use crate::JsonValue;

type Validation = Result<(), &'static str>;

fn string<'a>(value: &'a JsonValue, field: &str) -> Result<&'a str, &'static str> {
    value
        .get(field)
        .and_then(JsonValue::as_str)
        .ok_or("missing or non-string field")
}

fn integer(value: &JsonValue, field: &str) -> Result<u64, &'static str> {
    // u64::MAX rounds up to 2^64 as f64, so use an exclusive bound.
    match value.get(field) {
        Some(JsonValue::Number(number))
            if number.is_finite()
                && *number >= 0.0
                && number.fract() == 0.0
                && *number < 18_446_744_073_709_551_616.0 =>
        {
            Ok(*number as u64)
        }
        _ => Err("missing or non-u64 integer field"),
    }
}

fn decimal(value: &JsonValue, field: &str) -> Result<u128, &'static str> {
    let text = string(value, field)?;
    if text.is_empty()
        || !text.bytes().all(|byte| byte.is_ascii_digit())
        || (text.len() > 1 && text.starts_with('0'))
    {
        return Err("noncanonical unsigned decimal string");
    }
    text.parse().map_err(|_| "decimal exceeds u128")
}

// Independent requirements oracle, not a second production validator or a closed detail schema.
pub(crate) fn validate_error(value: &JsonValue) -> Validation {
    value.as_object().ok_or("error must be an object")?;
    let code = string(value, "code")?;
    if !matches!(
        code,
        "io_error"
            | "file_format_error"
            | "unsupported_version"
            | "unsupported_feature_flag"
            | "collection_not_found"
            | "collection_already_exists"
            | "invalid_embedding_dimension"
            | "embedding_missing"
            | "invalid_metadata_filter"
            | "invalid_jsonl_schema"
            | "missing_required_field"
            | "invalid_vector_length"
            | "embedding_model_mismatch"
            | "unsupported_metric"
            | "duplicate_external_id"
            | "collection_deleted_name_reserved"
            | "lock_contention"
            | "checksum_mismatch"
            | "benchmark_artifact_mode_conflict"
            | "corrupt_tail"
            | "index_corruption"
            | "recovery_error"
            | "resource_exhausted"
            | "permission_denied"
            | "invalid_search_params"
            | "invalid_search_weights"
            | "invalid_metadata_boost"
            | "document_not_found"
            | "chunk_not_found"
            | "cli_usage"
    ) {
        return Err("unknown error code");
    }
    string(value, "message")?;
    let recoverable = value
        .get("recoverable")
        .and_then(JsonValue::as_bool)
        .ok_or("missing or non-boolean recoverable")?;
    let action = match value.get("suggested_action") {
        Some(JsonValue::Null) => None,
        Some(JsonValue::String(text)) => Some(text.as_str()),
        _ => return Err("missing or invalid suggested_action"),
    };
    if code == "embedding_model_mismatch" && !recoverable {
        return Err("model conflicts are recoverable after explicit correction or cleanup");
    }
    let Some(detail) = value.get("detail") else {
        return if code == "resource_exhausted" {
            Err("resource detail is required")
        } else {
            Ok(())
        };
    };
    if code == "invalid_jsonl_schema"
        && detail.get("field").and_then(JsonValue::as_str) == Some("vectors_normalized")
        && !recoverable
    {
        return Err("manifest vector configuration can be retried after explicit correction");
    }
    detail.as_object().ok_or("detail must be an object")?;
    if detail.get("reason").and_then(JsonValue::as_str) == Some("writer_reopen_required")
        && (code != "recovery_error" || !recoverable || action.map_or(true, str::is_empty))
    {
        return Err("invalid writer-reopen classification or recovery advice");
    }
    if code == "resource_exhausted" {
        if !recoverable || action.map_or(true, str::is_empty) {
            return Err("invalid resource recovery advice");
        }
        if !matches!(
            string(detail, "resource")?,
            "memory" | "catalog" | "temporary_disk"
        ) {
            return Err("unknown resource");
        }
        let operation = string(detail, "operation")?;
        if !operation.starts_with(|ch: char| ch.is_ascii_lowercase())
            || !operation
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        {
            return Err("invalid open operation label");
        }
        let limit = decimal(detail, "limit_bytes")?;
        u64::try_from(limit).map_err(|_| "limit exceeds u64")?;
        if decimal(detail, "requested_bytes")? <= limit {
            return Err("requested must exceed the total limit");
        }
    }
    if detail.get("json_limit_reason").is_some()
        || detail.get("json_input_kind").is_some()
        || detail.get("limit_unit").is_some()
        || detail.get("weights_reason").and_then(JsonValue::as_str) == Some("json_processing_limit")
        || detail
            .get("metadata_boost_reason")
            .and_then(JsonValue::as_str)
            == Some("json_processing_limit")
    {
        let kind = string(detail, "json_input_kind")?;
        let expected_code = match kind {
            "manifest" | "collection_config" | "runtime_config" | "document_metadata"
            | "chunk_metadata" | "query_embedding" | "jsonl_record" | "benchmark_artifact" => {
                "invalid_jsonl_schema"
            }
            "filter" => "invalid_metadata_filter",
            "weights" => "invalid_search_weights",
            "metadata_boost" => "invalid_metadata_boost",
            "stored_json" => "recovery_error",
            _ => return Err("unknown JSON input kind"),
        };
        if code != expected_code {
            return Err("invalid JSON error classification");
        }
        for (kind_with_reason, field) in [
            ("weights", "weights_reason"),
            ("metadata_boost", "metadata_boost_reason"),
        ] {
            if kind == kind_with_reason && string(detail, field)? != "json_processing_limit" {
                return Err("missing JSON query-option reason");
            }
        }
        let limit = integer(detail, "limit")?;
        let reason = string(detail, "json_limit_reason")?;
        let unit = string(detail, "limit_unit")?;
        match reason {
            "depth_exceeded" if limit == 128 && unit == "container_levels" => {}
            "unit_bytes_exceeded" if limit == 8_388_608 && unit == "bytes" => {}
            "work_budget_exceeded" if limit <= 33_554_432 && unit == "bytes" => {}
            _ => return Err("invalid JSON reason, limit or unit"),
        }
        if detail.get("actual").is_some() {
            let actual = integer(detail, "actual")?;
            if reason != "work_budget_exceeded" && actual <= limit {
                return Err("observed JSON size must exceed the limit");
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_writer_reopen(value: &JsonValue) -> Validation {
    validate_error(value)?;
    if string(value, "code")? != "recovery_error"
        || value
            .get("detail")
            .and_then(|detail| detail.get("reason"))
            .and_then(JsonValue::as_str)
            != Some("writer_reopen_required")
    {
        return Err("known retained-file refusal must carry its conditional reason");
    }
    Ok(())
}
