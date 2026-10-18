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

// Recognize the action concepts in these English advice fixtures, not exact
// sentences or a public vocabulary. Extend the fixtures/matcher for a new safe
// phrasing rather than treating these test-only rules as a product text API.
fn affirmed_advice_words(action: &str) -> (Vec<String>, bool) {
    let normalized = action.to_ascii_lowercase().replace("don't", "do not");
    let mut words = Vec::new();
    let mut whole_write_retry = false;
    let is_action = |word: &str| {
        [
            "clos", "release", "confirm", "check", "verif", "reopen", "open", "preserv", "keep",
            "migrat", "reduc", "shrink", "free", "retry", "rerun", "truncat", "delet", "repair",
            "replay", "forc", "kill",
        ]
        .iter()
        .any(|stem| word.starts_with(stem))
    };
    for clause in normalized.split([';', '.', ',', ':', '\n', '\r']) {
        let tokens: Vec<_> = clause
            .split(|ch: char| !ch.is_ascii_alphabetic())
            .filter(|word| !word.is_empty())
            .collect();
        let mut last_action_negated = false;
        for (index, word) in tokens.iter().copied().enumerate() {
            if matches!(word, "but" | "then" | "however" | "instead") {
                last_action_negated = false;
            }
            let action_word = is_action(word);
            let negative_prefix =
                |token: &str| matches!(token, "not" | "never" | "avoid" | "without");
            let previous = index.checked_sub(1).map(|at| tokens[at]);
            let direct_negation = previous.is_some_and(negative_prefix)
                || (previous.is_some_and(|token| {
                    matches!(token, "to" | "ever" | "directly" | "immediately")
                }) && index
                    .checked_sub(2)
                    .is_some_and(|at| negative_prefix(tokens[at])));
            // Negation applies to the adjacent action and coordinated actions,
            // not to later verbs after "not hesitate/forget" or "without delay".
            let negated = action_word
                && (direct_negation
                    || (previous.is_some_and(|token| matches!(token, "or" | "and"))
                        && last_action_negated));
            if action_word {
                last_action_negated = negated;
            }
            if !negated {
                if word.starts_with("retry") || word.starts_with("rerun") {
                    // Keep the object of this affirmative action separate from
                    // a later prohibition such as "do not retry the whole import".
                    let object: Vec<_> = tokens[index + 1..]
                        .iter()
                        .copied()
                        .take_while(|token| !is_action(token))
                        .collect();
                    whole_write_retry |= object
                        .iter()
                        .any(|token| matches!(*token, "whole" | "entire"))
                        && object.iter().any(|token| {
                            ["import", "operat", "request", "batch"]
                                .iter()
                                .any(|stem| token.starts_with(stem))
                        });
                }
                words.push(word.to_owned());
            }
        }
    }
    (words, whole_write_retry)
}

fn validate_required_advice(code: &str, detail: &JsonValue, action: Option<&str>) -> Validation {
    let writer = detail.get("reason").and_then(JsonValue::as_str) == Some("writer_reopen_required");
    let kind = detail.get("json_input_kind").and_then(JsonValue::as_str);
    if !writer && code != "resource_exhausted" && kind.is_none() {
        return Ok(());
    }
    let action = action.ok_or("this refusal requires explicit recovery advice")?;
    let (words, whole_write_retry) = affirmed_advice_words(action);
    let has = |stems: &[&str]| {
        words
            .iter()
            .any(|word| stems.iter().any(|stem| word.starts_with(stem)))
    };
    if has(&["truncat", "delet", "repair", "replay"])
        || whole_write_retry
        || (has(&["forc", "kill"]) && has(&["reader"]))
    {
        return Err(
            "recovery advice must not recommend destructive recovery or whole-write replay",
        );
    }
    if writer
        && !(has(&["writer"])
            && has(&["clos", "release"])
            && has(&["confirm", "check", "verif"])
            && has(&["current"])
            && has(&["file", "path", "database"])
            && (has(&["reopen"]) || (has(&["open"]) && has(&["new", "fresh"]))))
    {
        return Err("writer advice must close the old writer, confirm the current file and reopen");
    }
    if code == "resource_exhausted"
        && !(has(&["release", "free", "reduc", "shrink"])
            && has(&["confirm", "check", "verif"])
            && has(&["commit"])
            && has(&["progress", "complet", "write", "record", "batch"]))
    {
        return Err("resource advice must release/reduce work and verify committed progress");
    }
    if kind == Some("stored_json") {
        if !(has(&["preserv", "keep"])
            && has(&["original"])
            && has(&["file", "database"])
            && has(&["compatib"])
            && has(&["version"])
            && has(&["migrat"]))
        {
            return Err("stored JSON advice must preserve the original and explicitly migrate with a compatible version");
        }
    } else if kind.is_some()
        && !(has(&["reduc", "shrink"]) && has(&["json", "input", "value", "footprint"]))
    {
        return Err("input JSON advice must reduce its value or processing footprint");
    }
    Ok(())
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
        Some(JsonValue::String(text)) if !text.trim().is_empty() && text.trim() != "null" => {
            Some(text.as_str())
        }
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
        && (code != "recovery_error" || !recoverable || action.is_none())
    {
        return Err("invalid writer-reopen classification or recovery advice");
    }
    if code == "resource_exhausted" {
        if !recoverable || action.is_none() {
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
        if action.is_none() {
            return Err("JSON limit refusals require explicit recovery advice");
        }
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
    validate_required_advice(code, detail, action)
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
