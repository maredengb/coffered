use std::collections::{BTreeMap, BTreeSet};

use super::BenchmarkQueryMode;
use crate::error::{Error, ErrorCode, Result};
use crate::json::JsonValue;
use crate::storage::FORMAT_VERSION;

pub(crate) fn validate_benchmark_result_contract(result: &JsonValue) -> Result<()> {
    let result_obj = result.as_object().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result must be a JSON object",
        )
        .with_detail("field", "benchmark_result")
    })?;
    reject_unknown_benchmark_result_top_level_fields(result_obj)?;
    for field in [
        "schema_version",
        "cofferdb_version",
        "git_commit",
        "file_format_version",
        "hardware",
        "result_file",
        "dataset_id",
        "dataset",
        "profile",
        "profile_params",
        "engine_params",
        "candidate_counts",
        "durability",
        "index_build_mode",
        "import_context",
        "metrics",
        "runs",
        "baseline_configs",
        "baseline_results",
        "baseline_status",
        "baseline_status_reason",
    ] {
        if result.get(field).is_none() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("benchmark result missing required field {field}"),
            )
            .with_detail("field", field));
        }
    }
    let index_build_mode = result
        .get("index_build_mode")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(
        index_build_mode,
        "inline_full_checkpoint" | "deferred_full_checkpoint" | "transient_only"
    ) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result index_build_mode must be a stable enum value",
        )
        .with_detail("field", "index_build_mode"));
    }
    validate_benchmark_version_fields(result)?;
    validate_benchmark_top_level_enums(result)?;
    validate_benchmark_engine_params(result)?;
    validate_benchmark_hardware(result)?;
    validate_benchmark_numeric_surfaces(result)?;
    validate_benchmark_warnings(result)?;
    let baseline_status = result
        .get("baseline_status")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(baseline_status, "completed" | "not_run") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result baseline_status must be completed or not_run",
        )
        .with_detail("field", "baseline_status"));
    }
    match (
        baseline_status,
        result
            .get("baseline_status_reason")
            .and_then(JsonValue::as_str),
        result.get("baseline_status_reason"),
    ) {
        ("completed", _, Some(JsonValue::Null)) => {}
        ("completed", _, _) => {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "completed benchmark baselines must use null baseline_status_reason",
            )
            .with_detail("field", "baseline_status_reason"));
        }
        ("not_run", Some(reason), _) if !reason.trim().is_empty() => {}
        ("not_run", _, _) => {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "not_run benchmark baselines must include baseline_status_reason",
            )
            .with_detail("field", "baseline_status_reason"));
        }
        _ => {}
    }
    let dataset = result
        .get("dataset")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result dataset must be an object",
            )
            .with_detail("field", "dataset")
        })?;
    for field in [
        "dataset_id",
        "version",
        "chunking",
        "embedding_model",
        "embedding_dim",
        "dataset_manifest_sha256",
        "documents_sha256",
        "chunks_sha256",
        "embeddings_sha256",
        "queries_sha256",
        "qrels_sha256",
        "query_embedding_source",
        "query_embedding_model",
        "query_embedding_dim",
        "query_embeddings_sha256",
    ] {
        if dataset.get(field).is_none() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                format!("benchmark result dataset missing required field {field}"),
            )
            .with_detail("field", format!("dataset.{field}")));
        }
    }
    validate_benchmark_dataset_identity(result, dataset)?;
    if !matches!(result.get("runs"), Some(JsonValue::Array(_))) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result runs must be an array",
        )
        .with_detail("field", "runs"));
    }
    if !matches!(result.get("baseline_configs"), Some(JsonValue::Array(_))) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result baseline_configs must be an array",
        )
        .with_detail("field", "baseline_configs"));
    }
    if !matches!(result.get("baseline_results"), Some(JsonValue::Array(_))) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result baseline_results must be an array",
        )
        .with_detail("field", "baseline_results"));
    }
    if baseline_status == "not_run"
        && result
            .get("baseline_results")
            .and_then(JsonValue::as_array)
            .map(|results| !results.is_empty())
            .unwrap_or(false)
    {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result baseline_results must be empty when baseline_status is not_run",
        )
        .with_detail("field", "baseline_results")
        .with_detail("baseline_status", "not_run"));
    }
    let query_embedding_source = result
        .get("dataset")
        .and_then(|dataset| dataset.get("query_embedding_source"))
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if query_embedding_source == "none" {
        let has_vector_run = result
            .get("runs")
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
            .iter()
            .any(|run| {
                matches!(
                    run.get("query_mode").and_then(JsonValue::as_str),
                    Some("vector_exact") | Some("hybrid")
                )
            });
        if has_vector_run {
            return Err(Error::new(
                ErrorCode::BenchmarkArtifactModeConflict,
                "benchmark result with query_embedding_source none must not contain vector or hybrid runs",
            )
            .with_detail("query_embedding_source", "none")
            .with_detail("field", "runs[].query_mode"));
        }
        let has_vector_baseline_result = result
            .get("baseline_results")
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
            .iter()
            .any(|run| {
                matches!(
                    run.get("query_mode").and_then(JsonValue::as_str),
                    Some("vector_exact") | Some("hybrid")
                )
            });
        if has_vector_baseline_result {
            return Err(Error::new(
                ErrorCode::BenchmarkArtifactModeConflict,
                "benchmark result with query_embedding_source none must not contain vector or hybrid baseline results",
            )
            .with_detail("query_embedding_source", "none")
            .with_detail("field", "baseline_results[].query_mode"));
        }
        let has_embedding_baseline = result
            .get("baseline_configs")
            .and_then(JsonValue::as_array)
            .unwrap_or(&[])
            .iter()
            .any(|baseline| {
                baseline
                    .get("requires_query_embedding")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false)
            });
        if has_embedding_baseline {
            return Err(Error::new(
                ErrorCode::BenchmarkArtifactModeConflict,
                "benchmark result with query_embedding_source none must not include embedding baselines",
            )
            .with_detail("query_embedding_source", "none")
            .with_detail("field", "baseline_configs[].requires_query_embedding"));
        }
        validate_none_source_has_no_vector_metrics(result)?;
    }
    validate_benchmark_baseline_configs(result, baseline_status == "completed")?;
    validate_benchmark_run_array(result, "runs")?;
    validate_benchmark_run_array(result, "baseline_results")?;
    validate_benchmark_baseline_links(result, baseline_status == "completed")?;
    validate_benchmark_cache_states(result, "runs", false)?;
    validate_benchmark_cache_states(result, "baseline_results", baseline_status == "completed")?;
    validate_benchmark_p0_sqlite_baseline_coverage(result, baseline_status == "completed")?;
    Ok(())
}

fn reject_unknown_benchmark_result_top_level_fields(obj: &crate::json::JsonObject) -> Result<()> {
    for field in obj.keys() {
        if ![
            "schema_version",
            "cofferdb_version",
            "git_commit",
            "file_format_version",
            "hardware",
            "status",
            "path",
            "result_file",
            "dataset_path",
            "dataset_id",
            "collection",
            "dataset",
            "profile",
            "profile_params",
            "engine_params",
            "candidate_counts",
            "durability",
            "index_build_mode",
            "import_context",
            "metrics",
            "warnings",
            "runs",
            "baseline_configs",
            "baseline_results",
            "baseline_status",
            "baseline_status_reason",
        ]
        .contains(&field.as_str())
        {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result contains an unknown top-level public field",
            )
            .with_detail("field", field));
        }
    }
    Ok(())
}

fn validate_none_source_has_no_vector_metrics(result: &JsonValue) -> Result<()> {
    let metrics = result
        .get("metrics")
        .and_then(JsonValue::as_object)
        .expect("benchmark metrics object is validated before source contract checks");
    reject_nonzero_none_source_vector_metrics(metrics, "metrics")?;

    for field in ["runs", "baseline_results"] {
        let Some(runs) = result.get(field).and_then(JsonValue::as_array) else {
            continue;
        };
        for run in runs {
            let Some(run_obj) = run.as_object() else {
                continue;
            };
            if let Some(metrics) = run_obj.get("metrics").and_then(JsonValue::as_object) {
                reject_nonzero_none_source_vector_metrics(metrics, &format!("{field}[].metrics"))?;
            }
            let Some(per_query) = run_obj.get("per_query").and_then(JsonValue::as_array) else {
                continue;
            };
            for item in per_query {
                let Some(item_obj) = item.as_object() else {
                    continue;
                };
                if let Some(execution_stats) = item_obj
                    .get("execution_stats")
                    .and_then(JsonValue::as_object)
                {
                    reject_nonzero_none_source_vector_metrics(
                        execution_stats,
                        &format!("{field}[].per_query[].execution_stats"),
                    )?;
                }
                if let Some(context_stats) = item_obj
                    .get("query_context")
                    .and_then(JsonValue::as_object)
                    .and_then(|context| context.get("execution_stats"))
                    .and_then(JsonValue::as_object)
                {
                    reject_nonzero_none_source_vector_metrics(
                        context_stats,
                        &format!("{field}[].per_query[].query_context.execution_stats"),
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn reject_nonzero_none_source_vector_metrics(
    metrics: &crate::json::JsonObject,
    detail_prefix: &str,
) -> Result<()> {
    for metric in ["vector_scan_count", "distance_ops"] {
        let value = metrics
            .get(metric)
            .and_then(JsonValue::as_f64)
            .unwrap_or_default();
        if value > 0.0 {
            return Err(Error::new(
                ErrorCode::BenchmarkArtifactModeConflict,
                "benchmark result with query_embedding_source none must not include vector metrics",
            )
            .with_detail("query_embedding_source", "none")
            .with_detail("field", format!("{detail_prefix}.{metric}")));
        }
    }
    Ok(())
}

fn validate_benchmark_version_fields(result: &JsonValue) -> Result<()> {
    if result.get("schema_version").and_then(JsonValue::as_str) != Some("0.1") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result schema_version must be 0.1",
        )
        .with_detail("field", "schema_version"));
    }
    for field in ["cofferdb_version", "git_commit"] {
        require_nonempty_string(
            result
                .as_object()
                .expect("benchmark result object is validated by callers"),
            field,
            field,
        )?;
    }
    let file_format_version = result
        .get("file_format_version")
        .and_then(JsonValue::as_f64)
        .unwrap_or_default();
    if file_format_version != FORMAT_VERSION as f64 {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result file_format_version must match the storage format version",
        )
        .with_detail("field", "file_format_version"));
    }
    Ok(())
}

fn validate_benchmark_top_level_enums(result: &JsonValue) -> Result<()> {
    let profile = result
        .get("profile")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !is_stable_profile_value(profile) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result profile must be a stable profile enum value",
        )
        .with_detail("field", "profile"));
    }
    let durability = result
        .get("durability")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(durability, "commit" | "fast") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result durability must be commit or fast",
        )
        .with_detail("field", "durability"));
    }
    Ok(())
}

fn is_stable_profile_value(value: &str) -> bool {
    crate::ResourceProfile::parse(value).is_some()
}

fn expected_benchmark_score_kind(
    engine: &str,
    query_mode: BenchmarkQueryMode,
) -> Option<&'static str> {
    match engine {
        "cofferdb" => Some(query_mode.score_kind()),
        "sqlite" => Some(query_mode.sqlite_score_kind()),
        _ => None,
    }
}

fn validate_benchmark_dataset_identity(
    result: &JsonValue,
    dataset: &crate::json::JsonObject,
) -> Result<()> {
    reject_unknown_benchmark_result_fields(
        dataset,
        "dataset",
        &[
            "dataset_id",
            "version",
            "chunking",
            "tokenizer",
            "embedding_model",
            "embedding_dim",
            "dataset_manifest_sha256",
            "documents_sha256",
            "chunks_sha256",
            "embeddings_sha256",
            "queries_sha256",
            "qrels_sha256",
            "query_embedding_source",
            "query_embedding_model",
            "query_embedding_dim",
            "query_embeddings_sha256",
        ],
    )?;
    validate_tokenizer_field(dataset, "dataset")?;
    let top_dataset_id = result
        .get("dataset_id")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    let dataset_id = dataset
        .get("dataset_id")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if top_dataset_id.trim().is_empty() || top_dataset_id != dataset_id {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result dataset_id must match dataset.dataset_id",
        )
        .with_detail("field", "dataset_id"));
    }
    for field in [
        "version",
        "dataset_manifest_sha256",
        "documents_sha256",
        "chunks_sha256",
        "queries_sha256",
        "qrels_sha256",
    ] {
        require_nonempty_string(dataset, field, format!("dataset.{field}"))?;
    }
    validate_benchmark_dataset_chunking(dataset)?;
    let query_embedding_source = dataset
        .get("query_embedding_source")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(query_embedding_source, "none" | "sidecar" | "inline") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result dataset.query_embedding_source must be none, sidecar, or inline",
        )
        .with_detail("field", "dataset.query_embedding_source"));
    }
    require_positive_integer(dataset, "embedding_dim", "dataset.embedding_dim")?;
    let embedding_model_is_null = matches!(dataset.get("embedding_model"), Some(JsonValue::Null));
    let embeddings_hash_is_null = matches!(dataset.get("embeddings_sha256"), Some(JsonValue::Null));
    let model_is_null = matches!(dataset.get("query_embedding_model"), Some(JsonValue::Null));
    let dim_is_null = matches!(dataset.get("query_embedding_dim"), Some(JsonValue::Null));
    let hash_is_null = matches!(
        dataset.get("query_embeddings_sha256"),
        Some(JsonValue::Null)
    );
    match query_embedding_source {
        "none" => {
            if !(embedding_model_is_null && embeddings_hash_is_null) {
                return Err(Error::new(
                    ErrorCode::BenchmarkArtifactModeConflict,
                    "benchmark result query_embedding_source none must use null corpus embedding model and hash",
                )
                .with_detail("field", "dataset.embedding_model"));
            }
            if !(model_is_null && dim_is_null && hash_is_null) {
                return Err(Error::new(
                    ErrorCode::BenchmarkArtifactModeConflict,
                    "benchmark result query_embedding_source none must use null model, dim, and hash",
                )
                .with_detail("field", "dataset.query_embedding_model"));
            }
        }
        "sidecar" | "inline" => {
            let model = dataset
                .get("query_embedding_model")
                .and_then(JsonValue::as_str)
                .unwrap_or_default();
            if model.trim().is_empty() {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result query embedding model must be non-empty",
                )
                .with_detail("field", "dataset.query_embedding_model"));
            }
            require_nonempty_string(dataset, "embedding_model", "dataset.embedding_model")?;
            require_nonempty_string(dataset, "embeddings_sha256", "dataset.embeddings_sha256")?;
            let embedding_dim = dataset
                .get("embedding_dim")
                .and_then(JsonValue::as_f64)
                .unwrap_or_default();
            let dim = dataset
                .get("query_embedding_dim")
                .and_then(JsonValue::as_f64)
                .unwrap_or_default();
            if !(dim.is_finite() && dim > 0.0 && dim.fract() == 0.0) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result query embedding dim must be a positive integer",
                )
                .with_detail("field", "dataset.query_embedding_dim"));
            }
            if embedding_dim != dim {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result embedding_dim must match query_embedding_dim",
                )
                .with_detail("field", "dataset.embedding_dim"));
            }
            if query_embedding_source == "sidecar" {
                let hash = dataset
                    .get("query_embeddings_sha256")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default();
                if hash.trim().is_empty() {
                    return Err(Error::new(
                        ErrorCode::MissingRequiredField,
                        "benchmark result sidecar query embeddings hash is required",
                    )
                    .with_detail("field", "dataset.query_embeddings_sha256"));
                }
            } else if !hash_is_null {
                return Err(Error::new(
                    ErrorCode::BenchmarkArtifactModeConflict,
                    "benchmark result inline query embeddings must use null query_embeddings_sha256",
                )
                .with_detail("field", "dataset.query_embeddings_sha256"));
            }
        }
        _ => unreachable!("query_embedding_source enum validated above"),
    }
    Ok(())
}

fn reject_unknown_benchmark_result_fields(
    obj: &crate::json::JsonObject,
    detail_prefix: &str,
    allowed: &[&str],
) -> Result<()> {
    for field in obj.keys() {
        if !allowed.contains(&field.as_str()) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result contains an unknown public field",
            )
            .with_detail("field", format!("{detail_prefix}.{field}")));
        }
    }
    Ok(())
}

fn validate_benchmark_dataset_chunking(dataset: &crate::json::JsonObject) -> Result<()> {
    match dataset.get("chunking") {
        Some(JsonValue::Null) => Ok(()),
        Some(JsonValue::Object(chunking)) => {
            let chunk_size = chunking
                .get("chunk_size")
                .and_then(JsonValue::as_f64)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::MissingRequiredField,
                        "benchmark result dataset.chunking requires chunk_size",
                    )
                    .with_detail("field", "dataset.chunking.chunk_size")
                })?;
            if !(chunk_size.is_finite() && chunk_size > 0.0 && chunk_size.fract() == 0.0) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result dataset.chunking.chunk_size must be a positive integer",
                )
                .with_detail("field", "dataset.chunking.chunk_size"));
            }
            let chunk_overlap = chunking
                .get("chunk_overlap")
                .and_then(JsonValue::as_f64)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::MissingRequiredField,
                        "benchmark result dataset.chunking requires chunk_overlap",
                    )
                    .with_detail("field", "dataset.chunking.chunk_overlap")
                })?;
            if !(chunk_overlap.is_finite() && chunk_overlap >= 0.0 && chunk_overlap.fract() == 0.0)
            {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result dataset.chunking.chunk_overlap must be a non-negative integer",
                )
                .with_detail("field", "dataset.chunking.chunk_overlap"));
            }
            if chunk_overlap >= chunk_size {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result dataset.chunking.chunk_overlap must be smaller than chunk_size",
                )
                .with_detail("field", "dataset.chunking.chunk_overlap"));
            }
            Ok(())
        }
        Some(_) => Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result dataset.chunking must be an object or null",
        )
        .with_detail("field", "dataset.chunking")),
        None => Err(Error::new(
            ErrorCode::MissingRequiredField,
            "benchmark result dataset missing required field chunking",
        )
        .with_detail("field", "dataset.chunking")),
    }
}

fn validate_benchmark_hardware(result: &JsonValue) -> Result<()> {
    let hardware = result
        .get("hardware")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result hardware must be an object",
            )
            .with_detail("field", "hardware")
        })?;
    reject_unknown_benchmark_result_fields(
        hardware,
        "hardware",
        &["cpu", "memory_gb", "storage", "os"],
    )?;
    for field in ["cpu", "storage", "os"] {
        let value = hardware
            .get(field)
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        if value.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result hardware.{field} must be a non-empty string"),
            )
            .with_detail("field", format!("hardware.{field}")));
        }
    }
    for field in [
        "cpu_model",
        "cpu_cores",
        "storage_type",
        "os_name",
        "os_version",
    ] {
        if hardware.get(field).is_some() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result hardware.{field} is not part of the v0.1 public hardware schema"),
            )
            .with_detail("field", format!("hardware.{field}")));
        }
    }
    let cpu = hardware
        .get("cpu")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !(cpu.to_ascii_lowercase().contains("core") && cpu.chars().any(|ch| ch.is_ascii_digit())) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result hardware.cpu must include CPU model and core count",
        )
        .with_detail("field", "hardware.cpu"));
    }
    let os = hardware
        .get("os")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !os.chars().any(|ch| ch.is_ascii_digit()) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result hardware.os must include operating system name and version",
        )
        .with_detail("field", "hardware.os"));
    }
    match hardware.get("memory_gb") {
        Some(JsonValue::Null) => Ok(()),
        Some(value) => {
            let memory_gb = value.as_f64().ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result hardware.memory_gb must be a number or null",
                )
                .with_detail("field", "hardware.memory_gb")
            })?;
            if memory_gb.is_finite() && memory_gb > 0.0 {
                Ok(())
            } else {
                Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result hardware.memory_gb must be positive when present",
                )
                .with_detail("field", "hardware.memory_gb"))
            }
        }
        None => Err(Error::new(
            ErrorCode::MissingRequiredField,
            "benchmark result hardware missing memory_gb",
        )
        .with_detail("field", "hardware.memory_gb")),
    }
}

fn validate_benchmark_warnings(result: &JsonValue) -> Result<()> {
    let Some(warnings) = result.get("warnings") else {
        return Ok(());
    };
    let warnings = warnings.as_array().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result warnings must be an array",
        )
        .with_detail("field", "warnings")
    })?;
    for warning in warnings {
        let obj = warning.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result warnings entries must be objects",
            )
            .with_detail("field", "warnings[]")
        })?;
        reject_unknown_benchmark_result_fields(obj, "warnings[]", &["code", "message", "reason"])?;
        require_nonempty_string(obj, "code", "warnings[].code")?;
        require_nonempty_string(obj, "message", "warnings[].message")?;
        require_string_or_null(obj, "reason", "warnings[].reason")?;
    }
    Ok(())
}

fn validate_benchmark_engine_params(result: &JsonValue) -> Result<()> {
    let params = benchmark_object(result, "engine_params")?;
    reject_unknown_benchmark_result_fields(
        params,
        "engine_params",
        &["text_index", "vector_index", "fusion"],
    )?;
    let text_index = require_object_field(params, "text_index", "engine_params.text_index")?;
    reject_unknown_benchmark_result_fields(
        text_index,
        "engine_params.text_index",
        &["index_kind", "tokenizer", "bm25"],
    )?;
    require_string_in(
        text_index,
        "index_kind",
        "engine_params.text_index.index_kind",
        &["text_bm25"],
    )?;
    let tokenizer = require_object_field(
        text_index,
        "tokenizer",
        "engine_params.text_index.tokenizer",
    )?;
    reject_unknown_benchmark_result_fields(
        tokenizer,
        "engine_params.text_index.tokenizer",
        &["name", "version", "profile", "language_hint", "config"],
    )?;
    require_string_in(
        tokenizer,
        "name",
        "engine_params.text_index.tokenizer.name",
        &[crate::index::text::DEFAULT_TOKENIZER_NAME],
    )?;
    require_string_in(
        tokenizer,
        "version",
        "engine_params.text_index.tokenizer.version",
        &[crate::index::text::DEFAULT_TOKENIZER_VERSION],
    )?;
    crate::TokenizerConfig::from_json(&JsonValue::Object(tokenizer.clone()))?;
    let bm25 = require_object_field(text_index, "bm25", "engine_params.text_index.bm25")?;
    reject_unknown_benchmark_result_fields(bm25, "engine_params.text_index.bm25", &["k1", "b"])?;
    require_positive_number(bm25, "k1", "engine_params.text_index.bm25.k1")?;
    require_positive_number(bm25, "b", "engine_params.text_index.bm25.b")?;

    let vector_index = require_object_field(params, "vector_index", "engine_params.vector_index")?;
    reject_unknown_benchmark_result_fields(
        vector_index,
        "engine_params.vector_index",
        &[
            "index_kind",
            "distance_metric",
            "vectors_normalized",
            "normalization_method",
        ],
    )?;
    require_string_in(
        vector_index,
        "index_kind",
        "engine_params.vector_index.index_kind",
        &["vector_exact_f32"],
    )?;
    require_string_in(
        vector_index,
        "distance_metric",
        "engine_params.vector_index.distance_metric",
        &["cosine", "dot_product", "euclidean"],
    )?;
    require_bool(
        vector_index,
        "vectors_normalized",
        "engine_params.vector_index.vectors_normalized",
    )?;
    require_nonempty_string(
        vector_index,
        "normalization_method",
        "engine_params.vector_index.normalization_method",
    )?;

    let fusion = require_object_field(params, "fusion", "engine_params.fusion")?;
    reject_unknown_benchmark_result_fields(
        fusion,
        "engine_params.fusion",
        &["algorithm", "score_scale"],
    )?;
    require_string_in(
        fusion,
        "algorithm",
        "engine_params.fusion.algorithm",
        &["weighted_sum"],
    )?;
    require_string_in(
        fusion,
        "score_scale",
        "engine_params.fusion.score_scale",
        &["engine_local"],
    )?;
    Ok(())
}

fn validate_benchmark_numeric_surfaces(result: &JsonValue) -> Result<()> {
    let profile_params = benchmark_object(result, "profile_params")?;
    reject_unknown_benchmark_result_fields(
        profile_params,
        "profile_params",
        &[
            "bm25_candidates",
            "vector_candidates",
            "fusion_candidates",
            "cache_budget_mb",
        ],
    )?;
    for field in [
        "bm25_candidates",
        "vector_candidates",
        "fusion_candidates",
        "cache_budget_mb",
    ] {
        require_positive_integer(profile_params, field, format!("profile_params.{field}"))?;
    }

    let candidate_counts = benchmark_object(result, "candidate_counts")?;
    reject_unknown_benchmark_result_fields(
        candidate_counts,
        "candidate_counts",
        &["bm25", "vector", "fusion"],
    )?;
    for field in ["bm25", "vector", "fusion"] {
        require_positive_integer(candidate_counts, field, format!("candidate_counts.{field}"))?;
    }
    for (candidate_field, profile_field) in [
        ("bm25", "bm25_candidates"),
        ("vector", "vector_candidates"),
        ("fusion", "fusion_candidates"),
    ] {
        let candidate_value = candidate_counts
            .get(candidate_field)
            .and_then(JsonValue::as_f64)
            .unwrap_or_default();
        let profile_value = profile_params
            .get(profile_field)
            .and_then(JsonValue::as_f64)
            .unwrap_or_default();
        if (candidate_value - profile_value).abs() > f64::EPSILON {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result candidate_counts must match profile_params",
            )
            .with_detail("field", format!("candidate_counts.{candidate_field}"))
            .with_detail("profile_field", format!("profile_params.{profile_field}")));
        }
    }

    let metrics = benchmark_object(result, "metrics")?;
    reject_unknown_benchmark_result_fields(
        metrics,
        "metrics",
        &[
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "p95_ms",
            "vector_scan_count",
            "distance_ops",
            "fusion_candidate_count",
            "import_time_ms",
            "index_size_mb",
            "memory_peak_mb",
        ],
    )?;
    for field in [
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "p95_ms",
        "vector_scan_count",
        "distance_ops",
        "fusion_candidate_count",
        "import_time_ms",
        "index_size_mb",
    ] {
        require_nonnegative_number(metrics, field, format!("metrics.{field}"))?;
    }
    require_nonnegative_number_or_null(metrics, "memory_peak_mb", "metrics.memory_peak_mb")?;

    let import_context = benchmark_object(result, "import_context")?;
    validate_tokenizer_field(import_context, "import_context")?;
    reject_unknown_benchmark_result_fields(
        import_context,
        "import_context",
        &[
            "batch_size",
            "index_build_mode",
            "import_time_ms",
            "import_command",
            "manifest_path",
            "database_path",
            "on_conflict",
            "summary",
            "durability",
            "tokenizer",
        ],
    )?;
    require_nonempty_string(
        import_context,
        "import_command",
        "import_context.import_command",
    )?;
    require_nonempty_string(
        import_context,
        "manifest_path",
        "import_context.manifest_path",
    )?;
    require_nonempty_string(
        import_context,
        "database_path",
        "import_context.database_path",
    )?;
    let on_conflict = import_context
        .get("on_conflict")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(on_conflict, "error" | "skip" | "replace") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result import_context.on_conflict must be error, skip, or replace",
        )
        .with_detail("field", "import_context.on_conflict"));
    }
    require_nonnegative_number(import_context, "batch_size", "import_context.batch_size")?;
    require_nonnegative_number(
        import_context,
        "import_time_ms",
        "import_context.import_time_ms",
    )?;
    let durability = import_context
        .get("durability")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(durability, "commit" | "fast") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result import_context.durability must be commit or fast",
        )
        .with_detail("field", "import_context.durability"));
    }
    let import_index_build_mode = import_context
        .get("index_build_mode")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(
        import_index_build_mode,
        "inline_full_checkpoint" | "deferred_full_checkpoint" | "transient_only"
    ) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result import_context.index_build_mode must be a stable enum value",
        )
        .with_detail("field", "import_context.index_build_mode"));
    }
    if import_context
        .get("summary")
        .and_then(JsonValue::as_object)
        .is_none()
    {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result import_context.summary must be an object",
        )
        .with_detail("field", "import_context.summary"));
    }
    Ok(())
}

fn benchmark_object<'a>(
    result: &'a JsonValue,
    field: &'static str,
) -> Result<&'a crate::json::JsonObject> {
    result
        .get(field)
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {field} must be an object"),
            )
            .with_detail("field", field)
        })
}

fn require_nonnegative_number(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    let value = obj.get(field).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("benchmark result missing {detail_field}"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    let number = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a number"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    if number.is_finite() && number >= 0.0 {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be non-negative"),
        )
        .with_detail("field", detail_field))
    }
}

fn require_positive_number(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    let value = obj.get(field).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("benchmark result missing {detail_field}"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    let number = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a number"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    if number.is_finite() && number > 0.0 {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be positive"),
        )
        .with_detail("field", detail_field))
    }
}

fn require_nonnegative_number_or_null(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    match obj.get(field) {
        Some(JsonValue::Null) => Ok(()),
        Some(_) => require_nonnegative_number(obj, field, detail_field.clone()),
        None => Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("benchmark result missing {detail_field}"),
        )
        .with_detail("field", detail_field)),
    }
}

fn require_positive_integer(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    let value = obj.get(field).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("benchmark result missing {detail_field}"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    let number = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a number"),
        )
        .with_detail("field", detail_field.clone())
    })?;
    if number.is_finite() && number > 0.0 && number.fract() == 0.0 {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a positive integer"),
        )
        .with_detail("field", detail_field))
    }
}

fn require_string_in(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
    allowed: &[&str],
) -> Result<()> {
    let detail_field = detail_field.into();
    let value = obj
        .get(field)
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a stable value"),
        )
        .with_detail("field", detail_field))
    }
}

fn require_object_field<'a>(
    obj: &'a crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<&'a crate::json::JsonObject> {
    let detail_field = detail_field.into();
    obj.get(field)
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {detail_field} must be an object"),
            )
            .with_detail("field", detail_field)
        })
}

fn validate_benchmark_baseline_configs(result: &JsonValue, require_verified: bool) -> Result<()> {
    let configs = result
        .get("baseline_configs")
        .and_then(JsonValue::as_array)
        .expect("benchmark baseline_configs array is validated before entry schema checks");
    for config in configs {
        let obj = config.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result baseline_configs entries must be objects",
            )
            .with_detail("field", "baseline_configs[]")
        })?;
        reject_unknown_benchmark_result_fields(
            obj,
            "baseline_configs[]",
            &[
                "name",
                "version",
                "run_mode",
                "mode",
                "requires_query_embedding",
                "top_k",
                "cache_states",
                "config_source",
                "checked_at",
                "config",
            ],
        )?;
        require_nonempty_string(obj, "name", "baseline_configs[].name")?;
        if require_verified {
            require_nonempty_string(obj, "version", "baseline_configs[].version")?;
        } else {
            require_string_or_null(obj, "version", "baseline_configs[].version")?;
        }
        require_nonempty_string(obj, "run_mode", "baseline_configs[].run_mode")?;
        require_nonempty_string(obj, "mode", "baseline_configs[].mode")?;
        require_bool(
            obj,
            "requires_query_embedding",
            "baseline_configs[].requires_query_embedding",
        )?;
        require_positive_integer(obj, "top_k", "baseline_configs[].top_k")?;
        let cache_states = obj
            .get("cache_states")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result baseline_configs[].cache_states must be an array",
                )
                .with_detail("field", "baseline_configs[].cache_states")
            })?;
        let mut has_cold = false;
        let mut has_warm = false;
        for state in cache_states {
            match state.as_str() {
                Some("cold") => has_cold = true,
                Some("warm") => has_warm = true,
                Some(_) | None => {
                    return Err(Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result baseline_configs[].cache_states entries must be cold or warm",
                    )
                    .with_detail("field", "baseline_configs[].cache_states"));
                }
            }
        }
        if !has_cold || !has_warm {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result baseline_configs[].cache_states must include cold and warm",
            )
            .with_detail("field", "baseline_configs[].cache_states"));
        }
        require_nonempty_string(obj, "config_source", "baseline_configs[].config_source")?;
        if require_verified {
            require_nonempty_string(obj, "checked_at", "baseline_configs[].checked_at")?;
        } else {
            require_string_or_null(obj, "checked_at", "baseline_configs[].checked_at")?;
        }
        if let Some(checked_at) = obj.get("checked_at").and_then(JsonValue::as_str) {
            validate_baseline_checked_at(checked_at)?;
        }
        let Some(config) = obj.get("config").and_then(JsonValue::as_object) else {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result baseline_configs[].config must be an object",
            )
            .with_detail("field", "baseline_configs[].config"));
        };
        reject_unknown_benchmark_result_fields(
            config,
            "baseline_configs[].config",
            &[
                "engine",
                "sqlite_cli",
                "fts",
                "fts_table",
                "fts_tokenizer",
                "fts_score",
                "fusion",
                "filter",
                "status",
                "vector_extension",
                "vector_extension_version",
                "vector_mode",
                "query_count",
                "cache_policy",
                "latency_scope",
            ],
        )?;
        for field in [
            "engine",
            "sqlite_cli",
            "fts",
            "fts_tokenizer",
            "fts_score",
            "fusion",
            "filter",
            "status",
        ] {
            require_nonempty_string(config, field, format!("baseline_configs[].config.{field}"))?;
        }
        let expected_status = if require_verified {
            "completed"
        } else {
            "not_run"
        };
        if config.get("status").and_then(JsonValue::as_str) != Some(expected_status) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result baseline config status must match baseline_status",
            )
            .with_detail("field", "baseline_configs[].config.status")
            .with_detail("expected", expected_status));
        }
        require_string_or_null(
            config,
            "vector_extension",
            "baseline_configs[].config.vector_extension",
        )?;
        require_string_or_null(
            config,
            "vector_mode",
            "baseline_configs[].config.vector_mode",
        )?;
        require_positive_integer(
            config,
            "query_count",
            "baseline_configs[].config.query_count",
        )?;
    }
    Ok(())
}

fn validate_baseline_checked_at(value: &str) -> Result<()> {
    let candidate = format!("{value}T00:00:00Z");
    if value.len() == 10 && crate::time::parse_rfc3339_seconds(&candidate).is_some() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result baseline_configs[].checked_at must be a YYYY-MM-DD date",
        )
        .with_detail("field", "baseline_configs[].checked_at"))
    }
}

fn validate_benchmark_baseline_links(result: &JsonValue, require_completed: bool) -> Result<()> {
    let configs = result
        .get("baseline_configs")
        .and_then(JsonValue::as_array)
        .expect("benchmark baseline_configs array is validated before baseline link checks");
    let results = result
        .get("baseline_results")
        .and_then(JsonValue::as_array)
        .expect("benchmark baseline_results array is validated before baseline link checks");

    let mut engines_by_name = BTreeMap::new();
    let mut query_counts_by_name = BTreeMap::new();
    for config in configs {
        let obj = config
            .as_object()
            .expect("benchmark baseline config objects are validated before link checks");
        let Some(name) = obj.get("name").and_then(JsonValue::as_str) else {
            continue;
        };
        let Some(engine) = obj
            .get("config")
            .and_then(JsonValue::as_object)
            .and_then(|config| config.get("engine"))
            .and_then(JsonValue::as_str)
        else {
            continue;
        };
        engines_by_name.insert(name.to_owned(), engine.to_owned());
        let query_count = obj
            .get("config")
            .and_then(JsonValue::as_object)
            .and_then(|config| config.get("query_count"))
            .and_then(JsonValue::as_f64)
            .unwrap_or_default() as usize;
        query_counts_by_name.insert(name.to_owned(), query_count);
    }

    if require_completed && engines_by_name.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "completed benchmark baselines must include baseline_configs entries",
        )
        .with_detail("field", "baseline_configs"));
    }

    let mut result_names = BTreeSet::new();
    for run in results {
        let run_obj = run
            .as_object()
            .expect("benchmark baseline result objects are validated before link checks");
        let Some(name) = run_obj.get("name").and_then(JsonValue::as_str) else {
            continue;
        };
        let Some(expected_engine) = engines_by_name.get(name) else {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark baseline result name must match a baseline config",
            )
            .with_detail("field", "baseline_results[].name")
            .with_detail("baseline_name", name));
        };
        let engine = run_obj
            .get("engine")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        if engine != expected_engine {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark baseline result engine must match baseline config",
            )
            .with_detail("field", "baseline_results[].engine")
            .with_detail("baseline_name", name));
        }
        if let Some(expected_query_count) = query_counts_by_name.get(name) {
            let actual_query_count = run_obj
                .get("per_query")
                .and_then(JsonValue::as_array)
                .map(|items| items.len())
                .unwrap_or(0);
            if actual_query_count != *expected_query_count {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark baseline result per_query count must match baseline config query_count",
                )
                .with_detail("field", "baseline_results[].per_query")
                .with_detail("baseline_name", name)
                .with_detail("expected_query_count", *expected_query_count)
                .with_detail("actual_query_count", actual_query_count));
            }
        }
        result_names.insert(name.to_owned());
    }

    if require_completed {
        for name in engines_by_name.keys() {
            if !result_names.contains(name) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "completed benchmark baseline config must have matching baseline results",
                )
                .with_detail("field", "baseline_results[].name")
                .with_detail("baseline_name", name));
            }
        }
    }
    Ok(())
}

fn validate_benchmark_p0_sqlite_baseline_coverage(
    result: &JsonValue,
    require_completed: bool,
) -> Result<()> {
    if !require_completed {
        return Ok(());
    }
    let query_embedding_source = result
        .get("dataset")
        .and_then(|dataset| dataset.get("query_embedding_source"))
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    let (expected_name, expected_requires_embedding, expected_modes): (&str, bool, &[&str]) =
        if query_embedding_source == "none" {
            ("sqlite-fts5-text-only", false, &["text_bm25"])
        } else {
            (
                "sqlite-fts5-sqlite-vec",
                true,
                &["text_bm25", "vector_exact", "hybrid"],
            )
        };

    let configs = result
        .get("baseline_configs")
        .and_then(JsonValue::as_array)
        .expect("benchmark baseline_configs array is validated before P0 coverage checks");
    let Some(config) = configs.iter().find_map(|config| {
        let obj = config.as_object()?;
        (obj.get("name").and_then(JsonValue::as_str) == Some(expected_name)).then_some(obj)
    }) else {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "completed benchmark baselines must include the P0 SQLite baseline config",
        )
        .with_detail("field", "baseline_configs[].name")
        .with_detail("baseline_name", expected_name));
    };
    if config
        .get("requires_query_embedding")
        .and_then(JsonValue::as_bool)
        != Some(expected_requires_embedding)
    {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "P0 SQLite baseline requires_query_embedding must match dataset query embedding source",
        )
        .with_detail("field", "baseline_configs[].requires_query_embedding")
        .with_detail("baseline_name", expected_name));
    }
    let config_obj = config
        .get("config")
        .and_then(JsonValue::as_object)
        .expect("benchmark baseline config object is validated before P0 coverage checks");
    if config_obj.get("engine").and_then(JsonValue::as_str) != Some("sqlite") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "P0 baseline config engine must be sqlite",
        )
        .with_detail("field", "baseline_configs[].config.engine")
        .with_detail("baseline_name", expected_name));
    }
    if config_obj.get("fts").and_then(JsonValue::as_str) != Some("fts5") {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "P0 baseline config fts engine must be fts5",
        )
        .with_detail("field", "baseline_configs[].config.fts")
        .with_detail("baseline_name", expected_name));
    }
    if expected_requires_embedding {
        for (field, expected) in [
            ("vector_extension", "sqlite-vec"),
            ("vector_mode", "exact_scan"),
            ("fusion", "weighted_sum"),
        ] {
            if config_obj.get(field).and_then(JsonValue::as_str) != Some(expected) {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "P0 sqlite-vec baseline config must record vector mode and fusion parameters",
                )
                .with_detail("field", format!("baseline_configs[].config.{field}"))
                .with_detail("baseline_name", expected_name));
            }
        }
    }
    let config_top_k = config
        .get("top_k")
        .and_then(JsonValue::as_f64)
        .expect("benchmark baseline config top_k is validated before P0 coverage checks")
        as usize;

    let results = result
        .get("baseline_results")
        .and_then(JsonValue::as_array)
        .expect("benchmark baseline_results array is validated before P0 coverage checks");
    let mut cache_states_by_mode: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut observed_max_top_k = 0usize;
    for run in results {
        let run_obj = run
            .as_object()
            .expect("benchmark baseline result objects are validated before P0 coverage checks");
        if run_obj.get("name").and_then(JsonValue::as_str) != Some(expected_name) {
            continue;
        }
        let query_mode = run_obj
            .get("query_mode")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        let cache_state = run_obj
            .get("cache_state")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        cache_states_by_mode
            .entry(query_mode)
            .or_default()
            .insert(cache_state);
        for item in run_obj
            .get("per_query")
            .and_then(JsonValue::as_array)
            .expect("benchmark baseline per_query array is validated before P0 coverage checks")
        {
            let top_k = item
                .get("query_context")
                .and_then(|context| context.get("top_k"))
                .and_then(JsonValue::as_f64)
                .expect(
                    "benchmark baseline query_context.top_k is validated before P0 coverage checks",
                ) as usize;
            observed_max_top_k = observed_max_top_k.max(top_k);
        }
    }
    if observed_max_top_k != 0 && observed_max_top_k != config_top_k {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "P0 baseline config top_k must match the max per-query top_k",
        )
        .with_detail("field", "baseline_configs[].top_k")
        .with_detail("baseline_name", expected_name)
        .with_detail("expected_top_k", observed_max_top_k)
        .with_detail("actual_top_k", config_top_k));
    }
    for expected_mode in expected_modes {
        let Some(states) = cache_states_by_mode.get(expected_mode) else {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "completed P0 SQLite baseline must include all required query modes",
            )
            .with_detail("field", "baseline_results[].query_mode")
            .with_detail("baseline_name", expected_name)
            .with_detail("expected_query_mode", *expected_mode));
        };
        if !(states.contains("cold") && states.contains("warm")) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "completed P0 SQLite baseline must include cold and warm results for every required query mode",
            )
            .with_detail("field", "baseline_results[].cache_state")
            .with_detail("baseline_name", expected_name)
            .with_detail("expected_query_mode", *expected_mode));
        }
    }
    Ok(())
}

fn validate_benchmark_run_array(result: &JsonValue, field: &'static str) -> Result<()> {
    let runs = result
        .get(field)
        .and_then(JsonValue::as_array)
        .expect("benchmark result array fields are validated before run schema checks");
    for run in runs {
        let run_obj = run.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {field} entries must be objects"),
            )
            .with_detail("field", format!("{field}[]"))
        })?;
        let run_allowed: &[&str] = if field == "baseline_results" {
            &[
                "name",
                "engine",
                "query_mode",
                "cache_state",
                "metrics",
                "per_query",
            ]
        } else {
            &[
                "engine",
                "query_mode",
                "cache_state",
                "metrics",
                "per_query",
            ]
        };
        reject_unknown_benchmark_result_fields(run_obj, &format!("{field}[]"), run_allowed)?;
        for required in [
            "engine",
            "query_mode",
            "cache_state",
            "metrics",
            "per_query",
        ] {
            if run_obj.get(required).is_none() {
                return Err(Error::new(
                    ErrorCode::MissingRequiredField,
                    format!("benchmark result {field} entry missing {required}"),
                )
                .with_detail("field", format!("{field}[].{required}")));
            }
        }
        if field == "baseline_results" {
            require_nonempty_string(run_obj, "name", "baseline_results[].name")?;
        }
        let engine = run_obj
            .get("engine")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        require_nonempty_string(run_obj, "engine", format!("{field}[].engine"))?;
        if field == "runs" && engine != "cofferdb" {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result runs[].engine must be cofferdb",
            )
            .with_detail("field", "runs[].engine"));
        }
        let query_mode = run_obj
            .get("query_mode")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        let query_mode = BenchmarkQueryMode::parse(query_mode).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result query_mode must be a stable enum value",
            )
            .with_detail("field", format!("{field}[].query_mode"))
        })?;
        validate_benchmark_run_metrics(run_obj, field)?;
        let per_query = run_obj
            .get("per_query")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("benchmark result {field}[].per_query must be an array"),
                )
                .with_detail("field", format!("{field}[].per_query"))
            })?;
        if per_query.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {field}[].per_query must not be empty"),
            )
            .with_detail("field", format!("{field}[].per_query")));
        }
        for item in per_query {
            let item_obj = item.as_object().ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("benchmark result {field}[].per_query entries must be objects"),
                )
                .with_detail("field", format!("{field}[].per_query[]"))
            })?;
            reject_unknown_benchmark_result_fields(
                item_obj,
                &format!("{field}[].per_query[]"),
                &[
                    "query_id",
                    "latency_ms",
                    "ranked_chunk_ids",
                    "score_kind",
                    "score_scale",
                    "score_comparable",
                    "scores",
                    "hit",
                    "execution_stats",
                    "query_context",
                ],
            )?;
            for required in [
                "query_id",
                "latency_ms",
                "ranked_chunk_ids",
                "score_kind",
                "score_scale",
                "score_comparable",
                "scores",
                "hit",
                "execution_stats",
                "query_context",
            ] {
                if item_obj.get(required).is_none() {
                    return Err(Error::new(
                        ErrorCode::MissingRequiredField,
                        format!("benchmark result per_query entry missing {required}"),
                    )
                    .with_detail("field", format!("{field}[].per_query[].{required}")));
                }
            }
            require_nonempty_string(
                item_obj,
                "query_id",
                format!("{field}[].per_query[].query_id"),
            )?;
            require_nonnegative_number(
                item_obj,
                "latency_ms",
                format!("{field}[].per_query[].latency_ms"),
            )?;
            require_nonempty_string(
                item_obj,
                "score_kind",
                format!("{field}[].per_query[].score_kind"),
            )?;
            let score_kind = item_obj
                .get("score_kind")
                .and_then(JsonValue::as_str)
                .unwrap_or_default();
            if expected_benchmark_score_kind(engine, query_mode)
                .map(|expected| score_kind != expected)
                .unwrap_or(false)
            {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result per_query score_kind must match engine and query_mode",
                )
                .with_detail("field", format!("{field}[].per_query[].score_kind"))
                .with_detail("engine", engine)
                .with_detail("query_mode", query_mode.as_str()));
            }
            require_nonempty_string(
                item_obj,
                "score_scale",
                format!("{field}[].per_query[].score_scale"),
            )?;
            require_bool(
                item_obj,
                "score_comparable",
                format!("{field}[].per_query[].score_comparable"),
            )?;
            if item_obj.get("score_scale").and_then(JsonValue::as_str) != Some("engine_local") {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result per_query score_scale must be engine_local",
                )
                .with_detail("field", format!("{field}[].per_query[].score_scale")));
            }
            if item_obj
                .get("score_comparable")
                .and_then(JsonValue::as_bool)
                != Some(false)
            {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result per_query score_comparable must be false",
                )
                .with_detail("field", format!("{field}[].per_query[].score_comparable")));
            }
            require_bool(item_obj, "hit", format!("{field}[].per_query[].hit"))?;
            let execution_stats = item_obj
                .get("execution_stats")
                .and_then(JsonValue::as_object)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result per_query execution_stats must be an object",
                    )
                    .with_detail("field", format!("{field}[].per_query[].execution_stats"))
                })?;
            reject_unknown_benchmark_result_fields(
                execution_stats,
                &format!("{field}[].per_query[].execution_stats"),
                &[
                    "vector_scan_count",
                    "distance_ops",
                    "fusion_candidate_count",
                ],
            )?;
            for metric in [
                "vector_scan_count",
                "distance_ops",
                "fusion_candidate_count",
            ] {
                require_nonnegative_number(
                    execution_stats,
                    metric,
                    format!("{field}[].per_query[].execution_stats.{metric}"),
                )?;
            }
            validate_benchmark_query_context(item_obj, field)?;
            let context = item_obj
                .get("query_context")
                .and_then(JsonValue::as_object)
                .expect("benchmark per-query context is validated before mode consistency checks");
            validate_benchmark_query_context_mode_consistency(context, field, query_mode)?;
            let ranked_chunk_count = item_obj
                .get("ranked_chunk_ids")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result ranked_chunk_ids must be an array",
                    )
                    .with_detail("field", format!("{field}[].per_query[].ranked_chunk_ids"))
                })?
                .iter()
                .map(|value| value.as_str())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result ranked_chunk_ids entries must be strings",
                    )
                    .with_detail("field", format!("{field}[].per_query[].ranked_chunk_ids"))
                })?
                .len();
            let score_count = item_obj
                .get("scores")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result scores must be an array",
                    )
                    .with_detail("field", format!("{field}[].per_query[].scores"))
                })?
                .iter()
                .map(|value| value.as_f64().filter(|number| number.is_finite()))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "benchmark result scores entries must be finite numbers",
                    )
                    .with_detail("field", format!("{field}[].per_query[].scores"))
                })?
                .len();
            if ranked_chunk_count != score_count {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result per_query scores must align with ranked_chunk_ids",
                )
                .with_detail("field", format!("{field}[].per_query[].scores")));
            }
            let top_k = context
                .get("top_k")
                .and_then(JsonValue::as_f64)
                .expect("benchmark query context top_k is validated before result count checks")
                as usize;
            if ranked_chunk_count > top_k {
                return Err(Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark result ranked_chunk_ids must not exceed query_context.top_k",
                )
                .with_detail("field", format!("{field}[].per_query[].ranked_chunk_ids"))
                .with_detail("top_k", top_k)
                .with_detail("ranked_chunk_count", ranked_chunk_count));
            }
        }
    }
    Ok(())
}

fn validate_benchmark_query_context_mode_consistency(
    context: &crate::json::JsonObject,
    field: &'static str,
    query_mode: BenchmarkQueryMode,
) -> Result<()> {
    let expected_text_present = query_mode.has_text();
    if context.get("text_present").and_then(JsonValue::as_bool) != Some(expected_text_present) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result query_context.text_present must match query_mode",
        )
        .with_detail(
            "field",
            format!("{field}[].per_query[].query_context.text_present"),
        )
        .with_detail("query_mode", query_mode.as_str()));
    }
    let expected_embedding_present = query_mode.has_vector();
    if context
        .get("embedding_present")
        .and_then(JsonValue::as_bool)
        != Some(expected_embedding_present)
    {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result query_context.embedding_present must match query_mode",
        )
        .with_detail(
            "field",
            format!("{field}[].per_query[].query_context.embedding_present"),
        )
        .with_detail("query_mode", query_mode.as_str()));
    }
    Ok(())
}

fn validate_benchmark_query_context(
    item_obj: &crate::json::JsonObject,
    field: &'static str,
) -> Result<()> {
    let detail_prefix = format!("{field}[].per_query[].query_context");
    let context = require_object_field(item_obj, "query_context", detail_prefix.clone())?;
    reject_unknown_benchmark_result_fields(
        context,
        &detail_prefix,
        &[
            "text_present",
            "tokenizer",
            "embedding_present",
            "filter_present",
            "weights",
            "requested_weights",
            "effective_weights",
            "weights_source",
            "renormalization_reason",
            "metadata_boost_present",
            "metadata_boost_enabled",
            "boost_config_hash",
            "boost_rule_count",
            "top_k",
            "profile",
            "candidate_limits",
            "execution_stats",
            "index_source",
            "index_rebuild_reason",
            "transient_rebuild",
        ],
    )?;
    for flag in [
        "text_present",
        "embedding_present",
        "filter_present",
        "metadata_boost_present",
        "metadata_boost_enabled",
    ] {
        require_bool(context, flag, format!("{detail_prefix}.{flag}"))?;
    }
    let tokenizer = context.get("tokenizer").ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            "query context requires tokenizer",
        )
        .with_detail("field", format!("{detail_prefix}.tokenizer"))
    })?;
    if field == "runs" && context.get("text_present").and_then(JsonValue::as_bool) == Some(true) {
        crate::TokenizerConfig::from_json(tokenizer)?;
    } else if !tokenizer.is_null() {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "non-CofferDB or vector-only query tokenizer must be null",
        )
        .with_detail("field", format!("{detail_prefix}.tokenizer")));
    }

    validate_weight_surface(context, "requested_weights", &detail_prefix)?;
    validate_weight_surface(context, "effective_weights", &detail_prefix)?;
    require_nonempty_string(
        context,
        "weights_source",
        format!("{detail_prefix}.weights_source"),
    )?;
    require_string_or_null(
        context,
        "renormalization_reason",
        format!("{detail_prefix}.renormalization_reason"),
    )?;

    let weights = require_object_field(context, "weights", format!("{detail_prefix}.weights"))?;
    reject_unknown_benchmark_result_fields(
        weights,
        &format!("{detail_prefix}.weights"),
        &[
            "present",
            "source",
            "requested",
            "effective",
            "renormalization_reason",
        ],
    )?;
    require_bool(
        weights,
        "present",
        format!("{detail_prefix}.weights.present"),
    )?;
    require_nonempty_string(weights, "source", format!("{detail_prefix}.weights.source"))?;
    let weights_prefix = format!("{detail_prefix}.weights");
    validate_weight_surface(weights, "requested", &weights_prefix)?;
    validate_weight_surface(weights, "effective", &weights_prefix)?;
    require_string_or_null(
        weights,
        "renormalization_reason",
        format!("{detail_prefix}.weights.renormalization_reason"),
    )?;

    require_string_or_null(
        context,
        "boost_config_hash",
        format!("{detail_prefix}.boost_config_hash"),
    )?;
    require_nonnegative_number(
        context,
        "boost_rule_count",
        format!("{detail_prefix}.boost_rule_count"),
    )?;
    require_positive_integer(context, "top_k", format!("{detail_prefix}.top_k"))?;
    let profile = context
        .get("profile")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !is_stable_profile_value(profile) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result query_context.profile must be a stable profile enum value",
        )
        .with_detail("field", format!("{detail_prefix}.profile")));
    }

    let candidate_limits = require_object_field(
        context,
        "candidate_limits",
        format!("{detail_prefix}.candidate_limits"),
    )?;
    reject_unknown_benchmark_result_fields(
        candidate_limits,
        &format!("{detail_prefix}.candidate_limits"),
        &["bm25", "vector", "fusion"],
    )?;
    for metric in ["bm25", "vector", "fusion"] {
        require_nonnegative_number(
            candidate_limits,
            metric,
            format!("{detail_prefix}.candidate_limits.{metric}"),
        )?;
    }

    if field == "runs" {
        let top_k = context
            .get("top_k")
            .and_then(JsonValue::as_f64)
            .expect("top_k was validated above");
        if top_k > crate::query::MAX_SEARCH_TOP_K as f64 {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result CofferDB top_k exceeds the supported exact JSON range",
            )
            .with_detail("field", format!("{detail_prefix}.top_k")));
        }
        let params = crate::ResourceProfile::parse(profile)
            .expect("profile was validated above")
            .params();
        for (branch, default_limit) in [
            ("bm25", params.bm25_candidates),
            ("vector", params.vector_candidates),
            ("fusion", params.fusion_candidates),
        ] {
            let expected = top_k.max(default_limit as f64);
            if candidate_limits.get(branch).and_then(JsonValue::as_f64) != Some(expected) {
                return Err(Error::new(ErrorCode::InvalidJsonlSchema,
                    "benchmark result CofferDB candidate limit must equal max(profile default, top_k)")
                    .with_detail("field", format!("{detail_prefix}.candidate_limits.{branch}")));
            }
        }
    }

    let execution_stats = require_object_field(
        context,
        "execution_stats",
        format!("{detail_prefix}.execution_stats"),
    )?;
    reject_unknown_benchmark_result_fields(
        execution_stats,
        &format!("{detail_prefix}.execution_stats"),
        &[
            "vector_scan_count",
            "distance_ops",
            "fusion_candidate_count",
        ],
    )?;
    for metric in [
        "vector_scan_count",
        "distance_ops",
        "fusion_candidate_count",
    ] {
        require_nonnegative_number(
            execution_stats,
            metric,
            format!("{detail_prefix}.execution_stats.{metric}"),
        )?;
    }

    let index_source = context
        .get("index_source")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if !matches!(
        index_source,
        "persistent_checkpoint"
            | "transient_rebuild"
            | "sqlite_fts5"
            | "sqlite_vec"
            | "sqlite_fts5_sqlite_vec"
    ) {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "benchmark result query_context.index_source must be a stable index source value",
        )
        .with_detail("field", format!("{detail_prefix}.index_source")));
    }
    require_string_or_null(
        context,
        "index_rebuild_reason",
        format!("{detail_prefix}.index_rebuild_reason"),
    )?;
    let transient_rebuild = require_object_field(
        context,
        "transient_rebuild",
        format!("{detail_prefix}.transient_rebuild"),
    )?;
    reject_unknown_benchmark_result_fields(
        transient_rebuild,
        &format!("{detail_prefix}.transient_rebuild"),
        &[
            "started",
            "completed",
            "cancelled",
            "budget_mb",
            "chunks_scanned",
            "bytes_read",
            "budget_exceeded",
        ],
    )?;
    for flag in ["started", "completed", "cancelled", "budget_exceeded"] {
        require_bool(
            transient_rebuild,
            flag,
            format!("{detail_prefix}.transient_rebuild.{flag}"),
        )?;
    }
    require_nonnegative_number(
        transient_rebuild,
        "budget_mb",
        format!("{detail_prefix}.transient_rebuild.budget_mb"),
    )?;
    require_nonnegative_number_or_null(
        transient_rebuild,
        "chunks_scanned",
        format!("{detail_prefix}.transient_rebuild.chunks_scanned"),
    )?;
    require_nonnegative_number_or_null(
        transient_rebuild,
        "bytes_read",
        format!("{detail_prefix}.transient_rebuild.bytes_read"),
    )?;
    Ok(())
}

fn validate_weight_surface(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_prefix: &str,
) -> Result<()> {
    let weights = require_object_field(obj, field, format!("{detail_prefix}.{field}"))?;
    reject_unknown_benchmark_result_fields(
        weights,
        &format!("{detail_prefix}.{field}"),
        &["bm25", "vector", "metadata"],
    )?;
    for weight in ["bm25", "vector", "metadata"] {
        require_nonnegative_number(weights, weight, format!("{detail_prefix}.{field}.{weight}"))?;
    }
    Ok(())
}

fn validate_benchmark_run_metrics(
    run_obj: &crate::json::JsonObject,
    field: &'static str,
) -> Result<()> {
    let metrics = run_obj
        .get("metrics")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {field}[].metrics must be an object"),
            )
            .with_detail("field", format!("{field}[].metrics"))
        })?;
    reject_unknown_benchmark_result_fields(
        metrics,
        &format!("{field}[].metrics"),
        &[
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "p95_ms",
            "vector_scan_count",
            "distance_ops",
            "fusion_candidate_count",
            "memory_peak_mb",
        ],
    )?;
    for metric in [
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "p95_ms",
        "vector_scan_count",
        "distance_ops",
        "fusion_candidate_count",
    ] {
        require_nonnegative_number(metrics, metric, format!("{field}[].metrics.{metric}"))?;
    }
    require_nonnegative_number_or_null(
        metrics,
        "memory_peak_mb",
        match field {
            "runs" => "runs[].metrics.memory_peak_mb",
            "baseline_results" => "baseline_results[].metrics.memory_peak_mb",
            _ => "metrics.memory_peak_mb",
        },
    )?;
    Ok(())
}

fn require_nonempty_string(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    let value = obj
        .get(field)
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    if value.trim().is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a non-empty string"),
        )
        .with_detail("field", detail_field));
    }
    Ok(())
}

fn require_string_or_null(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    match obj.get(field) {
        Some(JsonValue::Null) => Ok(()),
        Some(JsonValue::String(value)) if !value.trim().is_empty() => Ok(()),
        Some(JsonValue::String(_)) | Some(_) => Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a non-empty string or null"),
        )
        .with_detail("field", detail_field.clone())),
        None => Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("benchmark result missing {detail_field}"),
        )
        .with_detail("field", detail_field)),
    }
}

fn require_bool(
    obj: &crate::json::JsonObject,
    field: &str,
    detail_field: impl Into<String>,
) -> Result<()> {
    let detail_field = detail_field.into();
    if obj.get(field).and_then(JsonValue::as_bool).is_none() {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {detail_field} must be a boolean"),
        )
        .with_detail("field", detail_field));
    }
    Ok(())
}

fn validate_benchmark_cache_states(
    result: &JsonValue,
    field: &'static str,
    require_non_empty: bool,
) -> Result<()> {
    let runs = result
        .get(field)
        .and_then(JsonValue::as_array)
        .expect("benchmark result array fields are validated before cache-state checks");
    if require_non_empty && runs.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("benchmark result {field} must be non-empty when baselines completed"),
        )
        .with_detail("field", field));
    }
    let mut states_by_group: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for run in runs {
        let run_obj = run
            .as_object()
            .expect("benchmark result run objects are validated before cache-state checks");
        let engine = run_obj
            .get("engine")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("benchmark result {field}[].engine must be a string"),
                )
                .with_detail("field", format!("{field}[].engine"))
            })?;
        let query_mode = run_obj
            .get("query_mode")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("benchmark result {field}[].query_mode must be a string"),
                )
                .with_detail("field", format!("{field}[].query_mode"))
            })?;
        if BenchmarkQueryMode::parse(query_mode).is_none() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result query_mode must be a stable enum value",
            )
            .with_detail("field", format!("{field}[].query_mode")));
        }
        let cache_state = run_obj
            .get("cache_state")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    format!("benchmark result {field}[].cache_state must be a string"),
                )
                .with_detail("field", format!("{field}[].cache_state"))
            })?;
        if !matches!(cache_state, "cold" | "warm") {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark result cache_state must be cold or warm",
            )
            .with_detail("field", format!("{field}[].cache_state")));
        }
        let name = run_obj
            .get("name")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        states_by_group
            .entry(format!("{engine}\0{name}\0{query_mode}"))
            .or_default()
            .insert(cache_state.to_owned());
    }
    for states in states_by_group.values() {
        if !(states.contains("cold") && states.contains("warm")) {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                format!("benchmark result {field} must include cold and warm cache states"),
            )
            .with_detail("field", format!("{field}[].cache_state")));
        }
    }
    Ok(())
}

fn validate_tokenizer_field(obj: &crate::json::JsonObject, prefix: &str) -> Result<()> {
    let value = obj.get("tokenizer").ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            "complete tokenizer is required",
        )
        .with_detail("field", format!("{prefix}.tokenizer"))
    })?;
    crate::TokenizerConfig::from_json(value)
        .map(|_| ())
        .map_err(|err| {
            Error::new(err.code(), err.message())
                .with_detail("field", format!("{prefix}.tokenizer"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::object_from_pairs;

    #[test]
    fn result_validation_rejects_invalid_roots_without_mutation() {
        for seed in 0..100 {
            for value in [
                JsonValue::Null,
                JsonValue::Bool(seed % 2 == 0),
                JsonValue::Number(seed as f64),
                JsonValue::String(format!("artifact-{seed}")),
                JsonValue::Array(vec![JsonValue::Null; seed]),
            ] {
                let original = value.to_compact_string();
                let error = validate_benchmark_result_contract(&value).unwrap_err();
                assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
                assert_eq!(
                    error.detail(),
                    &[("field".into(), "benchmark_result".into())]
                );
                assert_eq!(value.to_compact_string(), original);
            }
        }
        let error = validate_benchmark_result_contract(&JsonValue::object()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::MissingRequiredField);
        assert_eq!(error.detail(), &[("field".into(), "schema_version".into())]);
    }

    #[test]
    fn result_validation_rejects_unknown_top_level_fields_before_missing_fields() {
        for seed in 0..100 {
            let field = format!("unexpected_{seed}");
            let value = object_from_pairs([(field.as_str(), JsonValue::Null)]);
            let original = value.to_compact_string();
            let error = validate_benchmark_result_contract(&value).unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
            assert_eq!(error.detail(), &[("field".into(), field)]);
            assert_eq!(value.to_compact_string(), original);
        }
    }
}
